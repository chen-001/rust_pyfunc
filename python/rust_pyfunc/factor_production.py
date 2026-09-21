"""生产因子的公共计算与写入逻辑；因子脚本只保留四个清晰的调用入口。

计算沿用完整研究截面；最终 H5/H1/DB 按股票表输出。DB 保留原值。
转换严格使用收尾的 float32 rank → 可交易缺失填补 → rolling 顺序。
"""
from contextlib import contextmanager
from importlib import import_module
from pathlib import Path
import fcntl
import hashlib
import json
import os
import shutil

import h5py
import numpy as np
import pandas as pd


class _ProductionRuntime:
    def __init__(self, settings):
        self.rp = settings.get("rp") or import_module(__package__)
        for key in _SETTINGS:
            if key in settings:
                setattr(self, key, settings[key])

    def limit_resources(self, n_jobs):
        if not 1 <= n_jobs <= 30:
            raise ValueError("每个系列的 CPU 预算必须在 1 至 30 之间")
        available = sorted(os.sched_getaffinity(0))
        cores = available[: min(n_jobs, len(available))]
        for task in Path("/proc/self/task").iterdir():
            try:
                os.sched_setaffinity(int(task.name), cores)
            except ProcessLookupError:
                pass
        return min(n_jobs, len(available))

    def raw_dates(self, start, end):
        return sorted(
            (
                int(p.name)
                for p in Path(self.level2_root).iterdir()
                if p.name.isdigit() and start <= int(p.name) <= end and (p / "transaction").is_dir()
            )
        )

    def raw_tasks(self, start, end):
        symbols = set(
            pd.read_csv(
                Path(self.calendar_root) / "basic_info/symbol_map.csv", dtype={"symbol": str}
            )["symbol"].str.zfill(6)
        )
        return [
            [date, path.name[:6]]
            for date in self.raw_dates(start, end)
            for path in sorted(
                (Path(self.level2_root) / str(date) / "transaction").glob("*_transaction.csv")
            )
            if path.name[:6] in symbols
        ]

    def prepare_output_axis(self, root, version, extra_symbols=()):
        """股票表只追加，保持旧 H5 列位置；新增上市股票不能被静默丢弃。"""
        directory = Path(root) / version
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "symbol_map.csv"
        previous = (
            pd.read_csv(path, dtype={"symbol": str})["symbol"].tolist() if path.exists() else []
        )
        current = (
            pd.read_csv(
                Path(self.calendar_root) / "basic_info/symbol_map.csv", dtype={"symbol": str}
            )["symbol"]
            .str.zfill(6)
            .tolist()
        )
        symbols = list(dict.fromkeys(previous + current + sorted(set(extra_symbols))))
        capacity_path = directory / "capacity.csv"
        if previous == symbols and capacity_path.exists():
            return
        if capacity_path.exists():
            capacity = pd.read_csv(capacity_path, index_col="name")["count"].to_dict()
        else:
            capacity = (
                pd.read_csv(Path(self.calendar_root) / "basic_info/capacity_config.csv")
                .iloc[0]
                .to_dict()
            )
        capacity["symbol_capacity"] = max(int(capacity["symbol_capacity"]), len(symbols))
        for file in directory.glob("*.h5"):
            with h5py.File(file, "r+") as handle:
                data = handle["data"]
                if data.shape[1] < capacity["symbol_capacity"]:
                    data.resize((data.shape[0], capacity["symbol_capacity"]))
        pd.DataFrame({"name": list(capacity), "count": list(capacity.values())}).to_csv(
            capacity_path, index=False
        )
        temporary = path.with_suffix(".tmp")
        pd.DataFrame({"symbol": symbols, "pos": range(len(symbols))}).to_csv(temporary, index=False)
        temporary.replace(path)

    def configuration_fingerprint(self):
        settings = dict(
            schema="explicit-universe-v2",
            module=self.rp.__name__,
            groups=self.pipeline_groups,
            base=self.names_save,
            stock_only_input=self.stock_only_input,
            gap5=self.names_gap5,
            gap1=self.names_gap1,
            db=self.names_in_db,
        )
        return hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()

    def write_selected_base(self, source_dirs, start_date, end_date):
        extra = set()
        if not self.stock_only_input:
            for directory in set(source_dirs.values()):
                extra.update(
                    (
                        str(code)[:6]
                        for code in self.rp.factor_store_v5_template(directory)["stocks"]
                    )
                )
        self.prepare_output_axis(self.base_hdf5_dir, self.base_factor_ver, extra)
        selected = set(self.names_save)
        for raw_name in dict.fromkeys((name.removesuffix("_fold") for name in self.names_save)):
            frame = self.rp.read_factor_from_colblk(
                source_dirs[raw_name], raw_name, start_date, end_date
            )
            frame.columns = [code[:6] for code in frame.columns]
            if raw_name in selected:
                self.rp.save_factor(
                    frame,
                    self.base_factor_ver,
                    raw_name,
                    start_date,
                    end_date,
                    1,
                    self.base_hdf5_dir,
                )
            if raw_name + "_fold" in selected:
                values = frame.to_numpy(dtype=np.float32)
                values[~np.isfinite(values)] = np.nan
                counts = np.sum(~np.isnan(values), axis=1, keepdims=True)
                means = np.divide(
                    np.nansum(values, axis=1, keepdims=True, dtype=np.float64),
                    counts,
                    out=np.full((len(values), 1), np.nan),
                    where=counts > 0,
                )
                folded = pd.DataFrame(
                    np.abs(values - means).astype(np.float32),
                    index=frame.index,
                    columns=frame.columns,
                )
                self.rp.save_factor(
                    folded,
                    self.base_factor_ver,
                    raw_name + "_fold",
                    start_date,
                    end_date,
                    1,
                    self.base_hdf5_dir,
                )

    def stock_input_root(self, start_date, end_date):
        """只给横截面计算器提供股票文件；文件级软链接避免复制原始数据。"""
        symbols = set(
            pd.read_csv(
                Path(self.calendar_root) / "basic_info/symbol_map.csv", dtype={"symbol": str}
            )["symbol"].str.zfill(6)
        )
        if not symbols or any((len(code) != 6 or not code.isdigit() for code in symbols)):
            raise ValueError("股票白名单为空或格式错误，拒绝读取全部证券")
        root = Path(self.colblk_store_dir) / "_stock_input"
        if root.is_symlink():
            raise ValueError("股票输入临时目录不能是目录软链接")
        if root.exists():
            shutil.rmtree(root)
        for date in self.raw_dates(start_date, end_date):
            for kind in ("transaction", "market_data"):
                directory = root / str(date) / kind
                directory.mkdir(parents=True, exist_ok=True)
                for code in sorted(symbols):
                    name = f"{code}_{date}_{kind}.csv"
                    source = Path(self.level2_root) / str(date) / kind / name
                    if source.is_file():
                        target = directory / name
                        if target.is_symlink():
                            target.unlink()
                        target.symlink_to(source.resolve())
        return (str(root.resolve()), symbols)

    def run_base(self, start_date, end_date, n_jobs=None, stock_only=None):
        if n_jobs is None:
            n_jobs = self.cpu_limit
        n_jobs = self.limit_resources(n_jobs)
        if stock_only is None:
            stock_only = self.stock_only_input
        cross = any((group["kind"] != "per_stock" for group in self.pipeline_groups))
        data_root, stock_codes = (
            self.stock_input_root(start_date, end_date)
            if stock_only and cross
            else (self.level2_root, None)
        )
        for group in self.pipeline_groups:
            store = (
                self.colblk_store_dir
                if group["dir"] is None
                else str(Path(self.colblk_store_dir) / group["dir"])
            )
            common = dict(
                pipeline=group["pipeline"],
                expected_result_length=len(group["names"]),
                trading_days=list(self.rp.td.trading_days),
                params=group["params"],
                update_mode=True,
                bind_cores=False,
                store_dir=store,
                store_factor_names=group["names"],
                data_root=data_root,
            )
            if group["kind"] == "per_stock":
                self.rp.run_factor_pipeline(
                    tasks=self.raw_tasks(start_date, end_date),
                    n_jobs=n_jobs,
                    backup_file="",
                    mode="multiprocess",
                    export_n_jobs=n_jobs,
                    **common,
                )
            else:
                self.rp.run_factor_pipeline_cross_section(
                    tasks=self.raw_dates(start_date, end_date),
                    n_jobs=max(1, n_jobs - 4),
                    n_workers=1,
                    vars_root=self.vars_root,
                    **common,
                )
            if stock_codes is not None:
                unexpected = {
                    code[:6] for code in self.rp.factor_store_v5_template(store)["stocks"]
                } - stock_codes
                if unexpected:
                    raise RuntimeError(f"计算结果含股票白名单之外的证券: {sorted(unexpected)}")
        self.write_selected_base(self.colblk_of, start_date, end_date)

    def run_db(self, start_date, end_date):
        self.prepare_output_axis(self.db_hdf5_dir, self.db_factor_ver)
        for name in self.names_in_db:
            frame = self.rp.read_factor(
                self.base_factor_ver, name, start_date, end_date, 1, self.base_hdf5_dir
            )
            self.rp.save_factor(
                frame, self.db_factor_ver, name, start_date, end_date, 1, self.db_hdf5_dir
            )

    def final_frames(self, names, start_date, end_date):
        """复用收尾引擎的 rank → 可交易缺失中位数填充 → rolling，保持 float32 口径。"""
        groups = {}
        for name in names:
            if name.endswith("_smooth_1"):
                source, stat, window = (name[:-9], "smooth", 1)
            else:
                prefix, window = name.rsplit("_smooth_", 1)
                source, stat = prefix.rsplit("_", 1)
                window = int(window)
                if stat not in ("mean", "max", "min", "std"):
                    raise ValueError("不支持的收尾算子: " + name)
            groups.setdefault(source, []).append((name, stat, window))
        days = [int(d) for d in self.rp.td.trading_days]
        first = next((i for i, date in enumerate(days) if date >= start_date))
        lookback = max((w for targets in groups.values() for _, _, w in targets), default=1)
        read_start = max(self.initial_start_date, days[max(0, first - lookback + 1)])
        restrict = self.rp.read_factor(
            "Base", "S_RESTRICT", read_start, end_date, 1, self.vars_root
        )
        for source, targets in groups.items():
            raw = self.rp.read_factor(
                self.base_factor_ver, source, read_start, end_date, 1, self.base_hdf5_dir
            )
            windows = sorted({window for _, _, window in targets if window > 1})
            values = raw.to_numpy(dtype=np.float32)
            values[~np.isfinite(values)] = np.nan
            mask = restrict.reindex(index=raw.index, columns=raw.columns).to_numpy(dtype=np.float32)
            block = np.asarray(
                self.rp.tail_v5_rank_fill_roll_block_f32(
                    np.ascontiguousarray(values), np.ascontiguousarray(mask), windows
                )
            )
            slots = {("smooth", 1): 0}
            for index, window in enumerate(windows):
                slots.update(
                    {
                        (stat, window): 1 + 4 * index + offset
                        for offset, stat in enumerate(("mean", "max", "min", "std"))
                    }
                )
            for name, stat, window in targets:
                yield (
                    name,
                    pd.DataFrame(
                        block[:, :, slots[stat, window]], index=raw.index, columns=raw.columns
                    ),
                )

    def run_h5(self, start_date, end_date):
        self.prepare_output_axis(self.final_hdf5_dir_gap5, self.final_factor_ver_gap5)
        for name, frame in self.final_frames(self.names_gap5, start_date, end_date):
            self.rp.save_factor(
                frame,
                self.final_factor_ver_gap5,
                name,
                start_date,
                end_date,
                1,
                self.final_hdf5_dir_gap5,
            )

    def run_h1(self, start_date, end_date):
        self.prepare_output_axis(self.final_hdf5_dir_gap1, self.final_factor_ver_gap1)
        for name, frame in self.final_frames(self.names_gap1, start_date, end_date):
            self.rp.save_factor(
                frame,
                self.final_factor_ver_gap1,
                name,
                start_date,
                end_date,
                1,
                self.final_hdf5_dir_gap1,
            )

    def cleanup_colblk(self):
        root = Path(self.colblk_store_dir).resolve()
        expected = (Path(self.output_root) / "temporary" / self.ver).resolve()
        if root != expected or Path(self.colblk_store_dir).is_symlink():
            raise ValueError("临时目录与本系列配置不一致，拒绝删除")
        if root.exists():
            shutil.rmtree(root)

    def run_update(self, start_date=None, end_date=None):
        self.limit_resources(self.cpu_limit)
        if not self.names_in_db:
            raise RuntimeError("尚未审核并写入 dailybank 名单，请先完成 hmokay4 后处理")
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        with self.state_path.with_suffix(".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            state = json.loads(self.state_path.read_text()) if self.state_path.exists() else {}
            fingerprint = self.configuration_fingerprint()
            if state and state.get("configuration") != fingerprint:
                raise RuntimeError("版本或因子名单已改变，不能沿用旧进度；请先完成对应历史重建")
            start = start_date or (
                int(state["completed_through"]) + 1 if state else self.initial_start_date
            )
            dates = self.raw_dates(start, end_date or 20991231)
            if dates:
                expected = {
                    int(d) for d in self.rp.td.trading_days if start <= int(d) <= max(dates)
                }
                missing = sorted(expected.difference(dates))
                if missing:
                    raise RuntimeError(f"原始数据缺少交易日，进度保持不变: {missing}")
            for date in dates:
                self.run_base(date, date)
                self.run_h5(date, date)
                self.run_h1(date, date)
                self.run_db(date, date)
                self.cleanup_colblk()
                state["completed_through"] = max(date, int(state.get("completed_through", 0)))
                state["module"] = self.rp.__name__
                state["configuration"] = fingerprint
                if self.stock_only_input:
                    state.setdefault("stock_only_from", date)
                state["calculation_universe"] = (
                    "stocks_only" if self.stock_only_input else "research_universe"
                )
                state["stock_universe_sha256"] = hashlib.sha256(
                    (Path(self.calendar_root) / "basic_info/symbol_map.csv").read_bytes()
                ).hexdigest()
                temporary = self.state_path.with_suffix(".tmp")
                temporary.write_text(json.dumps(state, ensure_ascii=False))
                temporary.replace(self.state_path)


_SETTINGS = [
    "ver",
    "base_factor_ver",
    "db_factor_ver",
    "final_factor_ver_gap5",
    "final_factor_ver_gap1",
    "base_hdf5_dir",
    "db_hdf5_dir",
    "final_hdf5_dir_gap5",
    "final_hdf5_dir_gap1",
    "colblk_store_dir",
    "state_path",
    "pipeline_groups",
    "colblk_of",
    "names_save",
    "names_tail_two",
    "names_gap5",
    "names_gap1",
    "names_in_db",
    "level2_root",
    "calendar_root",
    "vars_root",
    "output_root",
    "initial_start_date",
    "cpu_limit",
    "stock_only_input",
]


def factor_production_runtime(settings):
    """供本机验收流程复用底层逻辑；同事只需调用因子脚本中的四个入口。"""
    options = {key: settings[key] for key in [*_SETTINGS, "rp"] if key in settings}
    return _runtime(
        options.pop("base_factor_ver"), options.pop("base_hdf5_dir"),
        options.pop("calendar_root"), options.pop("vars_root"), **options,
    )


def _runtime(base_factor_ver, base_hdf5_dir, calendar_root, vars_root, **overrides):
    root = Path(base_hdf5_dir).parent
    ver = base_factor_ver.removesuffix("_base")
    settings = dict(
        ver=ver,
        output_root=str(root),
        base_factor_ver=base_factor_ver,
        base_hdf5_dir=base_hdf5_dir,
        db_factor_ver=ver,
        db_hdf5_dir=str(root / "dailybank"),
        final_factor_ver_gap5=ver,
        final_factor_ver_gap1=ver,
        final_hdf5_dir_gap5=str(root / "hf_data"),
        final_hdf5_dir_gap1=str(root / "hf_data1"),
        colblk_store_dir=str(root / "temporary" / ver),
        state_path=root / "state" / (ver + ".json"),
        level2_root="/ssd_data/stock",
        calendar_root=calendar_root,
        vars_root=vars_root,
        initial_start_date=20150105,
        cpu_limit=30,
        stock_only_input=False,
        pipeline_groups=[],
        names_save=[],
        names_tail_two=[],
        names_gap5=[],
        names_gap1=[],
        names_in_db=[],
    )
    settings.update(overrides)
    settings["colblk_of"] = {
        name: (
            settings["colblk_store_dir"]
            if group["dir"] is None
            else str(Path(settings["colblk_store_dir"]) / group["dir"])
        )
        for group in settings["pipeline_groups"]
        for name in group["names"]
    }
    return _ProductionRuntime(settings)


@contextmanager
def _stage(runtime):
    runtime.limit_resources(runtime.cpu_limit)
    os.environ["RUST_PYFUNC_LEVEL2_ROOT"] = runtime.level2_root
    os.environ["RUST_PYFUNC_CALENDAR_PATH"] = runtime.calendar_root
    os.environ["RUST_PYFUNC_VARS_DIR"] = runtime.vars_root
    os.environ["RUST_PYFUNC_BASIC_INFO_DIR"] = str(Path(runtime.calendar_root) / "basic_info")
    os.environ["RUST_PYFUNC_WORKER_BIN"] = str(
        Path(runtime.rp.__file__).parent / "rust_pyfunc_worker"
    )
    os.environ["RAYON_NUM_THREADS"] = str(runtime.cpu_limit)
    for key in (
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[key] = "1"
    runtime.state_path.parent.mkdir(parents=True, exist_ok=True)
    with runtime.state_path.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def write_selected_factor_base(
    pipeline_groups,
    names,
    start_date,
    end_date,
    *,
    base_factor_ver,
    base_hdf5_dir,
    colblk_store_dir,
    level2_root,
    calendar_root,
    vars_root,
    n_jobs=30,
):
    """计算指定日期，保存所需原值/fold；成功写入后删除临时 colblk。"""
    runtime = _runtime(
        base_factor_ver,
        base_hdf5_dir,
        calendar_root,
        vars_root,
        pipeline_groups=pipeline_groups,
        names_save=names,
        colblk_store_dir=colblk_store_dir,
        level2_root=level2_root,
        cpu_limit=n_jobs,
    )
    with _stage(runtime):
        runtime.run_base(start_date, end_date, n_jobs)
        runtime.cleanup_colblk()


def write_selected_factor_db(
    names,
    start_date,
    end_date,
    *,
    base_factor_ver,
    base_hdf5_dir,
    db_factor_ver,
    db_hdf5_dir,
    calendar_root,
    vars_root,
):
    """把入选原值写进 DB，只输出股票；后续预处理由使用者负责。"""
    runtime = _runtime(
        base_factor_ver,
        base_hdf5_dir,
        calendar_root,
        vars_root,
        names_in_db=names,
        db_factor_ver=db_factor_ver,
        db_hdf5_dir=db_hdf5_dir,
    )
    with _stage(runtime):
        runtime.run_db(start_date, end_date)


def write_selected_factor_final(
    names,
    start_date,
    end_date,
    *,
    base_factor_ver,
    base_hdf5_dir,
    final_factor_ver,
    final_hdf5_dir,
    calendar_root,
    vars_root,
):
    """完整截面排名、填补及 rolling 后，仅将入选股票结果写入 H5/H1。"""
    runtime = _runtime(
        base_factor_ver,
        base_hdf5_dir,
        calendar_root,
        vars_root,
        names_gap5=names,
        final_factor_ver_gap5=final_factor_ver,
        final_hdf5_dir_gap5=final_hdf5_dir,
    )
    with _stage(runtime):
        runtime.run_h5(start_date, end_date)


_RUST_DELETED = {
    "r20_det_ac1",
    "r20_abs_ac1",
    "br_ac1",
    "ct_b_skew",
    "ct_s_skew",
    "ct_s_ac1",
    "ct_d_skew",
    "ct_da_ac1",
}


def _build_col_names():
    """Rust 返回的 63 列列名（生成 71 列原始顺序后过滤 8 死列）。"""
    raw = []
    for a in ["mean", "std", "ac1", "trend"]:
        raw.append(f"imb1_{a}")
    for a in ["mean", "std", "skew", "ac1", "trend"]:
        raw.append(f"imb1_5_{a}")
    for a in ["skew", "ac1", "trend"]:
        raw.append(f"imb6_10_{a}")
    for a in ["mean", "std", "skew", "ac1"]:
        raw.append(f"vt_same_{a}")
    for a in ["mean", "std", "skew", "trend"]:
        raw.append(f"vt_opp_{a}")
    for a in ["mean", "std", "skew", "trend"]:
        raw.append(f"vs_same_{a}")
    for a in ["std", "skew", "ac1", "trend"]:
        raw.append(f"vs_opp_{a}")
    for a in ["mean", "skew", "ac1", "trend"]:
        raw.append(f"r10_det_{a}")
    raw.append("r10_abs_skew")
    for a in ["mean", "skew", "ac1", "trend"]:
        raw.append(f"r20_det_{a}")
    for a in ["ac1", "trend"]:
        raw.append(f"r20_abs_{a}")
    for s in ["pvol", "ivtrend", "sd10_same", "sd20_both"]:
        raw.append(s)
    for a in ["mean", "std", "skew", "ac1", "trend"]:
        raw.append(f"od_raw_{a}")
    for a in ["skew", "trend"]:
        raw.append(f"od_abs_{a}")
    for a in ["mean", "std", "ac1", "trend"]:
        raw.append(f"br_{a}")
    for a in ["mean", "skew"]:
        raw.append(f"ct_b_{a}")
    for a in ["mean", "skew", "ac1", "trend"]:
        raw.append(f"ct_s_{a}")
    for a in ["mean", "skew"]:
        raw.append(f"ct_d_{a}")
    for a in ["ac1", "trend"]:
        raw.append(f"ct_da_{a}")
    raw.append("am_a_mean")
    for s in ["tr_d", "bu_d", "se_d", "bs_dd", "bs_dda", "btr"]:
        raw.append(s)
    assert len(raw) == 71
    return [c for c in raw if c not in _RUST_DELETED]


def observable_order_factor_names():
    col_names = _build_col_names()
    prefixes = []
    for side in ["bid", "ask"]:
        for method in ["A", "B"]:
            for arr in ["seg", "pre5", "diff", "abs_diff"]:
                prefixes.append(f"{side}_{method}_{arr}")
    suffixes = []
    for s in [
        "mean",
        "median",
        "std",
        "skew",
        "kurt",
        "p5",
        "p25",
        "p75",
        "p95",
        "iqr",
        "cv",
        "autocorr1",
        "autocorr1_abs",
        "trend",
        "period_diff",
        "period_ratio",
    ]:
        for c in col_names:
            suffixes.append(f"{c}_{s}")
    n = len(col_names)
    for i in range(n):
        for j in range(i + 1, n):
            suffixes.append(f"{col_names[i]}_corr_{col_names[j]}")
    for s in ["lz_complexity", "entropy_1d", "max_range_product"]:
        for c in col_names:
            suffixes.append(f"{c}_{s}")
    return [f"{p}_{s}" for p in prefixes for s in suffixes]
