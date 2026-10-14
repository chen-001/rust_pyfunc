"""Self-contained factor-gate workflow for server 150.

Only ``factor_path``, ``exist_ic5_path`` and ``output_path`` are required in
``main``. Public data are read from ``/nas197/binary/stock/sz_alpha``.
"""

from __future__ import annotations

import hashlib
import logging
import json
import multiprocessing as mp
import os
import re
import sys
import textwrap
from pathlib import Path
from typing import Any, Iterable

for _name in ("MKL_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"
os.environ.setdefault("MPLBACKEND", "Agg")

import h5py
import numpy as np
import pandas as pd
from scipy.special import ndtri


BARRA_NAMES = [
    "residual_volatility", "momentum", "earnings_yield", "growth",
    "liquidity", "size", "leverage", "book_to_price", "beta",
    "non_linear_size",
]
DEFAULT_CORR_THRES = 0.6
SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")
WORKER_GATE = None


def configure_logger(path: Path) -> logging.Logger:
    logger = logging.getLogger("factor_gate")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    target = str(path.resolve())
    if not any(isinstance(item, logging.FileHandler) and item.baseFilename == target for item in logger.handlers):
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        file_handler = logging.FileHandler(path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
    return logger


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    frame.to_csv(temp, encoding="utf-8")
    os.replace(temp, path)


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    frame = frame.copy()
    if isinstance(frame.index, pd.MultiIndex):
        frame.index.names = [
            str(name) if name is not None else f"level_{number}"
            for number, name in enumerate(frame.index.names)
        ]
    else:
        frame.index.name = "date" if frame.index.name is None else str(frame.index.name)
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns.names = [
            str(name) if name is not None else f"level_{number}"
            for number, name in enumerate(frame.columns.names)
        ]
    else:
        frame.columns.name = "stock" if frame.columns.name is None else str(frame.columns.name)
    frame.to_parquet(temp, engine="pyarrow", compression="zstd")
    os.replace(temp, path)


def atomic_json(value: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp, path)


def safe_name(name: str) -> str:
    return SAFE_NAME.sub("_", str(name)).strip("._") or "factor"


def windcode(symbol: str) -> str:
    code = str(symbol).strip().upper().split(".")[0].zfill(6)
    if code.startswith(("0", "3")):
        return code + ".SZ"
    if code.startswith("6"):
        return code + ".SH"
    if code.startswith("8"):
        return code + ".BJ"
    return code


def worker_base(name: str) -> str:
    WORKER_GATE._calc_one_base(name)
    return name


def worker_perf(spec: tuple[str, str]) -> str:
    WORKER_GATE._backtest_one(spec[0], spec[1])
    return spec[0]


def worker_none(spec: tuple[str, str]) -> str:
    WORKER_GATE._calc_one_none(spec[0], spec[1])
    return spec[0]


def worker_final(name: str) -> str:
    WORKER_GATE._calc_one_final(name)
    return name


class FactorGate:
    """Complete factor-gate workflow on 150."""

    INPUT_ROOT = Path("/nas197/binary/stock/sz_alpha")
    HOLD_PRD = 5
    BENCHMARK = "000852.SH"

    def __init__(self, workers: int = 80):
        self.factor_path: Path | None = None
        self.exist_ic5_path: Path | None = None
        self.output_path: Path | None = None
        self.start_date = "2016-01-01"
        self.end_date = "2026-07-31"
        self.outsample_start_date: str | None = None
        self.corr_thres = DEFAULT_CORR_THRES
        self.workers = min(int(workers), 80)
        self.dates = pd.DatetimeIndex([])
        self.stocks = pd.Index([])
        self.restrict = pd.DataFrame()
        self.industry = pd.DataFrame()
        self.returns = pd.DataFrame()
        self.log_exret = pd.DataFrame()
        self.barra: list[pd.DataFrame] = []
        self.fill_base: list[pd.DataFrame] = []
        self._base_valid_mask: np.ndarray | None = None
        self._base_gram_inv: np.ndarray | None = None
        self.factor_files: dict[str, Path] = {}
        self.candidate_names: list[str] = []
        self.downstream_names: list[str] = []
        self.selection = pd.DataFrame()
        self.logger = logging.getLogger("factor_gate")

    def set_factor_path(self, path: str) -> "FactorGate":
        self.factor_path = self._resolve_factor_path(path) if path else None
        return self

    def set_exist_ic5_path(self, path: str) -> "FactorGate":
        self.exist_ic5_path = Path(path).expanduser() if path else None
        return self

    def set_output_path(self, path: str) -> "FactorGate":
        self.output_path = Path(path).expanduser() if path else None
        return self

    def set_start_date(self, date: str) -> "FactorGate":
        self.start_date = str(date)
        return self

    def set_end_date(self, date: str) -> "FactorGate":
        self.end_date = str(date)
        return self

    def set_outsample_start_date(self, date: str) -> "FactorGate":
        self.outsample_start_date = str(date) if date else None
        return self

    def set_corr_thres(self, value: float) -> "FactorGate":
        value = float(value)
        if not 0.0 < value <= 1.0:
            raise ValueError("corr_thres must be in (0, 1]")
        self.corr_thres = value
        return self

    def calc_base_resid(self) -> None:
        self._log("开始 calc_base_resid")
        self._validate_paths()
        self._load_context()
        self._read_candidate_names()
        self._prepare_base_linear_cache()
        self._run_pool(worker_base, self.candidate_names, "base_resid")
        self._log("完成 calc_base_resid")

    def backtest_base_resid(self) -> None:
        self._log("开始 backtest_base_resid")
        self._require_context()
        specs = [(name, "base_resid") for name in self.candidate_names]
        self._run_pool(worker_perf, specs, "base_perf")
        self._log("完成 backtest_base_resid")

    def get_greedy_list(self) -> None:
        self._log("开始 get_greedy_list")
        self._require_output()
        self.candidate_names = self._read_candidate_names()
        values = {}
        for name in self.candidate_names:
            perf = pd.read_csv(self._file("base_perf", name, ".csv"), index_col=0, parse_dates=True)
            values[name] = pd.to_numeric(perf["ic5"], errors="coerce")
        candidate_ic = pd.DataFrame(values).sort_index().loc[self.start_date:self.end_date]
        atomic_parquet(candidate_ic, self.output_path / "selection" / "candidate_base_ic5.parquet")
        mean_abs_ic5 = candidate_ic.mean().abs().sort_values(ascending=False, kind="mergesort")

        selected = []
        for name in mean_abs_ic5.index:
            if not selected:
                selected.append(name)
                continue
            corr = candidate_ic[selected].corrwith(candidate_ic[name]).abs().dropna()
            if corr.empty or float(corr.max()) <= self.corr_thres:
                selected.append(name)

        atomic_csv(
            pd.DataFrame({
                "factor_name": selected,
                "base_abs_mean_ic5": [float(mean_abs_ic5[name]) for name in selected],
                "greedy_rank": range(1, len(selected) + 1),
                "threshold": self.corr_thres,
            }),
            self.output_path / "selection" / f"greedy_corr_{self.corr_thres:.1f}.csv",
        )

        self.downstream_names = selected
        atomic_csv(
            pd.DataFrame({
                "factor_name": self.downstream_names,
                "base_abs_mean_ic5": [float(mean_abs_ic5[name]) for name in self.downstream_names],
                "union_rank": range(1, len(self.downstream_names) + 1),
            }),
            self.output_path / "selection" / "downstream_union.csv",
        )
        self._log(
            f"完成 get_greedy_list，corr_thres={self.corr_thres:.3f}, "
            f"downstream_union={len(self.downstream_names)}"
        )

    def get_top3(self) -> None:
        self._log("开始 get_top3")
        self._require_output()
        if not self.downstream_names:
            union_path = self.output_path / "selection" / "downstream_union.csv"
            self.downstream_names = pd.read_csv(union_path)["factor_name"].astype(str).tolist()
        candidate_ic = pd.read_parquet(self.output_path / "selection" / "candidate_base_ic5.parquet")
        existing = self._load_existing_ic5()
        excluded = set(self.candidate_names)
        pool = existing.drop(columns=sorted(excluded.intersection(existing.columns)), errors="ignore")
        rows = []
        for name in self.downstream_names:
            corr = pool.corrwith(candidate_ic[name]).abs().dropna().sort_values(ascending=False, kind="mergesort")
            row: dict[str, Any] = {
                "factor_name": name,
                "base_abs_mean_ic5": abs(float(candidate_ic[name].mean())),
                "excluded_existing_count": int(len(existing.columns) - pool.shape[1]),
                "remaining_existing_count": int(pool.shape[1]),
            }
            for number in range(3):
                row[f"B{number + 1}"] = str(corr.index[number]) if number < len(corr) else ""
                row[f"corr{number + 1}"] = float(corr.iloc[number]) if number < len(corr) else np.nan
            rows.append(row)
        self.selection = pd.DataFrame(rows).set_index("factor_name")
        atomic_csv(self.selection, self.output_path / "selection" / "selection_top3.csv")
        b_names = sorted({str(self.selection.loc[name, f"B{i}"]) for name in self.downstream_names for i in range(1, 4)})
        if "" in b_names:
            raise RuntimeError("some downstream factors have fewer than three B factors")
        atomic_csv(pd.DataFrame({"factor_name": b_names}), self.output_path / "selection" / "unique_b_names.csv")
        self._log(f"完成 get_top3，unique_B={len(b_names)}")

    def prepare_AB_none(self) -> None:
        self._log("开始 prepare_AB_none")
        self._require_context()
        self._load_selection()
        b_names = sorted({str(self.selection.loc[name, f"B{i}"]) for name in self.downstream_names for i in range(1, 4)})
        specs = [(name, "none") for name in self.downstream_names]
        specs.extend((name, "b_none") for name in b_names)
        self._run_pool(worker_none, specs, "none_cache")
        self._log("完成 prepare_AB_none")

    def calc_final_resid(self) -> None:
        self._log("开始 calc_final_resid")
        self._require_context()
        self._load_selection()
        self._run_pool(worker_final, self.downstream_names, "final_resid")
        self._log("完成 calc_final_resid")

    def backtest_final_resid(self) -> None:
        self._log("开始 backtest_final_resid")
        self._require_context()
        specs = []
        for name in self.downstream_names:
            specs.extend(((name, "final_resid1"), (name, "final_resid2")))
        self._run_pool(worker_perf, specs, "final_perf")
        self._log("完成 backtest_final_resid")

    def summarize(self) -> None:
        self._log("开始 summarize")
        self._require_output()
        self._load_selection()
        rows = []
        for name in self.candidate_names:
            row = {"factor_name": name}
            row.update({f"base_{key}": value for key, value in self._metrics(self._file("base_perf", name, ".csv")).items()})
            if name in self.selection.index:
                row.update(self.selection.loc[name].to_dict())
            for method in ("final_resid1", "final_resid2"):
                perf_path = self._file(f"{method}_perf", name, ".csv")
                if perf_path.exists():
                    row.update({f"{method}_{key}": value for key, value in self._metrics(perf_path).items()})
            rows.append(row)
        summary = pd.DataFrame(rows).set_index("factor_name")
        atomic_csv(summary, self.output_path / "summary.csv")
        atomic_json(
            {
                "status": "complete",
                "candidate_count": len(self.candidate_names),
                "downstream_union_count": len(self.downstream_names),
                "existing_ic5_count": int(self._load_existing_ic5().shape[1]),
                "unique_b_count": int(pd.read_csv(self.output_path / "selection" / "unique_b_names.csv").shape[0]),
                "start_date": self.start_date,
                "end_date": self.end_date,
                "outsample_start_date": self.outsample_start_date,
                "hold_prd": self.HOLD_PRD,
                "benchmark": self.BENCHMARK,
                "workers": self.workers,
                "threads_per_worker": 1,
                "final_resid1_x_columns": 13,
                "final_resid2_x_columns": 74,
                "final_resid2_basis": "65 Barra + 9 B = 74; B = 3 linear + 3 squares + 3 pairwise interactions",
                "candidate_names_sha256": hashlib.sha256("\n".join(self.candidate_names).encode("utf-8")).hexdigest(),
            },
            self.output_path / "manifest.json",
        )
        self._log("完成 summarize，流程结束")

    def _log(self, message: str, level: int = logging.INFO) -> None:
        self.logger.log(level, message)

    def _resolve_factor_path(self, path: str) -> Path:
        candidate = Path(path).expanduser()
        if candidate.exists():
            return candidate
        return self.INPUT_ROOT / "factors" / str(path)

    def _validate_paths(self) -> None:
        if self.output_path is not None:
            self.output_path.mkdir(parents=True, exist_ok=True)
            self.logger = configure_logger(self.output_path / "run.log")
            self._log(
                f"参数：factor_path={self.factor_path}, exist_ic5_path={self.exist_ic5_path}, "
                f"start={self.start_date}, end={self.end_date}, "
                f"outsample={self.outsample_start_date}, workers={self.workers}"
            )
        if self.factor_path is None or not self.factor_path.exists():
            raise FileNotFoundError(f"factor_path is invalid: {self.factor_path}")
        if self.exist_ic5_path is None or not self.exist_ic5_path.exists():
            raise FileNotFoundError(f"exist_ic5_path is invalid: {self.exist_ic5_path}")
        if self.output_path is None:
            raise ValueError("请先填写 output_path")
        if pd.Timestamp(self.start_date) > pd.Timestamp(self.end_date):
            raise ValueError("start_date must not be later than end_date")
        if self.outsample_start_date:
            if not pd.Timestamp(self.start_date) <= pd.Timestamp(self.outsample_start_date) <= pd.Timestamp(self.end_date):
                raise ValueError("outsample_start_date must be inside the backtest range")
        self._log("路径校验通过")

    def _require_output(self) -> None:
        if self.output_path is None:
            raise ValueError("请先填写 output_path")

    def _require_context(self) -> None:
        self._require_output()
        if self.restrict.empty:
            self._load_context()
        if not self.candidate_names:
            self._read_candidate_names()

    def _load_context(self) -> None:
        self._log("开始加载公共数据、Barra 和 H5 轴")
        calendar = pd.read_csv(self.INPUT_ROOT / "basic_info" / "calendar.csv", header=None).iloc[:, 0]
        all_dates = pd.to_datetime(calendar.astype(str), format="%Y%m%d")
        symbols = pd.read_csv(self.INPUT_ROOT / "basic_info" / "symbol_map.csv", dtype={"symbol": str})
        all_stocks = pd.Index([windcode(value) for value in symbols["symbol"]])
        start = pd.Timestamp(self.start_date)
        end = pd.Timestamp(self.end_date)
        end_position = int(np.searchsorted(all_dates.values, end.to_datetime64(), side="right"))
        end_position = min(len(all_dates), end_position + 20)
        mask = (all_dates >= start) & (np.arange(len(all_dates)) < end_position)
        self.dates = pd.DatetimeIndex(all_dates[mask])
        self.stocks = all_stocks
        self.restrict = self._read_panel("vars/Base/S_RESTRICT_bt.h5").loc[self.dates, self.stocks]
        self.industry = self._read_panel("vars/Base/SW_IND_CODE.h5").loc[self.dates, self.stocks]
        self.returns = self._read_panel("vars/Base/S_DQ_RET_VWAP_0930_1000.h5").loc[self.dates, self.stocks]
        log_return = np.log1p(self.returns)
        market_mean = log_return.where(self.restrict.shift(1) == 0).mean(axis=1)
        self.log_exret = log_return.sub(market_mean, axis=0)
        self.barra = [self._read_panel(f"vars/SzBa/{name}.h5").loc[self.dates, self.stocks].rank(axis=1, pct=True) for name in BARRA_NAMES]
        self.fill_base = [self._read_panel("vars/SzBa/size.h5").loc[self.dates, self.stocks].rank(axis=1, pct=True)]
        self.factor_files = self._build_factor_file_map()
        self._log(f"公共数据加载完成：dates={len(self.dates)}, stocks={len(self.stocks)}")

    def _read_panel(self, relative: str) -> pd.DataFrame:
        return self._read_h5(self.INPUT_ROOT / relative)

    def _read_h5(self, path: Path) -> pd.DataFrame:
        if not path.exists():
            raise FileNotFoundError(path)
        with h5py.File(path, "r") as store:
            values = store["data"][:len(self._all_dates()), :len(self._all_stocks())]
        return pd.DataFrame(values, index=self._all_dates(), columns=self._all_stocks())

    def _all_dates(self) -> pd.DatetimeIndex:
        calendar = pd.read_csv(self.INPUT_ROOT / "basic_info" / "calendar.csv", header=None).iloc[:, 0]
        return pd.DatetimeIndex(pd.to_datetime(calendar.astype(str), format="%Y%m%d"))

    def _all_stocks(self) -> pd.Index:
        symbols = pd.read_csv(self.INPUT_ROOT / "basic_info" / "symbol_map.csv", dtype={"symbol": str})
        return pd.Index([windcode(value) for value in symbols["symbol"]])

    def _build_factor_file_map(self) -> dict[str, Path]:
        root = self.INPUT_ROOT / "factors"
        paths = sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in {".h5", ".parquet", ".csv"})
        return {path.stem: path for path in paths}

    def _read_candidate_names(self) -> list[str]:
        if self.factor_path is None:
            raise ValueError("请先填写 factor_path")
        if self.factor_path.is_file():
            names = [self.factor_path.stem]
        else:
            factor_suffixes = {".h5", ".parquet", ".csv"}
            field_names = None
            fields_path = self.factor_path / "fields.csv"
            if fields_path.exists():
                fields = pd.read_csv(fields_path, usecols=["name"])
                field_names = set(fields["name"].astype(str))
            names = sorted(
                path.stem
                for path in self.factor_path.iterdir()
                if path.is_file()
                and path.suffix.lower() in factor_suffixes
                and (field_names is None or path.stem in field_names)
            )
        if not names:
            raise ValueError(f"factor_path contains no factor files: {self.factor_path}")
        if len(names) != len(set(names)):
            raise ValueError("candidate factor names are duplicated")
        self.candidate_names = names
        atomic_csv(pd.DataFrame({"factor_name": names}), self.output_path / "candidate_names.csv")
        return names

    def _raw_factor(self, name: str) -> pd.DataFrame:
        if self.factor_path is None:
            raise ValueError("factor_path is not set")
        if self.factor_path.is_file() and self.factor_path.stem == name:
            path = self.factor_path
        else:
            path = next((self.factor_path / f"{name}{suffix}" for suffix in (".h5", ".parquet", ".csv") if (self.factor_path / f"{name}{suffix}").exists()), None)
        if path is None:
            path = self.factor_files.get(name)
        if path is None or not path.exists():
            raise FileNotFoundError(f"raw factor file not found: {name}")
        if path.suffix.lower() == ".h5":
            value = self._read_h5(path)
        elif path.suffix.lower() == ".parquet":
            value = pd.read_parquet(path)
        else:
            value = pd.read_csv(path, index_col=0, parse_dates=True)
        value.index = pd.to_datetime(value.index)
        return value.reindex(index=self.dates, columns=self.stocks)

    def _file(self, kind: str, name: str, suffix: str) -> Path:
        folders = {
            "base_resid": "base_resid", "base_perf": "base_perf", "none": "none_cache",
            "b_none": "none_cache_existing", "final_resid1": "final_resid1",
            "final_resid1_perf": "final_resid1_perf", "final_resid2": "final_resid2",
            "final_resid2_perf": "final_resid2_perf",
        }
        return self.output_path / folders[kind] / (safe_name(name) + suffix)

    def _fill_industry_regression(self, value: pd.DataFrame) -> pd.DataFrame:
        values = value.to_numpy(dtype=float, copy=True)
        bases = np.stack([base.to_numpy(dtype=float) for base in self.fill_base], axis=2)
        industry = self.industry.to_numpy(dtype=float)
        with np.errstate(invalid="ignore"):
            levels = [industry // 100, industry // 10000, (~np.isnan(industry // 10000)).astype(float)]
        for codes in levels:
            for row in range(values.shape[0]):
                if len(np.unique(values[row][np.isfinite(values[row])])) < 10:
                    continue
                for code in np.unique(codes[row][np.isfinite(codes[row])]):
                    mask = codes[row] == code
                    valid = np.isfinite(values[row]) & np.isfinite(bases[row]).all(axis=1) & mask
                    missing = ~np.isfinite(values[row]) & np.isfinite(bases[row]).all(axis=1) & mask
                    if valid.sum() < 10 or not missing.any():
                        continue
                    x = np.column_stack((np.ones(valid.sum()), bases[row, valid]))
                    coef, *_ = np.linalg.lstsq(x, values[row, valid], rcond=None)
                    values[row, missing] = np.column_stack((np.ones(missing.sum()), bases[row, missing])) @ coef
        return pd.DataFrame(values, index=value.index, columns=value.columns)

    def _fill_industry_median(self, value: pd.DataFrame) -> pd.DataFrame:
        values = value.to_numpy(dtype=float, copy=True)
        source = values.copy()
        industry = self.industry.to_numpy(dtype=float)
        with np.errstate(invalid="ignore"):
            codes_list = [industry // 100, industry // 10000, np.zeros_like(industry)]
            valid_industry = np.isfinite(industry // 10000)
        values[~valid_industry] = np.nan
        for codes in codes_list:
            for row in range(values.shape[0]):
                missing = ~np.isfinite(values[row])
                for code in np.unique(codes[row][np.isfinite(codes[row])]):
                    group = codes[row] == code
                    source_values = source[row, group & np.isfinite(source[row])]
                    if source_values.size:
                        values[row, missing & group] = np.median(source_values)
        return pd.DataFrame(values, index=value.index, columns=value.columns)

    def _residual(self, value: pd.DataFrame, neutralizers: list[pd.DataFrame]) -> pd.DataFrame:
        y_all = value.to_numpy(dtype=float)
        x_all = [item.to_numpy(dtype=float) for item in neutralizers]
        output = np.full(value.shape, np.nan)
        for row in range(value.shape[0]):
            y = y_all[row]
            x = np.column_stack([item[row] for item in x_all])
            valid = np.isfinite(y) & np.isfinite(x).all(axis=1)
            if valid.sum() <= 10:
                continue
            if np.unique(y[valid]).size == 1:
                output[row, valid] = 0.5
                continue
            design = np.column_stack((np.ones(valid.sum()), x[valid]))
            coef, *_ = np.linalg.lstsq(design, y[valid], rcond=None)
            output[row, valid] = y[valid] - design @ coef
        result = pd.DataFrame(output, index=value.index, columns=value.columns)
        return result.rank(axis=1, pct=True)

    def _prepare_base_linear_cache(self) -> None:
        """预计算 base_resid 共用的 Barra 回归矩阵。"""
        if not self.candidate_names or self._base_gram_inv is not None:
            return

        template = self._fill_rank(
            self._raw_factor(self.candidate_names[0]), "none"
        )
        template_values = template.to_numpy(dtype=float)
        barra_values = [item.to_numpy(dtype=float) for item in self.barra]
        valid_mask = np.isfinite(template_values)
        for item in barra_values:
            valid_mask &= np.isfinite(item)

        feature_count = len(barra_values) + 1
        gram_inv = np.full(
            (len(self.dates), feature_count, feature_count),
            np.nan,
            dtype=float,
        )
        for row, valid in enumerate(valid_mask):
            if valid.sum() <= 10:
                continue
            y = template_values[row, valid]
            if np.unique(y).size == 1:
                continue
            design = np.column_stack(
                [
                    np.ones(valid.sum()),
                    *[item[row, valid] for item in barra_values],
                ]
            )
            gram = design.T @ design
            try:
                gram_inv[row] = np.linalg.inv(gram)
            except np.linalg.LinAlgError:
                gram_inv[row] = np.linalg.pinv(gram)

        self._base_valid_mask = valid_mask
        self._base_gram_inv = gram_inv
        self._log(
            f"base Barra 回归缓存完成：dates={len(self.dates)}，"
            f"features={feature_count}"
        )

    def _residual_with_base_cache(self, value: pd.DataFrame) -> pd.DataFrame:
        if self._base_valid_mask is None or self._base_gram_inv is None:
            return self._residual(value, self.barra)

        y_all = value.to_numpy(dtype=float)
        barra_values = [item.to_numpy(dtype=float) for item in self.barra]
        output = np.full(value.shape, np.nan)
        for row, valid in enumerate(self._base_valid_mask):
            if valid.sum() <= 10:
                continue
            y = y_all[row, valid]
            if np.unique(y).size == 1:
                output[row, valid] = 0.5
                continue
            gram_inv = self._base_gram_inv[row]
            if not np.isfinite(gram_inv).all():
                return self._residual(value, self.barra)
            design = np.column_stack(
                [
                    np.ones(valid.sum()),
                    *[item[row, valid] for item in barra_values],
                ]
            )
            coef = gram_inv @ (design.T @ y)
            output[row, valid] = y - design @ coef

        result = pd.DataFrame(output, index=value.index, columns=value.columns)
        return result.rank(axis=1, pct=True)

    def _fill_rank(self, raw: pd.DataFrame, neutralize: str) -> pd.DataFrame:
        value = raw.rank(axis=1, pct=True)
        value = self._fill_industry_regression(value)
        value = self._fill_industry_median(value)
        value = value.where(self.restrict == 0)
        value = value.rank(axis=1, pct=True)
        if neutralize == "none":
            return value
        if neutralize == "linear":
            return self._residual(value, self.barra)
        raise ValueError(f"unknown neutralize method: {neutralize}")

    def _calc_one_base(self, name: str) -> None:
        path = self._file("base_resid", name, ".parquet")
        if not path.exists():
            value = self._fill_rank(self._raw_factor(name), "none")
            value_mask = np.isfinite(value.to_numpy(dtype=float))
            if self._base_valid_mask is not None and np.array_equal(value_mask, self._base_valid_mask):
                value = self._residual_with_base_cache(value)
            else:
                value = self._residual(value, self.barra)
            atomic_parquet(value, path)

    def _calc_one_none(self, name: str, kind: str) -> None:
        path = self._file(kind, name, ".parquet")
        if not path.exists():
            atomic_parquet(self._fill_rank(self._raw_factor(name), "none"), path)

    def _normal_score(self, value: pd.DataFrame) -> pd.DataFrame:
        ranks = value.rank(axis=1, method="average")
        counts = value.notna().sum(axis=1).replace(0, np.nan)
        uniforms = ranks.sub(0.5).div(counts, axis=0).clip(1e-3, 1 - 1e-3)
        values = ndtri(uniforms.to_numpy(dtype=float))
        values[~value.notna().to_numpy()] = np.nan
        return pd.DataFrame(values, index=value.index, columns=value.columns)

    def _basis(self, values: np.ndarray) -> np.ndarray:
        parts = [values, values * values]
        parts.extend((values[:, left] * values[:, right])[:, None] for left in range(values.shape[1]) for right in range(left + 1, values.shape[1]))
        basis = np.column_stack(parts)
        mean = basis.mean(axis=0)
        std = basis.std(axis=0)
        keep = np.isfinite(std) & (std > 1e-10)
        return (basis[:, keep] - mean[keep]) / std[keep]

    def _block_locscale(self, factor: pd.DataFrame, b_values: list[pd.DataFrame]) -> pd.DataFrame:
        factor_z = self._normal_score(factor).to_numpy(dtype=float)
        barra_z = [self._normal_score(item.where(factor.notna())).to_numpy(dtype=float) for item in self.barra]
        b_z = [self._normal_score(item.where(factor.notna())).to_numpy(dtype=float) for item in b_values]
        output = np.full(factor.shape, np.nan)
        for row in range(len(factor.index)):
            y = factor_z[row]
            barra_x = np.column_stack([item[row] for item in barra_z])
            b_x = np.column_stack([item[row] for item in b_z])
            valid = np.isfinite(y) & np.isfinite(barra_x).all(axis=1) & np.isfinite(b_x).all(axis=1)
            if valid.sum() < 300:
                continue
            basis = np.column_stack((self._basis(barra_x[valid]), self._basis(b_x[valid])))
            design = np.column_stack((np.ones(valid.sum()), basis))
            coef, *_ = np.linalg.lstsq(design, y[valid], rcond=None)
            residual = y[valid] - design @ coef
            log_var = np.log(residual * residual + 1e-6)
            var_coef, *_ = np.linalg.lstsq(design, log_var, rcond=None)
            predicted = design @ var_coef
            lower, upper = np.quantile(predicted, (0.05, 0.95))
            scale = np.clip(np.exp(0.5 * np.clip(predicted, lower, upper)), 0.10, 10.0)
            output[row, valid] = residual / scale
        return pd.DataFrame(output, index=factor.index, columns=factor.columns).rank(axis=1, pct=True)

    def _calc_one_final(self, name: str) -> None:
        self._load_selection()
        a = pd.read_parquet(self._file("none", name, ".parquet"))
        b_names = [str(self.selection.loc[name, f"B{i}"]) for i in range(1, 4)]
        b_values = [pd.read_parquet(self._file("b_none", b_name, ".parquet")) for b_name in b_names]
        final1_path = self._file("final_resid1", name, ".parquet")
        if not final1_path.exists():
            atomic_parquet(self._residual(a, self.barra + b_values), final1_path)
        final2_path = self._file("final_resid2", name, ".parquet")
        if not final2_path.exists():
            atomic_parquet(self._block_locscale(a, b_values), final2_path)

    def _build_forward_return_ranks(
        self,
        horizons: tuple[int, ...],
        index: pd.Index | None = None,
    ) -> dict[int, pd.DataFrame]:
        result = {}
        for horizon in horizons:
            future = self.returns.rolling(horizon, min_periods=horizon).sum().shift(-horizon)
            rank = future.rank(axis=1, pct=True)
            if index is not None:
                rank = rank.reindex(index=index)
            rank.iloc[-horizon - 1:] = np.nan
            result[horizon] = rank
        return result

    def _compute_factor_ic(
        self,
        factor_rank: pd.DataFrame,
        return_rank: pd.DataFrame,
        horizon: int,
    ) -> pd.Series:
        factor_rank = factor_rank.reindex(index=return_rank.index, columns=return_rank.columns)
        unique_mask = factor_rank.nunique(axis=1, dropna=True) == 1
        ic = factor_rank.corrwith(return_rank, axis=1)
        ic.loc[unique_mask] = 0.0
        return ic.shift(horizon + 1)

    def _shift_perf_metric_cols(self, perf: pd.DataFrame) -> pd.DataFrame:
        end = perf.columns.get_loc("longshort_sharpe") + 1
        perf.iloc[:, :end] = perf.iloc[:, :end].shift(1)
        return perf

    def _performance(self, value: pd.DataFrame) -> pd.DataFrame:
        factor = value.reindex(index=self.dates, columns=self.stocks)
        factor.loc[factor.std(1) == 0] = np.nan

        factor_coverage = factor.count(axis=1)
        if (factor_coverage >= 10).any():
            first_not_empty_date = factor_coverage[factor_coverage >= 10].index[0]
        else:
            first_not_empty_date = factor_coverage.index[-1]
        first_not_empty_date = max(pd.Timestamp(self.start_date), first_not_empty_date)

        coverage = factor_coverage.loc[first_not_empty_date:]
        factor[(self.restrict == 1) | self.restrict.isna()] = np.nan
        after_coverage = factor.count(axis=1).loc[first_not_empty_date:]
        full_coverage = self.restrict[self.restrict == 0].count(1).loc[first_not_empty_date:]

        percentile = factor.rank(axis=1, method="average", pct=True)
        percentile.loc[percentile.count(1) <= 10] = np.nan
        group_number = 10 - ((1 - percentile) * 10) // 1
        daily_group_number = group_number.fillna(-9999).shift(1)
        weekly_group_number = (
            daily_group_number.resample("W")
            .first()
            .resample("D")
            .first()
            .shift(-6)
            .ffill()
            .reindex(daily_group_number.index)
        )
        weekly_group_number.loc[:pd.Timestamp(self.start_date), :] = np.nan

        output_list = []
        for number in range(11):
            weights = pd.DataFrame(0, index=self.dates, columns=self.stocks)
            if number == 0:
                weights[weekly_group_number > 0] = 1
            else:
                weights[weekly_group_number == number] = 1
            weights = (weights.T / weights.sum(1)).T

            ret = (weights * self.log_exret).sum(axis=1, min_count=1)
            mean = ret.expanding().mean() * 250
            std = ret.expanding().std() * 250 ** 0.5
            sharpe = mean / std
            count = weights[weights > 0].count(1).loc[first_not_empty_date:]
            cumret = ret.expanding().sum()
            max_cumret = cumret.expanding().max()
            drawdown = cumret - max_cumret
            max_drawdown = drawdown.expanding().min()

            group = pd.concat([ret, mean, std, sharpe, drawdown, max_drawdown, count], axis=1)
            group.columns = [
                f"group{number}_{key}"
                for key in ("ret", "mean", "std", "sharpe", "dd", "mdd", "count")
            ]
            output_list.append(group)

        perf = pd.concat(output_list, axis=1)
        perf["longshort_ret"] = perf["group10_ret"] - perf["group1_ret"]
        perf["longshort_mean"] = perf["longshort_ret"].expanding().mean() * 250
        perf["longshort_std"] = perf["longshort_ret"].expanding().std() * 250 ** 0.5
        perf["longshort_sharpe"] = perf["longshort_mean"] / perf["longshort_std"]
        perf["coverage"] = coverage
        perf["after_coverage"] = after_coverage
        perf["full_coverage"] = full_coverage
        perf["coverage_rate"] = after_coverage / full_coverage
        perf["mean_coverage"] = perf["coverage_rate"].expanding().mean()

        return_ranks = self._build_forward_return_ranks((5, 10, 20), index=self.dates)
        for horizon in (5, 10, 20):
            perf[f"ic{horizon}"] = self._compute_factor_ic(
                percentile, return_ranks[horizon], horizon
            ).reindex(perf.index)

        perf["TLQ"] = np.nan
        perf = perf.loc[self.start_date:self.end_date].copy()
        perf["TLQ"] = self._tlq(perf)
        perf["start_date"] = first_not_empty_date
        perf["end_date"] = pd.Series(perf.index, perf.index)
        perf["T"] = (perf["end_date"] - perf["start_date"]).dt.days / 365.25
        return self._shift_perf_metric_cols(perf)

    def _tlq(self, perf: pd.DataFrame, curve_points: int = 160) -> float:
        group_columns = [f"group{i}_ret" for i in range(1, 11)]
        if len(perf) < 2 or any(column not in perf.columns for column in group_columns):
            return np.nan

        curves = []
        for column in group_columns:
            values = pd.to_numeric(perf[column], errors="coerce").fillna(0.0).cumsum().to_numpy(dtype=float)
            positions = np.rint(np.linspace(0, len(values) - 1, curve_points)).astype(int)
            # 分段口径：取相邻两个采样点之间的增量；原来的 values[positions] 是水位口径
            curves.append(np.diff(np.concatenate([[0.0], values])[positions + 1]))
        curves = np.stack(curves).astype(float, copy=False)
        if not np.isfinite(curves).all():
            return np.nan

        gaps = np.diff(curves, axis=0)
        spread = np.max(curves, axis=0) - np.min(curves, axis=0)
        signed_endpoint = np.divide(
            curves[-1] - curves[0], spread,
            out=np.zeros_like(spread), where=spread > 1e-12,
        )
        movement = np.sum(np.abs(gaps), axis=0)
        violation_up = np.divide(
            np.sum(np.maximum(-gaps, 0), axis=0), movement,
            out=np.zeros_like(movement), where=movement > 1e-12,
        )
        violation_down = np.divide(
            np.sum(np.maximum(gaps, 0), axis=0), movement,
            out=np.zeros_like(movement), where=movement > 1e-12,
        )
        quality_up = np.mean(signed_endpoint * (1 - violation_up))
        quality_down = np.mean(-signed_endpoint * (1 - violation_down))
        return float(max(0.0, quality_up, quality_down))

    def _backtest_one(self, name: str, method: str) -> None:
        value = pd.read_parquet(self._file(method, name, ".parquet"))
        perf = self._performance(value)
        perf_kind = "base_perf" if method == "base_resid" else f"{method}_perf"
        atomic_csv(perf, self._file(perf_kind, name, ".csv"))
        self._plot(perf, name, method)

    def _plot(self, perf: pd.DataFrame, name: str, method: str) -> None:
        import matplotlib.pyplot as plt

        plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
        plt.rcParams["font.size"] = 16
        width = 12
        fig = plt.figure(figsize=(width * 1.41, width))
        wrapped_title = "\n".join(textwrap.wrap(f"{name}", width=80))
        fig.suptitle(wrapped_title, fontsize=20, y=0.96, fontweight="bold")

        ax = fig.add_axes([0.1, 0.47, 0.8, 0.4])
        info = perf.iloc[-1]
        group1_mean = info["group1_mean"]
        group10_mean = info["group10_mean"]
        ic5_mean = perf["ic5"].mean()
        tlq = float(info.get("TLQ", np.nan))
        ic_label = f"ic5 = {abs(ic5_mean):.3f}, TLQ = {tlq:.3f}"
        mean_coverage = info["mean_coverage"]
        if group1_mean > group10_mean:
            max_drawdown = info["group1_mdd"]
            ax.set_title(
                f"group1: {group1_mean:.1%}(mdd: {max_drawdown:.1%}), "
                f"group10: {group10_mean:.1%}, {ic_label}"
            )
        else:
            max_drawdown = info["group10_mdd"]
            ax.set_title(
                f"group1: {group1_mean:.1%}, "
                f"group10: {group10_mean:.1%}(mdd: {max_drawdown:.1%}), {ic_label}"
            )

        zero_line = pd.Series(0, index=perf.index)
        ax.plot(zero_line, linewidth=1, color="white")
        ax.plot(perf["group1_ret"].cumsum().ffill(), label="group1", linewidth=2, color="blue")
        for number in range(2, 10):
            column = f"group{number}_ret"
            if column in perf.columns:
                ax.plot(perf[column].cumsum().ffill(), label=f"group{number}", linewidth=1)
        ax.plot(perf["group1_ret"].cumsum().ffill(), linewidth=2, color="blue")
        ax.plot(perf["group10_ret"].cumsum().ffill(), label="group10", linewidth=2, color="red")
        ax.plot(perf["longshort_ret"].cumsum().ffill(), linewidth=2, color="black")
        if "group0_ret" in perf.columns:
            ax.plot(perf["group0_ret"].cumsum().ffill(), label="all", linewidth=3, color="grey")
        ax.legend(loc=2)

        ax2 = fig.add_axes([0.1, 0.1, 0.8, 0.3])
        ax2.set_title(f"T = {round(info['T'], 1)} years, mean_coverage = {mean_coverage:.1%}")
        ax2.plot(zero_line, linewidth=1, color="white")
        ax2.plot(perf["coverage_rate"], label="after_coverage", color="black")
        ax2.set_ylim([-0.05, 1.05])
        if self.outsample_start_date:
            marker = pd.Timestamp(self.outsample_start_date)
            ax.axvline(marker, color="red", linestyle="--", linewidth=1.5)
            ax2.axvline(marker, color="red", linestyle="--", linewidth=1.5)
        path = self.output_path / "plots" / method / f"{safe_name(name)}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150)
        plt.close(fig)

    def _metrics(self, path: Path) -> dict[str, Any]:
        frame = pd.read_csv(path, index_col=0, parse_dates=True)
        ic5 = pd.to_numeric(frame.get("ic5"), errors="coerce").dropna()
        ls = pd.to_numeric(frame.get("longshort_mean"), errors="coerce").dropna()
        tlq = pd.to_numeric(frame.get("TLQ"), errors="coerce").dropna()
        return {
            "abs_mean_ic5": abs(float(ic5.mean())) if len(ic5) else np.nan,
            "abs_longshort_annual": abs(float(ls.iloc[-1])) if len(ls) else np.nan,
            "TLQ": float(tlq.iloc[-1]) if len(tlq) else np.nan,
        }

    def _load_existing_ic5(self) -> pd.DataFrame:
        frame = pd.read_parquet(self.exist_ic5_path)
        frame.index = pd.to_datetime(frame.index)
        frame = frame.loc[self.start_date:self.end_date].sort_index().apply(pd.to_numeric, errors="coerce")
        if frame.columns.duplicated().any():
            raise ValueError("existing IC5 contains duplicate columns")
        if frame.shape[1] < 3:
            raise ValueError("existing IC5 must contain at least three factors")
        return frame

    def _load_selection(self) -> None:
        self._require_output()
        path = self.output_path / "selection" / "selection_top3.csv"
        if self.selection.empty and path.exists():
            self.selection = pd.read_csv(path, index_col=0)
        union_path = self.output_path / "selection" / "downstream_union.csv"
        if not self.downstream_names and union_path.exists():
            self.downstream_names = pd.read_csv(union_path)["factor_name"].astype(str).tolist()

    def _run_pool(self, worker: Any, items: Iterable[Any], label: str) -> None:
        items = list(items)
        if not items:
            return
        global WORKER_GATE
        WORKER_GATE = self
        if "fork" not in mp.get_all_start_methods():
            for item in items:
                worker(item)
            return
        count = min(self.workers, len(items))
        self._log(f"开始 {label}：任务数={len(items)}，进程数={count}")
        with mp.get_context("fork").Pool(count) as pool:
            try:
                for number, _ in enumerate(pool.imap_unordered(worker, items, chunksize=1), 1):
                    if number % 20 == 0 or number == len(items):
                        self._log(f"{label}: {number}/{len(items)}，workers={count}")
            except Exception:
                self._log(f"{label} 失败", logging.ERROR)
                raise


def main() -> None:
    fg = FactorGate()
    fg.set_factor_path("/nas197/binary/stock/sz_alpha/RDisk/result_temp/G3_v2/factors_original_name")
    fg.set_exist_ic5_path("/nas197/binary/stock/sz_alpha/RDisk/result_temp/sx7/weekly_train/2026-08-14/factor_info/resid_df_open_vwap2open_vwap-5.parquet")
    fg.set_output_path("/nas197/binary/stock/liquehao/file_deliver/g3_final_test")
    fg.set_start_date("2016-01-04")
    fg.set_end_date("2026-07-30")
    fg.set_outsample_start_date("2025-08-01")
    fg.set_corr_thres(0.6)
    fg.calc_base_resid()
    fg.backtest_base_resid()
    fg.get_greedy_list()
    fg.get_top3()
    fg.prepare_AB_none()
    fg.calc_final_resid()
    fg.backtest_final_resid()
    fg.summarize()


if __name__ == "__main__":
    main()
