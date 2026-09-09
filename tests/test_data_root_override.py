"""data_root 覆盖验证（三个 pipeline 的原始数据根参数）。

验证点：
1. reader 级：环境变量 RUST_PYFUNC_DATA_ROOT 覆盖 Level2 / 分钟数据根；
   根或日期缺失时直接报错，不再回退 /nas197/...。
2. pipeline 级：data_root 参数经 worker 子进程生效——用"只放一天数据的临时根"证明：
   临时根里没放的那天算不出东西（默认根里其实是有的）。

用法：python tests/test_data_root_override.py
"""

import os
import shutil
import tempfile

import rust_pyfunc as rp

DATE_OK = 20241231      # 临时根里放这一天
DATE_MISS = 20241230    # 临时根里不放这一天（默认根里存在）
CODE = "000001"
DEFAULT_ROOT = "/ssd_data"


def check(name: str, cond: bool) -> None:
    print(("✅ " if cond else "❌ ") + name)
    assert cond, name


def completed_dates(store: str) -> set:
    path = f"{store}/_completed_dates"
    return set(int(x) for x in open(path)) if os.path.exists(path) else set()


tmp = tempfile.mkdtemp(prefix="dsh_data_root_")
os.makedirs(f"{tmp}/stock", exist_ok=True)
os.makedirs(f"{tmp}/data", exist_ok=True)
os.symlink(f"{DEFAULT_ROOT}/stock/{DATE_OK}", f"{tmp}/stock/{DATE_OK}")
os.symlink(f"{DEFAULT_ROOT}/data/1min_factor_text", f"{tmp}/data/1min_factor_text")

try:
    # ============ 1. reader 级：环境变量覆盖 ============
    v_default = rp.py_distill(CODE, DATE_OK)
    os.environ["RUST_PYFUNC_DATA_ROOT"] = tmp
    check("环境变量覆盖后 py_distill 与默认根结果一致", rp.py_distill(CODE, DATE_OK) == v_default)
    try:
        rp.py_distill(CODE, DATE_MISS)
        check("临时根缺该日期应报错", False)
    except Exception:
        check("临时根缺该日期 → 报错（未回退）", True)
    os.environ["RUST_PYFUNC_DATA_ROOT"] = "/tmp/__no_such_root__"
    try:
        rp.py_distill(CODE, DATE_OK)
        check("数据根不存在应报错", False)
    except Exception:
        check("数据根不存在 → 报错（未回退 /nas197）", True)
    del os.environ["RUST_PYFUNC_DATA_ROOT"]

    # ============ 2. 横截面 pipeline：data_root 参数 ============
    names_cs = rp.py_cross_section_example_names()

    def run_cs(store: str, tasks: list) -> None:
        rp.run_factor_pipeline_cross_section(
            pipeline="cross_section_example",
            tasks=tasks,
            n_jobs=20,
            expected_result_length=len(names_cs),
            trading_days=list(rp.td.trading_days),
            store_dir=store,
            store_factor_names=names_cs,
            data_root=tmp,
        )

    store_cs = f"{tmp}/store_cs"
    run_cs(store_cs, [DATE_OK])
    info_cs = rp.factor_store_v5_info(store_cs)
    check(
        "cross_section：data_root 覆盖生效（临时根里的日期算出数据并投影）",
        completed_dates(store_cs) == {DATE_OK} and info_cs["is_projected"] and info_cs["record_count"] > 0,
    )
    try:
        run_cs(f"{tmp}/store_cs_miss", [DATE_MISS])
        check("cross_section：临时根缺该日期应报错", False)
    except RuntimeError:
        check("cross_section：临时根缺该日期 → 报错（未读默认根）", True)

    # ============ 3. 分钟 pipeline：data_root 参数 ============
    names_m = rp.py_minute_example_names()

    def run_minute(store: str, tasks: list) -> None:
        rp.run_factor_pipeline_minute(
            pipeline="minute_example",
            tasks=tasks,
            n_jobs=2,
            expected_result_length=len(names_m),
            trading_days=list(rp.td.trading_days),
            store_dir=store,
            store_factor_names=names_m,
            data_root=tmp,
        )

    store_m = f"{tmp}/store_m"
    run_minute(store_m, [DATE_OK])
    check("minute：data_root 覆盖生效", completed_dates(store_m) == {DATE_OK})
    # 负例用"不存在的根"：若覆盖被忽略，会照默认根正常出数（分钟数据是整天一个大 h5，
    # 单日缺失无法构造，故用整根不存在来证明确实没读默认根）。
    store_m_bad = f"{tmp}/store_m_bad"
    try:
        rp.run_factor_pipeline_minute(
            pipeline="minute_example",
            tasks=[DATE_OK],
            n_jobs=2,
            expected_result_length=len(names_m),
            trading_days=list(rp.td.trading_days),
            store_dir=store_m_bad,
            store_factor_names=names_m,
            data_root="/tmp/__no_such_root__",
        )
    except RuntimeError:
        pass
    check("minute：data_root 指向不存在根 → 无结果（未读默认根）", completed_dates(store_m_bad) == set())

    # ============ 4. Level2 per-stock pipeline：data_root 参数 ============
    names_l2 = rp.py_distill_names()

    def run_distill(store: str, root: str) -> None:
        rp.run_factor_pipeline(
            pipeline="distill",
            tasks=[[DATE_OK, CODE]],
            n_jobs=1,
            backup_file="",
            expected_result_length=len(names_l2),
            trading_days=list(rp.td.trading_days),
            store_dir=store,
            store_factor_names=names_l2,
            data_root=root,
        )

    store_ok = f"{tmp}/store_l2_ok"
    run_distill(store_ok, tmp)
    facs_ok = rp.factor_store_v5_read_factor(store_ok, 0)["factor"]
    check("level2：data_root 指向有效根 → 因子非空", len(facs_ok) > 0)

    store_bad = f"{tmp}/store_l2_bad"
    run_distill(store_bad, "/tmp/__no_such_root__")
    facs_bad = rp.factor_store_v5_read_factor(store_bad, 0)["factor"]
    check("level2：data_root 指向不存在根 → 因子全 NaN（读回为空）", len(facs_bad) == 0)

    # ============ 5. 不传 data_root：默认根仍可用（覆盖不残留） ============
    store_def = f"{tmp}/store_default"
    rp.run_factor_pipeline_cross_section(
        pipeline="cross_section_example",
        tasks=[DATE_OK],
        n_jobs=20,
        expected_result_length=len(names_cs),
        trading_days=list(rp.td.trading_days),
        store_dir=store_def,
        store_factor_names=names_cs,
    )
    check("不传 data_root → 默认根 /ssd_data 正常出数（覆盖已还原）", completed_dates(store_def) == {DATE_OK})

    print("\n🎉 data_root 覆盖验证全部通过")
finally:
    shutil.rmtree(tmp, ignore_errors=True)
