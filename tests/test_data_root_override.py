"""data_root 覆盖验证（三个 pipeline 的原始数据目录参数）。

验证点：
1. reader 级：环境变量 RUST_PYFUNC_LEVEL2_ROOT（及旧变量 RUST_PYFUNC_LEVEL2_PATH）
   覆盖 Level2 数据目录；目录或日期缺失时直接报错，不再回退 /nas197/...。
2. pipeline 级：data_root 参数就是"直接含数据文件的目录"（Level2 传 stock 目录，
   分钟传 1min_factor_text 目录），经 worker 子进程生效——用"只放一天数据的临时目录"证明：
   没放的那天算不出东西（默认目录里其实是有的）。

用法：python tests/test_data_root_override.py
"""

import os
import shutil
import tempfile

import rust_pyfunc as rp

DATE_OK = 20241231      # 临时目录里放这一天
DATE_MISS = 20241230    # 临时目录里不放这一天（默认目录里存在）
CODE = "000001"
DEFAULT_LEVEL2 = "/ssd_data/stock"
DEFAULT_MINUTE = "/ssd_data/data/1min_factor_text"


def check(name: str, cond: bool) -> None:
    print(("✅ " if cond else "❌ ") + name)
    assert cond, name


def completed_dates(store: str) -> set:
    path = f"{store}/_completed_dates"
    return set(int(x) for x in open(path)) if os.path.exists(path) else set()


tmp = tempfile.mkdtemp(prefix="dsh_data_root_")
level2_root = f"{tmp}/level2"
minute_root = f"{tmp}/minute"
os.makedirs(level2_root)
os.makedirs(minute_root)
os.symlink(f"{DEFAULT_LEVEL2}/{DATE_OK}", f"{level2_root}/{DATE_OK}")
for name in os.listdir(DEFAULT_MINUTE):
    os.symlink(f"{DEFAULT_MINUTE}/{name}", f"{minute_root}/{name}")

try:
    # ============ 1. reader 级：环境变量覆盖 Level2 目录 ============
    v_default = rp.py_distill(CODE, DATE_OK)
    os.environ["RUST_PYFUNC_LEVEL2_ROOT"] = level2_root
    check("环境变量覆盖后 py_distill 与默认目录结果一致", rp.py_distill(CODE, DATE_OK) == v_default)
    try:
        rp.py_distill(CODE, DATE_MISS)
        check("临时目录缺该日期应报错", False)
    except Exception:
        check("临时目录缺该日期 → 报错（未回退）", True)
    os.environ["RUST_PYFUNC_LEVEL2_ROOT"] = "/tmp/__no_such_dir__"
    try:
        rp.py_distill(CODE, DATE_OK)
        check("数据目录不存在应报错", False)
    except Exception:
        check("数据目录不存在 → 报错（未回退 /nas197）", True)
    del os.environ["RUST_PYFUNC_LEVEL2_ROOT"]
    os.environ["RUST_PYFUNC_LEVEL2_PATH"] = level2_root      # 旧变量兼容
    check("旧变量 RUST_PYFUNC_LEVEL2_PATH 仍生效", rp.py_distill(CODE, DATE_OK) == v_default)
    del os.environ["RUST_PYFUNC_LEVEL2_PATH"]

    # ============ 2. 横截面 pipeline：data_root = Level2 目录 ============
    names_cs = rp.py_cross_section_example_names()

    def run_cs(store: str, tasks: list, root: str) -> None:
        rp.run_factor_pipeline_cross_section(
            pipeline="cross_section_example",
            tasks=tasks,
            n_jobs=20,
            expected_result_length=len(names_cs),
            trading_days=list(rp.td.trading_days),
            store_dir=store,
            store_factor_names=names_cs,
            data_root=root,
        )

    store_cs = f"{tmp}/store_cs"
    run_cs(store_cs, [DATE_OK], level2_root)
    info_cs = rp.factor_store_v5_info(store_cs)
    check(
        "cross_section：data_root 覆盖生效（临时目录里的日期算出数据并投影）",
        completed_dates(store_cs) == {DATE_OK} and info_cs["is_projected"] and info_cs["record_count"] > 0,
    )
    try:
        run_cs(f"{tmp}/store_cs_miss", [DATE_MISS], level2_root)
        check("cross_section：临时目录缺该日期应报错", False)
    except RuntimeError:
        check("cross_section：临时目录缺该日期 → 报错（未读默认目录）", True)

    # ============ 3. 分钟 pipeline：data_root = 1min_factor_text 目录 ============
    names_m = rp.py_minute_example_names()

    def run_minute(store: str, tasks: list, root: str) -> None:
        rp.run_factor_pipeline_minute(
            pipeline="minute_example",
            tasks=tasks,
            n_jobs=2,
            expected_result_length=len(names_m),
            trading_days=list(rp.td.trading_days),
            store_dir=store,
            store_factor_names=names_m,
            data_root=root,
        )

    store_m = f"{tmp}/store_m"
    run_minute(store_m, [DATE_OK], minute_root)
    check("minute：data_root 覆盖生效", completed_dates(store_m) == {DATE_OK})
    # 负例用"不存在的目录"：若覆盖被忽略，会照默认目录正常出数（分钟数据是整天一个大 h5，
    # 单日缺失无法构造，故用整个目录不存在来证明确实没读默认目录）。
    store_m_bad = f"{tmp}/store_m_bad"
    try:
        run_minute(store_m_bad, [DATE_OK], "/tmp/__no_such_dir__")
    except RuntimeError:
        pass
    check("minute：data_root 指向不存在目录 → 无结果（未读默认目录）", completed_dates(store_m_bad) == set())

    # ============ 4. Level2 per-stock pipeline：data_root = Level2 目录 ============
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
    run_distill(store_ok, level2_root)
    facs_ok = rp.factor_store_v5_read_factor(store_ok, 0)["factor"]
    check("level2：data_root 指向有效目录 → 因子非空", len(facs_ok) > 0)

    store_bad = f"{tmp}/store_l2_bad"
    run_distill(store_bad, "/tmp/__no_such_dir__")
    facs_bad = rp.factor_store_v5_read_factor(store_bad, 0)["factor"]
    check("level2：data_root 指向不存在目录 → 因子全 NaN（读回为空）", len(facs_bad) == 0)

    # ============ 5. 不传 data_root：默认目录仍可用（覆盖不残留） ============
    store_def = f"{tmp}/store_default"
    run_cs(store_def, [DATE_OK], None)
    check("不传 data_root → 默认目录 /ssd_data/stock 正常出数（覆盖已还原）", completed_dates(store_def) == {DATE_OK})

    print("\n🎉 data_root 覆盖验证全部通过")
finally:
    shutil.rmtree(tmp, ignore_errors=True)
