"""hm95 切换时刻贡献 收尾：因子补到 2015-01-09 ~ 磁盘最新交易日，回测 2016-2025，只按中性化 RankIC 粗筛。

【新旧范式对照（本次重写的原因）】
  旧范式（已废弃）：Python go(date) 函数批量 + parquet + tail_pipeline_v4。
    因子名在 Python 侧拼（FIELD_NAMES × _gen_base_793()，15067 个字符串），
    计算走 dw.run_factor_rust_bin_minute 把 go 函数塞进 30 个 Python 进程，
    每个因子落一个 parquet（factor_data/hm95/ 下 15067 个文件），
    回测走 dw.tail_pipeline_v4 —— 一个旧入口：没有 colblk store 的续算/投影机制，
    base 只能吃 parquet 全期、没有独立的 base 轴，也没有 IC-only 快路径。
    这套写法只有 hm95 一个脚本还在用，与其余 32 个 hm 收尾脚本（hm94~hm110）不一致。
  新范式（本文件）：Rust 横截面 pipeline + colblk + tail_pipeline_engine。
    因子计算：rp.run_factor_pipeline_cross_section(pipeline="switch_moment")，
    与其余横截面收尾脚本同一个入口，per-date 多进程并行，产物直接写 colblk store。
    因子名：rp.py_switch_moment_names()，Python 侧不再拼名字（少一处与 Rust 顺序对不齐的风险）。
    存储：colblk store（/hdd/user_home_unsafe/chenzongwei/factor_store_hm95），
    支持断点续算（_completed_dates）+ 增量投影（incremental_projection=True）。
    回测：dw.tail_pipeline_engine，base 走独立轴、IC-only 粗筛、fulltest/doc 全在引擎内。

【1. ver 固定成 "hm95"】研究版 ver = sys.argv[0].split("/")[-1].split(".")[0]（从文件名取），
   改名成 hm95.py 后 ver 会跟着变，产物全部错位，所以这里写死。

【2. 计算区间：从 Level2 可取满五日窗口的首日，到磁盘最新 Level2 日期】
   每个因子使用含当日在内最近五个交易日的 Level2 逐笔和盘口，库内还原分钟字段。
   原始数据从 20150105 开始，故首个可计算日是 20150109。
   日期范围由实际存在的 transaction 与 market_data 文件确定，不读取分钟 H5。
   断点续算按日期粒度；中途断开后重跑会跳过已完成日期。

【3. 回测窗口 2016-01-01 ~ 2025-12-31；base 仍是 2015-01-01 ~ 最新全期】
   因子计算：算到磁盘上最新有数据的交易日（见上）。
   回测（engine 的 IC/组合 + fulltest 的图与指标）：只跑 bt_start ~ bt_end。
   base_part：用 base_start_date / base_end_date 单独指定 2015-01-01 ~ 最新，与回测轴解耦
   （引擎为 base 生成独立轴文件 {tail_root}/base_axis），所以 base 不必拆到流水线外面，
   仍在 postprocess 内部、fulltest/doc 之前落盘。

【4. IC-only 粗筛】selection_kwargs 只留中性化 RankIC 通道（照抄 hm100）：cut_num=35、
   cut2_rate=1.0，cut1_rate/raw_ret_rate/raw_ic_rate/ic_more_important_* 显式 None。
   触发 engine 的 IC-only 快路径：不算十分组/多空组合的收益率指标，也不算原始值 IC，
   上万列因子走批量粗筛，筛出 35 个之后只对这 35 个做 fulltest 与 doc。

【5. 单版 = 纯风格】本脚本 industry_neutralize=False，只对 10 列风格中性化。ver 保持无后缀
   "hm95"（约定：无后缀 = 纯风格，_ind = 行业 + 风格）；本系列只出这一版。

【6. n_jobs 走 FACTOR_N_JOBS】任务系统注入契约（factor_taskd：spawn 时设 FACTOR_N_JOBS），
   手动运行默认 200。这是提交到 factor-task 的脚本的固定接口，不是兼容分支。
   注意与旧范式的区别：旧脚本计算侧是写死的 30 进程 × go 内部 18 线程，与 n_jobs 无关；
   新入口的 n_jobs 是「总线程数」，由入口自己拆成 N 进程 × M 线程（如 200 → 4 × 50）。

用法：
  python hm95.py                # 全流程：计算（增量）+ 回测 + base/fulltest/doc
  python hm95.py --dry-run      # 只读干跑：打印日期范围、因子数、存储路径，不做计算与写入
  python .../factor_task.py submit .../hm95.py --n-jobs 200
"""
import os

# fulltest worker 由 rust 侧把脚本生成到 /tmp 再启动，Python 会把脚本所在目录放进 sys.path，
# /tmp 下任何与依赖同名的 .py 都会把真包顶掉（2026-10-15 被 /tmp/docx.py 顶掉 docx，
# worker 一启动就 ImportError 退出，父进程报成「读取 fulltest worker 结果失败」）。
# PYTHONSAFEPATH=1 让子进程不把脚本目录加进 sys.path；worker 自己会把 /home/chenzongwei 插进去。
os.environ.setdefault("PYTHONSAFEPATH", "1")
import sys

# 必须在 import numpy / design_whatever 之前设：父进程 import numpy 会自带 64 个闲置的
# OpenBLAS 线程，它们不吃 CPU，但按线程数看就会让 n_jobs 像是被突破了（生产 n_jobs=30 时
# 光父进程就 64 个）。这里统一压成单线程，让 n_jobs 真正等于整棵进程树的线程上限。
for _blas_var in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
):
    os.environ.setdefault(_blas_var, "1")

import design_whatever as dw
import numpy as np
import rust_pyfunc as rp


# ============================================================
# 1. 参数配置
# ============================================================

ver = "hm95"                     # 固定字符串，不从文件名取（纯风格版；行业版才加 _ind）
start_date = 20150101            # 因子计算起点的下界（真实起点见 _first_date_with_full_window）
bt_start = "2016-01-01"          # 回测窗口起点 = engine 日期轴起点
bt_end = "2025-12-31"            # 回测窗口终点 = engine 日期轴终点
base_start_date = "2015-01-01"   # base_part 的日期范围起点（与回测窗口解耦）
# 任务系统注入的并行度（factor_taskd 契约），手动运行默认 400
n_jobs = int(os.environ.get("FACTOR_N_JOBS", "400"))
# n_jobs 既是上限也是**目标**：设多少就实际占多少核，偏差不超过一成，且任何时刻不超过。
# 实测（2026-09-21，机器干净，/proc 每线程 utime+stime 增量）：
#   n_jobs=30  → 自动拆 2 进程 × 14 字段 → 实际 28 核（93%），单日 80 秒
#   n_jobs=200 → 10 进程 × 19 字段      → 实际 184 核（92%）
#   n_jobs=400 → 20 进程 × 19 字段      → 实际 365 核（91%），全期约 8.9 小时
# 进程数由 pipeline 自己按 n_jobs 算（每字段 1 线程时线程接近满载，再靠进程数凑够核数），
# 所以这里不要再显式传 n_workers，传了反而会破坏「n_jobs = 实际占用」这个契约。
# 旧实现（go 函数 30 进程）全期 25 小时，现在快约 2.8 倍。
# 单日模式（生产每天更新）有个已知限制：一天只有一个任务，只能落到一个进程里，而单进程最多
# 约 19 核。所以生产建议 n_jobs 直接设 19（参数与实际一致）；设 30 时实际仍是约 19 核。
# colblk 存储目录：本次从 parquet 范式切过来，是新建的空目录（写死不跟 ver 走）
# Set FACTOR_STORE_ROOT to /hdd/user_home_unsafe/chenzongwei for local-HDD tasks.
# Mount detection follows symlinks; NAS and HDD share the same store format.
store_root = os.environ.get("FACTOR_STORE_ROOT", "/nas197/user_home/chenzongwei")
store_dir = os.path.join(store_root, "factor_store_hm95")
# 回测缓存根（= 引擎默认值 _default_tail_v4_root(ver)）。必须显式传：wrapper 内部固定以
# no_pri=True 调 write_tail_v3_base_factors，它在 no_pri 分支里用 temp_root 去
# load_axis_labels —— 不传就会去读 /nas197/.../hm95_tail_v3（v4 引擎从不创建该目录），
# do_base_part=True 会在 base 阶段抛 FileNotFoundError。
# NAS uses a local NVMe working cache; HDD keeps its existing cache location.
# Explicit per-task cache overrides are honored without changing the store itself.
industry_neutralize = False
tail_root = dw.resolve_factor_storage_paths(
    store_dir, f"{ver}_ind" if industry_neutralize else ver,
    temp_root=os.environ.get("TAIL_TEMP_ROOT"),
)["temp_root"]
# 纯风格版：只对 10 列风格中性化。ver 保持无后缀 "hm95"；行业+风格版才用 hm95_ind。

selection_kwargs = {
    # ---- IC-only 触发条件：关闭除中性化 IC 之外的全部通道与指标 ----
    "cut1_rate": None,                 # 关闭「中性化收益」通道
    "raw_ret_rate": None,              # 关闭「原始值收益」通道
    "raw_ic_rate": None,               # 关闭「原始值 IC」通道
    "ic_more_important_gap5": None,    # 关闭「中性化收益 + IC 门槛」优先通道
    "ic_more_important_gap1": None,
    # ---- 唯一生效通道：中性化 IC ----
    "cover_rate": 0.5,
    "ic_point_neu_gap5": 0.01,         # 中性化 RankIC 门槛（gap5）
    "ic_point_neu_gap1": 0.006,        # 中性化 RankIC 门槛（gap1）
    "corr_point_neu": 0.6,             # 通道内 corr 去重阈值（序列 = 中性化 IC）
    "corr_point": 0.6,                 # 通道合并后的 corr 去重阈值（序列同样 = 中性化 IC）
    "cut_num": 60,                     # 候选池 = 60
    "cut2_rate": 1.0,                  # neu_ic 名额 = int(60 * 1.0) = 60
}
# 预检门槛沿用研究版 切换时刻贡献.py 的 2000，不是其它 hm 系列的 500。
# 这个家族的因子在每个交易日都有一个取值被上千只股票同时命中（实测研究版老库里
# amplitude_m1_bot_ac1 的每日众数计数均值 1510，本次重算 1693，同一量级），
# 用 500 会把 82012 个派生因子全部剔除、入选 0 个。研究版当年就是用 2000 跑出结果的。
majority_count_threshold = 2000.0
zero_max_threshold = 0.1
nan_max_threshold = 0.04

DRY_RUN = "--dry-run" in sys.argv[1:]


# ============================================================
# 2. 因子名：由 Rust 侧给出（Python 不再拼名字）
# ============================================================

names = rp.py_switch_moment_names()
_EXPECTED_LEN = 15067
assert len(names) == _EXPECTED_LEN, f"因子数 {len(names)} != {_EXPECTED_LEN}"
print(f"因子数: {len(names)}", flush=True)


# ============================================================
# 3. 目标 store 预检：已存在的 store 列名/列数必须与脚本严格一致（否则追加会错位）
#    首次运行 store 还不存在，由 pipeline 自己创建（update_mode=True 允许空目录起步）
# ============================================================

if os.path.exists(os.path.join(store_dir, "shard_0", "factors.idx")):
    _info = rp.factor_store_v5_info(store_dir)
    print(
        f"store {store_dir}: {_info['factor_count']} 列 / {_info['record_count']} 行 / "
        f"is_projected={_info['is_projected']} / proj_deltas={_info['proj_deltas']}",
        flush=True,
    )
    assert _info["factor_count"] == len(names), (
        f"列数不一致: store {_info['factor_count']} vs 脚本 {len(names)}"
    )
    assert list(_info["factor_names"]) == list(names), "列名/列序与 store 不一致（不能追加）"
    print("✅ 列名与列序和 store 完全一致，可以续算", flush=True)
else:
    print(f"store {store_dir} 不存在，本次为首次计算（由 pipeline 创建）", flush=True)


# ============================================================
# 4. 日期范围：起点 = 第一个能取满 5 日 Level2 窗口的交易日
# ============================================================


def _level2_dates():
    """列出有逐笔和盘口文件的 Level2 日期；五日窗口从 20150105 起算。"""
    root = "/ssd_data/stock"
    result = []
    for name in os.listdir(root):
        if not (name.isdigit() and int(name) >= 20150105):
            continue
        day = os.path.join(root, name)
        transaction = os.path.join(day, "transaction")
        market_data = os.path.join(day, "market_data")
        if not (os.path.isdir(transaction) and os.path.isdir(market_data)):
            continue
        if any(n.endswith("_transaction.csv") for n in os.listdir(transaction)) and any(
            n.endswith("_market_data.csv") for n in os.listdir(market_data)
        ):
            result.append(int(name))
    return sorted(result)


_available_dates = _level2_dates()
if len(_available_dates) < 5:
    raise RuntimeError("Level2 日期不足五天，无法计算 hm95")
start_date = _available_dates[4]
end_date = _available_dates[-1]
# 任务列表 = 交易日历里 [start_date, end_date] 之间的交易日（日历排到未来，必须截断）
tasks = [int(d) for d in rp.td.trading_days if start_date <= d <= end_date]
# base_part 的日期范围终点：磁盘上最新有 Level2 数据的交易日
base_end_date = f"{end_date // 10000}-{(end_date // 100) % 100:02d}-{end_date % 100:02d}"
print(
    f"日期范围: {start_date} ~ {end_date}（{len(tasks)} 个交易日）",
    flush=True,
)

# 预检：已有多少天完成、本次实际要算多少天（增量投影下只算缺的）
_completed = os.path.join(store_dir, "_completed_dates")
_done = {int(x) for x in open(_completed).read().split()} if os.path.exists(_completed) else set()
_todo = [d for d in tasks if d not in _done]
print(
    f"store 已有 {len(_done)} 天完成，本次待算 {len(_todo)} 天"
    + (f"（{_todo[0]} ~ {_todo[-1]}）" if _todo else "（无）"),
    flush=True,
)

if DRY_RUN:
    print(
        f"🔍 干跑：因子数 {len(names)}；计算 {start_date} ~ {end_date}（{len(tasks)} 天，"
        f"待算 {len(_todo)} 天）；回测 {bt_start} ~ {bt_end}；base {base_start_date} ~ {base_end_date}；"
        f"store {store_dir}；tail_root {tail_root}；n_jobs {n_jobs}（未做任何计算与写入）",
        flush=True,
    )
    raise SystemExit(0)


# ============================================================
# 5. 因子计算（横截面 pipeline "switch_moment"，增量投影）
# ============================================================

print(f"Step 1: 计算切换时刻贡献（横截面 pipeline switch_moment，n_jobs={n_jobs}）", flush=True)
rp.run_factor_pipeline_cross_section(
    pipeline="switch_moment",
    tasks=tasks,
    n_jobs=n_jobs,                  # 进程数由 pipeline 按 n_jobs 自己算（见上方说明）
    expected_result_length=_EXPECTED_LEN,
    trading_days=list(rp.td.trading_days),
    bind_cores=False,
    store_dir=store_dir,
    store_factor_names=names,
    update_mode=True,               # 断点续算：跳过 _completed_dates 里已完成的日期
    incremental_projection=True,    # base 不动，只把新日期写成 factors.proj.delta.<end>
)
_info = rp.factor_store_v5_info(store_dir)
print(
    f"✅ 计算完成: {_info['record_count']} 行, is_projected={_info['is_projected']}, "
    f"proj_deltas={_info['proj_deltas']}（0=纯 base，>0=base+增量 delta）",
    flush=True,
)


# ============================================================
# 6. 回测（IC-only 粗筛 → fulltest/doc）+ base 因子
#    回测窗口 2016-01-01 ~ 2025-12-31；base_part 用 base_start_date/base_end_date
#    单独指定 2015 ~ 最新全期 —— 这两个参数让 base 用「独立的轴文件」
#    （{tail_root}/base_axis），与回测轴（{tail_root}/meta）解耦。
# ============================================================

print("Step 2: 回测（IC-only，只算中性化 RankIC）", flush=True)
gap5_selected, gap1_selected, source_names, run_info = dw.tail_pipeline_engine(
    colblk_store_dir=store_dir,
    ver=ver,
    names=names,
    windows=[5, 10, 20],
    fold=True,
    n_jobs=n_jobs,
    temp_root=tail_root,            # 见上方说明：base/fulltest 需要它定位轴标签
    industry_neutralize=industry_neutralize,   # 纯风格版本传 False
    start_date=bt_start,            # 引擎日期轴 = 回测窗口
    backtest_start_date=bt_start,
    end_date=bt_end,
    do_base_part=True,              # base 走独立轴（下面两个参数），仍在 fulltest/doc 之前落盘
    base_start_date=base_start_date,
    base_end_date=base_end_date,
    do_fulltest_part=True,          # 只对筛出的 35 个做完整回测
    do_doc_part=True,               # doc 依赖 fulltest 产物
    selection_kwargs=selection_kwargs,
    majority_count_threshold=majority_count_threshold,
    zero_max_threshold=zero_max_threshold,
    nan_max_threshold=nan_max_threshold,
    tlq_drop_num=15,               # 按 base_resid 分段口径 TLQ 去掉最小的 15 个
    force_restart=False,
    return_run_info=True,
)
print(
    f"✅ 全流程完成: gap5={len(gap5_selected)}, gap1={len(gap1_selected)}, "
    f"source={len(source_names)}",
    flush=True,
)
