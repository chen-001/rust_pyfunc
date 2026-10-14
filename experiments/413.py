"""hm94 挂单猫 收尾：把因子区间补到 2015-01-01 ~ 最新数据，回测只跑 2016-2025，只按中性化 RankIC 粗筛。

与 挂单猫0701py.py（研究版）的区别：

【1. 全时段续算 + 增量投影】
  研究版（挂单猫0701.py）start_date=20170101 / end_date=20260531 都写死；本脚本起点提到
  2015-01-01（store factor_store_挂单猫0701c 的日期字典最小日期是 20170103，2015-2016 完全
  没数据，bt_start=2016-01-01 的回测窗口必须有这段），终点取**磁盘上最后一个真有 transaction
  目录的交易日**（自动，不写死）。
  store 现在已是 base + 2 个 delta 段（proj_deltas=2），追加只把新日期写成
  factors.proj.delta.<end>，base 一个字节都不动（否则 I/O ≈ 2× store 体积），读取端自动拼接。

【2. with_curvature=False —— 对齐旧 store 的 50400 列】
  store factor_store_挂单猫0701c 是 50400 列的旧口径（19 个单列统计后缀）；2026-07-13 commit
  f3c0cc0 给每组特征加了 curvature / quad_coef 两个二阶趋势指标，Rust observable_order 流水线
  默认输出 52416 列。拿 52416 列去写旧 store 会被拒（列名不匹配）；按 50400 硬截断则列会错位。
  → pipeline_params["with_curvature"]=False 让 Rust 侧**不输出**这两个指标，退回旧 19 后缀口径
  （50400 列），列名/列序与 store 完全一致（Rust 侧开关见 factor_pipeline.rs:221 / features.rs:956）。
  本文件的 names 是研究脚本 挂单猫0701py.py 的 get_names() 原样搬运（该 store 没有 rp.py_*_names()
  单一真相源），与 store 50400 列逐元素一致。

【3. 断点续算的口径（B1 per-stock）】
  rp.run_factor_pipeline 的 update_mode 是**逐 (date, code) cell** 过滤（内部 check_completed()
  扫 chunk 的 id 区），**不读** store 里的 _completed_dates 标记文件——那个文件只被
  run_factor_pipeline_cross_section / _minute / _regime 三个入口使用。所以本 store 没有该标记
  文件不影响续算，也不存在"退化为全量重算"的问题（这也是本脚本不照抄 hm100.py 那段
  `open(_completed_dates)` 预检的原因：B1 没有这个文件，照抄反而 FileNotFoundError）。

【4. IC-only 回测：只算中性化 RankIC】
  selection_kwargs 里 cut1_rate / raw_ret_rate / raw_ic_rate / ic_more_important_gap5 /
  ic_more_important_gap1 显式为 None 且 cut2_rate 有值 → 触发 tail_v4 的 IC-only 模式：
  跳过十分组/多空组合、不产出收益率指标，不跑原始值通道，只做预处理 + 中性化 + 中性化 RankIC；
  corr 去重只喂中性化 IC 宽表。名额 = int(cut_num * cut2_rate) = 35，只对这 35 个跑
  fulltest（do_fulltest_part=True）与 doc（do_doc_part=True），base 照常写出（do_base_part=True）。

【5. 两个版本：纯风格 / 行业+风格】
  同一份 colblk store，两个 ver：ver=hm94（industry_neutralize=False，只对 10 列风格中性化）
  与 ver=hm94_ind（industry_neutralize=True，10 风格 + 31 行业哑变量）。本文件与同目录
  hm94_ind.py 由同一份源码生成，逐字相同，只差 ver 与 industry_neutralize 两个常量。

【6. 回测窗口 2016-01-01 ~ 2025-12-31；base 仍是 2015 ~ 最新全期】
  base_part 用 base_start_date / base_end_date 单独指定，让 tail_pipeline_engine 为 base 生成
  独立轴文件（{tail_root}/base_axis/meta），与回测轴（{tail_root}/meta）解耦，所以 base 不必拆到
  流水线外面——它仍在 postprocess 内部、fulltest/doc **之前**落盘。

【7. temp_root 必须显式传】
  base_part 的 write_tail_v3_base_factors 用 temp_root 定位轴标签；不传会去读
  /nas197/.../{ver}_tail_v3（引擎从不创建该目录），base 阶段直接 FileNotFoundError。

用法：
  python hm94.py
  python .../factor_task.py submit .../hm94.py --n-jobs 200
"""
import os

import design_whatever as dw
import rust_pyfunc as rp


# ============================================================
# 2. 参数配置
# ============================================================

ver = "hm94"                  # 纯风格版 = "hm94"，行业版 = "hm94_ind"（两版只差本常量与 industry_neutralize）
industry_neutralize = False   # 纯风格版 False = 只中性化 10 列风格；行业版 True = 10 风格 + 31 行业哑变量
start_date = 20150101         # 因子计算起点（研究版是 20170101，这里补上 2015-2016）
bt_start = "2016-01-01"       # 回测窗口起点（engine 与 fulltest 都用它）
bt_end = "2025-12-31"         # 回测窗口终点（engine 与 fulltest 都用它）
base_start_date = "2015-01-01"  # base_part 的日期范围起点（与回测窗口解耦）
# n_jobs 由任务管理系统注入（factor_taskd 契约：spawn 时设 FACTOR_N_JOBS），
# 手动运行时用默认 200。这是提交到 factor-task 的脚本的固定接口，不是兼容分支。
n_jobs = int(os.environ.get("FACTOR_N_JOBS", "200"))
# 两版共用同一个 colblk store（写死名字，不跟 ver 走：行业版 ver=hm94_ind 但 store 仍是它）
store_dir = "/hdd/user_home_unsafe/chenzongwei/factor_store_hm94"
L2_ROOT = "/ssd_data/stock"   # Level2 数据根（transaction / market_data 都在它下面）
# 回测缓存根（= 引擎的默认值 _default_tail_v4_root(ver)）。
# 必须显式传给 tail_pipeline_engine：wrapper 内部固定以 no_pri=True 调
# write_tail_v3_base_factors，而它在 no_pri 分支里用 temp_root or _default_tail_v3_root(ver)
# 去 load_axis_labels —— 不传 temp_root 就会去读 /nas197/.../hm94_tail_v3（v4 引擎从不创建
# 该目录），do_base_part=True 会在 base 阶段抛 FileNotFoundError。
tail_root = f"/nas197/user_home_unsafe/chenzongwei/{ver}_tail_v4"

# observable_order pipeline 参数（照抄 挂单猫0701py.py，只多一个 with_curvature=False）
# with_curvature=False：不输出 curvature / quad_coef 两个二阶趋势指标，退回旧 19 统计量口径，
#   使输出列序与 store 挂单猫0701c 的 50400 列严格对齐（否则写入会被拒/错位）。
pipeline_params = {
    "q": 0.10,
    "drop_min": 10,
    "forward_sec": 180.0,
    "pre_minutes": 5.0,
    "rolling_n": 20,
    "autocorr_lag": 1,
    "with_curvature": False,
}

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
majority_count_threshold = 500.0
zero_max_threshold = 0.1
nan_max_threshold = 0.04


# ============================================================
# 3. 因子名生成（旧 19 统计量口径，与 store 挂单猫0701c 的 50400 列严格对齐）
#    研究脚本 挂单猫0701py.py 没有 rp.py_*_names()，names 从它原样搬运。
# ============================================================

_NCOLS = 63   # Rust DELETE_COLS(84) 后输出列数（第四轮加 8 死列，observable_order_metrics.rs）
# 19 = 16 个单列统计后缀（不含 curvature / quad_coef）+ lz_complexity / entropy_1d / max_range_product
_FEAT_PER_GROUP = 19 * _NCOLS + _NCOLS * (_NCOLS - 1) // 2
_RUST_DELETED = {  # Rust DELETE_COLS 第四轮删除的 8 死列（forward_sec=180 跨股诊断 ≥10/12 死列）
    "r20_det_ac1", "r20_abs_ac1", "br_ac1",
    "ct_b_skew", "ct_s_skew", "ct_s_ac1", "ct_d_skew", "ct_da_ac1",
}
_EXPECTED_LEN = _FEAT_PER_GROUP * 16  # 16 组 = 4套配置 × (seg+pre5+diff+abs_diff) = 50400


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


def get_names():
    col_names = _build_col_names()
    prefixes = []
    for side in ["bid", "ask"]:
        for method in ["A", "B"]:
            for arr in ["seg", "pre5", "diff", "abs_diff"]:
                prefixes.append(f"{side}_{method}_{arr}")
    suffixes = []
    for s in [
        "mean", "median", "std", "skew", "kurt", "p5", "p25", "p75", "p95",
        "iqr", "cv", "autocorr1", "autocorr1_abs", "trend", "period_diff", "period_ratio",
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


names = get_names()
assert len(names) == _EXPECTED_LEN, f"因子名数 {len(names)} != {_EXPECTED_LEN}"
print(f"因子数: {len(names)}（旧 19 统计量口径）", flush=True)


# ============================================================
# 4. 目标 store 预检：列名/列数必须与旧库严格一致（否则追加会错位）
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
    # 首次构建：store 还不存在，本次由 pipeline 直接建库，没有旧列名可断言。
    _info = {"record_count": 0}
    print(f"store 不存在: {store_dir} → 首次构建（本次直接建库）", flush=True)


# ============================================================
# 5. 任务列表：2015-01-01 ~ 最新有数据的交易日
# ============================================================

def _available_dates(start, end):
    """磁盘上真实存在 transaction 目录的交易日。"""
    return sorted(
        int(d)
        for d in os.listdir(L2_ROOT)
        if d.isdigit() and start <= int(d) <= end
        and os.path.isdir(os.path.join(L2_ROOT, d, "transaction"))
    )


def _scan_pairs(dates):
    """按目录扫 transaction 文件名取股票代码（只含真实存在的 (date, code)）。"""
    out = []
    for d in dates:
        tdir = os.path.join(L2_ROOT, str(d), "transaction")
        for f in os.listdir(tdir):
            code = f.split("_")[0]
            if code.isdigit():
                out.append([d, code])
    return out


end_date = _available_dates(start_date, 20991231)[-1]   # 数据最新交易日（自动，不写死）
# base_part 的日期范围终点：磁盘上最新有数据的交易日（回测窗口只到 bt_end，base 要全期）
base_end_date = f"{end_date // 10000}-{(end_date // 100) % 100:02d}-{end_date % 100:02d}"
print(f"日期范围: {start_date} ~ {end_date}（最后一个有数据的交易日）", flush=True)

# rp.read_level2_list 直接扫 /ssd_data/stock 各日期目录，结果就是磁盘现状，不再有
# 「预生成 parquet 更新滞后」的问题；下面这段补扫保留作保险，正常情况 _gap 为空。
pairs = rp.read_level2_list(start_date, end_date)
_have = {d for d, _ in pairs}
_gap = [d for d in _available_dates(start_date, end_date) if d not in _have]
if _gap:
    _extra = _scan_pairs(_gap)
    print(
        f"有 {len(_gap)} 天（{_gap[0]}~{_gap[-1]}）没扫到任务，按目录补扫 {len(_extra)} 个任务",
        flush=True,
    )
    pairs += _extra
print(f"任务数(date×code): {len(pairs)}（store 已有 {_info['record_count']} 行，"
      f"update_mode 会按 (date, code) 跳过已算的）", flush=True)


# ============================================================
# 6. 因子计算（增量投影）
# ============================================================

print("Step 1: 计算挂单猫", flush=True)
rp.run_factor_pipeline(
    pipeline="observable_order",
    tasks=pairs,
    n_jobs=n_jobs,
    backup_file="",
    expected_result_length=_EXPECTED_LEN,
    trading_days=rp.td.trading_days.tolist(),
    params=pipeline_params,
    update_mode=True,               # 断点续算：按 (date, code) 跳过已写入的
    bind_cores=False,
    mode="multiprocess",
    store_dir=store_dir,
    store_factor_names=names,
    export_n_jobs=80,
    incremental_projection=True,    # base 不动，只把新日期写成 factors.proj.delta.<end>
)
_info = rp.factor_store_v5_info(store_dir)
print(
    f"✅ 计算完成: {_info['record_count']} 行, is_projected={_info['is_projected']}, "
    f"proj_deltas={_info['proj_deltas']}（0=纯 base，>0=base+增量 delta）",
    flush=True,
)


# ============================================================
# 7. 回测（IC-only 粗筛 → fulltest/doc）+ base 因子
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
    industry_neutralize=industry_neutralize,   # 纯风格版 False；行业版 True
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
