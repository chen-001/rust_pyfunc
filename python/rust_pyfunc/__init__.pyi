"""
rust_pyfunc - 高性能Python函数库
=================================

基于Rust实现的高性能Python函数库，提供数学计算、时间序列分析、
文本处理、并行计算等功能。

主要模块：
- core_functions: 核心数学和统计函数
- time_series: 时间序列分析函数
- text_analysis: 文本处理函数
- parallel_computing: 并行计算和备份管理
- pandas_extensions: Pandas扩展函数
- tree_structures: 树结构相关类
"""

from typing import Any, List, Optional, Tuple
import numpy as np

def factor_production_runtime(settings: dict) -> Any: ...
def observable_order_factor_names() -> List[str]: ...
def okay_series_names(series: str) -> List[str]: ...
def okay_compute_stock(series: str, code: str, date: int) -> List[float]: ...
def okay_compute_cross(series: str, date: int) -> Tuple[List[str], List[float]]: ...
def write_selected_factor_base(
    pipeline_groups: list, names: List[str], start_date: int, end_date: int, *,
    base_factor_ver: str, base_hdf5_dir: str, colblk_store_dir: str,
    level2_root: str, calendar_root: str, vars_root: str,
    n_jobs: int = 30,
    minute_root: Optional[str] = None,
) -> None: ...
def write_selected_factor_db(
    names: List[str], start_date: int, end_date: int, *,
    base_factor_ver: str, base_hdf5_dir: str, db_factor_ver: str,
    db_hdf5_dir: str, calendar_root: str, vars_root: str,
) -> None: ...
def write_selected_factor_final(
    names: List[str], start_date: int, end_date: int, *,
    base_factor_ver: str, base_hdf5_dir: str, final_factor_ver: str,
    final_hdf5_dir: str, calendar_root: str, vars_root: str,
) -> None: ...

# 导入所有类型声明
from .core_functions import *
from .time_series import *
from .text_analysis import *
from .parallel_computing import *
from .pandas_extensions import *
from .pandas_correlation import *
from .tree_structures import *
from .trading_analysis import *
from .agent_trading_daily import *
from .agent_simulator import *
from .statistical_analysis import *
from .web_manager import *
from .copula import *
from .theme_cluster_factors import *
from .theme_feature_expansion import *
from .theme_feature_expansion_data import *
from .trading_data_utils import *
from .minute_data_reader import *



def py_individual_order_ratio_names() -> List[str]:
    ...


def py_individual_order_ratio(code: str, date: int) -> List[float]:
    ...
def py_orderbook_imb_refactor_names() -> List[str]:
    ...


def py_compute_orderbook_imb_refactor(code: str, date: int) -> List[float]:
    ...


def py_extreme_point_fit(
    code: str, date: int, with_curvature: bool = True
) -> List[float]:
    """单股单日完整因子值。with_curvature=True（默认）→ 27360 个；False → 26976 个（19 统计量旧口径）。"""
    ...


def py_extreme_point_fit_names(with_curvature: bool = True) -> List[str]:
    """完整因子名。with_curvature=True（默认）→ 27360 个；False → 26976 个（19 统计量旧口径）。"""
    ...


def py_dagpath(code: str, date: int) -> List[float]:
    """逐笔跳跃DAG最短路因子族（O(n) 三边版）单股单日完整因子值，固定 17384 个。"""
    ...


def py_dagpath_names() -> List[str]:
    """dagpath 全部因子名，17384 个，与 py_dagpath 输出逐位对应。"""
    ...


def py_extreme_point_fit_from_data(
    trade_arrays: npt.NDArray[np.float64],
    market_arrays: npt.NDArray[np.float64],
    with_curvature: bool = True,
) -> List[float]:
    """数组版极端点拟合因子（样例/分享用）；与 py_extreme_point_fit 逐位一致。

    trade_arrays: (n, 7) [exchtime_us, price, volume, turnover, flag, bid_order, ask_order]
    market_arrays: (m, 46) [exchtime_us, last_prc, volume, turnover, total_ask_vol,
        total_bid_vol, ask_prc1..10, ask_vol1..10, bid_prc1..10, bid_vol1..10]
    """
    ...


def py_distill(code: str, date: int, prev_date: Optional[int] = None) -> List[float]:
    ...


def py_distill_names() -> List[str]:
    ...


def py_hidden_arrange(code: str, date: int) -> List[float]:
    ...


def py_hidden_arrange_names() -> List[str]:
    ...


def py_geffe_leak(code: str, date: int) -> List[float]:
    """线性反馈移位寄存器的非线性组合（组合泄露）；固定 144612 列，算不出来给 NaN。"""
    ...


def py_geffe_leak_names() -> List[str]:
    ...


def py_lfsr_leak_ext1(date: int) -> Tuple[List[str], List[float]]:
    """线性反馈移位存储器的非线性组合（拓展一）：单日全市场，返回 (codes, vals)。

    每股 15744 列原始位流特征，再逐列做横截面 z-score（前两列计数保持原值）；
    盘口不足 300 行或逐笔不足 500 笔的股票不进当天输出。
    """
    ...


def py_lfsr_leak_ext1_names() -> List[str]:
    ...


def py_lfsr_ext2(date: int) -> Tuple[List[str], List[float]]:
    """线性反馈移位存储器的非线性组合（拓展二）：单日全市场，返回 (codes, vals)。

    每股 8506 列原始特征（2 计数 + 63 个拓展一入选列 + 8441 个新列），逐列做横截面 z-score
    （计数与 5 个 ctl_* 风格变量除外），再对 5 个风格变量 + ln 总市值做回归取残差、残差再
    z-score，得到 8499 个 _r 列，合计 17005 列。算不出来的股票整行 NaN，universe 一只不少。
    """
    ...


def py_lfsr_ext2_names() -> List[str]:
    ...


def py_distill_tick(code: str, date: int) -> List[float]:
    ...


def py_distill_tick_names() -> List[str]:
    ...


def py_hhi_event(code: str, date: int) -> List[float]:
    """A股创新高/新低 HHI 事件因子；140 = 2 方向 × 2 侧 × 7 窗口 × 5 统计量。"""
    ...


def py_hhi_event_names() -> List[str]:
    ...


def py_etp_event(code: str, date: int) -> List[float]:
    """A股创新高/新低熵变因子（hm54 现代版）；60 = 2 方向 × 2 侧 × 3 窗口 × 5 统计量。"""
    ...


def py_etp_event_names() -> List[str]:
    ...

def py_anneal_volume(code: str, date: int) -> List[float]:
    """柜员游戏因子；没有成交的分钟保留为 NaN，不作为零值参与分钟统计。"""
    ...

def py_anneal_volume_gpt(code: str, date: int) -> List[float]:
    ...


def py_anneal_volume_names() -> List[str]:
    ...

def py_hidden_arrange_from_data(
    trade_arrays: npt.NDArray[np.float64],
    market_arrays: npt.NDArray[np.float64],
) -> List[float]:
    """逐笔 + 盘口数组版柜员游戏三层因子；与 py_hidden_arrange 逐位一致，供样例/分享用。"""
    ...

def py_anneal_volume_from_data(
    exchtime_us: npt.NDArray[np.int64],
    price: npt.NDArray[np.float32],
    volume: npt.NDArray[np.float32],
    turnover: npt.NDArray[np.float32],
    flag: npt.NDArray[np.int32],
    bid_order: npt.NDArray[np.int64],
    ask_order: npt.NDArray[np.int64],
) -> List[float]:
    """逐笔数组版柜员游戏因子；与 py_anneal_volume 逐位一致，供样例/分享用。"""
    ...

def py_anneal_volume_market(code: str, date: int) -> List[float]:
    ...

def py_anneal_volume_market_names() -> List[str]:
    ...

def py_anneal_volume_market_from_data(
    time_sec: npt.NDArray[np.float64],
    ask_vols: npt.NDArray[np.float32],
    bid_vols: npt.NDArray[np.float32],
    total_ask_vol: npt.NDArray[np.float64],
    total_bid_vol: npt.NDArray[np.float64],
) -> List[float]:
    ...

def py_read_minute_data(field: str, date: int) -> Tuple[List[str], List[float], int, int]:
    ...


def run_factor_pipeline(
    pipeline: str,
    tasks: List[Tuple[int, str]],
    n_jobs: int,
    backup_file: str,
    expected_result_length: int,
    trading_days: List[int],
    params: Optional[dict] = None,
    update_mode: Optional[bool] = None,
    bind_cores: bool = True,
    backup_batch_size: Optional[int] = None,
    progress_log: Optional[bool] = None,
    mode: Optional[str] = None,
    export_names: Optional[List[str]] = None,
    export_dir: Optional[str] = None,
    export_n_jobs: int = 80,
    store_dir: Optional[str] = None,
    store_factor_names: Optional[List[str]] = None,
    data_root: Optional[str] = None,
    incremental_projection: Optional[bool] = None,
    direct_write: Optional[bool] = None,
    write_buffer_gb: float = 32.0,
) -> None:
    """Level2 per-(date, code) 因子批量计算入口。

    data_root: Level2 数据目录（默认 "/ssd_data/stock"），即直接包含各日期文件夹的目录：
      文件按 {data_root}/{date}/{subdir}/{code}_{date}_*.csv 读取（transaction / market_data）。
      显式指定后只查该目录，不再回退 /nas197/binary/stock/sz_alpha/stock。

    update_mode: True = 断点续算（按 (date, code) 粒度跳过已写入的任务）。
    params: pipeline 参数字典。observable_order / extreme_point_fit 额外支持
      `with_curvature`（默认 True）：
      False = 不输出 curvature / quad_coef 两个二阶趋势指标，退回 19 统计量口径
      （每组 19*n+C(n,2) 而非 21*n+C(n,2)），用于在 curvature 改动之前创建的旧 store 上续算
      （如「为人画像」的 26976 列旧 store；注意 extreme_point_fit 的 with_curvature=false
      对应 26976 列，true 对应 27360 列，必须与 store_factor_names 严格同口径）。
    incremental_projection: True = 增量投影（仅 store_dir 模式）。追加新任务时保留已有
      factors.proj 不动，新行写成 factors.proj.delta.<end>，读取端自动拼接 base + delta；
      默认 False = 追加即删投影、收尾整片重投影（I/O ≈ 2× store 体积）。
    direct_write: True = 在 store_dir 直接写最终列段，不生成 colblk 或执行投影。
      默认 None = 自动打开：给了 store_dir 且没显式要 incremental_projection 就直写；
      显式 False 保持旧流程。须使用新目录或已有同格式目录，不能转换旧库，
      不能与 incremental_projection=True 同用。只在完整列段发布后登记完成；
      崩溃时未封存的结果会在 update_mode=True 续算时重算。
    write_buffer_gb: 直接写入的存储缓冲预算，单位 GiB，默认 32.0。
      仅 direct_write=True 时有效，必须为有限正数；不包含 worker、IPC、
      结果队列和因子计算内存，因此不是整个进程的内存上限。
    """
    ...


def run_factor_pipeline_minute(
    pipeline: str,
    tasks: List[int],
    n_jobs: int,
    expected_result_length: int,
    trading_days: List[int],
    params: Optional[dict] = None,
    update_mode: Optional[bool] = None,
    bind_cores: bool = True,
    store_dir: Optional[str] = None,
    store_factor_names: Optional[List[str]] = None,
    data_root: Optional[str] = None,
    incremental_projection: Optional[bool] = None,
) -> None:
    """分钟数据 per-date 因子批量计算入口。

    data_root: 分钟数据目录（默认 "/ssd_data/data/1min_factor_text"），
      即直接包含 {field}.h5 与 calendar_map.csv / symbol_map.csv 的目录。
    update_mode: True = 断点续算（按 _completed_dates 跳过已完成日期）。
    incremental_projection: True = 增量投影。收尾执行投影：store 已有 base 时只把新日期写成
      factors.proj.delta.<end>（base 不动，读取端自动拼接）；store 尚无 base 时先建立 base
      （一次全量投影）。默认 False = 原行为（不投影，回测走在线转置）。
    """
    ...


def py_minute_example(date: int) -> Tuple[List[str], List[float]]:
    ...


def py_minute_example_names() -> List[str]:
    ...


def py_qhsk_anneal(date: int) -> Tuple[List[str], List[float]]:
    ...


def py_qhsk_anneal_names() -> List[str]:
    ...


def py_minute_capm(date: int) -> Tuple[List[str], List[float]]:
    ...


def py_minute_capm_all(date: int) -> Tuple[List[str], List[float]]:
    ...


def py_minute_capm_at(date: int, minute_index: int) -> Tuple[List[str], List[float]]:
    ...


def py_minute_capm_names() -> List[str]:
    ...


def py_minute_capm_all_names() -> List[str]:
    ...


def py_minute_capm_at_names() -> List[str]:
    ...


def py_microstructure_3s_features(code: str, date: int) -> List[float]:
    ...


def py_microstructure_3s_feature_names() -> List[str]:
    ...


def py_microstructure_capm(date: int) -> Tuple[List[str], List[float]]:
    ...


def py_microstructure_capm_codes(date: int, codes: List[str]) -> Tuple[List[str], List[float]]:
    ...


def py_microstructure_capm_names() -> List[str]:
    ...


def py_switch_moment_names() -> List[str]:
    """switch_moment（hm95 切换时刻贡献）15067 个因子名，顺序 = 字段字典序 × 793 基础名序。"""
    ...


def py_switch_moment(date: int) -> Tuple[List[str], List[float]]:
    """单日直算 switch_moment 因子（核对用）：返回 (codes, 扁平值)，扁平值长度 = len(codes) * 15067。"""
    ...


def py_multi_factor_capm(date: int) -> Tuple[List[str], List[float]]:
    ...


def py_multi_factor_capm_names() -> List[str]:
    ...


def run_factor_pipeline_cross_section(
    pipeline: str,
    tasks: List[int],
    n_jobs: int,
    expected_result_length: int,
    trading_days: List[int],
    params: Optional[dict] = None,
    n_workers: Optional[int] = None,
    update_mode: Optional[bool] = None,
    bind_cores: bool = True,
    store_dir: Optional[str] = None,
    store_factor_names: Optional[List[str]] = None,
    force_clear: Optional[bool] = None,
    data_root: Optional[str] = None,
    vars_root: Optional[str] = None,
    incremental_projection: Optional[bool] = None,
    direct_write: Optional[bool] = None,
    write_buffer_gb: float = 32.0,
) -> None:
    """横截面（一天全市场）因子批量计算入口。

    data_root: Level2 数据目录（默认 "/ssd_data/stock"），即直接包含各日期文件夹的目录：
      按 {data_root}/{date}/{subdir}/ 枚举与读取全市场数据。

    vars_root: 行业 / 日频变量目录（默认 "/ssd_data/data/vars"）。
      `{vars_root}/SzBa/industry.h5` 是行业分类的唯一源头，`{vars_root}/SzBa/calendar_map.csv`
      给出它的行轴日期。优先级：vars_root 参数 > 环境变量 RUST_PYFUNC_VARS_DIR > 默认值；
      路径写错直接报错，不会静默读到别的数据。行业表在每个 worker 进程内只读一次，
      之后所有日期共用内存里的表。少数因子额外读取的基础信息仍由
      RUST_PYFUNC_BASIC_INFO_DIR（默认 /ssd_data/data/basic_info）指定。

    update_mode: True = 断点续算（旧格式读取日期完成凭据；直接列段读取已提交清单）。
      None/False = 全量重跑：旧格式 store 非空且未传 force_clear=True 会直接报错。
      direct_write=True 不支持清空非空目录，force_clear 不适用；请续算或指定新目录。

    incremental_projection: True = 增量投影。追加新日期时保留已有投影（base 不动），
      收尾只把新行写成 factors.proj.delta.<end>；读取端自动拼接 base + delta。
      False/None = 旧行为：追加即删投影，收尾整片重投影（I/O ≈ 2× store 体积）。
      已有多年数据、只补算新日期时开这个开关可省掉整库重投影。
    direct_write: True = 在 store_dir 直接写最终列段，不生成 colblk 或执行投影。
      默认 None = 自动打开：给了 store_dir 且没显式要 incremental_projection 就直写；
      显式 False 保持旧流程。须使用新目录或已有同格式目录，不能转换旧库，
      不能与 incremental_projection=True 同用。只在完整列段发布后登记完成；
      崩溃时未封存的结果会在 update_mode=True 续算时重算。
    write_buffer_gb: 直接写入的存储缓冲预算，单位 GiB，默认 32.0。
      仅 direct_write=True 时有效，必须为有限正数；不包含 worker、IPC、
      结果队列和因子计算内存，因此不是整个进程的内存上限。
    """
    ...


def py_yupei_dist(date: int) -> Tuple[List[str], List[float]]:
    """单日全市场横截面因子（寻找玉佩-成交距离）: 28 张关联-差异矩阵 × 28 指标模块。

    返回 (codes, vals): codes 为当日 universe 股票代码列表,
    vals 为行主序 N×F 的因子值（顺序与 py_yupei_dist_names 一致）。
    线程上限 200（512 核共享机）; 设环境变量 RAYON_NUM_THREADS 可覆盖。
    """

def py_yupei_dist_names() -> List[str]:
    """因子名（与 py_yupei_dist 输出列顺序一致）。"""

def py_yupei_dist_lite(
    date: int, prev_date: Optional[int] = None
) -> Tuple[List[str], List[str], List[float]]:
    """单日全市场横截面因子·精简版：只算去重后幸存的 741 列。

    返回 (codes, names, vals)。names 一并返回——列序与全量版不同, 不要混用
    py_yupei_dist_names 的名单。相同列名取值与全量版逐位一致。
    prev_date 传入上一交易日才能算跨日因子（dyn_*），传 None 时这些列全为 NaN。
    线程上限 200; 设环境变量 RAYON_NUM_THREADS 可覆盖。
    """

def py_yupei_dist_lite_names() -> List[str]:
    """精简版因子名（与 py_yupei_dist_lite 输出列顺序一致, 共 741 个）。"""

def py_yupei_dist_emit_names() -> List[str]:
    """维护用：跑指标模块名收集（合成矩阵，不读数据），输出顺序 = names.rs 应存顺序。
    MATRIX_SPECS 改矩阵集之后用它再生成 names.rs。"""

def yupei_dist_backup_matrices(outdir: str, date: int) -> Tuple[int, int]:
    """读全市场逐笔 → 28 矩阵 + 统计量 → 写入 <outdir>/<date>/（codes/stats/mats）。

    返回 (n_stocks, n_matrices)。universe = symbol_map ∩ 当日 L2 文件
    （无成交/读不到的股票保留为空行，不做成交笔数筛选）。
    """

def yupei_dist_backup_factors(
    outdir: str, date: int, prev_date: Optional[int] = None
) -> Tuple[int, int]:
    """读备份 → 运行 28 个指标模块→ 写 factors.bin/names.txt/meta.json。

    prev_date 给定时从同备份目录加载前一日矩阵（dyn_* 跨日因子有值）。
    返回 (n_stocks, n_factors)。
    """

def yupei_dist_backup_read_matrix(
    outdir: str, date: int, name: str
) -> np.ndarray:
    """读单个矩阵为 (N, N) f32 数组（name 见 yupei_dist_backup_matrix_names）。"""

def yupei_dist_backup_read_stats(
    outdir: str, date: int
) -> Tuple[List[str], np.ndarray]:
    """读每股统计量: (codes, N×14 f64)。

    列序: n_trades, amount, total_vol, imb, ret, vol30, vwap, q95u,
    sum_w_cnt, sum_w_vol, sum_w_logvol, sum_w_flow, sum_w_urg, sum_w_ext。
    """

def yupei_dist_backup_read_factors(
    outdir: str, date: int
) -> Tuple[List[str], List[str], np.ndarray]:
    """读因子备份: (codes, names, N×F f32 数组)。"""

def yupei_dist_backup_verify(outdir: str, date: int) -> List[str]:
    """校验备份完整性: 返回缺失矩阵名清单（空 = 完整）。"""

def yupei_dist_backup_matrix_names() -> List[str]:
    """28 个矩阵名（与备份 mats/ 目录文件一致）。"""

def py_yupei_pair(
    date: int,
    mats: Optional[List[str]] = None,
    sig_temp: float = 0.3,
    sig_thr: float = 2.0,
    null_mc: int = 20000,
    exz_bins: int = 10,
) -> Tuple[List[str], List[str], List[float]]:
    """玉佩2 配对互补网络 —— 单日全市场内存管线（不落盘）。

    一次性完成：L2 逐笔 → 28 张有向交互矩阵（yupei_dist 上游）→ 每张指定矩阵
    独立跑 配对扫描(net) → 硬校验(check) → reduce 因子块 + topk K 档因子块，
    全部内存中进行，不读 codes.txt、不落任何中间文件。

    股票覆盖 = cross-section 标准口径（symbol_map ∩ 当日 L2 文件，≈5100 只）：
    无成交/读不到的股票保留行但整行 NaN，不做成交笔数筛选。

    topk 的 K 档固定 5 / 20 / 100 三档（2026-10 从九档裁掉 1/2/10/50/200/500，
    20241231 实测保留列逐位不变，少 720 列、单日快 29%）。

    参数:
      date      交易日 YYYYMMDD
      mats      要跑的矩阵名列表；None = 默认 6 张（cnt×3τ + rvol×3τ；rvol 已替换旧 vol 族）。
                可选值见 MATRIX_SPECS 28 张（cnt_t*/rvol_t*/logvol_t*/flow_t*/
                urg_t*/ext_*）。
      sig_temp  sigmoid 软化温度（默认 0.3）
      sig_thr   sigmoid 软化阈值（默认 2.0）
      null_mc   空模型蒙特卡洛抽样上限（默认 20000）
      exz_bins  (p,q) 条件网格边数（默认 10）

    返回 (codes, names, vals)：
      codes  当日全部股票代码（长度 N，含无成交股）
      names  因子名（长度 F，多矩阵时带 {mat}_ 前缀）
      vals   行主序 N×F，按 codes×names reshape 成 DataFrame 即可

    线程上限 200；设环境变量 RAYON_NUM_THREADS 可覆盖。
    """

def py_yupei_pair_split(
    date: int,
    tau: float = 1.0,
    weight: str = "cnt",
    combos: Optional[List[str]] = None,
    ks: Optional[List[int]] = None,
    sig_temp: float = 0.3,
    sig_thr: float = 2.0,
    null_mc: int = 20000,
    exz_bins: int = 10,
) -> Tuple[List[str], List[str], List[float]]:
    """玉佩2 双层版（主买/主卖拆分）—— 单日全市场，15 条链 × 653 列 = 9795 列。

    每笔成交按主动方向拆成主买（flag 66）/ 主卖（flag 83），配对组合 5 种 ×
    第三只方向 3 种 = 15 条链，列名前缀 sp{配对}{X}。

    返回 (codes, names, vals)，与 py_yupei_pair 同构。
    本函数会把一整天完整算一遍（20251031 实测单线程 2 小时），只要列名时用
    py_yupei_pair_split_names。
    """

def py_yupei_multi(
    date: int,
    tau: float = 1.0,
    weight: str = "cnt",
    ks: Optional[List[int]] = None,
    svd_k: int = 8,
    chains: Optional[List[str]] = None,
    sig_temp: float = 0.3,
    sig_thr: float = 2.0,
    null_mc: int = 20000,
    exz_bins: int = 10,
) -> Tuple[List[str], List[str], List[float]]:
    """多载体单日全市场计算；chains=None 运行全部方向链。

    ks 默认 [5, 20, 100]。返回股票代码、列名及按股票逐行展开的数值。
    线程数在新进程启动前通过 RAYON_NUM_THREADS 设置。
    """
    ...

def py_yupei_multi_v5(
    date: int,
    tau: float = 1.0,
    weight: str = "cnt",
    ks: Optional[List[int]] = None,
    svd_k: int = 8,
    sig_temp: float = 0.3,
    sig_thr: float = 2.0,
    null_mc: int = 20000,
    exz_bins: int = 10,
) -> Tuple[List[str], List[str], List[float]]:
    """四套载体、固定五条方向链；默认输出 12914 列。

    ks 默认 [5, 20]；可显式传 [5, 30] 或 [5, 20, 100] 等其他档位。
    ks 控制附加TopK档位，不改变其他因子内部固定的Top20/Top50等参数。
    等值加速路径默认启用；线程数在新进程启动前通过 RAYON_NUM_THREADS 设置。
    weight 为 "cnt"（笔数）或 "vol"（成交量平方根）；tau 为衰减秒数；
    svd_k 为分解成分数，sig_temp/sig_thr 为软阈值温度/阈值，null_mc 为
    空模型抽样规模（按100层分配，每层至少20次尝试），exz_bins 为条件网格边数。
    sig_temp/sig_thr 每次调用独立生效；并发调用之间互不覆盖。
    oc_ 为 Otsu 聚簇载体，os_ 为订单方向载体。TopK 各档共用同一排序规则。
    返回格式同 py_yupei_multi。
    """
    ...

def py_yupei_multi_v5_names(
    ks: Optional[List[int]] = None,
    svd_k: int = 8,
) -> List[str]:
    """v5 定稿版的列名（12914 个），不读 Level2、不算因子，毫秒级。

    与 py_yupei_multi_v5 的输出列逐字一致；流程脚本用它拿名字。
    ks 默认 [5, 20]，svd_k 默认 8，须与计算时传的一致。
    """
    ...

def py_yupei_union_all(date: int) -> Tuple[List[str], List[str], List[float]]:
    """玉佩四合一：yupei_dist_lite + yupei_gpt + yupei_pair_split + yupei_multi_v5 的因子并集。

    返回 (codes, names, vals)，vals 为 len(codes) × 20716 的行主序扁平 f32。
    列序：lite 741 | gpt 491 | split 9795 | v5 去掉与 split 同名的 3225 列后剩下的 9689 列。
    参数固定为四个函数各自 pipeline 的默认值；股票取四段并集，缺失段填 NaN。
    """
    ...

def py_yupei_union_all_names() -> List[str]:
    """四合一的 20716 个列名，不读数据，与 py_yupei_union_all 输出逐字一致。"""
    ...

def py_yupei_union_trim(date: int) -> Tuple[List[str], List[str], List[float]]:
    """玉佩四合一精简版。返回 (codes, names, vals)，vals 为 len(codes) × 18758 的扁平 f32。"""
    ...

def py_yupei_union_trim_names() -> List[str]:
    """精简版固定的 18758 个列名，不读数据。"""
    ...

def py_yupei_gpt(date: int) -> tuple[list[str], list[float]]:
    """玉佩精炼版单日计算：返回 (codes, 491列×股票数 扁平f32)。"""
    ...

def py_yupei_gpt_names() -> list[str]:
    """玉佩精炼版单日 491 列因子名。"""
    ...

def py_yupei_gpt_temporal(date: int) -> tuple[list[str], list[float]]:
    """玉佩精炼版历史阶段：29 个时序列（读 .ygsnap.json 快照）。"""
    ...

def py_yupei_gpt_temporal_names() -> list[str]:
    """玉佩精炼版 29 个历史时序列因子名。"""
    ...

def py_yupei_pair_split_names(
    combos: Optional[List[str]] = None,
    ks: Optional[List[int]] = None,
) -> List[str]:
    """15 条链的因子列名，不读 Level2、不算因子（毫秒级）。

    与 py_yupei_pair_split(...)[1] 逐字相同：列名只由代码里的字面量决定，
    与行情数据无关，所以用一份 n=64 的合成矩阵跑一遍名字收集即可。
    """

def py_pairwise_merge_metrics(
    times_a: List[int], big_a: List[int], times_b: List[int], big_b: List[int]
) -> Tuple[float, float, float, float]:
    """两股成交流双指针合并的 4 个配对指标 (X1 交替频率, X3 转移互信息, X8 大单聚集, X9 间隔CV)。

    times 升序（微秒）; big 非零 = 大单。
    """

def py_cross_section_example(date: int) -> Tuple[List[str], List[float]]:
    ...

def py_orderflow_ctx(date: int) -> Tuple[List[str], List[float]]:
    """orderflow_ctx 横截面因子：三时段×三类流×上下文算子，每股 10030 值。

    参数
    ----
    date : int
        交易日, 形如 20241231

    返回
    ----
    (codes, vals) : codes 为股票代码列表, vals 为每股 10030 个 f32 值
        （顺序与 py_orderflow_ctx_names 一致；缺的时段/流为 NaN）。
    """
    ...

def py_orderflow_ctx_names() -> List[str]:
    """返回 orderflow_ctx 的 10030 个因子名（{scope}_{stream}_{op}）。"""
    ...

def py_orderflow_ctx_v2(date: int) -> Tuple[List[str], List[float]]:
    """orderflow_ctx 修正版横截面因子，每股 9900 值（3 时段 × 3300）。

    相对 orderflow_ctx：tick_big 改用变价幅度自身的中位数；opw 改为末笔价严格高于首笔价；
    删除恒为常数的 odec_dom 流；xt_imb_mi / xt_imb_lead 改为按时间对齐成交与快照；
    删除与 f_n_tr 重复的 n_trades。

    参数
    ----
    date : int
        交易日, 形如 20241231

    返回
    ----
    (codes, vals) : codes 为股票代码列表, vals 为每股 9900 个 f32 值
        （顺序与 py_orderflow_ctx_v2_names 一致；缺的时段/流为 NaN）。
    """
    ...

def py_orderflow_ctx_v2_names() -> List[str]:
    """返回 orderflow_ctx_v2 的 9900 个因子名（{scope}_{stream}_{op}）。"""
    ...


def py_peaks(date: int) -> Tuple[List[str], List[float]]:
    """高峰-小峰截面因子（方案1）：读一天全市场逐笔 -> 983 因子 + 14 状态标量.

    参数
    ----
    date : int
        交易日, 形如 20241231

    返回
    ----
    (codes, vals) : codes 为股票代码列表, vals 为每股 997 个值
        （983 因子 + 14 状态标量，顺序与 py_peaks_names 一致）。
        缺失值已按「代理变量截面秩 -> 分位映射」补全。
    """
    ...


def py_peaks_from_data(
    codes: List[str],
    trade_arrays: List["numpy.ndarray"],
) -> Tuple[List[str], List[float]]:
    """v2 入口：从 Python 传入的 per-stock trade numpy 数组计算高峰-小峰截面因子。

    参数
    ----
    codes : List[str]
        股票代码列表
    trade_arrays : List[numpy.ndarray]
        每只股票一个 (n_i, 7) 数组，列序同 read_trade_fast：
        [time_sec, price, volume, turnover, flag, bid_order, ask_order]

    返回
    ----
    (codes, vals) 与 py_peaks 走同一计算核心；横截面基于传入子集，
    数值与 v1 全市场口径不同是预期的。
    """
    ...


def py_peaks_names() -> List[str]:
    """997 个名字：983 因子 + 14 状态标量（__st_ 前缀，回测时排除）。"""
    ...


def py_peaks_regime_fit(
    state_store_dir: str,
    date: int,
    window: int,
    trading_days: List[int],
) -> dict:
    """方案2 调试：用 state_store 窗口 [t−W, t−1] 拟合当日状态参数。"""


def py_state_read_row(state_store_dir: str, date: int) -> Optional[List[float]]:
    """方案2 调试：读某日标量行（14 个 f64）。"""


def run_factor_pipeline_regime(
    pipeline: str,
    tasks: List[int],
    n_jobs: int,
    expected_result_length: int,
    trading_days: List[int],
    state_store_dir: str,
    window: int = 59,
    store_dir: Optional[str] = None,
    store_factor_names: Optional[List[str]] = None,
) -> None:
    """方案2 顺序流水线：状态进计算过程（按日期顺序推进，预热期只存标量不输出因子）。

    参数
    ----
    pipeline : str
        流水线标识（"peaks_regime"）
    tasks : List[int]
        交易日列表（升序，顺序执行）
    n_jobs : int
        rayon 日内并行度 + 投影线程数
    expected_result_length : int
        每股因子数（= 997）
    trading_days : List[int]
        交易日历（窗口取最近 W 个交易日）
    state_store_dir : str
        标量表存储目录（每日 {date}.bin，14×f64，为 t+1 备料）
    window : int
        滚动窗口 W（默认 59，即 [t−59, t−1]）
    store_dir : Optional[str]
        colblk 列式存储目录
    store_factor_names : Optional[List[str]]
        因子名列表
    """


def py_pair_interaction(date: int) -> Tuple[List[str], List[float]]:
    """跨股票互动因子: 读一天全市场逐笔 -> 两两互动矩阵 -> 每股 6694 个因子.

    参数
    ----
    date : int
        交易日, 形如 20241231

    返回
    ----
    (codes, vals) : codes 为股票代码列表, vals 为每股 N_FACTORS 个因子值
    """
    ...


def py_pair_interaction_names() -> List[str]:
    """跨股票互动因子的因子名列表 (与 py_pair_interaction 列序一致)."""
    ...


def py_cross_section_example_from_data(
    codes: List[str],
    trade_arrays: List["numpy.ndarray"],
) -> Tuple[List[str], List[float]]:
    """v2 入口：从 Python 传入的 per-stock trade numpy 数组计算横截面因子。

    参数
    ----
    codes : List[str]
        股票代码列表
    trade_arrays : List[numpy.ndarray]
        每只股票一个 (n_i, 7) 数组，列序同 read_trade_fast：
        [time_sec, price, volume, turnover, flag, bid_order, ask_order]

    返回
    ----
    (codes, vals) 与 py_cross_section_example 完全一致。
    """
    ...


def py_relaynet_six_people(date: int) -> Tuple[List[str], List[float]]:
    """全量精确接力网络：单日全市场，477 张图，每股 13,574 个因子。

    返回股票代码和按股票逐行展开的因子值；列序见 py_relaynet_six_people_names。
    最短路和中介中心性遍历全部源节点，不使用源节点抽样。
    包含缓存、top-k、PageRank、生成森林和邻居距离修正，结果与旧 py_relaynet 有区别。
    独立下午分图的10个重名列改为 rlg_pm30_split_rd_thr20_*，列数和位置不变。
    在 import rust_pyfunc 前设置 RAYON_NUM_THREADS=50 可限制计算线程数。
    批量入口为 run_factor_pipeline_cross_section(pipeline="relaynet_six_people", ...)。
    """
    ...


def py_relaynet_six_people_names() -> List[str]:
    """13,574 个因子名，与 py_relaynet_six_people 的输出列一一对应。"""
    ...


def py_cross_section_example_names() -> List[str]:
    ...

def py_dijkstra(date: int) -> Tuple[List[str], List[float]]:
    """最优路径（Dijkstra）横截面因子：单日全市场，每股 7422 个值。

    27 种分桶口径 × 2 套时间窗口，把 Level2 逐笔按口径分桶当节点、
    相邻窗口连边跑最短路，从图与路径取连续量，最后横截面 z-score。
    """
    ...

def py_dijkstra_names() -> List[str]:
    """最优路径因子的名字（7422 个，顺序与 py_dijkstra 的输出列一一对应）。"""
    ...

def py_relaynet(date: int) -> Tuple[List[str], List[float]]:
    """接力网络（relaynet）横截面因子：单日全市场，每股 ~1200 个值。

    同一天全市场股票当节点，用逐笔对齐到 30 秒格子的价格变动构造多种
    共动相似度（收益相关/绝对收益/符号一致/上行尾部/主动买压/领先滞后），
    稀疏化成 ~50 张图后跑 Dijkstra/中心性/社区，逐股票输出因子。
    """
    ...

def py_relaynet_names() -> List[str]:
    """接力网络因子的名字（顺序与 py_relaynet 的输出列一一对应）。"""
    ...


def py_yhyb(date: int, approx: bool = False) -> Tuple[List[str], List[float]]:
    """一呼百应（yhyb）：tick 级事件响应网络横截面因子（v1 读盘，默认参数）。

    23 事件 × 4 时段 × 15 = 1380 个因子/股。事件/度量定义见 README.md。
    approx=True 时用近似统计（1 秒桶中点 + wmean + null 均值池化/x̄ 近似），
    仅用于评估妥协方案的截面影响，不用于生产。
    """
    ...


def py_yhyb_params(date: int, params: List[float]) -> Tuple[List[str], List[float]]:
    """同 py_yhyb，但使用自定义 17 个阈值参数（顺序见 src/yhyb_metrics.rs YhybParams）。

    调参流程：tune_thresholds.py 循环调用本函数扫描覆盖率。
    """
    ...


def py_yhyb_from_data(
    codes: List[str],
    trade_arrays: List["numpy.ndarray"],
    market_arrays: List["numpy.ndarray"],
    params: Optional[List[float]] = None,
    approx: bool = False,
) -> Tuple[List[str], List[float]]:
    """v2 入口：Python 传样例数据计算 yhyb 因子（无磁盘环境验证/调参）。

    参数
    ----
    codes : List[str]
        股票代码列表
    trade_arrays : List[numpy.ndarray]
        每股一个 (n_i, 8) 数组：列序同 read_trade_fast_us：
        [time_us, time_sec, price, volume, turnover, flag, bid_order, ask_order]
    market_arrays : List[numpy.ndarray]
        每股一个 (m_i, 25) 数组：
        [time_us, total_ask_vol, total_bid_vol, ask_prc1, bid_prc1,
         ask_vol1..10, bid_vol1..10]
    params : Optional[List[float]]
        可选 17 个阈值参数（默认 YhybParams::default()）
    approx : bool
        True 时用近似统计（1 秒桶中点 + wmean + null 均值池化/x̄ 近似），
        仅用于评估妥协方案的截面影响，不用于生产

    返回
    ----
    (codes, vals) 与 py_yhyb 完全一致（同一份纯计算核心）。
    """
    ...


def py_yhyb_names() -> List[str]:
    """合并输出全部因子名：1380 时段聚合 + 140 第 4 层网络因子 = 1520。"""
    ...


def py_yhyb_events(code: str, date: int) -> List[Tuple[str, Tuple[List[int], List[float]]]]:
    """单股单日事件时间线（调试/单例用）。

    返回 [(事件名, (时间us数组, 权重数组)), ...] 共 23 个事件类型。
    """
    ...


def py_yhyb_indtop(date: int, mode: int = 0) -> Tuple[List[str], List[float]]:
    """一呼百应补充因子（COR-39 行业头部配对池拓展，单日全市场）。

    mode 0（生产）：4176 因子 = G1a yhyb_ext_indtop10_*（1740，同伴池=同申万一级
    行业市值前 10）+ G1b yhyb_ext_mkttop310_*（1740，同伴池=全市场每行业市值前 10
    并集 ≈310）+ G3 yhyb_ext_mkttop310_spec_*（696 = G1b − 初版全市场，hit/rmed/rhit
    × fwd/bwd）。行业从 /ssd_data/data/vars/SzBa/industry.h5 直读，市值从
    daily_data/total_caps.parquet 直读。G3 基线在 pipeline 内重算初版无 L4 聚合，
    总耗时 ~77s/天（不慢于初版 82s）。
    mode 1（验证）：池=全市场，输出 1740，与初版 py_yhyb_params 逐位一致。
    """
    ...


def py_yhyb_indtop_names() -> List[str]:
    """补充因子名：4176 个（G1a 1740 + G1b 1740 + G3 696）。"""
    ...


def py_vsld(date: int) -> Tuple[List[str], List[float]]:
    """计算成交量分段领衔股相关横截面因子（v1 读盘）。

    算法：全市场逐笔混合排序→成交量累计100段→core序列→每段20领衔股
    →100×20相关矩阵→4套降维。笛卡尔全集 EP1×EP2×EP3×EP4=150 case，
    每 case 148 因子，共 22200 因子（= py_vsld_names 长度）。

    参数
    ----
    date : int
        8 位日期，如 20220819

    返回
    ----
    codes : List[str]
        股票代码列表
    vals : List[float]
        扁平因子值，按 [stock0:22200, stock1:22200, ...] 排布
    """
    ...

def py_vsld_names() -> List[str]:
    """返回 22200 个因子名（EP1_EP2_EP3_EP4_ 前缀 + 降维统计后缀）。"""
    ...

def py_vsld_from_data(
    codes: List[str],
    trade_arrays: List["numpy.ndarray"],
) -> Tuple[List[str], List[float]]:
    """v2 入口：Python 传 codes + per-stock 逐笔数组算因子（样例验证用）。

    参数
    ----
    codes : List[str]
        股票代码列表（长度 n_stocks）
    trade_arrays : List[numpy.ndarray]
        每股一个 (n_i, 7) 数组，列序与 read_trade_fast 一致：
        [time_sec, price, volume, turnover, flag, bid_order, ask_order]
    """
    ...

def py_hot_stock_pool(date: int) -> Tuple[List[str], List[float]]:
    """计算同热点股票池因子。

    参数
    ----
    date : int
        8 位日期，如 20220819

    返回
    ----
    codes : List[str]
        股票代码列表
    vals : List[float]
        row-major 因子值 (len(codes) × 13312)
    """
    ...

def py_hot_stock_pool_names() -> List[str]:
    ...

def py_hot_stock_pool_z_stats(date: int) -> Tuple[List[str], List[int]]:
    """返回每只股票在 8 个 (pi, hot/cold) 下的真实入选次数 z/z'。

    参数
    ----
    date : int
        8 位日期

    返回
    ----
    codes : List[str]
        股票代码列表
    z_matrix : List[int]
        row-major (n_codes × 8)，列顺序:
        [x60_y3_buy_hot, x60_y3_buy_cold, x60_y3_ba_hot, x60_y3_ba_cold,
         x15_y10_buy_hot, x15_y10_buy_cold, x15_y10_ba_hot, x15_y10_ba_cold]
    """
    ...

def py_hot_pool_ind(date: int) -> Tuple[List[str], List[float]]:
    """计算同热点股票池「行业维度拓展」补充因子（8656 个）。

    组 1 hotpool_ext_ind_rel_* (840)：全市场识别 + 行业内相对量回填 + 降维；
    组 2 hotpool_ext_ind_cooc_* (720)：共现 top-10 同行/跨行拆分；
    组 3 hotpool_ext_ind_heat_* (24)：行业热度聚合；
    组 4 hotpool_ext_indpool_* (7072)：行业内识别版（组=当秒同行入选者）。
    """
    ...

def py_hot_pool_ind_names() -> List[str]:
    ...

def py_hot_stock_pool_v2(date: int) -> Tuple[List[str], List[float]]:
    """方案一 v2：判别指标从绝对值改为标准化偏离度 D'=(D-μ)/σ。

    与 py_hot_stock_pool 的区别：分组时用标准化偏离度做百分位排序，
    而非原始 D 值。消除"冷门股永远冰点"问题。

    参数
    ----
    date : int
        8 位日期

    返回
    ----
    codes : List[str]
        股票代码列表
    vals : List[float]
        row-major 因子值 (len(codes) × 13312)
    """
    ...

def py_compute_drop_event_features(date: int) -> dict:
    """计算可观测挂单骤降事件的截面交互特征。

    参数
    ----
    date : int
        8 位日期，如 20220819

    返回
    ----
    dict:
        codes : List[str]   每事件的 "code_side"（如 "000001_bid"）
        times : List[float] 每事件的 epoch 秒
        features : List[float] row-major (n_events × 60)
        names : List[str]   60 个特征名
        n_features : int
    """
    ...


def py_drop_event_feature_names() -> List[str]:
    ...

def py_drop_event(date: int) -> Tuple[List[str], List[float]]:
    ...


def py_drop_event_names() -> List[str]:
    ...


def py_long_order(date: int) -> Tuple[List[str], List[float]]:
    ...


def py_long_order_names() -> List[str]:
    ...


def py_long_order_raw(
    date: int,
) -> Tuple[List[str], List[float], List[float]]:
    ...


def py_urgency(date: int) -> Tuple[List[str], List[float]]:
    ...


def py_urgency_ext(date: int) -> Tuple[List[str], List[float]]:
    ...


def py_urgency_ext_names() -> List[str]:
    ...


def py_urgency_ext_from_data(
    codes: List[str],
    trade_arrays: List["numpy.ndarray"],
) -> Tuple[List[str], List[float]]:
    """全市场数组版迫切交易笔数（样例/分享用）；与 py_urgency_ext 逐位一致。"""
    ...


def py_urgency_names() -> List[str]:
    ...


def py_urgency_from_data(
    codes: List[str],
    trade_arrays: List["numpy.ndarray"],
) -> Tuple[List[str], List[float]]:
    """全市场数组版迫切交易联动因子（样例/分享用）；与 py_urgency 逐位一致。

    trade_arrays: 每只股票一个 (n_i, 7) 数组，
    列序 [exchtime_us, price, volume, turnover, flag, bid_order, ask_order]。
    """
    ...


def personalized_meeting_feature_names() -> Tuple[List[str], List[str]]:
    ...


def personalized_meeting_features(
    date: int,
    symbol: str,
    universe_size: int = 100,
    selection_method: str = "code",
) -> Tuple[List[float], List[str], List[List[float]], List[str]]:
    ...


def orderbook_volume_cov_factors(
    bid_volumes: object,  # numpy.ndarray (n, 10)
    ask_volumes: object,  # numpy.ndarray (n, 10)
    window_size: int,
) -> Tuple[List[str], object]:
    ...


def illusion_liquidity_distance_factors(
    price: object,  # numpy.ndarray (n,)
    volume: object,  # numpy.ndarray (n,)
    turnover: object,  # numpy.ndarray (n,)
    timestamps_ns: object,  # numpy.ndarray (n,) of int64
    window: int,
) -> Tuple[List[str], object]:
    ...


def compute_corr_diff_features(
    large_corr: object,  # numpy.ndarray (n*ratio, n*ratio)
    small_corr: object,  # numpy.ndarray (n, n)
    ratio: int,
) -> Tuple[List[float], object]:
    ...

def compute_agent_validation_factors_py(
    market_timestamps: object,
    market_prices: object,
    market_volumes: object,
    market_flags: object,
    ob_timestamps: object,
    ob_bid_prc1: object,
    ob_ask_prc1: object,
    ob_bid_vol1: object,
    ob_ask_vol1: object,
    per_agent_market_indices: List[object],
    per_agent_directions: List[object],
    per_agent_volumes: List[object],
    fwd_horizons_sec: List[float],
    pt_horizons_sec: List[float],
) -> List[float]:
    ...


def tail_brc_halves_f32(signal: List[float], future: List[float]) -> Tuple[float, float]:
    """Balanced Rank Concordance 的单日「半段均值」（调试/独立验证用，不经过回测引擎）。

    返回 (S_t, L_t)：按因子升序把股票切成低/高两半，D_k 是第 k 个切点的标准化秩和偏离，
    S_t = mean_{k=1..floor(n/2)} D_k（从低端切），L_t = mean_{k=1..floor(n/2)} D_{n-k}（从高端切）。
    值域都是 [-1, 1]，D_k > 0 表示因子低值对应低收益。n < 2 或输入含 NaN/Inf 时返回 (nan, nan)。
    """
    ...


def tail_backtest_engine(
    colblk_store_dir: str,
    factor_names: List[str],
    factor_paths: List[str],
    dates: List[int],
    stocks: List[str],
    windows: List[int],
    fold: bool,
    n_jobs: int,
    min_valid: int,
    cache_root: str,
    style_vars_dir: str,
    ret_gap1_path: str,
    ret_sum_gap1_path: str,
    ret_gap5_path: str,
    ret_sum_gap5_path: str,
    restrict_path: str,
    index_ret_path: str,
    backtest_start: int,
    cover_rate: float = 0.97,
    ret_point_neu_gap5: float = 0.055,
    ret_point_neu_gap1: float = 0.08,
    ic_point_neu_gap5: float = 0.01,
    ic_point_neu_gap1: float = 0.006,
    ret_point_gap5: float = 0.1,
    ret_point_gap1: float = 0.13,
    ic_point_gap5: float = 0.03,
    ic_point_gap1: float = 0.02,
    ic_more_important_gap5: Optional[float] = 0.01,
    ic_more_important_gap1: Optional[float] = 0.006,
    majority_count_threshold: float = 200.0,
    zero_max_threshold: float = 0.01,
    nan_max_threshold: float = 0.04,
    save_all_metrics: bool = False,
    industry_neutralize: bool = True,
    industry_matrix: Optional[object] = None,
    ic_only: bool = False,
    io_concurrency: Optional[int] = None,
    io_batch_size: Optional[int] = None,
    io_batch_bytes: int = 67108864,
    low_coverage_markers: Optional[List[str]] = None,
    low_coverage_names: Optional[List[str]] = None,
) -> dict:
    """批量回测引擎。

    ic_only=True 时跳过收益回测（十分组/多空组合），只计算中性化 IC 序列与
    IC 汇总（summary 中收益字段为 0.0），用于“只按中性化 IC 选因子”的快速筛选。

    low_coverage_markers / low_coverage_names：低覆盖放行（默认 None = 关闭，行为不变）。
    因子名包含任一标记词（如 ["cxl"]）或在名单里的因子，缺失值不填中位数、只在有值股票上
    计算，不因原始覆盖率或缺失占比被淘汰（众数、零值照常判）。用于撤单类因子：
    2021-11-05 前沪市没有撤单，只有深市有值。各闸门读数写到
    {cache_root}/low_coverage_audit/{因子名}.json。只支持 v8 路径。
    """
    ...


# 版本信息
__version__ = "0.18.0"
__author__ = "chenzongwei"
__email__ = "noreply@example.com"

# 所有公开的函数和类
__all__ = [
    "py_yupei_union_trim",
    "py_yupei_union_trim_names",
    # 核心函数
    "trend",
    "trend_fast",
    "compute_integer_small_peak_features",
    "identify_segments",
    "find_max_range_product",
    "ols",
    "ols_predict",
    "ols_residuals",
    "max_range_loop",
    "min_range_loop",
    "rolling_volatility",
    "rolling_cv",
    "rolling_qcv",
    "compute_max_eigenvalue",
    "sum_as_string",
    "test_simple_function",
    "test_function",
    # 时间序列函数
    "dtw_distance",
    "fast_dtw_distance",
    "super_dtw_distance",
    "transfer_entropy",
    "rolling_dtw_distance",
    "find_local_peaks_within_window",
    "find_half_energy_time",
    "rolling_window_stat",
    "find_half_extreme_time",
    "fast_find_half_extreme_time",
    "super_find_half_extreme_time",
    "brachistochrone_curve",
    "brachistochrone_curve_v2",
    "rolling_lagged_regression",
    "rolling_lagged_regression_ridge_fast",
    "rolling_lagged_regression_ridge_simd",
    "rolling_lagged_regression_ridge_incremental",
    # 文本分析函数
    "vectorize_sentences",
    "jaccard_similarity",
    "min_word_edit_distance",
    "vectorize_sentences_list",
    # 并行计算函数
    "run_pools_queue",
    "query_backup",
    "query_backup_fast",
    "build_backup_column_block_cache_single_thread",
    "query_backup_single_column_compact_cached",
    "query_backup_single_column_compact_cached_from_cache_dir",
    "query_backup_codebook_cached",
    "query_backup_codebook_cached_from_cache_dir",
    # 列式因子存储 RPFBINV5（计算备份 + 回测读取统一格式）
    "factor_store_v5_open",
    "factor_store_v5_info",
    "factor_store_storage_profile",
    "factor_store_v5_register_group",
    "factor_store_v5_read_factor",
    "factor_store_v5_template",
    "factor_store_v5_export_factors_parquet",
    "factor_store_v5_verify_scatter_fast",
    "tail_backtest_engine",
    "tail_brc_halves_f32",
    # Pandas扩展函数
    "dataframe_corrwith",
    "dataframe_corrwith_single_thread",
    "rank_axis1",
    "fast_merge",
    "fast_merge_mixed",
    "fast_inner_join_dataframes",
    # 树结构类
    "PriceTree",
    "RollingFutureAccessor",
    "PriceTreeViz",
    # 交易分析函数
    "find_follow_volume_sum_same_price",
    "find_follow_volume_sum_same_price_and_flag",
    "mark_follow_groups",
    "mark_follow_groups_with_flag",
    "analyze_retreat_advance",
    "analyze_retreat_advance_v2",
    "calculate_large_order_nearby_small_order_time_gap",
    "order_contamination",
    "order_contamination_parallel",
    "order_contamination_bilateral",
    "trade_peak_analysis",
    "order_neighborhood_analysis",
    "calculate_trade_time_gap_and_price_percentile_ultra_sorted",
    "calculate_order_time_gap_and_price_percentile_ultra_sorted",
    "analyze_asks",
    "compute_non_breakthrough_stats",
    "compute_price_cycle_features",
    "compute_price_cycle_features_b_segments",
    "compute_price_cycle_features_b_segments_enhanced",
    "analyze_long_orders",
    "analyze_long_orders_python",
    "calculate_passive_order_features",
    "fit_hawkes_process",
    "hawkes_event_indicators",
    "analyze_hawkes_indicators",
    "compute_allo_microstructure_features",
    "reconstruct_limit_order_lifecycle",
    "reconstruct_limit_order_lifecycle_v2",
    "aggregate_agent_feature_triplets_to_daily_table",
    "get_agent_daily_feature_column_groups_for_get_features_factors",
    "pack_agent_daily_feature_table_for_get_features_factors",
    "compute_agent_trading_daily_feature_table",
    "compute_agent_trading_daily_feature_table_multi",
    "compute_agent_trading_daily_feature_table_for_get_features_factors",
    "compute_agent_trading_daily_feature_table_multi_for_get_features_factors",
    "compute_agent_validation_factors_py",
    # 统计分析函数
    "calculate_base_entropy",
    "calculate_shannon_entropy_change",
    "calculate_shannon_entropy_change_at_low",
    "calculate_window_entropy",
    "factor_correlation_by_date",
    "factor_grouping",
    "tail_v2_backtest_block",
    "tail_v2_backtest_block_f32",
    "tail_v2_neutralize_block",
    "tail_v2_neutralize_block_f32",
    "tail_v2_neutralize_block_f32_out",
    "tail_v4_neutralize_block_f32_out_legacy",
    "tail_v4_neutralize_block_exact",
    "TailV4LegacyStyleData",
    "tail_v2_rank_roll_factor_f32",
    "tail_v3_rank_roll_block_f32",
    "tail_v5_rank_fill_roll_block_f32",
    "NeutralizeStdSharedHandle",
    "neutralize_std_precompute_py",
    "neutralize_std_block_with_shared",
    "tail_v2_select_by_ic_corr_abs_f32",
    "tail_v2_select_by_ic_corr_f32",
    "tail_v2_select_by_ic_corr_par_f32",
    "tlq_gate_base_resid",
    "tlq_gate_rank_pct_rows",
    "tlq_gate_interleave_styles",
    "tail_v4_run_candidates",
    "tail_v4_run_fulltest_queue",
    "segment_and_correlate",
    "lz_complexity",
    "lz_complexity_detailed",
    "mutual_information_knn",
    "mutual_information_knn_chebyshev",
    "mutual_information_2d_knn",
    "mutual_information_2d_knn_chebyshev",
    # Python定义的pandas扩展函数
    "corrwith",
    "rank_axis1_df",
    "rank_axis0_df",
    "fast_rank",
    "fast_rank_axis1",
    "fast_rank_axis0",
    "fast_merge_df",
    "fast_inner_join_df",
    "fast_left_join_df",
    "fast_right_join_df",
    "fast_outer_join_df",
    "fast_join",
    "fast_merge_dataframe",
    # pandas相关性矩阵函数
    "fast_correlation_matrix_v2_df",
    "fast_corr_df",
    "correlation_matrix_df",
    # 因子中性化函数
    "batch_factor_neutralization",
    "batch_factor_neutralization_optimized",
    "batch_factor_neutralization_io_optimized",
    "batch_factor_neutralization_simple_math_optimized",
    "batch_factor_neutralization_parallel_optimized",
    "batch_factor_neutralization_ultimate_optimized",
    "batch_factor_neutralization_simple_fallback",
    # Web管理器
    "BackupWebManager",
    "check_port_available",
    "find_available_port",
    "start_web_manager",
    # 测试函数
    "haha",
    # Agent模拟器函数
    "simulate_momentum_agents_py",
    "simulate_buy_ratio_agents_py",
    "simulate_thematic_agents_py",
    "simulate_bottom_fishing_agents_py",
    "simulate_follow_flow_agents_py",
    "simulate_acceleration_follow_agents_py",
    "simulate_exhaustion_reversal_agents_py",
    "compute_theme_feature_expansion_from_minute",
    "prepare_minute_data_for_theme_feature_expansion",
    "py_read_minute_data",
    "run_factor_pipeline",
    "run_factor_pipeline_minute",
    "py_minute_example",
    "py_minute_example_names",
    "py_qhsk_anneal",
    "py_qhsk_anneal_names",
    "py_minute_capm",
    "py_minute_capm_all",
    "py_minute_capm_at",
    "py_minute_capm_names",
    "py_minute_capm_all_names",
    "py_minute_capm_at_names",
    "py_microstructure_3s_features",
    "py_microstructure_3s_feature_names",
    "py_microstructure_capm",
    "py_microstructure_capm_codes",
    "py_microstructure_capm_names",
    "py_switch_moment",
    "py_switch_moment_names",
    "py_multi_factor_capm",
    "py_multi_factor_capm_names",
    "run_factor_pipeline_cross_section",
    "py_relaynet_six_people",
    "py_relaynet_six_people_names",
    "py_cross_section_example",
    "py_cross_section_example_names",
    "py_orderflow_ctx",
    "py_orderflow_ctx_names",
    "py_orderflow_ctx_v2",
    "py_orderflow_ctx_v2_names",
    "py_dijkstra",
    "py_dijkstra_names",
    "py_relaynet",
    "py_relaynet_names",
    "py_peaks",
    "py_peaks_from_data",
    "py_peaks_names",
    "py_peaks_regime_fit",
    "py_state_read_row",
    "run_factor_pipeline_regime",
    "py_pair_interaction",
    "py_pair_interaction_names",
    "py_yhyb",
    "py_yhyb_params",
    "py_yhyb_from_data",
    "py_yhyb_names",
    "py_yhyb_events",
    "py_yhyb_indtop",
    "py_yhyb_indtop_names",
    "py_vsld",
    "py_vsld_names",
    "py_vsld_from_data",
    "py_hot_stock_pool",
    "py_hot_pool_ind",
    "py_hot_pool_ind_names",
    "py_hot_stock_pool_names",
    "py_compute_drop_event_features",
    "py_drop_event_feature_names",
    "py_drop_event",
    "py_drop_event_names",
    "py_long_order",
    "py_long_order_names",
    "py_long_order_raw",
    "py_urgency",
    "py_urgency_names",
    "py_urgency_ext",
    "py_urgency_ext_names",
    "get_theme_cluster_scatter_3d",
    "adjust_afternoon",
    "read_trade",
    "read_market",
    "read_market_pair",
    "read_minute_data",
    "py_extreme_point_fit_names",
    "py_dagpath",
    "py_dagpath_names",
    "py_distill",
    "py_distill_names",
    "py_hidden_arrange",
    "py_hidden_arrange_names",
    "py_geffe_leak",
    "py_geffe_leak_names",
    "py_lfsr_leak_ext1",
    "py_lfsr_leak_ext1_names",
    "py_lfsr_ext2",
    "py_lfsr_ext2_names",
    "py_distill_tick",
    "py_distill_tick_names",
    "py_hhi_event",
    "py_hhi_event_names",
    "py_etp_event",
    "py_etp_event_names",
    "get_features_factors",
    "get_features_names",
    "py_individual_order_ratio",
    "py_individual_order_ratio_names",
    "py_compute_orderbook_imb_refactor",
    "py_orderbook_imb_refactor_names",
    "py_extreme_point_fit",
    "py_extreme_point_fit_names",
    "py_dagpath",
    "py_dagpath_names",
    "dct_1d",
    "dct_2d",
    "personalized_meeting_feature_names",
    "personalized_meeting_features",
    "orderbook_volume_cov_factors",
    "illusion_liquidity_distance_factors",
    "compute_corr_diff_features",
    "py_anneal_regime_diff",
    "py_anneal_regime_diff_names",
    "py_anneal_regime_diff_from_data",
    "py_anneal_order_profile",
    "py_anneal_order_profile_names",
    "py_anneal_order_profile_from_data",
    "py_anneal_order_profile_v2",
    "py_anneal_order_profile_v2_names",
    "py_anneal_order_profile_v2_from_data",
    "py_anneal_order_profile_v3",
    "py_anneal_order_profile_v3_names",
    "py_anneal_order_profile_v3_from_data",
    "py_anneal_regime_excess",
    "py_anneal_regime_excess_names",
    "py_anneal_regime_excess_from_data",
    "py_anneal_regime_raw_excess",
    "py_anneal_regime_raw_excess_names",
    "py_anneal_regime_raw_excess_from_data",
    "py_anneal_cross_peer",
    "py_anneal_cross_peer_names",
    "py_anneal_cross_peer_from_data",
    "py_anneal_cross_peer_v2",
    "py_anneal_cross_peer_v2_names",
    "py_anneal_cross_peer_v2_from_data",
    "py_anneal_cross_peer_v3",
    "py_anneal_cross_peer_v3_names",
    "py_anneal_cross_peer_v3_profile",
    "py_anneal_cross_peer_v3_profile_names",
    "prepare_anneal_cross_peer_v3_history_from_store",
    "prepare_anneal_cross_peer_v3_history_from_h5",
]


def py_anneal_regime_diff(code: str, date: int) -> List[float]:
    """柜员游戏扩展系列1：时空尺度与行情条件可还原性差异（单股单日 16516 维）。"""
    ...


def py_anneal_regime_diff_names() -> List[str]:
    """返回柜员游戏扩展系列1全部 16516 个因子名。"""
    ...


def py_anneal_regime_diff_from_data(
    exchtime_us: "numpy.ndarray",
    price: "numpy.ndarray",
    volume: "numpy.ndarray",
    turnover: "numpy.ndarray",
    flag: "numpy.ndarray",
    bid_order: "numpy.ndarray",
    ask_order: "numpy.ndarray",
) -> List[float]:
    """v2 无磁盘依赖入口：传入逐笔成交列数组计算系列1全部 16516 个因子。"""
    ...


def py_anneal_order_profile(code: str, date: int) -> List[float]:
    """柜员游戏扩展系列2：多维订单画像与博弈序可还原性差异（单股单日 15384 维）。"""
    ...


def py_anneal_order_profile_names() -> List[str]:
    """返回柜员游戏扩展系列2全部 15384 个因子名。"""
    ...


def py_anneal_order_profile_from_data(
    exchtime_us: "numpy.ndarray",
    price: "numpy.ndarray",
    volume: "numpy.ndarray",
    turnover: "numpy.ndarray",
    flag: "numpy.ndarray",
    bid_order: "numpy.ndarray",
    ask_order: "numpy.ndarray",
) -> List[float]:
    """v2 无磁盘依赖入口：传入逐笔成交列数组计算系列2全部 15384 个因子。"""
    ...


def py_anneal_cross_peer(date: int) -> Tuple[List[str], List[float]]:
    """柜员游戏扩展系列3：全市场横截面联动与同伴可还原性（单日全市场，每股 11760 维）。"""
    ...


def py_anneal_cross_peer_names() -> List[str]:
    """返回柜员游戏扩展系列3全部 11760 个因子名。"""
    ...


def py_anneal_cross_peer_from_data(
    codes: List[str],
    trades_list: List[Optional["numpy.ndarray"]],
) -> Tuple[List[str], List[float]]:
    """v2 无磁盘依赖入口：传入股票代码列表与逐笔成交数组计算系列3横截面因子。"""
    ...


def py_anneal_cross_peer_v2(date: int) -> Tuple[List[str], List[float]]:
    """柜员游戏扩展系列3 修正版（单日全市场，每股 55680 维）：微秒时间切分，60 秒与 30 秒两种时间格，
    全天 / 下午 / 尾盘30分钟 三个时段；算不出来的值为 NaN。"""
    ...


def py_anneal_cross_peer_v2_names() -> List[str]:
    """返回柜员游戏扩展系列3 修正版全部 55680 个因子名。"""
    ...


def py_anneal_cross_peer_v2_from_data(
    codes: List[str],
    trades_list: List[Optional["numpy.ndarray"]],
) -> Tuple[List[str], List[float]]:
    """传数据版：trades_list 每项为 read_trade_fast_us(code, date, 0, True) 的 (n, 8) 数组，读不到传 None。"""
    ...


def py_anneal_cross_peer_v3(date: int) -> Tuple[List[str], List[float]]:
    """柜员游戏拓展3 v3（第二轮，每股 42268 维）；运行前要先调用 prepare_anneal_cross_peer_v3_history_*。"""
    ...


def py_anneal_cross_peer_v3_names() -> List[str]:
    """返回柜员游戏拓展3 v3 全部 42268 个因子名。"""
    ...


def py_anneal_cross_peer_v3_profile(date: int) -> Tuple[List[str], List[float]]:
    """柜员游戏拓展3 v3 第一轮：当天全市场每股 271 列中间数据。"""
    ...


def py_anneal_cross_peer_v3_profile_names() -> List[str]:
    """返回第一轮 271 个中间数据列名。"""
    ...


def prepare_anneal_cross_peer_v3_history_from_store(
    store_dir: str, task_dates: List[int], out_dir: str, n_proc: int = 16
) -> str:
    """研究期：把第一轮因子库里第二轮需要的中间数据取到临时目录。"""
    ...


def prepare_anneal_cross_peer_v3_history_from_h5(
    base_factor_ver: str, base_hdf5_dir: str, task_dates: List[int], out_dir: str
) -> str:
    """生产期：把 base h5 里第二轮需要的中间数据取到临时目录。"""
    ...


def factor_store_storage_profile(path: str) -> dict:
    """按实际挂载识别 NAS/本机存储，跟随软链接；支持尚未创建的 store 路径。"""
    ...


def py_anneal_order_profile_v2(code: str, date: int) -> List[float]:
    """柜员游戏扩展系列2 修正版（单股单日 7873 列）：微秒时间切分、打乱对照超额、显式成交量结构。"""
    ...


def py_anneal_order_profile_v2_names() -> List[str]:
    """返回柜员游戏扩展系列2 修正版全部 7873 个因子名。"""
    ...


def py_anneal_order_profile_v2_from_data(
    exchtime_us: "numpy.ndarray",
    price: "numpy.ndarray",
    volume: "numpy.ndarray",
    turnover: "numpy.ndarray",
    flag: "numpy.ndarray",
    bid_order: "numpy.ndarray",
    ask_order: "numpy.ndarray",
) -> List[float]:
    """无磁盘依赖入口：传入逐笔成交列数组计算修正版全部 7873 个因子。"""
    ...


def py_anneal_regime_excess(code: str, date: int) -> List[float]:
    """柜员游戏扩展系列1 新版（单股单日 44832 列）：微秒时间切分、打乱对照超额、30 秒分辨率、全天/下午/尾盘30分钟三段。"""
    ...


def py_anneal_regime_excess_names() -> List[str]:
    """返回柜员游戏扩展系列1 新版全部 44832 个因子名。"""
    ...


def py_anneal_regime_excess_from_data(
    exchtime_us: "numpy.ndarray",
    price: "numpy.ndarray",
    volume: "numpy.ndarray",
    turnover: "numpy.ndarray",
    flag: "numpy.ndarray",
    bid_order: "numpy.ndarray",
    ask_order: "numpy.ndarray",
) -> List[float]:
    """无磁盘依赖入口：传入逐笔成交列数组计算新版全部 44832 个因子。"""
    ...


def py_anneal_regime_raw_excess(code: str, date: int) -> List[float]:
    """柜员游戏扩展系列1 v3（单股单日 44694 列）：v1 原始退火指标 + 打乱对照超额 + 订单量结构，微秒时间、30 秒、全天/下午/尾盘30分钟。"""
    ...


def py_anneal_regime_raw_excess_names() -> List[str]:
    """返回柜员游戏扩展系列1 v3 全部 44694 个因子名。"""
    ...


def py_anneal_regime_raw_excess_from_data(
    exchtime_us: "numpy.ndarray",
    price: "numpy.ndarray",
    volume: "numpy.ndarray",
    turnover: "numpy.ndarray",
    flag: "numpy.ndarray",
    bid_order: "numpy.ndarray",
    ask_order: "numpy.ndarray",
) -> List[float]:
    """无磁盘依赖入口：传入逐笔成交列数组计算 v3 全部 44694 个因子。"""
    ...


def py_anneal_order_profile_v3(code: str, date: int) -> List[float]:
    """柜员游戏扩展系列2 第三版（单股单日 30045 列）：v1 原始退火指标 + v2 打乱对照超额 + 成交量结构与活跃度，5 个时段，分钟矩阵分全天 / 下午 / 尾盘三段降维。"""
    ...


def py_anneal_order_profile_v3_names() -> List[str]:
    """返回柜员游戏扩展系列2 第三版全部 30045 个因子名。"""
    ...


def py_anneal_order_profile_v3_from_data(
    exchtime_us: "numpy.ndarray",
    price: "numpy.ndarray",
    volume: "numpy.ndarray",
    turnover: "numpy.ndarray",
    flag: "numpy.ndarray",
    bid_order: "numpy.ndarray",
    ask_order: "numpy.ndarray",
) -> List[float]:
    """无磁盘依赖入口：传入逐笔成交列数组计算第三版全部 30045 个因子。"""
    ...
