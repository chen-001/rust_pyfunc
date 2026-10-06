//! hm060「拼团成功」—— 旧框架逐股系列的 Rust 复现。
//!
//! 旧文件：/home/chenzongwei/pythoncode/成果hm60_拼团成功/hm60_okay.py（560 行）
//!
//! ════════════════════════════════════════════════════════════════════
//! 一、旧 get_factor 做了什么（逐步对照）
//! ════════════════════════════════════════════════════════════════════
//! run_base 里 Go(dwo.TickDataBase) 的参数：get_many=0, read_full=0, change_time=1。
//!
//! 1) df = read_trade_data_single(with_retreat=0)
//!      → pd.read_csv(usecols=[date,exchtime,price,volume,turnover,flag,localtime,
//!                             ask_order,bid_order])，再丢掉 flag==32 的行；
//!      → deal_data：exchtime 从「epoch 微秒」变成 Timestamp（+8 小时后按北京钟点），
//!        localtime 同样处理，另加一列 delay。
//! 2) df = dwo.adjust_afternoon(df)   # 列形态分支
//!      上午 between_time('09:30:00','11:30:00')、下午 between_time('13:00:00','14:57:00')
//!      下午整体前移 90 分钟，然后 concat([上午, 下午]) 再 reset_index。
//!      pandas 的 between_time 是**微秒级闭区间**（_get_time_micros = ns%一天//1000）。
//! 3) if df.shape[0] <= 30: return [nan] * 837 * 2
//! 4) res = sing(df, 5) + sing(df, 1)
//!      sing(df, param):
//!        df, df_names = rp.order_neighborhood_analysis(
//!            df.ask_order.to_numpy(int), df.bid_order.to_numpy(int),
//!            df.volume.to_numpy(float), df.exchtime.to_numpy(int),
//!            'percentage', 1000, param, num_threads=8, timeout_ms=5000)
//!        df_buy  = df[df.订单类型 == 0]
//!        df_sell = df[df.订单类型 == 1]
//!        return one(df) + one(df_buy) + one(df_sell)
//!      one(frame) 先算 C(18,2)=153 个两两相关系数，再依次算
//!        18 个 mean、18 个 std/mean、18 个 skew、18 个 kurt、18 个 max、
//!        18 个 autocorr（= frame.corrwith(frame.shift(1))）、18 个 trend
//!        （= rp.trend_2d(frame.to_numpy(float), axis=0)），合计 279 个。
//! 5) names 的拼法：
//!      names = [base+'_all' for base] + [base+'_buy'] + [base+'_sell']   # 837
//!      names = [n+'_编号附近5perc' for n in names] + [n+'_编号附近1perc' ...]  # 1674
//!    所以前 837 个是 param=5，后 837 个是 param=1；每一段里 all / buy / sell 各 279。
//!
//! 整段包在 try/except 里，抛异常就返回 None（上游写整条 NaN）。所以
//! 邻居表为空（`order_neighborhood_analysis` 返回 0 行）时 pandas 会在
//! `df.max()` 上抛 ValueError，最终效果就是这 1674 个值全是 NaN。
//!
//! ════════════════════════════════════════════════════════════════════
//! 二、order_neighborhood_analysis 的算法（照着 src/order_neighborhood.rs 抄）
//! ════════════════════════════════════════════════════════════════════
//! · 聚合：按 ask_order / bid_order 汇总，ask 记 order_type=1、bid 记 0；
//!   量是 `volume[i] as f32` 顺序累加（**f32**，不是 f64），time 取该编号的最大 exchtime；
//!   最后按编号升序排序（HashMap 的随机迭代顺序被排序抹平，结果确定）。
//! · 邻域：percentage 模式下 range = max(1, trunc(|id| * param / 100))，
//!   邻居 = 排序后编号落在 [id-range, id+range] 的其它订单，按 order_type 分成 same / diff，
//!   收集顺序就是编号升序。
//! · 18 个指标见 metrics_of，逐行照抄 ultra_fast_compute_metrics。
//!
//! 注意：库里的 `order_neighborhood_analysis` 是 #[pyfunction]，worker 二进制不带
//! libpython，链路上不能碰 PyResult，所以本文件把算法整段搬过来自己实现，不改公共文件。
//!
//! ════════════════════════════════════════════════════════════════════
//! 三、时间与窗口（必须逐位对齐，不能偷懒）
//! ════════════════════════════════════════════════════════════════════
//! · CSV 的 exchtime 是「带 .0 的浮点串」，整数前缀就是 epoch 微秒；flag 同理按整数比 32。
//! · pandas 的 exchtime 时间戳不是「微秒 × 1000」：
//!     v = us / 1e6; base = (int64)v; frac = np.round(v - base, 9);
//!     ns = base * 1e9 + (int64)(frac * 1e9) + 8h
//!   （pandas/_libs/tslibs/conversion.pyx 的 cast_from_unit_vectorized，unit='s'→'ns'）。
//!   实测 1740964500130000 µs 对应 1740993300130000114 ns，比 us*1000 多 114 ns；
//!   1527816600000001 µs 少 46 ns。差到几百 ns，而 between_time 判的是「当天微秒
//!   = ns % 一天 // 1000」，正好可能把边界那一微秒判反，所以必须照抄这条链。
//! · between_time 的两端都闭，窗口是 [09:30:00, 11:30:00] 与 [13:00:00, 14:57:00]。
//! · 下午的行整体减 90 分钟（index 是 ns，减 5400e9）后再 concat，行序 = 上午（文件序）
//!   在前、下午（文件序）在后；这个顺序决定了聚合里 f32 累加的顺序，必须对上。
//!
//! ════════════════════════════════════════════════════════════════════
//! 四、pandas 语义对齐
//! ════════════════════════════════════════════════════════════════════
//! · frame.A.corr(frame.B)：Series.corr → nanops.nancorr → **np.corrcoef**（两遍法），
//!   走 cov/(sqrt(cov00)*sqrt(cov11))，本文件用 nan_corr 复刻。
//! · frame.corrwith(frame.shift(1))：走的是 DataFrame.corrwith 的 pearson 分支，
//!   num = Σ(left-mean_left)(right-mean_right)、dom = (count-1)*std_left*std_right，
//!   和 np.corrcoef 不是同一条路，本文件单列 autocorr_shift 复刻。
//! · mean/std/skew/kurt 都是 nanops 那条（无 NaN 时 count=n）：
//!   mean = Σ/n；var = Σ(x-mean)²/(n-1)；skew = ((n√(n-1))/(n-2))·m3/m2^1.5，
//!   m2==0 时取 0，n<3 给 NaN；kurt 的 adj = 3(n-1)²/((n-2)(n-3))，
//!   denominator==0 时取 0，n<4 给 NaN；两者都对 m2/m3 做 |x|<1e-14 归零。
//!   求和照抄 numpy 的成对求和（PW_BLOCKSIZE=128、8 路展开），见 pw_sum。
//! · trend 与 rp.trend_2d / calculate_trend_1d 一致（顺序求和，非成对）。
//!
//! ════════════════════════════════════════════════════════════════════
//! 五、已知差异
//! ════════════════════════════════════════════════════════════════════
//! · 旧代码给 order_neighborhood_analysis 传了 timeout_ms=5000，超时就返回 None
//!   （随后 pandas 抛异常 → 整天 NaN）。这是旧框架自己带进来的机器负载相关行为，
//!   本文件不设超时：四组对账里旧代码都没超时，走的是同一条计算路径。
//! · names() 的 1674 个名字里没有重名（已逐项核对），不需要 _dup2 后缀。
//!
//! 不改现有 hmXX_okay.py，也不动 src/order_neighborhood.rs 等公共文件。

use crate::data_paths;
use rayon::prelude::*;
use std::collections::HashMap;

/// 一个订单的 18 个邻域指标。
const NCOL: usize = 18;
/// 每个 (all/buy/sell) 段的指标数：153 个相关系数 + 7 组 × 18。
const N_BASE: usize = 279;
/// 最终每个股票日输出的因子个数。
const N_TOTAL: usize = 1674;

/// 一天的纳秒数。
const DAY_NS: i64 = 86_400_000_000_000;
/// 一天前的 8 小时偏移（epoch → 北京钟点）。
const EXCHANGE_OFFSET_NS: i64 = 8 * 3600 * 1_000_000_000;
/// 下午整体前移的纳秒数。
const AFTERNOON_SHIFT_NS: i64 = 90 * 60 * 1_000_000_000;

// between_time 判的是「当天微秒」，两端闭区间。
const MORNING_START_US: i64 = (9 * 3600 + 30 * 60) * 1_000_000;
const MORNING_END_US: i64 = (11 * 3600 + 30 * 60) * 1_000_000;
const AFTERNOON_START_US: i64 = (13 * 3600) * 1_000_000;
const AFTERNOON_END_US: i64 = (14 * 3600 + 57 * 60) * 1_000_000;

const COL_EXCHTIME: usize = 4;
const COL_VOLUME: usize = 8;
const COL_FLAG: usize = 10;
const COL_ASK_ORDER: usize = 13;
const COL_BID_ORDER: usize = 14;

/// 一行逐笔里本系列用得到的字段。
#[derive(Clone, Copy)]
struct Row {
    time: i64,
    volume: f64,
    ask: i64,
    bid: i64,
}

/// 聚合后的订单（32 字节对齐是库里的写法，这里只要字段一致）。
#[derive(Clone, Copy)]
struct Order {
    id: i64,
    volume: f32,
    time: i64,
    typ: u8,
}
const BASE_NAMES: [&str; 279] = [
    "corr_订单编号_订单类型", "corr_订单编号_同向成交量比", "corr_订单编号_异向成交量比", "corr_订单编号_同向均量比",
    "corr_订单编号_异向均量比", "corr_订单编号_总成交量比", "corr_订单编号_总均量比", "corr_订单编号_同向邻居数",
    "corr_订单编号_异向邻居数", "corr_订单编号_同向时差总和", "corr_订单编号_异向时差总和", "corr_订单编号_同向时差均值",
    "corr_订单编号_异向时差均值", "corr_订单编号_同向时差量相关", "corr_订单编号_异向时差量相关", "corr_订单编号_同向时间量相关",
    "corr_订单编号_异向时间量相关", "corr_订单类型_同向成交量比", "corr_订单类型_异向成交量比", "corr_订单类型_同向均量比",
    "corr_订单类型_异向均量比", "corr_订单类型_总成交量比", "corr_订单类型_总均量比", "corr_订单类型_同向邻居数",
    "corr_订单类型_异向邻居数", "corr_订单类型_同向时差总和", "corr_订单类型_异向时差总和", "corr_订单类型_同向时差均值",
    "corr_订单类型_异向时差均值", "corr_订单类型_同向时差量相关", "corr_订单类型_异向时差量相关", "corr_订单类型_同向时间量相关",
    "corr_订单类型_异向时间量相关", "corr_同向成交量比_异向成交量比", "corr_同向成交量比_同向均量比", "corr_同向成交量比_异向均量比",
    "corr_同向成交量比_总成交量比", "corr_同向成交量比_总均量比", "corr_同向成交量比_同向邻居数", "corr_同向成交量比_异向邻居数",
    "corr_同向成交量比_同向时差总和", "corr_同向成交量比_异向时差总和", "corr_同向成交量比_同向时差均值", "corr_同向成交量比_异向时差均值",
    "corr_同向成交量比_同向时差量相关", "corr_同向成交量比_异向时差量相关", "corr_同向成交量比_同向时间量相关", "corr_同向成交量比_异向时间量相关",
    "corr_异向成交量比_同向均量比", "corr_异向成交量比_异向均量比", "corr_异向成交量比_总成交量比", "corr_异向成交量比_总均量比",
    "corr_异向成交量比_同向邻居数", "corr_异向成交量比_异向邻居数", "corr_异向成交量比_同向时差总和", "corr_异向成交量比_异向时差总和",
    "corr_异向成交量比_同向时差均值", "corr_异向成交量比_异向时差均值", "corr_异向成交量比_同向时差量相关", "corr_异向成交量比_异向时差量相关",
    "corr_异向成交量比_同向时间量相关", "corr_异向成交量比_异向时间量相关", "corr_同向均量比_异向均量比", "corr_同向均量比_总成交量比",
    "corr_同向均量比_总均量比", "corr_同向均量比_同向邻居数", "corr_同向均量比_异向邻居数", "corr_同向均量比_同向时差总和",
    "corr_同向均量比_异向时差总和", "corr_同向均量比_同向时差均值", "corr_同向均量比_异向时差均值", "corr_同向均量比_同向时差量相关",
    "corr_同向均量比_异向时差量相关", "corr_同向均量比_同向时间量相关", "corr_同向均量比_异向时间量相关", "corr_异向均量比_总成交量比",
    "corr_异向均量比_总均量比", "corr_异向均量比_同向邻居数", "corr_异向均量比_异向邻居数", "corr_异向均量比_同向时差总和",
    "corr_异向均量比_异向时差总和", "corr_异向均量比_同向时差均值", "corr_异向均量比_异向时差均值", "corr_异向均量比_同向时差量相关",
    "corr_异向均量比_异向时差量相关", "corr_异向均量比_同向时间量相关", "corr_异向均量比_异向时间量相关", "corr_总成交量比_总均量比",
    "corr_总成交量比_同向邻居数", "corr_总成交量比_异向邻居数", "corr_总成交量比_同向时差总和", "corr_总成交量比_异向时差总和",
    "corr_总成交量比_同向时差均值", "corr_总成交量比_异向时差均值", "corr_总成交量比_同向时差量相关", "corr_总成交量比_异向时差量相关",
    "corr_总成交量比_同向时间量相关", "corr_总成交量比_异向时间量相关", "corr_总均量比_同向邻居数", "corr_总均量比_异向邻居数",
    "corr_总均量比_同向时差总和", "corr_总均量比_异向时差总和", "corr_总均量比_同向时差均值", "corr_总均量比_异向时差均值",
    "corr_总均量比_同向时差量相关", "corr_总均量比_异向时差量相关", "corr_总均量比_同向时间量相关", "corr_总均量比_异向时间量相关",
    "corr_同向邻居数_异向邻居数", "corr_同向邻居数_同向时差总和", "corr_同向邻居数_异向时差总和", "corr_同向邻居数_同向时差均值",
    "corr_同向邻居数_异向时差均值", "corr_同向邻居数_同向时差量相关", "corr_同向邻居数_异向时差量相关", "corr_同向邻居数_同向时间量相关",
    "corr_同向邻居数_异向时间量相关", "corr_异向邻居数_同向时差总和", "corr_异向邻居数_异向时差总和", "corr_异向邻居数_同向时差均值",
    "corr_异向邻居数_异向时差均值", "corr_异向邻居数_同向时差量相关", "corr_异向邻居数_异向时差量相关", "corr_异向邻居数_同向时间量相关",
    "corr_异向邻居数_异向时间量相关", "corr_同向时差总和_异向时差总和", "corr_同向时差总和_同向时差均值", "corr_同向时差总和_异向时差均值",
    "corr_同向时差总和_同向时差量相关", "corr_同向时差总和_异向时差量相关", "corr_同向时差总和_同向时间量相关", "corr_同向时差总和_异向时间量相关",
    "corr_异向时差总和_同向时差均值", "corr_异向时差总和_异向时差均值", "corr_异向时差总和_同向时差量相关", "corr_异向时差总和_异向时差量相关",
    "corr_异向时差总和_同向时间量相关", "corr_异向时差总和_异向时间量相关", "corr_同向时差均值_异向时差均值", "corr_同向时差均值_同向时差量相关",
    "corr_同向时差均值_异向时差量相关", "corr_同向时差均值_同向时间量相关", "corr_同向时差均值_异向时间量相关", "corr_异向时差均值_同向时差量相关",
    "corr_异向时差均值_异向时差量相关", "corr_异向时差均值_同向时间量相关", "corr_异向时差均值_异向时间量相关", "corr_同向时差量相关_异向时差量相关",
    "corr_同向时差量相关_同向时间量相关", "corr_同向时差量相关_异向时间量相关", "corr_异向时差量相关_同向时间量相关", "corr_异向时差量相关_异向时间量相关",
    "corr_同向时间量相关_异向时间量相关", "订单编号_mean", "订单类型_mean", "同向成交量比_mean",
    "异向成交量比_mean", "同向均量比_mean", "异向均量比_mean", "总成交量比_mean",
    "总均量比_mean", "同向邻居数_mean", "异向邻居数_mean", "同向时差总和_mean",
    "异向时差总和_mean", "同向时差均值_mean", "异向时差均值_mean", "同向时差量相关_mean",
    "异向时差量相关_mean", "同向时间量相关_mean", "异向时间量相关_mean", "订单编号_std_mean",
    "订单类型_std_mean", "同向成交量比_std_mean", "异向成交量比_std_mean", "同向均量比_std_mean",
    "异向均量比_std_mean", "总成交量比_std_mean", "总均量比_std_mean", "同向邻居数_std_mean",
    "异向邻居数_std_mean", "同向时差总和_std_mean", "异向时差总和_std_mean", "同向时差均值_std_mean",
    "异向时差均值_std_mean", "同向时差量相关_std_mean", "异向时差量相关_std_mean", "同向时间量相关_std_mean",
    "异向时间量相关_std_mean", "订单编号_skew", "订单类型_skew", "同向成交量比_skew",
    "异向成交量比_skew", "同向均量比_skew", "异向均量比_skew", "总成交量比_skew",
    "总均量比_skew", "同向邻居数_skew", "异向邻居数_skew", "同向时差总和_skew",
    "异向时差总和_skew", "同向时差均值_skew", "异向时差均值_skew", "同向时差量相关_skew",
    "异向时差量相关_skew", "同向时间量相关_skew", "异向时间量相关_skew", "订单编号_kurt",
    "订单类型_kurt", "同向成交量比_kurt", "异向成交量比_kurt", "同向均量比_kurt",
    "异向均量比_kurt", "总成交量比_kurt", "总均量比_kurt", "同向邻居数_kurt",
    "异向邻居数_kurt", "同向时差总和_kurt", "异向时差总和_kurt", "同向时差均值_kurt",
    "异向时差均值_kurt", "同向时差量相关_kurt", "异向时差量相关_kurt", "同向时间量相关_kurt",
    "异向时间量相关_kurt", "订单编号_max", "订单类型_max", "同向成交量比_max",
    "异向成交量比_max", "同向均量比_max", "异向均量比_max", "总成交量比_max",
    "总均量比_max", "同向邻居数_max", "异向邻居数_max", "同向时差总和_max",
    "异向时差总和_max", "同向时差均值_max", "异向时差均值_max", "同向时差量相关_max",
    "异向时差量相关_max", "同向时间量相关_max", "异向时间量相关_max", "订单编号_autocorr",
    "订单类型_autocorr", "同向成交量比_autocorr", "异向成交量比_autocorr", "同向均量比_autocorr",
    "异向均量比_autocorr", "总成交量比_autocorr", "总均量比_autocorr", "同向邻居数_autocorr",
    "异向邻居数_autocorr", "同向时差总和_autocorr", "异向时差总和_autocorr", "同向时差均值_autocorr",
    "异向时差均值_autocorr", "同向时差量相关_autocorr", "异向时差量相关_autocorr", "同向时间量相关_autocorr",
    "异向时间量相关_autocorr", "订单编号_trend", "订单类型_trend", "同向成交量比_trend",
    "异向成交量比_trend", "同向均量比_trend", "异向均量比_trend", "总成交量比_trend",
    "总均量比_trend", "同向邻居数_trend", "异向邻居数_trend", "同向时差总和_trend",
    "异向时差总和_trend", "同向时差均值_trend", "异向时差均值_trend", "同向时差量相关_trend",
    "异向时差量相关_trend", "同向时间量相关_trend", "异向时间量相关_trend",
];

/// 153 个相关系数对应的列下标对，顺序与 BASE_NAMES 的前 153 项一致。
const CORR_PAIRS: [(usize, usize); 153] = [
    (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8),
    (0, 9), (0, 10), (0, 11), (0, 12), (0, 13), (0, 14), (0, 15), (0, 16),
    (0, 17), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8),
    (1, 9), (1, 10), (1, 11), (1, 12), (1, 13), (1, 14), (1, 15), (1, 16),
    (1, 17), (2, 3), (2, 4), (2, 5), (2, 6), (2, 7), (2, 8), (2, 9),
    (2, 10), (2, 11), (2, 12), (2, 13), (2, 14), (2, 15), (2, 16), (2, 17),
    (3, 4), (3, 5), (3, 6), (3, 7), (3, 8), (3, 9), (3, 10), (3, 11),
    (3, 12), (3, 13), (3, 14), (3, 15), (3, 16), (3, 17), (4, 5), (4, 6),
    (4, 7), (4, 8), (4, 9), (4, 10), (4, 11), (4, 12), (4, 13), (4, 14),
    (4, 15), (4, 16), (4, 17), (5, 6), (5, 7), (5, 8), (5, 9), (5, 10),
    (5, 11), (5, 12), (5, 13), (5, 14), (5, 15), (5, 16), (5, 17), (6, 7),
    (6, 8), (6, 9), (6, 10), (6, 11), (6, 12), (6, 13), (6, 14), (6, 15),
    (6, 16), (6, 17), (7, 8), (7, 9), (7, 10), (7, 11), (7, 12), (7, 13),
    (7, 14), (7, 15), (7, 16), (7, 17), (8, 9), (8, 10), (8, 11), (8, 12),
    (8, 13), (8, 14), (8, 15), (8, 16), (8, 17), (9, 10), (9, 11), (9, 12),
    (9, 13), (9, 14), (9, 15), (9, 16), (9, 17), (10, 11), (10, 12), (10, 13),
    (10, 14), (10, 15), (10, 16), (10, 17), (11, 12), (11, 13), (11, 14), (11, 15),
    (11, 16), (11, 17), (12, 13), (12, 14), (12, 15), (12, 16), (12, 17), (13, 14),
    (13, 15), (13, 16), (13, 17), (14, 15), (14, 16), (14, 17), (15, 16), (15, 17),
    (16, 17),
];

// ══════════════════════════════════════════════════════════════════════
// numpy 的成对求和（numpy/core/src/umath/loops_utils.h: pairwise_sum_DOUBLE）
// ══════════════════════════════════════════════════════════════════════

const PW_BLOCKSIZE: usize = 128;

fn pw_sum(a: &[f64]) -> f64 {
    let n = a.len();
    if n < 8 {
        let mut res = 0.0f64;
        for i in 0..n {
            res += a[i];
        }
        return res;
    }
    if n <= PW_BLOCKSIZE {
        let mut r = [0.0f64; 8];
        r.copy_from_slice(&a[..8]);
        let stop = n - (n % 8);
        let mut i = 8usize;
        while i < stop {
            for j in 0..8 {
                r[j] += a[i + j];
            }
            i += 8;
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        let mut leftover = 0.0f64;
        for k in stop..n {
            leftover += a[k];
        }
        res += leftover;
        return res;
    }
    let mut n2 = n / 2;
    n2 -= n2 % 8;
    pw_sum(&a[..n2]) + pw_sum(&a[n2..])
}

// ══════════════════════════════════════════════════════════════════════
// pandas 的统计量（nanops 那条路，本系列所有指标列都没有 NaN）
// ══════════════════════════════════════════════════════════════════════

/// Series.corr / nanops.nancorr 的 pearson：np.corrcoef(a, b)[0, 1]。
fn nan_corr(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len();
    if n == 0 {
        return f64::NAN;
    }
    let nf = n as f64;
    let ma = pw_sum(a) / nf;
    let mb = pw_sum(b) / nf;
    let mut num = 0.0f64;
    let mut sa = 0.0f64;
    let mut sb = 0.0f64;
    for i in 0..n {
        let da = a[i] - ma;
        let db = b[i] - mb;
        num += da * db;
        sa += da * da;
        sb += db * db;
    }
    let fact = (n - 1) as f64;
    let c01 = num / fact;
    let s0 = (sa / fact).sqrt();
    let s1 = (sb / fact).sqrt();
    (c01 / s0) / s1
}

/// DataFrame.corrwith(shift(1)) 的 pearson 分支：
/// num = Σ(left-mean_l)(right-mean_r)，dom = (count-1)*std_l*std_r。
fn autocorr_shift(v: &[f64]) -> f64 {
    let n = v.len();
    if n < 2 {
        return f64::NAN;
    }
    let l = &v[1..];
    let r = &v[..n - 1];
    let m = l.len();
    if m < 2 {
        return f64::NAN;
    }
    let mf = m as f64;
    let ml = pw_sum(l) / mf;
    let mr = pw_sum(r) / mf;
    // left = left + right*0 之后第 0 行是 NaN，sum 时当 0，所以product数组前面补一个 0。
    let mut prod = Vec::with_capacity(m + 1);
    prod.push(0.0f64);
    for i in 0..m {
        prod.push((l[i] - ml) * (r[i] - mr));
    }
    let num = pw_sum(&prod);
    let mut s1 = 0.0f64;
    let mut s2 = 0.0f64;
    for i in 0..m {
        let d = l[i] - ml;
        s1 += d * d;
    }
    for i in 0..m {
        let d = r[i] - mr;
        s2 += d * d;
    }
    let dd = (m - 1) as f64;
    let sl = (s1 / dd).sqrt();
    let sr = (s2 / dd).sqrt();
    let dom = (dd * sl) * sr;
    num / dom
}

/// nanops.nanstd(ddof=1)：sqrt(Σ(x-mean)²/(n-1))。
fn std_of(v: &[f64], mean: f64, n: usize) -> f64 {
    if n < 2 {
        return f64::NAN;
    }
    let sqr: Vec<f64> = v.iter().map(|&x| (mean - x) * (mean - x)).collect();
    (pw_sum(&sqr) / (n - 1) as f64).sqrt()
}

fn zero_out_fperr(x: f64) -> f64 {
    if x.abs() < 1e-14 {
        0.0
    } else {
        x
    }
}

/// nanops.nanskew：调整后的 Fisher-Pearson 偏度 G1。
fn skew_of(v: &[f64], n: usize) -> f64 {
    if n == 0 {
        return f64::NAN;
    }
    let count = n as f64;
    let mean = pw_sum(v) / count;
    let mut a2 = Vec::with_capacity(n);
    let mut a3 = Vec::with_capacity(n);
    for &x in v {
        let d = x - mean;
        let d2 = d * d;
        a2.push(d2);
        a3.push(d2 * d);
    }
    let m2 = zero_out_fperr(pw_sum(&a2));
    let m3 = zero_out_fperr(pw_sum(&a3));
    let result = (count * (count - 1.0).sqrt() / (count - 2.0)) * (m3 / m2.powf(1.5));
    if n < 3 {
        return f64::NAN;
    }
    if m2 == 0.0 {
        0.0
    } else {
        result
    }
}

/// nanops.nankurt：调整后的 Fisher-Pearson 超额峰度 G2。
fn kurt_of(v: &[f64], n: usize) -> f64 {
    if n == 0 {
        return f64::NAN;
    }
    let count = n as f64;
    let mean = pw_sum(v) / count;
    let mut a2 = Vec::with_capacity(n);
    let mut a4 = Vec::with_capacity(n);
    for &x in v {
        let d = x - mean;
        let d2 = d * d;
        a2.push(d2);
        a4.push(d2 * d2);
    }
    let m2 = pw_sum(&a2);
    let m4 = pw_sum(&a4);
    let adj = 3.0 * (count - 1.0).powi(2) / ((count - 2.0) * (count - 3.0));
    let numerator = zero_out_fperr(count * (count + 1.0) * (count - 1.0) * m4);
    let denominator = zero_out_fperr((count - 2.0) * (count - 3.0) * m2.powi(2));
    if n < 4 {
        return f64::NAN;
    }
    if denominator == 0.0 {
        return 0.0;
    }
    numerator / denominator - adj
}

/// rp.trend_2d(axis=0) / calculate_trend_1d：列值与行号 1..n 的相关系数。
fn trend_of(v: &[f64]) -> f64 {
    let mut vals: Vec<f64> = Vec::with_capacity(v.len());
    let mut idx: Vec<f64> = Vec::with_capacity(v.len());
    for (i, &x) in v.iter().enumerate() {
        if x.is_finite() {
            vals.push(x);
            idx.push((i + 1) as f64);
        }
    }
    let n = vals.len();
    if n == 0 || n == 1 {
        return 0.0;
    }
    let nf = n as f64;
    let mx = vals.iter().sum::<f64>() / nf;
    let my = idx.iter().sum::<f64>() / nf;
    let mut cov = 0.0f64;
    let mut vx = 0.0f64;
    let mut vy = 0.0f64;
    for i in 0..n {
        let dx = vals[i] - mx;
        let dy = idx[i] - my;
        cov += dx * dy;
        vx += dx * dx;
        vy += dy * dy;
    }
    if vx == 0.0 || vy == 0.0 {
        return 0.0;
    }
    cov / (vx.sqrt() * vy.sqrt())
}

// ══════════════════════════════════════════════════════════════════════
// 逐笔读取（复刻 pandas 读 CSV + deal_data + adjust_afternoon）
// ══════════════════════════════════════════════════════════════════════

fn trim_ascii(raw: &[u8]) -> &[u8] {
    let mut s = 0usize;
    let mut e = raw.len();
    while s < e && raw[s].is_ascii_whitespace() {
        s += 1;
    }
    while e > s && raw[e - 1].is_ascii_whitespace() {
        e -= 1;
    }
    &raw[s..e]
}

/// 取字段的整数前缀。`"32.0"` 给 32，与旧框架 `df.flag != 32` 的数值比较一致。
fn int_prefix(raw: &[u8]) -> i64 {
    let b = trim_ascii(raw);
    let mut i = 0usize;
    let mut neg = false;
    if i < b.len() && (b[i] == b'-' || b[i] == b'+') {
        neg = b[i] == b'-';
        i += 1;
    }
    let mut v: i64 = 0;
    while i < b.len() && b[i].is_ascii_digit() {
        v = v * 10 + (b[i] - b'0') as i64;
        i += 1;
    }
    if neg {
        -v
    } else {
        v
    }
}

/// exchtime 列（形如 `1740964500130000.0`）：整数前缀就是微秒数。
fn us_prefix(raw: &[u8]) -> i64 {
    let b = trim_ascii(raw);
    if !b.contains(&b'e') && !b.contains(&b'E') {
        return int_prefix(b);
    }
    std::str::from_utf8(b)
        .ok()
        .and_then(|s| s.parse::<f64>().ok())
        .map(|v| v.round() as i64)
        .unwrap_or(0)
}

fn num(raw: &[u8]) -> f64 {
    let b = trim_ascii(raw);
    if b.is_empty() {
        return f64::NAN;
    }
    parse_f64_pandas(b)
}

fn pow10_table() -> &'static [f64; 309] {
    static T: std::sync::OnceLock<[f64; 309]> = std::sync::OnceLock::new();
    T.get_or_init(|| {
        let mut t = [1.0f64; 309];
        for (k, v) in t.iter_mut().enumerate() {
            *v = format!("1e{}", k).parse::<f64>().unwrap();
        }
        t
    })
}

/// pandas C 解析器 `precise_xstrtod` 的逐行复刻（与 hm033.rs / hm090.rs 里那份一致）。
fn parse_f64_pandas(b: &[u8]) -> f64 {
    const MAX_DIGITS: usize = 17;
    let n = b.len();
    let mut p = 0usize;
    while p < n && matches!(b[p], b' ' | b'\t' | b'\r' | b'\n') {
        p += 1;
    }
    let mut negative = false;
    if p < n && (b[p] == b'-' || b[p] == b'+') {
        negative = b[p] == b'-';
        p += 1;
    }
    let mut number = 0.0f64;
    let mut exponent: i32 = 0;
    let mut num_digits = 0usize;
    let mut num_decimals = 0usize;
    while p < n && b[p].is_ascii_digit() {
        if num_digits < MAX_DIGITS {
            number = number * 10.0 + (b[p] - b'0') as f64;
            num_digits += 1;
        } else {
            exponent += 1;
        }
        p += 1;
    }
    if p < n && b[p] == b'.' {
        p += 1;
        while num_digits < MAX_DIGITS && p < n && b[p].is_ascii_digit() {
            number = number * 10.0 + (b[p] - b'0') as f64;
            p += 1;
            num_digits += 1;
            num_decimals += 1;
        }
        if num_digits >= MAX_DIGITS {
            while p < n && b[p].is_ascii_digit() {
                p += 1;
            }
        }
        exponent -= num_decimals as i32;
    }
    if num_digits == 0 {
        return 0.0;
    }
    if negative {
        number = -number;
    }
    if p < n && (b[p] | 0x20) == b'e' {
        p += 1;
        let mut eneg = false;
        if p < n && (b[p] == b'-' || b[p] == b'+') {
            eneg = b[p] == b'-';
            p += 1;
        }
        let mut nn: i32 = 0;
        let mut nd = 0usize;
        while nd < MAX_DIGITS && p < n && b[p].is_ascii_digit() {
            nn = nn * 10 + (b[p] - b'0') as i32;
            nd += 1;
            p += 1;
        }
        if eneg {
            exponent -= nn;
        } else {
            exponent += nn;
        }
    }
    if exponent > 308 {
        f64::INFINITY
    } else if exponent > 0 {
        number * pow10_table()[exponent as usize]
    } else if exponent < -308 {
        if exponent < -616 {
            0.0
        } else {
            let t = pow10_table();
            (number / t[(-308 - exponent) as usize]) / t[308]
        }
    } else {
        number / pow10_table()[(-exponent) as usize]
    }
}

/// np.rint：四舍六入五成双。
fn rint_ties_even(x: f64) -> f64 {
    let t = x.trunc();
    let f = x - t;
    if f == 0.5 {
        if (t as i64) % 2 == 0 {
            t
        } else {
            t + 1.0
        }
    } else if f == -0.5 {
        if ((t as i64) % 2) == 0 {
            t
        } else {
            t - 1.0
        }
    } else {
        x.round()
    }
}

/// 复刻 pandas 的 `pd.to_timedelta(us/1e6, unit='s') + Timestamp('1970-01-01 08:00:00')`。
fn pandas_ns(us: i64) -> i64 {
    let v = us as f64 / 1e6;
    let base = v as i64;
    let frac = v - base as f64;
    let frac_rounded = rint_ties_even(frac * 1e9) / 1e9;
    base * 1_000_000_000 + (frac_rounded * 1e9) as i64 + EXCHANGE_OFFSET_NS
}

/// 读一个交易日的逐笔，复刻 `read_trade_data_single(with_retreat=0)` + `adjust_afternoon`。
/// 行序 = 上午（文件序）在前、下午（文件序）在后。
fn read_day(code: &str, date: i64) -> Option<Vec<Row>> {
    let filename = format!("{}_{}_transaction.csv", code, date);
    let mut data = None;
    for root in data_paths::level2_roots() {
        let path = root
            .join(date.to_string())
            .join("transaction")
            .join(&filename);
        if let Ok(bytes) = std::fs::read(&path) {
            data = Some(bytes);
            break;
        }
    }
    let data = data?;

    let mut morning: Vec<Row> = Vec::new();
    let mut afternoon: Vec<Row> = Vec::new();
    for (lineno, line) in data.split(|&b| b == b'\n').enumerate() {
        if lineno == 0 || line.is_empty() {
            continue;
        }
        let mut fields: [&[u8]; 15] = [&[][..]; 15];
        let mut commas = 0usize;
        let mut start = 0usize;
        for (i, &b) in line.iter().enumerate() {
            if b == b',' {
                if commas < 15 {
                    fields[commas] = &line[start..i];
                }
                commas += 1;
                start = i + 1;
            }
        }
        if commas < 15 {
            fields[commas] = &line[start..];
        }
        if commas + 1 < 15 {
            continue;
        }
        let exchtime_us = us_prefix(fields[COL_EXCHTIME]);
        if exchtime_us == 0 {
            continue;
        }
        if int_prefix(fields[COL_FLAG]) == 32 {
            continue;
        }
        let time = pandas_ns(exchtime_us);
        let tod_us = time.rem_euclid(DAY_NS) / 1000;
        let row = Row {
            time,
            volume: num(fields[COL_VOLUME]),
            ask: int_prefix(fields[COL_ASK_ORDER]),
            bid: int_prefix(fields[COL_BID_ORDER]),
        };
        if (MORNING_START_US..=MORNING_END_US).contains(&tod_us) {
            morning.push(row);
        } else if (AFTERNOON_START_US..=AFTERNOON_END_US).contains(&tod_us) {
            afternoon.push(Row {
                time: time - AFTERNOON_SHIFT_NS,
                ..row
            });
        }
    }
    morning.append(&mut afternoon);
    Some(morning)
}

// ══════════════════════════════════════════════════════════════════════
// 邻域指标（照抄 src/order_neighborhood.rs 的 ultra_fast_* 那一套）
// ══════════════════════════════════════════════════════════════════════

fn build_orders(rows: &[Row]) -> Vec<Order> {
    let mut map: HashMap<i64, (f32, i64, u8)> = HashMap::new();
    for r in rows {
        if r.ask != 0 {
            let vol = r.volume as f32;
            match map.get_mut(&r.ask) {
                Some(e) => {
                    e.0 += vol;
                    if r.time > e.1 {
                        e.1 = r.time;
                    }
                }
                None => {
                    map.insert(r.ask, (vol, r.time, 1));
                }
            }
        }
        if r.bid != 0 {
            let vol = r.volume as f32;
            match map.get_mut(&r.bid) {
                Some(e) => {
                    e.0 += vol;
                    if r.time > e.1 {
                        e.1 = r.time;
                    }
                }
                None => {
                    map.insert(r.bid, (vol, r.time, 0));
                }
            }
        }
    }
    let mut orders: Vec<Order> = map
        .into_iter()
        .map(|(id, (v, t, ty))| Order {
            id,
            volume: v,
            time: t,
            typ: ty,
        })
        .collect();
    orders.sort_unstable_by_key(|o| o.id);
    orders
}

fn binary_search_ge(orders: &[Order], target: i64) -> usize {
    let mut left = 0usize;
    let mut right = orders.len();
    while left < right {
        let mid = (left + right) >> 1;
        if orders[mid].id < target {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    left
}

fn binary_search_le(orders: &[Order], target: i64) -> usize {
    let mut left = 0usize;
    let mut right = orders.len();
    while left < right {
        let mid = (left + right) >> 1;
        if orders[mid].id <= target {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    if left > 0 {
        left - 1
    } else {
        0
    }
}

/// ultra_fast_compute_metrics 的逐行复刻。
fn metrics_of(target: &Order, all: &[Order], same: &[u32], diff: &[u32]) -> [f64; NCOL] {
    let mut m = [0.0f64; NCOL];
    m[0] = target.id as f64;
    m[1] = target.typ as f64;

    let same_count = same.len();
    let diff_count = diff.len();
    if same_count == 0 && diff_count == 0 {
        return m;
    }

    let target_vol = target.volume as f64;
    let target_time = target.time;
    let inv_target_vol = if target_vol > 0.0 {
        1.0 / target_vol
    } else {
        0.0
    };

    let mut same_vol_sum = 0.0f64;
    let mut same_time_sum = 0.0f64;
    for &idx in same.iter() {
        let nb = &all[idx as usize];
        same_vol_sum += nb.volume as f64;
        same_time_sum += (nb.time - target_time).unsigned_abs() as f64;
    }
    let mut diff_vol_sum = 0.0f64;
    let mut diff_time_sum = 0.0f64;
    for &idx in diff.iter() {
        let nb = &all[idx as usize];
        diff_vol_sum += nb.volume as f64;
        diff_time_sum += (nb.time - target_time).unsigned_abs() as f64;
    }

    let same_count_f = same_count as f64;
    let diff_count_f = diff_count as f64;
    let total_vol = same_vol_sum + diff_vol_sum;
    let total_count = same_count_f + diff_count_f;

    if target_vol > 0.0 {
        m[2] = same_vol_sum * inv_target_vol;
        m[3] = diff_vol_sum * inv_target_vol;
        m[4] = if same_count > 0 {
            same_vol_sum / same_count_f * inv_target_vol
        } else {
            0.0
        };
        m[5] = if diff_count > 0 {
            diff_vol_sum / diff_count_f * inv_target_vol
        } else {
            0.0
        };
        m[6] = total_vol * inv_target_vol;
        m[7] = if total_count > 0.0 {
            total_vol / total_count * inv_target_vol
        } else {
            0.0
        };
    }

    m[8] = same_count_f;
    m[9] = diff_count_f;

    const NS_TO_SEC: f64 = 1e-9;
    m[10] = same_time_sum * NS_TO_SEC;
    m[11] = diff_time_sum * NS_TO_SEC;
    m[12] = if same_count > 0 {
        same_time_sum / same_count_f * NS_TO_SEC
    } else {
        0.0
    };
    m[13] = if diff_count > 0 {
        diff_time_sum / diff_count_f * NS_TO_SEC
    } else {
        0.0
    };

    if same.len() >= 2 {
        m[14] = fast_correlation(target, all, same, true);
        m[16] = fast_correlation(target, all, same, false);
    }
    if diff.len() >= 2 {
        m[15] = fast_correlation(target, all, diff, true);
        m[17] = fast_correlation(target, all, diff, false);
    }
    m
}

/// fast_correlation + calculate_correlation_inline 的逐行复刻（顺序累加，不是成对求和）。
fn fast_correlation(target: &Order, all: &[Order], idx: &[u32], use_time_diff: bool) -> f64 {
    let n = idx.len();
    if n < 2 {
        return 0.0;
    }
    let target_time = target.time;
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    for &i in idx {
        let nb = &all[i as usize];
        let xv = if use_time_diff {
            (nb.time - target_time).unsigned_abs() as f64
        } else {
            (nb.time as f64) / 1_000_000_000.0
        };
        x.push(xv);
        y.push(nb.volume as f64);
    }
    let nf = n as f64;
    let mut sum_x = 0.0f64;
    let mut sum_y = 0.0f64;
    for i in 0..n {
        sum_x += x[i];
        sum_y += y[i];
    }
    let mean_x = sum_x / nf;
    let mean_y = sum_y / nf;
    let mut cov = 0.0f64;
    let mut var_x = 0.0f64;
    let mut var_y = 0.0f64;
    for i in 0..n {
        let dx = x[i] - mean_x;
        let dy = y[i] - mean_y;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }
    let std_x = var_x.sqrt();
    let std_y = var_y.sqrt();
    if std_x < 1e-15 || std_y < 1e-15 {
        0.0
    } else {
        cov / (std_x * std_y)
    }
}

/// 每个订单的 18 个指标，按编号升序返回。
fn compute_metrics(orders: &[Order], param: f64) -> Vec<[f64; NCOL]> {
    (0..orders.len())
        .into_par_iter()
        .map(|ti| {
            let t = &orders[ti];
            let range = (((t.id as f64) * param / 100.0).abs() as i64).max(1);
            let (min_id, max_id) = (t.id - range, t.id + range);
            let s = binary_search_ge(orders, min_id);
            let e = binary_search_le(orders, max_id);
            let mut same: Vec<u32> = Vec::new();
            let mut diff: Vec<u32> = Vec::new();
            if s <= e && s < orders.len() {
                let hi = e.min(orders.len() - 1);
                let span = hi - s + 1;
                same.reserve(span / 4);
                diff.reserve(span / 4);
                for i in s..=hi {
                    let nb = &orders[i];
                    if nb.id == t.id {
                        continue;
                    }
                    if nb.typ == t.typ {
                        same.push(i as u32);
                    } else {
                        diff.push(i as u32);
                    }
                }
            }
            metrics_of(t, orders, &same, &diff)
        })
        .collect()
}

// ══════════════════════════════════════════════════════════════════════
// one(frame)：一个子集上的 279 个指标
// ══════════════════════════════════════════════════════════════════════

/// sel = None 用全部行，Some(0) 用订单类型==0（buy），Some(1) 用订单类型==1（sell）。
fn one(m: &[[f64; NCOL]], sel: Option<u8>) -> Vec<f64> {
    let idx: Vec<usize> = match sel {
        None => (0..m.len()).collect(),
        Some(t) => {
            let tv = t as f64;
            (0..m.len()).filter(|&i| m[i][1] == tv).collect()
        }
    };
    let n = idx.len();
    let mut cols: Vec<Vec<f64>> = Vec::with_capacity(NCOL);
    for c in 0..NCOL {
        cols.push(idx.iter().map(|&i| m[i][c]).collect());
    }
    let mut out: Vec<f64> = Vec::with_capacity(N_BASE);
    if n == 0 {
        // pandas 在空表上会在 df.max() 抛 ValueError，旧代码整体返回 None → 全 NaN。
        out.resize(N_BASE, f64::NAN);
        return out;
    }
    for &(a, b) in CORR_PAIRS.iter() {
        out.push(nan_corr(&cols[a], &cols[b]));
    }
    let means: Vec<f64> = (0..NCOL).map(|c| pw_sum(&cols[c]) / n as f64).collect();
    out.extend_from_slice(&means);
    for c in 0..NCOL {
        out.push(std_of(&cols[c], means[c], n) / means[c]);
    }
    for c in 0..NCOL {
        out.push(skew_of(&cols[c], n));
    }
    for c in 0..NCOL {
        out.push(kurt_of(&cols[c], n));
    }
    for c in 0..NCOL {
        let mut mx = f64::NEG_INFINITY;
        for &x in cols[c].iter() {
            if x > mx {
                mx = x;
            }
        }
        out.push(mx);
    }
    for c in 0..NCOL {
        out.push(autocorr_shift(&cols[c]));
    }
    for c in 0..NCOL {
        out.push(trend_of(&cols[c]));
    }
    debug_assert_eq!(out.len(), N_BASE);
    out
}

fn all_nan() -> Vec<f32> {
    vec![f32::NAN; N_TOTAL]
}

/// 逐股入口。
pub fn compute_stock(code: &str, date: i64) -> Option<Vec<f32>> {
    let rows = match read_day(code, date) {
        Some(r) => r,
        None => return Some(all_nan()),
    };
    // 旧代码：adjust_afternoon 之后行数 <= 30 直接给 837*2 个 NaN。
    if rows.len() <= 30 {
        return Some(all_nan());
    }
    let orders = build_orders(&rows);
    if orders.is_empty() {
        return Some(all_nan());
    }
    // buy / sell 任一为空时，pandas 会在空表的 df.max() 上抛异常，整天 NaN。
    let n_buy = orders.iter().filter(|o| o.typ == 0).count();
    let n_sell = orders.len() - n_buy;
    if n_buy == 0 || n_sell == 0 {
        return Some(all_nan());
    }

    let mut out: Vec<f32> = Vec::with_capacity(N_TOTAL);
    for &param in [5.0f64, 1.0f64].iter() {
        let m = compute_metrics(&orders, param);
        for sel in [None, Some(0u8), Some(1u8)] {
            out.extend(one(&m, sel).into_iter().map(|v| v as f32));
        }
    }
    debug_assert_eq!(out.len(), N_TOTAL);
    Some(out)
}

/// hm060 全是逐股因子，没有横截面版本。
pub fn compute_cross(_date: i64) -> Option<(Vec<String>, Vec<f32>)> {
    None
}

/// 1674 个因子名，顺序与 compute_stock 的输出严格对齐。
pub fn names() -> Vec<String> {
    let mut out = Vec::with_capacity(N_TOTAL);
    for tail in ["编号附近5perc", "编号附近1perc"] {
        for seg in ["all", "buy", "sell"] {
            for b in BASE_NAMES.iter() {
                out.push(format!("{}_{}_{}", b, seg, tail));
            }
        }
    }
    out
}
