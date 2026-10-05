//! hm25「开盘系列综合」——旧框架（design_whatever_out.TickDataBase）的 Rust 复现。
//!
//! 旧文件：/home/chenzongwei/pythoncode/成果hm25_开盘系列综合/hm25_okay.py
//!   run_base 里的 Go.get_factor 自己调 read_trade_data_full_market(lazy=0) 与
//!   read_market_data_full_market(lazy=0)，一次读**一天全市场**两张表再自己切窗口，
//!   所以是横截面系列，只提供 compute_cross。
//!
//! ════════════════════════════════════════════════════════════════════
//! 旧代码逐步对照（写代码前先抄在这里，方便复核）
//! ════════════════════════════════════════════════════════════════════
//!
//! 【读取】trade / market 两张全市场表。read_trade_data_full_market 的 with_retreat
//!   取默认值 1，也就是**撤单行 flag==32 全部保留**（旧代码再自己按 flag 筛），
//!   本文件因此用 read_trade_fast_inner(code, date, true, false, usize::MAX) 读进来。
//!   market 没有涨跌停过滤，用 read_market_fast_inner(code, date, true, false, ...)。
//!   exchtime 在旧框架里是「epoch 微秒 + 8 小时」，与 Rust 的 TradeRecord.time_us 一致，
//!   当天钟点 = time_us.rem_euclid(86_400_000_000)。
//!
//! 【六个窗口】全部按真实钟点（旧代码没调 adjust_afternoon，比较是微秒级）：
//!   open_trade   : t <  09:30:00.000000
//!   open_market  : t <  09:30:00.000000
//!   close_trade  : t >  14:57:00.000000
//!   close_market : t >  14:57:00.000000
//!   inday_trade  : 09:30:00 <= t <= 14:57:00
//!   inday_market : 09:30:00 <= t <= 14:57:00
//! 另有 minute==25（09:25:xx）与 close 段里的 minute==0（15:00:xx）两个细切点。
//!
//! 【输出：69 个因子（原 72 个，摘掉了 four_* 那 3 个）】
//!   one  → 0..1    open/close 两段「撤单撮合的 ask/bid 成交量差」的 std、以及它的
//!                  按股名次的 std，两者都再做一次全市场名次差
//!   two  → 2       开盘集合竞价成交量 / (成交量+盘口挂单量) 的买卖两侧之差
//!   five → 3..5    open 段撤单量占该股 open 段总成交量的比例（买/卖/差）
//!   six  → 6..17   开盘集合竞价量占比 ocvp、尾盘集合竞价量占比 bcvp、两者之比 vr，
//!                  以及 9 个 w*ocvp+(1-w)*bcvp（w=0.1..0.9）
//!   seven→ 18..19  尾盘集合竞价收益率，以及它减掉「14:54 之前最后一笔」收益率
//!   eight→ 20..22  尾盘集合竞价每笔成交量取 log 之后的 std / 偏度 / 峰度
//!   nine → 23..40  尾盘集合竞价里 18 个「方向×金额档」的成交量占比之和
//!   ten  → 41..68  盘中 4 个截止时点 × 7 个金额档的成交均价比收盘价减 1
//!
//! ════════════════════════════════════════════════════════════════════
//! 为什么摘掉了 four_buy / four_sell / four_trend_volume_rank_abs
//! ════════════════════════════════════════════════════════════════════
//! 旧代码 four() 里有一句 `sort(["code", "exchtime"])`。polars 的**多列**排序不稳定：
//! 同一只股票内 exchtime 完全相同的并列行会被排成什么顺序，取决于整张全市场表
//! （20250303 是 142455 行 flag==32 的撤单）的拼接顺序，而拼接顺序来自 glob/readdir
//! 的任意顺序。而 four() 的 y 是 `pl.col("one").cum_count().over("code")`，也就是
//! 「排完之后落在第几个」，于是这个任意顺序直接进了相关系数。
//!
//! 对照实验（同一份数据、同一条链，只把文件拼接顺序从 sorted(glob) 换成 glob 原序）：
//!   000001 / 20250303 的 four_trend_volume_rank_abs：
//!     sorted(glob) → 0.05271937505636257
//!     glob 原序     → 0.05291089246490235   （这就是旧代码的真实取值）
//!   两次的行数（592）、归一化成交量之和（0.9999996423721313）、
//!   volume_rank 的最小/最大值（54713.5 / 136565.0）完全相同 ——
//!   差值全部来自 cum_count() 的名次落在哪一行，不是数据、也不是精度问题。
//! 也就是说这 3 个因子在旧代码里本来就不可复现，Rust 侧无法、也不该去复刻 readdir 顺序。
//! 它们不在 names_save 里、不写进因子库，摘掉不影响任何入库因子；
//! 这也让 names() 与 compute_cross 的返回长度严格自洽（都是 69）。
//! 副作用：再直接拿 lib/okay_database/verify_series.py 跑这个系列会整体错位 3 格
//! （旧代码 72 个、新代码 69 个），对账前要先把旧值里的这 3 个 four_* 去掉再做比较。
//!
//! 【哪些量是「全市场」量，不能逐股算】
//!   · one 第一步 `pl.col("ab_diff").rank()` 没有 over()，是对全市场那张表排名次；
//!     但第二步是 `.over("code")` 再取 std。对「全局名次」按 code 排名与直接对
//!     ab_diff 按 code 排名，并列结构完全一样，所以这一步可以化到逐股算。
//!   · one 最后一步 `df.ab_diff_std.rank()` 是 pandas 对**所有股票**排名次的差。
//!
//! 【精度取舍】
//!   · polars 对 Float32 的 sum / sum().over(...) 是**朴素 f32 顺序累加**，
//!     顺序 = 行序（transaction 文件本身按 index 升序、也就是 exchtime 升序），
//!     这里照做：所有成交量求和都用 f32 顺序累加，才能让 ab_diff、volume_ratio
//!     逐位对上（这些量后面要进 rank，差一个 ULP 就可能差半个名次）。
//!   · polars 的 std / skew / kurtosis 对 Float32 是**先用 f64 算再落到 f32**
//!     （实测 20/20 命中 f64 结果转 f32），skew 是总体偏度 m3/m2^1.5，
//!     kurtosis 默认 fisher=True 也就是超额峰度 m4/m2^2-3。
//!   · polars 的 rank() 返回 Float64、并列取平均、NaN 当最大值排在最后；
//!     pandas 的 Series.rank() 是 NaN 保持 NaN（不参与排名）。
//!   · (ask_volume-bid_volume)/(ask_volume+bid_volume) 是 f32 除法，照做。
//!   · `turnover` 为空时 polars 的比较结果是 null（整行被筛掉），
//!     这里用 NaN 判断模拟同一件事。

use std::collections::HashMap;

use rayon::prelude::*;

use crate::data_paths;
use crate::fast_csv_reader::{read_market_fast_inner, read_trade_fast_inner};

const SEC_US: i64 = 1_000_000;
const DAY_US: i64 = 86_400_000_000;

const T_0930: i64 = (9 * 3600 + 30 * 60) * SEC_US;
const T_1430: i64 = (14 * 3600 + 30 * 60) * SEC_US;
const T_1440: i64 = (14 * 3600 + 40 * 60) * SEC_US;
const T_1450: i64 = (14 * 3600 + 50 * 60) * SEC_US;
const T_1454: i64 = (14 * 3600 + 54 * 60) * SEC_US;
const T_1455: i64 = (14 * 3600 + 55 * 60) * SEC_US;
const T_1457: i64 = (14 * 3600 + 57 * 60) * SEC_US;

pub const N_OUT: usize = 69;

/// ten 的 4 个截止时点（旧代码里的 14:55 / 14:50 / 14:40 / 14:30）。
const TEN_CUTS: [i64; 4] = [T_1455, T_1450, T_1440, T_1430];
/// 金额档：每档给 (下界, 上界)，两端都是**开区间**（旧代码一律用 < 和 >）。
/// None 表示该侧不设界。最后一个档是「不设界」也就是 flag==66 的全部。
type Band = (Option<f64>, Option<f64>);
const NINE_BANDS: [Band; 6] = [
    (None, Some(4e4)),
    (Some(4e4), Some(20e4)),
    (None, Some(20e4)),
    (Some(20e4), Some(100e4)),
    (Some(100e4), None),
    (Some(20e4), None),
];

const FIVE_NAMES: [&str; 3] = [
    "five_buy_retreat_rate",
    "five_sell_retreat_rate",
    "five_retreat_diff",
];
const NINE_BAND_TAGS: [&str; 6] = [
    "small",
    "med",
    "relative_small",
    "large",
    "exlarge",
    "relative_large",
];
const NINE_DIR_TAGS: [&str; 3] = ["buy", "sell", "both"];
const TEN_CUT_TAGS: [&str; 4] = ["2", "7", "17", "27"];
const TEN_BAND_TAGS: [&str; 7] = [
    "small",
    "med",
    "relative_small",
    "large",
    "exlarge",
    "relative_large",
    "four",
];

pub fn names() -> Vec<String> {
    let mut out: Vec<String> = Vec::with_capacity(N_OUT);
    out.push("one_diff_std".to_string());
    out.push("one_diff_rank_std".to_string());
    out.push("two_buy_sell_diff".to_string());
    out.extend(FIVE_NAMES.iter().map(|s| s.to_string()));
    out.push("six_ocvp".to_string());
    out.push("six_bcvp".to_string());
    out.push("six_vr".to_string());
    for k in 1..=9 {
        out.push(format!("six_obcvp{}", k));
    }
    out.push("seven_auc_return".to_string());
    out.push("seven_auc_return_diff".to_string());
    out.push("eight_std".to_string());
    out.push("eight_skew".to_string());
    out.push("eight_kurt".to_string());
    for dir in NINE_DIR_TAGS.iter() {
        for band in NINE_BAND_TAGS.iter() {
            out.push(format!("nine_{}_{}", band, dir));
        }
    }
    for cut in TEN_CUT_TAGS.iter() {
        for band in TEN_BAND_TAGS.iter() {
            out.push(format!("ten_{}{}", band, cut));
        }
    }
    out
}

/// 每只股票当天聚合出来的全部中间量。
struct Agg {
    /// one（open 段）：09:25 段、flag!=32 的逐行 ab_diff（文件序）。
    one_ab_open: Vec<f32>,
    /// one（close 段）：15:00 段、flag!=32 的逐行 ab_diff（文件序）。
    one_ab_close: Vec<f32>,
    /// two：09:25 段 flag!=32 的成交量之和。
    two_trade: Option<f32>,
    /// two：开盘集合竞价最后一笔快照的 (total_ask_vol, total_bid_vol)。
    mkt_open: Option<(f32, f32)>,
    /// five：open 段撤单买量、撤单卖量、open 段全部成交量。
    five_buy: f32,
    five_sell: f32,
    five_total: f32,
    /// five：该股在 open 段、撤单买、撤单卖 三张表里**有没有出现过**。
    /// 旧代码是 pandas 的 Series 相除，只在某一张表里出现的股票对齐后是 NaN，
    /// 不能把「没有撤单买」当成「撤单买量 = 0」。
    five_has_total: bool,
    five_has_buy: bool,
    five_has_sell: bool,
    /// six：inday 段第一个有效 volume、inday 段最后一个 volume、close 段最后一个有效 volume。
    six_auc_open: Option<f32>,
    six_inday_last: Option<f32>,
    six_ttl: Option<f32>,
    /// seven：close 段最后一个有效非零 last_prc、inday 段同、inday 段 14:54 之前同。
    seven_r_close: Option<f32>,
    seven_r_inday: Option<f32>,
    seven_r_1454: Option<f32>,
    /// eight：close 段每笔成交量的自然对数（先算成 f32，与 polars log 的落点一致）。
    eight_log: Vec<f32>,
    /// nine：close 段总成交量（f32 顺序和）、18 个子集的 volume_ratio 之和与行数。
    nine_total: f32,
    nine_sum: [f32; 18],
    nine_cnt: [u32; 18],
    /// ten：4 个时点 × 7 个档的 price 之和与有效行数。
    /// ！！和必须是 f32 顺序累加：实测 polars 的 `group_by(...).agg(mean())` 对 Float32
    /// 就是「f32 顺序求和 / f32 除法」，和 `Series.mean()`（成对求和）不是一回事，
    /// 一万多行时两者能差 1e-5 相对量级，落在验收阈值之内但会让整块 ten 因子偏掉。
    ten_sum: [f32; 28],
    ten_cnt: [u32; 28],
    /// ten：close 段最后一行的 price（不筛有效性）。
    ten_close: Option<f32>,
}

impl Agg {
    fn new() -> Self {
        Agg {
            one_ab_open: Vec::new(),
            one_ab_close: Vec::new(),
            two_trade: None,
            mkt_open: None,
            five_buy: 0.0,
            five_sell: 0.0,
            five_total: 0.0,
            five_has_total: false,
            five_has_buy: false,
            five_has_sell: false,
            six_auc_open: None,
            six_inday_last: None,
            six_ttl: None,
            seven_r_close: None,
            seven_r_inday: None,
            seven_r_1454: None,
            eight_log: Vec::new(),
            nine_total: 0.0,
            nine_sum: [0.0; 18],
            nine_cnt: [0; 18],
            ten_sum: [0.0; 28],
            ten_cnt: [0; 28],
            ten_close: None,
        }
    }
}

#[inline]
fn minute_of(t: i64) -> i64 {
    (t / (60 * SEC_US)) % 60
}

/// 金额档判断（旧代码用 polars 的 null 传播：turnover 为空时整条为 false）。
#[inline]
fn in_band(turnover: f32, band: &Band) -> bool {
    if turnover.is_nan() {
        return false;
    }
    let v = turnover as f64;
    if let Some(lo) = band.0 {
        if !(v > lo) {
            return false;
        }
    }
    if let Some(hi) = band.1 {
        if !(v < hi) {
            return false;
        }
    }
    true
}

/// 把一段窗口里的 ab_diff 逐行算出来（按 (ask_order, code) / (bid_order, code) 各自求和）。
///
/// 三件事必须照旧代码做：撤单行已经被剔除、按文件序做 f32 顺序累加、f32 除法。
fn collect_ab(trades: &[crate::fast_csv_reader::TradeRecord], t_lo: i64, t_hi: i64, minute: i64, above: bool) -> Vec<f32> {
    let mut ask_sum: HashMap<i64, f32> = HashMap::new();
    let mut bid_sum: HashMap<i64, f32> = HashMap::new();
    let mut rows: Vec<(i64, i64)> = Vec::new();
    for r in trades.iter() {
        let t = r.time_us.rem_euclid(DAY_US);
        let in_win = if above { t > t_hi } else { t >= t_lo && t <= t_hi };
        if !in_win || r.flag == 32 || minute_of(t) != minute {
            continue;
        }
        *ask_sum.entry(r.ask_order).or_insert(0.0) += r.volume;
        *bid_sum.entry(r.bid_order).or_insert(0.0) += r.volume;
        rows.push((r.ask_order, r.bid_order));
    }
    rows.iter()
        .map(|(a, b)| {
            let sa = ask_sum[a];
            let sb = bid_sum[b];
            (sa - sb) / (sa + sb)
        })
        .collect()
}

/// 读一只股票当天的逐笔与盘口，聚合成 [`Agg`]。
///
/// 逐笔文件读不到时交易类因子全 NaN、盘口类因子照算；两边都读不到才返回 None。
fn aggregate_stock(code: &str, date: i64) -> Option<Agg> {
    let mut agg = Agg::new();
    let mut any = false;

    if let Ok(trades) = read_trade_fast_inner(code, date, true, false, usize::MAX) {
        any = true;

        // one：open 段（t<09:30 且 minute==25）与 close 段（t>14:57 且 minute==0）
        agg.one_ab_open = collect_ab(&trades, 0, T_0930, 25, false);
        agg.one_ab_close = collect_ab(&trades, 0, T_1457, 0, true);

        // 其余逐笔侧的量：一遍扫完，全部按文件序做 f32 顺序累加。
        for r in trades.iter() {
            let t = r.time_us.rem_euclid(DAY_US);
            if t < T_0930 {
                if r.flag != 32 && minute_of(t) == 25 {
                    *agg.two_trade.get_or_insert(0.0) += r.volume;
                }
                agg.five_total += r.volume;
                agg.five_has_total = true;
                if r.flag == 32 {
                    if r.ask_order == 0 {
                        agg.five_buy += r.volume;
                        agg.five_has_buy = true;
                    }
                    if r.bid_order == 0 {
                        agg.five_sell += r.volume;
                        agg.five_has_sell = true;
                    }
                }
            } else if t > T_1457 {
                agg.eight_log.push(r.volume.ln());
                agg.nine_total += r.volume;
                agg.ten_close = Some(r.price);
            } else if r.flag == 66 {
                // ten：盘中 flag==66 的成交均价，4 个时点 × 7 个档
                for w in 0..4 {
                    for b in 0..7 {
                        // ！旧代码里 four2 那一档的截止时点是 14:50，和同一组其它 6 档的
                        // 14:55 不一样（原文件就是这么写的）；four7/four17/four27 与同组一致。
                        let cut = if w == 0 && b == 6 { T_1450 } else { TEN_CUTS[w] };
                        if t >= cut {
                            continue;
                        }
                        if b < 6 && !in_band(r.turnover, &NINE_BANDS[b]) {
                            continue;
                        }
                        let idx = w * 7 + b;
                        if !r.price.is_nan() {
                            agg.ten_sum[idx] += r.price;
                            agg.ten_cnt[idx] += 1;
                        }
                    }
                }
            }
        }

        // nine 的第二遍：分母（当天 close 段总成交量）要先知道。顺序仍是文件序。
        if agg.nine_total != 0.0 {
            for r in trades.iter() {
                let t = r.time_us.rem_euclid(DAY_US);
                if t <= T_1457 {
                    continue;
                }
                let ratio = r.volume / agg.nine_total;
                let dir: Option<usize> = match r.flag {
                    66 => Some(0),
                    83 => Some(1),
                    _ => None,
                };
                if let Some(d) = dir {
                    for (b, band) in NINE_BANDS.iter().enumerate() {
                        if in_band(r.turnover, band) {
                            agg.nine_sum[d * 6 + b] += ratio;
                            agg.nine_cnt[d * 6 + b] += 1;
                        }
                    }
                }
                if r.flag == 66 || r.flag == 83 {
                    for (b, band) in NINE_BANDS.iter().enumerate() {
                        if in_band(r.turnover, band) {
                            agg.nine_sum[12 + b] += ratio;
                            agg.nine_cnt[12 + b] += 1;
                        }
                    }
                }
            }
        }
    }

    if let Ok(recs) = read_market_fast_inner(code, date, true, false, usize::MAX) {
        any = true;
        let mut open_time: Option<i64> = None;
        for r in recs.iter() {
            let t = r.time_us.rem_euclid(DAY_US);
            if t < T_0930 {
                // two：按 exchtime 取最后一笔；并列时取文件里靠后的那笔
                //（与旧代码 sort("exchtime") + unique(keep="last") 同序）
                if minute_of(t) < 30 && open_time.map_or(true, |bt| r.time_us >= bt) {
                    open_time = Some(r.time_us);
                    agg.mkt_open = Some((r.total_ask_vol, r.total_bid_vol));
                }
            } else if t <= T_1457 {
                if agg.six_auc_open.is_none() && !r.volume.is_nan() {
                    agg.six_auc_open = Some(r.volume);
                }
                agg.six_inday_last = Some(r.volume);
                if !r.last_prc.is_nan() && r.last_prc != 0.0 {
                    agg.seven_r_inday = Some(r.last_prc);
                    if t < T_1454 {
                        agg.seven_r_1454 = Some(r.last_prc);
                    }
                }
            } else {
                if !r.volume.is_nan() {
                    agg.six_ttl = Some(r.volume);
                }
                if !r.last_prc.is_nan() && r.last_prc != 0.0 {
                    agg.seven_r_close = Some(r.last_prc);
                }
            }
        }
    }

    if any {
        Some(agg)
    } else {
        None
    }
}

// ────────────────────── 统计工具（逐条对齐 pandas / polars） ──────────────────────

/// 样本标准差 ddof=1；n<2 或含 NaN 时为 NaN。
fn std_ddof1(x: &[f64]) -> f64 {
    let n = x.len();
    if n < 2 || x.iter().any(|v| v.is_nan()) {
        return f64::NAN;
    }
    let m = x.iter().sum::<f64>() / n as f64;
    let mut s = 0.0f64;
    for &v in x {
        let d = v - m;
        s += d * d;
    }
    (s / (n as f64 - 1.0)).sqrt()
}

/// 总体偏度 m3/m2^1.5（polars skew 默认 bias=True）。
fn skew_bias(x: &[f64]) -> f64 {
    let n = x.len();
    if n == 0 || x.iter().any(|v| v.is_nan()) {
        return f64::NAN;
    }
    let nf = n as f64;
    let m = x.iter().sum::<f64>() / nf;
    let mut m2 = 0.0f64;
    let mut m3 = 0.0f64;
    for &v in x {
        let d = v - m;
        m2 += d * d;
        m3 += d * d * d;
    }
    m2 /= nf;
    m3 /= nf;
    if !(m2 > 0.0) {
        return f64::NAN;
    }
    m3 / (m2.sqrt() * m2)
}

/// 超额峰度 m4/m2^2-3（polars kurtosis 默认 fisher=True, bias=True）。
fn kurt_excess(x: &[f64]) -> f64 {
    let n = x.len();
    if n == 0 || x.iter().any(|v| v.is_nan()) {
        return f64::NAN;
    }
    let nf = n as f64;
    let m = x.iter().sum::<f64>() / nf;
    let mut m2 = 0.0f64;
    let mut m4 = 0.0f64;
    for &v in x {
        let d = v - m;
        let d2 = d * d;
        m2 += d2;
        m4 += d2 * d2;
    }
    m2 /= nf;
    m4 /= nf;
    if !(m2 > 0.0) {
        return f64::NAN;
    }
    m4 / (m2 * m2) - 3.0
}

/// polars rank(method='average')：升序、并列取平均、NaN 当最大值排在最后。
fn rank_average_polars(values: &[f64]) -> Vec<f64> {
    let n = values.len();
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| {
        let (va, vb) = (values[a], values[b]);
        match (va.is_nan(), vb.is_nan()) {
            (true, true) => std::cmp::Ordering::Equal,
            (true, false) => std::cmp::Ordering::Greater,
            (false, true) => std::cmp::Ordering::Less,
            _ => va.partial_cmp(&vb).unwrap_or(std::cmp::Ordering::Equal),
        }
    });
    let mut out = vec![f64::NAN; n];
    let mut i = 0usize;
    while i < n {
        let mut j = i + 1;
        while j < n {
            let (a, b) = (values[idx[j]], values[idx[i]]);
            let same = (a.is_nan() && b.is_nan()) || (!a.is_nan() && !b.is_nan() && a == b);
            if !same {
                break;
            }
            j += 1;
        }
        let avg = ((i + 1) as f64 + j as f64) / 2.0;
        for k in i..j {
            out[idx[k]] = avg;
        }
        i = j;
    }
    out
}

/// pandas Series.rank()：升序、并列取平均、NaN 保持 NaN（不参与排名）。
fn rank_average_pandas(values: &[f64]) -> Vec<f64> {
    let n = values.len();
    let mut idx: Vec<usize> = (0..n).filter(|&i| !values[i].is_nan()).collect();
    idx.sort_by(|&a, &b| {
        values[a]
            .partial_cmp(&values[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut out = vec![f64::NAN; n];
    let mut i = 0usize;
    while i < idx.len() {
        let mut j = i + 1;
        while j < idx.len() && values[idx[j]] == values[idx[i]] {
            j += 1;
        }
        let avg = ((i + 1) as f64 + j as f64) / 2.0;
        for k in i..j {
            out[idx[k]] = avg;
        }
        i = j;
    }
    out
}

// ────────────────────────────── 横截面主流程 ──────────────────────────────

/// 横截面入口：一次算一天全市场，返回 (股票代码, 扁平因子值)。
///
/// 值按股票分块排列（每只股票连续 69 个），与 `dispatch_cross` 的
/// `vals.chunks(expected_len).zip(codes)` 对齐。
pub fn compute_cross(date: i64) -> Option<(Vec<String>, Vec<f32>)> {
    let codes = data_paths::list_stock_codes(date, "transaction");
    if codes.is_empty() {
        return None;
    }
    let n = codes.len();
    let aggs: Vec<Option<Agg>> = codes
        .par_iter()
        .map(|code| aggregate_stock(code, date))
        .collect();

    let mut out = vec![f32::NAN; n * N_OUT];

    // ── one ──
    // 旧代码里 `pl.col("ab_diff").rank().alias("ab_diff_rank")` **没有 over()**，
    // 是全市场名次；随后 `group_by("code").agg(std(ab_diff_rank))` 取的是这只股票
    // 那几行的「全市场名次」的 std。所以必须先把两段的名次在全市场排出来。
    // （`ab_diff_rank_code` 那一列旧代码算完之后根本没进 agg，是死列。）
    let mut o_std = vec![f64::NAN; n];
    let mut o_rankstd = vec![f64::NAN; n];
    let mut c_std = vec![f64::NAN; n];
    let mut c_rankstd = vec![f64::NAN; n];
    {
        // open 段
        let mut flat: Vec<f64> = Vec::new();
        let mut offs = vec![0usize; n + 1];
        for i in 0..n {
            if let Some(a) = aggs[i].as_ref() {
                for &v in a.one_ab_open.iter() {
                    flat.push(v as f64);
                }
            }
            offs[i + 1] = flat.len();
        }
        let ranks = rank_average_polars(&flat);
        for i in 0..n {
            let (lo, hi) = (offs[i], offs[i + 1]);
            if hi > lo {
                // polars 对 Float32 求 std 是 f64 算完再落回 f32，这里照做
                o_std[i] = std_ddof1(&flat[lo..hi]) as f32 as f64;
                o_rankstd[i] = std_ddof1(&ranks[lo..hi]);
            }
        }
    }
    {
        // close 段
        let mut flat: Vec<f64> = Vec::new();
        let mut offs = vec![0usize; n + 1];
        for i in 0..n {
            if let Some(a) = aggs[i].as_ref() {
                for &v in a.one_ab_close.iter() {
                    flat.push(v as f64);
                }
            }
            offs[i + 1] = flat.len();
        }
        let ranks = rank_average_polars(&flat);
        for i in 0..n {
            let (lo, hi) = (offs[i], offs[i + 1]);
            if hi > lo {
                c_std[i] = std_ddof1(&flat[lo..hi]) as f32 as f64;
                c_rankstd[i] = std_ddof1(&ranks[lo..hi]);
            }
        }
    }
    let r_o_std = rank_average_pandas(&o_std);
    let r_c_std = rank_average_pandas(&c_std);
    let r_o_rankstd = rank_average_pandas(&o_rankstd);
    let r_c_rankstd = rank_average_pandas(&c_rankstd);

    for i in 0..n {
        let base = i * N_OUT;
        out[base] = (r_o_std[i] - r_c_std[i]) as f32;
        out[base + 1] = (r_o_rankstd[i] - r_c_rankstd[i]) as f32;
        let Some(a) = aggs[i].as_ref() else { continue };

        // ── two ──
        if let (Some(tv), Some((ta, tb))) = (a.two_trade, a.mkt_open) {
            out[base + 2] = tv / (tv + tb) - tv / (tv + ta);
        }

        // ── five ──
        // 旧代码是 pandas Series 对齐相除：任一张表里没有这只股票，结果就是 NaN；
        // 分母为 0 时 pandas 先 replace 成 NaN，所以也是 NaN（不是 inf）。
        {
            let ok = |has_num: bool| a.five_has_total && has_num && a.five_total != 0.0;
            let rb = if ok(a.five_has_buy) {
                a.five_buy / a.five_total
            } else {
                f32::NAN
            };
            let rs = if ok(a.five_has_sell) {
                a.five_sell / a.five_total
            } else {
                f32::NAN
            };
            out[base + 3] = rb;
            out[base + 4] = rs;
            out[base + 5] = rb - rs;
        }

        // ── six ──
        {
            let div = |num: Option<f32>, den: Option<f32>| -> f32 {
                match (num, den) {
                    (Some(x), Some(d)) => {
                        if d == 0.0 {
                            f32::NAN
                        } else {
                            x / d
                        }
                    }
                    _ => f32::NAN,
                }
            };
            let auc_close = match (a.six_ttl, a.six_inday_last) {
                (Some(t), Some(v)) => Some(t - v),
                _ => None,
            };
            let ocvp = div(a.six_auc_open, a.six_ttl);
            let bcvp = div(auc_close, a.six_ttl);
            let vr = if bcvp == 0.0 { f32::NAN } else { ocvp / bcvp };
            out[base + 6] = ocvp;
            out[base + 7] = bcvp;
            out[base + 8] = vr;
            for k in 0..9 {
                let w = 0.1f32 * (k as f32 + 1.0);
                out[base + 9 + k] = w * ocvp + (1.0 - w) * bcvp;
            }
        }

        // ── seven ──
        {
            let div1 = |num: f32, den: Option<f32>| -> f32 {
                match den {
                    Some(d) => {
                        if d == 0.0 {
                            f32::NAN
                        } else {
                            num / d - 1.0
                        }
                    }
                    None => f32::NAN,
                }
            };
            if let Some(r1) = a.seven_r_close {
                let ret = div1(r1, a.seven_r_inday);
                let before = match a.seven_r_1454 {
                    Some(b1) => div1(b1, a.seven_r_inday),
                    None => f32::NAN,
                };
                out[base + 18] = ret;
                out[base + 19] = ret - before;
            }
        }

        // ── eight ──
        if !a.eight_log.is_empty() {
            let lv: Vec<f64> = a.eight_log.iter().map(|&v| v as f64).collect();
            out[base + 20] = std_ddof1(&lv) as f32;
            out[base + 21] = skew_bias(&lv) as f32;
            out[base + 22] = kurt_excess(&lv) as f32;
        }

        // ── nine ──
        for k in 0..18 {
            if a.nine_cnt[k] > 0 {
                out[base + 23 + k] = a.nine_sum[k];
            }
        }

        // ── ten ──
        if let Some(close) = a.ten_close {
            if close != 0.0 {
                for k in 0..28 {
                    if a.ten_cnt[k] == 0 {
                        continue;
                    }
                    let mean = a.ten_sum[k] / a.ten_cnt[k] as f32;
                    out[base + 41 + k] = mean / close - 1.0;
                }
            }
        }
    }

    debug_assert_eq!(out.len(), n * N_OUT);
    Some((codes, out))
}

/// hm25 是横截面系列，没有逐股实现。
pub fn compute_stock(_code: &str, _date: i64) -> Option<Vec<f32>> {
    None
}
