//! hm008 3 的倍数（divisible_perseverant）—— 旧框架（design_whatever_out.TickDataBase）的 Rust 复现。
//!
//! ════════════════════════════════════════════════════════════════════════
//! 旧代码要点（逐条对齐 hm8_okay.py 的 run_base）
//! ════════════════════════════════════════════════════════════════════════
//!
//! 【框架参数】get_many=0 / read_full=0 / change_time=1，Go 继承 dwo.TickDataBase，
//!   get_factor 里只读**单只股票**：df = self.read_trade_data_single(with_retreat=0)。
//!   所以是逐股系列，Rust 侧走 compute_stock。
//!
//! 【读入】read_trade_data_single 用 pd.read_csv(usecols=[...]) 读原始 CSV，
//!   丢掉 flag == 32 的撤单行，再 deal_data(change_time=1) 把 exchtime 变成
//!   Timestamp = 1970-01-01 08:00:00 + exchtime 微秒（即北京时间）。
//!   Rust 侧用 read_trade_fast_inner(code, date, true, false, usize::MAX) 拿同样的行
//!   （time_us 就是「epoch 微秒 + 8 小时」），再自己按 flag != 32 过滤。
//!
//! 【时间窗】df.set_index('exchtime').between_time(lo, hi)，两端都含，
//!   比较的是「当天的钟点」，精确到微秒。基准窗 09:30:00~14:57:00，
//!   另有 open30 09:30~10:00、close30 14:30~14:57、morning 09:30~11:30、
//!   afternoon 13:00~14:57，都是从**已经筛过基准窗的** df 上再切。
//!
//! 【指标】每个窗内：
//!   asks = df.groupby('ask_order').volume.sum()
//!   bids = df.groupby('bid_order').volume.sum()
//!   volume_sum = df.volume.sum()
//!   对 d = 2..31：
//!     sum_d = (asks[asks % d == 0].sum() - bids[bids % d == 0].sum()) / volume_sum
//!     std_d = (asks[asks % d == 0].std() - bids[bids % d == 0].std()) / volume_sum
//!   关键：`asks` 是「按委托编号分组后的成交量」这个 Series，`asks % d` 是对
//!   **值**（也就是那一组的成交量之和）取模，不是对委托编号取模。
//!   std 是 pandas 默认的样本标准差（ddof=1），空集合给 NaN。
//!   旧代码还有 df.price*100、df.volume/100 两行，前者没被用到，后者是个常数
//!   比例（分子分母同时缩放），这里照做以求逐位一致。
//!
//! 【输出 300 个值】5 个窗 × (sum_2..sum_31, std_2..std_31) = 300，
//!   窗的顺序是 all、open30、close30、morning、afternoon。

use crate::fast_csv_reader::{read_trade_fast_inner, TradeRecord};
use std::collections::HashMap;

/// 5 个时间窗 × 30 个除数 × 2 个统计量
pub const N_OUT: usize = 300;

const DAY_US: i64 = 86_400_000_000;

/// 时间窗（当天的钟点，单位微秒，闭区间），顺序与旧代码的 res 拼接顺序一致
const US: i64 = 1_000_000;
const WINDOWS: [(&str, i64, i64); 5] = [
    ("all", (9 * 3600 + 30 * 60) * US, (14 * 3600 + 57 * 60) * US),
    ("open30", (9 * 3600 + 30 * 60) * US, 10 * 3600 * US),
    ("close30", (14 * 3600 + 30 * 60) * US, (14 * 3600 + 57 * 60) * US),
    ("morning", (9 * 3600 + 30 * 60) * US, (11 * 3600 + 30 * 60) * US),
    ("afternoon", 13 * 3600 * US, (14 * 3600 + 57 * 60) * US),
];

const DIVISORS: std::ops::RangeInclusive<i64> = 2..=31;

pub fn names() -> Vec<String> {
    let mut out = Vec::with_capacity(N_OUT);
    for (w, _, _) in WINDOWS.iter() {
        for pre in ["sum", "std"] {
            for d in DIVISORS {
                out.push(format!("{pre}_{d}_{w}"));
            }
        }
    }
    out
}

/// 一个窗内的聚合结果：窗内总成交量 + 按委托编号聚好的成交量
struct WinAgg {
    vol_sum: f64,
    asks: Vec<f64>,
    bids: Vec<f64>,
}

fn agg_window(rows: &[&TradeRecord], lo_us: i64, hi_us: i64) -> WinAgg {
    let mut asks: HashMap<i64, f64> = HashMap::new();
    let mut bids: HashMap<i64, f64> = HashMap::new();
    let mut vol_sum = 0.0f64;
    for r in rows.iter() {
        let tod = r.time_us.rem_euclid(DAY_US);
        if tod < lo_us || tod > hi_us {
            continue;
        }
        // 旧代码的 df.volume = df.volume / 100
        let v = r.volume as f64 / 100.0;
        vol_sum += v;
        *asks.entry(r.ask_order).or_insert(0.0) += v;
        *bids.entry(r.bid_order).or_insert(0.0) += v;
    }
    // 只需要值；顺序对 sum / std 都没有影响
    WinAgg {
        vol_sum,
        asks: asks.into_values().collect(),
        bids: bids.into_values().collect(),
    }
}

/// 分组成交量里「能被 d 整除」的那些值之和（空集合给 0，与 pandas 一致）
fn sel_sum(vals: &[f64], d: i64) -> f64 {
    let dd = d as f64;
    let mut s = 0.0;
    for &v in vals.iter() {
        if v % dd == 0.0 {
            s += v;
        }
    }
    s
}

/// 同上那些值的样本标准差（ddof=1）；不足 2 个给 NaN（pandas 的行为）
fn sel_std(vals: &[f64], d: i64) -> f64 {
    let dd = d as f64;
    let mut n = 0usize;
    let mut s = 0.0;
    for &v in vals.iter() {
        if v % dd == 0.0 {
            n += 1;
            s += v;
        }
    }
    if n < 2 {
        return f64::NAN;
    }
    let m = s / n as f64;
    let mut ss = 0.0;
    for &v in vals.iter() {
        if v % dd == 0.0 {
            let t = v - m;
            ss += t * t;
        }
    }
    (ss / (n as f64 - 1.0)).sqrt()
}

/// 逐股入口：算不出来（文件缺失）返回 None，上游补 NaN 保证因子个数恒定。
pub fn compute_stock(code: &str, date: i64) -> Option<Vec<f32>> {
    let trades = read_trade_fast_inner(code, date, true, false, usize::MAX).ok()?;
    let rows: Vec<&TradeRecord> = trades.iter().filter(|r| r.flag != 32).collect();

    let mut out: Vec<f64> = Vec::with_capacity(N_OUT);
    for (_w, lo, hi) in WINDOWS.iter() {
        let agg = agg_window(&rows, *lo, *hi);
        for d in DIVISORS {
            let num = sel_sum(&agg.asks, d) - sel_sum(&agg.bids, d);
            out.push(num / agg.vol_sum);
        }
        for d in DIVISORS {
            let num = sel_std(&agg.asks, d) - sel_std(&agg.bids, d);
            out.push(num / agg.vol_sum);
        }
    }
    debug_assert_eq!(out.len(), N_OUT);
    Some(out.into_iter().map(|v| v as f32).collect())
}

/// hm008 全是逐股因子，没有横截面版本。
pub fn compute_cross(_date: i64) -> Option<(Vec<String>, Vec<f32>)> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_layout() {
        let n = names();
        assert_eq!(n.len(), N_OUT);
        assert_eq!(n[0], "sum_2_all");
        assert_eq!(n[29], "sum_31_all");
        assert_eq!(n[30], "std_2_all");
        assert_eq!(n[59], "std_31_all");
        assert_eq!(n[60], "sum_2_open30");
        assert_eq!(n[120], "sum_2_close30");
        assert_eq!(n[180], "sum_2_morning");
        assert_eq!(n[240], "sum_2_afternoon");
        assert_eq!(n[299], "std_31_afternoon");
    }
}
