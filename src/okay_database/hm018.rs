//! hm18 海日生残夜 —— 旧框架（design_whatever_out.TickDataBase）的 Rust 复现。
//!
//! 旧代码：/home/chenzongwei/pythoncode/成果hm18_海日生残夜/hm18_okay.py
//!
//! 一、旧代码的调用参数（run_base 里 run_factor_out 的实参）
//!     get_many=0、read_full=0、change_time=1
//!     → read_trade_data_single / read_market_data_single 都返回**单个** DataFrame，
//!       由旧代码自己切窗口；单股单日只读当天（look_back_window 默认 1）。
//!
//! 二、change_time=1 的含义（deal_data）
//!     只做一件事：把 exchtime（微秒整数）变成 Timestamp
//!         = exchtime/1e6 秒 + 1970-01-01 08:00:00，也就是北京时间。
//!     它**不做** adjust_afternoon 的下午时段平移。所以本文件读数据时
//!     with_afternoon_adjust=false，窗口按真实钟点的**微秒闭区间**筛
//!     （pandas 的 between_time 精确到纳秒且两端都含，不能按整秒截断）。
//!
//! 三、数据准备（旧代码 get_factor 开头）
//!     df     = read_market_data_single()            盘口快照，10 档量价
//!     trades = read_trade_data_single(with_retreat=0)  丢掉 flag==32 的撤单行
//!     df = df[df.last_prc != 0]                     丢掉没有最新价的快照
//!     两边都 set_index("exchtime").between_time("09:30","14:57")
//!     ！！撤单不能靠 Rust 读取侧的 with_retreat：库里那条判断按字节比 "32"，
//!        2025 年前后成交明细里撤单标记写的是 "32.0"，过滤不生效。
//!        所以这里用 with_retreat=true 读进来，再自己 filter(|r| r.flag != 32)。
//!
//! 四、五个时段（真实钟点，两端都含，微秒精度）
//!     all       [09:30:00, 14:57:00]
//!     open30    [09:30:00, 10:00:00]
//!     morning   [09:30:00, 11:30:00]
//!     afternoon [13:00:00, 14:57:00]
//!     close30   [14:30:00, 14:57:00]
//!
//! 五、每个时段 2 个指标（旧代码的 one()）
//!     price_close = 该时段成交里**出现次数最多**的成交价
//!                   （groupby('price').turnover.count() 再 sort_values().index[-1]）
//!     ask = skew(该时段所有 ask_prc{i} == price_close 的 ask_vol{i})
//!     bid = skew(该时段所有 bid_prc{i} == price_close 的 bid_vol{i})
//!     该时段的盘口行数或成交行数为 0 时，两个都是 NaN。
//!
//! 六、names_single 顺序（本文件 names() 必须逐个对上）
//!     ask_all, bid_all, ask_open30, bid_open30, ask_morning, bid_morning,
//!     ask_afternoon, bid_afternoon, ask_close30, bid_close30      共 10 个
//!     names_save（base 层要保存的）= ['ask_all', 'bid_open30']
//!     gap5 入选 = bid_open30_mean_smooth_5
//!     gap1 入选 = bid_open30_min_smooth_20, ask_all_min_smooth_20
//!
//! 七、一处已知的近似（已实测，写在这里免得以后重复排查）
//!     price_close 出现并列（两个价位成交笔数一样多）时，旧代码走的是
//!     numpy 的 argsort(kind='quicksort')，numpy 2.x 对 float64 数组用 SIMD 版本，
//!     并列顺序没有可复现的规则，任何确定性写法都不可能 100% 对上。
//!     本文件的选择是「并列时取价格最大的那个」。抽样 465 个时段的实测：
//!     7 个时段出现并列，其中 6 个 pandas 取到的正是最大价，1 个取到最小价。

use crate::fast_csv_reader::{
    read_market_fast_inner, read_trade_fast_inner, MarketRecord, TradeRecord,
};

/// 1 秒 = 1e6 微秒（窗口边界一律用微秒）
const US: i64 = 1_000_000;

/// 五个时段：(名字, 起始微秒, 结束微秒)，两端都含
const SESSIONS: [(&str, i64, i64); 5] = [
    ("all", (9 * 3600 + 30 * 60) * US, (14 * 3600 + 57 * 60) * US),
    ("open30", (9 * 3600 + 30 * 60) * US, 10 * 3600 * US),
    ("morning", (9 * 3600 + 30 * 60) * US, (11 * 3600 + 30 * 60) * US),
    ("afternoon", 13 * 3600 * US, (14 * 3600 + 57 * 60) * US),
    ("close30", (14 * 3600 + 30 * 60) * US, (14 * 3600 + 57 * 60) * US),
];

/// 每个时段 2 个指标（ask、bid），五个时段共 10 个
const PER_SESSION: usize = 2;
pub const N_OUT: usize = PER_SESSION * 5;

pub fn names() -> Vec<String> {
    let mut out = Vec::with_capacity(N_OUT);
    for (sess, _, _) in SESSIONS.iter() {
        out.push(format!("ask_{}", sess));
        out.push(format!("bid_{}", sess));
    }
    out
}

/// 北京时间的当日微秒数（time_us 已含 8 小时偏移）。
#[inline]
fn day_us(time_us: i64) -> i64 {
    time_us.rem_euclid(86_400_000_000)
}

/// 旧代码的 price_close：窗口中成交价里出现次数最多的那个价位。
///
/// 对应 pandas 的 `trades.groupby('price').turnover.count().sort_values().index[-1]`：
///   - groupby 默认丢 NaN 键（这里跳过非有限值）
///   - groupby(sort=True) 让价位按升序排列，所以升序扫描时「后来者优先」＝并列取最大价
/// 没有有效成交价时返回 None。
fn mode_price(prices: &[f32]) -> Option<f32> {
    let mut v: Vec<f32> = prices.iter().cloned().filter(|p| p.is_finite()).collect();
    if v.is_empty() {
        return None;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mut best_val = v[0];
    let mut best_cnt = 0usize;
    let mut i = 0usize;
    while i < v.len() {
        let mut j = i + 1;
        while j < v.len() && v[j] == v[i] {
            j += 1;
        }
        // >= 而不是 >：并列时保留价格更大的那个（升序遍历，后来者赢）
        if j - i >= best_cnt {
            best_cnt = j - i;
            best_val = v[i];
        }
        i = j;
    }
    Some(best_val)
}

/// pandas 的 Series.skew()（默认 bias=False，也就是 scipy 的修正 Fisher-Pearson）：
///     n < 3 或标准差为 0 时返回 NaN
///     g1 = (1/n) * Σ ((x-μ)/σ_pop)^3
///     结果 = g1 * sqrt(n(n-1)) / (n-2)
fn skew(values: &[f64]) -> f64 {
    let v: Vec<f64> = values.iter().cloned().filter(|x| x.is_finite()).collect();
    let n = v.len();
    if n < 3 {
        return f64::NAN;
    }
    let nf = n as f64;
    let mean = v.iter().sum::<f64>() / nf;
    let devs: Vec<f64> = v.iter().map(|x| x - mean).collect();
    let var = devs.iter().map(|d| d * d).sum::<f64>() / nf;
    let sd = var.sqrt();
    // pandas 是 dev/sd 再立方，sd==0 时 0/0=nan，结果仍是 NaN
    if !(sd > 0.0) {
        return f64::NAN;
    }
    let m3 = devs.iter().map(|d| (d / sd).powi(3)).sum::<f64>() / nf;
    m3 * (nf * (nf - 1.0)).sqrt() / (nf - 2.0)
}

/// 一个时段的 [ask, bid] 两个偏度，顺序与 names() 一致。
fn session_metrics(
    trades: &[&TradeRecord],
    mkts: &[&MarketRecord],
    lo_us: i64,
    hi_us: i64,
) -> [f64; PER_SESSION] {
    let window = lo_us..=hi_us;
    let prices: Vec<f32> = trades
        .iter()
        .filter(|r| window.contains(&day_us(r.time_us)))
        .map(|r| r.price)
        .collect();
    // 旧代码要求该时段的盘口行数（已去掉 last_prc==0）与成交行数都大于 0
    let n_mkt = mkts
        .iter()
        .filter(|m| window.contains(&day_us(m.time_us)) && m.last_prc != 0.0)
        .count();
    if n_mkt == 0 || prices.is_empty() {
        return [f64::NAN; PER_SESSION];
    }
    let price_close = match mode_price(&prices) {
        Some(p) => p,
        None => return [f64::NAN; PER_SESSION],
    };
    let mut asks: Vec<f64> = Vec::new();
    let mut bids: Vec<f64> = Vec::new();
    for m in mkts.iter() {
        if !window.contains(&day_us(m.time_us)) || m.last_prc == 0.0 {
            continue;
        }
        // 旧代码把 10 档揉成长表后按 price==price_close 过滤再取 vol，
        // 偏度与顺序无关，所以这里只要把命中的量收集齐即可。
        for i in 0..10 {
            if m.ask_prcs[i] == price_close {
                asks.push(m.ask_vols[i] as f64);
            }
            if m.bid_prcs[i] == price_close {
                bids.push(m.bid_vols[i] as f64);
            }
        }
    }
    [skew(&asks), skew(&bids)]
}

/// 逐股入口：读不到文件返回 None（上游会补 NaN，保证因子个数恒定）。
pub fn compute_stock(code: &str, date: i64) -> Option<Vec<f32>> {
    // with_afternoon_adjust=false：旧代码按真实钟点切窗口，不做下午平移。
    // with_retreat=true 只是先把撤单读进来，过滤在下面自己做（见文件头第三条）。
    let trades = read_trade_fast_inner(code, date, true, false, usize::MAX).ok()?;
    // with_high_low_limited=true：保留涨跌停快照。旧代码只丢 last_prc==0 的行，
    // 涨跌停那些行里另一侧的挂单是可能命中 price_close 的，不能提前过滤掉。
    let mkts = read_market_fast_inner(code, date, true, false, usize::MAX).ok()?;

    let trows: Vec<&TradeRecord> = trades.iter().filter(|r| r.flag != 32).collect();
    let mrows: Vec<&MarketRecord> = mkts.iter().collect();

    let mut out = Vec::with_capacity(N_OUT);
    for (_sess, lo_us, hi_us) in SESSIONS.iter() {
        for v in session_metrics(&trows, &mrows, *lo_us, *hi_us).iter() {
            out.push(*v as f32);
        }
    }
    debug_assert_eq!(out.len(), N_OUT);
    Some(out)
}

/// hm18 全是逐股因子，没有横截面版本。
pub fn compute_cross(_date: i64) -> Option<(Vec<String>, Vec<f32>)> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_order_matches_old_names_single() {
        assert_eq!(
            names(),
            vec![
                "ask_all",
                "bid_all",
                "ask_open30",
                "bid_open30",
                "ask_morning",
                "bid_morning",
                "ask_afternoon",
                "bid_afternoon",
                "ask_close30",
                "bid_close30",
            ]
        );
    }

    #[test]
    fn mode_price_takes_largest_on_tie() {
        assert_eq!(mode_price(&[1.0, 2.0, 2.0, 1.0]), Some(2.0));
        assert_eq!(mode_price(&[1.0, 3.0, 3.0, 1.0]), Some(3.0));
        assert_eq!(mode_price(&[]), None);
    }

    #[test]
    fn skew_follows_pandas() {
        // 样本不足
        assert!(skew(&[1.0, 2.0]).is_nan());
        // 全相等 → 标准差 0 → NaN
        assert!(skew(&[2.0, 2.0, 2.0, 2.0]).is_nan());
        // 对称样本偏度为 0
        let s = skew(&[1.0, 2.0, 3.0]);
        assert!(s.abs() < 1e-12, "got {s}");
    }
}
