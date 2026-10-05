//! hm003 网络延迟因子 —— 旧框架（design_whatever_out.TickDataBase）的 Rust 复现。
//!
//! 旧文件：/home/chenzongwei/pythoncode/成果hm3_网络延迟因子/hm3_okay.py
//! 参数：get_many=2 / read_full=0 / change_time=0 / use_old=True
//!
//! ════════════════════════════════════════════════════════════════════
//! 一、change_time=0 是这一系列的关键
//! ════════════════════════════════════════════════════════════════════
//! change_time=0 时框架**不调 deal_data**，所以 exchtime / localtime 两列保持原始
//! 微秒数没被转成 Timestamp。旧代码的 `df.delay = df.exchtime - df.localtime`
//! 因此是**微秒**单位的真实延迟，整列都是精确整数。
//! （对比 hm039 那种 change_time=1 的系列，exchtime 会被转成时间戳。）
//!
//! ════════════════════════════════════════════════════════════════════
//! 二、五个时段（get_many=2 的切分，真实钟点闭区间）
//! ════════════════════════════════════════════════════════════════════
//! 框架先算 time_judge = exchtime/1e6 秒 + 1970-01-01 08:00（北京时间），再 between_time：
//!   allday    全部行
//!   open30    [09:29:50, 10:00:10]
//!   morning   [09:29:50, 11:30:10]
//!   afternoon [12:59:50, 15:00:10]
//!   close30   [14:29:50, 15:00:10]
//! 段顺序 = [allday, open30, morning, afternoon, close30]，
//! names 也是 names1(allday)+names4(open30)+names2(morning)+names3(afternoon)+names5(close30)，
//! 两边顺序一致。
//! 撤单按 with_retreat=0 丢掉 flag==32 的行（自己按数值滤，不靠 fast_csv_reader 的 with_retreat）。
//!
//! ════════════════════════════════════════════════════════════════════
//! 三、每个时段 9 个指标（旧代码 one(df)，df.shape[0]<=1 时整段 NaN）
//! ════════════════════════════════════════════════════════════════════
//!   total_turnover   = turnover 之和
//!   common_act_buy   = flag==66 的 turnover 之和
//!   delay            = exchtime - localtime（微秒）
//!   delay_q10/q90    = delay 的 0.1/0.9 分位（pandas 线性插值）
//!   ask_delay_mean   = 按 ask_order 分组求 delay 均值，再对**各组的值**取均值
//!   ask_delay_std    = 按 ask_order 分组求 delay 样本标准差（ddof=1，单行组为 NaN，
//!                      最后 .mean() 会跳过 NaN）；bid 侧同理
//!   mean_diff        = bid_delay_mean_mean - ask_delay_mean_mean
//!   std_diff         = bid_delay_std_mean  - ask_delay_std_mean
//!   long/short       = delay >= q90 / delay <= q10
//!   long_turnover    = long.turnover.sum() / total_turnover
//!   short_turnover   = short.turnover.sum() / total_turnover
//!   ls_diff          = long_turnover - short_turnover
//!   ls_diff_act_buy  = long[flag==66] 占比 - short[flag==66] 占比
//!   short_delay_act_buy_turnover = short_turnover 里的 flag==66 那个占比
//!   bid_turnover     = 按 (bid_order, bid_delay) 分组求 turnover 之和；
//!                      bid_delay 在一个委托内是常数，所以等价于「每个委托一行」，
//!                      行上带该委托的 bid_delay
//!   bid_turnover_q10 = 上面那张表 bid_delay 列的 0.1 分位（对「每个委托」取分位）
//!   q10_mean         = Σ(turnover | bid_delay<=q10) / Σ turnover
//!   q90_mean         = 同上用 >= q90
//!   bid_turnover_diff = q90_mean - q10_mean
//! 返回顺序 = [common_act_buy, mean_diff, std_diff, short_delay_act_buy_turnover,
//!            ls_diff, ls_diff_act_buy, bid_turnover_q10_mean, bid_turnover_q90_mean,
//!            bid_turnover_diff]
//! 五个时段共 45 个值（names_save 是其中 27 个）。
//!
//! ════════════════════════════════════════════════════════════════════
//! 四、run_base 里还有一步跨股票正交（不在 get_factor 内，Rust 侧不实现）
//! ════════════════════════════════════════════════════════════════════
//! 旧代码算完 45 个原值后，对 27 个入选名里**非 common_act_buy 的那 22 个**做了
//! dwo.de_cross(该因子.rank(axis=1), common_act_buy_同段.rank(axis=1))：
//! 逐日把「该因子的截面排名」对「同段 common_act_buy 的截面排名」做带截距的 OLS，
//! 用残差覆盖原值。这一步在 hm3_okay_database.py 的 run_base 里用 Python 复刻
//! （verify_series 只对照 get_factor 的 45 个原值，不涉及这一层）。

use std::collections::HashMap;
use std::fs::File;
use std::io::Read;

// ============================================================================
// 时间常量
// ============================================================================

const SEC_NS: i64 = 1_000_000_000;
const DAY_NS: i64 = 86_400 * SEC_NS;
const BJ_OFFSET_NS: i64 = 8 * 3600 * SEC_NS;

const fn hms(h: i64, m: i64, s: i64) -> i64 {
    (h * 3600 + m * 60 + s) * SEC_NS
}

/// 五个时段（真实钟点闭区间），顺序 = 框架 get_many=2 的段顺序 = names 顺序
const SEGMENTS: [(i64, i64); 5] = [
    (0, DAY_NS),                     // allday
    (hms(9, 29, 50), hms(10, 0, 10)),   // open30
    (hms(9, 29, 50), hms(11, 30, 10)),  // morning
    (hms(12, 59, 50), hms(15, 0, 10)),  // afternoon
    (hms(14, 29, 50), hms(15, 0, 10)),  // close30
];

const PER_SEGMENT: usize = 9;
pub const N_OUT: usize = PER_SEGMENT * 5; // 45

const METRIC_NAMES: [&str; PER_SEGMENT] = [
    "common_act_buy",
    "mean_diff",
    "std_diff",
    "short_delay_act_buy_turnover",
    "ls_diff",
    "ls_diff_act_buy",
    "bid_turnover_q10_mean",
    "bid_turnover_q90_mean",
    "bid_turnover_diff",
];
const SEGMENT_NAMES: [&str; 5] = ["allday", "open30", "morning", "afternoon", "close30"];

/// names1(allday) + names4(open30) + names2(morning) + names3(afternoon) + names5(close30)
pub fn names() -> Vec<String> {
    let mut out = Vec::with_capacity(N_OUT);
    for s in SEGMENT_NAMES.iter() {
        for m in METRIC_NAMES.iter() {
            out.push(format!("{}_{}", m, s));
        }
    }
    out
}

// ============================================================================
// CSV 读取（列索引与 fast_csv_reader 的常量一致）
// ============================================================================

const TRD_COL_EXCHTIME: usize = 4;
const TRD_COL_LOCALTIME: usize = 5;
const TRD_COL_TURNOVER: usize = 9;
const TRD_COL_FLAG: usize = 10;
const TRD_COL_ASK_ORDER: usize = 13;
const TRD_COL_BID_ORDER: usize = 14;
const TRD_NCOL: usize = 15;

#[inline]
fn parse_i64_at(b: &[u8]) -> i64 {
    let mut i = 0usize;
    while i < b.len() && b[i] == b' ' {
        i += 1;
    }
    let mut neg = false;
    if i < b.len() && (b[i] == b'-' || b[i] == b'+') {
        neg = b[i] == b'-';
        i += 1;
    }
    let mut v: i64 = 0;
    while i < b.len() {
        let c = b[i];
        if !(b'0'..=b'9').contains(&c) {
            break;
        }
        v = v * 10 + (c - b'0') as i64;
        i += 1;
    }
    if neg {
        -v
    } else {
        v
    }
}

#[inline]
fn parse_f64_at(b: &[u8]) -> f64 {
    if b.is_empty() {
        return f64::NAN;
    }
    std::str::from_utf8(b)
        .ok()
        .and_then(|s| s.trim().parse::<f64>().ok())
        .unwrap_or(f64::NAN)
}

fn split_fields<'a>(line: &'a [u8], f: &mut [&'a [u8]]) {
    let n = f.len();
    for x in f.iter_mut() {
        *x = &line[..0];
    }
    let mut start = 0usize;
    let mut col = 0usize;
    for (i, &b) in line.iter().enumerate() {
        if b == b',' {
            if col < n {
                f[col] = &line[start..i];
            }
            col += 1;
            start = i + 1;
        }
    }
    if col < n {
        f[col] = &line[start..];
    }
}

/// 一条逐笔（只留要用的列）。clock 是北京时间当天钟点（纳秒），delay 是微秒。
#[derive(Clone, Copy)]
struct Row {
    clock: i64,
    delay: f64,
    turnover: f64,
    flag: i64,
    ask_order: i64,
    bid_order: i64,
}

fn read_trades(code: &str, date: i64) -> Option<Vec<Row>> {
    let filename = format!("{}_{}_transaction.csv", code, date);
    let mut path = None;
    for root in crate::data_paths::level2_roots() {
        let p = root.join(date.to_string()).join("transaction").join(&filename);
        if p.exists() {
            path = Some(p);
            break;
        }
    }
    let path = path?;
    let mut content = String::new();
    File::open(&path).ok()?.read_to_string(&mut content).ok()?;

    let mut out: Vec<Row> = Vec::with_capacity(content.len() / 120 + 1);
    let mut lines = content.lines();
    lines.next();
    for line in lines {
        if line.is_empty() {
            continue;
        }
        let bytes = line.as_bytes();
        let mut f: [&[u8]; TRD_NCOL] = [&[][..]; TRD_NCOL];
        split_fields(bytes, &mut f);
        let flag = parse_i64_at(f[TRD_COL_FLAG]);
        if flag == 32 {
            continue; // with_retreat=0
        }
        let ex = parse_i64_at(f[TRD_COL_EXCHTIME]);
        let lo = parse_i64_at(f[TRD_COL_LOCALTIME]);
        out.push(Row {
            clock: (ex * 1000 + BJ_OFFSET_NS).rem_euclid(DAY_NS),
            delay: (ex - lo) as f64,
            turnover: parse_f64_at(f[TRD_COL_TURNOVER]),
            flag,
            ask_order: parse_i64_at(f[TRD_COL_ASK_ORDER]),
            bid_order: parse_i64_at(f[TRD_COL_BID_ORDER]),
        });
    }
    Some(out)
}

// ============================================================================
// 小工具
// ============================================================================

/// pandas Series.quantile(q) 默认的线性插值
fn quantile_linear(sorted: &[f64], q: f64) -> f64 {
    let n = sorted.len();
    if n == 0 {
        return f64::NAN;
    }
    if n == 1 {
        return sorted[0];
    }
    let pos = q * (n as f64 - 1.0);
    let lo = pos.floor();
    let frac = pos - lo;
    let i = lo as usize;
    if i + 1 >= n {
        return sorted[n - 1];
    }
    sorted[i] + frac * (sorted[i + 1] - sorted[i])
}

/// 分组求均值的均值 + 分组样本标准差的均值（pandas 的 .std() 是 ddof=1，单元素组 NaN，
/// 最后 .mean() 跳过 NaN；两组都空时返回 NaN）
fn group_stats(ids: &[i64], vals: &[f64]) -> (f64, f64) {
    let mut cnt: HashMap<i64, usize> = HashMap::new();
    let mut sum: HashMap<i64, f64> = HashMap::new();
    for i in 0..ids.len() {
        *cnt.entry(ids[i]).or_insert(0) += 1;
        *sum.entry(ids[i]).or_insert(0.0) += vals[i];
    }
    // 组内均值（两遍法算方差，比 Σx² 稳）
    let mut mean: HashMap<i64, f64> = HashMap::with_capacity(cnt.len());
    for (k, &c) in cnt.iter() {
        mean.insert(*k, sum[k] / c as f64);
    }
    let mut ss: HashMap<i64, f64> = HashMap::with_capacity(cnt.len());
    for i in 0..ids.len() {
        let m = mean[&ids[i]];
        *ss.entry(ids[i]).or_insert(0.0) += (vals[i] - m) * (vals[i] - m);
    }
    // 均值：各组的组内均值再取平均（对每组累加一次，再除以组数）
    let mut mean_sum = 0.0f64;
    let mut mean_cnt = 0usize;
    for (k, _c) in cnt.iter() {
        mean_sum += mean[k];
        mean_cnt += 1;
    }
    let mean_of_means = if mean_cnt == 0 {
        f64::NAN
    } else {
        mean_sum / mean_cnt as f64
    };
    let mut std_mean = 0.0f64;
    let mut std_cnt = 0usize;
    for (k, &c) in cnt.iter() {
        if c < 2 {
            continue; // pandas 单元素组的 std 是 NaN，.mean() 跳过
        }
        let var = ss[k] / (c as f64 - 1.0);
        std_mean += var.sqrt();
        std_cnt += 1;
    }
    let std_of_stds = if std_cnt == 0 {
        f64::NAN
    } else {
        std_mean / std_cnt as f64
    };
    (mean_of_means, std_of_stds)
}

// ============================================================================
// 一个时段的 9 个指标
// ============================================================================

fn segment_metrics(rows: &[&Row]) -> [f64; PER_SEGMENT] {
    if rows.len() <= 1 {
        return [f64::NAN; PER_SEGMENT];
    }
    let n = rows.len();
    let mut total_turnover = 0.0f64;
    let mut common_act_buy = 0.0f64;
    let mut delays: Vec<f64> = Vec::with_capacity(n);
    let mut ask_ids: Vec<i64> = Vec::with_capacity(n);
    let mut bid_ids: Vec<i64> = Vec::with_capacity(n);
    let mut turnovers: Vec<f64> = Vec::with_capacity(n);
    let mut flags: Vec<i64> = Vec::with_capacity(n);
    for r in rows.iter() {
        total_turnover += r.turnover;
        if r.flag == 66 {
            common_act_buy += r.turnover;
        }
        delays.push(r.delay);
        ask_ids.push(r.ask_order);
        bid_ids.push(r.bid_order);
        turnovers.push(r.turnover);
        flags.push(r.flag);
    }

    let mut sorted_delay = delays.clone();
    sorted_delay.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let q10 = quantile_linear(&sorted_delay, 0.1);
    let q90 = quantile_linear(&sorted_delay, 0.9);

    let (ask_mean_mean, ask_std_mean) = group_stats(&ask_ids, &delays);
    let (bid_mean_mean, bid_std_mean) = group_stats(&bid_ids, &delays);
    let mean_diff = bid_mean_mean - ask_mean_mean;
    let std_diff = bid_std_mean - ask_std_mean;

    let mut long_turn = 0.0f64;
    let mut short_turn = 0.0f64;
    let mut long_act = 0.0f64;
    let mut short_act = 0.0f64;
    for i in 0..n {
        if delays[i] >= q90 {
            long_turn += turnovers[i];
            if flags[i] == 66 {
                long_act += turnovers[i];
            }
        }
        if delays[i] <= q10 {
            short_turn += turnovers[i];
            if flags[i] == 66 {
                short_act += turnovers[i];
            }
        }
    }
    let long_delay_turnover = long_turn / total_turnover;
    let short_delay_turnover = short_turn / total_turnover;
    let long_delay_act_buy_turnover = long_act / total_turnover;
    let short_delay_act_buy_turnover = short_act / total_turnover;
    let ls_diff = long_delay_turnover - short_delay_turnover;
    let ls_diff_act_buy = long_delay_act_buy_turnover - short_delay_act_buy_turnover;

    // bid_turnover：按 (bid_order, bid_delay) 分组，bid_delay 在委托内是常数，
    // 所以就是「每个委托一行」，行上带该委托的 bid_delay
    let mut bid_sum: HashMap<i64, f64> = HashMap::new();
    for i in 0..n {
        *bid_sum.entry(bid_ids[i]).or_insert(0.0) += turnovers[i];
    }
    // 每个委托的 delay（组内均值；组内常数，等价）
    let mut bid_cnt: HashMap<i64, usize> = HashMap::new();
    let mut bid_dsum: HashMap<i64, f64> = HashMap::new();
    for i in 0..n {
        *bid_cnt.entry(bid_ids[i]).or_insert(0) += 1;
        *bid_dsum.entry(bid_ids[i]).or_insert(0.0) += delays[i];
    }
    let mut orders: Vec<(f64, f64)> = Vec::with_capacity(bid_sum.len()); // (delay, turnover)
    let mut bid_turnover_sum = 0.0f64;
    for (k, &tv) in bid_sum.iter() {
        let d = bid_dsum[k] / bid_cnt[k] as f64;
        orders.push((d, tv));
        bid_turnover_sum += tv;
    }
    // HashMap 遍历顺序不定，先排序（只影响浮点求和次序，不影响结果量级）
    orders.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    let order_delays: Vec<f64> = orders.iter().map(|x| x.0).collect();
    let bq10 = quantile_linear(&order_delays, 0.1);
    let bq90 = quantile_linear(&order_delays, 0.9);
    let mut lo_sum = 0.0f64;
    let mut hi_sum = 0.0f64;
    for &(d, tv) in orders.iter() {
        if d <= bq10 {
            lo_sum += tv;
        }
        if d >= bq90 {
            hi_sum += tv;
        }
    }
    let bid_turnover_q10_mean = lo_sum / bid_turnover_sum;
    let bid_turnover_q90_mean = hi_sum / bid_turnover_sum;
    let bid_turnover_diff = bid_turnover_q90_mean - bid_turnover_q10_mean;

    [
        common_act_buy,
        mean_diff,
        std_diff,
        short_delay_act_buy_turnover,
        ls_diff,
        ls_diff_act_buy,
        bid_turnover_q10_mean,
        bid_turnover_q90_mean,
        bid_turnover_diff,
    ]
}

/// 逐股入口：算不出来返回 None（上游补 NaN，保证因子个数恒定）。
pub fn compute_stock(code: &str, date: i64) -> Option<Vec<f32>> {
    let rows = read_trades(code, date)?;
    let mut out: Vec<f32> = Vec::with_capacity(N_OUT);
    for &(lo, hi) in SEGMENTS.iter() {
        let sel: Vec<&Row> = rows
            .iter()
            .filter(|r| r.clock >= lo && r.clock <= hi)
            .collect();
        out.extend(segment_metrics(&sel).iter().map(|&v| v as f32));
    }
    debug_assert_eq!(out.len(), N_OUT);
    Some(out)
}

/// hm003 的逐股部分没有横截面版本（跨股票正交那一步在 Python 侧做）。
pub fn compute_cross(_date: i64) -> Option<(Vec<String>, Vec<f32>)> {
    None
}
