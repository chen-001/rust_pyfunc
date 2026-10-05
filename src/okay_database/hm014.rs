//! hm014 K线里程碑 —— 旧框架（design_whatever_out.TickDataBase）的 Rust 复现。
//!
//! 旧文件：/home/chenzongwei/pythoncode/成果hm14_k线里程碑/hm14_okay.py
//!
//! 旧代码要点（逐条对齐）：
//!
//! 【调用参数】get_many=0、read_full=0、change_time=1 → read_trade_data_single 返回**单个 DataFrame**，
//!   列是 [date, exchtime, localtime, price, volume, turnover, flag, ask_order, bid_order, code, delay]，
//!   exchtime 已被 deal_data 换成 Timestamp（= 原始 exchtime 微秒/1e6 + 1970-01-01 08:00，即北京时间墙钟）。
//!   with_retreat=0 → 丢掉 flag==32（撤单）的行。Rust 侧读进来后自己按数值过滤 flag（见下）。
//!
//! 【切窗】df = set_index("exchtime").between_time("09:30:00","14:57:00")
//!   df1 = between("09:30:00","11:30:00")      真实钟点
//!   df2 = between("13:00:00","14:57:00") 再整体前移 90 分钟
//!   df = concat([df1, df2])   ← **行顺序 = df1 全部在前、df2 全部在后**，price.last() 依赖这个顺序
//!   五个时段（都在平移后的时间轴上取）：
//!       all       = df 全量
//!       open30    = [09:30:00, 10:00:00]
//!       morning   = [09:30:00, 11:30:00]   （= df1 全部 + df2 里真实 13:00:00.000000 那一笔）
//!       afternoon = [11:30:00, 13:27:00]   （= df1 里真实 11:30:00.000000 那一笔 + df2 全部）
//!       close30   = [13:00:00, 13:27:00]   （= 真实 14:30:00 ~ 14:57:00）
//!   注意 (11:30:00, 13:00:00) 真实钟点之间的行（若有）在 concat 时被丢掉。
//!   边界是**闭区间且精确到微秒**，所以 Rust 侧一律用微秒整数比，不要按整秒截断。
//!
//! 【每个时段的 4 个指标】旧代码 sing(df)，顺序 [buy, sell, big_buy, big_sell]：
//!   asks = df.groupby('ask_order').volume.sum()
//!   bids = df.groupby('bid_order').volume.sum()
//!   ask_big = asks[asks >= asks.quantile(0.7)]，bid_big 同理（quantile 默认线性插值）
//!   df_ask_big = df[ask_order ∈ ask_big]，df_bid_big = df[bid_order ∈ bid_big]
//!   one(x_ask, x_bid) 里对 7 个时移 s ∈ {0,-30,-20,-10,+10,+20,+30} 秒各算一次：
//!       x_bid 整体平移 s 秒 → 取 flag==66 的行（主买）→ 按 1 分钟分箱求和 → shift(-1)
//!       x_ask 整体平移 s 秒 → 取 flag==83 的行（主卖）→ 同上
//!       price = **外层 df** 的 1 分钟 price.last()，vvs = **外层 df** 的 1 分钟 volume.sum()
//!       price 只保留 vvs > vvs.median() 的那些分钟
//!       corr = pearson(price, 平移后的分箱量)  成对完整观测
//!   返回 [corr0_66 - mean(corr_其他 6 个), corr0_83 - mean(其他 6 个)]
//!   sing = one(df, df) + one(df_ask_big, df_bid_big)
//!        = [buy, sell, big_buy, big_sell]
//!   注意 big 那一组用的是**两个不同的子集**：buy 用 bid_big，sell 用 ask_big。
//!
//! 【分箱细节】分箱是 pandas 在「平移后的时间轴」上按自然分钟对齐（origin 为当天 0 点），
//!   所以 s≠0 时箱子与真实钟点的分钟并不对齐，这正是这个因子的设计含义。
//!   shift(-1) 把箱子 b 的值换成箱子 b+1 的值，最后一箱变 NaN；随后 replace(0→NaN).dropna()，
//!   于是有效对 = {b : vvs[b] > median 且 平移分箱和[b+1] ≠ 0 且 b+1 在范围内}。
//!
//! 5 个时段 × 4 个指标 = 20 个输出，顺序与旧代码 names_single 完全一致。
//!
//! ════════════════════════════════════════════════════════════════════
//! 精度：price 必须按 f64 读，不能用 fast_csv_reader 的 f32
//! ════════════════════════════════════════════════════════════════════
//! price.last() 只取每分钟最后一笔的价格，相关系数里价格这一侧的方差很小，
//! 而结果本身常常是「两个大数相减得到的小数」（|值| 只有 1e-3 量级），
//! 所以价格上的相对误差会被放大。实测 f32 读 price 时，600519@20250304 的
//! open30_sell 相对误差 1.64e-4、open30_big_sell 1.63e-4，超过 1e-4 阈值；
//! 换成 f64 逐字节对照后逐值一致。故本文件自己解析 CSV（列索引与
//! fast_csv_reader 的常量一致），file 行序与 pandas 一致。

use std::collections::HashMap;
use std::fs::File;
use std::io::Read;

/// 逐笔 CSV 的列索引（0 基，与 fast_csv_reader 一致）
const TRD_COL_EXCHTIME: usize = 4;
const TRD_COL_PRICE: usize = 7;
const TRD_COL_VOLUME: usize = 8;
const TRD_COL_FLAG: usize = 10;
const TRD_COL_ASK_ORDER: usize = 13;
const TRD_COL_BID_ORDER: usize = 14;
const TRD_NCOL: usize = 15;

/// 五个时段的名字（旧代码 names 的前缀），顺序即输出顺序
const SESSION_NAMES: [&str; 5] = ["all", "open30", "morning", "afternoon", "close30"];

/// 每个时段 4 个指标，顺序与 sing() 的返回一致
const METRIC_NAMES: [&str; 4] = ["buy", "sell", "big_buy", "big_sell"];

const PER_SESSION: usize = 4;
pub const N_OUT: usize = PER_SESSION * 5;

/// 秒 → 微秒
const US: i64 = 1_000_000;
/// 一分钟的微秒数
const MIN_US: i64 = 60 * US;
/// 下午整体前移 90 分钟
const AFTERNOON_SHIFT_US: i64 = 90 * 60 * US;

/// 真实钟点的关键边界（当天秒数）
const T_0930: i64 = 9 * 3600 + 30 * 60; // 34200
const T_1000: i64 = 10 * 3600; // 36000
const T_1130: i64 = 11 * 3600 + 30 * 60; // 41400
const T_1300: i64 = 13 * 3600; // 46800
const T_1457: i64 = 14 * 3600 + 57 * 60; // 53820
/// 平移后 14:57 对应的标签 13:27
const T_1327_SHIFTED: i64 = T_1457 - 90 * 60; // 48420

/// 五个时段在「平移后时间轴」上的闭区间（当天微秒数）
const WINDOWS: [(i64, i64); 5] = [
    (i64::MIN, i64::MAX),                                    // all
    (T_0930 * US, T_1000 * US),                              // open30   09:30:00 ~ 10:00:00
    (T_0930 * US, T_1130 * US),                              // morning  09:30:00 ~ 11:30:00
    (T_1130 * US, T_1327_SHIFTED * US),                      // afternoon 11:30:00 ~ 13:27:00
    (T_1300 * US, T_1327_SHIFTED * US),                      // close30  13:00:00 ~ 13:27:00
];

/// 七个时移（秒），顺序与旧代码一致：0, -30, -20, -10, +10, +20, +30
const SHIFTS_SEC: [i64; 7] = [0, -30, -20, -10, 10, 20, 30];

/// 一行成交（只留这个因子用得到的字段）
#[derive(Clone, Copy)]
struct Row {
    /// 平移后的当天微秒数（下午段已减 90 分钟）
    t_us: i64,
    price: f64,
    volume: f64,
    flag: i32,
    ask: i64,
    bid: i64,
}

pub fn names() -> Vec<String> {
    let mut out = Vec::with_capacity(N_OUT);
    for sess in SESSION_NAMES.iter() {
        for m in METRIC_NAMES.iter() {
            out.push(format!("{}_{}", sess, m));
        }
    }
    out
}

#[inline]
fn day_us(time_us: i64) -> i64 {
    time_us.rem_euclid(86_400_000_000)
}

/// pandas Series.quantile 的默认线性插值：位置 = (n-1)*q，落在两个相邻值之间就线性插。
/// 传进来的切片必须已升序排好。
fn quantile_linear(sorted: &[f64], q: f64) -> f64 {
    let n = sorted.len();
    if n == 0 {
        return f64::NAN;
    }
    if n == 1 {
        return sorted[0];
    }
    let pos = (n as f64 - 1.0) * q;
    let lo = pos.floor();
    let hi = pos.ceil();
    let lo_i = lo as usize;
    if lo == hi {
        return sorted[lo_i.min(n - 1)];
    }
    let frac = pos - lo;
    sorted[lo_i] + (sorted[(lo_i + 1).min(n - 1)] - sorted[lo_i]) * frac
}

fn sort_f64(v: &mut [f64]) {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
}

/// pandas DataFrame.corr() 的 Pearson：成对完整观测，样本 < 2 或某侧方差为 0 返回 NaN。
fn pearson(x: &[f64], y: &[f64]) -> f64 {
    let mut n = 0usize;
    let mut sx = 0.0f64;
    let mut sy = 0.0f64;
    let mut sxx = 0.0f64;
    let mut syy = 0.0f64;
    let mut sxy = 0.0f64;
    for i in 0..x.len() {
        if !x[i].is_finite() || !y[i].is_finite() {
            continue;
        }
        n += 1;
        sx += x[i];
        sy += y[i];
        sxx += x[i] * x[i];
        syy += y[i] * y[i];
        sxy += x[i] * y[i];
    }
    if n < 2 {
        return f64::NAN;
    }
    let nf = n as f64;
    let cov = sxy - sx * sy / nf;
    let vx = sxx - sx * sx / nf;
    let vy = syy - sy * sy / nf;
    if vx <= 0.0 || vy <= 0.0 {
        return f64::NAN;
    }
    cov / (vx.sqrt() * vy.sqrt())
}

/// 一个来源（原始 df 或 big 子集）里按方向切好的 (时间, 成交量) 序列
struct Src {
    buy: Vec<(i64, f64)>,
    sell: Vec<(i64, f64)>,
}

impl Src {
    fn collect(rows: &[Row], pick: impl Fn(&Row) -> bool) -> Src {
        let mut buy = Vec::new();
        let mut sell = Vec::new();
        for r in rows.iter() {
            if !pick(r) {
                continue;
            }
            if r.flag == 66 {
                buy.push((r.t_us, r.volume));
            } else if r.flag == 83 {
                sell.push((r.t_us, r.volume));
            }
        }
        Src { buy, sell }
    }
}

/// 平移 delta 微秒后的 1 分钟分箱求和，shift(-1) 之后与 price 做 Pearson。
/// `keep` 是外层 df 的分箱序号（相对 b0），只保留 vvs > median 的那些。
fn corr_shifted(price: &[f64], keep: &[usize], b0: i64, src: &[(i64, f64)], delta_us: i64) -> f64 {
    if src.is_empty() {
        return f64::NAN;
    }
    let mut s0 = i64::MAX;
    let mut s1 = i64::MIN;
    for &(t, _) in src.iter() {
        let b = (t + delta_us).div_euclid(MIN_US);
        if b < s0 {
            s0 = b;
        }
        if b > s1 {
            s1 = b;
        }
    }
    let ns = (s1 - s0 + 1) as usize;
    let mut bins = vec![0.0f64; ns];
    for &(t, v) in src.iter() {
        let b = (t + delta_us).div_euclid(MIN_US);
        bins[(b - s0) as usize] += v;
    }
    let mut xs: Vec<f64> = Vec::with_capacity(keep.len());
    let mut ys: Vec<f64> = Vec::with_capacity(keep.len());
    for &i in keep.iter() {
        let b = b0 + i as i64;
        // shift(-1)：箱子 b 取 b+1 的值，最后一箱没有下一箱
        if b < s0 || b + 1 > s1 {
            continue;
        }
        let v = bins[(b + 1 - s0) as usize];
        if v == 0.0 {
            continue;
        }
        let p = price[i];
        if !p.is_finite() {
            continue;
        }
        xs.push(p);
        ys.push(v);
    }
    pearson(&xs, &ys)
}

/// 旧代码的 sing(df)：返回 [buy, sell, big_buy, big_sell]
fn sing(rows: &[Row]) -> [f64; PER_SESSION] {
    if rows.is_empty() {
        return [f64::NAN; PER_SESSION];
    }
    // ---- groupby ask_order / bid_order 的 volume 求和，再取 0.7 分位 ----
    let mut ask_sum: HashMap<i64, f64> = HashMap::new();
    let mut bid_sum: HashMap<i64, f64> = HashMap::new();
    for r in rows.iter() {
        *ask_sum.entry(r.ask).or_insert(0.0) += r.volume;
        *bid_sum.entry(r.bid).or_insert(0.0) += r.volume;
    }
    // 遍历 HashMap 求统计量前先排序，保证结果与进程无关
    let mut av: Vec<f64> = ask_sum.values().cloned().collect();
    let mut bv: Vec<f64> = bid_sum.values().cloned().collect();
    sort_f64(&mut av);
    sort_f64(&mut bv);
    let qa = quantile_linear(&av, 0.7);
    let qb = quantile_linear(&bv, 0.7);
    let is_big = |m: &HashMap<i64, f64>, k: i64, q: f64| -> bool {
        if !q.is_finite() {
            return false;
        }
        m.get(&k).map_or(false, |&v| v >= q)
    };
    // 两个子集是分开的：df_bid_big 决定 buy，df_ask_big 决定 sell
    let plain = Src::collect(rows, |_| true);
    let big = Src {
        buy: rows
            .iter()
            .filter(|r| r.flag == 66 && is_big(&bid_sum, r.bid, qb))
            .map(|r| (r.t_us, r.volume))
            .collect(),
        sell: rows
            .iter()
            .filter(|r| r.flag == 83 && is_big(&ask_sum, r.ask, qa))
            .map(|r| (r.t_us, r.volume))
            .collect(),
    };

    // ---- 外层 df 的 1 分钟 volume.sum() 与 price.last() ----
    let mut b0 = i64::MAX;
    let mut b1 = i64::MIN;
    for r in rows.iter() {
        let b = r.t_us.div_euclid(MIN_US);
        if b < b0 {
            b0 = b;
        }
        if b > b1 {
            b1 = b;
        }
    }
    let nb = (b1 - b0 + 1) as usize;
    let mut vvs = vec![0.0f64; nb];
    let mut price = vec![f64::NAN; nb];
    for r in rows.iter() {
        // 顺序遍历、后面的覆盖前面的 = pandas resample(...).last()
        let i = (r.t_us.div_euclid(MIN_US) - b0) as usize;
        vvs[i] += r.volume;
        price[i] = r.price;
    }
    let mut vsorted = vvs.clone();
    sort_f64(&mut vsorted);
    let med = quantile_linear(&vsorted, 0.5);
    let keep: Vec<usize> = (0..nb).filter(|&i| vvs[i] > med).collect();

    // ---- 7 个时移：第 0 个减去其余 6 个的均值 ----
    let one = |src: &Src| -> [f64; 2] {
        let mut buy_other = 0.0f64;
        let mut sell_other = 0.0f64;
        for &s in SHIFTS_SEC[1..].iter() {
            buy_other += corr_shifted(&price, &keep, b0, &src.buy, s * US);
            sell_other += corr_shifted(&price, &keep, b0, &src.sell, s * US);
        }
        let buy0 = corr_shifted(&price, &keep, b0, &src.buy, 0);
        let sell0 = corr_shifted(&price, &keep, b0, &src.sell, 0);
        [buy0 - buy_other / 6.0, sell0 - sell_other / 6.0]
    };
    let p = one(&plain);
    let b = one(&big);
    [p[0], p[1], b[0], b[1]]
}

// ============================================================================
// 逐笔 CSV 的 f64 读取（列索引与 fast_csv_reader 一致，price 保持 f64）
// ============================================================================

/// 一条逐笔成交（只留这个因子用得到的列）
#[derive(Clone, Copy, Default)]
struct TradeRow {
    /// 含 +8 小时偏移的 epoch 微秒（与 fast_csv_reader 的 time_us 同义）
    time_us: i64,
    price: f64,
    volume: f64,
    flag: i32,
    ask_order: i64,
    bid_order: i64,
}

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

/// 空单元格按 pandas 的读法给 NaN；其余交给 std 的正确舍入解析（与 pandas 的 strtod 一致）
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

/// 读 transaction CSV（f64）。行集与 pandas 读整张表一致：表头之后每一行都算，
/// exchtime 为空/0 的行也保留（它们落在 08:00，后面按窗口切时自然被丢掉），
/// 保持与 pandas 相同的行序。
fn read_trade_f64(code: &str, date: i64) -> Option<Vec<TradeRow>> {
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

    let mut out: Vec<TradeRow> = Vec::with_capacity(content.len() / 120 + 1);
    let mut lines = content.lines();
    lines.next(); // 表头
    for line in lines {
        if line.is_empty() {
            continue;
        }
        let bytes = line.as_bytes();
        let mut f: [&[u8]; TRD_NCOL] = [&[][..]; TRD_NCOL];
        let mut start = 0usize;
        let mut col = 0usize;
        for (i, &b) in bytes.iter().enumerate() {
            if b == b',' {
                if col < TRD_NCOL {
                    f[col] = &bytes[start..i];
                }
                col += 1;
                start = i + 1;
            }
        }
        if col < TRD_NCOL {
            f[col] = &bytes[start..];
        }
        out.push(TradeRow {
            time_us: parse_i64_at(f[TRD_COL_EXCHTIME]) + 8 * 3600 * US,
            price: parse_f64_at(f[TRD_COL_PRICE]),
            volume: parse_f64_at(f[TRD_COL_VOLUME]),
            flag: parse_i64_at(f[TRD_COL_FLAG]) as i32,
            ask_order: parse_i64_at(f[TRD_COL_ASK_ORDER]),
            bid_order: parse_i64_at(f[TRD_COL_BID_ORDER]),
        });
    }
    Some(out)
}

/// 逐股入口：算不出来返回 None（上游会补 NaN，保证因子个数恒定）。
pub fn compute_stock(code: &str, date: i64) -> Option<Vec<f32>> {
    // 自己按 f64 读 CSV：撤单过滤不靠库里那条按字节比 "32" 的判断
    // （2025 年前后成交明细里写的是 "32.0"，过滤不生效），这里按数值筛 flag；
    // price 也不能用 f32 的读取器，理由见文件头。读时不做事先的时段过滤，
    // 窗口全部在下面的两个 ranges 里按微秒闭区间切。
    let trades = read_trade_f64(code, date)?;
    if trades.is_empty() {
        return None;
    }

    // df1 段（真实 09:30:00~11:30:00）在前，df2 段（真实 13:00:00~14:57:00，前移 90 分钟）在后
    let mut rows: Vec<Row> = Vec::with_capacity(trades.len());
    for r in trades.iter() {
        if r.flag == 32 {
            continue;
        }
        let du = day_us(r.time_us);
        if du >= T_0930 * US && du <= T_1130 * US {
            rows.push(mk(r, du));
        }
    }
    for r in trades.iter() {
        if r.flag == 32 {
            continue;
        }
        let du = day_us(r.time_us);
        if du >= T_1300 * US && du <= T_1457 * US {
            rows.push(mk(r, du - AFTERNOON_SHIFT_US));
        }
    }
    if rows.is_empty() {
        return None;
    }

    let mut out = Vec::with_capacity(N_OUT);
    for &(lo, hi) in WINDOWS.iter() {
        let sub: Vec<Row> = rows
            .iter()
            .filter(|r| r.t_us >= lo && r.t_us <= hi)
            .cloned()
            .collect();
        out.extend(sing(&sub).iter().map(|&v| v as f32));
    }
    debug_assert_eq!(out.len(), N_OUT);
    Some(out)
}

#[inline]
fn mk(r: &TradeRow, t_us: i64) -> Row {
    Row {
        t_us,
        price: r.price,
        volume: r.volume,
        flag: r.flag,
        ask: r.ask_order,
        bid: r.bid_order,
    }
}

/// hm14 全是逐股因子，没有横截面版本。
pub fn compute_cross(_date: i64) -> Option<(Vec<String>, Vec<f32>)> {
    None
}
