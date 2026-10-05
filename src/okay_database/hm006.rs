//! hm6「动量反转投资者」—— 旧框架（design_whatever_out.TickDataBase）的 Rust 复现。
//!
//! 旧文件：/home/chenzongwei/pythoncode/成果hm6_动量反转投资者/hm6_okay.py
//!   run_base 里的 Go.get_factor，参数 get_many=0 / read_full=0 / change_time=1，
//!   也就是「一整天的逐笔」一张表，窗口由旧代码自己切。
//!
//! ════════════════════════════════════════════════════════════════════
//! 旧代码逐步对照（写代码前先抄在这里，方便复核）
//! ════════════════════════════════════════════════════════════════════
//! 1) df = read_trade_data_single(with_retreat=0)  → 丢掉 flag==32 的撤单行
//!    （规范提醒：Rust 侧 with_retreat 按字节比 "32"、2025 年写的是 "32.0"，
//!      所以这里自己按解析出的整数 flag 过滤，不用库里的参数）
//! 2) df.set_index('exchtime').between_time('09:30:00','14:57:00').sort_index()
//!    两端都含、精确到微秒；文件本身按 exchtime 升序，稳定排序后顺序不变。
//! 3) chg_1p/2p/5p/10p = np.sign(price - price.shift(int(n*f)))，
//!    f 分别是 0.01/0.02/0.05/0.1，shift 是**行数**位移（n = 窗口内行数）
//! 4) chg_1m/5m = np.sign(price - merge_asof(自身, 键=自身 exchtime+1min/5min,
//!    direction='backward'))，即「不超过 (当前时刻 - 1min/5min) 的最后一笔」的价格
//! 5) ask_small_point = 按 ask_order 分组的 turnover 之和，取 < 分组和的 0.3 分位数（线性插值）
//!    bid_small_point 同理按 bid_order
//! 6) asks = df.drop_duplicates('ask_order', keep='first')，bids 同理
//! 7) 五个时段（都是闭区间）：
//!      all       [09:30, 14:57]
//!      open30    [09:30, 10:00]
//!      close30   [14:30, 14:57]
//!      morning   [09:30, 11:30]
//!      afternoon [13:00, 14:57]
//! 8) 每笔成交按 (flag, chg_*) 归四类：flag==66 主买 / 83 主卖，
//!    chg==1 上行、-1 下行。每类给两个指标：
//!      rev_net_sell = (sell_up 的成交额和 - buy_down 的成交额和) / 该时段总成交额
//!      mom_buy_low  = (sell_down 成交额和 / sell_down 成交量和) / sell_down 价格均值
//!    「_small」版还要过一道订单号筛选：
//!      sell_up  行需 ask_order ∈ {asks.chg==1 且 flag==66} 且 bid_order ∈ 小单集合
//!      buy_down 行需 bid_order  ∈ {bids.chg==-1 且 flag==66} 且 bid_order ∈ 小单集合
//!      sell_down 行需 ask_order ∈ {asks.chg==-1 且 flag==83} 且 ask_order ∈ 小单集合
//!    注意 sell_up 筛的是 bid_order（旧代码就这么写的，照抄），小单集合取自全天窗口。
//! 9) 全天两个 std：把四类成交额按「分钟」重采样求和，f1 = 四者之和，
//!    std = (某类分钟成交额 / f1).std()（pandas 默认 ddof=1、跳过 NaN）
//!
//! names（names_single，132 个）：对 1p/2p/5p/10p/1m/5m 六个 tag，各 22 个：
//!   rev_net_sell_T / mom_buy_low_T / sell_up_std_T / buy_down_std_T
//!   rev_net_sell_T_open30 / mom_buy_low_T_open30 / rev_net_sell_T_close30 / mom_buy_low_T_close30
//!   rev_net_sell_T_morning / mom_buy_low_T_morning / rev_net_sell_T_afternoon / mom_buy_low_T_afternoon
//!   rev_net_sell_T_small / mom_buy_low_T_small 以及 _small 的四个时段版
//! names_save 见旧文件（38 个 base 名，本文件不涉及）。names 无重名。
//!
//! ════════════════════════════════════════════════════════════════════
//! 精度与已知偏差
//! ════════════════════════════════════════════════════════════════════
//! · 逐笔的 price/volume/turnover 一律按 f64 解析。rev_net_sell 是两个大额之和相减
//!   再除以总额，f32 的舍入会在这个减法里被放大，实测会跳出一致性阈值。
//! · 时间一律用整数微秒。旧代码走的是 pandas Timestamp（纳秒），
//!   `to_timedelta(exchtime/1e6,'s')` 对 float64 秒做转换时会引入 ±130ns 的误差
//!   （并不是精确的「微秒×1000」）。这个误差只可能影响「两笔成交恰好相差整 60s/整 300s」
//!   这种临界情形下 merge_asof 的归属；窗口边界都是整秒，整秒转换本身是精确的。
//!   实测四个 (日期,股票) 组合（含 2018 年）逐值一致，误差在 1e-12 以内。
//! · 排序：旧代码 sort_index 走 numpy quicksort，对「已经升序、带大量同时刻平局」的输入
//!   实测保持原顺序；这里用稳定排序，等价。
//! · 订单号分组的和、以及按分钟分箱的和，都按订单号/行号排序后累加，保证多次运行结果一致
//!   （HashMap 的遍历顺序不能用）。

use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::Read;

const DAY_US: i64 = 86_400_000_000;
const SEC_US: i64 = 1_000_000;
const MIN_US: i64 = 60 * SEC_US;
const OFFSET_US: i64 = 8 * 3600 * 1_000_000;

const LD_0930: i64 = (9 * 3600 + 30 * 60) * SEC_US;
const LD_1000: i64 = 10 * 3600 * SEC_US;
const LD_1130: i64 = (11 * 3600 + 30 * 60) * SEC_US;
const LD_1300: i64 = 13 * 3600 * SEC_US;
const LD_1430: i64 = (14 * 3600 + 30 * 60) * SEC_US;
const LD_1457: i64 = (14 * 3600 + 57 * 60) * SEC_US;

const N_TAG: usize = 6;
const N_SEG: usize = 5;
const PER_TAG: usize = 22;
pub const N_OUT: usize = PER_TAG * N_TAG;

const TAG_NAMES: [&str; N_TAG] = ["1p", "2p", "5p", "10p", "1m", "5m"];
/// 前四个 tag 用「行数比例」位移，后两个用时间回溯
const TAG_FRAC: [f64; N_TAG] = [0.01, 0.02, 0.05, 0.1, 0.0, 0.0];
const TAG_DELTA: [i64; N_TAG] = [0, 0, 0, 0, 60 * SEC_US, 300 * SEC_US];

/// 段顺序必须与旧 names 的拼接顺序一致
const SEG_SUF: [&str; N_SEG] = ["", "_open30", "_close30", "_morning", "_afternoon"];
const SEG_LO: [i64; N_SEG] = [LD_0930, LD_0930, LD_1430, LD_0930, LD_1300];
const SEG_HI: [i64; N_SEG] = [LD_1457, LD_1000, LD_1457, LD_1130, LD_1457];

pub fn names() -> Vec<String> {
    let mut out = Vec::with_capacity(N_OUT);
    for tag in TAG_NAMES.iter() {
        for (si, suf) in SEG_SUF.iter().enumerate() {
            out.push(format!("rev_net_sell_{}{}", tag, suf));
            out.push(format!("mom_buy_low_{}{}", tag, suf));
            if si == 0 {
                out.push(format!("sell_up_std_{}", tag));
                out.push(format!("buy_down_std_{}", tag));
            }
        }
        for suf in SEG_SUF.iter() {
            out.push(format!("rev_net_sell_{}_small{}", tag, suf));
            out.push(format!("mom_buy_low_{}_small{}", tag, suf));
        }
    }
    out
}

// ============================================================================
// 逐笔 CSV 的 f64 读取
// ============================================================================

/// 列索引（0 基，与 fast_csv_reader 一致；2018 年及以前末尾多一列 num，不影响前 15 列）
const TRD_COL_EXCHTIME: usize = 4;
const TRD_COL_PRICE: usize = 7;
const TRD_COL_VOLUME: usize = 8;
const TRD_COL_TURNOVER: usize = 9;
const TRD_COL_FLAG: usize = 10;
const TRD_COL_ASK_ORDER: usize = 13;
const TRD_COL_BID_ORDER: usize = 14;
const TRD_NCOL: usize = 15;

fn parse_i64_at(b: &[u8]) -> i64 {
    let mut i = 0usize;
    while i < b.len() && (b[i] == b' ' || b[i] == b'\r') {
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

/// 一条逐笔成交（时间已加 8 小时，是北京时间的 epoch 微秒）
#[derive(Clone, Copy, Default)]
struct Row {
    t: i64,
    price: f64,
    volume: f64,
    turnover: f64,
    flag: i64,
    ask_order: i64,
    bid_order: i64,
}

/// 读 transaction CSV（f64），行序保持文件顺序；这里就丢掉 flag==32 的撤单行。
fn read_rows(code: &str, date: i64) -> Option<Vec<Row>> {
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
        let flag = parse_i64_at(f[TRD_COL_FLAG]);
        if flag == 32 {
            continue; // with_retreat=0
        }
        out.push(Row {
            t: parse_i64_at(f[TRD_COL_EXCHTIME]) + OFFSET_US,
            price: parse_f64_at(f[TRD_COL_PRICE]),
            volume: parse_f64_at(f[TRD_COL_VOLUME]),
            turnover: parse_f64_at(f[TRD_COL_TURNOVER]),
            flag,
            ask_order: parse_i64_at(f[TRD_COL_ASK_ORDER]),
            bid_order: parse_i64_at(f[TRD_COL_BID_ORDER]),
        });
    }
    Some(out)
}

// ============================================================================
// pandas 语义的小工具
// ============================================================================

/// np.sign：NaN 原样传回（旧代码里的 np.sign(NaN) 也是 NaN，比较一律为假）
#[inline]
fn sign(x: f64) -> f64 {
    if x > 0.0 {
        1.0
    } else if x < 0.0 {
        -1.0
    } else if x == 0.0 {
        0.0
    } else {
        f64::NAN
    }
}

/// IEEE 除法：0/0 得 NaN，正数/0 得 +inf，与 numpy 一致
#[inline]
fn div(a: f64, b: f64) -> f64 {
    a / b
}

/// Series.mean()：空集是 NaN
#[inline]
fn mean(sum: f64, cnt: f64) -> f64 {
    if cnt == 0.0 {
        f64::NAN
    } else {
        sum / cnt
    }
}

/// np.percentile(..., method='linear')／Series.quantile 的线性插值
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
    let li = lo as usize;
    let hi_i = hi as usize;
    let frac = pos - lo;
    sorted[li] + (sorted[hi_i] - sorted[li]) * frac
}

/// 两趟算法求样本标准差（ddof=1，跳过 NaN，样本 < 2 给 NaN）
fn std_ddof1(values: &[f64]) -> f64 {
    let mut n = 0usize;
    let mut s = 0.0f64;
    for &v in values {
        if v.is_finite() {
            n += 1;
            s += v;
        }
    }
    if n < 2 {
        return f64::NAN;
    }
    let m = s / n as f64;
    let mut ss = 0.0f64;
    for &v in values {
        if v.is_finite() {
            let d = v - m;
            ss += d * d;
        }
    }
    (ss / (n as f64 - 1.0)).sqrt()
}

// ============================================================================
// 主计算
// ============================================================================

#[derive(Clone, Copy, Default)]
struct Acc {
    su_t: f64,  // sell_up 成交额
    bd_t: f64,  // buy_down 成交额
    su2_t: f64, // sell_up（过小单筛）
    bd2_t: f64, // buy_down（过小单筛）
    sd_t: f64,  // sell_down 成交额/成交量/价格和与笔数
    sd_v: f64,
    sd_p: f64,
    sd_n: f64,
    sd2_t: f64,
    sd2_v: f64,
    sd2_p: f64,
    sd2_n: f64,
}

/// 全天两分钟级 std：返回 (sell_up_std, buy_down_std)。
///
/// 旧代码是四个子集各做一次 resample('T').sum()，再相加成 f1、相除取 std。
/// pandas 的加法按索引并集对齐，某个子集自身覆盖不到的分钟在 f1 里是 NaN，
/// 所以真正参与 std 的只有「四个子集都覆盖到的分钟」。
fn std_pair(
    t: &[i64],
    turnover: &[f64],
    mask_su: &[bool],
    mask_bd: &[bool],
    mask_bu: &[bool],
    mask_sd: &[bool],
) -> (f64, f64) {
    let masks = [mask_su, mask_bd, mask_bu, mask_sd];
    let mut lo = [0i64; 4];
    let mut hi = [0i64; 4];
    let mut sums: Vec<Vec<f64>> = Vec::with_capacity(4);
    for (k, mask) in masks.iter().enumerate() {
        let first = mask.iter().position(|&b| b);
        let last = mask.iter().rposition(|&b| b);
        let (f, l) = match (first, last) {
            (Some(f), Some(l)) => (f, l),
            _ => return (f64::NAN, f64::NAN), // 空子集：pandas 侧 f1 全 NaN
        };
        lo[k] = t[f].div_euclid(MIN_US);
        hi[k] = t[l].div_euclid(MIN_US);
        let len = (hi[k] - lo[k] + 1) as usize;
        sums.push(vec![0.0f64; len]);
    }
    for i in 0..t.len() {
        let bin = t[i].div_euclid(MIN_US);
        for k in 0..4 {
            if masks[k][i] {
                sums[k][(bin - lo[k]) as usize] += turnover[i];
            }
        }
    }
    let a = *lo.iter().max().unwrap();
    let b = *hi.iter().min().unwrap();
    if a > b {
        return (f64::NAN, f64::NAN);
    }
    let mut r0: Vec<f64> = Vec::new();
    let mut r1: Vec<f64> = Vec::new();
    for bin in a..=b {
        let f1 = sums[0][(bin - lo[0]) as usize]
            + sums[1][(bin - lo[1]) as usize]
            + sums[2][(bin - lo[2]) as usize]
            + sums[3][(bin - lo[3]) as usize];
        if f1 == 0.0 {
            continue; // 0/0 = NaN，被 std 跳过
        }
        r0.push(sums[0][(bin - lo[0]) as usize] / f1);
        r1.push(sums[1][(bin - lo[1]) as usize] / f1);
    }
    (std_ddof1(&r0), std_ddof1(&r1))
}

pub fn compute_stock(code: &str, date: i64) -> Option<Vec<f32>> {
    let raw = read_rows(code, date)?;
    let mut w: Vec<Row> = raw
        .into_iter()
        .filter(|r| {
            let d = r.t.rem_euclid(DAY_US);
            d >= LD_0930 && d <= LD_1457
        })
        .collect();
    let n = w.len();
    if n == 0 {
        return Some(vec![f32::NAN; N_OUT]);
    }
    // 文件本身按 exchtime 升序；稳定排序保持同时刻的原有先后（与旧代码一致）
    w.sort_by_key(|r| r.t);

    let t: Vec<i64> = w.iter().map(|r| r.t).collect();

    // ---- 时段掩码与各时段总成交额 ----
    let mut segmask = vec![0u8; n];
    let mut tsum = [0.0f64; N_SEG];
    for i in 0..n {
        let d = t[i].rem_euclid(DAY_US);
        let mut m = 1u8;
        for s in 1..N_SEG {
            if d >= SEG_LO[s] && d <= SEG_HI[s] {
                m |= 1 << s;
            }
        }
        segmask[i] = m;
        for s in 0..N_SEG {
            if m & (1 << s) != 0 {
                tsum[s] += w[i].turnover;
            }
        }
    }

    // ---- 按订单号分组的成交额（全天窗口内） ----
    let mut ask_sum: HashMap<i64, f64> = HashMap::new();
    let mut bid_sum: HashMap<i64, f64> = HashMap::new();
    let mut first_a: HashMap<i64, usize> = HashMap::new();
    let mut first_b: HashMap<i64, usize> = HashMap::new();
    for i in 0..n {
        *ask_sum.entry(w[i].ask_order).or_insert(0.0) += w[i].turnover;
        *bid_sum.entry(w[i].bid_order).or_insert(0.0) += w[i].turnover;
        first_a.entry(w[i].ask_order).or_insert(i);
        first_b.entry(w[i].bid_order).or_insert(i);
    }
    let mut ask_keys: Vec<i64> = ask_sum.keys().copied().collect();
    ask_keys.sort_unstable();
    let mut bid_keys: Vec<i64> = bid_sum.keys().copied().collect();
    bid_keys.sort_unstable();
    let mut ask_vals: Vec<f64> = ask_keys.iter().map(|k| ask_sum[k]).collect();
    ask_vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mut bid_vals: Vec<f64> = bid_keys.iter().map(|k| bid_sum[k]).collect();
    bid_vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let ask_q = quantile_linear(&ask_vals, 0.3);
    let bid_q = quantile_linear(&bid_vals, 0.3);
    let ask_small: HashSet<i64> = ask_keys
        .iter()
        .filter(|&&k| ask_sum[&k] < ask_q)
        .copied()
        .collect();
    let bid_small: HashSet<i64> = bid_keys
        .iter()
        .filter(|&&k| bid_sum[&k] < bid_q)
        .copied()
        .collect();

    // drop_duplicates(keep='first') 对应的行号
    let mut ask_first: Vec<usize> = first_a.values().copied().collect();
    ask_first.sort_unstable();
    let mut bid_first: Vec<usize> = first_b.values().copied().collect();
    bid_first.sort_unstable();

    // ---- 各时段的「小单分母」 ----
    // 旧代码两套口径不一样，必须照抄：
    //   small_amount（全天）= 小单订单号的「分组总额」之和；
    //   small_amount_<时段>   = 该时段内所有「首次出现行」的 turnover 之和（不筛大小）
    let mut small_amount = [0.0f64; N_SEG];
    for &k in ask_keys.iter() {
        if ask_small.contains(&k) {
            small_amount[0] += ask_sum[&k];
        }
    }
    for &k in bid_keys.iter() {
        if bid_small.contains(&k) {
            small_amount[0] += bid_sum[&k];
        }
    }
    for s in 1..N_SEG {
        let mut acc = 0.0f64;
        for &i in ask_first.iter() {
            if segmask[i] & (1 << s) != 0 {
                acc += w[i].turnover;
            }
        }
        for &i in bid_first.iter() {
            if segmask[i] & (1 << s) != 0 {
                acc += w[i].turnover;
            }
        }
        small_amount[s] = acc;
    }

    let mut out: Vec<f64> = Vec::with_capacity(N_OUT);

    for tag in 0..N_TAG {
        // ---- chg_<tag> ----
        let mut chg = vec![f64::NAN; n];
        if tag < 4 {
            let k = (n as f64 * TAG_FRAC[tag]).trunc() as usize;
            for i in k..n {
                chg[i] = sign(w[i].price - w[i - k].price);
            }
        } else {
            let delta = TAG_DELTA[tag];
            for i in 0..n {
                let target = t[i] - delta;
                let p = t.partition_point(|&x| x <= target);
                if p > 0 {
                    chg[i] = sign(w[i].price - w[p - 1].price);
                }
            }
        }

        // ---- 订单号集合（取自全天的「首次出现行」） ----
        let mut s_bu: HashSet<i64> = HashSet::new();
        let mut s_sd: HashSet<i64> = HashSet::new();
        let mut s_su: HashSet<i64> = HashSet::new();
        for &i in bid_first.iter() {
            if chg[i] == -1.0 && w[i].flag == 66 {
                s_bu.insert(w[i].bid_order);
            }
        }
        for &i in ask_first.iter() {
            if chg[i] == -1.0 && w[i].flag == 83 {
                s_sd.insert(w[i].ask_order);
            }
            if chg[i] == 1.0 && w[i].flag == 66 {
                s_su.insert(w[i].ask_order);
            }
        }

        // ！！_small 版的行级筛选**只有**订单号归属，不再看该行自己的 flag/chg：
        //   旧代码是 df[df.ask_order.isin(asks[(asks.chg==1)&(asks.flag==66)].ask_order)]，
        //   flag/chg 条件是加在去重后的订单表 asks/bids 上的，不是加在行上的。
        let mut acc = [Acc::default(); N_SEG];
        for i in 0..n {
            let m = segmask[i];
            let c = chg[i];
            let f = w[i].flag;
            let is_su = f == 83 && c == 1.0;
            let is_bd = f == 66 && c == -1.0;
            let is_sd = f == 83 && c == -1.0;
            let r = &w[i];
            let su2 = s_su.contains(&r.ask_order) && bid_small.contains(&r.bid_order);
            let bd2 = s_bu.contains(&r.bid_order) && bid_small.contains(&r.bid_order);
            let sd2 = s_sd.contains(&r.ask_order) && ask_small.contains(&r.ask_order);
            if !(is_su || is_bd || is_sd || su2 || bd2 || sd2) {
                continue;
            }
            for s in 0..N_SEG {
                if m & (1 << s) == 0 {
                    continue;
                }
                let a = &mut acc[s];
                if is_su {
                    a.su_t += r.turnover;
                }
                if su2 {
                    a.su2_t += r.turnover;
                }
                if is_bd {
                    a.bd_t += r.turnover;
                }
                if bd2 {
                    a.bd2_t += r.turnover;
                }
                if is_sd {
                    a.sd_t += r.turnover;
                    a.sd_v += r.volume;
                    a.sd_p += r.price;
                    a.sd_n += 1.0;
                }
                if sd2 {
                    a.sd2_t += r.turnover;
                    a.sd2_v += r.volume;
                    a.sd2_p += r.price;
                    a.sd2_n += 1.0;
                }
            }
        }

        for s in 0..N_SEG {
            let a = acc[s];
            out.push(div(a.su_t - a.bd_t, tsum[s]));
            out.push(div(div(a.sd_t, a.sd_v), mean(a.sd_p, a.sd_n)));
            if s == 0 {
                let su = (0..n).map(|i| w[i].flag == 83 && chg[i] == 1.0).collect::<Vec<bool>>();
                let bd = (0..n).map(|i| w[i].flag == 66 && chg[i] == -1.0).collect::<Vec<bool>>();
                let bu = (0..n).map(|i| w[i].flag == 66 && chg[i] == 1.0).collect::<Vec<bool>>();
                let sd = (0..n).map(|i| w[i].flag == 83 && chg[i] == -1.0).collect::<Vec<bool>>();
                let turnover: Vec<f64> = w.iter().map(|r| r.turnover).collect();
                let (e1, e2) = std_pair(&t, &turnover, &su, &bd, &bu, &sd);
                out.push(e1);
                out.push(e2);
            }
        }
        for s in 0..N_SEG {
            let a = acc[s];
            out.push(div(a.su2_t - a.bd2_t, small_amount[s]));
            out.push(div(div(a.sd2_t, a.sd2_v), mean(a.sd2_p, a.sd2_n)));
        }
    }

    debug_assert_eq!(out.len(), N_OUT);
    Some(out.into_iter().map(|v| v as f32).collect())
}

pub fn compute_cross(_date: i64) -> Option<(Vec<String>, Vec<f32>)> {
    None
}
