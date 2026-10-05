//! hm7「后继无人」—— 旧框架（design_whatever_out.TickDataBase）的 Rust 复现。
//!
//! 旧文件：/home/chenzongwei/pythoncode/成果hm7_后继无人/hm7_okay.py
//!   run_base 的 Go.get_factor，参数 get_many=0 / read_full=0 / change_time=1 / use_old=True，
//!   逐股系列（check_my_kind == "level2"），**只用逐笔成交**，不碰盘口。
//!   （use_old 只影响旧框架「增量读取已有 base」那条分支，对单股单日的取值没有影响。）
//!
//! ════════════════════════════════════════════════════════════════════
//! 一、旧代码逐步对照（写代码前先抄在这里，方便复核）
//! ════════════════════════════════════════════════════════════════════
//! 1) df = read_trade_data_single(with_retreat=0)
//!    · 读逐笔 CSV（usecols 挑出 exchtime / volume / flag / ask_order / bid_order）,
//!      deal_data 把 exchtime 从 epoch 微秒换成 Timestamp（+8 小时，北京钟点）。
//!    · with_retreat=0 → 丢掉 flag == 32 的行。注意 pandas 是**按数值**比的
//!      （2025 年的明细写的是 "32.0"，数值比照样命中），所以本文件也按 f64 数值比，
//!      不能用 fast_csv_reader 那个按字节比 "32" 的 with_retreat。
//! 2) df = df.set_index(['exchtime']).between_time('09:30:00','14:57:00')，两端闭区间。
//! 3) 三个切法：
//!      按方向：df_buy = flag==66、df_sell = flag==83（其余 flag 只进「不分子集」的那份）
//!      按窗口：open30 [09:30,10:00]、close30 [14:30,14:57]、morning [09:30,11:30]、
//!              afternoon [13:00,14:57]，再加「全天」= 第 2 步之后的整份
//!    「全天 / open30 / close30 / morning / afternoon」×「不分子集 / _buy / _sell」
//!    = 15 份，res 里的拼接顺序就是 all, buy_all, sell_all, open30, buy_open30, ...,
//!    afternoon, buy_afternoon, sell_afternoon。
//! 4) 每份各做两次 groupby：
//!      asks = 该份.groupby('ask_order').volume.sum()
//!      bids = 该份.groupby('bid_order').volume.sum()
//!    · groupby 默认 dropna=True，key 为 NaN 的行不参与分组（本文件照做）。
//!    · 结果是一列「每个挂单号在该窗口内的成交量之和」。
//! 5) 三档参数 M = 1000 / 5000 / 10000（名字里写的是 param10 / param50 / param100）：
//!      asks_here = asks[asks.volume % M == 0]，bids_here 同理
//!      先把两份 volume 各自排序（下面说明为什么），再算：
//!        turnover_sum    = (Σask - Σbid) / (Σask + Σbid)
//!        turnover_mean   = (mean(ask) - mean(bid)) / (...)
//!        turnover_median = (median(ask) - median(bid)) / (...)
//!        turnover_std    = (std(ask) - std(bid)) / (...)   ← ddof=1，样本 <2 给 NaN
//!        cjbs            = (N_ask - N_bid) / (N_ask + N_bid)
//!    · 空集合：Σ = 0，mean/median/std = NaN，N = 0；np.divide 里 0/0 给 NaN、
//!      x/0 给 ±inf，NaN 传染。本文件的 np_div 逐条复刻这个行为。
//! 6) names = 225 个，顺序 = param（10/50/100）→ 窗口（all/open30/close30/morning/afternoon）
//!      → 方向（无后缀/_buy/_sell）→ 统计量（sum/mean/median/std/cjbs）。
//!    与 get_factor 返回的 res_10 + res_50 + res_100 逐个对齐。
//!
//! ════════════════════════════════════════════════════════════════════
//! 二、精度与确定性
//! ════════════════════════════════════════════════════════════════════
//! · volume 是整数，挂单量之和也当整数用 f64 存（远小于 2^53，精确）。
//!   「% M == 0」按 f64 取余做，整数值上等价于整数取余，也不会和 pandas 的
//!   int64 取余有差别（numpy 的浮点 % 与 Rust 的 % 在非负数上一致）。
//! · 每份的挂单量向量**先排序再用**：一是 median 需要有序，二是 HashMap 迭代顺序
//!   每次进程启动都不同，排序后才能保证同样输入永远出同样结果。
//! · mean 用「整数和 / 个数」（和 pandas 对 int64 的做法一致，先求和再除），
//!   std 用两遍法（先求均值再求平方偏差和），和 pandas 的 nanstd 一致。
//!
//! ════════════════════════════════════════════════════════════════════
//! 三、names_single 的 225 个名字互不重复，不需要 _dup 后缀；
//!    names_save 里没有 `_fold` 结尾的名字（旧文件没有 get_abs 那一步）。

use std::collections::HashMap;
use std::fs::File;
use std::io::Read;

/// 逐笔 CSV 的列索引（0 基，与 fast_csv_reader 的常量一致；2018 及更早末尾多一列 num）
const TRD_COL_EXCHTIME: usize = 4;
const TRD_COL_VOLUME: usize = 8;
const TRD_COL_FLAG: usize = 10;
const TRD_COL_ASK_ORDER: usize = 13;
const TRD_COL_BID_ORDER: usize = 14;
const TRD_NCOL: usize = 15;

const SEC_US: i64 = 1_000_000;
const DAY_US: i64 = 86_400 * SEC_US;
const OFFSET_8H: i64 = 8 * 3600 * SEC_US;

const LD_0930: i64 = (9 * 3600 + 30 * 60) * SEC_US;
const LD_1000: i64 = 10 * 3600 * SEC_US;
const LD_1130: i64 = (11 * 3600 + 30 * 60) * SEC_US;
const LD_1300: i64 = 13 * 3600 * SEC_US;
const LD_1430: i64 = (14 * 3600 + 30 * 60) * SEC_US;
const LD_1457: i64 = (14 * 3600 + 57 * 60) * SEC_US;

/// 窗口位（bit0..bit4 = all / open30 / close30 / morning / afternoon）
const WINDOWS: [(u8, &str); 5] = [
    (1, "_all"),
    (2, "_open30"),
    (4, "_close30"),
    (8, "_morning"),
    (16, "_afternoon"),
];
/// 方向位（bit0..bit2 = 不分子集 / _buy / _sell）
const SIDES: [(u8, &str); 3] = [(1, ""), (2, "_buy"), (4, "_sell")];
/// 三档参数：位掩码 M 与名字后缀
const PARAMS: [(f64, &str); 3] = [(1000.0, "_param10"), (5000.0, "_param50"), (10000.0, "_param100")];
/// 五个统计量（顺序即旧代码 res_*_* 里的顺序）
const STATS: [&str; 5] = ["turnover_sum", "turnover_mean", "turnover_median", "turnover_std", "cjbs"];

const PER_COMBO: usize = 3 * 5; // 一个 (窗口, 方向) 组合 × 3 档参数 × 5 个统计量
pub const N_OUT: usize = 15 * PER_COMBO; // 225

pub fn names() -> Vec<String> {
    let mut out: Vec<String> = Vec::with_capacity(N_OUT);
    for (_, psuf) in PARAMS.iter() {
        for (_, wsuf) in WINDOWS.iter() {
            for (_, ssuf) in SIDES.iter() {
                for st in STATS.iter() {
                    out.push(format!("{}{}{}{}", st, ssuf, wsuf, psuf));
                }
            }
        }
    }
    debug_assert_eq!(out.len(), N_OUT);
    out
}

// ============================================================================
// CSV 读取
// ============================================================================

#[inline]
fn parse_f64(b: &[u8]) -> f64 {
    if b.is_empty() {
        return f64::NAN;
    }
    std::str::from_utf8(b)
        .ok()
        .and_then(|s| s.trim().parse::<f64>().ok())
        .unwrap_or(f64::NAN)
}

fn transaction_path(code: &str, date: i64) -> Option<std::path::PathBuf> {
    let filename = format!("{}_{}_transaction.csv", code, date);
    for root in crate::data_paths::level2_roots() {
        let p = root.join(date.to_string()).join("transaction").join(&filename);
        if p.exists() {
            return Some(p);
        }
    }
    None
}

/// 一行逐笔（已经过滤完 flag==32 与 [09:30,14:57]）。
struct Row {
    wm: u8,
    sm: u8,
    ask: f64,
    bid: f64,
    vol: f64,
}

fn read_rows(code: &str, date: i64) -> Option<Vec<Row>> {
    let path = transaction_path(code, date)?;
    let mut rows: Vec<Row> = Vec::new();
    let mut content = String::new();
    File::open(&path).ok()?.read_to_string(&mut content).ok()?;

    let mut lines = content.lines();
    lines.next(); // 表头
    for raw in lines {
        let line = raw.trim_end_matches('\r');
        if line.is_empty() {
            continue;
        }
        // 定长栈缓冲切列，避免每行一次 Vec 分配（一只股票一天几十万行）
        let bytes = line.as_bytes();
        let mut f: [&[u8]; TRD_NCOL] = [&[][..]; TRD_NCOL];
        let mut n = 0usize;
        let mut start = 0usize;
        for (i, &b) in bytes.iter().enumerate() {
            if b == b',' {
                if n < TRD_NCOL {
                    f[n] = &bytes[start..i];
                }
                n += 1;
                start = i + 1;
            }
        }
        if n < TRD_NCOL {
            f[n] = &bytes[start..];
        }
        if n + 1 < TRD_NCOL {
            continue;
        }
        let ex = parse_f64(f[TRD_COL_EXCHTIME]);
        if !ex.is_finite() {
            continue;
        }
        let day = (ex.round() as i64 + OFFSET_8H).rem_euclid(DAY_US);
        if day < LD_0930 || day > LD_1457 {
            continue;
        }
        let flag = parse_f64(f[TRD_COL_FLAG]);
        // with_retreat=0：按数值丢掉 flag==32（NaN != 32，保留）
        if flag == 32.0 {
            continue;
        }
        let mut wm: u8 = 1; // all
        if day <= LD_1000 {
            wm |= 2;
        }
        if day >= LD_1430 {
            wm |= 4;
        }
        if day <= LD_1130 {
            wm |= 8;
        }
        if day >= LD_1300 {
            wm |= 16;
        }
        let mut sm: u8 = 1; // 不分子集
        if flag == 66.0 {
            sm |= 2;
        }
        if flag == 83.0 {
            sm |= 4;
        }
        rows.push(Row {
            wm,
            sm,
            ask: parse_f64(f[TRD_COL_ASK_ORDER]),
            bid: parse_f64(f[TRD_COL_BID_ORDER]),
            vol: parse_f64(f[TRD_COL_VOLUME]),
        });
    }
    if rows.is_empty() {
        return None;
    }
    Some(rows)
}

// ============================================================================
// pandas 聚合语义
// ============================================================================

/// np.divide：0/0 给 NaN，x/0（x≠0）给 ±inf，NaN 传染。
#[inline]
fn np_div(a: f64, b: f64) -> f64 {
    if b == 0.0 {
        if a == 0.0 || a.is_nan() {
            return f64::NAN;
        }
        return if a > 0.0 { f64::INFINITY } else { f64::NEG_INFINITY };
    }
    a / b
}

#[inline]
fn nan_sum(v: &[f64]) -> f64 {
    let mut s = 0.0f64;
    for &x in v.iter() {
        if !x.is_nan() {
            s += x;
        }
    }
    s
}

fn nan_mean(v: &[f64]) -> f64 {
    let mut s = 0.0f64;
    let mut n = 0usize;
    for &x in v.iter() {
        if !x.is_nan() {
            s += x;
            n += 1;
        }
    }
    if n == 0 {
        f64::NAN
    } else {
        s / n as f64
    }
}

fn nan_median(v: &[f64]) -> f64 {
    let ok: Vec<f64> = v.iter().copied().filter(|x| !x.is_nan()).collect();
    let n = ok.len();
    if n == 0 {
        return f64::NAN;
    }
    if n % 2 == 1 {
        ok[n / 2]
    } else {
        // 输入已排序（见 combo_values），这里直接取中间两个
        (ok[n / 2 - 1] + ok[n / 2]) / 2.0
    }
}

fn nan_std(v: &[f64]) -> f64 {
    let ok: Vec<f64> = v.iter().copied().filter(|x| !x.is_nan()).collect();
    let n = ok.len();
    if n <= 1 {
        return f64::NAN;
    }
    let mean = nan_sum(&ok) / n as f64;
    let mut sq = 0.0f64;
    for &x in ok.iter() {
        sq += (x - mean) * (x - mean);
    }
    (sq / (n - 1) as f64).sqrt()
}

/// f64 的位模式当 HashMap key；-0.0 归一成 0.0（pandas 里 -0.0 与 0.0 是同一个 key）。
#[inline]
fn key_bits(x: f64) -> u64 {
    if x == 0.0 {
        0u64
    } else {
        x.to_bits()
    }
}

/// 一个 (窗口, 方向) 组合的 15 个值：3 档参数 × 5 个统计量。
fn combo_values(rows: &[Row], wbit: u8, sbit: u8) -> [f64; PER_COMBO] {
    let mut ask: HashMap<u64, f64> = HashMap::new();
    let mut bid: HashMap<u64, f64> = HashMap::new();
    for r in rows.iter() {
        if r.wm & wbit == 0 || r.sm & sbit == 0 {
            continue;
        }
        // groupby 默认 dropna=True：key 为 NaN 的行不参与分组
        if !r.ask.is_nan() {
            *ask.entry(key_bits(r.ask)).or_insert(0.0) += r.vol;
        }
        if !r.bid.is_nan() {
            *bid.entry(key_bits(r.bid)).or_insert(0.0) += r.vol;
        }
    }
    // 先排序再用：median 要有序，同时保证 HashMap 迭代顺序不影响结果
    let mut av: Vec<f64> = ask.into_values().collect();
    let mut bv: Vec<f64> = bid.into_values().collect();
    av.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    bv.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let mut out = [f64::NAN; PER_COMBO];
    for (pi, &(m, _)) in PARAMS.iter().enumerate() {
        let af: Vec<f64> = av.iter().copied().filter(|v| v % m == 0.0).collect();
        let bf: Vec<f64> = bv.iter().copied().filter(|v| v % m == 0.0).collect();
        let sa = nan_sum(&af);
        let sb = nan_sum(&bf);
        let ma = nan_mean(&af);
        let mb = nan_mean(&bf);
        let da = nan_median(&af);
        let db = nan_median(&bf);
        let ta = nan_std(&af);
        let tb = nan_std(&bf);
        let na = af.len() as f64;
        let nb = bf.len() as f64;
        let base = pi * 5;
        out[base] = np_div(sa - sb, sa + sb);
        out[base + 1] = np_div(ma - mb, ma + mb);
        out[base + 2] = np_div(da - db, da + db);
        out[base + 3] = np_div(ta - tb, ta + tb);
        out[base + 4] = np_div(na - nb, na + nb);
    }
    out
}

// ============================================================================
// 逐股入口
// ============================================================================

/// 一只股票的一天：算出 225 个因子值。算不出来返回 None（上游补整条 NaN）。
pub fn compute_stock(code: &str, date: i64) -> Option<Vec<f32>> {
    let rows = read_rows(code, date)?;
    // 15 个组合各自 15 个值，按组合下标存
    let mut combos: Vec<[f64; PER_COMBO]> = Vec::with_capacity(15);
    for (wbit, _) in WINDOWS.iter() {
        for (sbit, _) in SIDES.iter() {
            combos.push(combo_values(&rows, *wbit, *sbit));
        }
    }
    // 旧代码的顺序是 param → 组合（all/buy_all/sell_all/...），不是组合 → param
    let mut out: Vec<f64> = Vec::with_capacity(N_OUT);
    for pi in 0..PARAMS.len() {
        for c in combos.iter() {
            out.extend_from_slice(&c[pi * 5..pi * 5 + 5]);
        }
    }
    debug_assert_eq!(out.len(), N_OUT);
    Some(out.into_iter().map(|v| v as f32).collect())
}

/// hm7 全是逐股因子，没有横截面版本。
pub fn compute_cross(_date: i64) -> Option<(Vec<String>, Vec<f32>)> {
    None
}
