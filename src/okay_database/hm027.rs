//! hm27 漫长订单及其衍生物 —— 旧框架（design_whatever_out.TickDataBase）的 Rust 复现。
//!
//! 旧文件：/home/chenzongwei/pythoncode/成果hm27_漫长订单及其衍生物/hm27_okay.py
//! （run_base 里的 Go 类；run_factor_out 的实参是 get_many=0 / read_full=0 / change_time=1）
//!
//! ════════════════════════════════════════════════════════════════════
//! 一、旧代码逐步对照（改代码前先核这一节）
//! ════════════════════════════════════════════════════════════════════
//! 1) df = read_trade_data_single(with_retreat=0)   —— 丢掉 flag==32 的撤单行
//!    change_time=1 走 deal_data，把 exchtime（微秒整数）变成
//!    Timestamp = exchtime/1e6 秒 + 1970-01-01 08:00:00（北京时区）。
//!    本文件不做 adjust_afternoon，窗口用真实钟点。
//!
//! 2) df = df.set_index('exchtime').between_time('09:30:00','14:57:00')
//!    df1 = df.between_time('09:30:00','11:30:00')          ← 上午，原时间
//!    df2 = df.between_time('13:00:00','14:57:00')          ← 下午
//!    df2.index = df2.index - Timedelta(hours=1, minutes=30) ← 下午整体回拨 1.5 小时
//!    df = pd.concat([df1, df2]).reset_index()
//!    ！！工作集是「上午 ∪ 下午（回拨后）」：11:30:00.000001 ~ 12:59:59.999999
//!      之间的成交**不参与**任何计算（既不在 df1 也不在 df2）。
//!    ！！回拨后的时间列名为 exchtime（reset_index 把索引名还原成列），
//!      后面所有 groupby 用的都是这个回拨后的时间。
//!
//! 3) 每个「订单号」的存活时长（秒）与成交额：
//!        asks_last/minus_first  = groupby('ask_order').exchtime 的 max - min
//!        bids_last/minus_first  = groupby('bid_order').exchtime 的 max - min
//!        ask_duration / bid_duration = 上面差值的 .dt.total_seconds()
//!        asks_amount  = groupby('ask_order').turnover.sum()
//!        bids_amount  = groupby('bid_order').turnover.sum()
//!    再按订单号 merge 回每一行（同一订单号的所有行共享同一份时长与成交额）。
//!    ask_order / bid_order 为 0 的行也照样成组（0 是「无对应委托」的占位值，
//!    不是缺失），groupby 不会把它们丢掉。
//!
//! 4) 每一行 6 个布尔标记：
//!        longsell_1p0/1p5/2p0 = ask_duration > mean + 1.0/1.5/2.0 * std
//!        longbuy_1p0/1p5/2p0  = bid_duration > mean + 1.0/1.5/2.0 * std
//!        largesell = asks_amount > asks_amount.quantile(0.8)
//!        largebuy  = bids_amount > bids_amount.quantile(0.8)
//!    （mean/std/quantile 都是 pandas 的，跳过 NaN，std 是 ddof=1，
//!      quantile 是默认的线性插值）
//!    ！！名字里的 largesell 对应 asks_amount、largebuy 对应 bids_amount。
//!
//! 5) 每个组合输出 exp(该组合的行成交额之和 / 全天成交额之和)：
//!        组合 = 卖出是否 longsell（1/0）× 档位（1p0/1p5/2p0）
//!             × 买入是否 longbuy（1/0）× largebuy(1/0) × largesell(1/0)
//!             × flag(66/83)，共 2×3×2×2×2×2 = 96 个。
//!    names_single 的顺序是：
//!        for 卖出族 in [longsell, shortsell]:
//!          for 档位 in [1p0, 1p5, 2p0]:
//!            for flag in [66, 83]:
//!              for largebuy in [large, small]:
//!                for largesell in [large, small]:
//!                  for 买入 in [longbuy, shortbuy]:        ← 共 16 个/档位
//!    已经与旧文件的 names 列表逐个核对过。
//!
//! ════════════════════════════════════════════════════════════════════
//! 二、一处已知的近似（不影响本机数据，写在这里免得以后重复排查）
//! ════════════════════════════════════════════════════════════════════
//! 旧代码的 Timestamp 是 exchtime/1e6 这个 **float64 秒** 再乘 1e9 转成纳秒的，
//! 1.74e9 秒量级上 float64 的分辨率约 2.4e-7 秒，所以旧值相对真实微秒有前后
//! 0.25 微秒的舍入。本文件直接用精确微秒。只有落在窗口边界 0.25 微秒以内的成交
//! 才会因此被纳入/排除不同，实测 000002@20180601、000001@20250303 在
//! 09:30:00 与 14:57:00 前后 100 微秒内都没有成交；订单时长是秒级差值，
//! 0.25 微秒的偏差相对量级 1e-7，也远小于 1e-4 的验收阈值。
//!
//! ════════════════════════════════════════════════════════════════════
//! 三、其它对齐细节
//! ════════════════════════════════════════════════════════════════════
//! · 撤单过滤按规范自己 filter flag != 32（不靠 read_trade_fast_inner 的
//!   with_retreat：库里那条判断按字节比 "32"，2025 年前后明细里写的是 "32.0"）。
//! · turnover 用 f64 解析（fast_csv_reader 的 TradeRecord.turnover 是 f32，
//!   全天成交额求和与分组求和都会被 f32 打偏，这里不用它）。
//! · 全天成交额之和用 numpy 的成对求和（pandas Series.sum 走的就是它）。
//! · 窗口内一行都没有时旧代码会在 `.dt` 那一步抛异常、被 except 吞掉返回 None，
//!   这里同样返回 None。

use std::collections::HashMap;
use std::fs::File;
use std::io::Read;

/// 逐笔 CSV 的列索引（0 基，与 fast_csv_reader 一致；2018 年及以前末尾多一列 num）
const TRD_COL_EXCHTIME: usize = 4;
const TRD_COL_TURNOVER: usize = 9;
const TRD_COL_FLAG: usize = 10;
const TRD_COL_ASK_ORDER: usize = 13;
const TRD_COL_BID_ORDER: usize = 14;
const TRD_NCOL: usize = 15;

const US: i64 = 1_000_000;
const DAY_US: i64 = 86_400_000_000;
const H8: i64 = 8 * 3600 * US;
const SHIFT_AFTERNOON: i64 = 90 * 60 * US; // 下午回拨 1 小时 30 分

const T_0930: i64 = (9 * 3600 + 30 * 60) * US;
const T_1130: i64 = (11 * 3600 + 30 * 60) * US;
const T_1300: i64 = 13 * 3600 * US;
const T_1457: i64 = (14 * 3600 + 57 * 60) * US;

const SELLS: [&str; 2] = ["longsell", "shortsell"];
const LEVELS: [(&str, f64); 3] = [("1p0", 1.0), ("1p5", 1.5), ("2p0", 2.0)];
const FLAGS: [i64; 2] = [66, 83];
const PER_LEVEL: usize = 16; // 2 flag × 2 largebuy × 2 largesell × 2 buy
pub const N_OUT: usize = 2 * 3 * PER_LEVEL;

/// 96 个因子名，顺序与旧 names_single 完全一致
pub fn names() -> Vec<String> {
    let mut out = Vec::with_capacity(N_OUT);
    for sell in SELLS.iter() {
        for (lvl, _) in LEVELS.iter() {
            for fl in FLAGS.iter() {
                for lb in ["large", "small"].iter() {
                    for ls in ["large", "small"].iter() {
                        for buy in ["longbuy", "shortbuy"].iter() {
                            out.push(format!(
                                "{}_{}_{}_{}_{}buy_{}sell_{}",
                                sell, lvl, buy, lvl, lb, ls, fl
                            ));
                        }
                    }
                }
            }
        }
    }
    out
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

/// 一条逐笔成交（只留计算要用的列）
#[derive(Clone, Copy, Default)]
struct TradeRow {
    time_us: i64,
    turnover: f64,
    flag: f64,
    ask_order: i64,
    bid_order: i64,
}

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
            time_us: parse_i64_at(f[TRD_COL_EXCHTIME]),
            turnover: parse_f64_at(f[TRD_COL_TURNOVER]),
            flag: parse_f64_at(f[TRD_COL_FLAG]),
            ask_order: parse_i64_at(f[TRD_COL_ASK_ORDER]),
            bid_order: parse_i64_at(f[TRD_COL_BID_ORDER]),
        });
    }
    Some(out)
}

// ============================================================================
// numpy 成对求和（pairwise_sum_FLOAT，blocksize 128、8 路展开）
// ============================================================================

fn pairwise_sum(a: &[f64]) -> f64 {
    let n = a.len();
    if n < 8 {
        let mut res = -0.0f64;
        for &x in a.iter() {
            res += x;
        }
        return res;
    }
    if n <= 128 {
        let mut r = [0.0f64; 8];
        r.copy_from_slice(&a[..8]);
        let mut i = 8usize;
        let stop = n - (n % 8);
        while i < stop {
            for k in 0..8 {
                r[k] += a[i + k];
            }
            i += 8;
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        while i < n {
            res += a[i];
            i += 1;
        }
        return res;
    }
    let mut n2 = n / 2;
    n2 -= n2 % 8;
    pairwise_sum(&a[..n2]) + pairwise_sum(&a[n2..])
}

/// pandas 的 mean/std（跳过 NaN，std 用 ddof=1）
fn nan_mean_std(v: &[f64]) -> (f64, f64) {
    let mut buf: Vec<f64> = Vec::with_capacity(v.len());
    for &x in v.iter() {
        if !x.is_nan() {
            buf.push(x);
        }
    }
    let cnt = buf.len();
    if cnt == 0 {
        return (f64::NAN, f64::NAN);
    }
    let avg = pairwise_sum(&buf) / (cnt as f64);
    let mut sq: Vec<f64> = Vec::with_capacity(cnt);
    for &x in buf.iter() {
        let d = avg - x;
        sq.push(d * d);
    }
    let std = (pairwise_sum(&sq) / ((cnt as f64) - 1.0)).sqrt();
    (avg, std)
}

/// pandas quantile(q)（默认线性插值）。v 里可以有 NaN（跳过），函数会自己拷贝排序。
fn nan_quantile(v: &[f64], q: f64) -> f64 {
    let mut arr: Vec<f64> = v.iter().copied().filter(|x| !x.is_nan()).collect();
    let n = arr.len();
    if n == 0 {
        return f64::NAN;
    }
    arr.sort_by(|a, b| a.total_cmp(b));
    let virtual_index = (n as f64 - 1.0) * q;
    if virtual_index >= (n as f64 - 1.0) {
        return arr[n - 1];
    }
    let lo = virtual_index.floor();
    let t = virtual_index - lo;
    let a = arr[lo as usize];
    let b = arr[(lo + 1.0) as usize];
    // numpy 的 _lerp：t >= 0.5 走 b - (b-a)*(1-t)
    if t >= 0.5 {
        b - (b - a) * (1.0 - t)
    } else {
        a + (b - a) * t
    }
}

/// 逐股入口：算不出来返回 None（上游补 NaN，保证因子个数恒定）。
pub fn compute_stock(code: &str, date: i64) -> Option<Vec<f32>> {
    let rows_all = read_trade_f64(code, date)?;
    if rows_all.is_empty() {
        return None;
    }

    // ── 丢掉撤单行，再按「上午 ∪ 下午（回拨 1.5 小时）」取窗口
    let mut time_key: Vec<i64> = Vec::new();
    let mut turnover: Vec<f64> = Vec::new();
    let mut flag: Vec<f64> = Vec::new();
    let mut ask_o: Vec<i64> = Vec::new();
    let mut bid_o: Vec<i64> = Vec::new();
    for r in rows_all.iter() {
        if r.flag == 32.0 {
            continue;
        }
        let t = r.time_us + H8;
        let d = t.rem_euclid(DAY_US);
        let key = if d >= T_0930 && d <= T_1130 {
            d
        } else if d >= T_1300 && d <= T_1457 {
            d - SHIFT_AFTERNOON
        } else {
            continue;
        };
        time_key.push(key);
        turnover.push(r.turnover);
        flag.push(r.flag);
        ask_o.push(r.ask_order);
        bid_o.push(r.bid_order);
    }
    let n = time_key.len();
    if n == 0 {
        return None;
    }

    // ── 按订单号聚合：时长（秒）与成交额之和
    fn group_dur_amt(
        keys: &[i64],
        times: &[i64],
        turn: &[f64],
    ) -> HashMap<i64, (f64, f64)> {
        let mut acc: HashMap<i64, (i64, i64, f64)> = HashMap::new();
        for i in 0..keys.len() {
            let e = acc
                .entry(keys[i])
                .or_insert((times[i], times[i], 0.0));
            if times[i] < e.0 {
                e.0 = times[i];
            }
            if times[i] > e.1 {
                e.1 = times[i];
            }
            e.2 += turn[i];
        }
        acc.into_iter()
            .map(|(k, (lo, hi, s))| (k, ((hi - lo) as f64 / 1e6, s)))
            .collect()
    }

    let ask_map = group_dur_amt(&ask_o, &time_key, &turnover);
    let bid_map = group_dur_amt(&bid_o, &time_key, &turnover);

    let mut ask_dur = vec![f64::NAN; n];
    let mut ask_amt = vec![f64::NAN; n];
    let mut bid_dur = vec![f64::NAN; n];
    let mut bid_amt = vec![f64::NAN; n];
    for i in 0..n {
        if let Some(&(d, a)) = ask_map.get(&ask_o[i]) {
            ask_dur[i] = d;
            ask_amt[i] = a;
        }
        if let Some(&(d, a)) = bid_map.get(&bid_o[i]) {
            bid_dur[i] = d;
            bid_amt[i] = a;
        }
    }

    let total_turnover = pairwise_sum(&turnover);
    let (ask_dur_mean, ask_dur_std) = nan_mean_std(&ask_dur);
    let (bid_dur_mean, bid_dur_std) = nan_mean_std(&bid_dur);
    let ask_q80 = nan_quantile(&ask_amt, 0.8);
    let bid_q80 = nan_quantile(&bid_amt, 0.8);

    // 3 个档位的阈值
    let mut ask_thr = [0.0f64; 3];
    let mut bid_thr = [0.0f64; 3];
    for (k, (_, mult)) in LEVELS.iter().enumerate() {
        ask_thr[k] = ask_dur_mean + ask_dur_std * mult;
        bid_thr[k] = bid_dur_mean + bid_dur_std * mult;
    }

    let mut out: Vec<f32> = Vec::with_capacity(N_OUT);
    for sell in SELLS.iter() {
        let want_long_sell = *sell == "longsell";
        for (k, _) in LEVELS.iter().enumerate() {
            // 先算好这一档每行的两个布尔（与旧代码 astype(int) 后 ==1 / ==0 等价）
            let is_long_sell: Vec<bool> = ask_dur.iter().map(|&d| d > ask_thr[k]).collect();
            let is_long_buy: Vec<bool> = bid_dur.iter().map(|&d| d > bid_thr[k]).collect();
            for fl in FLAGS.iter() {
                for lb in ["large", "small"].iter() {
                    let want_large_buy = *lb == "large";
                    for ls in ["large", "small"].iter() {
                        let want_large_sell = *ls == "large";
                        for buy in ["longbuy", "shortbuy"].iter() {
                            let want_long_buy = *buy == "longbuy";
                            let mut s = 0.0f64;
                            for i in 0..n {
                                if flag[i] != (*fl as f64) {
                                    continue;
                                }
                                if is_long_sell[i] != want_long_sell {
                                    continue;
                                }
                                if is_long_buy[i] != want_long_buy {
                                    continue;
                                }
                                if (ask_amt[i] > ask_q80) != want_large_sell {
                                    continue;
                                }
                                if (bid_amt[i] > bid_q80) != want_large_buy {
                                    continue;
                                }
                                s += turnover[i];
                            }
                            out.push((s / total_turnover).exp() as f32);
                        }
                    }
                }
            }
        }
    }
    debug_assert_eq!(out.len(), N_OUT);
    Some(out)
}

/// hm27 全是逐股因子，没有横截面版本。
pub fn compute_cross(_date: i64) -> Option<(Vec<String>, Vec<f32>)> {
    None
}
