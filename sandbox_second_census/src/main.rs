//! 第 06 期：成交落在每分钟的第几秒（sandbox）
//!
//! 口径
//! - 记录：flag 66（主买）或 83（主卖）且 turnover>0。flag 32 是撤单，剔除。
//! - 时段：连续竞价 9:30:00-11:30:00 与 13:00:00-14:57:00，两端都算，共 14,222 秒。
//!   上午 7,201 秒（9:30:00 到 11:30:00），下午 7,021 秒（13:00:00 到 14:57:00）。
//!   第 0 秒因此一天出现 239 次（上午 121 加下午 118），其余 59 个秒位各 237 次。
//! - 秒位：交易所时间戳的秒字段 0-59，不看分钟。
//! - 刻度：时间戳截到 10 毫秒。同一刻度上的多条记录算一次撮合。
//! - 分段：开盘 5 分钟、上午其余、午后、最后 5 分钟，四段。
//!
//! 输出
//! - cnt[60]      每个秒位的记录数
//! - tick[60]     每个秒位上的不同刻度数
//! - amt60[60]    每个秒位的成交金额（元）
//! - vol60[60]    每个秒位的成交股数
//! - cnt2[60]     每个秒位「每分钟笔数」的平方和，配 occ 可算分钟间的方差
//! - amt2[60]     同上，金额的平方和
//! - seg[4][60]   四段的每个秒位记录数
//!
//! 构建：cd sandbox_second_census && cargo build --release
//! 运行：./target/release/second_census 20240329 > out/20240329.json
//!       ./target/release/second_census 20240329 30   # 只跑前 30 只，用于小样本核对
mod fast_csv_reader;
use fast_csv_reader::read_ticks;
use rayon::prelude::*;
use serde::Serialize;

const N_SS: usize = 60;
const N_MIN: usize = 237;
const AM: usize = 7201;
const N_SEC: usize = 14222;
const SEC_0930: i64 = 34_200;
const SEC_1130: i64 = 41_400;
const SEC_1300: i64 = 46_800;
const SEC_1457: i64 = 53_820;

#[derive(Serialize)]
struct Meta {
    n_sec: usize,
    seg_s: Vec<u32>,
    occ: Vec<u32>,
    occ_seg: Vec<Vec<u32>>,
}

#[derive(Serialize)]
struct Rec {
    code: String,
    s: u64,
    e: u64,
    amt: f64,
    vol: f64,
    cnt: Vec<u32>,
    tick: Vec<u32>,
    amt60: Vec<f64>,
    vol60: Vec<f64>,
    cnt2: Vec<u64>,
    amt2: Vec<f64>,
    share1: Vec<f64>,
    share2: Vec<f64>,
    seg: Vec<Vec<u32>>,
}

#[derive(Serialize)]
struct Out {
    date: i64,
    n_codes: usize,
    n_valid: usize,
    n_fail: usize,
    meta: Meta,
    recs: Vec<Rec>,
}

#[inline]
fn seg_of(i: usize) -> usize {
    if i < 300 {
        0
    } else if i < AM {
        1
    } else if i < AM + 6720 {
        2
    } else {
        3
    }
}

/// 分钟序号。收盘那两秒（11:30:00 与 14:57:00）并入各自的上一分钟。
#[inline]
fn min_of(i: usize) -> usize {
    if i < AM {
        (i / 60).min(119)
    } else {
        120 + ((i - AM) / 60).min(116)
    }
}

#[inline]
fn sec_of(i: usize) -> i64 {
    if i < AM {
        SEC_0930 + i as i64
    } else {
        SEC_1300 + (i - AM) as i64
    }
}

fn meta() -> Meta {
    let mut occ = vec![0u32; N_SS];
    let mut occ_seg = vec![vec![0u32; N_SS]; 4];
    let mut seg_s = vec![0u32; 4];
    for i in 0..N_SEC {
        let ss = (sec_of(i) % 60) as usize;
        let g = seg_of(i);
        occ[ss] += 1;
        occ_seg[g][ss] += 1;
        seg_s[g] += 1;
    }
    Meta { n_sec: N_SEC, seg_s, occ, occ_seg }
}

fn read_one(date: i64, code: &str) -> Option<Rec> {
    let ticks = read_ticks(code, date).ok()?;
    if ticks.is_empty() {
        return None;
    }
    let mut cnt_s = vec![0u32; N_SEC];
    let mut tick_s = vec![0u32; N_SEC];
    let mut amt_s = vec![0f64; N_SEC];
    let mut vol_s = vec![0f64; N_SEC];
    let mut s = 0u64;
    let mut e = 0u64;
    let mut amt = 0.0f64;
    let mut vol = 0.0f64;
    let mut last: i64 = -1;
    for t in &ticks {
        if !(t.turnover > 0.0) {
            continue;
        }
        // 文件里的 exchtime 是纪元微秒，reader 已加 8 小时；取模到一天之内再分桶。
        let day_us = t.us.rem_euclid(86_400_000_000);
        let sec = day_us / 1_000_000;
        let tk = day_us / 10_000;
        amt += t.turnover;
        vol += t.vol;
        let i = if (SEC_0930..=SEC_1130).contains(&sec) {
            Some((sec - SEC_0930) as usize)
        } else if (SEC_1300..=SEC_1457).contains(&sec) {
            Some(AM + (sec - SEC_1300) as usize)
        } else {
            None
        };
        match i {
            Some(i) => {
                s += 1;
                cnt_s[i] += 1;
                amt_s[i] += t.turnover;
                vol_s[i] += t.vol;
                if tk != last {
                    last = tk;
                    e += 1;
                    tick_s[i] += 1;
                }
            }
            None => last = -1,
        }
    }
    if s == 0 {
        return None;
    }
    let mut cnt = vec![0u32; N_SS];
    let mut tick = vec![0u32; N_SS];
    let mut amt60 = vec![0f64; N_SS];
    let mut vol60 = vec![0f64; N_SS];
    let mut cnt2 = vec![0u64; N_SS];
    let mut amt2 = vec![0f64; N_SS];
    let mut tot_m = vec![0f64; N_MIN];
    let mut seg = vec![vec![0u32; N_SS]; 4];
    for i in 0..N_SEC {
        tot_m[min_of(i)] += cnt_s[i] as f64;
    }
    let mut share1 = vec![0f64; N_SS];
    let mut share2 = vec![0f64; N_SS];
    for i in 0..N_SEC {
        let ss = (sec_of(i) % 60) as usize;
        let g = seg_of(i);
        let c = cnt_s[i] as u64;
        let a = amt_s[i];
        cnt[ss] += cnt_s[i];
        tick[ss] += tick_s[i];
        amt60[ss] += a;
        vol60[ss] += vol_s[i];
        cnt2[ss] += c * c;
        amt2[ss] += a * a;
        let t = tot_m[min_of(i)];
        if t > 0.0 {
            let y = cnt_s[i] as f64 / t;
            share1[ss] += y;
            share2[ss] += y * y;
        }
        seg[g][ss] += cnt_s[i];
    }
    Some(Rec { code: code.to_string(), s, e, amt, vol, cnt, tick, amt60, vol60, cnt2, amt2, share1, share2, seg })
}

fn stock_universe(date: i64) -> Vec<String> {
    let basic = std::env::var("RUST_PYFUNC_BASIC_INFO_DIR").unwrap_or("/ssd_data/data/basic_info".into());
    let root = std::env::var("RUST_PYFUNC_LEVEL2_ROOT").unwrap_or("/ssd_data/stock".into());
    let dir = std::path::Path::new(&root).join(date.to_string()).join("transaction");
    let text = std::fs::read_to_string(std::path::Path::new(&basic).join("symbol_map.csv")).unwrap_or_default();
    let mut codes: Vec<String> = text
        .lines()
        .skip(1)
        .filter_map(|l| {
            let c = l.split(',').next().unwrap_or("").trim();
            (!c.is_empty() && c.len() <= 6 && c.bytes().all(|b| b.is_ascii_digit())).then(|| format!("{c:0>6}"))
        })
        .collect();
    codes.sort_unstable();
    codes.dedup();
    codes.into_iter().filter(|c| dir.join(format!("{c}_{date}_transaction.csv")).is_file()).collect()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let date: i64 = args.get(1).expect("用法: second_census <date> [limit]").parse().expect("date");
    let limit: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(0);
    let mut codes = stock_universe(date);
    if limit > 0 && limit < codes.len() {
        codes.truncate(limit);
    }
    eprintln!("date={} codes={}", date, codes.len());

    let parts: Vec<Rec> = codes.par_iter().filter_map(|c| read_one(date, c)).collect();
    if parts.is_empty() {
        panic!("没有可用的股票日");
    }
    let out = Out {
        date,
        n_codes: codes.len(),
        n_valid: parts.len(),
        n_fail: codes.len() - parts.len(),
        meta: meta(),
        recs: parts,
    };
    eprintln!("valid={} fail={}", out.n_valid, out.n_fail);
    println!("{}", serde_json::to_string(&out).unwrap());
}
