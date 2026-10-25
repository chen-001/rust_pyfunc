//! 成交的节奏（第 04 期，sandbox）
//!
//! 两个计数单位
//! - 记录：文件里的一行，flag 66 或 83 且 turnover>0。
//! - 事件：同一个时间戳里的几笔记录算一次。时间戳精度八天里有七天是 10 毫秒
//!   （微秒字段末四位恒为零），2025-11-27 是真微秒；统一截到 10 毫秒再判重。
//!
//! 时段：连续竞价 9:30:00-11:30:00 与 13:00:00-14:57:00，逐秒分桶共 14,220 个
//! （上午 7,200、下午 7,020）。14:57:00 之后进收盘集合竞价，15:00:00 那一笔
//! 撮合单独计数，不进任何秒桶。
//!
//! 输出
//! - hist / lz：每秒成交「记录」数的直方图与累计集中曲线（描述用）。
//! - fano：每秒成交「事件」数的方差除以均值。同样笔数随机撒开等于 1。
//! - sx2 / sem：Σ(每秒事件数²) 与 Σ(每秒事件数 × 当天全市场每秒事件数)，算相关系数用。
//!
//! 构建：cd sandbox_trade_rhythm && cargo build --release
//! 运行：./target/release/trade_rhythm 20240329 > out/20240329.json
//!       ./target/release/trade_rhythm 20240329 30   # 只跑前 30 只，用于小样本核对
mod fast_csv_reader;
use fast_csv_reader::read_ticks;
use rayon::prelude::*;
use serde::Serialize;

const T_AM: usize = 7200;
const T_PM: usize = 7020;
const T: usize = 14220;
const NB: usize = 33;
const NLZ: usize = 201;

const SEC_0930: i64 = 34_200;
const SEC_1130: i64 = 41_400;
const SEC_1300: i64 = 46_800;
const SEC_1457: i64 = 53_820;
const SEC_1500: i64 = 54_000;

#[derive(Serialize)]
struct Rec {
    code: String,
    s: u64,
    e: u64,
    nc: u64,
    nauc: u64,
    noth: u64,
    amt: f64,
    amt_c: f64,
    vol: f64,
    fano: f64,
    sx2: f64,
    sem: f64,
    hist: Vec<u32>,
    lz: Vec<f32>,
}

#[derive(Serialize)]
struct Market {
    events: f64,
    mean: f64,
    var: f64,
}

#[derive(Serialize)]
struct Out {
    date: i64,
    n_codes: usize,
    n_valid: usize,
    n_fail: usize,
    market: Market,
    recs: Vec<Rec>,
}

struct Part {
    code: String,
    s: u64,
    e: u64,
    nc: u64,
    nauc: u64,
    noth: u64,
    amt: f64,
    amt_c: f64,
    vol: f64,
    ecnt: Vec<u16>,
    hist: [u32; NB],
    lz: Vec<f32>,
}

#[inline]
fn bucket(sec: i64) -> Option<usize> {
    if (SEC_0930..=SEC_1130).contains(&sec) {
        Some(((sec - SEC_0930) as usize).min(T_AM - 1))
    } else if (SEC_1300..=SEC_1457).contains(&sec) {
        Some(T_AM + ((sec - SEC_1300) as usize).min(T_PM - 1))
    } else {
        None
    }
}

/// 累计曲线：把一天的秒按笔数从高到低排，取「最忙的前 f 份时间」占了多少。
fn lorenz(v: &mut [u32], out: &mut [f32]) {
    v.sort_unstable_by(|a, b| b.cmp(a));
    let tot: f64 = v.iter().map(|&x| x as f64).sum();
    let t = v.len() as f64;
    let n = out.len() - 1;
    let mut j = 0usize;
    let mut cum = 0.0f64;
    for (i, o) in out.iter_mut().enumerate() {
        let target = (i as f64) / (n as f64) * t;
        let k = (target.floor() as usize).min(v.len());
        while j < k {
            cum += v[j] as f64;
            j += 1;
        }
        let frac = target - target.floor();
        let extra = if j < v.len() && frac > 0.0 { v[j] as f64 * frac } else { 0.0 };
        *o = (if tot > 0.0 { (cum + extra) / tot } else { 0.0 }) as f32;
    }
}

fn read_one(date: i64, code: &str) -> Option<Part> {
    let ticks = read_ticks(code, date).ok()?;
    if ticks.is_empty() {
        return None;
    }
    let mut rcnt = vec![0u32; T];
    let mut ecnt = vec![0u16; T];
    let mut s = 0u64;
    let mut e = 0u64;
    let mut nc = 0u64;
    let mut nauc = 0u64;
    let mut noth = 0u64;
    let mut amt = 0.0f64;
    let mut amt_c = 0.0f64;
    let mut vol = 0.0f64;
    let mut last_tick: i64 = -1;
    for t in &ticks {
        if !(t.turnover > 0.0) {
            continue;
        }
        // 文件里的 exchtime 是纪元微秒，reader 已加 8 小时；取模到一天之内再分桶。
        let day_us = t.us.rem_euclid(86_400_000_000);
        let sec = day_us / 1_000_000;
        let tick = day_us / 10_000;
        amt += t.turnover;
        vol += t.vol;
        match bucket(sec) {
            Some(i) => {
                s += 1;
                amt_c += t.turnover;
                rcnt[i] += 1;
                if tick != last_tick {
                    last_tick = tick;
                    e += 1;
                    ecnt[i] += 1;
                }
            }
            None => {
                last_tick = -1;
                if sec < SEC_0930 {
                    nauc += 1;
                } else if sec == SEC_1500 {
                    nc += 1;
                } else {
                    noth += 1;
                }
            }
        }
    }
    if s == 0 {
        return None;
    }
    let mut hist = [0u32; NB];
    for &v in rcnt.iter() {
        hist[(v as usize).min(NB - 1)] += 1;
    }
    let mut lz = vec![0.0f32; NLZ];
    lorenz(&mut rcnt, &mut lz);
    Some(Part { code: code.to_string(), s, e, nc, nauc, noth, amt, amt_c, vol, ecnt, hist, lz })
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
    let date: i64 = args.get(1).expect("用法: trade_rhythm <date> [limit]").parse().expect("date");
    let limit: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(0);
    let mut codes = stock_universe(date);
    if limit > 0 && limit < codes.len() {
        codes.truncate(limit);
    }
    eprintln!("date={} codes={}", date, codes.len());

    let parts: Vec<Part> = codes.par_iter().filter_map(|c| read_one(date, c)).collect();
    if parts.is_empty() {
        panic!("没有可用的股票日");
    }

    // 全市场每秒的成交事件数，作为当天大家的共同作息
    let mut mkt = vec![0.0f64; T];
    for p in parts.iter() {
        for i in 0..T {
            mkt[i] += p.ecnt[i] as f64;
        }
    }
    let mtot: f64 = mkt.iter().sum();
    let mmean = mtot / T as f64;
    let mvar = mkt.iter().map(|&x| (x - mmean) * (x - mmean)).sum::<f64>() / T as f64;

    let n_valid = parts.len();
    let recs: Vec<Rec> = parts
        .into_par_iter()
        .map(|p| {
            let mut sx2 = 0.0f64;
            let mut sem = 0.0f64;
            for i in 0..T {
                let v = p.ecnt[i] as f64;
                if v > 0.0 {
                    sx2 += v * v;
                    sem += v * mkt[i];
                }
            }
            let mean = p.e as f64 / T as f64;
            let var = sx2 / T as f64 - mean * mean;
            Rec {
                code: p.code,
                s: p.s,
                e: p.e,
                nc: p.nc,
                nauc: p.nauc,
                noth: p.noth,
                amt: p.amt,
                amt_c: p.amt_c,
                vol: p.vol,
                fano: var / mean,
                sx2,
                sem,
                hist: p.hist.to_vec(),
                lz: p.lz,
            }
        })
        .collect();
    let out = Out { date, n_codes: codes.len(), n_valid, n_fail: codes.len() - n_valid, market: Market { events: mtot, mean: mmean, var: mvar }, recs };
    eprintln!("valid={} fail={} 全市场事件 {:.0} 每秒均值 {:.1}", out.n_valid, out.n_fail, mtot, mmean);
    println!("{}", serde_json::to_string(&out).unwrap());
}
