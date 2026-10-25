//! 成交节奏（第 04 期，sandbox）：一天全市场逐笔，逐股票日算「成交挤不挤」，
//! 并给每个股票日造一条「同笔数、同日内作息、内部打乱」的对照序列。
//!
//! 口径
//! - 成交：flag=66（主买）或 83（主卖）且 turnover>0。
//! - 连续竞价：9:30:00-11:30:00 与 13:00:00-14:57:00，逐秒分桶共 14,220 个（上午 7,200、下午 7,020）。
//!   14:57:00 之后进入收盘集合竞价，15:00:00 那一笔撮合单独计数，不进任何秒桶。
//! - 簇：同一个时间戳刻度里的几笔成交当成一个不可拆的原子。八个展示日里有七天时间戳精度
//!   是 10 毫秒（微秒字段末四位恒为零），只有 2025-11-27 是真微秒；为了八天可比，统一截到
//!   10 毫秒再分组。所以这一期量的是「10 毫秒以上」的成团。
//! - 对照：把这只股票的簇按「当天全市场逐秒成交占比」重新分配，笔数不变、簇大小不变、
//!   日内作息不变，唯一去掉的是「挤不挤」。
//! - 两个成团度：
//!   带作息 D_p = Σ_t (x_t - S p_t)^2 / S，对照的期望值 (Σ_c s_c^2)(1-Σp^2) / S
//!   去作息 D_u = Σ_t (x_t - S/T)^2 / S，对照的期望值 (Σ_c s_c^2)(1-1/T) / S
//!   两者相除就是「纯成团倍数」，泊松过程下等于 1，和笔数无关。
//!
//! 构建：cd sandbox_trade_rhythm && cargo build --release
//! 运行：./target/release/trade_rhythm 20240329 > out/20240329.json
//!       ./target/release/trade_rhythm 20240329 30   # 只跑前 30 只，用于小样本核对
mod fast_csv_reader;
use fast_csv_reader::read_ticks;
use rayon::prelude::*;
use serde::Serialize;
use std::sync::Arc;

const T_AM: usize = 7200;
const T_PM: usize = 7020;
const T: usize = 14220;
const NB: usize = 33;
const NDRAW: usize = 3;
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
    c: u64,
    s2: u64,
    nc: u64,
    nauc: u64,
    noth: u64,
    amt: f64,
    amt_c: f64,
    vol: f64,
    d_op: f64,
    d_sp: f64,
    d_ou: f64,
    d_su: f64,
    d_mc: f64,
    maxcs: u32,
    maxcsm: f64,
    hist: Vec<u32>,
    hist_s: Vec<f32>,
    lz: Vec<f32>,
    lz_s: Vec<f32>,
}

#[derive(Serialize)]
struct Market {
    trades: f64,
    p2: f64,
    t_eff: f64,
    top10_share: f64,
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
    c: u64,
    s2: u64,
    nc: u64,
    nauc: u64,
    noth: u64,
    amt: f64,
    amt_c: f64,
    vol: f64,
    maxcs: u32,
    counts: Vec<u32>,
    cs: [u32; 65],
    cs_big: Vec<(u32, u32)>,
    hist: [u32; NB],
    sx2: f64,
}

struct Rng(u64);

impl Rng {
    #[inline]
    fn new(seed: u64) -> Self {
        Rng(seed | 1)
    }
    #[inline]
    fn next(&mut self) -> f64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        ((x >> 11) as f64) * (1.0 / 9_007_199_254_740_992.0)
    }
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

#[inline]
fn flush(cur: u32, c: &mut u64, s2: &mut u64, cs: &mut [u32; 65], big: &mut Vec<(u32, u32)>) {
    if cur == 0 {
        return;
    }
    *c += 1;
    *s2 += (cur as u64) * (cur as u64);
    if cur <= 64 {
        cs[cur as usize] += 1;
    } else if let Some(e) = big.iter_mut().find(|e| e.0 == cur) {
        e.1 += 1;
    } else {
        big.push((cur, 1));
    }
}

fn read_one(date: i64, code: &str) -> Option<Part> {
    let ticks = read_ticks(code, date).ok()?;
    if ticks.is_empty() {
        return None;
    }
    let mut counts = vec![0u32; T];
    let mut s = 0u64;
    let mut c = 0u64;
    let mut s2 = 0u64;
    let mut nc = 0u64;
    let mut nauc = 0u64;
    let mut noth = 0u64;
    let mut amt = 0.0f64;
    let mut amt_c = 0.0f64;
    let mut vol = 0.0f64;
    let mut cs = [0u32; 65];
    let mut cs_big: Vec<(u32, u32)> = Vec::new();
    let mut last_ms: i64 = -1;
    let mut cur: u32 = 0;
    let mut last_bucket: usize = usize::MAX;
    let mut cnt_bucket: u32 = 0;
    let mut maxcs: u32 = 0;
    for t in &ticks {
        if !(t.turnover > 0.0) {
            continue;
        }
        // 文件里的 exchtime 是纪元微秒，reader 已加 8 小时；取模到一天之内再分桶。
        let day_us = t.us.rem_euclid(86_400_000_000);
        let sec = day_us / 1_000_000;
        let ms = day_us / 10_000;
        amt += t.turnover;
        vol += t.vol;
        match bucket(sec) {
            Some(i) => {
                s += 1;
                amt_c += t.turnover;
                counts[i] += 1;
                if ms != last_ms {
                    flush(cur, &mut c, &mut s2, &mut cs, &mut cs_big);
                    last_ms = ms;
                    cur = 0;
                    if i != last_bucket { last_bucket = i; cnt_bucket = 0; }
                    cnt_bucket += 1;
                    if cnt_bucket > maxcs { maxcs = cnt_bucket; }
                }
                cur += 1;
            }
            None => {
                flush(cur, &mut c, &mut s2, &mut cs, &mut cs_big);
                last_ms = -1;
                cur = 0;
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
    flush(cur, &mut c, &mut s2, &mut cs, &mut cs_big);
    if s == 0 {
        return None;
    }
    let mut hist = [0u32; NB];
    let mut sx2 = 0.0f64;
    for &v in counts.iter() {
        hist[(v as usize).min(NB - 1)] += 1;
        sx2 += (v as f64) * (v as f64);
    }
    Some(Part { code: code.to_string(), s, c, s2, nc, nauc, noth, amt, amt_c, vol, maxcs, counts, cs, cs_big, hist, sx2 })
}

/// 累计曲线：把一天的秒按成交笔数从高到低排，取「最忙的前 f 份时间」占了多少成交。
/// 输出 NLZ 个点，时间占比从 0 到 1。精确算，不靠直方图分箱。
fn lorenz(v: &mut [u32], out: &mut [f64]) {
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
        *o = if tot > 0.0 { (cum + extra) / tot } else { 0.0 };
    }
}

fn finish(p: Part, cum: &[f64], p2: f64) -> Rec {
    let s = p.s as f64;
    let mut sxp = 0.0f64;
    for i in 0..T {
        let v = p.counts[i];
        if v > 0 {
            sxp += (v as f64) * (cum[i + 1] - cum[i]);
        }
    }
    let d_op = (p.sx2 - 2.0 * s * sxp + s * s * p2) / s;
    // 对照的期望：每个簇独立按 p 落到各秒，E[Σ(x-Sp)^2] = (1-Σp^2) Σs^2
    let d_sp = (p.s2 as f64) * (1.0 - p2) / s;
    let d_ou = (p.sx2 - s * s / T as f64) / s;
    let d_su = (p.s2 as f64) * (1.0 - 1.0 / T as f64) / s;

    let mut lzv = vec![0.0f64; NLZ];
    let mut tmp: Vec<u32> = p.counts.clone();
    lorenz(&mut tmp, &mut lzv);

    let mut seed: u64 = 1469598103934665603;
    for b in p.code.bytes() {
        seed = (seed ^ b as u64).wrapping_mul(1099511628211);
    }
    let mut rng = Rng::new(seed);
    let mut hs = vec![0.0f64; NB];
    let mut lzs = vec![0.0f64; NLZ];
    let mut lzt = vec![0.0f64; NLZ];
    let mut d_mc = 0.0f64;
    let mut buf = vec![0u32; T];
    // 一秒只有 100 个 10 毫秒刻度，真实数据一秒的簇数被这个上限卡住。对照不设上限，
    // 所以顺手记下对照里一秒最多落几个簇，跟实际的 maxcs 比，就能看出上限有没有影响。
    let mut csec = vec![0u16; T];
    let mut maxcsm: u32 = 0;
    for _ in 0..NDRAW {
        buf.iter_mut().for_each(|x| *x = 0);
        csec.iter_mut().for_each(|x| *x = 0);
        for sz in 1..=64u32 {
            let cnt = p.cs[sz as usize];
            for _ in 0..cnt {
                let u = rng.next();
                let i = cum.partition_point(|&x| x <= u);
                csec[i - 1] += 1;
                if csec[i - 1] as u32 > maxcsm { maxcsm = csec[i - 1] as u32; }
                buf[i - 1] += sz;
            }
        }
        for (sz, cnt) in p.cs_big.iter() {
            for _ in 0..*cnt {
                let u = rng.next();
                let i = cum.partition_point(|&x| x <= u);
                csec[i - 1] += 1;
                if csec[i - 1] as u32 > maxcsm { maxcsm = csec[i - 1] as u32; }
                buf[i - 1] += *sz;
            }
        }
        let mut sx2m = 0.0f64;
        let mut sxpm = 0.0f64;
        for i in 0..T {
            let v = buf[i];
            hs[(v as usize).min(NB - 1)] += 1.0;
            if v > 0 {
                sx2m += (v as f64) * (v as f64);
                sxpm += (v as f64) * (cum[i + 1] - cum[i]);
            }
        }
        d_mc += (sx2m - 2.0 * s * sxpm + s * s * p2) / s;
        lorenz(&mut buf, &mut lzt);
        for i in 0..NLZ {
            lzs[i] += lzt[i];
        }
    }
    Rec {
        code: p.code,
        s: p.s,
        c: p.c,
        s2: p.s2,
        nc: p.nc,
        nauc: p.nauc,
        noth: p.noth,
        amt: p.amt,
        amt_c: p.amt_c,
        vol: p.vol,
        d_op,
        d_sp,
        d_ou,
        d_su,
        d_mc: d_mc / NDRAW as f64,
        maxcs: p.maxcs,
        maxcsm: maxcsm as f64,
        hist: p.hist.to_vec(),
        hist_s: hs.iter().map(|x| (x / NDRAW as f64) as f32).collect(),
        lz: lzv.iter().map(|x| *x as f32).collect(),
        lz_s: lzs.iter().map(|x| (x / NDRAW as f64) as f32).collect(),
    }
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

    let mut mkt = vec![0.0f64; T];
    for p in parts.iter() {
        for i in 0..T {
            mkt[i] += p.counts[i] as f64;
        }
    }
    let tot: f64 = mkt.iter().sum();
    let mut cum = vec![0.0f64; T + 1];
    for i in 0..T {
        cum[i + 1] = cum[i] + mkt[i] / tot;
    }
    let p2: f64 = (0..T).map(|i| { let q = mkt[i] / tot; q * q }).sum();
    let mut sorted = mkt.clone();
    sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
    let k10 = (T / 10).max(1);
    let top10: f64 = sorted[..k10].iter().sum::<f64>() / tot;
    let market = Market { trades: tot, p2, t_eff: 1.0 / p2, top10_share: top10 };

    let cum = Arc::new(cum);
    let n_valid = parts.len();
    let recs: Vec<Rec> = parts
        .into_par_iter()
        .map(|p| finish(p, &cum, p2))
        .collect();
    let out = Out { date, n_codes: codes.len(), n_valid, n_fail: codes.len() - n_valid, market, recs };
    eprintln!("valid={} fail={} p2={:.3e} t_eff={:.0}", out.n_valid, out.n_fail, p2, 1.0 / p2);
    println!("{}", serde_json::to_string(&out).unwrap());
}
