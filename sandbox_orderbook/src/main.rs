//! 盘口快照普查：每只股票每天读 market_data，统计快照数量与十档深度。
//! 口径：只取 9:30-15:00 的快照（不含 9:15-9:25 集合竞价），停牌/新股/一字板在 Python 侧用日频数据剔除。
use rayon::prelude::*;
use serde::Serialize;
use std::collections::HashSet;
use std::fs;
use std::path::Path;

const COL_TIME: usize = 3;
const COL_LAST: usize = 6;
const COL_TURNOVER: usize = 15;
const T_0930: i64 = 93_000_000;
const T_1500: i64 = 150_000_000;

#[derive(Serialize)]
struct Out {
    date: i64,
    n_files: usize,
    stocks: Vec<Stat>,
}

#[derive(Serialize, Clone, Default)]
struct Stat {
    code: String,
    n_snap: u32,
    n_lunch: u32,
    mcnt: Vec<u32>,
    gap_hist: [u32; 32],
    gap50: f64,
    amt50: f64,
    amt10: f64,
    amt90: f64,
    sh50: f64,
    l1: f64,
    l5: f64,
    ba: f64,
    spr50: f64,
    px50: f64,
    nlev50: f64,
    turnover: f64,
    lvl_a: [f64; 10],
    shr: [f64; 10],
    lvl_b: [f64; 10],
}

#[inline]
fn pf(b: &[u8]) -> f64 {
    std::str::from_utf8(b).ok().and_then(|s| s.trim().parse::<f64>().ok()).unwrap_or(f64::NAN)
}

#[inline]
fn pi(b: &[u8]) -> i64 {
    std::str::from_utf8(b).ok().and_then(|s| s.trim().parse::<i64>().ok()).unwrap_or(0)
}

#[inline]
fn pct(v: &mut [f64], q: f64) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    let i = (((v.len() - 1) as f64) * q).round() as usize;
    v[i]
}

fn stat_one(date: i64, code: &str) -> Option<Stat> {
    let path = format!("/ssd_data/stock/{}/market_data/{}_{}_market_data.csv", date, code, date);
    let data = fs::read(Path::new(&path)).ok()?;
    let mut amt = Vec::with_capacity(5000);
    let mut sh = Vec::with_capacity(5000);
    let mut l1 = Vec::with_capacity(5000);
    let mut l5 = Vec::with_capacity(5000);
    let mut ba = Vec::with_capacity(5000);
    let mut spr = Vec::with_capacity(5000);
    let mut px = Vec::with_capacity(5000);
    let mut nlev = Vec::with_capacity(5000);
    let mut times: Vec<i64> = Vec::with_capacity(5000);
    let mut la: Vec<Vec<f64>> = (0..10).map(|_| Vec::with_capacity(5000)).collect();
    let mut shrv: Vec<Vec<f64>> = (0..10).map(|_| Vec::with_capacity(5000)).collect();
    let mut lb: Vec<Vec<f64>> = (0..10).map(|_| Vec::with_capacity(5000)).collect();
    let mut turnover = 0.0f64;
    let mut lunch = 0u32;
    let mut mcnt = vec![0u32; 331];
    let mut gap_hist = [0u32; 32];
    let mut prev_sec: Option<f64> = None;
    let mut first = true;
    let mut start = 0usize;
    for i in 0..=data.len() {
        if i < data.len() && data[i] != b'\n' {
            continue;
        }
        let mut line = &data[start..i];
        if line.last() == Some(&b'\r') {
            line = &line[..line.len() - 1];
        }
        start = i + 1;
        if line.is_empty() {
            continue;
        }
        if first {
            first = false;
            continue;
        }
        let mut f: Vec<&[u8]> = Vec::with_capacity(64);
        let mut s = 0usize;
        for (j, &b) in line.iter().enumerate() {
            if b == b',' {
                f.push(&line[s..j]);
                s = j + 1;
            }
        }
        f.push(&line[s..]);
        if f.len() < 61 {
            continue;
        }
        let t = pi(f[COL_TIME]);
        if t < T_0930 || t > T_1500 {
            continue;
        }
        // 午休（11:30 之后到 13:00 之前）也在发快照，沪市 30 秒一个、深市 60 秒一个。
        // 这里只留连续竞价的快照，午休的单独计数，不进入深度统计。
        if t > 113_000_000 && t < 130_000_000 {
            lunch += 1;
            let (h, m) = ((t / 10_000_000) as usize, ((t / 100_000) % 100) as usize);
            let idx = 121 + (h - 11) * 60 + m - 31;
            if idx < 331 { mcnt[idx] += 1; }
            continue;
        }
        {
            let (h, m) = ((t / 10_000_000) as usize, ((t / 100_000) % 100) as usize);
            let idx = if t <= 113_000_000 { (h - 9) * 60 + m - 30 } else { 210 + (h - 13) * 60 + m };
            if idx < 331 { mcnt[idx] += 1; }
        }
        turnover = pf(f[COL_TURNOVER]);
        let mut qa = 0.0;
        let mut qb = 0.0;
        let mut sa = 0.0;
        let mut sb = 0.0;
        let mut a1 = 0.0;
        let mut b1 = 0.0;
        let mut lv = 0.0;
        let mut head = 0.0;
        for k in 0..10usize {
            let ap = 21 + 4 * k;
            let av = pf(f[ap + 1]);
            let bp = pf(f[ap + 2]);
            let bv = pf(f[ap + 3]);
            let app = pf(f[ap]);
            let bpp = pf(f[ap + 2]);
            la[k].push(app * av);
            lb[k].push(bpp * bv);
            if app > 0.0 {
                qa += app * av;
                sa += av;
                lv += 1.0;
            }
            if bpp > 0.0 {
                qb += bpp * bv;
                sb += bv;
                lv += 1.0;
            }
            if k == 0 {
                a1 = av;
                b1 = bv;
            }
            if k < 5 {
                head += av + bv;
            }
            let _ = bp;
        }
        let sec = (t / 10_000_000) as f64 * 3600.0
            + ((t / 100_000) % 100) as f64 * 60.0
            + ((t / 1000) % 100) as f64
            + (t % 1000) as f64 / 1000.0;
        if let Some(ps) = prev_sec {
            let d = sec - ps;
            if d > 0.0 && d < 600.0 {
                let r = d.round() as usize;
                gap_hist[if r > 30 { 31 } else { r }] += 1;
            }
        }
        prev_sec = Some(sec);
        let tot_sh = sa + sb;
        if tot_sh <= 0.0 {
            continue;
        }
        {
            let tot_amt: f64 = la.iter().zip(lb.iter()).map(|(x, y)| x[x.len() - 1] + y[y.len() - 1]).sum();
            if tot_amt > 0.0 {
                for k in 0..10 {
                    let v = la[k][la[k].len() - 1] + lb[k][lb[k].len() - 1];
                    shrv[k].push(v / tot_amt);
                }
            }
        }
        amt.push(qa + qb);
        sh.push(tot_sh);
        l1.push((a1 + b1) / tot_sh);
        l5.push(head / tot_sh);
        if sa > 0.0 && sb > 0.0 { ba.push(sb / sa); }
        let ap1 = pf(f[21]);
        let bp1 = pf(f[23]);
        spr.push(if ap1 > 0.0 && bp1 > 0.0 { ap1 - bp1 } else { f64::NAN });
        px.push(pf(f[COL_LAST]));
        nlev.push(lv);
        times.push(t);
    }
    let n = amt.len();
    if n < 100 {
        return None;
    }
    let _ = &times;
    amt.sort_by(|a, b| a.partial_cmp(b).unwrap());
    sh.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mut gaps: Vec<f64> = times.windows(2).map(|w| (w[1] - w[0]) as f64 / 1000.0).filter(|d| *d > 0.0 && *d < 600.0).collect();
    let mut l1s = l1;
    let mut l5s = l5;
    let mut bas: Vec<f64> = ba.into_iter().filter(|x| x.is_finite()).collect();
    let mut sprs: Vec<f64> = spr.into_iter().filter(|x| x.is_finite()).collect();
    let mut pxs = px;
    let mut levs = nlev;
    gaps.sort_by(|a, b| a.partial_cmp(b).unwrap());
    l1s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    l5s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    bas.sort_by(|a, b| a.partial_cmp(b).unwrap());
    sprs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    pxs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    levs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mut shr = [0.0f64; 10];
    for k in 0..10 {
        shrv[k].sort_by(|a, b| a.partial_cmp(b).unwrap());
        shr[k] = pct(&mut shrv[k], 0.5);
    }
    let mut lvl_a = [0.0f64; 10];
    let mut lvl_b = [0.0f64; 10];
    for k in 0..10 {
        la[k].sort_by(|a, b| a.partial_cmp(b).unwrap());
        lb[k].sort_by(|a, b| a.partial_cmp(b).unwrap());
        lvl_a[k] = pct(&mut la[k], 0.5);
        lvl_b[k] = pct(&mut lb[k], 0.5);
    }
    Some(Stat {
        code: code.to_string(),
        n_snap: n as u32,
        n_lunch: lunch,
        mcnt,
        gap_hist,
        gap50: pct(&mut gaps, 0.5),
        amt50: pct(&mut amt, 0.5),
        amt10: pct(&mut amt, 0.10),
        amt90: pct(&mut amt, 0.90),
        sh50: pct(&mut sh, 0.5),
        l1: pct(&mut l1s, 0.5),
        l5: pct(&mut l5s, 0.5),
        ba: pct(&mut bas, 0.5),
        spr50: pct(&mut sprs, 0.5),
        px50: pct(&mut pxs, 0.5),
        nlev50: pct(&mut levs, 0.5),
        turnover,
        lvl_a,
        lvl_b,
        shr,
    })
}

fn universe(date: i64) -> Vec<String> {
    let syms = fs::read_to_string("/ssd_data/data/basic_info/symbol_map.csv").unwrap_or_default();
    let set: HashSet<&str> = syms.lines().skip(1).filter_map(|l| l.split(',').next()).collect();
    let suffix = format!("_{}_market_data.csv", date);
    let mut v: Vec<String> = fs::read_dir(format!("/ssd_data/stock/{}/market_data", date))
        .map(|rd| rd.filter_map(|e| {
            let name = e.ok()?.file_name().into_string().ok()?;
            let code = name.strip_suffix(&suffix)?.to_string();
            if set.contains(code.as_str()) { Some(code) } else { None }
        }).collect())
        .unwrap_or_default();
    v.sort();
    v
}

fn main() {
    let date: i64 = std::env::args().nth(1).expect("need date").parse().unwrap();
    let codes = universe(date);
    let stocks: Vec<Stat> = codes.par_iter().filter_map(|c| stat_one(date, c)).collect();
    let out = Out { date, n_files: stocks.len(), stocks };
    println!("{}", serde_json::to_string(&out).unwrap());
}
