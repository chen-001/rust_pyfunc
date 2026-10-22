//! 撤单普查（sandbox）：一天全市场逐笔，逐股票日统计撤单与成交的对比。
//!
//! 口径
//! - 撤单：flag=32 且 volume>0，方向看 bid_order / ask_order 哪个非零。
//! - 成交：flag=66（主买）或 83（主卖）且 turnover>0。
//! - 时段：只取连续竞价 9:30:00-11:30:00 与 13:00:00-15:00:00，15:00 整点归入最后一桶。
//!   9:15-9:25 集合竞价的撤单与成交都不算（20240329 的 000001 有 688 笔集合竞价撤单）。
//! - 撤单没有价格（price 与 turnover 恒为 0），所以一切规模用股数，不用金额。
//! - 成批：同一毫秒内的撤单记录算一批。exchtime 字段是微秒，但八个展示日里有七天
//!   末三位恒为零（精度只到毫秒），2025-11－27 那天约一半记录带真微秒。按毫秒取整，
//!   八天口径才一致。
//!
//! 构建: cd sandbox_cancel_census && cargo build --release
//! 运行: ./target/release/cancel_census 20240329 > out/20240329.json
mod fast_csv_reader;
use fast_csv_reader::read_ticks;
use rayon::prelude::*;
use serde::Serialize;

const SEC_0930: i64 = 9 * 3600 + 30 * 60;
const SEC_1130: i64 = 11 * 3600 + 30 * 60;
const SEC_1300: i64 = 13 * 3600;
const SEC_1500: i64 = 15 * 3600;

const N_B5: usize = 48;
const BIN_LO: f64 = 1.5;
const BIN_W: f64 = 0.01;
const N_BIN: usize = 500;

#[derive(Serialize)]
struct StockStat {
    code: String,
    cvol_b: f64,
    cvol_s: f64,
    cn_b: u64,
    cn_s: u64,
    tvol_b: f64,
    tvol_s: f64,
    tn_b: u64,
    tn_s: u64,
    tamt: f64,
    cp50: f64,
    cp90: f64,
    tp50: f64,
    tp90: f64,
    n_batch: u64,
    b5_cvol: Vec<f64>,
    b5_tvol: Vec<f64>,
    b5_tamt: Vec<f64>,
    batch_n: Vec<u64>,
    batch_vol: Vec<f64>,
    batch_tick: Vec<u64>,
}

#[derive(Serialize)]
struct Out {
    date: i64,
    n_codes: usize,
    n_valid: usize,
    n_fail: usize,
    stocks: Vec<StockStat>,
    b5_cvol: Vec<f64>,
    b5_tvol: Vec<f64>,
    b5_tamt: Vec<f64>,
    b5_n: Vec<u64>,
    hist_c: Vec<u64>,
    hist_t: Vec<u64>,
    hist_lo: f64,
    hist_w: f64,
    batch_n: Vec<u64>,
    batch_vol: Vec<f64>,
    batch_tick: Vec<u64>,
}

struct Part {
    stat: StockStat,
    b5_cvol: [f64; N_B5],
    b5_tvol: [f64; N_B5],
    b5_tamt: [f64; N_B5],
    b5_n: [u64; N_B5],
    hist_c: [u32; N_BIN],
    hist_t: [u32; N_BIN],
    batch_n: [u64; 4],
    batch_vol: [f64; 4],
    batch_tick: [u64; 4],
}

fn stock_universe(date: i64) -> std::io::Result<Vec<String>> {
    use std::io::{Error, ErrorKind};
    let basic = std::env::var("RUST_PYFUNC_BASIC_INFO_DIR").unwrap_or("/ssd_data/data/basic_info".into());
    let root = std::env::var("RUST_PYFUNC_LEVEL2_ROOT").unwrap_or("/ssd_data/stock".into());
    let dir = std::path::Path::new(&root).join(date.to_string()).join("transaction");
    let text = std::fs::read_to_string(std::path::Path::new(&basic).join("symbol_map.csv"))
        .map_err(|_| Error::new(ErrorKind::InvalidData, "股票主表缺失"))?;
    let mut stocks: Vec<String> = text
        .lines()
        .skip(1)
        .filter_map(|l| {
            let c = l.split(',').next().unwrap_or("").trim();
            (!c.is_empty() && c.len() <= 6 && c.bytes().all(|b| b.is_ascii_digit()))
                .then(|| format!("{c:0>6}"))
        })
        .collect();
    stocks.sort_unstable();
    stocks.dedup();
    if stocks.is_empty() || !dir.is_dir() {
        return Err(Error::new(ErrorKind::InvalidData, "股票白名单为空或原始目录不存在"));
    }
    Ok(stocks
        .into_iter()
        .filter(|c| dir.join(format!("{c}_{date}_transaction.csv")).is_file())
        .collect())
}

#[inline]
fn bucket(sec: i64) -> Option<usize> {
    if (SEC_0930..SEC_1130).contains(&sec) {
        Some(((sec - SEC_0930) / 300) as usize)
    } else if (SEC_1300..=SEC_1500).contains(&sec) {
        Some(24 + (((sec - SEC_1300) / 300) as usize).min(23))
    } else {
        None
    }
}

#[inline]
fn bin_of(size: f64) -> Option<usize> {
    if size <= 0.0 {
        return None;
    }
    let b = ((size.log10() - BIN_LO) / BIN_W) as isize;
    Some(b.clamp(0, N_BIN as isize - 1) as usize)
}

#[inline]
fn q(v: &mut [f64], p: f64) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    let k = (((v.len() - 1) as f64) * p).round() as usize;
    *v.select_nth_unstable_by(k, |a, b| a.partial_cmp(b).unwrap()).1
}

fn stat_one(date: i64, code: &str) -> Option<Part> {
    let ticks = read_ticks(code, date).ok()?;
    let mut cvol = [0.0f64; 2];
    let mut cn = [0u64; 2];
    let mut tvol = [0.0f64; 2];
    let mut tn = [0u64; 2];
    let mut tamt = 0.0f64;
    let mut csz: Vec<f64> = Vec::new();
    let mut tsz: Vec<f64> = Vec::new();
    let mut b5_cvol = [0.0f64; N_B5];
    let mut b5_tvol = [0.0f64; N_B5];
    let mut b5_tamt = [0.0f64; N_B5];
    let mut b5_n = [0u64; N_B5];
    let mut hist_c = [0u32; N_BIN];
    let mut hist_t = [0u32; N_BIN];
    let mut batch_n = [0u64; 4];
    let mut batch_vol = [0.0f64; 4];
    let mut batch_tick = [0u64; 4];
    let mut n_batch = 0u64;

    let mut cur_us = i64::MIN;   // 毫秒
    let mut cur_cnt = 0u32;
    let mut cur_vol = 0.0f64;

    for t in &ticks {
        let sec = (t.us / 1_000_000).rem_euclid(86400);
        let Some(bi) = bucket(sec) else { continue };
        if t.flag == 32 {
            if t.vol <= 0.0 {
                continue;
            }
            let ms = t.us / 1000;
            let side = if t.bid_order > 0 {
                0
            } else if t.ask_order > 0 {
                1
            } else {
                continue;
            };
            cvol[side] += t.vol;
            cn[side] += 1;
            b5_cvol[bi] += t.vol;
            b5_n[bi] += 1;
            if let Some(k) = bin_of(t.vol) {
                hist_c[k] += 1;
            }
            csz.push(t.vol);
            if ms == cur_us {
                cur_cnt += 1;
                cur_vol += t.vol;
            } else {
                if cur_cnt > 0 {
                    let idx = match cur_cnt {
                        1 => 0,
                        2 => 1,
                        3..=10 => 2,
                        _ => 3,
                    };
                    batch_n[idx] += 1;
                    batch_vol[idx] += cur_vol;
                    batch_tick[idx] += cur_cnt as u64;
                    n_batch += 1;
                }
                cur_us = ms;
                cur_cnt = 1;
                cur_vol = t.vol;
            }
        } else if t.turnover > 0.0 {
            let side = if t.flag == 66 { 0 } else { 1 };
            tvol[side] += t.vol;
            tn[side] += 1;
            tamt += t.turnover;
            b5_tvol[bi] += t.vol;
            b5_tamt[bi] += t.turnover;
            if let Some(k) = bin_of(t.vol) {
                hist_t[k] += 1;
            }
            tsz.push(t.vol);
        }
    }
    if cur_cnt > 0 {
        let idx = match cur_cnt {
            1 => 0,
            2 => 1,
            3..=10 => 2,
            _ => 3,
        };
        batch_n[idx] += 1;
        batch_vol[idx] += cur_vol;
        batch_tick[idx] += cur_cnt as u64;
        n_batch += 1;
    }

    let (cp50, cp90) = (q(&mut csz, 0.5), q(&mut csz, 0.9));
    let (tp50, tp90) = (q(&mut tsz, 0.5), q(&mut tsz, 0.9));
    if cn[0] + cn[1] == 0 || tn[0] + tn[1] == 0 {
        return None;
    }
    Some(Part {
        stat: StockStat {
            code: code.to_string(),
            cvol_b: cvol[0],
            cvol_s: cvol[1],
            cn_b: cn[0],
            cn_s: cn[1],
            tvol_b: tvol[0],
            tvol_s: tvol[1],
            tn_b: tn[0],
            tn_s: tn[1],
            tamt,
            cp50,
            cp90,
            tp50,
            tp90,
            n_batch,
            b5_cvol: b5_cvol.to_vec(),
            b5_tvol: b5_tvol.to_vec(),
            b5_tamt: b5_tamt.to_vec(),
            batch_n: batch_n.to_vec(),
            batch_vol: batch_vol.to_vec(),
            batch_tick: batch_tick.to_vec(),
        },
        b5_cvol,
        b5_tvol,
        b5_tamt,
        b5_n,
        hist_c,
        hist_t,
        batch_n,
        batch_vol,
        batch_tick,
    })
}

fn main() {
    let date: i64 = std::env::args().nth(1).expect("用法: cancel_census <date>").parse().unwrap();
    let codes = stock_universe(date).expect("股票范围或原始目录检查失败");
    eprintln!("date={} codes={}", date, codes.len());

    let parts: Vec<Option<Part>> = codes.par_iter().map(|c| stat_one(date, c)).collect();

    let mut out = Out {
        date,
        n_codes: codes.len(),
        n_valid: 0,
        n_fail: 0,
        stocks: Vec::new(),
        b5_cvol: vec![0.0; N_B5],
        b5_tvol: vec![0.0; N_B5],
        b5_tamt: vec![0.0; N_B5],
        b5_n: vec![0; N_B5],
        hist_c: vec![0; N_BIN],
        hist_t: vec![0; N_BIN],
        hist_lo: BIN_LO,
        hist_w: BIN_W,
        batch_n: vec![0; 4],
        batch_vol: vec![0.0; 4],
        batch_tick: vec![0; 4],
    };
    for p in parts {
        match p {
            Some(p) => {
                out.n_valid += 1;
                out.stocks.push(p.stat);
                for i in 0..N_B5 {
                    out.b5_cvol[i] += p.b5_cvol[i];
                    out.b5_tvol[i] += p.b5_tvol[i];
                    out.b5_tamt[i] += p.b5_tamt[i];
                    out.b5_n[i] += p.b5_n[i];
                }
                for i in 0..N_BIN {
                    out.hist_c[i] += p.hist_c[i] as u64;
                    out.hist_t[i] += p.hist_t[i] as u64;
                }
                for i in 0..4 {
                    out.batch_n[i] += p.batch_n[i];
                    out.batch_vol[i] += p.batch_vol[i];
                    out.batch_tick[i] += p.batch_tick[i];
                }
            }
            None => out.n_fail += 1,
        }
    }
    if out.n_valid == 0 {
        eprintln!("没有一只股票产出结果，退出");
        std::process::exit(1);
    }
    eprintln!("valid={} fail={}", out.n_valid, out.n_fail);
    println!("{}", serde_json::to_string(&out).unwrap());
}
