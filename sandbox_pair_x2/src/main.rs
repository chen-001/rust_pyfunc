//! 纯 Rust 跨股票互动因子流水线（sandbox 性能验证版）
//! 目标: 20241231 单日全市场在 50 核下 1 分钟内完成 读取→计算→存储
//!
//! 性能设计（内存友好原则）:
//!   - 逐笔时间打包为 u32（秒14位+微秒/32 15位+大单1位），合并流带宽减到 1/4
//!   - X1/X3/X8/X9 只算上三角（对称），X8 用独立大单集合合并（~1% 成本）
//!   - X2/X5 的反方向 = 转置（一次 gemm 得到完整非对称矩阵）
//!   - 手写分块 f32 gemm（Bᵀ 转置 + i-block 并行）
//!   - 二跳特征 O(N²)（diag=行平方和, row_mean=矩阵×列和），不做 N³ 矩阵乘
//!   - 谱分解用确定性幂迭代（固定种子 + GS 正交化），f32 并行 matvec
//!
//! 运行: ./target/release/pair_x2_sandbox <date> <outdir> [limit] [--write-mats]
mod fast_csv_reader;
mod gemm;
use fast_csv_reader::{read_trade_fast, TradeRecord};
use gemm::{gemm_abt, transpose};
use rayon::prelude::*;
use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::io::Write;

const MIN_TRADES: usize = 200;
const BUCKET_1S: usize = 14220;
const SCALE_30: usize = 474;
const SCALE_60: usize = 237;
const SCALE_300: usize = 48;
const BIG10_T: usize = 1422;
const BIG30_T: usize = 474;
const X2_DT: [usize; 3] = [10, 30, 60];
const SPECTRAL_K: usize = 30;
const SPECTRAL_MAX_ITER: usize = 40;

fn days_from_civil(y: i64, m: u64, d: u64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400;
    let mp = (m + 9) % 12;
    let doy = (153 * mp + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy as i64;
    era * 146097 + doe - 719468
}

fn list_codes(date: i64) -> Vec<String> {
    for root in ["/ssd_data/stock", "/nas197/binary/stock/sz_alpha/stock"] {
        let dir = format!("{root}/{date}/transaction");
        if let Ok(entries) = std::fs::read_dir(&dir) {
            let mut v: Vec<(u64, String)> = Vec::new();
            for e in entries.flatten() {
                if let Some(code) = e.file_name().to_str().and_then(|n| n.split('_').next()) {
                    if code.bytes().all(|b| b.is_ascii_digit()) {
                        let size = e.metadata().map(|m| m.len()).unwrap_or(0);
                        v.push((size, code.to_string()));
                    }
                }
            }
            // 按文件大小降序（成交活跃度代理），同大小按代码序保证确定性
            v.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
            let mut codes: Vec<String> = v.into_iter().map(|(_, c)| c).collect();
            codes.sort();
            return codes;
        }
    }
    Vec::new()
}

// ============================================================
// 每股预处理
// ============================================================
struct StockData {
    code: String,
    times: Vec<u32>,
    big_times: Vec<u32>,
    n: usize,
    nbig: usize,
    th99: f64,
    turnover: f64,
    median_vol: f64,
    b1_idx: Vec<u16>,
    b1_cnt: Vec<u16>,
    p1: Vec<u32>,
    s30: Vec<f32>, p30: Vec<f32>,
    s60: Vec<f32>, p60: Vec<f32>, c60: Vec<f32>,
    s300: Vec<f32>, p300: Vec<f32>,
    big10: Vec<f32>, big30: Vec<f32>,
}

/// 毫秒打包时间（24bit，0..14220000）。间隔 = 1 次 u32 减法（毫秒单位），
/// X9 是变异系数（尺度无关）不受影响；X8 阈值用 30000ms。
/// 排序精度为毫秒级：同毫秒不同股票的成交按合并顺序稳定决胜。
#[inline]
fn pack_time(t_us: i64, day_start_us: i64) -> u32 {
    ((t_us - day_start_us) / 1000) as u32
}

fn per_stock_prep(code: &str, recs: &[TradeRecord], day_start_us: i64) -> Option<StockData> {
    let n = recs.len();
    if n < MIN_TRADES {
        return None;
    }
    let mut times = Vec::with_capacity(n);
    let mut vols = Vec::with_capacity(n);
    let mut turnover = 0.0f64;
    for r in recs {
        times.push(r.time_us);
        vols.push(r.volume as f32);
        turnover += r.turnover;
    }
    let mut sorted = vols.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let th99 = sorted[((n as f64) * 0.99) as usize] as f64;
    let median_vol = sorted[n / 2] as f64;

    let mut packed: Vec<u32> = Vec::with_capacity(n);
    let mut big_times: Vec<u32> = Vec::new();
    let mut p1 = vec![0u32; BUCKET_1S];
    let mut s30 = vec![0f32; SCALE_30]; let mut p30 = vec![0f32; SCALE_30];
    let mut s60 = vec![0f32; SCALE_60]; let mut p60 = vec![0f32; SCALE_60];
    let mut c60 = vec![0f32; SCALE_60];
    let mut s300 = vec![0f32; SCALE_300]; let mut p300 = vec![0f32; SCALE_300];
    let mut big10 = vec![0f32; BIG10_T]; let mut big30 = vec![0f32; BIG30_T];
    let mut cnt1 = vec![0u16; BUCKET_1S];
    let mut nbig = 0usize;
    let mut b1_idx = Vec::new();
    let mut b1_cnt = Vec::new();
    for r in recs.iter() {
        let big = (r.volume as f64) >= th99;
        packed.push(pack_time(r.time_us, day_start_us));
        if big {
            big_times.push(pack_time(r.time_us, day_start_us));
            nbig += 1;
        }
        let off = r.time_us - day_start_us;
        let sec = (off / 1_000_000) as usize;
        // 剔除秒0(开盘集合竞价)与秒14219(收盘边界秒)，与 Python 版 [:, 1:] 的窗口对齐
        if sec > 0 && sec < BUCKET_1S - 1 {
            if cnt1[sec] == 0 {
                b1_idx.push(sec as u16);
            }
            cnt1[sec] = (cnt1[sec] + 1).min(u16::MAX);
        }
        let sign: f32 = match r.flag { 66 => 1.0, 83 => -1.0, _ => 0.0 };
        let sv = r.volume as f32 * sign;
        let v = r.volume as f32;
        let i30 = off / 30_000_000;
        if i30 < SCALE_30 as i64 {
            let i = i30 as usize;
            s30[i] += sv; p30[i] = r.price as f32;
        }
        let i60 = off / 60_000_000;
        if i60 < SCALE_60 as i64 {
            let i = i60 as usize;
            s60[i] += sv; p60[i] = r.price as f32; c60[i] += 1.0;
        }
        let i300 = off / 300_000_000;
        if i300 < SCALE_300 as i64 {
            let i = i300 as usize;
            s300[i] += sv; p300[i] = r.price as f32;
        }
    }
    let mut run = 0u32;
    let mut bi = 0usize;
    for t in 0..BUCKET_1S {
        if bi < b1_idx.len() && b1_idx[bi] as usize == t {
            run += cnt1[t] as u32;
            b1_cnt.push(cnt1[t]);
            bi += 1;
        }
        p1[t] = run;
    }

    // 大订单 → 带符号 10s/30s 桶
    let mut fills: Vec<(i64, i64, f32, f32)> = Vec::with_capacity(n);
    for r in recs {
        let (oid, sign) = match r.flag {
            66 => (r.bid_order, 1.0f32),
            83 => (r.ask_order, -1.0f32),
            _ => continue,
        };
        if oid == 0 { continue; }
        fills.push((oid, r.time_us, r.volume as f32, sign));
    }
    fills.sort_by_key(|f| f.0);
    {
        let mut i = 0usize;
        let mut totals: Vec<(i64, f64)> = Vec::new();
        while i < fills.len() {
            let oid = fills[i].0;
            let mut tot = 0.0f64;
            let mut j = i;
            while j < fills.len() && fills[j].0 == oid {
                tot += fills[j].2 as f64;
                j += 1;
            }
            totals.push((oid, tot));
            i = j;
        }
        let mut ti = 0usize;
        for &(oid, t, vol, sign) in &fills {
            while ti + 1 < totals.len() && totals[ti].0 < oid { ti += 1; }
            if totals[ti].0 == oid && totals[ti].1 >= th99 {
                let i10 = ((t - day_start_us) / 10_000_000) as usize;
                if i10 < BIG10_T { big10[i10] += sign * vol; }
                let i30 = ((t - day_start_us) / 30_000_000) as usize;
                if i30 < BIG30_T { big30[i30] += sign * vol; }
            }
        }
    }

    Some(StockData {
        code: code.to_string(),
        times: packed, big_times, n, nbig,
        th99, turnover, median_vol,
        b1_idx, b1_cnt, p1,
        s30, p30, s60, p60, c60, s300, p300,
        big10, big30,
    })
}

// ============================================================
// 合并流指标
// ============================================================
#[derive(Clone, Copy)]
struct MState {
    ia: usize,
    ib: usize,
    trans: u64,
    cc: [u64; 4],
    gsum: u64,
    gsq: u64,
    gcnt: u64,
    prev_t: u32,
    prev_label: u8,
    has_prev: bool,
}

#[inline(always)]
fn mstate_step(s: &mut MState, a: &[u32], b: &[u32]) {
    let ta = if s.ia < a.len() { a[s.ia] } else { u32::MAX };
    let tb = if s.ib < b.len() { b[s.ib] } else { u32::MAX };
    let take_a = ta <= tb;
    let t = if take_a { ta } else { tb };
    let label: u8 = if take_a { 0 } else { 1 };
    s.ia += take_a as usize;
    s.ib += !take_a as usize;
    if s.has_prev {
        s.trans += (label != s.prev_label) as u64;
        s.cc[(s.prev_label as usize) << 1 | label as usize] += 1;
        let g = (t - s.prev_t) as u64;
        s.gsum += g;
        s.gsq += g * g;
        s.gcnt += 1;
    }
    s.prev_t = t;
    s.prev_label = label;
    s.has_prev = true;
}


#[inline]
fn mi_from_cc(cc: &[u64; 4], n: usize) -> f32 {
    if n <= 1 {
        return 0.0;
    }
    let denom = (n - 1) as f64;
    let (p00, p01, p10, p11) = (cc[0] as f64 / denom, cc[1] as f64 / denom, cc[2] as f64 / denom, cc[3] as f64 / denom);
    let (p0, p1) = (p00 + p01, p10 + p11);
    let mut mi = 0.0;
    for (pab, pa, pb) in [(p00, p0, p0), (p01, p0, p1), (p10, p1, p0), (p11, p1, p1)] {
        if pab > 0.0 && pa > 0.0 && pb > 0.0 {
            mi += pab * (pab / (pa * pb)).ln();
        }
    }
    mi as f32
}

#[inline]
fn merge_x139(a: &[u32], b: &[u32]) -> (f32, f32, f32) {
    let na = a.len();
    let nb = b.len();
    let n = na + nb;
    let mut trans: u64 = 0;
    let mut cc: [u64; 4] = [0; 4];
    let (mut gsum, mut gsq): (u64, u64) = (0, 0);
    let mut gcnt: u64 = 0;
    let (mut prev_t, mut prev_label): (u32, u8) = (0, 2);
    let mut has_prev = false;
    let (mut ia, mut ib) = (0usize, 0usize);
    while ia < na || ib < nb {
        let (label, t) = if ib >= nb || (ia < na && a[ia] <= b[ib]) {
            let l = (0u8, a[ia]);
            ia += 1;
            l
        } else {
            let l = (1u8, b[ib]);
            ib += 1;
            l
        };
        if has_prev {
            trans += (label != prev_label) as u64;
            cc[(prev_label as usize) << 1 | label as usize] += 1;
            let g = (t - prev_t) as u64; // ms
            gsum += g;
            gsq += g * g;
            gcnt += 1;
        }
        prev_t = t;
        prev_label = label;
        has_prev = true;
    }
    let x1 = if n > 1 { trans as f32 / (n - 1) as f32 } else { 0.0 };
    let x9 = if gcnt > 1 && gsum > 0 {
        let mean = gsum as f64 / gcnt as f64;
        let var = (gsq as f64 - gsum as f64 * gsum as f64 / gcnt as f64) / (gcnt - 1) as f64;
        if var > 0.0 { (var.sqrt() / mean) as f32 } else { 0.0 }
    } else {
        0.0
    };
    (x1, mi_from_cc(&cc, n), x9)
}

#[inline]
fn merge_x139_multi(a: &[u32], bs: &[&[u32]], out: &mut [(f32, f32, f32)]) {
    let mut st = [MState {
        ia: 0, ib: 0, trans: 0, cc: [0; 4], gsum: 0, gsq: 0, gcnt: 0,
        prev_t: 0, prev_label: 2, has_prev: false,
    }; 4];
    loop {
        let mut any = false;
        for k in 0..out.len() {
            let s = &mut st[k];
            if s.ia < a.len() || s.ib < bs[k].len() {
                mstate_step(s, a, bs[k]);
                any = true;
            }
        }
        if !any {
            break;
        }
    }
    for (k, o) in out.iter_mut().enumerate() {
        let s = &st[k];
        let n = a.len() + bs[k].len();
        let x1 = if n > 1 { s.trans as f32 / (n - 1) as f32 } else { 0.0 };
        let x3 = mi_from_cc(&s.cc, n);
        let x9 = if s.gcnt > 1 && s.gsum > 0 {
            let mean = s.gsum as f64 / s.gcnt as f64;
            let var = (s.gsq as f64 - s.gsum as f64 * s.gsum as f64 / s.gcnt as f64) / (s.gcnt - 1) as f64;
            if var > 0.0 { (var.sqrt() / mean) as f32 } else { 0.0 }
        } else {
            0.0
        };
        *o = (x1, x3, x9);
    }
}

#[inline]
fn merge_x8(a: &[u32], b: &[u32]) -> f32 {
    let mut h = [0u64; 33];
    let mut ngaps: u64 = 0;
    let mut last: u32 = u32::MAX;
    let (mut ia, mut ib) = (0usize, 0usize);
    while ia < a.len() || ib < b.len() {
        let ta = if ia < a.len() { a[ia] } else { u32::MAX };
        let tb = if ib < b.len() { b[ib] } else { u32::MAX };
        let take_a = ta <= tb;
        let t = if take_a { ta } else { tb };
        ia += take_a as usize;
        ib += !take_a as usize;
        if last != u32::MAX {
            let gap_ms = (t - last) as u64; // 毫秒
            let bin = if gap_ms >= 320_000 { 32usize } else { (gap_ms / 10_000) as usize };
            h[bin] += 1;
            ngaps += 1;
        }
        last = t;
    }
    if ngaps > 0 { (h[0] + h[1] + h[2]) as f32 / ngaps as f32 } else { 0.0 }
}

// ============================================================
// 矩阵工具
// ============================================================
/// numpy 线性插值分位数（与 np.percentile 默认 linear 一致）
fn np_quantile(s: &[f32], p: f64) -> f32 {
    let n = s.len();
    if n == 1 {
        return s[0];
    }
    let idx = p * (n - 1) as f64;
    let j = idx.floor() as usize;
    let frac = (idx - j as f64) as f32;
    let v0 = s[j];
    let v1 = s[(j + 1).min(n - 1)];
    v0 + (v1 - v0) * frac
}

fn winsorize_rows(M: &mut [f32], _n: usize, t: usize) {
    M.par_chunks_mut(t).for_each(|row| {
        let mut s = row.to_vec();
        s.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let lo = np_quantile(&s, 0.02);
        let hi = np_quantile(&s, 0.98);
        for v in row.iter_mut() {
            *v = v.max(lo).min(hi);
        }
    });
}

fn zscore_rows(M: &mut [f32], _n: usize, t: usize) {
    M.par_chunks_mut(t).for_each(|row| {
        let mean: f64 = row.iter().map(|&v| v as f64).sum::<f64>() / t as f64;
        let var: f64 = row.iter().map(|&v| (v as f64 - mean) * (v as f64 - mean)).sum::<f64>() / t as f64;
        if var <= 1e-24 {
            for v in row.iter_mut() { *v = 0.0; }
            return;
        }
        let sd = var.sqrt() as f32;
        for v in row.iter_mut() { *v = (*v - mean as f32) / sd; }
    });
}

/// 全矩阵 z-score（非对角）
fn zscore_mat(M: &mut [f32], n: usize) {
    let mut sum = 0.0f64;
    let mut sq = 0.0f64;
    let mut cnt = 0.0f64;
    for i in 0..n {
        for j in 0..n {
            if i != j {
                let v = M[i * n + j] as f64;
                sum += v;
                sq += v * v;
                cnt += 1.0;
            }
        }
    }
    let mu = sum / cnt;
    let sd = (sq / cnt - mu * mu).sqrt();
    if sd < 1e-15 {
        for v in M.iter_mut() { *v = 0.0; }
        return;
    }
    let mu32 = mu as f32;
    let sd32 = sd as f32;
    M.par_chunks_mut(n).enumerate().for_each(|(i, row)| {
        for (j, v) in row.iter_mut().enumerate() {
            *v = if i != j { (*v - mu32) / sd32 } else { 0.0 };
        }
    });
}

fn write_f32(path: &str, data: &[f32]) -> std::io::Result<()> {
    let mut f = std::fs::File::create(path)?;
    let bytes: &[u8] = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) };
    f.write_all(bytes)
}

// ============================================================
// 谱分解: 确定性幂迭代 + GS
// ============================================================
fn matvec_sym(M: &[f32], v: &[f32], n: usize, out: &mut [f32]) {
    out.par_chunks_mut(64).enumerate().for_each(|(ci, chunk)| {
        let i0 = ci * 64;
        for (ii, o) in chunk.iter_mut().enumerate() {
            let i = i0 + ii;
            let row = &M[i * n..(i + 1) * n];
            let mut acc = 0.0f32;
            for t in 0..n {
                acc = f32::mul_add(row[t], v[t], acc);
            }
            *o = acc;
        }
    });
}

fn top_eigenvectors(M: &[f32], n: usize, k: usize, max_iter: usize) -> Vec<f32> {
    let mut vecs = vec![0f32; n * k];
    let mut seed: u64 = 0x9E3779B97F4A7C15;
    for j in 0..k {
        let mut v: Vec<f32> = (0..n)
            .map(|_| {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((seed >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0
            })
            .collect();
        let mut w = vec![0f32; n];
        for _ in 0..max_iter {
            matvec_sym(M, &v, n, &mut w);
            for jj in 0..j {
                let u = &vecs[jj * n..(jj + 1) * n];
                let mut d = 0.0f32;
                for i in 0..n {
                    d += u[i] * w[i];
                }
                for i in 0..n {
                    w[i] -= d * u[i];
                }
            }
            let mut norm = 0.0f32;
            for i in 0..n {
                norm = f32::mul_add(w[i], w[i], norm);
            }
            norm = norm.sqrt();
            if norm < 1e-9 {
                v.iter_mut().for_each(|x| *x = 0.0);
                break;
            }
            for i in 0..n {
                w[i] /= norm;
            }
            let mut c = 0.0f32;
            for i in 0..n {
                c += w[i] * v[i];
            }
            v.copy_from_slice(&w);
            if c > 1.0 - 1e-4 {
                break;
            }
        }
        vecs[j * n..(j + 1) * n].copy_from_slice(&v);
    }
    for j in 0..k {
        let col = &mut vecs[j * n..(j + 1) * n];
        let mut mi = 0usize;
        for (i, &x) in col.iter().enumerate() {
            if x.abs() > col[mi].abs() {
                mi = i;
            }
        }
        if col[mi] < 0.0 {
            for x in col.iter_mut() {
                *x = -*x;
            }
        }
    }
    vecs
}

// ============================================================
// 主流程
// ============================================================
fn main() {
    let args: Vec<String> = std::env::args().collect();
    let date: i64 = args.get(1).map(|s| s.parse().unwrap()).unwrap_or(20241231);
    let outdir = args.get(2).cloned().unwrap_or_else(|| "/home/chenzongwei/pair_x_out".to_string());
    let write_mats = args.iter().any(|a| a == "--write-mats");
    let limit: Option<usize> = args.get(3).and_then(|s| s.parse().ok());
    let universe: Option<usize> = args.iter().position(|a| a == "--universe").map(|p| args[p + 1].parse().unwrap());
    let outdir = format!("{outdir}/{date}");
    std::fs::create_dir_all(&outdir).unwrap();

    let t_all = std::time::Instant::now();
    let mut codes = list_codes(date);
    if let Some(u) = universe {
        // 保留按文件大小排序后的前 u 只（先按大小排序，再截断，再恢复代码序）
        codes.sort_by(|a, b| {
            let (fa, fb) = (format!("/ssd_data/stock/{date}/transaction/{a}_{date}_transaction.csv"),
                            format!("/ssd_data/stock/{date}/transaction/{b}_{date}_transaction.csv"));
            let sa = std::fs::metadata(&fa).map(|m| m.len()).unwrap_or(0);
            let sb = std::fs::metadata(&fb).map(|m| m.len()).unwrap_or(0);
            sb.cmp(&sa).then_with(|| a.cmp(b))
        });
        codes.truncate(u);
        codes.sort();
    } else if let Some(l) = limit {
        codes.truncate(l);
    }
    eprintln!("[stage1] codes = {}", codes.len());

    let day_start_us = days_from_civil(date / 10000, (date / 100 % 100) as u64, (date % 100) as u64)
        * 86400 * 1_000_000
        + (9 * 3600 + 30 * 60) * 1_000_000;

    let t0 = std::time::Instant::now();
    let preps: Vec<Option<StockData>> = codes
        .par_iter()
        .map(|c| read_trade_fast(c, date).ok().and_then(|recs| per_stock_prep(c, &recs, day_start_us)))
        .collect();
    let stocks: Vec<StockData> = preps.into_iter().flatten().collect();
    let n = stocks.len();
    let total_trades: usize = stocks.iter().map(|s| s.n).sum();
    eprintln!("[stage2] valid = {n}, trades = {total_trades}, {:.1}s", t0.elapsed().as_secs_f32());

    // ---- stage3: 合并流 X1/X3/X8/X9 ----
    let t0 = std::time::Instant::now();
    let mut m_x1 = vec![0f32; n * n];
    let mut m_x3 = vec![0f32; n * n];
    let mut m_x8 = vec![0f32; n * n];
    let mut m_x9 = vec![0f32; n * n];
    let times: Vec<&[u32]> = stocks.iter().map(|s| s.times.as_slice()).collect();
    let bigs: Vec<&[u32]> = stocks.iter().map(|s| s.big_times.as_slice()).collect();
    m_x1.par_chunks_mut(n)
        .zip(m_x3.par_chunks_mut(n))
        .zip(m_x8.par_chunks_mut(n))
        .zip(m_x9.par_chunks_mut(n))
        .enumerate()
        .for_each(|(i, (((r1, r3), r8), r9))| {
            let ti = times[i];
            let bi = bigs[i];
            for j in 0..n {
                if i == j { continue; }
                let (x1, x3, x9) = merge_x139(ti, times[j]);
                r1[j] = x1; r3[j] = x3; r9[j] = x9;
                r8[j] = merge_x8(bi, bigs[j]);
            }
        });
    for i in 0..n {
        for j in (i + 1)..n {
            m_x1[j * n + i] = m_x1[i * n + j];
            m_x3[j * n + i] = m_x3[i * n + j];
            m_x8[j * n + i] = m_x8[i * n + j];
            m_x9[j * n + i] = m_x9[i * n + j];
        }
    }
    eprintln!("[stage3] merge {:.1}s", t0.elapsed().as_secs_f32());

    // ---- stage4: X2 ----
    let t0 = std::time::Instant::now();
    let t1s = BUCKET_1S;
    let mut c_dense = vec![0f32; n * t1s];
    c_dense.par_chunks_mut(t1s).enumerate().for_each(|(i, row)| {
        let s = &stocks[i];
        for (&idx, &cnt) in s.b1_idx.iter().zip(s.b1_cnt.iter()) {
            if idx > 0 {
                row[idx as usize] = cnt as f32;
            }
        }
    });
    let mut x2_mats: Vec<Vec<f32>> = Vec::new();
    for &dt in X2_DT.iter() {
        let mut w = vec![0f32; n * t1s];
        w.par_chunks_mut(t1s).enumerate().for_each(|(j, row)| {
            let p = &stocks[j].p1;
            for t in 0..t1s {
                let hi = (t + dt).min(t1s - 2);   // Python 版 T2=14219 → 上界 14218
                let lo = t.checked_sub(dt + 1);
                row[t] = p[hi] as f32 - lo.map(|x| p[x] as f32).unwrap_or(0.0);
            }
        });
        let mut wt = vec![0f32; t1s * n];
        transpose(n, t1s, &w, &mut wt);
        drop(w);
        let mut u = vec![0f32; n * n];
        gemm_abt(&c_dense, &wt, n, n, t1s, &mut u);
        for i in 0..n {
            let ni = stocks[i].n as f32;
            let row = &mut u[i * n..(i + 1) * n];
            for v in row.iter_mut() {
                *v /= ni;
            }
            row[i] = 0.0;
        }
        x2_mats.push(u);
    }
    drop(c_dense);
    eprintln!("[stage4] X2 {:.1}s", t0.elapsed().as_secs_f32());

    // ---- stage5: X4/X5/X6 ----
    let t0 = std::time::Instant::now();
    let mut x4_mats: Vec<Vec<f32>> = Vec::new();
    let mut x5lag_mats: Vec<Vec<f32>> = Vec::new();
    let mut x5str_mats: Vec<Vec<f32>> = Vec::new();
    let mut x6_mats: Vec<Vec<f32>> = Vec::new();
    for (si, (sname, t, kk)) in [("30", SCALE_30, 10usize), ("60", SCALE_60, 5usize), ("300", SCALE_300, 1usize)].iter().enumerate() {
        let (t, kk) = (*t, *kk);
        let mut s = vec![0f32; n * t];
        let mut p = vec![0f32; n * t];
        for (i, st) in stocks.iter().enumerate() {
            let (ss, pp): (&[f32], &[f32]) = match si {
                0 => (&st.s30, &st.p30),
                1 => (&st.s60, &st.p60),
                _ => (&st.s300, &st.p300),
            };
            s[i * t..(i + 1) * t].copy_from_slice(ss);
            p[i * t..(i + 1) * t].copy_from_slice(pp);
        }
        let t_trim = t - 2;
        let mut s2 = vec![0f32; n * t_trim];
        for i in 0..n {
            s2[i * t_trim..(i + 1) * t_trim].copy_from_slice(&s[i * t + 1..i * t + t - 1]);
        }
        drop(s);
        winsorize_rows(&mut s2, n, t_trim);
        zscore_rows(&mut s2, n, t_trim);
        let mut st2 = vec![0f32; t_trim * n];
        transpose(n, t_trim, &s2, &mut st2);
        let mut x4 = vec![0f32; n * n];
        gemm_abt(&s2, &st2, n, n, t_trim, &mut x4);
        let scale4 = 1.0 / (t_trim - 1) as f32;
        for v in x4.iter_mut() { *v *= scale4; }
        for i in 0..n { x4[i * n + i] = 0.0; }
        x4_mats.push(x4);

        let mut best_k = vec![0f32; n * n];
        let mut best_v = vec![0f32; n * n];
        for k in 1..=kk {
            let len = t_trim - k;
            let mut a = vec![0f32; n * len];
            let mut b = vec![0f32; n * len];
            for i in 0..n {
                a[i * len..(i + 1) * len].copy_from_slice(&s2[i * t_trim..i * t_trim + len]);
                b[i * len..(i + 1) * len].copy_from_slice(&s2[i * t_trim + k..i * t_trim + t_trim]);
            }
            let mut bt = vec![0f32; len * n];
            transpose(n, len, &b, &mut bt);
            drop(b);
            let mut ck = vec![0f32; n * n];
            gemm_abt(&a, &bt, n, n, len, &mut ck);
            let scale = 1.0 / len as f32;
            for v in ck.iter_mut() { *v *= scale; }
            let lag = k as f32 * sname.parse::<f32>().unwrap() / 60.0;
            for i in 0..n {
                for j in 0..n {
                    // +k 候选: |Ck[i][j]| ; −k 候选: |Ck[j][i]|（同一格子 (i,j) 的 −k 方向）
                    let vp = ck[i * n + j].abs();
                    let vm = ck[j * n + i].abs();
                    if vp > best_v[i * n + j] {
                        best_v[i * n + j] = vp;
                        best_k[i * n + j] = lag;
                    }
                    if vm > best_v[i * n + j] {
                        best_v[i * n + j] = vm;
                        best_k[i * n + j] = -lag;
                    }
                }
            }
        }
        for i in 0..n {
            best_k[i * n + i] = 0.0;
            best_v[i * n + i] = 0.0;
        }
        x5lag_mats.push(best_k);
        x5str_mats.push(best_v);

        let mut r = vec![0f32; n * t_trim];
        for i in 0..n {
            let pr = &p[i * t + 1..i * t + t - 1];
            let mut prev = pr[0];
            for (idx, &px) in pr.iter().enumerate() {
                let rr = if prev > 0.0 { px / prev - 1.0 } else { 0.0 };
                r[i * t_trim + idx] = rr;
                prev = px;
            }
        }
        drop(p);
        winsorize_rows(&mut r, n, t_trim);
        let mut zr = vec![0f32; n * (t_trim - 1)];
        for i in 0..n {
            zr[i * (t_trim - 1)..(i + 1) * (t_trim - 1)].copy_from_slice(&r[i * t_trim + 1..i * t_trim + t_trim]);
        }
        drop(r);
        zscore_rows(&mut zr, n, t_trim - 1);
        let mut zs1 = vec![0f32; n * (t_trim - 1)];
        for i in 0..n {
            zs1[i * (t_trim - 1)..(i + 1) * (t_trim - 1)].copy_from_slice(&s2[i * t_trim..i * t_trim + t_trim - 1]);
        }
        let mut zrt = vec![0f32; (t_trim - 1) * n];
        transpose(n, t_trim - 1, &zr, &mut zrt);
        drop(zr);
        let mut x6 = vec![0f32; n * n];
        gemm_abt(&zs1, &zrt, n, n, t_trim - 1, &mut x6);
        let scale6 = 1.0 / (t_trim - 2) as f32;
        for v in x6.iter_mut() { *v *= scale6; }
        for i in 0..n { x6[i * n + i] = 0.0; }
        x6_mats.push(x6);
    }
    eprintln!("[stage5] X4/X5/X6 {:.1}s", t0.elapsed().as_secs_f32());

    // ---- stage6: X7 ----
    let t0 = std::time::Instant::now();
    let mut x7_mats: Vec<Vec<f32>> = Vec::new();
    for (si, t) in [BIG10_T, BIG30_T].iter().enumerate() {
        let t = *t;
        let t_eff = t - 1; // 剔除第0桶与末桶（与 Python 版一致）
        let mut bp = vec![0f32; n * t_eff];
        let mut bn = vec![0f32; n * t_eff];
        let mut tot = vec![0f64; n];
        for (i, st) in stocks.iter().enumerate() {
            let big: &[f32] = if si == 0 { &st.big10 } else { &st.big30 };
            let mut s = 0.0f64;
            for idx in 1..t_eff {
                let v = big[idx];
                s += v.abs() as f64;
                if v > 0.0 { bp[i * t_eff + idx - 1] = v; } else { bn[i * t_eff + idx - 1] = -v; }
            }
            tot[i] = s;
        }
        let mut bpt = vec![0f32; t_eff * n];
        let mut bnt = vec![0f32; t_eff * n];
        transpose(n, t_eff, &bp, &mut bpt);
        transpose(n, t_eff, &bn, &mut bnt);
        let mut x7 = vec![0f32; n * n];
        let mut tmp = vec![0f32; n * n];
        gemm_abt(&bp, &bpt, n, n, t_eff, &mut x7);
        gemm_abt(&bn, &bnt, n, n, t_eff, &mut tmp);
        for i in 0..n {
            for j in 0..n {
                let denom = (tot[i] * tot[j]) as f32;
                x7[i * n + j] = if denom > 0.0 { (x7[i * n + j] + tmp[i * n + j]) / denom } else { 0.0 };
            }
            x7[i * n + i] = 0.0;
        }
        x7_mats.push(x7);
    }
    eprintln!("[stage6] X7 {:.1}s", t0.elapsed().as_secs_f32());

    // ---- stage7: 派生矩阵 + 组合矩阵 + 谱分解 ----
    let t0 = std::time::Instant::now();
    // 注册表: 名字 → 矩阵（行统计/交互层共用）
    let mut reg: BTreeMap<String, Vec<f32>> = BTreeMap::new();
    reg.insert("x1".into(), m_x1.clone());
    reg.insert("x3".into(), m_x3.clone());
    reg.insert("x8".into(), m_x8.clone());
    reg.insert("x9".into(), m_x9.clone());
    for (i, &dt) in X2_DT.iter().enumerate() {
        reg.insert(format!("x2_{dt}s"), x2_mats[i].clone());
        let mut sm = x2_mats[i].clone();
        let mut am = x2_mats[i].clone();
        for a in 0..n {
            for b in 0..n {
                let v = x2_mats[i][a * n + b];
                sm[a * n + b] = v + x2_mats[i][b * n + a];
                am[a * n + b] = v - x2_mats[i][b * n + a];
            }
        }
        reg.insert(format!("x2_{dt}s_sum"), sm);
        reg.insert(format!("x2_{dt}s_asy"), am);
    }
    for (i, sname) in ["30s", "60s", "300s"].iter().enumerate() {
        reg.insert(format!("x4_{sname}"), x4_mats[i].clone());
        reg.insert(format!("x5lag_{sname}"), x5lag_mats[i].clone());
        reg.insert(format!("x5str_{sname}"), x5str_mats[i].clone());
        reg.insert(format!("x6_{sname}"), x6_mats[i].clone());
    }
    for (i, sname) in ["10s", "30s"].iter().enumerate() {
        reg.insert(format!("x7_{sname}"), x7_mats[i].clone());
    }
    // 组合矩阵
    fn zsum(reg: &BTreeMap<String, Vec<f32>>, n: usize, names: &[&str]) -> Vec<f32> {
        let mut acc = vec![0f32; n * n];
        for nm in names {
            let m = reg.get(*nm).unwrap();
            let mut z = m.clone();
            zscore_mat(&mut z, n);
            for k in 0..n * n { acc[k] += z[k]; }
        }
        acc
    }
    let combo_defs: [(&str, Vec<&str>); 6] = [
        ("combo1", vec!["x4_60s", "x7_30s", "x1"]),
        ("combo2", vec!["x2_10s_sum", "x5str_60s"]),
        ("combo3", vec!["x3", "x8", "x9"]),
        ("combo4", vec!["x4_300s", "x5str_300s", "x2_60s_sum"]),
        ("combo5", vec!["x4_30s", "x7_10s", "x2_30s_sum"]),
        ("combo6", vec!["x5str_30s", "x5str_300s", "x4_60s"]),
    ];
    let mut combos: Vec<(String, Vec<f32>)> = Vec::new();
    for (nm, parts) in combo_defs.iter() {
        let mut c = zsum(&reg, n, &parts);
        if *nm == "combo2" {
            // 加上 x6_60s 的对称化
            let mut x6s = x6_mats[1].clone();
            for a in 0..n {
                for b in 0..n {
                    x6s[a * n + b] = (x6_mats[1][a * n + b] + x6_mats[1][b * n + a]) * 0.5;
                }
            }
            zscore_mat(&mut x6s, n);
            for k in 0..n * n { c[k] += x6s[k]; }
        }
        combos.push((nm.to_string(), c.clone()));
        reg.insert(nm.to_string(), c);
    }
    eprintln!("[stage7a] derived+combos {:.1}s", t0.elapsed().as_secs_f32());

    // 谱分解（9 对称 + 6 组合）
    let mut spectral_mats: Vec<(String, Vec<f32>)> = Vec::new();
    let sym_names: [&str; 9] = ["x1", "x3", "x8", "x9", "x4_30s", "x4_60s", "x4_300s", "x7_10s", "x7_30s"];
    let spec_names: Vec<&str> = sym_names.iter().copied().chain(combo_defs.iter().map(|(nm, _)| *nm)).collect();
    let t_spec = std::time::Instant::now();
    spectral_mats = spec_names
        .par_iter()
        .map(|nm| {
            let m = reg.get(*nm).unwrap();
            let mut mn = vec![0f32; n * n];
            for i in 0..n {
                let mut norm = 0.0f64;
                let row = &m[i * n..(i + 1) * n];
                for &v in row.iter() { norm += (v as f64) * (v as f64); }
                norm = norm.sqrt();
                if norm < 1e-9 { norm = 1.0; }
                let inv = (1.0 / norm) as f32;
                for j in 0..n {
                    mn[i * n + j] = m[i * n + j] * inv;
                }
            }
            let mut ms = vec![0f32; n * n];
            for i in 0..n {
                for j in 0..n {
                    ms[i * n + j] = (mn[i * n + j] + mn[j * n + i]) * 0.5;
                }
            }
            drop(mn);
            let vecs = top_eigenvectors(&ms, n, SPECTRAL_K, SPECTRAL_MAX_ITER);
            (nm.to_string(), vecs)
        })
        .collect();
    eprintln!("[stage7b] spectral {:.1}s", t_spec.elapsed().as_secs_f32());
    eprintln!("[stage7] total {:.1}s", t0.elapsed().as_secs_f32());

    // ---- stage8: 因子提取 ----
    let t0 = std::time::Instant::now();
    let (names, vals) = extract_factors(n, &reg, &combos, &spectral_mats, &stocks);
    eprintln!("[stage8] factors {:.1}s, n_factors = {}", t0.elapsed().as_secs_f32(), names.len());

    // ---- stage9: 输出 ----
    let t0 = std::time::Instant::now();
    let codes_out: Vec<String> = stocks.iter().map(|s| s.code.clone()).collect();
    std::fs::write(format!("{outdir}/codes2.txt"), codes_out.join("\n")).unwrap();
    std::fs::write(format!("{outdir}/names2.txt"), names.join("\n")).unwrap();
    write_f32(&format!("{outdir}/factors2.bin"), &vals).unwrap();
    if write_mats {
        for (nm, m) in reg.iter() {
            if nm.starts_with("combo") { continue; }
            write_f32(&format!("{outdir}/m2_{nm}.bin"), m).unwrap();
        }
    }
    eprintln!("[stage9] write {:.1}s", t0.elapsed().as_secs_f32());
    eprintln!("ALL DONE {:.1}s, n_stocks = {n}, n_factors = {}", t_all.elapsed().as_secs_f32(), names.len());
    println!("{{\"ok\":true,\"date\":{date},\"n\":{n},\"n_factors\":{},\"outdir\":\"{outdir}\"}}", names.len());
}

// ============================================================
// 因子提取（五层）
// ============================================================
const TOPK: [usize; 19] = [1, 2, 3, 5, 8, 10, 15, 20, 30, 50, 80, 100, 150, 200, 300, 500, 800, 1200, 2000];
const TOPK_ABS: [usize; 5] = [3, 10, 30, 100, 300];
const DEG_P: [f64; 8] = [0.70, 0.80, 0.85, 0.90, 0.95, 0.975, 0.99, 0.995];

fn row_feature_names(asym: bool) -> Vec<String> {
    let mut names: Vec<String> = Vec::new();
    for s in ["mean", "std", "skew", "kurt", "min", "max", "median", "iqr",
              "q01", "q02", "q05", "q10", "q25", "q75", "q90", "q95", "q98", "q99"] {
        names.push(s.to_string());
    }
    for s in ["mean_abs", "max_abs", "sum_abs", "q95_abs", "q99_abs"] {
        names.push(s.to_string());
    }
    for &k in TOPK.iter() {
        names.push(format!("topk{k}_mean"));
    }
    for &k in TOPK_ABS.iter() {
        names.push(format!("topk_abs{k}_mean"));
    }
    for &p in DEG_P.iter() {
        names.push(format!("deg_p{}", p * 100.0));
    }
    names.push("hhi".into());
    names.push("entropy".into());
    for s in ["cnt_pos", "cnt_neg", "sum_pos", "sum_neg", "mean_pos", "mean_neg", "topk3_pos_mean", "topk3_neg_mean"] {
        names.push(s.to_string());
    }
    if asym {
        for s in ["row_mean", "col_mean", "net", "row_topk5", "col_topk5", "row_topk50", "col_topk50", "row_topk300", "col_topk300"] {
            names.push(s.to_string());
        }
    }
    names
}

/// 每行特征（与 row_feature_names 顺序一致），并返回 topk/bottomk 索引供邻域特征用
struct RowFeats {
    names: Vec<String>,
    cols: Vec<Vec<f32>>,      // 每列 N 长
    topk_idx: [Vec<Vec<u32>>; 2],  // k=5, k=20 的 top 索引（每行）
    bottom_idx: [Vec<Vec<u32>>; 2], // k=5, k=20 的 bottom 索引
    mean_col: Vec<f32>,        // 行均值列（供 ratio 特征）
}

/// 直方图分位数（65536 桶, f32 位模式分桶, 确定性, 免全排序; 相对分辨率 ~0.8%）
fn hist_quantiles(vals: &[f32], ps: &[f64]) -> Vec<f32> {
    let mut hist = vec![0u64; 65536];
    for &v in vals.iter() {
        let b = ((v.to_bits() >> 16) & 0xFFFF) as usize;
        hist[b] += 1;
    }
    let total = vals.len() as f64;
    let mut out = Vec::with_capacity(ps.len());
    let mut acc = 0u64;
    let mut bi = 0usize;
    for &p in ps.iter() {
        let target = (p * total) as u64;
        while bi < 65536 && acc < target {
            acc += hist[bi];
            bi += 1;
        }
        let bits = ((bi as u32) << 16) | (1u32 << 15);
        out.push(f32::from_bits(bits.min(u32::MAX)));
    }
    out
}

fn row_features(m: &[f32], n: usize, asym: bool, prefix: &str) -> RowFeats {
    // 全局 |非对角| 分位数（直方图, 免全排序）
    let mut abs_all: Vec<f32> = Vec::with_capacity(n * n / 8);
    let mut step = 1usize;
    if n > 2000 { step = 2; }
    for i in (0..n).step_by(step) {
        let row = &m[i * n..(i + 1) * n];
        for j in 0..n {
            if i != j {
                abs_all.push(row[j].abs());
            }
        }
    }
    let deg_th: Vec<f32> = hist_quantiles(&abs_all, &DEG_P);
    drop(abs_all);

    let rows: Vec<Vec<(f32, u32)>> = (0..n)
        .into_par_iter()
        .map(|i| {
            let mut v: Vec<(f32, u32)> = (0..n)
                .filter(|&j| j != i)
                .map(|j| (m[i * n + j], j as u32))
                .collect();
            v.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            v
        })
        .collect();

    let names = row_feature_names(asym);
    let fcnt = names.len();
    let mut cols: Vec<Vec<f32>> = vec![Vec::with_capacity(n); fcnt];
    let mut topk_idx: [Vec<Vec<u32>>; 2] = [vec![Vec::new(); n], vec![Vec::new(); n]];
    let mut bottom_idx: [Vec<Vec<u32>>; 2] = [vec![Vec::new(); n], vec![Vec::new(); n]];
    let mut mean_col = vec![0f32; n];

    for i in 0..n {
        let r = &rows[i];
        let nn = r.len();
        let qv = |p: f64| -> f64 { r[((nn as f64) * p) as usize].0 as f64 };
        let mean: f64 = r.iter().map(|x| x.0 as f64).sum::<f64>() / nn as f64;
        let mut s2 = 0.0f64; let mut s3 = 0.0f64; let mut s4 = 0.0f64;
        for x in r.iter() {
            let d = x.0 as f64 - mean;
            s2 += d * d; s3 += d * d * d; s4 += d * d * d * d;
        }
        let std = (s2 / nn as f64).sqrt();
        let skew = if std > 1e-12 { s3 / nn as f64 / std.powi(3) } else { 0.0 };
        let kurt = if std > 1e-12 { s4 / nn as f64 / std.powi(4) - 3.0 } else { 0.0 };
        let mut feats: Vec<f64> = vec![
            mean, std, skew, kurt,
            r[0].0 as f64, r[nn - 1].0 as f64,
            qv(0.5),
            qv(0.75) - qv(0.25),
            qv(0.01), qv(0.02), qv(0.05), qv(0.10), qv(0.25), qv(0.75), qv(0.90), qv(0.95), qv(0.98), qv(0.99),
        ];
        let mean_abs: f64 = r.iter().map(|x| x.0.abs() as f64).sum::<f64>() / nn as f64;
        let mut abs_sorted: Vec<f32> = r.iter().map(|x| x.0.abs()).collect();
        abs_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let max_abs = abs_sorted[nn - 1] as f64;
        let sum_abs: f64 = abs_sorted.iter().map(|&x| x as f64).sum();
        let q95a = abs_sorted[((nn as f64) * 0.95) as usize] as f64;
        let q99a = abs_sorted[((nn as f64) * 0.99) as usize] as f64;
        let mut topk_feats: Vec<f64> = Vec::new();
        for &k in TOPK.iter() {
            let kk = k.min(nn);
            let s: f64 = r[nn - kk..].iter().map(|x| x.0 as f64).sum();
            topk_feats.push(s / kk as f64);
        }
        let mut topk_abs_feats: Vec<f64> = Vec::new();
        for &k in TOPK_ABS.iter() {
            let kk = k.min(nn);
            let s: f64 = abs_sorted[nn - kk..].iter().map(|&x| x as f64).sum();
            topk_abs_feats.push(s / kk as f64);
        }
        let mut deg_feats: Vec<f64> = Vec::new();
        for &th in deg_th.iter() {
            deg_feats.push(r.iter().filter(|x| x.0.abs() as f32 > th).count() as f64);
        }
        let sumabs = if sum_abs > 1e-30 { sum_abs } else { 1.0 };
        let mut hhi = 0.0f64;
        let mut ent = 0.0f64;
        for x in r.iter() {
            let w = x.0.abs() as f64 / sumabs;
            hhi += w * w;
            if w > 1e-300 { ent -= w * w.ln(); }
        }
        let mut cnt_pos = 0usize; let mut cnt_neg = 0usize;
        let mut sum_pos = 0.0f64; let mut sum_neg = 0.0f64;
        let mut posv: Vec<f32> = Vec::new(); let mut negv: Vec<f32> = Vec::new();
        for x in r.iter() {
            if x.0 > 0.0 { cnt_pos += 1; sum_pos += x.0 as f64; posv.push(x.0); }
            else if x.0 < 0.0 { cnt_neg += 1; sum_neg += x.0 as f64; negv.push(x.0); }
        }
        posv.sort_by(|a, b| a.partial_cmp(b).unwrap());
        negv.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let top3p = if posv.len() >= 3 { posv[posv.len() - 3..].iter().map(|&x| x as f64).sum::<f64>() / 3.0 } else { posv.iter().map(|&x| x as f64).sum::<f64>() / posv.len().max(1) as f64 };
        let top3n = if negv.len() >= 3 { negv[negv.len() - 3..].iter().map(|&x| x as f64).sum::<f64>() / 3.0 } else { negv.iter().map(|&x| x as f64).sum::<f64>() / negv.len().max(1) as f64 };
        feats.extend([
            mean_abs, max_abs, sum_abs, q95a, q99a,
            cnt_pos as f64, cnt_neg as f64, sum_pos, sum_neg,
            if cnt_pos > 0 { sum_pos / cnt_pos as f64 } else { 0.0 },
            if cnt_neg > 0 { sum_neg / cnt_neg as f64 } else { 0.0 },
            top3p, top3n, hhi, ent,
        ]);
        feats.extend(topk_feats);
        feats.extend(topk_abs_feats);
        feats.extend(deg_feats);
        if asym {
            let mut col: Vec<f64> = (0..n).filter(|&j| j != i).map(|j| m[j * n + i] as f64).collect();
            col.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let col_mean: f64 = col.iter().sum::<f64>() / nn as f64;
            let rtopk = |k: usize| -> f64 { let kk = k.min(nn); r[nn - kk..].iter().map(|x| x.0 as f64).sum::<f64>() / kk as f64 };
            let ctopk = |k: usize| -> f64 { let kk = k.min(nn); col[nn - kk..].iter().sum::<f64>() / kk as f64 };
            feats.extend([mean, col_mean, mean - col_mean, rtopk(5), ctopk(5), rtopk(50), ctopk(50), rtopk(300), ctopk(300)]);
        }
        for (f, &v) in feats.iter().enumerate() {
            cols[f].push(v as f32);
        }
        // topk/bottom 索引
        for (ki, &k) in [5usize, 20usize].iter().enumerate() {
            let kk = k.min(nn);
            topk_idx[ki][i] = r[nn - kk..].iter().map(|x| x.1).collect();
            bottom_idx[ki][i] = r[..kk].iter().map(|x| x.1).collect();
        }
        mean_col[i] = mean as f32;
    }
    let full_names: Vec<String> = names.iter().map(|nm| format!("{prefix}__{nm}")).collect();
    RowFeats { names: full_names, cols, topk_idx, bottom_idx, mean_col }
}

/// 稀疏图特征（阈值化）
fn graph_features(m: &[f32], n: usize, prefix: &str) -> Vec<(String, Vec<f32>)> {
    // 阈值 p95（直方图）
    let mut abs_all: Vec<f32> = Vec::with_capacity(n * n / 4);
    let mut step = 1usize;
    if n > 2000 { step = 2; }
    for i in (0..n).step_by(step) {
        let row = &m[i * n..(i + 1) * n];
        for j in 0..n {
            if i != j {
                abs_all.push(row[j].abs());
            }
        }
    }
    let th = hist_quantiles(&abs_all, &[0.95])[0];
    drop(abs_all);
    // CSR
    let mut edges: Vec<(u32, u32)> = Vec::new();
    for i in 0..n {
        for j in 0..n {
            if i != j && m[i * n + j].abs() > th {
                edges.push((i as u32, j as u32));
            }
        }
    }
    edges.sort_unstable();
    let nnz = edges.len();
    let mut row_ptr = vec![0u32; n + 1];
    for &(i, _) in edges.iter() {
        row_ptr[i as usize + 1] += 1;
    }
    for i in 0..n {
        row_ptr[i + 1] += row_ptr[i];
    }
    let mut col_idx = vec![0u32; nnz];
    for (k, &(_, j)) in edges.iter().enumerate() {
        col_idx[k] = j;
    }
    drop(edges);
    // 加权度 / 度
    let mut wdeg = vec![0f64; n];
    let mut deg = vec![0f64; n];
    for i in 0..n {
        for k in row_ptr[i] as usize..row_ptr[i + 1] as usize {
            let j = col_idx[k] as usize;
            wdeg[i] += m[i * n + j].abs() as f64;
            deg[i] += 1.0;
        }
    }
    // PageRank（列随机化幂迭代）
    let mut pagerank = vec![1.0 / n as f64; n];
    for _ in 0..40 {
        let mut nxt = vec![(1.0 - 0.85) / n as f64; n];
        for i in 0..n {
            if deg[i] == 0.0 { continue; }
            let w = 0.85 * pagerank[i] / deg[i];
            for k in row_ptr[i] as usize..row_ptr[i + 1] as usize {
                let j = col_idx[k] as usize;
                nxt[j] += w;
            }
        }
        pagerank = nxt;
    }
    // 聚类系数（采样 ≤50 邻居）
    let mut clustering = vec![0f64; n];
    let mut rng: u64 = 42;
    for i in 0..n {
        let d = row_ptr[i + 1] - row_ptr[i];
        if d < 2 { continue; }
        let nb: Vec<usize> = (row_ptr[i] as usize..row_ptr[i + 1] as usize).map(|k| col_idx[k] as usize).collect();
        let sample: Vec<usize> = if nb.len() <= 50 {
            nb.clone()
        } else {
            let mut s = nb.clone();
            // Fisher-Yates 前 50
            for k in 0..50 {
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let j = k + ((rng >> 33) as usize % (s.len() - k));
                s.swap(k, j);
            }
            s.truncate(50);
            s
        };
        let mut tri = 0usize;
        for a in 0..sample.len() {
            let ia = sample[a];
            for b in (a + 1)..sample.len() {
                let ib = sample[b];
                // 查边 ia→ib
                let (mut lo, mut hi) = (row_ptr[ia] as usize, row_ptr[ia + 1] as usize);
                while lo < hi {
                    let mid = (lo + hi) / 2;
                    if col_idx[mid] as usize == ib { tri += 1; break; }
                    if (col_idx[mid] as usize) < ib { lo = mid + 1; } else { hi = mid; }
                }
            }
        }
        let m = sample.len();
        clustering[i] = if m > 1 { 2.0 * tri as f64 / (m * (m - 1)) as f64 } else { 0.0 };
    }
    // 邻接谱嵌入 top-15（稀疏幂迭代, 免稠密矩阵）
    let adj_row_ptr = row_ptr.clone();
    let adj_col = col_idx.clone();
    let mut vecs = vec![0f32; n * 15];
    {
        let mut seed: u64 = 0x9E3779B97F4A7C15;
        for j in 0..15 {
            let mut v: Vec<f32> = (0..n)
                .map(|_| {
                    seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                    ((seed >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0
                })
                .collect();
            let mut w = vec![0f32; n];
            for _ in 0..30 {
                // 稀疏 matvec（并行行块）
                w.par_chunks_mut(128).enumerate().for_each(|(ci, chunk)| {
                    let i0 = ci * 128;
                    for (ii, o) in chunk.iter_mut().enumerate() {
                        let i = i0 + ii;
                        let mut acc = 0.0f32;
                        for k in adj_row_ptr[i] as usize..adj_row_ptr[i + 1] as usize {
                            acc += v[adj_col[k] as usize];
                        }
                        *o = acc;
                    }
                });
                for jj in 0..j {
                    let u = &vecs[jj * n..(jj + 1) * n];
                    let mut d = 0.0f32;
                    for i in 0..n {
                        d += u[i] * w[i];
                    }
                    for i in 0..n {
                        w[i] -= d * u[i];
                    }
                }
                let mut norm = 0.0f32;
                for i in 0..n {
                    norm = f32::mul_add(w[i], w[i], norm);
                }
                norm = norm.sqrt();
                if norm < 1e-9 {
                    v.iter_mut().for_each(|x| *x = 0.0);
                    break;
                }
                for i in 0..n {
                    w[i] /= norm;
                }
                let mut c = 0.0f32;
                for i in 0..n {
                    c += w[i] * v[i];
                }
                v.copy_from_slice(&w);
                if c > 1.0 - 1e-4 {
                    break;
                }
            }
            vecs[j * n..(j + 1) * n].copy_from_slice(&v);
        }
        for j in 0..15 {
            let col = &mut vecs[j * n..(j + 1) * n];
            let mut mi = 0usize;
            for (i, &x) in col.iter().enumerate() {
                if x.abs() > col[mi].abs() {
                    mi = i;
                }
            }
            if col[mi] < 0.0 {
                for x in col.iter_mut() {
                    *x = -*x;
                }
            }
        }
    }
    let mut out: Vec<(String, Vec<f32>)> = Vec::new();
    let mut push = |nm: &str, col: Vec<f32>| out.push((format!("graph_{prefix}__{nm}"), col));
    push("wdeg", wdeg.iter().map(|&v| v as f32).collect());
    push("deg", deg.iter().map(|&v| v as f32).collect());
    push("pagerank", pagerank.iter().map(|&v| v as f32).collect());
    push("clustering", clustering.iter().map(|&v| v as f32).collect());
    for j in 0..15 {
        push(&format!("ev{}", j + 1), vecs[j * n..(j + 1) * n].to_vec());
    }
    out
}

fn extract_factors(
    n: usize,
    reg: &BTreeMap<String, Vec<f32>>,
    combos: &[(String, Vec<f32>)],
    spectral_mats: &[(String, Vec<f32>)],
    stocks: &[StockData],
) -> (Vec<String>, Vec<f32>) {
    let mut names: Vec<String> = Vec::new();
    let mut cols: Vec<Vec<f32>> = Vec::new();
    let mut add = |nm: &str, col: Vec<f32>| {
        names.push(nm.to_string());
        cols.push(col);
    };

    // 矩阵顺序（与 Python 版一致的 21 基础矩阵）
    let mut order: Vec<(String, bool)> = Vec::new();
    for nm in ["x1", "x3", "x8", "x9"] { order.push((nm.to_string(), false)); }
    for dt in X2_DT.iter() { order.push((format!("x2_{dt}s"), true)); }
    for sname in ["30s", "60s", "300s"] {
        order.push((format!("x4_{sname}"), false));
    }
    for sname in ["30s", "60s", "300s"] {
        order.push((format!("x5lag_{sname}"), true));
        order.push((format!("x5str_{sname}"), true));
    }
    for sname in ["30s", "60s", "300s"] {
        order.push((format!("x6_{sname}"), true));
    }
    for sname in ["10s", "30s"] { order.push((format!("x7_{sname}"), false)); }
    for (nm, _) in combos.iter() { order.push((nm.clone(), false)); }

    // ---- L1 行画像 + 邻域索引 ----
    let tt = std::time::Instant::now();
    let mut rowf: Vec<RowFeats> = Vec::new();
    let mut mean_map: BTreeMap<String, Vec<f32>> = BTreeMap::new();
    for (nm, asym) in order.iter() {
        let m = reg.get(nm).unwrap();
        let mut rf = row_features(m, n, *asym, nm);
        mean_map.insert(nm.clone(), rf.mean_col.clone());
        for (fname, col) in rf.names.iter().zip(rf.cols.drain(..)) {
            add(fname, col);
        }
        rowf.push(rf);
    }
    eprintln!("    [f:L1 行画像] {:.1}s", tt.elapsed().as_secs_f32());

    // ---- L2 谱嵌入 ----
    for (nm, vecs) in spectral_mats.iter() {
        for j in 0..SPECTRAL_K {
            add(&format!("spec_{nm}__ev{}", j + 1), vecs[j * n..(j + 1) * n].to_vec());
        }
    }

    // ---- L3 图结构 ----
    let tt = std::time::Instant::now();
    let graph_names: [&str; 14] = [
        "x4_60s", "x7_30s", "x1", "x4_30s", "x9", "x3", "combo1", "combo2",
        "x2_10s", "x5str_60s", "x8", "x7_10s", "x4_300s", "combo3",
    ];
    for nm in graph_names.iter() {
        let m = reg.get(*nm).unwrap();
        for (fname, col) in graph_features(m, n, nm) {
            add(&fname, col);
        }
    }
    eprintln!("    [f:L3 图] {:.1}s", tt.elapsed().as_secs_f32());

    // ---- L4 交互 ----
    let tt = std::time::Instant::now();
    // L4a 派生矩阵
    let mut derived: Vec<(String, Vec<f32>)> = Vec::new();
    for dt in X2_DT.iter() {
        let sum = reg.get(&format!("x2_{dt}s_sum")).unwrap();
        let asy = reg.get(&format!("x2_{dt}s_asy")).unwrap();
        derived.push((format!("x2_{dt}s_sum"), sum.clone()));
        derived.push((format!("x2_{dt}s_asy"), asy.clone()));
    }
    for pair in [(10usize, 30usize), (10, 60), (30, 60)] {
        let (a, b) = pair;
        let sa = reg.get(&format!("x2_{a}s_sum")).unwrap();
        let sb = reg.get(&format!("x2_{b}s_sum")).unwrap();
        let aa = reg.get(&format!("x2_{a}s_asy")).unwrap();
        let ab = reg.get(&format!("x2_{b}s_asy")).unwrap();
        let mut d1 = vec![0f32; n * n];
        let mut d2 = vec![0f32; n * n];
        for k in 0..n * n {
            d1[k] = sa[k] - sb[k];
            d2[k] = aa[k] - ab[k];
        }
        derived.push((format!("x2_{a}s_sum_m_{b}s_sum"), d1));
        derived.push((format!("x2_{a}s_asy_m_{b}s_asy"), d2));
    }
    for (i, sname) in ["30s", "60s", "300s"].iter().enumerate() {
        let lag = reg.get(&format!("x5lag_{sname}")).unwrap();
        let str = reg.get(&format!("x5str_{sname}")).unwrap();
        let mut d = vec![0f32; n * n];
        for k in 0..n * n {
            d[k] = lag[k] * str[k];
        }
        derived.push((format!("x5lagxstr_{sname}"), d));
    }
    {
        let x6 = reg.get("x6_60s").unwrap();
        let mut d = vec![0f32; n * n];
        for i in 0..n {
            for j in 0..n {
                d[i * n + j] = x6[i * n + j] - x6[j * n + i];
            }
        }
        derived.push(("x6_60s_asy".into(), d));
    }
    {
        let a = reg.get("x4_30s").unwrap();
        let b = reg.get("x4_300s").unwrap();
        let mut d = vec![0f32; n * n];
        for k in 0..n * n { d[k] = a[k] - b[k]; }
        derived.push(("x4_30s_m_300s".into(), d));
    }
    {
        let a = reg.get("x7_10s").unwrap();
        let b = reg.get("x7_30s").unwrap();
        let mut d = vec![0f32; n * n];
        for k in 0..n * n { d[k] = a[k] - b[k]; }
        derived.push(("x7_10s_m_30s".into(), d));
    }
    {
        let a = reg.get("x5str_30s").unwrap();
        let b = reg.get("x5str_300s").unwrap();
        let mut d = vec![0f32; n * n];
        for k in 0..n * n { d[k] = a[k] - b[k]; }
        derived.push(("x5str_30s_m_300s".into(), d));
    }
    for (nm, d) in derived.iter() {
        // 每行统计: mean/std/topk5/topk50/max_abs + net（部分选择替代全排序）
        let rows: Vec<(f32, f32, f32, f32, f32, f32)> = (0..n)
            .into_par_iter()
            .map(|i| {
                let mut v: Vec<f32> = (0..n).filter(|&j| j != i).map(|j| d[i * n + j]).collect();
                let nn = v.len();
                let mean: f64 = v.iter().map(|&x| x as f64).sum::<f64>() / nn as f64;
                let std = (v.iter().map(|&x| (x as f64 - mean) * (x as f64 - mean)).sum::<f64>() / nn as f64).sqrt();
                let ma = v.iter().map(|x| x.abs() as f64).fold(0.0f64, f64::max);
                v.select_nth_unstable_by(nn - 5, |a, b| a.partial_cmp(b).unwrap());
                let t5: f64 = v[nn - 5..].iter().map(|&x| x as f64).sum::<f64>() / 5.0;
                v.select_nth_unstable_by(nn - 50, |a, b| a.partial_cmp(b).unwrap());
                let t50: f64 = v[nn - 50..].iter().map(|&x| x as f64).sum::<f64>() / 50.0;
                let col_mean: f64 = (0..n).filter(|&j| j != i).map(|j| d[j * n + i] as f64).sum::<f64>() / nn as f64;
                (mean as f32, std as f32, t5 as f32, t50 as f32, ma as f32, (mean - col_mean) as f32)
            })
            .collect();
        let mut c_mean = vec![0f32; n];
        let mut c_std = vec![0f32; n];
        let mut c_t5 = vec![0f32; n];
        let mut c_t50 = vec![0f32; n];
        let mut c_ma = vec![0f32; n];
        let mut c_net = vec![0f32; n];
        for (i, r) in rows.iter().enumerate() {
            c_mean[i] = r.0;
            c_std[i] = r.1;
            c_t5[i] = r.2;
            c_t50[i] = r.3;
            c_ma[i] = r.4;
            c_net[i] = r.5;
        }
        add(&format!("der_{nm}__mean"), c_mean);
        add(&format!("der_{nm}__std"), c_std);
        add(&format!("der_{nm}__topk5_mean"), c_t5);
        add(&format!("der_{nm}__topk50_mean"), c_t50);
        add(&format!("der_{nm}__max_abs"), c_ma);
        add(&format!("der_{nm}__net"), c_net);
    }
    eprintln!("    [f:L4a 派生] {:.1}s", tt.elapsed().as_secs_f32());

    // L4b 元素积
    let prod_pairs: [(&str, &str); 70] = [
        ("x4_60s", "x7_30s"), ("x4_30s", "x7_30s"), ("x4_300s", "x7_30s"), ("x4_60s", "x7_10s"),
        ("x1", "x4_60s"), ("x1", "x4_30s"), ("x3", "x4_60s"), ("x8", "x4_60s"), ("x9", "x4_60s"),
        ("x1", "x3"), ("x8", "x9"), ("x3", "x9"), ("x1", "x8"), ("x1", "x9"),
        ("x2_10s_sum", "x4_60s"), ("x2_30s_sum", "x4_60s"), ("x2_60s_sum", "x4_60s"),
        ("x2_10s_sum", "x7_30s"), ("x2_30s_sum", "x7_30s"), ("x2_60s_sum", "x7_30s"),
        ("x2_10s_sum", "x2_60s_sum"), ("x2_10s_asy", "x4_60s"), ("x2_10s_asy", "x7_30s"),
        ("x2_30s_asy", "x4_60s"), ("x2_60s_asy", "x4_60s"),
        ("x5str_30s", "x4_30s"), ("x5str_60s", "x4_60s"), ("x5str_300s", "x4_300s"),
        ("x5str_30s", "x6_30s"), ("x5str_60s", "x6_60s"), ("x5str_300s", "x6_300s"),
        ("x5lag_30s", "x4_30s"), ("x5lag_60s", "x4_60s"), ("x5lag_300s", "x4_300s"),
        ("x5lag_60s", "x6_60s"), ("x5lag_60s", "x5str_60s"),
        ("x6_30s", "x6_300s"), ("x7_10s", "x7_30s"), ("x6_60s", "x5str_60s"),
        ("x4_30s", "x4_300s"), ("x4_30s", "x4_60s"), ("x4_60s", "x4_300s"),
        ("x4_60s", "x6_60s"), ("x1", "x7_30s"), ("x3", "x7_30s"), ("x8", "x7_30s"),
        ("x2_10s", "x2_60s"), ("x5str_60s", "x1"), ("x6_60s", "x1"), ("x4_300s", "x7_30s"),
        ("combo1", "x4_60s"), ("combo1", "x7_30s"), ("combo2", "x6_60s"), ("combo3", "x4_60s"),
        ("combo4", "x5str_300s"), ("combo5", "x4_30s"), ("combo6", "x5str_30s"),
        ("combo1", "combo3"), ("combo2", "combo4"), ("combo5", "combo6"),
        ("x2_30s_sum", "x5str_60s"), ("x2_10s_asy", "x2_10s_asy"), ("x2_10s_sum", "x5str_60s"),
        ("x4_30s", "x5lag_30s"), ("x8", "x9"), ("x1", "x2_10s_sum"), ("x9", "x7_30s"),
        ("x2_10s", "x4_30s"), ("x2_60s", "x4_300s"), ("x5str_30s", "x5str_300s"),
    ];
    for (a, b) in prod_pairs.iter() {
        let ma = reg.get(*a).unwrap();
        let mb = reg.get(*b).unwrap();
        let mut p = vec![0f32; n * n];
        for k in 0..n * n { p[k] = ma[k] * mb[k]; }
        let rows: Vec<(f32, f32, f32, f32, f32)> = (0..n)
            .into_par_iter()
            .map(|i| {
                let mut v: Vec<f32> = (0..n).filter(|&j| j != i).map(|j| p[i * n + j]).collect();
                let nn = v.len();
                let mean: f64 = v.iter().map(|&x| x as f64).sum::<f64>() / nn as f64;
                let std = (v.iter().map(|&x| (x as f64 - mean) * (x as f64 - mean)).sum::<f64>() / nn as f64).sqrt();
                let mx = v.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                v.select_nth_unstable_by(nn - 5, |a, b| a.partial_cmp(b).unwrap());
                let t5: f64 = v[nn - 5..].iter().map(|&x| x as f64).sum::<f64>() / 5.0;
                v.select_nth_unstable_by(nn - 50, |a, b| a.partial_cmp(b).unwrap());
                let t50: f64 = v[nn - 50..].iter().map(|&x| x as f64).sum::<f64>() / 50.0;
                (mean as f32, std as f32, t5 as f32, t50 as f32, mx)
            })
            .collect();
        let mut c_mean = vec![0f32; n];
        let mut c_t5 = vec![0f32; n];
        let mut c_t50 = vec![0f32; n];
        let mut c_max = vec![0f32; n];
        let mut c_std = vec![0f32; n];
        for (i, r) in rows.iter().enumerate() {
            c_mean[i] = r.0;
            c_std[i] = r.1;
            c_t5[i] = r.2;
            c_t50[i] = r.3;
            c_max[i] = r.4;
        }
        add(&format!("prod_{a}_{b}__mean"), c_mean);
        add(&format!("prod_{a}_{b}__std"), c_std);
        add(&format!("prod_{a}_{b}__topk5_mean"), c_t5);
        add(&format!("prod_{a}_{b}__topk50_mean"), c_t50);
        add(&format!("prod_{a}_{b}__max"), c_max);
    }
    eprintln!("    [f:L4b 元素积] {:.1}s", tt.elapsed().as_secs_f32());

    // L4c 二跳 O(N²): diag = Σ_j M[i,j]² ; row_mean = Σ_k M[i,k]·colsum[k]
    let hop_pairs: [(&str, &str); 40] = [
        ("x4_60s", "x4_60s"), ("x7_30s", "x7_30s"), ("x1", "x1"), ("x3", "x3"),
        ("x8", "x8"), ("x9", "x9"), ("x4_30s", "x4_30s"), ("x4_300s", "x4_300s"),
        ("x7_10s", "x7_10s"), ("x2_10s_sum", "x2_10s_sum"), ("x2_60s_sum", "x2_60s_sum"),
        ("x5str_60s", "x5str_60s"), ("combo1", "combo1"), ("combo2", "combo2"), ("combo3", "combo3"),
        ("x4_60s", "x7_30s"), ("x4_60s", "x1"), ("x7_30s", "x1"), ("x3", "x9"), ("x1", "x9"),
        ("x8", "x4_60s"), ("x3", "x4_60s"), ("x2_10s_sum", "x4_60s"), ("x2_10s_sum", "x7_30s"),
        ("x6_60s", "x5str_60s"), ("x5str_60s", "x4_60s"), ("x4_300s", "x5str_300s"),
        ("x2_10s", "x2_10s"), ("x2_60s", "x2_60s"), ("x5lag_60s", "x5lag_60s"),
        ("x6_60s", "x6_60s"), ("x6_30s", "x6_30s"), ("x6_300s", "x6_300s"),
        ("combo4", "combo4"), ("combo5", "combo5"), ("combo6", "combo6"),
        ("x4_30s", "x7_10s"), ("x4_300s", "x7_30s"), ("x2_30s_sum", "x4_60s"), ("x1", "x7_30s"),
    ];
    let hop_res: Vec<(String, Vec<f32>, Vec<f32>)> = hop_pairs
        .par_iter()
        .map(|(a, b)| {
            let ma = reg.get(*a).unwrap();
            let mb = reg.get(*b).unwrap();
            let mut colsum_b = vec![0f64; n];
            for i in 0..n {
                let row = &mb[i * n..(i + 1) * n];
                let mut s = 0.0f64;
                for (j, &v) in row.iter().enumerate() {
                    if i != j { s += v as f64; }
                    colsum_b[j] += v as f64;   // 行主序累加列和（避免跨步访问）
                }
                let _ = s;
            }
            // 修正: 上面把对角也加了, 减掉
            for j in 0..n {
                colsum_b[j] -= mb[j * n + j] as f64;
            }
            let mut c_diag = vec![0f32; n];
            let mut c_rm = vec![0f32; n];
            for i in 0..n {
                let mut d2 = 0.0f64;
                let mut rm = 0.0f64;
                for j in 0..n {
                    let v = ma[i * n + j] as f64;
                    if i != j {
                        d2 += v * v;
                        rm += v * colsum_b[j];
                    }
                }
                c_diag[i] = d2 as f32;
                c_rm[i] = rm as f32;
            }
            (format!("{a}_{b}"), c_diag, c_rm)
        })
        .collect();
    for (nm, c_diag, c_rm) in hop_res.iter() {
        add(&format!("hop2_{nm}__diag"), c_diag.clone());
        add(&format!("hop2_{nm}__row_mean"), c_rm.clone());
    }

    eprintln!("    [f:L4c 二跳] {:.1}s", tt.elapsed().as_secs_f32());

    // L4d 跨矩阵行余弦
    let cos_pairs: [(&str, &str); 40] = [
        ("x4_60s", "x7_30s"), ("x1", "x3"), ("x8", "x9"), ("x1", "x4_60s"),
        ("x4_30s", "x4_60s"), ("x4_60s", "x4_300s"), ("x3", "x9"), ("x1", "x8"),
        ("x2_10s_sum", "x7_30s"), ("x2_10s_sum", "x4_60s"), ("x4_60s", "x5str_60s"),
        ("x7_10s", "x7_30s"), ("x4_300s", "x5str_300s"), ("x6_30s", "x6_300s"),
        ("x6_60s", "x5str_60s"), ("x3", "x4_60s"), ("x1", "x9"), ("x3", "x7_30s"),
        ("x8", "x4_60s"), ("x5str_30s", "x5str_300s"), ("x2_10s", "x2_60s"),
        ("x4_60s", "x6_60s"), ("x1", "x2_10s_sum"), ("x9", "x7_30s"),
        ("combo1", "x4_60s"), ("combo2", "x6_60s"), ("combo3", "x3"), ("combo4", "x4_300s"),
        ("combo1", "combo3"), ("x5str_60s", "x7_30s"), ("x4_30s", "x5str_30s"),
        ("x5lag_60s", "x6_60s"), ("x5lag_60s", "x4_60s"), ("x2_30s", "x4_60s"),
        ("x8", "x9"), ("x3", "x8"), ("x1", "x4_30s"), ("x6_60s", "x7_30s"),
        ("combo2", "combo4"), ("combo5", "combo6"),
    ];
    let cos_res: Vec<(String, Vec<f32>)> = cos_pairs
        .par_iter()
        .map(|(a, b)| {
            let ma = reg.get(*a).unwrap();
            let mb = reg.get(*b).unwrap();
            let c: Vec<f32> = (0..n)
                .into_par_iter()
                .map(|i| {
                    let mut dot = 0.0f64;
                    let mut na = 0.0f64;
                    let mut nb = 0.0f64;
                    for j in 0..n {
                        if i == j { continue; }
                        let x = ma[i * n + j] as f64;
                        let y = mb[i * n + j] as f64;
                        dot += x * y;
                        na += x * x;
                        nb += y * y;
                    }
                    let denom = (na * nb).sqrt();
                    if denom > 1e-15 { (dot / denom) as f32 } else { 0.0 }
                })
                .collect();
            (format!("{a}_{b}"), c)
        })
        .collect();
    for (nm, c) in cos_res.iter() {
        add(&format!("cos_{nm}"), c.clone());
    }
    eprintln!("    [f:L4d 余弦] {:.1}s", tt.elapsed().as_secs_f32());

    // ---- L5 邻域跨矩阵（主力扩张源）----
    // 对每对有序 (M,N), k∈{5,20}: mean_N 在 M 的 top/bottom k 邻居上
    let ks: [usize; 2] = [5, 20];
    let mut nb_tasks: Vec<(usize, usize)> = Vec::new();
    for mi in 0..order.len() {
        for ni in 0..order.len() {
            if mi != ni {
                nb_tasks.push((mi, ni));
            }
        }
    }
    let nb_res: Vec<(String, Vec<f32>)> = nb_tasks
        .par_iter()
        .flat_map_iter(|(mi, ni)| {
            let ((nm_m, _), rf_m) = (&order[*mi], &rowf[*mi]);
            let (nm_n, _) = &order[*ni];
            let m_n = reg.get(nm_n).unwrap();
            let mean_n = mean_map.get(nm_n).unwrap();
            let mut out = Vec::new();
            for (ki, &k) in ks.iter().enumerate() {
                let mut ctop = vec![0f32; n];
                let mut cbot = vec![0f32; n];
                let mut cr = vec![0f32; n];
                for i in 0..n {
                    let idxs = &rf_m.topk_idx[ki][i];
                    let mut st = 0.0f64;
                    for &j in idxs.iter() {
                        st += m_n[i * n + j as usize] as f64;
                    }
                    let tv = (st / idxs.len() as f64) as f32;
                    ctop[i] = tv;
                    let bidxs = &rf_m.bottom_idx[ki][i];
                    let mut sb = 0.0f64;
                    for &j in bidxs.iter() {
                        sb += m_n[i * n + j as usize] as f64;
                    }
                    cbot[i] = (sb / bidxs.len() as f64) as f32;
                    if ki == 1 {
                        cr[i] = tv / mean_n[i].abs().max(1e-9);
                    }
                }
                out.push((format!("nb_{nm_m}_{nm_n}_top{k}"), ctop));
                out.push((format!("nb_{nm_m}_{nm_n}_bot{k}"), cbot));
                if ki == 1 {
                    out.push((format!("nb_{nm_m}_{nm_n}_ratio20"), cr));
                }
            }
            out.into_iter()
        })
        .collect();
    for (nm, c) in nb_res.iter() {
        add(nm, c.clone());
    }

    eprintln!("    [f:L5 邻域] {:.1}s", tt.elapsed().as_secs_f32());

    // ---- 自特征 ----
    let tt = std::time::Instant::now();
    let mut self_names: Vec<&str> = Vec::new();
    let mut self_cols: Vec<Vec<f32>> = Vec::new();
    // cnt1 自相关（从 b1 稀疏重建）
    for (li, &lag) in [60usize, 300, 900].iter().enumerate() {
        let mut c = vec![0f32; n];
        for (i, st) in stocks.iter().enumerate() {
            let t = BUCKET_1S;
            let mut x = vec![0f32; t];
            for (&idx, &cnt) in st.b1_idx.iter().zip(st.b1_cnt.iter()) {
                x[idx as usize] = cnt as f32;
            }
            let (a, b) = (&x[..t - lag], &x[lag..]);
            let ma: f64 = a.iter().map(|&v| v as f64).sum::<f64>() / a.len() as f64;
            let mb: f64 = b.iter().map(|&v| v as f64).sum::<f64>() / b.len() as f64;
            let va: f64 = a.iter().map(|&v| (v as f64 - ma) * (v as f64 - ma)).sum::<f64>() / a.len() as f64;
            let vb: f64 = b.iter().map(|&v| (v as f64 - mb) * (v as f64 - mb)).sum::<f64>() / b.len() as f64;
            let den = (va * vb).sqrt();
            let num: f64 = a.iter().zip(b.iter()).map(|(&u, &v)| (u as f64 - ma) * (v as f64 - mb)).sum::<f64>() / a.len() as f64;
            c[i] = if den > 1e-15 { (num / den) as f32 } else { 0.0 };
        }
        self_names.push(["cnt1_ac60", "cnt1_ac300", "cnt1_ac900"][li]);
        self_cols.push(c);
    }
    // s60/s30 自相关
    let autos: [(&str, usize, usize); 5] = [
        ("s60", 1, 237), ("s60", 5, 237), ("s60", 30, 237), ("s30", 1, 474), ("s300", 1, 48),
    ];
    for &(which, lag, t) in autos.iter() {
        let mut c = vec![0f32; n];
        for (i, st) in stocks.iter().enumerate() {
            let x: &[f32] = match which {
                "s60" => &st.s60,
                "s30" => &st.s30,
                _ => &st.s300,
            };
            let (a, b) = (&x[..t - lag], &x[lag..]);
            let ma: f64 = a.iter().map(|&v| v as f64).sum::<f64>() / a.len() as f64;
            let mb: f64 = b.iter().map(|&v| v as f64).sum::<f64>() / b.len() as f64;
            let va: f64 = a.iter().map(|&v| (v as f64 - ma) * (v as f64 - ma)).sum::<f64>() / a.len() as f64;
            let vb: f64 = b.iter().map(|&v| (v as f64 - mb) * (v as f64 - mb)).sum::<f64>() / b.len() as f64;
            let den = (va * vb).sqrt();
            let num: f64 = a.iter().zip(b.iter()).map(|(&u, &v)| (u as f64 - ma) * (v as f64 - mb)).sum::<f64>() / a.len() as f64;
            c[i] = if den > 1e-15 { (num / den) as f32 } else { 0.0 };
        }
        self_names.push(match which {
            "s60" => match lag { 1 => "s60_ac1", 5 => "s60_ac5", _ => "s60_ac30" },
            "s30" => "s30_ac1",
            _ => "s300_ac1",
        });
        self_cols.push(c);
    }
    // ret60 std/skew/kurt; 活跃度剖面; 基础统计
    {
        let mut c_std = vec![0f32; n];
        let mut c_skew = vec![0f32; n];
        let mut c_kurt = vec![0f32; n];
        let mut c_active = vec![0f32; n];
        let mut c_ocr = vec![0f32; n];
        for (i, st) in stocks.iter().enumerate() {
            let p = &st.p60;
            let t = SCALE_60;
            let mut ret = vec![0f64; t];
            let mut prev = p[0] as f64;
            for k in 1..t {
                ret[k] = if prev > 0.0 { p[k] as f64 / prev - 1.0 } else { 0.0 };
                prev = p[k] as f64;
            }
            let mean = ret.iter().sum::<f64>() / t as f64;
            let var = ret.iter().map(|&v| (v - mean) * (v - mean)).sum::<f64>() / t as f64;
            let sd = var.sqrt();
            let m3 = ret.iter().map(|&v| (v - mean).powi(3)).sum::<f64>() / t as f64;
            let m4 = ret.iter().map(|&v| (v - mean).powi(4)).sum::<f64>() / t as f64;
            c_std[i] = sd as f32;
            c_skew[i] = if sd > 1e-12 { (m3 / sd.powi(3)) as f32 } else { 0.0 };
            c_kurt[i] = if sd > 1e-12 { (m4 / sd.powi(4) - 3.0) as f32 } else { 0.0 };
            c_active[i] = st.c60.iter().filter(|&&v| v > 0.0).count() as f32 / SCALE_60 as f32;
            let f30: f32 = st.c60[..30].iter().sum();
            let l30: f32 = st.c60[SCALE_60 - 30..].iter().sum();
            c_ocr[i] = f30 / (l30 + 1.0);
        }
        self_names.extend(["ret60_std", "ret60_skew", "ret60_kurt", "active_min_ratio", "open_close_ratio"]);
        self_cols.extend([c_std, c_skew, c_kurt, c_active, c_ocr]);
    }
    {
        let mut cols: Vec<Vec<f32>> = vec![Vec::with_capacity(n); 8];
        for (i, st) in stocks.iter().enumerate() {
            cols[0].push((st.n as f32).ln());
            cols[1].push((st.turnover as f32).ln());
            cols[2].push((st.median_vol as f32).ln());
            cols[3].push((st.th99 as f32).ln());
            cols[4].push(st.nbig as f32 / st.n as f32);
            cols[5].push((st.turnover / st.n as f64) as f32);
            cols[6].push(st.nbig as f32);
            cols[7].push(st.s60.iter().map(|&v| (v > 0.0) as i32 as f32).sum::<f32>() / SCALE_60 as f32);
        }
        self_names.extend(["log_n", "log_turnover", "log_median_vol", "log_th99", "big_fill_ratio", "avg_trade_size", "nbig", "s60_pos_min_ratio"]);
        self_cols.extend(cols);
    }
    for (nm, col) in self_names.iter().zip(self_cols) {
        add(&format!("self_{nm}"), col);
    }
    eprintln!("    [f:自特征] {:.1}s", tt.elapsed().as_secs_f32());

    // 组装 N×F 输出
    let fcnt = cols.len();
    let mut vals = vec![0f32; n * fcnt];
    for (f, col) in cols.iter().enumerate() {
        for i in 0..n {
            vals[i * fcnt + f] = col[i];
        }
    }
    (names, vals)
}
