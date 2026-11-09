//! 方向 1「今天对昨天」单日全市场探索（沙箱）。
//! 每只股票：今天和昨天的同一时段各变成字母序列 → 动态规划比对 → 路径特征 → 横截面合成。
//! 构建: cargo build --release -j 8
//! 运行: ./target/release/align_day_vs_day <今天> <昨天> > results/xxx.json
mod align;
mod fast_csv_reader;
mod letters;

use align::{align, Aln, Work, OP_D, OP_X, OP_Y};
use fast_csv_reader::{read_market_fast, read_trade_with_cancel, MarketRecord, TradeRecord};
use letters::*;
use rayon::prelude::*;
use serde::Serialize;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

const MAX_THREADS: usize = 50;
const SEC: i64 = 1_000_000;
const OPEN_US: i64 = (9 * 3600 + 30 * 60) * SEC;
const DAY_LEN: i64 = 14_220; // 09:30-11:30 + 13:00-14:57，下午已前移 90 分钟

/// (名字, 起始秒, 细格秒, 粗格秒)
const WINDOWS: [(&str, i64, i64, i64); 3] = [
    ("all", 0, 30, 60),
    ("pm", 7_200, 15, 30),
    ("tail", DAY_LEN - 1_800, 5, 10),
];
const ALPHA_TIME: [usize; 10] = [VOLSIG, MAG5, MAG5Y, SIZEL, RHYTHM, OID, PRICE, BOOKFLOW, BOOKSHAPE, CANCEL];
const ALPHA_VOL: [usize; 6] = [VOLSIG, MAG5, SIZEL, RHYTHM, OID, PRICE];
const ALPHA_COARSE: [usize; 4] = [VOLSIG, MAG5, OID, BOOKFLOW];
const N_STREAMS: usize = 20;

// 计分参数（match=4 写在计分表里）
const OPEN_LOOSE: i32 = 3;
const EXT_LOOSE: i32 = 1;
const OPEN_TIGHT: i32 = 8;
const EXT_TIGHT: i32 = 2;
const MAX_S: f32 = 4.0;

const GL_NAMES: [&str; 20] = ["score", "excess", "identity", "partial", "gap_open", "gap_bar",
    "lag_mean", "lag_abs", "lag_std", "lag_trend", "lag_h1_h2", "insert", "delete", "ins_del_asym",
    "insert_vw", "delete_vw", "lag_vw", "longest_run", "insert_center", "dp_gain"];
const GT_NAMES: [&str; 6] = ["score", "excess", "identity", "gap_open", "lag_mean", "lag_abs"];
const LC_NAMES: [&str; 6] = ["score", "span", "start", "offset", "identity", "vol_cov"];
const UG_NAMES: [&str; 2] = ["score", "identity"];
const N_FEAT: usize = 20 + 6 + 6 + 2;
const N_NAIVE: usize = 3; // 每个时段：分时成交量相关、秩相关、成交量重心时移（朴素替代物）
/// 横截面合成时使用的关键特征（在每条序列特征里的下标）
const KEY: [(usize, &str); 10] = [(0, "score"), (1, "excess"), (2, "identity"), (4, "gap_open"),
    (6, "lag_mean"), (7, "lag_abs"), (13, "ins_del_asym"), (19, "dp_gain"), (32, "ug_score"), (33, "ug_identity")];

#[derive(Serialize)]
struct Out { names: Vec<String>, codes: Vec<String>, vals: Vec<Vec<f32>> }

fn stock_codes_for_day(dir: &std::path::Path, symbol_map: &std::path::Path, date: i64, subdir: &str)
    -> std::io::Result<Vec<String>> {
    use std::io::{Error, ErrorKind};
    let invalid = |e| Error::new(ErrorKind::InvalidData, e);
    let mut reader = csv::Reader::from_path(symbol_map).map_err(invalid)?;
    let column = reader.headers().map_err(invalid)?.iter().position(|h| h == "symbol")
        .ok_or_else(|| Error::new(ErrorKind::InvalidData, "股票表缺少 symbol 列"))?;
    let mut stocks = std::collections::BTreeSet::new();
    for record in reader.records() {
        let record = record.map_err(invalid)?;
        let code = record.get(column).unwrap_or("").trim();
        if code.is_empty() || code.len() > 6 || !code.bytes().all(|b| b.is_ascii_digit()) {
            return Err(Error::new(ErrorKind::InvalidData, "股票表含无效代码"));
        }
        stocks.insert(format!("{code:0>6}"));
    }
    if stocks.is_empty() || !dir.is_dir() {
        return Err(Error::new(ErrorKind::InvalidData, "股票白名单为空或原始数据目录不存在"));
    }
    Ok(stocks.into_iter().filter(|code| dir.join(format!("{code}_{date}_{subdir}.csv")).is_file()).collect())
}

fn list_codes(date: i64, subdir: &str) -> std::io::Result<Vec<String>> {
    let root = std::env::var("RUST_PYFUNC_LEVEL2_ROOT").unwrap_or("/ssd_data/stock".into());
    let basic = std::env::var("RUST_PYFUNC_BASIC_INFO_DIR").unwrap_or("/ssd_data/data/basic_info".into());
    stock_codes_for_day(&std::path::Path::new(&root).join(date.to_string()).join(subdir),
        &std::path::Path::new(&basic).join("symbol_map.csv"), date, subdir)
}

/// 一天一只股票的全部格子：3 个时段 × 细格时间钟 / 粗格时间钟 / 成交量钟
struct DayBars { sets: Vec<[Bars; 3]> }

fn split_sorted(mut all: Vec<TradeRecord>) -> (Vec<TradeRecord>, Vec<TradeRecord>) {
    if !all.windows(2).all(|w| w[0].time_us <= w[1].time_us) { all.sort_by_key(|t| t.time_us); }
    let (mut tr, mut ca) = (Vec::with_capacity(all.len()), Vec::with_capacity(all.len() / 4));
    for t in all { if t.flag == 32 { ca.push(t); } else { tr.push(t); } }
    (tr, ca)
}

fn slice_by_time<'a, T>(v: &'a [T], lo: i64, hi: i64, tm: impl Fn(&T) -> i64) -> &'a [T] {
    let a = v.partition_point(|x| tm(x) < lo);
    let b = v.partition_point(|x| tm(x) <= hi);
    &v[a..b.max(a)]
}

fn build_day(trades: &[TradeRecord], cancels: &[TradeRecord], mkt: &[MarketRecord]) -> Option<DayBars> {
    let t0 = trades.first()?.time_us;
    let day_us = t0 - t0.rem_euclid(86_400 * SEC);
    let mut sets = Vec::with_capacity(3);
    for &(_, s, fine, coarse) in WINDOWS.iter() {
        let start = OPEN_US + s * SEC;
        let (lo, hi) = (day_us + start, day_us + OPEN_US + DAY_LEN * SEC);
        let tr = slice_by_time(trades, lo, hi, |t| t.time_us);
        let ca = slice_by_time(cancels, lo, hi, |t| t.time_us);
        let mk = slice_by_time(mkt, lo, hi, |t| t.time_us);
        let len = DAY_LEN - s;
        let (lf, lc) = ((len / fine) as usize, (len / coarse) as usize);
        sets.push([
            time_bars(tr, ca, mk, start, fine * SEC, lf, day_us),
            time_bars(tr, ca, mk, start, coarse * SEC, lc, day_us),
            volume_bars(tr, lf),
        ]);
    }
    Some(DayBars { sets })
}

struct Tables { t: Vec<Vec<i32>> }

/// 路径特征层：把比对路径翻译成行为差异。i 是今天的位置，j 是昨天的位置。
fn global_features(a: &[u8], b: &[u8], va: &[f64], vb: &[f64], ops: &[u8], r: &Aln, tbl: &[i32], ug: f32, e: f32, out: &mut [f32]) {
    let l = a.len() as f32;
    let score = r.score as f32;
    out[0] = score / (MAX_S * l);
    out[1] = (score - e) / (MAX_S * l);
    out[19] = (score - ug) / (MAX_S * l);
    let (mut i, mut j) = (r.i0, r.j0);
    let (mut same, mut partial, mut pairs) = (0f32, 0f32, 0f32);
    let (mut opens, mut gap_pos, mut last_op) = (0f32, 0f32, 3u8);
    let (mut lag_s, mut lag_a, mut lag_s2, mut lag_n) = (0f64, 0f64, 0f64, 0f64);
    let (mut lv_sum, mut lv_w) = (0f64, 0f64);
    let (mut sx, mut sxx, mut sxy) = (0f64, 0f64, 0f64);
    let (mut h1, mut h1n, mut h2, mut h2n) = (0f64, 0f64, 0f64, 0f64);
    let (mut ins_n, mut del_n, mut ins_pos) = (0f32, 0f32, 0f64);
    let (mut ins_v, mut del_v) = (0f64, 0f64);
    let (mut run, mut best_run) = (0f32, 0f32);
    let half = a.len() / 2;
    for &op in ops {
        match op {
            OP_D => {
                let (x, y) = (a[i], b[j]);
                if x != 0 || y != 0 {
                    pairs += 1.0;
                    if x == y { same += 1.0; } else if tbl[x as usize * align::SIZE + y as usize] > 0 { partial += 1.0; }
                }
                if x != 0 && x == y { run += 1.0; if run > best_run { best_run = run; } }
                else if x != 0 || y != 0 { run = 0.0; }
                if x != 0 && y != 0 {
                    let lag = i as f64 - j as f64;
                    lag_s += lag; lag_a += lag.abs(); lag_s2 += lag * lag; lag_n += 1.0;
                    lv_sum += va[i] * lag; lv_w += va[i];
                    sx += i as f64; sxx += (i * i) as f64; sxy += i as f64 * lag;
                    if i < half { h1 += lag; h1n += 1.0; } else { h2 += lag; h2n += 1.0; }
                }
                i += 1; j += 1;
            }
            OP_X => {
                if last_op != OP_X { opens += 1.0; }
                gap_pos += 1.0;
                if a[i] != 0 { ins_n += 1.0; ins_pos += i as f64; ins_v += va[i]; }
                i += 1;
            }
            _ => {
                debug_assert_eq!(op, OP_Y);
                if last_op != OP_Y { opens += 1.0; }
                gap_pos += 1.0;
                if b[j] != 0 { del_n += 1.0; del_v += vb[j]; }
                j += 1;
            }
        }
        last_op = op;
    }
    let lf = l as f64;
    if pairs > 0.0 { out[2] = same / pairs; out[3] = partial / pairs; }
    out[4] = opens / l;
    out[5] = gap_pos / (2.0 * l);
    if lag_n >= 5.0 {
        out[6] = (lag_s / lag_n / lf) as f32;
        out[7] = (lag_a / lag_n / lf) as f32;
        let var = (lag_s2 / lag_n - (lag_s / lag_n).powi(2)).max(0.0);
        out[8] = (var.sqrt() / lf) as f32;
        let cov = sxy / lag_n - (sx / lag_n) * (lag_s / lag_n);
        let vx = sxx / lag_n - (sx / lag_n).powi(2);
        if var > 1e-12 && vx > 1e-12 { out[9] = (cov / (var * vx).sqrt()) as f32; } else { out[9] = 0.0; }
        if h1n >= 2.0 && h2n >= 2.0 { out[10] = ((h1 / h1n - h2 / h2n) / lf) as f32; }
        if lv_w > 0.0 { out[16] = (lv_sum / lv_w / lf) as f32; }
    }
    let (na, nb) = (a.iter().filter(|&&x| x != 0).count() as f32, b.iter().filter(|&&x| x != 0).count() as f32);
    if na > 0.0 { out[11] = ins_n / na; }
    if nb > 0.0 { out[12] = del_n / nb; }
    out[13] = if ins_n + del_n > 0.0 { (ins_n - del_n) / (ins_n + del_n) } else { 0.0 };
    let (sa, sb): (f64, f64) = (va.iter().sum(), vb.iter().sum());
    if sa > 0.0 { out[14] = (ins_v / sa) as f32; }
    if sb > 0.0 { out[15] = (del_v / sb) as f32; }
    out[17] = best_run / l;
    if ins_n > 0.0 { out[18] = (ins_pos / ins_n as f64 / lf) as f32; }
}

fn local_features(a: &[u8], b: &[u8], va: &[f64], ops: &[u8], r: &Aln, out: &mut [f32]) {
    let l = a.len() as f32;
    out[0] = r.score as f32 / (MAX_S * l);
    out[1] = (r.i1 - r.i0) as f32 / l;
    out[2] = r.i0 as f32 / l;
    out[3] = (r.i0 as f32 - r.j0 as f32) / l;
    let (mut i, mut j, mut same, mut pairs) = (r.i0, r.j0, 0f32, 0f32);
    for &op in ops {
        match op {
            OP_D => { if a[i] != 0 || b[j] != 0 { pairs += 1.0; if a[i] == b[j] { same += 1.0; } } i += 1; j += 1; }
            OP_X => i += 1,
            _ => j += 1,
        }
    }
    if pairs > 0.0 { out[4] = same / pairs; }
    let s: f64 = va.iter().sum();
    if s > 0.0 { out[5] = (va[r.i0..r.i1].iter().sum::<f64>() / s) as f32; }
}

fn pearson(a: &[f64], b: &[f64]) -> f32 {
    let n = a.len() as f64;
    let (ma, mb) = (a.iter().sum::<f64>() / n, b.iter().sum::<f64>() / n);
    let (mut c, mut x, mut y) = (0.0, 0.0, 0.0);
    for (u, v) in a.iter().zip(b) { c += (u - ma) * (v - mb); x += (u - ma).powi(2); y += (v - mb).powi(2); }
    if x > 0.0 && y > 0.0 { (c / (x * y).sqrt()) as f32 } else { f32::NAN }
}
fn ranks(v: &[f64]) -> Vec<f64> {
    let mut idx: Vec<usize> = (0..v.len()).collect();
    idx.sort_by(|&x, &y| v[x].total_cmp(&v[y]).then(x.cmp(&y)));
    let mut r = vec![0.0; v.len()];
    let mut i = 0;
    while i < idx.len() {
        let mut j = i;
        while j + 1 < idx.len() && v[idx[j + 1]] == v[idx[i]] { j += 1; }
        let avg = (i + j) as f64 * 0.5;
        for k in i..=j { r[idx[k]] = avg; }
        i = j + 1;
    }
    r
}

/// 每只股票的完整计算。返回 N_RAW 列原始值和当日成交笔数（用于流动性分组）。
fn per_stock(code: &str, date: i64, prev: i64, tables: &Tables, n_raw: usize,
             read_ns: &AtomicU64, bar_ns: &AtomicU64, dp_ns: &AtomicU64) -> (Vec<f32>, f64) {
    let mut out = vec![f32::NAN; n_raw];
    let t0 = Instant::now();
    let (Ok(td), Ok(yd)) = (read_trade_with_cancel(code, date), read_trade_with_cancel(code, prev)) else {
        read_ns.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
        return (out, f64::NAN);
    };
    let mt = read_market_fast(code, date).unwrap_or_default();
    let my = read_market_fast(code, prev).unwrap_or_default();
    read_ns.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
    let t1 = Instant::now();
    let (tt, tc) = split_sorted(td);
    let (yt, yc) = split_sorted(yd);
    let n_trades = tt.len() as f64;
    let (Some(today), Some(yest)) = (build_day(&tt, &tc, &mt), build_day(&yt, &yc, &my)) else {
        bar_ns.fetch_add(t1.elapsed().as_nanos() as u64, Ordering::Relaxed);
        return (out, n_trades);
    };
    drop((tt, tc, yt, yc, mt, my));
    bar_ns.fetch_add(t1.elapsed().as_nanos() as u64, Ordering::Relaxed);

    let t2 = Instant::now();
    let mut w = Work::default();
    let (mut a, mut b, mut buf) = (Vec::new(), Vec::new(), Vec::new());
    let mut pos = 0usize;
    for (wi, _) in WINDOWS.iter().enumerate() {
        let (ts, ys) = (&today.sets[wi], &yest.sets[wi]);
        // 三组：细格时间钟 / 成交量钟 / 粗格时间钟
        let groups: [(&[usize], usize); 3] = [(&ALPHA_TIME, 0), (&ALPHA_VOL, 2), (&ALPHA_COARSE, 1)];
        for (alphas, si) in groups {
            for &al in alphas {
                let (tb, yb) = (&ts[si], &ys[si]);
                let (ca, cb) = (letters(tb, yb, al, &mut a, &mut buf), letters(yb, yb, al, &mut b, &mut buf));
                let l = a.len();
                let min_active = (l / 20).max(10);
                let slot = &mut out[pos..pos + N_FEAT];
                pos += N_FEAT;
                if ca < min_active || cb < min_active { continue; }
                let tbl = &tables.t[al];
                // 朴素基线：不允许错位，逐位计分；随机期望由两天字母频率给出
                let (mut ug, mut same, mut pairs) = (0i32, 0f32, 0f32);
                let (mut fa, mut fb) = ([0u32; align::SIZE], [0u32; align::SIZE]);
                for k in 0..l {
                    let (x, y) = (a[k], b[k]);
                    ug += tbl[x as usize * align::SIZE + y as usize];
                    fa[x as usize] += 1; fb[y as usize] += 1;
                    if x != 0 || y != 0 { pairs += 1.0; if x == y { same += 1.0; } }
                }
                let mut e = 0f64;
                for x in 0..align::SIZE { for y in 0..align::SIZE {
                    e += fa[x] as f64 * fb[y] as f64 * tbl[x * align::SIZE + y] as f64;
                } }
                let e = (e / l as f64) as f32;
                let band = l / 4;
                let (sl, ug_f) = (slot, ug as f32);
                let r = align(&a, &b, tbl, OPEN_LOOSE, EXT_LOOSE, band, false, &mut w);
                global_features(&a, &b, &tb.vol, &yb.vol, &w.ops, &r, tbl, ug_f, e, &mut sl[..20]);
                let rt = align(&a, &b, tbl, OPEN_TIGHT, EXT_TIGHT, band, false, &mut w);
                let mut tmp = [f32::NAN; 20];
                global_features(&a, &b, &tb.vol, &yb.vol, &w.ops, &rt, tbl, ug_f, e, &mut tmp);
                for (d, s) in [0usize, 1, 2, 4, 6, 7].iter().enumerate() { sl[20 + d] = tmp[*s]; }
                let rl = align(&a, &b, tbl, OPEN_LOOSE, EXT_LOOSE, band, true, &mut w);
                local_features(&a, &b, &tb.vol, &w.ops, &rl, &mut sl[26..32]);
                sl[32] = ug_f / (MAX_S * l as f32);
                sl[33] = if pairs > 0.0 { same / pairs } else { f32::NAN };
            }
        }
        // 朴素替代物：两天分时成交量相关（不做任何字母化和比对）
        let (va, vb) = (&ts[0].vol, &ys[0].vol);
        let active = va.iter().filter(|&&v| v > 0.0).count().min(vb.iter().filter(|&&v| v > 0.0).count());
        if active >= (va.len() / 20).max(10) {
            out[pos] = pearson(va, vb);
            out[pos + 1] = pearson(&ranks(va), &ranks(vb));
            let com = |v: &[f64]| { let s: f64 = v.iter().sum();
                v.iter().enumerate().map(|(i, x)| i as f64 * x).sum::<f64>() / s / v.len() as f64 };
            out[pos + 2] = (com(va) - com(vb)) as f32;
        }
        pos += N_NAIVE;
    }
    debug_assert_eq!(pos, n_raw);
    dp_ns.fetch_add(t2.elapsed().as_nanos() as u64, Ordering::Relaxed);
    (out, n_trades)
}

fn raw_names() -> Vec<String> {
    let mut v = Vec::new();
    for (win, _, _, _) in WINDOWS.iter() {
        for (clock, alphas) in [("t", &ALPHA_TIME[..]), ("v", &ALPHA_VOL[..]), ("t2x", &ALPHA_COARSE[..])] {
            for &al in alphas {
                let p = format!("{win}_{clock}_{}", ALPHA_NAMES[al]);
                for n in GL_NAMES { v.push(format!("{p}_gl_{n}")); }
                for n in GT_NAMES { v.push(format!("{p}_gt_{n}")); }
                for n in LC_NAMES { v.push(format!("{p}_lc_{n}")); }
                for n in UG_NAMES { v.push(format!("{p}_ug_{n}")); }
            }
        }
        v.push(format!("{win}_naive_volcorr"));
        v.push(format!("{win}_naive_volrankcorr"));
        v.push(format!("{win}_naive_vol_com_shift"));
    }
    v
}

/// 横截面百分位排名，空值保持空值，并列取平均。
fn pct_rank(col: &[f32]) -> Vec<f32> {
    let mut idx: Vec<usize> = (0..col.len()).filter(|&i| col[i].is_finite()).collect();
    idx.sort_by(|&x, &y| col[x].total_cmp(&col[y]).then(x.cmp(&y)));
    let mut r = vec![f32::NAN; col.len()];
    let n = idx.len();
    if n < 2 { return r; }
    let mut i = 0;
    while i < n {
        let mut j = i;
        while j + 1 < n && col[idx[j + 1]] == col[idx[i]] { j += 1; }
        let v = ((i + j) as f32 * 0.5) / (n - 1) as f32;
        for k in i..=j { r[idx[k]] = v; }
        i = j + 1;
    }
    r
}

/// 横截面合成层：跨字母表把排名聚合成每股一个数，再做流动性组内超额和时段对比。
fn cross_section(raw: &[Vec<f32>], n_trades: &[f64], names: &mut Vec<String>) -> Vec<Vec<f32>> {
    let n = raw.len();
    let mut extra: Vec<Vec<f32>> = Vec::new();
    // 流动性十组（当日逐笔笔数）
    let liq: Vec<f32> = n_trades.iter().map(|&v| v as f32).collect();
    let lr = pct_rank(&liq);
    let group: Vec<i32> = lr.iter().map(|&p| if p.is_finite() { ((p * 10.0) as i32).min(9) } else { -1 }).collect();
    let stride_win = (ALPHA_TIME.len() + ALPHA_VOL.len() + ALPHA_COARSE.len()) * N_FEAT + N_NAIVE;
    let mut rank_means: Vec<Vec<Vec<f32>>> = Vec::new();
    for (wi, (win, _, _, _)) in WINDOWS.iter().enumerate() {
        let mut per_key = Vec::new();
        for &(k, kn) in KEY.iter() {
            let ranks: Vec<Vec<f32>> = (0..ALPHA_TIME.len()).map(|a| {
                let c = wi * stride_win + a * N_FEAT + k;
                pct_rank(&raw.iter().map(|r| r[c]).collect::<Vec<_>>())
            }).collect();
            let (mut m, mut sd, mut mx, mut mn) = (vec![f32::NAN; n], vec![f32::NAN; n], vec![f32::NAN; n], vec![f32::NAN; n]);
            for i in 0..n {
                let (mut s, mut s2, mut c, mut hi, mut lo) = (0f32, 0f32, 0f32, f32::MIN, f32::MAX);
                for r in &ranks { let v = r[i]; if v.is_finite() { s += v; s2 += v * v; c += 1.0; hi = hi.max(v); lo = lo.min(v); } }
                if c >= 3.0 {
                    m[i] = s / c; sd[i] = (s2 / c - (s / c).powi(2)).max(0.0).sqrt(); mx[i] = hi; mn[i] = lo;
                }
            }
            // 流动性组内超额：减去同组（按当日成交笔数分十组）的均值
            let (mut gs, mut gc) = ([0f32; 10], [0f32; 10]);
            for i in 0..n { if group[i] >= 0 && m[i].is_finite() { gs[group[i] as usize] += m[i]; gc[group[i] as usize] += 1.0; } }
            let liqx: Vec<f32> = (0..n).map(|i| {
                if group[i] >= 0 && m[i].is_finite() && gc[group[i] as usize] >= 5.0 { m[i] - gs[group[i] as usize] / gc[group[i] as usize] } else { f32::NAN }
            }).collect();
            for (suffix, col) in [("rankmean", m.clone()), ("rankstd", sd), ("rankmax", mx), ("rankmin", mn), ("rankmean_liqx", liqx)] {
                names.push(format!("{win}_cmp_{kn}_{suffix}"));
                extra.push(col);
            }
            per_key.push(m);
        }
        rank_means.push(per_key);
    }
    // 时段对比：下午/尾盘 减 全天
    for (kid, &(_, kn)) in KEY.iter().enumerate() {
        for (wi, win) in [(1usize, "pm"), (2usize, "tail")] {
            let col: Vec<f32> = (0..n).map(|i| rank_means[wi][kid][i] - rank_means[0][kid][i]).collect();
            names.push(format!("{win}_minus_all_{kn}_rankmean"));
            extra.push(col);
        }
    }
    extra
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let date: i64 = args.get(1).expect("用法: align_day_vs_day <今天> <昨天>").parse().unwrap();
    let prev: i64 = args.get(2).expect("用法: align_day_vs_day <今天> <昨天>").parse().unwrap();
    let t_all = Instant::now();
    let codes = list_codes(date, "transaction").expect("股票范围或原始目录检查失败");
    eprintln!("codes = {}", codes.len());
    let mut names = raw_names();
    let n_raw = names.len();
    assert_eq!(n_raw, ((ALPHA_TIME.len() + ALPHA_VOL.len() + ALPHA_COARSE.len()) * N_FEAT + N_NAIVE) * WINDOWS.len());
    assert_eq!(ALPHA_TIME.len() + ALPHA_VOL.len() + ALPHA_COARSE.len(), N_STREAMS);
    let tables = Tables { t: (0..ALPHA_NAMES.len()).map(build_score_table).collect() };
    let pool = rayon::ThreadPoolBuilder::new().num_threads(MAX_THREADS).build().unwrap();
    let (read_ns, bar_ns, dp_ns) = (AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0));
    let res: Vec<(Vec<f32>, f64)> = pool.install(|| codes.par_iter()
        .map(|c| per_stock(c, date, prev, &tables, n_raw, &read_ns, &bar_ns, &dp_ns)).collect());
    let t_stock = t_all.elapsed().as_secs_f64();
    let (raw, n_tr): (Vec<Vec<f32>>, Vec<f64>) = res.into_iter().unzip();
    let extra = cross_section(&raw, &n_tr, &mut names);
    let mut vals = raw;
    names.push("meta_n_trades".into());
    for i in 0..vals.len() { for c in &extra { vals[i].push(c[i]); } vals[i].push(n_tr[i] as f32); }
    let t_cs = t_all.elapsed().as_secs_f64();
    let out = Out { names, codes, vals };
    let stdout = std::io::stdout();
    serde_json::to_writer(std::io::BufWriter::with_capacity(1 << 24, stdout.lock()), &out).unwrap();
    eprintln!("threads={MAX_THREADS} columns={} rows={}", out.names.len(), out.codes.len());
    eprintln!("per_stock_wall={t_stock:.2}s cs_done={t_cs:.2}s total={:.2}s", t_all.elapsed().as_secs_f64());
    eprintln!("thread_cpu_sum read={:.1}s bars={:.1}s dp={:.1}s",
        read_ns.load(Ordering::Relaxed) as f64 / 1e9, bar_ns.load(Ordering::Relaxed) as f64 / 1e9, dp_ns.load(Ordering::Relaxed) as f64 / 1e9);
}

#[cfg(test)]
mod feature_tests {
    use super::*;
    #[test]
    fn today_later_gives_positive_lag() {
        // 今天的模式比昨天整体晚 3 格：平均错位应为 +3 格，今天多出的格子集中在开头
        let y: Vec<u8> = (0..60).map(|i| ((i * 7) % 9 + 1) as u8).collect();
        let mut t = vec![1u8, 2, 3];
        t.extend_from_slice(&y[..57]);
        let tbl = build_score_table(VOLSIG);
        let mut w = Work::default();
        let r = align(&t, &y, &tbl, OPEN_LOOSE, EXT_LOOSE, 15, false, &mut w);
        let v = vec![1.0; 60];
        let mut o = [f32::NAN; 20];
        global_features(&t, &y, &v, &v, &w.ops, &r, &tbl, 0.0, 0.0, &mut o);
        assert!((o[6] * 60.0 - 3.0).abs() < 0.2, "lag_mean bars = {}", o[6] * 60.0);
        assert!(o[18] * 60.0 < 3.0, "insert_center bars = {}", o[18] * 60.0);
    }
}
