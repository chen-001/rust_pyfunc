// hm32 单股单日分阶段计时 + 无矩阵（O(n log n)）改法对照
use std::env;
use std::fs::File;
use std::io::Read;
use std::time::Instant;

const MKT_COL_EXCHTIME: usize = 4;
const MKT_COL_LAST_PRC: usize = 6;
const MKT_COL_VOLUME: usize = 14;
const MKT_COL_ASK_VOL1: usize = 22;
const MKT_COL_BID_VOL1: usize = 24;
const MKT_NCOL: usize = 61;
const US: i64 = 1_000_000;
const DAY_US: i64 = 86_400_000_000;
const ROOT: &str = "/ssd_data/stock";
const SESSIONS: [(&str, i64, i64); 5] = [
    ("allday", (9 * 3600 + 30 * 60) * US, (14 * 3600 + 57 * 60) * US),
    ("open30", (9 * 3600 + 30 * 60) * US, 10 * 3600 * US),
    ("morning", (9 * 3600 + 30 * 60) * US, (11 * 3600 + 30 * 60) * US),
    ("afternoon", 13 * 3600 * US, (14 * 3600 + 57 * 60) * US),
    ("close30", (14 * 3600 + 30 * 60) * US, (14 * 3600 + 57 * 60) * US),
];
const PER_GROUP: usize = 11;
const PER_SESSION: usize = PER_GROUP * 2;

#[derive(Clone, Copy, Default)]
struct MktRow { time_us: i64, last_prc: f64, volume: f64, ask_vol1: f64, bid_vol1: f64 }

#[inline]
fn parse_i64_at(b: &[u8]) -> i64 {
    let mut i = 0usize;
    while i < b.len() && b[i] == b' ' { i += 1; }
    let mut neg = false;
    if i < b.len() && (b[i] == b'-' || b[i] == b'+') { neg = b[i] == b'-'; i += 1; }
    let mut v: i64 = 0;
    while i < b.len() {
        let c = b[i];
        if !(b'0'..=b'9').contains(&c) { break; }
        v = v * 10 + (c - b'0') as i64;
        i += 1;
    }
    if neg { -v } else { v }
}

#[inline]
fn parse_f64_at(b: &[u8]) -> f64 {
    if b.is_empty() { return f64::NAN; }
    std::str::from_utf8(b).ok().and_then(|s| s.trim().parse::<f64>().ok()).unwrap_or(f64::NAN)
}

#[inline]
fn in_span(t: i64, lo: i64, hi: i64) -> bool { let d = t.rem_euclid(DAY_US); d >= lo && d <= hi }

fn read_market_f64(code: &str, date: i64) -> Option<Vec<MktRow>> {
    let filename = format!("{}_{}_market_data.csv", code, date);
    let p = std::path::Path::new(ROOT).join(date.to_string()).join("market_data").join(&filename);
    if !p.exists() { return None; }
    let mut content = String::new();
    File::open(&p).ok()?.read_to_string(&mut content).ok()?;
    let mut out: Vec<MktRow> = Vec::with_capacity(content.len() / 200 + 1);
    let mut lines = content.lines();
    lines.next();
    for line in lines {
        if line.is_empty() { continue; }
        let bytes = line.as_bytes();
        let mut f: [&[u8]; MKT_NCOL] = [&[][..]; MKT_NCOL];
        let mut start = 0usize;
        let mut col = 0usize;
        for (i, &b) in bytes.iter().enumerate() {
            if b == b',' {
                if col < MKT_NCOL { f[col] = &bytes[start..i]; }
                col += 1;
                start = i + 1;
            }
        }
        if col < MKT_NCOL { f[col] = &bytes[start..]; }
        out.push(MktRow {
            time_us: parse_i64_at(f[MKT_COL_EXCHTIME]) + 8 * 3600 * US,
            last_prc: parse_f64_at(f[MKT_COL_LAST_PRC]),
            volume: parse_f64_at(f[MKT_COL_VOLUME]),
            ask_vol1: parse_f64_at(f[MKT_COL_ASK_VOL1]),
            bid_vol1: parse_f64_at(f[MKT_COL_BID_VOL1]),
        });
    }
    Some(out)
}

fn pairwise_sum(a: &[f64]) -> f64 {
    let n = a.len();
    if n < 8 {
        let mut res = -0.0f64;
        for &x in a.iter() { res += x; }
        return res;
    }
    if n <= 128 {
        let mut r = [0.0f64; 8];
        r.copy_from_slice(&a[..8]);
        let mut i = 8usize;
        let stop = n - (n % 8);
        while i < stop {
            for k in 0..8 { r[k] += a[i + k]; }
            i += 8;
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        while i < n { res += a[i]; i += 1; }
        return res;
    }
    let mut n2 = n / 2;
    n2 -= n2 % 8;
    pairwise_sum(&a[..n2]) + pairwise_sum(&a[n2..])
}

fn nanmean(a: &[f64]) -> f64 {
    let mut buf: Vec<f64> = Vec::with_capacity(a.len());
    for &x in a.iter() { if !x.is_nan() { buf.push(x); } }
    if buf.is_empty() { return f64::NAN; }
    pairwise_sum(&buf) / (buf.len() as f64)
}

fn nanstd(a: &[f64]) -> f64 {
    let mut buf: Vec<f64> = Vec::with_capacity(a.len());
    for &x in a.iter() { if !x.is_nan() { buf.push(x); } }
    let cnt = buf.len();
    if cnt == 0 { return f64::NAN; }
    let avg = pairwise_sum(&buf) / (cnt as f64);
    let mut sq: Vec<f64> = Vec::with_capacity(cnt);
    for &x in buf.iter() { let d = avg - x; sq.push(d * d); }
    (pairwise_sum(&sq) / ((cnt as f64) - 1.0)).sqrt()
}

fn standardize(x: &[f64]) -> Vec<f64> {
    let mean = nanmean(x);
    let std = nanstd(x);
    let mut out: Vec<f64> = Vec::with_capacity(x.len());
    for &v in x.iter() { out.push((v - mean) / std); }
    out
}

fn matvec(m: &[f64], n: usize, x: &[f64], y: &mut [f64]) {
    for i in 0..n {
        let row = &m[i * n..(i + 1) * n];
        let mut s = 0.0f64;
        for j in 0..n { s += row[j] * x[j]; }
        y[i] = s;
    }
}

#[inline]
fn dot(x: &[f64], y: &[f64]) -> f64 {
    let mut s = 0.0f64;
    for i in 0..x.len() { s += x[i] * y[i]; }
    s
}

fn max_eigenvalue(m: &[f64], n: usize) -> (f64, Vec<f64>, usize, f64) {
    let mut v = vec![1.0f64 / (n as f64).sqrt(); n];
    let mut new_v = vec![0.0f64; n];
    let mut temp = vec![0.0f64; n];
    let mut iters = 0usize;
    let mut tmv = 0.0f64;
    let t0 = Instant::now();
    matvec(m, n, &v, &mut temp);
    tmv += t0.elapsed().as_secs_f64() * 1000.0;
    let mut eigenvalue = dot(&v, &temp);
    let max_iter = 30;
    let tolerance = 1e-4;
    for _ in 0..max_iter {
        iters += 1;
        let prev_eigenvalue = eigenvalue;
        let t1 = Instant::now();
        matvec(m, n, &v, &mut new_v);
        let norm = dot(&new_v, &new_v).sqrt();
        if norm < 1e-5 { break; }
        for x in new_v.iter_mut() { *x /= norm; }
        matvec(m, n, &new_v, &mut temp);
        tmv += t1.elapsed().as_secs_f64() * 1000.0;
        eigenvalue = dot(&new_v, &temp);
        std::mem::swap(&mut v, &mut new_v);
        let rel_error = (eigenvalue - prev_eigenvalue).abs();
        if rel_error < tolerance * eigenvalue.abs() { break; }
    }
    (eigenvalue, v, iters, tmv)
}

fn percentile_sorted(arr: &[f64], q: f64) -> f64 {
    let n = arr.len();
    if n == 0 { return f64::NAN; }
    let virtual_index = (n as f64 - 1.0) * q;
    if virtual_index >= (n as f64 - 1.0) { return arr[n - 1]; }
    if virtual_index < 0.0 { return arr[0]; }
    let lo = virtual_index.floor();
    let hi = lo + 1.0;
    let t = virtual_index - lo;
    let a = arr[lo as usize];
    let b = arr[hi as usize];
    if t >= 0.5 { b - (b - a) * (1.0 - t) } else { a + (b - a) * t }
}

fn vec_stats(eigenvalue: f64, v: &[f64]) -> [f64; PER_GROUP] {
    let n = v.len() as f64;
    let mean = pairwise_sum(v) / n;
    let mut dz: Vec<f64> = Vec::with_capacity(v.len());
    for &x in v.iter() { dz.push(x - mean); }
    let mut t2 = Vec::with_capacity(dz.len());
    let mut t3 = Vec::with_capacity(dz.len());
    let mut t4 = Vec::with_capacity(dz.len());
    for &d in dz.iter() { let d2 = d * d; t2.push(d2); t3.push(d2 * d); t4.push(d2 * d2); }
    let m2 = pairwise_sum(&t2) / n;
    let m3 = pairwise_sum(&t3) / n;
    let m4 = pairwise_sum(&t4) / n;
    let eps = f64::EPSILON;
    let thr = eps * mean;
    let zero = m2 <= thr * thr;
    let std = m2.sqrt();
    let skew = if zero { f64::NAN } else { m3 / m2.powf(1.5) };
    let kurt = { let base = if zero { f64::NAN } else { m4 / (m2 * m2) }; base - 3.0 };
    let mut vmax = f64::NEG_INFINITY;
    let mut vmin = f64::INFINITY;
    for &x in v.iter() { if x > vmax { vmax = x; } if x < vmin { vmin = x; } }
    let mut sorted: Vec<f64> = v.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let q1 = percentile_sorted(&sorted, 0.25);
    let q3 = percentile_sorted(&sorted, 0.75);
    [eigenvalue, mean, std, skew, kurt, vmax, vmin, percentile_sorted(&sorted, 0.5), q1, q3, q3 - q1]
}

fn haha_naive(zp: &[f64], zv: &[f64]) -> ([f64; PER_GROUP], f64, f64, usize, f64) {
    let n = zp.len();
    let t0 = Instant::now();
    let mut m = vec![0.0f64; n * n];
    let t_alloc = t0.elapsed().as_secs_f64() * 1000.0;
    let t1 = Instant::now();
    for i in 0..n {
        let pi = zp[i];
        let vi = zv[i];
        let row = &mut m[i * n..(i + 1) * n];
        for (j, cell) in row.iter_mut().enumerate() {
            *cell = if i >= j { (vi - zv[j]).abs() } else { (pi - zp[j]).abs() };
        }
    }
    let t_build = t1.elapsed().as_secs_f64() * 1000.0;
    let (ev, v, iters, tmv) = max_eigenvalue(&m, n);
    (vec_stats(ev, &v), t_alloc, t_build, iters, tmv)
}

struct Fenwick { t: Vec<f64> }
impl Fenwick {
    fn new(n: usize) -> Self { Fenwick { t: vec![0.0; n + 1] } }
    fn add(&mut self, pos: usize, v: f64) {
        let mut i = pos + 1;
        while i < self.t.len() { self.t[i] += v; i += i & i.wrapping_neg(); }
    }
    fn prefix(&self, k: usize) -> f64 {
        let mut i = k;
        let mut s = 0.0;
        while i > 0 { s += self.t[i]; i -= i & i.wrapping_neg(); }
        s
    }
}

fn ranks(vals: &[f64]) -> (Vec<usize>, Vec<f64>) {
    let n = vals.len();
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| vals[a].partial_cmp(&vals[b]).unwrap());
    let mut rank = vec![0usize; n];
    let mut sorted = vec![0.0f64; n];
    for (pos, &i) in idx.iter().enumerate() { rank[i] = pos; sorted[pos] = vals[i]; }
    (rank, sorted)
}

#[inline]
fn upper_bound(a: &[f64], q: f64) -> usize {
    let mut lo = 0usize;
    let mut hi = a.len();
    while lo < hi {
        let mid = (lo + hi) / 2;
        if a[mid] <= q { lo = mid + 1; } else { hi = mid; }
    }
    lo
}

fn matvec_free(zp: &[f64], zv: &[f64], rp_: &[usize], rv: &[usize], sp: &[f64], sv: &[f64], x: &[f64], y: &mut [f64]) {
    let n = x.len();
    {
        let mut bit_x = Fenwick::new(n);
        let mut bit_vx = Fenwick::new(n);
        let mut tot_x = 0.0f64;
        let mut tot_vx = 0.0f64;
        for i in 0..n {
            let q = zv[i];
            bit_x.add(rv[i], x[i]);
            bit_vx.add(rv[i], q * x[i]);
            tot_x += x[i];
            tot_vx += q * x[i];
            let k = upper_bound(sv, q);
            let lx = bit_x.prefix(k);
            let lvx = bit_vx.prefix(k);
            let gx = tot_x - lx;
            let gvx = tot_vx - lvx;
            y[i] = q * lx - lvx + gvx - q * gx;
        }
    }
    {
        let mut bit_x = Fenwick::new(n);
        let mut bit_px = Fenwick::new(n);
        let mut tot_x = 0.0f64;
        let mut tot_px = 0.0f64;
        for i in (0..n).rev() {
            if i + 1 < n {
                let j = i + 1;
                bit_x.add(rp_[j], x[j]);
                bit_px.add(rp_[j], zp[j] * x[j]);
                tot_x += x[j];
                tot_px += zp[j] * x[j];
            }
            let q = zp[i];
            let k = upper_bound(sp, q);
            let lx = bit_x.prefix(k);
            let lpx = bit_px.prefix(k);
            let gx = tot_x - lx;
            let gpx = tot_px - lpx;
            y[i] += q * lx - lpx + gpx - q * gx;
        }
    }
}

fn max_eigenvalue_free(zp: &[f64], zv: &[f64], n: usize) -> (f64, Vec<f64>, usize) {
    let (rp_, sp) = ranks(zp);
    let (rv, sv) = ranks(zv);
    let mut v = vec![1.0f64 / (n as f64).sqrt(); n];
    let mut new_v = vec![0.0f64; n];
    let mut temp = vec![0.0f64; n];
    matvec_free(zp, zv, &rp_, &rv, &sp, &sv, &v, &mut temp);
    let mut eigenvalue = dot(&v, &temp);
    let mut iters = 0usize;
    for _ in 0..30 {
        iters += 1;
        let prev = eigenvalue;
        matvec_free(zp, zv, &rp_, &rv, &sp, &sv, &v, &mut new_v);
        let norm = dot(&new_v, &new_v).sqrt();
        if norm < 1e-5 { break; }
        for x in new_v.iter_mut() { *x /= norm; }
        matvec_free(zp, zv, &rp_, &rv, &sp, &sv, &new_v, &mut temp);
        eigenvalue = dot(&new_v, &temp);
        std::mem::swap(&mut v, &mut new_v);
        if (eigenvalue - prev).abs() < 1e-4 * eigenvalue.abs() { break; }
    }
    (eigenvalue, v, iters)
}

fn haha_fast(zp: &[f64], zv: &[f64]) -> ([f64; PER_GROUP], usize) {
    let n = zp.len();
    let (ev, v, iters) = max_eigenvalue_free(zp, zv, n);
    (vec_stats(ev, &v), iters)
}

fn ms(d: std::time::Duration) -> f64 { d.as_secs_f64() * 1000.0 }

fn main() {
    let args: Vec<String> = env::args().collect();
    let date: i64 = args[1].parse().unwrap();
    let code = args[2].clone();
    let mode = args.get(3).cloned().unwrap_or_else(|| "both".to_string());

    let t0 = Instant::now();
    let rows = read_market_f64(&code, date).expect("no data");
    let t_read = t0.elapsed();
    let n = rows.len();
    println!("== {} {} mode={} rows_all={} read_ms={:.1}", code, date, mode, n, ms(t_read));

    let t1 = Instant::now();
    let mut volume = vec![0.0f64; n];
    for i in 0..n {
        let d = if i == 0 { f64::NAN } else { rows[i].volume - rows[i - 1].volume };
        let d = if d == 0.0 { 1.0 } else { d };
        volume[i] = d.ln();
    }
    let mut ask = vec![0.0f64; n];
    let mut bid = vec![0.0f64; n];
    for i in 0..n {
        let a = rows[i].ask_vol1;
        ask[i] = (if a == 0.0 { 1.0 } else { a }).ln();
        let b = rows[i].bid_vol1;
        bid[i] = (if b == 0.0 { 1.0 } else { b }).ln();
    }
    let price: Vec<f64> = rows.iter().map(|r| r.last_prc).collect();
    let t_xform = t1.elapsed();
    println!("   transform_ms={:.1}", ms(t_xform));

    let allday: Vec<usize> = (0..n)
        .filter(|&i| in_span(rows[i].time_us, SESSIONS[0].1, SESSIONS[0].2))
        .filter(|&i| rows[i].last_prc != 0.0)
        .collect();

    let mut tot_alloc = 0.0; let mut tot_build = 0.0; let mut tot_mv = 0.0; let mut tot_std = 0.0;
    let mut tot_iters = 0usize; let mut tot_n = 0usize;
    let mut tot_fast = 0.0;
    let mut naive_vals: Option<[[f64; PER_GROUP]; 2]> = None;
    let mut g_max_diff = 0.0f64;
    let mut dump_n: Vec<f64> = Vec::new();
    let mut dump_f: Vec<f64> = Vec::new();
    let mut any_bad = 0usize;

    for (si, (sess, lo, hi)) in SESSIONS.iter().enumerate() {
        let idx: Vec<usize> = allday.iter().copied()
            .filter(|&i| si == 0 || in_span(rows[i].time_us, *lo, *hi)).collect();
        let ns = idx.len();
        tot_n += ns;
        if ns == 0 { println!("   [{}] n=0", sess); continue; }
        let p: Vec<f64> = idx.iter().map(|&i| price[i]).collect();
        let v: Vec<f64> = idx.iter().map(|&i| volume[i]).collect();
        let a: Vec<f64> = idx.iter().map(|&i| ask[i]).collect();
        let b: Vec<f64> = idx.iter().map(|&i| bid[i]).collect();
        let ts = Instant::now();
        let zp = standardize(&p); let zv = standardize(&v);
        let za = standardize(&a); let zb = standardize(&b);
        let t_std = ts.elapsed();
        tot_std += ms(t_std);
        let has_nan = zp.iter().chain(zv.iter()).chain(za.iter()).chain(zb.iter()).any(|x| x.is_nan());

        let bad = zp.iter().chain(zv.iter()).chain(za.iter()).chain(zb.iter()).filter(|x| !x.is_finite()).count();
        if mode == "probe" {
            let f = |v: &[f64]| v.iter().filter(|x| !x.is_finite()).count();
            let nan = |v: &[f64]| v.iter().filter(|x| x.is_nan()).count();
            println!("PROBE {} {} {} n={} price_nf={}(nan {}) vol_nf={}(nan {}) ask_nf={}(nan {}) bid_nf={}(nan {})",
                code, date, sess, ns, f(&zp), nan(&zp), f(&zv), nan(&zv), f(&za), nan(&za), f(&zb), nan(&zb));
        }
        if mode == "check" {
            any_bad += bad;
            println!("CHECK {} {} {} n={} nonfinite={}", code, date, sess, ns, bad);
            continue;
        }
        let mut line = format!("   [{:9}] n={:5} std_ms={:6.1} nan={} nonfinite={}", sess, ns, ms(t_std), has_nan, bad);
        if mode != "fast" {
            let (s1, ta, tb, it1, mv1) = haha_naive(&zp, &zv);
            let (s2, _, _, it2, mv2) = haha_naive(&za, &zb);
            tot_alloc += ta; tot_build += tb; tot_mv += mv1 + mv2; tot_iters += it1 + it2;
            let mem = ns as f64 * ns as f64 * 8.0 / 1e6;
            line.push_str(&format!(" | naive alloc={:.1} build={:.1} matvec={:.1} iters={}+{} mem_MB={:.0} ev1={:.6}", ta, tb, mv1 + mv2, it1, it2, mem, s1[0]));
            naive_vals = Some([s1, s2]);
            dump_n.extend_from_slice(&s1);
            dump_n.extend_from_slice(&s2);
        }
        if mode != "naive" && !has_nan {
            let tf = Instant::now();
            let (s1f, itf1) = haha_fast(&zp, &zv);
            let (s2f, itf2) = haha_fast(&za, &zb);
            let tfa = ms(tf.elapsed());
            tot_fast += tfa;
            dump_f.extend_from_slice(&s1f);
            dump_f.extend_from_slice(&s2f);
            if let Some(nv) = naive_vals {
                let mut md = 0.0f64;
                for g in 0..2 {
                    for k in 0..PER_GROUP {
                        let a = nv[g][k];
                        let b = if g == 0 { s1f[k] } else { s2f[k] };
                        let d = (a - b).abs();
                        let d = if d.is_nan() { 0.0 } else { d };
                        if d > md { md = d; }
                    }
                }
                if md > g_max_diff { g_max_diff = md; }
                line.push_str(&format!(" | fast={:.1}ms iters={}+{} maxdiff={:.3e}", tfa, itf1, itf2, md));
            } else {
                line.push_str(&format!(" | fast={:.1}ms iters={}+{} ev1={:.6}", tfa, itf1, itf2, s1f[0]));
            }
        }
        println!("{}", line);
    }
    println!("   TOTAL naive: alloc={:.1} build={:.1} matvec={:.1} std={:.1} iters={} sum_n={}",
        tot_alloc, tot_build, tot_mv, tot_std, tot_iters, tot_n);
    if mode == "check" { println!("CHECK_TOTAL {} {} nonfinite={}", code, date, any_bad); return; }
    println!("   TOTAL fast_ms={:.1}  maxdiff_all={:.3e}", tot_fast, g_max_diff);
    if mode == "dump" {
        let j = |v: &[f64]| v.iter().map(|x| format!("{:.17e}", x)).collect::<Vec<_>>().join(",");
        println!("DUMP_NAIVE={}", j(&dump_n));
        println!("DUMP_FAST={}", j(&dump_f));
    }
}
