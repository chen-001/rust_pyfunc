//! v9 统计工具：幂迭代特征值、Lempel-Ziv、经验分布距离、时序统计栈。
use std::collections::HashMap;

pub fn mean(x: &[f64]) -> f64 {
    if x.is_empty() { return f64::NAN; }
    x.iter().sum::<f64>() / x.len() as f64
}
pub fn std(x: &[f64]) -> f64 {
    if x.len() < 2 { return f64::NAN; }
    let m = mean(x);
    (x.iter().map(|v| (v - m).powi(2)).sum::<f64>() / (x.len() as f64 - 1.0)).sqrt()
}
pub fn skew(x: &[f64]) -> f64 {
    let n = x.len();
    if n < 3 { return f64::NAN; }
    let m = mean(x);
    let s = (x.iter().map(|v| (v - m).powi(2)).sum::<f64>() / n as f64).sqrt();
    if s <= 1e-15 { return 0.0; }
    x.iter().map(|v| ((v - m) / s).powi(3)).sum::<f64>() / n as f64
}
pub fn kurt(x: &[f64]) -> f64 {
    let n = x.len();
    if n < 4 { return f64::NAN; }
    let m = mean(x);
    let s = (x.iter().map(|v| (v - m).powi(2)).sum::<f64>() / n as f64).sqrt();
    if s <= 1e-15 { return 0.0; }
    x.iter().map(|v| ((v - m) / s).powi(4)).sum::<f64>() / n as f64 - 3.0
}
pub fn quantile(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() { return f64::NAN; }
    let pos = (sorted.len() - 1) as f64 * q;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    if lo == hi { return sorted[lo]; }
    sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo as f64)
}
pub fn autocorr1(x: &[f64]) -> f64 {
    if x.len() < 3 { return f64::NAN; }
    let a = &x[..x.len() - 1];
    let b = &x[1..];
    let (ma, mb) = (mean(a), mean(b));
    let mut num = 0.0;
    let (mut da, mut db) = (0.0, 0.0);
    for i in 0..a.len() {
        num += (a[i] - ma) * (b[i] - mb);
        da += (a[i] - ma).powi(2);
        db += (b[i] - mb).powi(2);
    }
    if da <= 1e-15 || db <= 1e-15 { return f64::NAN; }
    num / (da * db).sqrt()
}
/// (斜率, R²) 对序号 1..n 做带截距回归。
pub fn trend(x: &[f64]) -> (f64, f64) {
    let n = x.len();
    if n < 3 { return (f64::NAN, f64::NAN); }
    let my = mean(x);
    let mx = (n as f64 + 1.0) / 2.0;
    let mut sxy = 0.0;
    let mut sxx = 0.0;
    for (i, v) in x.iter().enumerate() {
        let dx = (i as f64 + 1.0) - mx;
        sxy += dx * (v - my);
        sxx += dx * dx;
    }
    if sxx <= 1e-15 { return (f64::NAN, f64::NAN); }
    let slope = sxy / sxx;
    let alpha = my - slope * mx;
    let mut sse = 0.0;
    let mut sst = 0.0;
    for (i, v) in x.iter().enumerate() {
        let fit = alpha + slope * (i as f64 + 1.0);
        sse += (v - fit).powi(2);
        sst += (v - my).powi(2);
    }
    let r2 = if sst > 1e-15 { 1.0 - sse / sst } else { f64::NAN };
    (slope, r2)
}
/// 二阶差分的中位数绝对值（曲率），以及对二阶差分做回归的二次项系数。
pub fn curvature(x: &[f64]) -> (f64, f64) {
    let n = x.len();
    if n < 5 { return (f64::NAN, f64::NAN); }
    let mut d2: Vec<f64> = (0..n - 2).map(|i| x[i + 2] - 2.0 * x[i + 1] + x[i]).collect();
    d2.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let med = quantile(&d2, 0.5).abs();
    // 二次拟合 y = a t^2 + b t + c 的 a（用最小二乘解 3x3）
    let mut s = [0.0f64; 9];
    let mut r = [0.0f64; 3];
    for (i, v) in x.iter().enumerate() {
        let t = i as f64 + 1.0;
        let p = [t * t, t, 1.0];
        for a in 0..3 {
            for b in 0..3 {
                s[a * 3 + b] += p[a] * p[b];
            }
            r[a] += p[a] * v;
        }
    }
    let a = solve3(&s, &r);
    (med, a.unwrap_or(f64::NAN))
}
pub fn solve3(m: &[f64; 9], b: &[f64; 3]) -> Option<f64> {
    let mut a = *m;
    let mut x = *b;
    for c in 0..3 {
        let mut piv = c;
        for r in c + 1..3 {
            if a[r * 3 + c].abs() > a[piv * 3 + c].abs() { piv = r; }
        }
        if a[piv * 3 + c].abs() < 1e-12 { return None; }
        if piv != c {
            for k in 0..3 { a.swap(c * 3 + k, piv * 3 + k); }
            x.swap(c, piv);
        }
        let d = a[c * 3 + c];
        for k in c..3 { a[c * 3 + k] /= d; }
        x[c] /= d;
        for r in 0..3 {
            if r == c { continue; }
            let f = a[r * 3 + c];
            if f == 0.0 { continue; }
            for k in c..3 { a[r * 3 + k] -= f * a[c * 3 + k]; }
            x[r] -= f * x[c];
        }
    }
    Some(x[0])
}
pub fn period_diff_ratio(x: &[f64], frac: f64) -> (f64, f64) {
    let n = x.len();
    if n < 6 { return (f64::NAN, f64::NAN); }
    let k = ((n as f64 * frac) as usize).max(1).min(n / 2);
    let a = mean(&x[..k]);
    let b = mean(&x[n - k..]);
    let d = b - a;
    let r = if a.abs() > 1e-12 { b / a } else { f64::NAN };
    (d, r)
}
/// 基尼系数（非负序列）。
pub fn gini(x: &[f64]) -> f64 {
    let n = x.len();
    if n < 2 { return f64::NAN; }
    let mut s: Vec<f64> = x.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let tot: f64 = s.iter().sum();
    if tot <= 0.0 { return f64::NAN; }
    let mut acc = 0.0;
    for (i, v) in s.iter().enumerate() {
        acc += (2.0 * (i as f64 + 1.0) - n as f64 - 1.0) * v;
    }
    acc / (n as f64 * tot)
}

pub fn entropy_1d(x: &[f64], bins: usize) -> f64 {
    if x.len() < 5 { return f64::NAN; }
    let lo = x.iter().cloned().fold(f64::INFINITY, f64::min);
    let hi = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if hi <= lo { return 0.0; }
    let w = (hi - lo) / bins as f64;
    let mut h = vec![0.0f64; bins];
    for &v in x {
        let b = (((v - lo) / w) as usize).min(bins - 1);
        h[b] += 1.0;
    }
    let n = x.len() as f64;
    let mut e = 0.0;
    for c in h.iter() {
        if *c > 0.0 {
            let p = c / n;
            e -= p * p.ln();
        }
    }
    e / (bins as f64).ln()
}
pub fn lz_complexity(seq: &[u8]) -> f64 {
    if seq.len() < 8 { return f64::NAN; }
    let n = seq.len();
    let (mut i, mut l, mut k, mut kmax, mut c) = (0usize, 1usize, 1usize, 1usize, 1usize);
    while l + k <= n {
        if seq[i + k - 1] == seq[l + k - 1] {
            k += 1;
            if l + k > n { c += 1; break; }
        } else {
            if k > kmax { kmax = k; }
            i += 1;
            if i == l { c += 1; l += kmax; i = 0; k = 1; kmax = 1; } else { k = 1; }
        }
    }
    c as f64 / (n as f64 / (n as f64).log2())
}
pub fn max_range_product(x: &[f64]) -> f64 {
    let n = x.len();
    if n < 2 { return f64::NAN; }
    let mx = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mn = x.iter().cloned().fold(f64::INFINITY, f64::min);
    let mpos = x.iter().position(|v| *v == mx).unwrap() as f64;
    let mneg = x.iter().position(|v| *v == mn).unwrap() as f64;
    ((mx - mn) * (mpos - mneg)).abs() / n as f64
}
/// 21 个时序统计（顺序固定，与 names 一致）。
pub fn ts21(x: &[f64]) -> [f64; 21] {
    let mut out = [f64::NAN; 21];
    if x.is_empty() { return out; }
    let mut s = x.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let m = mean(x);
    let sd = std(x);
    out[0] = m;
    out[1] = quantile(&s, 0.5);
    out[2] = sd;
    out[3] = skew(x);
    out[4] = kurt(x);
    out[5] = quantile(&s, 0.05);
    out[6] = quantile(&s, 0.25);
    out[7] = quantile(&s, 0.75);
    out[8] = quantile(&s, 0.95);
    out[9] = out[7] - out[6];
    out[10] = if m.abs() > 1e-12 { sd / m.abs() } else { f64::NAN };
    out[11] = autocorr1(x);
    out[12] = out[11].abs();
    let (sl, _) = trend(x);
    out[13] = sl;
    let (curv, quad) = curvature(x);
    out[14] = curv;
    out[15] = quad;
    let (pd, pr) = period_diff_ratio(x, 0.25);
    out[16] = pd;
    out[17] = pr;
    out[18] = lz_complexity(&sign_seq(x, m));
    out[19] = entropy_1d(x, 20);
    out[20] = max_range_product(x);
    out
}
fn sign_seq(x: &[f64], m: f64) -> Vec<u8> {
    x.iter().map(|v| if *v >= m { 1u8 } else { 0u8 }).collect()
}
/// 极值事件的计数与跨度：> p90 的次数、首末位置跨度、最大连续段。
pub fn extreme_stats(x: &[f64]) -> (f64, f64, f64) {
    let n = x.len();
    if n < 10 { return (f64::NAN, f64::NAN, f64::NAN); }
    let mut s = x.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let hi = quantile(&s, 0.9);
    let lo = quantile(&s, 0.1);
    let mut cnt_hi = 0.0;
    let mut first = n;
    let mut last = 0usize;
    let mut run = 0usize;
    let mut best = 0usize;
    for (i, v) in x.iter().enumerate() {
        if *v > hi {
            cnt_hi += 1.0;
            first = first.min(i);
            last = last.max(i);
            run += 1;
            best = best.max(run);
        } else {
            run = 0;
        }
    }
    let _ = lo;
    (
        cnt_hi / n as f64,
        if last > first { (last - first) as f64 / n as f64 } else { 0.0 },
        best as f64 / n as f64,
    )
}
/// 幂迭代求最大特征值与特征向量（对称阵）。
pub fn power_eigen(a: &[f64], n: usize, iters: usize) -> (f64, Vec<f64>) {
    let mut v = vec![1.0 / (n as f64).sqrt(); n];
    let mut lambda = 0.0f64;
    for _ in 0..iters {
        let mut w = vec![0.0f64; n];
        for i in 0..n {
            let row = &a[i * n..i * n + n];
            let mut s = 0.0;
            for j in 0..n { s += row[j] * v[j]; }
            w[i] = s;
        }
        let norm = w.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm <= 1e-12 { return (0.0, v); }
        for x in w.iter_mut() { *x /= norm; }
        lambda = norm;
        v = w;
    }
    (lambda, v)
}
/// 前 k 个特征值（幂迭代加收缩）。
pub fn spectrum(a: &[f64], n: usize, k: usize, iters: usize) -> Vec<f64> {
    let mut m = a.to_vec();
    let mut evals = Vec::with_capacity(k);
    for _ in 0..k {
        let (l, v) = power_eigen(&m, n, iters);
        if l <= 1e-12 { break; }
        evals.push(l);
        for i in 0..n {
            for j in 0..n {
                m[i * n + j] -= l * v[i] * v[j];
            }
        }
    }
    while evals.len() < k { evals.push(0.0); }
    evals.sort_by(|x, y| y.partial_cmp(x).unwrap());
    evals
}
/// Pearson（成对完整观测）。
pub fn corr(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    if n < 3 { return f64::NAN; }
    let (ma, mb) = (mean(&a[..n]), mean(&b[..n]));
    let (mut num, mut da, mut db) = (0.0, 0.0, 0.0);
    for i in 0..n {
        let (x, y) = (a[i] - ma, b[i] - mb);
        num += x * y;
        da += x * x;
        db += y * y;
    }
    if da <= 1e-15 || db <= 1e-15 { return f64::NAN; }
    num / (da * db).sqrt()
}
/// DTW 距离（Sakoe-Chiba 带宽 w）。
pub fn dtw(a: &[f64], b: &[f64], w: usize) -> f64 {
    let (n, m) = (a.len(), b.len());
    if n == 0 || m == 0 { return f64::INFINITY; }
    let band = w.max(n.abs_diff(m)) + 1;
    let mut prev = vec![f64::INFINITY; m + 1];
    let mut cur = vec![f64::INFINITY; m + 1];
    prev[0] = 0.0;
    for i in 1..=n {
        for j in 0..=m { cur[j] = f64::INFINITY; }
        let lo = (i.saturating_sub(band)).max(1);
        let hi = (i + band).min(m);
        for j in lo..=hi {
            let cost = (a[i - 1] - b[j - 1]).abs();
            let best = prev[j].min(prev[j - 1]).min(cur[j - 1]);
            cur[j] = if best.is_finite() { cost + best } else { f64::INFINITY };
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    let d = prev[m];
    if d.is_finite() { d / (n.max(m) as f64) } else { f64::NAN }
}
/// 固定种子 xorshift64*，用于零分布置换。
pub struct Rng(u64);
impl Rng {
    pub fn new(seed: u64) -> Self { Rng(seed | 1) }
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }
    pub fn shuffle<T>(&mut self, v: &mut [T]) {
        let n = v.len();
        if n < 2 { return; }
        for i in (1..n).rev() {
            let j = (self.next_u64() % (i as u64 + 1)) as usize;
            v.swap(i, j);
        }
    }
}
/// 分层零分布：置换一个序列的时间顺序，看实际相关的排名分位。
pub fn permutation_rank(x: &[f64], y: &[f64], reps: usize, seed: u64) -> f64 {
    let real = corr(x, y);
    if !real.is_finite() { return f64::NAN; }
    let mut rng = Rng::new(seed);
    let mut buf = y.to_vec();
    let mut ge = 0.0;
    for _ in 0..reps {
        rng.shuffle(&mut buf);
        let c = corr(x, &buf);
        if c.is_finite() && c.abs() >= real.abs() { ge += 1.0; }
    }
    ge / reps as f64
}
/// 分段相关：按 col1 > col2 把行切成两段，各算一次 Pearson。
pub fn segment_corr(a: &[f64], b: &[f64]) -> (f64, f64) {
    let n = a.len().min(b.len());
    let mut xa = Vec::new();
    let mut ya = Vec::new();
    let mut xb = Vec::new();
    let mut yb = Vec::new();
    for i in 0..n {
        if a[i] > b[i] { xa.push(a[i]); ya.push(b[i]); } else { xb.push(a[i]); yb.push(b[i]); }
    }
    let ca = if xa.len() >= 5 { corr(&xa, &ya) } else { f64::NAN };
    let cb = if xb.len() >= 5 { corr(&xb, &yb) } else { f64::NAN };
    (ca, cb)
}
/// 直方图众数位置与集中度。
pub fn hist_conc(x: &[f64], bins: usize) -> (f64, f64) {
    if x.len() < 5 { return (f64::NAN, f64::NAN); }
    let lo = x.iter().cloned().fold(f64::INFINITY, f64::min);
    let hi = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if hi <= lo { return (0.0, 1.0); }
    let w = (hi - lo) / bins as f64;
    let mut h = vec![0.0f64; bins];
    for &v in x { h[(((v - lo) / w) as usize).min(bins - 1)] += 1.0; }
    let tot = x.len() as f64;
    let mut best = (0usize, 0.0f64);
    let mut hhi = 0.0;
    for (i, c) in h.iter().enumerate() {
        let p = c / tot;
        hhi += p * p;
        if *c > best.1 { best = (i, *c); }
    }
    (best.0 as f64 / bins as f64, hhi)
}
/// 用 map 统计计数并返回众数占比。
pub fn mode_share<T: std::hash::Hash + Eq + Copy>(v: &[T]) -> f64 {
    if v.is_empty() { return f64::NAN; }
    let mut m: HashMap<T, usize> = HashMap::new();
    let mut best = 0usize;
    for x in v {
        let c = m.entry(*x).or_insert(0);
        *c += 1;
        best = best.max(*c);
    }
    best as f64 / v.len() as f64
}
