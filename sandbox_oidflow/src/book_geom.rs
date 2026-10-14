//! 盘口轨迹几何的统计工具（sandbox v9 专用）。
//! 幂迭代求协方差阵特征值（只做几何量，不建完整谱分解）。

/// 对称阵 A(n x n) 的最大特征值与特征向量，shift 用于放缩（一般给 0）。
pub fn power_eigen(a: &[f64], n: usize, iters: usize) -> (f64, Vec<f64>) {
    let mut v = vec![1.0 / (n as f64).sqrt(); n];
    let mut lambda = 0.0f64;
    for _ in 0..iters {
        let mut w = vec![0.0f64; n];
        for i in 0..n {
            let mut s = 0.0;
            let row = &a[i * n..i * n + n];
            for j in 0..n {
                s += row[j] * v[j];
            }
            w[i] = s;
        }
        let norm = w.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm <= 1e-12 {
            return (0.0, v);
        }
        for x in w.iter_mut() {
            *x /= norm;
        }
        lambda = norm;
        v = w;
    }
    (lambda, v)
}

/// 依次剥离前 k 个成分后的剩余迹，以及各成分占比。返回 (特征值向量降序, 前 k 的占比)。
pub fn spectrum(a: &[f64], n: usize, k: usize, iters: usize) -> (Vec<f64>, f64) {
    let mut m = a.to_vec();
    let mut evals = Vec::with_capacity(k);
    for _ in 0..k {
        let (l, v) = power_eigen(&m, n, iters);
        if l <= 1e-12 {
            break;
        }
        evals.push(l);
        // 收缩：M <- M - l * v v^T
        for i in 0..n {
            for j in 0..n {
                m[i * n + j] -= l * v[i] * v[j];
            }
        }
    }
    while evals.len() < k {
        evals.push(0.0);
    }
    evals.sort_by(|x, y| y.partial_cmp(x).unwrap());
    let tot: f64 = evals.iter().sum();
    let share = if tot > 1e-12 { evals.iter().take(3).sum::<f64>() / tot } else { f64::NAN };
    (evals, share)
}

/// Lempel-Ziv 复杂度（LZ76 的近似：逐字符扫描，出现过的子串不再计数）。
pub fn lz_complexity(seq: &[u8]) -> f64 {
    if seq.is_empty() {
        return f64::NAN;
    }
    let n = seq.len();
    let mut i = 0usize;
    let mut l = 1usize;
    let mut k = 1usize;
    let mut k_max = 1usize;
    let mut c = 1usize;
    while l + k <= n {
        if seq[i + k - 1] == seq[l + k - 1] {
            k += 1;
            if l + k > n {
                c += 1;
                break;
            }
        } else {
            if k > k_max {
                k_max = k;
            }
            i += 1;
            if i == l {
                c += 1;
                l += k_max;
                i = 0;
                k = 1;
                k_max = 1;
            } else {
                k = 1;
            }
        }
    }
    let c = c as f64;
    // 归一化：随机序列的 LZ76 复杂度约为 n / log2(n)
    let norm = if n > 4 { (n as f64) / (n as f64).log2() } else { 1.0 };
    c / norm
}
