//! 相邻分钟截面相关系数的逐股贡献度因子
//!
//! 对输入的 (T, N) 字段矩阵，计算每相邻两行的截面 Pearson 相关系数，
//! 得到 corr 序列（长度 T-1）。再用 4 种方法将每个相关系数分解到每只股票，
//! 得到 4 个 (T-1) × N 贡献度矩阵。然后衍生出多种统计量作为因子。
//!
//! 4 种贡献度方法：
//! - m1: 加性分解 c_i = dx_i * dy_i / sqrt(Sxx * Syy)，sum(c_i) = r
//! - m2: 留一影响 r - r_{-i}（闭式更新）
//! - m3: 影响函数 IF = zx*zy - 0.5*r*(zx²+zy²)
//! - m4: 回归斜率 DFBETA = dx * resid / (Sxx * (1-h))

use ndarray::{s, Array2};
use numpy::{PyArray1, PyReadonlyArray2, PyReadonlyArray3};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rayon::prelude::*;

const STAT_NAMES: [&str; 8] = ["mean", "std", "skew", "kurt", "p5", "p95", "trend", "ac1"];
const METHOD_NAMES: [&str; 4] = ["m1", "m2", "m3", "m4"];

// ============================ 统计工具函数 ============================

/// 计算 8 个统计量: mean, std, skew, kurt(超额), p5, p95, trend, ac1
/// scratch 用于避免重复堆分配（替代 data.to_vec() + sort）
fn compute_col_stats_scratch(data: &[f64], scratch: &mut Vec<f64>) -> [f64; 8] {
    let n = data.len();
    if n == 0 {
        return [f64::NAN; 8];
    }
    let nf = n as f64;

    let mut sum = 0.0f64;
    for &v in data {
        sum += v;
    }
    let mean = sum / nf;

    let mut m2 = 0.0f64;
    let mut m3 = 0.0f64;
    let mut m4 = 0.0f64;
    for &v in data {
        let d = v - mean;
        let d2 = d * d;
        m2 += d2;
        m3 += d2 * d;
        m4 += d2 * d2;
    }
    let var = m2 / nf;
    let std = var.sqrt();
    let skew = if std > 1e-300 {
        (m3 / nf) / (std * std * std)
    } else {
        0.0
    };
    let kurt = if var > 1e-300 {
        (m4 / nf) / (var * var) - 3.0
    } else {
        0.0
    };

    let p5 = fast_percentile_scratch(data, 0.05, scratch);
    let p95 = fast_percentile_scratch(data, 0.95, scratch);

    let trend = corr_with_arange(data, 1.0);

    let ac1 = if n > 1 && m2 > 0.0 {
        let mut num = 0.0f64;
        for i in 0..n - 1 {
            num += (data[i] - mean) * (data[i + 1] - mean);
        }
        num / m2
    } else {
        0.0
    };

    [mean, std, skew, kurt, p5, p95, trend, ac1]
}

/// O(n) 分位数：select_nth 找到目标位置，再线性扫描找相邻值做插值
fn fast_percentile_scratch(data: &[f64], p: f64, scratch: &mut Vec<f64>) -> f64 {
    let n = data.len();
    if n == 0 {
        return f64::NAN;
    }
    if n == 1 {
        return data[0];
    }
    let idx_f = p * (n as f64 - 1.0);
    let lo = idx_f.floor() as usize;
    let hi = (lo + 1).min(n - 1);
    let frac = idx_f - lo as f64;

    scratch.clear();
    scratch.extend_from_slice(data);

    let (_, lo_ref, right) = scratch.select_nth_unstable_by(lo, |a, b| a.partial_cmp(b).unwrap());
    let lo_val = *lo_ref;

    if hi == lo {
        return lo_val;
    }

    let hi_val = right.iter().copied().fold(f64::INFINITY, f64::min);
    lo_val * (1.0 - frac) + hi_val * frac
}

/// data 与 [start, start+1, ..., start+n-1] 的 Pearson 相关系数
fn corr_with_arange(data: &[f64], start: f64) -> f64 {
    let n = data.len();
    if n <= 1 {
        return 0.0;
    }
    let nf = n as f64;
    let data_mean: f64 = data.iter().sum::<f64>() / nf;
    let idx_mean = start + (nf - 1.0) / 2.0;

    let mut cov = 0.0f64;
    let mut var_idx = 0.0f64;
    let mut var_data = 0.0f64;
    for (i, &v) in data.iter().enumerate() {
        let di = (start + i as f64) - idx_mean;
        let dd = v - data_mean;
        cov += di * dd;
        var_idx += di * di;
        var_data += dd * dd;
    }
    if var_idx > 0.0 && var_data > 0.0 {
        cov / (var_idx * var_data).sqrt()
    } else {
        0.0
    }
}

/// 两个等长切片的 Pearson 相关系数
fn pearson_corr(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len();
    if n == 0 {
        return 0.0;
    }
    let nf = n as f64;
    let mut sx = 0.0f64;
    let mut sy = 0.0f64;
    for i in 0..n {
        sx += x[i];
        sy += y[i];
    }
    let mx = sx / nf;
    let my = sy / nf;

    let mut cov = 0.0f64;
    let mut vx = 0.0f64;
    let mut vy = 0.0f64;
    for i in 0..n {
        let dx = x[i] - mx;
        let dy = y[i] - my;
        cov += dx * dy;
        vx += dx * dx;
        vy += dy * dy;
    }
    let denom = (vx * vy).sqrt();
    if denom > 0.0 {
        cov / denom
    } else {
        0.0
    }
}

// ============================ OLS 残差 ============================

/// OLS 残差：逐列回归 data[:, j] on target[:, j]（target 逐列不同，用于 orth2）
fn ols_resid_vector(data: &[f64], target: &[f64], rows: usize, n: usize) -> Vec<f64> {
    let mut resid = vec![0.0f64; rows * n];
    for j in 0..n {
        let mut d_mean = 0.0f64;
        let mut t_mean = 0.0f64;
        for t in 0..rows {
            d_mean += data[t * n + j];
            t_mean += target[t * n + j];
        }
        d_mean /= rows as f64;
        t_mean /= rows as f64;
        let mut t_ss = 0.0f64;
        let mut cov = 0.0f64;
        for t in 0..rows {
            let tc = target[t * n + j] - t_mean;
            let dc = data[t * n + j] - d_mean;
            t_ss += tc * tc;
            cov += tc * dc;
        }
        let beta = if t_ss > 1e-300 { cov / t_ss } else { 0.0 };
        let alpha = d_mean - beta * t_mean;
        for t in 0..rows {
            resid[t * n + j] = data[t * n + j] - alpha - beta * target[t * n + j];
        }
    }
    resid
}

/// OLS 残差：逐列回归 data[:, j] on 共享 target[t]（target 对所有股票相同，用于 orth1）
fn ols_resid_scalar(data: &[f64], target: &[f64], rows: usize, n: usize) -> Vec<f64> {
    let t_mean = target.iter().sum::<f64>() / rows as f64;
    let mut t_c = vec![0.0f64; rows];
    let mut t_ss = 0.0f64;
    for t in 0..rows {
        t_c[t] = target[t] - t_mean;
        t_ss += t_c[t] * t_c[t];
    }
    let inv_t_ss = if t_ss > 1e-300 { 1.0 / t_ss } else { 0.0 };
    let mut resid = vec![0.0f64; rows * n];
    for j in 0..n {
        let mut d_mean = 0.0f64;
        let mut dot_td = 0.0f64;
        for t in 0..rows {
            let v = data[t * n + j];
            d_mean += v;
            dot_td += t_c[t] * v;
        }
        d_mean /= rows as f64;
        let beta = dot_td * inv_t_ss;
        let alpha = d_mean - beta * t_mean;
        for t in 0..rows {
            resid[t * n + j] = data[t * n + j] - alpha - beta * target[t];
        }
    }
    resid
}

// ============================ 贡献度计算 ============================

/// 对一对相邻行 (x, y)，计算 4 种贡献度和 Pearson 相关系数
/// 返回 (m1, m2, m3, m4, r)，每个 Vec 长度 = n
fn compute_pair_contributions(
    x: &[f64],
    y: &[f64],
    n: usize,
) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, f64) {
    let nf = n as f64;

    let mut sum_x = 0.0f64;
    let mut sum_y = 0.0f64;
    for i in 0..n {
        sum_x += x[i];
        sum_y += y[i];
    }
    let mean_x = sum_x / nf;
    let mean_y = sum_y / nf;

    let mut sxx = 0.0f64;
    let mut syy = 0.0f64;
    let mut sxy = 0.0f64;
    for i in 0..n {
        let dx = x[i] - mean_x;
        let dy = y[i] - mean_y;
        sxx += dx * dx;
        syy += dy * dy;
        sxy += dx * dy;
    }

    let denom = (sxx * syy).sqrt();
    let r = if denom > 0.0 { sxy / denom } else { 0.0 };
    let b1 = if sxx > 0.0 { sxy / sxx } else { 0.0 };

    let mut m1 = vec![0.0f64; n];
    let mut m2 = vec![0.0f64; n];
    let mut m3 = vec![0.0f64; n];
    let mut m4 = vec![0.0f64; n];

    let factor = nf / (nf - 1.0);

    for i in 0..n {
        let dx = x[i] - mean_x;
        let dy = y[i] - mean_y;

        m1[i] = if denom > 0.0 { dx * dy / denom } else { 0.0 };

        let sxx_m = sxx - factor * dx * dx;
        let syy_m = syy - factor * dy * dy;
        let sxy_m = sxy - factor * dx * dy;
        let denom_m = (sxx_m * syy_m).sqrt();
        let r_m = if denom_m > 0.0 { sxy_m / denom_m } else { 0.0 };
        m2[i] = r - r_m;

        let zx = if sxx > 0.0 {
            dx * (nf / sxx).sqrt()
        } else {
            0.0
        };
        let zy = if syy > 0.0 {
            dy * (nf / syy).sqrt()
        } else {
            0.0
        };
        m3[i] = zx * zy - 0.5 * r * (zx * zx + zy * zy);

        let h = 1.0 / nf + dx * dx / sxx;
        let resid = dy - b1 * dx;
        let one_minus_h = 1.0 - h;
        m4[i] = if sxx > 0.0 && one_minus_h.abs() > 1e-300 {
            dx * resid / (sxx * one_minus_h)
        } else {
            0.0
        };
    }

    (m1, m2, m3, m4, r)
}

// ============================ 数组操作工具 ============================

/// 每行乘以 corr_seq[p]，得到逐元素乘积矩阵
fn element_wise_product(contrib: &[f64], corr_seq: &[f64], n: usize) -> Vec<f64> {
    let npairs = corr_seq.len();
    let mut result = vec![0.0f64; npairs * n];
    for p in 0..npairs {
        let c = corr_seq[p];
        let src = &contrib[p * n..(p + 1) * n];
        let dst = &mut result[p * n..(p + 1) * n];
        for j in 0..n {
            dst[j] = src[j] * c;
        }
    }
    result
}

/// 从行主序矩阵中选取指定行
fn select_rows(src: &[f64], indices: &[usize], n: usize) -> Vec<f64> {
    let k = indices.len();
    let mut result = vec![0.0f64; k * n];
    for (i, &idx) in indices.iter().enumerate() {
        result[i * n..(i + 1) * n].copy_from_slice(&src[idx * n..(idx + 1) * n]);
    }
    result
}

/// 两个行主序矩阵逐列 Pearson 相关
fn col_wise_corr(a: &[f64], b: &[f64], rows: usize, n: usize, parallel: bool) -> Vec<f64> {
    let compute = |j: usize| -> f64 {
        let mut col_a = vec![0.0f64; rows];
        let mut col_b = vec![0.0f64; rows];
        for t in 0..rows {
            col_a[t] = a[t * n + j];
            col_b[t] = b[t * n + j];
        }
        pearson_corr(&col_a, &col_b)
    };
    if parallel {
        (0..n).into_par_iter().map(compute).collect()
    } else {
        (0..n).map(compute).collect()
    }
}

/// 对行主序矩阵计算每列的 8 个统计量
fn compute_all_col_stats(data: &[f64], rows: usize, cols: usize, parallel: bool) -> Vec<[f64; 8]> {
    let compute = |j: usize| -> [f64; 8] {
        let mut col_buf = vec![0.0f64; rows];
        let mut scratch = Vec::with_capacity(rows);
        for t in 0..rows {
            col_buf[t] = data[t * cols + j];
        }
        compute_col_stats_scratch(&col_buf, &mut scratch)
    };
    if parallel {
        (0..cols).into_par_iter().map(compute).collect()
    } else {
        (0..cols).map(compute).collect()
    }
}

/// 计算相关系数矩阵 (cols × cols) 的每列统计量
/// GEMM 始终单线程（faer 已自带 SIMD），仅统计量部分按需并行
fn corr_matrix_col_stats(data: &[f64], rows: usize, cols: usize, parallel: bool) -> Vec<[f64; 8]> {
    use faer::{Mat, Parallelism};

    // Step 1: 逐列标准化
    let mut z = Mat::<f64>::zeros(rows, cols);
    for j in 0..cols {
        let mut mean = 0.0f64;
        for t in 0..rows {
            mean += data[t * cols + j];
        }
        mean /= rows as f64;
        let mut ss = 0.0f64;
        for t in 0..rows {
            let d = data[t * cols + j] - mean;
            ss += d * d;
        }
        let inv_std = if ss > 1e-300 {
            (rows as f64 / ss).sqrt()
        } else {
            0.0
        };
        for t in 0..rows {
            z[(t, j)] = (data[t * cols + j] - mean) * inv_std;
        }
    }

    // Step 2: 相关矩阵 = Z^T Z / rows，使用 faer GEMM
    let mut corr = Mat::<f64>::zeros(cols, cols);
    faer::linalg::matmul::matmul(
        &mut corr,
        z.as_ref().transpose(),
        z.as_ref(),
        None,
        1.0,
        if parallel {
            Parallelism::Rayon(0)
        } else {
            Parallelism::None
        },
    );

    // Step 3: 逐列统计量，复用 scratch
    let scale = 1.0 / rows as f64;
    let compute = |j: usize| -> [f64; 8] {
        let mut col_buf = vec![0.0f64; cols];
        let mut scratch = Vec::with_capacity(cols);
        for i in 0..cols {
            col_buf[i] = corr[(j, i)] * scale;
        }
        compute_col_stats_scratch(&col_buf, &mut scratch)
    };
    if parallel {
        (0..cols).into_par_iter().map(compute).collect()
    } else {
        (0..cols).map(compute).collect()
    }
}

// ============================ 硬拼凑统计 ============================

/// 单列硬拼凑计算（提取为独立函数，供 par_iter 复用）
fn hard_assemble_one_col(
    j: usize,
    contrib: &[f64],
    field: &[f64],
    npairs: usize,
    n: usize,
    kk: usize,
) -> (f64, [f64; 8], [f64; 8]) {
    let mut pairs: Vec<(f64, usize)> = vec![(0.0, 0); npairs];
    let mut contrib_vals = vec![0.0f64; kk];
    let mut field_vals = vec![0.0f64; kk];
    let mut indices = vec![0usize; kk];
    let mut scratch = Vec::with_capacity(kk);

    for p in 0..npairs {
        pairs[p] = (contrib[p * n + j], p);
    }
    pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

    for (i, &(val, p)) in pairs.iter().take(kk).enumerate() {
        contrib_vals[i] = val;
        field_vals[i] = field[p * n + j];
        indices[i] = p;
    }

    let ha_corr_j = pearson_corr(&contrib_vals[..kk], &field_vals[..kk]);

    let mut time_order: Vec<usize> = (0..kk).collect();
    time_order.sort_by(|&a, &b| indices[a].cmp(&indices[b]));
    let field_time: Vec<f64> = time_order.iter().map(|&i| field_vals[i]).collect();
    let time_j = compute_col_stats_scratch(&field_time, &mut scratch);

    let mut mag_order: Vec<usize> = (0..kk).collect();
    mag_order.sort_by(|&a, &b| {
        contrib_vals[a]
            .abs()
            .partial_cmp(&contrib_vals[b].abs())
            .unwrap()
            .reverse()
    });
    let field_mag: Vec<f64> = mag_order.iter().map(|&i| field_vals[i]).collect();
    let mag_j = compute_col_stats_scratch(&field_mag, &mut scratch);

    (ha_corr_j, time_j, mag_j)
}

/// 对 (npairs × n) 贡献度矩阵，逐列选取最大的 k 个值
fn hard_assemble_stats(
    contrib: &[f64],
    field: &[f64],
    npairs: usize,
    n: usize,
    k: usize,
    parallel: bool,
) -> (Vec<f64>, Vec<[f64; 8]>, Vec<[f64; 8]>) {
    let kk = k.min(npairs);
    let results: Vec<(f64, [f64; 8], [f64; 8])> = if parallel {
        (0..n)
            .into_par_iter()
            .map(|j| hard_assemble_one_col(j, contrib, field, npairs, n, kk))
            .collect()
    } else {
        (0..n)
            .map(|j| hard_assemble_one_col(j, contrib, field, npairs, n, kk))
            .collect()
    };

    let mut ha_corr = vec![0.0f64; n];
    let mut time_stats = vec![[0.0f64; 8]; n];
    let mut mag_stats = vec![[0.0f64; 8]; n];
    for (j, (corr, time, mag)) in results.into_iter().enumerate() {
        ha_corr[j] = corr;
        time_stats[j] = time;
        mag_stats[j] = mag;
    }
    (ha_corr, time_stats, mag_stats)
}

// ============================ 主计算 ============================

/// 处理单个矩阵：列统计 + 相关矩阵列统计 + field 相关
fn process_array(
    data: &[f64],
    rows: usize,
    n: usize,
    field_corr: &[f64],
    method: &str,
    type_name: &str,
    parallel: bool,
    factors: &mut Vec<(String, Vec<f64>)>,
) {
    let cs = compute_all_col_stats(data, rows, n, parallel);
    for si in 0..8 {
        let vals: Vec<f64> = cs.iter().map(|s| s[si]).collect();
        factors.push((format!("{}_{}_{}", method, type_name, STAT_NAMES[si]), vals));
    }

    let cms = corr_matrix_col_stats(data, rows, n, parallel);
    for si in 0..8 {
        let vals: Vec<f64> = cms.iter().map(|s| s[si]).collect();
        factors.push((
            format!("{}_{}_cm_{}", method, type_name, STAT_NAMES[si]),
            vals,
        ));
    }

    let fc = col_wise_corr(data, field_corr, rows, n, parallel);
    factors.push((format!("{}_{}_fcorr", method, type_name), fc));
}

/// 处理残差矩阵：列统计(7) + 相关矩阵统计(8) + (仅 full) top/bot 提取统计
fn process_residual(
    resid: &[f64],
    rows: usize,
    n: usize,
    prefix: &str,
    type_name: &str,
    top_idx: &[usize],
    bot_idx: &[usize],
    parallel: bool,
    factors: &mut Vec<(String, Vec<f64>)>,
) {
    let cs = compute_all_col_stats(resid, rows, n, parallel);
    for si in 1..8 {
        let vals: Vec<f64> = cs.iter().map(|s| s[si]).collect();
        factors.push((format!("{}_{}_{}", prefix, type_name, STAT_NAMES[si]), vals));
    }

    let cms = corr_matrix_col_stats(resid, rows, n, parallel);
    for si in 0..8 {
        let vals: Vec<f64> = cms.iter().map(|s| s[si]).collect();
        factors.push((
            format!("{}_{}_cm_{}", prefix, type_name, STAT_NAMES[si]),
            vals,
        ));
    }

    if !top_idx.is_empty() {
        let kk = top_idx.len();
        for (idx, name) in [(top_idx, "top"), (bot_idx, "bot")] {
            let subset = select_rows(resid, idx, n);
            let cs = compute_all_col_stats(&subset, kk, n, parallel);
            for si in 0..8 {
                let vals: Vec<f64> = cs.iter().map(|s| s[si]).collect();
                factors.push((format!("{}_full_{}_{}", prefix, name, STAT_NAMES[si]), vals));
            }
            let cms = corr_matrix_col_stats(&subset, kk, n, parallel);
            for si in 0..8 {
                let vals: Vec<f64> = cms.iter().map(|s| s[si]).collect();
                factors.push((
                    format!("{}_full_{}_cm_{}", prefix, name, STAT_NAMES[si]),
                    vals,
                ));
            }
        }
    }
}

fn compute_all_factors(
    field: Array2<f64>,
    top_k: usize,
    parallel: bool,
) -> Vec<(String, Vec<f64>)> {
    let t = field.nrows();
    let n = field.ncols();
    assert!(t >= 2, "field must have at least 2 rows");
    let npairs = t - 1;
    let k = top_k.min(npairs);

    let field_data: Vec<f64> = field.as_slice().unwrap().to_vec();

    // ===== Step 1: 相邻相关系数 & 4 种贡献度矩阵 =====
    let mut corr_seq = vec![0.0f64; npairs];
    let mut m_full: [Vec<f64>; 4] = [
        vec![0.0; npairs * n],
        vec![0.0; npairs * n],
        vec![0.0; npairs * n],
        vec![0.0; npairs * n],
    ];

    for p in 0..npairs {
        let x = &field_data[p * n..(p + 1) * n];
        let y = &field_data[(p + 1) * n..(p + 2) * n];
        let (c1, c2, c3, c4, r) = compute_pair_contributions(x, y, n);
        corr_seq[p] = r;
        let off = p * n;
        m_full[0][off..off + n].copy_from_slice(&c1);
        m_full[1][off..off + n].copy_from_slice(&c2);
        m_full[2][off..off + n].copy_from_slice(&c3);
        m_full[3][off..off + n].copy_from_slice(&c4);
    }

    // ===== Step 2: 按 corr 排序选取 top/bot =====
    let mut sorted_idx: Vec<usize> = (0..npairs).collect();
    sorted_idx.sort_by(|&a, &b| corr_seq[a].partial_cmp(&corr_seq[b]).unwrap());
    let bot_idx: Vec<usize> = sorted_idx[..k].to_vec();
    let top_idx: Vec<usize> = sorted_idx[npairs - k..].to_vec();

    // ===== Step 3: 衍生矩阵 =====
    let m_prod: [Vec<f64>; 4] = [
        element_wise_product(&m_full[0], &corr_seq, n),
        element_wise_product(&m_full[1], &corr_seq, n),
        element_wise_product(&m_full[2], &corr_seq, n),
        element_wise_product(&m_full[3], &corr_seq, n),
    ];
    let m_top: [Vec<f64>; 4] = [
        select_rows(&m_full[0], &top_idx, n),
        select_rows(&m_full[1], &top_idx, n),
        select_rows(&m_full[2], &top_idx, n),
        select_rows(&m_full[3], &top_idx, n),
    ];
    let m_bot: [Vec<f64>; 4] = [
        select_rows(&m_full[0], &bot_idx, n),
        select_rows(&m_full[1], &bot_idx, n),
        select_rows(&m_full[2], &bot_idx, n),
        select_rows(&m_full[3], &bot_idx, n),
    ];

    let field_full: Vec<f64> = field_data[..npairs * n].to_vec();
    let field_top: Vec<f64> = select_rows(&field_data, &top_idx, n);
    let field_bot: Vec<f64> = select_rows(&field_data, &bot_idx, n);
    let corr_top: Vec<f64> = top_idx.iter().map(|&i| corr_seq[i]).collect();
    let corr_bot: Vec<f64> = bot_idx.iter().map(|&i| corr_seq[i]).collect();

    let mut factors: Vec<(String, Vec<f64>)> = Vec::new();

    for mi in 0..4 {
        let method = METHOD_NAMES[mi];

        process_array(
            &m_full[mi],
            npairs,
            n,
            &field_full,
            method,
            "full",
            parallel,
            &mut factors,
        );
        process_array(
            &m_prod[mi],
            npairs,
            n,
            &field_full,
            method,
            "prod",
            parallel,
            &mut factors,
        );
        process_array(
            &m_top[mi],
            k,
            n,
            &field_top,
            method,
            "top",
            parallel,
            &mut factors,
        );
        process_array(
            &m_bot[mi],
            k,
            n,
            &field_bot,
            method,
            "bot",
            parallel,
            &mut factors,
        );

        for (data, type_name) in [(&m_full[mi][..], "full"), (&m_prod[mi][..], "prod")] {
            let (ha_corr, ha_time, ha_mag) =
                hard_assemble_stats(data, &field_data, npairs, n, k, parallel);
            factors.push((format!("{}_{}_ha_corr", method, type_name), ha_corr));
            for si in 0..8 {
                let time_vals: Vec<f64> = ha_time.iter().map(|s| s[si]).collect();
                let mag_vals: Vec<f64> = ha_mag.iter().map(|s| s[si]).collect();
                factors.push((
                    format!("{}_{}_ha_time_{}", method, type_name, STAT_NAMES[si]),
                    time_vals,
                ));
                factors.push((
                    format!("{}_{}_ha_mag_{}", method, type_name, STAT_NAMES[si]),
                    mag_vals,
                ));
            }
        }
    }

    // ===== Step 8: 正交化因子 =====
    {
        let r1_full = ols_resid_scalar(&field_full, &corr_seq, npairs, n);
        let r1_top = ols_resid_scalar(&field_top, &corr_top, k, n);
        let r1_bot = ols_resid_scalar(&field_bot, &corr_bot, k, n);
        process_residual(
            &r1_full,
            npairs,
            n,
            "orth1",
            "full",
            &top_idx,
            &bot_idx,
            parallel,
            &mut factors,
        );
        process_residual(
            &r1_top,
            k,
            n,
            "orth1",
            "top",
            &[],
            &[],
            parallel,
            &mut factors,
        );
        process_residual(
            &r1_bot,
            k,
            n,
            "orth1",
            "bot",
            &[],
            &[],
            parallel,
            &mut factors,
        );
    }

    for mi in 0..4 {
        let prefix = format!("{}_orth2", METHOD_NAMES[mi]);
        let target_top = select_rows(&m_prod[mi], &top_idx, n);
        let target_bot = select_rows(&m_prod[mi], &bot_idx, n);
        let r2_full = ols_resid_vector(&field_full, &m_prod[mi], npairs, n);
        let r2_top = ols_resid_vector(&field_top, &target_top, k, n);
        let r2_bot = ols_resid_vector(&field_bot, &target_bot, k, n);
        process_residual(
            &r2_full,
            npairs,
            n,
            &prefix,
            "full",
            &top_idx,
            &bot_idx,
            parallel,
            &mut factors,
        );
        process_residual(
            &r2_top,
            k,
            n,
            &prefix,
            "top",
            &[],
            &[],
            parallel,
            &mut factors,
        );
        process_residual(
            &r2_bot,
            k,
            n,
            &prefix,
            "bot",
            &[],
            &[],
            parallel,
            &mut factors,
        );
    }

    factors
}

// ============================ PyO3 接口 ============================

/// 单线程或并行（rayon，最多 30 线程）执行
fn run_with_pool<F, R>(parallel: bool, f: F) -> R
where
    F: FnOnce(bool) -> R + Send,
    R: Send,
{
    if parallel {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(18)
            .build()
            .unwrap();
        pool.install(|| f(true))
    } else {
        f(false)
    }
}

/// 计算相邻分钟截面相关系数的逐股贡献度因子
///
/// 参数
/// -----
/// field : np.ndarray (T, N)
/// top_k : int, 默认 300
/// parallel : bool, 默认 False
///     True 时启用 rayon 并行（最多 30 线程），适合单日快速更新
///
/// 返回
/// -----
/// dict[str, np.ndarray]  — 793 个因子
#[pyfunction]
#[pyo3(signature = (field, top_k=300, parallel=false))]
pub fn compute_corr_contribution_factors(
    py: Python,
    field: PyReadonlyArray2<f64>,
    top_k: usize,
    parallel: bool,
) -> PyResult<PyObject> {
    let field_arr = field.as_array().to_owned();
    let factors =
        py.allow_threads(|| run_with_pool(parallel, |p| compute_all_factors(field_arr, top_k, p)));

    let dict = PyDict::new(py);
    for (name, values) in factors {
        let arr = PyArray1::from_vec(py, values);
        dict.set_item(name, arr)?;
    }
    Ok(dict.into())
}

/// 批量计算多个字段的相邻分钟截面相关系数贡献度因子
///
/// 参数
/// -----
/// field_stack : np.ndarray (T, N, K)
/// field_names : list[str], 长度 K
/// top_k : int, 默认 300
/// parallel : bool, 默认 False
///     True 时启用 rayon 并行（最多 30 线程），适合单日快速更新
///
/// 返回
/// -----
/// dict[str, np.ndarray]  — K × 793 个因子
#[pyfunction]
#[pyo3(signature = (field_stack, field_names, top_k=300, parallel=false))]
pub fn compute_corr_contribution_multi(
    py: Python,
    field_stack: PyReadonlyArray3<f64>,
    field_names: Vec<String>,
    top_k: usize,
    parallel: bool,
) -> PyResult<PyObject> {
    let stack = field_stack.as_array();
    let (_t, _n, k) = stack.dim();
    assert_eq!(field_names.len(), k, "field_names length must match depth");

    // 先在 Rust 侧算完所有因子（不持 GIL），再回 Python 写 dict
    let all_factors: Vec<(String, Vec<(String, Vec<f64>)>)> = py.allow_threads(|| {
        run_with_pool(parallel, |par| {
            (0..k)
                .map(|i| {
                    let field_arr = stack.slice(s![.., .., i]).to_owned();
                    let prefix = field_names[i].clone();
                    let facs = compute_all_factors(field_arr, top_k, par);
                    (prefix, facs)
                })
                .collect()
        })
    });

    let dict = PyDict::new(py);

    for (prefix, facs) in all_factors {
        for (name, vals) in facs {
            let arr = PyArray1::from_vec(py, vals);
            dict.set_item(format!("{}_{}", prefix, name), arr)?;
        }
    }
    Ok(dict.into())
}
