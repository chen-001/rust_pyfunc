use numpy::{PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;

/// 自适应排序：小数组插入排序，大数组标准排序
fn sort_small(arr: &mut [f64]) {
    if arr.len() <= 32 {
        for i in 1..arr.len() {
            let key = arr[i];
            let mut j = i;
            while j > 0 && arr[j - 1] > key {
                arr[j] = arr[j - 1];
                j -= 1;
            }
            arr[j] = key;
        }
    } else {
        arr.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    }
}

/// 排序后线性插值 percentile
fn interp_pct(sorted: &[f64], pct: f64) -> f64 {
    let n = sorted.len() as f64;
    if n <= 1.0 { return sorted[0]; }
    let idx = pct / 100.0 * (n - 1.0);
    let lo = idx.floor() as usize;
    let hi = (lo + 1).min(sorted.len() - 1);
    let frac = idx - lo as f64;
    sorted[lo] * (1.0 - frac) + sorted[hi] * frac
}

/// percentile（拷贝后排序）
fn percentile(data: &[f64], pct: f64) -> f64 {
    let mut s = data.to_vec();
    sort_small(&mut s);
    interp_pct(&s, pct)
}

/// 两个向量的 pearson 相关系数
fn pearson_corr(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len();
    if n == 0 { return 0.0; }
    let xm = x.iter().sum::<f64>() / n as f64;
    let ym = y.iter().sum::<f64>() / n as f64;
    let mut xv = 0.0f64; let mut yv = 0.0f64; let mut cov = 0.0f64;
    for i in 0..n {
        let dx = x[i] - xm; let dy = y[i] - ym;
        xv += dx * dx; yv += dy * dy; cov += dx * dy;
    }
    let xs = xv.sqrt(); let ys = yv.sqrt();
    if xs > 0.0 && ys > 0.0 { cov / (xs * ys) } else { 0.0 }
}

/// 逐列相关系数（flat row-major 数组，shape (nrows, ncols)）
fn col_corr_flat(a: &[f64], b: &[f64], nrows: usize, ncols: usize) -> Vec<f64> {
    let mut res = vec![0.0; ncols];
    let nf = nrows as f64;
    for j in 0..ncols {
        let mut am = 0.0f64; let mut bm = 0.0f64;
        for i in 0..nrows { am += a[i * ncols + j]; bm += b[i * ncols + j]; }
        am /= nf; bm /= nf;
        let mut av = 0.0f64; let mut bv = 0.0f64; let mut cov = 0.0f64;
        for i in 0..nrows {
            let da = a[i * ncols + j] - am; let db = b[i * ncols + j] - bm;
            av += da * da; bv += db * db; cov += da * db;
        }
        let asd = (av / nf).sqrt(); let bsd = (bv / nf).sqrt();
        if asd > 0.0 && bsd > 0.0 { res[j] = (cov / nf) / (asd * bsd); }
    }
    res
}

#[pyfunction]
#[pyo3(signature = (large_corr, small_corr, ratio))]
pub fn compute_corr_diff_features(
    large_corr: PyReadonlyArray2<f64>,
    small_corr: PyReadonlyArray2<f64>,
    ratio: usize,
) -> PyResult<(Vec<f64>, Py<PyArray2<f64>>)> {
    let large = large_corr.as_array();
    let small = small_corr.as_array();
    let n = small.dim().0;
    let rr = ratio * ratio;
    let rr_f = rr as f64;
    let total = n * n;
    let total_f = total as f64;

    // 原始 slice 直接访问（避免 ndarray 边界检查开销）
    let large_s = large.as_slice().expect("large_corr must be C-contiguous");
    let large_nc = large.ncols();
    let small_s = small.as_slice().expect("small_corr must be C-contiguous");
    // small 是 n×n，所以 small_nc == n

    // ========== 一次性分配所有 flat buffer ==========
    let mut b_sum = vec![0.0f64; total];
    let mut b_sq = vec![0.0f64; total];     // sum of squares（用于方差）
    let mut b_cu = vec![0.0f64; total];     // sum of cubes（用于偏态）
    let mut b_mx = vec![f64::NEG_INFINITY; total];
    let mut b_mn = vec![f64::INFINITY; total];
    let mut b_amx = vec![0.0f64; total];    // abs max
    let mut b_sgn = vec![0.0f64; total];    // sign sum
    let mut b_dia = vec![0.0f64; total * ratio]; // diags[i*n*ratio + j*ratio + k]
    let mut b_iqr = vec![0.0f64; total];
    let mut b_peak = vec![0u32; total];     // z-score > 1 计数

    // 按块行处理：只保留当前块行的值，减少内存占用
    let mut row_buf = vec![0.0f64; n * rr]; // 每个 block col 的 rr 个值
    let mut buf_pos = vec![0usize; n];       // 每个 block col 的写入位置

    for bi in 0..n {
        // 重置写入位置
        for bj in 0..n { buf_pos[bj] = 0; }

        // 遍历块行内的 ratio 行（缓存友好：逐行顺序读取 large 矩阵）
        for a in 0..ratio {
            let row = bi * ratio + a;
            let row_off = row * large_nc;
            for bj in 0..n {
                let col_off = bj * ratio;
                let idx = bi * n + bj;
                for b in 0..ratio {
                    let v = unsafe { *large_s.get_unchecked(row_off + col_off + b) };

                    b_sum[idx] += v;
                    b_sq[idx] += v * v;
                    b_cu[idx] += v * v * v;
                    if v > b_mx[idx] { b_mx[idx] = v; }
                    if v < b_mn[idx] { b_mn[idx] = v; }
                    let av = v.abs();
                    if av > b_amx[idx] { b_amx[idx] = av; }
                    if v > 0.0 { b_sgn[idx] += 1.0; }
                    else if v < 0.0 { b_sgn[idx] -= 1.0; }

                    if a == b {
                        b_dia[idx * ratio + a] = v;
                    }

                    row_buf[bj * rr + buf_pos[bj]] = v;
                    buf_pos[bj] += 1;
                }
            }
        }

        // 对当前块行的每个 block col：计算 z-score peak 和 percentile
        for bj in 0..n {
            let idx = bi * n + bj;
            let mean = b_sum[idx] / rr_f;
            let variance = b_sq[idx] / rr_f - mean * mean;
            let std = if variance > 0.0 { variance.sqrt() } else { 0.0 };

            // z-score peak
            if std > 1e-10 {
                let thr = mean + std;
                let off = bj * rr;
                for k in 0..rr {
                    if unsafe { *row_buf.get_unchecked(off + k) } > thr {
                        b_peak[idx] += 1;
                    }
                }
            }

            // percentile（自适应排序后取 p25, p75）
            let sl = &mut row_buf[bj * rr..(bj + 1) * rr];
            sort_small(sl);
            b_iqr[idx] = interp_pct(sl, 75.0) - interp_pct(sl, 25.0);
        }
    }

    // ========== 计算 mean, std, skew ==========
    let mut b_mean = vec![0.0f64; total];
    let mut b_std = vec![0.0f64; total];
    let mut b_skew = vec![0.0f64; total];

    for idx in 0..total {
        let mean = b_sum[idx] / rr_f;
        b_mean[idx] = mean;
        let variance = b_sq[idx] / rr_f - mean * mean;
        b_std[idx] = if variance > 0.0 { variance.sqrt() } else { 0.0 };
        // skew: E[(X-μ)³] = E[X³] - 3μE[X²] + 2μ³
        let ex2 = b_sq[idx] / rr_f;
        let ex3 = b_cu[idx] / rr_f;
        let m3 = ex3 - 3.0 * mean * ex2 + 2.0 * mean * mean * mean;
        let s = b_std[idx];
        if s > 1e-10 {
            b_skew[idx] = m3 / (s * s * s);
        }
    }

    // ========== diff 及派生（单遍计算）==========
    let mut diff = vec![0.0f64; total];
    let mut diff_map = vec![0.0f64; total];
    let mut abs_small = vec![0.0f64; total];
    let mut b_range = vec![0.0f64; total];
    let mut sign_maj = vec![0.0f64; total];

    let mut diff_sum = 0.0f64;
    let mut diff_sq_sum = 0.0f64;
    let mut diff_max = 0.0f64;
    let mut diff_high_cnt = 0usize;
    let mut diff_low_cnt = 0usize;
    let mut std_sum = 0.0f64;
    let mut std_max = 0.0f64;
    let mut range_sum = 0.0f64;
    let mut range_max = 0.0f64;
    let mut iqr_sum = 0.0f64;
    let mut pos_ext_cnt = 0usize;
    let mut neg_ext_cnt = 0usize;
    let mut ext_asym_cnt = 0usize;
    let mut ext_ratio_sum = 0.0f64;
    let mut sgn_maj_sum = 0.0f64;
    let mut sgn_conflict_cnt = 0usize;
    let mut diff_vol_prod = 0.0f64;
    let mut pos_bias_cnt = 0usize;
    let mut neg_bias_cnt = 0usize;
    let mut pos_abs_sum = 0.0f64;
    let mut neg_abs_sum = 0.0f64;
    let mut edge_diff = 0.0f64;
    let mut edge_cnt = 0usize;
    let mut center_diff = 0.0f64;
    let mut center_cnt = 0usize;
    let mut nd_abs_sum = 0.0f64;
    let mut nd_sum = 0.0f64;
    let mut nd_ext_cnt = 0usize;

    let edge_size = n / 8;

    for i in 0..n {
        let small_row = i * n; // small_nc == n
        let is_edge_i = i < edge_size || i >= n - edge_size;
        for j in 0..n {
            let idx = i * n + j;
            let sv = small_s[small_row + j];
            let bm = b_mean[idx];
            let d = bm - sv;
            let dm = d.abs();
            let as_ = sv.abs();

            diff[idx] = d;
            diff_map[idx] = dm;
            abs_small[idx] = as_;
            b_range[idx] = b_mx[idx] - b_mn[idx];
            sign_maj[idx] = b_sgn[idx].abs() / rr_f;

            // 累积标量统计
            diff_sum += dm;
            diff_sq_sum += dm * dm;
            if dm > diff_max { diff_max = dm; }
            if dm > 0.1 { diff_high_cnt += 1; }
            if dm < 0.01 { diff_low_cnt += 1; }
            std_sum += b_std[idx];
            if b_std[idx] > std_max { std_max = b_std[idx]; }
            range_sum += b_range[idx];
            if b_range[idx] > range_max { range_max = b_range[idx]; }
            iqr_sum += b_iqr[idx];
            if b_mx[idx].abs() > 0.5 { pos_ext_cnt += 1; }
            if b_mn[idx].abs() > 0.5 { neg_ext_cnt += 1; }
            if b_mx[idx].abs() > b_mn[idx].abs() { ext_asym_cnt += 1; }
            ext_ratio_sum += b_mx[idx].abs() / (b_mn[idx].abs() + 1e-8);
            sgn_maj_sum += sign_maj[idx];
            if sign_maj[idx] < 0.6 { sgn_conflict_cnt += 1; }
            diff_vol_prod += b_std[idx] * dm;
            if d > 0.0 { pos_bias_cnt += 1; pos_abs_sum += d; }
            if d < 0.0 { neg_bias_cnt += 1; neg_abs_sum += d.abs(); }

            let is_edge = is_edge_i || j < edge_size || j >= n - edge_size;
            if is_edge { edge_diff += dm; edge_cnt += 1; }
            else { center_diff += dm; center_cnt += 1; }

            let nd = d / (as_ + 0.1);
            nd_abs_sum += nd.abs();
            nd_sum += nd;
            if nd.abs() > 0.5 { nd_ext_cnt += 1; }
        }
    }

    // trends
    let x_mean = (ratio - 1) as f64 / 2.0;
    let x_var: f64 = (0..ratio).map(|k| { let d = k as f64 - x_mean; d * d }).sum::<f64>() / ratio as f64;
    let x_std = x_var.sqrt();
    let mut trends = vec![0.0f64; total];
    let mut trend_sum = 0.0f64;
    let mut trend_sq = 0.0f64;
    let mut trend_pos = 0usize;
    let mut trend_neg = 0usize;

    for i in 0..n {
        for j in 0..n {
            let idx = i * n + j;
            let d_off = idx * ratio;
            let dm: f64 = (0..ratio).map(|k| b_dia[d_off + k]).sum::<f64>() / ratio as f64;
            let dv: f64 = (0..ratio).map(|k| { let d = b_dia[d_off + k] - dm; d * d }).sum::<f64>() / ratio as f64;
            let ds = dv.sqrt();
            if ds > 0.0 && x_std > 0.0 {
                let cov: f64 = (0..ratio).map(|k| (b_dia[d_off + k] - dm) * (k as f64 - x_mean)).sum::<f64>() / ratio as f64;
                trends[idx] = cov / (ds * x_std);
            }
            trend_sum += trends[idx];
        }
    }
    let trend_mean = trend_sum / total_f;
    for idx in 0..total {
        let d = trends[idx] - trend_mean;
        trend_sq += d * d;
        if trends[idx] > 0.2 { trend_pos += 1; }
        if trends[idx] < -0.2 { trend_neg += 1; }
    }
    let trend_std = (trend_sq / total_f).sqrt();

    // pearson corr
    let corr_dv = pearson_corr(&diff_map, &b_std);
    let corr_dr = pearson_corr(&diff_map, &b_range);

    let edge_m = if edge_cnt > 0 { edge_diff / edge_cnt as f64 } else { 0.0 };
    let center_m = if center_cnt > 0 { center_diff / center_cnt as f64 } else { 0.0 };
    let nd_mean = nd_sum / total_f;
    let mut nd_var = 0.0f64;
    for idx in 0..total {
        let nd = diff[idx] / (abs_small[idx] + 0.1);
        let d = nd - nd_mean;
        nd_var += d * d;
    }

    let scalar_features = vec![
        diff_sum / total_f, (diff_sq_sum / total_f - (diff_sum / total_f).powi(2)).sqrt(), diff_max,
        diff_high_cnt as f64 / total_f, diff_low_cnt as f64 / total_f,
        std_sum / total_f, std_max, range_sum / total_f, range_max, iqr_sum / total_f,
        pos_ext_cnt as f64 / total_f, neg_ext_cnt as f64 / total_f,
        ext_asym_cnt as f64 / total_f, ext_ratio_sum / total_f,
        trend_mean, trend_std, trend_pos as f64 / total_f, trend_neg as f64 / total_f,
        1.0, sgn_maj_sum / total_f, sgn_conflict_cnt as f64 / total_f,
        corr_dv, corr_dr, diff_vol_prod / total_f,
        pos_bias_cnt as f64 / total_f, neg_bias_cnt as f64 / total_f,
        pos_bias_cnt as f64 / total_f - neg_bias_cnt as f64 / total_f,
        if pos_bias_cnt > 0 && neg_bias_cnt > 0 {
            pos_abs_sum / pos_bias_cnt as f64 - neg_abs_sum / neg_bias_cnt as f64
        } else { 0.0 },
        0.0, 0.0, 0.0,
        edge_m, center_m, edge_m / (center_m + 1e-8),
        nd_abs_sum / total_f, (nd_var / total_f).sqrt(), nd_ext_cnt as f64 / total_f,
    ];

    // ========== 序列特征 (n, 16) ==========
    let abs_skew: Vec<f64> = b_skew.iter().map(|v| v.abs()).collect();
    let thr_skew = percentile(&abs_skew, 95.0);
    let thr_max = percentile(&b_amx, 95.0);

    // 扁平输出 buffer，row-major layout: series[i * 16 + col]
    let mut series = vec![0.0f64; n * 16];

    // 0: col_corr(b_mean, small)
    let cc0 = col_corr_flat(&b_mean, small_s, n, n);
    for j in 0..n { series[j * 16] = cc0[j]; }

    // 1: abs(b_mean - small).mean(axis=0)
    for j in 0..n {
        let mut s = 0.0f64;
        for i in 0..n { s += (b_mean[i * n + j] - small_s[i * n + j]).abs(); }
        series[j * 16 + 1] = s / n as f64;
    }

    // 2: top_skew_block_col_count
    for j in 0..n {
        let mut c = 0.0f64;
        for i in 0..n { if b_skew[i * n + j].abs() >= thr_skew { c += 1.0; } }
        series[j * 16 + 2] = c;
    }

    // 3: col_corr(b_std, small)
    let cc3 = col_corr_flat(&b_std, small_s, n, n);
    for j in 0..n { series[j * 16 + 3] = cc3[j]; }

    // 4: diag_align_abs_diff
    for i in 0..n {
        let idx = i * n + i;
        let dm: f64 = (0..ratio).map(|k| b_dia[idx * ratio + k]).sum::<f64>() / ratio as f64;
        series[i * 16 + 4] = (dm - small_s[i * n + i]).abs();
    }

    // 5: diag_align_sign_match
    for i in 0..n {
        let idx = i * n + i;
        let dm: f64 = (0..ratio).map(|k| b_dia[idx * ratio + k]).sum::<f64>() / ratio as f64;
        let sv = small_s[i * n + i];
        series[i * 16 + 5] = if dm.signum() == sv.signum() { 1.0 } else { 0.0 };
    }

    // 6: anti_diag_abs_diff
    for i in 0..n {
        let aj = n - 1 - i;
        let idx = i * n + aj;
        let dm: f64 = (0..ratio).map(|k| b_dia[idx * ratio + k]).sum::<f64>() / ratio as f64;
        series[i * 16 + 6] = (dm - small_s[i * n + aj]).abs();
    }

    // 7: col_sparse_ratio_005
    for j in 0..n {
        let c = (0..n).filter(|&i| small_s[i * n + j].abs() < 0.05).count();
        series[j * 16 + 7] = c as f64 / n as f64;
    }

    // 8: col_dense_ratio_03
    for j in 0..n {
        let c = (0..n).filter(|&i| small_s[i * n + j].abs() > 0.3).count();
        series[j * 16 + 8] = c as f64 / n as f64;
    }

    // 9: block_abs_max_col_std
    for j in 0..n {
        let m: f64 = (0..n).map(|i| b_amx[i * n + j]).sum::<f64>() / n as f64;
        let v: f64 = (0..n).map(|i| { let d = b_amx[i * n + j] - m; d * d }).sum::<f64>() / n as f64;
        series[j * 16 + 9] = v.sqrt();
    }

    // 10: top_block_absmax_col_mean
    for j in 0..n {
        let mut s = 0.0f64; let mut c = 0usize;
        for i in 0..n {
            if b_amx[i * n + j] >= thr_max { s += small_s[i * n + j]; c += 1; }
        }
        series[j * 16 + 10] = if c > 0 { s / c as f64 } else { 0.0 };
    }

    // 11: top_block_absmax_col_count
    for j in 0..n {
        series[j * 16 + 11] = (0..n).filter(|&i| b_amx[i * n + j] >= thr_max).count() as f64;
    }

    // 12: col_iqr（排序一次取 p75, p25）
    let mut col_buf = vec![0.0f64; n];
    for j in 0..n {
        for i in 0..n { col_buf[i] = small_s[i * n + j]; }
        sort_small(&mut col_buf[..n]);
        series[j * 16 + 12] = interp_pct(&col_buf[..n], 75.0) - interp_pct(&col_buf[..n], 25.0);
    }

    // 13: block_rank_vs_small_corr = 0（block_rank 恒为常数 (rr-1)/2，相关系数为 0）
    // 已初始化为 0

    // 14: block_peak_ratio_col_mean
    for j in 0..n {
        let mut s = 0.0f64;
        for i in 0..n { s += b_peak[i * n + j] as f64 / rr_f; }
        series[j * 16 + 14] = s / n as f64;
    }

    // 15: col_segment_trend
    let seg = ratio.max(2);
    let seg_size = n / seg;
    if seg_size > 0 {
        let idx_a: Vec<f64> = (0..seg).map(|s| s as f64).collect();
        let im = idx_a.iter().sum::<f64>() / seg as f64;
        let is_ = {
            let mut s = 0.0f64;
            for &v in &idx_a { let d = v - im; s += d * d; }
            (s / seg as f64).sqrt()
        };
        if is_ > 0.0 {
            for j in 0..n {
                let mut parts = vec![0.0f64; seg];
                for s in 0..seg {
                    let mut sum = 0.0f64;
                    for r in 0..seg_size { sum += small_s[(s * seg_size + r) * n + j]; }
                    parts[s] = sum / seg_size as f64;
                }
                let pm: f64 = parts.iter().sum::<f64>() / seg as f64;
                let pv: f64 = parts.iter().map(|&v| { let d = v - pm; d * d }).sum::<f64>() / seg as f64;
                let ps = pv.sqrt();
                if ps > 0.0 {
                    let cov: f64 = (0..seg).map(|s| (parts[s] - pm) * (idx_a[s] - im)).sum::<f64>() / seg as f64;
                    series[j * 16 + 15] = cov / (ps * is_);
                }
            }
        }
    }

    // 转换输出
    let mut arr = ndarray::Array2::zeros((n, 16));
    for i in 0..n {
        for j in 0..16 {
            arr[[i, j]] = series[i * 16 + j];
        }
    }
    let py_out = Python::with_gil(|py| PyArray2::from_array(py, &arr).to_owned());
    Ok((scalar_features, py_out))
}
