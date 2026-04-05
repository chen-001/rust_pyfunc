use numpy::{IntoPyArray, PyArray2, PyReadonlyArray1};
use pyo3::prelude::*;

/// 3x3 矩阵解析求逆（克莱姆法则）
fn invert3(m: &[[f64; 3]; 3]) -> Option<[[f64; 3]; 3]> {
    let (a, b, c, d, e, f, g, h, i) = (
        m[0][0], m[0][1], m[0][2], m[1][0], m[1][1], m[1][2], m[2][0], m[2][1], m[2][2],
    );
    let det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
    if det.abs() < 1e-15 {
        return None;
    }
    let inv_det = 1.0 / det;
    Some([
        [
            (e * i - f * h) * inv_det,
            (c * h - b * i) * inv_det,
            (b * f - c * e) * inv_det,
        ],
        [
            (f * g - d * i) * inv_det,
            (a * i - c * g) * inv_det,
            (c * d - a * f) * inv_det,
        ],
        [
            (d * h - e * g) * inv_det,
            (b * g - a * h) * inv_det,
            (a * e - b * d) * inv_det,
        ],
    ])
}

/// 提取对角矩阵并求逆
fn invert_diag3(m: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    [
        [1.0 / (m[0][0] + 1e-15), 0.0, 0.0],
        [0.0, 1.0 / (m[1][1] + 1e-15), 0.0],
        [0.0, 0.0, 1.0 / (m[2][2] + 1e-15)],
    ]
}

/// 二次型 d^T * inv * d
fn quadratic(d: &[f64; 3], inv: &[[f64; 3]; 3]) -> f64 {
    let mut s = 0.0f64;
    for i in 0..3 {
        let mut row_s = 0.0f64;
        for j in 0..3 {
            row_s += inv[i][j] * d[j];
        }
        s += d[i] * row_s;
    }
    s
}

/// 单个组合的计算结果
struct ComboResult {
    factor_m2: Vec<f64>,
    full_m2: Vec<f64>,
    factor_m3: Vec<f64>,
    full_m3: Vec<f64>,
    log_factor_m2: Vec<f64>,
    log_full_m2: Vec<f64>,
    log_factor_m3: Vec<f64>,
    log_full_m3: Vec<f64>,
}

/// 对一种 (return_type, standardize) 组合计算 m2 和 m3
/// - cum: true = cumulative, false = rolling
/// - std: true = standardize, false = raw
fn compute_combo(
    log_returns: &[f64],
    log_volume: &[f64],
    log_duration: &[f64],
    window: usize,
    do_standardize: bool,
) -> ComboResult {
    let n = log_returns.len();
    // 构建 X 矩阵 (N×3)
    let mut x = Vec::with_capacity(n * 3);
    for i in 0..n {
        x.push(log_returns[i]);
        x.push(log_volume[i]);
        x.push(log_duration[i]);
    }

    let w = window;
    if n < w {
        return ComboResult {
            factor_m2: vec![],
            full_m2: vec![],
            factor_m3: vec![],
            full_m3: vec![],
            log_factor_m2: vec![],
            log_full_m2: vec![],
            log_factor_m3: vec![],
            log_full_m3: vec![],
        };
    }

    // ===== M2: 二阶马氏距离 =====
    // 滚动均值和协方差
    // cumsum of X and outer(X, X)
    let mut cum = vec![0.0f64; (n + 1) * 3]; // cum[0]=0, cum[i+1]=sum(X[0..i+1])
    let mut cum2 = vec![0.0f64; (n + 1) * 9]; // cum2[0]=0, cum2[i+1]=sum of outer products

    for i in 0..n {
        for j in 0..3 {
            cum[(i + 1) * 3 + j] = cum[i * 3 + j] + x[i * 3 + j];
        }
        for j in 0..3 {
            for k in 0..3 {
                cum2[(i + 1) * 9 + j * 3 + k] =
                    cum2[i * 9 + j * 3 + k] + x[i * 3 + j] * x[i * 3 + k];
            }
        }
    }

    // 滚动均值: means[i] = mean of X[i-w..i] for i in [w..n]
    // cum[i] - cum[i-w] / w
    let n_windows = n.saturating_sub(w); // number of windows starting at w
    let mut means = vec![0.0f64; n_windows * 3];
    let mut covs = vec![0.0f64; n_windows * 9];

    for i in 0..n_windows {
        let end = (i + w) * 3;
        let start = i * 3;
        for j in 0..3 {
            means[i * 3 + j] = (cum[end + j] - cum[start + j]) / w as f64;
        }
        let end2 = (i + w) * 9;
        let start2 = i * 9;
        for j in 0..3 {
            for k in 0..3 {
                let mean2 = (cum2[end2 + j * 3 + k] - cum2[start2 + j * 3 + k]) / w as f64;
                covs[i * 9 + j * 3 + k] = mean2 - means[i * 3 + j] * means[i * 3 + k];
            }
        }
    }

    if do_standardize {
        // ===== standardize=True 路径 =====
        // 1. 滚动 std
        let mut stds = vec![0.0f64; n_windows * 3];
        for i in 0..n_windows {
            for j in 0..3 {
                let var = covs[i * 9 + j * 3 + j];
                stds[i * 3 + j] = (var + 1e-8).sqrt();
            }
        }

        // 2. z-score 标准化 X[w..]
        let n_std = n.saturating_sub(w);
        let mut x_std = vec![0.0f64; n_std * 3];
        for i in 0..n_std {
            for j in 0..3 {
                x_std[i * 3 + j] = (x[(i + w) * 3 + j] - means[i * 3 + j]) / stds[i * 3 + j];
            }
        }

        // 3. 对标准化数据重新计算滚动协方差
        let mut cum_s = vec![0.0f64; (n_std + 1) * 3];
        let mut cum2_s = vec![0.0f64; (n_std + 1) * 9];
        for i in 0..n_std {
            for j in 0..3 {
                cum_s[(i + 1) * 3 + j] = cum_s[i * 3 + j] + x_std[i * 3 + j];
            }
            for j in 0..3 {
                for k in 0..3 {
                    cum2_s[(i + 1) * 9 + j * 3 + k] =
                        cum2_s[i * 9 + j * 3 + k] + x_std[i * 3 + j] * x_std[i * 3 + k];
                }
            }
        }

        let nw2 = n_std.saturating_sub(w);
        let mut means_s = vec![0.0f64; nw2 * 3];
        let mut covs_s = vec![0.0f64; nw2 * 9];
        for i in 0..nw2 {
            for j in 0..3 {
                means_s[i * 3 + j] = (cum_s[(i + w) * 3 + j] - cum_s[i * 3 + j]) / w as f64;
            }
            for j in 0..3 {
                for k in 0..3 {
                    let m2 =
                        (cum2_s[(i + w) * 9 + j * 3 + k] - cum2_s[i * 9 + j * 3 + k]) / w as f64;
                    covs_s[i * 9 + j * 3 + k] = m2 - means_s[i * 3 + j] * means_s[i * 3 + k]
                        + 1e-8 * if j == k { 1.0 } else { 0.0 };
                }
            }
        }

        // M2 距离: diff = X_std[w..] - means_s, start_idx = 2*w
        let mut factor_m2 = Vec::with_capacity(nw2);
        let mut full_m2 = Vec::with_capacity(nw2);
        let mut log_factor_m2 = Vec::with_capacity(nw2);
        let mut log_full_m2 = Vec::with_capacity(nw2);
        for i in 0..nw2 {
            let d = [
                x_std[(i + w) * 3] - means_s[i * 3],
                x_std[(i + w) * 3 + 1] - means_s[i * 3 + 1],
                x_std[(i + w) * 3 + 2] - means_s[i * 3 + 2],
            ];
            let cov: [[f64; 3]; 3] = [
                [covs_s[i * 9], covs_s[i * 9 + 1], covs_s[i * 9 + 2]],
                [covs_s[i * 9 + 3], covs_s[i * 9 + 4], covs_s[i * 9 + 5]],
                [covs_s[i * 9 + 6], covs_s[i * 9 + 7], covs_s[i * 9 + 8]],
            ];
            let inv_cov = invert3(&cov).unwrap_or_else(|| invert_diag3(&cov));
            let d_full = quadratic(&d, &inv_cov).sqrt();
            let inv_diag = invert_diag3(&cov);
            let d_diag = quadratic(&d, &inv_diag).sqrt();
            let f_val = d_full - d_diag;
            factor_m2.push(f_val);
            full_m2.push(d_full);
            log_factor_m2.push((f_val + 1e-8).ln());
            log_full_m2.push((d_full + 1e-8).ln());
        }

        // ===== M3: 三阶矩 =====
        // diff2 = X_std[w..] - means_s (即 x_std[w+iw2] - means_s[i])
        // 已经在 nw2 个点上
        let diff2: Vec<[f64; 3]> = (0..nw2)
            .map(|i| {
                [
                    x_std[(i + w) * 3] - means_s[i * 3],
                    x_std[(i + w) * 3 + 1] - means_s[i * 3 + 1],
                    x_std[(i + w) * 3 + 2] - means_s[i * 3 + 2],
                ]
            })
            .collect();

        // 累积三阶张量 3×3×3
        let mut cum3 = vec![0.0f64; (nw2 + 1) * 27];
        for i in 0..nw2 {
            let d = &diff2[i];
            for j in 0..3 {
                for k in 0..3 {
                    for l in 0..3 {
                        cum3[(i + 1) * 27 + j * 9 + k * 3 + l] =
                            cum3[i * 27 + j * 9 + k * 3 + l] + d[j] * d[k] * d[l];
                    }
                }
            }
        }

        let nw3 = nw2.saturating_sub(w);
        let mut factor_m3 = Vec::with_capacity(nw3);
        let mut full_m3 = Vec::with_capacity(nw3);
        let mut log_factor_m3 = Vec::with_capacity(nw3);
        let mut log_full_m3 = Vec::with_capacity(nw3);
        for i in 0..nw3 {
            // 三阶矩均值
            let mut means3 = [[[0.0f64; 3]; 3]; 3];
            for j in 0..3 {
                for k in 0..3 {
                    for l in 0..3 {
                        means3[j][k][l] = (cum3[(i + w) * 27 + j * 9 + k * 3 + l]
                            - cum3[i * 27 + j * 9 + k * 3 + l])
                            / w as f64;
                    }
                }
            }

            // skewness_weight = 0.5
            let skew_w = 0.5;
            let base_cov: [[f64; 3]; 3] = [
                [
                    covs_s[(i + w) * 9],
                    covs_s[(i + w) * 9 + 1],
                    covs_s[(i + w) * 9 + 2],
                ],
                [
                    covs_s[(i + w) * 9 + 3],
                    covs_s[(i + w) * 9 + 4],
                    covs_s[(i + w) * 9 + 5],
                ],
                [
                    covs_s[(i + w) * 9 + 6],
                    covs_s[(i + w) * 9 + 7],
                    covs_s[(i + w) * 9 + 8],
                ],
            ];

            let mut cov_skew = base_cov;
            for j in 0..3 {
                for k in 0..3 {
                    if j != k {
                        cov_skew[j][k] *= 1.0 + skew_w * means3[j][j][k];
                    }
                }
                cov_skew[j][j] += 1e-8;
            }

            let d = diff2[i + w];
            let inv_cov = invert3(&cov_skew).unwrap_or_else(|| invert_diag3(&cov_skew));
            let d_full = quadratic(&d, &inv_cov).sqrt();
            let inv_diag = invert_diag3(&cov_skew);
            let d_diag = quadratic(&d, &inv_diag).sqrt();
            let f_val = d_full - d_diag;
            factor_m3.push(f_val);
            full_m3.push(d_full);
            log_factor_m3.push((f_val + 1e-8).ln());
            log_full_m3.push((d_full + 1e-8).ln());
        }

        // 对齐长度：nw2 和 nw3 都基于 n_std
        // M2 长度 = nw2, M3 长度 = nw3
        // go() 中用 min_len 对齐，这里直接返回各自长度
        // 但外面会用 min_len 截断
        // 注意：nw2 对应的原始索引从 2*w 开始
        // nw3 对应的原始索引从 3*w 开始
        // 但为了和 Python 对齐，factor_m2 的长度 = nw2，factor_m3 的长度 = nw3
        // 在组合时取 min
        ComboResult {
            factor_m2,
            full_m2,
            factor_m3,
            full_m3,
            log_factor_m2,
            log_full_m2,
            log_factor_m3,
            log_full_m3,
        }
    } else {
        // ===== standardize=False 路径 =====
        // M2: 直接在原始 X 上计算滚动协方差
        // covs 已经算好
        // diff = X[w+i] - means[i], i in [0..n_windows)
        let mut factor_m2 = Vec::with_capacity(n_windows);
        let mut full_m2 = Vec::with_capacity(n_windows);
        let mut log_factor_m2 = Vec::with_capacity(n_windows);
        let mut log_full_m2 = Vec::with_capacity(n_windows);
        for i in 0..n_windows {
            let d = [
                x[(i + w) * 3] - means[i * 3],
                x[(i + w) * 3 + 1] - means[i * 3 + 1],
                x[(i + w) * 3 + 2] - means[i * 3 + 2],
            ];
            let mut cov: [[f64; 3]; 3] = [
                [covs[i * 9], covs[i * 9 + 1], covs[i * 9 + 2]],
                [covs[i * 9 + 3], covs[i * 9 + 4], covs[i * 9 + 5]],
                [covs[i * 9 + 6], covs[i * 9 + 7], covs[i * 9 + 8]],
            ];
            // add regularization
            for j in 0..3 {
                cov[j][j] += 1e-8;
            }
            let inv_cov = invert3(&cov).unwrap_or_else(|| invert_diag3(&cov));
            let d_full = quadratic(&d, &inv_cov).sqrt();
            let inv_diag = invert_diag3(&cov);
            let d_diag = quadratic(&d, &inv_diag).sqrt();
            let f_val = d_full - d_diag;
            factor_m2.push(f_val);
            full_m2.push(d_full);
            log_factor_m2.push((f_val + 1e-8).ln());
            log_full_m2.push((d_full + 1e-8).ln());
        }

        // M3: 在去均值后的数据上累积三阶矩
        // diff = X[w+i] - means[i]
        let n_diff = n_windows; // n - w
        let diff: Vec<[f64; 3]> = (0..n_diff)
            .map(|i| {
                [
                    x[(i + w) * 3] - means[i * 3],
                    x[(i + w) * 3 + 1] - means[i * 3 + 1],
                    x[(i + w) * 3 + 2] - means[i * 3 + 2],
                ]
            })
            .collect();

        let mut cum3 = vec![0.0f64; (n_diff + 1) * 27];
        for i in 0..n_diff {
            let d = &diff[i];
            for j in 0..3 {
                for k in 0..3 {
                    for l in 0..3 {
                        cum3[(i + 1) * 27 + j * 9 + k * 3 + l] =
                            cum3[i * 27 + j * 9 + k * 3 + l] + d[j] * d[k] * d[l];
                    }
                }
            }
        }

        let nw3 = n_diff.saturating_sub(w);
        let mut factor_m3 = Vec::with_capacity(nw3);
        let mut full_m3 = Vec::with_capacity(nw3);
        let mut log_factor_m3 = Vec::with_capacity(nw3);
        let mut log_full_m3 = Vec::with_capacity(nw3);
        for i in 0..nw3 {
            let mut means3 = [[[0.0f64; 3]; 3]; 3];
            for j in 0..3 {
                for k in 0..3 {
                    for l in 0..3 {
                        means3[j][k][l] = (cum3[(i + w) * 27 + j * 9 + k * 3 + l]
                            - cum3[i * 27 + j * 9 + k * 3 + l])
                            / w as f64;
                    }
                }
            }

            let skew_w = 0.5;
            // 协方差来自 covs[i+w]
            let base_idx = (i + w) * 9;
            let base_cov: [[f64; 3]; 3] = [
                [covs[base_idx], covs[base_idx + 1], covs[base_idx + 2]],
                [covs[base_idx + 3], covs[base_idx + 4], covs[base_idx + 5]],
                [covs[base_idx + 6], covs[base_idx + 7], covs[base_idx + 8]],
            ];

            let mut cov_skew = base_cov;
            for j in 0..3 {
                for k in 0..3 {
                    if j != k {
                        cov_skew[j][k] *= 1.0 + skew_w * means3[j][j][k];
                    }
                }
                cov_skew[j][j] += 1e-8;
            }

            let d = diff[i + w];
            let inv_cov = invert3(&cov_skew).unwrap_or_else(|| invert_diag3(&cov_skew));
            let d_full = quadratic(&d, &inv_cov).sqrt();
            let inv_diag = invert_diag3(&cov_skew);
            let d_diag = quadratic(&d, &inv_diag).sqrt();
            let f_val = d_full - d_diag;
            factor_m3.push(f_val);
            full_m3.push(d_full);
            log_factor_m3.push((f_val + 1e-8).ln());
            log_full_m3.push((d_full + 1e-8).ln());
        }

        ComboResult {
            factor_m2,
            full_m2,
            factor_m3,
            full_m3,
            log_factor_m2,
            log_full_m2,
            log_factor_m3,
            log_full_m3,
        }
    }
}

#[pyfunction]
pub fn illusion_liquidity_distance_factors(
    py: Python<'_>,
    price: PyReadonlyArray1<f64>,
    volume: PyReadonlyArray1<f64>,
    turnover: PyReadonlyArray1<f64>,
    timestamps_ns: PyReadonlyArray1<i64>,
    window: usize,
) -> PyResult<(Vec<String>, Py<PyArray2<f64>>)> {
    let price = price.as_slice()?;
    let volume = volume.as_slice()?;
    let turnover = turnover.as_slice()?;
    let timestamps_ns = timestamps_ns.as_slice()?;

    let n = price.len();
    if n < 2 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "数据量不足: 需要≥2条数据, 当前只有{}条",
            n
        )));
    }

    // 预处理
    let mut duration = vec![0.0f64; n];
    for i in 1..n {
        let dt = (timestamps_ns[i] - timestamps_ns[i - 1]) as f64 / 1e9;
        duration[i] = dt.max(1e-9);
    }
    duration[0] = 1e-9;

    let log_volume: Vec<f64> = volume.iter().map(|v| v.ln()).collect();
    let log_duration: Vec<f64> = duration.iter().map(|d| d.ln()).collect();

    // 两种参考价格
    // cumulative: ref_price[i] = cumsum(turnover[0..=i]) / cumsum(volume[0..=i]), shift(1)
    let mut cum_to = vec![0.0f64; n];
    let mut cum_vo = vec![0.0f64; n];
    cum_to[0] = turnover[0];
    cum_vo[0] = volume[0];
    for i in 1..n {
        cum_to[i] = cum_to[i - 1] + turnover[i];
        cum_vo[i] = cum_vo[i - 1] + volume[i];
    }
    // shift(1): ref_price[i] = cum_to[i-1] / cum_vo[i-1]
    let ref_cum: Vec<f64> = (0..n)
        .map(|i| {
            if i > 0 && cum_vo[i - 1] > 0.0 {
                cum_to[i - 1] / cum_vo[i - 1]
            } else {
                f64::NAN
            }
        })
        .collect();

    // rolling: ref_price[i] = sum(turnover[i-w+1..=i]) / sum(volume[i-w+1..=i]), shift(1)
    // 即用滚动窗口 [i-w..i) 的数据
    let w = window;
    let ref_rol: Vec<f64> = (0..n)
        .map(|i| {
            if i == 0 {
                return f64::NAN;
            }
            let start = if i >= w { i - w } else { 0 };
            let mut st = 0.0f64;
            let mut sv = 0.0f64;
            for j in start..i {
                st += turnover[j];
                sv += volume[j];
            }
            if sv > 0.0 {
                st / sv
            } else {
                f64::NAN
            }
        })
        .collect();

    // 构造 log_returns for cumulative and rolling
    let log_ret_cum: Vec<f64> = (0..n)
        .map(|i| {
            if ref_cum[i].is_finite() && ref_cum[i] > 0.0 && price[i] > 0.0 {
                (price[i] / ref_cum[i]).ln()
            } else {
                f64::NAN
            }
        })
        .collect();

    let log_ret_rol: Vec<f64> = (0..n)
        .map(|i| {
            if ref_rol[i].is_finite() && ref_rol[i] > 0.0 && price[i] > 0.0 {
                (price[i] / ref_rol[i]).ln()
            } else {
                f64::NAN
            }
        })
        .collect();

    // 去除 NaN 行（需要在三种变量上同时有效）
    // 我们需要为4种组合分别去除NaN
    // 但为了简化，先找到 log_ret_cum 和 log_ret_rol 各自有效行
    // 然后每种组合各自去 NaN

    // 4种组合
    let combos: [(&str, bool); 4] = [
        ("cum_T", true),  // cumulative + standardize
        ("cum_F", false), // cumulative + no standardize
        ("rol_T", true),  // rolling + standardize
        ("rol_F", false), // rolling + no standardize
    ];

    let mut all_results: Vec<(&str, ComboResult, usize)> = Vec::new(); // (name, result, data_len)

    for (rt, do_std) in &combos {
        let lr = if rt.starts_with("cum") {
            &log_ret_cum
        } else {
            &log_ret_rol
        };

        // 去除 NaN 行
        let mut clean_lr = Vec::new();
        let mut clean_lv = Vec::new();
        let mut clean_ld = Vec::new();
        for i in 0..n {
            if lr[i].is_finite() && log_volume[i].is_finite() && log_duration[i].is_finite() {
                clean_lr.push(lr[i]);
                clean_lv.push(log_volume[i]);
                clean_ld.push(log_duration[i]);
            }
        }

        let result = compute_combo(&clean_lr, &clean_lv, &clean_ld, w, *do_std);
        all_results.push((rt, result, clean_lr.len()));
    }

    // 对齐：所有结果取 min_len
    // 每个 combo 有 m2 和 m3，各自长度不同
    // M2 的 factor/full 长度 = 各自的计算长度
    // M3 的 factor/full 长度 = 各自的计算长度
    // 最终按 min_len 对齐

    // 计算各 combo 的 m2/m3 的偏移
    // std=True: m2 长度 = n_clean - 2*w, m3 长度 = n_clean - 3*w
    // std=False: m2 长度 = n_clean - w, m3 长度 = n_clean - 2*w
    let mut min_len = usize::MAX;
    for (_, res, _) in &all_results {
        let l = std::cmp::min(res.factor_m2.len(), res.factor_m3.len());
        if l < min_len {
            min_len = l;
        }
    }
    if min_len == 0 || min_len == usize::MAX {
        min_len = 0;
    }

    // 列名: 原始16列 + log16列
    let col_names = vec![
        "factor_cum_T_m2",
        "full_cum_T_m2",
        "factor_cum_T_m3",
        "full_cum_T_m3",
        "factor_cum_F_m2",
        "full_cum_F_m2",
        "factor_cum_F_m3",
        "full_cum_F_m3",
        "factor_rol_T_m2",
        "full_rol_T_m2",
        "factor_rol_T_m3",
        "full_rol_T_m3",
        "factor_rol_F_m2",
        "full_rol_F_m2",
        "factor_rol_F_m3",
        "full_rol_F_m3",
        "log_factor_cum_T_m2",
        "log_full_cum_T_m2",
        "log_factor_cum_T_m3",
        "log_full_cum_T_m3",
        "log_factor_cum_F_m2",
        "log_full_cum_F_m2",
        "log_factor_cum_F_m3",
        "log_full_cum_F_m3",
        "log_factor_rol_T_m2",
        "log_full_rol_T_m2",
        "log_factor_rol_T_m3",
        "log_full_rol_T_m3",
        "log_factor_rol_F_m2",
        "log_full_rol_F_m2",
        "log_factor_rol_F_m3",
        "log_full_rol_F_m3",
    ];
    let names: Vec<String> = col_names.iter().map(|s| s.to_string()).collect();

    let ncols = 32;
    let mut data = vec![f64::NAN; min_len * ncols];

    for (ci, (_rt, res, _)) in all_results.iter().enumerate() {
        let base_col = ci * 4;
        let log_base_col = 16 + ci * 4;

        let m2_start = res.factor_m2.len().saturating_sub(min_len);
        let m3_start = res.factor_m3.len().saturating_sub(min_len);

        for i in 0..min_len {
            let mi2 = m2_start + i;
            let mi3 = m3_start + i;
            data[i * ncols + base_col] = res.factor_m2.get(mi2).copied().unwrap_or(f64::NAN);
            data[i * ncols + base_col + 1] = res.full_m2.get(mi2).copied().unwrap_or(f64::NAN);
            data[i * ncols + base_col + 2] = res.factor_m3.get(mi3).copied().unwrap_or(f64::NAN);
            data[i * ncols + base_col + 3] = res.full_m3.get(mi3).copied().unwrap_or(f64::NAN);
            data[i * ncols + log_base_col] =
                res.log_factor_m2.get(mi2).copied().unwrap_or(f64::NAN);
            data[i * ncols + log_base_col + 1] =
                res.log_full_m2.get(mi2).copied().unwrap_or(f64::NAN);
            data[i * ncols + log_base_col + 2] =
                res.log_factor_m3.get(mi3).copied().unwrap_or(f64::NAN);
            data[i * ncols + log_base_col + 3] =
                res.log_full_m3.get(mi3).copied().unwrap_or(f64::NAN);
        }
    }

    let arr = ndarray::Array2::from_shape_vec((min_len, ncols), data).unwrap();
    Ok((names, arr.into_pyarray(py).to_owned()))
}
