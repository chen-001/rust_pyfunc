use pyo3::prelude::*;

fn percentile_99(mut arr: Vec<f64>) -> f64 {
    if arr.is_empty() { return f64::MAX; }
    arr.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    let idx = (arr.len() as f64 * 0.99) as usize;
    arr[idx.min(arr.len() - 1)]
}

fn compute_5stats(thetas: &[f64]) -> Vec<f64> {
    if thetas.len() < 10 { return vec![f64::NAN; 5]; }
    let n = thetas.len() as f64;
    let mean = thetas.iter().sum::<f64>() / n;
    let var = thetas.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
    let n3 = (thetas.len() / 3).max(1);
    let trend = thetas[thetas.len() - n3..].iter().sum::<f64>() / n3 as f64
        - thetas[..n3].iter().sum::<f64>() / n3 as f64;
    vec![mean, var.sqrt(),
        thetas.iter().filter(|&&v| v < 30.0).count() as f64 / n,
        thetas.iter().filter(|&&v| v > 60.0).count() as f64 / n,
        trend]
}

/// 同一梯度方向：投影1次+排序1次 → 分别算3个pct的theta（不用合并第二遍，保证与旧公式完全一致）
fn slice_theta_3(
    data: &[[f64; 2]], i: usize, w: usize, pcts: &[f64],
    zx: f64, zy: f64, gx: f64, gy: f64,
) -> [Option<f64>; 3] {
    let mut fwd = [0.0f64; 1024]; let mut nf = 0usize;
    for j in i..i + w - 1 {
        let p = (data[j][0] - zx) * gx + (data[j][1] - zy) * gy;
        if p > 0.0 { fwd[nf] = p; nf += 1; }
    }
    if nf < 5 { return [None, None, None]; }
    fwd[..nf].sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    let mut out = [None, None, None];
    for pi in 0..3 {
        let idx = ((pcts[pi] / 100.0) * nf as f64) as usize;
        let delta = fwd[idx.min(nf - 1)];
        if delta <= 0.0 { continue; }
        let (mut sx, mut sy, mut sn) = (0.0f64, 0.0f64, 0usize);
        for j in i..i + w - 1 {
            let p = (data[j][0] - zx) * gx + (data[j][1] - zy) * gy;
            if p > 0.0 && p <= delta { sx += data[j][0]; sy += data[j][1]; sn += 1; }
        }
        if sn < 3 { continue; }
        let scx = sx / sn as f64 - zx; let scy = sy / sn as f64 - zy;
        let scn = (scx * scx + scy * scy).sqrt();
        if scn < 1e-10 { continue; }
        let ct = (gx * scx / scn + gy * scy / scn).clamp(-1.0, 1.0);
        out[pi] = Some(ct.acos().to_degrees());
    }
    out
}

/// 调试: 逐个位置比较旧公式和slice_theta_3的theta差值
#[pyfunction]
pub fn debug_compare_thetas(
    price: Vec<f64>, volume: Vec<f64>,
) -> Vec<f64> {
    if volume.is_empty() { return vec![]; }
    let mut dp = Vec::new(); let mut vol = Vec::new(); let mut cv = volume[0];
    for i in 1..price.len() {
        if (price[i] - price[i-1]).abs() > 1e-8 {
            dp.push(price[i] - price[i-1]); vol.push(cv); cv = volume[i];
        } else { cv += volume[i]; }
    }
    if dp.len() < 100 { return vec![]; }
    let dp99 = percentile_99(dp.iter().map(|v| v.abs()).collect());
    let vol99 = percentile_99(vol.clone());
    let mut dp_f = Vec::new(); let mut vol_f = Vec::new();
    for i in 0..dp.len() {
        if dp[i].abs() < dp99 && vol[i] < vol99 { dp_f.push(dp[i]); vol_f.push(vol[i]); }
    }
    if dp_f.len() < 100 { return vec![]; }

    let nd = dp_f.len();
    let mut s = [vec![[0.0f64; 2]; nd], vec![[0.0f64; 2]; nd], vec![[0.0f64; 2]; nd],
                 vec![[0.0f64; 2]; nd], vec![[0.0f64; 2]; nd]];
    for i in 0..nd {
        s[0][i] = [dp_f[i], (vol_f[i] + 1.0).ln()];
        s[1][i] = [dp_f[i], vol_f[i]];
        s[2][i] = [dp_f[i], 0.0];
        s[3][i] = [0.0, (vol_f[i] + 1.0).ln()];
        s[4][i] = [i as f64 / nd as f64, dp_f[i]];
    }
    let pcts = [10.0, 20.0, 30.0];
    let windows = [200usize, 500];
    let is2d = [true, true, false, false, true];
    let mut result = Vec::new();

    for si in 0..5 {
        let data = &s[si]; let two_d = is2d[si];
        for &w in &windows {
            if data.len() <= w { continue; }
            let nd_w = data.len() - w; let wf = w as f64;
            let ng = if two_d { 3 } else { 2 };

            let (mut rx, mut ry, mut rxx, mut rxy, mut ryy) = (0.0, 0.0, 0.0, 0.0, 0.0);
            for j in 0..w { rx+=data[j][0]; ry+=data[j][1]; rxx+=data[j][0]*data[j][0]; rxy+=data[j][0]*data[j][1]; ryy+=data[j][1]*data[j][1]; }
            let (mut x, mut y, mut xx, mut xy, mut yy) = (rx, ry, rxx, rxy, ryy);

            // theta_new[gi][pi], theta_old[gi][pi]
            let mut tn: Vec<Vec<Vec<f64>>> = vec![vec![Vec::new(); 3]; ng];
            let mut to: Vec<Vec<Vec<f64>>> = vec![vec![Vec::new(); 3]; ng];

            for i in 0..nd_w {
                let (zx, zy) = (data[i + w - 1][0], data[i + w - 1][1]);
                let (cmx, cmy) = (x / wf - zx, y / wf - zy); let cgn = (cmx*cmx + cmy*cmy).sqrt();
                let c_dir = if cgn > 1e-10 { Some((cmx/cgn, cmy/cgn)) } else { None };
                let (mx2, my2) = (x / wf, y / wf);
                let (vx, cxy2, vy) = (xx / wf - mx2*mx2, xy / wf - mx2*my2, yy / wf - my2*my2);
                let p_dir = if vx.abs() > 1e-15 || vy.abs() > 1e-15 {
                    let d = ((vx - vy).powi(2) + 4.0 * cxy2 * cxy2).sqrt();
                    let (evx, evy) = (cxy2, (vy - vx + d) / 2.0); let evn = (evx*evx + evy*evy).sqrt();
                    if evn > 1e-10 {
                        let (mut pgx, mut pgy) = (evx / evn, evy / evn);
                        if pgx*(zx-mx2) + pgy*(zy-my2) < 0.0 { pgx = -pgx; pgy = -pgy; }
                        Some((pgx, pgy))
                    } else { None }
                } else { None };
                let s_dir = if two_d {
                    let vx_s = xx / wf - mx2*mx2;
                    if vx_s.abs() > 1e-15 {
                        let slope = (xy / wf - mx2*my2) / vx_s; let sn = (1.0 + slope*slope).sqrt();
                        let (mut sgx, mut sgy) = (1.0 / sn, slope / sn);
                        if sgx*(zx-mx2) + sgy*(zy-my2) < 0.0 { sgx = -sgx; sgy = -sgy; }
                        Some((sgx, sgy))
                    } else { None }
                } else { None };

                let dirs = [c_dir, p_dir, s_dir];
                for gi in 0..ng {
                    let Some((gx, gy)) = dirs[gi] else { continue; };
                    // 新方法
                    let new_out = slice_theta_3(data, i, w, &pcts, zx, zy, gx, gy);
                    for pi in 0..3 { if let Some(th) = new_out[pi] { tn[gi][pi].push(th); } }
                    // 旧方法
                    let mut fwd = [0.0f64; 1024]; let mut nf = 0usize;
                    for j in i..i + w - 1 {
                        let p = (data[j][0] - zx) * gx + (data[j][1] - zy) * gy;
                        if p > 0.0 { fwd[nf] = p; nf += 1; }
                    }
                    if nf >= 5 {
                        fwd[..nf].sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
                        for pi in 0..3 {
                            let pct = pcts[pi];
                            let idx = ((pct / 100.0) * nf as f64) as usize;
                            let delta = fwd[idx.min(nf - 1)];
                            if delta <= 0.0 { continue; }
                            let (mut sx, mut sy, mut sn) = (0.0f64, 0.0f64, 0usize);
                            for j in i..i + w - 1 {
                                let p = (data[j][0] - zx) * gx + (data[j][1] - zy) * gy;
                                if p > 0.0 && p <= delta { sx += data[j][0]; sy += data[j][1]; sn += 1; }
                            }
                            if sn >= 3 {
                                let scx = sx / sn as f64 - zx; let scy = sy / sn as f64 - zy;
                                let scn = (scx * scx + scy * scy).sqrt();
                                if scn >= 1e-10 {
                                    let ct = (gx * scx / scn + gy * scy / scn).clamp(-1.0, 1.0);
                                    to[gi][pi].push(ct.acos().to_degrees());
                                }
                            }
                        }
                    }
                }
                if i + 1 < nd_w {
                    x += data[i + w][0] - data[i][0]; y += data[i + w][1] - data[i][1];
                    xx += data[i + w][0]*data[i + w][0] - data[i][0]*data[i][0];
                    xy += data[i + w][0]*data[i + w][1] - data[i][0]*data[i][1];
                    yy += data[i + w][1]*data[i + w][1] - data[i][1]*data[i][1];
                }
            }
            // 输出差异: 对每个(gi,pi), 返回 theta_new 和 theta_old 中不同的数量
            for gi in 0..ng {
                for pi in 0..3 {
                    let nv = tn[gi][pi].len();
                    let ov = to[gi][pi].len();
                    // 对比各个位置的theta
                    let mut diff_count = 0usize;
                    let mut max_diff = 0.0f64;
                    for k in 0..nv.min(ov) {
                        let d = (tn[gi][pi][k] - to[gi][pi][k]).abs();
                        if d > 1e-10 { diff_count += 1; if d > max_diff { max_diff = d; } }
                    }
                    result.push(nv as f64);           // 0: 新方法theta数
                    result.push(ov as f64);           // 1: 旧方法theta数
                    result.push(diff_count as f64);   // 2: 不同位置数
                    result.push(max_diff);            // 3: 最大差值
                }
            }
        }
    }
    result
}

/// 调试: 返回指定(space, window, pct, gradient)的所有theta值
#[pyfunction]
pub fn debug_theta(
    price: Vec<f64>, volume: Vec<f64>,
    space_idx: usize, window: usize, pct: f64, grad_idx: usize,
) -> Vec<f64> {
    if volume.is_empty() || window > 1024 { return vec![]; }
    let mut dp = Vec::new(); let mut vol = Vec::new(); let mut cv = volume[0];
    for i in 1..price.len() {
        if (price[i] - price[i-1]).abs() > 1e-8 {
            dp.push(price[i] - price[i-1]); vol.push(cv); cv = volume[i];
        } else { cv += volume[i]; }
    }
    if dp.len() < 100 { return vec![]; }
    let dp99 = percentile_99(dp.iter().map(|v| v.abs()).collect());
    let vol99 = percentile_99(vol.clone());
    let mut dp_f = Vec::new(); let mut vol_f = Vec::new();
    for i in 0..dp.len() {
        if dp[i].abs() < dp99 && vol[i] < vol99 { dp_f.push(dp[i]); vol_f.push(vol[i]); }
    }
    if dp_f.len() < 100 { return vec![]; }

    let nd = dp_f.len();
    let mut s = [vec![[0.0f64; 2]; nd], vec![[0.0f64; 2]; nd], vec![[0.0f64; 2]; nd],
                 vec![[0.0f64; 2]; nd], vec![[0.0f64; 2]; nd]];
    for i in 0..nd {
        s[0][i] = [dp_f[i], (vol_f[i] + 1.0).ln()];
        s[1][i] = [dp_f[i], vol_f[i]];
        s[2][i] = [dp_f[i], 0.0];
        s[3][i] = [0.0, (vol_f[i] + 1.0).ln()];
        s[4][i] = [i as f64 / nd as f64, dp_f[i]];
    }
    let data = &s[space_idx.min(4)];
    if data.len() <= window { return vec![]; }
    let w = window; let nd_w = data.len() - w;

    // 只用旧切片公式: 每次都重算一遍（不用slice_theta_3，不用增量更新）
    let pcts = vec![pct];
    let is2d = [true, true, false, false, true];
    let two_d = is2d[space_idx.min(4)];

    let mut thetas = Vec::new();
    let (mut rx, mut ry, mut rxx, mut rxy, mut ryy) = (0.0, 0.0, 0.0, 0.0, 0.0);
    for j in 0..w { rx+=data[j][0]; ry+=data[j][1]; rxx+=data[j][0]*data[j][0]; rxy+=data[j][0]*data[j][1]; ryy+=data[j][1]*data[j][1]; }

    let (mut x, mut y, mut xx, mut xy, mut yy) = (rx, ry, rxx, rxy, ryy);
    for i in 0..nd_w {
        let (zx, zy) = (data[i + w - 1][0], data[i + w - 1][1]);
        let wf = w as f64;
        let dir = match grad_idx {
            0 => {  // centroid
                let (cmx, cmy) = (x / wf - zx, y / wf - zy); let cgn = (cmx*cmx + cmy*cmy).sqrt();
                if cgn > 1e-10 { Some((cmx/cgn, cmy/cgn)) } else { None }
            }
            1 => {  // pca
                let (mx2, my2) = (x / wf, y / wf);
                let (vx, cxy2, vy) = (xx / wf - mx2*mx2, xy / wf - mx2*my2, yy / wf - my2*my2);
                if vx.abs() > 1e-15 || vy.abs() > 1e-15 {
                    let d = ((vx - vy).powi(2) + 4.0 * cxy2 * cxy2).sqrt();
                    let (evx, evy) = (cxy2, (vy - vx + d) / 2.0); let evn = (evx*evx + evy*evy).sqrt();
                    if evn > 1e-10 {
                        let (mut pgx, mut pgy) = (evx / evn, evy / evn);
                        if pgx*(zx-mx2) + pgy*(zy-my2) < 0.0 { pgx = -pgx; pgy = -pgy; }
                        Some((pgx, pgy))
                    } else { None }
                } else { None }
            }
            2 => { if !two_d { None } else {  // slope
                let (mx2, my2) = (x / wf, y / wf);
                let vx_s = xx / wf - mx2*mx2;
                if vx_s.abs() > 1e-15 {
                    let slope = (xy / wf - mx2*my2) / vx_s; let sn = (1.0 + slope*slope).sqrt();
                    let (mut sgx, mut sgy) = (1.0 / sn, slope / sn);
                    if sgx*(zx-mx2) + sgy*(zy-my2) < 0.0 { sgx = -sgx; sgy = -sgy; }
                    Some((sgx, sgy))
                } else { None }
            }}
            _ => None
        };
        if let Some((gx, gy)) = dir {
            // 原始slice_centroid: 投影+排序+第二遍
            let mut fwd = [0.0f64; 1024]; let mut nf = 0usize;
            for j in i..i + w - 1 {
                let p = (data[j][0] - zx) * gx + (data[j][1] - zy) * gy;
                if p > 0.0 { fwd[nf] = p; nf += 1; }
            }
            if nf >= 5 {
                fwd[..nf].sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
                let idx = ((pct / 100.0) * nf as f64) as usize;
                let delta = fwd[idx.min(nf - 1)];
                if delta > 0.0 {
                    let (mut sx, mut sy, mut sn) = (0.0f64, 0.0f64, 0usize);
                    for j in i..i + w - 1 {
                        let p = (data[j][0] - zx) * gx + (data[j][1] - zy) * gy;
                        if p > 0.0 && p <= delta { sx += data[j][0]; sy += data[j][1]; sn += 1; }
                    }
                    if sn >= 3 {
                        let scx = sx / sn as f64 - zx; let scy = sy / sn as f64 - zy;
                        let scn = (scx * scx + scy * scy).sqrt();
                        if scn >= 1e-10 {
                            let ct = (gx * scx / scn + gy * scy / scn).clamp(-1.0, 1.0);
                            thetas.push(ct.acos().to_degrees());
                        }
                    }
                }
            }
        }
        if i + 1 < nd_w {
            x += data[i + w][0] - data[i][0]; y += data[i + w][1] - data[i][1];
            xx += data[i + w][0]*data[i + w][0] - data[i][0]*data[i][0];
            xy += data[i + w][0]*data[i + w][1] - data[i][0]*data[i][1];
            yy += data[i + w][1]*data[i + w][1] - data[i][1]*data[i][1];
        }
    }
    thetas
}

#[pyfunction]
pub fn compute_divergence(
    price: Vec<f64>, volume: Vec<f64>,
    windows: Vec<usize>, pcts: Vec<f64>,
) -> Vec<f64> {
    if volume.is_empty() || windows.iter().any(|&w| w > 1024) { return vec![]; }
    // 价格变化段聚合
    let mut dp = Vec::new(); let mut vol = Vec::new(); let mut cv = volume[0];
    for i in 1..price.len() {
        if (price[i] - price[i-1]).abs() > 1e-8 {
            dp.push(price[i] - price[i-1]); vol.push(cv); cv = volume[i];
        } else { cv += volume[i]; }
    }
    if dp.len() < 100 { return vec![]; }
    let dp99 = percentile_99(dp.iter().map(|v| v.abs()).collect());
    let vol99 = percentile_99(vol.clone());
    let mut dp_f = Vec::new(); let mut vol_f = Vec::new();
    for i in 0..dp.len() {
        if dp[i].abs() < dp99 && vol[i] < vol99 { dp_f.push(dp[i]); vol_f.push(vol[i]); }
    }
    if dp_f.len() < 100 { return vec![]; }

    let nd = dp_f.len();
    // 预置5种空间
    let mut s = vec![vec![[0.0f64; 2]; nd]; 5];
    for i in 0..nd {
        s[0][i] = [dp_f[i], (vol_f[i] + 1.0).ln()];
        s[1][i] = [dp_f[i], vol_f[i]];
        s[2][i] = [dp_f[i], 0.0];
        s[3][i] = [0.0, (vol_f[i] + 1.0).ln()];
        s[4][i] = [i as f64 / nd as f64, dp_f[i]];
    }
    let is2d = [true, true, false, false, true];
    let mut result = Vec::new();

    for si in 0..5 {
        let data = &s[si]; let two_d = is2d[si];
        for &w in &windows {
            if data.len() <= w {
                let ng = if two_d { 3 } else { 2 };
                for _ in 0..ng { for _ in 0..3 { result.extend(vec![f64::NAN; 5]); } }
                continue;
            }
            let nd_w = data.len() - w;
            // 滚动初始化
            let (mut rx, mut ry, mut rxx, mut rxy, mut ryy) = (0.0, 0.0, 0.0, 0.0, 0.0);
            for j in 0..w { rx+=data[j][0]; ry+=data[j][1]; rxx+=data[j][0]*data[j][0]; rxy+=data[j][0]*data[j][1]; ryy+=data[j][1]*data[j][1]; }

            // theta[gi][pi] — gi=0 centroid, 1 pca, 2 slope; pi=0 10%, 1 20%, 2 30%
            let ng = if two_d { 3 } else { 2 };
            let mut theta: Vec<Vec<Vec<f64>>> = vec![vec![Vec::new(); 3]; ng];
            let (mut x, mut y, mut xx, mut xy, mut yy) = (rx, ry, rxx, rxy, ryy);
            for i in 0..nd_w {
                let (zx, zy) = (data[i + w - 1][0], data[i + w - 1][1]);
                let wf = w as f64;
                // centroid梯度
                let (cmx, cmy) = (x / wf - zx, y / wf - zy); let cgn = (cmx*cmx + cmy*cmy).sqrt();
                let c_dir = if cgn > 1e-10 { Some((cmx/cgn, cmy/cgn)) } else { None };
                // pca梯度
                let (mx2, my2) = (x / wf, y / wf);
                let (vx, cxy2, vy) = (xx / wf - mx2*mx2, xy / wf - mx2*my2, yy / wf - my2*my2);
                let p_dir = if vx.abs() > 1e-15 || vy.abs() > 1e-15 {
                    let d = ((vx - vy).powi(2) + 4.0 * cxy2 * cxy2).sqrt();
                    let (evx, evy) = (cxy2, (vy - vx + d) / 2.0); let evn = (evx*evx + evy*evy).sqrt();
                    if evn > 1e-10 {
                        let (mut pgx, mut pgy) = (evx / evn, evy / evn);
                        if pgx*(zx-mx2) + pgy*(zy-my2) < 0.0 { pgx = -pgx; pgy = -pgy; }
                        Some((pgx, pgy))
                    } else { None }
                } else { None };
                // slope梯度（仅2D）
                let s_dir = if two_d {
                    let vx_s = xx / wf - mx2*mx2;
                    if vx_s.abs() > 1e-15 {
                        let slope = (xy / wf - mx2*my2) / vx_s; let sn = (1.0 + slope*slope).sqrt();
                        let (mut sgx, mut sgy) = (1.0 / sn, slope / sn);
                        if sgx*(zx-mx2) + sgy*(zy-my2) < 0.0 { sgx = -sgx; sgy = -sgy; }
                        Some((sgx, sgy))
                    } else { None }
                } else { None };
                // 每个梯度方向：投影1次+排序1次 → 喂3个pct
                let dirs = [c_dir, p_dir, s_dir];
                for gi in 0..ng {
                    let Some((gx, gy)) = dirs[gi] else { continue; };
                    let out = slice_theta_3(data, i, w, &pcts, zx, zy, gx, gy);
                    for pi in 0..3 {
                        if let Some(th) = out[pi] { theta[gi][pi].push(th); }
                    }
                }
                if i + 1 < nd_w {
                    x += data[i + w][0] - data[i][0]; y += data[i + w][1] - data[i][1];
                    xx += data[i + w][0]*data[i + w][0] - data[i][0]*data[i][0];
                    xy += data[i + w][0]*data[i + w][1] - data[i][0]*data[i][1];
                    yy += data[i + w][1]*data[i + w][1] - data[i][1]*data[i][1];
                }
            }
            // 收集stats: 按 pct × gradient 顺序 → c10,p10,s10,c20,p20,s20,...
            for pi in 0..3 { for gi in 0..ng { result.extend(compute_5stats(&theta[gi][pi])); } }
        }
    }
    result
}
