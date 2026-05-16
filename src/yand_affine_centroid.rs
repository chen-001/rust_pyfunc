use pyo3::prelude::*;

/// 计算仿射质心相关特征的时间序列
/// 输入: 按快照组织的十档盘口数据
/// bid_prices, bid_vols, ask_prices, ask_vols: Vec<Vec<f64>> — [snapshots][10]
/// 
/// 返回: Vec<Vec<Vec<f64>>> — [feature_category][snapshot]
///   category 0: centroid_t (全局质心偏移)
///   category 1: centroid_rel (相对偏移)
///   category 2: bid_centroid偏移
///   category 3: ask_centroid偏移
///   category 4: oi1 (1档不平衡)
///   category 5: oi10 (10档不平衡)
///   category 6-11: level1-6各档不平衡
///   category 12: spread
///   category 13: depth
///   category 14-17: sc1, sc3, sc5, sc10 (slice-centroid 1,3,5,10 ticks)
#[pyfunction]
pub fn compute_affine_centroid(
    bid_prices: Vec<Vec<f64>>, bid_vols: Vec<Vec<f64>>,
    ask_prices: Vec<Vec<f64>>, ask_vols: Vec<Vec<f64>>,
) -> Vec<Vec<f64>> {
    let n = bid_prices.len();
    let mut categories: Vec<Vec<f64>> = (0..18).map(|_| Vec::with_capacity(n)).collect();

    for i in 0..n {
        let bp = &bid_prices[i];
        let bv = &bid_vols[i];
        let ap = &ask_prices[i];
        let av = &ask_vols[i];

        if bp.is_empty() || ap.is_empty() || bp[0] <= 0.0 || ap[0] <= 0.0 {
            for c in &mut categories { c.push(f64::NAN); }
            continue;
        }

        let mid = (bp[0] + ap[0]) / 2.0;
        
        // 有效数据
        let mut bid_p_valid = Vec::new(); let mut bid_v_valid = Vec::new();
        let mut ask_p_valid = Vec::new(); let mut ask_v_valid = Vec::new();
        for j in 0..10 {
            if j < bp.len() && bp[j] > 0.0 && bv[j] > 0.0 {
                bid_p_valid.push(bp[j]); bid_v_valid.push(bv[j]);
            }
            if j < ap.len() && ap[j] > 0.0 && av[j] > 0.0 {
                ask_p_valid.push(ap[j]); ask_v_valid.push(av[j]);
            }
        }
        if bid_p_valid.is_empty() || ask_p_valid.is_empty() {
            for c in &mut categories { c.push(f64::NAN); }
            continue;
        }

        // 加权质心
        let sum_bv: f64 = bid_v_valid.iter().sum();
        let sum_av: f64 = ask_v_valid.iter().sum();
        let bid_cen = bid_p_valid.iter().zip(bid_v_valid.iter()).map(|(p, v)| p * v).sum::<f64>() / sum_bv;
        let ask_cen = ask_p_valid.iter().zip(ask_v_valid.iter()).map(|(p, v)| p * v).sum::<f64>() / sum_av;
        let total_depth = sum_bv + sum_av;

        // 各档不平衡
        let mut level_imbs = [f64::NAN; 10];
        for j in 0..10.min(bv.len()).min(av.len()) {
            if bv[j] > 0.0 && av[j] > 0.0 {
                level_imbs[j] = (bv[j] - av[j]) / (bv[j] + av[j]);
            }
        }

        // 1档OI 和 10档OI
        let oi1 = if bv[0] + av[0] > 0.0 { (bv[0] - av[0]) / (bv[0] + av[0]) } else { f64::NAN };
        let oi10 = (sum_bv - sum_av) / (sum_bv + sum_av);

        let spread = ap[0] - bp[0];

        // Slice-centroid
        let mut sc_vals = [f64::NAN; 4];
        for (idx, delta_tick) in [1, 3, 5, 10].iter().enumerate() {
            let delta_price = *delta_tick as f64 * 0.01;
            let low = mid - delta_price;
            let high = mid + delta_price;

            let mut bid_inside = 0.0f64;
            let mut bid_wsum = 0.0f64;
            for j in 0..bid_p_valid.len() {
                if bid_p_valid[j] >= low {
                    bid_inside += bid_v_valid[j];
                    bid_wsum += bid_p_valid[j] * bid_v_valid[j];
                }
            }
            let mut ask_inside = 0.0f64;
            let mut ask_wsum = 0.0f64;
            for j in 0..ask_p_valid.len() {
                if ask_p_valid[j] <= high {
                    ask_inside += ask_v_valid[j];
                    ask_wsum += ask_p_valid[j] * ask_v_valid[j];
                }
            }
            if bid_inside + ask_inside > 0.0 {
                let bid_avg = if bid_inside > 0.0 { bid_wsum / bid_inside } else { mid };
                let ask_avg = if ask_inside > 0.0 { ask_wsum / ask_inside } else { mid };
                let sc = (bid_avg + ask_avg) / 2.0 - mid;
                sc_vals[idx] = sc / delta_price.max(0.001);
            }
        }

        // 填充categories
        categories[0].push((bid_cen + ask_cen) / 2.0 - mid);  // centroid_t
        categories[1].push(((bid_cen + ask_cen) / 2.0 - mid) / mid);  // centroid_rel
        categories[2].push(bid_cen - mid);  // bid偏移
        categories[3].push(ask_cen - mid);  // ask偏移
        categories[4].push(oi1);
        categories[5].push(oi10);
        for j in 0..6 { categories[6 + j].push(level_imbs[j]); }
        categories[12].push(spread);
        categories[13].push(total_depth);
        for j in 0..4 { categories[14 + j].push(sc_vals[j]); }
    }
    categories
}

/// 从时序数据计算统计因子 (替代Python的get_features_factors的简化版)
/// 输入: 时序数据 Vec<f64>
/// 返回: [mean, std, abs_mean, pos_ratio, early_mean, early_std, late_mean, late_std]
#[pyfunction]
pub fn compute_ts_stats(series: Vec<f64>) -> Vec<f64> {
    let valid: Vec<f64> = series.into_iter().filter(|v| v.is_finite()).collect();
    let n = valid.len();
    if n < 5 { return vec![]; }
    
    let mean = valid.iter().sum::<f64>() / n as f64;
    let var: f64 = valid.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n as f64;
    let std = var.sqrt();
    let abs_mean = valid.iter().map(|v| v.abs()).sum::<f64>() / n as f64;
    let pos_ratio = valid.iter().filter(|&&v| v > 0.0).count() as f64 / n as f64;
    
    let n3 = n / 3;
    let early: Vec<f64> = valid.iter().take(n3).copied().collect();
    let late: Vec<f64> = valid.iter().skip(2 * n3).take(n3).copied().collect();
    
    let early_mean = if early.is_empty() { f64::NAN } else { early.iter().sum::<f64>() / early.len() as f64 };
    let early_std = if early.len() > 1 { (early.iter().map(|v| (v - early_mean).powi(2)).sum::<f64>() / early.len() as f64).sqrt() } else { f64::NAN };
    let late_mean = if late.is_empty() { f64::NAN } else { late.iter().sum::<f64>() / late.len() as f64 };
    let late_std = if late.len() > 1 { (late.iter().map(|v| (v - late_mean).powi(2)).sum::<f64>() / late.len() as f64).sqrt() } else { f64::NAN };
    
    vec![mean, std, abs_mean, pos_ratio, early_mean, early_std, late_mean, late_std]
}
