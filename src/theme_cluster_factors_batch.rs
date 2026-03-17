/// 主题聚类因子批量计算模块 - 性能优化版本
///
/// 优化策略:
/// 1. 聚类结果缓存: 相同(k, segment)只计算一次聚类
/// 2. 并行参数搜索: 使用rayon并行处理不同参数组合
/// 3. SIMD距离计算: euclidean/cosine使用SIMD指令加速
/// 4. overlap匹配优化: 使用位图替代HashSet
use std::collections::HashMap;
use std::sync::Mutex;

use ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2, PyReadonlyArrayDyn};
use pyo3::prelude::*;
use rayon::prelude::*;

// ============================================================
// 全局缓存
// ============================================================

/// 聚类缓存键
#[derive(Hash, Eq, PartialEq, Clone, Debug)]
struct CacheKey {
    k: usize,
    segment_idx: usize,
}

/// 单日聚类结果（缓存用）
#[derive(Clone)]
struct CachedDayResult {
    labels: Vec<usize>,
    centers: Array2<f64>,
    features_clean: Array2<f64>,
    valid_indices: Vec<usize>,
    stats: ClusterStats,
    k: usize,
}

/// 全局聚类缓存
static CLUSTER_CACHE: once_cell::sync::Lazy<Mutex<HashMap<String, Vec<CachedDayResult>>>> =
    once_cell::sync::Lazy::new(|| Mutex::new(HashMap::new()));

// ============================================================
// SIMD优化距离计算
// ============================================================

/// 欧氏距离矩阵 (SIMD优化版本)
#[cfg(target_arch = "x86_64")]
fn cdist_euclidean_simd(a: &[f64], b: &[f64], k1: usize, k2: usize, d: usize) -> Vec<f64> {
    use std::arch::x86_64::*;
    
    let mut result = vec![0.0f64; k1 * k2];
    
    for i in 0..k1 {
        let a_row = &a[i * d..(i + 1) * d];
        for j in 0..k2 {
            let b_row = &b[j * d..(j + 1) * d];
            
            // SIMD计算
            let mut sum = 0.0f64;
            let chunks = d / 4;
            let remainder = d % 4;
            
            unsafe {
                for c in 0..chunks {
                    let offset = c * 4;
                    let va = _mm256_loadu_pd(a_row.as_ptr().add(offset));
                    let vb = _mm256_loadu_pd(b_row.as_ptr().add(offset));
                    let vdiff = _mm256_sub_pd(va, vb);
                    let vsq = _mm256_mul_pd(vdiff, vdiff);
                    
                    // 水平求和
                    let arr: [f64; 4] = std::mem::transmute(vsq);
                    sum += arr[0] + arr[1] + arr[2] + arr[3];
                }
            }
            
            // 处理剩余元素
            for r in 0..remainder {
                let diff = a_row[chunks * 4 + r] - b_row[chunks * 4 + r];
                sum += diff * diff;
            }
            
            result[i * k2 + j] = sum.sqrt();
        }
    }
    
    result
}

/// 欧氏距离矩阵 (非SIMD版本，作为fallback)
#[cfg(not(target_arch = "x86_64"))]
fn cdist_euclidean_simd(a: &[f64], b: &[f64], k1: usize, k2: usize, d: usize) -> Vec<f64> {
    let mut result = vec![0.0f64; k1 * k2];
    
    for i in 0..k1 {
        let a_row = &a[i * d..(i + 1) * d];
        for j in 0..k2 {
            let b_row = &b[j * d..(j + 1) * d];
            let mut sum = 0.0f64;
            for dd in 0..d {
                let diff = a_row[dd] - b_row[dd];
                sum += diff * diff;
            }
            result[i * k2 + j] = sum.sqrt();
        }
    }
    
    result
}

/// 余弦距离矩阵 (SIMD优化版本)
#[cfg(target_arch = "x86_64")]
fn cdist_cosine_simd(a: &[f64], b: &[f64], k1: usize, k2: usize, d: usize) -> Vec<f64> {
    use std::arch::x86_64::*;
    
    let mut result = vec![1.0f64; k1 * k2];
    
    // 预计算每个向量的模
    let mut a_norms = vec![0.0f64; k1];
    let mut b_norms = vec![0.0f64; k2];
    
    for i in 0..k1 {
        let a_row = &a[i * d..(i + 1) * d];
        let mut sum = 0.0f64;
        let chunks = d / 4;
        let remainder = d % 4;
        
        unsafe {
            for c in 0..chunks {
                let offset = c * 4;
                let va = _mm256_loadu_pd(a_row.as_ptr().add(offset));
                let vsq = _mm256_mul_pd(va, va);
                let arr: [f64; 4] = std::mem::transmute(vsq);
                sum += arr[0] + arr[1] + arr[2] + arr[3];
            }
        }
        for r in 0..remainder {
            sum += a_row[chunks * 4 + r].powi(2);
        }
        a_norms[i] = sum;
    }
    
    for j in 0..k2 {
        let b_row = &b[j * d..(j + 1) * d];
        let mut sum = 0.0f64;
        let chunks = d / 4;
        let remainder = d % 4;
        
        unsafe {
            for c in 0..chunks {
                let offset = c * 4;
                let vb = _mm256_loadu_pd(b_row.as_ptr().add(offset));
                let vsq = _mm256_mul_pd(vb, vb);
                let arr: [f64; 4] = std::mem::transmute(vsq);
                sum += arr[0] + arr[1] + arr[2] + arr[3];
            }
        }
        for r in 0..remainder {
            sum += b_row[chunks * 4 + r].powi(2);
        }
        b_norms[j] = sum;
    }
    
    // 计算点积
    for i in 0..k1 {
        let a_row = &a[i * d..(i + 1) * d];
        for j in 0..k2 {
            let b_row = &b[j * d..(j + 1) * d];
            
            let mut dot = 0.0f64;
            let chunks = d / 4;
            let remainder = d % 4;
            
            unsafe {
                for c in 0..chunks {
                    let offset = c * 4;
                    let va = _mm256_loadu_pd(a_row.as_ptr().add(offset));
                    let vb = _mm256_loadu_pd(b_row.as_ptr().add(offset));
                    let vdot = _mm256_mul_pd(va, vb);
                    let arr: [f64; 4] = std::mem::transmute(vdot);
                    dot += arr[0] + arr[1] + arr[2] + arr[3];
                }
            }
            for r in 0..remainder {
                dot += a_row[chunks * 4 + r] * b_row[chunks * 4 + r];
            }
            
            let denom = (a_norms[i] * b_norms[j]).sqrt();
            if denom > 1e-15 {
                result[i * k2 + j] = 1.0 - dot / denom;
            }
        }
    }
    
    result
}

#[cfg(not(target_arch = "x86_64"))]
fn cdist_cosine_simd(a: &[f64], b: &[f64], k1: usize, k2: usize, d: usize) -> Vec<f64> {
    let mut result = vec![1.0f64; k1 * k2];
    
    for i in 0..k1 {
        let a_row = &a[i * d..(i + 1) * d];
        for j in 0..k2 {
            let b_row = &b[j * d..(j + 1) * d];
            let mut dot = 0.0f64;
            let mut na = 0.0f64;
            let mut nb = 0.0f64;
            for dd in 0..d {
                dot += a_row[dd] * b_row[dd];
                na += a_row[dd] * a_row[dd];
                nb += b_row[dd] * b_row[dd];
            }
            let denom = (na * nb).sqrt();
            if denom > 1e-15 {
                result[i * k2 + j] = 1.0 - dot / denom;
            }
        }
    }
    
    result
}

/// 马氏距离矩阵 (优化版本 - 预计算协方差逆)
fn cdist_mahalanobis_opt(a: &[f64], b: &[f64], k1: usize, k2: usize, d: usize) -> Vec<f64> {
    let total = k1 + k2;
    
    // 计算均值
    let mut mean = vec![0.0f64; d];
    for i in 0..k1 {
        let a_row = &a[i * d..(i + 1) * d];
        for (j, &val) in a_row.iter().enumerate() {
            mean[j] += val;
        }
    }
    for i in 0..k2 {
        let b_row = &b[i * d..(i + 1) * d];
        for (j, &val) in b_row.iter().enumerate() {
            mean[j] += val;
        }
    }
    for j in 0..d {
        mean[j] /= total as f64;
    }
    
    // 计算协方差矩阵
    let mut cov = vec![0.0f64; d * d];
    for i in 0..k1 {
        let a_row = &a[i * d..(i + 1) * d];
        for j1 in 0..d {
            for j2 in 0..d {
                cov[j1 * d + j2] += (a_row[j1] - mean[j1]) * (a_row[j2] - mean[j2]);
            }
        }
    }
    for i in 0..k2 {
        let b_row = &b[i * d..(i + 1) * d];
        for j1 in 0..d {
            for j2 in 0..d {
                cov[j1 * d + j2] += (b_row[j1] - mean[j1]) * (b_row[j2] - mean[j2]);
            }
        }
    }
    
    let n_f = if total > 1 { (total - 1) as f64 } else { 1.0 };
    for j1 in 0..d {
        for j2 in 0..d {
            cov[j1 * d + j2] /= n_f;
        }
        cov[j1 * d + j1] += 1e-6; // 正则化
    }
    
    // 矩阵求逆 (Gauss-Jordan)
    let cov_inv = match invert_matrix_flat(&cov, d) {
        Some(inv) => inv,
        None => {
            // 回退欧氏距离
            return cdist_euclidean_simd(a, b, k1, k2, d);
        }
    };
    
    // 计算马氏距离
    let mut result = vec![0.0f64; k1 * k2];
    for i in 0..k1 {
        let a_row = &a[i * d..(i + 1) * d];
        for j in 0..k2 {
            let b_row = &b[j * d..(j + 1) * d];
            
            let mut diff = vec![0.0f64; d];
            for dd in 0..d {
                diff[dd] = a_row[dd] - b_row[dd];
            }
            
            let mut val = 0.0f64;
            for d1 in 0..d {
                let mut tmp = 0.0f64;
                for d2 in 0..d {
                    tmp += cov_inv[d1 * d + d2] * diff[d2];
                }
                val += diff[d1] * tmp;
            }
            result[i * k2 + j] = val.max(0.0).sqrt();
        }
    }
    
    result
}

/// 一维数组矩阵求逆
fn invert_matrix_flat(mat: &[f64], n: usize) -> Option<Vec<f64>> {
    let mut aug = vec![0.0f64; n * 2 * n];
    
    for i in 0..n {
        for j in 0..n {
            aug[i * 2 * n + j] = mat[i * n + j];
        }
        aug[i * 2 * n + n + i] = 1.0;
    }
    
    for col in 0..n {
        let mut max_row = col;
        let mut max_val = aug[col * 2 * n + col].abs();
        for row in col + 1..n {
            if aug[row * 2 * n + col].abs() > max_val {
                max_val = aug[row * 2 * n + col].abs();
                max_row = row;
            }
        }
        if max_val < 1e-12 {
            return None;
        }
        
        if max_row != col {
            for j in 0..2 * n {
                let tmp = aug[col * 2 * n + j];
                aug[col * 2 * n + j] = aug[max_row * 2 * n + j];
                aug[max_row * 2 * n + j] = tmp;
            }
        }
        
        let pivot = aug[col * 2 * n + col];
        for j in 0..2 * n {
            aug[col * 2 * n + j] /= pivot;
        }
        
        for row in 0..n {
            if row != col {
                let factor = aug[row * 2 * n + col];
                for j in 0..2 * n {
                    aug[row * 2 * n + j] -= factor * aug[col * 2 * n + j];
                }
            }
        }
    }
    
    let mut inv = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            inv[i * n + j] = aug[i * 2 * n + n + j];
        }
    }
    
    Some(inv)
}

/// 距离矩阵计算入口
fn compute_distance_matrix_flat(
    a: &[f64],
    b: &[f64],
    k1: usize,
    k2: usize,
    d: usize,
    metric: &str,
) -> Vec<f64> {
    match metric {
        "cosine" => cdist_cosine_simd(a, b, k1, k2, d),
        "mahalanobis" => cdist_mahalanobis_opt(a, b, k1, k2, d),
        _ => cdist_euclidean_simd(a, b, k1, k2, d),
    }
}

// ============================================================
// KMeans聚类 (优化版本)
// ============================================================

/// z-score标准化 (连续内存版本)
fn zscore_standardize_flat(data: &[f64], n: usize, d: usize) -> Vec<f64> {
    let mut result = vec![f64::NAN; n * d];
    
    for j in 0..d {
        let mut valid: Vec<f64> = Vec::with_capacity(n);
        for i in 0..n {
            let val = data[i * d + j];
            if val.is_finite() {
                valid.push(val);
            }
        }
        
        if valid.is_empty() {
            continue;
        }
        
        let mean = valid.iter().sum::<f64>() / valid.len() as f64;
        let var = valid.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / valid.len() as f64;
        let std = var.sqrt().max(1e-15);
        
        for i in 0..n {
            let val = data[i * d + j];
            if val.is_finite() {
                result[i * d + j] = (val - mean) / std;
            }
        }
    }
    
    result
}

/// KMeans++初始化 (优化版本)
fn kmeans_pp_init_flat(data: &[f64], n: usize, d: usize, k: usize, seed: u64) -> Vec<f64> {
    if n == 0 || k == 0 {
        return vec![0.0f64; k * d];
    }
    
    let mut centers = vec![0.0f64; k * d];
    let mut rng_state = seed;
    
    let mut next_rand = || -> f64 {
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (rng_state >> 33) as f64 / (1u64 << 31) as f64
    };
    
    // 第一个中心随机选取
    let first = (next_rand() * n as f64) as usize % n;
    centers[..d].copy_from_slice(&data[first * d..(first + 1) * d]);
    
    let mut dists = vec![f64::MAX; n];
    
    for c in 1..k {
        // 更新距离
        let prev_center = &centers[(c - 1) * d..c * d];
        for i in 0..n {
            let mut d2 = 0.0f64;
            for j in 0..d {
                let diff = data[i * d + j] - prev_center[j];
                d2 += diff * diff;
            }
            dists[i] = dists[i].min(d2);
        }
        
        let total: f64 = dists.iter().sum();
        if total < 1e-15 {
            centers[c * d..(c + 1) * d].copy_from_slice(&data[..d]);
            continue;
        }
        
        let target = next_rand() * total;
        let mut cum = 0.0f64;
        let mut chosen = 0;
        for i in 0..n {
            cum += dists[i];
            if cum >= target {
                chosen = i;
                break;
            }
        }
        centers[c * d..(c + 1) * d].copy_from_slice(&data[chosen * d..(chosen + 1) * d]);
    }
    
    centers
}

/// KMeans聚类 (优化版本)
fn kmeans_flat(data: &[f64], n: usize, d: usize, k: usize, n_init: usize, max_iter: usize) -> (Vec<usize>, Vec<f64>) {
    if n == 0 || k == 0 {
        return (vec![0usize; n], vec![0.0f64; k * d]);
    }
    
    let mut best_labels = vec![0usize; n];
    let mut best_centers = vec![0.0f64; k * d];
    let mut best_inertia = f64::MAX;
    
    for init in 0..n_init {
        let seed = 42 + init as u64 * 1000;
        let mut centers = kmeans_pp_init_flat(data, n, d, k, seed);
        let mut labels = vec![0usize; n];
        
        for _ in 0..max_iter {
            let mut changed = false;
            
            // 分配步骤
            for i in 0..n {
                let row = &data[i * d..(i + 1) * d];
                let mut min_dist = f64::MAX;
                let mut min_c = 0;
                
                for c in 0..k {
                    let center = &centers[c * d..(c + 1) * d];
                    let dist: f64 = (0..d).map(|j| (row[j] - center[j]).powi(2)).sum();
                    if dist < min_dist {
                        min_dist = dist;
                        min_c = c;
                    }
                }
                
                if labels[i] != min_c {
                    labels[i] = min_c;
                    changed = true;
                }
            }
            
            // 更新中心
            let mut new_centers = vec![0.0f64; k * d];
            let mut counts = vec![0usize; k];
            
            for i in 0..n {
                let c = labels[i];
                counts[c] += 1;
                for j in 0..d {
                    new_centers[c * d + j] += data[i * d + j];
                }
            }
            
            for c in 0..k {
                if counts[c] > 0 {
                    for j in 0..d {
                        new_centers[c * d + j] /= counts[c] as f64;
                    }
                }
            }
            
            centers = new_centers;
            
            if !changed {
                break;
            }
        }
        
        // 计算惯性
        let mut inertia = 0.0f64;
        for i in 0..n {
            let c = labels[i];
            for j in 0..d {
                let diff = data[i * d + j] - centers[c * d + j];
                inertia += diff * diff;
            }
        }
        
        if inertia < best_inertia {
            best_inertia = inertia;
            best_labels = labels;
            best_centers = centers;
        }
    }
    
    (best_labels, best_centers)
}

// ============================================================
// 聚类统计
// ============================================================

#[derive(Clone, Debug)]
struct ClusterStats {
    size: Vec<f64>,
    mean_return: Vec<f64>,
    std_return: Vec<f64>,
    total_amt: Vec<f64>,
    mean_act_buy_ratio: Vec<f64>,
    mean_intraday_vol: Vec<f64>,
}

/// 计算聚类统计量 (连续内存版本)
fn compute_cluster_stats_flat(features: &[f64], labels: &[usize], n: usize, d: usize, k: usize) -> ClusterStats {
    let mut stats = ClusterStats {
        size: vec![0.0; k],
        mean_return: vec![0.0; k],
        std_return: vec![0.0; k],
        total_amt: vec![0.0; k],
        mean_act_buy_ratio: vec![0.0; k],
        mean_intraday_vol: vec![0.0; k],
    };
    
    let col_ret = 0;
    let col_vol = 1;
    let col_amt = 2;
    let col_buy = 3;
    
    let mut ret_vals: Vec<Vec<f64>> = (0..k).map(|_| Vec::new()).collect();
    
    for i in 0..n {
        let c = labels[i];
        if c >= k {
            continue;
        }
        stats.size[c] += 1.0;
        
        let ret = features[i * d + col_ret];
        if ret.is_finite() {
            ret_vals[c].push(ret);
            stats.mean_return[c] += ret;
        }
        
        let amt = features[i * d + col_amt];
        if amt.is_finite() {
            stats.total_amt[c] += amt;
        }
        
        let buy = features[i * d + col_buy];
        if buy.is_finite() {
            stats.mean_act_buy_ratio[c] += buy;
        }
        
        let vol = features[i * d + col_vol];
        if vol.is_finite() {
            stats.mean_intraday_vol[c] += vol;
        }
    }
    
    for c in 0..k {
        let cnt = stats.size[c];
        if cnt > 0.0 {
            stats.mean_return[c] /= cnt;
            stats.mean_act_buy_ratio[c] /= cnt;
            stats.mean_intraday_vol[c] /= cnt;
        }
        if ret_vals[c].len() > 1 {
            let mean = stats.mean_return[c];
            let var: f64 = ret_vals[c].iter().map(|x| (x - mean).powi(2)).sum::<f64>()
                / (ret_vals[c].len() - 1) as f64;
            stats.std_return[c] = var.sqrt();
        }
    }
    
    stats
}

// ============================================================
// 单日聚类 (带缓存)
// ============================================================

/// 单日聚类 (返回缓存结构)
fn cluster_single_day_cached(features: &[f64], n: usize, d: usize, k: usize) -> CachedDayResult {
    // 找出没有NaN的行
    let mut valid_indices = Vec::with_capacity(n);
    for i in 0..n {
        let mut has_nan = false;
        for j in 0..d {
            if !features[i * d + j].is_finite() {
                has_nan = true;
                break;
            }
        }
        if !has_nan {
            valid_indices.push(i);
        }
    }
    
    let n_valid = valid_indices.len();
    if n_valid == 0 {
        // 无有效数据，返回空结果
        return CachedDayResult {
            labels: vec![usize::MAX; n],
            centers: Array2::zeros((1, d)),
            features_clean: Array2::zeros((0, d)),
            valid_indices: vec![],
            stats: ClusterStats {
                size: vec![],
                mean_return: vec![],
                std_return: vec![],
                total_amt: vec![],
                mean_act_buy_ratio: vec![],
                mean_intraday_vol: vec![],
            },
            k: 0,
        };
    }
    
    let actual_k = k.min(n_valid).max(1);  // 至少1个cluster
    
    // 构建清洗后的特征矩阵
    let mut clean = vec![0.0f64; n_valid * d];
    for (idx, &orig) in valid_indices.iter().enumerate() {
        for j in 0..d {
            clean[idx * d + j] = features[orig * d + j];
        }
    }
    
    // 标准化
    let scaled = zscore_standardize_flat(&clean, n_valid, d);
    
    // KMeans聚类
    let (labels_clean, centers) = kmeans_flat(&scaled, n_valid, d, actual_k, 10, 300);
    
    // 映射回原始索引的标签
    let mut full_labels = vec![usize::MAX; n];
    for (idx, &orig) in valid_indices.iter().enumerate() {
        full_labels[orig] = labels_clean[idx];
    }
    
    let stats = compute_cluster_stats_flat(&clean, &labels_clean, n_valid, d, actual_k);
    
    CachedDayResult {
        labels: full_labels,
        centers: Array2::from_shape_vec((actual_k, d), centers).unwrap_or_else(|_| Array2::zeros((actual_k, d))),
        features_clean: Array2::from_shape_vec((n_valid, d), clean).unwrap_or_else(|_| Array2::zeros((n_valid, d))),
        valid_indices,
        stats,
        k: actual_k,
    }
}

// ============================================================
// 跨日对齐 (优化版本)
// ============================================================

/// 重叠匹配 (优化版本 - 使用位图)
fn overlap_match_opt(labels_t: &[usize], labels_t1: &[usize], k: usize, n_stocks: usize) -> Vec<(usize, usize)> {
    // 预计算每个cluster的成员位图
    let mut cluster_members_t: Vec<Vec<bool>> = (0..k).map(|_| vec![false; n_stocks]).collect();
    let mut cluster_members_t1: Vec<Vec<bool>> = (0..k).map(|_| vec![false; n_stocks]).collect();
    
    for (s, &label) in labels_t.iter().enumerate() {
        if label < k {
            cluster_members_t[label][s] = true;
        }
    }
    for (s, &label) in labels_t1.iter().enumerate() {
        if label < k {
            cluster_members_t1[label][s] = true;
        }
    }
    
    let mut mapping = Vec::new();
    let mut used_t1 = vec![false; k];
    
    for i in 0..k {
        let members_i = &cluster_members_t[i];
        let mut best_j = None;
        let mut best_jaccard = -1.0f64;
        
        for j in 0..k {
            if used_t1[j] {
                continue;
            }
            let members_j = &cluster_members_t1[j];
            
            // 计算Jaccard
            let mut intersection = 0usize;
            let mut union_cnt = 0usize;
            for s in 0..n_stocks {
                let a = members_i[s];
                let b = members_j[s];
                if a && b {
                    intersection += 1;
                    union_cnt += 1;
                } else if a || b {
                    union_cnt += 1;
                }
            }
            
            let jaccard = intersection as f64 / (union_cnt as f64 + 1e-10);
            if jaccard > best_jaccard {
                best_jaccard = jaccard;
                best_j = Some(j);
            }
        }
        
        if let Some(j) = best_j {
            mapping.push((i, j));
            used_t1[j] = true;
        }
    }
    
    mapping
}

/// 对齐两天的聚类结果
fn align_two_days_opt(
    curr: &CachedDayResult,
    prev: &CachedDayResult,
    method: &str,
    distance_metric: &str,
    distance_threshold: f64,
    n_stocks: usize,
) -> Vec<usize> {
    let k = curr.k;
    
    let mapping_pairs: Vec<(usize, usize)> = match method {
        "overlap" => overlap_match_opt(&curr.labels, &prev.labels, k, n_stocks),
        _ => {
            // hungarian
            let curr_centers = curr.centers.as_slice().unwrap_or(&[]);
            let prev_centers = prev.centers.as_slice().unwrap_or(&[]);
            let k1 = curr.centers.nrows();
            let k2 = prev.centers.nrows();
            let d = curr.centers.ncols();
            
            let cost = compute_distance_matrix_flat(curr_centers, prev_centers, k1, k2, d, distance_metric);
            let cost_arr = Array2::from_shape_vec((k1, k2), cost).unwrap_or_else(|_| Array2::zeros((k1, k2)));
            hungarian_match(&cost_arr)
        }
    };
    
    // 构建映射表
    let mut mapping = vec![0usize; k];
    for &(from, to) in &mapping_pairs {
        if from < k {
            mapping[from] = to;
        }
    }
    
    // 距离阈值检查
    if distance_threshold > 0.0 && method != "overlap" {
        let curr_centers = curr.centers.as_slice().unwrap_or(&[]);
        let prev_centers = prev.centers.as_slice().unwrap_or(&[]);
        let k1 = curr.centers.nrows();
        let k2 = prev.centers.nrows();
        let d = curr.centers.ncols();
        
        let cost = compute_distance_matrix_flat(curr_centers, prev_centers, k1, k2, d, distance_metric);
        
        // 计算特征标准差的均值
        let mut std_sum = 0.0f64;
        for j in 0..d {
            let col = curr.centers.column(j);
            let mean = col.mean().unwrap_or(0.0);
            let var: f64 = col.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / k1 as f64;
            std_sum += var.sqrt();
        }
        let feature_std = std_sum / d as f64;
        let threshold = distance_threshold * feature_std;
        
        let max_id = *mapping.iter().max().unwrap_or(&0);
        let mut new_id = max_id + 1;
        
        for i in 0..k {
            let min_dist = (0..k2).map(|j| cost[i * k2 + j]).fold(f64::MAX, f64::min);
            if min_dist > threshold {
                mapping[i] = new_id;
                new_id += 1;
            }
        }
    }
    
    mapping
}

/// 对齐多日聚类结果
fn align_clusters_multi_days_opt(
    day_results: &[CachedDayResult],
    method: &str,
    distance_metric: &str,
    distance_threshold: f64,
    n_stocks: usize,
) -> Vec<Vec<usize>> {
    let n_days = day_results.len();
    let mut aligned_labels = Vec::with_capacity(n_days);
    
    aligned_labels.push(day_results[0].labels.clone());
    
    for d in 1..n_days {
        let mapping = align_two_days_opt(
            &day_results[d],
            &day_results[d - 1],
            method,
            distance_metric,
            distance_threshold,
            n_stocks,
        );
        
        let mut new_labels = vec![usize::MAX; n_stocks];
        for s in 0..n_stocks {
            let orig = day_results[d].labels[s];
            if orig < day_results[d].k {
                new_labels[s] = mapping[orig];
            }
        }
        aligned_labels.push(new_labels);
    }
    
    aligned_labels
}

/// 匈牙利算法
fn hungarian_match(cost: &Array2<f64>) -> Vec<(usize, usize)> {
    let n = cost.nrows();
    let m = cost.ncols();
    let size = n.max(m);
    
    let mut u = vec![0.0f64; size + 1];
    let mut v = vec![0.0f64; size + 1];
    let mut p = vec![0usize; size + 1];
    let mut way = vec![0usize; size + 1];
    
    let mut cost_padded = vec![vec![0.0f64; size]; size];
    for i in 0..n {
        for j in 0..m {
            cost_padded[i][j] = cost[[i, j]];
        }
    }
    
    for i in 1..=size {
        p[0] = i;
        let mut j0 = 0usize;
        let mut min_v = vec![f64::MAX; size + 1];
        let mut used = vec![false; size + 1];
        
        loop {
            used[j0] = true;
            let i0 = p[j0];
            let mut delta = f64::MAX;
            let mut j1 = 0usize;
            
            for j in 1..=size {
                if !used[j] {
                    let cur = if i0 > 0 && i0 <= size && j > 0 && j <= size {
                        cost_padded[i0 - 1][j - 1] - u[i0] - v[j]
                    } else {
                        0.0 - u[i0] - v[j]
                    };
                    if cur < min_v[j] {
                        min_v[j] = cur;
                        way[j] = j0;
                    }
                    if min_v[j] < delta {
                        delta = min_v[j];
                        j1 = j;
                    }
                }
            }
            
            for j in 0..=size {
                if used[j] {
                    u[p[j]] += delta;
                    v[j] -= delta;
                } else {
                    min_v[j] -= delta;
                }
            }
            
            j0 = j1;
            if p[j0] == 0 {
                break;
            }
        }
        
        loop {
            let j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
            if j0 == 0 {
                break;
            }
        }
    }
    
    let mut result = Vec::new();
    for j in 1..=size {
        if p[j] > 0 && p[j] <= n && j <= m {
            result.push((p[j] - 1, j - 1));
        }
    }
    result
}

// ============================================================
// 因子计算
// ============================================================

/// 线性回归斜率
fn linregress_slope(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len() as f64;
    if n < 2.0 {
        return f64::NAN;
    }
    let sx: f64 = x.iter().sum();
    let sy: f64 = y.iter().sum();
    let sxy: f64 = x.iter().zip(y.iter()).map(|(a, b)| a * b).sum();
    let sx2: f64 = x.iter().map(|a| a * a).sum();
    let denom = n * sx2 - sx * sx;
    if denom.abs() < 1e-15 {
        return f64::NAN;
    }
    (n * sxy - sx * sy) / denom
}

/// F4-F6: 主题动量因子
fn factor_theme_momentum_opt(
    theme_history: &[usize],
    n_stocks: usize,
    n_days: usize,
    stats_vec: &[ClusterStats],
    lookback: usize,
) -> Vec<f64> {
    let lb = lookback.min(n_days);
    let start_day = n_days - lb;
    let mut result = vec![f64::NAN; n_stocks * 3];
    
    for s in 0..n_stocks {
        let mut returns = Vec::new();
        let mut xs = Vec::new();
        
        for (idx, d) in (start_day..n_days).enumerate() {
            let theme = theme_history[s * n_days + d];
            if theme != usize::MAX && theme < stats_vec[d].mean_return.len() {
                returns.push(stats_vec[d].mean_return[theme]);
                xs.push(idx as f64);
            }
        }
        
        if !returns.is_empty() {
            // F4: 平均收益
            let mean: f64 = returns.iter().sum::<f64>() / returns.len() as f64;
            result[s * 3 + 0] = mean;
        }
        
        // F5: 倒数第二天的主题收益
        if n_days >= 2 {
            let d = n_days - 2;
            if d >= start_day {
                let theme = theme_history[s * n_days + d];
                if theme != usize::MAX && theme < stats_vec[d].mean_return.len() {
                    result[s * 3 + 1] = stats_vec[d].mean_return[theme];
                }
            }
        }
        
        // F6: 趋势斜率
        if returns.len() >= 2 {
            result[s * 3 + 2] = linregress_slope(&xs, &returns);
        }
    }
    
    result
}

/// F7-F10: 主题强度因子
fn factor_theme_strength_opt(
    theme_labels: &[usize],
    stats: &ClusterStats,
    n_stocks: usize,
) -> Vec<f64> {
    let mut result = vec![f64::NAN; n_stocks * 4];
    
    for s in 0..n_stocks {
        let theme = theme_labels[s];
        if theme != usize::MAX && theme < stats.size.len() {
            result[s * 4 + 0] = stats.size[theme];
            result[s * 4 + 1] = stats.total_amt[theme];
            result[s * 4 + 2] = stats.mean_return[theme];
            result[s * 4 + 3] = stats.mean_act_buy_ratio[theme];
        }
    }
    
    result
}

/// F11-F12: 主题内部排名因子
fn factor_theme_rank_opt(
    features: &[f64],
    theme_labels: &[usize],
    centers: &[f64],
    n_stocks: usize,
    d: usize,
    k: usize,
) -> Vec<f64> {
    let mut result = vec![f64::NAN; n_stocks * 2];
    
    // F11: 主题内收益率排名
    let mut theme_stocks: Vec<Vec<usize>> = vec![Vec::new(); k];
    for s in 0..n_stocks {
        let theme = theme_labels[s];
        if theme != usize::MAX && theme < k {
            theme_stocks[theme].push(s);
        }
    }
    
    for c in 0..k {
        let stocks = &theme_stocks[c];
        if stocks.len() <= 1 {
            if stocks.len() == 1 {
                result[stocks[0] * 2 + 0] = 0.5;
            }
            continue;
        }
        
        let mut rets: Vec<(usize, f64)> = stocks
            .iter()
            .map(|&s| (s, features[s * d + 0]))
            .filter(|(_, r)| r.is_finite())
            .collect();
        rets.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        
        let cnt = rets.len();
        for (rank, &(s, _)) in rets.iter().enumerate() {
            result[s * 2 + 0] = rank as f64 / (cnt - 1).max(1) as f64;
        }
    }
    
    // F12: 与主题中心距离
    let scaled = zscore_standardize_flat(features, n_stocks, d);
    for s in 0..n_stocks {
        let theme = theme_labels[s];
        if theme != usize::MAX && theme < k && theme * d + d <= centers.len() {
            let mut dist = 0.0f64;
            let mut valid = true;
            for j in 0..d {
                if !scaled[s * d + j].is_finite() {
                    valid = false;
                    break;
                }
                let diff = scaled[s * d + j] - centers[theme * d + j];
                dist += diff * diff;
            }
            if valid {
                result[s * 2 + 1] = dist.sqrt();
            }
        }
    }
    
    result
}

/// F13-F14: 主题切换因子
fn factor_theme_switch_opt(
    theme_history: &[usize],
    n_stocks: usize,
    n_days: usize,
    stats_vec: &[ClusterStats],
) -> Vec<f64> {
    let mut result = vec![f64::NAN; n_stocks * 2];
    
    if n_days < 2 {
        return result;
    }
    
    let curr_day = n_days - 1;
    let prev_day = n_days - 2;
    
    for s in 0..n_stocks {
        let curr_theme = theme_history[s * n_days + curr_day];
        let prev_theme = theme_history[s * n_days + prev_day];
        
        if curr_theme == usize::MAX || prev_theme == usize::MAX {
            continue;
        }
        
        if curr_theme == prev_theme {
            result[s * 2 + 0] = 0.0;
            result[s * 2 + 1] = 0.0;
        } else {
            let curr_stats = &stats_vec[curr_day];
            let prev_stats = &stats_vec[prev_day];
            
            if curr_theme < curr_stats.mean_return.len() && prev_theme < prev_stats.mean_return.len() {
                result[s * 2 + 0] = curr_stats.mean_return[curr_theme] - prev_stats.mean_return[prev_theme];
                result[s * 2 + 1] = curr_stats.total_amt[curr_theme] - prev_stats.total_amt[prev_theme];
            }
        }
    }
    
    result
}

/// F21: 主题熵因子
fn factor_theme_entropy_opt(
    theme_history: &[usize],
    n_stocks: usize,
    n_days: usize,
    lookback: usize,
) -> Vec<f64> {
    let lb = lookback.min(n_days);
    let start_day = n_days - lb;
    let mut result = vec![f64::NAN; n_stocks];
    
    for s in 0..n_stocks {
        let mut themes = Vec::new();
        for d in start_day..n_days {
            let t = theme_history[s * n_days + d];
            if t != usize::MAX {
                themes.push(t);
            }
        }
        
        if !themes.is_empty() {
            let mut counts = std::collections::HashMap::new();
            for &t in &themes {
                *counts.entry(t).or_insert(0usize) += 1;
            }
            let total = themes.len() as f64;
            let entropy: f64 = counts
                .values()
                .map(|&c| {
                    let p = c as f64 / total;
                    -p * (p + 1e-10).ln()
                })
                .sum();
            result[s] = entropy;
        }
    }
    
    result
}

/// F23-F26: 跨日主题演化因子
fn factor_theme_evolution_opt(
    theme_history: &[usize],
    n_stocks: usize,
    n_days: usize,
    stats_vec: &[ClusterStats],
    target_day_idx: usize,
    lookback: usize,
) -> Vec<f64> {
    let mut result = vec![f64::NAN; n_stocks * 4];
    
    let curr_idx = target_day_idx;
    if curr_idx >= n_days {
        return result;
    }
    
    let curr_stats = &stats_vec[curr_idx];
    let prev_stats = if curr_idx > 0 {
        Some(&stats_vec[curr_idx - 1])
    } else {
        None
    };
    
    for s in 0..n_stocks {
        let theme = theme_history[s * n_days + curr_idx];
        if theme == usize::MAX || theme >= curr_stats.total_amt.len() {
            continue;
        }
        
        let curr_amt = curr_stats.total_amt[theme];
        
        // F23: 主题热度
        let lb_start = if curr_idx > lookback { curr_idx - lookback } else { 0 };
        let mut past_amts = Vec::new();
        for d in lb_start..curr_idx {
            if theme < stats_vec[d].total_amt.len() {
                past_amts.push(stats_vec[d].total_amt[theme]);
            }
        }
        if !past_amts.is_empty() {
            let avg: f64 = past_amts.iter().sum::<f64>() / past_amts.len() as f64;
            result[s * 4 + 0] = curr_amt / (avg + 1e-10);
        } else {
            result[s * 4 + 0] = 1.0;
        }
        
        // F24: 动量持续性
        if past_amts.len() >= 3 {
            result[s * 4 + 1] = autocorr_lag1(&past_amts);
        } else {
            result[s * 4 + 1] = 0.0;
        }
        
        // F25: 主题扩散
        if let Some(ps) = prev_stats {
            if theme < ps.size.len() {
                let curr_size = curr_stats.size[theme];
                let prev_size = ps.size[theme];
                result[s * 4 + 2] = (curr_size - prev_size) / (prev_size + 1e-10);
            } else {
                result[s * 4 + 2] = 0.0;
            }
        } else {
            result[s * 4 + 2] = 0.0;
        }
        
        // F26: 领导者稳定性
        result[s * 4 + 3] = curr_stats.size[theme] / 4000.0;
    }
    
    result
}

/// 自相关系数
fn autocorr_lag1(data: &[f64]) -> f64 {
    let n = data.len();
    if n < 2 {
        return 0.0;
    }
    let mean: f64 = data.iter().sum::<f64>() / n as f64;
    let mut num = 0.0f64;
    let mut den = 0.0f64;
    for i in 0..n - 1 {
        num += (data[i] - mean) * (data[i + 1] - mean);
    }
    for x in data {
        den += (x - mean).powi(2);
    }
    if den.abs() < 1e-15 { 0.0 } else { num / den }
}

/// F17-F18: 微观偏离因子
fn factor_micro_deviation_opt(
    micro_metric: &[f64],
    prev_micro_metric: Option<&[f64]>,
    theme_labels: &[usize],
    prev_theme_labels: Option<&[usize]>,
    n_stocks: usize,
    k: usize,
) -> Vec<f64> {
    let mut result = vec![f64::NAN; n_stocks * 2];
    
    // 当日主题分组
    let mut theme_stocks: Vec<Vec<usize>> = vec![Vec::new(); k.max(1)];
    for s in 0..n_stocks {
        let t = theme_labels[s];
        if t != usize::MAX && t < k {
            theme_stocks[t].push(s);
        }
    }
    
    // 当日主题均值
    let mut theme_mean = vec![f64::NAN; k.max(1)];
    for c in 0..k {
        if !theme_stocks[c].is_empty() {
            let sum: f64 = theme_stocks[c]
                .iter()
                .filter_map(|&s| if micro_metric[s].is_finite() { Some(micro_metric[s]) } else { None })
                .sum();
            let cnt = theme_stocks[c].iter().filter(|&&s| micro_metric[s].is_finite()).count();
            if cnt > 0 {
                theme_mean[c] = sum / cnt as f64;
            }
        }
    }
    
    // 前日主题均值
    let mut prev_theme_mean = vec![f64::NAN; k.max(1)];
    if let (Some(prev_labels), Some(prev_micro)) = (prev_theme_labels, prev_micro_metric) {
        let mut prev_theme_stocks: Vec<Vec<usize>> = vec![Vec::new(); k.max(1)];
        for s in 0..n_stocks.min(prev_labels.len()) {
            let t = prev_labels[s];
            if t != usize::MAX && t < k {
                prev_theme_stocks[t].push(s);
            }
        }
        for c in 0..k {
            if !prev_theme_stocks[c].is_empty() {
                let sum: f64 = prev_theme_stocks[c]
                    .iter()
                    .filter_map(|&s| if s < prev_micro.len() && prev_micro[s].is_finite() { Some(prev_micro[s]) } else { None })
                    .sum();
                let cnt = prev_theme_stocks[c].iter().filter(|&&s| s < prev_micro.len() && prev_micro[s].is_finite()).count();
                if cnt > 0 {
                    prev_theme_mean[c] = sum / cnt as f64;
                }
            }
        }
    }
    
    for s in 0..n_stocks {
        let theme = theme_labels[s];
        if theme == usize::MAX || theme >= k {
            continue;
        }
        
        let val = micro_metric[s];
        let tm = theme_mean[theme];
        if val.is_finite() && tm.is_finite() {
            result[s * 2 + 0] = val - tm;
            
            if let (Some(prev_labels), Some(prev_micro)) = (prev_theme_labels, prev_micro_metric) {
                if s < prev_labels.len() {
                    let prev_theme = prev_labels[s];
                    if prev_theme != usize::MAX && prev_theme < k && s < prev_micro.len() && prev_micro[s].is_finite() && prev_theme_mean[prev_theme].is_finite() {
                        let prev_dev = prev_micro[s] - prev_theme_mean[prev_theme];
                        result[s * 2 + 1] = (val - tm) - prev_dev;
                    }
                }
            }
        }
    }
    
    result
}

// ============================================================
// 主入口: theme_cluster_factors_batch
// ============================================================

/// 单次参数组合的计算
fn compute_single_factor_set(
    day_results: &[CachedDayResult],
    features_arrays: &[Array2<f64>],
    micro_arrays: &[Array2<f64>],
    n_stocks: usize,
    n_days: usize,
    align_method: &str,
    distance_metric: &str,
    distance_threshold: f64,
    lookback: usize,
) -> Vec<f64> {
    // 跨日对齐
    let aligned_labels = align_clusters_multi_days_opt(
        day_results,
        align_method,
        distance_metric,
        distance_threshold,
        n_stocks,
    );
    
    // 构建 theme_history
    let mut theme_history = vec![usize::MAX; n_stocks * n_days];
    for d in 0..n_days {
        for s in 0..n_stocks {
            theme_history[s * n_days + d] = aligned_labels[d][s];
        }
    }
    
    // 构建对齐后的统计量
    let stats_vec: Vec<ClusterStats> = (0..n_days)
        .map(|d| {
            let aligned_k = aligned_labels[d]
                .iter()
                .filter(|&&l| l != usize::MAX)
                .cloned()
                .max()
                .unwrap_or(0)
                + 1;
            
            let valid = &day_results[d].valid_indices;
            let labels_for_clean: Vec<usize> = valid.iter().map(|&vi| aligned_labels[d][vi]).collect();
            let clean = day_results[d].features_clean.as_slice().unwrap_or(&[]);
            let n_valid = valid.len();
            let dd = day_results[d].features_clean.ncols();
            
            compute_cluster_stats_flat(clean, &labels_for_clean, n_valid, dd, aligned_k)
        })
        .collect();
    
    let target_day = n_days - 1;
    let target_labels = &aligned_labels[target_day];
    
    // 计算各因子
    let f_momentum = factor_theme_momentum_opt(&theme_history, n_stocks, n_days, &stats_vec, lookback);
    let f_strength = factor_theme_strength_opt(target_labels, &stats_vec[target_day], n_stocks);
    
    let aligned_k_target = target_labels.iter().filter(|&&l| l != usize::MAX).cloned().max().unwrap_or(0) + 1;
    let target_features = features_arrays[target_day].as_slice().unwrap_or(&[]);
    let target_centers = day_results[target_day].centers.as_slice().unwrap_or(&[]);
    let d = features_arrays[target_day].ncols();
    let f_rank = factor_theme_rank_opt(target_features, target_labels, target_centers, n_stocks, d, aligned_k_target);
    
    let f_switch = factor_theme_switch_opt(&theme_history, n_stocks, n_days, &stats_vec);
    let f_entropy = factor_theme_entropy_opt(&theme_history, n_stocks, n_days, lookback);
    let f_evolution = factor_theme_evolution_opt(&theme_history, n_stocks, n_days, &stats_vec, target_day, lookback);
    
    // 微观偏离因子
    let n_micro = if !micro_arrays.is_empty() { micro_arrays[0].ncols() } else { 0 };
    let mut micro_factors: Vec<Vec<f64>> = Vec::new();
    
    if n_micro > 0 && n_days >= 1 {
        let prev_labels = if n_days >= 2 { Some(aligned_labels[n_days - 2].as_slice()) } else { None };
        
        for m_idx in 0..n_micro {
            let curr_metric: Vec<f64> = micro_arrays[target_day].column(m_idx).iter().cloned().collect();
            let prev_metric: Option<Vec<f64>> = if n_days >= 2 && micro_arrays.len() > n_days - 2 {
                Some(micro_arrays[n_days - 2].column(m_idx).iter().cloned().collect())
            } else {
                None
            };
            
            let f = factor_micro_deviation_opt(
                &curr_metric,
                prev_metric.as_deref(),
                target_labels,
                prev_labels,
                n_stocks,
                aligned_k_target,
            );
            micro_factors.push(f);
        }
    }
    
    // 合并因子
    let n_base = 16;
    let n_micro_cols = n_micro * 2;
    let total_cols = n_base + n_micro_cols;
    let mut output = vec![f64::NAN; n_stocks * total_cols];
    
    for s in 0..n_stocks {
        // F4-F6
        output[s * total_cols + 0] = f_momentum[s * 3 + 0];
        output[s * total_cols + 1] = f_momentum[s * 3 + 1];
        output[s * total_cols + 2] = f_momentum[s * 3 + 2];
        // F7-F10
        output[s * total_cols + 3] = f_strength[s * 4 + 0];
        output[s * total_cols + 4] = f_strength[s * 4 + 1];
        output[s * total_cols + 5] = f_strength[s * 4 + 2];
        output[s * total_cols + 6] = f_strength[s * 4 + 3];
        // F11-F12
        output[s * total_cols + 7] = f_rank[s * 2 + 0];
        output[s * total_cols + 8] = f_rank[s * 2 + 1];
        // F13-F14
        output[s * total_cols + 9] = f_switch[s * 2 + 0];
        output[s * total_cols + 10] = f_switch[s * 2 + 1];
        // F21
        output[s * total_cols + 11] = f_entropy[s];
        // F23-F26
        output[s * total_cols + 12] = f_evolution[s * 4 + 0];
        output[s * total_cols + 13] = f_evolution[s * 4 + 1];
        output[s * total_cols + 14] = f_evolution[s * 4 + 2];
        output[s * total_cols + 15] = f_evolution[s * 4 + 3];
        // 微观偏离因子
        for (m_idx, mf) in micro_factors.iter().enumerate() {
            let col_offset = n_base + m_idx * 2;
            output[s * total_cols + col_offset] = mf[s * 2 + 0];
            output[s * total_cols + col_offset + 1] = mf[s * 2 + 1];
        }
    }
    
    output
}

/// 批量计算主题聚类因子
///
/// 参数:
/// - features_list: 多日特征矩阵列表, 每个shape=(n_stocks, 12)
/// - micro_metrics_list: 多日微观指标矩阵列表, 每个shape=(n_stocks, 8)
/// - k_values: 聚类数列表
/// - align_methods: 对齐方法列表 ("hungarian" / "overlap")
/// - distance_metrics: 距离度量列表 ("euclidean" / "cosine" / "mahalanobis")
/// - distance_thresholds: 距离阈值列表 (None用负数表示)
/// - lookback_values: 回溯天数列表
/// - n_threads: 并行线程数 (不超过10)
///
/// 返回:
///   (因子矩阵, 列名列表)
///   - 因子矩阵: shape=(n_stocks, n_factors * n_param_combos)
///   - 列名列表: 每列的名称，格式为 {factor_name}__k{k}_{align}_{dist}_{thresh}_{lookback}
#[pyfunction]
#[pyo3(signature = (features_list, micro_metrics_list, k_values, align_methods, distance_metrics, distance_thresholds, lookback_values, n_threads))]
pub fn theme_cluster_factors_batch(
    py: Python,
    features_list: Vec<PyReadonlyArray2<f64>>,
    micro_metrics_list: Vec<PyReadonlyArray2<f64>>,
    k_values: Vec<usize>,
    align_methods: Vec<String>,
    distance_metrics: Vec<String>,
    distance_thresholds: Vec<f64>,
    lookback_values: Vec<usize>,
    n_threads: usize,
) -> PyResult<(Py<PyArray2<f64>>, Vec<String>)> {
    let n_days = features_list.len();
    if n_days == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err("features_list不能为空"));
    }
    
    let n_stocks = features_list[0].as_array().nrows();
    
    // 转为owned ndarray
    let features_arrays: Vec<Array2<f64>> = features_list.iter().map(|f| f.as_array().to_owned()).collect();
    let micro_arrays: Vec<Array2<f64>> = micro_metrics_list.iter().map(|f| f.as_array().to_owned()).collect();
    
    let d = features_arrays[0].ncols();
    
    // 限制线程数
    let actual_threads = n_threads.min(10).max(1);
    
    // 构建参数组合
    let mut param_combos: Vec<(usize, String, String, f64, usize)> = Vec::new();
    for &k in &k_values {
        for align in &align_methods {
            for dist in &distance_metrics {
                for &thresh in &distance_thresholds {
                    for &lb in &lookback_values {
                        param_combos.push((k, align.clone(), dist.clone(), thresh, lb));
                    }
                }
            }
        }
    }
    
    let n_combos = param_combos.len();
    let n_micro = if !micro_arrays.is_empty() { micro_arrays[0].ncols() } else { 0 };
    let n_factors = 16 + n_micro * 2;
    let total_cols = n_factors * n_combos;
    
    // 生成列名
    let base_factor_names = vec![
        "theme_return_mean",        // F4
        "yesterday_theme_return",   // F5
        "theme_return_trend",       // F6
        "theme_size",               // F7
        "theme_amount",             // F8
        "theme_return",             // F9
        "theme_act_buy_ratio",      // F10
        "return_rank_in_theme",     // F11
        "distance_to_center",       // F12
        "switch_direction",         // F13
        "switch_strength",          // F14
        "theme_entropy",            // F21
        "theme_heat",               // F23
        "theme_momentum_persistence", // F24
        "theme_diffusion",          // F25
        "leader_stability",         // F26
    ];
    
    let micro_metric_names = vec![
        "act_buy_ratio", "bid_ask_imbalance", "vol_per_trade",
        "spread_ratio", "depth_imbalance", "vwap_deviation",
        "act_buy_vol_ratio", "big_order_ratio",
    ];
    
    let mut col_names: Vec<String> = Vec::with_capacity(total_cols);
    for (k, align, dist, thresh, lb) in &param_combos {
        let thresh_str = if *thresh < 0.0 { "None".to_string() } else { thresh.to_string() };
        let param_suffix = format!("__k{}_{}_{}_{}_{}", k, align, dist, thresh_str, lb);
        
        // 基础因子
        for name in &base_factor_names {
            col_names.push(format!("{}{}", name, param_suffix));
        }
        // 微观偏离因子
        for m_idx in 0..n_micro {
            let m_name = micro_metric_names.get(m_idx).unwrap_or(&"unknown");
            col_names.push(format!("{}_deviation{}", m_name, param_suffix));       // F17
            col_names.push(format!("{}_deviation_change{}", m_name, param_suffix)); // F18
        }
    }
    
    // 按k值分组, 预计算聚类
    let mut k_values_unique: Vec<usize> = k_values.iter().cloned().collect();
    k_values_unique.sort();
    k_values_unique.dedup();
    
    // 预计算所有k值对应的聚类结果
    let mut cluster_results_by_k: HashMap<usize, Vec<CachedDayResult>> = HashMap::new();
    
    for &k in &k_values_unique {
        let mut day_results: Vec<CachedDayResult> = Vec::with_capacity(n_days);
        for day_idx in 0..n_days {
            let features = features_arrays[day_idx].as_slice().unwrap_or(&[]);
            let result = cluster_single_day_cached(features, n_stocks, d, k);
            day_results.push(result);
        }
        cluster_results_by_k.insert(k, day_results);
    }
    
    // 创建线程池
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(actual_threads)
        .build()
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("创建线程池失败: {}", e)))?;
    
    // 并行计算各参数组合
    let results: Vec<Vec<f64>> = pool.install(|| {
        param_combos
            .into_par_iter()
            .map(|(k, align, dist, thresh, lb)| {
                let day_results = cluster_results_by_k.get(&k).unwrap();
                compute_single_factor_set(
                    day_results,
                    &features_arrays,
                    &micro_arrays,
                    n_stocks,
                    n_days,
                    &align,
                    &dist,
                    thresh,
                    lb,
                )
            })
            .collect()
    });
    
    // 合并结果为二维数组: (n_stocks, n_factors * n_combos)
    let mut output = vec![f64::NAN; n_stocks * total_cols];
    for (combo_idx, result) in results.iter().enumerate() {
        let col_offset = combo_idx * n_factors;
        for s in 0..n_stocks {
            for f in 0..n_factors {
                output[s * total_cols + col_offset + f] = result[s * n_factors + f];
            }
        }
    }
    
    let arr = Array2::from_shape_vec((n_stocks, total_cols), output)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("数组形状错误: {}", e)))?;
    
    Ok((arr.into_pyarray(py).to_owned(), col_names))
}

/// 批量计算主题聚类因子 (多时段并行版本)
///
/// 参数:
/// - segments_data: List[(segment_name, features_list, micro_list)]
///   - segment_name: 时段名称字符串
///   - features_list: List[ndarray], 每个shape=(n_stocks, 12)
///   - micro_list: List[ndarray], 每个shape=(n_stocks, 8)
/// - k_values: 聚类数列表
/// - align_methods: 对齐方法列表
/// - distance_metrics: 距离度量列表
/// - distance_thresholds: 距离阈值列表 (None用负数表示)
/// - lookback_values: 回溯天数列表
/// - n_threads: 并行线程数 (不超过10)
///
/// 返回:
///   (因子矩阵, 列名列表)
///   - 因子矩阵: shape=(n_stocks, total_cols)
///   - 列名列表: 格式为 {factor_name}__k{k}_{align}_{dist}_{thresh}_{lookback}__{segment}
#[pyfunction]
#[pyo3(signature = (segments_data, k_values, align_methods, distance_metrics, distance_thresholds, lookback_values, n_threads))]
pub fn theme_cluster_factors_batch_multi_segments(
    py: Python,
    segments_data: Vec<(String, Vec<PyReadonlyArray2<f64>>, Vec<PyReadonlyArray2<f64>>)>,
    k_values: Vec<usize>,
    align_methods: Vec<String>,
    distance_metrics: Vec<String>,
    distance_thresholds: Vec<f64>,
    lookback_values: Vec<usize>,
    n_threads: usize,
) -> PyResult<(Py<PyArray2<f64>>, Vec<String>)> {
    if segments_data.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err("segments_data不能为空"));
    }
    
    // 解析各时段数据
    let mut segment_info: Vec<(String, usize, Vec<Array2<f64>>, Vec<Array2<f64>>)> = Vec::new();
    let mut n_stocks = 0;
    
    for (seg_name, feat_list, micro_list) in segments_data {
        if feat_list.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!("时段 {} 的features_list为空", seg_name)));
        }
        
        let n_days = feat_list.len();
        let stocks = feat_list[0].as_array().nrows();
        
        if n_stocks == 0 {
            n_stocks = stocks;
        } else if stocks != n_stocks {
            return Err(pyo3::exceptions::PyValueError::new_err(
                format!("股票数不一致: {} vs {}", n_stocks, stocks)
            ));
        }
        
        let feat_arrays: Vec<Array2<f64>> = feat_list.iter().map(|f| f.as_array().to_owned()).collect();
        let micro_arrays: Vec<Array2<f64>> = micro_list.iter().map(|f| f.as_array().to_owned()).collect();
        
        segment_info.push((seg_name, n_days, feat_arrays, micro_arrays));
    }
    
    let n_segments = segment_info.len();
    let n_micro = if !segment_info.is_empty() && !segment_info[0].3.is_empty() {
        segment_info[0].3[0].ncols()
    } else {
        0
    };
    let n_factors = 16 + n_micro * 2;
    
    // 限制线程数
    let actual_threads = n_threads.min(10).max(1);
    
    // 构建参数组合
    let mut param_combos: Vec<(usize, String, String, f64, usize)> = Vec::new();
    for &k in &k_values {
        for align in &align_methods {
            for dist in &distance_metrics {
                for &thresh in &distance_thresholds {
                    for &lb in &lookback_values {
                        param_combos.push((k, align.clone(), dist.clone(), thresh, lb));
                    }
                }
            }
        }
    }
    let n_combos = param_combos.len();
    
    // 基础因子名
    let base_factor_names = vec![
        "theme_return_mean", "yesterday_theme_return", "theme_return_trend",
        "theme_size", "theme_amount", "theme_return", "theme_act_buy_ratio",
        "return_rank_in_theme", "distance_to_center",
        "switch_direction", "switch_strength", "theme_entropy",
        "theme_heat", "theme_momentum_persistence", "theme_diffusion", "leader_stability",
    ];
    let micro_metric_names = vec![
        "act_buy_ratio", "bid_ask_imbalance", "vol_per_trade",
        "spread_ratio", "depth_imbalance", "vwap_deviation",
        "act_buy_vol_ratio", "big_order_ratio",
    ];
    
    // 预计算各时段各k值的聚类结果
    let mut cluster_cache_by_segment: HashMap<String, HashMap<usize, Vec<CachedDayResult>>> = HashMap::new();
    
    for (seg_name, n_days, feat_arrays, _) in &segment_info {
        let mut k_cache: HashMap<usize, Vec<CachedDayResult>> = HashMap::new();
        let d = feat_arrays[0].ncols();
        
        for &k in &k_values {
            let mut day_results: Vec<CachedDayResult> = Vec::with_capacity(*n_days);
            for day_idx in 0..*n_days {
                let features = feat_arrays[day_idx].as_slice().unwrap_or(&[]);
                let result = cluster_single_day_cached(features, n_stocks, d, k);
                day_results.push(result);
            }
            k_cache.insert(k, day_results);
        }
        cluster_cache_by_segment.insert(seg_name.clone(), k_cache);
    }
    
    // 创建线程池
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(actual_threads)
        .build()
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("创建线程池失败: {}", e)))?;
    
    // 构建所有任务: (segment_idx, combo_idx, segment_name, k, align, dist, thresh, lb)
    let mut tasks: Vec<(usize, usize, String, usize, String, String, f64, usize)> = Vec::new();
    for (seg_idx, (seg_name, _, _, _)) in segment_info.iter().enumerate() {
        for (combo_idx, (k, align, dist, thresh, lb)) in param_combos.iter().enumerate() {
            tasks.push((seg_idx, combo_idx, seg_name.clone(), *k, align.clone(), dist.clone(), *thresh, *lb));
        }
    }
    
    // 并行计算所有任务
    let results: Vec<(usize, usize, Vec<f64>)> = pool.install(|| {
        tasks
            .into_par_iter()
            .map(|(seg_idx, combo_idx, seg_name, k, align, dist, thresh, lb)| {
                let k_cache = cluster_cache_by_segment.get(&seg_name).unwrap();
                let day_results = k_cache.get(&k).unwrap();
                let (_, _, feat_arrays, micro_arrays) = &segment_info[seg_idx];
                let n_days = feat_arrays.len();
                
                let result = compute_single_factor_set(
                    day_results,
                    feat_arrays,
                    micro_arrays,
                    n_stocks,
                    n_days,
                    &align,
                    &dist,
                    thresh,
                    lb,
                );
                (seg_idx, combo_idx, result)
            })
            .collect()
    });
    
    // 构建列名并整理输出
    let total_cols = n_segments * n_combos * n_factors;
    let mut col_names: Vec<String> = Vec::with_capacity(total_cols);
    let mut output = vec![f64::NAN; n_stocks * total_cols];
    
    // 按时段顺序生成列名和填充数据
    let mut global_col = 0;
    for (seg_idx, (seg_name, _, _, _)) in segment_info.iter().enumerate() {
        for (combo_idx, (k, align, dist, thresh, lb)) in param_combos.iter().enumerate() {
            let thresh_str = if *thresh < 0.0 { "None".to_string() } else { thresh.to_string() };
            let param_suffix = format!("__k{}_{}_{}_{}__{}", k, align, dist, thresh_str, seg_name);
            
            // 生成列名
            for name in &base_factor_names {
                col_names.push(format!("{}{}", name, param_suffix));
            }
            for m_idx in 0..n_micro {
                let m_name = micro_metric_names.get(m_idx).unwrap_or(&"unknown");
                col_names.push(format!("{}_deviation{}", m_name, param_suffix));
                col_names.push(format!("{}_deviation_change{}", m_name, param_suffix));
            }
            
            // 找到对应结果并填充
            if let Some((_, _, result)) = results.iter().find(|(si, ci, _)| *si == seg_idx && *ci == combo_idx) {
                for s in 0..n_stocks {
                    for f in 0..n_factors {
                        output[s * total_cols + global_col + f] = result[s * n_factors + f];
                    }
                }
            }
            global_col += n_factors;
        }
    }
    
    let arr = Array2::from_shape_vec((n_stocks, total_cols), output)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("数组形状错误: {}", e)))?;
    
    Ok((arr.into_pyarray(py).to_owned(), col_names))
}

/// 从分钟数据计算主题聚类因子（Rust内部做时段拆分和特征提取）
///
/// 优化点：
/// 1. Python只传入分钟原始数据，减少数据传输
/// 2. Rust内部做时段切片、特征提取、聚类、因子计算，全流程并行
/// 3. 各时段独立聚类
#[pyfunction]
#[pyo3(signature = (minute_data, segments, segment_bounds, k_values, align_methods, distance_metrics, distance_thresholds, lookback_values, n_threads))]
pub fn theme_cluster_factors_from_minute(
    py: Python,
    minute_data: PyReadonlyArrayDyn<f64>,  // shape: (n_days, n_minutes, n_stocks, n_fields)
    segments: Vec<String>,
    segment_bounds: Vec<(usize, usize)>,  // List[(start, end)]
    k_values: Vec<usize>,
    align_methods: Vec<String>,
    distance_metrics: Vec<String>,
    distance_thresholds: Vec<f64>,
    lookback_values: Vec<usize>,
    n_threads: usize,
) -> PyResult<(Py<PyArray2<f64>>, Vec<String>)> {
    let data = minute_data.as_array();
    let shape = data.shape();
    if shape.len() != 4 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "minute_data必须是4维数组 (n_days, n_minutes, n_stocks, n_fields)"
        ));
    }
    
    let n_days = shape[0];
    let n_minutes = shape[1];
    let n_stocks = shape[2];
    let n_fields = shape[3];
    
    if segments.len() != segment_bounds.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "segments和segment_bounds长度不一致"
        ));
    }
    
    // 将数据转为连续内存布局
    let data_vec: Vec<f64> = data.iter().cloned().collect();
    
    // 字段索引
    const FIELD_OPEN: usize = 0;
    const FIELD_CLOSE: usize = 1;
    const FIELD_AMOUNT: usize = 2;
    const FIELD_VOLUME: usize = 3;
    const FIELD_ACT_BUY_AMT: usize = 4;
    const FIELD_ACT_SELL_AMT: usize = 5;
    const FIELD_ACT_BUY_CNT: usize = 6;
    const FIELD_ACT_SELL_CNT: usize = 7;
    const FIELD_ACT_BUY_VOL: usize = 8;
    const FIELD_ACT_SELL_VOL: usize = 9;
    const FIELD_UP_TICK: usize = 10;
    const FIELD_DOWN_TICK: usize = 11;
    const FIELD_BID_SIZE1: usize = 12;
    const FIELD_ASK_SIZE1: usize = 13;
    const FIELD_BID_SIZE6: usize = 14;
    const FIELD_ASK_SIZE6: usize = 15;
    const FIELD_BID_VWAP: usize = 16;
    const FIELD_ASK_VWAP: usize = 17;
    const FIELD_BID_VOL1: usize = 18;
    const FIELD_ASK_VOL1: usize = 19;
    
    const N_FEATURES: usize = 12;
    const N_MICRO: usize = 8;
    const N_FACTORS: usize = 16 + N_MICRO * 2;  // 32
    
    // 提取单日单时段特征的函数
    let extract_features = |day_data: &[f64], start: usize, end: usize| -> Vec<f64> {
        let len = end - start;
        let mut features = vec![0.0f64; n_stocks * N_FEATURES];
        
        for s in 0..n_stocks {
            // 辅助函数：获取某字段某分钟的值
            let get_val = |field: usize, minute: usize| -> f64 {
                day_data[minute * n_stocks * n_fields + s * n_fields + field]
            };
            
            // 求和（跳过NaN）
            let sum_field = |field: usize| -> f64 {
                (start..end).map(|m| get_val(field, m)).filter(|&v| v.is_finite()).sum()
            };
            // 均值（跳过NaN）
            let mean_field = |field: usize| -> f64 {
                let vals: Vec<f64> = (start..end).map(|m| get_val(field, m)).filter(|&v| v.is_finite()).collect();
                if vals.is_empty() { f64::NAN } else { vals.iter().sum::<f64>() / vals.len() as f64 }
            };
            
            // 找最后一个有效的close值（用于daily_return）
            let last_valid_close = || -> (f64, usize) {
                for m in (start..end).rev() {
                    let c = get_val(FIELD_CLOSE, m);
                    if c.is_finite() { return (c, m); }
                }
                (f64::NAN, end - 1)
            };
            
            // 找最后一个有效的bid_vol1/ask_vol1位置
            let last_valid_pos = || -> usize {
                for m in (start..end).rev() {
                    let v = get_val(FIELD_BID_VOL1, m);
                    if v.is_finite() { return m; }
                }
                end - 1
            };
            
            // 0: daily_return = last_close / first_open - 1
            let first_open = get_val(FIELD_OPEN, start);
            let (last_close, _) = last_valid_close();
            features[s * N_FEATURES + 0] = if first_open > 0.0 && last_close.is_finite() {
                last_close / first_open - 1.0
            } else {
                f64::NAN
            };
            
            // 1: intraday_vol = std(log(close/close.shift(1)))
            if len > 1 {
                let mut log_rets = Vec::with_capacity(len - 1);
                for m in (start + 1)..end {
                    let prev_close = get_val(FIELD_CLOSE, m - 1);
                    let curr_close = get_val(FIELD_CLOSE, m);
                    if prev_close > 0.0 && curr_close > 0.0 {
                        log_rets.push((curr_close / prev_close).ln());
                    }
                }
                if log_rets.len() > 1 {
                    let mean: f64 = log_rets.iter().sum::<f64>() / log_rets.len() as f64;
                    let var: f64 = log_rets.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / log_rets.len() as f64;
                    features[s * N_FEATURES + 1] = var.sqrt();
                }
            }
            
            // 2: log_total_amt
            let total_amt = sum_field(FIELD_AMOUNT);
            features[s * N_FEATURES + 2] = if total_amt > 0.0 {
                total_amt.ln()
            } else {
                f64::NAN
            };
            
            // 3: act_buy_ratio
            let act_buy_amt = sum_field(FIELD_ACT_BUY_AMT);
            features[s * N_FEATURES + 3] = act_buy_amt / (total_amt + 1e-10);
            
            // 4: log_avg_trade_amt
            let total_cnt = sum_field(FIELD_ACT_BUY_CNT) + sum_field(FIELD_ACT_SELL_CNT);
            features[s * N_FEATURES + 4] = if total_amt > 0.0 && total_cnt > 0.0 {
                (total_amt / total_cnt).ln()
            } else {
                f64::NAN
            };
            
            // 5: min_ret_skew
            if len > 2 {
                let mut log_rets = Vec::with_capacity(len - 1);
                for m in (start + 1)..end {
                    let prev_close = get_val(FIELD_CLOSE, m - 1);
                    let curr_close = get_val(FIELD_CLOSE, m);
                    if prev_close > 0.0 && curr_close > 0.0 {
                        log_rets.push((curr_close / prev_close).ln());
                    }
                }
                if log_rets.len() > 2 {
                    let mean: f64 = log_rets.iter().sum::<f64>() / log_rets.len() as f64;
                    let std: f64 = {
                        let var: f64 = log_rets.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / log_rets.len() as f64;
                        var.sqrt()
                    };
                    if std > 1e-10 {
                        let skew: f64 = log_rets.iter().map(|x| ((x - mean) / std).powi(3)).sum::<f64>() / log_rets.len() as f64;
                        features[s * N_FEATURES + 5] = skew;
                    }
                }
            }
            
            // 6: up_tick_ratio
            let up_tick = sum_field(FIELD_UP_TICK);
            let down_tick = sum_field(FIELD_DOWN_TICK);
            let total_tick = up_tick + down_tick;
            features[s * N_FEATURES + 6] = if total_tick > 0.0 {
                up_tick / total_tick
            } else {
                0.5
            };
            
            // 7: bid_size1_mean
            features[s * N_FEATURES + 7] = mean_field(FIELD_BID_SIZE1);
            
            // 8: ask_size1_mean
            features[s * N_FEATURES + 8] = mean_field(FIELD_ASK_SIZE1);
            
            // 9: vwap_diff
            let bid_vwap = mean_field(FIELD_BID_VWAP);
            let ask_vwap = mean_field(FIELD_ASK_VWAP);
            features[s * N_FEATURES + 9] = bid_vwap - ask_vwap;
            
            // 10: last_bid_vol1 - 使用最后一个有效位置
            let last_pos = last_valid_pos();
            features[s * N_FEATURES + 10] = get_val(FIELD_BID_VOL1, last_pos);
            
            // 11: last_ask_vol1
            features[s * N_FEATURES + 11] = get_val(FIELD_ASK_VOL1, last_pos);
        }
        
        features
    };
    
    // 提取微观指标
    let extract_micro = |day_data: &[f64], start: usize, end: usize| -> Vec<f64> {
        let len = end - start;
        let mut micro = vec![0.0f64; n_stocks * N_MICRO];
        
        for s in 0..n_stocks {
            let get_val = |field: usize, minute: usize| -> f64 {
                day_data[minute * n_stocks * n_fields + s * n_fields + field]
            };
            let sum_field = |field: usize| -> f64 {
                (start..end).map(|m| get_val(field, m)).sum()
            };
            let mean_field = |field: usize| -> f64 {
                let sum: f64 = (start..end).map(|m| get_val(field, m)).sum();
                sum / len as f64
            };
            
            let total_amt = sum_field(FIELD_AMOUNT);
            let total_vol = sum_field(FIELD_ACT_BUY_VOL) + sum_field(FIELD_ACT_SELL_VOL);
            let total_cnt = sum_field(FIELD_ACT_BUY_CNT) + sum_field(FIELD_ACT_SELL_CNT);
            
            let act_buy_amt = sum_field(FIELD_ACT_BUY_AMT);
            let act_sell_amt = sum_field(FIELD_ACT_SELL_AMT);
            
            // 0: act_buy_ratio
            micro[s * N_MICRO + 0] = act_buy_amt / (total_amt + 1e-10);
            
            // 1: bid_ask_imbalance
            micro[s * N_MICRO + 1] = (act_buy_amt - act_sell_amt) / (total_amt + 1e-10);
            
            // 2: vol_per_trade
            micro[s * N_MICRO + 2] = if total_cnt > 0.0 && total_vol > 0.0 {
                (total_vol / total_cnt).ln()
            } else {
                f64::NAN
            };
            
            // 3: spread_ratio
            let bid_size1 = sum_field(FIELD_BID_SIZE1);
            let ask_size1 = sum_field(FIELD_ASK_SIZE1);
            let depth_total = bid_size1 + ask_size1;
            micro[s * N_MICRO + 3] = (ask_size1 - bid_size1) / (depth_total + 1e-10);
            
            // 4: depth_imbalance
            let bid_size6 = sum_field(FIELD_BID_SIZE6);
            let ask_size6 = sum_field(FIELD_ASK_SIZE6);
            let depth6_total = bid_size6 + ask_size6;
            micro[s * N_MICRO + 4] = (bid_size6 - ask_size6) / (depth6_total + 1e-10);
            
            // 5: vwap_deviation
            let bid_vwap = mean_field(FIELD_BID_VWAP);
            let ask_vwap = mean_field(FIELD_ASK_VWAP);
            micro[s * N_MICRO + 5] = bid_vwap - ask_vwap;
            
            // 6: act_buy_vol_ratio
            let act_buy_vol = sum_field(FIELD_ACT_BUY_VOL);
            micro[s * N_MICRO + 6] = act_buy_vol / (total_vol + 1e-10);
            
            // 7: big_order_ratio
            let median_amt: f64 = {
                let mut amts: Vec<f64> = (start..end).map(|m| get_val(FIELD_AMOUNT, m)).collect();
                amts.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                if amts.is_empty() { 0.0 } else { amts[amts.len() / 2] }
            };
            let big_order_amt: f64 = (start..end)
                .filter(|&m| get_val(FIELD_AMOUNT, m) > median_amt)
                .map(|m| get_val(FIELD_AMOUNT, m))
                .sum();
            micro[s * N_MICRO + 7] = big_order_amt / (total_amt + 1e-10);
        }
        
        micro
    };
    
    // 限制线程数
    let actual_threads = n_threads.min(10).max(1);
    
    // 构建参数组合
    let mut param_combos: Vec<(usize, String, String, f64, usize)> = Vec::new();
    for &k in &k_values {
        for align in &align_methods {
            for dist in &distance_metrics {
                for &thresh in &distance_thresholds {
                    for &lb in &lookback_values {
                        param_combos.push((k, align.clone(), dist.clone(), thresh, lb));
                    }
                }
            }
        }
    }
    let n_combos = param_combos.len();
    let n_segments = segments.len();
    
    // 基础因子名
    let base_factor_names = vec![
        "theme_return_mean", "yesterday_theme_return", "theme_return_trend",
        "theme_size", "theme_amount", "theme_return", "theme_act_buy_ratio",
        "return_rank_in_theme", "distance_to_center",
        "switch_direction", "switch_strength", "theme_entropy",
        "theme_heat", "theme_momentum_persistence", "theme_diffusion", "leader_stability",
    ];
    let micro_metric_names = vec![
        "act_buy_ratio", "bid_ask_imbalance", "vol_per_trade",
        "spread_ratio", "depth_imbalance", "vwap_deviation",
        "act_buy_vol_ratio", "big_order_ratio",
    ];
    
    // 预计算各时段各天的特征和微观指标
    let mut seg_features: Vec<Vec<Vec<f64>>> = vec![vec![Vec::new(); n_days]; n_segments];
    let mut seg_micro: Vec<Vec<Vec<f64>>> = vec![vec![Vec::new(); n_days]; n_segments];
    
    for seg_idx in 0..n_segments {
        let (start, end) = segment_bounds[seg_idx];
        for day_idx in 0..n_days {
            // 提取该天数据
            let day_offset = day_idx * n_minutes * n_stocks * n_fields;
            let day_data = &data_vec[day_offset..day_offset + n_minutes * n_stocks * n_fields];
            
            seg_features[seg_idx][day_idx] = extract_features(day_data, start, end);
            seg_micro[seg_idx][day_idx] = extract_micro(day_data, start, end);
        }
    }
    
    // 预计算各时段各k值的聚类结果
    let mut cluster_cache: Vec<HashMap<usize, Vec<CachedDayResult>>> = Vec::with_capacity(n_segments);
    for seg_idx in 0..n_segments {
        let mut k_cache: HashMap<usize, Vec<CachedDayResult>> = HashMap::new();
        
        for &k in &k_values {
            let mut day_results: Vec<CachedDayResult> = Vec::with_capacity(n_days);
            
            for day_idx in 0..n_days {
                let features = &seg_features[seg_idx][day_idx];
                let result = cluster_single_day_cached(features, n_stocks, N_FEATURES, k);
                day_results.push(result);
            }
            
            k_cache.insert(k, day_results);
        }
        cluster_cache.push(k_cache);
    }
    
    // 创建线程池
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(actual_threads)
        .build()
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("创建线程池失败: {}", e)))?;
    
    // 构建所有任务
    let mut tasks: Vec<(usize, usize, usize, String, String, f64, usize)> = Vec::new();
    for seg_idx in 0..n_segments {
        for (combo_idx, (k, align, dist, thresh, lb)) in param_combos.iter().enumerate() {
            tasks.push((seg_idx, combo_idx, *k, align.clone(), dist.clone(), *thresh, *lb));
        }
    }
    
    // 并行计算
    let results: Vec<(usize, usize, Vec<f64>)> = pool.install(|| {
        tasks
            .into_par_iter()
            .map(|(seg_idx, combo_idx, k, align, dist, thresh, lb)| {
                let k_cache = &cluster_cache[seg_idx];
                let day_results = k_cache.get(&k).unwrap();
                
                // 构建features_arrays和micro_arrays
                let features_arrays: Vec<Array2<f64>> = seg_features[seg_idx].iter()
                    .map(|f| Array2::from_shape_vec((n_stocks, N_FEATURES), f.clone()).unwrap())
                    .collect();
                let micro_arrays: Vec<Array2<f64>> = seg_micro[seg_idx].iter()
                    .map(|m| Array2::from_shape_vec((n_stocks, N_MICRO), m.clone()).unwrap())
                    .collect();
                
                let result = compute_single_factor_set(
                    day_results,
                    &features_arrays,
                    &micro_arrays,
                    n_stocks,
                    n_days,
                    &align,
                    &dist,
                    thresh,
                    lb,
                );
                (seg_idx, combo_idx, result)
            })
            .collect()
    });
    
    // 构建输出
    let total_cols = n_segments * n_combos * N_FACTORS;
    let mut col_names: Vec<String> = Vec::with_capacity(total_cols);
    let mut output = vec![f64::NAN; n_stocks * total_cols];
    
    let mut global_col = 0;
    for seg_idx in 0..n_segments {
        let seg_name = &segments[seg_idx];
        for (combo_idx, (k, align, dist, thresh, lb)) in param_combos.iter().enumerate() {
            let thresh_str = if *thresh < 0.0 { "None".to_string() } else { thresh.to_string() };
            let param_suffix = format!("__k{}_{}_{}_{}_lb{}__{}", k, align, dist, thresh_str, lb, seg_name);
            
            // 列名
            for name in &base_factor_names {
                col_names.push(format!("{}{}", name, param_suffix));
            }
            for m_idx in 0..N_MICRO {
                let m_name = micro_metric_names.get(m_idx).unwrap_or(&"unknown");
                col_names.push(format!("{}_deviation{}", m_name, param_suffix));
                col_names.push(format!("{}_deviation_change{}", m_name, param_suffix));
            }
            
            // 填充数据
            if let Some((_, _, result)) = results.iter().find(|(si, ci, _)| *si == seg_idx && *ci == combo_idx) {
                for s in 0..n_stocks {
                    for f in 0..N_FACTORS {
                        output[s * total_cols + global_col + f] = result[s * N_FACTORS + f];
                    }
                }
            }
            global_col += N_FACTORS;
        }
    }
    
    let arr = Array2::from_shape_vec((n_stocks, total_cols), output)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("数组形状错误: {}", e)))?;
    
    Ok((arr.into_pyarray(py).to_owned(), col_names))
}

/// 清除聚类缓存
#[pyfunction]
pub fn clear_theme_cluster_cache() {
    let mut cache = CLUSTER_CACHE.lock().unwrap();
    cache.clear();
}

/// 获取主题聚类因子的列名（不计算因子，仅生成列名）
#[pyfunction]
#[pyo3(signature = (segments, k_values, align_methods, distance_metrics, distance_thresholds, lookback_values))]
pub fn get_theme_cluster_factor_names(
    segments: Vec<String>,
    k_values: Vec<usize>,
    align_methods: Vec<String>,
    distance_metrics: Vec<String>,
    distance_thresholds: Vec<f64>,
    lookback_values: Vec<usize>,
) -> Vec<String> {
    const N_FEATURES: usize = 12;
    const N_MICRO: usize = 8;
    const N_FACTORS: usize = 16 + N_MICRO * 2;  // 32
    
    let base_factor_names = vec![
        "theme_return_mean", "yesterday_theme_return", "theme_return_trend",
        "theme_size", "theme_amount", "theme_return", "theme_act_buy_ratio",
        "return_rank_in_theme", "distance_to_center",
        "switch_direction", "switch_strength", "theme_entropy",
        "theme_heat", "theme_momentum_persistence", "theme_diffusion", "leader_stability",
    ];
    let micro_metric_names = vec![
        "act_buy_ratio", "bid_ask_imbalance", "vol_per_trade",
        "spread_ratio", "depth_imbalance", "vwap_deviation",
        "act_buy_vol_ratio", "big_order_ratio",
    ];
    
    // 构建参数组合
    let mut param_combos: Vec<(usize, String, String, f64, usize)> = Vec::new();
    for &k in &k_values {
        for align in &align_methods {
            for dist in &distance_metrics {
                for &thresh in &distance_thresholds {
                    for &lb in &lookback_values {
                        param_combos.push((k, align.clone(), dist.clone(), thresh, lb));
                    }
                }
            }
        }
    }
    
    let mut col_names: Vec<String> = Vec::new();
    
    for seg_name in &segments {
        for (k, align, dist, thresh, lb) in &param_combos {
            let thresh_str = if *thresh < 0.0 { "None".to_string() } else { thresh.to_string() };
            let param_suffix = format!("__k{}_{}_{}_{}_lb{}__{}", k, align, dist, thresh_str, lb, seg_name);
            
            // 基础因子列名
            for name in &base_factor_names {
                col_names.push(format!("{}{}", name, param_suffix));
            }
            // 微观指标偏差列名
            for m_name in &micro_metric_names {
                col_names.push(format!("{}_deviation{}", m_name, param_suffix));
                col_names.push(format!("{}_deviation_change{}", m_name, param_suffix));
            }
        }
    }
    
    col_names
}
