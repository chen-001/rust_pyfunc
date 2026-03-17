/// 主题聚类因子计算模块
///
/// 复现Python中 5_main.py 的 compute_all_factors 流程:
///   1) 截面标准化 + KMeans聚类
///   2) 跨日标签对齐（匈牙利 / 重叠）
///   3) 因子计算（动量 / 强度 / 排名 / 切换 / 熵 / 演化 / 微观偏离）
///
/// 输入: 多日特征矩阵 (n_stocks x n_features) 和微观指标矩阵
/// 输出: 因子矩阵 (n_stocks x n_factors)

use ndarray::{s, Array1, Array2, Axis};
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;

// ============================================================
// 辅助数学函数
// ============================================================

/// z-score截面标准化（按列标准化）
fn zscore_standardize(data: &Array2<f64>) -> Array2<f64> {
    let (n, d) = (data.nrows(), data.ncols());
    let mut result = Array2::<f64>::zeros((n, d));
    for j in 0..d {
        let col = data.column(j);
        let valid: Vec<f64> = col.iter().filter(|x| x.is_finite()).copied().collect();
        if valid.is_empty() {
            for i in 0..n {
                result[[i, j]] = f64::NAN;
            }
            continue;
        }
        let mean = valid.iter().sum::<f64>() / valid.len() as f64;
        let var = valid.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / valid.len() as f64;
        let std = var.sqrt().max(1e-15);
        for i in 0..n {
            if data[[i, j]].is_finite() {
                result[[i, j]] = (data[[i, j]] - mean) / std;
            } else {
                result[[i, j]] = f64::NAN;
            }
        }
    }
    result
}

/// 欧氏距离矩阵 (k1 x d) vs (k2 x d) -> (k1 x k2)
fn cdist_euclidean(a: &Array2<f64>, b: &Array2<f64>) -> Array2<f64> {
    let (k1, k2) = (a.nrows(), b.nrows());
    let mut result = Array2::<f64>::zeros((k1, k2));
    for i in 0..k1 {
        for j in 0..k2 {
            let mut sum = 0.0;
            for d in 0..a.ncols() {
                let diff = a[[i, d]] - b[[j, d]];
                sum += diff * diff;
            }
            result[[i, j]] = sum.sqrt();
        }
    }
    result
}

/// 余弦距离矩阵 = 1 - cosine_similarity
fn cdist_cosine(a: &Array2<f64>, b: &Array2<f64>) -> Array2<f64> {
    let (k1, k2) = (a.nrows(), b.nrows());
    let mut result = Array2::<f64>::zeros((k1, k2));
    for i in 0..k1 {
        for j in 0..k2 {
            let mut dot = 0.0;
            let mut na = 0.0;
            let mut nb = 0.0;
            for d in 0..a.ncols() {
                dot += a[[i, d]] * b[[j, d]];
                na += a[[i, d]] * a[[i, d]];
                nb += b[[j, d]] * b[[j, d]];
            }
            let denom = (na * nb).sqrt();
            if denom < 1e-15 {
                result[[i, j]] = 1.0;
            } else {
                result[[i, j]] = 1.0 - dot / denom;
            }
        }
    }
    result
}

/// 马氏距离矩阵
/// 将a和b堆叠后计算协方差矩阵, 求逆后计算马氏距离
fn cdist_mahalanobis(a: &Array2<f64>, b: &Array2<f64>) -> Array2<f64> {
    let d = a.ncols();
    let k1 = a.nrows();
    let k2 = b.nrows();
    let total = k1 + k2;

    // 堆叠计算协方差矩阵
    let mut combined = Array2::<f64>::zeros((total, d));
    for i in 0..k1 {
        for j in 0..d {
            combined[[i, j]] = a[[i, j]];
        }
    }
    for i in 0..k2 {
        for j in 0..d {
            combined[[k1 + i, j]] = b[[i, j]];
        }
    }

    // 计算均值
    let mut mean = vec![0.0f64; d];
    for j in 0..d {
        for i in 0..total {
            mean[j] += combined[[i, j]];
        }
        mean[j] /= total as f64;
    }

    // 计算协方差矩阵 + 正则化
    let mut cov = Array2::<f64>::zeros((d, d));
    for i in 0..total {
        for j1 in 0..d {
            for j2 in 0..d {
                cov[[j1, j2]] += (combined[[i, j1]] - mean[j1]) * (combined[[i, j2]] - mean[j2]);
            }
        }
    }
    let n_f = if total > 1 { (total - 1) as f64 } else { 1.0 };
    for j1 in 0..d {
        for j2 in 0..d {
            cov[[j1, j2]] /= n_f;
        }
        cov[[j1, j1]] += 1e-6; // 正则化
    }

    // 矩阵求逆 (Gauss-Jordan)
    let cov_inv = match invert_matrix(&cov) {
        Some(inv) => inv,
        None => return cdist_euclidean(a, b), // 求逆失败回退欧氏距离
    };

    // 计算马氏距离: d(x,y) = sqrt((x-y)^T * cov_inv * (x-y))
    let mut result = Array2::<f64>::zeros((k1, k2));
    for i in 0..k1 {
        for j in 0..k2 {
            let mut diff = vec![0.0f64; d];
            for dd in 0..d {
                diff[dd] = a[[i, dd]] - b[[j, dd]];
            }
            let mut val = 0.0;
            for d1 in 0..d {
                let mut tmp = 0.0;
                for d2 in 0..d {
                    tmp += cov_inv[[d1, d2]] * diff[d2];
                }
                val += diff[d1] * tmp;
            }
            result[[i, j]] = val.max(0.0).sqrt();
        }
    }
    result
}

/// Gauss-Jordan矩阵求逆
fn invert_matrix(mat: &Array2<f64>) -> Option<Array2<f64>> {
    let n = mat.nrows();
    if n != mat.ncols() {
        return None;
    }
    // 增广矩阵 [mat | I]
    let mut aug = Array2::<f64>::zeros((n, 2 * n));
    for i in 0..n {
        for j in 0..n {
            aug[[i, j]] = mat[[i, j]];
        }
        aug[[i, n + i]] = 1.0;
    }

    for col in 0..n {
        // 找主元
        let mut max_row = col;
        let mut max_val = aug[[col, col]].abs();
        for row in col + 1..n {
            if aug[[row, col]].abs() > max_val {
                max_val = aug[[row, col]].abs();
                max_row = row;
            }
        }
        if max_val < 1e-12 {
            return None;
        }
        // 交换行
        if max_row != col {
            for j in 0..2 * n {
                let tmp = aug[[col, j]];
                aug[[col, j]] = aug[[max_row, j]];
                aug[[max_row, j]] = tmp;
            }
        }
        // 归一化
        let pivot = aug[[col, col]];
        for j in 0..2 * n {
            aug[[col, j]] /= pivot;
        }
        // 消元
        for row in 0..n {
            if row != col {
                let factor = aug[[row, col]];
                for j in 0..2 * n {
                    aug[[row, j]] -= factor * aug[[col, j]];
                }
            }
        }
    }

    let mut inv = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            inv[[i, j]] = aug[[i, n + j]];
        }
    }
    Some(inv)
}

/// 距离矩阵（支持 euclidean / cosine / mahalanobis）
fn compute_distance_matrix(a: &Array2<f64>, b: &Array2<f64>, metric: &str) -> Array2<f64> {
    match metric {
        "cosine" => cdist_cosine(a, b),
        "mahalanobis" => cdist_mahalanobis(a, b),
        _ => cdist_euclidean(a, b),
    }
}

/// 匈牙利算法（Jonker-Volgenant方式的简化实现）
/// 输入: 代价矩阵 (n x n)
/// 输出: Vec<(row, col)> 的最优匹配
fn hungarian_match(cost: &Array2<f64>) -> Vec<(usize, usize)> {
    let n = cost.nrows();
    let m = cost.ncols();
    // 使用简单的贪心+调整实现（对于K<=30足够）
    // 完整匈牙利算法
    let size = n.max(m);
    let mut u = vec![0.0f64; size + 1];
    let mut v = vec![0.0f64; size + 1];
    let mut p = vec![0usize; size + 1]; // p[j] = 分配给列j的行
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
// KMeans聚类（简化实现）
// ============================================================

/// KMeans++初始化
fn kmeans_pp_init(data: &Array2<f64>, k: usize, seed: u64) -> Array2<f64> {
    let n = data.nrows();
    let d = data.ncols();
    let mut centers = Array2::<f64>::zeros((k, d));
    let mut rng_state = seed;

    // 简单的LCG随机数生成器
    let mut next_rand = || -> f64 {
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (rng_state >> 33) as f64 / (1u64 << 31) as f64
    };

    // 第一个中心随机选取
    let first = (next_rand() * n as f64) as usize % n;
    centers.row_mut(0).assign(&data.row(first));

    let mut dists = vec![f64::MAX; n];
    for c in 1..k {
        // 更新距离
        for i in 0..n {
            let mut d2 = 0.0;
            for j in 0..d {
                let diff = data[[i, j]] - centers[[c - 1, j]];
                d2 += diff * diff;
            }
            dists[i] = dists[i].min(d2);
        }
        let total: f64 = dists.iter().sum();
        if total < 1e-15 {
            centers.row_mut(c).assign(&data.row(0));
            continue;
        }
        let target = next_rand() * total;
        let mut cum = 0.0;
        let mut chosen = 0;
        for i in 0..n {
            cum += dists[i];
            if cum >= target {
                chosen = i;
                break;
            }
        }
        centers.row_mut(c).assign(&data.row(chosen));
    }
    centers
}

/// KMeans聚类
/// 返回 (labels: Vec<usize>, centers: Array2<f64>)
/// labels中类别从0开始
fn kmeans(data: &Array2<f64>, k: usize, n_init: usize, max_iter: usize) -> (Vec<usize>, Array2<f64>) {
    let n = data.nrows();
    let d = data.ncols();

    let mut best_labels = vec![0usize; n];
    let mut best_centers = Array2::<f64>::zeros((k, d));
    let mut best_inertia = f64::MAX;

    for init in 0..n_init {
        let seed = 42 + init as u64 * 1000;
        let mut centers = kmeans_pp_init(data, k, seed);
        let mut labels = vec![0usize; n];

        for _iter in 0..max_iter {
            // 分配步骤
            let mut changed = false;
            for i in 0..n {
                let mut min_dist = f64::MAX;
                let mut min_c = 0;
                for c in 0..k {
                    let mut dist = 0.0;
                    for j in 0..d {
                        let diff = data[[i, j]] - centers[[c, j]];
                        dist += diff * diff;
                    }
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
            let mut new_centers = Array2::<f64>::zeros((k, d));
            let mut counts = vec![0usize; k];
            for i in 0..n {
                let c = labels[i];
                counts[c] += 1;
                for j in 0..d {
                    new_centers[[c, j]] += data[[i, j]];
                }
            }
            for c in 0..k {
                if counts[c] > 0 {
                    for j in 0..d {
                        new_centers[[c, j]] /= counts[c] as f64;
                    }
                }
            }
            centers = new_centers;

            if !changed {
                break;
            }
        }

        // 计算惯性
        let mut inertia = 0.0;
        for i in 0..n {
            let c = labels[i];
            for j in 0..d {
                let diff = data[[i, j]] - centers[[c, j]];
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

/// 聚类统计结果
#[derive(Clone)]
struct ClusterStats {
    size: Vec<f64>,            // 每个cluster的股票数
    mean_return: Vec<f64>,      // 平均收益率
    std_return: Vec<f64>,       // 收益率标准差
    total_amt: Vec<f64>,        // 总成交额
    mean_act_buy_ratio: Vec<f64>, // 平均主动买入占比
    mean_intraday_vol: Vec<f64>,  // 平均日内波动率
}

/// 计算聚类统计量
/// features列顺序: daily_return(0), intraday_vol(1), log_total_amt(2), act_buy_ratio(3), ...
fn compute_cluster_stats(features: &Array2<f64>, labels: &[usize], k: usize) -> ClusterStats {
    let n = features.nrows();
    let mut stats = ClusterStats {
        size: vec![0.0; k],
        mean_return: vec![0.0; k],
        std_return: vec![0.0; k],
        total_amt: vec![0.0; k],
        mean_act_buy_ratio: vec![0.0; k],
        mean_intraday_vol: vec![0.0; k],
    };

    // 特征列索引 (与Python 1_daily_features.py 对应)
    // 0: daily_return, 1: intraday_vol, 2: log_total_amt, 3: act_buy_ratio
    let col_ret = 0;
    let col_vol = 1;
    let col_amt = 2;
    let col_buy = 3;

    let mut ret_vals: Vec<Vec<f64>> = (0..k).map(|_| Vec::new()).collect();

    for i in 0..n {
        let c = labels[i];
        if c < k {
            stats.size[c] += 1.0;
            let ret = features[[i, col_ret]];
            if ret.is_finite() {
                ret_vals[c].push(ret);
                stats.mean_return[c] += ret;
            }
            let amt = features[[i, col_amt]];
            if amt.is_finite() {
                stats.total_amt[c] += amt;
            }
            let buy = features[[i, col_buy]];
            if buy.is_finite() {
                stats.mean_act_buy_ratio[c] += buy;
            }
            let vol = features[[i, col_vol]];
            if vol.is_finite() {
                stats.mean_intraday_vol[c] += vol;
            }
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
// 单日聚类结果
// ============================================================

#[derive(Clone)]
struct DayClusterResult {
    labels: Vec<usize>,           // 每只股票的聚类标签 (0-based)
    centers: Array2<f64>,         // 聚类中心 (k x d) (标准化空间)
    stats: ClusterStats,          // 统计量
    k: usize,
    features_clean: Array2<f64>,  // 清洗后的特征
    valid_indices: Vec<usize>,    // 有效行索引（相对于输入矩阵）
}

/// 单日聚类
fn cluster_single_day(features: &Array2<f64>, k: usize) -> DayClusterResult {
    let n = features.nrows();
    let d = features.ncols();

    // 找出没有NaN的行
    let mut valid_indices = Vec::new();
    for i in 0..n {
        let mut has_nan = false;
        for j in 0..d {
            if !features[[i, j]].is_finite() {
                has_nan = true;
                break;
            }
        }
        if !has_nan {
            valid_indices.push(i);
        }
    }

    let n_valid = valid_indices.len();
    let actual_k = k.min(n_valid);

    // 构建清洗后的特征矩阵
    let mut clean = Array2::<f64>::zeros((n_valid, d));
    for (idx, &orig) in valid_indices.iter().enumerate() {
        for j in 0..d {
            clean[[idx, j]] = features[[orig, j]];
        }
    }

    // 标准化
    let scaled = zscore_standardize(&clean);

    // KMeans聚类
    let (labels_clean, centers) = kmeans(&scaled, actual_k, 10, 300);

    // 映射回原始索引的标签 (NaN行标签设为 usize::MAX)
    let mut full_labels = vec![usize::MAX; n];
    for (idx, &orig) in valid_indices.iter().enumerate() {
        full_labels[orig] = labels_clean[idx];
    }

    let stats = compute_cluster_stats(&clean, &labels_clean, actual_k);

    DayClusterResult {
        labels: full_labels,
        centers,
        stats,
        k: actual_k,
        features_clean: clean,
        valid_indices,
    }
}

// ============================================================
// 跨日对齐
// ============================================================

/// 重叠匹配（基于Jaccard相似度的贪心匹配）
fn overlap_match_fn(
    labels_t: &[usize],
    labels_t1: &[usize],
    k: usize,
    n_stocks: usize,
) -> Vec<(usize, usize)> {
    let mut mapping = Vec::new();
    let mut used_t1 = vec![false; k];

    for i in 0..k {
        let stocks_t: std::collections::HashSet<usize> = (0..n_stocks)
            .filter(|&s| labels_t[s] == i)
            .collect();
        let mut best_j = None;
        let mut best_jaccard = -1.0f64;

        for j in 0..k {
            if used_t1[j] {
                continue;
            }
            let stocks_t1: std::collections::HashSet<usize> = (0..n_stocks)
                .filter(|&s| labels_t1[s] == j)
                .collect();
            let intersection = stocks_t.intersection(&stocks_t1).count();
            let union = stocks_t.union(&stocks_t1).count();
            let jaccard = intersection as f64 / (union as f64 + 1e-10);
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
fn align_two_days(
    curr: &DayClusterResult,
    prev: &DayClusterResult,
    method: &str,
    distance_metric: &str,
    distance_threshold: f64, // <= 0 表示无阈值
    n_stocks: usize,
) -> Vec<usize> {
    // mapping[i] = 当日cluster i 映射到的 prev cluster id
    let k = curr.k;

    let mapping_pairs: Vec<(usize, usize)> = match method {
        "overlap" => overlap_match_fn(&curr.labels, &prev.labels, k, n_stocks),
        _ => {
            // hungarian (default)
            let cost = compute_distance_matrix(&curr.centers, &prev.centers, distance_metric);
            hungarian_match(&cost)
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
        let cost = compute_distance_matrix(&curr.centers, &prev.centers, distance_metric);
        // 计算特征标准差的均值
        let mut std_sum = 0.0;
        let d = curr.centers.ncols();
        for j in 0..d {
            let col = curr.centers.column(j);
            let mean = col.mean().unwrap_or(0.0);
            let var: f64 = col.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / k as f64;
            std_sum += var.sqrt();
        }
        let feature_std = std_sum / d as f64;
        let threshold = distance_threshold * feature_std;

        let max_id = *mapping.iter().max().unwrap_or(&0);
        let mut new_id = max_id + 1;

        for i in 0..k {
            let min_dist = (0..prev.k)
                .map(|j| cost[[i, j]])
                .fold(f64::MAX, f64::min);
            if min_dist > threshold {
                mapping[i] = new_id;
                new_id += 1;
            }
        }
    }

    mapping
}

/// 对齐多日聚类结果
/// 返回每天对齐后的标签 aligned_labels[day][stock]
fn align_clusters_multi_days(
    day_results: &[DayClusterResult],
    method: &str,
    distance_metric: &str,
    distance_threshold: f64,
    n_stocks: usize,
) -> Vec<Vec<usize>> {
    let n_days = day_results.len();
    let mut aligned_labels = Vec::with_capacity(n_days);

    // 第一天直接使用原始标签
    aligned_labels.push(day_results[0].labels.clone());

    for d in 1..n_days {
        let mapping = align_two_days(
            &day_results[d],
            &day_results[d - 1],
            method,
            distance_metric,
            distance_threshold,
            n_stocks,
        );
        // 应用映射
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
/// theme_history: (n_stocks, n_days), stats各天的统计量, lookback
fn factor_theme_momentum(
    theme_history: &Array2<usize>,     // (n_stocks x n_days)
    stats_vec: &[ClusterStats],         // 每天的统计量
    lookback: usize,
) -> Array2<f64> {
    let n_stocks = theme_history.nrows();
    let n_days = theme_history.ncols();
    let lb = lookback.min(n_days);
    let start_day = n_days - lb;

    let mut result = Array2::<f64>::from_elem((n_stocks, 3), f64::NAN);

    for s in 0..n_stocks {
        let mut returns = Vec::new();
        let mut xs = Vec::new();

        for (idx, d) in (start_day..n_days).enumerate() {
            let theme = theme_history[[s, d]];
            if theme != usize::MAX && theme < stats_vec[d].mean_return.len() {
                let ret = stats_vec[d].mean_return[theme];
                returns.push(ret);
                xs.push(idx as f64);
            }
        }

        if !returns.is_empty() {
            // F4: 平均收益
            let mean: f64 = returns.iter().sum::<f64>() / returns.len() as f64;
            result[[s, 0]] = mean;
        }

        // F5: 倒数第二天的主题收益
        if n_days >= 2 {
            let d = n_days - 2;
            if d >= start_day {
                let theme = theme_history[[s, d]];
                if theme != usize::MAX && theme < stats_vec[d].mean_return.len() {
                    result[[s, 1]] = stats_vec[d].mean_return[theme];
                }
            }
        }

        // F6: 趋势斜率
        if returns.len() >= 2 {
            result[[s, 2]] = linregress_slope(&xs, &returns);
        }
    }

    result
}

/// F7-F10: 主题强度因子
fn factor_theme_strength(
    theme_labels: &[usize],   // 目标日的主题标签
    stats: &ClusterStats,
) -> Array2<f64> {
    let n = theme_labels.len();
    let mut result = Array2::<f64>::from_elem((n, 4), f64::NAN);

    for s in 0..n {
        let theme = theme_labels[s];
        if theme != usize::MAX && theme < stats.size.len() {
            result[[s, 0]] = stats.size[theme];           // F7
            result[[s, 1]] = stats.total_amt[theme];      // F8
            result[[s, 2]] = stats.mean_return[theme];    // F9
            result[[s, 3]] = stats.mean_act_buy_ratio[theme]; // F10
        }
    }

    result
}

/// F11-F12: 主题内部排名因子
fn factor_theme_rank(
    features: &Array2<f64>,      // 目标日原始特征
    theme_labels: &[usize],      // 目标日主题标签
    centers: &Array2<f64>,       // 目标日聚类中心(标准化空间)
    k: usize,
) -> Array2<f64> {
    let n = features.nrows();
    let d = features.ncols();
    let mut result = Array2::<f64>::from_elem((n, 2), f64::NAN);

    // F11: 主题内收益率排名（使用第0列 daily_return）
    // 先按主题分组
    let mut theme_stocks: Vec<Vec<usize>> = vec![Vec::new(); k];
    for s in 0..n {
        let theme = theme_labels[s];
        if theme != usize::MAX && theme < k {
            theme_stocks[theme].push(s);
        }
    }

    for c in 0..k {
        let stocks = &theme_stocks[c];
        if stocks.len() <= 1 {
            if stocks.len() == 1 {
                result[[stocks[0], 0]] = 0.5;
            }
            continue;
        }
        // 收集收益率
        let mut rets: Vec<(usize, f64)> = stocks
            .iter()
            .map(|&s| (s, features[[s, 0]]))
            .filter(|(_, r)| r.is_finite())
            .collect();
        rets.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        let cnt = rets.len();
        for (rank, &(s, _)) in rets.iter().enumerate() {
            result[[s, 0]] = rank as f64 / (cnt - 1).max(1) as f64;
        }
    }

    // F12: 与主题中心距离（标准化空间）
    let scaled = zscore_standardize(features);
    for s in 0..n {
        let theme = theme_labels[s];
        if theme != usize::MAX && theme < k && theme < centers.nrows() {
            let mut dist = 0.0;
            let mut valid = true;
            for j in 0..d.min(centers.ncols()) {
                if !scaled[[s, j]].is_finite() {
                    valid = false;
                    break;
                }
                let diff = scaled[[s, j]] - centers[[theme, j]];
                dist += diff * diff;
            }
            if valid {
                result[[s, 1]] = dist.sqrt();
            }
        }
    }

    result
}

/// F13-F14: 主题切换因子
fn factor_theme_switch(
    theme_history: &Array2<usize>, // (n_stocks x n_days)
    stats_vec: &[ClusterStats],
) -> Array2<f64> {
    let n = theme_history.nrows();
    let n_days = theme_history.ncols();
    let mut result = Array2::<f64>::from_elem((n, 2), f64::NAN);

    if n_days < 2 {
        return result;
    }

    let curr_day = n_days - 1;
    let prev_day = n_days - 2;

    for s in 0..n {
        let curr_theme = theme_history[[s, curr_day]];
        let prev_theme = theme_history[[s, prev_day]];

        if curr_theme == usize::MAX || prev_theme == usize::MAX {
            continue;
        }

        if curr_theme == prev_theme {
            result[[s, 0]] = 0.0;
            result[[s, 1]] = 0.0;
        } else {
            let curr_stats = &stats_vec[curr_day];
            let prev_stats = &stats_vec[prev_day];

            if curr_theme < curr_stats.mean_return.len()
                && prev_theme < prev_stats.mean_return.len()
            {
                result[[s, 0]] =
                    curr_stats.mean_return[curr_theme] - prev_stats.mean_return[prev_theme];
                result[[s, 1]] =
                    curr_stats.total_amt[curr_theme] - prev_stats.total_amt[prev_theme];
            }
        }
    }

    result
}

/// F21: 主题熵因子
fn factor_theme_entropy(
    theme_history: &Array2<usize>, // (n_stocks x n_days)
    lookback: usize,
) -> Array1<f64> {
    let n = theme_history.nrows();
    let n_days = theme_history.ncols();
    let lb = lookback.min(n_days);
    let start_day = n_days - lb;
    let mut result = Array1::<f64>::from_elem(n, f64::NAN);

    for s in 0..n {
        let mut themes = Vec::new();
        for d in start_day..n_days {
            let t = theme_history[[s, d]];
            if t != usize::MAX {
                themes.push(t);
            }
        }

        if !themes.is_empty() {
            // 计算熵
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
fn factor_theme_evolution(
    theme_history: &Array2<usize>,
    stats_vec: &[ClusterStats],
    target_day_idx: usize,
    lookback: usize,
) -> Array2<f64> {
    let n = theme_history.nrows();
    let n_days = theme_history.ncols();
    let mut result = Array2::<f64>::from_elem((n, 4), f64::NAN);

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

    for s in 0..n {
        let theme = theme_history[[s, curr_idx]];
        if theme == usize::MAX || theme >= curr_stats.total_amt.len() {
            continue;
        }

        let curr_amt = curr_stats.total_amt[theme];

        // F23: 主题热度 = 当日成交额 / 过去平均成交额
        let lb_start = if curr_idx > lookback {
            curr_idx - lookback
        } else {
            0
        };
        let mut past_amts = Vec::new();
        for d in lb_start..curr_idx {
            if theme < stats_vec[d].total_amt.len() {
                past_amts.push(stats_vec[d].total_amt[theme]);
            }
        }
        if !past_amts.is_empty() {
            let avg: f64 = past_amts.iter().sum::<f64>() / past_amts.len() as f64;
            result[[s, 0]] = curr_amt / (avg + 1e-10);
        } else {
            result[[s, 0]] = 1.0;
        }

        // F24: 动量持续性 = 过去成交额序列的自相关
        if past_amts.len() >= 3 {
            result[[s, 1]] = autocorr_lag1(&past_amts);
        } else {
            result[[s, 1]] = 0.0;
        }

        // F25: 主题扩散 = (当日size - 前日size) / 前日size
        if let Some(ps) = prev_stats {
            if theme < ps.size.len() {
                let curr_size = curr_stats.size[theme];
                let prev_size = ps.size[theme];
                result[[s, 2]] = (curr_size - prev_size) / (prev_size + 1e-10);
            } else {
                result[[s, 2]] = 0.0;
            }
        } else {
            result[[s, 2]] = 0.0;
        }

        // F26: 领导者稳定性 = size / 4000
        result[[s, 3]] = curr_stats.size[theme] / 4000.0;
    }

    result
}

/// 自相关系数（lag=1）
fn autocorr_lag1(data: &[f64]) -> f64 {
    let n = data.len();
    if n < 2 {
        return 0.0;
    }
    let mean: f64 = data.iter().sum::<f64>() / n as f64;
    let mut num = 0.0;
    let mut den = 0.0;
    for i in 0..n - 1 {
        num += (data[i] - mean) * (data[i + 1] - mean);
    }
    for x in data {
        den += (x - mean).powi(2);
    }
    if den.abs() < 1e-15 {
        0.0
    } else {
        num / den
    }
}

/// F17-F18: 微观偏离因子（单个微观指标）
/// micro_metrics: (n_stocks,) 微观指标值
/// prev_micro_metrics: (n_stocks,) 前日微观指标值（可选）
fn factor_micro_deviation(
    micro_metric: &Array1<f64>,       // 当日微观指标
    prev_micro_metric: Option<&Array1<f64>>, // 前日微观指标
    theme_labels: &[usize],           // 当日主题标签
    prev_theme_labels: Option<&[usize]>, // 前日主题标签
    k: usize,
) -> Array2<f64> {
    let n = micro_metric.len();
    let mut result = Array2::<f64>::from_elem((n, 2), f64::NAN);

    // 按主题分组计算均值
    let mut theme_stocks: Vec<Vec<usize>> = vec![Vec::new(); k.max(1)];
    for s in 0..n {
        let t = theme_labels[s];
        if t != usize::MAX && t < k {
            theme_stocks[t].push(s);
        }
    }

    let mut theme_mean = vec![f64::NAN; k.max(1)];
    for c in 0..k {
        if !theme_stocks[c].is_empty() {
            let sum: f64 = theme_stocks[c]
                .iter()
                .filter_map(|&s| {
                    let v = micro_metric[s];
                    if v.is_finite() { Some(v) } else { None }
                })
                .sum();
            let cnt = theme_stocks[c]
                .iter()
                .filter(|&&s| micro_metric[s].is_finite())
                .count();
            if cnt > 0 {
                theme_mean[c] = sum / cnt as f64;
            }
        }
    }

    // 前日的主题分组和均值
    let mut prev_theme_mean = vec![f64::NAN; k.max(1)];
    if let (Some(prev_labels), Some(prev_micro)) = (prev_theme_labels, prev_micro_metric) {
        let mut prev_theme_stocks: Vec<Vec<usize>> = vec![Vec::new(); k.max(1)];
        for s in 0..n.min(prev_labels.len()) {
            let t = prev_labels[s];
            if t != usize::MAX && t < k {
                prev_theme_stocks[t].push(s);
            }
        }
        for c in 0..k {
            if !prev_theme_stocks[c].is_empty() {
                let sum: f64 = prev_theme_stocks[c]
                    .iter()
                    .filter_map(|&s| {
                        if s < prev_micro.len() {
                            let v = prev_micro[s];
                            if v.is_finite() { Some(v) } else { None }
                        } else { None }
                    })
                    .sum();
                let cnt = prev_theme_stocks[c]
                    .iter()
                    .filter(|&&s| s < prev_micro.len() && prev_micro[s].is_finite())
                    .count();
                if cnt > 0 {
                    prev_theme_mean[c] = sum / cnt as f64;
                }
            }
        }
    }

    for s in 0..n {
        let theme = theme_labels[s];
        if theme == usize::MAX || theme >= k {
            continue;
        }

        let val = micro_metric[s];
        let tm = theme_mean[theme];
        if val.is_finite() && tm.is_finite() {
            // F17: 当日偏离
            result[[s, 0]] = val - tm;

            // F18: 偏离变化
            if let Some(prev_labels) = prev_theme_labels {
                if let Some(prev_micro) = prev_micro_metric {
                    if s < prev_labels.len() {
                        let prev_theme = prev_labels[s];
                        if prev_theme != usize::MAX
                            && prev_theme < k
                            && s < prev_micro.len()
                            && prev_micro[s].is_finite()
                            && prev_theme_mean[prev_theme].is_finite()
                        {
                            let prev_dev = prev_micro[s] - prev_theme_mean[prev_theme];
                            result[[s, 1]] = (val - tm) - prev_dev;
                        }
                    }
                }
            }
        }
    }

    result
}

// ============================================================
// 主入口: compute_all_factors
// ============================================================

/// 计算所有主题聚类因子
///
/// 参数:
/// - features_list: Vec<(n_stocks, n_features)> 的特征矩阵列表，按日期升序排列
///   特征列顺序: daily_return(0), intraday_vol(1), log_total_amt(2), act_buy_ratio(3),
///               log_avg_trade_amt(4), min_ret_skew(5), up_tick_ratio(6),
///               bid_size1_mean(7), ask_size1_mean(8), vwap_diff(9),
///               last_bid_vol1(10), last_ask_vol1(11)
/// - micro_metrics_list: Vec<(n_stocks, n_micro_metrics)> 微观指标矩阵列表，按日期升序
///   微观指标列顺序: act_buy_ratio(0), bid_ask_imbalance(1), vol_per_trade(2),
///                   spread_ratio(3), depth_imbalance(4), vwap_deviation(5),
///                   act_buy_vol_ratio(6), big_order_ratio(7)
/// - k: 聚类数
/// - align_method: 对齐方法 ("hungarian" / "overlap")
/// - distance_metric: 距离度量 ("euclidean" / "cosine")
/// - distance_threshold: 距离阈值 (<=0 表示不使用)
/// - lookback: 回溯天数
///
/// 返回:
///   (n_stocks, n_factors) 的因子矩阵
///   因子顺序: F4, F5, F6, F7, F8, F9, F10, F11, F12, F13, F14, F21,
///             F23, F24, F25, F26, [微观偏离因子 F17_x, F18_x ...]
#[pyfunction]
#[pyo3(signature = (features_list, micro_metrics_list, k, align_method, distance_metric, distance_threshold, lookback))]
pub fn theme_cluster_factors(
    py: Python,
    features_list: Vec<PyReadonlyArray2<f64>>,
    micro_metrics_list: Vec<PyReadonlyArray2<f64>>,
    k: usize,
    align_method: &str,
    distance_metric: &str,
    distance_threshold: f64,
    lookback: usize,
) -> PyResult<Py<PyArray2<f64>>> {
    let n_days = features_list.len();
    if n_days == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "features_list不能为空",
        ));
    }

    let n_stocks = features_list[0].as_array().nrows();

    // 转为owned ndarray
    let features_arrays: Vec<Array2<f64>> = features_list
        .iter()
        .map(|f| f.as_array().to_owned())
        .collect();

    let micro_arrays: Vec<Array2<f64>> = micro_metrics_list
        .iter()
        .map(|f| f.as_array().to_owned())
        .collect();

    // 1. 逐日聚类
    let day_results: Vec<DayClusterResult> = features_arrays
        .iter()
        .map(|feat| cluster_single_day(feat, k))
        .collect();

    // 2. 跨日对齐
    let aligned_labels = align_clusters_multi_days(
        &day_results,
        align_method,
        distance_metric,
        distance_threshold,
        n_stocks,
    );

    // 构建 theme_history (n_stocks x n_days)
    let mut theme_history = Array2::<usize>::from_elem((n_stocks, n_days), usize::MAX);
    for d in 0..n_days {
        for s in 0..n_stocks {
            theme_history[[s, d]] = aligned_labels[d][s];
        }
    }

    // 构建对齐后的统计量
    let stats_vec: Vec<ClusterStats> = (0..n_days)
        .map(|d| {
            // 用对齐后的标签重新计算统计量
            let aligned_k = aligned_labels[d]
                .iter()
                .filter(|&&l| l != usize::MAX)
                .cloned()
                .max()
                .unwrap_or(0)
                + 1;
            // 用原始特征和对齐后标签
            let mut labels_for_clean: Vec<usize> = Vec::new();
            let valid = &day_results[d].valid_indices;
            for &vi in valid {
                labels_for_clean.push(aligned_labels[d][vi]);
            }
            compute_cluster_stats(&day_results[d].features_clean, &labels_for_clean, aligned_k)
        })
        .collect();

    // 获取对齐后的中心点（用映射重索引）
    let target_day = n_days - 1;
    let target_labels = &aligned_labels[target_day];

    // 3. 计算各因子
    let f_momentum = factor_theme_momentum(&theme_history, &stats_vec, lookback);
    let f_strength = factor_theme_strength(target_labels, &stats_vec[target_day]);

    // 对齐后的中心点
    let aligned_k_target = target_labels
        .iter()
        .filter(|&&l| l != usize::MAX)
        .cloned()
        .max()
        .unwrap_or(0)
        + 1;
    let f_rank = factor_theme_rank(
        &features_arrays[target_day],
        target_labels,
        &day_results[target_day].centers,
        aligned_k_target,
    );

    let f_switch = factor_theme_switch(&theme_history, &stats_vec);
    let f_entropy = factor_theme_entropy(&theme_history, lookback);
    let f_evolution = factor_theme_evolution(&theme_history, &stats_vec, target_day, lookback);

    // 微观偏离因子
    let n_micro = if !micro_arrays.is_empty() {
        micro_arrays[0].ncols()
    } else {
        0
    };

    let mut micro_factors: Vec<Array2<f64>> = Vec::new();
    if n_micro > 0 && n_days >= 1 {
        let prev_labels = if n_days >= 2 {
            Some(aligned_labels[n_days - 2].as_slice())
        } else {
            None
        };

        for m_idx in 0..n_micro {
            let curr_metric = micro_arrays[target_day].column(m_idx).to_owned();
            let prev_metric = if n_days >= 2 && micro_arrays.len() > n_days - 2 {
                Some(micro_arrays[n_days - 2].column(m_idx).to_owned())
            } else {
                None
            };

            let f = factor_micro_deviation(
                &curr_metric,
                prev_metric.as_ref(),
                target_labels,
                prev_labels,
                aligned_k_target,
            );
            micro_factors.push(f);
        }
    }

    // 4. 合并所有因子
    // 基础因子: F4(0), F5(1), F6(2), F7(3), F8(4), F9(5), F10(6),
    //           F11(7), F12(8), F13(9), F14(10), F21(11),
    //           F23(12), F24(13), F25(14), F26(15)
    let n_base = 16;
    let n_micro_cols = n_micro * 2; // 每个微观指标产生F17和F18
    let total_cols = n_base + n_micro_cols;

    let mut output = Array2::<f64>::from_elem((n_stocks, total_cols), f64::NAN);

    // F4-F6
    for s in 0..n_stocks {
        output[[s, 0]] = f_momentum[[s, 0]];
        output[[s, 1]] = f_momentum[[s, 1]];
        output[[s, 2]] = f_momentum[[s, 2]];
    }
    // F7-F10
    for s in 0..n_stocks {
        output[[s, 3]] = f_strength[[s, 0]];
        output[[s, 4]] = f_strength[[s, 1]];
        output[[s, 5]] = f_strength[[s, 2]];
        output[[s, 6]] = f_strength[[s, 3]];
    }
    // F11-F12
    for s in 0..n_stocks {
        output[[s, 7]] = f_rank[[s, 0]];
        output[[s, 8]] = f_rank[[s, 1]];
    }
    // F13-F14
    for s in 0..n_stocks {
        output[[s, 9]] = f_switch[[s, 0]];
        output[[s, 10]] = f_switch[[s, 1]];
    }
    // F21
    for s in 0..n_stocks {
        output[[s, 11]] = f_entropy[s];
    }
    // F23-F26
    for s in 0..n_stocks {
        output[[s, 12]] = f_evolution[[s, 0]];
        output[[s, 13]] = f_evolution[[s, 1]];
        output[[s, 14]] = f_evolution[[s, 2]];
        output[[s, 15]] = f_evolution[[s, 3]];
    }
    // 微观偏离因子
    for (m_idx, mf) in micro_factors.iter().enumerate() {
        let col_offset = n_base + m_idx * 2;
        for s in 0..n_stocks {
            output[[s, col_offset]] = mf[[s, 0]];
            output[[s, col_offset + 1]] = mf[[s, 1]];
        }
    }

    Ok(output.into_pyarray(py).to_owned())
}
