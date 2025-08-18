use nalgebra::{DMatrix, DVector};
use pyo3::prelude::*;

// 预计算所有5!排列的索引
const PERMUTATIONS: [[usize; 5]; 120] = generate_permutations();

const fn generate_permutations() -> [[usize; 5]; 120] {
    let mut result = [[0; 5]; 120];
    let mut count = 0;
    
    let mut a = 0;
    while a < 5 {
        let mut b = 0;
        while b < 5 {
            if b == a { b += 1; continue; }
            let mut c = 0;
            while c < 5 {
                if c == a || c == b { c += 1; continue; }
                let mut d = 0;
                while d < 5 {
                    if d == a || d == b || d == c { d += 1; continue; }
                    let mut e = 0;
                    while e < 5 {
                        if e == a || e == b || e == c || e == d { e += 1; continue; }
                        result[count] = [a, b, c, d, e];
                        count += 1;
                        e += 1;
                    }
                    d += 1;
                }
                c += 1;
            }
            b += 1;
        }
        a += 1;
    }
    result
}

// 预计算距离矩阵，避免重复计算
struct PrecomputedDistances {
    distances: Vec<Vec<f64>>,  // n×n距离矩阵
}

impl PrecomputedDistances {
    fn new(data: &[Vec<f64>]) -> Self {
        let n = data.len();
        let mut distances = vec![vec![0.0; n]; n];
        
        // 只计算上三角矩阵，利用对称性
        for i in 0..n {
            for j in (i + 1)..n {
                let dist = euclidean_distance(&data[i], &data[j]);
                distances[i][j] = dist;
                distances[j][i] = dist; // 对称性
            }
        }
        
        Self { distances }
    }
    
    fn get_distance(&self, i: usize, j: usize) -> f64 {
        self.distances[i][j]
    }
}

fn euclidean_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}

// 使用nalgebra优化的相关性矩阵计算
fn compute_correlation_matrix(matrix: &[Vec<f64>]) -> DMatrix<f64> {
    let n_cols = matrix.len();
    let n_rows = matrix[0].len();
    
    let mut data_matrix = DMatrix::zeros(n_rows, n_cols);
    for i in 0..n_cols {
        for j in 0..n_rows {
            data_matrix[(j, i)] = matrix[i][j];
        }
    }
    
    for j in 0..n_cols {
        let col = data_matrix.column(j);
        let mean = col.mean();
        for i in 0..n_rows {
            data_matrix[(i, j)] -= mean;
        }
    }
    
    let cov_matrix = data_matrix.transpose() * &data_matrix / (n_rows as f64);
    
    let mut correlation = DMatrix::zeros(n_cols, n_cols);
    for i in 0..n_cols {
        for j in 0..n_cols {
            let var_i = cov_matrix[(i, i)];
            let var_j = cov_matrix[(j, j)];
            
            if var_i > 1e-10 && var_j > 1e-10 {
                correlation[(i, j)] = cov_matrix[(i, j)] / (var_i * var_j).sqrt();
            } else {
                correlation[(i, j)] = if i == j { 1.0 } else { 0.0 };
            }
        }
    }
    
    correlation
}

fn compute_largest_eigenvalue(matrix: &DMatrix<f64>, max_iterations: usize) -> f64 {
    let n = matrix.nrows();
    let mut v = DVector::from_element(n, 1.0 / (n as f64).sqrt());
    let mut eigenvalue = 0.0;
    
    for _ in 0..max_iterations {
        let av = matrix * &v;
        let new_eigenvalue = v.dot(&av);
        
        let norm = av.norm();
        if norm > 1e-10 {
            v = av / norm;
        }
        
        if (new_eigenvalue - eigenvalue).abs() < 1e-8 {
            break;
        }
        eigenvalue = new_eigenvalue;
    }
    
    eigenvalue
}

struct ClusteringResult {
    labels: Vec<usize>,
    centers: Vec<Vec<f64>>,
    intra_cluster_distances: Vec<f64>,
    // 新增：预计算的统计信息
    cluster_sizes: Vec<f64>,
    silhouette_score: f64,
}

// 优化的聚类函数，一次性计算所有相关指标
fn optimized_clustering_with_stats(
    data: &[Vec<f64>], 
    k: usize, 
    max_iterations: usize
) -> ClusteringResult {
    let n_points = data.len();
    let n_dims = data[0].len();
    
    // 简单均匀初始化
    let mut centers = vec![vec![0.0; n_dims]; k];
    let step = n_points / k;
    for i in 0..k {
        let idx = (i * step).min(n_points - 1);
        centers[i] = data[idx].clone();
    }
    
    let mut labels = vec![0; n_points];
    
    // K-means迭代
    for _ in 0..max_iterations {
        let mut new_centers = vec![vec![0.0; n_dims]; k];
        let mut counts = vec![0; k];
        let mut changed = false;
        
        for (i, point) in data.iter().enumerate() {
            let mut min_dist = f64::INFINITY;
            let mut best_cluster = 0;
            
            for (j, center) in centers.iter().enumerate() {
                let dist = euclidean_distance(point, center);
                if dist < min_dist {
                    min_dist = dist;
                    best_cluster = j;
                }
            }
            
            if labels[i] != best_cluster {
                changed = true;
                labels[i] = best_cluster;
            }
        }
        
        if !changed { break; }
        
        for (i, point) in data.iter().enumerate() {
            let cluster = labels[i];
            counts[cluster] += 1;
            for j in 0..n_dims {
                new_centers[cluster][j] += point[j];
            }
        }
        
        for i in 0..k {
            if counts[i] > 0 {
                for j in 0..n_dims {
                    new_centers[i][j] /= counts[i] as f64;
                }
            }
        }
        
        centers = new_centers;
    }
    
    // 计算聚类大小
    let mut cluster_sizes = vec![0; k];
    for &label in &labels {
        cluster_sizes[label] += 1;
    }
    let cluster_sizes: Vec<f64> = cluster_sizes.into_iter().map(|x| x as f64).collect();
    
    // 计算簇内平均距离（到中心点）
    let mut cluster_distances = vec![0.0; k];
    let mut cluster_counts = vec![0; k];
    
    for (i, point) in data.iter().enumerate() {
        let cluster = labels[i];
        let dist = euclidean_distance(point, &centers[cluster]);
        cluster_distances[cluster] += dist;
        cluster_counts[cluster] += 1;
    }
    
    for i in 0..k {
        if cluster_counts[i] > 0 {
            cluster_distances[i] /= cluster_counts[i] as f64;
        }
    }
    
    // 预计算所有点对距离，用于轮廓系数计算
    let precomputed = PrecomputedDistances::new(data);
    
    // 优化的轮廓系数计算
    let silhouette_score = compute_optimized_silhouette_score(&precomputed, &labels, k);
    
    ClusteringResult {
        labels,
        centers,
        intra_cluster_distances: cluster_distances,
        cluster_sizes,
        silhouette_score,
    }
}

// 使用预计算距离的轮廓系数计算
fn compute_optimized_silhouette_score(
    precomputed: &PrecomputedDistances,
    labels: &[usize],
    n_clusters: usize,
) -> f64 {
    let n = labels.len();
    let mut silhouettes = vec![0.0; n];
    
    for i in 0..n {
        let cluster_i = labels[i];
        
        // 计算簇内平均距离
        let mut intra_dist = 0.0;
        let mut intra_count = 0;
        
        for j in 0..n {
            if i != j && labels[j] == cluster_i {
                intra_dist += precomputed.get_distance(i, j);
                intra_count += 1;
            }
        }
        
        if intra_count > 0 {
            intra_dist /= intra_count as f64;
        }
        
        // 计算最近邻簇的平均距离
        let mut min_inter_dist = f64::INFINITY;
        
        for other_cluster in 0..n_clusters {
            if other_cluster == cluster_i { continue; }
            
            let mut inter_dist = 0.0;
            let mut inter_count = 0;
            
            for j in 0..n {
                if labels[j] == other_cluster {
                    inter_dist += precomputed.get_distance(i, j);
                    inter_count += 1;
                }
            }
            
            if inter_count > 0 {
                inter_dist /= inter_count as f64;
                min_inter_dist = min_inter_dist.min(inter_dist);
            }
        }
        
        // 计算轮廓系数
        if intra_dist < min_inter_dist {
            silhouettes[i] = (min_inter_dist - intra_dist) / min_inter_dist;
        } else if intra_dist > min_inter_dist {
            silhouettes[i] = (min_inter_dist - intra_dist) / intra_dist;
        } else {
            silhouettes[i] = 0.0;
        }
    }
    
    silhouettes.iter().sum::<f64>() / n as f64
}

fn compute_entropy(values: &[f64]) -> f64 {
    if values.is_empty() { return 0.0; }
    
    let sum: f64 = values.iter().sum();
    if sum <= 0.0 { return 0.0; }
    
    let mut entropy = 0.0;
    for &val in values {
        if val > 0.0 {
            let p = val / sum;
            entropy -= p * p.ln();
        }
    }
    entropy
}

fn compute_center_correlations(centers: &[Vec<f64>]) -> f64 {
    let k = centers.len();
    if k < 2 { return 0.0; }
    
    let mut correlations = Vec::new();
    
    for i in 0..k {
        for j in (i + 1)..k {
            let center_i = &centers[i];
            let center_j = &centers[j];
            
            let mean_i: f64 = center_i.iter().sum::<f64>() / center_i.len() as f64;
            let mean_j: f64 = center_j.iter().sum::<f64>() / center_j.len() as f64;
            
            let mut covariance = 0.0;
            let mut var_i = 0.0;
            let mut var_j = 0.0;
            
            for k in 0..center_i.len() {
                let diff_i = center_i[k] - mean_i;
                let diff_j = center_j[k] - mean_j;
                covariance += diff_i * diff_j;
                var_i += diff_i * diff_i;
                var_j += diff_j * diff_j;
            }
            
            let correlation = if var_i * var_j > 1e-10 {
                covariance / (var_i * var_j).sqrt()
            } else {
                0.0
            };
            
            correlations.push(correlation);
        }
    }
    
    correlations.iter().sum::<f64>() / correlations.len() as f64
}

#[pyfunction]
pub fn analyze_sequence_permutations_optimized(
    sequence: Vec<f64>,
    window_size: Option<usize>,
    n_clusters: Option<usize>,
) -> PyResult<(Vec<Vec<f64>>, Vec<String>)> {
    let window_size = window_size.unwrap_or(5);
    let n_clusters = n_clusters.unwrap_or(3);
    
    if sequence.len() < window_size {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "序列长度必须大于等于窗口大小"
        ));
    }
    
    let n_windows = sequence.len() - window_size + 1;
    let mut results = vec![vec![f64::NAN; n_windows]; 9];
    
    let column_names = vec![
        "相关性矩阵均值".to_string(),
        "最大特征值".to_string(),
        "轮廓系数".to_string(),
        "聚类大小熵".to_string(),
        "最大聚类大小".to_string(),
        "簇内平均距离熵".to_string(),
        "簇内平均距离最大值".to_string(),
        "簇内平均距离最小值".to_string(),
        "聚类中心相关性均值".to_string(),
    ];
    
    // 处理每个滑动窗口（前4个输出位置为NaN）
    for window_idx in 0..n_windows {
        if window_idx < 4 {
            continue;
        }
        let window_data = &sequence[window_idx..window_idx + window_size];
        
        // 生成所有排列的矩阵 (5 x 120)
        let mut permutation_matrix = vec![vec![0.0; 120]; window_size];
        
        for (perm_idx, &perm) in PERMUTATIONS.iter().enumerate() {
            for (pos, &idx) in perm.iter().enumerate() {
                permutation_matrix[pos][perm_idx] = window_data[idx];
            }
        }
        
        // 计算相关性矩阵 (120 x 120)
        let correlation_matrix = compute_correlation_matrix(&permutation_matrix);
        
        // 1. 相关性矩阵均值
        let correlation_mean = correlation_matrix.iter().sum::<f64>() / (120 * 120) as f64;
        results[0][window_idx] = correlation_mean;
        
        // 2. 最大特征值
        let max_eigenvalue = compute_largest_eigenvalue(&correlation_matrix, 5);
        results[1][window_idx] = max_eigenvalue;
        
        // 准备聚类数据（转置矩阵，每列作为一个向量）
        let mut cluster_data = vec![vec![0.0; window_size]; 120];
        for i in 0..120 {
            for j in 0..window_size {
                cluster_data[i][j] = permutation_matrix[j][i];
            }
        }
        
        // 执行优化的聚类分析（一次性计算所有指标）
        let clustering_result = optimized_clustering_with_stats(&cluster_data, n_clusters, 3);
        
        // 3. 轮廓系数（已预计算）
        results[2][window_idx] = clustering_result.silhouette_score;
        
        // 4. 聚类大小熵
        let size_entropy = compute_entropy(&clustering_result.cluster_sizes);
        results[3][window_idx] = size_entropy;
        
        // 5. 最大聚类大小
        let max_cluster_size = clustering_result.cluster_sizes.iter().fold(0.0f64, |a, &b| a.max(b));
        results[4][window_idx] = max_cluster_size;
        
        // 6. 簇内平均距离熵
        let distance_entropy = compute_entropy(&clustering_result.intra_cluster_distances);
        results[5][window_idx] = distance_entropy;
        
        // 7. 簇内平均距离最大值
        let max_distance = clustering_result.intra_cluster_distances.iter().fold(0.0f64, |a, &b| a.max(b));
        results[6][window_idx] = max_distance;
        
        // 8. 簇内平均距离最小值
        let min_distance = clustering_result.intra_cluster_distances.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        results[7][window_idx] = if min_distance == f64::INFINITY { 0.0 } else { min_distance };
        
        // 9. 聚类中心相关性均值
        let center_correlation = compute_center_correlations(&clustering_result.centers);
        results[8][window_idx] = center_correlation;
    }
    
    Ok((results, column_names))
}