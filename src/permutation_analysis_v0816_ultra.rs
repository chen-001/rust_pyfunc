use pyo3::prelude::*;
use numpy::{PyArray1, PyArray2};
use ndarray::{Array1, Array2};
use nalgebra::{DMatrix, SymmetricEigen};
use rand::prelude::*;
use std::sync::Once;

// 全局初始化标志
static INIT: Once = Once::new();
static mut PERMUTATION_CACHE: Vec<Vec<usize>> = Vec::new();

fn initialize_permutations() {
    unsafe {
        INIT.call_once(|| {
            let mut perms = Vec::new();
            generate_all_permutations_heap(&mut vec![0, 1, 2, 3, 4], 5, &mut perms);
            PERMUTATION_CACHE = perms;
        });
    }
}

// 使用Heap算法生成排列
fn generate_all_permutations_heap(arr: &mut Vec<usize>, n: usize, result: &mut Vec<Vec<usize>>) {
    if n == 1 {
        result.push(arr.clone());
        return;
    }
    
    for i in 0..n {
        generate_all_permutations_heap(arr, n - 1, result);
        
        if n % 2 == 1 {
            arr.swap(0, n - 1);
        } else {
            arr.swap(i, n - 1);
        }
    }
}

fn get_cached_permutations() -> &'static Vec<Vec<usize>> {
    initialize_permutations();
    unsafe { &PERMUTATION_CACHE }
}

#[pyfunction]
pub fn analyze_sequence_permutations_v0816_ultra(
    py: Python,
    sequence: &PyArray1<f64>,
    window_size: Option<usize>,
    n_clusters: Option<usize>,
) -> PyResult<(Py<PyArray2<f64>>, Vec<String>)> {
    let window_size = window_size.unwrap_or(5);
    let n_clusters = n_clusters.unwrap_or(3);
    
    if window_size != 5 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "超级优化版本目前只支持window_size=5"
        ));
    }
    
    let binding = sequence.readonly();
    let sequence_arr = binding.as_array();
    let n = sequence_arr.len();
    
    if n < window_size {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "序列长度不能小于窗口大小"
        ));
    }
    
    let n_windows = n - window_size + 1;
    let indicator_names = vec![
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
    
    let mut results = Array2::<f64>::zeros((n_windows, 9));
    
    // 预分配内存，重用数据结构
    let permutations = get_cached_permutations();
    let n_perms = permutations.len();
    
    let mut perm_matrix = Array2::<f64>::zeros((5, n_perms));
    let mut corr_matrix = Array2::<f64>::zeros((n_perms, n_perms));
    
    // 预分配聚类相关的缓存
    let mut distance_cache = vec![vec![0.0; n_perms]; n_perms];
    let mut cluster_cache = ClusteringCache::new(n_perms, n_clusters);
    
    for i in 0..n_windows {
        let window_data = sequence_arr.slice(ndarray::s![i..i + window_size]);
        
        if i < 4 {
            for j in 0..9 {
                results[[i, j]] = f64::NAN;
            }
            continue;
        }
        
        let indicators = compute_window_indicators_ultra(
            &window_data.to_owned(),
            n_clusters,
            &mut perm_matrix,
            &mut corr_matrix,
            &mut distance_cache,
            &mut cluster_cache,
            permutations
        );
        
        for j in 0..9 {
            results[[i, j]] = indicators[j];
        }
    }
    
    let results_transposed = results.reversed_axes();
    let py_result = PyArray2::from_array(py, &results_transposed);
    
    Ok((py_result.to_owned(), indicator_names))
}

struct ClusteringCache {
    centroids: Array2<f64>,
    assignments: Vec<usize>,
    cluster_points: Vec<Vec<usize>>,
    temp_distances: Vec<f64>,
}

impl ClusteringCache {
    fn new(n_points: usize, n_clusters: usize) -> Self {
        Self {
            centroids: Array2::zeros((5, n_clusters)),
            assignments: vec![0; n_points],
            cluster_points: vec![Vec::with_capacity(n_points / n_clusters + 10); n_clusters],
            temp_distances: Vec::with_capacity(n_points),
        }
    }
    
    fn clear(&mut self) {
        for cluster in &mut self.cluster_points {
            cluster.clear();
        }
        self.temp_distances.clear();
    }
}

fn compute_window_indicators_ultra(
    window_data: &Array1<f64>,
    n_clusters: usize,
    perm_matrix: &mut Array2<f64>,
    corr_matrix: &mut Array2<f64>,
    distance_cache: &mut Vec<Vec<f64>>,
    cluster_cache: &mut ClusteringCache,
    permutations: &[Vec<usize>],
) -> Vec<f64> {
    // 快速检查唯一值
    let mut sorted_data = window_data.to_vec();
    sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mut unique_count = 1;
    for i in 1..sorted_data.len() {
        if (sorted_data[i] - sorted_data[i-1]).abs() > 1e-12 {
            unique_count += 1;
        }
    }
    
    if unique_count < 3 {
        return vec![f64::NAN; 9];
    }
    
    let n_perms = permutations.len();
    
    // 构建排列矩阵
    for (j, perm) in permutations.iter().enumerate() {
        for (i, &idx) in perm.iter().enumerate() {
            perm_matrix[[i, j]] = window_data[idx];
        }
    }
    
    // 高效相关性矩阵计算
    compute_correlation_matrix_ultra_fast(perm_matrix, corr_matrix);
    
    // 相关性矩阵均值（排除对角线）
    let mut sum = 0.0;
    let mut count = 0;
    for i in 0..n_perms {
        for j in 0..n_perms {
            if i != j {
                sum += corr_matrix[[i, j]];
                count += 1;
            }
        }
    }
    let corr_mean = sum / count as f64;
    
    // 使用nalgebra计算精确特征值（保证准确性）
    let max_eigenvalue = compute_max_eigenvalue_nalgebra(corr_matrix);
    
    // 预计算距离矩阵（重用缓存）
    compute_distance_matrix_cached(perm_matrix, distance_cache);
    
    // 高效聚类分析
    let (silhouette_score, cluster_sizes, intra_cluster_distances, centroids) = 
        perform_clustering_analysis_ultra(
            perm_matrix, 
            n_clusters, 
            distance_cache, 
            cluster_cache
        );
    
    // 快速熵计算
    let cluster_size_entropy = compute_entropy_optimized(&cluster_sizes.iter().map(|&x| x as f64).collect::<Vec<_>>());
    let max_cluster_size = *cluster_sizes.iter().max().unwrap_or(&0) as f64;
    
    let intra_dist_entropy = compute_entropy_optimized(&intra_cluster_distances);
    let intra_dist_max = intra_cluster_distances.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let intra_dist_min = intra_cluster_distances.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    
    let centroid_corr_mean = compute_centroid_correlation_ultra_fast(&centroids);
    
    vec![
        corr_mean,
        max_eigenvalue,
        silhouette_score,
        cluster_size_entropy,
        max_cluster_size,
        intra_dist_entropy,
        intra_dist_max,
        intra_dist_min,
        centroid_corr_mean,
    ]
}

fn compute_correlation_matrix_ultra_fast(data: &Array2<f64>, corr_matrix: &mut Array2<f64>) {
    let (n_features, n_cols) = data.dim();
    let n_features_f = n_features as f64;
    
    // 预计算所有列的统计量
    let mut means = vec![0.0; n_cols];
    let mut inv_stds = vec![0.0; n_cols];
    
    // 计算均值
    for j in 0..n_cols {
        let mut sum = 0.0;
        for i in 0..n_features {
            sum += data[[i, j]];
        }
        means[j] = sum / n_features_f;
    }
    
    // 计算标准差的倒数
    for j in 0..n_cols {
        let mut sum_sq = 0.0;
        for i in 0..n_features {
            let diff = data[[i, j]] - means[j];
            sum_sq += diff * diff;
        }
        let std_dev = (sum_sq / n_features_f).sqrt();
        inv_stds[j] = if std_dev > 1e-12 { 1.0 / std_dev } else { 0.0 };
    }
    
    // 计算相关性矩阵（利用对称性）
    for i in 0..n_cols {
        corr_matrix[[i, i]] = 1.0;
        
        for j in (i + 1)..n_cols {
            if inv_stds[i] == 0.0 || inv_stds[j] == 0.0 {
                corr_matrix[[i, j]] = 0.0;
                corr_matrix[[j, i]] = 0.0;
                continue;
            }
            
            let mut covariance = 0.0;
            for k in 0..n_features {
                covariance += (data[[k, i]] - means[i]) * (data[[k, j]] - means[j]);
            }
            covariance /= n_features_f;
            
            let correlation = covariance * inv_stds[i] * inv_stds[j];
            corr_matrix[[i, j]] = correlation;
            corr_matrix[[j, i]] = correlation;
        }
    }
}

fn compute_max_eigenvalue_nalgebra(matrix: &Array2<f64>) -> f64 {
    let n = matrix.nrows();
    let mut nalgebra_matrix = DMatrix::<f64>::zeros(n, n);
    
    // 复制矩阵数据
    for i in 0..n {
        for j in 0..n {
            nalgebra_matrix[(i, j)] = matrix[[i, j]];
        }
    }
    
    // 使用nalgebra的特征值分解（保证准确性）
    let eigen = SymmetricEigen::new(nalgebra_matrix);
    eigen.eigenvalues.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b))
}

fn compute_distance_matrix_cached(data: &Array2<f64>, distance_cache: &mut Vec<Vec<f64>>) {
    let (n_features, n_points) = data.dim();
    
    // 计算欧几里得距离矩阵（利用对称性）
    for i in 0..n_points {
        distance_cache[i][i] = 0.0;
        
        for j in (i + 1)..n_points {
            let mut distance_sq = 0.0;
            for k in 0..n_features {
                let diff = data[[k, i]] - data[[k, j]];
                distance_sq += diff * diff;
            }
            let distance = distance_sq.sqrt();
            distance_cache[i][j] = distance;
            distance_cache[j][i] = distance;
        }
    }
}

fn perform_clustering_analysis_ultra(
    data: &Array2<f64>,
    n_clusters: usize,
    distance_cache: &[Vec<f64>],
    cluster_cache: &mut ClusteringCache,
) -> (f64, Vec<usize>, Vec<f64>, Array2<f64>) {
    let (n_features, n_points) = data.dim();
    
    cluster_cache.clear();
    
    // K-means++初始化
    let mut rng = thread_rng();
    let first_center = rng.gen_range(0..n_points);
    
    // 选择第一个中心
    for i in 0..n_features {
        cluster_cache.centroids[[i, 0]] = data[[i, first_center]];
    }
    
    // 选择剩余中心（K-means++）
    for cluster_idx in 1..n_clusters {
        let mut max_min_distance = 0.0;
        let mut best_point = 0;
        
        for point_idx in 0..n_points {
            let mut min_distance = f64::INFINITY;
            
            for existing_cluster in 0..cluster_idx {
                let mut dist_sq = 0.0;
                for feature_idx in 0..n_features {
                    let diff = data[[feature_idx, point_idx]] - cluster_cache.centroids[[feature_idx, existing_cluster]];
                    dist_sq += diff * diff;
                }
                min_distance = min_distance.min(dist_sq);
            }
            
            if min_distance > max_min_distance {
                max_min_distance = min_distance;
                best_point = point_idx;
            }
        }
        
        for i in 0..n_features {
            cluster_cache.centroids[[i, cluster_idx]] = data[[i, best_point]];
        }
    }
    
    // K-means迭代（限制迭代次数保证性能）
    let max_iterations = 15;
    for _iteration in 0..max_iterations {
        let mut changed = false;
        
        // 分配点到最近的聚类中心
        for point_idx in 0..n_points {
            let mut min_distance = f64::INFINITY;
            let mut best_cluster = cluster_cache.assignments[point_idx];
            
            for cluster_idx in 0..n_clusters {
                let mut distance = 0.0;
                for feature_idx in 0..n_features {
                    let diff = data[[feature_idx, point_idx]] - cluster_cache.centroids[[feature_idx, cluster_idx]];
                    distance += diff * diff;
                }
                
                if distance < min_distance {
                    min_distance = distance;
                    best_cluster = cluster_idx;
                }
            }
            
            if cluster_cache.assignments[point_idx] != best_cluster {
                cluster_cache.assignments[point_idx] = best_cluster;
                changed = true;
            }
        }
        
        if !changed {
            break;
        }
        
        // 更新聚类中心
        for cluster_idx in 0..n_clusters {
            cluster_cache.cluster_points[cluster_idx].clear();
            
            for point_idx in 0..n_points {
                if cluster_cache.assignments[point_idx] == cluster_idx {
                    cluster_cache.cluster_points[cluster_idx].push(point_idx);
                }
            }
            
            if !cluster_cache.cluster_points[cluster_idx].is_empty() {
                for feature_idx in 0..n_features {
                    let mean = cluster_cache.cluster_points[cluster_idx]
                        .iter()
                        .map(|&point_idx| data[[feature_idx, point_idx]])
                        .sum::<f64>() / cluster_cache.cluster_points[cluster_idx].len() as f64;
                    cluster_cache.centroids[[feature_idx, cluster_idx]] = mean;
                }
            }
        }
    }
    
    // 计算轮廓系数
    let silhouette_score = compute_silhouette_score_ultra_fast(
        &cluster_cache.assignments,
        distance_cache,
        &cluster_cache.cluster_points,
        &mut cluster_cache.temp_distances
    );
    
    // 计算聚类大小
    let cluster_sizes: Vec<usize> = cluster_cache.cluster_points.iter().map(|cluster| cluster.len()).collect();
    
    // 计算簇内平均距离
    let intra_distances = compute_intra_distances_ultra_fast(
        distance_cache,
        &cluster_cache.cluster_points
    );
    
    (silhouette_score, cluster_sizes, intra_distances, cluster_cache.centroids.clone())
}

fn compute_silhouette_score_ultra_fast(
    assignments: &[usize],
    distance_cache: &[Vec<f64>],
    cluster_points: &[Vec<usize>],
    temp_distances: &mut Vec<f64>,
) -> f64 {
    let n_points = assignments.len();
    temp_distances.clear();
    temp_distances.reserve(n_points);
    
    for i in 0..n_points {
        let cluster = assignments[i];
        
        // 计算簇内平均距离
        let a = if cluster_points[cluster].len() > 1 {
            let mut sum = 0.0;
            let mut count = 0;
            for &j in &cluster_points[cluster] {
                if i != j {
                    sum += distance_cache[i][j];
                    count += 1;
                }
            }
            if count > 0 { sum / count as f64 } else { 0.0 }
        } else {
            0.0
        };
        
        // 计算到最近其他簇的平均距离
        let mut min_b = f64::INFINITY;
        for (other_cluster, other_points) in cluster_points.iter().enumerate() {
            if other_cluster != cluster && !other_points.is_empty() {
                let sum: f64 = other_points.iter().map(|&j| distance_cache[i][j]).sum();
                let avg = sum / other_points.len() as f64;
                min_b = min_b.min(avg);
            }
        }
        let b = if min_b.is_infinite() { 0.0 } else { min_b };
        
        // 轮廓系数
        let silhouette = if a.max(b) == 0.0 { 0.0 } else { (b - a) / a.max(b) };
        temp_distances.push(silhouette);
    }
    
    temp_distances.iter().sum::<f64>() / temp_distances.len() as f64
}

fn compute_intra_distances_ultra_fast(
    distance_cache: &[Vec<f64>],
    cluster_points: &[Vec<usize>],
) -> Vec<f64> {
    let mut distances = Vec::with_capacity(cluster_points.len());
    
    for cluster in cluster_points {
        if cluster.len() > 1 {
            let mut sum = 0.0;
            let mut count = 0;
            
            for i in 0..cluster.len() {
                for j in (i + 1)..cluster.len() {
                    sum += distance_cache[cluster[i]][cluster[j]];
                    count += 1;
                }
            }
            
            distances.push(sum / count as f64);
        } else {
            distances.push(0.0);
        }
    }
    
    distances
}

fn compute_entropy_optimized(values: &[f64]) -> f64 {
    let total: f64 = values.iter().sum();
    if total <= 0.0 {
        return 0.0;
    }
    
    let mut entropy = 0.0;
    for &value in values {
        if value > 0.0 {
            let prob = value / total;
            entropy -= prob * prob.ln();
        }
    }
    
    entropy
}

fn compute_centroid_correlation_ultra_fast(centroids: &Array2<f64>) -> f64 {
    let (n_features, n_centroids) = centroids.dim();
    if n_centroids < 2 {
        return f64::NAN;
    }
    
    let mut correlations = Vec::new();
    
    for i in 0..n_centroids {
        for j in (i + 1)..n_centroids {
            // 使用高效的相关性计算公式
            let mut sum_xy = 0.0;
            let mut sum_x = 0.0;
            let mut sum_y = 0.0;
            let mut sum_x_sq = 0.0;
            let mut sum_y_sq = 0.0;
            
            for k in 0..n_features {
                let x = centroids[[k, i]];
                let y = centroids[[k, j]];
                sum_xy += x * y;
                sum_x += x;
                sum_y += y;
                sum_x_sq += x * x;
                sum_y_sq += y * y;
            }
            
            let n = n_features as f64;
            let numerator = n * sum_xy - sum_x * sum_y;
            let denominator = ((n * sum_x_sq - sum_x * sum_x) * (n * sum_y_sq - sum_y * sum_y)).sqrt();
            
            if denominator > 1e-12 {
                correlations.push(numerator / denominator);
            }
        }
    }
    
    if correlations.is_empty() {
        f64::NAN
    } else {
        correlations.iter().sum::<f64>() / correlations.len() as f64
    }
}