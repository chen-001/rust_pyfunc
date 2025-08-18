use pyo3::prelude::*;
use numpy::{PyArray1, PyArray2};
use ndarray::{Array1, Array2};
use rand::prelude::*;

// 预计算的5!排列索引（120个排列）
const PERMUTATION_INDICES: [[usize; 5]; 120] = generate_permutation_indices_const();

// 编译时生成5!的所有排列索引
const fn generate_permutation_indices_const() -> [[usize; 5]; 120] {
    let mut result = [[0; 5]; 120];
    let mut count = 0;
    
    // 手动展开Heap算法生成所有排列
    let mut indices = [0, 1, 2, 3, 4];
    
    // 这里我们需要手动生成所有120个排列
    // 由于const fn的限制，我们简化为运行时生成
    result[0] = [0, 1, 2, 3, 4];
    // 其余119个排列会在运行时计算并缓存
    
    result
}

// 运行时生成排列缓存
static mut PERMUTATIONS_CACHE: Option<Vec<[usize; 5]>> = None;
static mut CACHE_INITIALIZED: bool = false;

fn get_permutation_indices() -> &'static Vec<[usize; 5]> {
    unsafe {
        if !CACHE_INITIALIZED {
            let mut perms = Vec::new();
            generate_all_permutations(&mut [0, 1, 2, 3, 4], 0, &mut perms);
            PERMUTATIONS_CACHE = Some(perms);
            CACHE_INITIALIZED = true;
        }
        PERMUTATIONS_CACHE.as_ref().unwrap()
    }
}

fn generate_all_permutations(arr: &mut [usize; 5], start: usize, result: &mut Vec<[usize; 5]>) {
    if start == arr.len() {
        result.push(*arr);
        return;
    }
    
    for i in start..arr.len() {
        arr.swap(start, i);
        generate_all_permutations(arr, start + 1, result);
        arr.swap(start, i);
    }
}

#[pyfunction]
pub fn analyze_sequence_permutations_v0816_optimized(
    py: Python,
    sequence: &PyArray1<f64>,
    window_size: Option<usize>,
    n_clusters: Option<usize>,
) -> PyResult<(Py<PyArray2<f64>>, Vec<String>)> {
    let window_size = window_size.unwrap_or(5);
    let n_clusters = n_clusters.unwrap_or(3);
    
    if window_size != 5 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "优化版本目前只支持window_size=5"
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
    
    // 预分配内存
    let mut perm_matrix = Array2::<f64>::zeros((5, 120));
    let mut corr_matrix = Array2::<f64>::zeros((120, 120));
    let mut temp_vec = Vec::with_capacity(120);
    
    // 获取预计算的排列索引
    let perm_indices = get_permutation_indices();
    
    for i in 0..n_windows {
        let window_data = sequence_arr.slice(ndarray::s![i..i + window_size]);
        
        if i < 4 {
            for j in 0..9 {
                results[[i, j]] = f64::NAN;
            }
            continue;
        }
        
        let indicators = compute_window_indicators_optimized(
            &window_data.to_owned(), 
            n_clusters,
            &mut perm_matrix,
            &mut corr_matrix,
            &mut temp_vec,
            perm_indices
        );
        
        for j in 0..9 {
            results[[i, j]] = indicators[j];
        }
    }
    
    let results_transposed = results.reversed_axes();
    let py_result = PyArray2::from_array(py, &results_transposed);
    
    Ok((py_result.to_owned(), indicator_names))
}

fn compute_window_indicators_optimized(
    window_data: &Array1<f64>,
    n_clusters: usize,
    perm_matrix: &mut Array2<f64>,
    corr_matrix: &mut Array2<f64>,
    temp_vec: &mut Vec<f64>,
    perm_indices: &[[usize; 5]],
) -> Vec<f64> {
    // 检查唯一值数量
    let mut unique_values = std::collections::HashSet::new();
    for &val in window_data.iter() {
        unique_values.insert(ordered_float::OrderedFloat(val));
    }
    
    if unique_values.len() < 3 {
        return vec![f64::NAN; 9];
    }
    
    // 使用预计算的排列索引构建排列矩阵
    for (j, perm_idx) in perm_indices.iter().enumerate() {
        for (i, &idx) in perm_idx.iter().enumerate() {
            perm_matrix[[i, j]] = window_data[idx];
        }
    }
    
    // 优化的相关性矩阵计算
    compute_correlation_matrix_optimized(perm_matrix, corr_matrix);
    let corr_mean = corr_matrix.sum() / (120.0 * 120.0);
    
    // 优化的特征值计算（幂迭代法）
    let max_eigenvalue = compute_max_eigenvalue_power_iteration(corr_matrix, 50);
    
    // 优化的聚类分析
    let (silhouette_score, cluster_sizes, intra_cluster_distances, centroids) = 
        perform_clustering_analysis_optimized(perm_matrix, n_clusters, temp_vec);
    
    let cluster_size_entropy = compute_entropy_fast(&cluster_sizes.iter().map(|&x| x as f64).collect::<Vec<_>>());
    let max_cluster_size = *cluster_sizes.iter().max().unwrap_or(&0) as f64;
    
    let intra_dist_entropy = compute_entropy_fast(&intra_cluster_distances);
    let intra_dist_max = intra_cluster_distances.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let intra_dist_min = intra_cluster_distances.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    
    let centroid_corr_mean = compute_centroid_correlation_mean_optimized(&centroids);
    
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

fn compute_correlation_matrix_optimized(data: &Array2<f64>, corr_matrix: &mut Array2<f64>) {
    let n_cols = data.ncols();
    
    // 预计算每列的均值和标准差
    let mut means = vec![0.0; n_cols];
    let mut stds = vec![0.0; n_cols];
    
    for j in 0..n_cols {
        let col = data.column(j);
        let mean = col.sum() / col.len() as f64;
        means[j] = mean;
        
        let variance = col.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / col.len() as f64;
        stds[j] = variance.sqrt();
    }
    
    // 计算相关性矩阵，利用对称性
    for i in 0..n_cols {
        corr_matrix[[i, i]] = 1.0;
        
        for j in (i + 1)..n_cols {
            if stds[i] == 0.0 || stds[j] == 0.0 {
                corr_matrix[[i, j]] = 0.0;
                corr_matrix[[j, i]] = 0.0;
                continue;
            }
            
            let col_i = data.column(i);
            let col_j = data.column(j);
            
            let mut covariance = 0.0;
            for k in 0..col_i.len() {
                covariance += (col_i[k] - means[i]) * (col_j[k] - means[j]);
            }
            covariance /= col_i.len() as f64;
            
            let correlation = covariance / (stds[i] * stds[j]);
            corr_matrix[[i, j]] = correlation;
            corr_matrix[[j, i]] = correlation;
        }
    }
}

fn compute_max_eigenvalue_power_iteration(matrix: &Array2<f64>, max_iterations: usize) -> f64 {
    let n = matrix.nrows();
    let mut v = Array1::<f64>::ones(n);
    let mut lambda = 0.0;
    
    // 归一化初始向量
    let norm = v.dot(&v).sqrt();
    v /= norm;
    
    for _ in 0..max_iterations {
        let mut v_new = Array1::<f64>::zeros(n);
        
        // 矩阵乘法 A * v
        for i in 0..n {
            for j in 0..n {
                v_new[i] += matrix[[i, j]] * v[j];
            }
        }
        
        // 计算特征值估计
        let new_lambda = v.dot(&v_new);
        
        // 归一化
        let norm = v_new.dot(&v_new).sqrt();
        if norm > 1e-12 {
            v_new /= norm;
        }
        
        // 检查收敛
        if (new_lambda - lambda).abs() < 1e-10 {
            break;
        }
        
        lambda = new_lambda;
        v = v_new;
    }
    
    lambda
}

fn perform_clustering_analysis_optimized(
    data: &Array2<f64>,
    n_clusters: usize,
    temp_vec: &mut Vec<f64>,
) -> (f64, Vec<usize>, Vec<f64>, Array2<f64>) {
    let (cluster_assignments, centroids) = kmeans_clustering_optimized(data, n_clusters);
    
    let silhouette_score = compute_silhouette_score_optimized(data, &cluster_assignments, &centroids, temp_vec);
    
    let cluster_sizes = compute_cluster_sizes_optimized(&cluster_assignments, n_clusters);
    
    let intra_cluster_distances = compute_intra_cluster_distances_optimized(data, &cluster_assignments, n_clusters);
    
    (silhouette_score, cluster_sizes, intra_cluster_distances, centroids)
}

fn kmeans_clustering_optimized(data: &Array2<f64>, k: usize) -> (Vec<usize>, Array2<f64>) {
    let (n_features, n_points) = data.dim();
    let mut rng = thread_rng();
    
    // 使用k-means++初始化
    let mut centroids = Array2::<f64>::zeros((n_features, k));
    
    // 选择第一个中心点
    let first_idx = rng.gen_range(0..n_points);
    for i in 0..n_features {
        centroids[[i, 0]] = data[[i, first_idx]];
    }
    
    // 选择剩余的中心点（简化版k-means++）
    for cluster_idx in 1..k {
        let mut max_distance = 0.0;
        let mut best_point = 0;
        
        for point_idx in 0..n_points {
            let mut min_dist_to_center = f64::INFINITY;
            
            for existing_cluster in 0..cluster_idx {
                let mut dist = 0.0;
                for feature_idx in 0..n_features {
                    let diff = data[[feature_idx, point_idx]] - centroids[[feature_idx, existing_cluster]];
                    dist += diff * diff;
                }
                min_dist_to_center = min_dist_to_center.min(dist);
            }
            
            if min_dist_to_center > max_distance {
                max_distance = min_dist_to_center;
                best_point = point_idx;
            }
        }
        
        for i in 0..n_features {
            centroids[[i, cluster_idx]] = data[[i, best_point]];
        }
    }
    
    let mut assignments = vec![0; n_points];
    let max_iterations = 20; // 减少迭代次数但保持准确性
    
    for _iteration in 0..max_iterations {
        let mut new_assignments = vec![0; n_points];
        let mut changed = false;
        
        // 分配点到最近的中心
        for point_idx in 0..n_points {
            let mut min_distance = f64::INFINITY;
            let mut best_cluster = 0;
            
            for cluster_idx in 0..k {
                let mut distance = 0.0;
                for feature_idx in 0..n_features {
                    let diff = data[[feature_idx, point_idx]] - centroids[[feature_idx, cluster_idx]];
                    distance += diff * diff;
                }
                
                if distance < min_distance {
                    min_distance = distance;
                    best_cluster = cluster_idx;
                }
            }
            
            new_assignments[point_idx] = best_cluster;
            if assignments[point_idx] != best_cluster {
                changed = true;
            }
        }
        
        if !changed {
            break;
        }
        assignments = new_assignments;
        
        // 更新中心点
        for cluster_idx in 0..k {
            let cluster_points: Vec<_> = assignments
                .iter()
                .enumerate()
                .filter(|(_, &assignment)| assignment == cluster_idx)
                .map(|(point_idx, _)| point_idx)
                .collect();
            
            if !cluster_points.is_empty() {
                for feature_idx in 0..n_features {
                    let mean = cluster_points
                        .iter()
                        .map(|&point_idx| data[[feature_idx, point_idx]])
                        .sum::<f64>() / cluster_points.len() as f64;
                    centroids[[feature_idx, cluster_idx]] = mean;
                }
            }
        }
    }
    
    (assignments, centroids)
}

fn compute_silhouette_score_optimized(
    data: &Array2<f64>,
    assignments: &[usize],
    centroids: &Array2<f64>,
    temp_vec: &mut Vec<f64>,
) -> f64 {
    let n_points = data.ncols();
    temp_vec.clear();
    temp_vec.reserve(n_points);
    
    for i in 0..n_points {
        let cluster = assignments[i];
        
        // 计算簇内平均距离（简化计算）
        let a = compute_intra_cluster_distance_single(data, assignments, cluster, i);
        
        // 计算到最近其他簇的距离
        let b = compute_nearest_cluster_distance_single(data, centroids, cluster, i);
        
        let silhouette = if a.max(b) == 0.0 {
            0.0
        } else {
            (b - a) / a.max(b)
        };
        
        temp_vec.push(silhouette);
    }
    
    temp_vec.iter().sum::<f64>() / temp_vec.len() as f64
}

fn compute_intra_cluster_distance_single(
    data: &Array2<f64>,
    assignments: &[usize],
    cluster: usize,
    point_idx: usize,
) -> f64 {
    let cluster_points: Vec<_> = assignments
        .iter()
        .enumerate()
        .filter(|(idx, &assignment)| assignment == cluster && *idx != point_idx)
        .map(|(idx, _)| idx)
        .collect();
    
    if cluster_points.is_empty() {
        return 0.0;
    }
    
    let mut total_distance = 0.0;
    let n_features = data.nrows();
    
    for &other_idx in &cluster_points {
        let mut distance = 0.0;
        for feature_idx in 0..n_features {
            let diff = data[[feature_idx, point_idx]] - data[[feature_idx, other_idx]];
            distance += diff * diff;
        }
        total_distance += distance.sqrt();
    }
    
    total_distance / cluster_points.len() as f64
}

fn compute_nearest_cluster_distance_single(
    data: &Array2<f64>,
    centroids: &Array2<f64>,
    current_cluster: usize,
    point_idx: usize,
) -> f64 {
    let mut min_distance = f64::INFINITY;
    let n_features = data.nrows();
    
    for cluster_idx in 0..centroids.ncols() {
        if cluster_idx != current_cluster {
            let mut distance = 0.0;
            for feature_idx in 0..n_features {
                let diff = data[[feature_idx, point_idx]] - centroids[[feature_idx, cluster_idx]];
                distance += diff * diff;
            }
            min_distance = min_distance.min(distance.sqrt());
        }
    }
    
    min_distance
}

fn compute_cluster_sizes_optimized(assignments: &[usize], n_clusters: usize) -> Vec<usize> {
    let mut sizes = vec![0; n_clusters];
    for &assignment in assignments {
        sizes[assignment] += 1;
    }
    sizes
}

fn compute_intra_cluster_distances_optimized(
    data: &Array2<f64>,
    assignments: &[usize],
    n_clusters: usize,
) -> Vec<f64> {
    let mut distances = Vec::with_capacity(n_clusters);
    let n_features = data.nrows();
    
    for cluster_idx in 0..n_clusters {
        let cluster_points: Vec<_> = assignments
            .iter()
            .enumerate()
            .filter(|(_, &assignment)| assignment == cluster_idx)
            .map(|(idx, _)| idx)
            .collect();
        
        if cluster_points.len() > 1 {
            let mut total_distance = 0.0;
            let mut count = 0;
            
            for i in 0..cluster_points.len() {
                for j in i + 1..cluster_points.len() {
                    let mut distance = 0.0;
                    for feature_idx in 0..n_features {
                        let diff = data[[feature_idx, cluster_points[i]]] - data[[feature_idx, cluster_points[j]]];
                        distance += diff * diff;
                    }
                    total_distance += distance.sqrt();
                    count += 1;
                }
            }
            
            distances.push(total_distance / count as f64);
        } else {
            distances.push(0.0);
        }
    }
    
    distances
}

fn compute_entropy_fast(values: &[f64]) -> f64 {
    let total: f64 = values.iter().sum();
    if total == 0.0 {
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

fn compute_centroid_correlation_mean_optimized(centroids: &Array2<f64>) -> f64 {
    let n_centroids = centroids.ncols();
    if n_centroids < 2 {
        return f64::NAN;
    }
    
    let mut correlations = Vec::new();
    let n_features = centroids.nrows();
    
    for i in 0..n_centroids {
        for j in i + 1..n_centroids {
            // 快速相关性计算
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
            
            if denominator != 0.0 {
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