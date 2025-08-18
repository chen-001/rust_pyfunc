use ndarray::{Array1, Array2, ArrayView1, Axis, s};
use pyo3::prelude::*;
use numpy::{PyArray1, PyArray2};

// 预计算所有5!排列的索引，使用编译时常量
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

// 高效的欧几里得距离计算
#[inline]
fn euclidean_distance_array(a: ArrayView1<f64>, b: ArrayView1<f64>) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}

// 预计算距离矩阵结构体
struct PrecomputedDistances {
    distances: Array2<f64>,
}

impl PrecomputedDistances {
    fn new(data: &Array2<f64>) -> Self {
        let n = data.nrows();
        let mut distances = Array2::zeros((n, n));
        
        // 只计算上三角矩阵，利用对称性
        for i in 0..n {
            for j in (i + 1)..n {
                let dist = euclidean_distance_array(data.row(i), data.row(j));
                distances[[i, j]] = dist;
                distances[[j, i]] = dist; // 对称性
            }
        }
        
        Self { distances }
    }
    
    #[inline]
    fn get_distance(&self, i: usize, j: usize) -> f64 {
        self.distances[[i, j]]
    }
}

// 使用ndarray优化的相关性矩阵计算
fn compute_correlation_matrix_ndarray(matrix: &Array2<f64>) -> Array2<f64> {
    let (n_rows, n_cols) = matrix.dim();
    
    // 计算列均值
    let means = matrix.mean_axis(Axis(0)).unwrap();
    
    // 中心化数据
    let mut centered = matrix.clone();
    for mut row in centered.rows_mut() {
        row -= &means;
    }
    
    // 计算协方差矩阵
    let cov_matrix = centered.t().dot(&centered) / (n_rows as f64);
    
    // 计算相关性矩阵
    let mut correlation = Array2::zeros((n_cols, n_cols));
    for i in 0..n_cols {
        for j in 0..n_cols {
            let var_i = cov_matrix[[i, i]];
            let var_j = cov_matrix[[j, j]];
            
            if var_i > 1e-10 && var_j > 1e-10 {
                correlation[[i, j]] = cov_matrix[[i, j]] / (var_i * var_j).sqrt();
            } else {
                correlation[[i, j]] = if i == j { 1.0 } else { 0.0 };
            }
        }
    }
    
    correlation
}

// 使用幂迭代法计算最大特征值
fn compute_largest_eigenvalue_ndarray(matrix: &Array2<f64>, max_iterations: usize) -> f64 {
    let n = matrix.nrows();
    let mut v = Array1::from_elem(n, 1.0 / (n as f64).sqrt());
    let mut eigenvalue = 0.0;
    
    for _ in 0..max_iterations {
        let av = matrix.dot(&v);
        let new_eigenvalue = v.dot(&av);
        
        let norm = av.mapv(|x| x * x).sum().sqrt();
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

// 聚类结果结构体
struct ClusteringResult {
    labels: Vec<usize>,
    centers: Array2<f64>,
    intra_cluster_distances: Vec<f64>,
    cluster_sizes: Vec<f64>,
    silhouette_score: f64,
}

// 优化的K-means聚类
fn optimized_clustering_ndarray(
    data: &Array2<f64>,
    k: usize,
    max_iterations: usize,
) -> ClusteringResult {
    let (n_points, n_dims) = data.dim();
    
    // 简单均匀初始化
    let mut centers = Array2::zeros((k, n_dims));
    let step = n_points / k;
    for i in 0..k {
        let idx = (i * step).min(n_points - 1);
        centers.row_mut(i).assign(&data.row(idx));
    }
    
    let mut labels = vec![0; n_points];
    
    // K-means迭代
    for _ in 0..max_iterations {
        let mut new_centers = Array2::zeros((k, n_dims));
        let mut counts = vec![0; k];
        let mut changed = false;
        
        // 分配点到最近的聚类中心
        for (i, point) in data.rows().into_iter().enumerate() {
            let mut min_dist = f64::INFINITY;
            let mut best_cluster = 0;
            
            for (j, center) in centers.rows().into_iter().enumerate() {
                let dist = euclidean_distance_array(point, center);
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
        
        // 更新聚类中心
        for (i, point) in data.rows().into_iter().enumerate() {
            let cluster = labels[i];
            counts[cluster] += 1;
            let mut center_row = new_centers.row_mut(cluster);
            center_row += &point;
        }
        
        for i in 0..k {
            if counts[i] > 0 {
                let mut center_row = new_centers.row_mut(i);
                center_row /= counts[i] as f64;
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
    
    // 计算簇内平均距离
    let mut cluster_distances = vec![0.0; k];
    let mut cluster_counts = vec![0; k];
    
    for (i, point) in data.rows().into_iter().enumerate() {
        let cluster = labels[i];
        let dist = euclidean_distance_array(point, centers.row(cluster));
        cluster_distances[cluster] += dist;
        cluster_counts[cluster] += 1;
    }
    
    for i in 0..k {
        if cluster_counts[i] > 0 {
            cluster_distances[i] /= cluster_counts[i] as f64;
        }
    }
    
    // 计算轮廓系数
    let precomputed = PrecomputedDistances::new(data);
    let silhouette_score = compute_silhouette_score_ndarray(&precomputed, &labels, k);
    
    ClusteringResult {
        labels,
        centers,
        intra_cluster_distances: cluster_distances,
        cluster_sizes,
        silhouette_score,
    }
}

// 优化的轮廓系数计算
fn compute_silhouette_score_ndarray(
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

// 计算熵
#[inline]
fn compute_entropy_fast(values: &[f64]) -> f64 {
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

// 计算聚类中心相关性
fn compute_center_correlations_ndarray(centers: &Array2<f64>) -> f64 {
    let k = centers.nrows();
    if k < 2 { return 0.0; }
    
    let mut correlations = Vec::new();
    
    for i in 0..k {
        for j in (i + 1)..k {
            let center_i = centers.row(i);
            let center_j = centers.row(j);
            
            let mean_i = center_i.mean().unwrap();
            let mean_j = center_j.mean().unwrap();
            
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
pub fn analyze_sequence_permutations_numpy(
    py: Python,
    sequence: &PyArray1<f64>,
    window_size: Option<usize>,
    n_clusters: Option<usize>,
) -> PyResult<(Py<PyArray2<f64>>, Vec<String>)> {
    let window_size = window_size.unwrap_or(5);
    let n_clusters = n_clusters.unwrap_or(3);
    
    let sequence_array = unsafe { sequence.as_array() };
    
    if sequence_array.len() < window_size {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "序列长度必须大于等于窗口大小"
        ));
    }
    
    let n_windows = sequence_array.len() - window_size + 1;
    let mut results = Array2::from_elem((9, n_windows), f64::NAN);
    
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
    for window_idx in 4..n_windows {
        let window_data = sequence_array.slice(s![window_idx..window_idx + window_size]);
        
        // 生成所有排列的矩阵 (120 x 5)
        let mut permutation_matrix = Array2::zeros((120, window_size));
        
        for (perm_idx, &perm) in PERMUTATIONS.iter().enumerate() {
            for (pos, &idx) in perm.iter().enumerate() {
                permutation_matrix[[perm_idx, pos]] = window_data[idx];
            }
        }
        
        // 计算相关性矩阵 (120 x 120)
        let correlation_matrix = compute_correlation_matrix_ndarray(&permutation_matrix);
        
        // 1. 相关性矩阵均值
        let correlation_mean = correlation_matrix.sum() / (120 * 120) as f64;
        results[[0, window_idx]] = correlation_mean;
        
        // 2. 最大特征值
        let max_eigenvalue = compute_largest_eigenvalue_ndarray(&correlation_matrix, 5);
        results[[1, window_idx]] = max_eigenvalue;
        
        // 执行优化的聚类分析
        let clustering_result = optimized_clustering_ndarray(&permutation_matrix, n_clusters, 3);
        
        // 3. 轮廓系数
        results[[2, window_idx]] = clustering_result.silhouette_score;
        
        // 4. 聚类大小熵
        let size_entropy = compute_entropy_fast(&clustering_result.cluster_sizes);
        results[[3, window_idx]] = size_entropy;
        
        // 5. 最大聚类大小
        let max_cluster_size = clustering_result.cluster_sizes.iter()
            .fold(0.0f64, |a, &b| a.max(b));
        results[[4, window_idx]] = max_cluster_size;
        
        // 6. 簇内平均距离熵
        let distance_entropy = compute_entropy_fast(&clustering_result.intra_cluster_distances);
        results[[5, window_idx]] = distance_entropy;
        
        // 7. 簇内平均距离最大值
        let max_distance = clustering_result.intra_cluster_distances.iter()
            .fold(0.0f64, |a, &b| a.max(b));
        results[[6, window_idx]] = max_distance;
        
        // 8. 簇内平均距离最小值
        let min_distance = clustering_result.intra_cluster_distances.iter()
            .fold(f64::INFINITY, |a, &b| a.min(b));
        results[[7, window_idx]] = if min_distance == f64::INFINITY { 0.0 } else { min_distance };
        
        // 9. 聚类中心相关性均值
        let center_correlation = compute_center_correlations_ndarray(&clustering_result.centers);
        results[[8, window_idx]] = center_correlation;
    }
    
    // 将结果转换为Python NumPy数组
    let py_array = PyArray2::from_owned_array(py, results);
    
    Ok((py_array.into(), column_names))
}