use nalgebra::{DMatrix, SymmetricEigen};
use ndarray::{Array1, Array2};
use numpy::{PyArray1, PyArray2};
use pyo3::prelude::*;
use rand::prelude::*;

#[pyfunction]
pub fn analyze_sequence_permutations_v0816_fixed(
    py: Python,
    sequence: &PyArray1<f64>,
    window_size: Option<usize>,
    n_clusters: Option<usize>,
) -> PyResult<(Py<PyArray2<f64>>, Vec<String>)> {
    let window_size = window_size.unwrap_or(5);
    let n_clusters = n_clusters.unwrap_or(3);

    let binding = sequence.readonly();
    let sequence_arr = binding.as_array();
    let n = sequence_arr.len();

    if n < window_size {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "序列长度不能小于窗口大小",
        ));
    }

    let n_windows = n - window_size + 1;
    let indicator_names = vec![
        "相关性矩阵偏度".to_string(),
        "相关性矩阵峰度".to_string(),
        "轮廓系数".to_string(),
        "聚类大小熵".to_string(),
        "最大聚类大小".to_string(),
        "簇内平均距离熵".to_string(),
        "簇内平均距离最大值".to_string(),
        "簇内平均距离最小值".to_string(),
        "聚类中心相关性均值".to_string(),
    ];

    let mut results = Array2::<f64>::zeros((n_windows, 9));

    for i in 0..n_windows {
        let window_data = sequence_arr.slice(ndarray::s![i..i + window_size]);

        if i < 4 {
            for j in 0..9 {
                results[[i, j]] = f64::NAN;
            }
            continue;
        }

        let indicators =
            compute_window_indicators_fixed(&window_data.to_owned(), window_size, n_clusters);

        for j in 0..9 {
            results[[i, j]] = indicators[j];
        }
    }

    let results_transposed = results.reversed_axes();
    let py_result = PyArray2::from_array(py, &results_transposed);

    Ok((py_result.to_owned(), indicator_names))
}

fn compute_window_indicators_fixed(
    window_data: &Array1<f64>,
    window_size: usize,
    n_clusters: usize,
) -> Vec<f64> {
    // 检查唯一值数量
    let mut unique_values = std::collections::HashSet::new();
    for &val in window_data.iter() {
        unique_values.insert(ordered_float::OrderedFloat(val));
    }

    if unique_values.len() < 3 {
        return vec![f64::NAN; 9];
    }

    // 生成所有排列（按照与Python itertools.permutations相同的顺序）
    let permutations = generate_permutations_fixed(window_size);

    if permutations.is_empty() {
        return vec![f64::NAN; 9];
    }

    // 构建排列矩阵：行是排列，列是特征（与Python实现一致）
    let n_perms = permutations.len();
    let mut perm_data = Array2::<f64>::zeros((n_perms, window_size));

    for (i, perm) in permutations.iter().enumerate() {
        for (j, &idx) in perm.iter().enumerate() {
            perm_data[[i, j]] = window_data[idx];
        }
    }

    // 1. 计算相关性矩阵（与Python numpy.corrcoef一致）
    let corr_matrix = compute_correlation_matrix_fixed(&perm_data);

    // 1-2. 计算相关性矩阵的统计指标（排除对角线元素）
    let mut off_diagonal_values = Vec::new();
    for i in 0..n_perms {
        for j in 0..n_perms {
            if i != j {
                off_diagonal_values.push(corr_matrix[[i, j]]);
            }
        }
    }

    let corr_skew = compute_skewness_fixed(&off_diagonal_values);
    let corr_kurt = compute_kurtosis_fixed(&off_diagonal_values);

    // 3-9. 聚类分析相关指标（使用固定种子确保可重现性）
    let (silhouette_score, cluster_sizes, intra_cluster_distances, centroids) =
        perform_clustering_analysis_fixed(&perm_data, n_clusters);

    let cluster_size_entropy =
        compute_entropy_fixed(&cluster_sizes.iter().map(|&x| x as f64).collect::<Vec<_>>());
    let max_cluster_size = *cluster_sizes.iter().max().unwrap_or(&0) as f64;

    let intra_dist_entropy = compute_entropy_fixed(&intra_cluster_distances);
    let intra_dist_max = intra_cluster_distances
        .iter()
        .fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let intra_dist_min = intra_cluster_distances
        .iter()
        .fold(f64::INFINITY, |a, &b| a.min(b));

    let centroid_corr_mean = compute_centroid_correlation_mean_fixed(&centroids);

    vec![
        corr_skew,
        corr_kurt,
        silhouette_score,
        cluster_size_entropy,
        max_cluster_size,
        intra_dist_entropy,
        intra_dist_max,
        intra_dist_min,
        centroid_corr_mean,
    ]
}

fn generate_permutations_fixed(n: usize) -> Vec<Vec<usize>> {
    // 生成与Python itertools.permutations相同顺序的排列
    let mut indices: Vec<usize> = (0..n).collect();
    let mut permutations = Vec::new();
    generate_permutations_recursive_fixed(&mut indices, 0, &mut permutations);
    permutations
}

fn generate_permutations_recursive_fixed(
    indices: &mut Vec<usize>,
    start: usize,
    permutations: &mut Vec<Vec<usize>>,
) {
    if start == indices.len() {
        permutations.push(indices.clone());
        return;
    }

    for i in start..indices.len() {
        indices.swap(start, i);
        generate_permutations_recursive_fixed(indices, start + 1, permutations);
        indices.swap(start, i);
    }
}

fn compute_correlation_matrix_fixed(data: &Array2<f64>) -> Array2<f64> {
    // 实现与numpy.corrcoef完全一致的相关性矩阵计算
    let (n_samples, n_features) = data.dim();
    let mut corr_matrix = Array2::<f64>::zeros((n_samples, n_samples));

    // 计算每个样本的均值
    let mut means = vec![0.0; n_samples];
    for i in 0..n_samples {
        means[i] = data.row(i).sum() / n_features as f64;
    }

    // 计算协方差矩阵和标准差
    for i in 0..n_samples {
        for j in i..n_samples {
            if i == j {
                corr_matrix[[i, j]] = 1.0;
            } else {
                let mut covariance = 0.0;
                let mut var_i = 0.0;
                let mut var_j = 0.0;

                for k in 0..n_features {
                    let diff_i = data[[i, k]] - means[i];
                    let diff_j = data[[j, k]] - means[j];
                    covariance += diff_i * diff_j;
                    var_i += diff_i * diff_i;
                    var_j += diff_j * diff_j;
                }

                let correlation = if var_i == 0.0 || var_j == 0.0 {
                    0.0
                } else {
                    covariance / (var_i * var_j).sqrt()
                };

                corr_matrix[[i, j]] = correlation;
                corr_matrix[[j, i]] = correlation;
            }
        }
    }

    corr_matrix
}

#[allow(dead_code)]
fn compute_max_eigenvalue_fixed(matrix: &Array2<f64>) -> f64 {
    let n = matrix.nrows();
    let mut nalgebra_matrix = DMatrix::<f64>::zeros(n, n);

    for i in 0..n {
        for j in 0..n {
            nalgebra_matrix[(i, j)] = matrix[[i, j]];
        }
    }

    let eigen = SymmetricEigen::new(nalgebra_matrix);
    eigen
        .eigenvalues
        .iter()
        .fold(f64::NEG_INFINITY, |a, &b| a.max(b))
}

fn perform_clustering_analysis_fixed(
    data: &Array2<f64>,
    n_clusters: usize,
) -> (f64, Vec<usize>, Vec<f64>, Array2<f64>) {
    let (cluster_assignments, centroids) = kmeans_clustering_fixed(data, n_clusters);

    let silhouette_score = compute_silhouette_score_fixed(data, &cluster_assignments, &centroids);

    let cluster_sizes = compute_cluster_sizes_fixed(&cluster_assignments, n_clusters);

    let intra_cluster_distances = compute_intra_cluster_distances_fixed(data, &cluster_assignments);

    (
        silhouette_score,
        cluster_sizes,
        intra_cluster_distances,
        centroids,
    )
}

fn kmeans_clustering_fixed(data: &Array2<f64>, k: usize) -> (Vec<usize>, Array2<f64>) {
    let (n_points, n_features) = data.dim();

    // 使用固定种子42确保可重现性（与Python sklearn一致）
    let mut rng = StdRng::seed_from_u64(42);

    // 随机初始化质心
    let mut centroids = Array2::<f64>::zeros((k, n_features));
    for i in 0..k {
        let random_point = rng.gen_range(0..n_points);
        for j in 0..n_features {
            centroids[[i, j]] = data[[random_point, j]];
        }
    }

    let mut assignments = vec![0; n_points];
    let max_iterations = 300; // 与sklearn默认值一致

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
                    let diff =
                        data[[point_idx, feature_idx]] - centroids[[cluster_idx, feature_idx]];
                    distance += diff * diff;
                }
                distance = distance.sqrt();

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
                        .map(|&point_idx| data[[point_idx, feature_idx]])
                        .sum::<f64>()
                        / cluster_points.len() as f64;
                    centroids[[cluster_idx, feature_idx]] = mean;
                }
            }
        }
    }

    (assignments, centroids)
}

fn compute_silhouette_score_fixed(
    data: &Array2<f64>,
    assignments: &[usize],
    _centroids: &Array2<f64>,
) -> f64 {
    let n_points = data.nrows();
    let mut silhouette_scores = Vec::new();

    for i in 0..n_points {
        let cluster = assignments[i];

        // 计算簇内平均距离
        let a = compute_intra_cluster_distance_fixed(data, assignments, cluster, i);

        // 计算到最近其他簇的平均距离
        let b = compute_nearest_cluster_distance_fixed(data, assignments, cluster, i);

        let silhouette = if a.max(b) == 0.0 {
            0.0
        } else {
            (b - a) / a.max(b)
        };

        silhouette_scores.push(silhouette);
    }

    silhouette_scores.iter().sum::<f64>() / silhouette_scores.len() as f64
}

fn compute_intra_cluster_distance_fixed(
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
    let n_features = data.ncols();

    for &other_idx in &cluster_points {
        let mut distance = 0.0;
        for feature_idx in 0..n_features {
            let diff = data[[point_idx, feature_idx]] - data[[other_idx, feature_idx]];
            distance += diff * diff;
        }
        total_distance += distance.sqrt();
    }

    total_distance / cluster_points.len() as f64
}

fn compute_nearest_cluster_distance_fixed(
    data: &Array2<f64>,
    assignments: &[usize],
    current_cluster: usize,
    point_idx: usize,
) -> f64 {
    let n_clusters = assignments.iter().max().unwrap_or(&0) + 1;
    let mut min_distance = f64::INFINITY;

    for cluster_idx in 0..n_clusters {
        if cluster_idx != current_cluster {
            let cluster_points: Vec<_> = assignments
                .iter()
                .enumerate()
                .filter(|(_, &assignment)| assignment == cluster_idx)
                .map(|(idx, _)| idx)
                .collect();

            if !cluster_points.is_empty() {
                let mut total_distance = 0.0;
                let n_features = data.ncols();

                for &other_idx in &cluster_points {
                    let mut distance = 0.0;
                    for feature_idx in 0..n_features {
                        let diff = data[[point_idx, feature_idx]] - data[[other_idx, feature_idx]];
                        distance += diff * diff;
                    }
                    total_distance += distance.sqrt();
                }

                let avg_distance = total_distance / cluster_points.len() as f64;
                min_distance = min_distance.min(avg_distance);
            }
        }
    }

    min_distance
}

fn compute_cluster_sizes_fixed(assignments: &[usize], n_clusters: usize) -> Vec<usize> {
    let mut sizes = vec![0; n_clusters];
    for &assignment in assignments {
        if assignment < n_clusters {
            sizes[assignment] += 1;
        }
    }
    sizes
}

fn compute_intra_cluster_distances_fixed(data: &Array2<f64>, assignments: &[usize]) -> Vec<f64> {
    let n_clusters = assignments.iter().max().unwrap_or(&0) + 1;
    let mut distances = Vec::new();
    let n_features = data.ncols();

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
                        let diff = data[[cluster_points[i], feature_idx]]
                            - data[[cluster_points[j], feature_idx]];
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

fn compute_entropy_fixed(values: &[f64]) -> f64 {
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

fn compute_centroid_correlation_mean_fixed(centroids: &Array2<f64>) -> f64 {
    let n_centroids = centroids.nrows();
    if n_centroids < 2 {
        return f64::NAN;
    }

    let mut correlations = Vec::new();
    let n_features = centroids.ncols();

    for i in 0..n_centroids {
        for j in i + 1..n_centroids {
            // 计算质心间的皮尔逊相关系数
            let mut sum_xy = 0.0;
            let mut sum_x = 0.0;
            let mut sum_y = 0.0;
            let mut sum_x_sq = 0.0;
            let mut sum_y_sq = 0.0;

            for k in 0..n_features {
                let x = centroids[[i, k]];
                let y = centroids[[j, k]];
                sum_xy += x * y;
                sum_x += x;
                sum_y += y;
                sum_x_sq += x * x;
                sum_y_sq += y * y;
            }

            let n = n_features as f64;
            let numerator = n * sum_xy - sum_x * sum_y;
            let denominator =
                ((n * sum_x_sq - sum_x * sum_x) * (n * sum_y_sq - sum_y * sum_y)).sqrt();

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

#[allow(dead_code)]
fn compute_std_fixed(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return f64::NAN;
    }

    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let variance =
        values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (values.len() - 1) as f64;

    variance.sqrt()
}

fn compute_skewness_fixed(values: &[f64]) -> f64 {
    if values.len() < 3 {
        return f64::NAN;
    }

    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;

    // 计算标准差（总体标准差，与scipy一致）
    let variance = values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n;
    let std_dev = variance.sqrt();

    if std_dev == 0.0 {
        return f64::NAN;
    }

    // 计算三阶标准化矩（与scipy.stats.skew一致）
    let m3 = values
        .iter()
        .map(|&x| ((x - mean) / std_dev).powi(3))
        .sum::<f64>()
        / n;

    m3
}

fn compute_kurtosis_fixed(values: &[f64]) -> f64 {
    if values.len() < 4 {
        return f64::NAN;
    }

    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;

    // 计算标准差（总体标准差，与scipy一致）
    let variance = values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n;
    let std_dev = variance.sqrt();

    if std_dev == 0.0 {
        return f64::NAN;
    }

    // 计算四阶标准化矩
    let m4 = values
        .iter()
        .map(|&x| ((x - mean) / std_dev).powi(4))
        .sum::<f64>()
        / n;

    // 计算超额峰度（减去3，与scipy.stats.kurtosis默认行为一致）
    let kurtosis = m4 - 3.0;

    kurtosis
}
