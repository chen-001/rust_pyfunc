use pyo3::prelude::*;
use numpy::{PyArray1, PyArray2};
use ndarray::{Array1, Array2, Axis};
use nalgebra::{DMatrix, SymmetricEigen};
use rand::prelude::*;
use std::collections::HashMap;

#[pyfunction]
pub fn analyze_sequence_permutations_v0816(
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
    
    for i in 0..n_windows {
        let window_data = sequence_arr.slice(ndarray::s![i..i + window_size]);
        
        if i < 4 {
            for j in 0..9 {
                results[[i, j]] = f64::NAN;
            }
            continue;
        }
        
        let indicators = compute_window_indicators(&window_data.to_owned(), window_size, n_clusters);
        
        for j in 0..9 {
            results[[i, j]] = indicators[j];
        }
    }
    
    let results_transposed = results.reversed_axes();
    let py_result = PyArray2::from_array(py, &results_transposed);
    
    Ok((py_result.to_owned(), indicator_names))
}

fn compute_window_indicators(window_data: &Array1<f64>, window_size: usize, n_clusters: usize) -> Vec<f64> {
    let unique_values: std::collections::HashSet<ordered_float::OrderedFloat<f64>> = 
        window_data.iter().map(|&x| ordered_float::OrderedFloat(x)).collect();
    
    if unique_values.len() < 3 {
        return vec![f64::NAN; 9];
    }
    
    let permutations = generate_permutations(window_data);
    
    if permutations.is_empty() {
        return vec![f64::NAN; 9];
    }
    
    let perm_matrix = Array2::from_shape_vec(
        (window_size, permutations.len()),
        permutations.into_iter().flatten().collect()
    ).unwrap();
    
    let corr_matrix = compute_correlation_matrix(&perm_matrix);
    let corr_mean = corr_matrix.mean().unwrap_or(f64::NAN);
    
    let max_eigenvalue = compute_max_eigenvalue(&corr_matrix);
    
    let (silhouette_score, cluster_sizes, intra_cluster_distances, centroids) = 
        perform_clustering_analysis(&perm_matrix, n_clusters);
    
    let cluster_size_entropy = compute_entropy(&cluster_sizes.iter().map(|&x| x as f64).collect::<Vec<_>>());
    let max_cluster_size = *cluster_sizes.iter().max().unwrap_or(&0) as f64;
    
    let intra_dist_entropy = compute_entropy(&intra_cluster_distances);
    let intra_dist_max = intra_cluster_distances.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let intra_dist_min = intra_cluster_distances.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    
    let centroid_corr_mean = compute_centroid_correlation_mean(&centroids);
    
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

fn generate_permutations(data: &Array1<f64>) -> Vec<Vec<f64>> {
    let mut indices: Vec<usize> = (0..data.len()).collect();
    let mut permutations = Vec::new();
    
    generate_permutations_recursive(&mut indices, 0, data, &mut permutations);
    
    permutations
}

fn generate_permutations_recursive(
    indices: &mut Vec<usize>,
    start: usize,
    data: &Array1<f64>,
    permutations: &mut Vec<Vec<f64>>,
) {
    if start == indices.len() {
        let perm: Vec<f64> = indices.iter().map(|&i| data[i]).collect();
        permutations.push(perm);
        return;
    }
    
    for i in start..indices.len() {
        indices.swap(start, i);
        generate_permutations_recursive(indices, start + 1, data, permutations);
        indices.swap(start, i);
    }
}

fn compute_correlation_matrix(data: &Array2<f64>) -> Array2<f64> {
    let n_cols = data.ncols();
    let mut corr_matrix = Array2::<f64>::zeros((n_cols, n_cols));
    
    for i in 0..n_cols {
        for j in i..n_cols {
            let col_i = data.column(i);
            let col_j = data.column(j);
            let corr = pearson_correlation(&col_i.to_owned(), &col_j.to_owned());
            corr_matrix[[i, j]] = corr;
            corr_matrix[[j, i]] = corr;
        }
    }
    
    corr_matrix
}

fn pearson_correlation(x: &Array1<f64>, y: &Array1<f64>) -> f64 {
    let n = x.len() as f64;
    if n == 0.0 {
        return f64::NAN;
    }
    
    let mean_x = x.mean().unwrap_or(0.0);
    let mean_y = y.mean().unwrap_or(0.0);
    
    let mut num = 0.0;
    let mut den_x = 0.0;
    let mut den_y = 0.0;
    
    for i in 0..x.len() {
        let dx = x[i] - mean_x;
        let dy = y[i] - mean_y;
        num += dx * dy;
        den_x += dx * dx;
        den_y += dy * dy;
    }
    
    if den_x == 0.0 || den_y == 0.0 {
        return f64::NAN;
    }
    
    num / (den_x * den_y).sqrt()
}

fn compute_max_eigenvalue(matrix: &Array2<f64>) -> f64 {
    let n = matrix.nrows();
    let mut nalgebra_matrix = DMatrix::<f64>::zeros(n, n);
    
    for i in 0..n {
        for j in 0..n {
            nalgebra_matrix[(i, j)] = matrix[[i, j]];
        }
    }
    
    let eigen = SymmetricEigen::new(nalgebra_matrix);
    eigen.eigenvalues.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b))
}

fn perform_clustering_analysis(
    data: &Array2<f64>,
    n_clusters: usize,
) -> (f64, Vec<usize>, Vec<f64>, Array2<f64>) {
    let (cluster_assignments, centroids) = kmeans_clustering(data, n_clusters);
    
    let silhouette_score = compute_silhouette_score(data, &cluster_assignments, &centroids);
    
    let cluster_sizes = compute_cluster_sizes(&cluster_assignments, n_clusters);
    
    let intra_cluster_distances = compute_intra_cluster_distances(data, &cluster_assignments, &centroids);
    
    (silhouette_score, cluster_sizes, intra_cluster_distances, centroids)
}

fn kmeans_clustering(data: &Array2<f64>, k: usize) -> (Vec<usize>, Array2<f64>) {
    let (n_features, n_points) = data.dim();
    let mut rng = thread_rng();
    
    let mut centroids = Array2::<f64>::zeros((n_features, k));
    for i in 0..k {
        for j in 0..n_features {
            centroids[[j, i]] = rng.gen_range(-1.0..1.0);
        }
    }
    
    let mut assignments = vec![0; n_points];
    let max_iterations = 100;
    
    for _iteration in 0..max_iterations {
        let mut new_assignments = vec![0; n_points];
        
        for point_idx in 0..n_points {
            let point = data.column(point_idx);
            let mut min_distance = f64::INFINITY;
            let mut best_cluster = 0;
            
            for cluster_idx in 0..k {
                let centroid = centroids.column(cluster_idx);
                let distance = euclidean_distance(&point.to_owned(), &centroid.to_owned());
                if distance < min_distance {
                    min_distance = distance;
                    best_cluster = cluster_idx;
                }
            }
            
            new_assignments[point_idx] = best_cluster;
        }
        
        if assignments == new_assignments {
            break;
        }
        assignments = new_assignments;
        
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

fn euclidean_distance(a: &Array1<f64>, b: &Array1<f64>) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}

fn compute_silhouette_score(
    data: &Array2<f64>,
    assignments: &[usize],
    centroids: &Array2<f64>,
) -> f64 {
    let n_points = data.ncols();
    let mut silhouette_scores = Vec::new();
    
    for i in 0..n_points {
        let point = data.column(i);
        let cluster = assignments[i];
        
        let a = compute_intra_cluster_distance(&point.to_owned(), data, assignments, cluster, i);
        
        let b = compute_nearest_cluster_distance(&point.to_owned(), centroids, cluster);
        
        let silhouette = if a.max(b) == 0.0 {
            0.0
        } else {
            (b - a) / a.max(b)
        };
        
        silhouette_scores.push(silhouette);
    }
    
    silhouette_scores.iter().sum::<f64>() / silhouette_scores.len() as f64
}

fn compute_intra_cluster_distance(
    point: &Array1<f64>,
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
    
    let total_distance: f64 = cluster_points
        .iter()
        .map(|&idx| euclidean_distance(point, &data.column(idx).to_owned()))
        .sum();
    
    total_distance / cluster_points.len() as f64
}

fn compute_nearest_cluster_distance(
    point: &Array1<f64>,
    centroids: &Array2<f64>,
    current_cluster: usize,
) -> f64 {
    let mut min_distance = f64::INFINITY;
    
    for cluster_idx in 0..centroids.ncols() {
        if cluster_idx != current_cluster {
            let centroid = centroids.column(cluster_idx);
            let distance = euclidean_distance(point, &centroid.to_owned());
            min_distance = min_distance.min(distance);
        }
    }
    
    min_distance
}

fn compute_cluster_sizes(assignments: &[usize], n_clusters: usize) -> Vec<usize> {
    let mut sizes = vec![0; n_clusters];
    for &assignment in assignments {
        sizes[assignment] += 1;
    }
    sizes
}

fn compute_intra_cluster_distances(
    data: &Array2<f64>,
    assignments: &[usize],
    centroids: &Array2<f64>,
) -> Vec<f64> {
    let n_clusters = centroids.ncols();
    let mut distances = Vec::new();
    
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
                    let point1 = data.column(cluster_points[i]);
                    let point2 = data.column(cluster_points[j]);
                    total_distance += euclidean_distance(&point1.to_owned(), &point2.to_owned());
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

fn compute_entropy(values: &[f64]) -> f64 {
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

fn compute_centroid_correlation_mean(centroids: &Array2<f64>) -> f64 {
    let n_centroids = centroids.ncols();
    if n_centroids < 2 {
        return f64::NAN;
    }
    
    let mut correlations = Vec::new();
    
    for i in 0..n_centroids {
        for j in i + 1..n_centroids {
            let centroid1 = centroids.column(i);
            let centroid2 = centroids.column(j);
            let corr = pearson_correlation(&centroid1.to_owned(), &centroid2.to_owned());
            if !corr.is_nan() {
                correlations.push(corr);
            }
        }
    }
    
    if correlations.is_empty() {
        f64::NAN
    } else {
        correlations.iter().sum::<f64>() / correlations.len() as f64
    }
}