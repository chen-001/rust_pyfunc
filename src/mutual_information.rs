use numpy::PyReadonlyArray1;
use pyo3::prelude::*;
use std::cmp::Ordering;

/// A data point in 2D space (X, Y)
#[derive(Clone, Copy, Debug)]
struct Point2D {
    x: f64,
    y: f64,
}

impl Point2D {
    fn new(x: f64, y: f64) -> Self {
        Point2D { x, y }
    }

    /// Calculate Euclidean distance to another point
    fn distance(&self, other: &Point2D) -> f64 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        (dx * dx + dy * dy).sqrt()
    }

    /// Calculate Chebyshev distance (max norm) to another point
    fn chebyshev_distance(&self, other: &Point2D) -> f64 {
        let dx = (self.x - other.x).abs();
        let dy = (self.y - other.y).abs();
        dx.max(dy)
    }
}

/// KD-Tree node for 2D points
#[derive(Debug)]
struct KDNode {
    point: Point2D,
    index: usize, // Original index of the point
    left: Option<Box<KDNode>>,
    right: Option<Box<KDNode>>,
    axis: usize, // 0 for x, 1 for y
}

impl KDNode {
    fn new(point: Point2D, index: usize, axis: usize) -> Self {
        KDNode {
            point,
            index,
            left: None,
            right: None,
            axis,
        }
    }
}

/// KD-Tree implementation for efficient k-nearest neighbor search
#[derive(Debug)]
struct KDTree {
    root: Option<Box<KDNode>>,
    points: Vec<Point2D>, // Store all points for distance calculations
}

impl KDTree {
    /// Build a KD-Tree from a list of points
    fn build(points: &[Point2D]) -> Self {
        let indices: Vec<usize> = (0..points.len()).collect();
        let root = if !indices.is_empty() {
            Some(Self::build_recursive(points, &indices, 0))
        } else {
            None
        };

        KDTree {
            root,
            points: points.to_vec(),
        }
    }

    /// Recursive build function
    fn build_recursive(points: &[Point2D], indices: &[usize], depth: usize) -> Box<KDNode> {
        if indices.is_empty() {
            panic!("Cannot build KD-Tree from empty index list");
        }

        let axis = depth % 2;
        let mut indices = indices.to_vec();

        // Sort by the axis and find median
        indices.sort_by(|&a, &b| {
            let coord_a = if axis == 0 { points[a].x } else { points[a].y };
            let coord_b = if axis == 0 { points[b].x } else { points[b].y };
            coord_a.partial_cmp(&coord_b).unwrap_or(Ordering::Equal)
        });

        let mid = indices.len() / 2;
        let median_index = indices[mid];
        let median_point = points[median_index];

        // Build left and right subtrees
        let left_indices = &indices[..mid];
        let right_indices = &indices[mid + 1..];

        let mut node = Box::new(KDNode::new(median_point, median_index, axis));

        if !left_indices.is_empty() {
            node.left = Some(Self::build_recursive(points, left_indices, depth + 1));
        }

        if !right_indices.is_empty() {
            node.right = Some(Self::build_recursive(points, right_indices, depth + 1));
        }

        node
    }

    /// Find k nearest neighbors using Euclidean distance
    fn k_nearest_euclidean(&self, target: &Point2D, k: usize) -> Vec<(usize, f64)> {
        let mut results = Vec::with_capacity(k);
        if let Some(ref root) = self.root {
            Self::search_recursive_euclidean(root, target, k, &mut results, 0.0);
        }
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        results
    }

    /// Find k nearest neighbors using Chebyshev distance
    fn k_nearest_chebyshev(&self, target: &Point2D, k: usize) -> Vec<(usize, f64)> {
        let mut results = Vec::with_capacity(k);
        if let Some(ref root) = self.root {
            Self::search_recursive_chebyshev(root, target, k, &mut results, 0.0);
        }
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        results
    }

    /// Recursive search for k nearest neighbors (Euclidean)
    fn search_recursive_euclidean(
        node: &KDNode,
        target: &Point2D,
        k: usize,
        results: &mut Vec<(usize, f64)>,
        best_dist: f64,
    ) {
        let dist = node.point.distance(target);
        if dist > 0.0 {
            // Don't add the point itself
            if results.len() < k {
                results.push((node.index, dist));
            } else {
                // Find current maximum distance
                let mut max_idx = 0;
                let mut max_dist = results[0].1;
                for (i, &(_, d)) in results.iter().enumerate() {
                    if d > max_dist {
                        max_dist = d;
                        max_idx = i;
                    }
                }
                if dist < max_dist {
                    results[max_idx] = (node.index, dist);
                }
            }
        }

        let axis = node.axis;
        let target_coord = if axis == 0 { target.x } else { target.y };
        let node_coord = if axis == 0 {
            node.point.x
        } else {
            node.point.y
        };

        // Determine which side to search first
        let near_subtree = if target_coord < node_coord {
            &node.left
        } else {
            &node.right
        };
        let far_subtree = if target_coord < node_coord {
            &node.right
        } else {
            &node.left
        };

        // Search near subtree
        if let Some(ref child) = near_subtree {
            Self::search_recursive_euclidean(child, target, k, results, best_dist);
        }

        // Check if we need to search far subtree
        let current_max_dist = if !results.is_empty() {
            results
                .iter()
                .fold(0.0, |max, &(_, d)| if d > max { d } else { max })
        } else {
            f64::INFINITY
        };

        let diff = (target_coord - node_coord).abs();
        if results.len() < k || diff < current_max_dist {
            if let Some(ref child) = far_subtree {
                Self::search_recursive_euclidean(child, target, k, results, best_dist);
            }
        }
    }

    /// Recursive search for k nearest neighbors (Chebyshev)
    fn search_recursive_chebyshev(
        node: &KDNode,
        target: &Point2D,
        k: usize,
        results: &mut Vec<(usize, f64)>,
        best_dist: f64,
    ) {
        let dist = node.point.chebyshev_distance(target);
        if dist > 0.0 {
            // Don't add the point itself
            if results.len() < k {
                results.push((node.index, dist));
            } else {
                // Find current maximum distance
                let mut max_idx = 0;
                let mut max_dist = results[0].1;
                for (i, &(_, d)) in results.iter().enumerate() {
                    if d > max_dist {
                        max_dist = d;
                        max_idx = i;
                    }
                }
                if dist < max_dist {
                    results[max_idx] = (node.index, dist);
                }
            }
        }

        let axis = node.axis;
        let target_coord = if axis == 0 { target.x } else { target.y };
        let node_coord = if axis == 0 {
            node.point.x
        } else {
            node.point.y
        };

        // Determine which side to search first
        let near_subtree = if target_coord < node_coord {
            &node.left
        } else {
            &node.right
        };
        let far_subtree = if target_coord < node_coord {
            &node.right
        } else {
            &node.left
        };

        // Search near subtree
        if let Some(ref child) = near_subtree {
            Self::search_recursive_chebyshev(child, target, k, results, best_dist);
        }

        // Check if we need to search far subtree
        let current_max_dist = if !results.is_empty() {
            results
                .iter()
                .fold(0.0, |max, &(_, d)| if d > max { d } else { max })
        } else {
            f64::INFINITY
        };

        let diff = (target_coord - node_coord).abs();
        if results.len() < k || diff < current_max_dist {
            if let Some(ref child) = far_subtree {
                Self::search_recursive_chebyshev(child, target, k, results, best_dist);
            }
        }
    }
}

/// K-nearest neighbors information for a point
#[derive(Debug)]
struct KNNInfo {
    kth_distance: f64,     // Distance to k-th nearest neighbor
    x_distances: Vec<f64>, // X-coordinates of k nearest neighbors
    y_distances: Vec<f64>, // Y-coordinates of k nearest neighbors
}

/// Calculate mutual information using KSG (Kraskov-Stögbauer-Grassberger) method 1
/// Uses Euclidean distance for k-nearest neighbor search
#[pyfunction]
pub fn mutual_information_knn(
    _py: Python,
    x: PyReadonlyArray1<f64>,
    y: PyReadonlyArray1<f64>,
    k: Option<usize>,
) -> PyResult<f64> {
    let x_slice = x.as_slice()?;
    let y_slice = y.as_slice()?;

    if x_slice.len() != y_slice.len() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "x and y must have the same length",
        ));
    }

    let n = x_slice.len();
    if n == 0 {
        return Ok(0.0);
    }

    let k = k.unwrap_or(3);
    if k >= n {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "k must be less than the length of the arrays",
        ));
    }

    let ksg_method1 = calculate_mi_ksg_method1(x_slice, y_slice, k);
    Ok(ksg_method1)
}

/// Calculate mutual information using KSG method 1 (Euclidean distance) with KD-Tree optimization
fn calculate_mi_ksg_method1(x: &[f64], y: &[f64], k: usize) -> f64 {
    let n = x.len();

    // Create 2D points from (x, y) pairs
    let mut points: Vec<Point2D> = Vec::with_capacity(n);
    for i in 0..n {
        points.push(Point2D::new(x[i], y[i]));
    }

    // Calculate k-nearest neighbors for all points using a single KD-Tree
    let psi_values = calculate_all_knn_info(&points, k);

    // Calculate digamma function values
    let digamma_n = digamma(n as f64);
    let digamma_k = digamma(k as f64);

    // Pre-sort x and y arrays for efficient range counting
    let mut x_sorted: Vec<(usize, f64)> = (0..n).map(|i| (i, x[i])).collect();
    let mut y_sorted: Vec<(usize, f64)> = (0..n).map(|i| (i, y[i])).collect();

    x_sorted.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
    y_sorted.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));

    // Calculate mutual information using KSG formula
    let mut mi_sum = 0.0;
    for i in 0..n {
        let psi = &psi_values[i];
        let epsilon = psi.kth_distance;

        // X-dimension: count points where |x_i - x_j| < ε_i using binary search
        let count_x = count_in_range(&x_sorted, x[i], epsilon);
        let neighbors_x = count_x.saturating_sub(1); // exclude self

        // Y-dimension: count points where |y_i - y_j| < ε_i using binary search
        let count_y = count_in_range(&y_sorted, y[i], epsilon);
        let neighbors_y = count_y.saturating_sub(1); // exclude self

        // Correct KSG method 1 formula
        let mi_i = digamma_k + digamma_n
            - digamma((neighbors_x + 1) as f64)
            - digamma((neighbors_y + 1) as f64);
        mi_sum += mi_i;
    }

    // Average over all points
    mi_sum / n as f64
}

/// Helper function: count points in range [value - epsilon, value + epsilon] using binary search
fn count_in_range(sorted_array: &[(usize, f64)], value: f64, epsilon: f64) -> usize {
    let min_val = value - epsilon;
    let max_val = value + epsilon;

    // Find left boundary (first index >= min_val)
    let left_idx = sorted_array.partition_point(|&(_, v)| v < min_val);

    // Find right boundary (first index > max_val)
    let right_idx = sorted_array.partition_point(|&(_, v)| v <= max_val);

    right_idx - left_idx
}

/// Calculate k-nearest neighbor information for all points using a single KD-Tree
fn calculate_all_knn_info(points: &[Point2D], k: usize) -> Vec<KNNInfo> {
    let n = points.len();

    // Build KD-Tree once for all points
    let kdtree = KDTree::build(points);

    // Calculate k-nearest neighbors for each point
    let mut all_knn_info = Vec::with_capacity(n);

    for idx in 0..n {
        let target = points[idx];

        // Find k nearest neighbors
        let neighbors = kdtree.k_nearest_euclidean(&target, k);

        // Get k-th distance
        let kth_distance = if !neighbors.is_empty() {
            neighbors[k - 1].1
        } else {
            0.0
        };

        let mut x_distances = Vec::with_capacity(k);
        let mut y_distances = Vec::with_capacity(k);

        for i in 0..k {
            if i < neighbors.len() {
                let neighbor_idx = neighbors[i].0;
                let _dist = neighbors[i].1;

                // Store normalized distances
                x_distances
                    .push((points[neighbor_idx].x - target.x).abs() / kth_distance.max(1e-10));
                y_distances
                    .push((points[neighbor_idx].y - target.y).abs() / kth_distance.max(1e-10));
            }
        }

        all_knn_info.push(KNNInfo {
            kth_distance: kth_distance.max(1e-10),
            x_distances,
            y_distances,
        });
    }

    all_knn_info
}

/// Calculate k-nearest neighbor information for all points using a single KD-Tree (Chebyshev)
fn calculate_all_knn_info_chebyshev(points: &[Point2D], k: usize) -> Vec<KNNInfo> {
    let n = points.len();

    // Build KD-Tree once for all points
    let kdtree = KDTree::build(points);

    // Calculate k-nearest neighbors for each point
    let mut all_knn_info = Vec::with_capacity(n);

    for idx in 0..n {
        let target = points[idx];

        // Find k nearest neighbors using Chebyshev distance
        let neighbors = kdtree.k_nearest_chebyshev(&target, k);

        let (kth_distance, x_distances, y_distances) = if neighbors.len() < k {
            let fallback = calculate_knn_info_chebyshev(points, idx, k);
            (
                fallback.kth_distance,
                fallback.x_distances,
                fallback.y_distances,
            )
        } else {
            let kth_distance = neighbors[k - 1].1;
            let mut x_distances = Vec::with_capacity(k);
            let mut y_distances = Vec::with_capacity(k);

            for i in 0..k {
                let neighbor_idx = neighbors[i].0;

                // Store normalized distances
                x_distances
                    .push((points[neighbor_idx].x - target.x).abs() / kth_distance.max(1e-10));
                y_distances
                    .push((points[neighbor_idx].y - target.y).abs() / kth_distance.max(1e-10));
            }

            (kth_distance, x_distances, y_distances)
        };

        all_knn_info.push(KNNInfo {
            kth_distance: kth_distance.max(1e-10),
            x_distances,
            y_distances,
        });
    }

    all_knn_info
}

/// Digamma function (psi) - approximation using asymptotic expansion
/// digamma(x) ≈ ln(x) - 1/(2x) - 1/(12x^2) + 1/(120x^4)
fn digamma(x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }

    let mut result = 0.0;
    let mut xx = x;
    let coef = -1.0 / 12.0;

    // Use reflection formula for small x
    if x < 1.0 {
        result += std::f64::consts::PI;
        result /= ((std::f64::consts::PI) * (1.0 - xx)).tan();
        xx += 1.0;
    }

    // Asymptotic expansion
    while xx < 8.0 {
        result -= 1.0 / xx;
        xx += 1.0;
    }

    let inv_xx = 1.0 / xx;
    let inv_xx2 = inv_xx * inv_xx;

    result += (xx - 0.5).ln() - 0.5 / xx - coef * inv_xx2;
    result
}

/// Calculate mutual information using KSG method 2 (Chebyshev distance)
#[pyfunction]
pub fn mutual_information_knn_chebyshev(
    _py: Python,
    x: PyReadonlyArray1<f64>,
    y: PyReadonlyArray1<f64>,
    k: Option<usize>,
) -> PyResult<f64> {
    let x_slice = x.as_slice()?;
    let y_slice = y.as_slice()?;

    if x_slice.len() != y_slice.len() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "x and y must have the same length",
        ));
    }

    let n = x_slice.len();
    if n == 0 {
        return Ok(0.0);
    }

    let k = k.unwrap_or(3);
    if k >= n {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "k must be less than the length of the arrays",
        ));
    }

    let ksg_method2 = calculate_mi_ksg_method2(x_slice, y_slice, k);
    Ok(ksg_method2)
}

/// Calculate mutual information using KSG method 2 (Chebyshev distance) with KD-Tree optimization
fn calculate_mi_ksg_method2(x: &[f64], y: &[f64], k: usize) -> f64 {
    let n = x.len();

    // Create 2D points from (x, y) pairs
    let mut points: Vec<Point2D> = Vec::with_capacity(n);
    for i in 0..n {
        points.push(Point2D::new(x[i], y[i]));
    }

    // Calculate k-nearest neighbors for all points using a single KD-Tree
    let psi_values = calculate_all_knn_info_chebyshev(&points, k);

    // Calculate digamma function values
    let digamma_n = digamma(n as f64);
    let digamma_k = digamma(k as f64);

    // Calculate mutual information using KSG (method 2) formula
    let mut mi_sum = 0.0;
    for i in 0..n {
        let psi = &psi_values[i];

        // X-dimension: count points where |x_i - x_j| < ε_i
        let mut count_x = 0;
        for j in 0..n {
            if i != j && (x[i] - x[j]).abs() < psi.kth_distance {
                count_x += 1;
            }
        }

        // Y-dimension: count points where |y_i - y_j| < ε_i
        let mut count_y = 0;
        for j in 0..n {
            if i != j && (y[i] - y[j]).abs() < psi.kth_distance {
                count_y += 1;
            }
        }

        let mi_i =
            digamma_k + digamma_n - digamma((count_x + 1) as f64) - digamma((count_y + 1) as f64);
        mi_sum += mi_i;
    }

    // Average over all points
    mi_sum / n as f64
}

/// Calculate k-nearest neighbor information for a point using Chebyshev distance
fn calculate_knn_info_chebyshev(points: &[Point2D], index: usize, k: usize) -> KNNInfo {
    let n = points.len();
    let target = points[index];

    // Calculate Chebyshev distances to all other points
    let mut distances: Vec<(usize, f64)> = Vec::with_capacity(n - 1);
    for i in 0..n {
        if i != index {
            let dist = target.chebyshev_distance(&points[i]);
            distances.push((i, dist));
        }
    }

    // Sort by distance
    distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));

    // Get k-th distance
    let kth_distance = distances[k - 1].1;

    let mut x_distances = Vec::with_capacity(k);
    let mut y_distances = Vec::with_capacity(k);

    for i in 0..k {
        let neighbor_idx = distances[i].0;

        // Store normalized distances
        x_distances.push((points[neighbor_idx].x - target.x).abs() / kth_distance);
        y_distances.push((points[neighbor_idx].y - target.y).abs() / kth_distance);
    }

    KNNInfo {
        kth_distance,
        x_distances,
        y_distances,
    }
}
