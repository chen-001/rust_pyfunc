use numpy::{IntoPyArray, PyReadonlyArray2};
use pyo3::prelude::*;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};

/// A simplified 2D point for row-level calculations
#[derive(Clone, Copy, Debug)]
struct Point2D {
    x: f64,
    y: f64,
}

impl Point2D {
    fn new(x: f64, y: f64) -> Self {
        Point2D { x, y }
    }

    /// Calculate Chebyshev distance (max norm) to another point
    fn chebyshev_distance(&self, other: &Point2D) -> f64 {
        let dx = (self.x - other.x).abs();
        let dy = (self.y - other.y).abs();
        dx.max(dy)
    }
}

/// A value with its original index for sorted arrays
#[derive(Clone, Copy, Debug)]
struct SortedValue {
    value: f64,
}

impl SortedValue {
    fn new(value: f64) -> Self {
        SortedValue { value }
    }
}

impl PartialEq<f64> for SortedValue {
    fn eq(&self, other: &f64) -> bool {
        self.value == *other
    }
}

impl PartialOrd<f64> for SortedValue {
    fn partial_cmp(&self, other: &f64) -> Option<Ordering> {
        self.value.partial_cmp(other)
    }
}

/// KD-Tree node for 2D points
#[derive(Debug)]
struct KDNode {
    point: Point2D,
    index: usize,
    left: Option<Box<KDNode>>,
    right: Option<Box<KDNode>>,
    axis: usize,
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

/// KD-Tree implementation
#[derive(Debug)]
struct KDTree {
    root: Option<Box<KDNode>>,
}

impl KDTree {
    fn build(points: &[Point2D]) -> Self {
        let mut indices: Vec<usize> = (0..points.len()).collect();
        let root = if !indices.is_empty() {
            Some(Self::build_recursive(points, &mut indices[..], 0))
        } else {
            None
        };

        KDTree { root }
    }

    fn build_recursive(points: &[Point2D], indices: &mut [usize], depth: usize) -> Box<KDNode> {
        if indices.is_empty() {
            panic!("Cannot build KD-Tree from empty index list");
        }

        let axis = depth % 2;
        let mid = indices.len() / 2;
        indices.select_nth_unstable_by(mid, |&a, &b| {
            let coord_a = if axis == 0 { points[a].x } else { points[a].y };
            let coord_b = if axis == 0 { points[b].x } else { points[b].y };
            coord_a.partial_cmp(&coord_b).unwrap_or(Ordering::Equal)
        });

        let (left_indices, right_with_median) = indices.split_at_mut(mid);
        let (median_index, right_indices) = {
            let (median_slot, rest) = right_with_median
                .split_first_mut()
                .expect("median should exist");
            (*median_slot, rest)
        };
        let median_point = points[median_index];

        let mut node = Box::new(KDNode::new(median_point, median_index, axis));

        if !left_indices.is_empty() {
            node.left = Some(Self::build_recursive(points, left_indices, depth + 1));
        }

        if !right_indices.is_empty() {
            node.right = Some(Self::build_recursive(points, right_indices, depth + 1));
        }

        node
    }

    fn k_nearest_chebyshev(&self, target: &Point2D, k: usize) -> Vec<(usize, f64)> {
        if k == 0 {
            return Vec::new();
        }

        let mut results = BinaryHeap::with_capacity(k);
        if let Some(ref root) = self.root {
            Self::search_recursive_chebyshev(root, target, k, &mut results);
        }

        let mut neighbors: Vec<(usize, f64)> = results
            .into_iter()
            .map(|neighbor| (neighbor.index, neighbor.distance))
            .collect();
        neighbors.sort_by(|a, b| compare_distance(a.1, b.1));
        neighbors
    }

    fn search_recursive_chebyshev(
        node: &KDNode,
        target: &Point2D,
        k: usize,
        results: &mut BinaryHeap<Neighbor>,
    ) {
        let dist = node.point.chebyshev_distance(target);
        if dist > 0.0 {
            Self::push_neighbor(results, Neighbor::new(node.index, dist), k);
        }

        let axis = node.axis;
        let target_coord = if axis == 0 { target.x } else { target.y };
        let node_coord = if axis == 0 {
            node.point.x
        } else {
            node.point.y
        };

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

        if let Some(ref child) = near_subtree {
            Self::search_recursive_chebyshev(child, target, k, results);
        }

        // Find current maximum distance
        let current_max_dist = if results.len() < k {
            f64::INFINITY
        } else {
            results
                .peek()
                .map(|neighbor| {
                    if neighbor.distance.is_nan() {
                        f64::INFINITY
                    } else {
                        neighbor.distance
                    }
                })
                .unwrap_or(f64::INFINITY)
        };

        let diff = (target_coord - node_coord).abs();
        if results.len() < k || diff < current_max_dist {
            if let Some(ref child) = far_subtree {
                Self::search_recursive_chebyshev(child, target, k, results);
            }
        }
    }

    fn push_neighbor(results: &mut BinaryHeap<Neighbor>, neighbor: Neighbor, k: usize) {
        if neighbor.distance.is_nan() {
            return;
        }

        if results.len() < k {
            results.push(neighbor);
            return;
        }

        if let Some(current_max) = results.peek() {
            if compare_distance(neighbor.distance, current_max.distance) == Ordering::Less {
                results.pop();
                results.push(neighbor);
            }
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct Neighbor {
    index: usize,
    distance: f64,
}

impl Neighbor {
    fn new(index: usize, distance: f64) -> Self {
        Neighbor { index, distance }
    }
}

fn compare_distance(a: f64, b: f64) -> Ordering {
    match (a.is_nan(), b.is_nan()) {
        (true, true) => Ordering::Equal,
        (true, false) => Ordering::Greater,
        (false, true) => Ordering::Less,
        (false, false) => a.partial_cmp(&b).unwrap_or(Ordering::Equal),
    }
}

impl PartialEq for Neighbor {
    fn eq(&self, other: &Self) -> bool {
        compare_distance(self.distance, other.distance) == Ordering::Equal
    }
}

impl Eq for Neighbor {}

impl PartialOrd for Neighbor {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(compare_distance(self.distance, other.distance))
    }
}

impl Ord for Neighbor {
    fn cmp(&self, other: &Self) -> Ordering {
        compare_distance(self.distance, other.distance)
    }
}

const TIE_EPSILON: f64 = 1e-12;

/// Cached digamma values reuse across row computations
struct DigammaCache {
    table: Vec<f64>,
    digamma_n: f64,
    digamma_k: f64,
}

impl DigammaCache {
    fn new(sample_count: usize, k: usize) -> Self {
        let mut table = vec![0.0_f64; sample_count + 2];
        for i in 0..=sample_count + 1 {
            table[i] = digamma(i as f64);
        }

        DigammaCache {
            table,
            digamma_n: digamma(sample_count as f64),
            digamma_k: digamma(k as f64),
        }
    }

    #[inline]
    fn digamma_n(&self) -> f64 {
        self.digamma_n
    }

    #[inline]
    fn digamma_k(&self) -> f64 {
        self.digamma_k
    }

    #[inline]
    fn marginal(&self, count: usize) -> f64 {
        let idx = (count + 1).min(self.table.len().saturating_sub(1));
        self.table[idx]
    }
}

struct DigammaCachePool {
    caches: HashMap<usize, DigammaCache>,
    k: usize,
}

impl DigammaCachePool {
    fn new(k: usize) -> Self {
        DigammaCachePool {
            caches: HashMap::new(),
            k,
        }
    }

    fn get(&mut self, sample_count: usize) -> &DigammaCache {
        self.caches
            .entry(sample_count)
            .or_insert_with(|| DigammaCache::new(sample_count, self.k))
    }
}

/// Binary search to find the leftmost index with value >= target
fn binary_search_left<T>(sorted_array: &[T], target: f64) -> usize
where
    T: PartialEq<f64> + PartialOrd<f64> + Copy,
{
    let mut left = 0;
    let mut right = sorted_array.len();

    while left < right {
        let mid = (left + right) / 2;
        if sorted_array[mid] < target {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    left
}

/// Binary search to find the rightmost index with value <= target
fn binary_search_right<T>(sorted_array: &[T], target: f64) -> usize
where
    T: PartialEq<f64> + PartialOrd<f64> + Copy,
{
    let mut left = 0;
    let mut right = sorted_array.len();

    while left < right {
        let mid = (left + right) / 2;
        if sorted_array[mid] <= target {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    left
}

/// Count points within range [min_val, max_val] for a specific original index using sorted array
fn count_in_range_1d(
    sorted_array: &[SortedValue],
    values: &[f64],
    original_index: usize,
    epsilon: f64,
) -> usize {
    let target_value = values[original_index];
    let min_val = target_value - epsilon;
    let max_val = target_value + epsilon;

    let left_idx = binary_search_left(sorted_array, min_val);
    let right_idx = binary_search_right(sorted_array, max_val);

    // right_idx is exclusive, so count is right_idx - left_idx
    right_idx - left_idx
}

/// Calculate mutual information for a single row pair (a_row, b_row) using KSG method
fn calculate_row_mi_ksg_fixed(
    a_row: ndarray::ArrayView1<f64>,
    b_row: ndarray::ArrayView1<f64>,
    k: usize,
    cache_pool: &mut DigammaCachePool,
) -> f64 {
    let n = a_row.len();

    if n != b_row.len() {
        return f64::NAN;
    }

    if n == 0 || k == 0 {
        return f64::NAN;
    }

    // Filter out entries containing non-finite values
    let mut points: Vec<Point2D> = Vec::with_capacity(n);
    let mut a_values: Vec<f64> = Vec::with_capacity(n);
    let mut b_values: Vec<f64> = Vec::with_capacity(n);
    for i in 0..n {
        let x = a_row[i];
        let y = b_row[i];
        if x.is_finite() && y.is_finite() {
            points.push(Point2D::new(x, y));
            a_values.push(x);
            b_values.push(y);
        }
    }

    let valid_n = points.len();
    if valid_n == 0 || valid_n <= k {
        return f64::NAN;
    }

    let digamma_cache = cache_pool.get(valid_n);

    // Build KD-Tree once for all points
    let kdtree = KDTree::build(&points);

    // Calculate k-nearest neighbors for all points using Chebyshev distance for consistent ε_i
    let mut epsilon_values = Vec::with_capacity(valid_n);

    for idx in 0..valid_n {
        let target = points[idx];
        let neighbors = kdtree.k_nearest_chebyshev(&target, k);

        let kth_distance = neighbors
            .get(k.saturating_sub(1))
            .map(|(_, dist)| *dist)
            .or_else(|| neighbors.last().map(|(_, dist)| *dist))
            .unwrap_or(0.0);

        // 使用略小于第k邻居的距离来避免包含边界点
        let epsilon = (kth_distance - TIE_EPSILON).max(1e-10);
        epsilon_values.push(epsilon);
    }

    // Pre-sort values for binary search optimization
    let mut a_sorted: Vec<SortedValue> = (0..valid_n)
        .map(|i| SortedValue::new(a_values[i]))
        .collect();
    a_sorted.sort_by(|a, b| a.value.partial_cmp(&b.value).unwrap_or(Ordering::Equal));

    let mut b_sorted: Vec<SortedValue> = (0..valid_n)
        .map(|i| SortedValue::new(b_values[i]))
        .collect();
    b_sorted.sort_by(|a, b| a.value.partial_cmp(&b.value).unwrap_or(Ordering::Equal));

    // Calculate mutual information using corrected KSG formula
    let mut mi_sum = 0.0;
    for i in 0..valid_n {
        let epsilon = epsilon_values[i];

        // 在X维度中计数：|x_i - x_j| < ε
        let count_x = count_in_range_1d(&a_sorted, &a_values, i, epsilon);
        let neighbors_x = count_x.saturating_sub(1);

        // 在Y维度中计数：|y_i - y_j| < ε
        let count_y = count_in_range_1d(&b_sorted, &b_values, i, epsilon);
        let neighbors_y = count_y.saturating_sub(1);

        // KSG互信息估计：I(X;Y) = ψ(k) + ψ(N) - <ψ(nx_i) + ψ(ny_i)>
        let mi_i = digamma_cache.digamma_k() + digamma_cache.digamma_n()
            - digamma_cache.marginal(neighbors_x)
            - digamma_cache.marginal(neighbors_y);

        mi_sum += mi_i;
    }

    // Average over all points
    mi_sum / valid_n as f64
}

/// Calculate mutual information for corresponding rows of two 2D arrays
/// using corrected KSG (Kraskov-Stögbauer-Grassberger) method.
/// This fixes the negative values issue by using proper 1D counting.
#[pyfunction]
pub fn mutual_information_2d_knn_fixed(
    _py: Python,
    a: PyReadonlyArray2<f64>,
    b: PyReadonlyArray2<f64>,
    k: Option<usize>,
) -> PyResult<PyObject> {
    let a_array = a.as_array();
    let b_array = b.as_array();

    let (n_rows_a, _n_cols_a) = a_array.dim();
    let (n_rows_b, _n_cols_b) = b_array.dim();

    if n_rows_a != n_rows_b {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "a and b must have the same number of rows",
        ));
    }

    if n_rows_a == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "arrays must not be empty",
        ));
    }

    let n = n_rows_a;
    let k = k.unwrap_or(3);

    if k == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "k must be positive",
        ));
    }

    // Prepare digamma cache pool reused across rows with varying valid sample counts
    let mut cache_pool = DigammaCachePool::new(k);

    // Calculate mutual information for each row
    let mut mi_results: Vec<f64> = Vec::with_capacity(n);

    for i in 0..n {
        // Get the i-th row from both arrays
        let a_row = a_array.row(i);
        let b_row = b_array.row(i);

        // Calculate mutual information for this specific row pair
        let mi = calculate_row_mi_ksg_fixed(a_row, b_row, k, &mut cache_pool);
        mi_results.push(mi);
    }

    Ok(pyo3::Python::with_gil(|py| {
        mi_results.into_pyarray(py).into()
    }))
}

/// Digamma function (psi) - approximation using asymptotic expansion
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
