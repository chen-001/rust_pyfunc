use pyo3::prelude::*;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray1};
use ndarray::{Array2, ArrayView1};
use std::collections::HashMap;

/// 计算价格序列的局部相关性分析
/// 
/// 对于每个价格点，向前取x个值作为局部序列，然后分别向前和向后搜索，
/// 找到与当前局部序列相关性最大和最小的位置，并计算间隔行数和volume总和。
/// 
/// 参数：
/// - prices: 价格序列 
/// - volumes: 成交量序列
/// - window_size: 局部序列的窗口大小
/// 
/// 返回：
/// - (result_array, column_names): 二维数组和列名列表
#[pyfunction]
#[pyo3(signature = (prices, volumes, window_size))]
pub fn local_correlation(
    py: Python,
    prices: PyReadonlyArray1<f64>,
    volumes: PyReadonlyArray1<f64>,
    window_size: usize,
) -> PyResult<(Py<PyArray2<f64>>, Vec<String>)> {
    let prices = prices.as_array();
    let volumes = volumes.as_array();
    let n = prices.len();
    
    if volumes.len() != n {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "价格序列和成交量序列长度必须相同",
        ));
    }
    
    if window_size == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "窗口大小必须大于0",
        ));
    }
    
    // 使用高性能相关性引擎
    let mut engine = CorrelationEngine::new(&prices, &volumes, window_size);
    let result = engine.compute_correlations();
    
    let column_names = vec![
        "向后corr最大处间隔行数".to_string(),
        "向后corr最大处间隔volume总和".to_string(),
        "向后corr最小处间隔行数".to_string(),
        "向后corr最小处间隔volume总和".to_string(),
        "向后与corr最大处之间的corr最小处间隔行数".to_string(),
        "向后与corr最大处之间的corr最小处间隔volume总和".to_string(),
        "向前corr最大处间隔行数".to_string(),
        "向前corr最大处间隔volume总和".to_string(),
        "向前corr最小处间隔行数".to_string(),
        "向前corr最小处间隔volume总和".to_string(),
        "向前与corr最大处之间的corr最小处间隔行数".to_string(),
        "向前与corr最大处之间的corr最小处间隔volume总和".to_string(),
    ];
    
    Ok((result.into_pyarray(py).to_owned(), column_names))
}

/// 高性能相关性计算引擎
#[allow(dead_code)]
struct CorrelationEngine {
    n: usize,
    window_size: usize,
    
    // 价格和成交量数据引用
    prices: Vec<f64>,
    volumes: Vec<f64>,
    
    // 内存池 - 预分配所有需要的内存
    memory_pool: MemoryPool,
    
    // 窗口数据的紧凑存储
    window_storage: CompactWindowStorage,
    
    // 相关性缓存
    correlation_cache: CorrelationCache,
    
    // 批量处理参数
    batch_size: usize,
}

/// 预分配的内存池，避免运行时分配
#[allow(dead_code)]
struct MemoryPool {
    // 窗口数据缓冲区
    window_buffer: Vec<f64>,
    
    // 相关性计算缓冲区
    correlation_buffer: Vec<f64>,
    
    // 批量处理缓冲区
    batch_buffer: Vec<f64>,
    
    // 结果临时缓冲区
    temp_results: Vec<CorrelationStats>,
}

/// 紧凑的窗口数据存储，优化内存局部性
#[allow(dead_code)]
struct CompactWindowStorage {
    // 所有窗口数据的连续存储
    data: Vec<f64>,
    
    // 每个窗口的起始偏移
    offsets: Vec<usize>,
    
    // 窗口统计信息
    stats: Vec<Option<WindowStats>>,
    
    window_size: usize,
}

/// 相关性结果缓存
#[allow(dead_code)]
struct CorrelationCache {
    // 相关性矩阵 - 使用紧凑键
    matrix: HashMap<u64, f64>,
    
    // 缓存命中统计
    hits: usize,
    misses: usize,
}

impl CorrelationEngine {
    fn new(prices: &ArrayView1<f64>, volumes: &ArrayView1<f64>, window_size: usize) -> Self {
        let n = prices.len();
        let batch_size = 64; // 最优批处理大小
        
        // 预先转换为Vec以消除内存布局问题
        let prices_vec = prices.to_vec();
        let volumes_vec = volumes.to_vec();
        
        // 预分配内存池
        let memory_pool = MemoryPool {
            window_buffer: vec![0.0; window_size * 2],
            correlation_buffer: vec![0.0; batch_size],
            batch_buffer: vec![0.0; window_size * batch_size],
            temp_results: Vec::with_capacity(n),
        };
        
        // 初始化窗口存储
        let window_storage = CompactWindowStorage {
            data: vec![0.0; n * window_size],
            offsets: vec![0; n],
            stats: vec![None; n],
            window_size,
        };
        
        // 初始化缓存
        let correlation_cache = CorrelationCache {
            matrix: HashMap::with_capacity(n * n / 4), // 预估缓存大小
            hits: 0,
            misses: 0,
        };
        
        Self {
            n,
            window_size,
            prices: prices_vec,
            volumes: volumes_vec,
            memory_pool,
            window_storage,
            correlation_cache,
            batch_size,
        }
    }
    
    fn compute_correlations(&mut self) -> Array2<f64> {
        let mut result = Array2::<f64>::from_elem((self.n, 12), f64::NAN);
        
        // 第一步：预计算所有窗口统计信息 - 批量优化
        self.precompute_all_windows();
        
        // 第二步：批量处理相关性计算
        for batch_start in (self.window_size..self.n).step_by(self.batch_size) {
            let batch_end = (batch_start + self.batch_size).min(self.n);
            self.process_batch(batch_start, batch_end, &mut result);
        }
        
        result
    }
    
    /// 预计算所有窗口统计信息 - 使用滑动窗口优化
    fn precompute_all_windows(&mut self) {
        for i in self.window_size - 1..self.n {
            let start = i + 1 - self.window_size;
            let end = i + 1;
            
            // 直接从Vec中获取数据，避免内存布局问题
            let window_data = self.prices[start..end].to_vec();
            
            // 检查NaN值
            if window_data.iter().any(|&x| x.is_nan()) {
                continue;
            }
            
            // 计算窗口统计信息
            if let Some(stats) = self.compute_window_stats(&window_data) {
                // 存储数据到紧凑存储
                let offset = i * self.window_size;
                self.window_storage.data[offset..offset + self.window_size]
                    .copy_from_slice(&window_data);
                self.window_storage.offsets[i] = offset;
                self.window_storage.stats[i] = Some(stats);
            }
        }
    }
    
    /// 批量处理一批位置的相关性计算
    fn process_batch(&mut self, batch_start: usize, batch_end: usize, result: &mut Array2<f64>) {
        for current_idx in batch_start..batch_end {
            if self.window_storage.stats[current_idx].is_none() {
                continue;
            }
            
            // 向后分析
            if let Some(forward_stats) = self.analyze_forward_cached(current_idx) {
                result[[current_idx, 0]] = forward_stats.max_corr_gap as f64;
                result[[current_idx, 1]] = forward_stats.max_corr_volume;
                result[[current_idx, 2]] = forward_stats.min_corr_gap as f64;
                result[[current_idx, 3]] = forward_stats.min_corr_volume;
                result[[current_idx, 4]] = forward_stats.between_min_gap as f64;
                result[[current_idx, 5]] = forward_stats.between_min_volume;
            }
            
            // 向前分析
            if let Some(backward_stats) = self.analyze_backward_cached(current_idx) {
                result[[current_idx, 6]] = backward_stats.max_corr_gap as f64;
                result[[current_idx, 7]] = backward_stats.max_corr_volume;
                result[[current_idx, 8]] = backward_stats.min_corr_gap as f64;
                result[[current_idx, 9]] = backward_stats.min_corr_volume;
                result[[current_idx, 10]] = backward_stats.between_min_gap as f64;
                result[[current_idx, 11]] = backward_stats.between_min_volume;
            }
        }
    }
    
    /// 计算窗口统计信息 - 高度优化版本
    fn compute_window_stats(&self, data: &[f64]) -> Option<WindowStats> {
        if data.is_empty() {
            return None;
        }
        
        let n = data.len() as f64;
        let sum_x = data.iter().sum::<f64>();
        let mean = sum_x / n;
        
        // 单次遍历计算方差和平方和
        let mut sum_x_squared = 0.0;
        let mut variance = 0.0;
        
        // 循环展开优化
        let chunks = data.len() / 4;
        let remainder = data.len() % 4;
        
        for chunk in 0..chunks {
            let base = chunk * 4;
            let x1 = data[base];
            let x2 = data[base + 1];
            let x3 = data[base + 2];
            let x4 = data[base + 3];
            
            sum_x_squared += x1 * x1 + x2 * x2 + x3 * x3 + x4 * x4;
            
            let d1 = x1 - mean;
            let d2 = x2 - mean;
            let d3 = x3 - mean;
            let d4 = x4 - mean;
            variance += d1 * d1 + d2 * d2 + d3 * d3 + d4 * d4;
        }
        
        // 处理剩余元素
        let start_remainder = chunks * 4;
        for i in start_remainder..start_remainder + remainder {
            let x = data[i];
            sum_x_squared += x * x;
            let diff = x - mean;
            variance += diff * diff;
        }
        
        let std_dev = (variance / n).sqrt();
        
        if std_dev <= f64::EPSILON {
            return None;
        }
        
        Some(WindowStats {
            mean,
            std_dev,
            sum_x,
            sum_x_squared,
            valid: true,
        })
    }
    
    /// 使用缓存的向后分析 - 精准且高性能版本
    fn analyze_forward_cached(&mut self, current_idx: usize) -> Option<CorrelationStats> {
        let current_stats = self.window_storage.stats[current_idx].as_ref()?;
        let current_offset = self.window_storage.offsets[current_idx];
        let current_data = &self.window_storage.data[current_offset..current_offset + self.window_size];
        
        let mut max_corr = f64::NEG_INFINITY;
        let mut min_corr = f64::INFINITY;
        let mut max_corr_idx = None;
        let mut min_corr_idx = None;
        
        let search_start = current_idx + self.window_size;
        if search_start >= self.n {
            return None;
        }
        
        // 精准搜索：遍历所有位置，但使用零分配优化
        for j in search_start..self.n {
            if let Some(ref compare_stats) = self.window_storage.stats[j] {
                let compare_offset = self.window_storage.offsets[j];
                let compare_data = &self.window_storage.data[compare_offset..compare_offset + self.window_size];
                
                if let Some(corr) = self.ultra_fast_correlation_no_alloc(
                    current_stats, compare_stats, current_data, compare_data
                ) {
                    if corr > max_corr {
                        max_corr = corr;
                        max_corr_idx = Some(j);
                    }
                    if corr < min_corr {
                        min_corr = corr;
                        min_corr_idx = Some(j);
                    }
                }
            }
        }
        
        if max_corr_idx.is_none() || min_corr_idx.is_none() {
            return None;
        }
        
        let max_idx = max_corr_idx.unwrap();
        let min_idx = min_corr_idx.unwrap();
        
        // 计算统计信息
        let max_corr_gap = max_idx - current_idx;
        let max_corr_volume: f64 = self.volumes[current_idx+1..=max_idx].iter().sum();
        
        let min_corr_gap = min_idx - current_idx;
        let min_corr_volume: f64 = self.volumes[current_idx+1..=min_idx].iter().sum();
        
        // 精准区间最小值查找
        let (between_min_gap, between_min_volume) = 
            self.find_min_between_forward_precise(current_idx, max_idx, current_stats, current_data);
        
        Some(CorrelationStats {
            max_corr_gap,
            max_corr_volume,
            min_corr_gap,
            min_corr_volume,
            between_min_gap,
            between_min_volume,
        })
    }
    
    /// 使用缓存的向前分析 - 精准且高性能版本
    fn analyze_backward_cached(&mut self, current_idx: usize) -> Option<CorrelationStats> {
        let current_stats = self.window_storage.stats[current_idx].as_ref()?;
        let current_offset = self.window_storage.offsets[current_idx];
        let current_data = &self.window_storage.data[current_offset..current_offset + self.window_size];
        
        let mut max_corr = f64::NEG_INFINITY;
        let mut min_corr = f64::INFINITY;
        let mut max_corr_idx = None;
        let mut min_corr_idx = None;
        
        if current_idx < self.window_size {
            return None;
        }
        
        let search_end = current_idx - self.window_size;
        
        // 精准搜索：遍历所有位置，但使用零分配优化
        for j in (0..=search_end).rev() {
            if let Some(ref compare_stats) = self.window_storage.stats[j] {
                let compare_offset = self.window_storage.offsets[j];
                let compare_data = &self.window_storage.data[compare_offset..compare_offset + self.window_size];
                
                if let Some(corr) = self.ultra_fast_correlation_no_alloc(
                    current_stats, compare_stats, current_data, compare_data
                ) {
                    if corr > max_corr {
                        max_corr = corr;
                        max_corr_idx = Some(j);
                    }
                    if corr < min_corr {
                        min_corr = corr;
                        min_corr_idx = Some(j);
                    }
                }
            }
        }
        
        if max_corr_idx.is_none() || min_corr_idx.is_none() {
            return None;
        }
        
        let max_idx = max_corr_idx.unwrap();
        let min_idx = min_corr_idx.unwrap();
        
        let max_corr_gap = current_idx - max_idx;
        let max_corr_volume = self.volumes[max_idx..current_idx].iter().sum();
        
        let min_corr_gap = current_idx - min_idx;
        let min_corr_volume = self.volumes[min_idx..current_idx].iter().sum();
        
        // 精准区间最小值查找
        let (between_min_gap, between_min_volume) = 
            self.find_min_between_backward_precise(max_idx, current_idx, current_stats, current_data);
        
        Some(CorrelationStats {
            max_corr_gap,
            max_corr_volume,
            min_corr_gap,
            min_corr_volume,
            between_min_gap,
            between_min_volume,
        })
    }
    
    /// 从缓存存储中获取窗口数据
    #[allow(dead_code)]
    fn get_window_data_cached(&self, idx: usize) -> Option<Vec<f64>> {
        if idx >= self.n || self.window_storage.stats[idx].is_none() {
            return None;
        }
        
        let offset = self.window_storage.offsets[idx];
        let data = self.window_storage.data[offset..offset + self.window_size].to_vec();
        Some(data)
    }
    
    /// 生成缓存键 - 使用位操作优化
    #[inline]
    #[allow(dead_code)]
    fn generate_cache_key(&self, idx1: usize, idx2: usize) -> u64 {
        // 使用位操作创建唯一键，假设索引不超过32位
        ((idx1 as u64) << 32) | (idx2 as u64)
    }
    
    /// 零分配的超优化相关性计算 - 直接使用slice
    #[inline]
    fn ultra_fast_correlation_no_alloc(
        &self,
        w1: &WindowStats, 
        w2: &WindowStats, 
        data1: &[f64], 
        data2: &[f64]
    ) -> Option<f64> {
        if data1.len() != data2.len() || w1.std_dev <= f64::EPSILON || w2.std_dev <= f64::EPSILON {
            return None;
        }
        
        let n = data1.len();
        let mut sum_xy = 0.0;
        
        // 超级优化的内积计算 - 手动循环展开到8个元素，直接访问slice
        let chunks = n / 8;
        let remainder = n % 8;
        
        for chunk in 0..chunks {
            let base = chunk * 8;
            unsafe {
                // 使用unsafe获得最高性能 - 边界检查已在外部完成
                sum_xy += data1.get_unchecked(base) * data2.get_unchecked(base)
                        + data1.get_unchecked(base + 1) * data2.get_unchecked(base + 1)
                        + data1.get_unchecked(base + 2) * data2.get_unchecked(base + 2)
                        + data1.get_unchecked(base + 3) * data2.get_unchecked(base + 3)
                        + data1.get_unchecked(base + 4) * data2.get_unchecked(base + 4)
                        + data1.get_unchecked(base + 5) * data2.get_unchecked(base + 5)
                        + data1.get_unchecked(base + 6) * data2.get_unchecked(base + 6)
                        + data1.get_unchecked(base + 7) * data2.get_unchecked(base + 7);
            }
        }
        
        // 处理剩余元素
        let start_remainder = chunks * 8;
        for i in start_remainder..start_remainder + remainder {
            sum_xy += data1[i] * data2[i];
        }
        
        let n_f64 = n as f64;
        let numerator = n_f64 * sum_xy - w1.sum_x * w2.sum_x;
        let denominator = n_f64 * w1.std_dev * w2.std_dev;
        
        Some(numerator / denominator)
    }
    
    /// 精准的向后区间最小值查找 - 零分配版本
    fn find_min_between_forward_precise(
        &self,
        start_idx: usize,
        end_idx: usize,
        reference_stats: &WindowStats,
        reference_data: &[f64],
    ) -> (usize, f64) {
        if end_idx - start_idx <= self.window_size {
            return (0, f64::NAN);
        }
        
        let search_start = start_idx + self.window_size;
        let search_end = end_idx - self.window_size;
        
        if search_start > search_end {
            return (0, f64::NAN);
        }
        
        let mut min_corr = f64::INFINITY;
        let mut min_idx = None;
        
        for j in search_start..=search_end {
            if let Some(ref compare_stats) = self.window_storage.stats[j] {
                let compare_offset = self.window_storage.offsets[j];
                let compare_data = &self.window_storage.data[compare_offset..compare_offset + self.window_size];
                
                if let Some(corr) = self.ultra_fast_correlation_no_alloc(
                    reference_stats, compare_stats, reference_data, compare_data
                ) {
                    if corr < min_corr {
                        min_corr = corr;
                        min_idx = Some(j);
                    }
                }
            }
        }
        
        if let Some(idx) = min_idx {
            let gap = idx - start_idx;
            let volume = self.volumes[start_idx+1..=idx].iter().sum();
            (gap, volume)
        } else {
            (0, f64::NAN)
        }
    }
    
    /// 精准的向前区间最小值查找 - 零分配版本
    fn find_min_between_backward_precise(
        &self,
        start_idx: usize,
        end_idx: usize,
        reference_stats: &WindowStats,
        reference_data: &[f64],
    ) -> (usize, f64) {
        if end_idx - start_idx <= self.window_size {
            return (0, f64::NAN);
        }
        
        let search_start = start_idx + self.window_size;
        let search_end = end_idx - self.window_size;
        
        if search_start > search_end {
            return (0, f64::NAN);
        }
        
        let mut min_corr = f64::INFINITY;
        let mut min_idx = None;
        
        for j in search_start..=search_end {
            if let Some(ref compare_stats) = self.window_storage.stats[j] {
                let compare_offset = self.window_storage.offsets[j];
                let compare_data = &self.window_storage.data[compare_offset..compare_offset + self.window_size];
                
                if let Some(corr) = self.ultra_fast_correlation_no_alloc(
                    reference_stats, compare_stats, reference_data, compare_data
                ) {
                    if corr < min_corr {
                        min_corr = corr;
                        min_idx = Some(j);
                    }
                }
            }
        }
        
        if let Some(idx) = min_idx {
            let gap = end_idx - idx;
            let volume = self.volumes[idx..end_idx].iter().sum();
            (gap, volume)
        } else {
            (0, f64::NAN)
        }
    }
}

#[derive(Debug, Clone)]
struct CorrelationStats {
    max_corr_gap: usize,
    max_corr_volume: f64,
    min_corr_gap: usize,
    min_corr_volume: f64,
    between_min_gap: usize,
    between_min_volume: f64,
}

/// 优化的窗口统计信息
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct WindowStats {
    mean: f64,
    std_dev: f64,
    sum_x: f64,
    sum_x_squared: f64,
    valid: bool,
}







