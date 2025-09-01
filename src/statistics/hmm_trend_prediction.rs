use pyo3::prelude::*;
use ndarray::{Array1, Array2, ArrayView1, s};
use std::f64::consts::E;

/// 线性回归结果结构体
#[derive(Debug, Clone)]
struct LinearRegressionResult {
    slope: f64,
    intercept: f64,
    r_squared: f64,
}

/// HMM状态：-1(下跌), 0(震荡), 1(上涨)
const STATE_DOWN: i32 = -1;
const STATE_SIDEWAYS: i32 = 0;
const STATE_UP: i32 = 1;

/// HMM预测结果
#[pyclass]
#[derive(Debug, Clone)]
pub struct HMMPredictionResult {
    #[pyo3(get)]
    pub state_predictions: Vec<Vec<f64>>,    // 每步的状态预测概率 [下跌, 震荡, 上涨]
    #[pyo3(get)]
    pub updated_state_probs: Vec<Vec<f64>>,  // 每步更新后的状态概率
    #[pyo3(get)]
    pub emission_probs: Vec<Vec<Vec<f64>>>,  // 每步的发射概率矩阵 [状态][观测]
    #[pyo3(get)]
    pub transition_probs: Vec<Vec<Vec<f64>>>, // 每步的状态转移概率矩阵
}

#[pymethods]
impl HMMPredictionResult {
    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("HMMPredictionResult(steps={})", self.state_predictions.len()))
    }
}

/// 执行线性回归分析
fn linear_regression(x: &[f64], y: &[f64]) -> LinearRegressionResult {
    let n = x.len() as f64;
    
    if n < 2.0 {
        return LinearRegressionResult {
            slope: 0.0,
            intercept: 0.0,
            r_squared: 0.0,
        };
    }
    
    let sum_x: f64 = x.iter().sum();
    let sum_y: f64 = y.iter().sum();
    let sum_xy: f64 = x.iter().zip(y.iter()).map(|(xi, yi)| xi * yi).sum();
    let sum_x2: f64 = x.iter().map(|xi| xi * xi).sum();
    let sum_y2: f64 = y.iter().map(|yi| yi * yi).sum();
    
    let mean_x = sum_x / n;
    let mean_y = sum_y / n;
    
    // 计算斜率和截距
    let slope = (sum_xy - n * mean_x * mean_y) / (sum_x2 - n * mean_x * mean_x);
    let intercept = mean_y - slope * mean_x;
    
    // 计算R²
    let ss_res: f64 = x.iter().zip(y.iter())
        .map(|(xi, yi)| {
            let predicted = slope * xi + intercept;
            (yi - predicted).powi(2)
        }).sum();
    
    let ss_tot: f64 = y.iter().map(|yi| (yi - mean_y).powi(2)).sum();
    
    let r_squared = if ss_tot == 0.0 { 0.0 } else { 1.0 - ss_res / ss_tot };
    
    LinearRegressionResult {
        slope,
        intercept,
        r_squared,
    }
}

/// 基于滚动线性回归的趋势判断
fn linear_regression_trend_analysis(
    prices: &[f64],
    window: usize,
    slope_threshold: f64,
    r2_threshold: f64,
) -> Vec<i32> {
    let n = prices.len();
    let mut states = vec![STATE_SIDEWAYS; n];
    
    // 对前window个数据点进行预热分析
    // 使用较小的窗口进行初始状态分析
    for i in 3..window.min(n) {
        let start_idx = (i as f64 * 0.5) as usize; // 使用前一半的数据作为窗口
        let window_prices = &prices[start_idx..i];
        
        if window_prices.len() < 3 {
            continue;
        }
        
        // 对价格取对数
        let log_prices: Vec<f64> = window_prices.iter().map(|p| p.ln()).collect();
        let x: Vec<f64> = (0..window_prices.len()).map(|j| j as f64).collect();
        
        // 执行线性回归
        let regression = linear_regression(&x, &log_prices);
        
        // 根据斜率和拟合优度判断状态，使用较宽松的阈值
        states[i] = if regression.r_squared < r2_threshold * 0.7 {
            STATE_SIDEWAYS  // 拟合度低，震荡
        } else if regression.slope > slope_threshold * 0.8 {
            STATE_UP        // 明确上涨
        } else if regression.slope < -slope_threshold * 0.8 {
            STATE_DOWN      // 明确下跌
        } else {
            STATE_SIDEWAYS  // 趋势不明确，震荡
        };
    }
    
    // 对剩余数据进行正常分析
    for i in window..n {
        // 取窗口内的数据
        let window_prices = &prices[i - window..i];
        
        // 对价格取对数
        let log_prices: Vec<f64> = window_prices.iter().map(|p| p.ln()).collect();
        let x: Vec<f64> = (0..window).map(|j| j as f64).collect();
        
        // 执行线性回归
        let regression = linear_regression(&x, &log_prices);
        
        // 根据斜率和拟合优度判断状态
        states[i] = if regression.r_squared < r2_threshold {
            STATE_SIDEWAYS  // 拟合度低，震荡
        } else if regression.slope > slope_threshold {
            STATE_UP        // 明确上涨
        } else if regression.slope < -slope_threshold {
            STATE_DOWN      // 明确下跌
        } else {
            STATE_SIDEWAYS  // 趋势不明确，震荡
        };
    }
    
    states
}

/// 应用状态平衡机制到转移矩阵
fn balance_transition_matrix(mut transition_matrix: Array2<f64>) -> Array2<f64> {
    let min_transition_prob = 0.1; // 最小转移概率
    
    for i in 0..3 {
        for j in 0..3 {
            if transition_matrix[[i, j]] < min_transition_prob {
                transition_matrix[[i, j]] = min_transition_prob;
            }
        }
        
        // 重新归一化
        let row_sum: f64 = transition_matrix.row(i).sum();
        if row_sum > 0.0 {
            for j in 0..3 {
                transition_matrix[[i, j]] /= row_sum;
            }
        }
    }
    
    transition_matrix
}

/// 初始化状态转移概率矩阵
fn initialize_transition_matrix(states: &[i32]) -> Array2<f64> {
    let mut transition_counts = Array2::<f64>::zeros((3, 3));
    
    // 统计状态转移次数
    for i in 1..states.len() {
        let prev_state_idx = state_to_index(states[i - 1]);
        let curr_state_idx = state_to_index(states[i]);
        transition_counts[[prev_state_idx, curr_state_idx]] += 1.0;
    }
    
    // 转换为概率矩阵（添加更强的拉普拉斯平滑）
    let mut transition_probs = Array2::<f64>::zeros((3, 3));
    for i in 0..3 {
        let row_sum: f64 = transition_counts.row(i).sum() + 9.0; // 更强的平滑
        for j in 0..3 {
            transition_probs[[i, j]] = (transition_counts[[i, j]] + 3.0) / row_sum;
        }
    }
    
    // 应用状态平衡机制
    balance_transition_matrix(transition_probs)
}

/// 初始化发射概率矩阵
fn initialize_emission_matrix(states: &[i32], price_changes: &[f64]) -> Array2<f64> {
    let mut emission_counts = Array2::<f64>::zeros((3, 3)); // [状态][观测（-1,0,1）]
    
    // 统计发射次数
    for i in 0..states.len().min(price_changes.len()) {
        let state_idx = state_to_index(states[i]);
        let obs_idx = if price_changes[i] > 0.001 { 2 } // 上涨
                     else if price_changes[i] < -0.001 { 0 } // 下跌
                     else { 1 }; // 震荡
        emission_counts[[state_idx, obs_idx]] += 1.0;
    }
    
    // 转换为概率矩阵（添加拉普拉斯平滑）
    let mut emission_probs = Array2::<f64>::zeros((3, 3));
    for i in 0..3 {
        let row_sum: f64 = emission_counts.row(i).sum() + 3.0; // 拉普拉斯平滑
        for j in 0..3 {
            emission_probs[[i, j]] = (emission_counts[[i, j]] + 1.0) / row_sum;
        }
    }
    
    emission_probs
}

/// 状态值转换为数组索引
fn state_to_index(state: i32) -> usize {
    match state {
        STATE_DOWN => 0,
        STATE_SIDEWAYS => 1,
        STATE_UP => 2,
        _ => 1, // 默认为震荡
    }
}

/// 索引转换为状态值
fn index_to_state(index: usize) -> i32 {
    match index {
        0 => STATE_DOWN,
        1 => STATE_SIDEWAYS,
        2 => STATE_UP,
        _ => STATE_SIDEWAYS,
    }
}

/// 计算观测索引
fn observation_to_index(price_change: f64) -> usize {
    if price_change > 0.001 { 2 }      // 上涨
    else if price_change < -0.001 { 0 } // 下跌
    else { 1 }                          // 震荡
}

/// 执行HMM前向算法预测
fn hmm_forward_prediction(
    transition_matrix: &Array2<f64>,
    emission_matrix: &Array2<f64>,
    current_state_probs: &Array1<f64>,
) -> Array1<f64> {
    // 计算下一时刻的状态概率预测
    let mut next_probs = Array1::<f64>::zeros(3);
    
    for next_state in 0..3 {
        for curr_state in 0..3 {
            next_probs[next_state] += current_state_probs[curr_state] * 
                                     transition_matrix[[curr_state, next_state]];
        }
    }
    
    next_probs
}

/// 使用观测值更新状态概率（贝叶斯更新）
fn update_state_probabilities(
    predicted_probs: &Array1<f64>,
    emission_matrix: &Array2<f64>,
    observation: f64,
) -> Array1<f64> {
    let obs_idx = observation_to_index(observation);
    let mut updated_probs = Array1::<f64>::zeros(3);
    
    // 计算似然
    let mut total_likelihood = 0.0;
    for state in 0..3 {
        let likelihood = predicted_probs[state] * emission_matrix[[state, obs_idx]];
        updated_probs[state] = likelihood;
        total_likelihood += likelihood;
    }
    
    // 归一化
    if total_likelihood > 0.0 {
        for state in 0..3 {
            updated_probs[state] /= total_likelihood;
        }
    } else {
        // 如果似然为0，使用均匀分布
        updated_probs.fill(1.0 / 3.0);
    }
    
    updated_probs
}

/// 更新转移概率矩阵（在线学习）- 修复收敛问题
fn update_transition_matrix(
    mut transition_matrix: Array2<f64>,
    prev_state_probs: &Array1<f64>,
    curr_state_probs: &Array1<f64>,
    learning_rate: f64,
) -> Array2<f64> {
    // 使用更保守的学习策略，防止过度收敛到单一状态
    let conservative_rate = learning_rate * 0.1; // 大幅降低学习率
    
    // 计算真实的状态转移
    // 找到最可能的前一状态和当前状态
    let prev_max_idx = prev_state_probs.iter().enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
    let curr_max_idx = curr_state_probs.iter().enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
    
    // 只对概率较高的转移进行轻微更新
    if prev_state_probs[prev_max_idx] > 0.6 && curr_state_probs[curr_max_idx] > 0.6 {
        // 增加观测到的转移概率
        transition_matrix[[prev_max_idx, curr_max_idx]] += conservative_rate;
        
        // 对同一行的其他状态稍微减少概率
        for j in 0..3 {
            if j != curr_max_idx {
                transition_matrix[[prev_max_idx, j]] *= (1.0 - conservative_rate * 0.5);
            }
        }
    }
    
    // 应用状态平衡机制，防止任何转移概率过小
    let min_prob = 0.15;
    for i in 0..3 {
        for j in 0..3 {
            if transition_matrix[[i, j]] < min_prob {
                transition_matrix[[i, j]] = min_prob;
            }
        }
    }
    
    // 重新归一化每行
    for i in 0..3 {
        let row_sum: f64 = transition_matrix.row(i).sum();
        if row_sum > 1e-10 {
            for j in 0..3 {
                transition_matrix[[i, j]] /= row_sum;
            }
        } else {
            for j in 0..3 {
                transition_matrix[[i, j]] = 1.0 / 3.0;
            }
        }
    }
    
    transition_matrix
}

/// 更新发射概率矩阵（在线学习）- 更保守的策略
fn update_emission_matrix(
    mut emission_matrix: Array2<f64>,
    state_probs: &Array1<f64>,
    observation: f64,
    learning_rate: f64,
) -> Array2<f64> {
    let obs_idx = observation_to_index(observation);
    let conservative_rate = learning_rate * 0.2; // 降低学习率
    
    // 找到最可能的状态
    let max_state_idx = state_probs.iter().enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
    
    // 只对概率较高的状态进行更新
    if state_probs[max_state_idx] > 0.5 {
        emission_matrix[[max_state_idx, obs_idx]] = 
            (1.0 - conservative_rate) * emission_matrix[[max_state_idx, obs_idx]] + 
            conservative_rate;
    }
    
    // 应用最小概率约束
    let min_emission_prob = 0.1;
    for i in 0..3 {
        for j in 0..3 {
            if emission_matrix[[i, j]] < min_emission_prob {
                emission_matrix[[i, j]] = min_emission_prob;
            }
        }
    }
    
    // 重新归一化每行
    for i in 0..3 {
        let row_sum: f64 = emission_matrix.row(i).sum();
        if row_sum > 1e-10 {
            for j in 0..3 {
                emission_matrix[[i, j]] /= row_sum;
            }
        } else {
            for j in 0..3 {
                emission_matrix[[i, j]] = 1.0 / 3.0;
            }
        }
    }
    
    emission_matrix
}

/// 主要的HMM趋势预测函数
#[pyfunction]
#[pyo3(signature = (prices, window=30, slope_threshold=0.0003, r2_threshold=0.4, learning_rate=0.1))]
pub fn hmm_trend_prediction(
    prices: Vec<f64>,
    window: usize,
    slope_threshold: f64,
    r2_threshold: f64,
    learning_rate: f64,
) -> PyResult<HMMPredictionResult> {
    if prices.len() < window + 10 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "价格序列长度太短，无法进行有效的HMM分析"
        ));
    }
    
    // 1. 使用前30%的数据进行趋势分析
    let training_length = (prices.len() as f64 * 0.3).max(window as f64 * 3.0) as usize;
    let training_prices = &prices[..training_length];
    
    // 执行趋势判断
    let initial_states = linear_regression_trend_analysis(
        training_prices, 
        window, 
        slope_threshold, 
        r2_threshold
    );
    
    // 计算价格变化率
    let mut price_changes = Vec::new();
    for i in 1..training_length {
        price_changes.push((prices[i] / prices[i-1]).ln());
    }
    
    // 2. 初始化HMM参数
    let mut transition_matrix = initialize_transition_matrix(&initial_states[window..]);
    let mut emission_matrix = initialize_emission_matrix(&initial_states[window..], &price_changes[window-1..]);
    
    // 初始状态概率基于训练数据的状态分布
    let mut state_counts = [0; 3];
    for &state in &initial_states[window..] {
        state_counts[state_to_index(state)] += 1;
    }
    
    // 计算状态分布，添加拉普拉斯平滑避免0概率
    let total_states = state_counts.iter().sum::<i32>() as f64;
    let mut current_state_probs = Array1::<f64>::zeros(3);
    for i in 0..3 {
        current_state_probs[i] = (state_counts[i] as f64 + 1.0) / (total_states + 3.0);
    }
    
    // 添加状态平衡机制 - 如果某个状态概率过低，提升它
    let min_prob = 0.15; // 最小状态概率阈值
    for i in 0..3 {
        if current_state_probs[i] < min_prob {
            current_state_probs[i] = min_prob;
        }
    }
    
    // 重新归一化
    let sum: f64 = current_state_probs.sum();
    for i in 0..3 {
        current_state_probs[i] /= sum;
    }
    
    // 3. 逐步预测和更新
    let mut state_predictions = Vec::new();
    let mut updated_state_probs = Vec::new();
    let mut emission_probs_history = Vec::new();
    let mut transition_probs_history = Vec::new();
    
    for i in training_length..prices.len() {
        // 预测下一时刻的状态概率
        let predicted_probs = hmm_forward_prediction(&transition_matrix, &emission_matrix, &current_state_probs);
        
        // 计算实际观测（价格变化）
        let observation = if i > 0 { (prices[i] / prices[i-1]).ln() } else { 0.0 };
        
        // 使用观测更新状态概率
        let updated_probs = update_state_probabilities(&predicted_probs, &emission_matrix, observation);
        
        // 保存结果
        state_predictions.push(predicted_probs.to_vec());
        updated_state_probs.push(updated_probs.to_vec());
        emission_probs_history.push(emission_matrix.to_shape((9,)).unwrap().to_vec().chunks(3).map(|chunk| chunk.to_vec()).collect::<Vec<_>>());
        transition_probs_history.push(transition_matrix.to_shape((9,)).unwrap().to_vec().chunks(3).map(|chunk| chunk.to_vec()).collect::<Vec<_>>());
        
        // 在线更新模型参数
        transition_matrix = update_transition_matrix(transition_matrix, &current_state_probs, &updated_probs, learning_rate);
        emission_matrix = update_emission_matrix(emission_matrix, &updated_probs, observation, learning_rate);
        
        // 更新当前状态概率
        current_state_probs = updated_probs;
    }
    
    Ok(HMMPredictionResult {
        state_predictions,
        updated_state_probs,
        emission_probs: emission_probs_history,
        transition_probs: transition_probs_history,
    })
}