use pyo3::prelude::*;
use numpy::{PyReadonlyArray1, PyArray1, IntoPyArray};
use std::collections::HashMap;

/// 拟合Hawkes自激点过程模型并计算多种指标
///
/// 该函数使用指数核函数 φ(u) = α * exp(-β * u) 拟合Hawkes过程,
/// 计算模型参数和各种金融指标,用于分析逐笔成交数据的自激特性。
///
/// 参数
/// ----------
/// event_times : numpy.ndarray
///     事件时间戳数组(单位:秒),需要是升序排列
/// event_volumes : numpy.ndarray
///     事件对应的成交量数组
/// initial_guess : Optional[Tuple[float, float, float]], optional
///     参数初始猜测值 (mu, alpha, beta)
///     - mu: 外生事件强度(基准强度)
///     - alpha: 自激强度系数
///     - beta: 核函数衰减率
///     默认为None,使用启发式方法自动初始化
/// max_iterations : int, optional
///     EM算法最大迭代次数,默认为1000
/// tolerance : float, optional
///     收敛容差,默认为1e-6
/// cluster_merge_threshold : float, optional
///     簇合并阈值(0-1),控制事件并入已有簇的宽松程度;
///     数值越大越容易把事件并入已有簇,形成更大的簇;
///     数值越小越严格,更倾向于创建新簇
///     默认0.8,建议范围[0.5, 0.95]
///
///     示例:
///     - threshold=0.3: 严格,只合并概率明显更高的事件,产生更多小簇
///     - threshold=0.8: 宽松,容易合并事件,允许产生大簇
///
/// 返回值
/// -------
/// dict
///     包含以下字段的字典:
///     - 'mu': 外生事件强度估计值
///     - 'alpha': 自激强度系数估计值
///     - 'beta': 核函数衰减率估计值
///     - 'branching_ratio': 分枝率 n = α/β
///     - 'mean_intensity': 无条件平均强度 Λ = μ/(1-n)
///     - 'exogenous_intensity': 外生强度 = μ
///     - 'endogenous_intensity': 内生强度 = Λ - μ
///     - 'expected_cluster_size': 期望簇大小 = 1/(1-n)
///     - 'half_life': 半衰期 = ln(2)/β
///     - 'mean_parent_child_interval': 父子平均间隔 = 1/β
///     - 'log_likelihood': 对数似然值
///     - 'event_intensities': 每个事件时刻的强度值
///     - 'root_probabilities': 每个事件是根节点(外生事件)的概率
///     - 'expected_children': 每个事件的预期子女数
///     - 'cluster_assignments': 每个事件所属的簇ID
///     - 'cluster_sizes': 每个簇的大小
///     - 'cluster_durations': 每个簇的持续时间
///     - 'cluster_volumes': 每个簇的成交量总和
///
/// 示例
/// -------
/// >>> import rust_pyfunc as rp
/// >>> import numpy as np
/// >>> # 模拟逐笔成交数据
/// >>> times = np.cumsum(np.random.exponential(0.1, 1000))
/// >>> volumes = np.random.lognormal(10, 1, 1000)
/// >>> result = rp.fit_hawkes_process(times, volumes)
/// >>> print(f"分枝率: {result['branching_ratio']:.3f}")
/// >>> print(f"期望簇大小: {result['expected_cluster_size']:.2f}")
/// >>> print(f"平均强度: {result['mean_intensity']:.3f}")
#[pyfunction]
#[pyo3(
    signature = (event_times, event_volumes, initial_guess=None, max_iterations=1000, tolerance=1e-6, cluster_merge_threshold=0.8, max_parent_search_window=200, parent_time_threshold_factor=10.0, merge_search_window=500, merge_time_threshold_factor=20.0, relax_factor_multiplier=3.0)
)]
pub fn fit_hawkes_process(
    py: Python,
    event_times: PyReadonlyArray1<f64>,
    event_volumes: PyReadonlyArray1<f64>,
    initial_guess: Option<(f64, f64, f64)>,
    max_iterations: usize,
    tolerance: f64,
    cluster_merge_threshold: f64,
    max_parent_search_window: usize,
    parent_time_threshold_factor: f64,
    merge_search_window: usize,
    merge_time_threshold_factor: f64,
    relax_factor_multiplier: f64,
) -> PyResult<PyObject> {
    // 输入验证
    let times = event_times.as_array();
    let volumes = event_volumes.as_array();

    if times.len() != volumes.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "event_times和event_volumes长度必须相同"
        ));
    }

    if times.len() < 10 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "数据点数量必须大于等于10"
        ));
    }

    // 事件时间排序（升序）
    let mut time_volume_pairs: Vec<(f64, f64)> = times.iter()
        .zip(volumes.iter())
        .map(|(&t, &v)| (t, v))
        .collect();
    time_volume_pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    let sorted_times: Vec<f64> = time_volume_pairs.iter().map(|(t, _)| *t).collect();
    let sorted_volumes: Vec<f64> = time_volume_pairs.iter().map(|(_, v)| *v).collect();

    // 时间窗口长度
    let t_start = sorted_times[0];
    let t_end = sorted_times[sorted_times.len() - 1];
    let t_total = t_end - t_start;

    // 初始参数估计 - 数据驱动的方法
    let (mu0, alpha0, beta0) = if let Some((mu_guess, alpha_guess, beta_guess)) = initial_guess {
        (mu_guess, alpha_guess, beta_guess)
    } else {
        // 数据驱动的启发式初始化
        let event_count = sorted_times.len() as f64;
        let avg_intensity = event_count / t_total; // 平均强度 λ = N/T

        // mu (外生强度): 假设约20-30%的事件是外生的
        let mu_init = avg_intensity * 0.25;

        // beta (衰减率): 基于事件时间间隔的分布
        // 典型金融数据：自激效应在几个事件间隔内衰减
        let median_interval = if sorted_times.len() >= 2 {
            let intervals: Vec<f64> = (1..sorted_times.len())
                .map(|i| sorted_times[i] - sorted_times[i-1])
                .collect();
            let mut sorted_intervals = intervals.clone();
            sorted_intervals.sort_by(|a, b| a.partial_cmp(b).unwrap());
            sorted_intervals[sorted_intervals.len() / 2]
        } else {
            t_total / event_count.max(1.0)
        };

        // 设半衰期约为中位数间隔的0.5倍: t_1/2 = ln(2)/β ≈ 0.5 * median_interval
        // 所以 β ≈ ln(2) / (0.5 * median_interval) = 2*ln(2)/median_interval
        let beta_init = (2.0 * 2.0f64.ln()) / median_interval.max(0.001);

        // alpha (自激强度): 基于期望分枝率n=0.3-0.6
        // n = α/β ⇒ α = n * β
        // 从用户notebook结果看，分枝率约为0.4-0.5
        let target_branching_ratio = 0.6; // 提高目标分枝率
        let alpha_init = target_branching_ratio * beta_init;

        (mu_init, alpha_init, beta_init)
    };

    // 使用EM算法或Nelder-Mead优化进行参数拟合
    let (mu, alpha, beta) = estimate_parameters_em(
        &sorted_times,
        mu0,
        alpha0,
        beta0,
        max_iterations,
        tolerance,
    );

    // 计算各项指标
    let branching_ratio = alpha / beta;
    let mean_intensity = mu / (1.0 - branching_ratio.max(0.0).min(0.95)); // 避免除零
    let exogenous_intensity = mu;
    let endogenous_intensity = mean_intensity - mu;
    let expected_cluster_size = 1.0 / (1.0 - branching_ratio.max(0.0).min(0.95));
    let half_life = (2.0f64.ln()) / beta;
    let mean_parent_child_interval = 1.0 / beta;

    // 计算事件级强度
    let mut event_intensities = Vec::with_capacity(sorted_times.len());
    let mut lambda = mu; // 初始强度
    event_intensities.push(lambda + alpha);

    for i in 1..sorted_times.len() {
        let delta_t = sorted_times[i] - sorted_times[i - 1];
        lambda = mu + (-beta * delta_t).exp() * (lambda - mu) + alpha;
        event_intensities.push(lambda);
    }

    // 计算对数似然
    let log_likelihood = calculate_log_likelihood(
        &sorted_times,
        t_start,
        t_end,
        mu,
        alpha,
        beta,
    );

    // 计算簇分配和指标
    let result = calculate_cluster_indicators(
        &sorted_times,
        &sorted_volumes,
        mu,
        alpha,
        beta,
        &event_intensities,
        cluster_merge_threshold,
        max_parent_search_window,
        parent_time_threshold_factor,
        merge_search_window,
        merge_time_threshold_factor,
        relax_factor_multiplier,
    );

    // 构建返回的字典
    let result_dict = pyo3::types::PyDict::new(py);
    result_dict.set_item("mu", mu)?;
    result_dict.set_item("alpha", alpha)?;
    result_dict.set_item("beta", beta)?;
    result_dict.set_item("branching_ratio", branching_ratio)?;
    result_dict.set_item("mean_intensity", mean_intensity)?;
    result_dict.set_item("exogenous_intensity", exogenous_intensity)?;
    result_dict.set_item("endogenous_intensity", endogenous_intensity)?;
    result_dict.set_item("expected_cluster_size", expected_cluster_size)?;
    result_dict.set_item("half_life", half_life)?;
    result_dict.set_item("mean_parent_child_interval", mean_parent_child_interval)?;
    result_dict.set_item("log_likelihood", log_likelihood)?;

    // 数组数据
    let intensities_array: &PyArray1<f64> = event_intensities.into_pyarray(py);
    result_dict.set_item("event_intensities", intensities_array)?;

    let root_probs_array: &PyArray1<f64> = result.root_probabilities.into_pyarray(py);
    result_dict.set_item("root_probabilities", root_probs_array)?;

    let expected_children_array: &PyArray1<f64> = result.expected_children.into_pyarray(py);
    result_dict.set_item("expected_children", expected_children_array)?;

    let cluster_assignments_array: &PyArray1<i32> = result.cluster_assignments.into_pyarray(py);
    result_dict.set_item("cluster_assignments", cluster_assignments_array)?;

    let cluster_sizes_array: &PyArray1<i32> = result.cluster_sizes.into_pyarray(py);
    result_dict.set_item("cluster_sizes", cluster_sizes_array)?;

    let cluster_durations_array: &PyArray1<f64> = result.cluster_durations.into_pyarray(py);
    result_dict.set_item("cluster_durations", cluster_durations_array)?;

    let cluster_volumes_array: &PyArray1<f64> = result.cluster_volumes.into_pyarray(py);
    result_dict.set_item("cluster_volumes", cluster_volumes_array)?;

    Ok(result_dict.into())
}

/// 计算Hawkes过程的事件级指标（需要价格数据）
///
/// 该函数在fit_hawkes_process的基础上,增加了需要价格数据的指标计算。
/// 额外计算的指标主要用于分析每个事件对价格的影响。
///
/// 参数
/// ----------
/// event_times : numpy.ndarray
///     事件时间戳数组(单位:秒),需要是升序排列
/// event_volumes : numpy.ndarray
///     事件对应的成交量数组
/// event_prices : numpy.ndarray
///     事件对应的价格数组
/// initial_guess : Optional[Tuple[float, float, float]], optional
///     参数初始猜测值 (mu, alpha, beta),默认为None
/// max_iterations : int, optional
///     EM算法最大迭代次数,默认为1000
/// tolerance : float, optional
///     收敛容差,默认为1e-6
/// cluster_merge_threshold : float, optional
///     簇合并阈值(0-1),控制事件并入已有簇的宽松程度;
///     数值越大越容易把事件并入已有簇,形成更大的簇;
///     数值越小越严格,更倾向于创建新簇
///     默认0.8,建议范围[0.5, 0.95]
///
/// 返回值
/// -------
/// dict
///     包含fit_hawkes_process的所有字段,以及：
///     - 'cluster_price_changes': 每个簇的价格变化(簇结束时价格 - 簇开始时价格)
///     - 'time_intervals': 连续事件间的时间间隔
///
/// 新指标解释
/// ------------
/// 1. 簇价格变化(cluster_price_changes): 每个成交簇从开始到结束的价格变化,
///    反映该簇交易活动对价格的影响方向和幅度。
///
/// 2. 时间间隔(time_intervals): 连续成交事件之间的时间间隔,
///    可用于分析市场活跃度的时间模式。
///
/// 示例
/// -------
/// >>> import rust_pyfunc as rp
/// >>> import numpy as np
/// >>> # 读取真实逐笔成交数据
/// >>> df = read_trade_data('000001', 20220101)
/// >>> df['time_seconds'] = (df.exchtime - df.exchtime.min()).dt.total_seconds()
/// >>> result = rp.hawkes_event_indicators(
/// ...     df.time_seconds.to_numpy(),
/// ...     df.volume.to_numpy(),
/// ...     df.price.to_numpy()
/// ... )
/// >>> # 分析大簇的价格影响
/// >>> large_clusters = np.array(result['cluster_sizes']) > 10
/// >>> price_changes = np.array(result['cluster_price_changes'])[large_clusters]
/// >>> print(f"大簇平均价格变化: {np.mean(price_changes):.4f}")
#[pyfunction]
#[pyo3(
    signature = (event_times, event_volumes, event_prices, initial_guess=None, max_iterations=1000, tolerance=1e-6, cluster_merge_threshold=0.8, max_parent_search_window=200, parent_time_threshold_factor=10.0, merge_search_window=500, merge_time_threshold_factor=20.0, relax_factor_multiplier=3.0)
)]
pub fn hawkes_event_indicators(
    py: Python,
    event_times: PyReadonlyArray1<f64>,
    event_volumes: PyReadonlyArray1<f64>,
    event_prices: PyReadonlyArray1<f64>,
    initial_guess: Option<(f64, f64, f64)>,
    max_iterations: usize,
    tolerance: f64,
    cluster_merge_threshold: f64,
    max_parent_search_window: usize,
    parent_time_threshold_factor: f64,
    merge_search_window: usize,
    merge_time_threshold_factor: f64,
    relax_factor_multiplier: f64,
) -> PyResult<PyObject> {
    // 事件时间和价格排序
    let times = event_times.as_array();
    let volumes = event_volumes.as_array();
    let prices = event_prices.as_array();

    let mut time_price_volume: Vec<(f64, f64, f64)> = times.iter()
        .zip(prices.iter())
        .zip(volumes.iter())
        .map(|((&t, &p), &v)| (t, p, v))
        .collect();
    time_price_volume.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    let sorted_times: Vec<f64> = time_price_volume.iter().map(|(t, _, _)| *t).collect();
    let sorted_prices: Vec<f64> = time_price_volume.iter().map(|(_, p, _)| *p).collect();
    let sorted_volumes: Vec<f64> = time_price_volume.iter().map(|(_, _, v)| *v).collect();

    // 使用相同的方法获取参数
    let (mu, alpha, beta) = estimate_parameters_em(
        &sorted_times,
        initial_guess.unwrap_or((0.5, 0.2, 1.0)).0,
        initial_guess.unwrap_or((0.5, 0.2, 1.0)).1,
        initial_guess.unwrap_or((0.5, 0.2, 1.0)).2,
        max_iterations,
        tolerance,
    );

    // 计算事件级强度
    let mut event_intensities = Vec::with_capacity(sorted_times.len());
    let mut lambda = mu;
    event_intensities.push(mu + alpha);

    for i in 1..sorted_times.len() {
        let delta_t = sorted_times[i] - sorted_times[i - 1];
        lambda = mu + (-beta * delta_t).exp() * (lambda - mu) + alpha;
        event_intensities.push(lambda);
    }

    // 计算簇指标
    let cluster_result = calculate_cluster_indicators(
        &sorted_times,
        &sorted_volumes,
        mu,
        alpha,
        beta,
        &event_intensities,
        cluster_merge_threshold,
        max_parent_search_window,
        parent_time_threshold_factor,
        merge_search_window,
        merge_time_threshold_factor,
        relax_factor_multiplier,
    );

    // 计算簇内价格变化
    let mut cluster_price_changes = vec![0.0; cluster_result.cluster_assignments.len()];
    let mut cluster_start_price: HashMap<i32, (f64, f64)> = HashMap::new();
    let mut cluster_end_price: HashMap<i32, (f64, f64)> = HashMap::new();

    for (i, &cluster_id) in cluster_result.cluster_assignments.iter().enumerate() {
        let time = sorted_times[i];
        let price = sorted_prices[i];

        cluster_start_price.entry(cluster_id)
            .and_modify(|(t, p)| { if time < *t { *t = time; *p = price; } })
            .or_insert((time, price));

        cluster_end_price.entry(cluster_id)
            .and_modify(|(t, p)| { if time > *t { *t = time; *p = price; } })
            .or_insert((time, price));
    }

    for (i, &cluster_id) in cluster_result.cluster_assignments.iter().enumerate() {
        if let (Some(&(_, start_price)), Some(&(_, end_price))) =
            (cluster_start_price.get(&cluster_id), cluster_end_price.get(&cluster_id)) {
            cluster_price_changes[i] = end_price - start_price;
        }
    }

    // 计算事件间时间间隔
    let mut time_intervals = vec![0.0; sorted_times.len()];
    for i in 1..sorted_times.len() {
        time_intervals[i] = sorted_times[i] - sorted_times[i-1];
    }

    // 构建结果字典
    let result_dict = pyo3::types::PyDict::new(py);

    // 计算各项指标
    let branching_ratio = alpha / beta;
    let mean_intensity = mu / (1.0 - branching_ratio.max(0.0).min(0.95));
    let exogenous_intensity = mu;
    let endogenous_intensity = mean_intensity - mu;
    let expected_cluster_size = 1.0 / (1.0 - branching_ratio.max(0.0).min(0.95));
    let half_life = (2.0f64.ln()) / beta;
    let mean_parent_child_interval = 1.0 / beta;

    // 基础指标
    result_dict.set_item("mu", mu)?;
    result_dict.set_item("alpha", alpha)?;
    result_dict.set_item("beta", beta)?;
    result_dict.set_item("branching_ratio", branching_ratio)?;
    result_dict.set_item("mean_intensity", mean_intensity)?;
    result_dict.set_item("exogenous_intensity", exogenous_intensity)?;
    result_dict.set_item("endogenous_intensity", endogenous_intensity)?;
    result_dict.set_item("expected_cluster_size", expected_cluster_size)?;
    result_dict.set_item("half_life", half_life)?;
    result_dict.set_item("mean_parent_child_interval", mean_parent_child_interval)?;

    // 计算对数似然
    let log_likelihood = calculate_log_likelihood(
        &sorted_times,
        sorted_times[0],
        sorted_times[sorted_times.len() - 1],
        mu,
        alpha,
        beta,
    );
    result_dict.set_item("log_likelihood", log_likelihood)?;

    // 数组数据
    let intensities_array: &PyArray1<f64> = event_intensities.into_pyarray(py);
    result_dict.set_item("event_intensities", intensities_array)?;

    let root_probs_array: &PyArray1<f64> = cluster_result.root_probabilities.into_pyarray(py);
    result_dict.set_item("root_probabilities", root_probs_array)?;

    let expected_children_array: &PyArray1<f64> = cluster_result.expected_children.into_pyarray(py);
    result_dict.set_item("expected_children", expected_children_array)?;

    let cluster_assignments_array: &PyArray1<i32> = cluster_result.cluster_assignments.into_pyarray(py);
    result_dict.set_item("cluster_assignments", cluster_assignments_array)?;

    let cluster_sizes_array: &PyArray1<i32> = cluster_result.cluster_sizes.into_pyarray(py);
    result_dict.set_item("cluster_sizes", cluster_sizes_array)?;

    let cluster_durations_array: &PyArray1<f64> = cluster_result.cluster_durations.into_pyarray(py);
    result_dict.set_item("cluster_durations", cluster_durations_array)?;

    let cluster_volumes_array: &PyArray1<f64> = cluster_result.cluster_volumes.into_pyarray(py);
    result_dict.set_item("cluster_volumes", cluster_volumes_array)?;

    // 添加新的字段
    let cluster_price_changes_array: &PyArray1<f64> = cluster_price_changes.into_pyarray(py);
    result_dict.set_item("cluster_price_changes", cluster_price_changes_array)?;

    let time_intervals_array: &PyArray1<f64> = time_intervals.into_pyarray(py);
    result_dict.set_item("time_intervals", time_intervals_array)?;

    Ok(result_dict.into())
}

/// EM算法估计Hawkes参数（带约束版本）
fn estimate_parameters_em(
    event_times: &[f64],
    mu0: f64,
    alpha0: f64,
    beta0: f64,
    max_iterations: usize,
    tolerance: f64,
) -> (f64, f64, f64) {
    let n = event_times.len();
    if n == 0 {
        return (mu0, alpha0, beta0);
    }

    let t_start = event_times[0];
    let t_end = event_times[n - 1];
    let total_time = t_end - t_start;
    let avg_interval = total_time / n as f64;

    // 更合理的初始值
    let mut mu = mu0.max(0.001).min(1000.0);
    let mut alpha = alpha0.max(0.001).min(1000.0);

    // β的初始值：基于平均间隔，假设自激作用主要在几个间隔内衰减
    let mut beta = if beta0 > 0.01 {
        beta0
    } else {
        // 启发式：β ≈ 5 / 平均间隔，使半衰期约为平均间隔的0.2倍
        (5.0 / avg_interval).max(0.01).min(1000.0)
    };

    // 确保初始分枝率合理
    if alpha / beta >= 0.95 {
        alpha = beta * 0.5; // 初始分枝率设为0.5
    }

    let mut prev_log_likelihood = calculate_log_likelihood(event_times, t_start, t_end, mu, alpha, beta);

    for _ in 0..max_iterations {
        let mu_old = mu;
        let alpha_old = alpha;
        let beta_old = beta;

        // E-step: 计算每个事件是背景还是触发
        let mut background_sum = 0.0;
        let mut trigger_contributions = vec![0.0; n];

        // 计算事件级强度
        let mut intensities = vec![mu + alpha; n];
        for i in 1..n {
            let delta_t = event_times[i] - event_times[i-1];
            intensities[i] = mu + (-beta * delta_t).exp() * (intensities[i-1] - mu) + alpha;
        }

        // 计算背景概率和触发贡献
        for i in 0..n {
            // 背景概率 p = μ / λ
            let p_background = (mu / intensities[i]).min(1.0).max(0.0);
            background_sum += p_background;

            // 触发概率：事件i作为父节点对事件j的触发贡献
            for j in (i+1..n).take(200) { // 限制范围以提高速度
                let delta_t = event_times[j] - event_times[i];
                if delta_t > 10.0 / beta { // 超过10倍衰减常数后贡献很小
                    break;
                }
                let contribution = alpha * (-beta * delta_t).exp();
                trigger_contributions[i] += contribution / intensities[j];
            }
        }

        // M-step: 更新参数
        mu = (background_sum / total_time).max(0.001).min(1000.0);

        let trigger_sum: f64 = trigger_contributions.iter().sum();
        let time_weighted_sum: f64 = (0..n).map(|i| {
            (i+1..n).map(|j| {
                let delta_t = event_times[j] - event_times[i];
                if delta_t > 10.0 / beta {
                    0.0
                } else {
                    let exp_term = (-beta * delta_t).exp();
                    exp_term * delta_t
                }
            }).sum::<f64>()
        }).sum();

        if trigger_sum > 0.0 && time_weighted_sum > 0.0 {
            alpha = (trigger_sum / n as f64).max(0.001).min(1000.0);
            beta = (trigger_sum / time_weighted_sum).max(0.01).min(1000.0);
        }

        // 关键约束：确保分枝率 n = α/β < 0.95（稳定性条件）
        let branching_ratio = alpha / beta;
        if branching_ratio >= 0.95 {
            // 调整alpha以保持分枝率在合理范围
            alpha = beta * 0.8; // 将分枝率设为0.8
        }

        // 计算当前对数似然并检查是否改进
        let current_log_likelihood = calculate_log_likelihood(event_times, t_start, t_end, mu, alpha, beta);

        // 如果对数似然没有改进，停止迭代
        if current_log_likelihood - prev_log_likelihood < tolerance {
            break;
        }
        prev_log_likelihood = current_log_likelihood;

        // 检查参数收敛
        if (mu - mu_old).abs() < tolerance &&
           (alpha - alpha_old).abs() < tolerance &&
           (beta - beta_old).abs() < tolerance {
            break;
        }
    }

    (mu, alpha, beta)
}

/// 计算对数似然
fn calculate_log_likelihood(
    event_times: &[f64],
    t_start: f64,
    t_end: f64,
    mu: f64,
    alpha: f64,
    beta: f64,
) -> f64 {
    let n = event_times.len();
    if n == 0 {
        return 0.0;
    }

    // 计算事件级强度
    let mut intensities = vec![mu + alpha; n];
    for i in 1..n {
        let delta_t = event_times[i] - event_times[i-1];
        intensities[i] = mu + (-beta * delta_t).exp() * (intensities[i-1] - mu) + alpha;
    }

    // 对数似然的事件部分
    let log_likelihood_event: f64 = intensities.iter()
        .map(|&lambda| lambda.ln())
        .sum();

    // 对数似然的积分部分
    let integral_part = mu * (t_end - t_start) +
        (alpha / beta) * (n as f64 - (0..n).map(|i| {
            (-beta * (t_end - event_times[i])).exp()
        }).sum::<f64>());

    log_likelihood_event - integral_part
}

/// 计算簇级别的指标
struct ClusterIndicators {
    root_probabilities: Vec<f64>,
    expected_children: Vec<f64>,
    cluster_assignments: Vec<i32>,
    cluster_sizes: Vec<i32>,
    cluster_durations: Vec<f64>,
    cluster_volumes: Vec<f64>,
}

fn calculate_cluster_indicators(
    event_times: &[f64],
    event_volumes: &[f64],
    mu: f64,
    alpha: f64,
    beta: f64,
    event_intensities: &[f64],
    cluster_merge_threshold: f64,
    max_parent_search_window: usize,
    parent_time_threshold_factor: f64,
    merge_search_window: usize,
    merge_time_threshold_factor: f64,
    relax_factor_multiplier: f64,
) -> ClusterIndicators {
    let n = event_times.len();
    let merge_threshold = cluster_merge_threshold.clamp(0.0, 1.0);
    let relax_factor = (1.0 - merge_threshold).max(0.0); // 阈值越小越宽松
    let mut root_probabilities = vec![0.0; n];
    let mut expected_children = vec![0.0; n];
    let mut cluster_assignments = vec![0; n];
    let mut cluster_sizes: HashMap<i32, usize> = HashMap::new();
    let mut cluster_start_time: HashMap<i32, f64> = HashMap::new();
    let mut cluster_end_time: HashMap<i32, f64> = HashMap::new();
    let mut cluster_total_volume: HashMap<i32, f64> = HashMap::new();

    // 计算每个事件的根概率和预期子女数
    for i in 0..n {
        root_probabilities[i] = mu / event_intensities[i];

        // 计算事件i作为父节点对后续事件的触发概率
        for j in (i+1..n).take(max_parent_search_window) { // 使用可调参数
            let delta_t = event_times[j] - event_times[i];
            if delta_t > parent_time_threshold_factor / beta {
                break;
            }
            expected_children[i] += alpha * (-beta * delta_t).exp() / event_intensities[j];
        }
    }

    // 改进的簇分配算法：更宽松的条件
    let mut next_cluster_id = 0;
    for i in 0..n {
        let root_prob = mu / event_intensities[i];
        let mut max_prob = root_prob;
        let mut parent_idx = -1; // -1表示背景

        // 搜索更多候选父节点（使用可调参数）
        // 限制时间窗口，并随着阈值降低进一步放宽
        let search_window =
            ((merge_search_window as f64 * (1.0 + relax_factor * 2.0)).round() as usize).min(i);
        let time_threshold = (merge_time_threshold_factor / beta) * (1.0 + relax_factor * relax_factor_multiplier);

        for j in (0..i).rev().take(search_window) {
            let delta_t = event_times[i] - event_times[j];
            if delta_t > time_threshold {
                break;
            }

            // 计算父节点概率
            let prob = alpha * (-beta * delta_t).exp() / event_intensities[i];

            if prob > max_prob {
                max_prob = prob;
                parent_idx = j as i32;
            }
        }

        // 更宽松的判断：如果非根概率相对根概率足够大，也分配为子节点
        // 条件：max_prob > root_prob * merge_threshold（阈值越大越容易合并）
        if parent_idx >= 0 && max_prob > root_prob * merge_threshold {
            // 分配给父节点的簇
            cluster_assignments[i] = cluster_assignments[parent_idx as usize];
            let cluster_id = cluster_assignments[i];

            // 更新簇的时间和卷
            if let Some(start_time) = cluster_start_time.get_mut(&cluster_id) {
                if event_times[i] < *start_time {
                    *start_time = event_times[i];
                }
            }
            if let Some(end_time) = cluster_end_time.get_mut(&cluster_id) {
                if event_times[i] > *end_time {
                    *end_time = event_times[i];
                }
            }
            if let Some(total_volume) = cluster_total_volume.get_mut(&cluster_id) {
                *total_volume += event_volumes[i];
            }
        } else {
            // 新簇
            cluster_assignments[i] = next_cluster_id;
            cluster_start_time.insert(next_cluster_id, event_times[i]);
            cluster_end_time.insert(next_cluster_id, event_times[i]);
            cluster_total_volume.insert(next_cluster_id, event_volumes[i]);
            next_cluster_id += 1;
        }

        // 更新簇大小
        *cluster_sizes.entry(cluster_assignments[i]).or_insert(0) += 1;
    }

    // 将HashMap转换为Vec
    // 首先需要找到最大的cluster_id，确保向量足够大
    let max_cluster_id = cluster_sizes.keys()
        .map(|&id| id)
        .max()
        .unwrap_or(-1);

    let vec_size = (max_cluster_id + 1) as usize;
    let mut cluster_sizes_vec = vec![0; vec_size];
    let mut cluster_durations_vec = vec![0.0; vec_size];
    let mut cluster_volumes_vec = vec![0.0; vec_size];

    for (&cluster_id, &size) in &cluster_sizes {
        let idx = cluster_id as usize;
        if idx < vec_size {
            cluster_sizes_vec[idx] = size as i32;

            if let (Some(&start), Some(&end)) = (cluster_start_time.get(&cluster_id), cluster_end_time.get(&cluster_id)) {
                cluster_durations_vec[idx] = end - start;
            }

            if let Some(&volume) = cluster_total_volume.get(&cluster_id) {
                cluster_volumes_vec[idx] = volume;
            }
        }
    }

    ClusterIndicators {
        root_probabilities,
        expected_children,
        cluster_assignments,
        cluster_sizes: cluster_sizes_vec,
        cluster_durations: cluster_durations_vec,
        cluster_volumes: cluster_volumes_vec,
    }
}
