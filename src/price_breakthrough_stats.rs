use numpy::{PyArray2, PyReadonlyArray1};
use pyo3::prelude::*;
use std::collections::HashSet;

/// 计算股票逐笔成交数据中"价格未突破上一分钟价格范围"的24个统计指标
///
/// 参数:
/// exchtime: 成交时间(纳秒)
/// volume: 成交量(支持浮点数)
/// price: 成交价格
/// flag: 主动买卖标识(66=主买, 83=主卖)
///
/// 返回:
/// (n×24的二维数组, 24个中文列名)
#[pyfunction]
pub fn compute_non_breakthrough_stats(
    py: Python,
    exchtime: PyReadonlyArray1<i64>,
    volume: PyReadonlyArray1<f64>,
    price: PyReadonlyArray1<f64>,
    flag: PyReadonlyArray1<i64>,
) -> PyResult<(Py<PyArray2<f64>>, Vec<String>)> {
    let exchtime = exchtime.as_array();
    let volume = volume.as_array();
    let price = price.as_array();
    let flag = flag.as_array();
    let n = exchtime.len();

    if n == 0 || volume.len() != n || price.len() != n || flag.len() != n {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "输入数组长度不一致或为空",
        ));
    }

    // 准备列名
    let column_names = vec![
        "未突破成交量总和".to_string(),
        "未突破成交量占本分钟比例".to_string(),
        "未突破成交量占上分钟比例".to_string(),
        "未突破交易笔数".to_string(),
        "未突破笔数占本分钟比例".to_string(),
        "未突破笔数占上分钟比例".to_string(),
        "未突破最后一笔秒数".to_string(),
        "未突破第一笔秒数".to_string(),
        "未突破时间跨度秒数".to_string(),
        "未突破时间段内总成交量".to_string(),
        "未突破时间段成交量占本分钟比例".to_string(),
        "未突破时间段内总交易笔数".to_string(),
        "未突破时间段笔数占本分钟比例".to_string(),
        "未突破主买笔数".to_string(),
        "未突破主买笔数比例".to_string(),
        "未突破主买成交量".to_string(),
        "未突破主买成交量比例".to_string(),
        "平均未突破时间跨度".to_string(),
        "未突破成交量与序号相关性".to_string(),
        "未突破标志与序号相关性".to_string(),
        "未突破秒数均值".to_string(),
        "未突破秒数标准差".to_string(),
        "未突破秒数偏度".to_string(),
        "未突破秒数峰度".to_string(),
        "分钟时间标记".to_string(),
    ];

    // 按分钟分组数据
    let minute_groups = group_by_minute(
        exchtime.as_slice().unwrap(),
        volume.as_slice().unwrap(),
        price.as_slice().unwrap(),
        flag.as_slice().unwrap(),
    );
    let num_minutes = minute_groups.len();

    if num_minutes == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "没有有效的分钟数据",
        ));
    }

    // 创建结果矩阵
    let mut result = vec![vec![f64::NAN; 25]; num_minutes];

    // 先为所有分钟设置时间标记（第25列）
    for i in 0..num_minutes {
        let minute_timestamp = minute_groups[i].exchtime[0];
        let minute_start = (minute_timestamp / (60 * 1_000_000_000)) * (60 * 1_000_000_000);
        result[i][24] = minute_start as f64;
    }

    // 计算每分钟的统计指标（从第1分钟开始，因为第0分钟没有上一分钟数据）
    for i in 1..num_minutes {
        let current_minute = &minute_groups[i];
        let prev_minute = &minute_groups[i - 1];

        // 构建上一分钟的价格集合
        let prev_prices: HashSet<OrderedFloat> =
            prev_minute.price.iter().map(|&p| OrderedFloat(p)).collect();

        if prev_prices.is_empty() {
            continue;
        }

        // 找出当前分钟中价格未突破的交易
        let mut breakthrough_indices = Vec::new();

        for (j, &curr_price) in current_minute.price.iter().enumerate() {
            if prev_prices.contains(&OrderedFloat(curr_price)) {
                breakthrough_indices.push(j);
            }
        }

        if breakthrough_indices.is_empty() {
            // 没有未突破的交易，前24个值保持NaN（时间标记已经设置）
            continue;
        }

        // 计算25个指标（包含时间标记）
        let minute_timestamp = minute_groups[i].exchtime[0]; // 获取时间戳用于函数调用
        let stats = calculate_breakthrough_stats(
            current_minute,
            prev_minute,
            &breakthrough_indices,
            minute_timestamp,
        );

        result[i] = stats;
    }

    // 转换为numpy数组
    let py_array = PyArray2::from_vec2(py, &result)?;

    Ok((py_array.into(), column_names))
}

/// 包装f64以实现Hash和Eq
#[derive(Debug, Clone, Copy, PartialEq)]
struct OrderedFloat(f64);

impl Eq for OrderedFloat {}

impl std::hash::Hash for OrderedFloat {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.to_bits().hash(state);
    }
}

impl OrderedFloat {
    #[allow(dead_code)]
    fn value(self) -> f64 {
        self.0
    }
}

/// 一分钟内的交易数据
#[derive(Debug, Clone)]
struct MinuteData {
    exchtime: Vec<i64>,
    volume: Vec<f64>,
    price: Vec<f64>,
    flag: Vec<i64>,
}

/// 将交易数据按分钟分组
fn group_by_minute(
    exchtime: &[i64],
    volume: &[f64],
    price: &[f64],
    flag: &[i64],
) -> Vec<MinuteData> {
    let mut groups: std::collections::BTreeMap<i64, MinuteData> = std::collections::BTreeMap::new();

    for i in 0..exchtime.len() {
        // 将纳秒时间戳转换为分钟
        let minute_timestamp = exchtime[i] / 1_000_000_000 / 60 * 60 * 1_000_000_000;

        let entry = groups
            .entry(minute_timestamp)
            .or_insert_with(|| MinuteData {
                exchtime: Vec::new(),
                volume: Vec::new(),
                price: Vec::new(),
                flag: Vec::new(),
            });

        entry.exchtime.push(exchtime[i]);
        entry.volume.push(volume[i]);
        entry.price.push(price[i]);
        entry.flag.push(flag[i]);
    }

    groups.into_values().collect()
}

/// 计算25个突破统计指标
fn calculate_breakthrough_stats(
    current: &MinuteData,
    prev: &MinuteData,
    breakthrough_indices: &[usize],
    minute_timestamp: i64,
) -> Vec<f64> {
    let mut stats = vec![f64::NAN; 25];

    if breakthrough_indices.is_empty() {
        return stats;
    }

    // 基本统计
    let current_total_volume: f64 = current.volume.iter().sum();
    let prev_total_volume: f64 = prev.volume.iter().sum();
    let current_total_trades = current.volume.len();
    let prev_total_trades = prev.volume.len();

    // 未突破的统计
    let breakthrough_volume: f64 = breakthrough_indices
        .iter()
        .map(|&i| current.volume[i])
        .sum();
    let breakthrough_trades = breakthrough_indices.len();

    // 列1: 未突破成交量总和
    stats[0] = breakthrough_volume;

    // 列2: 未突破成交量占本分钟比例
    if current_total_volume > 0.0 {
        stats[1] = breakthrough_volume / current_total_volume;
    }

    // 列3: 未突破成交量占上分钟比例
    if prev_total_volume > 0.0 {
        stats[2] = breakthrough_volume / prev_total_volume;
    }

    // 列4: 未突破交易笔数
    stats[3] = breakthrough_trades as f64;

    // 列5: 未突破笔数占本分钟比例
    if current_total_trades > 0 {
        stats[4] = breakthrough_trades as f64 / current_total_trades as f64;
    }

    // 列6: 未突破笔数占上分钟比例
    if prev_total_trades > 0 {
        stats[5] = breakthrough_trades as f64 / prev_total_trades as f64;
    }

    // 获取未突破交易的时间信息
    let breakthrough_times: Vec<i64> = breakthrough_indices
        .iter()
        .map(|&i| current.exchtime[i])
        .collect();

    if breakthrough_times.len() >= 2 {
        let first_time = *breakthrough_times.first().unwrap();
        let last_time = *breakthrough_times.last().unwrap();

        // 列7: 未突破最后一笔秒数
        stats[6] = ((last_time % (60 * 1_000_000_000)) / 1_000_000_000) as f64;

        // 列8: 未突破第一笔秒数
        stats[7] = ((first_time % (60 * 1_000_000_000)) / 1_000_000_000) as f64;

        // 列9: 未突破时间跨度
        stats[8] = stats[6] - stats[7];

        // 计算时间段内的统计(从第一次到最后一次未突破)
        let mut period_volume = 0.0f64;
        let mut period_trades = 0;

        for i in 0..current.exchtime.len() {
            if current.exchtime[i] >= first_time && current.exchtime[i] <= last_time {
                period_volume += current.volume[i];
                period_trades += 1;
            }
        }

        // 列10: 时间段内总成交量
        stats[9] = period_volume;

        // 列11: 时间段成交量占本分钟比例
        if current_total_volume > 0.0 {
            stats[10] = period_volume / current_total_volume;
        }

        // 列12: 时间段内总交易笔数
        stats[11] = period_trades as f64;

        // 列13: 时间段笔数占本分钟比例
        if current_total_trades > 0 {
            stats[12] = period_trades as f64 / current_total_trades as f64;
        }
    } else if breakthrough_times.len() == 1 {
        // 只有一笔未突破交易
        let time = breakthrough_times[0];
        let second = ((time % (60 * 1_000_000_000)) / 1_000_000_000) as f64;

        stats[6] = second; // 最后一笔
        stats[7] = second; // 第一笔
        stats[8] = 0.0; // 时间跨度

        stats[9] = breakthrough_volume; // 时间段成交量就是这一笔
        stats[10] = stats[1]; // 比例相同
        stats[11] = 1.0; // 时间段笔数就是1
        stats[12] = stats[4]; // 比例相同
    }

    // 列14: 未突破主买笔数
    let breakthrough_buy_trades = breakthrough_indices
        .iter()
        .filter(|&&i| current.flag[i] == 66)
        .count();
    stats[13] = breakthrough_buy_trades as f64;

    // 列15: 未突破主买笔数比例
    if breakthrough_trades > 0 {
        stats[14] = breakthrough_buy_trades as f64 / breakthrough_trades as f64;
    }

    // 列16: 未突破主买成交量
    let breakthrough_buy_volume: f64 = breakthrough_indices
        .iter()
        .filter(|&&i| current.flag[i] == 66)
        .map(|&i| current.volume[i])
        .sum();
    stats[15] = breakthrough_buy_volume;

    // 列17: 未突破主买成交量比例
    if breakthrough_volume > 0.0 {
        stats[16] = breakthrough_buy_volume / breakthrough_volume;
    }

    // 列18: 平均未突破时间跨度
    if breakthrough_trades > 0 {
        stats[17] = stats[8] / breakthrough_trades as f64;
    }

    // 列19-20: 相关性计算
    if breakthrough_indices.len() >= 2 {
        let volumes: Vec<f64> = breakthrough_indices
            .iter()
            .map(|&i| current.volume[i])
            .collect();
        let flags: Vec<f64> = breakthrough_indices
            .iter()
            .map(|&i| if current.flag[i] == 66 { 1.0 } else { 0.0 })
            .collect();
        let sequence: Vec<f64> = (1..=breakthrough_indices.len()).map(|i| i as f64).collect();

        stats[18] = pearson_correlation(&volumes, &sequence);
        stats[19] = pearson_correlation(&flags, &sequence);
    }

    // 列21-24: 秒数的统计量
    let seconds: Vec<f64> = breakthrough_indices
        .iter()
        .map(|&i| ((current.exchtime[i] % (60 * 1_000_000_000)) / 1_000_000_000) as f64)
        .collect();

    if !seconds.is_empty() {
        stats[20] = mean(&seconds);
        stats[21] = std_dev(&seconds);
        stats[22] = skewness(&seconds);
        stats[23] = kurtosis(&seconds);
    }

    // 列25: 分钟时间标记（该分钟的第0秒对应的纳秒时间戳）
    let minute_start = (minute_timestamp / (60 * 1_000_000_000)) * (60 * 1_000_000_000);
    stats[24] = minute_start as f64;

    stats
}

/// 计算皮尔逊相关系数
fn pearson_correlation(x: &[f64], y: &[f64]) -> f64 {
    if x.len() != y.len() || x.len() < 2 {
        return f64::NAN;
    }

    let _n = x.len() as f64;
    let mean_x = mean(x);
    let mean_y = mean(y);

    let mut numerator = 0.0;
    let mut sum_sq_x = 0.0;
    let mut sum_sq_y = 0.0;

    for i in 0..x.len() {
        let dx = x[i] - mean_x;
        let dy = y[i] - mean_y;
        numerator += dx * dy;
        sum_sq_x += dx * dx;
        sum_sq_y += dy * dy;
    }

    let denominator = (sum_sq_x * sum_sq_y).sqrt();
    if denominator == 0.0 {
        f64::NAN
    } else {
        numerator / denominator
    }
}

/// 计算均值
fn mean(data: &[f64]) -> f64 {
    if data.is_empty() {
        f64::NAN
    } else {
        data.iter().sum::<f64>() / data.len() as f64
    }
}

/// 计算标准差
fn std_dev(data: &[f64]) -> f64 {
    if data.len() < 2 {
        return f64::NAN;
    }

    let mean_val = mean(data);
    let variance =
        data.iter().map(|x| (x - mean_val).powi(2)).sum::<f64>() / (data.len() - 1) as f64;

    variance.sqrt()
}

/// 计算偏度
fn skewness(data: &[f64]) -> f64 {
    if data.len() < 3 {
        return f64::NAN;
    }

    let mean_val = mean(data);
    let std_val = std_dev(data);

    if std_val == 0.0 || std_val.is_nan() {
        return f64::NAN;
    }

    let n = data.len() as f64;
    let m3 = data
        .iter()
        .map(|x| ((x - mean_val) / std_val).powi(3))
        .sum::<f64>()
        / n;

    m3
}

/// 计算峰度
fn kurtosis(data: &[f64]) -> f64 {
    if data.len() < 4 {
        return f64::NAN;
    }

    let mean_val = mean(data);
    let std_val = std_dev(data);

    if std_val == 0.0 || std_val.is_nan() {
        return f64::NAN;
    }

    let n = data.len() as f64;
    let m4 = data
        .iter()
        .map(|x| ((x - mean_val) / std_val).powi(4))
        .sum::<f64>()
        / n;

    m4 - 3.0 // 减去3得到超额峰度
}
