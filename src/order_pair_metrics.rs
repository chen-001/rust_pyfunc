use ndarray::{Array2, ArrayView2};
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;
use std::collections::{HashMap, HashSet};

/// 订单信息结构体
#[derive(Debug, Clone)]
struct OrderInfo {
    order_id: i64,
    is_buy: bool,
    total_turnover: f64,
    total_volume: f64,
    avg_price: f64,
    first_time: f64,
    last_time: f64,
    counterparty_orders: Vec<i64>,
    mean_time: f64,
    trade_count: usize,
    // 新增：用于复杂指标计算
    all_times: Vec<f64>,
    all_prices: Vec<f64>,
    all_volumes: Vec<f64>,
    all_flags: Vec<i32>,
}

/// 成交记录结构体（按时间排序存储）
#[derive(Debug, Clone)]
struct SortedTrades {
    times: Vec<f64>,
    prices: Vec<f64>,
    volumes: Vec<f64>,
    turnovers: Vec<f64>,
    flags: Vec<i32>,
    bid_orders: Vec<i64>,
    ask_orders: Vec<i64>,
}

impl SortedTrades {
    fn from_records(records: Vec<TradeRecord>) -> Self {
        let mut sorted = records;
        sorted.sort_by(|a, b| a.time.partial_cmp(&b.time).unwrap_or(std::cmp::Ordering::Equal));
        
        let n = sorted.len();
        let mut times = Vec::with_capacity(n);
        let mut prices = Vec::with_capacity(n);
        let mut volumes = Vec::with_capacity(n);
        let mut turnovers = Vec::with_capacity(n);
        let mut flags = Vec::with_capacity(n);
        let mut bid_orders = Vec::with_capacity(n);
        let mut ask_orders = Vec::with_capacity(n);
        
        for t in sorted {
            times.push(t.time);
            prices.push(t.price);
            volumes.push(t.volume);
            turnovers.push(t.turnover);
            flags.push(t.flag);
            bid_orders.push(t.bid_order);
            ask_orders.push(t.ask_order);
        }
        
        SortedTrades { times, prices, volumes, turnovers, flags, bid_orders, ask_orders }
    }
    
    /// 二分查找第一个>=target的索引
    fn lower_bound(&self, target: f64) -> usize {
        self.times.partition_point(|&t| t < target)
    }
    
    /// 二分查找第一个>target的索引
    fn upper_bound(&self, target: f64) -> usize {
        self.times.partition_point(|&t| t <= target)
    }
}

/// 成交记录结构体
#[derive(Debug, Clone)]
struct TradeRecord {
    time: f64,
    price: f64,
    volume: f64,
    turnover: f64,
    flag: i32,
    bid_order: i64,
    ask_order: i64,
}

/// 计算序列趋势（与[1,2,3,...,n]的相关系数）
fn calculate_trend(values: &[f64]) -> f64 {
    let n = values.len();
    if n < 2 {
        return 0.0;
    }
    
    let mean_x = (n + 1) as f64 / 2.0;
    let mean_y: f64 = values.iter().sum::<f64>() / n as f64;
    
    let mut cov = 0.0;
    let mut var_x = 0.0;
    let mut var_y = 0.0;
    
    for (i, &y) in values.iter().enumerate() {
        let x = (i + 1) as f64;
        cov += (x - mean_x) * (y - mean_y);
        var_x += (x - mean_x).powi(2);
        var_y += (y - mean_y).powi(2);
    }
    
    if var_x == 0.0 || var_y == 0.0 {
        return 0.0;
    }
    
    cov / (var_x.sqrt() * var_y.sqrt())
}

/// 使用最近配对原则为今天订单找昨天的配对订单
/// tolerance: 配对时的容忍比例，默认0.001表示±0.1%
fn find_pairs(today_orders: &[i64], yesterday_orders: &[i64], tolerance: f64) -> Vec<(i64, i64)> {
    let mut pairs = Vec::new();
    let yest_sorted: Vec<i64> = {
        let mut v = yesterday_orders.to_vec();
        v.sort();
        v
    };
    
    for &today_id in today_orders {
        let lower = (today_id as f64 * (1.0 - tolerance)) as i64;
        let upper = (today_id as f64 * (1.0 + tolerance)) as i64;
        
        let left = yest_sorted.partition_point(|&x| x < lower);
        let right = yest_sorted.partition_point(|&x| x <= upper);
        
        if left < right {
            let mut best_match = yest_sorted[left];
            let mut min_diff = (best_match - today_id).abs();
            
            for i in left + 1..right {
                let diff = (yest_sorted[i] - today_id).abs();
                if diff < min_diff {
                    min_diff = diff;
                    best_match = yest_sorted[i];
                }
            }
            pairs.push((today_id, best_match));
        }
    }
    
    pairs
}

/// 计算订单配对指标
#[pyfunction(signature = (today_trades, yesterday_trades, tolerance=0.001))]
pub fn calculate_order_pair_metrics(
    py: Python,
    today_trades: PyReadonlyArray2<f64>,
    yesterday_trades: PyReadonlyArray2<f64>,
    tolerance: f64,
) -> PyResult<(Py<PyArray2<f64>>, Vec<String>)> {
    let today_trades = today_trades.as_array();
    let yesterday_trades = yesterday_trades.as_array();
    
    // 解析成交记录并按时间排序
    let today_records = parse_trades(today_trades);
    let yesterday_records = parse_trades(yesterday_trades);
    
    let today_sorted = SortedTrades::from_records(today_records.clone());
    let yest_sorted = SortedTrades::from_records(yesterday_records.clone());
    
    // 获取开盘收盘时间
    let today_open = *today_sorted.times.first().unwrap();
    let today_close = *today_sorted.times.last().unwrap();
    let yest_open = *yest_sorted.times.first().unwrap();
    let yest_close = *yest_sorted.times.last().unwrap();
    
    // 计算日均价格
    let today_avg_price: f64 = today_sorted.prices.iter().sum::<f64>() / today_sorted.prices.len() as f64;
    let yest_avg_price: f64 = yest_sorted.prices.iter().sum::<f64>() / yest_sorted.prices.len() as f64;
    
    // 计算全天价格极差
    let today_price_min = today_sorted.prices.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let today_price_max = today_sorted.prices.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let today_price_range = today_price_max - today_price_min;
    let yest_price_min = yest_sorted.prices.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let yest_price_max = yest_sorted.prices.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let yest_price_range = yest_price_max - yest_price_min;
    
    // 聚合订单信息
    let today_orders = aggregate_orders(&today_records);
    let yest_orders = aggregate_orders(&yesterday_records);
    
    // 分离买单和卖单
    let today_buy_orders: Vec<i64> = today_orders.values().filter(|o| o.is_buy).map(|o| o.order_id).collect();
    let today_sell_orders: Vec<i64> = today_orders.values().filter(|o| !o.is_buy).map(|o| o.order_id).collect();
    let yest_buy_orders: Vec<i64> = yest_orders.values().filter(|o| o.is_buy).map(|o| o.order_id).collect();
    let yest_sell_orders: Vec<i64> = yest_orders.values().filter(|o| !o.is_buy).map(|o| o.order_id).collect();
    
    // 配对
    let buy_pairs = find_pairs(&today_buy_orders, &yest_buy_orders, tolerance);
    let sell_pairs = find_pairs(&today_sell_orders, &yest_sell_orders, tolerance);
    
    // 合并所有配对
    let all_pairs: Vec<(i64, i64)> = buy_pairs.into_iter()
        .chain(sell_pairs.into_iter())
        .collect();
    
    // 预计算影子订单（为每个订单找到同时活跃的其他订单）
    let today_shadow = compute_shadow_orders(&today_orders);
    let yest_shadow = compute_shadow_orders(&yest_orders);
    
    // 计算指标
    let mut results = Array2::<f64>::zeros((all_pairs.len(), 33));
    
    for (idx, &(today_id, yest_id)) in all_pairs.iter().enumerate() {
        let today_order = today_orders.get(&today_id).unwrap();
        let yest_order = yest_orders.get(&yest_id).unwrap();
        
        results[[idx, 0]] = today_id as f64;
        
        // 计算各指标
        let metrics = compute_metrics(
            today_order,
            yest_order,
            &today_sorted,
            &yest_sorted,
            today_open,
            today_close,
            yest_open,
            yest_close,
            today_avg_price,
            yest_avg_price,
            today_price_range,
            yest_price_range,
            &today_shadow,
            &yest_shadow,
        );
        
        // 不带绝对值的指标
        for (i, &v) in metrics.iter().enumerate() {
            results[[idx, i + 1]] = v;
        }
        
        // 带绝对值的指标
        for (i, &v) in metrics.iter().enumerate() {
            results[[idx, i + 17]] = v.abs();
        }
    }
    
    // 列名
    let column_names = vec![
        "today_order_id".to_string(),
        // 不带绝对值版本
        "obs_displacement".to_string(),
        "turnover_diff".to_string(),
        "signed_turnover_diff".to_string(),
        "price_ratio_diff".to_string(),
        "impact1_diff".to_string(),
        "impact2_diff".to_string(),
        "buy_ratio_stability".to_string(),
        "avg_trade_amount_diff".to_string(),
        "trade_duration_diff".to_string(),
        "shadow_overlap".to_string(),
        "price_radiation_diff".to_string(),
        "counterparty_order_diff".to_string(),
        "same_direction_trend_diff".to_string(),
        "opposite_direction_trend_diff".to_string(),
        "buy_order_trend_diff".to_string(),
        "sell_order_trend_diff".to_string(),
        // 带绝对值版本
        "abs_obs_displacement".to_string(),
        "abs_turnover_diff".to_string(),
        "abs_signed_turnover_diff".to_string(),
        "abs_price_ratio_diff".to_string(),
        "abs_impact1_diff".to_string(),
        "abs_impact2_diff".to_string(),
        "abs_buy_ratio_stability".to_string(),
        "abs_avg_trade_amount_diff".to_string(),
        "abs_trade_duration_diff".to_string(),
        "abs_shadow_overlap".to_string(),
        "abs_price_radiation_diff".to_string(),
        "abs_counterparty_order_diff".to_string(),
        "abs_same_direction_trend_diff".to_string(),
        "abs_opposite_direction_trend_diff".to_string(),
        "abs_buy_order_trend_diff".to_string(),
        "abs_sell_order_trend_diff".to_string(),
    ];
    
    Ok((results.into_pyarray(py).to_owned(), column_names))
}

/// 解析成交记录
fn parse_trades(trades: ArrayView2<f64>) -> Vec<TradeRecord> {
    let n = trades.nrows();
    let mut records = Vec::with_capacity(n);
    
    for i in 0..n {
        records.push(TradeRecord {
            time: trades[[i, 0]],
            price: trades[[i, 1]],
            volume: trades[[i, 2]],
            turnover: trades[[i, 3]],
            flag: trades[[i, 4]] as i32,
            bid_order: trades[[i, 5]] as i64,
            ask_order: trades[[i, 6]] as i64,
        });
    }
    
    records
}

/// 聚合订单信息
/// 只为主动方创建订单记录（一个订单编号只对应一个 OrderInfo）
fn aggregate_orders(records: &[TradeRecord]) -> HashMap<i64, OrderInfo> {
    let mut orders: HashMap<i64, OrderInfo> = HashMap::new();
    
    for trade in records {
        // 只为主动方创建订单记录：
        // flag=66: 主买，bid_order 是主动方
        // flag=83: 主卖，ask_order 是主动方
        let (active_order_id, is_buy) = if trade.flag == 66 {
            (trade.bid_order, true)
        } else {
            (trade.ask_order, false)
        };
        
        let counterparty_order = if trade.flag == 66 {
            trade.ask_order
        } else {
            trade.bid_order
        };
        
        let entry = orders.entry(active_order_id).or_insert_with(|| OrderInfo {
            order_id: active_order_id,
            is_buy,
            total_turnover: 0.0,
            total_volume: 0.0,
            avg_price: 0.0,
            first_time: trade.time,
            last_time: trade.time,
            counterparty_orders: Vec::new(),
            mean_time: 0.0,
            trade_count: 0,
            all_times: Vec::new(),
            all_prices: Vec::new(),
            all_volumes: Vec::new(),
            all_flags: Vec::new(),
        });
        
        entry.total_turnover += trade.turnover;
        entry.total_volume += trade.volume;
        entry.last_time = entry.last_time.max(trade.time);
        entry.first_time = entry.first_time.min(trade.time);
        entry.counterparty_orders.push(counterparty_order);
        entry.trade_count += 1;
        entry.all_times.push(trade.time);
        entry.all_prices.push(trade.price);
        entry.all_volumes.push(trade.volume);
        entry.all_flags.push(trade.flag);
    }
    
    // 计算平均价格和平均时间
    for order in orders.values_mut() {
        order.avg_price = order.total_turnover / order.total_volume;
        order.mean_time = (order.first_time + order.last_time) / 2.0;
    }
    
    orders
}

/// 预计算每个订单的影子订单（同时活跃的其他订单）
fn compute_shadow_orders(orders: &HashMap<i64, OrderInfo>) -> HashMap<i64, HashSet<i64>> {
    let mut shadows: HashMap<i64, HashSet<i64>> = HashMap::new();
    
    if orders.is_empty() {
        return shadows;
    }
    
    // 将订单按时间排序（时间相同时按order_id排序，确保顺序确定）
    let mut order_list: Vec<&OrderInfo> = orders.values().collect();
    order_list.sort_by(|a, b| {
        match a.first_time.partial_cmp(&b.first_time) {
            Some(std::cmp::Ordering::Equal) => a.order_id.cmp(&b.order_id),
            Some(ord) => ord,
            None => a.order_id.cmp(&b.order_id),
        }
    });
    
    let n = order_list.len();
    
    // 使用滑动窗口找同时活跃的订单
    for i in 0..n {
        let order = order_list[i];
        let mut shadow = HashSet::new();
        
        // 向前查找
        for j in (0..i).rev() {
            let other = order_list[j];
            if other.last_time < order.first_time {
                break;
            }
            shadow.insert(other.order_id);
        }
        
        // 向后查找
        for j in (i + 1)..n {
            let other = order_list[j];
            if other.first_time > order.last_time {
                break;
            }
            shadow.insert(other.order_id);
        }
        
        shadows.insert(order.order_id, shadow);
    }
    
    shadows
}

/// 计算单个配对的所有指标
#[allow(clippy::too_many_arguments)]
fn compute_metrics(
    today_order: &OrderInfo,
    yest_order: &OrderInfo,
    today_sorted: &SortedTrades,
    yest_sorted: &SortedTrades,
    today_open: f64,
    today_close: f64,
    yest_open: f64,
    yest_close: f64,
    today_avg_price: f64,
    yest_avg_price: f64,
    today_price_range: f64,
    yest_price_range: f64,
    today_shadow: &HashMap<i64, HashSet<i64>>,
    yest_shadow: &HashMap<i64, HashSet<i64>>,
) -> [f64; 16] {
    let mut metrics = [0.0f64; 16];
    
    // 1. 开盘偏置位移
    let today_obs = (today_order.mean_time - today_open) / (today_close - today_open);
    let yest_obs = (yest_order.mean_time - yest_open) / (yest_close - yest_open);
    metrics[0] = yest_obs - today_obs;
    
    // 2. 订单体量差
    metrics[1] = yest_order.total_turnover - today_order.total_turnover;
    
    // 3. 带方向的订单体量差
    let yest_signed = if yest_order.is_buy { yest_order.total_turnover } else { -yest_order.total_turnover };
    let today_signed = if today_order.is_buy { today_order.total_turnover } else { -today_order.total_turnover };
    metrics[2] = yest_signed - today_signed;
    
    // 4. 成交价格差
    metrics[3] = yest_order.avg_price / yest_avg_price - today_order.avg_price / today_avg_price;
    
    // 5. 冲击1 - 之后10秒内最极端的价格变动
    let today_impact1 = calculate_impact1(today_order, today_sorted, today_order.avg_price);
    let yest_impact1 = calculate_impact1(yest_order, yest_sorted, yest_order.avg_price);
    metrics[4] = yest_impact1 - today_impact1;
    
    // 6. 冲击2 - 之后10秒的价格变动
    let today_impact2 = calculate_impact2(today_order, today_sorted, today_order.avg_price);
    let yest_impact2 = calculate_impact2(yest_order, yest_sorted, yest_order.avg_price);
    metrics[5] = yest_impact2 - today_impact2;
    
    // 7. 主买占比稳定性
    let today_buy_ratio = calculate_buy_ratio_before(today_order, today_sorted);
    let yest_buy_ratio = calculate_buy_ratio_before(yest_order, yest_sorted);
    metrics[6] = (yest_buy_ratio - today_buy_ratio).abs();
    
    // 8. 金额之差
    let yest_avg_trade = yest_order.total_turnover / yest_order.trade_count as f64;
    let today_avg_trade = today_order.total_turnover / today_order.trade_count as f64;
    metrics[7] = yest_avg_trade - today_avg_trade;
    
    // 9. 成交时长
    metrics[8] = (yest_order.last_time - yest_order.first_time) - (today_order.last_time - today_order.first_time);
    
    // 10. 影子订单重叠度
    metrics[9] = calculate_shadow_overlap(today_order.order_id, yest_order.order_id, today_shadow, yest_shadow);
    
    // 11. 价格辐射范围
    let today_prr = calculate_price_radiation(today_order, today_sorted, today_price_range);
    let yest_prr = calculate_price_radiation(yest_order, yest_sorted, yest_price_range);
    metrics[10] = yest_prr - today_prr;
    
    // 12. 对手方订单编号均值
    let yest_counter_mean = yest_order.counterparty_orders.iter().sum::<i64>() as f64 
        / yest_order.counterparty_orders.len() as f64;
    let today_counter_mean = today_order.counterparty_orders.iter().sum::<i64>() as f64 
        / today_order.counterparty_orders.len() as f64;
    metrics[11] = yest_counter_mean - today_counter_mean;
    
    // 13-16. 各类订单编号趋势
    let today_same = calculate_order_trend(today_order, today_sorted, today_order.is_buy);
    let yest_same = calculate_order_trend(yest_order, yest_sorted, yest_order.is_buy);
    metrics[12] = yest_same - today_same;
    
    let today_opposite = calculate_order_trend(today_order, today_sorted, !today_order.is_buy);
    let yest_opposite = calculate_order_trend(yest_order, yest_sorted, !yest_order.is_buy);
    metrics[13] = yest_opposite - today_opposite;
    
    let today_buy = calculate_order_trend(today_order, today_sorted, true);
    let yest_buy = calculate_order_trend(yest_order, yest_sorted, true);
    metrics[14] = yest_buy - today_buy;
    
    let today_sell = calculate_order_trend(today_order, today_sorted, false);
    let yest_sell = calculate_order_trend(yest_order, yest_sorted, false);
    metrics[15] = yest_sell - today_sell;
    
    metrics
}

/// 计算冲击1 - 之后10秒内最极端的价格变动
fn calculate_impact1(order: &OrderInfo, sorted: &SortedTrades, avg_price: f64) -> f64 {
    let start_idx = sorted.upper_bound(order.last_time);
    let end_idx = sorted.upper_bound(order.last_time + 10.0);
    
    if start_idx >= end_idx {
        return 0.0;
    }
    
    let mut max_deviation = 0.0f64;
    let mut last_price = 0.0f64;
    
    for i in start_idx..end_idx {
        let deviation = ((sorted.prices[i] - avg_price) / avg_price).abs();
        if deviation > max_deviation {
            max_deviation = deviation;
        }
        last_price = sorted.prices[i];
    }
    
    let sign = if last_price > avg_price { 1.0 } else { -1.0 };
    sign * max_deviation
}

/// 计算冲击2 - 之后10秒末的价格变动
fn calculate_impact2(order: &OrderInfo, sorted: &SortedTrades, avg_price: f64) -> f64 {
    let target_time = order.last_time + 10.0;
    let start_idx = sorted.upper_bound(order.last_time);
    
    if start_idx >= sorted.times.len() {
        return 0.0;
    }
    
    // 找最接近10秒后的价格
    let mut best_idx = start_idx;
    let mut min_diff = f64::INFINITY;
    
    for i in start_idx..sorted.times.len() {
        let diff = (sorted.times[i] - target_time).abs();
        if diff < min_diff {
            min_diff = diff;
            best_idx = i;
        }
        if sorted.times[i] > target_time + 10.0 {
            break;
        }
    }
    
    (sorted.prices[best_idx] - avg_price) / avg_price
}

/// 计算订单之前10秒内的主买占比
fn calculate_buy_ratio_before(order: &OrderInfo, sorted: &SortedTrades) -> f64 {
    let end_idx = sorted.lower_bound(order.first_time);
    let start_idx = sorted.lower_bound(order.first_time - 10.0);
    
    if start_idx >= end_idx {
        return 0.5;
    }
    
    let mut buy_turnover = 0.0f64;
    let mut total_turnover = 0.0f64;
    
    for i in start_idx..end_idx {
        total_turnover += sorted.turnovers[i];
        if sorted.flags[i] == 66 {
            buy_turnover += sorted.turnovers[i];
        }
    }
    
    if total_turnover > 0.0 {
        buy_turnover / total_turnover
    } else {
        0.5
    }
}

/// 计算影子订单重叠度（Jaccard相似度）
fn calculate_shadow_overlap(
    today_id: i64,
    yest_id: i64,
    today_shadow: &HashMap<i64, HashSet<i64>>,
    yest_shadow: &HashMap<i64, HashSet<i64>>,
) -> f64 {
    let today_neighbors = today_shadow.get(&today_id).unwrap();
    let yest_neighbors = yest_shadow.get(&yest_id).unwrap();
    
    let intersection = today_neighbors.intersection(yest_neighbors).count();
    let union = today_neighbors.union(yest_neighbors).count();
    
    if union == 0 {
        0.0
    } else {
        intersection as f64 / union as f64
    }
}

/// 计算价格辐射范围
fn calculate_price_radiation(order: &OrderInfo, sorted: &SortedTrades, price_range: f64) -> f64 {
    if price_range == 0.0 {
        return 0.0;
    }
    
    let start_idx = sorted.upper_bound(order.last_time);
    let end_idx = sorted.upper_bound(order.last_time + 10.0);
    
    if start_idx >= end_idx {
        return 0.0;
    }
    
    let mut max_price = f64::NEG_INFINITY;
    let mut min_price = f64::INFINITY;
    
    for i in start_idx..end_idx {
        max_price = max_price.max(sorted.prices[i]);
        min_price = min_price.min(sorted.prices[i]);
    }
    
    (max_price - min_price) / price_range
}

/// 计算订单编号趋势（is_buy: true=买单，false=卖单）
fn calculate_order_trend(order: &OrderInfo, sorted: &SortedTrades, is_buy: bool) -> f64 {
    let target_flag = if is_buy { 66 } else { 83 };
    
    let start_idx = sorted.upper_bound(order.last_time);
    let end_idx = sorted.upper_bound(order.last_time + 10.0);
    
    let mut order_ids: Vec<f64> = Vec::new();
    
    for i in start_idx..end_idx {
        if sorted.flags[i] == target_flag {
            let order_id = if is_buy { sorted.bid_orders[i] } else { sorted.ask_orders[i] };
            order_ids.push(order_id as f64);
        }
    }
    
    calculate_trend(&order_ids)
}

// ============== 新增：复杂指标计算函数 ==============
// 这些指标基于市场全局数据计算，反映订单成交时的市场状态

/// 1. 计算LZ复杂度（归一化）- 基于市场全局数据
/// 用该订单第一笔成交前N笔市场成交的价格变化序列计算
fn calculate_lz_complexity_market(sorted: &SortedTrades, order: &OrderInfo, window_size: usize) -> f64 {
    // 找到订单第一笔成交前的市场成交
    let end_idx = sorted.lower_bound(order.first_time);
    if end_idx == 0 {
        return 0.0;
    }
    
    let start_idx = end_idx.saturating_sub(window_size);
    if end_idx - start_idx < 3 {
        return 0.0;
    }
    
    // 提取价格序列
    let prices = &sorted.prices[start_idx..end_idx];
    let volumes = &sorted.volumes[start_idx..end_idx];
    let n = prices.len();
    
    // 编码为符号序列：(价格方向, 成交量方向) -> 4种符号
    let mut symbols: Vec<u8> = Vec::with_capacity(n - 1);
    for i in 1..n {
        let price_dir = if prices[i] >= prices[i - 1] { 1 } else { 0 };
        let vol_dir = if volumes[i] >= volumes[i - 1] { 1 } else { 0 };
        let symbol = (price_dir << 1) | vol_dir;
        symbols.push(symbol);
    }
    
    // LZ78复杂度计算
    let mut dict: HashSet<Vec<u8>> = HashSet::new();
    let mut w: Vec<u8> = Vec::new();
    let mut c = 0usize;
    
    for &s in &symbols {
        let mut new_w = w.clone();
        new_w.push(s);
        if dict.contains(&new_w) {
            w = new_w;
        } else {
            dict.insert(new_w);
            w.clear();
            c += 1;
        }
    }
    if !w.is_empty() {
        c += 1;
    }
    
    let n_symbols = symbols.len();
    if n_symbols == 0 {
        return 0.0;
    }
    let n_f64 = n_symbols as f64;
    c as f64 / (n_f64 / n_f64.log2().max(1.0))
}

/// 2. 计算分形维数（盒计数法）- 基于市场全局数据
/// 用该订单第一笔成交前N笔市场成交的价格-时间轨迹
fn calculate_fractal_dimension_market(sorted: &SortedTrades, order: &OrderInfo, window_size: usize) -> f64 {
    let end_idx = sorted.lower_bound(order.first_time);
    if end_idx == 0 {
        return 0.0;
    }
    
    let start_idx = end_idx.saturating_sub(window_size);
    if end_idx - start_idx < 5 {
        return 0.0;
    }
    
    let times = &sorted.times[start_idx..end_idx];
    let prices = &sorted.prices[start_idx..end_idx];
    
    // 归一化到单位正方形
    let t_min = times.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let t_max = times.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let p_min = prices.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let p_max = prices.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    
    let t_range = (t_max - t_min).max(1e-9);
    let p_range = (p_max - p_min).max(1e-9);
    
    let points: Vec<(f64, f64)> = times.iter()
        .zip(prices.iter())
        .map(|(&t, &p)| ((t - t_min) / t_range, (p - p_min) / p_range))
        .collect();
    
    // 盒计数法
    let mut counts: Vec<(f64, f64)> = Vec::new();
    
    for level in 2..=8 {
        let eps = 1.0 / level as f64;
        let mut boxes: HashSet<(i32, i32)> = HashSet::new();
        
        for (x, y) in &points {
            let bx = (x / eps).floor() as i32;
            let by = (y / eps).floor() as i32;
            boxes.insert((bx, by));
        }
        
        let n_boxes = boxes.len() as f64;
        if n_boxes > 0.0 {
            counts.push((eps.recip().ln(), n_boxes.ln()));
        }
    }
    
    if counts.len() < 2 {
        return 0.0;
    }
    
    // 线性拟合求斜率
    let n = counts.len() as f64;
    let sum_x: f64 = counts.iter().map(|(x, _)| x).sum();
    let sum_y: f64 = counts.iter().map(|(_, y)| y).sum();
    let sum_xy: f64 = counts.iter().map(|(x, y)| x * y).sum();
    let sum_xx: f64 = counts.iter().map(|(x, _)| x * x).sum();
    
    let denom = n * sum_xx - sum_x * sum_x;
    if denom.abs() < 1e-12 {
        return 0.0;
    }
    
    (n * sum_xy - sum_x * sum_y) / denom
}

/// 3. 计算Hurst指数（R/S分析）- 基于市场全局数据
/// 用该订单成交后10秒的市场价格序列
fn calculate_hurst_exponent_market(sorted: &SortedTrades, order: &OrderInfo) -> f64 {
    let start_idx = sorted.upper_bound(order.last_time);
    let end_idx = sorted.upper_bound(order.last_time + 10.0);
    
    if end_idx - start_idx < 10 {
        return 0.5;
    }
    
    let prices: Vec<f64> = sorted.prices[start_idx..end_idx].to_vec();
    let n = prices.len();
    
    let returns: Vec<f64> = (1..n)
        .map(|i| (prices[i] / prices[i - 1]).ln())
        .collect();
    
    if returns.is_empty() {
        return 0.5;
    }
    
    let mean_ret: f64 = returns.iter().sum::<f64>() / returns.len() as f64;
    
    let mut cum_dev: Vec<f64> = Vec::with_capacity(returns.len());
    let mut cumsum = 0.0;
    for &r in &returns {
        cumsum += r - mean_ret;
        cum_dev.push(cumsum);
    }
    
    let r_val = cum_dev.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b))
               - cum_dev.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    
    let var: f64 = returns.iter().map(|r| (r - mean_ret).powi(2)).sum::<f64>() / returns.len() as f64;
    let s_val = var.sqrt();
    
    if s_val < 1e-12 {
        return 0.5;
    }
    
    let rs = r_val / s_val;
    let h = rs.ln() / (returns.len() as f64 / 2.0).ln().max(1e-9);
    
    h.clamp(0.0, 1.0)
}

/// 4. 计算捕食压力 - 基于市场全局数据
/// 该订单成交前N笔市场中主动买/卖的比例
fn calculate_predation_pressure_market(sorted: &SortedTrades, order: &OrderInfo, window_size: usize) -> f64 {
    let end_idx = sorted.lower_bound(order.first_time);
    if end_idx == 0 {
        return 0.5;
    }
    
    let start_idx = end_idx.saturating_sub(window_size);
    if end_idx - start_idx < 3 {
        return 0.5;
    }
    
    // 统计主动买(flag=66)的成交金额
    let mut buy_turnover = 0.0f64;
    let mut total_turnover = 0.0f64;
    
    for i in start_idx..end_idx {
        total_turnover += sorted.turnovers[i];
        if sorted.flags[i] == 66 {
            buy_turnover += sorted.turnovers[i];
        }
    }
    
    if total_turnover < 1e-9 {
        return 0.5;
    }
    
    buy_turnover / total_turnover
}

/// 5. 计算市场成交间隔变异系数 - 基于市场全局数据
/// 该订单成交前N笔市场成交的时间间隔CV
fn calculate_interval_cv_market(sorted: &SortedTrades, order: &OrderInfo, window_size: usize) -> f64 {
    let end_idx = sorted.lower_bound(order.first_time);
    if end_idx == 0 {
        return 0.0;
    }
    
    let start_idx = end_idx.saturating_sub(window_size);
    if end_idx - start_idx < 3 {
        return 0.0;
    }
    
    let times = &sorted.times[start_idx..end_idx];
    let n = times.len();
    
    // 计算时间间隔
    let intervals: Vec<f64> = (1..n)
        .map(|i| times[i] - times[i - 1])
        .collect();
    
    if intervals.is_empty() {
        return 0.0;
    }
    
    let mean: f64 = intervals.iter().sum::<f64>() / intervals.len() as f64;
    
    if mean < 1e-9 {
        return 0.0;
    }
    
    let var: f64 = intervals.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / intervals.len() as f64;
    let std = var.sqrt();
    
    std / mean
}

/// 计算更多指标的订单配对函数
#[pyfunction(signature = (today_trades, yesterday_trades, tolerance=0.001))]
pub fn calculate_order_pair_metrics_more(
    py: Python,
    today_trades: PyReadonlyArray2<f64>,
    yesterday_trades: PyReadonlyArray2<f64>,
    tolerance: f64,
) -> PyResult<(Py<PyArray2<f64>>, Vec<String>)> {
    let today_trades = today_trades.as_array();
    let yesterday_trades = yesterday_trades.as_array();
    
    // 解析成交记录并按时间排序
    let today_records = parse_trades(today_trades);
    let yesterday_records = parse_trades(yesterday_trades);
    
    let today_sorted = SortedTrades::from_records(today_records.clone());
    let yest_sorted = SortedTrades::from_records(yesterday_records.clone());
    
    // 获取开盘收盘时间
    let today_open = *today_sorted.times.first().unwrap();
    let today_close = *today_sorted.times.last().unwrap();
    let yest_open = *yest_sorted.times.first().unwrap();
    let yest_close = *yest_sorted.times.last().unwrap();
    
    // 计算日均价格
    let today_avg_price: f64 = today_sorted.prices.iter().sum::<f64>() / today_sorted.prices.len() as f64;
    let yest_avg_price: f64 = yest_sorted.prices.iter().sum::<f64>() / yest_sorted.prices.len() as f64;
    
    // 计算全天价格极差
    let today_price_min = today_sorted.prices.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let today_price_max = today_sorted.prices.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let today_price_range = today_price_max - today_price_min;
    let yest_price_min = yest_sorted.prices.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let yest_price_max = yest_sorted.prices.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let yest_price_range = yest_price_max - yest_price_min;
    
    // 聚合订单信息
    let today_orders = aggregate_orders(&today_records);
    let yest_orders = aggregate_orders(&yesterday_records);
    
    // 分离买单和卖单
    let today_buy_orders: Vec<i64> = today_orders.values().filter(|o| o.is_buy).map(|o| o.order_id).collect();
    let today_sell_orders: Vec<i64> = today_orders.values().filter(|o| !o.is_buy).map(|o| o.order_id).collect();
    let yest_buy_orders: Vec<i64> = yest_orders.values().filter(|o| o.is_buy).map(|o| o.order_id).collect();
    let yest_sell_orders: Vec<i64> = yest_orders.values().filter(|o| !o.is_buy).map(|o| o.order_id).collect();
    
    // 配对
    let buy_pairs = find_pairs(&today_buy_orders, &yest_buy_orders, tolerance);
    let sell_pairs = find_pairs(&today_sell_orders, &yest_sell_orders, tolerance);
    
    // 合并所有配对
    let all_pairs: Vec<(i64, i64)> = buy_pairs.into_iter()
        .chain(sell_pairs.into_iter())
        .collect();
    
    // 预计算影子订单
    let today_shadow = compute_shadow_orders(&today_orders);
    let yest_shadow = compute_shadow_orders(&yest_orders);
    
    // 计算指标（原有16个 + 新增5个 = 21个指标）
    let mut results = Array2::<f64>::zeros((all_pairs.len(), 44)); // 1 + 21*2 + 1
    
    // 复杂指标的计算窗口大小（用成交前N笔市场成交）
    let window_size = 100usize;
    
    for (idx, &(today_id, yest_id)) in all_pairs.iter().enumerate() {
        let today_order = today_orders.get(&today_id).unwrap();
        let yest_order = yest_orders.get(&yest_id).unwrap();
        
        results[[idx, 0]] = today_id as f64;
        
        // 计算原有16个指标
        let metrics = compute_metrics(
            today_order,
            yest_order,
            &today_sorted,
            &yest_sorted,
            today_open,
            today_close,
            yest_open,
            yest_close,
            today_avg_price,
            yest_avg_price,
            today_price_range,
            yest_price_range,
            &today_shadow,
            &yest_shadow,
        );
        
        // 不带绝对值的原有指标 (列1-16)
        for (i, &v) in metrics.iter().enumerate() {
            results[[idx, i + 1]] = v;
        }
        
        // 带绝对值的原有指标 (列17-32)
        for (i, &v) in metrics.iter().enumerate() {
            results[[idx, i + 17]] = v.abs();
        }
        
        // 新增5个复杂指标 - 基于市场全局数据
        // 用订单成交前100笔市场成交计算
        let today_lz = calculate_lz_complexity_market(&today_sorted, today_order, window_size);
        let yest_lz = calculate_lz_complexity_market(&yest_sorted, yest_order, window_size);
        results[[idx, 33]] = yest_lz - today_lz;
        
        let today_fd = calculate_fractal_dimension_market(&today_sorted, today_order, window_size);
        let yest_fd = calculate_fractal_dimension_market(&yest_sorted, yest_order, window_size);
        results[[idx, 34]] = yest_fd - today_fd;
        
        let today_h = calculate_hurst_exponent_market(&today_sorted, today_order);
        let yest_h = calculate_hurst_exponent_market(&yest_sorted, yest_order);
        results[[idx, 35]] = yest_h - today_h;
        
        let today_pe = calculate_predation_pressure_market(&today_sorted, today_order, window_size);
        let yest_pe = calculate_predation_pressure_market(&yest_sorted, yest_order, window_size);
        results[[idx, 36]] = yest_pe - today_pe;
        
        let today_cv = calculate_interval_cv_market(&today_sorted, today_order, window_size);
        let yest_cv = calculate_interval_cv_market(&yest_sorted, yest_order, window_size);
        results[[idx, 37]] = yest_cv - today_cv;
        
        // 带绝对值的新增指标 (列38-42)
        results[[idx, 38]] = (yest_lz - today_lz).abs();
        results[[idx, 39]] = (yest_fd - today_fd).abs();
        results[[idx, 40]] = (yest_h - today_h).abs();
        results[[idx, 41]] = (yest_pe - today_pe).abs();
        results[[idx, 42]] = (yest_cv - today_cv).abs();
        
        // 第43列：订单成交笔数
        results[[idx, 43]] = today_order.trade_count as f64;
    }
    
    // 列名
    let column_names = vec![
        "today_order_id".to_string(),
        // 原有指标（不带绝对值）
        "obs_displacement".to_string(),
        "turnover_diff".to_string(),
        "signed_turnover_diff".to_string(),
        "price_ratio_diff".to_string(),
        "impact1_diff".to_string(),
        "impact2_diff".to_string(),
        "buy_ratio_stability".to_string(),
        "avg_trade_amount_diff".to_string(),
        "trade_duration_diff".to_string(),
        "shadow_overlap".to_string(),
        "price_radiation_diff".to_string(),
        "counterparty_order_diff".to_string(),
        "same_direction_trend_diff".to_string(),
        "opposite_direction_trend_diff".to_string(),
        "buy_order_trend_diff".to_string(),
        "sell_order_trend_diff".to_string(),
        // 原有指标（带绝对值）
        "abs_obs_displacement".to_string(),
        "abs_turnover_diff".to_string(),
        "abs_signed_turnover_diff".to_string(),
        "abs_price_ratio_diff".to_string(),
        "abs_impact1_diff".to_string(),
        "abs_impact2_diff".to_string(),
        "abs_buy_ratio_stability".to_string(),
        "abs_avg_trade_amount_diff".to_string(),
        "abs_trade_duration_diff".to_string(),
        "abs_shadow_overlap".to_string(),
        "abs_price_radiation_diff".to_string(),
        "abs_counterparty_order_diff".to_string(),
        "abs_same_direction_trend_diff".to_string(),
        "abs_opposite_direction_trend_diff".to_string(),
        "abs_buy_order_trend_diff".to_string(),
        "abs_sell_order_trend_diff".to_string(),
        // 新增复杂指标（不带绝对值）
        // 基于市场全局数据计算，反映订单成交时的市场状态
        "market_lz_complexity_diff".to_string(),
        "market_fractal_dim_diff".to_string(),
        "market_hurst_exp_diff".to_string(),
        "market_predation_pressure_diff".to_string(),
        "market_interval_cv_diff".to_string(),
        // 新增复杂指标（带绝对值）
        "abs_market_lz_complexity_diff".to_string(),
        "abs_market_fractal_dim_diff".to_string(),
        "abs_market_hurst_exp_diff".to_string(),
        "abs_market_predation_pressure_diff".to_string(),
        "abs_market_interval_cv_diff".to_string(),
        // 附加信息
        "trade_count".to_string(),
    ];
    
    Ok((results.into_pyarray(py).to_owned(), column_names))
}

// ============== V2版本：不区分买卖方向的配对 ==============

/// 聚合订单信息（V2版本）
/// 保留所有订单编号（bid_order 和 ask_order 都保留）
fn aggregate_orders_v2(records: &[TradeRecord]) -> HashMap<i64, OrderInfo> {
    let mut orders: HashMap<i64, OrderInfo> = HashMap::new();
    
    for trade in records {
        // 为 bid_order 创建或更新订单信息
        let bid_entry = orders.entry(trade.bid_order).or_insert_with(|| OrderInfo {
            order_id: trade.bid_order,
            is_buy: true, // 默认为买单（可能被后续成交更新）
            total_turnover: 0.0,
            total_volume: 0.0,
            avg_price: 0.0,
            first_time: trade.time,
            last_time: trade.time,
            counterparty_orders: Vec::new(),
            mean_time: 0.0,
            trade_count: 0,
            all_times: Vec::new(),
            all_prices: Vec::new(),
            all_volumes: Vec::new(),
            all_flags: Vec::new(),
        });
        
        bid_entry.total_turnover += trade.turnover;
        bid_entry.total_volume += trade.volume;
        bid_entry.last_time = bid_entry.last_time.max(trade.time);
        bid_entry.first_time = bid_entry.first_time.min(trade.time);
        bid_entry.counterparty_orders.push(trade.ask_order);
        bid_entry.trade_count += 1;
        bid_entry.all_times.push(trade.time);
        bid_entry.all_prices.push(trade.price);
        bid_entry.all_volumes.push(trade.volume);
        bid_entry.all_flags.push(trade.flag);
        
        // 为 ask_order 创建或更新订单信息
        let ask_entry = orders.entry(trade.ask_order).or_insert_with(|| OrderInfo {
            order_id: trade.ask_order,
            is_buy: false, // 默认为卖单（可能被后续成交更新）
            total_turnover: 0.0,
            total_volume: 0.0,
            avg_price: 0.0,
            first_time: trade.time,
            last_time: trade.time,
            counterparty_orders: Vec::new(),
            mean_time: 0.0,
            trade_count: 0,
            all_times: Vec::new(),
            all_prices: Vec::new(),
            all_volumes: Vec::new(),
            all_flags: Vec::new(),
        });
        
        ask_entry.total_turnover += trade.turnover;
        ask_entry.total_volume += trade.volume;
        ask_entry.last_time = ask_entry.last_time.max(trade.time);
        ask_entry.first_time = ask_entry.first_time.min(trade.time);
        ask_entry.counterparty_orders.push(trade.bid_order);
        ask_entry.trade_count += 1;
        ask_entry.all_times.push(trade.time);
        ask_entry.all_prices.push(trade.price);
        ask_entry.all_volumes.push(trade.volume);
        ask_entry.all_flags.push(trade.flag);
    }
    
    // 计算平均价格和平均时间，并确定订单方向
    for order in orders.values_mut() {
        order.avg_price = order.total_turnover / order.total_volume;
        order.mean_time = (order.first_time + order.last_time) / 2.0;
        
        // 根据flag统计确定订单方向
        let flag66_turnover: f64 = order.all_flags.iter()
            .zip(order.all_volumes.iter())
            .zip(order.all_prices.iter())
            .filter(|((&flag, _), _)| flag == 66)
            .map(|((_, &vol), &price)| vol * price)
            .sum();
        
        let flag83_turnover: f64 = order.all_flags.iter()
            .zip(order.all_volumes.iter())
            .zip(order.all_prices.iter())
            .filter(|((&flag, _), _)| flag == 83)
            .map(|((_, &vol), &price)| vol * price)
            .sum();
        
        // 如果flag=66的金额更多，说明主要作为买单
        order.is_buy = flag66_turnover >= flag83_turnover;
    }
    
    orders
}

/// 使用最近配对原则为今天订单找昨天的配对订单（V2版本）
/// 不区分买卖方向，仅根据编号最近配对
/// tolerance: 配对时的容忍比例，默认0.001表示±0.1%
fn find_pairs_v2(today_orders: &[i64], yesterday_orders: &[i64], tolerance: f64) -> Vec<(i64, i64)> {
    let mut pairs = Vec::new();
    let yest_sorted: Vec<i64> = {
        let mut v = yesterday_orders.to_vec();
        v.sort();
        v
    };
    
    for &today_id in today_orders {
        let lower = (today_id as f64 * (1.0 - tolerance)) as i64;
        let upper = (today_id as f64 * (1.0 + tolerance)) as i64;
        
        let left = yest_sorted.partition_point(|&x| x < lower);
        let right = yest_sorted.partition_point(|&x| x <= upper);
        
        if left < right {
            let mut best_match = yest_sorted[left];
            let mut min_diff = (best_match - today_id).abs();
            
            for i in left + 1..right {
                let diff = (yest_sorted[i] - today_id).abs();
                if diff < min_diff {
                    min_diff = diff;
                    best_match = yest_sorted[i];
                }
            }
            pairs.push((today_id, best_match));
        }
    }
    
    pairs
}

/// 计算订单配对指标（V2版本：不区分买卖方向）
#[pyfunction(signature = (today_trades, yesterday_trades, tolerance=0.001))]
pub fn calculate_order_pair_metrics_more_v2(
    py: Python,
    today_trades: PyReadonlyArray2<f64>,
    yesterday_trades: PyReadonlyArray2<f64>,
    tolerance: f64,
) -> PyResult<(Py<PyArray2<f64>>, Vec<String>)> {
    let today_trades = today_trades.as_array();
    let yesterday_trades = yesterday_trades.as_array();
    
    // 解析成交记录
    let today_records = parse_trades(today_trades);
    let yesterday_records = parse_trades(yesterday_trades);
    
    let today_sorted = SortedTrades::from_records(today_records.clone());
    let yest_sorted = SortedTrades::from_records(yesterday_records.clone());
    
    // 获取开盘收盘时间
    let today_open = *today_sorted.times.first().unwrap();
    let today_close = *today_sorted.times.last().unwrap();
    let yest_open = *yest_sorted.times.first().unwrap();
    let yest_close = *yest_sorted.times.last().unwrap();
    
    // 计算日均价格
    let today_avg_price: f64 = today_sorted.prices.iter().sum::<f64>() / today_sorted.prices.len() as f64;
    let yest_avg_price: f64 = yest_sorted.prices.iter().sum::<f64>() / yest_sorted.prices.len() as f64;
    
    // 计算全天价格极差
    let today_price_min = today_sorted.prices.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let today_price_max = today_sorted.prices.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let today_price_range = today_price_max - today_price_min;
    let yest_price_min = yest_sorted.prices.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let yest_price_max = yest_sorted.prices.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let yest_price_range = yest_price_max - yest_price_min;
    
    // 聚合订单信息（V2版本：保留所有订单编号）
    let today_orders = aggregate_orders_v2(&today_records);
    let yest_orders = aggregate_orders_v2(&yesterday_records);
    
    // 获取所有订单编号并排序（确保顺序确定）
    let mut today_all_orders: Vec<i64> = today_orders.keys().cloned().collect();
    let mut yest_all_orders: Vec<i64> = yest_orders.keys().cloned().collect();
    today_all_orders.sort();
    yest_all_orders.sort();
    
    // 配对（不区分买卖方向）
    let all_pairs = find_pairs_v2(&today_all_orders, &yest_all_orders, tolerance);
    
    // 预计算影子订单
    let today_shadow = compute_shadow_orders(&today_orders);
    let yest_shadow = compute_shadow_orders(&yest_orders);
    
    // 计算指标
    let mut results = Array2::<f64>::zeros((all_pairs.len(), 44));
    let window_size = 100usize;
    
    for (idx, &(today_id, yest_id)) in all_pairs.iter().enumerate() {
        let today_order = today_orders.get(&today_id).unwrap();
        let yest_order = yest_orders.get(&yest_id).unwrap();
        
        results[[idx, 0]] = today_id as f64;
        
        // 计算原有16个指标
        let metrics = compute_metrics(
            today_order,
            yest_order,
            &today_sorted,
            &yest_sorted,
            today_open,
            today_close,
            yest_open,
            yest_close,
            today_avg_price,
            yest_avg_price,
            today_price_range,
            yest_price_range,
            &today_shadow,
            &yest_shadow,
        );
        
        // 不带绝对值的原有指标 (列1-16)
        for (i, &v) in metrics.iter().enumerate() {
            results[[idx, i + 1]] = v;
        }
        
        // 带绝对值的原有指标 (列17-32)
        for (i, &v) in metrics.iter().enumerate() {
            results[[idx, i + 17]] = v.abs();
        }
        
        // 新增5个复杂指标
        let today_lz = calculate_lz_complexity_market(&today_sorted, today_order, window_size);
        let yest_lz = calculate_lz_complexity_market(&yest_sorted, yest_order, window_size);
        results[[idx, 33]] = yest_lz - today_lz;
        
        let today_fd = calculate_fractal_dimension_market(&today_sorted, today_order, window_size);
        let yest_fd = calculate_fractal_dimension_market(&yest_sorted, yest_order, window_size);
        results[[idx, 34]] = yest_fd - today_fd;
        
        let today_h = calculate_hurst_exponent_market(&today_sorted, today_order);
        let yest_h = calculate_hurst_exponent_market(&yest_sorted, yest_order);
        results[[idx, 35]] = yest_h - today_h;
        
        let today_pe = calculate_predation_pressure_market(&today_sorted, today_order, window_size);
        let yest_pe = calculate_predation_pressure_market(&yest_sorted, yest_order, window_size);
        results[[idx, 36]] = yest_pe - today_pe;
        
        let today_cv = calculate_interval_cv_market(&today_sorted, today_order, window_size);
        let yest_cv = calculate_interval_cv_market(&yest_sorted, yest_order, window_size);
        results[[idx, 37]] = yest_cv - today_cv;
        
        // 带绝对值的新增指标 (列38-42)
        results[[idx, 38]] = (yest_lz - today_lz).abs();
        results[[idx, 39]] = (yest_fd - today_fd).abs();
        results[[idx, 40]] = (yest_h - today_h).abs();
        results[[idx, 41]] = (yest_pe - today_pe).abs();
        results[[idx, 42]] = (yest_cv - today_cv).abs();
        
        // 第43列：订单成交笔数
        results[[idx, 43]] = today_order.trade_count as f64;
    }
    
    // 列名
    let column_names = vec![
        "today_order_id".to_string(),
        "obs_displacement".to_string(),
        "turnover_diff".to_string(),
        "signed_turnover_diff".to_string(),
        "price_ratio_diff".to_string(),
        "impact1_diff".to_string(),
        "impact2_diff".to_string(),
        "buy_ratio_stability".to_string(),
        "avg_trade_amount_diff".to_string(),
        "trade_duration_diff".to_string(),
        "shadow_overlap".to_string(),
        "price_radiation_diff".to_string(),
        "counterparty_order_diff".to_string(),
        "same_direction_trend_diff".to_string(),
        "opposite_direction_trend_diff".to_string(),
        "buy_order_trend_diff".to_string(),
        "sell_order_trend_diff".to_string(),
        "abs_obs_displacement".to_string(),
        "abs_turnover_diff".to_string(),
        "abs_signed_turnover_diff".to_string(),
        "abs_price_ratio_diff".to_string(),
        "abs_impact1_diff".to_string(),
        "abs_impact2_diff".to_string(),
        "abs_buy_ratio_stability".to_string(),
        "abs_avg_trade_amount_diff".to_string(),
        "abs_trade_duration_diff".to_string(),
        "abs_shadow_overlap".to_string(),
        "abs_price_radiation_diff".to_string(),
        "abs_counterparty_order_diff".to_string(),
        "abs_same_direction_trend_diff".to_string(),
        "abs_opposite_direction_trend_diff".to_string(),
        "abs_buy_order_trend_diff".to_string(),
        "abs_sell_order_trend_diff".to_string(),
        "market_lz_complexity_diff".to_string(),
        "market_fractal_dim_diff".to_string(),
        "market_hurst_exp_diff".to_string(),
        "market_predation_pressure_diff".to_string(),
        "market_interval_cv_diff".to_string(),
        "abs_market_lz_complexity_diff".to_string(),
        "abs_market_fractal_dim_diff".to_string(),
        "abs_market_hurst_exp_diff".to_string(),
        "abs_market_predation_pressure_diff".to_string(),
        "abs_market_interval_cv_diff".to_string(),
        "trade_count".to_string(),
    ];
    
    Ok((results.into_pyarray(py).to_owned(), column_names))
}

/// 计算订单配对指标（V2优化版本）
/// 通过预计算市场指标来加速
#[pyfunction(signature = (today_trades, yesterday_trades, tolerance=0.001))]
pub fn calculate_order_pair_metrics_more_v2_faster(
    py: Python,
    today_trades: PyReadonlyArray2<f64>,
    yesterday_trades: PyReadonlyArray2<f64>,
    tolerance: f64,
) -> PyResult<(Py<PyArray2<f64>>, Vec<String>)> {
    let today_trades = today_trades.as_array();
    let yesterday_trades = yesterday_trades.as_array();
    
    // 解析成交记录
    let today_records = parse_trades(today_trades);
    let yesterday_records = parse_trades(yesterday_trades);
    
    let today_sorted = SortedTrades::from_records(today_records.clone());
    let yest_sorted = SortedTrades::from_records(yesterday_records.clone());
    
    // 获取开盘收盘时间
    let today_open = *today_sorted.times.first().unwrap();
    let today_close = *today_sorted.times.last().unwrap();
    let yest_open = *yest_sorted.times.first().unwrap();
    let yest_close = *yest_sorted.times.last().unwrap();
    
    // 计算日均价格
    let today_avg_price: f64 = today_sorted.prices.iter().sum::<f64>() / today_sorted.prices.len() as f64;
    let yest_avg_price: f64 = yest_sorted.prices.iter().sum::<f64>() / yest_sorted.prices.len() as f64;
    
    // 计算全天价格极差
    let today_price_min = today_sorted.prices.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let today_price_max = today_sorted.prices.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let today_price_range = today_price_max - today_price_min;
    let yest_price_min = yest_sorted.prices.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let yest_price_max = yest_sorted.prices.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let yest_price_range = yest_price_max - yest_price_min;
    
    // 聚合订单信息（V2版本：保留所有订单编号）
    let today_orders = aggregate_orders_v2(&today_records);
    let yest_orders = aggregate_orders_v2(&yesterday_records);
    
    // 获取所有订单编号并排序（确保顺序确定）
    let mut today_all_orders: Vec<i64> = today_orders.keys().cloned().collect();
    let mut yest_all_orders: Vec<i64> = yest_orders.keys().cloned().collect();
    today_all_orders.sort();
    yest_all_orders.sort();
    
    // 配对（不区分买卖方向）
    let all_pairs = find_pairs_v2(&today_all_orders, &yest_all_orders, tolerance);
    
    // 预计算影子订单
    let today_shadow = compute_shadow_orders(&today_orders);
    let yest_shadow = compute_shadow_orders(&yest_orders);
    
    // 预计算所有订单的市场指标
    let window_size = 100usize;
    
    let mut today_market_metrics: HashMap<i64, [f64; 5]> = HashMap::new();
    for (&order_id, order) in &today_orders {
        let lz = calculate_lz_complexity_market(&today_sorted, order, window_size);
        let fd = calculate_fractal_dimension_market(&today_sorted, order, window_size);
        let h = calculate_hurst_exponent_market(&today_sorted, order);
        let pe = calculate_predation_pressure_market(&today_sorted, order, window_size);
        let cv = calculate_interval_cv_market(&today_sorted, order, window_size);
        today_market_metrics.insert(order_id, [lz, fd, h, pe, cv]);
    }
    
    let mut yest_market_metrics: HashMap<i64, [f64; 5]> = HashMap::new();
    for (&order_id, order) in &yest_orders {
        let lz = calculate_lz_complexity_market(&yest_sorted, order, window_size);
        let fd = calculate_fractal_dimension_market(&yest_sorted, order, window_size);
        let h = calculate_hurst_exponent_market(&yest_sorted, order);
        let pe = calculate_predation_pressure_market(&yest_sorted, order, window_size);
        let cv = calculate_interval_cv_market(&yest_sorted, order, window_size);
        yest_market_metrics.insert(order_id, [lz, fd, h, pe, cv]);
    }
    
    // 计算指标
    let mut results = Array2::<f64>::zeros((all_pairs.len(), 44));
    
    for (idx, &(today_id, yest_id)) in all_pairs.iter().enumerate() {
        let today_order = today_orders.get(&today_id).unwrap();
        let yest_order = yest_orders.get(&yest_id).unwrap();
        
        results[[idx, 0]] = today_id as f64;
        
        // 计算原有16个指标
        let metrics = compute_metrics(
            today_order,
            yest_order,
            &today_sorted,
            &yest_sorted,
            today_open,
            today_close,
            yest_open,
            yest_close,
            today_avg_price,
            yest_avg_price,
            today_price_range,
            yest_price_range,
            &today_shadow,
            &yest_shadow,
        );
        
        // 不带绝对值的原有指标 (列1-16)
        for (i, &v) in metrics.iter().enumerate() {
            results[[idx, i + 1]] = v;
        }
        
        // 带绝对值的原有指标 (列17-32)
        for (i, &v) in metrics.iter().enumerate() {
            results[[idx, i + 17]] = v.abs();
        }
        
        // 使用预计算的市场指标
        let today_mm = today_market_metrics.get(&today_id).unwrap();
        let yest_mm = yest_market_metrics.get(&yest_id).unwrap();
        
        results[[idx, 33]] = yest_mm[0] - today_mm[0];
        results[[idx, 34]] = yest_mm[1] - today_mm[1];
        results[[idx, 35]] = yest_mm[2] - today_mm[2];
        results[[idx, 36]] = yest_mm[3] - today_mm[3];
        results[[idx, 37]] = yest_mm[4] - today_mm[4];
        
        // 带绝对值的新增指标 (列38-42)
        results[[idx, 38]] = (yest_mm[0] - today_mm[0]).abs();
        results[[idx, 39]] = (yest_mm[1] - today_mm[1]).abs();
        results[[idx, 40]] = (yest_mm[2] - today_mm[2]).abs();
        results[[idx, 41]] = (yest_mm[3] - today_mm[3]).abs();
        results[[idx, 42]] = (yest_mm[4] - today_mm[4]).abs();
        
        // 第43列：订单成交笔数
        results[[idx, 43]] = today_order.trade_count as f64;
    }
    
    // 列名
    let column_names = vec![
        "today_order_id".to_string(),
        "obs_displacement".to_string(),
        "turnover_diff".to_string(),
        "signed_turnover_diff".to_string(),
        "price_ratio_diff".to_string(),
        "impact1_diff".to_string(),
        "impact2_diff".to_string(),
        "buy_ratio_stability".to_string(),
        "avg_trade_amount_diff".to_string(),
        "trade_duration_diff".to_string(),
        "shadow_overlap".to_string(),
        "price_radiation_diff".to_string(),
        "counterparty_order_diff".to_string(),
        "same_direction_trend_diff".to_string(),
        "opposite_direction_trend_diff".to_string(),
        "buy_order_trend_diff".to_string(),
        "sell_order_trend_diff".to_string(),
        "abs_obs_displacement".to_string(),
        "abs_turnover_diff".to_string(),
        "abs_signed_turnover_diff".to_string(),
        "abs_price_ratio_diff".to_string(),
        "abs_impact1_diff".to_string(),
        "abs_impact2_diff".to_string(),
        "abs_buy_ratio_stability".to_string(),
        "abs_avg_trade_amount_diff".to_string(),
        "abs_trade_duration_diff".to_string(),
        "abs_shadow_overlap".to_string(),
        "abs_price_radiation_diff".to_string(),
        "abs_counterparty_order_diff".to_string(),
        "abs_same_direction_trend_diff".to_string(),
        "abs_opposite_direction_trend_diff".to_string(),
        "abs_buy_order_trend_diff".to_string(),
        "abs_sell_order_trend_diff".to_string(),
        "market_lz_complexity_diff".to_string(),
        "market_fractal_dim_diff".to_string(),
        "market_hurst_exp_diff".to_string(),
        "market_predation_pressure_diff".to_string(),
        "market_interval_cv_diff".to_string(),
        "abs_market_lz_complexity_diff".to_string(),
        "abs_market_fractal_dim_diff".to_string(),
        "abs_market_hurst_exp_diff".to_string(),
        "abs_market_predation_pressure_diff".to_string(),
        "abs_market_interval_cv_diff".to_string(),
        "trade_count".to_string(),
    ];
    
    Ok((results.into_pyarray(py).to_owned(), column_names))
}
