#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Python版本的"以退为进"分析函数
用于与Rust版本进行结果一致性和性能对比
"""

import numpy as np
import pandas as pd
import time
from typing import Tuple, List, Set
import warnings
warnings.filterwarnings('ignore')

def find_local_peaks_python(prices: np.ndarray) -> List[int]:
    """
    Python版本：找到价格序列中的局部高点
    """
    peaks = []
    n = len(prices)
    
    if n < 3:
        return peaks
    
    for i in range(1, n-1):
        current_price = prices[i]
        
        # 向左查找第一个不同的价格
        left_different = False
        left_lower = False
        for j in range(i-1, -1, -1):
            if abs(prices[j] - current_price) > 1e-10:
                left_different = True
                left_lower = prices[j] < current_price
                break
        
        # 向右查找第一个不同的价格
        right_different = False
        right_lower = False
        for j in range(i+1, n):
            if abs(prices[j] - current_price) > 1e-10:
                right_different = True
                right_lower = prices[j] < current_price
                break
        
        # 如果左右两边的第一个不同价格都比当前价格低，则为局部高点
        if left_different and right_different and left_lower and right_lower:
            peaks.append(i)
    
    return peaks

def calculate_percentile_python(values: np.ndarray, percentile: float) -> float:
    """
    Python版本：计算数组的百分位数
    """
    return np.percentile(values, percentile)

def check_large_volume_near_peak_python(
    orderbook_times: np.ndarray,
    orderbook_prices: np.ndarray, 
    orderbook_volumes: np.ndarray,
    peak_price: float,
    peak_time: float,
    volume_threshold: float
) -> bool:
    """
    Python版本：检查局部高点附近是否有异常大的挂单量
    """
    time_window = 1.0 / 60.0  # 1分钟
    
    for i in range(len(orderbook_times)):
        time_diff = abs(orderbook_times[i] - peak_time)
        price_diff = abs(orderbook_prices[i] - peak_price)
        
        # 在时间窗口内且价格相近的挂单
        if time_diff <= time_window and price_diff < peak_price * 0.001:
            if orderbook_volumes[i] >= volume_threshold:
                return True
    
    return False

def find_breakthrough_point_python(
    trade_times: np.ndarray,
    trade_prices: np.ndarray,
    peak_idx: int,
    peak_price: float
) -> int:
    """
    Python版本：寻找突破点
    """
    n = len(trade_prices)
    
    # 从局部高点之后开始查找
    for i in range(peak_idx + 1, n):
        if trade_prices[i] > peak_price * 1.001:  # 突破局部高点0.1%以上
            return i
        
        # 设置最大搜索时间窗口
        time_diff = trade_times[i] - trade_times[peak_idx]
        if time_diff > 4.0 / 60.0:  # 4小时后仍未突破则放弃
            break
    
    return -1

class RetreatAdvanceProcess:
    """表示一个"以退为进"过程"""
    def __init__(self, peak_index: int, peak_price: float, start_time: float, 
                 end_time: float, start_index: int, end_index: int):
        self.peak_index = peak_index
        self.peak_price = peak_price
        self.start_time = start_time
        self.end_time = end_time
        self.start_index = start_index
        self.end_index = end_index

def identify_retreat_advance_processes_python(
    trade_times: np.ndarray,
    trade_prices: np.ndarray,
    trade_volumes: np.ndarray,
    trade_flags: np.ndarray,
    orderbook_times: np.ndarray,
    orderbook_prices: np.ndarray,
    orderbook_volumes: np.ndarray,
    local_peaks: List[int],
    volume_threshold: float
) -> List[RetreatAdvanceProcess]:
    """
    Python版本：识别"以退为进"过程
    """
    processes = []
    
    for peak_idx in local_peaks:
        peak_price = trade_prices[peak_idx]
        peak_time = trade_times[peak_idx]
        
        # 检查在局部高点附近1分钟内是否有异常大的挂单量
        has_large_volume = check_large_volume_near_peak_python(
            orderbook_times, orderbook_prices, orderbook_volumes,
            peak_price, peak_time, volume_threshold
        )
        
        if not has_large_volume:
            continue
        
        # 寻找过程的结束点：价格成功突破局部高点
        end_idx = find_breakthrough_point_python(
            trade_times, trade_prices, peak_idx, peak_price
        )
        
        if end_idx != -1:
            process = RetreatAdvanceProcess(
                peak_index=peak_idx,
                peak_price=peak_price,
                start_time=peak_time,
                end_time=trade_times[end_idx],
                start_index=peak_idx,
                end_index=end_idx
            )
            processes.append(process)
    
    return processes

def calculate_total_volume_python(trade_volumes: np.ndarray, start_idx: int, end_idx: int) -> float:
    """计算指定范围内的总成交量"""
    return np.sum(trade_volumes[start_idx:end_idx+1])

def find_first_large_volume_python(
    orderbook_times: np.ndarray,
    orderbook_prices: np.ndarray,
    orderbook_volumes: np.ndarray,
    peak_price: float,
    start_time: float,
    end_time: float
) -> float:
    """找到过程期间首次观察到的异常大挂单量"""
    for i in range(len(orderbook_times)):
        time = orderbook_times[i]
        price = orderbook_prices[i]
        volume = orderbook_volumes[i]
        
        # 在过程时间范围内且价格相近
        if start_time <= time <= end_time:
            price_diff = abs(price - peak_price)
            if price_diff < peak_price * 0.001:
                return volume
    
    return 0.0

def calculate_one_minute_volume_python(
    trade_times: np.ndarray,
    trade_volumes: np.ndarray,
    start_idx: int,
    start_time: float
) -> float:
    """计算过程开始后1分钟内的成交量"""
    volume = 0.0
    one_minute = 1.0 / 60.0
    
    for i in range(start_idx, len(trade_times)):
        time_diff = trade_times[i] - start_time
        if time_diff <= one_minute:
            volume += trade_volumes[i]
        else:
            break
    
    return volume

def calculate_buy_ratio_python(
    trade_flags: np.ndarray,
    trade_volumes: np.ndarray,
    start_idx: int,
    end_idx: int
) -> float:
    """计算主动买入成交量占比"""
    total_volume = 0.0
    buy_volume = 0.0
    
    for i in range(start_idx, end_idx + 1):
        volume = trade_volumes[i]
        total_volume += volume
        
        if trade_flags[i] > 0.0:
            buy_volume += volume
    
    return buy_volume / total_volume if total_volume > 0.0 else 0.0

def calculate_unique_prices_python(
    trade_prices: np.ndarray,
    start_idx: int,
    end_idx: int
) -> int:
    """计算过程期间的唯一价格数量"""
    unique_prices = set()
    
    for i in range(start_idx, end_idx + 1):
        # 使用价格的整数表示来避免浮点数精度问题
        price_key = round(trade_prices[i] * 1000)
        unique_prices.add(price_key)
    
    return len(unique_prices)

def calculate_max_decline_python(
    trade_prices: np.ndarray,
    start_idx: int,
    end_idx: int,
    peak_price: float
) -> float:
    """计算过程期间价格相对局部高点的最大下降比例"""
    max_decline = 0.0
    
    for i in range(start_idx, end_idx + 1):
        decline = (peak_price - trade_prices[i]) / peak_price
        if decline > max_decline:
            max_decline = decline
    
    return max_decline

def analyze_retreat_advance_python(
    trade_times: np.ndarray,
    trade_prices: np.ndarray,
    trade_volumes: np.ndarray,
    trade_flags: np.ndarray,
    orderbook_times: np.ndarray,
    orderbook_prices: np.ndarray,
    orderbook_volumes: np.ndarray,
    volume_percentile: float = 99.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Python版本的"以退为进"分析函数
    """
    # 验证输入数据长度一致性
    if not (len(trade_times) == len(trade_prices) == len(trade_volumes) == len(trade_flags)):
        raise ValueError("逐笔成交数据各列长度不一致")
    
    if not (len(orderbook_times) == len(orderbook_prices) == len(orderbook_volumes)):
        raise ValueError("盘口快照数据各列长度不一致")
    
    # 步骤1：找到所有局部高点
    local_peaks = find_local_peaks_python(trade_prices)
    
    # 步骤2：计算挂单量的百分位数阈值
    volume_threshold = calculate_percentile_python(orderbook_volumes, volume_percentile)
    
    # 步骤3：识别"以退为进"过程
    processes = identify_retreat_advance_processes_python(
        trade_times, trade_prices, trade_volumes, trade_flags,
        orderbook_times, orderbook_prices, orderbook_volumes,
        local_peaks, volume_threshold
    )
    
    # 步骤4：计算每个过程的6个指标
    process_volumes = []
    large_volumes = []
    one_min_volumes = []
    buy_ratios = []
    price_counts = []
    max_declines = []
    
    for process in processes:
        # 指标1：过程期间的成交量
        total_volume = calculate_total_volume_python(
            trade_volumes, process.start_index, process.end_index
        )
        process_volumes.append(total_volume)
        
        # 指标2：过程期间首次观察到的异常大挂单量
        first_large_volume = find_first_large_volume_python(
            orderbook_times, orderbook_prices, orderbook_volumes,
            process.peak_price, process.start_time, process.end_time
        )
        large_volumes.append(first_large_volume)
        
        # 指标3：过程开始后1分钟内的成交量
        one_min_volume = calculate_one_minute_volume_python(
            trade_times, trade_volumes, process.start_index, process.start_time
        )
        one_min_volumes.append(one_min_volume)
        
        # 指标4：过程期间的主动买入成交量占比
        buy_ratio = calculate_buy_ratio_python(
            trade_flags, trade_volumes, process.start_index, process.end_index
        )
        buy_ratios.append(buy_ratio)
        
        # 指标5：过程期间的价格种类数
        price_count = calculate_unique_prices_python(
            trade_prices, process.start_index, process.end_index
        )
        price_counts.append(float(price_count))
        
        # 指标6：过程期间价格相对局部高点的最大下降比例
        max_decline = calculate_max_decline_python(
            trade_prices, process.start_index, process.end_index, process.peak_price
        )
        max_declines.append(max_decline)
    
    return (
        np.array(process_volumes),
        np.array(large_volumes),
        np.array(one_min_volumes),
        np.array(buy_ratios),
        np.array(price_counts),
        np.array(max_declines)
    )

def compare_results(rust_results, python_results, tolerance=1e-6):
    """
    比较Rust和Python版本的结果
    """
    print("\n🔍 结果一致性检查:")
    print("=" * 50)
    
    if len(rust_results) != len(python_results):
        print(f"✗ 返回结果数量不一致: Rust={len(rust_results)}, Python={len(python_results)}")
        return False
    
    all_consistent = True
    result_names = [
        "过程成交量", "异常挂单量", "1分钟成交量", 
        "买入占比", "价格种类数", "最大下跌比例"
    ]
    
    for i, (rust_arr, python_arr, name) in enumerate(zip(rust_results, python_results, result_names)):
        if len(rust_arr) != len(python_arr):
            print(f"✗ {name}: 长度不一致 (Rust: {len(rust_arr)}, Python: {len(python_arr)})")
            all_consistent = False
            continue
        
        if len(rust_arr) == 0:
            print(f"✓ {name}: 都为空数组")
            continue
        
        # 计算差异
        diff = np.abs(rust_arr - python_arr)
        max_diff = np.max(diff)
        rel_diff = np.max(diff / (np.abs(rust_arr) + 1e-12))
        
        if max_diff < tolerance and rel_diff < tolerance:
            print(f"✓ {name}: 完全一致 (最大差异: {max_diff:.2e})")
        else:
            print(f"✗ {name}: 不一致 (最大差异: {max_diff:.2e}, 相对差异: {rel_diff:.2e})")
            # 显示前几个不一致的值
            inconsistent_indices = np.where(diff > tolerance)[0][:5]
            for idx in inconsistent_indices:
                print(f"    索引{idx}: Rust={rust_arr[idx]:.6f}, Python={python_arr[idx]:.6f}")
            all_consistent = False
    
    return all_consistent

def performance_comparison():
    """
    性能对比测试
    """
    print("\n🚀 性能对比测试")
    print("=" * 50)
    
    # 导入必要的库
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))
    
    try:
        from rust_pyfunc import analyze_retreat_advance
        import design_whatever as dw
    except ImportError as e:
        print(f"✗ 导入失败: {e}")
        return False
    
    # 加载真实数据
    print("加载测试数据...")
    date = 20220819
    symbol = '000001'
    
    trade_data = dw.read_l2_trade_data(start_date=date, symbols=[symbol], with_retreat=0)
    asks_data, _ = dw.read_l2_market_data_price_vol_pair(date=date, symbols=[symbol])
    
    # 预处理数据
    trade_times = (trade_data['exchtime'].dt.hour + 
                  trade_data['exchtime'].dt.minute / 60.0 + 
                  trade_data['exchtime'].dt.second / 3600.0).values.astype(np.float64)
    trade_prices = trade_data['price'].values.astype(np.float64)
    trade_volumes = trade_data['volume'].values.astype(np.float64)
    trade_flags = np.where(trade_data['flag'] == 66, 1.0, 
                          np.where(trade_data['flag'] == 83, -1.0, 0.0)).astype(np.float64)
    
    orderbook_times = (asks_data['exchtime'].dt.hour + 
                      asks_data['exchtime'].dt.minute / 60.0 + 
                      asks_data['exchtime'].dt.second / 3600.0).values.astype(np.float64)
    orderbook_prices = asks_data['price'].values.astype(np.float64)
    orderbook_volumes = asks_data['vol'].values.astype(np.float64)
    
    print(f"数据规模: 成交{len(trade_times):,}条, 盘口{len(orderbook_times):,}条")
    
    # 测试不同阈值
    thresholds = [90.0, 95.0]
    
    for threshold in thresholds:
        print(f"\n测试阈值: {threshold}%")
        print("-" * 30)
        
        # Rust版本性能测试
        rust_times = []
        for _ in range(3):  # 运行3次取平均
            start_time = time.time()
            rust_results = analyze_retreat_advance(
                trade_times, trade_prices, trade_volumes, trade_flags,
                orderbook_times, orderbook_prices, orderbook_volumes,
                volume_percentile=threshold
            )
            rust_time = time.time() - start_time
            rust_times.append(rust_time)
        
        avg_rust_time = np.mean(rust_times)
        
        # Python版本性能测试
        python_times = []
        for _ in range(3):  # 运行3次取平均
            start_time = time.time()
            python_results = analyze_retreat_advance_python(
                trade_times, trade_prices, trade_volumes, trade_flags,
                orderbook_times, orderbook_prices, orderbook_volumes,
                volume_percentile=threshold
            )
            python_time = time.time() - start_time
            python_times.append(python_time)
        
        avg_python_time = np.mean(python_times)
        
        # 性能对比
        speedup = avg_python_time / avg_rust_time
        
        print(f"Rust版本   : {avg_rust_time:.4f}秒 (±{np.std(rust_times):.4f})")
        print(f"Python版本 : {avg_python_time:.4f}秒 (±{np.std(python_times):.4f})")
        print(f"加速比     : {speedup:.2f}x")
        print(f"发现过程数 : Rust={len(rust_results[0])}, Python={len(python_results[0])}")
        
        # 结果一致性检查
        consistent = compare_results(rust_results, python_results)
        
        if consistent:
            print("✅ 结果完全一致!")
        else:
            print("❌ 结果存在差异!")
    
    return True

if __name__ == "__main__":
    performance_comparison()