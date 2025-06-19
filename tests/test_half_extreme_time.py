#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
import time
import sys
import os
from pathlib import Path

# 添加项目根目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent.parent))

# 导入 Rust 实现
from rust_pyfunc import find_half_extreme_time as rust_find_half_extreme_time

def python_find_half_extreme_time(times: np.ndarray, prices: np.ndarray, time_window: float = 5.0) -> np.ndarray:
    """
    Python 版本的 find_half_extreme_time 实现
    
    参数:
        times: 时间戳数组（单位：秒）
        prices: 价格数组
        time_window: 时间窗口大小（秒）
        
    返回:
        numpy.ndarray: 每个时间点达到最大变动一半所需的时间
    """
    n = len(times)
    result = np.full(n, time_window, dtype=np.float64)
    
    for i in range(n):
        current_time = times[i]
        current_price = prices[i]
        
        # 处理NaN和Inf值
        if not np.isfinite(current_price):
            result[i] = np.nan
            continue
            
        max_up = 0.0
        max_down = 0.0
        
        # 首先找到时间窗口内的最大上涨和下跌幅度
        for j in range(i, n):
            if i == j:
                continue
                
            time_diff = times[j] - current_time
            if time_diff > time_window:
                break
                
            # 跳过无效价格
            if not np.isfinite(prices[j]):
                continue
                
            # 计算价格变动比率
            price_ratio = (prices[j] - current_price) / current_price
            
            if price_ratio > 0:
                max_up = max(max_up, price_ratio)
            else:
                max_down = min(max_down, price_ratio)
        
        # 确定主要方向（取绝对值较大的）
        if abs(max_up) > abs(max_down):
            target_change = max_up / 2.0
            direction = 1
        else:
            target_change = max_down / 2.0
            direction = -1
        
        # 如果目标变化为0，则跳过（没有价格变动）
        if abs(target_change) < 1e-10:
            continue
        
        # 查找首次达到目标变化的时间
        for j in range(i, n):
            if i == j:
                continue
                
            time_diff = times[j] - current_time
            if time_diff > time_window:
                break
                
            # 跳过无效价格
            if not np.isfinite(prices[j]):
                continue
                
            price_ratio = (prices[j] - current_price) / current_price
            
            if (direction > 0 and price_ratio >= target_change) or \
               (direction < 0 and price_ratio <= target_change):
                result[i] = time_diff
                break
    
    return result

def load_stock_data(symbol: str = '600519.SH', start_date: str = '20230101', end_date: str = '20230131') -> pd.DataFrame:
    """
    加载股票数据
    
    参数:
        symbol: 股票代码
        start_date: 开始日期，格式为'YYYYMMDD'
        end_date: 结束日期，格式为'YYYYMMDD'
        
    返回:
        pd.DataFrame: 包含时间和价格的数据框
    """
    try:
        # 尝试使用 tushare 获取数据
        import tushare as ts
        pro = ts.pro_api()
        df = pro.daily(ts_code=symbol, start_date=start_date, end_date=end_date)
        df = df.sort_values('trade_date')
        
        # 转换为时间戳（秒）
        df['time'] = pd.to_datetime(df['trade_date']).astype('int64') // 10**9
        df['price'] = df['close']
        return df[['time', 'price']].sort_values('time')
        
    except ImportError:
        # 如果 tushare 不可用，使用本地测试数据
        print("Tushare 不可用，使用模拟数据")
        np.random.seed(42)
        n = 1000
        times = np.cumsum(np.random.uniform(0.1, 0.5, n))  # 随机时间间隔
        prices = 100 + np.cumsum(np.random.randn(n) * 0.5)  # 随机游走价格
        return pd.DataFrame({'time': times, 'price': prices})

def test_with_real_data():
    """使用真实股票数据测试"""
    print("\n=== 使用真实股票数据测试 ===")
    
    # 加载数据
    df = load_stock_data()
    times = df['time'].values.astype(np.float64)
    prices = df['price'].values.astype(np.float64)
    
    print(f"数据量: {len(times)} 条")
    
    # 测试 Rust 实现
    start = time.time()
    rust_result = rust_find_half_extreme_time(times, prices, time_window=3600*24)  # 24小时窗口
    rust_time = time.time() - start
    print(f"Rust 实现耗时: {rust_time:.4f} 秒")
    
    # 测试 Python 实现
    start = time.time()
    python_result = python_find_half_extreme_time(times, prices, time_window=3600*24)
    python_time = time.time() - start
    print(f"Python 实现耗时: {python_time:.4f} 秒")
    
    # 比较结果
    is_equal = np.allclose(rust_result, python_result, equal_nan=True, rtol=1e-5, atol=1e-5)
    print(f"结果一致性检查: {'通过' if is_equal else '失败'}")
    
    # 如果不一致，显示差异
    if not is_equal:
        diff = np.abs(rust_result - python_result)
        diff_indices = np.where(diff > 1e-5)[0]
        print(f"发现 {len(diff_indices)} 处不一致")
        for idx in diff_indices[:5]:  # 只显示前5处差异
            print(f"索引 {idx}: Rust={rust_result[idx]:.6f}, Python={python_result[idx]:.6f}, 差异={diff[idx]:.6f}")
    
    # 性能对比
    if python_time > 0 and rust_time > 0:
        speedup = python_time / rust_time
        print(f"性能提升: {speedup:.2f} 倍")
    
    return {
        'rust_time': rust_time,
        'python_time': python_time,
        'is_equal': is_equal,
        'data_size': len(times)
    }

def test_with_synthetic_data():
    """使用合成数据测试"""
    print("\n=== 使用合成数据测试 ===")
    
    # 生成测试数据
    np.random.seed(42)
    n = 10000  # 数据量
    times = np.cumsum(np.random.uniform(0.1, 0.5, n))  # 随机时间间隔
    prices = 100 + np.cumsum(np.random.randn(n) * 0.5)  # 随机游走价格
    
    print(f"数据量: {n} 条")
    
    # 测试 Rust 实现
    start = time.time()
    rust_result = rust_find_half_extreme_time(times, prices, time_window=60)  # 60秒窗口
    rust_time = time.time() - start
    print(f"Rust 实现耗时: {rust_time:.4f} 秒")
    
    # 测试 Python 实现
    start = time.time()
    python_result = python_find_half_extreme_time(times, prices, time_window=60)
    python_time = time.time() - start
    print(f"Python 实现耗时: {python_time:.4f} 秒")
    
    # 比较结果
    is_equal = np.allclose(rust_result, python_result, equal_nan=True, rtol=1e-5, atol=1e-5)
    print(f"结果一致性检查: {'通过' if is_equal else '失败'}")
    
    # 性能对比
    if python_time > 0 and rust_time > 0:
        speedup = python_time / rust_time
        print(f"性能提升: {speedup:.2f} 倍")
    
    return {
        'rust_time': rust_time,
        'python_time': python_time,
        'is_equal': is_equal,
        'data_size': n
    }

def test_edge_cases():
    """测试边界情况和异常值处理能力"""
    print("\n=== 测试边界情况和异常值 ===\n")
    
    # 测试用例1: NaN值处理
    print("1. NaN值处理测试")
    times = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    prices = np.array([10.0, np.nan, 30.0, 20.0, 15.0])
    
    rust_result = rust_find_half_extreme_time(times, prices, time_window=3.0)
    python_result = python_find_half_extreme_time(times, prices, time_window=3.0)
    
    rust_nans = np.isnan(rust_result)
    python_nans = np.isnan(python_result)
    
    print(f"Rust结果: {rust_result}")
    print(f"Python结果: {python_result}")
    print(f"NaN处理一致性: {np.array_equal(rust_nans, python_nans)}")
    print()
    
    # 测试用例2: 无限值处理
    print("2. 无限值处理测试")
    times = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    prices = np.array([10.0, np.inf, 30.0, -np.inf, 15.0])
    
    rust_result = rust_find_half_extreme_time(times, prices, time_window=3.0)
    python_result = python_find_half_extreme_time(times, prices, time_window=3.0)
    
    print(f"Rust结果: {rust_result}")
    print(f"Python结果: {python_result}")
    print(f"无限值处理一致性: {np.allclose(rust_result, python_result, equal_nan=True)}")
    print()
    
    # 测试用例3: 空数组处理
    print("3. 空数组处理测试")
    try:
        rust_result = rust_find_half_extreme_time(np.array([]), np.array([]), time_window=1.0)
        print(f"Rust结果: {rust_result}, 长度: {len(rust_result)}")
    except Exception as e:
        print(f"Rust异常: {e}")
        
    try:
        python_result = python_find_half_extreme_time(np.array([]), np.array([]), time_window=1.0)
        print(f"Python结果: {python_result}, 长度: {len(python_result)}")
    except Exception as e:
        print(f"Python异常: {e}")
    print()
    
    # 测试用例4: 单个值处理
    print("4. 单个值处理测试")
    times = np.array([1.0])
    prices = np.array([10.0])
    
    rust_result = rust_find_half_extreme_time(times, prices, time_window=1.0)
    python_result = python_find_half_extreme_time(times, prices, time_window=1.0)
    
    print(f"Rust结果: {rust_result}")
    print(f"Python结果: {python_result}")
    print(f"单值处理一致性: {np.allclose(rust_result, python_result)}")
    print()
    
    # 测试用例5: 平坦价格处理
    print("5. 平坦价格处理测试 (所有价格相同)")
    times = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    prices = np.array([10.0, 10.0, 10.0, 10.0, 10.0])
    
    rust_result = rust_find_half_extreme_time(times, prices, time_window=3.0)
    python_result = python_find_half_extreme_time(times, prices, time_window=3.0)
    
    print(f"Rust结果: {rust_result}")
    print(f"Python结果: {python_result}")
    print(f"平坦价格处理一致性: {np.allclose(rust_result, python_result)}")
    
    return {
        'nan_test': np.array_equal(rust_nans, python_nans),
        'inf_test': np.allclose(rust_result, python_result, equal_nan=True),
        'single_value_test': np.allclose(rust_result, python_result)
    }

def test_performance_scaling():
    """测试性能随数据量增加的变化"""
    print("\n=== 性能扩展性测试 ===\n")
    
    # 测试不同数据量
    sizes = [1000, 5000, 10000, 20000, 50000]
    rust_times = []
    python_times = []
    speedups = []
    
    for size in sizes:
        print(f"\n测试数据量: {size}")
        
        # 生成测试数据
        np.random.seed(42)
        times = np.cumsum(np.random.uniform(0.1, 0.5, size))  # 随机时间间隔
        prices = 100 + np.cumsum(np.random.randn(size) * 0.5)  # 随机游走价格
        
        # 测试 Rust 实现
        start = time.time()
        rust_find_half_extreme_time(times, prices, time_window=60)  # 60秒窗口
        rust_time = time.time() - start
        rust_times.append(rust_time)
        
        # 测试 Python 实现
        start = time.time()
        python_find_half_extreme_time(times, prices, time_window=60)
        python_time = time.time() - start
        python_times.append(python_time)
        
        # 计算加速比
        speedup = python_time / rust_time
        speedups.append(speedup)
        
        print(f"Rust 耗时: {rust_time:.4f} 秒")
        print(f"Python 耗时: {python_time:.4f} 秒")
        print(f"加速比: {speedup:.2f} 倍")
    
    # 打印性能统计信息
    print("\n性能统计信息:")
    print(f"数据量: {sizes}")
    print(f"Rust 耗时 (秒): {[f'{t:.4f}' for t in rust_times]}")
    print(f"Python 耗时 (秒): {[f'{t:.4f}' for t in python_times]}")
    print(f"加速比: {[f'{s:.2f}' for s in speedups]}")
    print(f"最大加速比: {max(speedups):.2f} 倍 (数据量: {sizes[speedups.index(max(speedups))]})") 
    print(f"平均加速比: {sum(speedups)/len(speedups):.2f} 倍")
    
    return {
        'sizes': sizes,
        'rust_times': rust_times,
        'python_times': python_times,
        'speedups': speedups
    }

if __name__ == "__main__":
    print("=== 开始测试 find_half_extreme_time 函数 ===")
    
    # 运行基础测试
    test_with_synthetic_data()
    test_with_real_data()
    
    # 边界情况测试
    test_edge_cases()
    
    # 性能测试
    test_performance_scaling()
