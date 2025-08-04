#!/usr/bin/env python3
"""
快速性能分析工具
"""

import sys
import numpy as np
import pandas as pd
import time
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "python"))

import rust_pyfunc as rp

def profile_components():
    """分析各个组件的性能开销"""
    print("=== 组件性能分析 ===")
    
    # 使用较小的数据集进行分析
    n = 50000
    window = '60s'
    
    print(f"创建 {n:,} 条测试数据...")
    
    # 创建测试数据
    start_time = pd.Timestamp('2022-08-19 09:30:00')
    np.random.seed(42)
    time_intervals = np.random.exponential(scale=100, size=n)  # 平均100ms间隔
    cumulative_times = np.cumsum(time_intervals)
    timestamps = pd.to_datetime(start_time.value + cumulative_times * 1e6, unit='ns')
    prices = 10.0 + np.cumsum(np.random.normal(0, 0.001, n))
    
    df = pd.DataFrame({'price': prices}, index=timestamps)
    print(f"数据创建完成，时间范围: {df.index[0]} 到 {df.index[-1]}")
    
    # 1. 测试时间转换开销
    print("\n--- 时间转换分析 ---")
    start = time.time()
    times_ns = df.index.astype(np.int64).to_numpy()
    time_conversion = time.time() - start
    print(f"时间转换为纳秒: {time_conversion:.4f}秒")
    
    # 2. 测试窗口转换
    start = time.time()
    window_ns = int(pd.Timedelta(window).total_seconds() * 1e9)
    window_conversion = time.time() - start
    print(f"窗口转换: {window_conversion:.6f}秒")
    
    # 3. 测试数据转换
    start = time.time()
    values = df.price.to_numpy().astype(np.float64)
    data_conversion = time.time() - start
    print(f"数据转换: {data_conversion:.4f}秒")
    
    # 4. 测试纯Rust函数调用
    print("\n--- Rust函数性能 ---")
    times_seconds = times_ns.astype(np.float64) / 1e9
    window_seconds = window_ns / 1e9
    
    # 测试不同操作的Rust性能
    operations = ['mean', 'count', 'first']
    for op in operations:
        start = time.time()
        result = rp.rolling_window_stat_backward(times_seconds, values, window_seconds, op, True)
        rust_time = time.time() - start
        print(f"Rust {op}: {rust_time:.4f}秒 ({n/rust_time:,.0f} 行/秒)")
    
    # 5. 测试Python包装器总开销
    print("\n--- Python包装器分析 ---")
    start = time.time()
    result_mean = df.price.rolling_past(window).mean()
    wrapper_time = time.time() - start
    print(f"Python包装器 mean: {wrapper_time:.4f}秒 ({n/wrapper_time:,.0f} 行/秒)")
    
    # 6. 对比pandas性能
    print("\n--- 与pandas对比 ---")
    start = time.time()
    pandas_result = df.price.rolling(window, closed='both').mean()
    pandas_time = time.time() - start
    print(f"Pandas rolling mean: {pandas_time:.4f}秒 ({n/pandas_time:,.0f} 行/秒)")
    
    print(f"\n性能比 (our/pandas): {wrapper_time/pandas_time:.2f}x")

def test_scaling():
    """测试不同数据规模的性能扩展性"""
    print("\n=== 扩展性分析 ===")
    
    sizes = [10000, 20000, 50000]
    window = '60s'
    
    for n in sizes:
        print(f"\n--- {n:,} 行数据 ---")
        
        # 创建数据
        start_time = pd.Timestamp('2022-08-19 09:30:00')
        np.random.seed(42)
        time_intervals = np.random.exponential(scale=100, size=n)
        cumulative_times = np.cumsum(time_intervals)
        timestamps = pd.to_datetime(start_time.value + cumulative_times * 1e6, unit='ns')
        prices = 10.0 + np.cumsum(np.random.normal(0, 0.001, n))
        df = pd.DataFrame({'price': prices}, index=timestamps)
        
        # 预热
        _ = df.price.rolling_past(window).mean()
        
        # 测试性能
        start = time.time()
        result = df.price.rolling_past(window).first()
        elapsed = time.time() - start
        
        print(f"  first(): {elapsed:.4f}秒 ({n/elapsed:,.0f} 行/秒)")

if __name__ == "__main__":
    profile_components()
    test_scaling()