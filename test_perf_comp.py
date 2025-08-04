#!/usr/bin/env python3
"""
对比rolling_past和rolling_future的性能差异
"""

import sys
import numpy as np
import pandas as pd
import time
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "python"))

import rust_pyfunc

def performance_comparison():
    """详细性能对比"""
    print("=== 详细性能对比测试 ===")
    
    # 测试不同数据大小
    data_sizes = [1000, 5000, 10000]
    window = '10s'
    
    for n in data_sizes:
        print(f"\n--- 数据大小: {n} 行 ---")
        
        # 创建测试数据
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=n, freq='1s')
        values = np.cumsum(np.random.randn(n) * 0.1) + 100
        df = pd.DataFrame({'value': values}, index=dates)
        
        # 预热
        _ = df.rolling_future(window).mean()
        _ = df.rolling_past(window).mean()
        
        # 测试rolling_future
        start_time = time.time()
        future_result = df.rolling_future(window).mean()
        future_time = time.time() - start_time
        
        # 测试rolling_past
        start_time = time.time()
        past_result = df.rolling_past(window).mean()
        past_time = time.time() - start_time
        
        # 测试pandas rolling作为基准
        start_time = time.time()
        pandas_result = df.rolling(window, closed='both').mean()
        pandas_time = time.time() - start_time
        
        print(f"  rolling_future时间: {future_time:.4f}秒")
        print(f"  rolling_past时间:   {past_time:.4f}秒")
        print(f"  pandas rolling时间: {pandas_time:.4f}秒")
        print(f"  性能比 (past/future): {past_time/future_time:.2f}x")
        print(f"  相对pandas (future): {future_time/pandas_time:.2f}x")
        print(f"  相对pandas (past):   {past_time/pandas_time:.2f}x")

def test_rust_function_directly():
    """直接测试Rust函数的性能"""
    print("\n=== 直接测试Rust函数性能 ===")
    
    n = 10000
    window_seconds = 10.0
    
    # 创建测试数据
    np.random.seed(42)
    times = np.arange(n, dtype=np.float64)  # 0, 1, 2, 3, ... 秒
    values = np.cumsum(np.random.randn(n) * 0.1) + 100
    
    # 测试rolling_window_stat (rolling_future使用的)
    start_time = time.time()
    future_result = rust_pyfunc.rolling_window_stat(times, values, window_seconds, "mean", False)
    future_rust_time = time.time() - start_time
    
    # 测试rolling_window_stat_backward (rolling_past使用的)
    start_time = time.time()
    past_result = rust_pyfunc.rolling_window_stat_backward(times, values, window_seconds, "mean", True)
    past_rust_time = time.time() - start_time
    
    print(f"数据大小: {n} 行")
    print(f"窗口大小: {window_seconds} 秒")
    print(f"rolling_window_stat (future): {future_rust_time:.4f}秒")
    print(f"rolling_window_stat_backward (past): {past_rust_time:.4f}秒")
    print(f"性能比 (past/future): {past_rust_time/future_rust_time:.2f}x")
    
    # 验证结果是否正确
    print(f"\n结果验证:")
    print(f"future结果前5个: {future_result[:5]}")
    print(f"past结果前5个: {past_result[:5]}")

def main():
    """主测试函数"""
    print("开始性能对比分析...")
    
    performance_comparison()
    test_rust_function_directly()

if __name__ == "__main__":
    main()