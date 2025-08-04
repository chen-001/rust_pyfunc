#!/usr/bin/env python3
"""
模拟真实股票数据场景的性能测试
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

def create_realistic_stock_data(n=100000):
    """创建模拟真实股票数据"""
    print(f"创建 {n} 条模拟股票交易数据...")
    
    # 创建交易时间 - 模拟一天的交易时间（9:30-15:00）
    start_time = pd.Timestamp('2022-08-19 09:30:00')
    
    # 生成随机的时间间隔（毫秒级）
    np.random.seed(42)
    time_intervals = np.random.exponential(scale=100, size=n)  # 平均100ms间隔
    cumulative_times = np.cumsum(time_intervals)
    
    # 转换为pandas时间戳
    timestamps = pd.to_datetime(start_time.value + cumulative_times * 1e6, unit='ns')
    
    # 生成价格数据 - 模拟股价随机游走
    initial_price = 10.0
    price_changes = np.random.normal(0, 0.001, n)  # 小幅价格变动
    prices = initial_price + np.cumsum(price_changes)
    
    # 创建DataFrame，模拟您的数据结构
    df = pd.DataFrame({
        'price': prices,
        'volume': np.random.randint(100, 10000, n),  # 随机成交量
        'amount': prices * np.random.randint(100, 10000, n)  # 成交金额
    }, index=timestamps)
    
    print(f"数据时间范围: {df.index[0]} 到 {df.index[-1]}")
    print(f"平均时间间隔: {df.index.to_series().diff().mean()}")
    
    return df

def test_performance_scenarios():
    """测试不同数据量的性能"""
    print("=== 真实股票数据场景性能测试 ===")
    
    # 测试不同数据量
    data_sizes = [10000, 50000, 100000, 200000]
    window = '60s'
    
    for n in data_sizes:
        print(f"\n--- 数据量: {n:,} 行 ---")
        
        # 创建测试数据
        df = create_realistic_stock_data(n)
        
        # 预热
        _ = df.price.rolling_past(window).mean()
        
        # 测试 first()
        start_time = time.time()
        result_first = df.price.rolling_past(window).first()
        first_time = time.time() - start_time
        
        # 测试 count()
        start_time = time.time()
        result_count = df.price.rolling_past(window).count()
        count_time = time.time() - start_time
        
        # 测试 mean()
        start_time = time.time()
        result_mean = df.price.rolling_past(window).mean()
        mean_time = time.time() - start_time
        
        # 与pandas对比
        start_time = time.time()
        pandas_mean = df.price.rolling(window, closed='both').mean()
        pandas_time = time.time() - start_time
        
        print(f"  rolling_past first(): {first_time:.4f}秒")
        print(f"  rolling_past count(): {count_time:.4f}秒")
        print(f"  rolling_past mean():  {mean_time:.4f}秒")
        print(f"  pandas rolling mean(): {pandas_time:.4f}秒")
        print(f"  性能比 (our/pandas): {mean_time/pandas_time:.2f}x")
        
        # 计算每秒处理的行数
        throughput_first = n / first_time if first_time > 0 else float('inf')
        throughput_count = n / count_time if count_time > 0 else float('inf')
        throughput_mean = n / mean_time if mean_time > 0 else float('inf')
        
        print(f"  处理速度 first(): {throughput_first:,.0f} 行/秒")
        print(f"  处理速度 count(): {throughput_count:,.0f} 行/秒")
        print(f"  处理速度 mean():  {throughput_mean:,.0f} 行/秒")

def profile_bottlenecks():
    """分析性能瓶颈"""
    print("\n=== 性能瓶颈分析 ===")
    
    n = 100000
    df = create_realistic_stock_data(n)
    window = '60s'
    
    print(f"分析 {n:,} 行数据的性能瓶颈...")
    
    # 测试时间转换的开销
    start_time = time.time()
    times_ns = df.index.astype(np.int64).to_numpy()
    conversion_time = time.time() - start_time
    
    # 测试窗口大小转换
    start_time = time.time()
    window_ns = int(pd.Timedelta(window).total_seconds() * 1e9)
    window_conversion_time = time.time() - start_time
    
    # 测试Rust函数调用
    values = df.price.to_numpy().astype(np.float64)
    times_seconds = times_ns.astype(np.float64) / 1e9
    window_seconds = window_ns / 1e9
    
    start_time = time.time()
    rust_result = rp.rolling_window_stat_backward(times_seconds, values, window_seconds, "mean", True)
    rust_time = time.time() - start_time
    
    # 测试Python包装器开销
    start_time = time.time()
    python_result = df.price.rolling_past(window).mean()
    python_wrapper_time = time.time() - start_time
    
    print(f"时间转换开销:     {conversion_time:.4f}秒 ({conversion_time/python_wrapper_time*100:.1f}%)")
    print(f"窗口转换开销:     {window_conversion_time:.6f}秒")
    print(f"Rust函数执行:     {rust_time:.4f}秒 ({rust_time/python_wrapper_time*100:.1f}%)")
    print(f"Python包装器总时间: {python_wrapper_time:.4f}秒")
    print(f"包装器开销:       {(python_wrapper_time-rust_time):.4f}秒 ({(python_wrapper_time-rust_time)/python_wrapper_time*100:.1f}%)")

def test_large_data():
    """测试大数据量场景"""
    print("\n=== 大数据量场景测试 ===")
    
    # 模拟一天完整的交易数据量
    n = 500000  # 50万条记录
    print(f"测试大数据量: {n:,} 行")
    
    df = create_realistic_stock_data(n)
    window = '60s'
    
    operations = ['first', 'count', 'mean', 'max', 'min']
    
    print(f"{'操作':<10} {'时间(秒)':<10} {'速度(行/秒)':<15}")
    print("-" * 40)
    
    for op in operations:
        start_time = time.time()
        result = getattr(df.price.rolling_past(window), op)()
        elapsed_time = time.time() - start_time
        throughput = n / elapsed_time if elapsed_time > 0 else float('inf')
        
        print(f"{op:<10} {elapsed_time:<10.4f} {throughput:<15,.0f}")

def main():
    """主测试函数"""
    print("开始真实股票数据场景性能测试...")
    
    test_performance_scenarios()
    profile_bottlenecks()
    test_large_data()

if __name__ == "__main__":
    main()