#!/usr/bin/env python3
"""
大数据集场景测试
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

def test_large_dataset():
    """测试大数据集性能"""
    print("=== 大数据集性能测试 ===")
    
    # 模拟50万条股票交易数据（接近真实场景）
    n = 500000
    window = '60s'
    
    print(f"创建 {n:,} 条模拟股票交易数据...")
    
    # 创建交易时间 - 模拟一天的交易时间（9:30-15:00）
    start_time = pd.Timestamp('2022-08-19 09:30:00')
    
    # 生成随机的时间间隔（毫秒级）
    np.random.seed(42)
    time_intervals = np.random.exponential(scale=50, size=n)  # 平均50ms间隔
    cumulative_times = np.cumsum(time_intervals)
    
    # 转换为pandas时间戳
    timestamps = pd.to_datetime(start_time.value + cumulative_times * 1e6, unit='ns')
    
    # 生成价格数据 - 模拟股价随机游走
    initial_price = 10.0
    price_changes = np.random.normal(0, 0.001, n)  # 小幅价格变动
    prices = initial_price + np.cumsum(price_changes)
    
    # 创建DataFrame
    df = pd.DataFrame({'price': prices}, index=timestamps)
    
    print(f"数据创建完成，时间范围: {df.index[0]} 到 {df.index[-1]}")
    print(f"数据量: {len(df):,} 行")
    
    # 测试不同操作的性能
    operations = ['first', 'count', 'mean', 'std']
    
    print(f"\n{'操作':<10} {'时间(秒)':<10} {'速度(行/秒)':<15}")
    print("-" * 40)
    
    for op in operations:
        # 预热
        _ = getattr(df.price.rolling_past(window), op)()
        
        # 实际测试
        start_time = time.time()
        result = getattr(df.price.rolling_past(window), op)()
        elapsed_time = time.time() - start_time
        throughput = n / elapsed_time if elapsed_time > 0 else float('inf')
        
        print(f"{op:<10} {elapsed_time:<10.4f} {throughput:<15,.0f}")
    
    # 与pandas对比
    print("\n--- 与pandas对比 ---")
    start_time = time.time()
    pandas_result = df.price.rolling(window, closed='both').mean()
    pandas_time = time.time() - start_time
    pandas_throughput = n / pandas_time
    
    print(f"pandas mean: {pandas_time:.4f}秒 ({pandas_throughput:,.0f} 行/秒)")
    
    # 计算我们的mean相对于pandas的性能比
    start_time = time.time()
    our_result = df.price.rolling_past(window).mean()
    our_time = time.time() - start_time
    our_throughput = n / our_time
    
    print(f"our mean:    {our_time:.4f}秒 ({our_throughput:,.0f} 行/秒)")
    print(f"性能比 (our/pandas): {our_time/pandas_time:.2f}x")

if __name__ == "__main__":
    test_large_dataset()