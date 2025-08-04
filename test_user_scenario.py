#!/usr/bin/env python3
"""
模拟用户提供的具体场景
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

def simulate_user_scenario():
    """模拟用户提供的具体使用场景"""
    print("=== 模拟用户场景测试 ===")
    
    # 模拟用户的数据结构
    # 假设是股票交易数据：df = p1.read_trade(code, date).set_index('exchtime')
    n = 200000  # 20万条数据，接近一天的交易数据量
    
    print(f"创建 {n:,} 条模拟交易数据...")
    
    # 创建时间序列 - 模拟真实的交易时间分布
    start_time = pd.Timestamp('2022-08-19 09:30:00')
    np.random.seed(42)
    
    # 使用更真实的时间间隔分布（股票交易不是均匀分布的）
    time_intervals = np.random.exponential(scale=80, size=n)  # 平均80ms间隔
    cumulative_times = np.cumsum(time_intervals)
    
    # 创建DataFrame，模拟用户的数据结构
    timestamps = pd.to_datetime(start_time.value + cumulative_times * 1e6, unit='ns')
    
    # 模拟价格数据（随机游走）
    initial_price = 10.50
    price_changes = np.random.normal(0, 0.002, n)
    prices = initial_price + np.cumsum(price_changes)
    
    df = pd.DataFrame({
        'price': prices,
        'volume': np.random.randint(100, 5000, n),
        'amount': prices * np.random.randint(100, 5000, n)
    }, index=timestamps)
    
    print(f"数据时间范围: {df.index[0]} 到 {df.index[-1]}")
    
    # 模拟用户代码中的操作
    print("\n--- 执行用户场景操作 ---")
    
    # 模拟: close_price = df.price.iloc[-1]
    close_price = df.price.iloc[-1]
    print(f"收盘价: {close_price:.4f}")
    
    # 模拟用户的具体操作
    window = '60s'
    
    operations = [
        ('first', lambda: df.price.rolling_past(window).first()),
        ('count', lambda: df.price.rolling_past(window).count())
    ]
    
    print(f"\n使用 {window} 窗口:")
    print(f"{'操作':<10} {'时间(秒)':<10} {'速度(行/秒)':<15} {'结果样本'}")
    print("-" * 60)
    
    for op_name, op_func in operations:
        # 预热
        _ = op_func()
        
        # 计时测试
        start_time = time.time()
        result = op_func()
        elapsed = time.time() - start_time
        throughput = n / elapsed if elapsed > 0 else float('inf')
        
        # 获取结果样本
        sample_result = result.iloc[-10:].values  # 最后10个值
        sample_str = f"[{sample_result[0]:.4f}...{sample_result[-1]:.4f}]"
        
        print(f"{op_name:<10} {elapsed:<10.4f} {throughput:<15,.0f} {sample_str}")
    
    # 验证结果的正确性
    print("\n--- 验证结果正确性 ---")
    
    # 用pandas rolling做对比验证（最后5个值）
    pandas_count = df.price.rolling(window, closed='both').count()
    our_count = df.price.rolling_past(window).count()
    
    print("最后5个count值对比:")
    print(f"pandas: {pandas_count.iloc[-5:].values}")
    print(f"our:    {our_count.iloc[-5:].values}")
    print(f"差异:   {np.abs(pandas_count.iloc[-5:].values - our_count.iloc[-5:].values)}")
    
    # 内存使用情况
    print(f"\n--- 内存使用 ---")
    print(f"DataFrame内存使用: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")

if __name__ == "__main__":
    simulate_user_scenario()