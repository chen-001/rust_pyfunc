#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试优化后的find_half_extreme_time函数的资源使用情况
"""

import numpy as np
import time
import sys
import os
import psutil
from pathlib import Path

# 添加项目根目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent.parent))

# 导入 Rust 实现
from rust_pyfunc import find_half_extreme_time

def test_resource_usage():
    """测试优化后函数的资源使用情况"""
    print("\n=== 测试资源使用情况 ===\n")
    
    # 生成测试数据
    size = 50000
    np.random.seed(42)
    times = np.cumsum(np.random.uniform(0.1, 0.5, size))  # 随机时间间隔
    prices = 100 + np.cumsum(np.random.randn(size) * 0.5)  # 随机游走价格
    
    print(f"数据量: {size}")
    
    # 获取当前进程
    process = psutil.Process()
    
    # 测试函数的 CPU 使用率
    print("测试优化后的实现...")
    cpu_percent_before = process.cpu_percent(interval=0.1)  # 初始化CPU测量
    memory_before = process.memory_info().rss / 1024 / 1024  # MB
    
    # 执行函数
    start_time = time.time()
    find_half_extreme_time(times, prices, time_window=60)
    elapsed_time = time.time() - start_time
    
    # 测量资源使用
    time.sleep(0.1)  # 等待一会儿以获取更准确的测量
    cpu_percent_after = process.cpu_percent(interval=0.5)
    memory_after = process.memory_info().rss / 1024 / 1024  # MB
    
    print(f"耗时: {elapsed_time:.4f} 秒")
    print(f"CPU 使用率: {cpu_percent_after:.2f}%")
    print(f"内存使用: {memory_after - memory_before:.2f} MB")
    
    # 监控CPU核心使用情况
    print("\nCPU核心数: ", psutil.cpu_count())
    print("物理CPU核心数: ", psutil.cpu_count(logical=False))
    print("逻辑CPU核心数: ", psutil.cpu_count(logical=True))
    
    # 查看CPU使用率
    per_cpu = psutil.cpu_percent(interval=0.1, percpu=True)
    print("\n各核心CPU使用率:")
    for i, usage in enumerate(per_cpu):
        print(f"核心 {i}: {usage:.2f}%", end="; ")
        if (i+1) % 5 == 0:  # 每行显示5个核心
            print()
    print("\n")
    
    # 计算活跃核心数量（使用率超过10%的核心）
    active_cores = sum(1 for usage in per_cpu if usage > 10)
    print(f"活跃核心数量 (>10%): {active_cores}")
    print(f"活跃核心占比: {active_cores/len(per_cpu)*100:.2f}%")
    
if __name__ == "__main__":
    test_resource_usage()
