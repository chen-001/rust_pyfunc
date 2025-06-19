#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
对优化后的 find_half_extreme_time 函数进行测试
比较修改后的 Rust 版本与 Python 版本的结果一致性和性能差异
"""

import numpy as np
import pandas as pd
import time
import sys
import os
import multiprocessing as mp
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
            time_diff = times[j] - current_time
            if time_diff > time_window:
                break
                
            # 跳过无效价格
            if not np.isfinite(prices[j]):
                continue
                
            # 计算价格变动比率
            price_ratio = (prices[j] - current_price) / current_price
            
            # 更新最大上涨和下跌幅度
            if price_ratio > max_up:
                max_up = price_ratio
            elif price_ratio < -max_down:
                max_down = -price_ratio
        
        # 确定主要方向（上涨或下跌）
        if max_up > max_down:
            target_ratio = max_up
            direction = 1.0  # 上涨
        else:
            target_ratio = max_down
            direction = -1.0  # 下跌
        
        # 如果目标变动为0，则保持默认值
        if target_ratio <= 0.0:
            continue
        
        half_ratio = target_ratio / 2.0 * direction
        
        # 查找首次达到目标变化的时间
        for j in range(i, n):
            time_diff = times[j] - current_time
            if time_diff > time_window:
                break
                
            # 跳过无效价格
            if not np.isfinite(prices[j]):
                continue
                
            price_ratio = (prices[j] - current_price) / current_price
            
            # 检查是否达到目标变动的一半
            if (direction > 0.0 and price_ratio >= half_ratio) or \
               (direction < 0.0 and price_ratio <= half_ratio):
                result[i] = time_diff
                break
    
    return result

# 定义为全局函数，以便多进程可以序列化
def _process_chunk_half_extreme(start_idx, end_idx, times, prices, time_window, n):
    result = np.full(end_idx - start_idx, time_window, dtype=np.float64)
    
    for local_i, i in enumerate(range(start_idx, end_idx)):
        current_time = times[i]
        current_price = prices[i]
        
        # 处理NaN和Inf值
        if not np.isfinite(current_price):
            result[local_i] = np.nan
            continue
            
        max_up = 0.0
        max_down = 0.0
        
        # 首先找到时间窗口内的最大上涨和下跌幅度
        for j in range(i, n):
            time_diff = times[j] - current_time
            if time_diff > time_window:
                break
                
            # 跳过无效价格
            if not np.isfinite(prices[j]):
                continue
                
            # 计算价格变动比率
            price_ratio = (prices[j] - current_price) / current_price
            
            # 更新最大上涨和下跌幅度
            if price_ratio > max_up:
                max_up = price_ratio
            elif price_ratio < -max_down:
                max_down = -price_ratio
        
        # 确定主要方向（上涨或下跌）
        if max_up > max_down:
            target_ratio = max_up
            direction = 1.0  # 上涨
        else:
            target_ratio = max_down
            direction = -1.0  # 下跌
        
        # 如果目标变动为0，则保持默认值
        if target_ratio <= 0.0:
            continue
        
        half_ratio = target_ratio / 2.0 * direction
        
        # 查找首次达到目标变化的时间
        for j in range(i, n):
            time_diff = times[j] - current_time
            if time_diff > time_window:
                break
                
            # 跳过无效价格
            if not np.isfinite(prices[j]):
                continue
                
            price_ratio = (prices[j] - current_price) / current_price
            
            # 检查是否达到目标变动的一半
            if (direction > 0.0 and price_ratio >= half_ratio) or \
               (direction < 0.0 and price_ratio <= half_ratio):
                result[local_i] = time_diff
                break
    
    return start_idx, result

def python_find_half_extreme_time_parallel(times: np.ndarray, prices: np.ndarray, time_window: float = 5.0) -> np.ndarray:
    """
    Python 版本的 find_half_extreme_time 多进程并行实现
    
    参数:
        times: 时间戳数组（单位：秒）
        prices: 价格数组
        time_window: 时间窗口大小（秒）
        
    返回:
        numpy.ndarray: 每个时间点达到最大变动一半所需的时间
    """
    n = len(times)
    
    # 将数据分成多个块进行处理
    cpu_count = mp.cpu_count()
    chunk_size = max(1, n // cpu_count)
    chunks = [(i, min(i + chunk_size, n), times, prices, time_window, n) for i in range(0, n, chunk_size)]
    
    # 使用进程池并行处理
    with mp.Pool(processes=cpu_count) as pool:
        results = pool.starmap(_process_chunk_half_extreme, chunks)
    
    # 组合结果
    final_result = np.full(n, time_window, dtype=np.float64)
    for start_idx, chunk_result in results:
        final_result[start_idx:start_idx + len(chunk_result)] = chunk_result
    
    return final_result

def generate_test_data(size, seed=42):
    """生成测试数据"""
    np.random.seed(seed)
    times = np.cumsum(np.random.uniform(0.1, 0.5, size))  # 随机时间间隔
    prices = 100 + np.cumsum(np.random.randn(size) * 0.5)  # 随机游走价格
    return times, prices

def test_correctness():
    """测试结果正确性"""
    print("\n=== 测试结果正确性 ===\n")
    
    # 生成不同规模的测试数据
    test_sizes = [100, 1000, 5000]
    
    for size in test_sizes:
        print(f"\n测试数据量: {size}")
        times, prices = generate_test_data(size)
        
        # 运行 Rust 和 Python 实现
        rust_result = rust_find_half_extreme_time(times, prices, time_window=10.0)
        python_result = python_find_half_extreme_time(times, prices, time_window=10.0)
        
        # 比较结果
        is_close = np.allclose(rust_result, python_result, equal_nan=True)
        max_diff = np.nanmax(np.abs(np.array(rust_result) - python_result))
        
        print(f"结果一致: {is_close}")
        print(f"最大差异: {max_diff}")
        
        # 如果结果不一致，打印详细信息
        if not is_close:
            # 找出差异较大的位置
            diff_indices = np.where(np.abs(np.array(rust_result) - python_result) > 1e-10)[0]
            if len(diff_indices) > 0:
                print(f"差异较大的位置数量: {len(diff_indices)}")
                for idx in diff_indices[:5]:  # 只打印前5个不同的结果
                    print(f"位置 {idx}: Rust={rust_result[idx]}, Python={python_result[idx]}")

def test_performance_with_parallel():
    """
    测试不同实现的性能并绘制比较图表
    比较: 
    1. 优化后的 Rust 实现
    2. Python 单线程实现
    3. Python 多进程实现
    """
    print("\n=== 性能比较测试 ===\n")
    
    # 测试不同数据量
    sizes = [1000, 5000, 10000, 20000, 50000]
    rust_times = []
    python_times = []
    python_parallel_times = []
    
    for size in sizes:
        print(f"\n测试数据量: {size}")
        times, prices = generate_test_data(size)
        
        # 测试 Rust 实现
        start = time.time()
        rust_find_half_extreme_time(times, prices, time_window=60)
        rust_time = time.time() - start
        rust_times.append(rust_time)
        print(f"Rust 耗时: {rust_time:.4f} 秒")
        
        # 测试 Python 单线程实现
        start = time.time()
        python_find_half_extreme_time(times, prices, time_window=60)
        python_time = time.time() - start
        python_times.append(python_time)
        print(f"Python 单线程耗时: {python_time:.4f} 秒")
        
        # 测试 Python 多进程实现
        if size <= 20000:  # 限制大型数据集的多进程测试，以避免内存问题
            start = time.time()
            python_find_half_extreme_time_parallel(times, prices, time_window=60)
            python_parallel_time = time.time() - start
            python_parallel_times.append(python_parallel_time)
            print(f"Python 多进程耗时: {python_parallel_time:.4f} 秒")
            print(f"Python 多进程与 Rust 比较: Rust 快 {python_parallel_time / rust_time:.2f} 倍")
        else:
            python_parallel_times.append(None)
            print("Python 多进程跳过 (数据量过大)")
        
        # 计算加速比
        rust_vs_python = python_time / rust_time
        print(f"Rust 比 Python 单线程快: {rust_vs_python:.2f} 倍")
    
    # 打印性能统计信息
    print("\n性能统计信息:")
    print(f"数据量: {sizes}")
    print(f"Rust 耗时 (秒): {[f'{t:.4f}' for t in rust_times]}")
    print(f"Python 单线程耗时 (秒): {[f'{t:.4f}' for t in python_times]}")
    print(f"Python 多进程耗时 (秒): {[f'{t if t is not None else None}' for t in python_parallel_times]}")
    
    # 计算平均加速比
    speedups = [py/ru for py, ru in zip(python_times, rust_times)]
    print(f"Rust 比 Python 单线程平均加速比: {sum(speedups)/len(speedups):.2f} 倍")
    print(f"最大加速比: {max(speedups):.2f} 倍 (数据量: {sizes[speedups.index(max(speedups))]})")

    # 生成可视化比较图表
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        # 创建性能比较图表
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("执行时间比较", "加速比"),
            specs=[[{"type": "scatter"}, {"type": "bar"}]]
        )
        
        # 执行时间比较
        fig.add_trace(
            go.Scatter(x=sizes, y=rust_times, mode='lines+markers', name='Rust 实现'),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=sizes, y=python_times, mode='lines+markers', name='Python 单线程'),
            row=1, col=1
        )
        
        # 添加 Python 多进程数据点 (忽略 None 值)
        valid_sizes = [s for s, t in zip(sizes, python_parallel_times) if t is not None]
        valid_times = [t for t in python_parallel_times if t is not None]
        if valid_sizes:
            fig.add_trace(
                go.Scatter(x=valid_sizes, y=valid_times, mode='lines+markers', name='Python 多进程'),
                row=1, col=1
            )
        
        # 加速比
        fig.add_trace(
            go.Bar(x=sizes, y=speedups, name='Rust 比 Python 单线程加速比'),
            row=1, col=2
        )
        
        # 更新布局
        fig.update_layout(
            title_text="find_half_extreme_time 性能比较",
            height=500,
            width=1000
        )
        
        # 更新 Y 轴标签
        fig.update_yaxes(title_text="执行时间 (秒)", row=1, col=1)
        fig.update_yaxes(title_text="加速比", row=1, col=2)
        fig.update_xaxes(title_text="数据量", row=1, col=1)
        fig.update_xaxes(title_text="数据量", row=1, col=2)
        
        # 保存图表
        fig.write_html('/home/chenzongwei/rustcode/rust_pyfunc/tests/half_extreme_time_performance.html')
        print("\n性能比较图表已保存至: half_extreme_time_performance.html")
    except ImportError:
        print("\n未安装 plotly，跳过生成可视化图表")

def test_resource_usage():
    """测试资源使用情况"""
    print("\n=== 测试资源使用情况 ===\n")
    
    # 生成大型测试数据以观察资源使用
    size = 50000
    times, prices = generate_test_data(size)
    
    print(f"数据量: {size}")
    
    try:
        import psutil
        process = psutil.Process()
        
        # 测试 Rust 函数的 CPU 使用率
        print("测试 Rust 实现...")
        cpu_percent_before = process.cpu_percent(interval=None)
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        start_time = time.time()
        rust_find_half_extreme_time(times, prices, time_window=60)
        rust_time = time.time() - start_time
        
        # 暂停一下以获取更精确的 CPU 使用率
        time.sleep(0.1)
        cpu_percent_after = process.cpu_percent(interval=0.1)
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        
        print(f"Rust 耗时: {rust_time:.4f} 秒")
        print(f"Rust CPU 使用率变化: {cpu_percent_after - cpu_percent_before:.2f}%")
        print(f"Rust 内存使用变化: {memory_after - memory_before:.2f} MB")
        
        # 等待一会儿确保系统资源恢复
        time.sleep(1)
        
        # 测试 Python 单线程函数的 CPU 使用率
        print("\n测试 Python 单线程实现...")
        cpu_percent_before = process.cpu_percent(interval=None)
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        start_time = time.time()
        python_find_half_extreme_time(times[:10000], prices[:10000], time_window=60)  # 使用较小的数据集以节省时间
        python_time = time.time() - start_time
        
        # 暂停一下以获取更精确的 CPU 使用率
        time.sleep(0.1)
        cpu_percent_after = process.cpu_percent(interval=0.1)
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        
        # 预估完整数据集的时间
        estimated_python_time = python_time * (size / 10000)
        
        print(f"Python 耗时 (10000 数据点): {python_time:.4f} 秒")
        print(f"Python 预估耗时 ({size} 数据点): {estimated_python_time:.4f} 秒")
        print(f"Python CPU 使用率变化: {cpu_percent_after - cpu_percent_before:.2f}%")
        print(f"Python 内存使用变化: {memory_after - memory_before:.2f} MB")
        
    except ImportError:
        print("未安装 psutil，跳过资源使用测试")
        
        # 仍然运行基本的时间测试
        start_time = time.time()
        rust_find_half_extreme_time(times, prices, time_window=60)
        rust_time = time.time() - start_time
        print(f"Rust 耗时: {rust_time:.4f} 秒")
        
        # 使用较小的数据集测试 Python 版本
        start_time = time.time()
        python_find_half_extreme_time(times[:10000], prices[:10000], time_window=60)
        python_time = time.time() - start_time
        estimated_python_time = python_time * (size / 10000)
        print(f"Python 耗时 (10000 数据点): {python_time:.4f} 秒")
        print(f"Python 预估耗时 ({size} 数据点): {estimated_python_time:.4f} 秒")

if __name__ == "__main__":
    print("=== 开始测试优化后的 find_half_extreme_time 函数 ===")
    
    # 测试修改后 Rust 函数结果正确性
    test_correctness()
    
    # 测试性能和资源使用情况
    test_performance_with_parallel()
    test_resource_usage()
