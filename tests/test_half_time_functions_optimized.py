#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
对优化后的 find_half_extreme_time 和 find_half_energy_time 函数进行测试
比较 Rust 版本与 Python 版本的结果一致性和性能差异
"""

import numpy as np
import pandas as pd
import time
import sys
import os
import psutil
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 添加项目根目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent.parent))

# 导入 Rust 实现
from rust_pyfunc import find_half_extreme_time, find_half_energy_time

# Python 实现的 find_half_extreme_time 函数
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

# Python 实现的 find_half_energy_time 函数
def python_find_half_energy_time(times: np.ndarray, prices: np.ndarray, time_window: float = 5.0) -> np.ndarray:
    """
    Python 版本的 find_half_energy_time 实现
    
    参数:
        times: 时间戳数组（单位：秒）
        prices: 价格数组
        time_window: 时间窗口大小（秒）
        
    返回:
        numpy.ndarray: 每个时间点达到半能量所需的时间
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
            
        final_energy = 0.0
        
        # 首先计算time_window秒后的最终能量
        for j in range(i, n):
            # 跳过当前点
            if j == i:
                continue
                
            # 检查时间间隔
            time_diff = times[j] - current_time
            if time_diff < time_window:
                continue
                
            # 检查价格是否为NaN或Inf
            if not np.isfinite(prices[j]):
                continue
                
            # 计算价格变动比率的绝对值
            final_energy = abs(prices[j] - current_price) / current_price
            break
            
        # 如果最终能量为0，设置为0并继续
        if final_energy <= 0.0:
            result[i] = 0.0
            continue
            
        # 计算一半能量的阈值
        half_energy = final_energy / 2.0
        
        # 再次遍历，找到第一次达到一半能量的时间
        for j in range(i, n):
            if j == i:
                continue
                
            time_diff = times[j] - current_time
            if time_diff > time_window:
                break
                
            # 检查价格是否为NaN或Inf
            if not np.isfinite(prices[j]):
                continue
                
            # 计算当前时刻的能量
            price_ratio = abs(prices[j] - current_price) / current_price
            
            # 如果达到一半能量
            if price_ratio >= half_energy:
                result[i] = time_diff
                break
    
    return result

def generate_test_data(size, seed=42):
    """生成测试数据"""
    np.random.seed(seed)
    times = np.cumsum(np.random.uniform(0.1, 0.5, size))  # 随机时间间隔
    prices = 100 + np.cumsum(np.random.randn(size) * 0.5)  # 随机游走价格
    return times, prices

def generate_special_test_data(size=1000, seed=42):
    """生成特殊测试数据，包含极端情况"""
    np.random.seed(seed)
    times = np.cumsum(np.random.uniform(0.1, 0.5, size))
    
    # 基础价格
    prices = 100 + np.cumsum(np.random.randn(size) * 0.5)
    
    # 添加一些极端价格变动
    for i in range(0, size, 50):
        if np.random.random() > 0.5:
            # 正向跳跃
            jump_idx = min(i + np.random.randint(1, 10), size - 1)
            prices[jump_idx:] += np.random.uniform(2.0, 5.0)
        else:
            # 负向跳跃
            jump_idx = min(i + np.random.randint(1, 10), size - 1)
            prices[jump_idx:] -= np.random.uniform(2.0, 5.0)
    
    # 添加一些无效值
    invalid_indices = np.random.choice(size, size // 20, replace=False)
    for idx in invalid_indices:
        if np.random.random() > 0.5:
            prices[idx] = np.nan
        else:
            prices[idx] = np.inf
    
    return times, prices

def test_extreme_correctness():
    """测试 find_half_extreme_time 函数结果正确性"""
    print("\n=== 测试 find_half_extreme_time 结果正确性 ===\n")
    
    # 生成不同规模的测试数据
    test_sizes = [100, 1000, 5000]
    
    all_correct = True
    
    for size in test_sizes:
        print(f"\n测试数据量: {size}")
        times, prices = generate_test_data(size)
        
        # 运行 Rust 和 Python 实现
        rust_result = find_half_extreme_time(times, prices, time_window=10.0)
        python_result = python_find_half_extreme_time(times, prices, time_window=10.0)
        
        # 比较结果
        is_close = np.allclose(rust_result, python_result, equal_nan=True)
        max_diff = np.nanmax(np.abs(np.array(rust_result) - python_result))
        
        print(f"结果一致: {is_close}")
        print(f"最大差异: {max_diff}")
        
        all_correct = all_correct and is_close
        
        # 如果结果不一致，打印详细信息
        if not is_close:
            # 找出差异较大的位置
            diff_indices = np.where(np.abs(np.array(rust_result) - python_result) > 1e-10)[0]
            if len(diff_indices) > 0:
                print(f"差异较大的位置数量: {len(diff_indices)}")
                for idx in diff_indices[:5]:  # 只打印前5个不同的结果
                    print(f"位置 {idx}: Rust={rust_result[idx]}, Python={python_result[idx]}")
    
    # 测试特殊情况
    print("\n测试特殊情况")
    times, prices = generate_special_test_data()
    
    # 运行 Rust 和 Python 实现
    rust_result = find_half_extreme_time(times, prices, time_window=10.0)
    python_result = python_find_half_extreme_time(times, prices, time_window=10.0)
    
    # 比较结果
    is_close = np.allclose(rust_result, python_result, equal_nan=True)
    
    # 对于NaN值的处理可能不同，所以单独检查
    nan_mask_rust = np.isnan(rust_result)
    nan_mask_python = np.isnan(python_result)
    nan_match = np.array_equal(nan_mask_rust, nan_mask_python)
    
    print(f"特殊情况结果一致: {is_close}")
    print(f"NaN处理一致: {nan_match}")
    
    all_correct = all_correct and is_close and nan_match
    
    print(f"\n总体结果: {'正确' if all_correct else '存在差异'}")
    
    return all_correct

def test_energy_correctness():
    """测试 find_half_energy_time 函数结果正确性"""
    print("\n=== 测试 find_half_energy_time 结果正确性 ===\n")
    
    # 生成不同规模的测试数据
    test_sizes = [100, 1000, 5000]
    
    all_correct = True
    
    for size in test_sizes:
        print(f"\n测试数据量: {size}")
        times, prices = generate_test_data(size)
        
        # 运行 Rust 和 Python 实现
        rust_result = find_half_energy_time(times, prices, time_window=10.0)
        python_result = python_find_half_energy_time(times, prices, time_window=10.0)
        
        # 比较结果
        is_close = np.allclose(rust_result, python_result, equal_nan=True)
        max_diff = np.nanmax(np.abs(np.array(rust_result) - python_result))
        
        print(f"结果一致: {is_close}")
        print(f"最大差异: {max_diff}")
        
        all_correct = all_correct and is_close
        
        # 如果结果不一致，打印详细信息
        if not is_close:
            # 找出差异较大的位置
            diff_indices = np.where(np.abs(np.array(rust_result) - python_result) > 1e-10)[0]
            if len(diff_indices) > 0:
                print(f"差异较大的位置数量: {len(diff_indices)}")
                for idx in diff_indices[:5]:  # 只打印前5个不同的结果
                    print(f"位置 {idx}: Rust={rust_result[idx]}, Python={python_result[idx]}")
    
    # 测试特殊情况
    print("\n测试特殊情况")
    times, prices = generate_special_test_data()
    
    # 运行 Rust 和 Python 实现
    rust_result = find_half_energy_time(times, prices, time_window=10.0)
    python_result = python_find_half_energy_time(times, prices, time_window=10.0)
    
    # 比较结果
    is_close = np.allclose(rust_result, python_result, equal_nan=True)
    
    # 对于NaN值的处理可能不同，所以单独检查
    nan_mask_rust = np.isnan(rust_result)
    nan_mask_python = np.isnan(python_result)
    nan_match = np.array_equal(nan_mask_rust, nan_mask_python)
    
    print(f"特殊情况结果一致: {is_close}")
    print(f"NaN处理一致: {nan_match}")
    
    all_correct = all_correct and is_close and nan_match
    
    print(f"\n总体结果: {'正确' if all_correct else '存在差异'}")
    
    return all_correct

def test_performance():
    """测试函数性能并绘制比较图表"""
    print("\n=== 性能比较测试 ===\n")
    
    # 测试不同数据量
    sizes = [1000, 5000, 10000, 20000, 50000]
    rust_extreme_times = []
    python_extreme_times = []
    rust_energy_times = []
    python_energy_times = []
    
    # 进行性能测试
    for size in sizes:
        print(f"\n测试数据量: {size}")
        times, prices = generate_test_data(size)
        
        # 测试 find_half_extreme_time
        print("测试 find_half_extreme_time:")
        
        # 测试 Rust 实现
        start = time.time()
        find_half_extreme_time(times, prices, time_window=60)
        rust_time = time.time() - start
        rust_extreme_times.append(rust_time)
        print(f"Rust 耗时: {rust_time:.4f} 秒")
        
        # 测试 Python 实现
        start = time.time()
        python_find_half_extreme_time(times, prices, time_window=60)
        python_time = time.time() - start
        python_extreme_times.append(python_time)
        print(f"Python 耗时: {python_time:.4f} 秒")
        
        # 计算加速比
        speedup = python_time / rust_time
        print(f"加速比: {speedup:.2f} 倍")
        
        # 测试 find_half_energy_time
        print("\n测试 find_half_energy_time:")
        
        # 测试 Rust 实现
        start = time.time()
        find_half_energy_time(times, prices, time_window=60)
        rust_time = time.time() - start
        rust_energy_times.append(rust_time)
        print(f"Rust 耗时: {rust_time:.4f} 秒")
        
        # 测试 Python 实现
        start = time.time()
        python_find_half_energy_time(times, prices, time_window=60)
        python_time = time.time() - start
        python_energy_times.append(python_time)
        print(f"Python 耗时: {python_time:.4f} 秒")
        
        # 计算加速比
        speedup = python_time / rust_time
        print(f"加速比: {speedup:.2f} 倍")
    
    # 计算加速比
    extreme_speedups = [py/ru for py, ru in zip(python_extreme_times, rust_extreme_times)]
    energy_speedups = [py/ru for py, ru in zip(python_energy_times, rust_energy_times)]
    
    # 打印性能统计信息
    print("\n===== 性能统计信息 =====")
    print(f"数据量: {sizes}")
    
    print("\nfind_half_extreme_time:")
    print(f"Rust 耗时 (秒): {[f'{t:.4f}' for t in rust_extreme_times]}")
    print(f"Python 耗时 (秒): {[f'{t:.4f}' for t in python_extreme_times]}")
    print(f"加速比: {[f'{s:.2f}' for s in extreme_speedups]}")
    print(f"平均加速比: {sum(extreme_speedups)/len(extreme_speedups):.2f} 倍")
    print(f"最大加速比: {max(extreme_speedups):.2f} 倍 (数据量: {sizes[extreme_speedups.index(max(extreme_speedups))]})")
    
    print("\nfind_half_energy_time:")
    print(f"Rust 耗时 (秒): {[f'{t:.4f}' for t in rust_energy_times]}")
    print(f"Python 耗时 (秒): {[f'{t:.4f}' for t in python_energy_times]}")
    print(f"加速比: {[f'{s:.2f}' for s in energy_speedups]}")
    print(f"平均加速比: {sum(energy_speedups)/len(energy_speedups):.2f} 倍")
    print(f"最大加速比: {max(energy_speedups):.2f} 倍 (数据量: {sizes[energy_speedups.index(max(energy_speedups))]})")
    
    # 生成可视化比较图表
    try:
        # 创建子图
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "find_half_extreme_time 执行时间", 
                "find_half_extreme_time 加速比",
                "find_half_energy_time 执行时间", 
                "find_half_energy_time 加速比"
            ),
            specs=[[{"type": "scatter"}, {"type": "bar"}],
                   [{"type": "scatter"}, {"type": "bar"}]]
        )
        
        # find_half_extreme_time 执行时间
        fig.add_trace(
            go.Scatter(x=sizes, y=rust_extreme_times, mode='lines+markers', name='Rust half_extreme_time'),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=sizes, y=python_extreme_times, mode='lines+markers', name='Python half_extreme_time'),
            row=1, col=1
        )
        
        # find_half_extreme_time 加速比
        fig.add_trace(
            go.Bar(x=sizes, y=extreme_speedups, name='half_extreme_time 加速比'),
            row=1, col=2
        )
        
        # find_half_energy_time 执行时间
        fig.add_trace(
            go.Scatter(x=sizes, y=rust_energy_times, mode='lines+markers', name='Rust half_energy_time'),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=sizes, y=python_energy_times, mode='lines+markers', name='Python half_energy_time'),
            row=2, col=1
        )
        
        # find_half_energy_time 加速比
        fig.add_trace(
            go.Bar(x=sizes, y=energy_speedups, name='half_energy_time 加速比'),
            row=2, col=2
        )
        
        # 更新布局
        fig.update_layout(
            title_text="Rust 函数性能比较",
            height=800,
            width=1000
        )
        
        # 更新 Y 轴标签
        fig.update_yaxes(title_text="执行时间 (秒)", row=1, col=1)
        fig.update_yaxes(title_text="加速比", row=1, col=2)
        fig.update_yaxes(title_text="执行时间 (秒)", row=2, col=1)
        fig.update_yaxes(title_text="加速比", row=2, col=2)
        
        # 更新 X 轴标签
        fig.update_xaxes(title_text="数据量", row=1, col=1)
        fig.update_xaxes(title_text="数据量", row=1, col=2)
        fig.update_xaxes(title_text="数据量", row=2, col=1)
        fig.update_xaxes(title_text="数据量", row=2, col=2)
        
        # 保存图表
        fig.write_html('/home/chenzongwei/rustcode/rust_pyfunc/tests/half_time_functions_performance.html')
        print("\n性能比较图表已保存至: half_time_functions_performance.html")
    except Exception as e:
        print(f"\n生成图表时出错: {e}")

def test_resource_usage():
    """测试资源使用情况"""
    print("\n=== 测试资源使用情况 ===\n")
    
    # 生成大型测试数据
    size = 50000
    times, prices = generate_test_data(size)
    
    print(f"数据量: {size}")
    
    try:
        import psutil
        process = psutil.Process()
        
        # 测试 find_half_extreme_time 函数的资源使用
        print("\n测试 find_half_extreme_time:")
        process.cpu_percent(interval=0.1)  # 初始化CPU测量
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        start_time = time.time()
        find_half_extreme_time(times, prices, time_window=60)
        elapsed_time = time.time() - start_time
        
        # 测量资源使用
        time.sleep(0.1)  # 等待一会儿以获取更准确的测量
        cpu_percent = process.cpu_percent(interval=0.1)
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        
        print(f"耗时: {elapsed_time:.4f} 秒")
        print(f"CPU 使用率: {cpu_percent:.2f}%")
        print(f"内存使用增加: {memory_after - memory_before:.2f} MB")
        
        # 显示CPU核心使用情况
        per_cpu = psutil.cpu_percent(interval=0.1, percpu=True)
        active_cores = sum(1 for usage in per_cpu if usage > 10)
        print(f"活跃核心数量 (>10%): {active_cores}")
        print(f"活跃核心占比: {active_cores/len(per_cpu)*100:.2f}%")
        
        # 让系统休息一下
        time.sleep(1)
        
        # 测试 find_half_energy_time 函数的资源使用
        print("\n测试 find_half_energy_time:")
        process.cpu_percent(interval=0.1)  # 初始化CPU测量
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        start_time = time.time()
        find_half_energy_time(times, prices, time_window=60)
        elapsed_time = time.time() - start_time
        
        # 测量资源使用
        time.sleep(0.1)  # 等待一会儿以获取更准确的测量
        cpu_percent = process.cpu_percent(interval=0.1)
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        
        print(f"耗时: {elapsed_time:.4f} 秒")
        print(f"CPU 使用率: {cpu_percent:.2f}%")
        print(f"内存使用增加: {memory_after - memory_before:.2f} MB")
        
        # 显示CPU核心使用情况
        per_cpu = psutil.cpu_percent(interval=0.1, percpu=True)
        active_cores = sum(1 for usage in per_cpu if usage > 10)
        print(f"活跃核心数量 (>10%): {active_cores}")
        print(f"活跃核心占比: {active_cores/len(per_cpu)*100:.2f}%")
        
    except ImportError:
        print("未安装 psutil，跳过资源使用测试")
        
        # 仍然运行基本的时间测试
        start_time = time.time()
        find_half_extreme_time(times, prices, time_window=60)
        elapsed_time = time.time() - start_time
        print(f"\nfind_half_extreme_time 耗时: {elapsed_time:.4f} 秒")
        
        start_time = time.time()
        find_half_energy_time(times, prices, time_window=60)
        elapsed_time = time.time() - start_time
        print(f"find_half_energy_time 耗时: {elapsed_time:.4f} 秒")

if __name__ == "__main__":
    print("=== 开始测试优化后的 find_half_extreme_time 和 find_half_energy_time 函数 ===")
    
    # 测试函数结果正确性
    extreme_correct = test_extreme_correctness()
    energy_correct = test_energy_correctness()
    
    if extreme_correct and energy_correct:
        print("\n结果验证通过，开始性能测试...")
        # 测试性能
        test_performance()
        
        # 测试资源使用情况
        test_resource_usage()
    else:
        print("\n结果验证未通过，需要修正函数实现!")
