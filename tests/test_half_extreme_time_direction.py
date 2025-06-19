#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import time
import sys
import os
import altair as alt
from pathlib import Path

# 添加项目根目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent.parent))

# 导入 Rust 实现
from rust_pyfunc import find_half_extreme_time as rust_find_half_extreme_time

def python_find_half_extreme_time(times: np.ndarray, prices: np.ndarray, time_window: float = 5.0, direction: str = "ignore") -> np.ndarray:
    """
    Python 版本的 find_half_extreme_time 实现，支持方向筛选
    
    参数:
        times: 时间戳数组（单位：秒）
        prices: 价格数组
        time_window: 时间窗口大小（秒）
        direction: 价格变动方向筛选
                 'pos' - 只考虑上涨方向
                 'neg' - 只考虑下跌方向
                 'ignore' - 同时考虑上涨和下跌，选择变动更大的方向（默认）
        
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
            
        max_up = 0.0      # 最大上涨幅度
        max_down = 0.0    # 最大下跌幅度
        
        # 第一次遍历：找到时间窗口内的最大上涨和下跌幅度
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
        
        # 根据方向参数确定主要方向
        if direction == "pos":
            # 只考虑上涨
            if max_up <= 0.0:
                # 没有上涨，设置为NaN
                result[i] = np.nan
                continue
            target_ratio = max_up
            dir_value = 1.0
        elif direction == "neg":
            # 只考虑下跌
            if max_down <= 0.0:
                # 没有下跌，设置为NaN
                result[i] = np.nan
                continue
            target_ratio = max_down
            dir_value = -1.0
        else:
            # 默认情况：选择变动更大的方向
            if max_up > max_down:
                target_ratio = max_up
                dir_value = 1.0
            else:
                target_ratio = max_down
                dir_value = -1.0
        
        # 如果目标变动为0，保持默认值并继续
        if target_ratio <= 0.0:
            continue
            
        half_ratio = target_ratio / 2.0 * dir_value
        found_half_time = False
            
        # 第二次遍历：找到首次达到一半变动的时间
        for j in range(i, n):
            time_diff = times[j] - current_time
            if time_diff > time_window:
                break
                
            # 跳过无效价格
            if not np.isfinite(prices[j]):
                continue
                
            # 计算当前时刻的价格变动比率
            price_ratio = (prices[j] - current_price) / current_price
            
            # 检查是否达到目标变动的一半
            if (dir_value > 0.0 and price_ratio >= half_ratio) or (dir_value < 0.0 and price_ratio <= half_ratio):
                result[i] = time_diff
                found_half_time = True
                break
    
    return result

def generate_synthetic_data(size: int = 1000, seed: int = 42) -> tuple:
    """
    生成合成测试数据
    
    参数:
        size: 数据点数量
        seed: 随机种子
        
    返回:
        tuple: (times, prices) 时间戳和价格数组
    """
    np.random.seed(seed)
    times = np.cumsum(np.random.uniform(0.1, 0.5, size))  # 随机时间间隔
    prices = 100 + np.cumsum(np.random.randn(size) * 0.5)  # 随机游走价格
    return times, prices

def test_direction_filtering():
    """测试方向筛选功能"""
    print("\n=== 方向筛选功能测试 ===\n")
    
    # 生成测试数据
    size = 1000
    times, prices = generate_synthetic_data(size)
    
    # 测试不同方向参数
    directions = ["ignore", "pos", "neg"]
    
    for direction in directions:
        print(f"\n测试方向参数: '{direction}'")
        
        # Rust实现
        rust_start = time.time()
        rust_result = rust_find_half_extreme_time(times, prices, time_window=5.0, direction=direction)
        rust_time = time.time() - rust_start
        
        # Python实现
        python_start = time.time()
        python_result = python_find_half_extreme_time(times, prices, time_window=5.0, direction=direction)
        python_time = time.time() - python_start
        
        # 计算结果一致性
        nan_mask_rust = np.isnan(rust_result)
        nan_mask_python = np.isnan(python_result)
        nan_consistency = np.array_equal(nan_mask_rust, nan_mask_python)
        
        # 对于非NaN值，计算值的一致性
        non_nan_consistency = True
        
        if nan_consistency and np.sum(~nan_mask_rust) > 0:  # 如果NaN标记一致且有非NaN值
            # 安全比较非NaN值
            rust_non_nan = np.array(rust_result)[~nan_mask_rust]
            python_non_nan = np.array(python_result)[~nan_mask_rust]  # 使用相同的掩码
            non_nan_consistency = np.allclose(
                rust_non_nan, 
                python_non_nan, 
                rtol=1e-5, 
                atol=1e-8
            )
        
        # 计算NaN值的数量
        nan_count_rust = np.sum(nan_mask_rust)
        nan_count_python = np.sum(nan_mask_python)
        
        # 输出结果
        print(f"Rust处理时间: {rust_time:.4f}秒")
        print(f"Python处理时间: {python_time:.4f}秒")
        print(f"加速比: {python_time/rust_time:.2f}倍")
        print(f"NaN值数量 - Rust: {nan_count_rust}, Python: {nan_count_python}")
        print(f"NaN位置一致性: {'一致' if nan_consistency else '不一致'}")
        print(f"非NaN值一致性: {'一致' if non_nan_consistency else '不一致'}")
        print(f"总体结果一致性: {'一致' if nan_consistency and non_nan_consistency else '不一致'}")
        
        # 如果不一致，提供更多信息
        if not (nan_consistency and non_nan_consistency):
            mismatch_indices = np.where(nan_mask_rust != nan_mask_python)[0]
            if len(mismatch_indices) > 0:
                print(f"前5个不一致的索引位置:")
                for idx in mismatch_indices[:5]:
                    print(f"  索引{idx}: Rust={rust_result[idx]}, Python={python_result[idx]}")
    
    return {
        "directions": directions,
        "consistency": [nan_consistency and non_nan_consistency for direction in directions]
    }

def test_performance_comparison():
    """比较不同方向参数的性能差异"""
    print("\n=== 方向筛选性能对比 ===\n")
    
    # 测试不同数据量
    sizes = [1000, 5000, 10000, 20000, 50000]
    directions = ["ignore", "pos", "neg"]
    
    # 存储结果
    results = {
        "sizes": sizes,
        "directions": directions,
        "rust_times": {d: [] for d in directions},
        "python_times": {d: [] for d in directions},
        "speedups": {d: [] for d in directions}
    }
    
    for size in sizes:
        print(f"\n测试数据量: {size}")
        
        # 生成测试数据
        times, prices = generate_synthetic_data(size)
        
        for direction in directions:
            # 测试 Rust 实现
            start = time.time()
            rust_find_half_extreme_time(times, prices, time_window=5.0, direction=direction)
            rust_time = time.time() - start
            results["rust_times"][direction].append(rust_time)
            
            # 测试 Python 实现
            start = time.time()
            python_find_half_extreme_time(times, prices, time_window=5.0, direction=direction)
            python_time = time.time() - start
            results["python_times"][direction].append(python_time)
            
            # 计算加速比
            speedup = python_time / rust_time
            results["speedups"][direction].append(speedup)
            
            print(f"方向参数 '{direction}': Rust耗时={rust_time:.4f}秒, Python耗时={python_time:.4f}秒, 加速比={speedup:.2f}倍")
    
    # 打印性能统计信息
    print("\n性能统计摘要:")
    for direction in directions:
        avg_speedup = sum(results["speedups"][direction]) / len(results["speedups"][direction])
        max_speedup = max(results["speedups"][direction])
        max_size_idx = results["speedups"][direction].index(max_speedup)
        
        print(f"方向参数 '{direction}':")
        print(f"  平均加速比: {avg_speedup:.2f}倍")
        print(f"  最大加速比: {max_speedup:.2f}倍 (数据量: {sizes[max_size_idx]})")
    
    return results

def visualize_performance(results):
    """可视化性能测试结果"""
    print("\n生成性能对比可视化图表...")
    
    # 准备数据
    df_list = []
    
    for direction in results["directions"]:
        for i, size in enumerate(results["sizes"]):
            df_list.append({
                "数据量": size,
                "方向参数": direction,
                "Rust耗时(秒)": results["rust_times"][direction][i],
                "Python耗时(秒)": results["python_times"][direction][i],
                "加速比": results["speedups"][direction][i]
            })
    
    df = pd.DataFrame(df_list)
    
    # 创建性能对比图表
    speedup_chart = alt.Chart(df).mark_line(point=True).encode(
        x=alt.X('数据量:Q', title='数据量'),
        y=alt.Y('加速比:Q', title='加速比 (Python时间/Rust时间)'),
        color=alt.Color('方向参数:N', title='方向参数'),
        tooltip=['数据量', '方向参数', '加速比', 'Rust耗时(秒)', 'Python耗时(秒)']
    ).properties(
        width=800,
        height=400,
        title='不同方向参数的性能加速比对比'
    ).interactive()
    
    # 准备时间对比数据
    time_df = []
    for row in df_list:
        time_df.append({
            "数据量": row["数据量"],
            "方向参数": row["方向参数"],
            "实现方式": "Rust",
            "耗时(秒)": row["Rust耗时(秒)"]
        })
        time_df.append({
            "数据量": row["数据量"],
            "方向参数": row["方向参数"],
            "实现方式": "Python",
            "耗时(秒)": row["Python耗时(秒)"]
        })
    
    time_df = pd.DataFrame(time_df)
    
    # 创建时间对比图表
    time_chart = alt.Chart(time_df).mark_line(point=True).encode(
        x=alt.X('数据量:Q', title='数据量'),
        y=alt.Y('耗时(秒):Q', title='耗时(秒)'),
        color=alt.Color('实现方式:N', title='实现方式'),
        strokeDash=alt.StrokeDash('方向参数:N', title='方向参数'),
        tooltip=['数据量', '方向参数', '实现方式', '耗时(秒)']
    ).properties(
        width=800,
        height=400,
        title='Rust与Python实现的耗时对比'
    ).interactive()
    
    # 保存图表
    combined_chart = alt.vconcat(speedup_chart, time_chart)
    combined_chart.save('/home/chenzongwei/rustcode/rust_pyfunc/tests/half_extreme_time_direction_performance.html')
    
    print("性能对比图表已保存到: half_extreme_time_direction_performance.html")

def test_specific_cases():
    """测试特定的案例，验证方向筛选的效果"""
    print("\n=== 特定案例测试 ===\n")
    
    # 案例1：明显的上涨趋势数据
    times1 = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])
    prices1 = np.array([100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0])  # 稳定上涨
    
    # 案例2：明显的下跌趋势数据
    times2 = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])
    prices2 = np.array([100.0, 99.0, 98.0, 97.0, 96.0, 95.0, 94.0, 93.0, 92.0])  # 稳定下跌
    
    # 案例3：先上涨后下跌
    times3 = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])
    prices3 = np.array([100.0, 102.0, 104.0, 103.0, 101.0, 99.0, 97.0, 96.0, 95.0])
    
    test_cases = [
        {"name": "上涨趋势", "times": times1, "prices": prices1},
        {"name": "下跌趋势", "times": times2, "prices": prices2},
        {"name": "先上涨后下跌", "times": times3, "prices": prices3}
    ]
    
    for case in test_cases:
        print(f"\n测试案例: {case['name']}")
        times = case["times"]
        prices = case["prices"]
        
        print("价格序列:", prices)
        
        for direction in ["ignore", "pos", "neg"]:
            rust_result = rust_find_half_extreme_time(times, prices, time_window=5.0, direction=direction)
            
            print(f"\n方向参数 '{direction}':")
            df = pd.DataFrame({
                "时间": times,
                "价格": prices,
                "半极端时间": rust_result
            })
            print(df)
            
            # 统计NaN值的数量
            nan_count = np.sum(np.isnan(rust_result))
            print(f"NaN值数量: {nan_count}")

if __name__ == "__main__":
    print("=== 开始测试 find_half_extreme_time 函数（带方向筛选）===")
    
    # 测试特定案例
    test_specific_cases()
    
    # 功能测试
    test_direction_filtering()
    
    # 性能测试
    results = test_performance_comparison()
    
    # 可视化结果
    visualize_performance(results)
    
    print("\n所有测试完成!")
