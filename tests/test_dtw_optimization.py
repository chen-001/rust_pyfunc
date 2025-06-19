#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试DTW距离函数的三种实现版本，比较它们的性能和结果一致性
使用股票的逐笔成交数据进行测试
"""

import sys
import time
import pandas as pd
import numpy as np
import design_whatever as dw
from rust_pyfunc import dtw_distance, fast_dtw_distance, super_dtw_distance

def run_test(s1, s2, radius=None, timeout_seconds=None, num_runs=5):
    """
    运行三个版本的DTW距离函数并比较结果和性能
    
    参数:
    s1, s2: 输入序列
    radius: 可选的窗口半径
    timeout_seconds: 超时时间（秒）
    num_runs: 运行次数，用于计算平均性能
    
    返回:
    结果字典，包含各版本的结果和性能指标
    """
    results = {}
    
    # 原始版本测试
    original_times = []
    for _ in range(num_runs):
        start = time.time()
        original_result = dtw_distance(s1, s2, radius, timeout_seconds)
        original_times.append(time.time() - start)
    
    # 快速版本测试
    fast_times = []
    for _ in range(num_runs):
        start = time.time()
        fast_result = fast_dtw_distance(s1, s2, radius, timeout_seconds)
        fast_times.append(time.time() - start)
    
    # 超级优化版本测试
    super_times = []
    for _ in range(num_runs):
        start = time.time()
        super_result = super_dtw_distance(s1, s2, radius, timeout_seconds)
        super_times.append(time.time() - start)
    
    # 计算平均耗时
    results['original'] = {
        'result': original_result,
        'time': sum(original_times) / len(original_times)
    }
    
    results['fast'] = {
        'result': fast_result,
        'time': sum(fast_times) / len(fast_times),
        'speedup': results['original']['time'] / (sum(fast_times) / len(fast_times)) if fast_times else float('inf'),
        'diff': abs(original_result - fast_result)
    }
    
    results['super'] = {
        'result': super_result,
        'time': sum(super_times) / len(super_times),
        'speedup': results['original']['time'] / (sum(super_times) / len(super_times)) if super_times else float('inf'),
        'diff': abs(original_result - super_result)
    }
    
    return results

def test_with_stock_data():
    """使用股票的逐笔成交数据测试DTW距离函数的性能和一致性"""
    print("\n==== 使用股票逐笔成交数据测试DTW距离函数 ====")
    
    # 读取股票逐笔成交数据
    print("正在读取股票数据...")
    trade_data = dw.read_l2_trade_data(start_date=20220819, symbols=['000001'], with_retreat=0)
    
    # 检查数据是否正确读取
    if trade_data is None or len(trade_data) == 0:
        print("未能成功读取股票数据，请检查数据源")
        return
    
    print(f"成功读取股票数据，共 {len(trade_data)} 条记录")
    print(f"数据列: {trade_data.columns.tolist()}")
    
    # 从交易数据提取价格序列，选取价格列(price)作为测试数据
    prices = trade_data['price'].values
    
    # 为了测试不同长度的序列，我们从完整序列中选取不同长度的子序列
    test_lengths = [100, 500, 1000, 2000]
    
    print("\n--- 测试不同长度的价格序列 ---")
    for length in test_lengths:
        if length > len(prices):
            print(f"警告: 请求的长度 {length} 超过了可用数据的长度 {len(prices)}")
            continue
            
        # 从序列中选取两个连续但不重叠的子序列
        s1 = prices[:length]
        s2 = prices[length:2*length] if 2*length <= len(prices) else prices[length:]
        
        print(f"\n长度 = {length} 的测试:")
        
        # 无窗口限制的测试
        print("无窗口限制...")
        results = run_test(s1, s2)
        
        print(f"原始DTW结果: {results['original']['result']:.6f}, 耗时: {results['original']['time']:.6f}秒")
        print(f"快速DTW结果: {results['fast']['result']:.6f}, 耗时: {results['fast']['time']:.6f}秒")
        print(f"超级DTW结果: {results['super']['result']:.6f}, 耗时: {results['super']['time']:.6f}秒")
        print(f"快速版差异: {results['fast']['diff']:.10f}, 加速比: {results['fast']['speedup']:.2f}倍")
        print(f"超级版差异: {results['super']['diff']:.10f}, 加速比: {results['super']['speedup']:.2f}倍")
        
        # 有窗口限制的测试 (半径为序列长度的10%)
        radius = max(int(length * 0.1), 1)
        print(f"\n带窗口限制 (radius={radius})...")
        results = run_test(s1, s2, radius)
        
        print(f"原始DTW结果: {results['original']['result']:.6f}, 耗时: {results['original']['time']:.6f}秒")
        print(f"快速DTW结果: {results['fast']['result']:.6f}, 耗时: {results['fast']['time']:.6f}秒")
        print(f"超级DTW结果: {results['super']['result']:.6f}, 耗时: {results['super']['time']:.6f}秒")
        print(f"快速版差异: {results['fast']['diff']:.10f}, 加速比: {results['fast']['speedup']:.2f}倍")
        print(f"超级版差异: {results['super']['diff']:.10f}, 加速比: {results['super']['speedup']:.2f}倍")

def test_with_synthetic_data():
    """使用合成数据测试DTW距离函数的性能和一致性"""
    print("\n==== 使用合成数据测试DTW距离函数 ====")
    
    # 创建具有相似形态但有噪声的合成时间序列
    np.random.seed(0)  # 确保可重复性
    length = 1000
    t = np.linspace(0, 10, length)
    s1 = np.sin(t) + 0.1 * np.random.randn(length)
    s2 = np.sin(t + 0.2) + 0.1 * np.random.randn(length)
    
    print(f"合成数据长度: {length}")
    
    # 无窗口限制的测试
    print("\n无窗口限制...")
    results = run_test(s1, s2)
    
    print(f"原始DTW结果: {results['original']['result']:.6f}, 耗时: {results['original']['time']:.6f}秒")
    print(f"快速DTW结果: {results['fast']['result']:.6f}, 耗时: {results['fast']['time']:.6f}秒")
    print(f"超级DTW结果: {results['super']['result']:.6f}, 耗时: {results['super']['time']:.6f}秒")
    print(f"快速版差异: {results['fast']['diff']:.10f}, 加速比: {results['fast']['speedup']:.2f}倍")
    print(f"超级版差异: {results['super']['diff']:.10f}, 加速比: {results['super']['speedup']:.2f}倍")
    
    # 有窗口限制的测试
    radius = 50
    print(f"\n带窗口限制 (radius={radius})...")
    results = run_test(s1, s2, radius)
    
    print(f"原始DTW结果: {results['original']['result']:.6f}, 耗时: {results['original']['time']:.6f}秒")
    print(f"快速DTW结果: {results['fast']['result']:.6f}, 耗时: {results['fast']['time']:.6f}秒")
    print(f"超级DTW结果: {results['super']['result']:.6f}, 耗时: {results['super']['time']:.6f}秒")
    print(f"快速版差异: {results['fast']['diff']:.10f}, 加速比: {results['fast']['speedup']:.2f}倍")
    print(f"超级版差异: {results['super']['diff']:.10f}, 加速比: {results['super']['speedup']:.2f}倍")

def main():
    """主函数，运行所有测试"""
    print("开始测试DTW距离函数优化...")
    
    # 测试合成数据
    test_with_synthetic_data()
    
    # 测试股票数据
    test_with_stock_data()
    
    print("\n测试完成!")

if __name__ == "__main__":
    main()
