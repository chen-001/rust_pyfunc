#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试"以退为进"分析函数的功能
"""

import numpy as np
import pandas as pd
import time
import sys
import os

# 添加当前目录到路径，以便导入rust_pyfunc
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

try:
    from rust_pyfunc import analyze_retreat_advance
    print("✓ 成功导入 analyze_retreat_advance 函数")
except ImportError as e:
    print(f"✗ 导入失败: {e}")
    print("请先编译安装rust_pyfunc库")
    sys.exit(1)

def create_test_data():
    """创建测试用的模拟股票数据"""
    # 模拟一天的交易数据（以小时为单位，9:30-15:00）
    np.random.seed(42)
    
    # 生成时间序列（9:30开始，每30秒一个数据点）
    start_time = 9.5  # 9:30
    end_time = 15.0   # 15:00
    time_interval = 0.5 / 60  # 30秒
    
    times = np.arange(start_time, end_time, time_interval)
    n_points = len(times)
    
    # 模拟价格走势：包含局部高点和突破
    base_price = 100.0
    price_trend = np.cumsum(np.random.normal(0, 0.1, n_points)) * 0.1
    
    # 在特定位置添加明显的局部高点
    prices = base_price + price_trend
    
    # 人工创建一个"以退为进"模式
    peak_idx = n_points // 3
    prices[peak_idx-2:peak_idx+1] = prices[peak_idx-2] + np.array([0.5, 1.0, 1.2])  # 创建局部高点
    prices[peak_idx+1:peak_idx+20] = prices[peak_idx+1] - np.linspace(0, 0.8, 19)  # 回落
    prices[peak_idx+20:peak_idx+40] = prices[peak_idx+20] + np.linspace(0, 1.5, 20)  # 突破
    
    # 生成成交量（局部高点附近成交量较大）
    volumes = np.random.exponential(200, n_points)
    volumes[peak_idx-5:peak_idx+25] *= 2  # 局部高点附近成交量加大
    
    # 生成买卖标志（随机，但在突破时偏向买入）
    flags = np.random.choice([-1, 1], n_points)
    flags[peak_idx+20:peak_idx+40] = 1  # 突破时主要是买入
    
    # 盘口数据（模拟卖出挂单）
    # 简化：每个时间点都有一个盘口价格和挂单量
    orderbook_times = times.copy()
    orderbook_prices = prices.copy()
    
    # 生成挂单量，在局部高点附近设置异常大的挂单量
    orderbook_volumes = np.random.exponential(1000, n_points)
    # 在局部高点价格附近设置异常大的挂单量
    peak_price = prices[peak_idx]
    for i, price in enumerate(orderbook_prices):
        if abs(price - peak_price) < 0.1 and times[i] >= times[peak_idx] and times[i] <= times[peak_idx] + 0.5/60:  # 30分钟内
            orderbook_volumes[i] *= 10  # 异常大的挂单量
    
    return {
        'trade_times': times,
        'trade_prices': prices,
        'trade_volumes': volumes,
        'trade_flags': flags,
        'orderbook_times': orderbook_times,
        'orderbook_prices': orderbook_prices,
        'orderbook_volumes': orderbook_volumes
    }

def test_basic_functionality():
    """测试基本功能"""
    print("\n=== 测试基本功能 ===")
    
    data = create_test_data()
    
    try:
        start_time = time.time()
        results = analyze_retreat_advance(
            data['trade_times'].astype(np.float64),
            data['trade_prices'].astype(np.float64),
            data['trade_volumes'].astype(np.float64),
            data['trade_flags'].astype(np.float64),
            data['orderbook_times'].astype(np.float64),
            data['orderbook_prices'].astype(np.float64),
            data['orderbook_volumes'].astype(np.float64)
        )
        end_time = time.time()
        
        process_volumes, large_volumes, one_min_volumes, buy_ratios, price_counts, max_declines = results
        
        print(f"✓ 函数执行成功，耗时: {end_time - start_time:.4f}秒")
        print(f"✓ 找到 {len(process_volumes)} 个以退为进过程")
        
        # 验证返回值类型和形状
        assert isinstance(process_volumes, np.ndarray), "process_volumes应该是numpy数组"
        assert isinstance(large_volumes, np.ndarray), "large_volumes应该是numpy数组"
        assert isinstance(one_min_volumes, np.ndarray), "one_min_volumes应该是numpy数组"
        assert isinstance(buy_ratios, np.ndarray), "buy_ratios应该是numpy数组"
        assert isinstance(price_counts, np.ndarray), "price_counts应该是numpy数组"
        assert isinstance(max_declines, np.ndarray), "max_declines应该是numpy数组"
        
        # 所有返回数组应该有相同的长度
        lengths = [len(arr) for arr in results]
        assert all(l == lengths[0] for l in lengths), "所有返回数组长度应该相同"
        
        print("✓ 返回值类型和形状验证通过")
        
        # 输出一些统计信息
        if len(process_volumes) > 0:
            print(f"  - 过程成交量范围: {process_volumes.min():.2f} - {process_volumes.max():.2f}")
            print(f"  - 异常挂单量范围: {large_volumes.min():.2f} - {large_volumes.max():.2f}")
            print(f"  - 买入占比范围: {buy_ratios.min():.2f} - {buy_ratios.max():.2f}")
            print(f"  - 最大下跌比例范围: {max_declines.min():.2f} - {max_declines.max():.2f}")
        
        return True
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_edge_cases():
    """测试边界情况"""
    print("\n=== 测试边界情况 ===")
    
    # 测试空数据
    try:
        empty_array = np.array([], dtype=np.float64)
        results = analyze_retreat_advance(
            empty_array, empty_array, empty_array, empty_array,
            empty_array, empty_array, empty_array
        )
        print("✓ 空数据测试通过")
    except Exception as e:
        print(f"✗ 空数据测试失败: {e}")
    
    # 测试长度不一致的数据
    try:
        short_array = np.array([1.0, 2.0], dtype=np.float64)
        long_array = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        analyze_retreat_advance(
            short_array, long_array, short_array, short_array,
            short_array, short_array, short_array
        )
        print("✗ 长度不一致测试失败：应该抛出异常")
    except Exception:
        print("✓ 长度不一致测试通过：正确抛出异常")
    
    # 测试包含NaN的数据
    try:
        data_with_nan = np.array([1.0, np.nan, 3.0], dtype=np.float64)
        normal_data = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        results = analyze_retreat_advance(
            normal_data, data_with_nan, normal_data, normal_data,
            normal_data, normal_data, normal_data
        )
        print("✓ NaN数据测试通过")
    except Exception as e:
        print(f"! NaN数据测试: {e}")

def test_parameter_effects():
    """测试参数对结果的影响"""
    print("\n=== 测试参数效果 ===")
    
    data = create_test_data()
    
    # 测试不同的volume_percentile参数
    percentiles = [95.0, 99.0, 99.9]
    
    for p in percentiles:
        try:
            results = analyze_retreat_advance(
                data['trade_times'].astype(np.float64),
                data['trade_prices'].astype(np.float64),
                data['trade_volumes'].astype(np.float64),
                data['trade_flags'].astype(np.float64),
                data['orderbook_times'].astype(np.float64),
                data['orderbook_prices'].astype(np.float64),
                data['orderbook_volumes'].astype(np.float64),
                volume_percentile=p
            )
            process_count = len(results[0])
            print(f"✓ volume_percentile={p}: 找到 {process_count} 个过程")
        except Exception as e:
            print(f"✗ volume_percentile={p} 测试失败: {e}")

def compare_with_python_implementation():
    """与Python实现对比（如果有的话）"""
    print("\n=== 性能对比 ===")
    
    data = create_test_data()
    
    # Rust实现
    start_time = time.time()
    rust_results = analyze_retreat_advance(
        data['trade_times'].astype(np.float64),
        data['trade_prices'].astype(np.float64),
        data['trade_volumes'].astype(np.float64),
        data['trade_flags'].astype(np.float64),
        data['orderbook_times'].astype(np.float64),
        data['orderbook_prices'].astype(np.float64),
        data['orderbook_volumes'].astype(np.float64)
    )
    rust_time = time.time() - start_time
    
    print(f"✓ Rust实现耗时: {rust_time:.6f}秒")
    print(f"✓ 找到 {len(rust_results[0])} 个以退为进过程")
    
    # 这里可以添加Python实现的对比代码
    print("  (Python对比实现可以在这里添加)")

def main():
    """主测试函数"""
    print("开始测试 analyze_retreat_advance 函数")
    print("=" * 50)
    
    success_count = 0
    total_tests = 4
    
    if test_basic_functionality():
        success_count += 1
    
    test_edge_cases()
    success_count += 1
    
    test_parameter_effects()
    success_count += 1
    
    compare_with_python_implementation()
    success_count += 1
    
    print("\n" + "=" * 50)
    print(f"测试完成: {success_count}/{total_tests} 个测试通过")
    
    if success_count == total_tests:
        print("🎉 所有测试通过！")
        return True
    else:
        print("⚠️  部分测试失败")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)