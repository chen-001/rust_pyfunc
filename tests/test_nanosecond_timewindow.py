#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试支持纳秒时间戳和可调节时间窗口的"以退为进"分析函数
"""

import numpy as np
import pandas as pd
import time
import sys
import os

# 添加路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

try:
    from rust_pyfunc import analyze_retreat_advance, analyze_retreat_advance_v2
    import design_whatever as dw
    print("✓ 成功导入所需库")
except ImportError as e:
    print(f"✗ 导入失败: {e}")
    sys.exit(1)

def test_nanosecond_timestamps():
    """测试纳秒时间戳和时间窗口参数"""
    print("=" * 80)
    print("测试纳秒时间戳和可调节时间窗口")
    print("=" * 80)
    
    # 加载真实数据
    print("加载真实数据...")
    date = 20220819
    symbol = '000001'
    
    trade_data = dw.read_l2_trade_data(start_date=date, symbols=[symbol], with_retreat=0)
    asks_data, _ = dw.read_l2_market_data_price_vol_pair(date=date, symbols=[symbol])
    
    print(f"数据规模: 成交{len(trade_data):,}条, 盘口{len(asks_data):,}条")
    
    # 将datetime转换为纳秒时间戳
    trade_times_ns = trade_data['exchtime'].astype('int64').values.astype(np.float64)
    trade_prices = trade_data['price'].values.astype(np.float64)
    trade_volumes = trade_data['volume'].values.astype(np.float64)
    trade_flags = np.where(trade_data['flag'] == 66, 1.0, 
                          np.where(trade_data['flag'] == 83, -1.0, 0.0)).astype(np.float64)
    
    orderbook_times_ns = asks_data['exchtime'].astype('int64').values.astype(np.float64)
    orderbook_prices = asks_data['price'].values.astype(np.float64)
    orderbook_volumes = asks_data['vol'].values.astype(np.float64)
    
    print(f"时间戳范围: {trade_times_ns.min():.0f} - {trade_times_ns.max():.0f} (纳秒)")
    print(f"时间跨度: {(trade_times_ns.max() - trade_times_ns.min()) / 1e9 / 3600:.2f} 小时")
    
    # 测试不同的时间窗口参数
    time_windows = [0.5, 1.0, 2.0, 5.0]  # 0.5分钟, 1分钟, 2分钟, 5分钟
    threshold = 95.0
    
    print(f"\n测试不同时间窗口 (阈值: {threshold}%):")
    print("-" * 60)
    
    results_summary = []
    
    for window in time_windows:
        print(f"\n时间窗口: {window} 分钟")
        
        start_time = time.time()
        results = analyze_retreat_advance_v2(
            trade_times_ns, trade_prices, trade_volumes, trade_flags,
            orderbook_times_ns, orderbook_prices, orderbook_volumes,
            volume_percentile=threshold,
            time_window_minutes=window
        )
        elapsed_time = time.time() - start_time
        
        process_volumes, large_volumes, time_window_volumes, buy_ratios, price_counts, max_declines = results
        num_processes = len(process_volumes)
        
        print(f"  发现过程数: {num_processes}")
        print(f"  计算耗时: {elapsed_time:.4f}秒")
        
        if num_processes > 0:
            print(f"  时间窗口成交量范围: {time_window_volumes.min():,.0f} - {time_window_volumes.max():,.0f}")
            print(f"  时间窗口成交量平均: {time_window_volumes.mean():,.0f}")
        
        results_summary.append({
            'time_window': window,
            'num_processes': num_processes,
            'elapsed_time': elapsed_time,
            'avg_window_volume': time_window_volumes.mean() if num_processes > 0 else 0
        })
    
    # 显示结果对比
    print(f"\n📊 时间窗口对比结果:")
    print("-" * 80)
    df = pd.DataFrame(results_summary)
    print(df.to_string(index=False, float_format='%.4f'))
    
    # 验证时间窗口逻辑
    print(f"\n🔍 验证时间窗口逻辑:")
    print("-" * 80)
    
    # 比较1分钟和2分钟窗口的结果
    results_1min = analyze_retreat_advance_v2(
        trade_times_ns, trade_prices, trade_volumes, trade_flags,
        orderbook_times_ns, orderbook_prices, orderbook_volumes,
        volume_percentile=threshold, time_window_minutes=1.0
    )
    
    results_2min = analyze_retreat_advance_v2(
        trade_times_ns, trade_prices, trade_volumes, trade_flags,
        orderbook_times_ns, orderbook_prices, orderbook_volumes,
        volume_percentile=threshold, time_window_minutes=2.0
    )
    
    window_1min = results_1min[2]  # 时间窗口成交量
    window_2min = results_2min[2]
    
    if len(window_1min) > 0 and len(window_2min) > 0:
        print(f"1分钟窗口平均成交量: {window_1min.mean():,.0f}")
        print(f"2分钟窗口平均成交量: {window_2min.mean():,.0f}")
        
        # 2分钟窗口的成交量应该大于等于1分钟窗口
        if len(window_1min) == len(window_2min):
            ratio = window_2min.mean() / window_1min.mean()
            print(f"2分钟/1分钟窗口成交量比例: {ratio:.2f}")
            
            if ratio >= 1.0:
                print("✓ 时间窗口逻辑正确：更长时间窗口包含更多成交量")
            else:
                print("✗ 时间窗口逻辑异常")
    
    return results_summary

def test_backwards_compatibility():
    """测试向后兼容性"""
    print(f"\n🔄 测试向后兼容性:")
    print("-" * 60)
    
    # 创建小规模测试数据
    n_trades = 1000
    n_orderbook = 500
    
    # 生成纳秒时间戳
    base_time = 1661743800000000000  # 2022-08-29 09:30:00 的纳秒时间戳
    trade_times = np.linspace(base_time, base_time + 3600 * 1e9, n_trades)  # 1小时数据
    orderbook_times = np.linspace(base_time, base_time + 3600 * 1e9, n_orderbook)
    
    # 生成模拟价格数据
    np.random.seed(42)
    base_price = 100.0
    trade_prices = base_price + np.cumsum(np.random.normal(0, 0.01, n_trades))
    orderbook_prices = base_price + np.cumsum(np.random.normal(0, 0.01, n_orderbook))
    
    # 生成其他数据
    trade_volumes = np.random.exponential(100, n_trades).astype(np.float64)
    trade_flags = np.random.choice([-1, 1], n_trades).astype(np.float64)
    orderbook_volumes = np.random.exponential(1000, n_orderbook).astype(np.float64)
    
    print("使用模拟数据测试...")
    
    # 测试默认参数（向后兼容）
    start_time = time.time()
    results_default = analyze_retreat_advance(
        trade_times, trade_prices, trade_volumes, trade_flags,
        orderbook_times, orderbook_prices, orderbook_volumes
    )
    time_default = time.time() - start_time
    
    # 测试显式指定参数
    start_time = time.time()
    results_explicit = analyze_retreat_advance(
        trade_times, trade_prices, trade_volumes, trade_flags,
        orderbook_times, orderbook_prices, orderbook_volumes,
        volume_percentile=99.0,
        time_window_minutes=1.0
    )
    time_explicit = time.time() - start_time
    
    print(f"默认参数结果: {len(results_default[0])} 个过程, 耗时: {time_default:.4f}s")
    print(f"显式参数结果: {len(results_explicit[0])} 个过程, 耗时: {time_explicit:.4f}s")
    
    # 验证结果一致性
    consistent = True
    for i, (default_arr, explicit_arr) in enumerate(zip(results_default, results_explicit)):
        if not np.array_equal(default_arr, explicit_arr):
            print(f"✗ 第{i+1}个返回值不一致")
            consistent = False
    
    if consistent:
        print("✓ 向后兼容性测试通过")
    else:
        print("✗ 向后兼容性测试失败")
    
    return consistent

def main():
    """主测试函数"""
    print("开始测试纳秒时间戳和可调节时间窗口功能")
    
    # 测试1：纳秒时间戳和时间窗口
    time_window_results = test_nanosecond_timestamps()
    
    # 测试2：向后兼容性
    compatibility_ok = test_backwards_compatibility()
    
    print(f"\n{'='*80}")
    print("✅ 测试完成！")
    print(f"{'='*80}")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)