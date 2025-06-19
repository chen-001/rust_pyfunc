#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
性能对比测试：Rust vs Python版本
使用真实数据的子集进行测试以避免超时
"""

import numpy as np
import pandas as pd
import time
import sys
import os

# 添加路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

try:
    from rust_pyfunc import analyze_retreat_advance
    import design_whatever as dw
    print("✓ 成功导入所需库")
except ImportError as e:
    print(f"✗ 导入失败: {e}")
    sys.exit(1)

# 导入Python版本的函数
from python_retreat_advance import analyze_retreat_advance_python

def load_sample_data(sample_ratio=0.1):
    """
    加载并采样真实数据
    """
    print(f"加载数据样本 (采样比例: {sample_ratio:.1%})...")
    
    date = 20220819
    symbol = '000001'
    
    # 读取完整数据
    trade_data = dw.read_l2_trade_data(start_date=date, symbols=[symbol], with_retreat=0)
    asks_data, _ = dw.read_l2_market_data_price_vol_pair(date=date, symbols=[symbol])
    
    # 随机采样
    np.random.seed(42)  # 确保可重复性
    
    trade_sample_size = int(len(trade_data) * sample_ratio)
    orderbook_sample_size = int(len(asks_data) * sample_ratio)
    
    trade_indices = np.sort(np.random.choice(len(trade_data), trade_sample_size, replace=False))
    orderbook_indices = np.sort(np.random.choice(len(asks_data), orderbook_sample_size, replace=False))
    
    trade_sample = trade_data.iloc[trade_indices].copy()
    orderbook_sample = asks_data.iloc[orderbook_indices].copy()
    
    # 预处理
    trade_times = (trade_sample['exchtime'].dt.hour + 
                  trade_sample['exchtime'].dt.minute / 60.0 + 
                  trade_sample['exchtime'].dt.second / 3600.0).values.astype(np.float64)
    trade_prices = trade_sample['price'].values.astype(np.float64)
    trade_volumes = trade_sample['volume'].values.astype(np.float64)
    trade_flags = np.where(trade_sample['flag'] == 66, 1.0, 
                          np.where(trade_sample['flag'] == 83, -1.0, 0.0)).astype(np.float64)
    
    orderbook_times = (orderbook_sample['exchtime'].dt.hour + 
                      orderbook_sample['exchtime'].dt.minute / 60.0 + 
                      orderbook_sample['exchtime'].dt.second / 3600.0).values.astype(np.float64)
    orderbook_prices = orderbook_sample['price'].values.astype(np.float64)
    orderbook_volumes = orderbook_sample['vol'].values.astype(np.float64)
    
    print(f"  采样后数据规模: 成交{len(trade_times):,}条, 盘口{len(orderbook_times):,}条")
    
    return (trade_times, trade_prices, trade_volumes, trade_flags,
            orderbook_times, orderbook_prices, orderbook_volumes)

def compare_results(rust_results, python_results, tolerance=1e-6):
    """比较结果一致性"""
    print("\n🔍 结果一致性检查:")
    
    result_names = [
        "过程成交量", "异常挂单量", "1分钟成交量", 
        "买入占比", "价格种类数", "最大下跌比例"
    ]
    
    if len(rust_results[0]) != len(python_results[0]):
        print(f"✗ 发现过程数不一致: Rust={len(rust_results[0])}, Python={len(python_results[0])}")
        return False
    
    if len(rust_results[0]) == 0:
        print("✓ 两个版本都未发现过程，结果一致")
        return True
    
    all_consistent = True
    
    for i, (rust_arr, python_arr, name) in enumerate(zip(rust_results, python_results, result_names)):
        diff = np.abs(rust_arr - python_arr)
        max_diff = np.max(diff)
        
        if max_diff < tolerance:
            print(f"✓ {name}: 完全一致 (最大差异: {max_diff:.2e})")
        else:
            print(f"✗ {name}: 存在差异 (最大差异: {max_diff:.2e})")
            all_consistent = False
    
    return all_consistent

def run_performance_test():
    """运行性能测试"""
    print("=" * 80)
    print("Rust vs Python 性能对比测试")
    print("=" * 80)
    
    # 测试不同的数据规模
    sample_ratios = [0.05, 0.1, 0.2]  # 5%, 10%, 20%
    
    results_summary = []
    
    for ratio in sample_ratios:
        print(f"\n{'='*60}")
        print(f"测试数据规模: {ratio:.1%}")
        print(f"{'='*60}")
        
        # 加载数据
        data = load_sample_data(ratio)
        trade_times, trade_prices, trade_volumes, trade_flags, orderbook_times, orderbook_prices, orderbook_volumes = data
        
        # 测试95%阈值
        threshold = 95.0
        print(f"\n使用阈值: {threshold}%")
        
        # Rust版本测试
        print("\n🦀 Rust版本测试:")
        rust_times = []
        rust_results = None
        
        for run in range(3):
            start_time = time.time()
            rust_results = analyze_retreat_advance(
                trade_times, trade_prices, trade_volumes, trade_flags,
                orderbook_times, orderbook_prices, orderbook_volumes,
                volume_percentile=threshold
            )
            rust_time = time.time() - start_time
            rust_times.append(rust_time)
            print(f"  运行 {run+1}: {rust_time:.4f}秒")
        
        avg_rust_time = np.mean(rust_times)
        rust_processes = len(rust_results[0])
        print(f"  平均耗时: {avg_rust_time:.4f}秒, 发现过程: {rust_processes}个")
        
        # Python版本测试
        print("\n🐍 Python版本测试:")
        python_times = []
        python_results = None
        
        for run in range(3):
            start_time = time.time()
            python_results = analyze_retreat_advance_python(
                trade_times, trade_prices, trade_volumes, trade_flags,
                orderbook_times, orderbook_prices, orderbook_volumes,
                volume_percentile=threshold
            )
            python_time = time.time() - start_time
            python_times.append(python_time)
            print(f"  运行 {run+1}: {python_time:.4f}秒")
        
        avg_python_time = np.mean(python_times)
        python_processes = len(python_results[0])
        print(f"  平均耗时: {avg_python_time:.4f}秒, 发现过程: {python_processes}个")
        
        # 性能对比
        if avg_rust_time > 0:
            speedup = avg_python_time / avg_rust_time
            print(f"\n📊 性能对比:")
            print(f"  Rust:   {avg_rust_time:.4f}秒")
            print(f"  Python: {avg_python_time:.4f}秒")
            print(f"  加速比: {speedup:.2f}x")
            
            # 结果一致性检查
            consistent = compare_results(rust_results, python_results)
            
            # 保存结果
            results_summary.append({
                'sample_ratio': ratio,
                'trade_records': len(trade_times),
                'orderbook_records': len(orderbook_times),
                'rust_time': avg_rust_time,
                'python_time': avg_python_time,
                'speedup': speedup,
                'rust_processes': rust_processes,
                'python_processes': python_processes,
                'consistent': consistent
            })
    
    # 显示总结
    print(f"\n{'='*80}")
    print("📈 性能测试总结")
    print(f"{'='*80}")
    
    df = pd.DataFrame(results_summary)
    print("\n详细结果:")
    print(df.to_string(index=False, float_format='%.4f'))
    
    print(f"\n🏆 关键发现:")
    print(f"  平均加速比: {df['speedup'].mean():.2f}x")
    print(f"  最大加速比: {df['speedup'].max():.2f}x")
    print(f"  结果一致性: {df['consistent'].all()}")
    
    return df

if __name__ == "__main__":
    results = run_performance_test()