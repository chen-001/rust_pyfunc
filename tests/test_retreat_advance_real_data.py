#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用真实股票数据测试"以退为进"分析函数
基于design_whatever库读取L2级别的逐笔成交和盘口快照数据
"""

import numpy as np
import pandas as pd
import time
import sys
import os
from datetime import datetime

# 添加当前目录到路径，以便导入rust_pyfunc
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

try:
    from rust_pyfunc import analyze_retreat_advance, analyze_retreat_advance_v2
    print("✓ 成功导入 analyze_retreat_advance 和 analyze_retreat_advance_v2 函数")
except ImportError as e:
    print(f"✗ 导入失败: {e}")
    print("请先编译安装rust_pyfunc库")
    sys.exit(1)

try:
    import design_whatever as dw
    print("✓ 成功导入 design_whatever 库")
except ImportError as e:
    print(f"✗ 导入design_whatever失败: {e}")
    print("请先安装design_whatever库")
    sys.exit(1)

def load_real_stock_data(date=20220819, symbol='000001'):
    """
    加载真实股票数据
    
    参数:
    - date: 日期，格式为YYYYMMDD
    - symbol: 股票代码
    
    返回:
    - trade_data: 逐笔成交数据 
    - asks_data: 卖方盘口数据
    """
    print(f"正在加载 {symbol} 在 {date} 的数据...")
    
    try:
        # 读取逐笔成交数据（不包含撤单）
        print("  - 读取逐笔成交数据...")
        trade_data = dw.read_l2_trade_data(
            start_date=date, 
            symbols=[symbol], 
            with_retreat=0
        )
        
        if trade_data.empty:
            print(f"  ✗ 没有找到 {symbol} 在 {date} 的逐笔成交数据")
            return None, None
            
        print(f"  ✓ 成功读取逐笔成交数据，共 {len(trade_data)} 条记录")
        
        # 读取盘口快照数据（挂单价格-挂单量对格式）
        print("  - 读取盘口快照数据...")
        asks_data, bids_data = dw.read_l2_market_data_price_vol_pair(
            date=date, 
            symbols=[symbol]
        )
        
        if asks_data.empty:
            print(f"  ✗ 没有找到 {symbol} 在 {date} 的盘口快照数据")
            return None, None
            
        print(f"  ✓ 成功读取盘口快照数据，卖方数据 {len(asks_data)} 条记录")
        
        return trade_data, asks_data
        
    except Exception as e:
        print(f"  ✗ 数据加载失败: {e}")
        return None, None

def preprocess_trade_data(trade_data, use_nanoseconds=False):
    """
    预处理逐笔成交数据
    
    参数:
    - use_nanoseconds: 是否使用纳秒时间戳（默认False使用小时）
    
    返回时间、价格、成交量、买卖标志的numpy数组
    """
    print("正在预处理逐笔成交数据...")
    
    if use_nanoseconds:
        # 转换为纳秒时间戳
        trade_times = trade_data['exchtime'].astype('int64').values.astype(np.float64)
        print(f"    - 使用纳秒时间戳: {trade_times.min():.0f} - {trade_times.max():.0f}")
    else:
        # 转换时间为小时形式（9:30 = 9.5）
        trade_times = trade_data['exchtime'].dt.hour + trade_data['exchtime'].dt.minute / 60.0 + trade_data['exchtime'].dt.second / 3600.0
        print(f"    - 时间范围: {trade_times.min():.2f} - {trade_times.max():.2f}")
    
    # 提取价格和成交量
    trade_prices = trade_data['price'].values
    trade_volumes = trade_data['volume'].values
    
    # 处理买卖标志：flag=66为主动买入(+1)，flag=83为主动卖出(-1)
    trade_flags = np.where(trade_data['flag'] == 66, 1.0, 
                          np.where(trade_data['flag'] == 83, -1.0, 0.0))
    
    print(f"  ✓ 预处理完成：")
    print(f"    - 价格范围: {trade_prices.min():.3f} - {trade_prices.max():.3f}")
    print(f"    - 成交量范围: {trade_volumes.min():.0f} - {trade_volumes.max():.0f}")
    print(f"    - 主动买入: {(trade_flags > 0).sum()} 笔")
    print(f"    - 主动卖出: {(trade_flags < 0).sum()} 笔")
    
    return trade_times if use_nanoseconds else trade_times.values, trade_prices, trade_volumes, trade_flags

def preprocess_orderbook_data(asks_data, use_nanoseconds=False):
    """
    预处理盘口快照数据（卖方挂单）
    
    参数:
    - use_nanoseconds: 是否使用纳秒时间戳（默认False使用小时）
    
    返回时间、价格、挂单量的numpy数组
    """
    print("正在预处理盘口快照数据...")
    
    if use_nanoseconds:
        # 转换为纳秒时间戳
        orderbook_times = asks_data['exchtime'].astype('int64').values.astype(np.float64)
        print(f"    - 使用纳秒时间戳: {orderbook_times.min():.0f} - {orderbook_times.max():.0f}")
    else:
        # 转换时间为小时形式
        orderbook_times = asks_data['exchtime'].dt.hour + asks_data['exchtime'].dt.minute / 60.0 + asks_data['exchtime'].dt.second / 3600.0
        print(f"    - 时间范围: {orderbook_times.min():.2f} - {orderbook_times.max():.2f}")
    
    # 提取价格和挂单量
    orderbook_prices = asks_data['price'].values
    orderbook_volumes = asks_data['vol'].values
    
    print(f"  ✓ 预处理完成：")
    print(f"    - 价格范围: {orderbook_prices.min():.3f} - {orderbook_prices.max():.3f}")
    print(f"    - 挂单量范围: {orderbook_volumes.min():.0f} - {orderbook_volumes.max():.0f}")
    print(f"    - 平均挂单量: {orderbook_volumes.mean():.0f}")
    
    return orderbook_times if use_nanoseconds else orderbook_times.values, orderbook_prices, orderbook_volumes

def analyze_with_real_data(trade_times, trade_prices, trade_volumes, trade_flags,
                          orderbook_times, orderbook_prices, orderbook_volumes,
                          volume_percentile=99.0, time_window_minutes=1.0, use_nanoseconds=False):
    """
    使用真实数据分析"以退为进"现象
    """
    print(f"\n开始分析'以退为进'现象（异常挂单量阈值: {volume_percentile}%，时间窗口: {time_window_minutes}分钟）...")
    
    start_time = time.time()
    
    try:
        if use_nanoseconds:
            # 使用v2版本（支持纳秒和可调时间窗口）
            results = analyze_retreat_advance_v2(
                trade_times.astype(np.float64),
                trade_prices.astype(np.float64),
                trade_volumes.astype(np.float64),
                trade_flags.astype(np.float64),
                orderbook_times.astype(np.float64),
                orderbook_prices.astype(np.float64),
                orderbook_volumes.astype(np.float64),
                volume_percentile=volume_percentile,
                time_window_minutes=time_window_minutes
            )
        else:
            # 使用原版本（小时单位，固定1分钟窗口）
            results = analyze_retreat_advance(
                trade_times.astype(np.float64),
                trade_prices.astype(np.float64),
                trade_volumes.astype(np.float64),
                trade_flags.astype(np.float64),
                orderbook_times.astype(np.float64),
                orderbook_prices.astype(np.float64),
                orderbook_volumes.astype(np.float64),
                volume_percentile=volume_percentile
            )
        
        end_time = time.time()
        
        process_volumes, large_volumes, time_window_volumes, buy_ratios, price_counts, max_declines = results
        
        print(f"✓ Rust版本分析完成，耗时: {end_time - start_time:.4f}秒")
        print(f"✓ 共发现 {len(process_volumes)} 个以退为进过程")
        
        return results
        
    except Exception as e:
        print(f"✗ 分析过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return None

def display_analysis_results(results):
    """
    展示分析结果
    """
    if results is None:
        return
        
    process_volumes, large_volumes, time_window_volumes, buy_ratios, price_counts, max_declines = results
    
    if len(process_volumes) == 0:
        print("\n📊 分析结果：未发现符合条件的以退为进过程")
        return
    
    print(f"\n📊 分析结果详情：")
    print("=" * 60)
    
    # 创建结果DataFrame便于查看
    results_df = pd.DataFrame({
        '过程序号': range(1, len(process_volumes) + 1),
        '过程成交量': process_volumes,
        '异常挂单量': large_volumes,
        '时间窗口成交量': time_window_volumes,
        '买入占比': buy_ratios,
        '价格种类数': price_counts.astype(int),
        '最大下跌比例': max_declines
    })
    
    print("各过程详细数据：")
    print(results_df.to_string(index=False, float_format='%.3f'))
    
    print(f"\n📈 统计摘要：")
    print(f"  过程成交量    - 均值: {process_volumes.mean():.0f}, 中位数: {np.median(process_volumes):.0f}")
    print(f"  异常挂单量    - 均值: {large_volumes.mean():.0f}, 中位数: {np.median(large_volumes):.0f}")
    print(f"  时间窗口成交量 - 均值: {time_window_volumes.mean():.0f}, 中位数: {np.median(time_window_volumes):.0f}")
    print(f"  买入占比      - 均值: {buy_ratios.mean():.3f}, 中位数: {np.median(buy_ratios):.3f}")
    print(f"  最大下跌比例  - 均值: {max_declines.mean():.3f}, 最大: {max_declines.max():.3f}")

def compare_rust_python_algorithms(trade_times, trade_prices, trade_volumes, trade_flags,
                                  orderbook_times, orderbook_prices, orderbook_volumes,
                                  volume_percentile=95.0, time_window_minutes=1.0, use_nanoseconds=False):
    """
    对比Rust版本和Python版本算法的结果
    """
    print(f"\n🔍 对比Rust和Python算法（阈值: {volume_percentile}%, 时间窗口: {time_window_minutes}分钟）...")
    
    # Rust版本
    print("\n--- Rust版本 ---")
    start_time = time.time()
    rust_results = analyze_with_real_data(
        trade_times, trade_prices, trade_volumes, trade_flags,
        orderbook_times, orderbook_prices, orderbook_volumes,
        volume_percentile=volume_percentile,
        time_window_minutes=time_window_minutes,
        use_nanoseconds=use_nanoseconds
    )
    rust_time = time.time() - start_time
    
    # Python版本
    print("\n--- Python版本 ---")
    start_time = time.time()
    python_results = analyze_retreat_advance_python(
        trade_times, trade_prices, trade_volumes, trade_flags,
        orderbook_times, orderbook_prices, orderbook_volumes,
        volume_percentile=volume_percentile,
        time_window_minutes=time_window_minutes,
        use_nanoseconds=use_nanoseconds
    )
    python_time = time.time() - start_time
    print(f"✓ Python版本分析完成，耗时: {python_time:.4f}秒")
    print(f"✓ 共发现 {len(python_results[0])} 个以退为进过程")
    
    # 对比结果
    print(f"\n🔍 结果对比：")
    print("-" * 60)
    
    if rust_results is None or python_results is None:
        print("❌ 无法进行对比，某个算法执行失败")
        return
    
    rust_count = len(rust_results[0])
    python_count = len(python_results[0])
    
    print(f"过程数量    - Rust: {rust_count}, Python: {python_count}")
    print(f"执行时间    - Rust: {rust_time:.4f}s, Python: {python_time:.4f}s")
    print(f"性能提升    - {python_time/rust_time:.1f}x 加速")
    
    if rust_count > 0 and python_count > 0 and rust_count == python_count:
        # 详细对比各项指标
        print(f"\n📊 指标对比（前10个过程）：")
        comparison_df = pd.DataFrame({
            'Rust过程成交量': rust_results[0][:10],
            'Python过程成交量': python_results[0][:10],
            'Rust时间窗口量': rust_results[2][:10],
            'Python时间窗口量': python_results[2][:10],
            'Rust买入占比': rust_results[3][:10],
            'Python买入占比': python_results[3][:10],
        })
        print(comparison_df.to_string(index=False, float_format='%.3f'))
        
        # 计算差异
        max_diff_volume = np.max(np.abs(rust_results[0] - python_results[0]))
        max_diff_window = np.max(np.abs(rust_results[2] - python_results[2]))
        max_diff_ratio = np.max(np.abs(rust_results[3] - python_results[3]))
        
        print(f"\n📈 最大差异：")
        print(f"  过程成交量: {max_diff_volume:.3f}")
        print(f"  时间窗口成交量: {max_diff_window:.3f}")
        print(f"  买入占比: {max_diff_ratio:.6f}")
        
        tolerance = 1e-6
        if max_diff_volume < tolerance and max_diff_window < tolerance and max_diff_ratio < tolerance:
            print("\n✅ 算法一致性验证通过！Rust和Python版本结果完全一致")
        else:
            print("\n⚠️  算法存在细微差异，可能由于浮点精度或实现细节导致")
    else:
        print(f"\n⚠️  过程数量不一致，无法进行详细对比")
    
    return rust_results, python_results

def test_multiple_thresholds(trade_times, trade_prices, trade_volumes, trade_flags,
                           orderbook_times, orderbook_prices, orderbook_volumes,
                           use_nanoseconds=False):
    """
    测试不同的异常挂单量阈值对结果的影响
    """
    print(f"\n🔍 测试不同异常挂单量阈值的影响...")
    
    thresholds = [95.0, 97.0, 99.0, 99.5, 99.9]
    threshold_results = {}
    
    for threshold in thresholds:
        print(f"\n测试阈值: {threshold}%")
        results = analyze_with_real_data(
            trade_times, trade_prices, trade_volumes, trade_flags,
            orderbook_times, orderbook_prices, orderbook_volumes,
            volume_percentile=threshold,
            time_window_minutes=1.0,
            use_nanoseconds=use_nanoseconds
        )
        
        if results is not None:
            process_count = len(results[0])
            threshold_results[threshold] = process_count
            print(f"  发现 {process_count} 个过程")
        else:
            threshold_results[threshold] = 0
    
    print(f"\n📊 阈值敏感性分析：")
    print("阈值 (%) | 发现过程数")
    print("-" * 20)
    for threshold, count in threshold_results.items():
        print(f"  {threshold:5.1f}  |     {count:3d}")

def main():
    """
    主测试函数
    """
    print("=" * 80)
    print("使用真实股票数据测试'以退为进'分析函数（支持纳秒时间戳和算法对比）")
    print("=" * 80)
    
    # 测试参数
    test_date = 20220819  # 可以修改为其他日期
    test_symbol = '000001'  # 平安银行，可以修改为其他股票
    
    # 步骤1：加载真实数据
    trade_data, asks_data = load_real_stock_data(test_date, test_symbol)
    
    if trade_data is None or asks_data is None:
        print("❌ 数据加载失败，测试结束")
        return False
    
    # 步骤2：数据预处理（小时格式，用于原版本测试）
    print("\n=== 测试原版本（小时时间戳） ===")
    trade_times_hour, trade_prices, trade_volumes, trade_flags = preprocess_trade_data(trade_data, use_nanoseconds=False)
    orderbook_times_hour, orderbook_prices, orderbook_volumes = preprocess_orderbook_data(asks_data, use_nanoseconds=False)
    
    # 步骤3：基本分析（原版本）
    results_hour = analyze_with_real_data(
        trade_times_hour, trade_prices, trade_volumes, trade_flags,
        orderbook_times_hour, orderbook_prices, orderbook_volumes,
        volume_percentile=95.0, time_window_minutes=1.0, use_nanoseconds=False
    )
    
    # 步骤4：展示结果
    print("\n--- 原版本结果 ---")
    display_analysis_results(results_hour)
    
    # 步骤5：数据预处理（纳秒格式，用于v2版本测试）
    print("\n=== 测试v2版本（纳秒时间戳） ===")
    trade_times_ns, trade_prices, trade_volumes, trade_flags = preprocess_trade_data(trade_data, use_nanoseconds=True)
    orderbook_times_ns, orderbook_prices, orderbook_volumes = preprocess_orderbook_data(asks_data, use_nanoseconds=True)
    
    # 步骤6：算法对比测试
    print("\n--- Rust vs Python 算法对比 ---")
    rust_results, python_results = compare_rust_python_algorithms(
        trade_times_ns, trade_prices, trade_volumes, trade_flags,
        orderbook_times_ns, orderbook_prices, orderbook_volumes,
        volume_percentile=95.0, time_window_minutes=1.0, use_nanoseconds=True
    )
    
    # 步骤7：不同时间窗口测试
    print("\n--- 不同时间窗口测试 ---")
    time_windows = [0.5, 1.0, 2.0, 5.0]
    for window in time_windows:
        print(f"\n🔍 测试时间窗口: {window} 分钟")
        window_results = analyze_with_real_data(
            trade_times_ns, trade_prices, trade_volumes, trade_flags,
            orderbook_times_ns, orderbook_prices, orderbook_volumes,
            volume_percentile=95.0, time_window_minutes=window, use_nanoseconds=True
        )
        if window_results:
            count = len(window_results[0])
            avg_window_volume = window_results[2].mean() if count > 0 else 0
            print(f"  发现 {count} 个过程，平均时间窗口成交量: {avg_window_volume:.0f}")
    
    # 步骤8：敏感性分析
    test_multiple_thresholds(
        trade_times_ns, trade_prices, trade_volumes, trade_flags,
        orderbook_times_ns, orderbook_prices, orderbook_volumes,
        use_nanoseconds=True
    )
    
    print("\n" + "=" * 80)
    print("✅ 全部测试完成！")
    print("=" * 80)
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)