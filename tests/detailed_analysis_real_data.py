#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
详细分析真实数据中发现的"以退为进"过程
"""

import numpy as np
import pandas as pd
import time
import sys
import os

# 添加当前目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

try:
    from rust_pyfunc import analyze_retreat_advance
    import design_whatever as dw
    print("✓ 成功导入所需库")
except ImportError as e:
    print(f"✗ 导入失败: {e}")
    sys.exit(1)

def load_and_preprocess_data(date=20220819, symbol='000001'):
    """加载并预处理数据"""
    print(f"加载 {symbol} 在 {date} 的数据...")
    
    # 读取数据
    trade_data = dw.read_l2_trade_data(start_date=date, symbols=[symbol], with_retreat=0)
    asks_data, _ = dw.read_l2_market_data_price_vol_pair(date=date, symbols=[symbol])
    
    # 预处理逐笔成交数据
    trade_times = (trade_data['exchtime'].dt.hour + 
                  trade_data['exchtime'].dt.minute / 60.0 + 
                  trade_data['exchtime'].dt.second / 3600.0).values
    trade_prices = trade_data['price'].values
    trade_volumes = trade_data['volume'].values
    trade_flags = np.where(trade_data['flag'] == 66, 1.0, 
                          np.where(trade_data['flag'] == 83, -1.0, 0.0))
    
    # 预处理盘口数据
    orderbook_times = (asks_data['exchtime'].dt.hour + 
                      asks_data['exchtime'].dt.minute / 60.0 + 
                      asks_data['exchtime'].dt.second / 3600.0).values
    orderbook_prices = asks_data['price'].values
    orderbook_volumes = asks_data['vol'].values
    
    return (trade_times, trade_prices, trade_volumes, trade_flags,
            orderbook_times, orderbook_prices, orderbook_volumes, trade_data)

def detailed_analysis():
    """详细分析"""
    print("=" * 80)
    print("详细分析'以退为进'现象")
    print("=" * 80)
    
    # 加载数据
    (trade_times, trade_prices, trade_volumes, trade_flags,
     orderbook_times, orderbook_prices, orderbook_volumes, trade_data) = load_and_preprocess_data()
    
    print(f"数据概况：")
    print(f"  - 逐笔成交记录: {len(trade_times):,} 条")
    print(f"  - 盘口快照记录: {len(orderbook_times):,} 条")
    print(f"  - 时间范围: {trade_times.min():.2f} - {trade_times.max():.2f}")
    print(f"  - 价格范围: {trade_prices.min():.3f} - {trade_prices.max():.3f}")
    
    # 分析不同阈值下的结果
    thresholds = [90.0, 95.0, 97.0, 98.0, 99.0]
    
    for threshold in thresholds:
        print(f"\n{'='*60}")
        print(f"分析阈值: {threshold}%")
        print(f"{'='*60}")
        
        start_time = time.time()
        results = analyze_retreat_advance(
            trade_times.astype(np.float64),
            trade_prices.astype(np.float64),
            trade_volumes.astype(np.float64),
            trade_flags.astype(np.float64),
            orderbook_times.astype(np.float64),
            orderbook_prices.astype(np.float64),
            orderbook_volumes.astype(np.float64),
            volume_percentile=threshold
        )
        analysis_time = time.time() - start_time
        
        process_volumes, large_volumes, one_min_volumes, buy_ratios, price_counts, max_declines = results
        num_processes = len(process_volumes)
        
        print(f"分析耗时: {analysis_time:.3f}秒")
        print(f"发现过程数: {num_processes}")
        
        if num_processes > 0:
            # 计算挂单量阈值
            volume_threshold = np.percentile(orderbook_volumes, threshold)
            print(f"挂单量阈值 ({threshold}%): {volume_threshold:,.0f}")
            
            # 详细统计
            print(f"\n📊 过程统计:")
            print(f"  过程成交量     - 范围: {process_volumes.min():,.0f} - {process_volumes.max():,.0f}")
            print(f"                - 平均: {process_volumes.mean():,.0f}, 中位数: {np.median(process_volumes):,.0f}")
            
            print(f"  异常挂单量     - 范围: {large_volumes.min():,.0f} - {large_volumes.max():,.0f}")
            print(f"                - 平均: {large_volumes.mean():,.0f}, 中位数: {np.median(large_volumes):,.0f}")
            
            print(f"  1分钟成交量    - 范围: {one_min_volumes.min():,.0f} - {one_min_volumes.max():,.0f}")
            print(f"                - 平均: {one_min_volumes.mean():,.0f}, 中位数: {np.median(one_min_volumes):,.0f}")
            
            print(f"  买入占比       - 范围: {buy_ratios.min():.3f} - {buy_ratios.max():.3f}")
            print(f"                - 平均: {buy_ratios.mean():.3f}, 中位数: {np.median(buy_ratios):.3f}")
            
            print(f"  价格种类数     - 范围: {price_counts.min():.0f} - {price_counts.max():.0f}")
            print(f"                - 平均: {price_counts.mean():.1f}, 中位数: {np.median(price_counts):.0f}")
            
            print(f"  最大下跌比例   - 范围: {max_declines.min():.4f} - {max_declines.max():.4f}")
            print(f"                - 平均: {max_declines.mean():.4f}, 中位数: {np.median(max_declines):.4f}")
            
            # 显示前10个过程的详细信息
            if num_processes <= 20:
                show_count = num_processes
            else:
                show_count = 10
                
            print(f"\n📋 前{show_count}个过程详情:")
            results_df = pd.DataFrame({
                '序号': range(1, show_count + 1),
                '成交量': process_volumes[:show_count],
                '异常挂单量': large_volumes[:show_count],
                '1分钟量': one_min_volumes[:show_count],
                '买入占比': buy_ratios[:show_count],
                '价格数': price_counts[:show_count].astype(int),
                '最大跌幅': max_declines[:show_count]
            })
            print(results_df.to_string(index=False, float_format='%.3f'))
            
            # 分析特征分布
            print(f"\n📈 特征分布分析:")
            
            # 买入占比分布
            high_buy_ratio = (buy_ratios > 0.6).sum()
            low_buy_ratio = (buy_ratios < 0.4).sum()
            print(f"  买入占比 > 60%: {high_buy_ratio} 个过程 ({high_buy_ratio/num_processes*100:.1f}%)")
            print(f"  买入占比 < 40%: {low_buy_ratio} 个过程 ({low_buy_ratio/num_processes*100:.1f}%)")
            
            # 下跌幅度分布
            small_decline = (max_declines < 0.01).sum()  # 小于1%
            medium_decline = ((max_declines >= 0.01) & (max_declines < 0.03)).sum()  # 1-3%
            large_decline = (max_declines >= 0.03).sum()  # 大于3%
            print(f"  最大下跌 < 1%:  {small_decline} 个过程 ({small_decline/num_processes*100:.1f}%)")
            print(f"  最大下跌 1-3%:  {medium_decline} 个过程 ({medium_decline/num_processes*100:.1f}%)")
            print(f"  最大下跌 > 3%:  {large_decline} 个过程 ({large_decline/num_processes*100:.1f}%)")
            
        else:
            print("未发现符合条件的过程")
    
    print(f"\n{'='*80}")
    print("✅ 详细分析完成")
    print(f"{'='*80}")

if __name__ == "__main__":
    detailed_analysis()