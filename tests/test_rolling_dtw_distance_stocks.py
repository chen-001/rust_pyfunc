#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import time
import rust_pyfunc as rp
import altair as alt
import design_whatever as dw

def python_rolling_dtw_distance(son, dragon, exchtime, minute_back):
    """
    使用Python实现滚动DTW距离计算
    
    参数说明：
    ----------
    son : array_like
        主要时间序列，将在此序列上滚动计算DTW距离
    dragon : array_like
        参考时间序列，用于计算DTW距离的模板
    exchtime : array_like
        时间戳数组，必须与son长度相同
    minute_back : float
        滚动窗口大小，以分钟为单位，表示每次计算使用的历史数据长度
        
    返回值：
    -------
    numpy.ndarray
        与son等长的数组，包含每个点的DTW距离
    """
    # 将son和dragon转换为numpy数组
    son_array = np.array(son)
    dragon_array = np.array(dragon)
    
    # 标准化dragon序列
    dragon_mean = np.mean(dragon_array)
    dragon_std = np.std(dragon_array)
    dragon_normalized = (dragon_array - dragon_mean) / dragon_std
    
    # 创建结果数组
    result = np.full_like(son_array, np.nan)
    
    # 定义一分钟的纳秒数
    one_minute_ns = 60.0 * 1_000_000_000.0
    
    # 计算滚动DTW距离
    for i in range(len(son_array)):
        current_time = exchtime[i]
        start_time = current_time - (minute_back * one_minute_ns)
        
        # 收集当前时间点前minute_back分钟内的数据
        segment_indices = []
        for j in range(i+1):
            if exchtime[j] > start_time and exchtime[j] <= current_time:
                segment_indices.append(j)
        
        segment = son_array[segment_indices]
        
        if len(segment) > 1 and len(dragon_normalized) > 1:
            # 标准化segment
            segment_mean = np.mean(segment)
            segment_std = np.std(segment)
            
            # 确保std不为零，避免除以零的错误
            if segment_std > 0:
                segment_normalized = (segment - segment_mean) / segment_std
                
                # 计算DTW距离
                try:
                    distance = rp.dtw_distance(segment_normalized, dragon_normalized)
                    result[i] = distance
                except Exception as e:
                    print(f"计算DTW距离时出错: {e}")
    
    return result

def test_with_real_stock_data():
    """测试Rust版本和Python版本在真实股票数据上的rolling_dtw_distance函数"""
    print("正在读取真实股票数据...")
    
    # 读取股票数据 - 逐笔成交数据
    try:
        # 读取某只股票的逐笔成交数据
        trade_data = dw.read_l2_trade_data(start_date=20220819, symbols=['000001'], with_retreat=0)
        
        # 确保数据按时间排序
        trade_data = trade_data.sort_values('exchtime')
        
        # 提取价格序列和时间戳
        prices = trade_data['price'].values
        times = trade_data['exchtime'].values.astype(np.float64)  # 将datetime64转为纳秒时间戳
        
        # 只使用前2000个数据点进行测试（以节省时间）
        n_samples = min(2000, len(prices))
        prices = prices[:n_samples]
        times = times[:n_samples]
        
        print(f"数据读取完成，使用前{n_samples}个数据点进行测试")
        
        # 创建一个模板波形（例如，波浪形状）
        dragon = np.sin(np.linspace(0, 2*np.pi, 40))
        
        # 设置滚动窗口大小（分钟）
        minute_back = 3.0
        
        # 测试Python版本性能
        print("正在测试Python版实现...")
        start_time = time.time()
        python_result = python_rolling_dtw_distance(prices, dragon, times, minute_back)
        python_time = time.time() - start_time
        print(f"Python版本耗时: {python_time:.4f}秒")
        
        # 测试Rust版本性能
        print("正在测试Rust版实现...")
        start_time = time.time()
        rust_result = rp.rolling_dtw_distance(prices, dragon, times, minute_back)
        rust_time = time.time() - start_time
        print(f"Rust版本耗时: {rust_time:.4f}秒")
        
        # 性能提升倍数
        speedup = python_time / rust_time
        print(f"Rust版本性能提升了 {speedup:.2f} 倍")
        
        # 验证结果是否一致
        rust_nans = np.isnan(rust_result)
        python_nans = np.isnan(python_result)
        
        print(f"Rust NaN数量: {np.sum(rust_nans)}")
        print(f"Python NaN数量: {np.sum(python_nans)}")
        
        # 计算非NaN结果的相关系数
        valid_mask = ~(np.isnan(rust_result) | np.isnan(python_result))
        if np.sum(valid_mask) > 1:
            corr = np.corrcoef(rust_result[valid_mask], python_result[valid_mask])[0, 1]
            print(f"Rust和Python结果的相关系数: {corr:.6f}")
            
            # 计算最大相对误差
            max_rel_diff = np.max(np.abs(rust_result[valid_mask] - python_result[valid_mask]) / 
                                 np.abs(python_result[valid_mask]))
            print(f"最大相对误差: {max_rel_diff:.2%}")
            
            # 创建DataFrame用于绘图
            df = pd.DataFrame({
                'Time': pd.to_datetime(times),
                'Price': prices,
                'Rust DTW': rust_result,
                'Python DTW': python_result
            })
            
            # 只保留有效的结果用于可视化
            df_valid = df.dropna(subset=['Rust DTW', 'Python DTW'])
            
            # 使用Altair绘制比较图
            price_chart = alt.Chart(df_valid).mark_line(color='black').encode(
                x='Time:T',
                y=alt.Y('Price:Q', title='价格'),
                tooltip=['Time:T', 'Price:Q']
            ).properties(
                width=800,
                height=200,
                title='股票价格'
            )
            
            dtw_comparison = alt.Chart(df_valid).transform_fold(
                ['Rust DTW', 'Python DTW'], 
                as_=['Method', 'DTW Distance']
            ).mark_line().encode(
                x='Time:T',
                y=alt.Y('DTW Distance:Q', title='DTW距离'),
                color='Method:N',
                tooltip=['Time:T', 'DTW Distance:Q', 'Method:N']
            ).properties(
                width=800,
                height=400,
                title='滚动DTW距离比较 - 真实股票数据'
            )
            
            # 组合图表
            combined_chart = alt.vconcat(
                price_chart, 
                dtw_comparison
            ).resolve_scale(
                x='shared'
            ).properties(
                title='股票数据滚动DTW距离分析'
            )
            
            # 保存为HTML文件
            combined_chart.save('stock_rolling_dtw_comparison.html')
            print("可视化结果已保存为 stock_rolling_dtw_comparison.html")
            
    except Exception as e:
        print(f"加载股票数据时出错: {e}")
        # 如果无法获取真实数据，使用模拟数据测试
        print("使用模拟数据进行测试...")
        
        # 创建模拟价格序列
        n = 2000
        times = np.arange(n) * 1_000_000_000  # 以秒为单位的时间戳
        prices = np.sin(np.linspace(0, 10, n)) + np.random.normal(0, 0.1, n)  # 添加一些噪声
        dragon = np.sin(np.linspace(0, 2*np.pi, 20))  # 一个正弦波作为模板
        
        # 设置滚动窗口大小（分钟）
        minute_back = 3.0
        
        # 测试Python版本性能
        start_time = time.time()
        python_result = python_rolling_dtw_distance(prices, dragon, times, minute_back)
        python_time = time.time() - start_time
        print(f"Python版本耗时: {python_time:.4f}秒")
        
        # 测试Rust版本性能
        start_time = time.time()
        rust_result = rp.rolling_dtw_distance(prices, dragon, times, minute_back)
        rust_time = time.time() - start_time
        print(f"Rust版本耗时: {rust_time:.4f}秒")
        
        # 性能提升倍数
        speedup = python_time / rust_time
        print(f"Rust版本性能提升了 {speedup:.2f} 倍")

if __name__ == "__main__":
    test_with_real_stock_data()
