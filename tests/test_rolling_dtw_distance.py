#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import time
import rust_pyfunc as rp
import matplotlib.pyplot as plt
import altair as alt

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
    minute_back : int
        滚动窗口大小，以分钟为单位，表示每次计算使用的历史数据长度
        
    返回值：
    -------
    pandas.Series
        与son等长的Series，包含每个点的DTW距离
    """
    # 将exchtime转换为pandas的datetime格式
    times = pd.to_datetime(exchtime, unit='ns')
    
    # 将son和dragon转换为pandas Series
    son_series = pd.Series(son, index=times)
    dragon_series = pd.Series(dragon)
    
    # 标准化dragon序列
    dragon_series = (dragon_series - dragon_series.mean()) / dragon_series.std()
    
    # 创建结果Series
    result = pd.Series(index=times, dtype=float)
    
    # 计算滚动DTW距离
    for i in times:
        # 获取当前时间点前minute_back分钟的数据
        start_time = i - pd.Timedelta(minutes=minute_back)
        segment = son_series[(son_series.index > start_time) & (son_series.index <= i)]
        
        if len(segment) > 1 and len(dragon_series) > 1:
            # 标准化segment
            segment = (segment - segment.mean()) / segment.std()
            
            # 计算DTW距离
            try:
                distance = rp.dtw_distance(segment.to_numpy(dtype=float), dragon_series.to_numpy(dtype=float))
                result.loc[i] = distance
            except:
                result.loc[i] = np.nan
        else:
            result.loc[i] = np.nan
    
    return result

def test_rolling_dtw_distance():
    """测试Rust版本和Python版本的rolling_dtw_distance函数"""
    print("开始测试rolling_dtw_distance函数...")
    
    # 生成测试数据
    n = 200
    times = pd.date_range('2023-01-01', periods=n, freq='T')
    son = np.sin(np.linspace(0, 10, n)) + np.random.normal(0, 0.1, n)  # 添加一些噪声
    dragon = np.sin(np.linspace(0, 2*np.pi, 20))  # 一个正弦波作为模板
    
    # 将时间戳转换为纳秒格式
    exchtime = times.astype(np.int64).values
    
    # 设置滚动窗口大小（分钟）
    minute_back = 10
    
    # 测试Python版本性能
    start_time = time.time()
    python_result = python_rolling_dtw_distance(son, dragon, exchtime, minute_back)
    python_time = time.time() - start_time
    print(f"Python版本耗时: {python_time:.4f}秒")
    
    # 测试Rust版本性能
    start_time = time.time()
    rust_result = rp.rolling_dtw_distance(son, dragon, exchtime, minute_back)
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
    
    # 创建一个差异分析数据框
    diff_df = pd.DataFrame({
        'Time': times,
        'Son': son,
        'Rust': rust_result,
        'Python': python_result,
        'Diff': np.abs(np.array(rust_result) - np.array(python_result))
    })
    
    # 打印差异最大的五个结果
    print("\n差异最大的五个结果:")
    valid_indices = ~(np.isnan(rust_result) | np.isnan(python_result))
    if np.any(valid_indices):
        max_diff_indices = np.argsort(-diff_df['Diff'].values)[0:5]
        print(diff_df.iloc[max_diff_indices][["Time", "Rust", "Python", "Diff"]])
    
    if np.array_equal(rust_nans, python_nans):
        valid_indices = ~rust_nans
        if np.any(valid_indices):
            max_diff = np.max(np.abs(np.array(rust_result)[valid_indices] - 
                                     np.array(python_result)[valid_indices]))
            print(f"\n最大误差: {max_diff:.6f}")
            
            # 使用相对误差容差进行比较
            relative_tolerance = 0.05  # 5%的相对误差
            max_rel_diff = np.max(np.abs(np.array(rust_result)[valid_indices] - 
                                      np.array(python_result)[valid_indices]) / 
                                 np.abs(np.array(python_result)[valid_indices]))
            print(f"最大相对误差: {max_rel_diff:.2%}")
            
            if max_rel_diff < relative_tolerance:
                print("✅ 测试通过: Rust版本和Python版本的结果在可接受的误差范围内")
            else:
                print("❌ 测试失败: Rust版本和Python版本的结果差异超出可接受范围")
        else:
            print("❌ 测试失败: 所有值都是NaN")
    else:
        print("❌ 测试失败: NaN位置不一致")
        # 打印NaN位置不一致的详情
        nan_diff = np.where(rust_nans != python_nans)[0]
        print(f"NaN位置不一致的索引: {nan_diff[:10]}... 共{len(nan_diff)}个")
        for idx in nan_diff[:5]:
            print(f"索引{idx}: Rust值={rust_result[idx]}, Python值={python_result[idx]}")
    
    # 计算相关系数
    valid_mask = ~(np.isnan(rust_result) | np.isnan(python_result))
    if np.sum(valid_mask) > 1:
        corr = np.corrcoef(np.array(rust_result)[valid_mask], np.array(python_result)[valid_mask])[0, 1]
        print(f"\nRust和Python结果的相关系数: {corr:.6f}")
        if corr > 0.99:
            print("两者结果高度相关，可能只是尺度不同")
            
            # 计算比例因子
            ratio = np.mean(np.array(rust_result)[valid_mask] / np.array(python_result)[valid_mask])
            print(f"Rust/Python平均比例: {ratio:.4f}")
    
    # 可视化比较
    # 创建DataFrame用于绘图
    df = pd.DataFrame({
        'Time': times,
        'Rust': rust_result,
        'Python': python_result
    })
    
    # 使用Altair绘制比较图
    chart = alt.Chart(df).transform_fold(
        ['Rust', 'Python'], 
        as_=['Method', 'DTW Distance']
    ).mark_line().encode(
        x='Time:T',
        y='DTW Distance:Q',
        color='Method:N',
        tooltip=['Time:T', 'DTW Distance:Q', 'Method:N']
    ).properties(
        title='Rolling DTW Distance Comparison',
        width=800,
        height=400
    ).interactive()
    
    # 保存为HTML文件
    chart.save('rolling_dtw_distance_comparison.html')
    print("结果比较图已保存为 rolling_dtw_distance_comparison.html")

if __name__ == "__main__":
    test_rolling_dtw_distance()
