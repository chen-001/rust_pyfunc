import numpy as np
import pandas as pd
import time
from datetime import datetime
import matplotlib.pyplot as plt
import altair as alt
import json

# 确保使用正确的Python路径
import sys
sys.path.append('/home/chenzongwei/rustcode/rust_pyfunc')
import rust_pyfunc
import design_whatever as dw

# 性能测试与比较函数
def compare_performance():
    print("比较find_half_extreme_time与fast_find_half_extreme_time的性能差异")
    print("="*80)
    
    # 读取股票的逐笔成交数据
    print("正在读取股票数据...")
    trade_data = dw.read_l2_trade_data(start_date=20220819, symbols=['000001'], with_retreat=0)
    print(f"读取完成，共{len(trade_data)}条记录")
    
    # 数据预处理
    print("正在预处理数据...")
    # 将时间转换为浮点数（纳秒）
    trade_data['time'] = pd.Series(trade_data['exchtime'].view(np.int64)).astype(np.float64)
    
    # 为了展示超时功能，取不同大小的数据样本
    sample_sizes = [1000, 5000, 10000, 50000]
    
    results = []
    
    # 在不同数据量下进行测试
    for size in sample_sizes:
        print(f"\n测试数据量: {size}")
        df = trade_data.head(size)
        
        # 使用原始函数计算
        print(f"原始函数 - 开始时间: {datetime.now().strftime('%H:%M:%S')}")
        start_time = time.time()
        result_original = rust_pyfunc.find_half_extreme_time(
            df['time'].values,
            df['price'].values,
            time_window=5.0
        )
        end_time = time.time()
        original_time = end_time - start_time
        print(f"  结束时间: {datetime.now().strftime('%H:%M:%S')}, 耗时: {original_time:.4f}秒")
        
        # 使用优化版函数计算
        print(f"优化版函数 - 开始时间: {datetime.now().strftime('%H:%M:%S')}")
        start_time = time.time()
        result_fast = rust_pyfunc.fast_find_half_extreme_time(
            df['time'].values,
            df['price'].values,
            time_window=5.0
        )
        end_time = time.time()
        fast_time = end_time - start_time
        print(f"  结束时间: {datetime.now().strftime('%H:%M:%S')}, 耗时: {fast_time:.4f}秒")
        
        # 验证结果正确性
        nan_count_original = np.sum(np.isnan(result_original))
        nan_count_fast = np.sum(np.isnan(result_fast))
        
        # 简化导致比较逻辑
        # 只计算非NaN值的差异率
        result_original_clean = result_original[~np.isnan(result_original)]
        result_fast_clean = result_fast[~np.isnan(result_fast)]
        
        # 取长度相同的一部分进行比较
        min_len = min(len(result_original_clean), len(result_fast_clean))
        if min_len > 0:
            # 取前 min_len 个元素进行比较
            original_sample = result_original_clean[:min_len]
            fast_sample = result_fast_clean[:min_len]
            
            # 计算相对误差
            with np.errstate(divide='ignore', invalid='ignore'):
                rel_errors = np.abs((original_sample - fast_sample) / np.maximum(original_sample, 1e-10))
            
            # 删除无效值
            rel_errors = rel_errors[np.isfinite(rel_errors)]
            
            mean_rel_error = np.mean(rel_errors) if len(rel_errors) > 0 else 0
            max_rel_error = np.max(rel_errors) if len(rel_errors) > 0 else 0
        else:
            mean_rel_error = 0
            max_rel_error = 0
        
        # 计算加速比
        speedup = original_time / fast_time if fast_time > 0 else float('inf')
        
        # 记录结果
        results.append({
            "数据量": size,
            "原始函数时间(秒)": original_time,
            "优化函数时间(秒)": fast_time,
            "加速比": speedup,
            "原始函数NaN数量": int(nan_count_original),
            "优化函数NaN数量": int(nan_count_fast), 
            "平均相对误差": float(mean_rel_error),
            "最大相对误差": float(max_rel_error)
        })
        
        print(f"  加速比: {speedup:.2f}倍")
        print(f"  原始函数NaN数量: {nan_count_original}/{size}")
        print(f"  优化函数NaN数量: {nan_count_fast}/{size}")
        print(f"  平均相对误差: {mean_rel_error:.6f}")
        print(f"  最大相对误差: {max_rel_error:.6f}")
    
    # 将结果转换为DataFrame以便于展示
    results_df = pd.DataFrame(results)
    print("\n性能比较结果:")
    print(results_df.to_string(index=False))
    
    # 使用Altair创建性能比较图表
    df_plot = pd.DataFrame({
        '数据量': sample_sizes * 2,
        '计算时间(秒)': [r['原始函数时间(秒)'] for r in results] + [r['优化函数时间(秒)'] for r in results],
        '函数版本': ['原始函数'] * len(sample_sizes) + ['优化函数'] * len(sample_sizes)
    })
    
    # 创建柱状图
    chart = alt.Chart(df_plot).mark_bar().encode(
        x=alt.X('数据量:N', title='数据量'),
        y=alt.Y('计算时间(秒):Q', title='计算时间(秒)'),
        color=alt.Color('函数版本:N', title='函数版本'),
        tooltip=['数据量', '计算时间(秒)', '函数版本']
    ).properties(
        title='半极端时间计算函数性能比较',
        width=600,
        height=400
    )
    
    # 创建加速比折线图
    speedup_df = pd.DataFrame({
        '数据量': sample_sizes,
        '加速比': [r['加速比'] for r in results]
    })
    
    speedup_chart = alt.Chart(speedup_df).mark_line(
        point=True
    ).encode(
        x=alt.X('数据量:N', title='数据量'),
        y=alt.Y('加速比:Q', title='加速比'),
        tooltip=['数据量', '加速比']
    ).properties(
        title='优化函数加速比',
        width=600,
        height=300
    )
    
    # 保存图表为HTML文件
    (chart & speedup_chart).save('performance_comparison.html')
    print("\n性能比较图表已保存为 performance_comparison.html")
    
    # 测试超时功能
    print("\n测试超时功能:")
    df_large = trade_data.head(100000)  # 取一个更大的数据集
    
    # 使用一个很短的超时时间
    timeout = 0.1  # 0.1秒超时
    
    print(f"设置超时时间为{timeout}秒")
    print(f"优化版函数 - 开始时间: {datetime.now().strftime('%H:%M:%S')}")
    start_time = time.time()
    result_timeout = rust_pyfunc.fast_find_half_extreme_time(
        df_large['time'].values,
        df_large['price'].values,
        time_window=5.0,
        timeout_seconds=timeout
    )
    end_time = time.time()
    timeout_time = end_time - start_time
    print(f"  结束时间: {datetime.now().strftime('%H:%M:%S')}, 耗时: {timeout_time:.4f}秒")
    print(f"  是否全部为NaN: {np.isnan(result_timeout).all()}")
    print(f"  NaN数量: {np.isnan(result_timeout).sum()}/{len(result_timeout)}")
    
    return results, results_df

# 运行比较
if __name__ == "__main__":
    results, results_df = compare_performance()
    
    # 将结果保存为JSON
    with open('performance_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    
    # 保存DataFrame结果
    results_df.to_csv('performance_results.csv', index=False)
    
    print("\n所有测试完成，结果已保存到performance_results.json和performance_results.csv")
