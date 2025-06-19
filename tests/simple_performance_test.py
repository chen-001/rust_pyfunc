import numpy as np
import pandas as pd
import time
from datetime import datetime
import altair as alt

# 确保使用正确的Python路径
import sys
sys.path.append('/home/chenzongwei/rustcode/rust_pyfunc')
import rust_pyfunc
import design_whatever as dw

def run_simple_performance_test():
    """运行简单的性能测试，比较find_half_extreme_time和fast_find_half_extreme_time函数"""
    print("比较find_half_extreme_time与fast_find_half_extreme_time的性能差异")
    print("="*80)
    
    # 读取股票的逐笔成交数据
    print("正在读取股票数据...")
    trade_data = dw.read_l2_trade_data(start_date=20220819, symbols=['000001'], with_retreat=0)
    print(f"读取完成，共{len(trade_data)}条记录")
    
    # 数据预处理 - 转换为numpy数组
    print("数据预处理...")
    # 将时间戳转换为浮点数
    times = pd.Series(trade_data['exchtime'].view(np.int64)).astype(np.float64).values
    prices = trade_data['price'].values
    
    # 测试数据大小
    sample_sizes = [1000, 5000, 10000, 50000]
    
    # 存储结果
    results = []
    
    # 在不同数据量下测试
    for size in sample_sizes:
        print(f"\n测试数据量: {size}")
        times_sample = times[:size]
        prices_sample = prices[:size]
        
        # 测试原始函数
        start_time = time.time()
        print(f"原始函数 - 开始时间: {datetime.now().strftime('%H:%M:%S')}")
        _ = rust_pyfunc.find_half_extreme_time(times_sample, prices_sample, time_window=5.0)
        end_time = time.time()
        original_time = end_time - start_time
        print(f"  结束时间: {datetime.now().strftime('%H:%M:%S')}, 耗时: {original_time:.4f}秒")
        
        # 测试优化函数
        start_time = time.time()
        print(f"优化函数 - 开始时间: {datetime.now().strftime('%H:%M:%S')}")
        _ = rust_pyfunc.fast_find_half_extreme_time(times_sample, prices_sample, time_window=5.0)
        end_time = time.time()
        fast_time = end_time - start_time
        print(f"  结束时间: {datetime.now().strftime('%H:%M:%S')}, 耗时: {fast_time:.4f}秒")
        
        # 计算加速比
        speedup = original_time / fast_time if fast_time > 0 else float('inf')
        print(f"  加速比: {speedup:.2f}倍")
        
        # 记录结果
        results.append({
            "数据量": size,
            "原始函数时间(秒)": original_time,
            "优化函数时间(秒)": fast_time,
            "加速比": speedup
        })
    
    # 结果转为DataFrame
    results_df = pd.DataFrame(results)
    print("\n性能比较结果:")
    print(results_df.to_string(index=False))
    
    # 绘制性能比较图表
    df_perf = pd.DataFrame({
        '数据量': [str(r['数据量']) for r in results] * 2,
        '计算时间(秒)': [r['原始函数时间(秒)'] for r in results] + [r['优化函数时间(秒)'] for r in results],
        '版本': ['原始版本'] * len(results) + ['优化版本'] * len(results)
    })
    
    # 创建条形图
    chart = alt.Chart(df_perf).mark_bar().encode(
        x='数据量:N',
        y='计算时间(秒):Q',
        color='版本:N',
        tooltip=['数据量', '计算时间(秒)', '版本']
    ).properties(
        title='半极端时间计算函数性能比较',
        width=600,
        height=400
    )
    
    # 保存图表
    chart.save('performance_comparison.html')
    print("性能比较图表已保存到 performance_comparison.html")
    
    # 测试超时机制
    print("\n测试超时机制:")
    large_sample_size = 100000
    times_large = times[:large_sample_size]
    prices_large = prices[:large_sample_size]
    
    timeout = 0.1  # 0.1秒超时
    print(f"设置超时时间为{timeout}秒")
    
    # 测试优化版本的超时
    start_time = time.time()
    print(f"优化版函数(带超时) - 开始时间: {datetime.now().strftime('%H:%M:%S')}")
    result_timeout = rust_pyfunc.fast_find_half_extreme_time(
        times_large, prices_large, time_window=5.0, timeout_seconds=timeout
    )
    end_time = time.time()
    timeout_time = end_time - start_time
    print(f"  结束时间: {datetime.now().strftime('%H:%M:%S')}, 耗时: {timeout_time:.4f}秒")
    
    # 检查超时后的结果是否全是NaN
    nan_count = np.sum(np.isnan(result_timeout))
    print(f"  NaN数量: {nan_count}/{large_sample_size}")
    print(f"  是否全部为NaN: {nan_count == large_sample_size}")
    
    return results_df

if __name__ == "__main__":
    df_results = run_simple_performance_test()
    # 保存结果到CSV
    df_results.to_csv('performance_results.csv', index=False)
    print("\n测试完成，结果已保存到 performance_results.csv")
