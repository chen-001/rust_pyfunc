import numpy as np
import pandas as pd
import time
from datetime import datetime
import altair as alt

# 导入模块
import sys
sys.path.append('/home/chenzongwei/rustcode/rust_pyfunc')
import rust_pyfunc
import design_whatever as dw

def test_follow_volume_sum_timeout():
    """测试查找相同价格后续成交量函数的超时机制"""
    print("测试find_follow_volume_sum_same_price函数的超时机制")
    print("="*80)
    
    # 读取股票的逐笔成交数据
    print("正在读取股票数据...")
    trade_data = dw.read_l2_trade_data(start_date=20220819, symbols=['000001'], with_retreat=0)
    print(f"读取完成，共{len(trade_data)}条记录")
    
    # 数据预处理
    print("数据预处理...")
    # 转换数据类型
    times = pd.Series(trade_data['exchtime'].view(np.int64)).astype(np.float64).values
    prices = trade_data['price'].values
    volumes = trade_data['volume'].astype(np.float64).values
    
    # 不同数据大小的测试
    sample_sizes = [1000, 10000, 50000, 100000]
    results = []
    
    print("\n正常执行测试（无超时）:")
    for size in sample_sizes:
        if size > len(trade_data):
            size = len(trade_data)
            
        print(f"\n测试数据量: {size}")
        times_sample = times[:size]
        prices_sample = prices[:size]
        volumes_sample = volumes[:size]
        
        # 无超时测试
        start_time = time.time()
        result = rust_pyfunc.find_follow_volume_sum_same_price(
            times_sample, prices_sample, volumes_sample, 
            time_window=0.1, check_price=True, filter_ratio=0.1
        )
        end_time = time.time()
        duration = end_time - start_time
        
        nan_count = np.sum(np.isnan(result))
        print(f"  执行时间: {duration:.4f}秒")
        print(f"  NaN数量: {nan_count}/{size} ({nan_count/size*100:.2f}%)")
        
        results.append({
            "数据量": size,
            "执行时间(秒)": duration,
            "是否超时": False,
            "NaN比例": nan_count/size
        })
    
    print("\n超时测试:")
    # 使用最大的数据量进行超时测试
    size = sample_sizes[-1]
    if size > len(trade_data):
        size = len(trade_data)
        
    times_sample = times[:size]
    prices_sample = prices[:size]
    volumes_sample = volumes[:size]
    
    # 设置一个很短的超时时间，确保会触发超时
    timeout = 0.01  # 10毫秒
    print(f"测试数据量: {size}, 超时设置: {timeout}秒")
    
    start_time = time.time()
    result_timeout = rust_pyfunc.find_follow_volume_sum_same_price(
        times_sample, prices_sample, volumes_sample,
        time_window=0.1, check_price=True, filter_ratio=0.1,
        timeout_seconds=timeout
    )
    end_time = time.time()
    duration = end_time - start_time
    
    nan_count = np.sum(np.isnan(result_timeout))
    is_all_nan = nan_count == size
    
    print(f"  执行时间: {duration:.4f}秒")
    print(f"  NaN数量: {nan_count}/{size} ({nan_count/size*100:.2f}%)")
    print(f"  是否全部为NaN: {is_all_nan}")
    
    results.append({
        "数据量": size,
        "执行时间(秒)": duration,
        "是否超时": True,
        "NaN比例": nan_count/size
    })
    
    # 创建结果可视化
    df_results = pd.DataFrame(results)
    
    # 创建执行时间图表
    chart_time = alt.Chart(df_results).mark_bar().encode(
        x=alt.X('数据量:N', title='数据量'),
        y=alt.Y('执行时间(秒):Q', title='执行时间(秒)'),
        color=alt.Color('是否超时:N', title='是否超时'),
        tooltip=['数据量', '执行时间(秒)', '是否超时', 'NaN比例']
    ).properties(
        title='find_follow_volume_sum_same_price函数执行时间',
        width=500
    )
    
    # 保存图表
    chart_time.save('volume_sum_timeout_test.html')
    print("\n测试图表已保存到 volume_sum_timeout_test.html")
    
    return df_results

if __name__ == "__main__":
    results = test_follow_volume_sum_timeout()
    print("\n测试完成!")
