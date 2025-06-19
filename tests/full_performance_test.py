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

def run_full_performance_test():
    """运行完整数据集性能测试，比较三个版本的半极端时间计算函数"""
    print("三个版本的半极端时间计算函数性能比较（完整数据集）")
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
    
    # 测试数据量 - 使用完整数据集
    n = len(trade_data)
    print(f"将使用全部{n}条数据测试性能")
    
    # 存储结果
    results = []
    warmup_size = 5000  # 预热大小
    
    # 预热 - 使用较小数据集
    print(f"\n预热阶段 - 使用{warmup_size}条数据...")
    _ = rust_pyfunc.find_half_extreme_time(times[:warmup_size], prices[:warmup_size])
    _ = rust_pyfunc.fast_find_half_extreme_time(times[:warmup_size], prices[:warmup_size])
    _ = rust_pyfunc.super_find_half_extreme_time(times[:warmup_size], prices[:warmup_size])
    
    # 正式测试 - 完整数据集
    print(f"\n正式测试阶段 - 使用全部{n}条数据")
    
    # 测试原始函数
    start_time = time.time()
    print(f"原始函数 - 开始时间: {datetime.now().strftime('%H:%M:%S')}")
    _ = rust_pyfunc.find_half_extreme_time(times, prices, time_window=5.0)
    end_time = time.time()
    original_time = end_time - start_time
    print(f"  结束时间: {datetime.now().strftime('%H:%M:%S')}, 耗时: {original_time:.4f}秒")
    
    # 测试优化函数
    start_time = time.time()
    print(f"快速优化函数 - 开始时间: {datetime.now().strftime('%H:%M:%S')}")
    _ = rust_pyfunc.fast_find_half_extreme_time(times, prices, time_window=5.0)
    end_time = time.time()
    fast_time = end_time - start_time
    print(f"  结束时间: {datetime.now().strftime('%H:%M:%S')}, 耗时: {fast_time:.4f}秒")
    
    # 测试超级优化函数
    start_time = time.time()
    print(f"超级优化函数 - 开始时间: {datetime.now().strftime('%H:%M:%S')}")
    _ = rust_pyfunc.super_find_half_extreme_time(times, prices, time_window=5.0)
    end_time = time.time()
    super_time = end_time - start_time
    print(f"  结束时间: {datetime.now().strftime('%H:%M:%S')}, 耗时: {super_time:.4f}秒")
    
    # 计算加速比
    fast_speedup = original_time / fast_time if fast_time > 0 else float('inf')
    super_speedup = original_time / super_time if super_time > 0 else float('inf')
    super_vs_fast = fast_time / super_time if super_time > 0 else float('inf')
    
    print(f"\n性能比较结果:")
    print(f"  原始函数耗时: {original_time:.4f}秒")
    print(f"  快速函数耗时: {fast_time:.4f}秒, 相比原始函数加速: {fast_speedup:.2f}倍")
    print(f"  超级函数耗时: {super_time:.4f}秒, 相比原始函数加速: {super_speedup:.2f}倍")
    print(f"  超级函数相比快速函数加速: {super_vs_fast:.2f}倍")
    
    # 记录结果
    result = {
        "数据量": n,
        "原始函数时间(秒)": original_time,
        "快速函数时间(秒)": fast_time,
        "超级函数时间(秒)": super_time,
        "快速函数加速比": fast_speedup,
        "超级函数加速比": super_speedup,
        "超级vs快速加速比": super_vs_fast
    }
    
    # 绘制性能比较图表
    df_perf = pd.DataFrame({
        '函数版本': ['原始版本', '快速版本', '超级版本'],
        '计算时间(秒)': [original_time, fast_time, super_time]
    })
    
    # 创建条形图
    chart = alt.Chart(df_perf).mark_bar().encode(
        x='函数版本:N',
        y='计算时间(秒):Q',
        color='函数版本:N',
        tooltip=['函数版本', '计算时间(秒)']
    ).properties(
        title=f'半极端时间计算函数性能比较 ({n}条数据)',
        width=600,
        height=400
    )
    
    # 保存图表
    chart.save('full_performance_comparison.html')
    print("性能比较图表已保存到 full_performance_comparison.html")
    
    return result

if __name__ == "__main__":
    try:
        result = run_full_performance_test()
        # 保存结果到CSV
        pd.DataFrame([result]).to_csv('full_performance_results.csv', index=False)
        print("\n测试完成，结果已保存到 full_performance_results.csv")
    except Exception as e:
        print(f"测试过程中发生错误: {e}")
