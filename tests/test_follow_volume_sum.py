import pandas as pd
import numpy as np
import time
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os
from pathlib import Path

# 添加项目根目录到PATH
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# 导入Rust函数
from rust_pyfunc import find_follow_volume_sum_same_price

# Python实现的相同功能函数
def py_find_follow_volume_sum_same_price(times, prices, volumes, time_window=0.1, check_price=True, filter_ratio=0.0):
    """
    Python版实现，功能与Rust版相同
    
    参数:
    ------
    times : numpy.ndarray
        时间戳数组（单位：秒）
    prices : numpy.ndarray
        价格数组
    volumes : numpy.ndarray
        成交量数组
    time_window : float, optional, default=0.1
        时间窗口（单位：秒）
    check_price : bool, optional, default=True
        是否检查价格是否相同
    filter_frequent_volumes : bool, optional, default=False
        是否过滤频繁出现的相同volume值
        
    返回:
    ------
    numpy.ndarray
        每一行在其后time_window秒内具有相同条件的行的volume总和
    """
    # 将时间转换为与Rust版本一致的格式
    times = times / 1.0e9
    n = len(times)
    result = np.zeros(n, dtype=float)
    
    # 首先计算每个点的volume总和，并将没有匹配项的点标记为NaN
    for i in range(n):
        current_time = times[i]
        current_price = prices[i]
        current_volume = volumes[i]
        sum_volume = current_volume  # 包含当前点的成交量
        has_match = False  # 记录是否有匹配
        
        # 检查之后的点
        for j in range(i + 1, n):
            # 如果时间差超过time_window秒，退出内层循环
            if times[j] - current_time > time_window:
                break
            
            # 根据check_price参数决定是否检查价格
            price_match = not check_price or abs(prices[j] - current_price) < 1e-10
            volume_match = abs(volumes[j] - current_volume) < 1e-10
            
            if price_match and volume_match:
                has_match = True
                sum_volume += volumes[j]
        
        # 如果没有找到匹配项，则设为NaN
        if not has_match:
            result[i] = np.nan
        else:
            result[i] = sum_volume
            
    # 如果需要过滤频繁出现的volume，统计非NaN点的volume频率
    volume_counts = {}
    if filter_ratio > 0.0:
        # 只统计非NaN点的volume频率
        for i in range(n):
            # 跳过NaN点
            if np.isnan(result[i]):
                continue
                
            current_volume = volumes[i]
            if current_volume in volume_counts:
                volume_counts[current_volume] += 1
            else:
                volume_counts[current_volume] = 1
        
        # 过滤出现频率最高的前30%的volume类型
        if volume_counts:
            # 将volume按照频率从高到低排序
            sorted_volumes = sorted(volume_counts.items(), key=lambda x: x[1], reverse=True)
            
            # 计算需要过滤的volume种类数量（根据filter_ratio参数）
            total_types = len(sorted_volumes)
            filter_count = int(np.ceil(total_types * filter_ratio))
            
            # 确保至少过滤一种如果有多种类型
            filter_count = 1 if filter_count == 0 and total_types > 0 else filter_count
            
            # 选取出现频率最高的前几种volume类型
            volume_to_filter = [vol for vol, _ in sorted_volumes[:filter_count]]
            
            # 将高频率volume对应的行设为NaN
            for i in range(n):
                if any(abs(volumes[i] - v) < 1e-10 for v in volume_to_filter):
                    result[i] = np.nan
    
    # 计算每个点在time_window内的volume总和
    for i in range(n):
        # 如果已经被标记为NaN，则跳过
        if filter_ratio > 0.0 and np.isnan(result[i]):
            continue
            
        current_time = times[i]
        current_price = prices[i]
        current_volume = volumes[i]
        sum_volume = current_volume  # 包含当前点的成交量
        
        # 检查之后的点
        for j in range(i + 1, n):
            # 如果时间差超过time_window秒，退出内层循环
            if times[j] - current_time > time_window:
                break
            
            # 根据check_price参数决定是否检查价格
            price_match = not check_price or abs(prices[j] - current_price) < 1e-10
            volume_match = abs(volumes[j] - current_volume) < 1e-10
            
            if price_match and volume_match:
                sum_volume += volumes[j]
        
        # 如果计算结果与该行的volume相等，说明没有找到其他匹配的点，将其设为NaN
        if abs(sum_volume - current_volume) < 1e-10:
            result[i] = np.nan
        else:
            result[i] = sum_volume
    
    return result

def run_test(size=10000, time_window=0.1, check_price=True, filter_ratio=0.0):
    """运行测试并比较结果和性能"""
    print(f"测试参数: size={size}, time_window={time_window}, check_price={check_price}, filter_ratio={filter_ratio}")
    
    # 生成测试数据
    np.random.seed(42)
    times = np.cumsum(np.abs(np.random.normal(0.01, 0.005, size)) * 1e9)  # 纳秒时间戳
    prices = np.random.choice(np.linspace(10.0, 20.0, 20), size)
    volumes = np.random.choice([100.0, 200.0, 300.0, 400.0, 500.0], size).astype(np.float64)  # 确保是float64类型
    
    # 测试Rust实现
    start_time = time.time()
    rust_result = find_follow_volume_sum_same_price(
        times, prices, volumes, 
        time_window=time_window, 
        check_price=check_price, 
        filter_ratio=filter_ratio
    )
    rust_time = time.time() - start_time
    
    # 测试Python实现
    start_time = time.time()
    py_result = py_find_follow_volume_sum_same_price(
        times, prices, volumes, 
        time_window=time_window, 
        check_price=check_price, 
        filter_ratio=filter_ratio
    )
    py_time = time.time() - start_time
    
    # 速度对比
    speedup = py_time / rust_time
    print(f"Rust实现用时: {rust_time:.6f} 秒")
    print(f"Python实现用时: {py_time:.6f} 秒")
    print(f"Rust速度提升倍数: {speedup:.2f}x")
    
    # 结果比较
    # 现在所有情况下都可能有NaN值，所以统一使用相同的比较方法
    # 首先检查NaN位置是否一致
    rust_nan_mask = np.isnan(rust_result)
    py_nan_mask = np.isnan(py_result)
    nan_match = np.array_equal(rust_nan_mask, py_nan_mask)
    
    # 如果有非NaN值，则检查这些值是否近似相等
    if np.any(~rust_nan_mask) and np.any(~py_nan_mask):
        non_nan_match = np.allclose(
            np.array(rust_result)[~rust_nan_mask], 
            py_result[~py_nan_mask], 
            rtol=1e-10, atol=1e-10
        )
    else:
        # 如果所有值都是NaN，我们已经比较了NaN位置，所以这里也是匹配的
        non_nan_match = True
    
    result_match = nan_match and non_nan_match
    
    print(f"结果一致: {result_match}")
    
    if not result_match:
        # 计算差异统计
        diff = np.abs(np.array(rust_result) - py_result)
        max_diff = np.nanmax(diff)
        mean_diff = np.nanmean(diff)
        print(f"最大差异: {max_diff}")
        print(f"平均差异: {mean_diff}")
    
    return {
        "rust_time": rust_time,
        "py_time": py_time,
        "speedup": speedup,
        "result_match": result_match
    }

def generate_performance_plot(results):
    """生成性能比较图表"""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["执行时间对比", "速度提升倍数"],
        specs=[[{"type": "bar"}, {"type": "bar"}]]
    )
    
    # 准备数据
    test_cases = list(results.keys())
    rust_times = [results[case]["rust_time"] for case in test_cases]
    py_times = [results[case]["py_time"] for case in test_cases]
    speedups = [results[case]["speedup"] for case in test_cases]
    
    # 执行时间对比
    fig.add_trace(
        go.Bar(name="Rust", x=test_cases, y=rust_times, marker_color='#2c3e50'),
        row=1, col=1
    )
    fig.add_trace(
        go.Bar(name="Python", x=test_cases, y=py_times, marker_color='#3498db'),
        row=1, col=1
    )
    
    # 速度提升倍数
    fig.add_trace(
        go.Bar(x=test_cases, y=speedups, marker_color='#e67e22'),
        row=1, col=2
    )
    
    # 更新布局
    fig.update_layout(
        title="Rust vs Python性能比较",
        height=500,
        width=1000,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    # Y轴标签
    fig.update_yaxes(title_text="执行时间 (秒)", row=1, col=1)
    fig.update_yaxes(title_text="速度提升倍数", row=1, col=2)
    
    # 保存HTML
    html_file = os.path.join(project_root, "tests", "follow_volume_sum_performance.html")
    fig.write_html(html_file)
    print(f"性能对比图表已保存至: {html_file}")

def main():
    # 运行不同测试用例
    results = {}
    
    # 测试用例1: 默认参数 (check_price=True, filter_ratio=0.0)
    results["\u9ed8\u8ba4\u53c2\u6570"] = run_test(size=100000, check_price=True, filter_ratio=0.0)
    print("\n" + "-" * 50 + "\n")
    
    # 测试用例2: 不检查价格 (check_price=False)
    results["\u4e0d\u68c0\u67e5\u4ef7\u683c"] = run_test(size=100000, check_price=False, filter_ratio=0.0)
    print("\n" + "-" * 50 + "\n")
    
    # 测试用例3: 过滤频繁volumes (filter_ratio=0.3)
    results["\u8fc7\u6ee4\u9891\u7e41volumes"] = run_test(size=100000, check_price=True, filter_ratio=0.3)
    print("\n" + "-" * 50 + "\n")
    
    # 测试用例4: 组合参数 (check_price=False, filter_ratio=0.3)
    results["\u7ec4\u5408\u53c2\u6570"] = run_test(size=100000, check_price=False, filter_ratio=0.3)
    
    # 生成性能比较图表
    generate_performance_plot(results)

if __name__ == "__main__":
    main()
