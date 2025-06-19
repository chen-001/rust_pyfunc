import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import altair as alt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os

# 添加项目路径到系统路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 导入rust函数
from rust_pyfunc import dataframe_corrwith, dataframe_corrwith_fast, dataframe_corrwith_turbo

# 设置随机种子以确保结果可复现
np.random.seed(42)

def create_test_dataframes(n_rows, n_cols, nan_ratio=0.05):
    """
    创建测试用的数据框，可以包含一定比例的NaN值
    
    参数:
    - n_rows: 行数
    - n_cols: 列数
    - nan_ratio: NaN值的比例
    
    返回:
    - df1, df2: 两个NumPy数组
    """
    # 创建基础数据
    df1 = np.random.randn(n_rows, n_cols)
    df2 = np.random.randn(n_rows, n_cols)
    
    # 添加一些相关性
    for i in range(n_cols):
        # 随机相关性强度
        corr_strength = np.random.uniform(-1, 1)
        df2[:, i] = df1[:, i] * corr_strength + np.random.randn(n_rows) * (1 - abs(corr_strength))
    
    # 添加NaN值
    if nan_ratio > 0:
        nan_mask1 = np.random.random(size=(n_rows, n_cols)) < nan_ratio
        nan_mask2 = np.random.random(size=(n_rows, n_cols)) < nan_ratio
        df1[nan_mask1] = np.nan
        df2[nan_mask2] = np.nan
    
    return df1, df2

def test_performance(sizes, n_cols=100, repeats=3):
    """
    测试不同数据大小下三个函数的性能
    
    参数:
    - sizes: 行数大小列表
    - n_cols: 列数
    - repeats: 重复测试次数
    
    返回:
    - 性能结果字典
    """
    results = {
        "sizes": sizes,
        "pandas": [],
        "rust_original": [],
        "rust_fast": [],
        "rust_turbo": [],
        "diff_original_fast": [],
        "diff_original_turbo": [],
        "diff_fast_turbo": []
    }
    
    for size in sizes:
        print(f"测试大小: {size}x{n_cols}")
        
        # 生成测试数据
        df1, df2 = create_test_dataframes(size, n_cols)
        
        # pandas实现
        pandas_times = []
        for _ in range(repeats):
            start = time.time()
            pandas_df1 = pd.DataFrame(df1)
            pandas_df2 = pd.DataFrame(df2)
            pandas_result = pandas_df1.corrwith(pandas_df2)
            end = time.time()
            pandas_times.append(end - start)
        
        # Rust原始实现
        rust_original_times = []
        for _ in range(repeats):
            start = time.time()
            rust_original_result = dataframe_corrwith(df1, df2)
            end = time.time()
            rust_original_times.append(end - start)
        
        # Rust优化实现
        rust_fast_times = []
        for _ in range(repeats):
            start = time.time()
            rust_fast_result = dataframe_corrwith_fast(df1, df2)
            end = time.time()
            rust_fast_times.append(end - start)
        
        # Rust Turbo实现
        rust_turbo_times = []
        for _ in range(repeats):
            start = time.time()
            rust_turbo_result = dataframe_corrwith_turbo(df1, df2)
            end = time.time()
            rust_turbo_times.append(end - start)
        
        # 验证结果一致性
        for i in range(n_cols):
            if np.isnan(pandas_result[i]) and np.isnan(rust_original_result[i]) and np.isnan(rust_fast_result[i]) and np.isnan(rust_turbo_result[i]):
                continue
            
            # 检查非NaN值的相对差异
            if not np.isnan(pandas_result[i]) and not np.isnan(rust_original_result[i]):
                assert abs(pandas_result[i] - rust_original_result[i]) < 1e-10, f"Pandas和Rust原始实现结果不一致: {pandas_result[i]} vs {rust_original_result[i]}"
            
            if not np.isnan(rust_original_result[i]) and not np.isnan(rust_fast_result[i]):
                assert abs(rust_original_result[i] - rust_fast_result[i]) < 1e-10, f"Rust原始和Fast实现结果不一致: {rust_original_result[i]} vs {rust_fast_result[i]}"
            
            if not np.isnan(rust_original_result[i]) and not np.isnan(rust_turbo_result[i]):
                assert abs(rust_original_result[i] - rust_turbo_result[i]) < 1e-10, f"Rust原始和Turbo实现结果不一致: {rust_original_result[i]} vs {rust_turbo_result[i]}"
        
        # 记录平均执行时间
        pandas_avg = np.mean(pandas_times)
        rust_original_avg = np.mean(rust_original_times)
        rust_fast_avg = np.mean(rust_fast_times)
        rust_turbo_avg = np.mean(rust_turbo_times)
        
        results["pandas"].append(pandas_avg)
        results["rust_original"].append(rust_original_avg)
        results["rust_fast"].append(rust_fast_avg)
        results["rust_turbo"].append(rust_turbo_avg)
        
        # 计算加速比
        diff_original_fast = rust_original_avg / rust_fast_avg
        diff_original_turbo = rust_original_avg / rust_turbo_avg
        diff_fast_turbo = rust_fast_avg / rust_turbo_avg
        
        results["diff_original_fast"].append(diff_original_fast)
        results["diff_original_turbo"].append(diff_original_turbo)
        results["diff_fast_turbo"].append(diff_fast_turbo)
        
        print(f"  Pandas: {pandas_avg:.4f}s")
        print(f"  Rust原始: {rust_original_avg:.4f}s")
        print(f"  Rust Fast: {rust_fast_avg:.4f}s")
        print(f"  Rust Turbo: {rust_turbo_avg:.4f}s")
        print(f"  Fast vs 原始加速比: {diff_original_fast:.2f}x")
        print(f"  Turbo vs 原始加速比: {diff_original_turbo:.2f}x")
        print(f"  Turbo vs Fast加速比: {diff_fast_turbo:.2f}x")
        print()
    
    return results

def plot_results_plotly(results):
    """使用Plotly绘制性能对比图"""
    # 创建子图布局
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('执行时间对比', '加速比对比'),
        vertical_spacing=0.15,
        specs=[[{"type": "scatter"}], [{"type": "bar"}]]
    )
    
    # 添加执行时间线图
    for name, color in zip(['pandas', 'rust_original', 'rust_fast', 'rust_turbo'], 
                          ['blue', 'green', 'red', 'purple']):
        fig.add_trace(
            go.Scatter(
                x=results['sizes'],
                y=results[name],
                mode='lines+markers',
                name=name,
                line=dict(color=color, width=2),
                marker=dict(size=8)
            ),
            row=1, col=1
        )
    
    # 添加加速比条形图
    x_labels = [f"{size}" for size in results['sizes']]
    bar_width = 0.2
    
    for i, (name, color) in enumerate(zip(
        ['diff_original_fast', 'diff_original_turbo', 'diff_fast_turbo'],
        ['orange', 'teal', 'magenta'])):
        
        # 调整x位置，使条形图并排显示
        x_pos = [j + (i - 1) * bar_width for j in range(len(results['sizes']))]
        
        fig.add_trace(
            go.Bar(
                x=x_labels,
                y=results[name],
                name=name.replace('diff_', ''),
                marker_color=color,
                width=bar_width
            ),
            row=2, col=1
        )
    
    # 更新布局
    fig.update_layout(
        title='DataFrame相关系数计算性能对比',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.05,
            xanchor="center",
            x=0.5
        ),
        height=800
    )
    
    # 更新轴标签
    fig.update_xaxes(title_text='数据框行数', row=1, col=1)
    fig.update_yaxes(title_text='执行时间 (秒)', row=1, col=1)
    fig.update_xaxes(title_text='数据框行数', row=2, col=1)
    fig.update_yaxes(title_text='加速比 (倍)', row=2, col=1)
    
    # 保存HTML文件
    fig.write_html('correlation_performance_comparison.html')
    
    return fig

def main():
    # 测试数据大小
    sizes = [1000, 5000, 10000, 20000, 50000]
    
    # 固定列数
    n_cols = 100
    
    # 进行性能测试
    results = test_performance(sizes, n_cols, repeats=3)
    
    # 使用Plotly绘制结果
    plot_results_plotly(results)
    
    # 打印最终结果
    print("\n最终性能结果:")
    print(f"数据大小: {sizes[-1]}x{n_cols}")
    print(f"Pandas vs Rust原始加速比: {results['pandas'][-1] / results['rust_original'][-1]:.2f}x")
    print(f"Pandas vs Rust Fast加速比: {results['pandas'][-1] / results['rust_fast'][-1]:.2f}x")
    print(f"Pandas vs Rust Turbo加速比: {results['pandas'][-1] / results['rust_turbo'][-1]:.2f}x")
    print(f"Rust原始 vs Fast加速比: {results['diff_original_fast'][-1]:.2f}x")
    print(f"Rust原始 vs Turbo加速比: {results['diff_original_turbo'][-1]:.2f}x")
    print(f"Rust Fast vs Turbo加速比: {results['diff_fast_turbo'][-1]:.2f}x")

if __name__ == "__main__":
    main()
