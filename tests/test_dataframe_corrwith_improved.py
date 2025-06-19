import numpy as np
import pandas as pd
import time
import altair as alt
from rust_pyfunc import dataframe_corrwith, dataframe_corrwith_fast

def test_small_dataframe():
    """测试小型数据框下两个函数的结果一致性"""
    # 创建两个简单的数据框
    df1 = pd.DataFrame({
        'A': [1.0, 2.0, 3.0, 4.0, 5.0],
        'B': [5.0, 4.0, 3.0, 2.0, 1.0],
        'C': [2.0, 4.0, 6.0, 8.0, 10.0]
    })
    df2 = pd.DataFrame({
        'A': [1.1, 2.2, 2.9, 4.1, 5.2],
        'B': [5.2, 4.1, 2.9, 2.1, 0.9],
        'D': [1.0, 2.0, 3.0, 4.0, 5.0]
    })
    
    # 使用标准版本
    result_standard = dataframe_corrwith(df1.values, df2.values)
    
    # 使用优化版本
    result_fast = dataframe_corrwith_fast(df1.values, df2.values)
    
    # 检查结果是否一致
    print("\n=== 小数据量测试结果 ===")
    print("标准版本结果:")
    print(pd.Series(result_standard, index=df1.columns))
    print("\n优化版本结果:")
    print(pd.Series(result_fast, index=df1.columns))
    
    # 计算差异
    diff = np.abs(result_standard - result_fast)
    print(f"\n最大差异: {diff.max()}")
    
    # 验证结果是否足够接近 (考虑浮点精度)
    assert np.allclose(result_standard, result_fast, rtol=1e-10, atol=1e-10), "结果不一致"
    print("✓ 小数据量结果验证通过")
    
    return True

def test_with_nan_values():
    """测试包含NaN值的情况"""
    # 创建包含NaN的数据框
    df1 = pd.DataFrame({
        'A': [1.0, 2.0, np.nan, 4.0, 5.0],
        'B': [5.0, np.nan, 3.0, 2.0, 1.0],
        'C': [2.0, 4.0, 6.0, np.nan, 10.0]
    })
    df2 = pd.DataFrame({
        'A': [1.1, 2.2, np.nan, 4.1, 5.2],
        'B': [5.2, np.nan, 2.9, 2.1, 0.9],
        'D': [1.0, 2.0, 3.0, 4.0, np.nan]
    })
    
    # 使用标准版本
    result_standard = dataframe_corrwith(df1.values, df2.values)
    
    # 使用优化版本
    result_fast = dataframe_corrwith_fast(df1.values, df2.values)
    
    # 比较结果
    print("\n=== 含NaN值测试结果 ===")
    print("标准版本结果:")
    print(pd.Series(result_standard, index=df1.columns))
    print("\n优化版本结果:")
    print(pd.Series(result_fast, index=df1.columns))
    
    # 验证结果是否足够接近
    assert np.allclose(result_standard, result_fast, equal_nan=True, rtol=1e-10, atol=1e-10), "结果不一致"
    print("✓ 含NaN值结果验证通过")
    
    return True

def benchmark_performance():
    """测试大数据框下的性能"""
    # 生成大型数据框: 5000 x 5000
    n_rows, n_cols = 5000, 5000
    
    # 使用随机数生成数据
    np.random.seed(42)
    df1_data = np.random.randn(n_rows, n_cols)
    df2_data = np.random.randn(n_rows, n_cols)
    
    # 添加一些NaN值作为测试
    nan_indices = np.random.choice(n_rows * n_cols, size=int(n_rows * n_cols * 0.01), replace=False)
    df1_data.ravel()[nan_indices] = np.nan
    
    nan_indices = np.random.choice(n_rows * n_cols, size=int(n_rows * n_cols * 0.01), replace=False)
    df2_data.ravel()[nan_indices] = np.nan
    
    print(f"\n=== 性能测试: {n_rows}x{n_cols} 数据框 ===")
    
    # 测量pandas corrwith性能
    print("开始测试pandas.DataFrame.corrwith性能...")
    pandas_start = time.time()
    pandas_result = pd.DataFrame(df1_data).corrwith(pd.DataFrame(df2_data))
    pandas_end = time.time()
    pandas_time = pandas_end - pandas_start
    print(f"Pandas原生corrwith耗时: {pandas_time:.4f}秒")
    
    # 测量原始版本corrwith性能
    print("开始测试Rust标准版corrwith性能...")
    rust_std_start = time.time()
    rust_std_result = dataframe_corrwith(df1_data, df2_data)
    rust_std_end = time.time()
    rust_std_time = rust_std_end - rust_std_start
    print(f"Rust标准版corrwith耗时: {rust_std_time:.4f}秒")
    print(f"相比Pandas速度提升: {pandas_time / rust_std_time:.2f}倍")
    
    # 测量优化版本corrwith性能
    print("开始测试Rust优化版corrwith_fast性能...")
    rust_fast_start = time.time()
    rust_fast_result = dataframe_corrwith_fast(df1_data, df2_data)
    rust_fast_end = time.time()
    rust_fast_time = rust_fast_end - rust_fast_start
    print(f"Rust优化版corrwith_fast耗时: {rust_fast_time:.4f}秒")
    
    # 计算速度提升
    std_vs_fast_speedup = rust_std_time / rust_fast_time
    pandas_vs_fast_speedup = pandas_time / rust_fast_time
    print(f"优化版相比标准版速度提升: {std_vs_fast_speedup:.2f}倍")
    print(f"优化版相比Pandas速度提升: {pandas_vs_fast_speedup:.2f}倍")
    
    # 验证结果差异
    # 转换pandas结果为numpy数组以便比较
    pandas_result_array = pandas_result.values
    max_diff_std_pandas = np.nanmax(np.abs(pandas_result_array - rust_std_result[:len(pandas_result_array)]))
    max_diff_fast_pandas = np.nanmax(np.abs(pandas_result_array - rust_fast_result[:len(pandas_result_array)]))
    max_diff_std_fast = np.nanmax(np.abs(rust_std_result - rust_fast_result))
    
    print(f"\n结果差异分析:")
    print(f"标准版与Pandas最大差异: {max_diff_std_pandas}")
    print(f"优化版与Pandas最大差异: {max_diff_fast_pandas}")
    print(f"标准版与优化版最大差异: {max_diff_std_fast}")
    
    # 验证两个Rust版本结果是否一致
    assert np.allclose(rust_std_result, rust_fast_result, equal_nan=True, rtol=1e-10, atol=1e-10), "标准版与优化版结果不一致"
    print("✓ 大数据量结果验证通过，标准版与优化版结果一致")
    
    # 创建性能对比可视化
    data = pd.DataFrame({
        '实现方式': ['Pandas', 'Rust标准版', 'Rust优化版'],
        '运行时间（秒）': [pandas_time, rust_std_time, rust_fast_time]
    })
    
    # 使用Altair创建条形图
    chart = alt.Chart(data).mark_bar().encode(
        x=alt.X('实现方式', title='实现方式'),
        y=alt.Y('运行时间（秒）', title='运行时间（秒）'),
        color='实现方式'
    ).properties(
        title=f'Pandas vs Rust corrwith性能对比 ({n_rows}x{n_cols})',
        width=500,
        height=300
    )
    
    # 保存图表为HTML文件
    chart.save('/home/chenzongwei/rustcode/rust_pyfunc/tests/corrwith_performance_improved.html')
    print(f"性能对比图已保存至 /home/chenzongwei/rustcode/rust_pyfunc/tests/corrwith_performance_improved.html")
    
    return pandas_time, rust_std_time, rust_fast_time

if __name__ == "__main__":
    test_small_dataframe()
    test_with_nan_values()
    pandas_time, rust_std_time, rust_fast_time = benchmark_performance()
    
    print("\n=== 总结 ===")
    print(f"在5000x5000数据量下:")
    print(f"Pandas原生corrwith耗时: {pandas_time:.4f}秒")
    print(f"Rust标准版corrwith耗时: {rust_std_time:.4f}秒")
    print(f"Rust优化版corrwith_fast耗时: {rust_fast_time:.4f}秒")
    print(f"优化版相比标准版速度提升: {rust_std_time/rust_fast_time:.2f}倍")
    print(f"优化版相比Pandas速度提升: {pandas_time/rust_fast_time:.2f}倍")
