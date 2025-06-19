import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import altair as alt
from rust_pyfunc import dataframe_corrwith

def test_small_dataframe():
    """测试小型数据框下结果的正确性"""
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
    
    # 使用pandas的corrwith
    pandas_result = df1.corrwith(df2)
    
    # 使用rust实现的corrwith
    rust_result = dataframe_corrwith(df1.values, df2.values)
    rust_result_series = pd.Series(rust_result, index=df1.columns)
    
    # 比较结果
    print("\n=== 小数据量测试结果 ===")
    print("Pandas corrwith结果:")
    print(pandas_result)
    print("\nRust corrwith结果:")
    print(rust_result_series)
    
    # 计算差异
    pd_values = pandas_result.iloc[:2].values  # 只取A和B两列的结果进行比较
    rust_values = rust_result[:2]
    diff = np.abs(pd_values - rust_values)
    print(f"\n最大差异: {diff.max()}")
    
    # 验证结果是否足够接近 (考虑浮点精度)
    assert np.allclose(pd_values, rust_values, rtol=1e-10, atol=1e-10), "结果不一致"
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
    
    # 使用pandas的corrwith
    pandas_result = df1.corrwith(df2)
    
    # 使用rust实现的corrwith
    rust_result = dataframe_corrwith(df1.values, df2.values)
    rust_result_series = pd.Series(rust_result, index=df1.columns)
    
    # 比较结果
    print("\n=== 含NaN值测试结果 ===")
    print("Pandas corrwith结果:")
    print(pandas_result)
    print("\nRust corrwith结果:")
    print(rust_result_series)
    
    # 检查是否都是NaN
    is_nan_pandas = pandas_result.isna()
    is_nan_rust = pd.Series([np.isnan(x) for x in rust_result], index=df1.columns)
    
    # 验证非NaN值是否足够接近
    for col in ['A', 'B']:
        if not is_nan_pandas[col] and not is_nan_rust[col]:
            assert np.isclose(pandas_result[col], rust_result_series[col], rtol=1e-10, atol=1e-10), f"列 {col} 结果不一致"
    
    # 验证NaN的位置是否一致
    assert np.array_equal(is_nan_pandas[:2], is_nan_rust[:2]), "NaN值位置不一致"
    
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
    
    # 转换为pandas数据框
    df1 = pd.DataFrame(df1_data)
    df2 = pd.DataFrame(df2_data)
    
    print(f"\n=== 性能测试: {n_rows}x{n_cols} 数据框 ===")
    
    # 测量pandas corrwith性能
    print("开始测试Pandas corrwith性能...")
    pandas_start = time.time()
    pandas_result = df1.corrwith(df2)
    pandas_end = time.time()
    pandas_time = pandas_end - pandas_start
    print(f"Pandas corrwith耗时: {pandas_time:.4f}秒")
    
    # 测量rust corrwith性能
    print("开始测试Rust corrwith性能...")
    rust_start = time.time()
    rust_result = dataframe_corrwith(df1_data, df2_data)
    rust_end = time.time()
    rust_time = rust_end - rust_start
    print(f"Rust corrwith耗时: {rust_time:.4f}秒")
    
    # 计算速度提升
    speedup = pandas_time / rust_time
    print(f"Rust版本速度提升: {speedup:.2f}倍")
    
    # 验证结果差异
    diff = np.abs(pandas_result.values - rust_result)
    max_diff = np.max(diff)
    print(f"最大结果差异: {max_diff}")
    
    assert np.allclose(pandas_result.values, rust_result, rtol=1e-10, atol=1e-10), "大数据量结果不一致"
    print("✓ 大数据量结果验证通过")
    
    # 创建性能对比可视化
    data = pd.DataFrame({
        '实现方式': ['Pandas', 'Rust'],
        '运行时间（秒）': [pandas_time, rust_time]
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
    chart.save('/home/chenzongwei/rustcode/rust_pyfunc/tests/corrwith_performance.html')
    print(f"性能对比图已保存至 /home/chenzongwei/rustcode/rust_pyfunc/tests/corrwith_performance.html")
    
    return speedup, pandas_time, rust_time

if __name__ == "__main__":
    test_small_dataframe()
    test_with_nan_values()
    speedup, pandas_time, rust_time = benchmark_performance()
    
    print("\n=== 总结 ===")
    print(f"在5000x5000数据量下:")
    print(f"Pandas corrwith耗时: {pandas_time:.4f}秒")
    print(f"Rust corrwith耗时: {rust_time:.4f}秒")
    print(f"Rust版本速度提升: {speedup:.2f}倍")
