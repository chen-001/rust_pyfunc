"""
测试优化后的dataframe_corrwith函数的性能和准确性
比较与pandas中df.corrwith的结果一致性和速度提升
"""

import pandas as pd
import numpy as np
import time
from rust_pyfunc.pandas_corrwith import corrwith as rust_corrwith
import rust_pyfunc  # 引入rust实现

# 设置随机种子，确保结果可重现
np.random.seed(42)

def test_small_dataframe():
    """测试小型数据集上的结果正确性"""
    print("=== 测试小型数据集上的准确性 ===")
    # 创建测试数据
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
    
    # 使用pandas计算相关系数
    pandas_result = df1.corrwith(df2)
    
    # 使用rust_corrwith计算相关系数
    rust_result = rust_corrwith(df1, df2)
    
    # 比较结果
    print("Pandas结果:")
    print(pandas_result)
    print("\nRust结果:")
    print(rust_result)
    
    # 过滤只保留两者都有的列
    common_cols = list(set(pandas_result.index) & set(rust_result.index))
    pandas_filtered = pandas_result[common_cols]
    rust_filtered = rust_result[common_cols]
    
    # 检查结果是否接近（考虑浮点数精度差异）
    is_close = np.allclose(pandas_filtered.values, rust_filtered.values, rtol=1e-5)
    print(f"\n结果一致性检查: {'通过' if is_close else '失败'}"
          f" (比较了共同的{len(common_cols)}列)")
    
    return is_close

def test_nan_handling():
    """测试NaN值处理的正确性"""
    print("\n=== 测试NaN值处理 ===")
    # 创建包含NaN的测试数据
    df1 = pd.DataFrame({
        'A': [1.0, 2.0, np.nan, 4.0, 5.0],
        'B': [5.0, 4.0, 3.0, np.nan, 1.0]
    })
    df2 = pd.DataFrame({
        'A': [1.1, np.nan, 2.9, 4.1, 5.2],
        'B': [5.2, 4.1, np.nan, 2.1, 0.9]
    })
    
    # 测试drop_na=True (默认)
    pandas_result = df1.corrwith(df2)
    rust_result = rust_corrwith(df1, df2)
    
    # 比较结果
    print("包含NaN值时结果:")
    print("Pandas结果:")
    print(pandas_result)
    print("\nRust结果:")
    print(rust_result)
    
    # 过滤只保留两者都有的列
    common_cols = list(set(pandas_result.index) & set(rust_result.index))
    pandas_filtered = pandas_result[common_cols]
    rust_filtered = rust_result[common_cols]
    
    # 检查结果是否接近
    is_close = np.allclose(pandas_filtered.values, rust_filtered.values, rtol=1e-5, equal_nan=True)
    print(f"\n结果一致性检查: {'通过' if is_close else '失败'}"
          f" (比较了共同的{len(common_cols)}列)")
    
    return is_close

def test_large_dataframe_performance():
    """测试大型数据集上的性能"""
    print("\n=== 测试大型数据集上的性能 (5000x5000) ===")
    
    # 创建大型测试数据集 (5000x5000)
    size = 5000
    print(f"创建大小为 {size}x{size} 的随机数据框...")
    
    # 生成随机数据并确保有少量NaN值
    data1 = np.random.rand(size, size)
    data2 = np.random.rand(size, size)
    
    # 在约0.1%的位置插入NaN值
    nan_indices = np.random.choice(size*size, size=size*size//1000, replace=False)
    data1.flat[nan_indices[:len(nan_indices)//2]] = np.nan
    data2.flat[nan_indices[len(nan_indices)//2:]] = np.nan
    
    # 创建DataFrame
    df1 = pd.DataFrame(data1)
    df2 = pd.DataFrame(data2)
    
    # 测量Pandas的性能
    print("运行pandas.DataFrame.corrwith...")
    start_time = time.time()
    pandas_result = df1.corrwith(df2)
    pandas_time = time.time() - start_time
    print(f"Pandas耗时: {pandas_time:.4f} 秒")
    
    # 测量rust_corrwith的性能
    print("运行rust_corrwith...")
    start_time = time.time()
    rust_result = rust_corrwith(df1, df2)
    rust_time = time.time() - start_time
    print(f"Rust耗时: {rust_time:.4f} 秒")
    
    # 计算速度提升比例
    speedup = pandas_time / rust_time
    print(f"性能提升: {speedup:.2f}x (Rust比Pandas快{speedup:.2f}倍)")
    
    # 检查结果一致性
    common_cols = list(set(pandas_result.index) & set(rust_result.index))
    pandas_filtered = pandas_result[common_cols]
    rust_filtered = rust_result[common_cols]
    
    is_close = np.allclose(pandas_filtered.values, rust_filtered.values, rtol=1e-5, equal_nan=True)
    print(f"大型数据集结果一致性检查: {'通过' if is_close else '失败'}"
          f" (比较了共同的{len(common_cols)}列)")
    
    # 随机抽样检查部分结果
    sample_indices = np.random.choice(common_cols, size=min(5, len(common_cols)), replace=False)
    print("\n随机抽样结果比较:")
    for idx in sample_indices:
        print(f"索引 {idx}: Pandas={pandas_filtered[idx]:.6f}, Rust={rust_filtered[idx]:.6f}, "
              f"差异={abs(pandas_filtered[idx] - rust_filtered[idx]):.10f}")
    
    return pandas_time, rust_time, is_close

def main():
    """运行所有测试"""
    print("开始测试dataframe_corrwith优化...")
    
    # 测试小规模数据集上的正确性
    small_test_passed = test_small_dataframe()
    
    # 测试NaN值处理
    nan_test_passed = test_nan_handling()
    
    # 测试大规模数据集的性能
    pandas_time, rust_time, large_test_passed = test_large_dataframe_performance()
    
    # 总结报告
    print("\n=== 测试总结 ===")
    print(f"小型数据集测试: {'通过' if small_test_passed else '失败'}")
    print(f"NaN处理测试: {'通过' if nan_test_passed else '失败'}")
    print(f"大型数据集测试: {'通过' if large_test_passed else '失败'}")
    print(f"性能提升: {pandas_time/rust_time:.2f}x")
    
    all_passed = small_test_passed and nan_test_passed and large_test_passed
    print(f"\n总体结果: {'全部通过' if all_passed else '存在失败项'}")

if __name__ == "__main__":
    main()
