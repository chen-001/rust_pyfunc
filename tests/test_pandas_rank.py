"""
测试pandas_rank模块的DataFrame封装函数
"""

import numpy as np
import pandas as pd
import time
from rust_pyfunc.pandas_rank import rank_axis1_df, rank_axis0_df, fast_rank


def test_dataframe_wrapper():
    """测试DataFrame封装函数"""
    print("=== 测试DataFrame封装函数 ===")
    
    # 创建测试DataFrame
    df = pd.DataFrame({
        'A': [3.0, 2.0, 1.0],
        'B': [1.0, 4.0, 2.0], 
        'C': [4.0, 1.0, 3.0],
        'D': [2.0, 3.0, 4.0]
    }, index=['row1', 'row2', 'row3'])
    
    print("原始DataFrame:")
    print(df)
    
    # 使用封装函数
    rust_result = rank_axis1_df(df)
    print("\nrust_pyfunc结果 (axis=1):")
    print(rust_result)
    
    # 使用pandas原生函数对比
    pandas_result = df.rank(axis=1, method='average', ascending=True, na_option='keep')
    print("\npandas原生结果 (axis=1):")
    print(pandas_result)
    
    # 检查一致性
    is_equal = np.allclose(rust_result.values, pandas_result.values, equal_nan=True)
    print(f"\n结果一致性: {is_equal}")
    print(f"索引一致性: {rust_result.index.equals(pandas_result.index)}")
    print(f"列名一致性: {rust_result.columns.equals(pandas_result.columns)}")
    
    assert is_equal, "DataFrame封装函数测试失败：数值不一致"
    assert rust_result.index.equals(pandas_result.index), "索引不一致" 
    assert rust_result.columns.equals(pandas_result.columns), "列名不一致"
    
    print("✓ DataFrame封装函数测试通过")


def test_axis0_wrapper():
    """测试axis=0的DataFrame封装函数"""
    print("\n=== 测试axis=0 DataFrame封装函数 ===")
    
    df = pd.DataFrame({
        'A': [3, 1, 2],
        'B': [1, 3, 2],
        'C': [2, 2, 1]
    })
    
    print("原始DataFrame:")
    print(df)
    
    # 使用封装函数 
    rust_result = rank_axis0_df(df)
    print("\nrust_pyfunc结果 (axis=0):")
    print(rust_result)
    
    # 使用pandas原生函数对比
    pandas_result = df.rank(axis=0, method='average', ascending=True, na_option='keep')
    print("\npandas原生结果 (axis=0):")
    print(pandas_result)
    
    # 检查一致性
    is_equal = np.allclose(rust_result.values, pandas_result.values, equal_nan=True)
    print(f"\n结果一致性: {is_equal}")
    
    assert is_equal, "axis=0 DataFrame封装函数测试失败"
    print("✓ axis=0 DataFrame封装函数测试通过")


def test_nan_dataframe():
    """测试包含NaN的DataFrame"""
    print("\n=== 测试包含NaN的DataFrame ===")
    
    df = pd.DataFrame({
        'A': [3.0, np.nan, 1.0],
        'B': [1.0, 4.0, np.nan], 
        'C': [np.nan, 1.0, 3.0],
        'D': [2.0, 3.0, 4.0]
    })
    
    print("原始DataFrame（包含NaN）:")
    print(df)
    
    # 测试不同的na_option
    for na_option in ['keep', 'top', 'bottom']:
        print(f"\n测试na_option='{na_option}':")
        
        rust_result = rank_axis1_df(df, na_option=na_option)
        pandas_result = df.rank(axis=1, method='average', ascending=True, na_option=na_option)
        
        print(f"rust_pyfunc结果:")
        print(rust_result)
        print(f"pandas原生结果:")
        print(pandas_result)
        
        is_equal = np.allclose(rust_result.values, pandas_result.values, equal_nan=True)
        print(f"结果一致性: {is_equal}")
        
        assert is_equal, f"NaN处理（{na_option}）测试失败"
    
    print("✓ NaN DataFrame测试通过")


def test_performance_comparison():
    """测试性能对比"""
    print("\n=== DataFrame封装函数性能测试 ===")
    
    # 创建中等规模的测试数据
    rows, cols = 1000, 2000
    print(f"创建测试DataFrame: {rows} x {cols}")
    
    np.random.seed(42)
    df = pd.DataFrame(
        np.random.randn(rows, cols),
        index=[f'row_{i}' for i in range(rows)],
        columns=[f'col_{i}' for i in range(cols)]
    )
    
    # pandas原生性能
    print("\n测试pandas原生性能...")
    start_time = time.time()
    pandas_result = df.rank(axis=1, method='average', ascending=True, na_option='keep')
    pandas_time = time.time() - start_time
    print(f"pandas耗时: {pandas_time:.4f}秒")
    
    # rust_pyfunc封装函数性能
    print("\n测试rust_pyfunc封装函数性能...")
    start_time = time.time()
    rust_result = rank_axis1_df(df, method="average", ascending=True, na_option="keep")
    rust_time = time.time() - start_time
    print(f"rust_pyfunc耗时: {rust_time:.4f}秒")
    
    # 性能提升
    speedup = pandas_time / rust_time
    print(f"\n性能提升: {speedup:.2f}倍")
    
    # 验证结果一致性（抽样检查）
    sample_size = min(100, rows)
    sample_indices = np.random.choice(rows, sample_size, replace=False)
    
    pandas_sample = pandas_result.iloc[sample_indices].values
    rust_sample = rust_result.iloc[sample_indices].values
    
    is_equal = np.allclose(pandas_sample, rust_sample, equal_nan=True)
    print(f"抽样结果一致性: {is_equal}")
    
    assert is_equal, "性能测试结果不一致"
    print("✓ DataFrame封装函数性能测试通过")
    
    return pandas_time, rust_time, speedup


def test_alias_functions():
    """测试别名函数"""
    print("\n=== 测试别名函数 ===")
    
    df = pd.DataFrame({
        'A': [3, 1, 2],
        'B': [1, 3, 2]
    })
    
    # 测试fast_rank别名
    result1 = fast_rank(df)
    result2 = rank_axis1_df(df)
    
    is_equal = np.allclose(result1.values, result2.values, equal_nan=True)
    print(f"fast_rank别名测试: {is_equal}")
    
    assert is_equal, "别名函数测试失败"
    print("✓ 别名函数测试通过")


def test_edge_cases():
    """测试边界情况"""
    print("\n=== 测试边界情况 ===")
    
    # 空DataFrame
    empty_df = pd.DataFrame()
    result = rank_axis1_df(empty_df)
    assert result.empty, "空DataFrame测试失败"
    print("✓ 空DataFrame测试通过")
    
    # 单行DataFrame
    single_row_df = pd.DataFrame([[1, 2, 3]], columns=['A', 'B', 'C'])
    result = rank_axis1_df(single_row_df)
    expected = single_row_df.rank(axis=1)
    is_equal = np.allclose(result.values, expected.values, equal_nan=True)
    assert is_equal, "单行DataFrame测试失败"
    print("✓ 单行DataFrame测试通过")
    
    # 单列DataFrame
    single_col_df = pd.DataFrame({'A': [1, 2, 3]})
    result = rank_axis1_df(single_col_df)
    expected = single_col_df.rank(axis=1)
    is_equal = np.allclose(result.values, expected.values, equal_nan=True)
    assert is_equal, "单列DataFrame测试失败"
    print("✓ 单列DataFrame测试通过")
    
    print("✓ 边界情况测试通过")


def main():
    """运行所有测试"""
    print("开始测试pandas_rank模块\n")
    
    try:
        # DataFrame封装函数测试
        test_dataframe_wrapper()
        
        # axis=0测试
        test_axis0_wrapper()
        
        # NaN处理测试
        test_nan_dataframe()
        
        # 性能测试
        pandas_time, rust_time, speedup = test_performance_comparison()
        
        # 别名函数测试
        test_alias_functions()
        
        # 边界情况测试
        test_edge_cases()
        
        print("\n" + "="*60)
        print("✓ 所有测试通过！")
        print("✓ DataFrame封装函数完全兼容pandas.DataFrame.rank()")
        print(f"✓ 性能提升: {speedup:.2f}倍")
        print(f"✓ pandas耗时: {pandas_time:.4f}秒")
        print(f"✓ rust_pyfunc耗时: {rust_time:.4f}秒")
        print("✓ 现在可以直接传入DataFrame，无需手动转换！")
        print("="*60)
        
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    if not success:
        exit(1)