"""
测试rank_axis1函数的正确性和性能
验证与pandas.DataFrame.rank(axis=1)的一致性，并测试性能提升
"""

import numpy as np
import pandas as pd
import time
import rust_pyfunc

def test_basic_functionality():
    """测试基本功能"""
    print("=== 测试基本功能 ===")
    
    # 创建测试数据
    data = np.array([
        [3.0, 1.0, 4.0, 2.0],
        [2.0, 4.0, 1.0, 3.0],
        [1.0, 2.0, 3.0, 4.0]
    ], dtype=np.float64)
    
    print("测试数据:")
    print(data)
    
    # pandas实现
    df = pd.DataFrame(data)
    pandas_result = df.rank(axis=1, method='average', ascending=True, na_option='keep')
    
    # Rust实现
    rust_result = rust_pyfunc.rank_axis1(data, method="average", ascending=True, na_option="keep")
    
    print("\npandas结果:")
    print(pandas_result.values)
    
    print("\nRust结果:")
    print(rust_result)
    
    # 检查一致性
    is_equal = np.allclose(pandas_result.values, rust_result, equal_nan=True)
    print(f"\n结果一致性: {is_equal}")
    
    assert is_equal, "基本功能测试失败：结果不一致"
    print("✓ 基本功能测试通过")

def test_nan_handling():
    """测试NaN值处理"""
    print("\n=== 测试NaN值处理 ===")
    
    # 创建包含NaN的测试数据
    data = np.array([
        [3.0, 1.0, np.nan, 2.0],
        [np.nan, 4.0, 1.0, 3.0],
        [1.0, np.nan, 3.0, np.nan]
    ], dtype=np.float64)
    
    print("测试数据（包含NaN）:")
    print(data)
    
    # 测试keep选项
    print("\n测试na_option='keep':")
    df = pd.DataFrame(data)
    pandas_result = df.rank(axis=1, method='average', ascending=True, na_option='keep')
    rust_result = rust_pyfunc.rank_axis1(data, method="average", ascending=True, na_option="keep")
    
    print("pandas结果:")
    print(pandas_result.values)
    print("Rust结果:")
    print(rust_result)
    
    is_equal = np.allclose(pandas_result.values, rust_result, equal_nan=True)
    print(f"结果一致性: {is_equal}")
    assert is_equal, "NaN处理（keep）测试失败"
    
    # 测试top选项
    print("\n测试na_option='top':")
    pandas_result_top = df.rank(axis=1, method='average', ascending=True, na_option='top')
    rust_result_top = rust_pyfunc.rank_axis1(data, method="average", ascending=True, na_option="top")
    
    print("pandas结果:")
    print(pandas_result_top.values)
    print("Rust结果:")
    print(rust_result_top)
    
    is_equal_top = np.allclose(pandas_result_top.values, rust_result_top, equal_nan=True)
    print(f"结果一致性: {is_equal_top}")
    assert is_equal_top, "NaN处理（top）测试失败"
    
    # 测试bottom选项
    print("\n测试na_option='bottom':")
    pandas_result_bottom = df.rank(axis=1, method='average', ascending=True, na_option='bottom')
    rust_result_bottom = rust_pyfunc.rank_axis1(data, method="average", ascending=True, na_option="bottom")
    
    print("pandas结果:")
    print(pandas_result_bottom.values)
    print("Rust结果:")
    print(rust_result_bottom)
    
    is_equal_bottom = np.allclose(pandas_result_bottom.values, rust_result_bottom, equal_nan=True)
    print(f"结果一致性: {is_equal_bottom}")
    assert is_equal_bottom, "NaN处理（bottom）测试失败"
    
    print("✓ NaN值处理测试通过")

def test_different_methods():
    """测试不同的排名方法"""
    print("\n=== 测试不同排名方法 ===")
    
    # 创建包含并列值的测试数据
    data = np.array([
        [1.0, 2.0, 2.0, 3.0],
        [3.0, 1.0, 1.0, 2.0],
        [2.0, 2.0, 2.0, 1.0]
    ], dtype=np.float64)
    
    print("测试数据（包含并列值）:")
    print(data)
    
    methods = ['average', 'min', 'max', 'first', 'dense']
    
    for method in methods:
        print(f"\n测试method='{method}':")
        
        df = pd.DataFrame(data)
        pandas_result = df.rank(axis=1, method=method, ascending=True, na_option='keep')
        rust_result = rust_pyfunc.rank_axis1(data, method=method, ascending=True, na_option="keep")
        
        print(f"pandas结果:")
        print(pandas_result.values)
        print(f"Rust结果:")
        print(rust_result)
        
        is_equal = np.allclose(pandas_result.values, rust_result, equal_nan=True)
        print(f"结果一致性: {is_equal}")
        assert is_equal, f"排名方法（{method}）测试失败"
    
    print("✓ 不同排名方法测试通过")

def test_ascending_parameter():
    """测试升序/降序参数"""
    print("\n=== 测试升序/降序参数 ===")
    
    data = np.array([
        [3.0, 1.0, 4.0, 2.0],
        [2.0, 4.0, 1.0, 3.0]
    ], dtype=np.float64)
    
    print("测试数据:")
    print(data)
    
    # 测试升序
    print("\n测试ascending=True:")
    df = pd.DataFrame(data)
    pandas_result_asc = df.rank(axis=1, method='average', ascending=True, na_option='keep')
    rust_result_asc = rust_pyfunc.rank_axis1(data, method="average", ascending=True, na_option="keep")
    
    print("pandas结果:")
    print(pandas_result_asc.values)
    print("Rust结果:")
    print(rust_result_asc)
    
    is_equal_asc = np.allclose(pandas_result_asc.values, rust_result_asc, equal_nan=True)
    print(f"结果一致性: {is_equal_asc}")
    assert is_equal_asc, "升序测试失败"
    
    # 测试降序
    print("\n测试ascending=False:")
    pandas_result_desc = df.rank(axis=1, method='average', ascending=False, na_option='keep')
    rust_result_desc = rust_pyfunc.rank_axis1(data, method="average", ascending=False, na_option="keep")
    
    print("pandas结果:")
    print(pandas_result_desc.values)
    print("Rust结果:")
    print(rust_result_desc)
    
    is_equal_desc = np.allclose(pandas_result_desc.values, rust_result_desc, equal_nan=True)
    print(f"结果一致性: {is_equal_desc}")
    assert is_equal_desc, "降序测试失败"
    
    print("✓ 升序/降序参数测试通过")

def test_performance():
    """测试性能对比"""
    print("\n=== 性能测试 ===")
    
    # 创建大规模测试数据（用户提到的2500*5500）
    rows, cols = 2500, 5500
    print(f"创建测试数据: {rows} x {cols}")
    
    np.random.seed(42)
    data = np.random.randn(rows, cols).astype(np.float64)
    
    # 添加一些NaN值
    nan_mask = np.random.random((rows, cols)) < 0.01  # 1%的NaN
    data[nan_mask] = np.nan
    
    print("测试数据创建完成")
    
    # pandas性能测试
    print("\n测试pandas性能...")
    df = pd.DataFrame(data)
    
    start_time = time.time()
    pandas_result = df.rank(axis=1, method='average', ascending=True, na_option='keep')
    pandas_time = time.time() - start_time
    
    print(f"pandas耗时: {pandas_time:.4f}秒")
    
    # Rust性能测试
    print("\n测试Rust性能...")
    
    start_time = time.time()
    rust_result = rust_pyfunc.rank_axis1(data, method="average", ascending=True, na_option="keep")
    rust_time = time.time() - start_time
    
    print(f"Rust耗时: {rust_time:.4f}秒")
    
    # 性能提升计算
    speedup = pandas_time / rust_time
    print(f"\n性能提升: {speedup:.2f}倍")
    
    # 验证结果一致性（抽样检查，因为数据量很大）
    print("\n验证结果一致性（抽样检查）...")
    sample_rows = min(100, rows)
    sample_indices = np.random.choice(rows, sample_rows, replace=False)
    
    pandas_sample = pandas_result.iloc[sample_indices].values
    rust_sample = rust_result[sample_indices]
    
    is_equal = np.allclose(pandas_sample, rust_sample, equal_nan=True)
    print(f"抽样结果一致性: {is_equal}")
    
    if is_equal:
        print("✓ 性能测试通过")
        print(f"✓ Rust实现比pandas快 {speedup:.2f} 倍")
    else:
        print("✗ 性能测试失败：结果不一致")
        assert False, "性能测试失败：结果不一致"
    
    return pandas_time, rust_time, speedup

def main():
    """运行所有测试"""
    print("开始测试rank_axis1函数\n")
    
    try:
        # 基本功能测试
        test_basic_functionality()
        
        # NaN值处理测试
        test_nan_handling()
        
        # 不同排名方法测试
        test_different_methods()
        
        # 升序/降序参数测试
        test_ascending_parameter()
        
        # 性能测试
        pandas_time, rust_time, speedup = test_performance()
        
        print("\n" + "="*50)
        print("✓ 所有测试通过！")
        print(f"✓ 功能完全兼容pandas.DataFrame.rank(axis=1)")
        print(f"✓ 性能提升: {speedup:.2f}倍")
        print(f"✓ pandas耗时: {pandas_time:.4f}秒")
        print(f"✓ Rust耗时: {rust_time:.4f}秒")
        print("="*50)
        
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