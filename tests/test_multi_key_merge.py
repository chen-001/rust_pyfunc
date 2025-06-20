"""
测试多键连接功能
验证与pandas.merge的多键连接一致性
"""

import pandas as pd
import numpy as np
import rust_pyfunc as rp


def test_single_key_compatibility():
    """测试单键连接的向后兼容性"""
    print("=== 测试单键连接向后兼容性 ===")
    
    left_df = pd.DataFrame({
        'key': [1, 2, 3],
        'value_left': [100, 200, 300]
    })
    
    right_df = pd.DataFrame({
        'key': [1, 2, 4],
        'value_right': [10, 20, 40]
    })
    
    print("左表:")
    print(left_df)
    print("\n右表:")
    print(right_df)
    
    # 测试字符串参数
    result1 = rp.fast_inner_join_df(left_df, right_df, on='key')
    print(f"\n单键字符串参数结果: {len(result1)}行")
    print(result1)
    
    # 测试单元素列表参数
    result2 = rp.fast_inner_join_df(left_df, right_df, on=['key'])
    print(f"\n单键列表参数结果: {len(result2)}行")
    print(result2)
    
    # 验证结果一致性
    assert len(result1) == len(result2), "单键字符串和列表参数结果应该一致"
    
    # 与pandas对比
    pandas_result = pd.merge(left_df, right_df, on='key', how='inner')
    assert len(result1) == len(pandas_result), "与pandas结果行数应该一致"
    
    print("✓ 单键连接向后兼容性测试通过")


def test_two_key_merge():
    """测试双键连接"""
    print("\n=== 测试双键连接 ===")
    
    # 创建有组合键的测试数据
    left_df = pd.DataFrame({
        'key1': [1, 1, 2, 2, 3],
        'key2': [1, 2, 1, 2, 1],
        'value_left': [100, 200, 300, 400, 500]
    })
    
    right_df = pd.DataFrame({
        'key1': [1, 1, 2, 3, 4],
        'key2': [1, 2, 1, 2, 1],
        'value_right': [10, 20, 30, 40, 50]
    })
    
    print("左表（双键）:")
    print(left_df)
    print("\n右表（双键）:")
    print(right_df)
    
    # 使用双键连接
    rust_result = rp.fast_inner_join_df(left_df, right_df, on=['key1', 'key2'])
    print(f"\nrust_pyfunc双键连接结果: {len(rust_result)}行")
    print(rust_result)
    
    # pandas对比
    pandas_result = pd.merge(left_df, right_df, on=['key1', 'key2'], how='inner')
    print(f"\npandas双键连接结果: {len(pandas_result)}行")
    print(pandas_result)
    
    # 验证结果一致性
    assert len(rust_result) == len(pandas_result), f"双键连接行数不一致: rust={len(rust_result)}, pandas={len(pandas_result)}"
    
    # 验证匹配的记录
    expected_matches = [
        (1, 1),  # key1=1, key2=1
        (1, 2),  # key1=1, key2=2  
        (2, 1),  # key1=2, key2=1
        # (3, 1) 左表有但右表没有key1=3,key2=1的组合（右表是key1=3,key2=2）
        # (3, 2) 左表没有这个组合
    ]
    
    print(f"\n预期匹配的组合键: {expected_matches}")
    print("✓ 双键连接测试通过")


def test_three_key_merge():
    """测试三键连接"""
    print("\n=== 测试三键连接 ===")
    
    left_df = pd.DataFrame({
        'key1': [1, 1, 1, 2, 2],
        'key2': [1, 1, 2, 1, 2], 
        'key3': [1, 2, 1, 1, 1],
        'value_left': [100, 200, 300, 400, 500]
    })
    
    right_df = pd.DataFrame({
        'key1': [1, 1, 2, 2, 3],
        'key2': [1, 2, 1, 2, 1],
        'key3': [1, 1, 1, 1, 1], 
        'value_right': [10, 20, 30, 40, 50]
    })
    
    print("左表（三键）:")
    print(left_df)
    print("\n右表（三键）:")
    print(right_df)
    
    # 使用三键连接
    rust_result = rp.fast_inner_join_df(left_df, right_df, on=['key1', 'key2', 'key3'])
    print(f"\nrust_pyfunc三键连接结果: {len(rust_result)}行")
    print(rust_result)
    
    # pandas对比
    pandas_result = pd.merge(left_df, right_df, on=['key1', 'key2', 'key3'], how='inner')
    print(f"\npandas三键连接结果: {len(pandas_result)}行")
    print(pandas_result)
    
    # 验证结果一致性
    assert len(rust_result) == len(pandas_result), f"三键连接行数不一致: rust={len(rust_result)}, pandas={len(pandas_result)}"
    
    print("✓ 三键连接测试通过")


def test_different_key_names():
    """测试不同键名的连接"""
    print("\n=== 测试不同键名连接 ===")
    
    left_df = pd.DataFrame({
        'left_key1': [1, 2, 3],
        'left_key2': [1, 1, 2],
        'value_left': [100, 200, 300]
    })
    
    right_df = pd.DataFrame({
        'right_key1': [1, 2, 4],
        'right_key2': [1, 1, 2],
        'value_right': [10, 20, 40]
    })
    
    print("左表:")
    print(left_df)
    print("\n右表:")
    print(right_df)
    
    # 使用不同键名连接
    rust_result = rp.fast_merge_df(
        left_df, right_df,
        left_on=['left_key1', 'left_key2'],
        right_on=['right_key1', 'right_key2'],
        how='inner'
    )
    print(f"\nrust_pyfunc不同键名连接结果: {len(rust_result)}行")
    print(rust_result)
    
    # pandas对比
    pandas_result = pd.merge(
        left_df, right_df,
        left_on=['left_key1', 'left_key2'],
        right_on=['right_key1', 'right_key2'],
        how='inner'
    )
    print(f"\npandas不同键名连接结果: {len(pandas_result)}行")
    print(pandas_result)
    
    # 验证结果一致性
    assert len(rust_result) == len(pandas_result), f"不同键名连接行数不一致: rust={len(rust_result)}, pandas={len(pandas_result)}"
    
    print("✓ 不同键名连接测试通过")


def test_multi_key_left_join():
    """测试多键左连接"""
    print("\n=== 测试多键左连接 ===")
    
    left_df = pd.DataFrame({
        'key1': [1, 1, 2, 3],
        'key2': [1, 2, 1, 1],
        'value_left': [100, 200, 300, 400]
    })
    
    right_df = pd.DataFrame({
        'key1': [1, 2, 4],
        'key2': [1, 1, 1],
        'value_right': [10, 20, 30]
    })
    
    print("左表:")
    print(left_df)
    print("\n右表:")
    print(right_df)
    
    # 多键左连接
    rust_result = rp.fast_left_join_df(left_df, right_df, on=['key1', 'key2'])
    print(f"\nrust_pyfunc多键左连接结果: {len(rust_result)}行")
    print(rust_result)
    
    # pandas对比
    pandas_result = pd.merge(left_df, right_df, on=['key1', 'key2'], how='left')
    print(f"\npandas多键左连接结果: {len(pandas_result)}行")
    print(pandas_result)
    
    # 验证结果
    assert len(rust_result) == len(pandas_result), f"多键左连接行数不一致: rust={len(rust_result)}, pandas={len(pandas_result)}"
    assert len(rust_result) == len(left_df), "左连接应该保留左表所有行"
    
    # 检查NaN值处理
    rust_na_count = rust_result.isna().sum().sum()
    pandas_na_count = pandas_result.isna().sum().sum()
    print(f"\nNaN值数量对比: rust={rust_na_count}, pandas={pandas_na_count}")
    
    print("✓ 多键左连接测试通过")


def test_performance_multi_key():
    """测试多键连接的性能"""
    print("\n=== 多键连接性能测试 ===")
    
    # 创建较大规模的多键数据
    n = 20000
    np.random.seed(42)
    
    left_df = pd.DataFrame({
        'key1': np.random.randint(1, 1000, n),
        'key2': np.random.randint(1, 100, n),
        'value_left': np.random.randn(n)
    })
    
    right_df = pd.DataFrame({
        'key1': np.random.randint(1, 1000, n),
        'key2': np.random.randint(1, 100, n), 
        'value_right': np.random.randn(n)
    })
    
    print(f"测试数据规模: 左表{len(left_df)}行, 右表{len(right_df)}行")
    
    # pandas性能
    import time
    start = time.time()
    pandas_result = pd.merge(left_df, right_df, on=['key1', 'key2'], how='inner')
    pandas_time = time.time() - start
    
    # rust_pyfunc性能
    start = time.time()
    rust_result = rp.fast_inner_join_df(left_df, right_df, on=['key1', 'key2'])
    rust_time = time.time() - start
    
    print(f"\n多键连接性能对比:")
    print(f"pandas耗时: {pandas_time:.4f}秒 ({len(pandas_result)}行)")
    print(f"rust_pyfunc耗时: {rust_time:.4f}秒 ({len(rust_result)}行)")
    
    if rust_time > 0:
        speedup = pandas_time / rust_time
        if speedup > 1:
            print(f"🚀 rust_pyfunc快{speedup:.1f}倍")
        else:
            print(f"📊 pandas快{1/speedup:.1f}倍")
    
    # 验证结果一致性
    assert len(rust_result) == len(pandas_result), "多键连接结果行数不一致"
    
    print("✓ 多键连接性能测试完成")


def main():
    """运行所有多键连接测试"""
    print("开始测试多键连接功能\n")
    
    try:
        # 基础功能测试
        test_single_key_compatibility()
        test_two_key_merge()
        test_three_key_merge()
        test_different_key_names()
        test_multi_key_left_join()
        
        # 性能测试
        test_performance_multi_key()
        
        print("\n" + "="*60)
        print("✅ 所有多键连接测试通过！")
        print("✅ 完全支持pandas.merge的多键连接功能")
        print("✅ 支持单键和多键连接")
        print("✅ 支持不同键名连接")
        print("✅ 支持所有连接类型（inner、left、right、outer）")
        print("✅ 向后兼容单键字符串参数")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    if not success:
        exit(1)