"""
测试DataFrame级别的字符串键合并功能
验证pandas_merge.py中的字符串键支持
"""

import pandas as pd
import numpy as np
import rust_pyfunc as rp
import time

def test_dataframe_string_key_merge():
    """测试DataFrame级别的字符串键合并"""
    print("=== 测试DataFrame字符串键合并 ===")
    
    # 创建包含字符串键的测试数据
    left_df = pd.DataFrame({
        'symbol': ['AAPL', 'GOOGL', 'MSFT', 'TSLA'],
        'sector': ['Technology', 'Technology', 'Technology', 'Auto'],
        'market_cap': [3000, 1800, 2800, 800]
    })
    
    right_df = pd.DataFrame({
        'symbol': ['AAPL', 'GOOGL', 'AMZN', 'TSLA'], 
        'price': [150.0, 135.0, 140.0, 250.0],
        'volume': [50000000, 25000000, 30000000, 40000000]
    })
    
    print("左表（股票基础信息）:")
    print(left_df)
    print("\n右表（价格数据）:")
    print(right_df)
    
    # 测试字符串键内连接
    print("\n测试字符串键内连接:")
    inner_result = rp.fast_inner_join_df(left_df, right_df, on='symbol')
    print(f"内连接结果: {len(inner_result)}行")
    print(inner_result)
    
    # 对比pandas结果
    pandas_inner = pd.merge(left_df, right_df, on='symbol', how='inner')
    print(f"\npandas内连接结果: {len(pandas_inner)}行")
    print(pandas_inner)
    
    # 验证结果一致性
    assert len(inner_result) == len(pandas_inner), "内连接结果行数不一致"
    print("✓ DataFrame字符串键内连接测试通过")
    
    # 测试左连接
    print("\n测试字符串键左连接:")
    left_result = rp.fast_left_join_df(left_df, right_df, on='symbol')
    print(f"左连接结果: {len(left_result)}行")
    print(left_result)
    
    pandas_left = pd.merge(left_df, right_df, on='symbol', how='left')
    print(f"\npandas左连接结果: {len(pandas_left)}行")
    print(pandas_left)
    
    assert len(left_result) == len(pandas_left), "左连接结果行数不一致"
    print("✓ DataFrame字符串键左连接测试通过")


def test_mixed_type_dataframe_merge():
    """测试混合类型键的DataFrame合并"""
    print("\n=== 测试混合类型键DataFrame合并 ===")
    
    # 创建混合类型键的数据
    left_df = pd.DataFrame({
        'market': ['SZ', 'SZ', 'SH', 'SH'],
        'stock_id': [1, 2, 1, 2],
        'stock_name': ['平安银行', '万科A', '浦发银行', '工商银行'],
        'sector': ['金融', '房地产', '金融', '金融']
    })
    
    right_df = pd.DataFrame({
        'market': ['SZ', 'SZ', 'SH', 'BJ'],
        'stock_id': [1, 2, 1, 1],
        'price': [10.5, 20.3, 12.8, 8.9],
        'volume': [1000000, 2000000, 1500000, 500000]
    })
    
    print("左表（混合类型键）:")
    print(left_df)
    print("\n右表（混合类型键）:")
    print(right_df)
    
    # 测试混合类型键合并
    print("\n测试混合类型键合并:")
    mixed_result = rp.fast_inner_join_df(left_df, right_df, on=['market', 'stock_id'])
    print(f"混合键合并结果: {len(mixed_result)}行")
    print(mixed_result)
    
    # 对比pandas结果
    pandas_mixed = pd.merge(left_df, right_df, on=['market', 'stock_id'], how='inner')
    print(f"\npandas混合键合并结果: {len(pandas_mixed)}行")
    print(pandas_mixed)
    
    assert len(mixed_result) == len(pandas_mixed), "混合键合并结果行数不一致"
    print("✓ DataFrame混合类型键合并测试通过")


def test_different_key_names_string():
    """测试不同键名的字符串合并"""
    print("\n=== 测试不同键名字符串合并 ===")
    
    left_df = pd.DataFrame({
        'left_symbol': ['AAPL', 'GOOGL', 'MSFT'],
        'company': ['Apple', 'Google', 'Microsoft'],
        'value': [100, 200, 300]
    })
    
    right_df = pd.DataFrame({
        'right_symbol': ['AAPL', 'GOOGL', 'AMZN'],
        'price': [150.0, 135.0, 140.0],
        'volume': [50000000, 25000000, 30000000]
    })
    
    print("左表:")
    print(left_df)
    print("\n右表:")
    print(right_df)
    
    # 测试不同键名合并
    print("\n测试不同键名合并:")
    diff_key_result = rp.fast_merge_df(
        left_df, right_df,
        left_on='left_symbol',
        right_on='right_symbol',
        how='inner'
    )
    print(f"不同键名合并结果: {len(diff_key_result)}行")
    print(diff_key_result)
    
    # 对比pandas结果
    pandas_diff_key = pd.merge(
        left_df, right_df,
        left_on='left_symbol',
        right_on='right_symbol',
        how='inner'
    )
    print(f"\npandas不同键名合并结果: {len(pandas_diff_key)}行")
    print(pandas_diff_key)
    
    assert len(diff_key_result) == len(pandas_diff_key), "不同键名合并结果行数不一致"
    print("✓ DataFrame不同键名字符串合并测试通过")


def test_all_join_types_string():
    """测试所有连接类型的字符串键合并"""
    print("\n=== 测试所有连接类型字符串键合并 ===")
    
    left_df = pd.DataFrame({
        'key': ['A', 'B', 'C'],
        'value_left': [100, 200, 300]
    })
    
    right_df = pd.DataFrame({
        'key': ['A', 'B', 'D'],
        'value_right': [10, 20, 40]
    })
    
    print("左表:")
    print(left_df)
    print("\n右表:")
    print(right_df)
    
    join_types = ['inner', 'left', 'right', 'outer']
    
    for join_type in join_types:
        print(f"\n--- {join_type.upper()} JOIN ---")
        
        if join_type == 'inner':
            result = rp.fast_inner_join_df(left_df, right_df, on='key')
        elif join_type == 'left':
            result = rp.fast_left_join_df(left_df, right_df, on='key')
        elif join_type == 'right':
            result = rp.fast_right_join_df(left_df, right_df, on='key')
        else:  # outer
            result = rp.fast_outer_join_df(left_df, right_df, on='key')
        
        print(f"rust结果行数: {len(result)}")
        print(result)
        
        # 对比pandas
        pandas_result = pd.merge(left_df, right_df, on='key', how=join_type)
        print(f"pandas结果行数: {len(pandas_result)}")
        print(pandas_result)
        
        assert len(result) == len(pandas_result), f"{join_type}连接结果行数不一致"
        print(f"✓ {join_type}连接测试通过")


def test_performance_string_dataframe():
    """测试字符串键DataFrame合并的性能"""
    print("\n=== 字符串键DataFrame合并性能测试 ===")
    
    # 创建较大规模的字符串键数据
    n = 20000
    symbols = [f"STOCK{i:04d}" for i in range(n)]
    
    left_df = pd.DataFrame({
        'symbol': symbols,
        'sector': [f"Sector{i%20}" for i in range(n)],
        'market_cap': np.random.randint(100, 10000, n)
    })
    
    # 创建50%重叠的右表
    right_symbols = symbols[n//2:] + [f"NEW{i:04d}" for i in range(n//2)]
    right_df = pd.DataFrame({
        'symbol': right_symbols,
        'price': np.random.uniform(10, 500, n),
        'volume': np.random.randint(100000, 10000000, n)
    })
    
    print(f"测试数据规模: 左表{len(left_df)}行, 右表{len(right_df)}行")
    
    # pandas性能
    start = time.time()
    pandas_result = pd.merge(left_df, right_df, on='symbol', how='inner')
    pandas_time = time.time() - start
    
    # rust_pyfunc性能
    start = time.time()
    rust_result = rp.fast_inner_join_df(left_df, right_df, on='symbol')
    rust_time = time.time() - start
    
    print(f"\n字符串键DataFrame合并性能对比:")
    print(f"pandas耗时: {pandas_time:.4f}秒 ({len(pandas_result)}行)")
    print(f"rust_pyfunc耗时: {rust_time:.4f}秒 ({len(rust_result)}行)")
    
    if rust_time > 0:
        speedup = pandas_time / rust_time
        if speedup > 1:
            print(f"🚀 rust_pyfunc快{speedup:.1f}倍")
        else:
            print(f"📊 pandas快{1/speedup:.1f}倍")
    
    # 验证结果一致性
    assert len(rust_result) == len(pandas_result), "字符串键合并结果行数不一致"
    print("✓ 字符串键DataFrame性能测试完成")


def main():
    """运行所有DataFrame字符串键测试"""
    print("开始测试DataFrame级别的字符串键合并功能\n")
    
    try:
        test_dataframe_string_key_merge()
        test_mixed_type_dataframe_merge()
        test_different_key_names_string()
        test_all_join_types_string()
        test_performance_string_dataframe()
        
        print("\n" + "="*70)
        print("✅ DataFrame字符串键合并功能测试完成！")
        print("✅ 支持DataFrame级别的字符串键合并")
        print("✅ 支持混合类型键合并（字符串+数值）")
        print("✅ 支持不同键名合并")
        print("✅ 支持所有连接类型")
        print("✅ 自动检测并处理字符串键")
        print("✅ 保持与pandas.merge的兼容性")
        print("="*70)
        
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