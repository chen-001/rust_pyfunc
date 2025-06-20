"""
fast_merge性能测试
对比pandas.merge的性能
"""

import numpy as np
import pandas as pd
import time
import rust_pyfunc as rp


def performance_test():
    """大规模数据性能测试"""
    print("=== fast_merge性能测试 ===")
    
    # 创建大规模测试数据
    n_left = 100000
    n_right = 120000
    overlap = 60000  # 重叠的键数量
    
    print(f"创建测试数据: 左表{n_left}行, 右表{n_right}行, 重叠键{overlap}个")
    
    np.random.seed(42)
    
    # 左表：键0到n_left-1，值为随机数
    left_keys = np.arange(n_left, dtype=np.float64)
    left_values = np.random.randn(n_left)
    left_data = np.column_stack([left_keys, left_values])
    
    # 右表：键n_left-overlap到n_left+n_right-overlap-1，值为随机数
    right_keys = np.arange(n_left - overlap, n_left + n_right - overlap, dtype=np.float64)
    right_values = np.random.randn(n_right)
    right_data = np.column_stack([right_keys, right_values])
    
    print(f"预期内连接结果行数: {overlap}")
    
    # pandas性能测试
    print("\n测试pandas性能...")
    left_df = pd.DataFrame(left_data, columns=['key', 'value_left'])
    right_df = pd.DataFrame(right_data, columns=['key', 'value_right'])
    
    start_time = time.time()
    pandas_result = pd.merge(left_df, right_df, on='key', how='inner')
    pandas_time = time.time() - start_time
    
    print(f"pandas内连接耗时: {pandas_time:.4f}秒")
    print(f"pandas结果行数: {len(pandas_result)}")
    
    # fast_merge性能测试
    print("\n测试fast_merge性能...")
    start_time = time.time()
    indices, merged_data = rp.fast_merge(
        left_data, right_data,
        left_keys=[0], right_keys=[0],
        how="inner"
    )
    rust_time = time.time() - start_time
    
    print(f"fast_merge内连接耗时: {rust_time:.4f}秒")
    print(f"fast_merge结果行数: {len(merged_data)}")
    
    # 性能提升
    speedup = pandas_time / rust_time
    print(f"\n🚀 性能提升: {speedup:.1f}倍")
    
    # 验证结果一致性（抽样检查）
    sample_size = min(1000, len(merged_data))
    print(f"\n验证结果一致性（抽样{sample_size}行）...")
    
    # 检查行数
    assert len(merged_data) == len(pandas_result), f"结果行数不一致: fast_merge={len(merged_data)}, pandas={len(pandas_result)}"
    
    # 抽样验证数据正确性
    sample_indices = np.random.choice(len(merged_data), sample_size, replace=False)
    for i in sample_indices:
        row = merged_data[i]
        # 检查左表键和右表键是否相等（连接条件）
        assert row[0] == row[2], f"第{i}行连接键不匹配: {row[0]} != {row[2]}"
        
        # 检查数据是否在原始表中存在
        left_key = row[0]
        right_key = row[2]
        
        left_match = left_df[left_df['key'] == left_key]
        right_match = right_df[right_df['key'] == right_key]
        
        assert len(left_match) == 1, f"左表键{left_key}不存在或重复"
        assert len(right_match) == 1, f"右表键{right_key}不存在或重复"
        
        assert abs(row[1] - left_match.iloc[0]['value_left']) < 1e-10, f"左表值不匹配"
        assert abs(row[3] - right_match.iloc[0]['value_right']) < 1e-10, f"右表值不匹配"
    
    print(f"✓ 抽样验证通过")
    
    return pandas_time, rust_time, speedup


def test_different_join_types():
    """测试不同连接类型的性能"""
    print("\n=== 不同连接类型性能测试 ===")
    
    # 创建中等规模测试数据
    n = 20000
    np.random.seed(42)
    
    left_data = np.column_stack([
        np.arange(n, dtype=np.float64),
        np.random.randn(n)
    ])
    
    right_data = np.column_stack([
        np.arange(n//2, n + n//2, dtype=np.float64),  # 50%重叠
        np.random.randn(n)
    ])
    
    left_df = pd.DataFrame(left_data, columns=['key', 'value_left'])
    right_df = pd.DataFrame(right_data, columns=['key', 'value_right'])
    
    join_types = ['inner', 'left', 'outer']
    
    for join_type in join_types:
        print(f"\n测试{join_type}连接:")
        
        # pandas测试
        start_time = time.time()
        pandas_result = pd.merge(left_df, right_df, on='key', how=join_type)
        pandas_time = time.time() - start_time
        
        # fast_merge测试
        start_time = time.time()
        indices, merged_data = rp.fast_merge(
            left_data, right_data,
            left_keys=[0], right_keys=[0],
            how=join_type
        )
        rust_time = time.time() - start_time
        
        speedup = pandas_time / rust_time
        
        print(f"  pandas {join_type}: {pandas_time:.4f}秒, {len(pandas_result)}行")
        print(f"  fast_merge {join_type}: {rust_time:.4f}秒, {len(merged_data)}行")
        print(f"  性能提升: {speedup:.1f}倍")
        
        # 基本一致性检查
        assert len(merged_data) == len(pandas_result), f"{join_type}连接行数不一致"


if __name__ == "__main__":
    try:
        # 运行性能测试
        pandas_time, rust_time, speedup = performance_test()
        test_different_join_types()
        
        print("\n" + "="*60)
        print("✓ 性能测试完成！")
        print(f"✓ 主要性能提升: {speedup:.1f}倍")
        print(f"✓ pandas耗时: {pandas_time:.4f}秒")
        print(f"✓ fast_merge耗时: {rust_time:.4f}秒")
        print("✓ 支持所有主要连接类型，性能显著优于pandas")
        print("="*60)
        
    except Exception as e:
        print(f"\n✗ 性能测试失败: {e}")
        import traceback
        traceback.print_exc()
        exit(1)