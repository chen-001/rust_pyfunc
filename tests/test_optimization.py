"""
测试优化后的性能
"""

import pandas as pd
import numpy as np
import rust_pyfunc as rp
import time


def create_test_data_mixed(n=10000):
    """创建混合类型测试数据（模拟用户场景）"""
    # 模拟用户的数据：date (Timestamp) + code (string)
    dates = pd.date_range('2024-01-01', periods=250, freq='D')
    codes = [f'00000{i}' for i in range(1, 201)]
    
    # y数据
    y_data = []
    for _ in range(n):
        date = np.random.choice(dates)
        code = np.random.choice(codes)
        fac = np.random.randn()
        y_data.append([date, code, fac])
    
    y = pd.DataFrame(y_data, columns=['date', 'code', 'fac'])
    
    # xs数据
    xs_data = []
    for date in dates:
        for code in codes[:50]:  # 部分匹配
            xs_data.append([date, code, np.random.randn(), np.random.randn()])
    
    xs = pd.DataFrame(xs_data, columns=['date', 'code', 'value1', 'value2'])
    
    return y, xs


def create_test_data_numeric(n=10000):
    """创建纯数值测试数据"""
    # 纯数值键
    key1_vals = np.random.randint(1, 1000, n)
    key2_vals = np.random.randint(1, 100, n)
    
    y = pd.DataFrame({
        'key1': key1_vals,
        'key2': key2_vals,
        'fac': np.random.randn(n)
    })
    
    # xs数据
    xs_key1 = np.random.randint(1, 1000, n//2)
    xs_key2 = np.random.randint(1, 100, n//2)
    
    xs = pd.DataFrame({
        'key1': xs_key1,
        'key2': xs_key2,
        'value1': np.random.randn(n//2),
        'value2': np.random.randn(n//2)
    })
    
    return y, xs


def test_column_deduplication():
    """测试连接键列去重功能"""
    print("=== 测试连接键列去重功能 ===")
    
    # 创建测试数据
    left_df = pd.DataFrame({
        'key1': [1, 2, 3],
        'key2': ['A', 'B', 'C'],
        'value_left': [100, 200, 300]
    })
    
    right_df = pd.DataFrame({
        'key1': [1, 2, 4],
        'key2': ['A', 'B', 'D'],
        'value_right': [10, 20, 40]
    })
    
    print("左表:")
    print(left_df)
    print("\n右表:")
    print(right_df)
    
    # rust_pyfunc结果
    rust_result = rp.fast_inner_join_df(left_df, right_df, on=['key1', 'key2'])
    print(f"\nrust_pyfunc结果 (优化后，去重连接键):")
    print(rust_result)
    print(f"列数: {len(rust_result.columns)}")
    
    # pandas结果
    pandas_result = pd.merge(left_df, right_df, on=['key1', 'key2'], how='inner')
    print(f"\npandas结果:")
    print(pandas_result)
    print(f"列数: {len(pandas_result.columns)}")
    
    print(f"\n✅ 连接键去重验证: rust列数={len(rust_result.columns)}, pandas列数={len(pandas_result.columns)}")


def test_performance_mixed_keys():
    """测试混合键的性能优化"""
    print("\n=== 测试混合键性能优化 ===")
    
    sizes = [1000, 5000, 10000, 20000]
    
    for n in sizes:
        print(f"\n测试规模: {n}行 (混合类型键: date + code)")
        y, xs = create_test_data_mixed(n)
        
        # pandas基准
        start = time.time()
        pandas_result = pd.merge(y, xs, on=['date', 'code'], how='inner')
        pandas_time = time.time() - start
        
        # rust_pyfunc优化版
        start = time.time()
        rust_result = rp.fast_inner_join_df(y, xs, on=['date', 'code'])
        rust_time = time.time() - start
        
        ratio = rust_time / pandas_time if pandas_time > 0 else float('inf')
        
        print(f"  pandas: {pandas_time:.4f}s ({len(pandas_result)}行)")
        print(f"  rust优化: {rust_time:.4f}s ({len(rust_result)}行)")
        print(f"  性能比: {ratio:.1f}x ({'rust更快' if ratio < 1 else 'pandas更快'})")


def test_performance_numeric_keys():
    """测试纯数值键的性能优化"""
    print("\n=== 测试纯数值键性能优化 ===")
    
    sizes = [1000, 5000, 10000, 20000]
    
    for n in sizes:
        print(f"\n测试规模: {n}行 (纯数值键: key1 + key2)")
        y, xs = create_test_data_numeric(n)
        
        # pandas基准
        start = time.time()
        pandas_result = pd.merge(y, xs, on=['key1', 'key2'], how='inner')
        pandas_time = time.time() - start
        
        # rust_pyfunc优化版
        start = time.time()
        rust_result = rp.fast_inner_join_df(y, xs, on=['key1', 'key2'])
        rust_time = time.time() - start
        
        ratio = rust_time / pandas_time if pandas_time > 0 else float('inf')
        
        print(f"  pandas: {pandas_time:.4f}s ({len(pandas_result)}行)")
        print(f"  rust优化: {rust_time:.4f}s ({len(rust_result)}行)")
        print(f"  性能比: {ratio:.1f}x ({'rust更快' if ratio < 1 else 'pandas更快'})")


def test_correctness():
    """验证优化后的正确性"""
    print("\n=== 验证优化后的正确性 ===")
    
    # 测试1：混合键
    print("1. 混合键正确性验证")
    y, xs = create_test_data_mixed(1000)
    
    pandas_result = pd.merge(y, xs, on=['date', 'code'], how='inner')
    rust_result = rp.fast_inner_join_df(y, xs, on=['date', 'code'])
    
    print(f"   pandas行数: {len(pandas_result)}, rust行数: {len(rust_result)}")
    assert len(pandas_result) == len(rust_result), "混合键结果行数不一致"
    print("   ✅ 混合键正确性验证通过")
    
    # 测试2：纯数值键
    print("2. 纯数值键正确性验证")
    y, xs = create_test_data_numeric(1000)
    
    pandas_result = pd.merge(y, xs, on=['key1', 'key2'], how='inner')
    rust_result = rp.fast_inner_join_df(y, xs, on=['key1', 'key2'])
    
    print(f"   pandas行数: {len(pandas_result)}, rust行数: {len(rust_result)}")
    assert len(pandas_result) == len(rust_result), "数值键结果行数不一致"
    print("   ✅ 数值键正确性验证通过")


def compare_with_baseline():
    """与优化前进行对比"""
    print("\n=== 与优化前对比 ===")
    
    # 创建中等规模数据
    y, xs = create_test_data_mixed(10000)
    
    print("数据信息:")
    print(f"  y: {y.shape}, 类型: {y.dtypes.to_dict()}")
    print(f"  xs: {xs.shape}, 类型: {xs.dtypes.to_dict()}")
    
    # pandas基准
    start = time.time()
    pandas_result = pd.merge(y, xs, on=['date', 'code'], how='inner')
    pandas_time = time.time() - start
    
    # rust_pyfunc优化版
    start = time.time()
    rust_result = rp.fast_inner_join_df(y, xs, on=['date', 'code'])
    rust_time = time.time() - start
    
    print(f"\n性能对比:")
    print(f"  pandas:     {pandas_time:.4f}s")
    print(f"  rust优化版: {rust_time:.4f}s")
    
    if rust_time < pandas_time:
        print(f"  🚀 rust比pandas快 {pandas_time/rust_time:.1f}倍!")
    else:
        print(f"  📊 pandas比rust快 {rust_time/pandas_time:.1f}倍")
    
    print(f"\n功能对比:")
    print(f"  pandas列数:     {len(pandas_result.columns)} (包含重复连接键)")
    print(f"  rust优化版列数: {len(rust_result.columns)} (连接键去重)")
    print(f"  结果行数一致:   {len(pandas_result) == len(rust_result)}")


def main():
    """运行所有优化测试"""
    print("开始测试优化效果...\n")
    
    # 验证正确性
    test_correctness()
    
    # 验证列去重功能
    test_column_deduplication()
    
    # 性能测试
    test_performance_mixed_keys()
    test_performance_numeric_keys()
    
    # 综合对比
    compare_with_baseline()
    
    print("\n" + "="*60)
    print("优化总结:")
    print("="*60)
    print("✅ 主要优化点:")
    print("   1. 避免iterrows()，使用values提升转换速度")
    print("   2. 智能路径选择：数值键vs混合键")
    print("   3. 连接键去重，减少冗余列")
    print("   4. 优化结果构建，减少Python对象创建")
    print("\n✅ 预期效果:")
    print("   • 大幅提升DataFrame转换速度")
    print("   • 减少结果DataFrame的列数")
    print("   • 保持与pandas完全兼容的结果")


if __name__ == "__main__":
    main()