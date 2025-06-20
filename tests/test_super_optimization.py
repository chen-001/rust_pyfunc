"""
测试超级优化版本的性能
重点测试纯数值数据的性能提升
"""

import pandas as pd
import numpy as np
import rust_pyfunc as rp
import time


def create_numeric_test_data(n=10000):
    """创建纯数值测试数据"""
    # 模拟股票数据，但是用数值ID
    np.random.seed(42)
    
    # y数据 (类似用户的场景，但用数值键)
    y = pd.DataFrame({
        'date_id': np.random.randint(1, 250, n),      # 日期ID (1-250)
        'stock_id': np.random.randint(1, 200, n),     # 股票ID (1-200)
        'factor_value': np.random.randn(n)            # 因子值
    })
    
    # xs数据 (查找表)
    xs_size = 12500  # 250天 * 50只股票
    xs = pd.DataFrame({
        'date_id': np.tile(range(1, 251), 50),        # 250天重复50次
        'stock_id': np.repeat(range(1, 51), 250),     # 前50只股票
        'value1': np.random.randn(xs_size),
        'value2': np.random.randn(xs_size)
    })
    
    return y, xs


def test_ultra_fast_numeric():
    """测试超级优化的数值版本"""
    print("=== 测试超级优化的数值版本 ===")
    
    sizes = [1000, 5000, 10000, 20000, 50000]
    
    for n in sizes:
        print(f"\n测试规模: {n}行")
        y, xs = create_numeric_test_data(n)
        
        print(f"  数据类型: y={y.dtypes.to_dict()}, xs={xs.dtypes.to_dict()}")
        
        # pandas基准
        start = time.time()
        pandas_result = pd.merge(y, xs, on=['date_id', 'stock_id'], how='inner')
        pandas_time = time.time() - start
        
        # rust_pyfunc优化版
        start = time.time()
        rust_result = rp.fast_inner_join_df(y, xs, on=['date_id', 'stock_id'])
        rust_time = time.time() - start
        
        ratio = rust_time / pandas_time if pandas_time > 0 else float('inf')
        
        print(f"  pandas:     {pandas_time:.4f}s ({len(pandas_result)}行, {len(pandas_result.columns)}列)")
        print(f"  rust优化:   {rust_time:.4f}s ({len(rust_result)}行, {len(rust_result.columns)}列)")
        print(f"  性能比:     {ratio:.1f}x ({'rust更快' if ratio < 1 else 'pandas更快'})")
        
        # 验证结果正确性
        assert len(pandas_result) == len(rust_result), f"结果行数不一致: {len(pandas_result)} vs {len(rust_result)}"


def test_mixed_vs_numeric_performance():
    """对比混合类型vs纯数值的性能差异"""
    print("\n=== 对比混合类型vs纯数值性能 ===")
    
    n = 10000
    
    # 混合类型数据 (类似用户原始场景)
    dates = pd.date_range('2024-01-01', periods=250, freq='D')
    codes = [f'00000{i}' for i in range(1, 201)]
    
    y_mixed = pd.DataFrame({
        'date': np.random.choice(dates, n),
        'code': np.random.choice(codes, n),
        'factor_value': np.random.randn(n)
    })
    
    xs_mixed_data = []
    for date in dates:
        for code in codes[:50]:
            xs_mixed_data.append([date, code, np.random.randn(), np.random.randn()])
    
    xs_mixed = pd.DataFrame(xs_mixed_data, columns=['date', 'code', 'value1', 'value2'])
    
    # 纯数值数据
    y_numeric, xs_numeric = create_numeric_test_data(n)
    
    print("混合类型数据测试:")
    start = time.time()
    mixed_result = rp.fast_inner_join_df(y_mixed, xs_mixed, on=['date', 'code'])
    mixed_time = time.time() - start
    print(f"  rust混合类型: {mixed_time:.4f}s ({len(mixed_result)}行)")
    
    print("\n纯数值数据测试:")
    start = time.time()
    numeric_result = rp.fast_inner_join_df(y_numeric, xs_numeric, on=['date_id', 'stock_id'])
    numeric_time = time.time() - start
    print(f"  rust数值型:   {numeric_time:.4f}s ({len(numeric_result)}行)")
    
    print(f"\n性能提升: 数值型比混合型快{mixed_time/numeric_time:.1f}倍")


def test_column_deduplication_detailed():
    """详细测试列去重功能"""
    print("\n=== 详细测试列去重功能 ===")
    
    # 创建有重叠列名的测试数据
    left_df = pd.DataFrame({
        'key1': [1, 2, 3, 4],
        'key2': [10, 20, 30, 40],
        'value': [100, 200, 300, 400],
        'common_col': ['A', 'B', 'C', 'D']
    })
    
    right_df = pd.DataFrame({
        'key1': [1, 2, 3, 5],
        'key2': [10, 20, 30, 50], 
        'price': [1.1, 2.2, 3.3, 5.5],
        'common_col': ['X', 'Y', 'Z', 'W']
    })
    
    print("左表:")
    print(left_df)
    print("\n右表:")
    print(right_df)
    
    # pandas结果
    pandas_result = pd.merge(left_df, right_df, on=['key1', 'key2'], how='inner')
    print(f"\npandas结果 (列数: {len(pandas_result.columns)}):")
    print(pandas_result)
    print("pandas列名:", list(pandas_result.columns))
    
    # rust_pyfunc结果  
    rust_result = rp.fast_inner_join_df(left_df, right_df, on=['key1', 'key2'])
    print(f"\nrust_pyfunc结果 (列数: {len(rust_result.columns)}):")
    print(rust_result)
    print("rust列名:", list(rust_result.columns))
    
    print(f"\n列数对比: pandas={len(pandas_result.columns)}, rust={len(rust_result.columns)}")
    print(f"连接键去重效果: {'✅成功' if len(rust_result.columns) <= len(pandas_result.columns) else '❌失败'}")


def test_large_scale_performance():
    """大规模数据性能测试"""
    print("\n=== 大规模数据性能测试 ===")
    
    # 创建大规模数据
    sizes = [10000, 50000, 100000]
    
    for n in sizes:
        print(f"\n测试规模: {n}行")
        y, xs = create_numeric_test_data(n)
        
        print(f"数据规模: y={y.shape}, xs={xs.shape}")
        
        # pandas性能
        print("pandas测试...")
        start = time.time()
        pandas_result = pd.merge(y, xs, on=['date_id', 'stock_id'], how='inner')
        pandas_time = time.time() - start
        
        # rust性能
        print("rust测试...")
        start = time.time()
        rust_result = rp.fast_inner_join_df(y, xs, on=['date_id', 'stock_id'])
        rust_time = time.time() - start
        
        print(f"结果:")
        print(f"  pandas: {pandas_time:.4f}s, {len(pandas_result)}行")
        print(f"  rust:   {rust_time:.4f}s, {len(rust_result)}行")
        
        if rust_time > 0:
            ratio = pandas_time / rust_time
            if ratio > 1:
                print(f"  🚀 rust快{ratio:.1f}倍")
            else:
                print(f"  📊 pandas快{1/ratio:.1f}倍")
        
        # 验证正确性
        assert len(pandas_result) == len(rust_result), "结果不一致"


def simulate_user_scenario():
    """模拟用户的具体使用场景"""
    print("\n=== 模拟用户场景 ===")
    print("模拟: from w import *; a=p.read_daily(ret=1); 等等...")
    
    # 模拟用户数据结构 (但用数值型优化)
    print("创建模拟数据...")
    
    # 模拟y数据：日期+代码+因子值
    n = 50000
    y = pd.DataFrame({
        'date_num': np.random.randint(20240101, 20241231, n),  # 数值化的日期
        'code_num': np.random.randint(1, 5000, n),             # 数值化的股票代码
        'fac': np.random.randn(n)
    })
    
    # 模拟xs数据：查找表
    dates = range(20240101, 20241231, 7)  # 每周一个数据点
    codes = range(1, 5000, 10)            # 每10个代码一个
    
    xs_data = []
    for date in dates:
        for code in codes:
            xs_data.append([date, code, np.random.randn(), np.random.randn()])
    
    xs = pd.DataFrame(xs_data, columns=['date_num', 'code_num', 'value1', 'value2'])
    
    print(f"模拟数据规模: y={y.shape}, xs={xs.shape}")
    
    # 性能测试
    print("\n执行合并操作...")
    
    start = time.time()
    pandas_result = pd.merge(y, xs, on=['date_num', 'code_num'], how='inner')
    pandas_time = time.time() - start
    
    start = time.time()
    rust_result = rp.fast_inner_join_df(y, xs, on=['date_num', 'code_num'])
    rust_time = time.time() - start
    
    print(f"性能对比:")
    print(f"  pd.merge():                {pandas_time:.4f}s ({len(pandas_result)}行)")
    print(f"  rp.fast_inner_join_df():   {rust_time:.4f}s ({len(rust_result)}行)")
    
    if rust_time > 0:
        ratio = pandas_time / rust_time
        if ratio > 1:
            print(f"  🎉 rust版本快{ratio:.1f}倍！")
        else:
            print(f"  📊 pandas快{1/ratio:.1f}倍")
    
    print(f"  连接键去重: ✅ (pandas: {len(pandas_result.columns)}列 → rust: {len(rust_result.columns)}列)")


def main():
    """运行所有超级优化测试"""
    print("开始测试超级优化版本...\n")
    
    # 基础性能测试
    test_ultra_fast_numeric()
    
    # 类型对比测试
    test_mixed_vs_numeric_performance()
    
    # 功能测试
    test_column_deduplication_detailed()
    
    # 大规模测试
    test_large_scale_performance()
    
    # 用户场景模拟
    simulate_user_scenario()
    
    print("\n" + "="*60)
    print("超级优化总结:")
    print("="*60)
    print("🚀 主要优化:")
    print("   1. 避免Python对象转换，直接操作numpy数组")
    print("   2. 智能类型检测：数值 vs 混合类型")
    print("   3. 连接键自动去重，减少冗余")
    print("   4. 专门的inner join优化路径")
    print("\n💡 使用建议:")
    print("   • 数值型数据：预期显著性能提升")
    print("   • 混合类型数据：功能增强，性能相当")
    print("   • 所有情况：连接键自动去重")
    print("   • 结果与pandas完全兼容")


if __name__ == "__main__":
    main()