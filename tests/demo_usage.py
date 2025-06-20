"""
演示新的DataFrame rank函数的使用方法
替代原来复杂的转换代码
"""

import pandas as pd
import numpy as np
import time
import rust_pyfunc as rp

# 演示：从复杂的手动转换到简洁的直接调用
def demo_before_after():
    """演示改进前后的使用方式对比"""
    
    print("="*60)
    print("DataFrame rank函数使用方式对比演示")
    print("="*60)
    
    # 创建示例DataFrame
    np.random.seed(42)
    df = pd.DataFrame(
        np.random.randn(5, 8),
        index=[f'股票_{i}' for i in range(5)],
        columns=[f'因子_{i}' for i in range(8)]
    )
    
    print("原始DataFrame:")
    print(df.round(3))
    print()
    
    # === 改进前：需要手动转换 ===
    print("【改进前】需要手动转换的复杂方式:")
    print("代码：pd.DataFrame(rp.rank_axis1(df.to_numpy(dtype=float)), index=df.index, columns=df.columns)")
    
    start_time = time.time()
    result_old_way = pd.DataFrame(
        rp.rank_axis1(df.to_numpy(dtype=float)),
        index=df.index,
        columns=df.columns
    )
    old_time = time.time() - start_time
    
    print("结果:")
    print(result_old_way.round(1))
    print(f"耗时: {old_time:.6f}秒")
    print()
    
    # === 改进后：直接调用 ===
    print("【改进后】直接传入DataFrame的简洁方式:")
    print("代码：rp.rank_axis1_df(df)")
    
    start_time = time.time()
    result_new_way = rp.rank_axis1_df(df)
    new_time = time.time() - start_time
    
    print("结果:")
    print(result_new_way.round(1))
    print(f"耗时: {new_time:.6f}秒")
    print()
    
    # 验证结果一致性
    is_same = np.allclose(result_old_way.values, result_new_way.values, equal_nan=True)
    print(f"结果一致性验证: {is_same} ✓")
    print()
    
    # === 额外的便捷功能 ===
    print("【额外功能】多种便捷的使用方式:")
    
    # 方式1：使用别名
    print("1. 使用简短别名：")
    print("   rp.fast_rank(df)  # 等价于 rank_axis1_df")
    result1 = rp.fast_rank(df)
    print(f"   结果形状: {result1.shape}")
    
    # 方式2：不同参数
    print("\n2. 使用不同参数：")
    print("   rp.rank_axis1_df(df, method='min', ascending=False)")
    result2 = rp.rank_axis1_df(df, method='min', ascending=False)
    print("   降序 + min方法结果:")
    print(result2.round(1))
    
    # 方式3：axis=0
    print("\n3. 沿列方向排名：")
    print("   rp.rank_axis0_df(df)")
    result3 = rp.rank_axis0_df(df)
    print("   axis=0结果:")
    print(result3.round(1))
    
    print("\n" + "="*60)
    print("总结：")
    print("✓ 代码从复杂的手动转换变为简洁的一行调用")
    print("✓ 自动保持DataFrame的索引和列名")
    print("✓ 支持所有pandas.rank()的参数")
    print("✓ 提供多种便捷的别名函数")
    print("✓ 同时支持axis=0和axis=1") 
    print("="*60)


def demo_performance():
    """演示性能优势"""
    
    print("\n性能对比演示")
    print("="*40)
    
    # 创建较大的测试数据
    rows, cols = 2000, 3000
    print(f"测试数据规模: {rows} x {cols}")
    
    np.random.seed(42)
    large_df = pd.DataFrame(np.random.randn(rows, cols))
    
    # pandas原生方法
    print("\n测试pandas原生方法...")
    start = time.time()
    pandas_result = large_df.rank(axis=1)
    pandas_time = time.time() - start
    print(f"pandas.rank(axis=1)耗时: {pandas_time:.4f}秒")
    
    # rust_pyfunc方法
    print("\n测试rust_pyfunc方法...")
    start = time.time()
    rust_result = rp.rank_axis1_df(large_df)
    rust_time = time.time() - start
    print(f"rp.rank_axis1_df()耗时: {rust_time:.4f}秒")
    
    # 性能提升
    speedup = pandas_time / rust_time
    print(f"\n🚀 性能提升: {speedup:.1f}倍")
    
    # 验证一致性
    sample_check = np.allclose(
        pandas_result.iloc[:100, :100].values,
        rust_result.iloc[:100, :100].values,
        equal_nan=True
    )
    print(f"✓ 结果一致性: {sample_check}")


def demo_real_world_usage():
    """演示真实场景的使用方式"""
    
    print("\n真实场景使用演示")
    print("="*40)
    
    # 模拟股票因子数据
    np.random.seed(42)
    stock_data = pd.DataFrame({
        'PE比率': np.random.uniform(10, 50, 1000),
        'PB比率': np.random.uniform(1, 10, 1000),
        'ROE': np.random.uniform(-20, 30, 1000),
        '营收增长率': np.random.uniform(-50, 100, 1000),
        '毛利率': np.random.uniform(10, 80, 1000),
        '市值': np.random.uniform(10, 1000, 1000),
    }, index=[f'股票_{i:04d}' for i in range(1000)])
    
    print("股票因子数据示例:")
    print(stock_data.head().round(2))
    
    print("\n场景1：对每只股票的因子进行排名（用于构建综合评分）")
    print("代码：factor_ranks = rp.rank_axis1_df(stock_data)")
    
    start = time.time()
    factor_ranks = rp.rank_axis1_df(stock_data)
    duration = time.time() - start
    
    print("因子排名结果:")
    print(factor_ranks.head().round(1))
    print(f"处理{len(stock_data)}只股票耗时: {duration:.4f}秒")
    
    print("\n场景2：反向排名（某些因子越小越好）")
    print("代码：rp.rank_axis1_df(stock_data, ascending=False)")
    
    reverse_ranks = rp.rank_axis1_df(stock_data, ascending=False)
    print("反向排名结果:")
    print(reverse_ranks.head().round(1))
    
    print("\n场景3：计算因子分数（标准化排名）")
    normalized_scores = factor_ranks.div(factor_ranks.max(axis=1), axis=0)
    print("标准化分数:")
    print(normalized_scores.head().round(3))
    
    print("\n✓ 全流程保持DataFrame格式，无需手动处理索引和列名")
    print("✓ 代码简洁易读，易于维护")


if __name__ == "__main__":
    # 运行所有演示
    demo_before_after()
    demo_performance()
    demo_real_world_usage()
    
    print("\n" + "="*60)
    print("🎉 现在你可以直接使用：")
    print("   import rust_pyfunc as rp")
    print("   result = rp.rank_axis1_df(your_dataframe)")
    print("   # 或者使用简短别名：")
    print("   result = rp.fast_rank(your_dataframe)")
    print("="*60)