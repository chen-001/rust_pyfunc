"""
简单的fast_merge使用演示
"""

import pandas as pd
import numpy as np
import time
import rust_pyfunc as rp

print("="*60)
print("🚀 rust_pyfunc.fast_merge 高性能数据表连接演示")
print("="*60)

# 创建示例数据
print("创建测试数据...")
left_df = pd.DataFrame({
    'stock_id': [1, 2, 3, 4, 5],
    'stock_name': ['平安银行', '万科A', '招商银行', '美的集团', '格力电器'],
    'market_cap': [3500, 2800, 5200, 4100, 3900]
})

right_df = pd.DataFrame({
    'stock_id': [1, 2, 3, 6, 7],
    'volume': [1000000, 2000000, 1500000, 800000, 1200000],
    'amount': [105.5, 211.2, 306.0, 126.4, 189.6]
})

print("\n左表（股票基础信息）:")
print(left_df)
print("\n右表（交易数据）:")
print(right_df)

# 测试不同连接类型
print("\n" + "="*40)
print("连接类型演示")
print("="*40)

# 1. 内连接
print("\n1. 内连接 (只保留两表都有的记录):")
inner_result = rp.fast_inner_join_df(left_df, right_df, on='stock_id')
print(f"结果行数: {len(inner_result)}")
print(inner_result)

# 2. 左连接
print("\n2. 左连接 (保留左表所有记录):")
left_result = rp.fast_left_join_df(left_df, right_df, on='stock_id')
print(f"结果行数: {len(left_result)}")
print(left_result)

# 3. 外连接
print("\n3. 外连接 (保留两表所有记录):")
outer_result = rp.fast_outer_join_df(left_df, right_df, on='stock_id')
print(f"结果行数: {len(outer_result)}")
print(outer_result)

# 性能对比
print("\n" + "="*40)
print("性能对比")
print("="*40)

# 创建较大数据进行性能测试
n = 50000
print(f"\n创建{n}行测试数据进行性能对比...")

np.random.seed(42)
large_left = pd.DataFrame({
    'key': range(n),
    'value_left': np.random.randn(n)
})

large_right = pd.DataFrame({
    'key': range(n//2, n + n//2),  # 50%重叠
    'value_right': np.random.randn(n)
})

# pandas性能
print("测试pandas.merge性能...")
start = time.time()
pandas_result = pd.merge(large_left, large_right, on='key', how='inner')
pandas_time = time.time() - start

# rust_pyfunc性能
print("测试rust_pyfunc性能...")
start = time.time()
rust_result = rp.fast_inner_join_df(large_left, large_right, on='key')
rust_time = time.time() - start

print(f"\n性能结果:")
print(f"pandas.merge耗时:     {pandas_time:.4f}秒 ({len(pandas_result)}行)")
print(f"fast_inner_join_df耗时: {rust_time:.4f}秒 ({len(rust_result)}行)")

if rust_time > 0:
    speedup = pandas_time / rust_time
    if speedup > 1:
        print(f"🚀 rust_pyfunc快{speedup:.1f}倍！")
    else:
        print(f"📊 在此规模下性能相近（pandas快{1/speedup:.1f}倍）")

print(f"结果一致性: {len(pandas_result) == len(rust_result)}")

# 使用建议
print("\n" + "="*40)
print("使用建议")
print("="*40)

print("\n✅ 推荐使用场景:")
print("  • 大规模数据表连接（>10万行）")
print("  • 外连接操作（rust_pyfunc在外连接上有明显优势）")
print("  • 需要频繁进行表连接的场景")
print("  • 数值型数据的连接")

print("\n📝 使用方法:")
print("  import rust_pyfunc as rp")
print("  ")
print("  # 基本用法")
print("  result = rp.fast_merge_df(left_df, right_df, on='key', how='inner')")
print("  ")
print("  # 便捷函数")
print("  result = rp.fast_inner_join_df(left_df, right_df, on='key')")
print("  result = rp.fast_left_join_df(left_df, right_df, on='key')")
print("  result = rp.fast_outer_join_df(left_df, right_df, on='key')")

print("\n⚠️  注意事项:")
print("  • 连接键必须是数值类型")
print("  • 非数值列会保持原有数据类型")
print("  • 列名冲突时会自动添加_left/_right后缀")

print("\n" + "="*60)
print("演示完成！🎉")
print("="*60)