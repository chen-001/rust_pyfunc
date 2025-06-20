import pandas as pd
import numpy as np
import time
from functools import reduce
import rust_pyfunc

def get_groups_python(df: pd.DataFrame, groups_num: int):
    """依据因子值，判断是在第几组（原始Python版本）"""
    if "group" in list(df.columns):
        df = df.drop(columns=["group"])
    df = df.sort_values(["fac"], ascending=True)
    each_group = round(df.shape[0] / groups_num)
    l = list(
        map(
            lambda x, y: [x] * y,
            list(range(1, groups_num + 1)),
            [each_group] * groups_num,
        )
    )
    l = reduce(lambda x, y: x + y, l)
    if len(l) < df.shape[0]:
        l = l + [groups_num] * (df.shape[0] - len(l))
    l = l[: df.shape[0]]
    df.insert(0, "group", l)
    return df

def test_factor_grouping_performance():
    """测试因子分组功能的性能对比"""
    print("正在生成测试数据...")
    
    # 模拟创建测试数据（类似用户提供的代码）
    np.random.seed(42)
    n_rows = 1_600_000
    
    # 生成日期数据（假设有200个交易日）
    dates = np.random.choice(range(20220101, 20220301), n_rows)
    
    # 生成因子数据
    factors = np.random.randn(n_rows) * 100
    
    # 创建DataFrame用于Python版本测试
    df = pd.DataFrame({
        'date': dates,
        'code': ['stock_' + str(i % 4000) for i in range(n_rows)],  # 假设4000只股票
        'fac': factors
    })
    
    print(f"测试数据大小: {n_rows:,} 行，{len(np.unique(dates))} 个交易日")
    print(f"内存使用: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    
    groups_num = 10
    
    print("\n=" * 60)
    print("开始Python版本测试...")
    start_time = time.time()
    
    result_python = df.groupby("date").apply(
        lambda x: get_groups_python(x, groups_num)
    )
    
    python_time = time.time() - start_time
    print(f"Python版本用时: {python_time:.3f} 秒")
    
    print("\n=" * 60)  
    print("开始Rust版本测试...")
    start_time = time.time()
    
    # 转换为numpy数组用于Rust函数
    dates_array = df['date'].values.astype(np.int64)
    factors_array = df['fac'].values.astype(np.float64)
    
    result_rust = rust_pyfunc.factor_grouping(dates_array, factors_array, groups_num)
    
    rust_time = time.time() - start_time
    print(f"Rust版本用时: {rust_time:.3f} 秒")
    
    print("\n=" * 60)
    print("性能对比结果:")
    print(f"Python版本: {python_time:.3f} 秒")
    print(f"Rust版本: {rust_time:.3f} 秒") 
    print(f"速度提升: {python_time / rust_time:.1f}x")
    
    # 验证结果正确性
    print("\n=" * 60)
    print("验证结果正确性...")
    
    # 取一个样本日期验证
    sample_date = dates[0]
    sample_mask = df['date'] == sample_date
    sample_df = df[sample_mask].copy()
    
    # Python结果
    python_sample = get_groups_python(sample_df, groups_num)['group'].values
    
    # Rust结果 
    rust_sample = result_rust[sample_mask]
    
    print(f"样本日期 {sample_date}:")
    print(f"Python结果前10个: {python_sample[:10]}")
    print(f"Rust结果前10个: {rust_sample[:10]}")
    
    # 检查分组分布
    python_groups = np.bincount(python_sample)[1:]  # 跳过0（如果有的话）
    rust_groups = np.bincount(rust_sample)[1:]
    
    print(f"Python各组数量: {python_groups}")
    print(f"Rust各组数量: {rust_groups}")
    
    # 检查结果一致性（允许小的差异，因为分组边界可能有差异）
    if len(python_groups) == len(rust_groups):
        print("✓ 分组数量一致")
    else:
        print("✗ 分组数量不一致")
        
    print("\n测试完成！")

if __name__ == "__main__":
    test_factor_grouping_performance()