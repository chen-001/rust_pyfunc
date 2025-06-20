"""
性能分析脚本
识别fast_inner_join_df的性能瓶颈
"""

import pandas as pd
import numpy as np
import rust_pyfunc as rp
import time


def create_test_data(n=50000):
    """创建测试数据，模拟用户的场景"""
    print(f"创建测试数据：{n}行")
    
    # 模拟y数据 (date, code, fac)
    dates = pd.date_range('2024-01-01', periods=250, freq='D')  # 250个交易日
    codes = [f'00000{i}' for i in range(1, 201)]  # 200只股票
    
    # 创建y数据
    y_data = []
    for _ in range(n):
        date = np.random.choice(dates)
        code = np.random.choice(codes)
        fac = np.random.randn()
        y_data.append([date, code, fac])
    
    y = pd.DataFrame(y_data, columns=['date', 'code', 'fac'])
    
    # 创建xs数据 (模拟较小的查找表)
    xs_data = []
    for date in dates:
        for code in codes[:50]:  # 只取50只股票，造成部分匹配
            xs_data.append([date, code, np.random.randn(), np.random.randn()])
    
    xs = pd.DataFrame(xs_data, columns=['date', 'code', 'value1', 'value2'])
    
    print(f"y数据形状: {y.shape}")
    print(f"xs数据形状: {xs.shape}")
    print(f"y数据类型: {y.dtypes.to_dict()}")
    print(f"xs数据类型: {xs.dtypes.to_dict()}")
    
    return y, xs


def profile_current_implementation(y, xs):
    """分析当前实现的各个步骤耗时"""
    print("\n=== 分析当前实现的性能瓶颈 ===")
    
    # 1. pandas baseline
    print("1. pandas.merge基准测试...")
    start = time.time()
    pandas_result = pd.merge(y, xs, on=['date', 'code'], how='inner')
    pandas_time = time.time() - start
    print(f"   pandas耗时: {pandas_time:.4f}秒, 结果: {len(pandas_result)}行")
    
    # 2. rust_pyfunc当前实现
    print("2. rust_pyfunc当前实现...")
    start = time.time()
    rust_result = rp.fast_inner_join_df(y, xs, on=['date', 'code'])
    rust_time = time.time() - start
    print(f"   rust_pyfunc耗时: {rust_time:.4f}秒, 结果: {len(rust_result)}行")
    
    print(f"3. 性能对比: pandas快{rust_time/pandas_time:.1f}倍")
    
    return pandas_time, rust_time, len(pandas_result)


def profile_data_conversion():
    """分析数据转换的耗时"""
    print("\n=== 分析数据转换耗时 ===")
    
    y, xs = create_test_data(10000)  # 较小数据集用于详细分析
    
    # 分析DataFrame转换为列表的耗时
    print("1. DataFrame to list转换...")
    start = time.time()
    
    left_data = []
    for _, row in y.iterrows():
        left_data.append(row.tolist())
    
    right_data = []
    for _, row in xs.iterrows():
        right_data.append(row.tolist())
    
    conversion_time = time.time() - start
    print(f"   转换耗时: {conversion_time:.4f}秒")
    
    # 分析键索引获取
    start = time.time()
    left_key_indices = [y.columns.get_loc('date'), y.columns.get_loc('code')]
    right_key_indices = [xs.columns.get_loc('date'), xs.columns.get_loc('code')]
    index_time = time.time() - start
    print(f"   索引获取: {index_time:.6f}秒")
    
    # 分析rust调用
    start = time.time()
    indices, merged_data = rp.fast_merge_mixed(
        left_data, right_data,
        left_keys=left_key_indices,
        right_keys=right_key_indices,
        how="inner"
    )
    rust_call_time = time.time() - start
    print(f"   rust调用: {rust_call_time:.4f}秒")
    
    # 分析结果重建
    start = time.time()
    result_dfs = []
    left_col_names = []
    for col in y.columns:
        left_col_names.append(f"{col}_left" if col in xs.columns else col)
    
    right_col_names = []
    for col in xs.columns:
        right_col_names.append(f"{col}_right" if col in y.columns else col)
    
    all_col_names = left_col_names + right_col_names
    
    for row in merged_data:
        result_dfs.append(dict(zip(all_col_names, row)))
    
    result_df = pd.DataFrame(result_dfs)
    rebuild_time = time.time() - start
    print(f"   结果重建: {rebuild_time:.4f}秒")
    
    total_analyzed = conversion_time + rust_call_time + rebuild_time
    print(f"   总计: {total_analyzed:.4f}秒")
    
    return {
        'conversion': conversion_time,
        'rust_call': rust_call_time, 
        'rebuild': rebuild_time,
        'total': total_analyzed
    }


def test_different_sizes():
    """测试不同数据规模下的性能"""
    print("\n=== 测试不同数据规模的性能 ===")
    
    sizes = [1000, 5000, 10000, 20000]
    results = []
    
    for n in sizes:
        print(f"\n测试规模: {n}行")
        y, xs = create_test_data(n)
        
        # pandas
        start = time.time()
        pandas_result = pd.merge(y, xs, on=['date', 'code'], how='inner')
        pandas_time = time.time() - start
        
        # rust_pyfunc
        start = time.time()
        rust_result = rp.fast_inner_join_df(y, xs, on=['date', 'code'])
        rust_time = time.time() - start
        
        ratio = rust_time / pandas_time if pandas_time > 0 else float('inf')
        
        results.append({
            'size': n,
            'pandas_time': pandas_time,
            'rust_time': rust_time,
            'ratio': ratio,
            'result_rows': len(pandas_result)
        })
        
        print(f"  pandas: {pandas_time:.4f}s, rust: {rust_time:.4f}s, ratio: {ratio:.1f}x")
    
    return results


def main():
    """运行性能分析"""
    print("开始性能分析...")
    
    # 详细的性能分析
    profile_data_conversion()
    
    # 不同规模测试
    test_different_sizes()
    
    print("\n=== 性能分析结论 ===")
    print("主要瓶颈:")
    print("1. DataFrame转列表的iterrows()操作非常慢")
    print("2. 结果重建时的字典创建和DataFrame构造耗时")
    print("3. 混合类型处理增加了额外开销")
    print("\n优化方向:")
    print("1. 避免iterrows()，使用values或to_numpy()") 
    print("2. 优化结果重建，减少Python对象创建")
    print("3. 对纯数值键提供快速路径")
    print("4. 去除重复的连接键列")


if __name__ == "__main__":
    main()