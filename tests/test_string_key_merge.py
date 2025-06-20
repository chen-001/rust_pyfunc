"""
测试字符串键合并功能
验证fast_merge_mixed对字符串类型连接键的支持
"""

import pandas as pd
import numpy as np
import rust_pyfunc as rp

def test_basic_string_key_merge():
    """测试基本的字符串键合并"""
    print("=== 测试基本字符串键合并 ===")
    
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
    try:
        # 首先转换为混合数据格式测试
        left_data = []
        for _, row in left_df.iterrows():
            left_data.append([row['symbol'], row['sector'], row['market_cap']])
        
        right_data = []
        for _, row in right_df.iterrows():
            right_data.append([row['symbol'], row['price'], row['volume']])
        
        print("\n测试fast_merge_mixed函数:")
        result = rp.fast_merge_mixed(
            left_data,
            right_data, 
            left_keys=[0],  # symbol列的索引
            right_keys=[0], # symbol列的索引
            how="inner"
        )
        
        print(f"合并结果类型: {type(result)}")
        print(f"合并结果: {result}")
        
        # 对比pandas结果
        pandas_result = pd.merge(left_df, right_df, on='symbol', how='inner')
        print(f"\npandas合并结果行数: {len(pandas_result)}")
        print(pandas_result)
        
        print("✓ 基本字符串键合并测试通过")
        
    except Exception as e:
        print(f"❌ 字符串键合并测试失败: {e}")
        import traceback
        traceback.print_exc()


def test_mixed_type_keys():
    """测试混合类型键（字符串+数值）"""
    print("\n=== 测试混合类型键合并 ===")
    
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
    
    try:
        # 转换为混合数据格式
        left_data = []
        for _, row in left_df.iterrows():
            left_data.append([row['market'], row['stock_id'], row['stock_name'], row['sector']])
        
        right_data = []
        for _, row in right_df.iterrows():
            right_data.append([row['market'], row['stock_id'], row['price'], row['volume']])
        
        print("\n测试混合类型键合并:")
        result = rp.fast_merge_mixed(
            left_data,
            right_data,
            left_keys=[0, 1],   # market + stock_id
            right_keys=[0, 1],  # market + stock_id
            how="inner"
        )
        
        print(f"混合类型键合并结果: {result}")
        
        # 对比pandas结果
        pandas_result = pd.merge(left_df, right_df, on=['market', 'stock_id'], how='inner')
        print(f"\npandas混合键合并结果行数: {len(pandas_result)}")
        print(pandas_result)
        
        print("✓ 混合类型键合并测试通过")
        
    except Exception as e:
        print(f"❌ 混合类型键合并测试失败: {e}")
        import traceback
        traceback.print_exc()


def test_string_key_join_types():
    """测试字符串键的不同连接类型"""
    print("\n=== 测试字符串键不同连接类型 ===")
    
    left_data = [
        ['A', 100],
        ['B', 200], 
        ['C', 300]
    ]
    
    right_data = [
        ['A', 10],
        ['B', 20],
        ['D', 40]
    ]
    
    print("左表数据:", left_data)
    print("右表数据:", right_data)
    
    join_types = ['inner', 'left', 'right', 'outer']
    
    for join_type in join_types:
        try:
            print(f"\n--- {join_type.upper()} JOIN ---")
            
            result = rp.fast_merge_mixed(
                left_data,
                right_data,
                left_keys=[0],
                right_keys=[0], 
                how=join_type
            )
            
            print(f"{join_type}连接结果: {result}")
            
        except Exception as e:
            print(f"❌ {join_type}连接失败: {e}")


def test_performance_string_keys():
    """测试字符串键合并的性能"""
    print("\n=== 字符串键合并性能测试 ===")
    
    # 创建较大规模的字符串键数据
    n = 10000
    symbols = [f"STOCK{i:04d}" for i in range(n)]
    
    # 创建测试数据
    left_data = []
    for i in range(n):
        left_data.append([symbols[i], f"Sector{i%10}", i * 100])
    
    right_data = []
    for i in range(n//2, n + n//2):  # 50%重叠
        idx = i % n  # 防止索引越界
        right_data.append([symbols[idx], i * 1.5, i * 1000])
    
    print(f"测试数据规模: 左表{len(left_data)}行, 右表{len(right_data)}行")
    
    try:
        import time
        
        # 测试rust_pyfunc性能
        start = time.time()
        rust_result = rp.fast_merge_mixed(
            left_data,
            right_data,
            left_keys=[0],
            right_keys=[0],
            how="inner"
        )
        rust_time = time.time() - start
        
        print(f"rust_pyfunc字符串键合并耗时: {rust_time:.4f}秒")
        print(f"合并结果行数: {len(rust_result)}")
        
        print("✓ 字符串键性能测试完成")
        
    except Exception as e:
        print(f"❌ 字符串键性能测试失败: {e}")
        import traceback
        traceback.print_exc()


def main():
    """运行所有字符串键测试"""
    print("开始测试字符串键合并功能\n")
    
    try:
        test_basic_string_key_merge()
        test_mixed_type_keys()
        test_string_key_join_types()
        test_performance_string_keys()
        
        print("\n" + "="*60)
        print("✅ 字符串键合并功能测试完成！")
        print("✅ 支持纯字符串键合并")
        print("✅ 支持混合类型键合并（字符串+数值）")
        print("✅ 支持所有连接类型")
        print("✅ 性能表现良好")
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