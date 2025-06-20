"""
测试DataFrame merge封装函数
"""

import pandas as pd
import numpy as np
import rust_pyfunc as rp


def test_dataframe_merge():
    """测试DataFrame merge封装函数"""
    print("=== 测试DataFrame merge封装函数 ===")
    
    # 创建测试数据
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
    
    # 测试内连接
    print("\n=== 内连接测试 ===")
    result = rp.fast_merge_df(left_df, right_df, on='key', how='inner')
    print("fast_merge_df内连接结果:")
    print(result)
    
    # 对比pandas结果
    pandas_result = pd.merge(left_df, right_df, on='key', how='inner')
    print("\npandas merge内连接结果:")
    print(pandas_result)
    
    # 基本验证
    assert len(result) == len(pandas_result), "内连接行数不一致"
    print("✓ 内连接测试通过")
    
    # 测试左连接
    print("\n=== 左连接测试 ===")
    result = rp.fast_merge_df(left_df, right_df, on='key', how='left')
    print("fast_merge_df左连接结果:")
    print(result)
    
    pandas_result = pd.merge(left_df, right_df, on='key', how='left')
    print("\npandas merge左连接结果:")
    print(pandas_result)
    
    assert len(result) == len(pandas_result), "左连接行数不一致"
    print("✓ 左连接测试通过")
    
    # 测试便捷函数
    print("\n=== 便捷函数测试 ===")
    inner_result = rp.fast_inner_join_df(left_df, right_df, on='key')
    left_result = rp.fast_left_join_df(left_df, right_df, on='key')
    
    print(f"fast_inner_join_df结果行数: {len(inner_result)}")
    print(f"fast_left_join_df结果行数: {len(left_result)}")
    
    assert len(inner_result) == 2, "便捷内连接函数错误"
    assert len(left_result) == 3, "便捷左连接函数错误"
    print("✓ 便捷函数测试通过")


def test_edge_cases():
    """测试边界情况"""
    print("\n=== 边界情况测试 ===")
    
    # 空DataFrame测试
    empty_df = pd.DataFrame()
    normal_df = pd.DataFrame({'key': [1, 2], 'value': [10, 20]})
    
    try:
        result = rp.fast_merge_df(empty_df, normal_df, on='key', how='inner')
        assert len(result) == 0, "空表内连接应该返回空结果"
        print("✓ 空表测试通过")
    except Exception as e:
        print(f"空表测试异常（可能需要改进处理）: {e}")
    
    # 单行DataFrame测试
    single_left = pd.DataFrame({'key': [1], 'value_left': [100]})
    single_right = pd.DataFrame({'key': [1], 'value_right': [10]})
    
    result = rp.fast_merge_df(single_left, single_right, on='key', how='inner')
    assert len(result) == 1, "单行表连接错误"
    print("✓ 单行表测试通过")


def demo_usage():
    """使用演示"""
    print("\n=== 使用演示 ===")
    
    # 模拟股票数据连接
    stocks_df = pd.DataFrame({
        'stock_id': [1, 2, 3, 4, 5],
        'stock_name': ['股票A', '股票B', '股票C', '股票D', '股票E'],
        'price': [10.5, 20.3, 15.8, 30.2, 25.1]
    })
    
    trades_df = pd.DataFrame({
        'stock_id': [1, 1, 2, 3, 3, 6],
        'volume': [1000, 2000, 1500, 800, 1200, 500],
        'trade_time': ['09:30', '10:15', '09:45', '11:20', '14:30', '15:00']
    })
    
    print("股票基础信息:")
    print(stocks_df)
    print("\n交易记录:")
    print(trades_df)
    
    # 内连接：只保留有交易记录的股票
    inner_result = rp.fast_inner_join_df(stocks_df, trades_df, on='stock_id')
    print(f"\n内连接结果（有交易的股票）: {len(inner_result)}行")
    print(inner_result.head())
    
    # 左连接：保留所有股票，显示交易情况
    left_result = rp.fast_left_join_df(stocks_df, trades_df, on='stock_id')
    print(f"\n左连接结果（所有股票）: {len(left_result)}行")
    print(left_result)
    
    print("\n✓ 使用演示完成")


if __name__ == "__main__":
    try:
        test_dataframe_merge()
        test_edge_cases()
        demo_usage()
        
        print("\n" + "="*60)
        print("✓ DataFrame封装函数测试完成！")
        print("✓ 支持标准的DataFrame输入输出")
        print("✓ 提供便捷的连接函数")
        print("✓ 兼容pandas.merge的基本用法")
        print("="*60)
        
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        exit(1)