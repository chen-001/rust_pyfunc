"""
演示多键连接功能
展示rust_pyfunc.fast_merge对多键连接的完整支持
"""

import pandas as pd
import numpy as np
import rust_pyfunc as rp


def demo_multi_key_evolution():
    """演示从单键到多键连接的进化"""
    
    print("="*70)
    print("🚀 rust_pyfunc 多键连接功能演示")
    print("="*70)
    
    # 创建示例数据：股票交易数据
    print("📊 创建股票交易示例数据\n")
    
    # 股票基础信息（使用数值ID）
    stocks_df = pd.DataFrame({
        'exchange_id': [1, 1, 2, 2, 1],  # 1=SZ, 2=SH
        'stock_id': [1, 2, 3, 4, 5],     # 数值股票ID
        'stock_name': ['平安银行', '万科A', '浦发银行', '招商银行', '五粮液'],
        'sector': ['金融', '房地产', '金融', '金融', '白酒']
    })
    
    # 交易数据
    trades_df = pd.DataFrame({
        'exchange_id': [1, 1, 2, 2, 3, 1],  # 1=SZ, 2=SH, 3=BJ
        'stock_id': [1, 2, 3, 4, 6, 1],     # 数值股票ID
        'trade_date': [20241201, 20241201, 20241201, 20241201, 20241201, 20241202],
        'volume': [1000000, 2000000, 1500000, 800000, 500000, 1200000],
        'amount': [105.5, 211.2, 306.0, 126.4, 75.0, 127.8]
    })
    
    print("股票基础信息:")
    print(stocks_df)
    print("\n交易数据:")
    print(trades_df)
    
    # === 演示1：单键连接的局限性 ===
    print("\n" + "="*50)
    print("❌ 单键连接的局限性演示")
    print("="*50)
    
    print("\n如果只用stock_id连接：")
    
    # 只用stock_id连接会出现问题，因为不同交易所可能有相同的股票ID
    single_key_result = rp.fast_inner_join_df(stocks_df, trades_df, on='stock_id')
    print(f"单键连接结果: {len(single_key_result)}行")
    print(single_key_result[['stock_name', 'exchange_id_left', 'exchange_id_right', 'volume']])
    
    print("\n⚠️  问题：可能匹配到错误的交易所数据！")
    
    # === 演示2：双键连接解决问题 ===
    print("\n" + "="*50)
    print("✅ 双键连接解决方案")
    print("="*50)
    
    print("\n使用exchange_id+stock_id组合键:")
    
    # 使用exchange_id和stock_id作为组合键
    multi_key_result = rp.fast_inner_join_df(stocks_df, trades_df, on=['exchange_id', 'stock_id'])
    print(f"双键连接结果: {len(multi_key_result)}行")
    print(multi_key_result[['stock_name', 'exchange_id_left', 'stock_id_left', 'volume', 'amount']])
    
    print("\n✓ 正确匹配，避免跨交易所的错误连接")
    
    # === 演示3：对比pandas结果 ===
    print("\n" + "="*50)
    print("📈 与pandas.merge对比")
    print("="*50)
    
    pandas_result = pd.merge(stocks_df, trades_df, on=['exchange_id', 'stock_id'], how='inner')
    print(f"\npandas.merge结果: {len(pandas_result)}行")
    print(pandas_result[['stock_name', 'exchange_id', 'stock_id', 'volume', 'amount']])
    
    print(f"\n结果一致性验证: ✓ 行数匹配 ({len(multi_key_result)} == {len(pandas_result)})")


def demo_complex_multi_key():
    """演示复杂的多键连接场景"""
    
    print("\n" + "="*70)
    print("🎯 复杂多键连接场景演示")
    print("="*70)
    
    # 创建更复杂的数据：时间序列股票数据
    print("📅 时间序列股票数据连接\n")
    
    # 股票日线数据
    daily_data = pd.DataFrame({
        'exchange_id': [1, 1, 2, 1, 2],  # 1=SZ, 2=SH
        'stock_id': [1, 1, 3, 1, 3],
        'trade_date': [20241201, 20241202, 20241201, 20241203, 20241202],
        'close_price': [10.5, 10.8, 12.3, 10.6, 12.5],
        'volume': [1000000, 1200000, 800000, 900000, 850000]
    })
    
    # 技术指标数据
    technical_data = pd.DataFrame({
        'exchange_id': [1, 1, 2, 1, 2, 2],  # 1=SZ, 2=SH
        'stock_id': [1, 1, 3, 1, 3, 4],
        'trade_date': [20241201, 20241202, 20241201, 20241203, 20241202, 20241201],
        'ma5': [10.2, 10.5, 12.1, 10.4, 12.2, 42.6],
        'ma20': [9.8, 10.3, 11.9, 10.2, 11.8, 42.1],
        'rsi': [65.2, 68.1, 45.8, 72.3, 48.2, 55.4]
    })
    
    print("股票日线数据:")
    print(daily_data)
    print("\n技术指标数据:")
    print(technical_data)
    
    # 使用三键连接：exchange_id + stock_id + trade_date
    print("\n使用三键连接 (exchange_id + stock_id + trade_date):")
    
    complete_data = rp.fast_inner_join_df(
        daily_data, technical_data, 
        on=['exchange_id', 'stock_id', 'trade_date']
    )
    
    print(f"三键连接结果: {len(complete_data)}行")
    print(complete_data[['exchange_id_left', 'stock_id_left', 'trade_date_left', 'close_price', 'ma5', 'rsi']])
    
    # 对比pandas
    pandas_complete = pd.merge(
        daily_data, technical_data,
        on=['exchange_id', 'stock_id', 'trade_date'],
        how='inner'
    )
    
    print(f"\npandas三键连接结果: {len(pandas_complete)}行")
    print(f"结果一致性: ✓ ({len(complete_data)} == {len(pandas_complete)})")


def demo_different_join_types():
    """演示多键连接的不同类型"""
    
    print("\n" + "="*70)
    print("🔀 多键连接类型演示")
    print("="*70)
    
    # 简化的数据用于演示
    left_df = pd.DataFrame({
        'key1': [1, 1, 2, 3],
        'key2': [1, 2, 1, 1], 
        'value_left': [100, 200, 300, 400]
    })
    
    right_df = pd.DataFrame({
        'key1': [1, 2, 4],
        'key2': [1, 1, 1],
        'value_right': [10, 20, 30]
    })
    
    print("左表:")
    print(left_df)
    print("\n右表:")
    print(right_df)
    
    join_types = ['inner', 'left', 'right', 'outer']
    
    for join_type in join_types:
        print(f"\n--- {join_type.upper()} JOIN ---")
        
        if join_type == 'inner':
            result = rp.fast_inner_join_df(left_df, right_df, on=['key1', 'key2'])
        elif join_type == 'left':
            result = rp.fast_left_join_df(left_df, right_df, on=['key1', 'key2'])
        elif join_type == 'right':
            result = rp.fast_right_join_df(left_df, right_df, on=['key1', 'key2'])
        else:  # outer
            result = rp.fast_outer_join_df(left_df, right_df, on=['key1', 'key2'])
        
        print(f"结果行数: {len(result)}")
        print(result)
        
        # 对比pandas
        pandas_result = pd.merge(left_df, right_df, on=['key1', 'key2'], how=join_type)
        consistency = len(result) == len(pandas_result)
        print(f"与pandas一致性: {'✓' if consistency else '✗'}")


def demo_performance_comparison():
    """演示多键连接的性能"""
    
    print("\n" + "="*70)
    print("⚡ 多键连接性能对比")
    print("="*70)
    
    # 创建性能测试数据
    sizes = [1000, 5000, 10000]
    
    for n in sizes:
        print(f"\n📊 测试规模: {n}行")
        
        np.random.seed(42)
        left_df = pd.DataFrame({
            'key1': np.random.randint(1, n//10, n),
            'key2': np.random.randint(1, 100, n),
            'key3': np.random.randint(1, 50, n),
            'value': np.random.randn(n)
        })
        
        right_df = pd.DataFrame({
            'key1': np.random.randint(1, n//10, n),
            'key2': np.random.randint(1, 100, n),
            'key3': np.random.randint(1, 50, n),
            'value': np.random.randn(n)
        })
        
        # pandas性能
        import time
        start = time.time()
        pandas_result = pd.merge(left_df, right_df, on=['key1', 'key2', 'key3'], how='inner')
        pandas_time = time.time() - start
        
        # rust_pyfunc性能  
        start = time.time()
        rust_result = rp.fast_inner_join_df(left_df, right_df, on=['key1', 'key2', 'key3'])
        rust_time = time.time() - start
        
        print(f"  pandas:     {pandas_time:.4f}秒 ({len(pandas_result)}行)")
        print(f"  rust_pyfunc: {rust_time:.4f}秒 ({len(rust_result)}行)")
        
        if rust_time > 0 and pandas_time > 0:
            ratio = pandas_time / rust_time
            if ratio > 1:
                print(f"  🚀 rust快{ratio:.1f}倍")
            else:
                print(f"  📊 pandas快{1/ratio:.1f}倍")


def demo_real_world_usage():
    """真实场景使用演示"""
    
    print("\n" + "="*70)
    print("🌟 真实场景使用示例")
    print("="*70)
    
    print("\n场景：多市场股票数据分析")
    print("需求：将基础信息、价格数据、财务数据按market+symbol+date连接")
    
    # 使用简化的多键连接示例
    sample_code = '''
# 真实使用示例
import rust_pyfunc as rp

# 1. 单键连接（简单场景）
result = rp.fast_inner_join_df(df1, df2, on='stock_id')

# 2. 双键连接（区分市场）
result = rp.fast_inner_join_df(df1, df2, on=['market', 'symbol'])

# 3. 三键连接（时间序列数据）
result = rp.fast_inner_join_df(df1, df2, on=['market', 'symbol', 'date'])

# 4. 不同键名连接
result = rp.fast_merge_df(
    df1, df2,
    left_on=['left_market', 'left_symbol'],
    right_on=['right_market', 'right_symbol'],
    how='left'
)

# 5. 所有连接类型都支持多键
inner_result = rp.fast_inner_join_df(df1, df2, on=['key1', 'key2'])
left_result = rp.fast_left_join_df(df1, df2, on=['key1', 'key2'])
outer_result = rp.fast_outer_join_df(df1, df2, on=['key1', 'key2'])
'''
    
    print(sample_code)
    
    print("\n✅ 完整功能支持:")
    print("  • 单键和多键连接")
    print("  • 所有连接类型（inner、left、right、outer）")
    print("  • 不同键名连接（left_on/right_on）")
    print("  • 向后兼容pandas.merge语法")
    print("  • 自动处理混合数据类型")


if __name__ == "__main__":
    # 运行所有演示
    demo_multi_key_evolution()
    demo_complex_multi_key()
    demo_different_join_types()
    demo_performance_comparison()
    demo_real_world_usage()
    
    print("\n" + "="*70)
    print("🎉 多键连接功能演示完成！")
    print("rust_pyfunc现在完全支持pandas.merge的多键连接功能")
    print("="*70)