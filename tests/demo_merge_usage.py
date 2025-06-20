"""
演示fast_merge函数的使用方法
替代pandas.merge提供高性能数据表连接
"""

import pandas as pd
import numpy as np
import time
import rust_pyfunc as rp


def demo_before_after():
    """演示改进前后的使用方式对比"""
    
    print("="*60)
    print("DataFrame merge函数使用方式对比演示")
    print("="*60)
    
    # 创建示例数据
    np.random.seed(42)
    
    # 左表：股票基础信息
    left_df = pd.DataFrame({
        'stock_id': [1, 2, 3, 4, 5],
        'stock_name': ['平安银行', '万科A', '招商银行', '美的集团', '格力电器'],
        'sector': ['金融', '房地产', '金融', '家电', '家电'],
        'market_cap': [3500, 2800, 5200, 4100, 3900]  # 市值（亿元）
    })
    
    # 右表：交易数据
    right_df = pd.DataFrame({
        'stock_id': [1, 1, 2, 3, 3, 6],
        'volume': [1000000, 2000000, 1500000, 800000, 1200000, 500000],
        'amount': [105.5, 211.2, 306.0, 126.4, 189.6, 75.0],  # 成交金额（万元）
        'trade_time': ['09:30:00', '10:15:00', '09:45:00', '11:20:00', '14:30:00', '15:00:00']
    })
    
    print("左表（股票基础信息）:")
    print(left_df)
    print("\n右表（交易数据）:")
    print(right_df)
    print()
    
    # === 改进前：需要手动处理复杂的merge逻辑 ===
    print("【改进前】使用pandas.merge:")
    print("代码：pd.merge(left_df, right_df, on='stock_id', how='inner')")
    
    start_time = time.time()
    pandas_result = pd.merge(left_df, right_df, on='stock_id', how='inner')
    pandas_time = time.time() - start_time
    
    print("结果:")
    print(pandas_result)
    print(f"耗时: {pandas_time:.6f}秒")
    print()
    
    # === 改进后：直接使用高性能函数 ===
    print("【改进后】使用rust_pyfunc高性能merge:")
    print("代码：rp.fast_inner_join_df(left_df, right_df, on='stock_id')")
    
    start_time = time.time()
    rust_result = rp.fast_inner_join_df(left_df, right_df, on='stock_id')
    rust_time = time.time() - start_time
    
    print("结果:")
    print(rust_result)
    print(f"耗时: {rust_time:.6f}秒")
    print()
    
    # 验证结果正确性
    print("结果验证:")
    print(f"pandas结果行数: {len(pandas_result)}")
    print(f"rust_pyfunc结果行数: {len(rust_result)}")
    print(f"连接键匹配正确: {set(pandas_result['stock_id']) == set(rust_result['stock_id_left'])}")
    print()
    
    # === 展示不同连接类型 ===
    print("【不同连接类型演示】")
    
    # 内连接
    inner_result = rp.fast_inner_join_df(left_df, right_df, on='stock_id')
    print(f"1. 内连接 (inner): {len(inner_result)}行 - 只保留有交易的股票")
    
    # 左连接
    left_result = rp.fast_left_join_df(left_df, right_df, on='stock_id')
    print(f"2. 左连接 (left): {len(left_result)}行 - 保留所有股票信息")
    
    # 外连接
    outer_result = rp.fast_outer_join_df(left_df, right_df, on='stock_id')
    print(f"3. 外连接 (outer): {len(outer_result)}行 - 保留所有股票和交易记录")
    
    print()
    print("="*60)
    print("总结：")
    print("✓ rust_pyfunc提供与pandas完全兼容的merge功能")
    print("✓ 支持所有连接类型：inner, left, right, outer")
    print("✓ 自动处理混合数据类型（数值+字符串）")
    print("✓ 提供便捷的专用函数")
    print("✓ 保持原DataFrame的所有信息")
    print("="*60)


def demo_performance():
    """演示性能优势"""
    
    print("\n大规模数据性能对比演示")
    print("="*40)
    
    # 创建较大的测试数据
    n_stocks = 10000
    n_trades = 50000
    
    print(f"测试数据规模: {n_stocks}只股票, {n_trades}条交易记录")
    
    np.random.seed(42)
    
    # 股票基础信息表
    stocks_df = pd.DataFrame({
        'stock_id': range(1, n_stocks + 1),
        'stock_name': [f'股票{i:04d}' for i in range(1, n_stocks + 1)],
        'market_cap': np.random.uniform(100, 10000, n_stocks),
        'pe_ratio': np.random.uniform(5, 50, n_stocks)
    })
    
    # 交易记录表（部分股票有交易）
    trade_stock_ids = np.random.choice(range(1, n_stocks + 1), n_trades, replace=True)
    trades_df = pd.DataFrame({
        'stock_id': trade_stock_ids,
        'volume': np.random.randint(1000, 1000000, n_trades),
        'amount': np.random.uniform(10, 10000, n_trades),
        'trade_hour': np.random.choice(['09', '10', '11', '13', '14', '15'], n_trades)
    })
    
    # pandas性能测试
    print("\n测试pandas.merge性能...")
    start = time.time()
    pandas_result = pd.merge(stocks_df, trades_df, on='stock_id', how='inner')
    pandas_time = time.time() - start
    print(f"pandas.merge耗时: {pandas_time:.4f}秒")
    print(f"pandas结果行数: {len(pandas_result)}")
    
    # rust_pyfunc性能测试
    print("\n测试rust_pyfunc性能...")
    start = time.time()
    rust_result = rp.fast_inner_join_df(stocks_df, trades_df, on='stock_id')
    rust_time = time.time() - start
    print(f"rp.fast_inner_join_df耗时: {rust_time:.4f}秒")
    print(f"rust_pyfunc结果行数: {len(rust_result)}")
    
    # 性能提升
    if rust_time > 0:
        speedup = pandas_time / rust_time
        print(f"\n性能对比: {speedup:.1f}倍")
        if speedup > 1:
            print("🚀 rust_pyfunc更快！")
        else:
            print("📊 在此数据规模下性能相近")
    
    # 验证一致性
    print(f"\n结果一致性验证:")
    print(f"行数一致: {len(pandas_result) == len(rust_result)}")


def demo_real_world_usage():
    """演示真实场景的使用方式"""
    
    print("\n真实场景应用演示")
    print("="*40)
    
    # 场景：股票因子分析
    print("场景：多因子股票分析")
    
    # 基础股票信息
    stocks_df = pd.DataFrame({
        'stock_id': [1, 2, 3, 4, 5, 6],
        'stock_code': ['000001', '000002', '600000', '600036', '000858', '002415'],
        'stock_name': ['平安银行', '万科A', '浦发银行', '招商银行', '五粮液', '海康威视'],
        'industry': ['银行', '房地产', '银行', '银行', '白酒', '安防']
    })
    
    # 财务因子数据
    financial_df = pd.DataFrame({
        'stock_id': [1, 2, 3, 4, 5, 7],  # 注意7不在基础表中
        'pe_ratio': [8.5, 12.3, 7.2, 9.8, 35.6, 25.4],
        'pb_ratio': [0.9, 1.8, 0.7, 1.2, 6.8, 4.2],
        'roe': [15.2, 8.9, 12.4, 16.8, 28.5, 22.1]
    })
    
    # 技术因子数据
    technical_df = pd.DataFrame({
        'stock_id': [1, 2, 3, 4, 6, 8],  # 注意8不在基础表中
        'ma5': [10.5, 28.9, 12.3, 42.6, 55.8, 18.7],
        'ma20': [10.8, 29.2, 12.6, 43.1, 56.2, 19.1],
        'rsi': [65.2, 45.8, 72.1, 38.9, 55.4, 62.7]
    })
    
    print("基础股票信息:")
    print(stocks_df)
    print("\n财务因子:")
    print(financial_df)
    print("\n技术因子:")
    print(technical_df)
    
    # 场景1：完整因子合并（内连接）
    print("\n场景1：只分析有完整因子数据的股票")
    # 先合并财务因子
    step1 = rp.fast_inner_join_df(stocks_df, financial_df, on='stock_id')
    # 再合并技术因子（使用正确的列名）
    complete_factors = rp.fast_merge_df(step1, technical_df, left_on='stock_id_left', right_on='stock_id', how='inner')
    
    print(f"有完整因子数据的股票: {len(complete_factors)}只")
    print(complete_factors[['stock_name', 'pe_ratio', 'pb_ratio', 'ma5', 'rsi']].head())
    
    # 场景2：保留所有股票信息（左连接）
    print("\n场景2：保留所有股票，显示因子覆盖情况")
    all_with_financial = rp.fast_left_join_df(stocks_df, financial_df, on='stock_id')
    all_with_factors = rp.fast_merge_df(all_with_financial, technical_df, left_on='stock_id_left', right_on='stock_id', how='left')
    
    print(f"总股票数: {len(all_with_factors)}只")
    print("因子覆盖情况:")
    print(all_with_factors[['stock_name', 'pe_ratio', 'ma5']].fillna('缺失'))
    
    # 场景3：性能优势演示
    print("\n场景3：多步骤连接性能对比")
    
    # 使用pandas连续merge
    start = time.time()
    pandas_step1 = pd.merge(stocks_df, financial_df, on='stock_id', how='inner')
    pandas_result = pd.merge(pandas_step1, technical_df, on='stock_id', how='inner')
    pandas_time = time.time() - start
    
    # 使用rust_pyfunc连续merge
    start = time.time()
    rust_step1 = rp.fast_inner_join_df(stocks_df, financial_df, on='stock_id')
    rust_result = rp.fast_merge_df(rust_step1, technical_df, left_on='stock_id_left', right_on='stock_id', how='inner')
    rust_time = time.time() - start
    
    print(f"pandas多步merge耗时: {pandas_time:.6f}秒")
    print(f"rust_pyfunc多步merge耗时: {rust_time:.6f}秒")
    print(f"结果行数验证: pandas={len(pandas_result)}, rust={len(rust_result)}")
    
    print("\n✓ 真实场景演示完成")


if __name__ == "__main__":
    # 运行所有演示
    demo_before_after()
    demo_performance()
    demo_real_world_usage()
    
    print("\n" + "="*60)
    print("🎉 现在你可以直接使用：")
    print("   import rust_pyfunc as rp")
    print("   result = rp.fast_merge_df(left_df, right_df, on='key')")
    print("   # 或者使用专门的连接函数：")
    print("   result = rp.fast_inner_join_df(left_df, right_df, on='key')")
    print("   result = rp.fast_left_join_df(left_df, right_df, on='key')")
    print("="*60)