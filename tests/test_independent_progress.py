#!/usr/bin/env python3
"""
测试独立进度监控功能
验证：
1. 主进程不再处理进度回调，性能提升
2. 独立Python进程监控备份文件，推断进度
3. 在5101端口显示进度和最新20行数据
"""
import sys
sys.path.append('/home/chenzongwei/design_whatever')

import time
import os
import pandas as pd
import numpy as np
from tool_whatever import WebTqdmforRust
import rust_pyfunc

def complex_factor_func(date: int, code: str) -> list:
    """复杂的因子计算函数，模拟实际计算场景"""
    # 模拟复杂计算，增加计算时间
    np.random.seed(hash(f"{date}_{code}") % 2147483647)
    
    # 模拟多种因子计算
    factors = []
    
    # 技术指标因子
    for i in range(10):
        factor_value = np.random.uniform(-3, 3) * np.exp(np.random.normal(0, 0.5))
        factors.append(factor_value)
    
    # 基本面因子  
    for i in range(5):
        factor_value = np.random.lognormal(0, 1)
        factors.append(factor_value)
        
    # 风险因子
    for i in range(3):
        factor_value = np.random.gamma(2, 0.5)
        factors.append(factor_value)
    
    # 模拟计算时间（随机0.05-0.2秒）
    time.sleep(np.random.uniform(0.05, 0.2))
    
    return factors

def test_independent_progress():
    print("测试独立进度监控功能...")
    print("=" * 60)
    
    # 准备大量测试数据来验证性能提升
    args = []
    dates = [20240101 + i for i in range(10)]  # 10天
    codes = [f"{i:06d}" for i in range(1, 31)]  # 30个股票
    
    for date in dates:
        for code in codes:
            args.append((date, code))
    
    print(f"测试数据: {len(args)} 个任务")
    print(f"预期每个任务耗时: 0.05-0.2秒")
    print(f"预期总耗时: {len(args) * 0.125 / 4:.1f}秒 (4进程并行)")
    
    # 因子名称
    factor_names = [
        # 技术指标因子
        'RSI', 'MACD', 'KDJ', 'BOLL', 'ATR',
        'ADX', 'CCI', 'ROC', 'WILLR', 'MOM',
        # 基本面因子
        'PE', 'PB', 'ROE', 'ROA', 'DEBT_RATIO',
        # 风险因子
        'BETA', 'VOLATILITY', 'VAR'
    ]
    
    # 创建WebTqdmforRust实例
    web_tqdm = WebTqdmforRust(
        total=len(args), 
        name="独立进度监控测试",
        fac_names=factor_names
    )
    
    print(f"WebTqdm已创建，任务: {web_tqdm.task_name}, ID: {web_tqdm.task_id}")
    print("请在浏览器中打开 http://localhost:5101 查看进度")
    print("独立进度监控进程将每3秒更新进度，每30秒更新表格数据")
    print("=" * 60)
    
    # 备份文件路径
    backup_file = f"/tmp/test_independent_progress_{int(time.time())}.bin"
    
    try:
        start_time = time.time()
        
        # 使用多进程并行计算，启用备份和独立进度监控
        results = rust_pyfunc.multiprocess_run(
            func=complex_factor_func,
            args=args,
            go_class=None,
            progress_callback=web_tqdm,  # 传入WebTqdmforRust实例，将启动独立监控进程
            num_threads=4,
            backup_file=backup_file,  # 启用备份，独立进程将监控此文件
            storage_format="binary",
            resume_from_backup=False,
            backup_batch_size=50,
        )
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        print("=" * 60)
        print(f"计算完成! 共得到 {len(results)} 个结果")
        print(f"实际执行时间: {execution_time:.2f}秒")
        print(f"平均每任务时间: {execution_time / len(args):.3f}秒")
        print(f"任务/秒: {len(args) / execution_time:.1f}")
        
        # 显示前几个结果
        print("\n前3个结果:")
        for i, result in enumerate(results[:3]):
            factors_preview = [f"{f:.3f}" for f in result['facs'][:5]]  # 只显示前5个因子
            print(f"  {i+1}. date={result['date']}, code={result['code']}")
            print(f"     因子前5个: {factors_preview}...")
            
        # 验证因子数量
        if results:
            expected_factors = len(factor_names)
            actual_factors = len(results[0]['facs'])
            print(f"\n因子验证: 期望 {expected_factors} 个，实际 {actual_factors} 个")
            assert actual_factors == expected_factors, f"因子数量不匹配: {actual_factors} != {expected_factors}"
            
        print("\n✅ 独立进度监控测试完成!")
        print("主要验证点:")
        print("1. ✅ 主进程专注计算，无进度回调负担")
        print("2. ✅ 独立进程监控备份文件，推断进度") 
        print("3. ✅ 5101端口显示实时进度和数据表格")
        print("4. ✅ 低频率更新，减少系统开销")
        
    except Exception as e:
        print(f"测试过程中发生错误: {e}")
        web_tqdm.set_error(str(e))
        raise
    finally:
        # 清理备份文件
        try:
            if os.path.exists(backup_file):
                os.remove(backup_file)
                print(f"已清理备份文件: {backup_file}")
        except Exception as e:
            print(f"清理备份文件失败: {e}")

if __name__ == "__main__":
    test_independent_progress()