#!/usr/bin/env python3
"""
测试WebTqdmforRust的表格数据更新功能
验证在5101端口页面上能看到最新备份的20行数据
"""
import sys
sys.path.append('/home/chenzongwei/design_whatever')

import time
import pandas as pd
import numpy as np
from tool_whatever import WebTqdmforRust
import rust_pyfunc

def simple_factor_func(date: int, code: str) -> list:
    """简单的因子计算函数，返回模拟的因子数据"""
    # 模拟计算一些因子
    np.random.seed(hash(f"{date}_{code}") % 2147483647)
    factors = [
        np.random.uniform(0.1, 2.0),  # 因子1: 收益率
        np.random.uniform(0.5, 1.5),  # 因子2: 波动率  
        np.random.uniform(-1, 1),     # 因子3: 动量
        np.random.uniform(0, 100),    # 因子4: 成交量
    ]
    
    # 模拟一些计算时间
    time.sleep(0.1)
    return factors

def test_web_table_update():
    print("测试WebTqdmforRust的表格数据更新功能...")
    
    # 准备测试数据
    args = []
    dates = [20240101, 20240102, 20240103]
    codes = ['000001', '000002', '600000', '600036']
    
    for date in dates:
        for code in codes:
            args.append((date, code))
    
    print(f"测试数据: {len(args)} 个任务")
    
    # 因子名称
    factor_names = ['收益率', '波动率', '动量', '成交量']
    
    # 创建WebTqdmforRust实例，包含因子名称
    web_tqdm = WebTqdmforRust(
        total=len(args), 
        name="因子计算测试",
        fac_names=factor_names
    )
    
    print(f"WebTqdm已启动，任务: {web_tqdm.task_name}, ID: {web_tqdm.task_id}")
    print("请在浏览器中打开 http://localhost:5101 查看进度")
    print("测试将每处理5个结果时更新一次表格数据...")
    
    try:
        # 使用多进程并行计算
        results = rust_pyfunc.multiprocess_run(
            func=simple_factor_func,
            args=args,
            go_class=None,
            progress_callback=web_tqdm,  # 传入WebTqdmforRust实例
            chunk_size=None,  # 异步模式不使用chunk_size
        )
        
        print(f"计算完成! 共得到 {len(results)} 个结果")
        
        # 显示前几个结果作为验证
        print("\n前5个结果:")
        for i, result in enumerate(results[:5]):
            print(f"  {i+1}. date={result['date']}, code={result['code']}, facs={result['facs'][:2]}...")
            
        # 最终标记任务完成
        web_tqdm.finish()
        
        print("测试完成！请检查5101端口页面是否显示了最新的20行数据")
        print("页面应该显示包含因子名称的展开数据表格")
        
    except Exception as e:
        print(f"测试过程中发生错误: {e}")
        web_tqdm.set_error(str(e))
        raise

if __name__ == "__main__":
    test_web_table_update()