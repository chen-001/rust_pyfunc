#!/usr/bin/env python3
"""
调试异步流水线实现
"""

import time
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

import rust_pyfunc as rf
import numpy as np

def debug_function(date, code):
    """调试函数：返回明确的结果"""
    print(f"DEBUG: 执行函数，date={date}, code={code}")
    result = [1.0, 2.0, 3.0]  # 简单的固定结果
    print(f"DEBUG: 函数返回: {result}")
    return result

def test_single_task():
    """测试单个任务"""
    print("=== 测试单个任务 ===")
    
    test_args = [[20240101, '000001']]  # 只有一个任务
    
    print(f"测试参数: {test_args}")
    
    try:
        results = rf.run_pools(
            func=debug_function,
            args=test_args,
            num_threads=2,
            backup_file=None,
        )
        
        print(f"返回结果形状: {results.shape}")
        print(f"返回结果内容:")
        print(results)
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_single_task()