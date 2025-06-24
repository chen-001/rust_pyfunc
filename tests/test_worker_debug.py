#!/usr/bin/env python3
"""
调试工作进程的输入输出
"""

import os
import sys
import json
import tempfile

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

# 导入工作进程模块
import worker_process

def test_worker_directly():
    """直接测试工作进程逻辑"""
    print("测试工作进程逻辑...")
    
    # 模拟设置函数
    function_code = """def calc(date, code):
    return [1.0, 2.0, 3.0]"""
    
    worker_process.set_function(function_code)
    
    # 模拟执行任务
    tasks = [{"date": 20240101, "code": "TEST001"}]
    
    print("执行任务...")
    result = worker_process.execute_tasks(tasks)
    
    print(f"工作进程返回结果: {result}")
    print(f"结果类型: {type(result)}")
    print(f"结果详情:")
    print(f"  - results: {result.get('results', [])}")
    print(f"  - results类型: {type(result.get('results', []))}")
    if result.get('results'):
        print(f"  - 第一个结果: {result['results'][0]}")
        print(f"  - 第一个结果类型: {type(result['results'][0])}")
        print(f"  - 第一个结果长度: {len(result['results'][0])}")
    print(f"  - errors: {result.get('errors', [])}")
    print(f"  - task_count: {result.get('task_count', 0)}")

if __name__ == "__main__":
    test_worker_directly()