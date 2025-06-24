#!/usr/bin/env python3
"""
详细调试工作进程
"""

import os
import sys
import json
import tempfile

# 添加项目路径  
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

# 导入工作进程模块
import worker_process

def test_worker_step_by_step():
    """逐步测试工作进程"""
    print("=== 逐步测试工作进程 ===")
    
    # 1. 检查初始状态
    print(f"1. 初始CALCULATE_FUNCTION: {worker_process.CALCULATE_FUNCTION}")
    print(f"1. 初始GO_CLASS_INSTANCE: {worker_process.GO_CLASS_INSTANCE}")
    
    # 2. 设置函数
    function_code = """def calc(date, code):
    print(f"calc called with date={date}, code={code}")
    result = [1.0, 2.0, 3.0]
    print(f"calc returning: {result}")
    return result"""
    
    print(f"2. 设置函数代码:\n{function_code}")
    worker_process.set_function(function_code)
    print(f"2. 设置后CALCULATE_FUNCTION: {worker_process.CALCULATE_FUNCTION}")
    print(f"2. CALCULATE_FUNCTION是否可调用: {callable(worker_process.CALCULATE_FUNCTION)}")
    
    # 3. 直接测试函数
    if callable(worker_process.CALCULATE_FUNCTION):
        print("3. 直接测试CALCULATE_FUNCTION...")
        try:
            direct_result = worker_process.CALCULATE_FUNCTION(20240101, "TEST001")
            print(f"3. 直接调用结果: {direct_result}")
        except Exception as e:
            print(f"3. 直接调用失败: {e}")
            import traceback
            traceback.print_exc()
    
    # 4. 测试execute_tasks
    print("4. 测试execute_tasks...")
    tasks = [{"date": 20240101, "code": "TEST001"}]
    
    # 添加调试到execute_tasks
    print("4. 准备执行任务...")
    print(f"4. 任务: {tasks}")
    
    result = worker_process.execute_tasks(tasks)
    
    print(f"4. execute_tasks返回: {result}")
    print(f"4. 结果详情:")
    for i, res in enumerate(result.get('results', [])):
        print(f"   结果{i}: {res} (类型: {type(res)}, 长度: {len(res)})")
    for i, err in enumerate(result.get('errors', [])):
        print(f"   错误{i}: {err}")

if __name__ == "__main__":
    test_worker_step_by_step()