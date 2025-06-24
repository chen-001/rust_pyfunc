#!/usr/bin/env python3
"""
调试函数执行问题
"""

import sys
import os
import traceback

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

def test_function_execution():
    """测试函数执行"""
    print("测试函数执行...")
    
    function_code = """def calc(date, code):
    return [1.0, 2.0, 3.0]"""
    
    print(f"函数代码:\n{function_code}")
    
    try:
        # 模拟worker_process.py中的逻辑
        if "def " in function_code:
            print("检测到函数定义，使用exec模式...")
            exec_globals = {}
            exec(function_code, exec_globals)
            print(f"exec后的globals: {list(exec_globals.keys())}")
            
            # 寻找定义的第一个函数
            func_items = [(name, obj) for name, obj in exec_globals.items() 
                         if callable(obj) and not name.startswith("__")]
            print(f"找到的可调用对象: {[name for name, obj in func_items]}")
            
            if func_items:
                func_name, func_obj = func_items[0]
                print(f"选择的函数: {func_name}")
                
                # 测试函数调用
                print("测试函数调用...")
                result = func_obj(20240101, "TEST001")
                print(f"函数调用结果: {result}")
                print(f"结果类型: {type(result)}")
                
                # 清理前再次测试
                print("再次测试函数调用...")
                result2 = func_obj(20240101, "TEST001")
                print(f"第二次调用结果: {result2}")
                
                # 现在清理
                exec_globals.clear()
                print("清理exec_globals后...")
                
                # 清理后测试
                print("清理后测试函数调用...")
                result3 = func_obj(20240101, "TEST001")
                print(f"清理后调用结果: {result3}")
                
            else:
                print("没有找到可调用的函数")
                
    except Exception as e:
        print(f"执行失败: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    test_function_execution()