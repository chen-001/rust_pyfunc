#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调试工作进程问题
"""

import subprocess
import json
import sys
import time

def test_worker_process_directly():
    """直接测试工作进程"""
    print("🔍 直接测试Python工作进程")
    
    # 启动工作进程
    python_path = "/home/chenzongwei/.conda/envs/chenzongwei311/bin/python"
    script_path = "/home/chenzongwei/rust_pyfunc/python/worker_process.py"
    
    try:
        proc = subprocess.Popen(
            [python_path, script_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        print(f"✅ 工作进程启动成功，PID: {proc.pid}")
        
        # 测试简单的任务
        test_request = {
            "tasks": [
                {"date": 20220101, "code": "000001"},
                {"date": 20220102, "code": "000002"}
            ],
            "function_code": """
def test_func(date, code):
    return [float(date % 1000), float(len(code)), 1.0]
"""
        }
        
        print(f"📤 发送测试请求...")
        request_json = json.dumps(test_request) + "\n"
        proc.stdin.write(request_json)
        proc.stdin.flush()
        
        print(f"📥 等待响应...")
        # 设置超时
        try:
            stdout, stderr = proc.communicate(timeout=10)
            
            print(f"📊 标准输出:")
            print(stdout)
            
            if stderr:
                print(f"⚠️ 标准错误:")
                print(stderr)
                
            print(f"🔄 进程返回码: {proc.returncode}")
            
        except subprocess.TimeoutExpired:
            print(f"⏰ 进程超时")
            proc.kill()
            stdout, stderr = proc.communicate()
            print(f"超时后的输出: {stdout}")
            print(f"超时后的错误: {stderr}")
        
    except Exception as e:
        print(f"❌ 启动工作进程失败: {e}")

def test_function_serialization():
    """测试函数序列化"""
    print(f"\n🔍 测试函数序列化")
    
    def test_func(date, code):
        return [float(date % 1000), float(len(code)), 1.0]
    
    # 测试获取源代码
    try:
        import inspect
        source = inspect.getsource(test_func)
        print(f"✅ 可以获取函数源代码:")
        print(source)
    except Exception as e:
        print(f"❌ 无法获取函数源代码: {e}")
    
    # 测试pickle
    try:
        import pickle
        pickle.dumps(test_func)
        print(f"✅ 函数可以pickle化")
    except Exception as e:
        print(f"❌ 函数无法pickle化: {e}")

if __name__ == "__main__":
    test_function_serialization()
    test_worker_process_directly()