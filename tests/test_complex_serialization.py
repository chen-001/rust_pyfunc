#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试复杂情况下的反序列化问题
"""

import sys
sys.path.insert(0, '/home/chenzongwei/rust_pyfunc/python')

import rust_pyfunc
import numpy as np

class MockGo:
    """模拟的Go类"""
    def __init__(self, date=None, code=None):
        self.date = date  
        self.code = code
        
    def get_factor(self):
        """模拟计算因子"""
        # 复杂的计算逻辑
        base = hash(f"{self.date}_{self.code}") % 1000
        factors = []
        for i in range(150):
            value = (base + i * 3.14159) % 100
            factors.append(float(value))
        return factors

def complex_function(go_instance, date, code):
    """复杂的函数，需要序列化传递"""
    try:
        # 设置Go实例的属性
        go_instance.date = date
        go_instance.code = code
        
        # 调用Go实例的方法
        result = go_instance.get_factor()
        
        # 进行额外的处理
        processed = [x * 1.01 + 0.001 for x in result]
        
        return processed
    except Exception as e:
        print(f"complex_function error: {e}")
        return [0.0] * 150

def test_complex_multiprocess():
    """测试复杂多进程执行"""
    print("开始测试复杂多进程反序列化...")
    
    # 创建Go实例
    go_instance = MockGo()
    
    # 准备测试数据
    test_args = [[20230101 + i, f"TEST{i:03d}"] for i in range(10)]
    
    try:
        print("调用 rust_pyfunc.run_pools 带Go类...")
        result = rust_pyfunc.run_pools(
            func=complex_function,
            args=test_args,
            go_class=go_instance,
            num_threads=2,
            backup_file=None
        )
        
        print(f"成功！结果形状: {result.shape}")
        print(f"第一行结果前5个值: {result[0, 2:7]}")  # 跳过date和code列
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()

def test_serialization_formats():
    """测试不同的序列化格式"""
    print("\\n测试序列化格式检测...")
    
    # 测试源代码格式
    def source_func(date, code):
        return [float(date % 100) + i for i in range(3)]
        
    test_args = [[20230101, "TEST001"]]
    
    try:
        result = rust_pyfunc.run_pools(
            func=source_func,
            args=test_args,
            num_threads=1
        )
        print(f"源代码格式成功！结果: {result[0, 2:]}")
    except Exception as e:
        print(f"源代码格式失败: {e}")

if __name__ == "__main__":
    test_complex_multiprocess()
    test_serialization_formats()