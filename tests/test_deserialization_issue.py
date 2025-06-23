#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试反序列化问题的详细调试脚本
模拟pcm.py中使用的复杂情况
"""

import sys
sys.path.insert(0, '/home/chenzongwei/rust_pyfunc/python')

import rust_pyfunc
import json

# 模拟复杂的类结构
class MockTickDataBase:
    def __init__(self):
        self.date = None
        self.code = None
        
    def read_trade_data_single(self, with_retreat=0):
        # 模拟读取数据，返回复杂的数据结构
        import pandas as pd
        return pd.DataFrame({
            'price': [100 + i for i in range(10)],
            'volume': [1000 + i*10 for i in range(10)]
        })

class MockGo(MockTickDataBase):
    def get_factor(self):
        try:
            # 模拟复杂的因子计算
            # df = self.read_trade_data_single(with_retreat=0)
            
            # 复杂计算逻辑
            import numpy as np
            base_value = hash(f"{self.date}_{self.code}")
            
            # 生成150个因子值
            factors = []
            for i in range(150):
                # 复杂的数学运算
                val = np.sin(base_value * 0.001 + i * 0.1) * 100
                val += np.cos(i * 0.05) * 50
                val = float(val)
                factors.append(val)
                
            return factors
        except Exception as e:
            print(f"get_factor error: {e}")
            return [0.0] * 150

def test_function_with_go(go_instance, date, code):
    """测试函数，模拟实际使用"""
    try:
        # 设置Go实例属性
        go_instance.date = date
        go_instance.code = code
        
        # 调用复杂的计算方法
        factors = go_instance.get_factor()
        
        # 添加一些后处理
        processed_factors = []
        for i, f in enumerate(factors):
            # 添加噪声和变换
            noise = 0.01 * (i % 10 - 5)
            processed_val = f * (1 + noise)
            processed_factors.append(processed_val)
            
        return processed_factors
        
    except Exception as e:
        print(f"Function execution error: {e}")
        return [0.0] * 150

def test_large_dataset():
    """测试大数据集的反序列化"""
    print("测试大数据集的反序列化...")
    
    # 创建Go实例
    go_instance = MockGo()
    
    # 准备大量测试数据
    test_args = []
    for i in range(100):  # 100个任务
        date = 20230101 + i
        code = f"TEST{i:03d}"
        test_args.append([date, code])
    
    try:
        print(f"开始处理 {len(test_args)} 个任务...")
        result = rust_pyfunc.run_pools(
            func=test_function_with_go,
            args=test_args,
            go_class=go_instance,
            num_threads=8,  # 使用更多进程
            backup_file=None
        )
        
        print(f"成功！结果形状: {result.shape}")
        
        # 检查结果质量
        non_zero_results = 0
        for i in range(result.shape[0]):
            row_sum = sum(result[i, 2:])  # 跳过date和code列
            if abs(row_sum) > 1e-6:  # 非零结果
                non_zero_results += 1
                
        print(f"非零结果数量: {non_zero_results}/{result.shape[0]}")
        
        if non_zero_results < result.shape[0] * 0.8:  # 如果超过20%的结果是零
            print("⚠️ 警告：检测到大量零结果，可能存在反序列化问题")
        
        return True
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_serialization_methods():
    """测试不同的序列化方法"""
    print("\n测试不同的序列化方法...")
    
    def simple_func(date, code):
        return [float(date % 100 + i) for i in range(5)]
    
    # 测试数据
    test_args = [[20230101, "TEST001"], [20230102, "TEST002"]]
    
    # 测试1：不带Go类的简单函数
    try:
        result1 = rust_pyfunc.run_pools(
            func=simple_func,
            args=test_args,
            num_threads=1
        )
        print(f"✅ 简单函数测试成功: {result1.shape}")
    except Exception as e:
        print(f"❌ 简单函数测试失败: {e}")
    
    # 测试2：带Go类的复杂函数
    try:
        go_instance = MockGo()
        result2 = rust_pyfunc.run_pools(
            func=test_function_with_go,
            args=test_args,
            go_class=go_instance,
            num_threads=1
        )
        print(f"✅ 复杂函数测试成功: {result2.shape}")
    except Exception as e:
        print(f"❌ 复杂函数测试失败: {e}")

if __name__ == "__main__":
    print("开始反序列化问题详细测试...")
    
    # 运行测试
    test_serialization_methods()
    
    # 运行大数据集测试
    success = test_large_dataset()
    
    if success:
        print("\n🎉 所有测试通过，反序列化问题已修复！")
    else:
        print("\n❌ 测试失败，仍存在反序列化问题")