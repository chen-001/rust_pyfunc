#!/usr/bin/env python3
"""
测试Fork模式并行执行功能
验证包含PyReadonlyArray1参数的函数可以正常并行执行
"""

import numpy as np
import rust_pyfunc as rp
from numpy.typing import NDArray
import time

def test_simple_function(date: int, code: str) -> list:
    """简单函数，可以序列化"""
    return [date * 0.1, len(code) * 0.2]

def test_numpy_function(arr: NDArray[np.float64], multiplier: float) -> list:
    """包含NumPy数组参数的函数，不可序列化"""
    return [float(np.sum(arr) * multiplier), float(np.mean(arr))]

def test_mixed_function(date: int, arr: NDArray[np.float64]) -> list:
    """混合参数函数"""
    return [date + np.sum(arr), date * np.mean(arr)]

def main():
    print("=== Fork模式并行执行测试 ===")
    
    # 准备测试数据 - run_pools期望的格式是[[date, code], ...]
    dates_codes = [[20220101, '000001'], [20220102, '000002'], [20220103, '000003']]
    
    print("\n1. 测试简单可序列化函数")
    try:
        result1 = rp.run_pools(
            func=test_simple_function,
            args=dates_codes,
            num_threads=2,
            backup_file="/tmp/test_simple.backup"
        )
        print(f"✅ 简单函数测试成功，结果形状: {result1.shape}")
        print(f"结果示例: {result1[0] if len(result1) > 0 else 'None'}")
    except Exception as e:
        print(f"❌ 简单函数测试失败: {e}")
    
    print("\n2. 测试包含NumPy数组的函数")
    try:
        # 创建一个包含数组参数的测试函数
        def numpy_wrapper(date: int, code: str) -> list:
            # 这里模拟调用rp.find_max_range_product等函数
            arr = np.array([float(date), len(code), 1.5, 2.0])
            return test_numpy_function(arr, 1.5)
        
        result2 = rp.run_pools(
            func=numpy_wrapper,
            args=dates_codes,
            num_threads=2,
            backup_file="/tmp/test_numpy.backup"
        )
        print(f"✅ NumPy函数测试成功，结果形状: {result2.shape}")
        print(f"结果示例: {result2[0] if len(result2) > 0 else 'None'}")
    except Exception as e:
        print(f"❌ NumPy函数测试失败: {e}")
    
    print("\n3. 测试真实的rust_pyfunc函数")
    try:
        def real_rust_function(date: int, code: str) -> list:
            # 测试实际包含PyReadonlyArray1的rust函数
            arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
            result = rp.find_max_range_product(arr)
            return [float(result), date * 0.001]
        
        start_time = time.time()
        result3 = rp.run_pools(
            func=real_rust_function,
            args=dates_codes,
            num_threads=2,
            backup_file="/tmp/test_rust.backup"
        )
        end_time = time.time()
        
        print(f"✅ 真实rust函数测试成功，结果形状: {result3.shape}")
        print(f"执行时间: {end_time - start_time:.3f}秒")
        print(f"结果示例: {result3[0] if len(result3) > 0 else 'None'}")
    except Exception as e:
        print(f"❌ 真实rust函数测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n4. 性能对比测试")
    try:
        def performance_test_func(date: int, code: str) -> list:
            # 创建较大的数组进行测试
            large_arr = np.random.random(1000)
            result = rp.find_max_range_product(large_arr)
            return [float(result), date, len(code)]
        
        # 准备更多测试数据
        large_test_data = [[20220100 + i, f'00000{i}'] for i in range(20)]
        
        print("开始大数据量测试...")
        start_time = time.time()
        result4 = rp.run_pools(
            func=performance_test_func,
            args=large_test_data,
            num_threads=4,
            backup_file="/tmp/test_performance.backup"
        )
        end_time = time.time()
        
        print(f"✅ 大数据量测试成功，结果形状: {result4.shape}")
        print(f"处理{len(large_test_data)}个任务耗时: {end_time - start_time:.3f}秒")
        print(f"平均每个任务: {(end_time - start_time) / len(large_test_data) * 1000:.1f}ms")
    except Exception as e:
        print(f"❌ 性能测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()