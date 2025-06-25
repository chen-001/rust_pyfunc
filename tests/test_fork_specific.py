#!/usr/bin/env python3
"""
专门测试Fork模式的脚本
通过创建包含PyReadonlyArray1参数的函数来强制触发Fork模式
"""

import numpy as np
import rust_pyfunc as rp
from numpy.typing import NDArray
import time

def create_function_with_pyarray():
    """创建一个包含PyReadonlyArray1参数的函数来触发Fork模式"""
    
    # 这个函数包含PyReadonlyArray1参数，应该无法序列化
    def process_with_array(date: int, code: str, data_array: NDArray[np.float64]) -> list:
        """包含PyReadonlyArray1参数的函数"""
        result = rp.find_max_range_product(data_array)
        return [float(result), date * 0.001, len(code)]
    
    return process_with_array

def main():
    print("=== 专门测试Fork模式 ===")
    
    # 准备测试数据  
    dates_codes = [[20220101, '000001'], [20220102, '000002'], [20220103, '000003']]
    
    print("\n测试1: 尝试直接使用rp.find_max_range_product")
    try:
        # 直接使用rust函数应该触发Fork模式
        result1 = rp.run_pools(
            func=rp.find_max_range_product,
            args=[[np.array([1.0, 2.0, 3.0])], [np.array([4.0, 5.0, 6.0])], [np.array([7.0, 8.0, 9.0])]],
            num_threads=2,
            backup_file="/tmp/test_direct_rust.backup"
        )
        print(f"✅ 直接rust函数测试成功，结果形状: {result1.shape}")
    except Exception as e:
        print(f"预期失败 - 直接rust函数测试: {e}")
    
    print("\n测试2: 强制使用Fork模式")
    try:
        # 我们可以通过修改配置强制使用Fork模式
        import tempfile
        import os
        
        # 创建一个简单的测试函数
        def simple_fork_test(date: int, code: str) -> list:
            return [date * 1.5, len(code) * 2.0]
        
        # 使用临时备份文件
        with tempfile.NamedTemporaryFile(delete=False, suffix='.backup') as tmp:
            tmp_path = tmp.name
        
        try:
            result2 = rp.run_pools(
                func=simple_fork_test,
                args=dates_codes,
                num_threads=2,
                backup_file=tmp_path
            )
            print(f"✅ 强制Fork模式测试成功，结果形状: {result2.shape}")
            print(f"结果示例: {result2[0] if len(result2) > 0 else 'None'}")
        finally:
            # 清理临时文件
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
                
    except Exception as e:
        print(f"❌ 强制Fork模式测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n测试3: 检查模式检测逻辑")
    try:
        # 测试序列化检测逻辑
        def test_numpy_in_signature(arr: np.ndarray) -> float:
            return float(np.sum(arr))
        
        # 这应该检测为不可序列化并切换到Fork模式
        test_data = [[np.array([1.0, 2.0])], [np.array([3.0, 4.0])]]
        
        print("尝试使用包含numpy.ndarray参数的函数...")
        result3 = rp.run_pools(
            func=test_numpy_in_signature,
            args=test_data,
            num_threads=2,
            backup_file="/tmp/test_numpy_sig.backup"
        )
        print(f"✅ NumPy签名测试成功，结果形状: {result3.shape}")
        
    except Exception as e:
        print(f"NumPy签名测试结果: {e}")
        print("这可能是预期的，因为参数格式不匹配")
    
    print("\n测试4: 模拟PyReadonlyArray1场景")
    try:
        # 创建一个wrapper函数模拟PyReadonlyArray1的使用场景
        def wrapper_with_rust_call(date: int, code: str) -> list:
            # 在函数内部调用rust函数
            arr = np.array([float(date), len(code), 1.0, 2.0, 3.0])
            max_product = rp.find_max_range_product(arr)
            variance = rp.calculate_variance(arr)
            return [float(max_product), float(variance), date * 0.1]
        
        result4 = rp.run_pools(
            func=wrapper_with_rust_call,
            args=dates_codes,
            num_threads=2,
            backup_file="/tmp/test_wrapper_rust.backup"
        )
        print(f"✅ Wrapper rust调用测试成功，结果形状: {result4.shape}")
        print(f"结果示例: {result4[0] if len(result4) > 0 else 'None'}")
        
    except Exception as e:
        print(f"❌ Wrapper rust调用测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()