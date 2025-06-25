#!/usr/bin/env python3
"""
测试scipy导致的进程卡死问题
"""

import os
import sys
import tempfile
import numpy as np

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

def test_simple_function():
    """测试1：简单函数（应该正常工作）"""
    print("=== 测试1：简单函数 ===")
    
    def simple_func(date, code):
        return [float(i) for i in range(10)]
    
    test_args = [[20240101, "TEST001"], [20240102, "TEST002"]]
    
    with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as tmp_file:
        backup_file = tmp_file.name
    
    try:
        import rust_pyfunc
        
        results = rust_pyfunc.run_pools(
            simple_func,
            test_args,
            backup_file=backup_file,
            num_threads=2
        )
        
        print(f"✅ 简单函数测试成功，结果数量: {len(results)}")
        return True
        
    except Exception as e:
        print(f"❌ 简单函数测试失败: {e}")
        return False
        
    finally:
        if os.path.exists(backup_file):
            os.unlink(backup_file)

def test_numpy_function():
    """测试2：包含numpy计算的函数"""
    print("\n=== 测试2：numpy函数 ===")
    
    def numpy_func(date, code):
        import numpy as np
        # 模拟一些numpy计算
        data1 = np.random.randn(100)
        data2 = np.random.randn(100)
        corr = np.corrcoef(data1, data2)[0, 1]
        return [corr, np.mean(data1), np.std(data2)]
    
    test_args = [[20240101, "TEST001"], [20240102, "TEST002"]]
    
    with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as tmp_file:
        backup_file = tmp_file.name
    
    try:
        import rust_pyfunc
        
        results = rust_pyfunc.run_pools(
            numpy_func,
            test_args,
            backup_file=backup_file,
            num_threads=2
        )
        
        print(f"✅ numpy函数测试成功，结果数量: {len(results)}")
        return True
        
    except Exception as e:
        print(f"❌ numpy函数测试失败: {e}")
        return False
        
    finally:
        if os.path.exists(backup_file):
            os.unlink(backup_file)

def test_scipy_function():
    """测试3：包含scipy计算的函数（可能会卡死）"""
    print("\n=== 测试3：scipy函数 ===")
    
    def scipy_func(date, code):
        import numpy as np
        from scipy import stats  # 关键：导入scipy
        
        # 模拟scipy计算
        data = np.random.randn(50)
        skew_val = stats.skew(data)
        kurt_val = stats.kurtosis(data)
        
        return [skew_val, kurt_val, np.mean(data)]
    
    test_args = [[20240101, "TEST001"], [20240102, "TEST002"]]
    
    with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as tmp_file:
        backup_file = tmp_file.name
    
    try:
        import rust_pyfunc
        
        print("开始scipy函数测试（可能会卡死）...")
        results = rust_pyfunc.run_pools(
            scipy_func,
            test_args,
            backup_file=backup_file,
            num_threads=2
        )
        
        print(f"✅ scipy函数测试成功，结果数量: {len(results)}")
        return True
        
    except Exception as e:
        print(f"❌ scipy函数测试失败: {e}")
        return False
        
    finally:
        if os.path.exists(backup_file):
            os.unlink(backup_file)

def test_rust_function():
    """测试4：包含rust函数调用的函数"""
    print("\n=== 测试4：rust函数调用 ===")
    
    def rust_func(date, code):
        import rust_pyfunc as rp
        import numpy as np
        
        # 模拟rust函数调用
        data1 = np.random.randn(100).astype(float)
        data2 = np.random.randn(100).astype(float)
        
        # 调用rust函数
        dtw_dist = rp.dtw_distance(data1, data2)
        
        return [dtw_dist, np.mean(data1), np.mean(data2)]
    
    test_args = [[20240101, "TEST001"], [20240102, "TEST002"]]
    
    with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as tmp_file:
        backup_file = tmp_file.name
    
    try:
        import rust_pyfunc
        
        results = rust_pyfunc.run_pools(
            rust_func,
            test_args,
            backup_file=backup_file,
            num_threads=2
        )
        
        print(f"✅ rust函数测试成功，结果数量: {len(results)}")
        return True
        
    except Exception as e:
        print(f"❌ rust函数测试失败: {e}")
        return False
        
    finally:
        if os.path.exists(backup_file):
            os.unlink(backup_file)

if __name__ == "__main__":
    print("开始逐步测试以找出问题原因...")
    
    # 按顺序执行测试
    test1_ok = test_simple_function()
    test2_ok = test_numpy_function() 
    test3_ok = test_scipy_function()  # 这个可能会卡死
    test4_ok = test_rust_function()
    
    print(f"\n=== 测试结果总结 ===")
    print(f"简单函数: {'✅' if test1_ok else '❌'}")
    print(f"numpy函数: {'✅' if test2_ok else '❌'}")  
    print(f"scipy函数: {'✅' if test3_ok else '❌'}")
    print(f"rust函数: {'✅' if test4_ok else '❌'}")
    
    if not test3_ok:
        print("\n⚠️ scipy函数测试失败，这很可能是万里长征脚本卡死的原因！")
    elif test1_ok and test2_ok and test3_ok and test4_ok:
        print("\n🎉 所有测试都通过，问题可能在其他地方")