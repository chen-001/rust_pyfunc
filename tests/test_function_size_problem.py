#!/usr/bin/env python3
"""
测试函数大小导致的问题
"""

import os
import sys
import tempfile

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

def test_small_function():
    """测试小函数"""
    print("=== 测试小函数 ===")
    
    def small_func(date, code):
        return [1.0, 2.0, 3.0]
    
    test_args = [[20170101, "000001"]]
    
    with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as tmp_file:
        backup_file = tmp_file.name
    
    try:
        import rust_pyfunc
        
        print("开始小函数测试...")
        results = rust_pyfunc.run_pools(
            small_func,
            test_args,
            backup_file=backup_file,
            num_threads=1
        )
        
        print(f"✅ 小函数测试成功，结果数量: {len(results)}")
        return True
        
    except Exception as e:
        print(f"❌ 小函数测试失败: {e}")
        return False
        
    finally:
        if os.path.exists(backup_file):
            os.unlink(backup_file)

def test_medium_function():
    """测试中等大小函数"""
    print("\n=== 测试中等大小函数 ===")
    
    def medium_func(date, code):
        import numpy as np
        import pandas as pd
        
        # 50行左右的函数
        def helper1():
            return np.array([1, 2, 3, 4, 5])
        
        def helper2():
            return pd.Series([1, 2, 3, 4, 5])
            
        def helper3():
            return {"a": 1, "b": 2}
        
        # 一些计算
        arr = helper1()
        series = helper2()
        dict_data = helper3()
        
        result1 = arr.mean()
        result2 = series.std()
        result3 = sum(dict_data.values())
        
        # 一些循环
        results = []
        for i in range(10):
            results.append(i * 2)
        
        final_result = [result1, result2, result3] + results[:5]
        return final_result
    
    test_args = [[20170101, "000001"]]
    
    with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as tmp_file:
        backup_file = tmp_file.name
    
    try:
        import rust_pyfunc
        
        print("开始中等函数测试...")
        results = rust_pyfunc.run_pools(
            medium_func,
            test_args,
            backup_file=backup_file,
            num_threads=1
        )
        
        print(f"✅ 中等函数测试成功，结果数量: {len(results)}")
        return True
        
    except Exception as e:
        print(f"❌ 中等函数测试失败: {e}")
        return False
        
    finally:
        if os.path.exists(backup_file):
            os.unlink(backup_file)

if __name__ == "__main__":
    print("开始测试函数大小对子进程的影响...")
    
    test1_ok = test_small_function()
    if test1_ok:
        test2_ok = test_medium_function()
    else:
        test2_ok = False
    
    print(f"\n=== 测试结果总结 ===")
    print(f"小函数测试: {'✅' if test1_ok else '❌'}")
    print(f"中等函数测试: {'✅' if test2_ok else '❌'}")
    
    if not test1_ok:
        print("\n⚠️ 连小函数都无法执行，问题很基础！")
        print("可能的原因：")
        print("1. 子进程启动失败")
        print("2. 函数序列化失败")
        print("3. 通信管道问题")
    elif test1_ok and not test2_ok:
        print("\n⚠️ 问题可能与函数复杂度相关")
    else:
        print("\n✅ 函数大小不是问题，需要查看其他差异")