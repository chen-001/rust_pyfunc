#!/usr/bin/env python3
"""
测试异常处理对子进程的影响
"""

import os
import sys
import tempfile

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

def test_exception_with_raise():
    """测试重新抛出异常"""
    print("=== 测试重新抛出异常 ===")
    
    def exception_raise_func(date, code):
        import numpy as np
        try:
            # 模拟可能出错的操作
            if code == "000001":  # 故意让这个测试出错
                non_existent_file = open("/non_existent_path/file.csv", "r")
            return [1.0, 2.0, 3.0]
        except Exception as e:
            # 像万里长征那样重新抛出异常
            raise
    
    test_args = [[20170101, "000001"]]
    
    with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as tmp_file:
        backup_file = tmp_file.name
    
    try:
        import rust_pyfunc
        
        print("开始异常重抛测试...")
        results = rust_pyfunc.run_pools(
            exception_raise_func,
            test_args,
            backup_file=backup_file,
            num_threads=1
        )
        
        print(f"结果: {results}")
        return True
        
    except Exception as e:
        print(f"❌ 异常重抛测试失败: {e}")
        return False
        
    finally:
        if os.path.exists(backup_file):
            os.unlink(backup_file)

def test_exception_with_return():
    """测试返回默认值而不抛出异常"""
    print("\n=== 测试返回默认值 ===")
    
    def exception_return_func(date, code):
        import numpy as np
        try:
            # 模拟可能出错的操作
            if code == "000001":  # 故意让这个测试出错
                non_existent_file = open("/non_existent_path/file.csv", "r")
            return [1.0, 2.0, 3.0]
        except Exception as e:
            # 返回默认值而不是重新抛出异常
            return [np.nan] * 10
    
    test_args = [[20170101, "000001"]]
    
    with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as tmp_file:
        backup_file = tmp_file.name
    
    try:
        import rust_pyfunc
        
        print("开始异常返回测试...")
        results = rust_pyfunc.run_pools(
            exception_return_func,
            test_args,
            backup_file=backup_file,
            num_threads=1
        )
        
        print(f"✅ 异常返回测试成功，结果数量: {len(results)}")
        if len(results) > 0:
            print(f"结果: {results[0][2:5]}")  # 显示前几个结果
        return True
        
    except Exception as e:
        print(f"❌ 异常返回测试失败: {e}")
        return False
        
    finally:
        if os.path.exists(backup_file):
            os.unlink(backup_file)

def test_successful_case():
    """测试成功情况"""
    print("\n=== 测试成功情况 ===")
    
    def success_func(date, code):
        import numpy as np
        try:
            # 成功的操作
            return [1.0, 2.0, 3.0]
        except Exception as e:
            raise
    
    test_args = [[20170101, "999999"]]  # 使用不同的code
    
    with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as tmp_file:
        backup_file = tmp_file.name
    
    try:
        import rust_pyfunc
        
        print("开始成功情况测试...")
        results = rust_pyfunc.run_pools(
            success_func,
            test_args,
            backup_file=backup_file,
            num_threads=1
        )
        
        print(f"✅ 成功情况测试通过，结果数量: {len(results)}")
        return True
        
    except Exception as e:
        print(f"❌ 成功情况测试失败: {e}")
        return False
        
    finally:
        if os.path.exists(backup_file):
            os.unlink(backup_file)

if __name__ == "__main__":
    print("开始测试异常处理对子进程的影响...")
    
    test1_ok = test_successful_case()
    test2_ok = test_exception_with_return()
    test3_ok = test_exception_with_raise()
    
    print(f"\n=== 测试结果总结 ===")
    print(f"成功情况: {'✅' if test1_ok else '❌'}")
    print(f"异常返回默认值: {'✅' if test2_ok else '❌'}")
    print(f"异常重新抛出: {'✅' if test3_ok else '❌'}")
    
    if test1_ok and test2_ok and not test3_ok:
        print("\n🎯 找到问题！重新抛出异常导致子进程崩溃！")
        print("解决方案：修改万里长征文件，让异常处理返回默认值而不是重新抛出")
    elif not test1_ok:
        print("\n⚠️ 基础功能有问题")
    else:
        print("\n🤔 异常处理不是主要问题，需要进一步分析")