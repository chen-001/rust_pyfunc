#!/usr/bin/env python3
"""
逐步测试找出问题
"""

import os
import sys
import tempfile
import time

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

def test_step1_basic():
    """步骤1：最基础测试"""
    print("=== 步骤1：最基础测试 ===")
    
    def simple_func(date, code):
        return [1.0, 2.0, 3.0]
    
    test_args = [[20170101, "000001"]]
    
    with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as tmp_file:
        backup_file = tmp_file.name
    
    try:
        import rust_pyfunc
        
        print("运行基础测试...")
        results = rust_pyfunc.run_pools(
            simple_func,
            test_args,
            backup_file=backup_file,
            num_threads=1
        )
        
        print(f"✅ 基础测试成功: {len(results)} 结果")
        return True
        
    except Exception as e:
        print(f"❌ 基础测试失败: {e}")
        return False
        
    finally:
        if os.path.exists(backup_file):
            os.unlink(backup_file)

def test_step2_multiple_threads():
    """步骤2：多线程测试"""
    print("\n=== 步骤2：多线程测试 ===")
    
    def simple_func(date, code):
        import time
        time.sleep(0.1)  # 模拟一些处理时间
        return [1.0, 2.0, 3.0]
    
    test_args = [[20170101, f"00000{i}"] for i in range(5)]
    
    with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as tmp_file:
        backup_file = tmp_file.name
    
    try:
        import rust_pyfunc
        
        print("运行多线程测试...")
        start_time = time.time()
        
        results = rust_pyfunc.run_pools(
            simple_func,
            test_args,
            backup_file=backup_file,
            num_threads=5
        )
        
        end_time = time.time()
        
        print(f"✅ 多线程测试成功: {len(results)} 结果, 耗时 {end_time-start_time:.2f}秒")
        return True
        
    except Exception as e:
        print(f"❌ 多线程测试失败: {e}")
        return False
        
    finally:
        if os.path.exists(backup_file):
            os.unlink(backup_file)

def test_step3_many_threads():
    """步骤3：大量线程测试"""
    print("\n=== 步骤3：大量线程测试 ===")
    
    def simple_func(date, code):
        import time
        time.sleep(0.05)  # 减少处理时间
        return [float(code[-1]), 2.0, 3.0]  # 返回一些可区分的值
    
    test_args = [[20170101, f"00000{i}"] for i in range(10)]
    
    with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as tmp_file:
        backup_file = tmp_file.name
    
    try:
        import rust_pyfunc
        
        print("运行大量线程测试...")
        start_time = time.time()
        
        results = rust_pyfunc.run_pools(
            simple_func,
            test_args,
            backup_file=backup_file,
            num_threads=50  # 使用50个线程
        )
        
        end_time = time.time()
        
        print(f"✅ 大量线程测试成功: {len(results)} 结果, 耗时 {end_time-start_time:.2f}秒")
        return True
        
    except Exception as e:
        print(f"❌ 大量线程测试失败: {e}")
        return False
        
    finally:
        if os.path.exists(backup_file):
            os.unlink(backup_file)

def test_step4_exceptions():
    """步骤4：异常处理测试"""
    print("\n=== 步骤4：异常处理测试 ===")
    
    def exception_func(date, code):
        if code == "000001":
            raise Exception("模拟异常")
        return [1.0, 2.0, 3.0]
    
    test_args = [[20170101, "000001"], [20170101, "000002"]]
    
    with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as tmp_file:
        backup_file = tmp_file.name
    
    try:
        import rust_pyfunc
        
        print("运行异常处理测试...")
        results = rust_pyfunc.run_pools(
            exception_func,
            test_args,
            backup_file=backup_file,
            num_threads=2
        )
        
        print(f"✅ 异常处理测试成功: {len(results)} 结果")
        return True
        
    except Exception as e:
        print(f"❌ 异常处理测试失败: {e}")
        return False
        
    finally:
        if os.path.exists(backup_file):
            os.unlink(backup_file)

if __name__ == "__main__":
    print("开始逐步测试...")
    
    # 逐步进行测试
    step1_ok = test_step1_basic()
    
    if step1_ok:
        step2_ok = test_step2_multiple_threads()
    else:
        print("步骤1失败，停止测试")
        sys.exit(1)
    
    if step2_ok:
        step3_ok = test_step3_many_threads()
    else:
        print("步骤2失败，跳过后续测试")
        step3_ok = False
        step4_ok = False
    
    if step3_ok:
        step4_ok = test_step4_exceptions()
    else:
        step4_ok = False
    
    print(f"\n=== 逐步测试结果 ===")
    print(f"步骤1 - 基础测试: {'✅' if step1_ok else '❌'}")
    print(f"步骤2 - 多线程测试: {'✅' if step2_ok else '❌'}")
    print(f"步骤3 - 大量线程测试: {'✅' if step3_ok else '❌'}")
    print(f"步骤4 - 异常处理测试: {'✅' if step4_ok else '❌'}")
    
    if step1_ok and step2_ok and not step3_ok:
        print("\n⚠️ 问题出现在大量线程时！")
        print("这可能是万里长征脚本卡住的根本原因")
    elif step1_ok and not step2_ok:
        print("\n⚠️ 问题出现在基本多线程时！")
    else:
        print(f"\n{'✅ 所有测试通过' if all([step1_ok, step2_ok, step3_ok, step4_ok]) else '⚠️ 存在问题'}")