#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试Broken pipe修复
"""

import sys
sys.path.append('/home/chenzongwei/rust_pyfunc/python')

import rust_pyfunc
import time


def test_function(date, code):
    """测试函数"""
    result = 0
    for i in range(100):
        result += hash(f"{date}_{code}_{i}") % 100
    
    return [
        float(date % 10000),
        float(len(code)),
        float(result % 1000),
        float((date + len(code)) % 500)
    ]


def test_multiprocess_stability():
    """测试多进程稳定性"""
    print("🔍 测试多进程稳定性...")
    
    # 创建足够多的任务来触发问题
    args = []
    for i in range(1000):
        args.append([20220101 + i % 100, f"{i+1:06d}"])
    
    print(f"测试数据: {len(args)} 个任务")
    
    try:
        start_time = time.time()
        result = rust_pyfunc.run_multiprocess(
            test_function,
            args,
            num_processes=8,
        )
        elapsed = time.time() - start_time
        
        print(f"✅ 执行成功!")
        print(f"结果数量: {len(result)}")
        print(f"耗时: {elapsed:.2f}秒")
        print(f"速度: {len(args)/elapsed:.0f} 任务/秒")
        
        return True
        
    except Exception as e:
        print(f"❌ 执行失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_run_pools_stability():
    """测试run_pools稳定性"""
    print(f"\n🔍 测试run_pools稳定性...")
    
    args = []
    for i in range(500):
        args.append([20220101 + i % 50, f"{i+1:06d}"])
    
    print(f"测试数据: {len(args)} 个任务")
    
    try:
        start_time = time.time()
        result = rust_pyfunc.run_pools(
            test_function,
            args,
            num_threads=6,
        )
        elapsed = time.time() - start_time
        
        print(f"✅ 执行成功!")
        print(f"结果数量: {len(result)}")
        print(f"耗时: {elapsed:.2f}秒")
        print(f"速度: {len(args)/elapsed:.0f} 任务/秒")
        
        return True
        
    except Exception as e:
        print(f"❌ 执行失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success1 = test_multiprocess_stability()
    success2 = test_run_pools_stability()
    
    if success1 and success2:
        print(f"\n🎉 所有测试通过! Broken pipe问题已修复")
    else:
        print(f"\n❌ 仍有问题需要解决")