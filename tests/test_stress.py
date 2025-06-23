#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
压力测试：模拟可能导致broken pipe的场景
"""

import sys
sys.path.append('/home/chenzongwei/rust_pyfunc/python')

import rust_pyfunc
import time


def complex_function(date, code):
    """复杂计算函数，可能导致进程问题"""
    import random
    import time
    
    # 随机一些计算复杂度
    computation_time = random.uniform(0.001, 0.01)
    time.sleep(computation_time)
    
    result = 0
    for i in range(1000):
        result += hash(f"{date}_{code}_{i}") % 100
    
    # 模拟一些可能出错的情况
    if date % 1000 == 999:  # 千分之一的概率
        # 模拟一个轻微的错误情况但不抛异常
        result = result * 1.1
    
    return [
        float(date % 10000),
        float(len(code)),
        float(result % 1000),
        float((date + len(code)) % 500)
    ]


def test_large_dataset():
    """测试大数据集"""
    print("🔍 压力测试：大数据集...")
    
    # 创建大量任务
    args = []
    for i in range(5000):  # 5000个任务
        args.append([20220101 + i % 365, f"{i+1:06d}"])
    
    print(f"测试数据: {len(args)} 个任务")
    
    try:
        start_time = time.time()
        result = rust_pyfunc.run_multiprocess(
            complex_function,
            args,
            num_processes=20,  # 更多进程
        )
        elapsed = time.time() - start_time
        
        print(f"✅ 大数据集测试成功!")
        print(f"结果数量: {len(result)}")
        print(f"耗时: {elapsed:.2f}秒")
        print(f"速度: {len(args)/elapsed:.0f} 任务/秒")
        
        return True
        
    except Exception as e:
        print(f"❌ 大数据集测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_high_concurrency():
    """测试高并发"""
    print(f"\n🔍 压力测试：高并发...")
    
    args = []
    for i in range(2000):
        args.append([20220101 + i % 100, f"{i+1:06d}"])
    
    print(f"测试数据: {len(args)} 个任务")
    
    try:
        start_time = time.time()
        result = rust_pyfunc.run_pools(
            complex_function,
            args,
            num_threads=50,  # 极高并发
        )
        elapsed = time.time() - start_time
        
        print(f"✅ 高并发测试成功!")
        print(f"结果数量: {len(result)}")
        print(f"耗时: {elapsed:.2f}秒")
        print(f"速度: {len(args)/elapsed:.0f} 任务/秒")
        
        return True
        
    except Exception as e:
        print(f"❌ 高并发测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("🧪 开始压力测试...")
    print("=" * 60)
    
    success1 = test_large_dataset()
    success2 = test_high_concurrency()
    
    print(f"\n" + "=" * 60)
    if success1 and success2:
        print(f"🎉 所有压力测试通过! 系统稳定性良好")
    else:
        print(f"❌ 压力测试中发现问题")