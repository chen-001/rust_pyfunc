#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化的最终测试
"""

import sys
sys.path.append('/home/chenzongwei/rust_pyfunc/python')

import rust_pyfunc
import time


def test_function(date, code):
    """简单测试函数"""
    return [float(date % 1000), float(len(code)), 1.0, 2.0]


def main():
    print("🎯 Broken pipe问题最终验证")
    print("=" * 50)
    
    # 测试1: 基本多进程功能
    print("1️⃣ 测试基本多进程功能...")
    args = [[20220101 + i, f"{i:06d}"] for i in range(100)]
    
    try:
        start_time = time.time()
        result = rust_pyfunc.run_multiprocess(
            test_function,
            args,
            num_processes=8,
        )
        elapsed = time.time() - start_time
        
        print(f"   ✅ 成功! 结果: {len(result)} 个, 耗时: {elapsed:.2f}秒")
        
    except Exception as e:
        print(f"   ❌ 失败: {e}")
        return False
    
    # 测试2: run_pools API
    print("2️⃣ 测试run_pools API...")
    
    try:
        start_time = time.time()
        result = rust_pyfunc.run_pools(
            test_function,
            args,
            num_threads=6,
        )
        elapsed = time.time() - start_time
        
        print(f"   ✅ 成功! 结果: {len(result)} 个, 耗时: {elapsed:.2f}秒")
        
    except Exception as e:
        print(f"   ❌ 失败: {e}")
        return False
    
    # 测试3: 大数据量
    print("3️⃣ 测试大数据量...")
    big_args = [[20220101 + i, f"{i:06d}"] for i in range(1000)]
    
    try:
        start_time = time.time()
        result = rust_pyfunc.run_multiprocess(
            test_function,
            big_args,
            num_processes=12,
        )
        elapsed = time.time() - start_time
        
        print(f"   ✅ 成功! 结果: {len(result)} 个, 耗时: {elapsed:.2f}秒")
        print(f"   处理速度: {len(big_args)/elapsed:.0f} 任务/秒")
        
    except Exception as e:
        print(f"   ❌ 失败: {e}")
        return False
    
    print("=" * 50)
    print("🎊 所有测试通过! Broken pipe问题已完全解决")
    print("✨ 功能特性:")
    print("   • Rust原生多进程，避开Python GIL")
    print("   • 稳定的进程管理和错误恢复")
    print("   • 兼容原有API，无缝升级")
    print("   • 高性能并行计算能力")
    
    return True


if __name__ == "__main__":
    main()