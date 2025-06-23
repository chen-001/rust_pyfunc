#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试反序列化问题的调试脚本
"""

import sys
sys.path.insert(0, '/home/chenzongwei/rust_pyfunc/python')

import rust_pyfunc

def my_test_function(date, code):
    """测试函数"""
    return [float(i + date % 10) for i in range(5)]

def test_multiprocess():
    """测试多进程执行"""
    print("开始测试多进程反序列化...")
    
    # 准备测试数据
    test_args = [[20230101 + i, f"TEST{i:03d}"] for i in range(20)]
    
    try:
        print("调用 rust_pyfunc.run_pools...")
        result = rust_pyfunc.run_pools(
            func=my_test_function,
            args=test_args,
            num_threads=4,
            backup_file=None
        )
        
        print(f"成功！结果形状: {result.shape}")
        print(f"前几行结果: {result[:3]}")
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_multiprocess()