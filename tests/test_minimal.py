#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import tempfile
import os
sys.path.append('/home/chenzongwei/rust_pyfunc/python')

import rust_pyfunc

def simple_func(date, code):
    return [1.0, 2.0]

print("开始最小测试...")

# 不使用备份文件的简单测试
try:
    args = [(20220101, "000001")]
    result = rust_pyfunc.run_pools(simple_func, args, num_threads=1)
    print(f"✓ 无备份测试成功，结果: {result}")
except Exception as e:
    print(f"❌ 无备份测试失败: {e}")
    import traceback
    traceback.print_exc()

print("测试完成")