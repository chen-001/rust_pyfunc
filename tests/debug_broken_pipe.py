#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调试Broken pipe问题
"""

import sys
sys.path.append('/home/chenzongwei/rust_pyfunc/python')

import rust_pyfunc


# 使用全局函数，确保可以被inspect.getsource获取
def global_test_function(date, code):
    """全局定义的测试函数"""
    result = 0
    for i in range(10):  # 减少计算量
        result += hash(f"{date}_{code}_{i}") % 100
    
    return [
        float(date % 10000),
        float(len(code)),
        float(result % 1000)
    ]


def test_minimal_case():
    """测试最小化用例"""
    print("🔍 测试最小化用例")
    
    # 只有2个任务
    args = [
        [20220101, "000001"],
        [20220102, "000002"],
    ]
    
    print(f"测试数据: {args}")
    
    try:
        print(f"🚀 开始执行...")
        result = rust_pyfunc.run_multiprocess(
            global_test_function,
            args,
            num_processes=1,  # 只用1个进程
        )
        
        print(f"✅ 执行成功!")
        print(f"结果: {result}")
        
    except Exception as e:
        print(f"❌ 执行失败: {e}")
        import traceback
        traceback.print_exc()


def test_function_inspection():
    """测试函数检查"""
    print(f"\n🔍 测试函数检查")
    
    # 测试inspect.getsource
    try:
        import inspect
        source = inspect.getsource(global_test_function)
        print(f"✅ 可以获取函数源代码:")
        print(f"长度: {len(source)} 字符")
        print(f"前100字符: {source[:100]}...")
    except Exception as e:
        print(f"❌ 无法获取函数源代码: {e}")


if __name__ == "__main__":
    test_function_inspection()
    test_minimal_case()