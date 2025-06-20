#!/usr/bin/env python3
"""
测试最简单的函数，验证构建是否成功
"""

print("开始测试简单函数...")

try:
    import rust_pyfunc
    print("✓ rust_pyfunc模块导入成功")
    
    # 测试简单函数
    if hasattr(rust_pyfunc, 'test_simple_function'):
        print("✓ test_simple_function函数已找到")
        
        result = rust_pyfunc.test_simple_function()
        print(f"函数返回值: {result}")
        
        if result == 42:
            print("✓ 函数返回值正确")
        else:
            print(f"✗ 函数返回值错误，期望42，实际{result}")
            
    else:
        print("✗ test_simple_function函数未找到")
        available_funcs = [f for f in dir(rust_pyfunc) if not f.startswith('__')]
        print(f"可用函数: {available_funcs[:10]}")
        
    # 同时测试factor_grouping函数
    if hasattr(rust_pyfunc, 'factor_grouping'):
        print("✓ factor_grouping函数也存在")
    else:
        print("✗ factor_grouping函数未找到")
        
except ImportError as e:
    print(f"✗ 导入失败: {e}")
except Exception as e:
    print(f"✗ 测试失败: {e}")
    import traceback
    traceback.print_exc()

print("简单函数测试完成!")