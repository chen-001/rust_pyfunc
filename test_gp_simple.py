#!/usr/bin/env python3
"""
测试GP相关维度算法的简单功能验证
"""

import sys
sys.path.insert(0, '/home/chenzongwei/.conda/envs/chenzongwei311/lib/python3.11/site-packages')

import numpy as np

def test_gp_function_imports():
    """测试GP函数是否可以导入"""
    try:
        import rust_pyfunc
        print("✓ rust_pyfunc 导入成功")

        # 检查GP相关函数
        gp_functions = [f for f in dir(rust_pyfunc) if 'gp' in f.lower()]
        print(f"GP相关函数: {gp_functions}")

        # 检查所有函数
        all_functions = [f for f in dir(rust_pyfunc) if not f.startswith('_')]
        print(f"总函数数量: {len(all_functions)}")

        # 查找correlation_dimension相关函数
        cd_functions = [f for f in all_functions if 'correlation_dimension' in f]
        print(f"相关维度函数: {cd_functions}")

        return True
    except Exception as e:
        print(f"✗ 导入失败: {e}")
        return False

def test_simple_gp_calculation():
    """测试简单的GP计算"""
    try:
        import rust_pyfunc

        # 生成简单的测试数据 - 逻辑斯蒂映射
        def logistic_map(x0, r, n):
            x = np.zeros(n)
            x[0] = x0
            for i in range(1, n):
                x[i] = r * x[i-1] * (1 - x[i-1])
            return x

        # 生成测试数据
        data = logistic_map(0.5, 3.8, 500)

        # 尝试使用GP函数
        if hasattr(rust_pyfunc, 'gp_correlation_dimension_auto'):
            print("✓ 找到 gp_correlation_dimension_auto 函数")
            result = rust_pyfunc.gp_correlation_dimension_auto(data)
            print(f"GP计算成功! 结果类型: {type(result)}")
            print(f"可用属性: {[attr for attr in dir(result) if not attr.startswith('_')]}")

            # 尝试访问各种可能的属性
            try:
                print(f"结果详情:")
                for attr in ['tau', 'm', 'theiler', 'd2_est', 'fit_r2', 'fit_start', 'fit_end']:
                    if hasattr(result, attr):
                        value = getattr(result, attr)
                        print(f"  {attr}: {value}")
                return True
            except Exception as e:
                print(f"访问属性时出错: {e}")
                return False
        else:
            print("✗ 未找到 gp_correlation_dimension_auto 函数")
            return False

    except Exception as e:
        print(f"✗ GP计算失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=== GP相关维度算法测试 ===")

    print("\n1. 测试函数导入...")
    import_ok = test_gp_function_imports()

    if import_ok:
        print("\n2. 测试GP计算...")
        calc_ok = test_simple_gp_calculation()

        if calc_ok:
            print("\n✓ 所有测试通过!")
        else:
            print("\n✗ 计算测试失败")
    else:
        print("\n✗ 导入测试失败")