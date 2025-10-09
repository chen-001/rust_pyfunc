#!/usr/bin/env python3
"""
测试GP相关维度算法是否正常工作
"""

import sys
sys.path.insert(0, '/home/chenzongwei/.conda/envs/chenzongwei311/lib/python3.11/site-packages')

import numpy as np

def test_gp_import():
    """测试GP函数导入"""
    try:
        import rust_pyfunc
        print("✓ rust_pyfunc 导入成功")

        # 检查GP相关函数
        gp_functions = [f for f in dir(rust_pyfunc) if 'gp' in f.lower()]
        print(f"GP相关函数: {gp_functions}")

        # 检查具体函数
        required_functions = [
            'gp_correlation_dimension_auto',
            'gp_correlation_dimension',
            'gp_create_default_options',
            'gp_create_options'
        ]

        for func in required_functions:
            if hasattr(rust_pyfunc, func):
                print(f"✓ 找到函数: {func}")
            else:
                print(f"✗ 缺少函数: {func}")

        return True
    except Exception as e:
        print(f"✗ 导入失败: {e}")
        return False

def test_simple_gp_calculation():
    """测试简单的GP计算"""
    try:
        import rust_pyfunc

        # 生成简单的测试数据
        np.random.seed(42)  # 固定种子确保确定性
        data = np.random.randn(200)  # 200个正态分布随机数

        print(f"测试数据长度: {len(data)}")
        print(f"数据范围: [{data.min():.4f}, {data.max():.4f}]")

        # 测试零参数版本
        if hasattr(rust_pyfunc, 'gp_correlation_dimension_auto'):
            print("\n测试零参数版本...")
            result = rust_pyfunc.gp_correlation_dimension_auto(data)
            print(f"✓ 计算成功! 结果类型: {type(result)}")

            # 访问结果属性
            attributes = ['tau', 'm', 'theiler', 'd2_est', 'fit_r2', 'fit_start', 'fit_end']
            for attr in attributes:
                if hasattr(result, attr):
                    value = getattr(result, attr)
                    print(f"  {attr}: {value}")

            return True
        else:
            print("✗ 缺少 gp_correlation_dimension_auto 函数")
            return False

    except Exception as e:
        print(f"✗ GP计算失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_logistic_map():
    """测试逻辑斯蒂映射数据"""
    try:
        import rust_pyfunc

        # 生成逻辑斯蒂映射数据
        def logistic_map(x0, r, n):
            x = np.zeros(n)
            x[0] = x0
            for i in range(1, n):
                x[i] = r * x[i-1] * (1 - x[i-1])
            return x

        data = logistic_map(0.5, 3.8, 500)  # 混沌参数

        print(f"\n逻辑斯蒂映射测试:")
        print(f"数据长度: {len(data)}")
        print(f"数据范围: [{data.min():.4f}, {data.max():.4f}]")

        result = rust_pyfunc.gp_correlation_dimension_auto(data)
        print(f"✓ 逻辑斯蒂映射计算成功!")
        print(f"  D₂估计值: {result.d2_est:.4f}")
        print(f"  τ: {result.tau}, m: {result.m}")
        print(f"  拟合质量: {result.fit_r2:.4f}")

        return True

    except Exception as e:
        print(f"✗ 逻辑斯蒂映射测试失败: {e}")
        return False

if __name__ == "__main__":
    print("=== GP相关维度算法简单测试 ===")

    print("\n1. 测试函数导入...")
    import_ok = test_gp_import()

    if import_ok:
        print("\n2. 测试随机数据计算...")
        random_ok = test_simple_gp_calculation()

        if random_ok:
            print("\n3. 测试逻辑斯蒂映射...")
            logistic_ok = test_logistic_map()

            if logistic_ok:
                print("\n✓ 所有测试通过!")
            else:
                print("\n✗ 逻辑斯蒂映射测试失败")
        else:
            print("\n✗ 随机数据测试失败")
    else:
        print("\n✗ 导入测试失败")