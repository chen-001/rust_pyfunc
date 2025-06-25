#!/usr/bin/env python3
"""
测试复杂函数的问题
"""

import os
import sys
import tempfile
import numpy as np

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

def test_complex_imports():
    """测试1：复杂导入（模拟万里长征的导入）"""
    print("=== 测试1：复杂导入 ===")
    
    def complex_import_func(date, code):
        import pure_ocean_breeze.jason as p
        import rust_pyfunc as rp
        import numpy as np
        import pandas as pd
        
        # 简单返回，只测试导入
        return [1.0, 2.0, 3.0]
    
    test_args = [[20240101, "TEST001"]]
    
    with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as tmp_file:
        backup_file = tmp_file.name
    
    try:
        import rust_pyfunc
        
        print("开始复杂导入测试...")
        results = rust_pyfunc.run_pools(
            complex_import_func,
            test_args,
            backup_file=backup_file,
            num_threads=2
        )
        
        print(f"✅ 复杂导入测试成功，结果数量: {len(results)}")
        return True
        
    except Exception as e:
        print(f"❌ 复杂导入测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        if os.path.exists(backup_file):
            os.unlink(backup_file)

def test_pandas_operations():
    """测试2：pandas操作"""
    print("\n=== 测试2：pandas操作 ===")
    
    def pandas_func(date, code):
        import numpy as np
        import pandas as pd
        
        # 模拟一些pandas操作
        data = {'values': np.random.randn(10)}
        series = pd.Series(data['values'])
        
        mean_val = series.mean()
        std_val = series.std()
        skew_val = series.skew()
        kurt_val = series.kurt()
        
        return [mean_val, std_val, skew_val, kurt_val]
    
    test_args = [[20240101, "TEST001"]]
    
    with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as tmp_file:
        backup_file = tmp_file.name
    
    try:
        import rust_pyfunc
        
        print("开始pandas操作测试...")
        results = rust_pyfunc.run_pools(
            pandas_func,
            test_args,
            backup_file=backup_file,
            num_threads=2
        )
        
        print(f"✅ pandas操作测试成功，结果数量: {len(results)}")
        return True
        
    except Exception as e:
        print(f"❌ pandas操作测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        if os.path.exists(backup_file):
            os.unlink(backup_file)

def test_large_function():
    """测试3：大型复杂函数（模拟万里长征的复杂度）"""
    print("\n=== 测试3：大型复杂函数 ===")
    
    def large_complex_func(date, code):
        import numpy as np
        import pandas as pd
        
        # 模拟复杂计算
        data1 = np.random.randn(100)
        data2 = np.random.randn(100)
        
        # 大量相关系数计算
        correlations = []
        for i in range(10):
            for j in range(i+1, 10):
                subset1 = data1[i*10:(i+1)*10]
                subset2 = data2[j*10:(j+1)*10]
                if len(subset1) > 1 and len(subset2) > 1:
                    corr = np.corrcoef(subset1, subset2)[0, 1]
                    if not np.isnan(corr):
                        correlations.append(corr)
        
        # pandas统计计算
        results = []
        for i, corr_data in enumerate([data1, data2]):
            series = pd.Series(corr_data)
            results.extend([
                series.mean(),
                series.std(), 
                series.skew(),
                series.kurt(),
                series.max(),
                series.min()
            ])
        
        # 添加相关系数
        results.extend(correlations[:10])  # 最多10个
        
        # 确保返回固定长度
        while len(results) < 50:
            results.append(0.0)
            
        return results[:50]  # 返回50个值
    
    test_args = [[20240101, "TEST001"]]
    
    with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as tmp_file:
        backup_file = tmp_file.name
    
    try:
        import rust_pyfunc
        
        print("开始大型复杂函数测试...")
        results = rust_pyfunc.run_pools(
            large_complex_func,
            test_args,
            backup_file=backup_file,
            num_threads=2
        )
        
        print(f"✅ 大型复杂函数测试成功，结果数量: {len(results)}")
        print(f"第一个结果长度: {len(results[0])-2} 个因子")  # -2是date和code
        return True
        
    except Exception as e:
        print(f"❌ 大型复杂函数测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        if os.path.exists(backup_file):
            os.unlink(backup_file)

def test_function_source_extraction():
    """测试4：函数源代码提取"""
    print("\n=== 测试4：函数源代码提取 ===")
    
    def test_func(date, code):
        # 这是一个复杂的函数，用来测试源代码提取
        import numpy as np
        import pandas as pd
        
        # 多行计算
        data = np.random.randn(100)
        series = pd.Series(data)
        
        result = [
            series.mean(),
            series.std(),
            series.skew()
        ]
        
        return result
    
    try:
        import inspect
        source = inspect.getsource(test_func)
        print(f"✅ 源代码提取成功，长度: {len(source)} 字符")
        print(f"源代码前100字符: {source[:100]}...")
        return True
        
    except Exception as e:
        print(f"❌ 源代码提取失败: {e}")
        return False

if __name__ == "__main__":
    print("开始调试复杂函数问题...")
    
    # 按顺序执行测试
    test1_ok = test_complex_imports()
    test2_ok = test_pandas_operations()
    test3_ok = test_large_function()
    test4_ok = test_function_source_extraction()
    
    print(f"\n=== 测试结果总结 ===")
    print(f"复杂导入: {'✅' if test1_ok else '❌'}")
    print(f"pandas操作: {'✅' if test2_ok else '❌'}")  
    print(f"大型函数: {'✅' if test3_ok else '❌'}")
    print(f"源代码提取: {'✅' if test4_ok else '❌'}")
    
    if not test1_ok:
        print("\n⚠️ 问题出在复杂导入上")
    elif not test2_ok:
        print("\n⚠️ 问题出在pandas操作上")
    elif not test3_ok:
        print("\n⚠️ 问题出在大型复杂函数上")
    elif not test4_ok:
        print("\n⚠️ 问题出在函数源代码提取上")
    else:
        print("\n🤔 所有基础测试都通过，问题可能更复杂")