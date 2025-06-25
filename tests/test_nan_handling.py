#!/usr/bin/env python3
"""
测试NaN/Inf值的处理
"""

import os
import sys
import tempfile
import numpy as np
import math

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

def test_nan_preservation():
    """测试NaN值是否被正确保留"""
    print("=== 测试NaN值保留 ===")
    
    def nan_func(date, code):
        import numpy as np
        import math
        
        # 返回包含各种特殊值的列表
        return [
            1.0,                # 正常值
            np.nan,             # numpy NaN
            float('nan'),       # python NaN  
            np.inf,             # 正无穷 -> 应转为NaN
            -np.inf,            # 负无穷 -> 应转为NaN
            2.5,                # 正常值
            0.0,                # 零值
            -1.5                # 负值
        ]
    
    test_args = [[20240101, "TEST001"]]
    
    with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as tmp_file:
        backup_file = tmp_file.name
    
    try:
        import rust_pyfunc
        
        print("开始NaN值保留测试...")
        results = rust_pyfunc.run_pools(
            nan_func,
            test_args,
            backup_file=backup_file,
            num_threads=1
        )
        
        if len(results) > 0:
            result = results[0]
            print(f"结果数量: {len(result)}")
            
            # 检查结果中的值
            values = result[2:]  # 跳过date和code
            print(f"返回的值: {values}")
            
            # 验证预期结果
            expected_pattern = [
                lambda x: x == 1.0,                    # 正常值
                lambda x: math.isnan(x),               # NaN
                lambda x: math.isnan(x),               # NaN
                lambda x: math.isnan(x),               # inf -> NaN
                lambda x: math.isnan(x),               # -inf -> NaN
                lambda x: x == 2.5,                    # 正常值
                lambda x: x == 0.0,                    # 零值
                lambda x: x == -1.5                    # 负值
            ]
            
            success = True
            for i, (value, check) in enumerate(zip(values, expected_pattern)):
                if not check(value):
                    print(f"❌ 第{i}个值不符合预期: {value}")
                    success = False
                else:
                    if math.isnan(value):
                        print(f"✅ 第{i}个值正确为NaN")
                    else:
                        print(f"✅ 第{i}个值正确: {value}")
            
            if success:
                print("✅ NaN值保留测试成功")
                return True
            else:
                print("❌ NaN值保留测试失败")
                return False
        else:
            print("❌ 没有返回结果")
            return False
        
    except Exception as e:
        print(f"❌ NaN值保留测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        if os.path.exists(backup_file):
            os.unlink(backup_file)

def test_large_array_with_nans():
    """测试包含大量NaN的数组"""
    print("\n=== 测试大量NaN数组 ===")
    
    def large_nan_func(date, code):
        import numpy as np
        
        # 创建包含大量NaN的数组
        data = np.random.randn(100)
        data[::5] = np.nan  # 每5个元素设置一个NaN
        
        # 返回前20个元素
        return data[:20].tolist()
    
    test_args = [[20240101, "TEST001"]]
    
    with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as tmp_file:
        backup_file = tmp_file.name
    
    try:
        import rust_pyfunc
        
        print("开始大量NaN数组测试...")
        results = rust_pyfunc.run_pools(
            large_nan_func,
            test_args,
            backup_file=backup_file,
            num_threads=1
        )
        
        if len(results) > 0:
            result = results[0]
            values = result[2:]  # 跳过date和code
            
            nan_count = sum(1 for x in values if math.isnan(x))
            finite_count = sum(1 for x in values if math.isfinite(x))
            
            print(f"数组长度: {len(values)}")
            print(f"NaN数量: {nan_count}")
            print(f"有限值数量: {finite_count}")
            
            # 验证每5个元素中有一个NaN
            expected_nan_count = len(values) // 5
            if abs(nan_count - expected_nan_count) <= 1:  # 允许1个误差
                print("✅ 大量NaN数组测试成功")
                return True
            else:
                print(f"❌ NaN数量不符合预期，期望约{expected_nan_count}个，实际{nan_count}个")
                return False
        else:
            print("❌ 没有返回结果")
            return False
        
    except Exception as e:
        print(f"❌ 大量NaN数组测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        if os.path.exists(backup_file):
            os.unlink(backup_file)

if __name__ == "__main__":
    print("开始测试NaN/Inf值处理...")
    
    test1_ok = test_nan_preservation()
    test2_ok = test_large_array_with_nans()
    
    print(f"\n=== 测试结果总结 ===")
    print(f"NaN值保留: {'✅' if test1_ok else '❌'}")
    print(f"大量NaN数组: {'✅' if test2_ok else '❌'}")
    
    if test1_ok and test2_ok:
        print("\n🎉 所有NaN处理测试通过！现在Python中可以正确获得np.nan值")
    else:
        print("\n⚠️ 某些测试失败，需要进一步调试")