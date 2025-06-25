#!/usr/bin/env python3
"""
简单的NaN测试
"""

import os
import sys
import tempfile
import numpy as np
import math

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

def test_simple_nan():
    """简单的NaN测试"""
    print("=== 简单NaN测试 ===")
    
    def simple_nan_func(date, code):
        import numpy as np
        # 返回简单的NaN值
        return [1.0, np.nan, 2.0]
    
    test_args = [[20240101, "TEST001"]]
    
    with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as tmp_file:
        backup_file = tmp_file.name
    
    try:
        import rust_pyfunc
        
        print("开始简单NaN测试...")
        results = rust_pyfunc.run_pools(
            simple_nan_func,
            test_args,
            backup_file=backup_file,
            num_threads=1
        )
        
        if len(results) > 0:
            result = results[0]
            values = result[2:]  # 跳过date和code
            print(f"返回的值: {values}")
            
            # 检查第二个值是否为NaN
            if len(values) >= 2 and math.isnan(values[1]):
                print("✅ NaN值正确保留！")
                print(f"第一个值: {values[0]} (期望: 1.0)")
                print(f"第二个值: NaN (正确!)")
                print(f"第三个值: {values[2]} (期望: 2.0)")
                return True
            else:
                print(f"❌ NaN值处理有问题: {values}")
                return False
        else:
            print("❌ 没有返回结果")
            return False
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        if os.path.exists(backup_file):
            os.unlink(backup_file)

if __name__ == "__main__":
    print("开始简单NaN测试...")
    success = test_simple_nan()
    
    if success:
        print("\n🎉 NaN处理修改成功！现在Python中可以正确获得np.nan值")
    else:
        print("\n⚠️ NaN处理仍有问题，需要进一步调试")