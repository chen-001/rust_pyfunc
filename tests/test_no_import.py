#!/usr/bin/env python3
"""
测试不导入pure_ocean_breeze的情况
"""

import os
import sys
import tempfile

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

def test_no_pure_ocean():
    """测试不导入pure_ocean_breeze"""
    print("=== 测试不导入pure_ocean_breeze ===")
    
    def no_import_func(date, code):
        import numpy as np
        return [1.0, 2.0, 3.0]
    
    test_args = [[20240101, "TEST001"]]
    
    with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as tmp_file:
        backup_file = tmp_file.name
    
    try:
        import rust_pyfunc
        
        print("开始测试（不导入pure_ocean_breeze）...")
        results = rust_pyfunc.run_pools(
            no_import_func,
            test_args,
            backup_file=backup_file,
            num_threads=1
        )
        
        print(f"✅ 测试成功，结果数量: {len(results)}")
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        if os.path.exists(backup_file):
            os.unlink(backup_file)

if __name__ == "__main__":
    test_no_pure_ocean()