#!/usr/bin/env python3
"""
最小化测试来定位问题
"""

import os
import sys
import tempfile

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

def test_minimal():
    """最小化测试"""
    print("=== 最小化测试 ===")
    
    def minimal_func(date, code):
        print(f"子进程开始执行: date={date}, code={code}")
        import pure_ocean_breeze.jason as p
        print("pure_ocean_breeze导入成功")
        return [1.0, 2.0, 3.0]
    
    test_args = [[20240101, "TEST001"]]
    
    with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as tmp_file:
        backup_file = tmp_file.name
    
    try:
        import rust_pyfunc
        
        print("开始最小化测试...")
        results = rust_pyfunc.run_pools(
            minimal_func,
            test_args,
            backup_file=backup_file,
            num_threads=1  # 只用1个进程
        )
        
        print(f"✅ 最小化测试成功，结果数量: {len(results)}")
        return True
        
    except Exception as e:
        print(f"❌ 最小化测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        if os.path.exists(backup_file):
            os.unlink(backup_file)

if __name__ == "__main__":
    test_minimal()