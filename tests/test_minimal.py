#!/usr/bin/env python3
"""
最小化测试
"""

import os
import sys
import tempfile

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

def simple_test():
    """最简单的测试"""
    
    def calc(date, code):
        return [1.0, 2.0, 3.0]
    
    test_args = [[20240101, "TEST001"]]
    
    with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as tmp_file:
        backup_file = tmp_file.name
    
    try:
        import rust_pyfunc
        
        results = rust_pyfunc.run_pools(
            calc,
            test_args,
            backup_file=backup_file,
            num_threads=1,
            backup_batch_size=1,
            storage_format="binary"
        )
        
        print(f"Results: {results}")
        print(f"Backup file size: {os.path.getsize(backup_file)} bytes")
        
        # 查询备份
        backup_results = rust_pyfunc.query_backup(
            backup_file,
            storage_format="binary"
        )
        print(f"Backup results: {backup_results}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        if os.path.exists(backup_file):
            os.unlink(backup_file)

if __name__ == "__main__":
    simple_test()