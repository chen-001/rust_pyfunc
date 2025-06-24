#!/usr/bin/env python3
"""
测试Jupyter中的日志显示
"""

import os
import sys
import tempfile

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

def test_jupyter_logging():
    """测试在Jupyter中显示日志"""
    print("开始测试Jupyter日志显示...")
    
    def simple_func(date, code):
        """简单的测试函数"""
        return [1.0, 2.0, 3.0]
    
    test_args = [[20240101, "TEST"]]
    
    with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as tmp_file:
        backup_file = tmp_file.name
    
    try:
        import rust_pyfunc
        
        print("即将调用run_pools，注意观察Rust日志输出：")
        results = rust_pyfunc.run_pools(
            simple_func,
            test_args,
            backup_file=backup_file,
            num_threads=2
        )
        
        print(f"执行完成，结果: {results[0]}")
        return True
        
    except Exception as e:
        print(f"测试失败: {e}")
        return False
        
    finally:
        # 清理临时文件
        if os.path.exists(backup_file):
            os.unlink(backup_file)

if __name__ == "__main__":
    success = test_jupyter_logging()
    if success:
        print("✅ Jupyter日志显示测试通过！")
    else:
        print("❌ 测试失败")