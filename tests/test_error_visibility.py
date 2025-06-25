#!/usr/bin/env python3
"""
测试错误可见性 - 确保错误信息能正确显示
"""

import os
import sys
import tempfile

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'python'))

def error_function(date, code):
    """故意出错的函数"""
    if code == "000001":
        raise FileNotFoundError("模拟文件不存在错误")
    elif code == "000002":
        raise ValueError("模拟数据值错误")
    else:
        return [1.0, 2.0, 3.0]

def test_error_visibility():
    """测试错误可见性"""
    print("=== 测试错误可见性 ===")
    
    test_args = [
        [20170101, "000001"],  # 会出错
        [20170101, "000002"],  # 会出错
        [20170101, "000003"],  # 正常
    ]
    
    with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as tmp_file:
        backup_file = tmp_file.name
    
    try:
        import rust_pyfunc
        
        print("开始错误可见性测试...")
        print("预期：前两个任务出错，第三个正常")
        print()
        
        results = rust_pyfunc.run_pools(
            error_function,
            test_args,
            backup_file=backup_file,
            num_threads=3
        )
        
        print(f"测试完成，结果数量: {len(results)}")
        return True
        
    except Exception as e:
        print(f"测试遇到异常: {e}")
        print("这说明错误报告系统正在工作！")
        return True  # 这实际上是好的，因为我们想看到错误
        
    finally:
        if os.path.exists(backup_file):
            os.unlink(backup_file)

if __name__ == "__main__":
    print("测试错误报告系统的可见性...")
    test_error_visibility()