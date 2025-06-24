#!/usr/bin/env python3
"""
调试函数返回值问题
"""

import os
import sys
import tempfile
import time

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

def test_function_return():
    """测试函数返回值"""
    print("开始函数返回值调试测试...")
    
    def debug_func(date, code):
        """带调试信息的测试函数"""
        print(f"DEBUG: debug_func called with date={date}, code={code}")
        result = [float(date + i) for i in range(10)]  # 返回10个结果
        print(f"DEBUG: debug_func returning {len(result)} items: {result[:3]}...")
        return result
    
    # 创建测试任务
    test_args = [[20240101, "TEST001"]]
    
    with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as tmp_file:
        backup_file = tmp_file.name
    
    try:
        import rust_pyfunc
        
        print(f"测试任务数: {len(test_args)}")
        print(f"备份文件: {backup_file}")
        
        start_time = time.time()
        
        # 执行计算
        results = rust_pyfunc.run_pools(
            debug_func,
            test_args,
            backup_file=backup_file,
            num_threads=1,  # 单线程便于调试
            backup_batch_size=1,
            storage_format="binary"
        )
        
        end_time = time.time()
        
        print(f"计算完成! 耗时: {end_time - start_time:.2f}秒")
        print(f"结果数量: {len(results)}")
        
        if len(results) > 0:
            print(f"返回结果: {results[0]}")
            print(f"返回结果长度: {len(results[0])}")
        
        # 检查备份文件
        if os.path.exists(backup_file):
            file_size = os.path.getsize(backup_file)
            print(f"备份文件大小: {file_size} bytes")
            
            if file_size > 0:
                try:
                    backup_results = rust_pyfunc.query_backup(
                        backup_file,
                        storage_format="binary"
                    )
                    print(f"从备份读取到 {len(backup_results)} 条结果")
                    if len(backup_results) > 0:
                        print(f"备份结果: {backup_results[0]}")
                        print(f"备份结果长度: {len(backup_results[0])}")
                except Exception as e:
                    print(f"读取备份失败: {e}")
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        if os.path.exists(backup_file):
            os.unlink(backup_file)

if __name__ == "__main__":
    test_function_return()