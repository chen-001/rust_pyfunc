#!/usr/bin/env python3
"""
调试备份文件内容
"""

import os
import sys
import tempfile

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

def debug_backup_content():
    """调试备份文件内容"""
    print("开始调试备份文件内容...")
    
    # 测试数据
    test_args = [[20240101 + i, f"code_{i}"] for i in range(10)]  # 10个测试任务
    
    def test_func(date, code):
        """测试函数"""
        return [float(date), float(len(code)), 999.0]
    
    # 创建临时备份文件
    with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as tmp_file:
        backup_file = tmp_file.name
    
    try:
        import rust_pyfunc
        
        # 执行计算
        print(f"执行{len(test_args)}个任务...")
        results = rust_pyfunc.run_pools(
            test_func,
            test_args,
            backup_file=backup_file,
            num_threads=2,
            backup_batch_size=5  # 小批次
        )
        
        print(f"计算完成，结果数量: {len(results)}")
        print(f"备份文件大小: {os.path.getsize(backup_file)} bytes")
        
        # 使用backup查询功能检查文件内容
        if hasattr(rust_pyfunc, 'query_backup'):
            print("\n=== 通过query_backup查询 ===")
            query_results = rust_pyfunc.query_backup(
                backup_file=backup_file,
                storage_format="binary"
            )
            print(f"查询到 {len(query_results)} 个结果")
            for i, result in enumerate(query_results[:3]):
                print(f"查询结果{i}: {result}")
        
        # 直接检查返回的结果
        print("\n=== 直接返回结果 ===")
        for i, result in enumerate(results[:3]):
            print(f"返回结果{i}: {result}")
            print(f"  类型: {type(result)}")
            print(f"  内容: {result}")
            if hasattr(result, 'shape'):
                print(f"  形状: {result.shape}")
        
        return True
        
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # 清理临时文件
        if os.path.exists(backup_file):
            os.unlink(backup_file)


if __name__ == "__main__":
    debug_backup_content()