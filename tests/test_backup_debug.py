#!/usr/bin/env python3
"""
调试备份文件生成问题
"""

import os
import sys
import tempfile
import time

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

def test_backup_creation():
    """测试备份文件的创建"""
    print("开始备份文件调试测试...")
    
    def simple_func(date, code):
        """简单的测试函数"""
        result = [float(date + i) for i in range(10)]  # 返回10个结果
        return result
    
    # 创建测试任务
    test_args = [[20240101, "TEST001"], [20240102, "TEST002"]]
    
    with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as tmp_file:
        backup_file = tmp_file.name
    
    try:
        import rust_pyfunc
        
        print(f"测试任务数: {len(test_args)}")
        print(f"备份文件: {backup_file}")
        
        # 检查备份文件初始状态
        print(f"执行前备份文件存在: {os.path.exists(backup_file)}")
        if os.path.exists(backup_file):
            print(f"执行前备份文件大小: {os.path.getsize(backup_file)} bytes")
        
        start_time = time.time()
        
        # 执行计算
        results = rust_pyfunc.run_pools(
            simple_func,
            test_args,
            backup_file=backup_file,
            num_threads=2,
            backup_batch_size=1,  # 小批次确保立即写入
            storage_format="binary"
        )
        
        end_time = time.time()
        
        print(f"计算完成! 耗时: {end_time - start_time:.2f}秒")
        print(f"结果数量: {len(results)}")
        
        # 检查备份文件状态
        print(f"执行后备份文件存在: {os.path.exists(backup_file)}")
        if os.path.exists(backup_file):
            file_size = os.path.getsize(backup_file)
            print(f"执行后备份文件大小: {file_size} bytes")
            
            if file_size > 0:
                print("✅ 备份文件有数据!")
                
                # 尝试直接读取备份
                try:
                    backup_results = rust_pyfunc.query_backup(
                        backup_file,
                        storage_format="binary"
                    )
                    print(f"从备份读取到 {len(backup_results)} 条结果")
                    if len(backup_results) > 0:
                        print(f"第一条备份结果: {backup_results[0]}")
                except Exception as e:
                    print(f"读取备份失败: {e}")
            else:
                print("❌ 备份文件为空!")
        
        if len(results) > 0:
            print(f"第一个结果: {results[0]}")
            print(f"结果长度: {len(results[0])}")
            
            # 验证结果格式
            if len(results[0]) >= 12:  # date, code, + 10个facs
                print("✅ 结果格式正确")
                return True
            else:
                print(f"❌ 结果格式错误，期望至少12个元素，实际: {len(results[0])}")
                return False
        else:
            print("❌ 没有结果返回")
            return False
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # 保留备份文件用于调试
        print(f"保留备份文件用于调试: {backup_file}")

if __name__ == "__main__":
    success = test_backup_creation()
    print(f"\n{'=' * 50}")
    if success:
        print("🎉 备份调试测试通过!")
        sys.exit(0)
    else:
        print("❌ 备份调试测试失败!")
        sys.exit(1)