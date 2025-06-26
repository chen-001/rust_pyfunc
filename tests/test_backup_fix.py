#!/usr/bin/env python3
"""
测试backup_pool_万里长征放宽参数20.bin文件读取修复效果
"""
import os
import sys
import numpy as np
import rust_pyfunc as rf

def test_backup_file_reading():
    """测试备份文件读取功能"""
    backup_file = "/home/chenzongwei/pythoncode/万里长征/backup_pool_万里长征放宽参数20.bin"
    
    print(f"测试文件: {backup_file}")
    
    # 检查文件是否存在
    if not os.path.exists(backup_file):
        print(f"❌ 文件不存在: {backup_file}")
        return False
    
    # 获取文件大小
    file_size_bytes = os.path.getsize(backup_file)
    file_size_mb = file_size_bytes / (1024 * 1024)
    file_size_gb = file_size_mb / 1024
    print(f"📁 文件大小: {file_size_mb:.1f} MB ({file_size_gb:.2f} GB)")
    
    try:
        # 检查备份文件是否存在（使用rust函数）
        exists = rf.backup_exists(backup_file, "binary")
        print(f"🔍 备份文件存在检查: {exists}")
        
        if not exists:
            print("❌ Rust函数无法识别备份文件")
            return False
        
        # 获取备份文件信息
        try:
            file_size, format_info = rf.get_backup_info(backup_file, "binary")
            print(f"📊 备份文件信息: 大小={file_size}字节, 格式={format_info}")
        except Exception as e:
            print(f"⚠️  获取基本信息失败: {e}")
        
        # 获取详细信息
        try:
            rows, cols, memory_gb, earliest_date, latest_date = rf.get_backup_dataset_info(backup_file, "binary")
            print(f"📈 数据集信息:")
            print(f"   - 行数: {rows:,}")
            print(f"   - 列数: {cols}")
            print(f"   - 预计内存: {memory_gb:.2f} GB")
            print(f"   - 日期范围: {earliest_date} ~ {latest_date}")
        except Exception as e:
            print(f"⚠️  获取详细信息失败: {e}")
        
        # 测试小量数据读取
        print("\n🔄 测试读取前1000行...")
        try:
            result = rf.query_backup(
                backup_file, 
                storage_format="binary",
                max_rows=1000
            )
            print(f"✅ 成功读取小量数据，形状: {result.shape}")
            print(f"   - 数据类型: {result.dtype}")
            if result.size > 0:
                print(f"   - 前几行预览:")
                for i in range(min(3, result.shape[0])):
                    print(f"     行{i}: {result[i][:min(5, result.shape[1])]}")
        except Exception as e:
            print(f"❌ 读取小量数据失败: {e}")
            return False
        
        # 测试默认内存限制读取
        print(f"\n🔄 测试使用默认64GB内存限制读取...")
        try:
            result = rf.query_backup(
                backup_file, 
                storage_format="binary"
            )
            print(f"✅ 成功读取完整数据，形状: {result.shape}")
            print(f"   - 实际行数: {result.shape[0]:,}")
            print(f"   - 实际列数: {result.shape[1]}")
            
            # 检查数据是否为空
            if result.size == 0:
                print("❌ 数据为空！")
                return False
            else:
                print(f"✅ 数据不为空，总元素数: {result.size:,}")
                return True
                
        except Exception as e:
            print(f"❌ 读取完整数据失败: {e}")
            
            # 如果仍然失败，尝试更大的内存限制
            print(f"\n🔄 尝试使用128GB内存限制...")
            try:
                result = rf.query_backup(
                    backup_file, 
                    storage_format="binary",
                    memory_limit_gb=128.0
                )
                print(f"✅ 使用128GB限制成功读取，形状: {result.shape}")
                return True
            except Exception as e2:
                print(f"❌ 使用128GB限制也失败: {e2}")
                return False
        
    except Exception as e:
        print(f"❌ 测试过程中发生错误: {e}")
        return False

if __name__ == "__main__":
    print("🚀 开始测试备份文件读取修复效果...")
    success = test_backup_file_reading()
    
    if success:
        print("\n🎉 测试成功！备份文件读取问题已修复")
        sys.exit(0)
    else:
        print("\n😞 测试失败，仍有问题需要解决")
        sys.exit(1)