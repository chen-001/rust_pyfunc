#!/usr/bin/env python3
"""
快速诊断备份文件问题
"""
import os
import rust_pyfunc as rf

backup_file = "/home/chenzongwei/pythoncode/万里长征/backup_pool_万里长征放宽参数20.bin"

print(f"文件: {backup_file}")
print(f"存在: {os.path.exists(backup_file)}")

if os.path.exists(backup_file):
    size_bytes = os.path.getsize(backup_file)
    size_mb = size_bytes / (1024 * 1024)
    print(f"大小: {size_mb:.1f} MB")
    
    # 检查rust能否识别
    try:
        exists = rf.backup_exists(backup_file, "binary")
        print(f"Rust识别: {exists}")
        
        if exists:
            # 获取详细信息
            rows, cols, memory_gb, earliest, latest = rf.get_backup_dataset_info(backup_file, "binary")
            print(f"行数: {rows:,}")
            print(f"列数: {cols}")
            print(f"预计内存: {memory_gb:.2f} GB")
            print(f"日期: {earliest} ~ {latest}")
            
            # 测试读取10行
            print("测试读取10行...")
            result = rf.query_backup(backup_file, storage_format="binary", max_rows=10)
            print(f"结果形状: {result.shape}")
            print("前3行:")
            for i in range(min(3, result.shape[0])):
                print(f"  {result[i][:5]}")
        
    except Exception as e:
        print(f"错误: {e}")