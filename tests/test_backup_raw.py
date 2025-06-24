#!/usr/bin/env python3
"""
直接检查备份文件原始内容
"""

import os
import sys
import tempfile
import struct

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

def inspect_backup_file():
    """检查备份文件的原始内容"""
    
    def calc(date, code):
        return [1.0, 2.0, 3.0]
    
    test_args = [[20240101, "TEST001"]]
    
    with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as tmp_file:
        backup_file = tmp_file.name
    
    try:
        import rust_pyfunc
        
        # 先执行计算
        results = rust_pyfunc.run_pools(
            calc,
            test_args,
            backup_file=backup_file,
            num_threads=1,
            backup_batch_size=1,
            storage_format="binary"
        )
        
        print(f"Results: {results}")
        
        # 检查备份文件原始内容
        if os.path.exists(backup_file):
            file_size = os.path.getsize(backup_file)
            print(f"Backup file size: {file_size} bytes")
            
            # 读取原始字节
            with open(backup_file, 'rb') as f:
                raw_data = f.read()
            
            print(f"Raw data length: {len(raw_data)} bytes")
            print(f"First 20 bytes: {raw_data[:20].hex()}")
            
            # 尝试解析长度前缀
            if len(raw_data) >= 4:
                length = struct.unpack('<I', raw_data[:4])[0]  # little-endian unsigned int
                print(f"Data length from header: {length} bytes")
                
                if len(raw_data) >= 4 + length:
                    payload = raw_data[4:4+length]
                    print(f"Payload length: {len(payload)} bytes")
                    print(f"Payload hex: {payload.hex()}")
                    
                    # 尝试使用rust_pyfunc查询
                    backup_results = rust_pyfunc.query_backup(
                        backup_file,
                        storage_format="binary"
                    )
                    print(f"Query results: {backup_results}")
                else:
                    print("File too short for declared payload length")
            else:
                print("File too short for length header")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # 保留文件以便进一步检查
        print(f"Backup file preserved at: {backup_file}")

if __name__ == "__main__":
    inspect_backup_file()