#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import tempfile
import os
import time
sys.path.append('/home/chenzongwei/rust_pyfunc/python')

import rust_pyfunc

def simple_analysis(date, code):
    print(f"处理 {date} {code}")
    return [1.0, 2.0, 3.0]

def test_with_debug():
    print("=== 调试测试开始 ===")
    with tempfile.NamedTemporaryFile(mode='w', suffix='.bin', delete=False) as f:
        backup_file = f.name
    
    print(f"备份文件: {backup_file}")
    
    try:
        args = [(20220101, "000001")]
        print(f"参数: {args}")
        
        print("调用 run_pools...")
        start_time = time.time()
        
        result = rust_pyfunc.run_pools(
            simple_analysis,
            args,
            backup_file=backup_file,
            storage_format="binary",
            num_threads=1,
            backup_batch_size=1
        )
        
        end_time = time.time()
        print(f"run_pools 完成，耗时: {end_time - start_time:.2f}秒")
        print(f"结果: {result}")
        
        # 检查文件是否创建
        if os.path.exists(backup_file):
            size = os.path.getsize(backup_file)
            print(f"备份文件大小: {size} 字节")
        else:
            print("⚠️ 备份文件未创建")
        
        print("测试查询备份...")
        backup_data = rust_pyfunc.query_backup(backup_file, storage_format="binary")
        print(f"查询结果: {backup_data}")
        
        print("✅ 测试完成")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if os.path.exists(backup_file):
            print(f"清理文件: {backup_file}")
            os.unlink(backup_file)

if __name__ == "__main__":
    test_with_debug()