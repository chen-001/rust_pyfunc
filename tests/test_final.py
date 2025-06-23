#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import tempfile
import os
import time
sys.path.append('/home/chenzongwei/rust_pyfunc/python')

import rust_pyfunc

def simple_analysis(date, code):
    return [1.0, 2.0, 3.0]

def test_storage_format(format_name):
    print(f"\n=== 测试 {format_name} 存储格式 ===")
    
    with tempfile.NamedTemporaryFile(mode='w', suffix=f'.{format_name}', delete=False) as f:
        backup_file = f.name
    
    try:
        args = [(20220101, "000001"), (20220102, "000002")]
        
        start_time = time.time()
        result = rust_pyfunc.run_pools(
            simple_analysis,
            args,
            backup_file=backup_file,
            storage_format=format_name,
            num_threads=1,
            backup_batch_size=1000
        )
        execution_time = time.time() - start_time
        
        print(f"✓ 执行完成，耗时: {execution_time:.3f}秒")
        print(f"✓ 结果数量: {len(result)}")
        
        # 检查备份文件
        if os.path.exists(backup_file):
            size = os.path.getsize(backup_file)
            print(f"✓ 备份文件大小: {size} 字节")
        
        # 测试查询
        start_time = time.time()
        backup_data = rust_pyfunc.query_backup(backup_file, storage_format=format_name)
        query_time = time.time() - start_time
        
        print(f"✓ 查询完成，耗时: {query_time:.3f}秒")
        print(f"✓ 查询结果数量: {len(backup_data)}")
        
        # 验证数据完整性
        assert len(backup_data) == len(args), f"数据不匹配: {len(backup_data)} != {len(args)}"
        print(f"✅ {format_name} 存储测试成功！")
        return True
        
    except Exception as e:
        print(f"❌ {format_name} 存储测试失败: {e}")
        return False
    finally:
        if os.path.exists(backup_file):
            os.unlink(backup_file)

def main():
    print("🚀 高性能存储系统最终测试")
    print("=" * 50)
    
    formats = ["json", "binary", "memory_map"]
    success_count = 0
    
    for fmt in formats:
        if test_storage_format(fmt):
            success_count += 1
    
    print("\n" + "=" * 50)
    print(f"测试结果: {success_count}/{len(formats)} 格式通过")
    
    if success_count == len(formats):
        print("🎉 所有存储格式测试成功！高性能存储系统就绪！")
    else:
        print("⚠️ 部分格式测试失败")

if __name__ == "__main__":
    main()