#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import tempfile
import os
sys.path.append('/home/chenzongwei/rust_pyfunc/python')

import rust_pyfunc

def simple_analysis(date, code):
    return [1.0, 2.0, 3.0]

def test_json():
    print("测试JSON存储...")
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        backup_file = f.name
    
    try:
        args = [(20220101, "000001"), (20220101, "000002")]
        
        result = rust_pyfunc.run_pools(
            simple_analysis,
            args,
            backup_file=backup_file,
            storage_format="json",
            num_threads=1
        )
        
        print(f"结果数量: {len(result)}")
        
        backup_data = rust_pyfunc.query_backup(backup_file, storage_format="json")
        print(f"备份数据数量: {len(backup_data)}")
        
        assert len(backup_data) == len(args)
        print("✓ JSON测试通过")
        return True
        
    except Exception as e:
        print(f"❌ JSON测试失败: {e}")
        return False
    finally:
        if os.path.exists(backup_file):
            os.unlink(backup_file)

def test_memory_map():
    print("测试内存映射存储...")
    with tempfile.NamedTemporaryFile(mode='w', suffix='.bin', delete=False) as f:
        backup_file = f.name
    
    try:
        args = [(20220101, "000001"), (20220101, "000002")]
        
        result = rust_pyfunc.run_pools(
            simple_analysis,
            args,
            backup_file=backup_file,
            storage_format="memory_map",
            num_threads=1
        )
        
        print(f"结果数量: {len(result)}")
        
        backup_data = rust_pyfunc.query_backup(backup_file, storage_format="memory_map")
        print(f"备份数据数量: {len(backup_data)}")
        
        assert len(backup_data) == len(args)
        print("✓ 内存映射测试通过")
        return True
        
    except Exception as e:
        print(f"❌ 内存映射测试失败: {e}")
        return False
    finally:
        if os.path.exists(backup_file):
            os.unlink(backup_file)

def main():
    print("开始简单存储测试")
    print("=" * 40)
    
    success_count = 0
    
    if test_json():
        success_count += 1
    
    if test_memory_map():
        success_count += 1
    
    print("=" * 40)
    print(f"成功测试: {success_count}/2")
    
    if success_count == 2:
        print("🎉 基本存储功能正常！")
    else:
        print("❌ 部分测试失败")

if __name__ == "__main__":
    main()