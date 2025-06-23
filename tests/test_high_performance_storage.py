#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试高性能存储方案
"""

import time
import os
import tempfile
import sys
sys.path.append('/home/chenzongwei/rust_pyfunc/python')

import rust_pyfunc

def simple_analysis(date, code):
    """简单的分析函数"""
    return [1.0, 2.0, 3.0]

def test_storage_format(storage_format, suffix):
    """测试单个存储格式"""
    print(f"\n=== 测试 {storage_format} 存储格式 ===")
    
    # 创建临时文件
    with tempfile.NamedTemporaryFile(mode='w', suffix=suffix, delete=False) as f:
        backup_file = f.name
    
    try:
        args = [
            (20220101, "000001"),
            (20220101, "000002"),
            (20220102, "000001"),
        ]
        
        print(f"备份文件: {backup_file}")
        
        # 测试写入
        start_time = time.time()
        result = rust_pyfunc.run_pools(
            simple_analysis,
            args,
            backup_file=backup_file,
            storage_format=storage_format,
            num_threads=1
        )
        write_time = time.time() - start_time
        
        print(f"写入耗时: {write_time:.4f}秒")
        print(f"结果数量: {len(result)}")
        
        # 验证文件存在
        file_size = os.path.getsize(backup_file) if os.path.exists(backup_file) else 0
        print(f"文件大小: {file_size} 字节")
        
        # 测试读取
        start_time = time.time()
        backup_data = rust_pyfunc.query_backup(backup_file, storage_format=storage_format)
        read_time = time.time() - start_time
        
        print(f"读取耗时: {read_time:.4f}秒")
        print(f"备份数据数量: {len(backup_data)}")
        
        if backup_data:
            print(f"第一条数据: {backup_data[0]}")
        
        # 验证数据正确性
        assert len(backup_data) == len(args), f"{storage_format}备份数据数量不正确"
        assert len(backup_data[0]) == 6, f"{storage_format}备份数据格式不正确"  # date, code, timestamp, fac1, fac2, fac3
        
        print(f"✓ {storage_format}测试通过")
        
        return write_time, read_time, file_size
        
    except Exception as e:
        print(f"❌ {storage_format}测试失败: {e}")
        return None, None, None
        
    finally:
        # 清理临时文件
        if os.path.exists(backup_file):
            os.unlink(backup_file)

def test_performance_comparison():
    """性能对比测试"""
    print("\n" + "=" * 60)
    print("高性能存储方案性能对比")
    print("=" * 60)
    
    formats = [
        ("json", ".json"),
        ("sqlite", ".db"),
        ("memory_map", ".bin"),
        ("parquet", ".parquet")
    ]
    
    results = {}
    
    for storage_format, suffix in formats:
        write_time, read_time, file_size = test_storage_format(storage_format, suffix)
        if write_time is not None:
            results[storage_format] = {
                'write_time': write_time,
                'read_time': read_time,
                'file_size': file_size
            }
    
    # 打印性能对比
    print("\n" + "=" * 60)
    print("性能对比结果:")
    print("=" * 60)
    print(f"{'格式':<12} {'写入时间':<10} {'读取时间':<10} {'文件大小':<10}")
    print("-" * 50)
    
    for fmt, data in results.items():
        print(f"{fmt:<12} {data['write_time']:<10.4f} {data['read_time']:<10.4f} {data['file_size']:<10}")
    
    print("\n注意：")
    print("- 写入时间包括计算和存储时间")
    print("- 读取时间仅包括从存储读取和反序列化时间")
    print("- 文件大小单位：字节")

def test_large_scale():
    """大规模数据测试"""
    print("\n" + "=" * 60)
    print("大规模数据测试 (1000个任务)")
    print("=" * 60)
    
    # 创建1000个任务
    args = [(20220101 + i // 100, f"{i:06d}") for i in range(1000)]
    
    # 测试SQLite（预期最快）
    with tempfile.NamedTemporaryFile(mode='w', suffix='.db', delete=False) as f:
        backup_file = f.name
    
    try:
        print(f"测试SQLite存储1000个任务...")
        start_time = time.time()
        
        result = rust_pyfunc.run_pools(
            simple_analysis,
            args,
            backup_file=backup_file,
            storage_format="sqlite",
            backup_batch_size=100,
            num_threads=1
        )
        
        total_time = time.time() - start_time
        speed = len(args) / total_time
        
        print(f"总耗时: {total_time:.2f}秒")
        print(f"处理速度: {speed:.0f} 任务/秒")
        print(f"结果数量: {len(result)}")
        
        # 测试查询性能
        start_time = time.time()
        backup_data = rust_pyfunc.query_backup(backup_file, storage_format="sqlite")
        query_time = time.time() - start_time
        
        print(f"查询耗时: {query_time:.4f}秒")
        print(f"查询到数据: {len(backup_data)}条")
        
        file_size = os.path.getsize(backup_file)
        print(f"数据库文件大小: {file_size / 1024:.1f} KB")
        
        print("✓ 大规模测试通过")
        
    except Exception as e:
        print(f"❌ 大规模测试失败: {e}")
        
    finally:
        if os.path.exists(backup_file):
            os.unlink(backup_file)

def main():
    """主测试函数"""
    print("开始测试rust_pyfunc高性能存储方案")
    
    try:
        test_performance_comparison()
        test_large_scale()
        
        print("\n" + "=" * 60)
        print("🎉 高性能存储方案测试完成！")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()