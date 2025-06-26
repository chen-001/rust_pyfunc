#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试大备份文件读取功能
测试四种增强的读取方法
"""

import sys
import os
sys.path.insert(0, '/home/chenzongwei/rust_pyfunc/python')

import rust_pyfunc as rp
import numpy as np
import time

def test_backup_dataset_info():
    """测试数据集信息获取功能"""
    print("=" * 50)
    print("测试1: 获取备份文件数据集信息")
    print("=" * 50)
    
    # 找一个备份文件进行测试
    backup_files = []
    for root, dirs, files in os.walk('/home/chenzongwei'):
        for file in files:
            if file.endswith('.backup') or file.endswith('.bak'):
                backup_files.append(os.path.join(root, file))
                if len(backup_files) >= 3:  # 只测试前3个
                    break
        if len(backup_files) >= 3:
            break
    
    if not backup_files:
        print("未找到任何备份文件，跳过测试")
        return
    
    for backup_file in backup_files:
        try:
            print(f"\n测试文件: {backup_file}")
            
            # 1. 使用新的get_backup_dataset_info函数
            rows, cols, memory_gb, min_date, max_date = rp.get_backup_dataset_info(backup_file, "binary")
            print(f"数据集信息:")
            print(f"  行数: {rows:,}")
            print(f"  列数: {cols}")
            print(f"  预计内存: {memory_gb:.2f} GB")
            print(f"  日期范围: {min_date} 到 {max_date}")
            
            # 检查是否为大文件
            if memory_gb > 2.0:
                print(f"⚠️  这是一个大文件 ({memory_gb:.1f}GB)，建议使用分块读取")
            
        except Exception as e:
            print(f"❌ 测试失败: {e}")
    
def test_chunked_reading():
    """测试分块读取功能"""
    print("\n" + "=" * 50)
    print("测试2: 分块读取功能")
    print("=" * 50)
    
    # 找一个相对较大的备份文件
    backup_files = []
    for root, dirs, files in os.walk('/home/chenzongwei'):
        for file in files:
            if file.endswith('.backup') or file.endswith('.bak'):
                file_path = os.path.join(root, file)
                if os.path.getsize(file_path) > 1000000:  # 大于1MB的文件
                    backup_files.append(file_path)
                    break
        if backup_files:
            break
    
    if not backup_files:
        print("未找到大型备份文件，跳过测试")
        return
    
    backup_file = backup_files[0]
    print(f"测试文件: {backup_file}")
    
    try:
        # 先获取数据集信息
        rows, cols, memory_gb, min_date, max_date = rp.get_backup_dataset_info(backup_file, "binary")
        print(f"总行数: {rows:,}")
        
        if rows == 0:
            print("文件为空，跳过分块测试")
            return
        
        # 2. 测试行范围读取
        print("\n2.1 测试行范围读取 (前100行)")
        start_time = time.time()
        result = rp.query_backup(backup_file, row_range=(0, 100), storage_format="binary")
        end_time = time.time()
        print(f"读取结果: {result.shape if result.size > 0 else '空数组'}")
        print(f"耗时: {end_time - start_time:.3f}秒")
        
        # 3. 测试最大行数限制
        print("\n2.2 测试最大行数限制 (最多50行)")
        start_time = time.time()
        result = rp.query_backup(backup_file, max_rows=50, storage_format="binary")
        end_time = time.time()
        print(f"读取结果: {result.shape if result.size > 0 else '空数组'}")
        print(f"耗时: {end_time - start_time:.3f}秒")
        
        # 4. 测试内存限制
        print("\n2.3 测试内存限制 (限制0.1GB)")
        start_time = time.time()
        try:
            result = rp.query_backup(backup_file, memory_limit_gb=0.1, storage_format="binary")
            print(f"读取结果: {result.shape if result.size > 0 else '空数组'}")
        except Exception as e:
            print(f"符合预期的内存限制错误: {e}")
        end_time = time.time()
        print(f"耗时: {end_time - start_time:.3f}秒")
        
        # 5. 测试较宽松的内存限制
        print("\n2.4 测试较宽松的内存限制 (限制2GB)")
        start_time = time.time()
        result = rp.query_backup(backup_file, memory_limit_gb=2.0, storage_format="binary")
        end_time = time.time()
        print(f"读取结果: {result.shape if result.size > 0 else '空数组'}")
        print(f"耗时: {end_time - start_time:.3f}秒")
        
    except Exception as e:
        print(f"❌ 分块读取测试失败: {e}")

def test_large_file_scenarios():
    """测试大文件场景"""
    print("\n" + "=" * 50)
    print("测试3: 大文件处理场景")
    print("=" * 50)
    
    # 寻找较大的备份文件
    large_files = []
    for root, dirs, files in os.walk('/home/chenzongwei'):
        for file in files:
            if file.endswith('.backup') or file.endswith('.bak'):
                file_path = os.path.join(root, file)
                file_size = os.path.getsize(file_path)
                if file_size > 5000000:  # 大于5MB
                    large_files.append((file_path, file_size))
        if len(large_files) >= 2:
            break
    
    if not large_files:
        print("未找到大型备份文件，创建模拟场景")
        return
    
    # 按文件大小排序，测试最大的
    large_files.sort(key=lambda x: x[1], reverse=True)
    
    for file_path, file_size in large_files[:2]:
        print(f"\n测试大文件: {file_path}")
        print(f"文件大小: {file_size / (1024*1024):.1f} MB")
        
        try:
            # 首先获取文件信息
            rows, cols, memory_gb, min_date, max_date = rp.get_backup_dataset_info(file_path, "binary")
            print(f"预估信息: {rows:,}行, {cols}列, {memory_gb:.2f}GB")
            
            # 如果预计内存使用超过1GB，使用分块读取
            if memory_gb > 1.0:
                print("💡 使用分块读取策略")
                
                # 计算合适的分块大小
                chunk_size = max(1000, int(rows * 0.01))  # 1%的数据作为一块
                print(f"分块大小: {chunk_size:,}行")
                
                # 读取第一块
                result = rp.query_backup(file_path, row_range=(0, chunk_size), storage_format="binary")
                print(f"第一块读取成功: {result.shape if result.size > 0 else '空数组'}")
                
                # 读取中间一块
                mid_start = rows // 2
                mid_end = min(mid_start + chunk_size, rows)
                result = rp.query_backup(file_path, row_range=(mid_start, mid_end), storage_format="binary")
                print(f"中间块读取成功: {result.shape if result.size > 0 else '空数组'}")
                
            else:
                print("✅ 文件大小适中，可以直接读取")
                result = rp.query_backup(file_path, storage_format="binary")
                print(f"完整读取成功: {result.shape if result.size > 0 else '空数组'}")
                
        except Exception as e:
            print(f"❌ 大文件测试失败: {e}")

def main():
    """主测试函数"""
    print("🧪 开始测试大备份文件读取功能")
    print("测试四种新的读取方法:")
    print("1. get_backup_dataset_info() - 获取数据集信息")
    print("2. query_backup(row_range=...) - 按行范围读取")
    print("3. query_backup(max_rows=...) - 限制最大行数")
    print("4. query_backup(memory_limit_gb=...) - 自定义内存限制")
    
    try:
        test_backup_dataset_info()
        test_chunked_reading() 
        test_large_file_scenarios()
        
        print("\n" + "=" * 50)
        print("✅ 所有测试完成!")
        print("=" * 50)
        
    except Exception as e:
        print(f"\n❌ 测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()