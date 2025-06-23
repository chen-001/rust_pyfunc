#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调试CSV问题
"""

import sys
import tempfile
import os
sys.path.append('/home/chenzongwei/rust_pyfunc/python')

import rust_pyfunc

def simple_analysis(date, code):
    return [1.0, 2.0, 3.0]

# 创建临时文件
with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
    backup_file = f.name

try:
    args = [
        (20220101, "000001"),
        (20220101, "000002"),
        (20220102, "000001"),
    ]
    
    print(f"CSV备份文件: {backup_file}")
    
    # 使用CSV格式进行备份
    result = rust_pyfunc.run_pools(
        simple_analysis,
        args,
        backup_file=backup_file,
        storage_format="csv",
        num_threads=1
    )
    
    print(f"CSV存储结果数量: {len(result)}")
    
    # 验证备份文件存在
    print(f"文件是否存在: {os.path.exists(backup_file)}")
    if os.path.exists(backup_file):
        with open(backup_file, 'r') as f:
            content = f.read()
            print("文件内容:")
            print(repr(content))
    
    # 查询CSV备份数据
    try:
        backup_data = rust_pyfunc.query_backup(backup_file)
        print(f"CSV备份数据数量: {len(backup_data)}")
        if backup_data:
            print(f"第一条数据: {backup_data[0]}")
    except Exception as e:
        print(f"查询失败: {e}")
        
finally:
    # 清理临时文件
    if os.path.exists(backup_file):
        os.unlink(backup_file)