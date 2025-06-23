#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调试复杂情况下的Broken pipe问题
"""

import sys
sys.path.append('/home/chenzongwei/rust_pyfunc/python')

import rust_pyfunc


def complex_function(date, code):
    """复杂计算函数"""
    result = 0
    for i in range(1000):  # 更多计算
        result += hash(f"{date}_{code}_{i}") % 100
    
    return [
        float(date % 10000),
        float(len(code)),
        float(result % 1000),
        float((date + len(code)) % 500)
    ]


def test_multiple_processes():
    """测试多进程情况"""
    print("🔍 测试多进程情况")
    
    # 更多任务
    args = [[20220101 + i, f"{i+1:06d}"] for i in range(20)]
    
    print(f"测试数据: {len(args)} 个任务")
    
    try:
        print(f"🚀 使用4个进程执行...")
        result = rust_pyfunc.run_multiprocess(
            complex_function,
            args,
            num_processes=4,
        )
        
        print(f"✅ 执行成功!")
        print(f"结果数量: {len(result)}")
        
    except Exception as e:
        print(f"❌ 执行失败: {e}")
        import traceback
        traceback.print_exc()


def test_with_backup():
    """测试带备份的情况"""
    print(f"\n🔍 测试带备份的情况")
    
    import tempfile
    
    args = [[20220101 + i, f"{i+1:06d}"] for i in range(10)]
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        backup_file = f.name
    
    try:
        print(f"🚀 使用备份执行...")
        result = rust_pyfunc.run_multiprocess(
            complex_function,
            args,
            num_processes=2,
            backup_file=backup_file,
            storage_format="json"
        )
        
        print(f"✅ 执行成功!")
        print(f"结果数量: {len(result)}")
        
    except Exception as e:
        print(f"❌ 执行失败: {e}")
        import traceback
        traceback.print_exc()
    finally:
        import os
        if os.path.exists(backup_file):
            os.unlink(backup_file)


def test_run_pools_api():
    """测试run_pools API"""
    print(f"\n🔍 测试run_pools API")
    
    args = [[20220101 + i, f"{i+1:06d}"] for i in range(15)]
    
    try:
        print(f"🚀 使用run_pools执行...")
        result = rust_pyfunc.run_pools(
            complex_function,
            args,
            num_threads=3,
        )
        
        print(f"✅ 执行成功!")
        print(f"结果数量: {len(result)}")
        
    except Exception as e:
        print(f"❌ 执行失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_multiple_processes()
    test_with_backup()
    test_run_pools_api()