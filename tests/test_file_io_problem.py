#!/usr/bin/env python3
"""
测试文件IO操作导致的子进程问题
"""

import os
import sys
import tempfile
import pandas as pd

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

def test_file_io_in_subprocess():
    """测试子进程中的文件IO操作"""
    print("=== 测试子进程中的文件IO操作 ===")
    
    def file_io_func(date, code):
        import os
        import pandas as pd
        
        # 模拟万里长征的文件读取操作
        file_name = "%s_%d_%s.csv" % (code, date, "transaction")
        file_path = os.path.join("/ssd_data/stock", str(date), "transaction", file_name)
        
        # 检查文件是否存在
        if not os.path.exists(file_path):
            print(f"文件不存在: {file_path}")
            return [0.0] * 10
            
        try:
            # 尝试读取文件
            df = pd.read_csv(file_path)
            return [1.0, 2.0, float(len(df))]
        except Exception as e:
            print(f"文件读取错误: {e}")
            return [0.0] * 10
    
    test_args = [[20170101, "000001"]]
    
    with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as tmp_file:
        backup_file = tmp_file.name
    
    try:
        import rust_pyfunc
        
        print("开始文件IO测试...")
        results = rust_pyfunc.run_pools(
            file_io_func,
            test_args,
            backup_file=backup_file,
            num_threads=1
        )
        
        print(f"✅ 文件IO测试成功，结果数量: {len(results)}")
        return True
        
    except Exception as e:
        print(f"❌ 文件IO测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        if os.path.exists(backup_file):
            os.unlink(backup_file)

def test_simple_no_io():
    """测试不涉及文件IO的函数"""
    print("\n=== 测试不涉及文件IO的函数 ===")
    
    def no_io_func(date, code):
        import numpy as np
        import pandas as pd
        
        # 不涉及文件IO，只做计算
        data = pd.Series([1, 2, 3, 4, 5])
        return [data.mean(), data.std(), data.sum()]
    
    test_args = [[20170101, "000001"]]
    
    with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as tmp_file:
        backup_file = tmp_file.name
    
    try:
        import rust_pyfunc
        
        print("开始无IO测试...")
        results = rust_pyfunc.run_pools(
            no_io_func,
            test_args,
            backup_file=backup_file,
            num_threads=1
        )
        
        print(f"✅ 无IO测试成功，结果数量: {len(results)}")
        return True
        
    except Exception as e:
        print(f"❌ 无IO测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        if os.path.exists(backup_file):
            os.unlink(backup_file)

def test_exception_handling():
    """测试异常处理对子进程的影响"""
    print("\n=== 测试异常处理 ===")
    
    def exception_func(date, code):
        import numpy as np
        
        try:
            # 故意引发异常
            result = 1 / 0
            return [result]
        except Exception as e:
            # 这里的异常处理可能有问题
            print(f"子进程中捕获异常: {e}")
            raise  # 重新抛出异常
    
    test_args = [[20170101, "000001"]]
    
    with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as tmp_file:
        backup_file = tmp_file.name
    
    try:
        import rust_pyfunc
        
        print("开始异常处理测试...")
        results = rust_pyfunc.run_pools(
            exception_func,
            test_args,
            backup_file=backup_file,
            num_threads=1
        )
        
        print(f"结果: {results}")
        return True
        
    except Exception as e:
        print(f"✅ 异常处理测试符合预期: {e}")
        return True
        
    finally:
        if os.path.exists(backup_file):
            os.unlink(backup_file)

if __name__ == "__main__":
    print("开始分析子进程快速关闭问题...")
    
    test1_ok = test_simple_no_io()
    test2_ok = test_file_io_in_subprocess()
    test3_ok = test_exception_handling()
    
    print(f"\n=== 测试结果总结 ===")
    print(f"无IO测试: {'✅' if test1_ok else '❌'}")
    print(f"文件IO测试: {'✅' if test2_ok else '❌'}")
    print(f"异常处理测试: {'✅' if test3_ok else '❌'}")
    
    if test1_ok and not test2_ok:
        print("\n⚠️ 问题很可能出现在文件IO操作上！")
        print("建议检查：")
        print("1. 文件路径是否正确")
        print("2. 文件权限是否足够")
        print("3. 文件是否真实存在")
        print("4. pandas读取大文件是否导致内存问题")
    elif not test1_ok:
        print("\n⚠️ 问题可能更基础，连简单函数都无法执行")
    else:
        print("\n🤔 需要进一步分析其他差异")