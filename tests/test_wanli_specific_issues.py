#!/usr/bin/env python3
"""
测试万里长征特定问题
"""

import os
import sys
import tempfile

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

def test_file_path_access():
    """测试文件路径访问问题"""
    print("=== 测试文件路径访问 ===")
    
    def file_access_func(date, code):
        import os
        import pandas as pd
        
        # 模拟万里长征的文件读取
        file_name = "%s_%d_%s.csv" % (code, date, "transaction")
        file_path = os.path.join("/ssd_data/stock", str(date), "transaction", file_name)
        
        print(f"尝试访问文件: {file_path}")
        
        # 检查路径是否存在
        if not os.path.exists(os.path.dirname(file_path)):
            print(f"目录不存在: {os.path.dirname(file_path)}")
            return [0.0] * 263
            
        if not os.path.exists(file_path):
            print(f"文件不存在: {file_path}")
            return [0.0] * 263
            
        try:
            # 尝试读取文件
            df = pd.read_csv(file_path)
            print(f"文件读取成功，行数: {len(df)}")
            return [1.0] * 263
        except Exception as e:
            print(f"文件读取失败: {e}")
            return [0.0] * 263
    
    test_args = [[20170101, "000001"]]
    
    with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as tmp_file:
        backup_file = tmp_file.name
    
    try:
        import rust_pyfunc
        
        print("开始文件路径访问测试...")
        results = rust_pyfunc.run_pools(
            file_access_func,
            test_args,
            backup_file=backup_file,
            num_threads=1
        )
        
        print(f"✅ 文件路径访问测试完成，结果数量: {len(results)}")
        return True
        
    except Exception as e:
        print(f"❌ 文件路径访问测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        if os.path.exists(backup_file):
            os.unlink(backup_file)

def test_large_return_array():
    """测试大型返回数组"""
    print("\n=== 测试大型返回数组 ===")
    
    def large_array_func(date, code):
        import numpy as np
        
        # 返回263个值，模拟万里长征
        results = []
        
        # 生成大量数据
        for i in range(263):
            if i % 10 == 0:
                results.append(np.nan)
            else:
                results.append(float(i))
        
        return results
    
    test_args = [[20170101, "000001"]]
    
    with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as tmp_file:
        backup_file = tmp_file.name
    
    try:
        import rust_pyfunc
        
        print("开始大型数组测试...")
        results = rust_pyfunc.run_pools(
            large_array_func,
            test_args,
            backup_file=backup_file,
            num_threads=1
        )
        
        print(f"✅ 大型数组测试成功，结果数量: {len(results)}")
        if len(results) > 0:
            print(f"返回值长度: {len(results[0])}")
        return True
        
    except Exception as e:
        print(f"❌ 大型数组测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        if os.path.exists(backup_file):
            os.unlink(backup_file)

def test_early_exit_detection():
    """测试是否能检测到早期退出"""
    print("\n=== 测试早期退出检测 ===")
    
    def early_exit_func(date, code):
        import sys
        print(f"子进程开始执行: date={date}, code={code}")
        
        # 可能导致早期退出的操作
        if code == "000001":
            print("检测到code=000001，可能会有问题")
            # 这里不做任何危险操作，只是标记
        
        print("子进程执行完成")
        return [1.0, 2.0, 3.0]
    
    test_args = [[20170101, "000001"]]
    
    with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as tmp_file:
        backup_file = tmp_file.name
    
    try:
        import rust_pyfunc
        
        print("开始早期退出检测测试...")
        results = rust_pyfunc.run_pools(
            early_exit_func,
            test_args,
            backup_file=backup_file,
            num_threads=1
        )
        
        print(f"✅ 早期退出检测测试成功，结果数量: {len(results)}")
        return True
        
    except Exception as e:
        print(f"❌ 早期退出检测测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        if os.path.exists(backup_file):
            os.unlink(backup_file)

if __name__ == "__main__":
    print("开始万里长征特定问题测试...")
    
    test1_ok = test_early_exit_detection()
    test2_ok = test_large_return_array()
    test3_ok = test_file_path_access()
    
    print(f"\n=== 测试结果总结 ===")
    print(f"早期退出检测: {'✅' if test1_ok else '❌'}")
    print(f"大型返回数组: {'✅' if test2_ok else '❌'}")
    print(f"文件路径访问: {'✅' if test3_ok else '❌'}")
    
    if not test3_ok:
        print("\n⚠️ 问题很可能出现在文件路径访问上！")
        print("可能的原因：")
        print("1. /ssd_data/stock 目录不存在或无权限")
        print("2. 文件路径格式错误")
        print("3. 文件不存在但代码没有正确处理")
    elif not test2_ok:
        print("\n⚠️ 问题可能与大型返回数组相关")
    elif not test1_ok:
        print("\n⚠️ 基础通信有问题")
    else:
        print("\n✅ 所有测试都通过，问题可能在其他地方")