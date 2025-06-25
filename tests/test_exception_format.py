#!/usr/bin/env python3
"""
测试异常情况下的数据格式问题
"""

import os
import sys
import tempfile

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

def test_consistent_return_format():
    """测试异常时的一致返回格式"""
    print("=== 测试一致返回格式 ===")
    
    def format_test_func(date, code):
        if code == "000001":
            # 第一个任务抛出异常，但我们在worker_process.py中会捕获并处理
            raise Exception("模拟异常")
        else:
            # 正常任务返回固定格式
            return [1.0, 2.0, 3.0]
    
    test_args = [
        [20170101, "000001"],  # 会异常
        [20170101, "000002"],  # 正常
        [20170101, "000003"],  # 正常
    ]
    
    with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as tmp_file:
        backup_file = tmp_file.name
    
    try:
        import rust_pyfunc
        
        print("开始格式测试...")
        results = rust_pyfunc.run_pools(
            format_test_func,
            test_args,
            backup_file=backup_file,
            num_threads=3
        )
        
        print(f"✅ 格式测试成功，结果数量: {len(results)}")
        
        # 检查结果格式
        for i, result in enumerate(results):
            print(f"结果{i}: 长度={len(result)}, 前5个值={result[:5]}")
        
        # 检查长度是否一致
        lengths = [len(result) for result in results]
        if len(set(lengths)) == 1:
            print("✅ 所有结果长度一致")
        else:
            print(f"❌ 结果长度不一致: {lengths}")
        
        return True
        
    except Exception as e:
        print(f"❌ 格式测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        if os.path.exists(backup_file):
            os.unlink(backup_file)

def test_wanli_exception_format():
    """测试万里长征式异常格式"""
    print("\n=== 测试万里长征式异常格式 ===")
    
    def wanli_exception_func(date, code):
        import numpy as np
        
        try:
            if code == "000001":
                # 模拟文件不存在异常
                raise FileNotFoundError("文件不存在")
            else:
                # 正常情况返回263个值
                return [float(i) for i in range(263)]
        except Exception as e:
            # 像万里长征一样返回263个NaN
            return [np.nan] * 263
    
    test_args = [
        [20170101, "000001"],  # 会异常
        [20170101, "000002"],  # 正常
    ]
    
    with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as tmp_file:
        backup_file = tmp_file.name
    
    try:
        import rust_pyfunc
        
        print("开始万里长征式异常测试...")
        results = rust_pyfunc.run_pools(
            wanli_exception_func,
            test_args,
            backup_file=backup_file,
            num_threads=2
        )
        
        print(f"✅ 万里长征式异常测试成功，结果数量: {len(results)}")
        
        # 检查结果
        for i, result in enumerate(results):
            values = result[2:]  # 跳过date和code
            nan_count = sum(1 for x in values if np.isnan(x))
            print(f"结果{i}: 总长度={len(result)}, 值长度={len(values)}, NaN数量={nan_count}")
        
        return True
        
    except Exception as e:
        print(f"❌ 万里长征式异常测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        if os.path.exists(backup_file):
            os.unlink(backup_file)

def test_mixed_return_lengths():
    """测试混合返回长度"""
    print("\n=== 测试混合返回长度 ===")
    
    def mixed_length_func(date, code):
        if code == "000001":
            return [1.0]  # 短结果
        elif code == "000002":
            return [1.0, 2.0, 3.0]  # 中等结果
        else:
            return [1.0, 2.0, 3.0, 4.0, 5.0]  # 长结果
    
    test_args = [
        [20170101, "000001"],
        [20170101, "000002"],
        [20170101, "000003"],
    ]
    
    with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as tmp_file:
        backup_file = tmp_file.name
    
    try:
        import rust_pyfunc
        
        print("开始混合长度测试...")
        results = rust_pyfunc.run_pools(
            mixed_length_func,
            test_args,
            backup_file=backup_file,
            num_threads=3
        )
        
        print(f"混合长度测试结果: {len(results)}")
        for i, result in enumerate(results):
            print(f"结果{i}: 长度={len(result)}")
        
        return True
        
    except Exception as e:
        print(f"❌ 混合长度测试失败: {e}")
        print("这可能就是万里长征问题的根源！")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        if os.path.exists(backup_file):
            os.unlink(backup_file)

if __name__ == "__main__":
    print("开始测试异常格式问题...")
    
    test1_ok = test_consistent_return_format()
    test2_ok = test_wanli_exception_format()
    test3_ok = test_mixed_return_lengths()
    
    print(f"\n=== 异常格式测试结果 ===")
    print(f"一致返回格式: {'✅' if test1_ok else '❌'}")
    print(f"万里长征式异常: {'✅' if test2_ok else '❌'}")
    print(f"混合返回长度: {'✅' if test3_ok else '❌'}")
    
    if not test3_ok:
        print("\n🎯 找到问题！混合返回长度导致NDArray形状错误！")
        print("解决方案：确保所有函数返回相同长度的结果")
    elif not test1_ok or not test2_ok:
        print("\n⚠️ 异常处理格式有问题")
    else:
        print("\n🤔 格式问题不是主因，需要进一步分析")