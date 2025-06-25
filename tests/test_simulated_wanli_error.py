#!/usr/bin/env python3
"""
模拟万里长征go函数来测试错误报告
"""

import os
import sys
import tempfile
import numpy as np

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'python'))

def simulated_wanli_go(date, code):
    """
    模拟万里长征go函数的行为和可能的错误
    基于万里长征脚本的names列表，应该返回1066个值
    """
    # 模拟万里长征的names列表长度（1066个因子）
    expected_length = 1066
    
    try:
        # 模拟各种可能的错误情况
        if code == "000001":
            # 模拟文件不存在错误
            raise FileNotFoundError(f"/ssd_data/stock/{date}/transaction/{code}_{date}_transaction.csv")
        
        elif code == "000002":
            # 模拟数据处理异常
            raise ValueError("数据格式错误：无法解析交易数据")
        
        elif code == "000003":
            # 模拟内存错误
            raise MemoryError("内存不足，无法处理大型数据集")
        
        elif code == "000004":
            # 模拟计算错误（如除零错误）
            raise ZeroDivisionError("计算过程中出现除零错误")
        
        elif code == "000005":
            # 模拟返回长度不正确的情况（这是万里长征的核心问题）
            return [1.0, 2.0, 3.0]  # 只返回3个值而不是1066个
        
        elif code == "000006":
            # 模拟部分计算成功但长度不够
            return [float(i) for i in range(500)]  # 只返回500个值
        
        else:
            # 正常情况，返回完整的1066个值
            return [float(i % 100) for i in range(expected_length)]
            
    except Exception as e:
        print(f"模拟万里长征go函数中捕获到异常: {e}")
        # 这里模拟万里长征原始代码的错误处理
        # 原始代码可能直接抛出异常，而不是返回NaN列表
        raise e  # 直接抛出，这会导致进程错误

def test_simulated_wanli_errors():
    """测试模拟的万里长征错误"""
    print("=== 测试模拟万里长征错误 ===")
    
    # 测试各种错误情况
    test_args = [
        [20170101, "000001"],  # 文件不存在
        [20170101, "000002"],  # 数据格式错误
        [20170101, "000003"],  # 内存错误
        [20170101, "000004"],  # 除零错误
        [20170101, "000005"],  # 返回长度错误
        [20170101, "000006"],  # 部分长度错误
        [20170101, "000007"],  # 正常情况
    ]
    
    with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as tmp_file:
        backup_file = tmp_file.name
    
    try:
        import rust_pyfunc
        
        print("开始模拟万里长征错误测试...")
        print("这应该触发各种错误并显示详细的错误信息...")
        
        results = rust_pyfunc.run_pools(
            simulated_wanli_go,
            test_args,
            backup_file=backup_file,
            num_threads=7
        )
        
        print(f"✅ 测试完成，结果数量: {len(results)}")
        
        # 检查结果
        for i, result in enumerate(results):
            if result:
                print(f"结果{i}: 长度={len(result)}")
            else:
                print(f"结果{i}: 空结果")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        print("\n🎯 这正是万里长征的问题！现在可以看到详细错误了：")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        if os.path.exists(backup_file):
            os.unlink(backup_file)

if __name__ == "__main__":
    print("开始模拟万里长征错误测试...")
    print("这将模拟万里长征脚本中可能出现的各种错误情况")
    print("包括文件不存在、数据错误、返回长度不一致等问题")
    print()
    
    success = test_simulated_wanli_errors()
    print(f"\n测试结果: {'✅ 成功' if success else '❌ 失败'}")
    
    if not success:
        print("\n📋 关键发现：")
        print("1. 错误报告系统正常工作，能显示详细错误信息")
        print("2. 万里长征的问题可能是：")
        print("   - 异常处理方式不当（直接抛出而不是返回固定长度）")
        print("   - 返回数组长度不一致")
        print("   - 某些股票数据文件不存在或格式错误")
        print("3. 解决方案：修改万里长征go函数的异常处理")