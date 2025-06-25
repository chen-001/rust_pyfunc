#!/usr/bin/env python3
"""
最小化对比测试
"""

import os
import sys
import tempfile

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

def test_pcm_style():
    """模拟pcm.py的风格"""
    print("=== 测试PCM风格 ===")
    
    def pcm_style_func(date, code):
        import numpy as np
        res = [float(i) for i in range(100)] + [np.nan] * 50
        return res
    
    test_args = [[20170101, "000001"]]
    
    with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as tmp_file:
        backup_file = tmp_file.name
    
    try:
        import rust_pyfunc
        
        print("开始PCM风格测试...")
        results = rust_pyfunc.run_pools(
            pcm_style_func,
            test_args,
            backup_file=backup_file,
            num_threads=1
        )
        
        print(f"✅ PCM风格测试成功，结果数量: {len(results)}")
        if len(results) > 0:
            print(f"结果长度: {len(results[0])}")
        return True
        
    except Exception as e:
        print(f"❌ PCM风格测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        if os.path.exists(backup_file):
            os.unlink(backup_file)

def test_wanli_style_simple():
    """模拟万里长征的风格但简化"""
    print("\n=== 测试万里长征风格(简化) ===")
    
    def wanli_style_func(date, code):
        import numpy as np
        import pandas as pd
        
        # 模拟万里长征的结构但不做实际文件读取
        try:
            # 模拟数据处理
            data1 = np.random.randn(100)
            data2 = np.random.randn(100)
            
            # 模拟相关系数计算
            corr = np.corrcoef(data1, data2)[0, 1]
            
            # 模拟pandas操作
            series1 = pd.Series(data1)
            series2 = pd.Series(data2)
            
            results = [
                corr,
                series1.mean(),
                series1.std(),
                series2.mean(),
                series2.std()
            ]
            
            # 返回263个值(万里长征的返回数量)
            while len(results) < 263:
                results.append(0.0)
                
            return results[:263]
            
        except Exception as e:
            return [np.nan] * 263
    
    test_args = [[20170101, "000001"]]
    
    with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as tmp_file:
        backup_file = tmp_file.name
    
    try:
        import rust_pyfunc
        
        print("开始万里长征风格测试...")
        results = rust_pyfunc.run_pools(
            wanli_style_func,
            test_args,
            backup_file=backup_file,
            num_threads=1
        )
        
        print(f"✅ 万里长征风格测试成功，结果数量: {len(results)}")
        if len(results) > 0:
            print(f"结果长度: {len(results[0])}")
        return True
        
    except Exception as e:
        print(f"❌ 万里长征风格测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        if os.path.exists(backup_file):
            os.unlink(backup_file)

if __name__ == "__main__":
    print("开始最小化对比测试...")
    
    test1_ok = test_pcm_style()
    if test1_ok:
        test2_ok = test_wanli_style_simple()
    else:
        test2_ok = False
    
    print(f"\n=== 对比结果 ===")
    print(f"PCM风格: {'✅' if test1_ok else '❌'}")
    print(f"万里长征风格(简化): {'✅' if test2_ok else '❌'}")
    
    if test1_ok and test2_ok:
        print("\n✅ 两种风格都可以工作，问题可能在于具体的实现细节")
        print("需要进一步检查万里长征文件中的具体操作")
    elif test1_ok and not test2_ok:
        print("\n⚠️ 万里长征风格存在问题，可能是：")
        print("1. 返回值数量过多(263个)")
        print("2. 复杂的numpy/pandas操作")
        print("3. 异常处理方式")
    else:
        print("\n❌ 基础功能都有问题，需要检查rust_pyfunc的基本运行")