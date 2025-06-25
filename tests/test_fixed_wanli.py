#!/usr/bin/env python3
"""
测试修复后的万里长征风格函数
"""

import os
import sys
import tempfile

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

def test_fixed_return_lengths():
    """测试修复后的返回长度一致性"""
    print("=== 测试修复后的返回长度一致性 ===")
    
    def fixed_wanli_func(date, code):
        import numpy as np
        
        # 模拟names列表（262个元素）
        names = [f"factor_{i}" for i in range(262)]
        
        try:
            if code == "000001":
                # 模拟文件不存在异常
                raise FileNotFoundError("文件不存在")
            else:
                # 正常情况，返回262个值
                res = [float(i) for i in range(262)]
                print(f"正常返回：res长度={len(res)}, names长度={len(names)}")
                # 确保返回长度与names一致
                if len(res) != len(names):
                    print(f"警告：res长度({len(res)})与names长度({len(names)})不一致")
                    if len(res) < len(names):
                        res.extend([np.nan] * (len(names) - len(res)))
                    else:
                        res = res[:len(names)]
                return res
        except Exception as e:
            print(f"异常返回：{e}")
            return [np.nan] * len(names)
    
    test_args = [
        [20170101, "000001"],  # 会异常
        [20170101, "000002"],  # 正常
        [20170101, "000003"],  # 正常
    ]
    
    with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as tmp_file:
        backup_file = tmp_file.name
    
    try:
        import rust_pyfunc
        
        print("开始修复后测试...")
        results = rust_pyfunc.run_pools(
            fixed_wanli_func,
            test_args,
            backup_file=backup_file,
            num_threads=3
        )
        
        print(f"✅ 修复后测试成功，结果数量: {len(results)}")
        
        # 检查所有结果长度是否一致
        lengths = [len(result) for result in results]
        print(f"所有结果长度: {lengths}")
        
        if len(set(lengths)) == 1:
            print("✅ 所有结果长度一致！")
            
            # 检查值的内容
            for i, result in enumerate(results):
                values = result[2:]  # 跳过date和code
                nan_count = sum(1 for x in values if np.isnan(x))
                finite_count = sum(1 for x in values if np.isfinite(x))
                print(f"结果{i}: 值长度={len(values)}, NaN={nan_count}, 有限值={finite_count}")
            
            return True
        else:
            print(f"❌ 结果长度不一致: {lengths}")
            return False
        
    except Exception as e:
        print(f"❌ 修复后测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        if os.path.exists(backup_file):
            os.unlink(backup_file)

def test_large_scale_fixed():
    """测试大规模修复后的函数"""
    print("\n=== 测试大规模修复后的函数 ===")
    
    def large_scale_func(date, code):
        import numpy as np
        
        names = [f"factor_{i}" for i in range(262)]
        
        try:
            # 模拟一些任务会失败
            if int(code) % 3 == 0:  # 每3个任务中有1个失败
                raise Exception("模拟失败")
            else:
                # 正常返回
                return [float(int(code) * i) for i in range(262)]
        except Exception as e:
            return [np.nan] * len(names)
    
    # 10个任务，其中一些会失败
    test_args = [[20170101, f"{i:06d}"] for i in range(10)]
    
    with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as tmp_file:
        backup_file = tmp_file.name
    
    try:
        import rust_pyfunc
        
        print("开始大规模测试...")
        results = rust_pyfunc.run_pools(
            large_scale_func,
            test_args,
            backup_file=backup_file,
            num_threads=10
        )
        
        print(f"✅ 大规模测试成功，结果数量: {len(results)}")
        
        # 检查结果
        lengths = [len(result) for result in results]
        print(f"长度统计: 最小={min(lengths)}, 最大={max(lengths)}, 唯一值={set(lengths)}")
        
        return len(set(lengths)) == 1  # 所有长度应该一致
        
    except Exception as e:
        print(f"❌ 大规模测试失败: {e}")
        return False
        
    finally:
        if os.path.exists(backup_file):
            os.unlink(backup_file)

if __name__ == "__main__":
    print("开始测试修复后的万里长征函数...")
    
    test1_ok = test_fixed_return_lengths()
    if test1_ok:
        test2_ok = test_large_scale_fixed()
    else:
        test2_ok = False
    
    print(f"\n=== 修复测试结果 ===")
    print(f"返回长度一致性: {'✅' if test1_ok else '❌'}")
    print(f"大规模测试: {'✅' if test2_ok else '❌'}")
    
    if test1_ok and test2_ok:
        print("\n🎉 修复成功！现在万里长征应该可以正常运行了")
        print("关键修复：")
        print("1. 确保正常情况下返回长度与names一致")
        print("2. 确保异常情况下返回长度与names一致")
        print("3. 所有返回结果长度都一致，避免NDArray形状错误")
    else:
        print("\n⚠️ 修复还不完善，需要进一步调试")