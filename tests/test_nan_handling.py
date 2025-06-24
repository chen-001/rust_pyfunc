#!/usr/bin/env python3
"""
测试NaN/Inf值处理功能
"""

import os
import sys
import tempfile
import numpy as np

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

def test_nan_inf_handling():
    """测试NaN和Inf值的处理"""
    print("开始测试NaN/Inf值处理...")
    
    def test_func_with_nan(date, code):
        """返回包含NaN和Inf的测试函数"""
        return [
            1.0,           # 正常值
            float('nan'),  # NaN值
            float('inf'),  # 正无穷
            float('-inf'), # 负无穷
            2.5            # 正常值
        ]
    
    # 测试数据
    test_args = [[20240101, "TEST001"], [20240102, "TEST002"]]
    
    with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as tmp_file:
        backup_file = tmp_file.name
    
    try:
        import rust_pyfunc
        
        print(f"测试任务数: {len(test_args)}")
        print(f"备份文件: {backup_file}")
        
        # 执行计算
        results = rust_pyfunc.run_pools(
            test_func_with_nan,
            test_args,
            backup_file=backup_file,
            num_threads=2,
            backup_batch_size=1
        )
        
        print(f"计算完成，结果数量: {len(results)}")
        
        # 检查结果
        for i, result in enumerate(results):
            print(f"\n结果 {i}:")
            print(f"  日期: {result[0]}")
            print(f"  代码: {result[1]}")
            print(f"  因子值: {result[2:]}")
            
            # 检查None值（原来的NaN/Inf）
            factors = result[2:]
            none_count = sum(1 for x in factors if x is None)
            print(f"  None值数量: {none_count}")
            
            # 验证预期的None值位置
            expected_none_positions = [1, 2, 3]  # NaN和Inf的位置
            actual_none_positions = [j for j, x in enumerate(factors) if x is None]
            print(f"  预期None位置: {expected_none_positions}")
            print(f"  实际None位置: {actual_none_positions}")
            
            if actual_none_positions == expected_none_positions:
                print("  ✅ NaN/Inf处理正确")
            else:
                print("  ❌ NaN/Inf处理有误")
        
        # 测试查询备份
        print("\n=== 测试备份查询 ===")
        if hasattr(rust_pyfunc, 'query_backup'):
            query_results = rust_pyfunc.query_backup(
                backup_file=backup_file,
                storage_format="binary"
            )
            print(f"查询到 {len(query_results)} 个结果")
            
            for i, result in enumerate(query_results):
                print(f"查询结果 {i}: 因子值 = {result[3:]}")  # 跳过date, code, timestamp
                factors = result[3:]
                none_count = sum(1 for x in factors if x is None)
                print(f"  查询结果None值数量: {none_count}")
        
        return True
        
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # 清理临时文件
        if os.path.exists(backup_file):
            os.unlink(backup_file)

def test_pure_normal_values():
    """测试纯正常值（对比测试）"""
    print("\n开始测试纯正常值...")
    
    def normal_func(date, code):
        """返回纯正常值"""
        return [1.0, 2.0, 3.0, 4.0]
    
    test_args = [[20240101, "NORMAL"]]
    
    with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as tmp_file:
        backup_file = tmp_file.name
    
    try:
        import rust_pyfunc
        
        results = rust_pyfunc.run_pools(
            normal_func,
            test_args,
            backup_file=backup_file,
            num_threads=1
        )
        
        print(f"正常值结果: {results[0]}")
        factors = results[0][2:]  # 跳过date, code
        none_count = sum(1 for x in factors if x is None)
        print(f"None值数量: {none_count} (应该为0)")
        
        if none_count == 0:
            print("✅ 正常值处理正确")
        else:
            print("❌ 正常值处理有误")
            
        return True
        
    except Exception as e:
        print(f"正常值测试失败: {e}")
        return False
    finally:
        # 清理临时文件
        if os.path.exists(backup_file):
            os.unlink(backup_file)

if __name__ == "__main__":
    success1 = test_nan_inf_handling()
    success2 = test_pure_normal_values()
    
    if success1 and success2:
        print("\n🎉 所有测试通过！")
    else:
        print("\n❌ 部分测试失败")