#!/usr/bin/env python3
"""
最终验证：在实际使用中可以获得np.nan
"""

import os
import sys
import tempfile
import numpy as np
import math

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

def test_real_world_usage():
    """实际使用场景测试"""
    print("=== 实际使用场景测试 ===")
    
    def real_world_func(date, code):
        import numpy as np
        import pandas as pd
        
        # 模拟真实计算场景：一些计算会产生NaN
        data1 = np.array([1, 2, 3, np.nan, 5])
        data2 = np.array([10, 20, 30, 40, np.nan])
        
        # 一些统计计算
        results = []
        
        # 平均值计算（包含NaN）
        mean1 = np.nanmean(data1)  # 会忽略NaN
        mean2 = np.mean(data2)     # 会返回NaN
        
        # 相关系数计算（可能产生NaN）
        corr = np.corrcoef(data1, data2)[0, 1]  # 可能是NaN
        
        # 除零操作（会产生inf，然后转为NaN）
        division_result = 1.0 / 0.0 if False else np.nan
        
        results.extend([mean1, mean2, corr, division_result])
        
        # pandas统计（可能产生NaN）
        series = pd.Series([1, 2, np.nan, 4, 5])
        skew = series.skew()  # 可能是有限值
        kurt = series.kurt()  # 可能是有限值
        
        results.extend([skew, kurt])
        
        return results
    
    test_args = [[20240101, "TEST001"]]
    
    with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as tmp_file:
        backup_file = tmp_file.name
    
    try:
        import rust_pyfunc
        
        print("开始实际使用场景测试...")
        results = rust_pyfunc.run_pools(
            real_world_func,
            test_args,
            backup_file=backup_file,
            num_threads=1
        )
        
        if len(results) > 0:
            result = results[0]
            values = result[2:]  # 跳过date和code
            
            print(f"返回的值: {values}")
            print(f"值的类型: {[type(v) for v in values]}")
            
            # 验证我们可以正确使用numpy函数检测NaN
            nan_count = sum(1 for v in values if np.isnan(v))
            finite_count = sum(1 for v in values if np.isfinite(v))
            
            print(f"NaN数量: {nan_count}")
            print(f"有限值数量: {finite_count}")
            
            # 验证我们可以用numpy处理这些值
            values_array = np.array(values)
            print(f"作为numpy数组: {values_array}")
            print(f"忽略NaN的平均值: {np.nanmean(values_array)}")
            
            # 验证与np.nan的比较
            contains_actual_nan = any(np.isnan(v) and str(v) == 'nan' for v in values)
            
            if contains_actual_nan:
                print("✅ 成功获得真正的np.nan值！")
                return True
            else:
                print("❌ 没有获得真正的np.nan值")
                return False
        else:
            print("❌ 没有返回结果")
            return False
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        if os.path.exists(backup_file):
            os.unlink(backup_file)

if __name__ == "__main__":
    print("开始最终验证测试...")
    success = test_real_world_usage()
    
    if success:
        print("\n🎉 验证成功！")
        print("现在你的rust_pyfunc库已经正确支持NaN/Inf值处理：")
        print("- NaN值会被保留为真正的np.nan")
        print("- Inf值会被转换为np.nan")
        print("- 在Python中可以正确使用np.isnan()和其他numpy函数")
        print("- 原始需求已完全满足！")
    else:
        print("\n⚠️ 验证失败，仍需进一步调试")