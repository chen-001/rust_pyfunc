#!/usr/bin/env python3
"""
缩小范围测试万里长征问题
"""

import os
import sys
import tempfile
import numpy as np

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

def test_rust_function_only():
    """测试1：仅调用rust函数"""
    print("=== 测试1：仅调用rust函数 ===")
    
    def rust_only_func(date, code):
        import rust_pyfunc as rp
        import numpy as np
        
        try:
            # 准备简单的测试数据
            trade_times = np.linspace(1000000000, 1000003600, 20)  # 20个点
            trade_prices = np.random.uniform(10.0, 15.0, 20)
            trade_volumes = np.random.uniform(100, 1000, 20)
            trade_flags = np.random.choice([66.0, 83.0], 20)
            
            asks_times = np.linspace(1000000000, 1000003600, 10)  # 10个点
            asks_prices = np.random.uniform(10.1, 15.1, 10)
            asks_volumes = np.random.uniform(100, 500, 10)
            
            # 调用rust函数
            result = rp.analyze_retreat_advance_v2(
                trade_times, trade_prices, trade_volumes, trade_flags,
                asks_times, asks_prices, asks_volumes,
                80, 1, 0, 5, False
            )
            
            return result[:5]  # 只返回前5个结果
            
        except Exception as e:
            print(f"rust函数调用错误: {e}")
            return [0.0] * 5
    
    test_args = [[20240101, "TEST001"]]
    
    with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as tmp_file:
        backup_file = tmp_file.name
    
    try:
        import rust_pyfunc
        
        print("开始rust函数测试...")
        results = rust_pyfunc.run_pools(
            rust_only_func,
            test_args,
            backup_file=backup_file,
            num_threads=2
        )
        
        print(f"✅ rust函数测试成功，结果数量: {len(results)}")
        return True
        
    except Exception as e:
        print(f"❌ rust函数测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        if os.path.exists(backup_file):
            os.unlink(backup_file)

def test_pandas_data_processing():
    """测试2：pandas数据处理"""
    print("\n=== 测试2：pandas数据处理 ===")
    
    def pandas_processing_func(date, code):
        import numpy as np
        import pandas as pd
        
        try:
            # 创建类似万里长征的数据结构
            n_trades = 1000
            trade_data = {
                'exchtime': pd.date_range('09:30:00', periods=n_trades, freq='1S'),
                'price': np.random.uniform(10.0, 15.0, n_trades),
                'volume': np.random.randint(100, 10000, n_trades),
                'flag': np.random.choice([66, 83], n_trades)
            }
            trade = pd.DataFrame(trade_data)
            
            # 进行一些pandas操作
            trade = trade.set_index("exchtime").between_time("09:30:00", "14:57:00").reset_index()
            
            # 基本统计
            price_max = trade.price.max()
            price_min = trade.price.min()
            volume_mean = trade.volume.mean()
            
            return [price_max, price_min, volume_mean]
            
        except Exception as e:
            print(f"pandas处理错误: {e}")
            return [0.0, 0.0, 0.0]
    
    test_args = [[20240101, "TEST001"]]
    
    with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as tmp_file:
        backup_file = tmp_file.name
    
    try:
        import rust_pyfunc
        
        print("开始pandas数据处理测试...")
        results = rust_pyfunc.run_pools(
            pandas_processing_func,
            test_args,
            backup_file=backup_file,
            num_threads=2
        )
        
        print(f"✅ pandas数据处理测试成功，结果数量: {len(results)}")
        return True
        
    except Exception as e:
        print(f"❌ pandas数据处理测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        if os.path.exists(backup_file):
            os.unlink(backup_file)

def test_pure_ocean_breeze_import():
    """测试3：pure_ocean_breeze导入"""
    print("\n=== 测试3：pure_ocean_breeze导入 ===")
    
    def pure_ocean_func(date, code):
        try:
            import pure_ocean_breeze.jason as p
            # 仅导入，不实际读取数据
            return [1.0, 2.0, 3.0]
            
        except Exception as e:
            print(f"pure_ocean_breeze导入错误: {e}")
            return [0.0, 0.0, 0.0]
    
    test_args = [[20240101, "TEST001"]]
    
    with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as tmp_file:
        backup_file = tmp_file.name
    
    try:
        import rust_pyfunc
        
        print("开始pure_ocean_breeze导入测试...")
        results = rust_pyfunc.run_pools(
            pure_ocean_func,
            test_args,
            backup_file=backup_file,
            num_threads=2
        )
        
        print(f"✅ pure_ocean_breeze导入测试成功，结果数量: {len(results)}")
        return True
        
    except Exception as e:
        print(f"❌ pure_ocean_breeze导入测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        if os.path.exists(backup_file):
            os.unlink(backup_file)

if __name__ == "__main__":
    print("开始缩小范围测试...")
    
    # 按顺序执行测试
    test1_ok = test_rust_function_only()
    if test1_ok:
        test2_ok = test_pandas_data_processing()
        if test2_ok:
            test3_ok = test_pure_ocean_breeze_import()
        else:
            test3_ok = False
    else:
        test2_ok = False
        test3_ok = False
    
    print(f"\n=== 测试结果总结 ===")
    print(f"rust函数调用: {'✅' if test1_ok else '❌'}")
    print(f"pandas数据处理: {'✅' if test2_ok else '❌'}")
    print(f"pure_ocean_breeze导入: {'✅' if test3_ok else '❌'}")
    
    if not test1_ok:
        print("\n⚠️ 问题出现在rust函数调用上！")
    elif not test2_ok:
        print("\n⚠️ 问题出现在pandas数据处理上！")
    elif not test3_ok:
        print("\n⚠️ 问题出现在pure_ocean_breeze导入上！")
    else:
        print("\n🤔 需要更详细的测试来定位问题")