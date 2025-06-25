#!/usr/bin/env python3
"""
专门测试万里长征文件的特定问题
"""

import os
import sys
import tempfile
import numpy as np

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

def test_complex_data_processing():
    """测试1：复杂数据处理（模拟万里长征的数据结构）"""
    print("=== 测试1：复杂数据处理 ===")
    
    def complex_data_func(date, code):
        import numpy as np
        import pandas as pd
        import rust_pyfunc as rp
        
        try:
            # 模拟trade数据结构
            n_trades = 1000
            trade_data = {
                'exchtime': pd.date_range('09:30:00', periods=n_trades, freq='1S'),
                'price': np.random.uniform(10.0, 15.0, n_trades),
                'volume': np.random.randint(100, 10000, n_trades),
                'flag': np.random.choice([66, 83], n_trades)
            }
            trade = pd.DataFrame(trade_data)
            
            # 模拟asks/bids数据结构  
            n_orders = 500
            asks_data = {
                'exchtime': pd.date_range('09:30:00', periods=n_orders, freq='2S'),
                'price': np.random.uniform(10.1, 15.1, n_orders),
                'vol': np.random.randint(100, 5000, n_orders),
                'number': np.random.randint(0, 3, n_orders)
            }
            asks = pd.DataFrame(asks_data)
            bids_data = {
                'exchtime': pd.date_range('09:30:00', periods=n_orders, freq='2S'),
                'price': np.random.uniform(9.9, 14.9, n_orders),
                'vol': np.random.randint(100, 5000, n_orders),
                'number': np.random.randint(0, 3, n_orders)
            }
            bids = pd.DataFrame(bids_data)
            
            # 过滤数据（模拟万里长征的操作）
            asks = asks[asks.number < 3]
            bids = bids[bids.number < 3]
            
            if trade.shape[0] > 0 and asks.shape[0] > 0 and bids.shape[0] > 0:
                start_time = asks.exchtime.astype(np.int64)[0] / 1e9
                price_max = trade.price.max()
                price_min = trade.price.min()
                
                # 模拟rust函数调用（但使用简单的数据代替）
                trade_times = np.linspace(start_time, start_time + 3600, 100)
                trade_prices = np.random.uniform(price_min, price_max, 100)
                trade_volumes = np.random.uniform(100, 1000, 100)
                trade_flags = np.random.choice([66.0, 83.0], 100)
                
                asks_times = np.linspace(start_time, start_time + 3600, 50)
                asks_prices = np.random.uniform(price_min + 0.01, price_max + 0.01, 50)
                asks_volumes = np.random.uniform(100, 500, 50)
                
                bids_times = np.linspace(start_time, start_time + 3600, 50)
                bids_prices = np.random.uniform(price_min - 0.01, price_max - 0.01, 50)
                bids_volumes = np.random.uniform(100, 500, 50)
                
                # 调用rust函数（模拟analyze_retreat_advance_v2）
                nines_asks = rp.analyze_retreat_advance_v2(
                    trade_times, trade_prices, trade_volumes, trade_flags,
                    asks_times, asks_prices, asks_volumes,
                    80, 1, 0, 5, False
                )
                
                nines_bids = rp.analyze_retreat_advance_v2(
                    trade_times, trade_prices, trade_volumes, trade_flags,
                    bids_times, bids_prices, bids_volumes,
                    80, 1, 0, 5, True
                )
                
                # 处理结果（模拟万里长征的数据处理）
                asks_dict = {
                    'amount_dura': nines_asks[0],
                    'amount_hang': nines_asks[1], 
                    'amount_future': nines_asks[2],
                    'act_buy_rate': nines_asks[3],
                    'price_kinds': nines_asks[4],
                    'price_rate': nines_asks[5],
                    'dura_seconds': nines_asks[6],
                    'dura_starts': nines_asks[7],
                    'peak_price': nines_asks[8]
                }
                
                # 进行大量相关系数计算（模拟万里长征的逻辑）
                correlations = []
                keys = list(asks_dict.keys())
                for i, key1 in enumerate(keys):
                    for j, key2 in enumerate(keys[i+1:], i+1):
                        if len(asks_dict[key1]) > 1 and len(asks_dict[key2]) > 1:
                            try:
                                corr = np.corrcoef(asks_dict[key1], asks_dict[key2])[0, 1]
                                if not np.isnan(corr):
                                    correlations.append(corr)
                                else:
                                    correlations.append(0.0)
                            except Exception:
                                correlations.append(0.0)
                
                # 返回结果
                result = correlations[:50]  # 取前50个
                while len(result) < 50:
                    result.append(0.0)
                
                return result
            else:
                return [0.0] * 50
                
        except Exception as e:
            print(f"函数内部错误: {e}")
            return [0.0] * 50
    
    test_args = [[20240101, "TEST001"]]
    
    with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as tmp_file:
        backup_file = tmp_file.name
    
    try:
        import rust_pyfunc
        
        print("开始复杂数据处理测试...")
        results = rust_pyfunc.run_pools(
            complex_data_func,
            test_args,
            backup_file=backup_file,
            num_threads=2
        )
        
        print(f"✅ 复杂数据处理测试成功，结果数量: {len(results)}")
        return True
        
    except Exception as e:
        print(f"❌ 复杂数据处理测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        if os.path.exists(backup_file):
            os.unlink(backup_file)

def test_large_numpy_correlations():
    """测试2：大量numpy相关系数计算"""
    print("\n=== 测试2：大量numpy相关系数计算 ===")
    
    def massive_corr_func(date, code):
        import numpy as np
        
        # 生成大量数据进行相关系数计算
        data_arrays = []
        for i in range(20):  # 20个数组
            data_arrays.append(np.random.randn(100))
        
        correlations = []
        # 进行两两相关系数计算（类似万里长征中的逻辑）
        for i in range(len(data_arrays)):
            for j in range(i+1, len(data_arrays)):
                try:
                    corr = np.corrcoef(data_arrays[i], data_arrays[j])[0, 1]
                    if not np.isnan(corr):
                        correlations.append(corr)
                    else:
                        correlations.append(0.0)
                except Exception:
                    correlations.append(0.0)
        
        # 返回前100个相关系数
        result = correlations[:100]
        while len(result) < 100:
            result.append(0.0)
            
        return result
    
    test_args = [[20240101, "TEST001"]]
    
    with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as tmp_file:
        backup_file = tmp_file.name
    
    try:
        import rust_pyfunc
        
        print("开始大量相关系数计算测试...")
        results = rust_pyfunc.run_pools(
            massive_corr_func,
            test_args,
            backup_file=backup_file,
            num_threads=2
        )
        
        print(f"✅ 大量相关系数计算测试成功，结果数量: {len(results)}")
        return True
        
    except Exception as e:
        print(f"❌ 大量相关系数计算测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        if os.path.exists(backup_file):
            os.unlink(backup_file)

def test_function_source_size():
    """测试3：大型函数源代码"""
    print("\n=== 测试3：大型函数源代码 ===")
    
    def huge_function(date, code):
        import numpy as np
        import pandas as pd
        import rust_pyfunc as rp
        
        # 这是一个巨大的函数，模拟万里长征的函数大小
        # 变量定义（模拟大量变量）
        var1 = np.random.randn(50)
        var2 = np.random.randn(50) 
        var3 = np.random.randn(50)
        var4 = np.random.randn(50)
        var5 = np.random.randn(50)
        var6 = np.random.randn(50)
        var7 = np.random.randn(50)
        var8 = np.random.randn(50)
        var9 = np.random.randn(50)
        var10 = np.random.randn(50)
        
        # 大量计算（模拟万里长征的计算量）
        corr1 = np.corrcoef(var1, var2)[0, 1] if len(var1) > 1 else 0.0
        corr2 = np.corrcoef(var1, var3)[0, 1] if len(var1) > 1 else 0.0
        corr3 = np.corrcoef(var1, var4)[0, 1] if len(var1) > 1 else 0.0
        corr4 = np.corrcoef(var1, var5)[0, 1] if len(var1) > 1 else 0.0
        corr5 = np.corrcoef(var1, var6)[0, 1] if len(var1) > 1 else 0.0
        corr6 = np.corrcoef(var1, var7)[0, 1] if len(var1) > 1 else 0.0
        corr7 = np.corrcoef(var1, var8)[0, 1] if len(var1) > 1 else 0.0
        corr8 = np.corrcoef(var1, var9)[0, 1] if len(var1) > 1 else 0.0
        corr9 = np.corrcoef(var1, var10)[0, 1] if len(var1) > 1 else 0.0
        corr10 = np.corrcoef(var2, var3)[0, 1] if len(var2) > 1 else 0.0
        corr11 = np.corrcoef(var2, var4)[0, 1] if len(var2) > 1 else 0.0
        corr12 = np.corrcoef(var2, var5)[0, 1] if len(var2) > 1 else 0.0
        corr13 = np.corrcoef(var2, var6)[0, 1] if len(var2) > 1 else 0.0
        corr14 = np.corrcoef(var2, var7)[0, 1] if len(var2) > 1 else 0.0
        corr15 = np.corrcoef(var2, var8)[0, 1] if len(var2) > 1 else 0.0
        corr16 = np.corrcoef(var2, var9)[0, 1] if len(var2) > 1 else 0.0
        corr17 = np.corrcoef(var2, var10)[0, 1] if len(var2) > 1 else 0.0
        corr18 = np.corrcoef(var3, var4)[0, 1] if len(var3) > 1 else 0.0
        corr19 = np.corrcoef(var3, var5)[0, 1] if len(var3) > 1 else 0.0
        corr20 = np.corrcoef(var3, var6)[0, 1] if len(var3) > 1 else 0.0
        
        # 处理NaN值
        corrs = [corr1, corr2, corr3, corr4, corr5, corr6, corr7, corr8, corr9, corr10,
                corr11, corr12, corr13, corr14, corr15, corr16, corr17, corr18, corr19, corr20]
        
        result = []
        for corr in corrs:
            if np.isnan(corr) or np.isinf(corr):
                result.append(0.0)
            else:
                result.append(float(corr))
        
        # 确保返回固定长度
        while len(result) < 50:
            result.append(0.0)
            
        return result[:50]
    
    test_args = [[20240101, "TEST001"]]
    
    with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as tmp_file:
        backup_file = tmp_file.name
    
    try:
        import rust_pyfunc
        
        print("开始大型函数源代码测试...")
        results = rust_pyfunc.run_pools(
            huge_function,
            test_args,
            backup_file=backup_file,
            num_threads=2
        )
        
        print(f"✅ 大型函数源代码测试成功，结果数量: {len(results)}")
        return True
        
    except Exception as e:
        print(f"❌ 大型函数源代码测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        if os.path.exists(backup_file):
            os.unlink(backup_file)

if __name__ == "__main__":
    print("开始万里长征特定问题测试...")
    
    # 按顺序执行测试
    test1_ok = test_complex_data_processing()
    test2_ok = test_large_numpy_correlations()
    test3_ok = test_function_source_size()
    
    print(f"\n=== 测试结果总结 ===")
    print(f"复杂数据处理: {'✅' if test1_ok else '❌'}")
    print(f"大量相关系数计算: {'✅' if test2_ok else '❌'}")
    print(f"大型函数源代码: {'✅' if test3_ok else '❌'}")
    
    if not test1_ok:
        print("\n⚠️ 问题出现在复杂数据处理上")
    elif not test2_ok:
        print("\n⚠️ 问题出现在大量相关系数计算上")
    elif not test3_ok:
        print("\n⚠️ 问题出现在大型函数源代码上")
    else:
        print("\n🎉 万里长征相关测试都通过，需要进一步分析")