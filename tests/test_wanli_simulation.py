#!/usr/bin/env python3
"""
模拟万里长征的实际运行
"""

import os
import sys
import tempfile
import time

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

def create_wanli_like_function():
    """创建类似万里长征的函数"""
    
    def go(date, code):
        import os
        import rust_pyfunc as rp
        import numpy as np
        import pandas as pd

        def adjust_afternoon(df: pd.DataFrame) -> pd.DataFrame:
            if df.index.name=='exchtime':
                df1=df.between_time('09:00:00','11:30:00')
                df2=df.between_time('13:00:00','15:00:00')
                df2.index=df2.index-pd.Timedelta(minutes=90)
                df=pd.concat([df1,df2])
            elif 'exchtime' in df.columns:
                df1=df.set_index('exchtime').between_time('09:00:00','11:30:00')
                df2=df.set_index('exchtime').between_time('13:00:00','15:00:00')
                df2.index=df2.index-pd.Timedelta(minutes=90)
                df=pd.concat([df1,df2]).reset_index()
            return df

        def read_trade(symbol:str, date:int,with_retreat:int=0)->pd.DataFrame:
            file_name = "%s_%d_%s.csv" % (symbol, date, "transaction")
            file_path = os.path.join("/ssd_data/stock", str(date), "transaction", file_name)
            
            # 检查文件是否存在
            if not os.path.exists(file_path):
                print(f"文件不存在: {file_path}")
                raise FileNotFoundError(f"File not found: {file_path}")
                
            df = pd.read_csv(
                file_path,
                dtype={"symbol": str},
                usecols=[
                    "exchtime",
                    "price",
                    "volume",
                    "turnover",
                    "flag",
                    "index", 
                    "localtime",
                    "ask_order",
                    "bid_order",
                ],
            )
            if not with_retreat:
                df=df[df.flag!=32]
            df.exchtime=pd.to_timedelta(df.exchtime/1e6,unit='s')+pd.Timestamp('1970-01-01 08:00:00')
            return df

        def read_market_pair(symbol:str, date:int):
            # 简化版本，直接抛出异常来模拟文件不存在
            raise FileNotFoundError(f"Market data file not found for {symbol} on {date}")

        try:
            print(f"开始处理 {code} on {date}")
            trade = read_trade(code, date, with_retreat=0)
            asks, bids = read_market_pair(code, date)
            
            # 这里永远不会执行到，因为文件不存在
            return [1.0] * 263
            
        except Exception as e:
            print(f"处理异常: {e}")
            return [np.nan] * 263
    
    return go

def test_multi_process_with_exceptions():
    """测试多进程处理异常情况"""
    print("=== 测试多进程异常处理 ===")
    
    go_func = create_wanli_like_function()
    
    # 模拟多个任务
    test_args = [
        [20170101, "000001"],
        [20170101, "000002"], 
        [20170101, "000003"],
        [20170101, "000004"],
        [20170101, "000005"]
    ]
    
    with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as tmp_file:
        backup_file = tmp_file.name
    
    try:
        import rust_pyfunc
        
        print("开始多进程异常处理测试...")
        print(f"测试任务数量: {len(test_args)}")
        print("启动多进程...")
        
        start_time = time.time()
        
        results = rust_pyfunc.run_pools(
            go_func,
            test_args,
            backup_file=backup_file,
            num_threads=50  # 使用50个线程，模拟万里长征
        )
        
        end_time = time.time()
        
        print(f"✅ 多进程异常处理测试完成")
        print(f"耗时: {end_time - start_time:.2f}秒")
        print(f"结果数量: {len(results)}")
        
        for i, result in enumerate(results):
            nan_count = sum(1 for x in result[2:] if np.isnan(x))
            print(f"任务{i}: {len(result)}个值, {nan_count}个NaN")
        
        return True
        
    except Exception as e:
        print(f"❌ 多进程异常处理测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        if os.path.exists(backup_file):
            os.unlink(backup_file)

def test_process_lifecycle():
    """测试进程生命周期"""
    print("\n=== 测试进程生命周期 ===")
    
    def lifecycle_func(date, code):
        import time
        import os
        print(f"进程 {os.getpid()} 开始处理 {code}")
        
        # 模拟一些处理时间
        time.sleep(0.1)
        
        if code in ["000001", "000002"]:
            # 模拟前几个任务出错
            raise Exception(f"模拟处理 {code} 时出错")
        
        print(f"进程 {os.getpid()} 完成处理 {code}")
        return [1.0, 2.0, 3.0]
    
    test_args = [
        [20170101, "000001"],  # 会出错
        [20170101, "000002"],  # 会出错
        [20170101, "000003"],  # 正常
        [20170101, "000004"],  # 正常
    ]
    
    with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as tmp_file:
        backup_file = tmp_file.name
    
    try:
        import rust_pyfunc
        
        print("开始进程生命周期测试...")
        results = rust_pyfunc.run_pools(
            lifecycle_func,
            test_args,
            backup_file=backup_file,
            num_threads=4
        )
        
        print(f"✅ 进程生命周期测试完成，结果数量: {len(results)}")
        return True
        
    except Exception as e:
        print(f"❌ 进程生命周期测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        if os.path.exists(backup_file):
            os.unlink(backup_file)

if __name__ == "__main__":
    print("开始模拟万里长征运行测试...")
    
    # 先测试简单的生命周期
    test1_ok = test_process_lifecycle()
    
    if test1_ok:
        # 再测试复杂的异常处理
        test2_ok = test_multi_process_with_exceptions()
    else:
        test2_ok = False
    
    print(f"\n=== 测试结果总结 ===")
    print(f"进程生命周期: {'✅' if test1_ok else '❌'}")
    print(f"多进程异常处理: {'✅' if test2_ok else '❌'}")
    
    if test1_ok and not test2_ok:
        print("\n⚠️ 问题出现在大量进程同时处理异常时")
        print("可能的原因：")
        print("1. 大量异常导致进程池不稳定")
        print("2. 文件访问权限问题") 
        print("3. 资源耗尽")
    elif not test1_ok:
        print("\n⚠️ 基础的进程处理就有问题")
    else:
        print("\n✅ 测试环境下一切正常，实际问题可能更复杂")