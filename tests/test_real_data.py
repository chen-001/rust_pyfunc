"""
使用真实盘口数据测试Rust版本的segment_and_correlate函数
"""

import numpy as np
import pandas as pd
import time
import sys
sys.path.append('/home/chenzongwei/pythoncode')

from rust_pyfunc import segment_and_correlate
from w import *
import design_whatever as dw

class TestSegmentCorrelate(dw.TickDataBase):
    def get_factor(self):
        # 读取盘口快照数据
        market_data = self.read_market_data_single()
        
        if market_data.empty:
            print("无盘口数据")
            return pd.DataFrame()
        
        # 提取买一和卖一价格
        bid1_price = market_data['bid_prc1'].values
        ask1_price = market_data['ask_prc1'].values
        
        # 过滤有效数据
        valid_mask = (bid1_price > 0) & (ask1_price > 0)
        bid1_price = bid1_price[valid_mask]
        ask1_price = ask1_price[valid_mask]
        
        if len(bid1_price) < 100:
            print(f"有效数据点太少: {len(bid1_price)}")
            return pd.DataFrame()
        
        print(f"股票: {self.symbol}, 日期: {self.date}")
        print(f"有效数据点: {len(bid1_price)}")
        print(f"bid1价格范围: [{np.min(bid1_price):.4f}, {np.max(bid1_price):.4f}]")
        print(f"ask1价格范围: [{np.min(ask1_price):.4f}, {np.max(ask1_price):.4f}]")
        
        # 测试不同的最小段长度
        min_lengths = [20, 50, 100]
        
        for min_length in min_lengths:
            print(f"\n--- 最小段长度: {min_length} ---")
            
            # Rust版本测试
            start_time = time.time()
            rust_bid_corrs, rust_ask_corrs = segment_and_correlate(
                bid1_price, ask1_price, min_length
            )
            rust_time = time.time() - start_time
            
            print(f"Rust版本:")
            print(f"  耗时: {rust_time:.6f} 秒")
            print(f"  bid1>ask1段数: {len(rust_bid_corrs)}")
            print(f"  ask1>bid1段数: {len(rust_ask_corrs)}")
            
            if len(rust_bid_corrs) > 0:
                print(f"  bid1>ask1段相关系数: 均值={np.mean(rust_bid_corrs):.4f}, 标准差={np.std(rust_bid_corrs):.4f}")
                print(f"    范围: [{np.min(rust_bid_corrs):.4f}, {np.max(rust_bid_corrs):.4f}]")
                print(f"    具体值: {[f'{x:.3f}' for x in rust_bid_corrs[:5]]}{'...' if len(rust_bid_corrs) > 5 else ''}")
            
            if len(rust_ask_corrs) > 0:
                print(f"  ask1>bid1段相关系数: 均值={np.mean(rust_ask_corrs):.4f}, 标准差={np.std(rust_ask_corrs):.4f}")
                print(f"    范围: [{np.min(rust_ask_corrs):.4f}, {np.max(rust_ask_corrs):.4f}]")
                print(f"    具体值: {[f'{x:.3f}' for x in rust_ask_corrs[:5]]}{'...' if len(rust_ask_corrs) > 5 else ''}")
        
        # 返回结果数据
        result_data = {
            'data_points': len(bid1_price),
            'bid1_min': np.min(bid1_price),
            'bid1_max': np.max(bid1_price),
            'ask1_min': np.min(ask1_price), 
            'ask1_max': np.max(ask1_price),
            'segments_20': len(rust_bid_corrs) + len(rust_ask_corrs),
            'rust_time': rust_time
        }
        
        return pd.DataFrame([result_data])

def main():
    print("=== 使用真实盘口数据测试 segment_and_correlate ===")
    
    # 测试多只股票多天数据
    test_dates = [20220819, 20220822, 20220823]
    test_symbols = ['000001', '600000', '000002']
    
    for date in test_dates:
        for symbol in test_symbols:
            print(f"\n{'='*60}")
            print(f"测试股票: {symbol}, 日期: {date}")
            print('='*60)
            
            try:
                result = dw.run_factor(
                    TestSegmentCorrelate,
                    f"test_segment_{symbol}_{date}",
                    ["test_factor"],
                    n_jobs=1,
                    start_date=date,
                    end_date=date,
                    level2_single_stock=1,
                    code=[symbol],  # 使用code参数而不是symbols
                    for_test=1
                )
                
                if not result.empty:
                    print("✅ 测试成功")
                else:
                    print("⚠️ 无结果数据")
                    
            except Exception as e:
                print(f"❌ 测试失败: {e}")
                continue
            
            # 只测试一个就够了，避免运行太久
            break
        break

if __name__ == "__main__":
    main()