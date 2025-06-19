"""
序列分段和相关系数计算函数演示
实现你想要的功能：当a和b互相反超时进行分段，计算每段内的相关系数
"""

import numpy as np
import pandas as pd
import time
from typing import Tuple, List
import matplotlib.pyplot as plt

def segment_and_correlate(a: np.ndarray, b: np.ndarray, min_length: int = 10) -> Tuple[List[float], List[float]]:
    """
    序列分段和相关系数计算函数
    
    输入两个等长的序列，根据大小关系进行分段，然后计算每段内的相关系数
    
    参数：
    - a: 第一个序列
    - b: 第二个序列  
    - min_length: 最小段长度，默认为10
    
    返回：
    - (a_greater_corrs, b_greater_corrs): 两个相关系数列表
    """
    if len(a) != len(b):
        raise ValueError("输入序列a和b的长度必须相等")
    
    if len(a) < 2:
        return [], []
    
    # 识别分段
    segments = []
    current_start = 0
    current_a_greater = a[0] > b[0]
    
    for i in range(1, len(a)):
        a_greater = a[i] > b[i]
        
        # 如果状态发生变化，结束当前段
        if a_greater != current_a_greater:
            if i - current_start >= min_length:
                segments.append((current_start, i, current_a_greater))
            current_start = i
            current_a_greater = a_greater
    
    # 添加最后一段
    if len(a) - current_start >= min_length:
        segments.append((current_start, len(a), current_a_greater))
    
    print(f"总共识别到 {len(segments)} 个符合条件的段（最小长度{min_length}）:")
    
    # 计算每段的相关系数
    a_greater_corrs = []
    b_greater_corrs = []
    
    for i, (start, end, a_greater) in enumerate(segments):
        # 提取段数据
        segment_a = a[start:end]
        segment_b = b[start:end]
        
        # 计算相关系数
        corr = np.corrcoef(segment_a, segment_b)[0, 1]
        
        segment_type = "a>b" if a_greater else "b>a"
        print(f"  段{i+1}: [{start}:{end}] 长度={end-start}, 类型={segment_type}, 相关系数={corr:.4f}")
        
        # 如果相关系数是NaN，跳过
        if not np.isnan(corr):
            if a_greater:
                a_greater_corrs.append(corr)
            else:
                b_greater_corrs.append(corr)
    
    return a_greater_corrs, b_greater_corrs

def simulate_market_data(n: int = 5000) -> Tuple[np.ndarray, np.ndarray]:
    """模拟盘口数据"""
    np.random.seed(42)
    base_price = 10.0
    
    # 模拟bid和ask价格序列
    # 创建趋势和随机波动
    trend = np.sin(np.arange(n) * 0.01) * 0.05  # 正弦趋势
    bid_noise = np.cumsum(np.random.randn(n) * 0.001)
    ask_noise = np.cumsum(np.random.randn(n) * 0.001) 
    
    bid_prices = base_price + trend + bid_noise
    ask_prices = base_price + trend + ask_noise + 0.01  # ask通常比bid高
    
    # 增加一些随机的反超情况
    for i in range(100, n, 200):
        if i + 50 < n:
            # 在某些区间让bid暂时超过ask（模拟异常情况）
            bid_prices[i:i+50] += 0.02
    
    return bid_prices.astype(np.float64), ask_prices.astype(np.float64)

def demo_with_real_data():
    """使用真实盘口数据进行演示"""
    try:
        import sys
        sys.path.append('/home/chenzongwei/pythoncode')
        import w
        import design_whatever as dw
        
        print("=== 尝试读取真实盘口数据 ===")
        
        # 创建数据读取类
        class TestData(dw.TickDataBase):
            def get_factor(self):
                market_data = self.read_market_data_single()
                
                if market_data.empty:
                    print("无数据")
                    return pd.DataFrame()
                
                # 使用bid1和ask1价格
                bid1 = market_data['bid_prc1'].values
                ask1 = market_data['ask_prc1'].values
                
                # 过滤有效数据
                valid_mask = (bid1 > 0) & (ask1 > 0)
                bid1 = bid1[valid_mask]
                ask1 = ask1[valid_mask]
                
                if len(bid1) < 100:
                    print(f"数据点不足: {len(bid1)}")
                    return pd.DataFrame()
                
                print(f"读取到 {len(bid1)} 个有效数据点")
                
                # 进行分段和相关系数计算
                start_time = time.time()
                a_corrs, b_corrs = segment_and_correlate(bid1, ask1, min_length=30)
                elapsed = time.time() - start_time
                
                print(f"\n计算耗时: {elapsed:.4f} 秒")
                print(f"bid1>ask1的段数: {len(a_corrs)}")
                print(f"ask1>bid1的段数: {len(b_corrs)}")
                
                if len(a_corrs) > 0:
                    print(f"bid1>ask1段的相关系数: 均值={np.mean(a_corrs):.4f}, 标准差={np.std(a_corrs):.4f}")
                    print(f"  相关系数范围: [{np.min(a_corrs):.4f}, {np.max(a_corrs):.4f}]")
                
                if len(b_corrs) > 0:
                    print(f"ask1>bid1段的相关系数: 均值={np.mean(b_corrs):.4f}, 标准差={np.std(b_corrs):.4f}")
                    print(f"  相关系数范围: [{np.min(b_corrs):.4f}, {np.max(b_corrs):.4f}]")
                
                return pd.DataFrame({'success': [1]})
        
        # 尝试读取一天的数据
        result = dw.run_factor(
            TestData,
            "demo_segment_correlate", 
            ["test"],
            n_jobs=1,
            start_date=20220819,
            end_date=20220819,
            symbols=['000001'],
            level2_single_stock=1,
            for_test=1
        )
        
    except Exception as e:
        print(f"真实数据读取失败: {e}")
        print("将使用模拟数据进行演示")
        return False
    
    return True

def demo_with_simulated_data():
    """使用模拟数据进行演示"""
    print("\n=== 模拟数据演示 ===")
    
    # 生成模拟数据
    bid_prices, ask_prices = simulate_market_data(3000)
    
    print(f"生成了 {len(bid_prices)} 个数据点")
    print(f"bid价格范围: [{np.min(bid_prices):.4f}, {np.max(bid_prices):.4f}]")
    print(f"ask价格范围: [{np.min(ask_prices):.4f}, {np.max(ask_prices):.4f}]")
    
    # 计算分段相关系数
    start_time = time.time()
    bid_greater_corrs, ask_greater_corrs = segment_and_correlate(bid_prices, ask_prices, min_length=50)
    elapsed = time.time() - start_time
    
    print(f"\n计算耗时: {elapsed:.4f} 秒")
    print(f"bid>ask的段数: {len(bid_greater_corrs)}")
    print(f"ask>bid的段数: {len(ask_greater_corrs)}")
    
    if len(bid_greater_corrs) > 0:
        print(f"\nbid>ask段的相关系数统计:")
        print(f"  均值: {np.mean(bid_greater_corrs):.4f}")
        print(f"  标准差: {np.std(bid_greater_corrs):.4f}")
        print(f"  范围: [{np.min(bid_greater_corrs):.4f}, {np.max(bid_greater_corrs):.4f}]")
        print(f"  具体值: {[f'{x:.3f}' for x in bid_greater_corrs]}")
    
    if len(ask_greater_corrs) > 0:
        print(f"\nask>bid段的相关系数统计:")
        print(f"  均值: {np.mean(ask_greater_corrs):.4f}")
        print(f"  标准差: {np.std(ask_greater_corrs):.4f}")
        print(f"  范围: [{np.min(ask_greater_corrs):.4f}, {np.max(ask_greater_corrs):.4f}]")
        print(f"  具体值: {[f'{x:.3f}' for x in ask_greater_corrs]}")

def main():
    """主函数"""
    print("=" * 60)
    print("序列分段和相关系数计算演示")
    print("=" * 60)
    
    # 首先尝试真实数据
    success = demo_with_real_data()
    
    # 如果真实数据失败，使用模拟数据
    if not success:
        demo_with_simulated_data()
    
    print("\n" + "=" * 60)
    print("演示完成")
    print("=" * 60)

if __name__ == "__main__":
    main()