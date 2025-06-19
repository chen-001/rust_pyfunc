import numpy as np
import time
import pandas as pd
from datetime import datetime

# 确保使用正确的Python路径
import sys
sys.path.append('/home/chenzongwei/rustcode/rust_pyfunc')

# 测试超时机制
def test_timeout_mechanism():
    print("测试 find_half_extreme_time 函数的超时机制")
    
    # 生成测试数据
    # 创建大数据集以确保计算足够慢
    n = 100000  # 数据点数量
    times = np.linspace(0, 10000, n) * 1e9  # 转换为纳秒
    prices = np.sin(np.linspace(0, 100, n)) * 10 + 100  # 生成正弦波价格数据
    
    # 测试正常情况（不设置超时）
    start_time = time.time()
    print(f"\n1. 不设置超时 - 开始时间: {datetime.now().strftime('%H:%M:%S')}")
    result1 = rust_pyfunc.find_half_extreme_time(times, prices, time_window=5.0)
    end_time = time.time()
    print(f"   结束时间: {datetime.now().strftime('%H:%M:%S')}, 耗时: {end_time - start_time:.2f}秒")
    print(f"   结果中NaN的数量: {np.isnan(result1).sum()}/{len(result1)}")
    print(f"   结果中的值范围: {np.nanmin(result1):.4f} 到 {np.nanmax(result1):.4f}")
    
    # 测试设置足够长的超时时间（不应该触发超时）
    start_time = time.time()
    print(f"\n2. 设置足够长的超时时间 (60秒) - 开始时间: {datetime.now().strftime('%H:%M:%S')}")
    result2 = rust_pyfunc.find_half_extreme_time(times, prices, time_window=5.0, timeout_seconds=60)
    end_time = time.time()
    print(f"   结束时间: {datetime.now().strftime('%H:%M:%S')}, 耗时: {end_time - start_time:.2f}秒")
    print(f"   结果中NaN的数量: {np.isnan(result2).sum()}/{len(result2)}")
    print(f"   结果中的值范围: {np.nanmin(result2):.4f} 到 {np.nanmax(result2):.4f}")
    
    # 测试设置较短的超时时间（应该触发超时）
    start_time = time.time()
    print(f"\n3. 设置较短的超时时间 (0.1秒) - 开始时间: {datetime.now().strftime('%H:%M:%S')}")
    result3 = rust_pyfunc.find_half_extreme_time(times, prices, time_window=5.0, timeout_seconds=0.1)
    end_time = time.time()
    print(f"   结束时间: {datetime.now().strftime('%H:%M:%S')}, 耗时: {end_time - start_time:.2f}秒")
    print(f"   结果中NaN的数量: {np.isnan(result3).sum()}/{len(result3)}")
    print(f"   是否全部为NaN: {np.isnan(result3).all()}")
    
    # 测试不同方向参数
    start_time = time.time()
    print(f"\n4. 测试direction参数 (pos) - 开始时间: {datetime.now().strftime('%H:%M:%S')}")
    result4 = rust_pyfunc.find_half_extreme_time(times[:1000], prices[:1000], time_window=5.0, direction="pos")
    end_time = time.time()
    print(f"   结束时间: {datetime.now().strftime('%H:%M:%S')}, 耗时: {end_time - start_time:.2f}秒")
    print(f"   结果中NaN的数量: {np.isnan(result4).sum()}/{len(result4)}")
    
    # 测试不同方向参数
    start_time = time.time()
    print(f"\n5. 测试direction参数 (neg) - 开始时间: {datetime.now().strftime('%H:%M:%S')}")
    result5 = rust_pyfunc.find_half_extreme_time(times[:1000], prices[:1000], time_window=5.0, direction="neg")
    end_time = time.time()
    print(f"   结束时间: {datetime.now().strftime('%H:%M:%S')}, 耗时: {end_time - start_time:.2f}秒")
    print(f"   结果中NaN的数量: {np.isnan(result5).sum()}/{len(result5)}")

# 运行测试
if __name__ == "__main__":
    import rust_pyfunc
    test_timeout_mechanism()
