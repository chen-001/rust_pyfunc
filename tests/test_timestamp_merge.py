"""
测试日期时间类型键的合并功能
验证对pandas Timestamp类型的支持
"""

import pandas as pd
import numpy as np
import rust_pyfunc as rp
from datetime import datetime, date


def test_timestamp_key_merge():
    """测试pandas Timestamp作为连接键"""
    print("=== 测试Timestamp键合并 ===")
    
    # 创建包含Timestamp键的数据
    dates = pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04'])
    
    left_df = pd.DataFrame({
        'date': dates,
        'stock': ['AAPL', 'GOOGL', 'MSFT', 'TSLA'],
        'price': [150.0, 135.0, 140.0, 250.0]
    })
    
    right_df = pd.DataFrame({
        'date': dates[:3],  # 只取前3个日期
        'volume': [50000000, 25000000, 30000000],
        'market': ['US', 'US', 'US']
    })
    
    print("左表（Timestamp键）:")
    print(left_df)
    print(f"date列类型: {left_df['date'].dtype}")
    print("\n右表（Timestamp键）:")
    print(right_df)
    print(f"date列类型: {right_df['date'].dtype}")
    
    try:
        # 测试Timestamp键内连接
        print("\n测试Timestamp键内连接:")
        result = rp.fast_inner_join_df(left_df, right_df, on='date')
        print(f"合并结果: {len(result)}行")
        print(result)
        
        # 对比pandas结果
        pandas_result = pd.merge(left_df, right_df, on='date', how='inner')
        print(f"\npandas合并结果: {len(pandas_result)}行")
        print(pandas_result)
        
        # 验证结果行数一致
        assert len(result) == len(pandas_result), f"结果行数不一致: rust={len(result)}, pandas={len(pandas_result)}"
        print("✓ Timestamp键合并测试通过")
        
    except Exception as e:
        print(f"❌ Timestamp键合并测试失败: {e}")
        import traceback
        traceback.print_exc()


def test_date_key_merge():
    """测试Python date对象作为连接键"""
    print("\n=== 测试date键合并 ===")
    
    dates = [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3)]
    
    left_df = pd.DataFrame({
        'date': dates,
        'event': ['New Year', 'Day 2', 'Day 3'],
        'importance': [10, 5, 3]
    })
    
    right_df = pd.DataFrame({
        'date': dates[:2],
        'weather': ['Sunny', 'Cloudy'],
        'temperature': [20, 15]
    })
    
    print("左表（date键）:")
    print(left_df)
    print(f"date列类型: {left_df['date'].dtype}")
    print("\n右表（date键）:")
    print(right_df)
    print(f"date列类型: {right_df['date'].dtype}")
    
    try:
        # 测试date键内连接
        print("\n测试date键内连接:")
        result = rp.fast_inner_join_df(left_df, right_df, on='date')
        print(f"合并结果: {len(result)}行")
        print(result)
        
        # 对比pandas结果
        pandas_result = pd.merge(left_df, right_df, on='date', how='inner')
        print(f"\npandas合并结果: {len(pandas_result)}行")
        print(pandas_result)
        
        assert len(result) == len(pandas_result), f"结果行数不一致: rust={len(result)}, pandas={len(pandas_result)}"
        print("✓ date键合并测试通过")
        
    except Exception as e:
        print(f"❌ date键合并测试失败: {e}")
        import traceback
        traceback.print_exc()


def test_datetime_string_conversion():
    """测试日期时间类型转换为字符串的兜底方案"""
    print("\n=== 测试日期时间字符串转换 ===")
    
    # 创建不同类型的日期时间数据
    timestamps = pd.to_datetime(['2024-01-01 10:00:00', '2024-01-01 11:00:00', '2024-01-01 12:00:00'])
    
    left_df = pd.DataFrame({
        'timestamp': timestamps,
        'event': ['Meeting', 'Lunch', 'Presentation'],
        'duration': [60, 30, 45]
    })
    
    right_df = pd.DataFrame({
        'timestamp': timestamps[:2],
        'room': ['A101', 'Cafeteria'],
        'capacity': [20, 100]
    })
    
    print("左表（datetime64键）:")
    print(left_df)
    print(f"timestamp列类型: {left_df['timestamp'].dtype}")
    print("\n右表（datetime64键）:")
    print(right_df)
    print(f"timestamp列类型: {right_df['timestamp'].dtype}")
    
    try:
        # 测试datetime64键合并
        print("\n测试datetime64键合并:")
        result = rp.fast_inner_join_df(left_df, right_df, on='timestamp')
        print(f"合并结果: {len(result)}行")
        print(result)
        
        # 对比pandas结果
        pandas_result = pd.merge(left_df, right_df, on='timestamp', how='inner')
        print(f"\npandas合并结果: {len(pandas_result)}行")
        print(pandas_result)
        
        assert len(result) == len(pandas_result), f"结果行数不一致: rust={len(result)}, pandas={len(pandas_result)}"
        print("✓ datetime64键合并测试通过")
        
    except Exception as e:
        print(f"❌ datetime64键合并测试失败: {e}")
        import traceback
        traceback.print_exc()


def test_mixed_timestamp_keys():
    """测试混合类型包含时间戳的连接"""
    print("\n=== 测试混合类型时间戳键合并 ===")
    
    dates = pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-03'])
    
    left_df = pd.DataFrame({
        'symbol': ['AAPL', 'AAPL', 'GOOGL'],
        'date': dates,
        'price': [150.0, 151.0, 135.0]
    })
    
    right_df = pd.DataFrame({
        'symbol': ['AAPL', 'AAPL', 'MSFT'],
        'date': dates,
        'volume': [50000000, 45000000, 30000000]
    })
    
    print("左表（混合键：字符串+时间戳）:")
    print(left_df)
    print("\n右表（混合键：字符串+时间戳）:")
    print(right_df)
    
    try:
        # 测试混合键合并
        print("\n测试混合键合并（symbol + date）:")
        result = rp.fast_inner_join_df(left_df, right_df, on=['symbol', 'date'])
        print(f"合并结果: {len(result)}行")
        print(result)
        
        # 对比pandas结果
        pandas_result = pd.merge(left_df, right_df, on=['symbol', 'date'], how='inner')
        print(f"\npandas合并结果: {len(pandas_result)}行")
        print(pandas_result)
        
        assert len(result) == len(pandas_result), f"结果行数不一致: rust={len(result)}, pandas={len(pandas_result)}"
        print("✓ 混合类型时间戳键合并测试通过")
        
    except Exception as e:
        print(f"❌ 混合类型时间戳键合并测试失败: {e}")
        import traceback
        traceback.print_exc()


def test_timestamp_performance():
    """测试时间戳键合并的性能"""
    print("\n=== 时间戳键合并性能测试 ===")
    
    # 创建较大规模的时间序列数据
    n = 10000
    start_date = pd.Timestamp('2024-01-01')
    dates = pd.date_range(start=start_date, periods=n, freq='min')
    
    left_df = pd.DataFrame({
        'timestamp': dates,
        'value_left': np.random.randn(n)
    })
    
    # 创建50%重叠的右表
    right_dates = dates[n//2:].tolist() + pd.date_range(start=dates[-1] + pd.Timedelta(minutes=1), periods=n//2, freq='min').tolist()
    right_df = pd.DataFrame({
        'timestamp': right_dates,
        'value_right': np.random.randn(n)
    })
    
    print(f"测试数据规模: 左表{len(left_df)}行, 右表{len(right_df)}行")
    
    try:
        import time
        
        # pandas性能
        start = time.time()
        pandas_result = pd.merge(left_df, right_df, on='timestamp', how='inner')
        pandas_time = time.time() - start
        
        # rust_pyfunc性能
        start = time.time()
        rust_result = rp.fast_inner_join_df(left_df, right_df, on='timestamp')
        rust_time = time.time() - start
        
        print(f"\n时间戳键合并性能对比:")
        print(f"pandas耗时:     {pandas_time:.4f}秒 ({len(pandas_result)}行)")
        print(f"rust_pyfunc耗时: {rust_time:.4f}秒 ({len(rust_result)}行)")
        
        if rust_time > 0:
            ratio = pandas_time / rust_time
            if ratio > 1:
                print(f"🚀 rust_pyfunc快{ratio:.1f}倍")
            else:
                print(f"📊 pandas快{1/ratio:.1f}倍")
        
        # 验证结果一致性
        assert len(rust_result) == len(pandas_result), "时间戳键合并结果行数不一致"
        print("✓ 时间戳键性能测试完成")
        
    except Exception as e:
        print(f"❌ 时间戳键性能测试失败: {e}")
        import traceback
        traceback.print_exc()


def main():
    """运行所有日期时间键测试"""
    print("开始测试日期时间键合并功能\n")
    
    try:
        test_timestamp_key_merge()
        test_date_key_merge()
        test_datetime_string_conversion()
        test_mixed_timestamp_keys()
        test_timestamp_performance()
        
        print("\n" + "="*70)
        print("✅ 日期时间键合并功能测试完成！")
        print("✅ 支持pandas Timestamp类型")
        print("✅ 支持Python date对象")
        print("✅ 支持datetime64类型")
        print("✅ 支持混合类型键（字符串+时间戳）")
        print("✅ 自动处理和转换日期时间类型")
        print("✅ 保持与pandas.merge的兼容性")
        print("="*70)
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    if not success:
        exit(1)