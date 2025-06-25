#!/usr/bin/env python3
"""
测试PyAny参数类型的兼容性
"""

import numpy as np
import rust_pyfunc as rp

def test_find_max_range_product():
    """测试find_max_range_product的PyAny参数"""
    print("=== 测试find_max_range_product ===")
    
    # 测试numpy数组
    arr_numpy = np.array([4.0, 2.0, 1.0, 3.0])
    start_idx, end_idx, max_product = rp.find_max_range_product(arr_numpy)
    print(f"numpy数组输入: {arr_numpy}")
    print(f"结果: start_idx={start_idx}, end_idx={end_idx}, max_product={max_product}")
    
    # 测试Python列表
    arr_list = [4.0, 2.0, 1.0, 3.0]
    start_idx2, end_idx2, max_product2 = rp.find_max_range_product(arr_list)
    print(f"Python列表输入: {arr_list}")
    print(f"结果: start_idx={start_idx2}, end_idx={end_idx2}, max_product={max_product2}")
    
    # 验证结果一致
    assert start_idx == start_idx2, f"结果不一致: {start_idx} vs {start_idx2}"
    assert end_idx == end_idx2, f"结果不一致: {end_idx} vs {end_idx2}"
    assert abs(max_product - max_product2) < 1e-10, f"结果不一致: {max_product} vs {max_product2}"
    print("✅ find_max_range_product测试通过\n")

def test_analyze_retreat_advance_v2():
    """测试analyze_retreat_advance_v2的PyAny参数"""
    print("=== 测试analyze_retreat_advance_v2 ===")
    
    # 测试numpy数组
    trade_times_np = np.array([1000000000.0, 1000000100.0, 1000000200.0])
    trade_prices_np = np.array([100.0, 101.0, 100.5])
    trade_volumes_np = np.array([1000.0, 1500.0, 1200.0])
    trade_flags_np = np.array([66.0, 66.0, 83.0])
    
    orderbook_times_np = np.array([999999999.0, 1000000050.0, 1000000150.0])
    orderbook_prices_np = np.array([100.0, 101.0, 100.5])
    orderbook_volumes_np = np.array([5000.0, 8000.0, 6000.0])
    
    print("测试numpy数组输入...")
    result_np = rp.analyze_retreat_advance_v2(
        trade_times=trade_times_np,
        trade_prices=trade_prices_np,
        trade_volumes=trade_volumes_np,
        trade_flags=trade_flags_np,
        orderbook_times=orderbook_times_np,
        orderbook_prices=orderbook_prices_np,
        orderbook_volumes=orderbook_volumes_np,
        volume_percentile=95.0,
        time_window_minutes=1.0,
        breakthrough_threshold=0.0,
        dedup_time_seconds=30.0,
        find_local_lows=False
    )
    print(f"numpy结果元组长度: {len(result_np)}")
    
    # 测试Python列表
    trade_times_list = [1000000000.0, 1000000100.0, 1000000200.0]
    trade_prices_list = [100.0, 101.0, 100.5]
    trade_volumes_list = [1000.0, 1500.0, 1200.0]
    trade_flags_list = [66.0, 66.0, 83.0]
    
    orderbook_times_list = [999999999.0, 1000000050.0, 1000000150.0]
    orderbook_prices_list = [100.0, 101.0, 100.5]
    orderbook_volumes_list = [5000.0, 8000.0, 6000.0]
    
    print("测试Python列表输入...")
    result_list = rp.analyze_retreat_advance_v2(
        trade_times=trade_times_list,
        trade_prices=trade_prices_list,
        trade_volumes=trade_volumes_list,
        trade_flags=trade_flags_list,
        orderbook_times=orderbook_times_list,
        orderbook_prices=orderbook_prices_list,
        orderbook_volumes=orderbook_volumes_list,
        volume_percentile=95.0,
        time_window_minutes=1.0,
        breakthrough_threshold=0.0,
        dedup_time_seconds=30.0,
        find_local_lows=False
    )
    print(f"列表结果元组长度: {len(result_list)}")
    
    # 验证结果一致
    assert len(result_np) == len(result_list), f"结果长度不一致: {len(result_np)} vs {len(result_list)}"
    for i, (arr_np, arr_list) in enumerate(zip(result_np, result_list)):
        assert len(arr_np) == len(arr_list), f"数组{i}长度不一致: {len(arr_np)} vs {len(arr_list)}"
    
    print("✅ analyze_retreat_advance_v2测试通过\n")

def main():
    print("=== PyAny参数类型兼容性测试 ===\n")
    
    try:
        test_find_max_range_product()
        test_analyze_retreat_advance_v2()
        print("🎉 所有测试通过！PyAny参数类型现在支持numpy数组和Python列表")
        print("✅ 两个函数都具备灵活的输入类型处理能力")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()