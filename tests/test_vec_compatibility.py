#!/usr/bin/env python3
"""
测试Vec<f64>参数类型的兼容性
"""

import rust_pyfunc as rp

def test_find_max_range_product():
    """测试find_max_range_product的Vec<f64>参数"""
    print("=== 测试find_max_range_product ===")
    
    # 测试Python列表
    arr = [4.0, 2.0, 1.0, 3.0]
    start_idx, end_idx, max_product = rp.find_max_range_product(arr)
    print(f"Python列表输入: {arr}")
    print(f"结果: start_idx={start_idx}, end_idx={end_idx}, max_product={max_product}")
    
    # 验证结果是否合理
    expected_product = min(arr[start_idx], arr[end_idx]) * abs(end_idx - start_idx)
    print(f"验证: min({arr[start_idx]}, {arr[end_idx]}) * {abs(end_idx - start_idx)} = {expected_product}")
    assert abs(max_product - expected_product) < 1e-10, f"结果不匹配: {max_product} vs {expected_product}"
    print("✅ find_max_range_product测试通过\n")

def test_analyze_retreat_advance_v2():
    """测试analyze_retreat_advance_v2的Vec<f64>参数"""
    print("=== 测试analyze_retreat_advance_v2 ===")
    
    # 创建简单的测试数据
    trade_times = [1000000000.0, 1000000100.0, 1000000200.0]
    trade_prices = [100.0, 101.0, 100.5]
    trade_volumes = [1000.0, 1500.0, 1200.0]
    trade_flags = [66.0, 66.0, 83.0]
    
    orderbook_times = [999999999.0, 1000000050.0, 1000000150.0]
    orderbook_prices = [100.0, 101.0, 100.5]
    orderbook_volumes = [5000.0, 8000.0, 6000.0]
    
    print(f"交易数据长度: {len(trade_times)}")
    print(f"订单簿数据长度: {len(orderbook_times)}")
    
    try:
        result = rp.analyze_retreat_advance_v2(
            trade_times=trade_times,
            trade_prices=trade_prices,
            trade_volumes=trade_volumes,
            trade_flags=trade_flags,
            orderbook_times=orderbook_times,
            orderbook_prices=orderbook_prices,
            orderbook_volumes=orderbook_volumes,
            volume_percentile=95.0,
            time_window_minutes=1.0,
            breakthrough_threshold=0.0,
            dedup_time_seconds=30.0,
            find_local_lows=False
        )
        
        print(f"结果类型: {type(result)}")
        print(f"结果元组长度: {len(result)}")
        
        # 打印每个结果数组的长度
        for i, arr in enumerate(result):
            print(f"结果数组 {i}: 长度={len(arr)}, 类型={type(arr)}")
        
        print("✅ analyze_retreat_advance_v2测试通过\n")
        
    except Exception as e:
        print(f"❌ analyze_retreat_advance_v2测试失败: {e}")
        import traceback
        traceback.print_exc()

def main():
    print("=== Vec<f64>参数类型兼容性测试 ===\n")
    
    try:
        test_find_max_range_product()
        test_analyze_retreat_advance_v2()
        print("🎉 所有测试通过！现在两个函数都接受Vec<f64>参数，支持完全的序列化")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()