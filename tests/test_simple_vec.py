#!/usr/bin/env python3
"""
简单测试Vec<f64>参数函数
"""

import rust_pyfunc as rp

def simple_wrapper(date: int, code: str) -> list:
    """简单的包装函数"""
    arr = [1.0, 2.0, 3.0, 4.0, 5.0]
    start_idx, end_idx, max_product = rp.find_max_range_product(arr)
    return [float(start_idx), float(end_idx), float(max_product)]

def main():
    print("=== 简单Vec<f64>参数测试 ===")
    
    # 单个任务测试
    dates_codes = [[20220101, '000001']]
    
    try:
        print("1. 直接调用测试")
        result = simple_wrapper(20220101, '000001')
        print(f"直接调用结果: {result}")
        
        print("\n2. 多进程调用测试")
        result2 = rp.run_pools(
            func=simple_wrapper,
            args=dates_codes,
            num_threads=1,
            backup_file="/tmp/test_vec_backup"
        )
        print(f"✅ 多进程测试成功，结果形状: {result2.shape}")
        print(f"结果: {result2[0] if len(result2) > 0 else 'None'}")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()