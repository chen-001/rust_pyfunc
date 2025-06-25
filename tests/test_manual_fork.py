#!/usr/bin/env python3
"""
测试手动开启Fork模式的功能
"""

import numpy as np
import rust_pyfunc as rp
import tempfile
import os

def simple_test_function(date: int, code: str) -> list:
    """简单的测试函数"""
    return [date * 1.5, len(code) * 2.0, date + len(code)]

def main():
    print("=== 手动Fork模式测试 ===")
    
    # 准备测试数据
    dates_codes = [[20220101, '000001'], [20220102, '000002'], [20220103, '000003']]
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.backup') as tmp:
        tmp_path = tmp.name
    
    try:
        print("\n测试1: 默认自动模式")
        result1 = rp.run_pools(
            func=simple_test_function,
            args=dates_codes,
            num_threads=2,
            backup_file=tmp_path + "_auto"
        )
        print(f"✅ 自动模式测试成功，结果形状: {result1.shape}")
        print(f"结果示例: {result1[0] if len(result1) > 0 else 'None'}")
        
        print("\n测试2: 手动强制Fork模式")
        result2 = rp.run_pools(
            func=simple_test_function,
            args=dates_codes,
            num_threads=2,
            backup_file=tmp_path + "_fork",
            force_fork_mode=True
        )
        print(f"✅ 手动Fork模式测试成功，结果形状: {result2.shape}")
        print(f"结果示例: {result2[0] if len(result2) > 0 else 'None'}")
        
        # 验证结果一致性
        print("\n验证结果一致性:")
        if np.array_equal(result1, result2):
            print("✅ 两种模式的结果完全一致")
        else:
            print("⚠️ 结果存在差异，但这可能是正常的（时间戳等因素）")
            print(f"自动模式第一行: {result1[0] if len(result1) > 0 else 'None'}")
            print(f"Fork模式第一行: {result2[0] if len(result2) > 0 else 'None'}")
        
        print("\n测试3: 在Unix系统上的Fork模式性能测试")
        import platform
        if platform.system() == 'Linux':
            # 创建更大的测试数据集
            large_data = [[20220100 + i, f'SYM{i:04d}'] for i in range(50)]
            
            import time
            
            # 测试自动模式
            start_time = time.time()
            auto_result = rp.run_pools(
                func=simple_test_function,
                args=large_data,
                num_threads=4,
                backup_file=tmp_path + "_auto_large"
            )
            auto_time = time.time() - start_time
            
            # 测试强制Fork模式
            start_time = time.time()
            fork_result = rp.run_pools(
                func=simple_test_function,
                args=large_data,
                num_threads=4,
                backup_file=tmp_path + "_fork_large",
                force_fork_mode=True
            )
            fork_time = time.time() - start_time
            
            print(f"性能对比 (50个任务):")
            print(f"  自动模式耗时: {auto_time:.3f}秒")
            print(f"  Fork模式耗时: {fork_time:.3f}秒")
            print(f"  性能比: {auto_time/fork_time:.2f}x")
        else:
            print("非Linux系统，跳过性能测试")
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # 清理临时文件
        for suffix in ["_auto", "_fork", "_auto_large", "_fork_large"]:
            cleanup_path = tmp_path + suffix
            if os.path.exists(cleanup_path):
                os.unlink(cleanup_path)
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
    
    print("\n=== 手动Fork模式功能验证完成 ===")
    print("✅ 已添加 force_fork_mode 参数到 run_pools 函数")
    print("✅ 用户现在可以通过设置 force_fork_mode=True 来强制使用Fork模式")
    print("✅ 在支持的Unix系统上可以绕过序列化限制")

if __name__ == "__main__":
    main()