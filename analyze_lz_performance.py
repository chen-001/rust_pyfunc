#!/usr/bin/env python3
"""
分析当前lz_complexity函数的性能瓶颈
"""

import numpy as np
import time
import sys
import os

# 添加路径以便导入rust_pyfunc
sys.path.insert(0, '/home/chenzongwei/rust_pyfunc/python')

def test_current_performance():
    """测试当前版本的性能"""
    print("=" * 60)
    print("分析当前lz_complexity函数性能")
    print("=" * 60)

    try:
        import rust_pyfunc

        # 测试不同长度的序列
        test_lengths = [1000, 5000, 10000, 50000, 100000]

        for length in test_lengths:
            print(f"\n测试序列长度: {length}")

            # 生成随机序列
            np.random.seed(42)
            seq = np.random.randn(length).astype(np.float64)

            # 测试离散化时间
            start_time = time.time()
            discrete_seq = np.where(seq > np.median(seq), 1, 0).astype(np.uint8)
            discretize_time = time.time() - start_time

            # 测试总计算时间
            start_time = time.time()
            result = rust_pyfunc.lz_complexity(seq, quantiles=[0.5])
            total_time = time.time() - start_time

            print(f"  离散化时间: {discretize_time:.4f}秒")
            print(f"  总计算时间: {total_time:.4f}秒")
            print(f"  LZ复杂度结果: {result:.6f}")
            print(f"  估算核心算法时间: {total_time - discretize_time:.4f}秒")

            if length == 100000:
                if total_time < 0.2:
                    print("  ✅ 当前版本已满足性能要求！")
                else:
                    print(f"  ❌ 当前版本不满足性能要求，需要优化")
                    print(f"  需要提升倍数: {total_time / 0.2:.1f}x")

    except ImportError as e:
        print(f"❌ 导入错误: {e}")
        print("请确保项目已正确构建")

def test_binary_sequence_performance():
    """专门测试二进制序列性能（最常见情况）"""
    print("\n" + "=" * 60)
    print("测试二进制序列性能（优化重点）")
    print("=" * 60)

    try:
        import rust_pyfunc

        # 测试不同长度的二进制序列
        test_lengths = [10000, 50000, 100000, 200000]

        for length in test_lengths:
            print(f"\n测试二进制序列长度: {length}")

            # 生成随机二进制序列
            np.random.seed(42)
            seq = np.random.randint(0, 2, length).astype(np.float64)

            # 测试计算时间
            start_time = time.time()
            result = rust_pyfunc.lz_complexity(seq)
            total_time = time.time() - start_time

            print(f"  计算时间: {total_time:.4f}秒")
            print(f"  LZ复杂度结果: {result:.6f}")

            if length == 100000:
                if total_time < 0.2:
                    print("  ✅ 二进制序列性能满足要求！")
                else:
                    print(f"  ❌ 二进制序列需要优化，当前: {total_time:.4f}秒")

    except ImportError as e:
        print(f"❌ 导入错误: {e}")

if __name__ == "__main__":
    test_current_performance()
    test_binary_sequence_performance()