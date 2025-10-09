#!/usr/bin/env python3
"""
调试大序列结果不一致问题
"""

import numpy as np
import time
import sys
sys.path.insert(0, '/home/chenzongwei/rust_pyfunc/python')

def lz_complexity_python_original(seq, k=None, normalize=True):
    """Python原版LZ76复杂度函数（完全按照用户提供的版本）"""
    # 统一成字符串
    if not isinstance(seq, str):
        # 将符号映射到紧致字母表
        vals = {v: i for i, v in enumerate(dict.fromkeys(seq))}
        s = ''.join(chr(vals[v]) for v in seq)  # 用字符编码避免连接耗时
        k_eff = len(vals)
    else:
        s = seq
        k_eff = len(set(s)) if k is None else k
    n = len(s)
    i = 0
    c = 0
    while i < n:
        j = i + 1
        # 扩到"新"为止：s[i:j] 不再出现在 s[:j-1]
        while j <= n and s[i:j] in s[:j-1]:
            j += 1
        c += 1
        i = j
    if not normalize:
        return c
    import math
    base = k if k is not None else k_eff
    return c * (math.log(n, base) if n > 1 else 0.0) / n

def discretize_sequence_python(seq, quantiles):
    """Python版本的离散化"""
    values = np.sort(seq)
    thresholds = []
    for q in quantiles:
        idx = int((len(seq) - 1) * q)
        thresholds.append(values[idx])

    discrete_seq = np.zeros(len(seq), dtype=int)
    for i, val in enumerate(seq):
        symbol = 0
        for j, threshold in enumerate(thresholds):
            if val <= threshold:
                symbol = j
                break
            symbol = j + 1
        discrete_seq[i] = symbol + 1  # 从1开始编号
    return discrete_seq

def test_large_sequence_consistency():
    """测试大序列的一致性"""
    print("大序列一致性测试")
    print("=" * 60)

    # 生成固定种子的测试数据
    np.random.seed(42)
    large_seq = np.random.randn(100000).astype(np.float64)
    quantiles = [0.5]

    print(f"序列长度: {len(large_seq)}")
    print(f"分位数: {quantiles}")
    print()

    # Python版本计算
    print("Python版本计算中...")
    start_time = time.time()

    # 离散化
    discrete_seq_python = discretize_sequence_python(large_seq, quantiles)

    # 计算LZ复杂度
    result_python = lz_complexity_python_original(discrete_seq_python)

    python_time = time.time() - start_time
    print(f"Python版本结果: {result_python:.6f}")
    print(f"Python版本时间: {python_time:.3f}秒")
    print()

    # Rust版本计算
    print("Rust版本计算中...")
    import rust_pyfunc
    start_time = time.time()
    result_rust = rust_pyfunc.lz_complexity(large_seq, quantiles=quantiles)
    rust_time = time.time() - start_time
    print(f"Rust版本结果:   {result_rust:.6f}")
    print(f"Rust版本时间:   {rust_time:.3f}秒")
    print()

    # 对比结果
    diff = abs(result_python - result_rust)
    print(f"绝对差值: {diff:.6f}")
    print(f"相对差值: {diff/result_python*100:.2f}%")

    if diff < 1e-10:
        print("✅ 结果完全一致！")
    elif diff < 1e-6:
        print("⚠️  结果基本一致（浮点精度差异）")
    else:
        print("❌ 结果不一致，需要修复！")

    return result_python, result_rust, discrete_seq_python

def debug_algorithm_differences():
    """调试算法差异"""
    print("\n算法差异调试")
    print("=" * 60)

    # 使用较小的序列进行详细对比
    np.random.seed(123)
    test_seq = np.random.randn(1000).astype(np.float64)
    quantiles = [0.5]

    print(f"测试序列长度: {len(test_seq)}")

    # Python版本
    discrete_python = discretize_sequence_python(test_seq, quantiles)
    result_python = lz_complexity_python_original(discrete_python)

    # Rust版本
    import rust_pyfunc
    result_rust = rust_pyfunc.lz_complexity(test_seq, quantiles=quantiles)

    print(f"Python结果: {result_python:.6f}")
    print(f"Rust结果:   {result_rust:.6f}")
    print(f"差值:       {abs(result_python - result_rust):.6f}")

    # 检查离散化是否一致
    print(f"\nPython离散化结果前10个: {discrete_python[:10]}")

    # 检查符号数量
    unique_symbols_python = len(set(discrete_python))
    print(f"Python唯一符号数: {unique_symbols_python}")

def main():
    """主函数"""
    try:
        result_python, result_rust, discrete_seq = test_large_sequence_consistency()

        if abs(result_python - result_rust) > 1e-6:
            debug_algorithm_differences()

    except Exception as e:
        print(f"测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()