#!/usr/bin/env python3
"""
GP相关维度算法演示
展示如何使用rust_pyfunc中的GP相关维度函数
"""

import sys
sys.path.insert(0, '/home/chenzongwei/.conda/envs/chenzongwei311/lib/python3.11/site-packages')

import numpy as np
import rust_pyfunc

def main():
    print("=== GP相关维度算法演示 ===\n")

    # 1. 零参数版本 - 用户只需提供数据
    print("1. 零参数版本演示")
    print("-" * 30)

    # 生成逻辑斯蒂映射数据（经典的混沌系统）
    def logistic_map(x0, r, n):
        x = np.zeros(n)
        x[0] = x0
        for i in range(1, n):
            x[i] = r * x[i-1] * (1 - x[i-1])
        return x

    # 生成1000个数据点
    data = logistic_map(0.5, 3.8, 1000)

    print(f"数据长度: {len(data)}")
    print(f"数据范围: [{data.min():.4f}, {data.max():.4f}]")

    # 计算GP相关维度
    result = rust_pyfunc.gp_correlation_dimension_auto(data)

    print(f"\n计算结果:")
    print(f"  延迟参数 τ: {result.tau}")
    print(f"  嵌入维数 m: {result.m}")
    print(f"  Theiler窗口: {result.theiler}")
    print(f"  相关维数 D₂: {result.d2_est:.4f}")
    print(f"  拟合质量 R²: {result.fit_r2:.4f}")
    print(f"  线性段范围: [{result.fit_start}, {result.fit_end}]")

    # 2. 自定义参数版本
    print("\n\n2. 自定义参数版本演示")
    print("-" * 30)

    # 创建自定义选项
    options = rust_pyfunc.gp_create_default_options()

    # 修改一些参数
    options.fnn_m_max = 10  # 限制最大嵌入维数
    options.n_r = 32        # 减少半径数量以加快计算
    options.fit_min_len = 8 # 增加最小拟合长度

    print(f"自定义参数:")
    print(f"  最大嵌入维数: {options.fnn_m_max}")
    print(f"  半径数量: {options.n_r}")
    print(f"  最小拟合长度: {options.fit_min_len}")

    # 使用自定义参数计算
    result_custom = rust_pyfunc.gp_correlation_dimension(data, options)

    print(f"\n自定义参数计算结果:")
    print(f"  延迟参数 τ: {result_custom.tau}")
    print(f"  嵌入维数 m: {result_custom.m}")
    print(f"  相关维数 D₂: {result_custom.d2_est:.4f}")
    print(f"  拟合质量 R²: {result_custom.fit_r2:.4f}")

    # 3. 对比不同系统
    print("\n\n3. 不同系统对比")
    print("-" * 30)

    systems = {
        "逻辑斯蒂映射(混沌)": logistic_map(0.5, 3.8, 500),
        "逻辑斯蒂映射(周期)": logistic_map(0.5, 3.2, 500),
        "随机噪声": np.random.randn(500),
        "正弦波": np.sin(np.linspace(0, 20*np.pi, 500))
    }

    print(f"{'系统名称':<15} {'D₂':<10} {'τ':<5} {'m':<5} {'R²':<8}")
    print("-" * 50)

    for name, data in systems.items():
        try:
            result = rust_pyfunc.gp_correlation_dimension_auto(data)
            print(f"{name:<15} {result.d2_est:<10.4f} {result.tau:<5} {result.m:<5} {result.fit_r2:<8.4f}")
        except Exception as e:
            print(f"{name:<15} 计算失败: {str(e)[:20]}...")

    print(f"\n=== 演示完成 ===")
    print(f"\n主要特点:")
    print(f"✓ 零参数设计：用户只需提供一维时间序列")
    print(f"✓ 完全确定性：所有参数自动选择，结果可复现")
    print(f"✓ 全面的诊断输出：包含中间结果便于分析")
    print(f"✓ 高性能实现：基于Rust的优化算法")

if __name__ == "__main__":
    main()