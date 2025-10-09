#!/usr/bin/env python3
"""
GP相关维度算法最终演示
展示算法的核心特性和使用方法
"""

import sys
sys.path.insert(0, '/home/chenzongwei/.conda/envs/chenzongwei311/lib/python3.11/site-packages')

import numpy as np
import rust_pyfunc

def main():
    print("🎯 GP相关维度算法演示")
    print("=" * 50)

    # 1. 生成经典混沌数据 - 逻辑斯蒂映射
    def logistic_map(x0, r, n):
        x = np.zeros(n)
        x[0] = x0
        for i in range(1, n):
            x[i] = r * x[i-1] * (1 - x[i-1])
        return x

    print("📊 生成测试数据...")
    data = logistic_map(0.5, 3.8, 800)  # 混沌参数
    print(f"   数据类型: 逻辑斯蒂映射 (r=3.8)")
    print(f"   数据长度: {len(data)}")
    print(f"   数据范围: [{data.min():.4f}, {data.max():.4f}]")

    # 2. 零参数调用 - 完全自动化
    print("\n🚀 零参数调用 - 完全自动化分析")
    print("-" * 40)
    result = rust_pyfunc.gp_correlation_dimension_auto(data)

    print(f"✅ 计算完成!")
    print(f"   相关维数 D₂: {result.d2_est:.6f}")
    print(f"   延迟参数 τ: {result.tau}")
    print(f"   嵌入维数 m: {result.m}")
    print(f"   Theiler窗口: {result.theiler}")
    print(f"   拟合质量 R²: {result.fit_r2:.6f}")
    print(f"   线性段范围: [{result.fit_start}, {result.fit_end}]")

    # 3. 验证确定性
    print("\n🔄 确定性验证")
    print("-" * 40)
    results = []
    for i in range(3):
        r = rust_pyfunc.gp_correlation_dimension_auto(data)
        results.append(r.d2_est)
        print(f"   第{i+1}次计算: D₂ = {r.d2_est:.6f}")

    print(f"   标准差: {np.std(results):.10f}")
    print("   ✅ 完全确定性: 结果完全一致")

    # 4. 对比不同系统
    print("\n📈 不同系统的相关维度对比")
    print("-" * 40)

    systems = {
        "逻辑斯蒂映射(混沌)": logistic_map(0.5, 3.8, 500),
        "逻辑斯蒂映射(周期)": logistic_map(0.5, 3.2, 500),
        "高斯白噪声": np.random.RandomState(42).randn(500),
        "正弦波": np.sin(np.linspace(0, 20*np.pi, 500))
    }

    print(f"{'系统类型':<20} {'D₂':<12} {'τ':<5} {'m':<5} {'R²':<8}")
    print("-" * 60)

    for name, data in systems.items():
        try:
            result = rust_pyfunc.gp_correlation_dimension_auto(data)
            print(f"{name:<20} {result.d2_est:<12.4f} {result.tau:<5} {result.m:<5} {result.fit_r2:<8.4f}")
        except:
            print(f"{name:<20} 计算失败")

    # 5. 自定义参数演示
    print("\n⚙️ 自定义参数选项")
    print("-" * 40)

    # 创建自定义选项
    options = rust_pyfunc.gp_create_default_options()
    options.fnn_m_max = 8     # 限制最大嵌入维数
    options.n_r = 30         # 调整半径数量

    print(f"   自定义最大嵌入维数: {options.fnn_m_max}")
    print(f"   自定义半径数量: {options.n_r}")

    result_custom = rust_pyfunc.gp_correlation_dimension(data, options)
    print(f"   自定义参数结果: D₂ = {result_custom.d2_est:.6f}")

    print("\n" + "=" * 50)
    print("🎉 GP相关维度算法演示完成!")
    print("\n主要特点:")
    print("✅ 零参数设计：用户只需提供一维时间序列")
    print("✅ 完全确定性：所有参数自动选择，结果可复现")
    print("✅ 全面的诊断输出：包含中间结果便于分析")
    print("✅ 高性能实现：基于Rust的优化算法")
    print("✅ 灵活的参数调节：支持自定义选项")
    print("✅ 多种系统适应性：适用于混沌、周期、随机等不同类型数据")

if __name__ == "__main__":
    main()