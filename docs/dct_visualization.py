"""
DCT变换直观演示：将时域路径分解为频域成分
"""
import numpy as np
import altair as alt
import pandas as pd


def dct_transform(values):
    """DCT-II变换"""
    n = len(values)
    result = []
    for k in range(min(n, 6)):  # 取前6个系数
        coef = sum(
            values[i] * np.cos(np.pi * k * (i + 0.5) / n)
            for i in range(n)
        )
        result.append(coef)
    return result


def reconstruct_from_dct(dct_coefs, n=8, k_limit=None):
    """从DCT系数重建信号（演示用）"""
    if k_limit is not None:
        dct_coefs = dct_coefs[:k_limit] + [0] * (len(dct_coefs) - k_limit)
    
    result = []
    for i in range(n):
        val = 0
        for k, coef in enumerate(dct_coefs[:n]):
            val += coef * np.cos(np.pi * k * (i + 0.5) / n)
        result.append(val * 2 / n)  # 归一化
    return result


# 定义几种典型的收益路径模式
patterns = {
    "单边上涨": [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
    "先涨后跌(倒V)": [0.02, 0.02, 0.01, 0, -0.01, -0.02, -0.02, -0.02],
    "先跌后涨(V型)": [-0.02, -0.02, -0.01, 0, 0.01, 0.02, 0.02, 0.02],
    "震荡走势": [0.02, -0.02, 0.02, -0.02, 0.02, -0.02, 0.02, -0.02],
    "尾盘拉升": [0, 0, 0, 0, 0, 0, 0.01, 0.03],
}

# ========== 图1: 不同模式的DCT分解 ==========
records = []
for name, path in patterns.items():
    dct = dct_transform(path)
    for i, v in enumerate(path):
        records.append({"模式": name, "时段": i, "收益率": v, "类型": "原始路径"})
    for k, c in enumerate(dct):
        records.append({"模式": name, "系数": f"DCT[{k}]", "DCT值": c, "类型": "DCT系数"})

df_path = pd.DataFrame([r for r in records if "收益率" in r])
df_dct = pd.DataFrame([r for r in records if "DCT值" in r])

# 原始路径图
chart_path = alt.Chart(df_path).mark_line(point=True).encode(
    x=alt.X("时段:O", title="时段 (0=开盘, 7=收盘)"),
    y=alt.Y("收益率:Q", title="收益率"),
    color=alt.Color("模式:N", legend=alt.Legend(orient="top")),
).properties(
    width=600, height=200,
    title="不同收益路径模式"
)

# DCT系数柱状图
chart_dct = alt.Chart(df_dct).mark_bar().encode(
    x=alt.X("系数:O", title="DCT系数索引"),
    y=alt.Y("DCT值:Q", title="系数值"),
    color=alt.Color("模式:N", legend=None),
    column=alt.Column("模式:N", title=""),
).properties(
    width=80, height=150,
).resolve_scale(y="independent")

# ========== 图2: DCT基函数可视化 ==========
basis_records = []
n = 8
for k in range(6):
    for i in range(n):
        basis_val = np.cos(np.pi * k * (i + 0.5) / n)
        basis_records.append({
            "系数k": f"DCT[{k}]",
            "时段": i,
            "基函数值": basis_val,
            "频率": "直流(均值)" if k == 0 else f"频率{k}"
        })

df_basis = pd.DataFrame(basis_records)

chart_basis = alt.Chart(df_basis).mark_line(point=True).encode(
    x=alt.X("时段:O", title="时段"),
    y=alt.Y("基函数值:Q", title="cos值"),
    color=alt.Color("系数k:N", legend=alt.Legend(orient="right")),
).properties(
    width=400, height=200,
).facet(
    "频率:N",
    columns=3
).resolve_scale(y="independent").properties(
    title="DCT基函数：每个系数对应的波形"
)

# ========== 图3: 从低频逐步重建信号 ==========
recon_records = []
example_path = patterns["先涨后跌(倒V)"]
example_dct = dct_transform(example_path)

for k_limit in [1, 2, 4, 6]:
    reconstructed = reconstruct_from_dct(example_dct, n=8, k_limit=k_limit)
    for i, v in enumerate(reconstructed):
        recon_records.append({
            "使用系数": f"前{k_limit}个",
            "时段": i,
            "重建值": v
        })
# 加入原始
for i, v in enumerate(example_path):
    recon_records.append({
        "使用系数": "原始信号",
        "时段": i,
        "重建值": v
    })

df_recon = pd.DataFrame(recon_records)

chart_recon = alt.Chart(df_recon).mark_line(point=True).encode(
    x=alt.X("时段:O"),
    y=alt.Y("重建值:Q"),
    color=alt.Color("使用系数:N", legend=alt.Legend(orient="top")),
).properties(
    width=600, height=200,
    title="从低频逐步重建信号 (先涨后跌示例)"
)

# ========== 输出解释 ==========
print("=" * 60)
print("DCT变换直观解释")
print("=" * 60)
print("""
【核心思想】
把8个时段的收益率路径，分解成6个不同频率的"波形叠加"。

【类比】
想象一条波浪曲线，DCT把它拆解成：
- 直流（平均值）：整体水位高低
- 低频波：大波浪（整体趋势方向）
- 高频波：小涟漪（快速抖动）

【每个系数的含义】
DCT[0] = 直流分量 = 整体收益均值（当天赚钱了吗？）
DCT[1] = 最低频 = 单边趋势（是单边涨/跌，还是震荡？）
DCT[2] = 低频 = 趋势拐点（是否有上午涨下午跌的转折？）
DCT[3] = 中频 = 盘中震荡
DCT[4] = 较高频 = 波动节奏
DCT[5] = 高频 = 噪声/抖动

【为什么有用？】
- 低频系数(k=0,1,2) = 主要走势特征
- 高频系数(k=4,5) = 噪声

trend_noise_ratio = 低频能量 / 高频能量
比值大 = 趋势清晰，噪声少
比值小 = 震荡混乱，噪声多
""")

print("\n保存图表...")
chart_path.save("/home/chenzongwei/rust_pyfunc/docs/dct_path_patterns.html")
chart_basis.save("/home/chenzongwei/rust_pyfunc/docs/dct_basis_functions.html")
chart_recon.save("/home/chenzongwei/rust_pyfunc/docs/dct_reconstruction.html")

print("图表已保存:")
print("  - dct_path_patterns.html (不同模式的DCT分解)")
print("  - dct_basis_functions.html (DCT基函数可视化)")
print("  - dct_reconstruction.html (从低频逐步重建信号)")

# 打印各模式的DCT系数
print("\n" + "=" * 60)
print("各模式的DCT系数对比")
print("=" * 60)
print(f"{'模式':<15} {'DCT[0]':>10} {'DCT[1]':>10} {'DCT[2]':>10} {'DCT[3]':>10} {'DCT[4]':>10} {'DCT[5]':>10}")
print("-" * 85)
for name, path in patterns.items():
    dct = dct_transform(path)
    print(f"{name:<15}", end="")
    for c in dct:
        print(f"{c:>10.4f}", end="")
    print()
