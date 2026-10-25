# -*- coding: utf-8 -*-
"""第 04 期：成交的节奏，四张图。"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyze import load

OUT = Path("/home/chenzongwei/hf-observation/2026-10-25-成交的节奏/images")
OUT.mkdir(parents=True, exist_ok=True)

for fp in fm.findSystemFonts():
    if "NotoSansCJK" in fp or "NotoSansSC" in fp:
        fm.fontManager.addfont(fp)
plt.rcParams["font.family"] = ["Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "white"
plt.rcParams["axes.edgecolor"] = "#BBBBBB"
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.color"] = "#E9E9E9"
plt.rcParams["axes.axisbelow"] = True

C_REAL = "#C0392B"
C_SUR = "#95A5A6"
CQ = ["#AED6F1", "#7FB3D5", "#5499C7", "#2E86C1", "#1B4F72"]
DATES = [20240205, 20240329, 20240813, 20240930, 20241008, 20250407, 20251127, 20260311]
LEG = {20240205: "02-05\n小盘危机", 20240329: "03-29\n常态A", 20240813: "08-13\n地量",
       20240930: "09-30\n924", 20241008: "10-08\n天量", 20250407: "04-07\n关税",
       20251127: "11-27\n常态B", 20260311: "03-11\n常态C"}


def quintile(df):
    q = np.zeros(len(df), dtype=int)
    for _, g in df.groupby("date"):
        q[g.index.values] = pd.qcut(g.s, 5, labels=False).values
    return q


def main():
    df, A = load()
    df["q"] = quintile(df)
    lam = df.groupby("q").lam.median()
    print("每秒笔数中位", lam.round(2).to_dict(), "样本", len(df))

    # ---------- 图一 ----------
    K = 15
    fig, axes = plt.subplots(2, 3, figsize=(17.5, 8.8))
    for qi, ax in enumerate(axes.flat):
        if qi >= 5:
            ax.axis("off")
            continue
        m = (df.q == qi).values
        h = A["H"][m].sum(0)
        hs = A["HS"][m].sum(0)
        pr = np.r_[h[:K], h[K:].sum()]
        ps = np.r_[hs[:K], hs[K:].sum()]
        pr, ps = pr / pr.sum(), ps / ps.sum()
        x = np.arange(K + 1)
        ax.bar(x - 0.2, pr, width=0.4, color=C_REAL, label="实际")
        ax.bar(x + 0.2, ps, width=0.4, color=C_SUR, label="同笔数对照")
        ax.set_xticks(x)
        ax.set_xticklabels([str(i) for i in range(K)] + [f"{K}+"])
        ax.set_title(f"第 {qi + 1} 组　每秒中位 {lam[qi]:.2f} 笔", fontsize=12.5)
        ax.set_xlabel("一秒钟里的成交笔数", fontsize=11)
        if qi % 3 == 0:
            ax.set_ylabel("占全天 14,220 秒的比例", fontsize=11)
        if qi == 0:
            ax.legend(frameon=False, fontsize=11)
        ax.text(0.97, 0.92, f"空秒 {pr[0] * 100:.0f}% / 对照 {ps[0] * 100:.0f}%",
                transform=ax.transAxes, ha="right", va="top", fontsize=11,
                bbox=dict(fc="white", ec="none", alpha=.85, pad=2))
    fig.suptitle("空着的秒比随机撒开更多，挤爆的秒也更多", fontsize=15, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    fig.savefig(OUT / "f1_per_second.png", dpi=150)
    plt.close(fig)

    # ---------- 图二 ----------
    fx = np.linspace(0, 1, A['LZ'].shape[1])
    lr = A["LZ"].mean(0)
    ls = A["LZS"].mean(0)
    half_r, half_s = np.interp(0.5, lr, fx), np.interp(0.5, ls, fx)
    top1_r, top1_s = np.interp(0.01, fx, lr), np.interp(0.01, fx, ls)
    fig, ax = plt.subplots(figsize=(11.8, 7.4))
    ax.plot(fx[1:] * 100, lr[1:] * 100, color=C_REAL, lw=2.6, label="实际")
    ax.plot(fx[1:] * 100, ls[1:] * 100, color=C_SUR, lw=2.4, ls="--", label="同笔数对照")
    ax.axhline(50, color="#999999", lw=0.9, ls=":")
    ax.plot([half_r * 100], [50], "o", color=C_REAL, ms=9)
    ax.plot([half_s * 100], [50], "o", color=C_SUR, ms=9)
    ax.annotate(f"实际 {half_r * 100:.1f}%", (half_r * 100, 50), xytext=(8, -30),
                textcoords="offset points", color=C_REAL, fontsize=12.5, fontweight="bold")
    ax.annotate(f"对照 {half_s * 100:.1f}%", (half_s * 100, 50), xytext=(14, -56),
                textcoords="offset points", color="#5D6D7E", fontsize=12.5)
    ax.set_xscale("log")
    ax.set_xlim(0.3, 100)
    ax.set_ylim(0, 100)
    ax.set_xlabel("最忙的那部分时间，占全天的比例（%，对数轴）", fontsize=12)
    ax.set_ylabel("这部分时间里发生的成交，占全天的比例（%）", fontsize=12)
    ax.set_title("一半的成交挤在最忙的几个百分点里", fontsize=15)
    ax.legend(frameon=False, fontsize=12, loc="lower right")
    ax.text(0.03, 0.06, f"最忙的 1% 时间（142 秒）里发生了 {top1_r * 100:.0f}% 的成交，对照是 {top1_s * 100:.0f}%",
            transform=ax.transAxes, fontsize=11.5, color="#333333")
    fig.tight_layout()
    fig.savefig(OUT / "f2_concentration.png", dpi=150)
    plt.close(fig)

    # ---------- 图三 ----------
    stat = df.groupby("q").ratio_p.describe(percentiles=[.1, .25, .5, .75, .9])
    x = np.arange(5)
    fig, ax = plt.subplots(figsize=(11.8, 7.2))
    ax.fill_between(x, stat["10%"], stat["90%"], color="#D6EAF8", label="十分位到九十分位")
    ax.fill_between(x, stat["25%"], stat["75%"], color="#85C1E9", label="中间一半")
    ax.plot(x, stat["50%"], "o-", color=C_REAL, lw=2.4, ms=9, label="中位")
    for i in range(5):
        ha, dx = ("right", -0.12) if i == 4 else ("left", 0.12)
        ax.text(i + dx, stat["50%"].iloc[i] + 0.018, f"{stat['50%'].iloc[i]:.2f}", ha=ha, fontsize=13,
                color=C_REAL, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"第 {i + 1} 组\n每秒 {lam[i]:.2f} 笔" for i in range(5)])
    ax.set_ylabel("纯成团倍数（实际 ÷ 同笔数对照）", fontsize=12)
    ax.set_title("活跃度差 13 倍，成团只多两成", fontsize=15)
    ax.legend(frameon=False, fontsize=11.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "f3_vs_activity.png", dpi=150)
    plt.close(fig)

    # ---------- 图四 ----------
    cm = df.groupby("date").common.median() * 100
    fig, ax = plt.subplots(figsize=(12.2, 6.8))
    cols = ["#5D6D7E" if cm[d] < 12 else "#C0392B" for d in DATES]
    ax.bar(range(8), [cm[d] for d in DATES], color=cols, width=.62)
    for i, d in enumerate(DATES):
        ax.text(i, cm[d] + 0.6, f"{cm[d]:.1f}%", ha="center", fontsize=12.5)
    ax.set_xticks(range(8))
    ax.set_xticklabels([LEG[d] for d in DATES], fontsize=11)
    ax.set_ylabel("成团里由全市场共同作息解释的比例（%）", fontsize=12)
    ax.set_ylim(0, max(cm) * 1.22)
    ax.set_title("常态日各走各的，天量日和暴跌日才会一起动", fontsize=15)
    fig.tight_layout()
    fig.savefig(OUT / "f4_common.png", dpi=150)
    plt.close(fig)

    print(f"一半成交落在最忙的 {half_r * 100:.2f}% 时间（对照 {half_s * 100:.2f}%）")
    print(f"最忙 1% 时间承载 {top1_r * 100:.1f}%（对照 {top1_s * 100:.1f}%）")
    print("按日 common：", cm.round(1).to_dict())
    print("图已写入", OUT)


if __name__ == "__main__":
    main()
