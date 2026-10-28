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
        pr = np.r_[h[:K], h[K:].sum()]
        pr = pr / pr.sum()
        x = np.arange(K + 1)
        ax.bar(x, pr, width=0.62, color=C_REAL)
        ax.set_xticks(x)
        ax.set_xticklabels([str(i) for i in range(K)] + [f"{K}+"])
        ax.set_title(f"第 {qi + 1} 组　每秒中位 {lam[qi]:.2f} 笔", fontsize=12.5)
        ax.set_xlabel("一秒钟里的成交笔数", fontsize=11)
        if qi % 3 == 0:
            ax.set_ylabel("占全天 14,220 秒的比例", fontsize=11)
        ax.text(0.97, 0.92, f"空着的秒 {pr[0] * 100:.0f}%",
                transform=ax.transAxes, ha="right", va="top", fontsize=12,
                bbox=dict(fc="white", ec="none", alpha=.85, pad=2))
    fig.suptitle("最冷清的五分之一，一天里近九成的秒没有成交", fontsize=15, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    fig.savefig(OUT / "f1_per_second.png", dpi=150)
    plt.close(fig)

    # ---------- 图二 ----------
    fx = np.linspace(0, 1, A['LZ'].shape[1])
    lr = A["LZ"].mean(0)
    half_r = np.interp(0.5, lr, fx)
    top1_r = np.interp(0.01, fx, lr)
    fig, ax = plt.subplots(figsize=(11.8, 7.4))
    for qi in range(5):
        m = (df.q == qi).values
        ax.plot(fx[1:] * 100, A["LZ"][m].mean(0)[1:] * 100, color=CQ[qi], lw=2.6,
                label=f"第 {qi + 1} 组　每秒 {lam[qi]:.2f} 笔")
    ax.axhline(50, color="#999999", lw=0.9, ls=":")
    ax.plot([half_r * 100], [50], "o", color=C_REAL, ms=10, zorder=5)
    ax.annotate(f"全体中位 {half_r * 100:.1f}%", (half_r * 100, 50), xytext=(10, -34),
                textcoords="offset points", color=C_REAL, fontsize=13, fontweight="bold")
    ax.set_xscale("log")
    ax.set_xlim(0.3, 100)
    ax.set_ylim(0, 100)
    ax.set_xlabel("最忙的那部分时间，占全天的比例（%，对数轴）", fontsize=12)
    ax.set_ylabel("这部分时间里发生的成交，占全天的比例（%）", fontsize=12)
    ax.set_title("越忙的股票，一半成交铺得越开", fontsize=15)
    ax.legend(frameon=False, fontsize=11.5, loc="lower right")
    ax.text(0.03, 0.06, f"最忙的 1% 时间（142 秒）里发生了 {top1_r * 100:.0f}% 的成交",
            transform=ax.transAxes, fontsize=12, color="#333333")
    fig.tight_layout()
    fig.savefig(OUT / "f2_concentration.png", dpi=150)
    plt.close(fig)

    # ---------- 图三 ----------
    stat = df.groupby("q").fano_n.describe(percentiles=[.1, .25, .5, .75, .9])
    x = np.arange(5)
    fig, ax = plt.subplots(figsize=(11.8, 7.2))
    ax.fill_between(x, stat["10%"], stat["90%"], color="#D6EAF8", label="十分位到九十分位")
    ax.fill_between(x, stat["25%"], stat["75%"], color="#85C1E9", label="中间一半")
    ax.plot(x, stat["50%"], "o-", color=C_REAL, lw=2.4, ms=9, label="中位")
    for i in range(5):
        ha, dx = ("right", -0.12) if i == 4 else ("left", 0.12)
        ax.text(i + dx, stat["50%"].iloc[i] + 0.06, f"{stat['50%'].iloc[i]:.2f}", ha=ha, fontsize=13,
                color=C_REAL, fontweight="bold")
    ax.axhline(1, color="#333333", lw=1.3, ls="--")
    ax.text(0.99, 0.10, "虚线 1 = 同样多笔成交随机撒开", transform=ax.transAxes, ha="right", fontsize=12, color="#333333")
    ax.set_xticks(x)
    ax.set_xticklabels([f"第 {i + 1} 组\n每秒 {lam[i]:.2f} 笔" for i in range(5)])
    ax.set_ylim(0, 16)
    ax.set_ylabel("方差 ÷ 均值 ÷（1 + 每秒笔数）", fontsize=12)
    ax.set_title("活跃度差 13 倍，成团几乎一样", fontsize=15)
    ax.legend(frameon=False, fontsize=11.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "f3_vs_activity.png", dpi=150)
    plt.close(fig)

    # ---------- 图四 ----------
    r2d = df.groupby("date").r2.median() * 100
    fig, ax = plt.subplots(figsize=(12.2, 6.8))
    EXT = {20240205, 20240930, 20241008, 20250407}
    cols = ["#C0392B" if d in EXT else "#5D6D7E" for d in DATES]
    ax.bar(range(8), [r2d[d] for d in DATES], color=cols, width=.62)
    for i, d in enumerate(DATES):
        ax.text(i, r2d[d] + 0.4, f"{r2d[d]:.1f}%", ha="center", fontsize=12.5)
    ax.set_xticks(range(8))
    ax.set_xticklabels([LEG[d] for d in DATES], fontsize=11)
    ax.set_ylabel("同一秒的成交能被大盘解释的比例（%）", fontsize=12)
    ax.set_ylim(0, max(r2d) * 1.25)
    ax.set_title("常态日一成不到，天量日三成跟大盘同步", fontsize=15)
    fig.tight_layout()
    fig.savefig(OUT / "f4_common.png", dpi=150)
    plt.close(fig)

    print(f"一半成交落在最忙的 {half_r * 100:.2f}% 时间")
    print(f"最忙 1% 时间承载 {top1_r * 100:.1f}%")
    print("按日 R2：", r2d.round(1).to_dict())
    print("无量纲版中位 %.2f，方差除均值中位 %.1f，r2 中位 %.3f" % (df.fano_n.median(), df.fano.median(), df.r2.median()))
    print("图已写入", OUT)


if __name__ == "__main__":
    main()
