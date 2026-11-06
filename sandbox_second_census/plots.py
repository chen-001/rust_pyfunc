# -*- coding: utf-8 -*-
"""第 06 期：成交落在每分钟的第几秒。五张图。"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyze import load, curves, seg_curves

OUT = Path("/home/chenzongwei/rust_pyfunc/sandbox_second_census/figs")
OUT.mkdir(exist_ok=True)

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
C_ALT = "#1F618D"
C_SUR = "#95A5A6"
CQ = ["#AED6F1", "#7FB3D5", "#5499C7", "#2E86C1", "#1B4F72"]
X = np.arange(60)


def band(ax, M, color=C_SUR, alpha=0.22):
    ax.fill_between(X, np.nanpercentile(M, 10, axis=0), np.nanpercentile(M, 90, axis=0),
                    color=color, alpha=alpha, linewidth=0, label="十分位到九十分位")


def style(ax, ylab, title=None):
    ax.set_xlim(-0.5, 59.5)
    ax.set_xticks([0, 10, 20, 30, 40, 50, 59])
    ax.set_xlabel("一分钟里的第几秒")
    ax.set_ylabel(ylab)
    if title:
        ax.set_title(title, loc="left", fontsize=12, pad=10)


def fig1(df, C):
    m = df.code.str[0]
    groups = [("沪市", m == "6", C_ALT), ("深市主板", m == "0", C_REAL), ("创业板", m == "3", "#E67E22")]
    fig, ax = plt.subplots(figsize=(10, 4.6))
    for name, mask, col in groups:
        ax.plot(X, np.median(C["inten"][mask.values], 0), color=col, lw=2.0, label=name)
    ax.axhline(1, color="#333333", lw=1, ls="--", label="均匀线")
    style(ax, "每秒平均笔数 ÷ 全天每秒平均", "深市主板每 3 秒跳一次，创业板晚一秒，沪市没有这个节奏")
    ax.legend(frameon=False, ncol=4, fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "f1_seconds.png", dpi=170)
    plt.close(fig)


def fig2(C):
    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.plot(X, np.nanmedian(C["disp"], 0), color=C_REAL, lw=2.0, label="全天量的波动")
    ax.plot(X, np.nanmedian(C["share_cv"], 0), color=C_ALT, lw=2.0, label="扣掉分钟总量后")
    ax.set_title("第 0 秒的量最不稳，份额反而最稳", loc="left", fontsize=12, pad=10)
    style(ax, "波动程度")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "f2_disp.png", dpi=170)
    plt.close(fig)


def fig3(C):
    fig, axes = plt.subplots(2, 1, figsize=(10, 6.4), sharex=True)
    ax = axes[0]
    ax.set_title("秒位上的差别全在撮合次数，每笔金额几乎不变", loc="left", fontsize=12, pad=10)
    ax.plot(X, np.nanmedian(C["inten"], 0), color=C_REAL, lw=2.0, label="记录数")
    ax.plot(X, np.nanmedian(C["inten_t"], 0), color=C_ALT, lw=2.0, label="不同的 10 毫秒刻度")
    ax.axhline(1, color="#333333", lw=1, ls="--")
    ax.set_xlim(-0.5, 59.5)
    ax.set_ylabel("相对全天平均")
    ax.legend(frameon=False, fontsize=9, title=None)
    ax = axes[1]
    ax.plot(X, np.nanmedian(C["amt_mult"], 0), color=C_REAL, lw=2.0, label="中位股票日")
    ax.axhline(1, color="#333333", lw=1, ls="--", label="全天平均每笔")
    ax.set_xlim(-0.5, 59.5)
    ax.set_xticks([0, 10, 20, 30, 40, 50, 59])
    ax.set_xlabel("一分钟里的第几秒")
    ax.set_ylabel("每笔金额 ÷ 全天平均")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "f3_pieces.png", dpi=170)
    plt.close(fig)


def fig4(df, C):
    m = df.code.str[0]
    groups = [("沪市", "6"), ("深市主板", "0"), ("创业板", "3")]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0), sharey=True)
    for ax, (name, p) in zip(axes, groups):
        mask = (m == p).values
        for k, col, lab in [(0, C_SUR, "最冷的两成"), (4, C_REAL, "最忙的两成")]:
            g = mask & (df.q == k).values
            ax.plot(X, np.median(C["inten"][g], 0), color=col, lw=1.9, label=lab)
        ax.axhline(1, color="#333333", lw=1, ls="--")
        ax.set_title(name, loc="left", fontsize=11)
        ax.set_xlim(-0.5, 59.5)
        ax.set_xticks([0, 20, 40, 59])
        ax.set_xlabel("一分钟里的第几秒")
    axes[0].set_ylabel("每秒平均笔数 ÷ 全天每秒平均")
    fig.suptitle("同一市场里，最冷和最忙的股票，秒位形状几乎重合", x=0.01, ha="left", fontsize=12)
    axes[0].legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "f4_activity.png", dpi=170)
    plt.close(fig)


def fig5(df, A):
    S = seg_curves(df, A)
    names = ["开盘 5 分钟", "上午其余", "午后", "最后 5 分钟"]
    cols = ["#E67E22", C_ALT, C_SUR, C_REAL]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    for k in range(4):
        ax.plot(X, np.nanmedian(S[:, k, :], 0), color=cols[k], lw=2.0, label=names[k])
    ax.axhline(1, color="#333333", lw=1, ls="--")
    style(ax, "每一段的每秒平均笔数 ÷ 该段平均", "开盘第一分钟的第 0 秒最挤，收盘前最后一分钟整体最平")
    ax.legend(frameon=False, ncol=4, fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "f5_segments.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    df, A = load()
    C = curves(df, A)
    fig1(df, C)
    fig2(C)
    fig3(C)
    fig4(df, C)
    fig5(df, A)
    print("股票日", len(df), "日期", sorted(df.date.unique()))
    print("图一 3 秒整倍数中位", round(float(np.median(np.median(C["inten"], 0)[::3])), 3),
          "其余", round(float(np.median(np.delete(np.median(C["inten"], 0), np.s_[::3]))), 3))
    print("图二 中位范围", np.round(np.nanmedian(C["disp"], 0)[[0, 1, 2, 10, 30, 59]], 2))
    print("图三 每笔金额倍数范围", np.round(np.nanmin(np.nanmedian(C["amt_mult"], 0)), 3),
          np.round(np.nanmax(np.nanmedian(C["amt_mult"], 0)), 3))
    print("完成")
