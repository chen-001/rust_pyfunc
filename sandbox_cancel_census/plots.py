# -*- coding: utf-8 -*-
"""第 03 期：挂单的去向（撤单）五张图。"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyze import load

OUT = Path("/home/chenzongwei/hf-observation/2026-10-22-挂单的去向/images")
OUT.mkdir(parents=True, exist_ok=True)
SB = Path("/home/chenzongwei/rust_pyfunc/sandbox_cancel_census")

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

DATES = [20240205, 20240329, 20240813, 20240930, 20241008, 20250407, 20251127, 20260311]
TIT = {20240205: "2024-02-05\n小盘股流动性危机", 20240329: "2024-03-29\n常态 A",
       20240813: "2024-08-13\n地量", 20240930: "2024-09-30\n924 行情",
       20241008: "2024-10-08\n历史天量", 20250407: "2025-04-07\n关税暴跌",
       20251127: "2025-11-27\n常态 B", 20260311: "2026-03-11\n常态 C"}
LEG = {20240205: "02-05 小盘危机", 20240329: "03-29 常态A", 20240813: "08-13 地量",
       20240930: "09-30 924", 20241008: "10-08 天量", 20250407: "04-07 关税",
       20251127: "11-27 常态B", 20260311: "03-11 常态C"}
C = {20240205: "#C0392B", 20240329: "#B8B8B8", 20240813: "#2E86C1", 20240930: "#E67E22",
     20241008: "#7B241C", 20250407: "#8E44AD", 20251127: "#9A9A9A", 20260311: "#757575"}

f = load()
f = f[(f.st == 1) & (f.hi != f.lo) & f.age_ok & f.amt.notna()].copy()
f["share"] = f.cancel_share * 100
print("样本股票日", len(f))

# ---------------- 图一：撤单比有多大
fig, axes = plt.subplots(2, 4, figsize=(19, 8.6), sharex=True, sharey=True)
for ax, d in zip(axes.flat, DATES):
    v = f[f.date == d].share
    ax.hist(v, bins=np.arange(20, 101, 1), color=C[d], alpha=.9, edgecolor="none")
    ax.axvline(50, color="#444444", ls="--", lw=1.3)
    ax.axvline(np.median(v), color="#111111", ls="-", lw=1.5)
    ax.set_title(TIT[d], fontsize=12, pad=6)
    ax.text(.04, .93, f"中位 {np.median(v):.1f}%", transform=ax.transAxes, fontsize=11.5, va="top",
            bbox=dict(fc="white", ec="none", alpha=.85, pad=1.5))
    ax.set_xlim(20, 100)
    ax.set_yticks([])
axes.flat[0].text(.04, .80, "虚线 50%：撤掉的\n和成交的一样多", transform=axes.flat[0].transAxes,
                  fontsize=10, va="top", color="#444444")
fig.supxlabel("撤单量占「撤单量加成交量」的比例 / %", fontsize=13, y=.02)
fig.suptitle("一半的挂单在成交之前就被撤走了", fontsize=17, y=.985)
plt.tight_layout(rect=[0, .045, 1, .95])
plt.savefig(OUT / "f1_cancel_share.png", dpi=125, bbox_inches="tight")
plt.close()

# ---------------- 图二：撤的是大单还是小单
hc = np.zeros(500)
ht = np.zeros(500)
for d in DATES:
    j = json.load(open(SB / f"out/{d}.json"))
    hc += np.array(j["hist_c"], float)
    ht += np.array(j["hist_t"], float)
x = 10 ** (1.5 + .01 * np.arange(500))
edges = np.arange(1.9, 6.01, 0.2)
mid, ratio, cn = [], [], []
for a, b in zip(edges[:-1], edges[1:]):
    m = (np.log10(x) >= a) & (np.log10(x) < b)
    mid.append(10 ** ((a + b) / 2))
    ratio.append(hc[m].sum() / max(ht[m].sum(), 1))
    cn.append(hc[m].sum() / hc.sum() * 100)
mid, ratio = np.array(mid), np.array(ratio)

fig, ax = plt.subplots(figsize=(15, 8.2))
ax.plot(mid, ratio, color="#C0392B", lw=2.6, marker="o", ms=5)
ax.axhline(1, color="#444444", ls="--", lw=1.3)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(80, 1.1e6)
ax.set_xlabel("一笔的股数（对数刻度）", fontsize=12.5)
ax.set_ylabel("撤单笔数 ÷ 成交笔数（对数刻度）", fontsize=12.5)
ax.set_title("一笔越大，撤的越多：一万五千股打平，五十万股的单撤掉的是成交掉的七倍", fontsize=15,
             loc="left", pad=12)
i1 = np.argmin(np.abs(ratio - 1))
ax.annotate(f"{mid[i1]:,.0f} 股打平", xy=(mid[i1], 1), xytext=(mid[i1] * .12, 1.35), fontsize=11.5,
            arrowprops=dict(arrowstyle="->", color="#666666", lw=1))
ax.annotate("一百股上下的一笔\n撤的只有成交的四分之一", xy=(100, ratio[0]), xytext=(150, .16),
            fontsize=11.5, arrowprops=dict(arrowstyle="->", color="#666666", lw=1))
ax.annotate(f"五十万股以上\n撤单是成交的 {ratio[-1]:.1f} 倍", xy=(mid[-1], ratio[-1]), xytext=(6e4, 1.6),
            fontsize=11.5, arrowprops=dict(arrowstyle="->", color="#666666", lw=1))
ax.grid(which="both", alpha=.3)
plt.tight_layout()
plt.savefig(OUT / "f2_cancel_size.png", dpi=125, bbox_inches="tight")
plt.close()
print("按规模档 撤/成：", np.round(ratio, 2), "\n档中点：", np.round(mid).astype(int))

# ---------------- 图三：一天之内撤单占比怎么变
fig, (a1, a2) = plt.subplots(2, 1, figsize=(16.5, 10), sharex=True,
                             gridspec_kw={"height_ratios": [1, 1], "hspace": .2})
sh = {}
for d in DATES:
    t = f[f.date == d]
    cv = np.stack(t.b5_cvol.values).sum(0)
    tv = np.stack(t.b5_tvol.values).sum(0)
    s = cv / (cv + tv) * 100
    sh[d] = s
    a1.plot(s, color=C[d], lw=1.9, label=LEG[d])
    a2.plot(s / s.mean(), color=C[d], lw=1.9)
for ax in (a1, a2):
    ax.set_xlim(0, 47)
    ax.axvline(23.5, color="#999999", lw=1, ls="--")
a1.set_ylabel("该 5 分钟桶的撤单占比 / %", fontsize=12.5)
a1.set_title("开盘那一桶四成多，之后一路往上爬，最后五分钟猛掉下来", fontsize=14, loc="left", pad=8)
a2.axhline(1, color="#666666", ls=":", lw=1.1)
a2.set_ylabel("除以当天的全天水平（倍）", fontsize=12.5)
a2.set_ylim(.68, 1.28)
a2.set_title("把每天的水平差除掉，八天的形状大体重合", fontsize=14, loc="left", pad=8)
ticks = list(range(0, 48, 6)) + [47]
lbl = ["9:30", "10:00", "10:30", "11:00", "11:30 | 13:00", "13:30", "14:00", "14:30", "15:00"]
a2.set_xticks(ticks)
a2.set_xticklabels(lbl)
fig.legend(*a1.get_legend_handles_labels(), fontsize=11, ncol=4, loc="lower center", frameon=False,
           bbox_to_anchor=(.5, .004))
fig.suptitle("一天之内，挂单被撤掉的比例先升后降", fontsize=16.5, y=.965)
plt.tight_layout(rect=[0, .06, 1, .945])
plt.savefig(OUT / "f3_intraday.png", dpi=125, bbox_inches="tight")
plt.close()
print("每桶撤单占比 首桶/第23桶/末桶：")
for d in DATES:
    print(f"  {d}  {sh[d][0]:.1f}  {sh[d][23]:.1f}  {sh[d][-1]:.1f}  全天 {sh[d].sum() / 48:.1f}")

# ---------------- 图四：买卖两边不对称
fig, ax = plt.subplots(figsize=(14.5, 8))
labels = ["跌超 5%", "跌 1-5%", "涨跌不到 1%", "涨 1-5%", "涨超 5%"]
bins = [-1, -.05, -.01, .01, .05, 1]
f["grp"] = pd.cut(f.ret, bins=bins, labels=labels)
data, pos = [], []
for lab in labels:
    t = f[f.grp == lab]
    v = (t.cvol_b / (t.cvol_b + t.cvol_s)).dropna() * 100
    data.append(v.values)
    pos.append(len(data))
bp = ax.boxplot(data, positions=pos, widths=.5, showfliers=False, whis=(10, 90), patch_artist=True,
                medianprops=dict(color="#111111", lw=2.2),
                whiskerprops=dict(color="#888888"), capprops=dict(color="#888888"))
for patch in bp["boxes"]:
    patch.set_facecolor("#DCDCDC")
    patch.set_edgecolor("#777777")
ax.axhline(50, color="#C0392B", ls="--", lw=1.4)
ax.set_xticks(pos)
ax.set_xticklabels([f"{lab}\n{len(data[i]):,} 个股票日" for i, lab in enumerate(labels)], fontsize=11.5)
ax.set_ylabel("撤买单占「撤买加撤卖」的比例 / %", fontsize=12.5)
ax.set_ylim(28, 73)
ax.set_title("跌得越多，撤掉的卖单越多；涨得越多，撤掉的买单越多", fontsize=15, loc="left", pad=12)
ax.text(5.36, 50.6, "对半", fontsize=10.5, color="#C0392B")
for i, d in enumerate(data):
    ax.text(pos[i], 29.4, f"中位 {np.median(d):.1f}%", ha="center", fontsize=11.5)
plt.tight_layout()
plt.savefig(OUT / "f4_side.png", dpi=125, bbox_inches="tight")
plt.close()

# ---------------- 图五：撤单是不是成批的
fig, ax = plt.subplots(figsize=(14.5, 8))
name = ["同一毫秒\n1 笔", "2 笔", "3 到 10 笔", "11 笔以上"]
bn = np.stack(f.batch_n.values).sum(0).astype(float)
bv = np.stack(f.batch_vol.values).sum(0)
bt = np.stack(f.batch_tick.values).sum(0).astype(float)
x = np.arange(4)
w = .26
for off, v, lab, col in [(-w, bn / bn.sum() * 100, "批次占比", "#4A6FA5"),
                         (0, bt / bt.sum() * 100, "撤单笔数占比", "#C0392B"),
                         (w, bv / bv.sum() * 100, "撤单量占比", "#E67E22")]:
    bars = ax.bar(x + off, v, w, label=lab, color=col, alpha=.9, edgecolor="none")
    for b, val in zip(bars, v):
        ax.text(b.get_x() + b.get_width() / 2, val + 1.2, f"{val:.1f}", ha="center", fontsize=10.5)
ax.set_xticks(x)
ax.set_xticklabels(name, fontsize=12)
ax.set_ylabel("占全部撤单的比例 / %", fontsize=12.5)
ax.set_ylim(0, 100)
ax.legend(fontsize=11.5, frameon=False, loc="upper right")
ax.set_title("成批撤的笔数占三分之一，但只占两成的量：一批一批撤的是小单", fontsize=14.5, loc="left", pad=12)
plt.tight_layout()
plt.savefig(OUT / "f5_batch.png", dpi=125, bbox_inches="tight")
plt.close()
print("图已生成", *sorted(p.name for p in OUT.glob("*.png")), sep="\n  ")
