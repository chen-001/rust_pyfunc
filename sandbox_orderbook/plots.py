# -*- coding: utf-8 -*-
"""盘口深度一期：三张图 + 数字汇总。"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd

OUT = Path("/home/chenzongwei/hf-observation/2026-10-20-盘口有多深/images")
OUT.mkdir(parents=True, exist_ok=True)
SB = Path("/home/chenzongwei/rust_pyfunc/sandbox_orderbook")

for f in fm.findSystemFonts():
    if "NotoSansCJK" in f or "NotoSansSC" in f:
        fm.fontManager.addfont(f)
plt.rcParams["font.family"] = ["Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "white"
plt.rcParams["axes.edgecolor"] = "#BBBBBB"
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.color"] = "#E6E6E6"
plt.rcParams["axes.axisbelow"] = True

DAYS = [20240205, 20240329, 20240813, 20240930, 20241008, 20250407, 20251127, 20260311]
LABEL = {20240205: "02-05 小盘危机", 20240329: "03-29 常态A", 20240813: "08-13 地量",
         20240930: "09-30 924", 20241008: "10-08 天量", 20250407: "04-07 关税",
         20251127: "11-27 常态B", 20260311: "03-11 常态C"}
COLOR = {20240205: "#C0392B", 20240329: "#B8B8B8", 20240813: "#2E86C1", 20240930: "#E67E22",
         20241008: "#7B241C", 20250407: "#8E44AD", 20251127: "#B8B8B8", 20260311: "#B8B8B8"}

f = pd.read_parquet(SB / "out/panel.parquet")
raw = pd.concat([pd.DataFrame(json.load(open(SB / f"out/{d}.json"))["stocks"]).assign(date=d) for d in DAYS])
raw["code"] = raw["code"].astype(str).str.zfill(6)

# ---------------- 图 1：快照数 + 间隔分布
fig = plt.figure(figsize=(13, 9.8))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 0.95], hspace=.42, wspace=.22)
a = fig.add_subplot(gs[0, 0])
b = fig.add_subplot(gs[0, 1])
cc = fig.add_subplot(gs[1, :])

a.hist(f.n_snap, bins=np.arange(0, 5050, 50), color="#4A6FA5", alpha=.85, edgecolor="none")
med = f.n_snap.median()
a.axvline(4802, color="#C0392B", ls="--", lw=1.4)
a.axvline(med, color="#222222", ls="-", lw=1.4)
a.set_ylim(0, 4100)
a.text(4788, 4020, "满额 4,802", ha="right", va="top", fontsize=10, color="#C0392B")
a.annotate(f"中位 {med:,.0f}", xy=(med, a.get_ylim()[1]*.40), xytext=(2050, a.get_ylim()[1]*.50),
           fontsize=11, color="#222222", arrowprops=dict(arrowstyle="->", color="#222222", lw=1))
a.set_xlabel("一只股票一个交易日的快照个数")
a.set_ylabel("股票日占比")
a.set_title("一只股票一天发 4,214 个快照，比满额少一成", fontsize=12.5, loc="left", pad=10)

g = f.groupby("date")
ns = g.n_snap.median().reindex(DAYS)
b.bar(range(8), ns.values, color=[COLOR[d] for d in DAYS], width=.62)
b.set_xticks(range(8)); b.set_xticklabels([LABEL[d] for d in DAYS], rotation=35, ha="right", fontsize=9)
b.set_ylabel("中位快照数"); b.set_ylim(0, 5200)
b.axhline(4802, color="#C0392B", ls="--", lw=1)
b.text(7.45, 4832, "满额 4,802", ha="right", va="bottom", fontsize=9, color="#C0392B")
for i, v in enumerate(ns.values):
    b.text(i, v + 90, f"{v:,.0f}", ha="center", fontsize=9)
b.set_title("地量那天，中位股票只发 3,243 个快照", fontsize=12.5, loc="left", pad=10)

# 间隔分布
gap = np.zeros(32)
for dd in DAYS:
    for s in json.load(open(SB / f"out/{dd}.json"))["stocks"]:
        gap += np.array(s["gap_hist"])
share = gap / gap.sum() * 100
xs = np.arange(2, 32)
vals = share[2:32]
vals[-1] = share[31]
cc.bar(xs, vals, color="#7B8FA8", width=.72)
cc.bar([3], [share[3]], color="#C0392B", width=.72)
cc.set_xticks(list(xs)); cc.set_xticklabels([str(x) for x in xs[:-1]] + ["30+"], fontsize=9)
cc.set_xlabel("相邻两个快照的间隔（秒，按整数秒分箱）")
cc.set_ylabel("占全部间隔的百分比（%）")
cc.set_ylim(0, 100)
cc.text(3, share[3] + 2, f"{share[3]:.1f}%", ha="center", fontsize=11, color="#222222")
for x in (2, 4, 6, 9, 12, 15):
    v = share[x]
    cc.text(x, v + 1.2, f"{v:.2f}%" if v < 1 else f"{v:.1f}%", ha="center", rotation=90,
            fontsize=8.5, color="#333333", va="bottom")
cc.text(17.5, 60, "2 秒和 4 秒几乎一样多，各占 0.57%，是时间戳的毫秒抖动，不是漏发。\n"
                  "6 秒、9 秒、12 秒才是真的少发了一条到三条。\n"
                  "5 秒、7 秒、8 秒、10 秒这些更细的柱子都不到 0.05%。", fontsize=10, color="#555555")
cc.set_title("十个间隔里有将近九个正好是 3 秒", fontsize=12.5, loc="left", pad=10)
fig.savefig(OUT / "f1_快照数.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ---------------- 图 2：深度量级
fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))
a = ax[0]
amt_wan = f.amt50 / 1e4
bins = np.logspace(np.log10(5), np.log10(200000), 90)
a.hist(amt_wan.clip(5, 200000), bins=bins, color="#4A6FA5", alpha=.85, edgecolor="none")
a.set_xscale("log")
for q, name, c in [(.1, "P10", "#777777"), (.5, "中位", "#111111"), (.9, "P90", "#C0392B")]:
    v = amt_wan.quantile(q)
    a.axvline(v, color=c, ls="--" if q != .5 else "-", lw=1.3)
    a.text(v*1.10, a.get_ylim()[1]*(.92 - .14*[.1, .5, .9].index(q)), f"{name} {v:,.0f} 万", fontsize=10, color=c,
           bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=.85))
a.set_xlabel("十档买卖挂单合计金额（万元，对数横轴）")
a.set_ylabel("股票日占比")
a.set_title("十档挂单金额中位 320 万元，P90 是 P10 的 19 倍", fontsize=12.5, loc="left", pad=10)

b = ax[1]
sh_wan = f.sh50 / 1e4
bins2 = np.logspace(np.log10(.3), np.log10(5000), 90)
b.hist(sh_wan.clip(.3, 5000), bins=bins2, color="#5D8A66", alpha=.85, edgecolor="none")
b.set_xscale("log")
for q, name, c in [(.1, "P10", "#777777"), (.5, "中位", "#111111"), (.9, "P90", "#C0392B")]:
    v = sh_wan.quantile(q)
    b.axvline(v, color=c, ls="--" if q != .5 else "-", lw=1.3)
    b.text(v*1.10, b.get_ylim()[1]*(.92 - .14*[.1, .5, .9].index(q)), f"{name} {v:,.1f} 万", fontsize=10, color=c,
           bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=.85))
b.set_xlabel("十档买卖挂单合计股数（万股，对数横轴）")
b.set_ylabel("股票日占比")
b.set_title("换成股数，两端差 103 倍", fontsize=12.5, loc="left", pad=10)
fig.tight_layout()
fig.savefig(OUT / "f3_深度量级.png", dpi=150)
plt.close(fig)

# ---------------- 图 3：深度构成与买卖
fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))
a = ax[0]
S = np.array([x for x in raw["shr"] if isinstance(x, list) and any(x)])
m = np.median(S, axis=0); m = m / m.sum() * 100
xs = np.arange(1, 11)
a.bar(xs, m, color="#4A6FA5", width=.68)
for x, v in zip(xs, m):
    a.text(x, v + .18, f"{v:.1f}", ha="center", fontsize=9)
a.set_xticks(xs); a.set_xlabel("档位（1 是最靠近成交价的一档）")
a.set_ylabel("该档在十档挂单里占的份额（%）")
a.set_ylim(0, 12.5)
a.axhline(10, color="#999999", ls=":", lw=1)
a.text(10.4, 10.15, "十档平均分摊 = 10%", ha="right", fontsize=9, color="#777777")
a.set_title("十档中间厚两头薄，最厚和最薄只差两个百分点", fontsize=12.5, loc="left", pad=10)

b = ax[1]
ba = f.ba.replace([np.inf, -np.inf], np.nan).dropna()
bins3 = np.logspace(np.log10(.1), np.log10(10), 80)
b.hist(ba.clip(.1, 10), bins=bins3, color="#8E6BB5", alpha=.85, edgecolor="none")
b.set_xscale("log")
b.axvline(1, color="#111111", lw=1.4)
med_ba = ba.median()
b.axvline(med_ba, color="#C0392B", ls="--", lw=1.3)
b.text(med_ba*1.1, b.get_ylim()[1]*.88, f"中位 {med_ba:.2f}", fontsize=10, color="#C0392B")
b.set_xlabel("买盘挂单 ÷ 卖盘挂单（对数横轴，1 表示两边一样厚）")
b.set_ylabel("股票日占比")
b.set_title("买卖大体对称，两成股票日相差一倍以上", fontsize=12.5, loc="left", pad=10)
fig.tight_layout()
fig.savefig(OUT / "f4_深度构成.png", dpi=150)
plt.close(fig)

print("图已存", OUT)
for p in sorted(OUT.glob("*.png")):
    print(p.name, p.stat().st_size)

# ---------------- 图 4：午休快照
import json as _json
D8 = [20240205, 20240329, 20240813, 20240930, 20241008, 20250407, 20251127, 20260311]
_tot = np.zeros(331); _n = 0; _lunch = {"沪": [], "深": []}
for _d in D8:
    _j = _json.load(open(SB / f"out/{_d}.json"))
    for s in _j["stocks"]:
        if s["mcnt"]:
            _tot += np.array(s["mcnt"]); _n += 1
            _lunch["沪" if s["code"][0] == "6" else "深"].append(s["n_lunch"])
avg = _tot / _n

fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))
a = ax[0]
mins = np.arange(331)
a.axvspan(121, 210, color="#EFE7DA", lw=0)
a.plot(mins, avg, color="#2F5D8C", lw=1.5)
a.axhline(20, color="#C0392B", ls="--", lw=1)
a.set_xticks([0, 60, 120, 210, 270, 330])
a.set_xticklabels(["9:30", "10:30", "11:30", "13:00", "14:00", "15:00"])
a.set_xlabel("一天里的时间（午休段用浅色标出）")
a.set_ylabel("每只股票每分钟的快照数")
a.set_ylim(0, 22.5)
a.text(122, 19.2, "每 3 秒一个 = 每分钟 20 个", color="#C0392B", fontsize=9.5)
a.annotate(f"午休每分钟 {np.median(avg[121:210]):.1f} 个\n90 分钟里一直没停",
           xy=(165, 2.0), xytext=(150, 9.0), fontsize=10, ha="center",
           arrowprops=dict(arrowstyle="->", color="#555555", lw=1))
a.annotate(f"交易时段每分钟 {np.median(avg[1:121]):.0f} 个", xy=(70, 17.2), xytext=(30, 12.0),
           fontsize=10, arrowprops=dict(arrowstyle="->", color="#555555", lw=1))
a.set_title("午休没有停发，只是从每 3 秒一个放慢到每分钟一两个", fontsize=12.5, loc="left", pad=10)

b = ax[1]
vals = [np.median(_lunch["沪"]), np.median(_lunch["深"])]
bars = b.bar(["沪市（6 开头）", "深市（0 和 3 开头）"], vals, color=["#C0392B", "#2E86C1"], width=.55)
for r, v, lab in zip(bars, vals, ["每 30 秒一个", "每 60 秒一个"]):
    b.text(r.get_x() + r.get_width()/2, v + 4, f"{v:.0f} 个\n{lab}", ha="center", fontsize=10)
b.axhline(1800, color="#999999", ls=":", lw=1.2)
b.text(1.45, 1830, "同样 90 分钟的交易时段是 1,800 个", ha="right", fontsize=9, color="#777777")
b.set_ylim(0, 2050)
b.set_ylabel("午休 90 分钟里的快照数")
b.set_title("同一个午休，沪市发了深市的两倍", fontsize=12.5, loc="left", pad=10)
fig.tight_layout()
fig.savefig(OUT / "f2_午休快照.png", dpi=150)
plt.close(fig)
print("f4 好了", np.median(_lunch["沪"]), np.median(_lunch["深"]))
