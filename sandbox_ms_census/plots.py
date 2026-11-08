# -*- coding: utf-8 -*-
"""第 07 期：毫秒里的成交。六张图 + numbers.json（文中每个数都从这里出）。
统一口径：全部用「合计」，即所选股票日的计数先加总再算比例。单只股票一个格子的计数太少，取中位数会变成整数台阶。"""
import json
import matplotlib; matplotlib.use("Agg")
import matplotlib.font_manager as fm, matplotlib.pyplot as plt
import numpy as np
from load import *

OUT = SB / "figs"; OUT.mkdir(exist_ok=True)
for fp in fm.findSystemFonts():
    if "NotoSansCJK" in fp or "NotoSansSC" in fp: fm.fontManager.addfont(fp)
plt.rcParams.update({"font.family": ["Noto Sans CJK SC", "DejaVu Sans"], "axes.unicode_minus": False,
    "figure.facecolor": "white", "axes.facecolor": "white", "axes.edgecolor": "#BBBBBB", "axes.grid": True,
    "grid.color": "#E9E9E9", "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False})
C = {"6": "#1F618D", "0": "#C0392B", "3": "#E67E22"}
NAME = {"6": "沪市", "0": "深市主板", "3": "创业板"}
GREY = "#95A5A6"; INK = "#333333"

Ms, As = {}, {}
for d in DATES: Ms[d], As[d] = load_day(d)
mk = np.concatenate([[c[0] for c in Ms[d]["codes"]] for d in DATES])
dates_arr = np.concatenate([[d] * Ms[d]["n_valid"] for d in DATES])
def G(k): return np.vstack([As[d][k] for d in DATES]).astype(float)
rec, ev, ev0, evx, amt = G("rec"), G("ev"), G("ev0"), G("evx"), G("amt100")
snap, occ, clk, gap, shuf = G("snap"), G("occ"), G("clk"), G("gap"), G("shuf")
pair, sd, sdv, bday, bmin, S = G("pair"), G("sd"), G("sdv"), G("bday"), G("bmin"), G("scal")
N = {}
N["股票日"] = int(len(S)); N["记录"] = int(S[:, 0].sum()); N["主动单"] = int(S[:, 1].sum()); N["10毫秒"] = int(S[:, 2].sum())
N["每张主动单记录数"] = S[:, 0].sum() / S[:, 1].sum()
N["10毫秒里含两张及以上主动单的占比"] = S[:, 3].sum() / S[:, 2].sum()
N["每个有成交的10毫秒主动单数"] = S[:, 1].sum() / S[:, 2].sum()
N["端点秒记录"] = int(S[:, 9].sum()); N["时间倒序记录"] = int(S[:, 8].sum())
D = daily()
r = np.concatenate([As[d]["amt_all"] / 1000 / D["amt"].loc[d, Ms[d]["codes"]].values for d in DATES])
N["对账中位比值"] = float(np.nanmedian(r)); N["对账最小比值"] = float(np.nanmin(r)); N["对账最大比值"] = float(np.nanmax(r)); N["对账超2%占比"] = float(np.mean(np.abs(r - 1) > .02))

CELLX = np.arange(100) * 10
# ---- 图一：100 个格子
def fig1():
    p = rec.sum(0) / rec.sum() * 100; a = amt.sum(0) / amt.sum() * 100
    imx, imn = int(p.argmax()), int(p.argmin()); jmx, jmn = int(a.argmax()), int(a.argmin())
    N.update({"图一_笔数最多格ms": imx * 10, "图一_笔数最多占比": p.max(), "图一_笔数最少格ms": imn * 10, "图一_笔数最少占比": p.min(), "图一_笔数最多最少比": p.max() / p.min(),
              "图一_金额最多格ms": jmx * 10, "图一_金额最多占比": a.max(), "图一_金额最少格ms": jmn * 10, "图一_金额最少占比": a.min(), "图一_金额最多最少比": a.max() / a.min(),
              "图一_前半秒笔数占比": p[:50].sum(), "图一_后半秒笔数占比": p[50:].sum(), "图一_前半秒金额占比": a[:50].sum(), "图一_后半秒金额占比": a[50:].sum()})
    days = {}
    for d in DATES:
        x = As[d]["rec"].sum(0).astype(float); x = x / x.sum() * 100
        days[str(d)] = (int(x.argmax()) * 10, float(x.max()), int(x.argmin()) * 10, float(x.min()), float(x.max() / x.min()), float(x[:50].sum()))
    N["图一_逐日(最多ms,占比,最少ms,占比,比,前半秒)"] = days
    M = np.vstack([As[d]["rec"].sum(0) / As[d]["rec"].sum() * 100 for d in DATES])
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.6), sharex=True)
    fig.suptitle(f"一秒的 100 个格子并不平均：最挤的格子比最冷的多 {p.max()/p.min()-1:.0%}，前半秒比后半秒挤", x=0.01, ha="left", fontsize=13)
    for ax, v, lab, i0, i1 in [(axes[0], p, "每格成交笔数占全天的 %", imx, imn), (axes[1], a, "每格成交金额占全天的 %", jmx, jmn)]:
        col = ["#C8D3DC"] * 100; col[i0] = "#C0392B"; col[i1] = "#1F618D"
        ax.bar(CELLX + 5, v, width=8.6, color=col, linewidth=0)
        ax.axhline(1, color=INK, lw=1, ls="--"); ax.set_ylabel(lab); ax.set_ylim(0.8, 1.2)
        ax.text(CELLX[i0] + 5, v[i0] + .008, f"最多 {i0*10}–{i0*10+9} 毫秒\n{v[i0]:.3f}%", ha="center", va="bottom", fontsize=9, color="#C0392B")
        ax.annotate(f"最少 {i1*10}–{i1*10+9} 毫秒，{v[i1]:.3f}%", xy=(CELLX[i1] + 5, v[i1]), xytext=(CELLX[i1] + 5, 1.09), ha="center", va="bottom", fontsize=9, color="#1F618D", arrowprops=dict(arrowstyle="-", color="#1F618D", lw=.8))
        ax.text(998, 1.004, "平均每格 1%", ha="right", va="bottom", fontsize=9, color=INK)
    axes[1].set_xlabel("一秒里的毫秒位置（每格 10 毫秒）"); axes[1].set_xlim(0, 1000); axes[1].set_xticks(range(0, 1001, 100))
    fig.tight_layout(rect=(0, 0, 1, .96)); fig.savefig(OUT / "f1_cells.png", dpi=170); plt.close(fig)

# ---- 图二：第 0 秒 vs 其余秒
def fig2():
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.7), sharey=True)
    fig.suptitle("沪市一秒四个高峰，每隔 250 毫秒一个；深市的高峰在一秒的开头，整分钟第 0 秒多出来的量也不在头 30 毫秒", x=0.01, ha="left", fontsize=13)
    for ax, m in zip(axes, "603"):
        i = mk == m; base = ev[i].sum() / 1_422_000
        y0 = ev0[i].sum(0) / 237 / base; yo = (ev[i].sum(0) - ev0[i].sum(0)) / (14220 - 237) / base
        ax.plot(CELLX + 5, y0, color=C[m], lw=1.6, label="每分钟第 0 秒")
        ax.plot(CELLX + 5, yo, color=GREY, lw=1.6, label="其余 59 秒")
        ax.axhline(1, color=INK, lw=1, ls="--"); ax.set_title(NAME[m], loc="left", fontsize=11); ax.set_xlim(0, 1000); ax.set_xticks(range(0, 1001, 250)); ax.set_xlabel("一秒里的毫秒位置"); ax.legend(frameon=False, fontsize=9)
        N[f"图二_{NAME[m]}_第0秒前3格均值"] = y0[:3].mean(); N[f"图二_{NAME[m]}_第0秒全秒均值"] = y0.mean(); N[f"图二_{NAME[m]}_第0秒最高格ms"] = int(y0.argmax()) * 10; N[f"图二_{NAME[m]}_第0秒最高"] = y0.max()
        N[f"图二_{NAME[m]}_其余秒最高格ms"] = int(yo.argmax()) * 10; N[f"图二_{NAME[m]}_其余秒最高"] = yo.max(); N[f"图二_{NAME[m]}_其余秒最低"] = yo.min(); N[f"图二_{NAME[m]}_其余秒最低格ms"] = int(yo.argmin()) * 10
        N[f"图二_{NAME[m]}_其余秒前3格均值"] = yo[:3].mean()
        if m == "6":
            N["图二_沪市_四峰位置ms"] = [int(a) * 10 for a in [yo[k:k + 25].argmax() + k for k in (0, 25, 50, 75)]]
            N["图二_沪市_四峰高度"] = [float(yo[k:k + 25].max()) for k in (0, 25, 50, 75)]
    axes[0].set_ylabel("相对全天每格平均的倍数")
    fig.tight_layout(rect=(0, 0, 1, .95)); fig.savefig(OUT / "f2_second0.png", dpi=170); plt.close(fig)
    # 跨日：沪市 4Hz 周期，深市早高峰
    rows = {}
    for d in DATES:
        o = {}
        mkd = np.array([c[0] for c in Ms[d]["codes"]])
        for m in "603":
            x = As[d]["ev"][mkd == m].sum(0).astype(float); x /= x.mean()
            f = np.abs(np.fft.rfft(x - x.mean()))
            o[NAME[m]] = dict(主频=int(f[1:].argmax()) + 1, 最高格ms=int(x.argmax()) * 10, 最高=float(x.max()), 最低格ms=int(x.argmin()) * 10, 最低=float(x.min()))
        rows[str(d)] = o
    N["图二_逐日"] = rows
    M = {}
    for m in "603":
        i = mk == m; a = ev[i]; rate = a / a.mean(1, keepdims=True); med = np.median(rate, 0); N[f"图二_{NAME[m]}_中位与合计曲线相关"] = float(np.corrcoef(med, a.sum(0))[0, 1])
    N["图二_沪市逐日主频(次每秒)"] = {str(d): rows[str(d)]["沪市"]["主频"] for d in DATES}

# ---- 图三：对齐到快照
def fig3():
    X3 = np.arange(300) * 10 + 5
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    fig.suptitle("成交跟着快照走：沪市三组股票的高峰跟着各自的快照挪；深市主板快照后 150 毫秒陡增，创业板要到 1 秒后", x=0.01, ha="left", fontsize=13)
    ax = axes[0]
    for m in "603":
        i = mk == m; y = snap[i].sum(0) / occ[i].sum(0); y = y / y.mean(); ax.plot(X3, y, color=C[m], lw=1.4, label=NAME[m])
        N[f"图三_{NAME[m]}_第1秒均值"] = y[:100].mean(); N[f"图三_{NAME[m]}_第2秒均值"] = y[100:200].mean(); N[f"图三_{NAME[m]}_第3秒均值"] = y[200:].mean()
        dy = np.diff(y); top = np.argsort(-dy)[:4] + 1; N[f"图三_{NAME[m]}_最大四次陡增(到达ms,前值,后值)"] = [[int(k) * 10, float(y[k - 1]), float(y[k])] for k in top]
        N[f"图三_{NAME[m]}_最高格ms"] = int(y.argmax()) * 10; N[f"图三_{NAME[m]}_最高"] = y.max()
        N[f"图三_{NAME[m]}_140ms"] = y[14]; N[f"图三_{NAME[m]}_150ms"] = y[15]; N[f"图三_{NAME[m]}_990ms"] = y[99]; N[f"图三_{NAME[m]}_1000ms"] = y[100]; N[f"图三_{NAME[m]}_1990ms"] = y[199]; N[f"图三_{NAME[m]}_2000ms"] = y[200]
    ax.axhline(1, color=INK, lw=1, ls="--"); ax.set_xlim(0, 3000); ax.set_xticks(range(0, 3001, 500)); ax.set_xlabel("距本股上一个快照的毫秒数"); ax.set_ylabel("相对 3 秒平均的倍数"); ax.legend(frameon=False, fontsize=9); ax.set_title("按自家快照对齐", loc="left", fontsize=11)
    ax = axes[1]
    dom = S[:, 5:8].argmax(1); cols = ["#7FB3D5", "#2E86C1", "#1B4F72"]; x100 = np.arange(30) * 100 + 50
    for p in range(3):
        i = (mk == "6") & (dom == p); y = clk[i].sum(0); y = y / y.mean()
        ax.plot(x100, y.reshape(30, 10).mean(1), color=cols[p], lw=1.8, label=f"快照在第 {p} 秒（{int(i.sum()):,} 个股票日）")
        N[f"图三_沪市相位{p}_每秒均值"] = y.reshape(3, 100).mean(1).tolist(); N[f"图三_沪市相位{p}_股票日数"] = int(i.sum())
    ax.axhline(1, color=INK, lw=1, ls="--"); ax.set_xlim(0, 3000); ax.set_xticks([0, 1000, 2000, 3000]); ax.set_xlabel("3 秒周期里的钟点位置（毫秒）"); ax.set_ylabel("相对 3 秒平均的倍数"); ax.set_ylim(.65, 1.62); ax.legend(frameon=False, fontsize=9, loc="upper center", ncol=1); ax.set_title("沪市按快照落在 3 秒周期的第几秒分三组", loc="left", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, .95)); fig.savefig(OUT / "f3_snapshot.png", dpi=170); plt.close(fig)
    # 跨日稳定性
    for m in "03":
        rows = []
        for d in DATES:
            mkd = np.array([c[0] for c in Ms[d]["codes"]]); y = As[d]["snap"][mkd == m].sum(0) / As[d]["occ"][mkd == m].sum(0); y = y / y.mean()
            rows.append([d, float(y[14]), float(y[15]), float(y[99]), float(y[100]), float(y[:100].mean()), float(y[100:200].mean())])
        N[f"图三_{NAME[m]}_逐日(日,140,150,990,1000,第1秒,第2秒)"] = rows
    S0 = S[mk == "6"]; N["图三_沪市相位分布(0,1,2)"] = np.bincount(S0[:, 5:8].argmax(1), minlength=3).tolist()
    N["图三_深市主板快照相位占比"] = float((S[mk == "0"][:, 5:8].argmax(1) == 0).mean()); N["图三_创业板快照相位占比"] = float((S[mk == "3"][:, 5:8].argmax(1) == 0).mean())

# ---- 图四：间隔
def fig4():
    fig, axes = plt.subplots(2, 1, figsize=(11, 7.2), sharex=False)
    fig.suptitle("两张主动单的间隔：落在同一个 10 毫秒的是随机的 10 倍左右，隔 3.00 秒的是随机的 2 倍", x=0.01, ha="left", fontsize=13)
    x = np.arange(500) * 10
    ax = axes[0]
    for m in "603":
        i = mk == m; a, b = gap[i].sum(0), shuf[i].sum(0); ratio = a / b; ax.plot(x, ratio, color=C[m], lw=1.1, label=NAME[m])
        N[f"图四_{NAME[m]}_间隔0比随机"] = ratio[0]; N[f"图四_{NAME[m]}_间隔10比随机"] = ratio[1]; N[f"图四_{NAME[m]}_间隔20比随机"] = ratio[2]; N[f"图四_{NAME[m]}_50ms内占比真实"] = a[:5].sum() / a.sum(); N[f"图四_{NAME[m]}_50ms内占比随机"] = b[:5].sum() / b.sum()
        for ms in [1000, 2000, 3000, 4000]: N[f"图四_{NAME[m]}_{ms}ms比随机"] = ratio[ms // 10]; N[f"图四_{NAME[m]}_{ms}ms邻居均值"] = float((ratio[ms // 10 - 3] + ratio[ms // 10 + 3]) / 2)
        N[f"图四_{NAME[m]}_500ms比随机"] = ratio[50]
    ax.set_yscale("log"); ax.axhline(1, color=INK, lw=1, ls="--"); ax.set_xlim(0, 4990); ax.set_xticks(range(0, 5001, 500)); ax.set_ylabel("真实 ÷ 分钟内随机（对数，一格 10 倍）"); ax.set_xlabel("相邻两张主动单的间隔（毫秒）")
    for ms in [1000, 2000, 3000, 4000]: ax.axvline(ms, color="#DDDDDD", lw=1, zorder=0)
    ax.legend(frameon=False, ncol=3, fontsize=9)
    ax = axes[1]
    for m in "603":
        i = mk == m; ratio = gap[i].sum(0) / shuf[i].sum(0); ax.plot(x[:51], ratio[:51], marker="o", ms=3.2, lw=1.1, color=C[m], label=NAME[m])
    ax.set_yscale("log"); ax.axhline(1, color=INK, lw=1, ls="--"); ax.set_xlim(-5, 505); ax.set_xticks(range(0, 501, 50)); ax.set_xlabel("放大看前 500 毫秒"); ax.set_ylabel("真实 ÷ 分钟内随机（对数）")
    fig.tight_layout(rect=(0, 0, 1, .96)); fig.savefig(OUT / "f4_gaps.png", dpi=170); plt.close(fig)
    rows = {}
    for d in DATES:
        mkd = np.array([c[0] for c in Ms[d]["codes"]]); o = {}
        for m in "603":
            a = As[d]["gap"][mkd == m].sum(0).astype(float); b = As[d]["shuf"][mkd == m].sum(0).astype(float); o[NAME[m]] = dict(间隔0=float(a[0] / b[0]), 间隔10=float(a[1] / b[1]), 间隔3000=float(a[300] / b[300]))
        rows[str(d)] = o
    N["图四_逐日比随机"] = rows

# ---- 图五：连发
def fig5():
    edges = Ms[DATES[0]]["gap_edges"]; L = []
    for a, b in zip(edges[:-1], edges[1:]):
        L.append(f"{a*10}" if b == a + 1 and a < 100 else (f"{a*10}–{b*10-10}" if b * 10 <= 1000 else f"{a/100:g}–{b/100:g}秒"))
    P = pair.sum(0); y = sdv.sum(0) / P; bd = bday.sum(0) / P; bm = bmin.sum(0) / P; ds = sd.sum(0) / P
    x = np.arange(13)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.9))
    fig.suptitle(f"间隔越短，下一单越像上一单：同一10毫秒的两张单 {y[0]:.0%} 方向和股数都相同，同一分钟打乱只有 {bm[0]:.0%}", x=0.01, ha="left", fontsize=13)
    ax = axes[0]
    ax.plot(x, y, color="#C0392B", lw=2.2, marker="o", ms=4, label="真实")
    ax.plot(x, bm, color=GREY, lw=1.8, marker="o", ms=3, label="同一分钟内打乱")
    ax.plot(x, bd, color="#BDC3C7", lw=1.5, ls="--", label="同一只股票全天打乱")
    ax.set_xticks(x); ax.set_xticklabels(L, rotation=35, ha="right", fontsize=9); ax.set_ylabel("方向和成交股数都相同的比例"); ax.set_xlabel("相邻两张主动单的间隔（毫秒）"); ax.set_ylim(0, .36); ax.legend(frameon=False, fontsize=9)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_title("方向和股数都相同", loc="left", fontsize=11)
    ax = axes[1]
    ax.plot(x, ds, color="#1F618D", lw=2.2, marker="o", ms=4, label="真实"); ax.axhline(0.5, color=INK, lw=1, ls="--", label="各占一半")
    ax.set_xticks(x); ax.set_xticklabels(L, rotation=35, ha="right", fontsize=9); ax.set_ylabel("方向相同的比例"); ax.set_xlabel("相邻两张主动单的间隔（毫秒）"); ax.set_ylim(.45, 1); ax.legend(frameon=False, fontsize=9)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_title("方向相同", loc="left", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, .94)); fig.savefig(OUT / "f5_repeat.png", dpi=170); plt.close(fig)
    N["图五_分组"] = L; N["图五_同10毫秒配对占相邻配对"] = float(P[0] / P.sum())
    for k, v in [("真实同向同量", y), ("同分钟打乱", bm), ("全天打乱", bd), ("真实同向", ds), ("配对数", P)]: N[f"图五_{k}"] = v.tolist()
    for m in "603":
        i = mk == m; N[f"图五_{NAME[m]}_同10毫秒"] = dict(真实同向同量=float(sdv[i, 0].sum() / pair[i, 0].sum()), 同分钟打乱=float(bmin[i, 0].sum() / pair[i, 0].sum()), 真实同向=float(sd[i, 0].sum() / pair[i, 0].sum()))

# ---- 图六：齐发
def fig6():
    real = sum(np.array(Ms[d]["cross"]["real_ex"], float) for d in DATES); fake = sum(np.array(Ms[d]["cross"]["fake_ex"], float) for d in DATES)
    nz = np.flatnonzero(real)
    def tail(a): return np.cumsum(a[::-1])[::-1]
    Tr, Tf = tail(real), tail(fake)
    n100r, n100f = Tr[100], Tf[100]
    N["图六_八天_100只以上_真实"] = float(n100r); N["图六_八天_100只以上_错开"] = float(n100f); N["图六_八天_50只以上_真实"] = float(Tr[50]); N["图六_八天_50只以上_错开"] = float(Tf[50])
    Ex = json.load(open(SB / "why4hz/fig6_example.json"))
    cnt = np.array(Ex["count"]); W = len(cnt) // 2; xs = (np.arange(len(cnt)) - W) * 10; others = np.delete(cnt, W)
    N["图六_例子"] = dict(日期=Ex["date"], 时间=Ex["time"], 方向="主动卖", 当时股票数=int(cnt[W]), 前后150毫秒中位=float(np.median(others)), 前后最高=int(others.max()), 这一天普通10毫秒中位=Ex["median_all"], 当天股票总数=Ex["n_codes"])
    fig = plt.figure(figsize=(14.5, 5.2)); gs = fig.add_gridspec(1, 4, width_ratios=[2.2, 1, 1, 1], wspace=.34)
    fig.suptitle("同一个 10 毫秒里，几百只股票一起被主动买或主动卖；把时间错开后，这种情况几乎消失", x=0.01, ha="left", fontsize=14)
    ax = fig.add_subplot(gs[0]); col = ["#C8D3DC"] * len(cnt); col[W] = "#C0392B"
    ax.bar(xs, cnt, width=8.4, color=col, linewidth=0)
    ax.text(0, cnt[W] + 8, f"{cnt[W]} 只股票", ha="center", va="bottom", color="#C0392B", fontsize=11)
    ax.axhline(Ex["median_all"], color=INK, lw=1, ls="--"); ax.text(16, 92, f"虚线：当天中位水平 {Ex['median_all']} 只", ha="left", va="bottom", fontsize=9.5, color=INK)
    ax.set_xlim(-158, 158); ax.set_ylim(0, 500); ax.set_xticks(range(-150, 151, 50)); ax.set_xlabel("相对这一刻的毫秒数（每根柱子是 10 毫秒）"); ax.set_ylabel("这 10 毫秒里有主动卖单的股票数")
    ax.set_title("一个例子：2024-03-29 上午 10:55:26.66 前后", loc="left", fontsize=11.5)
    for j, th in enumerate([100, 200, 300]):
        ax = fig.add_subplot(gs[j + 1]); r, f = float(Tr[th]), float(Tf[th]); N[f"图六_八天_{th}只以上_真实"] = r; N[f"图六_八天_{th}只以上_错开"] = f
        ax.bar([0, 1], [r, f], width=.62, color=["#C0392B", GREY]); ax.set_xticks([0, 1]); ax.set_xticklabels(["真实", "错开后"]); ax.set_ylim(0, r * 1.18); ax.grid(axis="x", visible=False)
        ax.text(0, r, f"{r:,.0f}", ha="center", va="bottom", fontsize=10.5, color="#C0392B"); ax.text(1, f, f"{f:,.0f}", ha="center", va="bottom", fontsize=10.5, color="#566573")
        ax.set_title(f"至少 {th} 只股票", loc="left", fontsize=11.5)
        if j == 0: ax.set_ylabel("这样的 10 毫秒有多少个（八天合计）")
    fig.text(.01, -.03, "错开后：每只股票的时间线整体挪 1 到 5 分钟，各自的节奏不变，只破坏“同时”。两段开盘后 5 分钟的成交已剔除。", fontsize=9.5, color="#566573")
    fig.savefig(OUT / "f6_cross.png", dpi=170, bbox_inches="tight"); plt.close(fig)
    rows = {}
    for d in DATES:
        c = Ms[d]["cross"]; r = np.array(c["real_ex"]); f = np.array(c["fake_ex"])
        top = np.array(c["top"][:100])
        rows[str(d)] = dict(最多股票数=int(np.flatnonzero(r)[-1]), 错开最多均值=float(np.mean(c["fake_max_ex"])), 错开最多峰值=int(max(c["fake_max_ex"])), 真实100=int(r[100:].sum()), 错开100=float(f[100:].sum()), 真实50=int(r[50:].sum()), 错开50=float(f[50:].sum()),
                            前100主买占比=float((top[:, 1] == 0).mean()), 前100整分钟占比=float(((top[:, 0] // 100) % 60 == 0).mean()), 前100沪市股票占比=float(top[:, 2].sum() / (top[:, 2] + top[:, 3]).sum()))
    N["图六_逐日"] = rows
    allt = []
    for d in DATES:
        t = np.array(Ms[d]["cross"]["top"][:1000]); allt.append(t)
    t = np.vstack(allt); ss = (t[:, 0] // 100) % 60; cc = t[:, 0] % 100
    N["图六_八天前1000_落在第0到9秒的占比"] = float((ss < 10).mean()); N["图六_八天前1000_0毫秒到99毫秒占比"] = float((cc < 10).mean()); N["图六_八天前1000_主买占比"] = float((t[:, 1] == 0).mean())
    # 开盘：全日 vs 剔除开盘最大值 / 13:00:00
    op = {}
    for d in DATES:
        rr = np.array(Ms[d]["cross"]["real"]); op[str(d)] = int(np.flatnonzero(rr)[-1])
    N["图六_含开盘全日最多股票数(13:00:00附近)"] = op
    single = 0; tot = 0
    for d in DATES:
        tp = np.array(Ms[d]["cross"]["top"][:100]); share = tp[:, 2] / (tp[:, 2] + tp[:, 3]); single += int(((share > .8) | (share < .2)).sum()); tot += 100
    N["图六_前10010毫秒八成以上来自一个交易所的占比"] = single / tot

def xlsx():
    import pandas as pd
    p = rec.sum(0) / rec.sum() * 100; a = amt.sum(0) / amt.sum() * 100
    cells = pd.DataFrame({"格子起点毫秒": CELLX, "成交笔数占比%": p, "成交金额占比%": a})
    for m in "603":
        x = ev[mk == m].sum(0); cells[NAME[m] + "主动单相对强度"] = x / x.mean()
    flat = []
    def walk(k, v):
        if isinstance(v, dict):
            for a1, b1 in v.items(): walk(f"{k} / {a1}", b1)
        else:
            flat.append((k, json.dumps(v, ensure_ascii=False, default=float) if isinstance(v, list) else v))
    for k, v in N.items(): walk(k, v)
    y = pd.DataFrame({"分组(毫秒)": N["图五_分组"], "真实同向同量": N["图五_真实同向同量"], "同分钟打乱": N["图五_同分钟打乱"], "全天打乱": N["图五_全天打乱"], "真实同向": N["图五_真实同向"], "配对数": N["图五_配对数"]})
    days = pd.DataFrame(N["图六_逐日"]).T.reset_index(names="日期")
    with pd.ExcelWriter(SB / "数字汇总.xlsx") as w:
        pd.DataFrame(flat, columns=["项目", "数值"]).to_excel(w, sheet_name="全部数字", index=False)
        cells.to_excel(w, sheet_name="100个格子", index=False)
        y.to_excel(w, sheet_name="下一单像不像", index=False)
        days.to_excel(w, sheet_name="全市场齐发逐日", index=False)

fig1(); fig2(); fig3(); fig4(); fig5(); fig6()
xlsx()
json.dump(N, open(SB / "numbers.json", "w"), ensure_ascii=False, indent=1, default=float)
print(json.dumps({k: v for k, v in N.items() if not isinstance(v, (dict, list))}, ensure_ascii=False, indent=0, default=float))
