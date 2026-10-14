"""hm94 全 35 个因子的两套 TLQ（水位口径 + 相邻采样点分段口径）与逐段质量曲线。

口径见 sandbox/src/tlq.rs：两套都用同一批 160 个等距采样点，
水位口径读累计曲线本身，分段口径读相邻采样点之间的增量（159 段，段间不重叠）。
"""
import time

import numpy as np
import pandas as pd

import dev_sandbox
import tlq_hm94_check as D

OUT_DIR = "/home/chenzongwei/pythoncode/成果hm94_挂单猫"
NER = "/home/chenzongwei/.local/share/fonts/noto-cjk-sc/NotoSerifCJKsc-Regular.otf"


def load_all():
    panels = D.build_panels()
    log_exret, week_first = D.build_derived(panels)
    br = dev_sandbox.sandbox_interleave_styles(
        [dev_sandbox.sandbox_rank_pct_rows(panels[f"barra_{n}"]) for n in D.BARRA_NAMES],
        panels["restrict"].shape[1],
    )
    sr = dev_sandbox.sandbox_rank_pct_rows(panels["size"])
    calendar = pd.read_csv(f"{D.ROOT}/basic_info/calendar.csv", header=None).iloc[:, 0]
    all_dates = pd.to_datetime(calendar.astype(str), format="%Y%m%d")
    end_pos = int(np.searchsorted(all_dates.values, pd.Timestamp(D.END).to_datetime64(), side="right"))
    end_pos = min(len(all_dates), end_pos + 20)
    rows = np.nonzero((all_dates >= pd.Timestamp(D.START)) & (np.arange(len(all_dates)) < end_pos))[0]
    dates = panels["_dates"]
    group_to = int(np.searchsorted(dates.values, pd.Timestamp(D.END).to_datetime64(), side="right"))
    return panels, log_exret, week_first, br, sr, rows, dates, group_to


def main():
    panels, log_exret, week_first, br, sr, rows, dates, group_to = load_all()
    n_stocks = panels["restrict"].shape[1]
    report = pd.read_csv(D.REPORT).set_index("factor_name")
    names = list(report["base_TLQ"].sort_values(ascending=False).index)

    lvl, seg, seg_stats, rets_all = {}, {}, {}, {}
    t0 = time.time()
    for name in names:
        factor = D._read_h5(f"{D.FACTOR_DIR}/{name}.h5", rows, n_stocks)
        a, b, rets, curves, stats = dev_sandbox.sandbox_tlq_base_resid(
            factor, log_exret, panels["restrict"], panels["industry"], br, len(D.BARRA_NAMES),
            sr, week_first, 0, group_to, 160)
        lvl[name], seg[name] = float(a), float(b)
        seg_stats[name] = np.asarray(stats, dtype=np.float64)
        rets_all[name] = np.asarray(rets, dtype=np.float64)
    print(f"35 个因子两套 TLQ 算完 {time.time()-t0:.1f}s（单因子 {(time.time()-t0)/35:.2f}s）", flush=True)

    cmp = pd.DataFrame({
        "水位TLQ": pd.Series(lvl),
        "分段TLQ": pd.Series(seg),
        "报告base_TLQ": report["base_TLQ"].reindex(names),
    })
    cmp["差(分段-水位)"] = cmp["分段TLQ"] - cmp["水位TLQ"]
    print(cmp.describe().loc[["mean", "50%", "max"]].round(3).to_string())
    print("秩相关 分段 vs 水位 %.3f；分段 vs 报告 %.3f；水位 vs 报告 %.3f" % (
        cmp["分段TLQ"].corr(cmp["水位TLQ"], method="spearman"),
        cmp["分段TLQ"].corr(cmp["报告base_TLQ"], method="spearman"),
        cmp["水位TLQ"].corr(cmp["报告base_TLQ"], method="spearman")))

    # 逐段质量矩阵（列 3 = 选中方向的质量）
    segq = pd.DataFrame({n: seg_stats[n][:, 3] for n in names},
                        index=[f"段{i+1:03d}" for i in range(seg_stats[names[0]].shape[0])]).T
    # 每段中点日期
    pos = np.rint(np.linspace(0, group_to - 1, 160)).astype(int)
    mid_dates = dates[[(pos[k] + pos[k + 1]) // 2 for k in range(len(pos) - 1)]]

    np.savez_compressed("/tmp/tlq_panels/two_tlq.npz",
                        names=np.array(names), lvl=np.array([lvl[n] for n in names]),
                        seg=np.array([seg[n] for n in names]), segq=segq.to_numpy())

    # ---------------- 画图 ----------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font_manager.fontManager.addfont(NER)
    plt.rcParams["font.sans-serif"] = [font_manager.FontProperties(fname=NER).get_name(), "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    picks = [names[0], names[1], names[5], names[12], names[-1]]
    fig = plt.figure(figsize=(15, 13))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1.5], hspace=0.28)

    ax = fig.add_subplot(gs[0])
    for n in picks:
        y = segq.loc[n].to_numpy()
        ax.plot(mid_dates, pd.Series(y).rolling(5, min_periods=1).mean(), lw=1.6,
                label=f"{n[:46]}  分段 {seg[n]:.3f} / 水位 {lvl[n]:.3f}")
    ax.axhline(0, color="#999", lw=1, ls="--")
    ax.set_title("逐段分层质量（横轴为每一段的日期中点，纵轴为该段内十组的端点分离×方向保留比例，5 段移动平均）",
                 fontsize=13)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(alpha=0.3)

    ax2 = fig.add_subplot(gs[1])
    order = cmp.sort_values("分段TLQ", ascending=False).index
    mat = segq.loc[order].to_numpy()
    im = ax2.imshow(mat, aspect="auto", cmap="RdYlGn", vmin=-0.6, vmax=0.6,
                    extent=[0, len(mid_dates), len(order), 0])
    ax2.set_yticks(np.arange(len(order)) + 0.5)
    ax2.set_yticklabels([f"{n[:42]}  ({seg[n]:.2f})" for n in order], fontsize=7)
    step = max(1, len(mid_dates) // 12)
    ticks = list(range(0, len(mid_dates), step))
    ax2.set_xticks(ticks)
    ax2.set_xticklabels([pd.Timestamp(mid_dates[i]).strftime("%Y-%m") for i in ticks], fontsize=9)
    ax2.set_title("全部 35 个因子的逐段质量热力图（按分段 TLQ 降序，绿=该段分层正向且少回穿，红=反向）", fontsize=13)
    fig.colorbar(im, ax=ax2, shrink=0.7, label="该段质量")
    png = f"{OUT_DIR}/hm94_逐段分层质量_20261013.png"
    fig.savefig(png, dpi=110, facecolor="white", bbox_inches="tight")
    print("图:", png, flush=True)

    xlsx = f"{OUT_DIR}/hm94_两套TLQ与逐段质量_20261013.xlsx"
    with pd.ExcelWriter(xlsx, engine="openpyxl") as w:
        cmp.round(4).to_excel(w, sheet_name="两套TLQ对比")
        segq.round(4).to_excel(w, sheet_name="逐段质量(35×159)")
        pd.DataFrame({"段": range(1, len(mid_dates) + 1),
                      "段中点日期": [pd.Timestamp(d).strftime("%Y-%m-%d") for d in mid_dates],
                      **{f"D_{n[:24]}": seg_stats[n][:, 0] for n in picks}}).round(4).to_excel(
            w, index=False, sheet_name="样例因子逐段D")
    print("表:", xlsx, flush=True)
    return cmp, segq, mid_dates


if __name__ == "__main__":
    main()
