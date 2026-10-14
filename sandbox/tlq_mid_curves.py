"""把分段 TLQ 落在 0.2~0.3 的因子的 base_resid 十分组累计收益曲线画出来，供人工看一眼。"""
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

import dev_sandbox
import tlq_hm94_check as D

NER = "/home/chenzongwei/.local/share/fonts/noto-cjk-sc/NotoSerifCJKsc-Regular.otf"
OUT_DIR = "/home/chenzongwei/pythoncode/成果hm94_挂单猫"
LO, HI = 0.2, 0.3


def main():
    z = np.load("/tmp/tlq_panels/two_tlq.npz", allow_pickle=True)
    tab = pd.DataFrame({"factor": list(z["names"]), "水位": z["lvl"], "分段": z["seg"]})
    picks = tab[(tab.分段 >= LO) & (tab.分段 <= HI)].sort_values("分段", ascending=False)

    panels = D.build_panels()
    log_exret, week_first = D.build_derived(panels)
    n_stocks = panels["restrict"].shape[1]
    br = dev_sandbox.sandbox_interleave_styles(
        [dev_sandbox.sandbox_rank_pct_rows(panels[f"barra_{n}"]) for n in D.BARRA_NAMES], n_stocks
    )
    sr = dev_sandbox.sandbox_rank_pct_rows(panels["size"])
    calendar = pd.read_csv(f"{D.ROOT}/basic_info/calendar.csv", header=None).iloc[:, 0]
    all_dates = pd.to_datetime(calendar.astype(str), format="%Y%m%d")
    end_pos = int(np.searchsorted(all_dates.values, pd.Timestamp(D.END).to_datetime64(), side="right"))
    end_pos = min(len(all_dates), end_pos + 20)
    rows = np.nonzero((all_dates >= pd.Timestamp(D.START)) & (np.arange(len(all_dates)) < end_pos))[0]
    dates = panels["_dates"]
    group_to = int(np.searchsorted(dates.values, pd.Timestamp(D.END).to_datetime64(), side="right"))

    curves = {}
    for name in picks.factor:
        factor = D._read_h5(f"{D.FACTOR_DIR}/{name}.h5", rows, n_stocks)
        lvl, seg, rets, cur, _stats = dev_sandbox.sandbox_tlq_base_resid(
            factor, log_exret, panels["restrict"], panels["industry"], br, len(D.BARRA_NAMES),
            sr, week_first, 0, group_to, 160,
        )
        curves[name] = np.asarray(cur)
        print(f"{name[:52]:52s} 分段 {seg:.4f} 水位 {lvl:.4f}", flush=True)

    font_manager.fontManager.addfont(NER)
    plt.rcParams["font.sans-serif"] = [font_manager.FontProperties(fname=NER).get_name(), "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    # 采样点对应的日期（160 个等距点）
    pos = np.rint(np.linspace(0, group_to - 1, 160)).astype(int)
    x = dates[pos]

    n = len(curves)
    ncol = 3
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(6.2 * ncol, 3.6 * nrow), squeeze=False)
    cmap = plt.get_cmap("coolwarm")
    for ax, (name, cur) in zip(axes.ravel(), curves.items()):
        for g in range(10):
            ax.plot(x, cur[:, g], color=cmap(g / 9), lw=1.5, label=f"G{g+1}" if g in (0, 4, 9) else None)
        ax.axhline(0, color="#999", lw=0.8, ls="--")
        seg = float(picks.loc[picks.factor == name, "分段"].iloc[0])
        lvl = float(picks.loc[picks.factor == name, "水位"].iloc[0])
        ax.set_title(f"{name}\n分段 TLQ {seg:.3f}（水位口径 {lvl:.3f}）", fontsize=9)
        ax.grid(alpha=0.25)
        ax.tick_params(labelsize=8)
        ax.legend(fontsize=7, loc="upper left")
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle(
        f"hm94 候选里分段 TLQ 落在 {LO}~{HI} 的 {n} 个因子：base_resid 十分组累计收益曲线（G1 蓝 → G10 红）",
        fontsize=14, y=1.002)
    fig.tight_layout()
    png = f"{OUT_DIR}/hm94_分段TLQ_0.2到0.3_分组回测图.png"
    fig.savefig(png, dpi=110, facecolor="white", bbox_inches="tight")
    print("图:", png, flush=True)

    xlsx = f"{OUT_DIR}/hm94_分段TLQ_0.2到0.3_明细.xlsx"
    with pd.ExcelWriter(xlsx, engine="openpyxl") as w:
        picks.round(4).to_excel(w, index=False, sheet_name="13个因子TLQ")
        pd.DataFrame({"日期": [pd.Timestamp(d).strftime("%Y-%m-%d") for d in x],
                      **{f"{n[:20]}_G{g+1}": curves[n][:, g] for n in list(curves)[:6] for g in range(10)}
                      }).round(6).to_excel(w, index=False, sheet_name="曲线采样点(前6个)")
    print("表:", xlsx, flush=True)


if __name__ == "__main__":
    main()
