"""对任意 hm 系列算两套 TLQ（水位 + 分段），出对比表；顺带画分段 TLQ 落在 0.2~0.3 的因子的分组曲线。

用法：python tlq_series_check.py hm97
数据：该系列的 hf_data 因子 + 上线因子报告 summary_{series}.csv
"""
import sys

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

import dev_sandbox
import tlq_hm94_check as D

NER = "/home/chenzongwei/.local/share/fonts/noto-cjk-sc/NotoSerifCJKsc-Regular.otf"
BAND = (0.2, 0.3)


def setup(series: str):
    D.FACTOR_DIR = f"{D.ROOT}/hf_data/{series}"
    D.REPORT = f"/nas197/user_home_share/上线因子/factor_report_20261010/{series}/summary_{series}.csv"
    return series


def run(series: str, out_dir: str, make_fig: bool = True):
    setup(series)
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
    report = pd.read_csv(D.REPORT).set_index("factor_name")
    names = list(report.index)

    lvl, seg, curves = {}, {}, {}
    for name in names:
        factor = D._read_h5(f"{D.FACTOR_DIR}/{name}.h5", rows, n_stocks)
        a, b, _rets, cur, _stats = dev_sandbox.sandbox_tlq_base_resid(
            factor, log_exret, panels["restrict"], panels["industry"], br, len(D.BARRA_NAMES),
            sr, week_first, 0, group_to, 160,
        )
        lvl[name], seg[name], curves[name] = float(a), float(b), np.asarray(cur)
        print(f"{name[:56]:56s} 水位 {a:.4f} 分段 {b:.4f} 报告 {report.loc[name,'base_TLQ']:.4f}", flush=True)

    df = pd.DataFrame({"factor_name": names, "水位TLQ": [lvl[n] for n in names],
                       "分段TLQ": [seg[n] for n in names]})
    df["报告base_TLQ"] = report["base_TLQ"].reindex(df.factor_name).to_numpy()
    df["差(分段-水位)"] = df["分段TLQ"] - df["水位TLQ"]
    df["水位名次"] = df["水位TLQ"].rank(ascending=False).astype(int)
    df["分段名次"] = df["分段TLQ"].rank(ascending=False).astype(int)
    df["名次变化"] = df["水位名次"] - df["分段名次"]
    out = df.sort_values("分段TLQ", ascending=False).reset_index(drop=True)

    xlsx = f"{out_dir}/{series}_两套TLQ全表对比.xlsx"
    with pd.ExcelWriter(xlsx, engine="openpyxl") as w:
        out.round(4).to_excel(w, index=False, sheet_name=f"{series}两套TLQ")
        pd.DataFrame({"项目": ["水位口径", "分段口径", "报告 base_TLQ", "水位 vs 报告", "分段 vs 报告",
                              "水位 vs 分段", "水位>0.6 个数", "分段>0.2 个数", "两者都过各自门槛"],
                      "值": ["公式作用在 160 个采样点的累计水位上（PDF 与报告口径）",
                            "公式作用在相邻采样点增量上（159 段）",
                            "上线因子报告记录值（水位口径）",
                            round(float(out.水位TLQ.corr(out.报告base_TLQ)), 4),
                            round(float(out.分段TLQ.corr(out.报告base_TLQ)), 4),
                            round(float(out.水位TLQ.corr(out.分段TLQ)), 4),
                            int((out.水位TLQ > 0.6).sum()), int((out.分段TLQ > 0.2).sum()),
                            int(((out.水位TLQ > 0.6) & (out.分段TLQ > 0.2)).sum())]}).to_excel(
            w, index=False, sheet_name="口径说明")
    print("\n" + out.round(4).to_string(index=False), flush=True)
    print("\n表:", xlsx, flush=True)

    band = out[(out.分段TLQ >= BAND[0]) & (out.分段TLQ <= BAND[1])].sort_values("分段TLQ", ascending=False)
    print(f"分段 TLQ 落在 {BAND[0]}~{BAND[1]} 的 {len(band)} 个；通过 0.2 的 {int((out.分段TLQ>0.2).sum())} / {len(out)}", flush=True)
    if make_fig and len(band):
        font_manager.fontManager.addfont(NER)
        plt.rcParams["font.sans-serif"] = [font_manager.FontProperties(fname=NER).get_name(), "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
        pos = np.rint(np.linspace(0, group_to - 1, 160)).astype(int)
        x = dates[pos]
        ncol = 3
        nrow = int(np.ceil(len(band) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(6.2 * ncol, 3.6 * nrow), squeeze=False)
        cmap = plt.get_cmap("coolwarm")
        for ax, name in zip(axes.ravel(), band.factor_name):
            cur = curves[name]
            for g in range(10):
                ax.plot(x, cur[:, g], color=cmap(g / 9), lw=1.5)
            ax.axhline(0, color="#999", lw=0.8, ls="--")
            ax.set_title(f"{name}\n分段 TLQ {seg[name]:.3f}（水位口径 {lvl[name]:.3f}）", fontsize=9)
            ax.grid(alpha=0.25)
            ax.tick_params(labelsize=8)
        for ax in axes.ravel()[len(band):]:
            ax.axis("off")
        fig.suptitle(f"{series} 候选里分段 TLQ 落在 {BAND[0]}~{BAND[1]} 的 {len(band)} 个因子：base_resid 十分组累计收益曲线（G1 蓝 → G10 红）",
                     fontsize=14, y=1.002)
        fig.tight_layout()
        png = f"{out_dir}/{series}_分段TLQ_0.2到0.3_分组回测图.png"
        fig.savefig(png, dpi=110, facecolor="white", bbox_inches="tight")
        print("图:", png, flush=True)
    np.savez_compressed(f"/tmp/tlq_panels/{series}_two_tlq.npz",
                        names=np.array(names), lvl=np.array([lvl[n] for n in names]),
                        seg=np.array([seg[n] for n in names]))
    return out


if __name__ == "__main__":
    series = sys.argv[1] if len(sys.argv) > 1 else "hm97"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "/home/chenzongwei/pythoncode/成果hm97_柜员游戏"
    run(series, out_dir)
