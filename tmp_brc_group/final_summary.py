"""汇总：自检 + 三组计数 + 不满足因子清单，输出 final_summary.txt。"""
import json

import numpy as np
import pandas as pd

HERE = "/home/chenzongwei/rust_pyfunc/tmp_brc_group"
inp = json.load(open("/home/chenzongwei/rust_pyfunc/tmp_brc_lead/group_extreme_input.json"))
df = pd.read_csv(f"{HERE}/repro_metrics.csv")
gs_all = np.load(f"{HERE}/group_sums.npy")
sc = pd.read_csv(f"{HERE}/selfcheck.csv").set_index("factor_name")
gs = {n: gs_all[i] for i, n in enumerate(df["factor_name"])}
TOL = 1e-12

lines = []
for key in ("brc", "orig"):
    sub = sc.loc[inp[key]]
    lines.append(f"{key}: 年化多头超额 对上 {int((sub.rel_hedge<1e-6).sum())}/35  "
                 f"最大相对误差 {sub.rel_hedge.max():.2e}；"
                 f"多空年化 对上 {int((sub.rel_ls<1e-6).sum())}/35  "
                 f"最大相对误差 {sub.rel_ls.max():.2e}")
lines.append(f"date_size 对上 {int((sc.calc_dsize==sc.tbl_dsize).sum())}/66；"
             f"ratio_mean 最大绝对差 {(sc.calc_ratio-sc.tbl_ratio).abs().max():.2e}")
lines.append("")

flags = {}
for name in inp["union"]:
    g = gs[name]
    li = 0 if g[0] > g[9] else 9
    si = 9 if li == 0 else 0
    rl, rs = np.delete(g, li), np.delete(g, si)
    flags[name] = dict(
        li=li, si=si,
        long_strict=bool(g[li] > rl.max()), long_tie=bool(g[li] >= rl.max() - TOL),
        short_strict=bool(g[si] < rs.min()), short_tie=bool(g[si] <= rs.min() + TOL),
        m_long=float(rl.max() - g[li]), m_short=float(g[si] - rs.min()))

for key in ("brc", "orig"):
    ns = inp[key]
    ls = sum(flags[n]["long_strict"] for n in ns)
    ss = sum(flags[n]["short_strict"] for n in ns)
    bs = sum(flags[n]["long_strict"] and flags[n]["short_strict"] for n in ns)
    lt = sum(flags[n]["long_tie"] for n in ns)
    st = sum(flags[n]["short_tie"] for n in ns)
    bt = sum(flags[n]["long_tie"] and flags[n]["short_tie"] for n in ns)
    lines.append(f"== {key} (35) ==")
    lines.append(f"  多头组是最高组: 严格 {ls} / 并列也算 {lt}")
    lines.append(f"  空头组是最低组: 严格 {ss} / 并列也算 {st}")
    lines.append(f"  两条同时成立:   严格 {bs} / 并列也算 {bt}")
    bad_l = [n for n in ns if not flags[n]["long_strict"]]
    bad_s = [n for n in ns if not flags[n]["short_strict"]]
    bad_b = [n for n in ns if not (flags[n]["long_strict"] and flags[n]["short_strict"])]
    lines.append(f"  多头不是最高 {len(bad_l)}: {bad_l[:10]}")
    lines.append(f"  空头不是最低 {len(bad_s)}: {bad_s[:10]}")
    lines.append(f"  两条不同时 {len(bad_b)}: {bad_b[:10]}")
    lines.append("")

lines.append(f"最小多头/空头极值间距: {min(abs(flags[n]['m_long']) for n in inp['union']):.4f} / "
             f"{min(abs(flags[n]['m_short']) for n in inp['union']):.4f}")
txt = "\n".join(lines)
open(f"{HERE}/final_summary.txt", "w").write(txt)
print(txt)

pd.DataFrame([dict(factor_name=n, list=("brc " if n in inp["brc"] else "") +
                   ("orig" if n in inp["orig"] else ""), **v) for n, v in flags.items()]
             ).to_csv(f"{HERE}/extreme_flags.csv", index=False)
