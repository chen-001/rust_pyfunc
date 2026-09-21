"""从复算结果统计三个计数。

多头组 = 引擎判定：组 1 与组 10 的逐日收益之和，大的一端为多头。
空头组 = 另一端。
极值比较用组收益曲线的累计和（引擎自己就是这么定多空的），
即 gs[i] = sum_t group_returns[i][t]，逐日收益填 0 的空组不影响排序。
"""
import json

import numpy as np
import pandas as pd

HERE = "/home/chenzongwei/rust_pyfunc/tmp_brc_group"
INPUT = "/home/chenzongwei/rust_pyfunc/tmp_brc_lead/group_extreme_input.json"
TOL = 1e-12

df = pd.read_csv(f"{HERE}/repro_metrics.csv")
gs_all = np.load(f"{HERE}/group_sums.npy")
gs = {n: gs_all[i] for i, n in enumerate(df["factor_name"])}
inp = json.load(open(INPUT))


def flags(g):
    long_i = 0 if g[0] > g[9] else 9
    short_i = 9 if long_i == 0 else 0
    rest_long = np.delete(g, long_i)
    rest_short = np.delete(g, short_i)
    long_top_strict = bool(g[long_i] > rest_long.max())
    long_top_tie = bool(g[long_i] >= rest_long.max() - TOL)
    short_bot_strict = bool(g[short_i] < rest_short.min())
    short_bot_tie = bool(g[short_i] <= rest_short.min() + TOL)
    return dict(long_i=long_i, short_i=short_i,
                long_top_strict=long_top_strict, long_top_tie=long_top_tie,
                short_bot_strict=short_bot_strict, short_bot_tie=short_bot_tie,
                both_strict=long_top_strict and short_bot_strict,
                both_tie=long_top_tie and short_bot_tie)


rows = []
for name, g in gs.items():
    rows.append(dict(factor_name=name, **flags(g),
                     g_long=float(g[flags(g)["long_i"]]),
                     g_best=float(g.max()), g_worst=float(g.min()),
                     spread=float(g.max() - g.min())))
res = pd.DataFrame(rows).set_index("factor_name")
res.to_csv(f"{HERE}/extreme_flags.csv")

print("== 三个计数（分母 35 / 35）==")
summary = {}
for key in ("brc", "orig"):
    sub = res.loc[inp[key]]
    n = len(sub)
    counts = {c: int(sub[c].sum()) for c in
              ("long_top_strict", "long_top_tie", "short_bot_strict", "short_bot_tie",
               "both_strict", "both_tie")}
    summary[key] = counts
    print(f"\n{key} (n={n}):")
    print(f"  多头组是最高组   严格 {counts['long_top_strict']}  并列也算 {counts['long_top_tie']}")
    print(f"  空头组是最低组   严格 {counts['short_bot_strict']}  并列也算 {counts['short_bot_tie']}")
    print(f"  两条同时成立     严格 {counts['both_strict']}  并列也算 {counts['both_tie']}")
    for c, lab in (("long_top_strict", "多头不是最高(严格)"),
                   ("short_bot_strict", "空头不是最低(严格)"),
                   ("both_strict", "两条不同时(严格)")):
        bad = list(sub.index[~sub[c]])[:10]
        print(f"  {lab}: {len(sub) - counts[c]} 个 {bad}")

json.dump(summary, open(f"{HERE}/extreme_counts.json", "w"), ensure_ascii=False, indent=2)
print("\n== 明细（并列/极端情况）==")
print(res[["long_i", "short_i", "long_top_strict", "short_bot_strict",
           "g_long", "g_best", "g_worst", "spread"]].to_string())
