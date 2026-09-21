"""稳健性检验：把每个因子的多头组累计收益扰动 ±delta，看三个计数是否变化。

delta 取复算与指标表之间最大偏差（年化）换算回累计收益后的若干倍。
"""
import json

import numpy as np
import pandas as pd

HERE = "/home/chenzongwei/rust_pyfunc/tmp_brc_group"
inp = json.load(open("/home/chenzongwei/rust_pyfunc/tmp_brc_lead/group_extreme_input.json"))
df = pd.read_csv(f"{HERE}/repro_metrics.csv")
gs_all = np.load(f"{HERE}/group_sums.npy")
selfcheck = pd.read_csv(f"{HERE}/selfcheck.csv").set_index("factor_name")
gs = {n: gs_all[i] for i, n in enumerate(df["factor_name"])}

# 复算与表之间在累计收益上的最大偏差（年化差 × 天数 / 250）
cum_err = (selfcheck["calc_ls"] - selfcheck["tbl_ls"]).abs() * selfcheck["calc_dsize"] / 250.0
print(f"累计收益口径的最大偏差 {cum_err.max():.3e}（因子 {cum_err.idxmax()}）")
DELTA = 5e-3  # 约 7 倍于上述偏差


def counts(delta):
    out = {}
    for key in ("brc", "orig"):
        n_long = n_short = n_both = 0
        for name in inp[key]:
            g = gs[name].copy()
            li = 0 if g[0] > g[9] else 9
            si = 9 if li == 0 else 0
            # 最坏情况：多头组被压低 delta（其它组不动）
            g[li] -= delta
            rl = np.delete(g, li)
            rs = np.delete(g, si)
            lt = bool(g[li] > rl.max())
            sb = bool(g[si] < rs.min())
            n_long += lt
            n_short += sb
            n_both += lt and sb
        out[key] = (n_long, n_short, n_both)
    return out


base = counts(0.0)
print("原始计数（brc / orig，各 35）:", base)
for d in (1e-3, 5e-3, 1e-2, 5e-2):
    c = counts(d)
    same = all(c[k] == base[k] for k in base)
    print(f"扰动 delta={d:.0e}: {c}  计数不变={same}")
