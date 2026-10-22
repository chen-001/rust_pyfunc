# -*- coding: utf-8 -*-
"""独立复算核对：用 pandas 另一条路径重算 30 只股票的五项统计，和 Rust 输出逐项对比。"""
import json
import numpy as np
import pandas as pd

D = 20240329
ROOT = f"/ssd_data/stock/{D}/transaction"
J = json.load(open(f"out/{D}.json"))
RUST = {s["code"]: s for s in J["stocks"]}
CODES = sorted(RUST)[::170]


def q(v, p):
    v = np.sort(np.asarray(v, float))
    return float(v[int(np.floor((len(v) - 1) * p + 0.5))])


def one(code):
    df = pd.read_csv(f"{ROOT}/{code}_{D}_transaction.csv",
                     usecols=["time", "exchtime", "volume", "turnover", "flag", "ask_order", "bid_order"])
    t = df["time"].to_numpy()
    inday = ((t >= 93000000) & (t < 113000000)) | ((t >= 130000000) & (t < 150001000))
    d = df[inday]
    c = d[d["flag"] == 32]
    tr = d[(d["flag"] != 32) & (d["turnover"] > 0)]
    cb, cs = c[c["bid_order"] > 0], c[c["ask_order"] > 0]
    tb, ts = tr[tr["flag"] == 66], tr[tr["flag"] == 83]
    grp = c.groupby("exchtime").size()
    cnt = grp.to_numpy()
    return dict(
        cvol_b=cb["volume"].sum(), cvol_s=cs["volume"].sum(), cn_b=len(cb), cn_s=len(cs),
        tvol_b=tb["volume"].sum(), tvol_s=ts["volume"].sum(), tn_b=len(tb), tn_s=len(ts),
        tamt=tr["turnover"].sum(),
        cp50=q(c["volume"], .5), cp90=q(c["volume"], .9), tp50=q(tr["volume"], .5), tp90=q(tr["volume"], .9),
        n_batch=len(grp), b1=int((cnt == 1).sum()), b2=int((cnt == 2).sum()),
        b3=int(((cnt >= 3) & (cnt <= 10)).sum()), b4=int((cnt >= 11).sum()),
        cbatch=int(cnt[cnt >= 2].sum()),
    )


KEY = ["cvol_b", "cvol_s", "cn_b", "cn_s", "tvol_b", "tvol_s", "tn_b", "tn_s", "tamt",
       "cp50", "cp90", "tp50", "tp90", "n_batch"]

rows, bad = [], 0
agg = dict(b1=0, b2=0, b3=0, b4=0, cbatch=0)
for code in CODES:
    m = one(code)
    r = RUST[code]
    diff = max(abs(m[k] - r[k]) / max(abs(r[k]), 1e-9) for k in KEY)
    bad += diff > 1e-9
    rows.append([code, m["cn_b"] + m["cn_s"], m["tvol_b"] + m["tvol_s"], diff])
    for k in agg:
        agg[k] += m[k]

print(f"抽查 {len(CODES)} 只股票，逐项相对误差 >1e-9 的：{bad} 只")
print(pd.DataFrame(rows, columns=["code", "撤单笔", "成交股数", "最大相对误差"]).to_string(index=False))
print("\n成批档位（这 30 只合计）")
print(f"  1 笔 {agg['b1']} 批  2 笔 {agg['b2']} 批  3-10 笔 {agg['b3']} 批  11 笔以上 {agg['b4']} 批")
print(f"  成批涉及的撤单笔数 {agg['cbatch']}，占这 30 只撤单总笔数 {sum(r[1] for r in rows)} 的 "
      f"{100 * agg['cbatch'] / sum(r[1] for r in rows):.2f}%")
