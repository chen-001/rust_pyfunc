# -*- coding: utf-8 -*-
"""把第 03 期要写进报告和 fun fact 的每个数字算一遍，存成 numbers.json。"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze import load

SB = Path("/home/chenzongwei/rust_pyfunc/sandbox_cancel_census")
DATES = [20240205, 20240329, 20240813, 20240930, 20241008, 20250407, 20251127, 20260311]

f = load()
f = f[(f.st == 1) & (f.hi != f.lo) & f.age_ok & f.amt.notna()].copy()
f["share"] = f.cancel_share * 100
N = {}

N["样本股票日"] = int(len(f))
N["八天各股票日"] = {int(d): int((f.date == d).sum()) for d in DATES}

# 1 撤单占比
N["撤单占比_中位"] = round(float(f.share.median()), 1)
N["撤单占比_P10"] = round(float(f.share.quantile(.1)), 1)
N["撤单占比_P90"] = round(float(f.share.quantile(.9)), 1)
N["撤单占比_八天中位"] = {int(d): round(float(f[f.date == d].share.median()), 1) for d in DATES}
N["撤单占比_四分之一分位"] = round(float(f.share.quantile(.25)), 1)
N["撤单占比_四分之三分位"] = round(float(f.share.quantile(.75)), 1)

# 2 笔数口径
N["全市场撤单笔数"] = int(f.cn.sum())
N["全市场成交笔数"] = int(f.tn.sum())
N["撤单笔数÷成交笔数"] = round(float(f.cn.sum() / f.tn.sum()), 3)

# 3 单笔规模
hc = np.zeros(500)
ht = np.zeros(500)
for d in DATES:
    j = json.load(open(SB / f"out/{d}.json"))
    hc += np.array(j["hist_c"], float)
    ht += np.array(j["hist_t"], float)
x = 10 ** (1.5 + .01 * np.arange(500))
edges = np.arange(1.9, 6.01, 0.2)
tab = []
for a, b in zip(edges[:-1], edges[1:]):
    m = (np.log10(x) >= a) & (np.log10(x) < b)
    tab.append([10 ** ((a + b) / 2), int(hc[m].sum()), int(ht[m].sum()), float(hc[m].sum() / max(ht[m].sum(), 1))])
N["规模档"] = [{"中点股数": round(t[0]), "撤单笔": t[1], "成交笔": t[2], "撤÷成": round(t[3], 2)} for t in tab]
N["一百股档撤÷成"] = round(tab[0][3], 2)
N["五十万股档撤÷成"] = round(tab[-1][3], 2)
N["打平股数"] = round(float(np.interp(1, [t[3] for t in tab], [t[0] for t in tab])))
N["单笔撤单股数中位"] = round(float(np.interp(.5, np.cumsum(hc) / hc.sum(), x)))
N["单笔成交股数中位"] = round(float(np.interp(.5, np.cumsum(ht) / ht.sum(), x)))

# 4 日内
b5 = {}
for d in DATES:
    t = f[f.date == d]
    cv = np.stack(t.b5_cvol.values).sum(0)
    tv = np.stack(t.b5_tvol.values).sum(0)
    s = cv / (cv + tv) * 100
    b5[int(d)] = [round(float(s[0]), 1), round(float(s[23]), 1), round(float(s[-1]), 1),
                  round(float(s.mean()), 1), round(float(s.max()), 1), round(float(s.argmax()), 0)]
N["日内"] = {k: v for k, v in b5.items()}
N["首桶中位"] = round(float(np.median([v[0] for v in b5.values()])), 1)
N["末桶中位"] = round(float(np.median([v[2] for v in b5.values()])), 1)
N["全天水平中位"] = round(float(np.median([v[3] for v in b5.values()])), 1)
N["末桶÷全天"] = round(float(np.median([v[2] / v[3] for v in b5.values()])), 2)

# 5 涨跌分组
labels = ["跌超 5%", "跌 1-5%", "涨跌不到 1%", "涨 1-5%", "涨超 5%"]
f["grp"] = pd.cut(f.ret, bins=[-1, -.05, -.01, .01, .05, 1], labels=labels)
g = []
for lab in labels:
    t = f[f.grp == lab]
    v = (t.cvol_b / (t.cvol_b + t.cvol_s)).dropna() * 100
    g.append([lab, int(len(v)), round(float(v.median()), 1), round(float(v.quantile(.1)), 0),
              round(float(v.quantile(.9)), 0)])
N["涨跌分组"] = g

# 6 成批
bn = np.stack(f.batch_n.values).sum(0).astype(float)
bv = np.stack(f.batch_vol.values).sum(0)
bt = np.stack(f.batch_tick.values).sum(0).astype(float)
N["成批"] = {"批次占比": [round(v, 1) for v in (bn / bn.sum() * 100)],
             "笔数占比": [round(v, 1) for v in (bt / bt.sum() * 100)],
             "量占比": [round(v, 1) for v in (bv / bv.sum() * 100)]}
N["成批_批次中一笔以上占比"] = round(float(bn[1:].sum() / bn.sum() * 100), 1)
N["成批_笔数占比"] = round(float(bt[1:].sum() / bt.sum() * 100), 1)
N["成批_量占比"] = round(float(bv[1:].sum() / bv.sum() * 100), 1)
N["成批_平均每批笔数"] = round(float(bt[1:].sum() / bn[1:].sum()), 2)

# 7 对账
f["ratio_amt"] = f.tamt / (f.amt * 1000)
N["对账_成交额比_中位"] = round(float(f.ratio_amt.median()), 4)
N["对账_成交额比_P5"] = round(float(f.ratio_amt.quantile(.05)), 3)
N["对账_成交额比_P95"] = round(float(f.ratio_amt.quantile(.95)), 3)
N["对账_八天中位"] = {int(d): round(float(f[f.date == d].ratio_amt.median()), 4) for d in DATES}

# 8 其他
N["全市场撤单量_亿股"] = round(float(f.cvol.sum() / 1e8), 2)
N["全市场成交量_亿股"] = round(float(f.tvol.sum() / 1e8), 2)
N["中位股票日撤单量_万股"] = round(float(f.cvol.median() / 1e4), 1)
N["中位股票日成交量_万股"] = round(float(f.tvol.median() / 1e4), 1)
N["撤买占全部撤单"] = round(float(f.cvol_b.sum() / f.cvol.sum() * 100), 1)

json.dump(N, open(SB / "numbers.json", "w"), ensure_ascii=False, indent=1)
for k, v in N.items():
    if k in ("规模档",):
        continue
    print(f"{k}: {v}")
