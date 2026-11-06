# -*- coding: utf-8 -*-
"""小样本出图检查：用 tiny.json 验证字体与版式，不用于报告。"""
import json
import numpy as np
import pandas as pd
from analyze import curves, seg_curves
import plots as P

o = json.load(open("out/tiny.json"))
m = o["meta"]
R = o["recs"]
df = pd.DataFrame({"code": [r["code"] for r in R], "date": 20240329, "s": [r["s"] for r in R], "e": [r["e"] for r in R]})
df["q"] = pd.qcut(df.s, 5, labels=False)
A = {"occ": np.array(m["occ"], float), "occ_seg": np.array(m["occ_seg"], float),
     "seg_s": np.array(m["seg_s"], float), "n_sec": m["n_sec"]}
for k in ["cnt", "tick", "amt60", "vol60", "cnt2", "amt2", "share1", "share2"]:
    A[k] = np.array([r[k] for r in R], float)
A["seg"] = np.array([r["seg"] for r in R], np.int32)
C = curves(df, A)
P.fig1(df, C); P.fig2(C); P.fig3(C); P.fig4(df, C); P.fig5(df, A)
print("ok")
