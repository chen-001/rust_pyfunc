# -*- coding: utf-8 -*-
"""把报告里用到的每个数都算出来，写 numbers.json 和 数字汇总.xlsx。"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze import load, T

OUT = Path("/home/chenzongwei/hf-observation/2026-10-25-成交的节奏")
LEG = {20240205: "小盘股流动性危机", 20240329: "常态 A", 20240813: "地量", 20240930: "924 行情",
       20241008: "历史天量", 20250407: "关税暴跌", 20251127: "常态 B", 20260311: "常态 C"}


def main():
    df, A = load()
    q = np.zeros(len(df), dtype=int)
    for _, g in df.groupby("date"):
        q[g.index.values] = pd.qcut(g.s, 5, labels=False).values
    df["q"] = q
    df["fano_net"] = df.fano * (1 - df.r2)
    fx = np.linspace(0, 1, A["LZ"].shape[1])

    N = {"股票日": int(len(df)),
         "中位记录数": float(df.s.median()), "中位事件数": float(df.e.median()),
         "记录除以事件_中位": float((df.s / df.e).median()),
         "共享时间戳的记录占比_中位": float((1 - df.e / df.s).median()),
         "八天_事件占记录": float(df.e.sum() / df.s.sum()),
         "对账_成交额比_中位": float((df.tamt / (df.amt * 1000)).median()),
         "对账_成交额比_最小": float((df.tamt / (df.amt * 1000)).min()),
         "对账_成交额比_最大": float((df.tamt / (df.amt * 1000)).max()),
         "fano_中位": float(df.fano.median()),
         "fano_分位": {k: float(df.fano.quantile(v)) for k, v in
                     [("10%", .1), ("25%", .25), ("50%", .5), ("75%", .75), ("90%", .9)]},
         "r2_中位": float(df.r2.median()),
         "fano_net_中位": float(df.fano_net.median()),
         "fano_并成一次_中位": float(df.fano_e.median())}
    lr = A["LZ"].mean(0)
    N["一半成交时间"] = float(np.interp(0.5, lr, fx))
    N["最忙1%成交占比"] = float(np.interp(0.01, fx, lr))

    rows = []
    for qi in range(5):
        m = (df.q == qi).values
        l1 = A["LZ"][m].mean(0)
        rows.append({"组": qi + 1, "每秒事件中位": df.lam[m].median(), "每秒记录中位": df.s[m].median() / T,
                     "空秒占比": float(A["H"][m].sum(0)[0] / A["H"][m].sum()),
                     "一半成交时间": float(np.interp(0.5, l1, fx)),
                     "最忙1%成交占比": float(np.interp(0.01, fx, l1)),
                     "fano_中位": df.fano[m].median(), "fano_25%": df.fano[m].quantile(.25),
                     "fano_75%": df.fano[m].quantile(.75), "fano_10%": df.fano[m].quantile(.1),
                     "fano_90%": df.fano[m].quantile(.9),
                     "r2_中位": df.r2[m].median(), "fano_net_中位": df.fano_net[m].median(),
                     "fano_并成一次_中位": df.fano_e[m].median()})
    Q = pd.DataFrame(rows)
    D = df.groupby("date").agg(股票日=("s", "size"), 中位记录数=("s", "median"), 每秒事件中位=("lam", "median"),
                              fano中位=("fano", "median"), fano并成一次=("fano_e", "median"), r2中位=("r2", "median"),
                              fano净中位=("fano_net", "median")).reset_index()
    D["对账比"] = [float((df[df.date == d].tamt / (df[df.date == d].amt * 1000)).median()) for d in D.date]
    D["名称"] = [LEG[d] for d in D.date]
    D = D[["date", "名称", "股票日", "中位记录数", "每秒事件中位", "fano中位", "fano并成一次", "r2中位", "fano净中位", "对账比"]]

    N["五分位"] = Q.to_dict(orient="records")
    N["按日"] = D.to_dict(orient="records")
    print(json.dumps({k: v for k, v in N.items() if not isinstance(v, list)}, ensure_ascii=False, indent=1))
    print(Q.round(3).to_string())
    print(D.round(3).to_string())

    (OUT / "numbers.json").write_text(json.dumps(N, ensure_ascii=False, indent=1), encoding="utf-8")
    with pd.ExcelWriter(OUT / "数字汇总.xlsx") as xw:
        Q.round(4).to_excel(xw, "活跃度五分位", index=False)
        D.round(4).to_excel(xw, "八个展示日", index=False)
        pd.DataFrame([{"项目": k, "数值": v} for k, v in N.items() if not isinstance(v, list)]).to_excel(
            xw, "汇总", index=False)
    print("写出", OUT / "numbers.json", OUT / "数字汇总.xlsx")


if __name__ == "__main__":
    main()
