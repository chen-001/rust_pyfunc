# -*- coding: utf-8 -*-
"""把报告里用到的每个数都算出来，写 numbers.json 和 数字汇总.xlsx。"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze import load, T

SB = Path("/home/chenzongwei/rust_pyfunc/sandbox_trade_rhythm")
OUT = Path("/home/chenzongwei/hf-observation/2026-10-25-成交的节奏")
LEG = {20240205: "小盘股流动性危机", 20240329: "常态 A", 20240813: "地量", 20240930: "924 行情",
       20241008: "历史天量", 20250407: "关税暴跌", 20251127: "常态 B", 20260311: "常态 C"}


def main():
    df, A = load()
    q = np.zeros(len(df), dtype=int)
    for _, g in df.groupby("date"):
        q[g.index.values] = pd.qcut(g.s, 5, labels=False).values
    df["q"] = q
    fx = np.linspace(0, 1, A["LZ"].shape[1])
    N = {}
    N["股票日"] = int(len(df))
    N["每天股票日"] = df.groupby("date").size().to_dict()
    N["中位成交笔数"] = float(df.s.median())
    N["中位每秒笔数"] = float(df.lam.median())
    N["中位事件数"] = float(df.c.median())
    N["记录数除以事件数_中位"] = float((df.s / df.c).median())
    N["同一刻度的记录占比_中位"] = float((1 - df.c / df.s).median())
    N["簇大小按笔加权_中位"] = float((df.s2 / df.s).median())
    N["对账_成交额比_中位"] = float((df.tamt / (df.amt * 1000)).median())
    N["对账_成交额比_最小"] = float((df.tamt / (df.amt * 1000)).min())
    N["对账_成交额比_最大"] = float((df.tamt / (df.amt * 1000)).max())
    N["对账_成交量比_中位"] = float((df.tvol / (df.vol * 100)).median())
    N["纯成团倍数_中位"] = float(df.ratio_p.median())
    N["纯成团倍数_均值"] = float(df.ratio_p.mean())
    N["纯成团倍数_分位"] = {k: float(df.ratio_p.quantile(v)) for k, v in
                        [("10%", .1), ("25%", .25), ("50%", .5), ("75%", .75), ("90%", .9)]}

    # 全样本累计曲线
    lr, ls = A["LZ"].mean(0), A["LZS"].mean(0)
    N["一半成交时间_实际"] = float(np.interp(0.5, lr, fx))
    N["一半成交时间_对照"] = float(np.interp(0.5, ls, fx))
    N["最忙1%_成交占比_实际"] = float(np.interp(0.01, fx, lr))
    N["最忙1%_成交占比_对照"] = float(np.interp(0.01, fx, ls))

    rows = []
    for qi in range(5):
        m = (df.q == qi).values
        h, hs = A["H"][m].sum(0), A["HS"][m].sum(0)
        l1, l2 = A["LZ"][m].mean(0), A["LZS"][m].mean(0)
        rp = df.ratio_p[m]
        rows.append({
            "组": qi + 1,
            "每秒笔数中位": float(df.lam[m].median()),
            "成交笔数中位": float(df.s[m].median()),
            "空秒占比_实际": float(h[0] / h.sum()),
            "空秒占比_对照": float(hs[0] / hs.sum()),
            "一半成交时间_实际": float(np.interp(0.5, l1, fx)),
            "一半成交时间_对照": float(np.interp(0.5, l2, fx)),
            "最忙1%成交占比_实际": float(np.interp(0.01, fx, l1)),
            "最忙1%成交占比_对照": float(np.interp(0.01, fx, l2)),
            "纯成团_中位": float(rp.median()),
            "纯成团_25%": float(rp.quantile(.25)),
            "纯成团_75%": float(rp.quantile(.75)),
            "纯成团_10%": float(rp.quantile(.1)),
            "纯成团_90%": float(rp.quantile(.9)),
            "共同作息份额_中位": float(df.common[m].median()),
        })
    Q = pd.DataFrame(rows)

    D = df.groupby("date").agg(股票日=("s", "size"), 中位成交笔数=("s", "median"), 每秒笔数中位=("lam", "median"),
                              纯成团中位=("ratio_p", "median"), 共同作息份额=("common", "median"),
                              对账比=("tamt", lambda x: 1.0))
    D["对账比"] = df.groupby("date").apply(lambda g: (g.tamt / (g.amt * 1000)).median(), include_groups=False)
    D["一半成交时间"] = [float(np.interp(0.5, A["LZ"][(df.date == d).values].mean(0), fx)) for d in D.index]
    D["一半成交时间_对照"] = [float(np.interp(0.5, A["LZS"][(df.date == d).values].mean(0), fx)) for d in D.index]
    D["名称"] = [LEG[d] for d in D.index]
    D = D.set_index("名称", append=True).reorder_levels([1, 0])

    N["五分位"] = Q.to_dict(orient="records")
    N["按日"] = D.reset_index().to_dict(orient="records")
    print(json.dumps({k: v for k, v in N.items() if not isinstance(v, list)}, ensure_ascii=False, indent=1))
    print(Q.round(3).to_string())
    print(D.round(3).to_string())

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "numbers.json").write_text(json.dumps(N, ensure_ascii=False, indent=1), encoding="utf-8")
    with pd.ExcelWriter(OUT / "数字汇总.xlsx") as xw:
        Q.round(4).to_excel(xw, "活跃度五分位", index=False)
        D.reset_index().round(4).to_excel(xw, "八个展示日", index=False)
        pd.DataFrame([{"项目": k, "数值": v} for k, v in N.items() if not isinstance(v, list)]).to_excel(
            xw, "汇总", index=False)
    print("写出", OUT / "numbers.json", OUT / "数字汇总.xlsx")


if __name__ == "__main__":
    main()
