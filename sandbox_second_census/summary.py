# -*- coding: utf-8 -*-
"""第 06 期：数字汇总表。"""
from pathlib import Path

import numpy as np
import pandas as pd

from analyze import load, curves, seg_curves

OUT = Path("/home/chenzongwei/hf-observation/2026-11-06-成交落在第几秒")
MKT = {"6": "沪市", "0": "深市主板", "3": "创业板"}
SEG = ["开盘 5 分钟", "上午其余", "午后", "最后 5 分钟"]


def stats(df, C, A):
    med = np.nanmedian(C["inten"], 0)
    other = np.nanmedian(np.delete(med, np.s_[::3]))
    tick = np.nanmedian(A["cnt"] / np.maximum(A["tick"], 1))
    multi = np.nanmedian(C["amt_mult"], 0)
    return {
        "股票日": len(df),
        "第0秒倍数": round(float(med[0]), 3),
        "第1秒倍数": round(float(med[1]), 3),
        "第3秒倍数": round(float(med[3]), 3),
        "第59秒倍数": round(float(med[59]), 3),
        "第0秒比第59秒": round(float(med[0] / med[59]), 2),
        "3秒整倍数比其他": round(float(np.nanmedian(med[::3]) / other), 3),
        "每刻度笔数": round(float(tick), 2),
        "每笔金额最低倍数": round(float(np.nanmin(multi)), 3),
        "每笔金额最高倍数": round(float(np.nanmax(multi)), 3),
        "全天量波动中位": round(float(np.nanmedian(C["disp"])), 2),
        "占比波动中位": round(float(np.nanmedian(C["share_cv"])), 2),
    }


def rows_by(df, C, A, col, names):
    out = []
    for k, g in df.groupby(col):
        idx = g.index.values
        d = stats(g, {x: y[idx] for x, y in C.items()}, {x: (y[idx] if isinstance(y, np.ndarray) and y.shape[0] == len(df) else y) for x, y in A.items()})
        out.append({"分组": names.get(k, k) if isinstance(names, dict) else f"{names}{k + 1}", **d})
    return pd.DataFrame(out)


def main():
    df, A = load()
    C = curves(df, A)
    S = seg_curves(df, A)
    df["mkt"] = df.code.str[0].map(MKT)

    t_day = rows_by(df.assign(k=df.date), C, A, "k", {})
    t_day.insert(0, "日期", t_day.pop("分组"))

    t_mkt = rows_by(df.assign(k=df.code.str[0]), C, A, "k", MKT)

    t_q = rows_by(df, C, A, "q", "第 ")

    seg_rows = []
    for k in range(4):
        v = np.nanmedian(S[:, k, :], 0)
        seg_rows.append({"分段": SEG[k], "第0秒": round(float(v[0]), 3), "第1秒": round(float(v[1]), 3),
                         "第3秒": round(float(v[3]), 3), "第59秒": round(float(v[59]), 3),
                         "段内最低": round(float(np.nanmin(v)), 3), "段内最高": round(float(np.nanmax(v)), 3)})
    t_seg = pd.DataFrame(seg_rows)

    t_all = pd.DataFrame([{"全部": "八天", **stats(df, C, A)}])

    with pd.ExcelWriter(OUT / "数字汇总.xlsx", engine="openpyxl") as w:
        t_all.to_excel(w, sheet_name="总览", index=False)
        t_day.to_excel(w, sheet_name="按日子", index=False)
        t_mkt.to_excel(w, sheet_name="按市场", index=False)
        t_q.to_excel(w, sheet_name="按活跃度", index=False)
        t_seg.to_excel(w, sheet_name="按分段", index=False)
    print(t_all.T.to_string())
    print()
    print(t_day.to_string(index=False))
    print()
    print(t_mkt.to_string(index=False))
    print()
    print(t_seg.to_string(index=False))


if __name__ == "__main__":
    main()
