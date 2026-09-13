# -*- coding: utf-8 -*-
"""沙箱宇宙准备：对每个测试日 D，用前一交易日 P 的总市值 + 申万一级行业分类，
选出 31 行业 × 市值前 10 = 310 只，输出 universe_{date}.txt（每行 "code ind"，code 为 6 位）。
行业取自 /home/chenzongwei/database/barra/barra_daily_together.parquet 的 ind_1..ind_31 argmax。
市值取自 read_daily(total_cap=1)。过滤 L2 无文件的股票（顺延到下一名）与 B 股/北交所。
"""
import os
import pandas as pd
import numpy as np
from pure_ocean_breeze.jason.data.read_data import read_daily

HERE = os.path.dirname(os.path.abspath(__file__))
BARRA = "/home/chenzongwei/database/barra/barra_daily_together.parquet"
L2_ROOT = "/ssd_data/stock"

DATES = [20240104, 20240603, 20241008, 20260105, 20260716]


def suffix(code6: str) -> str:
    return code6 + (".SH" if code6.startswith(("6", "9")) else ".SZ")


def main():
    tc = read_daily(total_cap=1)
    idx = tc.index
    barra = pd.read_parquet(BARRA, columns=["date", "code"] + [f"ind_{i}" for i in range(1, 32)])
    ind_cols = [f"ind_{i}" for i in range(1, 32)]
    barra["ind"] = barra[ind_cols].idxmax(axis=1).str.replace("ind_", "").astype(int)
    barra[barra[ind_cols].sum(axis=1) == 0] = 0  # 无行业置 0
    barra = barra[["date", "code", "ind"]]

    for d in DATES:
        dd = pd.Timestamp(f"{d // 10000}-{(d // 100) % 100:02d}-{d % 100:02d}")
        i = idx.get_loc(dd)
        prev = idx[i - 1]
        cap = tc.loc[prev].dropna()
        cap.name = "cap"
        b = barra[barra.date == prev][["code", "ind"]].drop_duplicates("code").set_index("code")
        df = cap.to_frame().join(b, how="inner")
        df = df[df.ind >= 1]
        df["code6"] = [c.split(".")[0] for c in df.index]
        # 过滤 B 股 / 北交所 / L2 无文件
        df = df[~df.code6.str.startswith(("900", "200", "4", "8"))]
        have = {f.split("_")[0] for f in os.listdir(f"{L2_ROOT}/{d}/transaction")}
        df = df[df.code6.isin(have)]
        # 每行业 top10（按总市值）
        top = df.sort_values("cap", ascending=False).groupby("ind").head(10)
        top = top.sort_values(["ind", "cap"], ascending=False)
        print(f"{d} prev={prev.date()} universe={len(top)} industries={top.ind.nunique()}")
        with open(f"{HERE}/universe_{d}.txt", "w") as f:
            for code6, ind in zip(top.code6, top.ind):
                f.write(f"{code6} {ind}\n")
        # 每行业前十检查
        print(top.groupby("ind").size().describe())
        # 全市场行业映射（供全市场同伴池对照实验）
        with open(f"{HERE}/barra_all_{d}.txt", "w") as f:
            for code6, ind in zip(df.code6, df.ind):
                f.write(f"{code6} {ind}\n")
        print(f"barra_all lines={len(df)}")


if __name__ == "__main__":
    main()
