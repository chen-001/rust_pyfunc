# -*- coding: utf-8 -*-
"""沙箱宇宙准备：对每个测试日 D，用前一交易日 P 的总市值 + 申万一级行业分类，
选出 31 行业 × 市值前 10 = 310 只，输出 universe_{date}.txt（每行 "code ind"，code 为 6 位）。
行业取自 {VARS_DIR}/SzBa/industry.h5：行对齐 calendar_map.csv 的交易日，列对齐 symbol_map.csv 的 pos，
取值即申万一级行业编号 1~31（NaN = 未知，剔除）。
市值取自 read_daily(total_cap=1)。过滤 L2 无文件的股票（顺延到下一名）与 B 股/北交所。
"""
import os
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from pure_ocean_breeze.jason.data.read_data import read_daily

HERE = os.path.dirname(os.path.abspath(__file__))
VARS_DIR = Path("/ssd_data/data/vars")
L2_ROOT = "/ssd_data/stock"

DATES = [20240104, 20240603, 20241008, 20260105, 20260716]


def suffix(code6: str) -> str:
    return code6 + (".SH" if code6.startswith(("6", "9")) else ".SZ")


def load_industry(date_ints):
    """按交易日读 industry.h5，返回 (date, code, ind) 三列 DataFrame（6 位代码，只留行业号 1~31）。"""
    szba = VARS_DIR / "SzBa"
    calendar = pd.read_csv(szba / "calendar_map.csv", dtype=np.int64)
    symbols = pd.read_csv(szba / "symbol_map.csv", dtype={"symbol": str, "pos": np.int64})
    code6 = symbols["symbol"].to_numpy()
    row_of = {int(v): r for r, v in enumerate(calendar.iloc[:, 0])}
    rows = np.array([row_of[int(d)] for d in date_ints], dtype=np.intp)
    uniq_rows = np.unique(rows)
    block_r = np.searchsorted(uniq_rows, rows)
    with h5py.File(szba / "industry.h5", "r") as h5:
        values = h5["data"][uniq_rows][:, : len(code6)][block_r]
    frames = []
    for k, d in enumerate(date_ints):
        v = values[k]
        keep = np.isfinite(v) & (v >= 1)
        frames.append(pd.DataFrame({"date": int(d), "code": code6[keep], "ind": v[keep].astype(int)}))
    return pd.concat(frames, ignore_index=True)


def main():
    tc = read_daily(total_cap=1)
    idx = tc.index
    prevs = []
    for d in DATES:
        dd = pd.Timestamp(f"{d // 10000}-{(d // 100) % 100:02d}-{d % 100:02d}")
        prevs.append(idx[idx.get_loc(dd) - 1])
    barra = load_industry([int(p.strftime("%Y%m%d")) for p in prevs])

    for d, prev in zip(DATES, prevs):
        cap = tc.loc[prev].dropna()
        cap.name = "cap"
        b = barra[barra.date == int(prev.strftime("%Y%m%d"))][["code", "ind"]].drop_duplicates("code").set_index("code")
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
