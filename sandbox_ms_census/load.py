# -*- coding: utf-8 -*-
"""第 07 期：共用的日频过滤与结果读取。过滤规则沿用第 06 期：
正常交易、上市满 60 个交易日、当日非一字板；股票只取 0/3/6 开头（沪、深主板、创业板）。"""
import json
from pathlib import Path
import h5py, numpy as np, pandas as pd

SB = Path("/home/chenzongwei/rust_pyfunc/sandbox_ms_census")
DATES = [20240205, 20240329, 20240813, 20240930, 20241008, 20250407, 20251127, 20260311]
ROOT = "/ssd_data/data/vars"

def _maps(d):
    cal = [int(x) for x in open(f"{ROOT}/{d}/calendar_map.csv").read().splitlines()[1:]]
    sym = [s.split(",")[0].zfill(6) for s in open(f"{ROOT}/{d}/symbol_map.csv").read().splitlines()[1:]]
    return cal, sym

def _field(d, f, i0, i1):
    cal, sym = _maps(d)
    with h5py.File(f"{ROOT}/{d}/{f}.h5", "r") as h:
        a = h["data"][i0:i1, :len(sym)]
    return pd.DataFrame(a, index=cal[i0:i1], columns=sym)

def daily():
    cal = _maps("DailyPrice")[0]
    i0 = next(i for i, x in enumerate(cal) if x >= 20230801)
    i1 = max(i for i, x in enumerate(cal) if x <= max(DATES)) + 1
    return {k: _field(f, v, i0, i1) for k, (f, v) in {
        "st": ("DailyPrice", "S_DQ_TRADESTATUS"), "hi": ("DailyPrice", "S_DQ_HIGH"),
        "lo": ("DailyPrice", "S_DQ_LOW"), "close": ("DailyPrice", "S_DQ_CLOSE"),
        "amt": ("DailyPrice", "S_DQ_AMOUNT")}.items()}

def valid_codes(D, d):
    idx = list(D["close"].index); j = idx.index(d)
    age = D["close"].iloc[max(0, j - 60):j].notna().sum() >= 60
    ok = (D["st"].loc[d] == 1) & (D["hi"].loc[d] != D["lo"].loc[d]) & age & D["amt"].loc[d].notna()
    return [c for c in ok.index[ok.values] if c[0] in "036"]

def write_valid():
    D = daily()
    for d in DATES:
        tdir = Path(f"/ssd_data/stock/{d}/transaction")
        cs = [c for c in valid_codes(D, d) if (tdir / f"{c}_{d}_transaction.csv").is_file()]
        (SB / f"out/valid_{d}.txt").write_text("\n".join(cs) + "\n")
        print(d, len(cs))

def load_day(d, tag=""):
    stem = SB / ("out/%d%s" % (d, "_" + tag if tag else ""))
    m = json.load(open(f"{stem}.json"))
    n = m["n_valid"]
    u = np.fromfile(f"{stem}.u32", np.uint32).reshape(n, m["n_col"])
    f = np.fromfile(f"{stem}.f64", np.float64).reshape(n, m["n_f"])
    A = {name: u[:, a:a + b] for name, a, b in m["layout"]}
    A["amt100"] = f[:, :100]; A["amt_all"] = f[:, 100]
    return m, A

if __name__ == "__main__":
    write_valid()
