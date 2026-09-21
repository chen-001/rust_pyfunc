"""按 收益口径/日期区间 跑因子。用法: run_variant.py <r5|r1> <start> <end> <out.csv>"""
import os, sys, glob, csv, time
import numpy as np
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from metrics_core import factor_metrics

FOLDERS = ["hm100", "hm101", "hm102", "hm103", "hm104", "hm105"]
FROOT = "/nas197/user_home_unsafe/chenzongwei/factor_data"
CACHE = os.path.join(HERE, "returns_cache.npz")
NWORK = 16
_G = {}


def _init():
    z = np.load(CACHE)
    _G["codes"] = [str(c) for c in z["codes"]]
    _G["r5"] = z[KEY][DATE_MASK]
    _G["dates"] = z["dates"][DATE_MASK]


def _work(path):
    folder = os.path.basename(os.path.dirname(path))
    name = os.path.basename(path)[:-8]
    m = factor_metrics(path, _G["codes"], _G["r5"], _G["dates"])
    if m is None:
        return None
    row = {"factor": folder + "/" + name, "folder": folder, "nd": m["nd"], "ic": m["ic"],
           "lambda": m["lam"], "lambda_short": m["l_short"], "lambda_long": m["l_long"],
           "ssm": m["ssm"], "gamma": m["gamma"], "share": m["share"],
           "share_ic": m["share_ic"], "resid_mean": m["resid"]}
    for d_ in range(10):
        row[f"r{d_+1}"] = m["r"][d_] * 1e4
        row[f"c{d_+1}"] = m["c"][d_]
        row[f"gsd{d_+1}"] = m["gsd"][d_] * 1e4
    return row


COLS = (["factor", "folder", "nd", "ic", "lambda", "lambda_short", "lambda_long",
         "ssm", "gamma", "share", "share_ic", "resid_mean"]
        + [f"r{i}" for i in range(1, 11)] + [f"c{i}" for i in range(1, 11)]
        + [f"gsd{i}" for i in range(1, 11)])

if __name__ == "__main__":
    KEY, START, END, OUT = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    z = np.load(CACHE)
    DATE_MASK = (z["dates"] >= START) & (z["dates"] <= END)
    files = []
    for f in FOLDERS:
        files += sorted(glob.glob(os.path.join(FROOT, f, "*.parquet")))
    t0 = time.time()
    rows = []
    with Pool(NWORK, initializer=_init) as p:
        for k, r in enumerate(p.imap_unordered(_work, files, chunksize=2), 1):
            if r is not None:
                rows.append(r)
    with open(OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS)
        w.writeheader()
        for r in sorted(rows, key=lambda x: x["factor"]):
            w.writerow(r)
    print(OUT, len(rows), "rows", round(time.time() - t0, 1), "s", flush=True)
