"""跑因子：逐因子读 parquet（算完即释放），输出 metrics.csv。"""
import os, sys, glob, csv, time
import numpy as np
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from metrics_core import factor_metrics

FOLDERS = ["hm100", "hm101", "hm102", "hm103", "hm104", "hm105"]
FROOT = "/nas197/user_home_unsafe/chenzongwei/factor_data"
CACHE = os.path.join(HERE, "returns_cache.npz")
OUT = os.path.join(HERE, "metrics.csv")
NWORK = 16

_G = {}


def _init():
    z = np.load(CACHE)
    _G["codes"] = [str(c) for c in z["codes"]]
    _G["r5"] = z["r5"]
    _G["dates"] = z["dates"]


def _work(path):
    folder = os.path.basename(os.path.dirname(path))
    name = os.path.basename(path)[:-8]
    m = factor_metrics(path, _G["codes"], _G["r5"], _G["dates"])
    if m is None:
        return None
    row = {"factor": folder + "/" + name, "folder": folder, "nd": m["nd"],
           "nbad": m["nbad"], "nmir_bad": m["nmir_bad"], "ic": m["ic"],
           "resid_mean": m["resid"], "max_mirror_dev": m["max_mirror_dev"],
           "lambda": m["lam"], "lambda_short": m["l_short"],
           "lambda_long": m["l_long"], "ssm": m["ssm"], "gamma": m["gamma"],
           "share": m["share"], "share_ic": m["share_ic"]}
    for d in range(10):
        row[f"r{d+1}"] = m["r"][d] * 1e4
        row[f"c{d+1}"] = m["c"][d]
        row[f"gsd{d+1}"] = m["gsd"][d] * 1e4
    return row


COLS = (["factor", "folder", "nd", "nbad", "nmir_bad", "ic", "resid_mean",
         "max_mirror_dev", "lambda", "lambda_short", "lambda_long", "ssm",
         "gamma", "share", "share_ic"]
        + [f"r{i}" for i in range(1, 11)]
        + [f"c{i}" for i in range(1, 11)]
        + [f"gsd{i}" for i in range(1, 11)])


def main():
    files = []
    for f in FOLDERS:
        files += sorted(glob.glob(os.path.join(FROOT, f, "*.parquet")))
    print("factors:", len(files), flush=True)
    t0 = time.time()
    rows = []
    with Pool(NWORK, initializer=_init) as p:
        for k, r in enumerate(p.imap_unordered(_work, files, chunksize=2), 1):
            if r is not None:
                rows.append(r)
            if k % 20 == 0:
                print(f"{k}/{len(files)} {time.time()-t0:.0f}s", flush=True)
    with open(OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS)
        w.writeheader()
        for r in sorted(rows, key=lambda x: x["factor"]):
            w.writerow(r)
    print("done", len(rows), "rows", round(time.time() - t0, 1), "s", flush=True)


if __name__ == "__main__":
    main()
