#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""跨日稳定性对照：未过滤口径 vs 股票白名单口径，都只统计「股票-股票」配对。

必须只在股票对上算。非个股那 280 行整行整列是 0，两天都在同样的位置是 0，
把它们算进去会把相关系数整体抬高（此前汇报里的 0.781 / 0.685 就是这么来的）。

用法: python experiments/yupei2/pair_crossday.py <未过滤目录> <过滤目录> <date1> <date2>
"""
import os
import sys
import json
import numpy as np
from scipy.stats import spearmanr

SM = set(l.split(",")[0]
         for l in open("/ssd_data/data/basic_info/symbol_map.csv").read().splitlines()[1:] if l)


def load(root, date):
    dd = os.path.join(root, str(date))
    codes = [l.strip() for l in open(os.path.join(dd, "codes.txt")) if l.strip()]
    n = len(codes)
    out = {"codes": codes, "n": n,
           "is_stock": np.array([c in SM for c in codes])}
    for nm in ("rho.bin", "G.bin", "Z.bin", "ZS.bin"):
        out[nm[:-4]] = np.fromfile(os.path.join(dd, nm), dtype="<f4").reshape(n, n)
    return out


def cross(root, d1, d2):
    A, B = load(root, d1), load(root, d2)
    common = sorted(set(A["codes"]) & set(B["codes"]))
    ia = {c: i for i, c in enumerate(A["codes"])}
    ib = {c: i for i, c in enumerate(B["codes"])}
    ca = np.array([ia[c] for c in common])
    cb = np.array([ib[c] for c in common])
    ok = np.array([c in SM for c in common])
    iu = np.triu_indices(len(common), 1)
    m = ok[iu[0]] & ok[iu[1]]
    r = {"共同股票": len(common), "共同配对": int(m.sum())}
    for key in ("rho", "G", "Z", "ZS"):
        # 先按需要对行、再按需要对列，避免整块 3746² 的大拷贝堆在一起
        a = A[key][ca][:, ca][iu[0][m], iu[1][m]]
        b = B[key][cb][:, cb][iu[0][m], iu[1][m]]
        r[key] = round(float(spearmanr(a, b).statistic), 4)
        del a, b
    return r


def main():
    un_root, so_root, d1, d2 = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    out = {"date_pair": [d1, d2]}
    for tag, root in (("未过滤", un_root), ("股票白名单", so_root)):
        r = cross(root, d1, d2)
        out[tag] = r
        print(f"  {tag:<8} 共同股票 {r['共同股票']}  共同配对 {r['共同配对']:,}  "
              f"Spearman(rho)={r['rho']}  Spearman(G)={r['G']}  "
              f"Spearman(Z)={r['Z']}  Spearman(ZS)={r['ZS']}")
    p = f"/tmp/yupei2_crossday_{d1}_{d2}.json"
    with open(p, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("写入", p)


if __name__ == "__main__":
    main()
