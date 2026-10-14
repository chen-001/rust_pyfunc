#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
"""测「每对取 h 前 K 只 X」这套放宽方案的覆盖率与因子质量。

K 默认取 1,2,5,10,20,50,100，可用第 5 个参数换成别的档位表。
对每一档看四件事：
  1 覆盖率：有多少只股票被选进过前 K（K=1 就是原来那套，只有 25%）
  2 集中度：被选中的次数前 10 名占多少
  3 跨日稳定性：两天共同股票上，mean_gain 的秩相关
  4 截面独立性：与现有相似度因子（r2_best_single / r2_best_pair / r3_basket_rho8）的秩相关

用法: python experiments/yupei2/pair_topk_test.py <topk根目录> <reduce根目录> <date1> <date2> [K表] [子目录名]
  例: python experiments/yupei2/pair_topk_test.py /tmp/yupei2_pair_topk /tmp/yupei2_pair_rs 20241231 20241230
"""
import os
import sys
import json
import numpy as np
from scipy.stats import rankdata

PAIR_ROOT = "/tmp/yupei2_pair_so"
SM = set(l.split(",")[0]
         for l in open("/ssd_data/data/basic_info/symbol_map.csv").read().splitlines()[1:] if l)


def load(d):
    names = [l.strip() for l in open(os.path.join(d, "names.txt")) if l.strip()]
    v = np.fromfile(os.path.join(d, "factors.bin"), dtype="<f4")
    return names, v.reshape(v.size // len(names), len(names))


def stock_codes(date):
    cs = [l.strip() for l in open(os.path.join(PAIR_ROOT, str(date), "codes.txt")) if l.strip()]
    return [c for c in cs if c in SM]


def sp(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 50:
        return float("nan")
    return float(np.corrcoef(rankdata(a[m]), rankdata(b[m]))[0, 1])


def main():
    tk_root, rd_root, d1, d2 = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    Ks = [int(x) for x in (sys.argv[5].split(",") if len(sys.argv) > 5
                           else ["1", "2", "5", "10", "20", "50", "100"])]
    sub = sys.argv[6] if len(sys.argv) > 6 else "topk"

    n1, V1 = load(os.path.join(tk_root, str(d1), sub))
    n2, V2 = load(os.path.join(tk_root, str(d2), sub))
    i1, i2 = {n: k for k, n in enumerate(n1)}, {n: k for k, n in enumerate(n2)}
    c1, c2 = stock_codes(d1), stock_codes(d2)
    assert len(c1) == V1.shape[0] and len(c2) == V2.shape[0], (len(c1), V1.shape[0], len(c2), V2.shape[0])
    common = sorted(set(c1) & set(c2))
    a1 = np.array([c1.index(c) for c in common])
    a2 = np.array([c2.index(c) for c in common])

    rn, RV = load(os.path.join(rd_root, str(d1), "reduce_rs"))
    ir = {n: k for k, n in enumerate(rn)}
    ref = {k: RV[a1, ir[k]] for k in ("r2_best_single", "r2_best_pair", "r3_basket_rho8")}

    print("=" * 96)
    print(f"每对取 h 前 K 只 X（{d1} / {d2}，股票 {V1.shape[0]} 只，共同 {len(common)} 只）")
    print("=" * 96)
    print(f"{'K':>5} {'覆盖率':>7} {'占比':>7} {'前10名占比':>10} {'跨日稳定性':>10} "
          f"{'vs best_single':>15} {'vs best_pair':>13} {'vs basket8':>11} {'选中次数中位':>12}")
    out = {}
    for K in Ks:
        if f"x0topk{K}_count" not in i1:
            print(f"{K:>5}   这一档不在产物里，跳过")
            continue
        ca = V1[:, i1[f"x0topk{K}_count"]]
        ga = V1[a1, i1[f"x0topk{K}_gain_mean"]]
        gb = V2[a2, i2[f"x0topk{K}_gain_mean"]]
        cov = int((ca > 0).sum())
        srt = np.sort(ca)[::-1]
        top10 = float(srt[:10].sum() / max(ca.sum(), 1))
        m = np.isfinite(ga) & np.isfinite(gb)
        sd = sp(ga, gb)
        row = {"K": K, "覆盖率": cov, "占比": round(cov / V1.shape[0], 4),
               "前10名占比": round(top10, 4),
               "跨日稳定性": None if sd != sd else round(sd, 4),
               "跨日比较股票数": int(m.sum()),
               "选中次数中位": float(np.median(ca[ca > 0])) if cov else 0.0}
        for k, v in ref.items():
            row["vs " + k] = round(sp(ga, v), 4) if cov else float("nan")
        out[f"K={K}"] = row
        print(f"{K:>5} {cov:>7} {cov/V1.shape[0]*100:>6.1f}% {top10:>10.4f} "
              f"{sd:>10.4f} {row['vs r2_best_single']:>15.4f} "
              f"{row['vs r2_best_pair']:>13.4f} {row['vs r3_basket_rho8']:>11.4f} "
              f"{row['选中次数中位']:>12.0f}")

    p = f"/tmp/yupei2_topk_test_{d1}.json"
    with open(p, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("写入", p)


if __name__ == "__main__":
    main()
