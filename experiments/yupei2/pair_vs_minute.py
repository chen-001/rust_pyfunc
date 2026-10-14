#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""配对互补矩阵 vs 分钟成交量相似度矩阵的对照。

回答用户提的那句「把这个矩阵和之前版本的矩阵，例如直接算 B 和 C 的 rho，
或者分钟成交量相似度指标之类的构造的那种 N×N 矩阵做对比与交互」。

分钟成交量相似度：读 /ssd_data/data/1min_factor_text 的 volume.h5 当天 240 根 K 线，
取 log1p 后按股票标准化，两两皮尔逊相关，得到 N×N。
对照看三件事：(1) 三个矩阵两两的全配对 Spearman；(2) 在 rho 十分位内，
分钟相关与互补量 ZS 的关系（扣掉 tick 相似度以后分钟量还有没有增量）；
(3) 分钟相关自己能不能单独还原出高互补配对。

运行: python experiments/yupei2/pair_vs_minute.py <pair_out_dir> <date>
"""
import os
import sys
import json
import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, "/home/chenzongwei/rust_pyfunc/python")
from rust_pyfunc.minute_data_reader import read_minute_data   # noqa: E402


def main():
    d, date = sys.argv[1], int(sys.argv[2])
    day = os.path.join(d, str(date))
    codes = [l.strip() for l in open(os.path.join(day, "codes.txt")) if l.strip()]
    n = len(codes)

    def rd(name):
        return np.fromfile(os.path.join(day, name), dtype="<f4").reshape(n, n)

    vol = read_minute_data("volume", date, date)
    miss = [c for c in codes if c not in vol.columns]
    # read_minute_data 会把每天最后 4 根 K 线置成 NaN（收盘集合竞价那一段），
    # 必须先去掉落进 NaN 的行，否则 isfinite 会对所有股票都判 False。
    sub = vol.iloc[:236].reindex(columns=codes).values.astype(np.float64)
    keep = np.isfinite(sub).all(axis=0) & (np.nanstd(sub, axis=0) > 0)
    X = np.log1p(np.where(keep, sub, 0.0))
    X = (X - X.mean(axis=0)) / np.maximum(X.std(axis=0), 1e-12)
    C = (X.T @ X) / X.shape[0]
    np.fill_diagonal(C, 1.0)
    idx = np.where(keep)[0]
    print(f"{date}: 分钟 volume 有值的股票 {len(idx)}/{n}，缺失 {len(miss)} 只")

    sel = lambda M: M[np.ix_(idx, idx)]           # noqa: E731
    iu = np.triu_indices(len(idx), 1)
    Cm = C[np.ix_(idx, idx)]
    rho, G, Z, ZS = sel(rd("rho.bin")), sel(rd("G.bin")), sel(rd("Z.bin")), sel(rd("ZS.bin"))

    out = {"date": date, "n": n, "n_minute": int(len(idx)), "n_missing_minute": len(miss)}
    for tag, a, b in [("minute_vs_rho", Cm, rho), ("minute_vs_G", Cm, G),
                      ("minute_vs_Z", Cm, Z), ("minute_vs_ZS", Cm, ZS),
                      ("rho_vs_G", rho, G), ("rho_vs_ZS", rho, ZS),
                      ("Z_vs_ZS", Z, ZS)]:
        out[tag] = float(spearmanr(a[iu], b[iu]).statistic)
        print(f"  Spearman({tag}) = {out[tag]:.4f}")

    bins = np.percentile(rho[iu], np.arange(0, 101, 10))
    per = []
    for k in range(10):
        m = (rho[iu] >= bins[k]) & (rho[iu] < bins[k + 1]) if k < 9 else (rho[iu] >= bins[k])
        if m.sum() < 2000:
            continue
        r = float(spearmanr(Cm[iu][m], ZS[iu][m]).statistic)
        per.append([k, int(m.sum()), r])
        print(f"  rho 第{k}档 n={m.sum()}: Spearman(分钟相关, ZS) = {r:.4f}")
    out["minute_vs_ZS_within_rho_decile"] = per

    # 分钟相关自己选出来的高互补配对命中率：按分钟相关取前 1% 配对，看其中有多少 ZS>2
    thr = np.percentile(Cm[iu], 99.0)
    hot = Cm[iu] >= thr
    out["top1pct_minute_corr_hit_ZSgt2"] = float((ZS[iu][hot] > 2.0).mean())
    out["top1pct_rho_hit_ZSgt2"] = float((ZS[iu][rho[iu] >= np.percentile(rho[iu], 99.0)] > 2.0).mean())
    out["base_ZSgt2_share"] = float((ZS[iu] > 2.0).mean())
    print(f"  分钟相关前 1% 配对里 ZS>2 占比 {out['top1pct_minute_corr_hit_ZSgt2']:.4f}；"
          f"rho 前 1% 里 {out['top1pct_rho_hit_ZSgt2']:.4f}；"
          f"全体基线 {out['base_ZSgt2_share']:.4f}")

    p = f"/tmp/yupei2_minute_vs_pair_{date}.json"
    with open(p, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("写入", p)


if __name__ == "__main__":
    main()
