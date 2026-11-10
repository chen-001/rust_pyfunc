"""方向 2 第三版：只读 Rust 产物做检查，不做任何因子合成。

用法：python validate_v3.py <prefix> [<prefix> ...]
每个 prefix 对应 <prefix>.f32（按列存放）+ <prefix>.meta.json。
检查：列数、命名约定（cxl）、撤单空值规则、跨日列、各族覆盖、市值与行业对账、2016 读取。
"""
import os

for k in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ[k] = "1"
import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/chenzongwei/.agents/skills/daily-data-reader")
import daily_data_reader as ddr


def load(prefix):
    meta = json.load(open(prefix + ".meta.json"))
    v = np.fromfile(prefix + ".f32", dtype="<f4")
    n_cols, n_codes = meta["n_cols"], meta["n_codes"]
    assert v.size == n_cols * n_codes, (v.size, n_cols, n_codes)
    return meta, v.reshape(n_cols, n_codes)


def family(name):
    for suf in ["__nrank_act__d1", "__nrank_act", "__nrank_cap", "__rank_ind"]:
        if name.endswith(suf):
            if suf == "__nrank_act" and any(x in name for x in ["__pm_minus_full__", "__tail_minus_full__", "__tail_minus_pm__"]):
                return "time_diff"
            return suf.strip("_")
    return "raw"


def check(prefix):
    meta, X = load(prefix)
    names = np.array(meta["names"])
    codes = np.array(meta["codes"])
    date = meta["date"]
    print(f"\n===== {prefix}  date={date} prev={meta['prev_date']} codes={len(codes)} cols={len(names)} dropped(today/prev)={meta['dropped_today']}/{meta['dropped_prev']}")
    assert len(names) == 30528, len(names)
    assert len(set(names)) == len(names), "列名重复"
    spec_cxl = {s["name"]: s["cxl"] for s in meta["specs"]}
    seq_of = np.array([n.split("__")[0] for n in names])
    is_cxl_col = np.array(["cxl" in n for n in names])
    assert all(spec_cxl[s] == c for s, c in zip(seq_of, is_cxl_col)), "命名约定不一致"
    print("命名约定：含 cxl 的列", int(is_cxl_col.sum()), "不含", int((~is_cxl_col).sum()))

    avail = np.array(meta["cxl_available"])
    sh = np.array([c.startswith("6") for c in codes])
    fam = np.array([family(n) for n in names])
    fin = np.isfinite(X)
    # 撤单空值规则：不可用股票的所有 cxl 列必须全空
    bad = fin[np.ix_(is_cxl_col, ~avail)].sum()
    print("撤单不可用股票数", int((~avail).sum()), "（沪", int((~avail & sh).sum()), "深", int((~avail & ~sh).sum()), "）；这些股票 cxl 列非空格数 =", int(bad))
    assert bad == 0
    for grp, mask in [("沪市", sh), ("深市", ~sh)]:
        if mask.sum():
            print(f"  {grp} cxl 列有值比例 {fin[np.ix_(is_cxl_col, mask)].mean():.4f}   非 cxl 列有值比例 {fin[np.ix_(~is_cxl_col, mask)].mean():.4f}")
    d1 = fam == "nrank_act__d1"
    for grp, mask in [("沪市", sh), ("深市", ~sh)]:
        if mask.sum():
            print(f"  跨日列 {grp}: cxl 有值比例 {fin[np.ix_(d1 & is_cxl_col, mask)].mean():.4f}  非 cxl {fin[np.ix_(d1 & ~is_cxl_col, mask)].mean():.4f}")

    rows = []
    for f in ["raw", "nrank_act", "nrank_cap", "rank_ind", "time_diff", "nrank_act__d1"]:
        m = fam == f
        cov = fin[m].sum(1)
        rows.append({"族": f, "列数": int(m.sum()), "每列有值股票数中位": float(np.median(cov)), "全空列数": int((cov == 0).sum()), "有值少于1000只的列数": int((cov < 1000).sum())})
    print(pd.DataFrame(rows).to_string(index=False))
    # 序列层覆盖（全天原值的 cov 特征）
    sc = []
    for s in meta["specs"]:
        j = list(names).index(f"{s['name']}__full__cov")
        sc.append((s["name"], int(fin[j].sum())))
    sc = pd.DataFrame(sc, columns=["seq", "full_cov_有值股票数"])
    print("各序列全天有值股票数（最少的 8 条）：")
    print(sc.sort_values("full_cov_有值股票数").head(8).to_string(index=False))

    # 市值、行业对账（Python 只核对输入，不参与计算）
    if meta["prev_date"]:
        cap = ddr.flow_cap(meta["prev_date"], meta["prev_date"]).iloc[0].reindex(codes).values.astype(float)
        mc = np.array(meta["prev_cap"], dtype=float)
        ok = np.isfinite(cap) & np.isfinite(mc)
        print("前一日流通市值对账：两边都有", int(ok.sum()), "最大相对差", float(np.nanmax(np.abs(mc[ok] / cap[ok] - 1))) if ok.any() else None)
    ind = ddr.industry(date).reindex(codes).values.astype(float)
    mi = np.array(meta["industry"], dtype=float)
    ok = np.isfinite(ind) & np.isfinite(mi)
    print("行业对账：两边都有", int(ok.sum()), "不一致", int((ind[ok] != mi[ok]).sum()), "Rust 侧行业为空", int((~np.isfinite(mi)).sum()))
    return meta, X, names, codes


if __name__ == "__main__":
    out = {}
    for p in sys.argv[1:]:
        out[p] = check(p)
