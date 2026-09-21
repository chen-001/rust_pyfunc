"""MPROB 端到端校验：用引擎自己的产物复现分组收益，再与 parquet 的 MPROB / SSM 列比对。

做法照 tmp_mono_metric/verifier/verify_ssm.md 第 3 节：
  引擎导出的 raw parquet（源因子值） + meta 轴 + 回测输入缓存
  → rp.tail_v5_rank_fill_roll_block_f32（rank/填充/rolling）  = rolled 阶段
  → rp.neutralize_std_block_py（标准中性化）                    = neu 阶段
  → 按 Rust 源码的分组规则重写分组收益
  → 用 ref_mprob.compute_mprob（A&S erf 版，与 Rust 同式）复算 MPROB / SSM

用法：
  python e2e_mprob.py <temp_root> <src_parquet_dir> [stages] [gaps]
    stages: 逗号分隔，默认 rolled,neu
    gaps  : 逗号分隔，默认 1,5
"""

import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/chenzongwei/rust_pyfunc/tmp_mono_metric/verify_mprob")
sys.path.insert(0, "/home/chenzongwei/design_whatever")

import rust_pyfunc as rp  # noqa: E402
from ref_mprob import compute_mprob, erf_as  # noqa: E402

STYLE = "/ssd_data/data/vars"
PORTF = 10
WINDOWS = [5, 10, 20]


def slot_order():
    o = ["smooth_1"]
    for w in WINDOWS:
        for st in ("mean", "max", "min", "std"):
            o.append(f"{st}_smooth_{w}")
    return o


def compute_ssm(gr):
    if len(gr) != 10:
        return float("nan")
    n = len(gr[0])
    if n == 0 or any(len(c) != n for c in gr):
        return float("nan")
    r = [sum(c) / n for c in gr]
    if r[9] < r[0]:
        r = r[::-1]

    def seg(i, j):
        s = r[i - 1:j]
        den = sum(abs(s[k] - s[k - 1]) for k in range(1, len(s)))
        if den <= 0:
            return 0.0
        return (s[-1] - s[0]) / den

    return min([seg(1, 10), seg(1, 5), seg(6, 10), seg(1, 4), seg(7, 10)])


def avg_ranks(v):
    order = np.argsort(v, kind="stable")
    ranks = np.empty(len(v), dtype=np.float64)
    sv = v[order]
    i = 0
    while i < len(sv):
        j = i + 1
        while j < len(sv) and sv[j] == sv[i]:
            j += 1
        ranks[order[i:j]] = (i + 1 + j) / 2.0
        i = j
    return ranks


def ordinal_ranks(v):
    order = np.argsort(v, kind="stable")
    r = np.empty(len(v), dtype=np.int64)
    r[order] = np.arange(len(v))
    return r


def ic_of(x, y):
    xx, yy = ordinal_ranks(x), ordinal_ranks(y)
    n = float(len(x))
    d = xx - yy
    return 1.0 - 6.0 * float((d * d).sum()) / (n * (n * n - 1.0))


def ic_mean(f, ret, rsu, gap, dates, restrict, backtest_start):
    """引擎逐字定义的 Spearman IC（IC 只依赖信号的序，用来判断中性化矩阵是否逐位一致）。"""
    T = ret.shape[0]
    eff = [t for t in range(1, T) if dates[t] > backtest_start and np.isfinite(f[t - 1]).any()]
    held = eff[0] - 1
    vals = []
    for lt, t in enumerate(eff):
        if lt % gap == 0:
            held = t - 1
        sig = f[held]
        ok = (np.isfinite(sig) & np.isfinite(ret[t])
              & np.isfinite(restrict[held]) & (restrict[held] == 0.0))
        if (lt + 1) % gap == 0:
            vals.append(ic_of(rsu[t][ok], sig[ok]))
    return float(np.nanmean(vals))


def main():
    base, src = sys.argv[1], sys.argv[2]
    stages = (sys.argv[3].split(",") if len(sys.argv) > 3 else ["rolled", "neu"])
    gaps = [int(x) for x in sys.argv[4].split(",")] if len(sys.argv) > 4 else [1, 5]

    meta = json.load(open(f"{base}/meta/input_fingerprints.json"))
    cfg = json.load(open(f"{base}/meta/tail_v4_config.json"))
    backtest_start = int(str(cfg["backtest_start_date"]).replace("-", ""))
    dates = np.load(f"{base}/meta/dates.npy").astype(np.int32)
    stocks = np.load(f"{base}/meta/stocks.npy", allow_pickle=True)
    restrict = np.load(meta["restrict"]["path"]).astype(np.float64)
    index_ret = np.load(meta["index_ret"]["path"]).astype(np.float64)

    from design_whatever.tail_v4 import _load_industry_matrix
    print("加载行业码矩阵...", flush=True)
    industry = _load_industry_matrix(dates, [str(s) for s in stocks.tolist()], industry_data_path=None)
    print("industry", industry.shape, flush=True)

    order = slot_order()

    def reproduce(f, ret, gap):
        T = ret.shape[0]
        eff = [t for t in range(1, T) if dates[t] > backtest_start and np.isfinite(f[t - 1]).any()]
        oc = ((np.isfinite(restrict)) & (restrict == 0.0)).sum(axis=1)
        gr = np.zeros((PORTF, len(eff)))
        ratio = np.full(len(eff), np.nan)
        held = eff[0] - 1
        for lt, t in enumerate(eff):
            if lt % gap == 0:
                held = t - 1
            sig, rr = f[held], ret[t]
            ok = (np.isfinite(sig) & np.isfinite(rr)
                  & np.isfinite(restrict[held]) & (restrict[held] == 0.0))
            fs, fr = sig[ok], rr[ok]
            k = fs.size
            if k < PORTF:
                continue
            if oc[t - 1] > 0:
                ratio[lt] = k / oc[t - 1]
            bucket = np.minimum((avg_ranks(fs) / k * PORTF).astype(np.int64), PORTF - 1)
            for b in range(PORTF):
                m = bucket == b
                gr[b, lt] = fr[m].mean() if m.any() else 0.0
        li, si = (0, PORTF - 1) if gr[0].sum() > gr[PORTF - 1].sum() else (PORTF - 1, 0)
        ls = gr[li] - gr[si]
        hedge = np.array([gr[li][lt] - index_ret[t] for lt, t in enumerate(eff)])
        g = gr.tolist()
        return dict(date_size=len(eff), ratio_mean=float(np.nanmean(ratio)),
                    annualized_return=float(np.nanmean(ls) * 250.0),
                    hedge_annualized_return=float(np.nanmean(hedge) * 250.0),
                    ssm=compute_ssm(g), mprob=compute_mprob(g, PORTF, erf_as))

    parqs = {}
    for st in stages:
        for g in gaps:
            p = f"{base}/metrics/summary_{st}_gap{g}_candidates.parquet"
            try:
                parqs[(st, g)] = pd.read_parquet(p)
            except Exception as e:
                print(f"跳过 {st} gap{g}: {e}", flush=True)

    # 源因子 = parquet 里的 factor_name 去掉已知 slot 后缀
    names = set()
    for d in parqs.values():
        names.update(d.factor_name.tolist())
    sources = sorted({n[: -(len(s) + 1)] for n in names for s in order if n.endswith("_" + s)})
    print(f"源因子 {len(sources)} 个（parquet 里的非 fold 面）", flush=True)

    res = []
    for source in sources:
        v = None
        try:
            v = pd.read_parquet(f"{src}/{source}.parquet")
        except Exception:
            print(f"  跳过 {source}（源 parquet 找不到）", flush=True)
            continue
        if "date" in v.columns:
            v = v.set_index("date")
        if not isinstance(v.index, pd.DatetimeIndex):
            v.index = pd.to_datetime(v.index)
        tpl_idx = pd.to_datetime(dates.astype(str), format="%Y%m%d")
        tpl_cols = pd.Index(stocks.tolist(), dtype=object)
        if not v.index.equals(tpl_idx) or not v.columns.equals(tpl_cols):
            v = v.reindex(index=tpl_idx, columns=tpl_cols)
        arr = v.to_numpy(dtype=np.float32, copy=True)
        arr[~np.isfinite(arr)] = np.nan
        rolled = np.asarray(rp.tail_v5_rank_fill_roll_block_f32(
            arr, restrict.astype(np.float32), [int(w) for w in WINDOWS]), dtype=np.float32)
        mats = {"rolled": rolled}
        if "neu" in stages:
            print(f"  {source}: 中性化中...", flush=True)
            mats["neu"] = np.asarray(rp.neutralize_std_block_py(
                rolled, industry, restrict.astype(np.float32), STYLE,
                [int(d) for d in dates], [str(s) for s in stocks.tolist()], False), dtype=np.float32)
        for (st, gap), parq in parqs.items():
            ret = np.load(meta[f"ret_gap{gap}"]["path"]).astype(np.float64)
            rsu = np.load(meta[f"ret_sum_gap{gap}"]["path"]).astype(np.float64)
            for slot, stat in enumerate(order):
                name = f"{source}_{stat}"
                row = parq[parq.factor_name == name]
                if row.empty:
                    continue
                row = row.iloc[0]
                got = reproduce(mats[st][:, :, slot], ret, gap)
                got["ic"] = ic_mean(mats[st][:, :, slot], ret, rsu, gap, dates, restrict, backtest_start)
                res.append((st, gap, name, got, row))
                dm = abs(got["mprob"] - float(row.MPROB))
                dsv = abs(got["ssm"] - float(row.SSM))
                da = abs(got["annualized_return"] - float(row.annualized_return))
                dic = abs(got["ic"] - float(row.IC_mean))
                print(f"  {st:6s} g{gap} {stat:14s} |dMPROB|={dm:.3e} |dSSM|={dsv:.3e}"
                      f" |dRet|={da:.3e} |dIC|={dic:.3e}", flush=True)

    print("=" * 110)
    if not res:
        print("没有任何可比对组合")
        return 1
    for st in stages:
        for g in gaps:
            sub = [(n, gg, r) for (s, gg2, n, gg, r) in res if s == st and gg2 == g]
            if not sub:
                continue
            dm = np.array([gg["mprob"] - float(r.MPROB) for _, gg, r in sub])
            ds = np.array([gg["ssm"] - float(r.SSM) for _, gg, r in sub])
            da = np.array([gg["annualized_return"] - float(r.annualized_return) for _, gg, r in sub])
            di = np.array([gg["ic"] - float(r.IC_mean) for _, gg, r in sub])
            bitm = sum(1 for _, gg, r in sub if gg["mprob"].hex() == float(r.MPROB).hex())
            bits = sum(1 for _, gg, r in sub if gg["ssm"].hex() == float(r.SSM).hex())
            biti = sum(1 for _, gg, r in sub if gg["ic"].hex() == float(r.IC_mean).hex())
            print(f"[{st} gap{g}] n={len(sub)}")
            print(f"   MPROB 最大|偏差| = {np.abs(dm).max():.3e}  逐位相同 {bitm}/{len(sub)}")
            print(f"   SSM   最大|偏差| = {np.abs(ds).max():.3e}  逐位相同 {bits}/{len(sub)}")
            print(f"   ann_ret 最大|偏差| = {np.abs(da).max():.3e}")
            print(f"   IC_mean 最大|偏差| = {np.abs(di).max():.3e}  逐位相同 {biti}/{len(sub)}   <- 信号序探针")
            bad = [(n, gg["mprob"], float(r.MPROB), gg["ic"] - float(r.IC_mean))
                   for n, gg, r in sub if abs(gg["mprob"] - float(r.MPROB)) > 1e-12]
            if bad:
                print(f"   MPROB 偏差 >1e-12 的 {len(bad)} 条（括号里是同一条的 IC 偏差）：")
                for n, a, b, dic in bad[:10]:
                    print(f"     {n}: py={a!r} rust={b!r}  dIC={dic:+.3e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
