"""敏感度实验：证明「中性化矩阵里少数股票换桶」足以解释 neu 阶段 MPROB 的残留偏差。

做法：取 observable_ratio_level 的 mean_smooth_5 / gap1（就是残留偏差最大的那条），
在某一交易日的信号行里，把相邻两个（跨桶边界的）股票的信号值互换，重算分组收益，
看 MPROB / SSM / annualized_return 各动多少。
"""

import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/chenzongwei/rust_pyfunc/tmp_mono_metric/verify_mprob")
sys.path.insert(0, "/home/chenzongwei/design_whatever")

import rust_pyfunc as rp  # noqa: E402
from ref_mprob import compute_mprob, erf_as  # noqa: E402

BASE = "/tmp/mprob_smoke_on_tail_v4"
SRC = "/hdd/user_home_unsafe/chenzongwei/factor_data/mprob_smoke_on"
SRCNAME = "microcapm_observable_ratio_level_capm_residual_zscore_mean_mean"
STAT = "mean_smooth_5"
_ORDER = ["smooth_1"] + [f"{st}_smooth_{w}" for w in (5, 10, 20) for st in ("mean", "max", "min", "std")]
SLOT = _ORDER.index(STAT)
GAP = 1
STYLE = "/ssd_data/data/vars"
WINDOWS = [5, 10, 20]
PORTF = 10

meta = json.load(open(f"{BASE}/meta/input_fingerprints.json"))
cfg = json.load(open(f"{BASE}/meta/tail_v4_config.json"))
BS = int(str(cfg["backtest_start_date"]).replace("-", ""))
dates = np.load(f"{BASE}/meta/dates.npy").astype(np.int32)
stocks = np.load(f"{BASE}/meta/stocks.npy", allow_pickle=True)
restrict = np.load(meta["restrict"]["path"]).astype(np.float64)
index_ret = np.load(meta["index_ret"]["path"]).astype(np.float64)
ret = np.load(meta[f"ret_gap{GAP}"]["path"]).astype(np.float64)
T = ret.shape[0]

from design_whatever.tail_v4 import _load_industry_matrix
industry = _load_industry_matrix(dates, [str(s) for s in stocks.tolist()], industry_data_path=None)


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


def compute_ssm(gr):
    n = len(gr[0])
    r = [sum(c) / n for c in gr]
    if r[9] < r[0]:
        r = r[::-1]

    def seg(i, j):
        s = r[i - 1:j]
        den = sum(abs(s[k] - s[k - 1]) for k in range(1, len(s)))
        return 0.0 if den <= 0 else (s[-1] - s[0]) / den

    return min([seg(1, 10), seg(1, 5), seg(6, 10), seg(1, 4), seg(7, 10)])


def reproduce(f):
    eff = [t for t in range(1, T) if dates[t] > BS and np.isfinite(f[t - 1]).any()]
    gr = np.zeros((PORTF, len(eff)))
    held = eff[0] - 1
    for lt, t in enumerate(eff):
        if lt % GAP == 0:
            held = t - 1
        sig, rr = f[held], ret[t]
        ok = (np.isfinite(sig) & np.isfinite(rr)
              & np.isfinite(restrict[held]) & (restrict[held] == 0.0))
        fs, fr = sig[ok], rr[ok]
        k = fs.size
        if k < PORTF:
            continue
        bucket = np.minimum((avg_ranks(fs) / k * PORTF).astype(np.int64), PORTF - 1)
        for b in range(PORTF):
            m = bucket == b
            gr[b, lt] = fr[m].mean() if m.any() else 0.0
    li, si = (0, PORTF - 1) if gr[0].sum() > gr[PORTF - 1].sum() else (PORTF - 1, 0)
    ls = gr[li] - gr[si]
    g = gr.tolist()
    return dict(ann=float(np.nanmean(ls) * 250.0), ssm=compute_ssm(g),
                mprob=compute_mprob(g, PORTF, erf_as))


df = pd.read_parquet(f"{SRC}/{SRCNAME}.parquet")
if "date" in df.columns:
    df = df.set_index("date")
df.index = pd.to_datetime(df.index)
ti = pd.to_datetime(dates.astype(str), format="%Y%m%d")
df = df.reindex(index=ti, columns=pd.Index(stocks.tolist(), dtype=object))
arr = df.to_numpy(dtype=np.float32, copy=True)
arr[~np.isfinite(arr)] = np.nan
rolled = np.asarray(rp.tail_v5_rank_fill_roll_block_f32(arr, restrict.astype(np.float32),
                                                        [int(w) for w in WINDOWS]), dtype=np.float32)
neu = np.asarray(rp.neutralize_std_block_py(rolled, industry, restrict.astype(np.float32), STYLE,
                                            [int(d) for d in dates], [str(s) for s in stocks.tolist()],
                                            False), dtype=np.float32)
mat = neu[:, :, SLOT]

base = reproduce(mat)
print(f"基线 {SRCNAME}_{STAT} gap{GAP}: MPROB={base['mprob']!r} SSM={base['ssm']!r} ann={base['ann']!r}")
parq = pd.read_parquet(f"{BASE}/metrics/summary_neu_gap{GAP}_candidates.parquet")
row = parq[parq.factor_name == f"{SRCNAME}_{STAT}"].iloc[0]
print(f"parquet:                    MPROB={float(row.MPROB)!r} SSM={float(row.SSM)!r} ann={float(row.annualized_return)!r}")
print(f"残留偏差: dMPROB={base['mprob'] - float(row.MPROB):+.3e} dSSM={base['ssm'] - float(row.SSM):+.3e}"
      f" dann={base['ann'] - float(row.annualized_return):+.3e}")

rng = np.random.default_rng(11)
eff = [t for t in range(1, T) if dates[t] > BS and np.isfinite(mat[t - 1]).any()]
print("\n互换相邻两个跨桶边界股票的信号值（每个交易日只换一对）：")
dm_list, ds_list, da_list = [], [], []
for trial in range(15):
    lt = int(rng.integers(0, len(eff)))
    held = eff[lt] - 1
    sig = mat[held]
    ok = (np.isfinite(sig) & np.isfinite(restrict[held]) & (restrict[held] == 0.0))
    idx = np.nonzero(ok)[0]
    k = idx.size
    r = avg_ranks(sig[idx])
    b = np.minimum((r / k * PORTF).astype(np.int64), PORTF - 1)
    # 找一个桶边界附近的下标对
    cand = np.nonzero(np.diff(b) > 0)[0]
    if cand.size == 0:
        continue
    p = int(cand[rng.integers(0, cand.size)])
    i0, i1 = idx[p], idx[p + 1]
    m2 = mat.copy()
    m2[held, i0], m2[held, i1] = m2[held, i1], m2[held, i0]
    got = reproduce(m2)
    dm = got["mprob"] - base["mprob"]
    ds = got["ssm"] - base["ssm"]
    da = got["ann"] - base["ann"]
    dm_list.append(dm); ds_list.append(ds); da_list.append(da)
    print(f"  第 {trial+1:2d} 次（t={dates[held]}，桶 {b[p]}→{b[p]+1} 边界）:"
          f" dMPROB={dm:+.3e}  dSSM={ds:+.3e}  dann={da:+.3e}")

dm = np.abs(np.array(dm_list)); ds = np.abs(np.array(ds_list)); da = np.abs(np.array(da_list))
print(f"\n单股换桶的影响：|dMPROB| 中位 {np.median(dm):.3e} 最大 {dm.max():.3e}；"
      f"|dSSM| 中位 {np.median(ds):.3e} 最大 {ds.max():.3e}；"
      f"|dann| 中位 {np.median(da):.3e} 最大 {da.max():.3e}")
print(f"对比实测残留偏差：|dMPROB|={abs(base['mprob'] - float(row.MPROB)):.3e}"
      f" |dSSM|={abs(base['ssm'] - float(row.SSM)):.3e}")
