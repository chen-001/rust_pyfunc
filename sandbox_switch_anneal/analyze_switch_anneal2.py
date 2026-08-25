"""补充实验: ①方向D特征正交化后增量IC ②方向C perm对照(同值集合随机排列的恢复步数)"""
import json
import numpy as np
import pandas as pd
from scipy import stats

DATES = [20240104, 20240603, 20241008, 20260105, 20260717]
OUT_DIR = "/home/chenzongwei/rust_pyfunc/sandbox_switch_anneal/out"
from pure_ocean_breeze.jason.data.read_data import read_daily

rets = read_daily(ret=1)
amounts = pd.read_parquet(
    "/home/chenzongwei/database/daily_data/amounts.parquet"
).reindex(rets.index)


def to_suffix(code: str) -> str:
    if code.startswith(("60", "68")):
        return code + ".SH"
    if code.startswith(("00", "30")):
        return code + ".SZ"
    return code + ".BJ"


frames = {}
cs_all = []
for d in DATES:
    js = json.load(open(f"{OUT_DIR}/out_{d}.json"))
    df = pd.DataFrame(js["d_vals"], index=js["d_codes"], columns=js["d_names"])
    df.index = pd.Index([to_suffix(c) for c in df.index], name="code")
    frames[d] = df
    for c in js["cs"]:
        c["c_vec"] = np.array(c["c_vec"], dtype=np.float64)
        cs_all.append({"date": d, **c})

# ============ ① 方向D: 退火特征正交掉"简单统计"后的残差 IC ============
print("== ① 方向D 增量检验: 每天把特征正交掉 [r0, std, ac1, trend] 的线性投影, 再算残差IC ==")
CTRL = ["r0", "std", "ac1", "trend"]
targets = ["steps50", "steps70", "steps90", "half_life", "final_r", "inertia",
           "declines", "dr_std", "ladder", "max_jump"]
rows = []
for d in DATES:
    df = frames[d]
    t = pd.Timestamp(str(d))
    r = rets.loc[t].reindex(df.index)
    row = {}
    for f in targets + ["mean", "morn_aft_diff"]:
        v = df[f].astype(float)
        base = df[CTRL].astype(float)
        m = v.notna() & r.notna() & np.isfinite(v) & np.isfinite(base).all(axis=1)
        if m.sum() < 100:
            row[f] = np.nan
            continue
        X = np.column_stack([np.ones(m.sum()), base[m].values])
        beta, *_ = np.linalg.lstsq(X, v[m].values, rcond=None)
        resid = v[m].values - X @ beta
        row[f] = stats.spearmanr(resid, r[m].values).statistic
    rows.append(row)
res_df = pd.DataFrame(rows, index=[str(d) for d in DATES])
print(res_df.round(4).to_string())
print("\n汇总(正交掉简单统计后):")
for f in res_df.columns:
    s = res_df[f]
    print(
        f"  {f:10s} mean_ic={s.mean():+.4f} mean_abs={s.abs().mean():.4f} "
        f"icir={s.mean()/s.std():+.3f} pos={float((s>0).mean()):.2f}"
    )

# 对照: 原始 IC(未正交)
print("\n对照(未正交, 5天):")
for f in targets + ["mean", "morn_aft_diff"]:
    ics = []
    for d in DATES:
        df = frames[d]
        t = pd.Timestamp(str(d))
        r = rets.loc[t].reindex(df.index)
        v = df[f].astype(float)
        m = v.notna() & r.notna() & np.isfinite(v)
        ics.append(stats.spearmanr(v[m], r[m]).statistic if m.sum() > 100 else np.nan)
    s = pd.Series(ics, dtype=float)
    print(
        f"  {f:10s} mean_ic={s.mean():+.4f} mean_abs={s.abs().mean():.4f} "
        f"icir={s.mean()/s.std():+.3f} pos={float((s>0).mean()):.2f}"
    )

# ============ ② 方向C perm 对照 ============
# 退火: 与 Rust 相同单步(确定性等价), 只测到达 r=0.70 的步数
def anneal_steps70(x, m_max, seed):
    n = len(x)
    mean = float(np.mean(x))
    sigma2 = float(np.mean((x - mean) ** 2))
    if sigma2 <= 0:
        return None
    rng = np.random.default_rng(seed)
    guess = np.sort(x).astype(np.float64)
    s = float(np.sum((guess - x) ** 2))
    denom = 2.0 * sigma2 * n
    ct_base = sigma2
    for t in range(m_max):
        i, j = rng.integers(0, n, 2)
        if i == j:
            j = rng.integers(0, n)
        ds = 2.0 * (guess[i] - guess[j]) * (x[i] - x[j])
        ct = ct_base * (1.0 - t / (m_max - 1))
        if ds < 0.0 or ds < ct:
            s += ds
            guess[i], guess[j] = guess[j], guess[i]
        cr = 1.0 - s / denom
        if cr >= 0.70:
            return t
    return None


print("\n== ② 方向C perm对照: 真实 c 向量 vs 同值集合随机排列(到达 r=0.70 步数) ==")
sel = sorted(cs_all, key=lambda c: c["c_top5pct"], reverse=True)[:2] + sorted(
    cs_all, key=lambda c: c["c_top5pct"]
)[:2]
print(f"   {'date':9s} {'t':>4s} {'top5pct':>8s} {'真实steps70':>12s} {'perm x3 (同值集合)':>28s}")
for c in sel:
    vec = c["c_vec"]
    perms = []
    for k in range(3):
        p = vec[np.random.default_rng(1234 + k).permutation(len(vec))]
        perms.append(anneal_steps70(p, 500_000, 999 + k))
    print(
        f"   {c['date']} {c['t']:4d} {c['c_top5pct']:8.3f} {c['c_steps70']:12d} "
        f"{str(perms):>28s}"
    )
