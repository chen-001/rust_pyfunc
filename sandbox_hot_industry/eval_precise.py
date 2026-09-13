"""同热点股票池「行业拓展」IC 评估
输入: sandbox_hot_industry/prec_{date}.json (5 天)
输出: results_ic_precise.json / 汇总表打印
核心比较: 行业内识别因子 vs 行业β基线 (原始IC / 行业内中性化IC / 行业残差IC)
"""
import json
import numpy as np
import pandas as pd

from pure_ocean_breeze.jason.data.read_data import read_daily

DATES = [20240104, 20240603, 20241008, 20260105, 20260710]
OUTDIR = "/home/chenzongwei/rust_pyfunc/sandbox_hot_industry"

r1 = read_daily(ret=1)
r5 = read_daily(ret=5)
r1.index = pd.to_datetime(r1.index)
r5.index = pd.to_datetime(r5.index)


def code_fmt(c):
    if c.startswith("6"):
        return c + ".SH"
    if c.startswith(("4", "8", "92")):
        return c + ".BJ"
    return c + ".SZ"


def next_day(dt, k=1):
    idx = r1.index
    pos = idx.searchsorted(dt)
    return idx[min(pos + k, len(idx) - 1)]


def spearman(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 50:
        return np.nan
    x = x[m]
    y = y[m]
    rx = pd.Series(x).rank().values
    ry = pd.Series(y).rank().values
    return float(np.corrcoef(rx, ry)[0, 1])


def ind_neutralize(x, inds):
    """行业内去均值+z化 — 等价于对行业哑变量截面OLS(含截距)的残差(单调变换不影响Spearman秩)"""
    out = np.full(len(x), np.nan)
    for i in np.unique(inds[inds > 0]):
        m = (inds == i) & np.isfinite(x)
        if m.sum() >= 5:
            mu = x[m].mean()
            sd = x[m].std()
            if sd > 1e-12:
                out[m] = (x[m] - mu) / sd
    return out


def build_day(date):
    d = json.load(open(f"{OUTDIR}/prec_{date}.json"))
    rows = d["stocks"]
    df = pd.DataFrame(
        {
            "code": [s["code"] for s in rows],
            "ind": [s["ind"] for s in rows],
            "hot_cnt": [s["hot_cnt"] for s in rows],
            "cold_cnt": [s["cold_cnt"] for s in rows],
            "rk_buy_h": [s["rk_buy_h"] for s in rows],
            "rk_vol_h": [s["rk_vol_h"] for s in rows],
            "rk_ba_h": [s["rk_ba_h"] for s in rows],
            "z_buy_h": [s["z_buy_h"] for s in rows],
            "z_vol_h": [s["z_vol_h"] for s in rows],
            "z_ba_h": [s["z_ba_h"] for s in rows],
            "rk_buy_p": [s.get("rk_buy_p", float("nan")) for s in rows],
            "rk_vol_p": [s.get("rk_vol_p", float("nan")) for s in rows],
            "rk_ba_p": [s.get("rk_ba_p", float("nan")) for s in rows],
        }
    )
    df["hot_peers"] = [s["hot_peers"] for s in rows]
    df["code_fmt"] = df["code"].map(code_fmt)

    dt = pd.Timestamp(f"{date // 10000}-{(date // 100) % 100:02d}-{date % 100:02d}")
    nx = next_day(dt)
    nx5 = next_day(nx, 4)
    # r1.loc[t] = close[t]/close[t-1]-1 (当日收益, 收盘可知)
    ret_D = r1.loc[dt]  # D 日收益（因子可用信息）
    ret_N = r1.loc[nx]  # 次日收益（目标）
    ret_D5 = r5.loc[dt]  # D 日 5 日收益
    ret_N5 = r5.loc[nx5]  # 5 日目标（D→D+5）
    if dt >= r1.index[-1]:  # 末日无次日收益 → 目标置 NaN
        ret_N = pd.Series(np.nan, index=r1.columns)
        ret_N5 = pd.Series(np.nan, index=r5.columns)

    df["ret_D"] = df["code_fmt"].map(ret_D).values
    df["ret_N"] = df["code_fmt"].map(ret_N).values
    df["ret_D5"] = df["code_fmt"].map(ret_D5).values
    df["ret_N5"] = df["code_fmt"].map(ret_N5).values

    # 行业平均收益（行业β 基线）
    ind_ret_today = df[df["ind"] > 0].groupby("ind")["ret_D"].mean()
    ind_ret_tmt = df[df["ind"] > 0].groupby("ind")["ret_N"].mean()
    ind_hot_mean = df[df["ind"] > 0].groupby("ind")["hot_cnt"].mean()
    df["ind_mret_today"] = df["ind"].map(ind_ret_today)
    df["ind_mret_tmt"] = df["ind"].map(ind_ret_tmt)

    ind_by_code = dict(zip(df["code"], df["ind"]))
    ind_all_rt = {v: float(np.nanmean(df["ret_D"][df["ind"] == v].values))
                  for v in df["ind"].unique() if v > 0}

    # 同伴聚合（热点组 top-10）
    peer_rt = df.set_index("code_fmt")["ret_D"]
    peer_rt_t = df.set_index("code_fmt")["ret_N"]
    N = len(df)
    cols = {
        "peer_ind_mean": np.full(N, np.nan),
        "peer_all_mean": np.full(N, np.nan),
        "peer_cross_mean": np.full(N, np.nan),
        "peer_ind_exc": np.full(N, np.nan),
        "peer_all_exc": np.full(N, np.nan),
        "peer_ind_w2": np.full(N, np.nan),
        "peer_ind_w5": np.full(N, np.nan),
        "peer_ind_only": np.full(N, np.nan),
        "peer_ind_sw": np.full(N, np.nan),
        "peer_cross_sw": np.full(N, np.nan),
        "peer_ind_exc_sw": np.full(N, np.nan),
        "peer_all_exc_sw": np.full(N, np.nan),
    }
    for i in range(N):
        peers = df["hot_peers"].iloc[i]
        if not peers:
            continue
        pf = [code_fmt(p["code"]) for p in peers]
        pind = [ind_by_code.get(p["code"], 0) for p in peers]
        cnts = np.array([p["cnt"] for p in peers], dtype=float)
        rt = np.array([peer_rt.get(c, np.nan) for c in pf], dtype=float)
        rt_t = np.array([peer_rt_t.get(c, np.nan) for c in pf], dtype=float)
        my_ind = df["ind"].iloc[i]
        same = np.array([v == my_ind for v in pind], dtype=bool)
        exc = np.array(
            [v - ind_all_rt.get(j, np.nan) if np.isfinite(v) else np.nan
             for v, j in zip(rt, pind)], dtype=float)
        exc_t = np.array(
            [v - ind_ret_tmt.get(j, np.nan) if np.isfinite(v) else np.nan
             for v, j in zip(rt_t, pind)], dtype=float)
        if same.sum() > 0:
            cols["peer_ind_mean"][i] = np.nanmean(rt[same])
            cols["peer_ind_exc"][i] = np.nanmean(exc[same])
            cols["peer_ind_sw"][i] = np.nanmean(rt_t[same])
            cols["peer_ind_exc_sw"][i] = np.nanmean(exc_t[same])
            w = cnts[same]
            cols["peer_ind_w2"][i] = np.nansum(rt[same] * w * 2) / np.nansum(w * 2)
            cols["peer_ind_w5"][i] = np.nansum(rt[same] * w * 5) / np.nansum(w * 5)
            cols["peer_ind_only"][i] = np.nansum(rt[same] * w) / np.nansum(w)
        if (~same).sum() > 0:
            cols["peer_cross_mean"][i] = np.nanmean(rt[~same])
            cols["peer_cross_sw"][i] = np.nanmean(rt_t[~same])
        cols["peer_all_mean"][i] = np.nanmean(rt)
        cols["peer_all_exc"][i] = np.nanmean(exc)
        cols["peer_all_exc_sw"][i] = np.nanmean(exc_t)

    for k, v in cols.items():
        df[k] = v

    # 同行业随机 10 只（不看同热点）
    rng = np.random.default_rng(date)
    rand_rt = np.full(N, np.nan)
    rand_rt_t = np.full(N, np.nan)
    for i in range(N):
        mi = df["ind"].iloc[i]
        if mi > 0:
            cand = df.index[df["ind"] == mi].tolist()
            if len(cand) >= 10:
                pick = rng.choice(cand, size=10, replace=False)
                rand_rt[i] = np.nanmean(df["ret_D"].iloc[pick].values)
                rand_rt_t[i] = np.nanmean(df["ret_N"].iloc[pick].values)
    df["rand_ind_mean"] = rand_rt
    df["rand_ind_sw"] = rand_rt_t

    # 行业内相对位置因子（已按行业构造 → 天然行业中性）
    df["ind_freq_ratio"] = df["hot_cnt"] / df["ind"].map(ind_hot_mean)
    df["ind_rank_buy"] = df["rk_buy_h"]
    df["ind_rank_vol"] = df["rk_vol_h"]
    df["ind_rank_ba"] = df["rk_ba_h"]
    df["ind_z_buy"] = df["z_buy_h"]
    df["ind_z_vol"] = df["z_vol_h"]
    df["ind_z_ba"] = df["z_ba_h"]
    # 全池(不分行业)排名 = 生产 f07/f28 原生口径
    df["pool_rank_buy"] = df["rk_buy_p"]
    df["pool_rank_vol"] = df["rk_vol_p"]
    df["pool_rank_ba"] = df["rk_ba_p"]

    # 共现top10 中同行业占比
    n_same = []
    for i in range(N):
        peers = df["hot_peers"].iloc[i]
        if not peers:
            continue
        pind = [ind_by_code.get(p["code"], 0) for p in peers]
        n_same.append(np.mean([v == df["ind"].iloc[i] for v in pind]))
    df["same_share"] = np.nan
    df.loc[df.index[: len(n_same)], "same_share"] = n_same

    return df, d["pool_hot"], d["pool_cold"]


FACTORS = [
    ("hot_cnt", "对照: 热点入选次数(未中性化)"),
    ("ind_mret_today", "基线: 行业β(D日行业平均收益)"),
    ("peer_ind_mean", "同伴: 同热点top10中同行业同伴当日收益均值"),
    ("peer_all_mean", "同伴: 全部top10同伴当日收益均值"),
    ("peer_cross_mean", "同伴: 跨行业同伴当日收益均值"),
    ("rand_ind_mean", "基线: 同行业随机10只当日收益均值"),
    ("peer_ind_exc", "超额: 同行业同伴收益-该行业平均收益"),
    ("peer_all_exc", "超额: 全部同伴收益-各自行业平均收益"),
    ("peer_ind_w2", "加权: 同行业同伴cooc权重x2"),
    ("peer_ind_w5", "加权: 同行业同伴cooc权重x5"),
    ("peer_ind_only", "加权: 同行业同伴cooc加权(排他)"),
    ("ind_freq_ratio", "行业内位置: 入选频率/行业均值"),
    ("ind_rank_buy", "行业内位置: 主买占比组内排名(行业内)"),
    ("ind_rank_vol", "行业内位置: 成交量组内排名(行业内)"),
    ("ind_rank_ba", "行业内位置: 盘口差组内排名(行业内)"),
    ("ind_z_buy", "行业内位置: 主买占比组内z(行业内)"),
    ("ind_z_ba", "行业内位置: 盘口差组内z(行业内)"),
    ("pool_rank_buy", "对照: 全池主买占比排名(f07原生口径)"),
    ("pool_rank_vol", "对照: 全池成交量排名(f28原生口径)"),
    ("pool_rank_ba", "对照: 全池盘口差排名(f20原生口径)"),
]

SW_FACTORS = [
    ("peer_ind_sw", "同期对照: 同行业同伴次日收益均值"),
    ("peer_cross_sw", "同期对照: 跨行业同伴次日收益均值"),
    ("rand_ind_sw", "同期对照: 同行业随机10只次日收益均值"),
    ("peer_ind_exc_sw", "同期对照: 同行业同伴次日-行业次日均值"),
    ("peer_all_exc_sw", "同期对照: 全部同伴次日超额"),
]

results = {}
for date in DATES:
    df, ph, pc = build_day(date)
    results[date] = {"df": df, "pool_hot": ph, "pool_cold": pc}
    print(
        f"date {date}: n={len(df)} pool_hot(mean={ph['mean_size']:.0f},max={ph['max_size']},inc={ph['total_inclusions']}) "
        f"pool_cold(mean={pc['mean_size']:.0f},max={pc['max_size']}) "
        f"共现top10同行业占比={np.nanmean(df['same_share']):.3f} "
        f"有同伴股票占比={df['peer_ind_mean'].notna().mean():.3f}"
    )


def ic_table(target_key="ret_N", factors=None, neutralize=False):
    factors = factors or FACTORS
    rows = []
    for fac, label in factors:
        ics = []
        ics_n = []
        cnt = []
        for date in DATES:
            df = results[date]["df"]
            x = df[fac].values.astype(float)
            y = df[target_key].values.astype(float)
            ics.append(spearman(x, y))
            xn = ind_neutralize(x, df["ind"].values)
            ics_n.append(spearman(xn, y))
            cnt.append(int(np.isfinite(x).sum()))
        ics = np.array(ics, dtype=float)
        ics_n = np.array(ics_n, dtype=float)
        rows.append(
            {
                "factor": fac,
                "label": label,
                "n_per_day": int(np.median(cnt)),
                "ic_mean": float(np.nanmean(ics)),
                "ic_abs_mean": float(np.nanmean(np.abs(ics))),
                "icir": float(np.nanmean(ics) / (np.nanstd(ics) + 1e-12)),
                "pos_frac": float(np.mean(ics > 0)),
                "ic_mean_neu": float(np.nanmean(ics_n)),
                "ic_abs_mean_neu": float(np.nanmean(np.abs(ics_n))),
                "icir_neu": float(np.nanmean(ics_n) / (np.nanstd(ics_n) + 1e-12)),
                "pos_frac_neu": float(np.mean(ics_n > 0)),
                "per_day": [round(float(v), 4) if np.isfinite(v) else None for v in ics],
                "per_day_neu": [round(float(v), 4) if np.isfinite(v) else None for v in ics_n],
            }
        )
    return rows


main_table = ic_table("ret_N", FACTORS)
sw_table = ic_table("ret_N", SW_FACTORS)
r5_table = ic_table("ret_N5", FACTORS)

# ---------- 诊断: 对当日收益 ret_D 横截面残差化后的 IC ----------
def ic_table_resid(target_key="ret_N", factors=None):
    rows = []
    for fac, label in (factors or FACTORS):
        ics = []
        for date in DATES:
            df = results[date]["df"]
            x = df[fac].values.astype(float)
            y = df[target_key].values.astype(float)
            rt = df["ret_D"].values.astype(float)
            m = np.isfinite(x) & np.isfinite(y) & np.isfinite(rt)
            if m.sum() < 100:
                ics.append(np.nan)
                continue
            X = np.stack([np.ones(m.sum()), rt[m]], axis=1)
            beta, *_ = np.linalg.lstsq(X, x[m], rcond=None)
            resid = x[m] - X @ beta
            ics.append(spearman(resid, y[m]))
        ics = np.array(ics, dtype=float)
        rows.append({"factor": fac, "label": label, "ic_mean_resid_retD": float(np.nanmean(ics)),
                     "per_day": [round(float(v), 4) if np.isfinite(v) else None for v in ics]})
    return rows


resid_table = ic_table_resid("ret_N", FACTORS)

# ---------- 因子间 spearman 相关（各日均值）----------
corr_rows = {}
key_facs = ["ind_mret_today", "hot_cnt", "ind_freq_ratio", "ind_rank_buy", "ind_rank_vol",
            "peer_ind_mean", "peer_ind_exc", "peer_all_exc", "rand_ind_mean", "ret_D"]
for a in key_facs:
    for b in key_facs:
        if a >= b:
            continue
        cs = []
        for date in DATES:
            df = results[date]["df"]
            cs.append(spearman(df[a].values.astype(float), df[b].values.astype(float)))
        corr_rows[f"{a}~{b}"] = float(np.nanmean(cs))

# ---------- 显式行业哑变量 OLS 残差 IC（以 ind_freq_ratio / ind_rank_buy 为例）----------
def ols_dummy_resid_ic(fac, target_key="ret_N"):
    ics = []
    for date in DATES:
        df = results[date]["df"]
        x = df[fac].values.astype(float)
        y = df[target_key].values.astype(float)
        inds = df["ind"].values
        m = np.isfinite(x) & np.isfinite(y) & (inds > 0)
        if m.sum() < 100:
            ics.append(np.nan)
            continue
        X = np.zeros((m.sum(), 32))
        X[:, 0] = 1.0
        for j, i in enumerate(np.where(m)[0]):
            if inds[i] <= 31:
                X[j, inds[i]] = 1.0
        beta, *_ = np.linalg.lstsq(X, x[m], rcond=None)
        resid = x[m] - X @ beta
        ics.append(spearman(resid, y[m]))
    ics = np.array(ics, dtype=float)
    return list(ics)


ols_resid = {
    "ind_freq_ratio": ols_dummy_resid_ic("ind_freq_ratio"),
    "ind_rank_buy": ols_dummy_resid_ic("ind_rank_buy"),
    "ind_rank_vol": ols_dummy_resid_ic("ind_rank_vol"),
    "peer_ind_exc": ols_dummy_resid_ic("peer_ind_exc"),
    "rand_ind_mean": ols_dummy_resid_ic("rand_ind_mean"),
}

out = {
    "dates": DATES,
    "main_r1": main_table,
    "main_r5": r5_table,
    "same_window": sw_table,
    "resid_retD": resid_table,
    "corr": corr_rows,
    "ols_dummy_resid_ic": ols_resid,
}
with open(f"{OUTDIR}/results_ic_precise.json", "w") as f:
    json.dump(out, f, indent=1, ensure_ascii=False, default=str)

pd.set_option("display.width", 260)
pd.set_option("display.max_columns", 30)
t = pd.DataFrame(main_table)
print("\n===== 主表: 因子(D日收盘可知) → 目标=D+1 收益 =====")
print(t[["factor", "n_per_day", "ic_mean", "ic_abs_mean", "icir", "pos_frac",
         "ic_mean_neu", "icir_neu", "pos_frac_neu"]].to_string(index=False))
print("\n===== 同期对照(同伴次日收益, 旧agent口径) =====")
t2 = pd.DataFrame(sw_table)
print(t2[["factor", "ic_mean", "ic_abs_mean", "icir", "pos_frac"]].to_string(index=False))
print("\n===== 5日目标(r5) =====")
t3 = pd.DataFrame(r5_table)
print(t3[["factor", "ic_mean", "ic_abs_mean", "icir", "pos_frac"]].to_string(index=False))
print("\n===== 控制当日收益(ret_D)残差化后 IC =====")
t4 = pd.DataFrame(resid_table)
print(t4[["factor", "ic_mean_resid_retD"]].to_string(index=False))
print("\n===== 显式行业哑变量 OLS 残差 IC (per-day, 应与行业内去均值一致) =====")
for k, v in ols_resid.items():
    print(" ", k, [round(x, 4) if np.isfinite(x) else None for x in v])
print("\n===== 关键因子间 spearman 相关(日均) =====")
for k, v in corr_rows.items():
    print(f"  {k}: {v:.3f}")
print("\nper-day IC (raw):")
for r in main_table:
    print(" ", r["factor"], r["per_day"])
