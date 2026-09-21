"""复现 hm104_brc_tail_v4 引擎 neu/gap5 通道的十分组收益。

口径（逐条对应 src/tail_v5_pipeline.rs）：
  - 因子从组合 store 根目录按 col_idx 读，落到回测轴（2430 天，= store 模板第 244 行起）
  - fold 变体: |raw - 当日非 NaN 均值|  (tail_v2_build._build_fold_values)
  - 预处理: rp.tail_v5_rank_fill_roll_block_f32(F, restrict, [5,10,20])  -> slot 0=_smooth_1,
    之后每个窗口 mean/max/min/std
  - 中性化: 引擎 v7/v8 走的是 factor_neutralize_std 那一支（neutralize_std_slot_f32_v2_resid），
    不是 tail_v5_neutralize_block_exact（后者是 legacy 那一支，结果对不上）。
    这里用同模块的 block 级包装 rp.neutralize_std_block_with_shared，precompute 一次全 run 复用。
  - 有效日: r = 1..T-1, dates[r] > 20160101, neu[r-1] 有有限值
  - held: local_t % 5 == 0 时 held = r-1
  - 过滤: signal[held] 有限 & ret[ r] 有限 & restrict[held] 有限且==0
  - 十分组: 组均收益 = 组内 ret 均值；rank/n*10 下取整, 夹到 9
  - 多头/空头 = 组 1 与组 10 的逐日收益之和，大的一端为多头
  - annualized_return = mean(多头-空头)*250 ; hedge_annualized_return = mean(多头-指数)*250
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import rust_pyfunc as rp

HERE = os.path.dirname(os.path.abspath(__file__))
STORE = "/hdd/user_home_unsafe/chenzongwei/factor_store_hot_stock_pool_v1_fix"
META = "/nas197/user_home_unsafe/chenzongwei/hm104_brc_tail_v4/meta"
BACKTEST = ("/home/chenzongwei/pythoncode/_tail_v2_shared/backtest_inputs/"
            "000905_20160104_20251231_9193_8c19aea9f4e01583")
SUMMARY = "/nas197/user_home_unsafe/chenzongwei/hm104_brc_tail_v4/metrics/summary_neu_gap5_candidates.parquet"
INPUT = "/home/chenzongwei/rust_pyfunc/tmp_brc_lead/group_extreme_input.json"
STYLE = "/ssd_data/data/vars"
INDUSTRY_CSV = "/nas197/binary/stock/sz_alpha/csv/vars/Base/SW_IND_CODE.csv"
BACKTEST_START = 20160101
GAP = 5
PORTF = 10
WINDOWS = [5, 10, 20]
STATS = ("mean", "max", "min", "std")


def parse_name(name):
    """派生名 -> (store 里的原始因子名, 是否 fold 变体, slot)。"""
    if name.endswith("_smooth_1"):
        variant, slot = name[: -len("_smooth_1")], 0
    else:
        for stat in STATS:
            token = f"_{stat}_smooth_"
            if token in name:
                variant, win = name.rsplit(token, 1)
                slot = 1 + WINDOWS.index(int(win)) * 4 + STATS.index(stat)
                break
        else:
            raise ValueError(f"无法解析派生名: {name}")
    is_fold = variant.endswith("_fold")
    return (variant[:-5] if is_fold else variant), is_fold, slot


def load_axes():
    dates = np.load(META + "/dates.npy").astype(np.int32)
    stocks = np.load(META + "/stocks.npy", allow_pickle=True).astype(str)
    return dates, stocks


def load_returns():
    ret = np.load(BACKTEST + "/ret_gap5.npy").astype(np.float32)
    ret_sum = np.load(BACKTEST + "/ret_sum_gap5.npy").astype(np.float32)
    restrict = np.load(BACKTEST + "/restrict.npy").astype(np.float32)
    index = np.load(BACKTEST + "/index_ret.npy").astype(np.float32)
    return ret, ret_sum, restrict, index


def load_industry(dates, stocks):
    """tail_v4._load_industry_matrix：SW_IND_CODE.csv 重排到模板轴 (T,N) f64。"""
    df = pd.read_csv(INDUSTRY_CSV, index_col=0).T
    df.index = pd.to_datetime(df.index)
    si = pd.Index(stocks)
    out = np.full((len(dates), len(stocks)), np.nan, dtype=np.float64)
    for i, d in enumerate(str(int(x)) for x in dates):
        ts = pd.Timestamp(f"{d[:4]}-{d[4:6]}-{d[6:8]}")
        if ts in df.index:
            out[i] = df.loc[ts].reindex(si).to_numpy(dtype=np.float64)
    return out


def fold_values(raw):
    """引擎 tail_v5_pipeline::build_fold_values：|raw - 当日非 NaN 均值|。

    注意与 tail_v2_build._build_fold_values 的差别：引擎先把均值截成 f32，
    再做 f32 减法（这里是逐位对齐表里指标的关键，f64 中间量会差 1e-5 量级）。
    """
    valid = np.isfinite(raw)
    counts = valid.sum(axis=1, keepdims=True)
    sums = np.where(valid, raw, 0.0).sum(axis=1, keepdims=True, dtype=np.float64)
    means = np.divide(sums, counts, out=np.full((raw.shape[0], 1), np.nan, dtype=np.float64),
                      where=counts > 0).astype(np.float32)
    out = np.full(raw.shape, np.nan, dtype=np.float32)
    np.subtract(raw, means, out=out, where=valid)
    np.abs(out, out=out)
    return out


def avg_rank(x):
    """平均名次（并列取平均），与 rank_both_radix_into 的 rk_avg 同义。"""
    _, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
    csum = np.cumsum(cnt)
    start = csum - cnt
    return ((start + csum + 1) / 2.0)[inv]


def group_returns(surface, ret, restrict, dates, gap=GAP, portf=PORTF):
    """返回 (group_returns(portf, T_eff), eff, ratio_values)。"""
    has = np.isfinite(surface).any(axis=1)
    eff = [r for r in range(1, dates.size)
           if dates[r] > BACKTEST_START and has[r - 1]]
    t_eff = len(eff)
    grp = np.zeros((portf, t_eff), dtype=np.float64)
    ratio = np.full(t_eff, np.nan, dtype=np.float64)
    if t_eff == 0:
        return grp, eff, ratio
    open_counts = np.isfinite(restrict).sum(axis=1)  # 占位，下面用真实定义覆盖
    open_counts = ((restrict == np.float32(0.0)) & np.isfinite(restrict)).sum(axis=1)
    held = eff[0] - 1
    for local_t, r in enumerate(eff):
        if local_t % gap == 0:
            held = r - 1
        sig = surface[held]
        rr = ret[r]
        rs = restrict[held]
        mask = np.isfinite(sig) & np.isfinite(rr) & np.isfinite(rs) & (rs == np.float32(0.0))
        n = int(mask.sum())
        if n < portf:
            continue
        vc = int(open_counts[r - 1])
        if vc > 0:
            ratio[local_t] = n / vc
        s = sig[mask]
        ranks = avg_rank(s)
        bucket = np.minimum((ranks / n * portf).astype(np.int64), portf - 1)
        rv = rr[mask].astype(np.float64)
        sums = np.bincount(bucket, weights=rv, minlength=portf)
        cnts = np.bincount(bucket, minlength=portf)
        grp[:, local_t] = np.where(cnts > 0, sums / np.maximum(cnts, 1), 0.0)
    return grp, eff, ratio


def metrics_from_groups(grp, eff, index):
    first = grp[0].sum()
    last = grp[PORTF - 1].sum()
    long_i, short_i = (0, PORTF - 1) if first > last else (PORTF - 1, 0)
    idx = index[np.asarray(eff, dtype=np.int64)]
    ls = grp[long_i] - grp[short_i]
    hedge = grp[long_i] - idx
    return {
        "long_i": long_i,
        "short_i": short_i,
        "group_sum": grp.sum(axis=1),
        "ls_mean250": float(np.nanmean(ls) * 250.0),
        "hedge_mean250": float(np.nanmean(hedge) * 250.0),
        "date_size": int(len(eff)),
    }


def main(limit=None):
    t0 = time.time()
    dates, stocks = load_axes()
    ret, ret_sum, restrict, index = load_returns()
    info = rp.factor_store_v5_info(STORE)
    names = list(info["factor_names"])
    col = {n: i for i, n in enumerate(names)}
    sd = np.asarray(rp.factor_store_v5_template(STORE)["dates"])
    off = int(np.searchsorted(sd, dates[0]))
    assert np.array_equal(sd[off:off + dates.size], dates), "store 模板轴与回测轴不匹配"
    t_axis, n_stocks = dates.size, stocks.size
    dates_list = dates.astype(np.int32).tolist()
    stocks_list = [str(s) for s in stocks]
    industry = load_industry(dates, stocks)
    shared = rp.neutralize_std_precompute_py(industry, restrict, STYLE, dates_list, stocks_list)
    print(f"[{time.time()-t0:.0f}s] neutralize precompute 完成", flush=True)

    inp = json.load(open(INPUT))
    todo = inp["union"] if limit is None else inp["union"][:limit]

    rows = []
    for k, name in enumerate(todo, 1):
        base, is_fold, slot = parse_name(name)
        rec = rp.factor_store_v5_read_factor(STORE, col[base])
        di = np.asarray(rec["date_id"], dtype=np.int64) - off
        ci = np.asarray(rec["code_id"], dtype=np.int64)
        v = np.asarray(rec["factor"], dtype=np.float32)
        keep = (di >= 0) & (di < t_axis)
        F = np.full((t_axis, n_stocks), np.nan, dtype=np.float32)
        F[di[keep], ci[keep]] = v[keep]
        del rec, di, ci, v, keep
        if is_fold:
            F = fold_values(F)
        block = rp.tail_v5_rank_fill_roll_block_f32(F, restrict, WINDOWS)
        surf = np.array(block[:, :, slot], dtype=np.float32, copy=True)
        del block, F
        neu = rp.neutralize_std_block_with_shared(surf[:, :, None], shared, False)
        neu = np.array(neu[:, :, 0], dtype=np.float32, copy=True)
        del surf
        grp, eff, ratio = group_returns(neu, ret, restrict, dates)
        m = metrics_from_groups(grp, eff, index)
        m["factor_name"] = name
        m["base"] = base
        m["is_fold"] = is_fold
        m["slot"] = slot
        m["ratio_mean"] = float(np.nanmean(ratio))
        rows.append(m)
        print(f"{k}/{len(todo)} {name} ls={m['ls_mean250']:.6f} hedge={m['hedge_mean250']:.6f} "
              f"n={m['date_size']}", flush=True)
        if k % 10 == 0:
            print(f"  [{time.time()-t0:.0f}s]", flush=True)

    out = pd.DataFrame([{k: v for k, v in r.items() if k != "group_sum"} for r in rows])
    np.save(os.path.join(HERE, "group_sums.npy"),
            np.stack([r["group_sum"] for r in rows]))
    out.to_csv(os.path.join(HERE, "repro_metrics.csv"), index=False)

    tbl = pd.read_parquet(SUMMARY)
    tbl = tbl[(tbl["stage"] == "neu") & (tbl["gap"] == 5)].set_index("factor_name")
    out = out.set_index("factor_name")
    tbl = tbl.loc[out.index]
    cmp = pd.DataFrame({
        "calc_ls": out["ls_mean250"],
        "tbl_ls": tbl["annualized_return"],
        "calc_hedge": out["hedge_mean250"],
        "tbl_hedge": tbl["hedge_annualized_return"],
        "calc_dsize": out["date_size"],
        "tbl_dsize": tbl["date_size"],
        "calc_ratio": out["ratio_mean"],
        "tbl_ratio": tbl["ratio_mean"],
    })
    for a, b in (("ls", "annualized_return"), ("hedge", "hedge_annualized_return")):
        denom = cmp[f"tbl_{a}"].abs().clip(lower=1e-12)
        cmp[f"rel_{a}"] = (cmp[f"calc_{a}"] - cmp[f"tbl_{a}"]).abs() / denom
    cmp.to_csv(os.path.join(HERE, "selfcheck.csv"))
    print("\n=== 自检 (66 因子) ===")
    for a, b in (("ls", "annualized_return"), ("hedge", "hedge_annualized_return")):
        ok = cmp[f"rel_{a}"] < 1e-6
        print(f"{b}: 对上 {int(ok.sum())}/{len(cmp)}  最大相对误差 {cmp[f'rel_{a}'].max():.3e}")
    print("date_size 对上:", int((cmp["calc_dsize"] == cmp["tbl_dsize"]).sum()), "/", len(cmp))
    print("ratio_mean 最大绝对差:", float((cmp["calc_ratio"] - cmp["tbl_ratio"]).abs().max()))
    print(f"总耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else None)
