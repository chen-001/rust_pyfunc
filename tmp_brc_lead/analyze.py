"""hm104 上对比 BRC / ssm1 / ssm2 / ssm3 / |IC| / 多头超额 六个选法的入选名单与指标。

BRC 值来自 IC-only 那一轮（hm104_brc_ic）——BRC 只要横截面、不依赖十分组，IC-only 模式
一样算，而且快得多。|RankIC| / 多头超额 / SSM / MPROB 统一从 hm104_mprob 那轮的完整回测
表取（同一批 derived 因子、同一套引擎配置）。

选法：
  brc      : metric = "BRC"（= min(BRC_S, BRC_L)），门槛 0，corr 0.5 去重，取前 35
  brc_ori  : metric = sign(S*L)*min(|S|,|L|)（方向无关版），其余同上
  ssm1     : metric = "SSM"，门槛 -1（无门槛，纯 SSM 降序），corr 0.5，取前 35
  ic       : metric = "IC_mean"（|IC| 降序），门槛 0.01，corr 0.5，取前 35
  hedge    : metric = "hedge_annualized_return"，门槛 0.055，corr 0.5，取前 35
  ssm2/ssm3: 直接读引擎跑出来的 selected/gap5_selected.parquet
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/chenzongwei/design_whatever")
from design_whatever.tail_v2_screen import _select_by_metric  # noqa: E402
from design_whatever.tail_v2_storage import make_tail_v2_paths  # noqa: E402
from design_whatever.tail_v4 import _load_ic_wide, _load_summary_json  # noqa: E402

NAS = Path("/nas197/user_home_unsafe/chenzongwei")
VER_BRC = "hm104_brc_ic"
VER_FULL = "hm104_mprob"
R_BRC = NAS / f"{VER_BRC}_tail_v4"
R_FULL = NAS / f"{VER_FULL}_tail_v4"
OUT = Path("/home/chenzongwei/rust_pyfunc/tmp_brc_lead")
TOPN = 35
COVER = 0.5
CORR = 0.5

t0 = time.time()


def log(m):
    print(f"[{time.time()-t0:7.1f}s] {m}", flush=True)


if not (R_BRC / "metrics" / "summary_neu_gap5_candidates.parquet").exists():
    print(f"BRC(IC-only) 那轮还没产出: {R_BRC}")
    sys.exit(2)

paths_b = make_tail_v2_paths(VER_BRC, R_BRC)
paths_f = make_tail_v2_paths(VER_FULL, R_FULL)
brc_tab = _load_summary_json(paths_b.metrics_dir / "summary_neu_gap5_candidates.json")
full = _load_summary_json(paths_f.metrics_dir / "summary_neu_gap5_candidates.json")
log(f"BRC 表 {brc_tab.shape} 列={list(brc_tab.columns)}")
log(f"完整表 {full.shape} 列={list(full.columns)}")

for c in ("BRC", "BRC_S", "BRC_L"):
    v = brc_tab[c].to_numpy()
    log(f"  {c}: 非空 {np.isfinite(v).sum()}/{len(v)} 范围[{np.nanmin(v):.4f}, {np.nanmax(v):.4f}]")
brc_tab["BRC_ORI"] = np.sign(brc_tab["BRC_S"] * brc_tab["BRC_L"]) * np.minimum(
    brc_tab["BRC_S"].abs(), brc_tab["BRC_L"].abs()
)

# 同一因子的 IC_mean 在两轮里必须一致（IC-only 与完整回测的 IC 序列同一套）
m = brc_tab[["factor_name", "IC_mean"]].merge(
    full[["factor_name", "IC_mean"]], on="factor_name", suffixes=("_ic", "_full")
)
dmax = float(np.nanmax(np.abs(m.IC_mean_ic.to_numpy() - m.IC_mean_full.to_numpy())))
log(f"两轮 IC_mean 逐因子最大差 = {dmax:.3e}（应接近 0）")

full["BRC"] = np.nan
full["BRC_S"] = np.nan
full["BRC_L"] = np.nan
full["BRC_ORI"] = np.nan
full = full.set_index("factor_name")
bi = brc_tab.set_index("factor_name")
common = full.index.intersection(bi.index)
for c in ("BRC", "BRC_S", "BRC_L", "BRC_ORI"):
    full.loc[common, c] = bi.loc[common, c].to_numpy()
full = full.reset_index()
log(f"并表后 BRC 非空 {np.isfinite(full.BRC.to_numpy()).sum()}/{len(full)}")

pool_brc = brc_tab[brc_tab.ratio_mean >= COVER].copy()
pool_full = full[full.ratio_mean >= COVER].copy()
log(f"BRC 池 {pool_brc.shape}；完整池 {pool_full.shape}")

dates = np.load(R_FULL / "meta" / "dates.npy", allow_pickle=False).astype(np.int32)
ic5 = _load_ic_wide(
    paths_f.ic_dir / "ic_neu_gap5.npy", paths_f.ic_dir / "ic_neu_gap5_names.json", dates, gap=5
)
log(f"ic_wide {ic5.shape}")


def pick(pool, metric, min_value, topn=TOPN):
    t = time.time()
    r = _select_by_metric(pool, ic5, metric, min_value, CORR)
    names = r.factor_name.tolist()[:topn]
    log(f"  {metric:28s} >= {min_value:<8} 去重后 {len(r):6d}，取前 {len(names)}，用时 {time.time()-t:.1f}s")
    return names


sel = {}
sel["brc"] = pick(pool_brc, "BRC", 0.0)
sel["brc_ori"] = pick(pool_brc, "BRC_ORI", 0.0)
_pre = OUT / "precomputed_picks.json"
if _pre.exists():
    sel.update(json.loads(_pre.read_text(encoding="utf-8")))
    log(f"  ic / hedge / ssm1 用预计算结果（{_pre}）")
else:
    sel["ssm1"] = pick(pool_full, "SSM", -1.0)
    sel["ic"] = pick(pool_full, "IC_mean", 0.01)
    sel["hedge"] = pick(pool_full, "hedge_annualized_return", 0.055)
for tag, ver in (("ssm2", "hm104_ssm2"), ("ssm3", "hm104_ssm3")):
    sel[tag] = pd.read_parquet(NAS / f"{ver}_tail_v4" / "selected" / "gap5_selected.parquet")[
        "factor_name"
    ].tolist()
    log(f"{tag}: {len(sel[tag])} 个")

# ---------------------------------------------------------------- 指标统计
idx = full.set_index("factor_name")
order = ["brc", "brc_ori", "ssm1", "ssm2", "ssm3", "ic", "hedge"]
label = {"brc": "BRC 前35", "brc_ori": "BRC方向无关版前35", "ssm1": "ssm1(纯SSM排序)前35",
         "ssm2": "ssm2(SSM>=0.5+|IC|)", "ssm3": "ssm3(SSM>=0.9+|IC|)",
         "ic": "|IC|前35", "hedge": "多头超额前35"}

rows, detail = [], {}
for k in order:
    sub = idx.reindex(sel[k])
    ic = sub["IC_mean"].abs().to_numpy(dtype=float)
    hd = sub["hedge_annualized_return"].to_numpy(dtype=float)
    brc = sub["BRC"].to_numpy(dtype=float)
    detail[k] = sub
    rows.append(dict(
        选法=label[k], n=len(sub),
        RankIC中位=np.nanmedian(ic), RankIC均值=np.nanmean(ic),
        RankIC_25=np.nanpercentile(ic, 25), RankIC_75=np.nanpercentile(ic, 75),
        多头超额中位=np.nanmedian(hd), 多头超额均值=np.nanmean(hd),
        多头超额_25=np.nanpercentile(hd, 25), 多头超额_75=np.nanpercentile(hd, 75),
        BRC中位=np.nanmedian(brc),
    ))
stat = pd.DataFrame(rows)
pd.set_option("display.width", 250, "display.max_columns", 50)
print()
print("=== 各选法 35 个入选因子的指标（|RankIC| / 多头超额 均为年化，来自完整回测表） ===")
print(stat.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

print()
print("=== 选法两两重合（交集个数） ===")
ov = pd.DataFrame(index=[label[k] for k in order], columns=[label[k] for k in order], dtype=int)
for a in order:
    for b in order:
        ov.loc[label[a], label[b]] = len(set(sel[a]) & set(sel[b]))
print(ov.to_string())

print()
print("=== 「二者重复」：|IC| 前35 与 多头超额前35 的重合 ===")
print(f"  |IC|前35 ∩ 多头超额前35 = {len(set(sel['ic']) & set(sel['hedge']))} 个")
for k in order:
    print(f"  {label[k]:22s} ∩ |IC|前35 = {len(set(sel[k]) & set(sel['ic'])):2d} 个；"
          f" ∩ 多头超额前35 = {len(set(sel[k]) & set(sel['hedge'])):2d} 个")

print()
print("=== BRC 前 35 明细（按 BRC 降序） ===")
cols = ["factor_name", "BRC", "BRC_S", "BRC_L", "IC_mean", "hedge_annualized_return", "SSM", "MPROB"]
d = detail["brc"].reset_index()[cols].copy()
d["IC_mean"] = d["IC_mean"].abs()
d = d.rename(columns={"IC_mean": "|RankIC|"})
print(d.to_string(index=False, float_format=lambda v: f"{v:.5f}"))

print()
print("=== BRC 与 |IC| / 多头超额 的关系（BRC 池 35704 个因子） ===")
from scipy.stats import spearmanr  # noqa: E402

a = pool_brc["BRC"].to_numpy(float)
ics = pool_brc["IC_mean"].to_numpy(float)
b = np.abs(ics)
# 多头超额只在完整回测表里有值（IC-only 那轮收益字段是 0），所以从并表后的 full 里取
fb = full[np.isfinite(full["BRC"].to_numpy(float))].copy()
fb["|BRC|"] = fb["BRC"].abs()
fb = fb.dropna(subset=["hedge_annualized_return"])
print(f"  BRC vs 带符号 IC_mean  Spearman = {spearmanr(a, ics).statistic:.4f}")
print(f"  |BRC| vs |RankIC|      Spearman = {spearmanr(np.abs(a), b).statistic:.4f}")
print(f"  BRC 与 IC_mean 的比值中位 = {np.nanmedian(a / np.where(np.abs(ics) > 1e-12, ics, np.nan)):.4f}"
      f"（BRC 带方向，约等于 IC 的 {np.nanmedian(a / np.where(np.abs(ics) > 1e-12, ics, np.nan)):.2f} 倍）")
print(f"  BRC vs 多头超额（{len(fb)} 个有值）Spearman = "
      f"{spearmanr(fb['BRC'].to_numpy(float), fb['hedge_annualized_return'].to_numpy(float)).statistic:.4f}")
print(f"  |BRC| vs 多头超额       Spearman = "
      f"{spearmanr(fb['|BRC|'].to_numpy(float), fb['hedge_annualized_return'].to_numpy(float)).statistic:.4f}")
print(f"  BRC 前35 里最小的 BRC = {min(idx.loc[n, 'BRC'] for n in sel['brc']):.5f}"
      f" → 折算 |IC| 约 {min(idx.loc[n, 'BRC'] for n in sel['brc']) / np.nanmedian(np.abs(a) / np.maximum(b, 1e-12)):.4f}")
print(f"  BRC 池里 |IC| < 0.01 的因子数 = {int((b < 0.01).sum())}（IC-only 模式门槛所致，不在池内）")
print(f"  BRC 前35 中 BRC_L < BRC_S（多头是弱侧）的个数 = "
      f"{sum(1 for n in sel['brc'] if idx.loc[n, 'BRC_L'] < idx.loc[n, 'BRC_S'])}/35")

stat.to_csv(OUT / "brc_vs_ssm_stats.csv", index=False)
ov.to_csv(OUT / "brc_vs_ssm_overlap.csv")
d.to_csv(OUT / "brc_top35_detail.csv", index=False)
with open(OUT / "brc_selections.json", "w", encoding="utf-8") as f:
    json.dump(sel, f, ensure_ascii=False, indent=2)
log(f"产物写到 {OUT}")
