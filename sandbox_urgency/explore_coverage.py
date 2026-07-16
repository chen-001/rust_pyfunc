# -*- coding: utf-8 -*-
"""第一阶段探索：两种差值方案在'全市场分位点'下的每股迫切交易覆盖率。

核心问题：哪种差值定义让"不存在迫切交易(无任何交易超过全市场阈值)的股票"更少。
- 方案A 比例: |ask-bid|/(ask+bid)   归一化到[0,1], 消除编号绝对大小影响
- 方案B 绝对值: |ask-bid|            跨股票不可比(编号范围差10倍), 可能被大编号股票主导

关键背景: 不同股票订单编号范围差异巨大(000001最大3895万, 688001仅493万),
所以全市场绝对值分位点会被大编号股票主导, 小编号股票可能一笔都达不到阈值.
"""
import os
import time
import numpy as np
import rust_pyfunc as rp

DATA_DIR = "/ssd_data/stock"
EPS = 1e-9


def list_codes(date):
    d = f"{DATA_DIR}/{date}/transaction"
    codes = []
    for fn in os.listdir(d):
        if fn.endswith("_transaction.csv"):
            code = fn.split("_")[0]
            if code.isdigit() and len(code) == 6:
                codes.append(code)
    return sorted(codes)


def explore_date(date):
    codes = list_codes(date)
    print(f"\n[{date}] {len(codes)} stocks", flush=True)
    ratios = []      # 每股 diff_ratio (float32)
    abss = []        # 每股 diff_abs (float32)
    valid_codes = []
    t0 = time.time()
    for i, code in enumerate(codes):
        try:
            a = rp.read_trade_fast(code, date, 0, True)  # (n,7): [5]bid [6]ask
        except Exception:
            continue
        a = a[a[:, 4] != 32]  # ⚠️ read_trade_fast 未过滤撤单(flag=32), 手动过滤
        if len(a) == 0:
            continue
        ask = a[:, 6]
        bid = a[:, 5]
        s = ask + bid
        da = np.abs(ask - bid)
        ratio = (da / (s + EPS)).astype(np.float32)
        abst = da.astype(np.float32)
        ratios.append(ratio)
        abss.append(abst)
        valid_codes.append(code)
        if (i + 1) % 1000 == 0:
            print(f"  {i+1}/{len(codes)} elapsed {time.time()-t0:.0f}s", flush=True)
    print(f"  loaded {len(valid_codes)} valid in {time.time()-t0:.0f}s", flush=True)

    full_ratio = np.concatenate(ratios)
    full_abs = np.concatenate(abss)
    n_total = len(full_ratio)
    print(f"  total trades: {n_total/1e6:.1f}M", flush=True)

    # ---- 方案A 比例 ----
    print("\n  [方案A 比例 |ask-bid|/(ask+bid)]")
    for q in [95, 90]:
        thr = np.percentile(full_ratio, q)
        cov = np.mean([np.any(r > thr) for r in ratios])
        print(f"    >q{q}%: thr={thr:.5f}  覆盖率={cov*100:.2f}%  "
              f"(无迫切交易股票 {(1-cov)*100:.2f}%)")
    for q in [10, 5]:
        thr = np.percentile(full_ratio, q)
        cov = np.mean([np.any(r < thr) for r in ratios])
        print(f"    <q{q}%: thr={thr:.5f}  覆盖率={cov*100:.2f}%  "
              f"(无迫切交易股票 {(1-cov)*100:.2f}%)")

    # ---- 方案B 绝对值 ----
    print("\n  [方案B 绝对值 |ask-bid|]")
    for q in [95, 90]:
        thr = np.percentile(full_abs, q)
        cov = np.mean([np.any(a > thr) for a in abss])
        print(f"    >q{q}%: thr={thr:.0f}  覆盖率={cov*100:.2f}%  "
              f"(无迫切交易股票 {(1-cov)*100:.2f}%)")
    for q in [10, 5]:
        thr = np.percentile(full_abs, q)
        cov = np.mean([np.any(a < thr) for a in abss])
        print(f"    <q{q}%: thr={thr:.0f}  覆盖率={cov*100:.2f}%  "
              f"(无迫切交易股票 {(1-cov)*100:.2f}%)")

    # ---- 诊断：绝对值方案无覆盖股票的编号范围 ----
    thr_a95 = np.percentile(full_abs, 95)
    no_cov = [c for c, a in zip(valid_codes, abss) if not np.any(a > thr_a95)]
    print(f"\n  诊断: ABS>q95% 无迫切交易股票 {len(no_cov)} 只 "
          f"(占 {len(no_cov)/len(valid_codes)*100:.1f}%)")
    if no_cov:
        # 抽样看这些股票的编号范围
        idx_map = {c: k for k, c in enumerate(valid_codes)}
        rngs = []
        for c in no_cov[:2000]:
            k = idx_map[c]
            rngs.append((abss[k].max(), c))
        rngs.sort()
        print(f"    这些股票最大绝对差值 min/median/max: "
              f"{rngs[0][0]:.0f} / {rngs[len(rngs)//2][0]:.0f} / {rngs[-1][0]:.0f}")
        print(f"    样本(差值最小的): {[(c, int(v)) for v, c in rngs[:5]]}")

    return full_ratio, full_abs, valid_codes, ratios, abss


if __name__ == "__main__":
    for date in [20220819, 20251231]:
        explore_date(date)
        print("=" * 78, flush=True)
