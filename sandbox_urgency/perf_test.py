# -*- coding: utf-8 -*-
"""第二阶段性能压测：全市场合并+时间排序+二分窗口扫描的可行性。

关键问题：每股470笔迫切种子，每笔扫全市场前后5秒窗口(~9万笔交易)。
测定"每笔种子窗口扫描耗时"，据此外推总耗时，判断是否需要Rust。
"""
import os, time
import numpy as np
import rust_pyfunc as rp

DATE = 20251231
EPS = 1e-9
WIN = 5.0  # 秒


def list_codes(date):
    d = f"/ssd_data/stock/{date}/transaction"
    return sorted(fn.split("_")[0] for fn in os.listdir(d)
                  if fn.endswith("_transaction.csv") and fn.split("_")[0].isdigit())


def main():
    codes = list_codes(DATE)
    print(f"{len(codes)} codes", flush=True)

    # 读全市场
    t0 = time.time()
    times, asks, bids, flags, vols, cids = [], [], [], [], [], []
    code_id = {}
    for code in codes:
        try:
            a = rp.read_trade_fast(code, DATE, 0, True)
        except Exception:
            continue
        a = a[a[:, 4] != 32]
        if len(a) == 0:
            continue
        cid = len(code_id)
        code_id[code] = cid
        times.append(a[:, 0]); asks.append(a[:, 6]); bids.append(a[:, 5])
        flags.append(a[:, 4]); vols.append(a[:, 2]); cids.append(np.full(len(a), cid, np.int32))
    time_all = np.concatenate(times); ask_all = np.concatenate(asks); bid_all = np.concatenate(bids)
    flag_all = np.concatenate(flags); vol_all = np.concatenate(vols); cid_all = np.concatenate(cids)
    n = len(time_all)
    print(f"loaded {len(code_id)} stocks, {n/1e6:.1f}M trades, {time.time()-t0:.0f}s", flush=True)

    # signed比例
    ratio = (ask_all - bid_all) / (ask_all + bid_all + EPS)
    abratio = np.abs(ratio)
    q95 = np.percentile(ratio, 95)
    is_sell_urgent = ratio > q95
    print(f"sell-urgent q95={q95:.4f} total={is_sell_urgent.sum()} "
          f"({is_sell_urgent.sum()/n*100:.1f}%)", flush=True)

    # 全市场按时间排序(一次)
    t0 = time.time()
    order = np.argsort(time_all, kind="stable")
    time_s = time_all[order]; abratio_s = abratio[order]; vol_s = vol_all[order]
    cid_s = cid_all[order]; su_s = is_sell_urgent[order]; flag_s = flag_all[order]
    print(f"global sort {time.time()-t0:.0f}s", flush=True)

    # 窗口内交易数分布(确认窗口规模)
    # 用全市场时间，估算每10秒窗口的交易数
    sample_wins = []
    for t in time_s[::len(time_s)//100][:50]:
        lo = np.searchsorted(time_s, t, "left")
        hi = np.searchsorted(time_s, t+10, "right")
        sample_wins.append(hi-lo)
    print(f"窗口规模(10s全市场): 中位{int(np.median(sample_wins))} "
          f"p90={int(np.percentile(sample_wins,90))} max={max(sample_wins)}", flush=True)

    # 对 000001 的卖方迫切种子算窗口
    for target in ["000001", "300750"]:
        if target not in code_id:
            continue
        tid = code_id[target]
        tmask = (cid_all == tid) & is_sell_urgent
        seed_idx = np.where(tmask)[0]
        seed_times = time_all[seed_idx]
        print(f"\n[{target}] sell-urgent seeds: {len(seed_times)}", flush=True)

        N = min(200, len(seed_times))
        t0 = time.time()
        for t_seed in seed_times[:N]:
            lo = np.searchsorted(time_s, t_seed - WIN, "left")
            mid = np.searchsorted(time_s, t_seed, "left")
            hi = np.searchsorted(time_s, t_seed + WIN, "right")
            # 前窗口 [lo,mid) 后窗口 [mid,hi)
            for a, b in [(lo, mid), (mid, hi)]:
                _n = b - a
                _nu = su_s[a:b].sum()
                _mass = abratio_s[a:b].sum()
                _vmass = np.dot(abratio_s[a:b], vol_s[a:b])
        dt = time.time() - t0
        per = dt / N * 1000
        print(f"  {N} seeds × 2窗口 × 4指标: {dt:.2f}s = {per:.2f}ms/seed", flush=True)
        # 外推
        total_seeds_single_dir = is_sell_urgent.sum()
        print(f"  外推(单方向全市场{total_seeds_single_dir/1e6:.1f}M种子): "
              f"{per*total_seeds_single_dir/1e3:.0f}s/天(Python单线程)", flush=True)


if __name__ == "__main__":
    main()
