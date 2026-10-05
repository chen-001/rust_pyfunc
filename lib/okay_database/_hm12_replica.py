"""hm12 我的算法的 Python 复刻，和真旧代码逐值比（临时，用完删）。"""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, '/tmp')
from mine_qs import mine                      # noqa: E402
sys.path.insert(0, "/home/chenzongwei/pythoncode")
sys.path.insert(0, "/home/chenzongwei/rust_pyfunc/lib/okay_database")
import design_whatever_out as dwo             # noqa: E402
import verify_series as vs                    # noqa: E402

LO = (9 * 3600 + 30 * 60) * 10 ** 9
HI = (14 * 3600 + 57 * 60) * 10 ** 9
SESS = [('all', LO, HI), ('open30', LO, 10 * 3600 * 10 ** 9),
        ('morning', LO, (11 * 3600 + 30 * 60) * 10 ** 9),
        ('afternoon', 13 * 3600 * 10 ** 9, HI),
        ('close30', (14 * 3600 + 30 * 60) * 10 ** 9, HI)]


PAIRS = {}
SUMS = {}


def calc(C, D):
    rc = dwo.TickDataBase(1, D, D, 1, symbol=C, get_many=0, read_full=0, change_time=1)
    df = rc.read_trade_data_single(with_retreat=0)
    df2 = rc.read_market_data_single()
    tr = df[['exchtime', 'volume', 'turnover', 'flag']].copy()
    tr = tr[tr.flag != 32]
    t_ns = tr.exchtime.to_numpy().astype('int64')
    tvol = tr.volume.to_numpy().astype(float)
    tto = tr.turnover.to_numpy()
    tflag = tr.flag.to_numpy()
    mk_ns = df2.exchtime.to_numpy().astype('int64')
    ask1 = df2.ask_prc1.to_numpy()
    bid1 = df2.bid_prc1.to_numpy()
    sel = np.where((mk_ns % 86400000000000 >= LO) & (mk_ns % 86400000000000 <= HI))[0]
    seen = set()
    mkd, labd, mka, mkb = [], [], [], []
    for pos, idx in enumerate(sel):
        v = mk_ns[idx]
        if v not in seen:
            seen.add(v); mkd.append(v); labd.append(pos)
            mka.append(ask1[idx]); mkb.append(bid1[idx])
    mkd = np.array(mkd)
    perm = mine(np.concatenate([t_ns, mkd]))
    nt = len(t_ns)
    labels = np.full(nt, -1, dtype=np.int64)
    cur = -1
    for i in perm:
        if i >= nt:
            cur = labd[i - nt]
        else:
            labels[i] = cur
    inwin = (t_ns % 86400000000000 >= LO) & (t_ns % 86400000000000 <= HI)
    # pandas groupby.sum 对 float64 用的是 Kahan 补偿求和，且按 frame 行序（= 排序后）加
    v66, to66, v83, to83 = {}, {}, {}, {}
    c66, ct66, c83, ct83 = {}, {}, {}, {}
    def add(sums, comp, L, x):
        c = comp.get(L, 0.0); s0 = sums.get(L, 0.0)
        y = x - c; t = s0 + y
        comp[L] = (t - s0) - y; sums[L] = t
    for i in perm:
        if i >= nt or not inwin[i] or labels[i] < 0:
            continue
        L = int(labels[i])
        if tflag[i] == 66:
            add(v66, c66, L, tvol[i]); add(to66, ct66, L, tto[i])
        elif tflag[i] == 83:
            add(v83, c83, L, tvol[i]); add(to83, ct83, L, tto[i])
    t_tod_all = t_ns % 86400000000000
    mk_tod = mkd % 86400000000000
    out = []
    PAIRS.clear(); SUMS.clear()
    SUMS.update(dict(v66=v66, t66=to66, v83=v83, t83=to83))
    for nm, s, e in SESS:
        # ！！旧代码的时段 label 集合来自「时段内所有行」（成交 + 盘口），
        #   所以某个成交若紧跟在时段开始之前那一笔快照后面，会把那个**时段外**的快照 label
        #   也带进这个时段（实测 600519@20250304 close30 的 label 4365，快照 14:29:58）。
        S_labels = set()
        for k in range(len(mkd)):
            if s <= mk_tod[k] <= e:
                S_labels.add(int(labd[k]))
        for i in range(nt):
            if s <= t_tod_all[i] <= e and labels[i] >= 0:
                S_labels.add(int(labels[i]))
        A, B = [], []
        for k in range(len(mkd)):
            L = int(labd[k])
            if L not in S_labels:
                continue
            if L in v66:
                p3 = to66[L] / v66[L]
                if not np.isnan(p3):
                    A.append((L, abs(p3 - mka[k])))
            if L in v83:
                p3 = to83[L] / v83[L]
                if not np.isnan(p3):
                    B.append((L, abs(mkb[k] - p3)))
        if not A or not B:
            out.append(np.nan); continue
        bd = dict(B)
        d = [np.nan if (x + bd[L]) == 0 else x / (x + bd[L]) for L, x in A if L in bd]
        PAIRS[nm] = sorted((L, (np.nan if (x + bd[L]) == 0 else x / (x + bd[L]))) for L, x in A if L in bd)
        out.append(np.nanmean(d) if d else np.nan)
    return out


if __name__ == "__main__":
    mod, cap = vs.load_old("/home/chenzongwei/pythoncode/成果hm12_成果hm13_梦想三级跳/hm12_okay.py")
    for C, D in [("000001", 20250303), ("600519", 20250304), ("000002", 20180601), ("300750", 20231020)]:
        old, _ = vs.old_values(mod, cap, C, D)
        old = [float(x) for x in old]
        new = calc(C, D)
        print(C, D)
        for i in range(5):
            rel = abs(old[i] - new[i]) / max(abs(old[i]), abs(new[i]), 1e-12)
            print(f"   #{i} 旧={old[i]!r} 我={new[i]!r} rel={rel:.2e}")
