"""第三版验收（只读 Rust 输出，不参与任何因子计算）：
1. 两个拆分任务的列拼回来后与冻结基准逐位相同（按 32 位整数比较，空值的位模式也要一致）
2. 撤单列空值：沪市 20211105 前、深市 20160104 前，凡列名含 cxl 的一律为空；跨日的撤单列要求前一日也有撤单数据
3. 2016 读取修复：20160630 深市 0 开头股票能读到数据
4. 列名约定、列数、读回耗时
用法：python validate_v3.py <检查项> ...
"""
import os
os.environ.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
import json, sys, time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = f"{ROOT}/out"


def load(run, date, mmap=True):
    m = json.load(open(f"{OUT}/{run}/{date}/meta.json"))
    p = f"{OUT}/{run}/{date}/vals.bin"
    if mmap:
        v = np.memmap(p, dtype="<u4", mode="r", shape=(m["n_rows"], m["n_cols"]))
    else:
        v = np.fromfile(p, dtype="<u4").reshape(m["n_rows"], m["n_cols"])
    return m, v


def check_split(base_run, parts, date):
    mb, vb = load(base_run, date)
    pos = {n: i for i, n in enumerate(mb["names"])}
    seen = 0
    for run in parts:
        mp, vp = load(run, date)
        assert mp["codes"] == mb["codes"], f"{run} 股票顺序与基准不同"
        idx = np.array([pos[n] for n in mp["names"]])
        seen += len(idx)
        for r0 in range(0, mb["n_rows"], 256):
            a = np.asarray(vb[r0:r0 + 256])[:, idx]
            b = np.asarray(vp[r0:r0 + 256])
            if not np.array_equal(a, b):
                bad = np.argwhere(a != b)[0]
                raise SystemExit(f"{run} 与基准不同：行 {r0 + bad[0]} 列 {mp['names'][bad[1]]}")
        print(f"{run}: {len(idx)} 列与基准逐位相同")
    print(f"拆分任务合计 {seen} 列，基准 {mb['n_cols']} 列，列名集合相同: {seen == mb['n_cols'] and len(set(sum([load(r, date)[0]['names'] for r in parts], []))) == mb['n_cols']}")


def check_cxl(run, date):
    m, v = load(run, date)
    names = np.array(m["names"])
    codes = np.array(m["codes"])
    f = v.view("<f4")
    is_cxl = np.array(["cxl" in n for n in names])
    sh = np.array([c.startswith("6") for c in codes])
    prev = m["prev_date"]
    print(f"{date}（前一交易日 {prev}）股票 {len(codes)}：沪 {sh.sum()} 深 {(~sh).sum()}；cxl 列 {is_cxl.sum()} / {len(names)}")
    xd_cxl = np.array([n.startswith("xd_cxl") for n in names])
    for mk, sel in [("沪", sh), ("深", ~sh)]:
        sub = np.asarray(f[sel])
        fin = np.isfinite(sub)
        cx = fin[:, is_cxl]
        nc = fin[:, ~is_cxl]
        # 有任何 cxl 值的股票数；cxl 列非空比例；cxl 列里等于 0 的格子数
        any_cxl = cx.any(axis=1).sum()
        zeros_cxl = int((sub[:, is_cxl] == 0).sum())
        print(f"  {mk}：有非撤单值的股票 {nc.any(axis=1).sum()}；有撤单值的股票 {any_cxl}；撤单列非空比例 {cx.mean():.4f}；"
              f"非撤单列非空比例 {nc.mean():.4f}；撤单列中恰为 0 的格子 {zeros_cxl}；跨日撤单列非空比例 {fin[:, xd_cxl].mean():.4f}")
        # 朴素量 active_share（缺数据必须空值）
        col = list(names).index("cxl_vol_d10_full_g1_naive_active_share")
        x = sub[:, col]
        print(f"    cxl_vol_d10_full_g1_naive_active_share：空值 {np.isnan(x).sum()}，为 0 {int((x == 0).sum())}，有值 {np.isfinite(x).sum()}")


def check_2016(run, date):
    m, v = load(run, date)
    f = v.view("<f4")
    names = m["names"]
    codes = np.array(m["codes"])
    col = names.index("flow_d10_full_g1_cnt_pair_ratio")
    x = np.asarray(f[:, col])
    for pre in ["000", "001", "002", "003", "300", "600", "601", "603"]:
        sel = np.array([c.startswith(pre) for c in codes])
        if sel.sum():
            print(f"  {date} {pre} 开头：股票 {sel.sum()}，flow 计数配对比有值 {np.isfinite(x[sel]).sum()}")


def check_names(run, date):
    m, _ = load(run, date)
    names = m["names"]
    assert len(names) == len(set(names)), "列名重复"
    pre = {}
    for n in names:
        p = n.split("_")[0]
        pre[p] = pre.get(p, 0) + 1
    print("列数", len(names), "其中 cxl", sum("cxl" in n for n in names), "前缀分布（序列块按字母名）", dict(sorted(pre.items(), key=lambda x: -x[1])[:8]))


def readback(run, date):
    t = time.time()
    m = json.load(open(f"{OUT}/{run}/{date}/meta.json"))
    t1 = time.time()
    v = np.fromfile(f"{OUT}/{run}/{date}/vals.bin", dtype="<f4")
    t2 = time.time()
    print(f"读回：meta {t1 - t:.2f}s，因子值 {v.nbytes / 1e9:.2f} GB 用 {t2 - t1:.2f}s（{v.nbytes / 1e9 / (t2 - t1):.2f} GB/s）")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "split":
        check_split(sys.argv[2], sys.argv[4:], int(sys.argv[3]))
    elif cmd == "cxl":
        check_cxl(sys.argv[2], int(sys.argv[3]))
    elif cmd == "y2016":
        check_2016(sys.argv[2], int(sys.argv[3]))
    elif cmd == "names":
        check_names(sys.argv[2], int(sys.argv[3]))
    elif cmd == "readback":
        readback(sys.argv[2], int(sys.argv[3]))
