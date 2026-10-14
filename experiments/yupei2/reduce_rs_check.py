#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""独立核对 Rust 版降维与 Python 版降维是否一致。

两边都是 N×F f32 行主序，列名各有一份 names.txt。检查四项：
  1 列名逐行一致（顺序也要一致）
  2 NaN（缺失值）出现的位置逐格一致
  3 两边都有限的位置上，数值最大绝对差与最大相对差
  4 有没有整列全空、有没有取值恒定

用法:
  python experiments/yupei2/reduce_rs_check.py <日期> [<python目录>] [<rust目录>]
默认目录是 /tmp/yupei2_pair_so/<日期>/reduce 和 /tmp/yupei2_pair_so/<日期>/reduce_rs
"""
import os
import sys
import numpy as np

ROOT = "/tmp/yupei2_pair_so"


def load(d, n):
    names = [l.strip() for l in open(os.path.join(d, "names.txt")) if l.strip()]
    v = np.fromfile(os.path.join(d, "factors.bin"), dtype="<f4")
    if v.size % len(names):
        raise SystemExit(f"{d}/factors.bin 有 {v.size} 个元素，不是 {len(names)} 的整数倍")
    rows = v.size // len(names)
    if n is not None and rows != n:
        raise SystemExit(f"{d}/factors.bin 有 {rows} 行，期望 {n} 行")
    return names, v.reshape(rows, len(names))


def main():
    date = int(sys.argv[1])
    n = 3964 if date == 20241231 else 3950
    pd = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, str(date), "reduce")
    rd = sys.argv[3] if len(sys.argv) > 3 else os.path.join("/tmp/yupei2_pair_rs", str(date), "reduce_rs")

    pn, pv = load(pd, n)
    rn, rv = load(rd, None)

    print(f"=== {date} ===")
    print(f"  Python  {os.path.join(pd,'factors.bin')}  {pv.shape}")
    print(f"  Rust    {os.path.join(rd,'factors.bin')}  {rv.shape}")

    # Rust 版只输出股票白名单内的代码，Python 版是全部行（非个股整行空值）。
    # 对齐口径：按代码表把 Python 侧也裁到白名单，再逐格比。
    codes = [l.strip() for l in open(os.path.join(ROOT, str(date), "codes.txt")) if l.strip()]
    sm = set(l.split(",")[0]
             for l in open("/ssd_data/data/basic_info/symbol_map.csv").read().splitlines()[1:] if l)
    keep = [i for i, c in enumerate(codes) if c in sm]
    if pv.shape[0] != rv.shape[0]:
        if len(keep) != rv.shape[0]:
            print(f"  [不通过] Rust 行数 {rv.shape[0]} 与白名单股票数 {len(keep)} 不符")
            return 1
        pv = pv[keep]
        print(f"  Python 侧已按白名单裁到 {pv.shape[0]} 行（原始 {n} 行）")

    ok = True
    if pn != rn:
        ok = False
        print(f"  [不通过] 列名不一致：Python {len(pn)} 列，Rust {len(rn)} 列")
        for i in range(max(len(pn), len(rn))):
            a = pn[i] if i < len(pn) else "<缺>"
            b = rn[i] if i < len(rn) else "<缺>"
            if a != b:
                print(f"    第 {i} 列：Python={a}  Rust={b}")
                break
    else:
        print(f"  [通过] 列名逐行一致，共 {len(pn)} 列")

    if pv.shape != rv.shape:
        print(f"  [不通过] 形状不同")
        raise SystemExit(1)

    na, nb = np.isnan(pv), np.isnan(rv)
    if not np.array_equal(na, nb):
        ok = False
        diff = int((na != nb).sum())
        print(f"  [不通过] NaN 位置有 {diff} 格不一致")
        for j in range(pv.shape[1]):
            d = int((na[:, j] != nb[:, j]).sum())
            if d:
                print(f"    第 {j} 列 {pn[j]}：{d} 格不同（Python 非空 {int((~na[:,j]).sum())}，Rust 非空 {int((~nb[:,j]).sum())}）")
    else:
        print(f"  [通过] NaN 位置逐格一致（每列非空数 {int((~na[:,0]).sum())} 到 {int((~na).sum(axis=0).max())}）")

    m = ~na & ~nb
    if m.any():
        d = np.abs(pv[m].astype(np.float64) - rv[m].astype(np.float64))
        denom = np.maximum(np.abs(pv[m].astype(np.float64)), 1e-12)
        print(f"  [数值] 两边都有限的位置 {int(m.sum()):,} 格")
        print(f"         最大绝对差 {d.max():.3e}   中位绝对差 {np.median(d):.3e}")
        print(f"         最大相对差 {(d/denom).max():.3e}")
        worst = int(np.argmax(d))
        col = np.where(m)[1][worst]
        print(f"         最差的那一格在第 {col} 列 {pn[col]}")
        if d.max() >= 1e-5:
            ok = False
            print("  [不通过] 最大绝对差超过 1e-5")
        else:
            print("  [通过] 最大绝对差在 1e-5 以内")

    for tag, v in (("Python", pv), ("Rust", rv)):
        e = int(np.isnan(v).all(axis=0).sum())
        c = 0
        for j in range(v.shape[1]):
            col = v[:, j]
            col = col[~np.isnan(col)]
            if col.size and np.unique(col).size == 1:
                c += 1
        print(f"  [{tag}] 整列全空 {e} 列，取值恒定 {c} 列")

    print("  结论：" + ("通过" if ok else "不通过"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
