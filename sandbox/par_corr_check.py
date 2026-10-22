"""并行版贪心去相关（dev_sandbox.par_select_by_ic_corr_f32）的对账与测速。

一致性判据：与已安装的 rust_pyfunc.tail_v2_select_by_ic_corr_f32 返回的 dict
逐键逐值比较，列表连顺序一起比（== 对 list 就是逐位比）。

用法：
  python par_corr_check.py            # 跑全部一致性用例 + 启动两种模式的测速子进程
  python par_corr_check.py perf serial|par   # 只跑一种测速（供上面用子进程调起）
"""

import json
import subprocess
import sys
import time

import numpy as np

import rust_pyfunc as rp
import dev_sandbox as sb

SERIAL = rp.tail_v2_select_by_ic_corr_f32
PAR = sb.par_select_by_ic_corr_f32


# ---------------------------------------------------------------- 对账工具


def first_diff(a, b):
    """返回第一个不一致的位置描述；完全一致返回 None。"""
    if type(a) is not type(b):
        return f"类型不同 {type(a)} vs {type(b)}"
    if isinstance(a, dict):
        if sorted(a) != sorted(b):
            return f"键不同 只在串行 {sorted(set(a) - set(b))} 只在并行 {sorted(set(b) - set(a))}"
        for k in a:
            d = first_diff(a[k], b[k])
            if d is not None:
                return f"[{k}] {d}"
        return None
    if isinstance(a, (list, tuple)):
        if len(a) != len(b):
            return f"长度不同 {len(a)} vs {len(b)}"
        for i, (x, y) in enumerate(zip(a, b)):
            d = first_diff(x, y)
            if d is not None:
                return f"下标 {i}: 串行 {x!r} 并行 {y!r}（{d}）"
        return None
    if a != b:
        return f"值不同 {a!r} vs {b!r}"
    return None


def call(fn, mat, kwargs):
    try:
        return ("ok", fn(mat, **kwargs))
    except Exception as exc:  # 报错也要比：两边都得报，且文字一致
        return ("err", f"{type(exc).__name__}: {exc}")


CASES = []


def case(name, mat, **kwargs):
    # on_uncomputable 在两个入口的默认值不同（串行 raise、并行 report，后者按任务规格），
    # 所以每个用例都显式传入，比较的才是同一套参数下的语义。
    kwargs.setdefault("on_uncomputable", "report")
    CASES.append((name, np.ascontiguousarray(mat, dtype=np.float32), kwargs))


def build_cases():
    rng = np.random.default_rng(20261022)

    # 单因子 / 空矩阵
    case("单因子 1×5", rng.standard_normal((1, 5)), threshold=0.6)
    case("空矩阵 0×7", np.zeros((0, 7)), threshold=0.6)

    # 全部保留 / 全部被拒
    case("全部保留 threshold=2.0", rng.standard_normal((40, 50)), threshold=2.0)
    base = rng.standard_normal((1, 60))
    same = np.repeat(base, 12, axis=0)
    case("全部被拒（同一行重复）", same, threshold=0.01)
    case("全部被拒（相关 1.0 门槛 0.6）", same + 0.0, threshold=0.6)

    # 随机小矩阵：n / dates / NaN 比例 / 门槛
    for n in (2, 7, 33, 129):
        for dates in (5, 100, 600):
            for nan in (0.0, 0.15, 0.6):
                for thr in (0.3, 0.6, 0.95):
                    for mor in (0.0, 0.5, 0.9):
                        m = rng.standard_normal((n, dates))
                        if nan:
                            m[rng.random((n, dates)) < nan] = np.nan
                        case(
                            f"随机 n={n} dates={dates} nan={nan} thr={thr} mor={mor}",
                            m,
                            threshold=thr,
                            min_overlap_ratio=mor,
                        )

    # 含 NaN 触发 uncomputable 与 low_overlap
    m = rng.standard_normal((6, 200))
    m[3, :] = np.nan                       # 整行全空 -> common == 0
    m[4, :190] = np.nan                    # 只剩 10 天 -> 交集少
    case("整行全空 + 短覆盖", m, threshold=0.6, min_overlap_ratio=0.5)

    m = rng.standard_normal((5, 200))
    m[2, 5] = 1.0                          # 只有 1 个有效 -> common == 1 < 2
    m[2, 6:] = np.nan
    case("common==1 -> too_few_common_days", m, threshold=0.6)
    case("common==1 min_common_days=1 放行", m, threshold=0.6, min_common_days=1)

    m = rng.standard_normal((4, 120))
    m[2] = 5.0                             # 常量行：交集上方差 0 -> NaN 相关
    case("零方差常量行 -> zero_variance_on_overlap", m, threshold=0.6)

    m = rng.standard_normal((8, 300))
    m[5, 150:] = np.nan                    # 交集占比 0.5，低于 0.6 -> low_overlap
    case("low_overlap 交集占比 0.5", m, threshold=0.6, min_overlap_ratio=0.6)

    # raise 模式：两边都要抛，且文案一致
    m = rng.standard_normal((4, 100))
    m[3, :] = np.nan
    case("raise 模式报错一致", m, threshold=0.6, on_uncomputable="raise")
    case("report 模式不报错", m, threshold=0.6, on_uncomputable="report")

    # 跨块（>2048 行）：块边界等价性
    proto = rng.standard_normal((64, 300))
    idx = rng.integers(0, 64, 5000)
    cross = proto[idx] + 0.35 * rng.standard_normal((5000, 300))
    case("跨块 5000×300（块边界）", cross, threshold=0.6)
    idx2 = rng.integers(0, 8, 4100)
    cross2 = rng.standard_normal((8, 240))[idx2] + 0.9 * rng.standard_normal((4100, 240))
    case("跨块 4100×240 强相关", cross2, threshold=0.6)

    # 深 base（入选数接近真实 hm94 的 1654 量级），且跨块边界
    rng2 = np.random.default_rng(99)
    deep = rng2.standard_normal((1000, 300))
    didx = np.arange(6000) % 1000
    dm = np.ascontiguousarray(deep[didx] + 0.35 * rng2.standard_normal((6000, 300)),
                              dtype=np.float32)
    dm[rng2.random((6000, 300)) < 0.1] = np.nan
    case("深 base 6000×300（约 1000 个入选、含 NaN）", dm, threshold=0.6, min_overlap_ratio=0.5)

    # 非连续内存 / 非法参数：两边都得报同样的错
    wide = np.ascontiguousarray(rng.standard_normal((30, 400)))
    case("非连续输入", wide[:, ::2], threshold=0.6)
    case("threshold=NaN", rng.standard_normal((5, 20)), threshold=float("nan"))
    case("min_overlap_ratio=1.5", rng.standard_normal((5, 20)), threshold=0.6, min_overlap_ratio=1.5)


def run_cases():
    bad = []
    for name, mat, kwargs in CASES:
        s = call(SERIAL, mat, kwargs)
        p = call(PAR, mat, kwargs)
        if s != p:
            bad.append((name, first_diff(s, p) or repr((s, p))))
    return bad


def run_thread_variants():
    """同一输入换线程数、以及连续两次调用，结果必须完全一致。"""
    rng = np.random.default_rng(7)
    proto = rng.standard_normal((32, 400))
    idx = rng.integers(0, 32, 3000)
    mat = np.ascontiguousarray(proto[idx] + 0.3 * rng.standard_normal((3000, 400)), dtype=np.float32)
    ref = SERIAL(mat, 0.6, 2, "report", 0.5)
    outs = {}
    for nt in (0, 1, 2, 8, 50):
        outs[nt] = PAR(mat, 0.6, 2, "report", 0.5, nt)
    again = PAR(mat, 0.6, 2, "report", 0.5, 50)
    return ref, outs, again


# ---------------------------------------------------------------- 测速子进程


def make_matrix(n_rows=77090, n_cols=2415, n_groups=200):
    """构造与 hm94 同量级的矩阵：同组强相关（约 0.9），靠提前 break 控制计算量。"""
    rng = np.random.default_rng(31415926)
    proto = rng.standard_normal((n_groups, n_cols), dtype=np.float32)
    group = np.arange(n_rows) % n_groups
    mat = np.empty((n_rows, n_cols), dtype=np.float32)
    step = 2000
    for s in range(0, n_rows, step):
        e = min(s + step, n_rows)
        noise = rng.standard_normal((e - s, n_cols), dtype=np.float32)
        mat[s:e] = 0.95 * proto[group[s:e]] + 0.312 * noise   # 同组相关约 0.9

    return mat


def vmsize():
    return int([l.split()[1] for l in open("/proc/self/status") if l.startswith("VmHWM")][0])


def perf(mode, n_rows=77090, n_cols=2415, n_groups=200, threshold=0.6):
    mat = make_matrix(n_rows, n_cols, n_groups)
    t0 = time.perf_counter()
    if mode == "serial":
        out = SERIAL(mat, threshold, 2, "report", 0.5)
    else:
        out = PAR(mat, threshold, 2, "report", 0.5, 50)
    secs = time.perf_counter() - t0
    print(json.dumps({
        "mode": mode,
        "shape": [n_rows, n_cols],
        "seconds": round(secs, 3),
        "vm_hwm_kb": vmsize(),
        "n_selected": len(out["selected"]),
        "n_uncomputable": len(out["uncomputable"]),
        "n_low_overlap": len(out["low_overlap"]),
        "selected_head": out["selected"][:5],
        "selected_tail": out["selected"][-5:],
    }))


def perf_sweep(n_rows=77090, n_cols=2415, n_groups=200, threshold=0.6):
    """同一份大矩阵上扫线程数，看扩展性；顺带核对 1 线程与多线程结果完全一样。"""
    mat = make_matrix(n_rows, n_cols, n_groups)
    ref = None
    out = []
    for nt in (1, 4, 8, 16, 32, 50):
        t0 = time.perf_counter()
        rep = PAR(mat, threshold, 2, "report", 0.5, nt)
        secs = time.perf_counter() - t0
        if ref is None:
            ref = rep
        out.append({"n_threads": nt, "seconds": round(secs, 3),
                    "same_as_1thread": rep["selected"] == ref["selected"]
                    and rep["n_dates"] == ref["n_dates"]})
    print(json.dumps({"shape": [n_rows, n_cols], "vm_hwm_kb": vmsize(),
                      "n_selected": len(ref["selected"]), "sweep": out}))


# ---------------------------------------------------------------- 主流程


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "perf":
        perf(sys.argv[2])
        return 0
    if len(sys.argv) > 1 and sys.argv[1] == "sweep":
        perf_sweep()
        return 0

    build_cases()
    bad = run_cases()
    print(f"一致性用例 {len(CASES)} 个，不一致 {len(bad)} 个")
    for name, why in bad:
        print(f"  ✗ {name}: {why}")

    ref, outs, again = run_thread_variants()
    for nt, out in outs.items():
        d = first_diff(ref, out)
        print(f"  线程数 {nt:>2}: {'一致' if d is None else '不一致 ' + d}")
    d2 = first_diff(outs[50], again)
    print(f"  同输入连跑两次(50 线程): {'一致' if d2 is None else '不一致 ' + d2}")

    res = []
    for mode in ("serial", "par"):
        raw = subprocess.run(
            [sys.executable, __file__, "perf", mode],
            capture_output=True, text=True, check=True,
        ).stdout.strip().splitlines()[-1]
        res.append(json.loads(raw))
        print(f"  测速 {mode}: {res[-1]['seconds']}s  VmHWM {res[-1]['vm_hwm_kb']/1024:.0f} MiB  "
              f"入选 {res[-1]['n_selected']}  uncomputable {res[-1]['n_uncomputable']}  "
              f"low_overlap {res[-1]['n_low_overlap']}")
    s, p = res[0], res[1]
    print(f"  加速比 {s['seconds'] / p['seconds']:.2f}×（{s['seconds']}s -> {p['seconds']}s）")
    same = (s["selected_head"] == p["selected_head"] and s["selected_tail"] == p["selected_tail"]
            and s["n_selected"] == p["n_selected"])
    print(f"  两种模式入选个数/首尾一致: {same}")
    raw = subprocess.run([sys.executable, __file__, "sweep"],
                         capture_output=True, text=True, check=True).stdout.strip().splitlines()[-1]
    sw = json.loads(raw)
    print("  线程数扫描（同一份 77090×2415 矩阵）: " +
          "  ".join(f"{r['n_threads']}线 {r['seconds']}s{'' if r['same_as_1thread'] else ' 不一致!'}"
                   for r in sw["sweep"]) +
          f"  VmHWM {sw['vm_hwm_kb']/1024:.0f} MiB")
    ok = (not bad and d2 is None and same
          and all(first_diff(ref, o) is None for o in outs.values())
          and all(r["same_as_1thread"] for r in sw["sweep"]))
    print("结论:", "全部一致" if ok else "有不一致")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
