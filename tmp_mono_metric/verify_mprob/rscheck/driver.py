"""Rust（逐字拷贝 src 的 compute_mprob）与 Python 参考实现的逐位对比驱动。

只写 verify_mprob/ 目录；Rust 源码用 sed 从 src/tail_v5_pipeline.rs 逐字抽取。
"""

import subprocess
import sys

import numpy as np

sys.path.insert(0, "/home/chenzongwei/rust_pyfunc/tmp_mono_metric/verify_mprob")
from ref_mprob import compute_mprob, erf_as  # noqa: E402

HERE = "/home/chenzongwei/rust_pyfunc/tmp_mono_metric/verify_mprob/rscheck"


def fmt(v: float) -> str:
    if np.isnan(v):
        return "NaN"
    if np.isinf(v):
        return "inf" if v > 0 else "-inf"
    return repr(float(v))


def build_cases():
    cases = []  # (portf, [cols])
    rng = np.random.default_rng(4242)

    # 随机：n 跨度大、漂移方向两向、尺度不同
    for n in (2, 3, 4, 5, 10, 17, 50, 120, 243, 2424):
        for drift in (0.0, 0.02, -0.02, 0.5, -0.5):
            for scale in (0.01, 1.0, 5.0):
                z = rng.standard_normal((10, n)) * scale + drift * np.arange(10)[:, None]
                cases.append((10, list(z)))
    # 纯噪声（不定向）+ 强均值回复（AR(-1)）
    for _ in range(30):
        cases.append((10, list(rng.standard_normal((10, 137)))))
    ar = rng.standard_normal((10, 137))
    for d in range(10):
        for t in range(1, 137):
            ar[d, t] = -0.98 * ar[d, t - 1] + 0.1 * ar[d, t]
    cases.append((10, list(ar)))

    # 退化：组间恒定阶梯 / 全平
    cases.append((10, [np.full(30, 8.0 * d) for d in range(10)]))
    cases.append((10, [np.full(30, 5.0) for _ in range(10)]))
    cases.append((10, [np.full(30, -8.0 * d) for d in range(10)]))
    # 阶梯 + 1e-7 确定性扰动（正反）
    lad = np.array([[8.0 * d + 1e-7 * ((d * 37 + t * 17) % 11) for t in range(30)] for d in range(10)])
    cases.append((10, list(lad)))
    cases.append((10, list(lad[::-1].copy())))

    # 真实感用例 A（seed 20240501）
    z = np.random.default_rng(20240501).standard_normal((10, 120)) + 0.02 * np.arange(10)[:, None]
    cases.append((10, list(z)))

    # LCG 真实感用例（rust-core 的 n=2424）
    M = (1 << 64) - 1
    st = 0x9E3779B97F4A7C15
    lcg = np.empty((10, 2424))
    for d in range(10):
        for t in range(2424):
            st = (6364136223846793005 * st + 1442695040888963407) & M
            lcg[d, t] = 0.00005 * d + 0.012 * (((st >> 11) / float(1 << 53)) * 2.0 - 1.0)
    cases.append((10, list(lcg)))

    # 病态：NaN / ±Inf / 不齐 / 组数不对 / portf_num 不对 / n=0 / n=1
    base = list(rng.standard_normal((10, 40)))
    for bad_val in (np.nan, np.inf, -np.inf):
        b = [c.copy() for c in base]
        b[3][7] = bad_val
        cases.append((10, b))
    ragged = [c.copy() for c in base]
    ragged[9] = ragged[9][:-1]
    cases.append((10, ragged))
    cases.append((9, [c.copy() for c in base[:9]]))
    cases.append((11, [c.copy() for c in base] + [base[0].copy()]))
    cases.append((5, [c.copy() for c in base]))
    cases.append((0, [c.copy() for c in base]))
    cases.append((10, [np.array([]) for _ in range(10)]))
    cases.append((10, [np.array([1.0]) for _ in range(10)]))
    cases.append((10, [np.array([1.0, 2.0]) for _ in range(10)]))
    return cases


def main():
    cases = build_cases()
    with open(f"{HERE}/cases.txt", "w") as f:
        for portf, cols in cases:
            f.write(f"{portf} {len(cols)}\n")
            for c in cols:
                f.write(str(len(c)) + " " + " ".join(fmt(x) for x in c) + "\n")
    out = subprocess.run([f"{HERE}/mprob_rs", f"{HERE}/cases.txt"], capture_output=True, text=True)
    if out.returncode != 0:
        print("RUST FAILED", out.returncode, out.stderr[:2000])
        return 1
    rvals = [float(x) for x in out.stdout.split()]
    assert len(rvals) == len(cases), (len(rvals), len(cases))

    pyvals = [compute_mprob([list(c) for c in cols], portf, erf_as) for portf, cols in cases]

    n_nan_rust = sum(1 for v in rvals if np.isnan(v))
    n_nan_py = sum(1 for v in pyvals if np.isnan(v))
    mism_nan, mism_bits = [], []
    for k, (a, b) in enumerate(zip(rvals, pyvals)):
        if np.isnan(a) and np.isnan(b):
            continue
        if np.isnan(a) != np.isnan(b):
            mism_nan.append(k)
        elif a != b:
            mism_bits.append((k, a, b, abs(a - b)))

    print(f"用例数 = {len(cases)}   Rust NaN = {n_nan_rust}   Python NaN = {n_nan_py}")
    print(f"NaN 判定不一致 = {len(mism_nan)}   非 NaN 但逐位不同 = {len(mism_bits)}")
    for k in mism_nan[:10]:
        print("  NaN 不一致 case", k, "rust", rvals[k], "py", pyvals[k])
    for k, a, b, d in mism_bits[:10]:
        print("  case", k, "rust", repr(a), "py", repr(b), "diff", d)
    ok = not mism_nan and not mism_bits
    print("结论:", "全部逐位一致（NaN 判定也一致）" if ok else "存在不一致")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
