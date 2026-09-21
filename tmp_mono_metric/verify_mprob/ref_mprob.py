"""MPROB 独立参考实现（校验用）。

逐条照 tmp_mono_metric/lead/MPROB_SPEC.md 第 1 节写，不看 Rust 实现。
两个版本：
  - compute_mprob(..., erf=erf_math)  : math.erf，作为「定义」的正确版本
  - compute_mprob(..., erf=erf_as)    : 照抄 A&S 7.1.26 五系数（与 src/copula.rs::erf 同式），
                                        用于和 Rust 逐位/近位对比
另有 compute_mprob_batch：把同一套定义向量化，用于 20000 次空值模拟。

只写 tmp_mono_metric/verify_mprob/ 目录。
"""

import math

import numpy as np

SQRT2 = math.sqrt(2.0)


# ---------- erf 两个版本 ----------

def erf_math(x: float) -> float:
    """定义版。"""
    return math.erf(x)


def erf_as(x: float) -> float:
    """A&S 7.1.26 五系数近似（照抄 src/copula.rs::erf 的表达式顺序）。"""
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    p = 0.3275911

    sign = -1.0 if x < 0.0 else 1.0
    xa = abs(x)
    t = 1.0 / (1.0 + p * xa)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.exp(-xa * xa)
    return sign * y


def _phi(x: float, erf) -> float:
    """标准正态 CDF：0.5 * (1 + erf(x / sqrt(2)))"""
    return 0.5 * (1.0 + erf(x / SQRT2))


# ---------- 参考实现（逐条照 SPEC 第 1 节） ----------

def compute_mprob(group_returns, portf_num: int = 10, erf=erf_math) -> float:
    """SPEC 第 1 节的字面实现。

    group_returns[d][t]：第 d 组（0..9 对应第 1..10 组）的逐日收益。
    """
    # 1
    if portf_num != 10 or len(group_returns) != 10:
        return float("nan")
    # 2
    n = len(group_returns[0])
    if n < 2 or any(len(col) != n for col in group_returns):
        return float("nan")
    # 3
    for col in group_returns:
        for v in col:
            if not math.isfinite(v):
                return float("nan")
    # 4
    r = [sum(col) / n for col in group_returns]
    # 5
    idx = [9 - d for d in range(10)] if r[9] < r[0] else list(range(10))
    # 6
    L = min(n - 1, int(math.floor(4.0 * (n / 100.0) ** (2.0 / 9.0))))
    # 7
    total = 0.0
    for i in range(10):
        for j in range(i + 1, 10):
            d = [group_returns[idx[j]][t] - group_returns[idx[i]][t] for t in range(n)]
            mean = sum(d) / n
            e = [x - mean for x in d]
            gamma0 = sum(x * x for x in e) / n
            s = 0.0
            for k in range(1, L + 1):
                gk = sum(e[t] * e[t - k] for t in range(k, n)) / n
                s += (1.0 - k / (L + 1.0)) * gk
            nw_var = (gamma0 + 2.0 * s) / n
            if not math.isfinite(nw_var) or nw_var <= 0.0:
                total += 0.0
            else:
                t_stat = mean / math.sqrt(nw_var)
                total += 2.0 * _phi(t_stat, erf) - 1.0
    # 8
    return total / 45.0


def _erf_as_vec(x):
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    p = 0.3275911
    xa = np.abs(x)
    t = 1.0 / (1.0 + p * xa)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-xa * xa)
    return np.where(x < 0.0, -y, y)


def compute_mprob_batch(arr, erf_vec=_erf_as_vec, chunk: int = 1000) -> np.ndarray:
    """同一套定义的向量化版本，arr shape=(S,10,T) → (S,)。

    定向规则 r[9] < r[0] 时整条倒序，等价于把 45 对全部取负
    （SPEC 第 1 节「为什么定向规则要和 SSM 一样」自己说明了这一点），
    所以这里只算恒等定向，最后按符号翻转；_selfcheck_batch 会拿字面实现验证该等价。
    """
    arr = np.asarray(arr, dtype=np.float64)
    S, G, T = arr.shape
    assert G == 10 and T >= 2
    L = min(T - 1, int(math.floor(4.0 * (T / 100.0) ** (2.0 / 9.0))))
    means = arr.mean(axis=2)                      # (S,10)
    w = np.array([1.0 - k / (L + 1.0) for k in range(1, L + 1)])

    acc = np.zeros(S)
    for i in range(10):
        for j in range(i + 1, 10):
            d = arr[:, j, :] - arr[:, i, :]       # (S,T)
            mu = d.mean(axis=1)
            e = d - mu[:, None]
            g0 = (e * e).mean(axis=1)
            s = np.zeros(S)
            for k in range(1, L + 1):
                s += w[k - 1] * (e[:, k:] * e[:, : T - k]).sum(axis=1) / T
            nw_var = (g0 + 2.0 * s) / T
            ok = np.isfinite(nw_var) & (nw_var > 0.0)
            t_stat = np.zeros(S)
            t_stat[ok] = mu[ok] / np.sqrt(nw_var[ok])
            contrib = np.where(ok, 2.0 * 0.5 * (1.0 + erf_vec(t_stat / SQRT2)) - 1.0, 0.0)
            acc += contrib
    out = acc / 45.0
    sign = np.where(means[:, 9] < means[:, 0], -1.0, 1.0)
    return out * sign


def l_for(n: int) -> int:
    return min(n - 1, int(math.floor(4.0 * (n / 100.0) ** (2.0 / 9.0))))


# ---------- 自测 ----------

def _selfcheck_batch():
    """向量化版 vs 字面版（含定向翻转）逐条对齐。"""
    rng = np.random.default_rng(7)
    a = rng.standard_normal((6, 10, 137))
    a[0] *= 3.0
    a[0, :, :] += np.linspace(-1.0, 1.0, 10)[:, None]   # 制造 r[9] < r[0] 的倒向情形
    a[1, :, :] += np.linspace(1.0, -1.0, 10)[:, None]   # 制造 r[9] > r[0] 的正向情形
    a[2] = np.tile(np.arange(10.0)[:, None], (1, 137))  # 零方差阶梯
    a[3, :, :] = 5.0
    vec = compute_mprob_batch(a, erf_vec=lambda x: np.vectorize(erf_math)(x))
    lit = np.array([compute_mprob(list(a[s]), 10, erf_math) for s in range(a.shape[0])])
    d = np.abs(vec - lit)
    return float(d.max()), vec, lit


def _selfcheck_null(seed=20240918, n_sims=20000, T=2424):
    """SPEC 第 1 节空值参考分布：20000 次 10 组 iid 标准正态，T=2424，取分位。"""
    rng = np.random.default_rng(seed)
    qs = [5, 25, 50, 75, 90, 95, 99]
    vals = []
    done = 0
    while done < n_sims:
        m = min(1000, n_sims - done)
        a = rng.standard_normal((m, 10, T))
        vals.append(compute_mprob_batch(a))
        done += m
    v = np.concatenate(vals)
    return v, np.percentile(v, qs), float(v.std(ddof=1))


def _realcase_A():
    """rust-core 要的用例 A：seed 20240501，10x120 标准正态 + 0.02*d 漂移。"""
    rng = np.random.default_rng(20240501)
    z = rng.standard_normal((10, 120))
    a = z + 0.02 * np.arange(10)[:, None]
    return a


if __name__ == "__main__":
    import sys

    print("== erf 两版本最大差（|x|<=8，4001 点） ==")
    xs = np.linspace(-8, 8, 4001)
    print("  max|erf_as - math.erf| =", max(abs(erf_as(float(x)) - erf_math(float(x))) for x in xs))

    dm, vec, lit = _selfcheck_batch()
    print("== 向量化 vs 字面实现 ==")
    print("  max|diff| =", dm)
    print("  vec =", np.array2string(vec, precision=17))
    print("  lit =", np.array2string(lit, precision=17))

    print("== L 公式 ==")
    for n in (2, 3, 50, 120, 242, 1212, 2424):
        print(f"  n={n} L={l_for(n)}")

    print("== 用例 B: group_returns[d][t]=d, n=50 ==")
    B = np.tile(np.arange(10.0)[:, None], (1, 50))
    print("  math.erf 版 =", repr(compute_mprob(list(B), 10, erf_math)))
    print("  A&S 版     =", repr(compute_mprob(list(B), 10, erf_as)))

    print("== 用例 C: 全 5.0, n=50 ==")
    C = np.full((10, 50), 5.0)
    print("  math.erf 版 =", repr(compute_mprob(list(C), 10, erf_math)))
    print("  A&S 版     =", repr(compute_mprob(list(C), 10, erf_as)))

    print("== SPEC 用例 1/2 的两种可能读法 ==")
    lad = np.arange(10.0) * 8.0
    one_day = lad[:, None]                       # (10,1) → n=1
    try:
        v1 = compute_mprob(list(one_day), 10, erf_math)
    except Exception as exc:  # noqa: BLE001
        v1 = f"raise {exc!r}"
    print("  单日(10x1)      =", repr(v1))
    multi = np.tile(lad[:, None], (1, 10))       # (10,10) 组间阶梯、时间上恒定
    print("  组间阶梯(10x10) =", repr(compute_mprob(list(multi), 10, erf_math)))
    rev = multi[::-1].copy()
    print("  反向阶梯(10x10) =", repr(compute_mprob(list(rev), 10, erf_math)))

    print("== 用例 A: seed 20240501, 10x120 ==")
    A = _realcase_A()
    va_math = compute_mprob(list(A), 10, erf_math)
    va_as = compute_mprob(list(A), 10, erf_as)
    print("  math.erf 版 =", repr(va_math))
    print("  A&S 版     =", repr(va_as))
    print("  差 =", abs(va_math - va_as))

    if "--null" in sys.argv:
        print("== 空值参考分布 20000 次, T=2424 ==")
        v, q, sd = _selfcheck_null()
        for name, val in zip(["5%", "25%", "50%", "75%", "90%", "95%", "99%"], q):
            print(f"  {name:>4} = {val:+.4f}")
        print("  sd =", round(sd, 4))
        print("  mean =", round(float(v.mean()), 4))
