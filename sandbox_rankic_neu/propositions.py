"""数学命题数值验证 (60 天样本, f64 机器精度级)。

P1  cov(a, Mb) == cov(Ma, b) == cov(Ma, Mb)  (X 含截距)
P2  ord(rank_pct(x)) == ord(x)  (f64 域, 引理1: 残差最终rank冗余)
P3  rank(Ma) != M*rank(a) 一般 (rank 与投影不可交换)
P4  OLS 残差的秩对 y 的仿射变换不变
P5  f32 cast 引入伪 tie (方案C 用 f32 resid 的差异来源)
P6  Spearman 域 cov 对称性失效: cov(ord(Ma),ord(r)) vs cov(ord(a),ord(Mr))
"""
import numpy as np
import pandas as pd
import rust_pyfunc as rp
import sys

sys.path.insert(0, "/home/chenzongwei/design_whatever")
import design_whatever as dw
sys.path.insert(0, ".")
from prototype import load_all, read_factor_matrix
import dev_sandbox_rankic as ds


def ord_rank(v):
    idx = np.argsort(v, kind="stable")
    out = np.empty(len(v), dtype=np.int64)
    out[idx] = np.arange(len(v))
    return out


def main():
    n_days = 60
    dates, stocks, ind, restrict, ret_sum1, ret_sum5, barra = load_all(n_days)
    ind = np.ascontiguousarray(ind); restrict = np.ascontiguousarray(restrict)
    barra = np.ascontiguousarray(barra)

    # 取第 40 天截面做命题验证 (该日有效股票 > 10)
    t = 40
    # 构造 X (含截距): [1, 10风格 rank pct]
    from prototype import rank_pct
    barra_r = np.stack([rank_pct(barra[:, :, i].copy()) for i in range(10)], axis=2)
    valid = (restrict[t] == 0) & np.all(np.isfinite(barra_r[t]), axis=1)
    n = valid.sum()
    X = np.concatenate([np.ones((n, 1)), barra_r[t][valid]], axis=1)
    M = np.eye(n) - X @ np.linalg.solve(X.T @ X, X.T)
    # 用真实因子做 a, 真实收益做 b
    raw = read_factor_matrix(0, n_days)
    ranked = pd.DataFrame(raw).rank(axis=1).values
    med = (np.isfinite(ranked).sum(axis=1) + 1.0) / 2.0
    ranked = np.where(np.isnan(ranked), med[:, None], ranked)
    a = ranked[t][valid].astype(np.float64)
    b = ret_sum1[t][valid].astype(np.float64)
    a = (a - a.mean()) / a.std()
    b = (b - b.mean()) / b.std()

    # P1: cov 对称性 (样本协方差, 分母 n)
    cov = lambda u, v: np.mean((u - u.mean()) * (v - v.mean()))
    Ma, Mb = M @ a, M @ b
    p1 = (cov(a, Mb), cov(Ma, b), cov(Ma, Mb))
    print("P1 cov(a,Mb)=%.6e cov(Ma,b)=%.6e cov(Ma,Mb)=%.6e  (max差=%.2e)"
          % (p1 + (max(p1) - min(p1),)))

    # P2: ord(rank_pct(x)) == ord(x)
    x = np.random.RandomState(0).randn(n)
    rp_x = rank_pct(x[None, :])[0]
    p2 = np.array_equal(ord_rank(rp_x), ord_rank(x))
    print("P2 ord(rank_pct(x))==ord(x):", p2)

    # P3: rank(Ma) vs M*rank(a)
    ra = ord_rank(a).astype(np.float64)
    ra = (ra - ra.mean()) / ra.std()
    rMa = ord_rank(Ma).astype(np.float64)
    rMa = (rMa - rMa.mean()) / rMa.std()
    mra = M @ ra
    print("P3 corr(ord(Ma), M*ord(a))=%.5f (一般 != 1, rank 与投影不可交换)"
          % np.corrcoef(rMa, mra)[0, 1])

    # P4: 残差秩对 y 仿射不变
    def resid_of(y):
        beta = np.linalg.solve(X.T @ X, X.T @ y)
        return y - X @ beta
    y1 = np.random.RandomState(1).randn(n)
    y2 = 3.7 * y1 - 1.2
    p4 = np.array_equal(ord_rank(resid_of(y1)), ord_rank(resid_of(y2)))
    print("P4 残差秩仿射不变:", p4)

    # P5: f32 cast 伪 tie
    e = resid_of(a)
    e32 = e.astype(np.float32).astype(np.float64)
    n_tie64 = n - len(np.unique(np.round(e, 12)))
    n_tie32 = n - len(np.unique(e32))
    print("P5 f64残差tie数=%d, f32 cast后tie数=%d (cast引入伪tie)" % (n_tie64, n_tie32))

    # P6: Spearman 域对称性失效 (真实链路口径)
    # A 分子 ~ cov(ord(resid_a), ord(r)); B 分子 ~ cov(ord(a), ord(resid_r))
    # resid_r = 收益侧完整链路的残差 (M b̃)
    b_rank = ord_rank(b).astype(np.float64)
    b_rank = (b_rank - b_rank.mean()) / b_rank.std()
    _, resid_r = ds.neutralize_ret(ret_sum1, ind, restrict, barra, True)
    resid_r = np.asarray(resid_r)[t][valid]
    e_r = ord_rank(resid_r).astype(np.float64)
    e_r = (e_r - e_r.mean()) / e_r.std()
    r_a = ord_rank(a).astype(np.float64)
    r_a = (r_a - r_a.mean()) / r_a.std()
    cA = cov(rMa, b_rank)          # A: ord(Ma) vs ord(r)
    cB = cov(r_a, e_r)             # B: ord(a) vs ord(resid_r)
    cF = cov(rMa, e_r)             # F: ord(Ma) vs ord(resid_r)
    print("P6 Spearman域: cov(ord(Ma),ord(r))=%.4f vs cov(ord(a),ord(Mr))=%.4f vs cov(ord(Ma),ord(Mr))=%.4f"
          % (cA, cB, cF))


if __name__ == "__main__":
    main()
