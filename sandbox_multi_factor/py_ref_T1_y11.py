# -*- coding: utf-8 -*-
"""T1-y11 组合的 Python 参考实现（numpy 向量化），与 sandbox Rust 实现对比验证。

读取 sandbox 导出的网格二进制（signals/market），实现：
- 滚动窗口（200 桶）增量维护 (y,F) 联合矩
- 时间序列单因子/多元暴露
- 横截面单因子/多元回归 + 全部 39 列统计量
输出每股每列均值，与 mf_sandbox compute 输出对比。
"""
import numpy as np

N_BINS = 4740
MIDDAY_BIN = 2400
N_FEATURES = 14
N_STOCKS = 5195
ROLLING_WINDOW = 200
MIN_HISTORY_OBS = 60
MIN_CS_STOCKS = 30

FACTORS = [0, 12, 2]  # T1: 主买占比 / 成交量 / 可观测占比
K = 3

# 列偏移（与 Rust col_* 一致）
def col_beta(k): return 0
def col_beta_t(k): return k
def col_r2(k): return 2*k
def col_adj_r2(k): return 2*k+1
def col_alpha(k): return 2*k+2
def col_lambda(k): return 2*k+3
def col_f_stat(k): return 3*k+3
def col_vif(k): return 3*k+4
def col_cond(k): return 4*k+4
def col_resid(k): return 4*k+5
def col_resid_z(k): return 4*k+6
def col_resid_rank(k): return 4*k+7
def col_resid_abs(k): return 4*k+8
def col_leverage(k): return 4*k+9
def col_cooks(k): return 4*k+10
def col_delta_r2(k): return 4*k+11
def col_resid_improve(k): return 4*k+12
def col_alpha_shift(k): return 4*k+13
def col_resid_corr(k): return 4*k+14
def col_beta_shift(k): return 4*k+15
def col_nested_f(k): return 5*k+15
def col_nested_p(k): return 6*k+15
def col_lambda_shift(k): return 7*k+15
N_COLS = 8*K+15

def load_grid(date, grid_dir):
    sig = np.fromfile(f"{grid_dir}/signals_{date}.bin", dtype=np.float32).reshape(N_FEATURES, N_BINS, N_STOCKS)
    mkt = np.fromfile(f"{grid_dir}/market_{date}.bin", dtype=np.float64).reshape(N_FEATURES, N_BINS)
    return sig, mkt

def rankdata_percentile(vals):
    """百分位排名（平局取中位排名），与 Rust percentile_ranks 一致。"""
    n = len(vals)
    if n == 0:
        return np.array([])
    order = np.argsort(vals, kind="stable")
    sorted_vals = vals[order]
    ranks = np.empty(n)
    i = 0
    denom = max(n - 1, 1)
    while i < n:
        j = i
        while j + 1 < n and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        r = (i + j) / 2.0 / denom
        ranks[order[i:j + 1]] = r
        i = j + 1
    return ranks

def solve_posdef(A, b):
    """批量解正定方程组。A: (N,k,k), b: (N,k)。奇异时返回 nan。"""
    A = A + np.eye(K) * 1e-12  # 对角抖动（Rust cholesky 阈值 1e-14）
    try:
        with np.errstate(all="ignore"):
            x = np.linalg.solve(A, b[:, :, None])[..., 0]
            return x
    except np.linalg.LinAlgError:
        return np.full(b.shape, np.nan)

def main():
    date = 20260717
    grid_dir = "grid_20260717"
    sig, mkt = load_grid(date, grid_dir)
    y = 11  # price_log_return_3s
    Y = sig[y]  # [4740, N_STOCKS]

    # 滚动矩：每 (y, F) 对每股票 6 个量
    # 布局：pairs[f] = (n, sx, sy, sxx, syy, sxy)，每项 [N_STOCKS]
    n = np.zeros(N_STOCKS, dtype=np.int32)
    sx = np.zeros(N_STOCKS)
    sy = np.zeros(N_STOCKS)
    sxx = np.zeros(N_STOCKS)
    syy = np.zeros(N_STOCKS)
    sxy = np.zeros((N_FEATURES, N_STOCKS))  # 每 F 的交叉矩（sx/sxx 每 F 不同）

    # 注意：sx/sxx 依赖 y 有效性（y valid 才累计 x），所以每 F 独立 → sx_f/sxx_f
    sx_f = np.zeros((N_FEATURES, N_STOCKS))
    sxx_f = np.zeros((N_FEATURES, N_STOCKS))

    # 输出时序 [N_COLS, N_BINS, N_STOCKS]
    out = np.full((N_COLS, N_BINS, N_STOCKS), np.nan, dtype=np.float32)

    # F-F 交叉矩（T1: (0,12),(0,2),(12,2)）
    cross_pairs = [(0, 12), (0, 2), (2, 12)]
    cn = np.zeros((3, N_STOCKS), dtype=np.int32)
    csx = np.zeros((3, N_STOCKS))
    csy = np.zeros((3, N_STOCKS))
    csxy = np.zeros((3, N_STOCKS))

    y_valid = np.isfinite(Y)  # [N_BINS, N_STOCKS]
    m_valid = np.isfinite(mkt)  # [14, N_BINS]

    for bin in range(N_BINS):
        if bin == MIDDAY_BIN:
            n[:] = 0; sx[:] = 0; sy[:] = 0; sxx[:] = 0; syy[:] = 0
            sxy[:] = 0; sx_f[:] = 0; sxx_f[:] = 0
            cn[:] = 0; csx[:] = 0; csy[:] = 0; csxy[:] = 0
        # 1) 减旧桶
        if bin >= ROLLING_WINDOW:
            old = bin - ROLLING_WINDOW
            if not (old < MIDDAY_BIN and bin >= MIDDAY_BIN):
                yv = Y[old]
                ymask = y_valid[old]
                sign = -1.0
                n += (ymask.astype(np.int32) * int(sign))  # n 用 int
                sy += sign * np.where(ymask, yv, 0.0)
                syy += sign * np.where(ymask, yv * yv, 0.0)
                for f in range(N_FEATURES):
                    x = mkt[f, old]
                    xv = x if np.isfinite(x) else 0.0
                    ok = ymask & np.isfinite(x)
                    sx_f[f] += sign * np.where(ok, xv, 0.0)
                    sxx_f[f] += sign * np.where(ok, xv * xv, 0.0)
                    sxy[f] += sign * np.where(ok, xv * yv, 0.0)
                # 交叉矩（y valid 且两端有效）
                for ci, (i, j) in enumerate(cross_pairs):
                    xi = mkt[i, old]; xj = mkt[j, old]
                    ok = ymask & np.isfinite(xi) & np.isfinite(xj)
                    cn[ci] += ok.astype(np.int32) * int(sign)
                    csx[ci] += sign * np.where(ok, xi, 0.0)
                    csy[ci] += sign * np.where(ok, xj, 0.0)
                    csxy[ci] += sign * np.where(ok, xi * xj, 0.0)

        # 2) 暴露
        nf = n.astype(np.float64)
        ok_n = nf >= MIN_HISTORY_OBS
        beta1 = np.full((N_FEATURES, N_STOCKS), np.nan)
        for f in range(N_FEATURES):
            sxx_c = sxx_f[f] - sx_f[f] * sx_f[f] / np.maximum(nf, 1)
            syy_c = syy - sy * sy / np.maximum(nf, 1)
            sxy_c = sxy[f] - sx_f[f] * sy / np.maximum(nf, 1)
            ok = ok_n & (sxx_c > 1e-18) & (syy_c > 1e-18)
            beta1[f] = np.where(ok, sxy_c / np.where(sxx_c > 0, sxx_c, np.nan), np.nan)

        ycur = Y[bin]
        # 3) 多元暴露（时间序列）：X'X 与 X'y
        XtX = np.full((N_STOCKS, K, K), np.nan)
        Xty = np.full((N_STOCKS, K), np.nan)
        syy_c_all = syy - sy * sy / np.maximum(nf, 1)
        mbeta = np.full((N_STOCKS, K), np.nan)
        mse = np.full(N_STOCKS, np.nan)
        for i in range(K):
            fi = FACTORS[i]
            Xty[:, i] = sxy[fi] - sx_f[fi] * sy / np.maximum(nf, 1)
            for j in range(K):
                fj = FACTORS[j]
                if i == j:
                    v = sxx_f[fi] - sx_f[fi] * sx_f[fi] / np.maximum(nf, 1)
                else:
                    ci = cross_pairs.index((min(fi, fj), max(fi, fj)))
                    v = csxy[ci] - csx[ci] * csy[ci] / np.maximum(nf, 1)
                XtX[:, i, j] = v
        ok_m = ok_n & (syy_c_all > 1e-18) & np.all(np.isfinite(XtX.reshape(N_STOCKS, -1)), axis=1)
        XtX_ok = np.where(np.isfinite(XtX), XtX, 0.0)
        beta_sol = solve_posdef(XtX_ok, np.where(np.isfinite(Xty), Xty, 0.0))
        sse_ts = np.maximum(syy_c_all - np.sum(beta_sol * np.where(np.isfinite(Xty), Xty, 0.0), axis=1), 0.0)
        dof_ts = nf - K - 1.0
        mse = sse_ts / np.maximum(dof_ts, 1e-12)
        mbeta = np.where(ok_m[:, None], beta_sol, np.nan)

        # 9) 加新桶
        yv_new = Y[bin]
        ymask = y_valid[bin]
        sign = 1.0
        n += ymask.astype(np.int32)
        sy += np.where(ymask, yv_new, 0.0)
        syy += np.where(ymask, yv_new * yv_new, 0.0)
        for f in range(N_FEATURES):
            x = mkt[f, bin]
            xv = x if np.isfinite(x) else 0.0
            ok = ymask & np.isfinite(x)
            sx_f[f] += sign * np.where(ok, xv, 0.0)
            sxx_f[f] += sign * np.where(ok, xv * xv, 0.0)
            sxy[f] += sign * np.where(ok, xv * yv_new, 0.0)
        for ci, (i, j) in enumerate(cross_pairs):
            xi = mkt[i, bin]; xj = mkt[j, bin]
            ok = ymask & np.isfinite(xi) & np.isfinite(xj)
            cn[ci] += ok.astype(np.int32)
            csx[ci] += np.where(ok, xi, 0.0)
            csy[ci] += np.where(ok, xj, 0.0)
            csxy[ci] += np.where(ok, xi * xj, 0.0)

        # 4) 横截面多元
        valid = np.isfinite(ycur) & np.all(np.isfinite(mbeta), axis=1)
        vs = np.where(valid)[0]
        if len(vs) < MIN_CS_STOCKS:
            continue
        yv = ycur[vs].astype(np.float64)
        B = mbeta[vs]  # [n, K]
        ncs = len(vs)
        sbeta = B.sum(axis=0)
        sbb = B.T @ B
        sby = B.T @ yv
        sy_all = yv.sum()
        syy_all = (yv * yv).sum()
        syy_c = syy_all - sy_all * sy_all / ncs
        XtX_cs = sbb - np.outer(sbeta, sbeta) / ncs
        Xty_cs = sby - sbeta * sy_all / ncs
        lam = np.linalg.solve(XtX_cs + np.eye(K) * 1e-12, Xty_cs)
        alpha = sy_all / ncs - lam @ sbeta / ncs
        sse = max(syy_c - lam @ Xty_cs, 0.0)
        dof = ncs - K - 1.0
        r2 = 1.0 - sse / syy_c
        adj_r2 = 1.0 - (1.0 - r2) * (ncs - 1.0) / dof
        f_stat = r2 / (1.0 - r2) * dof / K if 1.0 - r2 > 1e-18 else np.inf
        residual_std = np.sqrt(sse / dof)
        inv = np.linalg.inv(XtX_cs + np.eye(K) * 1e-12)
        # VIF / cond
        sd = np.sqrt(np.maximum(np.diag(XtX_cs), 0.0))
        corr = XtX_cs / np.outer(sd, sd)
        corr_inv = np.linalg.inv(corr + np.eye(K) * 1e-12)
        vif = np.maximum(np.diag(corr_inv), 0.0)
        ev = np.linalg.eigvalsh(corr)
        cond = ev[-1] / ev[0] if ev[0] > 1e-18 else np.inf

        # 5) 主基准单因子横截面 S(y)：y ~ beta1[y]
        byy = beta1[y, vs]
        ok_s = np.isfinite(byy) & np.isfinite(yv)
        if ok_s.sum() >= MIN_CS_STOCKS:
            b = byy[ok_s]; yy = yv[ok_s]
            n_s = ok_s.sum()
            lam_s = (b @ yy - b.sum() * yy.sum() / n_s) / (b @ b - b.sum()**2 / n_s)
            alpha_s = yy.mean() - lam_s * b.mean()
            sse_s = np.sum((yy - alpha_s - lam_s * b)**2)
            r2_s = 1.0 - sse_s / (yy @ yy - yy.sum()**2 / n_s)
        else:
            sse_s = np.nan; r2_s = np.nan; alpha_s = np.nan

        # 6) 辅助单因子（多元有效集上）
        sse_sk = np.full(K, np.nan)
        lam_sk = np.full(K, np.nan)
        for i in range(K):
            fi = FACTORS[i]
            b = beta1[fi, vs]
            okk = np.isfinite(b)
            if okk.sum() >= 30:
                bb = b[okk]; yy2 = yv[okk]
                n2 = okk.sum()
                lam2 = (bb @ yy2 - bb.sum() * yy2.sum() / n2) / (bb @ bb - bb.sum()**2 / n2)
                alpha2 = yy2.mean() - lam2 * bb.mean()
                sse_sk[i] = np.sum((yy2 - alpha2 - lam2 * bb)**2)
                lam_sk[i] = lam2

        # 7) 残差与 per-stock 统计
        fitted = alpha + B @ lam
        resid = yv - fitted
        z = resid / residual_std
        d = B - B.mean(axis=0)
        lev = 1.0 / ncs + np.einsum("ni,ij,nj->n", d, inv, d)
        one_minus_h = np.maximum(1.0 - lev, 1e-12)
        student = z / np.sqrt(one_minus_h)
        cooks = student**2 * lev / (2.0 * one_minus_h) if sse > 0 else np.full(ncs, np.nan)
        ranks = rankdata_percentile(resid)

        # 主基准残差（每股同股票）
        rs_all = np.full(N_STOCKS, np.nan)
        if np.isfinite(sse_s):
            rs_all[vs] = ycur[vs] - (alpha_s + lam_s * beta1[y, vs])
        rm = resid
        rs = rs_all[vs]
        okr = np.isfinite(rs)
        if okr.sum() > 30:
            a_ = rm[okr]; b_ = rs[okr]
            ca = np.corrcoef(a_, b_)[0, 1]
        else:
            ca = np.nan
        delta_r2 = r2 - r2_s
        resid_improve = (sse_s - sse) / sse_s if np.isfinite(sse_s) and sse_s > 0 else np.nan
        alpha_shift = alpha - alpha_s
        # 嵌套 F（每股相同）
        nested_f = np.full(K, np.nan)
        nested_p = np.full(K, np.nan)
        for i in range(K):
            if np.isfinite(sse_sk[i]) and dof > 0:
                num = max(sse_sk[i] - sse, 0.0) / (K - 1.0)
                den = sse / dof
                fv = num / den if den > 0 else np.inf
                nested_f[i] = fv
                # Wilson-Hilferty p 值
                x = fv * (K - 1.0)
                nu = K - 1.0
                zv = ((x / nu)**(1/3) - (1 - 2/(9*nu))) / np.sqrt(2/(9*nu))
                from math import erfc
                nested_p[i] = 0.5 * erfc(zv / np.sqrt(2))

        # 8) 写入时序
        out[col_beta(K):col_beta(K)+K, bin, vs] = mbeta[vs].T
        for i in range(K):
            se_i = np.sqrt(max(inv[i, i], 0.0)) * residual_std
            out[col_beta_t(K)+i, bin, vs] = mbeta[vs, i] / se_i if se_i > 0 else np.nan
            out[col_lambda(K)+i, bin, vs] = lam[i]
            out[col_vif(K)+i, bin, vs] = vif[i]
            bs = beta1[FACTORS[i], vs]
            out[col_beta_shift(K)+i, bin, vs] = mbeta[vs, i] - bs
            out[col_lambda_shift(K)+i, bin, vs] = lam[i] - lam_sk[i]
            out[col_nested_f(K)+i, bin, vs] = nested_f[i]
            out[col_nested_p(K)+i, bin, vs] = nested_p[i]
        out[col_r2(K), bin, vs] = r2
        out[col_adj_r2(K), bin, vs] = adj_r2
        out[col_alpha(K), bin, vs] = alpha
        out[col_f_stat(K), bin, vs] = f_stat
        out[col_cond(K), bin, vs] = cond
        out[col_resid(K), bin, vs] = resid
        out[col_resid_z(K), bin, vs] = z
        out[col_resid_rank(K), bin, vs] = ranks
        out[col_resid_abs(K), bin, vs] = np.abs(resid)
        out[col_leverage(K), bin, vs] = lev
        out[col_cooks(K), bin, vs] = cooks
        out[col_delta_r2(K), bin, vs] = delta_r2
        out[col_resid_improve(K), bin, vs] = resid_improve
        out[col_alpha_shift(K), bin, vs] = alpha_shift
        out[col_resid_corr(K), bin, vs] = ca

    # 每股每列均值
    means = np.full((N_COLS, N_STOCKS), np.nan)
    for c in range(N_COLS):
        col = out[c]
        cnt = np.sum(np.isfinite(col), axis=0)
        sm = np.nansum(col.astype(np.float64), axis=0)
        means[c] = np.where(cnt > 0, sm / np.maximum(cnt, 1), np.nan)
    np.save("py_ref_T1_y11_means.npy", means)
    np.save("py_ref_r2_ts.npy", out[col_r2(K)].astype(np.float64))  # [4740, 5195]
    np.save("py_ref_cond_ts.npy", out[col_cond(K)].astype(np.float64))
    np.save("py_ref_vif_ts.npy", out[col_vif(K)].astype(np.float64))
    print("saved py_ref_T1_y11_means.npy", means.shape)

if __name__ == "__main__":
    main()
