//! 多因子 CAPM 核心计算（sandbox 版）。
//!
//! - 5 套市场因子模型（T1/T2/T3 三因子 + F1/F2 五因子）
//! - 53 个 (模型, y) 组合（y 清单经真实数据探索校准）
//! - 每组合 39/55 列统计量时序（4740 桶），降维层（21 统计）在主项目实现
//!
//! 计算流程（对每个 y，14 路并行）：
//!   1. 滚动窗口（200 桶）增量维护 (y, F) 联合矩与 F-F 交叉矩
//!   2. 每桶解时间序列多元回归 → 暴露 β_k；单因子回归 → β(y, F_k)
//!   3. 每桶横截面回归（y ~ β 向量）→ λ/R²/残差/VIF/条件数等
//!   4. 主对比（vs y~market(y) 单因子）与因子级对比（vs 每个单因子）

use rayon::prelude::*;

pub const N_BINS: usize = 4_740;
pub const MIDDAY_BIN: usize = 2_400;
pub const N_FEATURES: usize = 14;
pub const ROLLING_WINDOW: usize = 200;
pub const MIN_HISTORY_OBS: u32 = 60;
pub const MIN_CS_STOCKS: usize = 30;
pub const MAX_FACTORS: usize = 5;

// ---------------------------------------------------------------------------
// 模型定义
// ---------------------------------------------------------------------------

pub struct ModelDef {
    pub name: &'static str,
    pub factors: [usize; MAX_FACTORS],
    pub k: usize,
    pub ys: &'static [usize],
}

/// 5 套模型。y 清单 = 数据探索校准后的 53 组合（同源变体 + 稳定高相关剔除，
/// 核心收益 y=11 全保留）。
pub const MODELS: [ModelDef; 5] = [
    ModelDef {
        name: "T1",
        factors: [0, 12, 2, 0, 0],
        k: 3,
        ys: &[0, 1, 2, 3, 5, 6, 8, 9, 11, 12],
    },
    ModelDef {
        name: "T2",
        factors: [1, 9, 6, 0, 0],
        k: 3,
        ys: &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    },
    ModelDef {
        name: "T3",
        factors: [3, 7, 13, 0, 0],
        k: 3,
        ys: &[0, 1, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13],
    },
    ModelDef {
        name: "F1",
        factors: [0, 12, 13, 2, 1],
        k: 5,
        ys: &[0, 1, 2, 3, 5, 6, 8, 11, 12, 13],
    },
    ModelDef {
        name: "F2",
        factors: [0, 12, 2, 6, 3],
        k: 5,
        ys: &[0, 1, 2, 3, 6, 8, 9, 11, 12],
    },
];

/// 组合总数 = 53。
pub const N_COMBOS: usize = 10 + 12 + 12 + 10 + 9;

// ---------------------------------------------------------------------------
// 列布局（每组合，K = 因子数）：共 15 + 8K 列
// ---------------------------------------------------------------------------

#[inline]
pub fn n_cols(k: usize) -> usize {
    15 + 8 * k
}
// 列布局：per-stock 列（4K+6）靠前，共享列（4K+9）靠后。
// per-stock: beta(K) | beta_t(K) | resid 6 | beta_shift(K) | lambda_shift(K)
// shared: r2 | adj_r2 | alpha | lambda(K) | f_stat | vif(K) | cond |
//         delta_r2 | resid_improve | alpha_shift | resid_corr | nested_f(K) | nested_p(K)
#[inline]
pub fn col_beta(k: usize) -> usize {
    0
}
#[inline]
pub fn col_beta_t(k: usize) -> usize {
    k
}
#[inline]
pub fn col_resid(k: usize) -> usize {
    2 * k
}
#[inline]
pub fn col_resid_z(k: usize) -> usize {
    2 * k + 1
}
#[inline]
pub fn col_resid_rank(k: usize) -> usize {
    2 * k + 2
}
#[inline]
pub fn col_resid_abs(k: usize) -> usize {
    2 * k + 3
}
#[inline]
pub fn col_leverage(k: usize) -> usize {
    2 * k + 4
}
#[inline]
pub fn col_cooks(k: usize) -> usize {
    2 * k + 5
}
#[inline]
pub fn col_beta_shift(k: usize) -> usize {
    2 * k + 6
}
#[inline]
pub fn col_lambda_shift(k: usize) -> usize {
    3 * k + 6
}
#[inline]
pub fn col_r2(k: usize) -> usize {
    4 * k + 6
}
#[inline]
pub fn col_adj_r2(k: usize) -> usize {
    4 * k + 7
}
#[inline]
pub fn col_alpha(k: usize) -> usize {
    4 * k + 8
}
#[inline]
pub fn col_lambda(k: usize) -> usize {
    4 * k + 9
}
#[inline]
pub fn col_f_stat(k: usize) -> usize {
    5 * k + 9
}
#[inline]
pub fn col_vif(k: usize) -> usize {
    5 * k + 10
}
#[inline]
pub fn col_cond(k: usize) -> usize {
    6 * k + 10
}
#[inline]
pub fn col_delta_r2(k: usize) -> usize {
    6 * k + 11
}
#[inline]
pub fn col_resid_improve(k: usize) -> usize {
    6 * k + 12
}
#[inline]
pub fn col_alpha_shift(k: usize) -> usize {
    6 * k + 13
}
#[inline]
pub fn col_resid_corr(k: usize) -> usize {
    6 * k + 14
}
#[inline]
pub fn col_nested_f(k: usize) -> usize {
    6 * k + 15
}
#[inline]
pub fn col_nested_p(k: usize) -> usize {
    7 * k + 15
}
// 校验：col_lambda_shift + k == n_cols(k)

/// 组合的列名（stat 部分，与 col_* 布局一致：per-stock 在前，shared 在后）。
pub fn col_names(k: usize, factors: &[usize; MAX_FACTORS]) -> Vec<String> {
    let mut out = Vec::with_capacity(n_cols(k));
    for i in 0..k {
        out.push(format!("beta_f{}", factors[i]));
    }
    for i in 0..k {
        out.push(format!("beta_t_f{}", factors[i]));
    }
    out.push("resid".into());
    out.push("resid_zscore".into());
    out.push("resid_rank".into());
    out.push("resid_abs".into());
    out.push("leverage".into());
    out.push("cooks".into());
    for i in 0..k {
        out.push(format!("beta_shift_f{}", factors[i]));
    }
    for i in 0..k {
        out.push(format!("lambda_shift_f{}", factors[i]));
    }
    out.push("r2".into());
    out.push("adj_r2".into());
    out.push("alpha".into());
    for i in 0..k {
        out.push(format!("lambda_f{}", factors[i]));
    }
    out.push("f_stat".into());
    for i in 0..k {
        out.push(format!("vif_f{}", factors[i]));
    }
    out.push("cond".into());
    out.push("delta_r2".into());
    out.push("resid_improve".into());
    out.push("alpha_shift".into());
    out.push("resid_corr".into());
    for i in 0..k {
        out.push(format!("nested_f_f{}", factors[i]));
    }
    for i in 0..k {
        out.push(format!("nested_p_f{}", factors[i]));
    }
    out
}

// ---------------------------------------------------------------------------
// 滚动矩
// ---------------------------------------------------------------------------

/// (y, F) 对的 6 个联合矩（x = market[F]，y = 个股信号）。
#[derive(Clone, Copy, Default)]
struct Rolling6 {
    n: u32,
    sx: f64,
    sy: f64,
    sxx: f64,
    syy: f64,
    sxy: f64,
}

/// 模型需要的 F_i × F_j 交叉矩（非对角），累计条件 = 个股 y 有效。
#[derive(Clone, Copy, Default)]
struct Cross2 {
    n: u32,
    sx: f64,
    sy: f64,
    sxy: f64,
}

/// 每股票滚动状态：14 个 (y,F) 对 + 若干 F-F 交叉对。
#[derive(Clone, Copy, Default)]
struct StockState {
    pairs: [Rolling6; N_FEATURES],
    cross: [Cross2; 23],
}

/// 从模型定义构建 F-F 交叉对表（去重）。(i, j) 升序。
pub fn build_cross_pairs() -> (Vec<(usize, usize)>, [[i8; N_FEATURES]; N_FEATURES]) {
    let mut set = std::collections::BTreeSet::new();
    for m in MODELS.iter() {
        for a in 0..m.k {
            for b in (a + 1)..m.k {
                let (i, j) = if m.factors[a] < m.factors[b] {
                    (m.factors[a], m.factors[b])
                } else {
                    (m.factors[b], m.factors[a])
                };
                set.insert((i, j));
            }
        }
    }
    let pairs: Vec<(usize, usize)> = set.into_iter().collect();
    let mut table = [[-1i8; N_FEATURES]; N_FEATURES];
    for (idx, &(i, j)) in pairs.iter().enumerate() {
        table[i][j] = idx as i8;
        table[j][i] = idx as i8;
    }
    (pairs, table)
}

/// 增量更新一对矩（加桶 sign=+1 / 减桶 sign=-1）。
#[inline]
fn add_pair(s: &mut Rolling6, x: f64, y: f64, sign: f64) {
    if !x.is_finite() || !y.is_finite() {
        return;
    }
    if sign > 0.0 {
        s.n += 1;
    } else {
        s.n = s.n.saturating_sub(1);
    }
    s.sx += sign * x;
    s.sy += sign * y;
    s.sxx += sign * x * x;
    s.syy += sign * y * y;
    s.sxy += sign * x * y;
}

/// 交叉矩更新：x = market[F_i]，y = market[F_j]，额外要求个股 y 有效。
#[inline]
fn add_cross(s: &mut Cross2, x: f64, y: f64, y_valid: bool, sign: f64) {
    if !y_valid || !x.is_finite() || !y.is_finite() {
        return;
    }
    if sign > 0.0 {
        s.n += 1;
    } else {
        s.n = s.n.saturating_sub(1);
    }
    s.sx += sign * x;
    s.sy += sign * y;
    s.sxy += sign * x * y;
}

/// 单因子暴露（从矩计算）。返回 (beta, alpha, corr, residual_std, n)。
fn exposure(p: &Rolling6) -> Option<(f64, f64, f64, f64, u32)> {
    if p.n < MIN_HISTORY_OBS {
        return None;
    }
    let n = p.n as f64;
    let sxx = p.sxx - p.sx * p.sx / n;
    let syy = p.syy - p.sy * p.sy / n;
    let sxy = p.sxy - p.sx * p.sy / n;
    if sxx <= 1e-18 || syy <= 1e-18 {
        return None;
    }
    let beta = sxy / sxx;
    let alpha = p.sy / n - beta * p.sx / n;
    let corr = (sxy / (sxx * syy).sqrt()).clamp(-1.0, 1.0);
    let sse = (syy - beta * sxy).max(0.0);
    Some((beta, alpha, corr, (sse / (n - 2.0)).sqrt(), p.n))
}

// ---------------------------------------------------------------------------
// 小矩阵工具（K ≤ 5，统一 MAX_FACTORS 步长）
// ---------------------------------------------------------------------------

/// Cholesky 分解（下三角 L，A = L L^T）。加对角抖动防奇异。返回 false = 失败。
fn cholesky(a: &[f64], n: usize, out: &mut [f64]) -> bool {
    out[..n * n].fill(0.0);
    for i in 0..n {
        for j in 0..=i {
            let mut sum = a[i * MAX_FACTORS + j];
            for k in 0..j {
                sum -= out[i * MAX_FACTORS + k] * out[j * MAX_FACTORS + k];
            }
            if i == j {
                if sum <= 1e-14 {
                    return false;
                }
                out[i * MAX_FACTORS + i] = sum.sqrt();
            } else {
                out[i * MAX_FACTORS + j] = sum / out[j * MAX_FACTORS + j];
            }
        }
    }
    true
}

/// 用 Cholesky 因子解 A x = b（A 对称正定）。
fn cholesky_solve(l: &[f64], n: usize, b: &[f64], x: &mut [f64]) {
    let mut z = [0.0f64; MAX_FACTORS];
    for i in 0..n {
        let mut sum = b[i];
        for k in 0..i {
            sum -= l[i * MAX_FACTORS + k] * z[k];
        }
        z[i] = sum / l[i * MAX_FACTORS + i];
    }
    for i in (0..n).rev() {
        let mut sum = z[i];
        for k in (i + 1)..n {
            sum -= l[k * MAX_FACTORS + i] * x[k];
        }
        x[i] = sum / l[i * MAX_FACTORS + i];
    }
}

/// 对称矩阵求逆（高斯消元，n ≤ 5，MAX_FACTORS 步长）。返回 false = 奇异。
fn invert_sym(a: &[f64], n: usize, out: &mut [f64]) -> bool {
    let w = 2 * n;
    let mut m = vec![0.0f64; n * w];
    for i in 0..n {
        for j in 0..n {
            m[i * w + j] = a[i * MAX_FACTORS + j];
        }
        m[i * w + n + i] = 1.0;
    }
    for col in 0..n {
        let mut piv = col;
        let mut best = m[col * w + col].abs();
        for r in col + 1..n {
            let v = m[r * w + col].abs();
            if v > best {
                best = v;
                piv = r;
            }
        }
        if best <= 1e-14 {
            return false;
        }
        if piv != col {
            for c in 0..w {
                m.swap(col * w + c, piv * w + c);
            }
        }
        let d = m[col * w + col];
        for c in 0..w {
            m[col * w + c] /= d;
        }
        for r in 0..n {
            if r != col {
                let f = m[r * w + col];
                if f != 0.0 {
                    for c in 0..w {
                        m[r * w + c] -= f * m[col * w + c];
                    }
                }
            }
        }
    }
    for i in 0..n {
        for j in 0..n {
            out[i * MAX_FACTORS + j] = m[i * w + n + j];
        }
    }
    true
}

/// 对称矩阵特征值（雅可比旋转，n ≤ 5，MAX_FACTORS 步长）。返回特征值（升序）。
fn eigen_sym(a: &[f64], n: usize, vals: &mut [f64]) {
    let mut v = [0.0f64; MAX_FACTORS * MAX_FACTORS];
    for i in 0..n {
        v[i * MAX_FACTORS + i] = 1.0;
    }
    let mut a = a.to_vec();
    for _iter in 0..64 {
        let mut p = 0;
        let mut q = 1;
        let mut max_off = 0.0f64;
        for i in 0..n {
            for j in (i + 1)..n {
                let x = a[i * MAX_FACTORS + j].abs();
                if x > max_off {
                    max_off = x;
                    p = i;
                    q = j;
                }
            }
        }
        if max_off < 1e-14 {
            break;
        }
        let app = a[p * MAX_FACTORS + p];
        let aqq = a[q * MAX_FACTORS + q];
        let apq = a[p * MAX_FACTORS + q];
        // Numerical Recipes 标准参数化：tau = (aqq-app)/(2apq)，t=tan(φ)
        let tau = (aqq - app) / (2.0 * apq);
        let mut t = 1.0 / (tau.abs() + (1.0 + tau * tau).sqrt());
        if tau < 0.0 {
            t = -t;
        }
        let c = 1.0 / (t * t + 1.0).sqrt();
        let s = t * c;
        // 非对角行更新（跳过 p/q），对角元素单独按雅可比公式更新
        for k in 0..n {
            if k == p || k == q {
                continue;
            }
            let akp = a[k * MAX_FACTORS + p];
            let akq = a[k * MAX_FACTORS + q];
            a[k * MAX_FACTORS + p] = c * akp - s * akq;
            a[p * MAX_FACTORS + k] = a[k * MAX_FACTORS + p];
            a[k * MAX_FACTORS + q] = s * akp + c * akq;
            a[q * MAX_FACTORS + k] = a[k * MAX_FACTORS + q];
            let vkp = v[k * MAX_FACTORS + p];
            let vkq = v[k * MAX_FACTORS + q];
            v[k * MAX_FACTORS + p] = c * vkp - s * vkq;
            v[k * MAX_FACTORS + q] = s * vkp + c * vkq;
        }
        // 对角元素：a_pp' = c²a_pp - 2sc·a_pq + s²a_qq
        a[p * MAX_FACTORS + p] = c * c * app - 2.0 * s * c * apq + s * s * aqq;
        a[q * MAX_FACTORS + q] = s * s * app + 2.0 * s * c * apq + c * c * aqq;
        a[p * MAX_FACTORS + q] = 0.0;
        a[q * MAX_FACTORS + p] = 0.0;
    }
    for i in 0..n {
        vals[i] = a[i * MAX_FACTORS + i];
    }
    vals[..n].sort_by(|x, y| x.total_cmp(y));
}

/// 时间序列多元回归：y ~ F1..FK，从矩解出暴露。
/// 返回 (mse, n)。
fn multi_exposure(
    pairs: &[Rolling6; N_FEATURES],
    cross: &[Cross2; 23],
    cross_idx: &[[i8; N_FEATURES]; N_FEATURES],
    factors: &[usize; MAX_FACTORS],
    k: usize,
    out_beta: &mut [f64],
) -> Option<(f64, u32)> {
    let p0 = &pairs[factors[0]];
    if p0.n < MIN_HISTORY_OBS {
        return None;
    }
    let n = p0.n as f64;
    // 注意：同一 y 的所有 pairs[f].sy/syy 相同（add_pair 的 y 相同）
    let mut xtx = [0.0f64; MAX_FACTORS * MAX_FACTORS];
    let mut xty = [0.0f64; MAX_FACTORS];
    for i in 0..k {
        let fi = factors[i];
        let pi = &pairs[fi];
        xty[i] = pi.sxy - pi.sx * p0.sy / n;
        for j in 0..k {
            let fj = factors[j];
            let pj = &pairs[fj];
            let sxy = if i == j {
                pi.sxx
            } else {
                let idx = cross_idx[fi][fj];
                if idx < 0 {
                    return None;
                }
                cross[idx as usize].sxy
            };
            xtx[i * MAX_FACTORS + j] = sxy - pi.sx * pj.sx / n;
        }
    }
    let syy_c = p0.syy - p0.sy * p0.sy / n;
    if syy_c <= 1e-18 {
        return None;
    }
    let mut l = [0.0f64; MAX_FACTORS * MAX_FACTORS];
    if !cholesky(&xtx, k, &mut l) {
        return None;
    }
    cholesky_solve(&l, k, &xty, out_beta);
    let mut sse = syy_c;
    for i in 0..k {
        sse -= out_beta[i] * xty[i];
    }
    if sse < 0.0 {
        sse = 0.0;
    }
    let dof = n - k as f64 - 1.0;
    if dof <= 0.0 {
        return None;
    }
    Some((sse / dof, p0.n))
}

// ---------------------------------------------------------------------------
// 横截面回归
// ---------------------------------------------------------------------------

/// 横截面单因子回归：y ~ beta。返回 (alpha, lambda, r2, sse, n)。
/// valid：可选强制样本集（嵌套检验用，None = 自动筛选）。
fn cs_single(
    y: &[f32],
    beta: &[f64],
    n_stocks: usize,
    force_valid: Option<&[usize]>,
) -> Option<(f64, f64, f64, f64, usize)> {
    let (mut n, mut sx, mut sy, mut sxx, mut syy, mut sxy) =
        (0usize, 0.0f64, 0.0, 0.0, 0.0, 0.0);
    match force_valid {
        Some(valid) => {
            for &s in valid {
                let b = beta[s];
                let v = y[s] as f64;
                if b.is_finite() && v.is_finite() {
                    n += 1;
                    sx += b;
                    sy += v;
                    sxx += b * b;
                    syy += v * v;
                    sxy += b * v;
                }
            }
        }
        None => {
            for s in 0..n_stocks {
                let b = beta[s];
                let v = y[s] as f64;
                if b.is_finite() && v.is_finite() {
                    n += 1;
                    sx += b;
                    sy += v;
                    sxx += b * b;
                    syy += v * v;
                    sxy += b * v;
                }
            }
        }
    }
    if n < MIN_CS_STOCKS {
        return None;
    }
    let nf = n as f64;
    let sxx_c = sxx - sx * sx / nf;
    let syy_c = syy - sy * sy / nf;
    let sxy_c = sxy - sx * sy / nf;
    if sxx_c <= 1e-18 || syy_c <= 1e-18 {
        return None;
    }
    let lambda = sxy_c / sxx_c;
    let alpha = sy / nf - lambda * sx / nf;
    let sse = (syy_c - lambda * sxy_c).max(0.0);
    let r2 = 1.0 - sse / syy_c;
    Some((alpha, lambda, r2, sse, n))
}

pub struct CsResult {
    pub n: usize,
    pub alpha: f64,
    pub lambda: [f64; MAX_FACTORS],
    pub r2: f64,
    pub adj_r2: f64,
    pub f_stat: f64,
    pub sse: f64,
    pub residual_std: f64,
    pub cond: f64,
    pub vif: [f64; MAX_FACTORS],
    pub valid: Vec<usize>,
    pub inv: [f64; MAX_FACTORS * MAX_FACTORS],
    pub syy_c: f64,
    pub beta: [f64; MAX_FACTORS],
    pub se: [f64; MAX_FACTORS],
}

impl Default for CsResult {
    fn default() -> Self {
        CsResult {
            n: 0,
            alpha: f64::NAN,
            lambda: [f64::NAN; MAX_FACTORS],
            r2: f64::NAN,
            adj_r2: f64::NAN,
            f_stat: f64::NAN,
            sse: f64::NAN,
            residual_std: f64::NAN,
            cond: f64::NAN,
            vif: [f64::NAN; MAX_FACTORS],
            valid: Vec::new(),
            inv: [0.0; MAX_FACTORS * MAX_FACTORS],
            syy_c: f64::NAN,
            beta: [f64::NAN; MAX_FACTORS],
            se: [f64::NAN; MAX_FACTORS],
        }
    }
}

/// 横截面多元回归：y ~ beta[K]。成功返回 true，结果写入 out。
#[allow(clippy::too_many_arguments)]
fn cs_multi(y: &[f32], betas: &[f64], k: usize, n_stocks: usize, out: &mut CsResult) -> bool {
    let (mut n, mut sy, mut syy) = (0usize, 0.0f64, 0.0);
    let mut sbeta = [0.0f64; MAX_FACTORS];
    let mut sbb = [0.0f64; MAX_FACTORS * MAX_FACTORS];
    let mut sby = [0.0f64; MAX_FACTORS];
    let mut valid = Vec::with_capacity(n_stocks);
    for s in 0..n_stocks {
        let v = y[s] as f64;
        if !v.is_finite() {
            continue;
        }
        let mut ok = true;
        for i in 0..k {
            if !betas[i * n_stocks + s].is_finite() {
                ok = false;
                break;
            }
        }
        if !ok {
            continue;
        }
        valid.push(s);
        n += 1;
        sy += v;
        syy += v * v;
        for i in 0..k {
            let b = betas[i * n_stocks + s];
            sbeta[i] += b;
            sby[i] += b * v;
            for j in 0..k {
                sbb[i * MAX_FACTORS + j] += b * betas[j * n_stocks + s];
            }
        }
    }
    if n < MIN_CS_STOCKS {
        return false;
    }
    let nf = n as f64;
    let syy_c = syy - sy * sy / nf;
    if syy_c <= 1e-18 {
        return false;
    }
    let mut xtx = [0.0f64; MAX_FACTORS * MAX_FACTORS];
    let mut xty = [0.0f64; MAX_FACTORS];
    for i in 0..k {
        xty[i] = sby[i] - sbeta[i] * sy / nf;
        for j in 0..k {
            xtx[i * MAX_FACTORS + j] = sbb[i * MAX_FACTORS + j] - sbeta[i] * sbeta[j] / nf;
        }
    }
    let mut l = [0.0f64; MAX_FACTORS * MAX_FACTORS];
    if !cholesky(&xtx, k, &mut l) {
        return false;
    }
    let mut beta = [0.0f64; MAX_FACTORS];
    cholesky_solve(&l, k, &xty, &mut beta);
    // 截距：alpha = ȳ - Σ λ_k · β̄_k（中心化回归的截距还原）
    let mut alpha = sy / nf;
    for i in 0..k {
        alpha -= beta[i] * sbeta[i] / nf;
    }
    let mut sse = syy_c;
    for i in 0..k {
        sse -= beta[i] * xty[i];
    }
    if sse < 0.0 {
        sse = 0.0;
    }
    let dof = nf - k as f64 - 1.0;
    let r2 = 1.0 - sse / syy_c;
    let adj_r2 = 1.0 - (1.0 - r2) * (nf - 1.0) / dof;
    let f_stat = if 1.0 - r2 > 1e-18 {
        r2 / (1.0 - r2) * dof / k as f64
    } else {
        f64::INFINITY
    };
    let residual_std = (sse / dof).sqrt();
    let mut inv = [0.0f64; MAX_FACTORS * MAX_FACTORS];
    if !invert_sym(&xtx, k, &mut inv) {
        return false;
    }
    // VIF：β 相关矩阵求逆的对角
    let mut corr = [0.0f64; MAX_FACTORS * MAX_FACTORS];
    let mut sd = [0.0f64; MAX_FACTORS];
    for i in 0..k {
        let v = xtx[i * MAX_FACTORS + i].max(0.0).sqrt();
        sd[i] = if v > 1e-18 { v } else { f64::NAN };
    }
    for i in 0..k {
        for j in 0..k {
            let d = sd[i] * sd[j];
            corr[i * MAX_FACTORS + j] = if d > 0.0 {
                xtx[i * MAX_FACTORS + j] / d
            } else {
                0.0
            };
        }
    }
    let mut corr_inv = [0.0f64; MAX_FACTORS * MAX_FACTORS];
    let mut vif = [f64::NAN; MAX_FACTORS];
    if invert_sym(&corr, k, &mut corr_inv) {
        for i in 0..k {
            vif[i] = corr_inv[i * MAX_FACTORS + i].max(0.0);
        }
    }
    let mut ev = [0.0f64; MAX_FACTORS];
    eigen_sym(&corr, k, &mut ev);
    let cond = if ev[0] > 1e-18 {
        ev[k - 1] / ev[0]
    } else {
        f64::INFINITY
    };
    out.n = n;
    out.alpha = alpha;
    for i in 0..k {
        out.lambda[i] = beta[i];
        out.vif[i] = vif[i];
    }
    out.r2 = r2;
    out.adj_r2 = adj_r2;
    out.f_stat = f_stat;
    out.sse = sse;
    out.residual_std = residual_std;
    out.cond = cond;
    out.valid = valid;
    out.inv = inv;
    out.syy_c = syy_c;
    out.beta = beta;
    for i in 0..k {
        let se = inv[i * MAX_FACTORS + i].max(0.0).sqrt() * residual_std;
        out.se[i] = se;
    }
    true
}

/// Wilson-Hilferty 近似：F 分布单侧 p 值（d2 大时精确）。
fn f_pvalue(f: f64, d1: f64, _d2: f64) -> f64 {
    if !f.is_finite() || f <= 0.0 {
        return f64::NAN;
    }
    let x = f * d1; // ~ chi2(d1)（d2→∞ 近似）
    let nu = d1;
    let z = ((x / nu).cbrt() - (1.0 - 2.0 / (9.0 * nu))) / (2.0 / (9.0 * nu)).sqrt();
    let zz = z.abs();
    let p = 0.231_641_9;
    let b1 = 0.319_381_530;
    let b2 = -0.356_563_782;
    let b3 = 1.781_477_937;
    let b4 = -1.821_255_978;
    let b5 = 1.330_274_429;
    let u = 1.0 / (1.0 + p * zz);
    let phi = (-0.5 * zz * zz).exp() / (2.0 * std::f64::consts::PI).sqrt();
    let upper =
        phi * (b1 * u + b2 * u.powi(2) + b3 * u.powi(3) + b4 * u.powi(4) + b5 * u.powi(5));
    upper.clamp(0.0, 1.0)
}

/// 残差百分位排名（平局取中位排名）。
fn percentile_ranks_in_place(values: &mut [(usize, f64)], ranks: &mut [f64]) {
    values.sort_unstable_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
    ranks.fill(f64::NAN);
    if values.is_empty() {
        return;
    }
    let denominator = values.len().saturating_sub(1).max(1) as f64;
    let mut start = 0;
    while start < values.len() {
        let mut end = start + 1;
        while end < values.len() && values[end].1 == values[start].1 {
            end += 1;
        }
        let rank = (start + end - 1) as f64 * 0.5 / denominator;
        for &(stock, _) in &values[start..end] {
            ranks[stock] = rank;
        }
        start = end;
    }
}

// ---------------------------------------------------------------------------
// 主计算：按 y 分组（14 路并行）
// ---------------------------------------------------------------------------

/// 一个 (模型, y) 组合的时序缓冲。
/// per-stock 列（β、残差系列等）按 [股票] 展开；共享列（R²、λ、VIF 等横截面
/// 全局量，每股相同）只存一份 [N_BINS]，21 统计阶段广播。
pub struct ComboBuf {
    pub n_per_stock: usize, // per-stock 列数（列索引靠前）
    pub n_shared: usize,    // 共享列数
    pub n_stocks: usize,
    pub per_stock: Vec<f32>, // [n_per_stock][N_BINS][n_stocks]
    pub shared: Vec<f32>,    // [n_shared][N_BINS]
}

impl Default for ComboBuf {
    fn default() -> Self {
        ComboBuf {
            n_per_stock: 0,
            n_shared: 0,
            n_stocks: 0,
            per_stock: Vec::new(),
            shared: Vec::new(),
        }
    }
}

fn madvise_huge(buf: &mut Vec<f32>) {
    if buf.is_empty() {
        return;
    }
    // 分配后、touch 前提示内核用透明大页（地址需页对齐，Vec 指针向下取整）
    unsafe {
        let raw = buf.as_mut_ptr() as usize;
        let page = 4096usize;
        let aligned = (raw + page - 1) & !(page - 1);
        let adj = aligned - raw;
        if buf.len() * 4 > adj {
            libc::madvise(
                aligned as *mut libc::c_void,
                buf.len() * 4 - adj,
                libc::MADV_HUGEPAGE,
            );
        }
    }
}

impl ComboBuf {
    fn new(n_per_stock: usize, n_shared: usize, n_stocks: usize) -> Self {
        let mut per_stock = Vec::with_capacity(n_per_stock * N_BINS * n_stocks);
        madvise_huge(&mut per_stock);
        per_stock.resize(n_per_stock * N_BINS * n_stocks, f32::NAN);
        let mut shared = Vec::with_capacity(n_shared * N_BINS);
        madvise_huge(&mut shared);
        shared.resize(n_shared * N_BINS, f32::NAN);
        ComboBuf {
            n_per_stock,
            n_shared,
            n_stocks,
            per_stock,
            shared,
        }
    }
    /// 总列数（per-stock + shared，与 col_names 顺序一致）。
    #[inline]
    fn cols(&self) -> usize {
        self.n_per_stock + self.n_shared
    }
    /// 写 per-stock 列（col < n_per_stock）。
    #[inline]
    fn write_ps(&mut self, col: usize, bin: usize, stock: usize, v: f32) {
        self.per_stock[((col * N_BINS + bin) * self.n_stocks) + stock] = v;
    }
    /// 写共享列（col 为全局列索引，n_per_stock <= col < cols）。
    #[inline]
    fn write_sh(&mut self, col: usize, bin: usize, v: f32) {
        self.shared[((col - self.n_per_stock) * N_BINS) + bin] = v;
    }
    /// 读任意列（全局列索引）指定桶、股票的值。
    #[inline]
    pub fn read(&self, col: usize, bin: usize, stock: usize) -> f32 {
        if col < self.n_per_stock {
            self.per_stock[((col * N_BINS + bin) * self.n_stocks) + stock]
        } else {
            self.shared[((col - self.n_per_stock) * N_BINS) + bin]
        }
    }
}

/// 每路输出：该 y 涉及的组合缓冲（模型序）。
pub struct RouteResult {
    pub y: usize,
    pub combos: Vec<ComboBuf>,
    pub model_indices: Vec<usize>,
}

/// 预分配全部 14 路组合缓冲（页错误与读盘/其他计算重叠）。
pub fn prealloc_combos(n_stocks: usize) -> Vec<Vec<ComboBuf>> {
    let mut y_models: [Vec<usize>; N_FEATURES] = std::array::from_fn(|_| Vec::new());
    for (mi, m) in MODELS.iter().enumerate() {
        for &y in m.ys {
            y_models[y].push(mi);
        }
    }
    (0..N_FEATURES)
        .into_par_iter()
        .map(|y| {
            y_models[y]
                .iter()
                .map(|&mi| {
                    let k = MODELS[mi].k;
                    ComboBuf::new(4 * k + 6, 4 * k + 9, n_stocks)
                })
                .collect()
        })
        .collect()
}

/// 处理单个 (模型, y) 组合：独立滚动矩 + 桶循环。返回组合缓冲。
#[allow(clippy::too_many_arguments)]
fn compute_one_combo(
    signals: &[f32],
    market: &[f64],
    n_stocks: usize,
    y: usize,
    mi: usize,
    mut buf: ComboBuf,
) -> ComboBuf {
    let m = &MODELS[mi];
    let k = m.k;
    let (cross_pairs, cross_idx) = build_cross_pairs();
    let mut states = vec![StockState::default(); n_stocks];
    let mut beta1 = vec![f64::NAN; N_FEATURES * n_stocks];
    let mut mbeta = vec![f64::NAN; MAX_FACTORS * n_stocks];
    let mut cs = CsResult::default();
    let mut rank_pairs: Vec<(usize, f64)> = Vec::with_capacity(n_stocks);
    let mut ranks = vec![f64::NAN; n_stocks];
    let mut tmp_resid = vec![f64::NAN; n_stocks];
    let mut tmp_z = vec![f64::NAN; n_stocks];
    let mut tmp_h = vec![f64::NAN; n_stocks];
    let mut tmp_cooks = vec![f64::NAN; n_stocks];

    let ysig_base = y * N_BINS * n_stocks;

    for bin in 0..N_BINS {
        if bin == MIDDAY_BIN {
            states.fill(StockState::default());
        }
        if bin >= ROLLING_WINDOW {
            let old = bin - ROLLING_WINDOW;
            if !(old < MIDDAY_BIN && bin >= MIDDAY_BIN) {
                let sign = -1.0;
                for s in 0..n_stocks {
                    let yv = signals[ysig_base + old * n_stocks + s] as f64;
                    let st = &mut states[s];
                    for f in 0..N_FEATURES {
                        add_pair(&mut st.pairs[f], market[f * N_BINS + old], yv, sign);
                    }
                    if yv.is_finite() {
                        for (ci, &(i, j)) in cross_pairs.iter().enumerate() {
                            add_cross(
                                &mut st.cross[ci],
                                market[i * N_BINS + old],
                                market[j * N_BINS + old],
                                true,
                                sign,
                            );
                        }
                    }
                }
            }
        }
        for s in 0..n_stocks {
            let st = &states[s];
            for f in 0..N_FEATURES {
                beta1[f * n_stocks + s] = match exposure(&st.pairs[f]) {
                    Some((b, _, _, _, _)) => b,
                    None => f64::NAN,
                };
            }
        }
        let ycur_base = ysig_base + bin * n_stocks;
        for s in 0..n_stocks {
            let mut bvec = [0.0f64; MAX_FACTORS];
            if let Some((_mse, _n)) = multi_exposure(
                &states[s].pairs,
                &states[s].cross,
                &cross_idx,
                &m.factors,
                k,
                &mut bvec,
            ) {
                for i in 0..k {
                    mbeta[i * n_stocks + s] = bvec[i];
                }
            } else {
                for i in 0..k {
                    mbeta[i * n_stocks + s] = f64::NAN;
                }
            }
        }
        let ycur = &signals[ycur_base..ycur_base + n_stocks];
        let beta_yy = &beta1[y * n_stocks..(y + 1) * n_stocks];
        let s_fit = cs_single(ycur, beta_yy, n_stocks, None);
        let ok = cs_multi(ycur, &mbeta, k, n_stocks, &mut cs);
        if !ok {
            // 无有效横截面：仍推进滚动窗口
            let sign = 1.0;
            for s in 0..n_stocks {
                let yv = signals[ysig_base + bin * n_stocks + s] as f64;
                let st = &mut states[s];
                for f in 0..N_FEATURES {
                    add_pair(&mut st.pairs[f], market[f * N_BINS + bin], yv, sign);
                }
                if yv.is_finite() {
                    for (ci, &(i, j)) in cross_pairs.iter().enumerate() {
                        add_cross(
                            &mut st.cross[ci],
                            market[i * N_BINS + bin],
                            market[j * N_BINS + bin],
                            true,
                            sign,
                        );
                    }
                }
            }
            continue;
        }
        // ---- 统计量计算与写入（与组合循环体一致） ----
        let mut sse_sk = [f64::NAN; MAX_FACTORS];
        let mut lam_sk = [f64::NAN; MAX_FACTORS];
        for i in 0..k {
            let fi = m.factors[i];
            let b = &beta1[fi * n_stocks..(fi + 1) * n_stocks];
            if let Some((_a, l, _r2, sse, _n)) = cs_single(ycur, b, n_stocks, Some(&cs.valid)) {
                sse_sk[i] = sse;
                lam_sk[i] = l;
            }
        }
        let n = cs.n;
        let nf = n as f64;
        let dof = nf - k as f64 - 1.0;
        let (r2_s, sse_s, alpha_s) = match s_fit {
            Some((a, _l, r2, sse, _n)) => (r2, sse, a),
            None => (f64::NAN, f64::NAN, f64::NAN),
        };
        let residual_std = cs.residual_std;
        let mut bmean = [0.0f64; MAX_FACTORS];
        for i in 0..k {
            let mut sm = 0.0;
            for &s in &cs.valid {
                sm += mbeta[i * n_stocks + s];
            }
            bmean[i] = sm / nf;
        }
        let mut sum_m = 0.0f64;
        let mut sum_s = 0.0f64;
        let mut sum_mm = 0.0f64;
        let mut sum_ss = 0.0f64;
        let mut sum_ms = 0.0f64;
        let mut cnt_r = 0usize;
        rank_pairs.clear();
        for &s in &cs.valid {
            let yv = ycur[s] as f64;
            let mut fitted = cs.alpha;
            for i in 0..k {
                fitted += cs.lambda[i] * mbeta[i * n_stocks + s];
            }
            let resid = yv - fitted;
            rank_pairs.push((s, resid));
            if let Some((a, l, _, _, _)) = s_fit {
                let rs = yv - (a + l * beta1[y * n_stocks + s]);
                sum_m += resid;
                sum_s += rs;
                sum_mm += resid * resid;
                sum_ss += rs * rs;
                sum_ms += resid * rs;
                cnt_r += 1;
            }
        }
        percentile_ranks_in_place(&mut rank_pairs, &mut ranks);
        let resid_corr = if cnt_r > 30 {
            let nf2 = cnt_r as f64;
            let vm = sum_mm - sum_m * sum_m / nf2;
            let vs = sum_ss - sum_s * sum_s / nf2;
            let vms = sum_ms - sum_m * sum_s / nf2;
            if vm > 1e-18 && vs > 1e-18 {
                (vms / (vm * vs).sqrt()).clamp(-1.0, 1.0)
            } else {
                f64::NAN
            }
        } else {
            f64::NAN
        };
        let b0 = col_beta(k);
        let bt0 = col_beta_t(k);
        let r0 = col_resid(k);
        let z0 = col_resid_z(k);
        let rk0 = col_resid_rank(k);
        let ab0 = col_resid_abs(k);
        let lv0 = col_leverage(k);
        let ck0 = col_cooks(k);
        let bs0 = col_beta_shift(k);
        let ls0 = col_lambda_shift(k);
        tmp_resid.fill(f64::NAN);
        tmp_z.fill(f64::NAN);
        tmp_h.fill(f64::NAN);
        tmp_cooks.fill(f64::NAN);
        for &s in &cs.valid {
            let yv = ycur[s] as f64;
            let mut fitted = cs.alpha;
            let mut h = 1.0 / nf;
            for i in 0..k {
                let b = mbeta[i * n_stocks + s];
                fitted += cs.lambda[i] * b;
            }
            let mut d = [0.0f64; MAX_FACTORS];
            for i in 0..k {
                d[i] = mbeta[i * n_stocks + s] - bmean[i];
            }
            for i in 0..k {
                for j in 0..k {
                    h += d[i] * cs.inv[i * MAX_FACTORS + j] * d[j];
                }
            }
            let resid = yv - fitted;
            let z = if residual_std > 0.0 {
                resid / residual_std
            } else {
                f64::NAN
            };
            let one_minus_h = (1.0 - h).max(1e-12);
            let studentized = if z.is_finite() {
                z / one_minus_h.sqrt()
            } else {
                f64::NAN
            };
            let cooks = if cs.sse > 0.0 && studentized.is_finite() {
                studentized * studentized * h / (2.0 * one_minus_h)
            } else {
                f64::NAN
            };
            tmp_resid[s] = resid;
            tmp_z[s] = z;
            tmp_h[s] = h;
            tmp_cooks[s] = cooks;
        }
        for i in 0..k {
            let fi = m.factors[i];
            let se = cs.se[i];
            for &s in &cs.valid {
                let bi = mbeta[i * n_stocks + s];
                buf.write_ps(b0 + i, bin, s, bi as f32);
                let t = if se > 0.0 && bi.is_finite() {
                    bi / se
                } else {
                    f64::NAN
                };
                buf.write_ps(bt0 + i, bin, s, t as f32);
                let bs = beta1[fi * n_stocks + s];
                buf.write_ps(bs0 + i, bin, s, (bi - bs) as f32);
                let ls = lam_sk[i];
                buf.write_ps(ls0 + i, bin, s, (cs.lambda[i] - ls) as f32);
            }
        }
        for &s in &cs.valid {
            buf.write_ps(r0, bin, s, tmp_resid[s] as f32);
            buf.write_ps(z0, bin, s, tmp_z[s] as f32);
            buf.write_ps(rk0, bin, s, ranks[s] as f32);
            buf.write_ps(ab0, bin, s, tmp_resid[s].abs() as f32);
            buf.write_ps(lv0, bin, s, tmp_h[s] as f32);
            buf.write_ps(ck0, bin, s, tmp_cooks[s] as f32);
        }
        buf.write_sh(col_r2(k), bin, cs.r2 as f32);
        buf.write_sh(col_adj_r2(k), bin, cs.adj_r2 as f32);
        buf.write_sh(col_alpha(k), bin, cs.alpha as f32);
        buf.write_sh(col_f_stat(k), bin, cs.f_stat as f32);
        buf.write_sh(col_cond(k), bin, cs.cond as f32);
        for i in 0..k {
            buf.write_sh(col_lambda(k) + i, bin, cs.lambda[i] as f32);
            buf.write_sh(col_vif(k) + i, bin, cs.vif[i] as f32);
        }
        let delta_r2 = cs.r2 - r2_s;
        let resid_improve = if sse_s > 0.0 && sse_s.is_finite() {
            (sse_s - cs.sse) / sse_s
        } else {
            f64::NAN
        };
        let alpha_shift = cs.alpha - alpha_s;
        buf.write_sh(col_delta_r2(k), bin, delta_r2 as f32);
        buf.write_sh(col_resid_improve(k), bin, resid_improve as f32);
        buf.write_sh(col_alpha_shift(k), bin, alpha_shift as f32);
        buf.write_sh(col_resid_corr(k), bin, resid_corr as f32);
        for i in 0..k {
            let sse_sk_i = sse_sk[i];
            if sse_sk_i.is_finite() && dof > 0.0 {
                let num = (sse_sk_i - cs.sse).max(0.0) / (k as f64 - 1.0);
                let den = cs.sse / dof;
                let f = if den > 0.0 { num / den } else { f64::INFINITY };
                buf.write_sh(col_nested_f(k) + i, bin, f as f32);
                buf.write_sh(
                    col_nested_p(k) + i,
                    bin,
                    f_pvalue(f, k as f64 - 1.0, dof) as f32,
                );
            } else {
                buf.write_sh(col_nested_f(k) + i, bin, f32::NAN);
                buf.write_sh(col_nested_p(k) + i, bin, f32::NAN);
            }
        }
        // 加新桶
        let sign = 1.0;
        for s in 0..n_stocks {
            let yv = signals[ysig_base + bin * n_stocks + s] as f64;
            let st = &mut states[s];
            for f in 0..N_FEATURES {
                add_pair(&mut st.pairs[f], market[f * N_BINS + bin], yv, sign);
            }
            if yv.is_finite() {
                for (ci, &(i, j)) in cross_pairs.iter().enumerate() {
                    add_cross(
                        &mut st.cross[ci],
                        market[i * N_BINS + bin],
                        market[j * N_BINS + bin],
                        true,
                        sign,
                    );
                }
            }
        }
    }
    buf
}

/// 主入口：全市场 14 指标网格 + 市场均值 → 53 组合时序缓冲。
/// signals 布局 [feature][bin][stock]（f32），market 布局 [feature][bin]（f64）。
/// prealloc 由 prealloc_combos 预分配（可与读盘重叠）。
pub fn compute_route_timeseries(
    signals: &[f32],
    market: &[f64],
    n_stocks: usize,
    mut prealloc: Vec<Vec<ComboBuf>>,
) -> Vec<RouteResult> {
    let mut y_models: [Vec<usize>; N_FEATURES] = std::array::from_fn(|_| Vec::new());
    for (mi, m) in MODELS.iter().enumerate() {
        for &y in m.ys {
            y_models[y].push(mi);
        }
    }
    // 53 个 (y, 模型) 任务并行，每个任务独立滚动矩
    let mut tasks: Vec<(usize, usize, usize, ComboBuf)> = Vec::new();
    for y in 0..N_FEATURES {
        for (ci, &mi) in y_models[y].iter().enumerate() {
            let buf = std::mem::take(&mut prealloc[y][ci]);
            tasks.push((y, mi, ci, buf));
        }
    }
    let results: Vec<(usize, usize, usize, ComboBuf)> = tasks
        .into_par_iter()
        .map(|(y, mi, ci, buf)| {
            let buf = compute_one_combo(signals, market, n_stocks, y, mi, buf);
            (y, mi, ci, buf)
        })
        .collect();
    // 重组回 [y][combo] 结构
    let mut out: Vec<RouteResult> = (0..N_FEATURES)
        .map(|y| RouteResult {
            y,
            combos: Vec::with_capacity(y_models[y].len()),
            model_indices: y_models[y].clone(),
        })
        .collect();
    for (y, _mi, ci, buf) in results {
        out[y].combos.push(buf);
        // 保持模型序：结果按任务序收集，任务序 = y 内 ci 升序 ✓
        let _ = ci;
    }
    out
}
