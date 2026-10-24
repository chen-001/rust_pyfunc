//! w1dist: 行分布对市场合并行分布的 1-Wasserstein 距离（分布形状指标）。
//! 每股: 先把行内非对角正权重归一成行分布 p_ij = w_ij / 行内正权重和
//! （对角剔除, NaN/负值按 0），再在 513 等距分位网格上算行分位对市场分位
//! 的 L1 均值:
//!   w1_i = mean_u |Q_i(u) - Q_mkt(u)|, u ∈ linspace(0,1,513)。
//! Q 用线性插值分位（同 numpy.quantile 默认 method='linear'）。
//! 市场合并分布 = 全部股票行分布 p 的正概率值合并（每矩阵算一次, 全量精确
//! 分位, 非抽样）。是对 p 合并、不是对原始 w 合并——生产 sym 是 S+Sᵀ 未归一,
//! 行尺度悬殊, 直接合并 w 会让市场分布被高强度行主导, w1 退化成行强度代理;
//! 归一成 p 后行尺度被消掉, 量的就是分布形状（双峰/重尾/平坦）。
//! 兄弟列 w1_{mat}_asym: 同积分的带符号版 mean_u (Q_i(u) - Q_mkt(u))（不取绝对值）:
//! 正=行分布比市场重尾/偏强, 负=轻尾/偏弱。注: ∫Q du 恰等于支撑均值,
//! 故 w1_asym ≈ 1/k_i - μ_mkt（k_i 为正伙伴数, μ_mkt 为市场合并分布均值）,
//! 实现仍走 513 网格积分保持与 w1 同口径。
//! 孤立行（非对角正权重和 ≤0）输出 NaN。全部 28 矩阵。
//! 成本: 市场端一次全量排序 O(N² log N²), 行端每行排序 O(N² log N)。

use crate::yupei_dist::indicator_ctx::{IndicatorCtx, IndicatorResult};
use rayon::prelude::*;

const GRID: usize = 513; // linspace(0,1,513), 步长 1/512

/// 线性插值分位数（与 np.quantile linear 一致）: pos = u*(m-1), 相邻两点线性插值。
fn qinterp(sorted: &[f64], u: f64) -> f64 {
    let m = sorted.len();
    if m == 0 {
        return f64::NAN;
    }
    if m == 1 {
        return sorted[0];
    }
    let pos = u * (m - 1) as f64;
    let k = pos.floor() as usize;
    let k = k.min(m - 2);
    let frac = pos - k as f64;
    sorted[k] + frac * (sorted[k + 1] - sorted[k])
}

/// 行内非对角正权重和（对角、NaN、负值都剔除）。
#[inline]
fn row_wsum(row: &[f32], i: usize, n: usize) -> f64 {
    let mut s = 0.0f64;
    for j in 0..n {
        let v = row[j];
        if j != i && v.is_finite() && v > 0.0 {
            s += v as f64;
        }
    }
    s
}

pub fn name() -> &'static str {
    "w1dist"
}

pub fn desc() -> &'static str {
    "W1距离: 行分布p vs 市场合并行分布, L1+带符号 (28矩阵)"
}

pub fn compute(ctx: &IndicatorCtx) -> Vec<IndicatorResult> {
    let n = ctx.n();
    let mut out = Vec::new();
    for mat in crate::yupei_dist::indicator_ctx::MATRIX_LIST {
        if !ctx.wants("w1_", mat) {
            continue;
        }
        let sym = match ctx.symmetric(mat) {
            Some(s) => s,
            None => continue,
        };
        // ---- 市场合并分布: 所有股票行分布 p 的正概率值合并（精确, 非抽样）----
        // 行归一用自算的非对角正权重和, 不用 row_sum_sym（后者会把 NaN 加进去
        // 导致行和为 NaN, 归一后的 p 全是 NaN 再喂给排序会 panic）。
        let mut mkt: Vec<f64> = Vec::with_capacity(n * n / 2);
        for i in 0..n {
            let row = &sym[i * n..(i + 1) * n];
            let wsum = row_wsum(row, i, n);
            if wsum <= 0.0 {
                continue;
            }
            let inv = 1.0 / wsum;
            for (j, &v) in row.iter().enumerate() {
                if j != i && v.is_finite() && v > 0.0 {
                    mkt.push(v as f64 * inv);
                }
            }
        }
        if mkt.is_empty() {
            let nan = vec![f32::NAN; n];
            if ctx.keep.is_none() || ctx.keep.as_ref().unwrap().contains(format!("w1_{mat}").as_str()) {
                out.push(IndicatorResult::new(format!("w1_{mat}"), nan.clone()));
            }
            if ctx.keep.is_none() || ctx.keep.as_ref().unwrap().contains(format!("w1_{mat}_asym").as_str()) {
                out.push(IndicatorResult::new(format!("w1_{mat}_asym"), nan));
            }
            continue;
        }
        mkt.par_sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        let mkt = &mkt[..];
        // 市场 513 网格分位（一次算好, 各行复用）
        let qs: Vec<f64> = (0..GRID)
            .map(|g| qinterp(mkt, g as f64 / (GRID - 1) as f64))
            .collect();
        let qs = &qs[..];

        // ---- 行分布 W1（同一次扫描出 L1 与带符号两列）----
        let both: Vec<(f32, f32)> = (0..n)
            .into_par_iter()
            .map(|i| {
                let row = &sym[i * n..(i + 1) * n];
                let wsum = row_wsum(row, i, n);
                if wsum <= 0.0 {
                    return (f32::NAN, f32::NAN);
                }
                let inv = 1.0 / wsum;
                let mut r: Vec<f64> = Vec::with_capacity(n / 2);
                for (j, &v) in row.iter().enumerate() {
                    if j != i && v.is_finite() && v > 0.0 {
                        r.push(v as f64 * inv);
                    }
                }
                if r.is_empty() {
                    return (f32::NAN, f32::NAN);
                }
                r.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
                let mut s_abs = 0.0f64;
                let mut s_sig = 0.0f64;
                for (g, &qm) in qs.iter().enumerate() {
                    let u = g as f64 / (GRID - 1) as f64;
                    let d = qinterp(&r, u) - qm;
                    s_abs += d.abs();
                    s_sig += d;
                }
                (
                    (s_abs / GRID as f64) as f32,
                    (s_sig / GRID as f64) as f32,
                )
            })
            .collect();
        let w1: Vec<f32> = both.iter().map(|t| t.0).collect();
        let asym: Vec<f32> = both.iter().map(|t| t.1).collect();
        if ctx.keep.is_none() || ctx.keep.as_ref().unwrap().contains(format!("w1_{mat}").as_str()) {
            out.push(IndicatorResult::new(format!("w1_{mat}"), w1));
        }
        if ctx.keep.is_none() || ctx.keep.as_ref().unwrap().contains(format!("w1_{mat}_asym").as_str()) {
            out.push(IndicatorResult::new(format!("w1_{mat}_asym"), asym));
        }
    }
    out
}
