//! strength: 连接强度（降维指标.md 一）。
//! Strength_i = Σ_j S_ij（对称矩阵行和）; MeanStrength = Σ/(N-1);
//! MedianStrength = 行中位数; Q90 = 行 90 分位。
//! 对称矩阵 = 有向矩阵 + 转置（S_dir + S_dirᵀ），用 ctx.symmetric() 缓存。

use crate::yupei_dist::indicator_ctx::{IndicatorCtx, IndicatorResult};
use rayon::prelude::*;

pub fn name() -> &'static str {
    "strength"
}

pub fn desc() -> &'static str {
    "强度: sum/mean/median/q90 (cnt_t1, rvol_t1)"
}

/// lite 模式下只推幸存列（keep=Some 时按完整列名过滤）。
fn emit(ctx: &IndicatorCtx, out: &mut Vec<IndicatorResult>, name: String, col: Vec<f32>) {
    match &ctx.keep {
        Some(k) if !k.contains(name.as_str()) => {}
        _ => out.push(IndicatorResult::new(name, col)),
    }
}

pub fn compute(ctx: &IndicatorCtx) -> Vec<IndicatorResult> {
    let n = ctx.n();
    let mut out = Vec::new();
    // 强度谱系矩阵: 6 种权重设计 × 多 τ（含 signed 族的 same 对称分量）
    for mat in crate::yupei_dist::indicator_ctx::MATRIX_LIST {
        if !ctx.wants("strength_", mat) {
            continue;
        }
        let sym = match ctx.symmetric(mat) {
            Some(s) => s,
            None => continue,
        };
        let mut sums = Vec::with_capacity(n);
        for i in 0..n {
            let row = &sym[i * n..(i + 1) * n];
            let s: f64 = row.iter().map(|&v| v as f64).sum();
            sums.push(s);
        }
        // 中位数 / q90: 只在**正权邻居**上取（行内其余位置是 0，不是邻居）。
        // 旧实现把整行 n 个元素一起排序，样本被 n-1-k 个 0 稀释：
        // 邻居数 < n/10 时 q90 直接落在 0 上，中位数更早就变 0，列退化成常数。
        let (medians, q90s): (Vec<f32>, Vec<f32>) = (0..n)
            .into_par_iter()
            .map(|i| {
                let row = &sym[i * n..(i + 1) * n];
                let mut nz: Vec<f32> = row.iter().copied().filter(|&v| v > 0.0).collect();
                if nz.is_empty() {
                    return (f32::NAN, f32::NAN);
                }
                nz.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let m = nz.len();
                let med = nz[m / 2];
                let q90 = nz[(((m as f64) * 0.9) as usize).min(m - 1)];
                (med, q90)
            })
            .unzip();
        let means: Vec<f32> = sums
            .iter()
            .map(|&s| (s / (n as f64 - 1.0).max(1.0)) as f32)
            .collect();
        let sums32: Vec<f32> = sums.iter().map(|&s| s as f32).collect();
        emit(ctx, &mut out, format!("strength_{mat}_sum"), sums32);
        emit(ctx, &mut out, format!("strength_{mat}_mean"), means);
        emit(ctx, &mut out, format!("strength_{mat}_median"), medians);
        emit(ctx, &mut out, format!("strength_{mat}_q90"), q90s);
    }
    out
}
