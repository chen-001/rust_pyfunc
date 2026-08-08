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
    "强度: sum/mean/median/q90 (cnt_t1, vol_t1)"
}

pub fn compute(ctx: &IndicatorCtx) -> Vec<IndicatorResult> {
    let n = ctx.n();
    let mut out = Vec::new();
    // 强度谱系矩阵: 6 种权重设计 × 多 τ（含 signed 族的 same 对称分量）
    for mat in crate::yupei_dist::indicator_ctx::MATRIX_LIST {
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
        // 中位数 / q90: 每行排序（行内并行）
        let (medians, q90s): (Vec<f32>, Vec<f32>) = (0..n).into_par_iter().map(|i| {
            let row = &sym[i * n..(i + 1) * n];
            let mut sorted: Vec<f32> = row.to_vec();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let nz = sorted.len();
            let med = if nz > 0 { sorted[nz / 2] } else { 0.0 };
            let q90 = if nz > 0 { sorted[(nz as f64 * 0.9) as usize] } else { 0.0 };
            (med, q90)
        }).unzip();
        let means: Vec<f32> = sums.iter().map(|&s| (s / (n as f64 - 1.0).max(1.0)) as f32).collect();
        let sums32: Vec<f32> = sums.iter().map(|&s| s as f32).collect();
        out.push(IndicatorResult::new(format!("strength_{mat}_sum"), sums32));
        out.push(IndicatorResult::new(format!("strength_{mat}_mean"), means));
        out.push(IndicatorResult::new(format!("strength_{mat}_median"), medians));
        out.push(IndicatorResult::new(format!("strength_{mat}_q90"), q90s));
    }
    out
}
