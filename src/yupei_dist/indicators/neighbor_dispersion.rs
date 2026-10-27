//! neighbor_dispersion: 邻居分歧（降维指标.md 七）。
//! NeighborReturnDispersion_i = sqrt(Σ_j p_ij·(r_j - mean_i)²),
//! mean_i = Σ_k p_ik·r_k（邻居加权收益均值）。

use crate::yupei_dist::indicator_ctx::{IndicatorCtx, IndicatorResult};
use rayon::prelude::*;

pub fn name() -> &'static str {
    "neighbor_dispersion"
}

pub fn desc() -> &'static str {
    "邻居分歧: sqrt(Σp·(r-mean)²) (cnt_t1)"
}

pub fn compute(ctx: &IndicatorCtx) -> Vec<IndicatorResult> {
    let n = ctx.n();
    let stats = ctx.stats();
    let r: Vec<f64> = stats.iter().map(|s| s.ret).collect();
    let mut out = Vec::new();
    for mat in crate::yupei_dist::indicator_ctx::MATRIX_LIST {
        if !ctx.wants("nbz_", mat) {
            continue;
        }
        let nm = format!("nbz_{mat}_return_dispersion");
        if let Some(k) = &ctx.keep {
            if !k.contains(nm.as_str()) {
                continue;
            }
        }
        let sym = match ctx.symmetric(mat) {
            Some(s) => s,
            None => continue,
        };
        let col: Vec<f32> = (0..n)
            .into_par_iter()
            .map(|i| {
                let row = &sym[i * n..(i + 1) * n];
                // 分母用**正项之和**: urg 族矩阵元素可为负（权重是带符号 z 分数），
                // 带符号行和会让 p = v/total 大于 1，权重和不再归一。
                let total: f64 = row.iter().map(|&v| (v as f64).max(0.0)).sum();
                if total <= 0.0 {
                    return f32::NAN;
                }
                let inv = 1.0 / total;
                let mut mean = 0.0f64;
                for (j, &v) in row.iter().enumerate() {
                    if v > 0.0 {
                        mean += v as f64 * inv * r[j];
                    }
                }
                let mut var = 0.0f64;
                for (j, &v) in row.iter().enumerate() {
                    if v > 0.0 {
                        let d = r[j] - mean;
                        var += v as f64 * inv * d * d;
                    }
                }
                var.sqrt() as f32
            })
            .collect();
        out.push(IndicatorResult::new(nm, col));
    }
    out
}
