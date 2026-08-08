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
        let sym = match ctx.symmetric(mat) {
            Some(s) => s,
            None => continue,
        };
        let row_sums = match ctx.row_sum_sym(mat) {
            Some(s) => s,
            None => continue,
        };
        let col: Vec<f32> = (0..n)
            .into_par_iter()
            .map(|i| {
                let total = row_sums[i];
                if total <= 0.0 {
                    return 0.0f32;
                }
                let inv = 1.0 / total;
                let row = &sym[i * n..(i + 1) * n];
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
        out.push(IndicatorResult::new(format!("nbz_{mat}_return_dispersion"), col));
    }
    out
}
