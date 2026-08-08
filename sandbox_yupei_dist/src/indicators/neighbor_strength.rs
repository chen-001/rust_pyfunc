//! neighbor_strength: 二阶网络效应（降维指标.md 八）。
//! NeighborStrength_i = Σ_j p_ij·Strength_j（Strength_j = cnt_t1 对称行和）。

use crate::indicator_ctx::{IndicatorCtx, IndicatorResult};
use rayon::prelude::*;

pub fn name() -> &'static str {
    "neighbor_strength"
}

pub fn desc() -> &'static str {
    "二阶网络: Σp·Strength (cnt_t1)"
}

pub fn compute(ctx: &IndicatorCtx) -> Vec<IndicatorResult> {
    let n = ctx.n();
    let mut out = Vec::new();
    for mat in crate::indicator_ctx::MATRIX_LIST {
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
                let mut acc = 0.0f64;
                for (j, &v) in row.iter().enumerate() {
                    if v > 0.0 {
                        acc += v as f64 * inv * row_sums[j];
                    }
                }
                acc as f32
            })
            .collect();
        out.push(IndicatorResult::new(format!("nbs_{mat}_neighbor_strength"), col));
    }
    out
}
