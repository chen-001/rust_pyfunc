//! isolation: 孤立度（降维指标.md 三十五）。
//! inv_strength = 1/(对称行和+ε); neg_top5 = -(Top5Mean)。cnt_t1 与 vol_t1 各两个。

use crate::yupei_dist::indicator_ctx::{IndicatorCtx, IndicatorResult};
use rayon::prelude::*;

pub fn name() -> &'static str {
    "isolation"
}

pub fn desc() -> &'static str {
    "孤立度: 1/strength 与 -top5_mean (cnt_t1, vol_t1)"
}

pub fn compute(ctx: &IndicatorCtx) -> Vec<IndicatorResult> {
    let n = ctx.n();
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
        let inv_s: Vec<f32> = row_sums
            .iter()
            .map(|&v| (1.0 / (v + 1e-9)) as f32)
            .collect();
        let neg_top5: Vec<f32> = (0..n)
            .into_par_iter()
            .map(|i| {
                let row = &sym[i * n..(i + 1) * n];
                let mut top: Vec<f32> = row.to_vec();
                top.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
                let s: f32 = top.iter().take(5).sum();
                -(s / 5.0)
            })
            .collect();
        out.push(IndicatorResult::new(format!("iso_{mat}_inv_strength"), inv_s));
        out.push(IndicatorResult::new(format!("iso_{mat}_neg_top5"), neg_top5));
    }
    out
}
