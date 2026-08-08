//! reciprocity: 互惠性（降维指标.md 三十一）。
//! Reciprocity_i = Σ_j min(S_ij, S_ji) / Σ_j max(S_ij, S_ji)（有向 cnt_t1; 分母为零输出 0）。

use crate::yupei_dist::indicator_ctx::{IndicatorCtx, IndicatorResult};
use rayon::prelude::*;

pub fn name() -> &'static str {
    "reciprocity"
}

pub fn desc() -> &'static str {
    "互惠性: Σmin/Σmax (cnt_t1 有向)"
}

pub fn compute(ctx: &IndicatorCtx) -> Vec<IndicatorResult> {
    let n = ctx.n();
    let mut out = Vec::new();
    for mat in crate::yupei_dist::indicator_ctx::MATRIX_LIST {
        let d = match ctx.matrix(mat) {
            Some(m) => m,
            None => continue,
        };
        let col: Vec<f32> = (0..n)
            .into_par_iter()
            .map(|i| {
                let row = &d[i * n..(i + 1) * n];
                let mut num = 0.0f64;
                let mut den = 0.0f64;
                for j in 0..n {
                    if i == j {
                        continue;
                    }
                    let a = row[j] as f64;
                    let b = d[j * n + i] as f64;
                    num += a.min(b);
                    den += a.max(b);
                }
                if den > 0.0 {
                    (num / den) as f32
                } else {
                    0.0
                }
            })
            .collect();
        out.push(IndicatorResult::new(format!("rec_{mat}_reciprocity"), col));
    }
    out
}
