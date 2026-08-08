//! cliqueness: Clique-ness（降维指标.md 十五）。
//! CliqueScore_i = Strength_i × 聚类系数（与 starness 同款聚类系数定义）。

use crate::indicator_ctx::{IndicatorCtx, IndicatorResult};
use rayon::prelude::*;

pub fn name() -> &'static str {
    "cliqueness"
}

pub fn desc() -> &'static str {
    "Clique-ness: Strength × 聚类系数 (cnt_t1)"
}

pub fn compute(ctx: &IndicatorCtx) -> Vec<IndicatorResult> {
    let n = ctx.n();
    let k = 50usize;
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
            let row = &sym[i * n..(i + 1) * n];
            let nbr: Vec<u32> = crate::topk_util::topk_indices(row, k);
            let m = nbr.len();
            if m < 2 {
                return 0.0f32;
            }
            let mut links = 0.0f64;
            let mut maxl = 0.0f64;
            for a in 0..m {
                for b in (a + 1)..m {
                    links += sym[nbr[a] as usize * n + nbr[b] as usize] as f64;
                    maxl += 1.0;
                }
            }
            let cc = if maxl > 0.0 { links / maxl } else { 0.0 };
            (row_sums[i] as f32) * (cc as f32)
        })
        .collect();
    out.push(IndicatorResult::new(format!("clique_{mat}_score"), col));
    }
    out
}
