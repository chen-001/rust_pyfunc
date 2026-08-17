//! network_momentum: 网络动量（降维指标.md 二十八）。
//! NetworkMomentum_i = Σ_j p_ij·r_j,t（r = 日内收益; p 来自 cnt_t1 对称行归一）。

use crate::yupei_dist::indicator_ctx::{IndicatorCtx, IndicatorResult};
use rayon::prelude::*;

pub fn name() -> &'static str {
    "network_momentum"
}

pub fn desc() -> &'static str {
    "网络动量: Σp·r_j (cnt_t1)"
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
                let mut acc = 0.0f64;
                for (j, &v) in row.iter().enumerate() {
                    if v > 0.0 {
                        acc += v as f64 * inv * r[j];
                    }
                }
                acc as f32
            })
            .collect();
        out.push(IndicatorResult::new(
            format!("mom_{mat}_network_momentum"),
            col,
        ));
    }
    out
}
