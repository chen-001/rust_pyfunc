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
        if !ctx.wants("mom_", mat) {
            continue;
        }
        let nm = format!("mom_{mat}_network_momentum");
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
                let mut acc = 0.0f64;
                for (j, &v) in row.iter().enumerate() {
                    if v > 0.0 {
                        acc += v as f64 * inv * r[j];
                    }
                }
                acc as f32
            })
            .collect();
        out.push(IndicatorResult::new(nm, col));
    }
    out
}
