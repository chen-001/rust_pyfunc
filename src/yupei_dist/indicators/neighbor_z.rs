//! neighbor_z: 邻居质量（降维指标.md 六）。
//! p_ij = S_ij/行和(对称); NeighborZ_i = Σ_j p_ij·Z_j（邻居的加权均值）。
//! 口径说明: 输出**没有做 z 标准化**，Z 是原始量:
//!   return = 日内收益率, turnover = 总成交量（股数，不是换手率）,
//!   volatility = 30s 桶收益波动, imbalance = 订单失衡, size = 成交额。
//! 所以 turnover / size 是日频水平量，量纲随全市场成交量漂移，跨日不可比。

use crate::yupei_dist::indicator_ctx::{IndicatorCtx, IndicatorResult};
use rayon::prelude::*;

pub fn name() -> &'static str {
    "neighbor_z"
}

pub fn desc() -> &'static str {
    "邻居质量: Σp·Z, Z∈{return,turnover,volatility,imbalance,size} (cnt_t1)"
}

pub fn compute(ctx: &IndicatorCtx) -> Vec<IndicatorResult> {
    let n = ctx.n();
    let stats = ctx.stats();
    // 每股 Z 值
    let z_ret: Vec<f64> = stats.iter().map(|s| s.ret).collect();
    let z_turn: Vec<f64> = stats.iter().map(|s| s.total_vol).collect();
    let z_vol: Vec<f64> = stats.iter().map(|s| s.vol30).collect();
    let z_imb: Vec<f64> = stats.iter().map(|s| s.imb).collect();
    let z_size: Vec<f64> = stats.iter().map(|s| s.amount).collect();

    let mut out = Vec::new();
    for mat in crate::yupei_dist::indicator_ctx::MATRIX_LIST {
        if !ctx.wants("nbz_", mat) {
            continue;
        }
        let sym = match ctx.symmetric(mat) {
            Some(s) => s,
            None => continue,
        };
        let neighbor = |z: &[f64]| -> Vec<f32> {
            (0..n)
                .into_par_iter()
                .map(|i| {
                    let row = &sym[i * n..(i + 1) * n];
                    // 分母用**正项之和**: urg 族矩阵元素可为负（权重是带符号 z 分数），
                    // 带符号行和会让 p = v/total 大于 1，权重和不再归一。
                    let total: f64 = row.iter().map(|&v| (v as f64).max(0.0)).sum();
                    if total <= 0.0 {
                        // 该行没有任何正权交互 → 邻居加权无定义
                        return f32::NAN;
                    }
                    let inv = 1.0 / total;
                    let mut acc = 0.0f64;
                    for (j, &v) in row.iter().enumerate() {
                        if v > 0.0 {
                            acc += v as f64 * inv * z[j];
                        }
                    }
                    acc as f32
                })
                .collect()
        };
        for (suffix, z) in [
            ("return", &z_ret),
            ("turnover", &z_turn),
            ("volatility", &z_vol),
            ("imbalance", &z_imb),
            ("size", &z_size),
        ] {
            let nm = format!("nbz_{mat}_{suffix}");
            if let Some(k) = &ctx.keep {
                if !k.contains(nm.as_str()) {
                    continue;
                }
            }
            out.push(IndicatorResult::new(nm, neighbor(z)));
        }
    }
    out
}
