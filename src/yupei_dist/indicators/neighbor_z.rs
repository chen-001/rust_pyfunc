//! neighbor_z: 邻居质量（降维指标.md 六）。
//! p_ij = S_ij/行和(对称 cnt_t1); NeighborZ_i = Σ_j p_ij·Z_j,
//! Z ∈ {日内收益 ret, 总成交量 total_vol(换手代理), vol30, 订单失衡 imb, 成交额 amount(规模代理)}。

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
        let sym = match ctx.symmetric(mat) {
            Some(s) => s,
            None => continue,
        };
        let row_sums = match ctx.row_sum_sym(mat) {
            Some(s) => s,
            None => continue,
        };
        let neighbor = |z: &[f64]| -> Vec<f32> {
            (0..n)
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
                            acc += v as f64 * inv * z[j];
                        }
                    }
                    acc as f32
                })
                .collect()
        };
        out.push(IndicatorResult::new(
            format!("nbz_{mat}_return"),
            neighbor(&z_ret),
        ));
        out.push(IndicatorResult::new(
            format!("nbz_{mat}_turnover"),
            neighbor(&z_turn),
        ));
        out.push(IndicatorResult::new(
            format!("nbz_{mat}_volatility"),
            neighbor(&z_vol),
        ));
        out.push(IndicatorResult::new(
            format!("nbz_{mat}_imbalance"),
            neighbor(&z_imb),
        ));
        out.push(IndicatorResult::new(
            format!("nbz_{mat}_size"),
            neighbor(&z_size),
        ));
    }
    out
}
