//! surplus: 交互盈余（关联与差异指标.md 17 节）。
//! 零模型（独立叠加）: null_ij = (Σw_i·Σw_j)·2τ/T, Σw = 每股成交笔数(sum_w_cnt),
//! τ = 1s, T = 14220s。S_surplus = S_sym - null（秩 1 修正）;
//! 因子 = 行和(S_surplus) / sqrt(行和(null)) 近似 z 分（注释: 若成交独立泊松,
//! S_ij 期望 = null_ij, 方差 ≈ null_ij, 故该比值近似标准化盈余）。

use crate::yupei_dist::indicator_ctx::{IndicatorCtx, IndicatorResult};
use rayon::prelude::*;

pub fn name() -> &'static str {
    "surplus"
}

pub fn desc() -> &'static str {
    "交互盈余 z: (行和S - 行和null)/sqrt(行和null), null=Σw_i·Σw_j·2τ/T (cnt_t1)"
}

pub fn compute(ctx: &IndicatorCtx) -> Vec<IndicatorResult> {
    let n = ctx.n();
    let stats = ctx.stats();
    let mut out = Vec::new();
    // 盈余对多矩阵: τ 不同 → c = 2τ/T 不同; 权重按矩阵家族取对应 sum_w
    for mat in crate::yupei_dist::indicator_ctx::MATRIX_LIST {
        let tau = crate::yupei_dist::indicator_ctx::matrix_tau(mat);
        let sym = match ctx.symmetric(mat) {
            Some(s) => s,
            None => continue,
        };
        // null 是秩 1: null_ij = c·w_i·w_j, c = 2τ/T
        let c = 2.0 * tau / 14220.0;
        let w: Vec<f64> = stats
            .iter()
            .map(|st| {
                if mat.starts_with("vol") {
                    st.sum_w_vol.max(0.0)
                } else if mat.starts_with("logvol") {
                    st.sum_w_logvol.max(0.0)
                } else if mat.starts_with("flow") {
                    st.sum_w_flow.max(0.0)
                } else if mat.starts_with("urg") {
                    st.sum_w_urg.max(0.0)
                } else if mat.starts_with("ext") {
                    st.sum_w_ext.max(0.0)
                } else {
                    st.sum_w_cnt.max(0.0)
                }
            })
            .collect();
        let total_w: f64 = w.iter().sum();
        let col: Vec<f32> = (0..n)
            .into_par_iter()
            .map(|i| {
                let row_sum: f64 = sym[i * n..(i + 1) * n].iter().map(|&v| v as f64).sum();
                let null_row = c * w[i] * total_w;
                if null_row <= 0.0 {
                    return 0.0f32;
                }
                ((row_sum - null_row) / null_row.sqrt()) as f32
            })
            .collect();
        out.push(IndicatorResult::new(format!("srp_{mat}_surplus_z"), col));
    }
    out
}
