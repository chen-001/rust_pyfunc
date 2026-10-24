//! jsmarg: 行分布对市场边际分布的 JS 散度（信息论增量指标）。
//! 每股: p_ij = S_ij / 行内非对角权重和（对角剔除; NaN/负值按 0）;
//! 市场边际 q_j = Strength_j / Σ_k Strength_k（Strength=对称行和, 含对角, 每张矩阵只算一次）;
//! JS_i = Σ_{j≠i} [ ½·p_ij·ln(p_ij/m_ij) + ½·q_j·ln(q_j/m_ij) ], m_ij = (p_ij+q_j)/2。
//! 截面含义: 伙伴关系结构偏离"全市场谁被链"共识的程度; 熵只量集中度, JS 量结构差异。
//! 行无非对角正权重（孤立行）或 q 行全 0 输出 NaN。全部 28 矩阵。

use crate::yupei_dist::indicator_ctx::{IndicatorCtx, IndicatorResult};
use rayon::prelude::*;

pub fn name() -> &'static str {
    "jsmarg"
}

pub fn desc() -> &'static str {
    "JS散度: 行分布p vs 市场边际q=Strength/ΣStrength (28矩阵)"
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
        // 市场边际 q，全市场只算一次；总强度 ≤0 时整列 NaN 保长度
        let total: f64 = row_sums.iter().sum();
        let js: Vec<f32> = if total <= 0.0 {
            vec![f32::NAN; n]
        } else {
            let inv_t = 1.0 / total;
            let q: Vec<f64> = row_sums.iter().map(|&s| s * inv_t).collect();
            (0..n)
                .into_par_iter()
                .map(|i| {
                    let row = &sym[i * n..(i + 1) * n];
                    let mut wsum = 0.0f64;
                    for (j, &v) in row.iter().enumerate() {
                        if j != i && v.is_finite() && v > 0.0 {
                            wsum += v as f64;
                        }
                    }
                    if wsum <= 0.0 {
                        return f32::NAN;
                    }
                    let inv_w = 1.0 / wsum;
                    let mut d = 0.0f64;
                    for (j, &v) in row.iter().enumerate() {
                        if j == i {
                            continue;
                        }
                        let qj = q[j];
                        let pij = if v.is_finite() && v > 0.0 {
                            v as f64 * inv_w
                        } else {
                            0.0
                        };
                        let m = 0.5 * (pij + qj);
                        if pij > 0.0 {
                            d += 0.5 * pij * (pij / m).ln();
                        }
                        if qj > 0.0 {
                            d += 0.5 * qj * (qj / m).ln();
                        }
                    }
                    d as f32
                })
                .collect()
        };
        out.push(IndicatorResult::new(format!("jsmarg_{mat}"), js));
    }
    out
}
