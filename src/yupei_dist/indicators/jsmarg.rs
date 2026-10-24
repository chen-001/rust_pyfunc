//! jsmarg: 行分布对市场边际分布的 JS 散度（信息论增量指标）。
//! 每股: p_ij = S_ij / 行内非对角权重和（对角剔除; NaN/负值按 0）;
//! 市场边际 q_j = Strength_j / Σ_k Strength_k（Strength=对称行和, 含对角, 每张矩阵只算一次）;
//! JS_i = Σ_{j≠i} [ ½·p_ij·ln(p_ij/m_ij) + ½·q_j·ln(q_j/m_ij) ], m_ij = (p_ij+q_j)/2。
//! 兄弟列 consens_{mat}_top20: 我 top50 邻居（按 w_ij 降序, 同权取小索引）里
//! 属于「全市场 strength 前 20」的伙伴的行权重质量占比 = Σ_{j∈top50(i)∩top20(mkt)} p_ij。
//! 与 jsmarg 是同一想法的两个算子: jsmarg 是全分布 JS 散度（"偏离共识多远"）,
//! consens 是头部共识质量占比（"我多少权重押在市场明星股上"）。
//! 截面含义: 伙伴关系结构偏离"全市场谁被链"共识的程度; 熵只量集中度, JS 量结构差异。
//! 行无非对角正权重（孤立行）或 q 行全 0 输出 NaN。全部 28 矩阵。

use crate::yupei_dist::indicator_ctx::{IndicatorCtx, IndicatorResult};
use rayon::prelude::*;

pub fn name() -> &'static str {
    "jsmarg"
}

pub fn desc() -> &'static str {
    "JS散度: p vs 市场边际q + top50∩市场top20权重占比 (28矩阵)"
}

pub fn compute(ctx: &IndicatorCtx) -> Vec<IndicatorResult> {
    let n = ctx.n();
    let mut out = Vec::new();
    for mat in crate::yupei_dist::indicator_ctx::MATRIX_LIST {
        if !ctx.wants_any(&["jsmarg_", "consens_"], mat) {
            continue;
        }
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
        // 市场 strength 前 20 的股票索引集合（j==i 处已被剔除, 对角误入无妨）
        let mut top20v: Vec<bool> = Vec::new();
        if total > 0.0 {
            let mut idx: Vec<usize> = (0..n).collect();
            idx.sort_unstable_by(|&a, &b| {
                row_sums[b]
                    .partial_cmp(&row_sums[a])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            top20v = vec![false; n];
            for &j in idx.iter().take(20.min(n)) {
                top20v[j] = true;
            }
        }
        let both: Vec<(f32, f32)> = if total <= 0.0 {
            vec![(f32::NAN, f32::NAN); n]
        } else {
            let inv_t = 1.0 / total;
            let q: Vec<f64> = row_sums.iter().map(|&s| s * inv_t).collect();
            let top20v = &top20v[..];
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
                        return (f32::NAN, f32::NAN);
                    }
                    let inv_w = 1.0 / wsum;
                    let mut d = 0.0f64;
                    // top-50 邻居小顶选择（值降序, 同权取小索引; 保持确定性）
                    let mut best: Vec<(f64, u32)> = Vec::with_capacity(51);
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
                        // top-50 候选（只对正权重入候选）
                        if pij > 0.0 {
                            if best.len() < 50 {
                                best.push((pij, j as u32));
                            } else {
                                // 找当前最小（值最小的; 并列时索引最大者先被踢, 保小索引）
                                let mut mi = 0usize;
                                for k in 1..best.len() {
                                    let bk = best[k];
                                    let bm = best[mi];
                                    if bk.0 < bm.0 || (bk.0 == bm.0 && bk.1 > bm.1) {
                                        mi = k;
                                    }
                                }
                                let bv = best[mi];
                                if pij > bv.0 || (pij == bv.0 && (j as u32) < bv.1) {
                                    best[mi] = (pij, j as u32);
                                }
                            }
                        }
                    }
                    let mut cons = 0.0f64;
                    for &(pij, j) in &best {
                        if top20v[j as usize] {
                            cons += pij;
                        }
                    }
                    (d as f32, cons as f32)
                })
                .collect()
        };
        let js: Vec<f32> = both.iter().map(|t| t.0).collect();
        let consens: Vec<f32> = both.iter().map(|t| t.1).collect();
        if ctx.keep.is_none() || ctx.keep.as_ref().unwrap().contains(format!("jsmarg_{mat}").as_str()) {
            out.push(IndicatorResult::new(format!("jsmarg_{mat}"), js));
        }
        if ctx.keep.is_none() || ctx.keep.as_ref().unwrap().contains(format!("consens_{mat}_top20").as_str()) {
            out.push(IndicatorResult::new(format!("consens_{mat}_top20"), consens));
        }
    }
    out
}
