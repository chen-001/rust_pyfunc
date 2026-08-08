//! assortativity: 网络同配性个股版（降维指标.md 十八）。
//! assort_corr: 每行 (S_ij, Strength_j) 的 Pearson 相关（j 遍历邻居, 确定性）;
//! nb_str_minus_mean = NeighborStrength_i - 全市场 Strength 均值。

use crate::indicator_ctx::{IndicatorCtx, IndicatorResult};
use rayon::prelude::*;

pub fn name() -> &'static str {
    "assortativity"
}

pub fn desc() -> &'static str {
    "同配性: 行内 corr(S_ij, Strength_j); 邻居强度减市场均值 (cnt_t1)"
}

pub fn compute(ctx: &IndicatorCtx) -> Vec<IndicatorResult> {
    let n = ctx.n();
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
    let mean_str = row_sums.iter().sum::<f64>() / (n as f64).max(1.0);
    let corr: Vec<f32> = (0..n)
        .into_par_iter()
        .map(|i| {
            let row = &sym[i * n..(i + 1) * n];
            let mut sx = 0.0f64;
            let mut sy = 0.0f64;
            let mut sxy = 0.0f64;
            let mut sxx = 0.0f64;
            let mut syy = 0.0f64;
            let mut cnt = 0usize;
            for j in 0..n {
                if i == j {
                    continue;
                }
                let x = row[j] as f64;
                let y = row_sums[j];
                sx += x;
                sy += y;
                sxy += x * y;
                sxx += x * x;
                syy += y * y;
                cnt += 1;
            }
            if cnt < 3 {
                return 0.0f32;
            }
            let c = cnt as f64;
            let num = sxy - sx * sy / c;
            let den = ((sxx - sx * sx / c) * (syy - sy * sy / c)).sqrt();
            if den > 1e-12 {
                (num / den) as f32
            } else {
                0.0
            }
        })
        .collect();
    // 邻居强度（Σp·Strength, 与 neighbor_strength 同款）
    let nb_str: Vec<f32> = (0..n)
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
                    acc += v as f64 * inv * row_sums[j];
                }
            }
            acc as f32
        })
        .collect();
    let minus_mean: Vec<f32> = nb_str.iter().map(|&v| v - mean_str as f32).collect();
    out.push(IndicatorResult::new(format!("assort_{mat}_corr"), corr));
    out.push(IndicatorResult::new(format!("assort_{mat}_nb_str_minus_mean"), minus_mean));
    }
    out
}
