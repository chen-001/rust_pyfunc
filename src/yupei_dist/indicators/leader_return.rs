//! leader_return: 邻居未来收益/领先者冲击（降维指标.md 二十八/二十九）。
//! leader_return_i = Σ_j (S_{j→i}/Σ_k S_{k→i})·r_j,t（有向 cnt_t1, 分母为列和）;
//! leader_shock_i = Σ_j p_{j→i}·Shock_j, Shock_j = 横截面日内收益 z 分。

use crate::yupei_dist::indicator_ctx::{IndicatorCtx, IndicatorResult};
use rayon::prelude::*;

pub fn name() -> &'static str {
    "leader_return"
}

pub fn desc() -> &'static str {
    "领先者收益/冲击 (cnt_t1 有向)"
}

pub fn compute(ctx: &IndicatorCtx) -> Vec<IndicatorResult> {
    let n = ctx.n();
    let mut out = Vec::new();
    let stats = ctx.stats();
    let r: Vec<f64> = stats.iter().map(|s| s.ret).collect();
    // Shock_j = (r_j - mean)/std
    let mean_r = r.iter().sum::<f64>() / (n as f64).max(1.0);
    let var_r = r.iter().map(|&x| (x - mean_r) * (x - mean_r)).sum::<f64>() / (n as f64).max(1.0);
    let std_r = var_r.sqrt().max(1e-12);
    let shock: Vec<f64> = r.iter().map(|&x| (x - mean_r) / std_r).collect();

    for mat in crate::yupei_dist::indicator_ctx::MATRIX_LIST {
        let d = match ctx.matrix(mat) {
            Some(m) => m,
            None => continue,
        };
        // 列和（Σ_k S_{k→i}）
        let mut col_sum = vec![0.0f64; n];
        let col_rows: Vec<Vec<(usize, f64)>> = (0..n)
            .into_par_iter()
            .map(|i| {
                let row = &d[i * n..(i + 1) * n];
                row.iter()
                    .enumerate()
                    .filter(|&(_, &v)| v != 0.0)
                    .map(|(j, &v)| (j, v as f64))
                    .collect()
            })
            .collect();
        for cr in col_rows.iter() {
            for &(j, v) in cr.iter() {
                col_sum[j] += v;
            }
        }
        let ret_col: Vec<f32> = (0..n)
            .into_par_iter()
            .map(|i| {
                let denom = col_sum[i];
                if denom <= 0.0 {
                    return 0.0f32;
                }
                let mut acc = 0.0f64;
                for j in 0..n {
                    if j == i {
                        continue;
                    }
                    let sji = d[j * n + i] as f64; // j 领先 i
                    if sji > 0.0 {
                        acc += (sji / denom) * r[j];
                    }
                }
                acc as f32
            })
            .collect();
        let shock_col: Vec<f32> = (0..n)
            .into_par_iter()
            .map(|i| {
                let denom = col_sum[i];
                if denom <= 0.0 {
                    return 0.0f32;
                }
                let mut acc = 0.0f64;
                for j in 0..n {
                    if j == i {
                        continue;
                    }
                    let sji = d[j * n + i] as f64;
                    if sji > 0.0 {
                        acc += (sji / denom) * shock[j];
                    }
                }
                acc as f32
            })
            .collect();
        out.push(IndicatorResult::new(format!("ldr_{mat}_return"), ret_col));
        out.push(IndicatorResult::new(format!("ldr_{mat}_shock"), shock_col));
    }
    out
}
