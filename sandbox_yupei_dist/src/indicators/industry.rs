//! industry: 行业内 vs 行业外（降维指标.md 二十五/二十七）。
//! within = Σ_{j: 同行业} S_ij; cross_ratio = (总-同行业)/总; mismatch = 1 - 同行业/总。
//! 行业未知(-1)的股票只统计已知行业邻居; 无已知行业邻居输出 NaN。遍历全部 37 矩阵。

use crate::indicator_ctx::{IndicatorCtx, IndicatorResult};
use rayon::prelude::*;

pub fn name() -> &'static str {
    "industry"
}

pub fn desc() -> &'static str {
    "行业: within/cross_ratio/mismatch (37 矩阵 + industry.bin)"
}

pub fn compute(ctx: &IndicatorCtx) -> Vec<IndicatorResult> {
    let n = ctx.n();
    let mut out = Vec::new();
    let ind = match ctx.industry() {
        Some(v) => v,
        None => {
            // 无行业数据（批量日期未提取）: 输出 NaN 列保持因子长度
            for mat in crate::indicator_ctx::MATRIX_LIST {
                let nan = vec![f32::NAN; n];
                out.push(IndicatorResult::new(format!("ind_{mat}_within"), nan.clone()));
                out.push(IndicatorResult::new(format!("ind_{mat}_cross_ratio"), nan.clone()));
                out.push(IndicatorResult::new(format!("ind_{mat}_mismatch"), nan.clone()));
            }
            return out;
        }
    };
    for mat in crate::indicator_ctx::MATRIX_LIST {
        let sym = match ctx.symmetric(mat) {
            Some(s) => s,
            None => continue,
        };
        let rows: Vec<(f32, f32)> = (0..n)
            .into_par_iter()
            .map(|i| {
                let ii = ind[i];
                if ii < 0 {
                    return (f32::NAN, f32::NAN);
                }
                let row = &sym[i * n..(i + 1) * n];
                let mut w = 0.0f64;
                let mut total = 0.0f64;
                let mut known = 0usize;
                for (j, &v) in row.iter().enumerate() {
                    if v <= 0.0 {
                        continue;
                    }
                    let ij = ind[j];
                    if ij < 0 {
                        continue;
                    }
                    known += 1;
                    total += v as f64;
                    if ij == ii {
                        w += v as f64;
                    }
                }
                if known == 0 {
                    return (f32::NAN, f32::NAN);
                }
                if total > 0.0 {
                    let cr = (total - w) / total;
                    (w as f32, cr as f32)
                } else {
                    (w as f32, 0.0)
                }
            })
            .collect();
        let within: Vec<f32> = rows.iter().map(|&(a, _)| a).collect();
        let cross: Vec<f32> = rows.iter().map(|&(_, b)| b).collect();
        let mism: Vec<f32> = rows.iter().map(|&(_, b)| b).collect();
        out.push(IndicatorResult::new(format!("ind_{mat}_within"), within));
        out.push(IndicatorResult::new(format!("ind_{mat}_cross_ratio"), cross));
        out.push(IndicatorResult::new(format!("ind_{mat}_mismatch"), mism));
    }
    out
}
