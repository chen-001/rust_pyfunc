//! starness: Star-ness（降维指标.md 十四）。
//! StarScore_i = Strength_i × (1 - 归一化聚类系数),
//! 归一化: CC / 全市场 CC 最大值（∈[0,1] 的保守近似, 注释说明）。

use crate::indicator_ctx::{IndicatorCtx, IndicatorResult};
use rayon::prelude::*;

pub fn name() -> &'static str {
    "starness"
}

pub fn desc() -> &'static str {
    "Star-ness: Strength × (1 - 归一化聚类系数) (cnt_t1)"
}

pub fn compute(ctx: &IndicatorCtx) -> Vec<IndicatorResult> {
    let n = ctx.n();
    let k = 50usize;
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
    // 聚类系数（与 clustering 模块同款加权定义: 邻居间实际连接/最大可能,
    // 用每行 top-50 邻居的加权平均连接强度）
    let coefs: Vec<f32> = (0..n)
        .into_par_iter()
        .map(|i| {
            let row = &sym[i * n..(i + 1) * n];
            let nbr: Vec<u32> = crate::topk_util::topk_indices(row, k);
            let m = nbr.len();
            if m < 2 {
                return 0.0f32;
            }
            let mut links = 0.0f64;
            let mut maxl = 0.0f64;
            for a in 0..m {
                for b in (a + 1)..m {
                    let ja = nbr[a] as usize;
                    let jb = nbr[b] as usize;
                    links += sym[ja * n + jb] as f64;
                    maxl += 1.0;
                }
            }
            if maxl > 0.0 {
                (links / maxl) as f32
            } else {
                0.0
            }
        })
        .collect();
    let max_cc = coefs.iter().cloned().fold(0.0f32, f32::max);
    let col: Vec<f32> = (0..n)
        .into_par_iter()
        .map(|i| {
            let cc = if max_cc > 0.0 { coefs[i] / max_cc } else { 0.0 };
            (row_sums[i] as f32) * (1.0 - cc)
        })
        .collect();
    out.push(IndicatorResult::new(format!("star_{mat}_score"), col));
    }
    out
}
