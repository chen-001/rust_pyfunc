//! topk: Top-K 类指标（降维指标.md 二）。
//! 对对称矩阵每行: Top1/3/5/10/20 Mean, Top3/Top5 Sum, MaxPartner(=Top1Mean),
//! BestPartnerGap = S_{(1)} - S_{(2)}。部分选择 + 行并行; 确定性: 同值按索引升序。

use crate::indicator_ctx::{IndicatorCtx, IndicatorResult};
use rayon::prelude::*;

pub fn name() -> &'static str {
    "topk"
}

pub fn desc() -> &'static str {
    "Top-K 均值/和/MaxPartner/BestPartnerGap (cnt_t1, vol_t1)"
}

/// 每行 top-k 索引（值降序, 同值索引升序; 确定性）
fn topk_indices(row: &[f32], k: usize) -> Vec<u32> {
    let n = row.len();
    let mut idx: Vec<u32> = (0..n as u32).collect();
    idx.sort_unstable_by(|&a, &b| {
        let va = row[a as usize];
        let vb = row[b as usize];
        vb.partial_cmp(&va)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.cmp(&b))
    });
    idx.truncate(k.min(n));
    idx
}

pub fn compute(ctx: &IndicatorCtx) -> Vec<IndicatorResult> {
    let n = ctx.n();
    let mut out = Vec::new();
    // Top-K 谱系: 无符号族多 τ + signed 族 same 分量
    for mat in crate::indicator_ctx::MATRIX_LIST {
        let sym = match ctx.symmetric(mat) {
            Some(s) => s,
            None => continue,
        };
        let top20: Vec<Vec<u32>> = (0..n)
            .into_par_iter()
            .map(|i| topk_indices(&sym[i * n..(i + 1) * n], 20))
            .collect();
        for (kk, suffix) in [
            (1usize, "top1_mean"),
            (3, "top3_mean"),
            (5, "top5_mean"),
            (10, "top10_mean"),
            (20, "top20_mean"),
        ] {
            let col: Vec<f32> = (0..n)
                .into_par_iter()
                .map(|i| {
                    let t = &top20[i];
                    if t.len() < kk {
                        return 0.0f32;
                    }
                    let s: f64 = t.iter().take(kk).map(|&j| sym[i * n + j as usize] as f64).sum();
                    (s / kk as f64) as f32
                })
                .collect();
            out.push(IndicatorResult::new(format!("topk_{mat}_{suffix}"), col));
        }
        for (kk, suffix) in [(3usize, "top3_sum"), (5, "top5_sum")] {
            let col: Vec<f32> = (0..n)
                .into_par_iter()
                .map(|i| {
                    let t = &top20[i];
                    t.iter().take(kk).map(|&j| sym[i * n + j as usize]).sum()
                })
                .collect();
            out.push(IndicatorResult::new(format!("topk_{mat}_{suffix}"), col));
        }
        let col: Vec<f32> = (0..n)
            .into_par_iter()
            .map(|i| {
                if top20[i].is_empty() {
                    0.0
                } else {
                    sym[i * n + top20[i][0] as usize]
                }
            })
            .collect();
        out.push(IndicatorResult::new(format!("topk_{mat}_maxpartner"), col));
        let col: Vec<f32> = (0..n)
            .into_par_iter()
            .map(|i| {
                let t = &top20[i];
                if t.len() < 2 {
                    return 0.0;
                }
                sym[i * n + t[0] as usize] - sym[i * n + t[1] as usize]
            })
            .collect();
        out.push(IndicatorResult::new(format!("topk_{mat}_bestpartnergap"), col));
    }
    out
}
