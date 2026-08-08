//! pagerank: PageRank（降维指标.md 十）。
//! 有向 cnt_t1（S_dir[i][j] = i→j）: 阻尼 0.85, 迭代 ≤100 次或 L1 变化 <1e-9;
//! 出度为零的行均分; 固定初始 1/N 均匀（确定性）。
//! 实现: 先把矩阵转置为列主序（next[j] = Σ_i rank_i·S_ij/outdeg_i 按列并行累加）。

use crate::indicator_ctx::{IndicatorCtx, IndicatorResult};
use rayon::prelude::*;

pub fn name() -> &'static str {
    "pagerank"
}

pub fn desc() -> &'static str {
    "PageRank (cnt_t1 有向, d=0.85)"
}

pub fn compute(ctx: &IndicatorCtx) -> Vec<IndicatorResult> {
    let n = ctx.n();
    let mut out = Vec::new();
    for mat in crate::indicator_ctx::MATRIX_LIST {
        let d = match ctx.matrix(mat) {
            Some(m) => m,
            None => continue,
        };
        let damping = 0.85f64;
    let init = 1.0 / n as f64;
    let mut rank = vec![init; n];
    let mut next = vec![0.0f64; n];
    // 出度
    let outdeg: Vec<f64> = (0..n)
        .into_par_iter()
        .map(|i| d[i * n..(i + 1) * n].iter().map(|&v| v as f64).sum())
        .collect();
    // 列主序转置（便于按列并行）
    let dt: Vec<f32> = (0..n)
        .into_par_iter()
        .flat_map_iter(|j| (0..n).map(move |i| d[i * n + j]))
        .collect();
    for _ in 0..100 {
        let base = (1.0 - damping) * init;
        let dangling: f64 = (0..n).filter(|&i| outdeg[i] <= 0.0).map(|i| rank[i]).sum();
        let add = dangling * damping / n as f64;
        // next[j] = base + add + damping·Σ_i rank_i·S_ij/outdeg_i
        next.par_chunks_mut(64).enumerate().for_each(|(ci, chunk)| {
            let j0 = ci * 64;
            for (jj, o) in chunk.iter_mut().enumerate() {
                let j = j0 + jj;
                let col = &dt[j * n..(j + 1) * n];
                let mut acc = 0.0f64;
                for i in 0..n {
                    let w = col[i] as f64;
                    if w > 0.0 && outdeg[i] > 0.0 {
                        acc += rank[i] * w / outdeg[i];
                    }
                }
                *o = base + add + damping * acc;
            }
        });
        let l1: f64 = (0..n).map(|i| (next[i] - rank[i]).abs()).sum();
        std::mem::swap(&mut rank, &mut next);
        if l1 < 1e-9 {
            break;
        }
    }
    let col: Vec<f32> = rank.iter().map(|&x| x as f32).collect();
    out.push(IndicatorResult::new(format!("pr_{mat}_pagerank"), col));
    }
    out
}
