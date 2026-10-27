//! cliqueness: Clique-ness（降维指标.md 十五）。
//! CliqueScore_i = Strength_i × 聚类系数。
//! 口径说明: 这里的"聚类系数"是**邻居对原始权重的算术平均**（links / C(m,2)，量纲 S²、
//! 无上界），与 clustering.rs 的 Zhang-Horvath 形式（先除全矩阵最大权、再取三次根）
//! 不是同一个量；m 是**实际正权邻居数**（top-k 不再用零值补位）。

use crate::yupei_dist::indicator_ctx::{IndicatorCtx, IndicatorResult};
use rayon::prelude::*;

pub fn name() -> &'static str {
    "cliqueness"
}

pub fn desc() -> &'static str {
    "Clique-ness: Strength × 聚类系数 (cnt_t1)"
}

pub fn compute(ctx: &IndicatorCtx) -> Vec<IndicatorResult> {
    let n = ctx.n();
    let k = 50usize;
    let mut out = Vec::new();
    for mat in crate::yupei_dist::indicator_ctx::MATRIX_LIST {
        if !ctx.wants("clique_", mat) {
            continue;
        }
        let sym = match ctx.symmetric(mat) {
            Some(s) => s,
            None => continue,
        };
        // 强度 = **正项**交互和: urg 族 sym 元素可为负（权重是带符号 z 分数），
        // 带符号行和会让"强度"变负、score 反向。
        let strength: Vec<f64> = (0..n)
            .into_par_iter()
            .map(|i| {
                sym[i * n..(i + 1) * n]
                    .iter()
                    .map(|&v| (v as f64).max(0.0))
                    .sum()
            })
            .collect();
        let col: Vec<f32> = (0..n)
            .into_par_iter()
            .map(|i| {
                let row = &sym[i * n..(i + 1) * n];
                let nbr: Vec<u32> = crate::yupei_dist::topk_util::topk_indices(row, k);
                let m = nbr.len();
                if m < 2 {
                    // 邻居不足 2 个 → 局部聚类系数无定义（旧实现返回 0）
                    return f32::NAN;
                }
                let mut links = 0.0f64;
                let mut maxl = 0.0f64;
                for a in 0..m {
                    for b in (a + 1)..m {
                        links += sym[nbr[a] as usize * n + nbr[b] as usize] as f64;
                        maxl += 1.0;
                    }
                }
                let cc = if maxl > 0.0 { links / maxl } else { 0.0 };
                (strength[i] as f32) * (cc as f32)
            })
            .collect();
        out.push(IndicatorResult::new(format!("clique_{mat}_score"), col));
    }
    out
}
