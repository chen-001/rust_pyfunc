//! motifs: 三角 motif（降维指标.md 三十）。
//! 有向 cnt_t1, 每行取 top-20 出边与 top-20 入边构建邻接;
//! feed_forward_i = #{j,k: i→j, j→k, i→k}; cycle_i = #{j,k: i→j, j→k, k→i};
//! reciprocal_i = #{j: i→j 且 j→i}（计数）。

use crate::yupei_dist::indicator_ctx::{IndicatorCtx, IndicatorResult};
use rayon::prelude::*;

pub fn name() -> &'static str {
    "motifs"
}

pub fn desc() -> &'static str {
    "三角 motif: feed_forward/cycle/reciprocal 计数 (cnt_t1 有向, top-20)"
}

pub fn compute(ctx: &IndicatorCtx) -> Vec<IndicatorResult> {
    let n = ctx.n();
    let mut out: Vec<IndicatorResult> = Vec::new();
    let k = 20usize;
    for mat in crate::yupei_dist::indicator_ctx::MATRIX_LIST {
        let d = match ctx.matrix(mat) {
            Some(m) => m,
            None => continue,
        };
        // 出边邻接: out_edges[i] = i 的前 k 大出边目标
        let out_edges: Vec<Vec<u32>> = (0..n)
            .into_par_iter()
            .map(|i| crate::yupei_dist::topk_util::topk_indices(&d[i * n..(i + 1) * n], k))
            .collect();
        // 邻接查询表（i 是否有向 j 的边）: 排序后二分
        let mut sorted_out: Vec<Vec<u32>> = out_edges.clone();
        for v in sorted_out.iter_mut() {
            v.sort_unstable();
        }
        let has_out = |i: usize, j: u32| -> bool { sorted_out[i].binary_search(&j).is_ok() };

        let rows: Vec<(f32, f32, f32)> = (0..n)
            .into_par_iter()
            .map(|i| {
                let mut f = 0u32;
                let mut c = 0u32;
                let mut r = 0u32;
                for &j in out_edges[i].iter() {
                    if has_out(j as usize, i as u32) {
                        r += 1;
                    }
                }
                for &j in out_edges[i].iter() {
                    for &l in out_edges[j as usize].iter() {
                        if l == i as u32 {
                            continue;
                        }
                        // i→j, j→l
                        if has_out(i, l) {
                            f += 1; // feed-forward: i→j, j→l, i→l
                        }
                        if has_out(l as usize, i as u32) {
                            c += 1; // cycle: i→j, j→l, l→i
                        }
                    }
                }
                (f as f32, c as f32, r as f32)
            })
            .collect();
        let ff: Vec<f32> = rows.iter().map(|&(a, _, _)| a).collect();
        let cy: Vec<f32> = rows.iter().map(|&(_, b, _)| b).collect();
        let rc: Vec<f32> = rows.iter().map(|&(_, _, c)| c).collect();
        out.push(IndicatorResult::new(format!("mot_{mat}_feed_forward"), ff));
        out.push(IndicatorResult::new(format!("mot_{mat}_cycle"), cy));
        out.push(IndicatorResult::new(format!("mot_{mat}_reciprocal"), rc));
    }
    out
}
