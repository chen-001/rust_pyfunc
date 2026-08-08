//! betweenness: 介数中心性（降维指标.md 十一, 近似）。
//! 每行取 top-50 邻居（对称 cnt_t1）建无权图; Brandes 算法逐源点并行;
//! 对每个源点只在邻居子图上做 BFS（无权近似, 注释说明）。

use crate::yupei_dist::indicator_ctx::{IndicatorCtx, IndicatorResult};
use rayon::prelude::*;

pub fn name() -> &'static str {
    "betweenness"
}

pub fn desc() -> &'static str {
    "介数中心性: top-50 邻居无权图, Brandes 并行 (cnt_t1)"
}

pub fn compute(ctx: &IndicatorCtx) -> Vec<IndicatorResult> {
    let n = ctx.n();
    let mut out = Vec::new();
    for mat in crate::yupei_dist::indicator_ctx::MATRIX_LIST {
        let sym = match ctx.symmetric(mat) {
            Some(s) => s,
            None => continue,
        };
        let k = 50usize;
    // 每行 top-50 邻居（确定性）
    let nbrs: Vec<Vec<u32>> = (0..n)
        .into_par_iter()
        .map(|i| {
            let row = &sym[i * n..(i + 1) * n];
            let mut idx: Vec<u32> = (0..n as u32).collect();
            idx.sort_unstable_by(|&a, &b| {
                row[a as usize]
                    .partial_cmp(&row[b as usize])
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| a.cmp(&b))
            });
            idx.truncate(k);
            idx
        })
        .collect();
    // 邻居查询表: 每行一个排序数组（二分查询）
    let mut sorted_nbrs: Vec<Vec<u32>> = nbrs.clone();
    for v in sorted_nbrs.iter_mut() {
        v.sort_unstable();
    }
    let is_edge = |a: usize, b: u32| -> bool { sorted_nbrs[a].binary_search(&b).is_ok() };

    // Brandes（无权, 只走邻居子图）: 对每个源点 s, BFS 计算依赖累加
    // 并行于源点; 每源点的 sigma/delta 数组本地分配（n=3964, k=50 图稀疏）
    let mut betweenness = vec![0.0f64; n];
    let contribs: Vec<Vec<(u32, f64)>> = (0..n)
        .into_par_iter()
        .map(|s| {
            let mut sigma = vec![0.0f64; n];
            let mut dist = vec![-1i32; n];
            let mut delta = vec![0.0f64; n];
            let mut queue: Vec<u32> = Vec::with_capacity(n);
            sigma[s] = 1.0;
            dist[s] = 0;
            queue.push(s as u32);
            let mut head = 0usize;
            let mut order: Vec<u32> = Vec::with_capacity(n);
            while head < queue.len() {
                let v = queue[head] as usize;
                head += 1;
                order.push(v as u32);
                let dv = dist[v];
                for &w in nbrs[v].iter() {
                    let w = w as usize;
                    if dist[w] < 0 {
                        dist[w] = dv + 1;
                        queue.push(w as u32);
                    }
                    if dist[w] == dv + 1 {
                        sigma[w] += sigma[v];
                    }
                }
            }
            // 反向累积依赖（只对可达节点; 确定性: 按 order 逆序）
            for &v32 in order.iter().rev() {
                let v = v32 as usize;
                for &w in nbrs[v].iter() {
                    let w = w as usize;
                    if dist[w] == dist[v] + 1 {
                        delta[w] += sigma[w] / sigma[v] * (1.0 + delta[v]);
                    }
                }
            }
            let mut out = Vec::new();
            for (i, &dv) in delta.iter().enumerate() {
                if i != s && dv > 0.0 {
                    out.push((i as u32, dv));
                }
            }
            out
        })
        .collect();
    for c in contribs.iter() {
        for &(j, v) in c.iter() {
            betweenness[j as usize] += v;
        }
    }
    // 归一化: 除以 (N-1)(N-2)/2（可选; 保留原始计数, 注释说明）
    let col: Vec<f32> = betweenness.iter().map(|&x| x as f32).collect();
    out.push(IndicatorResult::new(format!("btw_{mat}_betweenness"), col));
    }
    out
}
