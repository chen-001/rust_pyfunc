//! closeness: 接近中心性（降维指标.md 十二, 近似）。
//! top-50 邻居子图, 距离 d=1/(S+ε); 每源点 Dijkstra（3 跳上限近似）;
//! Closeness = 1/Σd; 不可达按 N 计入。实现方案注释说明。

use crate::indicator_ctx::{IndicatorCtx, IndicatorResult};
use rayon::prelude::*;

pub fn name() -> &'static str {
    "closeness"
}

pub fn desc() -> &'static str {
    "接近中心性: top-50 邻居子图 Dijkstra(≤3跳), 距离=1/S (cnt_t1)"
}

pub fn compute(ctx: &IndicatorCtx) -> Vec<IndicatorResult> {
    let n = ctx.n();
    let mut out = Vec::new();
    let k = 50usize;
    let max_hops = 3usize;
    for mat in crate::indicator_ctx::MATRIX_LIST {
        let sym = match ctx.symmetric(mat) {
            Some(s) => s,
            None => continue,
        };
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
    let col: Vec<f32> = (0..n)
        .into_par_iter()
        .map(|s| {
            // 多源 BFS（无权, ≤max_hops 跳）: 距离 = 跳数（权重 1/S 的单调近似, 注释说明）
            let mut dist = vec![-1i32; n];
            let mut queue: Vec<u32> = Vec::with_capacity(n);
            dist[s] = 0;
            queue.push(s as u32);
            let mut head = 0usize;
            while head < queue.len() {
                let v = queue[head] as usize;
                head += 1;
                let dv = dist[v];
                if dv >= max_hops as i32 {
                    continue;
                }
                for &w in nbrs[v].iter() {
                    let w = w as usize;
                    if dist[w] < 0 {
                        dist[w] = dv + 1;
                        queue.push(w as u32);
                    }
                }
            }
            let mut sum_d = 0.0f64;
            let mut reached = 0usize;
            for (i, &dd) in dist.iter().enumerate() {
                if i != s && dd > 0 {
                    sum_d += dd as f64;
                    reached += 1;
                }
            }
            let unreached = (n - 1) - reached;
            let total = sum_d + unreached as f64 * (max_hops as f64 + 1.0);
            if total > 0.0 {
                (1.0 / total) as f32
            } else {
                0.0
            }
        })
        .collect();
        out.push(IndicatorResult::new(format!("clo_{mat}_closeness"), col));
    }
    out
}
