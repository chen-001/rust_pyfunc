//! closeness: 接近中心性（降维指标.md 十二, 近似）。
//! top-50 邻居子图（只取正权边）, 距离 = 跳数（无权 BFS, ≤3 跳上限近似）;
//! Closeness = Wasserman-Faust 可达版 = (|R|/(N−1))·(|R|/Σ_{j∈R} d_ij)，值域 [0,1]。
//! 不可达节点不计入距离和，只在 |R|/(N−1) 这一项里体现（标准处理）。

use crate::yupei_dist::indicator_ctx::{IndicatorCtx, IndicatorResult};
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
    for mat in crate::yupei_dist::indicator_ctx::MATRIX_LIST {
        if !ctx.wants("clo_", mat) {
            continue;
        }
        let sym = match ctx.symmetric(mat) {
            Some(s) => s,
            None => continue,
        };
        let nbrs: Vec<Vec<u32>> = (0..n)
            .into_par_iter()
            .map(|i| crate::yupei_dist::topk_util::topk_select_indices(&sym[i * n..(i + 1) * n], k))
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
                // Wasserman-Faust 可达版接近中心性（不连通图的定义）:
                //   C(i) = (|R|/(N-1)) · (|R|/Σ_{j∈R} d_ij)
                // 值域 [0,1]，与 universe 大小无关，可跨日比较。
                // （旧实现是 1/Σd，缺 (N-1) 归一，绝对水平落在 1/(3N)~1/(4N)，
                //   随每天股票数漂移；不可达还按 max_hops+1=4 硬算距离。）
                if reached == 0 || n < 2 {
                    return f32::NAN;
                }
                let r = reached as f64;
                ((r / (n as f64 - 1.0)) * (r / sum_d)) as f32
            })
            .collect();
        out.push(IndicatorResult::new(format!("clo_{mat}_closeness"), col));
    }
    out
}
