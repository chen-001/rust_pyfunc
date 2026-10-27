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
        // lite 模式：该矩阵 btw_ 列幸存才往下算（单列全弃则整矩阵跳过）
        if !ctx.wants("btw_", mat) {
            continue;
        }
        let sym = match ctx.symmetric(mat) {
            Some(s) => s,
            None => continue,
        };
        let k = 50usize;
        // 每行 top-50 邻居（确定性; 只取正权边, 不用零值补位）
        let nbrs: Vec<Vec<u32>> = (0..n)
            .into_par_iter()
            .map(|i| crate::yupei_dist::topk_util::topk_select_indices(&sym[i * n..(i + 1) * n], k))
            .collect();
        // 邻居表已按索引升序? 不需要: Brandes 只遍历 nbrs，不做邻接查询。
        // （旧的 is_edge 闭包是死代码，已删——原意是无向图，实际跑的是有向图。）

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
                // 反向累积依赖（标准 Brandes）:
                //   δ(v) = Σ_{w ∈ succ(v)} (σ_v / σ_w)·(1 + δ(w))
                // 逆序处理保证后继的 δ(w) 已经算好。
                // 旧实现写成 delta[w] += (σ_w/σ_v)(1+δ(v))，方向反了——把依赖推给后继
                // 而不是从前驱收，数值会超过介数的理论上界 (n-1)(n-2)（实测归一化后 3.98）。
                for &v32 in order.iter().rev() {
                    let v = v32 as usize;
                    for &w in nbrs[v].iter() {
                        let w = w as usize;
                        if dist[w] == dist[v] + 1 && sigma[w] > 0.0 {
                            delta[v] += sigma[v] / sigma[w] * (1.0 + delta[w]);
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
        // 归一化: 有向介数除以 (N-1)(N-2)，值域 [0,1]，可跨日/跨 universe 比较。
        // （图是有向的: i 选 j 不保证 j 选 i，所以用有向归一因子，不是 (N-1)(N-2)/2。）
        let norm = ((n as f64 - 1.0) * (n as f64 - 2.0)).max(f64::MIN_POSITIVE);
        let col: Vec<f32> = betweenness
            .iter()
            .map(|&x| if n >= 3 { (x / norm) as f32 } else { f32::NAN })
            .collect();
        out.push(IndicatorResult::new(format!("btw_{mat}_betweenness"), col));
    }
    out
}
