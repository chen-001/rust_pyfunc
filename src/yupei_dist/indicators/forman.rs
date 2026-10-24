//! forman: Forman 离散曲率节点版（rho 交互矩阵网络算子，2025-10-31 提案）。
//! 在 top-50 对称并集无权图上（i 选 j 或 j 选 i 即连, 与 community 同口径）:
//!   边曲率 F_ij = 4 − d_i − d_j + 3·△_ij（△ = i,j 公共邻居数, 无权无方向）
//!   forman_{mat}_mean = 节点 i 所有关联边的 F 均值。
//! 负值 = 手里攥着两团之间的独木桥（结构瓶颈）; 高值 = 深陷互相抱团的集团内部。
//! 与 betweenness 的区别: betweenness 是全局最短路占比（近似）, Forman 是纯局部一条边一个数,
//! 且 −d_i−d_j 的度惩罚使"密集团成员"与"两团间的桥"可分（clustering 没有这一项）。
//! 稀疏化用 topk_select_indices（quickselect 部分选择, 确定性: 值降序, 同值索引升序）。

use crate::yupei_dist::indicator_ctx::{IndicatorCtx, IndicatorResult};
use crate::yupei_dist::topk_util::topk_select_indices;
use rayon::prelude::*;
use std::collections::BTreeSet;

pub fn name() -> &'static str {
    "forman"
}

pub fn desc() -> &'static str {
    "Forman 边曲率节点均值: 4-d_i-d_j+3*公共邻居, top-50 对称并集无权图 (全部28矩阵)"
}

/// 由每行 top-k 名单构建对称无权邻接表（并集）, 邻接按索引升序（确定性）。
fn union_adj(n: usize, nbrs: &[Vec<u32>]) -> Vec<Vec<u32>> {
    let mut edges: BTreeSet<(u32, u32)> = BTreeSet::new();
    for (i, list) in nbrs.iter().enumerate() {
        for &j in list {
            let (a, b) = if (i as u32) < j { (i as u32, j) } else { (j, i as u32) };
            if a != b {
                edges.insert((a, b));
            }
        }
    }
    let mut adj: Vec<Vec<u32>> = vec![Vec::new(); n];
    for &(a, b) in edges.iter() {
        adj[a as usize].push(b);
        adj[b as usize].push(a);
    }
    adj
}

pub fn compute(ctx: &IndicatorCtx) -> Vec<IndicatorResult> {
    let n = ctx.n();
    let mut out = Vec::new();
    let k = 50usize;
    for mat in crate::yupei_dist::indicator_ctx::MATRIX_LIST {
        let sym = match ctx.symmetric(mat) {
            Some(s) => s,
            None => continue,
        };
        // 每行 top-50（确定性, 行并行）
        let nbrs: Vec<Vec<u32>> = (0..n)
            .into_par_iter()
            .map(|i| topk_select_indices(&sym[i * n..(i + 1) * n], k))
            .collect();
        let adj = union_adj(n, &nbrs);
        let deg: Vec<usize> = adj.iter().map(|a| a.len()).collect();
        // 二分查找用的有序表 = adj 本身已按索引升序（BTreeSet 顺序插入）
        let col: Vec<f32> = (0..n)
            .into_par_iter()
            .map(|i| {
                let ni = &adj[i];
                if ni.is_empty() {
                    return f32::NAN;
                }
                let mut acc = 0.0f64;
                for &j in ni.iter() {
                    let ju = j as usize;
                    // 公共邻居数: 两有序表双指针
                    let (mut p, mut q, mut cn) = (0usize, 0usize, 0usize);
                    let (ai, bj) = (&adj[i], &adj[ju]);
                    while p < ai.len() && q < bj.len() {
                        match ai[p].cmp(&bj[q]) {
                            std::cmp::Ordering::Less => p += 1,
                            std::cmp::Ordering::Greater => q += 1,
                            std::cmp::Ordering::Equal => {
                                cn += 1;
                                p += 1;
                                q += 1;
                            }
                        }
                    }
                    acc += 4.0 - deg[i] as f64 - deg[ju] as f64 + 3.0 * cn as f64;
                }
                (acc / ni.len() as f64) as f32
            })
            .collect();
        out.push(IndicatorResult::new(format!("forman_{mat}_mean"), col));
    }
    out
}
