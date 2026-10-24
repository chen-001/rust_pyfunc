//! reach2: 二跳触达（rho 交互矩阵网络算子，2025-10-31 提案）。
//! 在 top-50 对称并集无权图上（i 选 j 或 j 选 i 即连, 与 community 同口径）:
//!   R2_i = | N(i) ∪ ⋃_{j∈N(i)} N(j) | / (n−1)（不含自身）
//! 原型上 R2 右偏严重（中位≈0.9，区分度集中在低值"深井股"），故输出变换版:
//!   reach2_{mat}_lgap = ln(2 − R2_i)   （单调减: 值越大 = 触达越差 = 越是深井股）
//! ln(2−R2) ∈ [0, ln2)，有界无 ε；高值端=低触达的井底股，分辨率高。
//! 兄弟列:
//!   reach3_{mat}_lgap = ln(2 − R3_i)，三跳位图并集版（最深井股的探测；R3 更趋近 1，
//!                       区分度进一步压向低值端）
//!   reach_{mat}_eff   = R2_i / (deg_i + 1)，每个直接邻居摊到多少二跳覆盖；
//!                       高 = 邻居带量（hub 高效触达），低 = 散铺在深井里。
//! 实现: 每行把邻居（三跳为邻居的邻居）的位图做并集（u64 桶）, 数 popcount。行并行, 确定性。

use crate::yupei_dist::indicator_ctx::{IndicatorCtx, IndicatorResult};
use crate::yupei_dist::topk_util::topk_select_indices;
use rayon::prelude::*;
use std::collections::BTreeSet;

pub fn name() -> &'static str {
    "reach2"
}

pub fn desc() -> &'static str {
    "二/三跳触达缺口+邻居效率: lgap=ln(2-R), eff=R2/(deg+1) (top-50并集无权图, 全部28矩阵)"
}

/// lite 模式列级过滤（同矩阵兄弟列只留一部分时按完整列名判定）。
#[inline]
fn kept(ctx: &IndicatorCtx, name: &str) -> bool {
    match &ctx.keep {
        None => true,
        Some(k) => k.contains(name),
    }
}

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
    let words = (n + 63) / 64;
    let mut out = Vec::new();
    let k = 50usize;
    for mat in crate::yupei_dist::indicator_ctx::MATRIX_LIST {
        // lite 模式：三族列任一幸存才算；reach3 位图最贵, 单列门
        if !ctx.wants_any(&["reach2_", "reach3_", "reach_"], mat) {
            continue;
        }
        let sym = match ctx.symmetric(mat) {
            Some(s) => s,
            None => continue,
        };
        let need2 = kept(ctx, &format!("reach2_{mat}_lgap"));
        let need3 = kept(ctx, &format!("reach3_{mat}_lgap"));
        let need_eff = kept(ctx, &format!("reach_{mat}_eff"));
        let nbrs: Vec<Vec<u32>> = (0..n)
            .into_par_iter()
            .map(|i| topk_select_indices(&sym[i * n..(i + 1) * n], k))
            .collect();
        let adj = union_adj(n, &nbrs);
        let rows: Vec<[f32; 3]> = (0..n)
            .into_par_iter()
            .map(|i| {
                let mut bm = vec![0u64; words];
                let mut bm3 = vec![0u64; words];
                let set = |bm: &mut Vec<u64>, x: u32| {
                    bm[(x >> 6) as usize] |= 1u64 << (x & 63);
                };
                let two = need2 || need_eff;
                if two {
                    set(&mut bm, i as u32); // 自身置位, 最后从计数里扣掉
                }
                if need3 {
                    set(&mut bm3, i as u32);
                }
                for &j in adj[i].iter() {
                    if two {
                        set(&mut bm, j);
                    }
                    if need3 {
                        set(&mut bm3, j);
                    }
                    for &q in adj[j as usize].iter() {
                        if two {
                            set(&mut bm, q);
                        }
                        if need3 {
                            set(&mut bm3, q);
                            for &r in adj[q as usize].iter() {
                                set(&mut bm3, r);
                            }
                        }
                    }
                }
                let den = (n - 1).max(1) as f64;
                let mut lgap = f32::NAN;
                let mut lgap3 = f32::NAN;
                let mut eff = f32::NAN;
                let mut r2 = 0.0f64;
                if two {
                    let cnt: u64 = bm.iter().map(|w| w.count_ones() as u64).sum();
                    r2 = cnt.saturating_sub(1) as f64 / den;
                    lgap = (2.0 - r2).ln() as f32;
                }
                if need3 {
                    let cnt3: u64 = bm3.iter().map(|w| w.count_ones() as u64).sum();
                    lgap3 = (2.0 - cnt3.saturating_sub(1) as f64 / den).ln() as f32;
                }
                if need_eff {
                    // 无邻居时 reach=0 → eff 按 0/(0+1)=0（深井最极端端）
                    eff = (r2 / (adj[i].len() as f64 + 1.0)) as f32;
                }
                [lgap, lgap3, eff]
            })
            .collect();
        if need2 {
            out.push(IndicatorResult::new(
                format!("reach2_{mat}_lgap"),
                rows.iter().map(|r| r[0]).collect(),
            ));
        }
        if need3 {
            out.push(IndicatorResult::new(
                format!("reach3_{mat}_lgap"),
                rows.iter().map(|r| r[1]).collect(),
            ));
        }
        if need_eff {
            out.push(IndicatorResult::new(
                format!("reach_{mat}_eff"),
                rows.iter().map(|r| r[2]).collect(),
            ));
        }
    }
    out
}
