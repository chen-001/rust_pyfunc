//! community: 社群结构（降维指标.md 十六/十七）。
//! Louvain 社群检测（对称 cnt_t1 加权, 稀疏化: 每行 top-50 邻居, 无向化）;
//! 确定性: 节点按索引序扫描, 社群 id 按首次出现顺序编号。
//! 输出: 社群大小, within, outside, purity, participation（连接系数）;
//! participation = 1 - Σ_c (k_ic/k_i)²。

use crate::indicator_ctx::{IndicatorCtx, IndicatorResult};
use rayon::prelude::*;

pub fn name() -> &'static str {
    "community"
}

pub fn desc() -> &'static str {
    "Louvain 社群: size/within/outside/purity/participation (cnt_t1)"
}

pub fn compute(ctx: &IndicatorCtx) -> Vec<IndicatorResult> {
    let n = ctx.n();
    let mut out = Vec::new();
    for mat in crate::indicator_ctx::MATRIX_LIST {
    let sym = match ctx.symmetric(mat) {
        Some(s) => s,
        None => continue,
    };
    // ---- 构建无向加权邻接表（每行 top-50; 边保留当 i∈top50(j) 或 j∈top50(i), 权重=(S_ij+S_ji)/2）
    let k = 50usize;
    let top: Vec<Vec<u32>> = (0..n)
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
    // 无向边: 用 BTreeSet 保证确定性
    let mut edges: std::collections::BTreeSet<(u32, u32)> = std::collections::BTreeSet::new();
    for i in 0..n {
        for &j in top[i].iter() {
            let (a, b) = if i < j as usize { (i as u32, j) } else { (j, i as u32) };
            if a != b {
                edges.insert((a, b));
            }
        }
    }
    // 邻接表（对称, 权重）
    let mut adj: Vec<Vec<(u32, f32)>> = vec![Vec::new(); n];
    let mut total_w = 0.0f64;
    for &(a, b) in edges.iter() {
        let w = (sym[a as usize * n + b as usize] + sym[b as usize * n + a as usize]) * 0.5;
        if w > 0.0 {
            adj[a as usize].push((b, w));
            adj[b as usize].push((a, w));
            total_w += w as f64;
        }
    }
    let m = total_w;
    // 节点总强度
    let k_i: Vec<f64> = (0..n)
        .map(|i| adj[i].iter().map(|&(_, w)| w as f64).sum())
        .collect();

    // ---- Louvain（确定性: 节点按索引序; 社群 id 为 0..n 的紧凑重编号）
    let mut comm = vec![0usize; n];
    for i in 0..n {
        comm[i] = i;
    }
    // 社群内边权和（Σ_tot）与内部边权
    let mut comm_tot: Vec<f64> = k_i.clone();
    for round in 0..12 {
        let mut moved = false;
        for i in 0..n {
            let c0 = comm[i];
            // 邻居社群及到它们的边权和
            let mut inc: std::collections::BTreeMap<usize, f64> = std::collections::BTreeMap::new();
            for &(j, w) in adj[i].iter() {
                *inc.entry(comm[j as usize]).or_insert(0.0) += w as f64;
            }
            let mut best_c = c0;
            let mut best_gain = 0.0f64;
            for (&c, &k_in) in inc.iter() {
                if c == c0 {
                    continue;
                }
                // 标准加权模块度增益（无向, γ=1）: ΔQ = k_in - k_i·Σ_tot/2m
                let gain = k_in - k_i[i] * comm_tot[c] / (2.0 * m);
                if gain > best_gain {
                    best_gain = gain;
                    best_c = c;
                }
            }
            if best_c != c0 {
                comm_tot[c0] -= k_i[i];
                comm_tot[best_c] += k_i[i];
                comm[i] = best_c;
                moved = true;
            }
        }
        // 聚合轮（简化: 只做节点移动轮, 不收缩图; 12 轮足够收敛）
        if !moved && round > 0 {
            break;
        }
    }
    // 社群 id 紧凑重编号（按最小节点索引序, 确定性）
    let mut comm_map: std::collections::BTreeMap<usize, usize> = std::collections::BTreeMap::new();
    for &c in comm.iter() {
        if !comm_map.contains_key(&c) {
            comm_map.insert(c, comm_map.len());
        }
    }
    let comm_c: Vec<usize> = comm.iter().map(|&c| comm_map[&c]).collect();
    let n_comm = comm_map.len();
    // 社群大小
    let mut size = vec![0usize; n_comm];
    for &c in comm_c.iter() {
        size[c] += 1;
    }

    // ---- 指标（行并行）
    let size_col: Vec<f32> = comm_c.iter().map(|&c| size[c] as f32).collect();
    let rows: Vec<(f32, f32, f32, f32)> = (0..n)
        .into_par_iter()
        .map(|i| {
            let ci = comm_c[i];
            let row = &sym[i * n..(i + 1) * n];
            let mut w = 0.0f64;
            let mut total = 0.0f64;
            let mut by_comm: std::collections::BTreeMap<usize, f64> = std::collections::BTreeMap::new();
            for (j, &v) in row.iter().enumerate() {
                if v <= 0.0 || j == i {
                    continue;
                }
                total += v as f64;
                if comm_c[j] == ci {
                    w += v as f64;
                }
                *by_comm.entry(comm_c[j]).or_insert(0.0) += v as f64;
            }
            let mut p = 0.0f64;
            if total > 0.0 {
                for &kc in by_comm.values() {
                    let r = kc / total;
                    p += r * r;
                }
            }
            (
                w as f32,
                (total - w) as f32,
                if total > 0.0 { (w / total) as f32 } else { 0.0 },
                (1.0 - p) as f32,
            )
        })
        .collect();
    let within: Vec<f32> = rows.iter().map(|&(a, _, _, _)| a).collect();
    let outside: Vec<f32> = rows.iter().map(|&(_, b, _, _)| b).collect();
    let purity: Vec<f32> = rows.iter().map(|&(_, _, c, _)| c).collect();
    let part: Vec<f32> = rows.iter().map(|&(_, _, _, d)| d).collect();
    out.push(IndicatorResult::new(format!("comm_{mat}_size"), size_col));
    out.push(IndicatorResult::new(format!("comm_{mat}_within"), within));
    out.push(IndicatorResult::new(format!("comm_{mat}_outside"), outside));
    out.push(IndicatorResult::new(format!("comm_{mat}_purity"), purity));
    out.push(IndicatorResult::new(format!("comm_{mat}_participation"), part));
    }
    out
}
