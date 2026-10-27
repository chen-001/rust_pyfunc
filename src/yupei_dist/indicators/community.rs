//! community: 社群结构（降维指标.md 十六/十七）。
//! Louvain 社群检测（对称 cnt_t1 加权, 稀疏化: 每行 top-50 邻居, 无向化）;
//! 确定性: 节点按索引序扫描, 社群 id 按首次出现顺序编号。
//! 输出: 社群大小, within, outside, purity, participation（连接系数）;
//! participation = 1 - Σ_c (k_ic/k_i)²。

use crate::yupei_dist::indicator_ctx::{IndicatorCtx, IndicatorResult};
use rayon::prelude::*;

pub fn name() -> &'static str {
    "community"
}

pub fn desc() -> &'static str {
    "Louvain 社群: size/within/outside/purity/participation (cnt_t1)"
}

pub fn compute(ctx: &IndicatorCtx) -> Vec<IndicatorResult> {
    let n = ctx.n();
    // 串行收集矩阵（symmetric 缓存顺序不变; lite 模式下无幸存列的矩阵跳过）
    let syms: Vec<Option<std::sync::Arc<Vec<f32>>>> = crate::yupei_dist::indicator_ctx::MATRIX_LIST
        .iter()
        .map(|m| {
            if !ctx.wants("comm_", m) {
                return None;
            }
            ctx.symmetric(m)
        })
        .collect();
    // 矩阵级并行（每矩阵内部 Louvain 串行; 确定性: 结果按矩阵索引收集）
    // 注: IndicatorCtx 含 RefCell 不可跨线程, keep 过滤在收集后做（compute_one 返回全部候选列）
    let per_mat: Vec<Vec<(String, Vec<f32>)>> = syms
        .into_par_iter()
        .enumerate()
        .map(|(mi, sym)| match sym {
            Some(s) => compute_one(n, crate::yupei_dist::indicator_ctx::MATRIX_LIST[mi], &s),
            None => Vec::new(),
        })
        .collect();
    per_mat
        .into_iter()
        .flatten()
        .filter(|(name, _)| match &ctx.keep {
            None => true,
            Some(k) => k.contains(name.as_str()),
        })
        .map(|(name, values)| IndicatorResult::new(name, values))
        .collect()
}

/// 单矩阵社群计算（top-50 邻接 + Louvain + 指标; 确定性; 返回候选列, keep 过滤在调用方）
fn compute_one(n: usize, mat: &str, sym: &[f32]) -> Vec<(String, Vec<f32>)> {
    let mut out = Vec::new();
    // ---- 构建无向加权邻接表（每行 top-50; 边保留当 i∈top50(j) 或 j∈top50(i), 权重=(S_ij+S_ji)/2）
    let k = 50usize;
    let top: Vec<Vec<u32>> = (0..n)
        .into_par_iter()
        .map(|i| crate::yupei_dist::topk_util::topk_select_indices(&sym[i * n..(i + 1) * n], k))
        .collect();
    // 无向边: 用 BTreeSet 保证确定性
    let mut edges: std::collections::BTreeSet<(u32, u32)> = std::collections::BTreeSet::new();
    for i in 0..n {
        for &j in top[i].iter() {
            let (a, b) = if i < j as usize {
                (i as u32, j)
            } else {
                (j, i as u32)
            };
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
    // 社群内边权和（Σ_tot）
    let mut comm_tot: Vec<f64> = k_i.clone();
    for _round in 0..12 {
        let mut moved = false;
        for i in 0..n {
            let c0 = comm[i];
            // i 到各邻居社群的边权和（Vec+排序替代 BTreeMap, 键升序同 → 结果逐位一致）
            let mut inc: Vec<(usize, f64)> = Vec::with_capacity(16);
            for &(j, w) in adj[i].iter() {
                inc.push((comm[j as usize], w as f64));
            }
            inc.sort_unstable_by_key(|&(c, _)| c);
            // 标准 Louvain 移动判据（Blondel 2008）: 先把 i 从原社群摘出，再对**每个候选
            // 社群（含原社群 c0）**算 ΔQ = k_in(i,c) − k_i·Σ_tot(c)/(2m)，取最大；
            // 只有最大 ΔQ > 0 才搬。
            // 旧实现把 c0 直接 continue、门槛写死 0，等于把"留在原社群"的增益当成 0，
            // 于是只要某个邻居社群的目标项为正就搬 —— 第一轮（每人自成社群）恰好等价，
            // 第二轮起判据就是错的: 12 轮每轮都在搬，最终划分取决于第 12 轮停在哪一步
            // （合成 20 张随机图的模块度 0.346 对标准写法 0.610）。
            let tot_c0 = comm_tot[c0] - k_i[i]; // i 摘出后的源社群 Σ_tot
            let mut best_c = c0;
            let mut best_gain = f64::NEG_INFINITY;
            let mut idx = 0usize;
            while idx < inc.len() {
                let c = inc[idx].0;
                let mut k_in = 0.0f64;
                while idx < inc.len() && inc[idx].0 == c {
                    k_in += inc[idx].1;
                    idx += 1;
                }
                let tot_c = if c == c0 { tot_c0 } else { comm_tot[c] };
                let gain = k_in - k_i[i] * tot_c / (2.0 * m);
                if gain > best_gain {
                    best_gain = gain;
                    best_c = c;
                }
            }
            let _ = _round;
            if best_c != c0 && best_gain > 1e-12 {
                comm_tot[c0] -= k_i[i];
                comm_tot[best_c] += k_i[i];
                comm[i] = best_c;
                moved = true;
            }
        }
        // 聚合轮（简化: 只做节点移动轮, 不收缩图; 标准判据下 ΔQ 单调增，必收敛）
        if !moved {
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
            let mut by_comm: Vec<(usize, f64)> = Vec::with_capacity(16);
            for (j, &v) in row.iter().enumerate() {
                if v <= 0.0 || j == i {
                    continue;
                }
                total += v as f64;
                if comm_c[j] == ci {
                    w += v as f64;
                }
                by_comm.push((comm_c[j], v as f64));
            }
            by_comm.sort_unstable_by_key(|&(c, _)| c);
            let mut p = 0.0f64;
            if total > 0.0 {
                let mut idx = 0usize;
                while idx < by_comm.len() {
                    let mut kc = 0.0f64;
                    let c = by_comm[idx].0;
                    while idx < by_comm.len() && by_comm[idx].0 == c {
                        kc += by_comm[idx].1;
                        idx += 1;
                    }
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
    out.push((format!("comm_{mat}_size"), size_col));
    out.push((format!("comm_{mat}_within"), within));
    out.push((format!("comm_{mat}_outside"), outside));
    out.push((format!("comm_{mat}_purity"), purity));
    out.push((format!("comm_{mat}_participation"), part));
    out
}
