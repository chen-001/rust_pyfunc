//! trophic: 营养层级违反度（rho 交互矩阵网络算子，2025-10-31 提案；无 S_dir 时用秩诱导方向）。
//! 有向边 w_{i→j} = S_sym_ij（j ∈ i 的 top50）; 方向完全由秩不对称产生——即使 w_ij=w_ji
//! 数值相同，各自是否选中对方也不同。在此有向加权图上:
//!   v_i = 入强 − 出强（被重仓程度）; Λ = D − (W+Wᵀ) 为对称权和拉普拉斯;
//!   解 Λh = ṽ（每连通分量内 v 去均值、锚定末节点, Jacobi 预条件 CG, 确定性）;
//!   分量内 h 平移到最低=1。
//!   trophic_{mat}_inco_i = mean_{j∈top50(i)} (h_j − h_i − 1)²
//! 值高 = 上下游关系错位（邻居层级差偏离 +1 的整齐链条）; 低 = 链条连贯。
//! 兄弟列:
//!   trophic_{mat}_span_i = max_j h_j − min_j h_j（j∈top50(i)），邻居高度的极差——
//!                          局部链高差，和 inco 是不同统计（span 不管均值位移只管展幅）。
//! 与 dep_top5 同属"秩互惠"族但视角不同：dep 讲"我重仓谁", inco 讲"邻居间层级是否连贯"。
//! 稀疏化用 topk_select_indices（quickselect 部分选择, 确定性: 值降序, 同值索引升序）。

use crate::yupei_dist::indicator_ctx::{IndicatorCtx, IndicatorResult};
use crate::yupei_dist::topk_util::topk_select_indices;
use rayon::prelude::*;
use std::collections::BTreeSet;

pub fn name() -> &'static str {
    "trophic"
}

pub fn desc() -> &'static str {
    "营养层级: inco=mean_{top50出边}(h_j-h_i-1)^2; span=top50邻居h极差 (Λh=入强-出强, 全部28矩阵)"
}

/// Jacobi 预条件 CG 解缩边拉普拉斯系统 A'x=b'（去锚节点）。
/// nodes: 分量内全局节点升序, 末位是锚（x≡0, 不参与求解）;
/// adj_w: 全局对称权邻接 Wt; deg_w: ΣWt。rhs 长度 = nodes.len()-1。
/// 确定性: 固定零初始、固定遍历顺序、固定迭代上限。
fn cg_solve(nodes: &[usize], adj_w: &[Vec<(u32, f64)>], deg_w: &[f64], rhs: &[f64]) -> Vec<f64> {
    let m = nodes.len() - 1;
    // 全局号→局部号
    let mut loc = vec![usize::MAX; adj_w.len()];
    for (l, &g) in nodes.iter().enumerate() {
        loc[g] = l;
    }
    let diag: Vec<f64> = nodes[..m].iter().map(|&g| deg_w[g].max(1e-12)).collect();
    // A'x: (Λx)_u = deg_u·x_u − Σ_{j} Wt_uj·x_j，j 为锚或不在分量时按0
    let mv = |x: &[f64], out: &mut Vec<f64>| {
        out.iter_mut().for_each(|v| *v = 0.0);
        for u in 0..m {
            let gu = nodes[u];
            let mut s = deg_w[gu] * x[u];
            for &(gj, w) in adj_w[gu].iter() {
                let l = loc[gj as usize];
                if l != usize::MAX && l < m {
                    s -= w * x[l];
                }
            }
            out[u] = s;
        }
    };
    let mut x = vec![0.0f64; m];
    let mut ax = vec![0.0; m];
    let mut r: Vec<f64> = rhs.to_vec();
    let mut z: Vec<f64> = r.iter().zip(&diag).map(|(r, d)| r / d).collect();
    let mut p = z.clone();
    let mut rz: f64 = r.iter().zip(&z).map(|(a, b)| a * b).sum();
    let tol = 1e-10 * rhs.iter().map(|v| v * v).sum::<f64>().sqrt().max(1e-12);
    for _ in 0..2000 {
        mv(&p, &mut ax);
        let pap: f64 = p.iter().zip(&ax).map(|(a, b)| a * b).sum();
        if pap.abs() < 1e-30 {
            break;
        }
        let alpha = rz / pap;
        for u in 0..m {
            x[u] += alpha * p[u];
            r[u] -= alpha * ax[u];
        }
        let rn: f64 = r.iter().map(|v| v * v).sum::<f64>().sqrt();
        if rn < tol {
            break;
        }
        let mut rz_new = 0.0;
        for u in 0..m {
            z[u] = r[u] / diag[u];
            rz_new += r[u] * z[u];
        }
        let beta = rz_new / rz;
        for u in 0..m {
            p[u] = z[u] + beta * p[u];
        }
        rz = rz_new;
    }
    x
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
        // 每行 top-50（确定性, 并行无妨）
        let nbrs: Vec<Vec<u32>> = (0..n)
            .into_par_iter()
            .map(|i| topk_select_indices(&sym[i * n..(i + 1) * n], k))
            .collect();
        // 有向弧 w_{i->j} = sym_ij（i 的 top50）；v_i = 入强-出强；
        // 对称权和 Wt_ij = sym_ij·(我选j) + sym_ij·(j选我)  —— 即 Σ_arc 权重
        let mut v_net = vec![0.0f64; n];
        let mut edges: BTreeSet<(u32, u32)> = BTreeSet::new();
        for i in 0..n {
            for &j in nbrs[i].iter() {
                let ju = j as usize;
                let w = sym[i * n + ju] as f64;
                v_net[ju] += w;
                v_net[i] -= w;
                let (a, b) = if (i as u32) < j { (i as u32, j) } else { (j, i as u32) };
                if a != b {
                    edges.insert((a, b));
                }
            }
        }
        // 无向权和邻接（Wt 计两条弧之和）与并查集求分量
        let mut adj_w: Vec<Vec<(u32, f64)>> = vec![Vec::new(); n];
        let mut deg_w = vec![0.0f64; n];
        for &(a, b) in edges.iter() {
            let (au, bu) = (a as usize, b as usize);
            let w = sym[au * n + bu] as f64
                * ((nbrs[au].contains(&b) as u32) + (nbrs[bu].contains(&a) as u32)) as f64;
            adj_w[au].push((b, w));
            adj_w[bu].push((a, w));
            deg_w[au] += w;
            deg_w[bu] += w;
        }
        // 并查集（确定性: 按索引升序 union）
        let mut par: Vec<usize> = (0..n).collect();
        fn find(par: &mut Vec<usize>, x: usize) -> usize {
            let mut r = x;
            while par[r] != r {
                r = par[r];
            }
            let mut c = x;
            while par[c] != r {
                let nxt = par[c];
                par[c] = r;
                c = nxt;
            }
            r
        }
        for &(a, b) in edges.iter() {
            let (ra, rb) = (find(&mut par, a as usize), find(&mut par, b as usize));
            if ra != rb {
                par[ra.max(rb)] = ra.min(rb);
            }
        }
        let mut comp_of = vec![usize::MAX; n];
        let mut comps: Vec<Vec<usize>> = Vec::new();
        for i in 0..n {
            let r = find(&mut par, i);
            if comp_of[r] == usize::MAX {
                comp_of[r] = comps.len();
                comps.push(Vec::new());
            }
            comps[comp_of[r]].push(i);
        }
        // 逐分量解 Λh=ṽ（去均值+锚末节点），分量内最低置 1
        let mut h = vec![0.0f64; n];
        for nodes in comps.iter() {
            if nodes.len() == 1 {
                h[nodes[0]] = 1.0;
                continue;
            }
            let m = nodes.len() - 1;
            let mean_v = nodes.iter().map(|&g| v_net[g]).sum::<f64>() / nodes.len() as f64;
            let rhs: Vec<f64> = nodes[..m].iter().map(|&g| v_net[g] - mean_v).collect();
            let x = cg_solve(nodes, &adj_w, &deg_w, &rhs);
            let mut vals: Vec<f64> = x;
            vals.push(0.0); // 锚节点 h=0
            let minv = vals.iter().cloned().fold(f64::INFINITY, f64::min);
            for (l, &g) in nodes.iter().enumerate() {
                h[g] = vals[l] - minv + 1.0;
            }
        }
        // 节点违反度 inco_i = mean_{j∈top50(i)} (h_j - h_i - 1)^2;
        // span_i = top50 邻居 h 的极差（max−min，不含自身）
        let rows: Vec<[f32; 2]> = (0..n)
            .map(|i| {
                if nbrs[i].is_empty() {
                    return [f32::NAN; 2];
                }
                let mut acc = 0.0f64;
                let (mut hi, mut lo) = (f64::NEG_INFINITY, f64::INFINITY);
                for &j in nbrs[i].iter() {
                    let hj = h[j as usize];
                    let d = hj - h[i] - 1.0;
                    acc += d * d;
                    if hj > hi {
                        hi = hj;
                    }
                    if hj < lo {
                        lo = hj;
                    }
                }
                [
                    (acc / nbrs[i].len() as f64) as f32,
                    (hi - lo) as f32,
                ]
            })
            .collect();
        let inco: Vec<f32> = rows.iter().map(|r| r[0]).collect();
        let span: Vec<f32> = rows.iter().map(|r| r[1]).collect();
        out.push(IndicatorResult::new(format!("trophic_{mat}_inco"), inco));
        out.push(IndicatorResult::new(format!("trophic_{mat}_span"), span));
    }
    out
}
