//! maxst: 最大生成树角色（提案① MaxST，ecol-econ 原型）。
//! 生产 sym 是未归一的双向强度（dir+dirᵀ，每股尺度差几个数量级），直接跑 Prim 会退化成
//! 星形（最强股吸走所有边，deg 分布均值2/中位1/最大上千）。先按行强度归一：
//! rho_ij = sym_ij / √(s_i·s_j)，s = 行正权重和（s<=0 的行整行边权按 0）——与原型 rho
//! 矩阵口径一致（跨行可比），Prim 再退化概率低得多。
//! 边权<=0/NaN 按 0，从归一化行强度最大的股出发做稠密 Prim（每轮扫 max，O(n²) 无堆），
//! 同权取小索引（确定性）。列: deg = 该股在 MaxST 上的度数。

use crate::yupei_dist::indicator_ctx::{IndicatorCtx, IndicatorResult};

pub fn name() -> &'static str {
    "maxst"
}

pub fn desc() -> &'static str {
    "MaxST 树角色: deg_MST（先按行强度归一 sym; 28 矩阵各一列; 稠密 Prim）"
}

/// 行正权重和（NaN/<=0 不计入）。
fn row_strength(sym: &[f32], n: usize) -> Vec<f32> {
    let mut s = vec![0.0f32; n];
    for i in 0..n {
        let row = &sym[i * n..(i + 1) * n];
        let mut acc = 0.0f32;
        for &v in row {
            if v > 0.0 {
                acc += v;
            }
        }
        s[i] = acc;
    }
    s
}

/// 归一化: w_ij = sym_ij / √(s_i·s_j)；s_i 或 s_j <=0 的边按 0（整行剔除）。
/// 返回 (权重矩阵, 归一化行强度)。
fn normalized(sym: &[f32], n: usize) -> (Vec<f32>, Vec<f32>) {
    let s = row_strength(sym, n);
    let mut w = vec![0.0f32; n * n];
    let mut rw = vec![0.0f32; n];
    for i in 0..n {
        let si = s[i];
        let row = &sym[i * n..(i + 1) * n];
        let wrow = &mut w[i * n..(i + 1) * n];
        let mut acc = 0.0f32;
        for j in 0..n {
            if i == j {
                continue;
            }
            let v = row[j];
            let sj = s[j];
            let mut val = 0.0f32;
            if v > 0.0 && si > 0.0 && sj > 0.0 {
                val = v / (si * sj).sqrt();
                if val.is_nan() {
                    val = 0.0;
                }
            }
            wrow[j] = val;
            acc += val;
        }
        rw[i] = acc;
    }
    (w, rw)
}

/// 稠密 Prim 最大生成树, 返回 (deg, depth, bott)。
/// deg = 树上度数; depth = 到根边数（根=0, 未到树=NaN）; bott = 入树边权（根/未到树=NaN）。
/// 输入为未归一对称矩阵（内部归一）。确定性: 每轮 argmax 同权取最小索引。
pub fn mst_stats(sym: &[f32], n: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let (w, strength) = normalized(sym, n);
    let root = strength
        .iter()
        .enumerate()
        .max_by(|a, b| {
            a.1.partial_cmp(b.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(b.0.cmp(&a.0))
        })
        .map(|(i, _)| i)
        .unwrap_or(0);
    // best[j] = j 到树内的最大边权; par[j] = 该边另一端（同权取小 par）
    let mut in_t = vec![false; n];
    let mut best = vec![0.0f32; n];
    let mut par = vec![0usize; n];
    let mut deg = vec![0u32; n];
    let mut depth = vec![f32::NAN; n];
    let mut bott = vec![f32::NAN; n];
    in_t[root] = true;
    depth[root] = 0.0;
    {
        let row = &w[root * n..(root + 1) * n];
        for j in 0..n {
            // 零权边不是边: 初值给 NEG_INFINITY，不让"权重 0 的坑"把节点接进树。
            // 旧实现初值 0.0，Prim 会把跨分量的零权桥也算一跳，depth 于是由索引顺序决定。
            best[j] = if row[j] > 0.0 { row[j] } else { f32::NEG_INFINITY };
            par[j] = root;
        }
        best[root] = f32::NEG_INFINITY;
    }
    for _ in 1..n {
        // 扫描取 max best（未入树; 同权小索引）
        let mut j = usize::MAX;
        let mut bv = f32::NEG_INFINITY;
        for c in 0..n {
            if !in_t[c] && best[c] > bv {
                bv = best[c];
                j = c;
            }
        }
        if j == usize::MAX {
            break; // 剩下的节点与已入树部分不连通（零权不算边）→ depth/bott 保持 NaN
        }
        in_t[j] = true;
        deg[j] += 1;
        deg[par[j]] += 1;
        depth[j] = depth[par[j]] + 1.0;
        bott[j] = w[j * n + par[j]];
        // 更新: 只接受正权边; w(j,k) > best[k] 或 (== 且 j < par[k]) → par=j
        let row = &w[j * n..(j + 1) * n];
        for k in 0..n {
            if in_t[k] {
                continue;
            }
            let wk = row[k];
            if wk > 0.0 && (wk > best[k] || (wk == best[k] && j < par[k])) {
                best[k] = wk;
                par[k] = j;
            }
        }
    }
    (
        deg.into_iter().map(|d| d as f32).collect(),
        depth,
        bott,
    )
}

/// 兼容旧签名: 只取度数。
pub fn mst_degrees(sym: &[f32], n: usize) -> Vec<f32> {
    mst_stats(sym, n).0
}

pub fn compute(ctx: &IndicatorCtx) -> Vec<IndicatorResult> {
    let n = ctx.n();
    let mut out = Vec::new();
    for mat in crate::yupei_dist::indicator_ctx::MATRIX_LIST {
        if !ctx.wants("maxst_", mat) {
            continue;
        }
        let sym = match ctx.symmetric(mat) {
            Some(s) => s,
            None => continue,
        };
        let (deg, depth, bott) = mst_stats(&sym, n);
        for (suffix, col) in [("deg", deg), ("depth", depth), ("bott", bott)] {
            let nm = format!("maxst_{mat}_{suffix}");
            if let Some(k) = &ctx.keep {
                if !k.contains(nm.as_str()) {
                    continue;
                }
            }
            out.push(IndicatorResult::new(nm, col));
        }
    }
    out
}
