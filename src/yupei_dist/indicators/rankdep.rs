//! rankdep: 单向依赖秩差 + 兄弟列（rho 交互矩阵网络算子，2025-10-31 提案）。
//! dep_{mat}_top5 = mean_{j∈top5(i)}( rank_j(i) − pos_i(j) )，其中 rank_j(i) 为 i 在 j 行
//!   按「值降序、同值索引升序」名次，j 未把 i 放入 top50 时按 51 计；pos_i(j) 为 j 在 i 行
//!   top5 内的名次（1..5）。值高 = 我重仓别人但别人眼里我没名（单向依赖）。
//! dep_{mat}_wmean = mean_{j∈top50(i)} (d_ij − d_ji)/max(d_ij,ε)，取原始有向矩阵 d
//!   （sym 对称下恒 0，必须用有向）。秩差之外的权重域视角。
//! dep_{mat}_admirer = |{j: i∈top50(j) 且 j∉top50(i)}| / 50，入向隐性影响力，
//!   与 dep_top5 同族但方向相反（dep 低值 ≈ admirer 高值）。
//! top-50 选择用 topk_select_indices（quickselect 部分选择, 确定性: 值降序同值索引升序）。

use crate::yupei_dist::indicator_ctx::{IndicatorCtx, IndicatorResult};
use crate::yupei_dist::topk_util::topk_select_indices;
use rayon::prelude::*;

pub fn name() -> &'static str {
    "rankdep"
}

pub fn desc() -> &'static str {
    "单向依赖: dep_top5=秩差; dep_wmean=有向权重不对称; dep_admirer=入向隐性影响力 (全部28矩阵)"
}

/// lite 模式下列级过滤：keep=None 恒真，Some 时按完整列名查名单
/// （同矩阵兄弟列可能只留一部分，矩阵级 wants 之外还要逐列过滤）。
#[inline]
fn kept(ctx: &IndicatorCtx, name: &str) -> bool {
    match &ctx.keep {
        None => true,
        Some(k) => k.contains(name),
    }
}

/// x 在 row 的排序中是否排在 i 之前（值降序, 同值索引升序; 与 topk 选择同口径）。
#[inline]
fn ranks_before(row: &[f32], x: usize, i: usize) -> bool {
    match row[x].partial_cmp(&row[i]).unwrap_or(std::cmp::Ordering::Equal) {
        std::cmp::Ordering::Greater => true,
        std::cmp::Ordering::Equal => x < i,
        std::cmp::Ordering::Less => false,
    }
}

/// i 在 j 行里的名次 = 1 + 排在 i 前面的列数；对角列 j 不参与排名（它不是邻居）。
/// O(n) 一次行扫描；只需 top5 各查一次，避免全行排序。
#[inline]
fn rank_in_row(row: &[f32], j: usize, i: usize) -> usize {
    let mut c = 0usize;
    for x in 0..row.len() {
        if x != j && x != i && ranks_before(row, x, i) {
            c += 1;
        }
    }
    c + 1
}

pub fn compute(ctx: &IndicatorCtx) -> Vec<IndicatorResult> {
    let n = ctx.n();
    let mut out = Vec::new();
    let k = 50usize;
    for mat in crate::yupei_dist::indicator_ctx::MATRIX_LIST {
        // lite 模式：该矩阵任一 dep_ 列幸存才往下算
        if !ctx.wants("dep_", mat) {
            continue;
        }
        let sym = match ctx.symmetric(mat) {
            Some(s) => s,
            None => continue,
        };
        let need_top5 = kept(ctx, &format!("dep_{mat}_top5"));
        let need_wmean = kept(ctx, &format!("dep_{mat}_wmean"));
        let need_admirer = kept(ctx, &format!("dep_{mat}_admirer"));
        // dep_wmean 需要原始有向矩阵（sym 对称下 w_ij−w_ji 恒 0）
        let dir = if need_wmean {
            match ctx.matrix(mat) {
                Some(d) => d,
                None => {
                    out.push(IndicatorResult::new(
                        format!("dep_{mat}_wmean"),
                        vec![f32::NAN; n],
                    ));
                    continue;
                }
            }
        } else {
            &sym[..]
        };
        // 每行 top-50（确定性, 行并行）；top5 = 其前 5
        let nbrs: Vec<Vec<u32>> = (0..n)
            .into_par_iter()
            .map(|i| topk_select_indices(&sym[i * n..(i + 1) * n], k))
            .collect();
        // adm 需要排序副本（二分反查）与入度；仅列幸存才建
        let (nbrs_sorted, indeg) = if need_admirer {
            let mut ns: Vec<Vec<u32>> = nbrs.clone();
            for v in ns.iter_mut() {
                v.sort_unstable();
            }
            let mut deg = vec![0u32; n];
            for l in nbrs.iter() {
                for &j in l.iter() {
                    deg[j as usize] += 1;
                }
            }
            (ns, deg)
        } else {
            (Vec::new(), Vec::new())
        };

        let rows: Vec<[f32; 3]> = (0..n)
            .into_par_iter()
            .map(|i| {
                let mut dep_v = f32::NAN;
                let mut wmean_v = f32::NAN;
                let mut adm_v = f32::NAN;
                // dep_top5
                if need_top5 && nbrs[i].len() >= 5 {
                    let mut acc = 0.0f64;
                    for (pos, &j) in nbrs[i].iter().take(5).enumerate() {
                        let ju = j as usize;
                        let r = rank_in_row(&sym[ju * n..(ju + 1) * n], ju, i);
                        acc += (r.min(k + 1) as f64) - (pos as f64 + 1.0);
                    }
                    dep_v = (acc / 5.0) as f32;
                }
                // dep_wmean: mean_j∈top50 (d_ij − d_ji) / max(d_ij, ε)
                if need_wmean {
                    let drow = &dir[i * n..(i + 1) * n];
                    let mut wm = 0.0f64;
                    for &j in nbrs[i].iter() {
                        let ju = j as usize;
                        let wij = drow[ju] as f64;
                        let wji = dir[ju * n + i] as f64;
                        wm += (wij - wji) / wij.max(1e-9);
                    }
                    wmean_v = if nbrs[i].is_empty() {
                        f32::NAN
                    } else {
                        (wm / nbrs[i].len() as f64) as f32
                    };
                }
                // dep_admirer: (被选中次数 − 互选次数) / 50
                if need_admirer {
                    let mutual = nbrs[i]
                        .iter()
                        .filter(|&&j| {
                            nbrs_sorted[j as usize].binary_search(&(i as u32)).is_ok()
                        })
                        .count();
                    adm_v = (indeg[i] - mutual as u32) as f32 / k as f32;
                }
                [dep_v, wmean_v, adm_v]
            })
            .collect();
        if need_top5 {
            out.push(IndicatorResult::new(
                format!("dep_{mat}_top5"),
                rows.iter().map(|r| r[0]).collect(),
            ));
        }
        if need_wmean {
            out.push(IndicatorResult::new(
                format!("dep_{mat}_wmean"),
                rows.iter().map(|r| r[1]).collect(),
            ));
        }
        if need_admirer {
            out.push(IndicatorResult::new(
                format!("dep_{mat}_admirer"),
                rows.iter().map(|r| r[2]).collect(),
            ));
        }
    }
    out
}
