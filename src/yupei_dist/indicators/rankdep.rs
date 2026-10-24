//! rankdep: 单向依赖秩差（rho 交互矩阵网络算子，2025-10-31 提案）。
//! dep_{mat}_top5 = mean_{j∈top5(i)}( rank_j(i) − pos_i(j) )，其中 rank_j(i) 为 i 在 j 行
//! 按「值降序、同值索引升序」名次，j 未把 i 放入 top50 时按 51 计；pos_i(j) 为 j 在 i 行
//! top5 内的名次（1..5）。
//! 值高 = 我重仓别人但别人眼里我没名（单向依赖）; 值低/负 = 伙伴全反选我前排（隐性影响力）。
//! 拓扑与权重无关的量纲：即使 S 对称（w_ij=w_ji），秩不对称依然非平凡。
//! top-50/top-5 选择用 topk_select_indices（quickselect 部分选择, 确定性: 值降序同值索引升序）。

use crate::yupei_dist::indicator_ctx::{IndicatorCtx, IndicatorResult};
use crate::yupei_dist::topk_util::topk_select_indices;
use rayon::prelude::*;

pub fn name() -> &'static str {
    "rankdep"
}

pub fn desc() -> &'static str {
    "单向依赖秩差: mean_{top5}(对方top50里我的名次(>50按51) - 我top5里对方名次) (全部28矩阵)"
}

/// x 在 row 的排序中是否排在 i 之前（值降序, 同值索引升序; 与 topk_indices 同口径）。
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
        let sym = match ctx.symmetric(mat) {
            Some(s) => s,
            None => continue,
        };
        // 每行 top-5（确定性）; dep 只需 top5 与 rank 反查，不建邻接
        let col: Vec<f32> = (0..n)
            .into_par_iter()
            .map(|i| {
                let row = &sym[i * n..(i + 1) * n];
                let top5 = topk_select_indices(row, 5);
                if top5.len() < 5 {
                    return f32::NAN;
                }
                let mut acc = 0.0f64;
                for (pos, &j) in top5.iter().enumerate() {
                    let ju = j as usize;
                    let rj = {
                        let r = rank_in_row(&sym[ju * n..(ju + 1) * n], ju, i);
                        if r > k {
                            k + 1
                        } else {
                            r
                        }
                    };
                    acc += (rj as f64) - (pos as f64 + 1.0);
                }
                (acc / 5.0) as f32
            })
            .collect();
        out.push(IndicatorResult::new(format!("dep_{mat}_top5"), col));
    }
    out
}
