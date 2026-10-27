//! isolation: 孤立度（降维指标.md 三十五）。
//! inv_strength = 1/(对称行和+ε); neg_top5 = -(Top5Mean)。cnt_t1 与 rvol_t1 各两个。

use crate::yupei_dist::indicator_ctx::{IndicatorCtx, IndicatorResult};
use rayon::prelude::*;

pub fn name() -> &'static str {
    "isolation"
}

pub fn desc() -> &'static str {
    "孤立度: 1/strength 与 -top5_mean (cnt_t1, rvol_t1)"
}

pub fn compute(ctx: &IndicatorCtx) -> Vec<IndicatorResult> {
    let n = ctx.n();
    let mut out = Vec::new();
    for mat in crate::yupei_dist::indicator_ctx::MATRIX_LIST {
        if !ctx.wants("iso_", mat) {
            continue;
        }
        let sym = match ctx.symmetric(mat) {
            Some(s) => s,
            None => continue,
        };
        let row_sums = match ctx.row_sum_sym(mat) {
            Some(s) => s,
            None => continue,
        };
        // 1/strength: 强度为 0（空行/该族无权重）时无定义，输出 NaN。
        // 旧实现用 1/(v+1e-9)，空行股票直接得到 1e9 这个哨兵数——既不是 inf 也不是 NaN，
        // 与其它列"0 填充"的约定也不一致，做 z 标准化时一只票就能吃掉整列。
        let inv_s: Vec<f32> = row_sums
            .iter()
            .map(|&v| if v > 0.0 { (1.0 / v) as f32 } else { f32::NAN })
            .collect();
        // neg_top5: 只在正权邻居上取前 5（旧实现排序整行，邻居不足 5 个时 0 会进来）
        let neg_top5: Vec<f32> = (0..n)
            .into_par_iter()
            .map(|i| {
                let row = &sym[i * n..(i + 1) * n];
                let mut top: Vec<f32> = row.iter().copied().filter(|&v| v > 0.0).collect();
                if top.is_empty() {
                    return f32::NAN;
                }
                top.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
                let take = top.len().min(5);
                -(top.iter().take(take).sum::<f32>() / take as f32)
            })
            .collect();
        emit(ctx, &mut out, format!("iso_{mat}_inv_strength"),
            inv_s,);
        emit(ctx, &mut out, format!("iso_{mat}_neg_top5"),
            neg_top5,);
    }
    out
}

/// 列级过滤: keep=None 全发; keep=Some 只发名单内列。
fn emit(ctx: &IndicatorCtx, out: &mut Vec<IndicatorResult>, name: String, values: Vec<f32>) {
    if let Some(k) = &ctx.keep {
        if !k.contains(name.as_str()) {
            return;
        }
    }
    out.push(IndicatorResult::new(name, values));
}
