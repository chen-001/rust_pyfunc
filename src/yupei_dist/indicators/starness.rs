//! starness: Star-ness（降维指标.md 十四）。
//! StarScore_i = Strength_i × (1 - 归一化聚类系数),
//! 归一化: CC / 全市场 CC 最大值（∈[0,1] 的保守近似）。
//! 口径说明: CC 是**邻居对原始权重的算术平均**（同 cliqueness，量纲 S²），
//! 与 clustering.rs 的 Zhang-Horvath 归一化定义不同；m 是实际正权邻居数。

use crate::yupei_dist::indicator_ctx::{IndicatorCtx, IndicatorResult};
use rayon::prelude::*;

pub fn name() -> &'static str {
    "starness"
}

pub fn desc() -> &'static str {
    "Star-ness: Strength × (1 - 归一化聚类系数) (cnt_t1)"
}

pub fn compute(ctx: &IndicatorCtx) -> Vec<IndicatorResult> {
    let n = ctx.n();
    let k = 50usize;
    let mut out = Vec::new();
    for mat in crate::yupei_dist::indicator_ctx::MATRIX_LIST {
        if !ctx.wants("star_", mat) {
            continue;
        }
        let sym = match ctx.symmetric(mat) {
            Some(s) => s,
            None => continue,
        };
        // 强度 = **正项**交互和（同 cliqueness 的说明）
        let strength: Vec<f64> = (0..n)
            .into_par_iter()
            .map(|i| {
                sym[i * n..(i + 1) * n]
                    .iter()
                    .map(|&v| (v as f64).max(0.0))
                    .sum()
            })
            .collect();
        // 聚类系数（与 clustering 模块同款加权定义: 邻居间实际连接/最大可能,
        // 用每行 top-50 邻居的加权平均连接强度）
        let coefs: Vec<f32> = (0..n)
            .into_par_iter()
            .map(|i| {
                let row = &sym[i * n..(i + 1) * n];
                let nbr: Vec<u32> = crate::yupei_dist::topk_util::topk_indices(row, k);
                let m = nbr.len();
                if m < 2 {
                    // 邻居不足 2 个 → 局部聚类系数无定义（旧实现返回 0）
                    return f32::NAN;
                }
                let mut links = 0.0f64;
                let mut maxl = 0.0f64;
                for a in 0..m {
                    for b in (a + 1)..m {
                        let ja = nbr[a] as usize;
                        let jb = nbr[b] as usize;
                        links += sym[ja * n + jb] as f64;
                        maxl += 1.0;
                    }
                }
                if maxl > 0.0 {
                    (links / maxl) as f32
                } else {
                    0.0
                }
            })
            .collect();
        let max_cc = coefs.iter().cloned().fold(0.0f32, f32::max);
        let col: Vec<f32> = (0..n)
            .into_par_iter()
            .map(|i| {
                let cc = if max_cc > 0.0 { coefs[i] / max_cc } else { 0.0 };
                (strength[i] as f32) * (1.0 - cc)
            })
            .collect();
        emit(ctx, &mut out, format!("star_{mat}_score"), col);
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
