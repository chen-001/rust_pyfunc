//! dynjs: 跨日行分布 JS 散度（rho 交互矩阵网络算子，跨日动态）。
//! 当日 p_i = max(sym_ij,0) 非对角权重 / 行和；前日 q_i 同样处理（prev 有向矩阵 S+Sᵀ）。
//! 两只股票集合不同：只在当日与前日都存在的 partner 上把两分布各自重新归一，
//! 再算 JS = 0.5·KL(p‖m) + 0.5·KL(q‖m)，m=(p+q)/2（KL 型 JS，与原生态算子同口径）。
//! 该股前日不存在、任一侧行和为 0、或共同支撑集上质量为 0 → NaN。
//! ctx.prev=None（批量首日）→ 整列 NaN 保长度，写法同 edge_dynamics。
//! prev 缺该矩阵 → 该矩阵列 NaN。

use crate::yupei_dist::indicator_ctx::{IndicatorCtx, IndicatorResult};
use rayon::prelude::*;

pub fn name() -> &'static str {
    "dynjs"
}

pub fn desc() -> &'static str {
    "跨日行分布JS散度: 共同支撑集重归一后 0.5*KL(p||m)+0.5*KL(q||m) (全部28矩阵)"
}

pub fn compute(ctx: &IndicatorCtx) -> Vec<IndicatorResult> {
    let n = ctx.n();
    let mut out = Vec::new();
    let prev = match ctx.prev {
        Some(ref p) => p,
        None => {
            // 无前一日数据（批量首日）: 输出 NaN 列保持因子长度
            for mat in crate::yupei_dist::indicator_ctx::MATRIX_LIST {
                out.push(IndicatorResult::new(
                    format!("dynjs_{mat}"),
                    vec![f32::NAN; n],
                ));
            }
            return out;
        }
    };
    let pn = prev.set.n;
    // 当前股票 → prev 索引（code 映射）
    let map: Vec<Option<usize>> = ctx.codes().iter().map(|c| prev.prev_idx(c)).collect();

    for mat in crate::yupei_dist::indicator_ctx::MATRIX_LIST {
        let sym = match ctx.symmetric(mat) {
            Some(s) => s,
            None => continue,
        };
        // prev 对称矩阵（手动 S+Sᵀ, 有向备份）
        let pd = match prev.set.mats.get(mat) {
            Some(m) => m.as_slice(),
            None => {
                out.push(IndicatorResult::new(
                    format!("dynjs_{mat}"),
                    vec![f32::NAN; n],
                ));
                continue;
            }
        };
        let mut psym = vec![0.0f32; pn * pn];
        psym.par_chunks_mut(pn).enumerate().for_each(|(i, row)| {
            for j in 0..pn {
                if i != j {
                    row[j] = pd[i * pn + j] + pd[j * pn + i];
                }
            }
        });
        // 前日行和（clip 非负）
        let rsp: Vec<f64> = (0..pn)
            .into_par_iter()
            .map(|i| {
                psym[i * pn..(i + 1) * pn]
                    .iter()
                    .map(|&v| (v as f64).max(0.0))
                    .sum()
            })
            .collect();

        let col: Vec<f32> = (0..n)
            .into_par_iter()
            .map(|i| {
                let pi = match map[i] {
                    Some(x) => x,
                    None => return f32::NAN,
                };
                let row = &sym[i * n..(i + 1) * n];
                let prow = &psym[pi * pn..(pi + 1) * pn];
                if rsp[pi] <= 0.0 {
                    return f32::NAN;
                }
                // 当日行和
                let mut rs = 0.0f64;
                for &v in row.iter() {
                    rs += (v as f64).max(0.0);
                }
                if rs <= 0.0 {
                    return f32::NAN;
                }
                // 共同支撑集（j 在 prev 中也存在）上各自重归一再算 JS
                let mut sa = 0.0f64;
                let mut sb = 0.0f64;
                for j in 0..n {
                    if let Some(pj) = map[j] {
                        sa += (row[j] as f64).max(0.0);
                        sb += (prow[pj] as f64).max(0.0);
                    }
                }
                if sa <= 0.0 || sb <= 0.0 {
                    return f32::NAN;
                }
                let mut js = 0.0f64;
                for j in 0..n {
                    if let Some(pj) = map[j] {
                        let a = (row[j] as f64).max(0.0) / sa;
                        let b = (prow[pj] as f64).max(0.0) / sb;
                        let m = 0.5 * (a + b);
                        if a > 0.0 && m > 0.0 {
                            js += 0.5 * a * (a / m).ln();
                        }
                        if b > 0.0 && m > 0.0 {
                            js += 0.5 * b * (b / m).ln();
                        }
                    }
                }
                js as f32
            })
            .collect();
        out.push(IndicatorResult::new(format!("dynjs_{mat}"), col));
    }
    out
}
