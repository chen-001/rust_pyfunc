//! concentration: 连接集中度（降维指标.md 三）。
//! p_ij = S_ij / 行和; HHI = Σp²; Entropy = -Σp·ln p; NormalizedEntropy = Entropy/ln(N-1);
//! Top3Share/Top5Share/Top10Share = TopK 和 / 行和。cnt_t1 与 rvol_t1。

use crate::yupei_dist::indicator_ctx::{IndicatorCtx, IndicatorResult};
use rayon::prelude::*;

pub fn name() -> &'static str {
    "concentration"
}

pub fn desc() -> &'static str {
    "集中度: HHI/熵/归一化熵/Top3·5·10占比 (cnt_t1, rvol_t1)"
}

/// lite 模式下只推幸存列。
fn emit(ctx: &IndicatorCtx, out: &mut Vec<IndicatorResult>, name: String, col: Vec<f32>) {
    match &ctx.keep {
        Some(k) if !k.contains(name.as_str()) => {}
        _ => out.push(IndicatorResult::new(name, col)),
    }
}

pub fn compute(ctx: &IndicatorCtx) -> Vec<IndicatorResult> {
    let n = ctx.n();
    let mut out = Vec::new();
    for mat in crate::yupei_dist::indicator_ctx::MATRIX_LIST {
        if !ctx.wants("conc_", mat) {
            continue;
        }
        let sym = match ctx.symmetric(mat) {
            Some(s) => s,
            None => continue,
        };
        let rows: Vec<(f32, f32, f32, f32, f32, f32)> = (0..n)
            .into_par_iter()
            .map(|i| {
                let row = &sym[i * n..(i + 1) * n];
                // 分母必须用**正项之和**：urg 族的权重是带符号的 z 分数，sym 元素可为负，
                // 带符号行和会让 p = v/total 大于 1（熵超过 ln(nz)，归一化熵 > 1）。
                let total: f64 = row.iter().map(|&v| (v as f64).max(0.0)).sum();
                if total <= 0.0 {
                    // 行和为 0（当天没成交 / 该族无权重）→ 分布无定义，输出 NaN
                    return (f32::NAN, f32::NAN, f32::NAN, f32::NAN, f32::NAN, f32::NAN);
                }
                let inv = 1.0 / total;
                let mut h = 0.0f64;
                let mut e = 0.0f64;
                let mut nz = 0usize; // 非零邻居数 = 该行分布的支撑集大小
                let mut s3 = 0.0f64;
                let mut s5 = 0.0f64;
                let mut s10 = 0.0f64;
                // top-10 部分选择
                let mut best: Vec<(f64, u32)> = Vec::with_capacity(10);
                for (j, &v) in row.iter().enumerate() {
                    let p = v as f64 * inv;
                    if p <= 0.0 {
                        continue;
                    }
                    nz += 1;
                    h += p * p;
                    e -= p * p.ln();
                    if best.len() < 10 {
                        best.push((p, j as u32));
                        if best.len() == 10 {
                            best.sort_by(|a, b| {
                                b.0.partial_cmp(&a.0).unwrap().then_with(|| a.1.cmp(&b.1))
                            });
                        }
                    } else if p > best[9].0 {
                        best[9] = (p, j as u32);
                        best.sort_by(|a, b| {
                            b.0.partial_cmp(&a.0).unwrap().then_with(|| a.1.cmp(&b.1))
                        });
                    }
                }
                for (p, _) in best.iter().take(10) {
                    s10 += p;
                }
                for (p, _) in best.iter().take(5) {
                    s5 += p;
                }
                for (p, _) in best.iter().take(3) {
                    s3 += p;
                }
                (
                    h as f32,
                    e as f32,
                    // 归一化熵: 分母是 ln(该行非零邻居数)，值域 [0,1]。
                    // （旧实现除以 ln(N-1)，分布只落在 k 个伙伴上时可到 ln k/ln(N-1)≈0.5，
                    //   几乎完全由邻居个数决定，不是集中度。）
                    if nz >= 2 { (e / (nz as f64).ln()) as f32 } else { 0.0 },
                    s3 as f32,
                    s5 as f32,
                    s10 as f32,
                )
            })
            .collect();
        let hhi: Vec<f32> = rows.iter().map(|&(a, _, _, _, _, _)| a).collect();
        let ent: Vec<f32> = rows.iter().map(|&(_, b, _, _, _, _)| b).collect();
        let nent: Vec<f32> = rows.iter().map(|&(_, _, c, _, _, _)| c).collect();
        let t3: Vec<f32> = rows.iter().map(|&(_, _, _, d, _, _)| d).collect();
        let t5: Vec<f32> = rows.iter().map(|&(_, _, _, _, e2, _)| e2).collect();
        let t10: Vec<f32> = rows.iter().map(|&(_, _, _, _, _, f)| f).collect();
        for (name, col) in [
            ("hhi", hhi),
            ("entropy", ent),
            ("norm_entropy", nent),
            ("top3_share", t3),
            ("top5_share", t5),
            ("top10_share", t10),
        ] {
            emit(ctx, &mut out, format!("conc_{mat}_{name}"), col);
        }
    }
    out
}
