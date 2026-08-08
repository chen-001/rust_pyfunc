//! concentration: 连接集中度（降维指标.md 三）。
//! p_ij = S_ij / 行和; HHI = Σp²; Entropy = -Σp·ln p; NormalizedEntropy = Entropy/ln(N-1);
//! Top3Share/Top5Share/Top10Share = TopK 和 / 行和。cnt_t1 与 vol_t1。

use crate::indicator_ctx::{IndicatorCtx, IndicatorResult};
use rayon::prelude::*;

pub fn name() -> &'static str {
    "concentration"
}

pub fn desc() -> &'static str {
    "集中度: HHI/熵/归一化熵/Top3·5·10占比 (cnt_t1, vol_t1)"
}

pub fn compute(ctx: &IndicatorCtx) -> Vec<IndicatorResult> {
    let n = ctx.n();
    let mut out = Vec::new();
    for mat in crate::indicator_ctx::MATRIX_LIST {
        let sym = match ctx.symmetric(mat) {
            Some(s) => s,
            None => continue,
        };
        let row_sums = match ctx.row_sum_sym(mat) {
            Some(s) => s,
            None => continue,
        };
        let ln_n = ((n as f64) - 1.0).max(1.0).ln();
        let rows: Vec<(f32, f32, f32, f32, f32, f32)> = (0..n)
            .into_par_iter()
            .map(|i| {
                let row = &sym[i * n..(i + 1) * n];
                let total = row_sums[i];
                if total <= 0.0 {
                    return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
                }
                let inv = 1.0 / total;
                let mut h = 0.0f64;
                let mut e = 0.0f64;
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
                    h += p * p;
                    e -= p * p.ln();
                    if best.len() < 10 {
                        best.push((p, j as u32));
                        if best.len() == 10 {
                            best.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap().then_with(|| a.1.cmp(&b.1)));
                        }
                    } else if p > best[9].0 {
                        best[9] = (p, j as u32);
                        best.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap().then_with(|| a.1.cmp(&b.1)));
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
                    if ln_n > 0.0 { (e / ln_n) as f32 } else { 0.0 },
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
            out.push(IndicatorResult::new(format!("conc_{mat}_{name}"), col));
        }
    }
    out
}
