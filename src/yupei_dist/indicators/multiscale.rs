//! multiscale: 时间尺度网络（降维指标.md 三十二/三十三）。
//! fast_strength = cnt_t02 对称行和; slow_strength = cnt_t30 对称行和;
//! fast_slow_ratio = fast/(slow+ε); scale_consistency = Jaccard(Top10_t02, Top10_t30)。

use crate::yupei_dist::indicator_ctx::{IndicatorCtx, IndicatorResult};
use rayon::prelude::*;

pub fn name() -> &'static str {
    "multiscale"
}

pub fn desc() -> &'static str {
    "多尺度: fast/slow strength, ratio, scale consistency (cnt_t02 vs cnt_t30)"
}

/// 每行 top-k 索引（值降序, 同值索引升序; 确定性）
fn topk(row: &[f32], k: usize) -> Vec<u32> {
    let mut idx: Vec<u32> = (0..row.len() as u32).collect();
    idx.sort_unstable_by(|&a, &b| {
        row[b as usize]
            .partial_cmp(&row[a as usize])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.cmp(&b))
    });
    idx.truncate(k);
    idx
}

pub fn compute(ctx: &IndicatorCtx) -> Vec<IndicatorResult> {
    let n = ctx.n();
    let mut out = Vec::new();
    // 快/慢 τ 配对: (快矩阵, 慢矩阵)
    for (fast_n, slow_n) in [
        ("cnt_t005", "cnt_t1"), ("cnt_t02", "cnt_t5"), ("cnt_t1", "cnt_t30"),
        ("vol_t02", "vol_t5"), ("vol_t1", "vol_t30"),
        ("logvol_t05", "logvol_t3"), ("logvol_t1", "logvol_t30"),
        ("flow_same_t02", "flow_same_t1"), ("flow_same_t05", "flow_same_t5"),
        ("urg_same_t1", "urg_same_t30"), ("ext_same_t1", "ext_same_t5"),
    ] {
        let fast = match ctx.symmetric(fast_n) {
            Some(s) => s,
            None => continue,
        };
        let slow = match ctx.symmetric(slow_n) {
            Some(s) => s,
            None => continue,
        };
        let fast_sum = match ctx.row_sum_sym(fast_n) {
            Some(s) => s,
            None => continue,
        };
        let slow_sum = match ctx.row_sum_sym(slow_n) {
            Some(s) => s,
            None => continue,
        };
        let fast_s: Vec<f32> = fast_sum.iter().map(|&v| v as f32).collect();
        let slow_s: Vec<f32> = slow_sum.iter().map(|&v| v as f32).collect();
        let ratio: Vec<f32> = fast_s
            .iter()
            .zip(slow_s.iter())
            .map(|(&f, &s)| f / (s + 1e-9))
            .collect();
        let top10_f: Vec<Vec<u32>> = (0..n)
            .into_par_iter()
            .map(|i| topk(&fast[i * n..(i + 1) * n], 10))
            .collect();
        let top10_s: Vec<Vec<u32>> = (0..n)
            .into_par_iter()
            .map(|i| topk(&slow[i * n..(i + 1) * n], 10))
            .collect();
        let jaccard: Vec<f32> = (0..n)
            .into_par_iter()
            .map(|i| {
                let a = &top10_f[i];
                let b = &top10_s[i];
                let mut inter = 0usize;
                for &x in a.iter() {
                    if b.contains(&x) {
                        inter += 1;
                    }
                }
                let union = a.len() + b.len() - inter;
                if union > 0 {
                    inter as f32 / union as f32
                } else {
                    0.0
                }
            })
            .collect();
        let tag = format!("ms_{fast_n}_{slow_n}");
        out.push(IndicatorResult::new(format!("{tag}_fast_strength"), fast_s));
        out.push(IndicatorResult::new(format!("{tag}_slow_strength"), slow_s));
        out.push(IndicatorResult::new(format!("{tag}_fast_slow_ratio"), ratio));
        out.push(IndicatorResult::new(format!("{tag}_scale_consistency"), jaccard));
    }
    out
}
