//! directionality: 方向性（降维指标.md 四）。
//! 用有向矩阵 S_dir: OutStrength=行和, InStrength=列和, NetLead=Out-In,
//! LeadRatio=(Out-In)/(Out+In+ε)。cnt_t1 与 cnt_t30。

use crate::indicator_ctx::{IndicatorCtx, IndicatorResult};
use rayon::prelude::*;

pub fn name() -> &'static str {
    "directionality"
}

pub fn desc() -> &'static str {
    "方向性: out/in strength, net lead, lead ratio (cnt_t1, cnt_t30 有向)"
}

pub fn compute(ctx: &IndicatorCtx) -> Vec<IndicatorResult> {
    let n = ctx.n();
    let mut out = Vec::new();
    // 方向性谱系: 全部 37 有向矩阵 + signed 族 9 个净有向（net = same - opp）
    let mut mats: Vec<&str> = crate::indicator_ctx::MATRIX_LIST.to_vec();
    mats.extend(crate::indicator_ctx::NET_LIST);
    for mat in mats {
        let d = match dir_matrix(ctx, mat) {
            Some(m) => m,
            None => continue,
        };
        let out_s: Vec<f64> = (0..n)
            .into_par_iter()
            .map(|i| d[i * n..(i + 1) * n].iter().map(|&v| v as f64).sum())
            .collect();
        let mut in_s = vec![0.0f64; n];
        let col_rows: Vec<Vec<(usize, f64)>> = (0..n)
            .into_par_iter()
            .map(|i| {
                let row = &d[i * n..(i + 1) * n];
                row.iter()
                    .enumerate()
                    .filter(|&(_, &v)| v != 0.0)
                    .map(|(j, &v)| (j, v as f64))
                    .collect()
            })
            .collect();
        for r in col_rows.iter() {
            for &(j, v) in r.iter() {
                in_s[j] += v;
            }
        }
        let net: Vec<f32> = out_s.iter().zip(in_s.iter()).map(|(o, x)| (o - x) as f32).collect();
        let ratio: Vec<f32> = out_s
            .iter()
            .zip(in_s.iter())
            .map(|(o, x)| {
                let denom = o + x + 1e-9;
                ((o - x) / denom) as f32
            })
            .collect();
        let out32: Vec<f32> = out_s.iter().map(|&v| v as f32).collect();
        let in32: Vec<f32> = in_s.iter().map(|&v| v as f32).collect();
        out.push(IndicatorResult::new(format!("dir_{mat}_out_strength"), out32));
        out.push(IndicatorResult::new(format!("dir_{mat}_in_strength"), in32));
        out.push(IndicatorResult::new(format!("dir_{mat}_net_lead"), net));
        out.push(IndicatorResult::new(format!("dir_{mat}_lead_ratio"), ratio));
    }
    out
}

/// 有向矩阵: name 为 "{family}_t{tau}"（如 "flow_t5"）→ ctx.matrix_net（same−opp）;
/// 否则为普通有向矩阵（S_dir）。
fn dir_matrix<'a>(ctx: &'a crate::indicator_ctx::IndicatorCtx, name: &str) -> Option<std::borrow::Cow<'a, [f32]>> {
    use std::borrow::Cow;
    if (name.starts_with("flow_") || name.starts_with("urg_") || name.starts_with("ext_"))
        && !name.contains("same")
        && !name.contains("opp")
    {
        let arc = ctx.matrix_net(name)?;
        return Some(Cow::Owned(arc.as_ref().clone()));
    }
    ctx.matrix(name).map(Cow::Borrowed)
}
