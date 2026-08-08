//! leadership: 领先关系细拆（降维指标.md 五）。
//! cnt_t1 有向: LeadBreadth = #{j: S_ij > S_ji}/(N-1);
//! LeadStrength = Σ_j max(S_ij - S_ji, 0)（净领先总量）;
//! LeadConcentration = 净领先分布 HHI（对正的 S_ij-S_ji 归一化）。

use crate::indicator_ctx::{IndicatorCtx, IndicatorResult};
use rayon::prelude::*;

pub fn name() -> &'static str {
    "leadership"
}

pub fn desc() -> &'static str {
    "leadership: breadth/strength/concentration (cnt_t1 有向)"
}

pub fn compute(ctx: &IndicatorCtx) -> Vec<IndicatorResult> {
    let n = ctx.n();
    let mut out = Vec::new();
    for mat in {
        let mut m: Vec<&str> = crate::indicator_ctx::MATRIX_LIST.to_vec();
        m.extend(crate::indicator_ctx::NET_LIST);
        m
    } {
        let d = match dir_matrix(ctx, mat) {
            Some(m) => m,
            None => continue,
        };
    let rows: Vec<(f32, f32, f32)> = (0..n)
        .into_par_iter()
        .map(|i| {
            let row = &d[i * n..(i + 1) * n];
            let mut cnt = 0usize;
            let mut s = 0.0f64;
            let mut s2 = 0.0f64;
            for j in 0..n {
                if i == j {
                    continue;
                }
                let net = row[j] as f64 - d[j * n + i] as f64;
                if net > 0.0 {
                    cnt += 1;
                    s += net;
                    s2 += net * net;
                }
            }
            (
                cnt as f32 / (n as f32 - 1.0).max(1.0),
                s as f32,
                if s > 0.0 { (s2 / (s * s)) as f32 } else { 0.0 },
            )
        })
        .collect();
    let breadth: Vec<f32> = rows.iter().map(|&(a, _, _)| a).collect();
    let strength: Vec<f32> = rows.iter().map(|&(_, b, _)| b).collect();
    let conc: Vec<f32> = rows.iter().map(|&(_, _, c)| c).collect();
    out.push(IndicatorResult::new(format!("lead_{mat}_breadth"), breadth));
    out.push(IndicatorResult::new(format!("lead_{mat}_strength"), strength));
    out.push(IndicatorResult::new(format!("lead_{mat}_concentration"), conc));
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
