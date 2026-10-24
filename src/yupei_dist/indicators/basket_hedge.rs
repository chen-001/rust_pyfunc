//! basket_hedge: 正向网络与反向网络（降维指标.md 三十四）。
//! basket = flow_same_t1 对称行和; hedge = flow_opp_t1 对称行和;
//! hedge_ratio = hedge/(basket+hedge+ε)。

use crate::yupei_dist::indicator_ctx::{IndicatorCtx, IndicatorResult};
use rayon::prelude::*;

pub fn name() -> &'static str {
    "basket_hedge"
}

pub fn desc() -> &'static str {
    "正反网络: basket/hedge/hedge_ratio (flow_same/opp_t1)"
}

pub fn compute(ctx: &IndicatorCtx) -> Vec<IndicatorResult> {
    let n = ctx.n();
    let mut out = Vec::new();
    // signed 族 same/opp 配对: flow×4τ, urg×3τ, ext×2τ
    for (fam, tau) in [
        ("flow", "t02"),
        ("flow", "t05"),
        ("flow", "t1"),
        ("flow", "t5"),
        ("urg", "t1"),
        ("urg", "t5"),
        ("urg", "t30"),
        ("ext", "t1"),
        ("ext", "t5"),
    ] {
        // 列名 bh_{fam}_{tau}_xxx: 矩阵名位置放 {fam}_{tau}
        if !ctx.wants("bh_", &format!("{fam}_{tau}")) {
            continue;
        }
        let same = match ctx.symmetric(&format!("{fam}_same_{tau}")) {
            Some(s) => s,
            None => continue,
        };
        let opp = match ctx.symmetric(&format!("{fam}_opp_{tau}")) {
            Some(s) => s,
            None => continue,
        };
        let basket: Vec<f32> = (0..n)
            .into_par_iter()
            .map(|i| same[i * n..(i + 1) * n].iter().sum())
            .collect();
        let hedge: Vec<f32> = (0..n)
            .into_par_iter()
            .map(|i| opp[i * n..(i + 1) * n].iter().sum())
            .collect();
        let ratio: Vec<f32> = basket
            .iter()
            .zip(hedge.iter())
            .map(|(&b, &h)| h / (b + h + 1e-9))
            .collect();
        let tag = format!("bh_{fam}_{tau}");
        for (suffix, col) in [("basket", basket), ("hedge", hedge), ("hedge_ratio", ratio)] {
            let nm = format!("{tag}_{suffix}");
            if ctx.keep.is_none() || ctx.keep.as_ref().unwrap().contains(nm.as_str()) {
                out.push(IndicatorResult::new(nm, col));
            }
        }
    }
    out
}
