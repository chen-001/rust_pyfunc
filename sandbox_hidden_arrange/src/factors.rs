//! 因子提取：从三层游戏结果提取基础因子（移植 Python factors.py）
use std::collections::HashMap;
use crate::Stage1Result;
use crate::stage2::{Stage2Result, GRAPH_NAMES, SCOPES};
use crate::stage3::Stage3Result;

const S1_SUFFIX: [&str; 11] = ["FINAL_CORR", "NET_GAIN", "NORM_GAIN", "AUC", "QUERY_EFF",
    "ACCEPT_RATIO", "ABS_CONTRAST_MEAN", "CONTRAST_ENERGY", "SCALE_CENTROID", "SCALE_ENTROPY", "PARENT_CHILD_SIGN"];
const S2_SUFFIX: [&str; 8] = ["FINAL_CORR", "NET_GAIN", "NORM_GAIN", "AUC", "QUERY_EFF",
    "ACCEPT_RATIO", "CONTRAST_ENERGY", "PATH_DISTANCE"];
const S3_STATS: [(&str, usize); 12] = [
    ("ENTROPY_MEAN", 0), ("ENTROPY_STD", 1), ("ENTROPY_P90", 2), ("ENTROPY_MAX", 3),
    ("TOP1_CONF_MEAN", 4), ("LOWCONF_RATIO50", 5), ("POSTVAR_MEAN", 6),
    ("TAIL20_ENTROPY", 7), ("TAIL10_ENTROPY", 8), ("TAIL05_ENTROPY", 9),
    ("TAIL01_ENTROPY", 10), ("TAIL10_CONF", 11)];
const FAMILIES_GRAPHS: [(&str, &[&str]); 8] = [
    ("TIME", &["TIME_R1", "TIME_R8", "TIME_R32"]),
    ("ACTIVE", &["ACTIVE_EXACT"]), ("PASSIVE", &["PASSIVE_EXACT"]),
    ("PRICE", &["PRICE_T0", "PRICE_T2"]), ("SIDE", &["SIDE_EXACT"]),
    ("BOOK", &["BOOK_K10", "BOOK_K30"]), ("QUEUE", &["QUEUE_K10"]), ("ARRIVAL", &["ARRIVAL_K10"]),
];

fn sd(a: f64, b: f64) -> f64 { if b.abs() > 1e-12 { a / b } else { 0.0 } }
fn gini(x: &[f64]) -> f64 {
    let mut s: Vec<f64> = x.iter().cloned().collect();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = s.len(); if n == 0 { return 0.0; }
    let sum: f64 = s.iter().sum();
    if sum.abs() < 1e-12 { return 0.0; }
    (2.0 * (1..=n).map(|i| i as f64 * s[i - 1]).sum::<f64>()) / (n as f64 * sum) - (n as f64 + 1.0) / n as f64
}
fn entropy_norm(probs: &[f64], l: usize) -> f64 {
    let p: Vec<f64> = probs.iter().filter(|&&x| x > 1e-12).cloned().collect();
    if p.is_empty() || l <= 1 { return 0.0; }
    -p.iter().map(|x| x * x.ln()).sum::<f64>() / (l as f64).ln()
}
fn auc(traj: &[f64], rho0: f64) -> f64 {
    if traj.len() <= 1 { return 0.0; }
    traj[1..].iter().map(|r| (r - rho0) / (1.0 - rho0 + 1e-12)).sum::<f64>() / (traj.len() - 1) as f64
}
fn percentile(x: &[f64], q: f64) -> f64 {
    let mut s = x.to_vec(); s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = s.len(); if n == 0 { return 0.0; }
    let pos = (n - 1) as f64 * q / 100.0;
    let lo = pos.floor() as usize; let hi = pos.ceil() as usize;
    if lo == hi { s[lo] } else { s[lo] + (pos - lo as f64) * (s[hi] - s[lo]) }
}
pub type Factors = Vec<(String, f64)>;
fn push(out: &mut Factors, name: &str, v: f64) { out.push((name.to_string(), v)); }

pub fn target_dist(y: &[f64]) -> Factors {
    let n = y.len();
    let mu = y.iter().sum::<f64>() / n as f64;
    let sdv = (y.iter().map(|v| (v - mu).powi(2)).sum::<f64>() / n as f64).sqrt();
    let med = percentile(y, 50.0);
    let mad_v: Vec<f64> = y.iter().map(|v| (v - med).abs()).collect();
    let mad = percentile(&mad_v, 50.0);
    let z: Vec<f64> = if sdv > 1e-12 { y.iter().map(|v| (v - mu) / sdv).collect() } else { vec![0f64; n] };
    let skew = if sdv > 1e-12 { z.iter().map(|v| v.powi(3)).sum::<f64>() / n as f64 } else { 0.0 };
    let kurt = if sdv > 1e-12 { z.iter().map(|v| v.powi(4)).sum::<f64>() / n as f64 - 3.0 } else { 0.0 };
    let total: f64 = y.iter().sum();
    let mut sy = y.to_vec(); sy.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let top_share = |frac: f64| { let k = ((frac * n as f64).ceil() as usize).max(1); sy[n - k..].iter().sum::<f64>() / (total.abs() + 1e-12) };
    let mut uniq = y.to_vec(); uniq.sort_by(|a, b| a.partial_cmp(b).unwrap()); uniq.dedup();
    let mut cnt: HashMap<i64, usize> = HashMap::new();
    for v in y.iter().map(|v| (v * 1e9) as i64) { *cnt.entry(v).or_default() += 1; }
    let shares: Vec<f64> = cnt.values().map(|&c| c as f64 / n as f64).collect();
    let mut f = vec![]; let mut p = |n: &str, v: f64| push(&mut f, n, v);
    p("TARGET_MEAN", mu); p("TARGET_STD", sdv); p("TARGET_MEDIAN", med); p("TARGET_MAD", mad);
    p("TARGET_SKEW", skew); p("TARGET_KURT", kurt);
    p("TARGET_MIN", y.iter().cloned().fold(f64::INFINITY, f64::min));
    p("TARGET_MAX", y.iter().cloned().fold(f64::NEG_INFINITY, f64::max));
    p("TARGET_Q75", percentile(y, 75.0)); p("TARGET_Q90", percentile(y, 90.0));
    p("TARGET_Q95", percentile(y, 95.0)); p("TARGET_Q99", percentile(y, 99.0));
    p("TARGET_UNIQUE_COUNT", uniq.len() as f64); p("TARGET_UNIQUE_RATIO", uniq.len() as f64 / n as f64);
    p("TARGET_VALUE_HHI", shares.iter().map(|s| s * s).sum());
    p("TARGET_GINI", gini(y));
    p("TARGET_TOP1_SHARE", top_share(0.01)); p("TARGET_TOP5_SHARE", top_share(0.05));
    p("TARGET_TOP10_SHARE", top_share(0.10));
    p("TARGET_MAX_SHARE", y.iter().cloned().fold(f64::NEG_INFINITY, f64::max) / (total.abs() + 1e-12));
    f
}

fn scale_features(level_ds: &[Vec<f64>]) -> (f64, f64) {
    if level_ds.is_empty() { return (0.0, 0.0); }
    let es: Vec<f64> = level_ds.iter().map(|lv| lv.iter().map(|d| d * d).sum::<f64>()).collect();
    let tot: f64 = es.iter().sum();
    if tot < 1e-12 { return (0.0, 0.0); }
    let pv: Vec<f64> = es.iter().map(|e| e / tot).collect();
    let l = es.len();
    let cent = (1..=l).zip(pv.iter()).map(|(i, pp)| i as f64 * pp).sum::<f64>();
    (cent, entropy_norm(&pv, l))
}

pub fn s1_branches(s1: &Stage1Result) -> Factors {
    let mut f = vec![];
    for b in &s1.branches {
        let name = format!("S1_{}_{}", b.axis, b.depth);
        let net = b.rho_final - b.rho0;
        let (cent, sent) = scale_features(&b.level_ds);
        let vals = [
            b.rho_final, net, sd(net, 1.0 - b.rho0), auc(&b.trajectory, b.rho0),
            sd(net, b.n_queries as f64),
            if b.n_internal > 0 { b.accepts.iter().filter(|&&x| x).count() as f64 / b.n_internal as f64 } else { 0.0 },
            if b.node_ds.is_empty() { 0.0 } else { b.node_ds.iter().map(|d| d.abs()).sum::<f64>() / b.node_ds.len() as f64 },
            if b.node_ds.is_empty() { 0.0 } else { b.node_ds.iter().map(|d| d * d).sum::<f64>() / b.node_ds.len() as f64 },
            cent, sent,
            if b.parent_child_sign.is_empty() { 0.0 } else { b.parent_child_sign.iter().filter(|(a, c)| a == c).count() as f64 / b.parent_child_sign.len() as f64 },
        ];
        for (suf, v) in S1_SUFFIX.iter().zip(vals.iter()) { push(&mut f, &format!("{}_{}", name, suf), *v); }
    }
    f
}

pub fn s1_summary(s1: &Stage1Result) -> Factors {
    let mut f = vec![]; let mut p = |n: &str, v: f64| push(&mut f, n, v);
    let rho0 = s1.rho0;
    let gains: Vec<f64> = s1.branches.iter().map(|b| b.rho_final - b.rho0).collect();
    p("S1_FINAL_CORR", s1.rho); p("S1_NET_GAIN", s1.rho - rho0); p("S1_NORM_GAIN", sd(s1.rho - rho0, 1.0 - rho0));
    let mut rhos: Vec<f64> = vec![rho0];
    rhos.extend(s1.branches.iter().map(|b| b.rho_final));
    rhos.extend(s1.votes.iter().map(|v| v.1));
    rhos.sort_by(|a, b| b.partial_cmp(a).unwrap());
    p("S1_BEST_MARGIN", if rhos.len() > 1 { rhos[0] - rhos[1] } else { 0.0 });
    let gm = gains.iter().sum::<f64>() / gains.len() as f64;
    p("S1_BRANCH_GAIN_MEAN", gm);
    p("S1_BRANCH_GAIN_STD", (gains.iter().map(|g| (g - gm).powi(2)).sum::<f64>() / gains.len() as f64).sqrt());
    p("S1_BRANCH_GAIN_MAX", gains.iter().cloned().fold(f64::NEG_INFINITY, f64::max));
    p("S1_BRANCH_GAIN_MIN", gains.iter().cloned().fold(f64::INFINITY, f64::min));
    p("S1_BRANCH_POS_RATIO", gains.iter().filter(|&&g| g > 0.0).count() as f64 / gains.len() as f64);
    let pos: Vec<f64> = gains.iter().map(|g| g.max(0.0)).collect();
    let pstot: f64 = pos.iter().sum();
    p("S1_BRANCH_GAIN_ENTROPY", if pstot > 1e-12 { entropy_norm(&pos.iter().map(|x| x / pstot).collect::<Vec<_>>(), 16) } else { 0.0 });
    p("S1_BRANCH_GAIN_HHI", if pstot > 1e-12 { pos.iter().map(|x| { let r = x / pstot; r * r }).sum() } else { 0.0 });
    let fam_gains = |fam: &str| -> Vec<f64> {
        s1.branches.iter().filter(|b| {
            let af = match b.axis.as_str() { "TIME" => "TIME", "ACTIVE_ORDER" => "ACTIVE", "PASSIVE_ORDER" => "PASSIVE", _ => "PRICE" };
            af == fam
        }).map(|b| b.rho_final - b.rho0).collect()
    };
    let fam_max = |fam: &str| fam_gains(fam).iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    for fam in ["TIME", "ACTIVE", "PASSIVE", "PRICE"] {
        p(&format!("S1_{}_FAMILY_MAX", fam), fam_max(fam));
        let g = fam_gains(fam); p(&format!("S1_{}_FAMILY_MEAN", fam), g.iter().sum::<f64>() / g.len() as f64);
    }
    p("S1_ACTIVE_MINUS_TIME", fam_max("ACTIVE") - fam_max("TIME"));
    p("S1_PASSIVE_MINUS_TIME", fam_max("PASSIVE") - fam_max("TIME"));
    p("S1_PRICE_MINUS_TIME", fam_max("PRICE") - fam_max("TIME"));
    let coarse: f64 = s1.branches.iter().filter(|b| b.depth == "N4" || b.depth == "N8").map(|b| (b.rho_final - b.rho0).max(0.0)).sum();
    let fine: f64 = s1.branches.iter().filter(|b| b.depth == "N32" || b.depth == "N16").map(|b| (b.rho_final - b.rho0).max(0.0)).sum();
    let tot = coarse + fine + 1e-12;
    p("S1_COARSE_SHARE", coarse / tot); p("S1_FINE_SHARE", fine / tot); p("S1_COARSE_MINUS_FINE", (coarse - fine) / tot);
    for (i, beta) in [1, 2, 5, 10].iter().enumerate() {
        p(&format!("S1_VOTE_B{}_CORR", beta), s1.votes[i].1);
        p(&format!("S1_VOTE_B{}_GAIN", beta), s1.votes[i].1 - rho0);
    }
    let pr = &s1.path_rhos;
    let pmean = pr.iter().sum::<f64>() / pr.len() as f64;
    p("S1_INIT_PATH_STD", (pr.iter().map(|x| (x - pmean).powi(2)).sum::<f64>() / pr.len() as f64).sqrt());
    p("S1_INIT_PATH_RANGE", pr.iter().cloned().fold(f64::NEG_INFINITY, f64::max) - pr.iter().cloned().fold(f64::INFINITY, f64::min));
    p("S1_TIME_ASC_MINUS_RANDOM", pr[0] - pr[3]);
    p("S1_TIME_DESC_MINUS_ASC", pr[1] - pr[0]);
    p("S1_ACTIVE_INIT_MINUS_TIME_INIT", pr[2] - pr[0]);
    f
}

pub fn s2_all(s2: &Stage2Result, rho0: f64) -> Factors {
    let mut f = vec![]; let mut p = |n: &str, v: f64| push(&mut f, n, v);
    let rho1 = s2.rho1;
    for b in &s2.branches {
        let name = format!("S2_{}_{}", b.graph, b.scope);
        let net = b.rho_final - rho1;
        let vals = [
            b.rho_final, net, sd(net, 1.0 - rho1), auc(&b.trajectory, rho1),
            sd(net, b.n_queries as f64),
            if b.accepts.is_empty() { 0.0 } else { b.accepts.iter().filter(|&&x| x).count() as f64 / b.accepts.len() as f64 },
            if b.node_ds.is_empty() { 0.0 } else { b.node_ds.iter().map(|d| d * d).sum::<f64>() / b.node_ds.len() as f64 },
            b.path_distance,
        ];
        for (suf, v) in S2_SUFFIX.iter().zip(vals.iter()) { let nm = format!("{}_{}", name, suf); p(&nm, *v); }
    }
    let find = |g: &str, sc: &str| s2.branches.iter().find(|b| b.graph == g && b.scope == sc);
    for g in GRAPH_NAMES.iter() {
        let gains: Vec<f64> = SCOPES.iter().map(|sc| find(g, sc).map(|b| b.rho_final - rho1).unwrap_or(0.0)).collect();
        p(&format!("S2_{}_GAIN_MAX_MODE", g), gains.iter().cloned().fold(f64::NEG_INFINITY, f64::max));
        let gm = gains.iter().sum::<f64>() / 3.0;
        p(&format!("S2_{}_GAIN_MEAN_MODE", g), gm);
        p(&format!("S2_{}_GAIN_STD_MODE", g), (gains.iter().map(|x| (x - gm).powi(2)).sum::<f64>() / 3.0).sqrt());
        p(&format!("S2_{}_LOCAL_GAIN", g), gains[0]); p(&format!("S2_{}_GLOBAL_GAIN", g), gains[2]);
        p(&format!("S2_{}_GLOBAL_MINUS_LOCAL", g), gains[2] - gains[0]);
    }
    let mut fam_gains: HashMap<&str, Vec<f64>> = HashMap::new();
    for (fam, graphs) in FAMILIES_GRAPHS.iter() {
        let mut gs = vec![];
        for g in graphs.iter() { for sc in SCOPES.iter() { if let Some(b) = find(g, sc) { gs.push(b.rho_final - rho1); } } }
        fam_gains.insert(fam, gs);
    }
    let all_pos: f64 = fam_gains.values().flat_map(|gs| gs.iter().map(|g| g.max(0.0))).sum::<f64>() + 1e-12;
    for fam in ["TIME", "ACTIVE", "PASSIVE", "PRICE", "SIDE", "BOOK", "QUEUE", "ARRIVAL"] {
        let gs = &fam_gains[fam];
        p(&format!("S2_{}_FAMILY_GAIN_MAX", fam), gs.iter().cloned().fold(f64::NEG_INFINITY, f64::max));
        p(&format!("S2_{}_FAMILY_GAIN_MEAN", fam), gs.iter().sum::<f64>() / gs.len() as f64);
        p(&format!("S2_{}_FAMILY_SHARE", fam), gs.iter().map(|g| g.max(0.0)).sum::<f64>() / all_pos);
        let gm = gs.iter().sum::<f64>() / gs.len() as f64;
        p(&format!("S2_{}_FAMILY_PARAM_STD", fam), if gs.len() > 1 { (gs.iter().map(|g| (g - gm).powi(2)).sum::<f64>() / gs.len() as f64).sqrt() } else { 0.0 });
    }
    let fmax2 = |fam: &str| -> f64 {
        let graphs = FAMILIES_GRAPHS.iter().find(|(fn_, _)| *fn_ == fam).unwrap().1;
        s2.branches.iter().filter(|b| graphs.contains(&b.graph.as_str())).map(|b| b.rho_final - rho1).fold(f64::NEG_INFINITY, f64::max)
    };
    for (a, b) in [("ACTIVE","TIME"),("PASSIVE","TIME"),("PRICE","TIME"),("SIDE","TIME"),("BOOK","TIME"),("QUEUE","TIME"),("ARRIVAL","TIME"),("ACTIVE","PASSIVE"),("BOOK","PRICE"),("QUEUE","BOOK"),("ARRIVAL","BOOK"),("SIDE","BOOK")] {
        p(&format!("S2_{}_MINUS_{}", a, b), fmax2(a) - fmax2(b));
    }
    p("S2_ACTIVE_TO_TIME_RATIO", sd(fmax2("ACTIVE"), fmax2("TIME").abs()));
    p("S2_PASSIVE_TO_TIME_RATIO", sd(fmax2("PASSIVE"), fmax2("TIME").abs()));
    p("S2_BOOK_TO_PRICE_RATIO", sd(fmax2("BOOK"), fmax2("PRICE").abs()));
    p("S2_QUEUE_TO_BOOK_RATIO", sd(fmax2("QUEUE"), fmax2("BOOK").abs()));
    p("S2_FINAL_CORR", s2.rho); p("S2_INCREMENTAL_GAIN", s2.rho - rho1); p("S2_TOTAL_GAIN", s2.rho - rho0);
    p("S2_NORM_INCREMENTAL_GAIN", sd(s2.rho - rho1, 1.0 - rho1));
    let mut all_rhos: Vec<f64> = vec![rho1, s2.rho];
    all_rhos.extend(s2.votes.iter().map(|v| v.1));
    all_rhos.sort_by(|a, b| b.partial_cmp(a).unwrap());
    p("S2_BEST_MARGIN", if all_rhos.len() > 1 { all_rhos[0] - all_rhos[1] } else { 0.0 });
    let gg: Vec<f64> = s2.branches.iter().map(|b| b.rho_final - rho1).collect();
    let ggm = gg.iter().sum::<f64>() / gg.len() as f64;
    p("S2_GRAPH_GAIN_MEAN", ggm);
    p("S2_GRAPH_GAIN_STD", (gg.iter().map(|g| (g - ggm).powi(2)).sum::<f64>() / gg.len() as f64).sqrt());
    p("S2_GRAPH_GAIN_MAX", gg.iter().cloned().fold(f64::NEG_INFINITY, f64::max));
    p("S2_GRAPH_POS_RATIO", gg.iter().filter(|&&g| g > 0.0).count() as f64 / gg.len() as f64);
    let shares: Vec<f64> = ["TIME", "ACTIVE", "PASSIVE", "PRICE", "SIDE", "BOOK", "QUEUE", "ARRIVAL"].iter().map(|fam| fmax2(fam).max(0.0)).collect();
    let stot: f64 = shares.iter().sum::<f64>() + 1e-12;
    let sp: Vec<f64> = shares.iter().map(|s| s / stot).collect();
    p("S2_GRAPH_MECHANISM_ENTROPY", entropy_norm(&sp, 8));
    p("S2_GRAPH_MECHANISM_HHI", sp.iter().map(|s| s * s).sum());
    let mut so = (0..8).collect::<Vec<usize>>(); so.sort_by(|&a, &b| shares[b].partial_cmp(&shares[a]).unwrap());
    p("S2_WINNER_MARGIN", (shares[so[0]] - shares[so[1]]) / stot);
    p("S2_TOP2_FAMILY_SHARE", (shares[so[0]] + shares[so[1]]) / stot);
    let lp: f64 = s2.branches.iter().filter(|b| b.scope == "LOCAL32").map(|b| (b.rho_final - rho1).max(0.0)).sum();
    let mp: f64 = s2.branches.iter().filter(|b| b.scope == "MESO8_D2").map(|b| (b.rho_final - rho1).max(0.0)).sum();
    let gp: f64 = s2.branches.iter().filter(|b| b.scope == "GLOBAL_D4").map(|b| (b.rho_final - rho1).max(0.0)).sum();
    let stt = lp + mp + gp + 1e-12;
    p("S2_LOCAL_SHARE", lp / stt); p("S2_MESO_SHARE", mp / stt); p("S2_GLOBAL_SHARE", gp / stt);
    p("S2_GLOBAL_MINUS_LOCAL", (gp / 15.0) - (lp / 32.0));
    for (i, beta) in [1, 2, 5, 10].iter().enumerate() {
        p(&format!("S2_VOTE_B{}_CORR", beta), s2.votes[i].1);
        p(&format!("S2_VOTE_B{}_GAIN", beta), s2.votes[i].1 - rho1);
    }
    f
}

pub fn s3_contrib_cross(s1: &Stage1Result, s2: &Stage2Result, s3: Option<&Stage3Result>) -> Factors {
    let mut f = vec![]; let mut p = |n: &str, v: f64| push(&mut f, n, v);
    if let Some(s3) = s3 {
        for t in s3.taus.iter() {
            let tn = format!("TAU{:03}", (t.tau * 100.0) as i64);
            for &ck in &[0usize, 8, 16, 32, 64] {
                if let Some(stats) = t.checkpoints.get(&ck) {
                    for (stat, idx) in S3_STATS.iter() { p(&format!("S3_{}_K{}_{}", tn, ck, stat), stats[*idx]); }
                }
            }
            let h = |k: usize| t.checkpoints.get(&k).map(|v| v[0]).unwrap_or(0.0);
            let h0 = h(0); let h8 = h(8); let h16 = h(16); let h32 = h(32); let h64 = h(64);
            p(&format!("S3_{}_TOTAL_INFO_GAIN", tn), h0 - h64);
            p(&format!("S3_{}_INFO_GAIN_0_8", tn), h0 - h8);
            p(&format!("S3_{}_INFO_GAIN_8_16", tn), h8 - h16);
            p(&format!("S3_{}_INFO_GAIN_16_32", tn), h16 - h32);
            p(&format!("S3_{}_INFO_GAIN_32_64", tn), h32 - h64);
            p(&format!("S3_{}_EARLY_INFO_SHARE8", tn), sd(h0 - h8, h0 - h64));
            p(&format!("S3_{}_EARLY_INFO_SHARE16", tn), sd(h0 - h16, h0 - h64));
            let ig = &t.info_gains;
            p(&format!("S3_{}_QUERY_INFO_GAIN_MAX", tn), ig.iter().cloned().fold(f64::NEG_INFINITY, f64::max));
            let igm = ig.iter().sum::<f64>() / ig.len() as f64;
            p(&format!("S3_{}_QUERY_INFO_GAIN_MEAN", tn), igm);
            p(&format!("S3_{}_QUERY_INFO_GAIN_STD", tn), (ig.iter().map(|x| (x - igm).powi(2)).sum::<f64>() / ig.len() as f64).sqrt());
            p(&format!("S3_{}_QUERY_INFO_GAIN_GINI", tn), gini(ig));
            p(&format!("S3_{}_PREDICTED_VARIANCE_MEAN", tn), t.pred_vars.iter().sum::<f64>() / t.pred_vars.len() as f64);
            p(&format!("S3_{}_FINAL_POSTERIOR_CORR", tn), t.rho3);
            p(&format!("S3_{}_FINAL_GAIN_VS_STAGE2", tn), t.rho3 - t.rho2);
            for (tp_i, tp) in ["DISAGREEMENT", "STRUCTURE", "ENTROPY", "RANDOM"].iter().enumerate() {
                let cnt = t.query_types.iter().filter(|&&q| q as usize == tp_i).count();
                p(&format!("S3_{}_{}_SHARE", tn, tp), cnt as f64 / t.query_types.len() as f64);
                let g: Vec<f64> = t.query_types.iter().enumerate().filter(|(_, &q)| q as usize == tp_i).map(|(i, _)| ig[i]).collect();
                p(&format!("S3_{}_{}_GAIN_MEAN", tn, tp), if g.is_empty() { 0.0 } else { g.iter().sum::<f64>() / g.len() as f64 });
            }
        }
    }
    let rho0 = s1.rho0; let rho1 = s1.rho; let rho2 = s2.rho;
    let g1 = rho1 - rho0; let g2 = rho2 - rho1;
    p("X_S1_GAIN", g1); p("X_S2_GAIN", g2);
    if let Some(s3) = s3 {
        for t in s3.taus.iter() {
            let tn = format!("TAU{:03}", (t.tau * 100.0) as i64);
            let g3 = t.rho3 - rho2;
            p(&format!("X_S3_GAIN_{}", tn), g3);
            p(&format!("X_S2_TO_S3_RATIO_{}", tn), sd(g2, g3.abs()));
        }
    }
    p("X_S1_SHARE_12", sd(g1, g1 + g2)); p("X_S2_SHARE_12", sd(g2, g1 + g2));
    p("X_S1_TO_S2_RATIO", sd(g1, g2.abs()));
    // ========== 交叉因子(25) ==========
    let s1gains: Vec<f64> = s1.branches.iter().map(|b| b.rho_final - b.rho0).collect();
    let s1pos: Vec<f64> = s1gains.iter().map(|g| g.max(0.0)).collect();
    let s1pstot: f64 = s1pos.iter().sum::<f64>() + 1e-12;
    let s1_bge = entropy_norm(&s1pos.iter().map(|x| x / s1pstot).collect::<Vec<_>>(), 16);
    let coarse: f64 = s1.branches.iter().filter(|b| b.depth == "N4" || b.depth == "N8").map(|b| (b.rho_final - b.rho0).max(0.0)).sum();
    let fine: f64 = s1.branches.iter().filter(|b| b.depth == "N32" || b.depth == "N16").map(|b| (b.rho_final - b.rho0).max(0.0)).sum();
    let ctot = coarse + fine + 1e-12;
    let s1_coarse = coarse / ctot; let s1_fine = fine / ctot;
    let rho1 = s2.rho1;
    let fam_gmax = |fam: &str| -> f64 {
        let graphs = FAMILIES_GRAPHS.iter().find(|(fn_, _)| *fn_ == fam).unwrap().1;
        s2.branches.iter().filter(|b| graphs.contains(&b.graph.as_str())).map(|b| b.rho_final - rho1).fold(f64::NEG_INFINITY, f64::max)
    };
    let fam_share = |fam: &str| fam_gmax(fam).max(0.0) / (fam_gmax(fam).max(0.0) + 1e-12);
    let shares8: Vec<f64> = ["TIME","ACTIVE","PASSIVE","PRICE","SIDE","BOOK","QUEUE","ARRIVAL"].iter().map(|f| fam_gmax(f).max(0.0)).collect();
    let stot8: f64 = shares8.iter().sum::<f64>() + 1e-12;
    let sp8: Vec<f64> = shares8.iter().map(|s| s / stot8).collect();
    let s2_me = entropy_norm(&sp8, 8);
    let fs = |fam: &str| fam_gmax(fam).max(0.0) / stot8;
    let active_minus_time = fam_gmax("ACTIVE") - fam_gmax("TIME");
    let book_minus_time = fam_gmax("BOOK") - fam_gmax("TIME");
    p("X_SCALE_ENTROPY_MINUS_GRAPH_ENTROPY", s1_bge - s2_me);
    p("X_COARSE_SHARE_X_ACTIVE_SHARE", s1_coarse * fs("ACTIVE"));
    p("X_COARSE_SHARE_X_BOOK_SHARE", s1_coarse * fs("BOOK"));
    p("X_FINE_SHARE_X_QUEUE_SHARE", s1_fine * fs("QUEUE"));
    p("X_FINE_SHARE_X_ARRIVAL_SHARE", s1_fine * fs("ARRIVAL"));
    p("X_ORDER_IDENTITY_DOMINANCE", (fam_gmax("ACTIVE") + fam_gmax("PASSIVE")) / 2.0 - fam_gmax("TIME"));
    p("X_BOOK_QUEUE_DOMINANCE", (fam_gmax("BOOK") + fam_gmax("QUEUE")) / 2.0 - fam_gmax("PRICE"));
    if let Some(s3) = s3 {
        for t in s3.taus.iter() {
            let tn = format!("TAU{:03}", (t.tau * 100.0) as i64);
            let last_k = t.checkpoints.keys().copied().max().unwrap_or(0);
            let ckl = t.checkpoints.get(&last_k).unwrap();
            let total_info = t.checkpoints[&0][0] - ckl[0];
            p(&format!("X_ACTIVE_GAIN_X_TAIL10_ENTROPY_{}", tn), fam_gmax("ACTIVE") * ckl[8]);
            p(&format!("X_BOOK_GAIN_X_TAIL10_CONF_{}", tn), fam_gmax("BOOK") * ckl[11]);
            p(&format!("X_GRAPH_ENTROPY_X_TOTAL_INFO_{}", tn), s2_me * total_info);
            p(&format!("X_SCALE_ENTROPY_X_FINAL_ENTROPY_{}", tn), s1_bge * ckl[0]);
            p(&format!("X_HIDDEN_LARGE_ORDER_{}", tn), active_minus_time * ckl[9]);
            p(&format!("X_VISIBLE_LARGE_LIQUIDITY_{}", tn), book_minus_time * ckl[12]);
        }
    }
    f
}

/// 汇总所有基础因子
pub fn extract_all(s1: &Stage1Result, s2: &Stage2Result, s3: Option<&Stage3Result>, y: &[f64]) -> Factors {
    let mut f = vec![];
    f.extend(target_dist(y));
    f.extend(s1_branches(s1));
    f.extend(s1_summary(s1));
    f.extend(s2_all(s2, s1.rho0));
    f.extend(s3_contrib_cross(s1, s2, s3));
    f
}
