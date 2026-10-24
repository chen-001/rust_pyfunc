// lite 门验证: 合成 n=8 MatrixSet（28 张随机矩阵），ctx.with_keep(全部 LITE 名单)，
// 跑我负责的 10 个模块 compute，检查产出列 ⊆ LITE（不会多推未幸存列）。
// 用法: lite_gate_check   输出每模块产出列数与违规列。
use rust_pyfunc::yupei_dist::indicator_ctx::{IndicatorCtx, MatrixSet};
use rust_pyfunc::yupei_dist::indicators::generated;
use rust_pyfunc::yupei_dist::lite_names::YUPEI_DIST_LITE_NAMES as LITE;
use rust_pyfunc::yupei_dist::matrix_stage::StockStats;
use std::collections::{HashMap, HashSet};

fn main() {
    let n = 8usize;
    let mut x = 0x9E3779B97F4A7C15u64;
    let mut next = move || {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        ((x >> 11) & 0xF_FFFF) as f32 / 0xF_FFFF as f32 + 0.01
    };
    let codes: Vec<String> = (0..n).map(|i| format!("S{i:04}")).collect();
    let stats = vec![StockStats::default(); n];
    let mut mats = HashMap::new();
    for name in rust_pyfunc::yupei_dist::indicator_ctx::MATRIX_LIST {
        mats.insert(name.to_string(), (0..n * n).map(|_| next()).collect());
    }
    let set = MatrixSet { n, codes, stats, mats, industry: None };
    let keep: HashSet<&'static str> = LITE.iter().copied().collect();
    let ctx = IndicatorCtx::with_keep(&set, None, keep.clone());
    let lite: HashSet<&str> = LITE.iter().copied().collect();
    let mods: Vec<(&str, fn(&IndicatorCtx) -> Vec<rust_pyfunc::yupei_dist::indicator_ctx::IndicatorResult>)> = vec![
        ("maxst", generated::maxst::compute),
        ("strength", generated::strength::compute),
        ("topk", generated::topk::compute),
        ("concentration", generated::concentration::compute),
        ("neighbor_strength", generated::neighbor_strength::compute),
        ("neighbor_dispersion", generated::neighbor_dispersion::compute),
        ("neighbor_z", generated::neighbor_z::compute),
        ("network_momentum", generated::network_momentum::compute),
        ("multiscale", generated::multiscale::compute),
        ("reciprocity", generated::reciprocity::compute),
    ];
    let mut bad = 0;
    for (mname, f) in mods {
        let out = f(&ctx);
        let extra: Vec<&str> = out.iter().map(|r| r.name.as_str()).filter(|nm| !lite.contains(*nm)).collect();
        if !extra.is_empty() {
            bad += extra.len();
            println!("{mname}: {} cols, EXTRA {extra:?}", out.len());
        } else {
            println!("{mname}: {} cols, all in LITE", out.len());
        }
    }
    println!("bad={bad}");
    // 全量模式（keep=None）不受影响: maxst 应产 84 列
    let ctx2 = IndicatorCtx::new(&set, None);
    println!("full-mode maxst cols = {}", generated::maxst::compute(&ctx2).len());
}
