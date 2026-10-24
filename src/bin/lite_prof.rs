// lite 版分段计时: 读数据+矩阵构建、对称化预热、每个指标模块各花多少秒。
// 用法: lite_prof [date]
use rust_pyfunc::yupei_dist::compute::compute_yupei_dist_partial;
use rust_pyfunc::yupei_dist::indicator_ctx::{IndicatorCtx, MatrixSet};
use rust_pyfunc::yupei_dist::lite_names::YUPEI_DIST_LITE_NAMES as LITE;
use rust_pyfunc::yupei_dist::matrix_stage::MATRIX_SPECS;
use std::collections::HashSet;
use std::time::Instant;

fn main() {
    let date: i64 = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(20251031);

    // partial 内部两段拆开计时: 读+预处理 vs 28矩阵
    let t0 = Instant::now();
    let codes = rust_pyfunc::data_paths::list_stock_codes(date, "transaction");
    let day_start_us = rust_pyfunc::yupei_dist::compute::day_start_us_of(date);
    use rayon::prelude::*;
    use rust_pyfunc::yupei_dist::matrix_stage::{prep_stock, StockPrep};
    let stocks: Vec<StockPrep> = codes
        .par_iter()
        .map(|c| {
            rust_pyfunc::fast_csv_reader::read_trade_fast_inner(c, date, false, true, usize::MAX)
                .ok()
                .and_then(|recs| prep_stock(c, &recs, day_start_us, 1))
                .unwrap_or_else(|| StockPrep::empty(c.clone()))
        })
        .collect();
    eprintln!("[read+prep] {:.1}s", t0.elapsed().as_secs_f64());
    let t = Instant::now();
    let (mats, stats) = rust_pyfunc::yupei_dist::matrix_stage::compute_matrices(&stocks, rust_pyfunc::yupei_dist::matrix_stage::FAM_ALL);
    eprintln!("[matrices] {:.1}s", t.elapsed().as_secs_f64());
    let t = Instant::now();
    let codes_out: Vec<String> = stocks.iter().map(|s| s.code.clone()).collect();
    let n = codes_out.len();
    let mut hm = std::collections::HashMap::new();
    for (m, spec) in mats.into_iter().zip(MATRIX_SPECS.iter()) {
        hm.insert(spec.name.to_string(), m);
    }
    let industry = rust_pyfunc::yupei_dist::industry::industry_of(date, &codes_out).unwrap();
    eprintln!("[industry+move] {:.1}s  n={}", t.elapsed().as_secs_f64(), n);

    let set = MatrixSet { n, codes: codes_out.clone(), stats, mats: hm, industry: Some(industry) };
    let keep: HashSet<&'static str> = LITE.iter().copied().collect();
    let ctx = IndicatorCtx::with_keep(&set, None, keep);

    // 预热: lite 里用到的矩阵先做对称化（模块里第一次用时才建, 这里单独计时）
    let used: HashSet<&str> = LITE
        .iter()
        .filter_map(|nm| MATRIX_SPECS.iter().find(|s| nm.contains(s.name)).map(|s| s.name))
        .collect();
    let t = Instant::now();
    for m in &used {
        let _ = ctx.symmetric(m);
    }
    eprintln!("[sym-prewarm] {} 张矩阵对称化: {:.1}s", used.len(), t.elapsed().as_secs_f64());

    // 并行跑模块（与 run_indicators_lite 同构）：先单独计 percolation 的 mat 循环
    let t = Instant::now();
    let per_mat: Vec<usize> = rust_pyfunc::yupei_dist::indicator_ctx::MATRIX_LIST
        .par_iter()
        .filter(|m| ctx.wants("perc_", m))
        .map(|m| m.len())
        .collect();
    eprintln!("[perc mats wanted] {} 张 (预热对称化)", per_mat.len());
    let t = Instant::now();
    let per_mod: Vec<(String, Vec<rust_pyfunc::yupei_dist::indicator_ctx::IndicatorResult>)> =
        rust_pyfunc::yupei_dist::indicators::all()
            .into_par_iter()
            .map(|def| (def.name.to_string(), (def.compute)(&ctx)))
            .collect();
    eprintln!("[all modules parallel] {:.1}s", t.elapsed().as_secs_f64());
    let mut ncol = 0usize;
    for (name, rs) in &per_mod {
        ncol += rs.len();
        eprintln!("  {name:<24} {} 列", rs.len());
    }
    eprintln!("总列数: {ncol}");
}
