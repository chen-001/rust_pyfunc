//! tail_perf_bench：tail_pipeline_engine 计算核心的 sandbox 性能基准。
//! 用法:
//!   tail_perf_bench replica [--fold] [--neu]    生产复刻（逐位对账用）
//!   tail_perf_bench fused   [--fold]            融合排序优化版回测（对账 + 测速）
//!   tail_perf_bench prerank [--fold]            预计算收益秩近似版（偏差 + 测速）
//!   tail_perf_bench rolling                     列主序 vs 行主序 rolling 对比
//!   tail_perf_bench hat                         中性化 hat 矩阵预计算原型（合成数据）
mod engine;
mod neu;
mod npy;
mod opt;
mod v2;
mod btopt;
mod preflightopt;

use std::time::Instant;

use ndarray::{Array2, ArrayView2, Axis};
use nalgebra::{Cholesky, DMatrix};

const DATA: &str = "/home/chenzongwei/pythoncode/tail_perf_lab/data";
const DATA_URGENCY: &str = "/home/chenzongwei/pythoncode/tail_perf_lab/data_urgency";

fn load_shared(fold: bool) -> engine::Shared {
    let dates = npy::as_i32_vec1(npy::load(&format!("{DATA}/dates.npy")));
    engine::Shared {
        cover_rate: 0.5,
        dates,
        windows: vec![5, 10, 20],
        fold,
        backtest_start: 20170201,
        ret_gap1: npy::as_f32_mat(npy::load(&format!("{DATA}/ret_gap1.npy"))),
        ret_sum_gap1: npy::as_f32_mat(npy::load(&format!("{DATA}/ret_sum_gap1.npy"))),
        ret_gap5: npy::as_f32_mat(npy::load(&format!("{DATA}/ret_gap5.npy"))),
        ret_sum_gap5: npy::as_f32_mat(npy::load(&format!("{DATA}/ret_sum_gap5.npy"))),
        restrict: npy::as_f32_mat(npy::load(&format!("{DATA}/restrict.npy"))),
        index_ret: npy::as_f32_vec1(npy::load(&format!("{DATA}/index_ret.npy"))).into(),
        majority_count_threshold: 200.0,
        zero_max_threshold: 0.1,
        nan_max_threshold: 0.04,
    }
}

fn load_factors() -> Vec<(String, Array2<f32>)> {
    let names = std::fs::read_to_string(format!("{DATA}/sample_names.txt")).unwrap();
    names
        .lines()
        .map(|nm| {
            let mat = npy::as_f32_mat(npy::load(&format!("{DATA}/factor_{nm}.npy")));
            (nm.to_string(), mat)
        })
        .collect()
}

fn derived_names(variant_name: &str, windows: &[usize]) -> Vec<String> {
    let mut names = vec![format!("{}_smooth_1", variant_name)];
    for &window in windows {
        names.push(format!("{}_mean_smooth_{}", variant_name, window));
        names.push(format!("{}_max_smooth_{}", variant_name, window));
        names.push(format!("{}_min_smooth_{}", variant_name, window));
        names.push(format!("{}_std_smooth_{}", variant_name, window));
    }
    names
}

fn engine_precompute_open(restrict: &ArrayView2<'_, f32>) -> Vec<usize> {
    (0..restrict.shape()[0])
        .map(|r| {
            restrict
                .row(r)
                .iter()
                .filter(|&&v| v.is_finite() && v == 0.0)
                .count()
        })
        .collect()
}

/// btopt 两个结果逐位一致判定（供 trunc 模式对账）。
fn btopt_results_eq(
    a: &engine::LegacyBacktestResult,
    b: &engine::LegacyBacktestResult,
) -> bool {
    btopt::result_bitwise_eq(a, b)
}

fn preflight_pass(
    slot: &ArrayView2<'_, f32>,
    shared: &engine::Shared,
) -> bool {
    engine::preflight_quality_check(
        &slot.view(),
        &shared.restrict.view(),
        shared.majority_count_threshold,
        shared.zero_max_threshold,
        shared.nan_max_threshold,
    )
    .passed
}

// ---- 融合排序版 variant 流水线：结构与 run_variant 一致，raw 回测换成 fused ----
fn run_variant_fused(
    variant_name: &str,
    ranked: Array2<f32>,
    shared: &engine::Shared,
    open_symbol_counts: &[usize],
    summaries: &mut Vec<(String, i32, [f64; 10])>,
    time_bt: &mut f64,
) {
    let names = derived_names(variant_name, &shared.windows);
    let mut slot_idx = 0usize;
    let mut process_slot = |slot: ArrayView2<'_, f32>| {
        let name = names[slot_idx].clone();
        slot_idx += 1;
        if !preflight_pass(&slot, shared) {
            return;
        }
        // 计时区与生产 legacy_backtest_gap1_gap5_single_slot 一致：
        // unique 检查 + effective indices + 两次 gap 回测
        let t = Instant::now();
        let slot_block = slot.insert_axis(Axis(2));
        let n_dates = slot.nrows();
        if n_dates >= 2 && has_enough_unique_view(&slot_block) {
            let eff = effective_indices_view(&slot_block, &shared.dates, shared.backtest_start);
            let r1 = opt::backtest_fused(
                &slot_block,
                &shared.ret_gap1.view(),
                &shared.ret_sum_gap1.view(),
                &shared.restrict.view(),
                &shared.index_ret.view(),
                &shared.dates,
                0,
                1,
                10,
                &eff,
                open_symbol_counts,
            );
            let r5 = opt::backtest_fused(
                &slot_block,
                &shared.ret_gap5.view(),
                &shared.ret_sum_gap5.view(),
                &shared.restrict.view(),
                &shared.index_ret.view(),
                &shared.dates,
                0,
                5,
                10,
                &eff,
                open_symbol_counts,
            );
            summaries.push((name.clone(), 1, r1.summary));
            summaries.push((name, 5, r5.summary));
        }
        *time_bt += t.elapsed().as_secs_f64();
    };

    process_slot(ranked.view());
    for &window in &shared.windows {
        let min_periods = std::cmp::max(1, window / 2);
        let (mean, max, min, std) =
            engine::rolling_stats_f32_serial(&ranked, window, min_periods);
        process_slot(mean.view());
        drop(mean);
        process_slot(max.view());
        drop(max);
        process_slot(min.view());
        drop(min);
        process_slot(std.view());
        drop(std);
    }
}

// ---- prerank 版 variant 流水线：结构与 fused 一致，IC 用预计算秩 ----
fn run_variant_prerank(
    variant_name: &str,
    ranked: Array2<f32>,
    shared: &engine::Shared,
    open_symbol_counts: &[usize],
    pr1: &opt::PrecomputedRetRanks,
    pr5: &opt::PrecomputedRetRanks,
    summaries: &mut Vec<(String, i32, [f64; 10])>,
    time_bt: &mut f64,
) {
    let names = derived_names(variant_name, &shared.windows);
    let mut slot_idx = 0usize;
    let mut process_slot = |slot: ArrayView2<'_, f32>| {
        let name = names[slot_idx].clone();
        slot_idx += 1;
        if !preflight_pass(&slot, shared) {
            return;
        }
        let t = Instant::now();
        let slot_block = slot.insert_axis(Axis(2));
        let n_dates = slot.nrows();
        if n_dates >= 2 && has_enough_unique_view(&slot_block) {
            let eff = effective_indices_view(&slot_block, &shared.dates, shared.backtest_start);
            let r1 = opt::backtest_prerank(
                &slot_block,
                &shared.ret_gap1.view(),
                &shared.restrict.view(),
                &shared.index_ret.view(),
                &shared.dates,
                0,
                1,
                10,
                &eff,
                open_symbol_counts,
                pr1,
            );
            let r5 = opt::backtest_prerank(
                &slot_block,
                &shared.ret_gap5.view(),
                &shared.restrict.view(),
                &shared.index_ret.view(),
                &shared.dates,
                0,
                5,
                10,
                &eff,
                open_symbol_counts,
                pr5,
            );
            summaries.push((name.clone(), 1, r1.summary));
            summaries.push((name, 5, r5.summary));
        }
        *time_bt += t.elapsed().as_secs_f64();
    };

    process_slot(ranked.view());
    for &window in &shared.windows {
        let min_periods = std::cmp::max(1, window / 2);
        let (mean, max, min, std) =
            engine::rolling_stats_f32_serial(&ranked, window, min_periods);
        process_slot(mean.view());
        drop(mean);
        process_slot(max.view());
        drop(max);
        process_slot(min.view());
        drop(min);
        process_slot(std.view());
        drop(std);
    }
}

fn has_enough_unique_view(factor: &ndarray::ArrayView3<'_, f32>) -> bool {
    let mut seen = std::collections::HashSet::<u32>::new();
    for raw_idx in 0..factor.shape()[0].saturating_sub(1) {
        for stock_idx in 0..factor.shape()[1] {
            let value = factor[[raw_idx, stock_idx, 0]];
            if value.is_finite() {
                seen.insert(value.to_bits());
                if seen.len() >= 10 {
                    return true;
                }
            }
        }
    }
    false
}

fn effective_indices_view(
    factor: &ndarray::ArrayView3<'_, f32>,
    dates: &[i32],
    backtest_start: i32,
) -> Vec<usize> {
    let n_dates = factor.shape()[0];
    let n_stocks = factor.shape()[1];
    let mut out = Vec::new();
    for raw_eff_idx in 1..n_dates {
        if dates[raw_eff_idx] <= backtest_start {
            continue;
        }
        let mut all_nan = true;
        for stock_idx in 0..n_stocks {
            if factor[[raw_eff_idx - 1, stock_idx, 0]].is_finite() {
                all_nan = false;
                break;
            }
        }
        if !all_nan {
            out.push(raw_eff_idx);
        }
    }
    out
}

fn print_summaries_json(tag: &str, summaries: &[(String, i32, [f64; 10])]) {
    let mut s = String::new();
    s.push('[');
    for (i, (name, gap, v)) in summaries.iter().enumerate() {
        if i > 0 {
            s.push(',');
        }
        s.push_str(&format!(
            "{{\"name\":\"{name}\",\"gap\":{gap},\"v\":[{},{},{},{},{},{},{},{},{},{}]}}",
            v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8], v[9]
        ));
    }
    s.push(']');
    println!("JSON {tag} {s}");
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mode = args.get(1).map(|s| s.as_str()).unwrap_or("replica");
    let fold = args.iter().any(|a| a == "--fold");
    let neu = args.iter().any(|a| a == "--neu");

    if mode == "rolling" {
        let factors = load_factors();
        let (_, raw) = &factors[0];
        let t = Instant::now();
        let ranked = engine::rank_axis1_average_f32_serial(raw);
        let rank_s = t.elapsed().as_secs_f64();
        for &w in &[5usize, 10, 20] {
            let t = Instant::now();
            let a = engine::rolling_stats_f32_serial(&ranked, w, w / 2);
            let t_serial = t.elapsed().as_secs_f64();
            let t = Instant::now();
            let b = engine::rolling_stats_f32_rowmajor(&ranked, w, w / 2);
            let t_row = t.elapsed().as_secs_f64();
            let (a1, _, _, _) = a;
            let (b1, _, _, _) = b;
            let diff = (&a1 - &b1).mapv(f64::from).fold(0.0f64, |m, x| m.max(x.abs()));
            println!(
                "rolling w={w}: serial={t_serial:.3}s rowmajor={t_row:.3}s speedup={:.2}x maxdiff={:e}",
                t_serial / t_row,
                diff
            );
        }
        println!("rank: {rank_s:.3}s");
        return;
    }

    if mode == "hat" {
        bench_hat();
        return;
    }

    if mode == "neu" {
        bench_neu();
        return;
    }

    if mode == "preflight" {
        // 现生产 radix preflight 成本测量 + 位图/zerofast 变体对照
        let shared = load_shared(true);
        let t = Instant::now();
        let bitmap = preflightopt::build_restrict_bitmap(&shared.restrict.view());
        println!("restrict bitmap build: {:.2}s", t.elapsed().as_secs_f64());
        let factors = load_factors();
        let (maj_thr, zero_thr, nan_thr) = (200.0_f64, 0.1_f64, 0.04_f64);
        for (name, raw) in &factors {
            let ranked = engine::rank_and_fill_missing_cross_sectional_median(raw);
            let folded = engine::build_fold_values(raw);
            let ranked_fold = engine::rank_and_fill_missing_cross_sectional_median(&folded);
            // 13+13 slots
            let mut slots: Vec<Array2<f32>> = vec![ranked.clone()];
            let mut slots_f: Vec<Array2<f32>> = vec![ranked_fold.clone()];
            for &w in &[5usize, 10, 20] {
                let (m, x, n2, s) = engine::rolling_stats_f32_serial(&ranked, w, w / 2);
                slots.push(m); slots.push(x); slots.push(n2); slots.push(s);
                let (m, x, n2, s) = engine::rolling_stats_f32_serial(&ranked_fold, w, w / 2);
                slots_f.push(m); slots_f.push(x); slots_f.push(n2); slots_f.push(s);
            }
            slots.extend(slots_f);
            let t = Instant::now();
            let mut reports_prod = Vec::new();
            for slot in &slots {
                reports_prod.push(preflightopt::preflight_prod(
                    &slot.view(), &shared.restrict.view(), maj_thr, zero_thr, nan_thr,
                ));
            }
            let t_prod = t.elapsed().as_secs_f64();
            let t = Instant::now();
            let mut reports_bm = Vec::new();
            for slot in &slots {
                reports_bm.push(preflightopt::preflight_bitmap(
                    &slot.view(), &bitmap, maj_thr, zero_thr, nan_thr,
                ));
            }
            let t_bm = t.elapsed().as_secs_f64();
            let t = Instant::now();
            let mut reports_zf = Vec::new();
            for slot in &slots {
                reports_zf.push(preflightopt::preflight_zerofast(
                    &slot.view(), &shared.restrict.view(), maj_thr, zero_thr, nan_thr,
                ));
            }
            let t_zf = t.elapsed().as_secs_f64();
            let eq_bm = reports_prod.iter().zip(reports_bm.iter()).all(|(a, b)| {
                a.passed == b.passed
                    && a.majority_count_mean == b.majority_count_mean
                    && a.zero_ratio_mean == b.zero_ratio_mean
                    && a.nan_ratio_mean == b.nan_ratio_mean
            });
            let npass = reports_prod.iter().filter(|r| r.passed).count();
            println!(
                "PREFLIGHT {name}: prod={t_prod:.3}s bitmap={t_bm:.3}s(eq={eq_bm}) zerofast={t_zf:.3}s slots={} passed={npass}",
                slots.len()
            );
        }
        return;
    }

    if mode == "trunc" {
        // 中性化日期截断实验: bt 只读 dates > backtest_start 的信号行 (最早 idx0-1)，
        // 更早的行永不读取 → 中性化只做 idx0-1 之后的行，bt 结果应逐位一致。
        let shared = load_shared(true);
        let pre = btopt::build_bt_precomputed(&shared.ret_sum_gap1, &shared.ret_sum_gap5);
        let open_counts = engine_precompute_open(&shared.restrict.view());
        // idx0 = 第一个 dates[idx] > backtest_start
        let idx0 = shared
            .dates
            .iter()
            .position(|&d| d > shared.backtest_start)
            .unwrap();
        let cut = idx0 - 1; // 保留信号行 idx0-1
        println!(
            "trunc: backtest_start={} idx0={} cut={} 跳过 {} / {} 行 ({:.1}%)",
            shared.backtest_start,
            idx0,
            cut,
            cut,
            shared.dates.len(),
            cut as f64 / shared.dates.len() as f64 * 100.0
        );
        let v2shared = v2::build_shared(DATA);
        let factors = load_factors();
        for (name, raw) in &factors {
            let ranked = engine::rank_and_fill_missing_cross_sectional_median(raw);
            let (m5, _x5, _n5, s5) = engine::rolling_stats_f32_serial(&ranked, 5, 2);
            let slots: Vec<(&str, &Array2<f32>)> =
                vec![("smooth_1", &ranked), ("mean_5", &m5), ("std_5", &s5)];
            for (tag, slot) in &slots {
                // 全量中性化
                let t = Instant::now();
                let (full, _) = v2::v2_slot(slot.view(), &v2shared, "opt2");
                let t_full = t.elapsed().as_secs_f64();
                // 截断中性化: 只处理 cut..T 行 (输出仍是全 T 矩阵, 前面为 NaN)
                let t = Instant::now();
                let sub = slot.slice(ndarray::s![cut.., ..]);
                let (sub_out, _) = v2::v2_slot(sub, &v2shared, "opt2");
                let mut trunc_out = Array2::<f32>::from_elem(slot.dim(), f32::NAN);
                trunc_out.slice_mut(ndarray::s![cut.., ..]).assign(&sub_out);
                let t_trunc = t.elapsed().as_secs_f64();
                // bt 对账
                let (pf1, pf5) = btopt::bt_gap1_gap5_prod(
                    full.view(),
                    shared.ret_gap1.view(),
                    shared.ret_sum_gap1.view(),
                    shared.ret_gap5.view(),
                    shared.ret_sum_gap5.view(),
                    shared.restrict.view(),
                    shared.index_ret.view(),
                    &shared.dates,
                    shared.backtest_start,
                    10,
                    &open_counts,
                    false,
                    &pre,
                );
                let (tf1, tf5) = btopt::bt_gap1_gap5_prod(
                    trunc_out.view(),
                    shared.ret_gap1.view(),
                    shared.ret_sum_gap1.view(),
                    shared.ret_gap5.view(),
                    shared.ret_sum_gap5.view(),
                    shared.restrict.view(),
                    shared.index_ret.view(),
                    &shared.dates,
                    shared.backtest_start,
                    10,
                    &open_counts,
                    false,
                    &pre,
                );
                let eq = btopt_results_eq(&pf1, &tf1) && btopt_results_eq(&pf5, &tf5);
                println!(
                    "TRUNC {name}::{tag}: full={t_full:.3}s trunc={t_trunc:.3}s speedup={:.2}x bt_bitwise_eq={eq}",
                    t_full / t_trunc
                );
            }
        }
        return;
    }

    if mode == "btfuse" {
        // 现生产 O1 回测路径: 逐面 per-slot vs 日循环外置多面批量版 (BT-P3)。
        let shared = load_shared(true);
        let t = Instant::now();
        let pre = btopt::build_bt_precomputed(&shared.ret_sum_gap1, &shared.ret_sum_gap5);
        println!("bt_precomputed: {:.2}s", t.elapsed().as_secs_f64());
        let open_counts = engine_precompute_open(&shared.restrict.view());
        let factors = load_factors();
        for (name, raw) in &factors {
            let ranked = engine::rank_and_fill_missing_cross_sectional_median(raw);
            let mut slots: Vec<Array2<f32>> = vec![ranked.clone()];
            for &w in &[5usize, 10, 20] {
                let (m, x, n2, s) = engine::rolling_stats_f32_serial(&ranked, w, w / 2);
                slots.push(m); slots.push(x); slots.push(n2); slots.push(s);
            }
            let t = Instant::now();
            let mut base: Vec<(engine::LegacyBacktestResult, engine::LegacyBacktestResult)> =
                Vec::new();
            for slot in &slots {
                base.push(btopt::bt_gap1_gap5_prod(
                    slot.view(),
                    shared.ret_gap1.view(), shared.ret_sum_gap1.view(),
                    shared.ret_gap5.view(), shared.ret_sum_gap5.view(),
                    shared.restrict.view(), shared.index_ret.view(),
                    &shared.dates, shared.backtest_start, 10, &open_counts, false, &pre,
                ));
            }
            let t_perface = t.elapsed().as_secs_f64();
            for &bb in &[4usize, 13] {
                let t = Instant::now();
                let mut outs: Vec<(engine::LegacyBacktestResult, engine::LegacyBacktestResult)> =
                    Vec::new();
                for chunk in slots.chunks(bb) {
                    let views: Vec<_> = chunk.iter().map(|s| s.view()).collect();
                    outs.extend(btopt::bt_gap1_gap5_batch(
                        &views,
                        shared.ret_gap1.view(), shared.ret_sum_gap1.view(),
                        shared.ret_gap5.view(), shared.ret_sum_gap5.view(),
                        shared.restrict.view(), shared.index_ret.view(),
                        &shared.dates, shared.backtest_start, 10, &open_counts, false, &pre,
                    ));
                }
                let t_batch = t.elapsed().as_secs_f64();
                let eq = base.iter().zip(outs.iter()).all(|(a, b)| {
                    btopt_results_eq(&a.0, &b.0) && btopt_results_eq(&a.1, &b.1)
                });
                println!(
                    "BTBATCH {name}: perface={t_perface:.3}s B={bb} batch={t_batch:.3}s speedup={:.2}x bitwise_eq={eq}",
                    t_perface / t_batch
                );
            }
        }
        return;
    }

    if mode == "neucheck" {
        bench_neucheck();
        return;
    }

    if mode == "v2" {
        v2::run_v2_bench(DATA);
        return;
    }

    if mode == "v2b" {
        // 用法: tail_perf_bench v2b
        v2::run_v2_batch_bench(DATA_URGENCY);
        return;
    }
    if mode == "v2bmt" {
        // 用法: tail_perf_bench v2bmt <workers逗号列表> <rounds> <batch>
        let workers: Vec<usize> = args[2].split(',').map(|a| a.parse::<usize>().unwrap()).collect();
        let rounds: usize = args.get(3).map(|a| a.parse().unwrap()).unwrap_or(3);
        let batch: usize = args.get(4).map(|a| a.parse().unwrap()).unwrap_or(4);
        v2::run_v2_batch_mt(DATA_URGENCY, &workers, rounds, batch);
        return;
    }
    if mode == "v2mt" {
        // 用法: tail_perf_bench v2mt [workers,...] [rounds]
        let workers: Vec<usize> = args
            .get(2)
            .map(|s| s.split(',').filter_map(|x| x.parse().ok()).collect())
            .unwrap_or_else(|| vec![1, 32, 96, 200]);
        let rounds: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(3);
        v2::run_v2_mt(DATA, &workers, rounds);
        return;
    }

    // replica / fused / prerank
    let shared = load_shared(fold);
    let factors = load_factors();

    for (name, raw) in &factors {
        match mode {
            "replica" => {
                let t = Instant::now();
                let stats = engine::run_factor_v7(name, raw.clone(), &shared, neu);
                println!(
                    "FACTOR {name} mode=replica fold={fold} neu={neu} total={:.3}s slots={} bt_calls={} | raw_cover={:.3}s rank_fill={:.3}s rolling={:.3}s preflight={:.3}s bt={:.3}s fold_step={:.3}s cov_before={:.4} cov_after={:.4}",
                    t.elapsed().as_secs_f64(),
                    stats.times.slots,
                    stats.times.bt_calls,
                    stats.times.raw_cover,
                    stats.times.rank_fill,
                    stats.times.rolling,
                    stats.times.preflight,
                    stats.times.bt_raw,
                    stats.times.fold,
                    stats.raw_cover_before_fill,
                    stats.raw_cover_after_fill,
                );
                print_summaries_json(name, &stats.raw_summaries);
            }
            "fused" => {
                let mut summaries = Vec::new();
                let mut time_bt = 0.0;
                let t0 = Instant::now();
                let ranked_raw = engine::rank_and_fill_missing_cross_sectional_median(raw);
                let open_counts = engine_precompute_open(&shared.restrict.view());
                run_variant_fused(
                    name,
                    ranked_raw,
                    &shared,
                    &open_counts,
                    &mut summaries,
                    &mut time_bt,
                );
                if fold {
                    let folded = engine::build_fold_values(raw);
                    let ranked_fold =
                        engine::rank_and_fill_missing_cross_sectional_median(&folded);
                    run_variant_fused(
                        &format!("{name}_fold"),
                        ranked_fold,
                        &shared,
                        &open_counts,
                        &mut summaries,
                        &mut time_bt,
                    );
                }
                println!(
                    "FACTOR {name} mode=fused fold={fold} total={:.3}s bt={:.3}s",
                    t0.elapsed().as_secs_f64(),
                    time_bt
                );
                print_summaries_json(name, &summaries);
            }
            "prerank" => {
                let mut summaries = Vec::new();
                let mut time_bt = 0.0;
                let t_pre = Instant::now();
                let pr1 = opt::precompute_ret_ranks(
                    &shared.ret_sum_gap1.view(),
                    &shared.restrict.view(),
                );
                let pr5 = opt::precompute_ret_ranks(
                    &shared.ret_sum_gap5.view(),
                    &shared.restrict.view(),
                );
                let pre_s = t_pre.elapsed().as_secs_f64();
                let t0 = Instant::now();
                let ranked_raw = engine::rank_and_fill_missing_cross_sectional_median(raw);
                let open_counts = engine_precompute_open(&shared.restrict.view());
                run_variant_prerank(
                    name,
                    ranked_raw,
                    &shared,
                    &open_counts,
                    &pr1,
                    &pr5,
                    &mut summaries,
                    &mut time_bt,
                );
                if fold {
                    let folded = engine::build_fold_values(raw);
                    let ranked_fold =
                        engine::rank_and_fill_missing_cross_sectional_median(&folded);
                    run_variant_prerank(
                        &format!("{name}_fold"),
                        ranked_fold,
                        &shared,
                        &open_counts,
                        &pr1,
                        &pr5,
                        &mut summaries,
                        &mut time_bt,
                    );
                }
                println!(
                    "FACTOR {name} mode=prerank fold={fold} precompute={pre_s:.3}s total={:.3}s bt={:.3}s",
                    t0.elapsed().as_secs_f64(),
                    time_bt
                );
                print_summaries_json(name, &summaries);
            }
            _ => unreachable!(),
        }
    }
}

/// 中性化 hat 矩阵预计算原型（合成数据）：
/// 生产 get_residual 对每个 (date, slot) 重建 X'X(p×p) 并 Cholesky 求解；
/// 优化：X 只随 date 变化（barra+行业哑变量），每 date 预计算一次
/// P = (X'X)^-1 X' (p×n)，之后每个 slot 只需两次 matvec。
fn bench_hat() {
    let t_dates = 2276usize;
    let n_stocks = 4300usize;
    let k = 10usize;
    let n_ind = 30usize;
    let p = k + n_ind;
    let slots = 26usize;

    let mut xs: Vec<DMatrix<f64>> = Vec::with_capacity(t_dates);
    let mut seed = 42u64;
    for _ in 0..t_dates {
        let data: Vec<f64> = (0..n_stocks * p)
            .map(|_| {
                seed = seed
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((seed >> 11) as f64) / ((1u64 << 53) as f64) - 0.5
            })
            .collect();
        xs.push(DMatrix::from_row_slice(n_stocks, p, &data));
    }
    let ys: Vec<DMatrix<f64>> = (0..slots)
        .map(|_| {
            let data: Vec<f64> = (0..t_dates * n_stocks)
                .map(|_| {
                    seed = seed
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    ((seed >> 11) as f64) / ((1u64 << 53) as f64) - 0.5
                })
                .collect();
            DMatrix::from_row_slice(t_dates, n_stocks, &data)
        })
        .collect();

    // (a) 生产语义：每 (date, slot) 重建 X'X + Cholesky
    let t = Instant::now();
    let mut resid_a = Vec::with_capacity(slots);
    for y in &ys {
        let mut resid = DMatrix::<f64>::zeros(t_dates, n_stocks);
        for d in 0..t_dates {
            let x = &xs[d];
            let xtx = x.transpose() * x;
            let xty = x.transpose() * &y.row(d).transpose();
            let chol = Cholesky::new(xtx).expect("chol");
            let beta = chol.solve(&xty);
            let yhat = x * &beta;
            for s in 0..n_stocks {
                resid[(d, s)] = y[(d, s)] - yhat[(s, 0)];
            }
        }
        resid_a.push(resid);
    }
    let t_a = t.elapsed().as_secs_f64();

    // (b) 优化：每 date 预计算一次 P = (X'X)^-1 X'
    let t = Instant::now();
    let mut ps: Vec<DMatrix<f64>> = Vec::with_capacity(t_dates);
    for x in &xs {
        let xtx = x.transpose() * x;
        let chol = Cholesky::new(xtx).expect("chol");
        let xt = x.transpose();
        let pinv = chol.solve(&xt);
        ps.push(pinv);
    }
    let t_pre = t.elapsed().as_secs_f64();

    let t = Instant::now();
    let mut resid_b = Vec::with_capacity(slots);
    for y in &ys {
        let mut resid = DMatrix::<f64>::zeros(t_dates, n_stocks);
        for d in 0..t_dates {
            let row = y.row(d);
            let beta = &ps[d] * &row.transpose();
            let yhat = &xs[d] * beta;
            for s in 0..n_stocks {
                resid[(d, s)] = row[s] - yhat[(s, 0)];
            }
        }
        resid_b.push(resid);
    }
    let t_apply = t.elapsed().as_secs_f64();

    let mut maxdiff = 0.0f64;
    for (a, b) in resid_a.iter().zip(resid_b.iter()) {
        for i in 0..a.nrows() * a.ncols() {
            maxdiff = maxdiff.max((a[i] - b[i]).abs());
        }
    }
    println!(
        "hat prototype: naive={t_a:.2}s ({:.2}s/slot) | precompute={t_pre:.2}s once + apply={t_apply:.2}s ({:.2}s/slot) | naive/apply={:.1}x | maxdiff={maxdiff:e}",
        t_a / slots as f64,
        t_apply / slots as f64,
        t_a / t_apply,
    );
}


fn bench_neu() {
    let barra_raw = npy::as_f64_3d(npy::load(&format!("{DATA}/barra_raw.npy")));
    let industry = npy::as_f64_mat(npy::load(&format!("{DATA}/industry.npy")));
    let restrict = npy::as_f32_mat(npy::load(&format!("{DATA}/restrict.npy")));
    let factors = load_factors();

    let t = Instant::now();
    let shared_prod = neu::neu_precompute(&industry, &restrict, &barra_raw);
    println!("neu_precompute (生产语义): {:.2}s", t.elapsed().as_secs_f64());

    let t = Instant::now();
    let shared_opt = neu::neu_precompute_opt(&industry, &restrict, &barra_raw);
    println!(
        "neu_precompute_opt (含每日 X/X'X/L/排序): {:.2}s",
        t.elapsed().as_secs_f64()
    );

    // 用真实 ranked+filled slot（与生产回测中进入中性化的 slot 形态一致）
    for (name, raw) in &factors {
        let ranked = engine::rank_and_fill_missing_cross_sectional_median(raw);
        let t = Instant::now();
        let (out_p, st) = neu::neutralize_slot_prod(ranked.view(), &shared_prod);
        let t_p = t.elapsed().as_secs_f64();
        let t = Instant::now();
        let (out_o, st_o) = neu::neutralize_slot_opt(ranked.view(), &shared_opt);
        let t_o = t.elapsed().as_secs_f64();
        let diff = (&out_p - &out_o).mapv(f64::from).fold(0.0f64, |m, x| m.max(x.abs()));
        println!(
            "NEU {name}: prod={t_p:.2}s [rank1={:.2} fill_ind={:.2} mask_clone={:.2} fills={:.2} restrict={:.2} rank2={:.2} residual={:.2} rank3={:.2} convert={:.2}]",
            st.rank1, st.fill_ind, st.mask_clone, st.fills, st.restrict, st.rank2, st.residual, st.rank3, st.convert
        );
        println!(
            "     opt ={t_o:.2}s [rank1={:.2} fill_ind={:.2} mask_clone={:.2} fills={:.2} restrict={:.2} rank2={:.2} residual={:.2} rank3={:.2} convert={:.2}] speedup={:.2}x maxdiff={:.3e}",
            st_o.rank1, st_o.fill_ind, st_o.mask_clone, st_o.fills, st_o.restrict, st_o.rank2, st_o.residual, st_o.rank3, st_o.convert,
            t_p / t_o,
            diff
        );
    }
}


/// 用公开 API 保存的 neu_out1.npy（生产中性化输出）与 sandbox 复刻逐位对账。
fn bench_neucheck() {
    let barra_raw = npy::as_f64_3d(npy::load(&format!("{DATA}/barra_raw.npy")));
    let industry = npy::as_f64_mat(npy::load(&format!("{DATA}/industry.npy")));
    let restrict = npy::as_f32_mat(npy::load(&format!("{DATA}/restrict.npy")));
    let shared = neu::neu_precompute(&industry, &restrict, &barra_raw);
    let factors = load_factors();
    let (_, raw) = &factors[0];
    let (out, st) = neu::neutralize_slot_prod(raw.view(), &shared);
    let prod_out = npy::as_f32_mat(npy::load(&format!("{DATA}/neu_out1.npy")));
    let diff = (&out - &prod_out).mapv(f64::from).fold(0.0f64, |m, x| m.max(x.abs()));
    let nan_mismatch = out
        .iter()
        .zip(prod_out.iter())
        .filter(|(a, b)| a.is_nan() != b.is_nan())
        .count();
    println!(
        "neucheck: sandbox={:.2}s maxdiff={:.3e} nan_mismatch={}",
        st.rank1 + st.fill_ind + st.mask_clone + st.fills + st.restrict + st.rank2 + st.residual + st.rank3 + st.convert,
        diff,
        nan_mismatch
    );
}
