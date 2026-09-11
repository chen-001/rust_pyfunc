//! drive8：端到端驱动 + 三档改造的「基线 vs v8」对照（Lead 独占）。
//!
//! 基线 `run_factor_v7_real`：与生产 v7 同形（物化 13 张 (T,N) 派生面 + 每面独立
//! preflight / 中性化 / 回测），但用**同一套**中性化(v3)与回测实现，保证与 v8 的差异
//! 只来自三档改造本身。
//!
//! 三档：结果聚合不再在主线程逐条串行处理——见 `aggregate_ic_*` 的两个版本。

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;

use ndarray::Array2;

use crate::engine::{self, LegacyBacktestResult};
use crate::pf8::{self, FreeMask};
use crate::t8::{self, T8Result};
use crate::v3::{self, V3Scratch, V3Shared};
use crate::v8::{self, V8Ctx, V8Scratch};

/// 基线（v7 形态）的共享上下文。
pub struct BaseCtx {
    pub shared: Arc<engine::Shared>,
    pub v3: Arc<V3Shared>,
    pub mask: Arc<FreeMask>,
    pub open_symbol_counts: Arc<Vec<usize>>,
    pub ic_only: bool,
    /// 生产 O1 回测路径所需的收益秩预排序（与生产 build_bt_precomputed 同源）
    pub bt_pre: Arc<crate::btopt::BtPrecomputed>,
}

impl BaseCtx {
    pub fn new(shared: Arc<engine::Shared>, v3: Arc<V3Shared>, ic_only: bool) -> Self {
        let mask = pf8::build_free_mask(&shared.restrict.view());
        let open_symbol_counts = Arc::new(
            (0..shared.restrict.nrows())
                .map(|r| {
                    shared
                        .restrict
                        .row(r)
                        .iter()
                        .filter(|&&v| v.is_finite() && v == 0.0)
                        .count()
                })
                .collect::<Vec<_>>(),
        );
        let bt_pre = Arc::new(crate::btopt::build_bt_precomputed(
            &shared.ret_sum_gap1,
            &shared.ret_sum_gap5,
        ));
        Self {
            shared,
            v3,
            mask: Arc::new(mask),
            open_symbol_counts,
            ic_only,
            bt_pre,
        }
    }
}

fn push_bt(
    res: &LegacyBacktestResult,
    name: &str,
    gap: i32,
    out: &mut Vec<(String, i32, [f64; 10])>,
) {
    out.push((name.to_string(), gap, res.summary));
}

/// 基线：v7 形态的整因子流水线（真实中性化）。
pub fn run_factor_v7_real(
    source_factor: &str,
    raw: Array2<f32>,
    ctx: &BaseCtx,
    v3sc: &mut V3Scratch,
) -> T8Result {
    let s = &ctx.shared;
    let mut result = T8Result {
        source_factor: source_factor.to_string(),
        ..Default::default()
    };

    let tp = Instant::now();
    let cover_before = engine::compute_raw_cover_rate(
        &raw.view(),
        &s.restrict.view(),
        &s.ret_gap1.view(),
        10,
    );
    result.raw_cover_before_fill = cover_before;
    let ranked_raw = engine::rank_and_fill_missing_cross_sectional_median(&raw, &s.restrict);
    let cover = engine::compute_raw_cover_rate(
        &ranked_raw.view(),
        &s.restrict.view(),
        &s.ret_gap1.view(),
        10,
    );
    result.raw_cover_after_fill = cover;
    t8::tick(&t8::T_PREP, tp);
    if cover < s.cover_rate {
        result.eliminated_by_raw_cover = true;
        return result;
    }

    variant_v7_real(source_factor, ranked_raw, ctx, v3sc, &mut result);
    if s.fold {
        let folded = engine::build_fold_values(&raw);
        let ranked_fold = engine::rank_and_fill_missing_cross_sectional_median(&folded, &s.restrict);
        variant_v7_real(
            &format!("{}_fold", source_factor),
            ranked_fold,
            ctx,
            v3sc,
            &mut result,
        );
    }
    result
}

fn variant_v7_real(
    variant_name: &str,
    ranked: Array2<f32>,
    ctx: &BaseCtx,
    v3sc: &mut V3Scratch,
    result: &mut T8Result,
) {
    let s = &ctx.shared;
    let names = t8::derived_names_for_variant(variant_name, &s.windows);

    let tr = Instant::now();
    let mut slots: Vec<Array2<f32>> = vec![ranked.clone()];
    for &w in &s.windows {
        let (m, x, n2, sd) = engine::rolling_stats_f32_serial(&ranked, w, std::cmp::max(1, w / 2));
        slots.push(m);
        slots.push(x);
        slots.push(n2);
        slots.push(sd);
    }
    t8::tick(&t8::T_ROLL, tr);

    for (si, slot) in slots.iter().enumerate() {
        let name = &names[si];
        let tpf = Instant::now();
        let pre = crate::preflightopt::preflight_prod(
            &slot.view(),
            &s.restrict.view(),
            s.majority_count_threshold,
            s.zero_max_threshold,
            s.nan_max_threshold,
        );
        t8::tick(&t8::T_PF, tpf);
        if !pre.passed {
            if pre.majority_count_mean > s.majority_count_threshold {
                result.preflight_maj_failed += 1;
            }
            if pre.zero_ratio_mean >= s.zero_max_threshold {
                result.preflight_zero_failed += 1;
            }
            if pre.nan_ratio_mean >= s.nan_max_threshold {
                result.preflight_nan_failed += 1;
            }
            continue;
        }
        result.any_window_passed_preflight = true;

        if !ctx.ic_only {
            let tb = Instant::now();
            let (r1, r5) = crate::btopt::bt_gap1_gap5_prod(
                slot.view(),
                s.ret_gap1.view(),
                s.ret_sum_gap1.view(),
                s.ret_gap5.view(),
                s.ret_sum_gap5.view(),
                s.restrict.view(),
                s.index_ret.view(),
                &s.dates,
                s.backtest_start,
                10,
                &ctx.open_symbol_counts,
                ctx.ic_only,
                &ctx.bt_pre,
            );
            push_bt(&r1, name, 1, &mut result.raw_summaries);
            push_bt(&r5, name, 5, &mut result.raw_summaries);
            t8::tick(&t8::T_BT, tb);
        }

        let tn = Instant::now();
        let (neu, _t) = v3::v3_slot_range(slot.view(), &ctx.v3, 0, slot.nrows(), v3sc);
        t8::tick(&t8::T_NEU, tn);
        let tb = Instant::now();
        let (n1, n5) = crate::btopt::bt_gap1_gap5_prod(
            neu.view(),
            s.ret_gap1.view(),
            s.ret_sum_gap1.view(),
            s.ret_gap5.view(),
            s.ret_sum_gap5.view(),
            s.restrict.view(),
            s.index_ret.view(),
            &s.dates,
            s.backtest_start,
            10,
            &ctx.open_symbol_counts,
            ctx.ic_only,
            &ctx.bt_pre,
        );
        push_bt(&n1, name, 1, &mut result.neu_summaries);
        push_bt(&n5, name, 5, &mut result.neu_summaries);
        t8::tick(&t8::T_BT, tb);
    }
}

// ---------------- 多线程驱动 ----------------

pub enum Mode {
    Base,
    V8,
}

/// 因子来源：内存夹具（小规模对账）或真实 colblk 库（全量）。
pub enum TaskSource {
    Mem(Arc<Vec<Array2<f32>>>),
    Store(Arc<crate::store8::Store8>),
}

impl TaskSource {
    pub fn get(&self, k: usize) -> Result<Array2<f32>, String> {
        match self {
            TaskSource::Mem(v) => Ok(v[k].clone()),
            TaskSource::Store(s) => s.read_factor(k),
        }
    }
}

/// 多线程跑 N 个任务，返回 (墙钟秒, 每个任务的结果)。
/// 三档：worker 自己完成结果整理，主线程只做收集（不做逐条串行处理）。
pub fn run_mt(
    mode: Mode,
    tasks: &[(String, usize)],
    src: &TaskSource,
    base: &Arc<BaseCtx>,
    v8ctx: &Arc<V8Ctx>,
    n_threads: usize,
) -> (f64, Vec<T8Result>) {
    let next = AtomicUsize::new(0);
    let n = tasks.len();
    let mut out: Vec<Option<T8Result>> = (0..n).map(|_| None).collect();
    let t0 = Instant::now();
    let slots: Vec<std::sync::Mutex<Option<T8Result>>> =
        (0..n).map(|_| std::sync::Mutex::new(None)).collect();
    std::thread::scope(|sc| {
        for _ in 0..n_threads {
            sc.spawn(|| {
                let mut v3sc = V3Scratch::new(base.shared.restrict.ncols());
                let mut v8sc = V8Scratch::new(
                    base.shared.restrict.ncols(),
                    &base.shared.windows,
                    v8ctx.block_rows,
                );
                loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    if i >= n {
                        break;
                    }
                    let (name, k) = &tasks[i];
                    let tr = Instant::now();
                    let raw = match src.get(*k) {
                        Ok(m) => m,
                        Err(e) => {
                            eprintln!("[worker] 读因子 {name} 失败: {e}");
                            continue;
                        }
                    };
                    t8::tick(&t8::T_READ, tr);
                    let tw = Instant::now();
                    let res = match mode {
                        Mode::Base => run_factor_v7_real(name, raw, base, &mut v3sc),
                        Mode::V8 => v8::run_factor_v8(name, raw, v8ctx, &mut v8sc),
                    };
                    t8::tick(&t8::T_WORK, tw);
                    *slots[i].lock().unwrap() = Some(res);
                }
            });
        }
    });
    let wall = t0.elapsed().as_secs_f64();
    for (i, s) in slots.into_iter().enumerate() {
        out[i] = s.into_inner().unwrap();
    }
    (wall, out.into_iter().map(|x| x.unwrap()).collect())
}

/// 三档·收尾聚合的基线版：逐元素 HashMap 查日期（生产 `write_ic_outputs` 的做法）。
pub fn aggregate_ic_hashmap(series: &[(String, Vec<i32>, Vec<f32>)]) -> (usize, usize) {
    use std::collections::{BTreeSet, HashMap};
    let mut all_dates = BTreeSet::<i32>::new();
    for (_, d, _) in series {
        for &x in d {
            all_dates.insert(x);
        }
    }
    let all_dates: Vec<i32> = all_dates.into_iter().collect();
    let pos: HashMap<i32, usize> = all_dates.iter().enumerate().map(|(i, &d)| (d, i)).collect();
    let mut data = vec![f32::NAN; all_dates.len() * series.len()];
    for (c, (_, d, v)) in series.iter().enumerate() {
        for (&dt, &val) in d.iter().zip(v.iter()) {
            if let Some(&r) = pos.get(&dt) {
                data[r * series.len() + c] = val;
            }
        }
    }
    (data.len(), data.iter().filter(|x| x.is_finite()).count())
}

/// 三档·收尾聚合的改造版：dates 已升序，用双指针归并（零哈希查找）。
pub fn aggregate_ic_merge(series: &[(String, Vec<i32>, Vec<f32>)]) -> (usize, usize) {
    use std::collections::BTreeSet;
    let mut all_dates = BTreeSet::<i32>::new();
    for (_, d, _) in series {
        for &x in d {
            all_dates.insert(x);
        }
    }
    let all_dates: Vec<i32> = all_dates.into_iter().collect();
    let mut data = vec![f32::NAN; all_dates.len() * series.len()];
    let ncols = series.len();
    for (c, (_, d, v)) in series.iter().enumerate() {
        let mut p = 0usize;
        for (k, &dt) in d.iter().enumerate() {
            while p < all_dates.len() && all_dates[p] < dt {
                p += 1;
            }
            if p < all_dates.len() && all_dates[p] == dt {
                data[p * ncols + c] = v[k];
            }
        }
    }
    (data.len(), data.iter().filter(|x| x.is_finite()).count())
}

/// 结果对账：v8 与基线逐位比较 summary 与 IC 序列。
pub fn compare_results(base: &T8Result, v8r: &T8Result) -> Vec<String> {
    let mut msgs = Vec::new();
    if base.eliminated_by_raw_cover != v8r.eliminated_by_raw_cover {
        msgs.push(format!("raw_cover 淘汰标记不一致"));
    }
    if base.any_window_passed_preflight != v8r.any_window_passed_preflight {
        msgs.push("any_window_passed_preflight 不一致".into());
    }
    if base.raw_summaries.len() != v8r.raw_summaries.len() {
        msgs.push(format!(
            "raw summary 条数 {} vs {}",
            base.raw_summaries.len(),
            v8r.raw_summaries.len()
        ));
    }
    if base.neu_summaries.len() != v8r.neu_summaries.len() {
        msgs.push(format!(
            "neu summary 条数 {} vs {}",
            base.neu_summaries.len(),
            v8r.neu_summaries.len()
        ));
    }
    let cmp = |a: &[(String, i32, [f64; 10])], b: &[(String, i32, [f64; 10])], tag: &str| {
        let mut out = Vec::new();
        let mut ka: Vec<_> = a.iter().collect();
        let mut kb: Vec<_> = b.iter().collect();
        ka.sort_by(|x, y| (x.0.clone(), x.1).cmp(&(y.0.clone(), y.1)));
        kb.sort_by(|x, y| (x.0.clone(), x.1).cmp(&(y.0.clone(), y.1)));
        for (x, y) in ka.iter().zip(kb.iter()) {
            if x.0 != y.0 || x.1 != y.1 {
                out.push(format!("{tag} 键不一致 {}:{} vs {}:{}", x.0, x.1, y.0, y.1));
                continue;
            }
            for i in 0..10 {
                let (p, q) = (x.2[i], y.2[i]);
                let same = (p.is_nan() && q.is_nan()) || p.to_bits() == q.to_bits();
                if !same {
                    out.push(format!("{tag} {} gap{} 第{}项 {p} vs {q}", x.0, x.1, i));
                }
            }
        }
        out
    };
    msgs.extend(cmp(&base.raw_summaries, &v8r.raw_summaries, "raw"));
    msgs.extend(cmp(&base.neu_summaries, &v8r.neu_summaries, "neu"));
    msgs
}

// ---------------- 端到端跑分 ----------------

/// 从夹具目录加载 engine::Shared（与 main.rs::load_shared 同构，但目录可传）。
pub fn load_shared_dir(data_dir: &str, fold: bool) -> engine::Shared {
    let dates = crate::npy::as_i32_vec1(crate::npy::load(&format!("{data_dir}/dates.npy")));
    // 与生产脚本对齐：majority_count_threshold 默认 2000（脚本值），可用 E2E_MAJ 覆盖
    let maj = std::env::var("E2E_MAJ").ok().and_then(|s| s.parse().ok()).unwrap_or(200.0);
    let bstart = std::env::var("E2E_BSTART").ok().and_then(|s| s.parse().ok()).unwrap_or(20170201);
    engine::Shared {
        cover_rate: 0.5,
        dates,
        windows: vec![5, 10, 20],
        fold,
        backtest_start: bstart,
        ret_gap1: crate::npy::as_f32_mat(crate::npy::load(&format!("{data_dir}/ret_gap1.npy"))),
        ret_sum_gap1: crate::npy::as_f32_mat(crate::npy::load(&format!(
            "{data_dir}/ret_sum_gap1.npy"
        ))),
        ret_gap5: crate::npy::as_f32_mat(crate::npy::load(&format!("{data_dir}/ret_gap5.npy"))),
        ret_sum_gap5: crate::npy::as_f32_mat(crate::npy::load(&format!(
            "{data_dir}/ret_sum_gap5.npy"
        ))),
        restrict: crate::npy::as_f32_mat(crate::npy::load(&format!("{data_dir}/restrict.npy"))),
        index_ret: crate::npy::as_f32_vec1(crate::npy::load(&format!("{data_dir}/index_ret.npy")))
            .into(),
        majority_count_threshold: maj,
        zero_max_threshold: 0.1,
        nan_max_threshold: 0.04,
    }
}

/// 端到端：同一批任务分别跑基线（v7 形态 + 真实中性化）与 v8（三档融合），
/// 逐位对账 summary，并打印墙钟对比。
pub fn run_e2e(data_dir: &str, n_threads: usize, n_tasks: usize, block_rows: usize) {
    let t0 = Instant::now();
    let shared = load_shared_dir(data_dir, true);
    let v2s = crate::v2::build_shared(data_dir);
    let v3s = crate::v3::v3_build(v2s);
    println!(
        "[e2e] 中性化预计算: {:.1}s（这段两种模式共用，不计入对比）",
        t0.elapsed().as_secs_f64()
    );
    let shared = Arc::new(shared);
    let v3s = Arc::new(v3s);

    let names_txt = std::fs::read_to_string(format!("{data_dir}/sample_names.txt")).unwrap();
    let base_names: Vec<String> = names_txt
        .lines()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();
    // ---- 任务表：store 模式（全量）或内存夹具模式 ----
    let use_store = std::env::var("E2E_STORE").is_ok();
    let mut fixtures = Vec::new();
    if !use_store {
        for nm in &base_names {
            fixtures.push(crate::npy::as_f32_mat(crate::npy::load(&format!(
                "{data_dir}/factor_{nm}.npy"
            ))));
        }
    }
    let (src, tasks): (TaskSource, Vec<(String, usize)>) = if use_store {
        let meta = std::env::var("STORE_META").unwrap_or("/home/chenzongwei/neu_lab/store_meta".into());
        let sdir = std::env::var("STORE_DIR")
            .unwrap_or("/hdd/user_home_unsafe/chenzongwei/factor_store_yupei_dist".into());
        let st = Arc::new(crate::store8::Store8::open(&meta, &sdir).expect("打开 store8"));
        let nf = st.n_factors();
        let take = if n_tasks == 0 { nf } else { n_tasks.min(nf) };
        let t: Vec<(String, usize)> =
            (0..take).map(|c| (st.factor_names[c].clone(), c)).collect();
        println!("[e2e] 来源=真实 colblk 库（{sdir}），库内 {nf} 个因子，本次跑 {take} 个");
        (TaskSource::Store(st), t)
    } else {
        let n_fx = fixtures.len();
        let fixtures = Arc::new(fixtures);
        let t: Vec<(String, usize)> = (0..n_tasks)
            .map(|i| {
                let k = i % n_fx;
                (format!("{}#{}", base_names[k], i), k)
            })
            .collect();
        (TaskSource::Mem(fixtures), t)
    };
    println!(
        "[e2e] 任务数={} 线程数={} 块行数={}",
        tasks.len(),
        n_threads,
        block_rows
    );

    let ic_only = std::env::var("E2E_IC_ONLY").is_ok();
    println!("[e2e] ic_only={ic_only}");
    let base = Arc::new(BaseCtx::new(shared.clone(), v3s.clone(), ic_only));
    let mut v8ctx_inner = v8::build_v8_ctx(shared.clone(), v3s.clone(), ic_only, false);
    v8ctx_inner.block_rows = block_rows;
    let v8ctx = Arc::new(v8ctx_inner);

    // 每种模式跑两轮：第一轮冷（缺页/缓存未热），第二轮热。对比用热轮。
    if std::env::var("E2E_SKIP_COLD").is_err() {
        let (cold_base, _) = run_mt(Mode::Base, &tasks, &src, &base, &v8ctx, n_threads);
        let (cold_v8, _) = run_mt(Mode::V8, &tasks, &src, &base, &v8ctx, n_threads);
        println!("[e2e] 冷启动轮: 基线 {:.1}s / v8 {:.1}s", cold_base, cold_v8);
    }

    t8::reset_timers();
    let (wall_base, res_base) = run_mt(Mode::Base, &tasks, &src, &base, &v8ctx, n_threads);
    t8::dump_timers("基线", tasks.len(), wall_base);
    println!(
        "[e2e] 基线(v7形态) 墙钟 {:.1}s  每因子 {:.2}s  吞吐 {:.3} 因子/s",
        wall_base,
        wall_base / tasks.len() as f64,
        tasks.len() as f64 / wall_base
    );

    t8::reset_timers();
    let (wall_v8, res_v8) = run_mt(Mode::V8, &tasks, &src, &base, &v8ctx, n_threads);
    t8::dump_timers("v8", tasks.len(), wall_v8);
    println!(
        "[e2e] v8(三档融合) 墙钟 {:.1}s  每因子 {:.2}s  吞吐 {:.3} 因子/s",
        wall_v8,
        wall_v8 / tasks.len() as f64,
        tasks.len() as f64 / wall_v8
    );
    println!(
        "[e2e] 加速比 = {:.2}×   端到端节省 {:.1}s ({:.1}%)",
        wall_base / wall_v8,
        wall_base - wall_v8,
        (wall_base - wall_v8) / wall_base * 100.0
    );

    // 对账
    let mut bad = 0usize;
    let mut msgs: Vec<String> = Vec::new();
    for (a, b) in res_base.iter().zip(res_v8.iter()) {
        let m = compare_results(a, b);
        if !m.is_empty() {
            bad += 1;
            if msgs.len() < 12 {
                msgs.extend(m);
            }
        }
    }
    if bad == 0 {
        println!("[e2e] 对账 PASS：{} 个因子的 raw/neu summary 全部逐位一致", res_base.len());
    } else {
        println!("[e2e] 对账 FAIL：{} / {} 个因子不一致", bad, res_base.len());
        for m in msgs.iter().take(12) {
            println!("       {m}");
        }
    }
}

/// 三档·收尾聚合的对照跑分（生产 `write_ic_outputs` 的 HashMap 版 vs 归并版）。
pub fn bench_aggregate(n_series: usize, n_dates: usize) {
    let mut series: Vec<(String, Vec<i32>, Vec<f32>)> = Vec::with_capacity(n_series);
    let mut seed: u64 = 0x9E3779B97F4A7C15;
    let mut rnd = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };
    for k in 0..n_series {
        let dates: Vec<i32> = (0..n_dates).map(|i| 20150105 + i as i32).collect();
        let vals: Vec<f32> = (0..n_dates).map(|_| (rnd() % 1000) as f32 / 1000.0).collect();
        series.push((format!("f{k}"), dates, vals));
    }
    let t = Instant::now();
    let (cells_a, fin_a) = aggregate_ic_hashmap(&series);
    let ta = t.elapsed().as_secs_f64();
    let t = Instant::now();
    let (_cells_b, fin_b) = aggregate_ic_merge(&series);
    let tb = t.elapsed().as_secs_f64();
    println!(
        "[agg] 序列={} 日期={} 矩阵={} 格({:.1} MB) | HashMap {:.2}s | 归并 {:.2}s | 加速 {:.2}× | 有效格 {} vs {}",
        n_series,
        n_dates,
        cells_a / n_dates,
        cells_a as f64 * 4.0 / 1e6,
        ta,
        tb,
        ta / tb,
        fin_a,
        fin_b
    );
}
