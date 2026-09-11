//! bt8_check：`bt8::BtAcc`（按日期块增量回测）与生产回测的**逐位对账** + 单线程计时。
//!
//! 对账口径（任务书）：
//!   summary 10 项 + ic_dates 全序列 + ic_values 全序列；
//!   f64/f32 一律 `to_bits` 比较，NaN 与 NaN 视为相同。
//! 双参照（两者本就逐位相同，同时打更有说服力）：
//!   * `engine::legacy_backtest_gap1_gap5_single_slot` —— 非 opt（逐日比较排序）
//!   * `btopt::bt_gap1_gap5_prod` —— 生产 O1 路径（收益秩预排序 `build_bt_precomputed`）
//!
//! 覆盖：3 个因子 × 13 个派生面（raw 侧）+ 每因子 2 个中性化面（v3::v3_slot），
//! 块大小 64 / 997 / 2818 各喂一遍；另含 ic_only 路径抽查与合成边界用例。
//!
//! 运行：
//!   V3_DATA=/home/chenzongwei/neu_lab/data_yupei ./target/release/bt8_check
//! 环境变量：
//!   BT8_FACTORS=3   参与对账的因子数（默认 3，最多 4）
//!   BT8_BENCH=1     是否跑「移植前(orig) / 移植后(bt8) / 参考引擎」单线程计时（默认 1）

use std::sync::Arc;
use std::time::Instant;

use ndarray::{s, Array2};

use tail_perf_bench::bt8::{BtAcc, BtCtx};
use tail_perf_bench::btopt::{self, BtPrecomputed};
use tail_perf_bench::engine::{self, LegacyBacktestResult};
use tail_perf_bench::t8;
use tail_perf_bench::{drive8, npy, v2, v3};

const PORTF_NUM: usize = 10;
const BLOCK_SIZES: [usize; 3] = [64, 997, 2818];
/// 每个因子额外对账的中性化面下标（0 = smooth_1，4 = mean_smooth_5）
const NEU_SLOTS: [usize; 2] = [0, 4];

fn same_f64(a: f64, b: f64) -> bool {
    (a.is_nan() && b.is_nan()) || a.to_bits() == b.to_bits()
}
fn same_f32(a: f32, b: f32) -> bool {
    (a.is_nan() && b.is_nan()) || a.to_bits() == b.to_bits()
}

/// 逐位比较一个结果对；不一致时往 fails 里追加明细，返回是否一致。
fn compare(
    tag: &str,
    got: &LegacyBacktestResult,
    want: &LegacyBacktestResult,
    fails: &mut Vec<String>,
) -> bool {
    let start = fails.len();
    for i in 0..10 {
        if !same_f64(got.summary[i], want.summary[i]) {
            fails.push(format!(
                "{tag} summary[{i}] {} vs {} (bits {:#018x} vs {:#018x})",
                got.summary[i],
                want.summary[i],
                got.summary[i].to_bits(),
                want.summary[i].to_bits()
            ));
        }
    }
    if got.ic_dates.len() != want.ic_dates.len() {
        fails.push(format!(
            "{tag} ic_dates 长度 {} vs {}",
            got.ic_dates.len(),
            want.ic_dates.len()
        ));
    } else if let Some(k) = got
        .ic_dates
        .iter()
        .zip(want.ic_dates.iter())
        .position(|(a, b)| a != b)
    {
        fails.push(format!(
            "{tag} ic_dates[{k}] {} vs {}",
            got.ic_dates[k], want.ic_dates[k]
        ));
    }
    if got.ic_values.len() != want.ic_values.len() {
        fails.push(format!(
            "{tag} ic_values 长度 {} vs {}",
            got.ic_values.len(),
            want.ic_values.len()
        ));
    } else if let Some(k) = got
        .ic_values
        .iter()
        .zip(want.ic_values.iter())
        .position(|(a, b)| !same_f32(*a, *b))
    {
        fails.push(format!(
            "{tag} ic_values[{k}] {} vs {} (bits {:#010x} vs {:#010x})",
            got.ic_values[k],
            want.ic_values[k],
            got.ic_values[k].to_bits(),
            want.ic_values[k].to_bits()
        ));
    }
    fails.len() == start
}

fn reference(
    shared: &engine::Shared,
    counts: &[usize],
    slot: &Array2<f32>,
    ic_only: bool,
) -> (LegacyBacktestResult, LegacyBacktestResult) {
    engine::legacy_backtest_gap1_gap5_single_slot(
        slot.view(),
        shared.ret_gap1.view(),
        shared.ret_sum_gap1.view(),
        shared.ret_gap5.view(),
        shared.ret_sum_gap5.view(),
        shared.restrict.view(),
        shared.index_ret.view(),
        &shared.dates,
        shared.backtest_start,
        PORTF_NUM,
        counts,
        ic_only,
    )
}

/// 参照 2：生产 O1 路径（收益秩预排序）。
fn reference_opt(
    shared: &engine::Shared,
    counts: &[usize],
    slot: &Array2<f32>,
    ic_only: bool,
    pre: &BtPrecomputed,
) -> (LegacyBacktestResult, LegacyBacktestResult) {
    btopt::bt_gap1_gap5_prod(
        slot.view(),
        shared.ret_gap1.view(),
        shared.ret_sum_gap1.view(),
        shared.ret_gap5.view(),
        shared.ret_sum_gap5.view(),
        shared.restrict.view(),
        shared.index_ret.view(),
        &shared.dates,
        shared.backtest_start,
        PORTF_NUM,
        counts,
        ic_only,
        pre,
    )
}

/// 按块喂 bt8（当前实现）。
fn run_bt8(
    slot: &Array2<f32>,
    gap: usize,
    block: usize,
    shared: &Arc<engine::Shared>,
    counts: &Arc<Vec<usize>>,
    ic_only: bool,
    pre: &Arc<BtPrecomputed>,
) -> LegacyBacktestResult {
    let mut acc = BtAcc::new(BtCtx {
        gap,
        shared: shared.clone(),
        open_symbol_counts: counts.clone(),
        ic_only,
        pre: pre.clone(),
    });
    let t = slot.nrows();
    let mut t0 = 0usize;
    while t0 < t {
        let t1 = (t0 + block).min(t);
        acc.push_block(&slot.slice(s![t0..t1, ..]), t0);
        t0 = t1;
    }
    acc.finish()
}

/// 按块喂 orig（改造前的冻结副本），仅用于计时对照。
fn run_orig(
    slot: &Array2<f32>,
    gap: usize,
    block: usize,
    shared: &Arc<engine::Shared>,
    counts: &Arc<Vec<usize>>,
    ic_only: bool,
) -> LegacyBacktestResult {
    let mut acc = orig::BtAccOrig::new(orig::BtCtxOrig {
        gap,
        shared: shared.clone(),
        open_symbol_counts: counts.clone(),
        ic_only,
    });
    let t = slot.nrows();
    let mut t0 = 0usize;
    while t0 < t {
        let t1 = (t0 + block).min(t);
        acc.push_block(&slot.slice(s![t0..t1, ..]), t0);
        t0 = t1;
    }
    acc.finish()
}

/// 合成数据边界用例：真实因子跑不到的路径。
///
/// 覆盖：`dates[t] <= backtest_start` 的行、signal 行全 NaN（不进 effective）、
/// 不同值不足 10（整面默认结果）、全 NaN 面（无有效日 → 默认结果）、
/// `stocks_num < portf_num`（ratio 保持 NaN / group 保持 0）、ret_sum 含 NaN（IC 排序的 NaN 分支）、
/// 块大小 1/3/8（与 T 不整除的块边界）。
fn synthetic_check(fails: &mut Vec<String>, n_cmp: &mut usize, n_fail: &mut usize) {
    use ndarray::Array1;

    let t = 8usize;
    let n = 24usize;
    let backtest_start = 101;
    let dates: Vec<i32> = vec![90, 101, 102, 103, 104, 105, 106, 107];

    let mut ret_gap1 = Array2::<f32>::from_elem((t, n), f32::NAN);
    let mut ret_gap5 = Array2::<f32>::from_elem((t, n), f32::NAN);
    let mut ret_sum_gap1 = Array2::<f32>::from_elem((t, n), f32::NAN);
    let mut ret_sum_gap5 = Array2::<f32>::from_elem((t, n), f32::NAN);
    let mut restrict = Array2::<f32>::from_elem((t, n), f32::NAN);
    let mut index_ret = vec![0.0f32; t];
    for r in 0..t {
        index_ret[r] = 0.001 * (r as f32) - 0.002;
        for s in 0..n {
            if (s * 3 + r) % 11 != 0 {
                ret_gap1[[r, s]] = s as f32 * 0.01 - 0.1 + r as f32 * 0.001;
                ret_gap5[[r, s]] = s as f32 * 0.02 - 0.2 + r as f32 * 0.002;
            }
            // ret_sum 留一部分 NaN：IC 的 filtered_future 里会出现 NaN（排序 NaN 分支）
            if (s + r) % 5 != 0 {
                ret_sum_gap1[[r, s]] = 0.5 * ret_gap1[[r, s]] + 0.01 * s as f32;
                ret_sum_gap5[[r, s]] = 0.5 * ret_gap5[[r, s]] + 0.01 * s as f32;
            }
            // 第 5 行只留 3 只可交易 → 该 held 行触发 stocks_num < portf_num
            restrict[[r, s]] = if r == 5 {
                if s < 3 { 0.0 } else { 1.0 }
            } else if (s + r) % 7 == 0 {
                f32::NAN
            } else if (s + r) % 3 == 0 {
                1.0
            } else {
                0.0
            };
        }
    }

    let shared = Arc::new(engine::Shared {
        cover_rate: 0.5,
        dates,
        windows: vec![5],
        fold: false,
        backtest_start,
        ret_gap1,
        ret_sum_gap1,
        ret_gap5,
        ret_sum_gap5,
        restrict,
        index_ret: Array1::from_vec(index_ret),
        majority_count_threshold: 200.0,
        zero_max_threshold: 0.1,
        nan_max_threshold: 0.04,
    });
    let counts: Arc<Vec<usize>> = Arc::new(
        (0..t)
            .map(|r| {
                shared
                    .restrict
                    .row(r)
                    .iter()
                    .filter(|&&v| v.is_finite() && v == 0.0)
                    .count()
            })
            .collect(),
    );

    // kind 0：正常面（≥10 个不同值，第 3 行全 NaN → 第 4 行不进 effective）
    // kind 1：只有 3 个不同值 → 整面默认结果
    // kind 2：全 NaN → 无有效日 → 默认结果
    let mk = |kind: u8| -> Array2<f32> {
        let mut slot = Array2::<f32>::from_elem((t, n), f32::NAN);
        for r in 0..t {
            for s in 0..n {
                slot[[r, s]] = match kind {
                    0 => ((r * 7 + s) % 13) as f32 * 0.1,
                    1 => ((r + s) % 3) as f32,
                    _ => f32::NAN,
                };
            }
        }
        if kind == 0 {
            for s in 0..n {
                slot[[3, s]] = f32::NAN;
            }
        }
        slot
    };

    let pre = Arc::new(btopt::build_bt_precomputed(
        &shared.ret_sum_gap1,
        &shared.ret_sum_gap5,
    ));

    let mut cmp = 0usize;
    let mut fail = 0usize;
    for kind in 0..3u8 {
        let slot = mk(kind);
        for ic_only in [false, true] {
            let (r1, r5) = reference(&shared, &counts, &slot, ic_only);
            let (o1, o5) = reference_opt(&shared, &counts, &slot, ic_only, &pre);
            let refs: [(usize, &LegacyBacktestResult, &LegacyBacktestResult); 2] =
                [(1, &r1, &o1), (5, &r5, &o5)];
            for (gap, want, want_opt) in refs.iter() {
                *n_cmp += 1;
                cmp += 1;
                let tag = format!("合成 kind{kind} ic_only={ic_only} gap{gap} 非opt参照vs opt参照");
                if !compare(&tag, want, want_opt, fails) {
                    *n_fail += 1;
                    fail += 1;
                }
            }
            for &block in [1usize, 3, 8].iter() {
                for (gap, want, want_opt) in refs.iter() {
                    let got = run_bt8(&slot, *gap, block, &shared, &counts, ic_only, &pre);
                    *n_cmp += 2;
                    cmp += 2;
                    let tag = format!("合成 kind{kind} ic_only={ic_only} gap{gap} block{block}");
                    if !compare(&format!("{tag} vs 非opt"), &got, want, fails) {
                        *n_fail += 1;
                        fail += 1;
                    }
                    if !compare(&format!("{tag} vs opt"), &got, want_opt, fails) {
                        *n_fail += 1;
                        fail += 1;
                    }
                }
            }
        }
    }
    println!(
        "  [合成边界] T=8 N=24 三种面（正常/少不同值/全 NaN）× ic_only × 块1/3/8：{}（{cmp} 次比较）",
        if fail == 0 { "PASS" } else { "FAIL" }
    );
}

fn main() {
    let data_dir = std::env::var("V3_DATA")
        .unwrap_or_else(|_| "/home/chenzongwei/neu_lab/data_yupei".to_string());
    let n_factors: usize = std::env::var("BT8_FACTORS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(3);
    let do_bench = std::env::var("BT8_BENCH").map(|v| v != "0").unwrap_or(true);

    println!("== bt8_check：BtAcc 逐位对账 ==");
    println!("数据目录: {data_dir}");

    let t0 = Instant::now();
    let shared = Arc::new(drive8::load_shared_dir(&data_dir, false));
    let counts: Arc<Vec<usize>> = Arc::new(
        (0..shared.restrict.nrows())
            .map(|r| {
                shared
                    .restrict
                    .row(r)
                    .iter()
                    .filter(|&&v| v.is_finite() && v == 0.0)
                    .count()
            })
            .collect(),
    );
    let t = shared.restrict.nrows();
    let n = shared.restrict.ncols();
    println!(
        "[加载] shared T={t} N={n} backtest_start={} 用时 {:.1}s",
        shared.backtest_start,
        t0.elapsed().as_secs_f64()
    );

    let t0 = Instant::now();
    let v3s = Arc::new(v3::v3_build(v2::build_shared(&data_dir)));
    println!("[加载] v3 中性化预计算 用时 {:.1}s", t0.elapsed().as_secs_f64());

    let t0 = Instant::now();
    let pre = Arc::new(btopt::build_bt_precomputed(
        &shared.ret_sum_gap1,
        &shared.ret_sum_gap5,
    ));
    println!(
        "[加载] 收益秩预排序 build_bt_precomputed 用时 {:.1}s（{} 行 × 2 gap）",
        t0.elapsed().as_secs_f64(),
        shared.ret_sum_gap1.nrows()
    );

    let names_txt = std::fs::read_to_string(format!("{data_dir}/sample_names.txt")).unwrap();
    let factor_names: Vec<String> = names_txt
        .lines()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();
    let n_factors = n_factors.min(factor_names.len());

    let mut n_cmp = 0usize;
    let mut n_fail = 0usize;
    let mut fails: Vec<String> = Vec::new();
    let mut bench_bt8 = 0.0f64;
    let mut bench_orig = 0.0f64;
    let mut bench_ref = 0.0f64;
    let mut bench_ref_opt = 0.0f64;
    let mut block_time = [0.0f64; 3];
    let mut sink = 0.0f64;

    // 先跑合成边界用例（不依赖真实数据，秒级）
    println!("\n[合成边界用例]");
    synthetic_check(&mut fails, &mut n_cmp, &mut n_fail);

    for (fi, nm) in factor_names.iter().take(n_factors).enumerate() {
        let tf = Instant::now();
        let raw = npy::as_f32_mat(npy::load(&format!("{data_dir}/factor_{nm}.npy")));
        let ranked = engine::rank_and_fill_missing_cross_sectional_median(&raw, &shared.restrict);
        let names = t8::derived_names_for_variant(nm, &shared.windows);
        assert_eq!(names.len(), 13, "派生面应为 13 个");
        let mut slots: Vec<Array2<f32>> = vec![ranked.clone()];
        for &w in &shared.windows {
            let (m, x, mn, sd) =
                engine::rolling_stats_f32_serial(&ranked, w, std::cmp::max(1, w / 2));
            slots.push(m);
            slots.push(x);
            slots.push(mn);
            slots.push(sd);
        }
        drop(raw);
        drop(ranked);
        println!(
            "\n[因子 {}/{}] {nm}  13 面构造用时 {:.1}s",
            fi + 1,
            n_factors,
            tf.elapsed().as_secs_f64()
        );

        // ---- raw 侧 13 面 × 3 块 × 2 gap（双参照：非 opt + 生产 opt）----
        for si in 0..slots.len() {
            let slot = &slots[si];
            let tr = Instant::now();
            let (r1, r5) = reference(&shared, &counts, slot, false);
            let ref_s = tr.elapsed().as_secs_f64();
            bench_ref += ref_s;
            let tr = Instant::now();
            let (o1, o5) = reference_opt(&shared, &counts, slot, false, &pre);
            bench_ref_opt += tr.elapsed().as_secs_f64();
            let refs: [(usize, &LegacyBacktestResult, &LegacyBacktestResult); 2] =
                [(1, &r1, &o1), (5, &r5, &o5)];
            let mut face_fail = 0usize;
            for (gap, want, want_opt) in refs.iter() {
                n_cmp += 1;
                let tag = format!("{nm}/{} gap{gap} 非opt参照vs opt参照", names[si]);
                if !compare(&tag, want, want_opt, &mut fails) {
                    face_fail += 1;
                    n_fail += 1;
                }
            }
            let tb = Instant::now();
            for (bi, &block) in BLOCK_SIZES.iter().enumerate() {
                let tk = Instant::now();
                for (gap, want, want_opt) in refs.iter() {
                    let got = run_bt8(slot, *gap, block, &shared, &counts, false, &pre);
                    n_cmp += 2;
                    let tag = format!("{nm}/{} gap{gap} block{block}", names[si]);
                    if !compare(&format!("{tag} vs 非opt"), &got, want, &mut fails) {
                        face_fail += 1;
                        n_fail += 1;
                    }
                    if !compare(&format!("{tag} vs opt"), &got, want_opt, &mut fails) {
                        face_fail += 1;
                        n_fail += 1;
                    }
                }
                block_time[bi] += tk.elapsed().as_secs_f64();
            }
            let bt_s = tb.elapsed().as_secs_f64();
            println!(
                "  [{si:2}] {:<42} ref {:>5.2}s | bt8×6 {:>5.2}s | {} (ic={} 项)",
                names[si],
                ref_s,
                bt_s,
                if face_fail == 0 { "PASS" } else { "FAIL" },
                r1.ic_values.len()
            );
        }

        // ---- 中性化面（v3）----
        for &si in NEU_SLOTS.iter() {
            let t_neu = Instant::now();
            let (neu, _times) = v3::v3_slot(slots[si].view(), &v3s);
            let (n1, n5) = reference(&shared, &counts, &neu, false);
            let (m1, m5) = reference_opt(&shared, &counts, &neu, false, &pre);
            let refs: [(usize, &LegacyBacktestResult, &LegacyBacktestResult); 2] =
                [(1, &n1, &m1), (5, &n5, &m5)];
            let mut face_fail = 0usize;
            for &block in &BLOCK_SIZES {
                for (gap, want, want_opt) in refs.iter() {
                    let got = run_bt8(&neu, *gap, block, &shared, &counts, false, &pre);
                    n_cmp += 2;
                    let tag = format!("{nm}/neu_{} gap{gap} block{block}", names[si]);
                    if !compare(&format!("{tag} vs 非opt"), &got, want, &mut fails) {
                        face_fail += 1;
                        n_fail += 1;
                    }
                    if !compare(&format!("{tag} vs opt"), &got, want_opt, &mut fails) {
                        face_fail += 1;
                        n_fail += 1;
                    }
                }
            }
            println!(
                "  [neu] {:<42} 中性化 {:.1}s  {}",
                names[si],
                t_neu.elapsed().as_secs_f64(),
                if face_fail == 0 { "PASS" } else { "FAIL" }
            );
        }

        // ---- ic_only 路径抽查（slot 0）----
        {
            let slot = &slots[0];
            let (i1, i5) = reference(&shared, &counts, slot, true);
            let (j1, j5) = reference_opt(&shared, &counts, slot, true, &pre);
            let refs: [(usize, &LegacyBacktestResult, &LegacyBacktestResult); 2] =
                [(1, &i1, &j1), (5, &i5, &j5)];
            let mut face_fail = 0usize;
            for &block in [64usize, 2818].iter() {
                for (gap, want, want_opt) in refs.iter() {
                    let got = run_bt8(slot, *gap, block, &shared, &counts, true, &pre);
                    n_cmp += 2;
                    let tag = format!("{nm}/{} ic_only gap{gap} block{block}", names[0]);
                    if !compare(&format!("{tag} vs 非opt"), &got, want, &mut fails) {
                        face_fail += 1;
                        n_fail += 1;
                    }
                    if !compare(&format!("{tag} vs opt"), &got, want_opt, &mut fails) {
                        face_fail += 1;
                        n_fail += 1;
                    }
                }
            }
            println!(
                "  [ico] {:<42} ic_only  {}",
                names[0],
                if face_fail == 0 { "PASS" } else { "FAIL" }
            );
        }

        // ---- 单线程计时：13 面（gap1+gap5）块 64，各跑 2 轮取最小值 ----
        // 第 1 轮顺带逐位比对「移植前(orig: 逐日比较排序) vs 移植后(bt8: pre 走查)」。
        if do_bench {
            let mut b_new = f64::INFINITY;
            let mut b_old = f64::INFINITY;
            for round in 0..2 {
                let t = Instant::now();
                let mut got_new: Vec<LegacyBacktestResult> = Vec::new();
                for slot in slots.iter() {
                    for gap in [1usize, 5] {
                        let r = run_bt8(slot, gap, 64, &shared, &counts, false, &pre);
                        sink += r.summary[5];
                        if round == 0 {
                            got_new.push(r);
                        }
                    }
                }
                b_new = b_new.min(t.elapsed().as_secs_f64());

                let t = Instant::now();
                let mut got_old: Vec<LegacyBacktestResult> = Vec::new();
                for slot in slots.iter() {
                    for gap in [1usize, 5] {
                        let r = run_orig(slot, gap, 64, &shared, &counts, false);
                        sink += r.summary[5];
                        if round == 0 {
                            got_old.push(r);
                        }
                    }
                }
                b_old = b_old.min(t.elapsed().as_secs_f64());

                if round == 0 {
                    for (k, (old, new)) in got_old.iter().zip(got_new.iter()).enumerate() {
                        n_cmp += 1;
                        let tag = format!("{nm} 移植前vs移植后 #{}", k / 2);
                        if !compare(&tag, old, new, &mut fails) {
                            n_fail += 1;
                        }
                    }
                }
            }
            println!(
                "  [计时] 13 面 × (gap1+gap5) 块64：移植前 {:.3}s | 移植后 {:.3}s | 加速 {:.2}×",
                b_old,
                b_new,
                b_old / b_new
            );
            bench_bt8 += b_new;
            bench_orig += b_old;
        }
    }

    println!("\n==== 对账汇总 ====");
    println!(
        "比较项 = 13 面 × {} 因子 × 3 块 × gap1/gap5 × 双参照（非 opt + 生产 opt）（+ 中性化面 + ic_only + 合成边界）：共 {} 次结果对比较",
        n_factors, n_cmp
    );
    if n_fail == 0 {
        println!("结果：全部 PASS —— 对 engine 非 opt 与 btopt 生产 opt 两个参照，summary 10 项 + ic_dates 全序列 + ic_values 全序列逐位一致");
    } else {
        println!("结果：FAIL {} / {} 次比较不一致；前 40 条明细：", n_fail, n_cmp);
        for m in fails.iter().take(40) {
            println!("    {m}");
        }
    }
    if do_bench {
        println!(
            "单线程计时（{} 个因子合计，13 面 × gap1+gap5，块64）：移植前 {:.3}s | 移植后 {:.3}s | 加速 {:.2}×",
            n_factors,
            bench_orig,
            bench_bt8,
            bench_orig / bench_bt8
        );
        println!(
            "参考实现同工作量：engine 非 opt {:.3}s | btopt 生产 opt {:.3}s",
            bench_ref, bench_ref_opt
        );
        println!(
            "对账期 bt8 六次喂块（gap1+gap5）耗时合计：块64 {:.2}s | 块997 {:.2}s | 块2818 {:.2}s",
            block_time[0], block_time[1], block_time[2]
        );
    }
    println!("(sink={sink:.1})");
    if n_fail > 0 {
        std::process::exit(1);
    }
}

// ---------------------------------------------------------------------------
// 「移植生产 O1 路径之前」的 bt8.rs 冻结副本（逐字照抄：每天对 filtered_future /
// filtered_signal 各做一次比较排序），只用于「移植前 vs 移植后」计时对照
// （bench 第 1 轮同时逐位比对两者，证明移植不改结果）。
// 当前实现见 `tail_perf_bench::bt8`。
// ---------------------------------------------------------------------------
#[allow(dead_code)]
mod orig {
    use std::cmp::Ordering;
    use std::collections::HashSet;
    use std::sync::Arc;

    use ndarray::{Array2, ArrayView2};

    use tail_perf_bench::engine::{
        self, annualized_sharpe_sample, max_drawdown_from_returns, nanmean_f64,
        nanstd_population, LegacyBacktestResult,
    };

    const EPS: f64 = 1e-12;

    fn ordinal_ranks(values: &[f32]) -> Vec<i64> {
        let mut indexed = values
            .iter()
            .copied()
            .enumerate()
            .collect::<Vec<(usize, f32)>>();
        indexed.sort_by(|lhs, rhs| match (lhs.1.is_nan(), rhs.1.is_nan()) {
            (true, true) => lhs.0.cmp(&rhs.0),
            (true, false) => Ordering::Greater,
            (false, true) => Ordering::Less,
            (false, false) => lhs
                .1
                .partial_cmp(&rhs.1)
                .unwrap_or(Ordering::Equal)
                .then_with(|| lhs.0.cmp(&rhs.0)),
        });
        let mut ranks = vec![0i64; values.len()];
        for (rank, (idx, _)) in indexed.iter().enumerate() {
            ranks[*idx] = rank as i64;
        }
        ranks
    }

    fn legacy_spearman_correlation(x: &[f32], y: &[f32]) -> f64 {
        if x.len() != y.len() || x.len() < 2 {
            return f64::NAN;
        }
        let xx = ordinal_ranks(x);
        let yy = ordinal_ranks(y);
        let n = x.len() as f64;
        let mut diff_sq_sum = 0.0;
        for idx in 0..x.len() {
            let diff = xx[idx] - yy[idx];
            diff_sq_sum += (diff * diff) as f64;
        }
        1.0 - 6.0 * diff_sq_sum / (n * (n * n - 1.0))
    }

    fn average_ranks(values: &[f32]) -> Vec<f64> {
        let mut indexed = values
            .iter()
            .copied()
            .enumerate()
            .collect::<Vec<(usize, f32)>>();
        indexed.sort_by(|lhs, rhs| {
            lhs.1
                .partial_cmp(&rhs.1)
                .unwrap_or(Ordering::Equal)
                .then_with(|| lhs.0.cmp(&rhs.0))
        });
        let mut ranks = vec![f64::NAN; values.len()];
        let mut start = 0usize;
        while start < indexed.len() {
            let value = indexed[start].1;
            let mut end = start + 1;
            while end < indexed.len() && indexed[end].1 == value {
                end += 1;
            }
            let avg_rank = (start + 1 + end) as f64 / 2.0;
            for item in indexed.iter().take(end).skip(start) {
                ranks[item.0] = avg_rank;
            }
            start = end;
        }
        ranks
    }

    fn default_result() -> LegacyBacktestResult {
        LegacyBacktestResult {
            summary: [f64::NAN; 10],
            ic_dates: Vec::new(),
            ic_values: Vec::new(),
        }
    }

    pub struct BtCtxOrig {
        pub gap: usize,
        pub shared: Arc<engine::Shared>,
        pub open_symbol_counts: Arc<Vec<usize>>,
        pub ic_only: bool,
    }

    impl BtCtxOrig {
        #[inline]
        pub fn ret(&self) -> &Array2<f32> {
            if self.gap == 1 {
                &self.shared.ret_gap1
            } else {
                &self.shared.ret_gap5
            }
        }
        #[inline]
        pub fn ret_sum(&self) -> &Array2<f32> {
            if self.gap == 1 {
                &self.shared.ret_sum_gap1
            } else {
                &self.shared.ret_sum_gap5
            }
        }
    }

    pub struct BtAccOrig {
        ctx: BtCtxOrig,
        n_stocks: usize,
        t_total: usize,
        local_t: usize,
        prev_row: Vec<f32>,
        held_signal: Vec<f32>,
        held_restrict_row: usize,
        distinct: HashSet<u32>,
        distinct_enough: bool,
        eff_idx: Vec<usize>,
        ic_dates: Vec<i32>,
        ic_values_f64: Vec<f64>,
        ic_values_f32: Vec<f32>,
        ratio_values: Vec<f64>,
        group_returns: Vec<Vec<f64>>,
        filtered_signal: Vec<f32>,
        filtered_ret: Vec<f32>,
        filtered_future: Vec<f32>,
        group_sums: Vec<f64>,
        group_counts: Vec<usize>,
    }

    impl BtAccOrig {
        pub fn new(ctx: BtCtxOrig) -> Self {
            let n_stocks = ctx.shared.restrict.ncols();
            let t_total = ctx.shared.restrict.nrows();
            let portf_num = 10usize;
            let ic_only = ctx.ic_only;
            Self {
                ctx,
                n_stocks,
                t_total,
                local_t: 0,
                prev_row: vec![f32::NAN; n_stocks],
                held_signal: vec![f32::NAN; n_stocks],
                held_restrict_row: 0,
                distinct: HashSet::new(),
                distinct_enough: false,
                eff_idx: Vec::new(),
                ic_dates: Vec::new(),
                ic_values_f64: Vec::new(),
                ic_values_f32: Vec::new(),
                ratio_values: Vec::new(),
                group_returns: if ic_only {
                    Vec::new()
                } else {
                    vec![Vec::new(); portf_num]
                },
                filtered_signal: Vec::with_capacity(n_stocks),
                filtered_ret: Vec::with_capacity(n_stocks),
                filtered_future: Vec::with_capacity(n_stocks),
                group_sums: vec![0.0; portf_num],
                group_counts: vec![0; portf_num],
            }
        }

        #[inline]
        fn row_has_finite(row: &[f32]) -> bool {
            row.iter().any(|v| v.is_finite())
        }

        pub fn push_block(&mut self, block: &ArrayView2<f32>, t0: usize) {
            let rows = block.nrows();
            let n = self.n_stocks;
            let gap = self.ctx.gap;
            let portf_num = 10usize;
            let mut sig_buf: Vec<f32> = vec![f32::NAN; n];

            for r in 0..rows {
                let t = t0 + r;
                let row = block.row(r);
                let row = row.as_slice().expect("slot 块必须行优先连续");

                if !self.distinct_enough && t + 1 < self.t_total {
                    for &v in row.iter() {
                        if v.is_finite() {
                            self.distinct.insert(v.to_bits());
                            if self.distinct.len() >= 10 {
                                self.distinct_enough = true;
                                break;
                            }
                        }
                    }
                }

                if t == 0 {
                    self.prev_row.copy_from_slice(row);
                    continue;
                }
                if self.ctx.shared.dates[t] <= self.ctx.shared.backtest_start {
                    self.prev_row.copy_from_slice(row);
                    continue;
                }

                if r > 0 {
                    sig_buf.copy_from_slice(block.row(r - 1).as_slice().unwrap());
                } else {
                    sig_buf.copy_from_slice(&self.prev_row);
                }
                let signal_all_nan = !Self::row_has_finite(&sig_buf);
                self.prev_row.copy_from_slice(row);
                if signal_all_nan {
                    continue;
                }

                if self.local_t % gap == 0 {
                    self.held_restrict_row = t - 1;
                    self.held_signal.copy_from_slice(&sig_buf);
                }
                let local_t = self.local_t;

                self.filtered_signal.clear();
                self.filtered_ret.clear();
                self.filtered_future.clear();
                for s in 0..n {
                    let signal_value = self.held_signal[s];
                    let ret_value = self.ctx.ret()[[t, s]];
                    let rv = self.ctx.shared.restrict[[self.held_restrict_row, s]];
                    let is_open = rv.is_finite() && rv == 0.0;
                    if signal_value.is_finite() && ret_value.is_finite() && is_open {
                        self.filtered_signal.push(signal_value);
                        self.filtered_ret.push(ret_value);
                        self.filtered_future.push(self.ctx.ret_sum()[[t, s]]);
                    }
                }

                if (local_t + 1) % gap == 0 {
                    let ic_value =
                        legacy_spearman_correlation(&self.filtered_future, &self.filtered_signal);
                    self.ic_dates.push(self.ctx.shared.dates[t]);
                    self.ic_values_f64.push(ic_value);
                    self.ic_values_f32.push(ic_value as f32);
                }

                let stocks_num = self.filtered_signal.len();
                if stocks_num < portf_num {
                    self.ratio_values.push(f64::NAN);
                    if !self.ctx.ic_only {
                        for b in 0..portf_num {
                            self.group_returns[b].push(0.0);
                        }
                    }
                    self.eff_idx.push(t);
                    self.local_t += 1;
                    continue;
                }

                let valid_symbol_num = self
                    .ctx
                    .open_symbol_counts
                    .get(t - 1)
                    .copied()
                    .unwrap_or(0);
                let ratio = if valid_symbol_num > 0 {
                    stocks_num as f64 / valid_symbol_num as f64
                } else {
                    f64::NAN
                };
                self.ratio_values.push(ratio);

                if self.ctx.ic_only {
                    self.eff_idx.push(t);
                    self.local_t += 1;
                    continue;
                }

                self.group_sums.iter_mut().for_each(|x| *x = 0.0);
                self.group_counts.iter_mut().for_each(|x| *x = 0);
                let ranks = average_ranks(&self.filtered_signal);
                for idx in 0..stocks_num {
                    let pct = ranks[idx] / stocks_num as f64;
                    let mut bucket = (pct * portf_num as f64).floor() as usize;
                    if bucket >= portf_num {
                        bucket = portf_num - 1;
                    }
                    self.group_sums[bucket] += self.filtered_ret[idx] as f64;
                    self.group_counts[bucket] += 1;
                }
                for bucket in 0..portf_num {
                    let v = if self.group_counts[bucket] == 0 {
                        0.0
                    } else {
                        self.group_sums[bucket] / self.group_counts[bucket] as f64
                    };
                    self.group_returns[bucket].push(v);
                }
                self.eff_idx.push(t);
                self.local_t += 1;
            }
        }

        pub fn finish(self) -> LegacyBacktestResult {
            if self.local_t == 0 || !self.distinct_enough {
                return default_result();
            }
            let date_size = self.eff_idx.len();
            let ic_mean = nanmean_f64(&self.ic_values_f64);
            let ic_std = nanstd_population(&self.ic_values_f64);
            let ir = if ic_std.is_nan() || ic_std <= EPS {
                f64::NAN
            } else {
                ic_mean.abs() / ic_std * (250.0 / self.ctx.gap as f64).sqrt()
            };
            let portf_num = 10usize;
            let summary = if self.ctx.ic_only {
                [
                    ic_mean,
                    ir,
                    0.0,
                    0.0,
                    0.0,
                    date_size as f64,
                    nanmean_f64(&self.ratio_values),
                    0.0,
                    0.0,
                    0.0,
                ]
            } else {
                let first_leg_cum = self.group_returns[0].iter().sum::<f64>();
                let last_leg_cum = self.group_returns[portf_num - 1].iter().sum::<f64>();
                let (long_idx, short_idx) = if first_leg_cum > last_leg_cum {
                    (0usize, portf_num - 1)
                } else {
                    (portf_num - 1, 0usize)
                };
                let mut ls_returns = vec![0.0_f64; date_size];
                let mut hedge_returns = vec![0.0_f64; date_size];
                for (local_t, &raw_eff_idx) in self.eff_idx.iter().enumerate() {
                    let long_ret = self.group_returns[long_idx][local_t];
                    let short_ret = self.group_returns[short_idx][local_t];
                    ls_returns[local_t] = long_ret - short_ret;
                    hedge_returns[local_t] =
                        long_ret - self.ctx.shared.index_ret[raw_eff_idx] as f64;
                }
                [
                    ic_mean,
                    ir,
                    nanmean_f64(&ls_returns) * 250.0,
                    annualized_sharpe_sample(&ls_returns),
                    max_drawdown_from_returns(&ls_returns),
                    date_size as f64,
                    nanmean_f64(&self.ratio_values),
                    nanmean_f64(&hedge_returns) * 250.0,
                    annualized_sharpe_sample(&hedge_returns),
                    max_drawdown_from_returns(&hedge_returns),
                ]
            };
            LegacyBacktestResult {
                summary,
                ic_dates: self.ic_dates,
                ic_values: self.ic_values_f32,
            }
        }
    }
}
