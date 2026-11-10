//! tail_v8_pipeline：三档改造的融合流水线（正式库版）。
//!
//! 相对 v7（`tail_v5_pipeline::process_task_with_values_v7`）的结构差异：
//!   v7：每个窗口先算完整 4 张 (T,N) 矩阵（各自 NaN 预填）→ 12 张落地
//!       → 每个面再被 preflight（重读 slot + restrict）→ 中性化（重读）→ 回测（重读）
//!   v8：按日期块（`BLOCK_ROWS` 行）产出 13 个面的**小块**（26 MB，落在 L3）
//!       → 同块就地 preflight 累积 → 同块批量中性化 → 同块回测累积
//!       全程不物化大矩阵，也不做 NaN 预填那趟白写。
//!
//! 一档：`tail_v8_roll`（流式产出）/ `tail_v8_preflight`（free-mask + 增量统计 + 快众数）
//! 二档：本文件（按日期块融合）+ `tail_v8_backtest`（增量回测）+ 中性化 range 入口
//! 三档：驱动层（`tail_backtest_engine` 的 worker 自行整理结果，主线程不做逐条串行）
//!
//! 逐位一致性：每个派生面的 preflight / 中性化 / raw 回测 / neu 回测 / summary 组装
//! 都与 v7 的对应步骤逐语句对齐，对账由引擎级 A/B 负责
//! （`/home/chenzongwei/neu_lab/ab_engine.py` 跑同一批因子比产物指纹）。

use std::sync::Arc;

use ndarray::{s, Array2, ArrayView2};

use crate::tail_v5_pipeline::{
    build_fold_values, compute_raw_cover_rate, default_legacy_backtest_result,
    derived_names_for_variant, precompute_open_symbol_counts,
    rank_and_fill_missing_cross_sectional_median, qualify_neu, qualify_raw, summary_from_row,
    IcRecord, SharedInputs, TailTask, TailTaskResult,
};
use crate::tail_v8_backtest::{BtAcc, BtCtx};
use crate::tail_v8_preflight::PfAcc;
use crate::tail_v8_roll::{self, RollScratch};

/// 日期块行数。64 行 × 7857 股 × 4B ≈ 2.0 MB；13 个面的块 ≈ 26 MB。
pub const BLOCK_ROWS: usize = 64;

/// 每线程一份的可复用缓冲。
pub struct V8Scratch {
    pub block_bufs: Vec<Array2<f32>>,
    pub roll: RollScratch,
    pub v3sc: crate::tail_v8_neu_v3::V3Scratch,
    pub block_rows: usize,
    pub n_stocks: usize,
}

impl V8Scratch {
    pub fn new(n_stocks: usize, windows: &[usize], block_rows: usize) -> Self {
        let ns = roll8_slot_count(windows);
        Self {
            block_bufs: (0..ns)
                .map(|_| Array2::<f32>::from_elem((block_rows, n_stocks), 0.0))
                .collect(),
            roll: RollScratch::new(0, n_stocks, windows, block_rows),
            v3sc: crate::tail_v8_neu_v3::V3Scratch::new(n_stocks),
            block_rows,
            n_stocks,
        }
    }
}

#[inline]
fn roll8_slot_count(windows: &[usize]) -> usize {
    tail_v8_roll::slot_count(windows)
}

/// 低覆盖放行时，单个派生面在 preflight 各闸门上的读数（写进对账产物）。
#[derive(Debug, Clone, serde::Serialize)]
pub struct LowCoverageFaceAudit {
    pub derived_name: String,
    pub majority_count_mean: f64,
    pub majority_count_threshold: f64,
    pub zero_ratio_mean: f64,
    pub zero_max_threshold: f64,
    pub nan_ratio_mean: f64,
    pub nan_max_threshold: f64,
    /// 缺失占比超过阈值、本该被淘汰、靠放行才留下
    pub nan_gate_bypassed: bool,
    /// 放行后 preflight 的最终结论（众数、零值两道照常判）
    pub preflight_passed: bool,
}

/// 低覆盖放行时，一个原始因子的对账记录。
#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct LowCoverageAudit {
    pub source_factor: String,
    /// 原始覆盖率（不填中位数，分母是当日可交易股票）
    pub raw_cover_rate: f64,
    pub cover_rate_threshold: f64,
    /// 覆盖率低于阈值、本该被淘汰、靠放行才留下
    pub cover_gate_bypassed: bool,
    /// 放行因子缺失值不填中位数（固定为 true，写出来方便对账）
    pub median_fill_skipped: bool,
    pub faces: Vec<LowCoverageFaceAudit>,
}

/// v8 融合版：与 `process_task_with_values_v7` 同语义。
pub fn process_task_with_values_v8(
    task: &TailTask,
    raw_values: Array2<f32>,
    shared: &SharedInputs,
    sc: &mut V8Scratch,
) -> Result<TailTaskResult, String> {
    process_task_with_values_v8_opts(task, raw_values, shared, sc, None)
}

/// v8 融合版，可选“低覆盖放行”（撤单类因子用）。
///
/// `low_cov = None`：与 `process_task_with_values_v8` 完全相同（默认路径，逐位不变）。
/// `low_cov = Some(audit)`：该因子在部分市场/时期天然没有数据（例如 2021-11-05 前沪市无撤单），
///   1) 缺失值不填横截面中位数，只在有值的股票上排名、滚动、中性化、回测；
///   2) 原始覆盖率闸门与 preflight 缺失占比闸门只记录读数、不淘汰；
///   3) 众数、零值两道闸门照常（不填中位数后，众数只统计真实的同值，不再被缺失撑大）；
///   4) 中性化输出在输入缺失的位置强制保持缺失（中性化内部的行业中位补齐不进入回测）。
///   各闸门读数写进 `audit`。
pub(crate) fn process_task_with_values_v8_opts(
    task: &TailTask,
    raw_values: Array2<f32>,
    shared: &SharedInputs,
    sc: &mut V8Scratch,
    mut low_cov: Option<&mut LowCoverageAudit>,
) -> Result<TailTaskResult, String> {
    let mut result = TailTaskResult {
        source_factor: task.source_factor.clone(),
        ..Default::default()
    };
    let allow_low_cov = low_cov.is_some();

    let raw_cover_before_fill = compute_raw_cover_rate(
        &raw_values.view(),
        &shared.restrict.view(),
        &shared.ret_gap1.view(),
        10,
    );
    let ranked_raw = if allow_low_cov {
        crate::tail_v2_rank_roll_factor::rank_axis1_average_f32_serial(&raw_values)
    } else {
        rank_and_fill_missing_cross_sectional_median(&raw_values, &shared.restrict)
    };
    let raw_cover_rate = compute_raw_cover_rate(
        &ranked_raw.view(),
        &shared.restrict.view(),
        &shared.ret_gap1.view(),
        10,
    );
    if let Some(audit) = low_cov.as_deref_mut() {
        audit.source_factor = task.source_factor.clone();
        audit.raw_cover_rate = raw_cover_rate;
        audit.cover_rate_threshold = shared.config.cover_rate;
        audit.cover_gate_bypassed = raw_cover_rate < shared.config.cover_rate;
        audit.median_fill_skipped = true;
    }
    if !allow_low_cov && raw_cover_rate < shared.config.cover_rate {
        return Ok(TailTaskResult {
            source_factor: task.source_factor.clone(),
            eliminated_by_raw_cover: true,
            raw_cover_before_fill,
            raw_cover_after_fill: raw_cover_rate,
            ..Default::default()
        });
    }
    result.raw_cover_before_fill = raw_cover_before_fill;
    result.raw_cover_after_fill = raw_cover_rate;

    let open_symbol_counts = precompute_open_symbol_counts(&shared.restrict.view());

    run_variant_v8(
        &task.source_factor,
        ranked_raw,
        shared,
        &open_symbol_counts,
        sc,
        &mut result,
        low_cov.as_deref_mut().map(|a| &mut a.faces),
    )?;

    if shared.fold {
        let folded = build_fold_values(&raw_values);
        let ranked_fold = if allow_low_cov {
            crate::tail_v2_rank_roll_factor::rank_axis1_average_f32_serial(&folded)
        } else {
            rank_and_fill_missing_cross_sectional_median(&folded, &shared.restrict)
        };
        run_variant_v8(
            &format!("{}_fold", task.source_factor),
            ranked_fold,
            shared,
            &open_symbol_counts,
            sc,
            &mut result,
            low_cov.as_deref_mut().map(|a| &mut a.faces),
        )?;
    }
    Ok(result)
}

fn run_variant_v8(
    variant_name: &str,
    ranked: Array2<f32>,
    shared: &SharedInputs,
    open_symbol_counts: &[usize],
    sc: &mut V8Scratch,
    result: &mut TailTaskResult,
    mut low_cov_faces: Option<&mut Vec<LowCoverageFaceAudit>>,
) -> Result<(), String> {
    let allow_low_cov = low_cov_faces.is_some();
    let (t, n) = ranked.dim();
    let names = derived_names_for_variant(variant_name, shared.windows.as_slice());
    let ns = names.len();
    result.derived_factor_count += ns;

    let mask = shared
        .free_mask
        .as_ref()
        .expect("v8 融合路径需要 SharedInputs::free_mask");
    let ns_shared = shared
        .neutralize_std_shared
        .as_ref()
        .expect("v8 融合路径需要 neutralize_std_shared");
    let pre = shared
        .bt_pre
        .as_ref()
        .expect("v8 融合路径需要 bt_pre（O1 回测路径）");
    let cfg = &shared.config;
    let ic_only = cfg.ic_only;
    let block = sc.block_rows.max(1);

    // 每面一套累积器
    let mut pf: Vec<PfAcc> = (0..ns)
        .map(|_| {
            let mut acc = PfAcc::new(
                t,
                n,
                cfg.majority_count_threshold,
                cfg.zero_max_threshold,
                cfg.nan_max_threshold,
            );
            if allow_low_cov {
                acc.set_ignore_nan(true);
            }
            acc
        })
        .collect();
    let mk_ctx = |gap: usize| BtCtx {
        gap,
        ret: if gap == 1 { &shared.ret_gap1 } else { &shared.ret_gap5 },
        ret_sum: if gap == 1 { &shared.ret_sum_gap1 } else { &shared.ret_sum_gap5 },
        restrict: &shared.restrict,
        index: &shared.index_ret,
        dates: shared.dates.as_slice(),
        backtest_start: shared.backtest_start,
        portf_num: 10,
        open_symbol_counts,
        ic_only,
        pre,
    };
    let mut raw_bt1: Vec<Option<BtAcc>> = (0..ns).map(|_| None).collect();
    let mut raw_bt5: Vec<Option<BtAcc>> = (0..ns).map(|_| None).collect();
    let mut neu_bt1: Vec<Option<BtAcc>> = (0..ns).map(|_| None).collect();
    let mut neu_bt5: Vec<Option<BtAcc>> = (0..ns).map(|_| None).collect();
    let mut alive = vec![true; ns];
    // metrics-only 需要"preflight 未过也要出 all_* 指标"，此时不做早停
    let early_stop = !cfg.save_all_metrics;

    for t0 in (0..t).step_by(block) {
        let t1 = (t0 + block).min(t);
        let rows = t1 - t0;
        tail_v8_roll::rolling_block(
            &ranked,
            shared.windows.as_slice(),
            t0,
            t1,
            &mut sc.roll,
            &mut sc.block_bufs,
        );

        // 1) preflight 统计（始终累积，保证 finish() 与生产逐位一致）
        for si in 0..ns {
            let blk = sc.block_bufs[si].slice(s![0..rows, ..]);
            pf[si].push_block(&blk, mask, t0);
            if alive[si] && early_stop && pf[si].definitely_failed() {
                alive[si] = false;
                raw_bt1[si] = None;
                raw_bt5[si] = None;
                neu_bt1[si] = None;
                neu_bt5[si] = None;
            }
        }

        // 2) raw 侧回测（ic_only 时生产会跳过，这里也跳过以省流量）
        if !ic_only {
            for si in 0..ns {
                if !alive[si] {
                    continue;
                }
                let blk = sc.block_bufs[si].slice(s![0..rows, ..]);
                let a = raw_bt1[si].get_or_insert_with(|| BtAcc::new(mk_ctx(1)));
                a.push_block(&blk, t0);
                let b = raw_bt5[si].get_or_insert_with(|| BtAcc::new(mk_ctx(5)));
                b.push_block(&blk, t0);
            }
        }

        // 3) 中性化同一块（批量）→ neu 侧回测
        let need: Vec<usize> = (0..ns).filter(|&i| alive[i]).collect();
        if !need.is_empty() {
            let views: Vec<ArrayView2<f32>> =
                need.iter().map(|&i| sc.block_bufs[i].slice(s![0..rows, ..])).collect();
            // v3（默认，比 v2 快 2.6~2.8×，数值逐位一致）；TAIL_NEU_V2=1 切回 v2 做诊断。
            // 行业与纯风格共用同一套逐日结构，只有残差那一步差一列行业 one-hot。
            let use_v2 = std::env::var("TAIL_NEU_V2").is_ok();
            let mut neutrals = if !use_v2 {
                let v3s = shared
                    .v3_shared
                    .as_ref()
                    .expect("v8 融合路径需要 v3_shared（可用 TAIL_NEU_V2=1 切回 v2）");
                crate::tail_v8_neu_v3::v3_slots_range(
                    &views,
                    v3s,
                    shared.industry_neutralize,
                    t0,
                    t1,
                    &mut sc.v3sc,
                )?
            } else {
                crate::factor_neutralize_std::neutralize_std_slots_f32_v2_resid_batch_range(
                    &views,
                    ns_shared,
                    shared.industry_neutralize,
                    t0,
                    t1,
                )?
            };
            if allow_low_cov {
                // 放行因子：输入缺失的位置，中性化输出也保持缺失（不让中性化内部的
                // 行业中位补齐把“没有数据”变成一个数进入回测）。
                for (k, &si) in need.iter().enumerate() {
                    let input = sc.block_bufs[si].slice(s![0..rows, ..]);
                    ndarray::Zip::from(&mut neutrals[k]).and(&input).for_each(|o, &x| {
                        if !x.is_finite() {
                            *o = f32::NAN;
                        }
                    });
                }
            }
            for (k, &si) in need.iter().enumerate() {
                let nb = neutrals[k].view();
                let a = neu_bt1[si].get_or_insert_with(|| BtAcc::new(mk_ctx(1)));
                a.push_block(&nb, t0);
                let b = neu_bt5[si].get_or_insert_with(|| BtAcc::new(mk_ctx(5)));
                b.push_block(&nb, t0);
            }
        }
    }

    // 4) 汇总（与 process_v7_slot_neu 的顺序/条件完全一致）
    let reports: Vec<crate::tail_v8_preflight::PfReport> =
        pf.into_iter().map(|a| a.finish()).collect();
    for si in 0..ns {
        let report = reports[si];
        if !report.passed {
            if report.majority_count_mean > cfg.majority_count_threshold {
                result.preflight_maj_failed_windows += 1;
            }
            if report.zero_ratio_mean >= cfg.zero_max_threshold {
                result.preflight_zero_failed_windows += 1;
            }
            if report.nan_ratio_mean >= cfg.nan_max_threshold {
                result.preflight_nan_failed_windows += 1;
            }
        } else {
            result.any_window_passed_preflight = true;
        }

        let derived_name = &names[si];
        if let Some(faces) = low_cov_faces.as_deref_mut() {
            faces.push(LowCoverageFaceAudit {
                derived_name: derived_name.clone(),
                majority_count_mean: report.majority_count_mean,
                majority_count_threshold: cfg.majority_count_threshold,
                zero_ratio_mean: report.zero_ratio_mean,
                zero_max_threshold: cfg.zero_max_threshold,
                nan_ratio_mean: report.nan_ratio_mean,
                nan_max_threshold: cfg.nan_max_threshold,
                nan_gate_bypassed: report.nan_ratio_mean >= cfg.nan_max_threshold,
                preflight_passed: report.passed,
            });
        }
        let raw_gap1_result =
            raw_bt1[si].take().map(|a| a.finish()).unwrap_or_else(default_legacy_backtest_result);
        let raw_gap5_result =
            raw_bt5[si].take().map(|a| a.finish()).unwrap_or_else(default_legacy_backtest_result);
        let neu_gap1_result =
            neu_bt1[si].take().map(|a| a.finish()).unwrap_or_else(default_legacy_backtest_result);
        let neu_gap5_result =
            neu_bt5[si].take().map(|a| a.finish()).unwrap_or_else(default_legacy_backtest_result);

        let mut raw_gap1_row =
            summary_from_row(derived_name, "rolled", 1, variant_name, &raw_gap1_result.summary);
        let mut raw_gap5_row =
            summary_from_row(derived_name, "rolled", 5, variant_name, &raw_gap5_result.summary);
        let mut neu_gap1_row =
            summary_from_row(derived_name, "neu", 1, variant_name, &neu_gap1_result.summary);
        let mut neu_gap5_row =
            summary_from_row(derived_name, "neu", 5, variant_name, &neu_gap5_result.summary);
        raw_gap1_row.preflight_passed = report.passed;
        raw_gap5_row.preflight_passed = report.passed;
        neu_gap1_row.preflight_passed = report.passed;
        neu_gap5_row.preflight_passed = report.passed;

        let raw_gap1_keep = report.passed && qualify_raw(&raw_gap1_row, 1, cfg);
        let raw_gap5_keep = report.passed && qualify_raw(&raw_gap5_row, 5, cfg);
        let neu_gap1_keep = report.passed && qualify_neu(&neu_gap1_row, 1, cfg);
        let neu_gap5_keep = report.passed && qualify_neu(&neu_gap5_row, 5, cfg);

        if raw_gap1_keep || raw_gap5_keep || neu_gap1_keep || neu_gap5_keep {
            result.passed = true;
        }

        // metrics-only：无论是否达到候选阈值都保存全部指标（ic_only 下 raw 未回测，跳过 raw all）
        if cfg.save_all_metrics {
            if !ic_only {
                result.all_raw_summary_gap1.push(raw_gap1_row.clone());
                result.all_raw_summary_gap5.push(raw_gap5_row.clone());
                result.all_raw_ic_gap1.push(IcRecord {
                    factor_name: derived_name.clone(),
                    dates: raw_gap1_result.ic_dates.clone(),
                    values: raw_gap1_result.ic_values.clone(),
                });
                result.all_raw_ic_gap5.push(IcRecord {
                    factor_name: derived_name.clone(),
                    dates: raw_gap5_result.ic_dates.clone(),
                    values: raw_gap5_result.ic_values.clone(),
                });
            }
            result.all_neu_summary_gap1.push(neu_gap1_row.clone());
            result.all_neu_summary_gap5.push(neu_gap5_row.clone());
            result.all_neu_ic_gap1.push(IcRecord {
                factor_name: derived_name.clone(),
                dates: neu_gap1_result.ic_dates.clone(),
                values: neu_gap1_result.ic_values.clone(),
            });
            result.all_neu_ic_gap5.push(IcRecord {
                factor_name: derived_name.clone(),
                dates: neu_gap5_result.ic_dates.clone(),
                values: neu_gap5_result.ic_values.clone(),
            });
        }

        if raw_gap1_keep {
            result.raw_summary_gap1.push(raw_gap1_row.clone());
            result.raw_ic_gap1.push(IcRecord {
                factor_name: derived_name.clone(),
                dates: raw_gap1_result.ic_dates.clone(),
                values: raw_gap1_result.ic_values.clone(),
            });
        }
        if raw_gap5_keep {
            result.raw_summary_gap5.push(raw_gap5_row.clone());
            result.raw_ic_gap5.push(IcRecord {
                factor_name: derived_name.clone(),
                dates: raw_gap5_result.ic_dates.clone(),
                values: raw_gap5_result.ic_values.clone(),
            });
        }
        if neu_gap1_keep {
            result.neu_summary_gap1.push(neu_gap1_row.clone());
        }
        if neu_gap5_keep {
            result.neu_summary_gap5.push(neu_gap5_row.clone());
        }
        if raw_gap1_keep || neu_gap1_keep {
            result.neu_ic_gap1.push(IcRecord {
                factor_name: derived_name.clone(),
                dates: neu_gap1_result.ic_dates.clone(),
                values: neu_gap1_result.ic_values.clone(),
            });
        }
        if raw_gap5_keep || neu_gap5_keep {
            result.neu_ic_gap5.push(IcRecord {
                factor_name: derived_name.clone(),
                dates: neu_gap5_result.ic_dates.clone(),
                values: neu_gap5_result.ic_values.clone(),
            });
        }
    }
    let _ = Arc::strong_count(&shared.restrict);
    Ok(())
}
