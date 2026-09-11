//! v8：三档改造的融合流水线（Lead 独占）。
//!
//! 相对 v7（`engine::run_factor_v7`）的结构差异：
//!   v7：每个窗口先算完整 4 张 (T,N) 矩阵（各自 NaN 预填）→ 12 张落地
//!       → 每个面再被 preflight（重读 slot + restrict）→ 中性化（重读）→ 回测（重读）
//!   v8：按日期块（BLOCK_ROWS 行）产出 13 个面的**小块**（26 MB，落在 L3）
//!       → 同块就地 preflight 累积 → 同块中性化 → 同块回测累积
//!       全程不物化大矩阵，也不做 NaN 预填那趟白写。
//!
//! 一档：roll8（流式产出、免预填）/ pf8（free-mask + 增量统计 + 快众数）
//! 二档：本文件（按日期块融合，不落地 12 张 slot 矩阵）
//! 三档：drive8.rs（驱动层去掉主线程串行）
//!
//! 对账：`src/bin/v8_check.rs` 逐位比对 v8 与基线（drive8::run_factor_v7_real）。

use std::sync::Arc;

use ndarray::{s, Array2};

use crate::bt8::{BtAcc, BtCtx};
use crate::engine;
use crate::neu8;
use crate::pf8::{self, FreeMask, PfAcc};
use crate::roll8::{self, RollScratch};
use crate::t8::{self, T8Result, BLOCK_ROWS};
use crate::v3::{V3Scratch, V3Shared};

/// v8 的共享上下文（Arc 字段，可跨线程共享；大矩阵不复制）。
pub struct V8Ctx {
    pub shared: Arc<engine::Shared>,
    pub v3: Arc<V3Shared>,
    pub mask: Arc<FreeMask>,
    pub open_symbol_counts: Arc<Vec<usize>>,
    pub ic_only: bool,
    /// 生产 O1 回测路径的收益秩预排序（bt8 移植后用）
    pub pre: Arc<crate::btopt::BtPrecomputed>,
    /// 一档第 4 条：填充前覆盖率是纯诊断量，默认不算（省一整趟 3 矩阵扫描）。
    pub want_cover_before_fill: bool,
    /// 二档：日期块行数。
    pub block_rows: usize,
}

impl V8Ctx {
    fn bt_ctx(&self, gap: usize) -> BtCtx {
        BtCtx {
            gap,
            shared: self.shared.clone(),
            open_symbol_counts: self.open_symbol_counts.clone(),
            ic_only: self.ic_only,
            pre: self.pre.clone(),
        }
    }
}

/// 便捷构造：把 engine::Shared + v3 预计算打包成 V8Ctx（含 free-mask 与每日可交易数）。
pub fn build_v8_ctx(
    shared: Arc<engine::Shared>,
    v3: Arc<V3Shared>,
    ic_only: bool,
    want_cover_before_fill: bool,
) -> V8Ctx {
    let mask = Arc::new(pf8::build_free_mask(&shared.restrict.view()));
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
    let pre = Arc::new(crate::btopt::build_bt_precomputed(
        &shared.ret_sum_gap1,
        &shared.ret_sum_gap5,
    ));
    V8Ctx {
        shared,
        v3,
        mask,
        open_symbol_counts,
        ic_only,
        pre,
        want_cover_before_fill,
        block_rows: BLOCK_ROWS,
    }
}

/// 每线程一份的可复用缓冲（跨块、跨 slot、跨因子复用，热路径零分配）。
pub struct V8Scratch {
    pub block_bufs: Vec<Array2<f32>>,
    pub roll: RollScratch,
    pub v3scratch: V3Scratch,
    pub block_rows: usize,
    pub n_stocks: usize,
}

impl V8Scratch {
    pub fn new(n_stocks: usize, windows: &[usize], block_rows: usize) -> Self {
        let ns = t8::slot_count(windows);
        Self {
            block_bufs: (0..ns)
                .map(|_| Array2::<f32>::from_elem((block_rows, n_stocks), 0.0))
                .collect(),
            roll: RollScratch::new(0, n_stocks, windows, block_rows),
            v3scratch: V3Scratch::new(n_stocks),
            block_rows,
            n_stocks,
        }
    }
}

/// 一个原始因子的完整 v8 流水线。
pub fn run_factor_v8(source_factor: &str, raw: Array2<f32>, ctx: &V8Ctx, sc: &mut V8Scratch) -> T8Result {
    let s = &ctx.shared;
    let mut result = T8Result {
        source_factor: source_factor.to_string(),
        ..Default::default()
    };

    let tp = std::time::Instant::now();
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

    run_variant_v8(source_factor, ranked_raw, ctx, sc, &mut result);

    if s.fold {
        let folded = engine::build_fold_values(&raw);
        let ranked_fold = engine::rank_and_fill_missing_cross_sectional_median(&folded, &s.restrict);
        run_variant_v8(
            &format!("{}_fold", source_factor),
            ranked_fold,
            ctx,
            sc,
            &mut result,
        );
    }
    result
}

fn run_variant_v8(
    variant_name: &str,
    ranked: Array2<f32>,
    ctx: &V8Ctx,
    sc: &mut V8Scratch,
    result: &mut T8Result,
) {
    let s = &ctx.shared;
    let t = ranked.nrows();
    let n = ranked.ncols();
    let names = t8::derived_names_for_variant(variant_name, &s.windows);
    let ns = names.len();
    let block = sc.block_rows.max(1);

    let mut pf: Vec<PfAcc> = (0..ns)
        .map(|_| {
            PfAcc::new(
                t,
                n,
                s.majority_count_threshold,
                s.zero_max_threshold,
                s.nan_max_threshold,
            )
        })
        .collect();
    let mut raw_bt1: Vec<Option<BtAcc>> = (0..ns).map(|_| None).collect();
    let mut raw_bt5: Vec<Option<BtAcc>> = (0..ns).map(|_| None).collect();
    let mut neu_bt1: Vec<Option<BtAcc>> = (0..ns).map(|_| None).collect();
    let mut neu_bt5: Vec<Option<BtAcc>> = (0..ns).map(|_| None).collect();
    let mut alive = vec![true; ns];

    for t0 in (0..t).step_by(block) {
        let t1 = (t0 + block).min(t);
        let rows = t1 - t0;
        let tr = std::time::Instant::now();
        roll8::rolling_block(&ranked, &s.windows, t0, t1, &mut sc.roll, &mut sc.block_bufs);
        t8::tick(&t8::T_ROLL, tr);

        for si in 0..ns {
            // preflight 统计始终累积（很便宜），保证 finish() 与生产逐位一致
            let blk = sc.block_bufs[si].slice(s![0..rows, ..]);
            let tpf = std::time::Instant::now();
            pf[si].push_block(&blk, &ctx.mask, t0);
            t8::tick(&t8::T_PF, tpf);
            if !alive[si] {
                continue;
            }
            if pf[si].definitely_failed() {
                alive[si] = false;
                raw_bt1[si] = None;
                raw_bt5[si] = None;
                neu_bt1[si] = None;
                neu_bt5[si] = None;
                continue;
            }

            // raw 侧回测（ic_only 时生产也会跳过，这里一并跳过以省流量）
            let tb_raw = std::time::Instant::now();
            if !ctx.ic_only {
                let a = raw_bt1[si].get_or_insert_with(|| BtAcc::new(ctx.bt_ctx(1)));
                a.push_block(&blk, t0);
                let b = raw_bt5[si].get_or_insert_with(|| BtAcc::new(ctx.bt_ctx(5)));
                b.push_block(&blk, t0);
            }
            t8::tick(&t8::T_BT, tb_raw);

            // 中性化同一块 → neu 侧回测
            let tn = std::time::Instant::now();
            let neu_blk = neu8::neutralize_block(&blk, &ctx.v3, t0, t1, &mut sc.v3scratch);
            t8::tick(&t8::T_NEU, tn);
            let tb = std::time::Instant::now();
            let nb = neu_blk.view();
            let a = neu_bt1[si].get_or_insert_with(|| BtAcc::new(ctx.bt_ctx(1)));
            a.push_block(&nb, t0);
            let b = neu_bt5[si].get_or_insert_with(|| BtAcc::new(ctx.bt_ctx(5)));
            b.push_block(&nb, t0);
            t8::tick(&t8::T_BT, tb);
        }
    }

    let reports: Vec<t8::PfReport> = pf.into_iter().map(|a| a.finish()).collect();
    for si in 0..ns {
        let report = reports[si];
        let name = &names[si];
        if !report.passed {
            if report.majority_count_mean > s.majority_count_threshold {
                result.preflight_maj_failed += 1;
            }
            if report.zero_ratio_mean >= s.zero_max_threshold {
                result.preflight_zero_failed += 1;
            }
            if report.nan_ratio_mean >= s.nan_max_threshold {
                result.preflight_nan_failed += 1;
            }
            continue;
        }
        result.any_window_passed_preflight = true;

        if !ctx.ic_only {
            if let Some(a) = raw_bt1[si].take() {
                result.raw_summaries.push((name.clone(), 1, a.finish().summary));
            }
            if let Some(b) = raw_bt5[si].take() {
                result.raw_summaries.push((name.clone(), 5, b.finish().summary));
            }
        }
        if let Some(a) = neu_bt1[si].take() {
            result.neu_summaries.push((name.clone(), 1, a.finish().summary));
        }
        if let Some(b) = neu_bt5[si].take() {
            result.neu_summaries.push((name.clone(), 5, b.finish().summary));
        }
    }
}

/// 便捷：从 `engine::Shared` 直接建 V8Ctx（不经 drive8::BaseCtx）。
pub fn v8_scratch_for(ctx: &V8Ctx) -> V8Scratch {
    V8Scratch::new(ctx.shared.restrict.ncols(), &ctx.shared.windows, ctx.block_rows)
}
