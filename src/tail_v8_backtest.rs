//! tail_v8_backtest：生产 O1 回测路径的「按日期块增量累积」实现（v8 二档消费侧）。
//!
//! 生产 `legacy_backtest_gap1_gap5_single_slot_opt` 需要整张 (T,N) slot；v8 融合流水线
//! 只有日期块，本模块把它的逐日循环拆成「喂块 → 累积 → finish」，状态只有 IC 序列 +
//! gen/stamp 走查缓冲 + held 信号行缓存，**不物化 slot 大矩阵**。
//!
//! 逐位一致性依据（与 `legacy_backtest_single_factor_with_effective_opt` 逐语句对齐）：
//! - `effective_raw_indices`：raw_eff_idx ∈ 1..T，`dates[t] > backtest_start` 且**第 t-1 行有有限值**；
//!   块边界处靠 `prev_row` 保留上一块最后一行。
//! - `local_t % gap == 0` 时刷新 held 行 = t-1；held 行的 slot 值缓存进 `held_signal`
//!   （held 行可能落在上一块），restrict 行直接读全局矩阵。
//! - `(local_t+1) % gap == 0` 时算 IC：收益秩走 `pre.orders_*[t]` 的 gen/stamp walk，
//!   信号秩用 `rank_both_radix`（同一次排序同时出 ordinal 秩与平均秩）。
//! - `stocks_num < portf_num` → `continue`（该日 group_returns 记 0.0、ratio 记 NaN）。
//! - `has_enough_unique_values`：扫 raw_idx ∈ 0..T-1 的有限值，够 10 个不同值即短路；
//!   不足则整面返回默认结果（全 NaN summary + 空 IC）。
//! - summary 12 项分 ic_only / 非 ic_only 两条路径，与生产逐语句相同。
//!
//! 自测：`rp.tail_v8_selfcheck("bt", data_dir)`。

use std::cell::RefCell;

use ndarray::{Array1, Array2, ArrayView2};

use crate::tail_v5_pipeline::{
    annualized_sharpe_sample, compute_mprob, compute_ssm, default_legacy_backtest_result,
    max_drawdown_from_returns, nanmean_f64, nanstd_population, rank_both_radix_into, BtPrecomputed,
    LegacyBacktestResult, EPS,
};

/// 「市场有效谓词」位图缓存（线程局部，跨累积器复用）。
///
/// 谓词 = `ret[t, s]` 有限 且 `restrict[held_row, s]` 有限且 == 0。
/// 它与「面」(slot) 和 raw/neu 变体**都无关**，只取决于
/// (ret 数组身份, restrict 数组身份, t, held_row)。
///
/// 同一 (gap, 日) 下 raw 与 neu 的有效股票集合之所以完全相同，正是因为该谓词相同，
/// 且 v3 中性化逐位保持 NaN 位置（signal 有限性一致）。于是：谓词按
/// (gap, 日) 算一次、位图被 raw/neu/各面共用；每个累积器只额外做
/// 「signal 有限性判定 + 收集」，结果与逐股全扫**逐位相同**（精确，非近似）。
///
/// 失效：`BtAcc::new` 清空（新一次回测/新一批共享输入 → 数组地址可能被复用）；
/// `push_block` 的块起点 t0 变化时清空（条目只在同一日期块内跨 raw/neu 使用）。
struct ValidPredCache {
    block_t0: usize,
    /// key = (ret 首元素地址, restrict 首元素地址, t, held_row, n)
    entries: std::collections::HashMap<(usize, usize, usize, usize, usize), Vec<u64>>,
}

impl ValidPredCache {
    #[inline]
    fn invalidate(&mut self) {
        self.entries.clear();
        self.block_t0 = usize::MAX;
    }

    #[inline]
    fn begin_block(&mut self, t0: usize) {
        if self.block_t0 != t0 {
            self.entries.clear();
            self.block_t0 = t0;
        }
    }

    /// 谓词位图是否已算过（命中即可跳过 ret/restrict 重扫）。
    #[inline]
    fn contains(&self, key: (usize, usize, usize, usize, usize)) -> bool {
        self.entries.contains_key(&key)
    }

    /// 命中取图（调用前必须 `contains` 为真）。
    #[inline]
    fn get(&self, key: (usize, usize, usize, usize, usize)) -> &[u64] {
        self.entries.get(&key).expect("谓词位图刚插入")
    }

    /// 首次扫描时顺带记下位图，供后续 raw/neu/各面复用。
    #[inline]
    fn store(&mut self, key: (usize, usize, usize, usize, usize), bits: Vec<u64>) {
        self.entries.insert(key, bits);
    }
}

thread_local! {
    /// 每线程一份：同一 (gap, 日) 下 raw/neu/各面共享同一张谓词位图。
    static VALID_PRED: RefCell<ValidPredCache> =
        RefCell::new(ValidPredCache { block_t0: usize::MAX, entries: std::collections::HashMap::new() });
}

pub struct BtCtx<'a> {
    pub gap: usize,
    pub ret: &'a Array2<f32>,
    pub ret_sum: &'a Array2<f32>,
    pub restrict: &'a Array2<f32>,
    pub index: &'a Array1<f32>,
    pub dates: &'a [i32],
    pub backtest_start: i32,
    pub portf_num: usize,
    pub open_symbol_counts: &'a [usize],
    pub ic_only: bool,
    pub pre: &'a BtPrecomputed,
}

pub struct BtAcc<'a> {
    ctx: BtCtx<'a>,
    n_stocks: usize,
    t_total: usize,
    local_t: usize,
    /// 上一块最后一行的 slot 值（块边界处取 signal 行用）
    prev_row: Vec<f32>,
    /// held signal 行缓存（held 行可能落在上一块）
    held_signal: Vec<f32>,
    held_restrict_row: usize,
    /// has_enough_unique_values 的增量判定
    distinct: std::collections::HashSet<u32>,
    distinct_enough: bool,
    /// O1 走查状态（跨块复用）
    gen: Vec<u32>,
    stamp: Vec<u32>,
    walk_buf: Vec<i64>,
    gen_id: u32,
    /// 逐日累积
    eff_idx: Vec<usize>,
    ic_dates: Vec<i32>,
    ic_values_f64: Vec<f64>,
    ic_values_f32: Vec<f32>,
    ratio_values: Vec<f64>,
    group_returns: Vec<Vec<f64>>,
    filtered_signal: Vec<f32>,
    filtered_ret: Vec<f32>,
    filtered_stock_idx: Vec<u32>,
    group_sums: Vec<f64>,
    group_counts: Vec<usize>,
    sig_buf: Vec<f32>,
    /// rank_both_radix_into 的复用缓冲（跨行复用，热路径零分配）
    rank_keys: Vec<u32>,
    rank_order: Vec<usize>,
    rank_tmp: Vec<usize>,
    rank_ordinal: Vec<i64>,
    rank_avg: Vec<f64>,
}

impl<'a> BtAcc<'a> {
    pub fn new(ctx: BtCtx<'a>) -> Self {
        let n_stocks = ctx.restrict.ncols();
        let t_total = ctx.restrict.nrows();
        let portf_num = ctx.portf_num;
        let ic_only = ctx.ic_only;
        // 新累积器 = 新一次回测：谓词位图缓存里的数组地址可能已失效，整体作废。
        VALID_PRED.with(|cell| cell.borrow_mut().invalidate());
        Self {
            ctx,
            n_stocks,
            t_total,
            local_t: 0,
            prev_row: vec![f32::NAN; n_stocks],
            held_signal: vec![f32::NAN; n_stocks],
            held_restrict_row: 0,
            distinct: std::collections::HashSet::new(),
            distinct_enough: false,
            gen: vec![0u32; n_stocks],
            stamp: vec![0u32; n_stocks],
            walk_buf: Vec::with_capacity(n_stocks),
            gen_id: 0,
            eff_idx: Vec::new(),
            ic_dates: Vec::new(),
            ic_values_f64: Vec::new(),
            ic_values_f32: Vec::new(),
            ratio_values: Vec::new(),
            group_returns: if ic_only { Vec::new() } else { vec![Vec::new(); portf_num] },
            filtered_signal: Vec::with_capacity(n_stocks),
            filtered_ret: Vec::with_capacity(n_stocks),
            filtered_stock_idx: Vec::with_capacity(n_stocks),
            group_sums: vec![0.0; portf_num],
            group_counts: vec![0; portf_num],
            sig_buf: vec![f32::NAN; n_stocks],
            rank_keys: Vec::with_capacity(n_stocks),
            rank_order: Vec::with_capacity(n_stocks),
            rank_tmp: Vec::with_capacity(n_stocks),
            rank_ordinal: Vec::with_capacity(n_stocks),
            rank_avg: Vec::with_capacity(n_stocks),
        }
    }

    #[inline]
    fn row_has_finite(row: &[f32]) -> bool {
        row.iter().any(|v| v.is_finite())
    }

    /// 喂入 slot 的第 [t0, t0+rows) 行（行优先、连续）。
    pub fn push_block(&mut self, block: &ArrayView2<f32>, t0: usize) {
        let rows = block.nrows();
        let n = self.n_stocks;
        let gap = self.ctx.gap;
        let portf_num = self.ctx.portf_num;

        for r in 0..rows {
            let t = t0 + r;
            let row = block.row(r);
            let row = row.as_slice().expect("slot 块必须行优先连续");

            // has_enough_unique_values：生产扫 raw_idx ∈ 0..T-1
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
            if self.ctx.dates[t] <= self.ctx.backtest_start {
                self.prev_row.copy_from_slice(row);
                continue;
            }

            // signal 行 = t-1（生产：effective 判定看 raw_eff_idx-1 行是否有有限值）
            if r > 0 {
                self.sig_buf.copy_from_slice(block.row(r - 1).as_slice().unwrap());
            } else {
                self.sig_buf.copy_from_slice(&self.prev_row);
            }
            let signal_all_nan = !Self::row_has_finite(&self.sig_buf);
            self.prev_row.copy_from_slice(row);
            if signal_all_nan {
                continue;
            }

            // ---- 进入生产的逐日主体 ----
            if self.local_t % gap == 0 {
                self.held_restrict_row = t - 1;
                self.held_signal.copy_from_slice(&self.sig_buf);
            }
            let local_t = self.local_t;

            self.filtered_signal.clear();
            self.filtered_ret.clear();
            self.filtered_stock_idx.clear();
            // 市场有效谓词（ret[t,:] 有限 且 restrict[held_row,:] 有限且 == 0）与面/变体无关：
            // 同一 (gap, 日) 下 raw/neu/各面共享同一张位图，各累积器只再做 signal 有限性
            // 判定 + 收集。逐股全扫的布尔式等价改写，结果逐位相同。
            let held_row = self.held_restrict_row;
            let ret_view = self.ctx.ret.row(t);
            let ret_row = ret_view.as_slice().unwrap_or_else(|| {
                panic!(
                    "ret 行必须行优先连续: shape={:?} strides={:?} is_std={} row_strides={:?} t={}",
                    self.ctx.ret.dim(),
                    self.ctx.ret.strides(),
                    self.ctx.ret.is_standard_layout(),
                    ret_view.strides(),
                    t
                )
            });
            let restrict_view = self.ctx.restrict.row(held_row);
            let restrict_row = restrict_view.as_slice().unwrap_or_else(|| {
                panic!(
                    "restrict 行必须行优先连续: shape={:?} strides={:?} is_std={} held_row={}",
                    self.ctx.restrict.dim(),
                    self.ctx.restrict.strides(),
                    self.ctx.restrict.is_standard_layout(),
                    held_row
                )
            });
            let pred_key = (
                self.ctx.ret.as_ptr() as usize,
                self.ctx.restrict.as_ptr() as usize,
                t,
                held_row,
                n,
            );
            {
                let signal = &self.held_signal;
                let out_signal = &mut self.filtered_signal;
                let out_ret = &mut self.filtered_ret;
                let out_idx = &mut self.filtered_stock_idx;
                VALID_PRED.with(|cell| {
                    let mut pred = cell.borrow_mut();
                    pred.begin_block(t0);
                    if pred.contains(pred_key) {
                        // 命中：同一 (gap, 日) 下 raw/neu/各面已算过谓词 → 只做 signal 判定 + 收集
                        let bits = pred.get(pred_key);
                        for s in 0..n {
                            if bits[s >> 6] & (1u64 << (s & 63)) == 0 {
                                continue;
                            }
                            let signal_value = signal[s];
                            if signal_value.is_finite() {
                                out_signal.push(signal_value);
                                out_ret.push(ret_row[s]);
                                out_idx.push(s as u32);
                            }
                        }
                    } else {
                        // 首次（该 (gap, 日) 的第一个累积器）：逐股全扫，同时把谓词位图记下来
                        // 供后续 raw/neu/各面复用 —— 首次不付出额外一趟扫描。
                        let mut bits = vec![0u64; (n + 63) >> 6];
                        for s in 0..n {
                            let ret_value = ret_row[s];
                            let rv = restrict_row[s];
                            let is_open = rv.is_finite() && rv == 0.0;
                            if ret_value.is_finite() && is_open {
                                bits[s >> 6] |= 1u64 << (s & 63);
                                let signal_value = signal[s];
                                if signal_value.is_finite() {
                                    out_signal.push(signal_value);
                                    out_ret.push(ret_value);
                                    out_idx.push(s as u32);
                                }
                            }
                        }
                        pred.store(pred_key, bits);
                    }
                });
            }

            let mut ranked_this_row = false;
            if (local_t + 1) % gap == 0 {
                // 收益秩：预排序全行 walk 出子集 ordinal 秩
                self.gen_id += 1;
                let gen_id = self.gen_id;
                let orders =
                    if gap == 1 { &self.ctx.pre.orders_g1 } else { &self.ctx.pre.orders_g5 };
                let order = &orders[t];
                for (pos, &stk) in self.filtered_stock_idx.iter().enumerate() {
                    self.gen[stk as usize] = gen_id;
                    self.stamp[stk as usize] = (pos + 1) as u32;
                }
                self.walk_buf.clear();
                self.walk_buf.resize(self.filtered_stock_idx.len(), 0);
                let mut counter = 0usize;
                for &stk in order {
                    if self.gen[stk as usize] == gen_id {
                        self.walk_buf[self.stamp[stk as usize] as usize - 1] = counter as i64;
                        counter += 1;
                    }
                }
                // 复用缓冲：keys/order/tmp/ordinal/avg 挂在 BtAcc 上跨行复用（零分配）。
                rank_both_radix_into(
                    &self.filtered_signal,
                    &mut self.rank_keys,
                    &mut self.rank_order,
                    &mut self.rank_tmp,
                    &mut self.rank_ordinal,
                    &mut self.rank_avg,
                );
                ranked_this_row = true;
                let nf = self.filtered_signal.len() as f64;
                let mut diff_sq_sum = 0.0;
                for idx in 0..self.filtered_signal.len() {
                    let diff = self.walk_buf[idx] - self.rank_ordinal[idx];
                    diff_sq_sum += (diff * diff) as f64;
                }
                let ic_value = if nf < 2.0 {
                    f64::NAN
                } else {
                    1.0 - 6.0 * diff_sq_sum / (nf * (nf * nf - 1.0))
                };
                self.ic_dates.push(self.ctx.dates[t]);
                self.ic_values_f64.push(ic_value);
                self.ic_values_f32.push(ic_value as f32);
            }

            let stocks_num = self.filtered_signal.len();
            if stocks_num < portf_num {
                self.eff_idx.push(t);
                self.ratio_values.push(f64::NAN);
                if !self.ctx.ic_only {
                    for b in 0..portf_num {
                        self.group_returns[b].push(0.0);
                    }
                }
                self.local_t += 1;
                continue;
            }

            let valid_symbol_num = self.ctx.open_symbol_counts.get(t - 1).copied().unwrap_or(0);
            let ratio = if valid_symbol_num > 0 {
                stocks_num as f64 / valid_symbol_num as f64
            } else {
                f64::NAN
            };
            self.ratio_values.push(ratio);
            self.eff_idx.push(t);

            if self.ctx.ic_only {
                self.local_t += 1;
                continue;
            }

            self.group_sums.iter_mut().for_each(|x| *x = 0.0);
            self.group_counts.iter_mut().for_each(|x| *x = 0);
            if !ranked_this_row {
                // 非 gap 日：本行只做一次排序，结果直接用于十分组（与旧实现一致）。
                rank_both_radix_into(
                    &self.filtered_signal,
                    &mut self.rank_keys,
                    &mut self.rank_order,
                    &mut self.rank_tmp,
                    &mut self.rank_ordinal,
                    &mut self.rank_avg,
                );
            }
            for idx in 0..stocks_num {
                let pct = self.rank_avg[idx] / stocks_num as f64;
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
            self.local_t += 1;
        }
    }

    pub fn finish(self) -> LegacyBacktestResult {
        if self.local_t == 0 || !self.distinct_enough {
            return default_legacy_backtest_result();
        }
        let date_size = self.eff_idx.len();
        let gap = self.ctx.gap;
        let portf_num = self.ctx.portf_num;
        let ic_mean = nanmean_f64(&self.ic_values_f64);
        let ic_std = nanstd_population(&self.ic_values_f64);
        let ir = if ic_std.is_nan() || ic_std <= EPS {
            f64::NAN
        } else {
            ic_mean.abs() / ic_std * (250.0 / gap as f64).sqrt()
        };
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
                f64::NAN,
                f64::NAN,
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
                hedge_returns[local_t] = long_ret - self.ctx.index[raw_eff_idx] as f64;
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
                compute_ssm(&self.group_returns, portf_num),
                compute_mprob(&self.group_returns, portf_num),
            ]
        };
        LegacyBacktestResult { summary, ic_dates: self.ic_dates, ic_values: self.ic_values_f32 }
    }
}

/// 自测：把 slot 按块喂给 BtAcc，与生产 `legacy_backtest_gap1_gap5_single_slot_opt` 逐位比较。
pub fn selfcheck(data_dir: &str) -> String {
    use ndarray_npy::read_npy;

    let dates: Vec<i32> = read_npy::<_, Array1<i32>>(format!("{data_dir}/dates.npy"))
        .map(|a| a.to_vec())
        .unwrap_or_default();
    let ret_gap1: Array2<f32> = read_npy(format!("{data_dir}/ret_gap1.npy")).unwrap();
    let ret_sum_gap1: Array2<f32> = read_npy(format!("{data_dir}/ret_sum_gap1.npy")).unwrap();
    let ret_gap5: Array2<f32> = read_npy(format!("{data_dir}/ret_gap5.npy")).unwrap();
    let ret_sum_gap5: Array2<f32> = read_npy(format!("{data_dir}/ret_sum_gap5.npy")).unwrap();
    let restrict: Array2<f32> = read_npy(format!("{data_dir}/restrict.npy")).unwrap();
    let index: Array1<f32> = read_npy(format!("{data_dir}/index_ret.npy")).unwrap();
    let pre = crate::tail_v5_pipeline::build_bt_precomputed(&ret_sum_gap1, &ret_sum_gap5).unwrap();
    let open_counts: Vec<usize> = (0..restrict.nrows())
        .map(|r| restrict.row(r).iter().filter(|&&v| v.is_finite() && v == 0.0).count())
        .collect();
    let backtest_start = 20150201i32;
    let names: Vec<String> = std::fs::read_to_string(format!("{data_dir}/sample_names.txt"))
        .unwrap_or_default()
        .split_whitespace()
        .map(|s| s.to_string())
        .collect();
    let (t, _n) = restrict.dim();
    let mut lines = Vec::new();
    let mut all_pass = true;
    let mut n_cmp = 0usize;
    let mut bt_secs = 0.0f64;
    let mut prod_secs = 0.0f64;
    for nm in names.iter().take(3) {
        let raw: Array2<f32> = read_npy(format!("{data_dir}/factor_{nm}.npy")).unwrap();
        let ranked =
            crate::tail_v5_pipeline::rank_and_fill_missing_cross_sectional_median(&raw, &restrict);
        for &bs in &[64usize, 997, 2818] {
            for &ic_only in &[false, true] {
                let t_prod = std::time::Instant::now();
                let want = crate::tail_v5_pipeline::legacy_backtest_gap1_gap5_single_slot_opt(
                    ranked.view(),
                    ret_gap1.view(),
                    ret_sum_gap1.view(),
                    ret_gap5.view(),
                    ret_sum_gap5.view(),
                    restrict.view(),
                    index.view(),
                    &dates,
                    backtest_start,
                    10,
                    &open_counts,
                    ic_only,
                    &pre,
                );
                prod_secs += t_prod.elapsed().as_secs_f64();
                let mk = |gap: usize| BtCtx {
                    gap,
                    ret: if gap == 1 { &ret_gap1 } else { &ret_gap5 },
                    ret_sum: if gap == 1 { &ret_sum_gap1 } else { &ret_sum_gap5 },
                    restrict: &restrict,
                    index: &index,
                    dates: &dates,
                    backtest_start,
                    portf_num: 10,
                    open_symbol_counts: &open_counts,
                    ic_only,
                    pre: &pre,
                };
                let t_bt = std::time::Instant::now();
                let mut a1 = BtAcc::new(mk(1));
                let mut a5 = BtAcc::new(mk(5));
                let mut t0 = 0usize;
                while t0 < t {
                    let t1 = (t0 + bs).min(t);
                    let blk = ranked.slice(ndarray::s![t0..t1, ..]);
                    a1.push_block(&blk, t0);
                    a5.push_block(&blk, t0);
                    t0 = t1;
                }
                let g1 = a1.finish();
                let g5 = a5.finish();
                bt_secs += t_bt.elapsed().as_secs_f64();
                for (tag, got, wantr) in [("gap1", &g1, &want.0), ("gap5", &g5, &want.1)] {
                    n_cmp += 1;
                    let mut bad = 0usize;
                    // 0..12：含新增的 SSM（下标 10）与 MPROB（下标 11），v8 融合路径与生产 opt 路径必须逐位一致。
                    for i in 0..12 {
                        let (p, q) = (got.summary[i], wantr.summary[i]);
                        let same = (p.is_nan() && q.is_nan()) || p.to_bits() == q.to_bits();
                        if !same {
                            bad += 1;
                        }
                    }
                    if got.ic_dates != wantr.ic_dates {
                        bad += 1;
                    }
                    if got.ic_values.len() != wantr.ic_values.len() {
                        bad += 1;
                    } else {
                        for (p, q) in got.ic_values.iter().zip(wantr.ic_values.iter()) {
                            let same = (p.is_nan() && q.is_nan()) || p.to_bits() == q.to_bits();
                            if !same {
                                bad += 1;
                            }
                        }
                    }
                    if bad > 0 {
                        all_pass = false;
                        lines.push(format!(
                            "  {nm} block={bs} ic_only={ic_only} {tag}: 不一致 {bad} 项（ic 条数 {} vs {}）",
                            got.ic_values.len(),
                            wantr.ic_values.len()
                        ));
                    }
                }
            }
        }
    }
    lines.insert(
        0,
        format!(
            "[tail_v8_backtest::selfcheck] 逐位一致: {}  ({n_cmp} 组比较)",
            if all_pass { "PASS" } else { "FAIL" }
        ),
    );
    // 同数据同轮次的粗略计时（增量路径 vs 生产整表路径），用于量化 P3a 收益。
    lines.push(format!(
        "[tail_v8_backtest::selfcheck] 计时: BtAcc 增量路径 {:.3}s / 生产整表路径 {:.3}s（{n_cmp} 组）",
        bt_secs, prod_secs
    ));
    lines.join("\n")
}
