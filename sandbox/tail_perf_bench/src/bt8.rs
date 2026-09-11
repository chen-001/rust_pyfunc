//! bt8：回测的「按日期块增量累积」实现（二档消费侧）。
//!
//! 现状：`engine::legacy_backtest_gap1_gap5_single_slot` 需要整张 (T,N) slot；
//! 二档融合流水线只有块。本模块把生产回测的逐日循环拆成「喂块 → 累积 → finish」，
//! 状态只有 IC 序列 + 少量缓存，**不物化 slot 大矩阵**。
//!
//! 硬性要求：`finish()` 与生产逐位一致（summary 10 项、ic_dates、ic_values）。
//! 自检由 `src/bin/bt8_check.rs` 负责：同时对齐
//! `engine::legacy_backtest_gap1_gap5_single_slot`（非 opt）与
//! `btopt::bt_gap1_gap5_prod`（生产 O1 路径）。
//!
//! ## 与生产 O1 路径对齐（收益秩预排序）
//!
//! 生产跑的是 `legacy_backtest_single_factor_with_effective_opt`：
//! `btopt::build_bt_precomputed` 先把 `ret_sum` 每一行的**全市场序**排好
//! （`orders_g1` / `orders_g5`，每 run 构建一次），回测时用
//! 「gen/stamp 标记 + 走一遍全市场序」在 O(N) 内得到**子集内的序数秩**，
//! 不必对 `filtered_future` 再排序。本模块原样移植该走查（对照 `btopt::bt_single_gap`）：
//!
//! * IC 日：`walk_buf` = 走 `pre.orders_g1/g5[raw_eff_idx]` 得到的 filtered_future 序数秩；
//!   `filtered_signal` 仍用**一次**基数排序同时得到序数秩与平均秩（≡ `rank_both_radix`）；
//! * 非 IC 日：十分组只需要平均秩，同样一次基数排序；
//! * 全程零分配：所有中间缓冲都是可复用字段（跨日、跨块、跨 slot 复用）。
//!
//! ## 排序口径（与 engine / btopt 完全同序）
//!
//! f32 → 单调 u32 键（±0 归一、NaN 统一 `u32::MAX`），稳定 LSD 基数排序（3 趟 × 11bit，
//! 2048 桶计数表常驻 L1；键在过滤循环里顺带生成，序数秩在最后一趟 scatter 里顺带写回）；
//! 键序 + 稳定性 ≡ `(值, 原下标)` 字典序 ≡ engine 里
//! `sort_by(partial_cmp.then(index))` 与 btopt 里 `mono_key32 + radix_sort_u32_keys`。
//!
//! 另外两项与逐位无关的热路径优化：块内 signal 行直接借用上一行切片、只在块边界
//! 维护 `prev_row`（原来每行整行 memcpy）；过滤循环按连续切片迭代而非二维下标。

use std::collections::HashSet;
use std::sync::Arc;

use ndarray::{Array2, ArrayView2};

use crate::btopt::BtPrecomputed;
use crate::engine::{
    self, annualized_sharpe_sample, max_drawdown_from_returns, nanmean_f64, nanstd_population,
    LegacyBacktestResult,
};

const EPS: f64 = 1e-12;

// ---------- 排序：与 engine 的 `average_ranks` / `ordinal_ranks` 完全同序 ----------

/// f32 → 单调 u32 排序键（与 btopt::mono_key32 + NaN 处理一致）。
///
/// * 非 NaN：`bits ^ 0x8000_0000`（正数）/ `!bits`（负数）→ 与数值序同序；
/// * ±0 归一成同一个键（engine 用 `partial_cmp`，±0 视为相等，靠下标定序）；
/// * NaN 统一为 `u32::MAX`（NaN 排最后；同键 + 稳定排序 ⇒ 保持原下标序）。
#[inline(always)]
fn sort_key(value: f32) -> u32 {
    if value.is_nan() {
        return u32::MAX;
    }
    sort_key_finite(value)
}

/// `sort_key` 的有限值快路径（过滤集里的 signal 已保证有限，省掉 NaN 分支）。
#[inline(always)]
fn sort_key_finite(value: f32) -> u32 {
    let bits = if value == 0.0 { 0u32 } else { value.to_bits() };
    if bits & 0x8000_0000 != 0 {
        !bits
    } else {
        bits ^ 0x8000_0000
    }
}

/// 基数排序位宽：3 趟 × 11bit（2048 桶，计数表 8KB 常驻 L1）。
/// 桶数只影响速度，不影响结果：LSD + 稳定 ⇒ 任意位宽切分都得到同一个全序。
const RADIX_BITS: u32 = 11;
const RADIX_SIZE: usize = 1 << RADIX_BITS;
const RADIX_MASK: u32 = (RADIX_SIZE as u32) - 1;
const RADIX_PASSES: u32 = 3;

/// 排序用的可复用缓冲（跨日、跨块、跨 slot 复用；热路径零分配）。
struct SortScratch {
    order: Vec<u32>,
    tmp: Vec<u32>,
    counts: [u32; RADIX_SIZE],
}

impl SortScratch {
    fn new() -> Self {
        Self {
            order: Vec::new(),
            tmp: Vec::new(),
            counts: [0; RADIX_SIZE],
        }
    }

    /// 稳定 LSD 基数排序：`keys` 与待排数组同序（下标 i 的键 = keys[i]）。
    /// 结果落在 `self.order`。`ord` 非空时，**最后一趟 scatter 顺带**写下序数秩
    /// （此时写的位置就是最终名次），省掉一趟随机写回。
    fn radix_sort(&mut self, keys: &[u32], mut ord: Option<&mut Vec<u32>>) {
        let n = keys.len();
        self.order.clear();
        self.order.extend(0..n as u32);
        if self.tmp.len() < n {
            self.tmp.resize(n, 0);
        } else {
            self.tmp.truncate(n);
        }
        for pass in 0..RADIX_PASSES {
            let shift = pass * RADIX_BITS;
            let last = pass == RADIX_PASSES - 1;
            self.counts[..].fill(0);
            for &i in self.order.iter() {
                self.counts[((keys[i as usize] >> shift) & RADIX_MASK) as usize] += 1;
            }
            let mut acc = 0u32;
            for c in self.counts.iter_mut() {
                let t = *c;
                *c = acc;
                acc += t;
            }
            if last {
                if let Some(ord) = ord.as_deref_mut() {
                    if ord.len() < n {
                        ord.resize(n, 0);
                    }
                    for &i in self.order.iter() {
                        let b = ((keys[i as usize] >> shift) & RADIX_MASK) as usize;
                        let pos = self.counts[b];
                        self.tmp[pos as usize] = i;
                        ord[i as usize] = pos;
                        self.counts[b] = pos + 1;
                    }
                    std::mem::swap(&mut self.order, &mut self.tmp);
                    continue;
                }
            }
            for &i in self.order.iter() {
                let b = ((keys[i as usize] >> shift) & RADIX_MASK) as usize;
                let pos = self.counts[b];
                self.tmp[pos as usize] = i;
                self.counts[b] = pos + 1;
            }
            std::mem::swap(&mut self.order, &mut self.tmp);
        }
    }

    /// 一次基数排序同时给出序数秩（0-based）与平均秩（1-based，f64）。
    /// ≡ btopt::rank_both_radix ≡ engine 的 ordinal_ranks + average_ranks。
    fn rank_both(
        &mut self,
        values: &[f32],
        keys: &[u32],
        ord: &mut Vec<u32>,
        avg: &mut Vec<f64>,
    ) {
        let n = values.len();
        self.radix_sort(keys, Some(ord));
        if avg.len() < n {
            avg.resize(n, f64::NAN);
        }
        let mut start = 0usize;
        while start < n {
            let value = values[self.order[start] as usize];
            let mut end = start + 1;
            while end < n && values[self.order[end] as usize] == value {
                end += 1;
            }
            let avg_rank = (start + 1 + end) as f64 / 2.0;
            for &i in self.order[start..end].iter() {
                avg[i as usize] = avg_rank;
            }
            start = end;
        }
    }

    /// 只算平均秩（非 IC 日的十分组用），仍是一次基数排序。
    fn average_ranks(&mut self, values: &[f32], keys: &[u32], out: &mut Vec<f64>) {
        let n = values.len();
        out.clear();
        if n == 0 {
            return;
        }
        self.radix_sort(keys, None);
        out.resize(n, f64::NAN);
        let mut start = 0usize;
        while start < n {
            let value = values[self.order[start] as usize];
            let mut end = start + 1;
            while end < n && values[self.order[end] as usize] == value {
                end += 1;
            }
            let avg_rank = (start + 1 + end) as f64 / 2.0;
            for &i in self.order[start..end].iter() {
                out[i as usize] = avg_rank;
            }
            start = end;
        }
    }
}

fn default_result() -> LegacyBacktestResult {
    LegacyBacktestResult {
        summary: [f64::NAN; 10],
        ic_dates: Vec::new(),
        ic_values: Vec::new(),
    }
}

/// 回测共享上下文（Arc 共享，不复制大矩阵）。
pub struct BtCtx {
    pub gap: usize,
    pub shared: Arc<engine::Shared>,
    pub open_symbol_counts: Arc<Vec<usize>>,
    pub ic_only: bool,
    /// 生产 O1 路径的收益秩预排序（`btopt::build_bt_precomputed`，每 run 构建一次）。
    pub pre: Arc<BtPrecomputed>,
}

impl BtCtx {
    #[inline]
    pub fn ret(&self) -> &Array2<f32> {
        if self.gap == 1 { &self.shared.ret_gap1 } else { &self.shared.ret_gap5 }
    }
}

/// 增量式回测累积器（一个 gap 一个）。
pub struct BtAcc {
    ctx: BtCtx,
    n_stocks: usize,
    t_total: usize,
    local_t: usize,
    /// 上一块最后一行的 slot 值（块内 r>0 直接用块内前一行，故只在块边界更新）
    prev_row: Vec<f32>,
    /// 缓存当前 held signal 行（生产里每行现取 factor[held_idx, :]）
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
    filtered_stock_idx: Vec<u32>,
    /// 与 filtered_signal 同序的排序键（在过滤循环里顺带生成，省一趟键扫描）
    filtered_keys: Vec<u32>,
    group_sums: Vec<f64>,
    group_counts: Vec<usize>,
    sc: SortScratch,
    ord_buf: Vec<u32>,
    ranks_buf: Vec<f64>,
    /// 走 precomputed order 用的「本日过滤集」标记（gen_id 递增，免清零）
    gen: Vec<u32>,
    stamp: Vec<u32>,
    walk_buf: Vec<u32>,
    gen_id: u32,
}

impl BtAcc {
    pub fn new(ctx: BtCtx) -> Self {
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
            eff_idx: Vec::with_capacity(t_total),
            ic_dates: Vec::with_capacity(t_total),
            ic_values_f64: Vec::with_capacity(t_total),
            ic_values_f32: Vec::with_capacity(t_total),
            ratio_values: Vec::with_capacity(t_total),
            group_returns: if ic_only {
                Vec::new()
            } else {
                (0..portf_num).map(|_| Vec::with_capacity(t_total)).collect()
            },
            filtered_signal: Vec::with_capacity(n_stocks),
            filtered_ret: Vec::with_capacity(n_stocks),
            filtered_stock_idx: Vec::with_capacity(n_stocks),
            filtered_keys: Vec::with_capacity(n_stocks),
            group_sums: vec![0.0; portf_num],
            group_counts: vec![0; portf_num],
            sc: SortScratch::new(),
            ord_buf: Vec::with_capacity(n_stocks),
            ranks_buf: Vec::with_capacity(n_stocks),
            gen: vec![0u32; n_stocks],
            stamp: vec![0u32; n_stocks],
            walk_buf: Vec::with_capacity(n_stocks),
            gen_id: 0,
        }
    }

    #[inline]
    fn row_has_finite(row: &[f32]) -> bool {
        row.iter().any(|v| v.is_finite())
    }

    /// 喂入 slot 的第 [t0, t0+rows) 行（块必须行优先连续，且与上一块首尾相接）。
    pub fn push_block(&mut self, block: &ArrayView2<f32>, t0: usize) {
        let rows = block.nrows();
        if rows == 0 {
            return;
        }
        let n = self.n_stocks;
        let gap = self.ctx.gap;
        let t_total = self.t_total;
        let backtest_start = self.ctx.shared.backtest_start;
        let portf_num = 10usize;
        let data = block.as_slice().expect("slot 块必须行优先连续");

        for r in 0..rows {
            let t = t0 + r;
            let row = &data[r * n..(r + 1) * n];

            // has_enough_unique_values：生产扫 raw_idx ∈ 0..T-1 的有限值，够 10 个即短路
            if !self.distinct_enough && t + 1 < t_total {
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

            // 生产 effective_raw_indices 只收 raw_eff_idx ∈ 1..T 且 dates > backtest_start
            if t == 0 || self.ctx.shared.dates[t] <= backtest_start {
                continue;
            }

            // signal 行 = t-1：块内直接借用前一行，块首用上一块最后一行
            let sig: &[f32] = if r > 0 { &data[(r - 1) * n..r * n] } else { &self.prev_row };
            if !Self::row_has_finite(sig) {
                continue;
            }

            // 有效日：进入生产的逐日主体
            if self.local_t % gap == 0 {
                self.held_restrict_row = t - 1;
                self.held_signal.copy_from_slice(sig);
            }
            let local_t = self.local_t;
            let ic_day = (local_t + 1) % gap == 0;

            let ret_view = self.ctx.ret().row(t);
            let ret_row = ret_view.as_slice().unwrap();
            let restrict_view = self.ctx.shared.restrict.row(self.held_restrict_row);
            let restrict_row = restrict_view.as_slice().unwrap();

            self.filtered_signal.clear();
            self.filtered_ret.clear();
            self.filtered_stock_idx.clear();
            self.filtered_keys.clear();
            let mut s = 0u32;
            for ((&signal_value, &rv), &ret_value) in self
                .held_signal
                .iter()
                .zip(restrict_row.iter())
                .zip(ret_row.iter())
            {
                if signal_value.is_finite() && ret_value.is_finite() && rv.is_finite() && rv == 0.0
                {
                    self.filtered_signal.push(signal_value);
                    self.filtered_ret.push(ret_value);
                    self.filtered_stock_idx.push(s);
                    // signal 已保证有限 → 走 sort_key 的快路径
                    self.filtered_keys.push(sort_key_finite(signal_value));
                }
                s += 1;
            }
            let stocks_num = self.filtered_signal.len();

            // ---- IC 日：走 precomputed order（生产 O1），不再排序 filtered_future ----
            let mut have_avg = false;
            if ic_day {
                self.gen_id += 1;
                let gid = self.gen_id;
                for (pos, &stk) in self.filtered_stock_idx.iter().enumerate() {
                    let k = stk as usize;
                    self.gen[k] = gid;
                    self.stamp[k] = (pos + 1) as u32;
                }
                let order = if gap == 1 {
                    &self.ctx.pre.orders_g1[t]
                } else {
                    &self.ctx.pre.orders_g5[t]
                };
                self.walk_buf.clear();
                self.walk_buf.resize(stocks_num, 0);
                let mut counter = 0u32;
                for &stk in order.iter() {
                    let k = stk as usize;
                    if self.gen[k] == gid {
                        self.walk_buf[self.stamp[k] as usize - 1] = counter;
                        counter += 1;
                    }
                }
                self.sc.rank_both(
                    &self.filtered_signal,
                    &self.filtered_keys,
                    &mut self.ord_buf,
                    &mut self.ranks_buf,
                );
                have_avg = true;

                let mut diff_sq_sum = 0.0f64;
                for idx in 0..stocks_num {
                    let diff = self.walk_buf[idx] as i64 - self.ord_buf[idx] as i64;
                    diff_sq_sum += (diff * diff) as f64;
                }
                let nf = stocks_num as f64;
                let ic_value = if nf < 2.0 {
                    f64::NAN
                } else {
                    1.0 - 6.0 * diff_sq_sum / (nf * (nf * nf - 1.0))
                };
                self.ic_dates.push(self.ctx.shared.dates[t]);
                self.ic_values_f64.push(ic_value);
                self.ic_values_f32.push(ic_value as f32);
            }

            if stocks_num < portf_num {
                // 生产：continue，group_returns[*][local_t] 保持 0.0；ratio_values[local_t] 保持 NaN
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

            if !have_avg {
                self.sc.average_ranks(
                    &self.filtered_signal,
                    &self.filtered_keys,
                    &mut self.ranks_buf,
                );
            }
            self.group_sums.iter_mut().for_each(|x| *x = 0.0);
            self.group_counts.iter_mut().for_each(|x| *x = 0);
            for idx in 0..stocks_num {
                let pct = self.ranks_buf[idx] / stocks_num as f64;
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

        // 块边界：只需要保留本块最后一行作为下一块块首的 signal 行
        self.prev_row.copy_from_slice(&data[(rows - 1) * n..rows * n]);
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
                hedge_returns[local_t] = long_ret - self.ctx.shared.index_ret[raw_eff_idx] as f64;
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
