//! pf8：一档的 preflight 侧——free-mask + 增量式统计累积器 + 快的众数算法。
//!
//! 现状（生产 `preflight_quality_check`）：每个派生面都把整张 (T,N) 重读一遍
//! （slot 88MB + restrict 88MB），且每天对全部有效值做 4 趟基数排序，只为求
//! 「最大重复次数」；每天还要 `HashMap::with_capacity(n)` 新建一张哈希表。
//!
//! 本文件的三件事：
//!  1. restrict → 1 字节/格 的 free-mask，只建一次，所有面复用（读流量 88MB → 7.9MB）；
//!  2. 统计按日期块增量累积（喂进来的块就是刚算出来的滚动块，数据还在 cache 里）；
//!  3. 众数用**开放寻址计数表**（u32 bits → u32 计数，线性探测，容量 2 的幂，跨日/跨块
//!     复用、零分配；跨日清表用 L1 常驻的「日代」数组，O(1)），替掉基数排序与逐日 HashMap。
//!
//! 硬性要求：`finish()` 的三项统计与 `engine::preflight_quality_check` **逐位一致**
//! （f64 按同一日期顺序累加），`passed` 判据也一致，含
//! `majority_count_threshold >= n_stocks` 时跳过众数的快路径。
//! 自测入口：`src/bin/pf8_check.rs`。

use std::sync::Arc;

use ndarray::{Array2, ArrayView2};

use crate::t8::PfReport;

/// 空槽哨兵：+inf 的 bit 模式。
///
/// 只有**有限**值参与众数计数，而有限 f32 的 bit 模式落在
/// `[0x0000_0000, 0x7F7F_FFFF] ∪ [0x8000_0000, 0xFF7F_FFFF]`，
/// 永远取不到 0x7F80_0000（+inf 的 bits）——所以它可安全当「空槽」标记。
/// （日代机制下它只用于建表初始化：`age[i] != gen` 才是「空」。）
const EMPTY_KEY: u32 = 0x7F80_0000;
/// 槽位打包 `(key << 32) | count`；空槽 = `(EMPTY_KEY << 32) | 0`。
const EMPTY_SLOT: u64 = (EMPTY_KEY as u64) << 32;
/// 斐波那契乘子（2^32 / φ），把 u32 bits 打散到高 32 位。
const HASH_MUL: u32 = 0x9E37_79B1;

/// 开放寻址计数表：u32（`value.to_bits()`）→ u32 计数，线性探测，容量取 2 的幂。
///
/// - 容量 = 2×n_stocks 向上取整到 2 的幂（负载因子 ≤ 0.5），跨日/跨块复用，零分配；
/// - 跨日清表用「日代」`age`（u8/槽，16KB，L1 常驻）：`age[i] == gen` 才算当日有效，
///   gen 每过一天 +1（O(1)），不必逐槽清零；u8 绕满一圈（255 天）才整体清一次 age；
/// - 键就是 `to_bits()`，与生产 `HashMap<u32, usize>` / 基数排序的分组键完全一致。
struct CountTable {
    slots: Vec<u64>,
    age: Vec<u8>,
    gen: u8,
    mask: usize,
    shift: u32,
}

impl CountTable {
    fn new(n_stocks: usize) -> Self {
        let cap = (n_stocks.max(16) * 2).next_power_of_two();
        Self {
            slots: vec![EMPTY_SLOT; cap],
            age: vec![0; cap],
            gen: 0,
            mask: cap - 1,
            shift: 32 - cap.trailing_zeros(),
        }
    }

    /// 进入新的一天：日代 +1（O(1)）；u8 绕回 0 时整体清一次 age。
    #[inline]
    fn begin_day(&mut self) {
        self.gen = self.gen.wrapping_add(1);
        if self.gen == 0 {
            self.age.fill(0);
            self.gen = 1;
        }
    }

    /// 键的起始探测槽（斐波那契散列取高位）。
    #[inline(always)]
    fn hash(&self, key: u32) -> usize {
        (key.wrapping_mul(HASH_MUL) >> self.shift) as usize
    }

    /// 计入一个有限值的 bits，返回它当前的计数。
    #[inline]
    fn bump(&mut self, key: u32) -> u32 {
        let i = self.hash(key);
        self.bump_at(key, i)
    }

    /// 从 `i` 起线性探测，计入一个有限值的 bits，返回它当前的计数。
    ///
    /// 日代不变式：当日插入过的槽位 age == gen 且当日不再变回；插入总是停在
    /// 探测序列上第一个 age != gen 的槽位，因此当日查找不会提前误判「空」。
    #[inline]
    fn bump_at(&mut self, key: u32, mut i: usize) -> u32 {
        let gen = self.gen;
        loop {
            if self.age[i] != gen {
                self.age[i] = gen;
                self.slots[i] = ((key as u64) << 32) | 1;
                return 1;
            }
            let s = self.slots[i];
            if (s >> 32) as u32 == key {
                let c = (s as u32) + 1;
                self.slots[i] = (s & 0xFFFF_FFFF_0000_0000) | (c as u64);
                return c;
            }
            i = (i + 1) & self.mask;
        }
    }
}

/// restrict → 1 字节/格的「可交易」掩码。1 = restrict 有限且 == 0（生产语义的 is_free）。
pub struct FreeMask {
    pub n_dates: usize,
    pub n_stocks: usize,
    pub bits: Vec<u8>,
    /// 保留一份 restrict 供对账/回退用（Arc 共享，不复制）。
    pub restrict: Arc<Array2<f32>>,
}

impl FreeMask {
    #[inline]
    pub fn is_free(&self, t: usize, s: usize) -> bool {
        self.bits[t * self.n_stocks + s] != 0
    }

    /// 第 t 行的掩码切片（push_block 热路径用，避免逐格乘加与越界检查）。
    #[inline]
    pub fn row(&self, t: usize) -> &[u8] {
        &self.bits[t * self.n_stocks..(t + 1) * self.n_stocks]
    }
}

pub fn build_free_mask(restrict: &ArrayView2<f32>) -> FreeMask {
    let (n_dates, n_stocks) = restrict.dim();
    let mut bits = vec![0u8; n_dates * n_stocks];
    for t in 0..n_dates {
        let dst = &mut bits[t * n_stocks..(t + 1) * n_stocks];
        match restrict.row(t).as_slice() {
            // 行优先连续时走 zip 循环，等价于逐格 `v.is_finite() && v == 0.0`，
            // 但无 2D 下标计算，便于自动向量化。
            Some(src) => {
                for (d, &v) in dst.iter_mut().zip(src.iter()) {
                    *d = (v.is_finite() && v == 0.0) as u8;
                }
            }
            None => {
                for s in 0..n_stocks {
                    let v = restrict[[t, s]];
                    dst[s] = (v.is_finite() && v == 0.0) as u8;
                }
            }
        }
    }
    FreeMask {
        n_dates,
        n_stocks,
        bits,
        restrict: Arc::new(restrict.to_owned()),
    }
}

/// 增量式 preflight 累积器：按日期块喂入，`finish()` 得到与生产完全相同的统计。
pub struct PfAcc {
    n_dates: usize,
    n_stocks: usize,
    thr_maj: f64,
    thr_zero: f64,
    thr_nan: f64,
    count_majority: bool,
    /// 众数计数表（跨日/跨块复用，零分配）。
    table: CountTable,
    majority_sum: f64,
    nan_sum: f64,
    zero_sum: f64,
    valid_dates: usize,
}

impl PfAcc {
    pub fn new(n_dates: usize, n_stocks: usize, thr_maj: f64, thr_zero: f64, thr_nan: f64) -> Self {
        Self {
            n_dates,
            n_stocks,
            thr_maj,
            thr_zero,
            thr_nan,
            // 与生产一致：阈值 ≥ 股票数时众数判定恒真，跳过全部计数。
            count_majority: thr_maj < n_stocks as f64,
            table: CountTable::new(n_stocks),
            majority_sum: 0.0,
            nan_sum: 0.0,
            zero_sum: 0.0,
            valid_dates: 0,
        }
    }

    /// 喂入第 [t0, t0+rows) 行（行优先、连续）。
    ///
    /// 逐日顺序累加 f64（与生产 t=0..T 的顺序一致），因此 `finish()` 逐位相同。
    pub fn push_block(&mut self, slot: &ArrayView2<f32>, mask: &FreeMask, t0: usize) {
        let rows = slot.nrows();
        let n = self.n_stocks;
        debug_assert_eq!(slot.ncols(), n);
        debug_assert_eq!(mask.n_stocks, n);
        for r in 0..rows {
            let t = t0 + r;
            debug_assert!(t < self.n_dates);
            let row = slot.row(r);
            let row = row.as_slice().expect("slot 块必须行优先连续");
            let mb = mask.row(t);

            let mut free_count = 0usize;
            let mut nan_count = 0usize;
            let mut zero_count = 0usize;
            let mut finite_count = 0usize;
            let mut max_count = 0usize;

            if self.count_majority {
                // 第 1 趟：free / nan / zero。掩码与判定写成无分支形式，便于自动向量化；
                // 语义与生产 `if is_free { free+=1; if !finite {nan+=1} else if v==0 {zero+=1} }`
                // 完全相同（整数计数，与顺序无关）。顺带数一下有限值个数：整行全 NaN
                // （生产里常见的「死因子」）时第 2 趟可直接跳过。
                for (&v, &m) in row.iter().zip(mb.iter()) {
                    let m = m as usize;
                    let finite = v.is_finite() as usize;
                    let is_zero = (v == 0.0) as usize;
                    free_count += m;
                    nan_count += m & (1 - finite);
                    zero_count += m & finite & is_zero;
                    finite_count += finite;
                }
                // 第 2 趟：众数（开放寻址计数）。只统计有限值，键 = to_bits()，
                // 与生产 `HashMap<u32, usize>` / 基数排序分组完全等价。
                // 直接迭代整行（`row.len() == n` 由上面的 debug_assert 保证）；
                // 实测 `.take(n)` 之类的迭代器适配器会让这个热循环慢 30%。
                self.table.begin_day();
                if finite_count > 0 {
                    for &v in row.iter() {
                        if v.is_finite() {
                            let c = self.table.bump(v.to_bits()) as usize;
                            if c > max_count {
                                max_count = c;
                            }
                        }
                    }
                }
                self.majority_sum += max_count as f64;
            } else {
                // 快路径（thr_maj >= n_stocks）：生产跳过众数统计，
                // majority_count_mean 直接返回 n_stocks（见 finish）。
                for (&v, &m) in row.iter().zip(mb.iter()) {
                    let m = m as usize;
                    let finite = v.is_finite() as usize;
                    let is_zero = (v == 0.0) as usize;
                    free_count += m;
                    nan_count += m & (1 - finite);
                    zero_count += m & finite & is_zero;
                }
            }

            if free_count > 0 {
                self.nan_sum += nan_count as f64 / free_count as f64;
                self.zero_sum += zero_count as f64 / free_count as f64;
                self.valid_dates += 1;
            }
        }
    }

    /// 保守下界早停：即使**剩余日期全部贡献 0** 也仍然不达标时返回 true。
    ///
    /// 依据：`nan_ratio_mean = nan_sum_final / valid_dates_final ≥ nan_sum_now / n_dates`
    /// （分子只增、分母 ≤ n_dates），zero 同理；majority 的均值也 ≥ majority_sum_now / n_dates。
    /// 用于二档融合里提前放弃该面的中性化/回测（preflight 统计仍继续累积，保证对账精确）。
    pub fn definitely_failed(&self) -> bool {
        if self.n_dates == 0 {
            return false;
        }
        let t = self.n_dates as f64;
        if self.nan_sum / t >= self.thr_nan || self.zero_sum / t >= self.thr_zero {
            return true;
        }
        if self.count_majority && self.majority_sum / t > self.thr_maj {
            return true;
        }
        false
    }

    pub fn finish(self) -> PfReport {
        let nan_ratio_mean = if self.valid_dates > 0 {
            self.nan_sum / self.valid_dates as f64
        } else {
            0.0
        };
        let zero_ratio_mean = if self.valid_dates > 0 {
            self.zero_sum / self.valid_dates as f64
        } else {
            0.0
        };
        if !self.count_majority {
            return PfReport {
                passed: zero_ratio_mean < self.thr_zero && nan_ratio_mean < self.thr_nan,
                majority_count_mean: self.n_stocks as f64,
                zero_ratio_mean,
                nan_ratio_mean,
            };
        }
        let majority_count_mean = if self.n_dates > 0 {
            self.majority_sum / self.n_dates as f64
        } else {
            0.0
        };
        PfReport {
            passed: majority_count_mean <= self.thr_maj
                && zero_ratio_mean < self.thr_zero
                && nan_ratio_mean < self.thr_nan,
            majority_count_mean,
            zero_ratio_mean,
            nan_ratio_mean,
        }
    }
}
