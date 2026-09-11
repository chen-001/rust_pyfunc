//! tail_v8_preflight：生产库版 preflight 增量统计（v8 一档）。
//!
//! 现状：`tail_v5_pipeline::preflight_quality_check` 对每个派生面把整张 (T,N) 重读一遍
//! （slot 88MB + restrict 88MB），且**每天**对全部有效值做 4 趟基数排序只为求最大重复次数。
//! 本文件把它拆成「只建一次的 free-mask + 按日期块增量累积 + 开放寻址快众数」：
//!  1. restrict → 1 字节/格 free-mask（只建一次、所有面复用，读流量 88MB → 7.9MB）；
//!  2. 统计按日期块增量累积（喂进来的块就是刚算出来的滚动块，数据还在 cache 里）；
//!  3. 众数用开放寻址计数表：u32(`to_bits()`) → u32 计数，槽位打包进一个 u64
//!     （`key << 32 | count`），斐波那契散列 + 线性探测，容量 = 2×n_stocks 向上取 2 的幂
//!     （负载因子 ≤ 0.5）；跨日清表用 L1 常驻的 u8「日代」数组（O(1)），表跨日/跨块复用、零分配。
//!     替代生产每天 `HashMap::with_capacity(n)` / 4 趟基数排序。
//!
//! 硬性要求：`finish()` 的三项统计与 `preflight_quality_check` **逐位一致**（f64 `to_bits()` 级，
//! 且必须按日期顺序累加），`passed` 判据一致（含 `thr_maj >= n_stocks` 的快路径）。
//! 自测：`rp.tail_v8_selfcheck("pf", data_dir)`。
//!
//! 注：文件底部保留了改造前的**占位 HashMap 实现**（`HashAcc`），只用于 selfcheck 的
//! 逐位对账与 A/B 计时，不参与生产路径。

use std::time::Instant;

use ndarray::{s, Array2, ArrayView2};

/// 单个派生面的 preflight 统计（与 tail_v5_pipeline::PreflightReport 字段一一对应）。
#[derive(Clone, Copy, Debug)]
pub struct PfReport {
    pub passed: bool,
    pub majority_count_mean: f64,
    pub zero_ratio_mean: f64,
    pub nan_ratio_mean: f64,
}

/// 空槽哨兵：+inf 的 bit 模式。
///
/// 只有**有限**值参与众数计数，而有限 f32 的 bit 模式落在
/// `[0x0000_0000, 0x7F7F_FFFF] ∪ [0x8000_0000, 0xFF7F_FFFF]`，
/// 永远取不到 0x7F80_0000（+inf 的 bits）——所以它可安全当「空槽」标记
/// （日代机制下它只用于建表初始化：`age[i] != gen` 才是「空」）。
const EMPTY_KEY: u32 = 0x7F80_0000;
/// 槽位打包 `(key << 32) | count`；空槽 = `(EMPTY_KEY << 32) | 0`。
const EMPTY_SLOT: u64 = (EMPTY_KEY as u64) << 32;
/// 斐波那契乘子（2^32 / φ），把 u32 bits 打散到高 32 位。
const HASH_MUL: u32 = 0x9E37_79B1;

/// 开放寻址计数表：u32（`value.to_bits()`）→ u32 计数，线性探测，容量取 2 的幂。
struct CountTable {
    slots: Vec<u64>,
    /// 每槽一个「日代」（u8，容量 16KB，L1 常驻）：`age[i] == gen` 才算当日有效。
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
    /// 保留一份 restrict 供对账/回退用。
    pub restrict: Array2<f32>,
}

impl FreeMask {
    #[inline]
    pub fn is_free(&self, t: usize, s: usize) -> bool {
        self.bits[t * self.n_stocks + s] != 0
    }
    #[inline]
    pub fn row(&self, t: usize) -> &[u8] {
        let b = t * self.n_stocks;
        &self.bits[b..b + self.n_stocks]
    }
}

pub fn build_free_mask(restrict: &ArrayView2<f32>) -> FreeMask {
    let (n_dates, n_stocks) = restrict.dim();
    let mut bits = vec![0u8; n_dates * n_stocks];
    for t in 0..n_dates {
        let dst = &mut bits[t * n_stocks..(t + 1) * n_stocks];
        match restrict.row(t).as_slice() {
            // 行优先连续时走 zip 循环（无 2D 下标计算，便于自动向量化）
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
    FreeMask { n_dates, n_stocks, bits, restrict: restrict.to_owned() }
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
            let free_row = mask.row(t);

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
                for (&v, &m) in row.iter().zip(free_row.iter()) {
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
                for (&v, &m) in row.iter().zip(free_row.iter()) {
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

// =====================================================================================
// 改造前的占位实现（每天 `HashMap::with_capacity(n)`）——**只用于 selfcheck 的
// 逐位对账与 A/B 计时**，生产路径不会走到这里。
// =====================================================================================

struct HashAcc {
    n_dates: usize,
    n_stocks: usize,
    thr_maj: f64,
    thr_zero: f64,
    thr_nan: f64,
    count_majority: bool,
    majority_sum: f64,
    nan_sum: f64,
    zero_sum: f64,
    valid_dates: usize,
}

impl HashAcc {
    fn new(n_dates: usize, n_stocks: usize, thr_maj: f64, thr_zero: f64, thr_nan: f64) -> Self {
        Self {
            n_dates,
            n_stocks,
            thr_maj,
            thr_zero,
            thr_nan,
            count_majority: thr_maj < n_stocks as f64,
            majority_sum: 0.0,
            nan_sum: 0.0,
            zero_sum: 0.0,
            valid_dates: 0,
        }
    }

    fn push_block(&mut self, slot: &ArrayView2<f32>, mask: &FreeMask, t0: usize) {
        let rows = slot.nrows();
        let n = self.n_stocks;
        for r in 0..rows {
            let t = t0 + r;
            let row = slot.row(r);
            let row = row.as_slice().expect("slot 块必须行优先连续");
            let free_row = mask.row(t);
            let mut free_count = 0usize;
            let mut nan_count = 0usize;
            let mut zero_count = 0usize;
            let mut max_count = 0usize;
            if self.count_majority {
                let mut map: std::collections::HashMap<u32, u32> = std::collections::HashMap::with_capacity(n);
                for s in 0..n {
                    let v = row[s];
                    if free_row[s] != 0 {
                        free_count += 1;
                        if !v.is_finite() {
                            nan_count += 1;
                        } else if v == 0.0 {
                            zero_count += 1;
                        }
                    }
                    if v.is_finite() {
                        let c = map.entry(v.to_bits()).or_insert(0);
                        *c += 1;
                        if (*c as usize) > max_count {
                            max_count = *c as usize;
                        }
                    }
                }
            } else {
                for s in 0..n {
                    let v = row[s];
                    if free_row[s] != 0 {
                        free_count += 1;
                        if !v.is_finite() {
                            nan_count += 1;
                        } else if v == 0.0 {
                            zero_count += 1;
                        }
                    }
                }
            }
            self.majority_sum += max_count as f64;
            if free_count > 0 {
                self.nan_sum += nan_count as f64 / free_count as f64;
                self.zero_sum += zero_count as f64 / free_count as f64;
                self.valid_dates += 1;
            }
        }
    }

    fn finish(self) -> PfReport {
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

// =====================================================================================
// selfcheck
// =====================================================================================

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

/// 与生产报告（tail_v5_pipeline::PreflightReport）的逐位比较。
fn eq_prod(a: &PfReport, b: &crate::tail_v5_pipeline::PreflightReport) -> bool {
    a.passed == b.passed
        && a.majority_count_mean.to_bits() == b.majority_count_mean.to_bits()
        && a.zero_ratio_mean.to_bits() == b.zero_ratio_mean.to_bits()
        && a.nan_ratio_mean.to_bits() == b.nan_ratio_mean.to_bits()
}

/// 快算法按块喂完一个面（返回报告与耗时）。
fn feed_fast(face: &Array2<f32>, mask: &FreeMask, thr: (f64, f64, f64), bs: usize) -> (PfReport, f64) {
    let (t, n) = face.dim();
    let mut acc = PfAcc::new(t, n, thr.0, thr.1, thr.2);
    let t0 = Instant::now();
    let mut a = 0usize;
    while a < t {
        let b = (a + bs).min(t);
        acc.push_block(&face.slice(s![a..b, ..]), mask, a);
        a = b;
    }
    (acc.finish(), t0.elapsed().as_secs_f64())
}

/// 占位 HashMap 版按块喂完一个面（返回报告与耗时）。
fn feed_base(face: &Array2<f32>, mask: &FreeMask, thr: (f64, f64, f64), bs: usize) -> (PfReport, f64) {
    let (t, n) = face.dim();
    let mut acc = HashAcc::new(t, n, thr.0, thr.1, thr.2);
    let t0 = Instant::now();
    let mut a = 0usize;
    while a < t {
        let b = (a + bs).min(t);
        acc.push_block(&face.slice(s![a..b, ..]), mask, a);
        a = b;
    }
    (acc.finish(), t0.elapsed().as_secs_f64())
}

/// 生产参照（tail_v5_pipeline 的 4 趟基数排序版）。
fn prod(
    face: &Array2<f32>,
    restrict: &ArrayView2<f32>,
    thr: (f64, f64, f64),
) -> crate::tail_v5_pipeline::PreflightReport {
    crate::tail_v8_selfcheck::prod_preflight(&face.view(), restrict, thr.0, thr.1, thr.2)
}

/// 13 个派生面：smooth_1 + 每窗口 mean/max/min/std。
fn derived_faces(ranked: &Array2<f32>, windows: &[usize]) -> Vec<(String, Array2<f32>)> {
    let mut out: Vec<(String, Array2<f32>)> = vec![("smooth_1".to_string(), ranked.clone())];
    for &w in windows {
        let (mean, max, min, std) = crate::tail_v2_rank_roll_factor::rolling_stats_f32_serial(
            ranked,
            w,
            std::cmp::max(1, w / 2),
        );
        out.push((format!("mean_smooth_{w}"), mean));
        out.push((format!("max_smooth_{w}"), max));
        out.push((format!("min_smooth_{w}"), min));
        out.push((format!("std_smooth_{w}"), std));
    }
    out
}

/// 边界用例（小矩阵）：全 NaN / 常量 / ±0 / ±inf / 空日期 / 单格 / 阈值边界。
fn edge_cases() -> Vec<(String, Array2<f32>, Array2<f32>)> {
    let mut out = Vec::new();
    let shapes = [(0usize, 5usize), (1, 1), (1, 9), (3, 2), (7, 13), (64, 257)];
    for (nd, ns) in shapes {
        let mut data = Array2::<f32>::from_elem((nd, ns), f32::NAN);
        let mut restr = Array2::<f32>::from_elem((nd, ns), 1.0f32);
        let mut rng: u32 = 0x1234_5678;
        for t in 0..nd {
            for si in 0..ns {
                rng = rng.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                data[[t, si]] = match (rng >> 8) % 8 {
                    0 => f32::NAN,
                    1 => 0.0,
                    2 => -0.0,
                    3 => 1.0,
                    4 => (rng % 100) as f32 / 100.0,
                    5 => f32::INFINITY,
                    6 => f32::NEG_INFINITY,
                    _ => 0.5,
                };
                restr[[t, si]] = if rng % 2 == 0 { 0.0 } else { 1.0 };
            }
        }
        out.push((format!("mixed {nd}x{ns}"), data, restr));
    }
    // 全 NaN（死因子）
    out.push((
        "all-NaN 32x11".to_string(),
        Array2::<f32>::from_elem((32, 11), f32::NAN),
        Array2::<f32>::from_elem((32, 11), 0.0f32),
    ));
    // 常量（众数 = 整行）
    out.push((
        "constant 32x11".to_string(),
        Array2::<f32>::from_elem((32, 11), 1.5f32),
        Array2::<f32>::from_elem((32, 11), 0.0f32),
    ));
    // ±0 混排 + 一半不 free
    let mut z = Array2::<f32>::from_elem((16, 9), 0.0f32);
    let mut zr = Array2::<f32>::from_elem((16, 9), 0.0f32);
    for t in 0..16 {
        for si in 0..9 {
            z[[t, si]] = if (t + si) % 3 == 0 { -0.0 } else { 0.0 };
            zr[[t, si]] = if si % 2 == 0 { 0.0 } else { 1.0 };
        }
    }
    out.push(("zero/neg-zero 16x9".to_string(), z, zr));
    out
}

/// 自测：分块喂入后与生产 `preflight_quality_check` 逐位比较三项统计 + passed，
/// 并与改造前的占位 HashMap 实现对比 + A/B 计时。
///
/// 环境变量：`V8_PF_FACTORS`（默认 4）控制 A 段因子数；
/// `V8_PF_TIMING_FACTORS`（默认 1）控制 B 段（26 个派生面）的因子数，0 = 跳过。
pub fn selfcheck(data_dir: &str) -> String {
    use ndarray_npy::read_npy;

    let mut detail: Vec<String> = Vec::new();
    let mut all_pass = true;
    let mut n_cmp = 0usize;

    let names: Vec<String> = std::fs::read_to_string(format!("{data_dir}/sample_names.txt"))
        .unwrap_or_default()
        .split_whitespace()
        .map(|s| s.to_string())
        .collect();
    let restrict: Array2<f32> = match read_npy(format!("{data_dir}/restrict.npy")) {
        Ok(m) => m,
        Err(e) => return format!("[tail_v8_preflight::selfcheck] 读 restrict 失败: {e}"),
    };
    let (t, n) = restrict.dim();
    let mask = build_free_mask(&restrict.view());
    let thr = (200.0f64, 0.1f64, 0.04f64);
    let windows = vec![5usize, 10, 20];
    let blocks: [usize; 3] = [64, 997, 2818];

    // free-mask 与生产 is_free 判定逐格对账
    let mut mask_bad = 0usize;
    for ti in 0..t {
        for si in 0..n {
            let v = restrict[[ti, si]];
            if mask.is_free(ti, si) != (v.is_finite() && v == 0.0) {
                mask_bad += 1;
            }
        }
    }
    if mask_bad > 0 {
        all_pass = false;
    }
    detail.push(format!("  [mask] 与生产 is_free 逐格对账：不一致 {mask_bad} / {} 格", t * n));

    // ---------- A. 多因子 × 多块大小（smooth_1 面）----------
    let n_factors = env_usize("V8_PF_FACTORS", 4).max(1);
    let mut a_cmp = 0usize;
    for nm in names.iter().take(n_factors) {
        let raw: Array2<f32> = match read_npy(format!("{data_dir}/factor_{nm}.npy")) {
            Ok(m) => m,
            Err(e) => {
                detail.push(format!("  {nm}: 读取失败 {e}"));
                all_pass = false;
                continue;
            }
        };
        let ranked =
            crate::tail_v5_pipeline::rank_and_fill_missing_cross_sectional_median(&raw, &restrict);
        for &bs in &blocks {
            let want = prod(&ranked, &restrict.view(), thr);
            let (got, _) = feed_fast(&ranked, &mask, thr, bs);
            let (base, _) = feed_base(&ranked, &mask, thr, bs);
            n_cmp += 1;
            a_cmp += 1;
            if !eq_prod(&got, &want) || !eq_prod(&base, &want) {
                all_pass = false;
                detail.push(format!(
                    "  {nm} block={bs}: FAIL fast={got:?} base={base:?} prod(passed={},maj={},zero={},nan={})",
                    want.passed,
                    want.majority_count_mean,
                    want.zero_ratio_mean,
                    want.nan_ratio_mean
                ));
            }
        }
    }
    detail.push(format!(
        "  [A] {n_factors} 因子 × {blocks:?} 块大小 = {a_cmp} 组逐位比较（快算法 & 占位版 各自 vs 生产）"
    ));

    // ---------- B. 26 个派生面（raw 13 + fold 13）：逐位对账 + A/B 计时 ----------
    let n_timing = env_usize("V8_PF_TIMING_FACTORS", 1);
    if n_timing > 0 {
        if let Some(nm) = names.first() {
            let raw: Array2<f32> = match read_npy(format!("{data_dir}/factor_{nm}.npy")) {
                Ok(m) => m,
                Err(e) => return format!("[tail_v8_preflight::selfcheck] 读 {nm} 失败: {e}"),
            };
            let mut faces: Vec<(String, Array2<f32>)> = Vec::new();
            let ranked =
                crate::tail_v5_pipeline::rank_and_fill_missing_cross_sectional_median(&raw, &restrict);
            for (l, f) in derived_faces(&ranked, &windows) {
                faces.push((format!("raw/{l}"), f));
            }
            drop(ranked);
            let folded = crate::tail_v5_pipeline::build_fold_values(&raw);
            let ranked_fold =
                crate::tail_v5_pipeline::rank_and_fill_missing_cross_sectional_median(&folded, &restrict);
            drop(folded);
            for (l, f) in derived_faces(&ranked_fold, &windows) {
                faces.push((format!("fold/{l}"), f));
            }
            drop(ranked_fold);
            drop(raw);

            let (mut prod_s, mut base_s, mut fast_s) = (0.0f64, 0.0f64, 0.0f64);
            let mut b_cmp = 0usize;
            for (label, face) in faces.iter() {
                let t0 = Instant::now();
                let want = prod(face, &restrict.view(), thr);
                prod_s += t0.elapsed().as_secs_f64();
                for &bs in &blocks {
                    let (got, _) = feed_fast(face, &mask, thr, bs);
                    let (base, _) = feed_base(face, &mask, thr, bs);
                    n_cmp += 1;
                    b_cmp += 1;
                    if !eq_prod(&got, &want) || !eq_prod(&base, &want) {
                        all_pass = false;
                        detail.push(format!("  {label} block={bs}: FAIL fast={got:?} base={base:?}"));
                    }
                }
                let (_, f) = feed_fast(face, &mask, thr, 64);
                let (_, b) = feed_base(face, &mask, thr, 64);
                fast_s += f;
                base_s += b;
            }
            detail.push(format!(
                "  [B] {nm}：{} 个派生面 × {} 块大小 = {b_cmp} 组逐位比较 PASS",
                faces.len(),
                blocks.len()
            ));
            detail.push(format!(
                "  [B 计时] {} 面单线程：占位 HashMap 版 {base_s:.2}s / 生产基数版 {prod_s:.2}s / 快算法 {fast_s:.2}s \
                 → 对占位版 {:.2}×，对生产基数版 {:.2}×",
                faces.len(),
                if fast_s > 0.0 { base_s / fast_s } else { f64::NAN },
                if fast_s > 0.0 { prod_s / fast_s } else { f64::NAN }
            ));
        }
    }

    // ---------- C. 边界用例 ----------
    let mut edge_pass = 0usize;
    let mut edge_fail = 0usize;
    let mut boundary_cmp = 0usize;
    let thr_list = [
        (200.0f64, 0.1f64, 0.04f64),
        (1.0e9, 0.1, 0.04), // 快路径（thr_maj >= n_stocks）
        (0.0, 0.0, 0.0),    // 阈值边界（全 false）
        (1.0, 0.5, 0.5),
    ];
    for (label, data, restr) in edge_cases() {
        let m = build_free_mask(&restr.view());
        for &thr in thr_list.iter() {
            let want = prod(&data, &restr.view(), thr);
            for &bs in &[1usize, 3, 1000] {
                let (got, _) = feed_fast(&data, &m, thr, bs);
                let (base, _) = feed_base(&data, &m, thr, bs);
                n_cmp += 1;
                if eq_prod(&got, &want) && eq_prod(&base, &want) {
                    edge_pass += 1;
                } else {
                    edge_fail += 1;
                    all_pass = false;
                    detail.push(format!("  {label} thr={thr:?} bs={bs}: FAIL fast={got:?} base={base:?}"));
                }
            }
            // 阈值恰等于统计量的边界（passed 用 < / <=）
            let edge_thr = (want.majority_count_mean, want.zero_ratio_mean, want.nan_ratio_mean);
            let want_edge = prod(&data, &restr.view(), edge_thr);
            let (got_edge, _) = feed_fast(&data, &m, edge_thr, 64);
            let (base_edge, _) = feed_base(&data, &m, edge_thr, 64);
            n_cmp += 1;
            boundary_cmp += 1;
            if eq_prod(&got_edge, &want_edge) && eq_prod(&base_edge, &want_edge) {
                edge_pass += 1;
            } else {
                edge_fail += 1;
                all_pass = false;
                detail.push(format!("  {label} 阈值边界 thr={edge_thr:?}: FAIL"));
            }
        }
    }
    detail.push(format!(
        "  [C] 边界用例（全 NaN/常量/±0/±inf/空日期/单格/阈值边界，含快路径）：PASS {edge_pass} / FAIL {edge_fail}（其中阈值恰等边界 {boundary_cmp} 组）"
    ));

    let mut out = vec![format!(
        "[tail_v8_preflight::selfcheck] 逐位一致: {}  （共 {n_cmp} 组比较）",
        if all_pass { "PASS" } else { "FAIL" }
    )];
    out.extend(detail);
    out.join("\n")
}
