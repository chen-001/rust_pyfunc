//! preflightopt.rs — 现生产 radix preflight (preflight_quality_check) 的逐位复刻
//! + 优化验证（sandbox 专用，不改生产代码）。
//!
//! 复刻对象: rust_pyfunc/src/tail_v5_pipeline.rs preflight_quality_check
//! 候选优化:
//!   P-A: zero/nan 统计与 majority 排序融合为单日单遍扫描（生产已是单遍, 此处对齐）
//!   P-B: majority 早退——当日"最大重复值"若已确定 ≤ 阈值所需的最小可能，
//!        即 segments 扫描中 max_count 已不可能超过 threshold 时……实际上
//!        majority_count_mean 需要精确均值参与判定, 不能早退。
//!        但：zero_ratio_mean >= zero_max 或 nan_ratio_mean >= nan_max 时因子已被淘汰,
//!        此时 majority 的精确值不影响 passed 判定——不过 majority_count_mean 仍要写入
//!        TailTaskResult.preflight_maj_failed_windows 计数与日志, 语义上不能跳过。
//!   P-C: 收益化简——radix 4-pass 排序按 to_bits 分组计数。相同 bits 必相邻。
//!        优化: 用 2-pass (16-bit) radix 减少排序趟数? u32 需 4 pass (8-bit)。
//!        16-bit 需 2 pass 但 count 表 65536 项, L2 友好性差, 实测。
//!   P-D: 行内并行 scan: restrict==0 判定提前计算位图 (全 run 一次), 每日直接
//!        读 bit 而非 f32 比较两次。restrict 是共享输入, 位图 T×N/8 ≈ 1.5MB。
//!        每日过滤时少读 49MB 的 restrict f32 平面 → 少一次全行比较。
//!        注意: 语义完全相同 (is_finite && ==0.0 ⇔ bit)。

use ndarray::{ArrayView2, Array2};

pub struct PreflightReport {
    pub passed: bool,
    pub majority_count_mean: f64,
    pub zero_ratio_mean: f64,
    pub nan_ratio_mean: f64,
}

fn radix_sort_u32_keys(keys: &[u32], order: &mut Vec<usize>, tmp: &mut Vec<usize>) {
    let n = order.len();
    if n < 2 {
        return;
    }
    tmp.clear();
    tmp.resize(n, 0);
    let mut count = [0usize; 256];
    for shift in (0..32).step_by(8) {
        count.fill(0);
        for &i in order.iter() {
            count[((keys[i] >> shift) & 0xff) as usize] += 1;
        }
        let mut acc = 0usize;
        for c in count.iter_mut() {
            let t = *c;
            *c = acc;
            acc += t;
        }
        for &i in order.iter() {
            let b = ((keys[i] >> shift) & 0xff) as usize;
            tmp[count[b]] = i;
            count[b] += 1;
        }
        std::mem::swap(order, tmp);
    }
}

/// 生产复刻（照抄 preflight_quality_check）。
pub fn preflight_prod(
    raw_values: &ArrayView2<f32>,
    restrict: &ArrayView2<f32>,
    majority_count_threshold: f64,
    zero_max_threshold: f64,
    nan_max_threshold: f64,
) -> PreflightReport {
    let n_dates = raw_values.shape()[0];
    let n_stocks = raw_values.shape()[1];
    let mut majority_sum: f64 = 0.0;
    let mut nan_ratio_sum: f64 = 0.0;
    let mut zero_ratio_sum: f64 = 0.0;
    let mut valid_date_count: usize = 0;

    let count_majority = majority_count_threshold < n_stocks as f64;
    if !count_majority {
        unimplemented!();
    }

    let mut keys: Vec<u32> = Vec::with_capacity(n_stocks);
    let mut order: Vec<usize> = Vec::with_capacity(n_stocks);
    let mut tmp: Vec<usize> = Vec::with_capacity(n_stocks);
    for t in 0..n_dates {
        keys.clear();
        let mut free_count: usize = 0;
        let mut nan_count: usize = 0;
        let mut zero_count: usize = 0;
        for s in 0..n_stocks {
            let val = raw_values[[t, s]];
            let is_free = restrict[[t, s]].is_finite() && restrict[[t, s]] == 0.0;
            if is_free {
                free_count += 1;
                if !val.is_finite() {
                    nan_count += 1;
                } else if val == 0.0 {
                    zero_count += 1;
                }
            }
            if val.is_finite() {
                keys.push(val.to_bits());
            }
        }
        let n = keys.len();
        order.clear();
        order.extend(0..n);
        if n >= 2 {
            radix_sort_u32_keys(&keys, &mut order, &mut tmp);
        }
        let mut max_count = 0usize;
        let mut start = 0usize;
        while start < n {
            let key = keys[order[start]];
            let mut end = start + 1;
            while end < n && keys[order[end]] == key {
                end += 1;
            }
            let c = end - start;
            if c > max_count {
                max_count = c;
            }
            start = end;
        }
        majority_sum += max_count as f64;
        if free_count > 0 {
            nan_ratio_sum += nan_count as f64 / free_count as f64;
            zero_ratio_sum += zero_count as f64 / free_count as f64;
            valid_date_count += 1;
        }
    }

    let majority_count_mean = if n_dates > 0 {
        majority_sum / n_dates as f64
    } else {
        0.0
    };
    let nan_ratio_mean = if valid_date_count > 0 {
        nan_ratio_sum / valid_date_count as f64
    } else {
        0.0
    };
    let zero_ratio_mean = if valid_date_count > 0 {
        zero_ratio_sum / valid_date_count as f64
    } else {
        0.0
    };

    PreflightReport {
        passed: majority_count_mean <= majority_count_threshold
            && zero_ratio_mean < zero_max_threshold
            && nan_ratio_mean < nan_max_threshold,
        majority_count_mean,
        zero_ratio_mean,
        nan_ratio_mean,
    }
}

/// P-D 优化: restrict 位图预计算（全 run 一次, 跨 factor/slot 摊销）。
pub struct RestrictBitmap {
    /// 每 (t, s) 1 bit: 1 = free (restrict 有限且 ==0)
    pub bits: Vec<u64>,
    pub n_stocks: usize,
    pub words_per_row: usize,
}

pub fn build_restrict_bitmap(restrict: &ArrayView2<f32>) -> RestrictBitmap {
    let (t, n) = restrict.dim();
    let words_per_row = n.div_ceil(64);
    let mut bits = vec![0u64; t * words_per_row];
    for ti in 0..t {
        let row = restrict.row(ti);
        let row = row.as_slice().unwrap();
        let base = ti * words_per_row;
        for s in 0..n {
            let v = row[s];
            if v.is_finite() && v == 0.0 {
                bits[base + s / 64] |= 1u64 << (s % 64);
            }
        }
    }
    RestrictBitmap {
        bits,
        n_stocks: n,
        words_per_row,
    }
}

/// P-D 版 preflight: 过滤判定读位图; 其余逻辑逐位一致。
pub fn preflight_bitmap(
    raw_values: &ArrayView2<f32>,
    bitmap: &RestrictBitmap,
    majority_count_threshold: f64,
    zero_max_threshold: f64,
    nan_max_threshold: f64,
) -> PreflightReport {
    let n_dates = raw_values.shape()[0];
    let n_stocks = raw_values.shape()[1];
    let mut majority_sum: f64 = 0.0;
    let mut nan_ratio_sum: f64 = 0.0;
    let mut zero_ratio_sum: f64 = 0.0;
    let mut valid_date_count: usize = 0;

    let mut keys: Vec<u32> = Vec::with_capacity(n_stocks);
    let mut order: Vec<usize> = Vec::with_capacity(n_stocks);
    let mut tmp: Vec<usize> = Vec::with_capacity(n_stocks);
    for t in 0..n_dates {
        keys.clear();
        let mut free_count: usize = 0;
        let mut nan_count: usize = 0;
        let mut zero_count: usize = 0;
        let row = raw_values.row(t);
        let row = row.as_slice().unwrap();
        let bits_row = &bitmap.bits[t * bitmap.words_per_row..(t + 1) * bitmap.words_per_row];
        for s in 0..n_stocks {
            let val = row[s];
            let is_free = (bits_row[s / 64] >> (s % 64)) & 1 == 1;
            if is_free {
                free_count += 1;
                if !val.is_finite() {
                    nan_count += 1;
                } else if val == 0.0 {
                    zero_count += 1;
                }
            }
            if val.is_finite() {
                keys.push(val.to_bits());
            }
        }
        let n = keys.len();
        order.clear();
        order.extend(0..n);
        if n >= 2 {
            radix_sort_u32_keys(&keys, &mut order, &mut tmp);
        }
        let mut max_count = 0usize;
        let mut start = 0usize;
        while start < n {
            let key = keys[order[start]];
            let mut end = start + 1;
            while end < n && keys[order[end]] == key {
                end += 1;
            }
            let c = end - start;
            if c > max_count {
                max_count = c;
            }
            start = end;
        }
        majority_sum += max_count as f64;
        if free_count > 0 {
            nan_ratio_sum += nan_count as f64 / free_count as f64;
            zero_ratio_sum += zero_count as f64 / free_count as f64;
            valid_date_count += 1;
        }
    }

    let majority_count_mean = if n_dates > 0 {
        majority_sum / n_dates as f64
    } else {
        0.0
    };
    let nan_ratio_mean = if valid_date_count > 0 {
        nan_ratio_sum / valid_date_count as f64
    } else {
        0.0
    };
    let zero_ratio_mean = if valid_date_count > 0 {
        zero_ratio_sum / valid_date_count as f64
    } else {
        0.0
    };

    PreflightReport {
        passed: majority_count_mean <= majority_count_threshold
            && zero_ratio_mean < zero_max_threshold
            && nan_ratio_mean < nan_max_threshold,
        majority_count_mean,
        zero_ratio_mean,
        nan_ratio_mean,
    }
}

/// P-C 优化: 计数排序替代 radix——bits 为 u32, 但横截面 rank 值域集中。
/// 思路不可移植 (因子值任意), 改为: majority 只需"最大重复计数"。
/// 若 zero 是众数 (常见: 淘汰因子多因 zero_ratio 或 majority=同一常数),
/// 先单遍统计 zero 计数 z 与 nan; 若 z > threshold 则当日 max_count=z 已定,
/// 跳过排序。否则仍需排序求精确 max_count (均值参与判定, 不能省)。
/// —— 该优化只在 z>threshold 的日子生效, 对"健康因子"无收益, 对"病态因子"
/// (恰是最常被 preflight 淘汰的) 收益大。生产 preflight 对全部 slot 跑,
/// 病态 slot 占比可观 (见 bench: 26 slot 中 preflight 淘汰 ~半数)。
pub fn preflight_zerofast(
    raw_values: &ArrayView2<f32>,
    restrict: &ArrayView2<f32>,
    majority_count_threshold: f64,
    zero_max_threshold: f64,
    nan_max_threshold: f64,
) -> PreflightReport {
    let n_dates = raw_values.shape()[0];
    let n_stocks = raw_values.shape()[1];
    let mut majority_sum: f64 = 0.0;
    let mut nan_ratio_sum: f64 = 0.0;
    let mut zero_ratio_sum: f64 = 0.0;
    let mut valid_date_count: usize = 0;
    let thr = majority_count_threshold as usize;

    let mut keys: Vec<u32> = Vec::with_capacity(n_stocks);
    let mut order: Vec<usize> = Vec::with_capacity(n_stocks);
    let mut tmp: Vec<usize> = Vec::with_capacity(n_stocks);
    for t in 0..n_dates {
        keys.clear();
        let mut free_count: usize = 0;
        let mut nan_count: usize = 0;
        let mut zero_count: usize = 0;
        let row = raw_values.row(t);
        let row = row.as_slice().unwrap();
        let rrow = restrict.row(t);
        let rrow = rrow.as_slice().unwrap();
        for s in 0..n_stocks {
            let val = row[s];
            let is_free = rrow[s].is_finite() && rrow[s] == 0.0;
            if is_free {
                free_count += 1;
                if !val.is_finite() {
                    nan_count += 1;
                } else if val == 0.0 {
                    zero_count += 1;
                }
            }
            if val.is_finite() {
                keys.push(val.to_bits());
            }
        }
        // 当日 majority 精确值: zero 组大小是 max_count 的下界;
        // 若 zero_count 已 > threshold, 精确值对判定无影响——但 majority_count_mean
        // 要精确写入统计, 因此仍需排序…… 除非连统计也可以证明不变:
        // max_count >= zero_count 恒成立; 当 zero_count > threshold 时,
        // 该日 precise max_count 仅用于 majority_sum (均值), 影响 majority_count_mean
        // 的精确值 —— 该值参与 passed 判定 AND preflight_maj_failed_windows 计数,
        // 二者在 majority_count_mean > threshold 时行为一致…… 但 mean 是逐日精确值
        // 平均, 跳过排序会改变 mean 的数值 (当日取 zero_count 而非真值) →
        // 当 mean 恰好落在 threshold 两侧边界时会改变判定。
        // 保守做法: zero_count > threshold 的日子 max_count 一定 > threshold,
        // 若"所有日子的 max_count 下界均值"已 > threshold, 则整体必淘汰,
        // 此时 precise 值不再影响 passed → 仍需两遍? 过于复杂。
        // 简化: 只对 zero_count > thr 的日子跳过排序, max_count 取 zero_count。
        // 语义偏差: majority_count_mean 在这些日子取了下界 → mean 偏小。
        // 因此该变体 NOT bitwise, 仅作上界收益测量。
        let n = keys.len();
        let max_count = if zero_count > thr {
            zero_count
        } else {
            order.clear();
            order.extend(0..n);
            if n >= 2 {
                radix_sort_u32_keys(&keys, &mut order, &mut tmp);
            }
            let mut mc = 0usize;
            let mut start = 0usize;
            while start < n {
                let key = keys[order[start]];
                let mut end = start + 1;
                while end < n && keys[order[end]] == key {
                    end += 1;
                }
                let c = end - start;
                if c > mc {
                    mc = c;
                }
                start = end;
            }
            mc
        };
        majority_sum += max_count as f64;
        if free_count > 0 {
            nan_ratio_sum += nan_count as f64 / free_count as f64;
            zero_ratio_sum += zero_count as f64 / free_count as f64;
            valid_date_count += 1;
        }
    }

    let majority_count_mean = if n_dates > 0 {
        majority_sum / n_dates as f64
    } else {
        0.0
    };
    let nan_ratio_mean = if valid_date_count > 0 {
        nan_ratio_sum / valid_date_count as f64
    } else {
        0.0
    };
    let zero_ratio_mean = if valid_date_count > 0 {
        zero_ratio_sum / valid_date_count as f64
    } else {
        0.0
    };

    PreflightReport {
        passed: majority_count_mean <= majority_count_threshold
            && zero_ratio_mean < zero_max_threshold
            && nan_ratio_mean < nan_max_threshold,
        majority_count_mean,
        zero_ratio_mean,
        nan_ratio_mean,
    }
}
