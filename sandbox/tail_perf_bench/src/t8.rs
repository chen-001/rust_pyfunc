//! t8：三档改造的共享类型与「冻结接口」约定。**Lead 独占，其他成员不要改本文件**。
//!
//! 目标（对应 2026-09-11 的测量结论）：
//!   一档：砍掉整矩阵空跑（NaN 预填、preflight 重复读、restrict 重复读、诊断性覆盖率扫描、
//!         众数统计的基数排序）
//!   二档：按日期块融合（rolling → preflight → 中性化 → 回测），不物化 12 张 slot 大矩阵
//!   三档：驱动层去掉主线程串行（序列化/落盘/汇总）
//!
//! 所有模块都必须满足：**结果与生产/复刻实现逐位一致**（f64 统计量允许逐位比较，
//! 不允许「近似」）。对账由各自的 src/bin/*_check.rs 负责。

use ndarray::Array2;

/// 日期块大小（行数）。64 行 × 7857 股 × 4B ≈ 2.0 MB；13 个 slot 的块 ≈ 26 MB，落在 L3 内。
pub const BLOCK_ROWS: usize = 64;

/// 一个因子跑完后的结果（字段与引擎 TailTaskResult 的关键部分对齐）。
#[derive(Clone, Default)]
pub struct T8Result {
    pub source_factor: String,
    pub eliminated_by_raw_cover: bool,
    pub raw_cover_before_fill: f64,
    pub raw_cover_after_fill: f64,
    pub any_window_passed_preflight: bool,
    pub preflight_maj_failed: usize,
    pub preflight_zero_failed: usize,
    pub preflight_nan_failed: usize,
    /// (derived_name, gap, summary[10])
    pub raw_summaries: Vec<(String, i32, [f64; 10])>,
    pub neu_summaries: Vec<(String, i32, [f64; 10])>,
}

/// 单个派生面的 preflight 统计（与 engine::PreflightReport 字段一一对应）。
#[derive(Clone, Copy, Debug)]
pub struct PfReport {
    pub passed: bool,
    pub majority_count_mean: f64,
    pub zero_ratio_mean: f64,
    pub nan_ratio_mean: f64,
}

/// 派生面命名（与生产 derived_names_for_variant 一致）：
/// slot 0 = `<variant>_smooth_1`；随后按 windows 顺序，每个窗口 4 个：mean/max/min/std。
pub fn derived_names_for_variant(variant_name: &str, windows: &[usize]) -> Vec<String> {
    let mut names = vec![format!("{}_smooth_1", variant_name)];
    for &w in windows {
        names.push(format!("{}_mean_smooth_{}", variant_name, w));
        names.push(format!("{}_max_smooth_{}", variant_name, w));
        names.push(format!("{}_min_smooth_{}", variant_name, w));
        names.push(format!("{}_std_smooth_{}", variant_name, w));
    }
    names
}

/// slot 总数 = 1 + 4 × windows.len()
pub fn slot_count(windows: &[usize]) -> usize {
    1 + 4 * windows.len()
}

/// 每个 slot 对应的窗口（slot 0 无窗口；其余 4 个一组）。
pub fn slot_window(windows: &[usize], slot_idx: usize) -> Option<usize> {
    if slot_idx == 0 {
        return None;
    }
    Some(windows[(slot_idx - 1) / 4])
}

// ---------------- 分阶段计时（两档对照用，AtomicU64 累加，开销 ~ns） ----------------
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

/// 准备（覆盖率 + rank/填充 + fold 构造）
pub static T_PREP: AtomicU64 = AtomicU64::new(0);
/// 滚动统计
pub static T_ROLL: AtomicU64 = AtomicU64::new(0);
/// preflight
pub static T_PF: AtomicU64 = AtomicU64::new(0);
/// 中性化
pub static T_NEU: AtomicU64 = AtomicU64::new(0);
/// 回测（raw + neu 合计）
pub static T_BT: AtomicU64 = AtomicU64::new(0);
/// worker 全时长
pub static T_WORK: AtomicU64 = AtomicU64::new(0);

#[inline]
pub fn tick(c: &AtomicU64, t0: Instant) {
    c.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
}

pub fn reset_timers() {
    for c in [&T_PREP, &T_ROLL, &T_PF, &T_NEU, &T_BT, &T_WORK] {
        c.store(0, Ordering::Relaxed);
    }
}

/// 打印每因子线程时间分解（秒/因子）。
pub fn dump_timers(tag: &str, n_factors: usize, wall: f64) {
    let per = |c: &AtomicU64| c.load(Ordering::Relaxed) as f64 / 1e9 / n_factors.max(1) as f64;
    println!(
        "[TIMER {tag}] 每因子线程秒: work={:.1} prep={:.1} roll={:.1} preflight={:.1} neutralize={:.1} backtest={:.1} | 墙钟/因子 {:.2}s",
        per(&T_WORK),
        per(&T_PREP),
        per(&T_ROLL),
        per(&T_PF),
        per(&T_NEU),
        per(&T_BT),
        wall / n_factors.max(1) as f64,
    );
}

/// 调试用：逐位比较两张同形状矩阵（NaN 对齐），返回 (是否一致, 不一致格数)。
pub fn bitwise_equal(a: &Array2<f32>, b: &Array2<f32>) -> (bool, usize) {    if a.dim() != b.dim() {
        return (false, usize::MAX);
    }
    let mut bad = 0usize;
    for (x, y) in a.iter().zip(b.iter()) {
        let same = (x.is_nan() && y.is_nan()) || x.to_bits() == y.to_bits();
        if !same {
            bad += 1;
        }
    }
    (bad == 0, bad)
}
