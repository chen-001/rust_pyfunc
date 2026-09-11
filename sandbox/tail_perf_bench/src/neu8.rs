//! neu8：中性化的「按日期块」入口（二档消费侧）。
//!
//! 二档融合流水线里只有第 [t0,t1) 行的块，没有整张 (T,N) slot 矩阵；
//! 本模块把它包成稳定接口，内部直接转发 `v3::v3_slot_range`（在 v3.rs 里，
//! 是 `v3_slot` 的按行区间版本），只触碰块内的行——**不物化大矩阵**。
//!
//! 逐位一致要求：对同一 slot，按块调用拼起来的输出必须与一次性 `v3_slot(整张)` 逐位相同
//! （含 NaN 位置）。自测入口：`src/bin/neu8_check.rs`。

use ndarray::{Array2, ArrayView2};

use crate::v3::{self, V3Scratch, V3Shared};

/// 对 slot 的第 [t0,t1) 行做中性化，返回 (t1-t0, N) 的块。
///
/// `shared` 的每日预计算按**绝对日期下标**索引，因此 `t0`/`t1` 必须传全局行号，
/// `slot_block` 只装第 [t0,t1) 行。
/// `scratch` 由调用方按线程持有一份并跨块复用（避免每块重新分配 10 个长度 N 的缓冲）；
/// 本函数不 new 任何 scratch，也不分配 (T,N) 级别的大对象。
pub fn neutralize_block(
    slot_block: &ArrayView2<f32>,
    shared: &V3Shared,
    t0: usize,
    t1: usize,
    scratch: &mut V3Scratch,
) -> Array2<f32> {
    assert_eq!(slot_block.nrows(), t1 - t0, "slot_block 行数必须等于 t1-t0");
    assert_eq!(slot_block.ncols(), shared.n_stocks, "slot_block 列数必须等于 N");
    assert!(t1 <= shared.v2.per_date.len(), "t1 超出日期维");
    let (out, _times) = v3::v3_slot_range(slot_block.view(), shared, t0, t1, scratch);
    out
}
