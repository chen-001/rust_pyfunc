//! 寻找玉佩-成交距离: 关联-差异矩阵（衰减事件场交互能量）+ 降维指标（网络因子）。
//! 纯 Rust 横截面因子族: 37 张矩阵 × 28 指标模块 = 2761 因子。
//! 供 run_factor_pipeline_cross_section 使用（compute_yupei_dist_xxx + pipeline 包装）。

pub mod compute;
pub mod indicator_ctx;
pub mod indicators;
pub mod industry;
pub mod matrix_stage;
pub mod names;
pub mod topk_util;
