//! 寻找玉佩-成交距离: 关联-差异矩阵 + 降维指标（网络因子）。
//!
//! 数据流（单日全市场, 20150105..20260717 批量）:
//!   1. Level2 逐笔 CSV（/ssd_data/stock/{date}/transaction/）按文件大小取 top-4000 只
//!   2. 每股预处理: 10ms 桶聚合, u[9] 权重场（cnt/vol/logvol/flow±/urg±/ext±）
//!   3. 37 张有向交互矩阵 S_dir[i][j]（i 领先 j 的衰减事件场能量）:
//!      cnt 9τ(0.05..30s), vol 6τ, logvol 4τ, flow/urg/ext same/opp 各 4/3/2τ
//!      SIMD 实现（phase1 7×ymm 递推 + phase2 AVX2 9×FMA, 争用下 105s/天）
//!   4. 28 个降维指标模块 → 2761 因子（强度/集中度/方向性/邻居质量/中心性/
//!      聚类/社群/行业/跨日动态/正反网络等, 全部遍历 37 矩阵）
//!   5. cross-section pipeline（run_factor_pipeline_cross_section, pipeline="yupei_dist"）
//!      → colblk 存储 → tail_pipeline_engine 回测
//!
//! 入口:
//!   - compute_yupei_dist_full(date): 单日全量 (codes, vals N×2761)
//!   - compute_yupei_dist_full_with_prev(date, prev): 带前一日矩阵（dyn_* 跨日因子）
//!   - py_yupei_dist / py_yupei_dist_names: Python 入口
//!
//! 正确性基准:
//!   - 37 矩阵经桶级前缀和对照朴素 Python（ratio ≈ 1.0000）
//!   - SIMD 版 vs 标量版最大相对差 9.7e-8
//!   - 正式库 vs sandbox 备份: 除 dyn（单日任务无 prev, 预期 NaN）外逐位一致
//!   - 指标并行化: 两次运行 md5 一致（确定性）
//!
//! 性能（20241231, universe 4000, 3964 股, 50 线程）:
//!   - 矩阵: 标量 212s → AVX2 131s → phase1+2 SIMD 105s（争用下）; 空机预估 ~45s
//!   - 指标: ~29s（clustering/edge_dynamics/community 并行化后）
//!   - 单日全流程空机预估 ~76s（2761 因子×37 矩阵; 原 60s 目标为 105 因子时代预算）
//!
//! 批量任务: 20150105..20260717 共 2801 天, 8 worker×50 线程, ~10h;
//! store: /hdd/user_home_unsafe/chenzongwei/factor_store_yupei_dist（~310GB）
//! 行业: /hdd/user_home_unsafe/chenzongwei/yupei_dist_backup/{date}/industry.bin（2803 天已提取）

pub mod compute;
pub mod indicator_ctx;
pub mod indicators;
pub mod industry;
pub mod matrix_stage;
pub mod matrix_store;
pub mod names;
pub mod topk_util;
