//! 多因子 CAPM 的路线时间序列编排（从 sandbox_multi_factor/src/multi_factor.rs 迁入主项目）。
//!
//! 历史背景: sandbox 版 compute_route_timeseries 按 y（14 路特征）分组嵌套编排 53 个
//! (y, 模型) 组合; 正式库 compute_multi_factor_capm_full_inner 改为扁平模型主序编排并
//! 优化了滚动矩共享。本模块保留 sandbox 的嵌套版编排（逐位一致, 供按 y 分组消费的
//! 场景使用）, 复用正式库的 compute_one_combo 计算内核。
//!
//! 结构: RouteResult[y] = { y, combos: 本 y 的各模型组合缓冲, model_indices: 模型序 }

use rayon::prelude::*;

use crate::multi_factor_capm_metrics::{compute_one_combo, ComboBuf, MODELS, N_FEATURES};

/// 单条路线（一个 y 特征）的结果: 各模型的组合缓冲与模型序。
pub struct RouteResult {
    pub y: usize,
    pub combos: Vec<ComboBuf>,
    pub model_indices: Vec<usize>,
}

/// 预分配全部 14 路组合缓冲（嵌套结构 [y][combo]; 页错误与读盘/其他计算重叠）。
/// 注意: 与 multi_factor_capm_metrics::prealloc_combos（扁平 53 路）不同, 本函数
/// 返回 sandbox 版嵌套布局, 专供 compute_route_timeseries 使用。
pub fn prealloc_route_combos(n_stocks: usize) -> Vec<Vec<ComboBuf>> {
    let mut y_models: [Vec<usize>; N_FEATURES] = std::array::from_fn(|_| Vec::new());
    for (mi, m) in MODELS.iter().enumerate() {
        for &y in m.ys {
            y_models[y].push(mi);
        }
    }
    (0..N_FEATURES)
        .into_par_iter()
        .map(|y| {
            y_models[y]
                .iter()
                .map(|&mi| {
                    let k = MODELS[mi].k;
                    ComboBuf::new(4 * k + 6, 4 * k + 9, n_stocks)
                })
                .collect()
        })
        .collect()
}

/// 53 个 (y, 模型) 任务并行（每任务独立滚动矩）, 结果按 y 分组返回。
/// 重组语义与 sandbox 一致: out[y].combos 顺序 = 该 y 的模型注册序。
pub fn compute_route_timeseries(
    signals: &[f32],
    market: &[f64],
    n_stocks: usize,
    mut prealloc: Vec<Vec<ComboBuf>>,
) -> Vec<RouteResult> {
    let mut y_models: [Vec<usize>; N_FEATURES] = std::array::from_fn(|_| Vec::new());
    for (mi, m) in MODELS.iter().enumerate() {
        for &y in m.ys {
            y_models[y].push(mi);
        }
    }
    // 53 个 (y, 模型) 任务并行，每个任务独立滚动矩
    let mut tasks: Vec<(usize, usize, usize, ComboBuf)> = Vec::new();
    for y in 0..N_FEATURES {
        for (ci, &mi) in y_models[y].iter().enumerate() {
            let buf = std::mem::take(&mut prealloc[y][ci]);
            tasks.push((y, mi, ci, buf));
        }
    }
    let results: Vec<(usize, usize, usize, ComboBuf)> = tasks
        .into_par_iter()
        .map(|(y, mi, ci, buf)| {
            let buf = compute_one_combo(signals, market, n_stocks, y, mi, buf);
            (y, mi, ci, buf)
        })
        .collect();
    // 重组回 [y][combo] 结构
    let mut out: Vec<RouteResult> = (0..N_FEATURES)
        .map(|y| RouteResult {
            y,
            combos: Vec::with_capacity(y_models[y].len()),
            model_indices: y_models[y].clone(),
        })
        .collect();
    for (y, _mi, ci, buf) in results {
        out[y].combos.push(buf);
        // 保持模型序：结果按任务序收集，任务序 = y 内 ci 升序 ✓
        let _ = ci;
    }
    out
}

// 注: (y, 模型) 任务划分与正式库扁平编排共 53 组合等价性由构造保证
// （两者调用同一 compute_one_combo, 仅分组方式不同）。
