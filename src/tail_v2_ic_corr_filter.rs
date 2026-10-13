//! tail_v2_ic_corr_filter: 候选因子的贪心去相关筛选。
//!
//! 2026-10 起：**不允许静默把「算不出来的相关性」当成 0（不相似）放行**。
//! 两个 IC 序列的日期轴交集为 0、交集天数少于 `min_common_days`、或交集上方差为 0 时，
//! 相关性在数学上无定义，此时：
//! - `tail_v2_select_by_ic_corr_f32` 会把每一对「算不出来」的因子（下标 + 共同交易日数 +
//!   原因）连同每个因子的有效日期数一起返回，`on_uncomputable="raise"` 时直接报错；
//! - 老入口 `tail_v2_select_by_ic_corr_abs_f32` 固定 `on_uncomputable="raise"`，
//!   它不再可能悄悄放过这种配对。
//!
//! 交集占比低于 `min_overlap_ratio`（默认 0.5）的配对不算错误，但会单独列在
//! `low_overlap` 里，由 Python 侧打印并在对账文件里留痕。

use ndarray::ArrayView2;
use numpy::PyReadonlyArray2;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

const EPS: f64 = 1e-12;

/// 相关性算不出来的原因。
#[derive(Clone, Copy, PartialEq)]
enum BadKind {
    NoCommonDate,
    TooFewCommonDays,
    ZeroVariance,
}

impl BadKind {
    fn as_str(self) -> &'static str {
        match self {
            BadKind::NoCommonDate => "no_common_date",
            BadKind::TooFewCommonDays => "too_few_common_days",
            BadKind::ZeroVariance => "zero_variance_on_overlap",
        }
    }
}

/// 绝对 Pearson 相关 + 共同有效天数。
///
/// 返回值第二项是**同时有限**的天数，第一项在该交集上算出来（无定义时为 NaN）。
#[inline]
fn pairwise_abs_corr_diag(x: &[f32], y: &[f32]) -> (f64, usize) {
    let mut n = 0usize;
    let mut sum_x = 0.0_f64;
    let mut sum_y = 0.0_f64;
    let mut sum_x2 = 0.0_f64;
    let mut sum_y2 = 0.0_f64;
    let mut sum_xy = 0.0_f64;

    for idx in 0..x.len() {
        let xv = x[idx] as f64;
        let yv = y[idx] as f64;
        if !xv.is_finite() || !yv.is_finite() {
            continue;
        }
        n += 1;
        sum_x += xv;
        sum_y += yv;
        sum_x2 += xv * xv;
        sum_y2 += yv * yv;
        sum_xy += xv * yv;
    }

    if n < 2 {
        return (f64::NAN, n);
    }

    let nf = n as f64;
    let cov = sum_xy - (sum_x * sum_y) / nf;
    let var_x = sum_x2 - (sum_x * sum_x) / nf;
    let var_y = sum_y2 - (sum_y * sum_y) / nf;
    if var_x <= EPS || var_y <= EPS {
        return (f64::NAN, n);
    }
    ((cov / (var_x.sqrt() * var_y.sqrt())).abs(), n)
}

struct GreedyReport {
    selected: Vec<usize>,
    n_dates: Vec<usize>,
    /// (候选下标, 已选下标, 共同交易日, 原因)
    uncomputable: Vec<(usize, usize, usize, BadKind)>,
    /// (候选下标, 已选下标, 共同交易日, 交集占比)
    low_overlap: Vec<(usize, usize, usize, f64)>,
}

fn greedy_select_diag(
    ic_by_factor: ArrayView2<'_, f32>,
    threshold: f64,
    min_common_days: usize,
    min_overlap_ratio: f64,
) -> GreedyReport {
    let n_factors = ic_by_factor.nrows();
    let mut n_dates = Vec::with_capacity(n_factors);
    for row in ic_by_factor.rows() {
        n_dates.push(row.iter().filter(|v| v.is_finite()).count());
    }
    let mut report = GreedyReport {
        selected: Vec::with_capacity(n_factors.min(64)),
        n_dates,
        uncomputable: Vec::new(),
        low_overlap: Vec::new(),
    };
    if n_factors == 0 {
        return report;
    }

    report.selected.push(0);

    for candidate_idx in 1..n_factors {
        let candidate = ic_by_factor.row(candidate_idx);
        let candidate_slice = candidate
            .as_slice()
            .expect("ic_by_factor must be row-major contiguous");
        let mut too_similar = false;
        for &chosen_idx in report.selected.iter() {
            let chosen = ic_by_factor.row(chosen_idx);
            let chosen_slice = chosen
                .as_slice()
                .expect("ic_by_factor must be row-major contiguous");
            let (corr_val, common) = pairwise_abs_corr_diag(candidate_slice, chosen_slice);

            let shorter = report.n_dates[candidate_idx].min(report.n_dates[chosen_idx]);
            if shorter > 0 {
                let ratio = common as f64 / shorter as f64;
                if ratio < min_overlap_ratio {
                    report.low_overlap.push((candidate_idx, chosen_idx, common, ratio));
                }
            }

            if common == 0 {
                report.uncomputable.push((
                    candidate_idx,
                    chosen_idx,
                    common,
                    BadKind::NoCommonDate,
                ));
                continue;
            }
            if common < min_common_days {
                report.uncomputable.push((
                    candidate_idx,
                    chosen_idx,
                    common,
                    BadKind::TooFewCommonDays,
                ));
                continue;
            }
            if !corr_val.is_finite() {
                report.uncomputable.push((
                    candidate_idx,
                    chosen_idx,
                    common,
                    BadKind::ZeroVariance,
                ));
                continue;
            }
            if corr_val >= threshold {
                too_similar = true;
                break;
            }
        }
        if !too_similar {
            report.selected.push(candidate_idx);
        }
    }
    report
}

fn build_report_dict<'py>(
    py: Python<'py>,
    report: &GreedyReport,
    threshold: f64,
    min_common_days: usize,
    min_overlap_ratio: f64,
) -> PyResult<&'py PyDict> {
    let out = PyDict::new(py);
    out.set_item("selected", report.selected.clone())?;
    out.set_item("n_dates", report.n_dates.clone())?;
    out.set_item("n_factors", report.n_dates.len())?;
    out.set_item("threshold", threshold)?;
    out.set_item("min_common_days", min_common_days)?;
    out.set_item("min_overlap_ratio", min_overlap_ratio)?;
    let bad = PyList::empty(py);
    for &(i, j, common, kind) in report.uncomputable.iter() {
        bad.append((i, j, common, kind.as_str()))?;
    }
    out.set_item("uncomputable", bad)?;
    let low = PyList::empty(py);
    for &(i, j, common, ratio) in report.low_overlap.iter() {
        low.append((i, j, common, ratio))?;
    }
    out.set_item("low_overlap", low)?;
    Ok(out)
}

fn check_input(view: &ArrayView2<'_, f32>, threshold: f32) -> PyResult<()> {
    if view.ndim() != 2 {
        return Err(PyValueError::new_err("ic_by_factor 必须是二维数组"));
    }
    if threshold.is_nan() {
        return Err(PyValueError::new_err("threshold 不能是 NaN"));
    }
    if !view.is_standard_layout() {
        return Err(PyValueError::new_err(
            "ic_by_factor 必须是 C contiguous；请在 Python 侧先做 np.ascontiguousarray",
        ));
    }
    Ok(())
}

fn uncomputable_message(report: &GreedyReport, threshold: f64) -> String {
    let mut text = format!(
        "贪心去相关遇到 {} 对「相关系数无法计算」的候选（阈值 {}）：这些配对的共同有效交易日不足，\
         相关性在数学上无定义，**不能**当成 0（不相似）放行。",
        report.uncomputable.len(),
        threshold
    );
    for &(i, j, common, kind) in report.uncomputable.iter().take(10) {
        text.push_str(&format!(
            "\n  候选下标 {} 与已留下标 {}：共同交易日 {}（{}，各自有效天数 {} / {}）",
            i, j, common, kind.as_str(), report.n_dates[i], report.n_dates[j]
        ));
    }
    if report.uncomputable.len() > 10 {
        text.push_str(&format!(
            "\n  ... 另有 {} 对，详见返回值 / selection_audit.json",
            report.uncomputable.len() - 10
        ));
    }
    text.push_str(
        "\n处理方式：先检查两边的 IC 序列日期轴（是否有一边几乎全空、或两边落在不同取点相位），\
         修好数据再说；确实要带着这种配对往下走时，显式传 allow_uncomputable_corr=True，\
         届时会打印警告并把放行记录写进对账文件。",
    );
    text
}

/// 带对账信息的贪心去相关：返回 dict（selected / n_dates / uncomputable / low_overlap / 阈值口径）。
///
/// `on_uncomputable`：
/// - `"raise"`（默认）遇到算不出来的配对直接抛 ValueError；
/// - `"report"` 只记录不抛错，由调用方决定怎么处理（Python 侧用它来生成带因子名的报错）。
#[pyfunction]
#[pyo3(signature = (ic_by_factor, threshold, min_common_days=2, on_uncomputable="raise", min_overlap_ratio=0.5))]
pub fn tail_v2_select_by_ic_corr_f32<'py>(
    py: Python<'py>,
    ic_by_factor: PyReadonlyArray2<'py, f32>,
    threshold: f32,
    min_common_days: usize,
    on_uncomputable: &str,
    min_overlap_ratio: f32,
) -> PyResult<&'py PyDict> {
    let view = ic_by_factor.as_array();
    check_input(&view, threshold)?;
    if min_overlap_ratio.is_nan() || !(0.0..=1.0).contains(&min_overlap_ratio) {
        return Err(PyValueError::new_err("min_overlap_ratio 必须在 0~1 之间"));
    }
    let mode = match on_uncomputable {
        "raise" | "report" => on_uncomputable,
        other => {
            return Err(PyValueError::new_err(format!(
                "on_uncomputable 只能是 raise 或 report，收到 {other}"
            )))
        }
    };
    let report = py.allow_threads(|| {
        greedy_select_diag(
            view,
            threshold as f64,
            min_common_days,
            min_overlap_ratio as f64,
        )
    });
    if mode == "raise" && !report.uncomputable.is_empty() {
        return Err(PyValueError::new_err(uncomputable_message(
            &report,
            threshold as f64,
        )));
    }
    build_report_dict(
        py,
        &report,
        threshold as f64,
        min_common_days,
        min_overlap_ratio as f64,
    )
}

/// 老入口：只返回保留下来的下标，行为与 `tail_v2_select_by_ic_corr_f32` 的 raise 模式一致。
///
/// 2026-10 起不再允许「算不出来的相关性 = 0（不相似）」这种静默放行。
#[pyfunction]
#[pyo3(signature = (ic_by_factor, threshold))]
pub fn tail_v2_select_by_ic_corr_abs_f32(
    ic_by_factor: PyReadonlyArray2<'_, f32>,
    threshold: f32,
) -> PyResult<Vec<usize>> {
    let view = ic_by_factor.as_array();
    check_input(&view, threshold)?;
    let report = greedy_select_diag(view, threshold as f64, 2, 0.0);
    if !report.uncomputable.is_empty() {
        return Err(PyValueError::new_err(uncomputable_message(
            &report,
            threshold as f64,
        )));
    }
    Ok(report.selected)
}
