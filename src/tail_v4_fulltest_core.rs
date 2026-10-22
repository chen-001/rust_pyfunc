//! tail_v4_fulltest_core —— fulltest 单因子回测「整表计算」的全量下沉。
//!
//! 被替换的 Python 段：`design_whatever/fulltest_whatever.py`
//! `PortfolioRetSummary.portf_ret_summary_update_mp` 里除画图/写表以外的**全部**环节：
//!
//!   1. ahead_rollover（信号与 restrict 列左移一位）
//!   2. `const_signal_df.reindex(S_restrict_df.index)`（按股票代码对齐到 restrict 轴）
//!   3. 日期区间筛选（`col > start_date && col <= end_date`）+ 全 NaN 日期剔除
//!   4. `valid_df` / `valid_symbol_num`
//!   5. `temp[restrict==1 | isnan(restrict)] = NaN`，再整表 rank（average, pct）
//!   6. `res_rank_ic = corrwith(f_rank, ret_resi, axis=0)`
//!   7. top/bottom 半区各自的 rank 与 IC
//!   8. `percentile`（与 5 同一张表，Python 里重复算了一遍，这里复用）
//!   9. `group_number` 十分组 + 首末组归位
//!  10. `weekly_group`（按 H_hori 取模后前向填充）
//!  11. 10 组权重与加权残差收益 `perf_df`
//!  12. 逐日循环（复用 `tail_v4_fulltest_rust::run_core`，口径不变）
//!  13. 全部汇总指标（IC_mean / IR / 多空年化 / 夏普 / 回撤 / 对冲 / 残差 IC / 多头 IC / ratio_mean）
//!
//! Python 侧只剩三件事：把数据从 h5/parquet 读进来（后续阶段再下沉）、画图、写 CSV。
//!
//! # 布局与精度
//! - 输入一律 **stocks_first=True**，即 (N_stocks, T_dates)，与 pandas DataFrame 原生一致。
//! - 内部统一转成 date-major（`data[t * n_s + i]`），整表 rank / 分组按日顺序访问，缓存友好。
//! - 全部按 float64 计算；与 pandas 的差异只可能来自浮点累加顺序（~1e-15 量级）。
//! - pandas 的 `rank(axis=0, pct=True)` 默认 `method='average'`、`na_option='keep'`，
//!   这里逐日复刻：非 NaN 子集按值排序，并列取平均秩（1-based），再除以非 NaN 个数。
//! - `fillna(-9999)` / `ffill` / `min/max` 广播的先后顺序照搬 Python，不做等价改写。

use std::collections::HashMap;

use ndarray::{Array1, Array2, ArrayView2};
use numpy::{PyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyString};

use crate::tail_v4_fulltest_rust::run_core;

const NAN: f64 = f64::NAN;

/// date-major 面板：`data[t * n_s + i]`。
struct Panel {
    data: Vec<f64>,
    n_t: usize,
    n_s: usize,
}

impl Panel {
    fn new(n_t: usize, n_s: usize) -> Self {
        Self {
            data: vec![NAN; n_t * n_s],
            n_t,
            n_s,
        }
    }

    /// (n_s, n_t) 的 stock-first 视图 → date-major。
    fn from_stocks_first(a: &ArrayView2<f64>) -> Self {
        let n_s = a.nrows();
        let n_t = a.ncols();
        let mut p = Self::new(n_t, n_s);
        for t in 0..n_t {
            for i in 0..n_s {
                p.data[t * n_s + i] = a[(i, t)];
            }
        }
        p
    }

    #[inline]
    fn get(&self, t: usize, i: usize) -> f64 {
        self.data[t * self.n_s + i]
    }

    #[inline]
    fn set(&mut self, t: usize, i: usize, v: f64) {
        self.data[t * self.n_s + i] = v;
    }

    #[inline]
    fn col(&self, t: usize) -> &[f64] {
        &self.data[t * self.n_s..(t + 1) * self.n_s]
    }

    #[inline]
    fn col_mut(&mut self, t: usize) -> &mut [f64] {
        &mut self.data[t * self.n_s..(t + 1) * self.n_s]
    }
}

/// pandas `rank(axis=0, method="average", pct=True)` 的单日实现（原地）。
/// NaN 位置保持 NaN；并列取平均秩；分母是当日非 NaN 个数。
fn rank_pct_average_in_place(col: &mut [f64]) {
    let vals = col.to_vec();
    let mut idx: Vec<u32> = Vec::with_capacity(vals.len());
    for (i, v) in vals.iter().enumerate() {
        if !v.is_nan() {
            idx.push(i as u32);
        }
    }
    let n = idx.len();
    if n == 0 {
        return;
    }
    idx.sort_by(|&a, &b| {
        vals[a as usize]
            .partial_cmp(&vals[b as usize])
            .expect("NaN 已在上面过滤")
    });
    let denom = n as f64;
    let mut k = 0usize;
    while k < n {
        let mut j = k;
        let v = vals[idx[k] as usize];
        while j + 1 < n && vals[idx[j + 1] as usize] == v {
            j += 1;
        }
        let pct = ((k + 1) as f64 + (j + 1) as f64) / 2.0 / denom;
        for m in k..=j {
            let i = idx[m] as usize;
            col[i] = pct;
        }
        k = j + 1;
    }
}

/// 逐列 Pearson 相关，跳过任一侧为 NaN 的位置；有效点 <2 或某一侧方差为 0 时返回 NaN。
/// 与 pandas `DataFrame.corrwith(axis=0)` 的两遍算法同构（先均值，再离差乘积和）。
fn pearson_nan(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len();
    let mut cnt = 0usize;
    let mut sx = 0.0f64;
    let mut sy = 0.0f64;
    for i in 0..n {
        let (a, b) = (x[i], y[i]);
        if a.is_nan() || b.is_nan() {
            continue;
        }
        cnt += 1;
        sx += a;
        sy += b;
    }
    if cnt < 2 {
        return NAN;
    }
    let mx = sx / cnt as f64;
    let my = sy / cnt as f64;
    let mut sxy = 0.0f64;
    let mut sxx = 0.0f64;
    let mut syy = 0.0f64;
    for i in 0..n {
        let (a, b) = (x[i], y[i]);
        if a.is_nan() || b.is_nan() {
            continue;
        }
        let dx = a - mx;
        let dy = b - my;
        sxy += dx * dy;
        sxx += dx * dx;
        syy += dy * dy;
    }
    if sxx == 0.0 || syy == 0.0 {
        return NAN;
    }
    sxy / (sxx.sqrt() * syy.sqrt())
}

fn nanmean(v: &[f64]) -> f64 {
    let mut s = 0.0f64;
    let mut n = 0usize;
    for &x in v {
        if !x.is_nan() {
            s += x;
            n += 1;
        }
    }
    if n == 0 {
        NAN
    } else {
        s / n as f64
    }
}

fn nanstd_ddof0(v: &[f64]) -> f64 {
    let m = nanmean(v);
    if m.is_nan() {
        return NAN;
    }
    let mut s = 0.0f64;
    let mut n = 0usize;
    for &x in v {
        if !x.is_nan() {
            let d = x - m;
            s += d * d;
            n += 1;
        }
    }
    if n == 0 {
        NAN
    } else {
        (s / n as f64).sqrt()
    }
}

/// pandas `Series.std()`（ddof=1）。
fn nanstd_ddof1(v: &[f64]) -> f64 {
    let m = nanmean(v);
    if m.is_nan() {
        return NAN;
    }
    let mut s = 0.0f64;
    let mut n = 0usize;
    for &x in v {
        if !x.is_nan() {
            let d = x - m;
            s += d * d;
            n += 1;
        }
    }
    if n < 2 {
        NAN
    } else {
        (s / (n - 1) as f64).sqrt()
    }
}

/// 与 Python `max_drawdown_cal` 逐行等价。
fn max_drawdown_cal(cum_ret: &[f64]) -> f64 {
    let mut max_ret = 0.0f64;
    let mut max_drawdown = 0.0f64;
    for &v in cum_ret {
        if v > max_ret {
            max_ret = v;
        }
        if max_drawdown < max_ret - v {
            max_drawdown = max_ret - v;
        }
    }
    max_drawdown
}

fn to_2d(v: &[f64], rows: usize, cols: usize) -> Array2<f64> {
    Array2::from_shape_vec((rows, cols), v.to_vec()).expect("形状与元素数一致")
}

#[pyfunction]
#[pyo3(signature = (
    signal, signal_stocks, ret, ret_sum, restrict, target_stocks,
    ret_resi, ret_resi_raw_1d, index_ret, index_ret_dates, dates_all,
    portf_num=10, h_hori=5, start_date="", end_date="",
))]
#[allow(clippy::too_many_arguments)]
pub fn tail_v4_fulltest_core_rust(
    py: Python<'_>,
    signal: PyReadonlyArray2<f64>,
    signal_stocks: Vec<String>,
    ret: PyReadonlyArray2<f64>,
    ret_sum: PyReadonlyArray2<f64>,
    restrict: PyReadonlyArray2<f64>,
    target_stocks: Vec<String>,
    ret_resi: PyReadonlyArray2<f64>,
    ret_resi_raw_1d: PyReadonlyArray2<f64>,
    index_ret: PyReadonlyArray1<f64>,
    index_ret_dates: Vec<String>,
    dates_all: Vec<String>,
    portf_num: usize,
    h_hori: usize,
    start_date: &str,
    end_date: &str,
) -> PyResult<PyObject> {
    if portf_num == 0 || h_hori == 0 {
        return Err(PyValueError::new_err("portf_num 与 h_hori 必须大于 0"));
    }
    let sig_a = signal.as_array();
    let ret_a = ret.as_array();
    let rsum_a = ret_sum.as_array();
    let rstr_a = restrict.as_array();
    let resi_a = ret_resi.as_array();
    let resi1_a = ret_resi_raw_1d.as_array();
    let idx_a = index_ret.as_array();

    let t0 = sig_a.ncols();
    if dates_all.len() != t0 {
        return Err(PyValueError::new_err(format!(
            "dates_all 长度 {} 与 signal 列数 {t0} 不一致",
            dates_all.len()
        )));
    }
    if signal_stocks.len() != sig_a.nrows() {
        return Err(PyValueError::new_err(format!(
            "signal_stocks 长度 {} 与 signal 行数 {} 不一致",
            signal_stocks.len(),
            sig_a.nrows()
        )));
    }
    let n_s = target_stocks.len();
    for (name, n_rows, n_cols) in [
        ("ret", ret_a.nrows(), ret_a.ncols()),
        ("ret_sum", rsum_a.nrows(), rsum_a.ncols()),
        ("restrict", rstr_a.nrows(), rstr_a.ncols()),
        ("ret_resi", resi_a.nrows(), resi_a.ncols()),
        ("ret_resi_raw_1d", resi1_a.nrows(), resi1_a.ncols()),
    ] {
        if n_rows != n_s || n_cols != t0 {
            return Err(PyValueError::new_err(format!(
                "{name} 形状 ({n_rows},{n_cols}) 与目标轴 ({n_s},{t0}) 不一致"
            )));
        }
    }
    if idx_a.len() != index_ret_dates.len() {
        return Err(PyValueError::new_err(format!(
            "index_ret 长度 {} 与 index_ret_dates 长度 {} 不一致",
            idx_a.len(),
            index_ret_dates.len()
        )));
    }
    let mut idx_pos: HashMap<&str, usize> = HashMap::with_capacity(index_ret_dates.len());
    for (i, d) in index_ret_dates.iter().enumerate() {
        idx_pos.insert(d.as_str(), i);
    }
    let mut idx_full = vec![NAN; t0];
    for t in 0..t0 {
        match idx_pos.get(dates_all[t].as_str()) {
            Some(&i) => idx_full[t] = idx_a[i],
            None => {
                return Err(PyValueError::new_err(format!(
                    "index_ret 缺少日期 {}",
                    dates_all[t]
                )))
            }
        }
    }

    // ---- 按股票代码对齐到 restrict 轴（对应 pandas 的 reindex）----
    let mut sig_pos: HashMap<&str, usize> = HashMap::with_capacity(signal_stocks.len());
    for (i, s) in signal_stocks.iter().enumerate() {
        sig_pos.insert(s.as_str(), i);
    }
    let align: Vec<Option<usize>> = target_stocks
        .iter()
        .map(|s| sig_pos.get(s.as_str()).copied())
        .collect();

    let mut sig_p = Panel::new(t0, n_s);
    for t in 0..t0 {
        for (j, slot) in align.iter().enumerate() {
            if let Some(i) = *slot {
                sig_p.data[t * n_s + j] = sig_a[(i, t)];
            }
        }
    }
    let ret_p = Panel::from_stocks_first(&ret_a);
    let rsum_p = Panel::from_stocks_first(&rsum_a);
    let rstr_p = Panel::from_stocks_first(&rstr_a);
    let resi_p = Panel::from_stocks_first(&resi_a);
    let resi1_p = Panel::from_stocks_first(&resi1_a);

    // ---- ahead_rollover + 日期筛选 + 全 NaN 日期剔除 ----
    //
    // Python 原式是
    //   DataFrame(values[:, :-1], columns=columns[1:])
    // 即**值不动、列名右移一位**：第 j 列保留原第 j 列的数值，却挂上原第 j+1 列的日期。
    // 等价说法是「t 日的信号记在 t+1 日名下」（信号前移一天，持仓在次日建立）。
    // 所以筛选后第 k 个日期 d_k = dates_all[t] 对应：
    //   signal / restrict 取原第 t-1 列，ret / ret_sum / ret_resi 取原第 t 列。
    let mut keep: Vec<usize> = Vec::new();
    for t in 1..t0 {
        let d = dates_all[t].as_str();
        if d <= start_date || d > end_date {
            continue;
        }
        let mut all_nan = true;
        for j in 0..n_s {
            if !sig_p.get(t - 1, j).is_nan() {
                all_nan = false;
                break;
            }
        }
        if all_nan {
            continue;
        }
        keep.push(t);
    }
    let nt = keep.len();

    let take_at = |p: &Panel, shift: usize| -> Panel {
        let mut q = Panel::new(nt, n_s);
        if nt > 0 {
            for (k, &t) in keep.iter().enumerate() {
                let src = t - shift;
                q.data[k * n_s..(k + 1) * n_s]
                    .copy_from_slice(&p.data[src * n_s..(src + 1) * n_s]);
            }
        }
        q
    };
    let sig_q = take_at(&sig_p, 1);
    let rstr_q = take_at(&rstr_p, 1);
    let ret_q = take_at(&ret_p, 0);
    let rsum_q = take_at(&rsum_p, 0);
    let resi_q = take_at(&resi_p, 0);
    let resi1_q = take_at(&resi1_p, 0);
    let idx_q: Vec<f64> = keep.iter().map(|&t| idx_full[t]).collect();
    let dates: Vec<String> = keep.iter().map(|&t| dates_all[t].clone()).collect();

    // ---- valid_symbol_num：restrict==1 → 0，NaN 记 0，其余 v+1；逐列求和 ----
    let mut valid_symbol_num = vec![0.0f64; nt];
    for t in 0..nt {
        let mut s = 0.0f64;
        for j in 0..n_s {
            let r = rstr_q.get(t, j);
            if r.is_nan() {
                continue;
            }
            s += if r == 1.0 { 0.0 } else { r + 1.0 };
        }
        valid_symbol_num[t] = s;
    }

    // ---- temp（restrict==1 或 NaN → NaN）与整表 rank ----
    let mut rank_p = Panel::new(nt, n_s);
    for t in 0..nt {
        for j in 0..n_s {
            let r = rstr_q.get(t, j);
            let s = sig_q.get(t, j);
            rank_p.set(t, j, if r == 1.0 || r.is_nan() { NAN } else { s });
        }
    }
    for t in 0..nt {
        rank_pct_average_in_place(rank_p.col_mut(t));
    }

    // ---- res_rank_ic / top_ic / bottom_ic ----
    let mut res_rank_ic = vec![NAN; nt];
    let mut top_ic = vec![NAN; nt];
    let mut bottom_ic = vec![NAN; nt];
    {
        let mut masked_f = vec![NAN; n_s];
        let mut masked_r = vec![NAN; n_s];
        for t in 0..nt {
            for j in 0..n_s {
                let f = rank_p.get(t, j);
                masked_f[j] = f;
                masked_r[j] = resi_q.get(t, j);
            }
            res_rank_ic[t] = pearson_nan(&masked_f, &masked_r);

            // top half：f_rank > 0.5 的位置保留，其余 NaN
            for j in 0..n_s {
                let f = rank_p.get(t, j);
                let ok = !f.is_nan() && f > 0.5;
                masked_f[j] = if ok { f } else { NAN };
                masked_r[j] = if ok { resi_q.get(t, j) } else { NAN };
            }
            rank_pct_average_in_place(&mut masked_r);
            top_ic[t] = pearson_nan(&masked_f, &masked_r);

            // bottom half：f_rank < 0.5 的位置保留，其余 NaN
            for j in 0..n_s {
                let f = rank_p.get(t, j);
                let ok = !f.is_nan() && f < 0.5;
                masked_f[j] = if ok { f } else { NAN };
                masked_r[j] = if ok { resi_q.get(t, j) } else { NAN };
            }
            rank_pct_average_in_place(&mut masked_r);
            bottom_ic[t] = pearson_nan(&masked_f, &masked_r);
        }
    }

    // ---- group_number（percentile 与 rank_p 是同一张表，Python 里重复算了一遍）----
    let mut group_p = Panel::new(nt, n_s);
    let pnum_f = portf_num as f64;
    for t in 0..nt {
        for j in 0..n_s {
            let p = rank_p.get(t, j);
            if p.is_nan() {
                continue;
            }
            group_p.set(t, j, pnum_f - ((1.0 - p) * pnum_f).floor());
        }
    }
    // 首末组归位：先按列最小值 → 1.0，再按列最大值 → 10.0（顺序照搬 Python）
    for t in 0..nt {
        let col = group_p.col(t);
        let mut mn = NAN;
        let mut mx = NAN;
        for &v in col.iter() {
            if v.is_nan() {
                continue;
            }
            if mn.is_nan() || v < mn {
                mn = v;
            }
            if mx.is_nan() || v > mx {
                mx = v;
            }
        }
        let cur = group_p.col_mut(t);
        if !mn.is_nan() {
            for v in cur.iter_mut() {
                if *v == mn {
                    *v = 1.0;
                }
            }
        }
        if !mx.is_nan() {
            for v in cur.iter_mut() {
                if *v == mx {
                    *v = 10.0;
                }
            }
        }
    }

    // ---- weekly_group：fillna(-9999) → 非格点日置 NaN → ffill ----
    let mut weekly_p = Panel::new(nt, n_s);
    for t in 0..nt {
        for j in 0..n_s {
            let v = group_p.get(t, j);
            weekly_p.set(t, j, if v.is_nan() { -9999.0 } else { v });
        }
    }
    for t in 0..nt {
        if t % h_hori != 0 {
            for j in 0..n_s {
                weekly_p.set(t, j, NAN);
            }
        }
    }
    for t in 1..nt {
        for j in 0..n_s {
            if weekly_p.get(t, j).is_nan() {
                let prev = weekly_p.get(t - 1, j);
                weekly_p.set(t, j, prev);
            }
        }
    }

    // ---- 10 组权重与加权残差收益 ----
    let mut perf = vec![NAN; portf_num * nt];
    let mut weight = vec![0.0f64; n_s];
    for gi in 1..=portf_num {
        let gi_f = gi as f64;
        for t in 0..nt {
            let mut sum = 0.0f64;
            for j in 0..n_s {
                let w = if weekly_p.get(t, j) == gi_f { 1.0 } else { 0.0 };
                weight[j] = w;
                sum += w;
            }
            let mut acc = 0.0f64;
            let mut cnt = 0usize;
            if sum > 0.0 {
                let inv = 1.0 / sum;
                for j in 0..n_s {
                    if weight[j] == 1.0 {
                        let r = resi1_q.get(t, j);
                        if !r.is_nan() {
                            acc += inv * r;
                            cnt += 1;
                        }
                    }
                }
            }
            perf[(gi - 1) * nt + t] = if cnt == 0 { NAN } else { acc };
        }
    }

    // ---- unique 检查：Python 在 len(unique_values) < portf_num 时提前 return ----
    let mut uniq: Vec<f64> = Vec::with_capacity(portf_num);
    'outer: for t in 0..nt {
        for j in 0..n_s {
            let v = sig_q.get(t, j);
            if v.is_nan() {
                continue;
            }
            if !uniq.iter().any(|&u| u == v) {
                uniq.push(v);
                if uniq.len() >= portf_num {
                    break 'outer;
                }
            }
        }
    }
    let skipped = uniq.len() < portf_num;

    // ---- 逐日循环（复用已有 Rust 核）----
    let (portf_ret, ic_values, ic_dates, stock_num, stock_num_dates) = if skipped || nt == 0 {
        (vec![0.0f64; portf_num * nt], Vec::new(), Vec::new(), Vec::new(), Vec::new())
    } else {
        let sig_2d = to_2d(&sig_q.data, nt, n_s);
        let ret_2d = to_2d(&ret_q.data, nt, n_s);
        let rsum_2d = to_2d(&rsum_q.data, nt, n_s);
        let rstr_2d = to_2d(&rstr_q.data, nt, n_s);
        let uniq_owned = uniq.clone();
        let out = run_core(
            py,
            &sig_2d.view(),
            &ret_2d.view(),
            &rsum_2d.view(),
            &rstr_2d.view(),
            &dates,
            portf_num,
            h_hori,
            Some(&uniq_owned),
            Some(false),
        )?;
        (
            out.portf_ret,
            out.ic_values,
            out.ic_dates,
            out.stock_num,
            out.stock_num_dates,
        )
    };
    // ---- 汇总指标 ----
    let ic_mean = nanmean(&ic_values);
    let ic_std = nanstd_ddof0(&ic_values);
    let icir = if ic_std == 0.0 || ic_std.is_nan() {
        NAN
    } else {
        ic_mean.abs() / ic_std * ((250.0 / h_hori as f64).sqrt())
    };

    // portf_ret 按 Python 的 fillna(0) 补齐空组
    let portf_ret_filled: Vec<f64> = portf_ret
        .iter()
        .map(|&v| if v.is_nan() { 0.0 } else { v })
        .collect();

    let ls_ret: Vec<f64> = (0..nt)
        .map(|t| {
            let g1 = portf_ret_filled[t];
            let gp = portf_ret_filled[(portf_num - 1) * nt + t];
            if ic_mean < 0.0 {
                g1 - gp
            } else {
                gp - g1
            }
        })
        .collect();
    let hedge_ret: Vec<f64> = (0..nt)
        .map(|t| {
            let g1 = portf_ret_filled[t];
            let gp = portf_ret_filled[(portf_num - 1) * nt + t];
            if ic_mean < 0.0 {
                g1 - idx_q[t]
            } else {
                gp - idx_q[t]
            }
        })
        .collect();

    let mut ls_cum = Vec::with_capacity(nt);
    let mut acc = 0.0f64;
    for &v in &ls_ret {
        acc += v;
        ls_cum.push(acc);
    }
    let mut hedge_cum = Vec::with_capacity(nt);
    let mut acc2 = 0.0f64;
    for &v in &hedge_ret {
        acc2 += v;
        hedge_cum.push(acc2);
    }

    let ls_mean = nanmean(&ls_ret);
    let ls_std = nanstd_ddof1(&ls_ret);
    let mean_annualized_return = ls_mean * 250.0 * 100.0;
    let mean_annualized_sharpe_ratio = if ls_std == 0.0 || ls_std.is_nan() {
        NAN
    } else {
        ls_mean / ls_std * 250.0f64.sqrt()
    };
    let max_drawdown = max_drawdown_cal(&ls_cum) * 100.0;

    let hedge_mean = nanmean(&hedge_ret);
    let hedge_std = nanstd_ddof1(&hedge_ret);
    let hedge_annualized_return = hedge_mean * 250.0 * 100.0;
    let hedge_annualized_sharpe_ratio = if hedge_std == 0.0 || hedge_std.is_nan() {
        NAN
    } else {
        hedge_mean / hedge_std * 250.0f64.sqrt()
    };
    let hedge_max_drawdown = max_drawdown_cal(&hedge_cum) * 100.0;

    let res_ic_mean = nanmean(&res_rank_ic);
    let long_only_series: Vec<f64> = if res_ic_mean > 0.0 {
        top_ic.clone()
    } else {
        bottom_ic.iter().map(|&v| -v).collect()
    };
    let long_only_ic = nanmean(&long_only_series);

    // ratio_mean：stock_num 只落在有信号的日期上，分母用同一批日期的 valid_symbol_num
    let mut pos_of: HashMap<&str, usize> = HashMap::with_capacity(dates.len());
    for (i, d) in dates.iter().enumerate() {
        pos_of.insert(d.as_str(), i);
    }
    let mut ratio_series = vec![NAN; stock_num_dates.len()];
    for (k, d) in stock_num_dates.iter().enumerate() {
        if let Some(&pos) = pos_of.get(d.as_str()) {
            let denom = valid_symbol_num[pos];
            if denom != 0.0 {
                ratio_series[k] = stock_num[k] as f64 / denom;
            }
        }
    }
    let ratio_mean = nanmean(&ratio_series);

    let sign = if ic_mean < 0.0 {
        "long small, short large"
    } else {
        "long large, short small"
    };

    let dict = PyDict::new(py);
    dict.set_item(
        "portf_ret",
        PyArray::from_owned_array(py, to_2d(&portf_ret_filled, portf_num, nt)),
    )?;
    dict.set_item("perf", PyArray::from_owned_array(py, to_2d(&perf, portf_num, nt)))?;
    dict.set_item("dates", PyList::new(py, dates.iter().map(|s| PyString::new(py, s))))?;
    dict.set_item("ic_values", PyList::new(py, ic_values.iter()))?;
    dict.set_item(
        "ic_dates",
        PyList::new(py, ic_dates.iter().map(|s| PyString::new(py, s))),
    )?;
    dict.set_item(
        "stock_num",
        PyList::new(py, stock_num.iter().map(|&v| v as i64)),
    )?;
    dict.set_item(
        "stock_num_dates",
        PyList::new(py, stock_num_dates.iter().map(|s| PyString::new(py, s))),
    )?;
    dict.set_item(
        "res_rank_ic_series",
        PyArray::from_owned_array(py, Array1::from(shift4(&res_rank_ic))),
    )?;
    dict.set_item(
        "long_only_ic_series",
        PyArray::from_owned_array(py, Array1::from(shift4(&long_only_series))),
    )?;
    dict.set_item(
        "top_ic",
        PyArray::from_owned_array(py, Array1::from(top_ic.clone())),
    )?;
    dict.set_item(
        "bottom_ic",
        PyArray::from_owned_array(py, Array1::from(bottom_ic.clone())),
    )?;
    dict.set_item("ls_ret", PyArray::from_owned_array(py, Array1::from(ls_ret)))?;
    dict.set_item(
        "hedge_ret",
        PyArray::from_owned_array(py, Array1::from(hedge_ret)),
    )?;
    dict.set_item("n_dates", nt)?;
    dict.set_item("portf_num", portf_num)?;
    dict.set_item("skipped", skipped)?;
    dict.set_item("ic_mean", ic_mean)?;
    dict.set_item("icir", icir)?;
    dict.set_item("annualized_return", mean_annualized_return / 100.0)?;
    dict.set_item("sharpe_ratio", mean_annualized_sharpe_ratio)?;
    dict.set_item("max_drawdown", max_drawdown / 100.0)?;
    dict.set_item("hedge_annualized_return", hedge_annualized_return / 100.0)?;
    dict.set_item(
        "hedge_annualized_sharpe_ratio",
        hedge_annualized_sharpe_ratio,
    )?;
    dict.set_item("hedge_max_drawdown", hedge_max_drawdown / 100.0)?;
    dict.set_item("res_rank_ic_mean", res_ic_mean)?;
    dict.set_item("long_only_ic", long_only_ic)?;
    // 左移后、未做日期筛选的完整日期轴：Python 那边 stock_num 表要 reindex 到它
    // （与 valid_df.columns 同长，比筛选后的日期多出末尾那一天）。
    dict.set_item(
        "dates_left",
        PyList::new(
            py,
            (1..t0).map(|t| PyString::new(py, &dates_all[t])),
        ),
    )?;
    dict.set_item(
        "index_ret_used",
        PyArray::from_owned_array(py, Array1::from(idx_q.clone())),
    )?;
    dict.set_item(
        "ratio_series",
        PyArray::from_owned_array(py, Array1::from(ratio_series)),
    )?;
    dict.set_item("ratio_mean", ratio_mean)?;
    dict.set_item("sign", sign)?;
    Ok(dict.into())
}

/// Python 里 `series.shift(4)`：前 4 个位置变 NaN，其余后移 4 位。
fn shift4(v: &[f64]) -> Vec<f64> {
    let mut out = vec![NAN; v.len()];
    for i in 4..v.len() {
        out[i] = v[i - 4];
    }
    out
}
