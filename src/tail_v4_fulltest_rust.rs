//! tail_v4_fulltest_rust —— fulltest 单因子回测「逐日循环」的 Rust 下沉。
//!
//! 被替换的 Python 段：`design_whatever/fulltest_whatever.py`
//! `PortfolioRetSummary.portf_ret_summary_update_mp` 第 **1804-1882** 行
//! （`for t in range(const_signal_arr.shape[1])` … `portf_ret_arr = np.array(portf_ret).T`）。
//!
//! 入口（生产用）：
//! ```python
//! import rust_pyfunc as rp
//! out = rp.tail_v4_fulltest_rust(
//!     signal, ret, ret_sum, restrict,   # 与 Python 侧同名数组同形（float64）
//!     list(dates), portf_num=10, h_hori=5,
//! )
//! portf_ret_df = pd.DataFrame(out["portf_ret"], index=range(1, portf_num + 1),
//!                             columns=const_signal_df.columns)
//! ic_df = pd.DataFrame(out["ic_values"], index=out["ic_dates"], columns=["ic"])
//! stock_num_df = pd.DataFrame(out["stock_num_list"], index=out["stock_num_dates"],
//!                             columns=["stock_num"])
//! ```
//!
//! # 输入契约（与 Python 段入口完全对齐）
//! 调用方必须在 Python 侧先完成「ahead_rollover 列左移 + 日期区间筛选」（原代码 1679-1725 行），
//! 即传入的 4 个二维数组与 `dates` 就是循环开始那一刻的
//! `const_signal_arr / ret_arr / ret_arr_sum / S_restrict_arr / dates`。
//! - 布局：`stocks_first=True` → (N_stocks, T_dates)（Python 里 `ret_df.values` 的原生形状）；
//!   `stocks_first=False` → (T_dates, N_stocks)（引擎 slot 轴）。默认 `None` = 按 `dates` 长度自动判定。
//! - 全部按 **float64** 读入（引擎里这些 DataFrame 都是 float64；f32 源数据可无损 `.astype(float64)`）。
//!   f32 会改变排序/求和的舍入，逐位一致要求 f64。
//! - `unique_values` 是 Python 侧 `np.unique(const_signal_arr[~np.isnan(...)])` 的结果，可选。
//!   原函数在 `len(unique_values) < 10` 时**提前 return**，因此生产路径上循环内的等值分支不可达；
//!   这里仍按规格照搬（传入 <10 个 unique 值时走 `signal == unique_values[k]` 分支）。
//!
//! # 逐位一致性的四个关键点
//! 1. **`rankdata_nonmiss`**：NaN/±Inf 位置保持 NaN；其余按 scipy `rankdata(method='average')`
//!    的**平均秩**（1-based，只在有限值子集内排名），再除以 `sum(~np.isnan(signal_arr_d))`
//!    ——循环内 signal_arr_d 已被 `~isnan` 过滤，故该除数恒等于 `stocks_num`。
//! 2. **分组边界**：`rank_pct >= k/portf_num` 且（`k < portf_num-1` 时 `< (k+1)/portf_num`，
//!    否则 `<= (k+1)/portf_num`）。本实现把它等价改写为「取满足 `k/P <= rank_pct` 的最大
//!    k∈[0,P-1]」，与上述两条比较式**逐值等价**（k/P 关于 k 单调，故条件区间不重不漏；
//!    `rank_pct == 1.0` 落到最后一组，正是 `<=` 的语义）。
//! 3. **`np.nanmean`**：组内收益无 NaN，`nanmean` 走 `np.mean` → numpy 的
//!    `DOUBLE_pairwise_sum`（`loops_utils.h.src`，PW_BLOCKSIZE=128，8 累加器 + 递归折半）。
//!    本文件 `pairwise_sum` 是它的逐字转写，因此 `sum/len` 与 numpy 逐位相同；空组 → NaN，
//!    与 `portf_ret_df.fillna(0)` 一致后输出 0.0。
//! 4. **IC**：原式 `1 - 6*dot(diff,diff)/(n*(n*n-1))`，其中 `diff = x.argsort().argsort() - y.argsort().argsort()`。
//!    `np.argsort(kind='quicksort')` 在 numpy 2.x 上对 f64 走 **x86-simd-sort（AVX512）**，
//!    并列值的序由 SIMD 算法决定（本机 numpy 2.3.5 + AVX512_SKX），纯 Rust 排序无法复现；
//!    实测真实因子数据上「稳定序数秩」与 numpy 的 IC 在 85%~100% 的 IC 日不同。
//!    故 IC 这里**回调 numpy 自身的 argsort**（每个 IC 日 4 次，占总量 <0.5% 时间），
//!    再在 Rust 里做**整数**点积（int64 点积无舍入，n≤7857 时 n*(n²-1) < 2^53，与 Python 的
//!    整数分母转 f64 完全一致）。
//!
//! 自测入口：`rp.tail_v4_fulltest_rust_selfcheck(...)`（见文件末），配套脚本
//! `tests/selfcheck_tail_v4_fulltest.py`。

use std::time::Instant;

use ndarray::{Array2, ArrayView2};
use numpy::{PyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyString};

/// numpy `pairwise_sum` 的块大小（loops_utils.h.src）。
const PW_BLOCKSIZE: usize = 128;

// ============================================================================
// numpy DOUBLE_pairwise_sum 的逐字转写（numpy/_core/src/umath/loops_utils.h.src）
// ============================================================================

/// numpy `DOUBLE_pairwise_sum(a, n, stride=1)` 的等价实现。
///
/// 逐位一致的关键：8 路累加器 + `((r0+r1)+(r2+r3))+((r4+r5)+(r6+r7))` 的合并次序，
/// 以及 `n2 -= n2 % 8` 的递归切分点，全部照抄；Rust 不会做浮点重结合，故结果一致。
pub fn pairwise_sum(a: &[f64]) -> f64 {
    pairwise_sum_range(a, 0, a.len())
}

fn pairwise_sum_range(a: &[f64], i0: usize, n: usize) -> f64 {
    if n < 8 {
        // numpy 从 -0.0 起累加（保留「全 -0」的符号）
        let mut res = -0.0f64;
        for i in 0..n {
            res += a[i0 + i];
        }
        res
    } else if n <= PW_BLOCKSIZE {
        let mut r = [
            a[i0],
            a[i0 + 1],
            a[i0 + 2],
            a[i0 + 3],
            a[i0 + 4],
            a[i0 + 5],
            a[i0 + 6],
            a[i0 + 7],
        ];
        let mut i = 8usize;
        let end = n - (n % 8);
        while i < end {
            r[0] += a[i0 + i];
            r[1] += a[i0 + i + 1];
            r[2] += a[i0 + i + 2];
            r[3] += a[i0 + i + 3];
            r[4] += a[i0 + i + 4];
            r[5] += a[i0 + i + 5];
            r[6] += a[i0 + i + 6];
            r[7] += a[i0 + i + 7];
            i += 8;
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        while i < n {
            res += a[i0 + i];
            i += 1;
        }
        res
    } else {
        let mut n2 = n / 2;
        n2 -= n2 % 8;
        pairwise_sum_range(a, i0, n2) + pairwise_sum_range(a, i0 + n2, n - n2)
    }
}

// ============================================================================
// 面板访问器：兼容 (N_stocks, T) 与 (T, N_stocks) 两种布局
// ============================================================================

struct Panel<'a> {
    view: ArrayView2<'a, f64>,
    stocks_first: bool,
    n_stocks: usize,
}

impl<'a> Panel<'a> {
    /// 把第 t 天（日期轴下标）的横截面按股票顺序拷进 `out`。
    #[inline]
    fn gather_into(&self, t: usize, out: &mut Vec<f64>) {
        out.clear();
        if out.capacity() < self.n_stocks {
            out.reserve(self.n_stocks);
        }
        if self.stocks_first {
            for v in self.view.column(t).iter() {
                out.push(*v);
            }
        } else {
            for v in self.view.row(t).iter() {
                out.push(*v);
            }
        }
        debug_assert_eq!(out.len(), self.n_stocks);
    }
}

/// 判定 (N,T) / (T,N) 布局；`hint=None` 时按 `dates` 长度自动判定。
fn resolve_layout(
    name: &str,
    view: &ArrayView2<f64>,
    n_dates: usize,
    hint: Option<bool>,
) -> PyResult<bool> {
    let (r, c) = (view.nrows(), view.ncols());
    match hint {
        Some(true) => {
            if c != n_dates {
                return Err(PyValueError::new_err(format!(
                    "{name} 形状 ({r},{c}) 与 dates 长度 {n_dates} 不符（stocks_first=True 要求列数=日期数）"
                )));
            }
            Ok(true)
        }
        Some(false) => {
            if r != n_dates {
                return Err(PyValueError::new_err(format!(
                    "{name} 形状 ({r},{c}) 与 dates 长度 {n_dates} 不符（stocks_first=False 要求行数=日期数）"
                )));
            }
            Ok(false)
        }
        None => {
            if r == n_dates && c == n_dates {
                return Err(PyValueError::new_err(format!(
                    "{name} 形状 ({r},{c}) 两个维度都等于 dates 长度，无法自动判定布局，请显式传 stocks_first"
                )));
            }
            if r == n_dates {
                Ok(false)
            } else if c == n_dates {
                Ok(true)
            } else {
                Err(PyValueError::new_err(format!(
                    "{name} 形状 ({r},{c}) 没有任何一维等于 dates 长度 {n_dates}"
                )))
            }
        }
    }
}

// ============================================================================
// IC：回调 numpy 的 argsort（并列值序由 numpy/AVX512 决定，必须用它自己）
// ============================================================================

/// `calc_spearman_correlation(x, y)`（szalpha/research_base/calc_func.py）的等价实现。
///
/// `x.argsort().argsort()` / `y.argsort().argsort()` 交给 numpy 执行（逐位复现并列值的序），
/// `np.dot(diff, diff)` 是 int64 精确整数点积，在 Rust 里用 i64 累加（无舍入），
/// 最后的 `1 - 6*dot/(n*(n*n-1))` 用 f64 按 Python 的求值顺序计算。
fn calc_spearman_correlation_np(py: Python<'_>, x: &[f64], y: &[f64]) -> PyResult<f64> {
    let n = x.len();
    // n<2 时 Python 会算 0/0 → nan（np.float64/int 除法）
    if n < 2 {
        return Ok(f64::NAN);
    }
    let numpy = py.import("numpy")?;
    let xa = PyArray1::from_slice(py, x);
    let ya = PyArray1::from_slice(py, y);
    let xx = xa.call_method0("argsort")?.call_method0("argsort")?;
    let yy = ya.call_method0("argsort")?.call_method0("argsort")?;
    let diff = xx.call_method1("__sub__", (yy,))?;
    let dot = numpy.call_method1("dot", (diff, diff))?;
    let dot_i: i64 = dot.extract()?;
    let nf = n as f64;
    Ok(1.0 - 6.0 * (dot_i as f64) / (nf * (nf * nf - 1.0)))
}

// ============================================================================
// 核心实现
// ============================================================================

/// 循环段全部输出。
pub struct FulltestOut {
    pub portf_num: usize,
    pub n_dates: usize,
    /// (portf_num, T) 行主序；空组/股票数不足日均已按 `fillna(0)` 填 0.0
    pub portf_ret: Vec<f64>,
    pub ic_values: Vec<f64>,
    pub ic_dates: Vec<String>,
    pub stock_num: Vec<usize>,
    pub stock_num_dates: Vec<String>,
}

#[allow(clippy::too_many_arguments)]
fn run_core(
    py: Python<'_>,
    signal: &ArrayView2<f64>,
    ret: &ArrayView2<f64>,
    ret_sum: &ArrayView2<f64>,
    restrict: &ArrayView2<f64>,
    dates: &[String],
    portf_num: usize,
    h_hori: usize,
    unique_values: Option<&[f64]>,
    stocks_first: Option<bool>,
) -> PyResult<FulltestOut> {
    if h_hori == 0 {
        return Err(PyValueError::new_err("h_hori 必须 > 0（Python 侧 t % H_hori 会 ZeroDivisionError）"));
    }
    let n_dates = dates.len();
    let sf = resolve_layout("signal", signal, n_dates, stocks_first)?;
    let n_stocks = if sf { signal.nrows() } else { signal.ncols() };
    let want = if sf { (n_stocks, n_dates) } else { (n_dates, n_stocks) };
    fn check_shape(name: &str, v: &ArrayView2<f64>, want: (usize, usize)) -> PyResult<()> {
        if (v.nrows(), v.ncols()) != want {
            return Err(PyValueError::new_err(format!(
                "{name} 形状 ({},{}) 与 signal 的 {want:?} 不一致",
                v.nrows(),
                v.ncols()
            )));
        }
        Ok(())
    }
    check_shape("ret", ret, want)?;
    check_shape("ret_sum", ret_sum, want)?;
    check_shape("restrict", restrict, want)?;

    let uniq: &[f64] = unique_values.unwrap_or(&[]);
    // None = 调用方不传 unique_values，按生产路径的保证处理：原函数在 len(unique_values) < 10
    // 时已提前 return，因此循环内等值分支不可达 → 走平均秩分支。
    let use_equality_branch = match unique_values {
        Some(u) => u.len() < 10,
        None => false,
    };
    if use_equality_branch && uniq.len() < portf_num {
        return Err(PyValueError::new_err(format!(
            "unique_values 只有 {} 个，少于 portf_num={}（生产路径上该分支不可达）",
            uniq.len(),
            portf_num
        )));
    }

    let sig_p = Panel { view: signal.clone(), stocks_first: sf, n_stocks };
    let ret_p = Panel { view: ret.clone(), stocks_first: sf, n_stocks };
    let rs_p = Panel { view: ret_sum.clone(), stocks_first: sf, n_stocks };
    let rstr_p = Panel { view: restrict.clone(), stocks_first: sf, n_stocks };

    // 复用缓冲（热路径零分配）
    let mut sig_held: Vec<f64> = Vec::with_capacity(n_stocks);
    let mut rstr_held: Vec<f64> = Vec::with_capacity(n_stocks);
    let mut ret_row: Vec<f64> = Vec::with_capacity(n_stocks);
    let mut rsum_row: Vec<f64> = Vec::with_capacity(n_stocks);
    let mut sig_d: Vec<f64> = Vec::with_capacity(n_stocks);
    let mut ret_d: Vec<f64> = Vec::with_capacity(n_stocks);
    let mut ic_x: Vec<f64> = Vec::with_capacity(n_stocks);
    let mut valid_idx: Vec<u32> = Vec::with_capacity(n_stocks);
    let mut pairs: Vec<(f64, u32)> = Vec::with_capacity(n_stocks);
    let mut group_of: Vec<u32> = vec![u32::MAX; n_stocks.max(1)];
    let mut group_bufs: Vec<Vec<f64>> = (0..portf_num)
        .map(|_| Vec::with_capacity(n_stocks / portf_num.max(1) + 16))
        .collect();

    let mut portf_ret = vec![0.0f64; portf_num * n_dates];
    let mut ic_values: Vec<f64> = Vec::new();
    let mut ic_dates: Vec<String> = Vec::new();
    let mut stock_num: Vec<usize> = Vec::new();
    let mut stock_num_dates: Vec<String> = Vec::new();

    let pnum_f = portf_num as f64;

    for t in 0..n_dates {
        // 原代码：持仓在 H_hori 天内保持不变（signal/restrict 取最近一个 t%H_hori==0 的列）
        if t % h_hori == 0 {
            sig_p.gather_into(t, &mut sig_held);
            rstr_p.gather_into(t, &mut rstr_held);
        }
        ret_p.gather_into(t, &mut ret_row);

        // b1 & b2 & b3：signal 非 NaN、ret 非 NaN、restrict == 0（NaN == 0 为假）
        valid_idx.clear();
        sig_d.clear();
        ret_d.clear();
        for i in 0..n_stocks {
            let s = sig_held[i];
            if s.is_nan() {
                continue;
            }
            let r = ret_row[i];
            if r.is_nan() {
                continue;
            }
            if rstr_held[i] != 0.0 {
                continue;
            }
            valid_idx.push(i as u32);
            sig_d.push(s);
            ret_d.push(r);
        }
        let stocks_num = sig_d.len();

        // IC：每 H_hori 天一次（在股票数检查之前，与原代码一致）
        if (t + 1) % h_hori == 0 {
            rs_p.gather_into(t, &mut rsum_row);
            ic_x.clear();
            for &i in valid_idx.iter() {
                ic_x.push(rsum_row[i as usize]);
            }
            let ic = calc_spearman_correlation_np(py, &ic_x, &sig_d)?;
            ic_values.push(ic);
            ic_dates.push(dates[t].clone());
        }

        // 有效股票不足 portf_num → 该日全部组合记 0
        if stocks_num < portf_num {
            continue;
        }
        stock_num.push(stocks_num);
        stock_num_dates.push(dates[t].clone());

        if use_equality_branch {
            // 照搬 `condition = signal_arr_d == unique_values[k]`
            for k in 0..portf_num {
                let target = uniq[k];
                let buf = &mut group_bufs[k];
                buf.clear();
                for i in 0..stocks_num {
                    if sig_d[i] == target {
                        buf.push(ret_d[i]);
                    }
                }
            }
        } else {
            // rankdata_nonmiss(signal_arr_d) / stocks_num
            pairs.clear();
            for i in 0..stocks_num {
                let v = sig_d[i];
                if v.is_finite() {
                    pairs.push((v, i as u32));
                }
            }
            pairs.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

            group_of[..stocks_num].fill(u32::MAX);
            let nf = stocks_num as f64;
            let mut start = 0usize;
            while start < pairs.len() {
                let value = pairs[start].0;
                let mut end = start + 1;
                while end < pairs.len() && pairs[end].0 == value {
                    end += 1;
                }
                // scipy rankdata(method='average')：并列组取名次的平均
                let avg_rank = (start + 1 + end) as f64 / 2.0;
                let pct = avg_rank / nf;
                // 等价于「>= k/P 且 (< 或 <=) (k+1)/P」两条比较式
                let mut k = 0usize;
                while k + 1 < portf_num && pct >= ((k + 1) as f64) / pnum_f {
                    k += 1;
                }
                for p in &pairs[start..end] {
                    group_of[p.1 as usize] = k as u32;
                }
                start = end;
            }
            for b in group_bufs.iter_mut() {
                b.clear();
            }
            // 按股票原顺序入组（与 ret_arr_d[condition] 的元素顺序一致 → pairwise_sum 才能逐位对齐）
            for i in 0..stocks_num {
                let g = group_of[i];
                if g != u32::MAX {
                    group_bufs[g as usize].push(ret_d[i]);
                }
            }
        }

        for (k, buf) in group_bufs.iter().enumerate() {
            let v = if buf.is_empty() {
                0.0
            } else {
                pairwise_sum(buf) / buf.len() as f64
            };
            portf_ret[k * n_dates + t] = v;
        }
    }

    Ok(FulltestOut {
        portf_num,
        n_dates,
        portf_ret,
        ic_values,
        ic_dates,
        stock_num,
        stock_num_dates,
    })
}

// ============================================================================
// PyO3 入口
// ============================================================================

/// fulltest 逐日回测循环（`fulltest_whatever.py` 1804-1882 行）的 Rust 版。
///
/// 返回 dict：
/// - `portf_ret`      : (portf_num, T) float64 ndarray（等价 `portf_ret_df.fillna(0).values`）
/// - `ic_values`      : list[float]（等价 `ic_value_list`）
/// - `ic_dates`       : list[str]（等价 `ic_date_list`）
/// - `stock_num_list` : list[int]（等价 `stock_num_list`）
/// - `stock_num_dates`: list[str]（等价 `stock_num_date_list`）
/// - `portf_num` / `n_dates`：回显
#[pyfunction]
#[pyo3(signature = (signal, ret, ret_sum, restrict, dates, portf_num=10, h_hori=5, unique_values=None, stocks_first=None))]
pub fn tail_v4_fulltest_rust(
    py: Python<'_>,
    signal: PyReadonlyArray2<f64>,
    ret: PyReadonlyArray2<f64>,
    ret_sum: PyReadonlyArray2<f64>,
    restrict: PyReadonlyArray2<f64>,
    dates: Vec<String>,
    portf_num: usize,
    h_hori: usize,
    unique_values: Option<PyReadonlyArray1<f64>>,
    stocks_first: Option<bool>,
) -> PyResult<PyObject> {
    let uniq_vec: Option<Vec<f64>> = unique_values.as_ref().map(|a| a.as_array().to_vec());
    let out = run_core(
        py,
        &signal.as_array(),
        &ret.as_array(),
        &ret_sum.as_array(),
        &restrict.as_array(),
        &dates,
        portf_num,
        h_hori,
        uniq_vec.as_deref(),
        stocks_first,
    )?;

    let arr = Array2::from_shape_vec((out.portf_num, out.n_dates), out.portf_ret)
        .map_err(|e| PyValueError::new_err(format!("portf_ret 形状组装失败: {e}")))?;
    let py_arr = PyArray::from_owned_array(py, arr);

    let dict = PyDict::new(py);
    dict.set_item("portf_ret", py_arr)?;
    dict.set_item("ic_values", PyList::new(py, out.ic_values.iter()))?;
    dict.set_item(
        "ic_dates",
        PyList::new(py, out.ic_dates.iter().map(|s| PyString::new(py, s))),
    )?;
    dict.set_item(
        "stock_num_list",
        PyList::new(py, out.stock_num.iter().map(|&v| v as i64)),
    )?;
    dict.set_item(
        "stock_num_dates",
        PyList::new(py, out.stock_num_dates.iter().map(|s| PyString::new(py, s))),
    )?;
    dict.set_item("portf_num", out.portf_num)?;
    dict.set_item("n_dates", out.n_dates)?;
    Ok(dict.into())
}

// ============================================================================
// 自测入口：Python 参考实现 vs Rust 实现，逐位对账 + 计时
// ============================================================================

fn cmp_flat(a: &[f64], b: &[f64]) -> (usize, f64, Option<usize>) {
    let mut bad = 0usize;
    let mut maxabs = 0.0f64;
    let mut first: Option<usize> = None;
    for i in 0..a.len().min(b.len()) {
        let (x, y) = (a[i], b[i]);
        let same = x.to_bits() == y.to_bits() || (x.is_nan() && y.is_nan());
        if !same {
            bad += 1;
            if first.is_none() {
                first = Some(i);
            }
            let d = (x - y).abs();
            if d.is_nan() || d > maxabs {
                maxabs = d;
            }
        }
    }
    (bad, maxabs, first)
}

fn get_f64_array(d: &PyDict, key: &str) -> PyResult<Vec<f64>> {
    let item = d.get_item(key).ok_or_else(|| {
        PyValueError::new_err(format!("Python 参考实现返回的 dict 缺少键 `{key}`"))
    })?;
    let ro: PyReadonlyArray2<f64> = item.extract().map_err(|_| {
        PyValueError::new_err(format!("键 `{key}` 不是 float64 二维 ndarray"))
    })?;
    Ok(ro.as_array().iter().copied().collect())
}

fn get_str_list(d: &PyDict, key: &str) -> PyResult<Vec<String>> {
    let item = d.get_item(key).ok_or_else(|| {
        PyValueError::new_err(format!("Python 参考实现返回的 dict 缺少键 `{key}`"))
    })?;
    item.extract::<Vec<String>>()
}

/// 对同一份输入，把 Python 参考实现与 Rust 实现逐位对账，并打印各自耗时。
///
/// `py_ref` 必须是 `py_ref(signal, ret, ret_sum, restrict, dates, portf_num, h_hori, unique_values)`
/// 形式的可调用对象，返回含 `portf_ret`(P,T float64 ndarray) / `ic_values` / `ic_dates` /
/// `stock_num_list` / `stock_num_dates` 的 dict。
#[pyfunction]
#[pyo3(signature = (py_ref, signal, ret, ret_sum, restrict, dates, portf_num=10, h_hori=5, unique_values=None, stocks_first=None, repeats=1))]
#[allow(clippy::too_many_arguments)]
pub fn tail_v4_fulltest_rust_selfcheck(
    py: Python<'_>,
    py_ref: &PyAny,
    signal: &PyAny,
    ret: &PyAny,
    ret_sum: &PyAny,
    restrict: &PyAny,
    dates: Vec<String>,
    portf_num: usize,
    h_hori: usize,
    unique_values: Option<&PyAny>,
    stocks_first: Option<bool>,
    repeats: usize,
) -> PyResult<String> {
    let sig_ro: PyReadonlyArray2<f64> = signal.extract()?;
    let ret_ro: PyReadonlyArray2<f64> = ret.extract()?;
    let rs_ro: PyReadonlyArray2<f64> = ret_sum.extract()?;
    let rstr_ro: PyReadonlyArray2<f64> = restrict.extract()?;
    let uniq_vec: Option<Vec<f64>> = match unique_values {
        Some(o) => Some(o.extract::<PyReadonlyArray1<f64>>()?.as_array().to_vec()),
        None => None,
    };

    // ---- Python 参考实现 ----
    let none = py.None();
    let uv_obj: &PyAny = match unique_values {
        Some(o) => o,
        None => none.as_ref(py),
    };
    let dates_list = PyList::new(py, dates.iter().map(|s| PyString::new(py, s)));
    let mut py_secs = f64::INFINITY;
    let mut ref_dict: Option<&PyDict> = None;
    let reps_py = repeats.max(1);
    for _ in 0..reps_py {
        let t0 = Instant::now();
        let r = py_ref.call1((
            signal,
            ret,
            ret_sum,
            restrict,
            dates_list,
            portf_num,
            h_hori,
            uv_obj,
        ))?;
        let el = t0.elapsed().as_secs_f64();
        if el < py_secs {
            py_secs = el;
        }
        ref_dict = Some(r.downcast::<PyDict>().map_err(|_| {
            PyValueError::new_err("Python 参考实现必须返回 dict")
        })?);
    }
    let ref_dict = ref_dict.unwrap();

    // ---- Rust 实现 ----
    let mut rust_secs = f64::INFINITY;
    let mut rust_out = None;
    for _ in 0..reps_py {
        let t0 = Instant::now();
        let o = run_core(
            py,
            &sig_ro.as_array(),
            &ret_ro.as_array(),
            &rs_ro.as_array(),
            &rstr_ro.as_array(),
            &dates,
            portf_num,
            h_hori,
            uniq_vec.as_deref(),
            stocks_first,
        )?;
        let el = t0.elapsed().as_secs_f64();
        if el < rust_secs {
            rust_secs = el;
        }
        rust_out = Some(o);
    }
    let rust_out = rust_out.unwrap();

    // ---- 对账 ----
    let mut lines: Vec<String> = Vec::new();
    let mut all_ok = true;
    lines.push(format!(
        "[tail_v4_fulltest_rust] 输入: dates={} portf_num={} H_hori={} stocks_first={:?}",
        dates.len(),
        portf_num,
        h_hori,
        stocks_first
    ));

    // portf_ret
    let py_pr = get_f64_array(ref_dict, "portf_ret")?;
    let (bad, maxabs, first) = cmp_flat(&rust_out.portf_ret, &py_pr);
    let shape_py = ref_dict
        .get_item("portf_ret")
        .and_then(|v| v.getattr("shape").ok())
        .map(|s| s.to_string())
        .unwrap_or_default();
    if rust_out.portf_ret.len() != py_pr.len() {
        all_ok = false;
        lines.push(format!(
            "  portf_ret  长度不一致: rust={} python={}",
            rust_out.portf_ret.len(),
            py_pr.len()
        ));
    } else if bad == 0 {
        lines.push(format!(
            "  portf_ret  ({}, {}) 共 {} 个值: 逐位一致 ✓  python.shape={}",
            rust_out.portf_num,
            rust_out.n_dates,
            py_pr.len(),
            shape_py
        ));
    } else {
        all_ok = false;
        lines.push(format!(
            "  portf_ret  ({}, {}): 不一致 ✗  {} / {} 个值不同, max|diff|={:.3e}, 首个下标={:?}",
            rust_out.portf_num,
            rust_out.n_dates,
            bad,
            py_pr.len(),
            maxabs,
            first
        ));
    }

    // ic_values
    let py_ic: Vec<f64> = ref_dict
        .get_item("ic_values")
        .ok_or_else(|| PyValueError::new_err("参考实现 dict 缺少 `ic_values`"))?
        .extract()?;
    let (bad_ic, max_ic, first_ic) = cmp_flat(&rust_out.ic_values, &py_ic);
    if rust_out.ic_values.len() != py_ic.len() {
        all_ok = false;
        lines.push(format!(
            "  ic_values  长度不一致: rust={} python={}",
            rust_out.ic_values.len(),
            py_ic.len()
        ));
    } else if bad_ic == 0 {
        lines.push(format!("  ic_values  ({}) : 逐位一致 ✓", py_ic.len()));
    } else {
        all_ok = false;
        lines.push(format!(
            "  ic_values  ({}) : 不一致 ✗  {} 个不同, max|diff|={:.3e}, 首个下标={:?}",
            py_ic.len(),
            bad_ic,
            max_ic,
            first_ic
        ));
    }

    // ic_dates / stock_num / stock_num_dates
    let py_icd = get_str_list(ref_dict, "ic_dates")?;
    if py_icd == rust_out.ic_dates {
        lines.push(format!("  ic_dates   ({}) : 一致 ✓", py_icd.len()));
    } else {
        all_ok = false;
        lines.push(format!(
            "  ic_dates   不一致 ✗  rust={} python={}（长度 {} vs {}）",
            rust_out.ic_dates.len(),
            py_icd.len(),
            rust_out.ic_dates.len(),
            py_icd.len()
        ));
    }
    let py_sn: Vec<i64> = ref_dict
        .get_item("stock_num_list")
        .ok_or_else(|| PyValueError::new_err("参考实现 dict 缺少 `stock_num_list`"))?
        .extract()?;
    let rust_sn: Vec<i64> = rust_out.stock_num.iter().map(|&v| v as i64).collect();
    if py_sn == rust_sn {
        lines.push(format!("  stock_num  ({}) : 一致 ✓", py_sn.len()));
    } else {
        all_ok = false;
        lines.push(format!(
            "  stock_num  不一致 ✗  rust={} python={}",
            rust_sn.len(),
            py_sn.len()
        ));
    }
    let py_snd = get_str_list(ref_dict, "stock_num_dates")?;
    if py_snd == rust_out.stock_num_dates {
        lines.push(format!("  stock_num_dates ({}) : 一致 ✓", py_snd.len()));
    } else {
        all_ok = false;
        lines.push("  stock_num_dates 不一致 ✗".to_string());
    }

    let speedup = if rust_secs > 0.0 { py_secs / rust_secs } else { f64::INFINITY };
    lines.push(format!(
        "  耗时: python={:.3}s  rust={:.3}s  加速={:.1}x  (repeats={}, 取最快)",
        py_secs, rust_secs, speedup, reps_py
    ));
    lines.push(if all_ok {
        "  结论: 逐位一致 ✓".to_string()
    } else {
        "  结论: 存在不一致 ✗".to_string()
    });
    Ok(lines.join("\n"))
}
