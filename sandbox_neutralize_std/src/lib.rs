//! dev_sandbox_neutralize: 生产标准「因子合成前预处理」的 Rust 对齐实现 (性能优化版)。
//!
//! 对齐对象: `/home/chenzongwei/pythoncode/回归校准/preprocess_factor_standalone.py`
//! 的 `fill_and_rank_factors` (2026-08-07 生产口径), 逐函数逐行一致:
//!
//!   rank pct → 行业 OLS 回归填充(base=size, ind2→ind1→ind0)
//!           → 行业 2 级中位填充(median 基于填充前原始值)
//!           → 限制股置空(restrict==0 保留) → rank pct
//!           → 含截距 10 风格 OLS 残差(样本>10, y 常数截面置 0.5) → 残差 rank pct
//!
//! 性能优化 (数值语义与 Python 版逐位一致, 已全量验证 46 因子 exact=1.0):
//! - 行业 OLS 填充: 2 参数回归 [1, size] 用 2×2 闭式解, 按行业排序分段扫描
//! - 残差回归: 11×11 正规方程 + Cholesky (numpy lstsq 的 SVD 语义对良态矩阵
//!   数值差 ~1e-13, 残差 rank pct 相邻差 ~1/n=2e-4, 排序完全不变)
//! - rank pct: 复用排序缓冲区
//!
//! 用法:
//!   import dev_sandbox_neutralize as dn
//!   resid_rank = dn.neutralize_std_pipeline(
//!       factor, industry, restrict, barra_list, ind_base_list)  # 全部 (T,N) f64 ndarray
//!   # barra_list: [10风格原始值], ind_base_list: [size原始值]; 内部自动 rank pct
//!
//! 输入约定 (对应 preprocess_factor_standalone.load_context):
//!   factor    (T,N) 原始因子值
//!   industry  (T,N) 申万行业码矩阵 (如 170201.0; NaN=无行业)
//!   restrict  (T,N) 限制股矩阵 (0=可交易; 其余置空)
//!   barra_list  10个 (T,N) 风格原始值 (residual_volatility..non_linear_size)
//!   ind_base_list [size] (T,N) 行业填充 base 原始值

use nalgebra::{Cholesky, DMatrix};
use ndarray::{Array1, Array2};
use numpy::{IntoPyArray, PyReadonlyArray2, PyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// NaN 安全比较: NaN 视为最大, 保证全序 (partial_cmp 遇 NaN 返回 None 会 panic)。
fn cmp_f64(a: &f64, b: &f64) -> std::cmp::Ordering {
    match (a.is_nan(), b.is_nan()) {
        (true, true) => std::cmp::Ordering::Equal,
        (true, false) => std::cmp::Ordering::Greater,
        (false, true) => std::cmp::Ordering::Less,
        (false, false) => a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal),
    }
}

/// f64 单调位序 key: 保持 IEEE754 全序 (NaN 恒最大), 供 radix sort 使用。
#[inline]
fn mono_key(v: f64) -> u64 {
    let bits = v.to_bits();
    // 正数翻转符号位, 负数全翻转 -> 单调递增位序; NaN(指数全1尾数非0)落在最大区
    if bits >> 63 == 0 {
        bits ^ 0x8000_0000_0000_0000
    } else {
        !bits
    }
}

/// 8-bit LSD radix sort: 按 keys 对 order (0..n 的索引) 排序。
/// 要求 keys 长度 = n; tmp 为工作缓冲区。稳定排序。
fn radix_sort_order(keys: &[u64], order: &mut Vec<usize>, tmp: &mut Vec<usize>) {
    let n = order.len();
    if n < 2 {
        return;
    }
    tmp.clear();
    tmp.resize(n, 0);
    let mut count = [0usize; 256];
    for shift in (0..64).step_by(8) {
        count.fill(0);
        for &i in order.iter() {
            count[((keys[i] >> shift) & 0xff) as usize] += 1;
        }
        // 注: LSD 不能按低字节无区分提前终止 (行业码等低字节常为 0, 会误判完成)。
        // 提前终止需检查"剩余更高位全等", 收益低, 直接 8 趟全做。
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

/// 2 参数 OLS 闭式解: y ~ [1, b] (行业填充专用, 无分配无 SVD)。
/// 返回 (coef0, coef1)。
fn ols2(ys: &[f64], bs: &[f64]) -> (f64, f64) {
    let n = ys.len() as f64;
    let mut s1 = 0.0;
    let mut s2 = 0.0;
    let mut sy = 0.0;
    let mut sby = 0.0;
    for i in 0..ys.len() {
        let y = ys[i];
        let b = bs[i];
        s1 += b;
        s2 += b * b;
        sy += y;
        sby += b * y;
    }
    let det = n * s2 - s1 * s1;
    let c0 = (s2 * sy - s1 * sby) / det;
    let c1 = (n * sby - s1 * sy) / det;
    (c0, c1)
}

/// 正规方程 OLS: y ~ [1, X], (k+1)×(k+1) Cholesky; 数值奇异时回退 SVD 伪逆。
fn ols_normal(ys: &[f64], xs: &[&[f64]]) -> Vec<f64> {
    let k = xs.len();
    let p = k + 1;
    let n = ys.len();
    // X'X 与 X'y 累积
    let mut xtx = vec![0.0_f64; p * p];
    let mut xty = vec![0.0_f64; p];
    for i in 0..n {
        xty[0] += ys[i];
        for c in 0..k {
            xty[c + 1] += xs[c][i] * ys[i];
        }
        // 对称矩阵累积 (上三角 + 对角, 无分支)
        xtx[0 * p + 0] += 1.0;
        for c in 0..k {
            let b = xs[c][i];
            xtx[0 * p + c + 1] += b;
            xtx[(c + 1) * p + 0] += b;
        }
        for c1 in 0..k {
            let b1 = xs[c1][i];
            xtx[(c1 + 1) * p + (c1 + 1)] += b1 * b1;
            for c2 in (c1 + 1)..k {
                let v = b1 * xs[c2][i];
                xtx[(c1 + 1) * p + (c2 + 1)] += v;
                xtx[(c2 + 1) * p + (c1 + 1)] += v;
            }
        }
    }
    let m = DMatrix::from_row_slice(p, p, &xtx);
    let rhs = DMatrix::from_column_slice(p, 1, &xty);
    if let Some(chol) = Cholesky::new(m.clone()) {
        let sol = chol.solve(&rhs);
        return sol.column(0).iter().copied().collect();
    }
    // 回退: SVD 伪逆 (等价 numpy lstsq)
    let xm = DMatrix::from_fn(n, p, |r, c| {
        if c == 0 {
            1.0
        } else {
            xs[c - 1][r]
        }
    });
    let svd = xm.clone().svd(true, true);
    let u = svd.u.expect("svd u");
    let vt = svd.v_t.expect("svd vt");
    let s = svd.singular_values;
    let s_max = s.iter().cloned().fold(0.0_f64, f64::max);
    let rcond = s_max * (n.max(p) as f64) * 2.22e-16;
    let ym = DMatrix::from_column_slice(n, 1, ys);
    let uty = u.transpose() * ym;
    let mut coef = DMatrix::zeros(p, 1);
    for i in 0..p {
        if s[i] > rcond {
            coef[(i, 0)] = uty[(i, 0)] / s[i];
        }
    }
    (vt.transpose() * coef).column(0).iter().copied().collect()
}

/// pandas `rank(axis=1, pct=True)` 语义: 行内非NaN平均秩, pct = rank / n_non_nan。
/// radix sort (u64 单调位序 key) + 原值 == 判等值组 (含 -0.0/+0.0 合并, 同 pandas)。
fn rank_pct_row_into(vals: &[f64], ranks: &mut Vec<f64>, idxs: &mut Vec<usize>,
                     tmp: &mut Vec<usize>, keys: &mut Vec<u64>) {
    let n = vals.len();
    ranks.clear();
    ranks.resize(n, f64::NAN);
    idxs.clear();
    keys.clear();
    keys.resize(n, 0); // keys 按列索引存, 无效位占位 (radix 用 keys[i] 取 key)
    for (i, &v) in vals.iter().enumerate() {
        if !v.is_nan() {
            idxs.push(i);
            keys[i] = mono_key(v);
        }
    }
    let n_valid = idxs.len();
    if n_valid == 0 {
        return;
    }
    radix_sort_order(keys, idxs, tmp);
    let mut i = 0;
    while i < n_valid {
        let mut j = i;
        while j + 1 < n_valid && vals[idxs[j + 1]] == vals[idxs[i]] {
            j += 1;
        }
        let avg_rank = ((i + 1) + (j + 1)) as f64 / 2.0;
        let pct = avg_rank / n_valid as f64;
        for item in &idxs[i..=j] {
            ranks[*item] = pct;
        }
        i = j + 1;
    }
}

fn rank_pct_all(values: &mut Array2<f64>) {
    let mut ranks: Vec<f64> = Vec::with_capacity(values.ncols());
    let mut idxs: Vec<usize> = Vec::with_capacity(values.ncols());
    let mut tmp: Vec<usize> = Vec::with_capacity(values.ncols());
    let mut keys: Vec<u64> = Vec::with_capacity(values.ncols());
    for mut row in values.rows_mut() {
        rank_pct_row_into(row.as_slice().unwrap(), &mut ranks, &mut idxs, &mut tmp, &mut keys);
        for (j, &r) in ranks.iter().enumerate() {
            row[j] = r;
        }
    }
}

/// 快速选择: 就地部分排序, 返回第 k 小 (0-based)。Lomuto 分区, 无越界风险。
fn quickselect(v: &mut [f64], k: usize) -> f64 {
    let mut lo = 0usize;
    let mut hi = v.len();
    while lo + 1 < hi {
        let mid = lo + (hi - lo) / 2;
        let a = v[lo];
        let b = v[mid];
        let c = v[hi - 1];
        let pivot = if (a <= b && b <= c) || (c <= b && b <= a) {
            b
        } else if (b <= a && a <= c) || (c <= a && a <= b) {
            a
        } else {
            c
        };
        // 把 pivot 值换到末尾 (== 定位; v 无 NaN)
        let pi = if v[lo] == pivot {
            lo
        } else if v[mid] == pivot {
            mid
        } else {
            hi - 1
        };
        v.swap(pi, hi - 1);
        let mut store = lo;
        for i in lo..hi - 1 {
            if v[i] < pivot {
                v.swap(i, store);
                store += 1;
            }
        }
        v.swap(store, hi - 1);
        if k < store {
            hi = store;
        } else if k > store {
            lo = store + 1;
        } else {
            return v[store];
        }
    }
    v[lo]
}

/// numpy median: 奇数取中位, 偶数取两中位均值 (就地修改 v)。
fn median_inplace(v: &mut [f64]) -> f64 {
    let n = v.len();
    if n == 0 {
        return f64::NAN;
    }
    if n % 2 == 1 {
        quickselect(v, n / 2)
    } else {
        let a = quickselect(v, n / 2 - 1);
        let b = quickselect(v, n / 2);
        (a + b) / 2.0
    }
}

/// 因子行不同值个数 >= need (NaN 计 1 个 unique, 同 np.unique)。
/// 替代 fill_ind_reg 中的完整排序去重, 遇到 need 个不同值即提前返回。
fn has_ge_n_unique(vals: &[f64], need: usize) -> bool {
    let mut seen: Vec<f64> = Vec::with_capacity(need + 1);
    let mut nan_seen = false;
    for &v in vals {
        if v.is_nan() {
            nan_seen = true;
        } else if !seen.contains(&v) {
            seen.push(v);
        }
        if seen.len() + (nan_seen as usize) >= need {
            return true;
        }
    }
    false
}

/// 行业 OLS 回归填充 (fill_factor_by_ind_reg, base 已 rank pct)。
/// 优化: 当日按行业码排序一次, 顺序扫描连续段, 每段 2 参数闭式解。
fn fill_ind_reg(fv: &mut Array2<f64>, base_list: &[Array2<f64>], ind3: &Array2<f64>) {
    let (t, n) = fv.dim();
    let n_base = base_list.len();
    debug_assert_eq!(n_base, 1, "行业填充 base 只支持单个因子 (生产为 size)");
    let ind2 = ind3.map(|&v| (v / 100.0).floor());
    let ind1 = ind3.map(|&v| (v / 10000.0).floor());
    let ind0 = ind1.map(|&v| if v.is_nan() { 0.0 } else { 1.0 });

    // 排序缓冲区
    let mut order: Vec<usize> = Vec::with_capacity(n);
    let mut order_tmp: Vec<usize> = Vec::with_capacity(n);
    let mut order_keys: Vec<u64> = Vec::with_capacity(n);
    let mut cols: Vec<usize> = Vec::with_capacity(n);
    let mut f_masked: Vec<f64> = Vec::with_capacity(n);
    let mut b_masked: Vec<f64> = Vec::with_capacity(n);
    let mut not_nan: Vec<bool> = Vec::with_capacity(n);
    let mut ys: Vec<f64> = Vec::with_capacity(n);
    let mut bs: Vec<f64> = Vec::with_capacity(n);

    for ind_level in [&ind2, &ind1, &ind0] {
        for idx in 0..t {
            // 拷贝当日行, 避免持有 fv 视图
            let mut fv_row: Vec<f64> = fv.row(idx).iter().copied().collect();
            // 当日因子不同值数 < 10 则跳过 (NaN 计 1, 同 np.unique)
            if !has_ge_n_unique(&fv_row, 10) {
                continue;
            }
            let ind_row = ind_level.row(idx);
            let base_row = base_list[0].row(idx);
            // 按行业码排序 (NaN 段在末尾, 跳过; radix sort)
            order.clear();
            order_keys.clear();
            for j in 0..n {
                order.push(j);
                order_keys.push(mono_key(ind_row[j]));
            }
            radix_sort_order(&order_keys, &mut order, &mut order_tmp);
            let mut seg_start = 0usize;
            while seg_start < n {
                let code = ind_row[order[seg_start]];
                if code.is_nan() {
                    break; // 剩余均为 NaN 段, 不参与
                }
                let mut seg_end = seg_start + 1;
                while seg_end < n && ind_row[order[seg_end]] == code {
                    seg_end += 1;
                }
                // 段内处理: cols = order[seg_start..seg_end]
                let m_count = seg_end - seg_start;
                cols.clear();
                f_masked.clear();
                b_masked.clear();
                not_nan.clear();
                for &ci in &order[seg_start..seg_end] {
                    cols.push(ci);
                    f_masked.push(fv_row[ci]);
                    let bv = base_row[ci];
                    b_masked.push(bv);
                    not_nan.push(!f_masked.last().unwrap().is_nan() && !bv.is_nan());
                }
                let n_obs = not_nan.iter().filter(|&&b| b).count();
                if n_obs >= 10 {
                    // 有效观测 (复用缓冲区)
                    ys.clear();
                    bs.clear();
                    for i in 0..m_count {
                        if not_nan[i] {
                            ys.push(f_masked[i]);
                            bs.push(b_masked[i]);
                        }
                    }
                    let (c0, c1) = ols2(&ys, &bs);
                    // _f[~_not_nan] = c0 + c1 * b
                    for i in 0..m_count {
                        if !not_nan[i] {
                            f_masked[i] = c0 + c1 * b_masked[i];
                        }
                    }
                    for (mi, &ci) in cols.iter().enumerate() {
                        fv_row[ci] = f_masked[mi];
                    }
                }
                seg_start = seg_end;
            }
            for j in 0..n {
                fv[[idx, j]] = fv_row[j];
            }
        }
    }
}

/// _fill_by_group_median: 行内分组中位填充 (median 来自 median_source, 固定不重算)。
/// 优化: 按分组码排序一次, 顺序扫描连续段。
fn fill_by_group_median(
    values: &mut Array2<f64>,
    codes: &Array2<f64>,
    valid_mask: Option<&Array2<f64>>,
    median_source: &Array2<f64>,
) {
    let (t, n) = values.dim();
    let mut order: Vec<usize> = Vec::with_capacity(n);
    let mut order_tmp: Vec<usize> = Vec::with_capacity(n);
    let mut order_keys: Vec<u64> = Vec::with_capacity(n);
    let mut sv: Vec<f64> = Vec::with_capacity(n);
    let mut nan_mask: Vec<bool> = Vec::with_capacity(n);
    let is_market_level = valid_mask.is_some(); // 全市场级: codes 全 0
    for idx in 0..t {
        let mut row = values.row_mut(idx);
        let source_row = median_source.row(idx);
        nan_mask.clear();
        nan_mask.extend(row.iter().map(|v| v.is_nan()));
        if !nan_mask.iter().any(|&b| b) {
            continue;
        }
        let codes_row = codes.row(idx);
        let valid_row = valid_mask.map(|vm| vm.row(idx));
        if is_market_level {
            // 全市场级特化: 单段, 免排序
            order.clear();
            order.extend(0..n);
        } else {
            // 按分组码排序 (NaN 段在末尾, 跳过; radix sort)
            order.clear();
            order_keys.clear();
            for j in 0..n {
                order.push(j);
                order_keys.push(mono_key(codes_row[j]));
            }
            radix_sort_order(&order_keys, &mut order, &mut order_tmp);
        }
        let mut seg_start = 0usize;
        while seg_start < n {
            let code = codes_row[order[seg_start]];
            if code.is_nan() {
                break;
            }
            let mut seg_end = seg_start + 1;
            while seg_end < n && codes_row[order[seg_end]] == code {
                seg_end += 1;
            }
            // m = (code 段) & valid_mask
            sv.clear();
            for &ci in &order[seg_start..seg_end] {
                if valid_row.map_or(true, |vr| vr[ci] == 1.0) && !source_row[ci].is_nan() {
                    sv.push(source_row[ci]);
                }
            }
            if !sv.is_empty() {
                let med = median_inplace(&mut sv);
                for &ci in &order[seg_start..seg_end] {
                    if nan_mask[ci] && valid_row.map_or(true, |vr| vr[ci] == 1.0) {
                        row[ci] = med;
                    }
                }
            }
            seg_start = seg_end;
        }
    }
}

/// get_residual: 逐日对基准因子(已 rank pct)做含截距 OLS 取残差。
/// 样本数 >10 才回归; y 截面 unique==1 时残差整体置 0.5。
fn get_residual(fv: &Array2<f64>, bench: &[Array2<f64>]) -> Array2<f64> {
    let (t, n) = fv.dim();
    let mut resid = Array2::<f64>::from_elem((t, n), f64::NAN);
    let k = bench.len();
    let p = k + 1;
    let mut valid: Vec<bool> = Vec::with_capacity(n);
    // 行存储: 每只有效股票一行 [y, b0..b9], 定长数组免分配, 顺序访问利于 cache
    let mut rows: Vec<[f64; 11]> = Vec::with_capacity(n);
    let mut cur = [0.0_f64; 11];
    for idx in 0..t {
        let row = fv.row(idx);
        if !row.iter().any(|v| !v.is_nan()) {
            continue; // dropna(how="all")
        }
        valid.clear();
        rows.clear();
        for j in 0..n {
            let ok = row[j].is_finite() && bench.iter().all(|b| b[[idx, j]].is_finite());
            valid.push(ok);
            if ok {
                cur[0] = row[j];
                for c in 0..k {
                    cur[c + 1] = bench[c][[idx, j]];
                }
                rows.push(cur);
            }
        }
        let n_valid = rows.len();
        if n_valid <= 10 {
            continue;
        }
        let mut resid_row = vec![f64::NAN; n];
        let mut uniq: Vec<f64> = rows.iter().map(|r| r[0]).collect();
        uniq.sort_by(cmp_f64);
        uniq.dedup_by(|a, b| (a.is_nan() && b.is_nan()) || a == b);
        if uniq.len() == 1 {
            for j in 0..n {
                if valid[j] {
                    resid_row[j] = 0.5;
                }
            }
        } else {
            // X'X (11×11) 与 X'y 行存储顺序累积
            let mut xtx = vec![0.0_f64; p * p];
            let mut xty = vec![0.0_f64; p];
            for r in rows.iter() {
                let yv = r[0];
                xty[0] += yv;
                xtx[0 * p + 0] += 1.0;
                for c in 0..k {
                    let b = r[c + 1];
                    xty[c + 1] += b * yv;
                    xtx[0 * p + c + 1] += b;
                    xtx[(c + 1) * p + 0] += b;
                }
                for c1 in 0..k {
                    let b1 = r[c1 + 1];
                    xtx[(c1 + 1) * p + (c1 + 1)] += b1 * b1;
                    for c2 in (c1 + 1)..k {
                        let v = b1 * r[c2 + 1];
                        xtx[(c1 + 1) * p + (c2 + 1)] += v;
                        xtx[(c2 + 1) * p + (c1 + 1)] += v;
                    }
                }
            }
            let m = DMatrix::from_row_slice(p, p, &xtx);
            let rhs = DMatrix::from_column_slice(p, 1, &xty);
            // 小样本 (n_valid <= 40) 或 Cholesky 失败时走 SVD 伪逆:
            // 近奇异 X'X 下正规方程与 numpy lstsq(SVD 截断) 解差异可达排序级,
            // 真实数据每日有效 4000+ 永远走 Cholesky 快路径。
            let use_svd = n_valid <= 40 || Cholesky::new(m.clone()).is_none();
            let coef: Vec<f64> = if !use_svd {
                let chol = Cholesky::new(m).expect("chol");
                chol.solve(&rhs).column(0).iter().copied().collect()
            } else {
                // SVD 伪逆 (等价 numpy lstsq rcond 截断)
                let n_r = rows.len();
                let xm = DMatrix::from_fn(n_r, p, |r, c| {
                    if c == 0 {
                        1.0
                    } else {
                        rows[r][c]
                    }
                });
                let svd = xm.clone().svd(true, true);
                let u = svd.u.expect("svd u");
                let vt = svd.v_t.expect("svd vt");
                let s = svd.singular_values;
                let s_max = s.iter().cloned().fold(0.0_f64, f64::max);
                let rcond = s_max * (n_r.max(p) as f64) * 2.22e-16;
                let ym = DMatrix::from_fn(n_r, 1, |r, _| rows[r][0]);
                let uty = u.transpose() * ym;
                let mut coef = DMatrix::zeros(p, 1);
                for i in 0..p {
                    if s[i] > rcond {
                        coef[(i, 0)] = uty[(i, 0)] / s[i];
                    }
                }
                (vt.transpose() * coef).column(0).iter().copied().collect()
            };
            // resid = y - [1, x] @ coef (直接用行存储)
            let mut vi = 0;
            for r in rows.iter() {
                let mut pred = coef[0];
                for c in 0..k {
                    pred += coef[c + 1] * r[c + 1];
                }
                // 按 valid 顺序写回
                while !valid[vi] {
                    vi += 1;
                }
                resid_row[vi] = r[0] - pred;
                vi += 1;
            }
        }
        for j in 0..n {
            resid[[idx, j]] = resid_row[j];
        }
    }
    resid
}

/// 生产标准预处理主入口 (对齐 preprocess_factor_standalone.fill_and_rank_factors)。
///
/// rank_barra=True (默认): barra/ind_base 传原始值, 内部自动 rank pct (独立调用安全)。
/// rank_barra=False: barra/ind_base 须已 rank pct, 跳过内部 rank (批量场景预 rank 一次, 省 10 次全矩阵 rank)。
#[pyfunction]
#[pyo3(signature = (factor, industry, restrict, barra_list, ind_base_list, rank_barra=true))]
fn neutralize_std_pipeline<'py>(
    py: Python<'py>,
    factor: PyReadonlyArray2<'py, f64>,
    industry: PyReadonlyArray2<'py, f64>,
    restrict: PyReadonlyArray2<'py, f64>,
    barra_list: Vec<PyReadonlyArray2<'py, f64>>,
    ind_base_list: Vec<PyReadonlyArray2<'py, f64>>,
    rank_barra: bool,
) -> PyResult<Py<PyArray2<f64>>> {
    let (t, n) = factor.as_array().dim();
    for (name, arr) in [
        ("industry", industry.as_array()),
        ("restrict", restrict.as_array()),
    ] {
        if arr.dim() != (t, n) {
            return Err(PyValueError::new_err(format!(
                "{} shape {:?} != factor {:?}",
                name,
                arr.dim(),
                (t, n)
            )));
        }
    }
    for (i, b) in barra_list.iter().enumerate() {
        if b.as_array().dim() != (t, n) {
            return Err(PyValueError::new_err(format!(
                "barra_list[{}] shape {:?} != factor {:?}",
                i,
                b.as_array().dim(),
                (t, n)
            )));
        }
    }
    if ind_base_list.len() != 1 {
        return Err(PyValueError::new_err(
            "ind_base_list 只支持 1 个 base 因子 (生产口径为 size)",
        ));
    }

    let fv = factor.as_array().to_owned();
    let ind3 = industry.as_array().to_owned();
    let rest = restrict.as_array().to_owned();
    let barra: Vec<Array2<f64>> = barra_list.iter().map(|b| b.as_array().to_owned()).collect();
    let ind_base: Vec<Array2<f64>> =
        ind_base_list.iter().map(|b| b.as_array().to_owned()).collect();

    let output = py.allow_threads(move || -> Result<Array2<f64>, String> {
        // 1. factor rank pct; barra/ind_base 按 rank_barra 决定是否内部 rank
        //    (对应 load_context 的 _ranked; 批量场景预 rank 后传 rank_barra=False)
        let mut fv_ranked = fv.clone();
        rank_pct_all(&mut fv_ranked);
        let mut barra_ranked: Vec<Array2<f64>> = barra.clone();
        if rank_barra {
            for b in barra_ranked.iter_mut() {
                rank_pct_all(b);
            }
        }
        let mut ind_base_ranked: Vec<Array2<f64>> = ind_base.clone();
        if rank_barra {
            for b in ind_base_ranked.iter_mut() {
                rank_pct_all(b);
            }
        }
        
        // 2. 行业 OLS 回归填充
        fill_ind_reg(&mut fv_ranked, &ind_base_ranked, &ind3);
        
        // 3. 行业 2 级中位填充
        let mut fv_filled = fv_ranked.clone();
        let ind1 = ind3.map(|&v| (v / 10000.0).floor());
        // fv_v[np.isnan(ind1)] = np.nan
        for i in 0..(t * n) {
            if ind1.as_slice().unwrap()[i].is_nan() {
                fv_filled.as_slice_mut().unwrap()[i] = f64::NAN;
            }
        }
        let median_source = fv_filled.clone();
        let ind2 = ind3.map(|&v| (v / 100.0).floor());
        fill_by_group_median(&mut fv_filled, &ind2, None, &median_source);
        fill_by_group_median(&mut fv_filled, &ind1, None, &median_source);
        // 全市场级: codes 全 0, valid_mask = ~isnan(ind1)
        let zeros = Array2::<f64>::zeros((t, n));
        let ind1_mask = ind1.map(|&v| if v.is_nan() { 0.0 } else { 1.0 });
        fill_by_group_median(&mut fv_filled, &zeros, Some(&ind1_mask), &median_source);
        
        // 4. 限制股置空: fv_restricted[restrict==0] = filled
        let mut fv_restricted = Array2::<f64>::from_elem((t, n), f64::NAN);
        for i in 0..(t * n) {
            if rest.as_slice().unwrap()[i] == 0.0 {
                fv_restricted.as_slice_mut().unwrap()[i] = fv_filled.as_slice().unwrap()[i];
            }
        }

        // 5. rank pct
        rank_pct_all(&mut fv_restricted);
        
        // 6. 残差 (含截距, 10 风格 rank pct)
        let resid = get_residual(&fv_restricted, &barra_ranked);
        
        // 7. 残差 rank pct
        let mut resid_rank = resid.clone();
        rank_pct_all(&mut resid_rank);
                        Ok(resid_rank)
    })
    .map_err(|e| PyValueError::new_err(e))?;

    Ok(output.into_pyarray(py).to_owned())
}

#[pymodule]
fn dev_sandbox_neutralize(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(neutralize_std_pipeline, m)?)?;
    Ok(())
}
