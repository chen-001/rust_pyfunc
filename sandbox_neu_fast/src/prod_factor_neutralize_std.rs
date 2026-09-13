// factor_neutralize_std: 生产标准「因子合成前预处理」的纯计算模块。
//
// 对齐对象: `/home/chenzongwei/pythoncode/回归校准/preprocess_factor_standalone.py`
// 的 `fill_and_rank_factors` (2026-08-07 生产口径):
//
//   rank pct → 行业 OLS 回归填充(base=size, ind2→ind1→ind0)
//           → 行业 2 级中位填充(median 基于填充前原始值)
//           → 限制股置空(restrict==0 保留) → rank pct
//           → 含截距 10 风格 OLS 残差(样本>10, y 常数截面置 0.5) → 残差 rank pct
//
// 性能优化 (46/46 hm80 因子全量验证与标准 Python 逐位一致):
// - rank pct: 8-bit LSD radix sort (u64 单调位序 key)
// - 行业 OLS 填充: 2 参数闭式解 + 行业码排序分段扫描
// - 行业中位填充: 排序分段 + quickselect 中位数
// - 残差回归: X'X 行存储顺序累积; n_valid>40 Cholesky, 小样本/奇异回退 SVD
//   (与 numpy lstsq 截断一致); industry_neutralize=True 时加一级行业 one-hot
//
// 已知边界: n_valid = p (11 样本 11 参数) 精确插值时残差恒 0, 数值噪声排序
// 无法跨实现对齐 (真实数据每日 4000+ 样本, 永不触发)。
//
// 入口: `neutralize_std_section(factor, industry, restrict, barra, ind_base,
// industry_neutralize)` —— 输入均为 (T,N) f64 模板轴矩阵, barra/ind_base 传原始值
// (内部 rank pct), 输出残差 rank pct (T,N)。

use crate::factor_neutralization_io_optimized::IOOptimizedStyleData;
use nalgebra::{Cholesky, DMatrix};
use ndarray::{s, Array2, Array3, ArrayView2, ArrayView3};
use numpy::{IntoPyArray, PyArray3, PyReadonlyArray2, PyReadonlyArray3};
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

/// pandas `rank(axis=1, pct=True)` 语义: 行内非NaN平均秩, pct = rank / n_non_nan。
/// radix sort (u64 单调位序 key) + 原值 == 判等值组 (含 -0.0/+0.0 合并, 同 pandas)。
fn rank_pct_row_into(
    vals: &[f64],
    ranks: &mut Vec<f64>,
    idxs: &mut Vec<usize>,
    tmp: &mut Vec<usize>,
    keys: &mut Vec<u64>,
) {
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
        rank_pct_row_into(
            row.as_slice().unwrap(),
            &mut ranks,
            &mut idxs,
            &mut tmp,
            &mut keys,
        );
        for (j, &r) in ranks.iter().enumerate() {
            row[j] = r;
        }
    }
}

/// f32 单调位序 key (与 mono_key 同构: 正数置符号位, 负数全翻转; ±0 合并)。
#[inline]
fn mono_key32(v: f32) -> u32 {
    let v = if v == 0.0 { 0.0 } else { v };
    let bits = v.to_bits();
    if bits >> 31 == 0 {
        bits ^ 0x8000_0000
    } else {
        !bits
    }
}

/// u32 key 的 8-bit LSD radix (4 趟)。初始 order 按 index 升序时稳定等价
/// (key, index) 总序 —— 与生产 radix_sort_order(64bit key) 的排序结果一致。
fn radix_sort_order32(keys: &[u32], order: &mut Vec<usize>, tmp: &mut Vec<usize>) {
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

/// O3e: 第一次 rank pct 直接在 f32 源上做 (u32 key 4-pass radix)。
/// f32→f64 是保序单射, 排序/并列组完全一致 ⇒ 输出 pct 值与 f64 版逐位一致。
fn rank_pct_all_from_f32(slot: &ArrayView2<f32>, out: &mut Array2<f64>) {
    let (t, n) = slot.dim();
    let mut idxs: Vec<usize> = Vec::with_capacity(n);
    let mut tmp: Vec<usize> = Vec::with_capacity(n);
    let mut keys: Vec<u32> = vec![0; n];
    for ti in 0..t {
        let mut out_row = out.row_mut(ti);
        let in_row = slot.row(ti);
        idxs.clear();
        for (i, &v) in in_row.iter().enumerate() {
            out_row[i] = f64::NAN;
            if !v.is_nan() {
                idxs.push(i);
                keys[i] = mono_key32(v);
            }
        }
        let n_valid = idxs.len();
        if n_valid == 0 {
            continue;
        }
        radix_sort_order32(&keys, &mut idxs, &mut tmp);
        let mut i = 0;
        while i < n_valid {
            let mut j = i;
            while j + 1 < n_valid && in_row[idxs[j + 1]] == in_row[idxs[i]] {
                j += 1;
            }
            let avg_rank = ((i + 1) + (j + 1)) as f64 / 2.0;
            let pct = avg_rank / n_valid as f64;
            for item in &idxs[i..=j] {
                out_row[*item] = pct;
            }
            i = j + 1;
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

/// 内存优化版 fill_by_group_median：median_source 就是 values 当前行自身。
///
/// 与 fill_by_group_median 完全相同的分组/排序/中位数/回填逻辑，但逐行把 source 先拷到
/// 一个长度 = N 的复用缓冲里（几十 KB），避免为 median_source 额外保留一整张 T×N f64。
/// 正确性依据：同一行内各组按 codes 分段互不重叠，先填的组不会影响后续组的 source 取值。
fn fill_by_group_median_inplace(
    values: &mut Array2<f64>,
    codes: &Array2<f64>,
    valid_mask: Option<&Array2<f64>>,
) {
    let (t, n) = values.dim();
    let mut order: Vec<usize> = Vec::with_capacity(n);
    let mut order_tmp: Vec<usize> = Vec::with_capacity(n);
    let mut order_keys: Vec<u64> = Vec::with_capacity(n);
    let mut sv: Vec<f64> = Vec::with_capacity(n);
    let mut nan_mask: Vec<bool> = Vec::with_capacity(n);
    let mut source_row: Vec<f64> = Vec::with_capacity(n);
    let is_market_level = valid_mask.is_some(); // 全市场级: codes 全 0
    for idx in 0..t {
        let mut row = values.row_mut(idx);
        source_row.clear();
        source_row.extend(row.iter().copied());
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

/// get_residual: 逐日对基准因子(已 rank pct)做 OLS 取残差。
///
/// industry=None (标准口径): X = [1, 10风格], 含截距。
/// industry=Some(ind1码):    X = [10风格, 一级行业 one-hot], 无显式截距
///                           (行业 one-hot 行和=1 隐含截距, 与 tail_pipeline_engine
///                           旧实现 industry_neutralize=True 口径一致; 行业码 NaN
///                           的股票哑变量全 0)。
/// 样本数 >10 才回归; y 截面 unique==1 时残差整体置 0.5。
fn get_residual(
    fv: &Array2<f64>,
    bench: &[Array2<f64>],
    industry: Option<&Array2<f64>>,
) -> Array2<f64> {
    let (t, n) = fv.dim();
    let mut resid = Array2::<f64>::from_elem((t, n), f64::NAN);
    let k = bench.len();
    let mut valid: Vec<bool> = Vec::with_capacity(n);
    // 行存储: 每只有效股票一行 [y, b0..b9], 定长数组免分配, 顺序访问利于 cache
    let mut rows: Vec<[f64; 11]> = Vec::with_capacity(n);
    let mut cur = [0.0_f64; 11];
    // 行业列索引 (有行业模式): -1 = 无行业 (哑变量全 0)
    let mut ind_cols: Vec<i32> = Vec::with_capacity(n);
    // 当日 unique 行业码 (排序)
    let mut ind_codes: Vec<f64> = Vec::with_capacity(40);
    let mut xtx: Vec<f64> = Vec::with_capacity(42 * 42);
    let mut xty: Vec<f64> = Vec::with_capacity(42);
    for idx in 0..t {
        let row = fv.row(idx);
        if !row.iter().any(|v| !v.is_nan()) {
            continue; // dropna(how="all")
        }
        // 当日 unique 一级行业码 (行业模式)
        if let Some(ind) = industry {
            ind_codes.clear();
            for j in 0..n {
                let c = ind[[idx, j]];
                if !c.is_nan() && !ind_codes.contains(&c) {
                    ind_codes.push(c);
                }
            }
            ind_codes.sort_by(cmp_f64);
        }
        let n_ind = if industry.is_some() {
            ind_codes.len()
        } else {
            0
        };
        // 特征列数: 无行业 = 截距 + 10 风格; 有行业 = 10 风格 + 行业 one-hot
        let p = if industry.is_some() { k + n_ind } else { k + 1 };
        valid.clear();
        rows.clear();
        ind_cols.clear();
        for j in 0..n {
            let ok = row[j].is_finite() && bench.iter().all(|b| b[[idx, j]].is_finite());
            valid.push(ok);
            if ok {
                cur[0] = row[j];
                for c in 0..k {
                    cur[c + 1] = bench[c][[idx, j]];
                }
                rows.push(cur);
                if let Some(ind) = industry {
                    let c = ind[[idx, j]];
                    let col = if c.is_nan() {
                        -1
                    } else {
                        match ind_codes.binary_search_by(|x| cmp_f64(x, &c)) {
                            Ok(pos) => pos as i32,
                            Err(_) => -1,
                        }
                    };
                    ind_cols.push(col);
                } else {
                    ind_cols.push(-1);
                }
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
            // X'X 与 X'y 行存储顺序累积
            xtx.clear();
            xtx.resize(p * p, 0.0);
            xty.clear();
            xty.resize(p, 0.0);
            for (i, r) in rows.iter().enumerate() {
                let yv = r[0];
                if industry.is_none() {
                    // 显式截距列 0
                    xty[0] += yv;
                    xtx[0] += 1.0;
                    for c in 0..k {
                        let b = r[c + 1];
                        xty[c + 1] += b * yv;
                        xtx[c + 1] += b;
                        xtx[(c + 1) * p] += b;
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
                } else {
                    // 风格列 0..k
                    for c in 0..k {
                        let b = r[c + 1];
                        xty[c] += b * yv;
                        xtx[c * p + c] += b * b;
                        for c2 in (c + 1)..k {
                            let v = b * r[c2 + 1];
                            xtx[c * p + c2] += v;
                            xtx[c2 * p + c] += v;
                        }
                    }
                    // 行业 one-hot 列 k + ic (ic >= 0; 无行业股票该行全 0)
                    let ic = ind_cols[i];
                    if ic >= 0 {
                        let col = (k as i32 + ic) as usize;
                        xty[col] += yv;
                        xtx[col * p + col] += 1.0;
                        for c in 0..k {
                            let b = r[c + 1];
                            xtx[c * p + col] += b;
                            xtx[col * p + c] += b;
                        }
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
                let xm = DMatrix::from_fn(n_r, p, |r_i, c| {
                    if industry.is_none() {
                        if c == 0 {
                            1.0
                        } else {
                            rows[r_i][c]
                        }
                    } else {
                        let ic = ind_cols[r_i];
                        if c < k {
                            rows[r_i][c + 1]
                        } else if ic >= 0 && c == k as usize + ic as usize {
                            1.0
                        } else {
                            0.0
                        }
                    }
                });
                let svd = xm.clone().svd(true, true);
                let u = svd.u.expect("svd u");
                let vt = svd.v_t.expect("svd vt");
                let s = svd.singular_values;
                let s_max = s.iter().cloned().fold(0.0_f64, f64::max);
                let rcond = s_max * (n_r.max(p) as f64) * 2.22e-16;
                let ym = DMatrix::from_fn(n_r, 1, |r_i, _| rows[r_i][0]);
                let uty = u.transpose() * ym;
                let mut coef = DMatrix::zeros(p, 1);
                for i in 0..p {
                    if s[i] > rcond {
                        coef[(i, 0)] = uty[(i, 0)] / s[i];
                    }
                }
                (vt.transpose() * coef).column(0).iter().copied().collect()
            };
            // resid = y - x @ coef (行存储顺序写回)
            let mut vi = 0;
            for (i, r) in rows.iter().enumerate() {
                let mut pred = 0.0;
                if industry.is_none() {
                    pred = coef[0];
                    for c in 0..k {
                        pred += coef[c + 1] * r[c + 1];
                    }
                } else {
                    for c in 0..k {
                        pred += coef[c] * r[c + 1];
                    }
                    let ic = ind_cols[i];
                    if ic >= 0 {
                        pred += coef[k + ic as usize];
                    }
                }
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

/// 标准中性化管线中「不随因子变化」的预计算量。
///
/// 在 build_shared_inputs 一次性展开并 Arc 共享，供 983 个因子复用，
/// 避免每个 worker 线程、每个因子、每个 slot 都重建 barra(10×T×N f64)
/// 与 template_style_positions(T×N Option<usize>)。
pub struct NeutralizeStdShared {
    pub(crate) industry: Array2<f64>,
    pub(crate) restrict_f64: Array2<f64>,
    /// 行业分级码 (纯 industry 派生，不随因子变化)
    pub(crate) ind1: Array2<f64>,
    pub(crate) ind2: Array2<f64>,
    pub(crate) zeros: Array2<f64>,
    pub(crate) ind1_mask: Array2<f64>,
    /// 10 风格 rank pct 后的模板轴矩阵 (纯 barra 派生)
    pub(crate) barra_ranked: Vec<Array2<f64>>,
    /// size(value_2) rank pct 后的模板轴矩阵 (行业填充 base，纯 size 派生)
    pub(crate) size_ranked: Array2<f64>,
    /// O2 优化 (2026-09): 5 个 (T,N) 行业分组排序。
    /// 0=ind2 1=ind1 2=ind0 供 fill_ind_reg; 3=ind2 4=ind1 供两级中位填充。
    /// 一次性预计算，v2 中性化按此顺序扫描连续段，替代每 slot 重新排序。
    pub(crate) orders: Vec<Array2<usize>>,
    /// O2 优化: 逐日 (p, 有效股票索引, 每股行业列号, X'X)。
    /// X'X 与 Cholesky 只依赖"有效集合"（restrict==0 且 10 风格全有限，经实测
    /// 与因子无关），一次性预计算；v2 的 OLS 每 slot 只做 y 侧累计 + 回代 + 残差。
    /// p=0 表示该日有效数 <=10（生产路径会跳过该日）。
    pub(crate) per_date: Vec<(usize, Vec<u32>, Vec<i32>, Vec<f64>)>,
    /// O3c 优化: 每日预分解 Cholesky (None = 分解失败, 该日回退生产 SVD 路径)。
    /// 与生产"每日两次 Cholesky::new"的解算逐位一致, 且跨 slot 共享为 0 摊销。
    pub(crate) chols: Vec<Option<Cholesky<f64, nalgebra::Dyn>>>,
    /// O3d 优化: 每日有效集 × 10 风格连续内存 (valid 顺序行主序)。
    /// 替代每 slot 从 10 张 (T,N) 大矩阵跳跃取数, 数值逐位一致。
    pub(crate) xdays: Vec<Vec<f64>>,
}

/// 从 style data + industry + restrict 预计算所有不随因子变化的量。
///
/// 模板轴 = (dates.len(), stocks.len())，与 neutralize_std_block 的输出轴一致。
pub fn neutralize_std_precompute(
    industry: &Array2<f64>,
    restrict: &Array2<f32>,
    style_data: &IOOptimizedStyleData,
    dates: &[i32],
    stocks: &[String],
) -> Result<NeutralizeStdShared, String> {
    let n_dates = dates.len();
    let n_stocks = stocks.len();
    if industry.dim() != (n_dates, n_stocks) || restrict.dim() != (n_dates, n_stocks) {
        return Err("neutralize_std_precompute industry/restrict 形状不匹配".to_string());
    }

    // 每日: 模板股票 -> style 当日集合行索引 (6 位码)
    let mut template_style_positions: Vec<Vec<Option<usize>>> = Vec::with_capacity(n_dates);
    for date_idx in 0..n_dates {
        let mut row = vec![None; n_stocks];
        if let Some(day_data) = style_data.data_by_date.get(&(dates[date_idx] as i64)) {
            for (stock_idx, stock) in stocks.iter().enumerate() {
                let code = stock.get(..6).unwrap_or(stock.as_str());
                if let Some(&style_idx) = day_data.stock_index_map.get(code) {
                    row[stock_idx] = Some(style_idx);
                }
            }
        }
        template_style_positions.push(row);
    }

    // barra 10 风格模板轴矩阵 (T,N) f64 (当日集合外 -> NaN)
    let mut barra_list: Vec<Array2<f64>> = (0..10)
        .map(|_| Array2::from_elem((n_dates, n_stocks), f64::NAN))
        .collect();
    for date_idx in 0..n_dates {
        let Some(day_data) = style_data.data_by_date.get(&(dates[date_idx] as i64)) else {
            continue;
        };
        let style_matrix = &day_data.style_matrix; // (n_day, 41)
        for (stock_idx, pos_opt) in template_style_positions[date_idx].iter().enumerate() {
            if let Some(pos) = pos_opt {
                for f in 0..10 {
                    barra_list[f][[date_idx, stock_idx]] = style_matrix[(*pos, f)];
                }
            }
        }
    }
    // size = value_2 (与生产 SzBa size.csv 同源, 已验证)
    let size_cube = barra_list[2].clone();

    // barra 10 风格与 size 的 rank pct (纯输入派生，与因子无关)
    let mut barra_ranked = barra_list;
    for b in barra_ranked.iter_mut() {
        rank_pct_all(b);
    }
    let mut size_ranked = size_cube;
    rank_pct_all(&mut size_ranked);

    // 行业分级码与 restrict (纯输入派生)。
    // 显式重排成标准行主序，避免上游 npy/view 非标准布局导致 as_slice() 为 None。
    let restrict_f64 = Array2::<f64>::from_shape_vec(
        restrict.dim(),
        restrict.iter().map(|&v| v as f64).collect(),
    )
    .expect("restrict 重排为标准行主序失败");
    let ind1 = industry.map(|&v| (v / 10000.0).floor());
    let ind2 = industry.map(|&v| (v / 100.0).floor());
    let zeros = Array2::<f64>::zeros((n_dates, n_stocks));
    let ind1_mask = ind1.map(|&v| if v.is_nan() { 0.0 } else { 1.0 });

    // ---- O2 优化: 预计算行业分组排序 (5 张) 与逐日 X'X ----
    let ind0 = ind1.map(|&v| if v.is_nan() { 0.0 } else { 1.0 });
    let levels = [&ind2, &ind1, &ind0, &ind2, &ind1];
    let mut orders = Vec::with_capacity(5);
    for codes in levels {
        let mut ord = Array2::<usize>::zeros((n_dates, n_stocks));
        for date_idx in 0..n_dates {
            let mut order: Vec<usize> = (0..n_stocks).collect();
            let mut keys: Vec<u64> = codes.row(date_idx).iter().map(|&v| mono_key(v)).collect();
            let mut tmp: Vec<usize> = Vec::new();
            radix_sort_order(&keys, &mut order, &mut tmp);
            for j in 0..n_stocks {
                ord[[date_idx, j]] = order[j];
            }
        }
        orders.push(ord);
    }
    // 逐日 X'X: 有效集合 = restrict==0 且 10 风格全有限。
    // 说明: 生产 get_residual 的 valid 还要求"因子值有限", 实测有效集合与理想集合
    // 100% 重合 (见 RANKIC_NEUTRALIZATION_REPORT 4.4); v2 若某日因子侧有 NaN 会
    // 回退该日为生产原路径, 保证结果与 v1 逐位一致。
    let mut per_date = Vec::with_capacity(n_dates);
    // O3c/O3d: 与 per_date 同构的每日预分解 Cholesky 与连续 X
    let mut chols = Vec::with_capacity(n_dates);
    let mut xdays = Vec::with_capacity(n_dates);
    for date_idx in 0..n_dates {
        let mut ind_codes: Vec<f64> = Vec::new();
        for j in 0..n_stocks {
            let c = ind1[[date_idx, j]];
            if !c.is_nan() && !ind_codes.contains(&c) {
                ind_codes.push(c);
            }
        }
        ind_codes.sort_by(cmp_f64);
        let p = 10 + ind_codes.len();
        let mut valid_idx: Vec<u32> = Vec::new();
        let mut valid_cols: Vec<i32> = Vec::new();
        for j in 0..n_stocks {
            let ok = restrict[[date_idx, j]].is_finite()
                && restrict[[date_idx, j]] == 0.0
                && barra_ranked.iter().all(|b| b[[date_idx, j]].is_finite());
            if ok {
                valid_idx.push(j as u32);
                let c_ind = ind1[[date_idx, j]];
                let col = if c_ind.is_nan() {
                    -1
                } else {
                    match ind_codes.binary_search_by(|x| cmp_f64(x, &c_ind)) {
                        Ok(pos) => pos as i32,
                        Err(_) => -1,
                    }
                };
                valid_cols.push(col);
            }
        }
        let n_valid = valid_idx.len();
        if n_valid <= 10 {
            per_date.push((0, valid_idx, valid_cols, Vec::new()));
            chols.push(None);
            xdays.push(Vec::new());
            continue;
        }
        let mut xtx = vec![0.0f64; p * p];
        let k = 10usize;
        // 连续 X: valid 顺序 × 10 风格 (行主序), 与生产 barra_ranked 取值逐位相同
        let mut xd = vec![0.0f64; n_valid * k];
        for (pos, &j) in valid_idx.iter().enumerate() {
            let ji = j as usize;
            for c in 0..k {
                let b = barra_ranked[c][[date_idx, ji]];
                xd[pos * k + c] = b;
                xtx[c * p + c] += b * b;
                for c2 in (c + 1)..k {
                    let v = b * barra_ranked[c2][[date_idx, ji]];
                    xtx[c * p + c2] += v;
                    xtx[c2 * p + c] += v;
                }
            }
            let ic = valid_cols[pos];
            if ic >= 0 {
                let col = 10 + ic as usize;
                xtx[col * p + col] += 1.0;
                for c in 0..k {
                    let b = barra_ranked[c][[date_idx, ji]];
                    xtx[c * p + col] += b;
                    xtx[col * p + c] += b;
                }
            }
        }
        chols.push(Cholesky::new(DMatrix::from_row_slice(p, p, &xtx)));
        per_date.push((p, valid_idx, valid_cols, xtx));
        xdays.push(xd);
    }

    Ok(NeutralizeStdShared {
        industry: industry.clone(),
        restrict_f64,
        ind1,
        ind2,
        zeros,
        ind1_mask,
        barra_ranked,
        size_ranked,
        orders,
        per_date,
        chols,
        xdays,
    })
}

/// 生产标准预处理主入口 (纯计算版, legacy 语义)。
///
/// 保留原 clone-heavy 实现作为「金标准」，用于与内存优化版做逐位一致性校验。
pub fn neutralize_std_section(
    factor: &Array2<f64>,
    shared: &NeutralizeStdShared,
    industry_neutralize: bool,
) -> Array2<f64> {
    let (t, n) = factor.dim();
    let industry = &shared.industry;
    let restrict = &shared.restrict_f64;
    let barra_ranked = &shared.barra_ranked;
    let ind_base_ranked = &shared.size_ranked;
    let ind1 = &shared.ind1;
    let ind2 = &shared.ind2;
    let zeros = &shared.zeros;
    let ind1_mask = &shared.ind1_mask;

    // 1. factor rank pct (barra/size 已预计算)
    let mut fv_ranked = factor.clone();
    rank_pct_all(&mut fv_ranked);

    // 2. 行业 OLS 回归填充
    fill_ind_reg(
        &mut fv_ranked,
        std::slice::from_ref(ind_base_ranked),
        industry,
    );

    // 3. 行业 2 级中位填充
    let mut fv_filled = fv_ranked.clone();
    for i in 0..(t * n) {
        if ind1.as_slice().unwrap()[i].is_nan() {
            fv_filled.as_slice_mut().unwrap()[i] = f64::NAN;
        }
    }
    let median_source = fv_filled.clone();
    fill_by_group_median(&mut fv_filled, ind2, None, &median_source);
    fill_by_group_median(&mut fv_filled, ind1, None, &median_source);
    fill_by_group_median(&mut fv_filled, zeros, Some(ind1_mask), &median_source);

    // 4. 限制股置空: restrict==0 保留
    let mut fv_restricted = Array2::<f64>::from_elem((t, n), f64::NAN);
    for i in 0..(t * n) {
        if restrict.as_slice().unwrap()[i] == 0.0 {
            fv_restricted.as_slice_mut().unwrap()[i] = fv_filled.as_slice().unwrap()[i];
        }
    }

    // 5. rank pct
    rank_pct_all(&mut fv_restricted);

    // 6. 残差 (默认含截距 10 风格; industry_neutralize=True 时加一级行业 one-hot 无截距)
    let resid = if industry_neutralize {
        get_residual(&fv_restricted, barra_ranked, Some(ind1))
    } else {
        get_residual(&fv_restricted, barra_ranked, None)
    };

    // 7. 残差 rank pct
    let mut resid_rank = resid;
    rank_pct_all(&mut resid_rank);
    resid_rank
}

/// 内存优化版标准中性化：输入 f64 矩阵直接原地复用为 fv_ranked。
///
/// 相对 legacy 的三处省内存改造（数值步骤与顺序完全不变）：
/// 1. 不再 clone factor → fv_ranked（调用方把 slot 转成 f64 后直接传入并接管）；
/// 2. 行业中位填充不再 clone 整张 median_source，改为逐行复用 N 长 source 缓冲；
/// 3. restrict 置空直接原地改写 fv_filled，不再新建 fv_restricted。
pub(crate) fn neutralize_std_section_owned(
    mut fv_ranked: Array2<f64>,
    shared: &NeutralizeStdShared,
    industry_neutralize: bool,
) -> Array2<f64> {
    let (t, n) = fv_ranked.dim();
    let industry = &shared.industry;
    let restrict = &shared.restrict_f64;
    let barra_ranked = &shared.barra_ranked;
    let ind_base_ranked = &shared.size_ranked;
    let ind1 = &shared.ind1;
    let ind2 = &shared.ind2;
    let zeros = &shared.zeros;
    let ind1_mask = &shared.ind1_mask;

    // 1. factor rank pct（原地）
    rank_pct_all(&mut fv_ranked);

    // 2. 行业 OLS 回归填充（原地）
    fill_ind_reg(
        &mut fv_ranked,
        std::slice::from_ref(ind_base_ranked),
        industry,
    );

    // 3. 行业 2 级中位填充。fv_ranked 同时充当 median_source：
    //    先按 ind1 NaN 置空（与 legacy 的 median_source 完全一致），再 clone 出待填矩阵。
    for i in 0..(t * n) {
        if ind1.as_slice().unwrap()[i].is_nan() {
            fv_ranked.as_slice_mut().unwrap()[i] = f64::NAN;
        }
    }
    let mut fv_filled = fv_ranked.clone();
    fill_by_group_median_inplace(&mut fv_filled, ind2, None);
    fill_by_group_median_inplace(&mut fv_filled, ind1, None);
    fill_by_group_median_inplace(&mut fv_filled, zeros, Some(ind1_mask));
    drop(fv_ranked); // median_source 用完即释放，进入残差阶段只需 1 张工作矩阵

    // 4. 限制股置空: restrict==0 保留，其余置 NaN（原地）
    for i in 0..(t * n) {
        if restrict.as_slice().unwrap()[i] != 0.0 {
            fv_filled.as_slice_mut().unwrap()[i] = f64::NAN;
        }
    }

    // 5. rank pct
    rank_pct_all(&mut fv_filled);

    // 6. 残差
    let resid = if industry_neutralize {
        get_residual(&fv_filled, barra_ranked, Some(ind1))
    } else {
        get_residual(&fv_filled, barra_ranked, None)
    };

    // 7. 残差 rank pct
    let mut resid_rank = resid;
    rank_pct_all(&mut resid_rank);
    resid_rank
}

/// block 级标准中性化: 对 rolled_block 的每个 slot 跑 neutralize_std_section。
///
/// 所有不随因子变化的量 (barra/size 的 rank、行业分级码、restrict、模板轴映射)
/// 已预计算进 shared，此函数只做每个 slot 的因子侧计算，不再逐因子展开 barra。
///
/// 输出: (T,N,F) 残差 rank pct (f32), 与生产标准 fill_and_rank_factors 输出一致。
pub fn neutralize_std_block(
    block: ArrayView3<'_, f32>,
    shared: &NeutralizeStdShared,
    industry_neutralize: bool,
) -> Result<Array3<f32>, String> {
    let (n_dates, n_stocks, n_factors) = block.dim();
    if shared.industry.dim() != (n_dates, n_stocks)
        || shared.restrict_f64.dim() != (n_dates, n_stocks)
    {
        return Err("neutralize_std_block industry/restrict 形状不匹配".to_string());
    }

    let mut output = Array3::<f32>::from_elem((n_dates, n_stocks, n_factors), f32::NAN);
    for factor_idx in 0..n_factors {
        let slot = neutralize_std_slot_f32(
            block.slice(s![.., .., factor_idx]),
            shared,
            industry_neutralize,
        )?;
        let out_slice = output.as_slice_mut().unwrap();
        let slot_slice = slot.as_slice().unwrap();
        for i in 0..(n_dates * n_stocks) {
            out_slice[i * n_factors + factor_idx] = slot_slice[i];
        }
    }
    Ok(output)
}

/// 单 slot 标准中性化：输入 (T,N) f32 → 输出 (T,N) f32。
///
/// 数值上与 neutralize_std_block 对应 slot 完全一致：
/// f32→f64 转换、legacy/owned 中性化步骤、f64→f32 回写规则均相同。
/// 供流式回测逐 slot 调用，避免同时物化整个 selected/neutralized block。
pub(crate) fn neutralize_std_slot_f32(
    slot: ArrayView2<'_, f32>,
    shared: &NeutralizeStdShared,
    industry_neutralize: bool,
) -> Result<Array2<f32>, String> {
    let (n_dates, n_stocks) = slot.dim();
    if shared.industry.dim() != (n_dates, n_stocks)
        || shared.restrict_f64.dim() != (n_dates, n_stocks)
    {
        return Err("neutralize_std_block industry/restrict 形状不匹配".to_string());
    }

    let factor_f64 = slot.map(|&v| v as f64);
    let resid_rank = neutralize_std_section_owned(factor_f64, shared, industry_neutralize);
    let mut output = Array2::<f32>::from_elem((n_dates, n_stocks), f32::NAN);
    let out_slice = output.as_slice_mut().unwrap();
    let rr = resid_rank.as_slice().unwrap();
    for i in 0..(n_dates * n_stocks) {
        let v = rr[i];
        out_slice[i] = if v.is_nan() { f32::NAN } else { v as f32 };
    }
    Ok(output)
}

/// pyo3 包装: block 级标准中性化 (验证/独立调用用)。
/// 输入: factor_block (T,N,F) f32, industry (T,N) f64, restrict (T,N) f32,
///       style_data_path, dates, stocks, industry_neutralize。
#[pyfunction]
#[pyo3(signature = (factor_block, industry, restrict, style_data_path, dates, stocks, industry_neutralize=false))]
pub fn neutralize_std_block_py<'py>(
    py: Python<'py>,
    factor_block: PyReadonlyArray3<'py, f32>,
    industry: PyReadonlyArray2<'py, f64>,
    restrict: PyReadonlyArray2<'py, f32>,
    style_data_path: String,
    dates: Vec<i32>,
    stocks: Vec<String>,
    industry_neutralize: bool,
) -> PyResult<Py<PyArray3<f32>>> {
    let block = factor_block.as_array().to_owned();
    let industry_owned = industry.as_array().to_owned();
    let restrict_owned = restrict.as_array().to_owned();
    let output = py.allow_threads(move || -> Result<Array3<f32>, String> {
        let style_data = IOOptimizedStyleData::load_from_parquet_io_optimized(&style_data_path)
            .map_err(|e| e.to_string())?;
        let shared = neutralize_std_precompute(
            &industry_owned,
            &restrict_owned,
            &style_data,
            &dates,
            &stocks,
        )?;
        neutralize_std_block(block.view(), &shared, industry_neutralize)
    });
    Ok(output
        .map_err(PyValueError::new_err)?
        .into_pyarray(py)
        .to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn next(seed: &mut u64) -> f64 {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*seed >> 11) as f64) / ((1u64 << 53) as f64) - 0.5
    }

    fn assert_f64_eq(a: &Array2<f64>, b: &Array2<f64>) {
        assert_eq!(a.dim(), b.dim());
        for (x, y) in a.iter().zip(b.iter()) {
            if x.is_nan() {
                assert!(y.is_nan(), "expected NaN");
            } else {
                assert!(y.is_finite(), "unexpected NaN");
                assert_eq!(x.to_bits(), y.to_bits());
            }
        }
    }

    fn assert_f32_eq(a: &Array2<f32>, b: &Array2<f32>) {
        assert_eq!(a.dim(), b.dim());
        for (x, y) in a.iter().zip(b.iter()) {
            if x.is_nan() {
                assert!(y.is_nan(), "expected NaN");
            } else {
                assert!(y.is_finite(), "unexpected NaN");
                assert_eq!(x.to_bits(), y.to_bits());
            }
        }
    }

    fn make_shared(t: usize, n: usize, seed: u64) -> (NeutralizeStdShared, Array2<f64>) {
        let mut s = seed;
        let mut mat = |nan_every: usize| -> Array2<f64> {
            let mut a = Array2::<f64>::from_elem((t, n), f64::NAN);
            for i in 0..t {
                for j in 0..n {
                    if (i * n + j) % nan_every != 0 {
                        a[[i, j]] = next(&mut s);
                    }
                }
            }
            a
        };
        // 行业码: 少量 NaN + 多级分组，覆盖 ind1/ind2 中位填充路径
        let mut industry = Array2::<f64>::from_elem((t, n), f64::NAN);
        for i in 0..t {
            for j in 0..n {
                industry[[i, j]] = if (i * n + j) % 11 == 0 {
                    f64::NAN
                } else {
                    ((j % 6 + 1) * 10000 + (j % 13 + 1) * 100 + (i % 3 + 1)) as f64
                };
            }
        }
        let mut restrict = Array2::<f64>::from_elem((t, n), 0.0);
        for i in 0..t {
            for j in 0..n {
                if (i * n + j) % 9 == 0 {
                    restrict[[i, j]] = 1.0;
                }
            }
        }
        let ind1 = industry.map(|&v| (v / 10000.0).floor());
        let ind2 = industry.map(|&v| (v / 100.0).floor());
        let zeros = Array2::<f64>::zeros((t, n));
        let ind1_mask = ind1.map(|&v| if v.is_nan() { 0.0 } else { 1.0 });
        let barra_ranked: Vec<Array2<f64>> = (0..10).map(|_| mat(13)).collect();
        let size_ranked = mat(17);
        let factor = mat(19);
        let shared = NeutralizeStdShared {
            industry,
            restrict_f64: restrict,
            ind1,
            ind2,
            zeros,
            ind1_mask,
            barra_ranked,
            size_ranked,
        };
        (shared, factor)
    }

    #[test]
    fn owned_section_matches_legacy_section_exact() {
        let (shared, factor) = make_shared(24, 60, 0x1111_2222_3333_4444);
        for industry_neutralize in [false, true] {
            let expected = neutralize_std_section(&factor, &shared, industry_neutralize);
            let actual = neutralize_std_section_owned(factor.clone(), &shared, industry_neutralize);
            assert_f64_eq(&expected, &actual);
        }
    }

    #[test]
    fn slot_f32_matches_legacy_block_cast_exact() {
        let (shared, _factor) = make_shared(24, 60, 0xaaaa_bbbb_cccc_dddd);
        let (t, n) = shared.industry.dim();
        let mut block = Array3::<f32>::from_elem((t, n, 5), f32::NAN);
        let mut seed = 0xdead_beef_cafe_f00d;
        for f in 0..5 {
            for i in 0..t {
                for j in 0..n {
                    if (i * n + j + f) % 5 != 0 {
                        block[[i, j, f]] = next(&mut seed) as f32;
                    }
                }
            }
        }
        for industry_neutralize in [false, true] {
            for f in 0..5 {
                let factor_f64 = block.slice(s![.., .., f]).map(|&v| v as f64);
                let expected_f64 =
                    neutralize_std_section(&factor_f64, &shared, industry_neutralize);
                let mut expected = Array2::<f32>::from_elem((t, n), f32::NAN);
                for i in 0..(t * n) {
                    let v = expected_f64.as_slice().unwrap()[i];
                    expected.as_slice_mut().unwrap()[i] =
                        if v.is_nan() { f32::NAN } else { v as f32 };
                }
                let actual = neutralize_std_slot_f32(
                    block.slice(s![.., .., f]),
                    &shared,
                    industry_neutralize,
                )
                .unwrap();
                assert_f32_eq(&expected, &actual);
            }
        }
    }
}

// ============================================================================
// O2 优化 (2026-09): 标准中性化 v2 —— 预计算行业排序 + 逐日 X'X。
//
// 与 v1 (neutralize_std_section_owned) 数值逐位一致:
//  - 行业 OLS 填充/两级中位填充: 排序由 shared.orders 预计算 (同一 radix 排序,
//    同一分段顺序), 段内计算与 v1 完全一致;
//  - 残差 OLS: X'X 由 shared.per_date 预计算 (累积顺序与 v1 完全相同), 每日只做
//    y 侧 xty 累计 + Cholesky/回退求解 + 残差写回;
//  - 若某日"理想有效集合"中存在因子侧 NaN (有效集合与因子理论上可能相关, 实测
//    100% 重合), 该日整体回退到 v1 生产原路径, 保证逐位一致;
//  - 最终残差 rank_pct 与 v1 相同 (回测对保序变换不变的另一优化未启用, 见
//    RANKIC_NEUTRALIZATION_REPORT 方案 C')。
// ============================================================================

/// 填充分段: 按预计算排序顺序扫描连续段 (与 fill_ind_reg 的 radix 排序等价)。
fn fill_ind_reg_pre(
    fv: &mut Array2<f64>,
    ind2: &Array2<f64>,
    ind1: &Array2<f64>,
    size_ranked: &Array2<f64>,
    orders: &[Array2<usize>],
) {
    // O3g (2026-09): 无整行拷贝 + 缓冲复用 + O3f 干净段跳过。与旧实现逐位一致:
    // 分组分段互不重叠, 行内直接读写与"先拷贝后回写"等价; 段内无填充目标时
    // 回归结果不会被使用, 跳过收集与回归。
    let (t, n) = fv.dim();
    let ind0 = ind1.map(|&v| if v.is_nan() { 0.0 } else { 1.0 });
    let levels = [ind2, ind1, &ind0];
    let mut ys: Vec<f64> = Vec::with_capacity(n);
    let mut bs: Vec<f64> = Vec::with_capacity(n);
    let mut obs: Vec<bool> = Vec::with_capacity(n);
    for (li, level) in levels.iter().enumerate() {
        for idx in 0..t {
            {
                let row_view = fv.row(idx);
                let row_slice = row_view.as_slice().unwrap();
                if !has_ge_n_unique(row_slice, 10) {
                    continue;
                }
            }
            let order_arr = orders[li].row(idx);
            let order = order_arr.as_slice().unwrap();
            let size_row = size_ranked.row(idx);
            let size_row = size_row.as_slice().unwrap();
            let level_row = level.row(idx);
            let level_row = level_row.as_slice().unwrap();
            let mut row_view = fv.row_mut(idx);
            let row = row_view.as_slice_mut().unwrap();
            let mut seg_start = 0usize;
            while seg_start < n {
                let code = level_row[order[seg_start]];
                if code.is_nan() {
                    break;
                }
                let mut seg_end = seg_start + 1;
                while seg_end < n && level_row[order[seg_end]] == code {
                    seg_end += 1;
                }
                // O3f: 无填充目标（所有位置 row 与 size 都有限）→ 跳过
                let mut has_target = false;
                for &ci in &order[seg_start..seg_end] {
                    if row[ci].is_nan() || size_row[ci].is_nan() {
                        has_target = true;
                        break;
                    }
                }
                if !has_target {
                    seg_start = seg_end;
                    continue;
                }
                ys.clear();
                bs.clear();
                obs.clear();
                for &ci in &order[seg_start..seg_end] {
                    let ok = !row[ci].is_nan() && !size_row[ci].is_nan();
                    obs.push(ok);
                    if ok {
                        ys.push(row[ci]);
                        bs.push(size_row[ci]);
                    }
                }
                if ys.len() >= 10 {
                    let (c0, c1) = ols2(&ys, &bs);
                    for (mi, &ci) in order[seg_start..seg_end].iter().enumerate() {
                        if !obs[mi] {
                            row[ci] = c0 + c1 * size_row[ci];
                        }
                    }
                }
                seg_start = seg_end;
            }
        }
    }
}

/// 分组中位填充 (预计算排序版, 语义与 fill_by_group_median_inplace 一致)。
fn group_median_fill_pre(
    values: &mut Array2<f64>,
    codes: &Array2<f64>,
    valid_mask: Option<&Array2<f64>>,
    orders: &Array2<usize>,
) {
    // O3g (2026-09): 无整行拷贝 + O3f 干净段跳过。与旧实现逐位一致:
    // 段内没有 (NaN 且 valid) 的填充目标时, 中位数不会被使用。
    let (t, n) = values.dim();
    let mut sv: Vec<f64> = Vec::with_capacity(n);
    let mut nan_mask: Vec<bool> = Vec::with_capacity(n);
    for idx in 0..t {
        let mut row_view = values.row_mut(idx);
        let row = row_view.as_slice_mut().unwrap();
        nan_mask.clear();
        let mut has_nan = false;
        for j in 0..n {
            let nn = row[j].is_nan();
            nan_mask.push(nn);
            if nn {
                has_nan = true;
            }
        }
        if !has_nan {
            continue;
        }
        let order_arr = orders.row(idx);
        let order = order_arr.as_slice().unwrap();
        let codes_row = codes.row(idx);
        let codes_row = codes_row.as_slice().unwrap();
        let valid_row = valid_mask.map(|vm| vm.row(idx));
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
            // O3f: 无填充目标 → 跳过收集与 quickselect
            let mut has_target = false;
            for &ci in &order[seg_start..seg_end] {
                let valid = valid_row.as_ref().map_or(true, |vr| vr[ci] == 1.0);
                if nan_mask[ci] && valid {
                    has_target = true;
                    break;
                }
            }
            if !has_target {
                seg_start = seg_end;
                continue;
            }
            sv.clear();
            for &ci in &order[seg_start..seg_end] {
                let valid = valid_row.as_ref().map_or(true, |vr| vr[ci] == 1.0);
                if valid && !row[ci].is_nan() {
                    sv.push(row[ci]);
                }
            }
            if !sv.is_empty() {
                let med = median_inplace(&mut sv);
                for &ci in &order[seg_start..seg_end] {
                    let valid = valid_row.as_ref().map_or(true, |vr| vr[ci] == 1.0);
                    if nan_mask[ci] && valid {
                        row[ci] = med;
                    }
                }
            }
            seg_start = seg_end;
        }
    }
}

/// v2 残差回归: 预计算 X'X; 因子侧 NaN 触发该日回退到 get_residual 原路径。

/// 生产语义的单日 OLS 残差 (因子侧有效集 = y有限 ∩ 风格有限, X'X 逐日实时累积)。
/// 与 get_residual 的单日逻辑完全一致, 但直接在全矩阵行上工作, 无临时切片。
fn ols_day_inline(
    fv_filled: &Array2<f64>,
    barra_ranked: &[Array2<f64>],
    ind1: &Array2<f64>,
    idx: usize,
) -> Vec<f64> {
    let n = fv_filled.ncols();
    let k = barra_ranked.len();
    let mut resid_row = vec![f64::NAN; n];
    let mut ind_codes: Vec<f64> = Vec::with_capacity(40);
    for j in 0..n {
        let c = ind1[[idx, j]];
        if !c.is_nan() && !ind_codes.contains(&c) {
            ind_codes.push(c);
        }
    }
    ind_codes.sort_by(cmp_f64);
    let n_ind = ind_codes.len();
    let p = k + n_ind;
    let mut valid: Vec<bool> = Vec::with_capacity(n);
    let mut rows: Vec<[f64; 11]> = Vec::with_capacity(n);
    let mut ind_cols: Vec<i32> = Vec::with_capacity(n);
    for j in 0..n {
        let ok = fv_filled[[idx, j]].is_finite()
            && barra_ranked.iter().all(|b| b[[idx, j]].is_finite());
        valid.push(ok);
        if ok {
            let mut cur = [0.0_f64; 11];
            cur[0] = fv_filled[[idx, j]];
            for c in 0..k {
                cur[c + 1] = barra_ranked[c][[idx, j]];
            }
            rows.push(cur);
            let c = ind1[[idx, j]];
            ind_cols.push(if c.is_nan() {
                -1
            } else {
                match ind_codes.binary_search_by(|x| cmp_f64(x, &c)) {
                    Ok(pos) => pos as i32,
                    Err(_) => -1,
                }
            });
        }
    }
    let n_valid = rows.len();
    if n_valid <= 10 {
        return resid_row;
    }
    let mut uniq: Vec<f64> = rows.iter().map(|r| r[0]).collect();
    uniq.sort_by(cmp_f64);
    uniq.dedup_by(|a, b| (a.is_nan() && b.is_nan()) || a == b);
    if uniq.len() == 1 {
        for j in 0..n {
            if valid[j] {
                resid_row[j] = 0.5;
            }
        }
        return resid_row;
    }
    let mut xtx: Vec<f64> = vec![0.0f64; p * p];
    let mut xty: Vec<f64> = vec![0.0f64; p];
    for (i, r) in rows.iter().enumerate() {
        let yv = r[0];
        for c in 0..k {
            let b = r[c + 1];
            xty[c] += b * yv;
            xtx[c * p + c] += b * b;
            for c2 in (c + 1)..k {
                let v = b * r[c2 + 1];
                xtx[c * p + c2] += v;
                xtx[c2 * p + c] += v;
            }
        }
        let ic = ind_cols[i];
        if ic >= 0 {
            let col = (k as i32 + ic) as usize;
            xty[col] += yv;
            xtx[col * p + col] += 1.0;
            for c in 0..k {
                let b = r[c + 1];
                xtx[c * p + col] += b;
                xtx[col * p + c] += b;
            }
        }
    }
    let m = DMatrix::from_row_slice(p, p, &xtx);
    let rhs = DMatrix::from_column_slice(p, 1, &xty);
    let use_svd = n_valid <= 40 || Cholesky::new(m.clone()).is_none();
    let coef: Vec<f64> = if !use_svd {
        let chol = Cholesky::new(m).expect("chol");
        chol.solve(&rhs).column(0).iter().copied().collect()
    } else {
        let n_r = rows.len();
        let xm = DMatrix::from_fn(n_r, p, |r_i, c| {
            let ic = ind_cols[r_i];
            if c < k {
                rows[r_i][c + 1]
            } else if ic >= 0 && c == k as usize + ic as usize {
                1.0
            } else {
                0.0
            }
        });
        let svd = xm.clone().svd(true, true);
        let u = svd.u.expect("svd u");
        let vt = svd.v_t.expect("svd vt");
        let sv = svd.singular_values;
        let s_max = sv.iter().cloned().fold(0.0_f64, f64::max);
        let rcond = s_max * (n_r.max(p) as f64) * 2.22e-16;
        let ym = DMatrix::from_fn(n_r, 1, |r_i, _| rows[r_i][0]);
        let uty = u.transpose() * ym;
        let mut coef = DMatrix::zeros(p, 1);
        for i in 0..p {
            if sv[i] > rcond {
                coef[(i, 0)] = uty[(i, 0)] / sv[i];
            }
        }
        (vt.transpose() * coef).column(0).iter().copied().collect()
    };
    let mut vi = 0;
    for (i, r) in rows.iter().enumerate() {
        let mut pred = 0.0;
        for c in 0..k {
            pred += coef[c] * r[c + 1];
        }
        let ic = ind_cols[i];
        if ic >= 0 {
            pred += coef[k + ic as usize];
        }
        while !valid[vi] {
            vi += 1;
        }
        resid_row[vi] = r[0] - pred;
        vi += 1;
    }
    resid_row
}

fn get_residual_v2(fv_filled: &Array2<f64>, shared: &NeutralizeStdShared) -> Array2<f64> {
    // O3 优化 (2026-09): O3a uniq 检查线性 min==max 扫描 (替代完整排序+去重);
    // O3c 每日 Cholesky 预分解共享 (替代每 slot 每日两次分解);
    // O3d 每日连续 X 内存 (10 风格列连续, 替代跨 10 张 (T,N) 矩阵跳跃取数)。
    // any_nan / n_valid<=40 / Cholesky 失败日回退生产路径 (ols_day_inline / SVD), 逐位一致。
    let (t, n) = fv_filled.dim();
    let barra_ranked = &shared.barra_ranked;
    let ind1 = &shared.ind1;
    let per_date = &shared.per_date;
    let mut resid = Array2::<f64>::from_elem((t, n), f64::NAN);
    let k = 10usize;
    let mut y_buf: Vec<f64> = Vec::with_capacity(n);
    let mut xty: Vec<f64> = Vec::with_capacity(64);
    for idx in 0..t {
        let (p, valid_idx, valid_cols, _xtx_pre) = &per_date[idx];
        if *p == 0 {
            continue;
        }
        // O3a: y unique 检查 (min==max 线性扫描) + 因子 NaN 检查 (触发回退)
        y_buf.clear();
        let mut any_nan = false;
        let mut mn = f64::INFINITY;
        let mut mx = f64::NEG_INFINITY;
        for &j in valid_idx {
            let y = fv_filled[[idx, j as usize]];
            if !y.is_finite() {
                any_nan = true;
                break;
            }
            if y < mn {
                mn = y;
            }
            if y > mx {
                mx = y;
            }
            y_buf.push(y);
        }
        if any_nan {
            // 回退: 生产语义单日 OLS (无切片, 与 get_residual 逐位一致)。
            // 触发场景: "理想集"内存在因子侧 NaN (稀疏覆盖/早期年份数据),
            // 此时生产 valid 集 = 理想集 ∩ y有限, X'X 与因子相关, 预计算不可用。
            let row = ols_day_inline(fv_filled, barra_ranked, ind1, idx);
            for j in 0..n {
                resid[[idx, j]] = row[j];
            }
            continue;
        }
        if mn == mx {
            for &j in valid_idx {
                resid[[idx, j as usize]] = 0.5;
            }
            continue;
        }
        // O3d: xty 用每日连续 X
        xty.clear();
        xty.resize(*p, 0.0);
        let xd = &shared.xdays[idx];
        for (pos, &yv) in y_buf.iter().enumerate() {
            let xrow = &xd[pos * k..pos * k + k];
            for c in 0..k {
                xty[c] += xrow[c] * yv;
            }
            let ic = valid_cols[pos];
            if ic >= 0 {
                xty[k + ic as usize] += yv;
            }
        }
        // O3c: 预分解 Cholesky; 失败/小样本回退生产 SVD 路径
        let use_svd = valid_idx.len() <= 40 || shared.chols[idx].is_none();
        let coef: Vec<f64> = if !use_svd {
            let chol = shared.chols[idx].as_ref().unwrap();
            let rhs = DMatrix::from_column_slice(*p, 1, &xty);
            chol.solve(&rhs).column(0).iter().copied().collect()
        } else {
            // 回退生产 SVD 路径 (n_valid 极小/近奇异时; 与 get_residual 相同)
            let n_r = valid_idx.len();
            let rows: Vec<[f64; 11]> = valid_idx
                .iter()
                .map(|&j| {
                    let ji = j as usize;
                    let mut cur = [0.0_f64; 11];
                    cur[0] = fv_filled[[idx, ji]];
                    for c in 0..k {
                        cur[c + 1] = barra_ranked[c][[idx, ji]];
                    }
                    cur
                })
                .collect();
            let xm = DMatrix::from_fn(n_r, *p, |r_i, c| {
                let ic = valid_cols[r_i];
                if c < k {
                    rows[r_i][c + 1]
                } else if ic >= 0 && c == k + ic as usize {
                    1.0
                } else {
                    0.0
                }
            });
            let svd = xm.clone().svd(true, true);
            let u = svd.u.expect("svd u");
            let vt = svd.v_t.expect("svd vt");
            let sv = svd.singular_values;
            let s_max = sv.iter().cloned().fold(0.0_f64, f64::max);
            let rcond = s_max * (n_r.max(*p) as f64) * 2.22e-16;
            let ym = DMatrix::from_fn(n_r, 1, |r_i, _| rows[r_i][0]);
            let uty = u.transpose() * ym;
            let mut coef = DMatrix::zeros(*p, 1);
            for i in 0..*p {
                if sv[i] > rcond {
                    coef[(i, 0)] = uty[(i, 0)] / sv[i];
                }
            }
            (vt.transpose() * coef).column(0).iter().copied().collect()
        };
        for (pos, &j) in valid_idx.iter().enumerate() {
            let ji = j as usize;
            let yv = y_buf[pos];
            let xrow = &xd[pos * k..pos * k + k];
            let mut pred = 0.0;
            for c in 0..k {
                pred += coef[c] * xrow[c];
            }
            let ic = valid_cols[pos];
            if ic >= 0 {
                pred += coef[k + ic as usize];
            }
            resid[[idx, ji]] = yv - pred;
        }
    }
    resid
}

/// v2 标准中性化核心 (与 neutralize_std_section_owned 数值逐位一致)。
pub(crate) fn neutralize_std_section_owned_v2(
    mut fv_ranked: Array2<f64>,
    shared: &NeutralizeStdShared,
    industry_neutralize: bool,
) -> Array2<f64> {
    let (t, n) = fv_ranked.dim();
    let ind = &shared.industry;
    let restrict = &shared.restrict_f64;
    let barra_ranked = &shared.barra_ranked;
    let size_ranked = &shared.size_ranked;
    let ind1 = &shared.ind1;
    let ind2 = &shared.ind2;
    let zeros = &shared.zeros;
    let ind1_mask = &shared.ind1_mask;

    rank_pct_all(&mut fv_ranked);
    fill_ind_reg_pre(&mut fv_ranked, ind2, ind1, size_ranked, &shared.orders[0..3]);
    for i in 0..(t * n) {
        if ind1.as_slice().unwrap()[i].is_nan() {
            fv_ranked.as_slice_mut().unwrap()[i] = f64::NAN;
        }
    }
    let mut fv_filled = fv_ranked.clone();
    group_median_fill_pre(&mut fv_filled, ind2, None, &shared.orders[3]);
    group_median_fill_pre(&mut fv_filled, ind1, None, &shared.orders[4]);
    group_median_fill_pre(&mut fv_filled, zeros, Some(ind1_mask), &shared.orders[0]);
    drop(fv_ranked);
    for i in 0..(t * n) {
        if restrict.as_slice().unwrap()[i] != 0.0 {
            fv_filled.as_slice_mut().unwrap()[i] = f64::NAN;
        }
    }
    rank_pct_all(&mut fv_filled);
    let resid = if industry_neutralize {
        get_residual_v2(&fv_filled, shared)
    } else {
        get_residual(&fv_filled, barra_ranked, None)
    };
    let mut resid_rank = resid;
    rank_pct_all(&mut resid_rank);
    resid_rank
}

/// C' 变体: 同 neutralize_std_section_owned_v2, 但不做最终残差 rank_pct
/// (Spearman/十分组对保序变换不变, 回测结果逐位一致, 见 RANKIC_NEUTRALIZATION_REPORT 方案 C')。
/// 引擎回测路径使用 (省 ~12% 中性化时间)。
pub(crate) fn neutralize_std_section_owned_v2_resid(
    mut fv_ranked: Array2<f64>,
    shared: &NeutralizeStdShared,
    industry_neutralize: bool,
) -> Array2<f64> {
    rank_pct_all(&mut fv_ranked);
    neutralize_std_section_owned_v2_resid_pre(fv_ranked, shared, industry_neutralize)
}

/// C' 变体 (rank 已完成): 填充 → 清理 → 中位填充 → restrict → rank → OLS 残差 (无最终 rank)。
fn neutralize_std_section_owned_v2_resid_pre(
    mut fv_ranked: Array2<f64>,
    shared: &NeutralizeStdShared,
    industry_neutralize: bool,
) -> Array2<f64> {
    let (t, n) = fv_ranked.dim();
    let ind = &shared.industry;
    let restrict = &shared.restrict_f64;
    let barra_ranked = &shared.barra_ranked;
    let size_ranked = &shared.size_ranked;
    let ind1 = &shared.ind1;
    let ind2 = &shared.ind2;
    let zeros = &shared.zeros;
    let ind1_mask = &shared.ind1_mask;

    fill_ind_reg_pre(&mut fv_ranked, ind2, ind1, size_ranked, &shared.orders[0..3]);
    for i in 0..(t * n) {
        if ind1.as_slice().unwrap()[i].is_nan() {
            fv_ranked.as_slice_mut().unwrap()[i] = f64::NAN;
        }
    }
    let mut fv_filled = fv_ranked.clone();
    group_median_fill_pre(&mut fv_filled, ind2, None, &shared.orders[3]);
    group_median_fill_pre(&mut fv_filled, ind1, None, &shared.orders[4]);
    group_median_fill_pre(&mut fv_filled, zeros, Some(ind1_mask), &shared.orders[0]);
    drop(fv_ranked);
    for i in 0..(t * n) {
        if restrict.as_slice().unwrap()[i] != 0.0 {
            fv_filled.as_slice_mut().unwrap()[i] = f64::NAN;
        }
    }
    rank_pct_all(&mut fv_filled);
    if industry_neutralize {
        get_residual_v2(&fv_filled, shared)
    } else {
        get_residual(&fv_filled, barra_ranked, None)
    }
}

/// C' 单 slot 变体: 输出残差 (f32, 不做最终 rank_pct)。
pub(crate) fn neutralize_std_slot_f32_v2_resid(
    slot: ArrayView2<'_, f32>,
    shared: &NeutralizeStdShared,
    industry_neutralize: bool,
) -> Result<Array2<f32>, String> {
    let (n_dates, n_stocks) = slot.dim();
    if shared.industry.dim() != (n_dates, n_stocks)
        || shared.restrict_f64.dim() != (n_dates, n_stocks)
    {
        return Err("neutralize_std_block industry/restrict 形状不匹配".to_string());
    }
    // O3e: 第一次 rank 直接在 f32 源做 (u32 4-pass radix, 与 f64 版逐位一致)
    let mut fv_ranked = Array2::<f64>::zeros((n_dates, n_stocks));
    rank_pct_all_from_f32(&slot, &mut fv_ranked);
    let resid = neutralize_std_section_owned_v2_resid_pre(
        fv_ranked,
        shared,
        industry_neutralize,
    );
    let mut output = Array2::<f32>::from_elem((n_dates, n_stocks), f32::NAN);
    let out_slice = output.as_slice_mut().unwrap();
    let rr = resid.as_slice().unwrap();
    for i in 0..(n_dates * n_stocks) {
        let v = rr[i];
        out_slice[i] = if v.is_nan() { f32::NAN } else { v as f32 };
    }
    Ok(output)
}

/// v2 单 slot 标准中性化: 输入 (T,N) f32 → 输出 (T,N) f32。
pub(crate) fn neutralize_std_slot_f32_v2(
    slot: ArrayView2<'_, f32>,
    shared: &NeutralizeStdShared,
    industry_neutralize: bool,
) -> Result<Array2<f32>, String> {
    let (n_dates, n_stocks) = slot.dim();
    if shared.industry.dim() != (n_dates, n_stocks)
        || shared.restrict_f64.dim() != (n_dates, n_stocks)
    {
        return Err("neutralize_std_block industry/restrict 形状不匹配".to_string());
    }
    let factor_f64 = slot.map(|&v| v as f64);
    let resid_rank = neutralize_std_section_owned_v2(factor_f64, shared, industry_neutralize);
    let mut output = Array2::<f32>::from_elem((n_dates, n_stocks), f32::NAN);
    let out_slice = output.as_slice_mut().unwrap();
    let rr = resid_rank.as_slice().unwrap();
    for i in 0..(n_dates * n_stocks) {
        let v = rr[i];
        out_slice[i] = if v.is_nan() { f32::NAN } else { v as f32 };
    }
    Ok(output)
}

// ==================== P3: 按日期小批量多面中性化 (2026-09-01) ====================
// 一个批次取 B 个面, 外层循环日期、内层循环面: 每日共享行 (行业码/排序/风格/
// restrict/有效集) 只装载一次, B 个面共用; 每面的 fv 行按日期流式处理, 每日
// 工作集 B×~72KB×2 + 共享 ~1.5MB, 全部驻留缓存。
// 语义与逐面路径逐位一致: 每个 (面, 日期) 的操作序列与生产完全相同
// (级别内段互不重叠→就地等价; 每级中位的 nan_mask 按当前行状态重建)。
// 实测 (urgency store, 200 线程隔离): neutralize 吞吐 11.1 → 61.5 slot/s (5.5×)。

/// 行级 rank (f32 源, u32 4-pass radix)。与 rank_pct_all_from_f32 逐位一致。
fn rank_pct_row_from_f32_in(
    in_row: &[f32],
    out_row: &mut [f64],
    idxs: &mut Vec<usize>,
    tmp: &mut Vec<usize>,
    keys: &mut Vec<u32>,
) {
    let n = in_row.len();
    for r in out_row.iter_mut() {
        *r = f64::NAN;
    }
    idxs.clear();
    for (i, &v) in in_row.iter().enumerate() {
        if !v.is_nan() {
            idxs.push(i);
            keys[i] = mono_key32(v);
        }
    }
    let n_valid = idxs.len();
    if n_valid == 0 {
        return;
    }
    radix_sort_order32(&keys, idxs, tmp);
    let mut i = 0;
    while i < n_valid {
        let mut j = i;
        while j + 1 < n_valid && in_row[idxs[j + 1]] == in_row[idxs[i]] {
            j += 1;
        }
        let avg_rank = ((i + 1) + (j + 1)) as f64 / 2.0;
        let pct = avg_rank / n_valid as f64;
        for item in &idxs[i..=j] {
            out_row[*item] = pct;
        }
        i = j + 1;
    }
}

/// 行级 f64 rank (就地写回), 与 rank_pct_all 逐行语义一致。
fn rank_pct_row_f64_in_place(
    row: &mut [f64],
    ranks: &mut Vec<f64>,
    idxs: &mut Vec<usize>,
    tmp: &mut Vec<usize>,
    keys: &mut Vec<u64>,
) {
    let vals: Vec<f64> = row.to_vec();
    rank_pct_row_into(&vals, ranks, idxs, tmp, keys);
    row.copy_from_slice(ranks);
}

/// 行级 fill_ind_reg (3 级顺序就地执行; 每级判唯一值/跳过干净段)。
#[allow(clippy::too_many_arguments)]
fn fill_ind_reg_row(
    row: &mut [f64],
    level_rows: [&[f64]; 3],
    size_row: &[f64],
    order_rows: [&[usize]; 3],
    ys: &mut Vec<f64>,
    bs: &mut Vec<f64>,
    obs: &mut Vec<bool>,
) {
    let n = row.len();
    for li in 0..3 {
        if !has_ge_n_unique(row, 10) {
            continue;
        }
        let level_row = level_rows[li];
        let order = order_rows[li];
        let mut seg_start = 0usize;
        while seg_start < n {
            let code = level_row[order[seg_start]];
            if code.is_nan() {
                break;
            }
            let mut seg_end = seg_start + 1;
            while seg_end < n && level_row[order[seg_end]] == code {
                seg_end += 1;
            }
            let mut has_target = false;
            for &ci in &order[seg_start..seg_end] {
                if row[ci].is_nan() || size_row[ci].is_nan() {
                    has_target = true;
                    break;
                }
            }
            if !has_target {
                seg_start = seg_end;
                continue;
            }
            ys.clear();
            bs.clear();
            obs.clear();
            for &ci in &order[seg_start..seg_end] {
                let ok = !row[ci].is_nan() && !size_row[ci].is_nan();
                obs.push(ok);
                if ok {
                    ys.push(row[ci]);
                    bs.push(size_row[ci]);
                }
            }
            if ys.len() >= 10 {
                let (c0, c1) = ols2(&ys, &bs);
                for (mi, &ci) in order[seg_start..seg_end].iter().enumerate() {
                    if !obs[mi] {
                        row[ci] = c0 + c1 * size_row[ci];
                    }
                }
            }
            seg_start = seg_end;
        }
    }
}

/// 行级中位填充 (单级; nan_mask 由调用方按"当前行状态"构建)。
fn median_fill_level_row(
    row: &mut [f64],
    codes_row: &[f64],
    valid_mask_row: Option<&[f64]>,
    order: &[usize],
    nan_mask: &[bool],
    sv: &mut Vec<f64>,
) {
    let n = row.len();
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
        let mut has_target = false;
        for &ci in &order[seg_start..seg_end] {
            let valid = valid_mask_row.map_or(true, |vr| vr[ci] == 1.0);
            if nan_mask[ci] && valid {
                has_target = true;
                break;
            }
        }
        if !has_target {
            seg_start = seg_end;
            continue;
        }
        sv.clear();
        for &ci in &order[seg_start..seg_end] {
            let valid = valid_mask_row.map_or(true, |vr| vr[ci] == 1.0);
            if valid && !row[ci].is_nan() {
                sv.push(row[ci]);
            }
        }
        if !sv.is_empty() {
            let med = median_inplace(sv);
            for &ci in &order[seg_start..seg_end] {
                let valid = valid_mask_row.map_or(true, |vr| vr[ci] == 1.0);
                if nan_mask[ci] && valid {
                    row[ci] = med;
                }
            }
        }
        seg_start = seg_end;
    }
}

/// ols_day_inline 的行级移植 (回退路径, 逐位一致)。
fn ols_day_inline_row(fv_row: &[f64], barra_row: &[&[f64]], ind1_row: &[f64]) -> Vec<f64> {
    let n = fv_row.len();
    let k = barra_row.len();
    let mut resid_row = vec![f64::NAN; n];
    let mut ind_codes: Vec<f64> = Vec::with_capacity(40);
    for j in 0..n {
        let c = ind1_row[j];
        if !c.is_nan() && !ind_codes.contains(&c) {
            ind_codes.push(c);
        }
    }
    ind_codes.sort_by(cmp_f64);
    let n_ind = ind_codes.len();
    let p = k + n_ind;
    let mut valid: Vec<bool> = Vec::with_capacity(n);
    let mut rows: Vec<[f64; 11]> = Vec::with_capacity(n);
    let mut ind_cols: Vec<i32> = Vec::with_capacity(n);
    for j in 0..n {
        let ok = fv_row[j].is_finite() && barra_row.iter().all(|b| b[j].is_finite());
        valid.push(ok);
        if ok {
            let mut cur = [0.0_f64; 11];
            cur[0] = fv_row[j];
            for c in 0..k {
                cur[c + 1] = barra_row[c][j];
            }
            rows.push(cur);
            let c = ind1_row[j];
            ind_cols.push(if c.is_nan() {
                -1
            } else {
                match ind_codes.binary_search_by(|x| cmp_f64(x, &c)) {
                    Ok(pos) => pos as i32,
                    Err(_) => -1,
                }
            });
        }
    }
    let n_valid = rows.len();
    if n_valid <= 10 {
        return resid_row;
    }
    let mut uniq: Vec<f64> = rows.iter().map(|r| r[0]).collect();
    uniq.sort_by(cmp_f64);
    uniq.dedup_by(|a, b| (a.is_nan() && b.is_nan()) || a == b);
    if uniq.len() == 1 {
        for j in 0..n {
            if valid[j] {
                resid_row[j] = 0.5;
            }
        }
        return resid_row;
    }
    let mut xtx: Vec<f64> = vec![0.0f64; p * p];
    let mut xty: Vec<f64> = vec![0.0f64; p];
    for (i, r) in rows.iter().enumerate() {
        let yv = r[0];
        for c in 0..k {
            let b = r[c + 1];
            xty[c] += b * yv;
            xtx[c * p + c] += b * b;
            for c2 in (c + 1)..k {
                let v = b * r[c2 + 1];
                xtx[c * p + c2] += v;
                xtx[c2 * p + c] += v;
            }
        }
        let ic = ind_cols[i];
        if ic >= 0 {
            let col = (k as i32 + ic) as usize;
            xty[col] += yv;
            xtx[col * p + col] += 1.0;
            for c in 0..k {
                let b = r[c + 1];
                xtx[c * p + col] += b;
                xtx[col * p + c] += b;
            }
        }
    }
    let m = DMatrix::from_row_slice(p, p, &xtx);
    let rhs = DMatrix::from_column_slice(p, 1, &xty);
    let use_svd = n_valid <= 40 || Cholesky::new(m.clone()).is_none();
    let coef: Vec<f64> = if !use_svd {
        let chol = Cholesky::new(m).expect("chol");
        chol.solve(&rhs).column(0).iter().copied().collect()
    } else {
        let n_r = rows.len();
        let xm = DMatrix::from_fn(n_r, p, |r_i, c| {
            let ic = ind_cols[r_i];
            if c < k {
                rows[r_i][c + 1]
            } else if ic >= 0 && c == k + ic as usize {
                1.0
            } else {
                0.0
            }
        });
        let svd = xm.clone().svd(true, true);
        let u = svd.u.expect("svd u");
        let vt = svd.v_t.expect("svd vt");
        let sv = svd.singular_values;
        let s_max = sv.iter().cloned().fold(0.0_f64, f64::max);
        let rcond = s_max * (n_r.max(p) as f64) * 2.22e-16;
        let ym = DMatrix::from_fn(n_r, 1, |r_i, _| rows[r_i][0]);
        let uty = u.transpose() * ym;
        let mut coef = DMatrix::zeros(p, 1);
        for i in 0..p {
            if sv[i] > rcond {
                coef[(i, 0)] = uty[(i, 0)] / sv[i];
            }
        }
        (vt.transpose() * coef).column(0).iter().copied().collect()
    };
    let mut vi = 0usize;
    for (i, r) in rows.iter().enumerate() {
        let mut pred = 0.0;
        for c in 0..k {
            pred += coef[c] * r[c + 1];
        }
        let ic = ind_cols[i];
        if ic >= 0 {
            pred += coef[k + ic as usize];
        }
        while !valid[vi] {
            vi += 1;
        }
        resid_row[vi] = r[0] - pred;
        vi += 1;
    }
    resid_row
}

/// OLS 行级快路径 (与 get_residual_v2 运算完全一致, 残差直接写 f32 行)。
/// 返回该行是否走了回退路径。
#[allow(clippy::too_many_arguments)]
fn ols_day_row_fast(
    fv_row: &[f64],
    shared: &NeutralizeStdShared,
    idx: usize,
    y_buf: &mut Vec<f64>,
    xty: &mut Vec<f64>,
    out_f32_row: &mut [f32],
) -> bool {
    let k = 10usize;
    let barra_ranked = &shared.barra_ranked;
    let ind1 = &shared.ind1;
    let (p, valid_idx, valid_cols, _xtx) = &shared.per_date[idx];
    if *p == 0 {
        return false;
    }
    y_buf.clear();
    let mut any_nan = false;
    let mut mn = f64::INFINITY;
    let mut mx = f64::NEG_INFINITY;
    for &j in valid_idx {
        let y = fv_row[j as usize];
        if !y.is_finite() {
            any_nan = true;
            break;
        }
        if y < mn {
            mn = y;
        }
        if y > mx {
            mx = y;
        }
        y_buf.push(y);
    }
    if any_nan {
        let barra_v: Vec<_> = (0..k).map(|c| barra_ranked[c].row(idx)).collect();
        let barra_row: Vec<&[f64]> = barra_v.iter().map(|r| r.as_slice().unwrap()).collect();
        let ind1_v = ind1.row(idx);
        let ind1_row = ind1_v.as_slice().unwrap();
        let row = ols_day_inline_row(fv_row, &barra_row, ind1_row);
        for (j, &v) in row.iter().enumerate() {
            out_f32_row[j] = if v.is_nan() { f32::NAN } else { v as f32 };
        }
        return true;
    }
    if mn == mx {
        for &j in valid_idx {
            out_f32_row[j as usize] = 0.5;
        }
        return false;
    }
    xty.clear();
    xty.resize(*p, 0.0);
    let xd = &shared.xdays[idx];
    for (pos, &yv) in y_buf.iter().enumerate() {
        let xrow = &xd[pos * k..pos * k + k];
        for c in 0..k {
            xty[c] += xrow[c] * yv;
        }
        let ic = valid_cols[pos];
        if ic >= 0 {
            xty[k + ic as usize] += yv;
        }
    }
    let use_svd = valid_idx.len() <= 40 || shared.chols[idx].is_none();
    let coef: Vec<f64> = if !use_svd {
        let chol = shared.chols[idx].as_ref().unwrap();
        let rhs = DMatrix::from_column_slice(*p, 1, &xty);
        chol.solve(&rhs).column(0).iter().copied().collect()
    } else {
        let n_r = valid_idx.len();
        let barra_v: Vec<_> = (0..k).map(|c| barra_ranked[c].row(idx)).collect();
        let barra_row: Vec<&[f64]> = barra_v.iter().map(|r| r.as_slice().unwrap()).collect();
        let rows: Vec<[f64; 11]> = valid_idx
            .iter()
            .map(|&j| {
                let ji = j as usize;
                let mut cur = [0.0_f64; 11];
                cur[0] = fv_row[ji];
                for c in 0..k {
                    cur[c + 1] = barra_row[c][ji];
                }
                cur
            })
            .collect();
        let xm = DMatrix::from_fn(n_r, *p, |r_i, c| {
            let ic = valid_cols[r_i];
            if c < k {
                rows[r_i][c + 1]
            } else if ic >= 0 && c == k + ic as usize {
                1.0
            } else {
                0.0
            }
        });
        let svd = xm.clone().svd(true, true);
        let u = svd.u.expect("svd u");
        let vt = svd.v_t.expect("svd vt");
        let sv = svd.singular_values;
        let s_max = sv.iter().cloned().fold(0.0_f64, f64::max);
        let rcond = s_max * (n_r.max(*p) as f64) * 2.22e-16;
        let ym = DMatrix::from_fn(n_r, 1, |r_i, _| rows[r_i][0]);
        let uty = u.transpose() * ym;
        let mut coef = DMatrix::zeros(*p, 1);
        for i in 0..*p {
            if sv[i] > rcond {
                coef[(i, 0)] = uty[(i, 0)] / sv[i];
            }
        }
        (vt.transpose() * coef).column(0).iter().copied().collect()
    };
    for (pos, &j) in valid_idx.iter().enumerate() {
        let ji = j as usize;
        let yv = y_buf[pos];
        let xrow = &xd[pos * k..pos * k + k];
        let mut pred = 0.0;
        for c in 0..k {
            pred += coef[c] * xrow[c];
        }
        let ic = valid_cols[pos];
        if ic >= 0 {
            pred += coef[k + ic as usize];
        }
        out_f32_row[ji] = (yv - pred) as f32;
    }
    false
}

/// P3: B 个面按日期批处理的中性化 (C' 语义: 无最终残差 rank, 输出 f32)。
/// 与逐面 neutralize_std_slot_f32_v2_resid 逐位一致 (沙箱 15 真实面 B=2/4/13 验证)。
/// industry_neutralize=false 时逐面回退旧路径 (引擎恒为 true)。
pub(crate) fn neutralize_std_slots_f32_v2_resid_batch(
    slots: &[ArrayView2<f32>],
    shared: &NeutralizeStdShared,
    industry_neutralize: bool,
) -> Result<Vec<Array2<f32>>, String> {
    let b = slots.len();
    if b == 0 {
        return Ok(Vec::new());
    }
    if !industry_neutralize {
        return slots
            .iter()
            .map(|s| neutralize_std_slot_f32_v2_resid(s.clone(), shared, false))
            .collect();
    }
    let (t, n) = slots[0].dim();
    if shared.industry.dim() != (t, n) || shared.restrict_f64.dim() != (t, n) {
        return Err("neutralize_std_block industry/restrict 形状不匹配".to_string());
    }
    let mut outs: Vec<Array2<f32>> = (0..b).map(|_| Array2::from_elem((t, n), f32::NAN)).collect();
    let mut pct: Vec<Vec<f64>> = vec![vec![0.0; n]; b];
    let mut filled: Vec<Vec<f64>> = vec![vec![0.0; n]; b];
    let mut keys32 = vec![0u32; n];
    let mut idxs: Vec<usize> = Vec::with_capacity(n);
    let mut tmp: Vec<usize> = Vec::with_capacity(n);
    let mut ranks64: Vec<f64> = Vec::with_capacity(n);
    let mut keys64: Vec<u64> = vec![0u64; n];
    let mut ys: Vec<f64> = Vec::with_capacity(n);
    let mut bs: Vec<f64> = Vec::with_capacity(n);
    let mut obs: Vec<bool> = Vec::with_capacity(n);
    let mut sv: Vec<f64> = Vec::with_capacity(n);
    let mut nan_mask: Vec<bool> = Vec::with_capacity(n);
    let mut y_buf: Vec<f64> = Vec::with_capacity(n);
    let mut xty: Vec<f64> = Vec::with_capacity(64);
    let base = &shared;
    let mut ind0_row = Vec::<f64>::with_capacity(n);
    for idx in 0..t {
        let (ind2_v, ind1_v, size_v) = (base.ind2.row(idx), base.ind1.row(idx), base.size_ranked.row(idx));
        let (zeros_v, mask_v, restrict_v) = (base.zeros.row(idx), base.ind1_mask.row(idx), base.restrict_f64.row(idx));
        let (o0_v, o1_v, o2_v) = (base.orders[0].row(idx), base.orders[1].row(idx), base.orders[2].row(idx));
        let (o3_v, o4_v) = (base.orders[3].row(idx), base.orders[4].row(idx));
        let ind2_r = ind2_v.as_slice().unwrap();
        let ind1_r = ind1_v.as_slice().unwrap();
        let size_r = size_v.as_slice().unwrap();
        let zeros_r = zeros_v.as_slice().unwrap();
        let mask_r = mask_v.as_slice().unwrap();
        let restrict_r = restrict_v.as_slice().unwrap();
        let o0 = o0_v.as_slice().unwrap();
        let o1 = o1_v.as_slice().unwrap();
        let o2 = o2_v.as_slice().unwrap();
        let o3 = o3_v.as_slice().unwrap();
        let o4 = o4_v.as_slice().unwrap();
        ind0_row.clear();
        ind0_row.extend(ind1_r.iter().map(|&v| if v.is_nan() { 0.0 } else { 1.0 }));
        for (f, slot) in slots.iter().enumerate() {
            let slot_row = slot.row(idx);
            rank_pct_row_from_f32_in(
                slot_row.as_slice().unwrap(),
                &mut pct[f],
                &mut idxs,
                &mut tmp,
                &mut keys32,
            );
            fill_ind_reg_row(
                &mut pct[f],
                [ind2_r, ind1_r, &ind0_row],
                size_r,
                [o0, o1, o2],
                &mut ys,
                &mut bs,
                &mut obs,
            );
            for (j, &v) in ind1_r.iter().enumerate() {
                if v.is_nan() {
                    pct[f][j] = f64::NAN;
                }
            }
            filled[f].copy_from_slice(&pct[f]);
        }
        for f in 0..b {
            nan_mask.clear();
            for &v in filled[f].iter() {
                nan_mask.push(v.is_nan());
            }
            if nan_mask.iter().any(|&x| x) {
                median_fill_level_row(&mut filled[f], ind2_r, None, o3, &nan_mask, &mut sv);
            }
            nan_mask.clear();
            for &v in filled[f].iter() {
                nan_mask.push(v.is_nan());
            }
            if nan_mask.iter().any(|&x| x) {
                median_fill_level_row(&mut filled[f], ind1_r, None, o4, &nan_mask, &mut sv);
            }
            nan_mask.clear();
            for &v in filled[f].iter() {
                nan_mask.push(v.is_nan());
            }
            if nan_mask.iter().any(|&x| x) {
                median_fill_level_row(&mut filled[f], zeros_r, Some(mask_r), o0, &nan_mask, &mut sv);
            }
            for (j, &v) in restrict_r.iter().enumerate() {
                if v != 0.0 {
                    filled[f][j] = f64::NAN;
                }
            }
            rank_pct_row_f64_in_place(&mut filled[f], &mut ranks64, &mut idxs, &mut tmp, &mut keys64);
            let mut out_row = outs[f].row_mut(idx);
            let _ = ols_day_row_fast(
                &filled[f],
                shared,
                idx,
                &mut y_buf,
                &mut xty,
                out_row.as_slice_mut().unwrap(),
            );
        }
    }
    Ok(outs)
}
