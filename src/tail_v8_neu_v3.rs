//! tail_v8_neu_v3：v3 中性化（沙箱 tail_perf_bench/src/v3.rs 的正式库移植）。
//!
//! ============================ 算法（沙箱已逐位验证） ============================
//! 进入中性化的 slot 是 `rank_and_fill_missing_cross_sectional_median` 的产物：所有
//! `restrict==0` 位置已被中位秩填满 → NaN 只可能出现在 `restrict!=0` 处。记
//!   P = {slot 有限}（生产 rank1 的域）、S = {restrict==0 且 ind1 有限}（填充后仍有限）、
//!   V = {restrict==0 且 10 风格全有限}（= per_date.valid_idx，OLS 有效集）。
//! 实测 |P|≈4135、|S|≈3651、|V|≈3624 且 V ⊆ S，于是：
//!   推论 1：3 级行业 OLS 填充 + 3 级中位填充只写 NaN 位置，而 S 上无 NaN、S 之外随后被
//!           `restrict!=0 → NaN` / `ind1 NaN → NaN` 抹掉 → 整段可跳过。
//!   推论 2：rank2 是 rank1 值的严格单调函数 → rank2 的序 = slot 值在 S 上的序，
//!           P 与第二次 u64 基数排序完全不需要，直接在 S 上排一次即可。
//!   推论 3：S/V 只依赖 restrict + 行业 + 风格，与因子/面无关 → 全 run 预计算一次。
//! 每 (因子, 面, 日) 只剩：gather |S| → 4 趟 u32 基数排序 → 组游走出 rank pct →
//! |V| 的 OLS（与生产 `ols_day_row_fast` 完全同一算术）。
//!
//! ============================ 按日多面批处理（P3b） ============================
//! `v3_slots_range` 外层是「日」，内层一次处理同日全部 b 个面（v8 里 b=13）。同一天 b 个面
//! 共用同一套「只随日期变化」的量：S/V/D 索引、o0c 分段、每日 Cholesky、连续 X（`xdays`）
//! 与 `valid_cols`。于是把原本每 (面,日) 各自完成的末步合并：
//!   ① `ols_check_*` 逐面判定（写 0.5 / 判 SVD / 判生产回退）；
//!   ② `ols_acc_day` 以「位置外层 / 面内层」累加 X'y —— 同一行 `xd` 只读一次供 13 面复用，
//!      每面看到的加数序列（pos 升序、列升序）与逐面版本逐字相同；
//!   ③ b 个 X'y 组成 (p × b) 的 RHS，一次 `Cholesky::solve_mut` 解出全部 b 列；
//!   ④ 逐面写回残差（pred 累加顺序不变）。
//! nalgebra 的 `solve_mut` 内部是 `for j in 0..b.ncols()` 逐列调用同一个下三角求解、列间无
//! 耦合 → 每列结果与单列 `chol.solve(&DMatrix::from_column_slice(p,1,..))` 逐位相同
//! （已用独立 nalgebra 程序对 p∈{10,11,12,38,41,55}、b∈{1,7,13}、带 leading dimension 的
//! 视图缓冲实测 0 bit 差异）。`shared.chols[idx]` 的每日预分解本身完全不动。
//!
//! 热路径零堆分配：`V3Scratch` 里新增 `b_y/b_xty/b_coef/rhs` 与回退路径的
//! `ind0_row/rank_vals/seen` 缓冲，容量一旦涨到最大值即不再分配；`ols_day_core` 的
//! `X'y` 也改为复用调用方缓冲。仅 `M_SVD`（|V|<=40 或 Cholesky 失败，实测罕见）与生产行级
//! 回退仍按面走原实现（逐位优先）。
//!
//! **逐位一致**：与 `crate::factor_neutralize_std::neutralize_std_slots_f32_v2_resid_batch`
//! 完全相同（含 NaN 位置）；出现洞（S 中有 NaN）或非快路径日时逐行回退到生产行级实现
//! （`v3_row_prod`）。
//!
//! ============================ 正式库移植说明 ============================
//! - `V3Shared::build(ns: Arc<NeutralizeStdShared>)`：只预计算压缩索引（几十 MB），
//!   `NeutralizeStdShared` 本身用 `Arc` 共享、**不复制**（它含 ~5GB 的 barra/orders/xdays）。
//! - 沙箱里 `shared.v2.base.X` / `shared.v2.X` 的访问已映射到 `NeutralizeStdShared` 的
//!   同名字段（industry/restrict_f64/ind1/ind2/zeros/ind1_mask/barra_ranked/size_ranked/
//!   orders/per_date/chols/xdays）。
//! - 沙箱 v2.rs 的行级 helper（rank_pct/radix/fill_ind_reg/median_fill/ols_day_*）在生产库里
//!   是私有的，无法跨模块引用，故原样复制到本文件（逐位一致，注释里标明出处）。
//! - 对外：`v3_slots_range`（v8 融合流水线入口，按日期块）、`v3_slot`（整张，对账/回落）、
//!   `V3Scratch`（每线程一份，跨块复用）、`selfcheck`（与生产 v2 逐位对账 + 计时）。
//!
//! 自测：`rp.tail_v8_selfcheck("neu3", data_dir)`。

use std::cmp::Ordering;
use std::sync::Arc;
use std::time::Instant;

use nalgebra::{Cholesky, DMatrix};
use ndarray::{Array2, ArrayView2};

use crate::factor_neutralize_std::NeutralizeStdShared;


// ==================== 基础原语与行级 helper（沙箱 v2.rs 原样移植，逐位一致） ====================

fn cmp_f64(a: &f64, b: &f64) -> Ordering {
    match (a.is_nan(), b.is_nan()) {
        (true, true) => Ordering::Equal,
        (true, false) => Ordering::Greater,
        (false, true) => Ordering::Less,
        (false, false) => a.partial_cmp(b).unwrap(),
    }
}

fn mono_key(v: f64) -> u64 {
    let b = v.to_bits();
    if v.is_nan() {
        u64::MAX
    } else if b >> 63 == 0 {
        b | (1u64 << 63)
    } else {
        !b
    }
}

fn mono_key32(v: f32) -> u32 {
    let b = v.to_bits();
    if v.is_nan() {
        u32::MAX
    } else if b >> 31 == 0 {
        b | (1u32 << 31)
    } else {
        !b
    }
}

fn radix_sort_order(keys: &[u64], order: &mut Vec<usize>, tmp: &mut Vec<usize>) {
    let n = order.len();
    if n <= 1 {
        return;
    }
    // 长度已是 n 时 `resize` 零开销（避免每次 memset 整个 tmp）
    tmp.resize(n, 0);
    let mut counts = [0usize; 256];
    for pass in 0..8 {
        let shift = pass * 8;
        counts.fill(0);
        for &i in order.iter() {
            counts[((keys[i] >> shift) & 0xff) as usize] += 1;
        }
        if counts.iter().any(|&c| c == n) {
            continue;
        }
        let mut sum = 0usize;
        for c in counts.iter_mut() {
            let t = *c;
            *c = sum;
            sum += t;
        }
        for &i in order.iter() {
            let d = ((keys[i] >> shift) & 0xff) as usize;
            tmp[counts[d]] = i;
            counts[d] += 1;
        }
        std::mem::swap(order, tmp);
    }
}

fn radix_sort_order32(keys: &[u32], order: &mut Vec<usize>, tmp: &mut Vec<usize>) {
    let n = order.len();
    if n <= 1 {
        return;
    }
    // 长度已是 n 时 `resize` 零开销（避免每次 memset 整个 tmp）
    tmp.resize(n, 0);
    let mut counts = [0usize; 256];
    for pass in 0..4 {
        let shift = pass * 8;
        counts.fill(0);
        for &i in order.iter() {
            counts[((keys[i] >> shift) & 0xff) as usize] += 1;
        }
        if counts.iter().any(|&c| c == n) {
            continue;
        }
        let mut sum = 0usize;
        for c in counts.iter_mut() {
            let t = *c;
            *c = sum;
            sum += t;
        }
        for &i in order.iter() {
            let d = ((keys[i] >> shift) & 0xff) as usize;
            tmp[counts[d]] = i;
            counts[d] += 1;
        }
        std::mem::swap(order, tmp);
    }
}

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

/// `seen` 由调用方提供（scratch 复用），热路径零分配。
fn has_ge_n_unique(vals: &[f64], need: usize, seen: &mut Vec<f64>) -> bool {
    seen.clear();
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
    keys.resize(n, 0);
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

/// 行级 fill_ind_reg (3 级顺序就地执行, 每级判断唯一值/跳过干净段)。
/// 与生产 fill_ind_reg_pre 对同一行状态的操作序列一致 (级别内段不重叠→就地等价)。
#[allow(clippy::too_many_arguments)]
fn fill_ind_reg_row(
    row: &mut [f64],
    level_rows: [&[f64]; 3],
    size_row: &[f64],
    order_rows: [&[usize]; 3],
    ys: &mut Vec<f64>,
    bs: &mut Vec<f64>,
    obs: &mut Vec<bool>,
    seen: &mut Vec<f64>,
) {
    let n = row.len();
    for li in 0..3 {
        if !has_ge_n_unique(row, 10, seen) {
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

/// 行级中位填充 (单级; nan_mask 由调用方按"当前行状态"构建, 与生产每 call 重建一致)。
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

/// 生产 ols_day_inline 的行级移植 (回退路径, 逐位一致)。
fn ols_day_inline_row(
    fv_row: &[f64],
    barra_row: &[&[f64]],
    ind1_row: &[f64],
) -> Vec<f64> {
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

/// OLS 行级快路径: 与 get_residual_v2 (O3a/c/d) 完全一致的运算, 残差直接写 f32 行。
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
        let barra_v: Vec<_> = (0..k).map(|c| shared.barra_ranked[c].row(idx)).collect();
        let barra_row: Vec<&[f64]> = barra_v.iter().map(|r| r.as_slice().unwrap()).collect();
        let ind1_v = shared.ind1.row(idx);
        let ind1_row = ind1_v.as_slice().unwrap();
        let row = ols_day_inline_row(fv_row, &barra_row, ind1_row);
        for (j, &v) in row.iter().enumerate() {
            out_f32_row[j] = if v.is_nan() { f32::NAN } else { v as f32 };
        }
        return true;
    }
    ols_day_core(y_buf, shared, idx, out_f32_row, xty);
    false
}

/// OLS 核心：生产 ols_day_row_fast 的 "mn==mx 之后" 段原样抽出，
/// 供行级路径与 V3 压缩路径共用（同一份代码 → 同一份 codegen → 逐位一致）。
/// `xty_buf` 由调用方提供（scratch 复用），热路径不再分配 `X'y`。
fn ols_day_core(
    y_buf: &[f64],
    shared: &NeutralizeStdShared,
    idx: usize,
    out_f32_row: &mut [f32],
    xty_buf: &mut Vec<f64>,
) {
    let k = 10usize;
    let (p, valid_idx, valid_cols, _xtx) = &shared.per_date[idx];
    if *p == 0 {
        return;
    }
    let mut mn = f64::INFINITY;
    let mut mx = f64::NEG_INFINITY;
    for &y in y_buf {
        if y < mn {
            mn = y;
        }
        if y > mx {
            mx = y;
        }
    }
    if mn == mx {
        for &j in valid_idx {
            out_f32_row[j as usize] = 0.5;
        }
        return;
    }
    xty_buf.clear();
    xty_buf.resize(*p, 0.0);
    let xty: &mut [f64] = &mut xty_buf[..];
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
        let rhs = DMatrix::from_column_slice(*p, 1, xty);
        chol.solve(&rhs).column(0).iter().copied().collect()
    } else {
        let n_r = valid_idx.len();
        let barra_v: Vec<_> = (0..k).map(|c| shared.barra_ranked[c].row(idx)).collect();
        let barra_row: Vec<&[f64]> = barra_v.iter().map(|r| r.as_slice().unwrap()).collect();
        let rows: Vec<[f64; 11]> = valid_idx
            .iter()
            .map(|&j| {
                let ji = j as usize;
                let mut cur = [0.0_f64; 11];
                cur[0] = y_buf[valid_idx.iter().position(|&x| x == j).unwrap()];
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
}

// ==================== OLS 末步拆分（准备 / 求解 / 写回），供按日批处理复用 ====================
//
// 拆分只改变"何时做"，不改变"怎么做"：准备段与 `ols_day_core` / `ols_day_core_style` 的前半段
// 逐位同序（mn/mx 扫描、X'y 同序累加），写回段与它们的末段逐位同序（同样的 pred 累加顺序、
// 同样的 f32 转换）。中间的 Cholesky solve 在 nalgebra 里是 `for j in 0..b.ncols()` 逐列调用
// 同一个 `solve_lower_triangular_vector_unchecked_mut` → 多列 RHS 与单列 RHS 的每列结果逐位相同。

/// 行业路径末步判定（只判定，不做 `X'y` 累加）：`M_DONE` / `M_READY` / `M_SVD`。
/// `X'y` 由 `ols_acc_day` 在「位置外层 / 面内层」里统一累加（同一天 13 面共用一次
/// `xd` / `valid_cols` 遍历，缓存友好），每面看到的加数序列与逐面版本完全一致。
fn ols_check_industry(
    y_v: &[f64],
    shared: &NeutralizeStdShared,
    idx: usize,
    out_f32_row: &mut [f32],
) -> u8 {
    let (p, valid_idx, _valid_cols, _xtx) = &shared.per_date[idx];
    if *p == 0 {
        return M_DONE;
    }
    let mut mn = f64::INFINITY;
    let mut mx = f64::NEG_INFINITY;
    for &y in y_v {
        if y < mn {
            mn = y;
        }
        if y > mx {
            mx = y;
        }
    }
    if mn == mx {
        for &j in valid_idx {
            out_f32_row[j as usize] = 0.5;
        }
        return M_DONE;
    }
    if valid_idx.len() <= 40 || shared.chols[idx].is_none() {
        M_SVD
    } else {
        M_READY
    }
}

/// 纯风格路径末步判定（只判定，不做 `X'y` 累加）：`M_DONE` / `M_READY` / `M_STYLE_FB`。
fn ols_check_style(
    y_v: &[f64],
    ns: &NeutralizeStdShared,
    idx: usize,
    out_f32_row: &mut [f32],
) -> u8 {
    let (pn, valid_idx, _valid_cols, _xtx) = &ns.per_date[idx];
    if *pn == 0 {
        return M_DONE;
    }
    let mut mn = f64::INFINITY;
    let mut mx = f64::NEG_INFINITY;
    for &y in y_v {
        if y < mn {
            mn = y;
        }
        if y > mx {
            mx = y;
        }
    }
    if mn == mx {
        for &j in valid_idx {
            out_f32_row[j as usize] = 0.5;
        }
        return M_DONE;
    }
    if valid_idx.len() <= 40 || ns.chols_style[idx].is_none() {
        M_STYLE_FB
    } else {
        M_READY
    }
}

/// 按日 13 面共用一次 X 遍历的 `X'y` 累加。
///
/// **逐位不变性**：对外层 pos 升序、内层对每个面 f 仍是 `xty_f[c] += xd[pos*k+c] * y_f[pos]`
/// （c 升序），随后 `xty_f[k+ic] += y_f[pos]` —— 与逐面版本的加数序列逐字相同，只是把同一行
/// `xd` 的读取在 13 个面之间复用（少 12/13 的 `xd` 访存）。行业与纯风格的列布局不同，各自成段。
fn ols_acc_day(
    sc: &mut V3Scratch,
    ns: &NeutralizeStdShared,
    idx: usize,
    industry_neutralize: bool,
    b: usize,
    p_alloc: usize,
) {
    let k = 10usize;
    let nv = ns.per_date[idx].1.len();
    if nv == 0 {
        return;
    }
    // 所有走压缩路径的面其 y 长度恒为 |V|（identity 日 |S| == |V|；其余按 v_from_s 收集）。
    debug_assert!((0..b).all(|f| sc.b_mode[f] != M_READY || sc.b_n[f] == nv));
    let stride = sc.b_stride;
    let xd = &ns.xdays[idx];
    let valid_cols = &ns.per_date[idx].2;
    {
        let modes: &[u8] = &sc.b_mode[..b];
        let y_all: &[f64] = &sc.b_y[..];
        let xty: &mut [f64] = &mut sc.b_xty[..];
        for f in 0..b {
            if modes[f] == M_READY {
                let base = f * p_alloc;
                for v in xty[base..base + p_alloc].iter_mut() {
                    *v = 0.0;
                }
            }
        }
        if industry_neutralize {
            for pos in 0..nv {
                let xrow = &xd[pos * k..pos * k + k];
                let ic = valid_cols[pos];
                for f in 0..b {
                    if modes[f] != M_READY {
                        continue;
                    }
                    let yv = y_all[f * stride + pos];
                    let base = f * p_alloc;
                    for c in 0..k {
                        xty[base + c] += xrow[c] * yv;
                    }
                    if ic >= 0 {
                        xty[base + k + ic as usize] += yv;
                    }
                }
            }
        } else {
            for pos in 0..nv {
                let xrow = &xd[pos * k..pos * k + k];
                for f in 0..b {
                    if modes[f] != M_READY {
                        continue;
                    }
                    let yv = y_all[f * stride + pos];
                    let base = f * p_alloc;
                    xty[base] += yv;
                    for c in 0..k {
                        xty[base + 1 + c] += xrow[c] * yv;
                    }
                }
            }
        }
    }
}

/// 行业路径残差写回：与 `ols_day_core` 末段逐位同序。
fn ols_write_industry(
    y_v: &[f64],
    shared: &NeutralizeStdShared,
    idx: usize,
    out_f32_row: &mut [f32],
    coef: &[f64],
) {
    let k = 10usize;
    let (_p, valid_idx, valid_cols, _xtx) = &shared.per_date[idx];
    let xd = &shared.xdays[idx];
    for (pos, &j) in valid_idx.iter().enumerate() {
        let ji = j as usize;
        let yv = y_v[pos];
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
}

/// 纯风格残差写回：与 `ols_day_core_style` 末段逐位同序（含 NaN 归一）。
fn ols_write_style(
    y_v: &[f64],
    ns: &NeutralizeStdShared,
    idx: usize,
    out_f32_row: &mut [f32],
    coef: &[f64],
) {
    let k = 10usize;
    let (_pn, valid_idx, _valid_cols, _xtx) = &ns.per_date[idx];
    let xd = &ns.xdays[idx];
    for (pos, &j) in valid_idx.iter().enumerate() {
        let ji = j as usize;
        let xrow = &xd[pos * k..pos * k + k];
        let mut pred = coef[0];
        for c in 0..k {
            pred += coef[c + 1] * xrow[c];
        }
        let r = y_v[pos] - pred;
        out_f32_row[ji] = if r.is_nan() { f32::NAN } else { r as f32 };
    }
}

/// 行级 f64 rank (就地写回), 与 rank_pct_all 的逐行语义一致。
/// `vals` 是调用方的 scratch 副本缓冲（避免每行一次 `to_vec` 分配）。
fn rank_pct_row_f64_in_place(
    row: &mut [f64],
    ranks: &mut Vec<f64>,
    idxs: &mut Vec<usize>,
    tmp: &mut Vec<usize>,
    keys: &mut Vec<u64>,
    vals: &mut Vec<f64>,
) {
    // 生产 rank_pct_row_into 从 vals 读入 idxs/keys 后写 ranks, 再拷回 row;
    // 就地时先取副本防读写交错 —— 直接复用 rank_pct_row_into 的读语义。
    vals.clear();
    vals.extend_from_slice(row);
    rank_pct_row_into(vals, ranks, idxs, tmp, keys);
    row.copy_from_slice(ranks);
}

// ==================== 行级临时缓冲 ====================

#[derive(Default, Clone, Copy)]
pub struct V3Times {
    pub gather: f64,
    pub sort: f64,
    pub walk: f64,
    pub ols: f64,
    pub fallback: f64,
    pub fast_rows: u64,
    pub slow_rows: u64,
}

impl V3Times {
    pub fn total(&self) -> f64 {
        self.gather + self.sort + self.walk + self.ols + self.fallback
    }
}

/// 按日多面批处理的末步状态。
///
/// - `M_DONE`：该面末步无需任何求解（`p == 0`，或 y 全等已直接写 0.5，或压缩路径空集）。
/// - `M_READY`：`X'y` 已按列累加进 `b_xty`，等待与同日其它面一起做一次 (p × b) 求解。
/// - `M_SVD`：行业路径需 SVD（日级判定：`|V| <= 40` 或 Cholesky 失败）→ 交 `ols_day_core`。
/// - `M_STYLE_FB`：纯风格路径 `|V| <= 40` 或 `chols_style` 缺失 → 交生产行级回退。
/// - `M_PROD`：压缩流水线失败（S 有洞 / 特殊位置不可精确复算）→ 交生产行级回退。
const M_DONE: u8 = 0;
const M_READY: u8 = 1;
const M_SVD: u8 = 2;
const M_STYLE_FB: u8 = 3;
const M_PROD: u8 = 4;

/// 行级临时缓冲（每线程一份，跨行复用，热路径零分配）。
pub struct V3Scratch {
    yv: Vec<f32>,
    pv: Vec<f32>,
    pi: Vec<u32>,
    keys: Vec<u32>,
    order: Vec<usize>,
    tmp: Vec<usize>,
    ybuf: Vec<f64>,
    y_v: Vec<f64>,
    xty: Vec<f64>,
    /// 纯风格行级回退：`get_residual_row_style` 的输出缓冲
    style_resid: Vec<f64>,
    // 回退路径缓冲
    pct: Vec<f64>,
    filled: Vec<f64>,
    idxs: Vec<usize>,
    keys64: Vec<u64>,
    ys: Vec<f64>,
    bs: Vec<f64>,
    obs: Vec<bool>,
    sv: Vec<f64>,
    nan_mask: Vec<bool>,
    y_buf_fb: Vec<f64>,
    // D 路径缓冲
    pct_full: Vec<f64>,
    touched: Vec<u32>,
    v_val: Vec<f64>,
    v_spos: Vec<u32>,
    d_val: Vec<f64>,
    d_spos: Vec<u32>,
    vpos: Vec<u32>,
    x_list: Vec<u32>,
    x_spos: Vec<u32>,
    x_val: Vec<f64>,
    group: Vec<u32>,
    // ---- 按日多面批处理（b 个面共用一次 gather 索引 / 有效集合 / 一次 solve）----
    /// `b_y` 每面步长（>= n_stocks），容量一旦涨到最大值就不再变
    b_stride: usize,
    /// `b_y` 已分配的面数
    b_cap: usize,
    /// 面 f 的 V 序 y 存于 `b_y[f * b_stride ..]`（长度 `b_n[f]`）
    b_y: Vec<f64>,
    b_n: Vec<usize>,
    b_mode: Vec<u8>,
    /// 列主序 p_alloc × b：面 f 的 `X'y` 存于 `b_xty[f * p_alloc ..]`
    b_xty: Vec<f64>,
    /// 列主序 p_alloc × b：面 f 的解存于 `b_coef[f * p_alloc ..]`
    b_coef: Vec<f64>,
    /// 复用的 RHS 矩阵（>= p_solve × b），`solve_mut` 直接写它 → 热路径零分配
    rhs: DMatrix<f64>,
    // ---- 生产回退路径的零分配缓冲 ----
    ind0_row: Vec<f64>,
    rank_vals: Vec<f64>,
    /// `has_ge_n_unique` 的唯一值去重缓冲
    seen: Vec<f64>,
}

impl V3Scratch {
    pub fn new(n: usize) -> Self {
        V3Scratch {
            yv: Vec::with_capacity(n),
            pv: Vec::with_capacity(n),
            pi: Vec::with_capacity(n),
            keys: Vec::with_capacity(n),
            order: Vec::with_capacity(n),
            tmp: Vec::with_capacity(n),
            ybuf: Vec::with_capacity(n),
            y_v: Vec::with_capacity(n),
            xty: Vec::with_capacity(64),
            style_resid: Vec::with_capacity(n),
            pct: Vec::with_capacity(n),
            filled: Vec::with_capacity(n),
            idxs: Vec::with_capacity(n),
            keys64: Vec::with_capacity(n),
            ys: Vec::with_capacity(n),
            bs: Vec::with_capacity(n),
            obs: Vec::with_capacity(n),
            sv: Vec::with_capacity(n),
            nan_mask: Vec::with_capacity(n),
            y_buf_fb: Vec::with_capacity(n),
            pct_full: vec![f64::NAN; n],
            touched: Vec::with_capacity(n),
            v_val: Vec::with_capacity(n),
            v_spos: Vec::with_capacity(n),
            d_val: Vec::with_capacity(64),
            d_spos: Vec::with_capacity(64),
            vpos: Vec::with_capacity(n),
            x_list: Vec::with_capacity(256),
            x_spos: Vec::with_capacity(256),
            x_val: Vec::with_capacity(256),
            group: Vec::with_capacity(64),
            b_stride: 0,
            b_cap: 0,
            b_y: Vec::new(),
            b_n: Vec::new(),
            b_mode: Vec::new(),
            b_xty: Vec::new(),
            b_coef: Vec::new(),
            rhs: DMatrix::zeros(0, 0),
            ind0_row: Vec::with_capacity(n),
            rank_vals: Vec::with_capacity(n),
            seen: Vec::with_capacity(16),
        }
    }

    /// 保证 `b_y` / `b_n` / `b_mode` 至少覆盖 b 个面、每面 n 个元素（只在容量不足时分配）。
    fn ensure_batch(&mut self, b: usize, n: usize) {
        if self.b_cap >= b && self.b_stride >= n {
            return;
        }
        self.b_cap = self.b_cap.max(b);
        self.b_stride = self.b_stride.max(n);
        self.b_y.resize(self.b_cap * self.b_stride, 0.0);
        self.b_n.resize(self.b_cap, 0);
        self.b_mode.resize(self.b_cap, M_DONE);
    }

    /// 保证 `b_xty` / `b_coef` / `rhs` 至少覆盖 b 面 × p 列（只在容量不足时分配）。
    fn ensure_xty(&mut self, b: usize, p: usize) {
        let need = b * p;
        if self.b_xty.len() < need {
            self.b_xty.resize(need, 0.0);
            self.b_coef.resize(need, 0.0);
        }
        if self.rhs.nrows() < p || self.rhs.ncols() < b {
            let nr = p.max(64);
            let nc = b.max(16);
            self.rhs = DMatrix::zeros(nr, nc);
        }
    }
}

// ==================== V3Shared（派生索引 + Arc<NeutralizeStdShared>） ====================
pub struct V3Shared {
    /// 中性化预计算（引擎持有，Arc 共享；不复制大矩阵）
    ns: Arc<NeutralizeStdShared>,
    pub n_stocks: usize,
    /// 每日 S（升序原始下标），尾部以 u32::MAX 填充到 n_stocks
    pub s_idx_flat: Vec<u32>,
    pub s_lens: Vec<u32>,
    /// 每日 V 位置 -> S 位置（len = |V|）
    pub v_from_s_flat: Vec<u32>,
    pub v_lens: Vec<u32>,
    pub v_offsets: Vec<u64>,
    /// 该日可走压缩路径（|V|>10 且 V ⊆ S）
    pub fast_ok: Vec<bool>,
    /// 该日 S == V（元素与顺序都一致）
    pub identity: Vec<bool>,
    pub small_days: usize,
    pub v_notin_s_days: usize,
    pub identity_days: usize,
    /// D = S \ V（size NaN 的位置），扁平存储（无填充，按 offsets 切）
    pub d_flat: Vec<u32>,
    pub d_offsets: Vec<u64>,
    pub d_lens: Vec<u32>,
    /// D 元素在 S 中的位置（与 d_flat 平行）
    pub d_s_pos_flat: Vec<u32>,
    /// 每日 ind2 排序（= orders[0] = orders[3]）压缩到"已上市"宇宙：|L| 长，尾部填充
    pub o0c_flat: Vec<u32>,
    pub o0c_lens: Vec<u32>,
    pub o0c_offsets: Vec<u64>,
    /// j -> 在 o0c 中的位置（T×N，u32::MAX = 不在 L）
    pub pos0_flat: Vec<u32>,
    /// j -> 在 S 中的位置（T×N，u32::MAX = 不在 S）
    pub s_pos_flat: Vec<u32>,
    /// 每日 V（= S\D）的升序原始下标，尾部 u32::MAX 填充
    pub v_idx_flat: Vec<u32>,
    /// j 是否属于 S（T×N，0/1）
    pub in_s_flat: Vec<u8>,
    /// j 是否属于 D（T×N，0/1）
    pub in_d_flat: Vec<u8>,
    /// o0c 的分段起点（按 ind2 码），扁平 + offsets
    pub seg_flat: Vec<u32>,
    pub seg_offsets: Vec<u64>,
    pub seg_lens: Vec<u32>,
    pub has_d: bool,
}

impl V3Shared {
    /// 预计算「只随日期变化」的压缩索引：S/V/D、o0c 分段、S 位置映射等。
    /// 与因子无关，全 run 建一次。
    pub fn build(ns: Arc<NeutralizeStdShared>) -> Result<V3Shared, String> {
        let (t, n) = ns.industry.dim();
        let mut s_idx_flat = vec![u32::MAX; t * n];
        let mut s_lens = vec![0u32; t];
        let mut v_lens = vec![0u32; t];
        let mut v_offsets = vec![0u64; t + 1];
        let mut v_from_s_flat: Vec<u32> = Vec::with_capacity(t * n);
        let mut fast_ok = vec![false; t];
        let mut identity = vec![false; t];
        let mut small_days = 0usize;
        let mut v_notin_s_days = 0usize;
        let mut identity_days = 0usize;
        let mut d_flat: Vec<u32> = Vec::new();
        let mut d_offsets: Vec<u64> = vec![0u64; t + 1];
        let mut d_lens = vec![0u32; t];
        let mut d_s_pos_flat: Vec<u32> = Vec::new();
        let mut o0c_flat: Vec<u32> = Vec::new();
        let mut o0c_lens = vec![0u32; t];
        let mut o0c_offsets: Vec<u64> = vec![0u64; t + 1];
        let mut pos0_flat = vec![u32::MAX; t * n];
        let mut s_pos_flat = vec![u32::MAX; t * n];
        let mut v_idx_flat = vec![u32::MAX; t * n];
        let mut in_s_flat = vec![0u8; t * n];
        let mut in_d_flat = vec![0u8; t * n];
        let mut seg_flat: Vec<u32> = Vec::new();
        let mut seg_offsets: Vec<u64> = vec![0u64; t + 1];
        let mut seg_lens = vec![0u32; t];
        let mut has_d = false;
        let ind1 = &ns.ind1;
        let restrict = &ns.restrict_f64;
        for idx in 0..t {
            let base = idx * n;
            let mut s_pos = vec![u32::MAX; n];
            let mut n_s = 0usize;
            for j in 0..n {
                if restrict[[idx, j]] == 0.0 && ind1[[idx, j]].is_finite() {
                    s_idx_flat[base + n_s] = j as u32;
                    s_pos[j] = n_s as u32;
                    s_pos_flat[base + j] = n_s as u32;
                    in_s_flat[base + j] = 1;
                    n_s += 1;
                }
            }
            s_lens[idx] = n_s as u32;
            // ---- D = S \ V ----
            let mut is_v = vec![false; n];
            for &j in ns.per_date[idx].1.iter() {
                is_v[j as usize] = true;
            }
            let mut nd = 0usize;
            let mut nvv = 0usize;
            for k in 0..n_s {
                let j = s_idx_flat[base + k];
                if !is_v[j as usize] {
                    d_flat.push(j);
                    d_s_pos_flat.push(k as u32);
                    in_d_flat[base + j as usize] = 1;
                    nd += 1;
                } else {
                    v_idx_flat[base + nvv] = j;
                    nvv += 1;
                }
            }
            d_lens[idx] = nd as u32;
            d_offsets[idx + 1] = d_offsets[idx] + nd as u64;
            if nd > 0 {
                has_d = true;
            }
            // ---- o0c: ind2 排序压缩到已上市宇宙 + 分段 ----
            {
                let ord = ns.orders[0].row(idx);
                let ord = ord.as_slice().unwrap();
                let mut lc = 0usize;
                let mut seg_start = 0usize;
                let mut nseg = 0usize;
                let mut prev_code = f64::NAN;
                for &jj in ord.iter() {
                    let code = ns.ind2[[idx, jj]];
                    if code.is_nan() {
                        break;
                    }
                    if lc > 0 && code != prev_code {
                        seg_flat.push(seg_start as u32);
                        nseg += 1;
                        seg_start = lc;
                    }
                    prev_code = code;
                    pos0_flat[idx * n + jj] = lc as u32;
                    o0c_flat.push(jj as u32);
                    lc += 1;
                }
                if lc > 0 {
                    seg_flat.push(seg_start as u32);
                    nseg += 1;
                }
                o0c_lens[idx] = lc as u32;
                o0c_offsets[idx + 1] = o0c_offsets[idx] + lc as u64;
                seg_lens[idx] = nseg as u32;
                seg_offsets[idx + 1] = seg_offsets[idx] + nseg as u64;
            }
            let vi = &ns.per_date[idx].1;
            v_lens[idx] = vi.len() as u32;
            v_offsets[idx + 1] = v_offsets[idx] + vi.len() as u64;
            let mut all_in_s = true;
            let mut is_identity = true;
            for (pos, &j) in vi.iter().enumerate() {
                let p = s_pos[j as usize];
                if p == u32::MAX {
                    all_in_s = false;
                }
                if p != pos as u32 {
                    is_identity = false;
                }
                v_from_s_flat.push(p);
            }
            if !all_in_s {
                v_notin_s_days += 1;
            }
            if is_identity && n_s == vi.len() {
                identity[idx] = true;
                identity_days += 1;
            }
            if vi.len() <= 10 {
                small_days += 1;
            }
            // 可走压缩路径的条件：V ⊆ S 且 |V| > 10。若 |V| == |S|（D 为空）用 v3_row_fast
            // （填充步骤对输出恒等）；否则用 v3_row_d（精确复算 D 的中位填充值）。
            fast_ok[idx] = all_in_s && vi.len() > 10;
        }
        Ok(V3Shared {
            ns,
            n_stocks: n,
            s_idx_flat,
            s_lens,
            v_from_s_flat,
            v_lens,
            v_offsets,
            fast_ok,
            identity,
            small_days,
            v_notin_s_days,
            identity_days,
            d_flat,
            d_offsets,
            d_lens,
            d_s_pos_flat,
            o0c_flat,
            o0c_lens,
            o0c_offsets,
            pos0_flat,
            s_pos_flat,
            v_idx_flat,
            in_s_flat,
            in_d_flat,
            seg_flat,
            seg_offsets,
            seg_lens,
            has_d,
        })
    }

    /// 需要时取回底层预计算（形状校验/诊断用）。
    pub fn ns(&self) -> &NeutralizeStdShared {
        &self.ns
    }
}

// ==================== 行级中性化（快路径 / 特殊位置路径 / 生产回退） ====================

/// 生产行级填充链（行业/纯风格共用）：rank1 → fill_ind_reg → ind1 NaN → 三级中位填充
/// → restrict 掩码 → rank2。结果留在 `sc.filled`，供两个末步（行业 OLS / 纯风格残差）复用。
/// 与 `neutralize_std_slots_f32_v2_resid_batch` 的单行完全一致。
fn v3_row_prod_fill(row: &[f32], idx: usize, shared: &V3Shared, sc: &mut V3Scratch) {
    let n = row.len();
    let base: &NeutralizeStdShared = &shared.ns;
    let ind2_v = base.ind2.row(idx);
    let ind1_v = base.ind1.row(idx);
    let size_v = base.size_ranked.row(idx);
    let zeros_v = base.zeros.row(idx);
    let mask_v = base.ind1_mask.row(idx);
    let restrict_v = base.restrict_f64.row(idx);
    let (ind2_r, ind1_r, size_r) = (
        ind2_v.as_slice().unwrap(),
        ind1_v.as_slice().unwrap(),
        size_v.as_slice().unwrap(),
    );
    let (zeros_r, mask_r, restrict_r) = (
        zeros_v.as_slice().unwrap(),
        mask_v.as_slice().unwrap(),
        restrict_v.as_slice().unwrap(),
    );
    let o0v = shared.ns.orders[0].row(idx);
    let o1v = shared.ns.orders[1].row(idx);
    let o2v = shared.ns.orders[2].row(idx);
    let o3v = shared.ns.orders[3].row(idx);
    let o4v = shared.ns.orders[4].row(idx);
    let (o0, o1, o2) = (
        o0v.as_slice().unwrap(),
        o1v.as_slice().unwrap(),
        o2v.as_slice().unwrap(),
    );
    let (o3, o4) = (o3v.as_slice().unwrap(), o4v.as_slice().unwrap());

    sc.ind0_row.clear();
    sc.ind0_row.extend(
        ind1_r
            .iter()
            .map(|&v| if v.is_nan() { 0.0 } else { 1.0 }),
    );

    sc.pct.clear();
    sc.pct.resize(n, f64::NAN);
    sc.keys.clear();
    sc.keys.resize(n, 0);
    rank_pct_row_from_f32_in(row, &mut sc.pct, &mut sc.idxs, &mut sc.tmp, &mut sc.keys);
    fill_ind_reg_row(
        &mut sc.pct,
        [ind2_r, ind1_r, &sc.ind0_row],
        size_r,
        [o0, o1, o2],
        &mut sc.ys,
        &mut sc.bs,
        &mut sc.obs,
        &mut sc.seen,
    );
    for (j, &v) in ind1_r.iter().enumerate() {
        if v.is_nan() {
            sc.pct[j] = f64::NAN;
        }
    }
    sc.filled.clear();
    sc.filled.extend_from_slice(&sc.pct);
    for pass in 0..3 {
        sc.nan_mask.clear();
        let mut any = false;
        for &v in sc.filled.iter() {
            let nn = v.is_nan();
            sc.nan_mask.push(nn);
            any |= nn;
        }
        if !any {
            break;
        }
        match pass {
            0 => median_fill_level_row(&mut sc.filled, ind2_r, None, o3, &sc.nan_mask, &mut sc.sv),
            1 => median_fill_level_row(&mut sc.filled, ind1_r, None, o4, &sc.nan_mask, &mut sc.sv),
            _ => median_fill_level_row(
                &mut sc.filled,
                zeros_r,
                Some(mask_r),
                o0,
                &sc.nan_mask,
                &mut sc.sv,
            ),
        }
    }
    for (j, &v) in restrict_r.iter().enumerate() {
        if v != 0.0 {
            sc.filled[j] = f64::NAN;
        }
    }
    rank_pct_row_f64_in_place(
        &mut sc.filled,
        &mut sc.sv,
        &mut sc.idxs,
        &mut sc.tmp,
        &mut sc.keys64,
        &mut sc.rank_vals,
    );
}

/// 生产行级回退（行业路径）：填充链 → 行业 OLS。
fn v3_row_prod(row: &[f32], idx: usize, shared: &V3Shared, out_row: &mut [f32], sc: &mut V3Scratch) {
    v3_row_prod_fill(row, idx, shared, sc);
    ols_day_row_fast(
        &sc.filled,
        &shared.ns,
        idx,
        &mut sc.y_buf_fb,
        &mut sc.xty,
        out_row,
    );
}

/// 生产行级回退（纯风格路径）：同一填充链 → `get_residual(.., None)` 的行级移植
/// （`factor_neutralize_std::get_residual_row_style`，task-10 产物，逐位验证过）。
fn v3_row_prod_style(
    row: &[f32],
    idx: usize,
    shared: &V3Shared,
    out_row: &mut [f32],
    sc: &mut V3Scratch,
) {
    v3_row_prod_fill(row, idx, shared, sc);
    style_residual_row(&sc.filled, &shared.ns, idx, out_row, &mut sc.style_resid);
}

/// 纯风格残差的生产行级实现：把整行交给 `get_residual_row_style`，再转 f32 写回。
/// 10 条风格行用栈上数组承载（无堆分配）。
fn style_residual_row(
    fv_row: &[f64],
    ns: &NeutralizeStdShared,
    idx: usize,
    out_row: &mut [f32],
    resid: &mut Vec<f64>,
) {
    let n = fv_row.len();
    // 10 条风格行视图用栈上数组承载（无堆分配）
    let views: [ndarray::ArrayView1<'_, f64>; 10] =
        std::array::from_fn(|c| ns.barra_ranked[c].row(idx));
    let mut bench_rows: [&[f64]; 10] = [&[]; 10];
    for c in 0..10 {
        bench_rows[c] = views[c].as_slice().unwrap();
    }
    resid.clear();
    resid.resize(n, f64::NAN);
    crate::factor_neutralize_std::get_residual_row_style(fv_row, &bench_rows, resid);
    for (j, &v) in resid.iter().enumerate() {
        out_row[j] = if v.is_nan() { f32::NAN } else { v as f32 };
    }
}

/// 含"特殊位置"行的精确路径。
///
/// 特殊集 X = {j ∈ S : slot[j] 为 NaN，或 size[j] 为 NaN}。
/// 对 j ∈ S\X（slot 有限且 size 有限）生产链路恒等于：pct1[j]（fill_ind_reg 不写它，
/// 中位填充也不写它），且其大小序 = slot 值序 → 直接由 P 排序给出。
/// 对 x ∈ X，生产链路为：
///   fill_ind_reg: obs=false（row 或 size 缺失）→ 写 c0+c1*size（size 缺失则写 NaN）
///                其中 (c0,c1) 是 x 所属 ind2 段在**该级**的 ols2（ys = 段内 row、size 均有限者）
///   若写后为 NaN（size 缺失）→ ind2 级中位填充写入"该 ind2 段 filled 值的中位数"
/// 这里把段级 (c0,c1) 与 sv 多重集精确复算（ys/bs 按 o0c 顺序累计 → ols2 逐位一致；
/// sv 多重集一致 → quickselect 结果一致），再与 S\X 的 pct1 序归并，
/// 得到与生产 rank2 完全相同的平均秩。任何前提不满足 → 返回 `M_PROD` 交回生产行级实现。
///
/// 末步不再就地求解，而是把 V 序 y 写进 `b_y[f]`（`M_READY`），由 `v3_day_batch` 与同日
/// 其它面合并成一次 (p × b) Cholesky solve。
fn v3_row_x_prep(
    row: &[f32],
    idx: usize,
    shared: &V3Shared,
    sc: &mut V3Scratch,
    t: &mut V3Times,
    f: usize,
) -> u8 {
    let n = row.len();
    let ns = shared.s_lens[idx] as usize;
    let ns_f = ns as f64;
    let s_idx = &shared.s_idx_flat[idx * n..idx * n + ns];
    let size_row = shared.ns.size_ranked.row(idx);
    let size_row = size_row.as_slice().unwrap();

    // ---- ① 排序 P（rank1 域） ----
    let st = Instant::now();
    sc.pv.clear();
    sc.pi.clear();
    for (j, &val) in row.iter().enumerate() {
        if val.is_finite() {
            sc.pi.push(j as u32);
            sc.pv.push(val);
        }
    }
    let np = sc.pv.len();
    t.gather += st.elapsed().as_secs_f64();
    if np == 0 {
        return M_DONE;
    }
    let st = Instant::now();
    sc.keys.clear();
    sc.keys.extend(sc.pv.iter().map(|&val| mono_key32(val)));
    sc.order.clear();
    sc.order.extend(0..np);
    radix_sort_order32(&sc.keys, &mut sc.order, &mut sc.tmp);
    t.sort += st.elapsed().as_secs_f64();

    // ---- ② 组游走：写 pct_full（全行 f64），收集 S\\X 的升序序列 ----
    let st = Instant::now();
    for &j in sc.touched.iter() {
        sc.pct_full[j as usize] = f64::NAN;
    }
    sc.touched.clear();
    sc.v_val.clear();
    sc.v_spos.clear();
    let np_f = np as f64;
    let s_pos = &shared.s_pos_flat[idx * n..idx * n + n];
    let mut k = 0usize;
    while k < np {
        let val = sc.pv[sc.order[k]];
        let mut e = k + 1;
        while e < np && sc.pv[sc.order[e]] == val {
            e += 1;
        }
        let pct = (((k + 1) + e) as f64 / 2.0) / np_f;
        for t2 in k..e {
            let j = sc.pi[sc.order[t2]] as usize;
            sc.pct_full[j] = pct;
            sc.touched.push(j as u32);
            if shared.in_s_flat[idx * n + j] != 0 && size_row[j].is_finite() {
                sc.v_val.push(pct);
                sc.v_spos.push(s_pos[j]);
            }
        }
        k = e;
    }
    t.walk += st.elapsed().as_secs_f64();

    // ---- ③ 特殊集 X = S \\ W（W = 上面收集到的 pct1 可用的位置） ----
    let st = Instant::now();
    sc.x_list.clear();
    sc.x_spos.clear();
    let nx_expected = ns - sc.v_val.len();
    if nx_expected == shared.d_lens[idx] as usize {
        let d0 = shared.d_offsets[idx] as usize;
        let nd = shared.d_lens[idx] as usize;
        sc.x_list.extend_from_slice(&shared.d_flat[d0..d0 + nd]);
        sc.x_spos
            .extend_from_slice(&shared.d_s_pos_flat[d0..d0 + nd]);
    } else {
        for kk in 0..ns {
            let j = s_idx[kk] as usize;
            if !row[j].is_finite() || !size_row[j].is_finite() {
                sc.x_list.push(j as u32);
                sc.x_spos.push(kk as u32);
            }
        }
    }
    let nx = sc.x_list.len();
    sc.x_val.clear();
    if nx > 0 {
        let pct_row = &sc.pct_full[0..n];
        if !has_ge_n_unique(pct_row, 10, &mut sc.seen) {
            t.fallback += st.elapsed().as_secs_f64();
            return M_PROD;
        }
        let o0c0 = shared.o0c_offsets[idx] as usize;
        let o0c = &shared.o0c_flat[o0c0..o0c0 + shared.o0c_lens[idx] as usize];
        let seg0 = shared.seg_offsets[idx] as usize;
        let nseg = shared.seg_lens[idx] as usize;
        let segs = &shared.seg_flat[seg0..seg0 + nseg];
        let mut last_a = usize::MAX;
        let mut last_c0 = 0.0f64;
        let mut last_c1 = 0.0f64;
        let mut last_med = f64::NAN;
        let mut last_ok = false;
        for xi in 0..nx {
            let j = sc.x_list[xi] as usize;
            let p = shared.pos0_flat[idx * n + j];
            if p == u32::MAX {
                t.fallback += st.elapsed().as_secs_f64();
                return M_PROD;
            }
            let p = p as usize;
            let si = match segs.binary_search(&(p as u32)) {
                Ok(x) => x,
                Err(x) => x - 1,
            };
            let a = segs[si] as usize;
            let b = if si + 1 < nseg {
                segs[si + 1] as usize
            } else {
                o0c.len()
            };
            if a != last_a {
                last_a = a;
                sc.ys.clear();
                sc.bs.clear();
                sc.obs.clear();
                for kk in a..b {
                    let i = o0c[kk] as usize;
                    let rv = sc.pct_full[i];
                    let sz = size_row[i];
                    let ok = !rv.is_nan() && !sz.is_nan();
                    sc.obs.push(ok);
                    if ok {
                        sc.ys.push(rv);
                        sc.bs.push(sz);
                    }
                }
                last_ok = sc.ys.len() >= 10;
                if last_ok {
                    let (c0, c1) = ols2(&sc.ys, &sc.bs);
                    last_c0 = c0;
                    last_c1 = c1;
                }
                sc.sv.clear();
                for kk in a..b {
                    let i = o0c[kk] as usize;
                    if sc.obs[kk - a] {
                        sc.sv.push(sc.pct_full[i]);
                    } else {
                        let sz = size_row[i];
                        if !sz.is_nan() {
                            if !last_ok {
                                t.fallback += st.elapsed().as_secs_f64();
                                return M_PROD;
                            }
                            sc.sv.push(last_c0 + last_c1 * sz);
                        }
                    }
                }
                last_med = if sc.sv.is_empty() {
                    f64::NAN
                } else {
                    median_inplace(&mut sc.sv)
                };
            }
            let sz = size_row[j];
            let val = if sz.is_finite() {
                if !last_ok {
                    t.fallback += st.elapsed().as_secs_f64();
                    return M_PROD;
                }
                last_c0 + last_c1 * sz
            } else {
                last_med
            };
            if val.is_nan() {
                t.fallback += st.elapsed().as_secs_f64();
                return M_PROD;
            }
            sc.x_val.push(val);
        }
    }
    t.walk += st.elapsed().as_secs_f64();

    // ---- ④ 归并 S\\X 序与 X 序 → rank2 的 rank pct（写 ybuf，S 顺序） ----
    let st = Instant::now();
    sc.ybuf.clear();
    sc.ybuf.resize(ns, f64::NAN);
    let nv = sc.v_val.len();
    if nx == 0 {
        let mut i = 0usize;
        while i < nv {
            let val = sc.v_val[i];
            let mut e = i + 1;
            while e < nv && sc.v_val[e] == val {
                e += 1;
            }
            let pct = (((i + 1) + e) as f64 / 2.0) / ns_f;
            for k2 in i..e {
                sc.ybuf[sc.v_spos[k2] as usize] = pct;
            }
            i = e;
        }
    } else {
        for a in 1..nx {
            let mut b = a;
            while b > 0 && sc.x_val[b - 1] > sc.x_val[b] {
                sc.x_val.swap(b - 1, b);
                sc.x_spos.swap(b - 1, b);
                b -= 1;
            }
        }
        let (mut i, mut kk, mut acc) = (0usize, 0usize, 0usize);
        while i < nv || kk < nx {
            let take_v = if kk >= nx {
                true
            } else if i >= nv {
                false
            } else {
                sc.v_val[i] <= sc.x_val[kk]
            };
            let val = if take_v { sc.v_val[i] } else { sc.x_val[kk] };
            sc.group.clear();
            while i < nv && sc.v_val[i] == val {
                sc.group.push(sc.v_spos[i]);
                i += 1;
            }
            while kk < nx && sc.x_val[kk] == val {
                sc.group.push(sc.x_spos[kk]);
                kk += 1;
            }
            let start = acc;
            acc += sc.group.len();
            let pct = (((start + 1) + acc) as f64 / 2.0) / ns_f;
            for &p in sc.group.iter() {
                sc.ybuf[p as usize] = pct;
            }
        }
        if acc != ns {
            return M_PROD;
        }
    }
    t.walk += st.elapsed().as_secs_f64();

    // ---- ⑤ 把 y 排到 V 顺序并存入批缓冲（末步求解由 v3_day_batch 统一做） ----
    let st = Instant::now();
    let o2 = shared.v_offsets[idx] as usize;
    let nvv = shared.v_lens[idx] as usize;
    let v_from_s = &shared.v_from_s_flat[o2..o2 + nvv];
    {
        let off = f * sc.b_stride;
        let dst = &mut sc.b_y[off..off + nvv];
        for (q, &p) in v_from_s.iter().enumerate() {
            dst[q] = sc.ybuf[p as usize];
        }
    }
    sc.b_n[f] = nvv;
    t.ols += st.elapsed().as_secs_f64();
    M_READY
}

/// 压缩快路径的"准备段"：gather S → 4 趟 u32 基数排序 → 组游走出 rank pct → 排到 V 顺序存入
/// `b_y[f]`。末步求解与写回不在这里做（由 `v3_day_batch` 与同日其它面合并成一次 solve）。
///
/// 返回值：`M_DONE`（S 为空，末步无需处理）、`M_PROD`（S 中有洞，交 `v3_row_x_prep`/生产回退）、
/// `M_READY`（y 已就绪）。
#[inline]
fn v3_row_fast_prep(
    row: &[f32],
    idx: usize,
    shared: &V3Shared,
    sc: &mut V3Scratch,
    t: &mut V3Times,
    f: usize,
) -> u8 {
    let n = row.len();
    let ns = shared.s_lens[idx] as usize;
    let s_idx = &shared.s_idx_flat[idx * n..idx * n + ns];

    // ---- ① gather S + 洞检测 ----
    let s = Instant::now();
    sc.yv.clear();
    for &j in s_idx {
        let v = row[j as usize];
        if !v.is_finite() {
            t.gather += s.elapsed().as_secs_f64();
            return M_PROD;
        }
        sc.yv.push(v);
    }
    t.gather += s.elapsed().as_secs_f64();
    if ns == 0 {
        return M_DONE;
    }

    // ---- ② 4 趟 u32 基数排序（|S| 而非 7857；全域等数字的趟自动跳过） ----
    let s = Instant::now();
    sc.keys.clear();
    sc.keys.extend(sc.yv.iter().map(|&v| mono_key32(v)));
    sc.order.clear();
    sc.order.extend(0..ns);
    radix_sort_order32(&sc.keys, &mut sc.order, &mut sc.tmp);
    t.sort += s.elapsed().as_secs_f64();

    // ---- ③ 组游走：直接产出 rank2 的 rank pct（无需第二次排序） ----
    let s = Instant::now();
    let ns_f = ns as f64;
    sc.ybuf.clear();
    sc.ybuf.resize(ns, f64::NAN);
    let mut k = 0usize;
    while k < ns {
        let v = sc.yv[sc.order[k]];
        let mut e = k + 1;
        while e < ns && sc.yv[sc.order[e]] == v {
            e += 1;
        }
        // 与生产 rank_pct_row_f64_in_place 同式：avg_rank = ((i+1)+(j+1))/2, pct = avg/n
        let avg_rank = ((k + 1) + e) as f64 / 2.0;
        let pct = avg_rank / ns_f;
        for t2 in k..e {
            sc.ybuf[sc.order[t2]] = pct;
        }
        k = e;
    }
    t.walk += s.elapsed().as_secs_f64();

    // ---- ④ 把 y 排到 V 顺序存入批缓冲（S==V 时直接整段拷贝） ----
    let s = Instant::now();
    let off = f * sc.b_stride;
    if shared.identity[idx] {
        sc.b_y[off..off + ns].copy_from_slice(&sc.ybuf[..ns]);
        sc.b_n[f] = ns;
    } else {
        let o2 = shared.v_offsets[idx] as usize;
        let nv = shared.v_lens[idx] as usize;
        let v_from_s = &shared.v_from_s_flat[o2..o2 + nv];
        {
            let dst = &mut sc.b_y[off..off + nv];
            for (q, &p) in v_from_s.iter().enumerate() {
                dst[q] = sc.ybuf[p as usize];
            }
        }
        sc.b_n[f] = nv;
    }
    t.ols += s.elapsed().as_secs_f64();
    M_READY
}

// ==================== 对外接口 ====================

/// 单个「面 × 日」的批处理调度：压缩快路径 → 特殊位置路径 → 生产行级回退。
/// `slots[f]` 是面 f 的日期块视图，`r` 是块内行号（绝对日期 `idx`）；`outs[f]` 是面 f 的输出。
///
/// **按日批处理**：同一天的 b 个面共用同一套「只随日期变化」的索引（S/V/D、o0c 分段、每日
/// Cholesky 与连续 X）。每面各自跑排序/秩变换得到 V 序 y 后，把 b 个 `X'y` 拼成 (p × b) 的
/// RHS，一次 `Cholesky::solve_mut` 解出全部 b 列系数，再统一写回。
///
/// 逐位不变性：每面的 mn/mx、`X'y` 累加顺序、pred 累加顺序都与拆分前逐字相同；nalgebra 的
/// `solve_mut` 对多列 RHS 是 `for j in 0..ncols` 逐列调用同一个下三角求解，列与列之间无耦合
/// → 每列结果与单列求解逐位相同。`shared.chols[idx]` 的每日预分解本身完全不动。
fn v3_day_batch(
    slots: &[ArrayView2<f32>],
    outs: &mut [Array2<f32>],
    r: usize,
    idx: usize,
    shared: &V3Shared,
    sc: &mut V3Scratch,
    times: &mut V3Times,
    industry_neutralize: bool,
) {
    let b = slots.len();
    let n = shared.n_stocks;
    sc.ensure_batch(b, n);

    // ---- 非快路径日：b 个面全部走生产行级实现 ----
    if !shared.fast_ok[idx] {
        let s = Instant::now();
        for f in 0..b {
            let row_v = slots[f].row(r);
            let row = row_v.as_slice().unwrap();
            let mut out_v = outs[f].row_mut(r);
            let out_row = out_v.as_slice_mut().unwrap();
            if industry_neutralize {
                v3_row_prod(row, idx, shared, out_row, sc);
            } else {
                v3_row_prod_style(row, idx, shared, out_row, sc);
            }
        }
        times.fallback += s.elapsed().as_secs_f64();
        times.slow_rows += b as u64;
        return;
    }

    // ---- 阶段 1：各面独立跑压缩流水线，产出 V 序 y（写进 b_y[f]） ----
    for f in 0..b {
        sc.b_mode[f] = M_DONE;
        sc.b_n[f] = 0;
        let row_v = slots[f].row(r);
        let row = row_v.as_slice().unwrap();
        let st = if shared.d_lens[idx] == 0 {
            let a = v3_row_fast_prep(row, idx, shared, sc, times, f);
            if a == M_PROD {
                v3_row_x_prep(row, idx, shared, sc, times, f)
            } else {
                a
            }
        } else {
            v3_row_x_prep(row, idx, shared, sc, times, f)
        };
        sc.b_mode[f] = st;
    }

    // ---- 阶段 1b：末步判定（写 0.5 / 判定 SVD / 判定生产回退，不做 X'y 累加） ----
    let p_day = shared.ns.per_date[idx].0;
    let p_alloc = p_day.max(11);
    sc.ensure_xty(b, p_alloc);
    let s_ols = Instant::now();
    for f in 0..b {
        if sc.b_mode[f] != M_READY {
            continue;
        }
        let off = f * sc.b_stride;
        let len = sc.b_n[f];
        let mut out_v = outs[f].row_mut(r);
        let out_row = out_v.as_slice_mut().unwrap();
        let mode = if industry_neutralize {
            ols_check_industry(&sc.b_y[off..off + len], &shared.ns, idx, out_row)
        } else {
            ols_check_style(&sc.b_y[off..off + len], &shared.ns, idx, out_row)
        };
        sc.b_mode[f] = mode;
    }

    // ---- 阶段 1c：X'y 累加（位置外层 / 面内层：同日 b 个面共用一次 xd/valid_cols 遍历） ----
    let n_ready = (0..b).filter(|&f| sc.b_mode[f] == M_READY).count();
    if n_ready > 0 {
        ols_acc_day(sc, &shared.ns, idx, industry_neutralize, b, p_alloc);
    }

    // ---- 阶段 2：一次 (p × b) Cholesky solve（只对 M_READY 的面填列） ----
    if n_ready > 0 {
        let p_solve = if industry_neutralize { p_day } else { 11 };
        let use_svd_industry = industry_neutralize
            && (shared.ns.per_date[idx].1.len() <= 40 || shared.ns.chols[idx].is_none());
        let chol: Option<&Cholesky<f64, nalgebra::Dyn>> = if industry_neutralize {
            if use_svd_industry {
                None
            } else {
                shared.ns.chols[idx].as_ref()
            }
        } else {
            shared.ns.chols_style[idx].as_ref()
        };
        match chol {
            Some(ch) => {
                {
                    let mut rhs_v = sc.rhs.view_mut((0, 0), (p_solve, b));
                    for f in 0..b {
                        if sc.b_mode[f] != M_READY {
                            continue;
                        }
                        let base = f * p_alloc;
                        for i in 0..p_solve {
                            rhs_v[(i, f)] = sc.b_xty[base + i];
                        }
                    }
                    ch.solve_mut(&mut rhs_v);
                    for f in 0..b {
                        if sc.b_mode[f] != M_READY {
                            continue;
                        }
                        let base = f * p_alloc;
                        for i in 0..p_solve {
                            sc.b_coef[base + i] = rhs_v[(i, f)];
                        }
                    }
                }
                // ---- 阶段 3：统一写回残差 ----
                for f in 0..b {
                    if sc.b_mode[f] != M_READY {
                        continue;
                    }
                    let off = f * sc.b_stride;
                    let len = sc.b_n[f];
                    let base = f * p_alloc;
                    let mut out_v = outs[f].row_mut(r);
                    let out_row = out_v.as_slice_mut().unwrap();
                    if industry_neutralize {
                        ols_write_industry(
                            &sc.b_y[off..off + len],
                            &shared.ns,
                            idx,
                            out_row,
                            &sc.b_coef[base..base + p_day],
                        );
                    } else {
                        ols_write_style(
                            &sc.b_y[off..off + len],
                            &shared.ns,
                            idx,
                            out_row,
                            &sc.b_coef[base..base + 11],
                        );
                    }
                }
            }
            None => {
                let nm = if industry_neutralize { M_SVD } else { M_STYLE_FB };
                for f in 0..b {
                    if sc.b_mode[f] == M_READY {
                        sc.b_mode[f] = nm;
                    }
                }
            }
        }
    }
    times.ols += s_ols.elapsed().as_secs_f64();

    // ---- 阶段 4：SVD（罕见）/ 生产行级回退 ----
    for f in 0..b {
        match sc.b_mode[f] {
            M_SVD => {
                let off = f * sc.b_stride;
                let len = sc.b_n[f];
                let mut out_v = outs[f].row_mut(r);
                let out_row = out_v.as_slice_mut().unwrap();
                let s = Instant::now();
                ols_day_core(&sc.b_y[off..off + len], &shared.ns, idx, out_row, &mut sc.xty);
                times.ols += s.elapsed().as_secs_f64();
                times.fast_rows += 1;
            }
            M_STYLE_FB | M_PROD => {
                let row_v = slots[f].row(r);
                let row = row_v.as_slice().unwrap();
                let mut out_v = outs[f].row_mut(r);
                let out_row = out_v.as_slice_mut().unwrap();
                let s = Instant::now();
                if industry_neutralize {
                    v3_row_prod(row, idx, shared, out_row, sc);
                } else {
                    v3_row_prod_style(row, idx, shared, out_row, sc);
                }
                times.fallback += s.elapsed().as_secs_f64();
                times.slow_rows += 1;
            }
            _ => {
                times.fast_rows += 1;
            }
        }
    }
}

/// 单面逐行中性化：把 `slot_block` 的第 r 行（= 绝对日期 `t0 + r`）算进 `out` 第 r 行。
fn v3_rows(
    slot_block: &ArrayView2<'_, f32>,
    shared: &V3Shared,
    t0: usize,
    t1: usize,
    sc: &mut V3Scratch,
    out: &mut Array2<f32>,
    times: &mut V3Times,
    industry_neutralize: bool,
) {
    let rows = t1 - t0;
    let blk = std::slice::from_ref(slot_block);
    let ob = std::slice::from_mut(out);
    for r in 0..rows {
        v3_day_batch(blk, ob, r, t0 + r, shared, sc, times, industry_neutralize);
    }
}

/// 整张版（对账 / 回落用）：输入 (T,N) f32 → 输出 (T,N) f32。
///
/// `industry_neutralize=true` 与生产 `neutralize_std_slots_f32_v2_resid_batch(slots, ns, true)`
/// 逐位一致；`false` 与 `(slots, ns, false)` 逐位一致（残差走 11 列 `[1, b0..b9]` 风格路径）。
pub fn v3_slot(
    slot: ArrayView2<'_, f32>,
    shared: &V3Shared,
    industry_neutralize: bool,
    sc: &mut V3Scratch,
) -> Result<Array2<f32>, String> {
    Ok(v3_slot_timed(slot, shared, industry_neutralize, sc)?.0)
}

/// 同 `v3_slot`，额外返回分段计时与快/慢路径行数（诊断用）。
pub fn v3_slot_timed(
    slot: ArrayView2<'_, f32>,
    shared: &V3Shared,
    industry_neutralize: bool,
    sc: &mut V3Scratch,
) -> Result<(Array2<f32>, V3Times), String> {
    let (t, n) = slot.dim();
    if n != shared.n_stocks || shared.ns.industry.dim() != (t, n) {
        return Err("v3_slot: slot 与 NeutralizeStdShared 形状不匹配".to_string());
    }
    let mut out = Array2::<f32>::from_elem((t, n), f32::NAN);
    let mut times = V3Times::default();
    v3_rows(&slot, shared, 0, t, sc, &mut out, &mut times, industry_neutralize);
    Ok((out, times))
}

/// 按日期块中性化（单面）：`slot_block` 是第 [t0,t1) 行的块视图（行数 = t1-t0），
/// 输出 (t1-t0, n)；`shared` 侧按绝对日期 idx 索引。与整张版逐位一致。
pub fn v3_slot_range(
    slot_block: ArrayView2<'_, f32>,
    shared: &V3Shared,
    industry_neutralize: bool,
    t0: usize,
    t1: usize,
    sc: &mut V3Scratch,
) -> Result<Array2<f32>, String> {
    let (rows, n) = slot_block.dim();
    let (t, sn) = shared.ns.industry.dim();
    if n != shared.n_stocks || sn != n {
        return Err("v3_slot_range: slot 与 NeutralizeStdShared 形状不匹配".to_string());
    }
    if t0 > t1 || t1 > t || t1 - t0 != rows {
        return Err(format!(
            "v3_slot_range 越界: t0={t0} t1={t1} t={t} rows={rows}"
        ));
    }
    let mut out = Array2::<f32>::from_elem((rows, n), f32::NAN);
    let mut times = V3Times::default();
    v3_rows(
        &slot_block,
        shared,
        t0,
        t1,
        sc,
        &mut out,
        &mut times,
        industry_neutralize,
    );
    Ok(out)
}

/// v8 融合流水线入口：B 个面一起按日期块中性化（每块一次调用，scratch 跨块复用）。
///
/// **日外层 / 面批处理**：同一天的 B 个面共用一次「只随日期变化」的索引（S/V/D、o0c 分段、
/// 每日 Cholesky 与连续 X），B 个 `X'y` 组成 (p × B) 的 RHS 一次解出，再一次性写回 B 行。
/// 数值路径与单面版完全一致（逐位）。
/// `industry_neutralize` 只影响残差末步（行业 one-hot vs 显式截距 + 10 风格），
/// 索引 / 压缩 / fast_ok / 填充链两边完全共用。
pub fn v3_slots_range(
    slots: &[ArrayView2<f32>],
    shared: &V3Shared,
    industry_neutralize: bool,
    t0: usize,
    t1: usize,
    sc: &mut V3Scratch,
) -> Result<Vec<Array2<f32>>, String> {
    let b = slots.len();
    if b == 0 {
        return Ok(Vec::new());
    }
    let (rows, n) = slots[0].dim();
    let (t, sn) = shared.ns.industry.dim();
    if sn != n || n != shared.n_stocks {
        return Err("v3_slots_range: slot 与 NeutralizeStdShared 形状不匹配".to_string());
    }
    if t0 > t1 || t1 > t || t1 - t0 != rows {
        return Err(format!(
            "v3_slots_range 越界: t0={t0} t1={t1} t={t} rows={rows}"
        ));
    }
    for s in slots {
        if s.dim() != (rows, n) {
            return Err("v3_slots_range: 各 slot 形状不一致".to_string());
        }
    }
    let mut outs: Vec<Array2<f32>> = (0..b)
        .map(|_| Array2::<f32>::from_elem((rows, n), f32::NAN))
        .collect();
    let mut times = V3Times::default();
    for r in 0..rows {
        let idx = t0 + r;
        v3_day_batch(slots, &mut outs, r, idx, shared, sc, &mut times, industry_neutralize);
    }
    Ok(outs)
}

/// 逐位比较（NaN 只要求位置一致；其余比 bit 模式）。
pub fn bitwise_equal(a: &Array2<f32>, b: &Array2<f32>) -> (bool, usize) {
    let mut mismatch = 0usize;
    for (x, y) in a.iter().zip(b.iter()) {
        if x.to_bits() != y.to_bits() && !(x.is_nan() && y.is_nan()) {
            mismatch += 1;
        }
    }
    (mismatch == 0, mismatch)
}


// ==================== 自测：与生产 v2 逐位对账 + 计时 ====================

/// 与生产 `neutralize_std_slots_f32_v2_resid_batch` 逐位比较：
/// 2 因子 × 13 面（smooth_1 + w=5/10/20 × {mean,max,min,std}）× 块 64/997/2818，
/// 并给出「生产 v2 整张 batch」vs「v3 按块」的单线程计时（13 面）。
pub fn selfcheck(data_dir: &str) -> String {
    use ndarray::Array1;
    use ndarray_npy::read_npy;

    let dates: Vec<i32> = match read_npy::<_, Array1<i32>>(format!("{data_dir}/dates.npy")) {
        Ok(v) => v.to_vec(),
        Err(e) => return format!("[neu3] 读 dates 失败: {e}"),
    };
    let stocks: Vec<String> = match std::fs::read_to_string(format!("{data_dir}/stocks.txt")) {
        Ok(t) => t
            .lines()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect(),
        Err(e) => return format!("[neu3] 读 stocks.txt 失败: {e}"),
    };
    let restrict: Array2<f32> = match read_npy(format!("{data_dir}/restrict.npy")) {
        Ok(v) => v,
        Err(e) => return format!("[neu3] 读 restrict 失败: {e}"),
    };
    let industry: Array2<f64> = match read_npy(format!("{data_dir}/industry.npy")) {
        Ok(v) => v,
        Err(e) => return format!("[neu3] 读 industry 失败: {e}"),
    };
    let style_vars_dir = std::env::var("V8_STYLE_PATH").unwrap_or_else(|_| {
        "/ssd_data/data/vars".to_string()
    });
    let style = match crate::factor_neutralization_io_optimized::IOOptimizedStyleData::
        load_from_vars_h5(&style_vars_dir)
    {
        Ok(s) => s,
        Err(e) => return format!("[neu3] 加载风格数据失败: {e}"),
    };
    let ns = match crate::factor_neutralize_std::neutralize_std_precompute(
        &industry, &restrict, &style, &dates, &stocks,
    ) {
        Ok(s) => s,
        Err(e) => return format!("[neu3] 预计算失败: {e}"),
    };
    let ns = Arc::new(ns);
    let tmr = Instant::now();
    let shared = match V3Shared::build(ns.clone()) {
        Ok(s) => s,
        Err(e) => return format!("[neu3] V3Shared::build 失败: {e}"),
    };
    let t_build = tmr.elapsed().as_secs_f64();

    let names: Vec<String> = std::fs::read_to_string(format!("{data_dir}/sample_names.txt"))
        .unwrap_or_default()
        .split_whitespace()
        .map(|s| s.to_string())
        .collect();
    if names.is_empty() {
        return "[neu3] sample_names.txt 为空".to_string();
    }
    let (t, n) = restrict.dim();
    let mut lines = vec![format!(
        "[neu3] V3Shared::build={t_build:.2}s T={t} N={n} 非快路径日={} S==V 日={} |V|<=10 日={} V⊄S 日={}",
        shared.fast_ok.iter().filter(|x| !**x).count(),
        shared.identity_days,
        shared.small_days,
        shared.v_notin_s_days,
    )];
    let mut all_pass = true;
    let mut n_cmp = 0usize;
    let mut mm_total = 0usize;
    let mut sc = V3Scratch::new(n);
    // 计时（仅第一个因子，单线程）：[0]=行业 [1]=纯风格
    let mut t_v2 = [0.0f64; 2];
    let mut t_v3 = [[0.0f64; 3]; 2];
    let mut t_v3_style_full = 0.0f64;
    let mut t_v2_range_style = 0.0f64;
    let blocks = [64usize, 997, 2818];
    let flags = [true, false];

    for (fi, nm) in names.iter().take(2).enumerate() {
        let raw: Array2<f32> = match read_npy(format!("{data_dir}/factor_{nm}.npy")) {
            Ok(v) => v,
            Err(e) => return format!("[neu3] 读 factor_{nm} 失败: {e}"),
        };
        let ranked =
            crate::tail_v5_pipeline::rank_and_fill_missing_cross_sectional_median(&raw, &restrict);
        // 13 面：smooth_1 + w=5/10/20 × {mean,max,min,std}
        let mut slots: Vec<Array2<f32>> = vec![ranked.clone()];
        for &w in &[5usize, 10, 20] {
            let (m, x, mn, sd) =
                crate::tail_v2_rank_roll_factor::rolling_stats_f32_serial(&ranked, w, w / 2);
            slots.push(m);
            slots.push(x);
            slots.push(mn);
            slots.push(sd);
        }
        let ns_views: Vec<ArrayView2<f32>> = slots.iter().map(|s| s.view()).collect();

        for (fi_flag, &flag) in flags.iter().enumerate() {
            let fname = if flag { "行业" } else { "纯风格" };
            let tmr = Instant::now();
            let v2_full =
                match crate::factor_neutralize_std::neutralize_std_slots_f32_v2_resid_batch(
                    &ns_views, &ns, flag,
                ) {
                    Ok(v) => v,
                    Err(e) => return format!("[neu3] 生产 v2 batch({fname}) 失败: {e}"),
                };
            if fi == 0 {
                t_v2[fi_flag] = tmr.elapsed().as_secs_f64();
            }

            for (bi, &bs) in blocks.iter().enumerate() {
                let mut cat: Vec<Array2<f32>> = (0..slots.len())
                    .map(|_| Array2::<f32>::from_elem((t, n), f32::NAN))
                    .collect();
                let tmr = Instant::now();
                let mut a = 0usize;
                while a < t {
                    let b1 = (a + bs).min(t);
                    let blk: Vec<ArrayView2<f32>> =
                        slots.iter().map(|s| s.slice(ndarray::s![a..b1, ..])).collect();
                    let outs = match v3_slots_range(&blk, &shared, flag, a, b1, &mut sc) {
                        Ok(v) => v,
                        Err(e) => return format!("[neu3] v3_slots_range({fname}) 失败: {e}"),
                    };
                    for k in 0..slots.len() {
                        cat[k].slice_mut(ndarray::s![a..b1, ..]).assign(&outs[k]);
                    }
                    a = b1;
                }
                if fi == 0 {
                    t_v3[fi_flag][bi] = tmr.elapsed().as_secs_f64();
                }
                let mut mm = 0usize;
                for k in 0..slots.len() {
                    mm += bitwise_equal(&v2_full[k], &cat[k]).1;
                }
                mm_total += mm;
                if mm > 0 {
                    all_pass = false;
                }
                n_cmp += 1;
                lines.push(format!("  {nm} 13 面 [{fname}] block={bs}: 不一致 {mm} 格"));
            }

            // 整张版 v3_slot 对账 + 快/慢路径统计
            let mut mm_full = 0usize;
            let mut fast_rows = 0u64;
            let mut slow_rows = 0u64;
            let tmr = Instant::now();
            for k in 0..slots.len() {
                let (out, tt) = match v3_slot_timed(slots[k].view(), &shared, flag, &mut sc) {
                    Ok(v) => v,
                    Err(e) => return format!("[neu3] v3_slot({fname}) 失败: {e}"),
                };
                mm_full += bitwise_equal(&v2_full[k], &out).1;
                fast_rows += tt.fast_rows;
                slow_rows += tt.slow_rows;
            }
            let t_full_v3 = tmr.elapsed().as_secs_f64();
            if fi == 0 && !flag {
                t_v3_style_full = t_full_v3;
            }
            mm_total += mm_full;
            if mm_full > 0 {
                all_pass = false;
            }
            n_cmp += 1;
            lines.push(format!(
                "  {nm} 13 面 [{fname}] 整张 v3_slot: 不一致 {mm_full} 格 ({t_full_v3:.3}s, 快路径行={fast_rows} 慢路径行={slow_rows})"
            ));

            // 纯风格：与 roll-dev 的行级 range 对照（对账 + 计时，bs=64）
            if !flag && fi == 0 {
                let bs = blocks[0];
                let mut cat: Vec<Array2<f32>> = (0..slots.len())
                    .map(|_| Array2::<f32>::from_elem((t, n), f32::NAN))
                    .collect();
                let tmr = Instant::now();
                let mut a = 0usize;
                while a < t {
                    let b1 = (a + bs).min(t);
                    let blk: Vec<ArrayView2<f32>> =
                        slots.iter().map(|s| s.slice(ndarray::s![a..b1, ..])).collect();
                    let outs = match crate::factor_neutralize_std::
                        neutralize_std_slots_f32_v2_resid_batch_range(&blk, &ns, false, a, b1)
                    {
                        Ok(v) => v,
                        Err(e) => return format!("[neu3] v2 range(false) 失败: {e}"),
                    };
                    for k in 0..slots.len() {
                        cat[k].slice_mut(ndarray::s![a..b1, ..]).assign(&outs[k]);
                    }
                    a = b1;
                }
                t_v2_range_style = tmr.elapsed().as_secs_f64();
                let mut mm = 0usize;
                for k in 0..slots.len() {
                    mm += bitwise_equal(&v2_full[k], &cat[k]).1;
                }
                mm_total += mm;
                if mm > 0 {
                    all_pass = false;
                }
                n_cmp += 1;
                lines.push(format!(
                    "  {nm} 13 面 [纯风格] v2 行级 range bs={bs}: 不一致 {mm} 格 ({t_v2_range_style:.3}s)"
                ));
            }
        }
    }
    lines.insert(
        0,
        format!(
            "[neu3] 逐位一致: {}  ({n_cmp} 组比较：2 因子 × 13 面 × 3 块 × [行业/纯风格] + 整张 + v2 range，不一致合计 {mm_total} 格)",
            if all_pass { "PASS" } else { "FAIL" }
        ),
    );
    lines.push(format!("  [计时] 13 面 生产 v2 整张 batch [行业]: {:.3}s", t_v2[0]));
    lines.push(format!(
        "  [计时] 13 面 生产 v2 整张 batch [纯风格]: {:.3}s",
        t_v2[1]
    ));
    for (bi, &bs) in blocks.iter().enumerate() {
        lines.push(format!(
            "  [计时] 13 面 v3 按块 [行业] bs={bs}: {:.3}s  (加速 {:.2}x)",
            t_v3[0][bi],
            t_v2[0] / t_v3[0][bi]
        ));
        lines.push(format!(
            "  [计时] 13 面 v3 按块 [纯风格] bs={bs}: {:.3}s  (加速 {:.2}x)",
            t_v3[1][bi],
            t_v2[1] / t_v3[1][bi]
        ));
    }
    lines.push(format!(
        "  [计时] 13 面 v3 整张 [纯风格]: {t_v3_style_full:.3}s"
    ));
    lines.push(format!(
        "  [计时] 13 面 v2 行级 range [纯风格] bs=64: {t_v2_range_style:.3}s  → v3 按块/它 = {:.2}x",
        t_v2_range_style / t_v3[1][0]
    ));
    lines.join("\n")
}

