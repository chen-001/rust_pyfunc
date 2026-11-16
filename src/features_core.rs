//! 纯 `std` 高性能序列统计特征与批量相关系数公共底座（零 `pyo3` / `ndarray` / `rayon` 依赖）。
//!
//! 设计目标：
//! 1. **正式库 `src/` 与独立探索沙箱 `sandbox_*/` 双端共用同一份源码**：
//!    - 正式库内通过 `crate::features` 或 `crate::features_core` 调用；
//!    - 独立 `sandbox_xxx` crate（无 `pyo3`/`ndarray` 依赖）直接在 `src/main.rs` 顶部挂载：
//!      `#[path = "../../src/features_core.rs"] mod features_core;`
//!      无需拷贝文件，杜绝沙箱手写慢速重复排序或逐对相关双层循环，升级正式库时只需改为 `use crate::features_core;`。
//! 2. **单列统计零堆分配、单次扫描 + 单次排序/部分选择**：
//!    - 同一列的均值、标准差、偏度、峰度在单次过滤有效值后一遍算出中心矩；
//!    - 同一列的中位数、各分位数（p5/p10/p25/p75/p90/p95）与 LZ 离散化阈值共用同一次排序（`n >= 64` 走 4 趟字节基数排序）；
//!    - 单次分位数查询走 `select_nth_unstable_by` 线性时间部分选择，不做全量排序。
//! 3. **多序列两两相关系数 `Z^T Z` 块矩阵加速（零临时分配，逐位精确）**：
//!    - 自动识别无缺失列与含 `NaN` 列；无缺失列统一去均值写入按 16 列对齐的行主序矩阵 `Z`，
//!      用 4×16 寄存器分块外积（沿列方向 16 路 SIMD 并行、沿时间行顺序累加）一次性算出上三角协方差，
//!      既跑满 AVX-512 向量吞吐，又保持与逐对顺序累加 **逐位一致（bit-exact）**；
//!    - 含 `NaN` 列自动回退零分配双遍配对扫描。

#![allow(dead_code)]

use std::cell::RefCell;
use std::cmp::Ordering;
use std::convert::TryInto;

const RETAINED_SCRATCH_FLOATS: usize = 262_144; // 1 MB

// ============================================================================
// 后缀自动机（LZ 复杂度专用，3 符号定长转移表，状态缓冲可跨列复用）
// ============================================================================

#[derive(Clone)]
pub(crate) struct SamState {
    len: usize,
    link: Option<usize>,
    transitions: [usize; 4],
}

impl SamState {
    #[inline]
    fn new(len: usize) -> Self {
        Self {
            len,
            link: None,
            transitions: [usize::MAX; 4],
        }
    }
    #[inline]
    fn get(&self, c: u8) -> Option<usize> {
        let t = self.transitions[c as usize];
        if t == usize::MAX {
            None
        } else {
            Some(t)
        }
    }
    #[inline]
    fn set(&mut self, c: u8, state: usize) {
        self.transitions[c as usize] = state;
    }
}

fn lz_complexity_simple(seq: &[u8]) -> usize {
    let n = seq.len();
    if n == 0 {
        return 0;
    }
    let mut complexity = 0;
    let mut i = 0;
    while i < n {
        let mut j = i + 1;
        while j <= n {
            let sub_len = j - i;
            let search_end = j - 1;
            if search_end < sub_len {
                break;
            }
            let mut found = false;
            for start_pos in 0..=(search_end - sub_len) {
                if seq[start_pos..start_pos + sub_len] == seq[i..j] {
                    found = true;
                    break;
                }
            }
            if found && j < n {
                j += 1;
            } else {
                break;
            }
        }
        complexity += 1;
        i = j;
    }
    complexity
}

pub(crate) fn lz_complexity_reused(seq: &[u8], states: &mut Vec<SamState>) -> usize {
    let n = seq.len();
    if n == 0 {
        return 0;
    }
    if n <= 64 {
        return lz_complexity_simple(seq);
    }
    states.clear();
    states.reserve(2 * n);
    states.push(SamState::new(0));
    let mut last = 0usize;
    let mut complexity = 0usize;
    let mut i = 0usize;
    while i < n {
        let mut state = 0usize;
        let mut j = i;
        while j < n {
            if let Some(next_state) = states[state].get(seq[j]) {
                state = next_state;
                j += 1;
            } else {
                break;
            }
        }
        if j == n {
            complexity += 1;
            break;
        }
        complexity += 1;
        let phrase_end = j + 1;
        for &c in &seq[i..phrase_end] {
            let cur_index = states.len();
            let cur_len = states[last].len + 1;
            states.push(SamState::new(cur_len));
            let mut p_opt = Some(last);
            while let Some(p_idx) = p_opt {
                if states[p_idx].get(c).is_some() {
                    break;
                }
                states[p_idx].set(c, cur_index);
                p_opt = states[p_idx].link;
            }
            if let Some(p_idx) = p_opt {
                let q_idx = states[p_idx].get(c).expect("transition must exist");
                if states[p_idx].len + 1 == states[q_idx].len {
                    states[cur_index].link = Some(q_idx);
                } else {
                    let clone_idx = states.len();
                    let mut cloned_state = states[q_idx].clone();
                    cloned_state.len = states[p_idx].len + 1;
                    states.push(cloned_state);
                    states[q_idx].link = Some(clone_idx);
                    states[cur_index].link = Some(clone_idx);
                    let mut current_opt = Some(p_idx);
                    while let Some(current) = current_opt {
                        if states[current].get(c) == Some(q_idx) {
                            states[current].set(c, clone_idx);
                            current_opt = states[current].link;
                        } else {
                            break;
                        }
                    }
                }
            } else {
                states[cur_index].link = Some(0);
            }
            last = cur_index;
        }
        i = phrase_end;
    }
    complexity
}

// ============================================================================
// 排序与部分选择工具
// ============================================================================

#[inline]
pub fn f32_sort_key(v: f32) -> u32 {
    let b = v.to_bits();
    if b >> 31 == 0 {
        b | 0x8000_0000
    } else {
        !b
    }
}

#[inline]
fn cmp_f32(a: &f32, b: &f32) -> Ordering {
    a.partial_cmp(b).unwrap_or(Ordering::Equal)
}

#[inline]
fn cmp_f64(a: &f64, b: &f64) -> Ordering {
    a.partial_cmp(b).unwrap_or(Ordering::Equal)
}

/// 4 趟 8 位稳定基数排序（仅用于非 NaN `f32`）。小数组自动走比较排序以省掉 1024 计数表开销。
pub fn sort_non_nan_f32(values: &mut [f32], radix_tmp: &mut Vec<f32>) {
    let n = values.len();
    if n < 2 {
        return;
    }
    if n < 96 {
        values.sort_unstable_by(cmp_f32);
        return;
    }
    radix_tmp.clear();
    radix_tmp.resize(n, 0.0);
    let mut cnt = [0usize; 256];
    for shift in [0u32, 8, 16, 24] {
        cnt.fill(0);
        for &v in values.iter() {
            cnt[((f32_sort_key(v) >> shift) & 0xff) as usize] += 1;
        }
        let mut acc = 0usize;
        for c in cnt.iter_mut() {
            let t = *c;
            *c = acc;
            acc += t;
        }
        for &v in values.iter() {
            let k = f32_sort_key(v);
            let b = ((k >> shift) & 0xff) as usize;
            radix_tmp[cnt[b]] = v;
            cnt[b] += 1;
        }
        values.swap_with_slice(radix_tmp);
    }
}

// ============================================================================
// 线程级复用缓冲
// ============================================================================

#[derive(Default)]
pub struct FeaturesScratch {
    pub(crate) col_major: Vec<f32>,
    pub(crate) valid: Vec<f32>,
    pub(crate) idx1: Vec<u32>,
    pub(crate) sorted: Vec<f32>,
    pub(crate) lz_sorted: Vec<f32>,
    pub(crate) radix: Vec<f32>,
    pub(crate) discrete: Vec<u8>,
    pub(crate) sam_states: Vec<SamState>,
    pub(crate) ent_counts: Vec<usize>,
    // 批量相关系数专用缓冲
    pub(crate) complete_cols: Vec<usize>,
    pub(crate) full_to_comp: Vec<usize>,
    pub(crate) z_mat: Vec<f32>,
    pub(crate) sqrt_var: Vec<f32>,
    // f64 专用缓冲
    pub(crate) valid_f64: Vec<f64>,
    pub(crate) complete_cols_f64: Vec<usize>,
    pub(crate) full_to_comp_f64: Vec<usize>,
    pub(crate) z_mat_f64: Vec<f64>,
    pub(crate) sqrt_var_f64: Vec<f64>,
}

impl FeaturesScratch {
    pub fn new() -> Self {
        Self::default()
    }

    fn trim_oversized(&mut self) {
        if self.col_major.capacity() > RETAINED_SCRATCH_FLOATS {
            self.col_major.clear();
            self.col_major.shrink_to(RETAINED_SCRATCH_FLOATS);
        }
        if self.z_mat.capacity() > RETAINED_SCRATCH_FLOATS {
            self.z_mat.clear();
            self.z_mat.shrink_to(RETAINED_SCRATCH_FLOATS);
        }
        if self.z_mat_f64.capacity() > RETAINED_SCRATCH_FLOATS / 2 {
            self.z_mat_f64.clear();
            self.z_mat_f64.shrink_to(RETAINED_SCRATCH_FLOATS / 2);
        }
    }
}

thread_local! {
    static TLS_FEATURES_SCRATCH: RefCell<FeaturesScratch> = RefCell::new(FeaturesScratch::new());
}

#[inline]
pub fn with_tls_scratch<R>(f: impl FnOnce(&mut FeaturesScratch) -> R) -> R {
    TLS_FEATURES_SCRATCH.with(|cell| {
        if let Ok(mut guard) = cell.try_borrow_mut() {
            let r = f(&mut guard);
            guard.trim_oversized();
            r
        } else {
            let mut tmp = FeaturesScratch::new();
            f(&mut tmp)
        }
    })
}

// ============================================================================
// 零分配一维基础算子（与 pandas / 旧 features.rs 逐位一致）
// ============================================================================

#[inline]
pub fn col_mean(col: &[f32]) -> f32 {
    let (sum, n) = col.iter().fold((0.0f32, 0usize), |(s, c), &v| {
        if v.is_nan() {
            (s, c)
        } else {
            (s + v, c + 1)
        }
    });
    if n == 0 {
        f32::NAN
    } else {
        sum / n as f32
    }
}

#[inline]
pub fn col_std(col: &[f32]) -> f32 {
    let (mut sum, mut n) = (0.0f32, 0usize);
    for &v in col {
        if !v.is_nan() {
            sum += v;
            n += 1;
        }
    }
    if n < 2 {
        return f32::NAN;
    }
    let mean = sum / n as f32;
    let mut s2 = 0.0f32;
    for &v in col {
        if !v.is_nan() {
            let d = v - mean;
            s2 += d * d;
        }
    }
    (s2 / (n - 1) as f32).sqrt()
}

#[inline]
pub fn col_skew(col: &[f32]) -> f32 {
    let (mut sum, mut n) = (0.0f32, 0usize);
    for &v in col {
        if !v.is_nan() {
            sum += v;
            n += 1;
        }
    }
    if n < 3 {
        return f32::NAN;
    }
    let mean = sum / n as f32;
    let nf = n as f32;
    let (mut s2, mut s3) = (0.0f32, 0.0f32);
    for &v in col {
        if !v.is_nan() {
            let d = v - mean;
            s2 += d * d;
            s3 += d * d * d;
        }
    }
    let k2 = s2 / (nf - 1.0);
    let k3 = nf * s3 / ((nf - 1.0) * (nf - 2.0));
    if k2.abs() < 1e-30 {
        0.0
    } else {
        k3 / k2.powf(1.5)
    }
}

#[inline]
pub fn col_kurt(col: &[f32]) -> f32 {
    let (mut sum, mut n) = (0.0f32, 0usize);
    for &v in col {
        if !v.is_nan() {
            sum += v;
            n += 1;
        }
    }
    if n < 4 {
        return f32::NAN;
    }
    let mean = sum / n as f32;
    let nf = n as f32;
    let (mut s2, mut s4) = (0.0f32, 0.0f32);
    for &v in col {
        if !v.is_nan() {
            let d = v - mean;
            let d2 = d * d;
            s2 += d2;
            s4 += d2 * d2;
        }
    }
    let k2 = s2 / (nf - 1.0);
    let k4 = nf * ((nf + 1.0) * s4 - 3.0 * (nf - 1.0) * s2 * s2 / nf)
        / ((nf - 1.0) * (nf - 2.0) * (nf - 3.0));
    if k2.abs() < 1e-30 {
        0.0
    } else {
        k4 / (k2 * k2)
    }
}

/// 单列中位数：利用 `select_nth_unstable_by` 在 `O(n)` 时间内定位中位数，零堆重分配，与全排序逐位一致。
#[inline]
pub fn col_median(col: &[f32]) -> f32 {
    with_tls_scratch(|sc| {
        sc.valid.clear();
        sc.valid.extend(col.iter().copied().filter(|v| !v.is_nan()));
        let n = sc.valid.len();
        if n == 0 {
            return f32::NAN;
        }
        let mid = n / 2;
        sc.valid.select_nth_unstable_by(mid, cmp_f32);
        let hi = sc.valid[mid];
        if n % 2 == 1 {
            hi
        } else {
            let lo = sc.valid[..mid]
                .iter()
                .copied()
                .fold(f32::NEG_INFINITY, |a, b| if cmp_f32(&b, &a) == Ordering::Greater { b } else { a });
            (lo + hi) / 2.0
        }
    })
}

/// 单列分位数：利用 `select_nth_unstable_by` 在 `O(n)` 时间内定位插值相邻两点，与 pandas 线性插值逐位一致。
#[inline]
pub fn col_quantile(col: &[f32], q: f32) -> f32 {
    with_tls_scratch(|sc| {
        sc.valid.clear();
        sc.valid.extend(col.iter().copied().filter(|v| !v.is_nan()));
        let n = sc.valid.len();
        if n == 0 {
            return f32::NAN;
        }
        if n == 1 {
            return sc.valid[0];
        }
        let pos = q * (n - 1) as f32;
        let lower = pos.floor() as usize;
        let upper = (lower + 1).min(n - 1);
        let frac = pos - lower as f32;
        sc.valid.select_nth_unstable_by(lower, cmp_f32);
        let v_lo = sc.valid[lower];
        let v_hi = if upper == lower {
            v_lo
        } else {
            sc.valid[lower + 1..].iter().copied().fold(
                f32::INFINITY,
                |a, b| if cmp_f32(&b, &a) == Ordering::Less { b } else { a },
            )
        };
        v_lo * (1.0 - frac) + v_hi * frac
    })
}

#[inline]
pub fn corr_pair(col_i: &[f32], col_j: &[f32]) -> f32 {
    let m = col_i.len().min(col_j.len());
    let (mut sum_i, mut sum_j, mut n) = (0.0f32, 0.0f32, 0usize);
    for k in 0..m {
        let a = col_i[k];
        let b = col_j[k];
        if !a.is_nan() && !b.is_nan() {
            sum_i += a;
            sum_j += b;
            n += 1;
        }
    }
    if n < 2 {
        return f32::NAN;
    }
    let mean_i = sum_i / n as f32;
    let mean_j = sum_j / n as f32;
    let (mut cov, mut var_i, mut var_j) = (0.0f32, 0.0f32, 0.0f32);
    for k in 0..m {
        let a = col_i[k];
        let b = col_j[k];
        if !a.is_nan() && !b.is_nan() {
            let di = a - mean_i;
            let dj = b - mean_j;
            cov += di * dj;
            var_i += di * di;
            var_j += dj * dj;
        }
    }
    if var_i == 0.0 || var_j == 0.0 {
        0.0
    } else {
        cov / (var_i.sqrt() * var_j.sqrt())
    }
}

#[inline]
pub fn trend_1d(col: &[f32]) -> f32 {
    let (mut sum_x, mut sum_y, mut n) = (0.0f32, 0.0f32, 0usize);
    for (i, &v) in col.iter().enumerate() {
        if !v.is_nan() {
            sum_x += (i + 1) as f32;
            sum_y += v;
            n += 1;
        }
    }
    if n < 2 {
        return 0.0;
    }
    let mean_x = sum_x / n as f32;
    let mean_y = sum_y / n as f32;
    let (mut cov, mut var_x, mut var_y) = (0.0f32, 0.0f32, 0.0f32);
    for (i, &v) in col.iter().enumerate() {
        if !v.is_nan() {
            let dx = (i + 1) as f32 - mean_x;
            let dy = v - mean_y;
            cov += dx * dy;
            var_x += dx * dx;
            var_y += dy * dy;
        }
    }
    if var_x == 0.0 || var_y == 0.0 {
        0.0
    } else {
        cov / (var_x.sqrt() * var_y.sqrt())
    }
}

#[inline]
fn solve3(mut m: [[f64; 4]; 3]) -> Option<[f64; 3]> {
    for k in 0..3 {
        let mut piv = k;
        for i in (k + 1)..3 {
            if m[i][k].abs() > m[piv][k].abs() {
                piv = i;
            }
        }
        if m[piv][k].abs() < 1e-30 {
            return None;
        }
        if piv != k {
            m.swap(piv, k);
        }
        for i in 0..3 {
            if i != k {
                let f = m[i][k] / m[k][k];
                for j in k..4 {
                    m[i][j] -= f * m[k][j];
                }
            }
        }
    }
    Some([m[0][3] / m[0][0], m[1][3] / m[1][1], m[2][3] / m[2][2]])
}

#[inline]
pub fn curvature_1d(col: &[f32]) -> f32 {
    let (mut sum_t, mut sum_y, mut n) = (0.0f64, 0.0f64, 0usize);
    for (i, &v) in col.iter().enumerate() {
        if !v.is_nan() {
            sum_t += (i + 1) as f64;
            sum_y += v as f64;
            n += 1;
        }
    }
    if n < 3 {
        return 0.0;
    }
    let nf = n as f64;
    let mean_t = sum_t / nf;
    let mean_y = sum_y / nf;
    let mut sum_q = 0.0f64;
    for (i, &v) in col.iter().enumerate() {
        if !v.is_nan() {
            let dt = (i + 1) as f64 - mean_t;
            sum_q += dt * dt;
        }
    }
    let mean_q = sum_q / nf;
    let (mut cov, mut var_y, mut var_q) = (0.0f64, 0.0f64, 0.0f64);
    for (i, &v) in col.iter().enumerate() {
        if !v.is_nan() {
            let dt = (i + 1) as f64 - mean_t;
            let q = dt * dt;
            let dy = v as f64 - mean_y;
            let dq = q - mean_q;
            cov += dy * dq;
            var_y += dy * dy;
            var_q += dq * dq;
        }
    }
    if var_y == 0.0 || var_q == 0.0 {
        0.0
    } else {
        (cov / (var_y.sqrt() * var_q.sqrt())) as f32
    }
}

#[inline]
pub fn quad_coef_1d(col: &[f32]) -> f32 {
    let (mut sum_t, mut sum_y, mut n) = (0.0f64, 0.0f64, 0usize);
    for (i, &v) in col.iter().enumerate() {
        if !v.is_nan() {
            sum_t += (i + 1) as f64;
            sum_y += v as f64;
            n += 1;
        }
    }
    if n < 4 {
        return 0.0;
    }
    let nf = n as f64;
    let mean_t = sum_t / nf;
    let mean_y = sum_y / nf;
    let (mut s_uu, mut s_uuu, mut s_uuuu, mut s_uy, mut s_uu_y, mut ss_tot) =
        (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for (i, &v) in col.iter().enumerate() {
        if !v.is_nan() {
            let t = (i + 1) as f64;
            let y = v as f64;
            let u = t - mean_t;
            let dy = y - mean_y;
            let uu = u * u;
            s_uu += uu;
            s_uuu += uu * u;
            s_uuuu += uu * uu;
            s_uy += u * y;
            s_uu_y += uu * y;
            ss_tot += dy * dy;
        }
    }
    if ss_tot <= 0.0 || s_uu <= 0.0 {
        return 0.0;
    }
    let b1 = s_uy / s_uu;
    let ss_res_lin = ss_tot - b1 * s_uy;
    let m = [
        [nf, 0.0, s_uu, nf * mean_y],
        [0.0, s_uu, s_uuu, s_uy],
        [s_uu, s_uuu, s_uuuu, s_uu_y],
    ];
    let sol = match solve3(m) {
        Some(s) => s,
        None => return 0.0,
    };
    let (c2, b2, a2) = (sol[0], sol[1], sol[2]);
    let ssr_quad = c2 * (nf * mean_y) + b2 * s_uy + a2 * s_uu_y - nf * mean_y * mean_y;
    let ss_res_quad = ss_tot - ssr_quad;
    let r2_lin = 1.0 - ss_res_lin / ss_tot;
    let r2_quad = 1.0 - ss_res_quad / ss_tot;
    let delta = (r2_quad - r2_lin).clamp(0.0, 1.0);
    let sign = if a2 >= 0.0 { 1.0 } else { -1.0 };
    (sign * delta) as f32
}

pub fn lz_complexity_1d(col: &[f32]) -> f32 {
    with_tls_scratch(|sc| {
        sc.valid.clear();
        sc.valid.extend(col.iter().copied().filter(|v| v.is_finite()));
        lz_from_finite_slice(&sc.valid, None, &mut sc.lz_sorted, &mut sc.radix, &mut sc.discrete, &mut sc.sam_states)
    })
}

fn lz_from_finite_slice(
    finite_vals: &[f32],
    already_sorted: Option<&[f32]>,
    lz_sorted: &mut Vec<f32>,
    radix_tmp: &mut Vec<f32>,
    discrete: &mut Vec<u8>,
    sam_states: &mut Vec<SamState>,
) -> f32 {
    let n = finite_vals.len();
    if n == 0 {
        return 0.0;
    }
    let sorted_ref: &[f32] = match already_sorted {
        Some(s) => s,
        None => {
            lz_sorted.clear();
            lz_sorted.extend_from_slice(finite_vals);
            sort_non_nan_f32(lz_sorted, radix_tmp);
            lz_sorted.as_slice()
        }
    };
    let sn = sorted_ref.len();
    let t0 = sorted_ref[((sn - 1) as f32 * 0.33f32) as usize];
    let t1 = sorted_ref[((sn - 1) as f32 * 0.66f32) as usize];

    discrete.clear();
    discrete.reserve(n);
    let mut seen = [false; 4];
    for &val in finite_vals {
        let sym = if val <= t0 {
            1u8
        } else if val <= t1 {
            2u8
        } else {
            3u8
        };
        seen[sym as usize] = true;
        discrete.push(sym);
    }

    let complexity = lz_complexity_reused(discrete, sam_states);
    if n <= 1 {
        return 0.0;
    }
    let k_eff = (seen[1] as usize + seen[2] as usize + seen[3] as usize) as f32;
    if k_eff < 3.0 {
        return f32::NAN;
    }
    let log_n_base_k = (n as f32).ln() / k_eff.ln();
    complexity as f32 * log_n_base_k / n as f32
}

pub fn binned_entropy_1d(col: &[f32], n_bins: usize) -> f32 {
    with_tls_scratch(|sc| {
        sc.valid.clear();
        let (mut min_val, mut max_val) = (f32::INFINITY, f32::NEG_INFINITY);
        for &v in col {
            if !v.is_nan() {
                sc.valid.push(v);
                min_val = min_val.min(v);
                max_val = max_val.max(v);
            }
        }
        binned_entropy_from_valid(&sc.valid, min_val, max_val, n_bins, &mut sc.ent_counts)
    })
}

#[inline]
fn binned_entropy_from_valid(
    valid: &[f32],
    min_val: f32,
    max_val: f32,
    n_bins: usize,
    counts: &mut Vec<usize>,
) -> f32 {
    if valid.is_empty() || n_bins == 0 {
        return 0.0;
    }
    if (max_val - min_val).abs() < f32::EPSILON {
        return 0.0;
    }
    let bin_width = (max_val - min_val) / n_bins as f32;
    counts.clear();
    counts.resize(n_bins, 0);
    for &v in valid {
        let mut idx = ((v - min_val) / bin_width).floor() as usize;
        if idx >= n_bins {
            idx = n_bins - 1;
        }
        counts[idx] += 1;
    }
    let total = valid.len() as f32;
    counts
        .iter()
        .map(|&c| {
            let p = c as f32 / total;
            if p > 0.0 {
                -p * p.ln()
            } else {
                0.0
            }
        })
        .sum()
}

pub fn max_range_product_strict(col: &[f32]) -> f32 {
    with_tls_scratch(|sc| {
        sc.valid.clear();
        sc.valid.extend(col.iter().copied().filter(|v| !v.is_nan()));
        max_range_from_valid(&sc.valid, col.len())
    })
}

#[inline]
fn max_range_from_valid(valid: &[f32], n_total: usize) -> f32 {
    let n = valid.len();
    if n < 2 || n_total == 0 {
        return 0.0;
    }
    let mut max_product = f32::NEG_INFINITY;
    let mut result = (0i64, 0i64);
    let mut left = 0usize;
    let mut right = n - 1;
    while left < right {
        let product = valid[left].min(valid[right]) * (right - left) as f32;
        if product > max_product {
            max_product = product;
            result = (left as i64, right as i64);
        }
        if valid[left] < valid[right] {
            left += 1;
        } else {
            right -= 1;
        }
    }
    for i in 0..n - 1 {
        let product = valid[i].min(valid[i + 1]) * 1.0;
        if product > max_product {
            max_product = product;
            result = (i as i64, (i + 1) as i64);
        }
    }
    let (i, j) = result;
    (i - j).abs() as f32 / n_total as f32
}

// ============================================================================
// 供沙箱与新因子直接调用的「一次扫描 + 单次排序」单列全统计汇总结构
// ============================================================================

#[derive(Clone, Copy, Debug)]
pub struct ColSummaryF64 {
    pub count: usize,
    pub mean: f64,
    pub std: f64,
    pub skew: f64,
    pub kurt: f64,
    pub min: f64,
    pub p5: f64,
    pub p10: f64,
    pub p25: f64,
    pub median: f64,
    pub p75: f64,
    pub p90: f64,
    pub p95: f64,
    pub max: f64,
}

impl ColSummaryF64 {
    pub const NAN: Self = Self {
        count: 0,
        mean: f64::NAN,
        std: f64::NAN,
        skew: f64::NAN,
        kurt: f64::NAN,
        min: f64::NAN,
        p5: f64::NAN,
        p10: f64::NAN,
        p25: f64::NAN,
        median: f64::NAN,
        p75: f64::NAN,
        p90: f64::NAN,
        p95: f64::NAN,
        max: f64::NAN,
    };
}

/// 单列一次过滤 + 一次中心矩扫描 + 单次排序提取全套矩与分位数（零堆重分配，适合 `sandbox_*` 直接调用）。
pub fn col_summary_f64(col: &[f64]) -> ColSummaryF64 {
    with_tls_scratch(|sc| {
        sc.valid_f64.clear();
        let mut sum = 0.0f64;
        for &v in col {
            if v.is_finite() {
                sc.valid_f64.push(v);
                sum += v;
            }
        }
        let n = sc.valid_f64.len();
        if n == 0 {
            return ColSummaryF64::NAN;
        }
        let nf = n as f64;
        let mean = sum / nf;
        let (mut s2, mut s3, mut s4) = (0.0f64, 0.0f64, 0.0f64);
        for &x in &sc.valid_f64 {
            let d = x - mean;
            let d2 = d * d;
            s2 += d2;
            s3 += d2 * d;
            s4 += d2 * d2;
        }
        let std = if n >= 2 { (s2 / (nf - 1.0)).sqrt() } else { f64::NAN };
        let skew = if n < 3 {
            f64::NAN
        } else {
            let k2 = s2 / (nf - 1.0);
            if k2.abs() < 1e-30 {
                0.0
            } else {
                let k3 = nf * s3 / ((nf - 1.0) * (nf - 2.0));
                k3 / k2.powf(1.5)
            }
        };
        let kurt = if n < 4 {
            f64::NAN
        } else {
            let k2 = s2 / (nf - 1.0);
            if k2.abs() < 1e-30 {
                0.0
            } else {
                let k4 = nf * ((nf + 1.0) * s4 - 3.0 * (nf - 1.0) * s2 * s2 / nf)
                    / ((nf - 1.0) * (nf - 2.0) * (nf - 3.0));
                k4 / (k2 * k2)
            }
        };

        sc.valid_f64.sort_unstable_by(cmp_f64);
        let s = &sc.valid_f64;
        let q_at = |q: f64| -> f64 {
            if n == 1 {
                return s[0];
            }
            let pos = q * (n - 1) as f64;
            let lo = pos.floor() as usize;
            let hi = (lo + 1).min(n - 1);
            let frac = pos - lo as f64;
            s[lo] * (1.0 - frac) + s[hi] * frac
        };
        let median = if n % 2 == 1 {
            s[n / 2]
        } else {
            0.5 * (s[n / 2 - 1] + s[n / 2])
        };
        ColSummaryF64 {
            count: n,
            mean,
            std,
            skew,
            kurt,
            min: s[0],
            p5: q_at(0.05),
            p10: q_at(0.10),
            p25: q_at(0.25),
            median,
            p75: q_at(0.75),
            p90: q_at(0.90),
            p95: q_at(0.95),
            max: s[n - 1],
        }
    })
}

// ============================================================================
// 批量相关系数矩阵上三角（4×16 寄存器分块 Z^T Z + 含 NaN 列零分配回退）
// ============================================================================

/// 计算列主序矩阵 `col_major[col * n_rows + row]` 的两两 Pearson 相关系数上三角
/// （按 `for i in 0..n_cols { for j in i+1..n_cols }` 顺序写入 `out`）。
///
/// 算法：
/// - 扫描每列是否含 `NaN`；所有无 `NaN` 列（绝大多数情形）一次性去均值排成 16 列对齐的行主序矩阵 `Z`，
///   随后用 4×16 寄存器分块外积（沿 `j` 方向 16 路 SIMD，沿 `t=0..n_rows` 顺序累加）批量算出协方差，
///   不仅消除 `O(C^2)` 次临时 `Vec` 堆分配和重复求均值/方差，还保持与 `corr_pair` **逐位一致（to_bits 相等）**！
/// - 涉及含 `NaN` 列的配对回退零分配 `corr_pair`。
pub fn batch_corr_upper_col_major_f32(
    col_major: &[f32],
    n_rows: usize,
    n_cols: usize,
    sc: &mut FeaturesScratch,
    out: &mut [f32],
) {
    let n_pairs = n_cols.saturating_mul(n_cols.saturating_sub(1)) / 2;
    assert_eq!(out.len(), n_pairs);
    if n_cols < 2 {
        return;
    }
    if n_rows < 2 {
        out.fill(f32::NAN);
        return;
    }

    sc.complete_cols.clear();
    sc.full_to_comp.clear();
    sc.full_to_comp.resize(n_cols, usize::MAX);
    sc.sqrt_var.clear();

    for c in 0..n_cols {
        let col = &col_major[c * n_rows..(c + 1) * n_rows];
        let mut sum = 0.0f32;
        let mut has_nan = false;
        for &v in col {
            if v.is_nan() {
                has_nan = true;
                break;
            }
            sum += v;
        }
        if !has_nan {
            let k = sc.complete_cols.len();
            sc.complete_cols.push(c);
            sc.full_to_comp[c] = k;
            sc.sqrt_var.push(sum / n_rows as f32); // 暂存 mean，下一步算中心化与 sqrt_var
        }
    }

    let m = sc.complete_cols.len();
    let stride = (m + 15) & !15; // 16-lane 对齐，便于 AVX-512 整块加载且越界位恒为 0
    if m >= 2 {
        sc.z_mat.clear();
        sc.z_mat.resize(n_rows * stride, 0.0f32);
        for k in 0..m {
            let c = sc.complete_cols[k];
            let col = &col_major[c * n_rows..(c + 1) * n_rows];
            let mean = sc.sqrt_var[k];
            let mut var = 0.0f32;
            for t in 0..n_rows {
                let d = col[t] - mean;
                sc.z_mat[t * stride + k] = d;
                var += d * d;
            }
            sc.sqrt_var[k] = var.sqrt();
        }

        // 先按 4×16 分块批量计算所有无 NaN 列对的相关系数并直接填进 out 对应槽位
        // 行号 ci < cj 对应上三角槽位：ci * (2 * n_cols - ci - 1) / 2 + (cj - ci - 1)
        #[inline(always)]
        fn pair_idx(i: usize, j: usize, n_cols: usize) -> usize {
            i * (2 * n_cols - i - 1) / 2 + (j - i - 1)
        }

        let z = &sc.z_mat[..];
        let mut i0 = 0usize;
        while i0 < m {
            let ib = (m - i0).min(4);
            let mut j0 = i0 & !15;
            while j0 < m {
                // 只算 j >= i0 + 1 所在块
                if j0 + 16 > i0 + 1 {
                    let mut acc0 = [0.0f32; 16];
                    let mut acc1 = [0.0f32; 16];
                    let mut acc2 = [0.0f32; 16];
                    let mut acc3 = [0.0f32; 16];
                    match ib {
                        4 => {
                            for t in 0..n_rows {
                                let row = &z[t * stride..t * stride + stride];
                                let a0 = row[i0];
                                let a1 = row[i0 + 1];
                                let a2 = row[i0 + 2];
                                let a3 = row[i0 + 3];
                                let b: &[f32; 16] = row[j0..j0 + 16].try_into().unwrap();
                                for l in 0..16 {
                                    let bv = b[l];
                                    acc0[l] += a0 * bv;
                                    acc1[l] += a1 * bv;
                                    acc2[l] += a2 * bv;
                                    acc3[l] += a3 * bv;
                                }
                            }
                        }
                        3 => {
                            for t in 0..n_rows {
                                let row = &z[t * stride..t * stride + stride];
                                let a0 = row[i0];
                                let a1 = row[i0 + 1];
                                let a2 = row[i0 + 2];
                                let b: &[f32; 16] = row[j0..j0 + 16].try_into().unwrap();
                                for l in 0..16 {
                                    let bv = b[l];
                                    acc0[l] += a0 * bv;
                                    acc1[l] += a1 * bv;
                                    acc2[l] += a2 * bv;
                                }
                            }
                        }
                        2 => {
                            for t in 0..n_rows {
                                let row = &z[t * stride..t * stride + stride];
                                let a0 = row[i0];
                                let a1 = row[i0 + 1];
                                let b: &[f32; 16] = row[j0..j0 + 16].try_into().unwrap();
                                for l in 0..16 {
                                    let bv = b[l];
                                    acc0[l] += a0 * bv;
                                    acc1[l] += a1 * bv;
                                }
                            }
                        }
                        _ => {
                            for t in 0..n_rows {
                                let row = &z[t * stride..t * stride + stride];
                                let a0 = row[i0];
                                let b: &[f32; 16] = row[j0..j0 + 16].try_into().unwrap();
                                for l in 0..16 {
                                    acc0[l] += a0 * b[l];
                                }
                            }
                        }
                    }
                    let accs = [&acc0, &acc1, &acc2, &acc3];
                    for ia in 0..ib {
                        let ki = i0 + ia;
                        let ci = sc.complete_cols[ki];
                        let svi = sc.sqrt_var[ki];
                        let l_end = (m - j0).min(16);
                        for l in 0..l_end {
                            let kj = j0 + l;
                            if kj <= ki {
                                continue;
                            }
                            let cj = sc.complete_cols[kj];
                            let svj = sc.sqrt_var[kj];
                            let val = if svi == 0.0 || svj == 0.0 {
                                0.0
                            } else {
                                accs[ia][l] / (svi * svj)
                            };
                            out[pair_idx(ci, cj, n_cols)] = val;
                        }
                    }
                }
                j0 += 16;
            }
            i0 += 4;
        }
    }

    // 补充任何涉及含 NaN 列（或 m < 2 时唯一无 NaN 列与其他列）的配对
    if m < n_cols {
        let mut idx = 0usize;
        for i in 0..n_cols {
            let ci_comp = sc.full_to_comp[i] != usize::MAX && m >= 2;
            let col_i = &col_major[i * n_rows..(i + 1) * n_rows];
            for j in (i + 1)..n_cols {
                let cj_comp = sc.full_to_comp[j] != usize::MAX && m >= 2;
                if !(ci_comp && cj_comp) {
                    let col_j = &col_major[j * n_rows..(j + 1) * n_rows];
                    out[idx] = corr_pair(col_i, col_j);
                }
                idx += 1;
            }
        }
    }
}

/// 行主序 `f32` 矩阵 `[n_rows × n_cols]` 的批量相关系数上三角便捷入口。
pub fn batch_corr_upper_row_major_f32(
    row_major: &[f32],
    n_rows: usize,
    n_cols: usize,
    out: &mut [f32],
) {
    assert_eq!(row_major.len(), n_rows * n_cols);
    with_tls_scratch(|sc| {
        transpose_row_to_col_f32(row_major, n_rows, n_cols, &mut sc.col_major);
        let col_major = std::mem::take(&mut sc.col_major);
        batch_corr_upper_col_major_f32(&col_major, n_rows, n_cols, sc, out);
        sc.col_major = col_major;
    });
}

/// 行主序 `f64` 矩阵 `[n_rows × n_cols]` 的批量相关系数上三角便捷入口（4×8 寄存器分块 `Z^T Z`，供 `sandbox_*` 与 `f64` 因子直接复用）。
pub fn batch_corr_upper_row_major_f64(
    row_major: &[f64],
    n_rows: usize,
    n_cols: usize,
    out: &mut [f64],
) {
    assert_eq!(row_major.len(), n_rows * n_cols);
    let n_pairs = n_cols.saturating_mul(n_cols.saturating_sub(1)) / 2;
    assert_eq!(out.len(), n_pairs);
    if n_cols < 2 {
        return;
    }
    if n_rows < 2 {
        out.fill(f64::NAN);
        return;
    }
    with_tls_scratch(|sc| {
        sc.complete_cols_f64.clear();
        sc.full_to_comp_f64.clear();
        sc.full_to_comp_f64.resize(n_cols, usize::MAX);
        sc.sqrt_var_f64.clear();

        for c in 0..n_cols {
            let mut sum = 0.0f64;
            let mut has_nan = false;
            for r in 0..n_rows {
                let v = row_major[r * n_cols + c];
                if !v.is_finite() {
                    has_nan = true;
                    break;
                }
                sum += v;
            }
            if !has_nan {
                let k = sc.complete_cols_f64.len();
                sc.complete_cols_f64.push(c);
                sc.full_to_comp_f64[c] = k;
                sc.sqrt_var_f64.push(sum / n_rows as f64);
            }
        }

        let m = sc.complete_cols_f64.len();
        let stride = (m + 7) & !7; // 8-lane f64 AVX-512 对齐
        if m >= 2 {
            sc.z_mat_f64.clear();
            sc.z_mat_f64.resize(n_rows * stride, 0.0f64);
            let mut vars = vec![0.0f64; m];
            for r in 0..n_rows {
                let src = &row_major[r * n_cols..(r + 1) * n_cols];
                let dst = &mut sc.z_mat_f64[r * stride..(r + 1) * stride];
                for k in 0..m {
                    let c = sc.complete_cols_f64[k];
                    let d = src[c] - sc.sqrt_var_f64[k];
                    dst[k] = d;
                    vars[k] += d * d;
                }
            }
            for k in 0..m {
                sc.sqrt_var_f64[k] = vars[k].sqrt();
            }

            #[inline(always)]
            fn pair_idx(i: usize, j: usize, n_cols: usize) -> usize {
                i * (2 * n_cols - i - 1) / 2 + (j - i - 1)
            }

            let z = &sc.z_mat_f64[..];
            let mut i0 = 0usize;
            while i0 < m {
                let ib = (m - i0).min(4);
                let mut j0 = i0 & !7;
                while j0 < m {
                    if j0 + 8 > i0 + 1 {
                        let mut acc0 = [0.0f64; 8];
                        let mut acc1 = [0.0f64; 8];
                        let mut acc2 = [0.0f64; 8];
                        let mut acc3 = [0.0f64; 8];
                        for t in 0..n_rows {
                            let row = &z[t * stride..t * stride + stride];
                            let b: &[f64; 8] = row[j0..j0 + 8].try_into().unwrap();
                            let a0 = row[i0];
                            for l in 0..8 {
                                acc0[l] += a0 * b[l];
                            }
                            if ib >= 2 {
                                let a1 = row[i0 + 1];
                                for l in 0..8 {
                                    acc1[l] += a1 * b[l];
                                }
                            }
                            if ib >= 3 {
                                let a2 = row[i0 + 2];
                                for l in 0..8 {
                                    acc2[l] += a2 * b[l];
                                }
                            }
                            if ib == 4 {
                                let a3 = row[i0 + 3];
                                for l in 0..8 {
                                    acc3[l] += a3 * b[l];
                                }
                            }
                        }
                        let accs = [&acc0, &acc1, &acc2, &acc3];
                        for ia in 0..ib {
                            let ki = i0 + ia;
                            let ci = sc.complete_cols_f64[ki];
                            let svi = sc.sqrt_var_f64[ki];
                            let l_end = (m - j0).min(8);
                            for l in 0..l_end {
                                let kj = j0 + l;
                                if kj <= ki {
                                    continue;
                                }
                                let cj = sc.complete_cols_f64[kj];
                                let svj = sc.sqrt_var_f64[kj];
                                out[pair_idx(ci, cj, n_cols)] = if svi == 0.0 || svj == 0.0 {
                                    0.0
                                } else {
                                    accs[ia][l] / (svi * svj)
                                };
                            }
                        }
                    }
                    j0 += 8;
                }
                i0 += 4;
            }
        }

        if m < n_cols {
            let mut idx = 0usize;
            for i in 0..n_cols {
                let ci_comp = sc.full_to_comp_f64[i] != usize::MAX && m >= 2;
                for j in (i + 1)..n_cols {
                    let cj_comp = sc.full_to_comp_f64[j] != usize::MAX && m >= 2;
                    if !(ci_comp && cj_comp) {
                        let (mut sa, mut sb, mut n) = (0.0f64, 0.0f64, 0usize);
                        for r in 0..n_rows {
                            let a = row_major[r * n_cols + i];
                            let b = row_major[r * n_cols + j];
                            if a.is_finite() && b.is_finite() {
                                sa += a;
                                sb += b;
                                n += 1;
                            }
                        }
                        out[idx] = if n < 2 {
                            f64::NAN
                        } else {
                            let ma = sa / n as f64;
                            let mb = sb / n as f64;
                            let (mut cov, mut va, mut vb) = (0.0f64, 0.0f64, 0.0f64);
                            for r in 0..n_rows {
                                let a = row_major[r * n_cols + i];
                                let b = row_major[r * n_cols + j];
                                if a.is_finite() && b.is_finite() {
                                    let da = a - ma;
                                    let db = b - mb;
                                    cov += da * db;
                                    va += da * da;
                                    vb += db * db;
                                }
                            }
                            if va == 0.0 || vb == 0.0 {
                                0.0
                            } else {
                                cov / (va.sqrt() * vb.sqrt())
                            }
                        };
                    }
                    idx += 1;
                }
            }
        }
    });
}

// ============================================================================
// 全套矩阵统计降维引擎（对齐 features::get_features_factors_rust_full_opts，逐位一致）
// ============================================================================

#[inline]
pub fn transpose_row_to_col_f32(
    row_major: &[f32],
    n_rows: usize,
    n_cols: usize,
    col_major: &mut Vec<f32>,
) {
    col_major.clear();
    col_major.resize(n_rows * n_cols, 0.0);
    const TILE: usize = 32;
    let mut r0 = 0usize;
    while r0 < n_rows {
        let r1 = (r0 + TILE).min(n_rows);
        let mut c0 = 0usize;
        while c0 < n_cols {
            let c1 = (c0 + TILE).min(n_cols);
            for r in r0..r1 {
                let src = &row_major[r * n_cols + c0..r * n_cols + c1];
                for (dc, &v) in src.iter().enumerate() {
                    col_major[(c0 + dc) * n_rows + r] = v;
                }
            }
            c0 += TILE;
        }
        r0 += TILE;
    }
}

/// 计算输出向量总长度（给定 `n_cols`、是否含阈值计数、是否含二阶趋势）。
#[inline]
pub fn features_output_len(
    n_cols: usize,
    with_threshold_counts: bool,
    with_curvature: bool,
) -> usize {
    if n_cols == 0 {
        return 0;
    }
    let n_groups = 19usize
        + if with_curvature { 2 } else { 0 }
        + if with_threshold_counts { 2 } else { 0 };
    let n_pairs = n_cols * (n_cols - 1) / 2;
    n_groups * n_cols + n_pairs
}

/// 对列主序矩阵 `col_major[col * n_rows + row]` 一次性提取全部通用序列特征与上三角相关系数，
/// 输出顺序与数值同 `features::get_features_factors_rust_full_opts` **逐位一致（to_bits 相等）**。
pub fn extract_features_col_major_f32(
    col_major: &[f32],
    n_rows: usize,
    n_cols: usize,
    with_threshold_counts: bool,
    with_curvature: bool,
    sc: &mut FeaturesScratch,
    out: &mut Vec<f32>,
) {
    out.clear();
    if n_rows == 0 || n_cols == 0 {
        return;
    }
    let total_len = features_output_len(n_cols, with_threshold_counts, with_curvature);
    out.resize(total_len, 0.0f32);

    // 各统计组在 out 中的起始偏移（与 push_group 顺序严格一一对应）
    let mut g = 0usize;
    let mut next_off = || {
        let o = g * n_cols;
        g += 1;
        o
    };
    let off_mean = next_off();
    let off_median = next_off();
    let off_std = next_off();
    let off_skew = next_off();
    let off_kurt = next_off();
    let off_p5 = next_off();
    let off_p25 = next_off();
    let off_p75 = next_off();
    let off_p95 = next_off();
    let off_iqr = next_off();
    let off_cv = next_off();
    let off_ac1 = next_off();
    let off_ac1_abs = next_off();
    let off_trend = next_off();
    let (off_curv, off_quad) = if with_curvature {
        (next_off(), next_off())
    } else {
        (0, 0)
    };
    let off_pdiff = next_off();
    let off_pratio = next_off();
    let (off_above_p90, off_below_p10) = if with_threshold_counts {
        (next_off(), next_off())
    } else {
        (0, 0)
    };
    let off_corr = g * n_cols;
    let n_pairs = n_cols * (n_cols - 1) / 2;
    let off_lz = off_corr + n_pairs;
    let off_ent = off_lz + n_cols;
    let off_maxr = off_ent + n_cols;

    let split = n_rows / 3;
    let n_bins = (n_rows as f32).log2().ceil() as usize + 1;

    for j in 0..n_cols {
        let c = &col_major[j * n_rows..(j + 1) * n_rows];

        // ---------- 第 1 遍：过滤非 NaN + 同步累积一阶量 ----------
        sc.valid.clear();
        sc.idx1.clear();
        sc.valid.reserve(n_rows);
        sc.idx1.reserve(n_rows);

        let mut mean_sum = 0.0f32;
        let mut first_sum = 0.0f32;
        let mut first_n = 0usize;
        let mut last_sum = 0.0f32;
        let mut last_n = 0usize;
        let mut ac_sum_i = 0.0f32;
        let mut ac_sum_j = 0.0f32;
        let mut ac_n = 0usize;
        let mut sum_t_f32 = 0.0f32;
        let mut sum_t_f64 = 0.0f64;
        let mut sum_y_f64 = 0.0f64;
        let mut min_val = f32::INFINITY;
        let mut max_val = f32::NEG_INFINITY;
        let mut all_finite = true;
        let mut prev = f32::NAN;

        for (i, &v) in c.iter().enumerate() {
            if !v.is_nan() {
                let idx1 = (i + 1) as u32;
                sc.valid.push(v);
                sc.idx1.push(idx1);
                mean_sum += v;
                if i < split {
                    first_sum += v;
                    first_n += 1;
                }
                if split > 0 && i >= n_rows - split {
                    last_sum += v;
                    last_n += 1;
                }
                sum_t_f32 += idx1 as f32;
                if with_curvature {
                    sum_t_f64 += idx1 as f64;
                    sum_y_f64 += v as f64;
                }
                min_val = min_val.min(v);
                max_val = max_val.max(v);
                if !v.is_finite() {
                    all_finite = false;
                }
                if i > 0 && !prev.is_nan() {
                    ac_sum_i += v;
                    ac_sum_j += prev;
                    ac_n += 1;
                }
            }
            prev = v;
        }

        let n = sc.valid.len();
        let mean = if n == 0 {
            f32::NAN
        } else {
            mean_sum / n as f32
        };
        let first_mean = if split > 0 && first_n > 0 {
            first_sum / first_n as f32
        } else {
            f32::NAN
        };
        let last_mean = if split > 0 && last_n > 0 {
            last_sum / last_n as f32
        } else {
            f32::NAN
        };
        out[off_mean + j] = mean;
        out[off_pdiff + j] = last_mean - first_mean;
        out[off_pratio + j] = last_mean / (first_mean.abs() + 1e-8);

        // ---------- 第 2 遍（仅扫 valid）：一遍同时算出 s2, s3, s4 ----------
        let (std_v, skew_v, kurt_v) = if n < 2 {
            (f32::NAN, f32::NAN, f32::NAN)
        } else {
            let nf = n as f32;
            let (mut s2, mut s3, mut s4) = (0.0f32, 0.0f32, 0.0f32);
            for &x in &sc.valid {
                let d = x - mean;
                let d2 = d * d;
                s2 += d2;
                s3 += d2 * d;
                s4 += d2 * d2;
            }
            let k2 = s2 / (nf - 1.0);
            let std_val = k2.sqrt();
            let skew_val = if n < 3 {
                f32::NAN
            } else if k2.abs() < 1e-30 {
                0.0
            } else {
                let k3 = nf * s3 / ((nf - 1.0) * (nf - 2.0));
                k3 / k2.powf(1.5)
            };
            let kurt_val = if n < 4 {
                f32::NAN
            } else if k2.abs() < 1e-30 {
                0.0
            } else {
                let k4 = nf * ((nf + 1.0) * s4 - 3.0 * (nf - 1.0) * s2 * s2 / nf)
                    / ((nf - 1.0) * (nf - 2.0) * (nf - 3.0));
                k4 / (k2 * k2)
            };
            (std_val, skew_val, kurt_val)
        };
        out[off_std + j] = std_v;
        out[off_skew + j] = skew_v;
        out[off_kurt + j] = kurt_v;
        out[off_cv + j] = std_v / (mean.abs() + 1e-8);

        // ---------- 单次排序：median + 4/6 分位 + LZ 阈值一次拿齐 ----------
        sc.sorted.clear();
        sc.sorted.extend_from_slice(&sc.valid);
        sort_non_nan_f32(&mut sc.sorted, &mut sc.radix);

        let (median_v, p5_v, p25_v, p75_v, p95_v) = if n == 0 {
            (f32::NAN, f32::NAN, f32::NAN, f32::NAN, f32::NAN)
        } else {
            let s = &sc.sorted;
            let med = if n % 2 == 0 {
                (s[n / 2 - 1] + s[n / 2]) / 2.0
            } else {
                s[n / 2]
            };
            let q_at = |q: f32| -> f32 {
                if n == 1 {
                    return s[0];
                }
                let pos = q * (n - 1) as f32;
                let lower = pos.floor() as usize;
                let upper = (lower + 1).min(n - 1);
                let frac = pos - lower as f32;
                s[lower] * (1.0 - frac) + s[upper] * frac
            };
            let p5 = q_at(0.05);
            let p25 = q_at(0.25);
            let p75 = q_at(0.75);
            let p95 = q_at(0.95);
            if with_threshold_counts {
                let p90 = q_at(0.90);
                let p10 = q_at(0.10);
                let (mut sa, mut na, mut sb, mut nb) = (0.0f32, 0usize, 0.0f32, 0usize);
                for &v in &sc.valid {
                    if v > p90 {
                        sa += v;
                        na += 1;
                    }
                    if v < p10 {
                        sb += v;
                        nb += 1;
                    }
                }
                out[off_above_p90 + j] = if na == 0 { 0.0 } else { sa / na as f32 };
                out[off_below_p10 + j] = if nb == 0 { 0.0 } else { sb / nb as f32 };
            } else if n == 0 {
                // no-op
            }
            (med, p5, p25, p75, p95)
        };
        if with_threshold_counts && n == 0 {
            out[off_above_p90 + j] = 0.0;
            out[off_below_p10 + j] = 0.0;
        }
        out[off_median + j] = median_v;
        out[off_p5 + j] = p5_v;
        out[off_p25 + j] = p25_v;
        out[off_p75 + j] = p75_v;
        out[off_p95 + j] = p95_v;
        out[off_iqr + j] = p75_v - p25_v;

        // ---------- autocorr1（零分配第二遍） ----------
        let ac1 = if n_rows < 2 || ac_n < 2 {
            f32::NAN
        } else {
            let mi = ac_sum_i / ac_n as f32;
            let mj = ac_sum_j / ac_n as f32;
            let (mut cov, mut vi, mut vj) = (0.0f32, 0.0f32, 0.0f32);
            let mut prev_v = c[0];
            for &v in &c[1..] {
                if !v.is_nan() && !prev_v.is_nan() {
                    let di = v - mi;
                    let dj = prev_v - mj;
                    cov += di * dj;
                    vi += di * di;
                    vj += dj * dj;
                }
                prev_v = v;
            }
            if vi == 0.0 || vj == 0.0 {
                0.0
            } else {
                cov / (vi.sqrt() * vj.sqrt())
            }
        };
        out[off_ac1 + j] = ac1;
        out[off_ac1_abs + j] = ac1.abs();

        // ---------- trend（扫描紧凑 valid + idx1，零分配） ----------
        out[off_trend + j] = if n < 2 {
            0.0
        } else {
            let mx = sum_t_f32 / n as f32;
            let my = mean;
            let (mut cov, mut vx, mut vy) = (0.0f32, 0.0f32, 0.0f32);
            for k in 0..n {
                let dx = sc.idx1[k] as f32 - mx;
                let dy = sc.valid[k] - my;
                cov += dx * dy;
                vx += dx * dx;
                vy += dy * dy;
            }
            if vx == 0.0 || vy == 0.0 {
                0.0
            } else {
                cov / (vx.sqrt() * vy.sqrt())
            }
        };

        // ---------- curvature + quad_coef（共享 mean_t / mean_y 与二次矩 u² 融合扫描） ----------
        if with_curvature {
            if n < 3 {
                out[off_curv + j] = 0.0;
                out[off_quad + j] = 0.0;
            } else {
                let nf = n as f64;
                let mean_t = sum_t_f64 / nf;
                let mean_y = sum_y_f64 / nf;
                // 先一遍求 s_uu = Σ (t - mean_t)²，故 mean_q = s_uu / nf（与原实现逐位一致）
                let mut s_uu = 0.0f64;
                for &idx1 in &sc.idx1 {
                    let u = idx1 as f64 - mean_t;
                    s_uu += u * u;
                }
                let mean_q = s_uu / nf;
                let (mut cov_q, mut var_y, mut var_q) = (0.0f64, 0.0f64, 0.0f64);
                let (mut s_uuu, mut s_uuuu, mut s_uy, mut s_uu_y) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
                for k in 0..n {
                    let t = sc.idx1[k] as f64;
                    let y = sc.valid[k] as f64;
                    let u = t - mean_t;
                    let uu = u * u;
                    let dy = y - mean_y;
                    let dq = uu - mean_q;
                    cov_q += dy * dq;
                    var_y += dy * dy;
                    var_q += dq * dq;
                    if n >= 4 {
                        s_uuu += uu * u;
                        s_uuuu += uu * uu;
                        s_uy += u * y;
                        s_uu_y += uu * y;
                    }
                }
                out[off_curv + j] = if var_y == 0.0 || var_q == 0.0 {
                    0.0
                } else {
                    (cov_q / (var_y.sqrt() * var_q.sqrt())) as f32
                };
                out[off_quad + j] = if n < 4 || var_y <= 0.0 || s_uu <= 0.0 {
                    0.0
                } else {
                    let b1 = s_uy / s_uu;
                    let ss_res_lin = var_y - b1 * s_uy;
                    let m3 = [
                        [nf, 0.0, s_uu, nf * mean_y],
                        [0.0, s_uu, s_uuu, s_uy],
                        [s_uu, s_uuu, s_uuuu, s_uu_y],
                    ];
                    match solve3(m3) {
                        Some([c2, b2, a2]) => {
                            let ssr_quad =
                                c2 * (nf * mean_y) + b2 * s_uy + a2 * s_uu_y - nf * mean_y * mean_y;
                            let ss_res_quad = var_y - ssr_quad;
                            let r2_lin = 1.0 - ss_res_lin / var_y;
                            let r2_quad = 1.0 - ss_res_quad / var_y;
                            let delta = (r2_quad - r2_lin).clamp(0.0, 1.0);
                            let sign = if a2 >= 0.0 { 1.0 } else { -1.0 };
                            (sign * delta) as f32
                        }
                        None => 0.0,
                    }
                };
            }
        }

        // ---------- lz_complexity + binned_entropy + max_range_product ----------
        out[off_lz + j] = if all_finite {
            lz_from_finite_slice(
                &sc.valid,
                Some(&sc.sorted),
                &mut sc.lz_sorted,
                &mut sc.radix,
                &mut sc.discrete,
                &mut sc.sam_states,
            )
        } else {
            let finite_only: Vec<f32> = sc.valid.iter().copied().filter(|v| v.is_finite()).collect();
            lz_from_finite_slice(
                &finite_only,
                None,
                &mut sc.lz_sorted,
                &mut sc.radix,
                &mut sc.discrete,
                &mut sc.sam_states,
            )
        };
        out[off_ent + j] =
            binned_entropy_from_valid(&sc.valid, min_val, max_val, n_bins, &mut sc.ent_counts);
        out[off_maxr + j] = max_range_from_valid(&sc.valid, n_rows);
    }

    // ---------- 批量相关系数矩阵上三角 ----------
    if n_cols >= 2 {
        batch_corr_upper_col_major_f32(
            col_major,
            n_rows,
            n_cols,
            sc,
            &mut out[off_corr..off_corr + n_pairs],
        );
    }
}

/// 生成与 `extract_features_col_major_f32` 完全对齐的列名列表。
pub fn generate_feature_names(
    col_names: &[String],
    with_threshold_counts: bool,
    with_curvature: bool,
) -> Vec<String> {
    let n_cols = col_names.len();
    if n_cols == 0 {
        return Vec::new();
    }
    let total_len = features_output_len(n_cols, with_threshold_counts, with_curvature);
    let mut names = Vec::with_capacity(total_len);
    #[inline]
    fn push_s(names: &mut Vec<String>, col_names: &[String], suffix: &str) {
        for cn in col_names {
            names.push(format!("{}_{}", cn, suffix));
        }
    }
    for sfx in [
        "mean",
        "median",
        "std",
        "skew",
        "kurt",
        "p5",
        "p25",
        "p75",
        "p95",
        "iqr",
        "cv",
        "autocorr1",
        "autocorr1_abs",
        "trend",
    ] {
        push_s(&mut names, col_names, sfx);
    }
    if with_curvature {
        push_s(&mut names, col_names, "curvature");
        push_s(&mut names, col_names, "quad_coef");
    }
    push_s(&mut names, col_names, "period_diff");
    push_s(&mut names, col_names, "period_ratio");
    if with_threshold_counts {
        push_s(&mut names, col_names, "mean_above_p90");
        push_s(&mut names, col_names, "mean_below_p10");
    }
    if n_cols >= 2 {
        for i in 0..n_cols {
            for j in (i + 1)..n_cols {
                names.push(format!("{}_corr_{}", col_names[i], col_names[j]));
            }
        }
    }
    push_s(&mut names, col_names, "lz_complexity");
    push_s(&mut names, col_names, "entropy_1d");
    push_s(&mut names, col_names, "max_range_product");
    names
}
