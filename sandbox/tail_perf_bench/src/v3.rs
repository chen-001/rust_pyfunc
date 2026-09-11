//! V3 中性化：把「只随日期变化」的结构全部预计算 + 把每行计算压缩到活跃集上。
//!
//! ============================ 算法洞察（用 yupei_dist 全量数据实测） ============================
//!
//! 进入中性化的 slot 是 `rank_and_fill_missing_cross_sectional_median` 的产物：
//! 所有 `restrict==0`（可交易）位置都已被中位秩填满 → NaN 只可能出现在 `restrict!=0` 处。
//! 记
//!   P = {slot 有限}                      —— 生产 rank1 的域
//!   S = {restrict==0 且 ind1 有限}        —— 填充/restrict 之后仍有限的位置
//!   V = {restrict==0 且 10 风格全有限}     —— OLS 有效集（生产 per_date.valid_idx）
//!
//! 实测 (2818 日 × 7857 股)：|P|≈4135, |S|≈3651, |V|≈3624，V ⊆ S（|V\S| = 0），
//! S \ P 只在窗口面偶发（占 S 的 0.02%~0.34%），smooth_1 面恒为 0。
//!
//! 推论 1（跳过全部填充）：中性化内部的 3 级行业 OLS 填充 + 3 级中位填充只**写 NaN 位置**，
//! 而 S 上无 NaN、S 之外的位置随后被 `restrict!=0 → NaN` / `ind1 NaN → NaN` 抹掉
//! → 这些步骤（生产耗时 ~52%）对输出恒等，可整段跳过。
//!
//! 推论 2（只排一次序）：生产 rank2 是对 filled 再做一次 rank pct。rank_pct 是原值的
//! 严格单调函数（组内相等、组间不等），因此 **rank2 的序 = slot 值在 S 上的序**，
//! 与 rank1 的域 P 无关。于是 P 完全不需要：直接在 S 上排序即可，
//! 生产的第二次 8 趟 u64 基数排序（~19% 耗时）被彻底消除。
//!
//! 推论 3（域与因子无关）：S/V 只依赖 restrict + 行业 + 风格，与因子、面无关
//! → 每日结构（活跃索引、压缩风格矩阵、Cholesky）全局预计算一次，
//! 而不是每个 (因子, 面) 重算一遍（2761 × 8 = 22088 次）。
//!
//! 结果：每 (因子, 面, 日) 只剩
//!   ① gather |S|≈3651 个值（顺带做洞检测）
//!   ② 4 趟 u32 基数排序（|S| 而非 7857）
//!   ③ 一次组游走，直接产出 rank2 的 rank pct
//!   ④ |V|≈3624 的 OLS（与生产 ols_day_row_fast 完全同一算术 → 逐位一致）
//!
//! 与生产 v2 路径**逐位一致**；出现洞（S 中有 NaN）时逐行回退到生产行级实现。

use std::time::Instant;

use nalgebra::{Cholesky, DMatrix, Dyn};
use ndarray::{Array2, ArrayView2};

use crate::v2::{
    fill_ind_reg_row, median_fill_level_row, mono_key32, ols_day_core, ols_day_row_fast,
    radix_sort_order32, rank_pct_row_f64_in_place, rank_pct_row_from_f32_in, V2Shared,
};

pub struct V3Shared {
    pub v2: V2Shared,
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
        }
    }
}

pub fn v3_build(v2s: V2Shared) -> V3Shared {
    let (t, n) = v2s.base.industry.dim();
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
    let ind1 = &v2s.base.ind1;
    let restrict = &v2s.base.restrict_f64;
    for idx in 0..t {
        let base = idx * n;
        let mut s_pos = vec![u32::MAX; n];
        let mut ns = 0usize;
        for j in 0..n {
            if restrict[[idx, j]] == 0.0 && ind1[[idx, j]].is_finite() {
                s_idx_flat[base + ns] = j as u32;
                s_pos[j] = ns as u32;
                s_pos_flat[base + j] = ns as u32;
                in_s_flat[base + j] = 1;
                ns += 1;
            }
        }
        s_lens[idx] = ns as u32;
        // ---- D = S \ V ----
        let mut is_v = vec![false; n];
        for &j in v2s.per_date[idx].1.iter() {
            is_v[j as usize] = true;
        }
        let mut nd = 0usize;
        let mut nvv = 0usize;
        for k in 0..ns {
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
            let ord = v2s.orders[0].row(idx);
            let ord = ord.as_slice().unwrap();
            let mut lc = 0usize;
            let mut seg_start = 0usize;
            let mut nseg = 0usize;
            let mut prev_code = f64::NAN;
            for &jj in ord.iter() {
                let code = v2s.base.ind2[[idx, jj]];
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
        let vi = &v2s.per_date[idx].1;
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
        if is_identity && ns == vi.len() {
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
    V3Shared {
        v2: v2s,
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
    }
}

/// 生产行级回退：与 neutralize_std_slots_f32_v2_resid_batch 的单行完全一致。
fn v3_row_prod(
    row: &[f32],
    idx: usize,
    shared: &V3Shared,
    out_row: &mut [f32],
    sc: &mut V3Scratch,
) {
    let n = row.len();
    let base = &shared.v2.base;
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
    let o0v = shared.v2.orders[0].row(idx);
    let o1v = shared.v2.orders[1].row(idx);
    let o2v = shared.v2.orders[2].row(idx);
    let o3v = shared.v2.orders[3].row(idx);
    let o4v = shared.v2.orders[4].row(idx);
    let (o0, o1, o2) = (
        o0v.as_slice().unwrap(),
        o1v.as_slice().unwrap(),
        o2v.as_slice().unwrap(),
    );
    let (o3, o4) = (o3v.as_slice().unwrap(), o4v.as_slice().unwrap());

    let ind0_row: Vec<f64> = ind1_r
        .iter()
        .map(|&v| if v.is_nan() { 0.0 } else { 1.0 })
        .collect();

    sc.pct.clear();
    sc.pct.resize(n, f64::NAN);
    sc.keys.clear();
    sc.keys.resize(n, 0);
    rank_pct_row_from_f32_in(row, &mut sc.pct, &mut sc.idxs, &mut sc.tmp, &mut sc.keys);
    fill_ind_reg_row(
        &mut sc.pct,
        [ind2_r, ind1_r, &ind0_row],
        size_r,
        [o0, o1, o2],
        &mut sc.ys,
        &mut sc.bs,
        &mut sc.obs,
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
    );
    ols_day_row_fast(
        &sc.filled,
        &shared.v2,
        idx,
        &mut sc.y_buf_fb,
        &mut sc.xty,
        out_row,
    );
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
/// 得到与生产 rank2 完全相同的平均秩。任何前提不满足 → 返回 false 交回生产行级实现。
fn v3_row_x(
    row: &[f32],
    idx: usize,
    shared: &V3Shared,
    out_row: &mut [f32],
    sc: &mut V3Scratch,
    t: &mut V3Times,
) -> bool {
    let n = row.len();
    let ns = shared.s_lens[idx] as usize;
    let ns_f = ns as f64;
    let s_idx = &shared.s_idx_flat[idx * n..idx * n + ns];
    let size_row = shared.v2.base.size_ranked.row(idx);
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
        return true;
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
        if !crate::v2::has_ge_n_unique_pub(pct_row, 10) {
            t.fallback += st.elapsed().as_secs_f64();
            return false;
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
                return false;
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
                    let (c0, c1) = crate::v2::ols2_pub(&sc.ys, &sc.bs);
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
                                return false;
                            }
                            sc.sv.push(last_c0 + last_c1 * sz);
                        }
                    }
                }
                last_med = if sc.sv.is_empty() {
                    f64::NAN
                } else {
                    crate::v2::median_inplace_pub(&mut sc.sv)
                };
            }
            let sz = size_row[j];
            let val = if sz.is_finite() {
                if !last_ok {
                    t.fallback += st.elapsed().as_secs_f64();
                    return false;
                }
                last_c0 + last_c1 * sz
            } else {
                last_med
            };
            if val.is_nan() {
                t.fallback += st.elapsed().as_secs_f64();
                return false;
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
            return false;
        }
    }
    t.walk += st.elapsed().as_secs_f64();

    // ---- ⑤ OLS（V 顺序） ----
    let st = Instant::now();
    let off = shared.v_offsets[idx] as usize;
    let nvv = shared.v_lens[idx] as usize;
    let v_from_s = &shared.v_from_s_flat[off..off + nvv];
    sc.y_v.clear();
    for &p in v_from_s {
        sc.y_v.push(sc.ybuf[p as usize]);
    }
    ols_day_core(&sc.y_v, &shared.v2, idx, out_row);
    t.ols += st.elapsed().as_secs_f64();
    true
}

/// 近似路径（对齐方案 B 的排序语义）：域 = V 中 slot 有限的位置，完全不做填充复算。
/// 域小于 V 时用「X'X 减去被剔除行的贡献 + 重新 Cholesky」求解，避免生产的 SVD 回退。
fn v3_row_approx(
    row: &[f32],
    idx: usize,
    shared: &V3Shared,
    out_row: &mut [f32],
    sc: &mut V3Scratch,
    t: &mut V3Times,
) -> bool {
    use nalgebra::{Cholesky, DMatrix};
    let n = row.len();
    let nv = shared.v_lens[idx] as usize;
    if nv == 0 {
        return true;
    }
    let v_idx = &shared.v_idx_flat[idx * n..idx * n + nv];
    // ① gather：只取有限值，记住它在 V 顺序中的位置
    let st = Instant::now();
    sc.yv.clear();
    sc.vpos.clear();
    for (k, &j) in v_idx.iter().enumerate() {
        let val = row[j as usize];
        if val.is_finite() {
            sc.yv.push(val);
            sc.vpos.push(k as u32);
        }
    }
    let m = sc.yv.len();
    t.gather += st.elapsed().as_secs_f64();
    if m == 0 {
        return true;
    }
    // ② 排序 + 组游走（分母 = m）
    let st = Instant::now();
    sc.keys.clear();
    sc.keys.extend(sc.yv.iter().map(|&val| mono_key32(val)));
    sc.order.clear();
    sc.order.extend(0..m);
    radix_sort_order32(&sc.keys, &mut sc.order, &mut sc.tmp);
    t.sort += st.elapsed().as_secs_f64();
    let st = Instant::now();
    let m_f = m as f64;
    sc.ybuf.clear();
    sc.ybuf.resize(m, f64::NAN);
    let mut k = 0usize;
    while k < m {
        let val = sc.yv[sc.order[k]];
        let mut e = k + 1;
        while e < m && sc.yv[sc.order[e]] == val {
            e += 1;
        }
        let pct = (((k + 1) + e) as f64 / 2.0) / m_f;
        for t2 in k..e {
            sc.ybuf[sc.order[t2]] = pct;
        }
        k = e;
    }
    t.walk += st.elapsed().as_secs_f64();

    // ③ OLS
    let st = Instant::now();
    let kk = 10usize;
    let (p, valid_idx, valid_cols, xtx_full) = &shared.v2.per_date[idx];
    if *p == 0 {
        t.ols += st.elapsed().as_secs_f64();
        return true;
    }
    let xd = &shared.v2.xdays[idx];
    if m == valid_idx.len() {
        // 域未退化：直接用生产 OLS 核心（y 已在 V 顺序）
        ols_day_core(&sc.ybuf, &shared.v2, idx, out_row);
        t.ols += st.elapsed().as_secs_f64();
        return true;
    }
    let mut xty = vec![0.0f64; *p];
    for (i, &pos) in sc.vpos.iter().enumerate() {
        let pos = pos as usize;
        let yv = sc.ybuf[i];
        let xrow = &xd[pos * kk..pos * kk + kk];
        for c in 0..kk {
            xty[c] += xrow[c] * yv;
        }
        let ic = valid_cols[pos];
        if ic >= 0 {
            xty[kk + ic as usize] += yv;
        }
    }
    let mut xtx = xtx_full.clone();
    {
        let mut keep = vec![false; valid_idx.len()];
        for &pos in sc.vpos.iter() {
            keep[pos as usize] = true;
        }
        for pos in 0..valid_idx.len() {
            if keep[pos] {
                continue;
            }
            let xrow = &xd[pos * kk..pos * kk + kk];
            for c in 0..kk {
                let b = xrow[c];
                xtx[c * *p + c] -= b * b;
                for c2 in (c + 1)..kk {
                    let vv = b * xrow[c2];
                    xtx[c * *p + c2] -= vv;
                    xtx[c2 * *p + c] -= vv;
                }
            }
            let ic = valid_cols[pos];
            if ic >= 0 {
                let col = kk + ic as usize;
                xtx[col * *p + col] -= 1.0;
                for c in 0..kk {
                    let b = xrow[c];
                    xtx[c * *p + col] -= b;
                    xtx[col * *p + c] -= b;
                }
            }
        }
    }
    let chol = match Cholesky::new(DMatrix::from_row_slice(*p, *p, &xtx)) {
        Some(c) => c,
        None => {
            t.ols += st.elapsed().as_secs_f64();
            return false;
        }
    };
    let coef = chol.solve(&DMatrix::from_column_slice(*p, 1, &xty));
    for (i, &pos) in sc.vpos.iter().enumerate() {
        let pos = pos as usize;
        let j = valid_idx[pos] as usize;
        let yv = sc.ybuf[i];
        let xrow = &xd[pos * kk..pos * kk + kk];
        let mut pred = 0.0;
        for c in 0..kk {
            pred += coef[c] * xrow[c];
        }
        let ic = valid_cols[pos];
        if ic >= 0 {
            pred += coef[kk + ic as usize];
        }
        out_row[j] = (yv - pred) as f32;
    }
    t.ols += st.elapsed().as_secs_f64();
    true
}

/// 单行快路径。返回 false = 需要回退。
/// 纯快路径核心（S 上无特殊位置时使用）：与 v3_row_fast 完全同一实现。
#[inline]
fn v3_row_fast_core(
    row: &[f32],
    idx: usize,
    shared: &V3Shared,
    out_row: &mut [f32],
    sc: &mut V3Scratch,
    t: &mut V3Times,
) -> bool {
    v3_row_fast(row, idx, shared, out_row, sc, t)
}

#[inline]
fn v3_row_fast(
    row: &[f32],
    idx: usize,
    shared: &V3Shared,
    out_row: &mut [f32],
    sc: &mut V3Scratch,
    t: &mut V3Times,
) -> bool {
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
            return false;
        }
        sc.yv.push(v);
    }
    t.gather += s.elapsed().as_secs_f64();
    if ns == 0 {
        return true;
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

    // ---- ④ OLS：把 y 排到 V 顺序（S==V 时直接复用） ----
    let s = Instant::now();
    if shared.identity[idx] {
        ols_day_core(&sc.ybuf, &shared.v2, idx, out_row);
    } else {
        let off = shared.v_offsets[idx] as usize;
        let nv = shared.v_lens[idx] as usize;
        let v_from_s = &shared.v_from_s_flat[off..off + nv];
        sc.y_v.clear();
        for &p in v_from_s {
            sc.y_v.push(sc.ybuf[p as usize]);
        }
        ols_day_core(&sc.y_v, &shared.v2, idx, out_row);
    }
    t.ols += s.elapsed().as_secs_f64();
    true
}

/// 单面中性化（快路径 + 逐行回退），与生产 v2 opt2 逐位一致。
pub fn v3_slot(slot: ArrayView2<'_, f32>, shared: &V3Shared) -> (Array2<f32>, V3Times) {
    let (t, n) = slot.dim();
    let mut out = Array2::<f32>::from_elem((t, n), f32::NAN);
    let mut sc = V3Scratch::new(n);
    let mut times = V3Times::default();
    for idx in 0..t {
        let row_v = slot.row(idx);
        let row = row_v.as_slice().unwrap();
        let mut out_v = out.row_mut(idx);
        let out_row = out_v.as_slice_mut().unwrap();
        if !shared.fast_ok[idx] {
            let s = Instant::now();
            v3_row_prod(row, idx, shared, out_row, &mut sc);
            times.fallback += s.elapsed().as_secs_f64();
            times.slow_rows += 1;
            continue;
        }
        let s = Instant::now();
        let ok = if shared.d_lens[idx] == 0 {
            v3_row_fast(row, idx, shared, out_row, &mut sc, &mut times)
                || v3_row_x(row, idx, shared, out_row, &mut sc, &mut times)
        } else {
            v3_row_x(row, idx, shared, out_row, &mut sc, &mut times)
        };
        if !ok {
            times.fallback += s.elapsed().as_secs_f64();
            v3_row_prod(row, idx, shared, out_row, &mut sc);
            times.slow_rows += 1;
        } else {
            times.fast_rows += 1;
        }
    }
    (out, times)
}

/// 按日期块中性化：只处理第 [t0,t1) 行，语义与 `v3_slot` 完全一致（逐位）。
/// 供二档融合流水线使用（不物化整张 slot 矩阵）。
pub fn v3_slot_range(
    slot_block: ArrayView2<'_, f32>,
    shared: &V3Shared,
    t0: usize,
    t1: usize,
    sc: &mut V3Scratch,
) -> (Array2<f32>, V3Times) {
    let rows = t1 - t0;
    let n = slot_block.ncols();
    let mut out = Array2::<f32>::from_elem((rows, n), f32::NAN);
    let mut times = V3Times::default();
    for r in 0..rows {
        let idx = t0 + r;
        let row_v = slot_block.row(r);
        let row = row_v.as_slice().unwrap();
        let mut out_v = out.row_mut(r);
        let out_row = out_v.as_slice_mut().unwrap();
        if !shared.fast_ok[idx] {
            let s = Instant::now();
            v3_row_prod(row, idx, shared, out_row, sc);
            times.fallback += s.elapsed().as_secs_f64();
            times.slow_rows += 1;
            continue;
        }
        let s = Instant::now();
        let ok = if shared.d_lens[idx] == 0 {
            v3_row_fast(row, idx, shared, out_row, sc, &mut times)
                || v3_row_x(row, idx, shared, out_row, sc, &mut times)
        } else {
            v3_row_x(row, idx, shared, out_row, sc, &mut times)
        };
        if !ok {
            times.fallback += s.elapsed().as_secs_f64();
            v3_row_prod(row, idx, shared, out_row, sc);
            times.slow_rows += 1;
        } else {
            times.fast_rows += 1;
        }
    }
    (out, times)
}

/// 多线程吞吐基准：workers 线程按 batch 粒度抢任务。
pub fn v3_mt(shared: &V3Shared, slots: &[Array2<f32>], workers: usize, batch: usize) -> f64 {
    use std::sync::atomic::{AtomicUsize, Ordering};
    let counter = AtomicUsize::new(0);
    let t0 = Instant::now();
    std::thread::scope(|scope| {
        for _ in 0..workers {
            scope.spawn(|| loop {
                let i = counter.fetch_add(1, Ordering::Relaxed);
                let start = i * batch;
                if start >= slots.len() {
                    break;
                }
                let end = (start + batch).min(slots.len());
                for s in &slots[start..end] {
                    let _ = v3_slot(s.view(), shared);
                }
            });
        }
    });
    t0.elapsed().as_secs_f64()
}

pub fn fmt_v3(t: &V3Times) -> String {
    format!(
        "total={:.3}s [gather={:.3} sort={:.3} walk={:.3} ols={:.3} fallback={:.3}] fast_rows={} slow_rows={}",
        t.total(),
        t.gather,
        t.sort,
        t.walk,
        t.ols,
        t.fallback,
        t.fast_rows,
        t.slow_rows
    )
}

pub fn bitwise_equal(a: &Array2<f32>, b: &Array2<f32>) -> (bool, usize) {
    let mut mismatch = 0usize;
    for (x, y) in a.iter().zip(b.iter()) {
        if x.to_bits() != y.to_bits() && !(x.is_nan() && y.is_nan()) {
            mismatch += 1;
        }
    }
    (mismatch == 0, mismatch)
}

/// 诊断：复刻生产行级实现并返回 filled（restrict 之后、rank2 之前）的有限性模式。
pub fn prod_filled_domain(
    row: &[f32],
    idx: usize,
    shared: &V3Shared,
) -> (Vec<bool>, usize, usize) {
    let n = row.len();
    let mut sc = V3Scratch::new(n);
    let base = &shared.v2.base;
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
    let o0v = shared.v2.orders[0].row(idx);
    let o1v = shared.v2.orders[1].row(idx);
    let o2v = shared.v2.orders[2].row(idx);
    let o3v = shared.v2.orders[3].row(idx);
    let o4v = shared.v2.orders[4].row(idx);
    let (o0, o1, o2) = (
        o0v.as_slice().unwrap(),
        o1v.as_slice().unwrap(),
        o2v.as_slice().unwrap(),
    );
    let (o3, o4) = (o3v.as_slice().unwrap(), o4v.as_slice().unwrap());
    let ind0_row: Vec<f64> = ind1_r
        .iter()
        .map(|&v| if v.is_nan() { 0.0 } else { 1.0 })
        .collect();
    sc.pct.clear();
    sc.pct.resize(n, f64::NAN);
    sc.keys.clear();
    sc.keys.resize(n, 0);
    rank_pct_row_from_f32_in(row, &mut sc.pct, &mut sc.idxs, &mut sc.tmp, &mut sc.keys);
    fill_ind_reg_row(
        &mut sc.pct,
        [ind2_r, ind1_r, &ind0_row],
        size_r,
        [o0, o1, o2],
        &mut sc.ys,
        &mut sc.bs,
        &mut sc.obs,
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
    let dom: Vec<bool> = sc.filled.iter().map(|v| v.is_finite()).collect();
    let nfin = dom.iter().filter(|x| **x).count();
    // S 内但非有限的位置数
    let ns = shared.s_lens[idx] as usize;
    let s_idx = &shared.s_idx_flat[idx * n..idx * n + ns];
    let s_notfin = s_idx.iter().filter(|&&j| !dom[j as usize]).count();
    let s_sizefin = s_idx
        .iter()
        .filter(|&&j| size_r[j as usize].is_finite())
        .count();
    (dom, nfin, s_notfin.max(s_sizefin * 0))
}

/// 诊断：按日统计不一致格子数，并给出前 3 个不一致日的细节。
pub fn debug_mismatch(
    prod: &Array2<f32>,
    got: &Array2<f32>,
    shared: &V3Shared,
    tag: &str,
    slot: &Array2<f32>,
) {
    let (t, n) = prod.dim();
    let mut bad_days = 0usize;
    let mut bad_ident = 0usize;
    let mut shown = 0usize;
    let mut tot_bad = 0usize;
    for idx in 0..t {
        let mut c = 0usize;
        let mut first = None;
        for j in 0..n {
            let a = prod[[idx, j]];
            let b = got[[idx, j]];
            if a.to_bits() != b.to_bits() && !(a.is_nan() && b.is_nan()) {
                c += 1;
                if first.is_none() {
                    first = Some(j);
                }
            }
        }
        if c > 0 {
            bad_days += 1;
            tot_bad += c;
            if shared.identity[idx] {
                bad_ident += 1;
            }
            if shown < 3 {
                shown += 1;
                let j = first.unwrap();
                let row_v = slot.row(idx);
                let (dom, nfin, _) = prod_filled_domain(row_v.as_slice().unwrap(), idx, shared);
                let n = prod.ncols();
                let s_idx = &shared.s_idx_flat[idx * n..idx * n + shared.s_lens[idx] as usize];
                let s_notfin = s_idx.iter().filter(|&&jj| !dom[jj as usize]).count();
                println!(
                    "     -> 生产 filled 有限数={nfin}  |S|={}  S内非有限={s_notfin}",
                    shared.s_lens[idx]
                );
                println!(
                    "  [dbg {tag}] date#{idx} identity={} |S|={} |V|={} bad={c} first_j={j} prod={:?} v3={:?}",
                    shared.identity[idx],
                    shared.s_lens[idx],
                    shared.v_lens[idx],
                    prod[[idx, j]],
                    got[[idx, j]]
                );
            }
        }
    }
    println!("  [dbg {tag}] 不一致日={bad_days}/{t}（其中 S==V 日={bad_ident}）总不一致格={tot_bad}");
}

/// 批量中性化：外层日期、内层 slot —— 每日共享结构（s_idx/in_s/xdays/per_date/chols）
/// 只从内存层次里取一次，被 B 个面复用（生产 B=4；这里可放到 13）。
/// 逐行调用与单面路径**同一个** v3_row_fast → 逐位一致。
pub fn v3_slots_batch(slots: &[ArrayView2<f32>], shared: &V3Shared) -> (Vec<Array2<f32>>, V3Times) {
    let b = slots.len();
    let (t, n) = slots[0].dim();
    let mut outs: Vec<Array2<f32>> = (0..b)
        .map(|_| Array2::<f32>::from_elem((t, n), f32::NAN))
        .collect();
    let mut scs: Vec<V3Scratch> = (0..b).map(|_| V3Scratch::new(n)).collect();
    let mut times = V3Times::default();
    let mut rowbuf: Vec<&[f32]> = Vec::with_capacity(b);
    for idx in 0..t {
        rowbuf.clear();
        for s in slots.iter() {
            rowbuf.push(s.as_slice().unwrap()[idx * n..idx * n + n].as_ref());
        }
        for f in 0..b {
            let mut out_v = outs[f].row_mut(idx);
            let out_row = out_v.as_slice_mut().unwrap();
            let sc = &mut scs[f];
            if !shared.fast_ok[idx] {
                let st = Instant::now();
                v3_row_prod(rowbuf[f], idx, shared, out_row, sc);
                times.fallback += st.elapsed().as_secs_f64();
                times.slow_rows += 1;
                continue;
            }
            let st = Instant::now();
            let ok = if shared.d_lens[idx] == 0 {
                v3_row_fast(rowbuf[f], idx, shared, out_row, sc, &mut times)
                    || v3_row_x(rowbuf[f], idx, shared, out_row, sc, &mut times)
            } else {
                v3_row_x(rowbuf[f], idx, shared, out_row, sc, &mut times)
            };
            if !ok {
                times.fallback += st.elapsed().as_secs_f64();
                v3_row_prod(rowbuf[f], idx, shared, out_row, sc);
                times.slow_rows += 1;
            } else {
                times.fast_rows += 1;
            }
        }
    }
    (outs, times)
}

/// 近似批量：域用 V，不做特殊位置复算（对齐方案 B 的排序语义）。
pub fn v3_slots_batch_approx(
    slots: &[ArrayView2<f32>],
    shared: &V3Shared,
) -> (Vec<Array2<f32>>, V3Times) {
    let b = slots.len();
    let (t, n) = slots[0].dim();
    let mut outs: Vec<Array2<f32>> = (0..b)
        .map(|_| Array2::<f32>::from_elem((t, n), f32::NAN))
        .collect();
    let mut scs: Vec<V3Scratch> = (0..b).map(|_| V3Scratch::new(n)).collect();
    let mut times = V3Times::default();
    for idx in 0..t {
        let rows: Vec<&[f32]> = slots
            .iter()
            .map(|s| s.as_slice().unwrap()[idx * n..idx * n + n].as_ref())
            .collect();
        for f in 0..b {
            let mut out_v = outs[f].row_mut(idx);
            let out_row = out_v.as_slice_mut().unwrap();
            let sc = &mut scs[f];
            if !shared.fast_ok[idx] {
                let st = Instant::now();
                v3_row_prod(rows[f], idx, shared, out_row, sc);
                times.fallback += st.elapsed().as_secs_f64();
                times.slow_rows += 1;
                continue;
            }
            let st = Instant::now();
            if !v3_row_approx(rows[f], idx, shared, out_row, sc, &mut times) {
                times.fallback += st.elapsed().as_secs_f64();
                v3_row_prod(rows[f], idx, shared, out_row, sc);
                times.slow_rows += 1;
            } else {
                times.fast_rows += 1;
            }
        }
    }
    (outs, times)
}

/// MT 对照：同一 slot 池，v2 生产批量路径 (B=batch) vs v3 快路径，同线程数。
/// rounds = 每个 worker 把整个池子跑几遍（池子构建很贵，只建一次）。
pub fn run_mt_compare(data_dir: &str, workers: usize, rounds: usize, batch: usize) {
    let v2s = crate::v2::build_shared(data_dir);
    let t = Instant::now();
    let v3s = v3_build(v2s);
    println!("v3 build: {:.2}s", t.elapsed().as_secs_f64());
    let names = std::fs::read_to_string(format!("{data_dir}/sample_names.txt")).unwrap();
    let restrict_m =
        crate::npy::as_f32_mat(crate::npy::load(&format!("{data_dir}/restrict.npy")));
    let mut slots: Vec<Array2<f32>> = Vec::new();
    for nm in names.lines() {
        let raw =
            crate::npy::as_f32_mat(crate::npy::load(&format!("{data_dir}/factor_{nm}.npy")));
        let ranked =
            crate::engine::rank_and_fill_missing_cross_sectional_median(&raw, &restrict_m);
        slots.push(ranked.clone());
        for &w in &[5usize, 10, 20] {
            let (m, _x, _n, _s) = crate::engine::rolling_stats_f32_rowmajor(&ranked, w, w / 2);
            slots.push(m);
        }
    }
    let n = slots.len();
    let total = n * rounds;
    println!(
        "slot 池: {n} 个 × {rounds} 轮 = {total} 次 (每个 {:.1}M 元素)",
        (slots[0].len() as f64) / 1e6
    );

    {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Arc;
        let counter = Arc::new(AtomicUsize::new(0));
        let shared = Arc::new(&v3s.v2);
        let slots_r: Arc<Vec<&Array2<f32>>> = Arc::new(slots.iter().collect());
        let t0 = Instant::now();
        std::thread::scope(|scope| {
            for _ in 0..workers {
                let counter = counter.clone();
                let shared = shared.clone();
                let slots_r = slots_r.clone();
                let ntasks = (slots_r.len() + batch - 1) / batch;
                scope.spawn(move || loop {
                    let i = counter.fetch_add(1, Ordering::Relaxed);
                    if i >= ntasks * rounds {
                        break;
                    }
                    let s0 = (i % ntasks) * batch;
                    let s1 = (s0 + batch).min(slots_r.len());
                    let views: Vec<_> = slots_r[s0..s1].iter().map(|s| s.view()).collect();
                    let _ = crate::v2::v2_slot_batch(&views, &shared);
                });
            }
        });
        let dt = t0.elapsed().as_secs_f64();
        println!(
            "V2(batch B={batch}) workers={workers}: {dt:.2}s  {:.3}s/slot  {:.1} slot/s",
            dt / total as f64,
            total as f64 / dt
        );
    }

    {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let counter = AtomicUsize::new(0);
        let t0 = Instant::now();
        std::thread::scope(|scope| {
            for _ in 0..workers {
                scope.spawn(|| {
                let ntasks = (slots.len() + batch - 1) / batch;
                loop {
                    let i = counter.fetch_add(1, Ordering::Relaxed);
                    if i >= ntasks * rounds {
                        break;
                    }
                    let s0 = (i % ntasks) * batch;
                    let s1 = (s0 + batch).min(slots.len());
                    let views: Vec<_> = slots[s0..s1].iter().map(|s| s.view()).collect();
                    let _ = v3_slots_batch(&views, &v3s);
                }
                });
            }
        });
        let dt = t0.elapsed().as_secs_f64();
        println!(
            "V3(batch B={batch}) workers={workers}: {dt:.2}s  {:.3}s/slot  {:.1} slot/s",
            dt / total as f64,
            total as f64 / dt
        );
    }

    // V3 单面（无批）对照
    {
        let t0 = Instant::now();
        let dt = {
            use std::sync::atomic::{AtomicUsize, Ordering};
            let counter = AtomicUsize::new(0);
            std::thread::scope(|scope| {
                for _ in 0..workers {
                    scope.spawn(|| loop {
                        let i = counter.fetch_add(1, Ordering::Relaxed);
                        if i >= slots.len() * rounds {
                            break;
                        }
                        let _ = v3_slot(slots[i % slots.len()].view(), &v3s);
                    });
                }
            });
            t0.elapsed().as_secs_f64()
        };
        println!(
            "V3(单面) workers={workers}: {dt:.2}s  {:.3}s/slot  {:.1} slot/s",
            dt / total as f64,
            total as f64 / dt
        );
    }
}

/// 单线程 B 扫描：生产批量路径 (v2_slot_batch) vs v3 批量/单面，同一 13 面集合。
pub fn run_b_sweep(data_dir: &str) {
    let v2s = crate::v2::build_shared(data_dir);
    let v3s = v3_build(v2s);
    let names = std::fs::read_to_string(format!("{data_dir}/sample_names.txt")).unwrap();
    let restrict_m =
        crate::npy::as_f32_mat(crate::npy::load(&format!("{data_dir}/restrict.npy")));
    let mut all: Vec<(String, Vec<Array2<f32>>)> = Vec::new();
    for nm in names.lines() {
        let raw =
            crate::npy::as_f32_mat(crate::npy::load(&format!("{data_dir}/factor_{nm}.npy")));
        let ranked =
            crate::engine::rank_and_fill_missing_cross_sectional_median(&raw, &restrict_m);
        let mut slots = vec![ranked.clone()];
        for &w in &[5usize, 10, 20] {
            let (m, x, n2, s) = crate::engine::rolling_stats_f32_rowmajor(&ranked, w, w / 2);
            slots.push(m);
            slots.push(x);
            slots.push(n2);
            slots.push(s);
        }
        all.push((nm.to_string(), slots));
    }
    println!("因子数={} 每因子 13 面", all.len());
    // 生产批量路径 B 扫描（单线程，13 面整体）
    for &b in &[1usize, 2, 4, 13] {
        let t0 = Instant::now();
        let mut tot = 0.0f64;
        for (_, slots) in all.iter() {
            for chunk in slots.chunks(b) {
                let views: Vec<_> = chunk.iter().map(|s| s.view()).collect();
                let _ = crate::v2::v2_slot_batch(&views, &v3s.v2);
                tot += 1.0;
            }
        }
        let dt = t0.elapsed().as_secs_f64();
        println!(
            "V2 生产批量 B={b:2}: {dt:.2}s  {:.3}s/因子  {:.1} 面/s",
            dt / all.len() as f64,
            tot / dt
        );
    }
    // v3
    for &b in &[1usize, 4, 13] {
        let t0 = Instant::now();
        for (_, slots) in all.iter() {
            for chunk in slots.chunks(b) {
                let views: Vec<_> = chunk.iter().map(|s| s.view()).collect();
                let _ = v3_slots_batch(&views, &v3s);
            }
        }
        let dt = t0.elapsed().as_secs_f64();
        println!("V3 批量 B={b:2}: {dt:.2}s  {:.3}s/因子", dt / all.len() as f64);
    }
    let t0 = Instant::now();
    for (_, slots) in all.iter() {
        for s in slots.iter() {
            let _ = v3_slot(s.view(), &v3s);
        }
    }
    let dt = t0.elapsed().as_secs_f64();
    println!("V3 单面   : {dt:.2}s  {:.3}s/因子", dt / all.len() as f64);
    // v3 近似（域 = V，跳过特殊位置复算；对齐方案 B 的排序语义）
    for &b in &[4usize, 13] {
        let t0 = Instant::now();
        for (_, slots) in all.iter() {
            for chunk in slots.chunks(b) {
                let views: Vec<_> = chunk.iter().map(|s| s.view()).collect();
                let _ = v3_slots_batch_approx(&views, &v3s);
            }
        }
        let dt = t0.elapsed().as_secs_f64();
        println!("V3 近似 B={b:2}: {dt:.2}s  {:.3}s/因子", dt / all.len() as f64);
    }
}

/// 近似档位的 IC 精度对照：baseline(生产 B=4) vs v3 近似，跑 ic_only 回测算 IC_mean。
pub fn run_ic_err(data_dir: &str, n_factors: usize) {
    let v2s = crate::v2::build_shared(data_dir);
    let v3s = v3_build(v2s);
    let dates = crate::npy::as_i32_vec1(crate::npy::load(&format!("{data_dir}/dates.npy")));
    let ret_g1 = crate::npy::as_f32_mat(crate::npy::load(&format!("{data_dir}/ret_gap1.npy")));
    let ret_s1 = crate::npy::as_f32_mat(crate::npy::load(&format!("{data_dir}/ret_sum_gap1.npy")));
    let ret_g5 = crate::npy::as_f32_mat(crate::npy::load(&format!("{data_dir}/ret_gap5.npy")));
    let ret_s5 = crate::npy::as_f32_mat(crate::npy::load(&format!("{data_dir}/ret_sum_gap5.npy")));
    let restrict = crate::npy::as_f32_mat(crate::npy::load(&format!("{data_dir}/restrict.npy")));
    let index_ret =
        crate::npy::as_f32_vec1(crate::npy::load(&format!("{data_dir}/index_ret.npy")));
    let pre = crate::btopt::build_bt_precomputed(&ret_s1, &ret_s5);
    let open_counts: Vec<usize> = (0..restrict.nrows())
        .map(|r| {
            restrict
                .row(r)
                .iter()
                .filter(|&&v| v.is_finite() && v == 0.0)
                .count()
        })
        .collect();
    let backtest_start = 20150201i32;
    let names = std::fs::read_to_string(format!("{data_dir}/sample_names.txt")).unwrap();
    let mut worst_g1 = 0.0f64;
    let mut worst_g5 = 0.0f64;
    let mut min_corr = 1.0f64;
    let mut n_cmp = 0usize;
    for nm in names.lines().take(n_factors) {
        let raw =
            crate::npy::as_f32_mat(crate::npy::load(&format!("{data_dir}/factor_{nm}.npy")));
        let ranked =
            crate::engine::rank_and_fill_missing_cross_sectional_median(&raw, &restrict);
        let mut slots = vec![ranked.clone()];
        for &w in &[5usize, 10, 20] {
            let (m, x, n2, s) = crate::engine::rolling_stats_f32_rowmajor(&ranked, w, w / 2);
            slots.push(m);
            slots.push(x);
            slots.push(n2);
            slots.push(s);
        }
        // baseline（生产 B=4）
        let mut base_out: Vec<Array2<f32>> = Vec::new();
        for chunk in slots.chunks(4) {
            let views: Vec<_> = chunk.iter().map(|s| s.view()).collect();
            base_out.extend(crate::v2::v2_slot_batch(&views, &v3s.v2).0);
        }
        // v3 近似
        let views: Vec<_> = slots.iter().map(|s| s.view()).collect();
        let approx_out = v3_slots_batch_approx(&views, &v3s).0;
        for (b, a) in base_out.iter().zip(approx_out.iter()) {
            for (slot, bt) in [(b, &pre)] {
                let _ = bt;
                let (r1b, r5b) = crate::btopt::bt_gap1_gap5_prod(
                    slot.view(),
                    ret_g1.view(),
                    ret_s1.view(),
                    ret_g5.view(),
                    ret_s5.view(),
                    restrict.view(),
                    ndarray::ArrayView1::from(&index_ret),
                    &dates,
                    backtest_start,
                    10,
                    &open_counts,
                    true,
                    &pre,
                );
                let (r1a, r5a) = crate::btopt::bt_gap1_gap5_prod(
                    a.view(),
                    ret_g1.view(),
                    ret_s1.view(),
                    ret_g5.view(),
                    ret_s5.view(),
                    restrict.view(),
                    ndarray::ArrayView1::from(&index_ret),
                    &dates,
                    backtest_start,
                    10,
                    &open_counts,
                    true,
                    &pre,
                );
                let d1 = (r1b.summary[0] - r1a.summary[0]).abs();
                let d5 = (r5b.summary[0] - r5a.summary[0]).abs();
                if d1.is_finite() && d1 > worst_g1 {
                    worst_g1 = d1;
                }
                if d5.is_finite() && d5 > worst_g5 {
                    worst_g5 = d5;
                }
                // 日 IC 序列相关
                let n = r1b.ic_values.len().min(r1a.ic_values.len());
                if n > 10 {
                    let (mut sx, mut sy, mut sxx, mut syy, mut sxy) = (0.0, 0.0, 0.0, 0.0, 0.0);
                    let mut cnt = 0.0;
                    for i in 0..n {
                        let x = r1b.ic_values[i] as f64;
                        let y = r1a.ic_values[i] as f64;
                        if x.is_finite() && y.is_finite() {
                            sx += x; sy += y; sxx += x * x; syy += y * y; sxy += x * y;
                            cnt += 1.0;
                        }
                    }
                    let cov = sxy / cnt - (sx / cnt) * (sy / cnt);
                    let vx = sxx / cnt - (sx / cnt) * (sx / cnt);
                    let vy = syy / cnt - (sy / cnt) * (sy / cnt);
                    if vx > 0.0 && vy > 0.0 {
                        let c = cov / (vx.sqrt() * vy.sqrt());
                        if c < min_corr {
                            min_corr = c;
                        }
                    }
                }
                n_cmp += 1;
            }
        }
    }
    println!(
        "IC 精度（{n_cmp} 个 (因子,面) 对比）: gap1 |ΔIC_mean| 最大={worst_g1:.3e}  gap5 最大={worst_g5:.3e}  日 IC 序列最小相关={min_corr:.8}"
    );
    println!("（阈值参考：gap5 ic_point_neu=0.01，gap1 ic_point_neu=0.006）");
}

// ============================================================================
// v3 近似 + IC 融合：算完当天残差立刻过滤 + 排序 + 累加 Σd²，不物化 (T,N) 残差矩阵
// ============================================================================

pub struct IcCtx {
    pub ret_g1: Array2<f32>,
    pub ret_s1: Array2<f32>,
    pub ret_g5: Array2<f32>,
    pub ret_s5: Array2<f32>,
    pub restrict: Array2<f32>,
    pub index_ret: Vec<f32>,
    pub dates: Vec<i32>,
    pub backtest_start: i32,
    pub pre: crate::btopt::BtPrecomputed,
    pub open_counts: Vec<usize>,
}

struct GapAcc {
    gap: usize,
    ratio_values: Vec<f64>,
    ic_dates: Vec<i32>,
    ic_values_f64: Vec<f64>,
    ic_values_f32: Vec<f32>,
    held_row: usize,
    held_idx: Vec<u32>,
    held_val: Vec<f32>,
    held_set: bool,
    gen: Vec<u32>,
    stamp: Vec<u32>,
    walk_buf: Vec<i64>,
    gen_id: u32,
    filt_sig: Vec<f32>,
    filt_ret: Vec<f32>,
    filt_stk: Vec<u32>,
    // IC 只需 ordinal 秩；用可复用缓冲避免 rank_both_radix 的 5 次堆分配 + 无用的 avg 组游走
    rk_keys: Vec<u32>,
    rk_order: Vec<usize>,
    rk_tmp: Vec<usize>,
    rk_ordinal: Vec<i64>,
}

impl GapAcc {
    fn new(gap: usize, n_stocks: usize, date_size: usize) -> Self {
        GapAcc {
            gap,
            ratio_values: vec![f64::NAN; date_size],
            ic_dates: Vec::new(),
            ic_values_f64: Vec::new(),
            ic_values_f32: Vec::new(),
            held_row: 0,
            held_idx: Vec::new(),
            held_val: Vec::new(),
            held_set: false,
            gen: vec![0u32; n_stocks],
            stamp: vec![0u32; n_stocks],
            walk_buf: Vec::with_capacity(n_stocks),
            gen_id: 0,
            filt_sig: Vec::with_capacity(n_stocks),
            filt_ret: Vec::with_capacity(n_stocks),
            filt_stk: Vec::with_capacity(n_stocks),
            rk_keys: Vec::with_capacity(n_stocks),
            rk_order: Vec::with_capacity(n_stocks),
            rk_tmp: Vec::with_capacity(n_stocks),
            rk_ordinal: Vec::with_capacity(n_stocks),
        }
    }
}

/// 单 slot 的「近似中性化 + IC」融合：逐日算残差（紧凑形式）→ 立刻过滤/排序/累加 IC。
/// 不物化 (T,N) 残差矩阵，IC 过滤在 |V'| 上做而不是全宽 N。
pub fn v3_slot_ic_approx(
    slot: ArrayView2<'_, f32>,
    shared: &V3Shared,
    ctx: &IcCtx,
) -> (crate::btopt::LegacyBacktestResult, crate::btopt::LegacyBacktestResult, V3Times) {
    let (t, n) = slot.dim();
    let mut times = V3Times::default();
    if t < 2 {
        return (
            crate::btopt::default_result_pub(),
            crate::btopt::default_result_pub(),
            times,
        );
    }
    let mut sc = V3Scratch::new(n);
    let mut any_finite = vec![false; t];
    let mut seen: std::collections::HashSet<u32> = std::collections::HashSet::new();
    let mut enough_unique = false;
    let mut eff: Vec<usize> = Vec::with_capacity(t);
    let mut prev_idx: Vec<u32> = Vec::new();
    let mut prev_val: Vec<f32> = Vec::new();
    let mut cur_idx: Vec<u32> = Vec::new();
    let mut cur_val: Vec<f32> = Vec::new();
    let mut g1 = GapAcc::new(1, n, t);
    let mut g5 = GapAcc::new(5, n, t);
    let mut g1_done = false;
    let mut g5_done = false;

    for idx in 0..t {
        let row_v = slot.row(idx);
        let row = row_v.as_slice().unwrap();
        let ok = v3_row_approx_compact(row, idx, shared, &mut sc, &mut times, &mut cur_idx, &mut cur_val);
        if !ok {
            // 奇异等极端情况：本 slot 交回调用方（返回空结果标记）
            times.slow_rows += 1;
            return (
                crate::btopt::default_result_pub(),
                crate::btopt::default_result_pub(),
                times,
            );
        }
        any_finite[idx] = !cur_val.is_empty();
        if !enough_unique && idx + 2 < t {
            for &v in cur_val.iter() {
                if seen.insert(v.to_bits()) && seen.len() >= 10 {
                    enough_unique = true;
                    break;
                }
            }
        }
        // 该日是否为 effective（dates[idx] > backtest_start 且上一日残差行非全 NaN）
        if idx >= 1 && ctx.dates[idx] > ctx.backtest_start && any_finite[idx - 1] {
            let local_t = eff.len();
            eff.push(idx);
            let held_row = idx - 1;
            if !g1_done {
                // gap=1：held 恒为前一日 → 直接用 prev，零拷贝
                g1.held_set = true;
                g1.held_row = held_row;
                g1_done = gap_step(&mut g1, ctx, local_t, idx, &prev_idx, &prev_val, held_row);
            }
            if !g5_done {
                if local_t % g5.gap == 0 {
                    g5.held_row = held_row;
                    g5.held_idx.clear();
                    g5.held_idx.extend_from_slice(&prev_idx);
                    g5.held_val.clear();
                    g5.held_val.extend_from_slice(&prev_val);
                    g5.held_set = true;
                }
                let hr = g5.held_row;
                let hi = std::mem::take(&mut g5.held_idx);
                let hv = std::mem::take(&mut g5.held_val);
                g5_done = gap_step(&mut g5, ctx, local_t, idx, &hi, &hv, hr);
                g5.held_idx = hi;
                g5.held_val = hv;
            }
        }
        // 轮换缓冲：cur -> prev，cur 复用
        std::mem::swap(&mut prev_idx, &mut cur_idx);
        std::mem::swap(&mut prev_val, &mut cur_val);
    }

    if !enough_unique {
        return (
            crate::btopt::default_result_pub(),
            crate::btopt::default_result_pub(),
            times,
        );
    }
    let r1 = crate::btopt::finish_result_pub(
        g1.ratio_values,
        g1.ic_dates,
        g1.ic_values_f64,
        g1.ic_values_f32,
        &eff,
        1,
    );
    let r5 = crate::btopt::finish_result_pub(
        g5.ratio_values,
        g5.ic_dates,
        g5.ic_values_f64,
        g5.ic_values_f32,
        &eff,
        5,
    );
    (r1, r5, times)
}

/// 单日 IC 步骤（照抄 bt_single_gap 的循环体）。返回 true 表示该 gap 已提前结束。
fn gap_step(
    acc: &mut GapAcc,
    ctx: &IcCtx,
    local_t: usize,
    raw_eff_idx: usize,
    held_idx: &[u32],
    held_val: &[f32],
    held_row: usize,
) -> bool {
    if !acc.held_set {
        return false;
    }
    let n_stocks = ctx.restrict.ncols();
    acc.filt_sig.clear();
    acc.filt_ret.clear();
    acc.filt_stk.clear();
    let ret_row = ctx.ret_g1.row(raw_eff_idx); // 占位，实际按 gap 选
    let _ = ret_row;
    let ret = if acc.gap == 1 { &ctx.ret_g1 } else { &ctx.ret_g5 };
    let rrow = ret.row(raw_eff_idx);
    let srow = ctx.restrict.row(held_row);
    for k in 0..held_idx.len() {
        let j = held_idx[k] as usize;
        let sig = held_val[k];
        let rv = rrow[j];
        let is_open = srow[j].is_finite() && srow[j] == 0.0;
        if sig.is_finite() && rv.is_finite() && is_open {
            acc.filt_sig.push(sig);
            acc.filt_ret.push(rv);
            acc.filt_stk.push(j as u32);
        }
    }
    if (local_t + 1) % acc.gap == 0 {
        acc.gen_id += 1;
        let orders = if acc.gap == 1 {
            &ctx.pre.orders_g1
        } else {
            &ctx.pre.orders_g5
        };
        let order = &orders[raw_eff_idx];
        for (pos, &stk) in acc.filt_stk.iter().enumerate() {
            acc.gen[stk as usize] = acc.gen_id;
            acc.stamp[stk as usize] = (pos + 1) as u32;
        }
        acc.walk_buf.clear();
        acc.walk_buf.resize(acc.filt_stk.len(), 0);
        let mut counter = 0usize;
        for &stk in order {
            if acc.gen[stk as usize] == acc.gen_id {
                acc.walk_buf[acc.stamp[stk as usize] as usize - 1] = counter as i64;
                counter += 1;
            }
        }
        // ordinal 秩（与 rank_both_radix 的第一半逐位一致，复用缓冲、不算 avg）
        {
            let m = acc.filt_sig.len();
            acc.rk_keys.clear();
            acc.rk_keys
                .extend(acc.filt_sig.iter().map(|&v| crate::v2::mono_key32(v)));
            acc.rk_order.clear();
            acc.rk_order.extend(0..m);
            crate::v2::radix_sort_order32(&acc.rk_keys, &mut acc.rk_order, &mut acc.rk_tmp);
            acc.rk_ordinal.clear();
            acc.rk_ordinal.resize(m, 0);
            for (rank, &idx) in acc.rk_order.iter().enumerate() {
                acc.rk_ordinal[idx] = rank as i64;
            }
        }
        let xx = &acc.rk_ordinal;
        let nf = acc.filt_sig.len() as f64;
        let mut diff_sq_sum = 0.0f64;
        for i in 0..acc.filt_sig.len() {
            let diff = acc.walk_buf[i] - xx[i];
            diff_sq_sum += (diff * diff) as f64;
        }
        let ic_value = if nf < 2.0 {
            f64::NAN
        } else {
            1.0 - 6.0 * diff_sq_sum / (nf * (nf * nf - 1.0))
        };
        acc.ic_dates.push(ctx.dates[raw_eff_idx]);
        acc.ic_values_f64.push(ic_value);
        acc.ic_values_f32.push(ic_value as f32);
    }
    let stocks_num = acc.filt_sig.len();
    if stocks_num < 10 {
        return false;
    }
    let valid_symbol_num = ctx.open_counts.get(raw_eff_idx - 1).copied().unwrap_or(0);
    if valid_symbol_num > 0 {
        acc.ratio_values[local_t] = stocks_num as f64 / valid_symbol_num as f64;
    }
    let _ = n_stocks;
    false
}

/// 逐日近似残差（紧凑形式）：把残差写进 res_idx/res_val（按股票下标升序）。
fn v3_row_approx_compact(
    row: &[f32],
    idx: usize,
    shared: &V3Shared,
    sc: &mut V3Scratch,
    t: &mut V3Times,
    res_idx: &mut Vec<u32>,
    res_val: &mut Vec<f32>,
) -> bool {
    use nalgebra::{Cholesky, DMatrix};
    res_idx.clear();
    res_val.clear();
    let n = row.len();
    let nv = shared.v_lens[idx] as usize;
    if nv == 0 {
        return true;
    }
    let v_idx = &shared.v_idx_flat[idx * n..idx * n + nv];
    let st = Instant::now();
    sc.yv.clear();
    sc.vpos.clear();
    for (k, &j) in v_idx.iter().enumerate() {
        let val = row[j as usize];
        if val.is_finite() {
            sc.yv.push(val);
            sc.vpos.push(k as u32);
        }
    }
    let m = sc.yv.len();
    t.gather += st.elapsed().as_secs_f64();
    if m == 0 {
        return true;
    }
    let st = Instant::now();
    sc.keys.clear();
    sc.keys.extend(sc.yv.iter().map(|&val| mono_key32(val)));
    sc.order.clear();
    sc.order.extend(0..m);
    radix_sort_order32(&sc.keys, &mut sc.order, &mut sc.tmp);
    t.sort += st.elapsed().as_secs_f64();
    let st = Instant::now();
    let m_f = m as f64;
    sc.ybuf.clear();
    sc.ybuf.resize(m, f64::NAN);
    let mut k = 0usize;
    while k < m {
        let val = sc.yv[sc.order[k]];
        let mut e = k + 1;
        while e < m && sc.yv[sc.order[e]] == val {
            e += 1;
        }
        let pct = (((k + 1) + e) as f64 / 2.0) / m_f;
        for t2 in k..e {
            sc.ybuf[sc.order[t2]] = pct;
        }
        k = e;
    }
    t.walk += st.elapsed().as_secs_f64();

    let st = Instant::now();
    let kk = 10usize;
    let (p, valid_idx, valid_cols, xtx_full) = &shared.v2.per_date[idx];
    if *p == 0 {
        t.ols += st.elapsed().as_secs_f64();
        return true;
    }
    let xd = &shared.v2.xdays[idx];
    // 结果写进紧凑缓冲
    let write_out = |sc: &mut V3Scratch,
                     coef: Option<&DMatrix<f64>>,
                     res_idx: &mut Vec<u32>,
                     res_val: &mut Vec<f32>| {
        for (i, &pos) in sc.vpos.iter().enumerate() {
            let pos = pos as usize;
            let j = valid_idx[pos] as usize;
            let yv = sc.ybuf[i];
            let v = match coef {
                None => (yv - 0.0) as f32,
                Some(c) => {
                    let xrow = &xd[pos * kk..pos * kk + kk];
                    let mut pred = 0.0;
                    for c2 in 0..kk {
                        pred += c[c2] * xrow[c2];
                    }
                    let ic = valid_cols[pos];
                    if ic >= 0 {
                        pred += c[kk + ic as usize];
                    }
                    (yv - pred) as f32
                }
            };
            res_idx.push(j as u32);
            res_val.push(v);
        }
    };
    if m == valid_idx.len() {
        // 域未退化：走生产 OLS 核心，但结果要落进紧凑缓冲 → 复用其系数计算
        let mut xty = vec![0.0f64; *p];
        for (i, &pos) in sc.vpos.iter().enumerate() {
            let pos = pos as usize;
            let yv = sc.ybuf[i];
            let xrow = &xd[pos * kk..pos * kk + kk];
            for c in 0..kk {
                xty[c] += xrow[c] * yv;
            }
            let ic = valid_cols[pos];
            if ic >= 0 {
                xty[kk + ic as usize] += yv;
            }
        }
        let mut mn = f64::INFINITY;
        let mut mx = f64::NEG_INFINITY;
        for &y in sc.ybuf.iter() {
            if y < mn {
                mn = y;
            }
            if y > mx {
                mx = y;
            }
        }
        if mn == mx {
            for &pos in sc.vpos.iter() {
                res_idx.push(valid_idx[pos as usize]);
                res_val.push(0.5);
            }
            t.ols += st.elapsed().as_secs_f64();
            return true;
        }
        let chol = shared.v2.chols[idx].as_ref().expect("fast path 需 Cholesky");
        let coef = chol.solve(&DMatrix::from_column_slice(*p, 1, &xty));
        write_out(sc, Some(&coef), res_idx, res_val);
        t.ols += st.elapsed().as_secs_f64();
        return true;
    }
    // 域退化：X'X 减去被剔除行 + 重新 Cholesky
    let mut xty = vec![0.0f64; *p];
    for (i, &pos) in sc.vpos.iter().enumerate() {
        let pos = pos as usize;
        let yv = sc.ybuf[i];
        let xrow = &xd[pos * kk..pos * kk + kk];
        for c in 0..kk {
            xty[c] += xrow[c] * yv;
        }
        let ic = valid_cols[pos];
        if ic >= 0 {
            xty[kk + ic as usize] += yv;
        }
    }
    let mut xtx = xtx_full.clone();
    {
        let mut keep = vec![false; valid_idx.len()];
        for &pos in sc.vpos.iter() {
            keep[pos as usize] = true;
        }
        for pos in 0..valid_idx.len() {
            if keep[pos] {
                continue;
            }
            let xrow = &xd[pos * kk..pos * kk + kk];
            for c in 0..kk {
                let b = xrow[c];
                xtx[c * *p + c] -= b * b;
                for c2 in (c + 1)..kk {
                    let vv = b * xrow[c2];
                    xtx[c * *p + c2] -= vv;
                    xtx[c2 * *p + c] -= vv;
                }
            }
            let ic = valid_cols[pos];
            if ic >= 0 {
                let col = kk + ic as usize;
                xtx[col * *p + col] -= 1.0;
                for c in 0..kk {
                    let b = xrow[c];
                    xtx[c * *p + col] -= b;
                    xtx[col * *p + c] -= b;
                }
            }
        }
    }
    let chol = match Cholesky::new(DMatrix::from_row_slice(*p, *p, &xtx)) {
        Some(c) => c,
        None => {
            t.ols += st.elapsed().as_secs_f64();
            return false;
        }
    };
    let coef = chol.solve(&DMatrix::from_column_slice(*p, 1, &xty));
    write_out(sc, Some(&coef), res_idx, res_val);
    t.ols += st.elapsed().as_secs_f64();
    true
}

/// 多 slot 融合跑：返回每 slot 的 (gap1, gap5) IC 结果与总计时。
pub fn v3_slots_ic_approx(
    slots: &[ArrayView2<'_, f32>],
    shared: &V3Shared,
    ctx: &IcCtx,
) -> (Vec<(crate::btopt::LegacyBacktestResult, crate::btopt::LegacyBacktestResult)>, V3Times) {
    let mut outs = Vec::with_capacity(slots.len());
    let mut tt = V3Times::default();
    for s in slots {
        let (r1, r5, t) = v3_slot_ic_approx(s.clone(), shared, ctx);
        tt.gather += t.gather;
        tt.sort += t.sort;
        tt.walk += t.walk;
        tt.ols += t.ols;
        tt.fallback += t.fallback;
        tt.fast_rows += t.fast_rows;
        tt.slow_rows += t.slow_rows;
        outs.push((r1, r5));
    }
    (outs, tt)
}

/// 融合对照：baseline(生产 B=4 中性化 + 独立 IC 回测) vs v3 近似+IC 融合。
pub fn run_ic_fused(data_dir: &str, n_factors: usize) {
    let v2s = crate::v2::build_shared(data_dir);
    let v3s = v3_build(v2s);
    let dates = crate::npy::as_i32_vec1(crate::npy::load(&format!("{data_dir}/dates.npy")));
    let ret_g1 = crate::npy::as_f32_mat(crate::npy::load(&format!("{data_dir}/ret_gap1.npy")));
    let ret_s1 = crate::npy::as_f32_mat(crate::npy::load(&format!("{data_dir}/ret_sum_gap1.npy")));
    let ret_g5 = crate::npy::as_f32_mat(crate::npy::load(&format!("{data_dir}/ret_gap5.npy")));
    let ret_s5 = crate::npy::as_f32_mat(crate::npy::load(&format!("{data_dir}/ret_sum_gap5.npy")));
    let restrict = crate::npy::as_f32_mat(crate::npy::load(&format!("{data_dir}/restrict.npy")));
    let index_ret = crate::npy::as_f32_vec1(crate::npy::load(&format!("{data_dir}/index_ret.npy")));
    let pre = crate::btopt::build_bt_precomputed(&ret_s1, &ret_s5);
    let open_counts: Vec<usize> = (0..restrict.nrows())
        .map(|r| {
            restrict
                .row(r)
                .iter()
                .filter(|&&v| v.is_finite() && v == 0.0)
                .count()
        })
        .collect();
    let ctx = IcCtx {
        ret_g1,
        ret_s1,
        ret_g5,
        ret_s5,
        restrict,
        index_ret,
        dates: dates.clone(),
        backtest_start: 20150201,
        pre,
        open_counts,
    };
    let names = std::fs::read_to_string(format!("{data_dir}/sample_names.txt")).unwrap();
    let mut t_base_neu = 0.0f64;
    let mut t_base_ic = 0.0f64;
    let mut t_fused = 0.0f64;
    let mut t_approx_only = 0.0f64;
    let mut worst_g1 = 0.0f64;
    let mut worst_g5 = 0.0f64;
    let mut n_cmp = 0usize;
    for nm in names.lines().take(n_factors) {
        let raw = crate::npy::as_f32_mat(crate::npy::load(&format!("{data_dir}/factor_{nm}.npy")));
        let restrict_m = crate::npy::as_f32_mat(crate::npy::load(&format!("{data_dir}/restrict.npy")));
        let ranked =
            crate::engine::rank_and_fill_missing_cross_sectional_median(&raw, &restrict_m);
        let mut slots = vec![ranked.clone()];
        for &w in &[5usize, 10, 20] {
            let (m, x, n2, s) = crate::engine::rolling_stats_f32_rowmajor(&ranked, w, w / 2);
            slots.push(m);
            slots.push(x);
            slots.push(n2);
            slots.push(s);
        }
        // ---- baseline：中性化（生产 B=4）----
        let t0 = Instant::now();
        let mut base_out: Vec<Array2<f32>> = Vec::new();
        for chunk in slots.chunks(4) {
            let views: Vec<_> = chunk.iter().map(|s| s.view()).collect();
            base_out.extend(crate::v2::v2_slot_batch(&views, &v3s.v2).0);
        }
        t_base_neu += t0.elapsed().as_secs_f64();
        // ---- baseline：独立 IC 回测 ----
        let t0 = Instant::now();
        let mut base_ic: Vec<(crate::btopt::LegacyBacktestResult, crate::btopt::LegacyBacktestResult)> =
            Vec::new();
        for s in base_out.iter() {
            base_ic.push(crate::btopt::bt_gap1_gap5_prod(
                s.view(),
                ctx.ret_g1.view(),
                ctx.ret_s1.view(),
                ctx.ret_g5.view(),
                ctx.ret_s5.view(),
                ctx.restrict.view(),
                ndarray::ArrayView1::from(&ctx.index_ret),
                &ctx.dates,
                ctx.backtest_start,
                10,
                &ctx.open_counts,
                true,
                &ctx.pre,
            ));
        }
        t_base_ic += t0.elapsed().as_secs_f64();
        // ---- v3 近似（不融合，仅中性化）----
        let views: Vec<_> = slots.iter().map(|s| s.view()).collect();
        let t0 = Instant::now();
        let _ = v3_slots_batch_approx(&views, &v3s);
        t_approx_only += t0.elapsed().as_secs_f64();
        // ---- v3 融合 ----
        let t0 = Instant::now();
        let (fused, _tt) = v3_slots_ic_approx(&views, &v3s, &ctx);
        t_fused += t0.elapsed().as_secs_f64();
        // ---- 对账 ----
        for ((b1, b5), (f1, f5)) in base_ic.iter().zip(fused.iter()) {
            let d1 = (b1.summary[0] - f1.summary[0]).abs();
            let d5 = (b5.summary[0] - f5.summary[0]).abs();
            if d1.is_finite() && d1 > worst_g1 {
                worst_g1 = d1;
            }
            if d5.is_finite() && d5 > worst_g5 {
                worst_g5 = d5;
            }
            n_cmp += 1;
        }
    }
    println!(
        "【{n_factors} 因子 × 13 面】baseline 中性化={t_base_neu:.2}s + IC={t_base_ic:.2}s = {:.2}s",
        t_base_neu + t_base_ic
    );
    println!(
        "                v3 近似(仅中性化)={t_approx_only:.2}s   + 独立 IC 估算 {:.2}s",
        t_base_ic
    );
    println!(
        "                v3 近似+IC 融合 = {t_fused:.2}s   合计加速 = {:.2}x   （单看 中性化+IC）",
        (t_base_neu + t_base_ic) / t_fused
    );
    println!(
        "                IC_mean 差: gap1 最大={worst_g1:.3e}  gap5 最大={worst_g5:.3e}  （{n_cmp} 个 (因子,面)）"
    );
}
