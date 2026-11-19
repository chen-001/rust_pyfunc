use super::pool;
use rayon::prelude::*;
use std::cell::RefCell;

pub(crate) const U16_NAN: u16 = 65535;
const SQRT_12: f32 = 3.4641016151377544;

#[inline]
fn f32_order_u32(v: f32) -> u32 {
    let bits = if v == 0.0 { 0u32 } else { v.to_bits() };
    if (bits & 0x8000_0000) != 0 {
        !bits
    } else {
        bits ^ 0x8000_0000
    }
}

#[inline]
fn pack_f32_idx(v: f32, idx: usize) -> u64 {
    ((f32_order_u32(v) as u64) << 32) | (idx as u64)
}

/// Branchless, stable 4-pass 8-bit LSB radix sort on `(keys_u32[0..cnt], idxs_u16[0..cnt])`.
/// Preserves exact ascending `idxs_u16` order on ties.
#[inline]
fn radix_sort_u32_idx(
    keys: &mut [u32],
    idxs: &mut [u16],
    tmp_k: &mut [u32],
    tmp_i: &mut [u16],
    cnt: usize,
) {
    if cnt <= 1 {
        return;
    }
    if cnt < 48 {
        for i in 1..cnt {
            let k = keys[i];
            let s = idxs[i];
            let mut j = i;
            while j > 0 && (keys[j - 1] > k || (keys[j - 1] == k && idxs[j - 1] > s)) {
                keys[j] = keys[j - 1];
                idxs[j] = idxs[j - 1];
                j -= 1;
            }
            keys[j] = k;
            idxs[j] = s;
        }
        return;
    }

    let mut hist = [[0u16; 256]; 4];
    for p in 0..cnt {
        let k = keys[p];
        hist[0][(k & 0xFF) as usize] += 1;
        hist[1][((k >> 8) & 0xFF) as usize] += 1;
        hist[2][((k >> 16) & 0xFF) as usize] += 1;
        hist[3][(k >> 24) as usize] += 1;
    }

    let cnt_u16 = cnt as u16;
    let mut in_tmp = false;

    for pass in 0..4usize {
        let mut trivial = false;
        let h = &hist[pass];
        for b in 0..256usize {
            if h[b] == cnt_u16 {
                trivial = true;
                break;
            }
            if h[b] != 0 {
                break;
            }
        }
        if trivial {
            continue;
        }

        let mut offsets = [0usize; 256];
        let mut sum = 0usize;
        for b in 0..256usize {
            offsets[b] = sum;
            sum += h[b] as usize;
        }

        let shift = pass * 8;
        if !in_tmp {
            for p in 0..cnt {
                let k = keys[p];
                let bucket = ((k >> shift) & 0xFF) as usize;
                let dst = offsets[bucket];
                offsets[bucket] = dst + 1;
                tmp_k[dst] = k;
                tmp_i[dst] = idxs[p];
            }
        } else {
            for p in 0..cnt {
                let k = tmp_k[p];
                let bucket = ((k >> shift) & 0xFF) as usize;
                let dst = offsets[bucket];
                offsets[bucket] = dst + 1;
                keys[dst] = k;
                idxs[dst] = tmp_i[p];
            }
        }
        in_tmp = !in_tmp;
    }

    if in_tmp {
        keys[..cnt].copy_from_slice(&tmp_k[..cnt]);
        idxs[..cnt].copy_from_slice(&tmp_i[..cnt]);
    }
}

pub(crate) struct SfThreadScratch {
    pub(crate) k_u32: Vec<u32>,
    pub(crate) s_u16: Vec<u16>,
    pub(crate) tmp_k: Vec<u32>,
    pub(crate) tmp_i: Vec<u16>,
    pub(crate) u64_keys: Vec<u64>,
    pub(crate) fin_buf: Vec<u32>,
    pub(crate) fold_vals: Vec<f32>,
    pub(crate) raw_mat_tl: Vec<f32>,
    pub(crate) raw_u16_tl: Vec<u16>,
    pub(crate) fold_u16_tl: Vec<u16>,
    pub(crate) dec_scratch: Vec<u8>,
    pub(crate) shuf_scratch: Vec<u8>,
    win_ring_rk: Vec<f32>,
    win_ring_pct: Vec<f32>,
    faces_day: Vec<f32>,
    cnt5: Vec<u16>,
    cnt10: Vec<u16>,
    cnt20: Vec<u16>,
    sum5: Vec<f64>,
    sum10: Vec<f64>,
    sum20: Vec<f64>,
    sq5: Vec<f64>,
    sq10: Vec<f64>,
    sq20: Vec<f64>,
    max5: Vec<f32>,
    max10: Vec<f32>,
    max20: Vec<f32>,
    min5: Vec<f32>,
    min10: Vec<f32>,
    min20: Vec<f32>,
    cnt_t: [Vec<u16>; 3],
    sum_t: [Vec<f64>; 3],
    sq_t: [Vec<f64>; 3],
    max_t: [Vec<f32>; 3],
    min_t: [Vec<f32>; 3],
    minp_t: [Vec<u16>; 3],
    rk_f32: Vec<f32>,
    e10_f32: Vec<f32>,
    ec2_f32: Vec<f32>,
    proj_c: Vec<f32>,
    cand_row: Vec<f32>,
    fp_b10_acc: Vec<Vec<f32>>,
    fp_c2_acc: Vec<Vec<f32>>,
    sig_b10_sum: [f64; 50],
    sig_c2_sum: [f64; 50],
}

impl SfThreadScratch {
    fn new() -> Self {
        Self {
            k_u32: Vec::new(),
            s_u16: Vec::new(),
            tmp_k: Vec::new(),
            tmp_i: Vec::new(),
            u64_keys: Vec::new(),
            fin_buf: Vec::new(),
            fold_vals: Vec::new(),
            raw_mat_tl: Vec::new(),
            raw_u16_tl: Vec::new(),
            fold_u16_tl: Vec::new(),
            dec_scratch: Vec::new(),
            shuf_scratch: Vec::new(),
            win_ring_rk: Vec::new(),
            win_ring_pct: Vec::new(),
            faces_day: Vec::new(),
            cnt5: Vec::new(),
            cnt10: Vec::new(),
            cnt20: Vec::new(),
            sum5: Vec::new(),
            sum10: Vec::new(),
            sum20: Vec::new(),
            sq5: Vec::new(),
            sq10: Vec::new(),
            sq20: Vec::new(),
            max5: Vec::new(),
            max10: Vec::new(),
            max20: Vec::new(),
            min5: Vec::new(),
            min10: Vec::new(),
            min20: Vec::new(),
            cnt_t: [Vec::new(), Vec::new(), Vec::new()],
            sum_t: [Vec::new(), Vec::new(), Vec::new()],
            sq_t: [Vec::new(), Vec::new(), Vec::new()],
            max_t: [Vec::new(), Vec::new(), Vec::new()],
            min_t: [Vec::new(), Vec::new(), Vec::new()],
            minp_t: [Vec::new(), Vec::new(), Vec::new()],
            rk_f32: Vec::new(),
            e10_f32: Vec::new(),
            ec2_f32: Vec::new(),
            proj_c: Vec::new(),
            cand_row: Vec::new(),
            fp_b10_acc: Vec::new(),
            fp_c2_acc: Vec::new(),
            sig_b10_sum: [0.0; 50],
            sig_c2_sum: [0.0; 50],
        }
    }

    pub(crate) fn ensure_matrix_tl(&mut self, tn: usize) {
        if self.raw_mat_tl.len() < tn {
            self.raw_mat_tl.resize(tn, f32::NAN);
            self.raw_u16_tl.resize(tn, U16_NAN);
            self.fold_u16_tl.resize(tn, U16_NAN);
        }
    }

    pub(crate) fn ensure(&mut self, n_stocks: usize, n_basis_c2: usize, d_fp: usize) {
        if self.fold_vals.len() < n_stocks {
            self.k_u32.resize(n_stocks, 0);
            self.s_u16.resize(n_stocks, 0);
            self.tmp_k.resize(n_stocks, 0);
            self.tmp_i.resize(n_stocks, 0);
            self.u64_keys.reserve(n_stocks.saturating_sub(self.u64_keys.capacity()));
            self.fin_buf.reserve(n_stocks.saturating_sub(self.fin_buf.capacity()));
            self.fold_vals.resize(n_stocks, f32::NAN);
            self.win_ring_rk.resize(40 * n_stocks, f32::NAN);
            self.win_ring_pct.resize(40 * n_stocks, f32::NAN);
            self.faces_day.resize(25 * n_stocks, f32::NAN);
            self.cnt5.resize(n_stocks, 0);
            self.cnt10.resize(n_stocks, 0);
            self.cnt20.resize(n_stocks, 0);
            self.sum5.resize(n_stocks, 0.0);
            self.sum10.resize(n_stocks, 0.0);
            self.sum20.resize(n_stocks, 0.0);
            self.sq5.resize(n_stocks, 0.0);
            self.sq10.resize(n_stocks, 0.0);
            self.sq20.resize(n_stocks, 0.0);
            self.max5.resize(n_stocks, f32::NEG_INFINITY);
            self.max10.resize(n_stocks, f32::NEG_INFINITY);
            self.max20.resize(n_stocks, f32::NEG_INFINITY);
            self.min5.resize(n_stocks, f32::INFINITY);
            self.min10.resize(n_stocks, f32::INFINITY);
            self.min20.resize(n_stocks, f32::INFINITY);
            for h in 0..3 {
                self.cnt_t[h].resize(n_stocks, 0);
                self.sum_t[h].resize(n_stocks, 0.0);
                self.sq_t[h].resize(n_stocks, 0.0);
                self.max_t[h].resize(n_stocks, f32::NEG_INFINITY);
                self.min_t[h].resize(n_stocks, f32::INFINITY);
                self.minp_t[h].resize(n_stocks, 2);
            }
            self.rk_f32.resize(n_stocks, 0.0);
            self.e10_f32.resize(n_stocks, 0.0);
            self.ec2_f32.resize(n_stocks, 0.0);
            self.cand_row.resize(n_stocks, f32::NAN);
        }
        if self.proj_c.len() < n_basis_c2 {
            self.proj_c.resize(n_basis_c2, 0.0);
        }
        if d_fp > 0 && (self.fp_b10_acc.len() < 50 || self.fp_b10_acc[0].len() < d_fp) {
            self.fp_b10_acc = vec![vec![0.0f32; d_fp]; 50];
            self.fp_c2_acc = vec![vec![0.0f32; d_fp]; 50];
        }
    }
}

thread_local! {
    pub(crate) static SF_SCRATCH: RefCell<SfThreadScratch> = RefCell::new(SfThreadScratch::new());
}

#[inline]
pub(crate) fn rank_slice_to_u16_radix(
    vals: &[f32],
    k_u32: &mut [u32],
    s_u16: &mut [u16],
    tmp_k: &mut [u32],
    tmp_i: &mut [u16],
    out_u16: &mut [u16],
) {
    let n = vals.len();
    let mut cnt = 0usize;
    for s in 0..n {
        out_u16[s] = U16_NAN;
        let v = vals[s];
        if v.is_finite() {
            k_u32[cnt] = f32_order_u32(v);
            s_u16[cnt] = s as u16;
            cnt += 1;
        }
    }
    if cnt < 10 {
        return;
    }
    radix_sort_u32_idx(k_u32, s_u16, tmp_k, tmp_i, cnt);
    let mut i = 0usize;
    while i < cnt {
        let hi0 = k_u32[i];
        let mut j = i + 1;
        while j < cnt && k_u32[j] == hi0 {
            j += 1;
        }
        let two_rk = (i + 1 + j).min(65534) as u16;
        for p in i..j {
            out_u16[s_u16[p] as usize] = two_rk;
        }
        i = j;
    }
}

#[inline]
pub(crate) fn rank_raw_mat_into_u16(
    raw_mat: &[f32],
    t_len: usize,
    n_stocks: usize,
    k_u32: &mut [u32],
    s_u16: &mut [u16],
    tmp_k: &mut [u32],
    tmp_i: &mut [u16],
    fold_vals: &mut [f32],
    raw_u16_out: &mut [u16],
    fold_u16_out: &mut [u16],
) {
    for t in 0..t_len {
        let off = t * n_stocks;
        let r_row = &raw_mat[off..off + n_stocks];
        let mut sum = 0.0f64;
        let mut cnt = 0usize;
        for s in 0..n_stocks {
            let v = r_row[s];
            if v.is_finite() {
                sum += v as f64;
                cnt += 1;
            }
        }
        if cnt >= 10 {
            let mean = (sum / (cnt as f64)) as f32;
            for s in 0..n_stocks {
                let v = r_row[s];
                fold_vals[s] = if v.is_finite() { (v - mean).abs() } else { f32::NAN };
            }
            rank_slice_to_u16_radix(
                r_row,
                k_u32,
                s_u16,
                tmp_k,
                tmp_i,
                &mut raw_u16_out[off..off + n_stocks],
            );
            rank_slice_to_u16_radix(
                &fold_vals[..n_stocks],
                k_u32,
                s_u16,
                tmp_k,
                tmp_i,
                &mut fold_u16_out[off..off + n_stocks],
            );
        } else {
            raw_u16_out[off..off + n_stocks].fill(U16_NAN);
            fold_u16_out[off..off + n_stocks].fill(U16_NAN);
        }
    }
}

#[inline]
fn load_u16_day_into_rings(
    u_src: &[u16],
    n_stocks: usize,
    r_rk: &mut [f32],
    r_pct: &mut [f32],
) {
    let mut cnt = 0usize;
    for s in 0..n_stocks {
        if u_src[s] != U16_NAN {
            cnt += 1;
        }
    }
    if cnt < 10 {
        r_rk[..n_stocks].fill(f32::NAN);
        r_pct[..n_stocks].fill(f32::NAN);
        return;
    }
    let inv_cnt = 1.0f32 / (cnt as f32);
    for s in 0..n_stocks {
        let code = u_src[s];
        if code == U16_NAN {
            r_rk[s] = f32::NAN;
            r_pct[s] = f32::NAN;
        } else {
            let avg_rk = (code as f32) * 0.5f32;
            r_rk[s] = avg_rk;
            r_pct[s] = (avg_rk - 0.5f32) * inv_cnt;
        }
    }
}

fn compute_25_faces_at_t(
    t_row: usize,
    n_stocks: usize,
    t_len: usize,
    win_len_turn: &[u8],
    sc: &mut SfThreadScratch,
) {
    let tn = t_len * n_stocks;
    let max_lag = (t_row + 1).min(40);
    let cur_rk = &sc.win_ring_rk[(t_row % 40) * n_stocks..(t_row % 40 + 1) * n_stocks];
    sc.faces_day[0..n_stocks].copy_from_slice(cur_rk);

    sc.cnt20[..n_stocks].fill(0);
    sc.sum20[..n_stocks].fill(0.0);
    sc.sq20[..n_stocks].fill(0.0);
    sc.max20[..n_stocks].fill(f32::NEG_INFINITY);
    sc.min20[..n_stocks].fill(f32::INFINITY);

    let w15_row = &win_len_turn[t_row * n_stocks..(t_row + 1) * n_stocks];
    let w30_row = &win_len_turn[tn + t_row * n_stocks..tn + (t_row + 1) * n_stocks];
    let w60_row = &win_len_turn[2 * tn + t_row * n_stocks..2 * tn + (t_row + 1) * n_stocks];

    for h in 0..3 {
        sc.cnt_t[h][..n_stocks].fill(0);
        sc.sum_t[h][..n_stocks].fill(0.0);
        sc.sq_t[h][..n_stocks].fill(0.0);
        sc.max_t[h][..n_stocks].fill(f32::NEG_INFINITY);
        sc.min_t[h][..n_stocks].fill(f32::INFINITY);
    }
    for s in 0..n_stocks {
        sc.minp_t[0][s] = ((w15_row[s] as u16) / 2).max(2);
        sc.minp_t[1][s] = ((w30_row[s] as u16) / 2).max(2);
        sc.minp_t[2][s] = ((w60_row[s] as u16) / 2).max(2);
    }

    for lag in 0..max_lag {
        let r_slot = (t_row + 40 - lag) % 40;
        let row_rk = &sc.win_ring_rk[r_slot * n_stocks..(r_slot + 1) * n_stocks];
        let row_pct = &sc.win_ring_pct[r_slot * n_stocks..(r_slot + 1) * n_stocks];

        if lag < 20 {
            for s in 0..n_stocks {
                let v = row_rk[s];
                if v.is_finite() {
                    let vf = v as f64;
                    sc.cnt20[s] += 1;
                    sc.sum20[s] += vf;
                    sc.sq20[s] += vf * vf;
                    if v > sc.max20[s] {
                        sc.max20[s] = v;
                    }
                    if v < sc.min20[s] {
                        sc.min20[s] = v;
                    }
                }
            }
            if lag == 4 {
                sc.cnt5[..n_stocks].copy_from_slice(&sc.cnt20[..n_stocks]);
                sc.sum5[..n_stocks].copy_from_slice(&sc.sum20[..n_stocks]);
                sc.sq5[..n_stocks].copy_from_slice(&sc.sq20[..n_stocks]);
                sc.max5[..n_stocks].copy_from_slice(&sc.max20[..n_stocks]);
                sc.min5[..n_stocks].copy_from_slice(&sc.min20[..n_stocks]);
            } else if lag == 9 {
                sc.cnt10[..n_stocks].copy_from_slice(&sc.cnt20[..n_stocks]);
                sc.sum10[..n_stocks].copy_from_slice(&sc.sum20[..n_stocks]);
                sc.sq10[..n_stocks].copy_from_slice(&sc.sq20[..n_stocks]);
                sc.max10[..n_stocks].copy_from_slice(&sc.max20[..n_stocks]);
                sc.min10[..n_stocks].copy_from_slice(&sc.min20[..n_stocks]);
            }
        }

        let lag_u8 = lag as u8;
        for s in 0..n_stocks {
            let v = row_pct[s];
            if v.is_finite() {
                let vf = v as f64;
                if lag_u8 < w15_row[s] {
                    sc.cnt_t[0][s] += 1;
                    sc.sum_t[0][s] += vf;
                    sc.sq_t[0][s] += vf * vf;
                    if v > sc.max_t[0][s] {
                        sc.max_t[0][s] = v;
                    }
                    if v < sc.min_t[0][s] {
                        sc.min_t[0][s] = v;
                    }
                }
                if lag_u8 < w30_row[s] {
                    sc.cnt_t[1][s] += 1;
                    sc.sum_t[1][s] += vf;
                    sc.sq_t[1][s] += vf * vf;
                    if v > sc.max_t[1][s] {
                        sc.max_t[1][s] = v;
                    }
                    if v < sc.min_t[1][s] {
                        sc.min_t[1][s] = v;
                    }
                }
                if lag_u8 < w60_row[s] {
                    sc.cnt_t[2][s] += 1;
                    sc.sum_t[2][s] += vf;
                    sc.sq_t[2][s] += vf * vf;
                    if v > sc.max_t[2][s] {
                        sc.max_t[2][s] = v;
                    }
                    if v < sc.min_t[2][s] {
                        sc.min_t[2][s] = v;
                    }
                }
            }
        }
    }

    if max_lag <= 4 {
        sc.cnt5[..n_stocks].copy_from_slice(&sc.cnt20[..n_stocks]);
        sc.sum5[..n_stocks].copy_from_slice(&sc.sum20[..n_stocks]);
        sc.sq5[..n_stocks].copy_from_slice(&sc.sq20[..n_stocks]);
        sc.max5[..n_stocks].copy_from_slice(&sc.max20[..n_stocks]);
        sc.min5[..n_stocks].copy_from_slice(&sc.min20[..n_stocks]);
    }
    if max_lag <= 9 {
        sc.cnt10[..n_stocks].copy_from_slice(&sc.cnt20[..n_stocks]);
        sc.sum10[..n_stocks].copy_from_slice(&sc.sum20[..n_stocks]);
        sc.sq10[..n_stocks].copy_from_slice(&sc.sq20[..n_stocks]);
        sc.max10[..n_stocks].copy_from_slice(&sc.max20[..n_stocks]);
        sc.min10[..n_stocks].copy_from_slice(&sc.min20[..n_stocks]);
    }

    #[inline]
    fn write_4_fixed(
        cnt: &[u16],
        sum: &[f64],
        sq: &[f64],
        mx: &[f32],
        mn: &[f32],
        min_p: usize,
        n_stocks: usize,
        out4: &mut [f32],
    ) {
        let (o_mean, r1) = out4.split_at_mut(n_stocks);
        let (o_max, r2) = r1.split_at_mut(n_stocks);
        let (o_min, o_std) = r2.split_at_mut(n_stocks);
        for s in 0..n_stocks {
            let c = cnt[s] as usize;
            if c >= min_p {
                let cf = c as f64;
                o_mean[s] = (sum[s] / cf) as f32;
                o_max[s] = mx[s];
                o_min[s] = mn[s];
                o_std[s] = if c > 1 {
                    (((sq[s] - (sum[s] * sum[s]) / cf) / ((c - 1) as f64)).max(0.0)).sqrt() as f32
                } else {
                    f32::NAN
                };
            } else {
                o_mean[s] = f32::NAN;
                o_max[s] = f32::NAN;
                o_min[s] = f32::NAN;
                o_std[s] = f32::NAN;
            }
        }
    }

    #[inline]
    fn write_4_turn(
        cnt: &[u16],
        sum: &[f64],
        sq: &[f64],
        mx: &[f32],
        mn: &[f32],
        min_p_arr: &[u16],
        n_stocks: usize,
        out4: &mut [f32],
    ) {
        let (o_mean, r1) = out4.split_at_mut(n_stocks);
        let (o_max, r2) = r1.split_at_mut(n_stocks);
        let (o_min, o_std) = r2.split_at_mut(n_stocks);
        for s in 0..n_stocks {
            let c = cnt[s] as usize;
            if c >= (min_p_arr[s] as usize) {
                let cf = c as f64;
                o_mean[s] = (sum[s] / cf) as f32;
                o_max[s] = mx[s];
                o_min[s] = mn[s];
                o_std[s] = if c > 1 {
                    (((sq[s] - (sum[s] * sum[s]) / cf) / ((c - 1) as f64)).max(0.0)).sqrt() as f32
                } else {
                    f32::NAN
                };
            } else {
                o_mean[s] = f32::NAN;
                o_max[s] = f32::NAN;
                o_min[s] = f32::NAN;
                o_std[s] = f32::NAN;
            }
        }
    }

    write_4_fixed(&sc.cnt5, &sc.sum5, &sc.sq5, &sc.max5, &sc.min5, 2, n_stocks, &mut sc.faces_day[n_stocks..5 * n_stocks]);
    write_4_fixed(&sc.cnt10, &sc.sum10, &sc.sq10, &sc.max10, &sc.min10, 5, n_stocks, &mut sc.faces_day[5 * n_stocks..9 * n_stocks]);
    write_4_fixed(&sc.cnt20, &sc.sum20, &sc.sq20, &sc.max20, &sc.min20, 10, n_stocks, &mut sc.faces_day[9 * n_stocks..13 * n_stocks]);
    for h in 0..3 {
        let base = (13 + h * 4) * n_stocks;
        write_4_turn(&sc.cnt_t[h], &sc.sum_t[h], &sc.sq_t[h], &sc.max_t[h], &sc.min_t[h], &sc.minp_t[h], n_stocks, &mut sc.faces_day[base..base + 4 * n_stocks]);
    }
}

fn compute_single_slot_at_t(
    slot_id: usize,
    t_row: usize,
    n_stocks: usize,
    t_len: usize,
    win_len_turn: &[u8],
    sc: &mut SfThreadScratch,
) {
    if slot_id == 0 {
        let cur_rk = &sc.win_ring_rk[(t_row % 40) * n_stocks..(t_row % 40 + 1) * n_stocks];
        sc.faces_day[0..n_stocks].copy_from_slice(cur_rk);
        return;
    }
    if slot_id < 13 {
        let grp = (slot_id - 1) / 4;
        let stat = (slot_id - 1) % 4;
        let (w, min_p) = match grp {
            0 => (5usize, 2usize),
            1 => (10usize, 5usize),
            _ => (20usize, 10usize),
        };
        let max_lag = (t_row + 1).min(w);
        sc.cnt20[..n_stocks].fill(0);
        sc.sum20[..n_stocks].fill(0.0);
        sc.sq20[..n_stocks].fill(0.0);
        sc.max20[..n_stocks].fill(f32::NEG_INFINITY);
        sc.min20[..n_stocks].fill(f32::INFINITY);

        for lag in 0..max_lag {
            let r_slot = (t_row + 40 - lag) % 40;
            let row_rk = &sc.win_ring_rk[r_slot * n_stocks..(r_slot + 1) * n_stocks];
            for s in 0..n_stocks {
                let v = row_rk[s];
                if v.is_finite() {
                    let vf = v as f64;
                    sc.cnt20[s] += 1;
                    sc.sum20[s] += vf;
                    sc.sq20[s] += vf * vf;
                    if v > sc.max20[s] {
                        sc.max20[s] = v;
                    }
                    if v < sc.min20[s] {
                        sc.min20[s] = v;
                    }
                }
            }
        }
        let out = &mut sc.faces_day[0..n_stocks];
        for s in 0..n_stocks {
            let c = sc.cnt20[s] as usize;
            if c >= min_p {
                let cf = c as f64;
                out[s] = match stat {
                    0 => (sc.sum20[s] / cf) as f32,
                    1 => sc.max20[s],
                    2 => sc.min20[s],
                    _ => {
                        if c > 1 {
                            let var = ((sc.sq20[s] - (sc.sum20[s] * sc.sum20[s]) / cf) / ((c - 1) as f64)).max(0.0);
                            var.sqrt() as f32
                        } else {
                            f32::NAN
                        }
                    }
                };
            } else {
                out[s] = f32::NAN;
            }
        }
        return;
    }

    let h = (slot_id - 13) / 4;
    let stat = (slot_id - 13) % 4;
    let tn = t_len * n_stocks;
    let max_lag = (t_row + 1).min(40);
    let w_row = &win_len_turn[h * tn + t_row * n_stocks..h * tn + (t_row + 1) * n_stocks];

    sc.cnt20[..n_stocks].fill(0);
    sc.sum20[..n_stocks].fill(0.0);
    sc.sq20[..n_stocks].fill(0.0);
    sc.max20[..n_stocks].fill(f32::NEG_INFINITY);
    sc.min20[..n_stocks].fill(f32::INFINITY);

    for lag in 0..max_lag {
        let r_slot = (t_row + 40 - lag) % 40;
        let row_pct = &sc.win_ring_pct[r_slot * n_stocks..(r_slot + 1) * n_stocks];
        let lag_u8 = lag as u8;
        for s in 0..n_stocks {
            if lag_u8 < w_row[s] {
                let v = row_pct[s];
                if v.is_finite() {
                    let vf = v as f64;
                    sc.cnt20[s] += 1;
                    sc.sum20[s] += vf;
                    sc.sq20[s] += vf * vf;
                    if v > sc.max20[s] {
                        sc.max20[s] = v;
                    }
                    if v < sc.min20[s] {
                        sc.min20[s] = v;
                    }
                }
            }
        }
    }
    let out = &mut sc.faces_day[0..n_stocks];
    for s in 0..n_stocks {
        let min_p = ((w_row[s] as usize) / 2).max(2);
        let c = sc.cnt20[s] as usize;
        if c >= min_p {
            let cf = c as f64;
            out[s] = match stat {
                0 => (sc.sum20[s] / cf) as f32,
                1 => sc.max20[s],
                2 => sc.min20[s],
                _ => {
                    if c > 1 {
                        let var = ((sc.sq20[s] - (sc.sum20[s] * sc.sum20[s]) / cf) / ((c - 1) as f64)).max(0.0);
                        var.sqrt() as f32
                    } else {
                        f32::NAN
                    }
                }
            };
        } else {
            out[s] = f32::NAN;
        }
    }
}

#[inline]
fn standardize_rank_row_barra_free(
    f_row: &[f32],
    sign: f32,
    keys: &mut Vec<u64>,
    out_u: &mut [f32],
) {
    let n = f_row.len();
    out_u.fill(f32::NAN);
    keys.clear();
    for s in 0..n {
        let v = f_row[s];
        if v.is_finite() {
            keys.push(pack_f32_idx(v, s));
        }
    }
    let cnt = keys.len();
    if cnt < 12 {
        return;
    }
    keys.sort_unstable();
    let inv_cnt = 1.0f32 / (cnt as f32);
    let scale = sign * SQRT_12;
    let mut i = 0usize;
    while i < cnt {
        let hi0 = keys[i] >> 32;
        let mut j = i + 1;
        while j < cnt && (keys[j] >> 32) == hi0 {
            j += 1;
        }
        let avg_rk = 0.5f32 * ((i + 1 + j) as f32);
        let u_val = ((avg_rk - 0.5f32) * inv_cnt - 0.5f32) * scale;
        for p in i..j {
            let s = (keys[p] as u32) as usize;
            out_u[s] = u_val;
        }
        i = j;
    }
}

#[inline]
fn neutralize_and_spearman_row(
    f_row: &[f32],
    ev_row: &[u8],
    q_day: &[f32],
    n_basis: usize,
    ret_ord_row: &[f32],
    n_stocks: usize,
    keys: &mut Vec<u64>,
    fin_buf: &mut Vec<u32>,
    rk_f32: &mut [f32],
    e_f32: &mut [f32],
    proj_c: &mut [f32],
    mut z_out_opt: Option<(&mut [f32], f32)>,
) -> (f32, f32) {
    fin_buf.clear();
    let mut tot_ev = 0usize;
    for s in 0..n_stocks {
        if ev_row[s] != 0 {
            tot_ev += 1;
            let v = f_row[s];
            if v.is_finite() {
                fin_buf.push(f32_order_u32(v));
            }
        }
    }
    if tot_ev < 200 || fin_buf.len() * 2 < tot_ev {
        return (f32::NAN, f32::NAN);
    }
    let med_key = if fin_buf.len() == tot_ev {
        0u32
    } else {
        let mid = fin_buf.len() / 2;
        fin_buf.select_nth_unstable(mid);
        fin_buf[mid]
    };

    keys.clear();
    for s in 0..n_stocks {
        rk_f32[s] = 0.0;
        if ev_row[s] != 0 {
            let v = f_row[s];
            let k_hi = if v.is_finite() { f32_order_u32(v) } else { med_key };
            keys.push(((k_hi as u64) << 32) | (s as u64));
        }
    }
    keys.sort_unstable();

    let cnt = keys.len();
    for (ord_idx, &kv) in keys.iter().enumerate() {
        let s = (kv as u32) as usize;
        rk_f32[s] = (ord_idx + 1) as f32;
    }

    let cnt_f = cnt as f64;
    let rk_norm_sq = cnt_f * (cnt_f + 1.0) * (2.0 * cnt_f + 1.0) / 6.0;
    let mean_rk = 0.5 * (cnt_f + 1.0);
    let centered_rk_sq = (rk_norm_sq - cnt_f * mean_rk * mean_rk).max(1e-9);

    let mut proj_sq = 0.0f64;
    for b in 0..n_basis {
        let q_row = &q_day[b * n_stocks..(b + 1) * n_stocks];
        let mut dot = 0.0f32;
        for s in 0..n_stocks {
            dot += rk_f32[s] * q_row[s];
        }
        proj_c[b] = dot;
        proj_sq += (dot as f64) * (dot as f64);
    }
    let resid_sq = rk_norm_sq - proj_sq;
    if resid_sq <= 1e-9 {
        return (f32::NAN, f32::NAN);
    }

    e_f32[..n_stocks].copy_from_slice(&rk_f32[..n_stocks]);
    for b in 0..n_basis {
        let c_b = proj_c[b];
        let q_row = &q_day[b * n_stocks..(b + 1) * n_stocks];
        for s in 0..n_stocks {
            e_f32[s] -= c_b * q_row[s];
        }
    }

    let sigma_ratio = (resid_sq / centered_rk_sq).clamp(0.01, 1.0).sqrt() as f32;
    let std_scale = (cnt_f / resid_sq).sqrt() as f32;

    if let Some((z_out, sign)) = z_out_opt.as_mut() {
        let sc_z = std_scale * (*sign);
        for s in 0..n_stocks {
            z_out[s] = if ev_row[s] != 0 { e_f32[s] * sc_z } else { f32::NAN };
        }
    }

    keys.clear();
    for s in 0..n_stocks {
        if ev_row[s] != 0 {
            keys.push(pack_f32_idx(e_f32[s], s));
        }
    }
    keys.sort_unstable();

    let mean_ord = 0.5f32 * ((cnt - 1) as f32);
    let norm_ord = (cnt_f * (cnt_f * cnt_f - 1.0) / 12.0).sqrt();
    if norm_ord <= 1e-9 {
        return (f32::NAN, f32::NAN);
    }
    let mut dot_sp = 0.0f64;
    for (ord_rk, &kv) in keys.iter().enumerate() {
        let s = (kv as u32) as usize;
        let u_x = ((ord_rk as f32) - mean_ord) as f64;
        dot_sp += u_x * (ret_ord_row[s] as f64);
    }
    ((dot_sp / norm_ord) as f32, sigma_ratio)
}

#[inline]
fn neutralize_and_spearman_dual_radix(
    f_row: &[f32],
    ev_row: &[u8],
    q14_day: &[f32],
    n_b10: usize,
    n_c2: usize,
    do_c2_proj: bool,
    do_c2_ic: bool,
    ret_ord_row: &[f32],
    n_stocks: usize,
    k_u32: &mut [u32],
    s_u16: &mut [u16],
    tmp_k: &mut [u32],
    tmp_i: &mut [u16],
    fin_buf: &mut Vec<u32>,
    rk_f32: &mut [f32],
    e10_f32: &mut [f32],
    ec2_f32: &mut [f32],
    proj_c: &mut [f32],
    mut fp_dual_opt: Option<(&[i32], &mut [f32], &mut [f32])>,
) -> (f32, f32, f32, f32) {
    fin_buf.clear();
    let mut tot_ev = 0usize;
    for s in 0..n_stocks {
        if ev_row[s] != 0 {
            tot_ev += 1;
            let v = f_row[s];
            if v.is_finite() {
                fin_buf.push(f32_order_u32(v));
            }
        }
    }
    if tot_ev < 200 || fin_buf.len() * 2 < tot_ev {
        return (f32::NAN, f32::NAN, f32::NAN, f32::NAN);
    }

    let med_key = if fin_buf.len() == tot_ev {
        0u32
    } else {
        let mid = fin_buf.len() / 2;
        fin_buf.select_nth_unstable(mid);
        fin_buf[mid]
    };

    let mut cnt = 0usize;
    for s in 0..n_stocks {
        rk_f32[s] = 0.0;
        if ev_row[s] != 0 {
            let v = f_row[s];
            k_u32[cnt] = if v.is_finite() { f32_order_u32(v) } else { med_key };
            s_u16[cnt] = s as u16;
            cnt += 1;
        }
    }
    radix_sort_u32_idx(k_u32, s_u16, tmp_k, tmp_i, cnt);

    for ord_idx in 0..cnt {
        rk_f32[s_u16[ord_idx] as usize] = (ord_idx + 1) as f32;
    }

    let cnt_f = cnt as f64;
    let rk_norm_sq = cnt_f * (cnt_f + 1.0) * (2.0 * cnt_f + 1.0) / 6.0;
    let mean_rk = 0.5 * (cnt_f + 1.0);
    let centered_rk_sq = (rk_norm_sq - cnt_f * mean_rk * mean_rk).max(1e-9);

    let mut proj_sq_10 = 0.0f64;
    let mut b = 0usize;
    while b + 4 <= n_b10 {
        let q0 = &q14_day[b * n_stocks..(b + 1) * n_stocks];
        let q1 = &q14_day[(b + 1) * n_stocks..(b + 2) * n_stocks];
        let q2 = &q14_day[(b + 2) * n_stocks..(b + 3) * n_stocks];
        let q3 = &q14_day[(b + 3) * n_stocks..(b + 4) * n_stocks];
        let (mut d0, mut d1, mut d2, mut d3) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
        for s in 0..n_stocks {
            let rv = rk_f32[s];
            d0 += rv * q0[s];
            d1 += rv * q1[s];
            d2 += rv * q2[s];
            d3 += rv * q3[s];
        }
        proj_c[b] = d0;
        proj_c[b + 1] = d1;
        proj_c[b + 2] = d2;
        proj_c[b + 3] = d3;
        proj_sq_10 += (d0 as f64) * (d0 as f64)
            + (d1 as f64) * (d1 as f64)
            + (d2 as f64) * (d2 as f64)
            + (d3 as f64) * (d3 as f64);
        b += 4;
    }
    while b < n_b10 {
        let q_row = &q14_day[b * n_stocks..(b + 1) * n_stocks];
        let mut dot = 0.0f32;
        for s in 0..n_stocks {
            dot += rk_f32[s] * q_row[s];
        }
        proj_c[b] = dot;
        proj_sq_10 += (dot as f64) * (dot as f64);
        b += 1;
    }
    let resid_sq_10 = rk_norm_sq - proj_sq_10;
    if resid_sq_10 <= 1e-9 {
        return (f32::NAN, f32::NAN, f32::NAN, f32::NAN);
    }

    e10_f32[..n_stocks].copy_from_slice(&rk_f32[..n_stocks]);
    let mut bs = 0usize;
    while bs + 4 <= n_b10 {
        let (c0_b, c1_b, c2_b, c3_b) = (proj_c[bs], proj_c[bs + 1], proj_c[bs + 2], proj_c[bs + 3]);
        let q0 = &q14_day[bs * n_stocks..(bs + 1) * n_stocks];
        let q1 = &q14_day[(bs + 1) * n_stocks..(bs + 2) * n_stocks];
        let q2 = &q14_day[(bs + 2) * n_stocks..(bs + 3) * n_stocks];
        let q3 = &q14_day[(bs + 3) * n_stocks..(bs + 4) * n_stocks];
        for s in 0..n_stocks {
            e10_f32[s] -= c0_b * q0[s] + c1_b * q1[s] + c2_b * q2[s] + c3_b * q3[s];
        }
        bs += 4;
    }
    while bs < n_b10 {
        let c_b = proj_c[bs];
        let q_row = &q14_day[bs * n_stocks..(bs + 1) * n_stocks];
        for s in 0..n_stocks {
            e10_f32[s] -= c_b * q_row[s];
        }
        bs += 1;
    }

    let sig_10 = (resid_sq_10 / centered_rk_sq).clamp(0.01, 1.0).sqrt() as f32;
    let std_10 = (cnt_f / resid_sq_10).sqrt() as f32;

    let mut sig_c2 = f32::NAN;
    let mut std_c2 = 0.0f32;
    let mut c2_valid = false;
    if do_c2_proj {
        let mut proj_sq_c2 = proj_sq_10;
        for b_idx in n_b10..n_c2 {
            let q_row = &q14_day[b_idx * n_stocks..(b_idx + 1) * n_stocks];
            let mut dot = 0.0f32;
            for s in 0..n_stocks {
                dot += rk_f32[s] * q_row[s];
            }
            proj_c[b_idx] = dot;
            proj_sq_c2 += (dot as f64) * (dot as f64);
        }
        let resid_sq_c2 = rk_norm_sq - proj_sq_c2;
        if resid_sq_c2 > 1e-9 {
            ec2_f32[..n_stocks].copy_from_slice(&e10_f32[..n_stocks]);
            for b_idx in n_b10..n_c2 {
                let c_b = proj_c[b_idx];
                let q_row = &q14_day[b_idx * n_stocks..(b_idx + 1) * n_stocks];
                for s in 0..n_stocks {
                    ec2_f32[s] -= c_b * q_row[s];
                }
            }
            sig_c2 = (resid_sq_c2 / centered_rk_sq).clamp(0.01, 1.0).sqrt() as f32;
            std_c2 = (cnt_f / resid_sq_c2).sqrt() as f32;
            c2_valid = true;
        }
    }

    if let Some((fp_cols, dst_10, dst_c2)) = fp_dual_opt.as_mut() {
        for (k_fp, &s_idx) in fp_cols.iter().enumerate() {
            if s_idx >= 0 {
                let s = s_idx as usize;
                if ev_row[s] != 0 {
                    dst_10[k_fp] = e10_f32[s] * std_10;
                    if c2_valid {
                        dst_c2[k_fp] = ec2_f32[s] * std_c2;
                    }
                } else {
                    dst_10[k_fp] = 0.0;
                    if c2_valid {
                        dst_c2[k_fp] = 0.0;
                    }
                }
            } else {
                dst_10[k_fp] = 0.0;
                if c2_valid {
                    dst_c2[k_fp] = 0.0;
                }
            }
        }
    }

    let mean_ord = 0.5f32 * ((cnt - 1) as f32);
    let norm_ord = (cnt_f * (cnt_f * cnt_f - 1.0) / 12.0).sqrt();
    if norm_ord <= 1e-9 {
        return (f32::NAN, f32::NAN, f32::NAN, f32::NAN);
    }

    let mut c1 = 0usize;
    for s in 0..n_stocks {
        if ev_row[s] != 0 {
            k_u32[c1] = f32_order_u32(e10_f32[s]);
            s_u16[c1] = s as u16;
            c1 += 1;
        }
    }
    radix_sort_u32_idx(k_u32, s_u16, tmp_k, tmp_i, c1);
    let mut dot_10 = 0.0f64;
    for ord_rk in 0..c1 {
        let s = s_u16[ord_rk] as usize;
        dot_10 += ((ord_rk as f32 - mean_ord) as f64) * (ret_ord_row[s] as f64);
    }
    let ic_10 = (dot_10 / norm_ord) as f32;

    let ic_c2 = if do_c2_ic && c2_valid {
        let mut c2 = 0usize;
        for s in 0..n_stocks {
            if ev_row[s] != 0 {
                k_u32[c2] = f32_order_u32(ec2_f32[s]);
                s_u16[c2] = s as u16;
                c2 += 1;
            }
        }
        radix_sort_u32_idx(k_u32, s_u16, tmp_k, tmp_i, c2);
        let mut dot_c2 = 0.0f64;
        for ord_rk in 0..c2 {
            let s = s_u16[ord_rk] as usize;
            dot_c2 += ((ord_rk as f32 - mean_ord) as f64) * (ret_ord_row[s] as f64);
        }
        (dot_c2 / norm_ord) as f32
    } else {
        f32::NAN
    };

    (ic_10, ic_c2, sig_10, sig_c2)
}

#[inline]
fn quantize_fp_vec(fp_vec: &[f32], dst_i8: &mut [i8]) -> f32 {
    let d_fp = fp_vec.len();
    if d_fp == 0 {
        return 0.0;
    }
    let mut mean = 0.0f64;
    for &v in fp_vec {
        mean += v as f64;
    }
    mean /= d_fp as f64;
    let mut var = 0.0f64;
    for &v in fp_vec {
        let d = (v as f64) - mean;
        var += d * d;
    }
    let sd = (var / (d_fp as f64)).sqrt();
    if sd <= 1e-8 {
        dst_i8.fill(0);
        return 0.0;
    }
    let inv_sd = 1.0 / sd;
    let mut q_sq_sum = 0.0f64;
    for k in 0..d_fp {
        let z = ((fp_vec[k] as f64 - mean) * inv_sd).clamp(-3.5, 3.5);
        let qi = (z * 36.0).round() as i8;
        dst_i8[k] = qi;
        let qf = qi as f64;
        q_sq_sum += qf * qf;
    }
    if q_sq_sum > 1e-9 {
        (1.0 / q_sq_sum.sqrt()) as f32
    } else {
        0.0
    }
}

pub(crate) fn eval_50faces_from_tl_u16(
    sc: &mut SfThreadScratch,
    t_len: usize,
    n_stocks: usize,
    win_len_turn: &[u8],
    n_b10: usize,
    n_c2: usize,
    has_c2: bool,
    g1_rows: &[i32],
    n_g1: usize,
    ev_g1: &[u8],
    q14_g1: &[f32],
    ret1_ord: &[f32],
    g5_from_g1: &[i32],
    n_g5: usize,
    ev_g5: &[u8],
    q14_g5: &[f32],
    ret5_ord: &[f32],
    fp_g1_map: &[i32],
    fp_stock_idx: &[i32],
    n_fp_dates: usize,
    fp_k_per_day: usize,
    ic5_b10_c: &mut [f32],
    mut ic5_c2_opt: Option<&mut [f32]>,
    mut ic1_b10_opt: Option<&mut [f32]>,
    fp10_i8_c: &mut [i8],
    fp10_sc_c: &mut [f32],
    rs10_sc_c: &mut [f32],
    mut fpc2_opt: Option<(&mut [i8], &mut [f32], &mut [f32])>,
    vd5_c: &mut [u16],
    vd1_c: &mut [u16],
) {
    let d_fp = n_fp_dates * fp_k_per_day;
    let has_g1 = ic1_b10_opt.is_some();
    ic5_b10_c.fill(f32::NAN);
    if let Some(s_c2) = ic5_c2_opt.as_deref_mut() {
        s_c2.fill(f32::NAN);
    }
    if let Some(s_g1) = ic1_b10_opt.as_deref_mut() {
        s_g1.fill(f32::NAN);
    }
    fp10_i8_c.fill(0);
    fp10_sc_c.fill(0.0);
    rs10_sc_c.fill(0.95);
    vd5_c.fill(0);
    vd1_c.fill(0);
    sc.sig_b10_sum.fill(0.0);
    sc.sig_c2_sum.fill(0.0);
    for m_loc in 0..50 {
        sc.fp_b10_acc[m_loc][..d_fp].fill(0.0);
        if has_c2 {
            sc.fp_c2_acc[m_loc][..d_fp].fill(0.0);
        }
    }

    for var_id in 0..2usize {
        sc.win_ring_rk[..40 * n_stocks].fill(f32::NAN);
        sc.win_ring_pct[..40 * n_stocks].fill(f32::NAN);

        let mut te1_ptr = 0usize;
        for t in 0..t_len {
            {
                let sc_ref = &mut *sc;
                let r_rk = &mut sc_ref.win_ring_rk[(t % 40) * n_stocks..(t % 40 + 1) * n_stocks];
                let r_pct = &mut sc_ref.win_ring_pct[(t % 40) * n_stocks..(t % 40 + 1) * n_stocks];
                let u_src = if var_id == 0 {
                    &sc_ref.raw_u16_tl[t * n_stocks..(t + 1) * n_stocks]
                } else {
                    &sc_ref.fold_u16_tl[t * n_stocks..(t + 1) * n_stocks]
                };
                load_u16_day_into_rings(u_src, n_stocks, r_rk, r_pct);
            }

            while te1_ptr < n_g1 && (g1_rows[te1_ptr] as usize) == t {
                let te1 = te1_ptr;
                let fp_slot = fp_g1_map[te1];
                let te5_idx = g5_from_g1[te1];
                if !has_g1 && fp_slot < 0 && te5_idx < 0 {
                    te1_ptr += 1;
                    continue;
                }

                compute_25_faces_at_t(t, n_stocks, t_len, win_len_turn, sc);

                let ev1_row = &ev_g1[te1 * n_stocks..(te1 + 1) * n_stocks];
                let q1_day = &q14_g1[te1 * n_c2 * n_stocks..(te1 + 1) * n_c2 * n_stocks];
                let r1_row = &ret1_ord[te1 * n_stocks..(te1 + 1) * n_stocks];
                let do_g1_c2_proj = has_c2 && fp_slot >= 0;

                let sc_ref = &mut *sc;
                for slot_id in 0..25usize {
                    let m_loc = var_id * 25 + slot_id;
                    let f_row = &sc_ref.faces_day[slot_id * n_stocks..(slot_id + 1) * n_stocks];

                    if has_g1 || fp_slot >= 0 {
                        let fp_opt = if fp_slot >= 0 {
                            let fs = fp_slot as usize;
                            let cols = &fp_stock_idx[fs * fp_k_per_day..(fs + 1) * fp_k_per_day];
                            let d10 = &mut sc_ref.fp_b10_acc[m_loc][fs * fp_k_per_day..(fs + 1) * fp_k_per_day];
                            let dc2 = &mut sc_ref.fp_c2_acc[m_loc][fs * fp_k_per_day..(fs + 1) * fp_k_per_day];
                            Some((cols, d10, dc2))
                        } else {
                            None
                        };

                        let (ic1_10, _, sig1_10, _) = neutralize_and_spearman_dual_radix(
                            f_row,
                            ev1_row,
                            q1_day,
                            n_b10,
                            n_c2,
                            do_g1_c2_proj,
                            false,
                            r1_row,
                            n_stocks,
                            &mut sc_ref.k_u32,
                            &mut sc_ref.s_u16,
                            &mut sc_ref.tmp_k,
                            &mut sc_ref.tmp_i,
                            &mut sc_ref.fin_buf,
                            &mut sc_ref.rk_f32[..n_stocks],
                            &mut sc_ref.e10_f32[..n_stocks],
                            &mut sc_ref.ec2_f32[..n_stocks],
                            &mut sc_ref.proj_c[..n_c2],
                            fp_opt,
                        );
                        if let Some(s_g1) = ic1_b10_opt.as_deref_mut() {
                            if ic1_10.is_finite() {
                                s_g1[m_loc * n_g1 + te1] = ic1_10;
                                vd1_c[m_loc] = vd1_c[m_loc].saturating_add(1);
                                sc_ref.sig_b10_sum[m_loc] += sig1_10 as f64;
                            }
                        }
                    }

                    if te5_idx >= 0 {
                        let p5 = te5_idx as usize;
                        let ev5_row = &ev_g5[p5 * n_stocks..(p5 + 1) * n_stocks];
                        let q5_day = &q14_g5[p5 * n_c2 * n_stocks..(p5 + 1) * n_c2 * n_stocks];
                        let r5_row = &ret5_ord[p5 * n_stocks..(p5 + 1) * n_stocks];
                        let (ic5_10, ic5_c2, sig5_10, sig5_c2) = neutralize_and_spearman_dual_radix(
                            f_row,
                            ev5_row,
                            q5_day,
                            n_b10,
                            n_c2,
                            has_c2,
                            has_c2,
                            r5_row,
                            n_stocks,
                            &mut sc_ref.k_u32,
                            &mut sc_ref.s_u16,
                            &mut sc_ref.tmp_k,
                            &mut sc_ref.tmp_i,
                            &mut sc_ref.fin_buf,
                            &mut sc_ref.rk_f32[..n_stocks],
                            &mut sc_ref.e10_f32[..n_stocks],
                            &mut sc_ref.ec2_f32[..n_stocks],
                            &mut sc_ref.proj_c[..n_c2],
                            None,
                        );
                        if ic5_10.is_finite() && (!has_c2 || ic5_c2.is_finite()) {
                            ic5_b10_c[m_loc * n_g5 + p5] = ic5_10;
                            if !has_g1 {
                                sc_ref.sig_b10_sum[m_loc] += sig5_10 as f64;
                            }
                            if let Some(s_c2) = ic5_c2_opt.as_deref_mut() {
                                s_c2[m_loc * n_g5 + p5] = ic5_c2;
                                sc_ref.sig_c2_sum[m_loc] += sig5_c2 as f64;
                            }
                            vd5_c[m_loc] = vd5_c[m_loc].saturating_add(1);
                        }
                    }
                }
                te1_ptr += 1;
            }
        }
    }

    for m_loc in 0..50usize {
        let vd_ref = if has_g1 { vd1_c[m_loc] as usize } else { vd5_c[m_loc] as usize };
        if vd_ref > 0 {
            rs10_sc_c[m_loc] = (sc.sig_b10_sum[m_loc] / (vd_ref as f64)).clamp(0.10, 1.0) as f32;
        }
        fp10_sc_c[m_loc] = quantize_fp_vec(&sc.fp_b10_acc[m_loc][..d_fp], &mut fp10_i8_c[m_loc * d_fp..(m_loc + 1) * d_fp]);
    }
    if let Some((fpc2_i8_c, fpc2_sc_c, rsc2_sc_c)) = fpc2_opt.as_mut() {
        fpc2_i8_c.fill(0);
        fpc2_sc_c.fill(0.0);
        rsc2_sc_c.fill(0.90);
        for m_loc in 0..50usize {
            let vd5 = vd5_c[m_loc] as usize;
            if vd5 > 0 {
                rsc2_sc_c[m_loc] = (sc.sig_c2_sum[m_loc] / (vd5 as f64)).clamp(0.10, 1.0) as f32;
            }
            fpc2_sc_c[m_loc] = quantize_fp_vec(&sc.fp_c2_acc[m_loc][..d_fp], &mut fpc2_i8_c[m_loc * d_fp..(m_loc + 1) * d_fp]);
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn sf_materialize_faces_u16(
    raw_u16_ptr: *const u16,
    fold_u16_ptr: *const u16,
    n_raw: usize,
    t_len: usize,
    n_stocks: usize,
    win_len_turn_ptr: *const u8,
    output_mode: i32,
    sub_idx_ptr: *const i32,
    n_sub: usize,
    eval_rows_ptr: *const i32,
    n_eval: usize,
    eval_valid_ptr: *const u8,
    style_q_ptr: *const f32,
    n_basis: usize,
    ret_ord_rk_ptr: *const f32,
    specs_ptr: *const i32,
    l_count: usize,
    out_z_ptr: *mut f32,
) {
    if l_count == 0 || n_sub == 0 {
        return;
    }
    let tn = t_len * n_stocks;
    let raw_u16 = std::slice::from_raw_parts(raw_u16_ptr, n_raw * tn);
    let fold_u16 = std::slice::from_raw_parts(fold_u16_ptr, n_raw * tn);
    let win_len_turn = std::slice::from_raw_parts(win_len_turn_ptr, 3 * tn);
    let sub_idx = std::slice::from_raw_parts(sub_idx_ptr, n_sub);
    let specs = std::slice::from_raw_parts(specs_ptr, l_count * 4);
    let out_z = std::slice::from_raw_parts_mut(out_z_ptr, l_count * n_sub * n_stocks);

    let (eval_rows, eval_valid, style_q, ret_ord_rk) = if output_mode == 1 && n_eval > 0 {
        (
            std::slice::from_raw_parts(eval_rows_ptr, n_eval),
            std::slice::from_raw_parts(eval_valid_ptr, n_eval * n_stocks),
            std::slice::from_raw_parts(style_q_ptr, n_eval * n_basis * n_stocks),
            std::slice::from_raw_parts(ret_ord_rk_ptr, n_eval * n_stocks),
        )
    } else {
        (&[][..], &[][..], &[][..], &[][..])
    };

    pool().install(|| {
        out_z
            .par_chunks_mut(n_stocks)
            .enumerate()
            .for_each(|(flat_idx, z_dst)| {
                SF_SCRATCH.with(|cell| {
                    let mut sc = cell.borrow_mut();
                    sc.ensure(n_stocks, n_basis.max(1), 0);
                    z_dst.fill(f32::NAN);

                    let li = flat_idx / n_sub;
                    let sub_ptr = flat_idx % n_sub;

                    let raw_idx = specs[li * 4] as usize;
                    let var_id = specs[li * 4 + 1] as usize;
                    let slot_id = specs[li * 4 + 2] as usize;
                    let sign = specs[li * 4 + 3] as f32;

                    let target_t = if output_mode == 0 {
                        sub_idx[sub_ptr] as usize
                    } else {
                        eval_rows[sub_idx[sub_ptr] as usize] as usize
                    };

                    let need_win = if slot_id == 0 {
                        1usize
                    } else if slot_id < 5 {
                        5usize
                    } else if slot_id < 9 {
                        10usize
                    } else if slot_id < 13 {
                        20usize
                    } else {
                        40usize
                    };
                    let t_start = (target_t + 1).saturating_sub(need_win);

                    let src_all = if var_id == 0 { &raw_u16[..] } else { &fold_u16[..] };
                    let src_mat = &src_all[raw_idx * tn..(raw_idx + 1) * tn];

                    for t in t_start..(target_t + 1) {
                        let sc_ref = &mut *sc;
                        let u_src = &src_mat[t * n_stocks..(t + 1) * n_stocks];
                        if slot_id < 13 {
                            let r_rk = &mut sc_ref.win_ring_rk[(t % 40) * n_stocks..(t % 40 + 1) * n_stocks];
                            let mut cnt = 0usize;
                            for s in 0..n_stocks {
                                let code = u_src[s];
                                if code == U16_NAN {
                                    r_rk[s] = f32::NAN;
                                } else {
                                    cnt += 1;
                                    r_rk[s] = (code as f32) * 0.5f32;
                                }
                            }
                            if cnt < 10 {
                                r_rk[..n_stocks].fill(f32::NAN);
                            }
                        } else {
                            let r_rk = &mut sc_ref.win_ring_rk[(t % 40) * n_stocks..(t % 40 + 1) * n_stocks];
                            let r_pct = &mut sc_ref.win_ring_pct[(t % 40) * n_stocks..(t % 40 + 1) * n_stocks];
                            load_u16_day_into_rings(u_src, n_stocks, r_rk, r_pct);
                        }
                    }

                    compute_single_slot_at_t(slot_id, target_t, n_stocks, t_len, win_len_turn, &mut sc);
                    let sc_ref = &mut *sc;
                    let f_row = &sc_ref.faces_day[0..n_stocks];

                    if output_mode == 0 {
                        standardize_rank_row_barra_free(f_row, sign, &mut sc_ref.u64_keys, z_dst);
                    } else {
                        let te = sub_idx[sub_ptr] as usize;
                        let ev_row = &eval_valid[te * n_stocks..(te + 1) * n_stocks];
                        let q_day = &style_q[te * n_basis * n_stocks..(te + 1) * n_basis * n_stocks];
                        let ret_ord_row = &ret_ord_rk[te * n_stocks..(te + 1) * n_stocks];
                        let _ = neutralize_and_spearman_row(
                            f_row,
                            ev_row,
                            q_day,
                            n_basis,
                            ret_ord_row,
                            n_stocks,
                            &mut sc_ref.u64_keys,
                            &mut sc_ref.fin_buf,
                            &mut sc_ref.rk_f32[..n_stocks],
                            &mut sc_ref.e10_f32[..n_stocks],
                            &mut sc_ref.proj_c[..n_basis],
                            Some((z_dst, sign)),
                        );
                    }
                });
            });
    });
}

#[no_mangle]
pub unsafe extern "C" fn sf_eval_squad_candidates_dual_ptrs(
    base_ptrs_ptr: *const *const f32,
    base_weights_ptr: *const f32,
    n_base: usize,
    cand_ptrs_ptr: *const *const f32,
    cand_weights_ptr: *const f32,
    min_fin_need: usize,
    k_fin: usize,
    n_sub: usize,
    n_stocks: usize,
    sub_eval_idx_ptr: *const i32,
    n_eval: usize,
    eval_valid_ptr: *const u8,
    q14_ptr: *const f32,
    n_b10: usize,
    n_c2: usize,
    need_c2: i32,
    ret_ord_rk_ptr: *const f32,
    out_ic_b10_ptr: *mut f32,
    out_ic_c2_ptr: *mut f32,
) {
    if k_fin == 0 || n_sub == 0 {
        return;
    }
    let do_c2 = need_c2 != 0;
    let sn = n_sub * n_stocks;
    let base_slices: Vec<&[f32]> = if n_base > 0 {
        let ptrs = std::slice::from_raw_parts(base_ptrs_ptr, n_base);
        ptrs.iter().map(|&p| std::slice::from_raw_parts(p, sn)).collect()
    } else {
        Vec::new()
    };
    let base_weights = if n_base > 0 {
        std::slice::from_raw_parts(base_weights_ptr, n_base)
    } else {
        &[][..]
    };
    let cand_ptrs = std::slice::from_raw_parts(cand_ptrs_ptr, k_fin);
    let cand_slices: Vec<&[f32]> = cand_ptrs.iter().map(|&p| std::slice::from_raw_parts(p, sn)).collect();
    let cand_weights = std::slice::from_raw_parts(cand_weights_ptr, k_fin);

    let sub_eval_idx = std::slice::from_raw_parts(sub_eval_idx_ptr, n_sub);
    let eval_valid = std::slice::from_raw_parts(eval_valid_ptr, n_eval * n_stocks);
    let q14 = std::slice::from_raw_parts(q14_ptr, n_eval * n_c2 * n_stocks);
    let ret_ord_rk = std::slice::from_raw_parts(ret_ord_rk_ptr, n_eval * n_stocks);
    let out_ic_b10 = std::slice::from_raw_parts_mut(out_ic_b10_ptr, k_fin * n_sub);
    let out_ic_c2 = std::slice::from_raw_parts_mut(out_ic_c2_ptr, k_fin * n_sub);

    pool().install(|| {
        out_ic_b10
            .par_iter_mut()
            .zip(out_ic_c2.par_iter_mut())
            .enumerate()
            .for_each(|(idx, (ic10_cell, icc2_cell))| {
                SF_SCRATCH.with(|cell| {
                    let mut sc = cell.borrow_mut();
                    sc.ensure(n_stocks, n_c2, 0);
                    let k = idx / n_sub;
                    let si = idx % n_sub;
                    let te = sub_eval_idx[si] as usize;
                    let row_off = si * n_stocks;

                    let f_c = &cand_slices[k][row_off..row_off + n_stocks];
                    let w_c = cand_weights[k];
                    let w2_c = w_c * w_c;

                    let sc_ref = &mut *sc;
                    for s in 0..n_stocks {
                        let mut tsum = 0.0f32;
                        let mut tw2 = 0.0f32;
                        let mut tcnt = 0usize;
                        for bi in 0..n_base {
                            let vb = base_slices[bi][row_off + s];
                            if vb.is_finite() {
                                let wb = base_weights[bi];
                                tsum += vb * wb;
                                tw2 += wb * wb;
                                tcnt += 1;
                            }
                        }
                        let vc = f_c[s];
                        if vc.is_finite() {
                            tsum += vc * w_c;
                            tw2 += w2_c;
                            tcnt += 1;
                        }
                        sc_ref.cand_row[s] = if tcnt >= min_fin_need {
                            tsum / tw2.max(1e-6).sqrt()
                        } else {
                            f32::NAN
                        };
                    }

                    let ev_row = &eval_valid[te * n_stocks..(te + 1) * n_stocks];
                    let q_day = &q14[te * n_c2 * n_stocks..(te + 1) * n_c2 * n_stocks];
                    let ret_ord_row = &ret_ord_rk[te * n_stocks..(te + 1) * n_stocks];

                    let (ic10, icc2, _, _) = neutralize_and_spearman_dual_radix(
                        &sc_ref.cand_row[..n_stocks],
                        ev_row,
                        q_day,
                        n_b10,
                        n_c2,
                        do_c2,
                        do_c2,
                        ret_ord_row,
                        n_stocks,
                        &mut sc_ref.k_u32,
                        &mut sc_ref.s_u16,
                        &mut sc_ref.tmp_k,
                        &mut sc_ref.tmp_i,
                        &mut sc_ref.fin_buf,
                        &mut sc_ref.rk_f32[..n_stocks],
                        &mut sc_ref.e10_f32[..n_stocks],
                        &mut sc_ref.ec2_f32[..n_stocks],
                        &mut sc_ref.proj_c[..n_c2],
                        None,
                    );
                    *ic10_cell = ic10;
                    *icc2_cell = icc2;
                });
            });
    });
}

#[no_mangle]
pub unsafe extern "C" fn sf_exhaustive_helper_scan_pure(
    ic_all_ptr: *const f32,
    m_total: usize,
    n_eval_stride: usize,
    win_end: usize,
    fp_i8_ptr: *const i8,
    fp_scale_ptr: *const f32,
    resid_scale_ptr: *const f32,
    d_fp: usize,
    col_sgn_ptr: *const i8,
    base_valid_ptr: *const u8,
    blocked_raws_ptr: *const i32,
    n_blocked_raws: usize,
    sq_ic_ptr: *const f32,
    sq_fp_ptr: *const f32,
    sq_norm_sq: f64,
    weights_ptr: *const f32,
    n_weights: usize,
    intra_corr_max: f64,
    ir_ann_sqrt: f64,
    mu_floor: f64,
    objective_mode: i32,
    min_rel_gain_scan: f64,
    prefer_eq_rel: f64,
    top_k: usize,
    out_top_idx_ptr: *mut i32,
    out_top_score_ptr: *mut f32,
) -> usize {
    let ic_all = std::slice::from_raw_parts(ic_all_ptr, m_total * n_eval_stride);
    let fp_i8 = std::slice::from_raw_parts(fp_i8_ptr, m_total * d_fp);
    let fp_scale = std::slice::from_raw_parts(fp_scale_ptr, m_total);
    let resid_scale = std::slice::from_raw_parts(resid_scale_ptr, m_total);
    let col_sgn = std::slice::from_raw_parts(col_sgn_ptr, m_total);
    let base_valid = std::slice::from_raw_parts(base_valid_ptr, m_total);
    let blocked_raws = if n_blocked_raws > 0 && !blocked_raws_ptr.is_null() {
        std::slice::from_raw_parts(blocked_raws_ptr, n_blocked_raws)
    } else {
        &[][..]
    };
    let sq_ic = std::slice::from_raw_parts(sq_ic_ptr, win_end);
    let sq_fp = std::slice::from_raw_parts(sq_fp_ptr, d_fp);
    let weights = std::slice::from_raw_parts(weights_ptr, n_weights);
    let out_top_idx = std::slice::from_raw_parts_mut(out_top_idx_ptr, top_k);
    let out_top_score = std::slice::from_raw_parts_mut(out_top_score_ptr, top_k);

    let n_raw_est = (m_total + 49) / 50;
    let mut raw_blocked = vec![0u8; n_raw_est + 1];
    for &r in blocked_raws {
        if r >= 0 && (r as usize) < raw_blocked.len() {
            raw_blocked[r as usize] = 1;
        }
    }

    let sq_norm = sq_norm_sq.sqrt().max(1e-9);
    let inv_base_norm = 1.0 / sq_norm;

    let n_raw_tot = m_total / 50;
    let raw_chunk = ((n_raw_tot + 799) / 800).max(8);
    let winners: Vec<(f32, i32)> = pool().install(|| {
        (0..n_raw_tot)
            .into_par_iter()
            .step_by(raw_chunk)
            .map(|r0| {
                let r1 = (r0 + raw_chunk).min(n_raw_tot);
                let mut local_top: Vec<(f32, i32)> = Vec::with_capacity(top_k + 1);
                let mut min_sc = f32::NEG_INFINITY;

                for r in r0..r1 {
                    if raw_blocked[r] != 0 {
                        continue;
                    }
                    let mut raw_best_sc = f32::NEG_INFINITY;
                    let mut raw_best_m = -1i32;

                    for m in (r * 50)..((r + 1) * 50) {
                        if base_valid[m] == 0 {
                            continue;
                        }
                        let scale_m = fp_scale[m] as f64;
                        let sig_m = resid_scale[m] as f64;
                        if scale_m <= 0.0 || sig_m <= 0.0 {
                            continue;
                        }
                        let sgn_f = col_sgn[m] as f64;
                        let fp_row = &fp_i8[m * d_fp..(m + 1) * d_fp];
                        let mut raw_dot = 0.0f32;
                        for k in 0..d_fp {
                            raw_dot += (fp_row[k] as f32) * sq_fp[k];
                        }
                        let dot_signed = sgn_f * scale_m * (raw_dot as f64);
                        if (dot_signed / sq_norm).abs() > intra_corr_max {
                            continue;
                        }

                        let ic_row = &ic_all[m * n_eval_stride..m * n_eval_stride + win_end];
                        let mut tot_cnt = 0usize;
                        let mut tot_sum_s = 0.0f64;
                        let mut tot_sum_m = 0.0f64;
                        let mut tot_sq_s = 0.0f64;
                        let mut tot_sq_m = 0.0f64;
                        let mut tot_cross = 0.0f64;

                        for t in 0..win_end {
                            let a = sq_ic[t];
                            let b = ic_row[t];
                            if a.is_finite() && b.is_finite() {
                                let af = a as f64;
                                let bf = (b as f64) * sgn_f;
                                tot_cnt += 1;
                                tot_sum_s += af;
                                tot_sum_m += bf;
                                tot_sq_s += af * af;
                                tot_sq_m += bf * bf;
                                tot_cross += af * bf;
                            }
                        }

                        if tot_cnt < win_end / 2 {
                            continue;
                        }
                        let cf_tot = tot_cnt as f64;

                        let mu_base_all = (tot_sum_s * inv_base_norm) / cf_tot;
                        if mu_base_all <= 0.0 {
                            continue;
                        }
                        let base_sc_all = if objective_mode == 1 {
                            mu_base_all
                        } else {
                            let sq_b = tot_sq_s * (inv_base_norm * inv_base_norm);
                            let sd_b = ((sq_b / cf_tot) - mu_base_all * mu_base_all).max(0.0).sqrt().max(1e-6);
                            let ir_b = (mu_base_all / sd_b) * ir_ann_sqrt;
                            if objective_mode == 0 {
                                ir_b
                            } else {
                                (mu_base_all * ir_b).max(0.0).sqrt()
                            }
                        };
                        let min_target = base_sc_all * (1.0 + min_rel_gain_scan);

                        let mut best_eff = f64::NEG_INFINITY;
                        for (wi, &w_f32) in weights.iter().take(4).enumerate() {
                            let w_eff = (w_f32 as f64) * sig_m;
                            let denom_var_cs = (sq_norm_sq + w_eff * w_eff + 2.0 * w_eff * dot_signed).max(0.05);
                            let inv_cs_norm = 1.0 / denom_var_cs.sqrt();

                            let mu_all = ((tot_sum_s + w_eff * tot_sum_m) * inv_cs_norm) / cf_tot;
                            if mu_all <= mu_floor {
                                continue;
                            }
                            let sc_all = if objective_mode == 1 {
                                mu_all
                            } else {
                                let sq_all = (tot_sq_s + w_eff * w_eff * tot_sq_m + 2.0 * w_eff * tot_cross)
                                    * (inv_cs_norm * inv_cs_norm);
                                let sd_all = ((sq_all / cf_tot) - mu_all * mu_all).max(0.0).sqrt().max(1e-6);
                                let ir_all = (mu_all / sd_all) * ir_ann_sqrt;
                                if objective_mode == 0 {
                                    ir_all
                                } else {
                                    (mu_all * ir_all).max(0.0).sqrt()
                                }
                            };

                            if sc_all > min_target {
                                let eff = if wi == 0 {
                                    sc_all
                                } else {
                                    sc_all / (1.0 + prefer_eq_rel)
                                };
                                if eff > best_eff {
                                    best_eff = eff;
                                }
                            }
                        }

                        if best_eff.is_finite() && best_eff > -1e8 {
                            let sc32 = best_eff as f32;
                            if sc32 > raw_best_sc {
                                raw_best_sc = sc32;
                                raw_best_m = m as i32;
                            }
                        }
                    }

                    if raw_best_m >= 0 && (local_top.len() < top_k || raw_best_sc > min_sc) {
                        local_top.push((raw_best_sc, raw_best_m));
                        if local_top.len() > top_k {
                            local_top.sort_unstable_by(|a, b| {
                                b.0.partial_cmp(&a.0)
                                    .unwrap_or(std::cmp::Ordering::Equal)
                                    .then_with(|| a.1.cmp(&b.1))
                            });
                            local_top.truncate(top_k);
                            min_sc = local_top.last().unwrap().0;
                        }
                    }
                }
                local_top.sort_unstable_by(|a, b| {
                    b.0.partial_cmp(&a.0)
                        .unwrap_or(std::cmp::Ordering::Equal)
                        .then_with(|| a.1.cmp(&b.1))
                });
                local_top
            })
            .reduce(Vec::new, |mut a, mut b| {
                a.append(&mut b);
                a.sort_unstable_by(|x, y| {
                    y.0.partial_cmp(&x.0)
                        .unwrap_or(std::cmp::Ordering::Equal)
                        .then_with(|| x.1.cmp(&y.1))
                });
                if a.len() > top_k {
                    a.truncate(top_k);
                }
                a
            })
    });

    let n_found = winners.len().min(top_k);
    for i in 0..n_found {
        out_top_score[i] = winners[i].0;
        out_top_idx[i] = winners[i].1;
    }
    n_found
}

#[no_mangle]
pub unsafe extern "C" fn sf_filter_helper_cands(
    part_idx_ptr: *const i32,
    n_cands: usize,
    fp_i8_ptr: *const i8,
    fp_scale_ptr: *const f32,
    resid_scale_ptr: *const f32,
    col_sgn_ptr: *const i8,
    d_fp: usize,
    ic_all_ptr: *const f32,
    n_t: usize,
    sq_fp_ptr: *const f32,
    sq_norm_sq: f64,
    sq_ic_clean_ptr: *const f32,
    other_fps_ptr: *const f32,
    other_proxy_uics_ptr: *const f32,
    n_locked: usize,
    inter_max_corr: f64,
    inter_mean_corr: f64,
    proxy_uic_cap: f64,
    out_ok1_ptr: *mut u8,
    out_ok05_ptr: *mut u8,
) {
    if n_cands == 0 {
        return;
    }
    let part_idx = std::slice::from_raw_parts(part_idx_ptr, n_cands);
    let sq_fp = std::slice::from_raw_parts(sq_fp_ptr, d_fp);
    let sq_ic_clean = std::slice::from_raw_parts(sq_ic_clean_ptr, n_t);
    let out_ok1 = std::slice::from_raw_parts_mut(out_ok1_ptr, n_cands);
    let out_ok05 = std::slice::from_raw_parts_mut(out_ok05_ptr, n_cands);

    let p_fp_i8 = fp_i8_ptr as usize;
    let p_fp_sc = fp_scale_ptr as usize;
    let p_res_sc = resid_scale_ptr as usize;
    let p_col_sgn = col_sgn_ptr as usize;
    let p_ic_all = ic_all_ptr as usize;

    let other_fps = if n_locked > 0 && !other_fps_ptr.is_null() {
        std::slice::from_raw_parts(other_fps_ptr, n_locked * d_fp)
    } else {
        &[][..]
    };
    let other_uics = if n_locked > 0 && !other_proxy_uics_ptr.is_null() {
        std::slice::from_raw_parts(other_proxy_uics_ptr, n_locked * n_t)
    } else {
        &[][..]
    };

    let base_cs_dots: Vec<f64> = (0..n_locked)
        .map(|l| {
            let row = &other_fps[l * d_fp..(l + 1) * d_fp];
            let mut acc = 0.0f64;
            for k in 0..d_fp {
                acc += (row[k] as f64) * (sq_fp[k] as f64);
            }
            acc
        })
        .collect();

    pool().install(|| {
        out_ok1
            .par_iter_mut()
            .zip(out_ok05.par_iter_mut())
            .enumerate()
            .for_each(|(ci, (o1, o05))| {
                *o1 = 0;
                *o05 = 0;
                let m_c = part_idx[ci] as usize;
                let sgn_c = *(p_col_sgn as *const i8).add(m_c) as f64;
                let fp_sc_c = *(p_fp_sc as *const f32).add(m_c) as f64;
                let sig_c = *(p_res_sc as *const f32).add(m_c) as f64;
                if fp_sc_c <= 0.0 || sig_c <= 0.0 {
                    return;
                }
                let fp_row = std::slice::from_raw_parts((p_fp_i8 as *const i8).add(m_c * d_fp), d_fp);
                let u_scale = sgn_c * fp_sc_c;

                let mut u_norm_sq = 0.0f64;
                let mut sq_dot_u = 0.0f64;
                for k in 0..d_fp {
                    let uk = (fp_row[k] as f64) * u_scale;
                    u_norm_sq += uk * uk;
                    sq_dot_u += (sq_fp[k] as f64) * uk;
                }
                let nm1_sq = sq_norm_sq + sig_c * sig_c * u_norm_sq + 2.0 * sig_c * sq_dot_u;
                let nm05_sq = sq_norm_sq + 0.25 * sig_c * sig_c * u_norm_sq + 1.0 * sig_c * sq_dot_u;
                if nm1_sq <= 1e-16 || nm05_sq <= 1e-16 {
                    return;
                }
                let inv_nm1 = 1.0 / nm1_sq.sqrt();
                let inv_nm05 = 1.0 / nm05_sq.sqrt();

                let mut pass1 = true;
                let mut pass05 = true;

                if n_locked > 0 {
                    let mut max1 = 0.0f64;
                    let mut sum1 = 0.0f64;
                    let mut max05 = 0.0f64;
                    let mut sum05 = 0.0f64;

                    for l in 0..n_locked {
                        let orow = &other_fps[l * d_fp..(l + 1) * d_fp];
                        let mut odot_i8 = 0.0f32;
                        for k in 0..d_fp {
                            odot_i8 += orow[k] * (fp_row[k] as f32);
                        }
                        let odot_u = (odot_i8 as f64) * u_scale;
                        let c1 = ((base_cs_dots[l] + sig_c * odot_u) * inv_nm1).abs();
                        let c05 = ((base_cs_dots[l] + 0.5 * sig_c * odot_u) * inv_nm05).abs();
                        if c1 > max1 {
                            max1 = c1;
                        }
                        if c05 > max05 {
                            max05 = c05;
                        }
                        sum1 += c1;
                        sum05 += c05;
                    }
                    if max1 > inter_max_corr || (n_locked >= 5 && (sum1 / (n_locked as f64)) > inter_mean_corr) {
                        pass1 = false;
                    }
                    if max05 > inter_max_corr || (n_locked >= 5 && (sum05 / (n_locked as f64)) > inter_mean_corr) {
                        pass05 = false;
                    }
                    if !pass1 && !pass05 {
                        return;
                    }

                    let ic_row = std::slice::from_raw_parts((p_ic_all as *const f32).add(m_c * n_t), n_t);
                    let w_signed = (sgn_c * sig_c) as f32;

                    let check_uic = |w_mult: f32| -> bool {
                        let mut comb = vec![0.0f32; n_t];
                        let mut fin_mask = vec![false; n_t];
                        let mut sum_v = 0.0f64;
                        let mut cnt_v = 0usize;
                        for t in 0..n_t {
                            let a = sq_ic_clean[t];
                            let b = ic_row[t];
                            let fa = a.is_finite();
                            let fb = b.is_finite();
                            if fa || fb {
                                let v = (if fa { a } else { 0.0 }) + w_mult * (if fb { b * w_signed } else { 0.0 });
                                comb[t] = v;
                                fin_mask[t] = true;
                                sum_v += v as f64;
                                cnt_v += 1;
                            }
                        }
                        if cnt_v < 10 {
                            return false;
                        }
                        let mean_v = (sum_v / (cnt_v as f64)) as f32;
                        let mut norm_sq = 0.0f64;
                        for t in 0..n_t {
                            if fin_mask[t] {
                                let d = comb[t] - mean_v;
                                comb[t] = d;
                                norm_sq += (d as f64) * (d as f64);
                            } else {
                                comb[t] = 0.0;
                            }
                        }
                        if norm_sq <= 1e-18 {
                            return false;
                        }
                        let inv_nm = (1.0 / norm_sq.sqrt()) as f32;
                        for t in 0..n_t {
                            comb[t] *= inv_nm;
                        }
                        for l in 0..n_locked {
                            let urow = &other_uics[l * n_t..(l + 1) * n_t];
                            let mut dot = 0.0f32;
                            for t in 0..n_t {
                                dot += urow[t] * comb[t];
                            }
                            if (dot.abs() as f64) > proxy_uic_cap {
                                return false;
                            }
                        }
                        true
                    };

                    if pass1 && !check_uic(1.0) {
                        pass1 = false;
                    }
                    if pass05 && !check_uic(0.5) {
                        pass05 = false;
                    }
                }

                if pass1 {
                    *o1 = 1;
                }
                if pass05 {
                    *o05 = 1;
                }
            });
    });
}

#[no_mangle]
pub unsafe extern "C" fn sf_combine_squads_from_uniq(
    u_uniq_ptr: *const f32,
    l_uniq: usize,
    t_len: usize,
    n_stocks: usize,
    sq_offsets_ptr: *const i32,
    sq_u_idx_ptr: *const i32,
    sq_weights_ptr: *const f32,
    n_sq: usize,
    out_raw_k46_ptr: *mut f32,
) {
    if n_sq == 0 || t_len == 0 {
        return;
    }
    let tn = t_len * n_stocks;
    let nnz = (*sq_offsets_ptr.add(n_sq)) as usize;
    let u_uniq = std::slice::from_raw_parts(u_uniq_ptr, l_uniq * tn);
    let sq_offsets = std::slice::from_raw_parts(sq_offsets_ptr, n_sq + 1);
    let sq_u_idx = std::slice::from_raw_parts(sq_u_idx_ptr, nnz);
    let sq_weights = std::slice::from_raw_parts(sq_weights_ptr, nnz);
    let out_raw_k46 = std::slice::from_raw_parts_mut(out_raw_k46_ptr, (n_sq + 1) * tn);

    let (squad_part, comp_part) = out_raw_k46.split_at_mut(n_sq * tn);

    pool().install(|| {
        squad_part
            .par_chunks_mut(n_stocks)
            .enumerate()
            .for_each(|(row_id, out_row)| {
                let g = row_id / t_len;
                let t = row_id % t_len;
                let p0 = sq_offsets[g] as usize;
                let p1 = sq_offsets[g + 1] as usize;
                let m_len = p1 - p0;
                if m_len == 1 {
                    let ui = sq_u_idx[p0] as usize;
                    let src = &u_uniq[(ui * t_len + t) * n_stocks..(ui * t_len + t + 1) * n_stocks];
                    out_row.copy_from_slice(src);
                    return;
                }
                let min_need = (m_len + 1) / 2;
                for s in 0..n_stocks {
                    let mut acc = 0.0f32;
                    let mut w2 = 0.0f32;
                    let mut cnt = 0usize;
                    for p in p0..p1 {
                        let ui = sq_u_idx[p] as usize;
                        let v = u_uniq[(ui * t_len + t) * n_stocks + s];
                        if v.is_finite() {
                            let w = sq_weights[p];
                            acc += v * w;
                            w2 += w * w;
                            cnt += 1;
                        }
                    }
                    out_row[s] = if cnt >= min_need {
                        acc / w2.max(1e-6).sqrt()
                    } else {
                        f32::NAN
                    };
                }
            });

        comp_part
            .par_chunks_mut(n_stocks)
            .enumerate()
            .for_each(|(t, out_c_row)| {
                let min_c = n_sq / 2;
                for s in 0..n_stocks {
                    let mut csum = 0.0f32;
                    let mut ccnt = 0usize;
                    for g in 0..n_sq {
                        let v = squad_part[(g * t_len + t) * n_stocks + s];
                        if v.is_finite() {
                            csum += v;
                            ccnt += 1;
                        }
                    }
                    out_c_row[s] = if ccnt >= min_c {
                        csum / (ccnt as f32)
                    } else {
                        f32::NAN
                    };
                }
            });
    });
}

#[no_mangle]
pub unsafe extern "C" fn sf_neutralize_block(
    in_mats_ptr: *const f32,
    k_count: usize,
    t_len: usize,
    n_stocks: usize,
    tradable_ptr: *const u8,
    style_q_full_ptr: *const f32,
    n_basis: usize,
    out_mats_ptr: *mut f32,
) {
    if k_count == 0 || t_len == 0 {
        return;
    }
    let tn = t_len * n_stocks;
    let in_mats = std::slice::from_raw_parts(in_mats_ptr, k_count * tn);
    let tradable = std::slice::from_raw_parts(tradable_ptr, tn);
    let style_q_full = std::slice::from_raw_parts(style_q_full_ptr, t_len * n_basis * n_stocks);
    let out_mats = std::slice::from_raw_parts_mut(out_mats_ptr, k_count * tn);

    pool().install(|| {
        out_mats
            .par_chunks_mut(n_stocks)
            .enumerate()
            .for_each(|(row_id, out_row)| {
                SF_SCRATCH.with(|cell| {
                    let mut sc = cell.borrow_mut();
                    sc.ensure(n_stocks, n_basis, 0);
                    out_row.fill(f32::NAN);

                    let k = row_id / t_len;
                    let t = row_id % t_len;
                    let in_row = &in_mats[(k * t_len + t) * n_stocks..(k * t_len + t + 1) * n_stocks];
                    let tr_row = &tradable[t * n_stocks..(t + 1) * n_stocks];
                    let q_day = &style_q_full[t * n_basis * n_stocks..(t + 1) * n_basis * n_stocks];

                    let sc_ref = &mut *sc;
                    sc_ref.fin_buf.clear();
                    let mut tot_tr = 0usize;
                    for s in 0..n_stocks {
                        if tr_row[s] != 0 {
                            tot_tr += 1;
                            let v = in_row[s];
                            if v.is_finite() {
                                sc_ref.fin_buf.push(f32_order_u32(v));
                            }
                        }
                    }
                    if tot_tr < 50 || sc_ref.fin_buf.len() < 12 {
                        return;
                    }
                    let med_key = if sc_ref.fin_buf.len() == tot_tr {
                        0u32
                    } else {
                        let mid = sc_ref.fin_buf.len() / 2;
                        sc_ref.fin_buf.select_nth_unstable(mid);
                        sc_ref.fin_buf[mid]
                    };

                    sc_ref.u64_keys.clear();
                    for s in 0..n_stocks {
                        sc_ref.rk_f32[s] = 0.0;
                        if tr_row[s] != 0 {
                            let v = in_row[s];
                            let k_hi = if v.is_finite() { f32_order_u32(v) } else { med_key };
                            sc_ref.u64_keys.push(((k_hi as u64) << 32) | (s as u64));
                        }
                    }
                    sc_ref.u64_keys.sort_unstable();
                    let cnt = sc_ref.u64_keys.len();
                    let inv_cnt = 1.0f32 / (cnt as f32);
                    let mut i = 0usize;
                    while i < cnt {
                        let hi0 = sc_ref.u64_keys[i] >> 32;
                        let mut j = i + 1;
                        while j < cnt && (sc_ref.u64_keys[j] >> 32) == hi0 {
                            j += 1;
                        }
                        let rk_pct = (0.5f32 * ((i + 1 + j) as f32)) * inv_cnt;
                        for p in i..j {
                            let s = (sc_ref.u64_keys[p] as u32) as usize;
                            sc_ref.rk_f32[s] = rk_pct;
                        }
                        i = j;
                    }

                    for b in 0..n_basis {
                        let q_row = &q_day[b * n_stocks..(b + 1) * n_stocks];
                        let mut dot = 0.0f32;
                        for s in 0..n_stocks {
                            dot += sc_ref.rk_f32[s] * q_row[s];
                        }
                        sc_ref.proj_c[b] = dot;
                    }

                    sc_ref.e10_f32[..n_stocks].copy_from_slice(&sc_ref.rk_f32[..n_stocks]);
                    for b in 0..n_basis {
                        let c_b = sc_ref.proj_c[b];
                        let q_row = &q_day[b * n_stocks..(b + 1) * n_stocks];
                        for s in 0..n_stocks {
                            sc_ref.e10_f32[s] -= c_b * q_row[s];
                        }
                    }

                    sc_ref.u64_keys.clear();
                    for s in 0..n_stocks {
                        if tr_row[s] != 0 {
                            sc_ref.u64_keys.push(pack_f32_idx(sc_ref.e10_f32[s], s));
                        }
                    }
                    sc_ref.u64_keys.sort_unstable();
                    let mut p0 = 0usize;
                    while p0 < cnt {
                        let hi0 = sc_ref.u64_keys[p0] >> 32;
                        let mut p1 = p0 + 1;
                        while p1 < cnt && (sc_ref.u64_keys[p1] >> 32) == hi0 {
                            p1 += 1;
                        }
                        let res_pct = (0.5f32 * ((p0 + 1 + p1) as f32)) * inv_cnt;
                        for q in p0..p1 {
                            let s = (sc_ref.u64_keys[q] as u32) as usize;
                            out_row[s] = res_pct;
                        }
                        p0 = p1;
                    }
                });
            });
    });
}

#[no_mangle]
pub unsafe extern "C" fn sf_fill_post_composite(
    k47_ptr: *mut f32,
    n_sq: usize,
    t_len: usize,
    n_stocks: usize,
) {
    let tn = t_len * n_stocks;
    let k47 = std::slice::from_raw_parts_mut(k47_ptr, (n_sq + 2) * tn);
    let (head, tail) = k47.split_at_mut((n_sq + 1) * tn);
    let src45 = &head[..n_sq * tn];
    let min_c = n_sq / 2;

    pool().install(|| {
        tail.par_chunks_mut(n_stocks)
            .enumerate()
            .for_each(|(t, dst_row)| {
                for s in 0..n_stocks {
                    let mut sum = 0.0f32;
                    let mut cnt = 0usize;
                    for g in 0..n_sq {
                        let v = src45[(g * t_len + t) * n_stocks + s];
                        if v.is_finite() {
                            sum += v;
                            cnt += 1;
                        }
                    }
                    dst_row[s] = if cnt >= min_c {
                        sum / (cnt as f32)
                    } else {
                        f32::NAN
                    };
                }
            });
    });
}

#[inline]
fn legacy_spearman_corr_u64(
    x: &[f32],
    y: &[f32],
    kx: &mut Vec<u64>,
    ky: &mut Vec<u64>,
    rk_y: &mut Vec<f64>,
) -> f32 {
    let n = x.len();
    if n < 10 {
        return f32::NAN;
    }
    kx.clear();
    ky.clear();
    for i in 0..n {
        kx.push(pack_f32_idx(x[i], i));
        ky.push(pack_f32_idx(y[i], i));
    }
    kx.sort_unstable();
    ky.sort_unstable();
    if rk_y.len() < n {
        rk_y.resize(n, 0.0);
    }
    let nf = n as f64;
    let mean = 0.5 * (nf - 1.0);
    let denom = nf * (nf * nf - 1.0) / 12.0;
    if denom <= 1e-12 {
        return f32::NAN;
    }
    for (r, &kv) in ky.iter().enumerate() {
        let orig_i = (kv as u32) as usize;
        rk_y[orig_i] = (r as f64) - mean;
    }
    let mut dot = 0.0f64;
    for (r, &kv) in kx.iter().enumerate() {
        let orig_i = (kv as u32) as usize;
        dot += ((r as f64) - mean) * rk_y[orig_i];
    }
    (dot / denom) as f32
}

#[no_mangle]
pub unsafe extern "C" fn sf_eval_tail_v4_exact_backtest(
    factor_ptr: *const f32,
    k_count: usize,
    n_dates: usize,
    n_stocks: usize,
    ret_gap_ptr: *const f32,
    ret_sum_gap_ptr: *const f32,
    restrict_ptr: *const f32,
    eff_indices_ptr: *const i32,
    date_size: usize,
    gap: usize,
    portf_num: usize,
    out_top_ret_ptr: *mut f64,
    out_bot_ret_ptr: *mut f64,
    out_mkt_ew_ret_ptr: *mut f64,
    out_ratio_ptr: *mut f64,
    out_ic_every_gap_ptr: *mut f32,
    out_ic_daily_ptr: *mut f32,
) {
    if k_count == 0 || date_size == 0 {
        return;
    }
    let dn = n_dates * n_stocks;
    let factor = std::slice::from_raw_parts(factor_ptr, k_count * dn);
    let ret_gap = std::slice::from_raw_parts(ret_gap_ptr, dn);
    let ret_sum_gap = std::slice::from_raw_parts(ret_sum_gap_ptr, dn);
    let restrict = std::slice::from_raw_parts(restrict_ptr, dn);
    let eff_indices = std::slice::from_raw_parts(eff_indices_ptr, date_size);

    let out_top_ret = std::slice::from_raw_parts_mut(out_top_ret_ptr, k_count * date_size);
    let out_bot_ret = std::slice::from_raw_parts_mut(out_bot_ret_ptr, k_count * date_size);
    let out_mkt_ew_ret = std::slice::from_raw_parts_mut(out_mkt_ew_ret_ptr, k_count * date_size);
    let out_ratio = std::slice::from_raw_parts_mut(out_ratio_ptr, k_count * date_size);
    let out_ic_every_gap = std::slice::from_raw_parts_mut(out_ic_every_gap_ptr, k_count * date_size);
    let out_ic_daily = std::slice::from_raw_parts_mut(out_ic_daily_ptr, k_count * date_size);

    let mut open_counts = vec![0usize; n_dates];
    for d in 0..n_dates {
        let r_row = &restrict[d * n_stocks..(d + 1) * n_stocks];
        let mut c = 0usize;
        for &v in r_row {
            if v.is_finite() && v == 0.0 {
                c += 1;
            }
        }
        open_counts[d] = c;
    }

    pool().install(|| {
        out_top_ret
            .par_chunks_mut(date_size)
            .zip(out_bot_ret.par_chunks_mut(date_size))
            .zip(out_mkt_ew_ret.par_chunks_mut(date_size))
            .zip(out_ratio.par_chunks_mut(date_size))
            .zip(out_ic_every_gap.par_chunks_mut(date_size))
            .zip(out_ic_daily.par_chunks_mut(date_size))
            .with_max_len(1)
            .enumerate()
            .for_each(|(ki, (((((top_c, bot_c), mkt_c), rat_c), icg_c), icd_c))| {
                top_c.fill(0.0);
                bot_c.fill(0.0);
                mkt_c.fill(0.0);
                rat_c.fill(f64::NAN);
                icg_c.fill(f32::NAN);
                icd_c.fill(f32::NAN);

                let f_mat = &factor[ki * dn..(ki + 1) * dn];
                let mut filtered_sig = Vec::<f32>::with_capacity(n_stocks);
                let mut filtered_ret = Vec::<f32>::with_capacity(n_stocks);
                let mut filtered_fut = Vec::<f32>::with_capacity(n_stocks);
                let mut daily_sig = Vec::<f32>::with_capacity(n_stocks);
                let mut daily_fut = Vec::<f32>::with_capacity(n_stocks);
                let mut kx = Vec::<u64>::with_capacity(n_stocks);
                let mut ky = Vec::<u64>::with_capacity(n_stocks);
                let mut ranks = vec![0.0f64; n_stocks];
                let mut group_sums = vec![0.0f64; portf_num];
                let mut group_counts = vec![0usize; portf_num];
                let mut rk_y = Vec::<f64>::with_capacity(n_stocks);

                let mut held_sig_row = (eff_indices[0] as usize).saturating_sub(1);
                let mut held_res_row = held_sig_row;

                for (local_t, &raw_eff_i32) in eff_indices.iter().enumerate() {
                    let raw_eff_idx = raw_eff_i32 as usize;
                    if local_t % gap == 0 {
                        held_sig_row = raw_eff_idx - 1;
                        held_res_row = raw_eff_idx - 1;
                    }

                    let sig_d = raw_eff_idx - 1;
                    let fut_d = sig_d + gap;
                    if fut_d < n_dates {
                        daily_sig.clear();
                        daily_fut.clear();
                        let fs_row = &f_mat[sig_d * n_stocks..(sig_d + 1) * n_stocks];
                        let rs_row = &restrict[sig_d * n_stocks..(sig_d + 1) * n_stocks];
                        let rf_row = &ret_gap[fut_d * n_stocks..(fut_d + 1) * n_stocks];
                        let rsum_row = &ret_sum_gap[fut_d * n_stocks..(fut_d + 1) * n_stocks];
                        for s in 0..n_stocks {
                            let sv = fs_row[s];
                            let rv = rf_row[s];
                            let rsum_v = rsum_row[s];
                            let is_open = rs_row[s].is_finite() && rs_row[s] == 0.0;
                            if sv.is_finite() && rv.is_finite() && rsum_v.is_finite() && is_open {
                                daily_sig.push(sv);
                                daily_fut.push(rsum_v);
                            }
                        }
                        icd_c[local_t] = legacy_spearman_corr_u64(&daily_fut, &daily_sig, &mut kx, &mut ky, &mut rk_y);
                    }

                    filtered_sig.clear();
                    filtered_ret.clear();
                    filtered_fut.clear();
                    let h_sig_slice = &f_mat[held_sig_row * n_stocks..(held_sig_row + 1) * n_stocks];
                    let h_res_slice = &restrict[held_res_row * n_stocks..(held_res_row + 1) * n_stocks];
                    let cur_ret_slice = &ret_gap[raw_eff_idx * n_stocks..(raw_eff_idx + 1) * n_stocks];
                    let cur_rsum_slice = &ret_sum_gap[raw_eff_idx * n_stocks..(raw_eff_idx + 1) * n_stocks];

                    let mut mkt_sum = 0.0f64;
                    for s in 0..n_stocks {
                        let sv = h_sig_slice[s];
                        let rv = cur_ret_slice[s];
                        let is_open = h_res_slice[s].is_finite() && h_res_slice[s] == 0.0;
                        if sv.is_finite() && rv.is_finite() && is_open {
                            filtered_sig.push(sv);
                            filtered_ret.push(rv);
                            filtered_fut.push(cur_rsum_slice[s]);
                            mkt_sum += rv as f64;
                        }
                    }

                    if (local_t + 1) % gap == 0 {
                        icg_c[local_t] = legacy_spearman_corr_u64(&filtered_fut, &filtered_sig, &mut kx, &mut ky, &mut rk_y);
                    }

                    let stocks_num = filtered_sig.len();
                    if stocks_num < portf_num {
                        continue;
                    }
                    mkt_c[local_t] = mkt_sum / (stocks_num as f64);

                    let valid_sym = open_counts[raw_eff_idx - 1];
                    if valid_sym > 0 {
                        rat_c[local_t] = (stocks_num as f64) / (valid_sym as f64);
                    }

                    kx.clear();
                    for (idx, &sv) in filtered_sig.iter().enumerate() {
                        kx.push(pack_f32_idx(sv, idx));
                    }
                    kx.sort_unstable();
                    if ranks.len() < stocks_num {
                        ranks.resize(stocks_num, 0.0);
                    }
                    let mut i = 0usize;
                    while i < stocks_num {
                        let hi0 = kx[i] >> 32;
                        let mut j = i + 1;
                        while j < stocks_num && (kx[j] >> 32) == hi0 {
                            j += 1;
                        }
                        let avg_rk = 0.5 * ((i + 1 + j) as f64);
                        for p in i..j {
                            let orig_i = (kx[p] as u32) as usize;
                            ranks[orig_i] = avg_rk;
                        }
                        i = j;
                    }

                    group_sums.fill(0.0);
                    group_counts.fill(0);
                    let s_num_f = stocks_num as f64;
                    let p_num_f = portf_num as f64;
                    for idx in 0..stocks_num {
                        let pct = ranks[idx] / s_num_f;
                        let mut bucket = (pct * p_num_f).floor() as usize;
                        if bucket >= portf_num {
                            bucket = portf_num - 1;
                        }
                        group_sums[bucket] += filtered_ret[idx] as f64;
                        group_counts[bucket] += 1;
                    }
                    if group_counts[portf_num - 1] > 0 {
                        top_c[local_t] = group_sums[portf_num - 1] / (group_counts[portf_num - 1] as f64);
                    }
                    if group_counts[0] > 0 {
                        bot_c[local_t] = group_sums[0] / (group_counts[0] as f64);
                    }
                }
            });
    });
}
