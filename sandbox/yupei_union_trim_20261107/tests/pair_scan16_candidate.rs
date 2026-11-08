//! Independent exact scan experiment: 16 score lanes, original 8 sum lanes.
//! The sibling `reference` module supplies the frozen fallback implementation.
//! This source is used only by standalone checks and kernel microbenchmarks.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

pub fn fused_hg_stats(
    hb: &mut [f32], gb: &mut [f32], wb: f32, wc: f32,
    p: &[f32], q: &[f32], addm: &[f32], b: usize, c: usize,
) -> (f32, f32, f32, f32, f32, u32) {
    #[cfg(target_arch = "x86_64")]
    if hb.len() >= 16 && hb.len() <= i32::MAX as usize
        && is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx2")
    {
        return unsafe { fused_hg_stats_512_keep8(hb, gb, wb, wc, p, q, addm, b, c) };
    }
    super::reference::fused_hg_stats(hb, gb, wb, wc, p, q, addm, b, c)
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn finite_hg_lanes(h: __m256, g: __m256) -> (__m256, __m256) {
    let sign = _mm256_set1_ps(-0.0);
    let inf = _mm256_set1_ps(f32::INFINITY);
    let ok_h = _mm256_cmp_ps::<_CMP_LT_OQ>(_mm256_andnot_ps(sign, h), inf);
    let ok_g = _mm256_cmp_ps::<_CMP_LT_OQ>(_mm256_andnot_ps(sign, g), inf);
    let ok = _mm256_and_ps(ok_h, ok_g);
    let excluded = _mm256_set1_ps(f32::NEG_INFINITY);
    (_mm256_blendv_ps(excluded, h, ok), _mm256_blendv_ps(excluded, g, ok))
}


// Independent experiment only. Baseline copy SHA256:
// 84284220c4b8adbfdfc5fe0edbac6c5e87dc13c33d2eaf5e4fda49f4098dbabc
// Do not replace with sixteen independent sum lanes: that changes rounding.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx2")]
unsafe fn fused_hg_stats_512_keep8(
    hb: &mut [f32], gb: &mut [f32], wb: f32, wc: f32,
    p: &[f32], q: &[f32], addm: &[f32], b: usize, c: usize,
) -> (f32, f32, f32, f32, f32, u32) {
    let n = hb.len();
    let wv = _mm512_set1_ps(wb);
    let cv = _mm512_set1_ps(wc);
    let zero16 = _mm512_setzero_ps();
    let neginf16 = _mm512_set1_ps(f32::NEG_INFINITY);
    let inf16 = _mm512_set1_ps(f32::INFINITY);
    let abs_mask = _mm512_set1_epi32(0x7fff_ffff);
    let bi = _mm512_set1_epi32(if b < n { b as i32 } else { -1 });
    let ci = _mm512_set1_epi32(if c < n { c as i32 } else { -1 });
    let mut ix = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    let step = _mm512_set1_epi32(16);
    let zero = _mm256_setzero_ps();
    let mut mh = _mm256_set1_ps(f32::NEG_INFINITY);
    let mut mg = _mm256_set1_ps(f32::NEG_INFINITY);
    let mut sh = zero;
    let mut sg = zero;
    let mut sq = zero;
    let mut cnt = 0u32;
    let mut x = 0usize;
    while x + 16 <= n {
        let a = _mm512_loadu_ps(p.as_ptr().add(x));
        let bv = _mm512_loadu_ps(q.as_ptr().add(x));
        let m = _mm512_loadu_ps(addm.as_ptr().add(x));
        let mut hv = _mm512_add_ps(
            _mm512_add_ps(_mm512_mul_ps(wv, a), _mm512_mul_ps(cv, bv)), m);
        // MAXPS chooses its second operand for equality and unordered inputs,
        // exactly as the baseline's `if p > q { p } else { q }` branch.
        let mut gv = _mm512_sub_ps(hv, _mm512_max_ps(a, bv));
        let ah = _mm512_castsi512_ps(_mm512_and_epi32(_mm512_castps_si512(hv), abs_mask));
        let ag = _mm512_castsi512_ps(_mm512_and_epi32(_mm512_castps_si512(gv), abs_mask));
        let valid = _mm512_cmp_ps_mask::<_CMP_LT_OQ>(ah, inf16)
            & _mm512_cmp_ps_mask::<_CMP_LT_OQ>(ag, inf16)
            & _mm512_cmpneq_epi32_mask(ix, bi) & _mm512_cmpneq_epi32_mask(ix, ci);
        hv = _mm512_mask_mov_ps(neginf16, valid, hv);
        gv = _mm512_mask_mov_ps(neginf16, valid, gv);
        _mm512_storeu_ps(hb.as_mut_ptr().add(x), hv);
        _mm512_storeu_ps(gb.as_mut_ptr().add(x), gv);
        let hp = _mm512_max_ps(hv, zero16);
        let gp = _mm512_max_ps(gv, zero16);
        let g2 = _mm512_mul_ps(gp, gp);
        // Extract using AVX512F's f64x4 operation (no AVX512DQ requirement).
        let hi_h = _mm256_castpd_ps(_mm512_extractf64x4_pd::<1>(_mm512_castps_pd(hv)));
        let hi_g = _mm256_castpd_ps(_mm512_extractf64x4_pd::<1>(_mm512_castps_pd(gv)));
        let hi_hp = _mm256_castpd_ps(_mm512_extractf64x4_pd::<1>(_mm512_castps_pd(hp)));
        let hi_gp = _mm256_castpd_ps(_mm512_extractf64x4_pd::<1>(_mm512_castps_pd(gp)));
        let hi_g2 = _mm256_castpd_ps(_mm512_extractf64x4_pd::<1>(_mm512_castps_pd(g2)));
        mh = _mm256_max_ps(mh, _mm512_castps512_ps256(hv));
        mg = _mm256_max_ps(mg, _mm512_castps512_ps256(gv));
        sh = _mm256_add_ps(sh, _mm512_castps512_ps256(hp));
        sg = _mm256_add_ps(sg, _mm512_castps512_ps256(gp));
        sq = _mm256_add_ps(sq, _mm512_castps512_ps256(g2));
        mh = _mm256_max_ps(mh, hi_h);
        mg = _mm256_max_ps(mg, hi_g);
        sh = _mm256_add_ps(sh, hi_hp);
        sg = _mm256_add_ps(sg, hi_gp);
        sq = _mm256_add_ps(sq, hi_g2);
        cnt += (_mm512_cmp_ps_mask::<_CMP_GT_OQ>(gv, zero16) as u32).count_ones();
        ix = _mm512_add_epi32(ix, step);
        x += 16;
    }
    // One remaining full eight-lane block stays in the vector reduction before
    // horizontal sums. Moving it to the scalar tail would change sum order.
    if x + 8 <= n {
        let a = _mm256_loadu_ps(p.as_ptr().add(x));
        let bv = _mm256_loadu_ps(q.as_ptr().add(x));
        let m = _mm256_loadu_ps(addm.as_ptr().add(x));
        let hv = _mm256_add_ps(
            _mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(wb), a),
                           _mm256_mul_ps(_mm256_set1_ps(wc), bv)), m);
        let gv = _mm256_sub_ps(hv, _mm256_max_ps(a, bv));
        let (hv, gv) = finite_hg_lanes(hv, gv);
        let ix8 = _mm256_setr_epi32(x as i32, (x+1) as i32, (x+2) as i32, (x+3) as i32,
                                    (x+4) as i32, (x+5) as i32, (x+6) as i32, (x+7) as i32);
        let excluded = _mm256_or_si256(
            _mm256_cmpeq_epi32(ix8, _mm512_castsi512_si256(bi)),
            _mm256_cmpeq_epi32(ix8, _mm512_castsi512_si256(ci)));
        let inf8 = _mm256_set1_ps(f32::NEG_INFINITY);
        let hv = _mm256_blendv_ps(hv, inf8, _mm256_castsi256_ps(excluded));
        let gv = _mm256_blendv_ps(gv, inf8, _mm256_castsi256_ps(excluded));
        _mm256_storeu_ps(hb.as_mut_ptr().add(x), hv);
        _mm256_storeu_ps(gb.as_mut_ptr().add(x), gv);
        mh = _mm256_max_ps(mh, hv);
        mg = _mm256_max_ps(mg, gv);
        sh = _mm256_add_ps(sh, _mm256_max_ps(hv, zero));
        let gp = _mm256_max_ps(gv, zero);
        sg = _mm256_add_ps(sg, gp);
        sq = _mm256_add_ps(sq, _mm256_mul_ps(gp, gp));
        cnt += (_mm256_movemask_ps(_mm256_cmp_ps::<_CMP_GT_OQ>(gv, zero)) as u32).count_ones();
        x += 8;
    }
    let mut b1 = [0f32; 8];
    let mut b2 = [0f32; 8];
    _mm256_storeu_ps(b1.as_mut_ptr(), mh);
    _mm256_storeu_ps(b2.as_mut_ptr(), mg);
    let (mut mx_h, mut mx_g) = (f32::NEG_INFINITY, f32::NEG_INFINITY);
    for i in 0..8 {
        if b1[i] > mx_h {
            mx_h = b1[i];
        }
        if b2[i] > mx_g {
            mx_g = b2[i];
        }
    }
    let mut b3 = [0f32; 8];
    let mut b4 = [0f32; 8];
    let mut b5 = [0f32; 8];
    _mm256_storeu_ps(b3.as_mut_ptr(), sh);
    _mm256_storeu_ps(b4.as_mut_ptr(), sg);
    _mm256_storeu_ps(b5.as_mut_ptr(), sq);
    let (mut s1, mut s2, mut s3) = (0f32, 0f32, 0f32);
    for i in 0..8 {
        s1 += b3[i];
        s2 += b4[i];
        s3 += b5[i];
    }
    while x < n {
        let mut h = wb * p[x] + wc * q[x] + addm[x];
        let mut g = h - if p[x] > q[x] { p[x] } else { q[x] };
        if x == b || x == c {
            h = f32::NEG_INFINITY;
            g = f32::NEG_INFINITY;
        }
        if !h.is_finite() || !g.is_finite() { h=f32::NEG_INFINITY; g=f32::NEG_INFINITY; }
        hb[x] = h;
        gb[x] = g;
        if h > mx_h {
            mx_h = h;
        }
        if h > 0.0 {
            s1 += h;
        }
        if g > mx_g {
            mx_g = g;
        }
        if g > 0.0 {
            s2 += g;
            s3 += g * g;
            cnt += 1;
        }
        x += 1;
    }
    (mx_h, mx_g, s1, s2, s3, cnt)
}

