//! Local exact-score scan helpers copied from yupei_pair SIMD. No ranking quantization.
#![allow(clippy::too_many_arguments)]
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
#[inline(always)]
pub fn have_avx2() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::is_x86_feature_detected!("avx2")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

pub fn fused_hg_stats(
    hb: &mut [f32],
    gb: &mut [f32],
    wb: f32,
    wc: f32,
    p: &[f32],
    q: &[f32],
    addm: &[f32],
    b: usize,
    c: usize,
) -> (f32, f32, f32, f32, f32, u32) {
    let n = hb.len();
    #[cfg(target_arch = "x86_64")]
    unsafe {
        if have_avx2() {
            let wv = _mm256_set1_ps(wb);
            let cv = _mm256_set1_ps(wc);
            let zero = _mm256_setzero_ps();
            let mut mh = _mm256_set1_ps(f32::NEG_INFINITY);
            let mut mg = _mm256_set1_ps(f32::NEG_INFINITY);
            let mut sh = _mm256_setzero_ps();
            let mut sg = _mm256_setzero_ps();
            let mut sq = _mm256_setzero_ps();
            let mut cnt = 0u32;
            let mut x = 0;
            let mut hbuf = [0f32; 8];
            let mut gbuf = [0f32; 8];
            while x + 8 <= n {
                let (hv, gv);
                if (b >= x && b < x + 8) || (c >= x && c < x + 8) {
                    // 含 b/c 的那块：这两个位置必须是 -inf，标量算这 8 个再按原 lane 累加
                    for i in 0..8 {
                        let xx = x + i;
                        let mut h = wb * p[xx] + wc * q[xx] + addm[xx];
                        let mut g = h - if p[xx] > q[xx] { p[xx] } else { q[xx] };
                        if xx == b || xx == c {
                            h = f32::NEG_INFINITY;
                            g = f32::NEG_INFINITY;
                        }
                        hbuf[i] = h;
                        gbuf[i] = g;
                    }
                    for i in 0..8 {
                        hb[x + i] = hbuf[i];
                        gb[x + i] = gbuf[i];
                    }
                    hv = _mm256_loadu_ps(hbuf.as_ptr());
                    gv = _mm256_loadu_ps(gbuf.as_ptr());
                } else {
                    let a = _mm256_loadu_ps(p.as_ptr().add(x));
                    let bv = _mm256_loadu_ps(q.as_ptr().add(x));
                    let m = _mm256_loadu_ps(addm.as_ptr().add(x));
                    // 不用 fma：与 fill_h 的两次舍入一致
                    hv = _mm256_add_ps(
                        _mm256_add_ps(_mm256_mul_ps(wv, a), _mm256_mul_ps(cv, bv)),
                        m,
                    );
                    gv = _mm256_sub_ps(hv, _mm256_max_ps(a, bv));
                    _mm256_storeu_ps(hb.as_mut_ptr().add(x), hv);
                    _mm256_storeu_ps(gb.as_mut_ptr().add(x), gv);
                }
                mh = _mm256_max_ps(mh, hv);
                mg = _mm256_max_ps(mg, gv);
                sh = _mm256_add_ps(sh, _mm256_max_ps(hv, zero));
                let gp = _mm256_max_ps(gv, zero);
                sg = _mm256_add_ps(sg, gp);
                sq = _mm256_add_ps(sq, _mm256_mul_ps(gp, gp));
                let mm = _mm256_movemask_ps(_mm256_cmp_ps::<_CMP_GT_OQ>(gv, zero)) as u32;
                cnt += mm.count_ones();
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
            return (mx_h, mx_g, s1, s2, s3, cnt);
        }
    }
    let (mut mx_h, mut mx_g) = (f32::NEG_INFINITY, f32::NEG_INFINITY);
    let (mut s1, mut s2, mut s3, mut cnt) = (0f32, 0f32, 0f32, 0u32);
    for x in 0..n {
        let mut h = wb * p[x] + wc * q[x] + addm[x];
        let mut g = h - if p[x] > q[x] { p[x] } else { q[x] };
        if x == b || x == c {
            h = f32::NEG_INFINITY;
            g = f32::NEG_INFINITY;
        }
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
    }
    (mx_h, mx_g, s1, s2, s3, cnt)
}

pub fn mass_acc_hg(
    em: &mut [f32],
    gm: &mut [f32],
    g2m: &mut [f32],
    sm: &mut [f32],
    sgm: &mut [f32],
    sg2m: &mut [f32],
    h: &[f32],
    g: &[f32],
    invh: f32,
    invg: f32,
    do_h: bool,
    do_g: bool,
) {
    let n = h.len();
    #[cfg(target_arch = "x86_64")]
    unsafe {
        if have_avx2() {
            let zero = _mm256_setzero_ps();
            let vh = _mm256_set1_ps(invh);
            let vg = _mm256_set1_ps(invg);
            let mut x = 0;
            while x + 8 <= n {
                let hv = _mm256_loadu_ps(h.as_ptr().add(x));
                let gv = _mm256_loadu_ps(g.as_ptr().add(x));
                if do_h {
                    let m = _mm256_cmp_ps::<_CMP_GT_OQ>(hv, zero);
                    let w = _mm256_mul_ps(hv, vh);
                    let wgg = _mm256_mul_ps(w, gv);
                    let wgg2 = _mm256_mul_ps(wgg, gv);
                    let e0 = _mm256_loadu_ps(em.as_ptr().add(x));
                    let g0 = _mm256_loadu_ps(gm.as_ptr().add(x));
                    let q0 = _mm256_loadu_ps(g2m.as_ptr().add(x));
                    _mm256_storeu_ps(
                        em.as_mut_ptr().add(x),
                        _mm256_add_ps(e0, _mm256_and_ps(w, m)),
                    );
                    _mm256_storeu_ps(
                        gm.as_mut_ptr().add(x),
                        _mm256_add_ps(g0, _mm256_and_ps(wgg, m)),
                    );
                    _mm256_storeu_ps(
                        g2m.as_mut_ptr().add(x),
                        _mm256_add_ps(q0, _mm256_and_ps(wgg2, m)),
                    );
                }
                if do_g {
                    let m = _mm256_cmp_ps::<_CMP_GT_OQ>(gv, zero);
                    let w = _mm256_mul_ps(gv, vg);
                    let wg = _mm256_mul_ps(w, gv);
                    let wg2 = _mm256_mul_ps(wg, gv);
                    let s0 = _mm256_loadu_ps(sm.as_ptr().add(x));
                    let s1 = _mm256_loadu_ps(sgm.as_ptr().add(x));
                    let s2 = _mm256_loadu_ps(sg2m.as_ptr().add(x));
                    _mm256_storeu_ps(
                        sm.as_mut_ptr().add(x),
                        _mm256_add_ps(s0, _mm256_and_ps(w, m)),
                    );
                    _mm256_storeu_ps(
                        sgm.as_mut_ptr().add(x),
                        _mm256_add_ps(s1, _mm256_and_ps(wg, m)),
                    );
                    _mm256_storeu_ps(
                        sg2m.as_mut_ptr().add(x),
                        _mm256_add_ps(s2, _mm256_and_ps(wg2, m)),
                    );
                }
                x += 8;
            }
            while x < n {
                if do_h {
                    let hv = h[x];
                    if hv > 0.0 {
                        let w = hv * invh;
                        let gg = g[x];
                        em[x] += w;
                        gm[x] += w * gg;
                        g2m[x] += w * gg * gg;
                    }
                }
                if do_g {
                    let gv = g[x];
                    if gv > 0.0 {
                        let w = gv * invg;
                        sm[x] += w;
                        sgm[x] += w * gv;
                        sg2m[x] += w * gv * gv;
                    }
                }
                x += 1;
            }
            return;
        }
    }
    for x in 0..n {
        if do_h {
            let hv = h[x];
            if hv > 0.0 {
                let w = hv * invh;
                let gg = g[x];
                em[x] += w;
                gm[x] += w * gg;
                g2m[x] += w * gg * gg;
            }
        }
        if do_g {
            let gv = g[x];
            if gv > 0.0 {
                let w = gv * invg;
                sm[x] += w;
                sgm[x] += w * gv;
                sg2m[x] += w * gv * gv;
            }
        }
    }
}

/// Exact top-K. Sampled thresholds only reduce the candidate sorting set;
/// if that set is too small the full exact selection runs. All X were already
/// scored, and no score is rounded. Genuine ties use canonical stock-code rank.
pub fn select_topk_keys(
    scores: &[f32],
    take: usize,
    code_rank: &[u32],
    kh: &mut [u64],
    mid_buf: &mut [u64],
) {
    let n = scores.len();
    if take == 0 || n == 0 {
        return;
    }
    #[cfg(target_arch = "x86_64")]
    unsafe {
        if have_avx2() && n >= 1024 && take * 4 < n {
            const NSAMP: usize = 128;
            let step = n / NSAMP;
            let mut samp = [0.0f32; NSAMP];
            for s in 0..NSAMP {
                samp[s] = *scores.get_unchecked(s * step);
            }
            let k1 = ((take * NSAMP) / n).saturating_sub(2).max(1);
            let k2 = (((take * NSAMP) / n) + 5).min(NSAMP - 1);
            samp.select_nth_unstable_by(k2, |a, b| b.total_cmp(a));
            let t2 = samp[k2];
            samp[..k2].select_nth_unstable_by(k1, |a, b| b.total_cmp(a));
            let t1 = samp[k1];
            if t2.is_finite() && t1.is_finite() && t1 >= t2 {
                let vt1 = _mm256_set1_ps(t1);
                let vt2 = _mm256_set1_ps(t2);
                let mut n_hi = 0usize;
                let mut n_mid = 0usize;
                let mut x = 0usize;
                while x + 8 <= n {
                    let v = _mm256_loadu_ps(scores.as_ptr().add(x));
                    let m2 = _mm256_movemask_ps(_mm256_cmp_ps::<_CMP_GE_OQ>(v, vt2)) as u32;
                    if m2 != 0 {
                        let m1 = _mm256_movemask_ps(_mm256_cmp_ps::<_CMP_GE_OQ>(v, vt1)) as u32;
                        let mut hi = m1;
                        while hi != 0 {
                            let l = hi.trailing_zeros() as usize;
                            let idx = x + l;
                            *kh.get_unchecked_mut(n_hi) =
                                super::key(*scores.get_unchecked(idx), code_rank[idx]);
                            n_hi += 1;
                            hi &= hi - 1;
                        }
                        let mut md = m2 & !m1;
                        while md != 0 {
                            let l = md.trailing_zeros() as usize;
                            let idx = x + l;
                            *mid_buf.get_unchecked_mut(n_mid) =
                                super::key(*scores.get_unchecked(idx), code_rank[idx]);
                            n_mid += 1;
                            md &= md - 1;
                        }
                    }
                    x += 8;
                }
                while x < n {
                    let sv = *scores.get_unchecked(x);
                    if sv >= t1 {
                        *kh.get_unchecked_mut(n_hi) = super::key(sv, code_rank[x]);
                        n_hi += 1;
                    } else if sv >= t2 {
                        *mid_buf.get_unchecked_mut(n_mid) = super::key(sv, code_rank[x]);
                        n_mid += 1;
                    }
                    x += 1;
                }
                if n_hi >= take {
                    if n_hi > take {
                        kh[..n_hi].select_nth_unstable(take - 1);
                    }
                    kh[..take].sort_unstable();
                    return;
                }
                if n_hi + n_mid >= take {
                    let need = take - n_hi;
                    let mid = &mut mid_buf[..n_mid];
                    if n_mid > need {
                        mid.select_nth_unstable(need - 1);
                    }
                    kh[n_hi..take].copy_from_slice(&mid[..need]);
                    kh[..take].sort_unstable();
                    return;
                }
            }
        }
    }
    for x in 0..n {
        kh[x] = super::key(scores[x], code_rank[x]);
    }
    kh.select_nth_unstable(take - 1);
    kh[..take].sort_unstable();
}
