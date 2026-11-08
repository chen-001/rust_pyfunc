//! Exact pair-network scan retaining only statistics consumed by net_one_mat.
//! h/g sanitization and original eight-lane sums are unchanged. The old caller
//! only checks max(h) != -inf and discards max(g); the finite mask is sufficient.
//! Standalone candidate only; keep the original general-purpose API untouched.
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

pub fn fused_hg_stats_no_max(
    hb: &mut [f32],
    gb: &mut [f32],
    wb: f32,
    wc: f32,
    p: &[f32],
    q: &[f32],
    addm: &[f32],
    b: usize,
    c: usize,
) -> (bool, f32, f32, f32, u32) {
    let n = hb.len();
    #[cfg(target_arch = "x86_64")]
    unsafe {
        if is_x86_feature_detected!("avx2") {
            let wv = _mm256_set1_ps(wb);
            let cv = _mm256_set1_ps(wc);
            let zero = _mm256_setzero_ps();
            let mut any_mask = _mm256_setzero_ps();
            let mut sh = _mm256_setzero_ps();
            let mut sg = _mm256_setzero_ps();
            let mut sq = _mm256_setzero_ps();
            let mut cnt = 0u32;
            let mut x = 0;
            let mut hbuf = [0f32; 8];
            let mut gbuf = [0f32; 8];
            while x + 8 <= n {
                let (mut hv, mut gv);
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
                }
                let sign = _mm256_set1_ps(-0.0);
                let inf = _mm256_set1_ps(f32::INFINITY);
                let ok_h = _mm256_cmp_ps::<_CMP_LT_OQ>(_mm256_andnot_ps(sign, hv), inf);
                let ok_g = _mm256_cmp_ps::<_CMP_LT_OQ>(_mm256_andnot_ps(sign, gv), inf);
                let ok = _mm256_and_ps(ok_h, ok_g);
                any_mask = _mm256_or_ps(any_mask, ok);
                let excluded = _mm256_set1_ps(f32::NEG_INFINITY);
                hv = _mm256_blendv_ps(excluded, hv, ok);
                gv = _mm256_blendv_ps(excluded, gv, ok);
                _mm256_storeu_ps(hb.as_mut_ptr().add(x), hv);
                _mm256_storeu_ps(gb.as_mut_ptr().add(x), gv);
                sh = _mm256_add_ps(sh, _mm256_max_ps(hv, zero));
                let gp = _mm256_max_ps(gv, zero);
                sg = _mm256_add_ps(sg, gp);
                sq = _mm256_add_ps(sq, _mm256_mul_ps(gp, gp));
                let mm = _mm256_movemask_ps(_mm256_cmp_ps::<_CMP_GT_OQ>(gv, zero)) as u32;
                cnt += mm.count_ones();
                x += 8;
            }
            let mut any_valid = _mm256_movemask_ps(any_mask) != 0;
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
                let valid = h.is_finite() && g.is_finite();
                any_valid |= valid;
                if !valid { h=f32::NEG_INFINITY; g=f32::NEG_INFINITY; }
                hb[x] = h;
                gb[x] = g;
                if h > 0.0 {
                    s1 += h;
                }
                if g > 0.0 {
                    s2 += g;
                    s3 += g * g;
                    cnt += 1;
                }
                x += 1;
            }
            return (any_valid, s1, s2, s3, cnt);
        }
    }
    let mut any_valid = false;
    let (mut s1, mut s2, mut s3, mut cnt) = (0f32, 0f32, 0f32, 0u32);
    for x in 0..n {
        let mut h = wb * p[x] + wc * q[x] + addm[x];
        let mut g = h - if p[x] > q[x] { p[x] } else { q[x] };
        if x == b || x == c {
            h = f32::NEG_INFINITY;
            g = f32::NEG_INFINITY;
        }
        let valid = h.is_finite() && g.is_finite();
        any_valid |= valid;
        if !valid { h=f32::NEG_INFINITY; g=f32::NEG_INFINITY; }
        hb[x] = h;
        gb[x] = g;
        if h > 0.0 {
            s1 += h;
        }
        if g > 0.0 {
            s2 += g;
            s3 += g * g;
            cnt += 1;
        }
    }
    (any_valid, s1, s2, s3, cnt)
}

