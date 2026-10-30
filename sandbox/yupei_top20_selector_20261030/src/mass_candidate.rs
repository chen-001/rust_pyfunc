#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
pub fn mass_acc_hg_512(
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
        if is_x86_feature_detected!("avx512f") {
            let zero = _mm512_setzero_ps();
            let vh = _mm512_set1_ps(invh);
            let vg = _mm512_set1_ps(invg);
            let mut x = 0;
            while x + 16 <= n {
                let hv = _mm512_loadu_ps(h.as_ptr().add(x));
                let gv = _mm512_loadu_ps(g.as_ptr().add(x));
                if do_h {
                    let m = _mm512_cmp_ps_mask::<_CMP_GT_OQ>(hv, zero);
                    let w = _mm512_mul_ps(hv, vh);
                    let wgg = _mm512_mul_ps(w, gv);
                    let wgg2 = _mm512_mul_ps(wgg, gv);
                    let e0 = _mm512_loadu_ps(em.as_ptr().add(x));
                    let g0 = _mm512_loadu_ps(gm.as_ptr().add(x));
                    let q0 = _mm512_loadu_ps(g2m.as_ptr().add(x));
                    _mm512_storeu_ps(
                        em.as_mut_ptr().add(x),
                        _mm512_add_ps(e0, _mm512_maskz_mov_ps(m, w)),
                    );
                    _mm512_storeu_ps(
                        gm.as_mut_ptr().add(x),
                        _mm512_add_ps(g0, _mm512_maskz_mov_ps(m, wgg)),
                    );
                    _mm512_storeu_ps(
                        g2m.as_mut_ptr().add(x),
                        _mm512_add_ps(q0, _mm512_maskz_mov_ps(m, wgg2)),
                    );
                }
                if do_g {
                    let m = _mm512_cmp_ps_mask::<_CMP_GT_OQ>(gv, zero);
                    let w = _mm512_mul_ps(gv, vg);
                    let wg = _mm512_mul_ps(w, gv);
                    let wg2 = _mm512_mul_ps(wg, gv);
                    let s0 = _mm512_loadu_ps(sm.as_ptr().add(x));
                    let s1 = _mm512_loadu_ps(sgm.as_ptr().add(x));
                    let s2 = _mm512_loadu_ps(sg2m.as_ptr().add(x));
                    _mm512_storeu_ps(
                        sm.as_mut_ptr().add(x),
                        _mm512_add_ps(s0, _mm512_maskz_mov_ps(m, w)),
                    );
                    _mm512_storeu_ps(
                        sgm.as_mut_ptr().add(x),
                        _mm512_add_ps(s1, _mm512_maskz_mov_ps(m, wg)),
                    );
                    _mm512_storeu_ps(
                        sg2m.as_mut_ptr().add(x),
                        _mm512_add_ps(s2, _mm512_maskz_mov_ps(m, wg2)),
                    );
                }
                x += 16;
            }
            super::baseline::mass_acc_hg(
                &mut em[x..], &mut gm[x..], &mut g2m[x..], &mut sm[x..],
                &mut sgm[x..], &mut sg2m[x..], &h[x..], &g[x..], invh, invg, do_h, do_g,
            );
            return;
        }
    }
    super::baseline::mass_acc_hg(em, gm, g2m, sm, sgm, sg2m, h, g, invh, invg, do_h, do_g);
}
