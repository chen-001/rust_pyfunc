//! Independent exact batching experiment: retain 1--4 pairs' mass updates.
//! Pair-major h/g buffers contain B consecutive n-element vectors. Each stock
//! still receives pair 0,1,2,3 in precisely the original arithmetic order.
//! Only complete 16-lane groups are batched. Original code handles each pair
//! in the remaining AVX2/scalar tail so NaN operand selection is not rewritten.
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[derive(Clone, Copy)]
pub struct PairScale { pub invh: f32, pub invg: f32, pub do_h: bool, pub do_g: bool }

#[allow(clippy::too_many_arguments)]
pub fn mass_acc_hg_batch4(
    em: &mut [f32], gm: &mut [f32], g2m: &mut [f32],
    sm: &mut [f32], sgm: &mut [f32], sg2m: &mut [f32],
    h: &[f32], g: &[f32], scales: &[PairScale],
) {
    let n = em.len();
    assert!(scales.len() <= 4);
    assert!(gm.len() == n && g2m.len() == n && sm.len() == n && sgm.len() == n && sg2m.len() == n);
    assert!(h.len() == n * scales.len() && g.len() == h.len());
    macro_rules! call { ($b:literal) => { mass_batch::<$b>(em, gm, g2m, sm, sgm, sg2m, h, g, scales) }; }
    match scales.len() { 0 => (), 1 => call!(1), 2 => call!(2), 3 => call!(3), 4 => call!(4), _ => unreachable!() }
}

#[allow(clippy::too_many_arguments)]
fn mass_batch<const B: usize>(
    em: &mut [f32], gm: &mut [f32], g2m: &mut [f32],
    sm: &mut [f32], sgm: &mut [f32], sg2m: &mut [f32],
    h: &[f32], g: &[f32], scales: &[PairScale],
) {
    let n = em.len();
    let mut x = 0;
    #[cfg(target_arch = "x86_64")]
    unsafe {
        static MASS_AVX512: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if n >= 16 && *MASS_AVX512.get_or_init(|| {
            std::env::var("YUPEI_MASS_AVX512").as_deref() != Ok("0") && is_x86_feature_detected!("avx512f")
        }) {
            x = mass_lanes16::<B>(em, gm, g2m, sm, sgm, sg2m, h, g, scales, x);
        }
    }
    if x < n {
        for (pair, meta) in scales.iter().enumerate() {
            super::reference::mass_acc_hg(
                &mut em[x..], &mut gm[x..], &mut g2m[x..], &mut sm[x..],
                &mut sgm[x..], &mut sg2m[x..],
                &h[pair * n + x..(pair + 1) * n],
                &g[pair * n + x..(pair + 1) * n],
                meta.invh, meta.invg, meta.do_h, meta.do_g,
            );
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[allow(clippy::too_many_arguments)]
unsafe fn mass_lanes16<const B: usize>(
    em: &mut [f32], gm: &mut [f32], g2m: &mut [f32],
    sm: &mut [f32], sgm: &mut [f32], sg2m: &mut [f32],
    h: &[f32], g: &[f32], scales: &[PairScale], mut x: usize,
) -> usize {
    let n = em.len();
    // The frozen memory-form add prioritizes its increment NaN. LLVM may
    // commute an intrinsic add after register batching, changing only payloads.
    // Preserve the executed operand order explicitly; no rounding/FMA changes.
    macro_rules! add_original_order {
        ($acc:expr, $inc:expr) => {{
            let out: __m512;
            std::arch::asm!(
                "vaddps {out}, {inc}, {acc}",
                out = lateout(zmm_reg) out, inc = in(zmm_reg) $inc, acc = in(zmm_reg) $acc,
                options(pure, nomem, nostack),
            );
            out
        }};
    }
    let zero = _mm512_setzero_ps();
    let vh: [__m512; B] = std::array::from_fn(|i| _mm512_set1_ps(scales[i].invh));
    let vg: [__m512; B] = std::array::from_fn(|i| _mm512_set1_ps(scales[i].invg));
    while x + 16 <= n {
        let mut e = _mm512_loadu_ps(em.as_ptr().add(x));
        let mut gg = _mm512_loadu_ps(gm.as_ptr().add(x));
        let mut gg2 = _mm512_loadu_ps(g2m.as_ptr().add(x));
        let mut s = _mm512_loadu_ps(sm.as_ptr().add(x));
        let mut sg = _mm512_loadu_ps(sgm.as_ptr().add(x));
        let mut sg2 = _mm512_loadu_ps(sg2m.as_ptr().add(x));
        for pair in 0..B {
            let hv = _mm512_loadu_ps(h.as_ptr().add(pair * n + x));
            let gv = _mm512_loadu_ps(g.as_ptr().add(pair * n + x));
            if scales[pair].do_h {
                let mask = _mm512_cmp_ps_mask::<_CMP_GT_OQ>(hv, zero);
                let w = _mm512_mul_ps(hv, vh[pair]);
                let wgg = _mm512_mul_ps(w, gv);
                let wgg2 = _mm512_mul_ps(wgg, gv);
                e = add_original_order!(e, _mm512_maskz_mov_ps(mask, w));
                gg = add_original_order!(gg, _mm512_maskz_mov_ps(mask, wgg));
                gg2 = add_original_order!(gg2, _mm512_maskz_mov_ps(mask, wgg2));
            }
            if scales[pair].do_g {
                let mask = _mm512_cmp_ps_mask::<_CMP_GT_OQ>(gv, zero);
                let w = _mm512_mul_ps(gv, vg[pair]);
                let wg = _mm512_mul_ps(w, gv);
                let wg2 = _mm512_mul_ps(wg, gv);
                s = add_original_order!(s, _mm512_maskz_mov_ps(mask, w));
                sg = add_original_order!(sg, _mm512_maskz_mov_ps(mask, wg));
                sg2 = add_original_order!(sg2, _mm512_maskz_mov_ps(mask, wg2));
            }
        }
        _mm512_storeu_ps(em.as_mut_ptr().add(x), e);
        _mm512_storeu_ps(gm.as_mut_ptr().add(x), gg);
        _mm512_storeu_ps(g2m.as_mut_ptr().add(x), gg2);
        _mm512_storeu_ps(sm.as_mut_ptr().add(x), s);
        _mm512_storeu_ps(sgm.as_mut_ptr().add(x), sg);
        _mm512_storeu_ps(sg2m.as_mut_ptr().add(x), sg2);
        x += 16;
    }
    x
}
