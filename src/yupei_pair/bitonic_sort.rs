#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn stage<const K: usize, const J: usize>(p: *mut u64) {
    if J >= 8 {
        for i in (0..512).step_by(8) {
            if i & J != 0 {
                continue;
            }
            let a = _mm512_loadu_si512(p.add(i) as *const _);
            let b = _mm512_loadu_si512(p.add(i ^ J) as *const _);
            let lo = _mm512_min_epu64(a, b);
            let hi = _mm512_max_epu64(a, b);
            let (a, b) = if i & K == 0 { (lo, hi) } else { (hi, lo) };
            _mm512_storeu_si512(p.add(i) as *mut _, a);
            _mm512_storeu_si512(p.add(i ^ J) as *mut _, b);
        }
    } else {
        let ix = _mm512_set_epi64(
            (7 ^ J) as i64,
            (6 ^ J) as i64,
            (5 ^ J) as i64,
            (4 ^ J) as i64,
            (3 ^ J) as i64,
            (2 ^ J) as i64,
            (1 ^ J) as i64,
            J as i64,
        );
        let mut mask = 0u8;
        for l in 0..8 {
            if (l & J == 0) == (l & K == 0) {
                mask |= 1 << l;
            }
        }
        for i in (0..512).step_by(8) {
            let a = _mm512_loadu_si512(p.add(i) as *const _);
            let b = _mm512_permutexvar_epi64(ix, a);
            let m = if K >= 8 && i & K != 0 { !mask } else { mask };
            let r = _mm512_mask_blend_epi64(m, _mm512_max_epu64(a, b), _mm512_min_epu64(a, b));
            _mm512_storeu_si512(p.add(i) as *mut _, r);
        }
    }
}
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn network(p: *mut u64) {
    stage::<2, 1>(p);
    stage::<4, 2>(p);
    stage::<4, 1>(p);
    stage::<8, 4>(p);
    stage::<8, 2>(p);
    stage::<8, 1>(p);
    stage::<16, 8>(p);
    stage::<16, 4>(p);
    stage::<16, 2>(p);
    stage::<16, 1>(p);
    stage::<32, 16>(p);
    stage::<32, 8>(p);
    stage::<32, 4>(p);
    stage::<32, 2>(p);
    stage::<32, 1>(p);
    stage::<64, 32>(p);
    stage::<64, 16>(p);
    stage::<64, 8>(p);
    stage::<64, 4>(p);
    stage::<64, 2>(p);
    stage::<64, 1>(p);
    stage::<128, 64>(p);
    stage::<128, 32>(p);
    stage::<128, 16>(p);
    stage::<128, 8>(p);
    stage::<128, 4>(p);
    stage::<128, 2>(p);
    stage::<128, 1>(p);
    stage::<256, 128>(p);
    stage::<256, 64>(p);
    stage::<256, 32>(p);
    stage::<256, 16>(p);
    stage::<256, 8>(p);
    stage::<256, 4>(p);
    stage::<256, 2>(p);
    stage::<256, 1>(p);
    stage::<512, 256>(p);
    stage::<512, 128>(p);
    stage::<512, 64>(p);
    stage::<512, 32>(p);
    stage::<512, 16>(p);
    stage::<512, 8>(p);
    stage::<512, 4>(p);
    stage::<512, 2>(p);
    stage::<512, 1>(p);
}
/// Sorts complete u64 keys, so float score bits and tie-breaking indices are unchanged.
/// Sentinel padding is never part of the returned 500 keys. No floating-point arithmetic.
pub fn sort_keys(keys: &mut [u64], scratch: &mut [u64]) {
    #[cfg(target_arch = "x86_64")]
    if keys.len() == 500 && scratch.len() >= 512 && is_x86_feature_detected!("avx512f") {
        scratch[..500].copy_from_slice(keys);
        scratch[500..512].fill(u64::MAX);
        unsafe {
            network(scratch.as_mut_ptr());
        }
        keys.copy_from_slice(&scratch[..500]);
        return;
    }
    keys.sort_unstable();
}
