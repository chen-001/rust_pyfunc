#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn stage<const N: usize, const K: usize, const J: usize>(p: *mut u64) {
    if J >= 8 {
        for i in (0..N).step_by(8) {
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
        for i in (0..N).step_by(8) {
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
unsafe fn network<const N: usize>(p: *mut u64) {
    stage::<N, 2, 1>(p);
    stage::<N, 4, 2>(p);
    stage::<N, 4, 1>(p);
    stage::<N, 8, 4>(p);
    stage::<N, 8, 2>(p);
    stage::<N, 8, 1>(p);
    stage::<N, 16, 8>(p);
    stage::<N, 16, 4>(p);
    stage::<N, 16, 2>(p);
    stage::<N, 16, 1>(p);
    stage::<N, 32, 16>(p);
    stage::<N, 32, 8>(p);
    stage::<N, 32, 4>(p);
    stage::<N, 32, 2>(p);
    stage::<N, 32, 1>(p);
    if N >= 64 {
        stage::<N, 64, 32>(p);
        stage::<N, 64, 16>(p);
        stage::<N, 64, 8>(p);
        stage::<N, 64, 4>(p);
        stage::<N, 64, 2>(p);
        stage::<N, 64, 1>(p);
    }
    if N >= 128 {
        stage::<N, 128, 64>(p);
        stage::<N, 128, 32>(p);
        stage::<N, 128, 16>(p);
        stage::<N, 128, 8>(p);
        stage::<N, 128, 4>(p);
        stage::<N, 128, 2>(p);
        stage::<N, 128, 1>(p);
    }
    if N >= 256 {
        stage::<N, 256, 128>(p);
        stage::<N, 256, 64>(p);
        stage::<N, 256, 32>(p);
        stage::<N, 256, 16>(p);
        stage::<N, 256, 8>(p);
        stage::<N, 256, 4>(p);
        stage::<N, 256, 2>(p);
        stage::<N, 256, 1>(p);
    }
    if N >= 512 {
        stage::<N, 512, 256>(p);
        stage::<N, 512, 128>(p);
        stage::<N, 512, 64>(p);
        stage::<N, 512, 32>(p);
        stage::<N, 512, 16>(p);
        stage::<N, 512, 8>(p);
        stage::<N, 512, 4>(p);
        stage::<N, 512, 2>(p);
        stage::<N, 512, 1>(p);
    }
}
/// Sorts complete u64 keys, so float score bits and tie-breaking indices are unchanged.
/// Sentinel padding is never part of the returned keys. No floating-point arithmetic.
pub fn sort_keys(keys: &mut [u64], scratch: &mut [u64]) {
    #[cfg(target_arch = "x86_64")]
    if keys.len() == 500 && scratch.len() >= 512 && is_x86_feature_detected!("avx512f") {
        scratch[..500].copy_from_slice(keys);
        scratch[500..512].fill(u64::MAX);
        unsafe {
            network::<512>(scratch.as_mut_ptr());
        }
        keys.copy_from_slice(&scratch[..500]);
        return;
    }
    // Three fresh-process microbenchmarks found the 128-lane network faster for
    // 100-key random, score-tied, and partitioned input. The 32-lane candidate
    // lost on 20-key input, so that length continues to use the standard sorter.
    // Read once, outside later hot calls; set before starting a baseline process.
    if keys.len() == 100 {
        static SMALL_SORT_ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if *SMALL_SORT_ENABLED
            .get_or_init(|| std::env::var("YUPEI_SMALL_SORT_BASELINE").as_deref() != Ok("1"))
            && sort_small_keys(keys, scratch)
        {
            return;
        }
    }
    keys.sort_unstable();
}

/// Candidate for the existing 20/100-key top-k callers. Padding only fills scratch;
/// it cannot change the selected set or the complete u64 score/index ordering.
/// The benchmark can exercise the 20-key candidate without enabling it by default.
pub(crate) fn sort_small_keys(keys: &mut [u64], scratch: &mut [u64]) -> bool {
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx512f") {
        let n = match keys.len() {
            20 if scratch.len() >= 32 => 32,
            100 if scratch.len() >= 128 => 128,
            _ => return false,
        };
        // std's sorter is already linear for monotone input. Preserve that case
        // before selecting the fixed network; comparisons use the entire key.
        if keys.len() == 100 {
            if keys.is_sorted() {
                return true;
            }
            if keys.windows(2).all(|pair| pair[0] >= pair[1]) {
                keys.reverse();
                return true;
            }
        }
        scratch[..keys.len()].copy_from_slice(keys);
        scratch[keys.len()..n].fill(u64::MAX);
        unsafe {
            if n == 32 {
                network::<32>(scratch.as_mut_ptr());
            } else {
                network::<128>(scratch.as_mut_ptr());
            }
        }
        keys.copy_from_slice(&scratch[..keys.len()]);
        return true;
    }
    #[allow(unreachable_code)]
    false
}
