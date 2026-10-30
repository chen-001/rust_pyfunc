// Standalone sandbox: same thresholds and candidate sets as production, delayed mid work.
use super::baseline::{have_avx2, pack_desc_idx_asc};
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

pub fn select_topk_keys(
    scores: &[f32],
    take: usize,
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
                let mut x = 0usize;
                // Preserve exact legacy t1 candidate set; defer mid candidates until needed.
                while x + 8 <= n {
                    let v = _mm256_loadu_ps(scores.as_ptr().add(x));
                    let mut hi = _mm256_movemask_ps(_mm256_cmp_ps::<_CMP_GE_OQ>(v, vt1)) as u32;
                    while hi != 0 {
                        let l = hi.trailing_zeros() as usize;
                        let idx = x + l;
                        *kh.get_unchecked_mut(n_hi) =
                            pack_desc_idx_asc(*scores.get_unchecked(idx), idx as u32);
                        n_hi += 1;
                        hi &= hi - 1;
                    }
                    x += 8;
                }
                while x < n {
                    let sv = *scores.get_unchecked(x);
                    if sv >= t1 {
                        *kh.get_unchecked_mut(n_hi) = pack_desc_idx_asc(sv, x as u32);
                        n_hi += 1;
                    }
                    x += 1;
                }
                if n_hi >= take {
                    if n_hi > take {
                        kh[..n_hi].select_nth_unstable(take - 1);
                    }
                    super::bitonic_sort::sort_keys(&mut kh[..take], mid_buf);
                    return;
                }
                let mut n_mid = 0usize;
                let mut x = 0usize;
                while x + 8 <= n {
                    let v = _mm256_loadu_ps(scores.as_ptr().add(x));
                    let above_lower = _mm256_cmp_ps::<_CMP_GE_OQ>(v, vt2);
                    let below_upper = _mm256_cmp_ps::<_CMP_LT_OQ>(v, vt1);
                    let mut md = _mm256_movemask_ps(_mm256_and_ps(above_lower, below_upper)) as u32;
                    while md != 0 {
                        let l = md.trailing_zeros() as usize;
                        let idx = x + l;
                        *mid_buf.get_unchecked_mut(n_mid) =
                            pack_desc_idx_asc(*scores.get_unchecked(idx), idx as u32);
                        n_mid += 1;
                        md &= md - 1;
                    }
                    x += 8;
                }
                while x < n {
                    let sv = *scores.get_unchecked(x);
                    if sv < t1 && sv >= t2 {
                        *mid_buf.get_unchecked_mut(n_mid) = pack_desc_idx_asc(sv, x as u32);
                        n_mid += 1;
                    }
                    x += 1;
                }
                if n_hi + n_mid >= take {
                    let need = take - n_hi;
                    let mid = &mut mid_buf[..n_mid];
                    if n_mid > need {
                        mid.select_nth_unstable(need - 1);
                    }
                    kh[n_hi..take].copy_from_slice(&mid[..need]);
                    super::bitonic_sort::sort_keys(&mut kh[..take], mid_buf);
                    return;
                }
            }
        }
    }
    for x in 0..n {
        kh[x] = pack_desc_idx_asc(scores[x], x as u32);
    }
    kh.select_nth_unstable(take - 1);
    super::bitonic_sort::sort_keys(&mut kh[..take], mid_buf);
}
