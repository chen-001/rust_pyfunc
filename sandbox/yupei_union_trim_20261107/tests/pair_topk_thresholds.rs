#![allow(dead_code)]
// Candidate 3 is frozen separately; active sandbox sources are candidate 2.
#[path = "../frozen/candidate3_src/src/yupei_pair/bitonic_sort.rs"]
mod bitonic_sort;
#[path = "../frozen/candidate3_src/src/yupei_pair/simd.rs"]
mod simd;

fn reference(scores: &[f32], take: usize) -> Vec<u64> {
    let mut keys: Vec<_> = scores.iter().enumerate()
        .map(|(i, &v)| simd::pack_desc_idx_asc(v, i as u32)).collect();
    keys.sort_unstable();
    keys.truncate(take);
    keys
}

fn assert_selector_matches(scores: &[f32], take: usize) -> simd::TopkSelectionDiag {
    let n = scores.len();
    let mut keys = vec![0u64; n];
    let mut diagnosed = vec![0u64; n];
    let mut scratch = vec![0u64; n.max(128)];
    let before: Vec<_> = scores.iter().map(|v| v.to_bits()).collect();
    let want = reference(scores, take);
    simd::select_topk_keys(scores, take, &mut keys, &mut scratch);
    let mut diag = simd::TopkSelectionDiag::default();
    simd::select_topk_keys_diagnosed(scores, take, &mut diagnosed, &mut scratch, &mut diag);
    assert_eq!(keys[..take], want, "ordinary n={n} take={take}");
    assert_eq!(diagnosed[..take], want, "diagnosed n={n} take={take}");
    assert_eq!(scores.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), before);
    let enabled = std::env::var("YUPEI_TOPK100_WIDE_FIRST").as_deref() == Ok("1");
    let high_first = std::env::var("YUPEI_TOPK_HIGH_FIRST").as_deref() != Ok("0");
    let eligible = take > 0 && n >= 1024 && take * 4 < n && high_first;
    assert_eq!(diag.calls, 1);
    assert_eq!(diag.high_first_calls, eligible as u64);
    assert_eq!(diag.wide_first_calls, (eligible && take == 100 && enabled) as u64);
    if eligible && take == 100 && enabled { assert_eq!(diag.retry_scans, 0); }
    diag
}

#[test]
fn full_key_order_matches_for_boundaries_ties_nonfinite_and_signed_zero() {
    let edge = 0.00005f32;
    let special = [
        0.0, -0.0, f32::INFINITY, f32::NEG_INFINITY,
        f32::from_bits(0x7fc00001), f32::from_bits(0x7fc12345),
        f32::from_bits(0xffc00001), f32::from_bits(0xffc12345),
        f32::from_bits(1), f32::from_bits(0x80000001),
        edge, f32::from_bits(edge.to_bits()-1), f32::from_bits(edge.to_bits()+1),
        -edge, -f32::from_bits(edge.to_bits()-1), -f32::from_bits(edge.to_bits()+1),
        f32::MAX, -f32::MAX, 1.0, -1.0,
    ];
    for n in [0usize, 1, 7, 20, 99, 100, 101, 400, 401, 1023, 1024, 1025, 5107, 8193] {
        for kind in 0..7 {
            let mut state = 0x815130198u64;
            let scores: Vec<f32> = (0..n).map(|i| {
                state ^= state << 13; state ^= state >> 7; state ^= state << 17;
                match kind {
                    0 => ((state % 100000) as f32 - 50000.0) / 30000.0,
                    1 => (i as f32) * 0.00001,
                    2 => -((i as f32) * 0.00001),
                    3 => 0.25,
                    4 => special[i % special.len()],
                    5 => if i < 3 { 0.25 } else { f32::NEG_INFINITY },
                    _ => f32::from_bits(state as u32),
                }
            }).collect();
            for take in [0usize, 1, 5, 20, 99, 100, n] {
                assert_selector_matches(&scores, take.min(n));
            }
        }
    }
}

#[test]
fn first_shortfall_and_full_fallback_are_counted_without_changing_output() {
    let n = 5107usize;
    let mut scores = vec![-10.0f32; n];
    // All sampled positions dominate the unsampled positions. Neither the old
    // second-best nor the wider eighth-best sampled threshold can contain 100.
    for i in 0..128 {
        scores[i * (n / 128)] = 10.0 + i as f32;
    }
    let diag = assert_selector_matches(&scores, 100);
    if std::env::var("YUPEI_TOPK_HIGH_FIRST").as_deref() != Ok("0") {
        assert_eq!(diag.first_shortfalls, 1);
        assert_eq!(diag.full_key_fallbacks, 1);
        let wide = std::env::var("YUPEI_TOPK100_WIDE_FIRST").as_deref() == Ok("1");
        assert_eq!(diag.first_candidates, if wide { 8 } else { 2 });
        assert_eq!(diag.retry_scans, if wide { 0 } else { 1 });
        assert_eq!(diag.retry_candidates, if wide { 0 } else { 8 });
    }
    println!("TOPK_THRESHOLD_DIAG={diag:?}");
}
