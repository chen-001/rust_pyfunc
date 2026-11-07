#![allow(dead_code)]
// Standalone Top100 selector benchmark. The Python runner alternates separate
// old/new processes because policy environment flags are cached by OnceLock.
// Candidate 3 is frozen separately; active sandbox sources are candidate 2.
#[path = "../frozen/candidate3_src/src/yupei_pair/bitonic_sort.rs"]
mod bitonic_sort;
#[path = "../frozen/candidate3_src/src/yupei_pair/simd.rs"]
mod simd;

fn main() {
    let args: Vec<_> = std::env::args().collect();
    let kind = args.get(1).map(String::as_str).unwrap_or("uniform_positive");
    let samples: usize = args.get(2).map(|v| v.parse().unwrap()).unwrap_or(64);
    let iterations: usize = args.get(3).map(|v| v.parse().unwrap()).unwrap_or(1024);
    assert!(samples > 0 && iterations > 0);
    let n = 5107usize;
    let take = 100usize;
    let mut state = 0x815130198u64;
    let scores: Vec<Vec<f32>> = (0..samples).map(|sample| {
        (0..n).map(|i| {
            state ^= state << 13; state ^= state >> 7; state ^= state << 17;
            let u = (state as u32 as f32) / u32::MAX as f32;
            match kind {
                "uniform_positive" => u,
                "uniform_signed" => 0.1 * u - 0.05,
                "repeated_32" => ((state % 32) as f32 - 16.0) * 0.0001,
                "narrow_near_ties" => 0.25 + (u - 0.5) * 0.0004,
                "sample_heavy_fallback" => {
                    if i % (n / 128) == 0 && i / (n / 128) < 128 {
                        10.0 + i as f32 + sample as f32 * 0.001
                    } else { -10.0 + u }
                }
                _ => panic!("unknown distribution {kind}"),
            }
        }).collect()
    }).collect();
    let mut keys = vec![0u64; n];
    let mut scratch = vec![0u64; n];
    let mut diag = simd::TopkSelectionDiag::default();
    let mut first_last = Vec::with_capacity(samples);
    // Exact reference comparisons and counters are outside the timed loop.
    // This pass also initializes policy/CPU dispatch and warms every input.
    for row in &scores {
        let mut want: Vec<u64> = row.iter().enumerate()
            .map(|(i, &v)| simd::pack_desc_idx_asc(v, i as u32)).collect();
        want.sort_unstable();
        simd::select_topk_keys_diagnosed(row, take, &mut keys, &mut scratch, &mut diag);
        assert_eq!(keys[..take], want[..take]);
        simd::select_topk_keys(row, take, &mut keys, &mut scratch);
        assert_eq!(keys[..take], want[..take]);
        first_last.push(keys[0] ^ keys[take - 1]);
    }
    let expected_checksum = (0..iterations).fold(0u64, |acc, i| acc.wrapping_add(first_last[i % samples]));
    let started = std::time::Instant::now();
    let mut checksum = 0u64;
    for i in 0..iterations {
        simd::select_topk_keys(std::hint::black_box(&scores[i % samples]), take, &mut keys, &mut scratch);
        checksum = checksum.wrapping_add(std::hint::black_box(keys[0] ^ keys[take - 1]));
    }
    let seconds = started.elapsed().as_secs_f64();
    assert_eq!(checksum, expected_checksum);
    println!(concat!(
        "{{\"measurement\":\"top100_threshold_kernel\",\"kind\":\"{}\",\"n\":{},\"take\":{},",
        "\"input_samples\":{},\"iterations\":{},\"seconds\":{:.9},\"nanoseconds_per_call\":{:.3},",
        "\"complete_keys_exactly_equal\":true,\"diagnostic_pass_timed\":false,",
        "\"diagnostics\":{{\"calls\":{},\"high_first_calls\":{},\"wide_first_calls\":{},",
        "\"first_shortfalls\":{},\"retry_scans\":{},\"full_key_fallbacks\":{},",
        "\"first_candidates\":{},\"retry_candidates\":{}}}}}"
    ), kind, n, take, samples, iterations, seconds, seconds * 1e9 / iterations as f64,
       diag.calls, diag.high_first_calls, diag.wide_first_calls, diag.first_shortfalls,
       diag.retry_scans, diag.full_key_fallbacks, diag.first_candidates, diag.retry_candidates);
}
