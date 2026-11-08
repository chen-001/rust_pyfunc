//! Standalone exactness check for the 16-score-lane / 8-sum-lane experiment.
//! No Rayon pool or performance measurement is started by this program.
#![allow(dead_code)]

#[path = "../frozen/src/yupei_pair/bitonic_sort.rs"]
mod bitonic_sort;
#[path = "../frozen/src/yupei_pair/simd.rs"]
mod reference;
#[path = "pair_scan16_candidate.rs"]
mod candidate;

type Stats = (f32, f32, f32, f32, f32, u32);

fn stats_bits(s: Stats) -> [u32; 6] {
    [s.0.to_bits(), s.1.to_bits(), s.2.to_bits(), s.3.to_bits(), s.4.to_bits(), s.5]
}

fn next(state: &mut u64) -> u32 {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    (*state >> 32) as u32
}

fn inputs(n: usize, pattern: usize, seed: u64) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let special = [
        0.0, -0.0, f32::INFINITY, f32::NEG_INFINITY,
        f32::from_bits(0x7fc00001), f32::from_bits(0xffc12345),
        f32::from_bits(0x7f800001), f32::from_bits(0xff800001),
        f32::MAX, -f32::MAX, f32::MIN_POSITIVE, -f32::MIN_POSITIVE,
        f32::from_bits(1), f32::from_bits(0x80000001),
        1.0, -1.0, f32::from_bits(1.0f32.to_bits() + 1),
        f32::from_bits(1.0f32.to_bits() - 1), 0.7, -0.7,
    ];
    let mut state = seed;
    let mut p = Vec::with_capacity(n);
    let mut q = Vec::with_capacity(n);
    let mut mask = Vec::with_capacity(n);
    for i in 0..n {
        let a = next(&mut state);
        let b = next(&mut state);
        let positive = |v: u32| (v % 1_000_001) as f32 / 1_000_000.0;
        let (pv, qv) = match pattern {
            0 | 4 | 5 => (positive(a), positive(b)),
            1 => (2.0 * positive(a) - 1.0, 2.0 * positive(b) - 1.0),
            2 => (if i % 2 == 0 { 0.0 } else { -0.0 },
                  if i % 3 == 0 { -0.0 } else { 0.0 }),
            3 => (0.3, 0.3),
            6 => (f32::from_bits(a), f32::from_bits(b)),
            7 => (special[(i + seed as usize) % special.len()],
                  special[(i * 7 + 3 + seed as usize) % special.len()]),
            8 => (f32::from_bits(a % 0x01000000), f32::from_bits(b % 0x01000000)),
            _ => (f32::from_bits(0x3f000000 + a % 16),
                  f32::from_bits(0x3f000000 + b % 16)),
        };
        p.push(pv);
        q.push(qv);
        mask.push(if pattern == 5 || (pattern == 4 && i % 3 == 0) {
            f32::NEG_INFINITY
        } else if pattern == 7 && i % 11 == 0 {
            f32::from_bits(0x7fc12345)
        } else if pattern == 2 && i % 2 == 0 {
            -0.0
        } else {
            0.0
        });
    }
    (p, q, mask)
}

fn assert_case(n: usize, pattern: usize, seed: u64, wb: f32, wc: f32, b: usize, c: usize) {
    let (p, q, mask) = inputs(n, pattern, seed);
    let p_before: Vec<_> = p.iter().map(|v| v.to_bits()).collect();
    let q_before: Vec<_> = q.iter().map(|v| v.to_bits()).collect();
    let m_before: Vec<_> = mask.iter().map(|v| v.to_bits()).collect();
    let (mut h0, mut g0, mut h1, mut g1) =
        (vec![7.0; n], vec![8.0; n], vec![9.0; n], vec![10.0; n]);
    let old = reference::fused_hg_stats(&mut h0, &mut g0, wb, wc, &p, &q, &mask, b, c);
    let new = candidate::fused_hg_stats(&mut h1, &mut g1, wb, wc, &p, &q, &mask, b, c);
    assert_eq!(stats_bits(old), stats_bits(new),
        "stats n={n} pattern={pattern} seed={seed} weights=({:x},{:x}) b={b} c={c}", wb.to_bits(), wc.to_bits());
    for (i, (&old, &new)) in h0.iter().chain(&g0).zip(h1.iter().chain(&g1)).enumerate() {
        assert_eq!(old.to_bits(), new.to_bits(),
            "arrays i={i} n={n} pattern={pattern} seed={seed} weights=({:x},{:x}) b={b} c={c}", wb.to_bits(), wc.to_bits());
    }
    assert_eq!(p_before, p.iter().map(|v| v.to_bits()).collect::<Vec<_>>());
    assert_eq!(q_before, q.iter().map(|v| v.to_bits()).collect::<Vec<_>>());
    assert_eq!(m_before, mask.iter().map(|v| v.to_bits()).collect::<Vec<_>>());
    // Repeat on reused output buffers: both overwrite completeness and exact
    // determinism are part of the check, including all NaN payloads.
    let again = candidate::fused_hg_stats(&mut h1, &mut g1, wb, wc, &p, &q, &mask, b, c);
    assert_eq!(stats_bits(old), stats_bits(again));
    assert!(h0.iter().chain(&g0).zip(h1.iter().chain(&g1))
        .all(|(a, b)| a.to_bits() == b.to_bits()));
}

fn main() {
    assert!(is_x86_feature_detected!("avx2"), "This comparison requires the baseline's AVX2 path");
    assert!(is_x86_feature_detected!("avx512f"), "AVX512F candidate path was not exercised");
    let weights = [
        (0.0, 0.0), (0.3, 0.7), (1.0, 1.0), (0.0, 1.0),
        (-0.7, 0.4), (1e20, 1e20), (f32::INFINITY, 0.0),
        (f32::from_bits(0x7fc00001), 1.0), (f32::from_bits(1), -0.0),
    ];
    let sizes = [0usize, 1, 2, 7, 8, 9, 15, 16, 17, 23, 24, 25, 31, 32, 33,
        63, 64, 65, 127, 128, 129, 255, 256, 257, 1023, 1024, 1025, 5107, 8193];
    let mut cases = 0usize;
    for n in sizes {
        let positions = [(0, 1), (7, 8), (15, 16), (n / 2, n.saturating_sub(1)),
            (n, usize::MAX), (n / 2, n / 2)];
        for pattern in 0..10 {
            for &(wb, wc) in &weights {
                for &(b, c) in &positions {
                    assert_case(n, pattern, 17, wb, wc, b, c);
                    cases += 1;
                }
            }
        }
    }
    // More independent values at the true market length and at SIMD tails.
    for seed in 1..=128u64 {
        for n in [31, 128, 5107] {
            for pattern in [0, 1, 6, 7, 8, 9] {
                let b = (seed as usize * 13) % n;
                let c = (seed as usize * 19 + 7) % n;
                assert_case(n, pattern, seed, 0.637, 0.814, b, c);
                cases += 1;
            }
        }
    }
    println!("{{\"exact_cases\":{cases},\"kernel_threads\":1,\"avx512_path_exercised\":true,\"h_g_arrays_bits_equal\":true,\"all_statistics_bits_equal\":true,\"repeated_calls_bits_equal\":true,\"inputs_unchanged\":true}}");
}
