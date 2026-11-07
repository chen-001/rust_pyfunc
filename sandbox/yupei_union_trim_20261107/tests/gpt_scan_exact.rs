//! Compare retained scan results with the frozen, unmodified implementation.
//! rustc --edition=2021 -O -C target-cpu=native tests/gpt_scan_exact.rs -o /tmp/gpt_scan_exact
//! The frozen source is the baseline saved before this optimization session.
#![allow(dead_code)]
use std::hint::black_box;
use std::time::Instant;
mod yupei_gpt {
    pub mod network {
        pub fn key(score: f32, code_rank: u32) -> u64 {
            let b = if score == 0. { 0.0f32.to_bits() } else { score.to_bits() };
            let ascending = if b & 0x8000_0000 != 0 { !b } else { b ^ 0x8000_0000 };
            ((!ascending as u64) << 32) | code_rank as u64
        }
    }
}
#[path = "../src/yupei_gpt/network_simd.rs"] mod optimized;
#[path = "../frozen/src/yupei_gpt/network_simd.rs"] mod frozen;

fn inputs(n: usize, pattern: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut state = 17u64;
    let mut next = || {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((state >> 32) % 100_000) as f32 / 50_000. - 1.
    };
    let mut p = Vec::with_capacity(n);
    let mut q = Vec::with_capacity(n);
    let mut addm = Vec::with_capacity(n);
    for i in 0..n {
        p.push(if pattern == 1 { 0. } else { next() });
        q.push(if pattern == 1 { -0. } else { next() });
        addm.push(if pattern == 2 || (pattern == 3 && i % 3 == 0) { f32::NEG_INFINITY } else { 0. });
    }
    (p, q, addm)
}
fn verify() {
    let mut cases = 0usize;
    for n in [0, 1, 2, 7, 8, 9, 15, 16, 17, 127, 1024, 5107, 8193] {
        for pattern in 0..4 {
            let (p, q, addm) = inputs(n, pattern);
            for (b, c) in [(0, 1), (n / 2, n.saturating_sub(1)), (n, n)] {
                for (wb, wc) in [(0., 0.), (0.3, 0.7), (1., 1.), (0., 1.)] {
                    let (mut h0, mut g0, mut h1, mut g1) = (vec![0.; n], vec![0.; n], vec![0.; n], vec![0.; n]);
                    let old = frozen::fused_hg_stats(&mut h0, &mut g0, wb, wc, &p, &q, &addm, b, c);
                    let new = optimized::fused_hg_stats(&mut h1, &mut g1, wb, wc, &p, &q, &addm, b, c);
                    assert_eq!(old.1.to_bits(), new.0.to_bits());
                    assert_eq!(old.2.to_bits(), new.1.to_bits());
                    assert_eq!(old.3.to_bits(), new.2.to_bits());
                    assert_eq!(old.5, new.3);
                    for (a, b) in h0.iter().chain(&g0).zip(h1.iter().chain(&g1)) {
                        assert_eq!(a.to_bits(), b.to_bits(), "n={n} pattern={pattern}");
                    }
                    cases += 1;
                }
            }
        }
    }
    println!("{{\"exact_cases\":{cases},\"reference\":\"frozen/src/yupei_gpt/network_simd.rs\",\"retained_statistics_bits_equal\":true,\"h_g_arrays_bits_equal\":true}}");
}
fn benchmark() {
    let n = 5107;
    let (p, q, addm) = inputs(n, 0);
    let (mut h, mut g) = (vec![0.; n], vec![0.; n]);
    let iterations = 12_000;
    let mut rows = Vec::new();
    for pass in 0..6 {
        let baseline = pass % 2 == 0;
        let timer = Instant::now();
        for i in 0..iterations {
            if baseline {
                black_box(frozen::fused_hg_stats(&mut h, &mut g, 0.3, 0.7,
                    black_box(&p), black_box(&q), black_box(&addm), i % n, (i + 17) % n));
            } else {
                black_box(optimized::fused_hg_stats(&mut h, &mut g, 0.3, 0.7,
                    black_box(&p), black_box(&q), black_box(&addm), i % n, (i + 17) % n));
            }
            black_box(&h);
            black_box(&g);
        }
        rows.push(format!("{{\"pass\":{pass},\"baseline\":{baseline},\"n\":{n},\"calls\":{iterations},\"elapsed_s\":{:.9}}}", timer.elapsed().as_secs_f64()));
    }
    println!("{{\"scope\":\"single_core_fused_hg_stats_microbenchmark\",\"runs\":[{}]}}", rows.join(","));
}
fn main() {
    if std::env::args().any(|s| s == "--bench") { benchmark(); } else { verify(); }
}
