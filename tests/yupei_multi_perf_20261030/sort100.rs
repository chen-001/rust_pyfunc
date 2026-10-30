// Standalone check: rustc --edition=2021 -O sort100.rs -o /tmp/yupei_sort100
// Uses only the sandbox sorter. No installed library or production source changes.
#[path = "../../sandbox/yupei_multi_perf_20261030/src/yupei_pair/bitonic_sort.rs"]
mod sorter;

use std::hint::black_box;
use std::time::Instant;

fn next(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

fn corpus(kind: usize, n: usize) -> Vec<Vec<u64>> {
    let mut seed = 0xC0FFEE12345678u64;
    (0..1024).map(|_| {
        let mut v: Vec<u64> = (0..n).map(|i| {
            let x = next(&mut seed);
            match kind {
                3 => ((x % 7) << 32) | i as u64, // quantized score ties, unique index
                4 => u64::MAX,                   // duplicate sentinel-valued inputs
                _ => x,
            }
        }).collect();
        if kind == 1 { v.sort_unstable(); }
        if kind == 2 { v.sort_unstable_by(|a, b| b.cmp(a)); }
        v
    }).collect()
}

fn bench(data: &[Vec<u64>], simd: bool, rounds: usize) -> f64 {
    let mut v = data[0].clone();
    let mut scratch = vec![0u64; 512];
    let clock = Instant::now();
    for i in 0..rounds {
        v.copy_from_slice(&data[i % data.len()]);
        if simd { sorter::sort_keys(black_box(&mut v), &mut scratch); }
        else { black_box(&mut v).sort_unstable(); }
        black_box(&v);
    }
    clock.elapsed().as_secs_f64() * 1e9 / rounds as f64
}

fn main() {
    std::env::set_var("YUPEI_SORT100_SIMD", "1");
    println!("avx512f={}", std::is_x86_feature_detected!("avx512f"));
    let labels = ["random", "sorted", "reverse", "quantized_ties", "all_max"];
    // Also exercise unchanged 500 path and fallback sizes.
    let mut checked = 0;
    for n in [0, 1, 5, 20, 99, 100, 101, 500] {
        for kind in 0..labels.len() {
            for mut actual in corpus(kind, n) {
                let mut expected = actual.clone();
                expected.sort_unstable();
                sorter::sort_keys(&mut actual, &mut [0u64; 512]);
                assert_eq!(actual, expected, "n={n} kind={kind}");
                checked += 1;
            }
        }
    }
    println!("exact_cases={checked}");
    for (kind, label) in labels.iter().enumerate() {
        let data = corpus(kind, 100);
        let mut baseline = Vec::new();
        let mut candidate = Vec::new();
        bench(&data, false, 5000);
        bench(&data, true, 5000);
        for round in 0..5 {
            // Alternate order to avoid consistently giving one version warm caches.
            if round % 2 == 0 {
                baseline.push(bench(&data, false, 100000));
                candidate.push(bench(&data, true, 100000));
            } else {
                candidate.push(bench(&data, true, 100000));
                baseline.push(bench(&data, false, 100000));
            }
        }
        baseline.sort_by(f64::total_cmp);
        candidate.sort_by(f64::total_cmp);
        println!("{label}: baseline_ns={:.1} simd_ns={:.1} speedup={:.3}",
                 baseline[2], candidate[2], baseline[2] / candidate[2]);
    }
}
