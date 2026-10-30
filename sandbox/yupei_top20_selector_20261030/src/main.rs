#![allow(dead_code)]
use std::{hint::black_box, time::Instant};
#[path = "../../../src/yupei_pair/bitonic_sort.rs"]
mod bitonic_sort;
#[path = "../../../src/yupei_pair/simd.rs"]
mod baseline;
mod candidate;

fn next(rng: &mut u64) -> f32 {
    *rng ^= *rng << 13;
    *rng ^= *rng >> 7;
    *rng ^= *rng << 17;
    (*rng >> 40) as f32 / 16_777_216.0
}

fn samples(n: usize, count: usize) -> Vec<Vec<f32>> {
    let mut rng = 123_456_789_u64;
    (0..count).map(|j| {
        let mut v = (0..n).map(|i| {
            let x = next(&mut rng);
            match j % 8 {
                0 => x * 2.0 - 1.0,
                1 => x * x * x * 0.3,
                2 => ((x - 0.5) * 0.0003).round() * 0.0001,
                3 => ((i as f32) * 0.013 + j as f32).sin() * 0.1 + x * 0.001,
                4 => (x * 30.0).round() * 0.0001 + (next(&mut rng) - 0.5) * 0.00009,
                5 => if x < 0.95 { f32::NEG_INFINITY } else { x },
                6 => -x * 0.002,
                _ => (x - 0.5) * 0.0001,
            }
        }).collect::<Vec<_>>();
        v[j % n] = f32::NEG_INFINITY;
        v[(j * 3 + 1) % n] = f32::NEG_INFINITY;
        v
    }).collect()
}

fn main() {
    let count = std::env::args().nth(1).and_then(|v| v.parse().ok()).unwrap_or(2048);
    let rounds = std::env::args().nth(2).and_then(|v| v.parse().ok()).unwrap_or(10);
    let bench_k = std::env::args().nth(3).and_then(|v| v.parse().ok()).unwrap_or(20);
    let data = samples(5195, count);
    let mut a = vec![0u64; 5195];
    let mut b = a.clone();
    let mut tmp = a.clone();
    let mut corrected = a.clone();
    let mut repaired_mismatch = 0;
    for &k in &[5, 20, 30, 100] {
        for v in &data {
            baseline::select_topk_keys(v, k, &mut a, &mut tmp);
            candidate::select_topk_keys(v, k, &mut b, &mut tmp);
            assert_eq!(&a[..k], &b[..k], "legacy-preserving candidate differs, k={k}");
            if k == 20 {
                for i in 0..v.len() { corrected[i] = baseline::pack_desc_idx_asc(v[i], i as u32); }
                corrected.select_nth_unstable(k - 1);
                corrected[..k].sort_unstable();
                repaired_mismatch += usize::from(a[..k] != corrected[..k]);
            }
        }
    }
    // Small inputs exercise the unchanged fallback; ties/negative zero/NaN/inf are deliberate.
    for n in [1, 7, 31, 128, 1000, 1024, 5195] {
        let mut v = samples(n, 1).pop().unwrap();
        for i in 0..n {
            if i % 11 == 0 { v[i] = -0.0; }
            if i % 13 == 0 { v[i] = f32::INFINITY; }
            if i % 17 == 0 { v[i] = f32::NAN; }
        }
        for k in [0, 1, 5, 20, n].map(|k| k.min(n)) {
            baseline::select_topk_keys(&v, k, &mut a[..n], &mut tmp[..n]);
            candidate::select_topk_keys(&v, k, &mut b[..n], &mut tmp[..n]);
            assert_eq!(&a[..k], &b[..k]);
        }
    }
    println!("{{\"kind\":\"correctness\",\"vectors\":{},\"ks\":[5,20,30,100],\"legacy_mismatches\":0,\"quantized_reference_different_vectors_k20\":{}}}", data.len(), repaired_mismatch);
    for family in 0..8 {
        let mut enough = 0;
        let mut attempted = 0;
        let mut discarded_mid = 0;
        for v in data.iter().skip(family).step_by(8) {
            let mut samp = (0..128).map(|j| v[j * (v.len() / 128)]).collect::<Vec<_>>();
            samp.select_nth_unstable_by(5, |a, b| b.total_cmp(a));
            let t2 = samp[5];
            samp[..5].select_nth_unstable_by(1, |a, b| b.total_cmp(a));
            let t1 = samp[1];
            if t1.is_finite() && t2.is_finite() && t1 >= t2 {
                attempted += 1;
                if v.iter().filter(|&&s| s >= t1).count() >= 20 {
                    enough += 1;
                    discarded_mid += v.iter().filter(|&&s| s >= t2 && s < t1).count();
                }
            }
        }
        println!("{{\"kind\":\"candidates\",\"family\":{},\"eligible_vectors\":{},\"high_sufficient_vectors\":{},\"discarded_mid_packs_avoided\":{}}}", family, attempted, enough, discarded_mid);
    }
    // Interleave implementations; report actual selector wall time, not full-day extrapolation.
    for (trial, faster) in [false, true, true, false].into_iter().enumerate() {
        let start = Instant::now();
        let mut digest = 0u64;
        for _ in 0..rounds {
            for v in &data {
                if faster { candidate::select_topk_keys(black_box(v), bench_k, &mut a, &mut tmp); }
                else { baseline::select_topk_keys(black_box(v), bench_k, &mut a, &mut tmp); }
                digest = digest.wrapping_add(black_box(a[0]));
            }
        }
        println!("{{\"kind\":\"timing\",\"trial\":{},\"k\":{},\"candidate\":{},\"calls\":{},\"seconds\":{},\"digest\":{}}}", trial, bench_k, faster, data.len() * rounds, start.elapsed().as_secs_f64(), digest);
    }
}
