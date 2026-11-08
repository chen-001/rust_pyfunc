//! Exact per-pair comparison and optional single-thread mass-only benchmark.
#![allow(dead_code)]
use std::hint::black_box;
use std::time::Instant;
#[path = "../frozen/src/yupei_pair/bitonic_sort.rs"] mod bitonic_sort;
#[path = "../frozen/src/yupei_pair/simd.rs"] mod reference;
#[cfg(not(integrated_mass4))]
#[path = "pair_mass4_candidate.rs"] mod candidate;
#[cfg(integrated_mass4)]
#[path = "../src/yupei_pair_mass_batch.rs"] mod candidate;
// The integrated source uses the crate's original mass helper for its tail.
// Route that unchanged helper to the frozen reference in this standalone test.
#[cfg(integrated_mass4)]
mod yupei_pair {
    pub(crate) mod simd { pub(crate) use crate::reference::mass_acc_hg; }
}
use candidate::PairScale;
type Acc = [Vec<f32>; 6];

fn rng(state: &mut u64) -> u32 {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    (*state >> 32) as u32
}
fn special(i: usize) -> f32 {
    [0.0, -0.0, f32::INFINITY, f32::NEG_INFINITY,
     f32::from_bits(0x7fc00001), f32::from_bits(0xffc12345),
     f32::from_bits(0x7f800001), f32::MAX, -f32::MAX,
     f32::from_bits(1), f32::from_bits(0x80000001), 0.3, -0.7][i % 13]
}
fn make_input(n: usize, batch: usize, pattern: usize, mut seed: u64) -> (Vec<f32>, Vec<f32>) {
    let mut h = Vec::with_capacity(n * batch);
    let mut g = Vec::with_capacity(n * batch);
    for i in 0..n * batch {
        let (a, b) = (rng(&mut seed), rng(&mut seed));
        let (hv, gv) = match pattern {
            1 => (f32::NEG_INFINITY, f32::NEG_INFINITY),
            2 => (special(i), special(i * 3 + 7)),
            3 => (f32::from_bits(a), f32::from_bits(b)),
            _ => ((a % 1_000_000) as f32 / 500_000.0,
                  (b % 1_000_000) as f32 / 500_000.0 - 1.0),
        };
        h.push(hv); g.push(gv);
    }
    (h, g)
}
fn initial(n: usize, pattern: usize) -> Acc {
    std::array::from_fn(|k| (0..n).map(|i| match pattern {
        1 => -0.0,
        2 => (i as f32 - k as f32 * 0.3) / 17.0,
        3 => special(i * 5 + k),
        4 => f32::from_bits((i as u32 % 0x800000) | if k % 2 == 0 { 0 } else { 0x80000000 }),
        _ => 0.0,
    }).collect())
}
fn metadata(batch: usize, flags: u32, pattern: usize) -> Vec<PairScale> {
    (0..batch).map(|p| {
        let (invh, invg) = match pattern {
            1 => (if p % 2 == 0 { -0.0 } else { 0.0 }, -0.0),
            2 => (special(p + 2), special(p + 4)),
            3 => (-0.2 - p as f32 * 0.01, 1e20),
            _ => (0.001 + p as f32 * 0.0003, 0.002 + p as f32 * 0.0007),
        };
        PairScale { invh, invg, do_h: flags & (1 << (2 * p)) != 0,
                    do_g: flags & (1 << (2 * p + 1)) != 0 }
    }).collect()
}
fn old(acc: &mut Acc, h: &[f32], g: &[f32], scales: &[PairScale]) {
    let [em, gm, g2m, sm, sgm, sg2m] = acc;
    let n = em.len();
    for (i, scale) in scales.iter().enumerate() {
        reference::mass_acc_hg(em, gm, g2m, sm, sgm, sg2m,
            &h[i * n..(i + 1) * n], &g[i * n..(i + 1) * n],
            scale.invh, scale.invg, scale.do_h, scale.do_g);
    }
}
fn new(acc: &mut Acc, h: &[f32], g: &[f32], scales: &[PairScale]) {
    let [em, gm, g2m, sm, sgm, sg2m] = acc;
    candidate::mass_acc_hg_batch4(em, gm, g2m, sm, sgm, sg2m, h, g, scales);
}
fn assert_exact(a: &Acc, b: &Acc, label: &str) {
    for (field, (left, right)) in a.iter().zip(b).enumerate() {
        for (x, (&u, &v)) in left.iter().zip(right).enumerate() {
            assert_eq!(u.to_bits(), v.to_bits(), "{label} field={field} x={x}");
        }
    }
}
fn verify_case(n: usize, batch: usize, pattern: usize, init: usize, meta: usize, flags: u32) {
    let (h, g) = make_input(n, batch, pattern, 19);
    let scales = metadata(batch, flags, meta);
    let start = initial(n, init);
    let (mut a, mut b) = (start.clone(), start.clone());
    let label = format!("n={n} batch={batch} pattern={pattern} init={init} meta={meta} flags={flags:x}");
    for _ in 0..2 {
        old(&mut a, &h, &g, &scales);
        new(&mut b, &h, &g, &scales);
        assert_exact(&a, &b, &label);
    }
    let mut repeat = start;
    for _ in 0..2 { new(&mut repeat, &h, &g, &scales); }
    assert_exact(&b, &repeat, "repeat determinism");
}
fn verify() {
    assert!(is_x86_feature_detected!("avx2"), "The reference AVX2 path must be exercised");
    assert!(is_x86_feature_detected!("avx512f"), "This two-backend check requires AVX512F hardware");
    let mut cases = 0;
    for n in [0, 1, 7, 8, 15, 16, 17, 23, 24, 31, 32, 33, 65, 5107] {
        for batch in 1..=4 {
            for pattern in 0..4 {
                for init in 0..5 {
                    for meta in 0..4 {
                        for flags in [0, 0x55, 0xaa, 0xff, 0xa5, 0x96, 0x3c, 0xc3] {
                            verify_case(n, batch, pattern, init, meta, flags);
                            cases += 1;
                        }
                    }
                }
            }
        }
    }
    for flags in 0..256 { verify_case(17, 4, 2, 3, 2, flags); cases += 1; }
    println!("{{\"exact_cases\":{cases},\"six_accumulators_bits_equal\":true,\"repeated_batches_bits_equal\":true,\"avx2_and_avx512f_supported\":true,\"kernel_threads\":1}}");
}
fn benchmark(args: &[String]) {
    let mode = &args[0];
    let access = &args[1];
    let kind = &args[2];
    let iterations: usize = args[3].parse().unwrap();
    assert!(mode == "reference" || mode == "candidate");
    assert!(access == "direct" || access == "staged");
    assert!(kind == "normal" || kind == "mixed_flags");
    let n = 5107;
    let scales = metadata(4, if kind == "normal" { 0xff } else { 0x96 }, 0);
    let samples: Vec<_> = (1..=32).map(|seed| make_input(n, 4, 0, seed)).collect();
    let mut acc = initial(n, 0);
    let mut hbuf = vec![0.0; if mode == "reference" { n } else { 4 * n }];
    let mut gbuf = hbuf.clone();
    let mut call = |i: usize| {
        let (h, g) = black_box(&samples[i % samples.len()]);
        if access == "direct" {
            if mode == "reference" { old(&mut acc, h, g, black_box(&scales)); }
            else { new(&mut acc, h, g, black_box(&scales)); }
        } else if mode == "reference" {
            for pair in 0..4 {
                hbuf.copy_from_slice(&h[pair * n..(pair + 1) * n]);
                gbuf.copy_from_slice(&g[pair * n..(pair + 1) * n]);
                old(&mut acc, black_box(&hbuf), black_box(&gbuf), &scales[pair..pair + 1]);
            }
        } else {
            // Identical producer-copy bytes to the reference staging case.
            for pair in 0..4 {
                hbuf[pair * n..(pair + 1) * n].copy_from_slice(&h[pair * n..(pair + 1) * n]);
                gbuf[pair * n..(pair + 1) * n].copy_from_slice(&g[pair * n..(pair + 1) * n]);
            }
            new(&mut acc, black_box(&hbuf), black_box(&gbuf), &scales);
        }
        black_box(&acc);
    };
    for i in 0..32 { call(i); }
    let timer = Instant::now();
    for i in 0..iterations { call(i); }
    let seconds = timer.elapsed().as_secs_f64();
    println!("{{\"mode\":\"{mode}\",\"access\":\"{access}\",\"kind\":\"{kind}\",\"n\":{n},\"batch\":4,\"iterations\":{iterations},\"seconds\":{seconds:.9},\"kernel_threads\":1}}");
}
fn main() {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.first().map(String::as_str) == Some("--bench") { benchmark(&args[1..]); }
    else { verify(); }
}
