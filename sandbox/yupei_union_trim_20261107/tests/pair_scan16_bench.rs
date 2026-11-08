//! Single-core kernel timing; this is not whole-market factor acceptance.
#![allow(dead_code)]
use std::hint::black_box;
use std::time::Instant;
#[path = "../frozen/src/yupei_pair/bitonic_sort.rs"]
mod bitonic_sort;
#[path = "../frozen/src/yupei_pair/simd.rs"]
mod reference;
#[path = "pair_scan16_candidate.rs"]
mod candidate;

fn bits(s: (f32, f32, f32, f32, f32, u32)) -> [u32; 6] {
    [s.0.to_bits(), s.1.to_bits(), s.2.to_bits(), s.3.to_bits(), s.4.to_bits(), s.5]
}

fn main() {
    assert!(is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx2"));
    let args: Vec<_> = std::env::args().collect();
    let mode = args.get(1).expect("reference or candidate");
    let kind = args.get(2).expect("positive, signed, or boundary");
    let iterations: usize = args.get(3).expect("iterations").parse().unwrap();
    assert!(iterations > 0);
    assert!(mode == "reference" || mode == "candidate");
    assert!(["positive", "signed", "boundary"].contains(&kind.as_str()));
    let n = 5107usize;
    let mut seed = 19u64;
    let mut next = || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (seed >> 32) as u32
    };
    let special = [0.0, -0.0, f32::INFINITY, f32::NEG_INFINITY,
        f32::from_bits(0x7fc00001), f32::from_bits(1),
        f32::from_bits(0x80000001), 0.5, f32::from_bits(0x3f000001)];
    let inputs: Vec<_> = (0..32).map(|sample| {
        let mut p = Vec::with_capacity(n);
        let mut q = Vec::with_capacity(n);
        let mut addm = Vec::with_capacity(n);
        for i in 0..n {
            let (a, b) = (next(), next());
            let pv = (a % 1_000_000) as f32 / 1_000_000.0;
            let qv = (b % 1_000_000) as f32 / 1_000_000.0;
            let (pv, qv) = match kind.as_str() {
                "signed" => (pv * 2.0 - 1.0, qv * 2.0 - 1.0),
                "boundary" if i % 7 == 0 =>
                    (special[(i + sample) % special.len()], special[(i * 3 + sample) % special.len()]),
                "boundary" => (f32::from_bits(0x3f000000 + a % 32), f32::from_bits(0x3f000000 + b % 32)),
                _ => (pv, qv),
            };
            p.push(pv);
            q.push(qv);
            addm.push(if i % 37 == 0 { f32::NEG_INFINITY } else { 0.0 });
        }
        (p, q, addm)
    }).collect();
    let (mut h, mut g, mut old_h, mut old_g) =
        (vec![0.0; n], vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    for (i, (p, q, addm)) in inputs.iter().enumerate() {
        let b = i * 17 % n;
        let c = (i * 31 + 7) % n;
        let old = reference::fused_hg_stats(&mut old_h, &mut old_g, 0.637, 0.814, p, q, addm, b, c);
        let new = candidate::fused_hg_stats(&mut h, &mut g, 0.637, 0.814, p, q, addm, b, c);
        assert_eq!(bits(old), bits(new));
        assert!(old_h.iter().chain(&old_g).zip(h.iter().chain(&g)).all(|(a, b)| a.to_bits() == b.to_bits()));
    }
    let run = |h: &mut [f32], g: &mut [f32], i: usize| {
        let (p, q, addm) = &inputs[i % inputs.len()];
        let b = i * 17 % n;
        let c = (i * 31 + 7) % n;
        if mode == "reference" {
            black_box(reference::fused_hg_stats(h, g, 0.637, 0.814,
                black_box(p), black_box(q), black_box(addm), b, c));
        } else {
            black_box(candidate::fused_hg_stats(h, g, 0.637, 0.814,
                black_box(p), black_box(q), black_box(addm), b, c));
        }
        black_box(h);
        black_box(g);
    };
    for i in 0..128 { run(&mut h, &mut g, i); }
    let timer = Instant::now();
    for i in 0..iterations { run(&mut h, &mut g, i); }
    let seconds = timer.elapsed().as_secs_f64();
    println!("{{\"mode\":\"{mode}\",\"kind\":\"{kind}\",\"n\":{n},\"input_samples\":{},\"iterations\":{iterations},\"seconds\":{seconds:.9},\"kernel_threads\":1,\"same_inputs_bits_equal\":true}}", inputs.len());
}
