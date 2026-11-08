//! Standalone R5 preparation probe; run through the matching Python driver.
//! The driver extracts the frozen loops/solver/row statistics without rewriting
//! their arithmetic. It replaces only the independent residual-row schedule
//! with sequential iteration, so this test uses one thread and no Rayon crate.
//! This is synthetic same-carrier/(a,b), three-c evidence, not full-day evidence.
use std::hint::black_box;
use std::time::Instant;

mod extracted {
    include!(env!("R5_GENERATED_REFERENCE"));
}
use extracted::{cached_fit, finish_r5, prepare_shared, reference_fit, Fit, R5};

struct Fixture {
    n: usize,
    rho: Vec<f32>,
    lnt: Vec<f64>,
    ok: Vec<bool>,
    // Different responses for each X direction; common rho/lnt/ok is the only
    // sharing premise. Do not use this cache across carriers or different masks.
    responses: [[Vec<f32>; 4]; 3],
}

fn next(seed: &mut u64) -> u32 {
    *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    (*seed >> 32) as u32
}

fn special(k: usize) -> f32 {
    [0.0, -0.0, f32::INFINITY, f32::NEG_INFINITY,
     f32::from_bits(0x7fc00001), f32::from_bits(0xffc12345),
     f32::from_bits(0x7f800001), f32::from_bits(1),
     f32::from_bits(0x80000001), f32::MAX, -f32::MAX][k % 11]
}

fn fixture(n: usize, pattern: usize) -> Fixture {
    let mut seed = 20261108;
    let mut rho = vec![0.0; n * n];
    for i in 0..n {
        rho[i * n + i] = 1.0;
        for j in i + 1..n {
            let r = match pattern {
                4 => special(i * 7 + j),
                6 => [0.0, -0.0, f32::from_bits(1), f32::from_bits(0x80000001)][(i + j) % 4],
                _ => (next(&mut seed) % 16001) as f32 / 10000.0 - 0.8,
            };
            rho[i * n + j] = r;
            rho[j * n + i] = r;
        }
    }
    let lnt = (0..n).map(|i| match pattern {
        2 if i % 3 == 0 => f64::NAN,
        2 if i % 3 == 1 => f64::INFINITY,
        5 => if i % 2 == 0 { f64::NAN } else { f64::NEG_INFINITY },
        6 => if i % 2 == 0 { -0.0 } else { f64::from_bits(1) },
        _ => 1.0 + (next(&mut seed) % 12000) as f64 / 1000.0,
    }).collect();
    let ok = (0..n).map(|i| pattern != 1 || i % 3 != 2).collect();
    let responses = std::array::from_fn(|c| std::array::from_fn(|field| {
        let mut w = vec![0.0; n * n];
        for i in 0..n {
            for j in i + 1..n {
                let y = if pattern == 3 {
                    special(i * 3 + j + c * 5 + field)
                } else {
                    (next(&mut seed) % 20001) as f32 / 7000.0 - 1.0
                        + c as f32 * 0.125 + field as f32 * 0.0625
                };
                w[i * n + j] = y;
                w[j * n + i] = y;
            }
        }
        w
    }));
    Fixture { n, rho, lnt, ok, responses }
}

fn bits(a: &[f64], b: &[f64], label: &str) {
    assert_eq!(a.len(), b.len(), "{label}: length");
    for (i, (&x, &y)) in a.iter().zip(b).enumerate() {
        assert_eq!(x.to_bits(), y.to_bits(), "{label}: index {i}");
    }
}

fn fit_equal(a: &Fit, b: &Fit) {
    assert_eq!(a.npair, b.npair);
    bits(&a.ln, &b.ln, "imputed lnt");
    for i in 0..5 { bits(&a.xtx[i], &b.xtx[i], "XTX"); }
    for i in 0..4 {
        bits(&a.xty[i], &b.xty[i], "XTy");
        bits(&a.betas[i], &b.betas[i], "beta");
    }
}

fn output_equal(a: &[R5; 4], b: &[R5; 4]) {
    for (a, b) in a.iter().zip(b) {
        bits(&a.res_max, &b.res_max, "res_max");
        bits(&a.res_top20, &b.res_top20, "res_top20");
        bits(&a.res_mean, &b.res_mean, "res_mean");
        bits(&a.res_hhi, &b.res_hhi, "res_hhi");
    }
}

fn check() {
    let mut cases = 0;
    for n in [2, 3, 7, 17, 65, 257] {
        for pattern in 0..7 {
            let f = fixture(n, pattern);
            let shared = prepare_shared(n, &f.rho, &f.lnt, &f.ok);
            for c in 0..3 {
                let w = f.responses[c].each_ref().map(Vec::as_slice);
                let old = reference_fit(n, w, &f.rho, &f.lnt, &f.ok);
                let new = cached_fit(n, w, &f.rho, &f.ok, &shared);
                fit_equal(&old, &new);
                fit_equal(&new, &cached_fit(n, w, &f.rho, &f.ok, &shared));
                output_equal(&finish_r5(n, w, &f.rho, &f.ok, &old),
                             &finish_r5(n, w, &f.rho, &f.ok, &new));
                cases += 1;
            }
        }
    }
    // Both interfaces must preserve the original no-valid-pair failure.
    let hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let mut empty_cases = 0;
    for n in [0, 1, 7] {
        let mut f = fixture(n, 0);
        f.ok.fill(false);
        let w = f.responses[0].each_ref().map(Vec::as_slice);
        let old = std::panic::catch_unwind(|| reference_fit(n, w, &f.rho, &f.lnt, &f.ok));
        let new = std::panic::catch_unwind(|| prepare_shared(n, &f.rho, &f.lnt, &f.ok));
        assert!(old.is_err() && new.is_err(), "empty domain must fail");
        empty_cases += 1;
    }
    std::panic::set_hook(hook);
    println!("{{\"chain_cases\":{cases},\"empty_domain_cases\":{empty_cases},\"xtx_xty_beta_bits_equal\":true,\"all_16_residual_columns_bits_equal\":true,\"repeated_calls_bits_equal\":true,\"kernel_threads\":1}}");
}

fn group(f: &Fixture, mode: &str) {
    let cache = if mode == "shared" {
        Some(prepare_shared(black_box(f.n), black_box(&f.rho), black_box(&f.lnt), black_box(&f.ok)))
    } else { None };
    for c in 0..3 {
        let w = black_box(f.responses[c].each_ref().map(Vec::as_slice));
        let fit = match &cache {
            Some(cache) => cached_fit(f.n, w, black_box(&f.rho), black_box(&f.ok), black_box(cache)),
            None => reference_fit(f.n, w, black_box(&f.rho), black_box(&f.lnt), black_box(&f.ok)),
        };
        black_box(fit);
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() { check(); return; }
    assert_eq!(args.len(), 3);
    assert_eq!(args[0], "--bench");
    let mode = &args[1];
    assert!(mode == "reference" || mode == "shared");
    let iterations: usize = args[2].parse().unwrap();
    assert!(iterations > 0);
    let f = fixture(5107, 0);
    group(&f, mode); // One unmeasured warm-up, including cache creation.
    let start = Instant::now();
    for _ in 0..iterations { group(black_box(&f), mode); }
    let seconds = start.elapsed().as_secs_f64();
    println!("{{\"mode\":\"{mode}\",\"n\":5107,\"chains\":3,\"iterations\":{iterations},\"seconds\":{seconds:.9},\"cache_construction_included\":true,\"scope\":\"normal_equations_and_solve_only\",\"kernel_threads\":1}}");
}
