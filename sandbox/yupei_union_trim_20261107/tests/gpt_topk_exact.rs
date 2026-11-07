//! Standalone check for the GPT top-K optimization. Build with
//! rustc --edition=2021 -O -C target-cpu=native tests/gpt_topk_exact.rs -o /tmp/gpt_topk_exact
//! Run without arguments for correctness, or with --bench for a short microbenchmark.
#![allow(dead_code)]
use std::hint::black_box;
use std::time::Instant;

mod yupei_gpt {
    pub mod network {
        // Same exact score/code ordering used by the production scan.
        pub fn key(score: f32, code_rank: u32) -> u64 {
            let b = if score == 0. { 0.0f32.to_bits() } else { score.to_bits() };
            let ascending = if b & 0x8000_0000 != 0 { !b } else { b ^ 0x8000_0000 };
            ((!ascending as u64) << 32) | code_rank as u64
        }
    }
}
#[path = "../src/yupei_gpt/network_simd.rs"]
mod simd;

struct Work {
    hk: Vec<u64>,
    mh: Vec<u64>,
    gk: Vec<u64>,
    mg: Vec<u64>,
    em: Vec<f32>,
    gm: Vec<f32>,
    g2m: Vec<f32>,
}
impl Work {
    fn new(n: usize) -> Self {
        Self {
            hk: vec![0; n], mh: vec![0; n], gk: vec![0; n], mg: vec![0; n],
            em: vec![0.; n], gm: vec![0.; n], g2m: vec![0.; n],
        }
    }
    fn run(&mut self, h: &[f32], g: &[f32], ranks: &[u32], ht: usize, gt: usize, sorted: bool) {
        simd::topk2_mass(h, g, ht, gt, ranks, &mut self.hk, &mut self.mh,
            &mut self.gk, &mut self.mg,
            Some(simd::MassAccMut { em: &mut self.em, gm: &mut self.gm,
                g2m: &mut self.g2m, invh: 0.00137 }), sorted);
    }
}

fn scores(n: usize, seed: u64, pattern: usize) -> (Vec<f32>, Vec<f32>, Vec<u32>) {
    let mut state = seed + 1;
    let mut next = || {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (state >> 32) as u32
    };
    let mut h = Vec::with_capacity(n);
    let mut g = Vec::with_capacity(n);
    for i in 0..n {
        let a = (next() % 100_000) as f32 / 100_000.;
        let b = (next() % 100_000) as f32 / 100_000.;
        let (hv, gv) = match pattern {
            0 => (a, b - 0.8),
            1 => (0.5, 0.), // every score ties: stock-code ordering decides
            2 => ((i % 7) as f32 - 3., (i % 3) as f32 - 1.),
            3 => (f32::NEG_INFINITY, f32::NEG_INFINITY),
            4 => (if i % 2 == 0 { -0. } else { 0. }, -0.),
            5 => (if i < 5 { a } else { f32::NEG_INFINITY }, b - 0.5),
            _ => (i as f32 / n as f32, 1. - i as f32 / n as f32),
        };
        h.push(hv);
        g.push(gv);
    }
    // Stock-code rank differs from physical position, including tied scores.
    let ranks = (0..n).rev().map(|x| x as u32).collect();
    (h, g, ranks)
}

fn verify() {
    let sizes = [1, 7, 8, 9, 20, 100, 127, 128, 511, 1023, 1024, 1025, 5107, 8193];
    let mut cases = 0usize;
    for n in sizes {
        for pattern in 0..7 {
            for seed in [0, 17, 713] {
                let (h, g, ranks) = scores(n, seed, pattern);
                let (ht, gt) = (20.min(n), 100.min(n));
                let (mut old, mut new) = (Work::new(n), Work::new(n));
                for _ in 0..3 {
                    old.run(&h, &g, &ranks, ht, gt, true);
                    new.run(&h, &g, &ranks, ht, gt, false);
                    assert_eq!(new.hk[0], old.hk[0], "H winner: n={n} pattern={pattern}");
                    let mut hh = new.hk[..ht].to_vec();
                    let mut gg = new.gk[..gt].to_vec();
                    hh.sort_unstable();
                    gg.sort_unstable();
                    assert_eq!(hh, old.hk[..ht], "H set: n={n} pattern={pattern}");
                    assert_eq!(gg, old.gk[..gt], "G set: n={n} pattern={pattern}");
                    for (a, b) in old.em.iter().chain(&old.gm).chain(&old.g2m)
                        .zip(new.em.iter().chain(&new.gm).chain(&new.g2m)) {
                        assert_eq!(a.to_bits(), b.to_bits(), "mass bits: n={n} pattern={pattern}");
                    }
                }
                // Independent full sort also verifies exact membership.
                for (v, actual, take) in [(&h, &old.hk, ht), (&g, &old.gk, gt)] {
                    let mut expected: Vec<_> = v.iter().zip(&ranks)
                        .map(|(&v, &r)| yupei_gpt::network::key(v, r)).collect();
                    expected.sort_unstable();
                    assert_eq!(actual[..take], expected[..take]);
                }
                cases += 1;
            }
        }
    }
    println!("{{\"exact_cases\":{cases},\"repetitions_per_case\":3,\"selected_sets_equal\":true,\"h_winner_equal\":true,\"mass_bits_equal\":true}}");
}

fn benchmark() {
    let n = 5107;
    let inputs: Vec<_> = (0..8).map(|seed| scores(n, seed, 0)).collect();
    let iterations = 6000;
    let mut work = Work::new(n);
    let mut rows = Vec::new();
    for pass in 0..4 {
        let sorted = pass % 2 == 0;
        let start = Instant::now();
        for i in 0..iterations {
            let (h, g, ranks) = &inputs[i % inputs.len()];
            work.run(black_box(h), black_box(g), ranks, 20, 100, sorted);
            black_box(&work.hk[..20]);
            black_box(&work.gk[..100]);
            black_box(&work.em);
        }
        rows.push(format!("{{\"pass\":{pass},\"sorted\":{sorted},\"n\":{n},\"calls\":{iterations},\"elapsed_s\":{:.9}}}", start.elapsed().as_secs_f64()));
    }
    println!("{{\"scope\":\"single_core_topk2_mass_microbenchmark\",\"runs\":[{}]}}", rows.join(","));
}

fn main() {
    if std::env::args().any(|s| s == "--bench") { benchmark(); } else { verify(); }
}
