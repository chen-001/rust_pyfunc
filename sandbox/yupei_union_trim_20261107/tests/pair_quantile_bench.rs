// Standalone kernel benchmark using the sandbox's already-built Rayon library.
// The full-market acceptance benchmark remains a separate measurement.
#[path = "../src/yupei_pair/exact_quantile.rs"]
mod exact_quantile;

use rayon::prelude::*;
use std::time::Instant;

fn main() {
    let n = std::env::args().nth(1).map(|v| v.parse().unwrap()).unwrap_or(1_000_000usize);
    let bins = 10usize;
    let pool = rayon::ThreadPoolBuilder::new().num_threads(50).build().unwrap();
    let mut records = Vec::new();
    for kind in ["random_f32", "repeated_f32", "ascending_f32"] {
        let mut state = 0x98461102u64;
        let values: Vec<f64> = (0..n).map(|i| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            match kind {
                "random_f32" => ((state as u32 as f32) / (u32::MAX as f32)) as f64,
                "repeated_f32" => ((state % 1024) as f32 / 1024.0) as f64,
                _ => (i as f32 / n as f32) as f64,
            }
        }).collect();
        for repeat in 1..=3 {
            let mut sorted = values.clone();
            let start = Instant::now();
            pool.install(|| sorted.par_sort_unstable_by(|a, b| a.partial_cmp(b).unwrap()));
            let reference: Vec<f64> = (1..bins).map(|k| sorted[k * n / bins]).collect();
            let sort_seconds = start.elapsed().as_secs_f64();
            let mut partitioned = values.clone();
            let start = Instant::now();
            let cuts = pool.install(|| {
                exact_quantile::exact_quantile_cuts(&mut partitioned, bins).unwrap()
            });
            let select_seconds = start.elapsed().as_secs_f64();
            let identical = cuts.iter().zip(&reference).all(|(a, b)| a.to_bits() == b.to_bits());
            assert!(identical, "cut mismatch for {kind} repeat {repeat}");
            std::hint::black_box(&cuts);
            records.push(format!(
                "{{\"kind\":\"{kind}\",\"repeat\":{repeat},\"sort_seconds\":{sort_seconds:.9},\"select_seconds\":{select_seconds:.9},\"cuts_bitwise_equal\":{identical}}}"
            ));
        }
    }
    println!(
        "{{\"measurement\":\"synthetic_quantile_kernel\",\"elements\":{n},\"bins\":{bins},\"rayon_threads\":50,\"excludes_input_generation_and_clone\":true,\"records\":[{}]}}",
        records.join(",")
    );
}
