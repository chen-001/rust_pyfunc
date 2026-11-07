//! Standalone exactness test and bounded single-core microbenchmark.
//! rustc --edition=2021 -C opt-level=3 -C target-cpu=native --test tests/sort_small_bitonic.rs -o /tmp/sort_small_tests
//! rustc --edition=2021 -C opt-level=3 -C target-cpu=native tests/sort_small_bitonic.rs -o /tmp/sort_small_bench
//! Benchmark emits JSON to stdout; run under taskset and redirect into sandbox results/speed/.

#[path = "../sandbox/yupei_union_trim_20261107/src/yupei_pair/bitonic_sort.rs"]
mod bitonic_sort;

use std::hint::black_box;
use std::time::Instant;

fn next(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

fn data(n: usize, pattern: &str, seed: u64) -> Vec<u64> {
    let mut state = seed;
    let size = if pattern == "partitioned_topk" { 5107 } else { n };
    let mut row: Vec<u64> = (0..size)
        .map(|i| {
            let value = next(&mut state);
            match pattern {
                "score_ties" => ((value & 7) << 32) | i as u64,
                "all_equal" => u64::MAX,
                _ => value,
            }
        })
        .collect();
    match pattern {
        "ascending" => row.sort_unstable(),
        "descending" => row.sort_unstable_by(|a, b| b.cmp(a)),
        "partitioned_topk" => {
            row.select_nth_unstable(n);
            row.truncate(n);
        }
        _ => {}
    }
    row
}

fn elapsed_per_sort(rows: &[Vec<u64>], repeats: usize, method: u8) -> f64 {
    let n = rows[0].len();
    let mut work = vec![0u64; n];
    let mut scratch = [0u64; 512];
    let start = Instant::now();
    for _ in 0..repeats {
        for row in rows {
            work.copy_from_slice(black_box(row));
            match method {
                0 => black_box(&mut work).sort_unstable(),
                1 => assert!(bitonic_sort::sort_small_keys(black_box(&mut work), &mut scratch)),
                2 => bitonic_sort::sort_keys(black_box(&mut work), &mut scratch),
                _ => unreachable!(),
            }
            black_box(&work);
        }
    }
    start.elapsed().as_nanos() as f64 / (repeats * rows.len()) as f64
}

fn main() {
    let patterns = ["random_u64", "score_ties", "ascending", "descending", "all_equal", "partitioned_topk"];
    let baseline_switch = std::env::var("YUPEI_SMALL_SORT_BASELINE").as_deref() == Ok("1");
    println!("{{\"schema\":\"yupei_small_sort_microbenchmark_v1\",\"cpu\":\"AMD EPYC 9754\",\"samples\":7,\"sorts_per_sample\":4096,\"copy_included\":true,\"baseline_switch\":{baseline_switch},\"cases\":[");
    let mut first = true;
    for n in [20, 100] {
        for pattern in patterns {
            let rows: Vec<_> = (1..=128).map(|seed| data(n, pattern, seed * 0x9e3779b9)).collect();
            // Warm both methods before the measured samples.
            black_box(elapsed_per_sort(&rows, 1, 0));
            black_box(elapsed_per_sort(&rows, 1, 1));
            black_box(elapsed_per_sort(&rows, 1, 2));
            let mut baseline = Vec::new();
            let mut candidate = Vec::new();
            let mut dispatch = Vec::new();
            for sample in 0..7 {
                if sample % 2 == 0 {
                    baseline.push(elapsed_per_sort(&rows, 32, 0));
                    candidate.push(elapsed_per_sort(&rows, 32, 1));
                    dispatch.push(elapsed_per_sort(&rows, 32, 2));
                } else {
                    dispatch.push(elapsed_per_sort(&rows, 32, 2));
                    candidate.push(elapsed_per_sort(&rows, 32, 1));
                    baseline.push(elapsed_per_sort(&rows, 32, 0));
                }
            }
            if !first { println!(","); }
            first = false;
            print!("{{\"length\":{n},\"pattern\":\"{pattern}\",\"baseline_ns\":{baseline:?},\"bitonic_ns\":{candidate:?},\"dispatch_ns\":{dispatch:?}}}");
        }
    }
    println!("\n]}}");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn complete_u64_order_matches_standard_sort() {
        for n in [20, 100] {
            for seed in 1..=2000 {
                let mut values = data(n, "random_u64", seed * 0x9e3779b9);
                // Include unsigned boundary values, duplicate keys, and sentinel-valued real keys.
                values[0] = 0;
                values[1] = u64::MAX;
                values[2] = 1u64 << 63;
                values[3] = (1u64 << 63) - 1;
                values[4] = values[5];
                let mut expected = values.clone();
                expected.sort_unstable();
                let mut scratch = [0xa5a5_a5a5_a5a5_a5a5u64; 513];
                assert!(bitonic_sort::sort_small_keys(&mut values, &mut scratch));
                assert_eq!(values, expected, "length={n} seed={seed}");
                let padded = if n == 20 { 32 } else { 128 };
                assert!(scratch[padded..].iter().all(|v| *v == 0xa5a5_a5a5_a5a5_a5a5));
            }
        }
    }

    #[test]
    fn adversarial_patterns_match_standard_sort() {
        for n in [20, 100] {
            for pattern in ["score_ties", "ascending", "descending", "all_equal", "partitioned_topk"] {
                for seed in 1..=64 {
                    let mut values = data(n, pattern, seed);
                    let mut expected = values.clone();
                    expected.sort_unstable();
                    let mut scratch = [0; 512];
                    assert!(bitonic_sort::sort_small_keys(&mut values, &mut scratch));
                    assert_eq!(values, expected, "length={n} pattern={pattern} seed={seed}");
                }
            }
        }
    }

    #[test]
    fn insufficient_scratch_and_other_lengths_preserve_input() {
        for n in [0, 1, 19, 20, 21, 99, 100, 101, 128, 500] {
            let mut values = data(n, "random_u64", 123);
            let original = values.clone();
            let mut scratch = [0; 19];
            assert!(!bitonic_sort::sort_small_keys(&mut values, &mut scratch));
            assert_eq!(values, original);
        }
    }

    #[test]
    fn existing_500_and_fallback_dispatch_are_unchanged() {
        for n in [0, 1, 20, 100, 499, 500, 501] {
            for seed in 1..=32 {
                let mut values = data(n, "random_u64", seed);
                let mut expected = values.clone();
                expected.sort_unstable();
                let mut scratch = [0; 512];
                bitonic_sort::sort_keys(&mut values, &mut scratch);
                assert_eq!(values, expected, "length={n} seed={seed}");
            }
        }
    }
}
