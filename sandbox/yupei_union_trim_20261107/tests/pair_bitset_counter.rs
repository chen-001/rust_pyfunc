// The runner extracts the production helper verbatim, so this checks the actual
// implementation without compiling the complete Python extension.
include!(env!("PAIR_BITSET_IMPLEMENTATION"));

use rayon::prelude::*;

fn old_count(a: &[u64], b: &[u64], n: usize, i: usize) -> (usize, usize) {
    let mut inter = 0usize;
    let mut union = 0usize;
    for j in 0..n {
        let pos = i * n + j;
        let av = (a[pos >> 6] >> (pos & 63)) & 1 == 1;
        let bv = (b[pos >> 6] >> (pos & 63)) & 1 == 1;
        inter += (av && bv) as usize;
        union += (av || bv) as usize;
    }
    (inter, union)
}

fn next(seed: &mut u64) -> u64 {
    *seed ^= *seed << 13;
    *seed ^= *seed >> 7;
    *seed ^= *seed << 17;
    *seed
}

#[test]
fn every_row_matches_with_unaligned_boundaries_diagonal_and_dirty_padding() {
    for n in [0usize, 1, 2, 3, 7, 31, 63, 64, 65, 127, 128, 129, 257, 5107] {
        let len = (n * n + 63) / 64;
        let mut seed = 0xABC8993102u64;
        for kind in 0..4 {
            let a: Vec<u64> = (0..len).map(|_| match kind {
                0 => 0, 1 => u64::MAX, _ => next(&mut seed),
            }).collect();
            let b: Vec<u64> = (0..len).map(|_| match kind {
                0 => 0, 2 => u64::MAX, _ => next(&mut seed),
            }).collect();
            for i in 0..n {
                assert_eq!(bitset_row_overlap_union(&a, &b, n, i), old_count(&a, &b, n, i), "n={n} row={i} kind={kind}");
            }
            if n == 0 {
                assert_eq!(bitset_row_overlap_union(&a, &b, n, 0), (0, 0));
            }
        }
    }
}

fn graph(n: usize, ok: &[bool], configured_k: usize, reverse: bool) -> Vec<u64> {
    let k = configured_k.min(ok.iter().filter(|v| **v).count().saturating_sub(1));
    let mut bits = vec![0u64; (n * n + 63) / 64];
    for i in 0..n {
        if !ok[i] { continue; }
        let mut neighbors: Vec<usize> = (0..n).filter(|&j| j != i && ok[j]).collect();
        if reverse { neighbors.reverse(); }
        for j in neighbors.into_iter().take(k) {
            let a = i * n + j;
            let b = j * n + i;
            bits[a >> 6] |= 1 << (a & 63);
            bits[b >> 6] |= 1 << (b & 63);
        }
    }
    bits
}

#[test]
fn graph_outputs_match_bits_for_empty_mask_truncated_degree_and_original_denominator() {
    for n in [1usize, 3, 63, 64, 65, 127, 129] {
        for mask_kind in 0..4 {
            let ok: Vec<bool> = (0..n).map(|i| match mask_kind {
                0 => false, 1 => i == 0, 2 => i % 3 == 0, _ => true,
            }).collect();
            for k in [5usize, 10, 20, 50, 100] {
                let a = graph(n, &ok, k, false);
                let b = graph(n, &ok, k, true);
                for i in 0..n {
                    let output = |counts: (usize, usize)| {
                        if !ok[i] { return [f64::NAN, f64::NAN]; }
                        [counts.0 as f64 / k as f64,
                         if counts.1 > 0 { counts.0 as f64 / counts.1 as f64 } else { f64::NAN }]
                    };
                    let old = output(old_count(&a, &b, n, i));
                    let new = output(bitset_row_overlap_union(&a, &b, n, i));
                    assert_eq!(old.map(f64::to_bits), new.map(f64::to_bits));
                }
            }
        }
    }
}

#[test]
#[ignore]
fn microbench() {
    let n = 5107usize;
    let len = (n * n + 63) / 64;
    let mut seed = 0xABC8993102u64;
    let a: Vec<u64> = (0..len).map(|_| next(&mut seed)).collect();
    let b: Vec<u64> = (0..len).map(|_| next(&mut seed)).collect();
    let pool = rayon::ThreadPoolBuilder::new().num_threads(50).build().unwrap();
    let mut rows = Vec::new();
    for repeat in 1..=3 {
        let started = std::time::Instant::now();
        let old: Vec<_> = pool.install(|| (0..n).into_par_iter().map(|i| old_count(&a, &b, n, i)).collect());
        let old_s = started.elapsed().as_secs_f64();
        let started = std::time::Instant::now();
        let new: Vec<_> = pool.install(|| (0..n).into_par_iter().map(|i| bitset_row_overlap_union(&a, &b, n, i)).collect());
        let new_s = started.elapsed().as_secs_f64();
        assert_eq!(old, new);
        std::hint::black_box(&new);
        rows.push(format!("{{\"repeat\":{repeat},\"old_seconds\":{old_s:.9},\"new_seconds\":{new_s:.9},\"counts_equal\":true}}"));
    }
    println!("PAIR_BITSET_BENCH={{\"measurement\":\"synthetic_bitset_overlap_one_graph\",\"n\":{n},\"rayon_threads\":50,\"records\":[{}]}}", rows.join(","));
}
