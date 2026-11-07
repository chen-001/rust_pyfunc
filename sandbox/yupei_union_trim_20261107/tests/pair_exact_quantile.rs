// Standalone lightweight check:
// rustc --edition=2021 --test tests/pair_exact_quantile.rs -O -o /tmp/pair_exact_quantile
// /tmp/pair_exact_quantile
#[path = "../src/yupei_pair/exact_quantile.rs"]
mod exact_quantile;

fn assert_matches_full_sort(values: &[f64], bins: usize) {
    let mut sorted = values.to_vec();
    sorted.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    let want: Vec<_> = (1..bins)
        .map(|k| sorted[(k * sorted.len() / bins).min(sorted.len() - 1)].to_bits())
        .collect();
    let mut partitioned = values.to_vec();
    let got = exact_quantile::exact_quantile_cuts(&mut partitioned, bins).unwrap();
    assert_eq!(got.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), want);
}

#[test]
fn cuts_match_sort_for_repeated_ranks_ties_and_varied_input_order() {
    for n in [1usize, 2, 3, 9, 10, 11, 127, 1024, 8193] {
        for seed in 0..5usize {
            let mut values: Vec<f64> = (0..n)
                .map(|i| (((i * 104729 + seed * 8191) % 65521) as i64 - 32760) as f64 / 8192.0)
                .collect();
            for bins in [1, 2, 3, 10, 23] {
                assert_matches_full_sort(&values, bins);
            }
            values.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
            assert_matches_full_sort(&values, 10);
            values.reverse();
            assert_matches_full_sort(&values, 10);
            for value in &mut values {
                *value = (*value * 2.0).round();
                if *value == 0.0 { *value = 0.0; }
            }
            assert_matches_full_sort(&values, 10);
        }
    }
    assert_matches_full_sort(&vec![0.0; 100], 10);
    assert_matches_full_sort(&vec![1.25; 100], 10);
}

#[test]
fn negative_zero_preserves_input_and_requests_original_sort() {
    let mut values = vec![1.0, -0.0, 0.0, -1.0];
    let before: Vec<_> = values.iter().map(|v| f64::to_bits(*v)).collect();
    assert!(exact_quantile::exact_quantile_cuts(&mut values, 10).is_none());
    assert_eq!(values.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), before);
}
