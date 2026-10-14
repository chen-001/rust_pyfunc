#[path = "../src/yupei_pair/bitonic_sort.rs"]
mod bitonic_sort;
#[test]
fn exact_random_and_edge_keys() {
    let mut seed = 123456789u64;
    for len in [0, 1, 127, 499, 500, 501, 512, 1024] {
        for case in 0..1000 {
            let mut values: Vec<u64> = (0..len)
                .map(|i| {
                    seed ^= seed << 13;
                    seed ^= seed >> 7;
                    seed ^= seed << 17;
                    match case % 6 {
                        0 => seed,
                        1 => u64::MAX,
                        2 => i as u64,
                        3 => (len - i) as u64,
                        4 => (seed % 10) << 32 | i as u64,
                        _ => 0,
                    }
                })
                .collect();
            let mut expected = values.clone();
            expected.sort_unstable();
            let mut scratch = vec![0; len.max(512)];
            bitonic_sort::sort_keys(&mut values, &mut scratch);
            assert_eq!(values, expected, "len={len} case={case}");
        }
    }
    // Insufficient scratch must take the ordinary exact sort.
    let mut v: Vec<u64> = (0..500).rev().collect();
    bitonic_sort::sort_keys(&mut v, &mut []);
    assert_eq!(v, (0..500).collect::<Vec<_>>());
}
