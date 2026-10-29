use yupei_refined::kernel::compute_kernel;
use yupei_refined::model::{Cell, Kernel, Stock};

fn stock(code: &str, cells: &[(u32, f32, f32)]) -> Stock {
    Stock {
        code: code.into(),
        industry: 1,
        cells: cells
            .iter()
            .map(|&(slot, buy, sell)| Cell { slot, buy, sell })
            .collect(),
        n_buy: 0,
        n_sell: 0,
        amount: 0.0,
        minute_buy: vec![],
        minute_sell: vec![],
        minute_ret: vec![],
        minute_close: vec![f64::NAN; 237],
    }
}
fn close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() <= 3e-6 * expected.abs().max(1.0),
        "{actual} != {expected}"
    );
}
fn session(slot: u32) -> usize {
    if slot < 720_000 {
        0
    } else {
        1
    }
}

/// Independent O(events²) reference: enumerate event pairs, not the block recurrence.
fn dense_reference(stocks: &[Stock], tau: f64) -> [Vec<f64>; 4] {
    let n = stocks.len();
    let mut result: [Vec<f64>; 4] = std::array::from_fn(|_| vec![0.0; n * n]);
    for (i, source) in stocks.iter().enumerate() {
        for (j, target) in stocks.iter().enumerate() {
            for a in &source.cells {
                for b in &target.cells {
                    if session(a.slot) != session(b.slot) || a.slot > b.slot {
                        continue;
                    }
                    let decay = if a.slot == b.slot {
                        0.5
                    } else {
                        (-((b.slot - a.slot) as f64) * 0.01 / tau).exp()
                    };
                    for (p, source_weight) in [a.buy, a.sell].iter().enumerate() {
                        for (q, target_weight) in [b.buy, b.sell].iter().enumerate() {
                            result[p * 2 + q][i * n + j] +=
                                *source_weight as f64 * *target_weight as f64 * decay;
                        }
                    }
                }
            }
        }
    }
    result
}
fn assert_same_bits(a: &Kernel, b: &Kernel) {
    for (x, y) in a.lead.iter().flatten().zip(b.lead.iter().flatten()) {
        assert_eq!(x.to_bits(), y.to_bits());
    }
    for (x, y) in a.rho.iter().flatten().zip(b.rho.iter().flatten()) {
        assert_eq!(x.to_bits(), y.to_bits());
    }
    for (x, y) in a.energy.iter().flatten().zip(b.energy.iter().flatten()) {
        assert_eq!(x.to_bits(), y.to_bits());
    }
}

#[test]
fn matches_independent_dense_events_and_cross_direction_diagonal() {
    let stocks = vec![
        stock(
            "A",
            &[
                (0, 1., 0.),
                (3, 2., 1.),
                (4095, 1., 2.),
                (4096, 3., 0.),
                (1_260_002, 1., 4.),
            ],
        ),
        stock(
            "B",
            &[
                (1, 0., 2.),
                (3, 1., 3.),
                (4096, 2., 0.),
                (1_260_001, 2., 1.),
            ],
        ),
        stock("C", &[(0, 1., 1.), (7, 2., 3.), (1_961_999, 1., 2.)]),
    ];
    for tau in [0.25, 1., 5.] {
        let actual = compute_kernel(&stocks, tau).unwrap();
        let expected = dense_reference(&stocks, tau);
        for k in 0..4 {
            for i in 0..9 {
                close(actual.lead[k][i] as f64, expected[k][i]);
            }
        }
        for a in 0..3 {
            for b in 0..3 {
                for i in 0..3 {
                    for j in 0..3 {
                        let value = actual.rho[a * 3 + b][i * 3 + j];
                        assert!(value.is_finite() && (0.0..=1.000003).contains(&value));
                        assert_eq!(value.to_bits(), actual.rho[b * 3 + a][j * 3 + i].to_bits());
                    }
                }
            }
        }
        // Separate event times/directions are not identical vectors, even for one stock.
        assert!(actual.rho[1 * 3 + 2][0] < 0.999);
        close(actual.rho[1 * 3 + 1][0] as f64, 1.0);
    }
}

#[test]
fn mixed_direction_normalization_does_not_create_the_100_bug() {
    let stocks = vec![stock("A", &[(2, 1., 10.)]), stock("B", &[(2, 10., 1.)])];
    let k = compute_kernel(&stocks, 1.).unwrap();
    close(k.lead[1][1] as f64, 0.5); // A buy -> B sell, not A sell -> B buy.
    close(k.lead[2][1] as f64, 50.0);
    close(k.rho[1 * 3 + 2][1] as f64, 1.0);
    for matrix in &k.rho {
        for &v in matrix {
            close(v as f64, 1.0);
        }
    }
}

#[test]
fn lunch_is_a_barrier_even_with_long_decay_and_tail_buckets_are_kept() {
    let stocks = vec![
        stock("A", &[(719_999, 2., 0.)]),
        stock("B", &[(1_260_000, 0., 3.), (1_961_999, 0., 4.)]),
        stock("C", &[(1_961_998, 5., 0.)]),
    ];
    let k = compute_kernel(&stocks, 1_000_000.).unwrap();
    assert_eq!(k.lead[1][1], 0.0);
    close(k.energy[1][0], 4.0); // final morning bucket
    close(
        k.lead[1][2 * 3 + 1] as f64,
        20. * (-0.01_f64 / 1_000_000.).exp(),
    );
    assert!(k.energy[2][1] >= 25.0); // final afternoon bucket contributes
}

#[test]
fn absent_sides_are_nan_and_duplicates_in_one_bucket_sum() {
    let stocks = vec![
        stock("A", &[(5, 1., 0.), (5, 2., 0.)]),
        stock("B", &[(6, 0., 4.)]),
        stock("C", &[]),
    ];
    let k = compute_kernel(&stocks, 1.).unwrap();
    close(k.energy[1][0], 9.0);
    assert_eq!(k.energy[2][0], 0.0);
    assert!(k.rho[2 * 3 + 1][0].is_nan());
    assert!(k.rho[2 * 3 + 2][0].is_nan());
    assert!(k.rho[0][2 * 3 + 2].is_nan());
    close(k.rho[1 * 3 + 2][1] as f64, (-0.01_f64).exp());
}

#[test]
fn deterministic_across_threads_and_permuting_stock_codes() {
    let stocks = vec![
        stock("000001", &[(9, 3., 1.), (4100, 1., 2.)]),
        stock("600000", &[(8, 1., 2.), (4099, 4., 0.)]),
        stock("300750", &[(7, 1., 3.), (1_260_000, 2., 3.)]),
    ];
    let one = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    let three = rayon::ThreadPoolBuilder::new()
        .num_threads(3)
        .build()
        .unwrap();
    let a = one.install(|| compute_kernel(&stocks, 1.).unwrap());
    let b = three.install(|| compute_kernel(&stocks, 1.).unwrap());
    let c = three.install(|| compute_kernel(&stocks, 1.).unwrap());
    assert_same_bits(&a, &b);
    assert_same_bits(&b, &c);
    let order = [2, 0, 1];
    let reordered: Vec<_> = order.iter().map(|&i| stocks[i].clone()).collect();
    let p = three.install(|| compute_kernel(&reordered, 1.).unwrap());
    for i in 0..3 {
        for j in 0..3 {
            for t in 0..4 {
                assert_eq!(
                    p.lead[t][i * 3 + j].to_bits(),
                    a.lead[t][order[i] * 3 + order[j]].to_bits()
                );
            }
            for t in 0..9 {
                assert_eq!(
                    p.rho[t][i * 3 + j].to_bits(),
                    a.rho[t][order[i] * 3 + order[j]].to_bits()
                );
            }
        }
    }
}

#[test]
fn malformed_inputs_fail_before_calculation() {
    let good = vec![stock("A", &[(0, 1., 1.)])];
    for tau in [0., -1., f64::NAN, f64::INFINITY] {
        assert!(compute_kernel(&good, tau).is_err());
    }
    assert!(compute_kernel(&[], 1.).is_err());
    for cells in [
        vec![(720_000, 1., 0.)],
        vec![(1_962_000, 1., 0.)],
        vec![(3, 1., 0.), (2, 1., 0.)],
        vec![(0, -1., 0.)],
    ] {
        assert!(compute_kernel(&[stock("bad", &cells)], 1.).is_err());
    }
}
