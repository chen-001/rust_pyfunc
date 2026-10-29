//! Independent dense reference: all ordered B,C and all X, without shared SIMD
//! helpers, top-K helpers, or network aggregate code.
use yupei_refined::model::{Kernel, NetworkOutput, Stock};
use yupei_refined::network::compute_network;

type Features = Vec<[Vec<f64>; 2]>;
fn stocks_and_features(n: usize) -> (Vec<Stock>, Features) {
    let mut seed = 1234567u64;
    let mut next = || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((seed >> 33) % 1000 + 1) as f64 / 1000.
    };
    let f: Features = (0..n)
        .map(|_| std::array::from_fn(|_| (0..7).map(|_| next()).collect()))
        .collect();
    let s = (0..n)
        .map(|i| Stock {
            code: format!("{:06}", i + 1),
            industry: (i % 3 + 1) as i32,
            cells: vec![],
            n_buy: 11 + i * 7,
            n_sell: 5 + i * 3,
            amount: 10.,
            minute_buy: vec![0.; 237],
            minute_sell: vec![0.; 237],
            minute_ret: vec![f64::NAN; 237],
            minute_close: vec![f64::NAN; 237],
        })
        .collect();
    (s, f)
}
fn kernel(f: &Features) -> Kernel {
    let n = f.len();
    let vectors: Vec<[Vec<f64>; 3]> = f
        .iter()
        .map(|v| {
            [
                v[0].iter().zip(&v[1]).map(|(a, b)| a + b).collect(),
                v[0].clone(),
                v[1].clone(),
            ]
        })
        .collect();
    let energy: [Vec<f64>; 3] = std::array::from_fn(|a| {
        vectors
            .iter()
            .map(|v| v[a].iter().map(|x| x * x).sum())
            .collect()
    });
    let rho = std::array::from_fn(|ab| {
        let (a, b) = (ab / 3, ab % 3);
        (0..n * n)
            .map(|k| {
                let (i, j) = (k / n, k % n);
                if energy[a][i] == 0. || energy[b][j] == 0. {
                    f32::NAN
                } else {
                    (vectors[i][a]
                        .iter()
                        .zip(&vectors[j][b])
                        .map(|(x, y)| x * y)
                        .sum::<f64>()
                        / (energy[a][i] * energy[b][j]).sqrt()) as f32
                }
            })
            .collect()
    });
    Kernel {
        n,
        lead: std::array::from_fn(|_| vec![0.; n * n]),
        energy,
        rho,
    }
}
fn col<'a>(out: &'a NetworkOutput, suffix: &str) -> &'a Vec<Option<f64>> {
    &out.columns
        .iter()
        .find(|c| c.name.ends_with(suffix))
        .unwrap()
        .values
}
fn close(a: Option<f64>, b: Option<f64>, eps: f64) {
    match (a, b) {
        (Some(a), Some(b)) => assert!((a - b).abs() <= eps, "{a} != {b}, tolerance {eps}"),
        (None, None) => (),
        _ => panic!("Missing-value mismatch: {a:?}, {b:?}"),
    }
}
fn q(v: &mut [f64], p: f64) -> Option<f64> {
    if v.is_empty() {
        return None;
    }
    v.sort_by(f64::total_cmp);
    let x = p * (v.len() - 1) as f64;
    let lo = x.floor() as usize;
    let hi = x.ceil() as usize;
    Some(v[lo] + (v[hi] - v[lo]) * (x - lo as f64))
}
struct Reference {
    pair_g: Vec<f32>,
    gmean: Vec<Option<f64>>,
    gcov: Vec<Option<f64>>,
    npos90: Vec<Option<f64>>,
    self95: Vec<Option<f64>>,
    wstd: Vec<Option<f64>>,
    hc: Vec<Option<f64>>,
    gc: Vec<Option<f64>>,
}
fn brute(stocks: &[Stock], k: &Kernel, a: usize, b: usize, c: usize) -> Reference {
    let n = k.n;
    let mut pair_g = vec![f32::NAN; n * n];
    let (mut rowg, mut rowcov, mut rownpos, mut rowself) = (
        vec![vec![]; n],
        vec![vec![]; n],
        vec![vec![]; n],
        vec![vec![]; n],
    );
    let (mut masses, mut hc, mut gc, mut eligible) = (
        vec![[0.; 3]; n],
        vec![0u64; n],
        vec![0u64; n],
        vec![0usize; n],
    );
    for i in 0..n {
        for j in 0..n {
            if i == j || k.energy[a][i] <= 0. || k.energy[b][j] <= 0. {
                continue;
            }
            let (ei, ej) = (k.energy[a][i], k.energy[b][j]);
            let d = (ei + ej + 2. * (ei * ej).sqrt() * k.rho[a * 3 + b][i * n + j] as f64).sqrt();
            let (wi, wj) = ((ei.sqrt() / d) as f32, (ej.sqrt() / d) as f32);
            let mut scores = vec![];
            for x in 0..n {
                if x == i || x == j || k.energy[c][x] <= 0. {
                    continue;
                }
                let (p, q) = (k.rho[a * 3 + c][i * n + x], k.rho[b * 3 + c][j * n + x]);
                let h = wi * p + wj * q;
                let g = h - p.max(q);
                scores.push((x, h, g, p));
                eligible[x] += 1;
            }
            if scores.is_empty() {
                continue;
            }
            let mut byh = scores.clone();
            byh.sort_by(|x, y| {
                y.1.total_cmp(&x.1)
                    .then(stocks[x.0].code.cmp(&stocks[y.0].code))
            });
            let mut byg = scores.clone();
            byg.sort_by(|x, y| {
                y.2.total_cmp(&x.2)
                    .then(stocks[x.0].code.cmp(&stocks[y.0].code))
            });
            rowg[i].push(byh[0].2 as f64);
            pair_g[i * n + j] = byh[0].2;
            rowself[i].push(byh[0].3 as f64);
            rowcov[i].push(
                scores.iter().map(|s| (s.2 as f64).max(0.)).sum::<f64>() / scores.len() as f64,
            );
            rownpos[i]
                .push(scores.iter().filter(|s| s.2 > 0.).count() as f64 / scores.len() as f64);
            for v in byh.iter().take(20) {
                hc[v.0] += 1;
            }
            for v in byg.iter().take(100) {
                gc[v.0] += 1;
            }
            let sh = scores.iter().map(|s| (s.1 as f64).max(0.)).sum::<f64>();
            if sh > 0. {
                for &(x, h, g, _) in &scores {
                    let w = (h as f64).max(0.) / sh;
                    masses[x][0] += w;
                    masses[x][1] += w * g as f64;
                    masses[x][2] += w * (g as f64).powi(2);
                }
            }
        }
    }
    let avg = |v: &Vec<f64>| {
        if v.is_empty() {
            None
        } else {
            Some(v.iter().sum::<f64>() / v.len() as f64)
        }
    };
    Reference {
        pair_g,
        gmean: rowg.iter().map(avg).collect(),
        gcov: rowcov.iter().map(avg).collect(),
        npos90: rownpos.iter_mut().map(|v| q(v, 0.9)).collect(),
        self95: rowself.iter_mut().map(|v| q(v, 0.95)).collect(),
        wstd: masses
            .iter()
            .map(|m| {
                if m[0] > 0. {
                    Some((m[2] / m[0] - (m[1] / m[0]).powi(2)).max(0.).sqrt())
                } else {
                    None
                }
            })
            .collect(),
        hc: (0..n)
            .map(|x| (eligible[x] > 0).then_some(hc[x] as f64))
            .collect(),
        gc: (0..n)
            .map(|x| (eligible[x] > 0).then_some(gc[x] as f64))
            .collect(),
    }
}
#[test]
fn exact_dense_reference_and_role_swap() {
    let (s, f) = stocks_and_features(12);
    let k = kernel(&f);
    for (chain, a, b, c) in [
        ("AAa", 0, 0, 0),
        ("pma", 1, 2, 0),
        ("mpm", 2, 1, 2),
        ("ppp", 1, 1, 1),
        ("mmm", 2, 2, 2),
    ] {
        let out = compute_network(&s, &k, chain).unwrap();
        let r = brute(&s, &k, a, b, c);
        for i in 0..s.len() {
            // n=12: top20 G mean is the mean of the entire eligible partner row.
            for (name, reference, tol) in [
                ("row_g_t20m", &r.gmean, 2e-7),
                ("row_gcov_mean", &r.gcov, 2e-7),
                ("row_gnpos_p90", &r.npos90, 1e-7),
                ("row_self_assoc_p95", &r.self95, 2e-7),
                ("e_w_std_soft", &r.wstd, 5e-6),
                ("x0topk20_count", &r.hc, 0.),
                ("gtopk100_count", &r.gc, 0.),
            ] {
                close(col(&out, name)[i], reference[i], tol);
            }
        }
        let expected_rho = matches!(chain, "AAa" | "ppp" | "mmm");
        assert_eq!(
            out.columns.iter().any(|v| v.name.ends_with("row_rho_skew")),
            expected_rho
        );
        assert_eq!(
            out.columns
                .iter()
                .filter(|v| v.name.contains("best_z_partner_reverse_rank"))
                .count(),
            if chain == "ppp" { 2 } else { 0 }
        );
    }
    let mut swapped_f = f.clone();
    for v in &mut swapped_f {
        v.swap(0, 1);
    }
    let mut swapped_s = s.clone();
    for v in &mut swapped_s {
        std::mem::swap(&mut v.n_buy, &mut v.n_sell);
    }
    let x = compute_network(&s, &k, "pma").unwrap();
    let y = compute_network(&swapped_s, &kernel(&swapped_f), "mpa").unwrap();
    for (cx, cy) in x.columns.iter().zip(&y.columns) {
        for i in 0..s.len() {
            close(cx.values[i], cy.values[i], 1e-9);
        }
    }
}
#[test]
fn ppp_best_z_partner_reverse_rank_matches_dense_reference() {
    let (mut stocks, f) = stocks_and_features(12);
    // One activity stratum makes Z the independently calculated global z-score.
    for stock in &mut stocks {
        stock.n_buy = 10;
    }
    let k = kernel(&f);
    let run = |threads| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(|| compute_network(&stocks, &k, "ppp").unwrap())
    };
    let out = run(1);
    let parallel = run(50);
    assert_eq!(
        serde_json::to_string(&out.columns).unwrap(),
        serde_json::to_string(&parallel.columns).unwrap()
    );
    let g = brute(&stocks, &k, 1, 1, 1).pair_g;
    let finite: Vec<_> = g
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .map(f64::from)
        .collect();
    let mean = finite.iter().sum::<f64>() / finite.len() as f64;
    let sd = (finite.iter().map(|v| v * v).sum::<f64>() / finite.len() as f64 - mean * mean).sqrt();
    let z: Vec<_> = g.iter().map(|&v| ((v as f64 - mean) / sd) as f32).collect();
    for i in 0..k.n {
        let mut partners: Vec<_> = (0..k.n)
            .filter(|&j| j != i && z[i * k.n + j].is_finite())
            .collect();
        partners.sort_by(|&p, &q| {
            z[i * k.n + q]
                .total_cmp(&z[i * k.n + p])
                .then(stocks[p].code.cmp(&stocks[q].code))
        });
        let p = partners[0];
        let reverse: Vec<_> = (0..k.n)
            .filter(|&j| j != p && z[p * k.n + j].is_finite())
            .collect();
        let count = reverse
            .iter()
            .filter(|&&j| z[p * k.n + j] > z[i * k.n + p])
            .count() as f64;
        close(col(&out, "best_z_partner_reverse_rank")[i], Some(count), 0.);
        close(
            col(&out, "best_z_partner_reverse_rank_fraction")[i],
            Some(count / reverse.len() as f64),
            0.,
        );
    }
    let mut absent = f.clone();
    for v in &mut absent {
        v[0].fill(0.);
    }
    let missing = compute_network(&stocks, &kernel(&absent), "ppp").unwrap();
    assert!(col(&missing, "best_z_partner_reverse_rank")
        .iter()
        .all(Option::is_none));
    assert!(col(&missing, "best_z_partner_reverse_rank_fraction")
        .iter()
        .all(Option::is_none));
}

#[test]
fn mixed_role_target_statistics_have_the_same_mathematical_definition() {
    let (stocks, f) = stocks_and_features(37);
    let k = kernel(&f);
    for (pm, mp) in [("pma", "mpa"), ("pmp", "mpp"), ("pmm", "mpm")] {
        let a = compute_network(&stocks, &k, pm).unwrap();
        let b = compute_network(&stocks, &k, mp).unwrap();
        for suffix in ["e_w_std_soft", "x0topk20_count", "gtopk100_count"] {
            for i in 0..k.n {
                close(
                    col(&a, suffix)[i],
                    col(&b, suffix)[i],
                    if suffix == "e_w_std_soft" { 5e-6 } else { 0. },
                );
            }
        }
    }
}
#[test]
fn one_sided_missing_direction_is_not_zero_or_wrong_role() {
    let (mut s, mut f) = stocks_and_features(8);
    f[0][0].fill(0.);
    s[0].n_buy = 0;
    f[1][1].fill(0.);
    s[1].n_sell = 0;
    let k = kernel(&f);
    let out = compute_network(&s, &k, "pma").unwrap();
    let r = brute(&s, &k, 1, 2, 0);
    assert!(!out.neighbors.valid[0]);
    assert!(col(&out, "row_gs_skew")[0].is_none());
    assert!(col(&out, "x0topk20_count")[0].is_some());
    for i in 0..s.len() {
        close(col(&out, "row_g_t20m")[i], r.gmean[i], 2e-7);
        close(col(&out, "row_self_assoc_p95")[i], r.self95[i], 2e-7);
    }
    // No sell stream exists anywhere: no valid C or X for pmm, and every
    // output is missing. This is different from an observed zero count.
    for (st, v) in s.iter_mut().zip(&mut f) {
        v[1].fill(0.);
        st.n_sell = 0;
    }
    let out = compute_network(&s, &kernel(&f), "pmm").unwrap();
    assert!(out
        .columns
        .iter()
        .flat_map(|c| &c.values)
        .all(Option::is_none));
    for a in out.audits.iter().filter(|a| a.layer == "cross_section") {
        assert_eq!(a.n, 0);
        assert!(a.value.is_none());
        assert!(a.detail.contains("Undefined"));
    }
}
#[test]
fn code_permutation_and_thread_count_determinism() {
    let (s, f) = stocks_and_features(37);
    let k = kernel(&f);
    let run = |threads| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(|| compute_network(&s, &k, "pmp").unwrap())
    };
    let x = run(1);
    let y = run(50);
    assert_eq!(
        serde_json::to_string(&x.columns).unwrap(),
        serde_json::to_string(&y.columns).unwrap()
    );
    assert_eq!(
        serde_json::to_string(&x.audits).unwrap(),
        serde_json::to_string(&y.audits).unwrap()
    );
    assert_eq!(
        serde_json::to_string(&x.neighbors).unwrap(),
        serde_json::to_string(&y.neighbors).unwrap()
    );
    let sr: Vec<_> = s.iter().cloned().rev().collect();
    let fr: Features = f.iter().cloned().rev().collect();
    let reversed = compute_network(&sr, &kernel(&fr), "pmp").unwrap();
    for (cx, cr) in x.columns.iter().zip(&reversed.columns) {
        for i in 0..s.len() {
            close(cx.values[i], cr.values[s.len() - 1 - i], 2e-5);
        }
    }
    for i in 0..s.len() {
        assert_eq!(
            x.neighbors.neighbors[i],
            reversed.neighbors.neighbors[s.len() - 1 - i]
        );
    }
}
#[test]
fn malformed_kernel_and_unknown_chain_fail_explicitly() {
    let (s, f) = stocks_and_features(5);
    let mut k = kernel(&f);
    assert!(compute_network(&s, &k, "oops").is_err());
    k.rho[0].clear();
    assert!(compute_network(&s, &k, "AAa").is_err());
}

#[test]
fn top100_diagnostic_uses_the_same_g_selected_pairs() {
    let (stocks, f) = stocks_and_features(109);
    let k = kernel(&f);
    let out = compute_network(&stocks, &k, "pma").unwrap();
    for i in [0, 37, 108] {
        let mut partners = Vec::new();
        for j in 0..k.n {
            if j == i {
                continue;
            }
            let (e1, e2) = (k.energy[1][i], k.energy[2][j]);
            let den = (e1 + e2 + 2. * (e1 * e2).sqrt() * k.rho[5][i * k.n + j] as f64).sqrt();
            let (w1, w2) = ((e1.sqrt() / den) as f32, (e2.sqrt() / den) as f32);
            let mut best: Option<(f32, f32, usize)> = None;
            for x in 0..k.n {
                if x == i || x == j {
                    continue;
                }
                let (p, q) = (k.rho[3][i * k.n + x], k.rho[6][j * k.n + x]);
                let h = w1 * p + w2 * q;
                let gain = h - p.max(q);
                if best
                    .as_ref()
                    .is_none_or(|t| h > t.0 || (h == t.0 && stocks[x].code < stocks[t.2].code))
                {
                    best = Some((h, gain, x));
                }
            }
            partners.push((j, best.unwrap().1));
        }
        partners.sort_by(|a, b| {
            b.1.total_cmp(&a.1)
                .then(stocks[a.0].code.cmp(&stocks[b.0].code))
        });
        partners.truncate(100);
        let rho = partners
            .iter()
            .map(|p| k.rho[5][i * k.n + p.0] as f64)
            .sum::<f64>()
            / 100.;
        let g = partners.iter().map(|p| p.1 as f64).sum::<f64>() / 100.;
        close(col(&out, "diag_rho_at_g_top100_mean")[i], Some(rho), 1e-9);
        close(col(&out, "diag_g_top100_mean")[i], Some(g), 1e-9);
    }
}

/// Reproducible export consumed by validate_reference.py; never writes on a
/// normal cargo test run. Synthetic input and timing are included in the file.
#[test]
#[ignore = "invoked explicitly by tests/validate_reference.py"]
fn write_python_reference_fixture() {
    let path = std::env::var("YUPEI_REFERENCE_JSON").expect("set YUPEI_REFERENCE_JSON");
    let n = std::env::var("YUPEI_REFERENCE_N")
        .unwrap_or("128".into())
        .parse()
        .unwrap();
    let jobs = std::env::var("YUPEI_REFERENCE_JOBS")
        .unwrap_or("1".into())
        .parse()
        .unwrap();
    let chain = std::env::var("YUPEI_REFERENCE_CHAIN").unwrap_or("pma".into());
    let (stocks, features) = stocks_and_features(n);
    let kernel = kernel(&features);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(jobs)
        .build()
        .unwrap();
    let timer = std::time::Instant::now();
    let out = pool.install(|| compute_network(&stocks, &kernel, &chain).unwrap());
    let elapsed = timer.elapsed().as_secs_f64();
    let data = serde_json::json!({"features":features,"stocks":stocks,"columns":out.columns,"rust_network_seconds":elapsed,"jobs":jobs,"chain":chain});
    std::fs::write(path, serde_json::to_vec(&data).unwrap()).unwrap();
}
