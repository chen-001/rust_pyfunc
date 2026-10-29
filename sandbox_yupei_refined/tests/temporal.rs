use yupei_refined::model::{FactorColumn, NeighborState, Snapshot};
use yupei_refined::temporal::temporal_columns;

fn calendar() -> Vec<i64> {
    vec![
        20240102, 20240103, 20240104, 20240105, 20240108, 20240109, 20240110, 20240111, 20240112,
        20240115, 20240116, 20240117, 20240118, 20240119, 20240122, 20240123, 20240124, 20240125,
        20240126, 20240129, 20240130,
    ]
}

fn snapshot(date: i64, t: usize) -> Snapshot {
    let codes: Vec<String> = (1..=4).map(|i| format!("{i:06}")).collect();
    let cols = [
        "row_gs_skew",
        "res_z_p90",
        "r4_ov_g_k100",
        "row_self_assoc_p95",
        "row_gnpos_p90",
        "best_z_partner_reverse_rank",
        "best_z_partner_reverse_rank_fraction",
    ];
    let mut columns: Vec<FactorColumn> = cols
        .iter()
        .map(|metric| FactorColumn {
            name: format!("yr_ppp_{metric}"),
            values: (0..4)
                .map(|i| {
                    Some(match *metric {
                        "row_self_assoc_p95" => t as f64 * (i + 1) as f64,
                        "row_gnpos_p90" => t as f64 / 100.0 + i as f64 / 10.0,
                        "best_z_partner_reverse_rank_fraction" => (t + i) as f64 / 100.0,
                        _ => t as f64 + i as f64,
                    })
                })
                .collect(),
        })
        .collect();
    columns.push(FactorColumn {
        name: "yr_AAm_res_g_max".into(),
        values: (0..4).map(|i| Some((2 * t + i) as f64)).collect(),
    });
    let edges = if t == 20 {
        vec![vec![2, 3], vec![0], vec![0], vec![0]]
    } else {
        vec![vec![1, 2], vec![0], vec![0], vec![]]
    };
    Snapshot {
        schema: "test-v1".into(),
        config_hash: "same-params".into(),
        date,
        codes: codes.clone(),
        columns,
        neighbors: vec![NeighborState {
            chain: "ppp".into(),
            valid: vec![true; 4],
            neighbors: edges
                .into_iter()
                .map(|v| v.into_iter().map(|i| codes[i].clone()).collect())
                .collect(),
        }],
    }
}

fn fixture() -> (Vec<i64>, Vec<Snapshot>, Snapshot) {
    let dates = calendar();
    let history = dates[..20]
        .iter()
        .enumerate()
        .map(|(t, &d)| snapshot(d, t))
        .collect();
    let current = snapshot(dates[20], 20);
    (dates, history, current)
}

fn values<'a>(columns: &'a [FactorColumn], name: &str) -> &'a [Option<f64>] {
    &columns
        .iter()
        .find(|c| c.name == name)
        .unwrap_or_else(|| panic!("missing {name}"))
        .values
}

fn close(value: Option<f64>, expected: f64) {
    assert!((value.expect("expected finite value") - expected).abs() < 1e-12);
}

#[test]
fn complete_windows_and_daily_cross_section_fold_match_hand_calculation() {
    let (dates, history, current) = fixture();
    let (out, audit) = temporal_columns(&current, &history, &dates, false).unwrap();
    assert_eq!(out.len(), 12);
    close(values(&out, "yr_ppp_row_gs_skew_mean_smooth_20")[0], 10.5);
    assert_eq!(values(&out, "yr_ppp_res_z_p90_min_smooth_5")[0], Some(16.0));
    close(values(&out, "yr_ppp_r4_ov_g_k100_mean_smooth_10")[0], 15.5);
    // Last ten dates t=11..20; last five dates t=16..20, inclusive.
    for i in 0..4 {
        close(
            values(&out, "yr_AAm_res_g_max_mean_smooth_10")[i],
            31.0 + i as f64,
        );
        close(
            values(&out, "yr_ppp_best_z_partner_reverse_rank_mean_smooth_5")[i],
            18.0 + i as f64,
        );
        close(
            values(
                &out,
                "yr_ppp_best_z_partner_reverse_rank_fraction_mean_smooth_5",
            )[i],
            0.18 + i as f64 / 100.0,
        );
    }
    assert_eq!(
        values(
            &out,
            "yr_ppp_row_self_assoc_p95_absdev_csmean_max_smooth_20"
        ),
        &[Some(30.0), Some(10.0), Some(10.0), Some(30.0)]
    );
    assert_eq!(
        values(&out, "yr_ppp_row_gnpos_p90_above_csmedian_persistence_10"),
        &[Some(0.0), Some(0.0), Some(1.0), Some(1.0)]
    );
    let delta = values(&out, "yr_ppp_row_gnpos_p90_change_vs_previous_mean20")[0].unwrap();
    assert!((delta - 0.105).abs() < 1e-12); // today's .20 minus prior mean .095
    assert_eq!(
        values(&out, "yr_ppp_positive_gs_neighbors_new_share_previous_day")[0],
        Some(0.5)
    );
    assert_eq!(
        values(&out, "yr_ppp_positive_gs_neighbors_lost_share_previous_day")[0],
        Some(0.5)
    );
    assert_eq!(
        values(&out, "yr_ppp_positive_gs_neighbors_jaccard_previous_day")[0],
        Some(1.0 / 3.0)
    );
    assert!(audit
        .iter()
        .any(|a| a.detail.contains("not asserted identical to legacy fold")));
}

#[test]
fn missing_dates_do_not_silently_use_older_observations() {
    let (dates, mut history, current) = fixture();
    history.remove(6); // 19 previous observations remain, but rolling20 needs this exact date.
    assert!(temporal_columns(&current, &history, &dates, false).is_err());
    let (out, audit) = temporal_columns(&current, &history, &dates, true).unwrap();
    assert!(values(&out, "yr_ppp_row_gs_skew_mean_smooth_20")
        .iter()
        .all(Option::is_none));
    assert_eq!(values(&out, "yr_ppp_res_z_p90_min_smooth_5")[0], Some(16.0));
    assert!(audit.iter().any(|a| a.detail.contains("20240110")));
    // Nineteen historical dates suffice for rolling20, but not previous_mean20.
    let (_, history, current) = fixture();
    let (out, _) = temporal_columns(&current, &history[1..], &dates, true).unwrap();
    close(values(&out, "yr_ppp_row_gs_skew_mean_smooth_20")[0], 10.5);
    assert!(
        values(&out, "yr_ppp_row_gnpos_p90_change_vs_previous_mean20")
            .iter()
            .all(Option::is_none)
    );
}

#[test]
fn no_history_is_an_error_unless_explicit_and_then_schema_is_fixed_null() {
    let (dates, history, current) = fixture();
    assert!(temporal_columns(&current, &[], &dates, false).is_err());
    let (empty, audit) = temporal_columns(&current, &[], &dates, true).unwrap();
    let (full, _) = temporal_columns(&current, &history, &dates, false).unwrap();
    assert_eq!(
        empty.iter().map(|c| &c.name).collect::<Vec<_>>(),
        full.iter().map(|c| &c.name).collect::<Vec<_>>()
    );
    assert!(empty.iter().flat_map(|c| &c.values).all(Option::is_none));
    assert!(audit.iter().any(|a| a.detail.contains("missing_dates")));
}

fn reorder(s: &mut Snapshot, order: &[usize]) {
    s.codes = order.iter().map(|&i| s.codes[i].clone()).collect();
    for c in &mut s.columns {
        c.values = order.iter().map(|&i| c.values[i]).collect();
    }
    for n in &mut s.neighbors {
        n.neighbors = order.iter().map(|&i| n.neighbors[i].clone()).collect();
        n.valid = order.iter().map(|&i| n.valid[i]).collect();
    }
}

#[test]
fn historical_alignment_is_by_code_and_not_by_array_position() {
    let (dates, mut history, mut current) = fixture();
    let (reference, _) = temporal_columns(&current, &history, &dates, false).unwrap();
    for (i, s) in history.iter_mut().enumerate() {
        if i % 2 == 0 {
            reorder(s, &[2, 0, 3, 1]);
        }
    }
    let order = [3, 1, 0, 2];
    reorder(&mut current, &order);
    history.reverse();
    let (out, _) = temporal_columns(&current, &history, &dates, false).unwrap();
    for (a, b) in reference.iter().zip(out) {
        assert_eq!(a.name, b.name);
        for (j, &i) in order.iter().enumerate() {
            match (a.values[i], b.values[j]) {
                (Some(x), Some(y)) => assert!((x - y).abs() < 1e-12),
                (None, None) => (),
                _ => panic!("missingness changed"),
            }
        }
    }
}

#[test]
fn invalid_or_missing_partner_is_not_counted_as_a_lost_link() {
    let (dates, mut history, mut current) = fixture();
    let b = current.codes[1].clone();
    let d = current.codes[3].clone();
    history[19].neighbors[0].neighbors[0] = vec![b.clone(), d];
    current.neighbors[0].neighbors[0] = vec![b];
    current.neighbors[0].valid[3] = false;
    current.neighbors[0].neighbors[3].clear();
    for c in &mut current.columns {
        c.values[3] = None;
    }
    let (out, _) = temporal_columns(&current, &history, &dates, false).unwrap();
    assert_eq!(
        values(&out, "yr_ppp_positive_gs_neighbors_new_share_previous_day")[0],
        Some(0.0)
    );
    assert_eq!(
        values(&out, "yr_ppp_positive_gs_neighbors_lost_share_previous_day")[0],
        Some(0.0)
    );
    assert_eq!(
        values(&out, "yr_ppp_positive_gs_neighbors_jaccard_previous_day")[0],
        Some(1.0)
    );
    assert_eq!(
        values(&out, "yr_ppp_positive_gs_neighbors_jaccard_previous_day")[3],
        None
    );
}

#[test]
fn empty_neighbor_denominators_and_individual_history_stay_undefined() {
    let (dates, mut history, mut current) = fixture();
    for s in history.iter_mut().chain(std::iter::once(&mut current)) {
        for row in &mut s.neighbors[0].neighbors {
            row.clear();
        }
    }
    history[18].columns[0].values[0] = None;
    let (out, _) = temporal_columns(&current, &history, &dates, false).unwrap();
    assert_eq!(values(&out, "yr_ppp_row_gs_skew_mean_smooth_20")[0], None);
    assert!(
        values(&out, "yr_ppp_positive_gs_neighbors_jaccard_previous_day")
            .iter()
            .all(Option::is_none)
    );
}

#[test]
fn permissive_history_never_permits_corrupt_or_future_snapshots() {
    let (dates, history, current) = fixture();
    let mut bad = history.clone();
    bad[0].config_hash = "other".into();
    assert!(temporal_columns(&current, &bad, &dates, true).is_err());
    let mut bad = history.clone();
    bad.push(history[0].clone());
    assert!(temporal_columns(&current, &bad, &dates, true).is_err());
    let mut bad = history.clone();
    bad.push(current.clone());
    assert!(temporal_columns(&current, &bad, &dates, true).is_err());
    let mut bad = current.clone();
    bad.codes[0] = bad.codes[1].clone();
    assert!(temporal_columns(&bad, &history, &dates, true).is_err());
    let mut bad = current.clone();
    bad.columns[0].values.pop();
    assert!(temporal_columns(&bad, &history, &dates, true).is_err());
    let mut bad = current.clone();
    bad.neighbors[0].valid.pop();
    assert!(temporal_columns(&bad, &history, &dates, true).is_err());
    let mut bad_dates = dates.clone();
    bad_dates.swap(0, 1);
    assert!(temporal_columns(&current, &history, &bad_dates, true).is_err());
}

#[test]
fn extension_placeholders_do_not_invent_uncomputed_chains() {
    let (dates, mut history, mut current) = fixture();
    for snapshot in history.iter_mut().chain(std::iter::once(&mut current)) {
        for chain in ["AAa", "mmm"] {
            snapshot.columns.push(FactorColumn {
                name: format!("yr_{chain}_row_gs_skew_industry_activity_residual"),
                values: vec![None; 4],
            });
        }
    }
    let (out, _) = temporal_columns(&current, &history, &dates, false).unwrap();
    assert_eq!(out.len(), 12);
    assert!(out.iter().all(|column| {
        column.name.starts_with("yr_ppp_") || column.name == "yr_AAm_res_g_max_mean_smooth_10"
    }));
}

#[test]
fn newly_listed_stock_has_no_fabricated_history_or_new_partner_count() {
    let (dates, mut history, current) = fixture();
    for snapshot in &mut history {
        reorder(snapshot, &[0, 1, 2]); // fourth code has no historical observations
    }
    let (out, _) = temporal_columns(&current, &history, &dates, false).unwrap();
    assert_eq!(values(&out, "yr_ppp_row_gs_skew_mean_smooth_20")[3], None);
    assert_eq!(
        values(&out, "yr_ppp_positive_gs_neighbors_new_share_previous_day")[3],
        None
    );
    // Today's D edge is outside the common universe, leaving C as A's only
    // comparable current neighbor. C already existed yesterday: no new edge.
    assert_eq!(
        values(&out, "yr_ppp_positive_gs_neighbors_new_share_previous_day")[0],
        Some(0.0)
    );
}
