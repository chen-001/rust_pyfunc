use yupei_refined::leadlag::leadlag_columns;
use yupei_refined::model::{Cell, FactorColumn, Stock};

fn stock(code: &str, industry: i32, cells: &[(u32, f32, f32)]) -> Stock {
    Stock {
        code: code.into(),
        industry,
        cells: cells
            .iter()
            .map(|&(slot, buy, sell)| Cell { slot, buy, sell })
            .collect(),
        n_buy: 0,
        n_sell: 0,
        amount: 0.,
        minute_buy: vec![],
        minute_sell: vec![],
        minute_ret: vec![],
        minute_close: vec![],
    }
}
fn value(columns: &[FactorColumn], name: &str, stock: usize) -> Option<f64> {
    columns
        .iter()
        .find(|c| c.name == name)
        .unwrap_or_else(|| panic!("missing {name}"))
        .values[stock]
}
fn close(a: f64, b: f64) {
    assert!((a - b).abs() < 2e-9 * b.abs().max(1.), "{a} != {b}");
}

#[test]
fn simultaneous_events_have_no_temporal_leader() {
    let stocks = vec![
        stock("A", 1, &[(1, 1., 10.)]),
        stock("B", 1, &[(1, 10., 1.)]),
        stock("C", 2, &[(1, 2., 3.)]),
    ];
    let result = leadlag_columns(&stocks).unwrap();
    assert_eq!(result.len(), 36);
    for column in &result {
        for &v in &column.values {
            if let Some(v) = v {
                close(v, 0.);
            }
        }
    }
}

#[test]
fn delayed_response_has_correct_sign_and_excludes_own_cross_flow() {
    // The own stock has both sides and strong internal directionality. Its
    // contribution must vanish from the market target before normalization.
    let stocks = vec![
        stock("A", 1, &[(0, 2., 0.), (20, 0., 7.)]),
        stock("B", 2, &[(100, 3., 4.)]),
    ];
    let c = leadlag_columns(&stocks).unwrap();
    close(
        value(&c, "yr_leadlag_market_buy_buy_t1", 0).unwrap(),
        (-1_f64).exp(),
    );
    close(
        value(&c, "yr_leadlag_market_buy_sell_t1", 0).unwrap(),
        (-1_f64).exp(),
    );
    close(
        value(&c, "yr_leadlag_market_sell_sell_t1", 0).unwrap(),
        (-0.8_f64).exp(),
    );
    close(
        value(&c, "yr_leadlag_market_sell_buy_t1", 1).unwrap(),
        -(-1_f64).exp(),
    );
    close(
        value(&c, "yr_leadlag_cross_industry_buy_sell_t1", 0).unwrap(),
        (-1_f64).exp(),
    );
    assert_eq!(value(&c, "yr_leadlag_industry_buy_buy_t1", 0), None);
    close(
        value(&c, "yr_leadlag_market_buy_buy_t0p1", 0).unwrap(),
        (-10_f64).exp(),
    );
    close(
        value(&c, "yr_leadlag_market_buy_buy_t5", 0).unwrap(),
        (-0.2_f64).exp(),
    );
}

#[test]
fn lunch_sessions_and_directional_absence_remain_explicit() {
    let stocks = vec![
        stock("A", 1, &[(719_999, 1., 0.)]),
        stock("B", 1, &[(1_260_000, 1., 0.)]),
    ];
    let c = leadlag_columns(&stocks).unwrap();
    close(value(&c, "yr_leadlag_market_buy_buy_t5", 0).unwrap(), 0.);
    close(value(&c, "yr_leadlag_industry_buy_buy_t1", 0).unwrap(), 0.);
    assert_eq!(value(&c, "yr_leadlag_market_buy_buy_t1_am", 0), None);
    assert_eq!(value(&c, "yr_leadlag_market_buy_buy_t1_pm", 0), None);
    assert_eq!(
        value(&c, "yr_leadlag_market_buy_buy_t1_pm_minus_am", 0),
        None
    );
    assert_eq!(value(&c, "yr_leadlag_market_buy_sell_t1", 0), None);
}

#[test]
fn singleton_target_is_undefined_and_unknown_industry_is_not_a_group() {
    let c = leadlag_columns(&[stock("A", 1, &[(0, 1., 2.), (100, 3., 4.)])]).unwrap();
    assert!(c.iter().all(|column| column.values[0].is_none()));
    let c = leadlag_columns(&[
        stock("A", -1, &[(0, 1., 2.)]),
        stock("B", -1, &[(100, 3., 4.)]),
    ])
    .unwrap();
    assert!(value(&c, "yr_leadlag_market_buy_buy_t1", 0).is_some());
    assert_eq!(value(&c, "yr_leadlag_industry_buy_buy_t1", 0), None);
    assert_eq!(value(&c, "yr_leadlag_cross_industry_buy_buy_t1", 0), None);
}
