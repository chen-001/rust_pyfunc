use yupei_refined::extensions::{price_columns, rank_difference, residualize};
use yupei_refined::model::{FactorColumn, Stock};

fn stock(i: usize) -> Stock {
    Stock {
        code: format!("{i:06}"),
        industry: 1,
        cells: Vec::new(),
        n_buy: 0,
        n_sell: 0,
        amount: 0.0,
        minute_buy: vec![0.0; 237],
        minute_sell: vec![0.0; 237],
        minute_ret: vec![123.0; 237],
        minute_close: vec![100.0; 237],
    }
}
fn value(columns: &[FactorColumn], stem: &str, h: usize) -> Option<f64> {
    columns
        .iter()
        .find(|c| c.name == format!("yr_{stem}_industry_exself_{h}m"))
        .unwrap()
        .values[0]
}
fn close(v: Option<f64>, x: f64) {
    assert!((v.expect("finite") - x).abs() < 1e-10);
}

#[test]
fn response_uses_later_minute_close_and_excludes_self_from_industry() {
    let mut stocks: Vec<_> = (0..5).map(stock).collect();
    stocks[0].minute_buy.fill(1.0);
    stocks[0].minute_close = (0..237).map(|m| 100.0 * (0.001 * m as f64).exp()).collect();
    // minute_ret is deliberately 123: using same-minute return must fail this.
    // Four flat peers imply zero leave-one-out industry response.
    let out = price_columns(&stocks);
    for h in [1, 5, 15] {
        close(value(&out, "buy_response_beta", h), 0.001 * h as f64);
    }
}

#[test]
fn no_response_window_crosses_lunch() {
    let mut stocks: Vec<_> = (0..5).map(stock).collect();
    stocks[0].minute_buy.fill(1.0);
    stocks[0].minute_close[120..].fill(10_000.0); // huge jump only across lunch
    let out = price_columns(&stocks);
    for h in [1, 5, 15] {
        close(value(&out, "buy_response_beta", h), 0.0);
    }
}

#[test]
fn missing_price_endpoints_are_not_forward_filled() {
    let mut stocks: Vec<_> = (0..5).map(stock).collect();
    stocks[0].minute_buy.fill(1.0);
    stocks[0].minute_close.fill(f64::NAN);
    stocks[0].minute_close[0] = 100.0;
    stocks[0].minute_close[2] = 102.0;
    let out = price_columns(&stocks);
    assert_eq!(value(&out, "buy_response_beta", 1), None);
    close(value(&out, "response_valid_minutes", 1), 0.0);
}

#[test]
fn twelve_samples_are_required_for_each_flow_direction_separately() {
    let mut stocks: Vec<_> = (0..5).map(stock).collect();
    stocks[0].minute_close = (0..237).map(|m| 100.0 * (0.001 * m as f64).exp()).collect();
    stocks[0].minute_buy[..11].fill(1.0);
    stocks[0].minute_sell[11..23].fill(1.0);
    let out = price_columns(&stocks);
    assert_eq!(value(&out, "buy_response_beta", 1), None);
    assert_eq!(value(&out, "buy_exhaustion", 1), None);
    close(value(&out, "sell_response_beta", 1), -0.001);
    close(value(&out, "sell_absorption", 1), 1.0);
    close(value(&out, "response_valid_minutes", 1), 23.0);
    stocks[0].minute_buy[23] = 1.0;
    close(
        value(&price_columns(&stocks), "buy_response_beta", 1),
        0.001,
    );
}

#[test]
fn rank_contrast_uses_only_the_same_valid_stock_intersection() {
    let a = [1.0, 2.0, 3.0, 4.0, f64::NAN];
    let b = [4.0, f64::NAN, 2.0, 1.0, 99.0];
    let d = rank_difference(&a, &b);
    assert_eq!([d[0], d[2], d[3]], [-1.0, 0.0, 1.0]);
    assert!(d[1].is_nan() && d[4].is_nan());
}

#[test]
fn residuals_are_industry_centered_and_keep_missing_targets_missing() {
    let mut stocks: Vec<_> = (0..12).map(stock).collect();
    for s in &mut stocks[6..] {
        s.industry = 2;
    }
    let y = [
        f64::NAN,
        5.0,
        1.0,
        4.0,
        2.0,
        3.0,
        100.0,
        105.0,
        101.0,
        104.0,
        102.0,
        103.0,
    ];
    let control: Vec<_> = (0..12).map(|i| (i % 6) as f64).collect();
    let result = residualize(&stocks, &y, &[&control]);
    assert!(result[0].is_nan());
    for range in [1..6, 6..12] {
        assert!(result[range.clone()].iter().all(|x| x.is_finite()));
        assert!(result[range].iter().sum::<f64>().abs() < 1e-10);
    }
    assert!(result
        .iter()
        .filter(|x| x.is_finite())
        .any(|x| x.abs() > 1e-3));
}
