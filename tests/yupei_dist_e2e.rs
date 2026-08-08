//! 寻找玉佩-成交距离 端到端回归测试: 合成 6 只股票 → 37 矩阵 → 2761 因子。
//! 运行: cargo test --release --test yupei_dist_e2e
use rust_pyfunc::fast_csv_reader::TradeRecord;
use rust_pyfunc::yupei_dist::indicator_ctx::{IndicatorCtx, MatrixSet};
use rust_pyfunc::yupei_dist::matrix_stage::{compute_matrices, prep_stock, MATRIX_SPECS};

fn mk_rec(t: f64, v: f64, flag: i32, day_start_us: i64) -> TradeRecord {
    TradeRecord {
        time_us: day_start_us + (t * 1e6) as i64,
        time_sec: t,
        price: 10.0,
        volume: v,
        turnover: v * 10.0,
        flag,
        bid_order: 100,
        ask_order: 90,
        index: 0,
    }
}

#[test]
fn end_to_end_synthetic() {
    let day_start_us = 1735637400000000i64; // 20241231 09:30 CST
    let stocks: Vec<_> = [
        (0.0, 100.0, 66), (0.5, 200.0, 83), (1.0, 150.0, 66),
        (2.0, 300.0, 83), (5.0, 250.0, 66), (10.0, 400.0, 83),
        (20.0, 350.0, 66), (30.0, 120.0, 83), (60.0, 500.0, 66),
    ]
    .iter()
    .enumerate()
    .map(|(i, &(t, v, f))| {
        let code = format!("00000{i}");
        let rec = mk_rec(t, v, f, day_start_us);
        prep_stock(&code, &[rec], day_start_us, 1).unwrap()
    })
    .collect();
    assert_eq!(stocks.len(), 9);

    let (mats, stats) = compute_matrices(&stocks);
    assert_eq!(mats.len(), MATRIX_SPECS.len());
    let n = stocks.len();
    for m in mats.iter() {
        assert!(m.iter().all(|&v| v.is_finite()));
        assert_eq!(m.len(), n * n);
    }

    let mut hm = std::collections::HashMap::new();
    for (m, spec) in mats.iter().zip(MATRIX_SPECS.iter()) {
        hm.insert(spec.name.to_string(), m.clone());
    }
    let set = MatrixSet {
        n,
        codes: (0..n).map(|i| format!("00000{i}")).collect(),
        stats,
        mats: hm,
        industry: None,
    };
    let ctx = IndicatorCtx::new(&set, None);
    let mut total = 0usize;
    for def in rust_pyfunc::yupei_dist::indicators::all() {
        for r in (def.compute)(&ctx) {
            assert_eq!(r.values.len(), n, "{}", r.name);
            total += 1;
        }
    }
    // 无行业/无 prev → ind/dyn 输出 NaN 列; 因子总数保持注册数
    assert_eq!(total, rust_pyfunc::yupei_dist::names::YUPEI_DIST_NAMES.len());
}
