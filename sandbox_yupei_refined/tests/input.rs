use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use yupei_refined::input::{read_stock, symbol_table, trade_tasks};
use yupei_refined::model::Config;

const DATE: i64 = 20240102;
const OPEN: i64 = 1_704_159_000_000_000; // 2024-01-02 09:30 Asia/Shanghai, UTC epoch microseconds
const HEADER: &str = "symbol,exchange,date,time,exchtime,type,index,price,volume,turnover,flag\n";
static NEXT: AtomicU64 = AtomicU64::new(0);
struct Temp(PathBuf);
impl Temp {
    fn new() -> Self {
        let p = std::env::temp_dir().join(format!(
            "yupei_refined_input_{}_{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&p).unwrap();
        Self(p)
    }
}
impl Drop for Temp {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}
fn row(ts: i64, index: i64, price: f64) -> String {
    format!("000001,SZ,{DATE},0,{ts},0,{index},{price},1,100,66\n")
}
fn write(temp: &Temp, name: &str, rows: &[String]) -> PathBuf {
    let path = temp.0.join(name);
    std::fs::write(&path, format!("{HEADER}{}", rows.concat())).unwrap();
    path
}

#[test]
fn session_boundaries_and_last_partial_bucket_are_retained_exactly() {
    let temp = Temp::new();
    let offsets = [
        -1,
        0,
        7_200_000_000 - 1,
        7_200_000_000,
        12_600_000_000,
        19_620_000_000 - 1,
        19_620_000_000,
    ];
    let rows: Vec<_> = offsets
        .into_iter()
        .enumerate()
        .map(|(i, t)| row(OPEN + t, i as i64, 100.0))
        .collect();
    let path = write(&temp, "trades.csv", &rows);
    let stock = read_stock("000001", &path, 1, DATE).unwrap();
    assert_eq!(stock.n_buy, 4);
    assert_eq!(
        stock.cells.iter().map(|c| c.slot).collect::<Vec<_>>(),
        vec![0, 719_999, 1_260_000, 1_961_999]
    );
    for m in [0, 119, 120, 236] {
        assert_eq!(stock.minute_close[m], 100.0);
    }
}

#[test]
fn timestamp_day_must_match_date_even_when_clock_matches() {
    let temp = Temp::new();
    for shift in [-86_400_000_000, 86_400_000_000] {
        let path = write(&temp, "wrong_day.csv", &[row(OPEN + shift, 1, 100.0)]);
        assert!(read_stock("000001", &path, 1, DATE).is_err());
    }
}

#[test]
fn trade_index_breaks_equal_timestamp_ties_independently_of_file_order() {
    let temp = Temp::new();
    let a = row(OPEN, 1, 100.0);
    let b = row(OPEN, 2, 101.0);
    let first = write(&temp, "a.csv", &[a.clone(), b.clone()]);
    let second = write(&temp, "b.csv", &[b, a]);
    let x = read_stock("000001", &first, 1, DATE).unwrap();
    let y = read_stock("000001", &second, 1, DATE).unwrap();
    assert_eq!(x.minute_close[0], 101.0);
    assert_eq!(x.minute_close[0], y.minute_close[0]);
    assert_eq!(x.minute_ret[0], y.minute_ret[0]);
    assert!((x.minute_ret[0] - 1.01f64.ln()).abs() < 1e-12);
}

#[test]
fn whitelist_is_fail_closed_and_nonstock_poison_is_never_read() {
    let temp = Temp::new();
    let white = temp.0.join("symbol_map.csv");
    assert!(symbol_table(&white).is_err());
    std::fs::write(&white, "symbol,pos\n").unwrap();
    assert!(symbol_table(&white).is_err());
    std::fs::write(&white, "symbol,pos\n000001,0\n000001,1\n").unwrap();
    assert!(symbol_table(&white).is_err());
    std::fs::write(&white, "symbol,pos\n000001,0\n").unwrap();
    let (whitelist, _) = symbol_table(&white).unwrap();
    let dir = temp.0.join(format!("{DATE}/transaction"));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(
        dir.join(format!("000001_{DATE}_transaction.csv")),
        format!("{HEADER}{}", row(OPEN, 1, 100.0)),
    )
    .unwrap();
    // Invalid UTF-8 would fail read_stock if this fund were ever read.
    std::fs::write(
        dir.join(format!("510300_{DATE}_transaction.csv")),
        [0xff, 0xfe, 0xff],
    )
    .unwrap();
    let cfg = Config {
        date: DATE,
        jobs: 1,
        level2_root: temp.0.clone(),
        basic_info_dir: temp.0.clone(),
        vars_dir: temp.0.clone(),
        history: Vec::new(),
        allow_incomplete_history: true,
        allow_uncomputable_corr: true,
        codes: None,
        chains: vec!["ppp".into()],
    };
    let (tasks, excluded) = trade_tasks(&cfg, &whitelist).unwrap();
    assert_eq!(excluded, 1);
    assert_eq!(tasks.len(), 1);
    assert_eq!(tasks[0].0, "000001");
    let stock = read_stock(&tasks[0].0, &tasks[0].1, 1, DATE).unwrap();
    assert_eq!(stock.n_buy, 1);
    assert_eq!(stock.amount, 100.0);
    assert!(trade_tasks(&cfg, &BTreeMap::new()).is_err());
}
