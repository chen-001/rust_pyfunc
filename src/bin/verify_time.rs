//! 验证 read_trade_fast_inner 新增的 time_us 字段精度（修复 f32 time_sec ±128s 问题）。
use rust_pyfunc::fast_csv_reader::read_trade_fast_inner;
use std::collections::HashSet;

fn main() {
    for code in ["000001", "600000", "300001"] {
        let trades = read_trade_fast_inner(code, 20251231, false, true, usize::MAX)
            .unwrap_or_default()
            .into_iter()
            .filter(|t| t.flag != 32)
            .collect::<Vec<_>>();
        if trades.is_empty() {
            println!("{code}: empty");
            continue;
        }
        println!("\n=== {code} n={} ===", trades.len());
        for t in trades.iter().take(5) {
            println!("  time_us={} time_sec={}", t.time_us, t.time_sec);
        }
        // 相邻正差 + 唯一时间数
        let mut prev = trades[0].time_us;
        let mut mindiff = i64::MAX;
        let mut uniq = HashSet::new();
        for t in &trades {
            if t.time_us > prev {
                mindiff = mindiff.min(t.time_us - prev);
            }
            uniq.insert(t.time_us);
            prev = t.time_us;
        }
        println!("  相邻正差 min={}us  unique times={}/{}",
                 mindiff, uniq.len(), trades.len());
    }
}
