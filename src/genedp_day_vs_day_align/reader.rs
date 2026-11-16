//! 统一委托给 `crate::fast_csv_reader` 双精度标准读取（消除重复 CSV 解析与页缓存泄漏）。
pub use crate::fast_csv_reader::{MarketRecordF64 as MarketRecord, TradeRecordF64 as TradeRecord};

#[inline]
pub fn read_trade_fast(code: &str, date: i64) -> std::io::Result<Vec<TradeRecord>> {
    crate::fast_csv_reader::read_trade_f64_inner(code, date, false, true)
}

/// 读逐笔并保留撤单记录（flag=32），沪深都有；已做下午平移并过滤到连续竞价时段。
#[inline]
pub fn read_trade_with_cancel(code: &str, date: i64) -> std::io::Result<Vec<TradeRecord>> {
    crate::fast_csv_reader::read_trade_f64_inner(code, date, true, true)
}

#[inline]
pub fn read_market_fast(code: &str, date: i64) -> std::io::Result<Vec<MarketRecord>> {
    crate::fast_csv_reader::read_market_f64_inner(code, date, false, true)
}
