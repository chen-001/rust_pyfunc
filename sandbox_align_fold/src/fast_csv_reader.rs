//! 统一挂载沙箱共享纯 std 读取底座（单次打开 + 线程内字节缓冲复用 + 页缓存释放）。
#[path = "../../sandbox_hidden_arrange/src/fast_csv_reader.rs"]
mod inner;

pub use inner::{MarketRecord, TradeRecord};

#[inline]
pub fn read_trade_fast(code: &str, date: i64) -> std::io::Result<Vec<TradeRecord>> {
    inner::read_trade_fast_inner(code, date, true, true)
}

#[inline]
pub fn read_market_fast(code: &str, date: i64) -> std::io::Result<Vec<MarketRecord>> {
    inner::read_market_fast_inner(code, date, true)
}
