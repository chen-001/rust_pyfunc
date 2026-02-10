//! Agent交易特征计算模块
//!
//! 基于Agent的买卖记录，计算21个过程指标：
//! 1. 成交后时间窗口内价格趋势
//! 2. 成交时两个订单编号的差值绝对值
//! 3. 成交前时间窗口内的成交笔数
//! 4. 成交后时间窗口内的成交笔数
//! 5. 相对日内均值的偏离
//! 6. 成交之前时间窗口内的波动率
//! 7. 成交之后时间窗口内的波动率
//! 8. 与同向其他Agent的同步率
//! 9. 截止此刻的累计P&L的最大回撤及持续时间
//! 10. 截止此刻的盈利集中度（Top 10%交易贡献的总利润占比）
//! 11. 截止此刻的盈亏不对称性（平均盈利/平均亏损）
//! 12. 截止此刻的平均持仓时长（FIFO原则）
//! 13. 仓位利用率（Exposure time / Total trading time）
//! 14. 截止此刻的交易密度
//! 15. 截止此刻的夏普比率
//! 16. 截止此刻的Sortino比率
//! 17. 截止此刻的Calmar比率
//! 18. 截止此刻的收益率
//! 19. 截止此刻的超额收益
//! 20. 截止此刻的持仓量
//! 21. 截止此刻的交易量
//!
//! 使用方式：
//! ```rust
//! use rust_pyfunc::agent_trading_features::compute_agent_features;
//! 
//! let features = compute_agent_features(
//!     &market_trades,
//!     &agent_trades,
//!     10000,  // 时间窗口：10秒
//!     None,   // 不计算同步率
//! );
//! ```

pub mod types;
pub mod metrics;
pub mod core;
pub mod py_bindings;

pub use types::*;
pub use metrics::*;
pub use core::*;
