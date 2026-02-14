//! Agent交易模拟器模块
//!
//! 基于Trait架构的Agent交易行为模拟器，支持多种类型的Agent：
//! - 动量型Agent：根据价格变化触发交易
//! - 主买占比型Agent：根据主买占比触发交易
//! - 抄底型/跟随型/加速跟随型/衰竭抄底型Agent
//! - 可扩展更多类型...
//!
//! 使用方式：
//! ```rust
//! use rust_pyfunc::agent_simulator::*;
//!
//! // 创建Agent
//! let mut agent = MomentumAgent::new(
//!     "momentum_3s", 3000, 0.02, false, 100, 1000, true
//! );
//!
//! // 运行模拟
//! let results = run_simulation(&market_trades, &mut [agent]);
//! ```

pub mod types;
pub mod trait_def;
pub mod utils;
pub mod momentum_agent;
pub mod buy_ratio_agent;
pub mod bottom_fishing_agent;
pub mod follow_flow_agent;
pub mod acceleration_follow_agent;
pub mod exhaustion_reversal_agent;
pub mod simulator;
pub mod py_bindings;

pub use types::*;
pub use trait_def::*;
pub use utils::*;
pub use momentum_agent::*;
pub use buy_ratio_agent::*;
pub use bottom_fishing_agent::*;
pub use follow_flow_agent::*;
pub use acceleration_follow_agent::*;
pub use exhaustion_reversal_agent::*;
pub use simulator::*;
