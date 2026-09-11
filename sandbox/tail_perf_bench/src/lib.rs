//! tail_perf_bench 库入口：把原有模块暴露出来，供 `src/bin/*_check.rs` 独立自测，
//! 以及新的三档改造模块（roll8 / pf8 / bt8 / neu8 / v8）使用。
//!
//! 约定：二进制入口 src/main.rs 通过 `use tail_perf_bench::...` 引用，不再自己 `mod`。
pub mod btopt;
pub mod engine;
pub mod neu;
pub mod npy;
pub mod opt;
pub mod preflightopt;
pub mod v2;
pub mod v3;

// ---- 三档改造（2026-09-11）----
pub mod bt8;
pub mod drive8;
pub mod neu8;
pub mod pf8;
pub mod roll8;
pub mod t8;
pub mod v8;
