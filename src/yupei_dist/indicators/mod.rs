//! 指标模块注册表（由 build.rs 自动生成）
pub mod generated;
pub use generated::{all, IndicatorDef};

/// 模块名 → 定义（供 CLI list/verify）
pub fn by_name(name: &str) -> Option<IndicatorDef> {
    all().into_iter().find(|d| d.name == name)
}
