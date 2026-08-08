//! 由 build.rs 自动生成，勿手改。
#[path = "strength.rs"] pub mod strength;
use crate::indicator_ctx::{IndicatorCtx, IndicatorResult};
pub struct IndicatorDef {
    pub name: &'static str,
    pub compute: fn(&IndicatorCtx) -> Vec<IndicatorResult>,
}
pub fn all() -> Vec<IndicatorDef> {
    vec![
        IndicatorDef { name: "strength", compute: strength::compute },
    ]
}
