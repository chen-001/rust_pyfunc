//! build.rs: 扫描 src/indicators/*.rs，自动生成模块注册表（src/indicators/generated.rs）。
//! 每个指标模块（文件）必须导出:
//!   pub fn name() -> &'static str
//!   pub fn desc() -> &'static str
//!   pub fn compute(ctx: &IndicatorCtx) -> Vec<IndicatorResult>
//! 新增指标文件后无需手改任何注册处，重新构建即自动注册。

use std::fs;
use std::path::Path;

fn main() {
    println!("cargo:rerun-if-changed=src/indicators");
    let dir = Path::new("src/indicators");
    let mut mods: Vec<String> = Vec::new();
    if let Ok(entries) = fs::read_dir(dir) {
        for e in entries.flatten() {
            let name = e.file_name().to_string_lossy().to_string();
            if name.ends_with(".rs") && name != "mod.rs" && name != "generated.rs" {
                mods.push(name.trim_end_matches(".rs").to_string());
            }
        }
    }
    mods.sort();
    let mut out = String::new();
    out.push_str("//! 由 build.rs 自动生成，勿手改。\n");
    for m in &mods {
        out.push_str(&format!("#[path = \"{m}.rs\"] pub mod {m};\n"));
    }
    out.push_str("use crate::indicator_ctx::{IndicatorCtx, IndicatorResult};\n");
    out.push_str("pub struct IndicatorDef {\n    pub name: &'static str,\n    pub compute: fn(&IndicatorCtx) -> Vec<IndicatorResult>,\n}\n");
    out.push_str("pub fn all() -> Vec<IndicatorDef> {\n    vec![\n");
    for m in &mods {
        out.push_str(&format!("        IndicatorDef {{ name: \"{m}\", compute: {m}::compute }},\n"));
    }
    out.push_str("    ]\n}\n");
    fs::write(dir.join("generated.rs"), out).unwrap();
}
