//! 开发沙箱：在这里编写新函数，秒级编译验证，确认无误后迁移到主项目。
//!
//! ── 迁移步骤 ──
//! 1. 将本文件的函数代码复制到主项目 src/your_fn_name.rs
//! 2. 在主项目 src/lib.rs 添加:  pub mod your_fn_name;
//! 3. 在主项目 src/lib.rs 的 #[pymodule] 里添加:
//!    m.add_function(wrap_pyfunction!(your_fn_name::your_fn, m)?)?;
//! 4. 运行 alter.sh（全量 LTO 构建，约 9 分钟，仅此一次）

use pyo3::prelude::*;

/// 示例函数，可删除后替换为你的新函数
#[pyfunction]
fn add(a: f64, b: f64) -> f64 {
    a + b
}

#[pymodule]
fn dev_sandbox(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(add, m)?)?;
    Ok(())
}
