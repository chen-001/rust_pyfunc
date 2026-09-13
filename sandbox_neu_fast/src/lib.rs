//! 在 sandbox 中 include 生产中性化实现并实验 fast path。
#![allow(dead_code, unused_imports, unused_variables, non_snake_case)]
use pyo3::prelude::*;

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use nalgebra::Dyn;

pub mod factor_neutralization_io_optimized {
    use nalgebra::DMatrix;
    use std::collections::HashMap;
    use std::sync::{Arc, Mutex};
    pub struct IOOptimizedStyleDayData {
        pub stocks: Vec<String>,
        pub style_matrix: DMatrix<f64>,
        pub regression_matrix: Option<Arc<DMatrix<f64>>>,
        pub regression_matrix_style_only: Option<Arc<DMatrix<f64>>>,
        pub stock_index_map: HashMap<String, usize>,
    }
    pub struct IOOptimizedStyleData {
        pub data_by_date: HashMap<i64, IOOptimizedStyleDayData>,
        pub file_cache: Arc<Mutex<Vec<u8>>>,
    }
    impl IOOptimizedStyleData {
        pub fn load_from_parquet_io_optimized(_path: &str) -> pyo3::PyResult<Self> {
            Err(pyo3::exceptions::PyRuntimeError::new_err("sandbox stub"))
        }
    }
}

pub mod prod_neu {
    include!("prod_factor_neutralize_std.rs");
    include!("fast_impl.rs");
    include!("backtest_impl.rs");
}

#[pymodule]
fn dev_sandbox_neu_fast(_py: pyo3::Python<'_>, m: &pyo3::types::PyModule) -> pyo3::PyResult<()> {
    use pyo3::prelude::*;
    m.add_class::<prod_neu::SharedHandle>()?;
    m.add_class::<prod_neu::SlotsHandle>()?;
    m.add_function(wrap_pyfunction!(prod_neu::sandbox_precompute, m)?)?;
    m.add_function(wrap_pyfunction!(prod_neu::sandbox_make_slots, m)?)?;
    m.add_function(wrap_pyfunction!(prod_neu::sandbox_baseline_batch, m)?)?;
    m.add_function(wrap_pyfunction!(prod_neu::sandbox_fast_batch, m)?)?;
    m.add_function(wrap_pyfunction!(prod_neu::sandbox_fast_batch_batched, m)?)?;
    m.add_function(wrap_pyfunction!(prod_neu::sandbox_fast_batch_v2, m)?)?;
    m.add_function(wrap_pyfunction!(prod_neu::sandbox_fast_approx_batch, m)?)?;
    m.add_function(wrap_pyfunction!(prod_neu::sandbox_fast_approx_batch_batched, m)?)?;
    m.add_class::<prod_neu::BtHandle>()?;
    m.add_function(wrap_pyfunction!(prod_neu::sandbox_make_bt_handle, m)?)?;
    m.add_function(wrap_pyfunction!(prod_neu::sandbox_backtest_ic_batch, m)?)?;
    m.add_function(wrap_pyfunction!(prod_neu::sandbox_baseline_profile, m)?)?;
    m.add_function(wrap_pyfunction!(prod_neu::sandbox_fast_profile, m)?)?;
    m.add_function(wrap_pyfunction!(prod_neu::sandbox_fast_profile_full, m)?)?;
    Ok(())
}
