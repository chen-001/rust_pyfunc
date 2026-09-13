mod fast_csv_reader;
mod urgency_ext;

use pyo3::prelude::*;

#[pymodule]
fn sandbox_urgency_ext_cluster(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(urgency_ext::py_urgency_ext, m)?)?;
    m.add_function(wrap_pyfunction!(urgency_ext::py_urgency_ext_names, m)?)?;
    m.add_function(wrap_pyfunction!(urgency_ext::py_bench_market, m)?)?;
    m.add_function(wrap_pyfunction!(urgency_ext::py_bench_stock, m)?)?;
    Ok(())
}
