//! Small Python harness around the production reader and the experimental writer.
use pyo3::prelude::*;
#[path = "../../../src/factor_store_segments.rs"]
mod factor_store_segments;
#[path = "../../../src/factor_store_v5.rs"]
mod factor_store_v5;
#[path = "../../../src/storage_policy.rs"]
mod storage_policy;
#[path = "../../../src/storage_read_cache.rs"]
mod storage_read_cache;
mod backup_reader {
    #[derive(Clone)]
    pub struct TaskResult {
        pub date: i64,
        pub code: String,
        pub timestamp: i64,
        pub facs: Vec<f32>,
    }
    pub fn read_existing_backup_with_filter(
        _: &str,
        _: Option<&std::collections::HashSet<i64>>,
    ) -> Result<std::collections::HashSet<(i64, String)>, std::io::Error> {
        Err(std::io::Error::other("bin backup is outside this sandbox"))
    }
}
mod backup_writer {
    pub fn save_results_to_backup(
        _: &[crate::backup_reader::TaskResult],
        _: &str,
        _: usize,
    ) -> Result<(), std::io::Error> {
        Err(std::io::Error::other("bin backup is outside this sandbox"))
    }
}

fn fail(e: String) -> PyErr { pyo3::exceptions::PyIOError::new_err(e) }
fn rows(dates: Vec<i64>, codes: Vec<String>, values: Vec<Vec<f32>>) -> PyResult<Vec<backup_reader::TaskResult>> {
    if dates.len() != codes.len() || dates.len() != values.len() {
        return Err(pyo3::exceptions::PyValueError::new_err("row count mismatch"));
    }
    Ok(dates.into_iter().zip(codes).zip(values).map(|((date, code), facs)| {
        backup_reader::TaskResult { date, code, timestamp: 0, facs }
    }).collect())
}

#[pyclass]
struct Writer { inner: Option<factor_store_segments::DirectSegmentWriter> }
impl Writer {
    fn get(&mut self) -> PyResult<&mut factor_store_segments::DirectSegmentWriter> {
        self.inner.as_mut().ok_or_else(|| pyo3::exceptions::PyValueError::new_err("writer closed"))
    }
}
#[pymethods]
impl Writer {
    #[new]
    fn new(path: &str, names: Vec<String>, buffer_bytes: usize) -> PyResult<Self> {
        Ok(Self { inner: Some(factor_store_segments::DirectSegmentWriter::open(path, &names, buffer_bytes).map_err(fail)?) })
    }
    fn append_rows(&mut self, dates: Vec<i64>, codes: Vec<String>, values: Vec<Vec<f32>>) -> PyResult<()> {
        self.get()?.append_batch(&rows(dates, codes, values)?).map_err(fail)
    }
    fn append_date(&mut self, date: i64, codes: Vec<String>, values: Vec<Vec<f32>>) -> PyResult<()> {
        let data = rows(vec![date; codes.len()], codes, values)?;
        self.get()?.append_date(date, &data).map_err(fail)
    }
    fn finish(&mut self) -> PyResult<()> { self.get()?.finish().map_err(fail) }
    fn completed(&mut self) -> PyResult<Vec<(i64, String)>> {
        let mut out: Vec<_> = self.get()?.check_completed().map_err(fail)?.into_iter().collect();
        out.sort(); Ok(out)
    }
    fn completed_dates(&mut self) -> PyResult<Vec<i64>> {
        let mut out: Vec<_> = self.get()?.completed_dates().into_iter().collect();
        out.sort_unstable(); Ok(out)
    }
    fn close(&mut self) { self.inner.take(); }
}

#[pyfunction]
fn write_legacy(path: &str, dates: Vec<i64>, codes: Vec<String>, values: Vec<Vec<f32>>, names: Vec<String>) -> PyResult<()> {
    let mut writer = factor_store_v5::FactorStoreWriter::open(path, &names, false).map_err(fail)?;
    writer.append_batch(&rows(dates, codes, values)?).map_err(fail)?;
    writer.finish_and_project(2).map_err(fail)
}

#[pyfunction]
fn read_matrix(path: &str, cols: Vec<usize>, batch: bool) -> PyResult<(Vec<i32>, Vec<String>, Vec<Vec<f32>>)> {
    let reader = factor_store_v5::FactorStoreReader::open(path).map_err(fail)?;
    let (dates, stocks) = reader.template_axes();
    let maps = reader.precompute_scatter_maps(&dates, &stocks);
    let matrices = if batch && !cols.is_empty() {
        if cols.windows(2).any(|p| p[1] != p[0] + 1) {
            return Err(pyo3::exceptions::PyValueError::new_err("batch columns must be contiguous"));
        }
        reader.read_factors_batch_v7_fast(cols[0], cols[0] + cols.len(), &dates, &stocks, &maps)
    } else {
        cols.iter().map(|&c| reader.read_factor_to_matrix_fast(c, &dates, &stocks, &maps)).collect()
    }.map_err(fail)?;
    Ok((dates, stocks, matrices.iter().map(|a| a.iter().copied().collect()).collect()))
}

#[pyfunction]
fn read_cached(path: &str, cols: Vec<usize>, batch: usize) -> PyResult<(usize, Vec<Vec<f32>>)> {
    let reader = factor_store_v5::FactorStoreReader::open(path).map_err(fail)?;
    let (dates, stocks) = reader.template_axes();
    let maps = reader.precompute_scatter_maps(&dates, &stocks);
    let cache = storage_read_cache::FactorReadCache::new(&reader, &cols, batch,
        dates.len() * stocks.len() * 4, 64 * 1024 * 1024).map_err(fail)?;
    let matrices = cols.iter().map(|&col| cache.take(&reader, col, &dates, &stocks, &maps)
        .map(|a| a.iter().copied().collect())).collect::<Result<Vec<_>, _>>().map_err(fail)?;
    Ok((cache.reads.load(std::sync::atomic::Ordering::Relaxed), matrices))
}

#[pymodule]
fn direct_segments_sandbox(_: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<Writer>()?;
    m.add_function(wrap_pyfunction!(write_legacy, m)?)?;
    m.add_function(wrap_pyfunction!(read_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(read_cached, m)?)?;
    m.add_function(wrap_pyfunction!(factor_store_v5::factor_store_v5_info, m)?)?;
    m.add_function(wrap_pyfunction!(factor_store_v5::factor_store_v5_register_group, m)?)?;
    Ok(())
}
