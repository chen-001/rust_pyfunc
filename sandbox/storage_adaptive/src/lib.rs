use pyo3::prelude::*;
mod storage_policy;
mod storage_read_cache;
mod factor_store_v5;
mod baseline_store;
mod backup_reader {
    #[derive(Clone)]
    pub struct TaskResult { pub date: i64, pub code: String, pub timestamp: i64, pub facs: Vec<f32> }
    pub fn read_existing_backup_with_filter(_: &str, _: Option<&std::collections::HashSet<i64>>) -> Result<std::collections::HashSet<(i64,String)>,std::io::Error> { Err(std::io::Error::other("bin backup is outside storage sandbox")) }
}
mod backup_writer {
    pub fn save_results_to_backup(_: &[crate::backup_reader::TaskResult], _: &str, _: usize) -> Result<(),std::io::Error> { Err(std::io::Error::other("bin backup is outside storage sandbox")) }
}
#[pyfunction]
fn append_store(path: String, dates: Vec<i64>, codes: Vec<String>, values: Vec<Vec<f32>>, names: Vec<String>, project: bool) -> PyResult<()> {
    if dates.len()!=codes.len() || dates.len()!=values.len() { return Err(pyo3::exceptions::PyValueError::new_err("row count mismatch")); }
    let mut w=factor_store_v5::FactorStoreWriter::open(&path,&names,true).map_err(pyo3::exceptions::PyIOError::new_err)?;
    let rows: Vec<_>=dates.into_iter().zip(codes).zip(values).map(|((date,code),facs)|backup_reader::TaskResult{date,code,timestamp:0,facs}).collect();
    w.append_batch(&rows).map_err(pyo3::exceptions::PyIOError::new_err)?;
    w.commit().map_err(pyo3::exceptions::PyIOError::new_err)?;
    if project { w.finish_and_project(1).map_err(pyo3::exceptions::PyIOError::new_err)?; }
    Ok(())
}
#[pyfunction]
fn read_matrix(path: String, cols: Vec<usize>, batch: bool) -> PyResult<Vec<Vec<f32>>> {
    let r=factor_store_v5::FactorStoreReader::open(&path).map_err(pyo3::exceptions::PyIOError::new_err)?;
    let (dates,stocks)=r.template_axes();let maps=r.precompute_scatter_maps(&dates,&stocks);
    let ms=if batch && !cols.is_empty() { r.read_factors_batch_v7_fast(cols[0],cols[0]+cols.len(),&dates,&stocks,&maps) } else { cols.iter().map(|&c|r.read_factor_to_matrix_fast(c,&dates,&stocks,&maps)).collect() }.map_err(pyo3::exceptions::PyIOError::new_err)?;
    Ok(ms.iter().map(|m|m.iter().copied().collect()).collect())
}
#[pyfunction]
fn bench_read(path:String, cols:Vec<usize>, mode:&str)->PyResult<(f64,f64,usize)> {
    use std::time::Instant;
    let fail=pyo3::exceptions::PyIOError::new_err;
    let mut checksum=0f64;let reads;
    let start=Instant::now();
    if mode=="baseline" {
        let r=baseline_store::FactorStoreReader::open(&path).map_err(fail)?;let (ds,cs)=r.template_axes();let maps=r.precompute_scatter_maps(&ds,&cs);
        for &c in &cols {let m=r.read_factor_to_matrix_fast(c,&ds,&cs,&maps).map_err(fail)?;checksum+=m.iter().filter(|v|v.is_finite()).map(|&v|v as f64).sum::<f64>();}reads=cols.len();
    } else {
        let r=factor_store_v5::FactorStoreReader::open(&path).map_err(fail)?;let (ds,cs)=r.template_axes();let maps=r.precompute_scatter_maps(&ds,&cs);
        let cache=storage_read_cache::FactorReadCache::new(&r,&cols,if mode=="batch" {4} else {1},ds.len()*cs.len()*4,64*1024*1024).map_err(fail)?;
        for &c in &cols {let m=cache.take(&r,c,&ds,&cs,&maps).map_err(fail)?;checksum+=m.iter().filter(|v|v.is_finite()).map(|&v|v as f64).sum::<f64>();}reads=cache.reads.load(std::sync::atomic::Ordering::Relaxed);
    }
    Ok((start.elapsed().as_secs_f64(),checksum,reads))
}
#[pymodule]
fn storage_sandbox(_: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(append_store,m)?)?;
    m.add_function(wrap_pyfunction!(bench_read,m)?)?;
    m.add_function(wrap_pyfunction!(read_matrix,m)?)?;
    m.add_function(wrap_pyfunction!(storage_policy::factor_store_storage_profile,m)?)?;
    m.add_function(wrap_pyfunction!(factor_store_v5::factor_store_v5_info,m)?)?;
    m.add_function(wrap_pyfunction!(factor_store_v5::factor_store_v5_read_factor,m)?)?;
    m.add_function(wrap_pyfunction!(factor_store_v5::factor_store_v5_smoke_proj,m)?)?;
    m.add_function(wrap_pyfunction!(factor_store_v5::factor_store_v5_register_group,m)?)?;
    Ok(())
}
