//! Isolated relaynet benchmark: production package is never installed or changed.
#![allow(dead_code, unused_variables)]
pub mod data_paths;
pub mod fast_csv_reader;
pub mod yupei_dist;
pub mod relaynet_baseline;
pub mod relaynet_fast;
pub mod relaynet_candidate;

use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;

#[pyfunction]
fn run<'py>(py: Python<'py>, date: i64, variant: &str) -> PyResult<(Vec<String>, &'py PyArray1<f32>)> {
    let (codes, values) = py.allow_threads(|| match variant {
        "baseline" => relaynet_baseline::compute_relaynet_full(date),
        "fast" => relaynet_fast::compute_relaynet_full(date),
        "candidate" => relaynet_candidate::compute_relaynet_full(date),
        _ => Err(std::io::Error::new(std::io::ErrorKind::InvalidInput, "variant must be baseline, fast or candidate")),
    }).map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
    Ok((codes, values.into_pyarray(py)))
}

#[pyfunction]
fn names() -> Vec<String> { relaynet_baseline::relaynet_names() }

#[pyfunction]
fn py_relaynet_experimental(date: i64) -> PyResult<(Vec<String>, Vec<f32>)> {
    relaynet_fast::py_relaynet(date)
}

#[pyfunction]
fn matrix_probe<'py>(py: Python<'py>, input: PyReadonlyArray2<f64>, kind: u8, variant: &str) -> PyResult<&'py PyArray2<f64>> {
    let input = input.as_array().to_owned();
    let result = py.allow_threads(|| match variant {
        "baseline" => relaynet_baseline::matrix_probe(input, kind),
        _ => relaynet_fast::matrix_probe(input, kind),
    });
    Ok(result.into_pyarray(py))
}

#[pyfunction]
fn stat_probe(py: Python, n: usize, edges: Vec<(usize, usize, f32)>, variant: &str) -> Vec<Vec<f64>> {
    py.allow_threads(|| match variant {
        "baseline" => relaynet_baseline::stat_probe(n, edges),
        _ => relaynet_fast::stat_probe(n, edges),
    })
}

#[pyfunction]
fn graph_probe(py: Python) -> PyResult<String> {
    py.allow_threads(relaynet_fast::graph_probe)
        .map_err(pyo3::exceptions::PyAssertionError::new_err)
}


#[pyfunction]
fn prep_probe<'py>(py: Python<'py>, code: &str, date: i64, variant: &str) -> PyResult<&'py PyArray1<f64>> {
    let result = py.allow_threads(|| match variant {
        "baseline" => relaynet_baseline::prep_probe(code, date),
        _ => relaynet_fast::prep_probe(code, date),
    }).map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
    Ok(result.into_pyarray(py))
}

#[pyfunction]
fn exact_path_probe(py: Python, n: usize, edges: Vec<(usize, usize, f32)>) -> Vec<Vec<f64>> {
    py.allow_threads(|| relaynet_fast::exact_path_probe(n, edges))
}

#[pyfunction]
fn weighted_graph_probe(py: Python, path: &str, parents: bool) -> PyResult<Vec<Vec<f64>>> {
    py.allow_threads(|| relaynet_fast::weighted_graph_probe(path, parents))
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))
}

#[pyfunction]
fn graph_batch_probe(py: Python, batch: usize) -> Vec<Vec<f64>> {
    py.allow_threads(|| relaynet_fast::graph_batch_probe(batch))
}

#[pyfunction]
fn exact_path_experiment_probe(py: Python, n: usize, edges: Vec<(usize,usize,f32)>, mode: u8) -> Vec<Vec<f64>> {
    py.allow_threads(|| relaynet_fast::exact_path_experiment_probe(n,edges,mode))
}

#[pyfunction]
fn weighted_graph_experiment_probe(py: Python,path: &str,mode: u8)->PyResult<Vec<Vec<f64>>>{
    py.allow_threads(|| relaynet_fast::weighted_graph_experiment_probe(path,mode))
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))
}

#[pymodule]
fn relaynet_perf_20261102(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(graph_metrics_probe, m)?)?;
    m.add_function(wrap_pyfunction!(graph_build_probe, m)?)?;
    m.add_function(wrap_pyfunction!(graph_mst_probe, m)?)?;
    m.add_function(wrap_pyfunction!(run, m)?)?;
    m.add_function(wrap_pyfunction!(small_stats_graph_probe, m)?)?;
    m.add_function(wrap_pyfunction!(binary_direction_state_probe, m)?)?;
    m.add_function(wrap_pyfunction!(relaynet_candidate::py_relaynet, m)?)?;
    m.add_function(wrap_pyfunction!(relaynet_candidate::py_relaynet_names, m)?)?;
    m.add_function(wrap_pyfunction!(py_relaynet_experimental, m)?)?;
    m.add_function(wrap_pyfunction!(directed_local_probe, m)?)?;
    m.add_function(wrap_pyfunction!(directed_local_graph_probe, m)?)?;
    m.add_function(wrap_pyfunction!(binary_probe, m)?)?;
    m.add_function(wrap_pyfunction!(binary_graph_probe, m)?)?;
    m.add_function(wrap_pyfunction!(directed_probe, m)?)?;
    m.add_function(wrap_pyfunction!(names, m)?)?;
    m.add_function(wrap_pyfunction!(matrix_probe, m)?)?;
    m.add_function(wrap_pyfunction!(stat_probe, m)?)?;
    m.add_function(wrap_pyfunction!(graph_probe, m)?)?;
    m.add_function(wrap_pyfunction!(prep_probe, m)?)?;
    m.add_function(wrap_pyfunction!(exact_path_probe, m)?)?;
    m.add_function(wrap_pyfunction!(weighted_graph_probe, m)?)?;
    m.add_function(wrap_pyfunction!(graph_batch_probe, m)?)?;
    m.add_function(wrap_pyfunction!(exact_path_experiment_probe, m)?)?;
    m.add_function(wrap_pyfunction!(weighted_graph_experiment_probe, m)?)?;
    Ok(())
}

#[pyfunction]
fn graph_metrics_probe(py: Python, input: PyReadonlyArray2<f64>, old: bool) -> Vec<Vec<f64>> {
    let input = input.as_array().to_owned();
    py.allow_threads(|| relaynet_fast::graph_metrics_probe(input, old))
}

#[pyfunction]
fn graph_build_probe<'py>(py: Python<'py>, input: PyReadonlyArray2<f64>, k: usize, mutual: bool) -> &'py PyArray2<f64> {
    let input = input.as_array().to_owned();
    py.allow_threads(|| relaynet_fast::graph_build_probe(input, k, mutual)).into_pyarray(py)
}

#[pyfunction]
fn graph_mst_probe<'py>(py: Python<'py>, input: PyReadonlyArray2<f64>) -> &'py PyArray2<f64> {
    let input = input.as_array().to_owned();
    py.allow_threads(|| relaynet_fast::graph_mst_probe(input)).into_pyarray(py)
}


#[pyfunction]
fn directed_probe(py: Python, n: usize, edges: Vec<(usize, usize, f32)>, old: bool) -> (bool, Vec<Vec<f64>>) {
    py.allow_threads(|| relaynet_fast::directed_probe(n, edges, old))
}


#[pyfunction]
fn binary_probe(py: Python, n: usize, edges: Vec<(usize, usize, f32)>, simd: bool) -> Vec<f64> {
    py.allow_threads(|| relaynet_fast::binary_probe(n, edges, simd))
}

#[pyfunction]
fn binary_graph_probe(py: Python, path: &str, simd: bool) -> PyResult<Vec<f64>> {
    py.allow_threads(|| relaynet_fast::binary_graph_probe(path, simd))
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))
}


#[pyfunction]
fn directed_local_probe(py:Python,n:usize,edges:Vec<(usize,usize,f32)>,old:bool)->Vec<Vec<f64>> {
    py.allow_threads(||relaynet_fast::directed_local_probe(n,edges,old))
}

#[pyfunction]
fn directed_local_graph_probe(py:Python,path:&str,old:bool)->PyResult<Vec<Vec<f64>>> {
    py.allow_threads(||relaynet_fast::directed_local_graph_probe(path,old))
        .map_err(|e|pyo3::exceptions::PyIOError::new_err(e.to_string()))
}


#[pyfunction]
fn binary_direction_state_probe(py:Python,n:usize,edges:Vec<(usize,usize,f32)>,src:usize,direction:bool,force_pull:bool)
    ->(Vec<usize>,Vec<f64>,Vec<i32>)
{
    py.allow_threads(||relaynet_fast::binary_direction_state_probe(n,edges,src,direction,force_pull))
}


#[pyfunction]
fn small_stats_graph_probe(py: Python, path: &str) -> PyResult<Vec<Vec<f64>>> {
    py.allow_threads(|| relaynet_fast::small_stats_graph_probe(path))
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))
}
