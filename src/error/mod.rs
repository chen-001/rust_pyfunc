use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

#[derive(Debug)]
pub struct TimeoutError {
    pub message: String,
    pub duration: f64,
}

impl std::fmt::Display for TimeoutError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "计算超时: {} (耗时 {}秒)", self.message, self.duration)
    }
}

impl std::error::Error for TimeoutError {}

impl From<TimeoutError> for PyErr {
    fn from(err: TimeoutError) -> PyErr {
        PyRuntimeError::new_err(format!("计算超时: {} (耗时 {}秒)", err.message, err.duration))
    }
}
