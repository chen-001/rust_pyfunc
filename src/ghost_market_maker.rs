use log::warn;
use nalgebra::DMatrix;
use ndarray::Array2;
use numpy::PyReadonlyArray2;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// 计算幽灵做市商因子（Ghost Market Maker's Regret）
///
/// 核心逻辑：强制割裂买卖两侧的联动，测量盘口僵硬度
/// 通过计算买卖盘口之间的"量子纠缠程度"来识别变盘信号
///
/// 参数:
///     bid_ask_volumes: 买卖盘量矩阵 (n_samples, 20)
///                      前10列为买方量(bid_vol1-10)，后10列为卖方量(ask_vol1-10)
///     window: 计算协方差的窗口大小
///
/// 返回:
///     因子值，衡量买卖盘口的量子纠缠程度
#[pyfunction]
#[pyo3(name = "calculate_ghost_market_maker_factor_rust")]
pub fn calculate_ghost_market_maker_factor_py(
    _py: Python<'_>,
    bid_ask_volumes: PyReadonlyArray2<f64>,
    window: usize,
) -> PyResult<f64> {
    let volumes = bid_ask_volumes.as_array();
    let factor = calculate_ghost_market_maker_factor(&volumes, window)
        .map_err(|e| PyValueError::new_err(e))?;
    Ok(factor)
}

/// Rust原生实现的核心计算函数
pub fn calculate_ghost_market_maker_factor(
    bid_ask_volumes: &ndarray::ArrayView2<f64>,
    window: usize,
) -> Result<f64, String> {
    // 检查数据形状
    let (n_samples, n_features) = bid_ask_volumes.dim();

    if n_features != 20 {
        return Err(format!(
            "输入数据必须是20列（10档买量+10档卖量），当前为{}列",
            n_features
        ));
    }

    if n_samples < window + 1 {
        return Ok(0.0);
    }

    // 计算对数变换（避免log(0)）
    let log_volumes: Array2<f64> = bid_ask_volumes.map(|&x| (x + 1.0).ln());

    // 计算差分（变化量）
    let mut log_diff = Array2::zeros((n_samples - 1, n_features));
    for i in 0..(n_samples - 1) {
        for j in 0..n_features {
            log_diff[[i, j]] = log_volumes[[i + 1, j]] - log_volumes[[i, j]];
        }
    }

    // 如果数据量不足，返回0
    if log_diff.nrows() < window {
        return Ok(0.0);
    }

    // 计算协方差矩阵 (20x20)
    let cov_matrix = calculate_covariance(&log_diff, window)?;

    // 检查矩阵是否正定，如果奇异则添加正则化
    if !is_positive_definite(&cov_matrix) {
        warn!("协方差矩阵接近奇异，添加正则化");
        let cov_matrix = add_regularization(cov_matrix, 1e-8)?;
        return calculate_factor_from_matrix(cov_matrix);
    }

    calculate_factor_from_matrix(cov_matrix)
}

/// 计算协方差矩阵
fn calculate_covariance(data: &Array2<f64>, window: usize) -> Result<Array2<f64>, String> {
    let n_rows = data.nrows();
    let n_cols = data.ncols();

    // 使用最近的window条数据
    let start_idx = if n_rows > window { n_rows - window } else { 0 };
    let window_data = data.slice(ndarray::s![start_idx.., ..]);

    // 计算均值
    let mut means = Vec::with_capacity(n_cols);
    for j in 0..n_cols {
        let sum: f64 = window_data.column(j).iter().sum();
        means.push(sum / window_data.nrows() as f64);
    }

    // 计算协方差矩阵
    let mut cov = Array2::zeros((n_cols, n_cols));

    for i in 0..n_cols {
        for j in 0..=i {
            let mut sum = 0.0;
            for k in 0..window_data.nrows() {
                sum += (window_data[[k, i]] - means[i]) * (window_data[[k, j]] - means[j]);
            }

            let cov_value = sum / (window_data.nrows() - 1) as f64;
            cov[[i, j]] = cov_value;
            cov[[j, i]] = cov_value; // 对称性
        }
    }

    Ok(cov)
}

/// 检查矩阵是否正定
fn is_positive_definite(matrix: &Array2<f64>) -> bool {
    let eigenvals = calculate_eigenvalues(matrix);
    eigenvals.iter().all(|&x| x > 1e-10)
}

/// 计算特征值（简化版本，用于检查正定性）
fn calculate_eigenvalues(matrix: &Array2<f64>) -> Vec<f64> {
    // 转换为nalgebra矩阵
    let n = matrix.nrows();
    let mut mat = DMatrix::zeros(n, n);

    for i in 0..n {
        for j in 0..n {
            mat[(i, j)] = matrix[[i, j]];
        }
    }

    // 计算特征值
    let eigen = mat.symmetric_eigen();
    eigen.eigenvalues.iter().map(|x| *x).collect()
}

/// 添加正则化
fn add_regularization(mut matrix: Array2<f64>, lambda: f64) -> Result<Array2<f64>, String> {
    let n = matrix.nrows();
    for i in 0..n {
        matrix[[i, i]] += lambda;
    }
    Ok(matrix)
}

/// 从协方差矩阵计算因子值
fn calculate_factor_from_matrix(cov_matrix: Array2<f64>) -> Result<f64, String> {
    // 分割矩阵
    let sigma_bb = cov_matrix.slice(ndarray::s![0..10, 0..10]).to_owned();
    let sigma_aa = cov_matrix.slice(ndarray::s![10..20, 10..20]).to_owned();
    // let sigma_ba = cov_matrix.slice(ndarray::s![0..10, 10..20]).to_owned();
    // let sigma_ab = cov_matrix.slice(ndarray::s![10..20, 0..10]).to_owned();

    // 构建割裂矩阵
    let mut sigma_split = Array2::zeros((20, 20));
    for i in 0..10 {
        for j in 0..10 {
            sigma_split[[i, j]] = sigma_bb[[i, j]];
            sigma_split[[i + 10, j + 10]] = sigma_aa[[i, j]];
        }
    }

    // 计算行列式
    let log_det_full = calculate_log_determinant(&cov_matrix)?;
    let log_det_split = calculate_log_determinant(&sigma_split)?;

    // 计算因子值
    let factor = log_det_split - log_det_full;

    Ok(factor)
}

/// 计算对数行列式（处理奇异矩阵）
fn calculate_log_determinant(matrix: &Array2<f64>) -> Result<f64, String> {
    // 使用特征值计算对数行列式
    let eigenvals = calculate_eigenvalues(matrix);

    // 过滤掉接近零的特征值
    let log_sum: f64 = eigenvals
        .iter()
        .filter(|&&x| x > 1e-10)
        .map(|x| x.ln())
        .sum();

    if log_sum.is_nan() || log_sum.is_infinite() {
        return Err("行列式计算失败，矩阵可能过于奇异".to_string());
    }

    Ok(log_sum)
}

/// 计算行列式（备用方法）
#[allow(dead_code)]
fn calculate_determinant(matrix: &Array2<f64>) -> Result<f64, String> {
    let n = matrix.nrows();
    let mut mat = DMatrix::zeros(n, n);

    for i in 0..n {
        for j in 0..n {
            mat[(i, j)] = matrix[[i, j]];
        }
    }

    let det = mat.determinant();
    Ok(det)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    #[test]
    fn test_calculate_ghost_market_maker_factor() {
        // 创建测试数据
        let mut data = Array2::zeros((100, 20));

        // 填充一些测试数据
        for i in 0..100 {
            for j in 0..20 {
                data[[i, j]] = (i + 1) as f64 * (j + 1) as f64 + (i as f64 * 0.1);
            }
        }

        let result = calculate_ghost_market_maker_factor(&data.view(), 60);
        assert!(result.is_ok());

        let factor = result.unwrap();
        println!("测试因子值: {}", factor);
    }

    #[test]
    fn test_positive_definite_check() {
        let matrix = arr2(&[[2.0, 1.0], [1.0, 2.0]]);
        assert!(is_positive_definite(&matrix));
    }
}
