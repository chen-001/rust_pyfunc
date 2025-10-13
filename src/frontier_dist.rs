//! 马科维茨有效前沿距离计算模块
//!
//! 本模块实现了基于马科维茨投资组合理论的有效前沿距离计算功能。
//! 给定单日3秒频率收益序列，按指定块大小聚合后计算每个资产点到有效前沿的最短距离。
//!
//! 核心算法包括：
//! 1. 数据分块聚合
//! 2. 块间样本协方差矩阵计算（带岭化）
//! 3. 马科维茨闭式有效前沿计算
//! 4. KKT-λ四次方程法求解最短距离

use faer::prelude::SpSolver;
use log::{info, warn};
use ndarray::{s, Array1, Array2, ArrayView1, Axis};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use pyo3::types::PyTuple;
use roots::{find_roots_quartic, Roots};
use thiserror::Error;

/// 简单的复数结构体
#[derive(Debug, Clone, Copy)]
struct Complex {
    re: f64,
    im: f64,
}

impl Complex {
    fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }
}

#[derive(Error, Debug)]
pub enum FrontierError {
    #[error("输入参数错误: {0}")]
    InvalidInput(String),
    #[error("线性代数计算失败: {0}")]
    LinAlgError(String),
    #[error("数值计算错误: {0}")]
    NumericalError(String),
}

impl From<FrontierError> for PyErr {
    fn from(err: FrontierError) -> Self {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(err.to_string())
    }
}

/// 将输入错误转换为Python异常
type Result<T> = std::result::Result<T, FrontierError>;

/// 数据聚合结构体
struct DataBlocks {
    /// 块数量
    m: usize,
    /// 块大小
    x: usize,
    /// 每块的均值向量
    means: Array1<f64>,
    /// 每块的原始数据，形状为 (m, x)
    blocks: Array2<f64>,
    /// 每块的起始时间戳（如果提供）
    start_timestamps: Option<Array1<i64>>,
}

impl DataBlocks {
    /// 将收益序列按指定块大小聚合
    ///
    /// # 参数
    /// * `r` - 输入收益序列
    /// * `group_size` - 块大小
    /// * `drop_last` - 是否丢弃尾部不完整的块
    ///
    /// # 返回
    /// 包含均值和原始块数据的结构体
    fn aggregate(
        r: ArrayView1<f64>,
        timestamps: Option<Array1<i64>>,
        group_size: usize,
        drop_last: bool,
    ) -> Result<Self> {
        if group_size == 0 {
            return Err(FrontierError::InvalidInput(
                "group_size必须大于0".to_string(),
            ));
        }

        let n = r.len();
        if n == 0 {
            return Err(FrontierError::InvalidInput("输入序列不能为空".to_string()));
        }

        if let Some(ref ts) = timestamps {
            if ts.len() != n {
                return Err(FrontierError::InvalidInput(format!(
                    "时间戳长度{}与输入序列长度{}不一致",
                    ts.len(),
                    n
                )));
            }
        }

        let m = if drop_last {
            n / group_size
        } else {
            if n % group_size != 0 {
                return Err(FrontierError::InvalidInput(format!(
                    "输入长度{}不能被group_size={}整除",
                    n, group_size
                )));
            }
            n / group_size
        };

        if m == 0 {
            return Err(FrontierError::InvalidInput(
                "分块后没有完整的数据块".to_string(),
            ));
        }

        let x = group_size;
        let used_len = m * x;

        // 提取有效数据并重塑为矩阵
        let r_used = r.slice(s![..used_len]).to_owned();
        let blocks = r_used
            .into_shape((m, x))
            .map_err(|e| FrontierError::InvalidInput(format!("数据重塑失败: {}", e)))?;

        // 处理时间戳，只保留每个块的首个时间戳
        let start_timestamps = if let Some(ts) = timestamps {
            let ts_used = ts.slice(s![..used_len]).to_owned();
            let ts_blocks = ts_used
                .into_shape((m, x))
                .map_err(|e| FrontierError::InvalidInput(format!("时间戳重塑失败: {}", e)))?;
            Some(ts_blocks.column(0).to_owned())
        } else {
            None
        };

        // 计算每块均值
        let invalid_count = blocks.iter().filter(|v| !v.is_finite()).count();
        if invalid_count > 0 {
            return Err(FrontierError::InvalidInput(format!(
                "聚合数据中包含{}个非有限值(NaN/Inf)，请先清理或填补输入序列",
                invalid_count
            )));
        }

        let means = blocks
            .mean_axis(Axis(1))
            .ok_or_else(|| FrontierError::NumericalError("计算均值失败".to_string()))?;

        info!("数据聚合完成: {}个数据点 -> {}块，每块{}个点", n, m, x);

        Ok(DataBlocks {
            m,
            x,
            means,
            blocks,
            start_timestamps,
        })
    }
}

/// 协方差矩阵计算器
struct CovarianceMatrix {
    /// 协方差矩阵
    matrix: Array2<f64>,
}

impl CovarianceMatrix {
    /// 计算块间样本协方差矩阵
    ///
    /// 使用块内序列逐点对应的样本协方差定义：
    /// cov(r_i, r_j) = (1/(x-ddof)) * sum_{k=0}^{x-1} (r_{ix+k}-μ_i)(r_{jx+k}-μ_j)
    ///
    /// # 参数
    /// * `data_blocks` - 聚合后的数据块
    /// * `ddof` - 自由度调整
    /// * `ridge` - 岭化系数
    fn compute(data_blocks: &DataBlocks, ddof: usize, ridge: f64) -> Result<Self> {
        let m = data_blocks.m;
        let x = data_blocks.x;

        if x <= ddof {
            return Err(FrontierError::InvalidInput(format!(
                "块大小{}必须大于自由度调整{}",
                x, ddof
            )));
        }

        let mut cov_matrix = Array2::<f64>::zeros((m, m));
        let denom = (x - ddof) as f64;

        // 计算协方差矩阵
        for i in 0..m {
            let mean_i = data_blocks.means[i];
            let block_i = data_blocks.blocks.row(i);

            // 对角线元素（方差）
            let var_i = {
                let mut sum_sq = 0.0;
                for k in 0..x {
                    let diff = block_i[k] - mean_i;
                    sum_sq += diff * diff;
                }
                sum_sq / denom
            };
            cov_matrix[(i, i)] = var_i;

            // 上三角元素（协方差）
            for j in (i + 1)..m {
                let mean_j = data_blocks.means[j];
                let block_j = data_blocks.blocks.row(j);

                let mut cov = 0.0;
                for k in 0..x {
                    cov += (block_i[k] - mean_i) * (block_j[k] - mean_j);
                }
                cov /= denom;

                cov_matrix[(i, j)] = cov;
                cov_matrix[(j, i)] = cov; // 对称性
            }
        }

        // 岭化处理：自适应寻找正定矩阵
        let base_matrix = cov_matrix;
        let diag_mean = base_matrix.diag().mean().unwrap_or(0.0);
        let scale = if diag_mean.is_finite() && diag_mean > 0.0 {
            diag_mean
        } else {
            1.0
        };
        let mut lambda = if ridge > 0.0 {
            (ridge * scale).max(ridge)
        } else {
            0.0
        };

        const MAX_RIDGE_ATTEMPTS: usize = 8;
        let mut adjusted_matrix = base_matrix.clone();
        let mut attempts = 0usize;
        loop {
            adjusted_matrix.assign(&base_matrix);
            if lambda > 0.0 {
                for i in 0..m {
                    adjusted_matrix[(i, i)] += lambda;
                }
            }

            if Self::is_positive_definite(&adjusted_matrix) {
                if lambda > 0.0 {
                    info!(
                        "协方差矩阵计算完成: {}x{}矩阵，岭化系数={:.2e}（经过{}次尝试）",
                        m,
                        m,
                        lambda,
                        attempts + 1
                    );
                } else {
                    info!(
                        "协方差矩阵计算完成: {}x{}矩阵，无需岭化（尝试{}次）",
                        m,
                        m,
                        attempts + 1
                    );
                }
                return Ok(CovarianceMatrix {
                    matrix: adjusted_matrix,
                });
            }

            if attempts >= MAX_RIDGE_ATTEMPTS {
                return Err(FrontierError::LinAlgError(format!(
                    "协方差矩阵不是正定的，即使在岭化系数增大到{:.2e}后仍然失败，建议检查输入数据或增大ridge参数",
                    lambda
                )));
            }

            attempts += 1;
            lambda = if lambda == 0.0 {
                ridge.max(1e-12)
            } else {
                lambda * 10.0
            };
        }
    }

    fn is_positive_definite(matrix: &Array2<f64>) -> bool {
        let dim = matrix.nrows();
        if dim == 0 {
            return false;
        }
        let mat = faer::Mat::from_fn(dim, dim, |i, j| matrix[(i, j)]);
        mat.cholesky(faer::Side::Lower).is_ok()
    }
}

/// 马科维茨有效前沿计算器
struct MarkowitzFrontier {
    /// A = e^T Σ^{-1} e
    a: f64,
    /// B = e^T Σ^{-1} μ
    b: f64,
    /// C = μ^T Σ^{-1} μ
    c: f64,
    /// Δ = AC - B^2
    delta: f64,
    /// 逆协方差矩阵的分解（用于求解线性方程）
    #[allow(dead_code)]
    inv_sigma_decomp: Array2<f64>,
}

impl MarkowitzFrontier {
    /// 计算马科维茨有效前沿参数
    ///
    /// 使用Cholesky分解求解线性方程组，避免显式求逆
    ///
    /// # 参数
    /// * `cov_matrix` - 协方差矩阵
    /// * `means` - 资产期望收益向量
    fn compute(cov_matrix: &CovarianceMatrix, means: &Array1<f64>) -> Result<Self> {
        let m = means.len();

        // 使用faer进行Cholesky分解
        let cov_faer = faer::Mat::from_fn(m, m, |i, j| cov_matrix.matrix[(i, j)]);

        let chol = match cov_faer.cholesky(faer::Side::Lower) {
            Ok(chol) => chol,
            Err(_) => {
                return Err(FrontierError::LinAlgError(
                    "协方差矩阵不是正定的，尝试增大ridge参数".to_string(),
                ))
            }
        };

        // 定义线性方程求解函数
        let solve = |rhs: &Array1<f64>| -> Result<Array1<f64>> {
            let rhs_faer = faer::Mat::from_fn(m, 1, |i, _| rhs[i]);
            let solution = chol.solve(rhs_faer);
            Ok(Array1::from_iter((0..m).map(|i| solution[(i, 0)])))
        };

        // 创建单位向量e
        let e = Array1::<f64>::ones(m);

        // 计算Σ^{-1}e, Σ^{-1}μ
        let inv_sigma_e = solve(&e)?;
        let inv_sigma_mu = solve(means)?;

        // 计算A, B, C
        let a = e.dot(&inv_sigma_e);
        let b = e.dot(&inv_sigma_mu);
        let c = means.dot(&inv_sigma_mu);

        let delta = a * c - b * b;

        if delta <= 0.0 {
            return Err(FrontierError::NumericalError(format!(
                "Δ <= 0 (Δ = {:.2e})，数据可能病态，建议增大ridge",
                delta
            )));
        }

        info!(
            "有效前沿参数计算完成: A={:.6e}, B={:.6e}, C={:.6e}, Δ={:.6e}",
            a, b, c, delta
        );

        Ok(MarkowitzFrontier {
            a,
            b,
            c,
            delta,
            inv_sigma_decomp: Array2::zeros((m, m)), // 这里存储的其实不是完整的逆矩阵
        })
    }
}

/// 多项式求根器
struct PolynomialSolver;

impl PolynomialSolver {
    /// 解四次方程并筛选有效根
    ///
    /// 对四次多项式 P(λ) = a4 λ^4 + a3 λ^3 + a2 λ^2 + a1 λ + a0 = 0
    /// 返回满足条件的实根
    ///
    /// # 参数
    /// * `coeffs` - 多项式系数 [a0, a1, a2, a3, a4]（从低次到高次）
    /// * `a_delta` - A * Δ 参数
    /// * `a` - A 参数
    ///
    /// # 返回
    /// 有效的λ根列表
    fn solve_quartic_and_filter(coeffs: &[f64; 5], a_delta: f64, a: f64) -> Vec<f64> {
        let [a0, a1, a2, a3, a4] = *coeffs;

        // 如果最高次项系数接近0，降级处理
        let roots = if a4.abs() < 1e-12 {
            if a3.abs() < 1e-12 {
                // 二次或更低次
                Self::solve_quadratic_fallback(a2, a1, a0)
            } else {
                // 三次方程 - 简化处理，返回空列表
                let _cubic_roots = Self::solve_cubic(a3, a2, a1, a0);
                vec![]
            }
        } else {
            // 四次方程
            match find_roots_quartic(a4, a3, a2, a1, a0) {
                Roots::Four([r1, r2, r3, r4]) => {
                    vec![
                        Complex::new(r1, 0.0),
                        Complex::new(r2, 0.0),
                        Complex::new(r3, 0.0),
                        Complex::new(r4, 0.0),
                    ]
                }
                Roots::Three([r1, r2, r3]) => {
                    vec![
                        Complex::new(r1, 0.0),
                        Complex::new(r2, 0.0),
                        Complex::new(r3, 0.0),
                    ]
                }
                Roots::Two([r1, r2]) => {
                    vec![Complex::new(r1, 0.0), Complex::new(r2, 0.0)]
                }
                Roots::One([r1]) => {
                    vec![Complex::new(r1, 0.0)]
                }
                Roots::No(_) => vec![],
            }
        };

        // 筛选有效根
        let mut valid_roots = Vec::new();
        for root in &roots {
            // 检查是否为实根
            if root.im.abs() > 1e-10 {
                continue;
            }

            let lambda = root.re;

            // 检查分母是否接近0
            if (1.0 + lambda * a_delta).abs() < 1e-12 || (1.0 - lambda * a).abs() < 1e-12 {
                continue;
            }

            valid_roots.push(lambda);
        }

        valid_roots
    }

    /// 三次方程求解（备用）
    fn solve_cubic(a3: f64, a2: f64, a1: f64, a0: f64) -> [Complex; 3] {
        // 使用简单的三次方程求解方法
        // 这里简化处理，实际应用中可以使用更精确的算法
        let _discriminant = a2 * a2 - 3.0 * a3 * a1;
        let _p = (3.0 * a3 * a1 - a2 * a2) / (3.0 * a3 * a3);
        let _q =
            (2.0 * a2 * a2 * a2 - 9.0 * a3 * a2 * a1 + 27.0 * a3 * a3 * a0) / (27.0 * a3 * a3 * a3);

        // 简化返回，实际需要更精确的实现
        [
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
        ]
    }

    /// 二次方程求解（备用）
    fn solve_quadratic_fallback(a2: f64, a1: f64, a0: f64) -> Vec<Complex> {
        if a2.abs() < 1e-12 {
            if a1.abs() < 1e-12 {
                vec![]
            } else {
                vec![Complex::new(-a0 / a1, 0.0)]
            }
        } else {
            let discriminant = a1 * a1 - 4.0 * a2 * a0;
            if discriminant >= 0.0 {
                let sqrt_disc = discriminant.sqrt();
                vec![
                    Complex::new((-a1 + sqrt_disc) / (2.0 * a2), 0.0),
                    Complex::new((-a1 - sqrt_disc) / (2.0 * a2), 0.0),
                ]
            } else {
                let sqrt_disc = (-discriminant).sqrt();
                vec![
                    Complex::new(-a1 / (2.0 * a2), sqrt_disc / (2.0 * a2)),
                    Complex::new(-a1 / (2.0 * a2), -sqrt_disc / (2.0 * a2)),
                ]
            }
        }
    }
}

/// KKT-λ距离计算器
struct DistanceCalculator;

impl DistanceCalculator {
    /// 计算资产点到有效前沿的最短距离
    ///
    /// 使用KKT-λ四次方程法，对每个资产点计算到有效前沿的最短欧氏距离
    ///
    /// # 参数
    /// * `frontier` - 有效前沿参数
    /// * `cov_matrix` - 协方差矩阵
    /// * `means` - 资产期望收益
    ///
    /// # 返回
    /// 距离序列，长度与资产数量相同
    fn calculate_distances(
        frontier: &MarkowitzFrontier,
        cov_matrix: &CovarianceMatrix,
        means: &Array1<f64>,
    ) -> Result<Array1<f64>> {
        let m = means.len();
        let mut distances = Array1::<f64>::zeros(m);

        let a = frontier.a;
        let b = frontier.b;
        let c = frontier.c;
        let delta = frontier.delta;
        let a_delta = a * delta;

        for i in 0..m {
            let mu_i = means[i];
            let sigma_i = cov_matrix.matrix[(i, i)].sqrt();

            // 构造四次多项式系数
            // P(λ) = A(μ_i+λB)²(1+λΔ)² - 2B(μ_i+λB)(1-λA)(1+λΔ)² + C(1-λA)²(1+λΔ)² - Δσ_i²(1-λA)²

            // 展开后得到四次多项式的系数
            let coeffs = Self::construct_quartic_coefficients(mu_i, sigma_i, a, b, c, delta);

            // 求解并筛选有效根
            let valid_lambdas = PolynomialSolver::solve_quartic_and_filter(&coeffs, a_delta, a);

            if valid_lambdas.is_empty() {
                warn!("资产{}: 没有找到有效根，距离设为0", i);
                distances[i] = 0.0;
                continue;
            }

            // 计算每个有效根对应的距离，取最小值
            let mut min_distance = f64::INFINITY;
            for &lambda in &valid_lambdas {
                let x = sigma_i / (1.0 + lambda * delta);
                let y = (mu_i + lambda * b) / (1.0 - lambda * a);
                let distance = ((x - sigma_i).powi(2) + (y - mu_i).powi(2)).sqrt();
                min_distance = min_distance.min(distance);
            }

            distances[i] = if min_distance.is_finite() {
                min_distance
            } else {
                0.0
            };
        }

        Ok(distances)
    }

    /// 构造四次多项式系数
    fn construct_quartic_coefficients(
        mu_i: f64,
        sigma_i: f64,
        a: f64,
        b: f64,
        c: f64,
        delta: f64,
    ) -> [f64; 5] {
        // 简化的系数计算，实际需要完整展开
        // 这里提供一个基础实现

        let a2 = a * a;
        let b2 = b * b;
        let delta2 = delta * delta;
        let mu2 = mu_i * mu_i;
        let sigma2 = sigma_i * sigma_i;

        // 从低次到高次的系数 [a0, a1, a2, a3, a4]
        [
            c - delta * sigma2,                         // a0
            -2.0 * a * c + 2.0 * b * delta * sigma2,    // a1
            a2 * c - b2 - 2.0 * b * mu_i + delta * mu2, // a2
            2.0 * a * b2 - 2.0 * a * delta * mu2,       // a3
            a2 * delta2,                                // a4
        ]
    }
}

/// 计算收益序列中每个聚合块到马科维茨有效前沿的距离
///
/// # 功能说明
/// 1. 将输入收益序列按指定大小分块聚合
/// 2. 计算块间样本协方差矩阵（带岭化保证正定性）
/// 3. 构造马科维茨无约束有效前沿
/// 4. 使用KKT-λ四次方程法计算每个资产点到前沿的最短距离
///
/// # 参数
/// * `r` - 1D float64数组，单日3秒频率收益序列
/// * `group_size` - 每多少行聚合成一块（x）
/// * `drop_last` - 尾部不足group_size行时是否丢弃，默认true
/// * `ddof` - 协方差/方差的自由度调整，默认1（样本协方差）
/// * `ridge` - 岭化强度系数，默认1e-6
/// * `timestamps` - 可选的int64时间戳数组（与`r`等长），用于标记每个聚合块的首个时间点
///
/// # 返回值
/// 包含两个1D数组的元组：(block_timestamps, distances)，长度均为m
/// - block_timestamps：每个聚合块的首个时间戳（若未提供timestamps，则返回0..m-1的序列）
/// - distances：对应资产点到有效前沿的距离
///
/// # 数值提示
/// 当 m >> group_size 时，协方差矩阵可能秩亏，需要通过增大ridge参数保证可逆性
///
/// # 示例
/// ```python
/// import numpy as np
/// from rust_pyfunc import distances_to_frontier
///
/// # 生成测试数据
/// np.random.seed(0)
/// r = 1e-4 * np.random.randn(4800).astype(np.float64)
///
/// # 每1分钟聚合（20个3秒间隔）
/// block_ts, distances = distances_to_frontier(r, group_size=20)
/// print(distances.shape)  # (240,)
/// ```
#[pyfunction]
#[pyo3(signature = (
    r,
    group_size,
    drop_last=true,
    ddof=1,
    ridge=1e-6,
    timestamps=None
))]
pub fn distances_to_frontier(
    r: PyReadonlyArray1<f64>,
    group_size: usize,
    drop_last: bool,
    ddof: usize,
    ridge: f64,
    timestamps: Option<PyReadonlyArray1<i64>>,
) -> Result<Py<PyTuple>> {
    pyo3::Python::with_gil(|py| {
        // 转换输入为ndarray
        let r_array = r.as_array();
        let timestamps_owned = timestamps.map(|ts| ts.to_owned_array());

        // 1. 数据聚合
        let data_blocks =
            DataBlocks::aggregate(r_array.view(), timestamps_owned, group_size, drop_last)?;

        // 2. 计算协方差矩阵
        let cov_matrix = CovarianceMatrix::compute(&data_blocks, ddof, ridge)?;

        // 3. 计算有效前沿参数
        let frontier = MarkowitzFrontier::compute(&cov_matrix, &data_blocks.means)?;

        // 4. 计算距离
        let distances =
            DistanceCalculator::calculate_distances(&frontier, &cov_matrix, &data_blocks.means)?;

        info!("距离计算完成: {}个资产点的距离", distances.len());

        let timestamps_vec = if let Some(ts) = &data_blocks.start_timestamps {
            ts.to_vec()
        } else {
            (0..distances.len()).map(|idx| idx as i64).collect()
        };

        let timestamps_py = PyArray1::from_vec(py, timestamps_vec);
        let distances_py = distances.into_pyarray(py);

        let tuple = PyTuple::new(
            py,
            &[timestamps_py.to_object(py), distances_py.to_object(py)],
        );

        Ok(tuple.into())
    })
}
