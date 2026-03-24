//! Copula函数模块
//!
//! 实现4种常用Copula:
//! - Gaussian Copula: 对称依赖，无尾部依赖
//! - t-Copula: 对称依赖，有尾部依赖
//! - Clayton Copula: 下尾依赖（左尾）
//! - Gumbel Copula: 上尾依赖（右尾）

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rand::prelude::*;
use std::f64::consts::PI;

// ==================== 辅助函数 ====================

/// 标准正态分布CDF (使用误差函数近似)
fn normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / std::f64::consts::SQRT_2))
}

/// 标准正态分布逆CDF (Beasley-Springer-Moro算法)
fn normal_inv_cdf(p: f64) -> f64 {
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }

    let a = [
        -3.969683028665376e+01,
        2.209460984245205e+02,
        -2.759285104469687e+02,
        1.383577518672690e+02,
        -3.066479806614716e+01,
        2.506628277459239e+00,
    ];
    let b = [
        -5.447609879822406e+01,
        1.615858368580409e+02,
        -1.556989798598866e+02,
        6.680131188771972e+01,
        -1.328068155288572e+01,
    ];
    let c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
        4.374664141464968e+00,
        2.938163982698783e+00,
    ];
    let d = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e+00,
        3.754408661907416e+00,
    ];

    let p_low = 0.02425;
    let p_high = 1.0 - p_low;

    if p < p_low {
        let q = (-2.0 * p.ln()).sqrt();
        let mut sum = c[0];
        for i in 1..6 {
            sum = sum * q + c[i];
        }
        let mut sum2 = d[0];
        for i in 1..4 {
            sum2 = sum2 * q + d[i];
        }
        sum / (sum2 * q + 1.0)
    } else if p <= p_high {
        let q = p - 0.5;
        let r = q * q;
        let mut sum = a[0];
        for i in 1..6 {
            sum = sum * r + a[i];
        }
        let mut sum2 = b[0];
        for i in 1..5 {
            sum2 = sum2 * r + b[i];
        }
        q * sum / (sum2 * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        let mut sum = c[0];
        for i in 1..6 {
            sum = sum * q + c[i];
        }
        let mut sum2 = d[0];
        for i in 1..4 {
            sum2 = sum2 * q + d[i];
        }
        -(sum / (sum2 * q + 1.0))
    }
}

/// 误差函数近似
fn erf(x: f64) -> f64 {
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();

    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

    sign * y
}

/// Gamma函数
fn gamma(x: f64) -> f64 {
    if x <= 0.0 {
        return f64::NAN;
    }
    ln_gamma(x).exp()
}

/// 不完全Gamma函数 (用于t分布)
fn incomplete_gamma(a: f64, x: f64) -> f64 {
    if x < 0.0 || a <= 0.0 {
        return 0.0;
    }
    if x == 0.0 {
        return 0.0;
    }

    let gam = gamma(a);
    if x < a + 1.0 {
        let mut ap = a;
        let mut sum = 1.0 / a;
        let mut del = sum;
        for _ in 0..200 {
            ap += 1.0;
            del *= x / ap;
            sum += del;
            if del.abs() < sum.abs() * 1e-12 {
                break;
            }
        }
        sum * (-x + a * x.ln() - gam.ln()).exp()
    } else {
        let mut b = x + 1.0 - a;
        let mut c = 1.0 / f64::MIN_POSITIVE;
        let mut d = 1.0 / b;
        let mut h = d;
        for i in 1..200 {
            let an = -i as f64 * (i as f64 - a);
            b += 2.0;
            d = an * d + b;
            if d.abs() < f64::MIN_POSITIVE {
                d = f64::MIN_POSITIVE;
            }
            c = b + an / c;
            if c.abs() < f64::MIN_POSITIVE {
                c = f64::MIN_POSITIVE;
            }
            d = 1.0 / d;
            h *= d * c;
            if (d * c - 1.0).abs() < 1e-12 {
                break;
            }
        }
        1.0 - (-x + a * x.ln() - gam.ln()).exp() * h
    }
}

/// 正则化不完全Beta函数 I_x(a, b)
fn regularized_incomplete_beta(x: f64, a: f64, b: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }

    let ln_beta = ln_gamma(a) + ln_gamma(b) - ln_gamma(a + b);
    let front = (a * x.ln() + b * (1.0 - x).ln() - ln_beta).exp() / a;

    // Use continued fraction (Lentz's method)
    if x < (a + 1.0) / (a + b + 2.0) {
        front * beta_cf(x, a, b)
    } else {
        1.0 - {
            let front2 = (b * (1.0 - x).ln() + a * x.ln() - ln_beta).exp() / b;
            front2 * beta_cf(1.0 - x, b, a)
        }
    }
}

fn beta_cf(x: f64, a: f64, b: f64) -> f64 {
    let max_iter = 200;
    let eps = 1e-14;
    let tiny = 1e-30;

    let mut c = 1.0;
    let mut d = 1.0 / (1.0 - (a + b) * x / (a + 1.0)).max(tiny);
    let mut h = d;

    for m in 1..=max_iter {
        let m_f64 = m as f64;

        // Even step
        let num = m_f64 * (b - m_f64) * x / ((a + 2.0 * m_f64 - 1.0) * (a + 2.0 * m_f64));
        d = 1.0 / (1.0 + num * d).max(tiny);
        c = (1.0 + num / c).max(tiny);
        h *= d * c;

        // Odd step
        let num =
            -((a + m_f64) * (a + b + m_f64) * x) / ((a + 2.0 * m_f64) * (a + 2.0 * m_f64 + 1.0));
        d = 1.0 / (1.0 + num * d).max(tiny);
        c = (1.0 + num / c).max(tiny);
        h *= d * c;

        if (d * c - 1.0).abs() < eps {
            break;
        }
    }
    h
}

/// t分布CDF
fn t_cdf(t: f64, df: f64) -> f64 {
    if df <= 0.0 {
        return f64::NAN;
    }
    let x = df / (df + t * t);
    let ibeta = regularized_incomplete_beta(x, df / 2.0, 0.5);
    if t >= 0.0 {
        1.0 - 0.5 * ibeta
    } else {
        0.5 * ibeta
    }
}

/// 二元正态分布CDF (1D条件积分 + Simpson法则)
fn bivariate_normal_cdf(h1: f64, h2: f64, rho: f64) -> f64 {
    if rho.abs() >= 1.0 {
        if rho >= 1.0 {
            return normal_cdf(h1.min(h2));
        } else {
            return (normal_cdf(h1) + normal_cdf(h2) - 1.0).max(0.0);
        }
    }

    let sqrt_1_rho2 = (1.0 - rho * rho).sqrt();
    let lower = -8.0_f64;
    let upper = h1;
    let n = 200; // must be even for Simpson's rule
    let h = (upper - lower) / n as f64;

    let integrand = |x: f64| -> f64 {
        let phi_x = (-0.5 * x * x).exp() / (2.0 * PI).sqrt();
        let arg = (h2 - rho * x) / sqrt_1_rho2;
        phi_x * normal_cdf(arg)
    };

    // Simpson's rule
    let mut sum = integrand(lower) + integrand(upper);
    for i in 1..n {
        let x = lower + i as f64 * h;
        if i % 2 == 0 {
            sum += 2.0 * integrand(x);
        } else {
            sum += 4.0 * integrand(x);
        }
    }
    (sum * h / 3.0).clamp(0.0, 1.0)
}

// ==================== Gaussian Copula ====================

/// Gaussian Copula CDF
/// 参数:
/// - u1, u2: 均匀分布值 [0, 1]
/// - rho: 相关系数 [-1, 1]
fn gaussian_copula_cdf(u1: f64, u2: f64, rho: f64) -> f64 {
    if rho.abs() >= 1.0 {
        return u1.min(u2);
    }
    let h1 = normal_inv_cdf(u1);
    let h2 = normal_inv_cdf(u2);
    bivariate_normal_cdf(h1, h2, rho)
}

/// Gaussian Copula PDF
fn gaussian_copula_pdf(u1: f64, u2: f64, rho: f64) -> f64 {
    if rho.abs() >= 1.0 {
        return f64::NAN;
    }
    let h1 = normal_inv_cdf(u1);
    let h2 = normal_inv_cdf(u2);
    let d = 1.0 - rho * rho;
    let exponent = (2.0 * rho * h1 * h2 - rho * rho * (h1 * h1 + h2 * h2)) / (2.0 * d);
    exponent.exp() / d.sqrt()
}

/// Gaussian Copula 采样
fn gaussian_copula_sample(rho: f64, n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut rng = rand::thread_rng();
    let mut u1 = Vec::with_capacity(n);
    let mut u2 = Vec::with_capacity(n);

    for _ in 0..n {
        let z1: f64 = rng.gen();
        let z2: f64 = rng.gen();
        let x1 = normal_inv_cdf(z1);
        let x2 = rho * x1 + (1.0 - rho * rho).sqrt() * normal_inv_cdf(z2);
        u1.push(normal_cdf(x1));
        u2.push(normal_cdf(x2));
    }
    (u1, u2)
}

// ==================== t-Copula ====================

/// t-Copula CDF
/// 参数:
/// - u1, u2: 均匀分布值 [0, 1]
/// - rho: 相关系数 [-1, 1]
/// - df: 自由度
fn t_copula_cdf(u1: f64, u2: f64, rho: f64, df: f64) -> f64 {
    if rho.abs() >= 1.0 {
        return u1.min(u2);
    }
    // 简化实现：使用数值积分
    let t1 = t_quantile(u1, df);
    let t2 = t_quantile(u2, df);
    bivariate_t_cdf(t1, t2, rho, df)
}

/// t分布分位数（利用与Beta分布的关系直接求解，避免牛顿迭代）
fn t_quantile(p: f64, df: f64) -> f64 {
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }
    if (p - 0.5).abs() < 1e-15 {
        return 0.0;
    }

    // t分位数与不完全Beta函数的关系:
    // 若p < 0.5: x_beta = I^{-1}(2p; df/2, 1/2), t = -sqrt(df*(1/x_beta - 1))
    // 若p > 0.5: x_beta = I^{-1}(2(1-p); df/2, 1/2), t = sqrt(df*(1/x_beta - 1))
    let (q, sign) = if p < 0.5 {
        (2.0 * p, -1.0)
    } else {
        (2.0 * (1.0 - p), 1.0)
    };

    // 反向不完全Beta: 求x使得I_x(a,b) = q，用牛顿迭代
    let a = df / 2.0;
    let b = 0.5;
    let mut x = inv_regularized_incomplete_beta(q, a, b);
    x = x.clamp(1e-15, 1.0 - 1e-15);

    sign * (df * (1.0 / x - 1.0)).sqrt()
}

/// 反向正则化不完全Beta函数: 求x使得I_x(a,b) = p
fn inv_regularized_incomplete_beta(p: f64, a: f64, b: f64) -> f64 {
    if p <= 0.0 {
        return 0.0;
    }
    if p >= 1.0 {
        return 1.0;
    }

    // 初始猜测: 使用正态近似
    let ln_a = ln_gamma(a);
    let ln_b = ln_gamma(b);
    let ln_ab = ln_gamma(a + b);

    let mut x = if a >= 1.0 && b >= 1.0 {
        let pp = if p < 0.5 { p } else { 1.0 - p };
        let t = (-2.0 * pp.ln()).sqrt();
        let s = t - (2.30753 + 0.27061 * t) / (1.0 + (0.99229 + 0.04481 * t) * t);
        let s = if p < 0.5 { -s } else { s };
        let lam = (s * s - 3.0) / 6.0;
        let h = 2.0 / (1.0 / (2.0 * a - 1.0) + 1.0 / (2.0 * b - 1.0));
        let w = s * (h + lam).sqrt() / h
            - (1.0 / (2.0 * b - 1.0) - 1.0 / (2.0 * a - 1.0)) * (lam + 5.0 / 6.0 - 2.0 / (3.0 * h));
        a / (a + b * (2.0 * w).exp())
    } else {
        let lna = (a / (a + b)).ln();
        let lnb = (b / (a + b)).ln();
        let t = (a * lna).exp() / a;
        let u = (b * lnb).exp() / b;
        let w = t + u;
        if p < t / w {
            (a * w * p).powf(1.0 / a)
        } else {
            1.0 - (b * w * (1.0 - p)).powf(1.0 / b)
        }
    };

    x = x.clamp(1e-15, 1.0 - 1e-15);

    // Halley迭代（比牛顿更快收敛）
    let afac = -(ln_a + ln_b - ln_ab);
    for _ in 0..20 {
        let err = regularized_incomplete_beta(x, a, b) - p;
        if err.abs() < 1e-14 {
            break;
        }
        let t = ((a - 1.0) * x.ln() + (b - 1.0) * (1.0 - x).ln() + afac).exp();
        if t == 0.0 {
            break;
        }
        let u = err / t;
        let corr = u / (1.0 - 0.5 * (u * ((a - 1.0) / x - (b - 1.0) / (1.0 - x))).min(1.0));
        x -= corr;
        x = x.clamp(1e-15, 1.0 - 1e-15);
    }
    x
}

/// t分布PDF（使用对数Gamma避免溢出）
fn t_pdf(x: f64, df: f64) -> f64 {
    let log_coef = ln_gamma((df + 1.0) / 2.0) - ln_gamma(df / 2.0) - 0.5 * (df * PI).ln();
    (log_coef - (df + 1.0) / 2.0 * (1.0 + x * x / df).ln()).exp()
}

/// 二元t分布CDF (1D条件积分 + Simpson法则)
fn bivariate_t_cdf(t1: f64, t2: f64, rho: f64, df: f64) -> f64 {
    if rho.abs() >= 1.0 {
        if rho >= 1.0 {
            return t_cdf(t1.min(t2), df);
        } else {
            return (t_cdf(t1, df) + t_cdf(t2, df) - 1.0).max(0.0);
        }
    }

    let sqrt_1_rho2 = (1.0 - rho * rho).sqrt();
    let lower = -15.0_f64;
    let upper = t1;
    let n = 200;
    let h = (upper - lower) / n as f64;

    let integrand = |x: f64| -> f64 {
        let pdf_x = t_pdf(x, df);
        let arg = (t2 - rho * x) / sqrt_1_rho2 * ((df + 1.0) / (df + x * x)).sqrt();
        pdf_x * t_cdf(arg, df + 1.0)
    };

    let mut sum = integrand(lower) + integrand(upper);
    for i in 1..n {
        let x = lower + i as f64 * h;
        if i % 2 == 0 {
            sum += 2.0 * integrand(x);
        } else {
            sum += 4.0 * integrand(x);
        }
    }
    (sum * h / 3.0).clamp(0.0, 1.0)
}

/// t-Copula PDF (数值稳定版本)
fn t_copula_pdf(u1: f64, u2: f64, rho: f64, df: f64) -> f64 {
    // 边界检查
    if u1 <= 0.0 || u1 >= 1.0 || u2 <= 0.0 || u2 >= 1.0 {
        return 0.0;
    }
    if rho.abs() >= 1.0 {
        return 0.0;
    }

    let t1 = t_quantile(u1, df);
    let t2 = t_quantile(u2, df);

    // 检查t值是否有效
    if !t1.is_finite() || !t2.is_finite() {
        return 0.0;
    }

    let z = (t1 * t1 + t2 * t2 - 2.0 * rho * t1 * t2) / (1.0 - rho * rho);

    // 检查z值
    if z < 0.0 || !z.is_finite() {
        return 0.0;
    }

    // 使用对数计算避免溢出
    let log_coef = ln_gamma((df + 2.0) / 2.0)
        - ln_gamma(df / 2.0)
        - (df * PI).ln()
        - 0.5 * (1.0 - rho * rho).ln();
    let log_pdf = log_coef - (df + 2.0) / 2.0 * (1.0 + z / df).ln();

    let c1 = t_pdf(t1, df);
    let c2 = t_pdf(t2, df);

    if c1 <= 0.0 || c2 <= 0.0 || !c1.is_finite() || !c2.is_finite() {
        return 0.0;
    }

    let result = log_pdf.exp() / (c1 * c2);

    // 限制结果范围
    if !result.is_finite() || result <= 0.0 {
        0.0
    } else {
        result.min(1000.0)
    }
}

/// 对数Gamma函数 (Lanczos近似)
fn ln_gamma(x: f64) -> f64 {
    if x <= 0.0 {
        return f64::NAN;
    }
    let cof = [
        76.18009172947146,
        -86.50532032941677,
        24.01409824083091,
        -1.231739572450155,
        0.1208650973866179e-2,
        -0.5395239384953e-5,
    ];
    let mut ser = 1.000000000190015;
    for j in 0..6 {
        ser += cof[j] / (x + (j + 1) as f64);
    }
    let tmp = x + 5.5;
    -(tmp) + (x + 0.5) * tmp.ln() + (2.5066282746310005 * ser / x).ln()
}

/// t-Copula 采样
fn t_copula_sample(rho: f64, df: f64, n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut rng = rand::thread_rng();
    let mut u1 = Vec::with_capacity(n);
    let mut u2 = Vec::with_capacity(n);

    for _ in 0..n {
        let z1: f64 = rng.gen();
        let z2: f64 = rng.gen();
        let x1 = normal_inv_cdf(z1);
        let x2 = rho * x1 + (1.0 - rho * rho).sqrt() * normal_inv_cdf(z2);

        // 生成chi-square变量
        let chi2 = sample_chi2(df, &mut rng);
        let scale = (df / chi2).sqrt();

        let t1 = x1 * scale;
        let t2 = x2 * scale;

        u1.push(t_cdf(t1, df));
        u2.push(t_cdf(t2, df));
    }
    (u1, u2)
}

/// 生成chi-square分布样本
fn sample_chi2(df: f64, rng: &mut ThreadRng) -> f64 {
    // 使用Gamma分布
    let shape = df / 2.0;
    let scale = 2.0;
    sample_gamma(shape, scale, rng)
}

/// 生成Gamma分布样本 (Marsaglia方法)
fn sample_gamma(shape: f64, scale: f64, rng: &mut ThreadRng) -> f64 {
    if shape < 1.0 {
        return sample_gamma(1.0 + shape, scale, rng) * rng.gen::<f64>().powf(1.0 / shape);
    }

    let d = shape - 1.0 / 3.0;
    let c = 1.0 / (9.0 * d).sqrt();

    loop {
        let x: f64 = normal_inv_cdf(rng.gen());
        let v = (1.0 + c * x).powi(3);
        if v > 0.0 {
            let u: f64 = rng.gen();
            if u < 1.0 - 0.0331 * x * x * x * x {
                return d * v * scale;
            }
            if u.ln() < 0.5 * x * x + d * (1.0 - v + v.ln()) {
                return d * v * scale;
            }
        }
    }
}

// ==================== Clayton Copula ====================

/// Clayton Copula CDF
/// 参数:
/// - u1, u2: 均匀分布值 [0, 1]
/// - theta: 依赖参数 (theta > 0)
fn clayton_copula_cdf(u1: f64, u2: f64, theta: f64) -> f64 {
    if theta <= 0.0 {
        return u1 * u2; // 独立情况
    }
    (u1.powf(-theta) + u2.powf(-theta) - 1.0)
        .max(0.0)
        .powf(-1.0 / theta)
}

/// Clayton Copula PDF
fn clayton_copula_pdf(u1: f64, u2: f64, theta: f64) -> f64 {
    if theta <= 0.0 {
        return 1.0;
    }
    let t = theta;
    let sum = u1.powf(-t) + u2.powf(-t) - 1.0;
    if sum <= 0.0 {
        return 0.0;
    }
    (1.0 + t) * u1.powf(-1.0 - t) * u2.powf(-1.0 - t) * sum.powf(-2.0 - 1.0 / t)
}

/// Clayton Copula 采样
fn clayton_copula_sample(theta: f64, n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut rng = rand::thread_rng();
    let mut u1 = Vec::with_capacity(n);
    let mut u2 = Vec::with_capacity(n);

    for _ in 0..n {
        let s: f64 = rng.gen();
        let t: f64 = rng.gen();

        let u1_val = s;
        let u2_val =
            (1.0 + s.powf(-theta) * (t.powf(-theta / (1.0 + theta)) - 1.0)).powf(-1.0 / theta);

        u1.push(u1_val);
        u2.push(u2_val);
    }
    (u1, u2)
}

/// Clayton尾部依赖系数
fn clayton_lower_tail_dependence(theta: f64) -> f64 {
    if theta <= 0.0 {
        return 0.0;
    }
    2.0_f64.powf(-1.0 / theta)
}

// ==================== Gumbel Copula ====================

/// Gumbel Copula CDF
/// 参数:
/// - u1, u2: 均匀分布值 [0, 1]
/// - theta: 依赖参数 (theta >= 1)
fn gumbel_copula_cdf(u1: f64, u2: f64, theta: f64) -> f64 {
    if theta < 1.0 {
        return u1 * u2;
    }
    let t1 = (-u1.ln()).powf(theta);
    let t2 = (-u2.ln()).powf(theta);
    (-(t1 + t2).powf(1.0 / theta)).exp()
}

/// Gumbel Copula PDF (数值稳定版本)
fn gumbel_copula_pdf(u1: f64, u2: f64, theta: f64) -> f64 {
    if theta < 1.0 {
        return 0.0;
    }

    // 边界检查
    if u1 <= 0.0 || u1 >= 1.0 || u2 <= 0.0 || u2 >= 1.0 {
        return 0.0;
    }

    // 使用对数计算避免溢出
    let ln_u1_safe = (-u1.ln()).max(1e-300);
    let ln_u2_safe = (-u2.ln()).max(1e-300);

    let ln_u1_pow = ln_u1_safe.powf(theta);
    let ln_u2_pow = ln_u2_safe.powf(theta);
    let s = ln_u1_pow + ln_u2_pow;

    if s <= 0.0 || !s.is_finite() {
        return 0.0;
    }

    let s_pow = s.powf(1.0 / theta);

    // 检查中间值
    if !s_pow.is_finite() || s_pow > 100.0 {
        return 0.0;
    }

    // 使用对数计算
    let log_c = -s_pow;
    let log_term1 = -ln_u1_safe.ln() - ln_u2_safe.ln();
    let log_term2 = (-2.0 + 2.0 / theta) * s.ln();
    let log_term3 = (1.0 - 1.0 / theta) * (ln_u1_pow.ln() + ln_u2_pow.ln());
    let log_term4 = (s_pow + theta - 1.0).max(1e-300).ln();

    let log_result = log_c + log_term1 + log_term2 + log_term3 + log_term4;

    if !log_result.is_finite() {
        return 0.0;
    }

    let result = log_result.exp();

    // 限制结果范围
    if !result.is_finite() || result <= 0.0 {
        0.0
    } else {
        result.min(100.0)
    }
}

/// Gumbel Copula 采样 (Marshall-Olkin方法 + 正稳定分布)
fn gumbel_copula_sample(theta: f64, n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut rng = rand::thread_rng();
    let mut u1 = Vec::with_capacity(n);
    let mut u2 = Vec::with_capacity(n);

    let alpha = 1.0 / theta;
    for _ in 0..n {
        let s = sample_positive_stable(alpha, &mut rng);
        let e1 = -rng.gen::<f64>().ln(); // Exp(1)
        let e2 = -rng.gen::<f64>().ln(); // Exp(1)
        let u1_val = (-(e1 / s).powf(alpha)).exp();
        let u2_val = (-(e2 / s).powf(alpha)).exp();
        u1.push(u1_val.clamp(1e-15, 1.0 - 1e-15));
        u2.push(u2_val.clamp(1e-15, 1.0 - 1e-15));
    }
    (u1, u2)
}

/// 正稳定分布采样 (Chambers-Mallows-Stuck method)
fn sample_positive_stable(alpha: f64, rng: &mut ThreadRng) -> f64 {
    if (alpha - 1.0).abs() < 1e-10 {
        return 1.0; // degenerate case: theta=1 means independence
    }
    let u = rng.gen::<f64>() * PI; // Uniform(0, pi)
    let e = -rng.gen::<f64>().ln(); // Exp(1)
    let s = (alpha * u).sin() / u.sin().powf(1.0 / alpha)
        * (((1.0 - alpha) * u).sin() / e).powf((1.0 - alpha) / alpha);
    s
}

/// Gumbel尾部依赖系数
fn gumbel_upper_tail_dependence(theta: f64) -> f64 {
    if theta < 1.0 {
        return 0.0;
    }
    2.0 - 2.0_f64.powf(1.0 / theta)
}

// ==================== Frank Copula ====================

/// Frank Copula CDF
/// 参数: theta ∈ (-∞, 0) ∪ (0, +∞)，支持正负相关
fn frank_copula_cdf(u1: f64, u2: f64, theta: f64) -> f64 {
    if theta.abs() < 1e-10 {
        return u1 * u2; // θ→0时退化为独立
    }
    let e1 = (-theta * u1).exp() - 1.0;
    let e2 = (-theta * u2).exp() - 1.0;
    let et = (-theta).exp() - 1.0;
    -1.0 / theta * (1.0 + e1 * e2 / et).ln()
}

/// Frank Copula PDF
fn frank_copula_pdf(u1: f64, u2: f64, theta: f64) -> f64 {
    if theta.abs() < 1e-10 {
        return 1.0;
    }
    if u1 <= 0.0 || u1 >= 1.0 || u2 <= 0.0 || u2 >= 1.0 {
        return 0.0;
    }
    let et = (-theta).exp();
    let e1 = (-theta * u1).exp();
    let e2 = (-theta * u2).exp();
    let numer = -theta * (et - 1.0) * e1 * e2;
    // 分母: ((et - 1) + (e1 - 1)(e2 - 1))^2
    let denom_base = (et - 1.0) + (e1 - 1.0) * (e2 - 1.0);
    let denom = denom_base * denom_base;
    if denom.abs() < 1e-300 || !denom.is_finite() {
        return 0.0;
    }
    let result = numer / denom;
    if !result.is_finite() || result <= 0.0 {
        0.0
    } else {
        result
    }
}

/// Frank Copula对数似然
fn frank_log_likelihood(u1: &[f64], u2: &[f64], theta: f64) -> f64 {
    if theta.abs() < 1e-10 {
        return 0.0; // 独立时对数似然为0
    }
    let mut loglik = 0.0;
    let mut valid_count = 0;
    for i in 0..u1.len() {
        let pdf = frank_copula_pdf(u1[i], u2[i], theta);
        if pdf > 0.0 && pdf.is_finite() {
            loglik += pdf.ln().max(-100.0).min(100.0);
            valid_count += 1;
        }
    }
    if valid_count == 0 {
        return f64::NEG_INFINITY;
    }
    loglik
}

/// Debye函数 D_1(x) = (1/x) * ∫₀ˣ t/(e^t - 1) dt
/// 使用Simpson法则数值积分，支持正负x
fn debye1(x: f64) -> f64 {
    if x.abs() < 1e-10 {
        return 1.0;
    }
    // 直接在[0, x]上积分，x可正可负
    let n = 400usize;
    let h = x / n as f64;

    let integrand = |t: f64| -> f64 {
        if t.abs() < 1e-15 {
            return 1.0; // lim_{t→0} t/(e^t-1) = 1
        }
        if t > 500.0 {
            return 0.0;
        }
        if t < -500.0 {
            return -t; // t/(e^t-1) ≈ -t when t→-∞
        }
        t / (t.exp() - 1.0)
    };

    let mut sum = integrand(0.0) + integrand(x);
    for i in 1..n {
        let t = i as f64 * h;
        if i % 2 == 0 {
            sum += 2.0 * integrand(t);
        } else {
            sum += 4.0 * integrand(t);
        }
    }
    // ∫₀ˣ f(t)dt / x
    sum * h / (3.0 * x)
}

/// 从Kendall's tau估计Frank Copula参数
/// τ = 1 - 4/θ * (1 - D₁(θ)) 需要数值求逆
fn estimate_frank_theta_from_tau(tau: f64) -> f64 {
    if tau.abs() < 1e-10 {
        return 0.0;
    }
    // 二分法求解 τ = 1 - 4/θ + 4/θ² * ∫₀^θ t/(e^t-1) dt
    let (mut lo, mut hi) = if tau > 0.0 {
        (0.01, 100.0)
    } else {
        (-100.0, -0.01)
    };

    // frank_tau_from_theta 在正负两侧都是单调递增函数（theta越大tau越大）
    for _ in 0..100 {
        let mid = (lo + hi) / 2.0;
        let tau_mid = frank_tau_from_theta(mid);
        if (tau_mid - tau).abs() < 1e-8 {
            return mid;
        }
        if tau_mid < tau {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    (lo + hi) / 2.0
}

/// 给定theta计算Frank Copula对应的Kendall's tau
/// τ = 1 - 4/θ + 4D₁(θ)/θ，其中负theta时D₁使用变换后的积分
fn frank_tau_from_theta(theta: f64) -> f64 {
    if theta.abs() < 1e-10 {
        return 0.0;
    }
    // D_1(θ)在负theta时需要特殊处理
    // 对于负θ: τ = 1 - 4/θ*(1 - D_1(θ))
    // D_1(θ)在我们的实现中已经用|θ|和变换后的被积函数处理了
    // 但公式中的θ仍需要带符号
    let d1 = debye1(theta);
    1.0 - 4.0 / theta * (1.0 - d1)
}

// ==================== 旋转Copula (处理负相关) ====================

/// 90°旋转Clayton Copula PDF
/// C_90(u,v) 用于负相关场景，theta > 0
fn rotated_clayton_90_pdf(u1: f64, u2: f64, theta: f64) -> f64 {
    clayton_copula_pdf(1.0 - u1, u2, theta)
}

/// 90°旋转Clayton Copula对数似然
fn rotated_clayton_90_log_likelihood(u1: &[f64], u2: &[f64], theta: f64) -> f64 {
    if theta <= 0.0 {
        return f64::NEG_INFINITY;
    }
    let mut loglik = 0.0;
    let mut valid_count = 0;
    for i in 0..u1.len() {
        let pdf = rotated_clayton_90_pdf(u1[i], u2[i], theta);
        if pdf > 0.0 && pdf.is_finite() {
            loglik += pdf.ln().max(-100.0).min(100.0);
            valid_count += 1;
        }
    }
    if valid_count == 0 {
        f64::NEG_INFINITY
    } else {
        loglik
    }
}

/// 90°旋转Gumbel Copula PDF
/// C_90(u,v) 用于负相关场景，theta >= 1
fn rotated_gumbel_90_pdf(u1: f64, u2: f64, theta: f64) -> f64 {
    gumbel_copula_pdf(1.0 - u1, u2, theta)
}

/// 90°旋转Gumbel Copula对数似然
fn rotated_gumbel_90_log_likelihood(u1: &[f64], u2: &[f64], theta: f64) -> f64 {
    if theta < 1.0 {
        return f64::NEG_INFINITY;
    }
    let mut loglik = 0.0;
    let mut valid_count = 0;
    for i in 0..u1.len() {
        let pdf = rotated_gumbel_90_pdf(u1[i], u2[i], theta);
        if pdf > 0.0 && pdf.is_finite() {
            loglik += pdf.ln().max(-100.0).min(100.0);
            valid_count += 1;
        }
    }
    if valid_count == 0 {
        f64::NEG_INFINITY
    } else {
        loglik
    }
}

// ==================== 参数估计 ====================

/// 从Kendall's tau估计相关系数
fn kendall_tau_to_rho(tau: f64) -> f64 {
    (tau * PI / 2.0).sin()
}

/// 计算Kendall's tau (O(n log n) 归并排序算法)
fn kendall_tau(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len().min(y.len());
    if n < 2 {
        return 0.0;
    }

    // 按x排序，相同x时按y排序
    let mut pairs: Vec<(f64, f64)> = x
        .iter()
        .zip(y.iter())
        .take(n)
        .map(|(&a, &b)| (a, b))
        .collect();
    pairs.sort_by(|a, b| {
        a.0.partial_cmp(&b.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
    });

    // 统计x相同的对数（tied in x）
    let mut ties_x = 0i64;
    let mut ties_xy = 0i64;
    let mut i = 0;
    while i < n {
        let mut j = i + 1;
        while j < n && pairs[j].0 == pairs[i].0 {
            j += 1;
        }
        let group = (j - i) as i64;
        ties_x += group * (group - 1) / 2;
        // 在x相同的组内，统计y也相同的
        let mut k = i;
        while k < j {
            let mut l = k + 1;
            while l < j && pairs[l].1 == pairs[k].1 {
                l += 1;
            }
            let sub = (l - k) as i64;
            ties_xy += sub * (sub - 1) / 2;
            k = l;
        }
        i = j;
    }

    // 提取y值，用归并排序计数不和谐对（排序后y_vals变为有序）
    let mut y_vals: Vec<f64> = pairs.iter().map(|p| p.1).collect();
    let swaps = merge_sort_count(&mut y_vals);

    let n0 = n as i64 * (n as i64 - 1) / 2;

    // 统计y的ties（y_vals已被merge_sort_count排好序）
    let mut ties_y = 0i64;
    let mut i = 0;
    while i < n {
        let mut j = i + 1;
        while j < n && y_vals[j] == y_vals[i] {
            j += 1;
        }
        let group = (j - i) as i64;
        ties_y += group * (group - 1) / 2;
        i = j;
    }

    let discordant = swaps as i64;
    let concordant = n0 - discordant - ties_x - ties_y + ties_xy;

    let denom = ((n0 - ties_x) as f64 * (n0 - ties_y) as f64).sqrt();
    if denom == 0.0 {
        return 0.0;
    }
    (concordant - discordant) as f64 / denom
}

/// 归并排序计数逆序对数（预分配buffer版本）
fn merge_sort_count(arr: &mut [f64]) -> usize {
    let mut buf = vec![0.0f64; arr.len()];
    merge_sort_inner(arr, &mut buf)
}

fn merge_sort_inner(arr: &mut [f64], buf: &mut [f64]) -> usize {
    let n = arr.len();
    if n <= 1 {
        return 0;
    }
    let mid = n / 2;
    let mut count = merge_sort_inner(&mut arr[..mid], &mut buf[..mid])
        + merge_sort_inner(&mut arr[mid..], &mut buf[mid..]);

    let (left, right) = arr.split_at(mid);
    let mut i = 0;
    let mut j = 0;
    let mut k = 0;
    while i < left.len() && j < right.len() {
        if left[i] <= right[j] {
            buf[k] = left[i];
            i += 1;
        } else {
            buf[k] = right[j];
            count += left.len() - i;
            j += 1;
        }
        k += 1;
    }
    while i < left.len() {
        buf[k] = left[i];
        i += 1;
        k += 1;
    }
    while j < right.len() {
        buf[k] = right[j];
        j += 1;
        k += 1;
    }
    arr[..k].copy_from_slice(&buf[..k]);
    count
}

/// 估计Gaussian Copula参数
fn estimate_gaussian_copula(x: &[f64], y: &[f64]) -> f64 {
    let tau = kendall_tau(x, y);
    kendall_tau_to_rho(tau).clamp(-0.9999, 0.9999)
}

/// 估计Clayton Copula参数
fn estimate_clayton_copula(x: &[f64], y: &[f64]) -> f64 {
    let tau = kendall_tau(x, y);
    if tau <= 0.0 {
        return 0.1; // 默认小正值
    }
    2.0 * tau / (1.0 - tau)
}

/// 估计Gumbel Copula参数
fn estimate_gumbel_copula(x: &[f64], y: &[f64]) -> f64 {
    let tau = kendall_tau(x, y);
    if tau <= 0.0 {
        return 1.0; // 最小值
    }
    1.0 / (1.0 - tau)
}

// ==================== Python接口 ====================

/// Gaussian Copula CDF
#[pyfunction]
#[pyo3(signature = (u1, u2, rho))]
pub fn gaussian_copula_cdf_py(u1: f64, u2: f64, rho: f64) -> PyResult<f64> {
    Ok(gaussian_copula_cdf(u1, u2, rho))
}

/// Gaussian Copula PDF
#[pyfunction]
#[pyo3(signature = (u1, u2, rho))]
pub fn gaussian_copula_pdf_py(u1: f64, u2: f64, rho: f64) -> PyResult<f64> {
    Ok(gaussian_copula_pdf(u1, u2, rho))
}

/// Gaussian Copula 批量CDF计算
#[pyfunction]
#[pyo3(signature = (u1, u2, rho))]
pub fn gaussian_copula_cdf_batch(
    py: Python,
    u1: PyReadonlyArray1<f64>,
    u2: PyReadonlyArray1<f64>,
    rho: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    let u1 = u1.as_slice()?;
    let u2 = u2.as_slice()?;
    let result: Vec<f64> = u1
        .iter()
        .zip(u2.iter())
        .map(|(&a, &b)| gaussian_copula_cdf(a, b, rho))
        .collect();
    Ok(result.into_pyarray(py).to_owned())
}

/// Gaussian Copula 批量PDF计算
#[pyfunction]
#[pyo3(signature = (u1, u2, rho))]
pub fn gaussian_copula_pdf_batch(
    py: Python,
    u1: PyReadonlyArray1<f64>,
    u2: PyReadonlyArray1<f64>,
    rho: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    let u1 = u1.as_slice()?;
    let u2 = u2.as_slice()?;
    let result: Vec<f64> = u1
        .iter()
        .zip(u2.iter())
        .map(|(&a, &b)| gaussian_copula_pdf(a, b, rho))
        .collect();
    Ok(result.into_pyarray(py).to_owned())
}

/// Gaussian Copula 采样
#[pyfunction]
#[pyo3(signature = (rho, n))]
pub fn gaussian_copula_sample_py(rho: f64, n: usize) -> PyResult<(Vec<f64>, Vec<f64>)> {
    Ok(gaussian_copula_sample(rho, n))
}

/// Gaussian Copula 参数估计
#[pyfunction]
#[pyo3(signature = (x, y))]
pub fn gaussian_copula_estimate(
    x: PyReadonlyArray1<f64>,
    y: PyReadonlyArray1<f64>,
) -> PyResult<f64> {
    Ok(estimate_gaussian_copula(x.as_slice()?, y.as_slice()?))
}

/// t-Copula CDF
#[pyfunction]
#[pyo3(signature = (u1, u2, rho, df))]
pub fn t_copula_cdf_py(u1: f64, u2: f64, rho: f64, df: f64) -> PyResult<f64> {
    Ok(t_copula_cdf(u1, u2, rho, df))
}

/// t-Copula PDF
#[pyfunction]
#[pyo3(signature = (u1, u2, rho, df))]
pub fn t_copula_pdf_py(u1: f64, u2: f64, rho: f64, df: f64) -> PyResult<f64> {
    Ok(t_copula_pdf(u1, u2, rho, df))
}

/// t-Copula 批量CDF计算
#[pyfunction]
#[pyo3(signature = (u1, u2, rho, df))]
pub fn t_copula_cdf_batch(
    py: Python,
    u1: PyReadonlyArray1<f64>,
    u2: PyReadonlyArray1<f64>,
    rho: f64,
    df: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    let u1 = u1.as_slice()?;
    let u2 = u2.as_slice()?;
    let result: Vec<f64> = u1
        .iter()
        .zip(u2.iter())
        .map(|(&a, &b)| t_copula_cdf(a, b, rho, df))
        .collect();
    Ok(result.into_pyarray(py).to_owned())
}

/// t-Copula 采样
#[pyfunction]
#[pyo3(signature = (rho, df, n))]
pub fn t_copula_sample_py(rho: f64, df: f64, n: usize) -> PyResult<(Vec<f64>, Vec<f64>)> {
    Ok(t_copula_sample(rho, df, n))
}

/// t-Copula 参数估计 (返回rho, 使用固定df)
#[pyfunction]
#[pyo3(signature = (x, y))]
pub fn t_copula_estimate(x: PyReadonlyArray1<f64>, y: PyReadonlyArray1<f64>) -> PyResult<f64> {
    Ok(estimate_gaussian_copula(x.as_slice()?, y.as_slice()?))
}

/// Clayton Copula CDF
#[pyfunction]
#[pyo3(signature = (u1, u2, theta))]
pub fn clayton_copula_cdf_py(u1: f64, u2: f64, theta: f64) -> PyResult<f64> {
    Ok(clayton_copula_cdf(u1, u2, theta))
}

/// Clayton Copula PDF
#[pyfunction]
#[pyo3(signature = (u1, u2, theta))]
pub fn clayton_copula_pdf_py(u1: f64, u2: f64, theta: f64) -> PyResult<f64> {
    Ok(clayton_copula_pdf(u1, u2, theta))
}

/// Clayton Copula 批量CDF计算
#[pyfunction]
#[pyo3(signature = (u1, u2, theta))]
pub fn clayton_copula_cdf_batch(
    py: Python,
    u1: PyReadonlyArray1<f64>,
    u2: PyReadonlyArray1<f64>,
    theta: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    let u1 = u1.as_slice()?;
    let u2 = u2.as_slice()?;
    let result: Vec<f64> = u1
        .iter()
        .zip(u2.iter())
        .map(|(&a, &b)| clayton_copula_cdf(a, b, theta))
        .collect();
    Ok(result.into_pyarray(py).to_owned())
}

/// Clayton Copula 采样
#[pyfunction]
#[pyo3(signature = (theta, n))]
pub fn clayton_copula_sample_py(theta: f64, n: usize) -> PyResult<(Vec<f64>, Vec<f64>)> {
    Ok(clayton_copula_sample(theta, n))
}

/// Clayton Copula 参数估计
#[pyfunction]
#[pyo3(signature = (x, y))]
pub fn clayton_copula_estimate(
    x: PyReadonlyArray1<f64>,
    y: PyReadonlyArray1<f64>,
) -> PyResult<f64> {
    Ok(estimate_clayton_copula(x.as_slice()?, y.as_slice()?))
}

/// Clayton Copula 下尾依赖系数
#[pyfunction]
#[pyo3(signature = (theta))]
pub fn clayton_lower_tail_dependence_py(theta: f64) -> PyResult<f64> {
    Ok(clayton_lower_tail_dependence(theta))
}

/// Gumbel Copula CDF
#[pyfunction]
#[pyo3(signature = (u1, u2, theta))]
pub fn gumbel_copula_cdf_py(u1: f64, u2: f64, theta: f64) -> PyResult<f64> {
    Ok(gumbel_copula_cdf(u1, u2, theta))
}

/// Gumbel Copula PDF
#[pyfunction]
#[pyo3(signature = (u1, u2, theta))]
pub fn gumbel_copula_pdf_py(u1: f64, u2: f64, theta: f64) -> PyResult<f64> {
    Ok(gumbel_copula_pdf(u1, u2, theta))
}

/// Gumbel Copula 批量CDF计算
#[pyfunction]
#[pyo3(signature = (u1, u2, theta))]
pub fn gumbel_copula_cdf_batch(
    py: Python,
    u1: PyReadonlyArray1<f64>,
    u2: PyReadonlyArray1<f64>,
    theta: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    let u1 = u1.as_slice()?;
    let u2 = u2.as_slice()?;
    let result: Vec<f64> = u1
        .iter()
        .zip(u2.iter())
        .map(|(&a, &b)| gumbel_copula_cdf(a, b, theta))
        .collect();
    Ok(result.into_pyarray(py).to_owned())
}

/// Gumbel Copula 采样
#[pyfunction]
#[pyo3(signature = (theta, n))]
pub fn gumbel_copula_sample_py(theta: f64, n: usize) -> PyResult<(Vec<f64>, Vec<f64>)> {
    Ok(gumbel_copula_sample(theta, n))
}

/// Gumbel Copula 参数估计
#[pyfunction]
#[pyo3(signature = (x, y))]
pub fn gumbel_copula_estimate(x: PyReadonlyArray1<f64>, y: PyReadonlyArray1<f64>) -> PyResult<f64> {
    Ok(estimate_gumbel_copula(x.as_slice()?, y.as_slice()?))
}

/// Gumbel Copula 上尾依赖系数
#[pyfunction]
#[pyo3(signature = (theta))]
pub fn gumbel_upper_tail_dependence_py(theta: f64) -> PyResult<f64> {
    Ok(gumbel_upper_tail_dependence(theta))
}

/// 计算Kendall's tau
#[pyfunction]
#[pyo3(signature = (x, y))]
pub fn copula_kendall_tau(x: PyReadonlyArray1<f64>, y: PyReadonlyArray1<f64>) -> PyResult<f64> {
    Ok(kendall_tau(x.as_slice()?, y.as_slice()?))
}

/// 将数据转换为均匀分布（经验CDF）
#[pyfunction]
#[pyo3(signature = (data))]
pub fn to_uniform(py: Python, data: PyReadonlyArray1<f64>) -> PyResult<Py<PyArray1<f64>>> {
    let data = data.as_slice()?;
    let n = data.len();
    let mut sorted_indices: Vec<usize> = (0..n).collect();
    sorted_indices.sort_by(|&i, &j| {
        data[i]
            .partial_cmp(&data[j])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut result = vec![0.0; n];
    for (rank, &idx) in sorted_indices.iter().enumerate() {
        result[idx] = (rank + 1) as f64 / (n + 1) as f64;
    }

    Ok(result.into_pyarray(py).to_owned())
}

/// 非参数估计下尾依赖系数（直接从数据计算）
/// 当X处于最差q分位时，Y也处于最差q分位的条件概率
fn empirical_lower_tail_dependence(x: &[f64], y: &[f64], q: f64) -> f64 {
    let n = x.len().min(y.len());
    if n == 0 {
        return f64::NAN;
    }

    // 计算分位数阈值
    let mut x_sorted: Vec<f64> = x.iter().take(n).cloned().collect();
    let mut y_sorted: Vec<f64> = y.iter().take(n).cloned().collect();
    x_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    y_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let x_threshold = x_sorted[(q * n as f64) as usize];
    let y_threshold = y_sorted[(q * n as f64) as usize];

    // 计算条件概率: P(Y < y_q | X < x_q)
    let mut both_below = 0usize;
    let mut x_below = 0usize;

    for i in 0..n {
        if x[i] <= x_threshold {
            x_below += 1;
            if y[i] <= y_threshold {
                both_below += 1;
            }
        }
    }

    if x_below == 0 {
        return 0.0;
    }
    both_below as f64 / x_below as f64
}

/// 非参数估计上尾依赖系数（直接从数据计算）
/// 当X处于最好(1-q)分位时，Y也处于最好(1-q)分位的条件概率
fn empirical_upper_tail_dependence(x: &[f64], y: &[f64], q: f64) -> f64 {
    let n = x.len().min(y.len());
    if n == 0 {
        return f64::NAN;
    }

    // 计算分位数阈值（上尾用1-q分位）
    let mut x_sorted: Vec<f64> = x.iter().take(n).cloned().collect();
    let mut y_sorted: Vec<f64> = y.iter().take(n).cloned().collect();
    x_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    y_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let x_threshold = x_sorted[((1.0 - q) * n as f64) as usize];
    let y_threshold = y_sorted[((1.0 - q) * n as f64) as usize];

    // 计算条件概率: P(Y > y_{1-q} | X > x_{1-q})
    let mut both_above = 0usize;
    let mut x_above = 0usize;

    for i in 0..n {
        if x[i] >= x_threshold {
            x_above += 1;
            if y[i] >= y_threshold {
                both_above += 1;
            }
        }
    }

    if x_above == 0 {
        return 0.0;
    }
    both_above as f64 / x_above as f64
}

/// 交叉尾部依赖：X处于最好(1-q)分位时，Y处于最差q分位的概率（负相关场景）
fn empirical_upper_tail_dependence_cross_lower(x: &[f64], y: &[f64], q: f64) -> f64 {
    let n = x.len().min(y.len());
    if n == 0 {
        return f64::NAN;
    }
    let mut x_sorted: Vec<f64> = x.iter().take(n).cloned().collect();
    let mut y_sorted: Vec<f64> = y.iter().take(n).cloned().collect();
    x_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    y_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let x_threshold = x_sorted[((1.0 - q) * n as f64) as usize];
    let y_threshold = y_sorted[(q * n as f64) as usize];
    let mut both = 0usize;
    let mut x_above = 0usize;
    for i in 0..n {
        if x[i] >= x_threshold {
            x_above += 1;
            if y[i] <= y_threshold {
                both += 1;
            }
        }
    }
    if x_above == 0 {
        0.0
    } else {
        both as f64 / x_above as f64
    }
}

/// 交叉尾部依赖：X处于最差q分位时，Y处于最好(1-q)分位的概率（负相关场景）
fn empirical_lower_tail_dependence_cross_upper(x: &[f64], y: &[f64], q: f64) -> f64 {
    let n = x.len().min(y.len());
    if n == 0 {
        return f64::NAN;
    }
    let mut x_sorted: Vec<f64> = x.iter().take(n).cloned().collect();
    let mut y_sorted: Vec<f64> = y.iter().take(n).cloned().collect();
    x_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    y_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let x_threshold = x_sorted[(q * n as f64) as usize];
    let y_threshold = y_sorted[((1.0 - q) * n as f64) as usize];
    let mut both = 0usize;
    let mut x_below = 0usize;
    for i in 0..n {
        if x[i] <= x_threshold {
            x_below += 1;
            if y[i] >= y_threshold {
                both += 1;
            }
        }
    }
    if x_below == 0 {
        0.0
    } else {
        both as f64 / x_below as f64
    }
}

/// 非参数估计尾部依赖（Python接口）
#[pyfunction]
#[pyo3(signature = (x, y, q=0.1))]
pub fn empirical_tail_dependence(
    py: Python,
    x: PyReadonlyArray1<f64>,
    y: PyReadonlyArray1<f64>,
    q: f64,
) -> PyResult<Py<PyDict>> {
    let dict = PyDict::new(py);
    let x_slice = x.as_slice()?;
    let y_slice = y.as_slice()?;

    let lower = empirical_lower_tail_dependence(x_slice, y_slice, q);
    let upper = empirical_upper_tail_dependence(x_slice, y_slice, q);

    dict.set_item("lower_tail", lower)?;
    dict.set_item("upper_tail", upper)?;
    dict.set_item("quantile", q)?;

    Ok(dict.into())
}

/// 批量估计所有Copula参数（增强版：包含非参数尾部依赖）
#[pyfunction]
#[pyo3(signature = (x, y, q=0.1))]
pub fn estimate_all_copulas(
    py: Python,
    x: PyReadonlyArray1<f64>,
    y: PyReadonlyArray1<f64>,
    q: f64,
) -> PyResult<Py<PyDict>> {
    let dict = PyDict::new(py);
    let x_slice = x.as_slice()?;
    let y_slice = y.as_slice()?;

    let tau = kendall_tau(x_slice, y_slice);

    // 基于tau直接估计参数（避免重复计算tau）
    let gaussian_rho = kendall_tau_to_rho(tau).clamp(-0.9999, 0.9999);
    let clayton_theta = if tau <= 0.0 {
        0.1
    } else {
        2.0 * tau / (1.0 - tau)
    };
    let gumbel_theta = if tau <= 0.0 { 1.0 } else { 1.0 / (1.0 - tau) };

    dict.set_item("kendall_tau", tau)?;
    dict.set_item("gaussian_rho", gaussian_rho)?;
    dict.set_item("t_rho", gaussian_rho)?;
    dict.set_item("clayton_theta", clayton_theta)?;
    dict.set_item("gumbel_theta", gumbel_theta)?;
    dict.set_item(
        "clayton_lower_tail_param",
        clayton_lower_tail_dependence(clayton_theta),
    )?;
    dict.set_item(
        "gumbel_upper_tail_param",
        gumbel_upper_tail_dependence(gumbel_theta),
    )?;

    // 非参数尾部依赖（直接从数据计算，这是真正有意义的！）
    let empirical_lower = empirical_lower_tail_dependence(x_slice, y_slice, q);
    let empirical_upper = empirical_upper_tail_dependence(x_slice, y_slice, q);
    dict.set_item("empirical_lower_tail", empirical_lower)?;
    dict.set_item("empirical_upper_tail", empirical_upper)?;
    dict.set_item("quantile", q)?;

    Ok(dict.into())
}

/// 自动搜索最优t-Copula自由度
fn optimize_t_copula_df(u1: &[f64], u2: &[f64], rho: f64) -> (f64, f64, f64) {
    // 搜索的df候选值
    let df_candidates = [3.0, 5.0, 10.0, 20.0, 50.0];

    let mut best_df = 4.0;
    let mut best_loglik = f64::NEG_INFINITY;
    let mut best_aic = f64::INFINITY;

    for &df in &df_candidates {
        let loglik = t_copula_log_likelihood(u1, u2, rho, df);
        if loglik.is_finite() {
            let aic = 2.0 * 2.0 - 2.0 * loglik; // 2个参数: rho, df
            if aic < best_aic {
                best_aic = aic;
                best_df = df;
                best_loglik = loglik;
            }
        }
    }

    (best_df, best_loglik, best_aic)
}

/// Copula综合分析函数（五种模型拟合 + 拟合优度比较 + 尾部依赖分析）
///
/// 支持5种Copula模型: Gaussian, t-Copula, Frank, Clayton(或Rotated-90), Gumbel(或Rotated-90)
/// 当 kendall_tau <= 0 时，Clayton/Gumbel 自动切换为90°旋转版本以适配负相关
///
/// 返回三个字典：params(设定参数), estimates(估计结果), evaluation(评价建议)
///
/// 【params】参数信息
///   - quantile:         用户指定的分位数阈值，用于经验尾部依赖计算
///   - t_df_auto:        是否自动优化t-Copula自由度 (true/false)
///   - t_df:             t-Copula自由度，取值范围 (0, +∞)，越小尾部越厚
///   - n_samples:        有效样本量
///
/// 【estimates】从数据估计的结果
///
///   Gaussian Copula:
///   - gaussian_rho:             由 tau 通过 sin(tau * π/2) 转换得到，取值 [-1, 1]
///                               绝对值越大相关越强，0 = 独立
///   - gaussian_log_likelihood:  对数似然，取值 (-∞, +∞)，越大拟合越好
///
///   t-Copula:
///   - t_rho:                    与 gaussian_rho 相同（同源于 tau 转换），取值 [-1, 1]
///   - t_log_likelihood:         对数似然，取值 (-∞, +∞)，越大拟合越好
///
///   Frank Copula:
///   - frank_theta:              Frank参数，取值 (-∞, +∞)
///                               > 0 正相关，< 0 负相关，绝对值越大相关越强，→0 趋近独立
///                               Frank 无尾部依赖，适合描述"中间段有相关但极端不联动"
///   - frank_log_likelihood:     对数似然，取值 (-∞, +∞)，越大拟合越好
///
///   Clayton Copula (正相关时) / Rotated-Clayton-90 (负相关时):
///   - clayton_type:             "Clayton"(tau>0) 或 "Rotated-Clayton-90"(tau<=0)
///   - clayton_theta:            Clayton参数，取值 (0, +∞)
///                               越大依赖越强，→0 趋近独立
///                               注意：负相关时参数由 |tau| 估计，仍为正值
///   - clayton_log_likelihood:   对数似然，取值 (-∞, +∞)，越大拟合越好
///   - clayton_tail_dependence:  理论尾部依赖系数 = 2^(-1/θ)，取值 [0, 1]
///                               原版Clayton: 衡量同向下尾（X和Y同时极小）的联动概率
///                               Rotated-90: 衡量交叉尾部（X极大时Y极小）的联动概率
///                               越接近1联动越强，越接近0越独立
///
///   Gumbel Copula (正相关时) / Rotated-Gumbel-90 (负相关时):
///   - gumbel_type:              "Gumbel"(tau>0) 或 "Rotated-Gumbel-90"(tau<=0)
///   - gumbel_theta:             Gumbel参数，取值 [1, +∞)
///                               越大依赖越强，=1 时为独立
///                               注意：负相关时参数由 |tau| 估计，仍 >= 1
///   - gumbel_log_likelihood:    对数似然，取值 (-∞, +∞)，越大拟合越好
///   - gumbel_tail_dependence:   理论尾部依赖系数 = 2 - 2^(1/θ)，取值 [0, 1]
///                               原版Gumbel: 衡量同向上尾（X和Y同时极大）的联动概率
///                               Rotated-90: 衡量交叉尾部（X极小时Y极大）的联动概率
///                               越接近1联动越强，越接近0越独立
///
///   非参数经验尾部依赖:
///   - empirical_lower_tail:       P(Y<q分位 | X<q分位)，取值 [0, 1]
///                                 衡量两者同时处于极低分位的概率，越大说明同跌联动越强
///   - empirical_upper_tail:       P(Y>1-q分位 | X>1-q分位)，取值 [0, 1]
///                                 衡量两者同时处于极高分位的概率，越大说明同涨联动越强
///   - empirical_cross_lower_tail: P(Y<q分位 | X>1-q分位)，取值 [0, 1]
///                                 X极大时Y极小的概率，负相关核心指标，越大反向联动越强
///   - empirical_cross_upper_tail: P(X<q分位 | Y>1-q分位)，取值 [0, 1]
///                                 X极小时Y极大的概率，负相关核心指标，越大反向联动越强
///
/// 【evaluation】模型评价和建议
///   - gaussian_aic / t_aic / frank_aic / clayton_aic / gumbel_aic:
///                               AIC信息准则 = 2k - 2*loglik，取值 (-∞, +∞)
///                               越小（越负）拟合越好，不同模型间可直接比较
///   - best_copula:              AIC最小的模型名称
///   - best_aic:                 最佳AIC值
///   - best_avg_loglik:          最佳模型的每样本平均对数似然，取值 [0, +∞)
///                               = best_loglik / n_samples
///                               独立时 = 0，越大说明依赖越强
///                               跨例子可直接比较：无论最佳模型是哪个族、样本量多少
///                               例如 A=0.015 vs B=0.032 → B的依赖关系更强
///   - tail_asymmetry:           尾部不对称度，取值 (-1, 1)
///                               正相关时 = empirical_lower - empirical_upper，>0 共跌型，<0 共涨型
///                               负相关时 = cross_lower - cross_upper，>0 卖强买弱联动更显著
///   - tail_type:                尾部类型的文字描述
///                               正相关: "共跌型" / "共涨型" / "对称尾部依赖" / "近独立"
///                               负相关: "反向下尾型" / "反向上尾型" / "对称反向联动" / "弱联动"
#[pyfunction]
#[pyo3(signature = (x, y, q=0.1, df=None))]
pub fn copula_analysis(
    py: Python,
    x: PyReadonlyArray1<f64>,
    y: PyReadonlyArray1<f64>,
    q: f64,
    df: Option<f64>,
) -> PyResult<Py<PyDict>> {
    let result = PyDict::new(py);
    let x_slice = x.as_slice()?;
    let y_slice = y.as_slice()?;
    let n = x_slice.len().min(y_slice.len());

    // ==================== 基础统计 ====================
    let tau = kendall_tau(x_slice, y_slice);
    let pearson_corr = pearson_correlation(x_slice, y_slice);

    // ==================== 转换为均匀分布 ====================
    let u1 = to_uniform_slice(x_slice);
    let u2 = to_uniform_slice(y_slice);

    // ==================== 基于tau直接估计参数（避免重复计算tau）====================
    let gaussian_rho = kendall_tau_to_rho(tau).clamp(-0.9999, 0.9999);
    let gaussian_loglik = gaussian_log_likelihood(&u1, &u2, gaussian_rho);
    let gaussian_aic = 2.0 * 1.0 - 2.0 * gaussian_loglik; // 1个参数

    let t_rho = gaussian_rho;
    let (t_df, t_loglik, t_aic) = match df {
        Some(user_df) => {
            let loglik = t_copula_log_likelihood(&u1, &u2, t_rho, user_df);
            let aic = 2.0 * 2.0 - 2.0 * loglik;
            (user_df, loglik, aic)
        }
        None => optimize_t_copula_df(&u1, &u2, t_rho),
    };

    // ==================== Frank Copula (支持正负相关) ====================
    let frank_theta = estimate_frank_theta_from_tau(tau);
    let frank_loglik = frank_log_likelihood(&u1, &u2, frank_theta);
    let frank_aic = 2.0 * 1.0 - 2.0 * frank_loglik;

    // ==================== Clayton / Gumbel (根据tau正负选择原版或旋转版) ====================
    let is_negative = tau <= 0.0;
    let tau_abs = tau.abs().max(0.01); // 用|tau|来估计旋转copula的参数

    let (clayton_theta_clamped, clayton_loglik, clayton_label) = if !is_negative {
        let theta = (2.0 * tau / (1.0 - tau)).max(0.01);
        let loglik = clayton_log_likelihood(&u1, &u2, theta);
        (theta, loglik, "Clayton")
    } else {
        let theta = (2.0 * tau_abs / (1.0 - tau_abs)).max(0.01);
        let loglik = rotated_clayton_90_log_likelihood(&u1, &u2, theta);
        (theta, loglik, "Rotated-Clayton-90")
    };
    let clayton_aic = 2.0 * 1.0 - 2.0 * clayton_loglik;
    let clayton_tail = clayton_lower_tail_dependence(clayton_theta_clamped);

    let (gumbel_theta_clamped, gumbel_loglik, gumbel_label) = if !is_negative {
        let theta = (1.0 / (1.0 - tau)).max(1.0);
        let loglik = gumbel_log_likelihood(&u1, &u2, theta);
        (theta, loglik, "Gumbel")
    } else {
        let theta = (1.0 / (1.0 - tau_abs)).max(1.0);
        let loglik = rotated_gumbel_90_log_likelihood(&u1, &u2, theta);
        (theta, loglik, "Rotated-Gumbel-90")
    };
    let gumbel_aic = 2.0 * 1.0 - 2.0 * gumbel_loglik;
    let gumbel_tail = gumbel_upper_tail_dependence(gumbel_theta_clamped);

    // ==================== 非参数尾部依赖 ====================
    let empirical_lower = empirical_lower_tail_dependence(x_slice, y_slice, q);
    let empirical_upper = empirical_upper_tail_dependence(x_slice, y_slice, q);
    // 交叉尾部依赖（负相关特有）：X极大时Y极小的概率
    let empirical_cross_lower = empirical_upper_tail_dependence_cross_lower(x_slice, y_slice, q);
    let empirical_cross_upper = empirical_lower_tail_dependence_cross_upper(x_slice, y_slice, q);

    // ==================== 模型选择建议 ====================
    let best_aic = gaussian_aic
        .min(t_aic)
        .min(clayton_aic)
        .min(gumbel_aic)
        .min(frank_aic);
    let best_copula = if (gaussian_aic - best_aic).abs() < 1e-10 {
        "Gaussian"
    } else if (t_aic - best_aic).abs() < 1e-10 {
        "t-Copula"
    } else if (frank_aic - best_aic).abs() < 1e-10 {
        "Frank"
    } else if (clayton_aic - best_aic).abs() < 1e-10 {
        clayton_label
    } else {
        gumbel_label
    };

    // ==================== 尾部依赖特征总结 ====================
    let (tail_asymmetry, tail_type) = if is_negative {
        let cross_asym = empirical_cross_lower - empirical_cross_upper;
        let tt = if cross_asym > 0.1 {
            "负相关-反向下尾型（X涨时Y跌联动更强）"
        } else if cross_asym < -0.1 {
            "负相关-反向上尾型（X跌时Y涨联动更强）"
        } else if empirical_cross_lower > 0.3 || empirical_cross_upper > 0.3 {
            "负相关-对称反向联动"
        } else {
            "负相关-弱联动（极端行情反向性不显著）"
        };
        (cross_asym, tt)
    } else {
        let asym = empirical_lower - empirical_upper;
        let tt = if asym > 0.1 {
            "共跌型（熊市联动更强）"
        } else if asym < -0.1 {
            "共涨型（牛市联动更强）"
        } else if empirical_lower > 0.3 || empirical_upper > 0.3 {
            "对称尾部依赖"
        } else {
            "近独立（极端行情不联动）"
        };
        (asym, tt)
    };

    // ==================== 构建三个输出字典 ====================

    // 1. params: 用户设定的参数
    let params = PyDict::new(py);
    params.set_item("quantile", q)?;
    params.set_item("t_df_auto", df.is_none())?;
    params.set_item("t_df", t_df)?;
    params.set_item("n_samples", n)?;

    // 2. estimates: 从数据估计的结果
    let estimates = PyDict::new(py);
    // Gaussian Copula
    estimates.set_item("gaussian_rho", gaussian_rho)?;
    estimates.set_item("gaussian_log_likelihood", gaussian_loglik)?;
    // t-Copula
    estimates.set_item("t_rho", t_rho)?;
    estimates.set_item("t_log_likelihood", t_loglik)?;
    // Frank Copula
    estimates.set_item("frank_theta", frank_theta)?;
    estimates.set_item("frank_log_likelihood", frank_loglik)?;
    // Clayton / Rotated-Clayton
    estimates.set_item("clayton_type", clayton_label)?;
    estimates.set_item("clayton_theta", clayton_theta_clamped)?;
    estimates.set_item("clayton_log_likelihood", clayton_loglik)?;
    estimates.set_item("clayton_tail_dependence", clayton_tail)?;
    // Gumbel / Rotated-Gumbel
    estimates.set_item("gumbel_type", gumbel_label)?;
    estimates.set_item("gumbel_theta", gumbel_theta_clamped)?;
    estimates.set_item("gumbel_log_likelihood", gumbel_loglik)?;
    estimates.set_item("gumbel_tail_dependence", gumbel_tail)?;
    // 非参数尾部依赖
    estimates.set_item("empirical_lower_tail", empirical_lower)?;
    estimates.set_item("empirical_upper_tail", empirical_upper)?;
    estimates.set_item("empirical_cross_lower_tail", empirical_cross_lower)?;
    estimates.set_item("empirical_cross_upper_tail", empirical_cross_upper)?;

    // 3. evaluation: 模型评价和建议
    let evaluation = PyDict::new(py);
    evaluation.set_item("gaussian_aic", gaussian_aic)?;
    evaluation.set_item("t_aic", t_aic)?;
    evaluation.set_item("frank_aic", frank_aic)?;
    evaluation.set_item("clayton_aic", clayton_aic)?;
    evaluation.set_item("gumbel_aic", gumbel_aic)?;
    // 最佳模型
    evaluation.set_item("best_copula", best_copula)?;
    evaluation.set_item("best_aic", best_aic)?;
    // 最佳模型的每样本平均对数似然（跨例子可比的依赖强度指标）
    let best_loglik = if (gaussian_aic - best_aic).abs() < 1e-10 {
        gaussian_loglik
    } else if (t_aic - best_aic).abs() < 1e-10 {
        t_loglik
    } else if (frank_aic - best_aic).abs() < 1e-10 {
        frank_loglik
    } else if (clayton_aic - best_aic).abs() < 1e-10 {
        clayton_loglik
    } else {
        gumbel_loglik
    };
    evaluation.set_item("best_avg_loglik", best_loglik / n as f64)?;
    // 尾部特征总结
    evaluation.set_item("tail_asymmetry", tail_asymmetry)?;
    evaluation.set_item("tail_type", tail_type)?;

    // 组装最终结果
    result.set_item("params", params)?;
    result.set_item("estimates", estimates)?;
    result.set_item("evaluation", evaluation)?;

    Ok(result.into())
}

/// 计算Pearson相关系数
fn pearson_correlation(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len().min(y.len()) as f64;
    if n < 2.0 {
        return f64::NAN;
    }

    let mean_x: f64 = x.iter().take(n as usize).sum::<f64>() / n;
    let mean_y: f64 = y.iter().take(n as usize).sum::<f64>() / n;

    let mut cov = 0.0;
    let mut var_x = 0.0;
    let mut var_y = 0.0;

    for i in 0..n as usize {
        let dx = x[i] - mean_x;
        let dy = y[i] - mean_y;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }

    if var_x == 0.0 || var_y == 0.0 {
        return 0.0;
    }
    cov / (var_x * var_y).sqrt()
}

/// 将数据转换为均匀分布
fn to_uniform_slice(data: &[f64]) -> Vec<f64> {
    let n = data.len();
    let mut sorted_indices: Vec<usize> = (0..n).collect();
    sorted_indices.sort_by(|&i, &j| {
        data[i]
            .partial_cmp(&data[j])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut result = vec![0.0; n];
    for (rank, &idx) in sorted_indices.iter().enumerate() {
        result[idx] = (rank + 1) as f64 / (n + 1) as f64;
    }
    result
}

/// Gaussian Copula对数似然 (数值稳定版本)
fn gaussian_log_likelihood(u1: &[f64], u2: &[f64], rho: f64) -> f64 {
    if rho.abs() >= 1.0 {
        return f64::NEG_INFINITY;
    }

    let mut loglik = 0.0;
    let mut valid_count = 0;

    for i in 0..u1.len() {
        let pdf = gaussian_copula_pdf(u1[i], u2[i], rho);
        if pdf > 0.0 && pdf.is_finite() {
            loglik += pdf.ln().max(-100.0).min(100.0); // 限制对数值范围
            valid_count += 1;
        }
    }

    if valid_count == 0 {
        return f64::NEG_INFINITY;
    }
    loglik
}

/// t-Copula对数似然 (数值稳定版本)
fn t_copula_log_likelihood(u1: &[f64], u2: &[f64], rho: f64, df: f64) -> f64 {
    if rho.abs() >= 1.0 || df <= 0.0 {
        return f64::NEG_INFINITY;
    }

    let mut loglik = 0.0;
    let mut valid_count = 0;

    for i in 0..u1.len() {
        let pdf = t_copula_pdf(u1[i], u2[i], rho, df);
        if pdf > 0.0 && pdf.is_finite() {
            loglik += pdf.ln().max(-100.0).min(100.0);
            valid_count += 1;
        }
    }

    if valid_count == 0 {
        return f64::NEG_INFINITY;
    }
    loglik
}

/// Clayton Copula对数似然 (数值稳定版本)
fn clayton_log_likelihood(u1: &[f64], u2: &[f64], theta: f64) -> f64 {
    if theta <= 0.0 {
        return f64::NEG_INFINITY;
    }

    let mut loglik = 0.0;
    let mut valid_count = 0;

    for i in 0..u1.len() {
        let pdf = clayton_copula_pdf(u1[i], u2[i], theta);
        if pdf > 0.0 && pdf.is_finite() {
            loglik += pdf.ln().max(-100.0).min(100.0);
            valid_count += 1;
        }
    }

    if valid_count == 0 {
        return f64::NEG_INFINITY;
    }
    loglik
}

/// Gumbel Copula对数似然 (数值稳定版本)
fn gumbel_log_likelihood(u1: &[f64], u2: &[f64], theta: f64) -> f64 {
    if theta < 1.0 {
        return f64::NEG_INFINITY;
    }

    let mut loglik = 0.0;
    let mut valid_count = 0;

    for i in 0..u1.len() {
        let pdf = gumbel_copula_pdf(u1[i], u2[i], theta);
        if pdf > 0.0 && pdf.is_finite() {
            loglik += pdf.ln().max(-100.0).min(100.0);
            valid_count += 1;
        }
    }

    if valid_count == 0 {
        return f64::NEG_INFINITY;
    }
    loglik
}
