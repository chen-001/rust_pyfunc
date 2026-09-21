/// Abramowitz & Stegun 7.1.26 的误差函数近似（五系数），与 `copula.rs::erf` 同一份系数。
/// 最大绝对误差 1.5e-7。
fn erf_as_7_1_26(x: f64) -> f64 {
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

/// 标准正态分布函数 `Φ(x) = 0.5 * (1 + erf(x / sqrt(2)))`，erf 走 A&S 7.1.26。
fn std_normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf_as_7_1_26(x / std::f64::consts::SQRT_2))
}

/// 概率版双边单调度 MPROB。
///
/// 输入 `group_returns[d][t]` 是第 d 组的逐日收益。与 [`compute_ssm`] 用同一套端点规则定向
/// （第 10 组时间均值低于第 1 组时整条倒序），然后对 45 个组对各自算一条价差序列的
/// Newey-West（1994 自动选阶）t 统计量，转成正态概率后等权平均：
///
/// ```text
/// d_t     = group_returns[idx[j]][t] - group_returns[idx[i]][t]      （i < j，45 对）
/// mean    = Σ d_t / n,   e_t = d_t - mean
/// gamma_0 = Σ e_t² / n,  gamma_k = Σ_{t=k..n-1} e_t·e_{t-k} / n
/// nw_var  = (gamma_0 + 2·Σ_{k=1..L} (1 - k/(L+1))·gamma_k) / n
/// 该对贡献 = 2·Φ(mean / sqrt(nw_var)) - 1
/// MPROB   = Σ(45 对贡献) / 45
/// ```
///
/// `L = min(n-1, floor(4·(n/100)^(2/9)))` 是 Newey-West 的自动滞后阶（n=2424 时 L=8）。
///
/// 与 SSM 的分工：SSM 只看组均收益曲线的形状（样本均值的符号比较），MPROB 看每条价差
/// 序列的信噪比，所以「10bp ± 1bp」会比「0.01bp ± 5bp」拿到更高的分。值域 [-1, 1]；
/// 定向规则与 SSM 相同，因此整条曲线取负时严格反号（`Φ(-t) = 1-Φ(t)`），是方向无关的形状分。
///
/// 精度：erf 用 A&S 7.1.26 近似（最大绝对误差 1.5e-7）→ Φ 的绝对误差 ≤ 7.5e-8
/// → 每个组对 `2·Φ(t)-1` 的绝对误差 ≤ 1.5e-7（系数 2 把它放大一倍），45 对等权平均后
/// MPROB 的绝对误差 ≤ 1.5e-7（各对误差同向的最坏情形）。
///
/// 无法计算时返回 `NaN`：组数不是 10、任一列长度不等、日期数 n < 2、出现 NaN/±Inf。
///
/// 单对 `nw_var` 非有限或 ≤ 0 时该对贡献 0.0，这是**刻意的保守兜底**：Newey-West 方差
/// 为 0 说明该价差序列在这段窗口里没有可用波动，给不出置信度；为负则是强均值回复下
/// Bartlett 加权和把 gamma_0 抵消掉的结果，同样是「没有可用信息」。所以严格恒定的阶梯
/// （任意两组价差逐日不变，e_t ≡ 0 → gamma_0 = 0 → nw_var = 0）得 0.0 而不是 1.0，
/// 值域上界 1.0 只在「价差均值远大于其 Newey-West 标准误」的极限下取到。
/// 真实数据的组间价差逐日有波动，这条只会在「整段恒定」或「强均值回复」两种病态情形触发。
pub(crate) fn compute_mprob(group_returns: &[Vec<f64>], portf_num: usize) -> f64 {
    if portf_num != 10 || group_returns.len() != portf_num {
        return f64::NAN;
    }
    let n = group_returns[0].len();
    if n < 2 || group_returns.iter().any(|v| v.len() != n) {
        return f64::NAN;
    }
    if group_returns
        .iter()
        .any(|col| col.iter().any(|v| !v.is_finite()))
    {
        return f64::NAN;
    }
    // 时间均值 + 定向：与 compute_ssm 同一套端点规则。
    let mut r = [0.0_f64; 10];
    for (d, col) in group_returns.iter().enumerate() {
        r[d] = col.iter().sum::<f64>() / n as f64;
    }
    let mut idx = [0usize; 10];
    for (d, slot) in idx.iter_mut().enumerate() {
        *slot = if r[9] < r[0] { 9 - d } else { d };
    }
    // Newey-West(1994) 自动选阶。
    let l = (((4.0 * (n as f64 / 100.0).powf(2.0 / 9.0)).floor()) as usize).min(n - 1);
    let l_f = l as f64;
    // e_t 缓冲复用（n 最大到回测窗口长度，45 对循环里不再分配）。
    let mut e = vec![0.0_f64; n];
    let mut acc = 0.0_f64;
    for i in 0..10 {
        for j in (i + 1)..10 {
            let a = &group_returns[idx[i]];
            let b = &group_returns[idx[j]];
            let mut mean = 0.0_f64;
            for t in 0..n {
                mean += b[t] - a[t];
            }
            mean /= n as f64;
            for t in 0..n {
                e[t] = b[t] - a[t] - mean;
            }
            let mut gamma0 = 0.0_f64;
            for v in e.iter() {
                gamma0 += v * v;
            }
            gamma0 /= n as f64;
            let mut acov = 0.0_f64;
            for k in 1..=l {
                let mut gamma_k = 0.0_f64;
                for t in k..n {
                    gamma_k += e[t] * e[t - k];
                }
                gamma_k /= n as f64;
                acov += (1.0 - k as f64 / (l_f + 1.0)) * gamma_k;
            }
            let nw_var = (gamma0 + 2.0 * acov) / n as f64;
            if !nw_var.is_finite() || nw_var <= 0.0 {
                continue;
            }
            acc += 2.0 * std_normal_cdf(mean / nw_var.sqrt()) - 1.0;
        }
    }
    acc / 45.0
}
