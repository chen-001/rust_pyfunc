//! spectral: 谱隙与第二集体模式（统计物理扩展，20251031 原型验证版）。
//! 对对称矩阵 S_sym = S_dir + S_dirᵀ（对角 0）做幂迭代:
//!   λ1,v1 = 主模式（同 eigenvector.rs，≤40 次或 cos>0.9999）;
//!   λ2,v2 = 第二模式（deflated 幂迭代，每步 w -= v1·(v1·w) 后归一，≤60 次）。
//! 每股输出:
//!   eig2_{mat}_coord = v2_i（符号约定: 绝对值最大元素取正，同 eigenvector.rs）;
//!   eig2_{mat}_part  = N·v2_i²（该模式内的参与度，等于 |v2|·√IPR2 的单调等价）;
//!   spec_{mat}_gap   = (λ1−λ2)/λ1 全市场标量，逐股写同一值。
//! 物理含义: gap 收缩 → 市场模式之外有竞争的第二集体模式（分裂/板块化）;
//! v2 坐标与参与度给出该次模式的载股，与行强度类指标弱相关（原型 worst|rho|≈0.4）。
//! 确定性: splitmix64 固定种子 LCG 初始化（与 eigenvector.rs 同款）。

use crate::yupei_dist::indicator_ctx::{IndicatorCtx, IndicatorResult};
use rayon::prelude::*;

pub fn name() -> &'static str {
    "spectral"
}

pub fn desc() -> &'static str {
    "谱隙+第二模式: deflated幂迭代 v2/N·v2²/gap (28矩阵)"
}

/// 确定性 LCG 初始化（与 eigenvector.rs 相同种子与步进）
fn seed_vec(n: usize) -> Vec<f64> {
    let mut seed: u64 = 0x9E3779B97F4A7C15;
    (0..n)
        .map(|_| {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((seed >> 33) as f64 / u32::MAX as f64) * 2.0 - 1.0
        })
        .collect()
}

/// 行并行 matvec（f32 矩阵 × f64 向量，f64 累加）
fn matvec(m: &[f32], v: &[f64], n: usize) -> Vec<f64> {
    let mut w = vec![0.0f64; n];
    w.par_chunks_mut(64).enumerate().for_each(|(ci, chunk)| {
        let i0 = ci * 64;
        for (ii, o) in chunk.iter_mut().enumerate() {
            let i = i0 + ii;
            let row = &m[i * n..(i + 1) * n];
            let mut acc = 0.0f64;
            for t in 0..n {
                acc += row[t] as f64 * v[t];
            }
            *o = acc;
        }
    });
    w
}

/// deflate = Some(v1) 时每步正交化去主模式方向。
/// 返回 (Rayleigh 商 λ, 单位向量); 收敛: |Δλ| ≤ 1e-8·max(1,λ) 或达 iters 上限。
fn power_mode(m: &[f32], n: usize, iters: usize, deflate: Option<&[f64]>) -> (f64, Vec<f64>) {
    let mut v = seed_vec(n);
    let mut lam = 0.0f64;
    for _ in 0..iters {
        let mut w = matvec(m, &v, n);
        if let Some(u) = deflate {
            let dot: f64 = w.iter().zip(u.iter()).map(|(a, b)| a * b).sum();
            w.iter_mut().zip(u.iter()).for_each(|(a, b)| *a -= dot * b);
        }
        let l: f64 = w.iter().map(|&x| x * x).sum::<f64>().sqrt();
        if l < 1e-12 {
            break;
        }
        w.iter_mut().for_each(|x| *x /= l);
        v = w;
        if (l - lam).abs() <= 1e-8 * l.max(1.0) {
            break;
        }
        lam = l;
    }
    // Rayleigh 商得带符号 λ（deflate 后范数法只给 |λ|）
    let av = matvec(m, &v, n);
    let lam_rq: f64 = av.iter().zip(v.iter()).map(|(a, b)| a * b).sum();
    (lam_rq, v)
}

/// 符号统一: 最大绝对值元素为正
fn fix_sign(v: &mut [f64]) {
    let mut mi = 0usize;
    for (i, &x) in v.iter().enumerate() {
        if x.abs() > v[mi].abs() {
            mi = i;
        }
    }
    if v[mi] < 0.0 {
        v.iter_mut().for_each(|x| *x = -*x);
    }
}

pub fn compute(ctx: &IndicatorCtx) -> Vec<IndicatorResult> {
    let n = ctx.n();
    let mut out = Vec::new();
    if n == 0 {
        return out;
    }
    for mat in crate::yupei_dist::indicator_ctx::MATRIX_LIST {
        let sym = match ctx.symmetric(mat) {
            Some(s) => s,
            None => continue,
        };
        let (lam1, mut v1) = power_mode(&sym, n, 40, None);
        fix_sign(&mut v1);
        let (lam2, mut v2) = power_mode(&sym, n, 60, Some(&v1));
        fix_sign(&mut v2);
        let gap = if lam1.abs() > 1e-12 {
            ((lam1 - lam2) / lam1) as f32
        } else {
            f32::NAN
        };
        out.push(IndicatorResult::new(
            format!("eig2_{mat}_coord"),
            v2.iter().map(|&x| x as f32).collect(),
        ));
        out.push(IndicatorResult::new(
            format!("eig2_{mat}_part"),
            v2.iter().map(|&x| (x * x * n as f64) as f32).collect(),
        ));
        out.push(IndicatorResult::new(
            format!("spec_{mat}_gap"),
            vec![gap; n],
        ));
    }
    out
}
