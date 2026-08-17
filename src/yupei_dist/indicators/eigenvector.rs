//! eigenvector: 特征向量中心性（降维指标.md 九）。
//! 对称矩阵幂迭代求主特征向量: splitmix64 固定种子初始化, 每次迭代归一化,
//! ≤40 次或 cos 收敛>0.9999; 符号统一（最大绝对值元素为正）。行并行 matvec。cnt_t1 与 vol_t1。

use crate::yupei_dist::indicator_ctx::{IndicatorCtx, IndicatorResult};
use rayon::prelude::*;

pub fn name() -> &'static str {
    "eigenvector"
}

pub fn desc() -> &'static str {
    "特征向量中心性: 确定性幂迭代 (cnt_t1, vol_t1)"
}

pub fn compute(ctx: &IndicatorCtx) -> Vec<IndicatorResult> {
    let n = ctx.n();
    let mut out = Vec::new();
    for mat in crate::yupei_dist::indicator_ctx::MATRIX_LIST {
        let sym = match ctx.symmetric(mat) {
            Some(s) => s,
            None => continue,
        };
        let col = power_iteration(&sym, n);
        out.push(IndicatorResult::new(format!("eig_{mat}_centrality"), col));
    }
    out
}

fn power_iteration(m: &[f32], n: usize) -> Vec<f32> {
    // 确定性种子初始化
    let mut seed: u64 = 0x9E3779B97F4A7C15;
    let mut v: Vec<f64> = (0..n)
        .map(|_| {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((seed >> 33) as f64 / u32::MAX as f64) * 2.0 - 1.0
        })
        .collect();
    let mut w = vec![0.0f64; n];
    for _ in 0..40 {
        // matvec（行并行）
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
        // 归一化 + 收敛判定
        let norm: f64 = w.iter().map(|&x| x * x).sum::<f64>().sqrt();
        if norm < 1e-12 {
            break;
        }
        let inv = 1.0 / norm;
        let mut c = 0.0f64;
        for i in 0..n {
            w[i] *= inv;
            c += w[i] * v[i];
        }
        std::mem::swap(&mut v, &mut w);
        if c > 0.9999 {
            break;
        }
    }
    // 符号统一: 最大绝对值元素为正
    let mut mi = 0usize;
    for (i, &x) in v.iter().enumerate() {
        if x.abs() > v[mi].abs() {
            mi = i;
        }
    }
    if v[mi] < 0.0 {
        v.iter_mut().for_each(|x| *x = -*x);
    }
    v.iter().map(|&x| x as f32).collect()
}
