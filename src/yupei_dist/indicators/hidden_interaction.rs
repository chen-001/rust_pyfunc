//! hidden_interaction: 收益率相关性无法解释的 interaction（降维指标.md 二十六）。
//! 全局 OLS: y_ij = log1p(S_ij)（对称 cnt_t1; log 变换抑制尺度）对
//! X = [1, 同行业(0/1), |log amount_i - log amount_j|, |total_vol_i - total_vol_j|, |ret_i - ret_j|]。
//! 残差 ε_ij = y - Xβ; hidden_degree_pos_i = Σ_j max(ε_ij, 0), neg = Σ_j max(-ε_ij, 0)。
//! 实现: 行并行累加 X'X / X'y（f64）→ 全局归约 → 高斯消元解正规方程 → 行并行残差和。

use crate::yupei_dist::indicator_ctx::{IndicatorCtx, IndicatorResult};
use rayon::prelude::*;

pub fn name() -> &'static str {
    "hidden_interaction"
}

pub fn desc() -> &'static str {
    "残差化交互: OLS 后 hidden degree (cnt_t1)"
}

const P: usize = 5; // 特征数（含截距）

pub fn compute(ctx: &IndicatorCtx) -> Vec<IndicatorResult> {
    let n = ctx.n();
    let mut out = Vec::new();
    let stats = ctx.stats();
    let ind = ctx.industry();
    // 每股特征
    let f_size: Vec<f64> = stats.iter().map(|s| s.amount.max(1.0).ln()).collect();
    let f_liq: Vec<f64> = stats.iter().map(|s| s.total_vol).collect();
    let f_ret: Vec<f64> = stats.iter().map(|s| s.ret).collect();

    for mat in crate::yupei_dist::indicator_ctx::MATRIX_LIST {
        let sym = match ctx.symmetric(mat) {
            Some(s) => s,
            None => continue,
        };
    // ---- 第一遍: X'X (P×P) 与 X'y (P), 行并行局部累加后归约
    let local: Vec<([f64; P * P], [f64; P])> = (0..n)
        .into_par_iter()
        .map(|i| {
            let row = &sym[i * n..(i + 1) * n];
            let mut xx = [0.0f64; P * P];
            let mut xy = [0.0f64; P];
            for j in (i + 1)..n {
                let s = row[j] as f64;
                if s <= 0.0 {
                    continue;
                }
                let y = (1.0 + s).ln();
                let x = feature_vec(i, j, ind, &f_size, &f_liq, &f_ret);
                for a in 0..P {
                    xy[a] += x[a] * y;
                    for b in 0..P {
                        xx[a * P + b] += x[a] * x[b];
                    }
                }
            }
            (xx, xy)
        })
        .collect();
    let mut xx = [0.0f64; P * P];
    let mut xy = [0.0f64; P];
    for (lxx, lxy) in local.iter() {
        for a in 0..P * P {
            xx[a] += lxx[a];
        }
        for a in 0..P {
            xy[a] += lxy[a];
        }
    }
    // 高斯消元解 β
    let beta = solve(&xx, &xy);

    // ---- 第二遍: 残差和
    let col_pos: Vec<(f32, f32)> = (0..n)
        .into_par_iter()
        .map(|i| {
            let row = &sym[i * n..(i + 1) * n];
            let mut pos = 0.0f64;
            let mut neg = 0.0f64;
            for j in 0..n {
                if j == i {
                    continue;
                }
                let s = row[j] as f64;
                if s <= 0.0 {
                    continue;
                }
                let y = (1.0 + s).ln();
                let x = feature_vec(i, j, ind, &f_size, &f_liq, &f_ret);
                let mut pred = 0.0f64;
                for a in 0..P {
                    pred += x[a] * beta[a];
                }
                let e = y - pred;
                if e > 0.0 {
                    pos += e;
                } else {
                    neg += -e;
                }
            }
            // 输出相对规模: 除以 (N-1) 使量级可比（注释说明）
            let inv = 1.0 / (n as f64 - 1.0).max(1.0);
            ((pos * inv) as f32, (neg * inv) as f32)
        })
        .collect();
    let pos: Vec<f32> = col_pos.iter().map(|&(p, _)| p).collect();
    let neg: Vec<f32> = col_pos.iter().map(|&(_, q)| q).collect();
    out.push(IndicatorResult::new(format!("hid_{mat}_degree_pos"), pos));
    out.push(IndicatorResult::new(format!("hid_{mat}_degree_neg"), neg));
    }
    out
}

#[inline]
fn feature_vec(
    i: usize,
    j: usize,
    ind: Option<&[i16]>,
    f_size: &[f64],
    f_liq: &[f64],
    f_ret: &[f64],
) -> [f64; P] {
    let same_ind = match ind {
        Some(v) => {
            let (a, b) = (v[i], v[j]);
            if a >= 0 && b >= 0 && a == b {
                1.0
            } else {
                0.0
            }
        }
        None => 0.0,
    };
    [
        1.0,
        same_ind,
        (f_size[i] - f_size[j]).abs(),
        (f_liq[i] - f_liq[j]).abs(),
        (f_ret[i] - f_ret[j]).abs(),
    ]
}

/// 高斯消元解 P×P 线性方程组
fn solve(a: &[f64; P * P], b: &[f64; P]) -> [f64; P] {
    let mut m = [[0.0f64; P]; P];
    let mut v = [0.0f64; P];
    for i in 0..P {
        for j in 0..P {
            m[i][j] = a[i * P + j];
        }
        v[i] = b[i];
    }
    for col in 0..P {
        // 选主元
        let mut piv = col;
        for r in (col + 1)..P {
            if m[r][col].abs() > m[piv][col].abs() {
                piv = r;
            }
        }
        if m[piv][col].abs() < 1e-18 {
            continue;
        }
        m.swap(col, piv);
        v.swap(col, piv);
        let inv = 1.0 / m[col][col];
        for j in col..P {
            m[col][j] *= inv;
        }
        v[col] *= inv;
        for r in 0..P {
            if r != col {
                let f = m[r][col];
                if f != 0.0 {
                    for j in col..P {
                        m[r][j] -= f * m[col][j];
                    }
                    v[r] -= f * v[col];
                }
            }
        }
    }
    v
}
