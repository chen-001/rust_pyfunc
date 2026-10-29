//! Small deterministic statistics for the refined network; no random sampling.
use crate::yupei_gpt::model::AuditRow;
use rayon::prelude::*;

pub(super) fn mean(v: &[f64]) -> f64 {
    if v.is_empty() {
        f64::NAN
    } else {
        v.iter().sum::<f64>() / v.len() as f64
    }
}
pub(super) fn quantile_sorted(v: &[f64], p: f64) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    let x = p * (v.len() - 1) as f64;
    let lo = x.floor() as usize;
    let hi = x.ceil() as usize;
    v[lo] + (v[hi] - v[lo]) * (x - lo as f64)
}
pub(super) fn topmean_sorted(v: &[f64], k: usize) -> f64 {
    mean(&v[v.len().saturating_sub(k)..])
}
pub(super) fn skew(v: &[f64]) -> f64 {
    if v.len() < 3 {
        return f64::NAN;
    }
    let m = mean(v);
    let var = v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / v.len() as f64;
    if var <= 1e-24 {
        f64::NAN
    } else {
        v.iter().map(|x| (x - m).powi(3)).sum::<f64>() / v.len() as f64 / var.powf(1.5)
    }
}
pub(super) fn hhi(v: &[f64]) -> f64 {
    let s = v.iter().filter(|x| **x > 0.).sum::<f64>();
    if s > 0. {
        v.iter().filter(|x| **x > 0.).map(|x| (x / s).powi(2)).sum()
    } else {
        f64::NAN
    }
}
pub(super) fn ranks(v: &[f64]) -> Vec<f64> {
    let mut idx: Vec<_> = (0..v.len()).collect();
    idx.sort_unstable_by(|&i, &j| v[i].total_cmp(&v[j]).then(i.cmp(&j)));
    let mut out = vec![0.; v.len()];
    let mut k = 0;
    while k < idx.len() {
        let mut end = k + 1;
        while end < idx.len() && v[idx[end]] == v[idx[k]] {
            end += 1;
        }
        let r = (k + end - 1) as f64 / 2.;
        for t in k..end {
            out[idx[t]] = r;
        }
        k = end;
    }
    out
}
pub(super) fn correlation(x: &[f64], y: &[f64]) -> Option<f64> {
    if x.len() != y.len() || x.len() < 3 {
        return None;
    }
    let (mx, my) = (mean(x), mean(y));
    let (mut xx, mut yy, mut xy) = (0., 0., 0.);
    for (&a, &b) in x.iter().zip(y) {
        let (a, b) = (a - mx, b - my);
        xx += a * a;
        yy += b * b;
        xy += a * b;
    }
    (xx > 1e-24 && yy > 1e-24).then(|| (xy / (xx * yy).sqrt()).clamp(-1., 1.))
}
pub(super) fn rank_audit(name: String, x: &[f64], y: &[f64]) -> AuditRow {
    let pairs: Vec<_> = x
        .iter()
        .zip(y)
        .filter(|(a, b)| a.is_finite() && b.is_finite())
        .map(|(&a, &b)| (a, b))
        .collect();
    let a: Vec<_> = pairs.iter().map(|p| p.0).collect();
    let b: Vec<_> = pairs.iter().map(|p| p.1).collect();
    let value = correlation(&ranks(&a), &ranks(&b));
    AuditRow {
        name,
        layer: "cross_section".into(),
        n: pairs.len(),
        value,
        detail: if pairs.len() < 3 {
            "Undefined: fewer than 3 common valid stocks".into()
        } else if value.is_none() {
            "Undefined: zero cross-sectional rank variance on one or both sides".into()
        } else {
            "Spearman ranks aligned by identical stock index; ties use average rank".into()
        },
    }
}

/// Six regression columns including the intercept. Pair covariates are invariant
/// to exchanging (B,a) with (C,b); no code-position-specific lnB/lnC coefficients.
#[derive(Clone)]
pub(super) struct Ridge {
    pub mean_x: [f64; 5],
    pub scale: [f64; 5],
    pub mean_y: f64,
    pub beta: [f64; 5],
    pub n: usize,
}
impl Ridge {
    pub fn residual(&self, x: [f64; 5], y: f64) -> f64 {
        y - self.mean_y
            - (0..5)
                .map(|k| self.beta[k] * (x[k] - self.mean_x[k]) / self.scale[k])
                .sum::<f64>()
    }
}
/// Center/scale each nonconstant regressor, then solve (X'X/n + 1e-6 I)b=X'y/n.
/// The intercept is unpenalized (centering); constant columns have scale=1 and
/// zero covariance. Cholesky is valid because the positive ridge makes the
/// standardized normal matrix positive definite. Tiny/nonfinite pivots are an
/// explicit failed fit, never replaced by coefficients of zero.
#[allow(dead_code)]
pub(super) fn fit_ridge<F: FnMut(&mut dyn FnMut([f64; 5], f64))>(mut each: F) -> Option<Ridge> {
    let (mut n, mut sx, mut sy) = (0usize, [0.; 5], 0.);
    each(&mut |x, y| {
        if y.is_finite() && x.iter().all(|v| v.is_finite()) {
            n += 1;
            sy += y;
            for k in 0..5 {
                sx[k] += x[k];
            }
        }
    });
    if n < 8 {
        return None;
    }
    let mx = sx.map(|v| v / n as f64);
    let my = sy / n as f64;
    let mut cov = [[0.; 5]; 5];
    let mut cy = [0.; 5];
    each(&mut |x, y| {
        if y.is_finite() && x.iter().all(|v| v.is_finite()) {
            for a in 0..5 {
                let xa = x[a] - mx[a];
                cy[a] += xa * (y - my);
                for b in 0..=a {
                    cov[a][b] += xa * (x[b] - mx[b]);
                }
            }
        }
    });
    let scale = std::array::from_fn(|k| {
        let s = (cov[k][k] / n as f64).sqrt();
        if s > 1e-12 {
            s
        } else {
            1.
        }
    });
    let mut a = [[0.; 5]; 5];
    let mut b = [0.; 5];
    for i in 0..5 {
        b[i] = cy[i] / (n as f64 * scale[i]);
        for k in 0..=i {
            let v = cov[i][k] / (n as f64 * scale[i] * scale[k]);
            a[i][k] = v;
            a[k][i] = v;
        }
        a[i][i] += 1e-6;
    }
    let mut l = [[0.; 5]; 5];
    for i in 0..5 {
        for k in 0..=i {
            let s = a[i][k] - (0..k).map(|t| l[i][t] * l[k][t]).sum::<f64>();
            if i == k {
                if !s.is_finite() || s <= 0. {
                    return None;
                }
                l[i][k] = s.sqrt();
            } else {
                l[i][k] = s / l[k][k];
            }
        }
    }
    let mut z = [0.; 5];
    for i in 0..5 {
        z[i] = (b[i] - (0..i).map(|k| l[i][k] * z[k]).sum::<f64>()) / l[i][i];
    }
    let mut beta = [0.; 5];
    for i in (0..5).rev() {
        beta[i] = (z[i] - (i + 1..5).map(|k| l[k][i] * beta[k]).sum::<f64>()) / l[i][i];
    }
    if !beta.iter().all(|v| v.is_finite()) {
        return None;
    }
    Some(Ridge {
        mean_x: mx,
        scale,
        mean_y: my,
        beta,
        n,
    })
}

/// Parallel two-pass ridge over a flat index domain; `at` returns the
/// (covariates, y) of a finite observation or None. Chunk partial sums are
/// combined in index order, so results are deterministic across runs.
pub(super) fn fit_ridge_par<F>(nn: usize, at: &F) -> Option<Ridge>
where
    F: Fn(usize) -> Option<([f64; 5], f64)> + Sync,
{
    const CHUNK: usize = 1 << 16;
    type Sums = (usize, [f64; 5], f64);
    let pass1: Vec<Sums> = (0..nn.div_ceil(CHUNK))
        .into_par_iter()
        .map(|blk| {
            let mut acc: Sums = (0, [0.; 5], 0.);
            for k in blk * CHUNK..((blk + 1) * CHUNK).min(nn) {
                if let Some((x, y)) = at(k) {
                    acc.0 += 1;
                    acc.2 += y;
                    for j in 0..5 {
                        acc.1[j] += x[j];
                    }
                }
            }
            acc
        })
        .collect();
    let (mut n, mut sx, mut sy) = (0usize, [0.; 5], 0.);
    for (cn, cx, cy) in pass1 {
        n += cn;
        sy += cy;
        for j in 0..5 {
            sx[j] += cx[j];
        }
    }
    if n < 8 {
        return None;
    }
    let mx = sx.map(|v| v / n as f64);
    let my = sy / n as f64;
    type Cov = ([[f64; 5]; 5], [f64; 5]);
    let pass2: Vec<Cov> = (0..nn.div_ceil(CHUNK))
        .into_par_iter()
        .map(|blk| {
            let mut cov = [[0.; 5]; 5];
            let mut cy = [0.; 5];
            for k in blk * CHUNK..((blk + 1) * CHUNK).min(nn) {
                if let Some((x, y)) = at(k) {
                    for a in 0..5 {
                        let xa = x[a] - mx[a];
                        cy[a] += xa * (y - my);
                        for b in 0..=a {
                            cov[a][b] += xa * (x[b] - mx[b]);
                        }
                    }
                }
            }
            (cov, cy)
        })
        .collect();
    let mut cov = [[0.; 5]; 5];
    let mut cy = [0.; 5];
    for (cc, cy2) in pass2 {
        for a in 0..5 {
            cy[a] += cy2[a];
            for b in 0..=a {
                cov[a][b] += cc[a][b];
            }
        }
    }
    let scale = std::array::from_fn(|k| {
        let s = (cov[k][k] / n as f64).sqrt();
        if s > 1e-12 {
            s
        } else {
            1.
        }
    });
    let mut a = [[0.; 5]; 5];
    let mut b = [0.; 5];
    for i in 0..5 {
        b[i] = cy[i] / (n as f64 * scale[i]);
        for k in 0..=i {
            let v = cov[i][k] / (n as f64 * scale[i] * scale[k]);
            a[i][k] = v;
            a[k][i] = v;
        }
        a[i][i] += 1e-6;
    }
    let mut l = [[0.; 5]; 5];
    for i in 0..5 {
        for k in 0..=i {
            let s = a[i][k] - (0..k).map(|t| l[i][t] * l[k][t]).sum::<f64>();
            if i == k {
                if !s.is_finite() || s <= 0. {
                    return None;
                }
                l[i][k] = s.sqrt();
            } else {
                l[i][k] = s / l[k][k];
            }
        }
    }
    let mut z = [0.; 5];
    for i in 0..5 {
        z[i] = (b[i] - (0..i).map(|k| l[i][k] * z[k]).sum::<f64>()) / l[i][i];
    }
    let mut beta = [0.; 5];
    for i in (0..5).rev() {
        beta[i] = (z[i] - (i + 1..5).map(|k| l[k][i] * beta[k]).sum::<f64>()) / l[i][i];
    }
    if !beta.iter().all(|v| v.is_finite()) {
        return None;
    }
    Some(Ridge {
        mean_x: mx,
        scale,
        mean_y: my,
        beta,
        n,
    })
}
