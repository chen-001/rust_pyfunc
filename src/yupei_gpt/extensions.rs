//! Selective contrasts and price-response extensions, composed entirely here.
use crate::yupei_gpt::model::{FactorColumn, Kernel, Stock};
use std::collections::BTreeMap;

pub fn ranks(x: &[f64]) -> Vec<f64> {
    let mut order: Vec<_> = x
        .iter()
        .enumerate()
        .filter(|(_, v)| v.is_finite())
        .map(|(i, &v)| (i, v))
        .collect();
    order.sort_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
    let mut out = vec![f64::NAN; x.len()];
    let n = order.len();
    if n < 2 {
        return out;
    }
    let mut i = 0;
    while i < n {
        let mut j = i + 1;
        while j < n && order[j].1 == order[i].1 {
            j += 1;
        }
        for k in i..j {
            out[order[k].0] = (i + j - 1) as f64 / 2.0 / (n - 1) as f64;
        }
        i = j;
    }
    out
}
fn values(cols: &[FactorColumn], name: &str, n: usize) -> Vec<f64> {
    cols.iter()
        .find(|c| c.name == name)
        .map(|c| c.values.iter().map(|v| v.unwrap_or(f64::NAN)).collect())
        .unwrap_or_else(|| vec![f64::NAN; n])
}
fn mean(x: &[f64]) -> f64 {
    let mut s = 0.0;
    let mut n = 0;
    for &v in x {
        if v.is_finite() {
            s += v;
            n += 1;
        }
    }
    if n > 0 {
        s / n as f64
    } else {
        f64::NAN
    }
}

/// A and B are ranked on exactly the same finite stock intersection.
pub fn rank_difference(a: &[f64], b: &[f64]) -> Vec<f64> {
    let a: Vec<_> = a
        .iter()
        .zip(b)
        .map(|(&x, &y)| if y.is_finite() { x } else { f64::NAN })
        .collect();
    let b: Vec<_> = a
        .iter()
        .zip(b)
        .map(|(&x, &y)| if x.is_finite() { y } else { f64::NAN })
        .collect();
    ranks(&a)
        .into_iter()
        .zip(ranks(&b))
        .map(|(x, y)| x - y)
        .collect()
}

/// End-of-day response: future here is later IN THE SAME SESSION and date.
/// Minute close is never forward-filled; missing endpoints invalidate that pair.
pub fn price_columns(stocks: &[Stock]) -> Vec<FactorColumn> {
    let n = stocks.len();
    let mut output = Vec::new();
    for horizon in [1usize, 5, 15] {
        let mut future = vec![vec![f64::NAN; 237]; n];
        for (i, s) in stocks.iter().enumerate() {
            for m in 0..237 - horizon {
                if m < 120 && m + horizon >= 120 {
                    continue;
                }
                let p = s.minute_close[m];
                let q = s.minute_close[m + horizon];
                if p.is_finite() && q.is_finite() && p > 0.0 && q > 0.0 {
                    future[i][m] = (q / p).ln();
                }
            }
        }
        let mut group: BTreeMap<i32, (Vec<f64>, Vec<usize>)> = BTreeMap::new();
        for (i, s) in stocks.iter().enumerate() {
            if s.industry < 1 {
                continue;
            }
            let (sum, count) = group
                .entry(s.industry)
                .or_insert_with(|| (vec![0.0; 237], vec![0; 237]));
            for m in 0..237 {
                if future[i][m].is_finite() {
                    sum[m] += future[i][m];
                    count[m] += 1;
                }
            }
        }
        let mut buy_beta = vec![f64::NAN; n];
        let mut sell_beta = buy_beta.clone();
        let mut absorb = buy_beta.clone();
        let mut exhaust = buy_beta.clone();
        let mut valid_count = buy_beta.clone();
        for (i, s) in stocks.iter().enumerate() {
            let Some((sum, count)) = group.get(&s.industry) else {
                continue;
            };
            let (mut xy_b, mut xx_b, mut xy_s, mut xx_s, mut sw, mut bw, mut a, mut e) =
                (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
            let (mut nb, mut ns) = (0, 0);
            for m in 0..237 {
                let y = future[i][m];
                let total = s.minute_buy[m] + s.minute_sell[m];
                if !y.is_finite() || total <= 0.0 || count[m] < 4 {
                    continue;
                }
                let y = y - (sum[m] - y) / (count[m] - 1) as f64;
                let x = (s.minute_buy[m] - s.minute_sell[m]) / total;
                if x > 0.0 {
                    xy_b += x * y;
                    xx_b += x * x;
                    bw += x;
                    e += if y <= 0.0 { x } else { 0.0 };
                    nb += 1;
                }
                if x < 0.0 {
                    xy_s += x * y;
                    xx_s += x * x;
                    sw -= x;
                    a += if y >= 0.0 { -x } else { 0.0 };
                    ns += 1;
                }
            }
            valid_count[i] = (nb + ns) as f64;
            if nb >= 12 && xx_b > 1e-12 {
                buy_beta[i] = xy_b / xx_b;
                exhaust[i] = e / bw;
            }
            if ns >= 12 && xx_s > 1e-12 {
                sell_beta[i] = xy_s / xx_s;
                absorb[i] = a / sw;
            }
        }
        for (stem, v) in [
            ("buy_response_beta", buy_beta),
            ("sell_response_beta", sell_beta),
            ("sell_absorption", absorb),
            ("buy_exhaustion", exhaust),
            ("response_valid_minutes", valid_count),
        ] {
            output.push(FactorColumn::new(
                format!("yr_{stem}_industry_exself_{horizon}m"),
                v,
            ));
        }
    }
    output
}

pub fn cross_columns(
    stocks: &[Stock],
    kernel: &Kernel,
    base: &[FactorColumn],
) -> Vec<FactorColumn> {
    let n = stocks.len();
    let mut out = Vec::new();
    let count: Vec<_> = stocks
        .iter()
        .map(|s| ((s.n_buy + s.n_sell) as f64).ln_1p())
        .collect();
    let amount: Vec<_> = stocks.iter().map(|s| s.amount.ln_1p()).collect();
    let imbalance: Vec<_> = stocks
        .iter()
        .map(|s| {
            if s.n_buy + s.n_sell > 0 {
                (s.n_buy as f64 - s.n_sell as f64) / (s.n_buy + s.n_sell) as f64
            } else {
                f64::NAN
            }
        })
        .collect();
    let energy: Vec<_> = (0..n)
        .map(|i| {
            if kernel.energy[1][i] > 0.0 && kernel.energy[2][i] > 0.0 {
                (kernel.energy[1][i] / kernel.energy[2][i]).ln()
            } else {
                f64::NAN
            }
        })
        .collect();
    for (name, v) in [
        ("log_trade_count", count.clone()),
        ("log_amount", amount.clone()),
        ("count_imbalance", imbalance.clone()),
        ("log_buy_sell_energy_ratio", energy),
    ] {
        out.push(FactorColumn::new(format!("yr_control_{name}"), v));
    }
    for stem in [
        "row_rho_skew",
        "row_gs_skew",
        "row_gnpos_p90",
        "row_gcov_mean",
        "e_w_std_soft",
        "res_z_p90",
        "r4_ov_g_k100",
        "row_self_assoc_p95",
    ] {
        let buy = values(base, &format!("yr_ppp_{stem}"), n);
        let sell = values(base, &format!("yr_mmm_{stem}"), n);
        out.push(FactorColumn::new(
            format!("yr_buy_minus_sell_rank_{stem}"),
            rank_difference(&buy, &sell),
        ));
    }
    let pm = values(base, "yr_pmp_row_gcov_mean", n);
    let mp = values(base, "yr_mpm_row_gcov_mean", n);
    out.push(FactorColumn::new(
        "yr_cross_role_rank_gcov",
        rank_difference(&pm, &mp),
    ));
    // Within-industry demeaning before simple activity/flow controls. Values are
    // generated here, not in the Python wrapper or the validation report.
    for chain in ["AAa", "ppp", "mmm"] {
        for stem in ["row_gs_skew", "row_gnpos_p90", "res_z_p90"] {
            let y = values(base, &format!("yr_{chain}_{stem}"), n);
            out.push(FactorColumn::new(
                format!("yr_{chain}_{stem}_industry_activity_residual"),
                residualize(stocks, &y, &[&count, &amount, &imbalance]),
            ));
        }
    }
    out
}

/// OLS of industry-demeaned ranks on demeaned ranked activity controls.
/// Ridge only stabilizes a singular control matrix; undefined targets stay NaN.
pub fn residualize(stocks: &[Stock], y: &[f64], controls: &[&[f64]]) -> Vec<f64> {
    let n = y.len();
    let p = controls.len();
    let valid: Vec<_> = (0..n)
        .filter(|&i| {
            stocks[i].industry > 0 && y[i].is_finite() && controls.iter().all(|x| x[i].is_finite())
        })
        .collect();
    let mut result = vec![f64::NAN; n];
    if valid.len() < p + 8 {
        return result;
    }
    let mut columns: Vec<Vec<f64>> = std::iter::once(y)
        .chain(controls.iter().copied())
        .map(|v| {
            let mut a = vec![f64::NAN; n];
            for &i in &valid {
                a[i] = v[i];
            }
            ranks(&a)
        })
        .collect();
    let mut groups: BTreeMap<i32, Vec<usize>> = BTreeMap::new();
    for &i in &valid {
        groups.entry(stocks[i].industry).or_default().push(i);
    }
    for col in &mut columns {
        for indices in groups.values() {
            let avg = indices.iter().map(|&i| col[i]).sum::<f64>() / indices.len() as f64;
            for &i in indices {
                col[i] -= avg;
            }
        }
    }
    let mut gram = vec![vec![0.0; p + 1]; p];
    for &i in &valid {
        for j in 0..p {
            for k in 0..p {
                gram[j][k] += columns[j + 1][i] * columns[k + 1][i];
            }
            gram[j][p] += columns[j + 1][i] * columns[0][i];
        }
    }
    for j in 0..p {
        gram[j][j] += 1e-10;
    }
    for j in 0..p {
        let pivot = (j..p)
            .max_by(|&a, &b| gram[a][j].abs().total_cmp(&gram[b][j].abs()))
            .unwrap();
        gram.swap(j, pivot);
        let d = gram[j][j];
        if d.abs() < 1e-18 {
            return result;
        }
        for k in j..=p {
            gram[j][k] /= d;
        }
        for i in 0..p {
            if i == j {
                continue;
            }
            let f = gram[i][j];
            for k in j..=p {
                gram[i][k] -= f * gram[j][k];
            }
        }
    }
    for &i in &valid {
        if groups[&stocks[i].industry].len() >= 4 {
            result[i] = columns[0][i] - (0..p).map(|j| gram[j][p] * columns[j + 1][i]).sum::<f64>();
        }
    }
    result
}

pub fn interaction_columns(cols: &[FactorColumn], n: usize) -> Vec<FactorColumn> {
    let breadth = ranks(&values(cols, "yr_mmm_row_gnpos_p90", n));
    let absorb = ranks(&values(cols, "yr_sell_absorption_industry_exself_5m", n));
    let buy = ranks(&values(cols, "yr_ppp_row_gnpos_p90", n));
    let exhaust = ranks(&values(cols, "yr_buy_exhaustion_industry_exself_5m", n));
    let mut out = Vec::new();
    for (name, a, b) in [
        ("sell_network_absorption", breadth, absorb),
        ("buy_network_exhaustion", buy, exhaust),
    ] {
        out.push(FactorColumn::new(
            format!("yr_{name}"),
            a.into_iter()
                .zip(b)
                .map(|(x, y)| (x - 0.5) * (y - 0.5))
                .collect(),
        ));
    }
    out
}

pub fn spearman(a: &[f64], b: &[f64]) -> (usize, Option<f64>) {
    let pairs: Vec<_> = a
        .iter()
        .zip(b)
        .filter(|(x, y)| x.is_finite() && y.is_finite())
        .map(|(&x, &y)| (x, y))
        .collect();
    let n = pairs.len();
    if n < 3 {
        return (n, None);
    }
    let x = ranks(&pairs.iter().map(|p| p.0).collect::<Vec<_>>());
    let y = ranks(&pairs.iter().map(|p| p.1).collect::<Vec<_>>());
    let mx = mean(&x);
    let my = mean(&y);
    let xx = x.iter().map(|v| (v - mx).powi(2)).sum::<f64>();
    let yy = y.iter().map(|v| (v - my).powi(2)).sum::<f64>();
    if xx <= 1e-20 || yy <= 1e-20 {
        return (n, None);
    }
    let xy = x
        .iter()
        .zip(y)
        .map(|(v, w)| (v - mx) * (w - my))
        .sum::<f64>();
    (n, Some(xy / (xx * yy).sqrt()))
}
