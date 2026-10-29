//! Exact all-market two-member complementarity network.
//!
//! Each row B always uses direction a; each partner C always uses direction b.
//! Every eligible ordered B,C scans every eligible X (excluding B,C), including
//! mixed directions. There is no code-order role assignment or sparse shortcut.
//! Same-direction chains are also scanned as ordered pairs: X counts are twice
//! the legacy unordered counts; normalized/row statistics are unaffected by it.
//!
//! G is the improvement at the X maximizing combined association H; GS is the
//! maximum improvement across X. These are distinct objectives and are retained.
//! Z uses exact all-pair activity-stratum moments, not legacy random draws.
//! Residuals use standardized symmetric covariates and ridge (see stats module),
//! hence their numerical values intentionally differ from the legacy residuals.
use crate::model::{AuditRow, FactorColumn, Kernel, NeighborState, NetworkOutput, Stock};
use rayon::prelude::*;
use std::io;
#[path = "network_simd.rs"]
mod simd;
#[path = "network_stats.rs"]
mod stats;
use stats::{fit_ridge, hhi, mean, quantile_sorted, rank_audit, skew, topmean_sorted};

const BLOCK_ROWS: usize = 16; // Independent of the Rayon thread count.
const NAN: f64 = f64::NAN;
#[derive(Clone, Copy)]
struct Pair {
    g: f32,
    gs: f32,
    gnpos: f32,
    gcov: f32,
    self_assoc: f32,
}
impl Pair {
    fn missing() -> Self {
        Self {
            g: f32::NAN,
            gs: f32::NAN,
            gnpos: f32::NAN,
            gcov: f32::NAN,
            self_assoc: f32::NAN,
        }
    }
}
struct Block {
    mass: [Vec<f32>; 3],
    h_count: Vec<u64>,
    g_count: Vec<u64>,
    incident: Vec<u64>,
    pairs: u64,
}
fn invalid(s: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, s.into())
}
fn directions(chain: &str) -> io::Result<(usize, usize, usize)> {
    if ![
        "AAa", "AAp", "AAm", "ppa", "ppp", "ppm", "pma", "pmp", "pmm", "mpa", "mpp", "mpm", "mma",
        "mmp", "mmm",
    ]
    .contains(&chain)
    {
        return Err(invalid(format!("Unsupported network chain {chain}")));
    }
    let d = |c: u8| match c {
        b'A' | b'a' => 0,
        b'p' => 1,
        b'm' => 2,
        _ => unreachable!(),
    };
    let b = chain.as_bytes();
    Ok((d(b[0]), d(b[1]), d(b[2])))
}
fn activity(s: &Stock, d: usize) -> f64 {
    match d {
        0 => s.n_buy as f64 + s.n_sell as f64,
        1 => s.n_buy as f64,
        _ => s.n_sell as f64,
    }
}
fn bins(v: &[f64]) -> Vec<usize> {
    let mut finite: Vec<_> = v.iter().copied().filter(|x| x.is_finite()).collect();
    finite.sort_unstable_by(f64::total_cmp);
    let cuts: Vec<_> = (1..10)
        .map(|k| quantile_sorted(&finite, k as f64 / 10.))
        .collect();
    v.iter()
        .map(|&x| cuts.partition_point(|&c| x > c))
        .collect()
}
fn sorted_values(it: impl Iterator<Item = f64>) -> Vec<f64> {
    let mut v: Vec<_> = it.filter(|x| x.is_finite()).collect();
    v.sort_unstable_by(f64::total_cmp);
    v
}
// Exact f32 order; the low half resolves genuine equal scores by stock code.
fn key(score: f32, code_rank: u32) -> u64 {
    let b = if score == 0. {
        0.0f32.to_bits()
    } else {
        score.to_bits()
    };
    let ascending = if b & 0x8000_0000 != 0 {
        !b
    } else {
        b ^ 0x8000_0000
    };
    ((!ascending as u64) << 32) | code_rank as u64
}
fn top_keys(scores: &[f32], take: usize, code_rank: &[u32], out: &mut [u64], scratch: &mut [u64]) {
    simd::select_topk_keys(scores, take, code_rank, out, scratch);
}
fn top_neighbors(row: &[f32], i: usize, k: usize, codes: &[Stock], positive: bool) -> Vec<usize> {
    let mut indices: Vec<_> = (0..row.len())
        .filter(|&j| j != i && row[j].is_finite() && (!positive || row[j] > 0.))
        .collect();
    let cmp = |&a: &usize, &b: &usize| {
        row[b]
            .total_cmp(&row[a])
            .then(codes[a].code.cmp(&codes[b].code))
    };
    if indices.len() > k {
        indices.select_nth_unstable_by(k - 1, cmp);
        indices.truncate(k);
    }
    indices.sort_unstable_by(cmp);
    indices
}
fn overlap(a: &[usize], b: &[usize]) -> f64 {
    if a.is_empty() || b.is_empty() {
        return NAN;
    }
    // Effective-K denominator prevents an artificial dilution in small universes.
    a.iter().filter(|x| b.contains(x)).count() as f64 / a.len().min(b.len()) as f64
}
fn reverse_rank(row: &[f32], partner: usize, threshold: f32) -> (f64, f64) {
    let (mut greater, mut valid) = (0usize, 0usize);
    for (k, &score) in row.iter().enumerate() {
        if k != partner && score.is_finite() {
            valid += 1;
            greater += (score > threshold) as usize;
        }
    }
    if valid == 0 || !threshold.is_finite() {
        (NAN, NAN)
    } else {
        (greater as f64, greater as f64 / valid as f64)
    }
}

pub fn compute_network(
    stocks: &[Stock],
    kernel: &Kernel,
    chain: &str,
) -> io::Result<NetworkOutput> {
    let (a, b, c) = directions(chain)?;
    let n = stocks.len();
    if n == 0 || kernel.n != n {
        return Err(invalid(
            "Network requires a nonempty stock universe matching Kernel.n",
        ));
    }
    let nn = n
        .checked_mul(n)
        .ok_or_else(|| invalid("Network matrix size overflow"))?;
    if kernel.energy.iter().any(|v| v.len() != n) || kernel.rho.iter().any(|v| v.len() != nn) {
        return Err(invalid("Kernel energy/rho dimensions do not match stocks"));
    }
    if kernel
        .energy
        .iter()
        .flatten()
        .any(|&x| !x.is_finite() || x < 0.)
    {
        return Err(invalid("Kernel energy must be finite and nonnegative"));
    }
    let mut code_order: Vec<_> = (0..n).collect();
    code_order.sort_unstable_by(|&i, &j| stocks[i].code.cmp(&stocks[j].code));
    if code_order
        .windows(2)
        .any(|w| stocks[w[0]].code == stocks[w[1]].code)
    {
        return Err(invalid("Duplicate stock code in network universe"));
    }
    let mut code_rank = vec![0u32; n];
    for (rank, &i) in code_order.iter().enumerate() {
        code_rank[i] = rank as u32;
    }
    let (ea, eb, ex) = (&kernel.energy[a], &kernel.energy[b], &kernel.energy[c]);
    let va: Vec<_> = ea.iter().map(|x| *x > 0.).collect();
    let vb: Vec<_> = eb.iter().map(|x| *x > 0.).collect();
    let vx: Vec<_> = ex.iter().map(|x| *x > 0.).collect();
    let sa: Vec<_> = ea.iter().map(|x| x.sqrt()).collect();
    let sb: Vec<_> = eb.iter().map(|x| x.sqrt()).collect();
    let rbc = &kernel.rho[a * 3 + b];
    let pr = &kernel.rho[a * 3 + c];
    let qr = &kernel.rho[b * 3 + c];
    // A missing direction has a finite placeholder only inside the vector kernel;
    // its -infinity mask excludes it from both numerator and eligible-X count.
    // Missing rho despite positive directional energies is a malformed Kernel.
    for (mat, vi, label) in [(pr, &va, "a/c"), (qr, &vb, "b/c")] {
        for i in 0..n {
            if !vi[i] {
                continue;
            }
            for x in 0..n {
                if vx[x] && !mat[i * n + x].is_finite() {
                    return Err(invalid(format!(
                        "Nonfinite {label} rho with positive energies: {} vs {}",
                        stocks[i].code, stocks[x].code
                    )));
                }
            }
        }
    }
    let p: Vec<_> = pr
        .iter()
        .map(|&v| if v.is_finite() { v } else { 0. })
        .collect();
    let q: Vec<_> = qr
        .iter()
        .map(|&v| if v.is_finite() { v } else { 0. })
        .collect();
    let addm: Vec<_> = vx
        .iter()
        .map(|&v| if v { 0. } else { f32::NEG_INFINITY })
        .collect();
    let nx = vx.iter().filter(|&&x| x).count();
    let mut pairs = vec![Pair::missing(); nn];
    let blocks: Vec<Block> = pairs
        .par_chunks_mut(BLOCK_ROWS * n)
        .enumerate()
        .map(|(blk, rows)| {
            let mut result = Block {
                mass: std::array::from_fn(|_| vec![0.; n]),
                h_count: vec![0; n],
                g_count: vec![0; n],
                incident: vec![0; n],
                pairs: 0,
            };
            let (mut hb, mut gb) = (vec![0.; n], vec![0.; n]);
            let (mut hk, mut gk) = (vec![0u64; n], vec![0u64; n]);
            let mut scratch = vec![0u64; n];
            for (offset, row) in rows.chunks_mut(n).enumerate() {
                let i = blk * BLOCK_ROWS + offset;
                if !va[i] {
                    continue;
                }
                for j in 0..n {
                    if i == j || !vb[j] {
                        continue;
                    }
                    let nv = nx - (vx[i] as usize) - (vx[j] as usize);
                    if nv == 0 {
                        continue;
                    }
                    let r = rbc[i * n + j] as f64;
                    if !r.is_finite() {
                        continue;
                    }
                    let d2 = ea[i] + eb[j] + 2. * r * sa[i] * sb[j];
                    if !d2.is_finite() || d2 <= 0. {
                        continue;
                    }
                    let d = d2.sqrt();
                    let (wi, wj) = ((sa[i] / d) as f32, (sb[j] / d) as f32);
                    let pi = &p[i * n..(i + 1) * n];
                    let qj = &q[j * n..(j + 1) * n];
                    let (_mh, mg, sh, sg, _sq, npos) =
                        simd::fused_hg_stats(&mut hb, &mut gb, wi, wj, pi, qj, &addm, i, j);
                    let ht = 20.min(nv);
                    let gt = 100.min(nv);
                    top_keys(&hb, ht, &code_rank, &mut hk, &mut scratch);
                    top_keys(&gb, gt, &code_rank, &mut gk, &mut scratch);
                    let x = code_order[hk[0] as u32 as usize];
                    let h = hb[x];
                    row[j] = Pair {
                        g: h - pi[x].max(qj[x]),
                        gs: mg,
                        gnpos: npos as f32 / nv as f32,
                        gcov: sg / nv as f32,
                        self_assoc: pi[x],
                    };
                    for &k in &hk[..ht] {
                        result.h_count[code_order[k as u32 as usize]] += 1;
                    }
                    for &k in &gk[..gt] {
                        result.g_count[code_order[k as u32 as usize]] += 1;
                    }
                    if sh > 0. && sh.is_finite() {
                        let [em, gm, g2] = &mut result.mass;
                        simd::mass_acc_hg(
                            em,
                            gm,
                            g2,
                            &mut [],
                            &mut [],
                            &mut [],
                            &hb,
                            &gb,
                            1. / sh,
                            0.,
                            true,
                            false,
                        );
                    }
                    result.pairs += 1;
                    result.incident[i] += 1;
                    result.incident[j] += 1;
                }
            }
            result
        })
        .collect();
    drop(p);
    drop(q);
    let (mut total_pairs, mut mass, mut hc, mut gc, mut incident) = (
        0u64,
        std::array::from_fn::<_, 3, _>(|_| vec![0f64; n]),
        vec![0u64; n],
        vec![0u64; n],
        vec![0u64; n],
    );
    for block in blocks {
        total_pairs += block.pairs;
        for x in 0..n {
            for k in 0..3 {
                mass[k][x] += block.mass[k][x] as f64;
            }
            hc[x] += block.h_count[x];
            gc[x] += block.g_count[x];
            incident[x] += block.incident[x];
        }
    }
    let ln_a: Vec<_> = stocks.iter().map(|s| activity(s, a).ln_1p()).collect();
    let ln_b: Vec<_> = stocks.iter().map(|s| activity(s, b).ln_1p()).collect();
    let le_a: Vec<_> = ea.iter().map(|e| e.ln_1p()).collect();
    let le_b: Vec<_> = eb.iter().map(|e| e.ln_1p()).collect();
    // Pool activity cuts across the two roles, making strata invariant under
    // exchanging the directional roles, without swapping the actual stocks.
    let mut pool = ln_a.clone();
    pool.extend_from_slice(&ln_b);
    let dec = bins(&pool);
    let cell = |i: usize, j: usize| {
        let x = dec[i];
        let y = dec[n + j];
        x.min(y) * 10 + x.max(y)
    };
    let mut moments = vec![(0usize, 0f64, 0f64); 100];
    let (mut alln, mut allsum, mut allsq) = (0usize, 0f64, 0f64);
    for i in 0..n {
        for j in 0..n {
            let v = pairs[i * n + j].g as f64;
            if v.is_finite() {
                let t = &mut moments[cell(i, j)];
                t.0 += 1;
                t.1 += v;
                t.2 += v * v;
                alln += 1;
                allsum += v;
                allsq += v * v;
            }
        }
    }
    let global = if alln > 0 {
        let m = allsum / alln as f64;
        (m, (allsq / alln as f64 - m * m).max(0.).sqrt())
    } else {
        (NAN, NAN)
    };
    let z_fallback_pairs: usize = moments.iter().filter(|v| v.0 < 30).map(|v| v.0).sum();
    let moments: Vec<_> = moments
        .iter()
        .map(|&(count, sum, sq)| {
            if count >= 30 {
                let m = sum / count as f64;
                (m, (sq / count as f64 - m * m).max(0.).sqrt())
            } else {
                global
            }
        })
        .collect();
    let mut z = vec![f32::NAN; nn];
    z.par_chunks_mut(n).enumerate().for_each(|(i, row)| {
        for j in 0..n {
            let v = pairs[i * n + j].g as f64;
            let (m, sd) = moments[cell(i, j)];
            if v.is_finite() && sd > 1e-12 {
                row[j] = ((v - m) / sd) as f32;
            }
        }
    });
    let cov = |i: usize, j: usize| {
        [
            rbc[i * n + j] as f64,
            ln_a[i] + ln_b[j],
            (ln_a[i] - ln_b[j]).abs(),
            le_a[i] + le_b[j],
            (le_a[i] - le_b[j]).abs(),
        ]
    };
    let fit_g = fit_ridge(|visit| {
        for i in 0..n {
            for j in 0..n {
                let y = pairs[i * n + j].g as f64;
                if y.is_finite() {
                    visit(cov(i, j), y);
                }
            }
        }
    });
    let fit_z = fit_ridge(|visit| {
        for i in 0..n {
            for j in 0..n {
                let y = z[i * n + j] as f64;
                if y.is_finite() {
                    visit(cov(i, j), y);
                }
            }
        }
    });
    // Keep legacy OR-union graph overlap as well as a clearly named directed
    // variant. OR is a topology reduction only; it never swaps scoring roles.
    let tops: Vec<_> = (0..n)
        .into_par_iter()
        .map(|i| {
            let g: Vec<_> = (0..n).map(|j| pairs[i * n + j].g).collect();
            let r: Vec<_> = (0..n)
                .map(|j| {
                    if g[j].is_finite() {
                        rbc[i * n + j]
                    } else {
                        f32::NAN
                    }
                })
                .collect();
            [
                top_neighbors(&r, i, 100, stocks, false),
                top_neighbors(&g, i, 100, stocks, false),
                top_neighbors(&z[i * n..(i + 1) * n], i, 100, stocks, false),
            ]
        })
        .collect();
    let union: Vec<Vec<Vec<usize>>> = (0..3)
        .map(|k| {
            let mut adj = vec![Vec::new(); n];
            for i in 0..n {
                for &j in &tops[i][k] {
                    adj[i].push(j);
                    adj[j].push(i);
                }
            }
            for v in &mut adj {
                v.sort_unstable();
                v.dedup();
            }
            adj
        })
        .collect();
    let rowdata: Vec<_> = (0..n)
        .into_par_iter()
        .map(|i| {
            let row = &pairs[i * n..(i + 1) * n];
            let zr = &z[i * n..(i + 1) * n];
            let rr = &rbc[i * n..(i + 1) * n];
            let mut vals = [NAN; 32];
            let mut neighbor = Vec::new();
            let mut audit_r = [NAN; 3];
            let mut audit_g = [NAN; 3];
            if !va[i] {
                return (vals, neighbor, audit_r, audit_g);
            }
            let g = sorted_values(row.iter().map(|v| v.g as f64));
            if g.is_empty() {
                return (vals, neighbor, audit_r, audit_g);
            }
            let gs = sorted_values(row.iter().map(|v| v.gs as f64));
            let gn = sorted_values(row.iter().map(|v| v.gnpos as f64));
            let self_ass = sorted_values(row.iter().map(|v| v.self_assoc as f64));
            let rv = sorted_values((0..n).filter(|&j| j != i && vb[j]).map(|j| rr[j] as f64));
            vals[0] = skew(&rv);
            vals[1] = skew(&gs);
            vals[2] = quantile_sorted(&gn, 0.75);
            vals[3] = quantile_sorted(&gn, 0.9);
            let rg = sorted_values((0..n).filter_map(|j| {
                fit_g
                    .as_ref()
                    .map(|f| f.residual(cov(i, j), row[j].g as f64))
            }));
            let rz = sorted_values(
                (0..n).filter_map(|j| fit_z.as_ref().map(|f| f.residual(cov(i, j), zr[j] as f64))),
            );
            vals[5] = quantile_sorted(&rz, 0.9);
            vals[6] = rg.last().copied().unwrap_or(NAN);
            vals[7] = topmean_sorted(&rg, 20);
            let gsv: Vec<_> = row.iter().map(|p| p.gs).collect();
            for (k, out) in [(1, 8), (2, 9)] {
                if !union[0][i].is_empty() && !union[k][i].is_empty() {
                    vals[out] = union[0][i]
                        .iter()
                        .filter(|&&j| union[k][i].binary_search(&j).is_ok())
                        .count() as f64
                        / 100.;
                }
            }
            vals[22] = overlap(&tops[i][0], &tops[i][1]);
            vals[23] = overlap(&tops[i][0], &tops[i][2]);
            vals[12] = quantile_sorted(&self_ass, 0.95);
            vals[13] = mean(&sorted_values(row.iter().map(|p| p.gcov as f64)));
            vals[14] = hhi(&gs);
            vals[15] = quantile_sorted(&g, 0.9);
            vals[16] = topmean_sorted(&g, 20);
            vals[17] = if vals[14].is_finite() {
                1. / vals[14]
            } else {
                NAN
            };
            // Continuous weighted Jaccard on nonnegative rho and G row masses.
            let sr = (0..n)
                .filter(|&j| row[j].g.is_finite())
                .map(|j| (rr[j] as f64).max(0.))
                .sum::<f64>();
            let sg = g.iter().map(|v| v.max(0.)).sum::<f64>();
            if sr > 0. && sg > 0. {
                let (mut lo, mut hi) = (0., 0.);
                for j in 0..n {
                    if row[j].g.is_finite() {
                        let x = (rr[j] as f64).max(0.) / sr;
                        let y = (row[j].g as f64).max(0.) / sg;
                        lo += x.min(y);
                        hi += x.max(y);
                    }
                }
                if hi > 0. {
                    vals[18] = lo / hi;
                }
            }
            let mut same = Vec::new();
            let mut cross = Vec::new();
            for j in 0..n {
                let v = row[j].g as f64;
                if !v.is_finite() || stocks[i].industry <= 0 || stocks[j].industry <= 0 {
                    continue;
                }
                if stocks[i].industry == stocks[j].industry {
                    same.push(v)
                } else {
                    cross.push(v)
                }
            }
            vals[19] = mean(&same);
            vals[20] = mean(&cross);
            vals[21] = vals[19] - vals[20];
            neighbor = top_neighbors(&gsv, i, 100, stocks, true)
                .into_iter()
                .map(|j| stocks[j].code.clone())
                .collect();
            // Both aggregations use exactly the same valid pair edges.
            let ar = sorted_values(
                (0..n)
                    .filter(|&j| row[j].g.is_finite())
                    .map(|j| rr[j] as f64),
            );
            let chosen = &tops[i][1]; // G-selected partners, shared by both sides.
            let same_set_rho = if chosen.is_empty() {
                NAN
            } else {
                chosen.iter().map(|&j| rr[j] as f64).sum::<f64>() / chosen.len() as f64
            };
            let same_set_g = if chosen.is_empty() {
                NAN
            } else {
                chosen.iter().map(|&j| row[j].g as f64).sum::<f64>() / chosen.len() as f64
            };
            audit_r = [mean(&ar), ar.last().copied().unwrap_or(NAN), same_set_rho];
            audit_g = [mean(&g), g.last().copied().unwrap_or(NAN), same_set_g];
            vals[24..27].copy_from_slice(&audit_r);
            vals[27..30].copy_from_slice(&audit_g);
            if chain == "ppp" {
                // Z top100 already includes finite negative values and resolves
                // ties by stock code. Reuse its first member, then scan only that
                // partner's row: O(N²) total work, with no extra dense matrix.
                if let Some(&p) = tops[i][2].first() {
                    (vals[30], vals[31]) = reverse_rank(&z[p * n..(p + 1) * n], p, zr[p]);
                }
            }
            (vals, neighbor, audit_r, audit_g)
        })
        .collect();
    let names = [
        "row_rho_skew",
        "row_gs_skew",
        "row_gnpos_p75",
        "row_gnpos_p90",
        "e_w_std_soft",
        "res_z_p90",
        "res_g_max",
        "res_g_t20m",
        "r4_ov_g_k100",
        "r4_ov_z_k100",
        "x0topk20_count",
        "gtopk100_count",
        "row_self_assoc_p95",
        "row_gcov_mean",
        "row_gs_hhip",
        "row_g_p90",
        "row_g_t20m",
        "row_gs_effective_neighbors",
        "row_rho_g_weighted_overlap",
        "same_industry_g_mean",
        "cross_industry_g_mean",
        "same_minus_cross_g_mean",
        "r4_ov_g_directed_k100",
        "r4_ov_z_directed_k100",
        "diag_rho_row_mean",
        "diag_rho_row_max",
        "diag_rho_at_g_top100_mean",
        "diag_g_row_mean",
        "diag_g_row_max",
        "diag_g_top100_mean",
        "best_z_partner_reverse_rank",
        "best_z_partner_reverse_rank_fraction",
    ];
    let mut columns = Vec::new();
    for (k, name) in names.iter().enumerate() {
        if k == 0 && !(a == b && b == c) {
            continue;
        }
        if k >= 30 && chain != "ppp" {
            continue;
        }
        let values = (0..n)
            .map(|i| match k {
                4 => {
                    let den = mass[0][i];
                    if vx[i] && total_pairs > incident[i] && den > 0. {
                        (mass[2][i] / den - (mass[1][i] / den).powi(2))
                            .max(0.)
                            .sqrt()
                    } else {
                        NAN
                    }
                }
                10 => {
                    if vx[i] && total_pairs > incident[i] {
                        hc[i] as f64
                    } else {
                        NAN
                    }
                }
                11 => {
                    if vx[i] && total_pairs > incident[i] {
                        gc[i] as f64
                    } else {
                        NAN
                    }
                }
                _ => rowdata[i].0[k],
            })
            .collect();
        columns.push(FactorColumn::new(format!("yr_{chain}_{name}"), values));
    }
    let mut audits = Vec::new();
    if chain == "ppp" {
        audits.push(AuditRow{name:"yr_ppp_best_z_partner_reverse_rank_definition".into(),layer:"definition".into(),n:alln,value:None,detail:"For each i, choose finite Z[i,p] row maximum (negative values are valid; ties by ascending stock code). Count finite k != p with Z[p,k] strictly greater than Z[i,p]. Fraction divides by all finite k != p in p's row, including i. Missing best partner or empty reverse domain is undefined. Preserves the legacy reverse-rank reduction, applied to the new activity-stratified Z.".into()});
    }
    audits.push(AuditRow{name:format!("yr_{chain}_z_global_fallback_pair_fraction"),layer:"coverage".into(),n:alln,value:if alln>0{Some(z_fallback_pairs as f64/alln as f64)}else{None},detail:format!("{z_fallback_pairs} ordered pairs use global G moments because their pooled-activity stratum has fewer than 30 observations. Undefined zero standard deviations remain missing.")});
    audits.push(AuditRow{name:format!("yr_{chain}_graph_definition"),layer:"definition".into(),n:alln,value:None,detail:"r4_ov_*_k100 retains the legacy undirected OR of each row top100 and intersection/100 (can exceed one); *_directed_k100 compares role-specific outgoing rows and divides by effective K. Scoring remains fully ordered; graph OR never assigns roles by code.".into()});
    audits.push(AuditRow{name:format!("yr_{chain}_diagnostic_columns"),layer:"definition".into(),n:alln,value:None,detail:"Six diag_* columns are diagnostic controls, not additional alpha proposals. Top100 uses the same G-selected partner set for both rho and G; means/maxima use the same full valid-pair domain.".into()});
    // Numeric layer: ten rho bins; conditional-mean residual variance / total G
    // variance. This is a diagnostic (in-sample), not an orthogonality guarantee.
    let mut rho_samples = Vec::with_capacity(alln);
    for k in 0..nn {
        if pairs[k].g.is_finite() && rbc[k].is_finite() {
            rho_samples.push(rbc[k] as f64);
        }
    }
    rho_samples.sort_unstable_by(f64::total_cmp);
    let cuts: Vec<_> = (1..10)
        .map(|k| quantile_sorted(&rho_samples, k as f64 / 10.))
        .collect();
    let mut cm = [(0usize, 0f64, 0f64); 10];
    for k in 0..nn {
        let y = pairs[k].g as f64;
        if y.is_finite() && rbc[k].is_finite() {
            let t = &mut cm[cuts.partition_point(|&cut| rbc[k] as f64 > cut)];
            t.0 += 1;
            t.1 += y;
            t.2 += y * y;
        }
    }
    let within = cm
        .iter()
        .filter(|t| t.0 > 0)
        .map(|&(n, s, q)| (q - s * s / n as f64).max(0.))
        .sum::<f64>();
    let total = if alln > 0 {
        (allsq - allsum * allsum / alln as f64).max(0.)
    } else {
        0.
    };
    let ratio = if alln >= 3 && total > 1e-24 {
        Some((within / total).sqrt())
    } else {
        None
    };
    audits.push(AuditRow{name:format!("yr_{chain}_g_rho_conditional_residual_sd_ratio"),layer:"numeric".into(),n:alln,value:ratio,detail:if ratio.is_some(){"sqrt(within-rho-decile G variance / total G variance), exact ordered valid pairs; descriptive in-sample diagnostic".into()}else{"Undefined: fewer than 3 valid pairs or zero G variance".into()}});
    for (k, tag) in ["rowmean", "rowmax", "top100mean"].iter().enumerate() {
        let r: Vec<_> = rowdata.iter().map(|v| v.2[k]).collect();
        let g: Vec<_> = rowdata.iter().map(|v| v.3[k]).collect();
        audits.push(rank_audit(
            format!("yr_{chain}_g_rho_{tag}_rank_corr"),
            &r,
            &g,
        ));
    }
    for (tag, fit) in [("g", fit_g.as_ref()), ("z", fit_z.as_ref())] {
        audits.push(AuditRow{name:format!("yr_{chain}_res_{tag}_ridge"),layer:"regression".into(),n:fit.map(|f|f.n).unwrap_or(0),value:fit.map(|_|1e-6),detail:if fit.is_some(){"Centered standardized [rho, sum log1p directional counts, abs log-count difference, sum log1p directional energies, abs log-energy difference]; ridge=1e-6 on slopes, intercept unpenalized. Exact all-pair activity strata for Z (min 30 else global). New residual definition, not legacy-equivalent.".into()}else{"Undefined regression: fewer than 8 valid pairs or failed positive-definite solve".into()}});
    }
    audits.push(AuditRow{name:format!("yr_{chain}_ordered_pair_count"),layer:"coverage".into(),n:total_pairs as usize,value:Some(total_pairs as f64),detail:"Every ordered direction-valid B,C with at least one valid X is scanned; X excludes both members. Same-direction counts include both orders.".into()});
    Ok(NetworkOutput {
        columns,
        neighbors: NeighborState {
            chain: chain.into(),
            valid: rowdata.iter().map(|r| r.2[0].is_finite()).collect(),
            neighbors: rowdata.into_iter().map(|r| r.1).collect(),
        },
        audits,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn reverse_rank_uses_strict_comparison_and_valid_reverse_domain() {
        let row = [f32::NAN, -2., -1., -1., -0.5, f32::INFINITY];
        // The chosen negative threshold remains valid; equality is not a win.
        assert_eq!(reverse_rank(&row, 0, -1.), (1., 0.25));
        assert_eq!(reverse_rank(&row, 0, -3.), (4., 1.));
        assert_eq!(reverse_rank(&row, 0, 0.), (0., 0.));
        // A finite diagonal must never enter the reverse comparison domain.
        assert_eq!(reverse_rank(&[99., -2., -1.], 0, -1.), (0., 0.));
        let missing = reverse_rank(&[f32::NAN; 3], 1, -1.);
        assert!(missing.0.is_nan() && missing.1.is_nan());
    }
    #[test]
    fn best_z_partner_accepts_negative_values_and_breaks_ties_by_code() {
        let stocks: Vec<_> = ["000004", "000003", "000002", "000001"]
            .iter()
            .map(|&code| Stock {
                code: code.into(),
                industry: 1,
                cells: vec![],
                n_buy: 1,
                n_sell: 1,
                amount: 1.,
                minute_buy: vec![],
                minute_sell: vec![],
                minute_ret: vec![],
                minute_close: vec![],
            })
            .collect();
        let row = [f32::NAN, -1., -1., -2.];
        assert_eq!(top_neighbors(&row, 0, 100, &stocks, false), vec![2, 1, 3]);
        assert!(top_neighbors(&[f32::NAN; 4], 0, 100, &stocks, false).is_empty());
    }
    #[test]
    fn accelerated_selection_is_exact_with_ties_and_missing_x() {
        let n = 4099;
        let ranks: Vec<_> = (0..n as u32).rev().collect();
        for kind in 0..4 {
            let scores: Vec<f32> = (0..n)
                .map(|i| match kind {
                    0 => (((i * 719) % 1009) as f32 - 700.) / 1000.,
                    1 => {
                        if i % 3 == 0 {
                            f32::NEG_INFINITY
                        } else {
                            0.001
                        }
                    }
                    2 => {
                        if i % 100 == 0 {
                            1.
                        } else {
                            f32::NEG_INFINITY
                        }
                    }
                    _ => 1. + ((i * 719) % 1009) as f32 * 1e-7,
                })
                .collect();
            let mut all: Vec<_> = scores
                .iter()
                .zip(&ranks)
                .map(|(&s, &r)| key(s, r))
                .collect();
            all.sort_unstable();
            for take in [1, 20, 100, 1000] {
                let mut result = vec![0; n];
                let mut scratch = vec![0; n];
                top_keys(&scores, take, &ranks, &mut result, &mut scratch);
                assert_eq!(&result[..take], &all[..take]);
            }
        }
    }
}
