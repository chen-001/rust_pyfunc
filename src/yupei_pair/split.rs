//! 玉佩2 主买/主卖方向拆分：配对组合 5 种 × 第三只 3 种 = 15 条链。
//!
//! ## 口径
//! 每只股票的全部成交先按主动方向拆成 主买（P，flag 66）与 主卖（M，flag 83），
//! 得到四个分量矩阵（见 `split_mat.rs`）: PP / PM / MP / MM。
//! 配对的两个位置（B、C）各取一个方向集合、第三只 X 再取一个:
//!   全部 = {P, M}、主买 = {P}、主卖 = {M}
//!
//! 三个位置各有一套方向组合，所以交互量也要按位置算:
//!   X_ab[i][j]   = Σ_{p∈a, q∈b} S_pq[i][j]                （i 的 p 领先 j 的 q）
//!   e_i^a        = 2·X_aa[i][i]                            （i 的方向 a 自能量）
//!   R(a,b)[i][j] = (X_ab[i][j] + X_ba[j][i]) / √(e_i^a·e_j^b)
//!
//! 配对网络需要三套: 行 B(a) 对列 X(c)、行 C(b) 对列 X(c)、行 B(a) 对列 C(b)，
//! 分别是 R(a,c)、R(b,c)、R(a,b)。自能量也分两套（B 用 a、C 用 b）。
//!
//! ## 和现有配对网络的关系
//! `net_one_mat` 原本只吃一个对称 `rho` 和一个 `e`。这里用 `SplitRho` 把三套取数
//! 矩阵和两套能量喂进去；不拆分时传 None，行为与原来逐位一致。
//!
//! ## 注意
//! 方向未知的成交（flag 既非 66 也非 83）不进四个分量，所以"全部"在这里等于 P+M。
//! A 股 L2 逐笔的 flag 只有 66/83（撤单 32 已在读取时过滤）。

use std::io;

use pyo3::prelude::*;
use rayon::prelude::*;

use super::check::check_net;
use super::net::{net_one_mat, SplitRho};
use super::reduce::{matset_from_net, produce_one_mat, Out, SoftCfg};
use super::split_mat::{compute_split_mats, PmWeight, MM, MP, PM, PP};
use super::topk::{finalize_total, topk_one_mat};

/// 方向集合：全部 / 主买 / 主卖。
pub const D_ALL: usize = 0;
pub const D_P: usize = 1;
pub const D_M: usize = 2;

/// 第三只的三种方向（顺序即输出顺序）。
pub const X_DIRS: [usize; 3] = [D_ALL, D_P, D_M];

/// 五种配对组合 (B 的方向, C 的方向, 标签)。
pub const BC_COMBOS: [(usize, usize, &str); 5] = [
    (D_ALL, D_ALL, "AA"),
    (D_P, D_P, "pp"),
    (D_P, D_M, "pm"),
    (D_M, D_P, "mp"),
    (D_M, D_M, "mm"),
];

pub fn dir_tag(d: usize) -> &'static str {
    match d {
        D_P => "p",
        D_M => "m",
        _ => "a",
    }
}

/// 方向集合 d 是否包含某一侧（side: 0=主买, 1=主卖）。
#[inline]
fn has(d: usize, side: usize) -> bool {
    match d {
        D_P => side == 0,
        D_M => side == 1,
        _ => true,
    }
}

/// X_ab[i][j] = Σ_{p∈a, q∈b} S_pq[i][j]。
#[inline]
fn x_ab(m: &[Vec<f32>; 4], n: usize, a: usize, b: usize, i: usize, j: usize) -> f64 {
    let k = i * n + j;
    let (ap, am) = (has(a, 0), has(a, 1));
    let (bp, bm) = (has(b, 0), has(b, 1));
    let mut v = 0.0f64;
    if ap && bp {
        v += m[PP][k] as f64;
    }
    if ap && bm {
        v += m[PM][k] as f64;
    }
    if am && bp {
        v += m[MP][k] as f64;
    }
    if am && bm {
        v += m[MM][k] as f64;
    }
    v
}

/// 方向 a 的自能量 e_i^a = 2·X_aa[i][i]。
pub fn energy(m: &[Vec<f32>; 4], n: usize, a: usize) -> Vec<f64> {
    (0..n).map(|i| 2.0 * x_ab(m, n, a, a, i, i)).collect()
}

/// 方向组合 (a, b) 的相关矩阵 R(a,b)[i][j]，**不镜像**（a≠b 时本来就不对称）。
/// 行 i 取方向 a、列 j 取方向 b；e_i^a、e_j^b 为两侧自能量。
pub fn rho_ab(m: &[Vec<f32>; 4], n: usize, a: usize, b: usize) -> Vec<f32> {
    let ea = energy(m, n, a);
    let eb = energy(m, n, b);
    let mut rho = vec![0.0f32; n * n];
    rho.par_chunks_mut(n).enumerate().for_each(|(i, row)| {
        row[i] = 1.0;
        let si = ea[i].sqrt();
        if !(si > 0.0) {
            return;
        }
        for j in 0..n {
            if i == j {
                continue;
            }
            let sj = eb[j].sqrt();
            if !(sj > 0.0) {
                continue;
            }
            let num = x_ab(m, n, a, b, i, j) + x_ab(m, n, b, a, j, i);
            row[j] = (num / (si * sj)) as f32;
        }
    });
    rho
}

/// 由 (e, rho) 反造一份伪 S_dir，使 `net_one_mat` 内部算出的 e、rho 与目标一致:
///   S_dir[i][i] = e_i/2；S_dir[i][j] = S_dir[j][i] = rho_ij·√(e_i e_j)/2。
/// `rho` 必须是镜像对称的（net 内部会再镜像一次，不影响结果）。
fn fake_sdir(e: &[f64], rho: &[f32], n: usize) -> Vec<f32> {
    let mut s = vec![0.0f32; n * n];
    for i in 0..n {
        s[i * n + i] = (e[i] / 2.0) as f32;
    }
    for i in 0..n {
        for j in (i + 1)..n {
            let v = (rho[i * n + j] as f64 * (e[i] * e[j]).sqrt() / 2.0) as f32;
            s[i * n + j] = v;
            s[j * n + i] = v;
        }
    }
    s
}

/// 单日全市场：5 种配对组合 × 3 种第三只方向，共 15 条因子链。
///
/// * `tau`    —— 衰减时间常数（秒）
/// * `weight` —— "cnt"（笔数）或 "vol"（√成交量）
/// * `combos` —— 只跑这些标签（如 ["AAa","ppp","pmp"]）；空 = 全部 15 条
/// * `ks`     —— topk 的 K 档
#[allow(clippy::too_many_arguments)]
pub(crate) fn compute_split_full(
    date: i64,
    tau: f64,
    weight: PmWeight,
    combos: &[String],
    ks: &[usize],
    sig: SoftCfg,
    null_mc: usize,
    exz_bins: usize,
) -> io::Result<(Vec<String>, Vec<String>, Vec<f32>)> {
    crate::yupei_dist::compute::with_compute_pool(|| {
        compute_split_inner(date, tau, weight, combos, ks, sig, null_mc, exz_bins)
    })?
}

#[allow(clippy::too_many_arguments)]
fn compute_split_inner(
    date: i64,
    tau: f64,
    weight: PmWeight,
    combos: &[String],
    ks: &[usize],
    sig: SoftCfg,
    null_mc: usize,
    exz_bins: usize,
) -> io::Result<(Vec<String>, Vec<String>, Vec<f32>)> {
    let t_all = std::time::Instant::now();
    super::reduce::set_soft(sig);

    // ---- 1. 上游：逐笔预处理（含方向拆分场）+ 行业；need=0 不算 28 张矩阵 ----
    let (codes, n, stocks, stats, _hm, industry) =
        crate::yupei_dist::compute::compute_yupei_dist_partial_need(date, 0)?;

    // ---- 2. 四分量矩阵 ----
    let m4 = compute_split_mats(&stocks, tau as f32, weight);
    eprintln!(
        "[yupei_split] date={date} n={n} tau={tau} weight={:?} 四分量完成 {:.1}s",
        weight,
        t_all.elapsed().as_secs_f32()
    );

    // ---- 3. 活跃度分层（空模型用）+ 协变量 ----
    let ntrades: Vec<f64> = stats.iter().map(|s| s.n_trades).collect();
    let mut sorted_nt = ntrades.clone();
    sorted_nt.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let cuts: Vec<f64> = (1..10)
        .map(|k| sorted_nt[(k * n / 10).min(n - 1)])
        .collect();
    let dec: Vec<usize> = ntrades.iter().map(|&v| super::net::decile_of(v, &cuts)).collect();
    let lnt: Vec<f64> = ntrades
        .iter()
        .map(|&v| if v.is_nan() { f64::NAN } else { v.max(1.0).ln() })
        .collect();
    let has_lnt = lnt.iter().any(|x| x.is_finite());
    let industry_i32: Vec<i32> = industry.iter().map(|&v| v as i32).collect();
    let industry_opt = Some(industry_i32.as_slice());
    let lnt_opt = Some(lnt.as_slice());

    // ---- 4. 15 条链（配对网络之间互相独立，并行跑）----
    type ChainOut = io::Result<(String, Vec<String>, Vec<Vec<f32>>)>;
    let mut jobs: Vec<(usize, usize, usize, String)> = Vec::new();
    for &(a, b, tag) in BC_COMBOS.iter() {
        for &c_dir in X_DIRS.iter() {
            let short = format!("{tag}{}", dir_tag(c_dir));
            let label = format!("sp{short}");
            // combos 里写 "AAa" 或 "spAAa" 都认
            if !combos.is_empty() && !combos.iter().any(|x| *x == short || *x == label) {
                continue;
            }
            jobs.push((a, b, c_dir, label));
        }
    }
    if jobs.is_empty() {
        return Err(io::Error::new(io::ErrorKind::InvalidInput, "combos 过滤后没有可跑的链"));
    }
    eprintln!("[yupei_split] 准备跑 {} 条链", jobs.len());

    let results: Vec<ChainOut> = jobs
        .par_iter()
        .map(|(a, b, c_dir, label)| {
            let (a, b, c_dir) = (*a, *b, *c_dir);
            // 三套取数矩阵 + 两套自能量
            let rbx = rho_ab(&m4, n, a, c_dir); // 行 B(a) 对列 X(c)
            let rcx = rho_ab(&m4, n, b, c_dir); // 行 C(b) 对列 X(c)
            let rbc = rho_ab(&m4, n, a, b); // 行 B(a) 对列 C(b)
            let e_a = energy(&m4, n, a);
            let e_b = energy(&m4, n, b);

            // 伪 S_dir 用 (a,b) 组合（给 net 内部的 rho 与 reduce 用）
            let mut rho_sym = rbc.clone();
            for i in 0..n {
                for j in 0..i {
                    rho_sym[i * n + j] = rho_sym[j * n + i];
                }
            }
            let sdir = fake_sdir(&e_a, &rho_sym, n);

            let mask: Vec<bool> = e_a.iter().map(|&v| v > 0.0).collect();
            let stock_idx: Vec<usize> = (0..n).filter(|&i| mask[i]).collect();
            let n_stocks = stock_idx.len();

            let sp = SplitRho {
                b_row: &rbx,
                c_row: &rcx,
                bc: &rbc,
            };
            let mut net = net_one_mat(
                &sdir,
                n,
                &dec,
                &mask,
                &stock_idx,
                null_mc,
                exz_bins,
                label,
                Some(ks),
                Some((&sp, &e_a, &e_b)),
            );
            let fails = check_net(&net, n, &mask, label);
            if fails > 0 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("check 失败 {fails} 项（chain={label}）"),
                ));
            }
            let tk_total = net.topk_total.take();
            let ms = matset_from_net(net, n);
            let mut out = Out::new();
            let pre = format!("{label}_");
            produce_one_mat(
                &ms, &pre, n, &mask, industry_opt, lnt_opt, has_lnt, &mut out,
            );
            let (tn, tc) = match tk_total {
                Some(t) => finalize_total(&t, n, &mask, ks, &pre),
                None => topk_one_mat(&ms.rho, &ms.e, n, &mask, ks, &pre),
            };
            let mut names = out.names;
            names.extend(tn);
            let mut cols: Vec<Vec<f32>> = out
                .cols
                .into_iter()
                .map(|c| c.into_iter().map(|v| v as f32).collect())
                .collect();
            cols.extend(tc);
            eprintln!("[yupei_split:{label}] {} 列（活跃 {n_stocks}/{n}）", names.len());
            Ok((label.clone(), names, cols))
        })
        .collect();

    // ---- 5. 按 jobs 顺序拼接 ----
    let mut names: Vec<String> = Vec::new();
    let mut cols: Vec<Vec<f32>> = Vec::new();
    for r in results {
        let (_lab, ns, cs) = r?;
        names.extend(ns);
        cols.extend(cs);
    }
    let f = names.len();
    if f == 0 {
        return Err(io::Error::new(io::ErrorKind::Other, "无因子产出"));
    }
    let mut vals = vec![f32::NAN; n * f];
    vals.par_chunks_mut(f).enumerate().for_each(|(i, row)| {
        for (j, col) in cols.iter().enumerate() {
            row[j] = col[i];
        }
    });
    eprintln!(
        "[yupei_split] 完成：{} 股票 × {} 因子，总 {:.1}s",
        n,
        f,
        t_all.elapsed().as_secs_f32()
    );
    Ok((codes, names, vals))
}

/// Python 入口：单日全市场主买/主卖方向拆分因子。
#[pyfunction]
#[pyo3(signature = (date, tau=1.0, weight="cnt", combos=None, ks=None, sig_temp=0.3, sig_thr=2.0, null_mc=20000, exz_bins=10))]
pub fn py_yupei_pair_split(
    date: i64,
    tau: f64,
    weight: &str,
    combos: Option<Vec<String>>,
    ks: Option<Vec<usize>>,
    sig_temp: f64,
    sig_thr: f64,
    null_mc: usize,
    exz_bins: usize,
) -> PyResult<(Vec<String>, Vec<String>, Vec<f32>)> {
    let w = match weight {
        "cnt" => PmWeight::Cnt,
        "vol" => PmWeight::Vol,
        other => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "weight 只支持 cnt / vol，收到 {other}"
            )))
        }
    };
    let ks = ks.unwrap_or_else(|| vec![5, 20, 100]);
    compute_split_full(
        date,
        tau,
        w,
        &combos.unwrap_or_default(),
        &ks,
        SoftCfg {
            temp: sig_temp,
            thr: sig_thr,
        },
        null_mc,
        exz_bins,
    )
    .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("{e:?}")))
}
