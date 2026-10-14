//! 玉佩2 双层网络（配对互补）—— 内存管线总控。
//!
//! 一个函数跑完「L2 逐笔 → 37 张有向交互矩阵 → 配对网络（net）→ 硬校验（check）
//! → 因子合成（reduce + topk）」全流程，等价于 sandbox_yupei_pair 的
//! `net → check → reduce → topk` 四步 CLI，但全部在内存中流动、不落任何中间文件。
//!
//! 数据流（单日、全市场）：
//!   1. `compute_yupei_dist_partial(date)` —— cross-section 标准口径枚举股票
//!      （symbol_map ∩ 当日 L2 文件），每股 10ms 桶 u[9] 权重场 + 37 张 n×n 有向矩阵
//!   2. 每张指定矩阵独立一条链：sdir → net_one_mat（rho/X0/G/Z/EXZ/…22 张）
//!      → check_net（硬校验）→ matset_from_net → produce_one_mat（reduce 因子块）
//!      + topk_one_mat（K 档因子块）
//!   3. 各矩阵的因子块按 `mats` 参数顺序拼接（确定性），输出行主序 N×F
//!
//! 股票覆盖：universe = symbol_map ∩ 当日 L2 文件（≈5100 只，与 cross-section
//! 完全一致，不再读 codes.txt、不做成交筛选）。读不到逐笔或 0 笔成交的股票
//! 保留在 universe 中（矩阵行/列全 0 → 配对中自然被掩掉，因子输出 NaN，
//! 但**行在**——满足"覆盖全部股票"）。
//!
//! 用法（Python）：
//!   codes, names, vals = rp.py_yupei_pair(20241231, ["cnt_t1","vol_t5"])
//!   # vals 行主序 len(codes) × len(names)，reshape 成 DataFrame 即可

pub(crate) mod check;
pub(crate) mod net;
pub(crate) mod reduce;
pub(crate) mod reduce2;
pub(crate) mod simd;
pub(crate) mod topk;

use std::io;

use pyo3::prelude::*;
use rayon::prelude::*;

use reduce::{matset_from_net, produce_one_mat, set_soft, Out, SoftCfg};
use topk::topk_one_mat;

/// topk 的默认 K 档（与 sandbox 一致：九档）。
const KS_DEFAULT: [usize; 9] = [1, 2, 5, 10, 20, 50, 100, 200, 500];

/// 默认跑的矩阵集合：cnt 全九档 τ + vol 六档 τ（C 组口径）。
/// MATRIX_SPECS 里还有 logvol/flow/urg/ext 族，需要时显式传 mats 即可。
pub const DEFAULT_MATS: [&str; 15] = [
    "cnt_t005", "cnt_t01", "cnt_t02", "cnt_t05", "cnt_t1", "cnt_t3",
    "cnt_t5", "cnt_t10", "cnt_t30", "vol_t02", "vol_t05", "vol_t1",
    "vol_t3", "vol_t5", "vol_t30",
];

/// 单日全市场配对互补因子（内存管线版）。
///
/// * `mats` —— 要跑的矩阵名（必须是 `compute_matrices` 产出的 37 个名字里的）
/// * `sig`  —— sigmoid 软化参数（E 组参数刷选用，默认 (0.3, 2.0)）
/// * `null_mc` —— 空模型蒙特卡洛抽样上限（默认 20000）
/// * `exz_bins` —— (p,q) 条件网格边数（默认 10）
///
/// 返回 `(codes, names, vals)`：codes 是当日全部股票（含无成交空股），
/// vals 行主序 len(codes) × len(names)；无数据股票整行 NaN 但行保留。
pub(crate) fn compute_yupei_pair_full(
    date: i64,
    mats: &[String],
    sig: SoftCfg,
    null_mc: usize,
    exz_bins: usize,
) -> io::Result<(Vec<String>, Vec<String>, Vec<f32>)> {
    crate::yupei_dist::compute::with_compute_pool(|| {
        compute_yupei_pair_inner(date, mats, sig, null_mc, exz_bins)
    })?
}

fn compute_yupei_pair_inner(
    date: i64,
    mats: &[String],
    sig: SoftCfg,
    null_mc: usize,
    exz_bins: usize,
) -> io::Result<(Vec<String>, Vec<String>, Vec<f32>)> {
    let t_all = std::time::Instant::now();
    set_soft(sig);

    // ---- 1. 上游：cross-section 口径枚举 + 逐笔预处理 + 37 张矩阵 ----
    // 只算请求的矩阵用到的权重家族。默认 15 张只要 cnt+vol，其余 22 张不参与计算。
    let need = {
        let m = crate::yupei_dist::matrix_stage::family_mask_of(mats);
        if m == 0 { crate::yupei_dist::matrix_stage::FAM_ALL } else { m }
    };
    let (codes, n, stats, hm, industry) =
        crate::yupei_dist::compute::compute_yupei_dist_partial_need(date, need)?;
    eprintln!("[yupei_pair] date={date} universe n={n}，矩阵 {} 张", hm.len());

    // 每股活跃度（成交笔数）→ 十分位分层（空模型用）+ ln(成交笔数)（R5 协变量）
    let ntrades: Vec<f64> = stats.iter().map(|s| s.n_trades).collect();
    let lnt: Vec<f64> = ntrades
        .iter()
        .map(|&v| if v.is_nan() { f64::NAN } else { v.max(1.0).ln() })
        .collect();
    let has_lnt = lnt.iter().any(|x| x.is_finite());
    let industry_i32: Vec<i32> = industry.iter().map(|&v| v as i32).collect();
    let industry_opt = Some(industry_i32.as_slice());
    let lnt_opt = Some(lnt.as_slice());

    let mut sorted_nt = ntrades.clone();
    sorted_nt.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let cuts: Vec<f64> = (1..10)
        .map(|k| sorted_nt[(k * n / 10).min(n - 1)])
        .collect();
    let dec: Vec<usize> = ntrades.iter().map(|&v| net::decile_of(v, &cuts)).collect();

    // ---- 2. 每张矩阵一条链：net → check → matset → reduce + topk ----
    // 矩阵之间完全独立、并行执行；每矩阵返回 (names, cols_f32)，按 mats 顺序合并。
    type MatOut = io::Result<(Vec<String>, Vec<Vec<f32>>)>;
    let results: Vec<MatOut> = mats
        .par_iter()
        .map(|mat| {
            let sdir = hm.get(mat.as_str()).ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("矩阵 {mat} 不在 MATRIX_SPECS 的 37 张里"),
                )
            })?;
            if sdir.len() != n * n {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("矩阵 {mat} 有 {} 个元素，期望 n*n={}", sdir.len(), n * n),
                ));
            }
            // 自能量 E>0 = 当天有成交的股票 = 配对参与集与候选 X 集
            let e: Vec<f64> = (0..n).map(|i| 2.0 * sdir[i * n + i] as f64).collect();
            let mask: Vec<bool> = e.iter().map(|&v| v > 0.0).collect();
            let stock_idx: Vec<usize> = (0..n).filter(|&i| mask[i]).collect();
            let n_stocks = stock_idx.len();

            // ---- net ----
            let net = net::net_one_mat(sdir, n, &dec, &mask, &stock_idx, null_mc, exz_bins, mat);

            // ---- check（硬校验，任一失败即中止该矩阵）----
            let fails = check::check_net(&net, n, &mask, mat);
            if fails > 0 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("check 失败 {fails} 项（mat={mat}）"),
                ));
            }

            // ---- reduce：legacy + reduce2 扩展块（列 f64 → f32）----
            let ms = matset_from_net(net, n);
            let mut out = Out::new();
            let pre = format!("{mat}_");
            produce_one_mat(
                &ms, &pre, n, &mask,
                industry_opt, lnt_opt, has_lnt,
                &mut out,
            );
            // ---- topk：K 档 × 双口径（rho/e 复用 ms 里的）----
            let (tn, tc) = topk_one_mat(&ms.rho, &ms.e, n, &mask, &KS_DEFAULT, &pre);

            // 合并 names/cols（reduce cols 是 f64，topk cols 是 f32，统一成 f32）
            let mut names = out.names;
            names.extend(tn);
            let mut cols: Vec<Vec<f32>> = out
                .cols
                .into_iter()
                .map(|c| c.into_iter().map(|v| v as f32).collect())
                .collect();
            cols.extend(tc);
            eprintln!(
                "[yupei_pair:{mat}] {} 因子块完成（活跃 {n_stocks}/{n}）",
                names.len()
            );
            Ok((names, cols))
        })
        .collect();

    // ---- 3. 按 mats 顺序拼接所有因子列 ----
    let mut names: Vec<String> = Vec::new();
    let mut cols: Vec<Vec<f32>> = Vec::new();
    for r in results {
        let (ns, cs) = r?;
        names.extend(ns);
        cols.extend(cs);
    }
    let f = names.len();
    if f == 0 {
        return Err(io::Error::new(io::ErrorKind::Other, "无因子产出"));
    }

    // ---- 4. 行主序 vals：n × f；universe 全保留（无数据股票整行 NaN 但行在）----
    let mut vals = vec![f32::NAN; n * f];
    vals.par_chunks_mut(f).enumerate().for_each(|(i, row)| {
        for (j, col) in cols.iter().enumerate() {
            row[j] = col[i];
        }
    });
    eprintln!(
        "[yupei_pair] 完成：{} 股票 × {} 因子，总 {:.1}s",
        n,
        f,
        t_all.elapsed().as_secs_f32()
    );
    Ok((codes, names, vals))
}

/// Python 入口：单日全市场配对互补因子。
///
/// `mats=None` 默认 DEFAULT_MATS 15 张（cnt×9τ + vol×6τ）。
/// 返回 `(codes, names, vals)`，vals 行主序 len(codes)×len(names)。
#[pyfunction]
#[pyo3(signature = (date, mats=None, sig_temp=0.3, sig_thr=2.0, null_mc=20000, exz_bins=10))]
pub fn py_yupei_pair(
    date: i64,
    mats: Option<Vec<String>>,
    sig_temp: f64,
    sig_thr: f64,
    null_mc: usize,
    exz_bins: usize,
) -> PyResult<(Vec<String>, Vec<String>, Vec<f32>)> {
    let mats = mats.unwrap_or_else(|| DEFAULT_MATS.iter().map(|s| s.to_string()).collect());
    compute_yupei_pair_full(
        date,
        &mats,
        SoftCfg { temp: sig_temp, thr: sig_thr },
        null_mc,
        exz_bins,
    )
    .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("{e:?}")))
}
