//! hm79「交流之苦」的 Rust 复现：盘口 10 档量差分 → 窗口协方差因子 → 特征 → 入选 base。
//!
//! 旧文件：/home/chenzongwei/pythoncode/成果hm79_交流之苦/hm79_okay.py
//! 形态：模块级 `go(date, code)`（逐股，数据来源是 **盘口快照 market_data**，不是逐笔）。
//!   `dwo.run_factor_rust_bin(go, base_factor_ver, names_save_unfold, n_jobs, ...)`
//!   run_factor_out 的注释那一段里 get_many=0 / read_full=0 / change_time=1，
//!   也就是一整天的盘口一张表，窗口由旧代码自己切。属于**逐股**系列。
//!
//! ════════════════════════════════════════════════════════════════════
//! 一、旧代码逐步对照（写代码前先抄在这里，方便复核）
//! ════════════════════════════════════════════════════════════════════
//! 1) market_data = rp.adjust_afternoon(rp.read_market(code, date))
//!                    .drop_duplicates(subset=['exchtime'])
//!    · rp.read_market 只读 exchtime / last_prc / volume / ask_prc1..10 / ask_vol1..10
//!      / bid_prc1..10 / bid_vol1..10，并丢掉 `ask_prc1 == 0 or bid_prc1 == 0` 的行
//!      （NaN 不等于 0，所以空值行是**保留**的，这里用同样的 `!= 0.0` 语义）。
//!    · rp.adjust_afternoon(only_inday=1)：按当日钟点切 [09:30:00, 11:30:00] 与
//!      [13:00:00, 14:57:00]（between_time，**微秒级闭区间**），下午整体前移 90 分钟，
//!      再 concat（上午全部在前、下午全部在后）。顺序有意义，见下面第 3 步。
//!    · drop_duplicates 保留第一次出现：下午 13:00:00 平移后正好等于上午 11:30:00，
//!      这些行会被丢掉。
//! 2) 对 windows ∈ {60, 10} 各算一次：
//!      s = f'_{windows}w'
//!      bid = market_data[bid_vol1..10].values.astype(float)
//!      ask = market_data[ask_vol1..10].values.astype(float)
//!      names, factors = rp.orderbook_volume_cov_factors(bid, ask, windows)
//!      DataFrame(factors, index=timestamps[windows-1:], columns=[f'{n}{s}' for n in names])
//!    其中 timestamps = market_data['exchtime'].values（就是上面那个行序）。
//!    factors 每行 23 列，行数 = N - windows + 1（N 是 market_data 的行数）。
//! 3) df = pd.concat([one(60), one(10)], axis=1)[cols_60w + cols_10w]
//!    · 两个 DataFrame 的下标是同一串 timestamps 的两个后缀，concat(axis=1) 取并集，
//!      实测并集是**升序**的 timestamps[9:]（已用小例子确认过 pandas 的行为）。
//!      于是 df 有 N-9 行：第 r 行对应 timestamps[9+r]；
//!      60w 的前 50 行是 NaN（因为 timestamps[59:] 从第 59 个开始）。
//!    · 列顺序 = COLS_60W_BASE（23 个）+ COLS_10W_BASE（16 个），共 39 列。
//!      两组的列名都直接取自 rp.orderbook_volume_cov_factors 的 23 个列名，
//!      所以 60w 块就是 23 个输出按序，10w 块是其中 16 个（下标见 COL10_IDX）。
//! 4) appdf = market_data.set_index('exchtime')[['volume', 'last_prc']].reindex(index=df.index)
//!    appdf.volume = appdf.volume.diff()            → 第一行 NaN
//! 5) res, _ = rp.get_features_factors(df, append_for_corr=appdf,
//!                                     with_lyapunov_exponent=False,
//!                                     with_complexity=False)
//!    res 的长度 = 18 组单列统计量 × 39 列 + corr 上三角 C(41, 2)
//!              = 702 + 820 = 1522；名字顺序见 python/rust_pyfunc/trading_data_utils.py
//!              的 get_features_factors。
//! 6) res = {name: value}; res = [res[name] for name in names_save_unfold]
//!    names_save_unfold = 旧 names_save 去掉尾部 `_fold`；共 78 项。
//!
//! ════════════════════════════════════════════════════════════════════
//! 二、名字与重名
//! ════════════════════════════════════════════════════════════════════
//! 旧 names_save 有 78 项，去掉 `_fold` 以后只剩 76 个不同的名字：
//!   · `volume_corr_last_prc`（第 19 项）与第 31 项 `volume_corr_last_prc_fold` 撞名
//!   · `b_diag_60w_iqr`（第 58 项）与第 17 项 `b_diag_60w_iqr_fold` 撞名
//! 两两取值完全相同（旧代码是同一个字典键查了两次），列式 store 不接受重名，
//! 所以按规范给**第二次及以后**出现的重名加 `_dup2` 后缀，值不变、列数仍是 78。
//!
//! ════════════════════════════════════════════════════════════════════
//! 三、实现要点
//! ════════════════════════════════════════════════════════════════════
//! · 盘口量一律按 f64 解析 CSV（不走 fast_csv_reader 的 f32）。旧代码里的
//!   `market_data[bid_cols].values` 是 float64，档位量本身是整数，f64/f32 都能精确表示，
//!   但 window 协方差里的 sum_xx = Σ v_i·v_j 会累到 1e9 量级，f32 只有 7 位有效数字，
//!   再进 Cholesky / 特征值就会被 log_det < -23 这类阈值翻面，所以统一 f64。
//! · rp.orderbook_volume_cov_factors 是 #[pyfunction]，worker 二进制不带 libpython，
//!   不能在链路里调（会链接失败且构建表面还成功）。这里把它的**纯 Rust 主体逐行抄一份**
//!   到本文件（cov_window_factors），算法、求和顺序、增量更新的顺序完全一致，因此
//!   与 .so 里的那条通路逐位相同。src/orderbook_volume_cov_factors.rs 一行没动。
//! · 特征走 f64 通路 crate::features_f64::get_features_factors_rust_full_ex，
//!   append_for_corr 传 (volume.diff(), last_prc)，开关取
//!   with_threshold_counts=true / with_curvature=false / with_max_min=false /
//!   with_lyapunov_exponent=false，与旧 go 的调用一一对应。
//!   （该入口与 Python get_features_factors 的一致性由
//!     lib/okay_database/verify_features_f64_switches.py 覆盖，2026-10-06 跑过：结论通过。）
//! · 因子下标不写死，改成运行时在 features 返回的名字里按名字查一次并缓存，
//!   避免手抄 1522 项下标出错。
//! · 注意 f64 特征通路固定会在**最后**再追加 lz_complexity / entropy_1d /
//!   max_range_product 三组（共 3×39=117 项），而旧 go 传的是 with_complexity=False。
//!   这三组排在 corr 块之后、且入选的 78 项全部落在它前面，所以按名字取下标不受影响；
//!   库里这条通路的输出总长因此是 1522+117=1639，而不是 1522。
//! · 已知差异：没有。四组 (日期, 股票) 对账全部「通过」。

use crate::data_paths;
use crate::features_f64::{self, FeatureOpts};
use ndarray::{Array2, ArrayView2};
use std::collections::HashSet;
use std::sync::OnceLock;

/// 盘口 CSV 的列索引（0 基，与 python/rust_pyfunc/trading_data_utils.py 读出来的列名一致）。
/// 2018/2023 的盘口文件末尾多一列 `num`（共 62 列），不影响前 61 列。
const MKT_COL_EXCHTIME: usize = 4;
const MKT_COL_LAST_PRC: usize = 6;
const MKT_COL_VOLUME: usize = 14;
const MKT_ASK_PRC_BASE: usize = 21; // ask_prc1=21, ask_vol1=22, bid_prc1=23, bid_vol1=24, ...
const MKT_NCOL: usize = 61;

const DAY_US: i64 = 86_400_000_000;
const SEC_US: i64 = 1_000_000;
const AM_START_US: i64 = (9 * 3600 + 30 * 60) * SEC_US;
const AM_END_US: i64 = (11 * 3600 + 30 * 60) * SEC_US;
const PM_START_US: i64 = 13 * 3600 * SEC_US;
const PM_END_US: i64 = (14 * 3600 + 57 * 60) * SEC_US;
const PM_SHIFT_US: i64 = 90 * 60 * SEC_US;

/// 两个窗口。
const WIN60: usize = 60;
const WIN10: usize = 10;
/// 每个窗口的协方差输出列数（rp.orderbook_volume_cov_factors 固定 23 个）。
const COV_NCOL: usize = 23;
/// 合并后的列数：23 个 60w + 16 个 10w。
const N_DATA_COL: usize = 39;

/// COLS_10W_BASE 那 16 个列名在 23 个协方差输出里的下标。
const COL10_IDX: [usize; 16] = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 19, 20, 21, 22];

/// df 的 39 个列名，顺序 = COLS_60W_BASE（各加 `_60w`）+ COLS_10W_BASE（各加 `_10w`）。
/// 这些名字会进 get_features_factors 生成的因子名，必须与旧文件逐字一致，
/// 否则按名字取 78 项入选因子时一个都查不到（2026-10-06 踩过一次）。
const DATA_COLS: [&str; N_DATA_COL] = [
    "ba_spl_60w",
    "oe_spl_60w",
    "hl_spl_60w",
    "bla_cpl_60w",
    "bha_cpl_60w",
    "adj_cpl_60w",
    "diag_spl_60w",
    "b_oe_60w",
    "b_hl_60w",
    "b_adj_60w",
    "b_diag_60w",
    "a_oe_60w",
    "a_hl_60w",
    "a_adj_60w",
    "a_diag_60w",
    "oe_bma_60w",
    "oe_dabs_60w",
    "hl_bma_60w",
    "hl_dabs_60w",
    "adj_bma_60w",
    "adj_dabs_60w",
    "diag_bma_60w",
    "diag_dabs_60w",
    "bla_cpl_10w",
    "bha_cpl_10w",
    "adj_cpl_10w",
    "diag_spl_10w",
    "b_oe_10w",
    "b_hl_10w",
    "b_adj_10w",
    "b_diag_10w",
    "a_oe_10w",
    "a_hl_10w",
    "a_adj_10w",
    "a_diag_10w",
    "adj_bma_10w",
    "adj_dabs_10w",
    "diag_bma_10w",
    "diag_dabs_10w",
];

/// 旧 names_save 去掉 `_fold` 之后的 78 项，顺序就是旧 go 的返回顺序。
/// 注意里面有 2 组重名（见文件头第二节），names() 会给第二次出现加 `_dup2`。
const SRC: [&str; 78] = [
    "diag_bma_60w_median",
    "adj_bma_60w_period_diff",
    "bha_cpl_10w_p95",
    "hl_bma_60w_trend",
    "oe_bma_60w_corr_bla_cpl_10w",
    "diag_dabs_10w_median",
    "b_adj_60w_iqr",
    "b_adj_60w_median",
    "hl_dabs_60w_corr_adj_bma_60w",
    "a_diag_60w_period_ratio",
    "diag_spl_60w_period_ratio",
    "hl_bma_60w_p5",
    "b_oe_60w_corr_last_prc",
    "b_hl_60w_corr_a_diag_10w",
    "hl_dabs_60w_p25",
    "b_adj_10w_p75",
    "b_diag_60w_iqr",
    "a_hl_10w_trend",
    "volume_corr_last_prc",
    "adj_bma_10w_p75",
    "adj_cpl_10w_corr_a_adj_10w",
    "adj_bma_60w_mean_above_p90",
    "a_diag_60w_period_diff",
    "adj_cpl_10w_corr_volume",
    "oe_dabs_60w_mean_above_p90",
    "adj_cpl_60w_kurt",
    "adj_cpl_60w_period_diff",
    "adj_cpl_60w_p25",
    "b_oe_60w_corr_a_diag_10w",
    "b_oe_60w_mean_above_p90",
    "volume_corr_last_prc",
    "bla_cpl_10w_p95",
    "b_diag_10w_p5",
    "bha_cpl_10w_corr_last_prc",
    "hl_bma_60w_median",
    "b_diag_10w_corr_volume",
    "b_adj_10w_cv",
    "diag_bma_10w_period_ratio",
    "bla_cpl_10w_period_ratio",
    "oe_dabs_60w_std",
    "b_hl_60w_kurt",
    "b_adj_60w_period_diff",
    "diag_bma_60w_corr_bla_cpl_10w",
    "diag_bma_60w_corr_volume",
    "oe_bma_60w_period_diff",
    "adj_bma_60w_mean",
    "hl_spl_60w_corr_a_hl_60w",
    "adj_cpl_10w_iqr",
    "a_diag_10w_corr_diag_dabs_10w",
    "bla_cpl_10w_std",
    "b_diag_60w_period_diff",
    "diag_spl_10w_p75",
    "oe_bma_60w_p75",
    "adj_cpl_60w_p5",
    "adj_bma_10w_cv",
    "a_adj_60w_corr_diag_dabs_60w",
    "diag_spl_60w_period_diff",
    "b_diag_60w_iqr",
    "b_oe_60w_period_ratio",
    "oe_bma_60w_p25",
    "diag_bma_10w_p75",
    "b_oe_60w_corr_adj_bma_60w",
    "b_diag_10w_median",
    "oe_spl_60w_corr_a_oe_60w",
    "oe_spl_60w_corr_b_adj_60w",
    "bha_cpl_60w_autocorr1",
    "b_diag_10w_kurt",
    "diag_bma_60w_p95",
    "adj_bma_10w_period_ratio",
    "bha_cpl_10w_autocorr1",
    "diag_bma_10w_mean",
    "diag_bma_60w_p75",
    "bha_cpl_60w_period_ratio",
    "b_diag_60w_cv",
    "adj_bma_60w_corr_volume",
    "adj_bma_10w_corr_diag_dabs_10w",
    "b_adj_10w_p5",
    "diag_spl_60w_iqr",
];

/// 因子名：第二次及以后出现的重名加 `_dup2`（见文件头第二节）。
pub fn names() -> Vec<String> {
    let mut seen: HashSet<&str> = HashSet::new();
    let mut out = Vec::with_capacity(SRC.len());
    for n in SRC.iter() {
        if seen.insert(n) {
            out.push((*n).to_string());
        } else {
            out.push(format!("{}_dup2", n));
        }
    }
    out
}

// ============================================================================
// 盘口 CSV 的 f64 读取
// ============================================================================

/// 一行盘口快照里计算要用的列。
#[derive(Clone, Copy, Default)]
struct MktRow {
    /// exchtime（微秒）+ 8 小时，即 pandas 那边的交易所本地时间。
    time_us: i64,
    ask_prc1: f64,
    bid_prc1: f64,
    last_prc: f64,
    volume: f64,
    ask_vol: [f64; 10],
    bid_vol: [f64; 10],
}

/// 按整数前缀解析（exchtime 在 CSV 里是 "1740960939000000.0" 这种浮点串）。
fn parse_i64_at(b: &[u8]) -> i64 {
    let mut i = 0usize;
    let mut neg = false;
    if i < b.len() && (b[i] == b'-' || b[i] == b'+') {
        neg = b[i] == b'-';
        i += 1;
    }
    let mut v: i64 = 0;
    while i < b.len() && b[i].is_ascii_digit() {
        v = v * 10 + (b[i] - b'0') as i64;
        i += 1;
    }
    if neg {
        -v
    } else {
        v
    }
}

/// 与 pandas 的浮点读取等价的解析：空串给 NaN，其余交给正确舍入的 str::parse。
/// 档位量是整数、价格两位小数，都在 17 位有效数字以内，不存在 precise_xstrtod 的偏差。
fn parse_f64_at(b: &[u8]) -> f64 {
    std::str::from_utf8(b)
        .ok()
        .and_then(|s| s.trim().parse::<f64>().ok())
        .unwrap_or(f64::NAN)
}

/// 读 market_data CSV（f64）。行集与 pandas 读整张表一致（不过滤 status、不过滤涨跌停）。
fn read_market_rows(code: &str, date: i64) -> Option<Vec<MktRow>> {
    let filename = format!("{}_{}_market_data.csv", code, date);
    let mut path = None;
    for root in data_paths::level2_roots() {
        let p = root.join(date.to_string()).join("market_data").join(&filename);
        if p.exists() {
            path = Some(p);
            break;
        }
    }
    let path = path?;
    let content = std::fs::read_to_string(&path).ok()?;

    let mut out: Vec<MktRow> = Vec::with_capacity(content.len() / 220 + 1);
    let mut lines = content.lines();
    lines.next()?; // 表头
    for line in lines {
        if line.is_empty() {
            continue;
        }
        let bytes = line.as_bytes();
        let mut f: [&[u8]; MKT_NCOL] = [&b""[..]; MKT_NCOL];
        let mut col = 0usize;
        let mut start = 0usize;
        for (i, &b) in bytes.iter().enumerate() {
            if b == b',' {
                if col < MKT_NCOL {
                    f[col] = &bytes[start..i];
                }
                col += 1;
                start = i + 1;
            }
        }
        if col < MKT_NCOL {
            f[col] = &bytes[start..];
            col += 1;
        }
        if col < MKT_NCOL {
            // 列数不够的行 pandas 会直接报错；真实文件不会出现，这里跳过保护一下。
            continue;
        }

        let mut row = MktRow {
            time_us: parse_i64_at(f[MKT_COL_EXCHTIME]) + 8 * 3600 * SEC_US,
            ..Default::default()
        };
        row.last_prc = parse_f64_at(f[MKT_COL_LAST_PRC]);
        row.volume = parse_f64_at(f[MKT_COL_VOLUME]);
        row.ask_prc1 = parse_f64_at(f[MKT_ASK_PRC_BASE]);
        row.bid_prc1 = parse_f64_at(f[MKT_ASK_PRC_BASE + 2]);
        for k in 0..10 {
            let base = MKT_ASK_PRC_BASE + k * 4;
            row.ask_vol[k] = parse_f64_at(f[base + 1]);
            row.bid_vol[k] = parse_f64_at(f[base + 3]);
        }
        out.push(row);
    }
    Some(out)
}

// ============================================================================
// rp.orderbook_volume_cov_factors 的纯 Rust 主体（逐行抄自
// src/orderbook_volume_cov_factors.rs，只去掉 pyo3 外壳）
// ============================================================================

use nalgebra::DMatrix;

fn extract_submatrix(mat: &DMatrix<f64>, rows: &[usize], cols: &[usize]) -> DMatrix<f64> {
    let nr = rows.len();
    let nc = cols.len();
    let mut data = Vec::with_capacity(nr * nc);
    for &c in cols {
        for &r in rows {
            data.push(mat[(r, c)]);
        }
    }
    DMatrix::from_vec(nr, nc, data)
}

fn log_det_cholesky(mat: &DMatrix<f64>) -> Option<f64> {
    let chol = mat.clone().cholesky()?;
    let log_det: f64 = 2.0 * chol.l().diagonal().iter().map(|x| x.ln()).sum::<f64>();
    if log_det.is_finite() {
        Some(log_det)
    } else {
        None
    }
}

fn schur_factor(cov: &DMatrix<f64>, x_idx: &[usize], y_idx: &[usize]) -> Option<f64> {
    let sigma_xx = extract_submatrix(cov, x_idx, x_idx);
    let sigma_yy = extract_submatrix(cov, y_idx, y_idx);
    let sigma_xy = extract_submatrix(cov, x_idx, y_idx);

    let log_det_y = log_det_cholesky(&sigma_yy)?;
    if log_det_y < (-23.0) {
        return None;
    }

    let xx_chol = sigma_xx.clone().cholesky()?;
    let log_det_xx: f64 = 2.0 * xx_chol.l().diagonal().iter().map(|x| x.ln()).sum::<f64>();
    if !log_det_xx.is_finite() || log_det_xx < (-23.0) {
        return None;
    }

    let solved = xx_chol.solve(&sigma_xy);
    let schur = &sigma_yy - &sigma_xy.transpose() * &solved;

    let log_det_schur = log_det_cholesky(&schur)?;
    if log_det_schur < (-23.0) {
        return None;
    }

    let _ = log_det_xx;
    Some(log_det_y - log_det_schur)
}

fn log_sum_eigenvalues(eigenvalues: &[f64]) -> f64 {
    eigenvalues
        .iter()
        .filter(|&&x| x > 1e-10)
        .map(|x| x.ln())
        .sum()
}

fn extract_eigenvalues(mat: &DMatrix<f64>) -> Vec<f64> {
    mat.clone()
        .symmetric_eigen()
        .eigenvalues
        .iter()
        .copied()
        .collect()
}

fn eigval_diff_precomputed(eig_full: &[f64], cov_split: &DMatrix<f64>) -> Option<f64> {
    let sum_full = log_sum_eigenvalues(eig_full);
    let eig_split = extract_eigenvalues(cov_split);
    let sum_split = log_sum_eigenvalues(&eig_split);

    if !sum_full.is_finite() || !sum_split.is_finite() {
        return None;
    }
    Some(sum_split - sum_full)
}

fn build_adjacent_mask(d: usize, cov: &DMatrix<f64>) -> DMatrix<f64> {
    let mut m = DMatrix::zeros(d, d);
    for i in 0..d {
        m[(i, i)] = cov[(i, i)];
        if i + 1 < d {
            m[(i, i + 1)] = cov[(i, i + 1)];
            m[(i + 1, i)] = cov[(i + 1, i)];
        }
    }
    m
}

fn build_anti_diag_mask(n_half: usize, d: usize, cov: &DMatrix<f64>) -> DMatrix<f64> {
    let mut m = cov.clone();
    for j in 0..n_half {
        let k = d - 1 - j;
        m[(j, k)] = 0.0;
        m[(k, j)] = 0.0;
    }
    m
}

fn zero_cross_blocks(mat: &mut DMatrix<f64>, rows: &[usize], cols: &[usize]) {
    for &i in rows {
        for &j in cols {
            mat[(i, j)] = 0.0;
            mat[(j, i)] = 0.0;
        }
    }
}

/// 从 20x20 协方差矩阵算 15 个基础因子（抄自源文件的同名函数）。
fn compute_all_factors(cov20: &DMatrix<f64>) -> [Option<f64>; 15] {
    let mut results = [None; 15];

    let eig20_vals = extract_eigenvalues(cov20);

    results[0] = schur_factor(
        cov20,
        &(0..10).collect::<Vec<_>>(),
        &(10..20).collect::<Vec<_>>(),
    );
    results[1] = schur_factor(
        cov20,
        &(0..20).step_by(2).collect::<Vec<_>>(),
        &(1..20).step_by(2).collect::<Vec<_>>(),
    );
    results[2] = schur_factor(
        cov20,
        &(0..5).chain(10..15).collect::<Vec<_>>(),
        &(5..10).chain(15..20).collect::<Vec<_>>(),
    );

    let bid_low: Vec<usize> = (0..5).collect();
    let ask_high: Vec<usize> = (15..20).collect();
    let bid_high: Vec<usize> = (5..10).collect();
    let ask_low: Vec<usize> = (10..15).collect();

    {
        let mut cov_split = cov20.clone();
        zero_cross_blocks(&mut cov_split, &bid_high, &ask_low);
        zero_cross_blocks(&mut cov_split, &bid_low, &ask_low);
        zero_cross_blocks(&mut cov_split, &bid_high, &ask_high);
        results[3] = eigval_diff_precomputed(&eig20_vals, &cov_split);
    }
    {
        let mut cov_split = cov20.clone();
        zero_cross_blocks(&mut cov_split, &bid_low, &ask_high);
        zero_cross_blocks(&mut cov_split, &bid_low, &ask_low);
        zero_cross_blocks(&mut cov_split, &bid_high, &ask_high);
        results[4] = eigval_diff_precomputed(&eig20_vals, &cov_split);
    }
    {
        let cov_split = build_adjacent_mask(20, cov20);
        results[5] = eigval_diff_precomputed(&eig20_vals, &cov_split);
    }
    {
        let cov_split = build_anti_diag_mask(10, 20, cov20);
        results[6] = eigval_diff_precomputed(&eig20_vals, &cov_split);
    }

    let bid_rows: Vec<usize> = (0..10).collect();
    let cov_bid = extract_submatrix(cov20, &bid_rows, &bid_rows);
    let ask_rows: Vec<usize> = (10..20).collect();
    let cov_ask = extract_submatrix(cov20, &ask_rows, &ask_rows);

    let eig_bid_vals = extract_eigenvalues(&cov_bid);
    let eig_ask_vals = extract_eigenvalues(&cov_ask);

    results[7] = schur_factor(
        &cov_bid,
        &(0..10).step_by(2).collect::<Vec<_>>(),
        &(1..10).step_by(2).collect::<Vec<_>>(),
    );
    results[8] = schur_factor(
        &cov_bid,
        &(0..5).collect::<Vec<_>>(),
        &(5..10).collect::<Vec<_>>(),
    );
    {
        let cov_split = build_adjacent_mask(10, &cov_bid);
        results[9] = eigval_diff_precomputed(&eig_bid_vals, &cov_split);
    }
    {
        let cov_split = build_anti_diag_mask(5, 10, &cov_bid);
        results[10] = eigval_diff_precomputed(&eig_bid_vals, &cov_split);
    }

    results[11] = schur_factor(
        &cov_ask,
        &(0..10).step_by(2).collect::<Vec<_>>(),
        &(1..10).step_by(2).collect::<Vec<_>>(),
    );
    results[12] = schur_factor(
        &cov_ask,
        &(0..5).collect::<Vec<_>>(),
        &(5..10).collect::<Vec<_>>(),
    );
    {
        let cov_split = build_adjacent_mask(10, &cov_ask);
        results[13] = eigval_diff_precomputed(&eig_ask_vals, &cov_split);
    }
    {
        let cov_split = build_anti_diag_mask(5, 10, &cov_ask);
        results[14] = eigval_diff_precomputed(&eig_ask_vals, &cov_split);
    }

    results
}

/// 盘口量协方差因子的窗口主体，输出 row-major 的 (n - window + 1) × 23。
/// 列序与 rp.orderbook_volume_cov_factors 的 get_column_names() 一致。
fn cov_window_factors(bid: ArrayView2<f64>, ask: ArrayView2<f64>, window_size: usize) -> Vec<f64> {
    let n = bid.nrows();
    let d = 20;
    let n_windows = if n >= window_size { n - window_size + 1 } else { 0 };
    if n_windows == 0 {
        return Vec::new();
    }

    let mut diff_data = vec![0.0f64; n * d];
    for i in 1..n {
        for j in 0..10 {
            diff_data[i * d + j] = bid[(i, j)] - bid[(i - 1, j)];
            diff_data[i * d + 10 + j] = ask[(i, j)] - ask[(i - 1, j)];
        }
    }

    let mut result_data = vec![f64::NAN; n_windows * COV_NCOL];

    let mut sum_x = vec![0.0f64; d];
    let mut sum_xx = vec![0.0f64; d * d];
    for t in 0..window_size {
        for i in 0..d {
            let vi = diff_data[t * d + i];
            sum_x[i] += vi;
            for j in 0..=i {
                let vj = diff_data[t * d + j];
                sum_xx[i * d + j] += vi * vj;
            }
        }
    }

    let ws_f = window_size as f64;
    let ddof = (window_size - 1) as f64;
    let mut cov_colmaj = vec![0.0f64; d * d];

    for w in 0..n_windows {
        let mut has_nan_inf = false;
        for j in 0..d {
            let sj = sum_x[j];
            for i in 0..d {
                let si = sum_x[i];
                let (lo, hi) = if i >= j { (j, i) } else { (i, j) };
                let sxx = sum_xx[hi * d + lo];
                let cov_val = (sxx - si * sj / ws_f) / ddof;
                if !cov_val.is_finite() {
                    has_nan_inf = true;
                    break;
                }
                cov_colmaj[j * d + i] = cov_val;
            }
            if has_nan_inf {
                break;
            }
        }

        if !has_nan_inf {
            let cov = DMatrix::from_vec(d, d, cov_colmaj.clone());
            let factors = compute_all_factors(&cov);

            for i in 0..15 {
                if let Some(v) = factors[i] {
                    result_data[w * COV_NCOL + i] = v;
                }
            }

            let b_oe = factors[7];
            let a_oe = factors[11];
            let b_hl = factors[8];
            let a_hl = factors[12];
            let b_adj = factors[9];
            let a_adj = factors[13];
            let b_diag = factors[10];
            let a_diag = factors[14];

            if let (Some(b), Some(a)) = (b_oe, a_oe) {
                result_data[w * COV_NCOL + 15] = b - a;
                result_data[w * COV_NCOL + 16] = (b - a).abs();
            }
            if let (Some(b), Some(a)) = (b_hl, a_hl) {
                result_data[w * COV_NCOL + 17] = b - a;
                result_data[w * COV_NCOL + 18] = (b - a).abs();
            }
            if let (Some(b), Some(a)) = (b_adj, a_adj) {
                result_data[w * COV_NCOL + 19] = b - a;
                result_data[w * COV_NCOL + 20] = (b - a).abs();
            }
            if let (Some(b), Some(a)) = (b_diag, a_diag) {
                result_data[w * COV_NCOL + 21] = b - a;
                result_data[w * COV_NCOL + 22] = (b - a).abs();
            }
        }

        if w < n_windows - 1 {
            let old_t = w;
            let new_t = w + window_size;
            for i in 0..d {
                let vi_old = diff_data[old_t * d + i];
                sum_x[i] -= vi_old;
                for j in 0..=i {
                    let vj_old = diff_data[old_t * d + j];
                    sum_xx[i * d + j] -= vi_old * vj_old;
                }
            }
            for i in 0..d {
                let vi_new = diff_data[new_t * d + i];
                sum_x[i] += vi_new;
                for j in 0..=i {
                    let vj_new = diff_data[new_t * d + j];
                    sum_xx[i * d + j] += vi_new * vj_new;
                }
            }
        }
    }

    result_data
}

// ============================================================================
// 主流程
// ============================================================================

/// 旧 go 的输出长度（= names_save 去掉 `_fold` 后的项数）。
const N_OUT: usize = 78;

/// 名字 → 下标的缓存。第一次调用时在 features 返回的名字里按名字查一次。
static IDX_CACHE: OnceLock<Vec<usize>> = OnceLock::new();

fn src_indices(generated: &[String]) -> &'static Vec<usize> {
    IDX_CACHE.get_or_init(|| {
        SRC.iter()
            .map(|n| generated.iter().position(|x| x == n).unwrap_or(usize::MAX))
            .collect()
    })
}

/// 按 SRC 的顺序把 78 项取出来。任何一步缺数据都返回 78 个 NaN。
fn full_values(code: &str, date: i64) -> Vec<f64> {
    match try_full_values(code, date) {
        Some(v) if v.len() == N_OUT => v,
        _ => vec![f64::NAN; N_OUT],
    }
}

fn try_full_values(code: &str, date: i64) -> Option<Vec<f64>> {
    let raw = read_market_rows(code, date)?;

    // rp.read_market：丢掉 ask_prc1 == 0 or bid_prc1 == 0 的行（NaN != 0，空值行保留）
    let kept: Vec<MktRow> = raw
        .into_iter()
        .filter(|r| r.ask_prc1 != 0.0 && r.bid_prc1 != 0.0)
        .collect();

    // rp.adjust_afternoon：上午 [09:30, 11:30]、下午 [13:00, 14:57] 微秒闭区间，
    // 下午整体前移 90 分钟，上午全部在前、下午全部在后。
    let mut rows: Vec<MktRow> = Vec::with_capacity(kept.len());
    for r in kept.iter() {
        let t = r.time_us.rem_euclid(DAY_US);
        if t >= AM_START_US && t <= AM_END_US {
            rows.push(*r);
        }
    }
    for r in kept.iter() {
        let t = r.time_us.rem_euclid(DAY_US);
        if t >= PM_START_US && t <= PM_END_US {
            let mut q = *r;
            q.time_us -= PM_SHIFT_US;
            rows.push(q);
        }
    }
    drop(kept);

    // .drop_duplicates(subset=['exchtime'])：保留第一次出现
    let mut seen: HashSet<i64> = HashSet::with_capacity(rows.len());
    let mkt: Vec<MktRow> = rows.into_iter().filter(|r| seen.insert(r.time_us)).collect();

    let n = mkt.len();
    if n <= WIN10 - 1 {
        return None;
    }
    let n_rows = n - (WIN10 - 1);

    let mut bid_arr = Array2::<f64>::zeros((n, 10));
    let mut ask_arr = Array2::<f64>::zeros((n, 10));
    for (i, r) in mkt.iter().enumerate() {
        for k in 0..10 {
            bid_arr[[i, k]] = r.bid_vol[k];
            ask_arr[[i, k]] = r.ask_vol[k];
        }
    }

    let f10 = cov_window_factors(bid_arr.view(), ask_arr.view(), WIN10);
    let f60 = cov_window_factors(bid_arr.view(), ask_arr.view(), WIN60);
    let n60 = if n >= WIN60 { n - WIN60 + 1 } else { 0 };

    // 拼 df：N-9 行 × 39 列；第 r 行对应 mkt[9+r]，60w 从第 50 行才开始有值。
    let mut data = Array2::<f64>::from_elem((n_rows, N_DATA_COL), f64::NAN);
    for r in 0..n_rows {
        if r + (WIN10 - 1) >= (WIN60 - 1) {
            let w = r + (WIN10 - 1) - (WIN60 - 1);
            if w < n60 {
                for c in 0..COV_NCOL {
                    data[[r, c]] = f60[w * COV_NCOL + c];
                }
            }
        }
        for (k, &c) in COL10_IDX.iter().enumerate() {
            data[[r, COV_NCOL + k]] = f10[r * COV_NCOL + c];
        }
    }

    // append_for_corr：volume.diff() 与 last_prc
    let mut app = Array2::<f64>::from_elem((n_rows, 2), f64::NAN);
    for r in 0..n_rows {
        let i = r + (WIN10 - 1);
        if r > 0 {
            app[[r, 0]] = mkt[i].volume - mkt[i - 1].volume;
        }
        app[[r, 1]] = mkt[i].last_prc;
    }

    let col_names: Vec<String> = DATA_COLS.iter().map(|s| (*s).to_string()).collect();
    let app_names: Vec<String> = vec!["volume".to_string(), "last_prc".to_string()];

    let opts = FeatureOpts {
        with_threshold_counts: true,
        with_curvature: false,
        with_max_min: false,
        with_lyapunov_exponent: false,
    };
    let (vals, gen_names) = features_f64::get_features_factors_rust_full_ex(
        &data.view(),
        &col_names,
        opts,
        Some((&app.view(), app_names.as_slice())),
    );
    let idx = src_indices(&gen_names);
    Some(
        idx.iter()
            .map(|&i| if i < vals.len() { vals[i] } else { f64::NAN })
            .collect(),
    )
}

/// 逐股入口。
pub fn compute_stock(code: &str, date: i64) -> Option<Vec<f32>> {
    let vals = full_values(code, date);
    Some(vals.iter().map(|&v| v as f32).collect())
}

/// hm79 是逐股系列，没有横截面。
pub fn compute_cross(_date: i64) -> Option<(Vec<String>, Vec<f32>)> {
    None
}
