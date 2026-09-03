//! COR-15「切换时刻贡献」补充因子（正式版，仅入选因子）：柜员游戏退火 s80 × 2。
//!
//! 判定结论（skill B 模式 1，supplement_ver=切换时刻贡献_v2_supp_supplement_anneal）：
//! 43 个候选因子中仅 `qhsk_ext_aseq_buy_ratio_s80` 与 `qhsk_ext_aseq_up_tick_ratio_s80`
//! 值得补充（gap5 中性化 |IC| 0.0470 / 0.0451 > 入选下限 0.0226，且未被初版高相关高 IC 因子阻挡）。
//! 本模块只实现这两个因子；评估目录保留全量 43 因子数据（不注册组合 store）。
//!
//! 计算口径（与评估版逐位一致，保证 accepted group 因子值与正式代码产出完全一致）：
//!   每只股票取过去 5 个交易日（含当日）的分钟字段序列（每天有效行 = 前 236 根 1 分钟 K 线，
//!   末 4 根集合竞价行置 NaN 不参与），自身 z-score 后做「柜员游戏」模拟退火：
//!     guess = 升序排列、target = 原始时序；固定种子 xorshift64 + 线性降温 Metropolis
//!     （ΔS = 2·(g_i−g_j)·(t_i−t_j)，ct = σ²·(1−step/m_max)，σ²=1），m_max = min(2000, 10N)，
//!     每步 O(1) 增量更新 r = Σ(g·t)/N；
//!   s80 = 退火过程中 r 首次达到 0.8 的步数（达不到记 m_max+1）——度量“序列离可完美还原
//!   还有多远”的爬升速度：噪声越小、趋势越强的序列爬得越快。
//!   字段：buy_ratio = 主动买入金额/成交金额（分母 0 → 0）；up_tick_ratio = 上涨笔数/(上涨+下跌笔数)。
//!   N < 100 或序列方差为 0 → 输出 NaN。
//!
//! 股票轴：设环境变量 RUST_PYFUNC_AXIS_ALLOWLIST（每行一个 6 位代码的文本文件）时输出
//! 白名单全集（排序），缺失/无数据的股票输出 2 个 NaN——保证 store 股票轴与初版一致；
//! 未设白名单时输出当日有值的股票。
//!
//! 全程确定性：固定随机种子、排序全 total_cmp、无 HashMap 遍历。

use crate::minute_data_reader::{last_n_trading_dates, read_minute_field_multi_day};
use ndarray::Array3;
use pyo3::prelude::*;
use std::collections::HashSet;
use std::fs;
use std::io;
use std::sync::OnceLock;

pub const N_FACTORS: usize = 2;
const MIN_PER_DAY: usize = 240;
const N_TAIL_NAN: usize = 4;
/// 每天参与计算的有效分钟行数（末 4 根集合竞价行置 NaN）。
const VALID_MIN: usize = MIN_PER_DAY - N_TAIL_NAN;
/// 固定随机种子（与评估版一致，保证因子值逐位相同）。
const SEED: u64 = 0x9E37_79B9_7F4A_7C15;

static AXIS_ALLOWLIST: OnceLock<Option<HashSet<String>>> = OnceLock::new();

fn axis_allowlist() -> Option<&'static HashSet<String>> {
    AXIS_ALLOWLIST
        .get_or_init(|| {
            std::env::var("RUST_PYFUNC_AXIS_ALLOWLIST").ok().and_then(|p| {
                fs::read_to_string(p).ok().map(|s| {
                    s.lines()
                        .map(|l| l.trim().to_string())
                        .filter(|x| !x.is_empty())
                        .collect()
                })
            })
        })
        .as_ref()
}

/// 因子名（唯一真相源，与 accepted group 名单一致）。
pub fn qhsk_anneal_names() -> Vec<String> {
    vec![
        "qhsk_ext_aseq_buy_ratio_s80".to_string(),
        "qhsk_ext_aseq_up_tick_ratio_s80".to_string(),
    ]
}

/// 读取单字段多日 → f32 Array3（ndays, 240, n_stocks），末 4 行已由 reader 置 NaN。
fn read_field_f32(field: &str, dates: &[i64]) -> io::Result<(Vec<String>, Array3<f32>)> {
    let (codes, arr) = read_minute_field_multi_day(field, dates)?;
    Ok((codes, arr.mapv(|v| v as f32)))
}

/// buy_ratio = act_buy_amount_sum / amount（缺失→NaN；精确分母 0 且分子有限→0）。
fn derive_buy_ratio(dates: &[i64]) -> io::Result<(Vec<String>, Array3<f32>)> {
    let (codes, buy) = read_field_f32("act_buy_amount_sum", dates)?;
    let (_, amt) = read_field_f32("amount", dates)?;
    let n_days = dates.len();
    let mut out = Array3::<f32>::from_elem((n_days, MIN_PER_DAY, buy.shape()[2]), f32::NAN);
    for d in 0..n_days {
        for i in 0..MIN_PER_DAY {
            for s in 0..out.shape()[2] {
                let a = amt[(d, i, s)];
                let b = buy[(d, i, s)];
                if a.is_finite() && b.is_finite() {
                    out[(d, i, s)] = if a != 0.0 { b / a } else { 0.0 };
                }
            }
        }
    }
    Ok((codes, out))
}

/// up_tick_ratio = up_tick_count / (up_tick_count + down_tick_count)（缺失→NaN；分母 0→0）。
fn derive_up_tick(dates: &[i64]) -> io::Result<(Vec<String>, Array3<f32>)> {
    let (codes, up) = read_field_f32("up_tick_count", dates)?;
    let (_, down) = read_field_f32("down_tick_count", dates)?;
    let n_days = dates.len();
    let mut out = Array3::<f32>::from_elem((n_days, MIN_PER_DAY, up.shape()[2]), f32::NAN);
    for d in 0..n_days {
        for i in 0..MIN_PER_DAY {
            for s in 0..out.shape()[2] {
                let u = up[(d, i, s)];
                let w = down[(d, i, s)];
                if u.is_finite() && w.is_finite() {
                    let denom = u + w;
                    out[(d, i, s)] = if denom != 0.0 { u / denom } else { 0.0 };
                }
            }
        }
    }
    Ok((codes, out))
}

#[inline]
fn xorshift64(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

/// 单股单字段：5 日窗口分钟序列 → (r0, r_final, s80, auc, half)。N<100 或方差 0 → None。
/// 与评估版逐行一致（含顺序累加与 f64/f32 精度），保证 s80 值与评估 store 逐位相同。
fn anneal_metrics(series: &[f32], n: usize) -> Option<[f32; 5]> {
    if n < 100 {
        return None;
    }
    let n_f = n as f64;
    let mut mean = 0f64;
    for k in 0..n {
        mean += series[k] as f64;
    }
    mean /= n_f;
    let mut ss = 0f64;
    for k in 0..n {
        let dz = series[k] as f64 - mean;
        ss += dz * dz;
    }
    let var = ss / n_f;
    if var <= 0.0 {
        return None;
    }
    let inv = 1.0 / var.sqrt();
    let mut target = vec![0f32; n];
    for k in 0..n {
        target[k] = ((series[k] as f64 - mean) * inv) as f32;
    }
    let mut guess = target.clone();
    guess.sort_by(|a, b| a.total_cmp(b));
    let mut r = 0f64;
    for k in 0..n {
        r += guess[k] as f64 * target[k] as f64;
    }
    r /= n_f;
    let r0 = r;
    let m_max = (2000usize).min(10 * n);
    let mut state = SEED;
    let mut r_at: Vec<f64> = Vec::with_capacity(m_max + 1);
    r_at.push(r);
    for step in 0..m_max {
        let i = (xorshift64(&mut state) % n as u64) as usize;
        let mut j = (xorshift64(&mut state) % (n as u64 - 1)) as usize;
        if j >= i {
            j += 1;
        }
        let ds = 2.0 * (guess[i] - guess[j]) * (target[i] - target[j]);
        let ct = 1.0 * (1.0 - step as f64 / m_max as f64);
        if (ds as f64) < ct {
            guess.swap(i, j);
            r -= ds as f64 / (2.0 * n_f);
        }
        r_at.push(r);
    }
    let r_final = r;
    let half_target = r0 + 0.5 * (r_final - r0);
    let mut s80 = (m_max + 1) as f32;
    let mut half = (m_max + 1) as f32;
    let mut auc = 0f64;
    for (idx, &rv) in r_at.iter().enumerate().skip(1) {
        if s80 as usize > m_max && rv >= 0.8 {
            s80 = idx as f32;
        }
        if half as usize > m_max && rv >= half_target {
            half = idx as f32;
        }
        auc += rv;
    }
    let auc = (auc / m_max as f64) as f32;
    Some([r0 as f32, r_final as f32, s80, auc, half])
}

/// 核心唯一真相源：pipeline 与 Python 入口的共同调用点。
pub fn compute_qhsk_anneal_full(date: i64) -> io::Result<(Vec<String>, Vec<f32>)> {
    let dates = last_n_trading_dates(date, 5)?;
    if dates.is_empty() {
        return Ok((Vec::new(), Vec::new()));
    }
    let n_days = dates.len();

    let (codes_all, buy_ratio) = derive_buy_ratio(&dates)?;
    let (_, up_tick) = derive_up_tick(&dates)?;
    let n_stocks = codes_all.len();
    let fields: [&Array3<f32>; 2] = [&buy_ratio, &up_tick];

    let al = axis_allowlist();
    // 代码 → 读者列号（仅查找用，不遍历，无确定性风险）
    let code_to_col: std::collections::HashMap<String, usize> = codes_all
        .iter()
        .enumerate()
        .map(|(i, c)| (c.clone(), i))
        .collect();

    // 输出轴：设白名单时 = 白名单全集（排序），缺失/无数据的股票输出 2 个 NaN。
    let output_list: Vec<(String, Option<usize>)> = match al {
        Some(set) => {
            let mut v: Vec<&String> = set.iter().collect();
            v.sort();
            v.iter()
                .map(|c| ((*c).clone(), code_to_col.get(*c).copied()))
                .collect()
        }
        None => codes_all
            .iter()
            .enumerate()
            .filter(|(s, _)| {
                (0..n_days).any(|d| (0..VALID_MIN).any(|t| buy_ratio[(d, t, *s)].is_finite()))
            })
            .map(|(s, c)| (c.clone(), Some(s)))
            .collect(),
    };

    let mut out_codes: Vec<String> = Vec::with_capacity(output_list.len());
    let mut out_vals: Vec<f32> = Vec::with_capacity(output_list.len() * N_FACTORS);
    let mut series: Vec<f32> = vec![0.0; n_days * VALID_MIN];
    for (code, s_opt) in output_list {
        out_codes.push(code);
        let Some(s) = s_opt else {
            out_vals.extend([f32::NAN; N_FACTORS]);
            continue;
        };
        for fi in 0..2 {
            let mut n = 0usize;
            for d in 0..n_days {
                for t in 0..VALID_MIN {
                    let v = fields[fi][(d, t, s)];
                    if v.is_finite() {
                        series[n] = v;
                        n += 1;
                    }
                }
            }
            match anneal_metrics(&series, n) {
                Some(m) => out_vals.push(m[2]), // s80
                None => out_vals.push(f32::NAN),
            }
        }
    }
    Ok((out_codes, out_vals))
}

// ============================================================
// Python 包装
// ============================================================

#[pyfunction]
pub fn py_qhsk_anneal(_py: Python<'_>, date: i64) -> PyResult<(Vec<String>, Vec<f32>)> {
    compute_qhsk_anneal_full(date).map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("{e:?}")))
}

#[pyfunction]
pub fn py_qhsk_anneal_names() -> Vec<String> {
    qhsk_anneal_names()
}
