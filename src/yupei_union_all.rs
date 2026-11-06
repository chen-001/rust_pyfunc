//! 玉佩四合一：yupei_dist_lite + yupei_gpt + yupei_pair_split + yupei_multi_v5 的因子并集。
//!
//! 列序（共 20716 列）：
//!   1. yupei_dist_lite 741 列（prev=None，跨日 dyn_* / dynjs_* 列为 NaN，与单跑一致）
//!   2. yupei_gpt 491 列（设了 YUPEI_GPT_SNAP_DIR 时照常写快照，供 yupei_gpt_temporal 用）
//!   3. yupei_pair_split 9795 列（15 条链，ks=[5,20,100]，参数与它的 pipeline 一致）
//!   4. yupei_multi_v5 去掉与第 3 段同名的 3225 列后剩下的 9689 列，按 v5 原顺序
//!      （or_/oc_/os_ 三套载体 9675 列 + ld_AA_* 6 列 + sv_comp* 8 列）
//!
//! 做法：四个原函数依次调用，按股票代码对齐后横向拼接。每一列都直接来自原函数，
//! 与单独调用逐位一致。20241231 实测 split 与 v5 的 3225 列同名列逐位相等，
//! 所以重复列保留 split 那份、丢掉 v5 那份不损失任何信息。
//!
//! 行（股票）取四段代码的并集、升序；某段缺这只股票时该段整段填 NaN（不当 0）。
//! 任何一段计算失败，整天报错并写明是哪一段——不允许静默少一段列。

use std::collections::{BTreeSet, HashMap, HashSet};
use std::io;

use pyo3::prelude::*;

/// split 段的 top-k 档位，与 pipeline_yupei_pair_split 一致。
const SPLIT_KS: [usize; 3] = [5, 20, 100];
/// v5 段的 top-k 档位与分解成分数，与 pipeline_yupei_multi_v5 一致。
const V5_KS: [usize; 2] = [5, 20];
const V5_SVD_K: usize = 8;

fn v5_chains() -> Vec<String> {
    crate::yupei_multi::V5_CHAINS.iter().map(|s| s.to_string()).collect()
}

/// v5 中要保留的列（下标, 名字）：去掉在 split 里出现过的同名列。
fn v5_keep(v5_names: &[String], split_names: &[String]) -> Vec<(usize, String)> {
    let dup: HashSet<&str> = split_names.iter().map(|s| s.as_str()).collect();
    v5_names
        .iter()
        .enumerate()
        .filter(|(_, n)| !dup.contains(n.as_str()))
        .map(|(i, n)| (i, n.clone()))
        .collect()
}

/// 列名单源：不读数据，毫秒级。与 compute_yupei_union_all 的输出列逐字一致。
pub fn yupei_union_all_names() -> Vec<String> {
    let mut names: Vec<String> = crate::yupei_dist::lite_names::YUPEI_DIST_LITE_NAMES
        .iter()
        .map(|s| s.to_string())
        .collect();
    names.extend(crate::yupei_gpt::names::daily_names());
    let split = crate::yupei_pair::split::split_factor_names(&[], &SPLIT_KS);
    let v5 = crate::yupei_multi::multi_factor_names(&V5_KS, V5_SVD_K, Some(&v5_chains()));
    let keep = v5_keep(&v5, &split);
    names.extend(split);
    names.extend(keep.into_iter().map(|(_, n)| n));
    names
}

/// 一段结果：代码、列名、行主序数值。
struct Part {
    tag: &'static str,
    codes: Vec<String>,
    names: Vec<String>,
    vals: Vec<f32>,
    /// 只取这些列（None = 全取）
    pick: Option<Vec<usize>>,
}

fn tag_err(tag: &str, e: io::Error) -> io::Error {
    io::Error::new(e.kind(), format!("{tag} 段计算失败: {e}"))
}

/// 单日全市场四段因子并集。返回 (codes, names, vals 行主序 len(codes) × len(names))。
pub fn compute_yupei_union_all(date: i64) -> io::Result<(Vec<String>, Vec<String>, Vec<f32>)> {
    let verbose = std::env::var("YUPEI_STAGE").is_ok();
    let t0 = std::time::Instant::now();
    let log = |tag: &str| {
        if verbose {
            eprintln!("[yupei_union_all] {tag} 完成，累计 {:.1}s", t0.elapsed().as_secs_f64());
        }
    };

    // ---- 1. 四段依次算（顺序执行，避免四份大矩阵同时占内存）----
    let (lite_codes, lite_vals) = crate::yupei_dist::compute::compute_yupei_dist_lite(date, None)
        .map_err(|e| tag_err("yupei_dist_lite", e))?;
    let lite = Part {
        tag: "yupei_dist_lite",
        codes: lite_codes,
        names: crate::yupei_dist::lite_names::YUPEI_DIST_LITE_NAMES
            .iter()
            .map(|s| s.to_string())
            .collect(),
        vals: lite_vals,
        pick: None,
    };
    log("yupei_dist_lite");

    let (gpt_codes, gpt_vals) =
        crate::yupei_gpt::compute::compute_full(date).map_err(|e| tag_err("yupei_gpt", e))?;
    let gpt = Part {
        tag: "yupei_gpt",
        codes: gpt_codes,
        names: crate::yupei_gpt::names::daily_names(),
        vals: gpt_vals,
        pick: None,
    };
    log("yupei_gpt");

    let (split_codes, split_names, split_vals) = crate::yupei_pair::split::compute_split_full(
        date,
        1.0,
        crate::yupei_pair::split_mat::PmWeight::Cnt,
        &[],
        &SPLIT_KS,
        crate::yupei_pair::reduce::SoftCfg { temp: 0.3, thr: 2.0 },
        20000,
        10,
    )
    .map_err(|e| tag_err("yupei_pair_split", e))?;
    log("yupei_pair_split");

    let (v5_codes, v5_names, v5_vals) = crate::yupei_multi::compute_multi_full(
        date,
        1.0,
        "cnt",
        &V5_KS,
        crate::yupei_pair::reduce::SoftCfg { temp: 0.3, thr: 2.0 },
        20000,
        10,
        V5_SVD_K,
        Some(&v5_chains()),
    )
    .map_err(|e| tag_err("yupei_multi_v5", e))?;
    log("yupei_multi_v5");

    let keep = v5_keep(&v5_names, &split_names);
    let split = Part {
        tag: "yupei_pair_split",
        codes: split_codes,
        names: split_names,
        vals: split_vals,
        pick: None,
    };
    let v5 = Part {
        tag: "yupei_multi_v5",
        codes: v5_codes,
        names: keep.iter().map(|(_, n)| n.clone()).collect(),
        vals: v5_vals,
        pick: Some(keep.iter().map(|(i, _)| *i).collect()),
    };
    // v5 的原始列数（行宽）要按原始列名数算，不是保留后的
    let v5_width = v5_names.len();

    let parts = [lite, gpt, split, v5];

    // ---- 2. 校验每段数值长度 = 行数 × 行宽 ----
    for p in parts.iter() {
        let width = if p.tag == "yupei_multi_v5" { v5_width } else { p.names.len() };
        if p.vals.len() != p.codes.len() * width {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "{} 段数值长度 {} ≠ 股票数 {} × 列数 {}",
                    p.tag,
                    p.vals.len(),
                    p.codes.len(),
                    width
                ),
            ));
        }
    }

    // ---- 3. 股票取并集（升序），按代码对齐拼接 ----
    let all_codes: Vec<String> = parts
        .iter()
        .flat_map(|p| p.codes.iter().cloned())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    let names: Vec<String> = parts.iter().flat_map(|p| p.names.iter().cloned()).collect();
    let f = names.len();
    let n = all_codes.len();
    let row_of: HashMap<&str, usize> =
        all_codes.iter().enumerate().map(|(i, c)| (c.as_str(), i)).collect();

    let mut vals = vec![f32::NAN; n * f];
    let mut col0 = 0usize;
    for p in parts.iter() {
        let width = if p.tag == "yupei_multi_v5" { v5_width } else { p.names.len() };
        let out_w = p.names.len();
        for (r, code) in p.codes.iter().enumerate() {
            let src = &p.vals[r * width..(r + 1) * width];
            let dst_start = row_of[code.as_str()] * f + col0;
            let dst = &mut vals[dst_start..dst_start + out_w];
            match &p.pick {
                None => dst.copy_from_slice(src),
                Some(idx) => {
                    for (d, &j) in dst.iter_mut().zip(idx.iter()) {
                        *d = src[j];
                    }
                }
            }
        }
        col0 += out_w;
    }
    if verbose {
        let counts: Vec<String> = parts
            .iter()
            .map(|p| format!("{}={}只", p.tag, p.codes.len()))
            .collect();
        eprintln!(
            "[yupei_union_all] date={date} 并集 {n} 只 × {f} 列（{}），总 {:.1}s",
            counts.join(", "),
            t0.elapsed().as_secs_f64()
        );
    }
    Ok((all_codes, names, vals))
}

/// Python 入口：单日四合一因子。返回 (codes, names, vals)，vals 为行主序扁平 f32。
#[pyfunction]
pub fn py_yupei_union_all(date: i64) -> PyResult<(Vec<String>, Vec<String>, Vec<f32>)> {
    compute_yupei_union_all(date).map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("{e:?}")))
}

/// Python 入口：四合一的列名（20716 个），不读数据。
#[pyfunction]
pub fn py_yupei_union_all_names() -> Vec<String> {
    yupei_union_all_names()
}
