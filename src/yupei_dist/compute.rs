//! 正式库入口: 单日全市场横截面因子计算（寻找玉佩-成交距离）。
//!
//! 流程（与 sandbox `yupei_dist factors` 完全一致）:
//!   1. 枚举代码（按文件大小降序, universe=4000 截断, 代码序恢复确定性）
//!   2. 并行读逐笔 + 每股预处理（10ms 桶, u[9] 权重场）
//!   3. 37 张关联-差异矩阵（衰减事件场交互能量）
//!   4. 28 个降维指标模块 → 2761 个横截面因子
//!   5. 组装 (codes, vals N×F) 供 pipeline fan-out
//!
//! 行业数据: 读备份目录 {BACKUP_DIR}/{date}/industry.bin（u64 n + n×i16, 与 codes 对齐）;
//! 缺失时行业模块输出 NaN（因子长度不变）。
//! 前一日矩阵: cross-section pipeline 单日任务无 prev（批量首日 dyn_* 输出 NaN）。

use std::collections::HashMap;
use std::path::Path;

use pyo3::prelude::*;
use rayon::prelude::*;

use super::indicator_ctx::{IndicatorCtx, MatrixSet};
use super::matrix_stage::{compute_matrices, prep_stock, StockPrep, MATRIX_SPECS};
use super::names::YUPEI_DIST_NAMES;

/// 因子宇宙：全市场（不设股票数上限；MIN_TRADES=200 过滤低活跃股）
/// 最少成交笔数（低于此剔除）
pub const MIN_TRADES: usize = 200;
/// 行业备份目录（python 一次性预提取全部交易日）
pub const BACKUP_DIR: &str = "/hdd/user_home_unsafe/chenzongwei/yupei_dist_backup";

/// 单日全市场横截面因子: (codes, vals) — vals 行主序 N×F（F=2761）。
pub fn compute_yupei_dist_full(date: i64) -> std::io::Result<(Vec<String>, Vec<f32>)> {
    let (codes_out, n, stats, hm, industry) = compute_yupei_dist_partial(date)?;
    let set = MatrixSet { n, codes: codes_out.clone(), stats, mats: hm, industry };
    let ctx = IndicatorCtx::new(&set, None);
    let vals = run_indicators(&ctx, n, &codes_out)?;
    Ok((codes_out, vals))
}

/// 单日前半段（读数据 + 矩阵 + 行业）: 返回组装 MatrixSet 所需的部件。
pub fn compute_yupei_dist_partial(
    date: i64,
) -> std::io::Result<(Vec<String>, usize, Vec<crate::yupei_dist::matrix_stage::StockStats>, HashMap<String, Vec<f32>>, Option<Vec<i16>>)> {
    // ---- 1. 代码枚举（按文件大小降序 → 全市场不截断 → 代码序）----
    let mut codes = list_codes_by_size(date);
    codes.sort();
    let day_start_us = day_start_us_of(date);

    // ---- 2. 并行读 + 每股预处理 ----
    let preps: Vec<Option<StockPrep>> = codes
        .par_iter()
        .map(|c| {
            crate::fast_csv_reader::read_trade_fast_inner(c, date, false, true, usize::MAX)
                .ok()
                .and_then(|recs| prep_stock(c, &recs, day_start_us, MIN_TRADES))
        })
        .collect();
    let stocks: Vec<StockPrep> = preps.into_iter().flatten().collect();
    if stocks.is_empty() {
        return Err(std::io::Error::new(std::io::ErrorKind::Other, "无有效股票"));
    }
    let codes_out: Vec<String> = stocks.iter().map(|s| s.code.clone()).collect();
    let n = codes_out.len();

    // ---- 3. 37 张矩阵 + 统计量 ----
    let (mats, stats) = compute_matrices(&stocks);
    let mut hm: HashMap<String, Vec<f32>> = HashMap::with_capacity(MATRIX_SPECS.len());
    for (m, spec) in mats.iter().zip(MATRIX_SPECS.iter()) {
        hm.insert(spec.name.to_string(), m.clone());
    }

    // ---- 4. 行业（可选; 缺失 → None → 行业模块输出 NaN）----
    // industry.bin 按 symbol_map 全市场顺序存储: 读入后按 codes 重排（缺失 -1）
    let industry = super::industry::load_industry_all(
        &Path::new(BACKUP_DIR).join(date.to_string()).join("industry.bin"),
    )
    .ok()
    .map(|all: std::collections::HashMap<String, i16>| {
        codes_out.iter().map(|c| all.get(c).copied().unwrap_or(-1)).collect()
    });

    Ok((codes_out, n, stats, hm, industry))
}

/// 后半段: 跑全部指标模块 + 组装 N×F（行主序）+ 长度/顺序校验。
pub fn run_indicators(ctx: &IndicatorCtx, n: usize, codes_out: &[String]) -> std::io::Result<Vec<f32>> {
    let mut cols: Vec<Vec<f32>> = Vec::new();
    for def in super::indicators::all() {
        for r in (def.compute)(ctx) {
            debug_assert_eq!(r.values.len(), n);
            cols.push(r.values);
        }
    }
    let nf = cols.len();
    if nf != YUPEI_DIST_NAMES.len() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::Other,
            format!("因子数不匹配: 计算 {nf} != 注册 {}. 检查指标模块与 names.rs 一致性", YUPEI_DIST_NAMES.len()),
        ));
    }
    let mut vals = vec![0.0f32; n * nf];
    for (fi, col) in cols.iter().enumerate() {
        for i in 0..n {
            vals[i * nf + fi] = col[i];
        }
    }
    Ok(vals)
}

/// 单日全市场横截面因子（带前一日矩阵）: dyn_* 跨日因子有值。
/// prev_date 的 37 张矩阵从备份目录 {BACKUP_DIR}/{prev_date}/mats/*.bin 读取（f32 行主序）。
pub fn compute_yupei_dist_full_with_prev(date: i64, prev_date: Option<i64>) -> std::io::Result<(Vec<String>, Vec<f32>)> {
    let (codes_out, n, stats, hm, industry) = compute_yupei_dist_partial(date)?;
    // 加载前一日矩阵
    let prev = match prev_date {
        Some(pd) => {
            let pdir = Path::new(BACKUP_DIR).join(pd.to_string());
            let pcodes: Vec<String> = std::fs::read_to_string(pdir.join("codes.txt"))?
                .lines()
                .map(|l| l.trim().to_string())
                .filter(|l| !l.is_empty())
                .collect();
            let mut pmats: HashMap<String, Vec<f32>> = HashMap::new();
            for spec in MATRIX_SPECS.iter() {
                let p = pdir.join("mats").join(format!("{}.bin", spec.name));
                if p.exists() {
                    let raw = std::fs::read(&p)?;
                    let data: Vec<f32> = unsafe {
                        std::slice::from_raw_parts(raw.as_ptr() as *const f32, raw.len() / 4).to_vec()
                    };
                    pmats.insert(spec.name.to_string(), data);
                }
            }
            let pset = super::indicator_ctx::PrevMats {
                n: pcodes.len(),
                codes: pcodes,
                mats: pmats,
            };
            Some(super::indicator_ctx::PrevDay::new(Box::leak(Box::new(pset))))
        }
        None => None,
    };
    let set = MatrixSet { n, codes: codes_out.clone(), stats, mats: hm, industry };
    let ctx = super::indicator_ctx::IndicatorCtx::new(&set, prev);
    let vals = run_indicators(&ctx, n, &codes_out)?;
    Ok((codes_out, vals))
}

/// 因子名（与 compute 输出顺序一致）
pub fn yupei_dist_names() -> Vec<String> {
    YUPEI_DIST_NAMES.iter().map(|s| s.to_string()).collect()
}

/// Python 入口: 单日全市场 2761 个横截面因子 (codes, vals)。
#[pyfunction]
pub fn py_yupei_dist(date: i64) -> PyResult<(Vec<String>, Vec<f32>)> {
    compute_yupei_dist_full(date)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("{e:?}")))
}

/// Python 拿因子名（2761 个, 顺序与 py_yupei_dist 输出一致）。
#[pyfunction]
pub fn py_yupei_dist_names() -> Vec<String> {
    yupei_dist_names()
}

/// 枚举当日全市场代码（按文件大小降序; 同大小按代码序）
pub fn list_codes_by_size(date: i64) -> Vec<String> {
    for root in ["/ssd_data/stock", "/nas197/binary/stock/sz_alpha/stock"] {
        let dir = format!("{root}/{date}/transaction");
        if let Ok(entries) = std::fs::read_dir(&dir) {
            let mut v: Vec<(u64, String)> = Vec::new();
            for e in entries.flatten() {
                if let Some(code) = e.file_name().to_str().and_then(|n| n.split('_').next()) {
                    if code.bytes().all(|b| b.is_ascii_digit()) {
                        let size = e.metadata().map(|m| m.len()).unwrap_or(0);
                        v.push((size, code.to_string()));
                    }
                }
            }
            v.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
            let codes: Vec<String> = v.into_iter().map(|(_, c)| c).collect();
            if !codes.is_empty() {
                return codes;
            }
        }
    }
    Vec::new()
}

/// 日起点（UTC 零点 + 9.5h; 与 fast_csv_reader 的 +8h 偏移配合, 9:30 开盘 = 0）
pub fn day_start_us_of(date: i64) -> i64 {
    days_from_civil(date / 10000, (date / 100 % 100) as u64, (date % 100) as u64)
        * 86400
        * 1_000_000
        + (9 * 3600 + 30 * 60) * 1_000_000
}

fn days_from_civil(y: i64, m: u64, d: u64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400;
    let mp = (m + 9) % 12;
    let doy = (153 * mp + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy as i64;
    era * 146097 + doe - 719468
}
