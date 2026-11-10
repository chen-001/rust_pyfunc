//! 玉佩四合一精简版（2026-11-07），列序固定：lite 115、GPT 455、split 9795、multi 8393。
//! 按四个原函数的单独回测入选名单裁剪计算，值与沙箱 v6 的验收口径一致。
//! 通过共用入口对齐股票代码；完整 20716 列版继续使用 yupei_union_all。

use pyo3::prelude::*;

pub const N_FACTORS: usize = 18_758;

/// lite 段保留的指标家族（列名第一个下划线前的部分）：单独回测里有入选的 13 个。
const LITE_KEEP_FAMS: [&str; 13] = [
    "assort", "bh", "clo", "conc", "forman", "jsmarg", "lead", "mom", "nbz", "star", "topk",
    "trophic", "w1",
];
/// lite 段删掉的矩阵：28 张里单独回测无一入选的 14 张（任何家族落在这些矩阵上的列都删）。
const LITE_DROP_MATS: [&str; 14] = [
    "rvol_t005", "rvol_t30", "logvol_t05", "flow_same_t02", "flow_opp_t02", "flow_same_t05",
    "flow_opp_t05", "flow_opp_t1", "urg_same_t1", "urg_opp_t1", "urg_opp_t5", "ext_same_t1",
    "ext_opp_t1", "ext_opp_t5",
];
/// v5 段跳过的 (载体前缀, 链)：单独回测无一入选。
pub(crate) const V5_DROP_CHAINS: [(&str, &str); 2] = [("or_", "sppma"), ("oc_", "sppma")];

/// lite 段精简后的列（按 741 列原顺序）。
pub(crate) fn lite_trim_names() -> Vec<&'static str> {
    crate::yupei_dist::lite_names::YUPEI_DIST_LITE_NAMES
        .iter()
        .copied()
        .filter(|n| {
            let fam = n.split('_').next().unwrap_or("");
            let wrapped = format!("_{n}_");
            LITE_KEEP_FAMS.contains(&fam)
                && !LITE_DROP_MATS.iter().any(|m| wrapped.contains(&format!("_{m}_")))
        })
        .collect()
}

/// v5 全量列名里，精简版不算的列（跳过的链 + ld_AA_*）。
pub(crate) fn v5_trimmed_out(name: &str) -> bool {
    name.starts_with("ld_AA_")
        || V5_DROP_CHAINS
            .iter()
            .any(|(tag, chain)| name.starts_with(&format!("{tag}{chain}_")))
}

pub fn compute_yupei_union_trim(date: i64) -> std::io::Result<(Vec<String>, Vec<String>, Vec<f32>)> {
    // The old full entry retains its opt-in precision experiment. This entry
    // has a fixed exact-precision contract.
    if std::env::var_os("YUPEI_BF16").is_some() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "yupei_union_trim 不支持 YUPEI_BF16 精度实验",
        ));
    }
    let run = || crate::yupei_union_all::compute_union(date, true);
    let configured = std::env::var("RAYON_NUM_THREADS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok());
    // Reuse the worker's assigned pool, including smaller allocations. Do
    // not initialize an uncapped global pool for a standalone Python call.
    let max_threads = 50;
    let has_pool = rayon::current_thread_index().is_some()
        || configured.is_some_and(|n| (1..=max_threads).contains(&n));
    let (codes, names, vals) = if has_pool && rayon::current_num_threads() <= max_threads {
        run()?
    } else {
        static POOL: std::sync::OnceLock<Result<rayon::ThreadPool, String>> =
            std::sync::OnceLock::new();
        let pool = POOL.get_or_init(|| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(max_threads)
                .build()
                .map_err(|err| err.to_string())
        });
        pool.as_ref().map_err(|err| std::io::Error::other(err.clone()))?.install(run)?
    };
    if names.len() != N_FACTORS || vals.len() != codes.len() * N_FACTORS {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("yupei_union_trim 需要 {N_FACTORS} 列，实际 {} 列", names.len()),
        ));
    }
    Ok((codes, names, vals))
}

#[pyfunction]
pub fn py_yupei_union_trim(date: i64) -> PyResult<(Vec<String>, Vec<String>, Vec<f32>)> {
    compute_yupei_union_trim(date).map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))
}

#[pyfunction]
pub fn py_yupei_union_trim_names() -> Vec<String> {
    crate::yupei_union_all::union_names(true)
}
