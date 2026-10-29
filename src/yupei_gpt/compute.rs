//! 玉佩精炼版（GPT 改进版）单日核心：与 sandbox_yupei_refined 同一套计算，
//! 去掉 CLI/JSON 产物层，供 run_factor_pipeline_cross_section 使用。
//!
//! 一天的输出 = 网络列 + 控制/对比/价格响应/交互 + leadlag，共 491 列
//! （见 names.rs）。29 个历史时序列不在本函数内；它们由
//! compute_temporal_full 消费 .ygsnap.json 快照单独成 pipeline。
use crate::yupei_gpt::model::*;
use crate::yupei_gpt::{extensions, input, kernel, leadlag, network, temporal};
use rayon::prelude::*;
use std::collections::BTreeMap;
use std::io;
use std::time::Instant;

pub const SCHEMA: &str = "yupei-gpt-v1";
pub const CHAINS: [&str; 15] = [
    "AAa", "AAp", "AAm", "ppa", "ppp", "ppm", "pma", "pmp", "pmm", "mpa", "mpp", "mpm", "mma",
    "mmp", "mmm",
];

/// 快照里只保留 temporal 所需列（SPECS 基础列 + 三条链的宽度列）与邻居状态，
/// 避免每天落一份全量 491 列 JSON。
const SLIM_BASES: [&str; 16] = [
    "yr_AAa_row_rho_skew",
    "yr_ppp_row_gs_skew",
    "yr_ppp_res_z_p90",
    "yr_ppp_r4_ov_g_k100",
    "yr_mmm_r4_ov_z_k100",
    "yr_ppp_row_self_assoc_p95",
    "yr_ppa_row_gnpos_p90",
    "yr_AAp_e_w_std_soft",
    "yr_pmp_x0topk20_count",
    "yr_AAp_gtopk100_count",
    "yr_AAm_res_g_max",
    "yr_ppp_best_z_partner_reverse_rank",
    "yr_ppp_best_z_partner_reverse_rank_fraction",
    "yr_AAa_row_gnpos_p90",
    "yr_ppp_row_gnpos_p90",
    "yr_mmm_row_gnpos_p90",
];

pub struct DailyCore {
    pub codes: Vec<String>,
    pub columns: Vec<FactorColumn>,
    pub snapshot: Snapshot,
    pub calendar: Vec<i64>,
    pub timings: BTreeMap<String, f64>,
    pub audits: Vec<AuditRow>,
}

fn default_config(date: i64) -> Config {
    Config {
        date,
        level2_root: crate::data_paths::level2_roots()
            .into_iter()
            .next()
            .unwrap_or_else(|| std::path::PathBuf::from("/ssd_data/stock")),
        basic_info_dir: std::env::var_os("RUST_PYFUNC_BASIC_INFO_DIR")
            .map(Into::into)
            .unwrap_or_else(|| std::path::PathBuf::from("/ssd_data/data/basic_info")),
        vars_dir: crate::data_paths::vars_root(),
        snap_dir: std::env::var_os("YUPEI_GPT_SNAP_DIR").map(Into::into),
        allow_incomplete_history: true,
        allow_uncomputable_corr: true,
        codes: None,
        chains: CHAINS.iter().map(|s| s.to_string()).collect(),
    }
}

/// 与沙箱 compute() 同序：读白名单→逐笔→核→12 个家族并行网络→结构性精简
/// →扩展列→leadlag→快照。历史列不在这里算。
pub fn compute_daily(config: &Config) -> io::Result<DailyCore> {
    let mut chain_set = config.chains.clone();
    chain_set.sort();
    chain_set.dedup();
    if chain_set.is_empty()
        || chain_set.len() != config.chains.len()
        || chain_set.iter().any(|c| !CHAINS.contains(&c.as_str()))
    {
        return Err(input::bad(
            "chains must be nonempty, unique supported directions",
        ));
    }
    let total = Instant::now();
    let mut times = BTreeMap::new();
    let mut audits = Vec::new();
    let timer = Instant::now();
    if std::env::var_os("YUPEI_GPT_VERBOSE").is_some() {
        eprintln!("Loading whitelist stocks for {}", config.date);
    }
    let (stocks, manifest, calendar) = input::load(config)?;
    times.insert("read_and_aggregate".into(), timer.elapsed().as_secs_f64());
    let n = stocks.len();
    let codes: Vec<_> = stocks.iter().map(|s| s.code.clone()).collect();
    let hash = input::sha256(
        format!(
            "{SCHEMA}|{}|{}|{:?}|{:?}|{:?}|{:?}|tau=1;slot=10ms;continuous=237min",
            manifest.whitelist_sha256,
            manifest.industry_symbol_sha256,
            config.codes,
            chain_set,
            config.level2_root.canonicalize()?,
            config.vars_dir.canonicalize()?
        )
        .as_bytes(),
    );
    let timer = Instant::now();
    if std::env::var_os("YUPEI_GPT_VERBOSE").is_some() {
        eprintln!("Computing corrected event kernel for {n} stocks");
    }
    let kernel = kernel::compute_kernel(&stocks, 1.0)?;
    times.insert("event_kernel".into(), timer.elapsed().as_secs_f64());
    let mut columns = Vec::new();
    let mut neighbors = Vec::new();
    // Chain families: a==b chains scan half the ordered pairs; a!=b chains are
    // scanned once together with their role-mirror (e.g. pmp serves mpp).
    let mut done: std::collections::BTreeSet<String> = Default::default();
    let mut work: Vec<(String, Option<String>)> = Vec::new();
    for chain in &config.chains {
        if done.contains(chain) {
            continue;
        }
        let mirror = network::mirror_name(chain)
            .filter(|m| config.chains.contains(m) && !done.contains(m));
        done.insert(chain.clone());
        if let Some(m) = &mirror {
            done.insert(m.clone());
        }
        work.push((chain.clone(), mirror));
    }
    let verbose = std::env::var_os("YUPEI_GPT_VERBOSE").is_some();
    if verbose {
        eprintln!("Computing {} network families in parallel", work.len());
    }
    let timer = Instant::now();
    // One pipeline per family: scan -> family stats -> per-chain posts, so a
    // finished family's post runs while other families still stream scans.
    let posted: Vec<io::Result<Vec<(String, NetworkOutput)>>> = work
        .par_iter()
        .map(|(chain, mirror)| {
            let bundle = network::scan_family(&stocks, &kernel, chain, mirror.is_some())?;
            let fam = network::family_stats_of(&kernel, &bundle);
            let mut out = vec![(
                chain.clone(),
                network::post_family(&stocks, &kernel, &bundle, &fam, chain, false)?,
            )];
            if let Some(m) = mirror {
                out.push((
                    m.clone(),
                    network::post_family(&stocks, &kernel, &bundle, &fam, m, true)?,
                ));
            }
            Ok(out)
        })
        .collect();
    let net_elapsed = timer.elapsed().as_secs_f64();
    if verbose {
        eprintln!("network families finished {net_elapsed:.3}s");
    }
    times.insert("network".into(), net_elapsed);
    let mut outputs: BTreeMap<String, NetworkOutput> = BTreeMap::new();
    for result in posted {
        for (name, out) in result? {
            outputs.insert(name, out);
        }
    }
    for chain in &config.chains {
        let out = outputs
            .remove(chain)
            .expect("every chain produced by its family");
        if ["AAa", "ppp", "mmm"].contains(&chain.as_str()) {
            neighbors.push(out.neighbors);
        }
        columns.extend(out.columns);
        audits.extend(out.audits);
    }
    // Structural pruning, decided from formulas rather than this day's ranks.
    // Effective neighbor count is exactly 1/HHI. Ordered pm/mp pairs have the
    // same target-X aggregate after exchanging the two member roles.
    columns.retain(|col| {
        !col.name.ends_with("_row_gs_hhip")
            && !(col.name.starts_with("yr_mp")
                && ["_e_w_std_soft", "_x0topk20_count", "_gtopk100_count"]
                    .iter()
                    .any(|suffix| col.name.ends_with(suffix)))
    });
    let timer = Instant::now();
    columns.extend(extensions::cross_columns(&stocks, &kernel, &columns));
    drop(kernel);
    columns.extend(extensions::price_columns(&stocks));
    columns.extend(extensions::interaction_columns(&columns, n));
    times.insert("price_and_contrasts".into(), timer.elapsed().as_secs_f64());
    let timer = Instant::now();
    columns.extend(leadlag::leadlag_columns(&stocks)?);
    times.insert("leadlag".into(), timer.elapsed().as_secs_f64());
    times.insert("total_compute".into(), total.elapsed().as_secs_f64());
    let snapshot = Snapshot {
        schema: SCHEMA.into(),
        config_hash: hash,
        date: config.date,
        codes: codes.clone(),
        columns: columns.clone(),
        neighbors,
    };
    let _ = manifest;
    Ok(DailyCore {
        codes,
        columns,
        snapshot,
        calendar,
        timings: times,
        audits,
    })
}

/// 瘦身快照：只留 temporal SPECS 所需基础列 + 三条链宽度列 + 邻居状态。
fn slim_snapshot(snap: &Snapshot) -> Snapshot {
    let keep: std::collections::BTreeSet<&str> = SLIM_BASES.iter().copied().collect();
    Snapshot {
        schema: snap.schema.clone(),
        config_hash: snap.config_hash.clone(),
        date: snap.date,
        codes: snap.codes.clone(),
        columns: snap
            .columns
            .iter()
            .filter(|c| keep.contains(c.name.as_str()))
            .cloned()
            .collect(),
        neighbors: snap.neighbors.clone(),
    }
}

fn snap_path(dir: &std::path::Path, date: i64) -> std::path::PathBuf {
    dir.join(format!("{date}.ygsnap.json"))
}

fn write_snapshot(dir: &std::path::Path, snap: &Snapshot) -> io::Result<()> {
    std::fs::create_dir_all(dir)?;
    let slim = slim_snapshot(snap);
    let tmp = snap_path(dir, snap.date).with_extension("tmp");
    serde_json::to_writer(io::BufWriter::new(std::fs::File::create(&tmp)?), &slim)
        .map_err(|e| input::bad(e.to_string()))?;
    std::fs::rename(&tmp, snap_path(dir, snap.date))?;
    Ok(())
}

/// pipeline 主入口：一天全市场，491 列（无历史列）。顺手把瘦身快照写入
/// snap_dir（若设置），供 yupei_gpt_temporal 第二阶段使用。
pub fn compute_full(date: i64) -> io::Result<(Vec<String>, Vec<f32>)> {
    let config = default_config(date);
    let core = compute_daily(&config)?;
    if let Some(dir) = &config.snap_dir {
        write_snapshot(dir, &core.snapshot)?;
    }
    let mut vals = Vec::with_capacity(core.codes.len() * core.columns.len());
    for i in 0..core.codes.len() {
        for col in &core.columns {
            vals.push(col.values[i].map(|v| v as f32).unwrap_or(f32::NAN));
        }
    }
    Ok((core.codes, vals))
}

fn load_snapshot(path: &std::path::Path) -> io::Result<Snapshot> {
    let data = std::fs::read(path)?;
    serde_json::from_slice(&data)
        .map_err(|e| input::bad(format!("snapshot {}: {e}", path.display())))
}

/// 第二阶段：一天 29 个历史时序列。需要 snap_dir 里当前日与前最多 30 个
/// 交易日的 .ygsnap.json；缺哪些交易日由 temporal 层按 allow_incomplete
/// 规则保留 null（pipeline 恒 true）。
pub fn compute_temporal_full(date: i64) -> io::Result<(Vec<String>, Vec<f32>)> {
    let config = default_config(date);
    let snap_dir = config
        .snap_dir
        .clone()
        .ok_or_else(|| input::bad("YUPEI_GPT_SNAP_DIR must point at the snapshot directory"))?;
    // 只读行业日历与交易日历，不重算网络：先生成一个轻量 Snapshot 壳。
    let cur_path = snap_path(&snap_dir, date);
    if !cur_path.is_file() {
        return Err(input::bad(format!(
            "temporal stage needs {} produced by the main pipeline first",
            cur_path.display()
        )));
    }
    let current: Snapshot = load_snapshot(&cur_path)?;
    let dates = input::calendar(&config.basic_info_dir.join("calendar.csv"), false)?;
    let today = dates
        .iter()
        .position(|&d| d == date)
        .ok_or_else(|| input::bad(format!("date {date} not in trading calendar")))?;
    let mut history = Vec::new();
    for &d in &dates[today.saturating_sub(30)..today] {
        let p = snap_path(&snap_dir, d);
        if p.is_file() {
            let snap = load_snapshot(&p)?;
            if snap.schema == current.schema && snap.config_hash == current.config_hash {
                history.push(snap);
            }
        }
    }
    let (tcols, _audits) =
        temporal::temporal_columns(&current, &history, &dates, config.allow_incomplete_history)?;
    let mut vals = Vec::with_capacity(current.codes.len() * tcols.len());
    for i in 0..current.codes.len() {
        for col in &tcols {
            vals.push(col.values[i].map(|v| v as f32).unwrap_or(f32::NAN));
        }
    }
    Ok((current.codes.clone(), vals))
}
