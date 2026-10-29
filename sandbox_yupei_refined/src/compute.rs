use crate::{extensions, input, kernel, leadlag, model::*, network, temporal};
use rayon::prelude::*;
use serde::Serialize;
use std::{collections::BTreeMap, io, path::Path, time::Instant};

pub const SCHEMA: &str = "yupei-refined-v1";
pub const CHAINS: [&str; 15] = [
    "AAa", "AAp", "AAm", "ppa", "ppp", "ppm", "pma", "pmp", "pmm", "mpa", "mpp", "mpm", "mma",
    "mmp", "mmm",
];
#[derive(Serialize)]
pub struct ResultData {
    pub schema: String,
    pub date: i64,
    pub codes: Vec<String>,
    pub columns: Vec<FactorColumn>,
    pub manifest: input::InputManifest,
    pub audits: Vec<AuditRow>,
    pub timings_seconds: BTreeMap<String, f64>,
    pub source_sha256: String,
    pub config_hash: String,
    pub history_dates: Vec<i64>,
    pub snapshot_path: String,
    pub notes: Vec<String>,
}

pub fn source_hash() -> String {
    input::sha256(
        concat!(
            include_str!("model.rs"),
            include_str!("input.rs"),
            include_str!("kernel.rs"),
            include_str!("network.rs"),
            include_str!("network_simd.rs"),
            include_str!("network_stats.rs"),
            include_str!("extensions.rs"),
            include_str!("temporal.rs"),
            include_str!("leadlag.rs"),
            include_str!("compute.rs"),
            include_str!("../Cargo.toml")
        )
        .as_bytes(),
    )
}

pub fn compute_yupei_refined(config: &Config) -> io::Result<(ResultData, Snapshot)> {
    if config.jobs == 0 {
        return Err(input::bad("jobs must be positive"));
    }
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
    if config.history.is_empty() && !config.allow_incomplete_history {
        return Err(input::bad(
            "explicit history required; use allow_incomplete_history for daily exploration",
        ));
    }
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(config.jobs)
        .build()
        .map_err(|e| input::bad(e.to_string()))?;
    pool.install(|| compute(config))
}
fn compute(config: &Config) -> io::Result<(ResultData, Snapshot)> {
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
    let source = source_hash();
    let mut canonical_codes = config.codes.clone();
    if let Some(codes) = &mut canonical_codes {
        codes.sort();
    }
    let mut canonical_chains = config.chains.clone();
    canonical_chains.sort();
    let hash = input::sha256(
        format!(
            "{SCHEMA}|{source}|{}|{}|{:?}|{:?}|{:?}|{:?}|tau=1;slot=10ms;continuous=237min",
            manifest.whitelist_sha256,
            manifest.industry_symbol_sha256,
            canonical_codes,
            canonical_chains,
            config.level2_root.canonicalize()?,
            config.vars_dir.canonicalize()?
        )
        .as_bytes(),
    );
    let mut history = Vec::new();
    for path in &config.history {
        let data = std::fs::read(path)?;
        let snap: Snapshot = serde_json::from_slice(&data)
            .map_err(|e| input::bad(format!("snapshot {}: {e}", path.display())))?;
        if snap.schema != SCHEMA || snap.config_hash != hash || snap.date >= config.date {
            return Err(input::bad(format!(
                "incompatible or future snapshot {}",
                path.display()
            )));
        }
        history.push(snap);
    }
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
    // Scans run in parallel across families; post-processing is parallel too.
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
    let post_elapsed = timer.elapsed().as_secs_f64();
    if verbose {
        eprintln!("network families finished {post_elapsed:.3}s");
    }
    let scan_elapsed = post_elapsed;
    for chain in &config.chains {
        times.insert(format!("network_{chain}"), (scan_elapsed + post_elapsed) / 15.0);
    }
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
    let snapshot = Snapshot {
        schema: SCHEMA.into(),
        config_hash: hash.clone(),
        date: config.date,
        codes: codes.clone(),
        columns: columns.clone(),
        neighbors,
    };
    let timer = Instant::now();
    let (tcols, taudits) = temporal::temporal_columns(
        &snapshot,
        &history,
        &calendar,
        config.allow_incomplete_history,
    )?;
    columns.extend(tcols);
    audits.extend(taudits);
    times.insert("temporal".into(), timer.elapsed().as_secs_f64());
    // Direct marginal comparison against explicit activity controls, finite
    // intersections only. Constant/all-missing columns are undefined, not zero.
    for col in &columns {
        let v: Vec<_> = col.values.iter().map(|x| x.unwrap_or(f64::NAN)).collect();
        let valid = col.values.iter().flatten().count();
        let mut best: Option<(f64, usize, String)> = None;
        for control in columns.iter().filter(|c| c.name.starts_with("yr_control_")) {
            if control.name == col.name {
                continue;
            }
            let other: Vec<_> = control
                .values
                .iter()
                .map(|x| x.unwrap_or(f64::NAN))
                .collect();
            let (count, corr) = extensions::spearman(&v, &other);
            if let Some(corr) = corr {
                if best.as_ref().is_none_or(|b| corr.abs() > b.0.abs()) {
                    best = Some((corr, count, control.name.clone()));
                }
            }
        }
        audits.push(AuditRow{name:col.name.clone(),layer:"coverage_and_max_control_rankcorr".into(),n:valid,value:best.as_ref().map(|b|b.0),detail:match best{Some((_,overlap,name))=>format!("largest absolute signed correlation vs {name}; common stocks={overlap}; {} total stocks",n),None=>"undefined: fewer than 3 common finite values or zero rank variance".into()}});
    }
    let undefined_corr: Vec<_> = audits
        .iter()
        .filter(|a| {
            a.value.is_none()
                && ["cross_section", "coverage_and_max_control_rankcorr"]
                    .contains(&a.layer.as_str())
        })
        .collect();
    if !undefined_corr.is_empty() {
        if !config.allow_uncomputable_corr {
            return Err(input::bad(format!("{} diagnostic correlations are undefined (first={} common/valid={}): {}. Set allow_uncomputable_corr=true / --allow-uncomputable-corr to preserve nulls and audited warnings; never zero.",undefined_corr.len(),undefined_corr[0].name,undefined_corr[0].n,undefined_corr[0].detail)));
        }
        eprintln!("WARNING: explicitly allowed {} undefined diagnostic correlations; each remains null with reason and valid counts in audits",undefined_corr.len());
    }
    let undefined_count = undefined_corr.len();
    audits.push(AuditRow{name:"allow_uncomputable_corr".into(),layer:"explicit_policy".into(),n:undefined_count,value:Some(if config.allow_uncomputable_corr{1.0}else{0.0}),detail:"Undefined correlations are never zero and never eligible for ranking; each affected item has a separate audit row.".into()});
    times.insert("total_compute".into(), total.elapsed().as_secs_f64());
    let result=ResultData{schema:SCHEMA.into(),date:config.date,codes,columns,manifest,audits,timings_seconds:times,source_sha256:source,config_hash:hash,history_dates:history.iter().map(|s|s.date).collect(),snapshot_path:String::new(),notes:vec![
        "Exploration only: no task submission, backtest, installed-library mutation, or inferred predictive performance.".into(),
        "10ms event-count kernel; sessions [09:30,11:30),[13:00,14:57); mixed directions correctly normalized and own/partner roles explicit.".into(),
        "yr_ is a new definition namespace. Exact all-BC/all-X pairs; same-direction target counts use ordered pairs (twice legacy unordered counts).".into(),
        "Price response uses later same-day same-session minute closes, industry ex-self reference, >=12 qualifying minutes per side; it is available only after session close.".into(),
        "History is supplied explicitly; incomplete windows remain null. A single-day run cannot establish IC or strategy returns.".into()]};
    Ok((result, snapshot))
}

pub fn write_result(config: &Config, path: &Path) -> io::Result<()> {
    let (mut result, snapshot) = compute_yupei_refined(config)?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let snapshot_path = path.with_extension("snapshot.json");
    result.snapshot_path = snapshot_path.display().to_string();
    serde_json::to_writer(
        std::io::BufWriter::new(std::fs::File::create(&snapshot_path)?),
        &snapshot,
    )
    .map_err(|e| input::bad(e.to_string()))?;
    serde_json::to_writer(
        std::io::BufWriter::new(std::fs::File::create(path)?),
        &result,
    )
    .map_err(|e| input::bad(e.to_string()))?;
    eprintln!(
        "Saved {} stocks x {} columns to {}",
        result.codes.len(),
        result.columns.len(),
        path.display()
    );
    Ok(())
}
