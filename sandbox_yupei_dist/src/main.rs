//! yupei_dist CLI:
//!   yupei_dist matrices <date> <outdir> [--min-trades N] [--universe N] [--threads T]
//!       读全市场逐笔 → 每股预处理 → 21 张关联-差异矩阵 + 统计量 → 二进制备份
//!   yupei_dist factors <date> <outdir> [--prev-date D] [--threads T]
//!       读备份 → 运行全部降维指标模块 → factors.bin + names.txt + report.txt
//!   yupei_dist all <date> <outdir> [--prev-date D] ...   （matrices + factors, 基准目标）
//!   yupei_dist list                         列出已注册指标模块
//!   yupei_dist verify <date> <outdir>       核对 spec.json 与注册表/备份
//! 并行度: RAYON_NUM_THREADS 环境变量或 --threads（全局线程池）

use rayon::prelude::*;

mod cli_util;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("用法见文件头注释");
        std::process::exit(1);
    }
    match args[1].as_str() {
        "matrices" => cmd_matrices(&args[2..]),
        "factors" => cmd_factors(&args[2..]),
        "all" => cmd_all(&args[2..]),
        "list" => cmd_list(),
        "verify" => cmd_verify(&args[2..]),
        other => {
            eprintln!("未知子命令: {other}");
            std::process::exit(1);
        }
    }
}

fn parse_flag(args: &[String], flag: &str) -> Option<String> {
    args.iter().position(|a| a == flag).map(|i| args[i + 1].clone())
}

fn parse_threads(args: &[String]) {
    if let Some(t) = parse_flag(args, "--threads") {
        let n: usize = t.parse().unwrap_or(50);
        let _ = rayon::ThreadPoolBuilder::new().num_threads(n).build_global();
    } else if let Ok(t) = std::env::var("RAYON_NUM_THREADS") {
        if let Ok(n) = t.parse::<usize>() {
            let _ = rayon::ThreadPoolBuilder::new().num_threads(n).build_global();
        }
    }
}

fn cmd_matrices(args: &[String]) {
    parse_threads(args);
    let date: i64 = args[0].parse().unwrap();
    let outdir = args[1].clone();
    let min_trades: usize = parse_flag(args, "--min-trades").map(|s| s.parse().unwrap()).unwrap_or(200);
    // 默认 universe = 4000: 按文件大小（成交活跃度代理）取前 4000 只,
    // 覆盖 ~96% 全市场逐笔; 是 50 并行 1 分钟预算下的设计参数（spec.json 记录）。
    let universe: Option<usize> = parse_flag(args, "--universe").map(|s| s.parse().unwrap()).or(Some(4000));
    let t_all = std::time::Instant::now();

    // ---- 枚举代码（按文件大小降序 → 可选 universe 截断 → 恢复代码序保证确定性）----
    let mut codes = cli_util::list_codes_by_size(date);
    if let Some(u) = universe {
        codes.truncate(u);
        codes.sort();
    }
    eprintln!("[m1] codes = {}", codes.len());

    let day_start_us = cli_util::day_start_us(date);

    // ---- 并行读 + 每股预处理 ----
    let t0 = std::time::Instant::now();
    let preps: Vec<Option<yupei_dist::matrix_stage::StockPrep>> = codes
        .par_iter()
        .map(|c| {
            yupei_dist::fast_csv_reader::read_trade_fast(c, date)
                .ok()
                .and_then(|recs| yupei_dist::matrix_stage::prep_stock(c, &recs, day_start_us, min_trades))
        })
        .collect();
    let stocks: Vec<yupei_dist::matrix_stage::StockPrep> = preps.into_iter().flatten().collect();
    let n = stocks.len();
    let total_trades: usize = stocks.iter().map(|s| s.n_trades).sum();
    let secs = t0.elapsed().as_secs_f32();
    eprintln!("[m2] valid = {n}, trades = {total_trades}, read+prep {secs:.1}s");
    if n == 0 {
        eprintln!("无有效股票");
        std::process::exit(1);
    }

    // ---- 矩阵计算 ----
    let t0 = std::time::Instant::now();
    let (mats, stats) = yupei_dist::matrix_stage::compute_matrices(&stocks);
    let secs = t0.elapsed().as_secs_f32();
    eprintln!("[m3] matrices 21×{n}×{n} {secs:.1}s");

    // ---- 写备份 ----
    let t0 = std::time::Instant::now();
    let dir = std::path::Path::new(&outdir).join(date.to_string());
    std::fs::create_dir_all(&dir).unwrap();
    let codes_out: Vec<String> = stocks.iter().map(|s| s.code.clone()).collect();
    yupei_dist::matrix_store::write_codes(&dir, &codes_out).unwrap();
    yupei_dist::matrix_store::write_stats(&dir, &stats).unwrap();
    for (m, name) in mats.iter().zip(yupei_dist::matrix_stage::MATRIX_SPECS.iter()) {
        yupei_dist::matrix_store::write_matrix(&dir, name.name, m).unwrap();
    }
    let secs = t0.elapsed().as_secs_f32();
    eprintln!("[m4] write backup {secs:.1}s");
    eprintln!("[matrices] done {:.1}s total, n = {n}", t_all.elapsed().as_secs_f32());
    println!("{{\"ok\":true,\"date\":{date},\"n\":{n},\"outdir\":\"{outdir}\"}}");
}

fn cmd_factors(args: &[String]) {
    parse_threads(args);
    let date: i64 = args[0].parse().unwrap();
    let outdir = args[1].clone();
    let prev_date: Option<i64> = parse_flag(args, "--prev-date").map(|s| s.parse().unwrap());
    let t_all = std::time::Instant::now();

    let ctx = yupei_dist::load_ctx(&outdir, date, prev_date).expect("加载备份失败");
    eprintln!("[f1] ctx loaded: n = {}, mats = {}", ctx.n(), ctx.set.mats.len());

    // 行业文件（可选）
    let ind_path = std::path::Path::new(&outdir).join(date.to_string()).join("industry.bin");
    if ind_path.exists() {
        let inds = yupei_dist::industry::load_industry_bin(&ind_path, ctx.codes()).unwrap_or_default();
        eprintln!("[f1b] industry loaded: {} 已知", inds.iter().filter(|&&v| v >= 0).count());
    }

    let defs = yupei_dist::indicators::all();
    eprintln!("[f2] modules = {}", defs.len());

    let mut all_names: Vec<String> = Vec::new();
    let mut all_vals: Vec<Vec<f32>> = Vec::new();
    let mut report: Vec<String> = Vec::new();
    for def in &defs {
        let t0 = std::time::Instant::now();
        let results = (def.compute)(&ctx);
        let dt = t0.elapsed().as_secs_f32();
        let nf = results.len();
        for r in results {
            if r.values.len() != ctx.n() {
                eprintln!("!! {}.{} 长度错误: {} != {}", def.name, r.name, r.values.len(), ctx.n());
                std::process::exit(1);
            }
            all_names.push(r.name);
            all_vals.push(r.values);
        }
        report.push(format!("{:<28} {dt:7.3}s  {nf} factors", def.name));
        eprintln!("{:<28} {dt:7.3}s  {nf} factors", def.name);
    }

    // 组装 N×F
    let n = ctx.n();
    let nf = all_names.len();
    let mut vals = vec![0.0f32; n * nf];
    for (fi, col) in all_vals.iter().enumerate() {
        for i in 0..n {
            vals[i * nf + fi] = col[i];
        }
    }

    let dir = std::path::Path::new(&outdir).join(date.to_string());
    yupei_dist::matrix_store::write_factors(&dir, ctx.codes(), &all_names, &vals).unwrap();
    let rep = report.join("\n");
    std::fs::write(dir.join("report.txt"), format!("{rep}\n")).unwrap();
    let secs = t_all.elapsed().as_secs_f32();
    eprintln!("[factors] done {secs:.1}s total, n_factors = {nf}");
    println!("{{\"ok\":true,\"date\":{date},\"n\":{n},\"n_factors\":{nf},\"outdir\":\"{outdir}\"}}");
}

fn cmd_all(args: &[String]) {
    let date: i64 = args[0].parse().unwrap();
    let outdir = args[1].clone();
    // 去掉 date/outdir 后把其余参数传给两个子命令
    let rest: Vec<String> = args[2..].to_vec();
    cmd_matrices(&[date.to_string(), outdir.clone()].into_iter().chain(rest.iter().cloned()).collect::<Vec<_>>());
    let rest2: Vec<String> = args[2..].to_vec();
    cmd_factors(&[date.to_string(), outdir.clone()].into_iter().chain(rest2.iter().cloned()).collect::<Vec<_>>());
}

fn cmd_list() {
    let defs = yupei_dist::indicators::all();
    println!("共 {} 个指标模块:", defs.len());
    for d in &defs {
        println!("  {}", d.name);
    }
}

fn cmd_verify(args: &[String]) {
    let date: i64 = args[0].parse().unwrap();
    let outdir = args[1].clone();
    // 1) 备份完整性
    let dir = std::path::Path::new(&outdir).join(date.to_string());
    let mut ok = true;
    for name in yupei_dist::matrix_store::expected_matrix_names() {
        let p = dir.join("mats").join(format!("{name}.bin"));
        if !p.exists() {
            eprintln!("缺失矩阵: {name}");
            ok = false;
        }
    }
    // 2) 注册模块
    let defs = yupei_dist::indicators::all();
    println!("已注册模块 ({})", defs.len());
    // 3) spec.json 对比
    let spec_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("spec.json");
    if spec_path.exists() {
        let spec: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&spec_path).unwrap()).unwrap();
        let spec_mods = spec["modules"].as_array().map(|a| a.iter().map(|m| m["module"].as_str().unwrap().to_string()).collect::<Vec<_>>()).unwrap_or_default();
        let have: Vec<String> = defs.iter().map(|d| d.name.to_string()).collect();
        for m in &spec_mods {
            if !have.contains(m) {
                eprintln!("spec 要求但未注册: {m}");
                ok = false;
            }
        }
        for m in &have {
            if !spec_mods.contains(m) {
                eprintln!("已注册但不在 spec: {m}");
            }
        }
    } else {
        eprintln!("warning: spec.json 不存在");
    }
    println!("verify {}", if ok { "OK" } else { "FAIL" });
}
