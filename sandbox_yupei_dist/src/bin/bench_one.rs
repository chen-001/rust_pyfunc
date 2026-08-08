//! bench_one <module> <date> <outdir> [--prev-date D] [--threads T]
//! 单独计时某个指标模块（指标开发/优化用）; 输出各因子统计与耗时。

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        eprintln!("用法: bench_one <module> <date> <outdir> [--prev-date D] [--threads T]");
        std::process::exit(1);
    }
    let module = args[1].clone();
    let date: i64 = args[2].parse().unwrap();
    let outdir = args[3].clone();
    let prev_date: Option<i64> = args.iter().position(|a| a == "--prev-date").map(|i| args[i + 1].parse().unwrap());
    if let Some(t) = args.iter().position(|a| a == "--threads") {
        let n: usize = args[t + 1].parse().unwrap();
        let _ = rayon::ThreadPoolBuilder::new().num_threads(n).build_global();
    }

    let def = yupei_dist::indicators::by_name(&module).unwrap_or_else(|| {
        eprintln!("模块不存在: {module}");
        std::process::exit(1);
    });
    let ctx = yupei_dist::load_ctx(&outdir, date, prev_date).expect("加载备份失败");
    eprintln!("ctx: n = {}, mats = {}, prev = {}", ctx.n(), ctx.set.mats.len(), ctx.prev.is_some());

    // 预热（mmap 页缓存 + 缓存对称化）
    let _ = (def.compute)(&ctx);

    let t0 = std::time::Instant::now();
    let results = (def.compute)(&ctx);
    let dt = t0.elapsed().as_secs_f32();
    println!("模块 {module}: {dt:.3}s");
    for r in results {
        let v = &r.values;
        let n = v.len();
        let mut finite = 0usize;
        let (mut s, mut s2) = (0.0f64, 0.0f64);
        let mut mn = f64::INFINITY;
        let mut mx = f64::NEG_INFINITY;
        let mut nan = 0usize;
        for &x in v.iter() {
            if x.is_finite() {
                finite += 1;
                s += x as f64;
                s2 += x as f64 * x as f64;
                mn = mn.min(x as f64);
                mx = mx.max(x as f64);
            } else {
                nan += 1;
            }
        }
        let mean = if finite > 0 { s / finite as f64 } else { f64::NAN };
        let sd = if finite > 1 { ((s2 - s * s / finite as f64) / (finite - 1) as f64).sqrt() } else { f64::NAN };
        let zero = v.iter().filter(|&&x| x == 0.0).count();
        println!("  {}: n={} finite={} nan={} zero={} mean={:.4} sd={:.4} min={:.4} max={:.4}",
                 r.name, n, finite, nan, zero, mean, sd, mn, mx);
    }
}
