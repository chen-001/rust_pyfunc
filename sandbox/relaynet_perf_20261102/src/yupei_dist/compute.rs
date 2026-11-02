pub const MAX_THREADS: usize = 200;

/// 在 200 线程的局部池里跑 f。外部已设 RAYON_NUM_THREADS 时尊重外部设置
/// （pipeline worker 由启动方设 RAYON_NUM_THREADS = n_jobs/n_workers, 走这条路）。
/// 用局部池而不是 build_global：全局池只建一次, 进程里别的函数先建过就会静默失效。
pub fn with_compute_pool<T: Send>(f: impl FnOnce() -> T + Send) -> std::io::Result<T> {
    if std::env::var("RAYON_NUM_THREADS").is_ok() {
        return Ok(f());
    }
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(MAX_THREADS)
        .build()
        .map_err(|e| {
            std::io::Error::new(std::io::ErrorKind::Other, format!("线程池创建失败: {e}"))
        })?;
    Ok(pool.install(f))
}


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
