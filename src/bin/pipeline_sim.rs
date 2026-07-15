//! 流水线调度模拟：验证"多进程异步循环 + 错峰 + 计算占位"能否避免磁盘争抢。
//!
//! 模拟用户的真实场景：每个进程实例 = 一个 worker，循环处理多个日期，
//! 每天执行 [读逐笔 → 读盘口 → busy计算]。多实例由 bash 并发启动。
//!
//! argv[1] = 日期列表（逗号分隔），如 "20240126,20240223"
//! argv[2] = 线程数（默认 20）
//! argv[3] = compute_sec（每天计算占位秒数，默认 0）
//!
//! 关键输出：每天的"读取 wall"——若回到单进程水平(~8s)说明无争抢。
use rust_pyfunc::fast_csv_reader::{read_market_fast_inner, read_trade_fast_inner};

use rayon::prelude::*;
use std::collections::BTreeSet;
use std::fs;

fn list_codes(date: i64, subdir: &str) -> Vec<String> {
    let dir = format!("/ssd_data/stock/{date}/{subdir}");
    let mut set = BTreeSet::new();
    if let Ok(entries) = fs::read_dir(&dir) {
        for e in entries.flatten() {
            let name = e.file_name().into_string().unwrap_or_default();
            if let Some(code) = name.split('_').next() {
                if code.chars().all(|c| c.is_ascii_digit()) {
                    set.insert(code.to_string());
                }
            }
        }
    }
    set.into_iter().collect()
}

/// busy loop 模拟 CPU 密集计算（不用 sleep，否则不占 CPU 核）。
fn busy_compute(sec: f64) {
    let target = std::time::Duration::from_secs_f64(sec);
    let start = std::time::Instant::now();
    let mut x: u64 = 12345;
    while start.elapsed() < target {
        // 伪随机乘法，防止编译器优化掉
        x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    }
    std::hint::black_box(x);
}

fn main() {
    let dates: Vec<i64> = std::env::args()
        .nth(1)
        .unwrap_or_default()
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    let n_threads: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(20);
    let compute_sec: f64 = std::env::args()
        .nth(3)
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.0);

    rayon::ThreadPoolBuilder::new()
        .num_threads(n_threads)
        .build_global()
        .unwrap();

    let pid = std::process::id();
    eprintln!("[pid{pid}] 线程={n_threads} compute={compute_sec}s 日期数={}", dates.len());

    let total_start = std::time::Instant::now();
    let mut sum_read = 0.0f64;
    let mut n_trades_total = 0usize;

    for &date in &dates {
        // 取股票并集
        let mut codes: BTreeSet<String> = BTreeSet::new();
        for c in list_codes(date, "transaction") {
            codes.insert(c);
        }
        for c in list_codes(date, "market_data") {
            codes.insert(c);
        }
        let codes: Vec<String> = codes.into_iter().collect();

        // 读逐笔
        let t = std::time::Instant::now();
        let trades: Vec<Vec<_>> = codes
            .par_iter()
            .map(|c| read_trade_fast_inner(c, date, false, true, usize::MAX).unwrap_or_default())
            .collect();
        let r_wall = t.elapsed().as_secs_f64();
        let n_trades: usize = trades.iter().map(|v| v.len()).sum();

        // 读盘口
        let t = std::time::Instant::now();
        let md: Vec<Vec<_>> = codes
            .par_iter()
            .map(|c| read_market_fast_inner(c, date, false, true, usize::MAX).unwrap_or_default())
            .collect();
        let m_wall = t.elapsed().as_secs_f64();

        sum_read += r_wall + m_wall;
        n_trades_total += n_trades;
        eprintln!(
            "[pid{pid}] {date}: 读逐笔 {r_wall:.1}s + 读盘口 {m_wall:.1}s = {rw:.1}s | 逐笔{n_trades}条",
            rw = r_wall + m_wall
        );

        // busy 计算（模拟因子计算，占 CPU）
        if compute_sec > 0.0 {
            busy_compute(compute_sec);
        }

        // 显式 drop，释放内存（循环多天不累积）
        drop(trades);
        drop(md);
    }

    let total = total_start.elapsed().as_secs_f64();
    eprintln!(
        "[pid{pid}] ✅ 完成: 总用时 {total:.1}s, 纯读取 {sum_read:.1}s ({pct:.0}%), 逐笔共 {n_trades_total} 条",
        pct = sum_read / total * 100.0
    );
}
