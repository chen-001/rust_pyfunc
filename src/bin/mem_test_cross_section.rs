//! 内存测试：用 Rust 读取一整天全市场逐笔 + 盘口，测量峰值内存占用。
//!
//! 模拟横截面 pipeline 的真实场景：
//!   - 200 线程 rayon 池
//!   - 并行读取一天所有股票的 transaction + market_data
//!   - 两个数据集同时驻留内存（横截面计算需要）
//!
//! parallel_threshold = usize::MAX → 单文件走 read_to_string（纯堆内存，无 mmap 干扰）
//! 这与 pipeline_order_pair_hm90 实际调用的方式一致。
use rust_pyfunc::fast_csv_reader::{read_market_fast_inner, read_trade_fast_inner};

use rayon::prelude::*;
use std::collections::BTreeSet;
use std::fs;

const KB: f64 = 1024.0;
const GB_KB: f64 = 1024.0 * 1024.0; // KB → GiB
const CLK_TCK: f64 = 100.0;

/// 进程累计 CPU 时间（user+sys，秒）。读 /proc/self/stat 字段14,15。
fn cpu_time_secs() -> f64 {
    let stat = fs::read_to_string("/proc/self/stat").unwrap_or_default();
    let after_comm = stat.rsplit_once(')').map(|(_, s)| s).unwrap_or("");
    let f: Vec<&str> = after_comm.split_whitespace().collect();
    // after_comm 索引: 0=state(字段3) ... 11=utime(字段14) 12=stime(字段15)
    let utime: u64 = f.get(11).and_then(|x| x.parse().ok()).unwrap_or(0);
    let stime: u64 = f.get(12).and_then(|x| x.parse().ok()).unwrap_or(0);
    (utime + stime) as f64 / CLK_TCK
}

fn vm_field(name: &str) -> u64 {
    let s = fs::read_to_string("/proc/self/status").unwrap_or_default();
    for line in s.lines() {
        if let Some(rest) = line.strip_prefix(name) {
            return rest
                .split_whitespace()
                .next()
                .and_then(|x| x.parse().ok())
                .unwrap_or(0);
        }
    }
    0
}

/// 列出某天某子目录下的全部股票代码（从文件名提取）。
fn list_codes(date: i64, subdir: &str) -> Vec<String> {
    let dir = format!("/ssd_data/stock/{date}/{subdir}");
    let mut set = BTreeSet::new();
    if let Ok(entries) = fs::read_dir(&dir) {
        for e in entries.flatten() {
            let name = e.file_name().into_string().unwrap_or_default();
            // 文件名格式: 000001_20251231_transaction.csv
            if let Some(code) = name.split('_').next() {
                if code.chars().all(|c| c.is_ascii_digit()) {
                    set.insert(code.to_string());
                }
            }
        }
    }
    set.into_iter().collect()
}

fn main() {
    // 线程数：argv[2]，默认 200
    let n_threads: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(200);
    rayon::ThreadPoolBuilder::new()
        .num_threads(n_threads)
        .build_global()
        .unwrap();

    let date: i64 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(20251231);

    // 取 transaction + market_data 股票代码并集（最全）
    let mut codes: BTreeSet<String> = BTreeSet::new();
    for c in list_codes(date, "transaction") {
        codes.insert(c);
    }
    for c in list_codes(date, "market_data") {
        codes.insert(c);
    }
    let codes: Vec<String> = codes.into_iter().collect();
    println!("日期 {date}，线程数 {n_threads}，股票代码并集 {} 只", codes.len());

    let base_rss = vm_field("VmRSS:");
    let base_hwm = vm_field("VmHWM:");
    println!("baseline VmRSS = {:.2} GiB, VmHWM = {:.2} GiB\n", base_rss as f64 / GB_KB, base_hwm as f64 / GB_KB);

    // ============ 阶段 1：并行读全部逐笔成交 ============
    let t0 = std::time::Instant::now();
    let cpu0 = cpu_time_secs();
    let all_trades: Vec<Vec<_>> = codes
        .par_iter()
        .map(|code| {
            read_trade_fast_inner(code, date, false, true, usize::MAX).unwrap_or_default()
        })
        .collect();
    let n_trades: usize = all_trades.iter().map(|v| v.len()).sum();
    let ok_trades = all_trades.iter().filter(|v| !v.is_empty()).count();
    let wall_trades = t0.elapsed().as_secs_f64();
    let cpu_trades = cpu_time_secs() - cpu0;
    let rss1 = vm_field("VmRSS:");
    let hwm1 = vm_field("VmHWM:");
    let net_trades_gib = n_trades as f64 * 40.0 / (KB * KB * KB);
    println!("【阶段1 逐笔 transaction】");
    println!("  记录数: {n_trades} 条 ({ok_trades}/{}) 只有效)", codes.len());
    println!("  wall: {:.2}s  |  CPU(user+sys, {n_threads}核): {:.1}s  |  CPU利用率: {:.0}%",
        wall_trades, cpu_trades, cpu_trades / wall_trades / n_threads as f64 * 100.0);
    println!("  结构体净内存: {net_trades_gib:.2} GiB ({n_trades} × 40B)");
    println!("  VmRSS: {:.2} GiB (较 baseline +{:.2} GiB)", rss1 as f64 / GB_KB, (rss1 - base_rss) as f64 / GB_KB);
    println!("  VmHWM: {:.2} GiB (较 baseline +{:.2} GiB)\n", hwm1 as f64 / GB_KB, (hwm1 - base_hwm) as f64 / GB_KB);

    // ============ 阶段 2：逐笔仍持有，再并行读全部盘口 ============
    let t1 = std::time::Instant::now();
    let cpu1 = cpu_time_secs();
    let all_md: Vec<Vec<_>> = codes
        .par_iter()
        .map(|code| {
            read_market_fast_inner(code, date, false, true, usize::MAX).unwrap_or_default()
        })
        .collect();
    let n_md: usize = all_md.iter().map(|v| v.len()).sum();
    let ok_md = all_md.iter().filter(|v| !v.is_empty()).count();
    let wall_md = t1.elapsed().as_secs_f64();
    let cpu_md = cpu_time_secs() - cpu1;
    let rss2 = vm_field("VmRSS:");
    let hwm2 = vm_field("VmHWM:");
    let net_md_gib = n_md as f64 * 184.0 / (KB * KB * KB);
    println!("【阶段2 盘口 market_data（逐笔仍持有）】");
    println!("  记录数: {n_md} 条 ({ok_md} 只有效)");
    println!("  wall: {:.2}s  |  CPU(user+sys, {n_threads}核): {:.1}s  |  CPU利用率: {:.0}%",
        wall_md, cpu_md, cpu_md / wall_md / n_threads as f64 * 100.0);
    println!("  结构体净内存: {net_md_gib:.2} GiB ({n_md} × 184B)");
    println!("  VmRSS: {:.2} GiB (较 baseline +{:.2} GiB)", rss2 as f64 / GB_KB, (rss2 - base_rss) as f64 / GB_KB);
    println!("  VmHWM: {:.2} GiB (较 baseline +{:.2} GiB)\n", hwm2 as f64 / GB_KB, (hwm2 - base_hwm) as f64 / GB_KB);

    // ============ 汇总 ============
    let net_total = (n_trades * 40 + n_md * 184) as f64 / (KB * KB * KB);
    println!("================= 汇总（逐笔 + 盘口全量驻留）=================");
    println!("总记录: 逐笔 {n_trades} + 盘口 {n_md} = {} 条", n_trades + n_md);
    println!("结构体净内存合计: {net_total:.2} GiB");
    println!("实测 VmRSS: {:.2} GiB", rss2 as f64 / GB_KB);
    println!("实测峰值 VmHWM: {:.2} GiB", hwm2 as f64 / GB_KB);
    println!("=============================================================");
}
