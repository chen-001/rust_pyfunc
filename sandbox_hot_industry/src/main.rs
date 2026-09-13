//! 同热点股票池「行业拓展」单日全市场探索（sandbox）。
//!
//! 只实现 x15y10_ba 一个参数组合（与正式库 PARAM_CONFIGS[3] 完全一致）：
//! - x=15 秒窗口，D = 窗口内 (bid_order - ask_order) 的均值（每秒先算 per-sec 均值，再对 15 秒求均值；
//!   空秒 bid_ask_mean 记 0.0，与正式库 RollingCache 行为一致）
//! - 每 2 秒采样一次；历史窗口 120 秒（最多 60 个 2 秒采样点，含当前值）
//! - z-score = (D - 历史均值) / 历史标准差；z > +1.5 → 热点组，z < -1.5 → 冰点组
//! - 要求窗口成交笔数 >= 10（14 秒滚动窗口 trade_cnt 合计）
//!
//! 输出（stdout JSON，日志走 stderr）：
//! - 每股：hot_cnt / cold_cnt（2 秒级入选次数）
//! - 行业内相对位置特征：每次热点/冰点入选时，该股在「组内同行业子集」中的
//!   主买占比(buy15) / 成交量(vol15) / 盘口差(D) 排名百分位均值 与 z 分均值
//! - 共现 top10 同伴（同框次数）：hot_peers / cold_peers
//!
//! 用法: ./target/release/hot_industry_sandbox <date> <industry_csv> [--no-cooc] [--dump-steps]
//!   industry_csv: 每行 "6位代码,行业号(1..31,0=未知)"

mod fast_csv_reader;

use fast_csv_reader::read_trade_fast;
use rayon::prelude::*;
use serde::Serialize;
use std::collections::BTreeSet;

// ---------------- 与正式库一致的常量 ----------------
const ADJUSTED_SECONDS: usize = 14221;
const SEC_OFFSET: i64 = 9 * 3600 + 30 * 60; // 34200
const MORNING_END: i64 = 11 * 3600 + 30 * 60; // 41400
const AFTERNOON_START: i64 = MORNING_END + 1; // 41401
const AFTERNOON_END: i64 = MORNING_END + (14 * 3600 + 57 * 60 - 13 * 3600); // 48420
const STEP: usize = 2;
const WIN: usize = 15;
const HIST_WIN: usize = 120; // 秒
const Z_THRESH: f64 = 1.5;
const MIN_TRADES: u32 = 10;
const K_PEERS: usize = 10;

/// 2 秒采样步数：sec = 15, 17, ..., 14219
const N_STEPS: usize = (ADJUSTED_SECONDS - 15) / STEP;

// ---------------- 时间工具（与正式库 sec_to_idx / cst_midnight_epoch 一致） ----------------
fn days_from_civil(y: i64, m: u32, d: u32) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400;
    let mp = (m as i64 + 9) % 12;
    let doy = (153 * mp + 2) / 5 + d as i64 - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146097 + doe - 719468
}

fn cst_midnight_epoch(date: i64) -> i64 {
    days_from_civil(date / 10000, ((date / 100) % 100) as u32, (date % 100) as u32) * 86400
}

fn sec_to_idx(epoch: f32, day_mid: i64) -> Option<usize> {
    let e = (epoch as i64) - day_mid;
    if e < SEC_OFFSET || e > AFTERNOON_END {
        return None;
    }
    if e <= MORNING_END {
        Some((e - SEC_OFFSET) as usize)
    } else {
        Some((MORNING_END - SEC_OFFSET + 1 + e - AFTERNOON_START) as usize)
    }
}

// ---------------- 数据 ----------------
struct StockVals {
    code: String,
    ind: u8, // 1..31 申万一级行业, 0 = 未知
    d15: Vec<f32>,   // [N_STEPS] 15s 窗口 bid_ask 均值
    buy15: Vec<f32>, // [N_STEPS] 15s 窗口主买占比
    vol15: Vec<f32>, // [N_STEPS] 15s 窗口成交量合计
    cnt15: Vec<u32>, // [N_STEPS] 15s 窗口成交笔数
    hot: Vec<u8>,    // [N_STEPS] 热/冰 标记
    cold: Vec<u8>,
}

fn list_codes(date: i64) -> Vec<String> {
    let dir = format!("/ssd_data/stock/{date}/transaction");
    let mut set = BTreeSet::new();
    if let Ok(entries) = std::fs::read_dir(&dir) {
        for e in entries.flatten() {
            let name = e.file_name().into_string().unwrap_or_default();
            if let Some(code) = name.split('_').next() {
                if code.bytes().all(|b| b.is_ascii_digit()) {
                    set.insert(code.to_string());
                }
            }
        }
    }
    set.into_iter().collect()
}

/// 读行业映射文件（6位代码,行业号）
fn load_industry(path: &str) -> std::collections::HashMap<String, u8> {
    let mut m = std::collections::HashMap::new();
    if let Ok(content) = std::fs::read_to_string(path) {
        for line in content.lines().skip(1) {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            if let Some((code, ind)) = line.split_once(',') {
                if let Ok(v) = ind.trim().parse::<i32>() {
                    m.insert(code.trim().to_string(), v.clamp(0, 31) as u8);
                }
            }
        }
    }
    m
}

/// 单股：逐笔 → 每秒聚合 → 15s 滚动 → 120s z-score → 热/冰标记。
/// 逻辑与正式库 build_stock_data + RollingCache::compute + 方向二 z-score 完全一致。
/// 默认 mode=production：秒索引与正式库完全一致（time_sec 经 f32 量化，128s 网格）；
/// mode=precise 时用微秒精确整数秒（文档语义的"15 秒窗口"）。
fn per_stock(code: &str, date: i64, ind: u8, precise: bool) -> Option<StockVals> {
    let trades = read_trade_fast(code, date).ok()?;
    if trades.is_empty() {
        return None;
    }
    let day_mid = cst_midnight_epoch(date);
    let n_sec = ADJUSTED_SECONDS;

    let mut buy_vol = vec![0.0f64; n_sec];
    let mut tot_vol = vec![0.0f64; n_sec];
    let mut ba_sum = vec![0.0f64; n_sec];
    let mut ba_cnt = vec![0u32; n_sec];

    for t in &trades {
        // 与正式库一致的秒格：time_us/1e6 取整秒 → f32（128s 网格量化）→ 再减 day_mid
        let e = if precise {
            (t.time_us / 1_000_000) - day_mid
        } else {
            let sec_f = (t.time_us / 1_000_000) as f64;
            ((sec_f as f32) as i64) - day_mid
        };
        if e < SEC_OFFSET || e > AFTERNOON_END {
            continue;
        }
        let idx = if e <= MORNING_END {
            (e - SEC_OFFSET) as usize
        } else {
            (MORNING_END - SEC_OFFSET + 1 + e - AFTERNOON_START) as usize
        };
        let v = t.volume;
        tot_vol[idx] += v;
        if t.flag == 66 {
            buy_vol[idx] += v;
        }
        ba_sum[idx] += (t.bid_order - t.ask_order) as f64;
        ba_cnt[idx] += 1;
    }

    // per-sec bid_ask 均值（空秒 = 0.0，与正式库 SecStat::default 一致）
    let mut ba_sec = vec![0.0f32; n_sec];
    for i in 0..n_sec {
        if ba_cnt[i] > 0 {
            ba_sec[i] = (ba_sum[i] / ba_cnt[i] as f64) as f32;
        }
    }

    let mut d15 = vec![f32::NAN; N_STEPS];
    let mut buy15 = vec![f32::NAN; N_STEPS];
    let mut vol15 = vec![f32::NAN; N_STEPS];
    let mut cnt15 = vec![0u32; N_STEPS];
    let mut hot = vec![0u8; N_STEPS];
    let mut cold = vec![0u8; N_STEPS];

    // 逐秒精确滚动（窗口 [sec-14, sec]），仅存 2 秒网格点；与正式库 RollingCache 完全一致
    let mut s_buy = 0.0f64;
    let mut s_vol = 0.0f64;
    let mut s_ba = 0.0f64;
    let mut s_cnt: u32 = 0;
    let mut hist: std::collections::VecDeque<f32> =
        std::collections::VecDeque::with_capacity(HIST_WIN / STEP + 1);

    for sec in 0..ADJUSTED_SECONDS {
        s_buy += buy_vol[sec];
        s_vol += tot_vol[sec];
        s_ba += ba_sec[sec] as f64;
        s_cnt += ba_cnt[sec];
        if sec >= WIN {
            let old = sec - WIN;
            s_buy -= buy_vol[old];
            s_vol -= tot_vol[old];
            s_ba -= ba_sec[old] as f64;
            s_cnt -= ba_cnt[old];
        }
        if sec >= 15 && (sec - 15) % STEP == 0 {
            let i = (sec - 15) / STEP;
            if i >= N_STEPS {
                break;
            }
            d15[i] = (s_ba / WIN as f64) as f32;
            buy15[i] = if s_vol > 0.0 {
                (s_buy / s_vol) as f32
            } else {
                f32::NAN
            };
            vol15[i] = s_vol as f32;
            cnt15[i] = s_cnt;

            // z-score（与正式库一致：先 push 当前值到历史，再算 z）
            let dv = d15[i];
            if dv.is_finite() {
                hist.push_back(dv);
                while hist.len() > HIST_WIN / STEP {
                    hist.pop_front();
                }
            }
            if !dv.is_finite() || cnt15[i] < MIN_TRADES {
                continue;
            }
            if hist.len() < 5 {
                continue;
            }
            let hmean: f64 = hist.iter().map(|v| *v as f64).sum::<f64>() / hist.len() as f64;
            let hvar: f64 = hist
                .iter()
                .map(|v| (*v as f64 - hmean).powi(2))
                .sum::<f64>()
                / hist.len() as f64;
            let hstd = hvar.sqrt();
            if hstd < 1e-8 {
                continue;
            }
            let z = (dv as f64 - hmean) / hstd;
            if z > Z_THRESH {
                hot[i] = 1;
            } else if z < -Z_THRESH {
                cold[i] = 1;
            }
        }
    }

    Some(StockVals {
        code: code.to_string(),
        ind,
        d15,
        buy15,
        vol15,
        cnt15,
        hot,
        cold,
    })
}

// ---------------- 输出结构 ----------------
#[derive(Serialize)]
struct PeerOut {
    code: String,
    cnt: u32,
}

#[derive(Serialize)]
struct StockOut {
    code: String,
    ind: u8,
    hot_cnt: u32,
    cold_cnt: u32,
    rk_buy_h: f32,
    rk_vol_h: f32,
    rk_ba_h: f32,
    z_buy_h: f32,
    z_vol_h: f32,
    z_ba_h: f32,
    rk_buy_c: f32,
    rk_vol_c: f32,
    rk_ba_c: f32,
    rk_buy_p: f32,
    rk_vol_p: f32,
    rk_ba_p: f32,
    hot_peers: Vec<PeerOut>,
    cold_peers: Vec<PeerOut>,
}

#[derive(Serialize)]
struct PoolStat {
    n_pools_total: usize,
    mean_size: f64,
    max_size: usize,
    total_inclusions: usize,
    /// 池大小直方图（分位点 [0,100,300,1000,2000,3000,4000,+∞)）
    hist: Vec<usize>,
}

#[derive(Serialize)]
struct Out {
    date: i64,
    n_stocks: usize,
    pool_hot: PoolStat,
    pool_cold: PoolStat,
    stocks: Vec<StockOut>,
}

/// 为组内同行业子集分配排名特征（rank_pct 与 z 分），累计到 sums。
/// bucket: 组内同行业成员; val: 排序键; sums/cnts 为每股累计。
fn accumulate_ranks(
    bucket: &mut Vec<u32>,
    stocks: &[StockVals],
    step: usize,
    val_sel: fn(&StockVals, usize) -> f32,
    rk_sums: &mut [f64],
    z_sums: Option<&mut [f64]>,
    cnts: &mut [f32],
) {
    let n = bucket.len();
    if n == 0 {
        return;
    }
    let mut zz = z_sums;
    bucket.sort_by(|&a, &b| {
        val_sel(&stocks[a as usize], step)
            .partial_cmp(&val_sel(&stocks[b as usize], step))
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    // 组内均值/标准差（用于 z 分）
    let mut m = 0.0f64;
    for &s in bucket.iter() {
        m += val_sel(&stocks[s as usize], step) as f64;
    }
    m /= n as f64;
    let var = if n >= 2 {
        bucket
            .iter()
            .map(|&s| {
                let v = val_sel(&stocks[s as usize], step) as f64;
                (v - m) * (v - m)
            })
            .sum::<f64>()
            / n as f64
    } else {
        0.0
    };
    let sd = var.sqrt();
    for (rank, &s) in bucket.iter().enumerate() {
        let idx = s as usize;
        rk_sums[idx] += (rank as f64 + 0.5) / n as f64;
        if let Some(zz) = zz.as_deref_mut() {
            if sd > 1e-12 {
                zz[idx] += (val_sel(&stocks[idx], step) as f64 - m) / sd;
            }
        }
        cnts[idx] += 1.0;
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("用法: hot_industry_sandbox <date> <industry_csv> [--no-cooc]");
        std::process::exit(1);
    }
    let date: i64 = args[1].parse().expect("date");
    let ind_path = args[2].clone();
    let do_cooc = !args.iter().any(|a| a == "--no-cooc");
    let precise = args.iter().any(|a| a == "--precise");
    eprintln!("mode = {}", if precise { "precise (文档语义 15s 窗口)" } else { "production (f32 量化网格,与正式库一致)" });

    let codes = list_codes(date);
    eprintln!("codes = {}", codes.len());

    let ind_map = load_industry(&ind_path);
    eprintln!("industry map = {}", ind_map.len());

    let t0 = std::time::Instant::now();
    let vals: Vec<Option<StockVals>> = codes
        .par_iter()
        .map(|c| {
            let ind = *ind_map.get(c).unwrap_or(&0);
            per_stock(c, date, ind, precise)
        })
        .collect();

    // 调试：--debug-code 000001 打印单股细节
    if let Some(ci) = args.iter().position(|a| a == "--debug-code") {
        if let Some(code) = args.get(ci + 1) {
            for (v, c) in vals.iter().zip(codes.iter()) {
                if c == code {
                    if let Some(s) = v {
                        eprintln!("debug {}: hot_cnt={} cold_cnt={}", code, s.hot.iter().map(|&x| x as u32).sum::<u32>(), s.cold.iter().map(|&x| x as u32).sum::<u32>());
                        for i in [40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 104, 105] {
                            if i < N_STEPS {
                                eprintln!("  i={} sec={} d15={} cnt15={} hot={} cold={}", i, 15 + 2 * i, s.d15[i], s.cnt15[i], s.hot[i], s.cold[i]);
                            }
                        }
                    }
                    break;
                }
            }
        }
    }
    eprintln!("per-stock done in {:?}", t0.elapsed());

    let stocks: Vec<StockVals> = vals.into_iter().flatten().collect();
    let n = stocks.len();
    eprintln!("valid stocks = {}", n);
    if n == 0 {
        return;
    }

    // 按 step 分桶（热/冰）
    let mut hot_buckets: Vec<Vec<u32>> = vec![Vec::new(); N_STEPS];
    let mut cold_buckets: Vec<Vec<u32>> = vec![Vec::new(); N_STEPS];
    hot_buckets
        .par_iter_mut()
        .enumerate()
        .for_each(|(step, b)| {
            for si in 0..n {
                if stocks[si].hot[step] != 0 {
                    b.push(si as u32);
                }
            }
        });
    cold_buckets
        .par_iter_mut()
        .enumerate()
        .for_each(|(step, b)| {
            for si in 0..n {
                if stocks[si].cold[step] != 0 {
                    b.push(si as u32);
                }
            }
        });

    let mut stat_hot = PoolStat {
        n_pools_total: 0,
        mean_size: 0.0,
        max_size: 0,
        total_inclusions: 0,
        hist: vec![0; 7],
    };
    let mut stat_cold = PoolStat {
        n_pools_total: 0,
        mean_size: 0.0,
        max_size: 0,
        total_inclusions: 0,
        hist: vec![0; 7],
    };
    let pool_bins = |sz: usize| -> usize {
        let edges = [0usize, 100, 300, 1000, 2000, 3000, 4000];
        for (i, &e) in edges.iter().enumerate() {
            if sz < e {
                return i;
            }
        }
        6
    };
    for b in &hot_buckets {
        stat_hot.n_pools_total += 1;
        stat_hot.mean_size += b.len() as f64;
        stat_hot.max_size = stat_hot.max_size.max(b.len());
        stat_hot.total_inclusions += b.len();
        stat_hot.hist[pool_bins(b.len())] += 1;
    }
    if stat_hot.n_pools_total > 0 {
        stat_hot.mean_size /= stat_hot.n_pools_total as f64;
    }
    for b in &cold_buckets {
        stat_cold.n_pools_total += 1;
        stat_cold.mean_size += b.len() as f64;
        stat_cold.max_size = stat_cold.max_size.max(b.len());
        stat_cold.total_inclusions += b.len();
        stat_cold.hist[pool_bins(b.len())] += 1;
    }
    if stat_cold.n_pools_total > 0 {
        stat_cold.mean_size /= stat_cold.n_pools_total as f64;
    }
    eprintln!(
        "pool hot: mean={:.1} max={} total_inc={} hist={:?} | cold: mean={:.1} max={} total_inc={} hist={:?}",
        stat_hot.mean_size,
        stat_hot.max_size,
        stat_hot.total_inclusions,
        stat_hot.hist,
        stat_cold.mean_size,
        stat_cold.max_size,
        stat_cold.total_inclusions,
        stat_cold.hist
    );

    // 共现矩阵（扁平 N×N）与行业内排名累计器
    let mut cooc_hot = vec![0u32; n * n];
    let mut cooc_cold = vec![0u32; n * n];
    let mut rk_buy_h = vec![0.0f64; n];
    let mut rk_vol_h = vec![0.0f64; n];
    let mut rk_ba_h = vec![0.0f64; n];
    let mut z_buy_h = vec![0.0f64; n];
    let mut z_vol_h = vec![0.0f64; n];
    let mut z_ba_h = vec![0.0f64; n];
    let mut cnt_h = vec![0.0f32; n];
    let mut rk_buy_c = vec![0.0f64; n];
    let mut rk_vol_c = vec![0.0f64; n];
    let mut rk_ba_c = vec![0.0f64; n];
    let mut z_buy_c = vec![0.0f64; n];
    let mut z_vol_c = vec![0.0f64; n];
    let mut z_ba_c = vec![0.0f64; n];
    let mut cnt_c = vec![0.0f32; n];
    // 全池(不分行业)排名 — 生产 f07/f28 的原始口径, 用于与行业内排名对比
    let mut rk_buy_p = vec![0.0f64; n];
    let mut rk_vol_p = vec![0.0f64; n];
    let mut rk_ba_p = vec![0.0f64; n];
    let mut cnt_p = vec![0.0f32; n];

    let t1 = std::time::Instant::now();
    let mut ind_buf: Vec<Vec<u32>> = vec![Vec::new(); 32];
    for step in 0..N_STEPS {
        // ---- 热点组 ----
        let pool = &hot_buckets[step];
        if !pool.is_empty() {
            if do_cooc {
                for &m in pool.iter() {
                    let base = m as usize * n;
                    for &o in pool.iter() {
                        if o != m {
                            cooc_hot[base + o as usize] += 1;
                        }
                    }
                }
            }
            for b in ind_buf.iter_mut() {
                b.clear();
            }
            for &m in pool.iter() {
                let ind = stocks[m as usize].ind as usize;
                if ind >= 1 && ind <= 31 {
                    ind_buf[ind].push(m);
                }
            }
            for ind in 1..=31usize {
                let b = &mut ind_buf[ind];
                if b.is_empty() {
                    continue;
                }
                accumulate_ranks(b, &stocks, step, |s, i| s.buy15[i], &mut rk_buy_h, Some(&mut z_buy_h), &mut cnt_h);
                accumulate_ranks(b, &stocks, step, |s, i| s.vol15[i], &mut rk_vol_h, Some(&mut z_vol_h), &mut cnt_h);
                accumulate_ranks(b, &stocks, step, |s, i| s.d15[i], &mut rk_ba_h, Some(&mut z_ba_h), &mut cnt_h);
            }
            // 全池(不分行业)排名 — 生产口径 f07/f28/f20 类比
            let mut full_pool: Vec<u32> = pool.clone();
            accumulate_ranks(&mut full_pool, &stocks, step, |s, i| s.buy15[i], &mut rk_buy_p, None, &mut cnt_p);
            accumulate_ranks(&mut full_pool, &stocks, step, |s, i| s.vol15[i], &mut rk_vol_p, None, &mut cnt_p);
            accumulate_ranks(&mut full_pool, &stocks, step, |s, i| s.d15[i], &mut rk_ba_p, None, &mut cnt_p);
        }
        // ---- 冰点组 ----
        let pool = &cold_buckets[step];
        if !pool.is_empty() {
            if do_cooc {
                for &m in pool.iter() {
                    let base = m as usize * n;
                    for &o in pool.iter() {
                        if o != m {
                            cooc_cold[base + o as usize] += 1;
                        }
                    }
                }
            }
            for b in ind_buf.iter_mut() {
                b.clear();
            }
            for &m in pool.iter() {
                let ind = stocks[m as usize].ind as usize;
                if ind >= 1 && ind <= 31 {
                    ind_buf[ind].push(m);
                }
            }
            for ind in 1..=31usize {
                let b = &mut ind_buf[ind];
                if b.is_empty() {
                    continue;
                }
                accumulate_ranks(b, &stocks, step, |s, i| s.buy15[i], &mut rk_buy_c, Some(&mut z_buy_c), &mut cnt_c);
                accumulate_ranks(b, &stocks, step, |s, i| s.vol15[i], &mut rk_vol_c, Some(&mut z_vol_c), &mut cnt_c);
                accumulate_ranks(b, &stocks, step, |s, i| s.d15[i], &mut rk_ba_c, Some(&mut z_ba_c), &mut cnt_c);
            }
        }
    }
    eprintln!("cooc+rank pass done in {:?}", t1.elapsed());

    // 组装输出
    let mut out_stocks = Vec::with_capacity(n);
    for si in 0..n {
        let s = &stocks[si];
        let hc = cnt_h[si];
        let cc = cnt_c[si];
        let hot_peers = if do_cooc {
            top10(&cooc_hot[si * n..(si + 1) * n], si, n, &stocks)
        } else {
            Vec::new()
        };
        let cold_peers = if do_cooc {
            top10(&cooc_cold[si * n..(si + 1) * n], si, n, &stocks)
        } else {
            Vec::new()
        };
        out_stocks.push(StockOut {
            code: s.code.clone(),
            ind: s.ind,
            hot_cnt: s.hot.iter().map(|&v| v as u32).sum(),
            cold_cnt: s.cold.iter().map(|&v| v as u32).sum(),
            rk_buy_h: if hc > 0.0 { (rk_buy_h[si] / hc as f64) as f32 } else { f32::NAN },
            rk_vol_h: if hc > 0.0 { (rk_vol_h[si] / hc as f64) as f32 } else { f32::NAN },
            rk_ba_h: if hc > 0.0 { (rk_ba_h[si] / hc as f64) as f32 } else { f32::NAN },
            z_buy_h: if hc > 0.0 { (z_buy_h[si] / hc as f64) as f32 } else { f32::NAN },
            z_vol_h: if hc > 0.0 { (z_vol_h[si] / hc as f64) as f32 } else { f32::NAN },
            z_ba_h: if hc > 0.0 { (z_ba_h[si] / hc as f64) as f32 } else { f32::NAN },
            rk_buy_c: if cc > 0.0 { (rk_buy_c[si] / cc as f64) as f32 } else { f32::NAN },
            rk_vol_c: if cc > 0.0 { (rk_vol_c[si] / cc as f64) as f32 } else { f32::NAN },
            rk_ba_c: if cc > 0.0 { (rk_ba_c[si] / cc as f64) as f32 } else { f32::NAN },
            rk_buy_p: if cnt_p[si] > 0.0 { (rk_buy_p[si] / cnt_p[si] as f64) as f32 } else { f32::NAN },
            rk_vol_p: if cnt_p[si] > 0.0 { (rk_vol_p[si] / cnt_p[si] as f64) as f32 } else { f32::NAN },
            rk_ba_p: if cnt_p[si] > 0.0 { (rk_ba_p[si] / cnt_p[si] as f64) as f32 } else { f32::NAN },
            hot_peers,
            cold_peers,
        });
    }

    let out = Out {
        date,
        n_stocks: n,
        pool_hot: stat_hot,
        pool_cold: stat_cold,
        stocks: out_stocks,
    };
    println!("{}", serde_json::to_string(&out).unwrap());
}

fn top10(row: &[u32], self_idx: usize, n: usize, stocks: &[StockVals]) -> Vec<PeerOut> {
    let mut cands: Vec<(usize, u32)> = row
        .iter()
        .enumerate()
        .filter(|&(i, &c)| i != self_idx && c > 0)
        .map(|(i, &c)| (i, c))
        .collect();
    cands.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    cands.truncate(K_PEERS);
    cands.iter()
        .map(|(i, c)| PeerOut {
            code: stocks[*i].code.clone(),
            cnt: *c,
        })
        .collect()
}
