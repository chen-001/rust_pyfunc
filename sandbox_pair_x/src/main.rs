//! 单日全市场"跨股票互动指标"两两矩阵（sandbox 探索版）
//!
//! 遵循 cross-section-explore 规范：
//!   - sandbox 独立 crate（不碰正式库 src/lib.rs / alter.sh）
//!   - 复用 sandbox_hidden_arrange 的简化 fast_csv_reader（已改为 read_to_end）
//!   - rayon 文件级并行读全市场 + per-stock 特征（步骤①+②）
//!   - 跨股票交互（步骤③）→ 写盘（步骤④）；日志走 eprintln!，结果走文件
//! 预留 cross-section-pipeline 升级路径（B 类因子规范）：
//!   - per_stock_prep 把每股逐笔**重采样到统一时间网格**（1秒/30秒/60秒/300秒桶，
//!     大订单桶），避免全量逐笔跨股票对齐的内存爆炸
//!   - 升级正式库时：核心逻辑原样搬，套 compute_xxx_full 壳 + 5 处注册
//! 输出说明：N×N 矩阵（约 5000×5000）远超 JSON 承载能力，采用二进制 .bin
//!   （f32 小端, 行主序）+ meta.json + stats.csv，Python 用 np.fromfile 加载。
//!
//! 构建: cd sandbox_pair_x && cargo build --release
//! 运行: ./target/release/pair_x_sandbox <date> <outdir> [limit(调试:只处理前N只)]
//! 输出文件:
//!   meta.json            参数与文件清单（Python 入口）
//!   codes.txt            有效股票代码（按代码序, N 行）
//!   stats.csv            code,n,th99,turnover,median_vol,nbig
//!   mat_x1.bin / mat_x3.bin / mat_x8.bin / mat_x9.bin   N×N f32 矩阵
//!   cnt1.bin             N×14220 1秒桶计数
//!   {s,c,v,p}{30,60,300}.bin   多尺度桶: 净主动量/笔数/量/末价
//!   big10.bin / big30.bin     大订单成交带符号桶（10s/30s）
mod fast_csv_reader;
use fast_csv_reader::{read_trade_fast, TradeRecord};
use rayon::prelude::*;
use serde_json::json;
use std::collections::BTreeSet;
use std::io::Write;

const MIN_TRADES: usize = 200;   // 每股最少成交笔数
const BUCKET_1S: usize = 14220;  // 237 分钟 × 60
const BIG10_T: usize = 1422;     // 10 秒桶数
const BIG30_T: usize = 474;      // 30 秒桶数
// (桶宽秒, 桶数)
const SCALES: [(usize, usize); 3] = [(30, 474), (60, 237), (300, 48)];

/// 公历日期 -> 1970-01-01 起的天数 (Howard Hinnant 算法)
fn days_from_civil(y: i64, m: u64, d: u64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400;
    let mp = (m + 9) % 12;
    let doy = (153 * mp + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy as i64;
    era * 146097 + doe - 719468
}

/// 步骤① 枚举某日全市场股票代码（横截面因子枚举用, 双数据根目录回退）
fn list_codes(date: i64, subdir: &str) -> Vec<String> {
    for root in ["/ssd_data/stock", "/nas197/binary/stock/sz_alpha/stock"] {
        let dir = format!("{root}/{date}/{subdir}");
        if let Ok(entries) = std::fs::read_dir(&dir) {
            let mut set = BTreeSet::new();
            for e in entries.flatten() {
                if let Some(code) = e.file_name().to_str().and_then(|n| n.split('_').next()) {
                    if code.bytes().all(|b| b.is_ascii_digit()) {
                        set.insert(code.to_string());
                    }
                }
            }
            if !set.is_empty() {
                return set.into_iter().collect();
            }
        }
    }
    Vec::new()
}

/// 每股预处理结果（步骤②输出）
struct StockPrep {
    code: String,
    times: Vec<i64>,   // 成交时间(微秒, 已做下午平移)
    big: Vec<u8>,      // 是否大单(单笔量 >= p99)
    n: usize,
    th99: f64,
    turnover: f64,
    median_vol: f64,
    nbig: usize,       // 大订单成交笔数
    cnt1: Vec<f32>,    // 1 秒桶计数
    s30: Vec<f32>, c30: Vec<f32>, v30: Vec<f32>, p30: Vec<f32>,
    s60: Vec<f32>, c60: Vec<f32>, v60: Vec<f32>, p60: Vec<f32>,
    s300: Vec<f32>, c300: Vec<f32>, v300: Vec<f32>, p300: Vec<f32>,
    big10: Vec<f32>,   // 大订单成交 10 秒桶(带符号: 主动买+, 主动卖-)
    big30: Vec<f32>,
}

/// 步骤② per-stock 预处理：压缩逐笔 + 重采样到统一时间网格 + 大单/大订单识别
/// （B 类因子规范：跨股票对齐前先降维到时间网格）
fn per_stock_prep(code: &str, recs: &[TradeRecord], day_start_us: i64) -> Option<StockPrep> {
    let n = recs.len();
    if n < MIN_TRADES {
        return None;
    }
    let mut times = Vec::with_capacity(n);
    let mut vols = Vec::with_capacity(n);
    let mut turnover = 0.0f64;
    for r in recs {
        times.push(r.time_us);
        vols.push(r.volume as f32);
        turnover += r.turnover;
    }
    // 单笔量分位数
    let mut sorted = vols.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let th99 = sorted[((n as f64) * 0.99) as usize] as f64;
    let median_vol = sorted[n / 2] as f64;
    let big: Vec<u8> = vols.iter().map(|&v| if (v as f64) >= th99 { 1u8 } else { 0u8 }).collect();

    let mut cnt1 = vec![0f32; BUCKET_1S];
    let mut s30 = vec![0f32; 474]; let mut c30 = vec![0f32; 474];
    let mut v30 = vec![0f32; 474]; let mut p30 = vec![0f32; 474];
    let mut s60 = vec![0f32; 237]; let mut c60 = vec![0f32; 237];
    let mut v60 = vec![0f32; 237]; let mut p60 = vec![0f32; 237];
    let mut s300 = vec![0f32; 48]; let mut c300 = vec![0f32; 48];
    let mut v300 = vec![0f32; 48]; let mut p300 = vec![0f32; 48];
    for r in recs.iter() {
        let idx1 = ((r.time_us - day_start_us) / 1_000_000) as usize;
        if idx1 < BUCKET_1S {
            cnt1[idx1] += 1.0;
        }
        let sign: f32 = match r.flag { 66 => 1.0, 83 => -1.0, _ => 0.0 };
        let sv = r.volume as f32 * sign;
        let v = r.volume as f32;
        let i30 = ((r.time_us - day_start_us) / 30_000_000) as usize;
        if i30 < 474 { s30[i30] += sv; c30[i30] += 1.0; v30[i30] += v; p30[i30] = r.price as f32; }
        let i60 = ((r.time_us - day_start_us) / 60_000_000) as usize;
        if i60 < 237 { s60[i60] += sv; c60[i60] += 1.0; v60[i60] += v; p60[i60] = r.price as f32; }
        let i300 = ((r.time_us - day_start_us) / 300_000_000) as usize;
        if i300 < 48 { s300[i300] += sv; c300[i300] += 1.0; v300[i300] += v; p300[i300] = r.price as f32; }
    }

    // 大订单: 攻击方订单号(66主买→bid_order, 83主卖→ask_order)聚合总量 >= th99 的订单,
    // 其所有成交 = 大单成交 → 带符号时间桶
    let mut fills: Vec<(i64, i64, f32, f32)> = Vec::with_capacity(n); // (oid, t, vol, sign)
    for r in recs {
        let (oid, sign) = match r.flag {
            66 => (r.bid_order, 1.0f32),
            83 => (r.ask_order, -1.0f32),
            _ => continue,
        };
        if oid == 0 { continue; }
        fills.push((oid, r.time_us, r.volume as f32, sign));
    }
    fills.sort_by_key(|f| f.0);
    let mut big10 = vec![0f32; BIG10_T];
    let mut big30 = vec![0f32; BIG30_T];
    let mut nbig = 0usize;
    {
        // 每订单总量
        let mut i = 0usize;
        let mut totals: Vec<(i64, f64)> = Vec::new();
        while i < fills.len() {
            let oid = fills[i].0;
            let mut tot = 0.0f64;
            let mut j = i;
            while j < fills.len() && fills[j].0 == oid {
                tot += fills[j].2 as f64;
                j += 1;
            }
            totals.push((oid, tot));
            i = j;
        }
        // 双指针: 对每个成交, 查所属订单是否大单
        let mut ti = 0usize;
        for &(oid, t, vol, sign) in &fills {
            while ti + 1 < totals.len() && totals[ti].0 < oid { ti += 1; }
            if totals[ti].0 == oid && totals[ti].1 >= th99 {
                let i10 = ((t - day_start_us) / 10_000_000) as usize;
                if i10 < BIG10_T { big10[i10] += sign * vol; }
                let i30 = ((t - day_start_us) / 30_000_000) as usize;
                if i30 < BIG30_T { big30[i30] += sign * vol; }
                nbig += 1;
            }
        }
    }

    Some(StockPrep {
        code: code.to_string(),
        times, big, n,
        th99, turnover, median_vol, nbig,
        cnt1,
        s30, c30, v30, p30,
        s60, c60, v60, p60,
        s300, c300, v300, p300,
        big10, big30,
    })
}

/// 步骤③ 跨股票交互（两两合并流指标）
/// X1 交替频率 / X3 转移互信息 / X8 大单聚集(相邻大单间隔<30s比例) / X9 间隔CV
#[derive(Clone, Copy)]
struct PairOut { x1: f32, x3: f32, x8: f32, x9: f32 }

#[inline]
fn pairwise_merge_metrics(times_a: &[i64], big_a: &[u8], times_b: &[i64], big_b: &[u8]) -> PairOut {
    let na = times_a.len();
    let nb = times_b.len();
    let n = na + nb;
    let mut trans: u64 = 0;
    let (mut c00, mut c01, mut c10, mut c11): (u64, u64, u64, u64) = (0, 0, 0, 0);
    let mut h = [0u64; 33]; // 大单间隔直方图: 10s 一箱, 第32箱为溢出
    let mut nbig_gaps: u64 = 0;
    let mut last_big: i64 = i64::MIN;
    let (mut gap_mean, mut gap_m2): (f64, f64) = (0.0, 0.0);
    let mut gap_cnt: u64 = 0;
    let (mut prev_t, mut prev_label): (i64, u8) = (0, 2);
    let mut has_prev = false;
    let (mut ia, mut ib) = (0usize, 0usize);
    while ia < na || ib < nb {
        let (label, t, big) = if ib >= nb || (ia < na && times_a[ia] <= times_b[ib]) {
            let l = (0u8, times_a[ia], big_a[ia] != 0);
            ia += 1;
            l
        } else {
            let l = (1u8, times_b[ib], big_b[ib] != 0);
            ib += 1;
            l
        };
        if has_prev {
            if label != prev_label { trans += 1; }
            match (prev_label, label) {
                (0, 0) => c00 += 1, (0, 1) => c01 += 1,
                (1, 0) => c10 += 1, _ => c11 += 1,
            }
            let g = (t - prev_t) as f64;
            gap_cnt += 1;
            let d = g - gap_mean;
            gap_mean += d / gap_cnt as f64;
            gap_m2 += d * (g - gap_mean);
        }
        if big {
            if last_big != i64::MIN {
                let bg = (t - last_big) / 10_000_000;
                let bin = if bg >= 32 { 32usize } else { bg as usize };
                h[bin] += 1;
                nbig_gaps += 1;
            }
            last_big = t;
        }
        prev_t = t;
        prev_label = label;
        has_prev = true;
    }
    // X1 交替频率
    let x1 = if n > 1 { trans as f32 / (n - 1) as f32 } else { 0.0 };
    // X3 转移互信息 (nats)
    let x3 = {
        if n <= 1 {
            0.0
        } else {
            let denom = (n - 1) as f64;
            let (p00, p01, p10, p11) = (c00 as f64 / denom, c01 as f64 / denom, c10 as f64 / denom, c11 as f64 / denom);
            let (p0, p1) = (p00 + p01, p10 + p11);
            let mut mi = 0.0;
            for (pab, pa, pb) in [(p00, p0, p0), (p01, p0, p1), (p10, p1, p0), (p11, p1, p1)] {
                if pab > 0.0 && pa > 0.0 && pb > 0.0 {
                    mi += pab * (pab / (pa * pb)).ln();
                }
            }
            mi as f32
        }
    };
    // X8 大单聚集: 相邻大单间隔 < 30s 的比例
    let x8 = if nbig_gaps > 0 { (h[0] + h[1] + h[2]) as f32 / nbig_gaps as f32 } else { 0.0 };
    // X9 间隔CV
    let x9 = if gap_cnt > 1 && gap_mean > 0.0 {
        ((gap_m2 / (gap_cnt - 1) as f64).sqrt() / gap_mean) as f32
    } else {
        0.0
    };
    PairOut { x1, x3, x8, x9 }
}

fn write_f32_mat(path: &str, mat: &[f32]) -> std::io::Result<()> {
    let mut f = std::fs::File::create(path)?;
    let bytes: &[u8] = unsafe { std::slice::from_raw_parts(mat.as_ptr() as *const u8, mat.len() * 4) };
    f.write_all(bytes)
}

fn write_f32_rows(path: &str, rows: &[Vec<f32>]) -> std::io::Result<()> {
    let mut f = std::fs::File::create(path)?;
    for r in rows {
        let bytes: &[u8] = unsafe { std::slice::from_raw_parts(r.as_ptr() as *const u8, r.len() * 4) };
        f.write_all(bytes)?;
    }
    Ok(())
}

/// 步骤①→④ 全流程：读全市场 → per-stock → 两两交互 → 写盘
fn compute_pair_interaction_full(date: i64, outdir: &str, limit: Option<usize>) -> std::io::Result<usize> {
    std::fs::create_dir_all(outdir)?;

    let mut codes = list_codes(date, "transaction");
    if let Some(l) = limit {
        codes.truncate(l);
    }
    eprintln!("[step1] codes = {}", codes.len());

    // 09:30 (北京时间) 当日微秒起点。
    // 注意: reader 的 time_us = exchtime + 8h（即"北京时间墙钟"的 epoch 表示），
    // 因此这里只加 9.5h，不能再加 8h。
    let day_start_us = days_from_civil(date / 10000, (date / 100 % 100) as u64, (date % 100) as u64)
        * 86400 * 1_000_000
        + (9 * 3600 + 30 * 60) * 1_000_000;

    // ①+② rayon 并行读全市场 + per-stock 预处理（B 类: 重采样到时间网格）
    let t0 = std::time::Instant::now();
    let preps: Vec<Option<StockPrep>> = codes.par_iter()
        .map(|c| read_trade_fast(c, date).ok().and_then(|recs| per_stock_prep(c, &recs, day_start_us)))
        .collect();
    let valid: Vec<StockPrep> = preps.into_iter().flatten().collect();
    eprintln!("[step2] valid = {} (read+prep {:.1}s)", valid.len(), t0.elapsed().as_secs_f32());

    let n = valid.len();

    // ④a 写每股桶数组与统计
    let mut codes_out = Vec::with_capacity(n);
    let mut stats = String::new();
    stats.push_str("code,n,th99,turnover,median_vol,nbig\n");
    let mut cnt1_rows = Vec::with_capacity(n);
    let mut s30_rows = Vec::with_capacity(n); let mut c30_rows = Vec::with_capacity(n);
    let mut v30_rows = Vec::with_capacity(n); let mut p30_rows = Vec::with_capacity(n);
    let mut s60_rows = Vec::with_capacity(n); let mut c60_rows = Vec::with_capacity(n);
    let mut v60_rows = Vec::with_capacity(n); let mut p60_rows = Vec::with_capacity(n);
    let mut s300_rows = Vec::with_capacity(n); let mut c300_rows = Vec::with_capacity(n);
    let mut v300_rows = Vec::with_capacity(n); let mut p300_rows = Vec::with_capacity(n);
    let mut big10_rows = Vec::with_capacity(n);
    let mut big30_rows = Vec::with_capacity(n);
    let mut n_trades_total: u64 = 0;
    for s in &valid {
        codes_out.push(s.code.clone());
        stats.push_str(&format!("{},{},{:.2},{:.2},{:.2},{}\n", s.code, s.n, s.th99, s.turnover, s.median_vol, s.nbig));
        n_trades_total += s.n as u64;
        cnt1_rows.push(s.cnt1.clone());
        s30_rows.push(s.s30.clone()); c30_rows.push(s.c30.clone());
        v30_rows.push(s.v30.clone()); p30_rows.push(s.p30.clone());
        s60_rows.push(s.s60.clone()); c60_rows.push(s.c60.clone());
        v60_rows.push(s.v60.clone()); p60_rows.push(s.p60.clone());
        s300_rows.push(s.s300.clone()); c300_rows.push(s.c300.clone());
        v300_rows.push(s.v300.clone()); p300_rows.push(s.p300.clone());
        big10_rows.push(s.big10.clone());
        big30_rows.push(s.big30.clone());
    }
    std::fs::write(format!("{outdir}/codes.txt"), codes_out.join("\n"))?;
    std::fs::write(format!("{outdir}/stats.csv"), stats)?;
    write_f32_rows(&format!("{outdir}/cnt1.bin"), &cnt1_rows)?;
    write_f32_rows(&format!("{outdir}/s30.bin"), &s30_rows)?;
    write_f32_rows(&format!("{outdir}/c30.bin"), &c30_rows)?;
    write_f32_rows(&format!("{outdir}/v30.bin"), &v30_rows)?;
    write_f32_rows(&format!("{outdir}/p30.bin"), &p30_rows)?;
    write_f32_rows(&format!("{outdir}/s60.bin"), &s60_rows)?;
    write_f32_rows(&format!("{outdir}/c60.bin"), &c60_rows)?;
    write_f32_rows(&format!("{outdir}/v60.bin"), &v60_rows)?;
    write_f32_rows(&format!("{outdir}/p60.bin"), &p60_rows)?;
    write_f32_rows(&format!("{outdir}/s300.bin"), &s300_rows)?;
    write_f32_rows(&format!("{outdir}/c300.bin"), &c300_rows)?;
    write_f32_rows(&format!("{outdir}/v300.bin"), &v300_rows)?;
    write_f32_rows(&format!("{outdir}/p300.bin"), &p300_rows)?;
    write_f32_rows(&format!("{outdir}/big10.bin"), &big10_rows)?;
    write_f32_rows(&format!("{outdir}/big30.bin"), &big30_rows)?;
    eprintln!("[step4a] per-stock files written, total trades = {}", n_trades_total);

    // ③ 两两交互 (rayon 按行并行, par_chunks_mut 每行独立写, 无竞争)
    let comp: Vec<(Vec<i64>, Vec<u8>)> = valid.into_iter()
        .map(|s| (s.times, s.big))
        .collect();
    let comp = std::sync::Arc::new(comp);
    let mut m1 = vec![0f32; n * n];
    let mut m3 = vec![0f32; n * n];
    let mut m8 = vec![0f32; n * n];
    let mut m9 = vec![0f32; n * n];
    let t1 = std::time::Instant::now();
    m1.par_chunks_mut(n)
        .zip(m3.par_chunks_mut(n))
        .zip(m8.par_chunks_mut(n))
        .zip(m9.par_chunks_mut(n))
        .enumerate()
        .for_each(|(i, (((row1, row3), row8), row9))| {
            let sa = &comp[i];
            for j in 0..n {
                if i == j { continue; }
                let sb = &comp[j];
                let o = pairwise_merge_metrics(&sa.0, &sa.1, &sb.0, &sb.1);
                row1[j] = o.x1;
                row3[j] = o.x3;
                row8[j] = o.x8;
                row9[j] = o.x9;
            }
        });
    eprintln!("[step3] pairwise done {:.1}s", t1.elapsed().as_secs_f32());

    // ④b 写矩阵
    write_f32_mat(&format!("{outdir}/mat_x1.bin"), &m1)?;
    write_f32_mat(&format!("{outdir}/mat_x3.bin"), &m3)?;
    write_f32_mat(&format!("{outdir}/mat_x8.bin"), &m8)?;
    write_f32_mat(&format!("{outdir}/mat_x9.bin"), &m9)?;

    // meta.json: 参数与文件清单（Python 加载入口）
    let meta = json!({
        "date": date,
        "n": n,
        "min_trades": MIN_TRADES,
        "bucket_1s": BUCKET_1S,
        "scales": { "30": 474, "60": 237, "300": 48 },
        "big_buckets": { "10": BIG10_T, "30": BIG30_T },
        "matrices": ["x1", "x3", "x8", "x9"],
        "files": {
            "codes": "codes.txt", "stats": "stats.csv",
            "mats": ["mat_x1.bin", "mat_x3.bin", "mat_x8.bin", "mat_x9.bin"],
            "cnt1": "cnt1.bin",
            "scales": ["s30.bin","c30.bin","v30.bin","p30.bin","s60.bin","c60.bin","v60.bin","p60.bin","s300.bin","c300.bin","v300.bin","p300.bin"],
            "big": ["big10.bin", "big30.bin"]
        }
    });
    std::fs::write(format!("{outdir}/meta.json"), serde_json::to_string_pretty(&meta)?)?;
    Ok(n)
}

fn main() {
    let date: i64 = std::env::args().nth(1).map(|s| s.parse().unwrap()).unwrap_or(20260717);
    let outdir = std::env::args().nth(2).unwrap_or_else(|| "/home/chenzongwei/pair_x_out".to_string());
    let outdir = format!("{outdir}/{date}");
    let limit = std::env::args().nth(3).map(|s| s.parse::<usize>().unwrap());
    match compute_pair_interaction_full(date, &outdir, limit) {
        Ok(n) => {
            // 结果通道: 单行 JSON（Python 可直接加载）
            println!("{}", json!({"ok": true, "date": date, "outdir": outdir, "n": n}));
        }
        Err(e) => {
            eprintln!("error: {e:?}");
            println!("{}", json!({"ok": false, "date": date, "outdir": outdir, "error": format!("{e:?}")}));
            std::process::exit(1);
        }
    }
}
