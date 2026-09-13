//! 同热点股票池「行业维度拓展」sandbox —— 4 组补充因子单日全市场计算。
//!
//! 组 1 hotpool_ext_ind_rel_*   (840): 全市场识别后, 每股每次入选记 5 个行业内相对量,
//!      收盘后按行业当日全部入选记录回填排名/z/频率比, 再走 21 统计量降维。
//! 组 2 hotpool_ext_ind_cooc_*  (720): 初版全市场共现 top-10 按同行/跨行拆分聚合。
//! 组 3 hotpool_ext_ind_heat_*  ( 24): 行业层热度聚合映射到每股。
//! 组 4 hotpool_ext_indpool_*  (7072): 识别层行业内化——组=当秒同行业入选者,
//!      40 维组特征全部在行业内桶上计算, 降维 + 同行共现。
//!
//! 用法: ./target/release/hotpool_ind_sandbox <date> <industry_csv> [--only CODE]
//! 输出: out/{date}/facs.bin (n×8656 f32) + codes.txt + names.json

mod fast_csv_reader;
mod reduce;

use fast_csv_reader::read_trade_fast;
use rayon::prelude::*;
use std::collections::BTreeSet;

// ---------------- 与正式库一致的常量 ----------------
const FEAT_PER_INCLUSION: usize = 40;
const N_PARAM_COMBOS: usize = 4;
const BASIC_FEAT_N: usize = 11;
const SEC_OFFSET: i64 = 9 * 3600 + 30 * 60; // 34200
const MORNING_END: i64 = 11 * 3600 + 30 * 60; // 41400
const AFTERNOON_START: i64 = MORNING_END + 1; // 41401
const AFTERNOON_END: i64 = MORNING_END + (14 * 3600 + 57 * 60 - 13 * 3600); // 48420
const ADJUSTED_SECONDS: usize =
    ((MORNING_END - SEC_OFFSET + 1) + (AFTERNOON_END - AFTERNOON_START + 1)) as usize;
const SECOND_STEP: usize = 2;
const HIST_WIN: usize = 120;
const Z_THRESH: f64 = 1.5;
const MIN_TRADES: u32 = 10;

const PARAM_CONFIGS: [(usize, usize, u32); N_PARAM_COMBOS] = [
    (60, 0, 10), // x=60, buy_ratio
    (60, 1, 10), // x=60, bid_ask
    (15, 0, 10), // x=15, buy_ratio
    (15, 1, 10), // x=15, bid_ask
];
const COMBO_LABELS: [&str; 4] = ["x60y3_buy", "x60y3_ba", "x15y10_buy", "x15y10_ba"];

// 因子块大小
const G1_PER: usize = 5 * 21; // 5 特征 × 21 统计
const G2_PER: usize = BASIC_FEAT_N * 2 * 4 + 2; // 11×2×4 + 2 结构量 = 90
const G3_PER: usize = 6;
const G4_REDUCED_PER: usize = FEAT_PER_INCLUSION * 21; // 40×21 = 840
const G4_COOC_PER: usize = BASIC_FEAT_N * 2 * 4; // 88
pub const N_FACTORS: usize = 4 * 2 * G1_PER
    + 4 * 2 * G2_PER
    + 4 * G3_PER
    + 4 * 2 * G4_REDUCED_PER
    + 4 * G4_COOC_PER; // 840+720+24+6720+352 = 8656

// ---------------- 时间工具 ----------------
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
#[inline]
fn sec_to_idx(epoch: f32, day_midnight_cst: i64) -> Option<usize> {
    let e = (epoch as i64) - day_midnight_cst;
    if e < SEC_OFFSET || e > AFTERNOON_END {
        return None;
    }
    if e <= MORNING_END {
        Some((e - SEC_OFFSET) as usize)
    } else {
        Some((MORNING_END - SEC_OFFSET + 1 + e - AFTERNOON_START) as usize)
    }
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

// ---------------- 数据结构（与正式库一致） ----------------
#[derive(Clone, Copy, Default)]
struct SecStat {
    buy_ratio: f32,
    buy_vol: f32,
    bid_ask_mean: f32,
    ret_val: f32,
    volume: f32,
    trade_cnt: u32,
    last_price: f32,
    first_price: f32,
    has_data: bool,
}

struct StockData {
    code: String,
    ind: u8,
    secs: Vec<SecStat>,
    basic_feats: [f32; BASIC_FEAT_N],
}

struct RollingCache {
    data: Vec<f32>, // 10 * ADJUSTED_SECONDS
}

impl RollingCache {
    #[inline]
    fn get_by_x(&self, x: usize, field_type: u8, sec: usize) -> f32 {
        let base = match (field_type, x) {
            (0, 15) => 0,
            (0, 60) => 1,
            (1, 15) => 2,
            (1, 60) => 3,
            (2, 15) => 4,
            (2, 60) => 5,
            (3, 15) => 6,
            (3, 60) => 7,
            (4, 15) => 8,
            (4, 60) => 9,
            _ => return f32::NAN,
        };
        self.data[base * ADJUSTED_SECONDS + sec]
    }

    fn compute(secs: &[SecStat]) -> Self {
        let n = ADJUSTED_SECONDS;
        let total = 10 * n;
        let mut data = vec![f32::NAN; total];
        // buy_ratio 滚动: sum(buy_vol)/sum(volume)
        for &(col, win) in &[(0usize, 15usize), (1, 60)] {
            let base = col * n;
            let mut sum_buy: f64 = 0.0;
            let mut sum_vol: f64 = 0.0;
            for sec in 0..n {
                let s = &secs[sec];
                if s.has_data {
                    sum_buy += s.buy_vol as f64;
                    sum_vol += s.volume as f64;
                }
                if sec >= win {
                    let old = &secs[sec - win];
                    if old.has_data {
                        sum_buy -= old.buy_vol as f64;
                        sum_vol -= old.volume as f64;
                    }
                }
                if sum_vol > 0.0 && sec >= win - 1 {
                    data[base + sec] = (sum_buy / sum_vol) as f32;
                }
            }
        }
        // bid_ask / volume
        let configs: [(usize, usize, fn(&SecStat) -> f32, bool); 4] = [
            (2, 15, |s: &SecStat| s.bid_ask_mean, false),
            (3, 60, |s: &SecStat| s.bid_ask_mean, false),
            (6, 15, |s: &SecStat| s.volume, true),
            (7, 60, |s: &SecStat| s.volume, true),
        ];
        for &(col, win, getter, is_sum) in &configs {
            let base = col * n;
            let mut sum: f64 = 0.0;
            let mut cnt: u32 = 0;
            for sec in 0..n {
                let v = getter(&secs[sec]);
                if v.is_finite() {
                    sum += v as f64;
                    cnt += 1;
                }
                if sec >= win {
                    let old = getter(&secs[sec - win]);
                    if old.is_finite() {
                        sum -= old as f64;
                        cnt -= 1;
                    }
                }
                if cnt > 0 && sec >= win - 1 {
                    data[base + sec] = if is_sum { sum as f32 } else { (sum / cnt as f64) as f32 };
                }
            }
        }
        // trade_cnt
        for &(col, win) in &[(8usize, 15usize), (9, 60)] {
            let base = col * n;
            let mut sum: u32 = 0;
            for sec in 0..n {
                sum += secs[sec].trade_cnt;
                if sec >= win {
                    sum -= secs[sec - win].trade_cnt;
                }
                data[base + sec] = sum as f32;
            }
        }
        // ret 窗口首末价变动
        for &(col, win) in &[(4usize, 15usize), (5, 60)] {
            let base = col * n;
            let mut dq: std::collections::VecDeque<(usize, f32, f32)> =
                std::collections::VecDeque::with_capacity(win);
            for sec in 0..n {
                let s = &secs[sec];
                if s.has_data && s.first_price > 0.0 {
                    dq.push_back((sec, s.first_price, s.last_price));
                }
                while let Some(&(front_sec, _, _)) = dq.front() {
                    if front_sec + win <= sec {
                        dq.pop_front();
                    } else {
                        break;
                    }
                }
                if sec >= win - 1 {
                    if let (Some(&(_, fp, _)), Some(&(_, _, lp))) = (dq.front(), dq.back()) {
                        if fp > 0.0 {
                            data[base + sec] = (lp - fp) / fp;
                        }
                    }
                }
            }
        }
        Self { data }
    }
}

// ---------------- per-stock 构建 ----------------
fn build_stock_data(code: &str, date: i64, ind: u8) -> Option<StockData> {
    let trades = read_trade_fast(code, date).ok()?;
    if trades.is_empty() {
        return None;
    }
    let n_secs = ADJUSTED_SECONDS;
    let day_mid = cst_midnight_epoch(date);
    let mut buy_vol = vec![0.0f64; n_secs];
    let mut total_vol = vec![0.0f64; n_secs];
    let mut bid_ask_sum = vec![0.0f64; n_secs];
    let mut bid_ask_cnt = vec![0u32; n_secs];
    let mut first_prices = vec![0.0f32; n_secs];
    let mut last_prices = vec![0.0f32; n_secs];
    let mut has_data = vec![false; n_secs];

    for t in &trades {
        let idx = match sec_to_idx((t.time_sec as f32), day_mid) {
            Some(i) => i,
            None => continue,
        };
        has_data[idx] = true;
        let vol = t.volume as f64;
        total_vol[idx] += vol;
        if t.flag == 66 {
            buy_vol[idx] += vol;
        }
        bid_ask_sum[idx] += (t.bid_order - t.ask_order) as f64;
        bid_ask_cnt[idx] += 1;
        if first_prices[idx] == 0.0 {
            first_prices[idx] = t.price as f32;
        }
        last_prices[idx] = t.price as f32;
    }

    let mut secs = Vec::with_capacity(n_secs);
    for i in 0..n_secs {
        let tv = total_vol[i];
        if tv > 0.0 {
            secs.push(SecStat {
                buy_ratio: (buy_vol[i] / tv) as f32,
                buy_vol: buy_vol[i] as f32,
                bid_ask_mean: if bid_ask_cnt[i] > 0 {
                    (bid_ask_sum[i] / bid_ask_cnt[i] as f64) as f32
                } else {
                    f32::NAN
                },
                ret_val: if first_prices[i] > 0.0 {
                    (last_prices[i] - first_prices[i]) / first_prices[i]
                } else {
                    f32::NAN
                },
                volume: tv as f32,
                trade_cnt: bid_ask_cnt[i],
                last_price: last_prices[i],
                first_price: first_prices[i],
                has_data: true,
            });
        } else {
            secs.push(SecStat::default());
        }
    }
    let basic = compute_basic_features(&secs);
    Some(StockData {
        code: code.to_string(),
        ind,
        secs,
        basic_feats: basic,
    })
}

fn safe_std(v: &[f32]) -> f32 {
    let finite: Vec<f32> = v.iter().filter(|x| x.is_finite()).copied().collect();
    if finite.len() < 2 {
        return f32::NAN;
    }
    let mean = finite.iter().sum::<f32>() / finite.len() as f32;
    (finite.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / finite.len() as f32).sqrt()
}
fn rolling_std_short(data: &[f32], len: usize) -> Vec<f32> {
    let n = data.len();
    let mut out = vec![f32::NAN; n];
    if n < len || len < 2 {
        return out;
    }
    for i in (len - 1)..n {
        out[i] = safe_std(&data[i + 1 - len..=i]);
    }
    out
}
fn rolling_buy_ratio(buy_vols: &[f32], total_vols: &[f32], len: usize) -> Vec<f32> {
    let n = buy_vols.len();
    let mut out = vec![f32::NAN; n];
    if n < len {
        return out;
    }
    let mut sb: f64 = 0.0;
    let mut sv: f64 = 0.0;
    for i in 0..n {
        if buy_vols[i].is_finite() && total_vols[i].is_finite() {
            sb += buy_vols[i] as f64;
            sv += total_vols[i] as f64;
        }
        if i >= len {
            if buy_vols[i - len].is_finite() && total_vols[i - len].is_finite() {
                sb -= buy_vols[i - len] as f64;
                sv -= total_vols[i - len] as f64;
            }
        }
        if sv > 0.0 && i >= len - 1 {
            out[i] = (sb / sv) as f32;
        }
    }
    out
}

fn compute_basic_features(secs: &[SecStat]) -> [f32; BASIC_FEAT_N] {
    let rets: Vec<f32> = secs
        .iter()
        .map(|s| if s.has_data { s.ret_val } else { f32::NAN })
        .collect();
    let bid_asks: Vec<f32> = secs
        .iter()
        .map(|s| if s.has_data { s.bid_ask_mean } else { f32::NAN })
        .collect();
    let vols: Vec<f32> = secs.iter().map(|s| s.volume).collect();
    let total_buy: f64 = secs.iter().map(|s| s.buy_vol as f64).sum();
    let total_vol: f64 = secs.iter().map(|s| s.volume as f64).sum();
    let g01 = if total_vol > 0.0 {
        (total_buy / total_vol) as f32
    } else {
        f32::NAN
    };
    let first_p = secs
        .iter()
        .find(|s| s.has_data && s.first_price > 0.0)
        .map(|s| s.first_price);
    let last_p = secs
        .iter()
        .rfind(|s| s.has_data && s.last_price > 0.0)
        .map(|s| s.last_price);
    let g02 = match (first_p, last_p) {
        (Some(fp), Some(lp)) if fp > 0.0 => (lp - fp) / fp,
        _ => f32::NAN,
    };
    let buy_vols_arr: Vec<f32> = secs.iter().map(|s| s.buy_vol).collect();
    let ret_15s_std = rolling_std_short(&rets, 15);
    let ret_60s_std = rolling_std_short(&rets, 60);
    let br_15_series = rolling_buy_ratio(&buy_vols_arr, &vols, 15);
    let br_60_series = rolling_buy_ratio(&buy_vols_arr, &vols, 60);
    let br_15s_std = rolling_std_short(&br_15_series, 15);
    let br_60s_std = rolling_std_short(&br_60_series, 15);
    let ba_15s_std = rolling_std_short(&bid_asks, 15);
    let ba_60s_std = rolling_std_short(&bid_asks, 60);
    let vol_15s_std = rolling_std_short(&vols, 15);
    let vol_60s_std = rolling_std_short(&vols, 60);
    [
        g01,
        g02,
        safe_std(&ret_15s_std),
        safe_std(&ret_60s_std),
        safe_std(&br_15s_std),
        safe_std(&br_60s_std),
        safe_std(&ba_15s_std),
        safe_std(&ba_60s_std),
        total_vol as f32,
        safe_std(&vol_15s_std),
        safe_std(&vol_60s_std),
    ]
}

// ---------------- 统计工具（与正式库一致） ----------------
fn mean(v: &[f32]) -> f32 {
    let n = v.len();
    if n == 0 {
        return f32::NAN;
    }
    v.iter().sum::<f32>() / n as f32
}
fn std(v: &[f32]) -> f32 {
    let n = v.len();
    if n < 2 {
        return f32::NAN;
    }
    let m = mean(v);
    (v.iter().map(|x| (x - m).powi(2)).sum::<f32>() / n as f32).sqrt()
}
fn skew(v: &[f32]) -> f32 {
    let n = v.len();
    if n < 3 {
        return f32::NAN;
    }
    let m = mean(v);
    let s = std(v);
    if s < 1e-12 {
        return 0.0;
    }
    let m3 = v.iter().map(|x| (x - m).powi(3)).sum::<f32>() / n as f32;
    m3 / s.powi(3)
}
fn kurtosis(v: &[f32]) -> f32 {
    let n = v.len();
    if n < 4 {
        return f32::NAN;
    }
    let m = mean(v);
    let s = std(v);
    if s < 1e-12 {
        return 0.0;
    }
    let m4 = v.iter().map(|x| (x - m).powi(4)).sum::<f32>() / n as f32;
    m4 / s.powi(4) - 3.0
}
fn percentile_sorted(sorted: &[f32], p: f32) -> f32 {
    let n = sorted.len();
    if n == 0 {
        return f32::NAN;
    }
    let idx = (p * (n - 1) as f32) as usize;
    sorted[idx.min(n - 1)]
}
fn rank_pct_in(sorted: &[f32], val: f32) -> f32 {
    if sorted.is_empty() || !val.is_finite() {
        return f32::NAN;
    }
    let pos = sorted.partition_point(|&v| v < val);
    pos as f32 / sorted.len().max(1) as f32
}
fn herfindahl(v: &[f32]) -> f32 {
    let total: f32 = v.iter().sum();
    if total <= 0.0 {
        return f32::NAN;
    }
    v.iter().map(|x| (x / total).powi(2)).sum()
}
fn top_k_concentration(v: &[f32], k: usize) -> f32 {
    let total: f32 = v.iter().sum();
    if total <= 0.0 {
        return f32::NAN;
    }
    let mut sorted: Vec<f32> = v.to_vec();
    sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let top_sum: f32 = sorted.iter().take(k).sum();
    top_sum / total
}
#[inline]
fn corr_fast(a: &[f32], b: &[f32]) -> f32 {
    let mut n: i32 = 0;
    let mut sa = 0.0f64;
    let mut sb = 0.0f64;
    let mut saa = 0.0f64;
    let mut sbb = 0.0f64;
    let mut sab = 0.0f64;
    for i in 0..a.len() {
        let av = a[i];
        let bv = b[i];
        if av.is_finite() && bv.is_finite() {
            n += 1;
            let avd = av as f64;
            let bvd = bv as f64;
            sa += avd;
            sb += bvd;
            saa += avd * avd;
            sbb += bvd * bvd;
            sab += avd * bvd;
        }
    }
    if n < 3 {
        return f32::NAN;
    }
    let nf = n as f64;
    let num = sab - sa * sb / nf;
    let den = ((saa - sa * sa / nf) * (sbb - sb * sb / nf)).sqrt();
    if den < 1e-15 {
        return f32::NAN;
    }
    (num / den) as f32
}
#[inline(always)]
fn neighbor_mean_inline(values: &[f32], stock_rank: usize, k: usize) -> f32 {
    let n = values.len();
    if n < 2 {
        return f32::NAN;
    }
    let start = if stock_rank >= k / 2 { stock_rank - k / 2 } else { 0 };
    let end = (start + k).min(n);
    let mut sum = 0.0f32;
    let mut cnt = 0usize;
    for i in start..end {
        if i != stock_rank && values[i].is_finite() {
            sum += values[i];
            cnt += 1;
        }
    }
    if cnt == 0 {
        f32::NAN
    } else {
        sum / cnt as f32
    }
}
fn mean_std(v: &[f32]) -> (f32, f32) {
    let n = v.len();
    if n < 2 {
        return (f32::NAN, f32::NAN);
    }
    let m = v.iter().sum::<f32>() / n as f32;
    let var = v.iter().map(|x| (x - m).powi(2)).sum::<f32>() / n as f32;
    (m, var.sqrt())
}

/// 组特征 34 维（与正式库 build_group_features_arr 一致）
#[allow(clippy::too_many_arguments)]
fn build_group_features_arr(
    br_finite: &[f32],
    ba_finite: &[f32],
    ret_finite: &[f32],
    vol_finite: &[f32],
    sto_buy: &[f32],
    sto_ba: &[f32],
    sto_ret: &[f32],
    br_sorted: &[f32],
    ba_sorted: &[f32],
    mean_br: f32,
    mean_ba: f32,
    mkt_mean_d: f32,
    mkt_mean_ba: f32,
) -> [f32; 34] {
    let mut f = [f32::NAN; 34];
    f[0] = mean_br;
    f[1] = std(br_finite);
    f[2] = skew(br_finite);
    f[3] = kurtosis(br_finite);
    f[4] = if br_sorted.len() >= 2 {
        percentile_sorted(&br_sorted, 0.90) - percentile_sorted(&br_sorted, 0.10)
    } else {
        f32::NAN
    };
    f[10] = if mkt_mean_d.is_finite() && mean_br.is_finite() {
        mean_br - mkt_mean_d
    } else {
        f32::NAN
    };
    f[11] = corr_fast(sto_buy, sto_ret);
    f[13] = mean_ba;
    f[14] = std(ba_finite);
    f[15] = skew(ba_finite);
    f[16] = kurtosis(ba_finite);
    f[17] = if ba_sorted.len() >= 2 {
        percentile_sorted(&ba_sorted, 0.90) - percentile_sorted(&ba_sorted, 0.10)
    } else {
        f32::NAN
    };
    f[23] = if mkt_mean_ba.is_finite() && mean_ba.is_finite() {
        mean_ba - mkt_mean_ba
    } else {
        f32::NAN
    };
    f[24] = corr_fast(sto_ba, sto_ret);
    f[26] = herfindahl(vol_finite);
    f[27] = top_k_concentration(vol_finite, 3);
    f[30] = mean(ret_finite);
    f[31] = std(ret_finite);
    f[32] = skew(ret_finite);
    f
}

/// per-stock 填充 A08-A10, B08-B10, C03（与正式库 fill_per_stock_arr 一致）
#[allow(clippy::too_many_arguments)]
fn fill_per_stock_arr(
    feats: &mut [f32; 34],
    br_sorted: &[f32],
    ba_sorted: &[f32],
    vol_sorted: &[f32],
    sb: f32,
    sba: f32,
    sv: f32,
    br_m: f32,
    br_s: f32,
    ba_m: f32,
    ba_s: f32,
) {
    feats[7] = rank_pct_in(br_sorted, sb);
    feats[8] = if br_s > 1e-12 && sb.is_finite() {
        (sb - br_m) / br_s
    } else {
        f32::NAN
    };
    feats[9] = if sb.is_finite() && !br_sorted.is_empty() {
        sb - br_sorted[br_sorted.len() / 2]
    } else {
        f32::NAN
    };
    feats[20] = rank_pct_in(ba_sorted, sba);
    feats[21] = if ba_s > 1e-12 && sba.is_finite() {
        (sba - ba_m) / ba_s
    } else {
        f32::NAN
    };
    feats[22] = if sba.is_finite() && !ba_sorted.is_empty() {
        sba - ba_sorted[ba_sorted.len() / 2]
    } else {
        f32::NAN
    };
    feats[28] = rank_pct_in(vol_sorted, sv);
}

// ---------------- 主计算 ----------------
type Feat40 = [f32; FEAT_PER_INCLUSION];
type Rel5 = [f32; 5]; // [rk_buy, rk_vol, rk_ba, z_norm, freq]

struct PiOut {
    // 组 1: rel[gt][stock] = Vec<[buy, vol, ba, z, seq]>  (前 4 回填)
    rel_hot: Vec<Vec<[f32; 4]>>,
    rel_cold: Vec<Vec<[f32; 4]>>,
    // 组 2: 全市场 pool_log
    pool_log_hot: Vec<Vec<usize>>,
    pool_log_cold: Vec<Vec<usize>>,
    // 组 3: 每股 hot/cold 入选次数 + 行业统计
    hot_cnt: Vec<u32>,
    cold_cnt: Vec<u32>,
    hot_sum_ind: [u32; 32],
    cold_sum_ind: [u32; 32],
    hot_nstock_ind: [u32; 32],
    cold_nstock_ind: [u32; 32],
    mkt_hot_total: u32,
    mkt_cold_total: u32,
    // 组 4: 行业内桶累积 + 同行 pool log
    indpool_hot: Vec<Vec<Feat40>>,
    indpool_cold: Vec<Vec<Feat40>>,
    ind_pool_log_hot: [Vec<Vec<usize>>; 32],
    ind_pool_log_cold: [Vec<Vec<usize>>; 32],
}

fn compute_pi(
    pi: usize,
    x: usize,
    d_type: usize,
    min_trades: u32,
    caches: &[RollingCache],
    stocks: &[StockData],
) -> PiOut {
    let n_valid = stocks.len();
    let d_field: u8 = if d_type == 0 { 0 } else { 1 };

    let mut rel_hot: Vec<Vec<[f32; 4]>> = vec![Vec::new(); n_valid];
    let mut rel_cold: Vec<Vec<[f32; 4]>> = vec![Vec::new(); n_valid];
    let mut pool_log_hot: Vec<Vec<usize>> = Vec::new();
    let mut pool_log_cold: Vec<Vec<usize>> = Vec::new();
    let mut hot_cnt = vec![0u32; n_valid];
    let mut cold_cnt = vec![0u32; n_valid];
    let mut hot_sum_ind = [0u32; 32];
    let mut cold_sum_ind = [0u32; 32];
    let mut hot_nstock_ind = [0u32; 32];
    let mut cold_nstock_ind = [0u32; 32];
    let mut mkt_hot_total: u32 = 0;
    let mut mkt_cold_total: u32 = 0;
    let mut indpool_hot: Vec<Vec<Feat40>> = vec![Vec::new(); n_valid];
    let mut indpool_cold: Vec<Vec<Feat40>> = vec![Vec::new(); n_valid];
    let mut ind_pool_log_hot: [Vec<Vec<usize>>; 32] = std::array::from_fn(|_| Vec::new());
    let mut ind_pool_log_cold: [Vec<Vec<usize>>; 32] = std::array::from_fn(|_| Vec::new());

    // 组 4 状态：per gt per 行业
    let mut prev_mean_br = [[f32::NAN; 32]; 2];
    let mut prev2_mean_br = [[f32::NAN; 32]; 2];
    let mut prev_mean_ba = [[f32::NAN; 32]; 2];
    let mut prev2_mean_ba = [[f32::NAN; 32]; 2];
    let mut prev_sec = [[usize::MAX; 32]; 2];
    let mut prev_bucket: [[Vec<usize>; 32]; 2] = std::array::from_fn(|_| std::array::from_fn(|_| Vec::new()));
    let mut stay_seconds = vec![[0u32; 2]; n_valid];
    let mut stay_seen = vec![[false; 2]; n_valid]; // 本 gt 本股已更新过 stay 的标记

    // z-score 历史
    let mut d_hist: Vec<std::collections::VecDeque<f32>> = (0..n_valid)
        .map(|_| std::collections::VecDeque::with_capacity(HIST_WIN / SECOND_STEP + 1))
        .collect();

    let mut buf_all_vals_d = vec![f32::NAN; n_valid];
    let mut buf_all_ba_vals = vec![f32::NAN; n_valid];
    let mut ind_buf: [Vec<usize>; 32] = std::array::from_fn(|_| Vec::new());

    for sec in (15..ADJUSTED_SECONDS).step_by(SECOND_STEP) {
        if sec < x - 1 {
            continue;
        }
        for (si, cache) in caches.iter().enumerate() {
            let dv = cache.get_by_x(x, d_field, sec);
            let bv = cache.get_by_x(x, 1, sec);
            buf_all_vals_d[si] = dv;
            buf_all_ba_vals[si] = bv;
            if dv.is_finite() {
                let h = &mut d_hist[si];
                h.push_back(dv);
                while h.len() > HIST_WIN / SECOND_STEP {
                    h.pop_front();
                }
            }
        }
        let mut top_pairs: Vec<(usize, f32, f32)> = Vec::new(); // (stock, D, z)
        let mut bot_pairs: Vec<(usize, f32, f32)> = Vec::new();
        for si in 0..n_valid {
            let dv = buf_all_vals_d[si];
            if !dv.is_finite() {
                continue;
            }
            let trades = caches[si].get_by_x(x, 4, sec);
            if trades < min_trades as f32 {
                continue;
            }
            let h = &d_hist[si];
            if h.len() < 5 {
                continue;
            }
            let hmean: f64 = h.iter().map(|v| *v as f64).sum::<f64>() / h.len() as f64;
            let hvar: f64 = h.iter().map(|v| (*v as f64 - hmean).powi(2)).sum::<f64>() / h.len() as f64;
            let hstd = hvar.sqrt();
            if hstd < 1e-8 {
                continue;
            }
            let zscore = (dv as f64 - hmean) / hstd;
            if zscore > Z_THRESH {
                top_pairs.push((si, dv, zscore as f32));
            } else if zscore < -Z_THRESH {
                bot_pairs.push((si, dv, zscore as f32));
            }
        }
        if top_pairs.is_empty() && bot_pairs.is_empty() {
            continue;
        }
        top_pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        bot_pairs.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        let top_indices: Vec<usize> = top_pairs.iter().map(|(i, _, _)| *i).collect();
        let bottom_indices: Vec<usize> = bot_pairs.iter().map(|(i, _, _)| *i).collect();

        let mkt_total_vol: f32 = (0..n_valid).map(|i| caches[i].get_by_x(x, 3, sec)).sum();
        let mkt_mean_d = {
            let vf: Vec<f32> = buf_all_vals_d.iter().copied().filter(|v| v.is_finite()).collect();
            mean(&vf)
        };
        let mkt_mean_ba = {
            let vf: Vec<f32> = buf_all_ba_vals
                .iter()
                .copied()
                .filter(|v| v.is_finite())
                .collect();
            mean(&vf)
        };

        // ---- 全市场组（初版口径）：只为组 1 记录 rel + 组 2 pool_log + 组 3 计数 ----
        for (gt, pairs) in [0usize, 1usize].iter().zip([&top_pairs, &bot_pairs].iter()) {
            let gt = *gt;
            let pool_idx: Vec<usize> = pairs.iter().map(|(i, _, _)| *i).collect();
            if pool_idx.is_empty() {
                continue;
            }
            pool_log_hot_or_cold(gt, &pool_idx, &mut pool_log_hot, &mut pool_log_cold);
            // rel 记录（行业未知 ind=0 不记录 → 因子 NaN，符合设计）
            for &(si, _dv, z) in pairs.iter() {
                if stocks[si].ind < 1 {
                    continue;
                }
                let buy = caches[si].get_by_x(x, 0, sec);
                let vol = caches[si].get_by_x(x, 3, sec);
                let ba = caches[si].get_by_x(x, 1, sec);
                let rec = [buy, vol, ba, z];
                if gt == 0 {
                    rel_hot[si].push(rec);
                } else {
                    rel_cold[si].push(rec);
                }
            }
            // 组 3 计数
            for &si in pool_idx.iter() {
                if gt == 0 {
                    hot_cnt[si] += 1;
                } else {
                    cold_cnt[si] += 1;
                }
            }
        }

        // ---- 组 4：行业内桶 ----
        for (gt, pairs) in [0usize, 1usize].iter().zip([&top_pairs, &bot_pairs].iter()) {
            let gt = *gt;
            let pool_idx: Vec<usize> = pairs.iter().map(|(i, _, _)| *i).collect();
            if pool_idx.is_empty() {
                continue;
            }
            // 分桶
            for b in ind_buf.iter_mut() {
                b.clear();
            }
            for &si in pool_idx.iter() {
                let ind = stocks[si].ind as usize;
                if ind >= 1 && ind <= 31 {
                    ind_buf[ind].push(si);
                }
            }
            for ind in 1..=31usize {
                let bucket = &ind_buf[ind];
                if bucket.is_empty() {
                    continue;
                }
                let n_b = bucket.len();
                let sto_buy: Vec<f32> = bucket.iter().map(|&i| caches[i].get_by_x(x, 0, sec)).collect();
                let sto_ba: Vec<f32> = bucket.iter().map(|&i| caches[i].get_by_x(x, 1, sec)).collect();
                let sto_ret: Vec<f32> = bucket.iter().map(|&i| caches[i].get_by_x(x, 2, sec)).collect();
                let sto_vol: Vec<f32> = bucket.iter().map(|&i| caches[i].get_by_x(x, 3, sec)).collect();
                let br_finite: Vec<f32> = sto_buy.iter().copied().filter(|v| v.is_finite()).collect();
                let ba_finite: Vec<f32> = sto_ba.iter().copied().filter(|v| v.is_finite()).collect();
                let ret_finite: Vec<f32> = sto_ret.iter().copied().filter(|v| v.is_finite()).collect();
                let vol_finite: Vec<f32> = sto_vol
                    .iter()
                    .copied()
                    .filter(|v| v.is_finite() && *v > 0.0)
                    .collect();
                let mean_br = mean(&br_finite);
                let mean_ba = mean(&ba_finite);
                let pool_total_vol: f32 = sto_vol.iter().sum();
                let c04_val = if mkt_total_vol > 0.0 {
                    pool_total_vol / mkt_total_vol
                } else {
                    f32::NAN
                };
                let mut br_sorted: Vec<f32> = br_finite.clone();
                br_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let mut ba_sorted: Vec<f32> = ba_finite.clone();
                ba_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let vol_sorted: Vec<f32> = {
                    let mut v = vol_finite.clone();
                    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                    v
                };
                let (br_m, br_s) = mean_std(&br_sorted);
                let (ba_m, ba_s) = mean_std(&ba_sorted);
                let mut group_feats: [f32; 34] = build_group_features_arr(
                    &br_finite, &ba_finite, &ret_finite, &vol_finite, &sto_buy, &sto_ba,
                    &sto_ret, &br_sorted, &ba_sorted, mean_br, mean_ba, mkt_mean_d, mkt_mean_ba,
                );
                // 桶级差分（A06/A07/B06/B07）
                group_feats[5] = if prev_mean_br[gt][ind].is_finite() && mean_br.is_finite() {
                    mean_br - prev_mean_br[gt][ind]
                } else {
                    f32::NAN
                };
                group_feats[6] = if prev_mean_br[gt][ind].is_finite()
                    && prev2_mean_br[gt][ind].is_finite()
                    && mean_br.is_finite()
                {
                    (mean_br - prev_mean_br[gt][ind]) - (prev_mean_br[gt][ind] - prev2_mean_br[gt][ind])
                } else {
                    f32::NAN
                };
                group_feats[18] = if prev_mean_ba[gt][ind].is_finite() && mean_ba.is_finite() {
                    mean_ba - prev_mean_ba[gt][ind]
                } else {
                    f32::NAN
                };
                group_feats[19] = if prev_mean_ba[gt][ind].is_finite()
                    && prev2_mean_ba[gt][ind].is_finite()
                    && mean_ba.is_finite()
                {
                    (mean_ba - prev_mean_ba[gt][ind]) - (prev_mean_ba[gt][ind] - prev2_mean_ba[gt][ind])
                } else {
                    f32::NAN
                };

                // 与上一秒该行业桶的连续性
                let (grp_overlap, grp_gap, grp_prev_avail) = if prev_sec[gt][ind] != usize::MAX {
                    // 双指针交集
                    let mut sorted_bucket = bucket.clone();
                    sorted_bucket.sort_unstable();
                    let prev = &prev_bucket[gt][ind];
                    let (mut i, mut j, mut ov) = (0usize, 0usize, 0usize);
                    while i < sorted_bucket.len() && j < prev.len() {
                        match sorted_bucket[i].cmp(&prev[j]) {
                            std::cmp::Ordering::Equal => {
                                ov += 1;
                                i += 1;
                                j += 1;
                            }
                            std::cmp::Ordering::Less => i += 1,
                            std::cmp::Ordering::Greater => j += 1,
                        }
                    }
                    (ov, sec - prev_sec[gt][ind], true)
                } else {
                    (0usize, 0usize, false)
                };

                // stay_seconds 更新（行业内桶口径）
                let contiguous = grp_prev_avail && grp_gap == SECOND_STEP;
                let prev_is_sorted = prev_bucket[gt][ind].windows(2).all(|w| w[0] <= w[1]);
                for &si in bucket.iter() {
                    if !stay_seen[si][gt] {
                        stay_seen[si][gt] = true;
                        let in_prev = if contiguous {
                            prev_is_sorted
                                && prev_bucket[gt][ind].binary_search(&si).is_ok()
                        } else {
                            false
                        };
                        if contiguous && in_prev {
                            stay_seconds[si][gt] += 1;
                        } else {
                            stay_seconds[si][gt] = if contiguous { 0 } else { 1 };
                        }
                    }
                }
                // 清理本步 seen 标记
                for &si in bucket.iter() {
                    stay_seen[si][gt] = false;
                }
                let stay_ge3 = bucket.iter().filter(|&&si| stay_seconds[si][gt] >= 3).count();
                group_feats[33] = stay_ge3 as f32 / n_b.max(1) as f32;

                // per-stock 组装 40 维
                for (rank_i, &stock_i) in bucket.iter().enumerate() {
                    let rank_pct = rank_i as f32 / n_b.max(1) as f32;
                    let mut per_stock = group_feats;
                    fill_per_stock_arr(
                        &mut per_stock, &br_sorted, &ba_sorted, &vol_sorted, sto_buy[rank_i],
                        sto_ba[rank_i], sto_vol[rank_i], br_m, br_s, ba_m, ba_s,
                    );
                    per_stock[12] = neighbor_mean_inline(&sto_buy, rank_i, 3);
                    per_stock[25] = neighbor_mean_inline(&sto_ba, rank_i, 3);
                    per_stock[29] = c04_val;
                    let (cf01, cf02, cf03, cf04, cf05) = if grp_prev_avail {
                        (
                            grp_overlap as f32,
                            grp_overlap as f32 / std::cmp::min(n_b, prev_bucket[gt][ind].len()).max(1) as f32,
                            rank_pct, // 近似: 与上一秒桶内排名差（桶级 prev 未存逐股排名, 用当前排名近似）
                            if grp_gap == SECOND_STEP { 2.0 } else { 1.0 },
                            grp_gap as f32,
                        )
                    } else {
                        (f32::NAN, f32::NAN, f32::NAN, f32::NAN, f32::NAN)
                    };
                    let mut all_feats: Feat40 = [f32::NAN; FEAT_PER_INCLUSION];
                    all_feats[..34].copy_from_slice(&per_stock);
                    all_feats[34] = cf01;
                    all_feats[35] = cf02;
                    all_feats[36] = cf03;
                    all_feats[37] = cf04;
                    all_feats[38] = cf05;
                    all_feats[39] = f32::NAN;
                    if gt == 0 {
                        indpool_hot[stock_i].push(all_feats);
                    } else {
                        indpool_cold[stock_i].push(all_feats);
                    }
                }

                // 同行 pool log
                if gt == 0 {
                    ind_pool_log_hot[ind].push(bucket.clone());
                } else {
                    ind_pool_log_cold[ind].push(bucket.clone());
                }

                // 更新桶级 prev 状态
                prev2_mean_br[gt][ind] = prev_mean_br[gt][ind];
                prev_mean_br[gt][ind] = mean_br;
                prev2_mean_ba[gt][ind] = prev_mean_ba[gt][ind];
                prev_mean_ba[gt][ind] = mean_ba;
                prev_sec[gt][ind] = sec;
                prev_bucket[gt][ind] = {
                    let mut s = bucket.clone();
                    s.sort_unstable();
                    s
                };
            }
        }
    }

    // 组 3 行业统计
    for si in 0..n_valid {
        let ind = stocks[si].ind as usize;
        if ind >= 1 && ind <= 31 {
            if hot_cnt[si] > 0 {
                hot_sum_ind[ind] += hot_cnt[si];
                hot_nstock_ind[ind] += 1;
            }
            if cold_cnt[si] > 0 {
                cold_sum_ind[ind] += cold_cnt[si];
                cold_nstock_ind[ind] += 1;
            }
        }
    }
    mkt_hot_total = hot_sum_ind.iter().sum();
    mkt_cold_total = cold_sum_ind.iter().sum();

    PiOut {
        rel_hot,
        rel_cold,
        pool_log_hot,
        pool_log_cold,
        hot_cnt,
        cold_cnt,
        hot_sum_ind,
        cold_sum_ind,
        hot_nstock_ind,
        cold_nstock_ind,
        mkt_hot_total,
        mkt_cold_total,
        indpool_hot,
        indpool_cold,
        ind_pool_log_hot,
        ind_pool_log_cold,
    }
}

fn pool_log_hot_or_cold(
    gt: usize,
    pool_idx: &[usize],
    pool_log_hot: &mut Vec<Vec<usize>>,
    pool_log_cold: &mut Vec<Vec<usize>>,
) {
    if gt == 0 {
        pool_log_hot.push(pool_idx.to_vec());
    } else {
        pool_log_cold.push(pool_idx.to_vec());
    }
}

/// 组 1 回填：行业内当日全部入选记录的排名百分位 / 截面 z / 频率比
fn backfill_rel(rel: &mut [Vec<[f32; 4]>], stocks: &[StockData]) -> Vec<Vec<[f32; 5]>> {
    let n = rel.len();
    // 行业入选次数与有效股票数
    let mut ind_inc_cnt = [0u32; 32];
    let mut ind_nstock = [0u32; 32];
    for si in 0..n {
        let ind = stocks[si].ind as usize;
        if ind >= 1 && ind <= 31 {
            ind_inc_cnt[ind] += rel[si].len() as u32;
            ind_nstock[ind] += 1;
        }
    }
    // per 行业收集排序回填（4 类值）
    for ind in 1..=31usize {
        if ind_inc_cnt[ind] < 2 {
            continue;
        }
        for kind in 0..4usize {
            let mut vals: Vec<(f32, usize, usize)> = Vec::with_capacity(ind_inc_cnt[ind] as usize);
            for si in 0..n {
                if stocks[si].ind as usize != ind {
                    continue;
                }
                for (k, rec) in rel[si].iter().enumerate() {
                    if rec[kind].is_finite() {
                        vals.push((rec[kind], si, k));
                    }
                }
            }
            if vals.len() < 2 {
                continue;
            }
            vals.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
            let sorted: Vec<f32> = vals.iter().map(|v| v.0).collect();
            if kind == 3 {
                // z 截面标准化
                let (m, s) = mean_std(&sorted);
                if s > 1e-12 {
                    for &(v, si, k) in vals.iter() {
                        rel[si][k][3] = ((v - m) / s) as f32;
                    }
                } else {
                    for &(_, si, k) in vals.iter() {
                        rel[si][k][3] = f32::NAN;
                    }
                }
            } else {
                let mut seq_pos = 0usize;
                for &(v, si, k) in vals.iter() {
                    while seq_pos < sorted.len() && sorted[seq_pos] < v {
                        seq_pos += 1;
                    }
                    rel[si][k][kind] = seq_pos as f32 / sorted.len() as f32;
                }
            }
        }
    }
    // freq: 截至第 k 条的累计次数 / 行业人均入选次数
    let mut out: Vec<Vec<[f32; 5]>> = vec![Vec::new(); n];
    for si in 0..n {
        let ind = stocks[si].ind as usize;
        let denom = if ind >= 1 && ind <= 31 && ind_nstock[ind] > 0 {
            ind_inc_cnt[ind] as f32 / ind_nstock[ind] as f32
        } else {
            f32::NAN
        };
        for (k, rec) in rel[si].iter().enumerate() {
            let mut r5 = [f32::NAN; 5];
            r5[..4].copy_from_slice(rec);
            r5[4] = if denom.is_finite() && denom > 0.0 {
                (k + 1) as f32 / denom
            } else {
                f32::NAN
            };
            out[si].push(r5);
        }
    }
    out
}

/// 组 1/组 4 降维：每列 21 统计量（stat-major）
fn reduce_seq(seq: &[[f32; 5]]) -> Vec<f32> {
    let z = seq.len();
    if z == 0 {
        return vec![f32::NAN; 21 * 5];
    }
    let mut flat = Vec::with_capacity(z * 5);
    for r in seq.iter() {
        flat.extend_from_slice(r);
    }
    reduce::reduce_21_flat(&flat, z, 5)
}

fn reduce_seq40(seq: &[Feat40]) -> Vec<f32> {
    let z = seq.len();
    if z == 0 {
        return vec![f32::NAN; 21 * FEAT_PER_INCLUSION];
    }
    let mut flat = Vec::with_capacity(z * FEAT_PER_INCLUSION);
    for r in seq.iter() {
        flat.extend_from_slice(r);
    }
    reduce::reduce_21_flat(&flat, z, FEAT_PER_INCLUSION)
}

/// 组 2：初版共现 top-10 同行/跨行拆分（返回 hot 块 + cold 块，共 2×G2_PER）
fn g2_cooc(stock_i: usize, pi_out: &PiOut, stocks: &[StockData]) -> [f32; G2_PER * 2] {
    let n_valid = stocks.len();
    let mut out = [f32::NAN; G2_PER * 2];
    let mut buf = vec![0u32; n_valid];
    let self_ind = stocks[stock_i].ind;
    let mut pairs_h: Vec<(usize, u32)> = Vec::new();
    for pool in pi_out.pool_log_hot.iter() {
        for &other in pool.iter() {
            if other != stock_i {
                buf[other] += 1;
            }
        }
    }
    for (si, &cnt) in buf.iter().enumerate() {
        if cnt > 0 && si != stock_i {
            pairs_h.push((si, cnt));
        }
    }
    pairs_h.sort_by(|a, b| b.1.cmp(&a.1));
    let top10_h: Vec<(usize, u32)> = pairs_h.iter().take(10).copied().collect();
    for &(si, _) in pairs_h.iter() {
        buf[si] = 0;
    }
    let mut pairs_c: Vec<(usize, u32)> = Vec::new();
    for pool in pi_out.pool_log_cold.iter() {
        for &other in pool.iter() {
            if other != stock_i {
                buf[other] += 1;
            }
        }
    }
    for (si, &cnt) in buf.iter().enumerate() {
        if cnt > 0 && si != stock_i {
            pairs_c.push((si, cnt));
        }
    }
    pairs_c.sort_by(|a, b| b.1.cmp(&a.1));
    let top10_c: Vec<(usize, u32)> = pairs_c.iter().take(10).copied().collect();

    for (gt, top10) in [0usize, 1usize].iter().zip([&top10_h, &top10_c].iter()) {
        let gt = *gt;
        let same: Vec<usize> = top10
            .iter()
            .filter(|(si, _)| stocks[*si].ind == self_ind && self_ind >= 1)
            .map(|(si, _)| *si)
            .collect();
        let cross: Vec<usize> = top10
            .iter()
            .filter(|(si, _)| stocks[*si].ind != self_ind || self_ind < 1)
            .map(|(si, _)| *si)
            .collect();
        // 结构量
        let ratio = same.len() as f32 / 10.0;
        let cntsum: f32 = top10
            .iter()
            .filter(|(si, _)| stocks[*si].ind == self_ind && self_ind >= 1)
            .map(|(_, c)| *c as f32)
            .sum();
        let base = gt * G2_PER;
        out[base + BASIC_FEAT_N * 2 * 4] = ratio;
        out[base + BASIC_FEAT_N * 2 * 4 + 1] = cntsum;
        for j in 0..BASIC_FEAT_N {
            let same_ms = ms_of(&same, j, stocks);
            let cross_ms = ms_of(&cross, j, stocks);
            for s in 0..2usize {
                let (sv, cv) = (same_ms[s], cross_ms[s]);
                let idx = base + j * 8 + s * 4;
                out[idx] = sv;
                out[idx + 1] = cv;
                out[idx + 2] = if sv.is_finite() && cv.is_finite() {
                    sv - cv
                } else {
                    f32::NAN
                };
                out[idx + 3] = if sv.is_finite() && cv.is_finite() {
                    (sv - cv).abs()
                } else {
                    f32::NAN
                };
            }
        }
    }
    out
}

fn ms_of(peers: &[usize], j: usize, stocks: &[StockData]) -> [f32; 2] {
    let col: Vec<f32> = peers
        .iter()
        .map(|&si| stocks[si].basic_feats[j])
        .filter(|v| v.is_finite())
        .collect();
    if col.len() >= 2 {
        let m = col.iter().sum::<f32>() / col.len() as f32;
        let var = col.iter().map(|v| (v - m).powi(2)).sum::<f32>() / col.len() as f32;
        [m, var.sqrt()]
    } else {
        [f32::NAN, f32::NAN]
    }
}

/// 组 4 共现：同行共现 top-10（初版 4 口径）
fn g4_cooc(stock_i: usize, pi_out: &PiOut, stocks: &[StockData]) -> [f32; BASIC_FEAT_N * 2 * 4] {
    let n_valid = stocks.len();
    let mut out = [f32::NAN; BASIC_FEAT_N * 2 * 4];
    let self_ind = stocks[stock_i].ind as usize;
    if self_ind < 1 || self_ind > 31 {
        return out;
    }
    let mut buf = vec![0u32; n_valid];
    for pool in pi_out.ind_pool_log_hot[self_ind].iter() {
        for &other in pool.iter() {
            if other != stock_i {
                buf[other] += 1;
            }
        }
    }
    let mut pairs_h: Vec<(usize, u32)> = buf
        .iter()
        .enumerate()
        .filter(|&(si, &cnt)| si != stock_i && cnt > 0)
        .map(|(si, &cnt)| (si, cnt))
        .collect();
    pairs_h.sort_by(|a, b| b.1.cmp(&a.1));
    let top10_h: Vec<usize> = pairs_h.iter().take(10).map(|(si, _)| *si).collect();
    for &(si, _) in pairs_h.iter() {
        buf[si] = 0;
    }
    for pool in pi_out.ind_pool_log_cold[self_ind].iter() {
        for &other in pool.iter() {
            if other != stock_i {
                buf[other] += 1;
            }
        }
    }
    let mut pairs_c: Vec<(usize, u32)> = buf
        .iter()
        .enumerate()
        .filter(|&(si, &cnt)| si != stock_i && cnt > 0)
        .map(|(si, &cnt)| (si, cnt))
        .collect();
    pairs_c.sort_by(|a, b| b.1.cmp(&a.1));
    let top10_c: Vec<usize> = pairs_c.iter().take(10).map(|(si, _)| *si).collect();

    let hot_ms = ms_all(&top10_h, stocks);
    let cold_ms = ms_all(&top10_c, stocks);
    for j in 0..BASIC_FEAT_N {
        for s in 0..2usize {
            let hv = hot_ms[s * BASIC_FEAT_N + j];
            let cv = cold_ms[s * BASIC_FEAT_N + j];
            let idx = (j * 2 + s) * 4;
            out[idx] = hv;
            out[idx + 1] = cv;
            out[idx + 2] = if hv.is_finite() && cv.is_finite() {
                hv - cv
            } else {
                f32::NAN
            };
            out[idx + 3] = if hv.is_finite() && cv.is_finite() {
                (hv - cv).abs()
            } else {
                f32::NAN
            };
        }
    }
    out
}

fn ms_all(peers: &[usize], stocks: &[StockData]) -> [f32; BASIC_FEAT_N * 2] {
    let mut out = [f32::NAN; BASIC_FEAT_N * 2];
    for j in 0..BASIC_FEAT_N {
        let col: Vec<f32> = peers
            .iter()
            .map(|&si| stocks[si].basic_feats[j])
            .filter(|v| v.is_finite())
            .collect();
        if col.len() >= 2 {
            let m = col.iter().sum::<f32>() / col.len() as f32;
            let var = col.iter().map(|v| (v - m).powi(2)).sum::<f32>() / col.len() as f32;
            out[j] = m;
            out[BASIC_FEAT_N + j] = var.sqrt();
        }
    }
    out
}

/// 组 3：行业热度聚合（每股返回 [sum, nratio, share, z, own, NaN]；spread 由调用方按行业表回填）
fn g3_heat(
    si: usize,
    pi_out: &PiOut,
    stocks: &[StockData],
    ind_nstock: &[u32; 32],
    sum_m: f32,
    sum_sd: f32,
) -> [f32; G3_PER] {
    let mut out = [f32::NAN; G3_PER];
    let ind = stocks[si].ind as usize;
    if ind < 1 || ind > 31 {
        return out;
    }
    let hs = pi_out.hot_sum_ind[ind] as f32;
    let nratio = if ind_nstock[ind] > 0 {
        pi_out.hot_nstock_ind[ind] as f32 / ind_nstock[ind] as f32
    } else {
        f32::NAN
    };
    let share = if pi_out.mkt_hot_total > 0 {
        hs / pi_out.mkt_hot_total as f32
    } else {
        f32::NAN
    };
    let z = if sum_sd > 1e-12 { (hs - sum_m) / sum_sd } else { f32::NAN };
    let own = if hs > 0.0 {
        pi_out.hot_cnt[si] as f32 / hs
    } else {
        f32::NAN
    };
    out[0] = hs;
    out[1] = nratio;
    out[2] = share;
    out[3] = z;
    out[4] = own;
    out[5] = f32::NAN; // spread 由调用方回填
    out
}

// ---------------- 因子名 ----------------
pub fn factor_names() -> Vec<String> {
    let mut names = Vec::with_capacity(N_FACTORS);
    let rel_feats = ["rk_buy", "rk_vol", "rk_ba", "z", "freq"];
    let stat_names = [
        "mean", "median", "std", "skew", "kurt", "p5", "p25", "p75", "p95", "iqr", "cv",
        "autocorr1", "autocorr1_abs", "trend", "curvature", "quad_coef", "period_diff",
        "period_ratio", "lz_complexity", "entropy_1d", "max_range_product",
    ];
    let basic_names = [
        "total_buy_ratio",
        "total_return",
        "ret_15s_std",
        "ret_60s_std",
        "buy_ratio_15s_std",
        "buy_ratio_60s_std",
        "bid_ask_15s_std",
        "bid_ask_60s_std",
        "total_volume",
        "vol_15s_std",
        "vol_60s_std",
    ];
    // 组 1（值布局 stat-major：stat 外层、feat 内层，与 reduce_21_flat 一致）
    for combo in COMBO_LABELS.iter() {
        for grp in ["hot", "cold"].iter() {
            for stat in stat_names.iter() {
                for feat in rel_feats.iter() {
                    names.push(format!("hotpool_ext_ind_rel_{combo}_{grp}_{feat}_{stat}"));
                }
            }
        }
    }
    // 组 2
    for combo in COMBO_LABELS.iter() {
        for grp in ["hot", "cold"].iter() {
            for b in basic_names.iter() {
                for s in ["mean", "std"].iter() {
                    for kind in ["same", "cross", "diff", "absdiff"].iter() {
                        names.push(format!("hotpool_ext_ind_cooc_{combo}_{grp}_{b}_{s}_{kind}"));
                    }
                }
            }
            names.push(format!("hotpool_ext_ind_cooc_{combo}_{grp}_struct_ratio"));
            names.push(format!("hotpool_ext_ind_cooc_{combo}_{grp}_struct_cntsum"));
        }
    }
    // 组 3
    for combo in COMBO_LABELS.iter() {
        for metric in ["sum", "nratio", "share", "z", "own", "spread"].iter() {
            names.push(format!("hotpool_ext_ind_heat_{combo}_{metric}"));
        }
    }
    // 组 4（值布局 stat-major：stat 外层、col 内层，与 reduce_21_flat 一致）
    let col_names: Vec<String> = (0..FEAT_PER_INCLUSION).map(|i| format!("f{i:02}")).collect();
    for combo in COMBO_LABELS.iter() {
        for grp in ["hot", "cold"].iter() {
            for stat in stat_names.iter() {
                for cn in col_names.iter() {
                    names.push(format!("hotpool_ext_indpool_{combo}_{grp}_{cn}_{stat}"));
                }
            }
        }
    }
    for combo in COMBO_LABELS.iter() {
        for b in basic_names.iter() {
            for s in ["mean", "std"].iter() {
                for t in ["hot", "cold", "diff", "abs_diff"].iter() {
                    names.push(format!("hotpool_ext_indpool_cooc_{combo}_{b}_{s}_{t}"));
                }
            }
        }
    }
    assert_eq!(names.len(), N_FACTORS);
    names
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("用法: hotpool_ind_sandbox <date> <industry_csv> [--only CODE]");
        std::process::exit(1);
    }
    let date: i64 = args[1].parse().expect("date");
    let ind_path = args[2].clone();
    let only: Option<String> = args
        .iter()
        .position(|a| a == "--only")
        .map(|i| args.get(i + 1).cloned().unwrap_or_default());

    let ind_map = load_industry(&ind_path);
    let mut codes = list_codes(date);
    if let Some(c) = &only {
        codes = codes.into_iter().filter(|x| x == c).collect();
    }
    eprintln!("date={} codes={} ind_map={}", date, codes.len(), ind_map.len());

    let t0 = std::time::Instant::now();
    let stocks: Vec<Option<StockData>> = codes
        .par_iter()
        .map(|code| {
            let ind = *ind_map.get(code).unwrap_or(&0);
            build_stock_data(code, date, ind)
        })
        .collect();
    let mut valid_stocks: Vec<StockData> = Vec::new();
    for s in stocks.into_iter() {
        if let Some(sd) = s {
            if sd.secs.iter().any(|s| s.has_data) {
                valid_stocks.push(sd);
            }
        }
    }
    let n_valid = valid_stocks.len();
    eprintln!("per-stock done in {:?}, valid={}", t0.elapsed(), n_valid);
    if n_valid == 0 {
        return;
    }

    let t1 = std::time::Instant::now();
    let rolling_caches: Vec<RollingCache> = valid_stocks
        .par_iter()
        .map(|sd| RollingCache::compute(&sd.secs))
        .collect();
    eprintln!("rolling cache done in {:?}", t1.elapsed());

    let t2 = std::time::Instant::now();
    let pi_outs: Vec<PiOut> = PARAM_CONFIGS
        .par_iter()
        .enumerate()
        .map(|(pi, &(x, d_type, min_trades))| compute_pi(pi, x, d_type, min_trades, &rolling_caches, &valid_stocks))
        .collect();
    eprintln!("main loop done in {:?}", t2.elapsed());

    // 组 1 回填
    let t3 = std::time::Instant::now();
    let mut rel_back: Vec<(Vec<Vec<[f32; 5]>>, Vec<Vec<[f32; 5]>>)> = Vec::new();
    for po in pi_outs.iter() {
        let mut rh = po.rel_hot.clone();
        let mut rc = po.rel_cold.clone();
        let bfh = backfill_rel(&mut rh, &valid_stocks);
        let bfc = backfill_rel(&mut rc, &valid_stocks);
        rel_back.push((bfh, bfc));
    }
    eprintln!("g1 backfill done in {:?}", t3.elapsed());

    // 组 3 预计算：行业有效股票数 + heat_sum 截面 mean/std + spread 行业表
    let mut ind_nstock = [0u32; 32];
    for s in valid_stocks.iter() {
        let ind = s.ind as usize;
        if ind >= 1 && ind <= 31 {
            ind_nstock[ind] += 1;
        }
    }
    let g3_meta: Vec<(f32, f32, [f32; 32])> = pi_outs
        .iter()
        .map(|po| {
            let sums: Vec<f32> = (1..=31).map(|i| po.hot_sum_ind[i] as f32).collect();
            let (m, sd) = mean_std(&sums);
            let mut spread_ind = [f32::NAN; 32];
            for i in 1..=31usize {
                spread_ind[i] = po.hot_sum_ind[i] as f32 - po.cold_sum_ind[i] as f32;
            }
            (m, sd, spread_ind)
        })
        .collect();

    // 组装每股因子
    let t4 = std::time::Instant::now();
    let all_factors: Vec<Vec<f32>> = (0..n_valid)
        .into_par_iter()
        .map(|stock_i| {
            let mut facs = vec![f32::NAN; N_FACTORS];
            let mut off = 0usize;
            // 组 1
            for pi in 0..N_PARAM_COMBOS {
                for seq in [&rel_back[pi].0[stock_i], &rel_back[pi].1[stock_i]] {
                    let vals = reduce_seq(seq);
                    facs[off..off + vals.len()].copy_from_slice(&vals);
                    off += vals.len();
                }
            }
            // 组 2
            for pi in 0..N_PARAM_COMBOS {
                let vals = g2_cooc(stock_i, &pi_outs[pi], &valid_stocks);
                facs[off..off + vals.len()].copy_from_slice(&vals);
                off += vals.len();
            }
            // 组 3
            for pi in 0..N_PARAM_COMBOS {
                let (m, sd, spread_ind) = &g3_meta[pi];
                let mut vals = g3_heat(stock_i, &pi_outs[pi], &valid_stocks, &ind_nstock, *m, *sd);
                let ind = valid_stocks[stock_i].ind as usize;
                if ind >= 1 && ind <= 31 {
                    vals[5] = spread_ind[ind];
                }
                facs[off..off + vals.len()].copy_from_slice(&vals);
                off += vals.len();
            }
            // 组 4 降维
            for pi in 0..N_PARAM_COMBOS {
                for seq in [&pi_outs[pi].indpool_hot[stock_i], &pi_outs[pi].indpool_cold[stock_i]] {
                    let vals = reduce_seq40(seq);
                    facs[off..off + vals.len()].copy_from_slice(&vals);
                    off += vals.len();
                }
            }
            // 组 4 共现
            for pi in 0..N_PARAM_COMBOS {
                let vals = g4_cooc(stock_i, &pi_outs[pi], &valid_stocks);
                facs[off..off + vals.len()].copy_from_slice(&vals);
                off += vals.len();
            }
            facs
        })
        .collect();
    eprintln!("assemble done in {:?}", t4.elapsed());

    // 输出
    let out_dir = format!("out/{}", date);
    std::fs::create_dir_all(&out_dir).expect("mkdir");
    let names = factor_names();
    std::fs::write(
        format!("{}/names.json", out_dir),
        serde_json::to_string(&names).unwrap(),
    )
    .expect("write names");
    let codes_out: Vec<String> = valid_stocks.iter().map(|s| s.code.clone()).collect();
    std::fs::write(
        format!("{}/codes.txt", out_dir),
        codes_out.join("\n"),
    )
    .expect("write codes");
    let mut bin = Vec::with_capacity(n_valid * N_FACTORS * 4);
    for facs in all_factors.iter() {
        for v in facs.iter() {
            bin.extend_from_slice(&v.to_le_bytes());
        }
    }
    std::fs::write(format!("{}/facs.bin", out_dir), &bin).expect("write facs");
    eprintln!("total done in {:?}, wrote {} x {}", t0.elapsed(), n_valid, N_FACTORS);
}
