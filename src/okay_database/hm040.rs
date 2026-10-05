//! hm40「洗澡 & 保洁员应聘」——旧框架（design_whatever_out.TickDataBase，Level2 **逐笔**）的 Rust 复现。
//!
//! 旧文件：/home/chenzongwei/pythoncode/成果hm40_洗澡&保洁员应聘/hm40_okay.py
//!   run_base 的 Go.get_factor，get_many=0 / read_full=0 / change_time=1，
//!   read_trade_data_single(with_retreat=0)，也就是单张成交明细表，
//!   因此本文件只有 compute_stock，没有 compute_cross。
//!
//! ════════════════════════════════════════════════════════════════════
//! 一、旧代码逐步对照
//! ════════════════════════════════════════════════════════════════════
//! 1) df = read_trade_data_single(with_retreat=0)
//!      · 丢掉 flag == 32（撤单）。库里 read_trade_fast_inner 的 with_retreat 按字节比 "32"，
//!        对 2025 年的 "32.0" 不生效，所以本文件按规范自己按数值过滤。
//!      · change_time=1 → deal_data：exchtime 整数微秒 + 8h 变成 Timestamp。
//!        本文件保留微秒整数（time_us = exchtime_us + 8h），窗口按微秒闭区间切。
//!      · 文件不存在 → 旧代码 df is None → get_factor 返回 None。
//!
//! 2) 下午平移（只做一次，作用于整张表）：
//!        df1 = df[09:30:00 ~ 11:30:00]
//!        df2 = df[13:00:00 ~ 14:57:00] 再整体减 90 分钟（变成 11:30 ~ 13:27）
//!        df  = concat([df1, df2])
//!    然后在这张平移后的表上切 5 个窗口（between_time 两端都含、精确到微秒）：
//!        allday = df            （真实 09:30~11:30 + 13:00~14:57）
//!        open30 = [09:30,10:00] （真实）
//!        morning = [09:30,11:30]（真实）
//!        afternoon = [11:30,13:27]（平移后 = 真实 13:00~14:57）
//!        close30 = [13:00,13:27]  （平移后 = 真实 14:30~14:57）
//!    窗口内 exchtime 再减窗口内最小值变成「相对秒数」，所以 90 分钟平移只在 allday
//!    这张混合表里有意义（下午的行要减 90 分钟再和上午的行一起排序）。
//!
//! 3) 每个窗口算 108 个值 = sing1(54) + sing2(54)：
//!        sing1 = cal_high(flag==83) + cal_low(flag==66)     → 命名前缀 smart_high / smart_low
//!        sing2 = cal_high(flag==66) + cal_low(flag==83)     → 命名前缀 fool_high  / fool_low
//!    （sing1 与 sing2 逐行相同，只有 6 处 flag 常量互换，已用 diff 核过。）
//!    每个 cal_* 返回 27 个数，返回前把 ±inf 抹成 NaN（np.isinf）。
//!
//! 4) 价格分箱 pd.cut(df.price, bins=10, labels=range(10))：
//!        mn, mx = 价格的最小/最大
//!        mn == mx 时：mn -= 0.001*|mn|（mn==0 时减 0.001），mx 同理加；
//!                     再 linspace(mn, mx, 11)，**不做** bins[0] 的额外平移
//!        mn <  mx 时：bins = linspace(mn, mx, 11)，bins[0] -= (mx-mn)*0.001
//!        落箱：code = searchsorted(bins, x, side="left") - 1（即「小于 x 的边数 - 1」），
//!              code 为 0 或 11（越界）→ NaN
//!     本文件按源码复刻，并用真实离散价格与 pandas 对照过 300 组随机数据、零差异。
//!     · pd.cut 对空数组会 raise；这里只在窗口行数 > 10 时才调，不会遇到。
//!     · 价格含 ±inf 时 pandas 抛 ValueError → 整股返回 None。
//!
//! 5) 箱号之后：
//!        high    = 出现过的最大箱号；subhigh = 比 high 小的箱号里最大的那个
//!        low     = 出现过的最小箱号；sublow  = 比 low 大的箱号里最小的那个
//!        high_time = 箱号 == high 的**第一行**（按行序，也就是时间序）的相对秒数
//!        df_subhigh_real = 所有 rel >= high_time 的行（注意是**所有行**，不只是次高价的行）
//!        df_sublow_real  = 所有 rel >= low_time 的行
//!     次高价箱不存在（或 real 为空）时，该 cal_* 整组 27 个值给 NaN。
//!
//! 6) **旧代码的一处笔误必须照抄**：cal_high 里 `inner83_volume_high_mean` 被赋值两次，
//!    第二次用的是次高价箱的均值（volume_subhigh_83_mean / volume_subhigh_mean），
//!    把第一次的值覆盖掉了；而返回列表里这个名字出现了两次（第 13、16 位），
//!    所以两处都是「次高价」那个值。cal_low 里的 `inner66_volume_low_mean` 同理。
//!    本文件照样算两次、写同一个值，保证与旧值一致。
//!
//! 7) 因子名：4 组 × 27 名 × 5 窗口 = 540，但 `smart_high_inner83_volume_high_mean`
//!    这类名字在每组里出现两次（对应上面那个笔误），所以 540 个名字里有 20 个是重名。
//!    列式 store 不接受重名，按规范给第二次及以后的出现加 `_dup2` 后缀（值不变）。
//!
//! ════════════════════════════════════════════════════════════════════
//! 二、数值细节
//! ════════════════════════════════════════════════════════════════════
//! · 逐笔价格/成交量按 **f64** 自己解析 CSV：旧框架用 pandas 读的是 float64，
//!   库里的 read_trade_fast_inner 是 f32，价格分箱的边界会被末位差带偏。
//! · pandas 的 Series.sum()/mean() 是 numpy 的成对求和（NaN 先填 0、按原位参与归约），
//!   本文件照 sample 复刻（与 hm051 同一份 pair_sum）。空选择的 sum 是 0.0、mean 是 NaN。
//! · 相对秒数用 (time_us - min)/1e6：pandas 那条链是 Timestamp 相减再 total_seconds()，
//!   多一次 ns 取整，误差 ~2e-9 秒，远低于 1e-4 的验收门槛。
//! · 因子个数恒定：算不出来整条 NaN，不条件跳过。

use std::collections::HashSet;
use std::fs::File;
use std::io::Read;

const SEC_US: i64 = 1_000_000;
/// 下午整体前移 90 分钟
const AFTERNOON_SHIFT: i64 = 90 * 60 * SEC_US;

/// 窗口数、每窗口因子数、总因子数
const N_WIN: usize = 5;
const N_PER_WIN: usize = 108;
pub const N_OUT: usize = N_WIN * N_PER_WIN;

/// 每个 cal_* 的 27 个量（顺序 = 旧代码返回列表的顺序）
const N_CAL: usize = 27;

/// 4 组 × 27 个名字（顺序 = 旧 names 的前 108 项）
const BLOCKS: [[&str; N_CAL]; 4] = [
    [
        "smart_high_volume_sum",
        "smart_high_volume_mean",
        "smart_high_time_mean",
        "smart_high_volume_83_sum",
        "smart_high_volume_83_mean",
        "smart_high_time_83_mean",
        "smart_high_volume_real_sum",
        "smart_high_volume_real_mean",
        "smart_high_time_real_mean",
        "smart_high_volume_real_83_sum",
        "smart_high_volume_real_83_mean",
        "smart_high_time_real_83_mean",
        "smart_high_inner83_volume_high_sum",
        "smart_high_inner83_volume_high_mean",
        "smart_high_inner83_time_high_mean",
        "smart_high_inner83_volume_subhigh_sum",
        "smart_high_inner83_volume_high_mean",
        "smart_high_inner83_time_subhigh_mean",
        "smart_high_inner83_volume_real_sum",
        "smart_high_inner83_volume_real_mean",
        "smart_high_inner83_time_real_mean",
        "smart_high_innerreal_volume_sum",
        "smart_high_innerreal_volume_mean",
        "smart_high_innerreal_time_mean",
        "smart_high_innerreal_volume_83_sum",
        "smart_high_innerreal_volume_83_mean",
        "smart_high_innerreal_time_83_mean",
    ],
    [
        "smart_low_volume_sum",
        "smart_low_volume_mean",
        "smart_low_time_mean",
        "smart_low_volume_66_sum",
        "smart_low_volume_66_mean",
        "smart_low_time_66_mean",
        "smart_low_volume_real_sum",
        "smart_low_volume_real_mean",
        "smart_low_time_real_mean",
        "smart_low_volume_real_66_sum",
        "smart_low_volume_real_66_mean",
        "smart_low_time_real_66_mean",
        "smart_low_inner66_volume_low_sum",
        "smart_low_inner66_volume_low_mean",
        "smart_low_inner66_time_low_mean",
        "smart_low_inner66_volume_sublow_sum",
        "smart_low_inner66_volume_low_mean",
        "smart_low_inner66_time_sublow_mean",
        "smart_low_inner66_volume_real_sum",
        "smart_low_inner66_volume_real_mean",
        "smart_low_inner66_time_real_mean",
        "smart_low_innerreal_volume_sum",
        "smart_low_innerreal_volume_mean",
        "smart_low_innerreal_time_mean",
        "smart_low_innerreal_volume_66_sum",
        "smart_low_innerreal_volume_66_mean",
        "smart_low_innerreal_time_66_mean",
    ],
    [
        "fool_high_volume_sum",
        "fool_high_volume_mean",
        "fool_high_time_mean",
        "fool_high_volume_66_sum",
        "fool_high_volume_66_mean",
        "fool_high_time_66_mean",
        "fool_high_volume_real_sum",
        "fool_high_volume_real_mean",
        "fool_high_time_real_mean",
        "fool_high_volume_real_66_sum",
        "fool_high_volume_real_66_mean",
        "fool_high_time_real_66_mean",
        "fool_high_inner66_volume_high_sum",
        "fool_high_inner66_volume_high_mean",
        "fool_high_inner66_time_high_mean",
        "fool_high_inner66_volume_subhigh_sum",
        "fool_high_inner66_volume_high_mean",
        "fool_high_inner66_time_subhigh_mean",
        "fool_high_inner66_volume_real_sum",
        "fool_high_inner66_volume_real_mean",
        "fool_high_inner66_time_real_mean",
        "fool_high_innerreal_volume_sum",
        "fool_high_innerreal_volume_mean",
        "fool_high_innerreal_time_mean",
        "fool_high_innerreal_volume_66_sum",
        "fool_high_innerreal_volume_66_mean",
        "fool_high_innerreal_time_66_mean",
    ],
    [
        "fool_low_volume_sum",
        "fool_low_volume_mean",
        "fool_low_time_mean",
        "fool_low_volume_83_sum",
        "fool_low_volume_83_mean",
        "fool_low_time_83_mean",
        "fool_low_volume_real_sum",
        "fool_low_volume_real_mean",
        "fool_low_time_real_mean",
        "fool_low_volume_real_83_sum",
        "fool_low_volume_real_83_mean",
        "fool_low_time_real_83_mean",
        "fool_low_inner83_volume_low_sum",
        "fool_low_inner83_volume_low_mean",
        "fool_low_inner83_time_low_mean",
        "fool_low_inner83_volume_sublow_sum",
        "fool_low_inner83_volume_low_mean",
        "fool_low_inner83_time_sublow_mean",
        "fool_low_inner83_volume_real_sum",
        "fool_low_inner83_volume_real_mean",
        "fool_low_inner83_time_real_mean",
        "fool_low_innerreal_volume_sum",
        "fool_low_innerreal_volume_mean",
        "fool_low_innerreal_time_mean",
        "fool_low_innerreal_volume_83_sum",
        "fool_low_innerreal_volume_83_mean",
        "fool_low_innerreal_time_83_mean",
    ],
];

/// 窗口后缀（顺序 = 旧 names 的 5 个块）
const WINDOWS: [&str; N_WIN] = ["allday", "open30", "morning", "afternoon", "close30"];

/// 因子名：第二次及以后的重名加 `_dup2`（旧 names_single 里有 20 个重名，
/// 来自 cal_high/cal_low 里那个被赋值两次的 `inner*_volume_*_mean`）。
pub fn names() -> Vec<String> {
    let mut seen: HashSet<String> = HashSet::new();
    let mut out = Vec::with_capacity(N_OUT);
    for w in WINDOWS.iter() {
        for b in BLOCKS.iter() {
            for m in b.iter() {
                let base = format!("{}_{}", m, w);
                let mut name = base.clone();
                let mut k = 1usize;
                while seen.contains(&name) {
                    k += 1;
                    name = format!("{}_dup{}", base, k);
                }
                seen.insert(name.clone());
                out.push(name);
            }
        }
    }
    out
}

// ============================================================================
// f64 逐笔读取
// ============================================================================

/// 一条逐笔成交（只用得上这 4 列）
#[derive(Clone, Copy, Default)]
struct TradeRow {
    time_us: i64,
    price: f64,
    volume: f64,
    flag: f64,
}

/// transaction CSV 的列：symbol,datetime,date,time,exchtime,localtime,index,price,volume,turnover,flag,...
const TRD_NCOL: usize = 15;
const TRD_COL_EXCHTIME: usize = 4;
const TRD_COL_PRICE: usize = 7;
const TRD_COL_VOLUME: usize = 8;
const TRD_COL_FLAG: usize = 10;

/// pandas 读空单元格给 NaN
#[inline]
fn parse_f64_at(b: &[u8]) -> f64 {
    if b.is_empty() {
        return f64::NAN;
    }
    std::str::from_utf8(b)
        .ok()
        .and_then(|s| s.trim().parse::<f64>().ok())
        .unwrap_or(f64::NAN)
}

#[inline]
fn parse_i64_at(b: &[u8]) -> i64 {
    let mut i = 0usize;
    while i < b.len() && (b[i] == b' ' || b[i] == b'+') {
        i += 1;
    }
    let mut neg = false;
    if i < b.len() && b[i] == b'-' {
        neg = true;
        i += 1;
    }
    let mut v: i64 = 0;
    while i < b.len() && b[i].is_ascii_digit() {
        v = v * 10 + (b[i] - b'0') as i64;
        i += 1;
    }
    if neg {
        -v
    } else {
        v
    }
}

/// 读一天的成交明细（f64）。表头之后每行都读，和 pandas 读整张表一致。
fn read_trade_f64(code: &str, date: i64) -> Option<Vec<TradeRow>> {
    let filename = format!("{}_{}_transaction.csv", code, date);
    let mut path = None;
    for root in crate::data_paths::level2_roots() {
        let p = root.join(date.to_string()).join("transaction").join(&filename);
        if p.exists() {
            path = Some(p);
            break;
        }
    }
    let path = path?;
    let mut content = String::new();
    File::open(&path).ok()?.read_to_string(&mut content).ok()?;

    let mut out: Vec<TradeRow> = Vec::with_capacity(content.len() / 120 + 1);
    let mut lines = content.lines();
    lines.next();
    for line in lines {
        if line.is_empty() {
            continue;
        }
        let bytes = line.as_bytes();
        let mut f: [&[u8]; TRD_NCOL] = [&[][..]; TRD_NCOL];
        let mut start = 0usize;
        let mut col = 0usize;
        for (i, &b) in bytes.iter().enumerate() {
            if b == b',' {
                if col < TRD_NCOL {
                    f[col] = &bytes[start..i];
                }
                col += 1;
                start = i + 1;
            }
        }
        if col < TRD_NCOL {
            f[col] = &bytes[start..];
        }
        out.push(TradeRow {
            // deal_data：exchtime（微秒）+ 8 小时
            time_us: parse_i64_at(f[TRD_COL_EXCHTIME]) + 8 * 3600 * SEC_US,
            price: parse_f64_at(f[TRD_COL_PRICE]),
            volume: parse_f64_at(f[TRD_COL_VOLUME]),
            flag: parse_f64_at(f[TRD_COL_FLAG]),
        });
    }
    Some(out)
}

// ============================================================================
// numpy 成对求和 / pandas 的 sum、mean
// ============================================================================

const PW_BLOCKSIZE: usize = 128;

/// numpy 的浮点成对求和（见 hm051.rs 的同一份实现与说明）。
fn pair_sum(a: &[f64]) -> f64 {
    let n = a.len();
    if n < 8 {
        let mut res = 0.0f64;
        for v in &a[..n] {
            res += *v;
        }
        return res;
    }
    if n <= PW_BLOCKSIZE {
        let mut r = [0.0f64; 8];
        r.copy_from_slice(&a[..8]);
        let mut i = 8usize;
        while i < n - (n % 8) {
            for k in 0..8 {
                r[k] += a[i + k];
            }
            i += 8;
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        while i < n {
            res += a[i];
            i += 1;
        }
        return res;
    }
    let mut n2 = n / 2;
    n2 -= n2 % 8;
    pair_sum(&a[..n2]) + pair_sum(&a[n2..])
}

/// pandas Series.sum()：NaN 填 0、按原位参与成对求和；空选择给 0.0。
fn nansum(v: &[f64]) -> f64 {
    let mut buf = vec![0.0f64; v.len()];
    for (i, &x) in v.iter().enumerate() {
        buf[i] = if x.is_nan() { 0.0 } else { x };
    }
    pair_sum(&buf)
}

/// pandas Series.mean()：NaN 填 0 后成对求和再除以非 NaN 个数；空选择给 NaN。
fn nanmean(v: &[f64]) -> f64 {
    let cnt = v.iter().filter(|x| !x.is_nan()).count();
    if cnt == 0 {
        return f64::NAN;
    }
    nansum(v) / cnt as f64
}

// ============================================================================
// pd.cut(price, bins=10, labels=range(10))
// ============================================================================

/// 返回每行的箱号（0..9），价格 NaN 或越界给 -1。
/// 价格含 ±inf 时返回 None（对应 pandas 抛 ValueError）。
fn cut_codes(prices: &[f64]) -> Option<Vec<i32>> {
    let mut mn = f64::INFINITY;
    let mut mx = f64::NEG_INFINITY;
    for &p in prices {
        if p.is_nan() {
            continue;
        }
        if p < mn {
            mn = p;
        }
        if p > mx {
            mx = p;
        }
    }
    if !mn.is_finite() || !mx.is_finite() {
        return None;
    }
    let mut bins = [0.0f64; 11];
    if mn == mx {
        mn -= if mn != 0.0 { 0.001 * mn.abs() } else { 0.001 };
        mx += if mx != 0.0 { 0.001 * mx.abs() } else { 0.001 };
        let step = (mx - mn) / 10.0;
        for i in 0..10 {
            bins[i] = (i as f64) * step + mn;
        }
        bins[10] = mx;
    } else {
        let step = (mx - mn) / 10.0;
        for i in 0..10 {
            bins[i] = (i as f64) * step + mn;
        }
        bins[10] = mx;
        bins[0] -= (mx - mn) * 0.001;
    }
    let mut out = Vec::with_capacity(prices.len());
    for &p in prices {
        if p.is_nan() {
            out.push(-1);
            continue;
        }
        // searchsorted(bins, p, side="left") = 严格小于 p 的边数
        let mut i = 0usize;
        while i < 11 && bins[i] < p {
            i += 1;
        }
        out.push(if i == 0 || i == 11 { -1 } else { (i - 1) as i32 });
    }
    Some(out)
}

// ============================================================================
// 窗口内的一个 cal_* 组（27 个值）
// ============================================================================

/// 窗口内的一行：相对秒数、箱号（-1 = NaN）、成交量、flag
#[derive(Clone, Copy)]
struct WRow {
    rel: f64,
    code: i32,
    volume: f64,
    flag: f64,
}

fn gather_volume(w: &[WRow], idx: &[usize]) -> Vec<f64> {
    idx.iter().map(|&i| w[i].volume).collect()
}

fn gather_rel(w: &[WRow], idx: &[usize]) -> Vec<f64> {
    idx.iter().map(|&i| w[i].rel).collect()
}

/// cal_high：以「出现过的最大箱号」为 high，次大箱号为 subhigh。
/// `flag_in` 是这一组用的内层 flag（sing1 的 cal_high 是 83，sing2 的是 66）。
fn cal_high(w: &[WRow], flag_in: f64) -> [f64; N_CAL] {
    let nan27 = [f64::NAN; N_CAL];
    let high = match w.iter().filter(|r| r.code >= 0).map(|r| r.code).max() {
        Some(v) => v,
        None => return nan27,
    };
    let idx_high: Vec<usize> = (0..w.len()).filter(|&i| w[i].code == high).collect();
    if idx_high.is_empty() {
        return nan27;
    }
    let volume_high_sum = nansum(&gather_volume(w, &idx_high));
    let volume_high_mean = nanmean(&gather_volume(w, &idx_high));
    let time_high_mean = nanmean(&gather_rel(w, &idx_high));

    let idx_hf: Vec<usize> = idx_high
        .iter()
        .cloned()
        .filter(|&i| w[i].flag == flag_in)
        .collect();
    let volume_hf_sum = nansum(&gather_volume(w, &idx_hf));
    let volume_hf_mean = nanmean(&gather_volume(w, &idx_hf));
    let time_hf_mean = nanmean(&gather_rel(w, &idx_hf));

    let high_time = w[idx_high[0]].rel;
    let subhigh = w
        .iter()
        .filter(|r| r.code >= 0 && r.code < high)
        .map(|r| r.code)
        .max();
    let subhigh = match subhigh {
        Some(v) => v,
        None => return nan27,
    };
    let idx_sh: Vec<usize> = (0..w.len()).filter(|&i| w[i].code == subhigh).collect();
    let idx_real: Vec<usize> = (0..w.len()).filter(|&i| w[i].rel >= high_time).collect();
    if idx_sh.is_empty() || idx_real.is_empty() {
        return nan27;
    }
    let volume_sh_sum = nansum(&gather_volume(w, &idx_sh));
    let volume_sh_mean = nanmean(&gather_volume(w, &idx_sh));
    let time_sh_mean = nanmean(&gather_rel(w, &idx_sh));

    let idx_shf: Vec<usize> = idx_sh
        .iter()
        .cloned()
        .filter(|&i| w[i].flag == flag_in)
        .collect();
    let volume_shf_sum = nansum(&gather_volume(w, &idx_shf));
    let volume_shf_mean = nanmean(&gather_volume(w, &idx_shf));
    let time_shf_mean = nanmean(&gather_rel(w, &idx_shf));

    // df_subhigh_real：**所有** rel >= high_time 的行
    let volume_real_sum = nansum(&gather_volume(w, &idx_real));
    let volume_real_mean = nanmean(&gather_volume(w, &idx_real));
    let time_real_mean = nanmean(&gather_rel(w, &idx_real));

    let idx_realf: Vec<usize> = idx_real
        .iter()
        .cloned()
        .filter(|&i| w[i].flag == flag_in)
        .collect();
    let volume_realf_sum = nansum(&gather_volume(w, &idx_realf));
    let volume_realf_mean = nanmean(&gather_volume(w, &idx_realf));
    let time_realf_mean = nanmean(&gather_rel(w, &idx_realf));

    [
        volume_sh_sum / volume_high_sum,
        volume_sh_mean / volume_high_mean,
        time_sh_mean - time_high_mean,
        volume_shf_sum / volume_hf_sum,
        volume_shf_mean / volume_hf_mean,
        time_shf_mean - time_hf_mean,
        volume_real_sum / volume_high_sum,
        volume_real_mean / volume_high_mean,
        time_real_mean - time_high_mean,
        volume_realf_sum / volume_hf_sum,
        volume_realf_mean / volume_hf_mean,
        time_realf_mean - time_hf_mean,
        volume_hf_sum / volume_high_sum,
        // 旧代码这里被覆盖成「次高价箱」的均值，返回列表里这个名字出现两次，两处都是这个值
        volume_shf_mean / volume_sh_mean,
        time_hf_mean - time_high_mean,
        volume_shf_sum / volume_sh_sum,
        volume_shf_mean / volume_sh_mean,
        time_shf_mean - time_sh_mean,
        volume_realf_sum / volume_real_sum,
        volume_realf_mean / volume_real_mean,
        time_realf_mean - time_real_mean,
        volume_real_sum / volume_sh_sum,
        volume_real_mean / volume_sh_mean,
        time_real_mean - time_sh_mean,
        volume_realf_sum / volume_shf_sum,
        volume_realf_mean / volume_shf_mean,
        time_realf_mean - time_shf_mean,
    ]
}

/// cal_low：以「出现过的最小箱号」为 low，次小箱号为 sublow。
fn cal_low(w: &[WRow], flag_in: f64) -> [f64; N_CAL] {
    let nan27 = [f64::NAN; N_CAL];
    let low = match w.iter().filter(|r| r.code >= 0).map(|r| r.code).min() {
        Some(v) => v,
        None => return nan27,
    };
    let idx_low: Vec<usize> = (0..w.len()).filter(|&i| w[i].code == low).collect();
    if idx_low.is_empty() {
        return nan27;
    }
    let volume_low_sum = nansum(&gather_volume(w, &idx_low));
    let volume_low_mean = nanmean(&gather_volume(w, &idx_low));
    let time_low_mean = nanmean(&gather_rel(w, &idx_low));

    let idx_lf: Vec<usize> = idx_low
        .iter()
        .cloned()
        .filter(|&i| w[i].flag == flag_in)
        .collect();
    let volume_lf_sum = nansum(&gather_volume(w, &idx_lf));
    let volume_lf_mean = nanmean(&gather_volume(w, &idx_lf));
    let time_lf_mean = nanmean(&gather_rel(w, &idx_lf));

    let low_time = w[idx_low[0]].rel;
    let sublow = w
        .iter()
        .filter(|r| r.code >= 0 && r.code > low)
        .map(|r| r.code)
        .min();
    let sublow = match sublow {
        Some(v) => v,
        None => return nan27,
    };
    let idx_sl: Vec<usize> = (0..w.len()).filter(|&i| w[i].code == sublow).collect();
    let idx_real: Vec<usize> = (0..w.len()).filter(|&i| w[i].rel >= low_time).collect();
    if idx_sl.is_empty() || idx_real.is_empty() {
        return nan27;
    }
    let volume_sl_sum = nansum(&gather_volume(w, &idx_sl));
    let volume_sl_mean = nanmean(&gather_volume(w, &idx_sl));
    let time_sl_mean = nanmean(&gather_rel(w, &idx_sl));

    let idx_slf: Vec<usize> = idx_sl
        .iter()
        .cloned()
        .filter(|&i| w[i].flag == flag_in)
        .collect();
    let volume_slf_sum = nansum(&gather_volume(w, &idx_slf));
    let volume_slf_mean = nanmean(&gather_volume(w, &idx_slf));
    let time_slf_mean = nanmean(&gather_rel(w, &idx_slf));

    let volume_real_sum = nansum(&gather_volume(w, &idx_real));
    let volume_real_mean = nanmean(&gather_volume(w, &idx_real));
    let time_real_mean = nanmean(&gather_rel(w, &idx_real));

    let idx_realf: Vec<usize> = idx_real
        .iter()
        .cloned()
        .filter(|&i| w[i].flag == flag_in)
        .collect();
    let volume_realf_sum = nansum(&gather_volume(w, &idx_realf));
    let volume_realf_mean = nanmean(&gather_volume(w, &idx_realf));
    let time_realf_mean = nanmean(&gather_rel(w, &idx_realf));

    [
        volume_sl_sum / volume_low_sum,
        volume_sl_mean / volume_low_mean,
        time_sl_mean - time_low_mean,
        volume_slf_sum / volume_lf_sum,
        volume_slf_mean / volume_lf_mean,
        time_slf_mean - time_lf_mean,
        volume_real_sum / volume_low_sum,
        volume_real_mean / volume_low_mean,
        time_real_mean - time_low_mean,
        volume_realf_sum / volume_lf_sum,
        volume_realf_mean / volume_lf_mean,
        time_realf_mean - time_lf_mean,
        volume_lf_sum / volume_low_sum,
        volume_slf_mean / volume_sl_mean,
        time_lf_mean - time_low_mean,
        volume_slf_sum / volume_sl_sum,
        volume_slf_mean / volume_sl_mean,
        time_slf_mean - time_sl_mean,
        volume_realf_sum / volume_real_sum,
        volume_realf_mean / volume_real_mean,
        time_realf_mean - time_real_mean,
        volume_real_sum / volume_sl_sum,
        volume_real_mean / volume_sl_mean,
        time_real_mean - time_sl_mean,
        volume_realf_sum / volume_slf_sum,
        volume_realf_mean / volume_slf_mean,
        time_realf_mean - time_slf_mean,
    ]
}

/// 把 ±inf 抹成 NaN（旧代码的 arr[np.isinf(arr)] = np.nan）
fn clean_inf(v: &mut [f64]) {
    for x in v.iter_mut() {
        if x.is_infinite() {
            *x = f64::NAN;
        }
    }
}

// ============================================================================
// 逐股入口
// ============================================================================

/// 窗口行的筛选与下午平移。
///
/// 关键在 allday 这张混合表：df = concat([morning, afternoon 减 90 分钟])，
/// 再在这张表上按**平移后**的钟点切窗口，所以：
///   · morning   [09:30,11:30] 平移后 → 上午的行 + 恰好在 13:00:00.000000 的下午行
///                （它平移后正好是 11:30:00.000000，落在右端点上）
///   · afternoon [11:30,13:27] 平移后 → 下午的行 + 恰好在 11:30:00.000000 的上午行
///   · allday    = 上午 [09:30,11:30] + 下午 [13:00,14:57]
///   · open30    [09:30,10:00]（平移后）= 上午 [09:30,10:00]
///   · close30   [13:00,13:27]（平移后）= 真实 [14:30,14:57]
/// 000001@20250303 的 41 处不一致就是漏了这两个整点端点，补上后逐值对上。
fn in_window(win: usize, day_us: i64) -> bool {
    let s = |h: i64, m: i64| (h * 3600 + m * 60) * SEC_US;
    match win {
        0 => (s(9, 30)..=s(11, 30)).contains(&day_us) || (s(13, 0)..=s(14, 57)).contains(&day_us),
        1 => (s(9, 30)..=s(10, 0)).contains(&day_us),
        // 上午窗口还要收下「平移后正好 11:30」的那一笔下午开盘成交
        2 => (s(9, 30)..=s(11, 30)).contains(&day_us) || day_us == s(13, 0),
        // 下午窗口还要收下「11:30:00.000000」整点的上午收市成交
        3 => (s(13, 0)..=s(14, 57)).contains(&day_us) || day_us == s(11, 30),
        _ => (s(14, 30)..=s(14, 57)).contains(&day_us),
    }
}

/// 逐股入口：算不出来返回 None（上游补 NaN，保证因子个数恒定）。
pub fn compute_stock(code: &str, date: i64) -> Option<Vec<f32>> {
    let trades = read_trade_f64(code, date)?;
    let rows: Vec<&TradeRow> = trades.iter().filter(|r| r.flag != 32.0).collect();

    let mut out: Vec<f32> = Vec::with_capacity(N_OUT);
    for win in 0..N_WIN {
        // 收集窗口内的行（保持 CSV 行序），并做下午平移
        let mut raw: Vec<(i64, f64, f64, f64)> = Vec::new();
        for r in rows.iter() {
            let day_us = r.time_us.rem_euclid(86_400_000_000);
            if !in_window(win, day_us) {
                continue;
            }
            let shifted = if day_us >= 13 * 3600 * SEC_US {
                r.time_us - AFTERNOON_SHIFT
            } else {
                r.time_us
            };
            raw.push((shifted, r.price, r.volume, r.flag));
        }
        let t0 = raw.iter().map(|x| x.0).min().unwrap_or(0);
        let w: Vec<WRow> = raw
            .iter()
            .map(|&(t, _p, v, f)| WRow {
                rel: (t - t0) as f64 / 1e6,
                code: 0,
                volume: v,
                flag: f,
            })
            .collect();
        // 分箱要的是价格，单独传（WRow 里不存价格）
        let prices: Vec<f64> = raw.iter().map(|x| x.1).collect();
        let vals = match window_values(&w, &prices) {
            Some(v) => v,
            None => return None,
        };
        out.extend(vals.iter().map(|&v| v as f32));
    }
    debug_assert_eq!(out.len(), N_OUT);
    Some(out)
}

/// 一个窗口的 108 个值：sing1(cal_high 83 + cal_low 66) + sing2(cal_high 66 + cal_low 83)。
/// 行数 <= 10 时整窗 108 个 NaN。pd.cut 失败（价格含 inf）时返回 None。
fn window_values(w: &[WRow], prices: &[f64]) -> Option<[f64; N_PER_WIN]> {
    let mut out = [f64::NAN; N_PER_WIN];
    if w.len() <= 10 {
        return Some(out);
    }
    let codes = cut_codes(prices)?;
    let wr: Vec<WRow> = w
        .iter()
        .zip(codes.iter())
        .map(|(r, &c)| WRow {
            rel: r.rel,
            code: c,
            volume: r.volume,
            flag: r.flag,
        })
        .collect();
    let mut s1 = [0.0f64; 54];
    s1[..N_CAL].copy_from_slice(&cal_high(&wr, 83.0));
    s1[N_CAL..].copy_from_slice(&cal_low(&wr, 66.0));
    clean_inf(&mut s1);
    let mut s2 = [0.0f64; 54];
    s2[..N_CAL].copy_from_slice(&cal_high(&wr, 66.0));
    s2[N_CAL..].copy_from_slice(&cal_low(&wr, 83.0));
    clean_inf(&mut s2);
    out[..54].copy_from_slice(&s1);
    out[54..].copy_from_slice(&s2);
    Some(out)
}

/// hm40 全是逐股因子，没有横截面版本。
pub fn compute_cross(_date: i64) -> Option<(Vec<String>, Vec<f32>)> {
    None
}
