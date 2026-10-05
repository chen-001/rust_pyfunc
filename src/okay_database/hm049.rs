//! hm49「龙生九子」（旧框架 design_whatever_out.TickDataBase）的 Rust 复现。
//!
//! 旧文件：/home/chenzongwei/pythoncode/成果hm49_龙生九子/hm49_okay.py
//!   run_base 的 Go.get_factor，参数 get_many=0 / read_full=0 / change_time=1，
//!   with_will=1 / with_will_back_days=1（默认）—— 因子 = 今天盘口帧与**上一个交易日**
//!   盘口帧的滚动相关、以及由这些相关挑出来的三段窗口的统计量。
//!   旧框架的 will 机制在 design_whatever_out.cal_factor_one_stock：
//!   先把回看日跑一遍（只写 self.will、返回 None），再把 will 传给目标日。
//!   Rust 侧 compute_stock(code, date) 自己回看上一个交易日、两天的盘口各读一遍即可。
//!
//! ════════════════════════════════════════════════════════════════════
//! 旧代码逐步对照（写代码前先抄在这里，方便复核）
//! ════════════════════════════════════════════════════════════════════
//! 1) df = read_market_data_single()
//!      · 索引 = pd.to_datetime(data["datetime"])，change_time=1 只把 exchtime 列转成
//!        Timestamp，索引不变。
//! 2) df = dwo.adjust_afternoon(df.between_time("09:30:00","14:57:00"))
//!      · 先按 datetime 索引切 [09:30:00, 14:57:00]（两端都含、微秒精度）
//!      · adjust_afternoon 走 'exchtime' in df.columns 那条分支：
//!          reset_index() 后 exchtime 回到列上 → set_index('exchtime') →
//!          df1 = [09:30:00, 11:30:00]、df2 = [13:00:00, 14:57:00] 整体 -90 分钟 →
//!          concat([df1, df2]).reset_index()
//!        实测 datetime 列与 exchtime+8h 逐行完全相同（差值 0 ns），exchtime→Timestamp
//!        也精确到微秒，所以这里统一用「当日微秒数」做时间轴，下午减 90 分钟。
//! 3) df.volume = df.volume.diff()      # 成交量列是累计量，差分后是每个快照的增量
//!    df = df.set_index("exchtime")     # 索引变成「前移后的当日时刻」，范围 09:30~13:27
//!    df_morning   = df.between_time("09:30:00","11:30:00")   # 前移后 = 真实上午
//!    df_afternoon = df.between_time("13:00:00","14:57:00")   # 前移后 = 真实 14:30~14:57
//! 4) sing(df_w, is_all=0) 对 df_w = df_morning / df_afternoon / df（全天）各算一遍：
//!      will = 上一日整天帧，按 [df_w 时刻的 min, max] 过滤
//!      minute_price_yesterday = will.last_prc.resample("1min").last().dropna()
//!      minute_price_today     = df_w.last_prc.resample("1min").last().dropna()
//!      res = one(df_w, minute_price_yesterday) + one(df_w, minute_price_today)
//! 5) one(df_w, minute_price)：
//!      num = len(minute_price)
//!      sim = rolling_correlation_vectorized(df_w.last_prc, minute_price)
//!            （旧代码里那个 numpy 版滑动相关，逐行照抄）
//!      df = df_w.assign(sim=sim).reset_index(drop=True)
//!      df_max  = 第 (sim==sim.max()) 个位置起往前 num 行   → get_stats2(df_max, res)
//!      df_min  = 第 (sim==sim.min()) 个位置起往前 num 行   → get_stats2(df_min, res)
//!      df_zero = 第 (|sim|==|sim|.min()) 个位置起往前 num 行 → get_stats2(df_zero, res)
//!      res = get_stats(df)（在整段上算一次），三个子窗共用
//!      每个 get_stats2 出 30 个数，一个 one() 出 90 个数
//! 6) 于是 540 个输出 = 3 个窗口 × 2 个 minute_price × 90，
//!    名字顺序：morning(180) + afternoon(180) + all(180)，
//!    每 180 里再分 corrmax_yesterday / corrmin_yesterday / corrzero_yesterday /
//!    corrmax_today / corrmin_today / corrzero_today 六组 × 30。
//!    ！！is_all=1 那条带 dtw 的分支是死代码：三处调用都传 is_all=0，
//!      所以 rp.rolling_dtw_distance 根本不会被调用，本文件不需要 DTW。
//! names_save 里的 80 个名字全部落在 namesp 的前 4 组和偏度类里，不含
//! freq_price / rare_price / freq_rare_price / freq_rare_price_rate（见下「已知差异」）。
//!
//! ════════════════════════════════════════════════════════════════════
//! 精度与已知不可复现
//! ════════════════════════════════════════════════════════════════════
//! · 一律 f64 解析 CSV（不用 fast_csv_reader 的 f32）：窗口里成交量是累计量的差分，
//!   f32 下 in/out 相减的误差会直接进 cpv/crv/volume_volatility。
//! · pandas 的 sum/mean/std 走 numpy 的成对求和（blocksize 128、8 路展开），
//!   np.cumsum 是顺序累加，groupby().sum() 是 Kahan 补偿求和 —— 三者在本文件里分别
//!   用 pw_sum / 顺序累加 / kahan_sum 照抄，不能混用。
//! · pandas Series.corr → nanops.nancorr → np.corrcoef，即
//!   (c01/sqrt(c00))/sqrt(c11)。三个点积在 numpy 那边由 BLAS dgemm 累加，这里用 pw_sum
//!   代替，只差最后几位（相对 1e-15 量级）。
//! · np.log 走 AVX-512 的 SVML，与 Rust 的 f64::ln 有约 0.01%~0.2% 的输入差 1 ulp。
//!   实测对最终值没有可见影响（见对账结论）。
//! · **已知差异**：x.groupby("last_prc").volume.sum().sort_values() 里，
//!   numpy 2.x 对 float64 的 argsort(kind='quicksort') 走 x86-simd-sort 的 AVX-512
//!   实现，并列元素的先后不是稳定排序（本机 pandas 2.3.3 + numpy 2.3.5 实测：
//!   用 longdouble 复刻标量 quicksort 逐位一致，float64 则不一致，说明确实走了 SIMD）。
//!   这里用稳定排序代替，规则是「组内成交量和的极值并列时取价格最大的/最小的」。
//!   受影响的是 30 个统计量里的第 15~18 个（freq_price / rare_price /
//!   freq_rare_price / freq_rare_price_rate），共 72/540 列；实测 288 个窗口里
//!   极小值并列 25 次、其中 4 次与 numpy 的选择不同。这 4 个量**不在** names_save /
//!   names_in_db / gap5 / gap1 任何一张名单里，不会进因子库。

use std::fs::File;
use std::io::Read;

/// 盘口 CSV 的列索引（0 基）
const MKT_COL_EXCHTIME: usize = 4;
const MKT_COL_LAST_PRC: usize = 6;
const MKT_COL_VOLUME: usize = 14;
const MKT_COL_W_ASK: usize = 17;
const MKT_COL_W_BID: usize = 18;
const MKT_ASK_PRC_BASE: usize = 21; // ask_prc1=21, ask_vol1=22, bid_prc1=23, bid_vol1=24, ...
const MKT_NCOL: usize = 61;

const US: i64 = 1_000_000;
const DAY_US: i64 = 86_400 * US;
const MIN_US: i64 = 60 * US;
const SHIFT90: i64 = 90 * MIN_US;

const LD_0930: i64 = (9 * 3600 + 30 * 60) * US;
const LD_1130: i64 = (11 * 3600 + 30 * 60) * US;
const LD_1300: i64 = 13 * 3600 * US;
const LD_1457: i64 = (14 * 3600 + 57 * 60) * US;

/// 一个时段的 30 个统计量
pub const PER_GROUP: usize = 30;
const N_WINDOW: usize = 3;
const N_MP: usize = 2;
pub const N_OUT: usize = N_WINDOW * N_MP * 3 * PER_GROUP; // 3*2*3*30 = 540

const NAMESP: [&str; PER_GROUP] = [
    "ret_pri",
    "ret",
    "volatility_pri",
    "volatility",
    "volume",
    "sim",
    "cpv_pri",
    "crv_pri",
    "cpv",
    "crv",
    "volume_volatility_pri",
    "volume_volatility",
    "high_volume",
    "low_volume",
    "high_low_volume",
    "high_volume_rate_pri",
    "low_volume_rate_pri",
    "high_volume_rate",
    "low_volume_rate",
    "high_low_volume_rate_pri",
    "high_low_volume_rate",
    "ask_bid_imbalance1",
    "ask_bid_imbalance5",
    "weighted_prc_imbalance",
    "ask_bid_imbalance1_rate_pri",
    "ask_bid_imbalance1_rate",
    "ask_bid_imbalance5_rate_pri",
    "ask_bid_imbalance5_rate",
    "weighted_prc_imbalance_rate_pri",
    "weighted_prc_imbalance_rate",
];

const GROUPS: [&str; 6] = [
    "corrmax_yesterday",
    "corrmin_yesterday",
    "corrzero_yesterday",
    "corrmax_today",
    "corrmin_today",
    "corrzero_today",
];

pub fn names() -> Vec<String> {
    let mut out = Vec::with_capacity(N_OUT);
    for w in ["morning", "afternoon", "all"] {
        for g in GROUPS.iter() {
            for n in NAMESP.iter() {
                out.push(format!("{}_{}_{}", n, g, w));
            }
        }
    }
    out
}

// ============================================================================
// CSV 读取
// ============================================================================

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
    while i < b.len() && b[i] == b' ' {
        i += 1;
    }
    let mut neg = false;
    if i < b.len() && (b[i] == b'-' || b[i] == b'+') {
        neg = b[i] == b'-';
        i += 1;
    }
    let mut v: i64 = 0;
    while i < b.len() {
        let c = b[i];
        if !c.is_ascii_digit() {
            break;
        }
        v = v * 10 + (c - b'0') as i64;
        i += 1;
    }
    if neg {
        -v
    } else {
        v
    }
}

// ============================================================================
// 一帧（一天的盘口，下午已前移 90 分钟）
// ============================================================================

#[derive(Clone, Default)]
struct Frame {
    t: Vec<i64>,
    last_prc: Vec<f64>,
    vol: Vec<f64>,
    w_ask: Vec<f64>,
    w_bid: Vec<f64>,
    av: Vec<[f64; 5]>,
    bv: Vec<[f64; 5]>,
    /// 只有 one() 里算出来的那一段才带 sim；其余地方是空 vec
    sim: Vec<f64>,
}

impl Frame {
    fn len(&self) -> usize {
        self.t.len()
    }

    /// between_time(lo, hi)：两端都含、微秒精度，保持原行序（pandas 是布尔掩码）
    fn select(&self, lo: i64, hi: i64) -> Frame {
        let mut out = Frame::default();
        for i in 0..self.len() {
            if self.t[i] >= lo && self.t[i] <= hi {
                out.push_from(self, i);
            }
        }
        out
    }

    /// iloc[a:b]
    fn slice(&self, a: usize, b: usize) -> Frame {
        let mut out = Frame::default();
        for i in a..b {
            out.push_from(self, i);
        }
        out
    }

    fn push_from(&mut self, src: &Frame, i: usize) {
        self.t.push(src.t[i]);
        self.last_prc.push(src.last_prc[i]);
        self.vol.push(src.vol[i]);
        self.w_ask.push(src.w_ask[i]);
        self.w_bid.push(src.w_bid[i]);
        self.av.push(src.av[i]);
        self.bv.push(src.bv[i]);
        if !src.sim.is_empty() {
            self.sim.push(src.sim[i]);
        }
    }

    fn t_range(&self) -> Option<(i64, i64)> {
        if self.t.is_empty() {
            return None;
        }
        let mut lo = self.t[0];
        let mut hi = self.t[0];
        for &x in self.t.iter() {
            if x < lo {
                lo = x;
            }
            if x > hi {
                hi = x;
            }
        }
        Some((lo, hi))
    }
}

/// 读一天的盘口（f64），做 between_time + 下午前移 + 成交量差分。
/// 顺序严格按旧代码：先上午段（原行序）再下午段（原行序、时刻 -90 分钟）。
fn read_day(code: &str, date: i64) -> Option<Frame> {
    let filename = format!("{}_{}_market_data.csv", code, date);
    let mut path = None;
    for root in crate::data_paths::level2_roots() {
        let p = root.join(date.to_string()).join("market_data").join(&filename);
        if p.exists() {
            path = Some(p);
            break;
        }
    }
    let path = path?;

    let mut content = String::new();
    File::open(&path).ok()?.read_to_string(&mut content).ok()?;

    let mut f = Frame::default(); // 上午段
    let mut raw_vol: Vec<f64> = Vec::new();
    let mut aft = Frame::default(); // 下午段（时刻已前移）
    let mut aft_vol: Vec<f64> = Vec::new();
    let mut lines = content.lines();
    lines.next(); // 表头
    for line in lines {
        if line.is_empty() {
            continue;
        }
        let bytes = line.as_bytes();
        let mut fl: [&[u8]; MKT_NCOL] = [&[][..]; MKT_NCOL];
        let mut start = 0usize;
        let mut col = 0usize;
        for (i, &b) in bytes.iter().enumerate() {
            if b == b',' {
                if col < MKT_NCOL {
                    fl[col] = &bytes[start..i];
                }
                col += 1;
                start = i + 1;
            }
        }
        if col < MKT_NCOL {
            fl[col] = &bytes[start..];
        }
        let ex = parse_i64_at(fl[MKT_COL_EXCHTIME]);
        let t = (ex + 8 * 3600 * US).rem_euclid(DAY_US);
        // 外层 between_time("09:30:00","14:57:00")；中间的午休段两边都不收
        let in_morning = t >= LD_0930 && t <= LD_1130;
        let in_afternoon = t >= LD_1300 && t <= LD_1457;
        if !in_morning && !in_afternoon {
            continue;
        }
        let mut av = [0.0f64; 5];
        let mut bv = [0.0f64; 5];
        for k in 0..5 {
            let base = MKT_ASK_PRC_BASE + k * 4;
            av[k] = parse_f64_at(fl[base + 1]);
            bv[k] = parse_f64_at(fl[base + 3]);
        }
        let vol = parse_f64_at(fl[MKT_COL_VOLUME]);
        let slot = if in_morning { &mut f } else { &mut aft };
        slot.t.push(if in_morning { t } else { t - SHIFT90 });
        slot.last_prc.push(parse_f64_at(fl[MKT_COL_LAST_PRC]));
        slot.w_ask.push(parse_f64_at(fl[MKT_COL_W_ASK]));
        slot.w_bid.push(parse_f64_at(fl[MKT_COL_W_BID]));
        slot.av.push(av);
        slot.bv.push(bv);
        if in_morning {
            raw_vol.push(vol);
        } else {
            aft_vol.push(vol);
        }
    }
    // concat([df1, df2])
    append_frame(&mut f, &aft);
    raw_vol.extend_from_slice(&aft_vol);
    let n = f.len();
    // volume 差分（在拼接后的顺序上做）
    let mut dv = vec![f64::NAN; n];
    for i in 1..n {
        dv[i] = raw_vol[i] - raw_vol[i - 1];
    }
    f.vol = dv;
    Some(f)
}

fn append_frame(dst: &mut Frame, src: &Frame) {
    dst.t.extend_from_slice(&src.t);
    dst.last_prc.extend_from_slice(&src.last_prc);
    dst.vol.extend_from_slice(&src.vol);
    dst.w_ask.extend_from_slice(&src.w_ask);
    dst.w_bid.extend_from_slice(&src.w_bid);
    dst.av.extend_from_slice(&src.av);
    dst.bv.extend_from_slice(&src.bv);
}

// ============================================================================
// numpy / pandas 原语
// ============================================================================

const PW_BLOCKSIZE: usize = 128;

/// numpy 的成对求和（loops_utils.h 的 pairwise_sum_DOUBLE：blocksize 128、8 路展开）
fn pw_sum(a: &[f64]) -> f64 {
    let n = a.len();
    if n < 8 {
        let mut res = -0.0f64;
        for &v in a.iter() {
            res += v;
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
    pw_sum(&a[..n2]) + pw_sum(&a[n2..])
}

/// pandas Series.sum()：NaN 记 0 后做成对求和（位置不能压缩，成对求和对位置敏感）
fn nan_sum(a: &[f64]) -> f64 {
    let z: Vec<f64> = a.iter().map(|&x| if x.is_nan() { 0.0 } else { x }).collect();
    pw_sum(&z)
}

fn nan_count(a: &[f64]) -> usize {
    a.iter().filter(|x| !x.is_nan()).count()
}

/// pandas Series.mean()：sum(带 0 填充的整段) / 非空个数
fn nan_mean(a: &[f64]) -> f64 {
    let c = nan_count(a);
    if c == 0 {
        return f64::NAN;
    }
    nan_sum(a) / c as f64
}

fn nan_max(a: &[f64]) -> f64 {
    let mut m = f64::NAN;
    for &v in a.iter() {
        if !v.is_nan() && (m.is_nan() || v > m) {
            m = v;
        }
    }
    m
}

fn nan_min(a: &[f64]) -> f64 {
    let mut m = f64::NAN;
    for &v in a.iter() {
        if !v.is_nan() && (m.is_nan() || v < m) {
            m = v;
        }
    }
    m
}

/// pandas Series.std()：两遍法。avg = 成对求和(0 填充)/count，
/// 平方偏差和也是成对求和，NaN 位置置 0（长度不变，成对求和的位置不能变）。
fn nan_std(a: &[f64], ddof: usize) -> f64 {
    let k = nan_count(a);
    if k == 0 {
        return f64::NAN;
    }
    if k <= ddof {
        return f64::NAN;
    }
    let avg = nan_sum(a) / k as f64;
    let sq: Vec<f64> = a
        .iter()
        .map(|&x| if x.is_nan() { 0.0 } else { (avg - x) * (avg - x) })
        .collect();
    (pw_sum(&sq) / (k as f64 - ddof as f64)).sqrt()
}

/// pandas Series.pct_change()：先 pad（前向填充）再 x/x.shift(1)-1
fn pct_change(a: &[f64]) -> Vec<f64> {
    let n = a.len();
    let mut x = a.to_vec();
    let mut last = f64::NAN;
    for v in x.iter_mut() {
        if v.is_nan() {
            *v = last;
        } else {
            last = *v;
        }
    }
    let mut out = vec![f64::NAN; n];
    for i in 1..n {
        out[i] = x[i] / x[i - 1] - 1.0;
    }
    out
}

/// np.log(x.volume).replace([inf,-inf], nan)
fn log_replace_inf(a: &[f64]) -> Vec<f64> {
    a.iter()
        .map(|&x| {
            let l = x.ln();
            if l.is_infinite() {
                f64::NAN
            } else {
                l
            }
        })
        .collect()
}

/// pandas Series.corr（pearson）：align(inner) → nancorr → np.corrcoef[0,1]
fn corr(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len().min(y.len());
    let mut xs: Vec<f64> = Vec::with_capacity(n);
    let mut ys: Vec<f64> = Vec::with_capacity(n);
    for i in 0..n {
        if !x[i].is_nan() && !y[i].is_nan() {
            xs.push(x[i]);
            ys.push(y[i]);
        }
    }
    let k = xs.len();
    if k < 2 {
        return f64::NAN;
    }
    let mx = pw_sum(&xs) / k as f64;
    let my = pw_sum(&ys) / k as f64;
    let mut xx: Vec<f64> = Vec::with_capacity(k);
    let mut yy: Vec<f64> = Vec::with_capacity(k);
    let mut xy: Vec<f64> = Vec::with_capacity(k);
    for i in 0..k {
        let a = xs[i] - mx;
        let b = ys[i] - my;
        xx.push(a * a);
        yy.push(b * b);
        xy.push(a * b);
    }
    let s = 1.0f64 / (k as f64 - 1.0);
    let c00 = pw_sum(&xx) * s;
    let c11 = pw_sum(&yy) * s;
    let c01 = pw_sum(&xy) * s;
    let r = (c01 / c00.sqrt()) / c11.sqrt();
    if r.is_nan() {
        r
    } else {
        r.clamp(-1.0, 1.0)
    }
}

/// pandas resample("1min").last().dropna() → 每分钟最后一个非空 last_prc
/// （origin='start_day'、closed='left'、label='left'；空桶是 NaN，dropna 丢掉）
fn resample_1min_last(t: &[i64], v: &[f64]) -> Vec<f64> {
    if t.is_empty() {
        return Vec::new();
    }
    let first = *t.iter().min().unwrap();
    let last = *t.iter().max().unwrap();
    let lo = first.div_euclid(MIN_US) * MIN_US;
    let loff = last.rem_euclid(MIN_US);
    let hi = if loff > 0 {
        last + (MIN_US - loff)
    } else {
        last + MIN_US
    };
    let nbin = ((hi - lo) / MIN_US) as usize;
    let mut vals = vec![f64::NAN; nbin];
    for i in 0..t.len() {
        if v[i].is_nan() {
            continue; // group_last 只让非空值覆盖
        }
        let idx = ((t[i] - lo) / MIN_US) as usize;
        if idx < nbin {
            vals[idx] = v[i];
        }
    }
    vals.into_iter().filter(|x| !x.is_nan()).collect()
}

/// 旧代码 rolling_correlation_vectorized 的逐行复刻。
/// long 比 short 短时（result_len <= 0）旧代码会抛异常 → 整股整日 NaN，这里返回 None。
fn rolling_corr_vec(long: &[f64], short: &[f64]) -> Option<Vec<f64>> {
    let sl = short.len();
    let ll = long.len();
    if sl == 0 || ll + 1 < sl {
        return None;
    }
    let rl = ll - sl + 1;
    let sm = pw_sum(short) / sl as f64;
    let sq: Vec<f64> = short.iter().map(|&x| (x - sm) * (x - sm)).collect();
    let sv = (pw_sum(&sq) / sl as f64).sqrt();
    let sn: Vec<f64> = short.iter().map(|&x| (x - sm) / sv).collect();

    let mut cs = Vec::with_capacity(ll + 1);
    let mut csq = Vec::with_capacity(ll + 1);
    cs.push(0.0);
    csq.push(0.0);
    let mut acc = 0.0f64;
    let mut acc2 = 0.0f64;
    for &x in long.iter() {
        acc += x;
        acc2 += x * x;
        cs.push(acc);
        csq.push(acc2);
    }
    let mut means = Vec::with_capacity(rl);
    let mut stds = Vec::with_capacity(rl);
    for i in 0..rl {
        let m = (cs[sl + i] - cs[i]) / sl as f64;
        let var = (csq[sl + i] - csq[i]) / sl as f64 - m * m;
        means.push(m);
        stds.push(var.sqrt());
    }
    let mut out = vec![f64::NAN; ll];
    let mut buf = vec![0.0f64; sl];
    for i in 0..rl {
        for j in 0..sl {
            buf[j] = (long[i + j] - means[i]) / stds[i] * sn[j];
        }
        out[sl - 1 + i] = pw_sum(&buf) / sl as f64;
    }
    Some(out)
}

/// pandas groupby(key).sum()：NaN 键丢掉，组内按行序做 Kahan 补偿求和。
/// 返回「组键升序」的 (键, 和)。
fn group_sum_kahan(keys: &[f64], vals: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let mut ks: Vec<f64> = Vec::new();
    let mut seen: std::collections::HashSet<u64> = std::collections::HashSet::new();
    for &k in keys.iter() {
        if k.is_nan() {
            continue;
        }
        if seen.insert(k.to_bits()) {
            ks.push(k);
        }
    }
    ks.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mut pos: std::collections::HashMap<u64, usize> = std::collections::HashMap::new();
    for (p, &k) in ks.iter().enumerate() {
        pos.insert(k.to_bits(), p);
    }
    let mut sums = vec![0.0f64; ks.len()];
    let mut comp = vec![0.0f64; ks.len()];
    for i in 0..keys.len() {
        let k = keys[i];
        if k.is_nan() {
            continue;
        }
        let v = vals[i];
        if v.is_nan() {
            continue; // group_sum 只统计非空值，min_count=0 时全空组的和是 0
        }
        let p = pos[&k.to_bits()];
        let y = v - comp[p];
        let t = sums[p] + y;
        let mut c = t - sums[p] - y;
        if c.is_nan() {
            c = 0.0; // pandas GH#53606：val 是 ±inf 时补偿项会变 NaN，置 0
        }
        comp[p] = c;
        sums[p] = t;
    }
    (ks, sums)
}

/// 上升序键数组按值稳定排序后取 index[-1]：并列的最大值里取**键最大**的那个。
fn last_argmax(v: &[f64]) -> usize {
    let mut best = 0usize;
    for i in 1..v.len() {
        if v[i] >= v[best] {
            best = i;
        }
    }
    best
}

/// 同上的 index[0]：并列的最小值里取**键最小**的那个。
fn first_argmin_stable(v: &[f64]) -> usize {
    let mut best = 0usize;
    for i in 1..v.len() {
        if v[i] < v[best] {
            best = i;
        }
    }
    best
}

// ============================================================================
// 30 个统计量
// ============================================================================

/// get_stats 里被 get_stats2 用到的部分（其余量算了也没人用）
struct Res {
    ret: f64,
    volatility: f64,
    volume: f64,
    cpv: f64,
    crv: f64,
    volume_volatility: f64,
    high_volume: f64,
    low_volume: f64,
    high_low_volume: f64,
    high_volume_rate: f64,
    low_volume_rate: f64,
    high_low_volume_rate: f64,
    ab1: f64,
    ab5: f64,
    wpi: f64,
    ab1r: f64,
    ab5r: f64,
    wpir: f64,
}

#[inline]
fn a1_of(f: &Frame, i: usize) -> f64 {
    f.av[i][0] - f.bv[i][0]
}

#[inline]
fn a5_of(f: &Frame, i: usize) -> f64 {
    let a = f.av[i][0] + f.av[i][1] + f.av[i][2] + f.av[i][3] + f.av[i][4];
    let b = f.bv[i][0] + f.bv[i][1] + f.bv[i][2] + f.bv[i][3] + f.bv[i][4];
    a - b
}

#[inline]
fn a1r_of(f: &Frame, i: usize) -> f64 {
    (f.av[i][0] - f.bv[i][0]) / (f.av[i][0] + f.bv[i][0])
}

#[inline]
fn a5r_of(f: &Frame, i: usize) -> f64 {
    let a = f.av[i][0] + f.av[i][1] + f.av[i][2] + f.av[i][3] + f.av[i][4];
    let b = f.bv[i][0] + f.bv[i][1] + f.bv[i][2] + f.bv[i][3] + f.bv[i][4];
    (a - b) / (a + b)
}

#[inline]
fn wpr_of(f: &Frame, i: usize) -> f64 {
    (f.w_ask[i] - f.w_bid[i]) / (f.w_ask[i] + f.w_bid[i])
}

fn get_stats(f: &Frame) -> Res {
    let n = f.len();
    let lg = log_replace_inf(&f.vol);
    let pc = pct_change(&f.last_prc);
    let ret = f.last_prc[n - 1] / f.last_prc[0] - 1.0;
    let volatility = nan_std(&pc, 1);
    let volume = nan_sum(&f.vol);
    let cpv = corr(&f.last_prc, &lg);
    let crv = corr(&pc, &lg);
    let volume_volatility = nan_std(&lg, 1);
    let hp = nan_max(&f.last_prc);
    let lp = nan_min(&f.last_prc);
    let sel = |want: f64| -> Vec<f64> {
        (0..n)
            .filter(|&i| f.last_prc[i] == want)
            .map(|i| f.vol[i])
            .collect()
    };
    let high_volume = nan_sum(&sel(hp));
    let low_volume = nan_sum(&sel(lp));
    let high_low_volume = high_volume - low_volume;
    let high_volume_rate = high_volume / volume;
    let low_volume_rate = low_volume / volume;
    let high_low_volume_rate = high_volume_rate - low_volume_rate;
    let a1: Vec<f64> = (0..n).map(|i| a1_of(f, i)).collect();
    let a5: Vec<f64> = (0..n).map(|i| a5_of(f, i)).collect();
    let wp: Vec<f64> = (0..n).map(|i| f.w_ask[i] - f.w_bid[i]).collect();
    let a1r: Vec<f64> = (0..n).map(|i| a1r_of(f, i)).collect();
    let a5r: Vec<f64> = (0..n).map(|i| a5r_of(f, i)).collect();
    let wpr: Vec<f64> = (0..n).map(|i| wpr_of(f, i)).collect();
    Res {
        ret,
        volatility,
        volume,
        cpv,
        crv,
        volume_volatility,
        high_volume,
        low_volume,
        high_low_volume,
        high_volume_rate,
        low_volume_rate,
        high_low_volume_rate,
        ab1: nan_mean(&a1),
        ab5: nan_mean(&a5),
        wpi: nan_mean(&wp),
        ab1r: nan_mean(&a1r),
        ab5r: nan_mean(&a5r),
        wpir: nan_mean(&wpr),
    }
}

fn get_stats2(f: &Frame, r: &Res) -> [f64; PER_GROUP] {
    let n = f.len();
    let lg = log_replace_inf(&f.vol);
    let pc = pct_change(&f.last_prc);
    let ret_pri = f.last_prc[n - 1] / f.last_prc[0] - 1.0;
    let ret = ret_pri / 12.0 * 237.0 - r.ret;
    let volatility_pri = nan_std(&pc, 1);
    let volatility = volatility_pri - r.volatility;
    let volume = nan_sum(&f.vol) / r.volume;
    let sim = f.sim.last().copied().unwrap_or(f64::NAN);
    let cpv_pri = corr(&f.last_prc, &lg);
    let crv_pri = corr(&pc, &lg);
    let cpv = cpv_pri - r.cpv;
    let crv = crv_pri - r.crv;
    let volume_volatility_pri = nan_std(&lg, 1);
    let volume_volatility = volume_volatility_pri - r.volume_volatility;
    let hp = nan_max(&f.last_prc);
    let lp = nan_min(&f.last_prc);
    let sel = |want: f64| -> Vec<f64> {
        (0..n)
            .filter(|&i| f.last_prc[i] == want)
            .map(|i| f.vol[i])
            .collect()
    };
    let high_volume = nan_sum(&sel(hp)) / r.high_volume;
    let low_volume = nan_sum(&sel(lp)) / r.low_volume;
    let high_low_volume = ((high_volume - low_volume) / r.high_low_volume).exp();
    let high_volume_rate_pri = high_volume / volume;
    let low_volume_rate_pri = low_volume / volume;
    let high_volume_rate = high_volume_rate_pri - r.high_volume_rate;
    let low_volume_rate = low_volume_rate_pri - r.low_volume_rate;
    let high_low_volume_rate_pri = high_volume_rate_pri - low_volume_rate_pri;
    let high_low_volume_rate = high_low_volume_rate_pri - r.high_low_volume_rate;
    let a1: Vec<f64> = (0..n).map(|i| a1_of(f, i)).collect();
    let a5: Vec<f64> = (0..n).map(|i| a5_of(f, i)).collect();
    let wp: Vec<f64> = (0..n).map(|i| f.w_ask[i] - f.w_bid[i]).collect();
    let a1r: Vec<f64> = (0..n).map(|i| a1r_of(f, i)).collect();
    let a5r: Vec<f64> = (0..n).map(|i| a5r_of(f, i)).collect();
    let wpr: Vec<f64> = (0..n).map(|i| wpr_of(f, i)).collect();
    let (ks, vs) = group_sum_kahan(&f.last_prc, &f.vol);
    let (freq_price, rare_price) = if ks.is_empty() {
        (f64::NAN, f64::NAN)
    } else {
        // 稳定排序后取 index[-1] / index[0]：并列时取键最大 / 最小的（见文件头「已知差异」）
        (ks[last_argmax(&vs)], ks[first_argmin_stable(&vs)])
    };
    let freq_rare_price = freq_price - rare_price;
    let freq_rare_price_rate = freq_rare_price / rare_price;
    [
        ret_pri,
        ret,
        volatility_pri,
        volatility,
        volume,
        sim,
        cpv_pri,
        crv_pri,
        cpv,
        crv,
        volume_volatility_pri,
        volume_volatility,
        high_volume,
        low_volume,
        high_low_volume,
        high_volume_rate_pri,
        low_volume_rate_pri,
        high_volume_rate,
        low_volume_rate,
        high_low_volume_rate_pri,
        high_low_volume_rate,
        nan_mean(&a1) / r.ab1,
        nan_mean(&a5) / r.ab5,
        nan_mean(&wp) / r.wpi,
        nan_mean(&a1r),
        nan_mean(&a1r) - r.ab1r,
        nan_mean(&a5r),
        nan_mean(&a5r) - r.ab5r,
        nan_mean(&wpr),
        nan_mean(&wpr) - r.wpir,
    ]
}

// ============================================================================
// one / sing / 入口
// ============================================================================

/// 第一个非 NaN 的最大值位置（pandas：df[df.sim == df.sim.max()].index[0]）
fn first_argmax(a: &[f64]) -> Option<usize> {
    let m = nan_max(a);
    if m.is_nan() {
        return None;
    }
    (0..a.len()).find(|&i| a[i] == m)
}

fn first_argmin(a: &[f64]) -> Option<usize> {
    let m = nan_min(a);
    if m.is_nan() {
        return None;
    }
    (0..a.len()).find(|&i| a[i] == m)
}

/// one(df_w, minute_price) → 90 个数
fn one(w: &Frame, minute_price: &[f64]) -> Option<Vec<f64>> {
    let num = minute_price.len();
    let sim = rolling_corr_vec(&w.last_prc, minute_price)?;
    let mut w2 = w.clone();
    w2.sim = sim;
    // 全部 last_prc 都是 NaN 时旧代码取 xp.index[-1] 会抛 IndexError → 整股 NaN
    if group_sum_kahan(&w2.last_prc, &w2.vol).0.is_empty() {
        return None;
    }
    let res = get_stats(&w2);
    let ix_max = first_argmax(&w2.sim)?;
    let ix_min = first_argmin(&w2.sim)?;
    let absim: Vec<f64> = w2.sim.iter().map(|x| x.abs()).collect();
    let ix_zero = first_argmin(&absim)?;
    let mut out = Vec::with_capacity(3 * PER_GROUP);
    for ix in [ix_max, ix_min, ix_zero] {
        if ix + 1 < num {
            return None; // 旧代码 iloc 负数下标会拿到别的行，实测不会走到；保守起见整股 NaN
        }
        let sub = w2.slice(ix + 1 - num, ix + 1);
        out.extend_from_slice(&get_stats2(&sub, &res));
    }
    Some(out)
}

fn sing(w: &Frame, will: &Frame) -> Option<Vec<f64>> {
    let (tmin, tmax) = w.t_range()?;
    let wsub = will.select(tmin, tmax);
    let mp_y = resample_1min_last(&wsub.t, &wsub.last_prc);
    let mp_t = resample_1min_last(&w.t, &w.last_prc);
    let a = one(w, &mp_y)?;
    let b = one(w, &mp_t)?;
    let mut out = a;
    out.extend_from_slice(&b);
    Some(out)
}

// ============================================================================
// 交易日历
// ============================================================================

pub fn trading_days() -> &'static [i64] {
    static DAYS: std::sync::OnceLock<Vec<i64>> = std::sync::OnceLock::new();
    DAYS.get_or_init(|| {
        let dir = crate::data_paths::basic_info_path("");
        match std::fs::read_to_string(dir.join("calendar.csv")) {
            Ok(s) => s
                .lines()
                .filter_map(|l| l.trim().parse::<i64>().ok())
                .collect(),
            Err(_) => Vec::new(),
        }
    })
}

pub fn last_trading_day(days: &[i64], date: i64) -> Option<i64> {
    let idx = days.partition_point(|&d| d < date);
    if idx == 0 {
        None
    } else {
        Some(days[idx - 1])
    }
}

pub fn compute_cross(_date: i64) -> Option<(Vec<String>, Vec<f32>)> {
    None
}

pub fn compute_stock(code: &str, date: i64) -> Option<Vec<f32>> {
    let days = trading_days();
    let prev = last_trading_day(days, date)?;
    let y = read_day(code, prev)?;
    let d = read_day(code, date)?;

    let m = d.select(LD_0930, LD_1130);
    let a = d.select(LD_1300, LD_1457);
    let r1 = sing(&m, &y)?;
    let r2 = sing(&a, &y)?;
    let r3 = sing(&d, &y)?;
    let mut out = r1;
    out.extend_from_slice(&r2);
    out.extend_from_slice(&r3);
    if out.len() != N_OUT {
        return None;
    }
    Some(out.into_iter().map(|v| v as f32).collect())
}
