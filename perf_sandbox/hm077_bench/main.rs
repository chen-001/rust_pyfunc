use std::time::Instant;
// ────────────────────────── 常量 ──────────────────────────

/// 一天的纳秒数。
const DAY_NS: i64 = 86_400_000_000_000;
/// exchtime（微秒，UTC）转成 pandas Timestamp 后要加的北京时区偏移。
const OFFSET_NS: i64 = 8 * 3600 * 1_000_000_000;
/// 下午时段整体前移 90 分钟。
const SHIFT_NS: i64 = 90 * 60 * 1_000_000_000;
const MORNING_START_NS: i64 = 34_200_000_000_000; // 09:30:00
const MORNING_END_NS: i64 = 41_400_000_000_000; // 11:30:00
const AFTERNOON_START_NS: i64 = 46_800_000_000_000; // 13:00:00
const AFTERNOON_END_NS: i64 = 53_820_000_000_000; // 14:57:00

/// ultra_sorted 输出的 27 列列名（旧代码把中文列名过了 pypinyin，这里直接写拼音）。
/// 只作对照记录，计算里用下标访问。
#[allow(dead_code)]
const FEATS: [&str; 27] = [
    "zui4jin4shi2jian1jian4ge2",
    "ping2jun1shi2jian1jian4ge2_1%",
    "ping2jun1shi2jian1jian4ge2_2%",
    "ping2jun1shi2jian1jian4ge2_3%",
    "ping2jun1shi2jian1jian4ge2_4%",
    "ping2jun1shi2jian1jian4ge2_5%",
    "ping2jun1shi2jian1jian4ge2_10%",
    "ping2jun1shi2jian1jian4ge2_20%",
    "ping2jun1shi2jian1jian4ge2_30%",
    "ping2jun1shi2jian1jian4ge2_40%",
    "ping2jun1shi2jian1jian4ge2_50%",
    "ping2jun1shi2jian1jian4ge2_quan2bu4",
    "jia4ge2fen1wei4shu4_10%",
    "jia4ge2fen1wei4shu4_20%",
    "jia4ge2fen1wei4shu4_30%",
    "jia4ge2fen1wei4shu4_40%",
    "jia4ge2fen1wei4shu4_50%",
    "jia4ge2fen1wei4shu4_60%",
    "jia4ge2fen1wei4shu4_70%",
    "jia4ge2fen1wei4shu4_80%",
    "jia4ge2fen1wei4shu4_90%",
    "jia4ge2fen1wei4shu4_quan2bu4",
    "ding4dan1bian1hao4",
    "mai3dan1biao1zhi4",
    "ding4dan1zong3liang4",
    "ding4dan1shi2jian1",
    "ding4dan1jia4ge2",
];
/// 「买单标识」列（0-based）。
const C_BUY_FLAG: usize = 23;
/// 「订单总量」列（groupby 的键）。
const C_ORDER_VOLUME: usize = 24;

/// 参与全部统计的 24 列：13 个时间间隔列（含 trend 序列）+ 11 个价格分位数列（含 trend 序列）。
const N24: [&str; 24] = [
    "zui4jin4shi2jian1jian4ge2",
    "ping2jun1shi2jian1jian4ge2_1%",
    "ping2jun1shi2jian1jian4ge2_2%",
    "ping2jun1shi2jian1jian4ge2_3%",
    "ping2jun1shi2jian1jian4ge2_4%",
    "ping2jun1shi2jian1jian4ge2_5%",
    "ping2jun1shi2jian1jian4ge2_10%",
    "ping2jun1shi2jian1jian4ge2_20%",
    "ping2jun1shi2jian1jian4ge2_30%",
    "ping2jun1shi2jian1jian4ge2_40%",
    "ping2jun1shi2jian1jian4ge2_50%",
    "ping2jun1shi2jian1jian4ge2_quan2bu4",
    "shi2jian1jian4ge2_trendxu4lie4",
    "jia4ge2fen1wei4shu4_10%",
    "jia4ge2fen1wei4shu4_20%",
    "jia4ge2fen1wei4shu4_30%",
    "jia4ge2fen1wei4shu4_40%",
    "jia4ge2fen1wei4shu4_50%",
    "jia4ge2fen1wei4shu4_60%",
    "jia4ge2fen1wei4shu4_70%",
    "jia4ge2fen1wei4shu4_80%",
    "jia4ge2fen1wei4shu4_90%",
    "jia4ge2fen1wei4shu4_quan2bu4",
    "jia4ge2fen1wei4shu4_trendxu4lie4",
];
/// 一个方向 960 个值里入选的 55 个名字（= 旧 names_save 去掉 `_fold`，顺序原样）。
const NAMES: [&str; 55] = [
    "mai3dan1shi2jian1jian4ge210perc_kurt_sameside",
    "mai3dan1_jia4ge2fen1wei4shu410perc_trend_volumeju4he2_sameside",
    "mai3dan1_jia4ge2fen1wei4shu4trendxu4lie4_trend_volumeju4he2_sameside",
    "mai3dan1shi2jian1jian4ge210perc_mean_volumeju4he2_diffside",
    "mai3dan1_corr_jia4ge2fen1wei4shu430perc_jia4ge2_sameside",
    "mai3dan1_jia4ge2fen1wei4shu490perc_kurt_sameside",
    "mai3dan1jia4ge2fen1wei4shu4trendxu4lie4_mean_sameside",
    "mai3dan1_jia4ge2fen1wei4shu480perc_kurt_diffside",
    "mai3dan1shi2jian1jian4ge2trendxu4lie4_trend_volumeju4he2_sameside",
    "mai3dan1shi2jian1jian4ge2trendxu4lie4_mean_volumeju4he2_diffside",
    "mai3dan1_corr_shi2jian1jian4ge21perc_jia4ge2_trend_diffside",
    "mai3dan1_corr_shi2jian1jian4ge2quan2bu4_jia4ge2_autocorr_diffside",
    "mai3dan1shi2jian1jian4ge240perc_kurt_diffside",
    "mai3dan1_corr_shi2jian1jian4ge2quan2bu4_jia4ge2_trend_volumeju4he2_diffside",
    "mai3dan1_jia4ge2fen1wei4shu460perc_trend_volumeju4he2_diffside",
    "mai3dan1_jia4ge2fen1wei4shu420perc_skew_diffside",
    "mai3dan1shi2jian1jian4ge25perc_mean_diffside",
    "mai3dan1_corr_shi2jian1jian4ge250perc_jia4ge2_trend_volumeju4he2_sameside",
    "mai3dan1jia4ge2fen1wei4shu4quan2bu4_std_sameside",
    "mai3dan1shi2jian1jian4ge2trendxu4lie4_std_diffside",
    "mai3dan1shi2jian1jian4ge22perc_mean_volumeju4he2_diffside",
    "mai3dan1shi2jian1jian4ge24perc_mean_diffside",
    "mai3dan1_jia4ge2fen1wei4shu490perc_trend_volumeju4he2_sameside",
    "mai3dan1_corr_jia4ge2fen1wei4shu450perc_cheng2jiao1shi2jian1_diffside",
    "mai3dan1shi2jian1jian4ge220perc_mean_volumeju4he2_diffside",
    "mai3dan1shi2jian1jian4ge2trendxu4lie4_skew_sameside",
    "mai3dan1shi2jian1jian4ge2trendxu4lie4_trend_diffside",
    "mai3dan1shi2jian1jian4ge23perc_kurt_sameside",
    "mai3dan1shi2jian1jian4ge250perc_kurt_diffside",
    "mai3dan1shi2jian1jian4ge2quan2bu4_mean_volumeju4he2_sameside",
    "mai3dan1_jia4ge2fen1wei4shu450perc_trend_volumeju4he2_diffside",
    "mai3dan1_corr_shi2jian1jian4ge250perc_jia4ge2_trend_volumeju4he2_diffside",
    "mai3dan1_corr_shi2jian1jian4ge2quan2bu4_jia4ge2_trend_volumeju4he2_sameside",
    "mai3dan1_corr_shi2jian1jian4ge2trendxu4lie4_jia4ge2_diffside",
    "mai3dan1_corr_jia4ge2fen1wei4shu4trendxu4lie4_ding4dan1bian1hao4_sameside",
    "mai3dan1_corr_zui4jin4shi2jian1jian4ge2_jia4ge2_sameside",
    "mai3dan1shi2jian1jian4ge2trendxu4lie4_skew_diffside",
    "mai3dan1_corr_shi2jian1jian4ge250perc_cheng2jiao1shi2jian1_diffside",
    "mai3dan1_corr_shi2jian1jian4ge2quan2bu4_ding4dan1bian1hao4_diffside",
    "mai3dan1_corr_jia4ge2fen1wei4shu4trendxu4lie4_ding4dan1bian1hao4_diffside",
    "mai3dan1shi2jian1jian4ge2trendxu4lie4_mean_diffside",
    "mai3dan1jia4ge2fen1wei4shu480perc_mean_diffside",
    "mai3dan1shi2jian1jian4ge21perc_mean_volumeju4he2_diffside",
    "mai3dan1shi2jian1jian4ge220perc_mean_sameside",
    "mai3dan1_jia4ge2fen1wei4shu450perc_kurt_diffside",
    "mai3dan1shi2jian1jian4ge2trendxu4lie4_autocorr_sameside",
    "mai3dan1_corr_jia4ge2fen1wei4shu480perc_cheng2jiao1shi2jian1_sameside",
    "mai3dan1_jia4ge2fen1wei4shu410perc_trend_diffside",
    "mai3dan1_jia4ge2fen1wei4shu460perc_trend_volumeju4he2_sameside",
    "mai3dan1_corr_shi2jian1jian4ge22perc_jia4ge2_diffside",
    "mai3dan1_jia4ge2fen1wei4shu4quan2bu4_trend_volumeju4he2_diffside",
    "mai3dan1_corr_shi2jian1jian4ge250perc_ding4dan1bian1hao4_diffside",
    "mai3dan1_corr_zui4jin4shi2jian1jian4ge2_ding4dan1bian1hao4_diffside",
    "mai3dan1_corr_shi2jian1jian4ge21perc_ding4dan1bian1hao4_diffside",
    "mai3dan1_jia4ge2fen1wei4shu4trendxu4lie4_trend_volumeju4he2_diffside",
];

/// 上面 55 个名字在「两个方向拼起来的 1920 个值」里的下标。
const IDX: [usize; 55] = [
    270, 805, 815, 1638, 15, 285, 215, 1244, 804, 1644, 1273, 1259, 1233, 1763,
    1770, 1214, 1157, 802, 238, 1188, 1634, 1156, 813, 1049, 1639, 252, 1284, 267,
    1234, 683, 1769, 1762, 803, 972, 71, 0, 1212, 1042, 1019, 1031, 1164, 1172,
    1633, 199, 1241, 300, 92, 1285, 810, 962, 1774, 1018, 1008, 1009, 1775,
];

/// 一个方向的输出长度（8 块 × 24 + 8 块 × 24，按上面第五条的拼装顺序）。
const ONE_LEN: usize = 960;
/// 两个方向拼起来的总长度。
const FULL_LEN: usize = ONE_LEN * 2;

pub fn names() -> Vec<String> {
    NAMES.iter().map(|s| s.to_string()).collect()
}
// ────────────────────────── 数值小工具 ──────────────────────────

/// numpy 的成对求和（PW_BLOCKSIZE=128、8 路展开），pandas 的 .sum()/.mean() 走这条。
fn pairwise_sum(a: &[f64]) -> f64 {
    let n = a.len();
    if n < 8 {
        let mut r = 0.0f64;
        for i in 0..n {
            r += a[i];
        }
        return r;
    }
    if n <= 128 {
        let mut r = [a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]];
        let mut i = 8usize;
        let stop = n - (n % 8);
        while i < stop {
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
    pairwise_sum(&a[..n2]) + pairwise_sum(&a[n2..])
}

/// np.corrcoef(a, b)[0,1]：两遍法（np.average 的成对求和 → 中心化 → 点积/(n-1)）
/// → `(c/sa)/sb` → clip 到 [-1,1]。成对完整观测由调用方事先丢掉 NaN。
fn np_corrcoef(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len();
    if n == 0 {
        return f64::NAN;
    }
    let fact = n as f64 - 1.0;
    let ma = pairwise_sum(a) / n as f64;
    let mb = pairwise_sum(b) / n as f64;
    let (mut daa, mut dbb, mut dab) = (0.0f64, 0.0f64, 0.0f64);
    for i in 0..n {
        let x = a[i] - ma;
        let y = b[i] - mb;
        daa += x * x;
        dbb += y * y;
        dab += x * y;
    }
    let r = (dab / fact) / (daa / fact).sqrt() / (dbb / fact).sqrt();
    if r.is_nan() {
        f64::NAN
    } else {
        r.clamp(-1.0, 1.0)
    }
}

/// Series.corr(other)：成对完整观测 + np.corrcoef。
fn series_corr(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    let mut x: Vec<f64> = Vec::with_capacity(n);
    let mut y: Vec<f64> = Vec::with_capacity(n);
    for i in 0..n {
        if !a[i].is_nan() && !b[i].is_nan() {
            x.push(a[i]);
            y.push(b[i]);
        }
    }
    if x.is_empty() {
        return f64::NAN;
    }
    np_corrcoef(&x, &y)
}

/// DataFrame.corrwith(DataFrame) 的逐列口径：num/dom，dom = (count-1)*std_l*std_r。
fn df_corrwith(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    let mut la: Vec<f64> = Vec::with_capacity(n);
    let mut lb: Vec<f64> = Vec::with_capacity(n);
    for i in 0..n {
        if !a[i].is_nan() && !b[i].is_nan() {
            la.push(a[i]);
            lb.push(b[i]);
        }
    }
    let cnt = la.len();
    if cnt == 0 {
        return f64::NAN;
    }
    let ma = pairwise_sum(&la) / cnt as f64;
    let mb = pairwise_sum(&lb) / cnt as f64;
    let (mut num, mut va, mut vb) = (0.0f64, 0.0f64, 0.0f64);
    for i in 0..cnt {
        let da = la[i] - ma;
        let db = lb[i] - mb;
        num += da * db;
        va += da * da;
        vb += db * db;
    }
    let d = cnt as f64 - 1.0;
    let dom = d * (va / d).sqrt() * (vb / d).sqrt();
    num / dom
}

/// rp.trend（trend_2d 的一维版）：丢掉非有限值后与「原始位置 +1」做 Pearson；
/// 样本 < 2 或任一侧方差为 0 给 0.0（不是 NaN）。
fn trend_1d(v: &[f64]) -> f64 {
    let mut values: Vec<f64> = Vec::new();
    let mut idx: Vec<f64> = Vec::new();
    for (i, &x) in v.iter().enumerate() {
        if x.is_finite() {
            values.push(x);
            idx.push((i + 1) as f64);
        }
    }
    let n = values.len();
    if n < 2 {
        return 0.0;
    }
    let mean_x = values.iter().sum::<f64>() / n as f64;
    let mean_y = idx.iter().sum::<f64>() / n as f64;
    let (mut cov, mut vx, mut vy) = (0.0f64, 0.0f64, 0.0f64);
    for i in 0..n {
        let dx = values[i] - mean_x;
        let dy = idx[i] - mean_y;
        cov += dx * dy;
        vx += dx * dx;
        vy += dy * dy;
    }
    if vx == 0.0 || vy == 0.0 {
        return 0.0;
    }
    cov / (vx.sqrt() * vy.sqrt())
}

/// 从 features_f64 的 (vals, names) 里按名字取值。
fn pick(vals: &[f64], names: &[String], want: &str) -> f64 {
    for (i, n) in names.iter().enumerate() {
        if n == want {
            return vals[i];
        }
    }
    f64::NAN
}
// ────────────────────────── 读数据 ──────────────────────────

#[derive(Clone, Copy)]
struct Row {
    ns: i64,
    price: f64,
    volume: f64,
    turnover: f64,
    flag: i32,
    ask: i64,
    bid: i64,
}

/// 去掉字段两端的空白（含行尾的 `\r`）。
fn trim_ascii(raw: &[u8]) -> &[u8] {
    let mut s = 0usize;
    let mut e = raw.len();
    while s < e && raw[s].is_ascii_whitespace() {
        s += 1;
    }
    while e > s && raw[e - 1].is_ascii_whitespace() {
        e -= 1;
    }
    &raw[s..e]
}

/// 取字段的整数前缀（`"32.0"` 给 32）。
fn int_prefix(raw: &[u8]) -> i64 {
    let b = trim_ascii(raw);
    let mut i = 0usize;
    let mut neg = false;
    if i < b.len() && (b[i] == b'-' || b[i] == b'+') {
        neg = b[i] == b'-';
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

fn pow10_table() -> &'static [f64; 309] {
    static T: std::sync::OnceLock<[f64; 309]> = std::sync::OnceLock::new();
    T.get_or_init(|| {
        let mut t = [1.0f64; 309];
        for (k, v) in t.iter_mut().enumerate() {
            *v = format!("1e{}", k).parse::<f64>().unwrap();
        }
        t
    })
}

/// pandas C 解析器 `precise_xstrtod` 的逐行复刻（与 hm033/hm090 里那份一致）。
fn parse_f64_pandas(b: &[u8]) -> f64 {
    const MAX_DIGITS: usize = 17;
    let n = b.len();
    let mut p = 0usize;
    while p < n && matches!(b[p], b' ' | b'\t' | b'\r' | b'\n') {
        p += 1;
    }
    let mut negative = false;
    if p < n && (b[p] == b'-' || b[p] == b'+') {
        negative = b[p] == b'-';
        p += 1;
    }
    let mut number = 0.0f64;
    let mut exponent: i32 = 0;
    let mut num_digits = 0usize;
    let mut num_decimals = 0usize;
    while p < n && b[p].is_ascii_digit() {
        if num_digits < MAX_DIGITS {
            number = number * 10.0 + (b[p] - b'0') as f64;
            num_digits += 1;
        } else {
            exponent += 1;
        }
        p += 1;
    }
    if p < n && b[p] == b'.' {
        p += 1;
        while num_digits < MAX_DIGITS && p < n && b[p].is_ascii_digit() {
            number = number * 10.0 + (b[p] - b'0') as f64;
            p += 1;
            num_digits += 1;
            num_decimals += 1;
        }
        if num_digits >= MAX_DIGITS {
            while p < n && b[p].is_ascii_digit() {
                p += 1;
            }
        }
        exponent -= num_decimals as i32;
    }
    if num_digits == 0 {
        return 0.0;
    }
    if negative {
        number = -number;
    }
    if p < n && (b[p] | 0x20) == b'e' {
        p += 1;
        let mut eneg = false;
        if p < n && (b[p] == b'-' || b[p] == b'+') {
            eneg = b[p] == b'-';
            p += 1;
        }
        let mut nn: i32 = 0;
        let mut nd = 0usize;
        while nd < MAX_DIGITS && p < n && b[p].is_ascii_digit() {
            nn = nn * 10 + (b[p] - b'0') as i32;
            nd += 1;
            p += 1;
        }
        if eneg {
            exponent -= nn;
        } else {
            exponent += nn;
        }
    }
    if exponent > 308 {
        f64::INFINITY
    } else if exponent > 0 {
        number * pow10_table()[exponent as usize]
    } else if exponent < -308 {
        if exponent < -616 {
            0.0
        } else {
            let t = pow10_table();
            (number / t[(-308 - exponent) as usize]) / t[308]
        }
    } else {
        number / pow10_table()[(-exponent) as usize]
    }
}

/// pandas conversion.pyx 的 cast_from_unit_vectorized（unit='s' → 'ns'）：
///   base = (int64)val; frac = val - base; frac = np.round(frac, 9);
///   out = base*1e9 + (int64)(frac*1e9)
/// 不是「微秒 × 1000」，这条链自带 ±120ns 噪声。
fn pandas_time_ns(exchtime_us: f64) -> i64 {
    let val = exchtime_us / 1e6;
    let base = val as i64;
    let frac = val - base as f64;
    let y = (frac * 1e9).round_ties_even() / 1e9;
    base.wrapping_mul(1_000_000_000)
        .wrapping_add((y * 1e9) as i64)
}

/// 读一个交易日的逐笔，复刻 dwo.read_trade + dwo.adjust_afternoon。
///
/// 行序 = 上午全部（文件序）在前、下午全部（文件序）在后（下午整体减 90 分钟）。
fn read_day(code: &str, date: i64) -> Option<Vec<Row>> {
    const COL_EXCHTIME: usize = 4;
    const COL_PRICE: usize = 7;
    const COL_VOLUME: usize = 8;
    const COL_TURNOVER: usize = 9;
    const COL_FLAG: usize = 10;
    const COL_ASK_ORDER: usize = 13;
    const COL_BID_ORDER: usize = 14;

    let filename = format!("{}_{}_transaction.csv", code, date);
    let mut data = None;
    for root in vec![std::path::PathBuf::from("/ssd_data/stock")] {
        let path = root
            .join(date.to_string())
            .join("transaction")
            .join(&filename);
        if let Ok(bytes) = std::fs::read(&path) {
            data = Some(bytes);
            break;
        }
    }
    let data = data?;

    let mut morning: Vec<Row> = Vec::new();
    let mut afternoon: Vec<Row> = Vec::new();
    for (lineno, line) in data.split(|&b| b == b'\n').enumerate() {
        if lineno == 0 || line.is_empty() {
            continue;
        }
        let mut fields: [&[u8]; 15] = [&[][..]; 15];
        let mut commas = 0usize;
        let mut start = 0usize;
        for (i, &b) in line.iter().enumerate() {
            if b == b',' {
                if commas < 15 {
                    fields[commas] = &line[start..i];
                }
                commas += 1;
                start = i + 1;
            }
        }
        if commas < 15 {
            fields[commas] = &line[start..];
        }
        if commas + 1 < 15 {
            continue;
        }
        let exchtime_us = parse_f64_pandas(trim_ascii(fields[COL_EXCHTIME]));
        if exchtime_us == 0.0 {
            continue;
        }
        let flag = int_prefix(fields[COL_FLAG]) as i32;
        if flag == 32 {
            continue;
        }
        let ns = pandas_time_ns(exchtime_us).wrapping_add(OFFSET_NS);
        let tod = ns.rem_euclid(DAY_NS);
        let mut row = Row {
            ns,
            price: parse_f64_pandas(trim_ascii(fields[COL_PRICE])),
            volume: parse_f64_pandas(trim_ascii(fields[COL_VOLUME])),
            turnover: parse_f64_pandas(trim_ascii(fields[COL_TURNOVER])),
            flag,
            ask: int_prefix(fields[COL_ASK_ORDER]),
            bid: int_prefix(fields[COL_BID_ORDER]),
        };
        if (MORNING_START_NS..=MORNING_END_NS).contains(&tod) {
            morning.push(row);
        } else if (AFTERNOON_START_NS..=AFTERNOON_END_NS).contains(&tod) {
            row.ns -= SHIFT_NS;
            afternoon.push(row);
        }
    }
    morning.append(&mut afternoon);
    if morning.is_empty() {
        return None;
    }
    Some(morning)
}
// ────────────────────────── 排序后的当日数据 ──────────────────────────

/// hm077 的 v2 通路只用 ns / price / volume / ask / bid；turnover 与 flag 保留下来是为了
/// 与 hm076 的 Day 保持同一形状（读盘阶段两者本就一样），标 dead_code 免得刷警告。
#[allow(dead_code)]
struct Day {
    ns: Vec<i64>,
    price: Vec<f64>,
    volume: Vec<f64>,
    turnover: Vec<f64>,
    flag: Vec<i32>,
    ask: Vec<i64>,
    bid: Vec<i64>,
}

fn build_day(rows: Vec<Row>) -> Day {
    let n = rows.len();
    let mut order: Vec<usize> = (0..n).collect();
    // np.lexsort((exchtime, volume))：主键 volume、次键 exchtime，稳定。
    order.sort_by(|&i, &j| {
        rows[i]
            .volume
            .partial_cmp(&rows[j].volume)
            .unwrap()
            .then(rows[i].ns.cmp(&rows[j].ns))
    });

    let mut d = Day {
        ns: Vec::with_capacity(n),
        price: Vec::with_capacity(n),
        volume: Vec::with_capacity(n),
        turnover: Vec::with_capacity(n),
        flag: Vec::with_capacity(n),
        ask: Vec::with_capacity(n),
        bid: Vec::with_capacity(n),
    };
    for &i in order.iter() {
        d.ns.push(rows[i].ns);
        d.price.push(rows[i].price);
        d.volume.push(rows[i].volume);
        d.turnover.push(rows[i].turnover);
        d.flag.push(rows[i].flag);
        d.ask.push(rows[i].ask);
        d.bid.push(rows[i].bid);
    }
    d
}
// ────────── rp.calculate_order_time_gap_and_price_percentile_ultra_sorted_v2 ──────────

/// v2 的订单分组：按「下单方向」分买卖两侧（不再用成交 flag）。
struct OrderVolumeGroupV2 {
    indices: Vec<usize>,
    times: Vec<f64>,
    prices: Vec<f64>,
    /// true = 买单（bid_order 那一支）
    order_types: Vec<bool>,
    ask_indices: Vec<usize>,
    bid_indices: Vec<usize>,
}

impl OrderVolumeGroupV2 {
    fn new() -> Self {
        Self {
            indices: Vec::new(),
            times: Vec::new(),
            prices: Vec::new(),
            order_types: Vec::new(),
            ask_indices: Vec::new(),
            bid_indices: Vec::new(),
        }
    }

    fn add_record(&mut self, orig_idx: usize, time: f64, price: f64, is_bid: bool) {
        let group_idx = self.indices.len();
        self.indices.push(orig_idx);
        self.times.push(time);
        self.prices.push(price);
        self.order_types.push(is_bid);
        if is_bid {
            self.bid_indices.push(group_idx);
        } else {
            self.ask_indices.push(group_idx);
        }
    }

    /// 基于订单类型的时间距离计算（照抄 v2 源码）。
    fn find_nearest_records_ultra_fast_v2(
        &self,
        current_group_idx: usize,
        target_indices: &[usize],
        max_records: usize,
    ) -> Vec<(f64, f64)> {
        if target_indices.is_empty() {
            return Vec::new();
        }
        let current_time = self.times[current_group_idx];
        let mut time_distances: Vec<(f64, f64)> = Vec::new();
        for &target_idx in target_indices.iter() {
            if target_idx != current_group_idx {
                let time_diff = (current_time - self.times[target_idx]).abs();
                let price = self.prices[target_idx];
                time_distances.push((time_diff, price));
            }
        }
        time_distances.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        if time_distances.len() > max_records {
            time_distances.truncate(max_records);
        }
        time_distances
    }

    /// 基于订单类型的批量计算指标（照抄 v2 源码）。
    fn compute_all_indicators_ultra_fast_v2(
        &self,
        results: &mut [f64],
        min_count: usize,
        use_flag: &str,
    ) {
        let group_size = self.indices.len();
        if group_size < min_count {
            return;
        }
        for current_group_idx in 0..group_size {
            let current_is_bid = self.order_types[current_group_idx];
            let target_indices: Vec<usize> = match use_flag {
                "same" => {
                    if current_is_bid {
                        self.bid_indices
                            .iter()
                            .filter(|&&idx| idx != current_group_idx)
                            .cloned()
                            .collect()
                    } else {
                        self.ask_indices
                            .iter()
                            .filter(|&&idx| idx != current_group_idx)
                            .cloned()
                            .collect()
                    }
                }
                "diff" => {
                    if current_is_bid {
                        self.ask_indices.clone()
                    } else {
                        self.bid_indices.clone()
                    }
                }
                _ => (0..group_size)
                    .filter(|&idx| idx != current_group_idx)
                    .collect(),
            };
            if target_indices.len() < min_count {
                continue;
            }
            let time_distances = self.find_nearest_records_ultra_fast_v2(
                current_group_idx,
                &target_indices,
                target_indices.len(),
            );
            if time_distances.len() >= min_count {
                let orig_idx = self.indices[current_group_idx];
                let current_price = self.prices[current_group_idx];
                calculate_order_indicators_ultra_fast(
                    &mut results[orig_idx * 22..orig_idx * 22 + 22],
                    &time_distances,
                    current_price,
                );
            }
        }
    }
}
/// 超高速指标计算（照抄 src/order_records_ultra_sorted.rs）。
fn calculate_order_indicators_ultra_fast(
    result_row: &mut [f64],
    time_distances: &[(f64, f64)],
    current_price: f64,
) {
    let total_count = time_distances.len();
    if total_count == 0 {
        return;
    }
    result_row[0] = time_distances[0].0;

    let percentages = [0.01, 0.02, 0.03, 0.04, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50];
    let mut cumulative_sum = 0.0;
    let mut last_count = 0;
    for (idx, &pct) in percentages.iter().enumerate() {
        let count = ((total_count as f64 * pct).ceil() as usize)
            .max(1)
            .min(total_count);
        for j in last_count..count {
            cumulative_sum += time_distances[j].0;
        }
        result_row[idx + 1] = cumulative_sum / count as f64;
        last_count = count;
    }
    for j in last_count..total_count {
        cumulative_sum += time_distances[j].0;
    }
    result_row[11] = cumulative_sum / total_count as f64;

    let price_percentages = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00];
    let mut all_prices: Vec<f64> = Vec::with_capacity(total_count);
    for &(_, price) in time_distances.iter() {
        all_prices.push(price);
    }
    all_prices.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    for (idx, &pct) in price_percentages.iter().enumerate() {
        let count = if pct == 1.0 {
            total_count
        } else {
            ((total_count as f64 * pct).ceil() as usize)
                .max(1)
                .min(total_count)
        };
        let prices_slice = &all_prices[..count];
        let rank = match prices_slice.binary_search_by(|&p| p.partial_cmp(&current_price).unwrap()) {
            Ok(pos) => pos,
            Err(pos) => pos,
        };
        result_row[idx + 12] = rank as f64 / prices_slice.len() as f64;
    }
}

fn find_volume_ranges(volumes: &[f64]) -> Vec<(f64, usize, usize)> {
    if volumes.is_empty() {
        return Vec::new();
    }
    let mut ranges = Vec::new();
    let mut current_volume = volumes[0];
    let mut start_idx = 0;
    for i in 1..volumes.len() {
        if volumes[i] != current_volume {
            ranges.push((current_volume, start_idx, i));
            current_volume = volumes[i];
            start_idx = i;
        }
    }
    ranges.push((current_volume, start_idx, volumes.len()));
    ranges
}
/// 平铺的 (n × 27) 矩阵，等价于 rp.calculate_order_time_gap_and_price_percentile_ultra_sorted_v2。
/// exchtime 传进来是纳秒 i64（df.exchtime.to_numpy(float) 的值），内部 `/ 1e9` 变秒。
fn ultra_sorted_v2(
    volume: &[f64],
    exchtime_ns: &[i64],
    price: &[f64],
    ask_order: &[i64],
    bid_order: &[i64],
    min_count: usize,
    use_flag: &str,
) -> Vec<f64> {
    let n = volume.len();
    let exchtime_data: Vec<f64> = exchtime_ns.iter().map(|&t| t as f64 / 1e9).collect();

    // 1) 订单聚合：ask 分支在前、bid 分支在后，各写各的；键只有 order_id。
    let mut orders: Vec<(i64, bool, f64, f64, f64)> = Vec::new(); // (order_id, is_bid, vol, price, time)
    let mut order_map: std::collections::HashMap<i64, usize> = std::collections::HashMap::new();
    for i in 0..n {
        if ask_order[i] != 0 {
            let order_id = ask_order[i];
            if let Some(&oi) = order_map.get(&order_id) {
                let old_vol = orders[oi].2;
                let old_price = orders[oi].3;
                let new_vol = old_vol + volume[i];
                orders[oi].2 = new_vol;
                orders[oi].3 = (old_vol * old_price + volume[i] * price[i]) / new_vol;
                orders[oi].4 = orders[oi].4.max(exchtime_data[i]);
            } else {
                let oi = orders.len();
                orders.push((order_id, false, volume[i], price[i], exchtime_data[i]));
                order_map.insert(order_id, oi);
            }
        }
        if bid_order[i] != 0 {
            let order_id = bid_order[i];
            if let Some(&oi) = order_map.get(&order_id) {
                let old_vol = orders[oi].2;
                let old_price = orders[oi].3;
                let new_vol = old_vol + volume[i];
                orders[oi].2 = new_vol;
                orders[oi].3 = (old_vol * old_price + volume[i] * price[i]) / new_vol;
                orders[oi].4 = orders[oi].4.max(exchtime_data[i]);
            } else {
                let oi = orders.len();
                orders.push((order_id, true, volume[i], price[i], exchtime_data[i]));
                order_map.insert(order_id, oi);
            }
        }
    }

    // 2) 按订单 volume、再按订单时间排序（不稳定排序，与源码一致）
    orders.sort_unstable_by(|a, b| {
        a.2.partial_cmp(&b.2)
            .unwrap()
            .then(a.4.partial_cmp(&b.4).unwrap())
    });

    // 3) 按订单 volume 分组
    let order_volumes: Vec<f64> = orders.iter().map(|o| o.2).collect();
    let order_ranges = find_volume_ranges(&order_volumes);

    let mut order_groups: Vec<OrderVolumeGroupV2> = Vec::new();
    for (_, start_idx, end_idx) in order_ranges.iter() {
        let mut group = OrderVolumeGroupV2::new();
        for i in *start_idx..*end_idx {
            let (_, is_bid, _, p, time) = orders[i];
            group.add_record(i, time, p, is_bid);
        }
        order_groups.push(group);
    }

    // 4) 逐组算 22 个指标
    let mut order_results = vec![f64::NAN; orders.len() * 22];
    for group in order_groups.iter() {
        group.compute_all_indicators_ultra_fast_v2(&mut order_results, min_count, use_flag);
    }

    // 5) 映射回逐笔行；ask 分支先写、bid 分支后写（后者覆盖前者，原样保留）
    let mut results = vec![f64::NAN; n * 27];
    for i in 0..n {
        if ask_order[i] != 0 {
            if let Some(&oi) = order_map.get(&ask_order[i]) {
                for j in 0..22 {
                    results[i * 27 + j] = order_results[oi * 22 + j];
                }
                results[i * 27 + 22] = ask_order[i] as f64;
                results[i * 27 + 23] = 0.0;
                results[i * 27 + 24] = orders[oi].2;
                results[i * 27 + 25] = orders[oi].4;
                results[i * 27 + 26] = orders[oi].3;
            }
        }
        if bid_order[i] != 0 {
            if let Some(&oi) = order_map.get(&bid_order[i]) {
                for j in 0..22 {
                    results[i * 27 + j] = order_results[oi * 22 + j];
                }
                results[i * 27 + 22] = bid_order[i] as f64;
                results[i * 27 + 23] = 1.0;
                results[i * 27 + 24] = orders[oi].2;
                results[i * 27 + 25] = orders[oi].4;
                results[i * 27 + 26] = orders[oi].3;
            }
        }
    }
    results
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let code = args.get(1).cloned().unwrap_or_else(|| "000001".to_string());
    let date: i64 = args.get(2).map(|s| s.parse().unwrap()).unwrap_or(20220819);
    let t0 = Instant::now();
    let rows = read_day(&code, date).expect("no data");
    let t_read = t0.elapsed();
    let n = rows.len();
    let t1 = Instant::now();
    let d = build_day(rows);
    let t_build = t1.elapsed();
    println!("rows={} read_ms={:.1} build_day_ms={:.1}", n, t_read.as_secs_f64() * 1000.0, t_build.as_secs_f64() * 1000.0);
    for dir in ["same", "diff"] {
        let t = Instant::now();
        let flat = ultra_sorted_v2(&d.volume, &d.ns, &d.price, &d.ask, &d.bid, 100, dir);
        println!("ultra_sorted_v2[{}] = {:.1} ms  cells={}", dir, t.elapsed().as_secs_f64() * 1000.0, flat.len() / 27);
    }
}
