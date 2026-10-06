//! hm69「本福特定律」——旧框架（design_whatever_out.TickDataBase）的 Rust 复现。
//!
//! 旧文件：/home/chenzongwei/pythoncode/成果hm57_hm69_本福特定律/hm69_okay.py
//! 数据来源：Level2 逐笔。run_base 里是 go 型的 `class Go(dwo.TickDataBase)`，
//! 参数 get_many=0 / read_full=0 / change_time=1，也就是「一整天的逐笔」一张表，
//! 窗口由旧代码自己切。逐股系列（kind='per_stock'）。
//!
//! ════════════════════════════════════════════════════════════════════
//! 一、旧 get_factor 逐步对照（写代码前先抄在这里，方便复核）
//! ════════════════════════════════════════════════════════════════════
//! 1) trade_data = read_trade_data_single(with_retreat=1)
//!      with_retreat=1 → **不丢** flag==32 的撤单行（旧代码后面还要用撤单）。
//!      change_time=1 → deal_data 只把 exchtime/localtime 从 epoch 微秒换成 Timestamp。
//!      读的是 CSV 的 9 列 date/exchtime/price/volume/turnover/flag/localtime/ask_order/bid_order。
//! 2) trade_data = trade_data.set_index('exchtime')
//! 3) trade_data = dwo.adjust_afternoon(trade_data)
//!      以 exchtime 为索引切两段：[09:30:00, 11:30:00] 与 [13:00:00, 14:57:00]（两端都含），
//!      下午段索引整体减 90 分钟，然后 concat（上午段在前、各自保持原行序）。
//!      窗口之外的行整行丢掉；调整后所有时刻落在 09:30:00 ~ 13:27:00。
//! 4) 五个时段（都是上面那张表的子集，闭区间）：
//!      res1 allday    整张调整后的表
//!      res2 open30    09:30:00 ~ 10:00:00
//!      res3 morning   09:30:00 ~ 11:30:00
//!      res4 afternoon 11:30:00 ~ 13:27:00   （= 真实 13:00:00 ~ 14:57:00）
//!      res5 close30   13:00:00 ~ 13:27:00   （= 真实 14:30:00 ~ 14:57:00）
//!    注意 morning 的上界与 afternoon 的下界都是 11:30:00，调整后正好落在
//!    真实 13:00:00（有 4 行）上，所以同一个交易日的这两段会有一处重叠。
//! 5) 每个时段跑一次 one(trade_data)，用同一套 10 条序列算 20 个统计量：
//!      成交量        price != 0 的 volume
//!      成交金额      price != 0 的 turnover
//!      主动买入成交量 flag == 66 且 price != 0 的 volume
//!      主动卖出成交量 flag == 83 且 price != 0 的 volume
//!      买单编号      price != 0 的 bid_order
//!      卖单编号      price != 0 的 ask_order
//!      买单撤单量    flag == 32 且 bid_order > 0 的 volume
//!      卖单撤单量    flag == 32 且 ask_order > 0 的 volume
//!      买单撤单编号  flag == 32 且 bid_order > 0 的 bid_order
//!      卖单撤单编号  flag == 32 且 ask_order > 0 的 ask_order
//!    对每条序列：
//!      positive = 序列里 > 0 的值（NaN 不算）
//!      first_digits = positive.astype(str).str.slice(0, 1)  ← 首**字符**，不是首位有效数字
//!      若 positive 为空 → 该序列 20 个统计量全 NaN
//!      observed[d] = first_digits 里字符 d 的占比（value_counts(normalize=True) 后
//!                    reindex 到 '1'..'9'，没出现的给 NaN）
//!      theoretical[d] = log10(1 + 1/d)
//!      deviation[d] = observed[d] - theoretical[d]
//!    20 个统计量的顺序（names_single 就是按这个顺序拼的）：
//!      correlation          observed.corr(theoretical)
//!      deviation_trend      deviation.reset_index().corr().iloc[0,1]
//!      deviation_trend_abs  同上，用 |deviation|
//!      deviation_std        deviation.std()          （ddof=1，跳过 NaN）
//!      deviation_abs_std    |deviation|.std()
//!      deviation_skew       deviation.skew()         （调整 Fisher-Pearson G1）
//!      deviation_abs_skew
//!      deviation_max        deviation.max()
//!      deviation_abs_max
//!      deviation_min
//!      deviation_abs_min
//!      deviation_abs_mean   |deviation|.mean()
//!      deviation_median
//!      deviation_abs_median
//!      deviation_autocorr1  deviation.autocorr(1) = deviation.corr(deviation.shift(1))
//!      deviation_abs_autocorr1
//!      deviation_autocorr2 / deviation_abs_autocorr2
//!      deviation_autocorr3 / deviation_abs_autocorr3
//!    再补 4 组各 20 个，都是上面 20 个统计量按位相减/取绝对值：
//!      ask&bid_order_diff            = stats(买单编号) - stats(卖单编号)
//!      ask&bid_order_diff_abs        = |上一条|
//!      buy_cancels&sell_cancels_diff = stats(买单撤单编号) - stats(卖单撤单编号)
//!      ..._abs
//!    每个时段 = 10×20 + 4×20 = 280 个值；五个时段共 1400 个。
//! 6) 名称：10 条序列（见上）+ 4 组差值的名字打头，后接 20 个统计量名；
//!    再按时段加前缀/后缀 —— 第一段加 `all_` 前缀，其余四段加 `_open30` / `_morning` /
//!    `_afternoon` / `_close30` 后缀。顺序 = 时段在前的拼接顺序。
//!
//! ════════════════════════════════════════════════════════════════════
//! 二、时间戳：必须照抄 pandas 的「微秒 → 纳秒」这一步
//! ════════════════════════════════════════════════════════════════════
//! deal_data 里是 `pd.to_timedelta(df.exchtime / 1e6, unit="s") + Timestamp("1970-01-01 08:00:00")`。
//! pandas 走 conversion.pyx 的 cast_from_unit_vectorized：
//!     val  = exchtime_us / 1e6              （f64 除法）
//!     base = (int64)val                     （向零截断）
//!     frac = np.round(val - base, 9)        （= rint(x*1e9)/1e9，四舍六入五成双）
//!     ns   = base * 1e9 + (int64)(frac * 1e9)
//! 再 + 8 小时的纳秒偏移。这条链**不是**整数运算：实测 000001@20250303 的 76488 行里
//! 72787 行的 ns 与「微秒 × 1000」差 1~114 ns，其中约一半是负的。
//! between_time 比的是**当日微秒**（DatetimeIndex._get_time_micros：ns 先对 86400e9 取模
//! 再整除 1000，向下取整），所以那 1ns 的负噪声会把「正好落在 09:30:00.000000」这样的行
//! 推到 09:29:59.999999，直接改变窗口归属。必须逐位复刻，不能拿整微秒顶替。
//!
//! ════════════════════════════════════════════════════════════════════
//! 三、首字符：pandas 的 astype(str) 不是「首位有效数字」
//! ════════════════════════════════════════════════════════════════════
//! positive.astype(str).str.slice(0,1) 取的是 Python/numpy 浮点（或整数）字符串的**第一个字符**：
//!   · x >= 1 且 x < 1e16 → 定点写法，首字符 = 整数部分首位（= 首位有效数字）
//!   · 0 < x < 1e-4       → 科学计数法（'1e-05'），首字符 = 首位有效数字
//!   · 1e-4 <= x < 1      → 定点写法（'0.001'），首字符是 '0'
//! 实测四组 (日期,股票) 的 3604577 个正值上，「repr(float(v))[0]」与 pandas 的
//! astype(str)[0] 逐值相同，所以这里按上面三条规则算，不真的做字符串格式化
//! （只有 1e-4 那个分支才退化到 format!("{:e}")）。
//! 注意 '0'、'inf'、'nan' 这些首字符不在 '1'..'9' 里，不进任何直方图柱，
//! 但**要计入分母**（value_counts(normalize=True) 的分母是全部正值个数）。
//!
//! ════════════════════════════════════════════════════════════════════
//! 四、统计量的算法（对着 pandas 2.3.3 源码抄）
//! ════════════════════════════════════════════════════════════════════
//! · observed.corr(theoretical) 和 deviation.autocorr(lag)：Series.corr → nanops.nancorr
//!   → np.corrcoef（两遍法：先均值后 Σ(dx·dy)），成对丢 NaN，有效点 < 2 或方差为 0 给 NaN。
//! · deviation.reset_index().corr().iloc[0,1]：DataFrame.corr → libalgos.nancorr，
//!   是 **Welford** 增量算法，且除数写成 sqrt(ssqdmx * ssqdmy)（不再除以 n-1），
//!   掩码用 isfinite（inf 也算无效）。照抄，别用两遍法顶替。
//! · std：nanvar 两遍法，ddof=1，有效点 <= ddof 给 NaN。
//! · skew：nanskew，mean = Σx(count)/count，m2/m3 = Σ(adjusted²)/Σ(adjusted³)（掩码处置 0），
//!   最后 (count·sqrt(count-1)/(count-2))·(m3/m2^1.5)；|m2|、|m3| < 1e-14 抹成 0，
//!   m2 == 0 直接返回 0.0。count == 2 时除零给出 ±inf（numpy 在这个标量分支里
//!   不会把 count<3 改成 NaN），所以这里也照原样让 f64 除零。
//! · median：去掉 NaN 后排序取中位（偶数取两个中间值的平均）。
//! · 空的序列（该窗口里该条序列一个正值都没有）→ 20 个统计量全 NaN。
//!
//! ════════════════════════════════════════════════════════════════════
//! 五、精度与已知差异
//! ════════════════════════════════════════════════════════════════════
//! · 求和/求均值的累加顺序与 numpy 的成对求和不同，差在 1e-16 相对量级；
//!   四组对账实测 1400 个值全部一致（相对误差远小于 1e-4）。
//! · volume / turnover / order 号只用「是否 > 0」和「首字符」，不做算术，
//!   所以不需要 pandas 的 precise_xstrtod；这里仍按它解析（保证与旧值同源）。
//! · 掩码用 isfinite：inf 也算无效（DataFrame.corr 那一路）；Series.corr 那一路
//!   nancorr 用的是 notna，inf 有效——本文件的四个入参都不会是 inf。

use crate::data_paths;

const N_STATS: usize = 20;
const N_SERIES: usize = 10;
const N_EXTRA: usize = 4;
/// 每个时段 280 个值
const PER_WINDOW: usize = (N_SERIES + N_EXTRA) * N_STATS;
const N_WINDOWS: usize = 5;
/// 总共 1400 个
pub const N_OUT: usize = PER_WINDOW * N_WINDOWS;

/// 20 个统计量的名字，顺序即 names_single 里的顺序。
const STAT_NAMES: [&str; N_STATS] = [
    "correlation",
    "deviation_trend",
    "deviation_trend_abs",
    "deviation_std",
    "deviation_abs_std",
    "deviation_skew",
    "deviation_abs_skew",
    "deviation_max",
    "deviation_abs_max",
    "deviation_min",
    "deviation_abs_min",
    "deviation_abs_mean",
    "deviation_median",
    "deviation_abs_median",
    "deviation_autocorr1",
    "deviation_abs_autocorr1",
    "deviation_autocorr2",
    "deviation_abs_autocorr2",
    "deviation_autocorr3",
    "deviation_abs_autocorr3",
];

/// 10 条序列的名字前缀（顺序 = one() 里 data_dict/ns 的顺序）
const SERIES_NAMES: [&str; N_SERIES] = [
    "volume",
    "turnover",
    "volume66",
    "volume83",
    "bid_order",
    "ask_order",
    "buy_cancels",
    "sell_cancels",
    "buy_cancels_order",
    "sell_cancels_order",
];

/// 4 组按位相减/取绝对值的名字前缀
const EXTRA_NAMES: [&str; N_EXTRA] = [
    "ask&bid_order_diff",
    "ask&bid_order_diff_abs",
    "buy_cancels&sell_cancels_diff",
    "buy_cancels&sell_cancels_diff_abs",
];

/// 时段后缀：第一个是 `all_` 前缀，其余是 `_xxx` 后缀。
const WINDOW_TAGS: [&str; N_WINDOWS] = ["all", "_open30", "_morning", "_afternoon", "_close30"];

pub fn names() -> Vec<String> {
    let mut base: Vec<String> = Vec::with_capacity((N_SERIES + N_EXTRA) * N_STATS);
    for s in SERIES_NAMES.iter() {
        for st in STAT_NAMES.iter() {
            base.push(format!("{}_{}", s, st));
        }
    }
    for e in EXTRA_NAMES.iter() {
        for st in STAT_NAMES.iter() {
            base.push(format!("{}_{}", e, st));
        }
    }
    let mut out = Vec::with_capacity(N_OUT);
    for (w, tag) in WINDOW_TAGS.iter().enumerate() {
        for b in base.iter() {
            if w == 0 {
                out.push(format!("all_{}", b));
            } else {
                out.push(format!("{}{}", b, tag));
            }
        }
    }
    out
}

// ════════════════════════════════════════════════════════════════════
// 时间与窗口常量（单位：纳秒 / 微秒）
// ════════════════════════════════════════════════════════════════════

const DAY_NS: i64 = 86_400_000_000_000;
/// 北京时区偏移 8 小时
const OFFSET_NS: i64 = 8 * 3600 * 1_000_000_000;
/// 下午整体前移 90 分钟
const SHIFT_NS: i64 = 90 * 60 * 1_000_000_000;

const US: i64 = 1_000;
const SEC_US: i64 = 1_000_000;
const D_0930: i64 = (9 * 3600 + 30 * 60) * SEC_US;
const D_1000: i64 = 10 * 3600 * SEC_US;
const D_1130: i64 = (11 * 3600 + 30 * 60) * SEC_US;
/// 调整后的 13:00:00 = 真实 14:30:00
const D_1300: i64 = 13 * 3600 * SEC_US;
/// 调整后的 13:27:00 = 真实 14:57:00
const D_1327: i64 = (13 * 3600 + 27 * 60) * SEC_US;

/// 窗口掩码（一行可能同时属于多个时段）
const W_DAY: u8 = 1;
const W_OPEN30: u8 = 2;
const W_MORNING: u8 = 4;
const W_AFTERNOON: u8 = 8;
const W_CLOSE30: u8 = 16;

/// pandas 的「epoch 微秒（f64）→ 纳秒（i64）」（conversion.pyx cast_from_unit_vectorized）。
///
/// 注意入参 `val` 是 **`exchtime_us / 1e6`（秒）**，不是微秒：pandas 那条链是
/// `pd.to_timedelta(df.exchtime / 1e6, unit="s")`，先做 f64 除法再看小数部分，
/// 直接把微秒整数乘 1e9 会得到一个完全不同的纳秒值（而且是溢出后的垃圾）。
/// val = us/1e6；base = 向零截断；frac = rint(frac*1e9)/1e9；ns = base*1e9 + (i64)(frac*1e9)。
#[inline]
fn pandas_ns(val: f64) -> i64 {
    let base = val as i64;
    let frac = val - base as f64;
    let frac = (frac * 1e9).round_ties_even() / 1e9;
    base * 1_000_000_000 + (frac * 1e9) as i64
}

// ════════════════════════════════════════════════════════════════════
// 读盘与切窗口
// ════════════════════════════════════════════════════════════════════

/// 调整后的一行：adjusted 当日微秒 + 窗口掩码 + one() 用到的 6 个字段。
struct Row {
    tod: i64,
    mask: u8,
    price: f64,
    volume: f64,
    turnover: f64,
    flag: i64,
    bid: i64,
    ask: i64,
}

/// 读一个交易日的逐笔，复刻 `read_trade_data_single(with_retreat=1)` + `adjust_afternoon`。
///
/// 三件事都不借 fast_csv_reader：
///  · 撤单：这一系列 with_retreat=1，**保留** flag==32（后面要用），所以不能过滤；
///  · 窗口：between_time 是微秒级闭区间，且带 pandas 的纳秒噪声（见文件头第二节）；
///  · 精度：exchtime 要按 f64 走 1e6 除法再截断，整数路径会差。
fn read_rows(code: &str, date: i64) -> Option<Vec<Row>> {
    const COL_EXCHTIME: usize = 4;
    const COL_PRICE: usize = 7;
    const COL_VOLUME: usize = 8;
    const COL_TURNOVER: usize = 9;
    const COL_FLAG: usize = 10;
    const COL_ASK_ORDER: usize = 13;
    const COL_BID_ORDER: usize = 14;

    let filename = format!("{}_{}_transaction.csv", code, date);
    let mut data = None;
    for root in data_paths::level2_roots() {
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

    let mut rows: Vec<Row> = Vec::new();
    for (lineno, line) in data.split(|&b| b == b'\n').enumerate() {
        if lineno == 0 || line.is_empty() {
            continue; // 表头 / 空行
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
            continue; // 列数不够，pandas 也读不出这一行
        }

        let exch_us = us_prefix(fields[COL_EXCHTIME]);
        if exch_us <= 0 {
            continue; // 空的 exchtime 在 pandas 里是 NaT，between_time 一定选不上
        }
        // ns = pandas 的 Timestamp 值（含 8 小时偏移）；先做 exchtime_us / 1e6 的 f64 除法
        let ns = pandas_ns(exch_us as f64 / 1e6) + OFFSET_NS;
        let tod = ns.rem_euclid(DAY_NS) / US;
        // adjust_afternoon：上午段原样、下午段前移 90 分钟，窗口外的行整行丢掉
        let adj_tod = if (D_0930..=D_1130).contains(&tod) {
            tod
        } else if ((13 * 3600) * SEC_US..=(14 * 3600 + 57 * 60) * SEC_US).contains(&tod) {
            tod - SHIFT_NS / US
        } else {
            continue;
        };
        // 五个时段都是在**调整后**的时刻上再切一次闭区间（morning 的上界与
        // afternoon 的下界都是 11:30，所以真实 13:00:00 那一行同时属于这两段）
        let mut mask = W_DAY;
        if (D_0930..=D_1000).contains(&adj_tod) {
            mask |= W_OPEN30;
        }
        if (D_0930..=D_1130).contains(&adj_tod) {
            mask |= W_MORNING;
        }
        if (D_1130..=D_1327).contains(&adj_tod) {
            mask |= W_AFTERNOON;
        }
        if (D_1300..=D_1327).contains(&adj_tod) {
            mask |= W_CLOSE30;
        }

        rows.push(Row {
            tod: adj_tod,
            mask,
            price: num(fields[COL_PRICE]),
            volume: num(fields[COL_VOLUME]),
            turnover: num(fields[COL_TURNOVER]),
            flag: int_prefix(fields[COL_FLAG]),
            bid: int_prefix(fields[COL_BID_ORDER]),
            ask: int_prefix(fields[COL_ASK_ORDER]),
        });
    }
    Some(rows)
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

/// 取字段的整数前缀。`"32.0"` 给 32，与旧框架 `df.flag == 32` 的数值比较一致。
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

/// exchtime 列（形如 `1740964500130000.0`）：整数前缀就是微秒数。
fn us_prefix(raw: &[u8]) -> i64 {
    let b = trim_ascii(raw);
    if !b.contains(&b'e') && !b.contains(&b'E') {
        return int_prefix(b);
    }
    std::str::from_utf8(b)
        .ok()
        .and_then(|s| s.parse::<f64>().ok())
        .map(|v| v.round() as i64)
        .unwrap_or(0)
}

/// 小数转 f64，照抄 pandas 的 C 解析器；空字段给 NaN（pandas 也把空字段读成 NaN）。
fn num(raw: &[u8]) -> f64 {
    let b = trim_ascii(raw);
    if b.is_empty() {
        return f64::NAN;
    }
    parse_f64_pandas(b)
}

/// 10 的幂表（与 pandas 的 parse 用同一套）。
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

/// pandas C 解析器 `precise_xstrtod` 的逐行复刻（与 hm033.rs / hm090.rs 里的同名函数一致，
/// 这里复制一份免得跨系列互相依赖）。这一系列不做量级算术，但为了与旧值同源仍照抄。
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

// ════════════════════════════════════════════════════════════════════
// 首字符
// ════════════════════════════════════════════════════════════════════

/// `str(x)[0]`：见文件头第三节。
#[inline]
fn first_char(x: f64) -> u8 {
    if !x.is_finite() {
        // 'inf' / 'nan' 的首字符，都不在 '1'..'9' 里
        return b'i';
    }
    if x >= 1.0e16 || x < 0.0001 {
        // Python/numpy 在这两端用科学计数法，首字符 = 首位有效数字
        let s = format!("{:e}", x);
        return s.as_bytes()[0];
    }
    if x >= 1.0 {
        // 定点写法，首字符 = 整数部分首位（x < 1e16 < 2^53，as u64 是精确截断）
        let n = x as u64;
        let k = n.ilog10();
        return b'0' + (n / 10u64.pow(k)) as u8;
    }
    // 1e-4 <= x < 1：'0.001' 这种定点写法的首字符就是 '0'
    b'0'
}

// ════════════════════════════════════════════════════════════════════
// 统计量（全部按 NaN 语义，掩码处不参与）
// ════════════════════════════════════════════════════════════════════

#[inline]
fn valid_count(x: &[f64]) -> usize {
    x.iter().filter(|v| !v.is_nan()).count()
}

/// nanmean：pandas 是「掩码处置 0 后整体求和 / 有效个数」。
fn nanmean(x: &[f64]) -> f64 {
    let c = valid_count(x);
    if c == 0 {
        return f64::NAN;
    }
    let s: f64 = x.iter().map(|v| if v.is_nan() { 0.0 } else { *v }).sum();
    s / c as f64
}

/// nanvar：两遍法，ddof 默认 1；有效点 <= ddof 给 NaN。
fn nanvar(x: &[f64], ddof: usize) -> f64 {
    let c = valid_count(x);
    if c <= ddof {
        return f64::NAN;
    }
    let avg = nanmean(x);
    let mut s = 0.0f64;
    for v in x.iter() {
        if !v.is_nan() {
            let d = avg - *v;
            s += d * d;
        }
    }
    s / (c - ddof) as f64
}

fn nanstd(x: &[f64], ddof: usize) -> f64 {
    nanvar(x, ddof).sqrt()
}

/// nanskew：调整 Fisher-Pearson G1（照抄 pandas/core/nanops.py 的 nanskew）。
fn nanskew(x: &[f64]) -> f64 {
    let c = valid_count(x);
    let mean = if c == 0 {
        f64::NAN
    } else {
        let s: f64 = x.iter().map(|v| if v.is_nan() { 0.0 } else { *v }).sum();
        s / c as f64
    };
    let mut m2 = 0.0f64;
    let mut m3 = 0.0f64;
    for v in x.iter() {
        // 掩码处：pandas 是 (0 - mean) 再 putmask 成 0，等价于直接取 0
        let adj = if v.is_nan() { 0.0 } else { *v - mean };
        m2 += adj * adj;
        m3 += adj * adj * adj;
    }
    if m2.abs() < 1e-14 {
        m2 = 0.0;
    }
    if m3.abs() < 1e-14 {
        m3 = 0.0;
    }
    if m2 == 0.0 {
        return 0.0; // pandas：标量分支 `dtype.type(0) if m2 == 0 else result`
    }
    let cnt = c as f64;
    (cnt * (cnt - 1.0).sqrt() / (cnt - 2.0)) * (m3 / m2.powf(1.5))
}

fn nanmax(x: &[f64]) -> f64 {
    let mut m = f64::NAN;
    for v in x.iter() {
        if !v.is_nan() && (m.is_nan() || *v > m) {
            m = *v;
        }
    }
    m
}

fn nanmin(x: &[f64]) -> f64 {
    let mut m = f64::NAN;
    for v in x.iter() {
        if !v.is_nan() && (m.is_nan() || *v < m) {
            m = *v;
        }
    }
    m
}

/// nanmedian：去掉 NaN 后排序取中位（偶数取两个中间值的平均）。
fn nanmedian(x: &[f64]) -> f64 {
    let mut v: Vec<f64> = x.iter().cloned().filter(|z| !z.is_nan()).collect();
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    }
}

/// Series.corr → nanops.nancorr → **np.corrcoef**（两遍法），成对丢 NaN。
fn corrcoef(x: &[f64], y: &[f64]) -> f64 {
    let mut xs: Vec<f64> = Vec::new();
    let mut ys: Vec<f64> = Vec::new();
    for i in 0..x.len() {
        if !x[i].is_nan() && !y[i].is_nan() {
            xs.push(x[i]);
            ys.push(y[i]);
        }
    }
    let n = xs.len();
    if n < 2 {
        return f64::NAN;
    }
    let nf = n as f64;
    let mx = xs.iter().sum::<f64>() / nf;
    let my = ys.iter().sum::<f64>() / nf;
    let mut sxx = 0.0f64;
    let mut syy = 0.0f64;
    let mut sxy = 0.0f64;
    for i in 0..n {
        let dx = xs[i] - mx;
        let dy = ys[i] - my;
        sxx += dx * dx;
        syy += dy * dy;
        sxy += dx * dy;
    }
    if sxx <= 0.0 || syy <= 0.0 {
        return f64::NAN;
    }
    sxy / (sxx.sqrt() * syy.sqrt())
}

/// DataFrame.corr → libalgos.nancorr 的 **Welford** 版（掩码用 isfinite）。
fn corr_welford(x: &[f64], y: &[f64]) -> f64 {
    let mut nobs: i64 = 0;
    let mut meanx = 0.0f64;
    let mut meany = 0.0f64;
    let mut ssqdmx = 0.0f64;
    let mut ssqdmy = 0.0f64;
    let mut covxy = 0.0f64;
    for i in 0..x.len() {
        if x[i].is_finite() && y[i].is_finite() {
            let vx = x[i];
            let vy = y[i];
            nobs += 1;
            let dx = vx - meanx;
            let dy = vy - meany;
            meanx += 1.0 / nobs as f64 * dx;
            meany += 1.0 / nobs as f64 * dy;
            ssqdmx += (vx - meanx) * dx;
            ssqdmy += (vy - meany) * dy;
            covxy += (vx - meanx) * dy;
        }
    }
    let divisor = (ssqdmx * ssqdmy).sqrt();
    if divisor != 0.0 {
        covxy / divisor
    } else {
        f64::NAN
    }
}

/// `dev.autocorr(lag)` = `dev.corr(dev.shift(lag))`，成对丢掉 NaN。
fn autocorr(dev: &[f64], lag: usize) -> f64 {
    let n = dev.len();
    let x: Vec<f64> = dev[lag..].to_vec();
    let y: Vec<f64> = dev[..n - lag].to_vec();
    corrcoef(&x, &y)
}

/// 一条序列在一个时段里的 20 个统计量。
fn series_stats(vals: &[f64]) -> [f64; N_STATS] {
    let mut out = [f64::NAN; N_STATS];

    // observed：先数首字符直方图，再除以全部正值个数
    let mut total = 0usize;
    let mut hits = [0usize; 9];
    for &v in vals.iter() {
        if v > 0.0 {
            total += 1;
            let c = first_char(v);
            if (b'1'..=b'9').contains(&c) {
                hits[(c - b'1') as usize] += 1;
            }
        }
    }
    if total == 0 {
        return out; // 旧代码：len(first_digits.dropna()) == 0 → [nan]*20
    }

    let mut obs = [f64::NAN; 9];
    for k in 0..9 {
        if hits[k] > 0 {
            obs[k] = hits[k] as f64 / total as f64;
        }
    }
    let theo: [f64; 9] = std::array::from_fn(|k| (1.0 + 1.0 / (k as f64 + 1.0)).log10());
    let mut dev = [0.0f64; 9];
    let mut devabs = [0.0f64; 9];
    for k in 0..9 {
        dev[k] = obs[k] - theo[k];
        devabs[k] = dev[k].abs();
    }
    let xs: [f64; 9] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];

    out[0] = corrcoef(&obs, &theo);
    out[1] = corr_welford(&xs, &dev);
    out[2] = corr_welford(&xs, &devabs);
    out[3] = nanstd(&dev, 1);
    out[4] = nanstd(&devabs, 1);
    out[5] = nanskew(&dev);
    out[6] = nanskew(&devabs);
    out[7] = nanmax(&dev);
    out[8] = nanmax(&devabs);
    out[9] = nanmin(&dev);
    out[10] = nanmin(&devabs);
    out[11] = nanmean(&devabs);
    out[12] = nanmedian(&dev);
    out[13] = nanmedian(&devabs);
    out[14] = autocorr(&dev, 1);
    out[15] = autocorr(&devabs, 1);
    out[16] = autocorr(&dev, 2);
    out[17] = autocorr(&devabs, 2);
    out[18] = autocorr(&dev, 3);
    out[19] = autocorr(&devabs, 3);
    out
}

// ════════════════════════════════════════════════════════════════════
// 入口
// ════════════════════════════════════════════════════════════════════

/// 取第 `which` 条序列（顺序见 SERIES_NAMES）在某个时段里的原始值。
fn collect(rows: &[Row], mask: u8, which: usize, out: &mut Vec<f64>) {
    out.clear();
    for r in rows.iter() {
        if r.mask & mask == 0 {
            continue;
        }
        match which {
            0 => {
                if r.price != 0.0 {
                    out.push(r.volume)
                }
            }
            1 => {
                if r.price != 0.0 {
                    out.push(r.turnover)
                }
            }
            2 => {
                if r.flag == 66 && r.price != 0.0 {
                    out.push(r.volume)
                }
            }
            3 => {
                if r.flag == 83 && r.price != 0.0 {
                    out.push(r.volume)
                }
            }
            4 => {
                if r.price != 0.0 {
                    out.push(r.bid as f64)
                }
            }
            5 => {
                if r.price != 0.0 {
                    out.push(r.ask as f64)
                }
            }
            6 => {
                if r.flag == 32 && r.bid > 0 {
                    out.push(r.volume)
                }
            }
            7 => {
                if r.flag == 32 && r.ask > 0 {
                    out.push(r.volume)
                }
            }
            8 => {
                if r.flag == 32 && r.bid > 0 {
                    out.push(r.bid as f64)
                }
            }
            _ => {
                if r.flag == 32 && r.ask > 0 {
                    out.push(r.ask as f64)
                }
            }
        }
    }
}

/// 逐股入口：算不出来返回 None（上游会补 NaN，保证因子个数恒定）。
pub fn compute_stock(code: &str, date: i64) -> Option<Vec<f32>> {
    let rows = read_rows(code, date)?;
    let mut out = Vec::with_capacity(N_OUT);
    let mut buf: Vec<f64> = Vec::new();
    // 五个时段：all / open30 / morning / afternoon / close30
    for mask in [
        W_DAY,
        W_OPEN30,
        W_MORNING,
        W_AFTERNOON,
        W_CLOSE30,
    ] {
        let mut per = [[f64::NAN; N_STATS]; N_SERIES];
        for s in 0..N_SERIES {
            collect(&rows, mask, s, &mut buf);
            per[s] = series_stats(&buf);
        }
        for s in 0..N_SERIES {
            out.extend(per[s].iter().map(|&v| v as f32));
        }
        // ask&bid_order_diff（买单编号 - 卖单编号）
        for k in 0..N_STATS {
            out.push((per[4][k] - per[5][k]) as f32);
        }
        for k in 0..N_STATS {
            out.push((per[4][k] - per[5][k]).abs() as f32);
        }
        // buy_cancels&sell_cancels_diff（买单撤单编号 - 卖单撤单编号）
        for k in 0..N_STATS {
            out.push((per[8][k] - per[9][k]) as f32);
        }
        for k in 0..N_STATS {
            out.push((per[8][k] - per[9][k]).abs() as f32);
        }
    }
    debug_assert_eq!(out.len(), N_OUT);
    if out.len() != N_OUT {
        return None;
    }
    Some(out)
}

/// hm69 全是逐股因子，没有横截面版本。
pub fn compute_cross(_date: i64) -> Option<(Vec<String>, Vec<f32>)> {
    None
}
