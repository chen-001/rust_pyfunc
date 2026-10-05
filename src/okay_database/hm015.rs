//! hm15 收盘分歧 —— 旧框架（design_whatever_out.TickDataBase）的 Rust 复现。
//!
//! ════════════════════════════════════════════════════════════════════════
//! 一、旧代码的调用参数（/home/chenzongwei/pythoncode/成果hm15_成果hm16_收盘分歧/hm15_okay.py）
//! ════════════════════════════════════════════════════════════════════════
//!   run_factor_out(..., get_many=0, read_full=0, change_time=1, use_old=True)
//!   → read_trade_data_single(with_retreat=0) 走 `get_many==0 && read_full==0` 那条分支：
//!     只做 deal_data（把 exchtime 微秒转成 1970-01-01 的 Timestamp），返回**单个** DataFrame，
//!     不做任何时段切分，也不做 adjust_afternoon。所以 Rust 侧用原始钟点自己切窗口。
//!   → with_retreat=0 丢掉 flag==32 的撤单行（数值比较，32 与 "32.0" 都吃）。
//!
//! ════════════════════════════════════════════════════════════════════════
//! 二、get_factor() 的算法（逐行对齐旧代码）
//! ════════════════════════════════════════════════════════════════════════
//!   1. df = between_time("09:30:00", "14:57:00")        真实钟点、微秒闭区间
//!   2. df1 = between_time("09:30:00", "11:30:00")       上午段
//!      df2 = between_time("13:00:00", "14:57:00")       下午段，index 前移 90 分钟
//!      df  = concat([df1, df2])                         拼成 09:30 ~ 13:27 的连续序列
//!   3. df.flag    = np.sign(70 - df.flag)                flag<70→+1，>70→-1，==70→0
//!      df.turnover = df.turnover * df.flag
//!      df.turnover = df.turnover - df.turnover.min()     **整段共用一个最小值**
//!   4. 按 1 分钟分箱（resample("1T")，左闭左标签，label 是分钟起点）：
//!        price = 该分钟**最后一笔**的 price（last 跳过空值，本数据无空值）
//!        std   = 该分钟 turnover 的样本标准差（ddof=1，样本 < 2 为 NaN）
//!        mean  = 该分钟 turnover 的均值（样本 0 为 NaN）
//!      disag = std / mean
//!   5. df1 = concat([price, disag]).dropna()
//!        → 只留「样本数 >= 2 且 mean != 0」的分钟；mean == 0 必然 std == 0，0/0 是 NaN
//!   6. df2 = concat([price.shift(1), disag]).dropna()
//!        price.shift(1) 是**位置**平移，移的是 dropna 之后那一串分钟，
//!        所以 lag1 取的是「上一个有成交的分钟」的收盘价，不是自然上一分钟。
//!        位置 0 的那一行被 dropna 丢掉。
//!   7. 五个窗口（df 的 index 已经是前移后的时刻）分别取 df1 / df2 两列：
//!        all       09:30 ~ 13:27
//!        open30    09:30 ~ 10:00
//!        morning   09:30 ~ 11:30
//!        afternoon 11:30 ~ 13:27   ← 对应真实下午 13:00 ~ 14:57
//!        close30   13:00 ~ 13:27   ← 对应真实 14:30 ~ 14:57
//!      **df1 与 df2 交替**：all_lag0, all_lag1, open30_lag0, open30_lag1, ...，
//!      每个时段 8 个指标，五个时段共 80 个值。
//!   8. 每个 (窗口, lag) 组合上算 8 个指标（旧代码的 one(df)）：
//!        corr_price_disag = df.price 与 df.disag 的皮尔逊相关（成对完整观测）
//!        trends_mean      = 同一价格分组后，各组的 (行号 vs disag) 相关系数的均值
//!        trends_skew      = 上者的偏度（pandas 的无偏口径）
//!        dura_mean/skew/median/min = 同一价格分组内「disag 最小那一分钟」到
//!                           「disag 最大那一分钟」的秒数差的四个统计量
//!        ret_std_adj      = (price[i]/price[i-1] - 1)/disag[i] 的标准差（ddof=1）
//!      分组只保留「出现次数 > 2」（也就是 >= 3 次）的价格；一个都不剩时 8 个值全 NaN。
//!      这个窗口一行都没有时 8 个值也全 NaN。
//!
//!   names_save（旧代码要存进 base 的名单，15 个）见 hm15_okay.py 的 names_save。
//!
//! ════════════════════════════════════════════════════════════════════════
//! 三、几个必须写清楚的坑
//! ════════════════════════════════════════════════════════════════════════
//! 【时间边界用微秒】pandas 的 between_time 是闭区间，精度到纳秒。窗口端点
//!   09:30:00.000000 / 11:30:00.000000 / 13:00:00.000000 / 14:57:00.000000 都算在内，
//!   14:57:00.000001 必须排除。这里 day_us（当天微秒）直接和这些整数端点比，不做整秒截断。
//!
//! 【price / turnover 必须用 f64 读】read_trade_fast_inner 的 TradeRecord 里
//!   price/turnover 是 f32。旧代码第 3 步要做 `turnover - turnover.min()`，成交额量级
//!   1e7 减到 1e1 量级会把 f32 的 6e-8 相对误差放大成 1e-3 量级，过不了 verify 的
//!   1e-4 相对误差阈值。所以本文件自带一个 f64 版本的逐笔读取，语义与
//!   read_trade_fast_inner(code, date, true, false, usize::MAX) 一致：
//!     不按 with_retreat 过滤（那条按字节比 "32"，"32.0" 不生效），自己按数值过滤 flag != 32；
//!     不做下午平移；没有该股该日文件时返回 None。
//!
//! 【std 用两遍算法】pandas 的 std(ddof=1) 与本文件的「先算均值再求偏差平方和」一致，
//!   用 sum(x^2) - sum(x)^2/n 那种一遍公式在这批数据上会有 1e-14 量级的偏差（实测），
//!   虽然离阈值还远，仍按 pandas 的口径走两遍。
//!
//! 【偏度是无偏口径】pandas 2.3 的 Series.skew() 是 g1 * sqrt(n(n-1))/(n-2)，
//!   不是 m3/m2^1.5；n < 3 或 m2 <= 0 时返回 NaN。
//!
//! 【分组的确定性】价格分组用 HashMap 计数，但所有需要遍历的地方都先按价格升序排好
//!   再算，保证同一输入每次结果一致。分组内按 disag 排序用稳定排序，并列时保持时间先后。
//!
//! 【恒定的因子个数】算不出来返回 None（上游补一列 NaN），绝不条件跳过、绝不少输出几列。
//!   任何非有限值（inf/-inf）一律写成 NaN，与旧框架落盘时的 replace(inf, nan) 口径一致。

use std::collections::HashMap;
use std::fs;

use crate::data_paths;

// transaction.csv 的列下标（0-based，与文件头一致）：
//   symbol,datetime,date,time,exchtime,localtime,index,price,volume,turnover,flag,
//   order_type,func_code,ask_order,bid_order
const COL_EXCHTIME: usize = 4;
const COL_PRICE: usize = 7;
const COL_TURNOVER: usize = 9;
const COL_FLAG: usize = 10;

const DAY_US: i64 = 86_400_000_000;
const MIN_US: i64 = 60_000_000;
/// 下午段前移 90 分钟
const SHIFT_US: i64 = 90 * 60 * 1_000_000;
/// 交易所时间戳的 8 小时偏移（CSV 里的 exchtime 是 epoch 微秒 UTC）
const OFFSET_US: i64 = 8 * 3600 * 1_000_000;

/// 窗口边界（当天零点起的微秒，闭区间两端）
const OPEN_US: i64 = (9 * 3600 + 30 * 60) * 1_000_000; // 09:30:00
const AM_END_US: i64 = (11 * 3600 + 30 * 60) * 1_000_000; // 11:30:00
const PM_START_US: i64 = 13 * 3600 * 1_000_000; // 13:00:00
const LAST_US: i64 = (14 * 3600 + 57 * 60) * 1_000_000; // 14:57:00

/// 一天最多 1440 个分钟箱
const NMIN: usize = 1440;

/// 每个 (窗口, lag) 8 个指标，五个窗口 × 2 个 lag = 80
const PER_SESSION: usize = 8;
pub const N_OUT: usize = PER_SESSION * 5 * 2;

const METRIC_NAMES: [&str; PER_SESSION] = [
    "corr_price_disag",
    "trends_mean",
    "trends_skew",
    "dura_mean",
    "dura_skew",
    "dura_median",
    "dura_min",
    "ret_std_adj",
];

/// 五个窗口按旧代码的拼接顺序；区间是「前移后」的分钟数，闭区间
const SESSIONS: [(&str, i64, i64); 5] = [
    ("all", 570, 807),       // 09:30 ~ 13:27
    ("open30", 570, 600),    // 09:30 ~ 10:00
    ("morning", 570, 690),   // 09:30 ~ 11:30
    ("afternoon", 690, 807), // 11:30 ~ 13:27（真实下午）
    ("close30", 780, 807),   // 13:00 ~ 13:27（真实 14:30~14:57）
];

pub fn names() -> Vec<String> {
    let mut out = Vec::with_capacity(N_OUT);
    for (sess, _, _) in SESSIONS.iter() {
        for lag in 0..2 {
            for m in METRIC_NAMES.iter() {
                out.push(format!("{}_lag{}_{}", sess, lag, m));
            }
        }
    }
    out
}

/// 一行逐笔（只用得到旧代码真正参与计算的四列）
#[derive(Clone, Copy)]
struct RawRow {
    /// epoch 微秒，已加 8 小时（当天的北京钟点 = rem_euclid(DAY_US)）
    time_us: i64,
    price: f64,
    turnover: f64,
    flag: i32,
}

fn parse_f64(bytes: &[u8]) -> f64 {
    // 与 pandas 的 read_csv 一致：都用正确舍入的十进制解析；空串/非数字 → NaN
    std::str::from_utf8(bytes)
        .ok()
        .and_then(|s| s.trim().parse::<f64>().ok())
        .unwrap_or(f64::NAN)
}

/// 拆一行成 15 个字段（不足的补空切片）
#[inline]
fn split_fields(line: &[u8]) -> [&[u8]; 15] {
    let mut fields: [&[u8]; 15] = [&[][..]; 15];
    let mut start = 0usize;
    let mut col = 0usize;
    for (i, &b) in line.iter().enumerate() {
        if b == b',' {
            if col < 15 {
                fields[col] = &line[start..i];
            }
            col += 1;
            start = i + 1;
        }
    }
    if col < 15 {
        fields[col] = &line[start..];
    }
    fields
}

/// 解析一行；返回 None 表示这行不进计算（表头、空行、字段缺失、撤单行）
#[inline]
fn parse_line(line: &[u8]) -> Option<RawRow> {
    if line.is_empty() {
        return None;
    }
    let fields = split_fields(line);
    let exchtime = parse_f64(fields[COL_EXCHTIME]);
    let price = parse_f64(fields[COL_PRICE]);
    let turnover = parse_f64(fields[COL_TURNOVER]);
    let flag = parse_f64(fields[COL_FLAG]);
    if !exchtime.is_finite() || !price.is_finite() || !turnover.is_finite() || !flag.is_finite() {
        return None;
    }
    // CSV 里的 exchtime 是 "1740964500130000.0" 这种浮点写法，取整到微秒
    let exchtime_us = exchtime.round() as i64;
    if exchtime_us == 0 {
        return None;
    }
    let flag = flag.round() as i32;
    // with_retreat=0：数值比较，能吃下 "32" 和 "32.0" 两种写法
    if flag == 32 {
        return None;
    }
    Some(RawRow {
        time_us: exchtime_us + OFFSET_US,
        price,
        turnover,
        flag,
    })
}

/// 自己读逐笔成交，price/turnover 保持 f64（TradeRecord 的 f32 会过不了对账）。
/// 语义与 read_trade_fast_inner(code, date, true, false, usize::MAX) 一致，且保持文件行序。
fn read_trade_f64(code: &str, date: i64) -> Option<Vec<RawRow>> {
    let filename = format!("{}_{}_transaction.csv", code, date);
    let mut path = None;
    for root in data_paths::level2_roots() {
        let p = root.join(date.to_string()).join("transaction").join(&filename);
        if p.exists() {
            path = Some(p);
            break;
        }
    }
    let data = fs::read(path?).ok()?;

    let mut out: Vec<RawRow> = Vec::with_capacity(data.len() / 80 + 1);
    let mut start = 0usize;
    for i in 0..data.len() {
        if data[i] == b'\n' {
            let line = &data[start..i];
            let line = if line.last() == Some(&b'\r') {
                &line[..line.len() - 1]
            } else {
                line
            };
            if let Some(r) = parse_line(line) {
                out.push(r);
            }
            start = i + 1;
        }
    }
    if start < data.len() {
        let line = &data[start..];
        let line = if line.last() == Some(&b'\r') {
            &line[..line.len() - 1]
        } else {
            line
        };
        if let Some(r) = parse_line(line) {
            out.push(r);
        }
    }
    Some(out)
}

/// np.sign(70 - flag)：flag<70 → +1，>70 → -1，==70 → 0
#[inline]
fn flag_sign(flag: i32) -> f64 {
    if flag < 70 {
        1.0
    } else if flag > 70 {
        -1.0
    } else {
        0.0
    }
}

/// f64 的相等键（-0.0 与 0.0 归一到同一个键，和 pandas 的哈希口径一致）
#[inline]
fn key_of(v: f64) -> u64 {
    if v == 0.0 {
        0.0f64.to_bits()
    } else {
        v.to_bits()
    }
}

/// 皮尔逊相关（成对完整观测）。样本 < 2、任一侧方差 <= 0 时返回 NaN。
fn pearson(x: &[f64], y: &[f64]) -> f64 {
    let mut n = 0usize;
    let mut sx = 0.0f64;
    let mut sy = 0.0f64;
    let mut vx = 0.0f64;
    let mut vy = 0.0f64;
    let mut cxy = 0.0f64;
    for i in 0..x.len() {
        if x[i].is_nan() || y[i].is_nan() {
            continue;
        }
        n += 1;
        sx += x[i];
        sy += y[i];
    }
    if n < 2 {
        return f64::NAN;
    }
    let nf = n as f64;
    let mx = sx / nf;
    let my = sy / nf;
    for i in 0..x.len() {
        if x[i].is_nan() || y[i].is_nan() {
            continue;
        }
        let dx = x[i] - mx;
        let dy = y[i] - my;
        vx += dx * dx;
        vy += dy * dy;
        cxy += dx * dy;
    }
    if vx <= 0.0 || vy <= 0.0 {
        return f64::NAN;
    }
    cxy / (vx.sqrt() * vy.sqrt())
}

/// Series.mean(skipna=True)
fn mean_skip(x: &[f64]) -> f64 {
    let mut n = 0usize;
    let mut s = 0.0f64;
    for &v in x.iter() {
        if v.is_nan() {
            continue;
        }
        n += 1;
        s += v;
    }
    if n == 0 {
        return f64::NAN;
    }
    s / n as f64
}

/// Series.skew()（pandas 2.3 的无偏口径），skipna=True
fn skew_skip(x: &[f64]) -> f64 {
    let vals: Vec<f64> = x.iter().copied().filter(|v| !v.is_nan()).collect();
    let n = vals.len();
    if n < 3 {
        return f64::NAN;
    }
    let nf = n as f64;
    let m = vals.iter().sum::<f64>() / nf;
    let mut m2 = 0.0f64;
    let mut m3 = 0.0f64;
    for &v in vals.iter() {
        let d = v - m;
        m2 += d * d;
        m3 += d * d * d;
    }
    m2 /= nf;
    m3 /= nf;
    if m2 <= 0.0 {
        return f64::NAN;
    }
    m3 / m2.powf(1.5) * (nf * (nf - 1.0)).sqrt() / (nf - 2.0)
}

/// Series.median()（线性插值），skipna=True
fn median_skip(x: &[f64]) -> f64 {
    let mut vals: Vec<f64> = x.iter().copied().filter(|v| !v.is_nan()).collect();
    let n = vals.len();
    if n == 0 {
        return f64::NAN;
    }
    vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    if n % 2 == 1 {
        vals[n / 2]
    } else {
        (vals[n / 2 - 1] + vals[n / 2]) / 2.0
    }
}

/// Series.min(skipna=True)
fn min_skip(x: &[f64]) -> f64 {
    let mut best = f64::NAN;
    for &v in x.iter() {
        if v.is_nan() {
            continue;
        }
        if best.is_nan() || v < best {
            best = v;
        }
    }
    best
}

/// Series.std(ddof=1, skipna=True)
fn std_skip(x: &[f64]) -> f64 {
    let n = x.iter().filter(|v| !v.is_nan()).count();
    if n < 2 {
        return f64::NAN;
    }
    let nf = n as f64;
    let m = x.iter().filter(|v| !v.is_nan()).sum::<f64>() / nf;
    let mut s = 0.0f64;
    for &v in x.iter() {
        if v.is_nan() {
            continue;
        }
        let d = v - m;
        s += d * d;
    }
    (s / (nf - 1.0)).sqrt()
}

/// np.corrcoef(range(n), x)[0,1]
fn trend_cal(x: &[f64]) -> f64 {
    let n = x.len();
    if n < 2 {
        return f64::NAN;
    }
    let nums: Vec<f64> = (0..n).map(|i| i as f64).collect();
    pearson(&nums, x)
}

/// 非有限值（inf/-inf）写成 NaN，与旧框架落盘时的 replace(inf, nan) 一致
#[inline]
fn finite_or_nan(v: f64) -> f64 {
    if v.is_finite() {
        v
    } else {
        f64::NAN
    }
}

/// 旧代码的 one(df)：minutes 是分钟序号（升序、唯一），price/disag 与之一一对应。
/// 返回 8 个指标，顺序与 METRIC_NAMES 一致。
fn session_metrics(minutes: &[i64], price: &[f64], disag: &[f64]) -> [f64; PER_SESSION] {
    let nan8 = [f64::NAN; PER_SESSION];
    let n = minutes.len();
    if n == 0 {
        return nan8;
    }
    let corr = pearson(price, disag);

    // 每个价格出现的次数（价格用 f64 位模式做键，精确相等）
    let mut cnt: HashMap<u64, u32> = HashMap::new();
    for i in 0..n {
        *cnt.entry(key_of(price[i])).or_insert(0) += 1;
    }

    // 只留出现次数 > 2 的价格；按行号（时间）顺序收集
    let mut gid_of: HashMap<u64, usize> = HashMap::new();
    let mut gid_price: Vec<f64> = Vec::new();
    let mut g_rows: Vec<Vec<usize>> = Vec::new();
    for i in 0..n {
        let k = key_of(price[i]);
        if cnt.get(&k).copied().unwrap_or(0) <= 2 {
            continue;
        }
        let g = match gid_of.get(&k) {
            Some(&g) => g,
            None => {
                let g = gid_price.len();
                gid_of.insert(k, g);
                gid_price.push(price[i]);
                g_rows.push(Vec::new());
                g
            }
        };
        g_rows[g].push(i);
    }
    if gid_price.is_empty() {
        // 旧代码 else 分支：8 个值全 NaN（连 corr 也给 NaN）
        return nan8;
    }

    // 遍历顺序固定：按价格升序
    let mut order: Vec<usize> = (0..gid_price.len()).collect();
    order.sort_by(|&a, &b| {
        gid_price[a]
            .partial_cmp(&gid_price[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // trends：组内 (行号 vs disag) 的相关系数
    let mut trends: Vec<f64> = Vec::with_capacity(order.len());
    for &g in order.iter() {
        let vals: Vec<f64> = g_rows[g].iter().map(|&i| disag[i]).collect();
        trends.push(trend_cal(&vals));
    }

    // dura：组内按 disag 升序排（稳定，并列保持时间先后），首末分钟的秒数差
    let mut duras: Vec<f64> = Vec::with_capacity(order.len());
    for &g in order.iter() {
        let rows = &mut g_rows[g];
        rows.sort_by(|&a, &b| {
            disag[a]
                .partial_cmp(&disag[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let lo = minutes[rows[0]];
        let hi = minutes[rows[rows.len() - 1]];
        duras.push((hi - lo) as f64 * 60.0);
    }

    // ret_std_adj：(price[i]/price[i-1] - 1)/disag[i] 的标准差
    let mut rets: Vec<f64> = Vec::with_capacity(n.saturating_sub(1));
    for i in 1..n {
        rets.push((price[i] / price[i - 1] - 1.0) / disag[i]);
    }

    [
        corr,
        mean_skip(&trends),
        skew_skip(&trends),
        mean_skip(&duras),
        skew_skip(&duras),
        median_skip(&duras),
        min_skip(&duras),
        std_skip(&rets),
    ]
}

/// 逐股入口：算不出来返回 None（上游会补 NaN，保证因子个数恒定）。
pub fn compute_stock(code: &str, date: i64) -> Option<Vec<f32>> {
    let rows = read_trade_f64(code, date)?;
    if rows.is_empty() {
        return None;
    }

    // ── 第 1 步：真实钟点切 [09:30:00, 14:57:00] ──
    // 旧代码先判断这个窗口非空（空则 get_factor 返回 None），再做后面的切分
    let mut any_sel = false;
    for r in rows.iter() {
        let d = r.time_us.rem_euclid(DAY_US);
        if d >= OPEN_US && d <= LAST_US {
            any_sel = true;
            break;
        }
    }
    if !any_sel {
        return None;
    }

    // ── 第 2 步：上午段 + 下午段（前移 90 分钟），上午在前 ──
    let mut day: Vec<i64> = Vec::with_capacity(rows.len());
    let mut price: Vec<f64> = Vec::with_capacity(rows.len());
    let mut turn: Vec<f64> = Vec::with_capacity(rows.len());
    let mut sgn: Vec<f64> = Vec::with_capacity(rows.len());
    for r in rows.iter() {
        let d = r.time_us.rem_euclid(DAY_US);
        if d >= OPEN_US && d <= AM_END_US {
            day.push(d);
            price.push(r.price);
            turn.push(r.turnover);
            sgn.push(flag_sign(r.flag));
        }
    }
    for r in rows.iter() {
        let d = r.time_us.rem_euclid(DAY_US);
        if d >= PM_START_US && d <= LAST_US {
            day.push(d - SHIFT_US);
            price.push(r.price);
            turn.push(r.turnover);
            sgn.push(flag_sign(r.flag));
        }
    }
    if day.is_empty() {
        // 旧代码此时各窗口都取不到东西，80 个值全 NaN
        return Some(vec![f32::NAN; N_OUT]);
    }

    // ── 第 3 步：符号化成交额，整段减去全局最小值 ──
    let mut adj: Vec<f64> = turn
        .iter()
        .zip(sgn.iter())
        .map(|(&t, &s)| t * s)
        .collect();
    let mut min_t = f64::INFINITY;
    for v in adj.iter() {
        if *v < min_t {
            min_t = *v;
        }
    }
    for v in adj.iter_mut() {
        *v -= min_t;
    }

    // ── 第 4 步：按分钟分箱 ──
    let mut bins: Vec<Vec<f64>> = vec![Vec::new(); NMIN];
    let mut last_price = vec![f64::NAN; NMIN];
    for i in 0..day.len() {
        let b = (day[i] / MIN_US) as usize;
        if b >= NMIN {
            continue;
        }
        bins[b].push(adj[i]);
        last_price[b] = price[i];
    }

    // ── 第 5 步：price / std / mean → disag ──
    // all_min / all_price = 有成交的分钟（旧代码 price.dropna() 之后那一串，
    //   也就是 price.shift(1) 真正作用的那一串，**不是**下面 kept 那一串）；
    // kept = 再经过 concat([price, std/mean]).dropna() 之后剩下的分钟（样本 >= 2）。
    let mut all_min: Vec<i64> = Vec::new();
    let mut all_price: Vec<f64> = Vec::new();
    let mut kept_pos: Vec<usize> = Vec::new();
    let mut kept_dis: Vec<f64> = Vec::new();
    for b in 0..NMIN {
        let v = &bins[b];
        let n = v.len();
        if n == 0 {
            continue;
        }
        let pos = all_min.len();
        all_min.push(b as i64);
        all_price.push(last_price[b]);
        if n < 2 {
            continue;
        }
        let nf = n as f64;
        let mean = v.iter().sum::<f64>() / nf;
        let mut s = 0.0f64;
        for &x in v.iter() {
            let d = x - mean;
            s += d * d;
        }
        let sd = (s / (nf - 1.0)).sqrt();
        let dis = sd / mean;
        // dropna：std/mean 为 NaN（mean == 0 时必然 std == 0，0/0）的分钟丢掉
        if dis.is_nan() || last_price[b].is_nan() {
            continue;
        }
        kept_pos.push(pos);
        kept_dis.push(dis);
    }

    // ── 第 6~8 步：五个窗口 × (lag0, lag1) 各 8 个指标 ──
    let mut out = Vec::with_capacity(N_OUT);
    let m = kept_pos.len();
    let mut buf_min: Vec<i64> = Vec::new();
    let mut buf_price: Vec<f64> = Vec::new();
    let mut buf_dis: Vec<f64> = Vec::new();
    for (_sess, lo, hi) in SESSIONS.iter() {
        for lag in 0..2 {
            buf_min.clear();
            buf_price.clear();
            buf_dis.clear();
            for i in 0..m {
                let p = kept_pos[i];
                let minute = all_min[p];
                if minute < *lo || minute > *hi {
                    continue;
                }
                // lag1 是 price 序列的位置平移一位；整串的第一个位置是 NaN，被 dropna 丢掉
                let pi = if lag == 0 {
                    p
                } else {
                    if p == 0 {
                        continue;
                    }
                    p - 1
                };
                buf_min.push(minute);
                buf_price.push(all_price[pi]);
                buf_dis.push(kept_dis[i]);
            }
            let vals = session_metrics(&buf_min, &buf_price, &buf_dis);
            for v in vals.iter() {
                out.push(finite_or_nan(*v) as f32);
            }
        }
    }
    debug_assert_eq!(out.len(), N_OUT);
    Some(out)
}

/// hm15 全是逐股因子，没有横截面版本。
pub fn compute_cross(_date: i64) -> Option<(Vec<String>, Vec<f32>)> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_order() {
        let n = names();
        assert_eq!(n.len(), N_OUT);
        assert_eq!(n[0], "all_lag0_corr_price_disag");
        assert_eq!(n[7], "all_lag0_ret_std_adj");
        assert_eq!(n[8], "all_lag1_corr_price_disag");
        assert_eq!(n[16], "open30_lag0_corr_price_disag");
        assert_eq!(n[79], "close30_lag1_ret_std_adj");
    }

    #[test]
    fn flag_sign_matches_python() {
        assert_eq!(flag_sign(66), 1.0);
        assert_eq!(flag_sign(83), -1.0);
        assert_eq!(flag_sign(70), 0.0);
    }

    #[test]
    fn retreat_and_parse() {
        // flag==32（2025 年写法 "32.0"）必须被丢掉
        let line = b"000001,2025-03-03 09:30:00.000,20250303,93000000,1740965400000000.0,1740965400000000.0,1,11.45,100,1145,32.0,,,0,1";
        assert!(parse_line(line).is_none());
        let line = b"000001,2025-03-03 09:30:00.000,20250303,93000000,1740965400000000.0,1740965400000000.0,1,11.45,100,1145,66.0,,,0,1";
        let r = parse_line(line).unwrap();
        assert_eq!(r.flag, 66);
        assert_eq!(r.price, 11.45);
        assert!((r.turnover - 1145.0).abs() < 1e-12);
        assert_eq!(r.time_us, 1740965400000000 + OFFSET_US);
    }

    #[test]
    fn skew_is_unbiased() {
        // pandas 2.3: Series([0.1,0.5,-0.3,0.9,0.2,-0.7]).skew() == -0.1306398578424021
        let v = [0.1, 0.5, -0.3, 0.9, 0.2, -0.7];
        let s = skew_skip(&v);
        assert!((s - (-0.1306398578424021)).abs() < 1e-12, "skew={}", s);
    }
}
