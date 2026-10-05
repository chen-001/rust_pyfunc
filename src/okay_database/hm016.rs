//! hm16 收盘分歧 —— 旧框架（design_whatever_out.TickDataBase）的 Rust 复现。
//!
//! ════════════════════════════════════════════════════════════════════════
//! 一、旧代码的调用参数（/home/chenzongwei/pythoncode/成果hm15_成果hm16_收盘分歧/hm16_okay.py）
//! ════════════════════════════════════════════════════════════════════════
//!   run_factor_out(..., get_many=0, read_full=0, change_time=1, use_old=True)
//!   → read_trade_data_single(with_retreat=0) 走 `get_many==0 && read_full==0` 那条分支：
//!     只做 deal_data（把 exchtime 微秒转成 1970-01-01 的 Timestamp），返回**单个** DataFrame，
//!     不做任何时段切分，也不做 adjust_afternoon。所以 Rust 侧用
//!     with_afternoon_adjust=false 读原始钟点，自己按钟点切窗口。
//!   → with_retreat=0 丢掉 flag==32 的撤单行。
//!
//! ════════════════════════════════════════════════════════════════════════
//! 二、get_factor() 的算法（逐行对齐旧代码）
//! ════════════════════════════════════════════════════════════════════════
//!   1. df = df[df.flag != 32]                       撤单行去掉
//!   2. df = between_time("09:30:00", "14:57:00")    真实钟点、微秒闭区间
//!   3. df1 = between_time("09:30:00", "11:30:00")   上午段
//!      df2 = between_time("13:00:00", "14:57:00")   下午段，index 前移 90 分钟
//!      df  = concat([df1, df2])                     拼成一段 09:30~13:27 的连续序列
//!      （上午 11:30 之后、下午 13:00 之前的行被丢掉，因为两段都不含）
//!   4. df.flag    = np.sign(70 - df.flag)           flag<70→+1，>70→-1，==70→0
//!      df.turnover = df.turnover * df.flag
//!      df.turnover = df.turnover - df.turnover.min()   **整段共用一个最小值**
//!   5. df.price = np.log(df.price).diff().replace(0, np.nan)
//!      df = df.dropna()                            第 0 行 diff 是 NaN，必然丢掉
//!   6. 按 1 分钟分箱（resample("1T")，左闭左标签）：
//!        brown_price = price.diff 的 std  (ddof=1，样本<2 为 NaN)
//!        disag       = turnover.std() / turnover.mean()   (std 用 ddof=1)
//!      df4 = concat([brown_price, disag]).dropna()
//!        → 只有「样本数 >= 2 的分钟」留在 df4 里，空分钟/单笔分钟都不算
//!   7. 五个窗口各取 df4 两列的协方差（ddof=1，样本<2 为 NaN）：
//!        all       全部 df4
//!        open30    09:30:00 ~ 10:00:00      （df4 的 index 已经是前移后的时刻）
//!        morning   09:30:00 ~ 11:30:00
//!        afternoon 11:30:00 ~ 13:27:00      ← 注意不是 13:00，前移后这里是真实下午
//!        close30   13:00:00 ~ 13:27:00      ← 对应真实 14:30~14:57
//!   8. 返回 10 个值：5 个协方差 + 5 个绝对值（顺序见 names()）
//!
//!   names_save（旧代码要存进 base 的名单，5 个）：
//!      abs_cov_brown_close30, cov_brown_morning, cov_brown_open30,
//!      cov_brown_afternoon,  cov_brown_close30
//!
//! ════════════════════════════════════════════════════════════════════════
//! 三、两个必须写清楚的坑
//! ════════════════════════════════════════════════════════════════════════
//! 【时间边界用微秒】pandas 的 between_time 是闭区间，精度到纳秒。旧代码的窗口
//!   09:30:00 ~ 14:57:00 里，14:57:00.000001 这种必须排除，按整秒截断会错纳进来。
//!   这里 day_us（当天微秒）直接和 09:30:00.000000 / 11:30:00.000000 / 13:00:00.000000 /
//!   14:57:00.000000 比，不做任何整秒截断。
//!
//!   顺带说明为什么「精确微秒」和 pandas 的判断等价：pandas 把 exchtime 先按
//!   `to_timedelta(exchtime/1e6, unit="s")` 转成纳秒，浮点转换会带来最多 ~114ns 的偏差
//!   （实测 20250303 000001 最大 114ns）。但只有「正好落在整秒上」的逐笔才会贴着窗口边界，
//!   而整秒时刻的小数部分是 0，浮点转换在那里没有偏差（实测这些行的 pandas 时间就等于精确值）。
//!   其余逐笔离整秒至少 1 微秒，远大于 114ns 的偏差，所以两种判断不会分岔。
//!
//! 【price / turnover 必须用 f64 读】read_trade_fast_inner 的 TradeRecord 里
//!   price/turnover 是 f32。旧代码第 4 步要做 `turnover - turnover.min()`，成交量金额量级
//!   1e7 减到 1e1 量级会把 f32 的 6e-8 相对误差放大成 1e-3 量级，直接过不了
//!   verify_series.py 的 1e-4 相对误差阈值（实测把输入列换成 f32 后，10 个值里有 4 个
//!   相对误差在 1e-3 上下）。所以本文件自带一个 f64 版本的逐笔读取，
//!   语义与 read_trade_fast_inner(code, date, true, false, usize::MAX) 完全一致：
//!     不按 with_retreat 过滤（那条按字节比 "32"，2025 年前后写成 "32.0" 就不生效），
//!     自己做数值过滤 flag != 32；不做下午平移；只丢弃 pandas dropna 会丢的行。
//!   同理，pandas 的 read_csv 用 f64 解析，这里也用 f64（std 的 parse 与 pandas 都是
//!   正确舍入，两种解析结果一致）。
//!
//! 【dropna 的口径】旧代码读进来的是 usecols=[date, exchtime, price, volume, turnover,
//!   flag, localtime, ask_order, bid_order] 这 9 列，随后 df.dropna() 会把这 9 列里
//!   任一为空的行整行丢掉。这里逐列检查字段是否为空，口径与之一致（本文件只用到
//!   price/turnover/flag/exchtime，其余几列只为「这一行会不会被 dropna 丢掉」而检查）。
//!
//! ════════════════════════════════════════════════════════════════════════
//! 四、恒定的因子个数
//! ════════════════════════════════════════════════════════════════════════
//!   算不出来（没有该股该日文件、或上午段一笔都没有）返回 None，由上游补一列 NaN，
//!   绝不条件跳过、也绝不少输出几列。任何非有限值（inf/-inf）一律写成 NaN，
//!   与旧框架 `df.replace(inf, nan).replace(-inf, nan)` 落盘时的口径一致。

use std::fs;

use crate::data_paths;

// transaction.csv 的列下标（0-based，与文件头一致）：
//   symbol,datetime,date,time,exchtime,localtime,index,price,volume,turnover,flag,
//   order_type,func_code,ask_order,bid_order
const COL_DATE: usize = 2;
const COL_EXCHTIME: usize = 4;
const COL_LOCALTIME: usize = 5;
const COL_PRICE: usize = 7;
const COL_VOLUME: usize = 8;
const COL_TURNOVER: usize = 9;
const COL_FLAG: usize = 10;
const COL_ASK_ORDER: usize = 13;
const COL_BID_ORDER: usize = 14;
/// 旧代码 usecols 里的 9 列，任列为空 → pandas 会 dropna 掉这一行
const DROPNA_COLS: [usize; 9] = [
    COL_DATE,
    COL_EXCHTIME,
    COL_LOCALTIME,
    COL_PRICE,
    COL_VOLUME,
    COL_TURNOVER,
    COL_FLAG,
    COL_ASK_ORDER,
    COL_BID_ORDER,
];

const DAY_US: i64 = 86_400_000_000;
const MIN_US: i64 = 60_000_000;
/// 下午段前移 90 分钟
const SHIFT_US: i64 = 90 * 60 * 1_000_000;
/// 交易所时间戳的 8 小时偏移（CSV 里的 exchtime 是 epoch 微秒 UTC）
const OFFSET_US: i64 = 8 * 3600 * 1_000_000;

/// 窗口边界（当天零点起的微秒，闭区间两端）
const OPEN_US: i64 = (9 * 3600 + 30 * 60) * 1_000_000; // 09:30:00.000000
const AM_END_US: i64 = (11 * 3600 + 30 * 60) * 1_000_000; // 11:30:00.000000
const PM_START_US: i64 = 13 * 3600 * 1_000_000; // 13:00:00.000000
const LAST_US: i64 = (14 * 3600 + 57 * 60) * 1_000_000; // 14:57:00.000000

/// 分箱范围：前移后的序列落在 09:30 ~ 13:27（分钟数 570 ~ 807）
const FIRST_MIN: i64 = 9 * 60 + 30; // 570
const LAST_MIN: i64 = 13 * 60 + 27; // 807
const NBINS: usize = (LAST_MIN - FIRST_MIN + 1) as usize; // 238

/// 每个时段 10 个指标（5 个协方差 + 5 个绝对值）
pub const N_OUT: usize = 10;

const PER_SESSION: [&str; 5] = [
    "cov_brown_all",
    "cov_brown_open30",
    "cov_brown_morning",
    "cov_brown_afternoon",
    "cov_brown_close30",
];

/// 五个窗口在「前移后分钟数」上的闭区间；all 用 (FIRST_MIN, LAST_MIN) 覆盖全部。
/// 顺序必须与 PER_SESSION 一致，也就是与旧代码 get_factor 的返回顺序一致。
const WINDOWS: [(i64, i64); 5] = [
    (FIRST_MIN, LAST_MIN),  // all
    (570, 600),             // open30  09:30 ~ 10:00
    (570, 690),             // morning 09:30 ~ 11:30
    (690, 807),             // afternoon 11:30 ~ 13:27（真实下午，前移后）
    (780, 807),             // close30 13:00 ~ 13:27（真实 14:30~14:57）
];

pub fn names() -> Vec<String> {
    let mut out = Vec::with_capacity(N_OUT);
    for m in PER_SESSION.iter() {
        out.push((*m).to_string());
    }
    for m in PER_SESSION.iter() {
        out.push(format!("abs_{}", m));
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

/// 解析一行，返回 None 表示这行被丢掉（表头、空行、pandas 会 dropna 掉的行、撤单行）
#[inline]
fn parse_line(line: &[u8]) -> Option<RawRow> {
    if line.is_empty() {
        return None;
    }
    let fields = split_fields(line);
    // pandas 的 dropna：9 列里任一为空就整行丢掉
    for &k in DROPNA_COLS.iter() {
        if fields[k].is_empty() {
            return None;
        }
    }
    let exchtime = parse_f64(fields[COL_EXCHTIME]);
    if !exchtime.is_finite() {
        return None;
    }
    // CSV 里的 exchtime 是 "1740964500130000.0" 这种浮点写法，取整到微秒
    let exchtime_us = exchtime.round() as i64;
    if exchtime_us == 0 {
        return None;
    }
    let price = parse_f64(fields[COL_PRICE]);
    let turnover = parse_f64(fields[COL_TURNOVER]);
    if !price.is_finite() || !turnover.is_finite() {
        return None;
    }
    let flag = parse_f64(fields[COL_FLAG]);
    if !flag.is_finite() {
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
/// 语义与 read_trade_fast_inner(code, date, true, false, usize::MAX) 一致。
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

/// 两列的样本协方差（ddof=1）。样本 < 2 返回 NaN；某列方差为 0 时按 numpy 口径返回 0
/// （注意与 corr 不同，这里**不**因为方差为 0 给 NaN）。
fn cov(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len();
    if n < 2 || y.len() != n {
        return f64::NAN;
    }
    let nf = n as f64;
    let mx = x.iter().sum::<f64>() / nf;
    let my = y.iter().sum::<f64>() / nf;
    let mut s = 0.0f64;
    for i in 0..n {
        s += (x[i] - mx) * (y[i] - my);
    }
    s / (nf - 1.0)
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

/// 逐股入口：算不出来返回 None（上游会补 NaN，保证因子个数恒定）。
pub fn compute_stock(code: &str, date: i64) -> Option<Vec<f32>> {
    let rows = read_trade_f64(code, date)?;
    if rows.is_empty() {
        return None;
    }

    // ── 第 2、3 步：真实钟点切窗口 + 下午段前移 90 分钟，按 df1 再 df2 拼接 ──
    let mut seq: Vec<(i64, f64, f64, i32)> = Vec::with_capacity(rows.len());
    let mut n_morning = 0usize;
    for r in rows.iter() {
        let day = r.time_us.rem_euclid(DAY_US);
        if day >= OPEN_US && day <= AM_END_US {
            seq.push((day, r.price, r.turnover, r.flag));
            n_morning += 1;
        }
    }
    // 旧代码第 56 行的门槛 `df.shape[0] > 0 and df1.shape[0] > 0`：
    // df1（上午段）为空就整条返回 None
    if n_morning == 0 {
        return None;
    }
    for r in rows.iter() {
        let day = r.time_us.rem_euclid(DAY_US);
        if day >= PM_START_US && day <= LAST_US {
            seq.push((day - SHIFT_US, r.price, r.turnover, r.flag));
        }
    }

    // ── 第 4 步：flag 变号、乘 turnover、整段减去全局最小值 ──
    let mut to: Vec<f64> = seq
        .iter()
        .map(|&(_, _, t, f)| t * flag_sign(f))
        .collect();
    let mut min_to = f64::INFINITY;
    for v in to.iter() {
        if *v < min_to {
            min_to = *v;
        }
    }
    for v in to.iter_mut() {
        *v -= min_to;
    }

    // ── 第 5、6 步：log(price).diff().replace(0, nan) → dropna → 按分钟分箱 ──
    // 第 0 行是 NaN 必然丢掉；diff 正好为 0 的行也丢掉（那一行的 turnover 不参与分箱）
    let mut bins: Vec<Vec<(f64, f64)>> = vec![Vec::new(); NBINS];
    let mut prev_logp = f64::NAN;
    for i in 0..seq.len() {
        let logp = seq[i].1.ln();
        let d = logp - prev_logp;
        prev_logp = logp;
        if i == 0 {
            continue;
        }
        if d.is_nan() || d == 0.0 {
            continue;
        }
        let minute = seq[i].0 / MIN_US;
        let idx = minute - FIRST_MIN;
        if idx < 0 || idx >= LAST_MIN - FIRST_MIN + 1 {
            continue;
        }
        bins[idx as usize].push((to[i], d));
    }

    // 每个分钟箱：brown_price = diff 的 std(ddof=1)，disag = std/mean(ddof=1)
    // df4 只保留样本数 >= 2 的分钟（std 为 NaN 的行会被 concat(...).dropna() 丢掉）
    let mut brown = vec![f64::NAN; NBINS];
    let mut disag = vec![f64::NAN; NBINS];
    let mut kept = vec![false; NBINS];
    for b in 0..NBINS {
        let v = &bins[b];
        let n = v.len();
        if n < 2 {
            continue;
        }
        let nf = n as f64;
        let mean_to = v.iter().map(|x| x.0).sum::<f64>() / nf;
        let mean_d = v.iter().map(|x| x.1).sum::<f64>() / nf;
        let var_to = v.iter().map(|x| (x.0 - mean_to) * (x.0 - mean_to)).sum::<f64>() / (nf - 1.0);
        let var_d = v.iter().map(|x| (x.1 - mean_d) * (x.1 - mean_d)).sum::<f64>() / (nf - 1.0);
        brown[b] = var_d.sqrt();
        disag[b] = var_to.sqrt() / mean_to;
        // dropna 只丢 NaN（inf 不丢），口径与 pandas 一致
        kept[b] = !brown[b].is_nan() && !disag[b].is_nan();
    }

    // ── 第 7 步：五个窗口各取两列协方差 ──
    let mut out = Vec::with_capacity(N_OUT);
    let mut covs = [0.0f64; 5];
    for (w, &(lo, hi)) in WINDOWS.iter().enumerate() {
        let mut x: Vec<f64> = Vec::new();
        let mut y: Vec<f64> = Vec::new();
        for b in 0..NBINS {
            if !kept[b] {
                continue;
            }
            let minute = FIRST_MIN + b as i64;
            if minute < lo || minute > hi {
                continue;
            }
            x.push(brown[b]);
            y.push(disag[b]);
        }
        covs[w] = finite_or_nan(cov(&x, &y));
    }

    // ── 第 8 步：5 个协方差 + 5 个绝对值 ──
    for v in covs.iter() {
        out.push(*v as f32);
    }
    for v in covs.iter() {
        out.push(finite_or_nan(v.abs()) as f32);
    }
    debug_assert_eq!(out.len(), N_OUT);
    Some(out)
}

/// hm16 全是逐股因子，没有横截面版本。
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
        assert_eq!(n[0], "cov_brown_all");
        assert_eq!(n[4], "cov_brown_close30");
        assert_eq!(n[5], "abs_cov_brown_all");
        assert_eq!(n[9], "abs_cov_brown_close30");
    }

    #[test]
    fn flag_sign_matches_python() {
        assert_eq!(flag_sign(66), 1.0);
        assert_eq!(flag_sign(83), -1.0);
        assert_eq!(flag_sign(70), 0.0);
    }

    #[test]
    fn dropna_columns_and_retreat() {
        // flag==32（2025 年写法 "32.0"）必须被丢掉
        let line = b"000001,2025-03-03 09:30:00.000,20250303,93000000,1740965400000000.0,1740965400000000.0,1,11.45,100,1145,32.0,,,0,1";
        assert!(parse_line(line).is_none());
        // 正常行
        let line = b"000001,2025-03-03 09:30:00.000,20250303,93000000,1740965400000000.0,1740965400000000.0,1,11.45,100,1145,66.0,,,0,1";
        let r = parse_line(line).unwrap();
        assert_eq!(r.flag, 66);
        assert_eq!(r.price, 11.45);
        assert!((r.turnover - 1145.0).abs() < 1e-12);
        // ask_order 为空 → pandas 会 dropna，这里也要丢
        let line = b"000001,2025-03-03 09:30:00.000,20250303,93000000,1740965400000000.0,1740965400000000.0,1,11.45,100,1145,66.0,,,,";
        assert!(parse_line(line).is_none());
    }
}
