//! 高速 CSV 读取器：替代 pandas.read_csv 的 Rust 多线程实现。
//!
//! 设计目标（run_factor_pipeline 优化方案 Phase 1）：
//! - 用 memmap2 内存映射文件，热数据走 OS page cache
//! - 大文件按字节块切分，rayon 并行解析（消除 pandas 单线程瓶颈）
//! - 读取时即完成全部预处理：flag!=32 过滤、exchtime→epoch 秒、下午时段平移
//! - 列裁剪：只解析 hm90.go 等流水线需要的 7 列
//! - 输出连续内存的 Vec<TradeRecord>，对后续 Rust 算法友好（零拷贝）
//!
//! 与 python/rust_pyfunc/trading_data_utils.py 中 read_trade 的语义对齐：
//!   read_trade(symbol, date, with_retreat=0) 返回 DataFrame，列含
//!   exchtime(已转 Timestamp)/price/volume/turnover/flag/index/localtime/ask_order/bid_order
//!   - with_retreat=0 (默认)：过滤 flag!=32
//!   - exchtime 由微秒整数转为 Timestamp（= 微秒/1e6 秒 + 1970-01-01 08:00:00 偏移）
//!   - 不做 adjust_afternoon（那由调用方显式调用）
//!
//! 本模块的 read_trade_fast 直接返回 epoch 秒（float64），跳过 Timestamp 往返转换。
//! adjust_afternoon 由参数 with_afternoon_adjust 控制，读取时直接平移（避免 pandas 索引操作）。

use memmap2::Mmap;
use rayon::prelude::*;
use std::fs::File;
use std::io::Read;
use std::path::Path;

/// 单条逐笔成交记录（已预处理）。
///
/// 字段顺序与 hm90.go 的 prepare() 中 np.column_stack 顺序一致：
///   [time, price, volume, turnover, flag, bid_order, ask_order]
/// 其中 time 为 epoch 秒（float64，与 prepare 中 exchtime.astype(int64)//1e9 一致）。
#[derive(Clone, Copy, Debug, Default)]
#[repr(C)]
pub struct TradeRecord {
    /// 成交时间，epoch 秒（exchtime 微秒整数 / 1e6，整数秒精度）
    pub time_sec: f32,
    /// 成交时间，epoch 微秒（time_sec 的精确版，含 8h 偏移 + 下午前移）。
    /// time_sec 在 2025 年 epoch(1.7e9) 附近 f32 精度仅 ±128s，无法做毫秒级窗口对齐；
    /// 需要精确时间（窗口二分、延迟区间）的因子请用 time_us。
    pub time_us: i64,
    pub price: f32,
    pub volume: f32,
    pub turnover: f32,
    /// 成交标志（i32）：66=主买, 83=主卖, 32=撤单（已被过滤）
    pub flag: i32,
    pub bid_order: i64,
    pub ask_order: i64,
}

/// adjust_afternoon 的默认时间窗口（only_inday=1）。
/// 上午保留 [09:30:00, 11:30:00]；下午 [13:00:00, 14:57:00] 整体前移 90 分钟。
const AFTERNOON_START_SEC: i64 = 13 * 3600; // 13:00:00 当天零点起的秒数
const AFTERNOON_END_SEC: i64 = 14 * 3600 + 57 * 60; // 14:57:00
const MORNING_START_SEC: i64 = 9 * 3600 + 30 * 60; // 09:30:00
const MORNING_END_SEC: i64 = 11 * 3600 + 30 * 60; // 11:30:00
const AFTERNOON_SHIFT_SEC: i64 = 90 * 60; // 下午时段前移 90 分钟

/// 列在 CSV 中的 0-based 索引（与 transaction CSV 头一致）。
///   symbol,datetime,date,time,exchtime,localtime,index,price,volume,turnover,flag,order_type,func_code,ask_order,bid_order
///     0      1     2    3      4         5       6     7      8       9       10     11         12        13        14
const COL_EXCHTIME: usize = 4;
const COL_PRICE: usize = 7;
const COL_VOLUME: usize = 8;
const COL_TURNOVER: usize = 9;
const COL_FLAG: usize = 10;
const COL_ASK_ORDER: usize = 13;
const COL_BID_ORDER: usize = 14;

/// resolve 股票数据路径，复用 read_trade 的多路径搜索逻辑。
///
/// 优先级：环境变量 RUST_PYFUNC_LEVEL2_PATH > /ssd_data/stock > /nas197/binary/stock/sz_alpha/stock
fn resolve_stock_path(date: i64, subdir: &str, filename: &str) -> std::io::Result<String> {
    // 环境变量优先
    if let Ok(env_path) = std::env::var("RUST_PYFUNC_LEVEL2_PATH") {
        let p = Path::new(&env_path)
            .join(date.to_string())
            .join(subdir)
            .join(filename);
        if p.exists() {
            return Ok(p.to_string_lossy().into_owned());
        }
        return Err(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!("RUST_PYFUNC_LEVEL2_PATH 指定路径不存在: {}", p.display()),
        ));
    }
    // 默认两路径
    for root in ["/ssd_data/stock", "/nas197/binary/stock/sz_alpha/stock"] {
        let p = Path::new(root)
            .join(date.to_string())
            .join(subdir)
            .join(filename);
        if p.exists() {
            return Ok(p.to_string_lossy().into_owned());
        }
    }
    Err(std::io::Error::new(
        std::io::ErrorKind::NotFound,
        format!(
            "股票数据文件未找到: {}/{}/{}/{}",
            "{root}", date, subdir, filename
        ),
    ))
}

/// 手写快速解析 i64（比 std::from_str 快，无 Unicode 校验开销）。
/// 输入为纯 ASCII 数字字节切片，允许前导空格。
#[inline]
fn parse_i64_fast(bytes: &[u8]) -> i64 {
    let mut neg = false;
    let mut i = 0;
    // 跳过前导空白
    while i < bytes.len() && bytes[i] == b' ' {
        i += 1;
    }
    if i < bytes.len() && (bytes[i] == b'-' || bytes[i] == b'+') {
        neg = bytes[i] == b'-';
        i += 1;
    }
    let mut val: i64 = 0;
    while i < bytes.len() {
        let c = bytes[i];
        if c < b'0' || c > b'9' {
            break;
        }
        val = val * 10 + (c - b'0') as i64;
        i += 1;
    }
    if neg {
        -val
    } else {
        val
    }
}

/// 手写快速解析 f64。优先尝试整数快速路径，失败再走 std 解析。
#[inline]
fn parse_f64_fast(bytes: &[u8]) -> f64 {
    // 快速整数路径（无小数点、无 e/E、无 inf/nan）
    let mut has_dot = false;
    let mut has_exp = false;
    let mut all_digits_or_sign = true;
    for (i, &c) in bytes.iter().enumerate() {
        if c == b'.' {
            has_dot = true;
            continue;
        }
        if c == b'e' || c == b'E' {
            has_exp = true;
            continue;
        }
        if c == b'-' || c == b'+' {
            // 仅首位或 e 后允许
            if i != 0 && bytes[i - 1] != b'e' && bytes[i - 1] != b'E' {
                all_digits_or_sign = false;
                break;
            }
            continue;
        }
        if c < b'0' || c > b'9' {
            all_digits_or_sign = false;
            break;
        }
    }
    if all_digits_or_sign && !has_exp {
        if !has_dot {
            // 纯整数
            return parse_i64_fast(bytes) as f64;
        }
        // 含小数点：手动解析
        return parse_decimal_fast(bytes);
    }
    // 退化到 std（处理科学计数、inf、nan、空串等）
    std::str::from_utf8(bytes)
        .ok()
        .and_then(|s| s.trim().parse::<f64>().ok())
        .unwrap_or(f64::NAN)
}

/// 手写小数 f64 解析（如 "12.34"、"-0.00001"）。
#[inline]
fn parse_decimal_fast(bytes: &[u8]) -> f64 {
    let mut i = 0;
    let mut neg = false;
    if i < bytes.len() && (bytes[i] == b'-') {
        neg = true;
        i += 1;
    } else if i < bytes.len() && bytes[i] == b'+' {
        i += 1;
    }
    let mut int_part: f64 = 0.0;
    while i < bytes.len() && bytes[i] >= b'0' && bytes[i] <= b'9' {
        int_part = int_part * 10.0 + (bytes[i] - b'0') as f64;
        i += 1;
    }
    let mut frac: f64 = 0.0;
    let mut frac_scale: f64 = 1.0;
    if i < bytes.len() && bytes[i] == b'.' {
        i += 1;
        while i < bytes.len() && bytes[i] >= b'0' && bytes[i] <= b'9' {
            frac = frac * 10.0 + (bytes[i] - b'0') as f64;
            frac_scale *= 10.0;
            i += 1;
        }
    }
    let mut result = int_part + frac / frac_scale;
    if neg {
        result = -result;
    }
    result
}

/// 解析单行（不带回车换行），返回 Option<TradeRecord>（None 表示被过滤或空行）。
///
/// with_retreat=false 时过滤 flag==32；exchtime 微秒整数转 epoch 秒；
/// with_afternoon_adjust=true 时做下午时段平移（时间超出 [09:30,11:30]∪[13:00,14:57] 的行被过滤）。
#[inline]
fn parse_line(line: &[u8], with_retreat: bool, with_afternoon_adjust: bool) -> Option<TradeRecord> {
    if line.is_empty() {
        return None;
    }
    // 按逗号切分字段。已知结构 15 列，用游标遍历避免分配 Vec。
    // 我们需要的列：4(exchtime), 7(price), 8(volume), 9(turnover), 10(flag), 13(ask_order), 14(bid_order)
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

    // 解析 flag
    let flag_bytes = fields[COL_FLAG];
    // flag==32 过滤：仅当 with_retreat=false
    // flag 字段可能是 "32"、"66"、"83" 或空字符串
    if !with_retreat && flag_bytes == b"32" {
        return None;
    }
    let flag = if flag_bytes.is_empty() {
        0
    } else {
        parse_i64_fast(flag_bytes) as i32
    };

    // exchtime：微秒整数（UTC）→ 带交易所时区偏移的 epoch 整数秒。
    // 与 read_trade + hm90.go 的 prepare() 语义对齐：
    //   read_trade: exchtime = to_timedelta(微秒/1e6) + Timestamp('1970-01-01 08:00:00')
    //   prepare:    time_sec = exchtime.astype(int64) // 1e9
    //             = (微秒数*1000 + 28800e9) // 1e9   （Timestamp int64 是纳秒）
    //             = (微秒数 + 28800e6) // 1e6         （整除截断小数部分）
    // 故输出 = (exchtime_us + 28800_000_000) / 1_000_000 的整数部分。
    let exchtime_us = parse_i64_fast(fields[COL_EXCHTIME]);
    // 跳过 CSV 表头及无效时间行；未做 afternoon adjust 时也必须过滤。
    if exchtime_us == 0 {
        return None;
    }
    const EXCHANGE_OFFSET_US: i64 = 8 * 3600 * 1_000_000;
    let total_us = exchtime_us + EXCHANGE_OFFSET_US;

    // 精确微秒时间（含 afternoon 平移）。final_time_sec 由它派生为整数秒。
    // time_us 保留微秒精度，供需要毫秒级跨股时间对齐的因子使用。
    let final_time_us: Option<i64> = if with_afternoon_adjust {
        let day_offset = ((exchtime_us / 1_000_000) + 8 * 3600).rem_euclid(86400);
        if day_offset >= AFTERNOON_START_SEC && day_offset <= AFTERNOON_END_SEC {
            // 下午时段：前移 90 分钟
            Some(total_us - AFTERNOON_SHIFT_SEC * 1_000_000)
        } else if day_offset >= MORNING_START_SEC && day_offset <= MORNING_END_SEC {
            // 上午时段：保留
            Some(total_us)
        } else {
            // 集合竞价前或收盘后：过滤
            None
        }
    } else {
        Some(total_us)
    };
    let final_time_us = final_time_us?;
    let final_time_sec = (final_time_us / 1_000_000) as f64;

    Some(TradeRecord {
        time_sec: final_time_sec as f32,
        time_us: final_time_us,
        price: parse_f64_fast(fields[COL_PRICE]) as f32,
        volume: parse_f64_fast(fields[COL_VOLUME]) as f32,
        turnover: parse_f64_fast(fields[COL_TURNOVER]) as f32,
        flag,
        bid_order: parse_i64_fast(fields[COL_BID_ORDER]),
        ask_order: parse_i64_fast(fields[COL_ASK_ORDER]),
    })
}

/// 在字节切片中找到第 N 个换行符后的位置（作为并行块起点）。
/// 返回该换行符之后第一个字节的索引（确保不在行中间切分）。
fn find_line_boundary(data: &[u8], approx_pos: usize) -> usize {
    if approx_pos >= data.len() {
        return data.len();
    }
    // 从 approx_pos 向后找第一个 \n，返回其后位置
    let search = &data[approx_pos..];
    match search.iter().position(|&b| b == b'\n') {
        Some(offset) => approx_pos + offset + 1,
        None => data.len(),
    }
}

/// 内部核心：读 transaction CSV → Vec<TradeRecord>。
///
/// - `with_retreat`: true 保留 flag==32 的撤单记录；false 过滤（与 read_trade 默认一致）
/// - `with_afternoon_adjust`: true 则读时做 adjust_afternoon 平移并过滤非盘中数据
/// - `parallel_threshold`: 文件大于此字节数时启用多线程并行解析（默认 8MB）
pub fn read_trade_fast_inner(
    code: &str,
    date: i64,
    with_retreat: bool,
    with_afternoon_adjust: bool,
    parallel_threshold: usize,
) -> std::io::Result<Vec<TradeRecord>> {
    let filename = format!("{}_{}_transaction.csv", code, date);
    let path = resolve_stock_path(date, "transaction", &filename)?;

    let file = File::open(&path)?;
    let meta = file.metadata()?;
    let file_size = meta.len() as usize;

    // 小文件直接用 read_to_string（避免 mmap 的设置开销）
    if file_size < parallel_threshold {
        let mut content = String::new();
        File::open(&path)?.read_to_string(&mut content)?;
        return Ok(parse_chunk(
            content.as_bytes(),
            with_retreat,
            with_afternoon_adjust,
        ));
    }

    // 大文件用 mmap + 多线程并行解析
    let mmap = unsafe { Mmap::map(&file)? };
    let data = &mmap[..];

    // 跳过表头行
    let body_start = data
        .iter()
        .position(|&b| b == b'\n')
        .map(|p| p + 1)
        .unwrap_or(0);

    // 切分块数：基于文件大小和可用 CPU 数决定，避免过度切分
    let n_threads = rayon::current_num_threads().min(32).max(1);
    let chunk_size = (data.len() - body_start) / n_threads + 1;

    let chunks: Vec<(usize, usize)> = (0..n_threads)
        .map(|i| {
            let start = body_start + i * chunk_size;
            let raw_end = start + chunk_size;
            let start = if i == 0 {
                start
            } else {
                find_line_boundary(data, start)
            };
            let end = if i == n_threads - 1 {
                data.len()
            } else {
                find_line_boundary(data, raw_end.min(data.len()))
            };
            (start.min(end), end)
        })
        .filter(|(s, e)| s < e)
        .collect();

    // 并行解析各块
    let mut partials: Vec<Vec<TradeRecord>> = chunks
        .par_iter()
        .map(|&(s, e)| parse_chunk(&data[s..e], with_retreat, with_afternoon_adjust))
        .collect();

    // 合并结果（预分配总容量减少 reallocation）
    let total: usize = partials.iter().map(|v| v.len()).sum();
    let mut result = Vec::with_capacity(total);
    for p in partials.drain(..) {
        result.extend(p);
    }
    Ok(result)
}

/// 解析一个字节块（可能含多行）→ Vec<TradeRecord>。
fn parse_chunk(data: &[u8], with_retreat: bool, with_afternoon_adjust: bool) -> Vec<TradeRecord> {
    // 估算行数（按平均行长）预分配
    let est_lines = data.len() / 80 + 1;
    let mut out = Vec::with_capacity(est_lines);
    let mut start = 0usize;
    for i in 0..data.len() {
        if data[i] == b'\n' {
            let line = &data[start..i];
            // 去掉行尾 \r
            let line = if line.last() == Some(&b'\r') {
                &line[..line.len() - 1]
            } else {
                line
            };
            if let Some(rec) = parse_line(line, with_retreat, with_afternoon_adjust) {
                out.push(rec);
            }
            start = i + 1;
        }
    }
    // 处理最后一段（无结尾换行的情况）
    if start < data.len() {
        let line = &data[start..];
        let line = if line.last() == Some(&b'\r') {
            &line[..line.len() - 1]
        } else {
            line
        };
        if let Some(rec) = parse_line(line, with_retreat, with_afternoon_adjust) {
            out.push(rec);
        }
    }
    out
}

const MKT_COL_EXCHTIME: usize = 4;
const MKT_COL_LAST_PRC: usize = 6;
const MKT_COL_HIGH_LIMITED: usize = 12;
const MKT_COL_LOW_LIMITED: usize = 13;
const MKT_COL_VOLUME: usize = 14;
const MKT_COL_TURNOVER: usize = 15;
const MKT_COL_TOTAL_ASK_VOL: usize = 19;
const MKT_COL_TOTAL_BID_VOL: usize = 20;
const MKT_COL_ASK_PRC_BASE: usize = 21; // ask_prc1=21, ask_vol1=22, bid_prc1=23, bid_vol1=24, ...


/// 单条 market_data 快照记录（已预处理）。
/// 包含 10 档 ask/bid 量价，以及常用字段。
#[derive(Clone, Copy, Debug, Default)]
pub struct MarketRecord {
    /// 成交时间，epoch 秒（exchtime 微秒 + 8h 偏移，再整除 1e6，与 read_trade_fast 一致）
    pub time_sec: f32,
    /// 精确到微秒的快照时间（含 8h 偏移及可选的下午时段平移）。
    /// 3 秒及更细粒度分桶必须使用本字段，不能使用精度不足的 `time_sec`。
    pub time_us: i64,
    pub last_prc: f32,
    pub volume: f32,
    pub turnover: f32,
    /// 全部档位卖单挂单量（含 10 档之外）
    pub total_ask_vol: f32,
    /// 全部档位买单挂单量（含 10 档之外）
    pub total_bid_vol: f32,
    /// 10 档卖价 [ask_prc1, ..., ask_prc10]
    pub ask_prcs: [f32; 10],
    /// 10 档卖量
    pub ask_vols: [f32; 10],
    /// 10 档买价
    pub bid_prcs: [f32; 10],
    /// 10 档买量
    pub bid_vols: [f32; 10],
}

/// 解析 market_data 的一行。
/// with_high_low_limited=false 时过滤涨跌停（ask_prc1==0 或 bid_prc1==0）。
/// with_afternoon_adjust=true 时做下午时段平移（同 read_trade_fast）。
#[inline]
fn parse_market_line(
    line: &[u8],
    with_high_low_limited: bool,
    with_afternoon_adjust: bool,
) -> Option<MarketRecord> {
    if line.is_empty() {
        return None;
    }
    // market_data 有 61 列，按逗号切分
    let mut fields: [&[u8]; 61] = [&[][..]; 61];
    let mut start = 0usize;
    let mut col = 0usize;
    for (i, &b) in line.iter().enumerate() {
        if b == b',' {
            if col < 61 {
                fields[col] = &line[start..i];
            }
            col += 1;
            start = i + 1;
        }
    }
    if col < 61 {
        fields[col] = &line[start..];
    }

    // 涨跌停过滤
    if !with_high_low_limited {
        let ask_prc1 = parse_f64_fast(fields[MKT_COL_ASK_PRC_BASE]);
        let bid_prc1 = parse_f64_fast(fields[MKT_COL_ASK_PRC_BASE + 2]);
        if ask_prc1 == 0.0 || bid_prc1 == 0.0 {
            return None;
        }
    }

    // exchtime → epoch 秒（与 read_trade_fast 相同的偏移逻辑）。
    // 过滤 exchtime=0 的无效行（market_data 的集合竞价前空快照可能有 exchtime=0）。
    let exchtime_us = parse_i64_fast(fields[MKT_COL_EXCHTIME]);
    if exchtime_us == 0 {
        return None;
    }
    let total_us = exchtime_us + 8 * 3600 * 1_000_000;
    let final_time_us = if with_afternoon_adjust {
        let day_offset = ((exchtime_us / 1_000_000) + 8 * 3600).rem_euclid(86400);
        if day_offset >= AFTERNOON_START_SEC && day_offset <= AFTERNOON_END_SEC {
            Some(total_us - AFTERNOON_SHIFT_SEC * 1_000_000)
        } else if day_offset >= MORNING_START_SEC && day_offset <= MORNING_END_SEC {
            Some(total_us)
        } else {
            None
        }
    } else {
        Some(total_us)
    };

    let time_us = final_time_us?;
    let time_sec = (time_us / 1_000_000) as f64;

    // 解析 10 档 ask/bid（每档 4 列：ask_prc, ask_vol, bid_prc, bid_vol）
    let mut ask_prcs = [0.0f32; 10];
    let mut ask_vols = [0.0f32; 10];
    let mut bid_prcs = [0.0f32; 10];
    let mut bid_vols = [0.0f32; 10];
    for i in 0..10 {
        let base = MKT_COL_ASK_PRC_BASE + i * 4;
        ask_prcs[i] = parse_f64_fast(fields[base]) as f32;
        ask_vols[i] = parse_f64_fast(fields[base + 1]) as f32;
        bid_prcs[i] = parse_f64_fast(fields[base + 2]) as f32;
        bid_vols[i] = parse_f64_fast(fields[base + 3]) as f32;
    }

    Some(MarketRecord {
        time_sec: time_sec as f32,
        time_us,
        last_prc: parse_f64_fast(fields[MKT_COL_LAST_PRC]) as f32,
        volume: parse_f64_fast(fields[MKT_COL_VOLUME]) as f32,
        turnover: parse_f64_fast(fields[MKT_COL_TURNOVER]) as f32,
        total_ask_vol: parse_f64_fast(fields[MKT_COL_TOTAL_ASK_VOL]) as f32,
        total_bid_vol: parse_f64_fast(fields[MKT_COL_TOTAL_BID_VOL]) as f32,
        ask_prcs,
        ask_vols,
        bid_prcs,
        bid_vols,
    })
}

/// 解析 market_data 的字节块 → Vec<MarketRecord>。
fn parse_market_chunk(
    data: &[u8],
    with_high_low_limited: bool,
    with_afternoon_adjust: bool,
) -> Vec<MarketRecord> {
    let est_lines = data.len() / 200 + 1; // market_data 行较长
    let mut out = Vec::with_capacity(est_lines);
    let mut start = 0usize;
    for i in 0..data.len() {
        if data[i] == b'\n' {
            let line = &data[start..i];
            let line = if line.last() == Some(&b'\r') {
                &line[..line.len() - 1]
            } else {
                line
            };
            if let Some(rec) = parse_market_line(line, with_high_low_limited, with_afternoon_adjust)
            {
                out.push(rec);
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
        if let Some(rec) = parse_market_line(line, with_high_low_limited, with_afternoon_adjust) {
            out.push(rec);
        }
    }
    out
}

/// 内部核心：读 market_data CSV → Vec<MarketRecord>。
///
/// - `with_high_low_limited`: true 保留涨跌停；false 过滤（ask_prc1==0 或 bid_prc1==0）
/// - `with_afternoon_adjust`: true 做下午时段平移
/// - `parallel_threshold`: 文件大于此字节数时启用多线程解析
pub fn read_market_fast_inner(
    code: &str,
    date: i64,
    with_high_low_limited: bool,
    with_afternoon_adjust: bool,
    parallel_threshold: usize,
) -> std::io::Result<Vec<MarketRecord>> {
    let filename = format!("{}_{}_market_data.csv", code, date);
    let path = resolve_stock_path(date, "market_data", &filename)?;

    let file = File::open(&path)?;
    let meta = file.metadata()?;
    let file_size = meta.len() as usize;

    if file_size < parallel_threshold {
        let mut content = String::new();
        File::open(&path)?.read_to_string(&mut content)?;
        return Ok(parse_market_chunk(
            content.as_bytes(),
            with_high_low_limited,
            with_afternoon_adjust,
        ));
    }

    let mmap = unsafe { Mmap::map(&file)? };
    let data = &mmap[..];
    let body_start = data
        .iter()
        .position(|&b| b == b'\n')
        .map(|p| p + 1)
        .unwrap_or(0);

    let n_threads = rayon::current_num_threads().min(16).max(1);
    let chunk_size = (data.len() - body_start) / n_threads + 1;
    let chunks: Vec<(usize, usize)> = (0..n_threads)
        .map(|i| {
            let start = body_start + i * chunk_size;
            let raw_end = start + chunk_size;
            let start = if i == 0 {
                start
            } else {
                find_line_boundary(data, start)
            };
            let end = if i == n_threads - 1 {
                data.len()
            } else {
                find_line_boundary(data, raw_end.min(data.len()))
            };
            (start.min(end), end)
        })
        .filter(|(s, e)| s < e)
        .collect();

    let mut partials: Vec<Vec<MarketRecord>> = chunks
        .par_iter()
        .map(|&(s, e)| {
            parse_market_chunk(&data[s..e], with_high_low_limited, with_afternoon_adjust)
        })
        .collect();

    let total: usize = partials.iter().map(|v| v.len()).sum();
    let mut result = Vec::with_capacity(total);
    for p in partials.drain(..) {
        result.extend(p);
    }
    Ok(result)
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_i64() {
        assert_eq!(parse_i64_fast(b"1780622100020000"), 1780622100020000i64);
        assert_eq!(parse_i64_fast(b"32"), 32);
        assert_eq!(parse_i64_fast(b"66"), 66);
        assert_eq!(parse_i64_fast(b"-5"), -5);
        assert_eq!(parse_i64_fast(b""), 0);
        assert_eq!(parse_i64_fast(b"  42 "), 42);
    }

    #[test]
    fn test_parse_f64() {
        assert_eq!(parse_f64_fast(b"12.34"), 12.34);
        assert!((parse_f64_fast(b"0.00001") - 0.00001).abs() < 1e-12);
        assert_eq!(parse_f64_fast(b"100"), 100.0);
        assert!((parse_f64_fast(b"-0.5") + 0.5).abs() < 1e-12);
        assert!(parse_f64_fast(b"").is_nan());
    }

    #[test]
    fn test_parse_line_basic() {
        // 真实 CSV 行格式（含空字段 order_type,func_code）
        let line = b"000001,2026-06-05 09:30:00.000,20260605,93000000,1780622580000000,1780622580000000,1,13.50,100,1350.0,66,,,0,302";
        let rec = parse_line(line, false, false).expect("应解析成功");
        assert_eq!(rec.flag, 66);
        assert!((rec.price - 13.50).abs() < 1e-9);
        assert_eq!(rec.volume, 100.0);
        assert!((rec.turnover - 1350.0).abs() < 1e-9);
        assert_eq!(rec.bid_order, 302);
        assert_eq!(rec.ask_order, 0);
        // time_sec = 1780622580000000 / 1e6 = 1780622580.0
        assert!((rec.time_sec - 1780622580.0).abs() < 1e-6);
    }

    #[test]
    fn test_parse_line_flag32_filtered() {
        let line = b"000001,2026-06-05 09:15:00.020,20260605,91500020,1780622100020000,1780622100042000,305,0.0,100,0.0,32,,,0,302";
        // with_retreat=false → 过滤
        assert!(parse_line(line, false, false).is_none());
        // with_retreat=true → 保留
        assert!(parse_line(line, true, false).is_some());
    }

    #[test]
    fn test_afternoon_shift() {
        // 构造一个东八区下午 13:30:00 的成交时间。
        // 东八区 13:30:00 当天零点起的秒数 = 13*3600+30*60 = 48600
        // day_offset = (UTC秒 + 8*3600) % 86400 = 48600 → UTC秒 = 48600 - 28800 = 19800
        let utc_sec = 48600i64 - 8 * 3600;
        let exchtime_us = utc_sec * 1_000_000i64;
        let line = format!(
            "000001,t,20260605,0,{},{},1,10.0,100,1000.0,66,,,0,5",
            exchtime_us, exchtime_us
        );
        let rec = parse_line(line.as_bytes(), false, true).expect("应解析成功");
        // 输出 = (UTC秒 + 8h) - 90min = (19800+28800) - 5400 = 43200
        assert!((rec.time_sec - 43200.0).abs() < 1e-6);
    }

    #[test]
    fn test_morning_kept() {
        // 东八区上午 10:00:00 = day_offset 36000，应保留不平移
        // UTC秒 = 36000 - 28800 = 7200
        let utc_sec = 36000i64 - 8 * 3600;
        let exchtime_us = utc_sec * 1_000_000i64;
        let line = format!(
            "000001,t,20260605,0,{},{},1,10.0,100,1000.0,66,,,0,5",
            exchtime_us, exchtime_us
        );
        let rec = parse_line(line.as_bytes(), false, true).expect("应解析成功");
        // 输出 = UTC秒 + 8h = 7200 + 28800 = 36000
        assert!((rec.time_sec - 36000.0).abs() < 1e-6);
    }

    #[test]
    fn test_pre_open_filtered() {
        // 东八区 09:15:00（集合竞价前）= day_offset 33300，应被过滤
        let utc_sec = 33300i64 - 8 * 3600;
        let exchtime_us = utc_sec * 1_000_000i64;
        let line = format!(
            "000001,t,20260605,0,{},{},1,10.0,100,1000.0,66,,,0,5",
            exchtime_us, exchtime_us
        );
        assert!(parse_line(line.as_bytes(), false, true).is_none());
    }

    #[test]
    fn test_find_line_boundary() {
        let data = b"abc\ndef\nghi\njkl";
        // 从位置 5 开始找边界，应定位到第二个 \n 之后（位置 8）
        assert_eq!(find_line_boundary(data, 5), 8);
        // 超出范围
        assert_eq!(find_line_boundary(data, 100), data.len());
    }
}
