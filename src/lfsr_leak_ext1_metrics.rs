//! 线性反馈移位存储器的非线性组合（拓展一）—— 位流 × 算子 × 视角大发散
//!
//! 来源：sandbox_lfsr_leak 单日全市场探索（原版只在沙箱里输出矩阵给 Python 看）。
//! 正式版改成单股 per-stock pipeline：只算本股逐笔/盘口，不做任何跨股票运算，
//! 输出恒定 15744 列，算不出来给 NaN。跨股票的标准化/排名留给回测里的 rank 去做。
//!
//! 构成（名字由 build_specs 生成，与该函数同序）：
//!   2 个计数（盘口快照数、逐笔笔数）
//!   + 盘口 33 条位流 × 7 算子（rate / ac1,5,10 / run / tasym / spcv / ctx6）
//!   + 逐笔 13 条位流 × 4 算子（rate / ac1 / run / ctx6）
//!   + 16 组二元流 × 5 算子（corr / lead / and / xor / edge）
//!   + 10 个三元组 × 256 种三元布尔函数 × 6 统计（rate / ac1 / 泄露 / 泄露超额 / 边际 mid / 边际 imb）
//!   + 5 个钟控掩码 × 3 条被控流 × 2 统计 + 4 条半日差异 × 2 统计 + 2 时段 × 5 条流
//!
//! 与沙箱唯一的差别：沙箱最后做了一次全市场逐列 z-score，这里不做（那是存储之后的事）。
//! 读取器与算子都从沙箱原样搬来，逐位相同。
use pyo3::prelude::*;
use std::sync::OnceLock;

mod reader {
    //! 精简版数据读取（单线程，无 pyo3/rayon），从 fast_csv_reader.rs 移植。
    //! 读取 transaction / market_data CSV，做 adjust_afternoon 平移。
    //! 只保留因子实际用到的字段：盘口不解析第 6~10 档与 last_prc/volume，
    //! 逐笔不解析 time_sec 与委托序号 index。

    const AFTERNOON_START_SEC: i64 = 13 * 3600;
    const AFTERNOON_END_SEC: i64 = 14 * 3600 + 57 * 60;
    const MORNING_START_SEC: i64 = 9 * 3600 + 30 * 60;
    const MORNING_END_SEC: i64 = 11 * 3600 + 30 * 60;
    const AFTERNOON_SHIFT_SEC: i64 = 90 * 60;
    const SHIFT_US: i64 = 90 * 60 * 1_000_000;
    const OFFSET_US: i64 = 8 * 3600 * 1_000_000;

    const COL_EXCHTIME: usize = 4;
    const COL_PRICE: usize = 7;
    const COL_VOLUME: usize = 8;
    const COL_TURNOVER: usize = 9;
    const COL_FLAG: usize = 10;
    const COL_ASK_ORDER: usize = 13;
    const COL_BID_ORDER: usize = 14;

    const MKT_COL_EXCHTIME: usize = 4;
    const MKT_COL_TURNOVER: usize = 15;
    const MKT_COL_TOTAL_ASK_VOL: usize = 19;
    const MKT_COL_TOTAL_BID_VOL: usize = 20;
    const MKT_COL_ASK_PRC_BASE: usize = 21;

    #[derive(Clone, Copy, Debug, Default)]
    pub struct TradeRecord {
        pub time_us: i64,
        pub price: f64,
        pub volume: f64,
        pub turnover: f64,
        pub flag: i32,
        pub bid_order: i64,
        pub ask_order: i64,
    }

    #[derive(Clone, Copy, Debug, Default)]
    pub struct MarketRecord {
        pub time_us: i64,
        pub turnover: f64,
        pub total_ask_vol: f64,
        pub total_bid_vol: f64,
        pub ask_prc1: f64,
        pub bid_prc1: f64,
        pub ask_vols: [f64; 5],
        pub bid_vols: [f64; 5],
    }

    fn resolve_stock_path(date: i64, subdir: &str, filename: &str) -> std::io::Result<String> {
        let roots = crate::data_paths::level2_roots();
        for root in &roots {
            let p = root.join(date.to_string()).join(subdir).join(filename);
            if p.exists() {
                return Ok(p.to_string_lossy().into_owned());
            }
        }
        Err(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!("股票数据文件未找到: date={date}, subdir={subdir}, file={filename}"),
        ))
    }

    #[inline]
    fn parse_i64_fast(bytes: &[u8]) -> i64 {
        let mut neg = false;
        let mut i = 0;
        while i < bytes.len() && bytes[i] == b' ' { i += 1; }
        if i < bytes.len() && (bytes[i] == b'-' || bytes[i] == b'+') { neg = bytes[i] == b'-'; i += 1; }
        let mut val: i64 = 0;
        while i < bytes.len() {
            let c = bytes[i];
            if c < b'0' || c > b'9' { break; }
            val = val * 10 + (c - b'0') as i64;
            i += 1;
        }
        if neg { -val } else { val }
    }

    /// 一遍扫描解析数字：整数走 i64 累加再转 f64，带小数点的走整数部分 + 小数部分/量纲。
    /// 与旧版（先分类再分别走 parse_i64_fast / parse_decimal_fast）逐位相同；
    /// 遇到空格、字母、指数等异常字符时退回 trim().parse()，与旧版兜底一致。
    #[inline]
    fn parse_f64_fast(bytes: &[u8]) -> f64 {
        let n = bytes.len();
        let mut i = 0;
        let mut neg = false;
        if i < n && (bytes[i] == b'-' || bytes[i] == b'+') { neg = bytes[i] == b'-'; i += 1; }
        let mut iv: i64 = 0;
        let mut int_part = 0.0f64;
        while i < n {
            let c = bytes[i];
            if c < b'0' || c > b'9' { break; }
            let d = (c - b'0') as i64;
            iv = iv.wrapping_mul(10).wrapping_add(d);
            int_part = int_part * 10.0 + d as f64;
            i += 1;
        }
        if i == n {
            return (if neg { -iv } else { iv }) as f64;
        }
        if bytes[i] == b'.' {
            i += 1;
            let mut frac = 0.0f64;
            let mut scale = 1.0f64;
            while i < n {
                let c = bytes[i];
                if c < b'0' || c > b'9' { break; }
                frac = frac * 10.0 + (c - b'0') as f64;
                scale *= 10.0;
                i += 1;
            }
            if i == n {
                let r = int_part + frac / scale;
                return if neg { -r } else { r };
            }
        }
        std::str::from_utf8(bytes).ok().and_then(|s| s.trim().parse::<f64>().ok()).unwrap_or(f64::NAN)
    }

    #[inline]
    fn parse_line(line: &[u8], code: &str, date: i64) -> Option<TradeRecord> {
        if line.is_empty() { return None; }
        let mut fields: [&[u8]; 15] = [&[][..]; 15];
        let mut start = 0;
        let mut col = 0;
        for i in memchr::memchr_iter(b',', line) {
            if col < 15 { fields[col] = &line[start..i]; }
            col += 1;
            start = i + 1;
        }
        if col < 15 { fields[col] = &line[start..]; }
        // 非本股票 / 非本日期行直接丢（0=symbol, 2=date；date 解析成整数再比，覆盖 "20251031.0"）
        if fields[0] != code.as_bytes() || parse_i64_fast(fields[2]) != date { return None; }
        // 先解析成数值再比较：部分日期的 flag 列写 "32.0"，按字节比 b"32" 会漏掉撤单
        let flag = if fields[COL_FLAG].is_empty() { 0 } else { parse_i64_fast(fields[COL_FLAG]) as i32 };
        if flag == 32 { return None; }
        let exchtime_us = parse_i64_fast(fields[COL_EXCHTIME]);
        let total_us = exchtime_us + OFFSET_US;
        let day_offset = ((exchtime_us / 1_000_000) + 8 * 3600).rem_euclid(86400);
        let final_us = if day_offset >= AFTERNOON_START_SEC && day_offset <= AFTERNOON_END_SEC { total_us - SHIFT_US }
            else if day_offset >= MORNING_START_SEC && day_offset <= MORNING_END_SEC { total_us }
            else { return None; };
        Some(TradeRecord {
            time_us: final_us,
            price: parse_f64_fast(fields[COL_PRICE]),
            volume: parse_f64_fast(fields[COL_VOLUME]),
            turnover: parse_f64_fast(fields[COL_TURNOVER]),
            flag,
            bid_order: parse_i64_fast(fields[COL_BID_ORDER]),
            ask_order: parse_i64_fast(fields[COL_ASK_ORDER]),
        })
    }

    fn parse_chunk(data: &[u8], code: &str, date: i64) -> Vec<TradeRecord> {
        let mut out = Vec::with_capacity(data.len() / 130 + 1);
        let mut start = 0;
        for nl in memchr::memchr_iter(b'\n', data) {
            let line = &data[start..nl];
            let line = if line.last() == Some(&b'\r') { &line[..line.len() - 1] } else { line };
            if let Some(r) = parse_line(line, code, date) { out.push(r); }
            start = nl + 1;
        }
        if start < data.len() {
            let line = &data[start..];
            let line = if line.last() == Some(&b'\r') { &line[..line.len() - 1] } else { line };
            if let Some(r) = parse_line(line, code, date) { out.push(r); }
        }
        out
    }

    pub fn read_trade_fast(code: &str, date: i64) -> std::io::Result<Vec<TradeRecord>> {
        let filename = format!("{}_{}_transaction.csv", code, date);
        let path = resolve_stock_path(date, "transaction", &filename)?;
        let data = std::fs::read(&path)?;
        Ok(parse_chunk(&data, code, date))
    }

    #[inline]
    fn parse_market_line(line: &[u8], code: &str, date: i64) -> Option<MarketRecord> {
        if line.is_empty() { return None; }
        // 只解析到第 41 列（第 5 档挂量的卖量序号位），后面的列不看，逗号扫到那里就停
        const NEED_MAX: usize = MKT_COL_ASK_PRC_BASE + 4 * 4 + 3;
        let mut fields: [&[u8]; 61] = [&[][..]; 61];
        let mut start = 0;
        let mut col = 0;
        for i in memchr::memchr_iter(b',', line) {
            if col > NEED_MAX { break; }
            fields[col] = &line[start..i];
            col += 1;
            start = i + 1;
        }
        if col <= NEED_MAX { fields[col] = &line[start..]; }
        // 非本股票 / 非本日期行直接丢（0=symbol, 2=date）
        if fields[0] != code.as_bytes() || parse_i64_fast(fields[2]) != date { return None; }
        let ask_prc1 = parse_f64_fast(fields[MKT_COL_ASK_PRC_BASE]);
        let bid_prc1 = parse_f64_fast(fields[MKT_COL_ASK_PRC_BASE + 2]);
        if ask_prc1 == 0.0 || bid_prc1 == 0.0 { return None; }
        let exchtime_us = parse_i64_fast(fields[MKT_COL_EXCHTIME]);
        if exchtime_us == 0 { return None; }
        let total_us = exchtime_us + OFFSET_US;
        let day_offset = ((exchtime_us / 1_000_000) + 8 * 3600).rem_euclid(86400);
        let final_us = if day_offset >= AFTERNOON_START_SEC && day_offset <= AFTERNOON_END_SEC { total_us - SHIFT_US }
            else if day_offset >= MORNING_START_SEC && day_offset <= MORNING_END_SEC { total_us }
            else { return None; };
        let mut ask_vols = [0f64; 5];
        let mut bid_vols = [0f64; 5];
        for i in 0..5 {
            let base = MKT_COL_ASK_PRC_BASE + i * 4;
            ask_vols[i] = parse_f64_fast(fields[base + 1]);
            bid_vols[i] = parse_f64_fast(fields[base + 3]);
        }
        Some(MarketRecord {
            time_us: final_us,
            turnover: parse_f64_fast(fields[MKT_COL_TURNOVER]),
            total_ask_vol: parse_f64_fast(fields[MKT_COL_TOTAL_ASK_VOL]),
            total_bid_vol: parse_f64_fast(fields[MKT_COL_TOTAL_BID_VOL]),
            ask_prc1,
            bid_prc1,
            ask_vols,
            bid_vols,
        })
    }

    fn parse_market_chunk(data: &[u8], code: &str, date: i64) -> Vec<MarketRecord> {
        let mut out = Vec::with_capacity(data.len() / 200 + 1);
        let mut start = 0;
        for nl in memchr::memchr_iter(b'\n', data) {
            let line = &data[start..nl];
            let line = if line.last() == Some(&b'\r') { &line[..line.len() - 1] } else { line };
            if let Some(r) = parse_market_line(line, code, date) { out.push(r); }
            start = nl + 1;
        }
        if start < data.len() {
            let line = &data[start..];
            let line = if line.last() == Some(&b'\r') { &line[..line.len() - 1] } else { line };
            if let Some(r) = parse_market_line(line, code, date) { out.push(r); }
        }
        out
    }

    pub fn read_market_fast(code: &str, date: i64) -> std::io::Result<Vec<MarketRecord>> {
        let filename = format!("{}_{}_market_data.csv", code, date);
        let path = resolve_stock_path(date, "market_data", &filename)?;
        let data = std::fs::read(&path)?;
        Ok(parse_market_chunk(&data, code, date))
    }
}
use reader::{read_market_fast, read_trade_fast, MarketRecord, TradeRecord};

// ---------- 基础算子 ----------
fn rate(s: &[u8]) -> f64 {
    if s.is_empty() { return f64::NAN; }
    s.iter().map(|&b| b as f64).sum::<f64>() / s.len() as f64
}
fn ac_lag(v: &[u8], lag: usize) -> f64 {
    let n = v.len();
    if n <= lag + 10 { return f64::NAN; }
    let m = n - lag;
    let (mut s1, mut s2, mut sx, mut v1, mut v2) = (0., 0., 0., 0., 0.);
    for i in 0..m {
        let a = v[i] as f64; let b = v[i + lag] as f64;
        s1 += a; s2 += b; sx += a * b; v1 += a * a; v2 += b * b;
    }
    let mf = m as f64;
    let cov = sx / mf - (s1 / mf) * (s2 / mf);
    let va = v1 / mf - (s1 / mf) * (s1 / mf);
    let vb = v2 / mf - (s2 / mf) * (s2 / mf);
    if va <= 1e-12 || vb <= 1e-12 { return f64::NAN; }
    cov / (va * vb).sqrt()
}
fn corr_bits(a: &[u8], b: &[u8]) -> f64 { ac_ab(a, b, 0) }
fn ac_ab(a: &[u8], b: &[u8], lag: usize) -> f64 {
    let n = a.len().min(b.len());
    if n <= lag + 10 { return f64::NAN; }
    let m = n - lag;
    let (mut s1, mut s2, mut sx, mut v1, mut v2) = (0., 0., 0., 0., 0.);
    for i in 0..m {
        let x = a[i] as f64; let y = b[i + lag] as f64;
        s1 += x; s2 += y; sx += x * y; v1 += x * x; v2 += y * y;
    }
    let mf = m as f64;
    let cov = sx / mf - (s1 / mf) * (s2 / mf);
    let va = v1 / mf - (s1 / mf) * (s1 / mf);
    let vb = v2 / mf - (s2 / mf) * (s2 / mf);
    if va <= 1e-12 || vb <= 1e-12 { return f64::NAN; }
    cov / (va * vb).sqrt()
}
fn run_mean(s: &[u8]) -> f64 {
    if s.len() < 10 { return f64::NAN; }
    let (mut runs, mut len) = (0usize, 1usize);
    for i in 1..s.len() {
        if s[i] == s[i - 1] { len += 1; } else { runs += 1; len = 1; }
    }
    runs += 1;
    s.len() as f64 / runs as f64
}
fn trans_asym(s: &[u8]) -> f64 {
    if s.len() < 30 { return f64::NAN; }
    let (mut up, mut dn) = (0u32, 0u32);
    for i in 1..s.len() {
        if s[i - 1] == 0 && s[i] == 1 { up += 1; }
        if s[i - 1] == 1 && s[i] == 0 { dn += 1; }
    }
    (up as f64 - dn as f64) / (up + dn) as f64
}
fn spacing_cv(s: &[u8]) -> f64 {
    let pos: Vec<usize> = (0..s.len()).filter(|&i| s[i] == 1).collect();
    if pos.len() < 10 { return f64::NAN; }
    let gaps: Vec<f64> = pos.windows(2).map(|w| (w[1] - w[0]) as f64).collect();
    let m = gaps.iter().sum::<f64>() / gaps.len() as f64;
    if m <= 0.0 { return f64::NAN; }
    let sd = (gaps.iter().map(|g| (g - m).powi(2)).sum::<f64>() / gaps.len() as f64).sqrt();
    sd / m
}
fn ctx_acc(s: &[u8], k: usize) -> f64 {
    let n = s.len();
    if n < 4000 { return f64::NAN; }
    let size = 1usize << k;
    let mut idx = vec![0usize; n - k];
    let mut cur = 0usize;
    for i in 0..n - k {
        cur = ((cur << 1) | s[i] as usize) & (size - 1);
        if i + 1 >= k { idx[i + 1 - k] = cur; }
    }
    let nxt = &s[k..];
    let half = (n - k) / 2;
    let mut cnt1 = vec![0u32; size];
    let mut tot = vec![0u32; size];
    for j in 0..half {
        tot[idx[j]] += 1;
        cnt1[idx[j]] += nxt[j] as u32;
    }
    let (mut hit, mut m) = (0u64, 0u64);
    for j in half..n - k {
        let c = idx[j];
        if tot[c] == 0 { continue; }
        if (2 * cnt1[c] > tot[c]) as u8 == nxt[j] { hit += 1; }
        m += 1;
    }
    if m == 0 { return f64::NAN; }
    hit as f64 / m as f64
}
fn and_rate(a: &[u8], b: &[u8]) -> f64 {
    let n = a.len().min(b.len());
    if n < 30 { return f64::NAN; }
    (0..n).filter(|&i| a[i] == 1 && b[i] == 1).count() as f64 / n as f64
}
fn xor_rate(a: &[u8], b: &[u8]) -> f64 {
    let n = a.len().min(b.len());
    if n < 30 { return f64::NAN; }
    (0..n).filter(|&i| a[i] != b[i]).count() as f64 / n as f64
}
fn cond_edge(x: &[u8], y: &[u8]) -> f64 {
    let n = x.len().min(y.len());
    if n < 60 { return f64::NAN; }
    let (mut n1, mut h1, mut n0, mut h0) = (0u32, 0u32, 0u32, 0u32);
    for i in 0..n {
        if x[i] == 1 { n1 += 1; h1 += y[i] as u32; } else { n0 += 1; h0 += y[i] as u32; }
    }
    if n1 < 20 || n0 < 20 { return f64::NAN; }
    h1 as f64 / n1 as f64 - h0 as f64 / n0 as f64
}
fn delta_bits(v: &[f64]) -> Vec<u8> { v.windows(2).map(|w| (w[1] > w[0]) as u8).collect() }
fn delta_neg(v: &[f64]) -> Vec<u8> { v.windows(2).map(|w| (w[1] < w[0]) as u8).collect() }
fn sign_pos(v: &[f64]) -> Vec<u8> { v.iter().map(|&x| (x > 0.0) as u8).collect() }
fn quantile(v: &[f64], q: f64) -> f64 {
    let mut w = v.to_vec();
    let k = ((w.len() - 1) as f64 * q) as usize;
    let (_lo, e, _hi) = w.select_nth_unstable_by(k, |a, b| a.partial_cmp(b).unwrap());
    *e
}
fn binarize_med(v: &[f64]) -> Vec<u8> {
    if v.is_empty() { return Vec::new(); }
    let m = quantile(v, 0.5);
    v.iter().map(|&x| (x > m) as u8).collect()
}
fn binarize_q(v: &[f64], q: f64) -> Vec<u8> {
    if v.is_empty() { return Vec::new(); }
    let m = quantile(v, q);
    v.iter().map(|&x| (x > m) as u8).collect()
}
/// Δ²变号流：相邻方向位是否翻转
fn flip2(d: &[u8]) -> Vec<u8> {
    d.windows(2).map(|w| (w[0] != w[1]) as u8).collect()
}
fn sample_at(src_t: &[i64], src_v: &[u8], grid: &[i64]) -> Vec<u8> {
    let mut out = Vec::with_capacity(grid.len());
    let mut j = 0usize;
    for &t in grid {
        while j + 1 < src_t.len() && src_t[j + 1] <= t { j += 1; }
        out.push(if src_t[j] <= t { src_v[j] } else { 0 });
    }
    out
}

// ---------- 三元布尔函数（256 全枚举）的闭式统计 ----------
// 每个三元组先算 8 格联合：cnt, 目标=下一拍Δmid 的命中, 目标=下一拍Δimb 的命中, 8x8 转移
struct Cells {
    cnt: [u32; 8],
    tm: [u32; 8],   // tgt mid_up 命中
    ti: [u32; 8],   // tgt imbimp_up 命中
    jt: [u32; 64],  // 状态转移计数
    marg: [f64; 3], // 三路输入各自的 1 率
    n: usize,
}
fn cells_of(a: &[u8], b: &[u8], c: &[u8], tm_next: &[u8], ti_next: &[u8]) -> Cells {
    let n = a.len().min(b.len()).min(c.len()).min(tm_next.len()).min(ti_next.len());
    let mut cs = Cells { cnt: [0; 8], tm: [0; 8], ti: [0; 8], jt: [0; 64], marg: [0.; 3], n };
    if n == 0 { return cs; }
    let mut prev = 8usize;
    for i in 0..n {
        let cell = ((a[i] as usize) << 2) | ((b[i] as usize) << 1) | c[i] as usize;
        cs.cnt[cell] += 1;
        cs.tm[cell] += tm_next[i] as u32;
        cs.ti[cell] += ti_next[i] as u32;
        cs.marg[0] += a[i] as f64;
        cs.marg[1] += b[i] as f64;
        cs.marg[2] += c[i] as f64;
        if prev < 8 { cs.jt[prev * 8 + cell] += 1; }
        prev = cell;
    }
    for m in cs.marg.iter_mut() { *m /= n as f64; }
    cs
}
/// 统计类别：rate, ac1, leakmax, leakmax超额(乘积基线), edge_mid, edge_imb
fn tern_stat(cs: &Cells, func: usize, stat: usize) -> f64 {
    let n = cs.n;
    if n < 200 { return f64::NAN; }
    let nf = n as f64;
    let bit = |cell: usize| (func >> cell) & 1;
    match stat {
        0 => { // rate
            (0..8).map(|c| bit(c) as f64 * cs.cnt[c] as f64).sum::<f64>() / nf
        }
        1 => { // ac1：用 8x8 转移表算 z 序列的一阶自相关
            let p1 = (0..8).map(|c| bit(c) as f64 * cs.cnt[c] as f64).sum::<f64>() / nf;
            let npairs = (n - 1) as f64;
            let p11 = (0..8).flat_map(|c1| (0..8).map(move |c2| (c1, c2)))
                .filter(|&(c1, c2)| bit(c1) == 1 && bit(c2) == 1)
                .map(|(c1, c2)| cs.jt[c1 * 8 + c2] as f64).sum::<f64>() / npairs;
            let var = p1 * (1.0 - p1);
            if var <= 1e-12 { return f64::NAN; }
            (p11 - p1 * p1) / var
        }
        2 | 3 => { // leakmax / 超额（对三路各算 P(z==xi)，取最大）
            let mut mx = f64::MIN;
            let mut mxb = f64::MIN;
            for who in 0..3 {
                let mut lk = 0.0;
                let mut lkb = 0.0;
                for c in 0..8 {
                    let xbit = (c >> (2 - who)) & 1;
                    // 乘积边际基线下该格概率
                    let mut pb = 1.0;
                    for w in 0..3 {
                        let xb = (c >> (2 - w)) & 1;
                        let p = cs.marg[w];
                        pb *= if xb == 1 { p } else { 1.0 - p };
                    }
                    if bit(c) == xbit {
                        lk += cs.cnt[c] as f64 / nf;
                        lkb += pb;
                    }
                }
                if lk > mx { mx = lk; }
                if lkb > mxb { mxb = lkb; }
            }
            if stat == 2 { mx } else { mx - mxb }
        }
        4 | 5 => { // edge：|E[tgt|z=1] - E[tgt|z=0]|
            let tg = if stat == 4 { &cs.tm } else { &cs.ti };
            let (mut n1, mut h1, mut n0, mut h0) = (0u32, 0u32, 0u32, 0u32);
            for c in 0..8 {
                if bit(c) == 1 { n1 += cs.cnt[c]; h1 += tg[c]; }
                else { n0 += cs.cnt[c]; h0 += tg[c]; }
            }
            if n1 < 30 || n0 < 30 { return f64::NAN; }
            h1 as f64 / n1 as f64 - h0 as f64 / n0 as f64
        }
        _ => f64::NAN,
    }
}

// ---------- Spec 驱动：特征名与计算同序 ----------
#[derive(Clone)]
enum Op {
    Count,                 // i: 0=nsnap 1=ntrade
    Rate, Ac(usize), Run, Tasym, SpCv, Ctx(usize),
    Corr0, CorrL, AndR, XorR, Edge,
    Tern(usize),           // 三元 stat 类别
    TTRate,                // 真值表 8 格熵
    ClockRate, ClockAc,    // 钟控：mask[i] & src[i]
    HalfRate, HalfAc,      // 前后半日差异
    Sess(usize),           // 时段率：b=0 下午段, b=1 尾盘段
}
#[derive(Clone)]
struct Spec { name: String, op: Op, a: usize, b: usize, c: usize }

struct Env {
    seqs: Vec<Vec<u8>>,        // 盘口 L 网格上的流
    tseqs: Vec<Vec<u8>>,       // 逐笔网格上的流
    trips: Vec<Cells>,         // 三元组联合统计
    masks: Vec<Vec<u8>>,       // 钟控掩码（L 网格）
    tg_mid: Vec<u8>,
    tg_imb: Vec<u8>,
    segs: [Vec<Vec<u8>>; 2],   // [下午段, 尾盘段] 各条 seqs 的子序列
    nsnap: usize,
    ntrade: usize,
}

fn eval(sp: &Spec, e: &Env) -> f64 {
    match &sp.op {
        Op::Count => if sp.a == 0 { e.nsnap as f64 } else { e.ntrade as f64 },
        Op::Rate => rate(&e.seqs[sp.a]),
        Op::Ac(k) => ac_lag(&e.seqs[sp.a], *k),
        Op::Run => run_mean(&e.seqs[sp.a]),
        Op::Tasym => trans_asym(&e.seqs[sp.a]),
        Op::SpCv => spacing_cv(&e.seqs[sp.a]),
        Op::Ctx(k) => ctx_acc(&e.seqs[sp.a], *k),
        Op::Corr0 => corr_bits(&e.seqs[sp.a], &e.seqs[sp.b]),
        Op::CorrL => ac_ab(&e.seqs[sp.a], &e.seqs[sp.b], 1),
        Op::AndR => and_rate(&e.seqs[sp.a], &e.seqs[sp.b]),
        Op::XorR => xor_rate(&e.seqs[sp.a], &e.seqs[sp.b]),
        Op::Edge => cond_edge(&e.seqs[sp.a], &e.seqs[sp.b]),
        Op::Tern(stat) => tern_stat(&e.trips[sp.a], sp.b, *stat),
        Op::TTRate => { // 三元组的 8 格模式熵
            let cs = &e.trips[sp.a];
            if cs.n < 200 { return f64::NAN; }
            let nf = cs.n as f64;
            - (0..8).filter(|&c| cs.cnt[c] > 0)
                .map(|c| { let f = cs.cnt[c] as f64 / nf; f * f.ln() }).sum::<f64>()
        }
        Op::ClockRate | Op::ClockAc => {
            let m = &e.masks[sp.a];
            let s = &e.seqs[sp.b];
            let n = m.len().min(s.len());
            let sub: Vec<u8> = (0..n).filter(|&i| m[i] == 1).map(|i| s[i]).collect();
            if matches!(sp.op, Op::ClockRate) { rate(&sub) } else { ac_lag(&sub, 1) }
        }
        Op::HalfRate | Op::HalfAc => {
            let s = &e.seqs[sp.a];
            let h = s.len() / 2;
            if h < 100 { return f64::NAN; }
            if matches!(sp.op, Op::HalfRate) { rate(&s[h..]) - rate(&s[..h]) }
            else { ac_lag(&s[h..], 1) - ac_lag(&s[..h], 1) }
        }
        Op::Sess(b) => rate(&e.segs[*b][sp.a]),
    }
}

const SN: &[&str] = &[
    "imb1p", "dimb1", "imbtp", "dimbt", "midu", "midd", "sprd", "spru",
    "depu", "bid1u", "ask1u", "tosp", "vosp", "imb5p", "dimb5",
    "c1sgn", "c1big", "flip",
    // v3 增补（对齐 py_geffe_leak 的字段覆盖）
    "tbidd", "taskd", "tbidq", "taskq", "imb1q", "imbtq", "midgt", "sprdq",
    "fimb", "fmid", "fsprd", "cpu", "cto", "agree", "simbt", "simb1", "bimb1", "imbq",
];
// 0 imb1p 1 dimb1 2 imbtp 3 dimbt 4 midu 5 midd 6 sprd 7 spru 8 depu
// 9 bid1u 10 ask1u 11 tosp 12 vosp 13 imb5p 14 dimb5 15 c1sgn 16 c1big 17 flip
// 18 Δ总买量↑ 19 Δ总卖量↑ 20 总买量>q75 21 总卖量>q75 22 一档不平衡>q75 23 总不平衡>q75
// 24 中间价>中位 25 价差>q75 26 一档不平衡Δ²变号 27 中间价Δ²变号 28 价差Δ²变号
// 29 采样成交价↑ 30 采样成交额>中位 31 方向==盘口偏多(agree) 32 主买&总偏多 33 主买&一档偏多
// 34 大单&一档偏多 35 |一档不平衡|>q75
const TN: &[&str] = &[
    "tsgn", "tbig8", "tbig5", "tdpx", "tdvol", "tdto", "rgap", "turg", "tbb",
    "togt", "toq7", "vgt", "vq7", "ogt", "oq7", "dgt", "dq7",
];
// 9 成交额>中位 10 成交额>q75 11 成交量>中位 12 成交量>q75
// 13 volume%100>中位(碎单) 14 volume%100>q75 15 |卖单号-买单号|>中位 16 |单号差|>q75
// 注：原 tddt(时间间隔缩短)恒为0是死流，第6位改为 rgap=间隔>中位
const TRIPS: &[(usize, usize, usize)] = &[
    (1, 4, 0),   // T0: Δimb1, Δmid↑, imb1偏多      —— 原 g1/mj 组
    (15, 4, 0),  // T1: 成交方向@快照, Δmid↑, imb1偏多 —— 原 gc 组
    (9, 10, 0),  // T2: Δbid1量↑, Δask1量↑, imb1偏多  —— 两侧挂单的攻防
    (8, 4, 6),   // T3: Δ深度↑, Δmid↑, 价差收窄      —— 深度×价格×价差
    (3, 11, 2),  // T4: Δ总不平衡↑, 成交速度↑, 总偏多  —— 量价速
    (14, 13, 4), // T5: Δ5档不平衡↑, 5档偏多, Δmid↑   —— 深档结构
    (16, 15, 4), // T6: 大单@快照, 方向@快照, Δmid↑   —— 成交发起方结构
    // v3 增补（对齐 py_geffe_leak 的域）
    (31, 4, 2),  // T7: agree(方向==总偏多), Δmid↑, 总偏多 —— 同边信号
    (18, 19, 23),// T8: Δ总买量, Δ总卖量, 总挂单q75 —— 总深度攻防
    (30, 29, 31),// T9: 采样成交额, 采样价↑, agree —— 采样三元
];
const TERN_STATS: &[&str] = &["rate", "ac1", "lk", "lkx", "edgeM", "edgeI"];
const PAIRS: &[(usize, usize)] = &[
    (1, 4), (1, 0), (4, 0), (6, 4), (8, 1), (4, 8), (15, 4), (15, 0),
    (16, 15), (1, 8), (3, 11), (9, 10), (14, 4), (2, 4), (5, 4), (7, 6),
];
const UNARY: &[usize] = &[0, 1, 2, 3, 4, 6, 8, 9, 10, 11, 13, 14, 15, 16, 17];
// v3 新增流上轻量算子（rate/ac1/run/ctx6）
const UNARY_NEW: &[usize] = &[18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35];
const UNARY_T: &[usize] = &[0, 1, 3, 6, 7]; // tseqs 下标：tsgn,tbig8,tdpx,rgap,turg
const UNARY_TNEW: &[usize] = &[9, 10, 11, 12, 13, 14, 15, 16];

fn build_specs() -> Vec<Spec> {
    let mut v = Vec::with_capacity(12000);
    let mut push = |name: String, op: Op, a: usize, b: usize, c: usize| {
        v.push(Spec { name, op, a, b, c });
    };
    push("nsnap".into(), Op::Count, 0, 0, 0);
    push("ntrade".into(), Op::Count, 1, 0, 0);
    // 一元：盘口流
    for &i in UNARY {
        let s = SN[i];
        push(format!("u_{s}_rate"), Op::Rate, i, 0, 0);
        for k in [1usize, 5, 10] { push(format!("u_{s}_ac{k}"), Op::Ac(k), i, 0, 0); }
        push(format!("u_{s}_run"), Op::Run, i, 0, 0);
        push(format!("u_{s}_tasym"), Op::Tasym, i, 0, 0);
        push(format!("u_{s}_spcv"), Op::SpCv, i, 0, 0);
        push(format!("u_{s}_ctx6"), Op::Ctx(6), i, 0, 0);
    }
    // 一元：逐笔流（存进 seqs 后段，eval 里统一按 seqs 索引 → 记录偏移在 env 构造时对齐）
    // —— 逐笔流单独编号：specs 用 a = 100 + tidx，eval 特判
    for &i in UNARY_T {
        let s = TN[i];
        push(format!("u_{s}_rate"), Op::Rate, 100 + i, 0, 0);
        push(format!("u_{s}_ac1"), Op::Ac(1), 100 + i, 0, 0);
        push(format!("u_{s}_run"), Op::Run, 100 + i, 0, 0);
        push(format!("u_{s}_ctx6"), Op::Ctx(6), 100 + i, 0, 0);
    }
    // v3 新增盘口流：轻量四算子
    for &i in UNARY_NEW {
        let s = SN[i];
        push(format!("u_{s}_rate"), Op::Rate, i, 0, 0);
        push(format!("u_{s}_ac1"), Op::Ac(1), i, 0, 0);
        push(format!("u_{s}_run"), Op::Run, i, 0, 0);
        push(format!("u_{s}_ctx6"), Op::Ctx(6), i, 0, 0);
    }
    // v3 新增逐笔流：轻量四算子
    for &i in UNARY_TNEW {
        let s = TN[i];
        push(format!("u_{s}_rate"), Op::Rate, 100 + i, 0, 0);
        push(format!("u_{s}_ac1"), Op::Ac(1), 100 + i, 0, 0);
        push(format!("u_{s}_run"), Op::Run, 100 + i, 0, 0);
        push(format!("u_{s}_ctx6"), Op::Ctx(6), 100 + i, 0, 0);
    }
    // 时段率：下午段/尾盘段在核心流上的率
    for (b, tag) in [(0usize, "pm"), (1, "tl")] {
        for &i in &[1usize, 0, 15, 4, 31] {
            push(format!("sp_{tag}_{}_rate", SN[i]), Op::Sess(b), i, 0, 0);
        }
    }
    // 二元
    for &(i, j) in PAIRS {
        let (a, b) = (SN[i], SN[j]);
        push(format!("b_{a}_{b}_corr"), Op::Corr0, i, j, 0);
        push(format!("b_{a}_{b}_lead"), Op::CorrL, i, j, 0);
        push(format!("b_{a}_{b}_and"), Op::AndR, i, j, 0);
        push(format!("b_{a}_{b}_xor"), Op::XorR, i, j, 0);
        push(format!("b_{a}_{b}_edge"), Op::Edge, i, j, 0);
    }
    // 三元 256 全枚举
    for (ti, _) in TRIPS.iter().enumerate() {
        for f in 0..256usize {
            for (si, st) in TERN_STATS.iter().enumerate() {
                push(format!("T{ti}_f{f:03}_{st}"), Op::Tern(si), ti, f, 0);
            }
        }
        push(format!("tt_{ti}_pattH"), Op::TTRate, ti, 0, 0);
    }
    // 钟控：masks 0=大单@快照 1=mid移动 2=imb翻转 3=价差收窄 4=深度增加
    const CLOCK_SRC: &[usize] = &[1, 0, 15]; // dimb1, imb1p, c1sgn
    const MASKN: &[&str] = &["big", "mv", "fl", "sp", "dp"];
    for (mi, mn) in MASKN.iter().enumerate() {
        for &si in CLOCK_SRC {
            push(format!("ck_{mn}_{}_rate", SN[si]), Op::ClockRate, mi, si, 0);
            push(format!("ck_{mn}_{}_ac1", SN[si]), Op::ClockAc, mi, si, 0);
        }
    }
    // 半日差异
    for &i in &[1usize, 0, 15, 4] {
        push(format!("hf_{}_rate", SN[i]), Op::HalfRate, i, 0, 0);
        push(format!("hf_{}_ac1", SN[i]), Op::HalfAc, i, 0, 0);
    }
    v
}

fn build_env(mkt: &[MarketRecord], tr: &[TradeRecord]) -> Option<Env> {
    if mkt.len() < 300 || tr.len() < 500 { return None; }
    let mut ts: Vec<i64> = Vec::with_capacity(mkt.len());
    let (mut imb1, mut imbt, mut mid, mut spread, mut dep, mut b1v, mut a1v, mut imb5): (
        Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>
    ) = (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let mut to_v: Vec<f64> = Vec::new();
    let mut tbid: Vec<f64> = Vec::new();
    let mut task: Vec<f64> = Vec::new();
    for s in mkt {
        let (b, a) = (s.total_bid_vol, s.total_ask_vol);
        let (b1, a1) = (s.bid_vols[0], s.ask_vols[0]);
        let (bp, ap) = (s.bid_prc1, s.ask_prc1);
        if b + a <= 0.0 || b1 + a1 <= 0.0 || bp <= 0.0 || ap <= 0.0 || ap <= bp { continue; }
        ts.push(s.time_us);
        imb1.push((b1 - a1) / (b1 + a1));
        imbt.push((b - a) / (b + a));
        mid.push((ap + bp) / 2.0);
        spread.push(ap - bp);
        dep.push(b1 + a1);
        b1v.push(b1); a1v.push(a1);
        to_v.push(s.turnover);
        tbid.push(b);
        task.push(a);
        let b5: f64 = s.bid_vols[..5].iter().sum();
        let a5: f64 = s.ask_vols[..5].iter().sum();
        imb5.push((b5 - a5) / (b5 + a5 + 1e-9));
    }
    if ts.len() < 300 { return None; }
    let imb1p = sign_pos(&imb1);
    let dimb1 = delta_bits(&imb1);
    let imbtp = sign_pos(&imbt);
    let dimbt = delta_bits(&imbt);
    let midu = delta_bits(&mid);
    let midd = delta_neg(&mid);
    let sprd = delta_neg(&spread);
    let spru = delta_bits(&spread);
    let depu = delta_bits(&dep);
    let bid1u = delta_bits(&b1v);
    let ask1u = delta_bits(&a1v);
    // 成交速度流：相邻快照 Δturnover/Δt > 中位数
    let mut tospeed: Vec<f64> = Vec::with_capacity(ts.len() - 1);
    let mut dt_v: Vec<f64> = Vec::with_capacity(ts.len() - 1);
    for i in 1..ts.len() {
        let dt = (ts[i] - ts[i - 1]) as f64;
        dt_v.push(dt);
        tospeed.push((to_v[i] - to_v[i - 1]) / dt.max(1.0));
    }
    let med_sp = quantile(&tospeed, 0.5);
    let tosp: Vec<u8> = tospeed.iter().map(|&v| (v > med_sp) as u8).collect();
    let med_dt = quantile(&dt_v, 0.5);
    let vosp: Vec<u8> = dt_v.iter().map(|&v| (v < med_dt) as u8).collect(); // 快照更密=事件密集
    let imb5p = sign_pos(&imb5);
    let dimb5 = delta_bits(&imb5);
    // 翻转流：imb1p 在相邻快照间换边
    let flip: Vec<u8> = imb1p.windows(2).map(|w| (w[0] != w[1]) as u8).collect();
    // ---- 逐笔流 ----
    let tsgn: Vec<u8> = tr.iter().map(|t| (t.flag == 66) as u8).collect();
    let tos: Vec<f64> = tr.iter().map(|t| t.turnover).collect();
    let q80 = quantile(&tos, 0.8);
    let q50 = quantile(&tos, 0.5);
    let tbig8: Vec<u8> = tos.iter().map(|&v| (v > q80) as u8).collect();
    let tbig5: Vec<u8> = tos.iter().map(|&v| (v > q50) as u8).collect();
    let tdpx: Vec<u8> = tr.windows(2).map(|w| (w[1].price > w[0].price) as u8).collect();
    let tdvol: Vec<u8> = tr.windows(2).map(|w| (w[1].volume > w[0].volume) as u8).collect();
    let tdto: Vec<u8> = tr.windows(2).map(|w| (w[1].turnover > w[0].turnover) as u8).collect();
    let gaps_f64: Vec<f64> = tr.windows(2).map(|w| (w[1].time_us - w[0].time_us) as f64).collect();
    let turg: Vec<u8> = tr.iter().map(|t| (t.ask_order > t.bid_order) as u8).collect();
    let tbb: Vec<u8> = (0..tr.len()).map(|i| (tsgn[i] == 1 && tbig8[i] == 1) as u8).collect();
    // 采样到盘口网格（用 L = len(ts)-1 对齐 Δ 网格：取 ts[1..] 时刻）
    let tr_t: Vec<i64> = tr.iter().map(|t| t.time_us).collect();
    let grid: Vec<i64> = ts[1..].to_vec();
    let c1sgn = sample_at(&tr_t, &tsgn, &grid);
    let c1big = sample_at(&tr_t, &tbig8, &grid);
    // ---- 全部对齐到 L = len-1 的 Δ 网格 ----
    // ---- v3 新增流（对齐 py_geffe_leak 的字段覆盖） ----
    let tbidd = delta_bits(&tbid);
    let taskd = delta_bits(&task);
    let tbidq = binarize_q(&tbid[1..], 0.75);
    let taskq = binarize_q(&task[1..], 0.75);
    let imb1q = binarize_q(&imb1[1..], 0.75);
    let imbtq = binarize_q(&imbt[1..], 0.75);
    let midgt = binarize_med(&mid[1..]);
    let sprdq = binarize_q(&spread[1..], 0.75);
    let fimb = flip2(&dimb1);
    let fmid = flip2(&midu);
    let fsprd = flip2(&sprd);
    let c2to = sample_at(&tr_t, &tbig5, &grid);           // 采样成交额>中位
    let tdpx_full: Vec<u8> = {
        let mut v = vec![0u8; tr.len()];
        for i in 1..tr.len() { v[i] = (tr[i].price > tr[i - 1].price) as u8; }
        v
    };
    let c_px = sample_at(&tr_t, &tdpx_full, &grid);       // 采样成交价↑
    let agree: Vec<u8> = (0..c1sgn.len()).map(|i| (c1sgn[i] == imbtp[i + 1]) as u8).collect();
    let simbt: Vec<u8> = (0..c1sgn.len()).map(|i| c1sgn[i] & imbtp[i + 1]).collect();
    let simb1: Vec<u8> = (0..c1sgn.len()).map(|i| c1sgn[i] & imb1p[i + 1]).collect();
    let bimb1: Vec<u8> = (0..c1sgn.len()).map(|i| c1big[i] & imb1p[i + 1]).collect();
    let imb1absq = binarize_q(&imb1[1..].iter().map(|v| v.abs()).collect::<Vec<_>>(), 0.75);

    let seqs: Vec<Vec<u8>> = vec![
        imb1p[1..].to_vec(), dimb1, imbtp[1..].to_vec(), dimbt,
        midu.clone(), midd, sprd, spru, depu,
        bid1u, ask1u, tosp, vosp, imb5p[1..].to_vec(), dimb5,
        c1sgn.clone(), c1big.clone(), flip,
        tbidd, taskd, tbidq, taskq, imb1q, imbtq, midgt, sprdq,
        fimb, fmid, fsprd, c_px, c2to, agree, simbt, simb1, bimb1, imb1absq,
    ];
    let togt = binarize_med(&tos);
    let toq7 = binarize_q(&tos, 0.75);
    let vols: Vec<f64> = tr.iter().map(|t| t.volume).collect();
    let vgt = binarize_med(&vols);
    let vq7 = binarize_q(&vols, 0.75);
    let o100: Vec<f64> = tr.iter().map(|t| t.volume % 100.0).collect();
    let ogt = binarize_med(&o100);
    let oq7 = binarize_q(&o100, 0.75);
    let ad: Vec<f64> = tr.iter().map(|t| (t.ask_order - t.bid_order).abs() as f64).collect();
    let dgt = binarize_med(&ad);
    let dq7 = binarize_q(&ad, 0.75);
    let tseqs: Vec<Vec<u8>> = vec![tsgn, tbig8, tbig5, tdpx, tdvol, tdto,
        binarize_med(&gaps_f64), turg, tbb,
        togt, toq7, vgt, vq7, ogt, oq7, dgt, dq7];
    let tg_mid: Vec<u8> = seqs[4][1..].to_vec();      // 下一拍 midu
    let tg_imb: Vec<u8> = seqs[1][1..].to_vec();      // 下一拍 dimb1
    let trips: Vec<Cells> = TRIPS.iter().map(|&(ia, ib, ic)| {
        cells_of(&seqs[ia][..seqs[ia].len() - 1], &seqs[ib][..seqs[ib].len() - 1],
                 &seqs[ic][..seqs[ic].len() - 1], &tg_mid, &tg_imb)
    }).collect();
    // 钟控掩码（L-1 网格）
    let masks: Vec<Vec<u8>> = vec![
        seqs[16][..seqs[16].len() - 1].to_vec(),
        seqs[4][..seqs[4].len() - 1].to_vec(),
        seqs[17][..seqs[17].len() - 1].to_vec(),
        seqs[6][..seqs[6].len() - 1].to_vec(),
        seqs[8][..seqs[8].len() - 1].to_vec(),
    ];
    // 时段子序列：对齐后时钟 11:30-14:57（adjust_afternoon 把下午前移 90 分钟）
    // segs[0]=下午段(时钟>=11:30), segs[1]=尾盘段(时钟>=14:27)
    let mk_seg = |lo_min: i64| -> Vec<Vec<u8>> {
        seqs.iter().map(|s| {
            let mut o = Vec::new();
            for i in 0..s.len().min(grid.len()) {
                let tod = (grid[i] / 1_000_000).rem_euclid(86400);
                if tod >= lo_min * 60 { o.push(s[i]); }
            }
            o
        }).collect()
    };
    let segs = [mk_seg(11 * 60 + 30), mk_seg(14 * 60 + 27)];
    Some(Env { seqs, tseqs, trips, masks, tg_mid, tg_imb, segs, nsnap: ts.len(), ntrade: tr.len() })
}

fn eval_env(sp: &Spec, e: &Env) -> f64 {
    if sp.a >= 100 { // 逐笔网格流
        let s = &e.tseqs[sp.a - 100];
        return match &sp.op {
            Op::Rate => rate(s),
            Op::Ac(k) => ac_lag(s, *k),
            Op::Run => run_mean(s),
            Op::Ctx(k) => ctx_acc(s, *k),
            _ => f64::NAN,
        };
    }
    eval(sp, e)
}


// ---------- 对外入口 ----------

/// 固定输出长度（由 build_specs 决定，改动会破坏已有 store 的列名对齐）
pub const OUT_LEN: usize = 15744;

fn specs() -> &'static [Spec] {
    static S: OnceLock<Vec<Spec>> = OnceLock::new();
    S.get_or_init(build_specs)
}

/// 列名与输出严格对齐（进程内只算一次）
pub fn lfsr_leak_ext1_names() -> Vec<String> {
    static N: OnceLock<Vec<String>> = OnceLock::new();
    N.get_or_init(|| specs().iter().map(|s| s.name.clone()).collect()).clone()
}

/// 单股单日核心：读盘 → 建位流 → 算全部特征。
/// 输出恒为 OUT_LEN 个 f32；盘口不足 300 行或逐笔不足 500 笔时整行 NaN。
pub fn compute_lfsr_leak_ext1_full(code: &str, date: i64) -> std::io::Result<Vec<f32>> {
    let mkt = read_market_fast(code, date)?;
    let tr = read_trade_fast(code, date)?;
    Ok(compute_lfsr_leak_ext1_from_records(&mkt, &tr))
}

/// 无 IO 内核：读盘版与传数组版都走这里，保证两入口一致。
pub fn compute_lfsr_leak_ext1_from_records(
    mkt: &[MarketRecord],
    tr: &[TradeRecord],
) -> Vec<f32> {
    match build_env(mkt, tr) {
        Some(env) => {
            let out: Vec<f32> = specs().iter().map(|sp| eval_env(sp, &env) as f32).collect();
            debug_assert_eq!(out.len(), OUT_LEN, "lfsr_leak_ext1 列数必须恒定");
            out
        }
        None => vec![f32::NAN; OUT_LEN],
    }
}

/// 单股单日调试入口（错误抛异常，可见完整栈）
#[pyfunction]
pub fn py_lfsr_leak_ext1(code: &str, date: i64) -> PyResult<Vec<f32>> {
    compute_lfsr_leak_ext1_full(code, date)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("{e:?}")))
}

#[pyfunction]
pub fn py_lfsr_leak_ext1_names() -> Vec<String> {
    lfsr_leak_ext1_names()
}
