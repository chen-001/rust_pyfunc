//! 第三版读取器副本（沙箱）。和第一版的区别：
//!   1. 股票代码按数值比较（2016 年部分文件把 000001 写成 1）；
//!   2. 保留撤单（只看标记 32，不看价格）；
//!   3. 每条记录带“是否下午”标记，下午段据此排除 11:30:00 那一秒的上午成交。
//! 时间：只保留 09:30:00-11:30:00.999 与 13:00:00-14:57:00.999 两段连续竞价；
//! t 是距 09:30 的微秒数，下午已前移 90 分钟（13:00 对应 7200 秒）。
//! 底层文件打开、字节缓冲复用与页缓存释放统一委托 `crate::fast_csv_reader`。

const US: i64 = 1_000_000;
const DAY_US: i64 = 86_400 * US;
const OFFSET_US: i64 = 8 * 3600 * US;
const MORNING_START: i64 = (9 * 3600 + 30 * 60) * US;
const MORNING_END_SEC: i64 = 11 * 3600 + 30 * 60;
const AFTERNOON_START: i64 = 13 * 3600 * US;
const AFTERNOON_END_SEC: i64 = 14 * 3600 + 57 * 60;
const SHIFT: i64 = 90 * 60 * US;

#[derive(Clone, Copy, Debug)]
pub struct Ev {
    pub t: i64,
    pub pm: bool,
    pub flag: i32,
    pub price: f64,
    pub volume: f64,
    pub bid: i64,
    pub ask: i64,
    pub index: i64,
}

#[derive(Clone, Copy, Debug)]
pub struct Snap {
    pub t: i64,
    pub pm: bool,
    /// 买量合计 - 卖量合计
    pub lv: f32,
}



#[inline]
fn pi(bytes: &[u8]) -> i64 {
    let mut neg = false;
    let mut i = 0;
    while i < bytes.len() && bytes[i] == b' ' {
        i += 1;
    }
    if i < bytes.len() && (bytes[i] == b'-' || bytes[i] == b'+') {
        neg = bytes[i] == b'-';
        i += 1;
    }
    let mut v: i64 = 0;
    while i < bytes.len() {
        let c = bytes[i];
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

#[inline]
fn pf(bytes: &[u8]) -> f64 {
    let mut i = 0;
    let mut neg = false;
    if i < bytes.len() && bytes[i] == b'-' {
        neg = true;
        i += 1;
    } else if i < bytes.len() && bytes[i] == b'+' {
        i += 1;
    }
    let mut ip = 0.0f64;
    let mut any = false;
    while i < bytes.len() && bytes[i].is_ascii_digit() {
        ip = ip * 10.0 + (bytes[i] - b'0') as f64;
        i += 1;
        any = true;
    }
    let mut fr = 0.0f64;
    let mut sc = 1.0f64;
    if i < bytes.len() && bytes[i] == b'.' {
        i += 1;
        while i < bytes.len() && bytes[i].is_ascii_digit() {
            fr = fr * 10.0 + (bytes[i] - b'0') as f64;
            sc *= 10.0;
            i += 1;
            any = true;
        }
    }
    if i < bytes.len() && (bytes[i] == b'e' || bytes[i] == b'E') {
        return std::str::from_utf8(bytes).ok().and_then(|s| s.trim().parse::<f64>().ok()).unwrap_or(f64::NAN);
    }
    if !any {
        return f64::NAN;
    }
    let r = ip + fr / sc;
    if neg {
        -r
    } else {
        r
    }
}

/// 交易所时间（微秒）→ (距 09:30 的微秒, 是否下午)；不在两段连续竞价内返回 None
#[inline]
fn session_time(exch_us: i64) -> Option<(i64, bool)> {
    let tod = (exch_us + OFFSET_US).rem_euclid(DAY_US);
    let sec = tod / US;
    if sec >= AFTERNOON_START / US && sec <= AFTERNOON_END_SEC {
        Some((tod - MORNING_START - SHIFT, true))
    } else if tod >= MORNING_START && sec <= MORNING_END_SEC {
        Some((tod - MORNING_START, false))
    } else {
        None
    }
}

fn split<'a, const N: usize>(line: &'a [u8], out: &mut [&'a [u8]; N]) -> usize {
    let mut start = 0;
    let mut col = 0;
    for (i, &b) in line.iter().enumerate() {
        if b == b',' {
            if col < N {
                out[col] = &line[start..i];
            }
            col += 1;
            start = i + 1;
        }
    }
    if col < N {
        out[col] = &line[start..];
    }
    col + 1
}

fn lines(data: &[u8], mut f: impl FnMut(&[u8])) {
    let mut start = 0;
    for i in 0..data.len() {
        if data[i] == b'\n' {
            let l = &data[start..i];
            f(if l.last() == Some(&b'\r') { &l[..l.len() - 1] } else { l });
            start = i + 1;
        }
    }
    if start < data.len() {
        let l = &data[start..];
        f(if l.last() == Some(&b'\r') { &l[..l.len() - 1] } else { l });
    }
}

/// 逐笔成交 + 撤单（标记 32）。code 按数值比较。
pub fn read_events(code: &str, date: i64) -> std::io::Result<Vec<Ev>> {
    crate::fast_csv_reader::with_transaction_csv_bytes(code, date, |buf| {
        let code_num: i64 = code.parse().unwrap_or(-1);
        let mut out = Vec::with_capacity(buf.len() / 90 + 1);
        let mut first = true;
        lines(buf, |line| {
            if first {
                first = false;
                return;
            }
            if line.is_empty() {
                return;
            }
            let mut f: [&[u8]; 15] = [&[][..]; 15];
            split(line, &mut f);
            if pi(f[0]) != code_num || pi(f[2]) != date {
                return;
            }
            let Some((t, pm)) = session_time(pi(f[4])) else { return };
            out.push(Ev {
                t,
                pm,
                flag: if f[10].is_empty() { 0 } else { pi(f[10]) as i32 },
                price: pf(f[7]),
                volume: pf(f[8]),
                ask: pi(f[13]),
                bid: pi(f[14]),
                index: pi(f[6]),
            });
        });
        out
    })
}

/// 盘口快照：只取时间与买卖量合计。code 按数值比较。
pub fn read_snaps(code: &str, date: i64) -> std::io::Result<Vec<Snap>> {
    crate::fast_csv_reader::with_market_csv_bytes(code, date, |buf| {
        let code_num: i64 = code.parse().unwrap_or(-1);
        let mut out = Vec::with_capacity(buf.len() / 400 + 1);
        let mut first = true;
        lines(buf, |line| {
            if first {
                first = false;
                return;
            }
            if line.is_empty() {
                return;
            }
            let mut f: [&[u8]; 24] = [&[][..]; 24];
            split(line, &mut f);
            if pi(f[0]) != code_num || pi(f[2]) != date {
                return;
            }
            let ask1 = pf(f[21]);
            let bid1 = pf(f[23]);
            if ask1 == 0.0 || bid1 == 0.0 {
                return;
            }
            let ex = pi(f[4]);
            if ex == 0 {
                return;
            }
            let Some((t, pm)) = session_time(ex) else { return };
            out.push(Snap { t, pm, lv: (pf(f[20]) - pf(f[19])) as f32 });
        });
        out
    })
}
