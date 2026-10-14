//! 单股单日「撤单-挂单-成交」三流分解 + 封板时段撤买量对比（sandbox_cancel_flow）
//!
//! 逐笔成交文件带 flag=32 的撤单记录，撤单记录价格恒为 0，但带方向（ask_order / bid_order 二选一）。
//! 于是全股层面可以按时间戳把撤单和成交投进快照桶，做三条流的分解：
//!   新增买单 = Δ(total_bid_vol) + 撤买量 + 主动卖成交量(flag 83)
//!   新增卖单 = Δ(total_ask_vol) + 撤卖量 + 主动买成交量(flag 66)
//! flag 66 = 主买（成交价打在 ask1），flag 83 = 主卖（打在 bid1）。
//!
//! 另出封板时段的撤买量对比：封板判定 = last_prc 等于涨停价 且 ask_vol1 = 0。
//! 封板期间买盘全压在涨停价那一档，所以该时段的撤买量就是那一档的撤单量（不需要撤单价格）。
//!
//! 【涨停价口径有分期，改这支程序时不许绕开】
//! 快照里的 high_limited 沪深两市在 20250228 及以前都有值；沪市从 20250303 起全为 0（深市一直正常）。
//! 所以取第一张非零的，取不到先查外部对照表（环境变量 CF_LIMIT_TABLE 指的两列文件 code,high_limited），
//! 再取不到就标成未知并打警告，绝不退回「昨收乘比例」。ST 是 5%、科创板 20%、创业板 20%、北交所 30%，
//! 按 1.1 算会漏掉而且不报错。撤单面板那支程序（src/bin/panel.rs）不依赖涨停价。
//!
//! 【读取闸门三条，删掉就等于把静默错误放回来】
//! 一、文件本身有一百行以上数据、解析后却是空的，直接报错，不往下走。
//!     20250303 起部分文件把数值写成小数（如 32.0、1740964500130000.0），用整数严格解析会全失败，
//!     表现就是整个文件读出来是空的。所以整数一律先按小数解析再取整（见 pi）。
//! 二、逐只股票失败要打印原因，并统计失败几只、共几只。
//! 三、全部失败就退出码一，不产出任何结果，不留半成品。
//!
//! 构建: cd sandbox_cancel_flow && cargo build --release
//! 运行: ./target/release/cancel_flow_sandbox 20220819 000001,002209,... > out.json
//!       CF_LIMIT_TABLE=table.csv ./target/release/cancel_flow_sandbox 20250303 600000
//! 日志走 stderr，JSON 走 stdout。

use rayon::prelude::*;
use serde::Serialize;

#[derive(Clone, Copy, Default)]
struct T {
    t: i64,
    price: f64,
    vol: f64,
    flag: i32,
    bid_order: i64,
    ask_order: i64,
}

#[derive(Clone, Copy, Default)]
struct M {
    t: i64,
    last: f64,
    tot_bid: f64,
    tot_ask: f64,
    ask_p1: f64,
    ask_v1: f64,
    bid_p1: f64,
    bid_v1: f64,
    hl: f64,
    ll: f64,
    prev: f64,
}

fn in_session(us: i64) -> bool {
    let sec = ((us / 1_000_000) + 8 * 3600).rem_euclid(86400);
    (sec >= 9 * 3600 + 30 * 60 && sec <= 11 * 3600 + 30 * 60) || (sec >= 13 * 3600 && sec <= 15 * 3600)
}

// 20250303 起部分文件把数值写成小数（如 32.0、1740964500130000.0），整数直接 parse 会失败，
// 失败后所有行的时间都是 0，会被静默丢掉，整个文件读出来是空的。所以一律走小数再取整。
#[inline]
fn pi(s: &str) -> i64 { s.trim().parse::<f64>().ok().map(|v| v as i64).unwrap_or(0) }

fn load_trades(path: &str) -> std::io::Result<Vec<T>> {
    let content = std::fs::read_to_string(path)?;
    let raw = content.lines().count().saturating_sub(1);
    let mut out = Vec::with_capacity(content.len() / 120);
    for line in content.lines().skip(1) {
        let f: Vec<&str> = line.split(',').collect();
        if f.len() < 15 { continue; }
        let t: i64 = pi(f[4]);
        if t == 0 || !in_session(t) { continue; }
        out.push(T {
            t,
            price: f[7].parse().unwrap_or(0.0),
            vol: f[8].parse().unwrap_or(0.0),
            flag: pi(f[10]) as i32,
            ask_order: pi(f[13]),
            bid_order: pi(f[14]),
        });
    }
    // 读到 0 行就必须停下，不能静默往下走（数值写法变化时会表现为整文件解析为空）
    if out.is_empty() && raw > 100 {
        return Err(std::io::Error::new(std::io::ErrorKind::InvalidData,
            format!("有 {raw} 行数据但解析后为空，数值写法可能变了")));
    }
    Ok(out)
}

fn load_market(path: &str) -> std::io::Result<Vec<M>> {
    let content = std::fs::read_to_string(path)?;
    let raw = content.lines().count().saturating_sub(1);
    let mut out = Vec::with_capacity(content.len() / 600);
    for line in content.lines().skip(1) {
        let f: Vec<&str> = line.split(',').collect();
        if f.len() < 25 { continue; }
        let t: i64 = pi(f[4]);
        if t == 0 || !in_session(t) { continue; }
        let ask_p1: f64 = f[21].parse().unwrap_or(0.0);
        let bid_p1: f64 = f[23].parse().unwrap_or(0.0);
        if ask_p1 == 0.0 && bid_p1 == 0.0 { continue; }
        out.push(M {
            t,
            last: f[6].parse().unwrap_or(0.0),
            prev: f[8].parse().unwrap_or(0.0),
            hl: f[12].parse().unwrap_or(0.0),
            ll: f[13].parse().unwrap_or(0.0),
            tot_ask: f[19].parse().unwrap_or(0.0),
            tot_bid: f[20].parse().unwrap_or(0.0),
            ask_p1,
            ask_v1: f[22].parse().unwrap_or(0.0),
            bid_p1,
            bid_v1: f[24].parse().unwrap_or(0.0),
        });
    }
    if out.is_empty() && raw > 100 {
        return Err(std::io::Error::new(std::io::ErrorKind::InvalidData,
            format!("有 {raw} 行数据但解析后为空，数值写法可能变了")));
    }
    Ok(out)
}

#[derive(Serialize)]
struct StockOut {
    code: String,
    n_mkt: usize,
    n_trade: usize,
    n_intervals: usize,
    secs_covered: f64,
    cancel_bid_vol: f64,
    cancel_ask_vol: f64,
    active_buy_vol: f64,
    active_sell_vol: f64,
    implied_new_bid: f64,
    implied_new_ask: f64,
    neg_bid_n: usize,
    neg_ask_n: usize,
    neg_bid_pct: f64,
    neg_ask_pct: f64,
    neg_bid_vol: f64,
    neg_ask_vol: f64,
    neg_bid_time: Vec<i64>,
    neg_bid_dt: Vec<i64>,
    all_dt: Vec<i64>,
    seal_secs: f64,
    seal_cancel_bid: f64,
    seal_trade_vol: f64,
    open_secs: f64,
    open_cancel_bid: f64,
    open_trade_vol: f64,
    seal_cb_per_sec: f64,
    seal_v66: f64,
    seal_v83: f64,
    seal_iv: usize,
    seal_zero_sell: usize,
    clear_p25: f64,
    clear_p50: f64,
    clear_p75: f64,
    #[serde(skip)]
    clear_times: Vec<f64>,
    seal_cancel_bid_share: f64,
    open_cb_per_sec: f64,
    seal_break_n: usize,
    iv_seal: usize,
    iv_open: usize,
    neg_bid_n_seal: usize,
    neg_bid_n_open: usize,
    neg_bid_pct_seal: f64,
    neg_bid_pct_open: f64,
    seal_cb_open_ratio: f64,
    open_cb_open_ratio: f64,
    hl: f64,
    prev: f64,
    hl_source: String,
    limit_price_unknown: bool,
}

fn one(code: &str, date: i64, table: &std::collections::HashMap<String, f64>) -> Result<StockOut, String> {
    let tx = format!("/ssd_data/stock/{}/{}/{}", date, "transaction", format!("{}_{}_transaction.csv", code, date));
    let md = format!("/ssd_data/stock/{}/{}/{}", date, "market_data", format!("{}_{}_market_data.csv", code, date));
    let trades = load_trades(&tx).map_err(|e| format!("{code} 逐笔: {e}"))?;
    let mkt_all = load_market(&md).map_err(|e| format!("{code} 快照: {e}"))?;
    let stride: usize = std::env::var("CF_STRIDE").ok().and_then(|s| s.parse().ok()).unwrap_or(1);
    // stride 用来把快照按更大的间隔合并，看残差是不是快照抖动造成的
    let mkt: Vec<M> = if stride > 1 { mkt_all.into_iter().step_by(stride).collect() } else { mkt_all };
    if mkt.len() < 3 { return Err(format!("{code} 快照有效行不足 3")); }

    let mut o = StockOut {
        code: code.to_string(), n_mkt: mkt.len(), n_trade: trades.len(), n_intervals: 0,
        secs_covered: 0.0, cancel_bid_vol: 0.0, cancel_ask_vol: 0.0, active_buy_vol: 0.0,
        active_sell_vol: 0.0, implied_new_bid: 0.0, implied_new_ask: 0.0, neg_bid_n: 0,
        neg_ask_n: 0, neg_bid_pct: 0.0, neg_ask_pct: 0.0, neg_bid_vol: 0.0, neg_ask_vol: 0.0, neg_bid_time: Vec::new(), neg_bid_dt: Vec::new(), all_dt: Vec::new(),
        seal_secs: 0.0, seal_cancel_bid: 0.0, seal_trade_vol: 0.0, open_secs: 0.0,
        open_cancel_bid: 0.0, open_trade_vol: 0.0, seal_cb_per_sec: 0.0, open_cb_per_sec: 0.0,
        seal_v66: 0.0, seal_v83: 0.0, seal_cancel_bid_share: 0.0,
        seal_iv: 0, seal_zero_sell: 0, clear_p25: 0.0, clear_p50: 0.0, clear_p75: 0.0, clear_times: Vec::new(),
        iv_seal: 0, iv_open: 0, neg_bid_n_seal: 0, neg_bid_n_open: 0,
        neg_bid_pct_seal: 0.0, neg_bid_pct_open: 0.0, seal_cb_open_ratio: 0.0, open_cb_open_ratio: 0.0,
        seal_break_n: 0, hl: 0.0, prev: 0.0, hl_source: String::new(), limit_price_unknown: false,
    };
    // 涨停价口径分期：快照里的 high_limited 沪深两市在 20250228 及以前都有值，
    // 沪市从 20250303 起全为 0（深市一直正常）。所以取第一张非零的，取不到就先查外部对照表，
    // 再取不到就标成未知，绝不拿昨收乘比例顶替（ST 是 5%、科创板 20%，乘 1.1 会漏且不报错）。
    let mut hl = 0.0;
    for m in &mkt { if m.hl > 0.0 { hl = m.hl; break; } }
    let mut src = "snapshot";
    if hl <= 0.0 {
        match table.get(code) { Some(v) if *v > 0.0 => { hl = *v; src = "table"; }, _ => { src = "unknown"; } }
    }
    o.hl = hl;
    o.hl_source = src.to_string();
    o.limit_price_unknown = src == "unknown";
    o.prev = mkt[0].prev;

    let mut p = 0usize;
    let mut was_seal = false;
    for i in 1..mkt.len() {
        let (t0, t1) = (mkt[i - 1].t, mkt[i].t);
        let dt = t1 - t0;
        // 跳过午休、停牌等大间隔，避免把跨段时长记到某一种状态上
        if dt <= 0 || dt > 60_000_000 { continue; }
        let mut cb = 0.0; let mut cs = 0.0; let mut ab = 0.0; let mut asell = 0.0;
        while p < trades.len() && trades[p].t <= t0 { p += 1; }
        let mut q = p;
        while q < trades.len() && trades[q].t <= t1 {
            let tr = trades[q];
            if tr.flag == 32 {
                if tr.bid_order > 0 { cb += tr.vol; } else if tr.ask_order > 0 { cs += tr.vol; }
            } else if tr.flag == 66 { ab += tr.vol; } else if tr.flag == 83 { asell += tr.vol; }
            q += 1;
        }
        o.n_intervals += 1;
        o.secs_covered += dt as f64 / 1e6;
        o.cancel_bid_vol += cb;
        o.cancel_ask_vol += cs;
        o.active_buy_vol += ab;
        o.active_sell_vol += asell;

        let db = mkt[i].tot_bid - mkt[i - 1].tot_bid;
        let da = mkt[i].tot_ask - mkt[i - 1].tot_ask;
        let nb = db + cb + asell;
        let na = da + cs + ab;
        o.implied_new_bid += nb;
        o.implied_new_ask += na;
        o.all_dt.push(dt);
        if nb < 0.0 { o.neg_bid_n += 1; o.neg_bid_vol += -nb; o.neg_bid_time.push(t1); o.neg_bid_dt.push(dt); }
        if na < 0.0 { o.neg_ask_n += 1; o.neg_ask_vol += -na; }

        let is_seal = o.hl > 0.0 && (mkt[i].last - o.hl).abs() < 0.005 && mkt[i].ask_v1 == 0.0;
        if is_seal && !was_seal { o.seal_break_n += 1; }
        was_seal = is_seal;
        if is_seal {
            o.iv_seal += 1;
            if nb < 0.0 { o.neg_bid_n_seal += 1; }
            o.seal_secs += dt as f64 / 1e6;
            o.seal_cancel_bid += cb;
            o.seal_trade_vol += ab + asell;
            o.seal_v66 += ab;
            o.seal_v83 += asell;
            // 清空时间 = 该刻买一挂单量 / 该刻主动卖速率（每秒股数）
            o.seal_iv += 1;
            if asell <= 0.0 { o.seal_zero_sell += 1; } else {
                let rate = asell / (dt as f64 / 1e6);
                if rate > 0.0 { o.clear_times.push(mkt[i].bid_v1 / rate); }
            }
        } else {
            o.iv_open += 1;
            if nb < 0.0 { o.neg_bid_n_open += 1; }
            o.open_secs += dt as f64 / 1e6;
            o.open_cancel_bid += cb;
            o.open_trade_vol += ab + asell;
        }
    }
    if o.n_intervals > 0 {
        o.neg_bid_pct = 100.0 * o.neg_bid_n as f64 / o.n_intervals as f64;
        o.neg_ask_pct = 100.0 * o.neg_ask_n as f64 / o.n_intervals as f64;
    }
    if o.seal_secs > 0.0 {
        o.seal_cb_per_sec = o.seal_cancel_bid / o.seal_secs;
        o.seal_cancel_bid_share = o.seal_cancel_bid / (o.seal_cancel_bid + o.seal_trade_vol);
    }
    if o.open_secs > 0.0 { o.open_cb_per_sec = o.open_cancel_bid / o.open_secs; }
    if o.iv_seal > 0 { o.neg_bid_pct_seal = 100.0 * o.neg_bid_n_seal as f64 / o.iv_seal as f64; }
    if o.iv_open > 0 { o.neg_bid_pct_open = 100.0 * o.neg_bid_n_open as f64 / o.iv_open as f64; }
    o.seal_cb_open_ratio = o.seal_cancel_bid / (o.seal_cancel_bid + o.seal_trade_vol + 1e-9);
    o.open_cb_open_ratio = o.open_cancel_bid / (o.open_cancel_bid + o.open_trade_vol + 1e-9);
    if !o.clear_times.is_empty() {
        o.clear_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let q = |p: f64| o.clear_times[((o.clear_times.len() as f64 - 1.0) * p).round() as usize];
        o.clear_p25 = q(0.25); o.clear_p50 = q(0.5); o.clear_p75 = q(0.75);
    }
    Ok(o)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let date: i64 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(20220819);
    let codes: Vec<String> = args
        .get(2)
        .map(|s| s.split(',').map(|x| x.trim().to_string()).filter(|x| !x.is_empty()).collect())
        .unwrap_or_else(|| vec!["000001".to_string()]);

    // 可选的涨停价对照表，两列 code,high_limited（表里只有断点之后的沪市股票即可）
    let table: std::collections::HashMap<String, f64> = std::env::var("CF_LIMIT_TABLE").ok()
        .and_then(|p| std::fs::read_to_string(p).ok())
        .map(|t| t.lines().filter_map(|l| {
            let mut it = l.split(',');
            let c = it.next()?.trim().to_string();
            let v: f64 = it.next()?.trim().parse().ok()?;
            if c.len() == 6 { Some((c, v)) } else { None }
        }).collect())
        .unwrap_or_default();
    if !table.is_empty() { eprintln!("limit table loaded: {} codes", table.len()); }
    let results: Vec<(String, Result<StockOut, String>)> = codes.par_iter()
        .map(|c| (c.clone(), one(c, date, &table))).collect();
    let mut stocks: Vec<StockOut> = Vec::new();
    let mut fails = 0usize;
    for (_, r) in results {
        match r { Ok(s) => stocks.push(s), Err(e) => { fails += 1; if fails <= 10 { eprintln!("跳过 {e}"); } } }
    }
    if !codes.is_empty() && stocks.is_empty() {
        eprintln!("全部 {} 只都读失败，停下，不产出结果", codes.len());
        std::process::exit(1);
    }
    eprintln!("读取失败 {} 只 / 共 {} 只", fails, codes.len());
    stocks.sort_by(|a, b| a.code.cmp(&b.code));

    let unk = stocks.iter().filter(|s| s.limit_price_unknown).count();
    let from_snap = stocks.iter().filter(|s| s.hl_source == "snapshot").count();
    let from_tab = stocks.iter().filter(|s| s.hl_source == "table").count();
    eprintln!("date={} stocks={} 涨停价来源 快照={} 对照表={} 未知={}", date, stocks.len(), from_snap, from_tab, unk);
    if unk > 0 { eprintln!("警告 {} 只股票拿不到涨停价，这些票的封板段一律按未封板统计，结果是错的，请补对照表", unk); }
    println!("{}", serde_json::to_string_pretty(&stocks).unwrap());
}
