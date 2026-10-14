//! 订单号坐标轴探索（sandbox v6 定稿）。
//! 按「被动订单号」把逐笔成交聚合成订单对象，再按「主动订单号 + 时刻」聚合成撮合批次。
//! 实证（20220819 全市场，4734 只）：
//!   · 同一订单号出现在多个撮合时刻的比例 = 0（订单对象都是「一个瞬间、一次打光」）
//!   · 主动订单平均被拆成 1.59 笔成交，跨过约 1.12 个价位
//!   · 被动订单里只有 23% 会被多笔成交吃掉（一单对多人）
//! 构建: cd /home/chenzongwei/rust_pyfunc/sandbox_oidflow && cargo build --release
//! 运行: ./target/release/oidflow_sandbox 20220819 > out_v6_20220819.json
mod fast_csv_reader;
use fast_csv_reader::{read_trade_fast, TradeRecord};
use rayon::prelude::*;
use serde::Serialize;

// diag: [n_fill, n_order, n_multi_order, n_batch, span_min, share_multi_event, mean_touches]
const NAMES: [&str; 7] = [
    "fa_frag",     // 一个被动订单平均被几笔成交吃掉（全市场约 2.2）
    "fa_frag_do",  // 同上（订单等权）
    "fa_fragcv",   // 各被动订单被吃笔数的变异系数
    "fa_rep5m",    // 10% 分位量以上的被动订单里，5 分钟内被再次吃掉的比例
    "fa_topsz",    // 成交笔数最多的 10% 被动订单，占全部成交笔数的比例
    "fa_batch",    // 一个主动订单平均被拆成几笔成交
    "fa_levels",   // 主动订单跨过的价格档位数 / 其成交笔数
];

#[derive(Serialize)]
struct Out {
    names: Vec<String>,
    codes: Vec<String>,
    vals: Vec<Vec<f32>>,
    diag: Vec<Vec<f32>>,
}

#[derive(Clone, Copy)]
struct Fill {
    t: f64,
    p: f64,
    v: f64,
    oid: i64,
}

fn stock_codes_for_day(
    dir: &std::path::Path,
    symbol_map: &std::path::Path,
    date: i64,
    subdir: &str,
) -> std::io::Result<Vec<String>> {
    use std::io::{Error, ErrorKind};
    let invalid = |e: &str| Error::new(ErrorKind::InvalidData, e.to_string());
    let text = std::fs::read_to_string(symbol_map).map_err(|e| invalid(&e.to_string()))?;
    let mut lines = text.lines();
    let header = lines.next().ok_or_else(|| invalid("股票表为空"))?;
    let col = header
        .split(',')
        .position(|h| h.trim() == "symbol")
        .ok_or_else(|| invalid("股票表缺少 symbol 列"))?;
    let mut stocks = std::collections::BTreeSet::new();
    for line in lines {
        let code = line.split(',').nth(col).unwrap_or("").trim();
        if code.is_empty() {
            continue;
        }
        if code.len() > 6 || !code.bytes().all(|b| b.is_ascii_digit()) {
            return Err(invalid("股票表含无效代码"));
        }
        stocks.insert(format!("{code:0>6}"));
    }
    if stocks.is_empty() || !dir.is_dir() {
        return Err(invalid("股票白名单为空或原始数据目录不存在"));
    }
    Ok(stocks
        .into_iter()
        .filter(|code| dir.join(format!("{code}_{date}_{subdir}.csv")).is_file())
        .collect())
}

fn list_codes(date: i64) -> std::io::Result<Vec<String>> {
    let root = std::env::var("RUST_PYFUNC_LEVEL2_ROOT").unwrap_or("/ssd_data/stock".into());
    let basic =
        std::env::var("RUST_PYFUNC_BASIC_INFO_DIR").unwrap_or("/ssd_data/data/basic_info".into());
    let dir = std::path::Path::new(&root).join(date.to_string()).join("transaction");
    let map = std::path::Path::new(&basic).join("symbol_map.csv");
    stock_codes_for_day(&dir, &map, date, "transaction")
}

fn mean(x: &[f64]) -> f64 {
    if x.is_empty() { return f64::NAN; }
    x.iter().sum::<f64>() / x.len() as f64
}

fn std(x: &[f64]) -> f64 {
    if x.len() < 2 { return f64::NAN; }
    let m = mean(x);
    (x.iter().map(|v| (v - m).powi(2)).sum::<f64>() / (x.len() as f64 - 1.0)).sqrt()
}

fn fill_series(trades: &[TradeRecord]) -> Vec<Fill> {
    let mut out: Vec<Fill> = Vec::with_capacity(trades.len());
    for r in trades {
        let buy = r.flag == 66;
        if !buy && r.flag != 83 {
            continue;
        }
        let oid = if buy { r.bid_order } else { r.ask_order };
        if oid <= 0 || !(r.volume > 0.0) {
            continue;
        }
        out.push(Fill { t: r.time_sec, p: r.price, v: r.volume, oid });
    }
    out.sort_by(|a, b| a.t.partial_cmp(&b.t).unwrap());
    out
}

fn per_stock(trades: &[TradeRecord]) -> Option<([f64; 7], [f64; 7])> {
    let fills = fill_series(trades);
    if fills.len() < 300 {
        return None;
    }
    let n = fills.len();

    // ① 按被动订单号聚合
    let mut pairs: Vec<(i64, usize)> = fills.iter().enumerate().map(|(i, f)| (f.oid, i)).collect();
    pairs.sort_by_key(|x| x.0);

    let mut frag_vw_num = 0.0f64;
    let mut frag_vw_den = 0.0f64;
    let mut frag_do: Vec<f64> = Vec::new();
    let mut counts: Vec<(f64, f64)> = Vec::new(); // (被吃笔数, 成交总量)
    let mut n_order = 0.0f64;
    let mut n_share_multi = 0.0f64;
    let mut _touches = 0.0f64;
    // 重复吃单候选（按量）存 (max_gap_sec, 成交量)
    let mut touches_vol: Vec<(f64, f64)> = Vec::new();

    let mut i = 0;
    while i < pairs.len() {
        let mut j = i + 1;
        while j < pairs.len() && pairs[j].0 == pairs[i].0 {
            j += 1;
        }
        let (mut lo, mut hi) = (pairs[i].1, pairs[i].1);
        for t in i..j {
            lo = lo.min(pairs[t].1);
            hi = hi.max(pairs[t].1);
        }
        let k = (hi - lo + 1) as f64;
        let mut vol = 0.0f64;
        let mut ts: Vec<f64> = Vec::new();
        for x in lo..=hi {
            vol += fills[x].v;
            if ts.last().map(|p| *p != fills[x].t).unwrap_or(true) {
                ts.push(fills[x].t);
            }
        }
        if vol > 0.0 {
            frag_vw_num += k * vol;
            frag_vw_den += vol;
            frag_do.push(k);
            counts.push((k, vol));
            n_order += 1.0;
            _touches += ts.len() as f64;
            if ts.len() >= 2 {
                n_share_multi += 1.0;
            }
            let mut mg = 0.0f64;
            if ts.len() >= 2 {
                for w in 1..ts.len() {
                    mg = mg.max(ts[w] - ts[w - 1]);
                }
            }
            touches_vol.push((mg, vol));
        }
        i = j;
    }
    if n_order < 100.0 || frag_vw_den <= 0.0 {
        return None;
    }
    let frag_do_mean = mean(&frag_do);
    let frag_cv = std(&frag_do) / frag_do_mean.max(1e-9);

    // 重复吃单：10% 分位量以上、且 5 分钟内有再次被吃
    let mut vols: Vec<f64> = touches_vol.iter().map(|x| x.1).collect();
    vols.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let v_cut = vols[(vols.len() as f64 * 0.10) as usize];
    let (mut rep_all, mut rep_hit) = (0.0f64, 0.0f64);
    for &(mg, vol) in &touches_vol {
        if vol >= v_cut {
            rep_all += 1.0;
            if mg > 0.0 && mg <= 150.0 {
                rep_hit += 1.0;
            }
        }
    }

    // 集中度：笔数最多的 10% 订单占全部笔数
    let total_k: f64 = counts.iter().map(|x| x.0).sum();
    let mut ks: Vec<f64> = counts.iter().map(|x| x.0).collect();
    ks.sort_by(|a, b| b.partial_cmp(a).unwrap());
    let topk = ((n_order * 0.10) as usize).max(1);
    let topsz: f64 = ks[..topk].iter().sum::<f64>() / total_k.max(1e-9);

    // ② 撮合批次：连续成交里时刻变或被动订单号变就开新批次
    let mut batches: Vec<(usize, usize)> = Vec::new();
    let mut s = 0usize;
    for i in 1..n {
        if fills[i].t != fills[i - 1].t || fills[i].oid != fills[i - 1].oid {
            batches.push((s, i));
            s = i;
        }
    }
    batches.push((s, n));
    let mut batch_sz: Vec<f64> = Vec::with_capacity(batches.len());
    let (mut lvl_num, mut lvl_den) = (0.0f64, 0.0f64);
    for &(a, b) in &batches {
        batch_sz.push((b - a) as f64);
        let mut lvls = 1usize;
        for i in a + 1..b {
            if fills[i].p != fills[i - 1].p {
                lvls += 1;
            }
        }
        let vol: f64 = fills[a..b].iter().map(|f| f.v).sum();
        lvl_num += lvls as f64 * vol;
        lvl_den += vol;
    }

    let vals = [
        frag_vw_num / frag_vw_den.max(1e-9),
        frag_do_mean,
        frag_cv,
        rep_hit / rep_all.max(1e-9),
        topsz,
        mean(&batch_sz),
        lvl_num / lvl_den.max(1e-9),
    ];
    for x in vals.iter() {
        if !x.is_finite() {
            return None;
        }
    }
    let diag = [
        n as f64,
        n_order,
        n_share_multi,
        batches.len() as f64,
        (fills[n - 1].t - fills[0].t) / 60.0,
        n_share_multi / n_order,
        _touches / n_order,
    ];
    Some((vals, diag))
}

fn main() {
    let date: i64 = std::env::args()
        .nth(1)
        .expect("用法: oidflow_sandbox <date>")
        .parse()
        .expect("date 必须是整数");
    let codes = list_codes(date).expect("股票范围或原始目录检查失败");
    eprintln!("codes = {}", codes.len());

    let rows: Vec<(String, [f64; 7], [f64; 7])> = codes
        .par_iter()
        .filter_map(|c| {
            let trades = read_trade_fast(c, date).ok()?;
            let (v, d) = per_stock(&trades)?;
            Some((c.clone(), v, d))
        })
        .collect();

    let mut out = Out {
        names: NAMES.iter().map(|s| s.to_string()).collect(),
        codes: Vec::with_capacity(rows.len()),
        vals: Vec::with_capacity(rows.len()),
        diag: Vec::with_capacity(rows.len()),
    };
    for (c, v, d) in rows {
        out.codes.push(c);
        out.vals.push(v.iter().map(|&x| x as f32).collect());
        out.diag.push(d.iter().map(|&x| x as f32).collect());
    }
    eprintln!("valid = {}", out.codes.len());
    println!("{}", serde_json::to_string(&out).unwrap());
}
