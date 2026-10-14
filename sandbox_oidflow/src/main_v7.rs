//! 订单号坐标轴探索（sandbox v7）。
//! 在 v6 的基础上把 flag=32 的撤单行读回来，按「订单号」把撤单量和成交量放在同一个订单对象上。
//! 关键事实：这批数据里撤单行带订单号、带量，但 price 与 turnover 恒为 0，且有 flag=32 的行只出现在
//! 连续竞价与收盘竞价里；而同一订单号从不跨越多个撮合时刻（v6 实测 share_multi_event = 0）。
//! 真正空着的角度：一个订单的撤单量相对它自己的成交量是多少（订单级撤单率），现有库只有按桶汇总的撤单流。
//! 构建: cd /home/chenzongwei/rust_pyfunc/sandbox_oidflow && cargo build --release
//! 运行: ./target/release/oidflow_sandbox 20220819 > out_v7_20220819.json
mod fast_csv_reader;
use fast_csv_reader::{read_trade_fast, TradeRecord};
use rayon::prelude::*;
use serde::Serialize;

const NAMES: [&str; 5] = [
    "fa_cxl_share_do",  // 有过撤单的订单，撤单量占（撤单+成交）量的比例（订单等权）
    "fa_cxl_share_vw",  // 同上（按该订单总量加权）
    "fa_cxl_vol_share", // 全股票：撤单总量 /（撤单总量 + 成交总量）
    "fa_cxl_ratio_do",  // 有撤单订单里，撤单量 > 成交量 的比例
    "fa_cxl_frag",      // 一个订单平均被撤销几次
];

#[derive(Serialize)]
struct Out {
    names: Vec<String>,
    codes: Vec<String>,
    vals: Vec<Vec<f32>>,
    diag: Vec<Vec<f32>>,
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

fn per_stock(trades: &[TradeRecord]) -> Option<([f64; 5], [f64; 5])> {
    // 订单号 -> (成交笔数, 成交量, 撤单笔数, 撤单量)
    let mut m: std::collections::HashMap<i64, [f64; 4]> = std::collections::HashMap::new();
    let mut n_trade = 0.0f64;
    for r in trades {
        let buy = r.flag == 66;
        let sell = r.flag == 83;
        if !buy && !sell && r.flag != 32 {
            continue;
        }
        let oid = if buy || r.flag == 32 {
            if r.bid_order > 0 { r.bid_order } else { r.ask_order }
        } else {
            r.ask_order
        };
        if oid <= 0 || !(r.volume > 0.0) {
            continue;
        }
        let e = m.entry(oid).or_insert([0.0; 4]);
        if buy || sell {
            e[0] += 1.0;
            e[1] += r.volume;
            n_trade += 1.0;
        } else {
            e[2] += 1.0;
            e[3] += r.volume;
        }
    }
    if n_trade < 300.0 || m.is_empty() {
        return None;
    }
    let mut share: Vec<f64> = Vec::new();
    let mut wnum = 0.0f64;
    let mut wden = 0.0f64;
    let mut cxl_total = 0.0f64;
    let mut trade_total = 0.0f64;
    let mut frag: Vec<f64> = Vec::new();
    let mut heavy = 0.0f64;
    let mut n_cxl_order = 0.0f64;
    for (_, e) in m.iter() {
        trade_total += e[1];
        cxl_total += e[3];
        if e[2] > 0.0 {
            n_cxl_order += 1.0;
            let tot = e[1] + e[3];
            if tot > 0.0 {
                let s = e[3] / tot;
                share.push(s);
                wnum += s * tot;
                wden += tot;
                if e[3] > e[1] {
                    heavy += 1.0;
                }
            }
            frag.push(e[2]);
        }
    }
    if share.len() < 50 {
        // 撤单行太少的股票：仍然可以给出全股票口径，但订单级口径缺失
        return None;
    }
    let total = cxl_total + trade_total;
    let vals = [
        mean(&share),
        wnum / wden.max(1e-9),
        cxl_total / total.max(1e-9),
        heavy / n_cxl_order.max(1e-9),
        mean(&frag),
    ];
    for x in vals.iter() {
        if !x.is_finite() {
            return None;
        }
    }
    let diag = [
        n_trade,
        n_cxl_order,
        cxl_total,
        trade_total,
        m.len() as f64,
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

    let rows: Vec<(String, [f64; 5], [f64; 5])> = codes
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
