//! 订单号坐标轴探索（sandbox v7b）。
//! 把 flag=32 的撤单行也读回来（读取器默认丢弃），按「订单号」把撤单与成交放在同一个订单对象上。
//! 构建: cd /home/chenzongwei/rust_pyfunc/sandbox_oidflow && cargo build --release
//! 运行: ./target/release/oidflow_sandbox 20220819 > out_v7_20220819.json
mod fast_csv_reader;
use rayon::prelude::*;
use serde::Serialize;

const NAMES: [&str; 5] = [
    "fa_cxl_share_do",  // 有撤单的订单，撤单量 /（撤单量 + 成交量）（订单等权）
    "fa_cxl_share_vw",  // 同上（按订单总量加权）
    "fa_cxl_vol_share", // 全股票：撤单总量 /（撤单总量 + 成交总量）
    "fa_cxl_heavy_do",  // 有撤单的订单里，撤单量 > 成交量 的比例
    "fa_cxl_frag",      // 一个订单平均被撤销几次
];

#[derive(Serialize)]
struct Out {
    names: Vec<String>,
    codes: Vec<String>,
    vals: Vec<Vec<f32>>,
    diag: Vec<Vec<f32>>,
}

#[derive(Clone, Copy)]
struct Row {
    oid: i64,
    vol: f64,
    is_cxl: bool,
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

/// 直接读 CSV 全字段：返回 (订单号, 成交量, 是否撤单)。撤单用 bid_order（跟成交一样：撤单行的
/// bid_order 是原始委托号，ask_order 恒为 0）。
fn read_orders(code: &str, date: i64) -> Option<Vec<Row>> {
    let root = std::env::var("RUST_PYFUNC_LEVEL2_ROOT").unwrap_or("/ssd_data/stock".into());
    let path = format!("{}/{}/transaction/{}_{}_transaction.csv", root, date, code, date);
    let text = std::fs::read_to_string(&path).ok()?;
    let mut out = Vec::with_capacity(text.len() / 90);
    for line in text.lines().skip(1) {
        let f: Vec<&str> = line.split(',').collect();
        if f.len() < 15 {
            continue;
        }
        let flag: i32 = match f[10].parse() {
            Ok(v) => v,
            Err(_) => continue,
        };
        if flag != 66 && flag != 83 && flag != 32 {
            continue;
        }
        let vol: f64 = match f[8].parse() {
            Ok(v) => v,
            Err(_) => continue,
        };
        if !(vol > 0.0) {
            continue;
        }
        let bid: i64 = f[14].parse().unwrap_or(0);
        let ask: i64 = f[13].parse().unwrap_or(0);
        let oid = if bid > 0 { bid } else { ask };
        if oid <= 0 {
            continue;
        }
        out.push(Row { oid, vol, is_cxl: flag == 32 });
    }
    Some(out)
}

fn mean(x: &[f64]) -> f64 {
    if x.is_empty() { return f64::NAN; }
    x.iter().sum::<f64>() / x.len() as f64
}

fn per_stock(rows: &[Row]) -> Option<([f64; 5], [f64; 5])> {
    let mut m: std::collections::HashMap<i64, [f64; 4]> = std::collections::HashMap::new();
    let (mut n_trade, mut n_cxl_row) = (0.0f64, 0.0f64);
    for r in rows {
        let e = m.entry(r.oid).or_insert([0.0; 4]);
        if r.is_cxl {
            e[2] += 1.0;
            e[3] += r.vol;
            n_cxl_row += 1.0;
        } else {
            e[0] += 1.0;
            e[1] += r.vol;
            n_trade += 1.0;
        }
    }
    if n_trade < 300.0 || n_cxl_row < 50.0 {
        return None;
    }
    let mut share: Vec<f64> = Vec::new();
    let (mut wnum, mut wden) = (0.0f64, 0.0f64);
    let (mut cxl_total, mut trade_total) = (0.0f64, 0.0f64);
    let mut frag: Vec<f64> = Vec::new();
    let (mut heavy, mut n_ord_cxl) = (0.0f64, 0.0f64);
    for (_, e) in m.iter() {
        cxl_total += e[3];
        trade_total += e[1];
        if e[2] > 0.0 {
            n_ord_cxl += 1.0;
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
        return None;
    }
    let total = cxl_total + trade_total;
    let vals = [
        mean(&share),
        wnum / wden.max(1e-9),
        cxl_total / total.max(1e-9),
        heavy / n_ord_cxl.max(1e-9),
        mean(&frag),
    ];
    for x in vals.iter() {
        if !x.is_finite() {
            return None;
        }
    }
    let diag = [n_trade, n_cxl_row, cxl_total, trade_total, m.len() as f64];
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
            let r = read_orders(c, date)?;
            let (v, d) = per_stock(&r)?;
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
