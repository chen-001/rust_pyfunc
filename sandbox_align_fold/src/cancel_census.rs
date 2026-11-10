//! 撤单数据覆盖普查：按日期、沪深、时段，统计能算出撤单字母的股票比例。只读数据，不产出因子。
//! 用法：cancel_census <输出csv> <日期...>
#[path = "fast_csv_reader.rs"]
mod fast_csv_reader;
#[path = "letters.rs"]
mod letters;
use letters::{build_sequences, Win};
use rayon::prelude::*;
use std::collections::BTreeSet;
use std::io::Write;
use std::path::Path;

const CANCEL: usize = 5; // letters 里 cancel 字母的下标

fn wins() -> Vec<Win> {
    vec![
        Win { name: "full", raw_lo: 34200, raw_hi: 53820, adj_lo: 34200, adj_hi: 48420, bar: 30 },
        Win { name: "pm", raw_lo: 46800, raw_hi: 53820, adj_lo: 41400, adj_hi: 48420, bar: 30 },
        Win { name: "tail", raw_lo: 52020, raw_hi: 53820, adj_lo: 46620, adj_hi: 48420, bar: 15 },
    ]
}

fn whitelist() -> BTreeSet<String> {
    let mut r = csv::Reader::from_path("/ssd_data/data/basic_info/symbol_map.csv").expect("股票表");
    let col = r.headers().unwrap().iter().position(|h| h == "symbol").expect("symbol 列");
    r.records().map(|x| format!("{:0>6}", x.unwrap().get(col).unwrap().trim())).collect()
}

#[derive(Default, Clone, Copy)]
struct S { traded: u32, any_cancel: u32, both: u32, valid: u32, active_sum: u64, zero_side: u32, cancel_n: u64 }

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let out = &a[1];
    let pool = rayon::ThreadPoolBuilder::new().num_threads(50).build().unwrap();
    let wl = whitelist();
    let ws = wins();
    let mut f = std::fs::File::create(out).unwrap();
    writeln!(f, "date,market,window,stock_files,market_data_files,traded,any_cancel,both_side,letter_valid,letter_all_zero,mean_active_letters,cancel_records").unwrap();
    for d in &a[2..] {
        let date: i64 = d.parse().unwrap();
        let dir = format!("/nvme/level2/{d}");
        if !Path::new(&dir).join("transaction").is_dir() { eprintln!("{d} 目录不存在"); continue; }
        let codes: Vec<String> = wl.iter().filter(|c| Path::new(&dir).join("transaction").join(format!("{c}_{d}_transaction.csv")).is_file()).cloned().collect();
        let t0 = std::time::Instant::now();
        let rows: Vec<(usize, bool, Vec<Option<(u32, u32, u32, u32, u64)>>)> = pool.install(|| codes.par_iter().map(|c| {
            let m = if c.starts_with('6') { 0 } else { 1 };
            let md = Path::new(&dir).join("market_data").join(format!("{c}_{d}_market_data.csv")).is_file();
            let v = match fast_csv_reader::read_trade_fast(c, date) {
                Err(_) => vec![None; 3],
                Ok(tr) => ws.iter().map(|w| {
                    let inw: Vec<_> = tr.iter().filter(|t| t.raw_sec >= w.raw_lo && t.raw_sec <= w.raw_hi).collect();
                    let tn = inw.iter().filter(|t| t.flag != 32).count() as u32;
                    if tn == 0 { return None; }
                    let (mut b, mut s) = (0u32, 0u32);
                    for t in &inw { if t.flag == 32 { if t.bid_order > 0 && t.ask_order == 0 { b += 1 } else if t.ask_order > 0 && t.bid_order == 0 { s += 1 } } }
                    let sq = build_sequences(&tr, &[], w);
                    let cs = sq[CANCEL].as_ref();
                    let (nb, ns) = cs.map(|q| (q.side.iter().filter(|&&x| x > 0).count() as u32, q.side.iter().filter(|&&x| x < 0).count() as u32)).unwrap_or((0, 0));
                    Some((tn, b, s, (nb << 16) | ns, (b + s) as u64))
                }).collect(),
            };
            (m, md, v)
        }).collect());
        for m in 0..2 {
            let names = ["SH", "SZ"];
            let sel: Vec<_> = rows.iter().filter(|r| r.0 == m).collect();
            let md_n = sel.iter().filter(|r| r.1).count();
            for (wi, w) in ws.iter().enumerate() {
                let mut x = S::default();
                for r in &sel {
                    if let Some((_, b, s, packed, n)) = r.2[wi] {
                        x.traded += 1;
                        let (nb, ns) = (packed >> 16, packed & 0xffff);
                        if b + s > 0 { x.any_cancel += 1; }
                        if b > 0 && s > 0 { x.both += 1; }
                        if nb >= 3 && ns >= 3 { x.valid += 1; }
                        if nb + ns == 0 { x.zero_side += 1; }
                        x.active_sum += (nb + ns) as u64;
                        x.cancel_n += n;
                    }
                }
                writeln!(f, "{d},{},{},{},{},{},{},{},{},{},{:.1},{}", names[m], w.name, sel.len(), md_n, x.traded, x.any_cancel, x.both, x.valid, x.zero_side,
                    x.active_sum as f64 / x.traded.max(1) as f64, x.cancel_n).unwrap();
            }
        }
        f.flush().unwrap();
        eprintln!("{d} 完成 {:.1}s 股票 {}", t0.elapsed().as_secs_f64(), codes.len());
    }
}
