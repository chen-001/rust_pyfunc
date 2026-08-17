// 隔离测试3: 完全复刻二进制的 per_stock_prep 打包 + merge_x139
mod fast_csv_reader;
use fast_csv_reader::{read_trade_fast, TradeRecord};

fn days_from_civil(y: i64, m: u64, d: u64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400;
    let mp = (m + 9) % 12;
    let doy = (153 * mp + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy as i64;
    era * 146097 + doe - 719468
}

#[inline]
fn pack_time(t_us: i64, day_start_us: i64, big: bool) -> u32 {
    let off = t_us - day_start_us;
    let sec = (off / 1_000_000) as u32;
    let us15 = ((off % 1_000_000) >> 5) as u32;
    (sec << 16) | (us15 << 1) | (big as u32)
}

// 与 main.rs 完全一致的打包预处理
fn prep(recs: &[TradeRecord], day_start_us: i64) -> (Vec<u32>, Vec<u32>) {
    let n = recs.len();
    let mut vols: Vec<f32> = recs.iter().map(|r| r.volume as f32).collect();
    let mut sorted = vols.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let th99 = sorted[((n as f64) * 0.99) as usize] as f64;
    let mut packed = Vec::with_capacity(n);
    let mut big_times = Vec::new();
    for r in recs.iter() {
        let big = (r.volume as f64) >= th99;
        packed.push(pack_time(r.time_us, day_start_us, big));
        if big {
            big_times.push(pack_time(r.time_us, day_start_us, false));
        }
    }
    (packed, big_times)
}

fn merge_x139(a: &[u32], b: &[u32]) -> (f32, f32, f32) {
    let na = a.len(); let nb = b.len(); let n = na + nb;
    let mut trans: u64 = 0;
    let (mut c00, mut c01, mut c10, mut c11): (u64, u64, u64, u64) = (0, 0, 0, 0);
    let (mut gsum, mut gsq): (f64, f64) = (0.0, 0.0);
    let mut gcnt: u64 = 0;
    let (mut prev_p, mut prev_label): (u32, u8) = (0, 2);
    let mut has_prev = false;
    let (mut ia, mut ib) = (0usize, 0usize);
    while ia < na || ib < nb {
        let (label, t) = if ib >= nb || (ia < na && a[ia] <= b[ib]) {
            let l = (0u8, a[ia]); ia += 1; l
        } else {
            let l = (1u8, b[ib]); ib += 1; l
        };
        if has_prev {
            if label != prev_label { trans += 1; }
            match (prev_label, label) {
                (0, 0) => c00 += 1, (0, 1) => c01 += 1,
                (1, 0) => c10 += 1, _ => c11 += 1,
            }
            let g = (t - prev_p) as f64 * 32.0;
            gsum += g; gsq += g * g; gcnt += 1;
        }
        prev_p = t; prev_label = label; has_prev = true;
    }
    let x9 = if gcnt > 1 && gsum > 0.0 {
        let mean = gsum / gcnt as f64;
        let var = (gsq - gsum * gsum / gcnt as f64) / (gcnt - 1) as f64;
        if var > 0.0 { (var.sqrt() / mean) as f32 } else { 0.0 }
    } else { 0.0 };
    (trans as f32 / (n - 1) as f32, 0.0, x9)
}

fn main() {
    let date = 20241231i64;
    let day_start = days_from_civil(date / 10000, (date / 100 % 100) as u64, (date % 100) as u64)
        * 86400 * 1_000_000 + (9 * 3600 + 30 * 60) * 1_000_000;
    let a = read_trade_fast("000096", date).unwrap();
    let b = read_trade_fast("000752", date).unwrap();
    let (pa, ba) = prep(&a, day_start);
    let (pb, bb) = prep(&b, day_start);
    let (x1, _, x9) = merge_x139(&pa, &pb);
    println!("复刻二进制: X1={x1:.5} (旧=?)  X9={x9:.4} (旧=2.6749 二进制=22.77)");
    println!("pa len={} pb len={} ba len={} bb len={}", pa.len(), pb.len(), ba.len(), bb.len());
}
