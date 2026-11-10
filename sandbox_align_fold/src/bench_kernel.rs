//! 单线程测第三版各计算核的单位成本（用真实字母序列，非因子产出）。
#[path = "fast_csv_reader.rs"]
mod fast_csv_reader;
#[path = "letters.rs"]
mod letters;
#[path = "fold.rs"]
mod fold;
use fold::{Table, NEG};
use letters::{build_sequences, Win};
use std::time::Instant;

fn solve_i16(n: usize, h: usize, active: &[bool], side: &[i8], r: &mut Vec<i16>, z: &mut Vec<i16>) -> i16 {
    let st = n + 1;
    r.clear(); r.resize(st * st, 0); z.clear(); z.resize(n + 1, i16::MIN / 2);
    for j in 0..n {
        if !active[j] || j < h + 1 { for i in 0..=j { r[i * st + j + 1] = r[i * st + j]; } continue; }
        let km = j - h - 1;
        for k in 0..=km { z[k] = if side[k] * side[j] == -1 { 1 + r[(k + 1) * st + j] } else { i16::MIN / 2 }; }
        for i in 0..=j {
            let row = &mut r[i * st..(i + 1) * st];
            let mut b = row[j];
            if i <= km { for (x, y) in row[i..=km].iter().zip(&z[i..=km]) { b = b.max(*x + *y); } }
            row[j + 1] = b;
        }
    }
    r[n]
}

/// 配对概率所需的“内部配分函数”前半段（加权所有合法配对方案）
fn inside(n: usize, h: usize, side: &[i8], bz: f64, q: &mut Vec<f64>, z: &mut Vec<f64>) -> f64 {
    let st = n + 1;
    q.clear(); q.resize(st * st, 1.0); z.clear(); z.resize(n + 1, 0.0);
    for j in 0..n {
        if j < h + 1 { for i in 0..=j { q[i * st + j + 1] = q[i * st + j]; } continue; }
        let km = j - h - 1;
        for k in 0..=km { z[k] = if side[k] * side[j] == -1 { bz * q[(k + 1) * st + j] } else { 0.0 }; }
        for i in 0..=j {
            let row = &mut q[i * st..(i + 1) * st];
            let mut s = row[j];
            if i <= km { for (x, y) in row[i..=km].iter().zip(&z[i..=km]) { s += *x * *y; } }
            row[j + 1] = s;
        }
    }
    q[n]
}

/// 买卖二分图最大权匹配（允许任意交叉），匈牙利算法。
fn hungarian(c: &[Vec<i64>]) -> i64 {
    let (n, m) = (c.len(), c[0].len());
    let (mut u, mut v, mut p, mut way) = (vec![0i64; n + 1], vec![0i64; m + 1], vec![0usize; m + 1], vec![0usize; m + 1]);
    for i in 1..=n {
        p[0] = i;
        let (mut j0, mut mv, mut used) = (0usize, vec![i64::MAX; m + 1], vec![false; m + 1]);
        loop {
            used[j0] = true;
            let (i0, mut d, mut j1) = (p[j0], i64::MAX, 0usize);
            for j in 1..=m { if !used[j] {
                let cur = -c[i0 - 1][j - 1] - u[i0] - v[j];
                if cur < mv[j] { mv[j] = cur; way[j] = j0; }
                if mv[j] < d { d = mv[j]; j1 = j; }
            } }
            for j in 0..=m { if used[j] { u[p[j]] += d; v[j] -= d; } else { mv[j] -= d; } }
            j0 = j1;
            if p[j0] == 0 { break; }
        }
        loop { let j1 = way[j0]; p[j0] = p[j1]; j0 = j1; if j0 == 0 { break; } }
    }
    (1..=m).filter(|&j| p[j] != 0).map(|j| c[p[j] - 1][j - 1]).sum()
}

fn main() {
    let codes = ["000001", "600000", "300750", "002594", "688981", "603259", "000858", "601318"];
    let date = 20241231;
    let (mut trd, mut mkd, mut let_t) = (0f64, 0f64, 0f64);
    let wins = [
        ("full474", Win { name: "full", raw_lo: 34200, raw_hi: 53820, adj_lo: 34200, adj_hi: 48420, bar: 30 }),
        ("tail5s360", Win { name: "tail", raw_lo: 52020, raw_hi: 53820, adj_lo: 46620, adj_hi: 48420, bar: 5 }),
        ("pm30s234", Win { name: "pm", raw_lo: 46800, raw_hi: 53820, adj_lo: 41400, adj_hi: 48420, bar: 30 }),
        ("tail15s120", Win { name: "tail", raw_lo: 52020, raw_hi: 53820, adj_lo: 46620, adj_hi: 48420, bar: 15 }),
    ];
    let mut t = Table::new();
    let (mut r16, mut z16, mut q, mut zf) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let mut tot = vec![[0f64; 7]; wins.len()];
    let (mut cnt, mut act) = (0f64, 0f64);
    for c in codes {
        let t0 = Instant::now(); let tr = fast_csv_reader::read_trade_fast(c, date).unwrap(); trd += t0.elapsed().as_secs_f64();
        let t0 = Instant::now(); let mk = fast_csv_reader::read_market_fast(c, date).unwrap(); mkd += t0.elapsed().as_secs_f64();
        for (wi, (_, w)) in wins.iter().enumerate() {
            let t0 = Instant::now(); let sq = build_sequences(&tr, &mk, w); let_t += t0.elapsed().as_secs_f64();
            for a in [0usize, 1, 5, 6] {
                let Some(s) = &sq[a] else { continue };
                let n = s.side.len();
                let active: Vec<bool> = s.side.iter().map(|&x| x != 0).collect();
                let w1 = |k: usize, j: usize| if s.side[k] * s.side[j] == -1 { 1 } else { NEG };
                let tau = (n as f64 / 8.0).max(1.0);
                let dec: Vec<i32> = (0..=n).map(|d| ((1000.0 * (-(d as f64) / tau).exp()).round() as i32).max(1)).collect();
                let w2 = |k: usize, j: usize| if s.side[k] * s.side[j] == -1 { dec[j - k] } else { NEG };
                let t0 = Instant::now(); let x = t.solve(n, 3, &active, &w1); tot[wi][0] += t0.elapsed().as_secs_f64();
                let t0 = Instant::now(); let mut p = vec![-1i32; n + 1]; t.traceback(n, 3, &w1, true, &mut p); tot[wi][1] += t0.elapsed().as_secs_f64();
                let t0 = Instant::now(); let y = solve_i16(n, 3, &active, &s.side, &mut r16, &mut z16); tot[wi][2] += t0.elapsed().as_secs_f64();
                assert_eq!(x, y as i32);
                let t0 = Instant::now(); t.solve(n, 3, &active, &w2); tot[wi][3] += t0.elapsed().as_secs_f64();
                // 压缩掉中性字母后的序列再解一次
                let idx: Vec<usize> = (0..n).filter(|&i| active[i]).collect();
                let cs: Vec<i8> = idx.iter().map(|&i| s.side[i]).collect();
                let on = vec![true; cs.len()];
                let w3 = |k: usize, j: usize| if cs[k] * cs[j] == -1 && idx[j] - idx[k] > 3 { 1 } else { NEG };
                let t0 = Instant::now(); let z2 = t.solve(cs.len(), 0, &on, &w3); tot[wi][4] += t0.elapsed().as_secs_f64();
                assert_eq!(z2, x);
                cnt += n as f64; act += cs.len() as f64;
                let t0 = Instant::now(); let v = inside(n, 3, &s.side, 1.05, &mut q, &mut zf); tot[wi][5] += t0.elapsed().as_secs_f64();
                assert!(v.is_finite());
                let bu: Vec<usize> = (0..n).filter(|&i| s.side[i] > 0).collect();
                let se: Vec<usize> = (0..n).filter(|&i| s.side[i] < 0).collect();
                if !bu.is_empty() && !se.is_empty() {
                    let (l, r) = if bu.len() <= se.len() { (&bu, &se) } else { (&se, &bu) };
                    let m: Vec<Vec<i64>> = l.iter().map(|&i| r.iter().map(|&j| { let d = i.abs_diff(j); if d > 3 { dec[d] as i64 } else { 0 } }).collect()).collect();
                    let t0 = Instant::now(); hungarian(&m); tot[wi][6] += t0.elapsed().as_secs_f64();
                }
            }
        }
    }
    let n = (codes.len() * 4) as f64;
    println!("读逐笔 {:.2} ms/股, 读盘口 {:.2} ms/股, 字母化(10种) {:.2} ms/股/时段, 活跃字母占比 {:.3}", trd * 1e3 / codes.len() as f64, mkd * 1e3 / codes.len() as f64, let_t * 1e3 / (codes.len() * wins.len()) as f64, act / cnt);
    println!("单位 ms/次: 时段, 求解i32, 倒推, 求解i16, 衰减i32, 压缩后i32, 配分内部f64, 匈牙利");
    for (i, (nm, _)) in wins.iter().enumerate() {
        println!("{nm}: {}", tot[i].iter().map(|x| format!("{:.3}", x * 1e3 / n)).collect::<Vec<_>>().join(" "));
    }
}
