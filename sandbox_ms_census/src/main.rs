//! 第 07 期：毫秒里的成交（sandbox）
//!
//! 口径
//! - 记录：flag 66（主买）/ 83（主卖）且 turnover>0。
//! - 时段：连续竞价 [9:30:00, 11:30:00) 与 [13:00:00, 14:57:00)，共 14,220 秒 = 1,422,000 个 10 毫秒刻度。
//! - 刻度：exchtime 四舍五入到 10 毫秒（2025-11-27 的时间戳带 ±0.25 微秒的存储误差，截断会错一格）。
//! - 事件（主动单）：同一刻度、同一主动方委托号（主买 bid_order，主卖 ask_order）的记录合并成一次。
//! - 随机种子由代码和日期决定，结果可复现。
//!
//! 每股票日输出一行 u32（列布局见 Layout）和一行 f64（100 格金额 + 全天金额）。
//! 全市场同刻度齐发（角度 6）在 main 里合并后计算。
mod reader;
use rayon::prelude::*;
use reader::{read_snaps, read_trades};
use serde::Serialize;
use std::io::Write;

const N_TICK: usize = 1_422_000;
const AM_TICK: usize = 720_000;
const N_CELL: usize = 100;
const N_SNAP: usize = 300;
const N_GAP: usize = 500;
const N_MIN: usize = 237;
const N_HIST: usize = 3000;
const N_PLACEBO: usize = 20;
const GAP_EDGES: [u32; 14] = [0, 1, 2, 3, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 6000];
const N_B: usize = 13;

// 列偏移：cell 四组、snap 两组、clock3、gap 两组、相似度五组、标量
const C_REC: usize = 0;
const C_EV: usize = C_REC + N_CELL;
const C_EV0: usize = C_EV + N_CELL;
const C_EVX: usize = C_EV0 + N_CELL;
const C_SNAP: usize = C_EVX + N_CELL;
const C_OCC: usize = C_SNAP + N_SNAP;
const C_CLK: usize = C_OCC + N_SNAP;
const C_GAP: usize = C_CLK + N_SNAP;
const C_SHUF: usize = C_GAP + N_GAP;
const C_PAIR: usize = C_SHUF + N_GAP;
const C_SD: usize = C_PAIR + N_B;
const C_SDV: usize = C_SD + N_B;
const C_BDAY: usize = C_SDV + N_B;
const C_BMIN: usize = C_BDAY + N_B;
const C_SCAL: usize = C_BMIN + N_B;
// 标量：nrec, nev, nticks, nticks_multi_order, nsnap, phase0, phase1, phase2, unsorted, edge
const N_SCAL: usize = 10;
const N_COL: usize = C_SCAL + N_SCAL;
const N_F: usize = N_CELL + 1;

#[derive(Clone, Copy)]
struct Ev {
    t: u32,
    dir: u8,
    vol: f64,
}

struct Rng(u64);
impl Rng {
    fn new(code: &str, date: i64, salt: u64) -> Self {
        let mut h: u64 = 0xcbf29ce484222325 ^ salt.wrapping_mul(0x9E3779B97F4A7C15) ^ (date as u64);
        for b in code.bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
        Rng(h | 1)
    }
    #[inline]
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }
    #[inline]
    fn below(&mut self, n: usize) -> usize {
        ((self.next() >> 32) * n as u64 >> 32) as usize
    }
}

/// 日内微秒（已是东八区）转 10 毫秒序号；非连续竞价返回 None。
#[inline]
fn to_idx(day_cs: i64) -> Option<usize> {
    const A0: i64 = 3_420_000;
    const A1: i64 = 4_140_000;
    const P0: i64 = 4_680_000;
    const P1: i64 = 5_382_000;
    if (A0..A1).contains(&day_cs) {
        Some((day_cs - A0) as usize)
    } else if (P0..P1).contains(&day_cs) {
        Some(AM_TICK + (day_cs - P0) as usize)
    } else {
        None
    }
}

#[inline]
fn same_sess(a: u32, b: u32) -> bool {
    ((a as usize) < AM_TICK) == ((b as usize) < AM_TICK)
}

#[inline]
fn gap_bin(g: u32) -> usize {
    let mut k = N_B;
    for i in 0..N_B {
        if g >= GAP_EDGES[i] && g < GAP_EDGES[i + 1] {
            k = i;
            break;
        }
    }
    k
}

struct Part {
    code: String,
    u: Vec<u32>,
    f: Vec<f64>,
    marks: Vec<(u32, u8)>,
}

fn one(date: i64, code: &str) -> Option<Part> {
    let recs = read_trades(code, date).ok()?;
    let mut u = vec![0u32; N_COL];
    let mut f = vec![0f64; N_F];
    let mut evs: Vec<Ev> = Vec::with_capacity(recs.len() / 2 + 1);
    let mut last_key: (u32, u8, i64) = (u32::MAX, 0, -1);
    let mut last_t: i64 = -1;
    let mut last_tick: u32 = u32::MAX;
    let mut orders_in_tick = 0u32;
    for r in &recs {
        f[N_CELL] += r.turnover;
        let Some(i) = to_idx(r.cs) else {
            if (r.cs == 4_140_000) || (r.cs == 5_382_000) {
                u[C_SCAL + 9] += 1;
            }
            continue;
        };
        if (i as i64) < last_t {
            u[C_SCAL + 8] += 1;
        }
        last_t = i as i64;
        let cell = i % 100;
        u[C_REC + cell] += 1;
        f[cell] += r.turnover;
        u[C_SCAL] += 1;
        let t = i as u32;
        if t != last_tick {
            last_tick = t;
            orders_in_tick = 0;
            u[C_SCAL + 2] += 1;
        }
        let key = (t, r.dir, r.oid);
        if key == last_key {
            evs.last_mut().unwrap().vol += r.vol;
        } else {
            last_key = key;
            orders_in_tick += 1;
            if orders_in_tick == 2 {
                u[C_SCAL + 3] += 1;
            }
            evs.push(Ev { t, dir: r.dir, vol: r.vol });
        }
    }
    if evs.is_empty() {
        return None;
    }
    u[C_SCAL + 1] = evs.len() as u32;

    // 角度 0/1：100 个格子
    for e in &evs {
        let i = e.t as usize;
        let cell = i % 100;
        u[C_EV + cell] += 1;
        if (i / 100) % 60 == 0 {
            u[C_EV0 + cell] += 1;
        }
        if i >= 30_000 {
            u[C_EVX + cell] += 1;
        }
        // 3 秒钟点格
        u[C_CLK + ((i / 100) % 3) * 100 + cell] += 1;
    }

    // 角度 2：对齐到本股上一个快照
    if let Ok(snaps) = read_snaps(code, date) {
        let mut st: Vec<u32> = snaps.iter().filter_map(|&cs| to_idx(cs).map(|x| x as u32)).collect();
        st.sort_unstable();
        st.dedup();
        u[C_SCAL + 4] = st.len() as u32;
        for &s in &st {
            u[C_SCAL + 5 + ((s as usize / 100) % 3)] += 1;
        }
        let mut diff = vec![0i64; N_SNAP + 1];
        for w in st.windows(2) {
            if same_sess(w[0], w[1]) {
                let g = ((w[1] - w[0]) as usize).min(N_SNAP);
                diff[0] += 1;
                diff[g] -= 1;
            }
        }
        let mut c = 0i64;
        for d in 0..N_SNAP {
            c += diff[d];
            u[C_OCC + d] = c as u32;
        }
        let mut p = 0usize;
        for e in &evs {
            while p + 1 < st.len() && st[p + 1] <= e.t {
                p += 1;
            }
            if st.is_empty() || st[p] > e.t || p + 1 >= st.len() || !same_sess(st[p], st[p + 1]) || !same_sess(st[p], e.t) {
                continue;
            }
            let d = (e.t - st[p]) as usize;
            if d < N_SNAP {
                u[C_SNAP + d] += 1;
            }
        }
    }

    // 角度 3：相邻主动单间隔，真实 vs 分钟内随机重排
    for w in evs.windows(2) {
        if same_sess(w[0].t, w[1].t) {
            let g = (w[1].t - w[0].t) as usize;
            if g < N_GAP {
                u[C_GAP + g] += 1;
            }
        }
    }
    let mut rng = Rng::new(code, date, 1);
    let mut per_min = vec![0u32; N_MIN];
    for e in &evs {
        per_min[e.t as usize / 6000] += 1;
    }
    let mut sh: Vec<u32> = Vec::with_capacity(evs.len());
    for (m, &n) in per_min.iter().enumerate() {
        for _ in 0..n {
            sh.push((m * 6000 + rng.below(6000)) as u32);
        }
    }
    sh.sort_unstable();
    for w in sh.windows(2) {
        if same_sess(w[0], w[1]) {
            let g = (w[1] - w[0]) as usize;
            if g < N_GAP {
                u[C_SHUF + g] += 1;
            }
        }
    }

    // 角度 4：下一单像不像上一单
    let attrs: Vec<(u8, f64)> = evs.iter().map(|e| (e.dir, e.vol)).collect();
    let mut a_day = attrs.clone();
    let mut r2 = Rng::new(code, date, 2);
    for i in (1..a_day.len()).rev() {
        let j = r2.below(i + 1);
        a_day.swap(i, j);
    }
    let mut a_min = attrs.clone();
    let mut r3 = Rng::new(code, date, 3);
    let mut s = 0usize;
    while s < evs.len() {
        let m = evs[s].t as usize / 6000;
        let mut e = s + 1;
        while e < evs.len() && evs[e].t as usize / 6000 == m {
            e += 1;
        }
        for i in (s + 1..e).rev() {
            let j = s + r3.below(i - s + 1);
            a_min.swap(i, j);
        }
        s = e;
    }
    for i in 0..evs.len() - 1 {
        if !same_sess(evs[i].t, evs[i + 1].t) {
            continue;
        }
        let k = gap_bin(evs[i + 1].t - evs[i].t);
        if k >= N_B {
            continue;
        }
        u[C_PAIR + k] += 1;
        let (d0, v0) = attrs[i];
        let (d1, v1) = attrs[i + 1];
        if d0 == d1 {
            u[C_SD + k] += 1;
            if v0 == v1 {
                u[C_SDV + k] += 1;
            }
        }
        if a_day[i].0 == a_day[i + 1].0 && a_day[i].1 == a_day[i + 1].1 {
            u[C_BDAY + k] += 1;
        }
        if a_min[i].0 == a_min[i + 1].0 && a_min[i].1 == a_min[i + 1].1 {
            u[C_BMIN + k] += 1;
        }
    }

    // 角度 6：本股每个刻度、每个方向只算一次
    let mut marks: Vec<(u32, u8)> = evs.iter().map(|e| (e.t, e.dir)).collect();
    marks.sort_unstable();
    marks.dedup();
    Some(Part { code: code.to_string(), u, f, marks })
}

#[derive(Serialize)]
struct Cross {
    real: Vec<u64>,
    fake: Vec<f64>,
    real_sh: Vec<u64>,
    fake_sh: Vec<f64>,
    real_sz: Vec<u64>,
    fake_sz: Vec<f64>,
    real_ex: Vec<u64>,
    fake_ex: Vec<f64>,
    fake_max: Vec<u32>,
    fake_max_ex: Vec<u32>,
    top: Vec<(u32, u8, u32, u32)>,
}

#[inline]
fn in_open5(t: u32) -> bool {
    let t = t as usize;
    t < 30_000 || (AM_TICK..AM_TICK + 30_000).contains(&t)
}

fn add_marks(cnt: &mut [u16], p: &Part, shift: i64, ex: bool) {
    for &(t, d) in &p.marks {
        if ex && in_open5(t) {
            continue;
        }
        let i = (t as i64 + shift).rem_euclid(N_TICK as i64) as usize;
        cnt[i * 2 + (d as usize)] += 1;
    }
}

fn hist(cnt: &[u16], from_tick: usize, out: &mut [u64]) {
    for (j, &c) in cnt.iter().enumerate() {
        if j / 2 >= from_tick {
            out[(c as usize).min(N_HIST - 1)] += 1;
        }
    }
}

fn cross(parts: &[Part], date: i64) -> Cross {
    let is_sh: Vec<bool> = parts.iter().map(|p| p.code.starts_with('6')).collect();
    // ex=true 时，上午和午后开盘后 5 分钟的成交不进真实也不进对照，避免开盘尖峰被错开后混进对照组
    let build = |shifts: &[i64], ex: bool| -> (Vec<u16>, Vec<u16>) {
        let mut a = vec![0u16; N_TICK * 2];
        let mut b = vec![0u16; N_TICK * 2];
        for (k, p) in parts.iter().enumerate() {
            add_marks(if is_sh[k] { &mut a } else { &mut b }, p, shifts[k], ex);
        }
        (a, b)
    };
    let sum = |a: &[u16], b: &[u16]| -> Vec<u16> { a.iter().zip(b).map(|(x, y)| x + y).collect() };
    let zero = vec![0i64; parts.len()];
    let (a, b) = build(&zero, false);
    let t = sum(&a, &b);
    let (ax, bx) = build(&zero, true);
    let tx = sum(&ax, &bx);
    let mut real = vec![0u64; N_HIST];
    let mut real_sh = vec![0u64; N_HIST];
    let mut real_sz = vec![0u64; N_HIST];
    let mut real_ex = vec![0u64; N_HIST];
    hist(&t, 0, &mut real);
    hist(&ax, 0, &mut real_sh);
    hist(&bx, 0, &mut real_sz);
    hist(&tx, 0, &mut real_ex);

    let mut idx: Vec<usize> = (0..tx.len()).filter(|&j| tx[j] > 0).collect();
    idx.sort_unstable_by_key(|&j| std::cmp::Reverse((tx[j], usize::MAX - j)));
    let top = idx.iter().take(5000).map(|&j| ((j / 2) as u32, (j % 2) as u8, ax[j] as u32, bx[j] as u32)).collect();

    let reps: Vec<_> = (0..N_PLACEBO)
        .into_par_iter()
        .map(|rep| {
            let shifts: Vec<i64> = parts
                .iter()
                .map(|p| {
                    let mut r = Rng::new(&p.code, date, 100 + rep as u64);
                    let m = 1 + r.below(5) as i64;
                    let s = if r.below(2) == 0 { m } else { -m };
                    s * 6000
                })
                .collect();
            let (a, b) = build(&shifts, false);
            let t = sum(&a, &b);
            let (ax, bx) = build(&shifts, true);
            let tx = sum(&ax, &bx);
            let mut h0 = vec![0u64; N_HIST];
            let mut h1 = vec![0u64; N_HIST];
            let mut h2 = vec![0u64; N_HIST];
            let mut h3 = vec![0u64; N_HIST];
            hist(&t, 0, &mut h0);
            hist(&ax, 0, &mut h1);
            hist(&bx, 0, &mut h2);
            hist(&tx, 0, &mut h3);
            let mx = *t.iter().max().unwrap() as u32;
            let mx_ex = *tx.iter().max().unwrap() as u32;
            (h0, h1, h2, h3, mx, mx_ex)
        })
        .collect();
    let avg = |pick: fn(&(Vec<u64>, Vec<u64>, Vec<u64>, Vec<u64>, u32, u32)) -> &Vec<u64>| -> Vec<f64> {
        let mut o = vec![0f64; N_HIST];
        for r in &reps {
            for (x, y) in o.iter_mut().zip(pick(r)) {
                *x += *y as f64 / N_PLACEBO as f64;
            }
        }
        o
    };
    Cross {
        real,
        fake: avg(|r| &r.0),
        real_sh,
        fake_sh: avg(|r| &r.1),
        real_sz,
        fake_sz: avg(|r| &r.2),
        real_ex,
        fake_ex: avg(|r| &r.3),
        fake_max: reps.iter().map(|r| r.4).collect(),
        fake_max_ex: reps.iter().map(|r| r.5).collect(),
        top,
    }
}

#[derive(Serialize)]
struct Meta {
    date: i64,
    n_codes: usize,
    n_valid: usize,
    n_col: usize,
    n_f: usize,
    codes: Vec<String>,
    layout: Vec<(String, usize, usize)>,
    gap_edges: Vec<u32>,
    cross: Cross,
}

fn universe(date: i64) -> Vec<String> {
    let valid = format!("out/valid_{date}.txt");
    let text = std::fs::read_to_string(&valid).unwrap_or_else(|_| panic!("缺少 {valid}"));
    text.lines().map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let date: i64 = args[1].parse().expect("date");
    let limit: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(0);
    let tag = args.get(3).cloned().unwrap_or_default();
    let mut codes = universe(date);
    if limit > 0 && limit < codes.len() {
        // 小样本：沪深创均匀抽
        let step = codes.len() / limit;
        codes = codes.into_iter().step_by(step.max(1)).take(limit).collect();
    }
    let n_codes = codes.len();
    eprintln!("date={date} codes={n_codes}");
    let parts: Vec<Part> = codes.par_iter().filter_map(|c| one(date, c)).collect();
    eprintln!("valid={}", parts.len());
    let cr = cross(&parts, date);
    let layout: Vec<(String, usize, usize)> = [
        ("rec", C_REC, N_CELL),
        ("ev", C_EV, N_CELL),
        ("ev0", C_EV0, N_CELL),
        ("evx", C_EVX, N_CELL),
        ("snap", C_SNAP, N_SNAP),
        ("occ", C_OCC, N_SNAP),
        ("clk", C_CLK, N_SNAP),
        ("gap", C_GAP, N_GAP),
        ("shuf", C_SHUF, N_GAP),
        ("pair", C_PAIR, N_B),
        ("sd", C_SD, N_B),
        ("sdv", C_SDV, N_B),
        ("bday", C_BDAY, N_B),
        ("bmin", C_BMIN, N_B),
        ("scal", C_SCAL, N_SCAL),
    ]
    .iter()
    .map(|(a, b, c)| (a.to_string(), *b, *c))
    .collect();
    let stem = if tag.is_empty() { format!("out/{date}") } else { format!("out/{date}_{tag}") };
    let mut fu = std::io::BufWriter::new(std::fs::File::create(format!("{stem}.u32")).unwrap());
    let mut ff = std::io::BufWriter::new(std::fs::File::create(format!("{stem}.f64")).unwrap());
    for p in &parts {
        for x in &p.u {
            fu.write_all(&x.to_le_bytes()).unwrap();
        }
        for x in &p.f {
            ff.write_all(&x.to_le_bytes()).unwrap();
        }
    }
    let meta = Meta {
        date,
        n_codes,
        n_valid: parts.len(),
        n_col: N_COL,
        n_f: N_F,
        codes: parts.iter().map(|p| p.code.clone()).collect(),
        layout,
        gap_edges: GAP_EDGES.to_vec(),
        cross: cr,
    };
    std::fs::write(format!("{stem}.json"), serde_json::to_string(&meta).unwrap()).unwrap();
}
