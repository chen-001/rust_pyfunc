//! 线性反馈移位存储器的非线性组合 —— 拓展二（正式版，横截面 pipeline 名 lfsr_ext2）
//!
//! 来源：sandbox_lfsr_ext2 的单日全市场探索。走横截面范式，任务粒度是「一天」：
//!   ① 逐股读逐笔 + 盘口，算拓展一回测入选的 63 列（与拓展一逐位相同）与本股原始序列
//!   ② 全市场汇总直方图，求 7 个变量的当天分位数门槛（每笔成交额、相邻两笔间隔、价差、
//!      一档挂单金额、一档不平衡、成交速度、中间价非零变动幅度）
//!   ③ 逐股按门槛造位流，算 8441 个拓展二新列
//!   ④ 逐列横截面 z-score（2 个计数与 5 个 ctl_* 风格变量除外），再把每列对
//!      「5 个风格变量 + ln 总市值」做横截面回归，残差再 z-score，得到 8499 个 _r 列
//!
//! 输出恒定 17005 列 = 2 个计数 + 63 个拓展一入选列 + 8441 个新列（共 8506）+ 8499 个残差列。
//! 算不出来的股票整行 NaN（横截面要求当天 universe 每只股票都有一行）。
//!
//! 与拓展一的 6 条实质差别（拓展二研究讨论的 6 条）：
//!  1. 三元布尔函数去重：256 种只留「三个输入都真的用到」且「互为取反只留一个」的 109 种；
//!     线性可加的 rate 不再按函数枚举，改成直接输出 8 格频率。
//!  2. 涨/平/跌三态：价格、挂单量等变化不再把「没变」和「跌」混在 0 里，
//!     单独统计变动频率、变动中的方向、事件时间下的方向自相关。
//!  3. 门槛改成「当天全市场」分位数：成交额、价差、挂单金额、成交速度等，用全市场汇总直方图求分位数。
//!  4. 风格剔除：每列再多输出一个对 ln 价格 / ln 成交额 / 日内波动 / 最小报价单位占比 /
//!     ln 笔数 / ln 总市值 做横截面回归后的残差版（后缀 _r）。
//!  5. 预测目标拉长 + 样本外：未来 60 秒、300 秒中间价收益，事件时间「下一次真实价格变动方向」；
//!     8 格条件均值前半段学、后半段验（及反过来），给出样本外相关。
//!  6. 多日合并（沙箱的 pool 模式）不进正式库：pipeline 的粒度就是一天，跨日合并用不着。
use crate::fast_csv_reader::{
    read_market_f64_inner, read_trade_f64_inner, MarketRecordF64 as MarketRecord,
    TradeRecordF64 as TradeRecord,
};
use crate::lfsr_leak_ext1_metrics as ext1;
use pyo3::prelude::*;
use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::OnceLock;

// ====================== 全市场分位数门槛 ======================
// 每个变量在当天全市场所有观测上汇总成一张直方图（变换后的值等距分 NB 格），再取分位数。
const NB: usize = 4000;
const NV: usize = 7;
const QS: [f64; 5] = [0.2, 0.5, 0.8, 0.95, 0.99];
// 变量：0 每笔成交额(元) 1 相邻两笔间隔(微秒) 2 价差(bp) 3 一档挂单金额(元) 4 |一档不平衡|
//       5 成交速度(元/秒) 6 中间价非零变动幅度(bp)
const VAR_RANGE: [(f64, f64, bool); NV] = [
    (-1.0, 10.0, true),
    (0.0, 11.0, true),
    (-2.0, 4.5, true),
    (0.0, 11.0, true),
    (0.0, 1.0, false),
    (0.0, 11.0, true),
    (-3.0, 4.5, true),
];
#[inline]
fn hv_fwd(v: usize, x: f64) -> f64 {
    if VAR_RANGE[v].2 {
        (x + 1e-12).log10()
    } else {
        x
    }
}
#[inline]
fn hv_inv(v: usize, y: f64) -> f64 {
    if VAR_RANGE[v].2 {
        10f64.powf(y)
    } else {
        y
    }
}
#[inline]
fn hbin(v: usize, x: f64) -> usize {
    let (lo, hi, _) = VAR_RANGE[v];
    let y = hv_fwd(v, x);
    let b = ((y - lo) / (hi - lo) * NB as f64).floor();
    if !(b >= 0.0) {
        0
    } else if b >= NB as f64 {
        NB - 1
    } else {
        b as usize
    }
}
type Thr = [[f64; 5]; NV];

// ====================== 单日单股的原始序列 ======================
struct Raw {
    ts: Vec<i64>,
    mid: Vec<f64>,
    imb1: Vec<f64>,
    imbt: Vec<f64>,
    imb5: Vec<f64>,
    spread: Vec<f64>,
    dep1: Vec<f64>,
    b1v: Vec<f64>,
    a1v: Vec<f64>,
    tbid: Vec<f64>,
    task: Vec<f64>,
    to_cum: Vec<f64>,
    dep1amt: Vec<f64>,
    tr: Vec<TradeRecord>,
}
/// 有效快照口径与拓展一完全一致（保证股票集合与拓展一相同）
fn build_raw(mkt: &[MarketRecord], tr: Vec<TradeRecord>) -> Option<Raw> {
    if mkt.len() < 300 || tr.len() < 500 {
        return None;
    }
    let n = mkt.len();
    let mut r = Raw {
        ts: Vec::with_capacity(n),
        mid: Vec::with_capacity(n),
        imb1: Vec::with_capacity(n),
        imbt: Vec::with_capacity(n),
        imb5: Vec::with_capacity(n),
        spread: Vec::with_capacity(n),
        dep1: Vec::with_capacity(n),
        b1v: Vec::with_capacity(n),
        a1v: Vec::with_capacity(n),
        tbid: Vec::with_capacity(n),
        task: Vec::with_capacity(n),
        to_cum: Vec::with_capacity(n),
        dep1amt: Vec::with_capacity(n),
        tr,
    };
    for s in mkt {
        let (b, a) = (s.total_bid_vol, s.total_ask_vol);
        let (b1, a1) = (s.bid_vols[0], s.ask_vols[0]);
        let (bp, ap) = (s.bid_prcs[0], s.ask_prcs[0]);
        if b + a <= 0.0 || b1 + a1 <= 0.0 || bp <= 0.0 || ap <= 0.0 || ap <= bp {
            continue;
        }
        r.ts.push(s.time_us);
        r.imb1.push((b1 - a1) / (b1 + a1));
        r.imbt.push((b - a) / (b + a));
        r.mid.push((ap + bp) / 2.0);
        r.spread.push(ap - bp);
        r.dep1.push(b1 + a1);
        r.b1v.push(b1);
        r.a1v.push(a1);
        r.tbid.push(b);
        r.task.push(a);
        r.to_cum.push(s.turnover);
        r.dep1amt.push(b1 * bp + a1 * ap);
        let b5: f64 = s.bid_vols[..5].iter().sum();
        let a5: f64 = s.ask_vols[..5].iter().sum();
        r.imb5.push((b5 - a5) / (b5 + a5 + 1e-9));
    }
    if r.ts.len() < 300 {
        return None;
    }
    Some(r)
}
/// 把一只股票当天的观测加进全市场直方图
fn add_hist(r: &Raw, h: &mut [Vec<u64>]) {
    for t in &r.tr {
        h[0][hbin(0, t.turnover)] += 1;
    }
    for w in r.tr.windows(2) {
        h[1][hbin(1, (w[1].time_us - w[0].time_us) as f64 + 1.0)] += 1;
    }
    for i in 0..r.ts.len() {
        h[2][hbin(2, r.spread[i] / r.mid[i] * 1e4)] += 1;
        h[3][hbin(3, r.dep1amt[i])] += 1;
        h[4][hbin(4, r.imb1[i].abs())] += 1;
    }
    for i in 1..r.ts.len() {
        let dt = ((r.ts[i] - r.ts[i - 1]) as f64 / 1e6).max(1e-3);
        h[5][hbin(5, ((r.to_cum[i] - r.to_cum[i - 1]).max(0.0)) / dt + 1.0)] += 1;
        let dm = (r.mid[i] - r.mid[i - 1]).abs() / r.mid[i - 1] * 1e4;
        if dm > 0.0 {
            h[6][hbin(6, dm)] += 1;
        }
    }
}
fn thr_from_hist(h: &[Vec<u64>]) -> Thr {
    let mut t = [[f64::NAN; 5]; NV];
    for v in 0..NV {
        let tot: u64 = h[v].iter().sum();
        if tot == 0 {
            continue;
        }
        let (lo, hi, _) = VAR_RANGE[v];
        for (qi, &q) in QS.iter().enumerate() {
            let need = (q * tot as f64).ceil() as u64;
            let mut cum = 0u64;
            for b in 0..NB {
                cum += h[v][b];
                if cum >= need {
                    t[v][qi] = hv_inv(v, lo + (b as f64 + 1.0) / NB as f64 * (hi - lo));
                    break;
                }
            }
        }
    }
    // 变量 1/5 记录时 +1，门槛减回去
    for qi in 0..5 {
        t[1][qi] -= 1.0;
        t[5][qi] -= 1.0;
    }
    t
}

// ====================== 位流 ======================
// 快照 Δ 网格：下标 i 表示第 i→i+1 张快照之间，取值在 t_{i+1} 时刻已知；目标都是 t_{i+1} 之后的未来。
const SN3: &[&str] = &[
    "mid", "imb1", "imbt", "imb5", "spr", "dep1", "bid1", "ask1", "tbid", "task",
];
const BN: &[&str] = &[
    "imb1p", "dimb1", "imbtp", "midu", "c1sgn", "sprd", "depu", "bid1u", "ask1u", "dimbt",
    "dimb5", "imb5p", "tbidd", "taskd", "agree",
    "spdq50", "spdq80", "spdq95", "sprq50", "sprq80", "depq50", "depq80", "aimbq80", "bigmv",
    "btr95", "btr80", "midchg", "imb1chg", "tpxu",
];
const TBN: &[&str] = &["amtq50", "amtq80", "amtq95", "amtq99", "burst", "odd"];
struct Flows {
    s3: Vec<Vec<i8>>, // 快照三态（涨1/平0/跌-1）
    b: Vec<Vec<u8>>,  // 快照 0/1 位流
    r60: Vec<f32>,
    r300: Vec<f32>,
    nd: Vec<f32>,   // 目标：60s/300s 收益(bp)、下一次真实变动方向(±1)
    tpx: Vec<i8>,   // 逐笔价格三态
    tbuy: Vec<u8>,  // 逐笔买方向
    tb: Vec<Vec<u8>>,
    tr60: Vec<f32>, // 逐笔时刻往后 60s 收益
}
#[inline]
fn tri(a: f64, b: f64) -> i8 {
    if b > a {
        1
    } else if b < a {
        -1
    } else {
        0
    }
}
/// at[k] 时刻往后 h 的中间价（最后一张 ts<=at+h 的快照）相对 base[k] 的对数收益(bp)；超出当天给 NaN
fn fwd_ret(ts: &[i64], mid: &[f64], at: &[i64], base: &[f64], h_us: i64) -> Vec<f32> {
    let last = *ts.last().unwrap();
    let mut j = 0usize;
    at.iter()
        .zip(base)
        .map(|(&t, &b)| {
            let tt = t + h_us;
            if tt > last {
                return f32::NAN;
            }
            while j + 1 < ts.len() && ts[j + 1] <= tt {
                j += 1;
            }
            ((mid[j] / b).ln() * 1e4) as f32
        })
        .collect()
}
fn build_flows(r: &Raw, thr: &Thr) -> Flows {
    let n = r.ts.len();
    let l = n - 1;
    let seq3 = [
        &r.mid, &r.imb1, &r.imbt, &r.imb5, &r.spread, &r.dep1, &r.b1v, &r.a1v, &r.tbid, &r.task,
    ];
    let s3: Vec<Vec<i8>> = seq3
        .iter()
        .map(|v| (0..l).map(|i| tri(v[i], v[i + 1])).collect())
        .collect();
    // 逐笔采样到快照网格
    let tr = &r.tr;
    let mut c1sgn = vec![0u8; l];
    let mut tpxu = vec![0u8; l];
    let mut btr95 = vec![0u8; l];
    let mut btr80 = vec![0u8; l];
    let mut j = 0usize; // 第一笔 time > t_i 的下标
    while j < tr.len() && tr[j].time_us <= r.ts[0] {
        j += 1;
    }
    for i in 0..l {
        let t1 = r.ts[i + 1];
        while j < tr.len() && tr[j].time_us <= t1 {
            if tr[j].turnover > thr[0][3] {
                btr95[i] = 1;
            }
            if tr[j].turnover > thr[0][2] {
                btr80[i] = 1;
            }
            j += 1;
        }
        if j > 0 {
            c1sgn[i] = (tr[j - 1].flag == 66) as u8;
            tpxu[i] = (j >= 2 && tr[j - 1].price > tr[j - 2].price) as u8;
        }
    }
    let gt = |v: &Vec<f64>, k: usize| -> u8 { (v[k] > 0.0) as u8 };
    let mut b: Vec<Vec<u8>> = vec![Vec::with_capacity(l); BN.len()];
    for i in 0..l {
        let k = i + 1;
        let dt = ((r.ts[k] - r.ts[i]) as f64 / 1e6).max(1e-3);
        let spd = (r.to_cum[k] - r.to_cum[i]).max(0.0) / dt;
        let sprbp = r.spread[k] / r.mid[k] * 1e4;
        let dm = (r.mid[k] - r.mid[i]).abs() / r.mid[i] * 1e4;
        let vals: [u8; 29] = [
            gt(&r.imb1, k),
            (s3[1][i] == 1) as u8,
            gt(&r.imbt, k),
            (s3[0][i] == 1) as u8,
            c1sgn[i],
            (s3[4][i] == -1) as u8,
            (s3[5][i] == 1) as u8,
            (s3[6][i] == 1) as u8,
            (s3[7][i] == 1) as u8,
            (s3[2][i] == 1) as u8,
            (s3[3][i] == 1) as u8,
            gt(&r.imb5, k),
            (s3[8][i] == 1) as u8,
            (s3[9][i] == 1) as u8,
            (c1sgn[i] == gt(&r.imbt, k)) as u8,
            (spd > thr[5][1]) as u8,
            (spd > thr[5][2]) as u8,
            (spd > thr[5][3]) as u8,
            (sprbp > thr[2][1]) as u8,
            (sprbp > thr[2][2]) as u8,
            (r.dep1amt[k] > thr[3][1]) as u8,
            (r.dep1amt[k] > thr[3][2]) as u8,
            (r.imb1[k].abs() > thr[4][2]) as u8,
            (dm > 0.0 && dm > thr[6][2]) as u8,
            btr95[i],
            btr80[i],
            (s3[0][i] != 0) as u8,
            (s3[1][i] != 0) as u8,
            tpxu[i],
        ];
        for (q, &x) in vals.iter().enumerate() {
            b[q].push(x);
        }
    }
    // 目标
    let at: Vec<i64> = r.ts[1..].to_vec();
    let base: Vec<f64> = r.mid[1..].to_vec();
    let r60 = fwd_ret(&r.ts, &r.mid, &at, &base, 60_000_000);
    let r300 = fwd_ret(&r.ts, &r.mid, &at, &base, 300_000_000);
    let mut nd = vec![f32::NAN; l];
    let mut nxt = f32::NAN;
    for i in (0..l).rev() {
        nd[i] = nxt; // 下标 i 之后（j>=i+1）第一个非零中间价变动
        if s3[0][i] != 0 {
            nxt = s3[0][i] as f32;
        }
    }
    // 逐笔网格
    let nt = tr.len();
    let mut tpx = vec![0i8; nt];
    for i in 1..nt {
        tpx[i] = tri(tr[i - 1].price, tr[i].price);
    }
    let tbuy: Vec<u8> = tr.iter().map(|t| (t.flag == 66) as u8).collect();
    let mut tb: Vec<Vec<u8>> = vec![Vec::with_capacity(nt); TBN.len()];
    for i in 0..nt {
        let a = tr[i].turnover;
        let gap = if i > 0 {
            (tr[i].time_us - tr[i - 1].time_us) as f64
        } else {
            f64::INFINITY
        };
        let v = [
            (a > thr[0][1]) as u8,
            (a > thr[0][2]) as u8,
            (a > thr[0][3]) as u8,
            (a > thr[0][4]) as u8,
            (gap < thr[1][0]) as u8,
            (tr[i].volume % 100.0 != 0.0) as u8,
        ];
        for (q, &x) in v.iter().enumerate() {
            tb[q].push(x);
        }
    }
    // 逐笔时刻的中间价（最后一张 ts<=t 的快照），之前没有快照的用第一张
    let mut jm = 0usize;
    let tat: Vec<i64> = tr.iter().map(|t| t.time_us).collect();
    let tbase: Vec<f64> = tat
        .iter()
        .map(|&t| {
            while jm + 1 < r.ts.len() && r.ts[jm + 1] <= t {
                jm += 1;
            }
            r.mid[jm]
        })
        .collect();
    let tr60 = fwd_ret(&r.ts, &r.mid, &tat, &tbase, 60_000_000);
    Flows {
        s3,
        b,
        r60,
        r300,
        nd,
        tpx,
        tbuy,
        tb,
        tr60,
    }
}

// ====================== 算子 ======================
fn rate_u8(s: &[u8]) -> f64 {
    if s.len() < 30 {
        return f64::NAN;
    }
    s.iter().map(|&x| x as u64).sum::<u64>() as f64 / s.len() as f64
}
fn ac1_f(v: &[f64]) -> f64 {
    let n = v.len();
    if n < 30 {
        return f64::NAN;
    }
    let m = n - 1;
    let (mut s1, mut s2, mut sx, mut v1, mut v2) = (0., 0., 0., 0., 0.);
    for i in 0..m {
        let (a, b) = (v[i], v[i + 1]);
        s1 += a;
        s2 += b;
        sx += a * b;
        v1 += a * a;
        v2 += b * b;
    }
    let mf = m as f64;
    let cov = sx / mf - (s1 / mf) * (s2 / mf);
    let va = v1 / mf - (s1 / mf).powi(2);
    let vb = v2 / mf - (s2 / mf).powi(2);
    if va <= 1e-12 || vb <= 1e-12 {
        return f64::NAN;
    }
    cov / (va * vb).sqrt()
}
fn ac1_u8(s: &[u8]) -> f64 {
    ac1_f(&s.iter().map(|&x| x as f64).collect::<Vec<_>>())
}
fn run_u8(s: &[u8]) -> f64 {
    if s.len() < 30 {
        return f64::NAN;
    }
    let runs = 1 + s.windows(2).filter(|w| w[0] != w[1]).count();
    s.len() as f64 / runs as f64
}
/// E[y | x=1] - E[y | x=0]，跳过 NaN 目标
fn edge_bin(x: &[u8], y: &[f32]) -> f64 {
    let (mut n1, mut s1, mut n0, mut s0) = (0u32, 0f64, 0u32, 0f64);
    for (&a, &t) in x.iter().zip(y) {
        if !t.is_finite() {
            continue;
        }
        if a == 1 {
            n1 += 1;
            s1 += t as f64;
        } else {
            n0 += 1;
            s0 += t as f64;
        }
    }
    if n1 < 20 || n0 < 20 {
        return f64::NAN;
    }
    s1 / n1 as f64 - s0 / n0 as f64
}
/// E[y | s=+1] - E[y | s=-1]
fn edge_tri(x: &[i8], y: &[f32]) -> f64 {
    let (mut n1, mut s1, mut n0, mut s0) = (0u32, 0f64, 0u32, 0f64);
    for (&a, &t) in x.iter().zip(y) {
        if !t.is_finite() {
            continue;
        }
        if a == 1 {
            n1 += 1;
            s1 += t as f64;
        } else if a == -1 {
            n0 += 1;
            s0 += t as f64;
        }
    }
    if n1 < 20 || n0 < 20 {
        return f64::NAN;
    }
    s1 / n1 as f64 - s0 / n0 as f64
}
/// 前后两半各算一次 edge，同号取均值、异号记 0（衡量这条关系在一天里稳不稳）
fn edge_stab(x: &[u8], y: &[f32]) -> f64 {
    let h = x.len() / 2;
    let e1 = edge_bin(&x[..h], &y[..h]);
    let e2 = edge_bin(&x[h..], &y[h..]);
    if !e1.is_finite() || !e2.is_finite() {
        return f64::NAN;
    }
    if e1 * e2 > 0.0 {
        (e1 + e2) / 2.0
    } else {
        0.0
    }
}

// ====================== 三元组：8 格统计 ======================
const NT: usize = 3; // 目标个数：r60, r300, nd
#[derive(Clone)]
struct C3 {
    n: usize,
    cnt: [u32; 8],
    jt: [u32; 64],
    marg: [f64; 3],
    // 全样本 & 前后两半：每格目标的 计数/和；每半的 Σy²
    tn: [[u32; 8]; NT],
    ts: [[f64; 8]; NT],
    hn: [[[u32; 8]; NT]; 2],
    hs: [[[f64; 8]; NT]; 2],
    hss: [[f64; NT]; 2],
}
fn c3_of(a: &[u8], b: &[u8], c: &[u8], tg: [&[f32]; NT]) -> C3 {
    let n = a.len();
    let mut z = C3 {
        n,
        cnt: [0; 8],
        jt: [0; 64],
        marg: [0.; 3],
        tn: [[0; 8]; NT],
        ts: [[0.; 8]; NT],
        hn: [[[0; 8]; NT]; 2],
        hs: [[[0.; 8]; NT]; 2],
        hss: [[0.; NT]; 2],
    };
    if n == 0 {
        return z;
    }
    let half = n / 2;
    let mut prev = 8usize;
    for i in 0..n {
        let cell = ((a[i] as usize) << 2) | ((b[i] as usize) << 1) | c[i] as usize;
        z.cnt[cell] += 1;
        z.marg[0] += a[i] as f64;
        z.marg[1] += b[i] as f64;
        z.marg[2] += c[i] as f64;
        if prev < 8 {
            z.jt[prev * 8 + cell] += 1;
        }
        prev = cell;
        let h = (i >= half) as usize;
        for k in 0..NT {
            let y = tg[k][i];
            if !y.is_finite() {
                continue;
            }
            let y = y as f64;
            z.tn[k][cell] += 1;
            z.ts[k][cell] += y;
            z.hn[h][k][cell] += 1;
            z.hs[h][k][cell] += y;
            z.hss[h][k] += y * y;
        }
    }
    for m in z.marg.iter_mut() {
        *m /= n as f64;
    }
    z
}
/// 在 h_train 半段学「每格的目标均值」，在另一半算预测值与真实目标的相关
fn c3_oos(z: &C3, k: usize, h_train: usize) -> f64 {
    let ht = 1 - h_train;
    let tot_n: u32 = z.hn[h_train][k].iter().sum();
    if tot_n < 100 {
        return f64::NAN;
    }
    let gm = z.hs[h_train][k].iter().sum::<f64>() / tot_n as f64;
    let mut pred = [0f64; 8];
    for c in 0..8 {
        let nn = z.hn[h_train][k][c];
        // 格内样本少就向整体均值收缩
        pred[c] = (z.hs[h_train][k][c] + 20.0 * gm) / (nn as f64 + 20.0);
    }
    let (mut n, mut sp, mut sy, mut spp, mut spy) = (0f64, 0f64, 0f64, 0f64, 0f64);
    for c in 0..8 {
        let nn = z.hn[ht][k][c] as f64;
        n += nn;
        sp += pred[c] * nn;
        spp += pred[c] * pred[c] * nn;
        sy += z.hs[ht][k][c];
        spy += pred[c] * z.hs[ht][k][c];
    }
    if n < 100.0 {
        return f64::NAN;
    }
    let syy = z.hss[ht][k];
    let cov = spy / n - sp / n * sy / n;
    let vp = spp / n - (sp / n).powi(2);
    let vy = syy / n - (sy / n).powi(2);
    if vp <= 1e-18 || vy <= 1e-18 {
        return f64::NAN;
    }
    cov / (vp * vy).sqrt()
}
/// 109 个「三个输入都用到、且取反只留一个（f(000)=0）」的三元布尔函数
fn canon_funcs() -> Vec<usize> {
    (0..256usize)
        .filter(|&f| {
            if f & 1 == 1 {
                return false;
            }
            (0..3).all(|k| (0..8).any(|c| ((f >> c) & 1) != ((f >> (c ^ (1 << k))) & 1)))
        })
        .collect()
}
const FSTATS: &[&str] = &["ac1", "lks", "lksx", "e60", "end"];
fn func_stat(z: &C3, f: usize, st: usize) -> f64 {
    let n = z.n;
    if n < 200 {
        return f64::NAN;
    }
    let nf = n as f64;
    let bit = |c: usize| (f >> c) & 1;
    match st {
        0 => {
            let p1 = (0..8).map(|c| bit(c) as f64 * z.cnt[c] as f64).sum::<f64>() / nf;
            let mut p11 = 0.0;
            for c1 in 0..8 {
                if bit(c1) == 1 {
                    for c2 in 0..8 {
                        if bit(c2) == 1 {
                            p11 += z.jt[c1 * 8 + c2] as f64;
                        }
                    }
                }
            }
            p11 /= (n - 1) as f64;
            let var = p1 * (1.0 - p1);
            if var <= 1e-12 {
                return f64::NAN;
            }
            (p11 - p1 * p1) / var
        }
        1 | 2 => {
            // 取反不变的「泄露」：max_i |2·P(z==x_i) - 1|；lksx 再减去三路独立时的同一量
            let (mut mx, mut mxb) = (0f64, 0f64);
            for who in 0..3 {
                let (mut lk, mut lkb) = (0.0, 0.0);
                for c in 0..8 {
                    let xbit = (c >> (2 - who)) & 1;
                    let mut pb = 1.0;
                    for w in 0..3 {
                        let p = z.marg[w];
                        pb *= if (c >> (2 - w)) & 1 == 1 {
                            p
                        } else {
                            1.0 - p
                        };
                    }
                    if bit(c) == xbit {
                        lk += z.cnt[c] as f64 / nf;
                        lkb += pb;
                    }
                }
                mx = mx.max((2.0 * lk - 1.0).abs());
                mxb = mxb.max((2.0 * lkb - 1.0).abs());
            }
            if st == 1 {
                mx
            } else {
                mx - mxb
            }
        }
        3 | 4 => {
            let k = if st == 3 { 0 } else { 2 };
            let (mut n1, mut s1, mut n0, mut s0) = (0u32, 0f64, 0u32, 0f64);
            for c in 0..8 {
                if bit(c) == 1 {
                    n1 += z.tn[k][c];
                    s1 += z.ts[k][c];
                } else {
                    n0 += z.tn[k][c];
                    s0 += z.ts[k][c];
                }
            }
            if n1 < 30 || n0 < 30 {
                return f64::NAN;
            }
            s1 / n1 as f64 - s0 / n0 as f64
        }
        _ => f64::NAN,
    }
}
// 三元组（下标指 BN）。前 10 组沿用拓展一的含义，个股分位门槛换成全市场门槛；后 4 组新增。
const TRIPS2: &[(usize, usize, usize)] = &[
    (1, 3, 0),
    (4, 3, 0),
    (7, 8, 0),
    (6, 3, 5),
    (9, 15, 2),
    (10, 11, 3),
    (25, 4, 3),
    (14, 3, 2),
    (12, 13, 21),
    (25, 28, 14),
    (24, 4, 0),
    (16, 26, 2),
    (19, 21, 0),
    (22, 0, 4),
];

// ====================== 特征清单（名字与计算同序） ======================
/// 拓展一回测入选的 63 列，原样搬进拓展二（名字必须在拓展一的 build_specs 里真实存在）
const KEEP: &[&str] = &[
    "T0_f070_lkx", "T0_f080_lkx", "T0_f223_lkx", "T0_f249_edgeI", "T1_f014_edgeM", "T1_f175_edgeM",
    "T2_f036_rate", "T2_f087_lk", "T2_f158_lkx", "T2_f178_lkx", "T2_f188_rate", "T4_f015_ac1",
    "T4_f039_ac1", "T4_f045_ac1", "T4_f100_ac1", "T4_f128_ac1", "T4_f136_ac1", "T4_f172_edgeM",
    "T4_f200_ac1", "T4_f243_ac1", "T5_f048_lkx", "T6_f028_lk", "T8_f011_edgeM", "T8_f041_edgeM",
    "T8_f047_rate", "T8_f064_lk", "T8_f072_lk", "T8_f077_ac1", "T8_f079_ac1", "T8_f092_ac1",
    "T8_f093_edgeM", "T8_f131_edgeM", "T8_f138_edgeM", "T8_f149_edgeM", "T8_f153_rate",
    "T8_f163_ac1", "T8_f171_edgeM", "T8_f229_rate", "T8_f240_ac1", "T8_f243_ac1", "T9_f248_ac1",
    "b_midd_midu_edge", "ck_dp_c1sgn_ac1", "u_cto_run", "u_dimbt_ac5", "u_dimbt_spcv",
    "u_imb1p_run", "u_midu_ac10", "u_midu_ac5", "u_tbidd_ac1", "u_tbig8_run", "u_togt_ac1",
    "u_togt_ctx6", "u_togt_run", "u_tosp_ac1", "u_tosp_ac10", "u_tosp_ac5", "u_tosp_run",
    "u_tosp_spcv", "u_vgt_ctx6", "u_vgt_run", "u_vq7_ac1", "u_vq7_run",
];
/// 拓展二新列的名字（5 个风格变量 + 8441 个新列里的 8436 个）
fn new_names() -> Vec<String> {
    let mut v = Vec::new();
    for s in ["lp", "lamt", "rv", "tick", "lnt"] {
        v.push(format!("ctl_{s}"));
    }
    for s in SN3 {
        for o in ["chg", "dir", "ac1c", "acchg", "e60", "end"] {
            v.push(format!("s3_{s}_{o}"));
        }
    }
    for o in ["chg", "dir", "ac1c", "acchg", "e60"] {
        v.push(format!("s3_tpx_{o}"));
    }
    for s in BN {
        for o in ["rate", "ac1", "run", "e60", "e300", "end", "e60s"] {
            v.push(format!("b_{s}_{o}"));
        }
    }
    for s in TBN {
        for o in ["rate", "ac1", "run", "buyc", "e60"] {
            v.push(format!("tb_{s}_{o}"));
        }
    }
    for o in ["rate", "ac1", "run", "e60"] {
        v.push(format!("tb_buy_{o}"));
    }
    let cf = canon_funcs();
    for k in 0..TRIPS2.len() {
        for c in 0..8 {
            v.push(format!("X{k}_c{c}"));
        }
        for t in ["m60", "m300", "mnd"] {
            for c in 0..8 {
                v.push(format!("X{k}_{t}_c{c}"));
            }
        }
        for t in ["patH", "oos60", "oos300", "oosnd"] {
            v.push(format!("X{k}_{t}"));
        }
        for &f in &cf {
            for s in FSTATS {
                v.push(format!("X{k}_f{f:03}_{s}"));
            }
        }
    }
    v
}
/// 风格变量 5 个：ln 中间价均值、ln 成交额、1 分钟收益标准差、最小报价单位占中间价、ln 笔数
fn ctl_feats(r: &Raw) -> [f64; 5] {
    let n = r.ts.len();
    let mmid = r.mid.iter().sum::<f64>() / n as f64;
    let amt: f64 = r.tr.iter().map(|t| t.turnover).sum();
    let mut rets = Vec::new();
    let mut next_t = r.ts[0] + 60_000_000;
    let mut last_mid = r.mid[0];
    for i in 0..n {
        if r.ts[i] >= next_t {
            rets.push((r.mid[i] / last_mid).ln() * 1e4);
            last_mid = r.mid[i];
            while next_t <= r.ts[i] {
                next_t += 60_000_000;
            }
        }
    }
    let rv = if rets.len() > 10 {
        let m = rets.iter().sum::<f64>() / rets.len() as f64;
        (rets.iter().map(|x| (x - m).powi(2)).sum::<f64>() / rets.len() as f64).sqrt()
    } else {
        f64::NAN
    };
    [
        mmid.ln(),
        (amt + 1.0).ln(),
        rv,
        (0.01 / mmid).ln(),
        (r.tr.len() as f64).ln(),
    ]
}
fn new_feats(fl: &Flows, ctl: [f64; 5], cf: &[usize], out: &mut Vec<f64>) {
    out.extend_from_slice(&ctl);
    let mid3 = &fl.s3[0];
    for (si, s) in fl.s3.iter().enumerate() {
        let nz: Vec<f64> = s.iter().filter(|&&x| x != 0).map(|&x| x as f64).collect();
        let (nu, ndn) = (
            nz.iter().filter(|&&x| x > 0.0).count(),
            nz.iter().filter(|&&x| x < 0.0).count(),
        );
        out.push(if s.len() >= 30 {
            nz.len() as f64 / s.len() as f64
        } else {
            f64::NAN
        });
        out.push(if nz.len() >= 20 {
            (nu as f64 - ndn as f64) / nz.len() as f64
        } else {
            f64::NAN
        });
        out.push(ac1_f(&nz));
        out.push(ac1_u8(&s.iter().map(|&x| (x != 0) as u8).collect::<Vec<_>>()));
        out.push(edge_tri(s, &fl.r60));
        // 自身是中间价时，「下一次真实变动」就是事件时间下的反转/延续
        out.push(if si == 0 {
            edge_tri(mid3, &fl.nd)
        } else {
            edge_tri(s, &fl.nd)
        });
    }
    {
        let s = &fl.tpx;
        let nz: Vec<f64> = s.iter().filter(|&&x| x != 0).map(|&x| x as f64).collect();
        let (nu, ndn) = (
            nz.iter().filter(|&&x| x > 0.0).count(),
            nz.iter().filter(|&&x| x < 0.0).count(),
        );
        out.push(if s.len() >= 30 {
            nz.len() as f64 / s.len() as f64
        } else {
            f64::NAN
        });
        out.push(if nz.len() >= 20 {
            (nu as f64 - ndn as f64) / nz.len() as f64
        } else {
            f64::NAN
        });
        out.push(ac1_f(&nz));
        out.push(ac1_u8(&s.iter().map(|&x| (x != 0) as u8).collect::<Vec<_>>()));
        out.push(edge_tri(s, &fl.tr60));
    }
    for b in &fl.b {
        out.push(rate_u8(b));
        out.push(ac1_u8(b));
        out.push(run_u8(b));
        out.push(edge_bin(b, &fl.r60));
        out.push(edge_bin(b, &fl.r300));
        out.push(edge_bin(b, &fl.nd));
        out.push(edge_stab(b, &fl.r60));
    }
    let buyf: Vec<f32> = fl.tbuy.iter().map(|&x| x as f32).collect();
    for b in &fl.tb {
        out.push(rate_u8(b));
        out.push(ac1_u8(b));
        out.push(run_u8(b));
        out.push(edge_bin(b, &buyf));
        out.push(edge_bin(b, &fl.tr60));
    }
    out.push(rate_u8(&fl.tbuy));
    out.push(ac1_u8(&fl.tbuy));
    out.push(run_u8(&fl.tbuy));
    out.push(edge_bin(&fl.tbuy, &fl.tr60));
    for &(ia, ib, ic) in TRIPS2 {
        let z = c3_of(&fl.b[ia], &fl.b[ib], &fl.b[ic], [&fl.r60, &fl.r300, &fl.nd]);
        let nf = z.n as f64;
        let ok = z.n >= 200;
        for c in 0..8 {
            out.push(if ok { z.cnt[c] as f64 / nf } else { f64::NAN });
        }
        for k in 0..NT {
            let tot: u32 = z.tn[k].iter().sum();
            let gm = if tot > 0 {
                z.ts[k].iter().sum::<f64>() / tot as f64
            } else {
                f64::NAN
            };
            for c in 0..8 {
                out.push(if ok && z.tn[k][c] >= 30 {
                    z.ts[k][c] / z.tn[k][c] as f64 - gm
                } else {
                    f64::NAN
                });
            }
        }
        out.push(if ok {
            -(0..8)
                .filter(|&c| z.cnt[c] > 0)
                .map(|c| {
                    let f = z.cnt[c] as f64 / nf;
                    f * f.ln()
                })
                .sum::<f64>()
        } else {
            f64::NAN
        });
        for k in 0..NT {
            let (a, b) = (c3_oos(&z, k, 0), c3_oos(&z, k, 1));
            out.push(if a.is_finite() && b.is_finite() {
                (a + b) / 2.0
            } else {
                f64::NAN
            });
        }
        for &f in cf {
            for st in 0..FSTATS.len() {
                out.push(func_stat(&z, f, st));
            }
        }
    }
}

// ====================== 横截面：z-score + 风格残差 ======================
fn zscore_cols(mat: &mut [f64], nr: usize, nc: usize, skip: &[bool]) {
    let stats: Vec<Option<(f64, f64)>> = (0..nc)
        .into_par_iter()
        .map(|j| {
            if skip[j] {
                return None;
            }
            let fin: Vec<f64> = (0..nr).map(|i| mat[i * nc + j]).filter(|x| x.is_finite()).collect();
            if fin.len() < 30 {
                return None;
            }
            let m = fin.iter().sum::<f64>() / fin.len() as f64;
            let s = (fin.iter().map(|x| (x - m).powi(2)).sum::<f64>() / fin.len() as f64).sqrt();
            if s < 1e-12 {
                None
            } else {
                Some((m, s))
            }
        })
        .collect();
    mat.par_chunks_mut(nc).for_each(|r| {
        for (j, x) in r.iter_mut().enumerate() {
            if skip[j] {
                continue;
            }
            match stats[j] {
                Some((m, s)) if x.is_finite() => *x = (*x - m) / s,
                Some(_) => {}
                None => *x = f64::NAN,
            }
        }
    });
}
/// 小矩阵高斯消元（带部分主元），解 A x = b
fn solve(mut a: Vec<f64>, mut b: Vec<f64>, k: usize) -> Option<Vec<f64>> {
    for c in 0..k {
        let p = (c..k).max_by(|&i, &j| a[i * k + c].abs().partial_cmp(&a[j * k + c].abs()).unwrap())?;
        if a[p * k + c].abs() < 1e-10 {
            return None;
        }
        if p != c {
            for t in 0..k {
                a.swap(p * k + t, c * k + t);
            }
            b.swap(p, c);
        }
        for r in c + 1..k {
            let f = a[r * k + c] / a[c * k + c];
            for t in c..k {
                a[r * k + t] -= f * a[c * k + t];
            }
            b[r] -= f * b[c];
        }
    }
    let mut x = vec![0.0; k];
    for c in (0..k).rev() {
        let s: f64 = (c + 1..k).map(|t| a[c * k + t] * x[t]).sum();
        x[c] = (b[c] - s) / a[c * k + c];
    }
    Some(x)
}
/// 每列（已 z-score，先截到 ±5）对风格变量做横截面 OLS，残差再 z-score
fn resid_cols(z: &[f64], nr: usize, nc: usize, ctl: &[Vec<f64>]) -> Vec<f64> {
    let k = ctl[0].len() + 1;
    let cols: Vec<Vec<f64>> = (0..nc)
        .into_par_iter()
        .map(|j| {
            let mut out = vec![f64::NAN; nr];
            let mut xtx = vec![0.0; k * k];
            let mut xty = vec![0.0; k];
            let mut rows = Vec::new();
            let mut x = vec![0.0; k];
            for i in 0..nr {
                let y = z[i * nc + j];
                if !y.is_finite() || ctl[i].iter().any(|v| !v.is_finite()) {
                    continue;
                }
                let y = y.clamp(-5.0, 5.0);
                x[0] = 1.0;
                x[1..].copy_from_slice(&ctl[i]);
                for a in 0..k {
                    xty[a] += x[a] * y;
                    for b in 0..k {
                        xtx[a * k + b] += x[a] * x[b];
                    }
                }
                rows.push(i);
            }
            if rows.len() < 100 {
                return out;
            }
            let beta = match solve(xtx, xty, k) {
                Some(b) => b,
                None => return out,
            };
            let mut res = Vec::with_capacity(rows.len());
            for &i in &rows {
                let y = z[i * nc + j].clamp(-5.0, 5.0);
                let p = beta[0] + (1..k).map(|a| beta[a] * ctl[i][a - 1]).sum::<f64>();
                res.push(y - p);
            }
            let m = res.iter().sum::<f64>() / res.len() as f64;
            let s = (res.iter().map(|v| (v - m).powi(2)).sum::<f64>() / res.len() as f64).sqrt();
            if s < 1e-12 {
                return out;
            }
            for (&i, &v) in rows.iter().zip(&res) {
                out[i] = (v - m) / s;
            }
            out
        })
        .collect();
    let mut m = vec![f64::NAN; nr * nc];
    for (j, c) in cols.iter().enumerate() {
        for i in 0..nr {
            m[i * nc + j] = c[i];
        }
    }
    m
}
/// 当日 ln 总市值：6 位代码 → ln(总市值)。
/// 只认日历里存在的当天，日期缺失时返回空表（该日 8499 个残差列给 NaN，绝不拿前一天的市值顶替）。
fn logcap_of_day(date: i64) -> HashMap<String, f64> {
    let (dates, codes) = crate::yhyb_indtop_metrics::cap_maps();
    let r = match dates.iter().position(|&d| d == date) {
        Some(r) => r,
        None => return HashMap::new(),
    };
    codes
        .iter()
        .zip(crate::yhyb_indtop_metrics::cap_row(r, codes.len()))
        .filter(|(_, v)| v.is_finite() && *v > 0.0)
        .map(|(c, v)| (c.clone(), v.ln()))
        .collect()
}

// ====================== 对外入口（横截面：一天全市场） ======================

/// 基础列数：2 计数 + 63 个拓展一保留列 + 8441 个新列
const BASE_LEN: usize = 8506;
/// 输出列数 = 基础列数 + 残差列数（基础列里 7 列不参与回归，故残差列 8499）
pub const OUT_LEN: usize = 17005;
/// 不参与 z-score 与残差的列：2 个计数与 5 个风格变量
const N_SKIP: usize = 7;

fn keep_specs() -> Vec<&'static ext1::Spec> {
    static S: OnceLock<Vec<&'static ext1::Spec>> = OnceLock::new();
    S.get_or_init(|| ext1::specs_of_names(KEEP)).clone()
}

fn canon() -> &'static [usize] {
    static C: OnceLock<Vec<usize>> = OnceLock::new();
    C.get_or_init(canon_funcs)
}

/// 列名与输出严格对齐（进程内只算一次）：2 计数 + 拓展一保留列 + 新列 + 残差列
pub fn lfsr_ext2_names() -> Vec<String> {
    static N: OnceLock<Vec<String>> = OnceLock::new();
    N.get_or_init(|| {
        // 基础列：2 个计数 + 拓展一保留列 + 拓展二新列
        let mut base: Vec<String> = Vec::with_capacity(BASE_LEN);
        base.push("nsnap".into());
        base.push("ntrade".into());
        base.extend(KEEP.iter().map(|s| s.to_string()));
        base.extend(new_names());
        assert_eq!(base.len(), BASE_LEN, "拓展二基础列数必须是 {BASE_LEN}");
        // 残差列：基础列里除 2 个计数与 5 个风格变量之外，每一列都再来一个 _r
        let ctl_idx = 2 + KEEP.len();
        let mut v = base.clone();
        for (j, n) in base.iter().enumerate() {
            if j < 2 || (j >= ctl_idx && j < ctl_idx + 5) {
                continue;
            }
            v.push(format!("{n}_r"));
        }
        assert_eq!(v.len(), OUT_LEN, "拓展二列数必须是 {OUT_LEN}");
        v
    })
    .clone()
}

/// 单日全市场核心：读当天全市场 → 全市场门槛 → 每股特征 → 横截面 z-score 与风格残差 → (codes, vals)。
/// codes 是当天 universe 的全部股票（不是「算得出来的那些」），
/// vals 按股票顺序展开，长度恒为 codes.len() * OUT_LEN。
pub fn compute_lfsr_ext2_full(date: i64) -> std::io::Result<(Vec<String>, Vec<f32>)> {
    let codes = ext1::list_codes(date)?;
    let specs = keep_specs();
    let cf = canon();
    let nr = codes.len();
    let nc = BASE_LEN;
    let ctl_idx = 2 + KEEP.len();
    // ① 逐股读数据 + 拓展一保留列 + 原始序列；有效口径与拓展一完全一致（parts 与 codes 同序）
    let parts: Vec<Option<(Vec<f64>, Raw)>> = codes
        .par_iter()
        .map(|code| {
            let mkt = read_market_f64_inner(code, date, false, true).ok()?;
            let tr = read_trade_f64_inner(code, date, false, true).ok()?;
            let env = ext1::build_env(&mkt, &tr)?;
            let keep: Vec<f64> = specs.iter().map(|sp| ext1::eval_one(sp, &env)).collect();
            let raw = build_raw(&mkt, tr)?;
            let mut v = Vec::with_capacity(nc);
            v.push(raw.ts.len() as f64);
            v.push(raw.tr.len() as f64);
            v.extend(keep);
            Some((v, raw))
        })
        .collect();
    let valid = parts.iter().filter(|p| p.is_some()).count();
    // 整日一只都算不出来（典型是 2016-01-07 那种只有十几分钟交易的熔断日）：
    // 仍然发全 universe 的 NaN 行，绝不返回空 batch —— 空的结果会被父进程判成
    // 「这一天没完成」而让整次提交失败，而这一天本来就应该有行、只是全为空值。
    if valid == 0 {
        eprintln!("lfsr_ext2 {date}: 整日无有效股票（多半是熔断日），发 {nr} 行全 NaN");
        return Ok((codes, vec![f32::NAN; nr * OUT_LEN]));
    }
    // ② 全市场门槛：只用有效股票的观测
    let mut h = vec![vec![0u64; NB]; NV];
    for p in parts.iter().flatten() {
        add_hist(&p.1, &mut h);
    }
    let thr = thr_from_hist(&h);
    // ③ 逐股位流特征
    let rows: Vec<Option<Vec<f64>>> = parts
        .into_par_iter()
        .map(|p| {
            p.map(|(mut v, raw)| {
                let fl = build_flows(&raw, &thr);
                new_feats(&fl, ctl_feats(&raw), cf, &mut v);
                debug_assert_eq!(v.len(), nc, "拓展二基础列数必须恒定");
                v
            })
        })
        .collect();
    // ④ 摊成矩阵：算不出来的股票整行 NaN，保证当天 universe 每只股票都有一行
    let mut mat = vec![f64::NAN; nr * nc];
    for (i, r) in rows.into_iter().enumerate() {
        if let Some(v) = r {
            mat[i * nc..(i + 1) * nc].copy_from_slice(&v);
        }
    }
    // 风格变量：ctl_* 五列原始值（未被 z-score 动过）+ ln 总市值（当天市值缺失则不加）
    let logcap = logcap_of_day(date);
    let has_cap = !logcap.is_empty();
    if !has_cap {
        eprintln!("lfsr_ext2 {date}: 市值表里没有这一天，{}-{} 个残差列给 NaN", nc - N_SKIP, OUT_LEN);
    }
    let ctl: Vec<Vec<f64>> = (0..nr)
        .map(|i| {
            let mut c: Vec<f64> = (0..5).map(|t| mat[i * nc + ctl_idx + t]).collect();
            if has_cap {
                c.push(*logcap.get(&codes[i]).unwrap_or(&f64::NAN));
            }
            c
        })
        .collect();
    // ⑤ 横截面 z-score（计数与风格变量除外）→ 风格回归残差
    let mut skip = vec![false; nc];
    skip[0] = true;
    skip[1] = true;
    for j in ctl_idx..ctl_idx + 5 {
        skip[j] = true;
    }
    zscore_cols(&mut mat, nr, nc, &skip);
    let res = resid_cols(&mat, nr, nc, &ctl);
    let res_keep: Vec<usize> = (0..nc).filter(|&j| !skip[j]).collect();
    let mut vals: Vec<f32> = Vec::with_capacity(nr * OUT_LEN);
    for i in 0..nr {
        vals.extend(mat[i * nc..(i + 1) * nc].iter().map(|&x| x as f32));
        for &j in &res_keep {
            vals.push(res[i * nc + j] as f32);
        }
    }
    eprintln!("lfsr_ext2 {date}: {} 只（有效 {valid}）× {OUT_LEN} 列", nr);
    Ok((codes, vals))
}

/// 单日全市场调试入口（错误抛异常，可见完整栈）
#[pyfunction]
pub fn py_lfsr_ext2(date: i64) -> PyResult<(Vec<String>, Vec<f32>)> {
    compute_lfsr_ext2_full(date).map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("{e:?}")))
}

#[pyfunction]
pub fn py_lfsr_ext2_names() -> Vec<String> {
    lfsr_ext2_names()
}
