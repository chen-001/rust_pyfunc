//! 切换时刻 × 模拟退火 —— 用真实数据验证 Multica COR-15 评论中的退火拓展想法。
//!
//! 【背景】旧 agent 实测"相邻两分钟截面排名直接退火"与 Pearson/Spearman r 高度冗余;
//! 本程序把旧 agent 未验证的改进方向真正实现并逐个实测:
//!   方向 D(重点): 每只股票日内"有成交分钟"的 buy_ratio 序列 → 柜员游戏式退火
//!     guess = sort(序列), target = 原序列, xorshift64 固定种子 + 线性降温 Metropolis,
//!     r = 1 - Σ(guess-target)²/(2σ²N)。测量"单序列内部时间结构的可恢复性"。
//!   方向 C(重点): 低 r 时刻的全市场 m1 贡献度向量 c_i = dx·dy/√(Sxx·Syy)
//!     对 c 做同样退火, 测"贡献度结构恢复难度", 并对照 c 的集中度(Gini/top5%)
//!     检验是否真的区分"局部切换 vs 全局漂移"。
//!   方向 B(顺带): 低 r 时刻截面排名用"贪心+退火"(优先交换错位最大的对)恢复, 数阶梯。
//!
//! 【数据】/ssd_data/stock/{date}/transaction/{code}_{date}_transaction.csv 逐笔,
//!   read_trade_fast 已做 adjust_afternoon(下午 13:00-14:56 平移到 11:30-14:56)。
//!   分钟分桶: idx = (time_us - 09:30:00) / 60s, 上午 idx 0..119, 下午 idx 120..236, 共 237。
//!
//! 【输出】stdout 一个 JSON(日志走 stderr)。Python 侧加载做 IC 分析。
//!
//! 构建: cd sandbox_switch_anneal && cargo build --release
//! 运行: ./target/release/switch_anneal_sandbox 20240104 > out_20240104.json

mod fast_csv_reader;
use fast_csv_reader::{read_trade_fast, TradeRecord};
use rayon::prelude::*;
use serde::Serialize;
use std::collections::BTreeSet;

const N_MIN: usize = 237; // 分钟数: 上午120 + 下午117
const M_CAP: usize = 500_000; // 退火步数上限(参考正式库 M_MAX_SCALAR)
const SEED: u64 = 0x9E37_79B9_7F4A_7C15; // 与正式库同一固定种子
const MIN_US: i64 = 60 * 1_000_000;
const N_CS_MOMENTS: usize = 3; // 每个日期选 3 个低 r 时刻做方向 C/B

// ============================================================================
// xorshift64 — 与正式库 annnealing 相同的确定性 PRNG
// ============================================================================

struct XorShift64 {
    state: u64,
}

impl XorShift64 {
    #[inline]
    fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 { 0xDEAD_BEEF_DEAD_BEEF } else { seed },
        }
    }

    #[inline]
    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    #[inline]
    fn next_index(&mut self, n: usize) -> usize {
        if n == 0 {
            return 0;
        }
        ((self.next_u64() >> 32).wrapping_mul(n as u64) >> 32) as usize
    }
}

// ============================================================================
// 单序列退火(柜员游戏核心, 方向 D 与方向 C 共用)
// ============================================================================

#[derive(Serialize, Clone, Copy)]
pub struct SeqFeat {
    pub r0: f32,        // 初始-目标匹配度 = 1 - Σ(sort-x)²/(2σ²N)
    pub steps50: u32,   // 达到 r>=0.50 的步数(未到则 M)
    pub steps70: u32,
    pub steps90: u32,
    pub half_life: u32, // 达到 (1+r0)/2 的步数
    pub final_r: f32,
    pub inertia: f32,   // Σ(1-r) 曲线下面积
    pub declines: u32,  // r 下降步数
    pub dr_std: f32,    // Δr 标准差
    pub ladder: u32,    // 阶梯数: 单步 Δr>=0.01 的次数
    pub max_jump: f32,  // 最大单步跳升
}

pub fn anneal_seq(x: &[f32], m_max: usize) -> Option<SeqFeat> {
    let n = x.len();
    if n < 8 {
        return None;
    }
    let mean = x.iter().map(|&v| v as f64).sum::<f64>() / n as f64;
    let sigma2 = x
        .iter()
        .map(|&v| {
            let d = v as f64 - mean;
            d * d
        })
        .sum::<f64>()
        / n as f64;
    if sigma2 <= 0.0 {
        return None;
    }
    let mut guess: Vec<f32> = x.to_vec();
    guess.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mut s: f64 = guess
        .iter()
        .zip(x.iter())
        .map(|(g, t)| {
            let d = *g as f64 - *t as f64;
            d * d
        })
        .sum();
    let denom = 2.0 * sigma2 * n as f64;
    let inv_denom = 1.0 / denom;
    let r0 = 1.0 - s * inv_denom;
    let hl_target = (1.0 + r0) * 0.5;
    let s_tol = (sigma2 * n as f64 * 1e-10).max(1e-12);
    let ct_base = sigma2; // C1_FRAC = 1.0
    let inv_m = if m_max > 1 { 1.0 / (m_max as f64 - 1.0) } else { 0.0 };
    let mut rng = XorShift64::new(SEED);
    let (mut st50, mut st70, mut st90, mut hl) = (u32::MAX, u32::MAX, u32::MAX, u32::MAX);
    let mut prev_r = r0;
    let mut final_r = r0;
    let mut inertia = 0.0f64;
    let mut decl = 0u32;
    let (mut dr_s1, mut dr_s2) = (0.0f64, 0.0f64);
    let mut drn = 0u32;
    let mut ladder = 0u32;
    let mut max_jump = 0.0f32;
    let mut t = 0usize;
    while t < m_max {
        let i = rng.next_index(n);
        let mut j = rng.next_index(n);
        if j == i {
            j = rng.next_index(n);
        }
        if i != j {
            let gi = guess[i];
            let gj = guess[j];
            let ti = x[i];
            let tj = x[j];
            let ds = 2.0f64 * (gi as f64 - gj as f64) * (ti as f64 - tj as f64);
            let ct = ct_base * (1.0 - t as f64 * inv_m).max(0.0);
            if ds < 0.0 || ds < ct {
                s += ds;
                guess.swap(i, j);
            }
        }
        let cr = 1.0 - s * inv_denom;
        final_r = cr;
        if t > 0 {
            let dv = (cr - prev_r) as f32;
            drn += 1;
            dr_s1 += dv as f64;
            dr_s2 += (dv as f64) * (dv as f64);
            if dv < 0.0 {
                decl += 1;
            }
            if dv >= 0.01 {
                ladder += 1;
                max_jump = max_jump.max(dv);
            }
        }
        if hl == u32::MAX && cr >= hl_target {
            hl = t as u32;
        }
        if st50 == u32::MAX && cr >= 0.50 {
            st50 = t as u32;
        }
        if st70 == u32::MAX && cr >= 0.70 {
            st70 = t as u32;
        }
        if st90 == u32::MAX && cr >= 0.90 {
            st90 = t as u32;
        }
        inertia += 1.0 - cr;
        prev_r = cr;
        if s <= s_tol {
            break;
        }
        t += 1;
    }
    let dr_std = if drn >= 2 {
        ((dr_s2 - dr_s1 * dr_s1 / drn as f64) / (drn as f64 - 1.0))
            .max(0.0)
            .sqrt() as f32
    } else {
        f32::NAN
    };
    Some(SeqFeat {
        r0: r0 as f32,
        steps50: if st50 == u32::MAX { m_max as u32 } else { st50 },
        steps70: if st70 == u32::MAX { m_max as u32 } else { st70 },
        steps90: if st90 == u32::MAX { m_max as u32 } else { st90 },
        half_life: if hl == u32::MAX { m_max as u32 } else { hl },
        final_r: final_r as f32,
        inertia: inertia as f32,
        declines: decl,
        dr_std,
        ladder,
        max_jump,
    })
}

/// 自适应步数: M = min(cap, max(2000, N²×10)) — 与正式库 adaptive_m_max 同式
fn adaptive_m_max(n: usize) -> usize {
    (n * n * 10).max(2000).min(M_CAP)
}

// ============================================================================
// 方向 D: 每股日内分钟 buy_ratio 序列退火 + 对照特征
// ============================================================================

#[derive(Serialize, Clone, Copy, Default)]
pub struct StockFeatD {
    pub n_min: u32,      // 有成交分钟数
    pub r0: f32,
    pub steps50: u32,
    pub steps70: u32,
    pub steps90: u32,
    pub half_life: u32,
    pub final_r: f32,
    pub inertia: f32,
    pub declines: u32,
    pub dr_std: f32,
    pub ladder: u32,
    pub max_jump: f32,
    pub mean: f32,       // 对照: 全天 buy_ratio 均值(原始237分钟, 无值跳过)
    pub std: f32,        // 对照
    pub ac1: f32,        // 对照
    pub trend: f32,      // 对照: 分钟索引线性斜率
    pub morn_aft_diff: f32, // 对照: 上午均值 - 下午均值
}

pub const D_NAMES: &[&str] = &[
    "n_min", "r0", "steps50", "steps70", "steps90", "half_life", "final_r", "inertia",
    "declines", "dr_std", "ladder", "max_jump", "mean", "std", "ac1", "trend", "morn_aft_diff",
];

/// 读逐笔 → 分钟聚合 + 方向 D 退火特征 + 分钟矩阵(供截面)
pub fn per_stock_d(trades: &[TradeRecord]) -> Option<(StockFeatD, Vec<f64>, Vec<f64>)> {
    if trades.is_empty() {
        return None;
    }
    // 以首笔成交为 t_open(与正式库 compute_anneal_volume_full 同口径),
    // 分钟索引 = (time_us - t_open) / 60s; reader 已剔除盘前/盘后, t_open 在 09:30 后
    let t_open = trades[0].time_us;
    let mut buy = vec![0.0f64; N_MIN];
    let mut amt = vec![0.0f64; N_MIN];
    for t in trades {
        let idx = (t.time_us - t_open) / MIN_US;
        if idx < 0 || (idx as usize) >= N_MIN {
            continue;
        }
        let i = idx as usize;
        amt[i] += t.turnover;
        if t.flag == 66 {
            buy[i] += t.turnover;
        }
    }
    // 压缩序列(有成交分钟) — 退火输入
    let mut xs: Vec<f32> = Vec::new();
    for i in 0..N_MIN {
        if amt[i] > 0.0 {
            xs.push((buy[i] / amt[i]) as f32);
        }
    }
    let nv = xs.len();
    if nv < 30 {
        return None;
    }
    let feat = anneal_seq(&xs, adaptive_m_max(nv))?;
    // 对照统计(原始 237 序列, 无值跳过)
    let mut vals: Vec<(usize, f32)> = Vec::new(); // (分钟idx, ratio)
    for i in 0..N_MIN {
        if amt[i] > 0.0 {
            vals.push((i, (buy[i] / amt[i]) as f32));
        }
    }
    let mean = vals.iter().map(|&(_, v)| v as f64).sum::<f64>() / vals.len() as f64;
    let std = (vals
        .iter()
        .map(|&(_, v)| {
            let d = v as f64 - mean;
            d * d
        })
        .sum::<f64>()
        / (vals.len() as f64 - 1.0))
        .sqrt();
    let mut ac1 = f32::NAN;
    if vals.len() >= 3 {
        let mut sxy = 0.0f64;
        let mut sxx = 0.0f64;
        for w in vals.windows(2) {
            let a = w[0].1 as f64 - mean;
            let b = w[1].1 as f64 - mean;
            sxy += a * b;
            sxx += a * a;
        }
        ac1 = (sxy / sxx) as f32;
    }
    let mut trend = f32::NAN;
    if vals.len() >= 3 {
        let mut sxx = 0.0f64;
        let mut sxy = 0.0f64;
        let mx = vals.iter().map(|&(i, _)| i as f64).sum::<f64>() / vals.len() as f64;
        let my = mean;
        for &(i, v) in &vals {
            let dx = i as f64 - mx;
            sxx += dx * dx;
            sxy += dx * (v as f64 - my);
        }
        if sxx > 1e-12 {
            trend = (sxy / sxx) as f32;
        }
    }
    let mut morn_s = 0.0f64;
    let mut morn_n = 0u32;
    let mut aft_s = 0.0f64;
    let mut aft_n = 0u32;
    for &(i, v) in &vals {
        if i < 120 {
            morn_s += v as f64;
            morn_n += 1;
        } else {
            aft_s += v as f64;
            aft_n += 1;
        }
    }
    let m_a_diff = if morn_n > 0 && aft_n > 0 {
        (morn_s / morn_n as f64 - aft_s / aft_n as f64) as f32
    } else {
        f32::NAN
    };
    Some((
        StockFeatD {
            n_min: nv as u32,
            r0: feat.r0,
            steps50: feat.steps50,
            steps70: feat.steps70,
            steps90: feat.steps90,
            half_life: feat.half_life,
            final_r: feat.final_r,
            inertia: feat.inertia,
            declines: feat.declines,
            dr_std: feat.dr_std,
            ladder: feat.ladder,
            max_jump: feat.max_jump,
            mean: mean as f32,
            std: std as f32,
            ac1,
            trend,
            morn_aft_diff: m_a_diff,
        },
        buy,
        amt,
    ))
}

// ============================================================================
// 截面部分: r 序列 / m1 贡献度 / 方向 C / 方向 B
// ============================================================================

/// 截面 Pearson r(两分钟 buy_ratio, 两分钟都有成交的股票)
fn cross_r(x: &[f32], y: &[f32]) -> f64 {
    let n = x.len();
    if n < 8 {
        return f64::NAN;
    }
    let mx = x.iter().map(|&v| v as f64).sum::<f64>() / n as f64;
    let my = y.iter().map(|&v| v as f64).sum::<f64>() / n as f64;
    let mut sxy = 0.0f64;
    let mut sxx = 0.0f64;
    let mut syy = 0.0f64;
    for i in 0..n {
        let dx = x[i] as f64 - mx;
        let dy = y[i] as f64 - my;
        sxy += dx * dy;
        sxx += dx * dx;
        syy += dy * dy;
    }
    let den = (sxx * syy).sqrt();
    if den < 1e-20 {
        return f64::NAN;
    }
    sxy / den
}

/// m1 贡献度向量 c_i = dx_i·dy_i / √(Sxx·Syy), 保证 Σc_i = r
fn m1_contrib(x: &[f32], y: &[f32]) -> Vec<f32> {
    let n = x.len();
    let mx = x.iter().map(|&v| v as f64).sum::<f64>() / n as f64;
    let my = y.iter().map(|&v| v as f64).sum::<f64>() / n as f64;
    let sxx = x.iter().map(|&v| {
        let d = v as f64 - mx;
        d * d
    }).sum::<f64>();
    let syy = y.iter().map(|&v| {
        let d = v as f64 - my;
        d * d
    }).sum::<f64>();
    let den = (sxx * syy).sqrt();
    (0..n)
        .map(|i| ((x[i] as f64 - mx) * (y[i] as f64 - my) / den) as f32)
        .collect()
}

/// Gini 系数(升序公式, 用 |c|)
fn gini_of(v: &[f32]) -> f64 {
    let mut s: Vec<f64> = v.iter().map(|&x| (x as f64).abs()).collect();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = s.len();
    let sum: f64 = s.iter().sum();
    if sum <= 0.0 {
        return f64::NAN;
    }
    let mut g = 0.0f64;
    for (i, &x) in s.iter().enumerate() {
        g += (2.0 * (i as f64 + 1.0) - n as f64 - 1.0) * x;
    }
    g / (n as f64 * sum)
}

/// top5% 集中度: |c| 降序前 5% 的份额
fn top5_share(v: &[f32]) -> f64 {
    let mut s: Vec<f64> = v.iter().map(|&x| (x as f64).abs()).collect();
    s.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let n = s.len();
    let sum: f64 = s.iter().sum();
    let k = ((n as f64) * 0.05).ceil() as usize;
    if sum <= 0.0 || k == 0 {
        return f64::NAN;
    }
    s[..k.min(n)].iter().sum::<f64>() / sum
}

#[derive(Serialize)]
pub struct MomentCS {
    pub t: usize,   // 分钟对索引: 该分钟与前一分钟
    pub r: f64,     // 该时刻截面 Pearson r
    pub n: usize,   // 股票数
    // 方向 C: m1 贡献度向量退火
    pub c_r0: f32,
    pub c_steps50: u32,
    pub c_steps70: u32,
    pub c_steps90: u32,
    pub c_half_life: u32,
    pub c_final_r: f32,
    pub c_inertia: f32,
    pub c_declines: u32,
    pub c_ladder: u32,
    pub c_gini: f64,
    pub c_top5pct: f64,
    pub c_std: f32,
    pub c_kurt: f64,
    pub c_curve: Vec<(u32, f32)>, // 采样曲线 (step, r), ~200 点
    pub c_vec: Vec<f32>,          // m1 贡献度向量本体(perm 对照用)
    // 方向 B: 贪心+退火 截面排名恢复
    pub b_r0: f32,
    pub b_final_r: f32,
    pub b_steps70: u32,
    pub b_steps90: u32,
    pub b_ladder: u32,
    pub b_curve: Vec<(u32, f32)>,
}

/// 方向 B: 贪心+退火恢复截面排名。90% 步贪心(交换错位最大对), 10% 随机扰动。
fn greedy_anneal(rank_x: &[f32], rank_y: &[f32], steps: usize) -> Option<(f32, f32, u32, u32, u32, Vec<(u32, f32)>)> {
    let n = rank_x.len();
    if n < 10 {
        return None;
    }
    let my = rank_y.iter().map(|&v| v as f64).sum::<f64>() / n as f64;
    let var_y = rank_y
        .iter()
        .map(|&v| {
            let d = v as f64 - my;
            d * d
        })
        .sum::<f64>()
        / n as f64;
    if var_y <= 0.0 {
        return None;
    }
    let mut g = rank_x.to_vec();
    let denom = 2.0 * var_y * n as f64;
    let inv_denom = 1.0 / denom;
    let mut s: f64 = g
        .iter()
        .zip(rank_y.iter())
        .map(|(a, b)| {
            let d = *a as f64 - *b as f64;
            d * d
        })
        .sum();
    let r0 = 1.0 - s * inv_denom;
    let mut rng = XorShift64::new(SEED + 1);
    let (mut st70, mut st90) = (u32::MAX, u32::MAX);
    let mut ladder = 0u32;
    let mut prev_r = r0;
    let mut curve: Vec<(u32, f32)> = Vec::new();
    let stride = (steps / 200).max(1);
    for step in 0..steps {
        if rng.next_index(10) == 0 {
            // 随机扰动
            let i = rng.next_index(n);
            let mut j = rng.next_index(n);
            if j == i {
                j = rng.next_index(n);
            }
            let ds = 2.0f64 * (g[i] as f64 - g[j] as f64)
                * (rank_y[i] as f64 - rank_y[j] as f64);
            if ds < 0.0 {
                s += ds;
                g.swap(i, j);
            }
        } else {
            // 贪心: i = 错位最大
            let mut i = 0usize;
            let mut best = -1.0f64;
            for k in 0..n {
                let d = (g[k] as f64 - rank_y[k] as f64).abs();
                if d > best {
                    best = d;
                    i = k;
                }
            }
            // j = 使 ΔS 最小的位置
            let gi = g[i];
            let ti = rank_y[i];
            let mut best_ds = f64::INFINITY;
            let mut best_j = usize::MAX;
            for k in 0..n {
                if k == i {
                    continue;
                }
                let ds = 2.0 * (gi as f64 - g[k] as f64) * (ti as f64 - rank_y[k] as f64);
                if ds < best_ds {
                    best_ds = ds;
                    best_j = k;
                }
            }
            if best_j != usize::MAX && best_ds < 0.0 {
                s += best_ds;
                g.swap(i, best_j);
            }
        }
        let cr = 1.0 - s * inv_denom;
        if st70 == u32::MAX && cr >= 0.70 {
            st70 = step as u32;
        }
        if st90 == u32::MAX && cr >= 0.90 {
            st90 = step as u32;
        }
        if cr - prev_r >= 0.02 {
            ladder += 1;
        }
        prev_r = cr;
        if step % stride == 0 {
            curve.push((step as u32, cr as f32));
        }
        if cr >= 0.995 {
            break;
        }
    }
    Some((
        r0 as f32,
        prev_r as f32,
        if st70 == u32::MAX { steps as u32 } else { st70 },
        if st90 == u32::MAX { steps as u32 } else { st90 },
        ladder,
        curve,
    ))
}

/// 密集排名: 值 → 1..N 排名(并列用平均排名)
fn dense_rank(v: &[f32]) -> Vec<f32> {
    let mut idx: Vec<usize> = (0..v.len()).collect();
    idx.sort_by(|&a, &b| v[a].partial_cmp(&v[b]).unwrap_or(std::cmp::Ordering::Equal));
    let mut rk = vec![0.0f32; v.len()];
    let n = v.len();
    let mut i = 0;
    while i < n {
        let mut j = i;
        while j < n && v[idx[j]] == v[idx[i]] {
            j += 1;
        }
        let avg = ((i + 1) + (j)) as f32 * 0.5; // 平均排名 1..N
        for k in i..j {
            rk[idx[k]] = avg;
        }
        i = j;
    }
    rk
}

// ============================================================================
// 主流程
// ============================================================================

fn list_codes(date: i64) -> Vec<String> {
    for root in ["/ssd_data/stock", "/nas197/binary/stock/sz_alpha/stock"] {
        let dir = format!("{root}/{date}/transaction");
        if let Ok(entries) = std::fs::read_dir(&dir) {
            let mut set = BTreeSet::new();
            for e in entries.flatten() {
                if let Some(code) = e.file_name().to_str().and_then(|n| n.split('_').next()) {
                    if code.bytes().all(|b| b.is_ascii_digit()) {
                        set.insert(code.to_string());
                    }
                }
            }
            if !set.is_empty() {
                return set.into_iter().collect();
            }
        }
    }
    Vec::new()
}

fn main() {
    let date: i64 = std::env::args()
        .nth(1)
        .expect("用法: switch_anneal_sandbox <date>")
        .parse()
        .unwrap();
    let codes = list_codes(date);
    eprintln!("date={} codes={}", date, codes.len());

    // ① 并行: 读逐笔 → 方向 D + 分钟矩阵
    let per: Vec<Option<(StockFeatD, Vec<f64>, Vec<f64>)>> = codes
        .par_iter()
        .map(|c| read_trade_fast(c, date).ok().and_then(|tr| per_stock_d(&tr)))
        .collect();

    let mut d_codes = Vec::new();
    let mut d_vals: Vec<Vec<f32>> = Vec::new();
    let mut mx: Vec<(Vec<f64>, Vec<f64>)> = Vec::new(); // 全部股票(含无效)的分钟矩阵
    for (c, p) in codes.iter().zip(per.iter()) {
        match p {
            Some((f, buy, amt)) => {
                d_codes.push(c.clone());
                let mut row = Vec::with_capacity(D_NAMES.len());
                row.push(f.n_min as f32);
                row.push(f.r0);
                row.push(f.steps50 as f32);
                row.push(f.steps70 as f32);
                row.push(f.steps90 as f32);
                row.push(f.half_life as f32);
                row.push(f.final_r);
                row.push(f.inertia);
                row.push(f.declines as f32);
                row.push(f.dr_std);
                row.push(f.ladder as f32);
                row.push(f.max_jump);
                row.push(f.mean);
                row.push(f.std);
                row.push(f.ac1);
                row.push(f.trend);
                row.push(f.morn_aft_diff);
                d_vals.push(row);
                mx.push((buy.clone(), amt.clone()));
            }
            None => mx.push((vec![0.0; N_MIN], vec![0.0; N_MIN])),
        }
    }
    eprintln!("direction D valid = {}", d_codes.len());

    // ② 全天相邻分钟截面 r 序列
    let mut r_t: Vec<usize> = Vec::new();
    let mut r_v: Vec<f64> = Vec::new();
    let mut r_n: Vec<usize> = Vec::new();
    for t in 1..N_MIN {
        let mut xs: Vec<f32> = Vec::new();
        let mut ys: Vec<f32> = Vec::new();
        for (buy, amt) in mx.iter() {
            if amt[t - 1] > 0.0 && amt[t] > 0.0 {
                xs.push((buy[t - 1] / amt[t - 1]) as f32);
                ys.push((buy[t] / amt[t]) as f32);
            }
        }
        if xs.len() >= 400 {
            let r = cross_r(&xs, &ys);
            r_t.push(t);
            r_v.push(r);
            r_n.push(xs.len());
        }
    }
    eprintln!("cross-section pairs = {}", r_t.len());

    // ③ 选 3 个低 r 时刻(股票数 >= 1500, 间隔 >= 5 分钟)
    let mut order: Vec<usize> = (0..r_t.len()).collect();
    order.sort_by(|&a, &b| r_v[a].partial_cmp(&r_v[b]).unwrap_or(std::cmp::Ordering::Equal));
    let mut picks: Vec<usize> = Vec::new();
    for &o in &order {
        if r_n[o] < 1500 {
            continue;
        }
        if picks.iter().all(|&p| (r_t[p] as i64 - r_t[o] as i64).abs() >= 5) {
            picks.push(o);
            if picks.len() >= N_CS_MOMENTS {
                break;
            }
        }
    }
    eprintln!("picked low-r moments = {:?}", picks.iter().map(|&i| (r_t[i], r_v[i], r_n[i])).collect::<Vec<_>>());

    // ④ 每个选中时刻: 方向 C(m1 向量退火) + 方向 B(贪心+退火排名恢复)
    let mut cs: Vec<MomentCS> = Vec::new();
    for &o in &picks {
        let t = r_t[o];
        let mut xs: Vec<f32> = Vec::new();
        let mut ys: Vec<f32> = Vec::new();
        let mut ok_codes: Vec<usize> = Vec::new();
        for (k, (buy, amt)) in mx.iter().enumerate() {
            if amt[t - 1] > 0.0 && amt[t] > 0.0 {
                xs.push((buy[t - 1] / amt[t - 1]) as f32);
                ys.push((buy[t] / amt[t]) as f32);
                ok_codes.push(k);
            }
        }
        let r = cross_r(&xs, &ys);
        // 方向 C: m1 向量退火
        let cvec = m1_contrib(&xs, &ys);
        let m_adapt = adaptive_m_max(cvec.len());
        let cf = anneal_seq(&cvec, m_adapt);
        // 采样曲线
        let mut c_curve: Vec<(u32, f32)> = Vec::new();
        if let Some(f) = cf {
            // 重新跑一遍只采集曲线(简单起见用第二遍, 确定性同种子同路径)
            let mut rng = XorShift64::new(SEED);
            let mut guess: Vec<f32> = cvec.clone();
            guess.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let n = cvec.len() as f64;
            let mean_c = cvec.iter().map(|&v| v as f64).sum::<f64>() / n;
            let sigma2 = cvec.iter().map(|&v| {
                let d = v as f64 - mean_c;
                d * d
            }).sum::<f64>() / n;
            let denom = 2.0 * sigma2 * n;
            let mut s: f64 = guess.iter().zip(cvec.iter()).map(|(g, t2)| {
                let d = *g as f64 - *t2 as f64;
                d * d
            }).sum();
            let ct_base = sigma2;
            let inv_m = 1.0 / (m_adapt as f64 - 1.0);
            let stride = (m_adapt / 200).max(1);
            let mut t2 = 0usize;
            while t2 < m_adapt {
                let i = rng.next_index(cvec.len());
                let mut j = rng.next_index(cvec.len());
                if j == i {
                    j = rng.next_index(cvec.len());
                }
                let ds = 2.0f64 * (guess[i] as f64 - guess[j] as f64)
                    * (cvec[i] as f64 - cvec[j] as f64);
                let ct = ct_base * (1.0 - t2 as f64 * inv_m).max(0.0);
                if ds < 0.0 || ds < ct {
                    s += ds;
                    guess.swap(i, j);
                }
                if t2 % stride == 0 {
                    c_curve.push((t2 as u32, (1.0 - s / denom) as f32));
                }
                t2 += 1;
                if s <= (sigma2 * n * 1e-10).max(1e-12) {
                    break;
                }
            }
        }
        // c 的统计
        let c_std = if cvec.len() > 1 {
            let mc = cvec.iter().map(|&v| v as f64).sum::<f64>() / cvec.len() as f64;
            ((cvec.iter().map(|&v| {
                let d = v as f64 - mc;
                d * d
            }).sum::<f64>()
                / (cvec.len() as f64 - 1.0))
                .sqrt()) as f32
        } else {
            f32::NAN
        };
        let c_kurt = if cvec.len() > 3 {
            let mc = cvec.iter().map(|&v| v as f64).sum::<f64>() / cvec.len() as f64;
            let m2 = cvec.iter().map(|&v| {
                let d = v as f64 - mc;
                d * d
            }).sum::<f64>() / cvec.len() as f64;
            let m4 = cvec.iter().map(|&v| {
                let d = (v as f64 - mc).powi(2);
                d * d
            }).sum::<f64>() / cvec.len() as f64;
            if m2 > 1e-20 { m4 / (m2 * m2) - 3.0 } else { f64::NAN }
        } else {
            f64::NAN
        };
        // 方向 B: 排名贪心+退火
        let rx = dense_rank(&xs);
        let ry = dense_rank(&ys);
        let b = greedy_anneal(&rx, &ry, 30000);
        let (b_r0, b_final, b_st70, b_st90, b_ladder, b_curve) = match b {
            Some(v) => v,
            None => {
                (f32::NAN, f32::NAN, 30000, 30000, 0, Vec::new())
            }
        };
        cs.push(MomentCS {
            t,
            r,
            n: xs.len(),
            c_r0: cf.map(|f| f.r0).unwrap_or(f32::NAN),
            c_steps50: cf.map(|f| f.steps50).unwrap_or(u32::MAX),
            c_steps70: cf.map(|f| f.steps70).unwrap_or(u32::MAX),
            c_steps90: cf.map(|f| f.steps90).unwrap_or(u32::MAX),
            c_half_life: cf.map(|f| f.half_life).unwrap_or(u32::MAX),
            c_final_r: cf.map(|f| f.final_r).unwrap_or(f32::NAN),
            c_inertia: cf.map(|f| f.inertia).unwrap_or(f32::NAN),
            c_declines: cf.map(|f| f.declines).unwrap_or(0),
            c_ladder: cf.map(|f| f.ladder).unwrap_or(0),
            c_gini: gini_of(&cvec),
            c_top5pct: top5_share(&cvec),
            c_std,
            c_kurt,
            c_curve,
            c_vec: cvec,
            b_r0,
            b_final_r: b_final,
            b_steps70: b_st70,
            b_steps90: b_st90,
            b_ladder,
            b_curve,
        });
    }

    // ⑤ 输出 JSON
    #[derive(Serialize)]
    struct Out {
        date: i64,
        n_codes: usize,
        d_names: Vec<String>,
        d_codes: Vec<String>,
        d_vals: Vec<Vec<f32>>,
        r_seq: Vec<(usize, f64, usize)>,
        cs: Vec<MomentCS>,
    }
    let out = Out {
        date,
        n_codes: codes.len(),
        d_names: D_NAMES.iter().map(|s| s.to_string()).collect(),
        d_codes,
        d_vals,
        r_seq: r_t.into_iter().zip(r_v.into_iter().zip(r_n)).map(|(a, (b, c))| (a, b, c)).collect(),
        cs,
    };
    println!("{}", serde_json::to_string(&out).unwrap());
    eprintln!("done");
}
