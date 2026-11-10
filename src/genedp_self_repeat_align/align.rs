//! 第 2 层：带宽限制的自身局部比对（Smith-Waterman 动态规划）。
//!
//! 只比对“某位置 i”和“它后面 lag 位的位置 j=i+lag”，lag 从 1 到 max_lag，
//! 因此主对角线（自己和自己，lag=0）的平凡匹配天然被排除；lag 变化对应插入/缺失。
//! 计分用“稀有度”：字母越少见，配上加分越多（四分之一比特）；失配罚分随字母分布自适应，
//! 保证完全随机序列的期望得分为负（否则局部比对会无限延长）。
//! 显著门槛 T = 4*log2(n*L) + 16：随机序列出现这么高分的期望次数约 1/16 以下。
//!
//! 三种实现：
//! - `self_align_ref`：参考实现（逐格计算），第三版冻结基准用它；
//! - `self_align_fast`：提速版，按 64 个错位一组先找出“可能非零”的格子，只算这些格子。
//!   被跳过的格子在参考实现里必然得 0 且不触发任何上报，所以结果逐位相同（有自检）；
//! - `self_align_affine`：开缺口与延长缺口分开计分（Gotoh 三状态），参考和提速模式都用它。

/// 诊断计数（不影响结果）：上报片段条数、排序去重耗时（纳秒）。
pub static PROF_EMIT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static PROF_FINISH_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Seg {
    pub score: i32,
    pub si: u32, // 第一份拷贝起点
    pub sl: u32, // 起点处的 lag
    pub ei: u32, // 第一份拷贝终点（最高分处）
    pub el: u32, // 终点处的 lag
    pub matches: u32,
    pub mism: u32,
    pub gaps: u32,
}

#[derive(Clone, Copy, Default)]
struct Meta {
    best: i32,
    si: u32,
    sl: u32,
    bi: u32,
    bl: u32,
    ma: u32,
    mi: u32,
    ga: u32,
    mab: u32,
    mib: u32,
    gab: u32,
}

pub struct AlignOut {
    pub segs: Vec<Seg>,
    pub threshold: i32,
    pub pp: f64,            // 字母分布的 Σp²（随机两字母相同的概率）
    pub lag_hits: Vec<u64>, // 每个 lag 上字母相同的位置数（供朴素对照用）
    pub max_lag: usize,
    pub n_emitted: u64, // 诊断：去重前上报条数
}

/// 默认目标相同率 80%（第一、二版只用这个）。
pub fn scoring(a: &[u32], k: usize) -> Option<(Vec<i32>, i32, f64)> {
    scoring_q(a, k, 0.8)
}

/// 按字母频率给出配对分与失配罚分（四分之一比特）。
/// 目标模型：真实重复里 q 比例的位置相同。配对分 = 4*log2(q/p)，失配分 = 4*log2((1-q)/(1-Σp²))。
/// 这样随机序列的得分分布有确定的尺度：得分 S 的片段，随机出现的期望次数约为 n*L*2^(-S/4)。
pub fn scoring_q(a: &[u32], k: usize, q: f64) -> Option<(Vec<i32>, i32, f64)> {
    let n = a.len() as f64;
    let mut cnt = vec![0u32; k];
    for &x in a {
        cnt[x as usize] += 1;
    }
    let mut ms = vec![0i32; k];
    let (mut s_sum, mut pp) = (0.0f64, 0.0f64);
    for (c, m) in cnt.iter().zip(ms.iter_mut()) {
        if *c == 0 {
            continue;
        }
        let p = *c as f64 / n;
        *m = ((4.0 * (q / p).log2()).round() as i32).max(1);
        pp += p * p;
        s_sum += p * p * (*m as f64);
    }
    if pp >= 0.9999 {
        return None; // 全是同一个字母，无法比对
    }
    let mut mism = ((-4.0 * ((1.0 - q) / (1.0 - pp)).log2()).round() as i32).max(1);
    // 字母极度集中时配对分被截到 1，这里再抬高罚分，保证随机序列期望得分为负
    while s_sum - (1.0 - pp) * mism as f64 >= -1e-9 {
        mism += 1;
    }
    Some((ms, mism.min(1000), pp))
}

/// 插入/缺失罚分：比失配贵一半，gap_mult 再放大（大值 = 不允许插入缺失）。
#[inline]
pub fn gap_cost(mism: i32, gap_mult: i32) -> i32 {
    ((mism + mism / 2 + 1).saturating_mul(gap_mult)).min(1_000_000)
}

/// 片段上报器。提速版打开“同起点去重缓存”：同一起点 (si,sl) 已上报过排序不差的片段时，
/// 新片段直接丢弃。最终每个起点只保留排序最前的一条；排序键 (起点, 分数, 最高点) 相同的片段
/// 来自同一个格子的记录，其余字段也必然相同，所以丢弃不改变输出。参考版关闭缓存。
struct Emitter {
    out: Vec<Seg>,
    cache: Vec<(u32, u32, i32, u32, u32)>,
    on: bool,
    pushed: u64,
}

impl Emitter {
    fn new(on: bool) -> Self {
        Emitter { out: Vec::new(), cache: if on { vec![(u32::MAX, u32::MAX, 0, 0, 0); 4096] } else { Vec::new() }, on, pushed: 0 }
    }
}

#[inline]
fn emit(em: &mut Emitter, m: &Meta) {
    if em.on {
        let h = ((m.si as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ (m.sl as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F)) >> 52;
        let slot = &mut em.cache[h as usize];
        if slot.0 == m.si && slot.1 == m.sl {
            // 已有片段的排序键 (-分数, 终点, 终点错位) 不大于新片段：新片段不可能成为该起点的保留项
            if (-slot.2, slot.3, slot.4) <= (-m.best, m.bi, m.bl) {
                return;
            }
        }
        *slot = (m.si, m.sl, m.best, m.bi, m.bl);
    }
    em.pushed += 1;
    em.out.push(Seg {
        score: m.best,
        si: m.si,
        sl: m.sl,
        ei: m.bi,
        el: m.bl,
        matches: m.mab,
        mism: m.mib,
        gaps: m.gab,
    });
}

fn finish(em: Emitter, seg_cap: usize, threshold: i32, pp: f64, lag_hits: Vec<u64>, l: usize) -> AlignOut {
    let t = std::time::Instant::now();
    PROF_EMIT.fetch_add(em.pushed, std::sync::atomic::Ordering::Relaxed);
    let n_emitted = em.pushed;
    let mut out = em.out;
    // 同一条链可能分叉、重复上报：按起点去重，保留最高分
    out.sort_unstable_by(|x, y| (x.si, x.sl, -x.score, x.ei, x.el).cmp(&(y.si, y.sl, -y.score, y.ei, y.el)));
    out.dedup_by(|b, a| a.si == b.si && a.sl == b.sl);
    out.sort_unstable_by(|x, y| (-x.score, x.si, x.sl, x.ei, x.el).cmp(&(-y.score, y.si, y.sl, y.ei, y.el)));
    out.truncate(seg_cap);
    PROF_FINISH_NS.fetch_add(t.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
    AlignOut { segs: out, threshold, pp, lag_hits, max_lag: l, n_emitted }
}

struct Ctx {
    ms: Vec<i32>,
    mism: i32,
    g: i32,
    threshold: i32,
    x_drop: i32,
}

/// 单个格子的线性缺口递推（参考与提速共用，保证两者算式完全相同）。返回本格得分。
/// PRUNE=true（提速版）时，被放弃的对角线链若被同起点、分数更高（或同分同最高点）的
/// 链在本格接续，就不上报：那条接续链结束时一定会上报一个不差于它的片段，而最终每个
/// 起点只保留排序最前的一条，所以输出不变。参考版 PRUNE=false。
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn cell_linear<const PRUNE: bool>(
    i: usize,
    lag: usize,
    eq: bool,
    ai: u32,
    dh: i32,
    uh: i32,
    lh: i32,
    c: &Ctx,
    prev_m: &[Meta],
    cur_m: &mut [Meta],
    out: &mut Emitter,
) -> i32 {
    let s = if eq { c.ms[ai as usize] } else { -c.mism };
    let vd = if dh > 0 || eq { dh + s } else { 0 };
    let vu = if uh > 0 { uh - c.g } else { 0 };
    let vl = if lh > 0 { lh - c.g } else { 0 };
    let h = vd.max(vu).max(vl).max(0);
    let from_d = h > 0 && h == vd;
    let from_u = h > 0 && !from_d && h == vu;
    // 对角线前驱的链没有被延续：它在这里结束
    if dh > 0 && !from_d && prev_m[lag].best >= c.threshold {
        let pd = &prev_m[lag];
        let dominated = PRUNE && h > 0 && {
            let m0 = if from_u { &prev_m[lag + 1] } else { &cur_m[lag - 1] };
            m0.si == pd.si && m0.sl == pd.sl && (m0.best > pd.best || (m0.best == pd.best && m0.bi == pd.bi && m0.bl == pd.bl))
        };
        if !dominated {
            emit(out, pd);
        }
    }
    if h == 0 {
        return 0;
    }
    let mut m = if from_d {
        if dh > 0 {
            prev_m[lag]
        } else {
            Meta { si: i as u32, sl: lag as u32, ..Meta::default() }
        }
    } else if from_u {
        prev_m[lag + 1]
    } else {
        cur_m[lag - 1]
    };
    if from_d {
        if eq {
            m.ma += 1;
        } else {
            m.mi += 1;
        }
    } else {
        m.ga += 1;
    }
    if h > m.best {
        m.best = h;
        m.bi = i as u32;
        m.bl = lag as u32;
        m.mab = m.ma;
        m.mib = m.mi;
        m.gab = m.ga;
    }
    if m.best - h > c.x_drop {
        if m.best >= c.threshold {
            emit(out, &m);
        }
        return 0;
    }
    cur_m[lag] = m;
    h
}

/// 带宽限制的自身局部比对（默认参数），第一、二版入口。
pub fn self_align(a: &[u32], k: usize, max_lag: usize, gap_mult: i32, seg_cap: usize) -> Option<AlignOut> {
    self_align_ref(a, k, max_lag, 0.8, gap_mult, seg_cap, true)
}

/// use_xdrop=false 只用于和完整表格对照：关掉“分数比最高点跌太多就截断”。
pub fn self_align_x(a: &[u32], k: usize, max_lag: usize, gap_mult: i32, seg_cap: usize, use_xdrop: bool) -> Option<AlignOut> {
    self_align_ref(a, k, max_lag, 0.8, gap_mult, seg_cap, use_xdrop)
}

fn setup(a: &[u32], k: usize, max_lag: usize, q: f64, gap_mult: i32, use_xdrop: bool) -> Option<(Ctx, f64, usize)> {
    let n = a.len();
    if n < 8 || max_lag == 0 {
        return None;
    }
    let (ms, mism, pp) = scoring_q(a, k, q)?;
    let g = gap_cost(mism, gap_mult);
    let l = max_lag.min(n - 1);
    let threshold = (4.0 * ((n * l) as f64).log2()).ceil() as i32 + 16;
    let x_drop = if use_xdrop { threshold } else { i32::MAX };
    Some((Ctx { ms, mism, g, threshold, x_drop }, pp, l))
}

/// 参考实现：逐格计算。
pub fn self_align_ref(a: &[u32], k: usize, max_lag: usize, q: f64, gap_mult: i32, seg_cap: usize, use_xdrop: bool) -> Option<AlignOut> {
    let n = a.len();
    let (c, pp, l) = setup(a, k, max_lag, q, gap_mult, use_xdrop)?;
    let threshold = c.threshold;
    let mut prev_h = vec![0i32; l + 2];
    let mut cur_h = vec![0i32; l + 2];
    let mut prev_m = vec![Meta::default(); l + 2];
    let mut cur_m = vec![Meta::default(); l + 2];
    let mut lag_hits = vec![0u64; l + 2];
    let mut out = Emitter::new(false);

    for i in 0..n {
        let lim = l.min(n - 1 - i);
        // 错位超出序列末端的带子上，残留的片段此时结束
        for lag in (lim + 1)..=l {
            if prev_h[lag] > 0 {
                if prev_m[lag].best >= threshold {
                    emit(&mut out, &prev_m[lag]);
                }
                prev_h[lag] = 0;
            }
        }
        cur_h[0] = 0;
        let ai = a[i];
        for lag in 1..=lim {
            let eq = ai == a[i + lag];
            if eq {
                lag_hits[lag] += 1;
            }
            let dh = prev_h[lag];
            let uh = prev_h[lag + 1];
            let lh = cur_h[lag - 1];
            if !eq && dh == 0 && uh == 0 && lh == 0 {
                cur_h[lag] = 0;
                continue;
            }
            cur_h[lag] = cell_linear::<false>(i, lag, eq, ai, dh, uh, lh, &c, &prev_m, &mut cur_m, &mut out);
        }
        // 本行没有算到的带子要清零，下一行才不会读到陈旧值
        for v in cur_h[(lim + 1)..=(l + 1)].iter_mut() {
            *v = 0;
        }
        std::mem::swap(&mut prev_h, &mut cur_h);
        std::mem::swap(&mut prev_m, &mut cur_m);
    }
    for lag in 1..=l {
        if prev_h[lag] > 0 && prev_m[lag].best >= threshold {
            emit(&mut out, &prev_m[lag]);
        }
    }
    Some(finish(out, seg_cap, threshold, pp, lag_hits, l))
}

/// 提速版：每行按 64 个错位一组，先用“字母相同 / 上一行对应格非零”两张位图找出候选格，
/// 再沿“左邻非零”向右补候选。只有候选格可能非零或触发上报，其余格子在参考实现里
/// 都走“全零直接跳过”分支，所以逐位等价。
pub fn self_align_fast(a: &[u32], k: usize, max_lag: usize, q: f64, gap_mult: i32, seg_cap: usize) -> Option<AlignOut> {
    let n = a.len();
    let (c, pp, l) = setup(a, k, max_lag, q, gap_mult, true)?;
    let threshold = c.threshold;
    let mut prev_h = vec![0i32; l + 2];
    let mut cur_h = vec![0i32; l + 2];
    let mut prev_m = vec![Meta::default(); l + 2];
    let mut cur_m = vec![Meta::default(); l + 2];
    let mut lag_hits = vec![0u64; l + 2];
    let mut out = Emitter::new(true);

    for i in 0..n {
        let lim = l.min(n - 1 - i);
        for lag in (lim + 1)..=l {
            if prev_h[lag] > 0 {
                if prev_m[lag].best >= threshold {
                    emit(&mut out, &prev_m[lag]);
                }
                prev_h[lag] = 0;
            }
        }
        cur_h[0] = 0;
        let ai = a[i];
        let row = &a[i..];
        let mut lag0 = 1usize;
        let mut carry = false;
        while lag0 <= lim {
            let len = 64.min(lim - lag0 + 1);
            let mut cand: u64 = 0;
            {
                let ar = &row[lag0..lag0 + len];
                let pd = &prev_h[lag0..lag0 + len];
                let pu = &prev_h[lag0 + 1..lag0 + len + 1];
                for k2 in 0..len {
                    let bit = (ar[k2] == ai) | (pd[k2] > 0) | (pu[k2] > 0);
                    cand |= (bit as u64) << k2;
                }
            }
            for v in cur_h[lag0..lag0 + len].iter_mut() {
                *v = 0;
            }
            if carry {
                cand |= 1;
            }
            while cand != 0 {
                let b = cand.trailing_zeros() as usize;
                cand &= cand - 1;
                let lag = lag0 + b;
                let eq = ai == row[lag];
                if eq {
                    lag_hits[lag] += 1;
                }
                let dh = prev_h[lag];
                let uh = prev_h[lag + 1];
                let lh = cur_h[lag - 1];
                if !eq && dh == 0 && uh == 0 && lh == 0 {
                    continue;
                }
                let h = cell_linear::<true>(i, lag, eq, ai, dh, uh, lh, &c, &prev_m, &mut cur_m, &mut out);
                cur_h[lag] = h;
                if h > 0 && b + 1 < len {
                    cand |= 1u64 << (b + 1);
                }
            }
            carry = cur_h[lag0 + len - 1] > 0;
            lag0 += len;
        }
        for v in cur_h[(lim + 1)..=(l + 1)].iter_mut() {
            *v = 0;
        }
        std::mem::swap(&mut prev_h, &mut cur_h);
        std::mem::swap(&mut prev_m, &mut cur_m);
    }
    for lag in 1..=l {
        if prev_h[lag] > 0 && prev_m[lag].best >= threshold {
            emit(&mut out, &prev_m[lag]);
        }
    }
    Some(finish(out, seg_cap, threshold, pp, lag_hits, l))
}

/// 开缺口与延长缺口分开计分（Gotoh 三状态）。开缺口 = 线性版缺口罚分，延长 = 失配罚分的一半（至少 1）。
/// E：竖向缺口链（来自上一行 lag+1），F：横向缺口链（来自本行 lag-1）。
pub fn self_align_affine(a: &[u32], k: usize, max_lag: usize, q: f64, seg_cap: usize, use_xdrop: bool, dedup_cache: bool) -> Option<AlignOut> {
    let n = a.len();
    let (c, pp, l) = setup(a, k, max_lag, q, 1, use_xdrop)?;
    let threshold = c.threshold;
    let go = c.g;
    let ge = (c.mism / 2).max(1);
    let (mut ph, mut ch) = (vec![0i32; l + 2], vec![0i32; l + 2]);
    let (mut pe, mut ce) = (vec![0i32; l + 2], vec![0i32; l + 2]);
    let mut cf = vec![0i32; l + 2];
    let (mut pm, mut cm) = (vec![Meta::default(); l + 2], vec![Meta::default(); l + 2]);
    let (mut pem, mut cem) = (vec![Meta::default(); l + 2], vec![Meta::default(); l + 2]);
    let mut cfm = vec![Meta::default(); l + 2];
    let mut lag_hits = vec![0u64; l + 2];
    let mut out = Emitter::new(dedup_cache);
    for i in 0..n {
        let lim = l.min(n - 1 - i);
        for lag in (lim + 1)..=l {
            if ph[lag] > 0 {
                if pm[lag].best >= threshold {
                    emit(&mut out, &pm[lag]);
                }
                ph[lag] = 0;
            }
            pe[lag] = 0;
        }
        ch[0] = 0;
        cf[0] = 0;
        let ai = a[i];
        for lag in 1..=lim {
            let eq = ai == a[i + lag];
            if eq {
                lag_hits[lag] += 1;
            }
            let dh = ph[lag];
            // 竖向缺口：从上一行 lag+1 的 H 开口，或延长上一行 lag+1 的 E
            let e_open = if ph[lag + 1] > 0 { ph[lag + 1] - go } else { 0 };
            let e_ext = if pe[lag + 1] > 0 { pe[lag + 1] - ge } else { 0 };
            let e = e_open.max(e_ext).max(0);
            // 横向缺口：从本行 lag-1 的 H 开口，或延长本行 lag-1 的 F
            let f_open = if ch[lag - 1] > 0 { ch[lag - 1] - go } else { 0 };
            let f_ext = if cf[lag - 1] > 0 { cf[lag - 1] - ge } else { 0 };
            let f = f_open.max(f_ext).max(0);
            if e > 0 {
                let mut m = if e == e_open { pm[lag + 1] } else { pem[lag + 1] };
                m.ga += 1;
                cem[lag] = m;
            }
            if f > 0 {
                let mut m = if f == f_open { cm[lag - 1] } else { cfm[lag - 1] };
                m.ga += 1;
                cfm[lag] = m;
            }
            ce[lag] = e;
            cf[lag] = f;
            if !eq && dh == 0 && e == 0 && f == 0 {
                ch[lag] = 0;
                continue;
            }
            let s = if eq { c.ms[ai as usize] } else { -c.mism };
            let vd = if dh > 0 || eq { dh + s } else { 0 };
            let h = vd.max(e).max(f).max(0);
            let from_d = h > 0 && h == vd;
            let from_e = h > 0 && !from_d && h == e;
            if dh > 0 && !from_d && pm[lag].best >= threshold {
                emit(&mut out, &pm[lag]);
            }
            if h == 0 {
                ch[lag] = 0;
                continue;
            }
            let mut m = if from_d {
                let mut m = if dh > 0 { pm[lag] } else { Meta { si: i as u32, sl: lag as u32, ..Meta::default() } };
                if eq {
                    m.ma += 1;
                } else {
                    m.mi += 1;
                }
                m
            } else if from_e {
                cem[lag]
            } else {
                cfm[lag]
            };
            if h > m.best {
                m.best = h;
                m.bi = i as u32;
                m.bl = lag as u32;
                m.mab = m.ma;
                m.mib = m.mi;
                m.gab = m.ga;
            }
            if m.best - h > c.x_drop {
                if m.best >= threshold {
                    emit(&mut out, &m);
                }
                ch[lag] = 0;
                continue;
            }
            ch[lag] = h;
            cm[lag] = m;
        }
        for lag in (lim + 1)..=(l + 1) {
            ch[lag] = 0;
            ce[lag] = 0;
            cf[lag] = 0;
        }
        std::mem::swap(&mut ph, &mut ch);
        std::mem::swap(&mut pe, &mut ce);
        std::mem::swap(&mut pm, &mut cm);
        std::mem::swap(&mut pem, &mut cem);
    }
    for lag in 1..=l {
        if ph[lag] > 0 && pm[lag].best >= threshold {
            emit(&mut out, &pm[lag]);
        }
    }
    Some(finish(out, seg_cap, threshold, pp, lag_hits, l))
}

/// 贪心挑选互不重复的片段：按分数从高到低，新片段至少一半的位置未被覆盖才保留。
/// 返回入选片段和每个位置是否被覆盖。
pub fn select_nonoverlap(segs: &[Seg], n: usize) -> (Vec<Seg>, Vec<bool>) {
    let mut cov = vec![false; n];
    let mut picked = Vec::new();
    for s in segs {
        let (a0, a1) = (s.si as usize, s.ei as usize);
        let (b0, b1) = ((s.si + s.sl) as usize, (s.ei + s.el) as usize);
        let total = (a1 - a0 + 1) + (b1 - b0 + 1);
        let mut fresh = 0usize;
        for p in (a0..=a1).chain(b0..=b1.min(n - 1)) {
            if !cov[p] {
                fresh += 1;
            }
        }
        if fresh * 2 < total {
            continue;
        }
        for p in (a0..=a1).chain(b0..=b1.min(n - 1)) {
            cov[p] = true;
        }
        picked.push(*s);
    }
    (picked, cov)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 独立对照：完整二维表格的同一递推，返回全表最高分（不走滚动数组和提前跳过）。
    fn brute_best(a: &[u32], k: usize, max_lag: usize, gap_mult: i32) -> i32 {
        let n = a.len();
        let (ms, mism, _) = scoring(a, k).unwrap();
        let g = gap_cost(mism, gap_mult);
        let mut h = vec![vec![0i32; n + 1]; n + 1];
        let mut best = 0;
        for i in 1..=n {
            for j in (i + 1)..=n {
                if j - i > max_lag {
                    continue;
                }
                let s = if a[i - 1] == a[j - 1] { ms[a[i - 1] as usize] } else { -mism };
                let d = if h[i - 1][j - 1] > 0 || s > 0 { h[i - 1][j - 1] + s } else { 0 };
                let u = if i >= 1 && j - (i - 1) <= max_lag && h[i - 1][j] > 0 { h[i - 1][j] - g } else { 0 };
                let l = if j - 1 > i && h[i][j - 1] > 0 { h[i][j - 1] - g } else { 0 };
                h[i][j] = d.max(u).max(l).max(0);
                best = best.max(h[i][j]);
            }
        }
        best
    }

    /// 独立对照：完整表格的 Gotoh 三状态，返回全表最高分。
    fn brute_affine(a: &[u32], k: usize, max_lag: usize, q: f64) -> i32 {
        let n = a.len();
        let (ms, mism, _) = scoring_q(a, k, q).unwrap();
        let go = gap_cost(mism, 1);
        let ge = (mism / 2).max(1);
        let (mut h, mut e, mut f) = (vec![vec![0i32; n + 1]; n + 1], vec![vec![0i32; n + 1]; n + 1], vec![vec![0i32; n + 1]; n + 1]);
        let mut best = 0;
        for i in 1..=n {
            for j in (i + 1)..=n {
                if j - i > max_lag {
                    continue;
                }
                let ok_up = j - (i - 1) <= max_lag;
                let eo = if ok_up && h[i - 1][j] > 0 { h[i - 1][j] - go } else { 0 };
                let ee = if ok_up && e[i - 1][j] > 0 { e[i - 1][j] - ge } else { 0 };
                e[i][j] = eo.max(ee).max(0);
                let fo = if j - 1 > i && h[i][j - 1] > 0 { h[i][j - 1] - go } else { 0 };
                let fe = if j - 1 > i && f[i][j - 1] > 0 { f[i][j - 1] - ge } else { 0 };
                f[i][j] = fo.max(fe).max(0);
                let s = if a[i - 1] == a[j - 1] { ms[a[i - 1] as usize] } else { -mism };
                let d = if h[i - 1][j - 1] > 0 || s > 0 { h[i - 1][j - 1] + s } else { 0 };
                h[i][j] = d.max(e[i][j]).max(f[i][j]).max(0);
                best = best.max(h[i][j]);
            }
        }
        best
    }

    fn rand_seq(n: usize, k: u32, mut x: u64) -> Vec<u32> {
        (0..n)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                (x % k as u64) as u32
            })
            .collect()
    }

    /// 全表最高分要等于滚动实现上报的最高分（只要最高分过了门槛）。
    #[test]
    fn matches_bruteforce() {
        let mut checked = 0;
        for seed in 1..400u64 {
            let mut a = rand_seq(300, 4, seed * 7919 + 1);
            // 植入一段重复，让最高分过门槛
            let unit: Vec<u32> = a[20..60].to_vec();
            let pos = 80 + (seed as usize % 100);
            a[pos..pos + 40].copy_from_slice(&unit);
            for &(lag, gm) in &[(64usize, 1i32), (128, 1), (64, 1000)] {
                let b = brute_best(&a, 4, lag, gm);
                let o = self_align_x(&a, 4, lag, gm, 100000, false).unwrap();
                if b >= o.threshold {
                    let got = o.segs.iter().map(|s| s.score).max().unwrap_or(0);
                    assert_eq!(got, b, "seed={seed} lag={lag} gm={gm}");
                    checked += 1;
                }
            }
        }
        assert!(checked > 100, "植入样本过门槛的太少: {checked}");
    }

    #[test]
    fn affine_matches_bruteforce() {
        let mut checked = 0;
        for seed in 1..300u64 {
            let mut a = rand_seq(260, 5, seed * 104729 + 3);
            let unit: Vec<u32> = a[10..55].to_vec();
            let pos = 70 + (seed as usize % 90);
            a[pos..pos + 45].copy_from_slice(&unit);
            // 在拷贝里删掉 / 插入一个字母，制造缺口
            if seed % 2 == 0 {
                a.remove(pos + 20);
            } else {
                a.insert(pos + 20, 0);
            }
            for &lag in &[48usize, 100] {
                let b = brute_affine(&a, 5, lag, 0.8);
                let o = self_align_affine(&a, 5, lag, 0.8, 100000, false, false).unwrap();
                let oc = self_align_affine(&a, 5, lag, 0.8, 100000, true, true).unwrap();
                let or = self_align_affine(&a, 5, lag, 0.8, 100000, true, false).unwrap();
                assert_eq!(oc.segs, or.segs, "去重缓存改变了结果 seed={seed}");
                if b >= o.threshold {
                    let got = o.segs.iter().map(|s| s.score).max().unwrap_or(0);
                    assert_eq!(got, b, "seed={seed} lag={lag}");
                    checked += 1;
                }
            }
        }
        assert!(checked > 100, "植入样本过门槛的太少: {checked}");
    }

    /// 提速版与参考版：片段列表（含顺序、所有字段）与每个错位的相同计数逐项相等。
    #[test]
    fn fast_equals_reference() {
        let mut cases = 0;
        for seed in 1..250u64 {
            let k = [2u32, 4, 20, 36, 300][seed as usize % 5];
            let n = 200 + (seed as usize * 37) % 1500;
            let mut a = rand_seq(n, k, seed * 31 + 7);
            // 低复杂度段、串联重复、长同字母段，覆盖各种分支
            for p in (n / 5)..(n / 5 + n / 10) {
                a[p] = 0;
            }
            let unit = rand_seq(1 + seed as usize % 9, k, seed);
            for p in (n / 2)..(n / 2 + n / 6).min(n) {
                a[p] = unit[p % unit.len()];
            }
            for &(lag, q, gm) in &[(1usize, 0.8, 1i32), (63, 0.8, 1), (64, 0.65, 1), (65, 0.9, 1), (128, 0.8, 1000), (512, 0.8, 1), (n / 2, 0.8, 1)] {
                let r = self_align_ref(&a, k as usize, lag, q, gm, 20000, true);
                let f = self_align_fast(&a, k as usize, lag, q, gm, 20000);
                match (r, f) {
                    (None, None) => {}
                    (Some(r), Some(f)) => {
                        assert_eq!(r.segs, f.segs, "seed={seed} lag={lag}");
                        assert_eq!(r.lag_hits, f.lag_hits, "seed={seed} lag={lag}");
                        assert_eq!(r.threshold, f.threshold);
                        cases += 1;
                    }
                    _ => panic!("一边有结果一边没有 seed={seed}"),
                }
            }
        }
        assert!(cases > 1500);
    }

    #[test]
    fn finds_tandem_repeat_period() {
        let mut a = rand_seq(1000, 20, 99);
        let unit: Vec<u32> = rand_seq(7, 20, 12345);
        for c in 0..30 {
            for (o, &u) in unit.iter().enumerate() {
                a[300 + c * 7 + o] = u;
            }
        }
        let o = self_align(&a, 20, 64, 1, 100000).unwrap();
        let (p, cov) = select_nonoverlap(&o.segs, a.len());
        assert!(!p.is_empty());
        let top = p[0];
        assert_eq!(top.sl % 7, 0, "周期应为 7 的倍数, 实际 {}", top.sl);
        let inside = (300..510).filter(|&i| cov[i]).count();
        assert!(inside >= 190, "重复区覆盖不足: {inside}/210");
        let outside = (0..250).filter(|&i| cov[i]).count();
        assert!(outside <= 12, "随机区被误判过多: {outside}");
    }

    #[test]
    fn random_sequence_no_hits() {
        let a = rand_seq(5000, 20, 2024);
        let o = self_align(&a, 20, 128, 1, 100000).unwrap();
        assert!(o.segs.len() <= 1, "随机序列显著片段过多: {}", o.segs.len());
    }
}
