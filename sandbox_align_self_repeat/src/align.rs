//! 第 2 层：带宽限制的自身局部比对（Smith-Waterman 动态规划）。
//!
//! 只比对“某位置 i”和“它后面 lag 位的位置 j=i+lag”，lag 从 1 到 max_lag，
//! 因此主对角线（自己和自己，lag=0）的平凡匹配天然被排除；lag 变化对应插入/缺失。
//! 计分用“稀有度”：字母越少见，配上加分越多（四分之一比特）；失配罚分随字母分布自适应，
//! 保证完全随机序列的期望得分为负（否则局部比对会无限延长）。
//! 显著门槛 T = 4*log2(n*L) + 16：随机序列出现这么高分的期望次数约 1/16 以下。

#[derive(Clone, Copy, Debug)]
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
    pub pp: f64,          // 字母分布的 Σp²（随机两字母相同的概率）
    pub lag_hits: Vec<u64>, // 每个 lag 上字母相同的位置数（供朴素对照用）
    pub max_lag: usize,
}

/// 按字母频率给出配对分与失配罚分（四分之一比特）。
/// 目标模型：真实重复里 80% 的位置相同。配对分 = 4*log2(q/p)，失配分 = 4*log2((1-q)/(1-Σp²))。
/// 这样随机序列的得分分布有确定的尺度：得分 S 的片段，随机出现的期望次数约为 n*L*2^(-S/4)。
pub fn scoring(a: &[u32], k: usize) -> Option<(Vec<i32>, i32, f64)> {
    const Q: f64 = 0.8;
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
        *m = ((4.0 * (Q / p).log2()).round() as i32).max(1);
        pp += p * p;
        s_sum += p * p * (*m as f64);
    }
    if pp >= 0.9999 {
        return None; // 全是同一个字母，无法比对
    }
    let mut mism = ((-4.0 * ((1.0 - Q) / (1.0 - pp)).log2()).round() as i32).max(1);
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

#[inline]
fn emit(out: &mut Vec<Seg>, m: &Meta) {
    out.push(Seg {
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

/// 带宽限制的自身局部比对，返回去重前的所有显著片段（已排序、去重后）。
/// max_lag 为最大错位；gap_mult 放大插入/缺失罚分（大值 = 不允许插入缺失）。
pub fn self_align(a: &[u32], k: usize, max_lag: usize, gap_mult: i32, seg_cap: usize) -> Option<AlignOut> {
    self_align_x(a, k, max_lag, gap_mult, seg_cap, true)
}

/// use_xdrop=false 只用于和完整表格对照：关掉“分数比最高点跌太多就截断”。
pub fn self_align_x(a: &[u32], k: usize, max_lag: usize, gap_mult: i32, seg_cap: usize, use_xdrop: bool) -> Option<AlignOut> {
    let n = a.len();
    if n < 8 || max_lag == 0 {
        return None;
    }
    let (ms, mism, pp) = scoring(a, k)?;
    let g = gap_cost(mism, gap_mult);
    let l = max_lag.min(n - 1);
    let threshold = (4.0 * ((n * l) as f64).log2()).ceil() as i32 + 16;
    let x_drop = if use_xdrop { threshold } else { i32::MAX };

    let mut prev_h = vec![0i32; l + 2];
    let mut cur_h = vec![0i32; l + 2];
    let mut prev_m = vec![Meta::default(); l + 2];
    let mut cur_m = vec![Meta::default(); l + 2];
    let mut lag_hits = vec![0u64; l + 2];
    let mut out: Vec<Seg> = Vec::new();

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
            let s = if eq { ms[ai as usize] } else { -mism };
            let vd = if dh > 0 || eq { dh + s } else { 0 };
            let vu = if uh > 0 { uh - g } else { 0 };
            let vl = if lh > 0 { lh - g } else { 0 };
            let h = vd.max(vu).max(vl).max(0);
            let from_d = h > 0 && h == vd;
            let from_u = h > 0 && !from_d && h == vu;
            // 对角线前驱的链没有被延续：它在这里结束
            if dh > 0 && !from_d && prev_m[lag].best >= threshold {
                emit(&mut out, &prev_m[lag]);
            }
            if h == 0 {
                cur_h[lag] = 0;
                continue;
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
            if m.best - h > x_drop {
                if m.best >= threshold {
                    emit(&mut out, &m);
                }
                cur_h[lag] = 0;
                continue;
            }
            cur_h[lag] = h;
            cur_m[lag] = m;
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

    // 同一条链可能分叉、重复上报：按起点去重，保留最高分
    out.sort_unstable_by(|x, y| (x.si, x.sl, -x.score, x.ei, x.el).cmp(&(y.si, y.sl, -y.score, y.ei, y.el)));
    out.dedup_by(|b, a| a.si == b.si && a.sl == b.sl);
    out.sort_unstable_by(|x, y| (-x.score, x.si, x.sl, x.ei, x.el).cmp(&(-y.score, y.si, y.sl, y.ei, y.el)));
    out.truncate(seg_cap);
    Some(AlignOut { segs: out, threshold, pp, lag_hits, max_lag: l })
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
