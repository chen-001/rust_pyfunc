//! 第三版结构特征层：把一个配对方案（可含交叉）或一组真实挂撤弧压成固定个数的数。
use super::v3_kernels::{Ctx, H};

pub const PAIR_FEATS: [&str; 19] = [
    "score", "pair_ratio", "unp_net", "unp_center", "depth_max", "depth_mean", "span_mean", "branch", "hairpin", "stem",
    "bf_share", "pair_center", "gain_bps", "span_p10", "span_p50", "span_p90", "span_cv", "cross_ratio", "crossed_share",
];
pub const NAIVE: [&str; 9] = ["active_share", "buy_share", "bound", "letter_imb", "vol_imb", "alternation", "walk_range", "walk_end", "greedy_pair"];
pub const ARC_FEATS: [&str; 19] = [
    "arcs_per_min", "in_window_share", "span_mean", "span_p10", "span_p50", "span_p90", "span_cv", "depth_mean", "depth_max",
    "top_share", "leaf_share", "cross_ratio", "volw_span", "center", "fast_share", "buy_share", "log_vol", "unmapped_share",
    "burst",
];

/// 配对特征。nested=true 时配对互不交叉，用一次扫描求深度；否则逐对比较求包含与交叉。
/// out 至少 17 个；len>=19 时再写交叉两项。
pub fn pair_feats(c: &Ctx, pairs: &[(usize, usize)], score_norm: f32, nested: bool, out: &mut [f32]) {
    let n = c.n;
    let np = pairs.len();
    out.iter_mut().for_each(|x| *x = f32::NAN);
    out[0] = score_norm;
    let bound = c.n_buy.min(c.n_sell) as f64;
    let active = (c.n_buy + c.n_sell) as f64;
    out[1] = (np as f64 / bound) as f32;
    let mut partner = vec![-1i64; n];
    for &(a, b) in pairs {
        partner[a] = b as i64;
        partner[b] = a as i64;
    }
    let (mut unp, mut unp_net, mut unp_pos) = (0u32, 0f64, 0f64);
    for p in 0..n {
        if partner[p] < 0 && c.side[p] != 0 {
            unp += 1;
            unp_net += c.side[p] as f64;
            unp_pos += p as f64;
        }
    }
    out[2] = (unp_net / active) as f32;
    if unp > 0 { out[3] = (unp_pos / unp as f64 / n as f64) as f32; }
    if np == 0 { return; }
    // 每对的“被包含次数+1”作为深度；“不包含任何配对”记为发夹；交叉对数
    let mut depth = vec![0u32; np];
    let mut leaf = vec![true; np];
    let (mut cross_pairs, mut crossed) = (0u64, vec![false; np]);
    if nested {
        let mut stack: Vec<usize> = Vec::new();
        let idx_of: std::collections::BTreeMap<usize, usize> = pairs.iter().enumerate().map(|(q, &(a, _))| (a, q)).collect();
        for p in 0..n {
            let q = partner[p];
            if q > p as i64 {
                let id = idx_of[&p];
                if let Some(&top) = stack.last() { leaf[top] = false; }
                stack.push(id);
                depth[id] = stack.len() as u32;
            } else if q >= 0 {
                stack.pop();
            }
        }
    } else {
        for a in 0..np {
            let (ka, ja) = pairs[a];
            let mut d = 1u32;
            for b in 0..np {
                if a == b { continue; }
                let (kb, jb) = pairs[b];
                if kb < ka && ja < jb { d += 1; }
                if ka < kb && jb < ja { leaf[a] = false; }
                if ka < kb && kb < ja && ja < jb {
                    cross_pairs += 1;
                    crossed[a] = true;
                    crossed[b] = true;
                }
            }
            depth[a] = d;
        }
    }
    let pf = np as f64;
    let nf = n as f64;
    let mut spans: Vec<usize> = pairs.iter().map(|&(a, b)| b - a).collect();
    let (mut sum_d, mut max_d, mut top, mut hp, mut cont, mut bf, mut center, mut gain) = (0u64, 0u32, 0u32, 0u32, 0u32, 0u32, 0f64, 0f64);
    for (q, &(a, b)) in pairs.iter().enumerate() {
        sum_d += depth[q] as u64;
        max_d = max_d.max(depth[q]);
        if depth[q] == 1 { top += 1; }
        if leaf[q] { hp += 1; }
        if a + 1 < b - 1 && partner[a + 1] == (b - 1) as i64 { cont += 1; }
        let (bi, si) = if c.side[a] > 0 { bf += 1; (a, b) } else { (b, a) };
        center += (a + b) as f64 / 2.0;
        gain += ((c.price[si] / c.price[bi]) as f64 - 1.0) * 1e4;
    }
    let ssum: f64 = spans.iter().map(|&s| s as f64).sum();
    let smean = ssum / pf;
    let svar = spans.iter().map(|&s| (s as f64 - smean).powi(2)).sum::<f64>() / pf;
    spans.sort_unstable();
    let q = |f: f64| spans[((np - 1) as f64 * f).floor() as usize] as f64 / nf;
    out[4] = (max_d as f64 / pf) as f32;
    out[5] = (sum_d as f64 / pf) as f32;
    out[6] = (smean / nf) as f32;
    out[7] = (top as f64 / pf) as f32;
    out[8] = (hp as f64 / pf) as f32;
    out[9] = (pf / (pf - cont as f64).max(1.0)) as f32;
    out[10] = (bf as f64 / pf) as f32;
    out[11] = (center / pf / nf) as f32;
    out[12] = (gain / pf) as f32;
    out[13] = q(0.1) as f32;
    out[14] = q(0.5) as f32;
    out[15] = q(0.9) as f32;
    out[16] = if smean > 0.0 { (svar.sqrt() / smean) as f32 } else { f32::NAN };
    if out.len() >= 19 {
        if nested {
            out[17] = 0.0;
            out[18] = 0.0;
        } else {
            out[17] = (cross_pairs as f64 / pf) as f32;
            out[18] = (crossed.iter().filter(|&&x| x).count() as f64 / pf) as f32;
        }
    }
}

pub fn naive_feats(c: &Ctx, out: &mut [f32]) {
    out.iter_mut().for_each(|x| *x = f32::NAN);
    let n = c.n;
    let active = (c.n_buy + c.n_sell) as f64;
    out[0] = (active / n as f64) as f32;
    if active == 0.0 { return; }
    out[1] = (c.n_buy as f64 / active) as f32;
    out[2] = (c.n_buy.min(c.n_sell) as f64 / active) as f32;
    out[3] = ((c.n_buy - c.n_sell) as f64 / active) as f32;
    let (mut vb, mut vs) = (0f64, 0f64);
    for i in 0..n {
        if c.side[i] > 0 { vb += c.vol[i] as f64; } else if c.side[i] < 0 { vs += c.vol[i] as f64; }
    }
    if vb + vs > 0.0 { out[4] = ((vb - vs) / (vb + vs)) as f32; }
    let (mut prev, mut flips, mut adj, mut walk, mut lo, mut hi, mut greedy) = (0i8, 0u32, 0u32, 0i32, 0i32, 0i32, 0i32);
    let mut stack: Vec<usize> = Vec::new();
    for i in 0..n {
        let x = c.side[i];
        if x == 0 { continue; }
        if prev != 0 {
            adj += 1;
            if prev != x { flips += 1; }
        }
        prev = x;
        walk += x as i32;
        lo = lo.min(walk);
        hi = hi.max(walk);
        match stack.last() {
            Some(&t) if c.side[t] != x && i - t - 1 >= H => {
                stack.pop();
                greedy += 1;
            }
            _ => stack.push(i),
        }
    }
    if adj > 0 { out[5] = flips as f32 / adj as f32; }
    out[6] = ((hi - lo) as f64 / active) as f32;
    out[7] = (walk as f64 / active) as f32;
    let b = c.n_buy.min(c.n_sell);
    if b > 0 { out[8] = greedy as f32 / b as f32; }
}

/// 树状数组（计数）
struct Fen(Vec<i64>);
impl Fen {
    fn new(n: usize) -> Self { Fen(vec![0; n + 1]) }
    fn add(&mut self, i: usize, v: i64) {
        let mut i = i + 1;
        while i < self.0.len() { self.0[i] += v; i += i & i.wrapping_neg(); }
    }
    fn sum(&self, i: usize) -> i64 {
        // [0, i)
        let (mut i, mut s) = (i, 0);
        while i > 0 { s += self.0[i]; i -= i & i.wrapping_neg(); }
        s
    }
}

/// 真实挂撤弧特征。start/end 为微秒；win_len_us 为时段长度；bar_us 用于“同一时间片结束的弧”的爆发度。
pub fn arc_feats(arcs: &[(i64, i64, bool, f64, bool, bool)], win_lo_us: i64, win_len_us: i64, bar_us: i64, out: &mut [f32]) {
    // 元组：(开始, 结束, 是否买撤, 撤单量, 挂单是否在时段内, 是否无法定位)
    out.iter_mut().for_each(|x| *x = f32::NAN);
    let np = arcs.len();
    let minutes = win_len_us as f64 / 60e6;
    out[0] = (np as f64 / minutes) as f32;
    if np == 0 { return; }
    let pf = np as f64;
    let len = win_len_us as f64;
    let mut spans: Vec<i64> = arcs.iter().map(|a| a.1 - a.0).collect();
    let smean = spans.iter().map(|&s| s as f64).sum::<f64>() / pf;
    let svar = spans.iter().map(|&s| (s as f64 - smean).powi(2)).sum::<f64>() / pf;
    spans.sort_unstable();
    let q = |f: f64| spans[((np - 1) as f64 * f).floor() as usize] as f64 / len;
    // 包含与交叉：按端点离散化后用树状数组
    let mut ends: Vec<i64> = arcs.iter().map(|a| a.1).collect();
    ends.sort_unstable();
    ends.dedup();
    let rank = |e: i64| ends.partition_point(|&x| x < e);
    let mut order: Vec<usize> = (0..np).collect();
    order.sort_unstable_by(|&a, &b| arcs[a].0.cmp(&arcs[b].0).then(arcs[b].1.cmp(&arcs[a].1)).then(a.cmp(&b)));
    let mut contained_by = vec![0i64; np];
    let mut contains_any = vec![false; np];
    let mut crossings = 0i64;
    let mut crossed = vec![false; np];
    {
        // 开始严格更早、结束严格更晚 → 包含；开始更早、结束落在本弧内部（严格） → 交叉
        let mut fen = Fen::new(ends.len());
        let mut g = 0;
        while g < np {
            let s0 = arcs[order[g]].0;
            let mut h2 = g;
            while h2 < np && arcs[order[h2]].0 == s0 { h2 += 1; }
            for &a in &order[g..h2] {
                let (s, e) = (arcs[a].0, arcs[a].1);
                let total = fen.sum(ends.len());
                contained_by[a] = total - fen.sum(rank(e) + 1).min(total);
                let c = fen.sum(rank(e)) - fen.sum(ends.partition_point(|&x| x <= s));
                if c > 0 { crossed[a] = true; }
                crossings += c;
            }
            for &a in &order[g..h2] { fen.add(rank(arcs[a].1), 1); }
            g = h2;
        }
        // 反向：开始严格更晚、结束严格更早 → 本弧包含别的弧；被后面的弧交叉
        let mut fen2 = Fen::new(ends.len());
        let mut g = np;
        while g > 0 {
            let s0 = arcs[order[g - 1]].0;
            let mut h2 = g;
            while h2 > 0 && arcs[order[h2 - 1]].0 == s0 { h2 -= 1; }
            for &a in &order[h2..g] {
                if fen2.sum(rank(arcs[a].1)) > 0 { contains_any[a] = true; }
            }
            for &a in &order[h2..g] { fen2.add(rank(arcs[a].1), 1); }
            g = h2;
        }
    }
    // 对称标记“被交叉”：在正向扫描里，若弧 b 与更早的弧 a 交叉，a 也应记为被交叉
    {
        let mut by_start: Vec<(i64, i64, usize)> = arcs.iter().enumerate().map(|(i, a)| (a.0, a.1, i)).collect();
        by_start.sort_unstable();
        // 对每个弧 a，看是否存在开始在 (s_a, e_a) 内、结束晚于 e_a 的弧：开始区间内结束时刻的最大值 > e_a
        let starts: Vec<i64> = by_start.iter().map(|x| x.0).collect();
        // 稀疏表求区间最大结束时刻
        let m = by_start.len();
        let lg = (usize::BITS - m.leading_zeros()) as usize;
        let mut sp: Vec<Vec<i64>> = vec![by_start.iter().map(|x| x.1).collect()];
        for l in 1..lg {
            let prev = &sp[l - 1];
            let w = 1 << l;
            let row: Vec<i64> = (0..m.saturating_sub(w - 1)).map(|i| prev[i].max(prev[i + (w >> 1)])).collect();
            sp.push(row);
        }
        let qmax = |l: usize, r: usize| -> i64 {
            let k = (usize::BITS - (r - l).leading_zeros() - 1) as usize;
            sp[k][l].max(sp[k][r - (1 << k)])
        };
        for &(s, e, i) in &by_start {
            let l = starts.partition_point(|&x| x <= s);
            let r = starts.partition_point(|&x| x < e);
            if l < r && qmax(l, r) > e { crossed[i] = true; }
        }
    }
    let (mut sd, mut md, mut top, mut leafc) = (0f64, 0i64, 0f64, 0f64);
    for a in 0..np {
        let d = contained_by[a] + 1;
        sd += d as f64;
        md = md.max(d);
        if d == 1 { top += 1.0; }
        if !contains_any[a] { leafc += 1.0; }
    }
    let (mut vw, mut vs, mut center, mut fast, mut buy, mut lv, mut unm, mut inw) = (0f64, 0f64, 0f64, 0f64, 0f64, 0f64, 0f64, 0f64);
    let mut burst: std::collections::BTreeMap<i64, u32> = std::collections::BTreeMap::new();
    for a in arcs {
        let s = (a.1 - a.0) as f64;
        vw += a.3 * s;
        vs += a.3;
        center += ((a.0 + a.1) as f64 / 2.0 - win_lo_us as f64) / len;
        if a.1 - a.0 < 3_000_000 { fast += 1.0; }
        if a.2 { buy += 1.0; }
        lv += (1.0 + a.3).ln();
        if a.5 { unm += 1.0; }
        if a.4 { inw += 1.0; }
        *burst.entry((a.1 - win_lo_us) / bar_us).or_insert(0) += 1;
    }
    out[1] = (inw / pf) as f32;
    out[2] = (smean / len) as f32;
    out[3] = q(0.1) as f32;
    out[4] = q(0.5) as f32;
    out[5] = q(0.9) as f32;
    out[6] = if smean > 0.0 { (svar.sqrt() / smean) as f32 } else { f32::NAN };
    out[7] = (sd / pf) as f32;
    out[8] = (md as f64 / pf) as f32;
    out[9] = (top / pf) as f32;
    out[10] = (leafc / pf) as f32;
    out[11] = (crossings as f64 / pf) as f32;
    out[12] = if vs > 0.0 { (vw / vs / len) as f32 } else { f32::NAN };
    out[13] = (center / pf) as f32;
    out[14] = (fast / pf) as f32;
    out[15] = (buy / pf) as f32;
    out[16] = (lv / pf) as f32;
    out[17] = (unm / pf) as f32;
    out[18] = (*burst.values().max().unwrap() as f64 / pf) as f32;
}
