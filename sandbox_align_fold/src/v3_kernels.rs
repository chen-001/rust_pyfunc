//! 第三版计算核。每个核都有“基准写法”和“提速写法”两条路，结果必须逐字节相同：
//!  - 无交叉配对：基准在原序列上算；提速版先去掉中性字母（它们不能配对，不改变任何区间的最优值），
//!    再把每一行按买、卖字母分开连续存放，内层只扫对侧字母。
//!  - 连续嵌套奖励：同一份代码，基准传入全部位置，提速版只传非中性位置。
//!  - 配对概率：精确计算所有配对方案的加权和（不抽样）。累加顺序在定义里写死为“按对侧字母分 4 路累加”，
//!    基准逐个位置判断，提速版用对侧字母的连续数组，所以数值逐字节相同。
use crate::fold::{Table, NEG};
use crate::v3_letters::Seq;

pub const RULES: [&str; 12] = ["cnt", "dec", "vol", "rna", "profit", "loss", "bf", "sf", "short", "h10", "gain", "stack"];
pub const H: usize = 3;

#[derive(Clone, Copy, PartialEq)]
pub enum Mode {
    Base,
    Fast,
}

pub struct Ctx {
    pub n: usize,
    pub side: Vec<i8>,
    pub side2: Option<Vec<i8>>,
    pub strength: Vec<u8>,
    pub price: Vec<f32>,
    pub vol: Vec<f32>,
    pub vint: Vec<i32>,
    pub decay: Vec<i32>,
    pub active: Vec<bool>,
    pub act: Vec<usize>,
    pub n_buy: i32,
    pub n_sell: i32,
    pub sum_vb: i64,
    pub sum_vs: i64,
    pub lshort: usize,
}

pub fn make_ctx(s: &Seq, side2: Option<&[i8]>) -> Ctx {
    let n = s.side.len();
    let side: Vec<i8> = match side2 {
        Some(b) => (0..n).map(|i| if b[i] != 0 { s.side[i] } else { 0 }).collect(),
        None => s.side.clone(),
    };
    let (mut sumv, mut cnt) = (0f64, 0f64);
    for i in 0..n {
        if side[i] != 0 { sumv += s.vol[i] as f64; cnt += 1.0; }
    }
    let mean = if cnt > 0.0 && sumv > 0.0 { sumv / cnt } else { 1.0 };
    let vint: Vec<i32> = s.vol.iter().map(|&v| ((v as f64 / mean * 100.0) as i32).clamp(1, 3000)).collect();
    let tau = (n as f64 / 8.0).max(1.0);
    let decay = (0..=n).map(|d| ((1000.0 * (-(d as f64) / tau).exp()).round() as i32).max(1)).collect();
    let (mut n_buy, mut n_sell, mut sum_vb, mut sum_vs) = (0, 0, 0i64, 0i64);
    for i in 0..n {
        if side[i] > 0 { n_buy += 1; sum_vb += vint[i] as i64; } else if side[i] < 0 { n_sell += 1; sum_vs += vint[i] as i64; }
    }
    let active: Vec<bool> = side.iter().map(|&x| x != 0).collect();
    let act = (0..n).filter(|&i| active[i]).collect();
    Ctx {
        n, side, side2: side2.map(|b| b.to_vec()), strength: s.strength.clone(), price: s.price.clone(), vol: s.vol.clone(),
        vint, decay, active, act, n_buy, n_sell, sum_vb, sum_vs, lshort: (n / 8).max(H + 1),
    }
}

pub fn rule_h(rule: usize) -> usize {
    if rule == 9 { 10 } else { H }
}

/// 配对权重（原序列下标，k<j）；不可配对返回 NEG。
#[inline(always)]
pub fn weight(c: &Ctx, rule: usize, k: usize, j: usize) -> i32 {
    let (sk, sj) = (c.side[k], c.side[j]);
    if sk as i32 * sj as i32 != -1 { return NEG; }
    if let Some(s2) = &c.side2 {
        if s2[k] as i32 * s2[j] as i32 != -1 { return NEG; }
    }
    match rule {
        0 | 9 => 1,
        1 => c.decay[j - k],
        2 => c.vint[k].min(c.vint[j]),
        3 => c.strength[k] as i32 + c.strength[j] as i32 - 1,
        4 | 5 => {
            let (b, s) = if sk > 0 { (c.price[k], c.price[j]) } else { (c.price[j], c.price[k]) };
            if (rule == 4 && s > b) || (rule == 5 && s < b) { 1 } else { NEG }
        }
        6 => if sk > 0 { 1 } else { NEG },
        7 => if sk < 0 { 1 } else { NEG },
        8 => if j - k <= c.lshort { 1 } else { NEG },
        10 => {
            let (b, s) = if sk > 0 { (c.price[k], c.price[j]) } else { (c.price[j], c.price[k]) };
            if s > b { ((((s / b) as f64 - 1.0) * 1e4).round() as i32).clamp(1, 100_000) } else { NEG }
        }
        11 => 2,
        _ => NEG,
    }
}

pub fn rule_ub(c: &Ctx, rule: usize) -> f64 {
    let m = c.n_buy.min(c.n_sell) as f64;
    match rule {
        1 => 1000.0 * m,
        2 => c.sum_vb.min(c.sum_vs) as f64,
        3 | 11 => 3.0 * m,
        _ => m,
    }
}

// ---------------- 提速版无交叉配对 ----------------

/// 得分类型：小分值规则用 16 位整数（一次处理的数翻倍），其余用 32 位；最大值运算对两者都是精确的。
pub trait Sc: Copy + Ord + Default + std::ops::Add<Output = Self> {
    const NEG: Self;
    fn from_w(v: i32) -> Self;
    fn to_i32(self) -> i32;
}
impl Sc for i32 {
    const NEG: i32 = NEG;
    #[inline] fn from_w(v: i32) -> i32 { if v > NEG / 2 { v } else { NEG } }
    #[inline] fn to_i32(self) -> i32 { self }
}
impl Sc for i16 {
    const NEG: i16 = i16::MIN / 2;
    #[inline] fn from_w(v: i32) -> i16 { if v > NEG / 2 { v as i16 } else { i16::MIN / 2 } }
    #[inline] fn to_i32(self) -> i32 { self as i32 }
}

#[derive(Default)]
pub struct FastTable<T: Sc> {
    st: usize,
    /// 按列存放：c[x*st + i] = 区间 [i, x) 的最优得分，保证内层循环全是连续内存
    c: Vec<T>,
    best: Vec<T>,
    kmax: Vec<i64>,
    bpos: Vec<usize>,
    spos: Vec<usize>,
    stack: Vec<(usize, usize)>,
}

impl<T: Sc> FastTable<T> {
    /// act：参与计算的位置（原下标，递增）；side 为原序列方向；h 按原下标计。
    /// 对每个 j，先取上一列，再按对侧字母 k 逐个更新所有 i<=k 的行（这一步在 i 上连续）。
    pub fn solve<W: Fn(usize, usize) -> i32>(&mut self, act: &[usize], side: &[i8], h: usize, w: &W) -> i32 {
        let m = act.len();
        let st = m + 1;
        self.st = st;
        self.bpos.clear();
        self.spos.clear();
        for (x, &o) in act.iter().enumerate() {
            if side[o] > 0 { self.bpos.push(x); } else { self.spos.push(x); }
        }
        self.kmax.clear();
        let mut p = 0usize;
        for j in 0..m {
            while p < m && act[p] + h + 1 <= act[j] { p += 1; }
            self.kmax.push(p as i64 - 1);
        }
        let zero = T::default();
        self.c.clear();
        self.c.resize(st * st, zero);
        self.best.clear();
        self.best.resize(st, zero);
        let (c, best) = (&mut self.c[..], &mut self.best[..]);
        for j in 0..m {
            best[..=j].copy_from_slice(&c[j * st..j * st + j + 1]);
            let km = self.kmax[j];
            if km >= 0 {
                let km = km as usize;
                let opp = if side[act[j]] > 0 { &self.spos } else { &self.bpos };
                for &k in opp.iter().take_while(|&&k| k <= km) {
                    let v = w(act[k], act[j]);
                    if v <= NEG / 2 { continue; }
                    let zt = T::from_w(v) + c[j * st + k + 1];
                    let col = &c[k * st..k * st + k + 1];
                    for (b, &r) in best[..=k].iter_mut().zip(col) {
                        *b = (*b).max(r + zt);
                    }
                }
            }
            c[(j + 1) * st..(j + 1) * st + j + 1].copy_from_slice(&best[..=j]);
        }
        c[m * st].to_i32()
    }

    pub fn traceback<W: Fn(usize, usize) -> i32>(&mut self, act: &[usize], w: &W, partner: &mut [i32]) {
        let st = self.st;
        let m = act.len();
        let c = &self.c;
        let at = |i: usize, x: usize| c[x * st + i].to_i32();
        self.stack.clear();
        self.stack.push((0, m));
        while let Some((i, x0)) = self.stack.pop() {
            let mut x = x0;
            while x > i {
                let j = x - 1;
                let cur = at(i, x);
                if cur == at(i, j) { x -= 1; continue; }
                let km = self.kmax[j] as usize;
                let mut found = usize::MAX;
                for k in (i..=km).rev() {
                    let v = w(act[k], act[j]);
                    if v > NEG / 2 && at(i, k) + v + at(k + 1, j) == cur { found = k; break; }
                }
                partner[act[found]] = act[j] as i32;
                partner[act[j]] = act[found] as i32;
                self.stack.push((found + 1, j));
                x = found;
            }
        }
    }
}

/// 分值上限小（不超过 3 × 序列长度的一半）的规则走 16 位
pub fn small_score_rule(rule: usize) -> bool {
    matches!(rule, 0 | 3 | 4 | 5 | 6 | 7 | 8 | 9)
}

// ---------------- 连续嵌套奖励（同一份代码，传入不同位置集合） ----------------

#[derive(Default)]
pub struct StackBuf {
    r: Vec<i32>,
    s: Vec<i32>,
    z: Vec<i32>,
    kmax: Vec<i64>,
}

/// 每对基础 2 分，若紧贴着内层也配对（原序列上相邻）再加 1 分。返回 (得分, 配对表)。
pub fn stack_fold(c: &Ctx, pos: &[usize], buf: &mut StackBuf, partner: &mut [i32]) -> i32 {
    let m = pos.len();
    let st = m + 1;
    buf.r.clear();
    buf.r.resize(st * st, 0);
    buf.s.clear();
    buf.s.resize(m * m, NEG);
    buf.z.clear();
    buf.z.resize(m + 1, NEG);
    buf.kmax.clear();
    let mut p = 0usize;
    for j in 0..m {
        while p < m && pos[p] + H + 1 <= pos[j] { p += 1; }
        buf.kmax.push(p as i64 - 1);
    }
    let (r, s, z) = (&mut buf.r[..], &mut buf.s[..], &mut buf.z[..]);
    let stacked = |s: &[i32], k: usize, j: usize| -> i32 {
        if k + 1 < j && pos[k + 1] == pos[k] + 1 && pos[j - 1] + 1 == pos[j] { s[(k + 1) * m + j - 1] } else { NEG }
    };
    for j in 0..m {
        let km = buf.kmax[j];
        if km < 0 {
            for i in 0..=j { r[i * st + j + 1] = r[i * st + j]; }
            continue;
        }
        let km = km as usize;
        for k in 0..=km {
            let v = weight(c, 11, pos[k], pos[j]);
            if v > NEG / 2 {
                let inner = r[(k + 1) * st + j];
                let sk = stacked(s, k, j);
                let best = if sk > NEG / 2 { inner.max(sk + 1) } else { inner };
                s[k * m + j] = v + best;
                z[k] = s[k * m + j];
            } else {
                z[k] = NEG;
            }
        }
        for i in 0..=j {
            let mut best = r[i * st + j];
            if i <= km {
                for (a, b) in r[i * st + i..i * st + km + 1].iter().zip(&z[i..=km]) {
                    best = best.max(*a + *b);
                }
            }
            r[i * st + j + 1] = best;
        }
    }
    // 倒推：区间状态与配对状态两种；配对内部若“紧贴内层配对”与“内部自由”同分，固定优先紧贴
    enum St { Seg(usize, usize), Pair(usize, usize) }
    let mut stack = vec![St::Seg(0, m)];
    while let Some(stt) = stack.pop() {
        match stt {
            St::Seg(i, x0) => {
                let mut x = x0;
                while x > i {
                    let j = x - 1;
                    let cur = r[i * st + x];
                    if cur == r[i * st + j] { x -= 1; continue; }
                    let km = buf.kmax[j] as usize;
                    let mut found = usize::MAX;
                    for k in (i..=km).rev() {
                        let sv = s[k * m + j];
                        if sv > NEG / 2 && r[i * st + k] + sv == cur { found = k; break; }
                    }
                    stack.push(St::Pair(found, j));
                    x = found;
                }
            }
            St::Pair(k, j) => {
                partner[pos[k]] = pos[j] as i32;
                partner[pos[j]] = pos[k] as i32;
                let v = weight(c, 11, pos[k], pos[j]);
                let sk = stacked(s, k, j);
                if sk > NEG / 2 && s[k * m + j] == v + sk + 1 {
                    stack.push(St::Pair(k + 1, j - 1));
                } else {
                    stack.push(St::Seg(k + 1, j));
                }
            }
        }
    }
    r[m]
}

// ---------------- 配对概率（精确） ----------------

pub const PROB_FEATS: [&str; 8] = ["exp_pair_ratio", "pos_entropy", "confident_share", "exp_span", "exp_bf", "exp_center", "unp_prob", "free_energy"];
const SIG: f64 = 2.0;
const ISIG: f64 = 0.5;
const ISIG2: f64 = 0.25;
const BW: f64 = std::f64::consts::E;

#[derive(Default)]
pub struct ProbBuf {
    r: Vec<f64>,
    a: Vec<f64>,
    b: Vec<f64>,
    zz: Vec<f64>,
    acc: Vec<f64>,
    rb: Vec<f64>,
    rs: Vec<f64>,
    bb: Vec<f64>,
    bs: Vec<f64>,
    pos_sum: Vec<f64>,
    pos_ent: Vec<f64>,
    pos_max: Vec<f64>,
}

struct ProbAcc {
    sp: f64,
    sspan: f64,
    sbf: f64,
    scenter: f64,
}

#[inline]
fn record_pair(buf_sum: &mut [f64], buf_ent: &mut [f64], buf_max: &mut [f64], acc: &mut ProbAcc, side: &[i8], k: usize, j: usize, p: f64) {
    if p > 0.0 {
        let e = -p * p.ln();
        for q in [k, j] {
            buf_sum[q] += p;
            buf_ent[q] += e;
            if p > buf_max[q] { buf_max[q] = p; }
        }
        acc.sp += p;
        acc.sspan += p * (j - k) as f64;
        if side[k] > 0 { acc.sbf += p; }
        acc.scenter += p * (k + j) as f64 * 0.5;
    }
}

fn prob_finish(c: &Ctx, z: f64, buf: &ProbBuf, acc: &ProbAcc) -> [f32; 8] {
    let n = c.n;
    let mut out = [f32::NAN; 8];
    if !(z.is_finite() && z > 0.0) { return out; }
    let bound = c.n_buy.min(c.n_sell) as f64;
    let (mut hs, mut conf, mut unp, mut na) = (0f64, 0f64, 0f64, 0f64);
    for i in 0..n {
        if !c.active[i] { continue; }
        let u = (1.0 - buf.pos_sum[i]).max(0.0);
        let mut h = buf.pos_ent[i];
        if u > 0.0 { h -= u * u.ln(); }
        hs += h;
        if buf.pos_max[i] > 0.5 { conf += 1.0; }
        unp += u;
        na += 1.0;
    }
    out[0] = (acc.sp / bound) as f32;
    out[1] = (hs / na) as f32;
    out[2] = (conf / na) as f32;
    if acc.sp > 0.0 {
        out[3] = (acc.sspan / acc.sp / n as f64) as f32;
        out[4] = (acc.sbf / acc.sp) as f32;
        out[5] = (acc.scenter / acc.sp / n as f64) as f32;
    }
    out[6] = (unp / na) as f32;
    out[7] = ((z.ln() + n as f64 * SIG.ln()) / n as f64) as f32;
    out
}

fn prob_reset(c: &Ctx, buf: &mut ProbBuf) {
    let n = c.n;
    for v in [&mut buf.pos_sum, &mut buf.pos_ent, &mut buf.pos_max] {
        v.clear();
        v.resize(n, 0.0);
    }
    buf.zz.clear();
    buf.zz.resize(n + 1, 0.0);
    buf.acc.clear();
    buf.acc.resize(n + 1, 0.0);
}

/// 基准写法：逐个位置判断能否配对。
pub fn prob_base(c: &Ctx, buf: &mut ProbBuf) -> [f32; 8] {
    let n = c.n;
    let st = n + 1;
    let side = &c.side;
    prob_reset(c, buf);
    buf.r.clear();
    buf.r.resize(st * st, 1.0);
    let ok = |k: usize, j: usize| side[k] as i32 * side[j] as i32 == -1;
    {
        let (r, zz) = (&mut buf.r[..], &mut buf.zz[..]);
        for j in 0..n {
            let elig = side[j] != 0 && j >= H + 1;
            let kmax = if elig { j - H - 1 } else { 0 };
            if elig {
                for k in 0..=kmax {
                    if ok(k, j) { zz[k] = BW * r[(k + 1) * st + j] * ISIG2; }
                }
            }
            for i in 0..=j {
                let mut s = r[i * st + j] * ISIG;
                if elig && i <= kmax {
                    let mut acc = [0f64; 4];
                    let mut t = 0usize;
                    for k in i..=kmax {
                        if ok(k, j) { acc[t & 3] += r[i * st + k] * zz[k]; t += 1; }
                    }
                    s += (acc[0] + acc[1]) + (acc[2] + acc[3]);
                }
                r[i * st + j + 1] = s;
            }
        }
    }
    let z = buf.r[n];
    let mut pa = ProbAcc { sp: 0.0, sspan: 0.0, sbf: 0.0, scenter: 0.0 };
    if z.is_finite() && z > 0.0 {
        buf.a.clear();
        buf.a.resize(st * st, 0.0);
        buf.b.clear();
        buf.b.resize(st * st, 0.0);
        buf.a[n] = 1.0;
        let (r, a, b, zz, acc) = (&buf.r[..], &mut buf.a[..], &mut buf.b[..], &mut buf.zz[..], &mut buf.acc[..]);
        for j in (0..n).rev() {
            let x = j + 1;
            let elig = side[j] != 0 && j >= H + 1;
            let kmax = if elig { j - H - 1 } else { 0 };
            if elig {
                for k in 0..=kmax {
                    if ok(k, j) { zz[k] = BW * r[(k + 1) * st + j] * ISIG2; acc[k] = 0.0; }
                }
            }
            for i in 0..=j {
                let o = a[i * st + x] + b[i * st + x];
                if o == 0.0 { continue; }
                a[i * st + j] += o * ISIG;
                if elig && i <= kmax {
                    for k in i..=kmax {
                        if ok(k, j) {
                            b[i * st + k] += o * zz[k];
                            acc[k] += o * r[i * st + k];
                        }
                    }
                }
            }
            if elig {
                for k in 0..=kmax {
                    if ok(k, j) {
                        a[(k + 1) * st + j] += acc[k] * BW * ISIG2;
                        let p = acc[k] * zz[k] / z;
                        record_pair(&mut buf.pos_sum, &mut buf.pos_ent, &mut buf.pos_max, &mut pa, side, k, j, p);
                    }
                }
            }
        }
    }
    prob_finish(c, z, buf, &pa)
}

/// 提速写法：对侧字母的数值放在连续数组里，按 4 路批量累加（与基准的累加顺序完全一致）；
/// 区间表按列存放（rc[x*st+i] = 区间 [i,x)），按列读写都连续。
pub fn prob_fast(c: &Ctx, buf: &mut ProbBuf) -> [f32; 8] {
    let n = c.n;
    let st = n + 1;
    let side = &c.side;
    prob_reset(c, buf);
    let mut bpos = Vec::new();
    let mut spos = Vec::new();
    let mut cb = Vec::with_capacity(n + 1);
    let mut cs = Vec::with_capacity(n + 1);
    let mut slot = vec![usize::MAX; n];
    for x in 0..n {
        cb.push(bpos.len());
        cs.push(spos.len());
        if side[x] > 0 { slot[x] = bpos.len(); bpos.push(x); } else if side[x] < 0 { slot[x] = spos.len(); spos.push(x); }
    }
    cb.push(bpos.len());
    cs.push(spos.len());
    let (nb, ns) = (bpos.len().max(1), spos.len().max(1));
    buf.r.clear();
    buf.r.resize(st * st, 1.0);
    buf.rb.clear();
    buf.rb.resize(st * nb, 1.0);
    buf.rs.clear();
    buf.rs.resize(st * ns, 1.0);
    let mut zo = vec![0f64; nb.max(ns)];
    {
        let (rc, rb, rs) = (&mut buf.r[..], &mut buf.rb[..], &mut buf.rs[..]);
        for j in 0..n {
            let elig = side[j] != 0 && j >= H + 1;
            let buy_j = side[j] > 0;
            let (opp, cnt, nopp) = if buy_j { (&spos, &cs, ns) } else { (&bpos, &cb, nb) };
            let tmax = if elig { cnt[j - H] } else { 0 };
            for t in 0..tmax {
                zo[t] = BW * rc[j * st + opp[t] + 1] * ISIG2;
            }
            let ropp: &[f64] = if buy_j { &*rs } else { &*rb };
            for i in 0..=j {
                let mut s = rc[j * st + i] * ISIG;
                let t0 = cnt[i];
                if elig && t0 < tmax {
                    let row = &ropp[i * nopp + t0..i * nopp + tmax];
                    let zs = &zo[t0..tmax];
                    let mut acc = [0f64; 4];
                    let mut ch_r = row.chunks_exact(4);
                    let mut ch_z = zs.chunks_exact(4);
                    for (a4, b4) in (&mut ch_r).zip(&mut ch_z) {
                        acc[0] += a4[0] * b4[0];
                        acc[1] += a4[1] * b4[1];
                        acc[2] += a4[2] * b4[2];
                        acc[3] += a4[3] * b4[3];
                    }
                    for (l, (a1, b1)) in ch_r.remainder().iter().zip(ch_z.remainder()).enumerate() {
                        acc[l] += a1 * b1;
                    }
                    s += (acc[0] + acc[1]) + (acc[2] + acc[3]);
                }
                rc[(j + 1) * st + i] = s;
            }
            let x = j + 1;
            if x < n && side[x] != 0 {
                let sl = slot[x];
                if side[x] > 0 { for i in 0..=j { rb[i * nb + sl] = rc[x * st + i]; } } else { for i in 0..=j { rs[i * ns + sl] = rc[x * st + i]; } }
            }
        }
    }
    let z = buf.r[n * st];
    let mut pa = ProbAcc { sp: 0.0, sspan: 0.0, sbf: 0.0, scenter: 0.0 };
    if z.is_finite() && z > 0.0 {
        buf.a.clear();
        buf.a.resize(st * st, 0.0);
        buf.bb.clear();
        buf.bb.resize(st * nb, 0.0);
        buf.bs.clear();
        buf.bs.resize(st * ns, 0.0);
        buf.a[n * st] = 1.0;
        let mut acc = vec![0f64; nb.max(ns)];
        let (rc, rb, rs, ac, bb, bs) = (&buf.r[..], &buf.rb[..], &buf.rs[..], &mut buf.a[..], &mut buf.bb[..], &mut buf.bs[..]);
        for j in (0..n).rev() {
            let x = j + 1;
            let elig = side[j] != 0 && j >= H + 1;
            let buy_j = side[j] > 0;
            let (opp, cnt, nopp) = if buy_j { (&spos, &cs, ns) } else { (&bpos, &cb, nb) };
            let tmax = if elig { cnt[j - H] } else { 0 };
            for t in 0..tmax {
                zo[t] = BW * rc[j * st + opp[t] + 1] * ISIG2;
                acc[t] = 0.0;
            }
            let xs = if x < n { side[x] } else { 0 };
            for i in 0..=j {
                // 读 O(i,x)：直接累加部分 + （若第 x 个字母有方向）配对左端累加部分
                let o = if xs != 0 {
                    let sl = slot[x];
                    ac[x * st + i] + if xs > 0 { bb[i * nb + sl] } else { bs[i * ns + sl] }
                } else {
                    ac[x * st + i] + 0.0
                };
                if o == 0.0 { continue; }
                ac[j * st + i] += o * ISIG;
                let t0 = cnt[i];
                if elig && t0 < tmax {
                    let (bopp, ropp): (&mut [f64], &[f64]) = if buy_j { (&mut *bs, rs) } else { (&mut *bb, rb) };
                    let brow = &mut bopp[i * nopp + t0..i * nopp + tmax];
                    let rrow = &ropp[i * nopp + t0..i * nopp + tmax];
                    let zs = &zo[t0..tmax];
                    let a2 = &mut acc[t0..tmax];
                    for q in 0..brow.len() {
                        brow[q] += o * zs[q];
                        a2[q] += o * rrow[q];
                    }
                }
            }
            for t in 0..tmax {
                let k = opp[t];
                ac[j * st + k + 1] += acc[t] * BW * ISIG2;
                let p = acc[t] * zo[t] / z;
                record_pair(&mut buf.pos_sum, &mut buf.pos_ent, &mut buf.pos_max, &mut pa, side, k, j, p);
            }
        }
    }
    prob_finish(c, z, buf, &pa)
}

// ---------------- 允许交叉：买卖二分图最大权匹配（匈牙利算法） ----------------

pub fn cross_match(c: &Ctx) -> (i64, Vec<(usize, usize)>) {
    let n = c.n;
    let bu: Vec<usize> = (0..n).filter(|&i| c.side[i] > 0).collect();
    let se: Vec<usize> = (0..n).filter(|&i| c.side[i] < 0).collect();
    if bu.is_empty() || se.is_empty() { return (0, Vec::new()); }
    let (rows, cols) = if bu.len() <= se.len() { (&bu, &se) } else { (&se, &bu) };
    let wt = |a: usize, b: usize| -> i64 {
        let (k, j) = if a < b { (a, b) } else { (b, a) };
        if j - k > H { c.decay[j - k] as i64 } else { 0 }
    };
    let (nr, nc) = (rows.len(), cols.len());
    let wm: Vec<i64> = rows.iter().flat_map(|&a| cols.iter().map(move |&b| (a, b))).map(|(a, b)| wt(a, b)).collect();
    let (mut u, mut v, mut p, mut way) = (vec![0i64; nr + 1], vec![0i64; nc + 1], vec![0usize; nc + 1], vec![0usize; nc + 1]);
    let mut minv = vec![0i64; nc + 1];
    let mut used = vec![false; nc + 1];
    for i in 1..=nr {
        p[0] = i;
        let mut j0 = 0usize;
        minv.iter_mut().for_each(|x| *x = i64::MAX);
        used.iter_mut().for_each(|x| *x = false);
        loop {
            used[j0] = true;
            let i0 = p[j0];
            let (mut d, mut j1) = (i64::MAX, 0usize);
            for j in 1..=nc {
                if !used[j] {
                    let cur = -wm[(i0 - 1) * nc + j - 1] - u[i0] - v[j];
                    if cur < minv[j] { minv[j] = cur; way[j] = j0; }
                    if minv[j] < d { d = minv[j]; j1 = j; }
                }
            }
            for j in 0..=nc {
                if used[j] { u[p[j]] += d; v[j] -= d; } else { minv[j] -= d; }
            }
            j0 = j1;
            if p[j0] == 0 { break; }
        }
        loop {
            let j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
            if j0 == 0 { break; }
        }
    }
    let mut pairs = Vec::new();
    let mut total = 0i64;
    for j in 1..=nc {
        if p[j] != 0 {
            let (a, b) = (rows[p[j] - 1], cols[j - 1]);
            let w = wt(a, b);
            if w > 0 { total += w; pairs.push(if a < b { (a, b) } else { (b, a) }); }
        }
    }
    pairs.sort_unstable();
    (total, pairs)
}

// ---------------- H 型假结：全序列切成四段，第一、三段之间和第二、四段之间各一条无交叉配对，两条互相交叉 ----------------

#[derive(Default)]
pub struct HpkBuf {
    h: Vec<i16>,
    g: Vec<i16>,
}

/// 保留空函数：H 型假结已无提速分支（布局改动对两种写法通用）
pub fn set_hpk_fast(_v: bool) {}

pub fn hpk(c: &Ctx, buf: &mut HpkBuf) -> (i32, Vec<(usize, usize)>) {
    let n = c.n;
    let okm: Vec<i16> = (0..n * n).map(|q| { let (k, j) = (q / n, q % n); if k < j && weight(c, 0, k, j) > NEG / 2 && j - k > H { 1 } else { 0 } }).collect();
    let ok = |k: usize, j: usize| -> i16 { okm[k * n + j] };
    let n1 = n + 1;
    // h(x,y,z)：[x,y) 与 [z,n) 之间的最优无交叉配对数（x<=y<=z）。按 [x][z][y] 存放，y 方向连续。
    buf.h.clear();
    buf.h.resize(n1 * n1 * n1, 0);
    let hi = |x: usize, y: usize, z: usize| (x * n1 + z) * n1 + y;
    for x in 0..=n {
        for z in (0..n).rev() {
            for y in x + 1..=z {
                let mut v = buf.h[hi(x, y - 1, z)].max(buf.h[hi(x, y, z + 1)]);
                let w = ok(y - 1, z);
                if w > 0 { v = v.max(buf.h[hi(x, y - 1, z + 1)] + w); }
                buf.h[hi(x, y, z)] = v;
            }
        }
    }
    // g_z(x,y)：[0,x) 与 [y,z) 之间的最优无交叉配对数，逐个 z 计算并与 h 组合；同分取 (z,x,y) 最先出现者
    buf.g.clear();
    buf.g.resize(n1 * n1, 0);
    let (mut best, mut arg) = (0i32, (0usize, 0usize, 0usize));
    for z in 1..=n {
        let gi = |x: usize, y: usize| x * n1 + y;
        for x in 0..=z {
            for y in (x..=z).rev() {
                let v = if x == 0 || y == z { 0 } else {
                    let mut v = buf.g[gi(x - 1, y)].max(buf.g[gi(x, y + 1)]);
                    let w = ok(x - 1, y);
                    if w > 0 { v = v.max(buf.g[gi(x - 1, y + 1)] + w); }
                    v
                };
                buf.g[gi(x, y)] = v;
            }
        }
        for x in 1..z {
            for y in x + 1..z {
                let (a, b) = (buf.g[gi(x, y)] as i32, buf.h[hi(x, y, z)] as i32);
                if a > 0 && b > 0 && a + b > best { best = a + b; arg = (x, y, z); }
            }
        }
    }
    let mut pairs = Vec::new();
    if best > 0 {
        let (x0, y0, z0) = arg;
        // 重算 g_{z0} 并倒推第一条茎
        let gi = |x: usize, y: usize| x * n1 + y;
        for x in 0..=z0 {
            for y in (x..=z0).rev() {
                let v = if x == 0 || y == z0 { 0 } else {
                    let mut v = buf.g[gi(x - 1, y)].max(buf.g[gi(x, y + 1)]);
                    let w = ok(x - 1, y);
                    if w > 0 { v = v.max(buf.g[gi(x - 1, y + 1)] + w); }
                    v
                };
                buf.g[gi(x, y)] = v;
            }
        }
        let (mut x, mut y) = (x0, y0);
        while x > 0 && y < z0 {
            let cur = buf.g[gi(x, y)];
            if cur == 0 { break; }
            if cur == buf.g[gi(x - 1, y)] { x -= 1; } else if cur == buf.g[gi(x, y + 1)] { y += 1; } else { pairs.push((x - 1, y)); x -= 1; y += 1; }
        }
        let (mut y, mut z) = (y0, z0);
        while y > x0 && z < n {
            let cur = buf.h[hi(x0, y, z)];
            if cur == 0 { break; }
            if cur == buf.h[hi(x0, y - 1, z)] { y -= 1; } else if cur == buf.h[hi(x0, y, z + 1)] { z += 1; } else { pairs.push((y - 1, z)); y -= 1; z += 1; }
        }
        pairs.sort_unstable();
    }
    (best, pairs)
}

// ---------------- 无交叉配对的统一入口 ----------------

pub struct Kern {
    pub base: Table,
    pub fast: FastTable<i32>,
    pub fast16: FastTable<i16>,
    pub stack: StackBuf,
    pub prob: ProbBuf,
    pub hpk: HpkBuf,
    pub partner: Vec<i32>,
}

impl Kern {
    pub fn new() -> Self {
        Kern { base: Table::new(), fast: FastTable::default(), fast16: FastTable::default(), stack: StackBuf::default(), prob: ProbBuf::default(), hpk: HpkBuf::default(), partner: Vec::new() }
    }
}

/// 返回 (归一化得分, 配对列表[按左端递增])
pub fn nested(mode: Mode, k: &mut Kern, c: &Ctx, rule: usize) -> (f32, Vec<(usize, usize)>) {
    let n = c.n;
    k.partner.clear();
    k.partner.resize(n + 1, -1);
    let score = if rule == 11 {
        let all: Vec<usize>;
        let pos: &[usize] = if mode == Mode::Base { all = (0..n).collect(); &all } else { &c.act };
        stack_fold(c, pos, &mut k.stack, &mut k.partner)
    } else {
        let h = rule_h(rule);
        macro_rules! run {
            ($r:expr) => {{
                let w = |a: usize, b: usize| weight(c, $r, a, b);
                run_nested(mode, k, c, $r, h, &w)
            }};
        }
        match rule { 0 => run!(0), 1 => run!(1), 2 => run!(2), 3 => run!(3), 4 => run!(4), 5 => run!(5), 6 => run!(6), 7 => run!(7), 8 => run!(8), 9 => run!(9), _ => run!(10) }
    };
    let ub = rule_ub(c, rule);
    let sn = if ub > 0.0 { (score as f64 / ub) as f32 } else { f32::NAN };
    let pairs = (0..n).filter(|&p| k.partner[p] > p as i32).map(|p| (p, k.partner[p] as usize)).collect();
    (sn, pairs)
}

#[inline(always)]
fn run_nested<W: Fn(usize, usize) -> i32>(mode: Mode, k: &mut Kern, c: &Ctx, rule: usize, h: usize, w: &W) -> i32 {
    let n = c.n;
    {
        let w = |a: usize, b: usize| w(a, b);
        match mode {
            Mode::Base => {
                let s = k.base.solve(n, h, &c.active, &w);
                k.base.traceback(n, h, &w, true, &mut k.partner[..n]);
                s
            }
            Mode::Fast if small_score_rule(rule) => {
                let s = k.fast16.solve(&c.act, &c.side, h, &w);
                k.fast16.traceback(&c.act, &w, &mut k.partner);
                s
            }
            Mode::Fast => {
                let s = k.fast.solve(&c.act, &c.side, h, &w);
                k.fast.traceback(&c.act, &w, &mut k.partner);
                s
            }
        }
    }
}

/// 二层折叠：第一层是计数规则的结果；在第一层没配上的字母上再做一次无交叉配对，两层之间允许交叉。
pub fn two_layer(mode: Mode, k: &mut Kern, c: &Ctx, layer1: &[(usize, usize)]) -> (f32, Vec<(usize, usize)>) {
    let n = c.n;
    let mut used = vec![false; n];
    for &(a, b) in layer1 { used[a] = true; used[b] = true; }
    let rest: Vec<usize> = (0..n).filter(|&i| c.active[i] && !used[i]).collect();
    let m = rest.len();
    let mut pairs: Vec<(usize, usize)> = layer1.to_vec();
    if m >= 2 {
        let sub_side: Vec<i8> = rest.iter().map(|&i| c.side[i]).collect();
        let mut part = vec![-1i32; m + 1];
        match mode {
            Mode::Base => {
                let w = |a: usize, b: usize| if sub_side[a] as i32 * sub_side[b] as i32 == -1 && rest[b] - rest[a] > H { 1 } else { NEG };
                let all = vec![true; m];
                k.base.solve(m, 0, &all, &w);
                k.base.traceback(m, 0, &w, true, &mut part[..m]);
            }
            Mode::Fast => {
                // 提速表按原下标判断间隔，这里直接传原下标
                let w = |a: usize, b: usize| if c.side[a] as i32 * c.side[b] as i32 == -1 { 1 } else { NEG };
                k.fast16.solve(&rest, &c.side, H, &w);
                let mut full = vec![-1i32; n + 1];
                k.fast16.traceback(&rest, &w, &mut full);
                for (q, &o) in rest.iter().enumerate() {
                    if full[o] >= 0 { part[q] = rest.iter().position(|&x| x == full[o] as usize).unwrap() as i32; }
                }
            }
        }
        for q in 0..m {
            if part[q] > q as i32 { pairs.push((rest[q], rest[part[q] as usize])); }
        }
    }
    pairs.sort_unstable();
    let bound = c.n_buy.min(c.n_sell) as f64;
    let sn = if bound > 0.0 { (pairs.len() as f64 / bound) as f32 } else { f32::NAN };
    (sn, pairs)
}
