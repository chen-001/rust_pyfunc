//! 第三版计算核。每个核都有“基准写法”和“提速写法”两条路，结果必须逐字节相同：
//!  - 无交叉配对：基准在原序列上算；提速版先去掉中性字母（它们不能配对，不改变任何区间的最优值），
//!    再把每一行按买、卖字母分开连续存放，内层只扫对侧字母，并做 4 路寄存器分块消除重复读写。
//!  - 连续嵌套奖励：基准走原始行主序表；提速版在非中性位置上按列存放 `i16` 并做对侧过滤与 4 路分块。
//!  - 配对概率：精确计算所有配对方案的加权和（不抽样）。累加顺序在定义里写死为“按对侧字母分 4 路累加”，
//!    基准逐个位置判断，提速版用对侧字母的连续数组，所以数值逐字节相同。
use crate::fold::{Table, NEG};
use crate::v3_letters::Seq;
use std::cell::RefCell;

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
    pub act_bpos: Vec<usize>,
    pub act_spos: Vec<usize>,
    pub act_kmax_h3: Vec<i64>,
    pub act_kmax_h10: Vec<i64>,
    pub n_buy: i32,
    pub n_sell: i32,
    pub sum_vb: i64,
    pub sum_vs: i64,
    pub lshort: usize,
}

fn build_kmax(pos: &[usize], h: usize) -> Vec<i64> {
    let m = pos.len();
    let mut kmax = Vec::with_capacity(m);
    let mut p = 0usize;
    for j in 0..m {
        while p < m && pos[p] + h + 1 <= pos[j] {
            p += 1;
        }
        kmax.push(p as i64 - 1);
    }
    kmax
}

pub fn make_ctx(s: &Seq, side2: Option<&[i8]>) -> Ctx {
    let n = s.side.len();
    let side: Vec<i8> = match side2 {
        Some(b) => (0..n).map(|i| if b[i] != 0 { s.side[i] } else { 0 }).collect(),
        None => s.side.clone(),
    };
    let (mut sumv, mut cnt) = (0f64, 0f64);
    for i in 0..n {
        if side[i] != 0 {
            sumv += s.vol[i] as f64;
            cnt += 1.0;
        }
    }
    let mean = if cnt > 0.0 && sumv > 0.0 { sumv / cnt } else { 1.0 };
    let vint: Vec<i32> = s.vol.iter().map(|&v| ((v as f64 / mean * 100.0) as i32).clamp(1, 3000)).collect();
    let tau = (n as f64 / 8.0).max(1.0);
    let decay = (0..=n).map(|d| ((1000.0 * (-(d as f64) / tau).exp()).round() as i32).max(1)).collect();
    let (mut n_buy, mut n_sell, mut sum_vb, mut sum_vs) = (0, 0, 0i64, 0i64);
    for i in 0..n {
        if side[i] > 0 {
            n_buy += 1;
            sum_vb += vint[i] as i64;
        } else if side[i] < 0 {
            n_sell += 1;
            sum_vs += vint[i] as i64;
        }
    }
    let active: Vec<bool> = side.iter().map(|&x| x != 0).collect();
    let act: Vec<usize> = (0..n).filter(|&i| active[i]).collect();
    let mut act_bpos = Vec::with_capacity(n_buy as usize);
    let mut act_spos = Vec::with_capacity(n_sell as usize);
    for (x, &o) in act.iter().enumerate() {
        if side[o] > 0 {
            act_bpos.push(x);
        } else {
            act_spos.push(x);
        }
    }
    let act_kmax_h3 = build_kmax(&act, H);
    let act_kmax_h10 = build_kmax(&act, 10);
    Ctx {
        n,
        side,
        side2: side2.map(|b| b.to_vec()),
        strength: s.strength.clone(),
        price: s.price.clone(),
        vol: s.vol.clone(),
        vint,
        decay,
        active,
        act,
        act_bpos,
        act_spos,
        act_kmax_h3,
        act_kmax_h10,
        n_buy,
        n_sell,
        sum_vb,
        sum_vs,
        lshort: (n / 8).max(H + 1),
    }
}

pub fn rule_h(rule: usize) -> usize {
    if rule == 9 { 10 } else { H }
}

/// 配对权重（原序列下标，k<j）；不可配对返回 NEG。
#[inline(always)]
pub fn weight(c: &Ctx, rule: usize, k: usize, j: usize) -> i32 {
    let (sk, sj) = (c.side[k], c.side[j]);
    if sk as i32 * sj as i32 != -1 {
        return NEG;
    }
    if let Some(s2) = &c.side2 {
        if s2[k] as i32 * s2[j] as i32 != -1 {
            return NEG;
        }
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
            if s > b {
                ((((s / b) as f64 - 1.0) * 1e4).round() as i32).clamp(1, 100_000)
            } else {
                NEG
            }
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
    #[inline]
    fn from_w(v: i32) -> i32 {
        if v > NEG / 2 { v } else { NEG }
    }
    #[inline]
    fn to_i32(self) -> i32 {
        self
    }
}
impl Sc for i16 {
    const NEG: i16 = i16::MIN / 2;
    #[inline]
    fn from_w(v: i32) -> i16 {
        if v > NEG / 2 { v as i16 } else { i16::MIN / 2 }
    }
    #[inline]
    fn to_i32(self) -> i32 {
        self as i32
    }
}

/// 32 元素对齐的 4 路寄存器分块更新：列长按 32 对齐、下三角填 `T::NEG`（使 `col[i] + z_q < 0 <= best[i]` 对 `i > k_q` 恒不胜出），
/// 消除全部尾部碎循环与标量余数，直接编译为单条无分支 32 路 `i16` / 双 16 路 `i32` 向量指令块。
#[inline(always)]
fn apply_kz_4way<T: Sc>(c: &[T], st: usize, best: &mut [T], kz: &[(usize, T)]) {
    let mut idx = 0usize;
    while idx + 4 <= kz.len() {
        let (k0, z0) = kz[idx];
        let (k1, z1) = kz[idx + 1];
        let (k2, z2) = kz[idx + 2];
        let (k3, z3) = kz[idx + 3];
        let len = (k3 + 1 + 31) & !31;
        let col0 = &c[k0 * st..k0 * st + len];
        let col1 = &c[k1 * st..k1 * st + len];
        let col2 = &c[k2 * st..k2 * st + len];
        let col3 = &c[k3 * st..k3 * st + len];
        for ((b_blk, c0), ((c1, c2), c3)) in best[..len]
            .chunks_exact_mut(32)
            .zip(col0.chunks_exact(32))
            .zip(col1.chunks_exact(32).zip(col2.chunks_exact(32)).zip(col3.chunks_exact(32)))
        {
            for l in 0..32 {
                let m01 = (c0[l] + z0).max(c1[l] + z1);
                let m23 = (c2[l] + z2).max(c3[l] + z3);
                b_blk[l] = b_blk[l].max(m01.max(m23));
            }
        }
        idx += 4;
    }
    while idx < kz.len() {
        let (k, zt) = kz[idx];
        let len = (k + 1 + 31) & !31;
        let col = &c[k * st..k * st + len];
        for (b_blk, ck) in best[..len].chunks_exact_mut(32).zip(col.chunks_exact(32)) {
            for l in 0..32 {
                b_blk[l] = b_blk[l].max(ck[l] + zt);
            }
        }
        idx += 1;
    }
}

#[allow(dead_code)]
#[derive(Default)]
pub struct FastTable<T: Sc> {
    st: usize,
    /// 按列存放（st 按 32 对齐）：c[x*st + i] = 区间 [i, x) 的最优得分，i > x 填 T::NEG 供定长向量块无分支读取
    c: Vec<T>,
    best: Vec<T>,
    kmax: Vec<i64>,
    bpos: Vec<usize>,
    spos: Vec<usize>,
    valid_kz: Vec<(usize, T)>,
    stack: Vec<(usize, usize)>,
}

#[allow(dead_code)]
impl<T: Sc> FastTable<T> {
    /// 通用入口（供 two_layer 等自定义子序列调用）
    pub fn solve<W: Fn(usize, usize) -> i32>(&mut self, act: &[usize], side: &[i8], h: usize, w: &W) -> i32 {
        let m = act.len();
        self.bpos.clear();
        self.spos.clear();
        for (x, &o) in act.iter().enumerate() {
            if side[o] > 0 {
                self.bpos.push(x);
            } else {
                self.spos.push(x);
            }
        }
        self.kmax.clear();
        let mut p = 0usize;
        for j in 0..m {
            while p < m && act[p] + h + 1 <= act[j] {
                p += 1;
            }
            self.kmax.push(p as i64 - 1);
        }
        let st = (m + 1 + 31) & !31;
        self.st = st;
        let zero = T::default();
        let total = (m + 1) * st;
        if self.c.len() < total {
            self.c.resize(total, T::NEG);
        }
        if self.best.len() < st {
            self.best.resize(st, T::NEG);
        }
        let (c, best, kz) = (&mut self.c[..], &mut self.best[..], &mut self.valid_kz);
        c[0] = zero;
        c[1..st].fill(T::NEG);
        for j in 0..m {
            best[..j + 1].copy_from_slice(&c[j * st..j * st + j + 1]);
            let km = self.kmax[j];
            if km >= 0 {
                let km = km as usize;
                let aj = act[j];
                let opp = if side[aj] > 0 { &self.spos } else { &self.bpos };
                kz.clear();
                let cj = &c[j * st..j * st + j + 1];
                for &k in opp {
                    if k > km {
                        break;
                    }
                    let v = w(act[k], aj);
                    if v > NEG / 2 {
                        kz.push((k, T::from_w(v) + cj[k + 1]));
                    }
                }
                apply_kz_4way(c, st, best, kz);
            }
            let x = j + 1;
            c[x * st..x * st + x].copy_from_slice(&best[..x]);
            c[x * st + x] = zero;
            c[x * st + x + 1..x * st + st].fill(T::NEG);
        }
        c[m * st].to_i32()
    }

    /// 复用 Ctx 已预计算的 `act_bpos` / `act_spos` / `kmax`，并对常见无 side2 规则消掉逐对分支
    #[inline(always)]
    pub fn solve_ctx<W: Fn(usize, usize) -> i32>(&mut self, ctx: &Ctx, rule: usize, kmax: &[i64], w: &W) -> i32 {
        let act = &ctx.act;
        let side = &ctx.side;
        let m = act.len();
        let st = (m + 1 + 31) & !31;
        self.st = st;
        let zero = T::default();
        let total = (m + 1) * st;
        if self.c.len() < total {
            self.c.resize(total, T::NEG);
        }
        if self.best.len() < st {
            self.best.resize(st, T::NEG);
        }
        let has_s2 = ctx.side2.is_some();
        let (c, best, kz) = (&mut self.c[..], &mut self.best[..], &mut self.valid_kz);
        c[0] = zero;
        c[1..st].fill(T::NEG);
        for j in 0..m {
            best[..j + 1].copy_from_slice(&c[j * st..j * st + j + 1]);
            let km = kmax[j];
            if km >= 0 {
                let km = km as usize;
                let aj = act[j];
                let buy_j = side[aj] > 0;
                // 规则 6（买在前）在 j 为买时无合法配对；规则 7（卖在前）在 j 为卖时无合法配对
                if !((rule == 6 && buy_j) || (rule == 7 && !buy_j)) {
                    let opp = if buy_j { &ctx.act_spos } else { &ctx.act_bpos };
                    let hi = opp.partition_point(|&k| k <= km);
                    let opp_slice = &opp[..hi];
                    kz.clear();
                    let cj = &c[j * st..j * st + j + 1];
                    if !has_s2 {
                        match rule {
                            0 | 6 | 7 | 9 => {
                                let one = T::from_w(1);
                                for &k in opp_slice {
                                    kz.push((k, one + cj[k + 1]));
                                }
                            }
                            8 => {
                                let one = T::from_w(1);
                                let lo = opp_slice.partition_point(|&k| act[k] + ctx.lshort < aj);
                                for &k in &opp_slice[lo..] {
                                    kz.push((k, one + cj[k + 1]));
                                }
                            }
                            1 => {
                                let dec = &ctx.decay;
                                for &k in opp_slice {
                                    kz.push((k, T::from_w(dec[aj - act[k]]) + cj[k + 1]));
                                }
                            }
                            2 => {
                                let vint = &ctx.vint;
                                let vj = vint[aj];
                                for &k in opp_slice {
                                    kz.push((k, T::from_w(vint[act[k]].min(vj)) + cj[k + 1]));
                                }
                            }
                            3 => {
                                let stg = &ctx.strength;
                                let sj = stg[aj] as i32 - 1;
                                for &k in opp_slice {
                                    kz.push((k, T::from_w(stg[act[k]] as i32 + sj) + cj[k + 1]));
                                }
                            }
                            _ => {
                                for &k in opp_slice {
                                    let v = w(act[k], aj);
                                    if v > NEG / 2 {
                                        kz.push((k, T::from_w(v) + cj[k + 1]));
                                    }
                                }
                            }
                        }
                    } else {
                        for &k in opp_slice {
                            let v = w(act[k], aj);
                            if v > NEG / 2 {
                                kz.push((k, T::from_w(v) + cj[k + 1]));
                            }
                        }
                    }
                    apply_kz_4way(c, st, best, kz);
                }
            }
            let x = j + 1;
            c[x * st..x * st + x].copy_from_slice(&best[..x]);
            c[x * st + x] = zero;
            c[x * st + x + 1..x * st + st].fill(T::NEG);
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
                if cur == at(i, j) {
                    x -= 1;
                    continue;
                }
                let km = self.kmax[j] as usize;
                let mut found = usize::MAX;
                for k in (i..=km).rev() {
                    let v = w(act[k], act[j]);
                    if v > NEG / 2 && at(i, k) + v + at(k + 1, j) == cur {
                        found = k;
                        break;
                    }
                }
                partner[act[found]] = act[j] as i32;
                partner[act[j]] = act[found] as i32;
                self.stack.push((found + 1, j));
                x = found;
            }
        }
    }

    #[inline(always)]
    pub fn traceback_ctx<W: Fn(usize, usize) -> i32>(
        &mut self,
        ctx: &Ctx,
        kmax: &[i64],
        w: &W,
        partner: &mut [i32],
    ) {
        let act = &ctx.act;
        let side = &ctx.side;
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
                if cur == at(i, j) {
                    x -= 1;
                    continue;
                }
                let km = kmax[j] as usize;
                let aj = act[j];
                let opp = if side[aj] > 0 { &ctx.act_spos } else { &ctx.act_bpos };
                let hi = opp.partition_point(|&k| k <= km);
                let mut found = usize::MAX;
                for &k in opp[..hi].iter().rev() {
                    if k < i {
                        break;
                    }
                    let v = w(act[k], aj);
                    if v > NEG / 2 && at(i, k) + v + at(k + 1, j) == cur {
                        found = k;
                        break;
                    }
                }
                partner[act[found]] = aj as i32;
                partner[aj] = act[found] as i32;
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

// ---------------- 连续嵌套奖励（基准走原行主序，提速走列主序 i16 + 4 路分块） ----------------

#[derive(Default)]
pub struct StackBuf {
    r: Vec<i32>,
    s: Vec<i32>,
    z: Vec<i32>,
    kmax: Vec<i64>,
    r16: Vec<i16>,
    s16: Vec<i16>,
    best16: Vec<i16>,
    kz16: Vec<(usize, i16)>,
}

/// 基准写法：每对基础 2 分，若紧贴着内层也配对（原序列上相邻）再加 1 分。
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
        while p < m && pos[p] + H + 1 <= pos[j] {
            p += 1;
        }
        buf.kmax.push(p as i64 - 1);
    }
    let (r, s, z) = (&mut buf.r[..], &mut buf.s[..], &mut buf.z[..]);
    let stacked = |s: &[i32], k: usize, j: usize| -> i32 {
        if k + 1 < j && pos[k + 1] == pos[k] + 1 && pos[j - 1] + 1 == pos[j] {
            s[(k + 1) * m + j - 1]
        } else {
            NEG
        }
    };
    for j in 0..m {
        let km = buf.kmax[j];
        if km < 0 {
            for i in 0..=j {
                r[i * st + j + 1] = r[i * st + j];
            }
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
    enum St {
        Seg(usize, usize),
        Pair(usize, usize),
    }
    let mut stack = vec![St::Seg(0, m)];
    while let Some(stt) = stack.pop() {
        match stt {
            St::Seg(i, x0) => {
                let mut x = x0;
                while x > i {
                    let j = x - 1;
                    let cur = r[i * st + x];
                    if cur == r[i * st + j] {
                        x -= 1;
                        continue;
                    }
                    let km = buf.kmax[j] as usize;
                    let mut found = usize::MAX;
                    for k in (i..=km).rev() {
                        let sv = s[k * m + j];
                        if sv > NEG / 2 && r[i * st + k] + sv == cur {
                            found = k;
                            break;
                        }
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

/// 提速版连续嵌套奖励：列主序 `i16` + 仅扫对侧位置 + 4 路分块，得分与倒推配对跟基准逐位相同。
pub fn stack_fold_fast(c: &Ctx, buf: &mut StackBuf, partner: &mut [i32]) -> i32 {
    let pos = &c.act;
    let m = pos.len();
    if m == 0 {
        return 0;
    }
    let st = (m + 1 + 31) & !31;
    let total = (m + 1) * st;
    if buf.r16.len() < total {
        buf.r16.resize(total, i16::NEG);
    }
    if buf.s16.len() < m * m {
        buf.s16.resize(m * m, i16::NEG);
    }
    buf.s16[..m * m].fill(i16::NEG);
    if buf.best16.len() < st {
        buf.best16.resize(st, i16::NEG);
    }
    let has_s2 = c.side2.is_some();
    let kmax = &c.act_kmax_h3;
    let (rc, s, best, kz) = (&mut buf.r16[..], &mut buf.s16[..], &mut buf.best16[..], &mut buf.kz16);
    rc[0] = 0;
    rc[1..st].fill(i16::NEG);
    let stacked = |s: &[i16], k: usize, j: usize| -> i16 {
        if k + 1 < j && pos[k + 1] == pos[k] + 1 && pos[j - 1] + 1 == pos[j] {
            s[(j - 1) * m + (k + 1)]
        } else {
            i16::NEG
        }
    };
    for j in 0..m {
        best[..j + 1].copy_from_slice(&rc[j * st..j * st + j + 1]);
        let km = kmax[j];
        if km >= 0 {
            let km = km as usize;
            let aj = pos[j];
            let opp = if c.side[aj] > 0 { &c.act_spos } else { &c.act_bpos };
            let hi = opp.partition_point(|&k| k <= km);
            kz.clear();
            for &k in &opp[..hi] {
                if has_s2 && weight(c, 11, pos[k], aj) <= NEG / 2 {
                    continue;
                }
                let inner = rc[j * st + k + 1];
                let sk = stacked(s, k, j);
                let b = if sk > i16::NEG / 2 { inner.max(sk + 1) } else { inner };
                let sv = 2i16 + b;
                s[j * m + k] = sv;
                kz.push((k, sv));
            }
            apply_kz_4way(rc, st, best, kz);
        }
        let x = j + 1;
        rc[x * st..x * st + x].copy_from_slice(&best[..x]);
        rc[x * st + x] = 0;
        rc[x * st + x + 1..x * st + st].fill(i16::NEG);
    }

    enum St {
        Seg(usize, usize),
        Pair(usize, usize),
    }
    let mut stack = vec![St::Seg(0, m)];
    while let Some(stt) = stack.pop() {
        match stt {
            St::Seg(i, x0) => {
                let mut x = x0;
                while x > i {
                    let j = x - 1;
                    let cur = rc[x * st + i];
                    if cur == rc[j * st + i] {
                        x -= 1;
                        continue;
                    }
                    let km = kmax[j] as usize;
                    let aj = pos[j];
                    let opp = if c.side[aj] > 0 { &c.act_spos } else { &c.act_bpos };
                    let hi = opp.partition_point(|&k| k <= km);
                    let mut found = usize::MAX;
                    for &k in opp[..hi].iter().rev() {
                        if k < i {
                            break;
                        }
                        let sv = s[j * m + k];
                        if sv > i16::NEG / 2 && rc[k * st + i] + sv == cur {
                            found = k;
                            break;
                        }
                    }
                    stack.push(St::Pair(found, j));
                    x = found;
                }
            }
            St::Pair(k, j) => {
                partner[pos[k]] = pos[j] as i32;
                partner[pos[j]] = pos[k] as i32;
                let sk = stacked(s, k, j);
                if sk > i16::NEG / 2 && s[j * m + k] == 2 + sk + 1 {
                    stack.push(St::Pair(k + 1, j - 1));
                } else {
                    stack.push(St::Seg(k + 1, j));
                }
            }
        }
    }
    rc[m * st] as i32
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
    bpos: Vec<usize>,
    spos: Vec<usize>,
    cb: Vec<usize>,
    cs: Vec<usize>,
    slot: Vec<usize>,
    zo: Vec<f64>,
    acc_opp: Vec<f64>,
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
            if p > buf_max[q] {
                buf_max[q] = p;
            }
        }
        acc.sp += p;
        acc.sspan += p * (j - k) as f64;
        if side[k] > 0 {
            acc.sbf += p;
        }
        acc.scenter += p * (k + j) as f64 * 0.5;
    }
}

fn prob_finish(c: &Ctx, z: f64, buf: &ProbBuf, acc: &ProbAcc) -> [f32; 8] {
    let n = c.n;
    let mut out = [f32::NAN; 8];
    if !(z.is_finite() && z > 0.0) {
        return out;
    }
    let bound = c.n_buy.min(c.n_sell) as f64;
    let (mut hs, mut conf, mut unp, mut na) = (0f64, 0f64, 0f64, 0f64);
    for i in 0..n {
        if !c.active[i] {
            continue;
        }
        let u = (1.0 - buf.pos_sum[i]).max(0.0);
        let mut h = buf.pos_ent[i];
        if u > 0.0 {
            h -= u * u.ln();
        }
        hs += h;
        if buf.pos_max[i] > 0.5 {
            conf += 1.0;
        }
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
                    if ok(k, j) {
                        zz[k] = BW * r[(k + 1) * st + j] * ISIG2;
                    }
                }
            }
            for i in 0..=j {
                let mut s = r[i * st + j] * ISIG;
                if elig && i <= kmax {
                    let mut acc = [0f64; 4];
                    let mut t = 0usize;
                    for k in i..=kmax {
                        if ok(k, j) {
                            acc[t & 3] += r[i * st + k] * zz[k];
                            t += 1;
                        }
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
                    if ok(k, j) {
                        zz[k] = BW * r[(k + 1) * st + j] * ISIG2;
                        acc[k] = 0.0;
                    }
                }
            }
            for i in 0..=j {
                let o = a[i * st + x] + b[i * st + x];
                if o == 0.0 {
                    continue;
                }
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
/// 区间表按列存放（rc[x*st+i] = 区间 [i,x)），全部临时数组复用在 ProbBuf 内，且 i > opp[tmax-1] 直接走连续向量缩放。
pub fn prob_fast(c: &Ctx, buf: &mut ProbBuf) -> [f32; 8] {
    let n = c.n;
    let st = n + 1;
    let side = &c.side;
    prob_reset(c, buf);
    buf.bpos.clear();
    buf.spos.clear();
    buf.cb.clear();
    buf.cs.clear();
    buf.slot.clear();
    buf.slot.resize(n, usize::MAX);
    for x in 0..n {
        buf.cb.push(buf.bpos.len());
        buf.cs.push(buf.spos.len());
        if side[x] > 0 {
            buf.slot[x] = buf.bpos.len();
            buf.bpos.push(x);
        } else if side[x] < 0 {
            buf.slot[x] = buf.spos.len();
            buf.spos.push(x);
        }
    }
    buf.cb.push(buf.bpos.len());
    buf.cs.push(buf.spos.len());
    let (nb, ns) = (buf.bpos.len().max(1), buf.spos.len().max(1));
    let nmax = nb.max(ns);
    if buf.r.len() < st * st {
        buf.r.resize(st * st, 1.0);
    }
    if buf.rb.len() < st * nb {
        buf.rb.resize(st * nb, 1.0);
    }
    if buf.rs.len() < st * ns {
        buf.rs.resize(st * ns, 1.0);
    }
    if buf.zo.len() < nmax {
        buf.zo.resize(nmax, 0.0);
    }
    {
        let (rc, rb, rs) = (&mut buf.r[..], &mut buf.rb[..], &mut buf.rs[..]);
        let (bpos, spos, cb, cs, slot, zo) = (&buf.bpos[..], &buf.spos[..], &buf.cb[..], &buf.cs[..], &buf.slot[..], &mut buf.zo[..]);
        rc[0] = 1.0;
        if n > 0 && side[0] != 0 {
            let sl = slot[0];
            if side[0] > 0 {
                rb[sl] = 1.0;
            } else {
                rs[sl] = 1.0;
            }
        }
        for j in 0..n {
            let elig = side[j] != 0 && j >= H + 1;
            let buy_j = side[j] > 0;
            let (opp, cnt, nopp) = if buy_j { (spos, cs, ns) } else { (bpos, cb, nb) };
            let tmax = if elig { cnt[j - H] } else { 0 };
            for t in 0..tmax {
                zo[t] = BW * rc[j * st + opp[t] + 1] * ISIG2;
            }
            let i_lim = if tmax > 0 { opp[tmax - 1] + 1 } else { 0 };
            let ropp: &[f64] = if buy_j { &*rs } else { &*rb };
            for i in 0..i_lim {
                let mut s = rc[j * st + i] * ISIG;
                let t0 = cnt[i];
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
                rc[(j + 1) * st + i] = s;
            }
            for i in i_lim..(j + 1) {
                rc[(j + 1) * st + i] = rc[j * st + i] * ISIG;
            }
            rc[(j + 1) * st + j + 1] = 1.0;
            let x = j + 1;
            if x < n && side[x] != 0 {
                let sl = slot[x];
                if side[x] > 0 {
                    for i in 0..x {
                        rb[i * nb + sl] = rc[x * st + i];
                    }
                    rb[x * nb + sl] = 1.0;
                } else {
                    for i in 0..x {
                        rs[i * ns + sl] = rc[x * st + i];
                    }
                    rs[x * ns + sl] = 1.0;
                }
            }
        }
    }
    let z = buf.r[n * st];
    let mut pa = ProbAcc { sp: 0.0, sspan: 0.0, sbf: 0.0, scenter: 0.0 };
    if z.is_finite() && z > 0.0 {
        if buf.a.len() < st * st {
            buf.a.resize(st * st, 0.0);
        }
        for x in 0..st {
            buf.a[x * st..x * st + x + 1].fill(0.0);
        }
        if buf.bb.len() < st * nb {
            buf.bb.resize(st * nb, 0.0);
        }
        buf.bb[..st * nb].fill(0.0);
        if buf.bs.len() < st * ns {
            buf.bs.resize(st * ns, 0.0);
        }
        buf.bs[..st * ns].fill(0.0);
        buf.a[n * st] = 1.0;
        if buf.acc_opp.len() < nmax {
            buf.acc_opp.resize(nmax, 0.0);
        }
        let (rc, rb, rs, ac, bb, bs) = (
            &buf.r[..],
            &buf.rb[..],
            &buf.rs[..],
            &mut buf.a[..],
            &mut buf.bb[..],
            &mut buf.bs[..],
        );
        let (bpos, spos, cb, cs, slot, zo, acc) = (
            &buf.bpos[..],
            &buf.spos[..],
            &buf.cb[..],
            &buf.cs[..],
            &buf.slot[..],
            &mut buf.zo[..],
            &mut buf.acc_opp[..],
        );
        for j in (0..n).rev() {
            let x = j + 1;
            let elig = side[j] != 0 && j >= H + 1;
            let buy_j = side[j] > 0;
            let (opp, cnt, nopp) = if buy_j { (spos, cs, ns) } else { (bpos, cb, nb) };
            let tmax = if elig { cnt[j - H] } else { 0 };
            for t in 0..tmax {
                zo[t] = BW * rc[j * st + opp[t] + 1] * ISIG2;
                acc[t] = 0.0;
            }
            let i_lim = if tmax > 0 { opp[tmax - 1] + 1 } else { 0 };
            let xs = if x < n { side[x] } else { 0 };
            let sl_x = if xs != 0 { slot[x] } else { 0 };
            {
                let (bopp, bsame, nsame, ropp): (&mut [f64], &[f64], usize, &[f64]) = if buy_j {
                    (&mut *bs, &*bb, nb, rs)
                } else {
                    (&mut *bb, &*bs, ns, rb)
                };
                for i in 0..i_lim {
                    let o = if xs > 0 {
                        ac[x * st + i] + if buy_j { bsame[i * nsame + sl_x] } else { bopp[i * nopp + sl_x] }
                    } else if xs < 0 {
                        ac[x * st + i] + if buy_j { bopp[i * nopp + sl_x] } else { bsame[i * nsame + sl_x] }
                    } else {
                        ac[x * st + i]
                    };
                    if o == 0.0 {
                        continue;
                    }
                    ac[j * st + i] += o * ISIG;
                    let t0 = cnt[i];
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
            if xs > 0 {
                for i in i_lim..x {
                    let o = ac[x * st + i] + bb[i * nb + sl_x];
                    if o != 0.0 {
                        ac[j * st + i] += o * ISIG;
                    }
                }
            } else if xs < 0 {
                for i in i_lim..x {
                    let o = ac[x * st + i] + bs[i * ns + sl_x];
                    if o != 0.0 {
                        ac[j * st + i] += o * ISIG;
                    }
                }
            } else {
                for i in i_lim..x {
                    let o = ac[x * st + i];
                    if o != 0.0 {
                        ac[j * st + i] += o * ISIG;
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

// ---------------- 允许交叉：买卖二分图最大权匹配（匈牙利算法，线程内复用 i32 缓冲） ----------------

#[derive(Default)]
struct HungBuf {
    bu: Vec<usize>,
    se: Vec<usize>,
    cost: Vec<i32>,
    u: Vec<i32>,
    v: Vec<i32>,
    p: Vec<usize>,
    way: Vec<usize>,
    minv: Vec<i32>,
    used: Vec<bool>,
    visited: Vec<usize>,
}

thread_local! {
    static HUNG_TLS: RefCell<HungBuf> = RefCell::new(HungBuf::default());
}

pub fn cross_match(c: &Ctx) -> (i64, Vec<(usize, usize)>) {
    let n = c.n;
    HUNG_TLS.with(|cell| {
        let b = &mut *cell.borrow_mut();
        b.bu.clear();
        b.se.clear();
        for i in 0..n {
            if c.side[i] > 0 {
                b.bu.push(i);
            } else if c.side[i] < 0 {
                b.se.push(i);
            }
        }
        if b.bu.is_empty() || b.se.is_empty() {
            return (0, Vec::new());
        }
        let bu_rows = b.bu.len() <= b.se.len();
        let (rows, cols) = if bu_rows { (&b.bu[..], &b.se[..]) } else { (&b.se[..], &b.bu[..]) };
        let (nr, nc) = (rows.len(), cols.len());
        b.cost.clear();
        b.cost.reserve(nr * nc);
        for &ra in rows {
            for &cb in cols {
                let d = ra.abs_diff(cb);
                b.cost.push(if d > H { -c.decay[d] } else { 0 });
            }
        }
        b.u.clear();
        b.u.resize(nr + 1, 0);
        b.v.clear();
        b.v.resize(nc + 1, 0);
        b.p.clear();
        b.p.resize(nc + 1, 0);
        b.way.clear();
        b.way.resize(nc + 1, 0);
        b.minv.clear();
        b.minv.resize(nc + 1, 0);
        b.used.clear();
        b.used.resize(nc + 1, false);

        const INF: i32 = i32::MAX / 2;
        for i in 1..=nr {
            b.p[0] = i;
            let mut j0 = 0usize;
            b.minv[1..=nc].fill(INF);
            b.used[..=nc].fill(false);
            b.visited.clear();
            loop {
                b.used[j0] = true;
                b.visited.push(j0);
                let i0 = b.p[j0];
                let u_i0 = b.u[i0];
                let crow = &b.cost[(i0 - 1) * nc..i0 * nc];
                let (mut d, mut j1) = (INF, 0usize);
                for j in 1..=nc {
                    if !b.used[j] {
                        let cur = crow[j - 1] - u_i0 - b.v[j];
                        if cur < b.minv[j] {
                            b.minv[j] = cur;
                            b.way[j] = j0;
                        }
                        if b.minv[j] < d {
                            d = b.minv[j];
                            j1 = j;
                        }
                    }
                }
                for &j in &b.visited {
                    b.u[b.p[j]] += d;
                    b.v[j] -= d;
                }
                for m in &mut b.minv[1..=nc] {
                    *m -= d;
                }
                j0 = j1;
                if b.p[j0] == 0 {
                    break;
                }
            }
            loop {
                let j1 = b.way[j0];
                b.p[j0] = b.p[j1];
                j0 = j1;
                if j0 == 0 {
                    break;
                }
            }
        }
        let mut pairs = Vec::with_capacity(nr);
        let mut total = 0i64;
        for j in 1..=nc {
            if b.p[j] != 0 {
                let (ra, cb) = (rows[b.p[j] - 1], cols[j - 1]);
                let dist = ra.abs_diff(cb);
                let w = if dist > H { c.decay[dist] as i64 } else { 0 };
                if w > 0 {
                    total += w;
                    pairs.push(if ra < cb { (ra, cb) } else { (cb, ra) });
                }
            }
        }
        pairs.sort_unstable();
        (total, pairs)
    })
}

// ---------------- H 型假结：全序列切成四段，第一、三段之间和第二、四段之间各一条无交叉配对，两条互相交叉 ----------------

#[derive(Default)]
pub struct HpkBuf {
    h: Vec<i16>,
    g: Vec<i16>,
    ok: Vec<u8>,
}

/// 保留空函数：H 型假结已无提速分支（32 路 i16 向量化布局对两种写法通用且逐位精确）
pub fn set_hpk_fast(_v: bool) {}

pub fn hpk(c: &Ctx, buf: &mut HpkBuf) -> (i32, Vec<(usize, usize)>) {
    let n = c.n;
    if n < 2 {
        return (0, Vec::new());
    }
    buf.ok.clear();
    buf.ok.resize(n * n, 0);
    for k in 0..n {
        if c.side[k] == 0 {
            continue;
        }
        let row = &mut buf.ok[k * n..(k + 1) * n];
        for j in (k + H + 1)..n {
            if weight(c, 0, k, j) > NEG / 2 {
                row[j] = 1;
            }
        }
    }
    let n1 = n + 1;
    let total_sz = n1 * n1 * n1;

    // 1) h(x,y,z)：[x,y) 与 [z,n) 之间的最优无交叉配对数（x<=y<=z）。
    //    按 [z][y][x] 存放，使最内层 x=0..y 无跨元素依赖，直接走 32 路 i16 向量取最大。
    buf.h.clear();
    buf.h.resize(total_sz, 0);
    let hi = |x: usize, y: usize, z: usize| (z * n1 + y) * n1 + x;
    {
        let h = &mut buf.h[..];
        let ok = &buf.ok[..];
        for z in (0..n).rev() {
            for y in 1..=z {
                let w = ok[(y - 1) * n + z];
                let z_base = z * n1 * n1;
                let zp_base = (z + 1) * n1 * n1;
                let (cur_z, next_z) = h[z_base..zp_base + n1 * n1].split_at_mut(n1 * n1);
                let (prev_y_part, dst_part) = cur_z.split_at_mut(y * n1);
                let prev_y = &prev_y_part[(y - 1) * n1..(y - 1) * n1 + y];
                let dst = &mut dst_part[..y];
                let next_zy = &next_z[y * n1..y * n1 + y];
                if w > 0 {
                    let diag = &next_z[(y - 1) * n1..(y - 1) * n1 + y];
                    for x in 0..y {
                        dst[x] = prev_y[x].max(next_zy[x]).max(diag[x] + 1);
                    }
                } else {
                    for x in 0..y {
                        dst[x] = prev_y[x].max(next_zy[x]);
                    }
                }
            }
        }
    }

    // 2) g(x,y,z)：[0,x) 与 [y,z) 之间的最优无交叉配对数（0<=x<=y<=z<=n）。
    //    递推与 z 无关（z 仅是 y==z 时的 0 边界），按 [x][y][z] 存放一次算完，最内层 z=y+1..=n 同样走 32 路 i16 向量取最大。
    buf.g.clear();
    buf.g.resize(total_sz, 0);
    let gi = |x: usize, y: usize, z: usize| (x * n1 + y) * n1 + z;
    {
        let g = &mut buf.g[..];
        let ok = &buf.ok[..];
        for x in 1..=n {
            let x_base = (x - 1) * n1 * n1;
            let (prev_x_plane, rest) = g[x_base..].split_at_mut(n1 * n1);
            let cur_x_plane = &mut rest[..n1 * n1];
            for y in (x..n).rev() {
                let w = ok[(x - 1) * n + y];
                let z0 = y + 1;
                let len = n + 1 - z0;
                let (dst_and_before, after_y) = cur_x_plane.split_at_mut((y + 1) * n1);
                let dst = &mut dst_and_before[y * n1 + z0..y * n1 + z0 + len];
                let next_y = &after_y[z0..z0 + len];
                let prev_x = &prev_x_plane[y * n1 + z0..y * n1 + z0 + len];
                if w > 0 {
                    let diag = &prev_x_plane[(y + 1) * n1 + z0..(y + 1) * n1 + z0 + len];
                    for k in 0..len {
                        dst[k] = prev_x[k].max(next_y[k]).max(diag[k] + 1);
                    }
                } else {
                    for k in 0..len {
                        dst[k] = prev_x[k].max(next_y[k]);
                    }
                }
            }
        }
    }

    // 3) 同分严格取 (z, x, y) 最先出现者；利用 g 在 y 上单调不增、h 在 y 上单调不减的上界安全跳过不可能破纪录的 (z, x)
    let (mut best, mut arg) = (0i32, (0usize, 0usize, 0usize));
    {
        let (g, h) = (&buf.g[..], &buf.h[..]);
        for z in 2..=n {
            for x in 1..(z - 1) {
                let g_ub = g[gi(x, x + 1, z)] as i32;
                if g_ub <= 0 {
                    continue;
                }
                let h_ub = h[hi(x, z, z)] as i32;
                if h_ub <= 0 || g_ub + h_ub <= best {
                    continue;
                }
                for y in (x + 1)..z {
                    let a = g[gi(x, y, z)] as i32;
                    let b = h[hi(x, y, z)] as i32;
                    if a > 0 && b > 0 && a + b > best {
                        best = a + b;
                        arg = (x, y, z);
                    }
                }
            }
        }
    }

    let mut pairs = Vec::new();
    if best > 0 {
        let (x0, y0, z0) = arg;
        let (mut x, mut y) = (x0, y0);
        while x > 0 && y < z0 {
            let cur = buf.g[gi(x, y, z0)];
            if cur == 0 {
                break;
            }
            if cur == buf.g[gi(x - 1, y, z0)] {
                x -= 1;
            } else if cur == buf.g[gi(x, y + 1, z0)] {
                y += 1;
            } else {
                pairs.push((x - 1, y));
                x -= 1;
                y += 1;
            }
        }
        let (mut y, mut z) = (y0, z0);
        while y > x0 && z < n {
            let cur = buf.h[hi(x0, y, z)];
            if cur == 0 {
                break;
            }
            if cur == buf.h[hi(x0, y - 1, z)] {
                y -= 1;
            } else if cur == buf.h[hi(x0, y, z + 1)] {
                z += 1;
            } else {
                pairs.push((y - 1, z));
                y -= 1;
                z += 1;
            }
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
        Kern {
            base: Table::new(),
            fast: FastTable::default(),
            fast16: FastTable::default(),
            stack: StackBuf::default(),
            prob: ProbBuf::default(),
            hpk: HpkBuf::default(),
            partner: Vec::new(),
        }
    }
}

/// 返回 (归一化得分, 配对列表[按左端递增])
pub fn nested(mode: Mode, k: &mut Kern, c: &Ctx, rule: usize) -> (f32, Vec<(usize, usize)>) {
    let n = c.n;
    k.partner.clear();
    k.partner.resize(n + 1, -1);
    let score = if rule == 11 {
        if mode == Mode::Fast && c.act.len() <= 8000 {
            stack_fold_fast(c, &mut k.stack, &mut k.partner)
        } else {
            let all: Vec<usize>;
            let pos: &[usize] = if mode == Mode::Base {
                all = (0..n).collect();
                &all
            } else {
                &c.act
            };
            stack_fold(c, pos, &mut k.stack, &mut k.partner)
        }
    } else {
        let h = rule_h(rule);
        macro_rules! run {
            ($r:expr) => {{
                let w = |a: usize, b: usize| weight(c, $r, a, b);
                run_nested(mode, k, c, $r, h, &w)
            }};
        }
        match rule {
            0 => run!(0),
            1 => run!(1),
            2 => run!(2),
            3 => run!(3),
            4 => run!(4),
            5 => run!(5),
            6 => run!(6),
            7 => run!(7),
            8 => run!(8),
            9 => run!(9),
            _ => run!(10),
        }
    };
    let ub = rule_ub(c, rule);
    let sn = if ub > 0.0 { (score as f64 / ub) as f32 } else { f32::NAN };
    let pairs = (0..n)
        .filter(|&p| k.partner[p] > p as i32)
        .map(|p| (p, k.partner[p] as usize))
        .collect();
    (sn, pairs)
}

#[inline(always)]
fn run_nested<W: Fn(usize, usize) -> i32>(mode: Mode, k: &mut Kern, c: &Ctx, rule: usize, h: usize, w: &W) -> i32 {
    let n = c.n;
    let kmax = if h == 10 { &c.act_kmax_h10 } else { &c.act_kmax_h3 };
    match mode {
        Mode::Base => {
            let s = k.base.solve(n, h, &c.active, w);
            k.base.traceback(n, h, w, true, &mut k.partner[..n]);
            s
        }
        Mode::Fast if small_score_rule(rule) || ((rule == 1 || rule == 2) && rule_ub(c, rule) <= 15_000.0) => {
            let s = k.fast16.solve_ctx(c, rule, kmax, w);
            k.fast16.traceback_ctx(c, kmax, w, &mut k.partner);
            s
        }
        Mode::Fast => {
            let s = k.fast.solve_ctx(c, rule, kmax, w);
            k.fast.traceback_ctx(c, kmax, w, &mut k.partner);
            s
        }
    }
}

/// 二层折叠：第一层是计数规则的结果；在第一层没配上的字母上再做一次无交叉配对，两层之间允许交叉。
pub fn two_layer(mode: Mode, k: &mut Kern, c: &Ctx, layer1: &[(usize, usize)]) -> (f32, Vec<(usize, usize)>) {
    let n = c.n;
    let mut used = vec![false; n];
    for &(a, b) in layer1 {
        used[a] = true;
        used[b] = true;
    }
    let rest: Vec<usize> = (0..n).filter(|&i| c.active[i] && !used[i]).collect();
    let m = rest.len();
    let mut pairs: Vec<(usize, usize)> = layer1.to_vec();
    if m >= 2 {
        let sub_side: Vec<i8> = rest.iter().map(|&i| c.side[i]).collect();
        let mut part = vec![-1i32; m + 1];
        match mode {
            Mode::Base => {
                let w = |a: usize, b: usize| {
                    if sub_side[a] as i32 * sub_side[b] as i32 == -1 && rest[b] - rest[a] > H {
                        1
                    } else {
                        NEG
                    }
                };
                let all = vec![true; m];
                k.base.solve(m, 0, &all, &w);
                k.base.traceback(m, 0, &w, true, &mut part[..m]);
            }
            Mode::Fast => {
                let w = |a: usize, b: usize| if c.side[a] as i32 * c.side[b] as i32 == -1 { 1 } else { NEG };
                k.fast16.solve(&rest, &c.side, H, &w);
                let mut full = vec![-1i32; n + 1];
                k.fast16.traceback(&rest, &w, &mut full);
                for (q, &o) in rest.iter().enumerate() {
                    if full[o] >= 0 {
                        part[q] = rest.iter().position(|&x| x == full[o] as usize).unwrap() as i32;
                    }
                }
            }
        }
        for q in 0..m {
            if part[q] > q as i32 {
                pairs.push((rest[q], rest[part[q] as usize]));
            }
        }
    }
    pairs.sort_unstable();
    let bound = c.n_buy.min(c.n_sell) as f64;
    let sn = if bound > 0.0 { (pairs.len() as f64 / bound) as f32 } else { f32::NAN };
    (sn, pairs)
}
