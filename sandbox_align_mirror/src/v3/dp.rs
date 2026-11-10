//! 第 2、3 层：计分表、镜像变换、比对动态规划（标量版，两种引擎共用的“完整路径”部分也在这里）。
//! 空位一律按“开启 o、延长 e”计价；o == e 时与线性空位完全等价（最高分逐位相同）。
//! 所有分数都是整数，|分数| 远小于 16000，计算结果与批量版逐位相同。

pub const K: usize = 8;
pub type Tab = [[i32; K]; K];
pub const NEG: i32 = -16000;
pub const NEG_HALF: i32 = -8000;

#[derive(Clone, Copy)]
pub struct Cfg {
    pub o: i32,
    pub e: i32,
    pub minsep: usize,
    pub size_swap: bool,
    pub tier_strict: bool,
    pub theta_mode: usize,
}
pub const BASE: Cfg = Cfg { o: 2, e: 2, minsep: 4, size_swap: false, tier_strict: false, theta_mode: 0 };

#[inline]
pub fn comp(a: usize, t: usize, size_swap: bool) -> usize {
    if a < 2 {
        return a;
    }
    let tier = (a - 2) / 2;
    let side = (a - 2) % 2;
    let tier2 = if size_swap { t - 1 - tier } else { tier };
    2 + 2 * tier2 + (1 - side)
}

fn base_score(a: usize, b: usize, strict: bool) -> i32 {
    if a == 0 || b == 0 {
        return if a == b { 0 } else { -1 };
    }
    if a == 1 || b == 1 {
        return if a == b { 0 } else { -1 };
    }
    let (ta, sa) = ((a - 2) / 2, (a - 2) % 2);
    let (tb, sb) = ((b - 2) / 2, (b - 2) % 2);
    if sa != sb {
        return -3;
    }
    let d = ta.abs_diff(tb);
    if strict {
        return if d == 0 { 2 } else { -1 };
    }
    match d {
        0 => 2,
        1 => 1,
        _ => 0,
    }
}

/// tab[a][b]：前面（或模板/昨天）的字母 a 对后面（今天）的字母 b；use_comp 时 b 先做互换
pub fn make_tab(t: usize, cfg: &Cfg, use_comp: bool) -> Tab {
    let mut tab = [[0i32; K]; K];
    let t = t.max(1);
    for a in 0..K {
        for b in 0..K {
            let maxl = 2 + 2 * t;
            if a >= maxl || b >= maxl {
                continue;
            }
            let tb = if use_comp { comp(b, t, cfg.size_swap) } else { b };
            tab[a][b] = base_score(a, tb, cfg.tier_strict);
        }
    }
    tab
}

#[inline]
pub fn side_of(a: u8) -> i16 {
    if a >= 2 {
        if (a - 2) % 2 == 0 {
            1
        } else {
            -1
        }
    } else {
        0
    }
}

/// BLAST 式归一常数 λ：只用两边字母组成解出；算不出返回 None
pub fn karlin_lambda(sa: &[u8], sb: &[u8], tab: &Tab) -> Option<f64> {
    let mut ca = [0f64; K];
    let mut cb = [0f64; K];
    for &x in sa {
        ca[x as usize] += 1.0;
    }
    for &x in sb {
        cb[x as usize] += 1.0;
    }
    let (na, nb) = (sa.len() as f64, sb.len() as f64);
    if na == 0.0 || nb == 0.0 {
        return None;
    }
    let mut e = 0.0;
    let mut pos = false;
    for a in 0..K {
        for b in 0..K {
            if ca[a] > 0.0 && cb[b] > 0.0 {
                e += ca[a] / na * cb[b] / nb * tab[a][b] as f64;
                pos |= tab[a][b] > 0;
            }
        }
    }
    if e >= 0.0 || !pos {
        return None;
    }
    let f = |l: f64| -> f64 {
        let mut s = 0.0;
        for a in 0..K {
            for b in 0..K {
                if ca[a] > 0.0 && cb[b] > 0.0 {
                    s += ca[a] / na * cb[b] / nb * (l * tab[a][b] as f64).exp();
                }
            }
        }
        s - 1.0
    };
    let (mut lo, mut hi) = (1e-4, 1.0);
    while f(hi) < 0.0 {
        hi *= 2.0;
        if hi > 64.0 {
            return None;
        }
    }
    for _ in 0..40 {
        let m = (lo + hi) / 2.0;
        if f(m) < 0.0 {
            lo = m
        } else {
            hi = m
        }
    }
    Some((lo + hi) / 2.0)
}

pub fn bits(score: i32, lambda: Option<f64>, cells: f64) -> f32 {
    match lambda {
        Some(l) if cells >= 1.0 => ((l * score as f64 - cells.ln()) / std::f64::consts::LN_2) as f32,
        _ => f32::NAN,
    }
}

// ---------------- 完整路径版 ----------------

#[derive(Clone, Copy)]
pub struct Cell {
    pub v: i32,
    pub i0: u16,
    pub j0: u16,
    pub pairs: u16,
    pub matched: u16,
    pub gaps: u16,
    pub bs: i16,
    pub mv: f64,
}
const ZERO: Cell = Cell { v: 0, i0: 0, j0: 0, pairs: 0, matched: 0, gaps: 0, bs: 0, mv: 0.0 };
const NEGC: Cell = Cell { v: NEG, ..ZERO };

#[derive(Clone, Copy)]
pub struct Best {
    pub score: i32,
    pub i: usize,
    pub j: usize,
    pub c: Cell,
}
impl Default for Best {
    fn default() -> Self {
        Best { score: 0, i: 0, j: 0, c: ZERO }
    }
}

#[derive(Default)]
pub struct Scratch {
    h0: Vec<Cell>,
    h1: Vec<Cell>,
    f0: Vec<Cell>,
    f1: Vec<Cell>,
    pub center: Vec<i32>,
    a: Vec<i32>,
}

#[inline]
fn gap_from(src: Cell, v: i32) -> Cell {
    let mut c = src;
    c.v = v;
    c.gaps = c.gaps.saturating_add(1);
    c
}

/// 选择规则（两种引擎一致）：对角 ≥ 上 且 ≥ 左 取对角；否则 上 ≥ 左 取上；否则取左；不为正就空
#[inline]
fn choose(dv: i32, diag: Cell, f: Cell, e: Cell) -> (u8, i32) {
    if dv >= f.v && dv >= e.v {
        (0, dv)
    } else if f.v >= e.v {
        let _ = diag;
        (1, f.v)
    } else {
        (2, e.v)
    }
}

/// 嵌套比对（M 倒序互换、R 只倒序）：G(i,j) 以 (i,j) 为最外层配对、从中心向外长。
/// want_center 时记录每个镜像中心（i0+j0）的最高分。
#[allow(clippy::too_many_arguments)]
pub fn nested_full(s: &[u8], vol: &[f32], tab: &Tab, o: i32, e: i32, minsep: usize, want_center: bool, sc: &mut Scratch) -> Best {
    let n = s.len();
    for v in [&mut sc.h0, &mut sc.h1] {
        v.clear();
        v.resize(n + 1, ZERO);
    }
    for v in [&mut sc.f0, &mut sc.f1] {
        v.clear();
        v.resize(n + 1, NEGC);
    }
    sc.center.clear();
    if want_center {
        sc.center.resize(2 * n + 1, 0);
    }
    let (mut ph, mut ch) = (&mut sc.h0, &mut sc.h1);
    let (mut pf, mut cf) = (&mut sc.f0, &mut sc.f1);
    let center = &mut sc.center;
    let mut best = Best::default();
    let ms = minsep.max(1);
    for i in (0..n).rev() {
        let start = i + ms;
        if start >= n {
            continue;
        }
        let row = &tab[s[i] as usize];
        ch[start - 1] = ZERO;
        cf[start - 1] = NEGC;
        let mut lh = ZERO;
        let mut ee = NEGC;
        for j in start..n {
            let p = row[s[j] as usize];
            // 左（同一行）
            let eo = lh.v - o;
            let ex = ee.v - e;
            ee = if eo >= ex { gap_from(lh, eo) } else { gap_from(ee, ex) };
            // 上（i+1 行）
            let fo = ph[j].v - o;
            let fx = pf[j].v - e;
            let ff = if fo >= fx { gap_from(ph[j], fo) } else { gap_from(pf[j], fx) };
            let dg = ph[j - 1];
            let dv = dg.v + p;
            let (k, v) = choose(dv, dg, ff, ee);
            let h = if v <= 0 {
                ZERO
            } else {
                match k {
                    0 => {
                        let mut c = if dg.v > 0 { dg } else { Cell { i0: i as u16, j0: j as u16, ..ZERO } };
                        c.v = v;
                        c.pairs += 1;
                        if p > 0 {
                            c.matched += 1;
                            c.bs += side_of(s[i]);
                            c.mv += vol[i] as f64 + vol[j] as f64;
                        }
                        c
                    }
                    1 => ff,
                    _ => ee,
                }
            };
            ch[j] = h;
            cf[j] = ff;
            lh = h;
            if h.v > 0 {
                if want_center {
                    let kk = h.i0 as usize + h.j0 as usize;
                    if h.v > center[kk] {
                        center[kk] = h.v;
                    }
                }
                if h.v > best.score {
                    best = Best { score: h.v, i, j, c: h };
                }
            }
        }
        std::mem::swap(&mut ph, &mut ch);
        std::mem::swap(&mut pf, &mut cf);
    }
    best
}

/// 同向比对（C 只互换、D 都不做）：两段都从左往右读，后段起点必须在前段当前位置之后（不重叠）
#[allow(clippy::too_many_arguments)]
pub fn forward_full(s: &[u8], vol: &[f32], tab: &Tab, o: i32, e: i32, minsep: usize, sc: &mut Scratch) -> Best {
    let n = s.len();
    for v in [&mut sc.h0, &mut sc.h1] {
        v.clear();
        v.resize(n + 1, ZERO);
    }
    for v in [&mut sc.f0, &mut sc.f1] {
        v.clear();
        v.resize(n + 1, NEGC);
    }
    let (mut ph, mut ch) = (&mut sc.h0, &mut sc.h1);
    let (mut pf, mut cf) = (&mut sc.f0, &mut sc.f1);
    let mut best = Best::default();
    let ms = minsep.max(1);
    for i in 0..n {
        let start = i + ms;
        if start >= n {
            break;
        }
        let row = &tab[s[i] as usize];
        ch[start - 1] = ZERO;
        cf[start - 1] = NEGC;
        let mut lh = ZERO;
        let mut ee = NEGC;
        let iu = i as u16;
        for j in start..n {
            let p = row[s[j] as usize];
            let lvalid = lh.v > 0 && lh.j0 > iu;
            let evalid = ee.v > NEG_HALF && ee.j0 > iu;
            let eo = if lvalid { lh.v - o } else { NEG };
            let ex = if evalid { ee.v - e } else { NEG };
            ee = if eo >= ex { gap_from(lh, eo) } else { gap_from(ee, ex) };
            let up = ph[j];
            let upf = pf[j];
            let uvalid = i > 0 && up.v > 0 && up.j0 > iu;
            let fvalid = i > 0 && upf.v > NEG_HALF && upf.j0 > iu;
            let fo = if uvalid { up.v - o } else { NEG };
            let fx = if fvalid { upf.v - e } else { NEG };
            let ff = if fo >= fx { gap_from(up, fo) } else { gap_from(upf, fx) };
            let dg = if i > 0 { ph[j - 1] } else { ZERO };
            let dvalid = dg.v > 0 && dg.j0 > iu;
            let dv = if dvalid { dg.v } else { 0 } + p;
            let (k, v) = choose(dv, dg, ff, ee);
            let h = if v <= 0 {
                ZERO
            } else {
                match k {
                    0 => {
                        let mut c = if dvalid { dg } else { Cell { i0: iu, j0: j as u16, ..ZERO } };
                        c.v = v;
                        c.pairs += 1;
                        if p > 0 {
                            c.matched += 1;
                            c.bs += side_of(s[i]);
                            c.mv += vol[i] as f64 + vol[j] as f64;
                        }
                        c
                    }
                    1 => ff,
                    _ => ee,
                }
            };
            ch[j] = h;
            cf[j] = ff;
            lh = h;
            if h.v > best.score {
                best = Best { score: h.v, i, j, c: h };
            }
        }
        std::mem::swap(&mut ph, &mut ch);
        std::mem::swap(&mut pf, &mut cf);
    }
    best
}

/// 两条序列的局部比对（跨日、同行业用）：a = 变换后的昨天/模板，b = 今天；线性空位 g。
/// 路径信息里 i0/j0 记 b（今天）上的起点；bs 记今天字母的方向。
pub fn rect_full(a: &[u8], b: &[u8], vb: &[f32], tab: &Tab, g: i32, sc: &mut Scratch) -> Best {
    let nb = b.len();
    for v in [&mut sc.h0, &mut sc.h1] {
        v.clear();
        v.resize(nb + 1, ZERO);
    }
    let (mut ph, mut ch) = (&mut sc.h0, &mut sc.h1);
    let mut best = Best::default();
    for (x, &ax) in a.iter().enumerate() {
        let row = &tab[ax as usize];
        ch[0] = ZERO;
        for y in 1..=nb {
            let p = row[b[y - 1] as usize];
            let dg = ph[y - 1];
            let dv = dg.v + p;
            let uv = ph[y].v - g;
            let lv = ch[y - 1].v - g;
            let h = if dv >= uv && dv >= lv {
                if dv > 0 {
                    let mut c = if dg.v > 0 { dg } else { Cell { i0: x as u16, j0: (y - 1) as u16, ..ZERO } };
                    c.v = dv;
                    c.pairs += 1;
                    if p > 0 {
                        c.matched += 1;
                        c.bs += side_of(b[y - 1]);
                        c.mv += vb[y - 1] as f64;
                    }
                    c
                } else {
                    ZERO
                }
            } else if uv >= lv {
                if uv > 0 {
                    gap_from(ph[y], uv)
                } else {
                    ZERO
                }
            } else if lv > 0 {
                gap_from(ch[y - 1], lv)
            } else {
                ZERO
            };
            ch[y] = h;
            if h.v > best.score {
                best = Best { score: h.v, i: x, j: y - 1, c: h };
            }
        }
        std::mem::swap(&mut ph, &mut ch);
    }
    best
}

/// 固定切点外扩（M 专用，线性空位）：前段倒着读、后段顺着读。返回 (最高分, 覆盖字母数)
pub fn extend_from_split(s: &[u8], c: usize, tab: &Tab, g: i32, sc: &mut Scratch) -> (i32, usize) {
    let n = s.len();
    if c == 0 || c >= n {
        return (0, 0);
    }
    let lb = n - c;
    sc.a.clear();
    sc.a.resize(lb + 1, 0);
    for y in 1..=lb {
        sc.a[y] = -(g * y as i32);
    }
    let mut best = (0i32, 0usize);
    for x in 1..=c {
        let row = &tab[s[c - x] as usize];
        let mut diag = sc.a[0];
        sc.a[0] = -(g * x as i32);
        for y in 1..=lb {
            let up = sc.a[y];
            let v = (diag + row[s[c + y - 1] as usize]).max(up - g).max(sc.a[y - 1] - g);
            diag = up;
            sc.a[y] = v;
            if v > best.0 {
                best = (v, x + y);
            }
        }
    }
    best
}

// ---------------- 只算最高分（打乱基准用；基准引擎逐条算，提速引擎批量算） ----------------

pub fn nested_score(s: &[u8], tab: &Tab, o: i32, e: i32, minsep: usize) -> i32 {
    let n = s.len();
    let mut ph = vec![0i32; n + 1];
    let mut ch = vec![0i32; n + 1];
    let mut pf = vec![NEG; n + 1];
    let mut cf = vec![NEG; n + 1];
    let ms = minsep.max(1);
    let mut best = 0;
    for i in (0..n).rev() {
        let start = i + ms;
        if start >= n {
            continue;
        }
        let row = &tab[s[i] as usize];
        ch[start - 1] = 0;
        cf[start - 1] = NEG;
        let mut lh = 0;
        let mut ee = NEG;
        for j in start..n {
            ee = (lh - o).max(ee - e);
            let ff = (ph[j] - o).max(pf[j] - e);
            let h = (ph[j - 1] + row[s[j] as usize]).max(ff).max(ee).max(0);
            ch[j] = h;
            cf[j] = ff;
            lh = h;
            best = best.max(h);
        }
        std::mem::swap(&mut ph, &mut ch);
        std::mem::swap(&mut pf, &mut cf);
    }
    best
}

pub fn forward_score(s: &[u8], tab: &Tab, o: i32, e: i32, minsep: usize) -> i32 {
    let n = s.len();
    let mut ph = vec![0i32; n + 1];
    let mut ch = vec![0i32; n + 1];
    let mut pj = vec![0i32; n + 1];
    let mut cj = vec![0i32; n + 1];
    let mut pf = vec![NEG; n + 1];
    let mut cf = vec![NEG; n + 1];
    let mut pfj = vec![0i32; n + 1];
    let mut cfj = vec![0i32; n + 1];
    let ms = minsep.max(1);
    let mut best = 0;
    for i in 0..n {
        let start = i + ms;
        if start >= n {
            break;
        }
        let row = &tab[s[i] as usize];
        let ii = i as i32;
        ch[start - 1] = 0;
        cj[start - 1] = 0;
        cf[start - 1] = NEG;
        cfj[start - 1] = 0;
        let (mut lh, mut lj) = (0i32, 0i32);
        let (mut ev, mut ej) = (NEG, 0i32);
        for j in start..n {
            let p = row[s[j] as usize];
            let eo = if lh > 0 && lj > ii { lh - o } else { NEG };
            let ex = if ev > NEG_HALF && ej > ii { ev - e } else { NEG };
            if eo >= ex {
                ev = eo;
                ej = lj;
            } else {
                ev = ex;
            }
            let (up, uj, upf, upfj) = if i > 0 { (ph[j], pj[j], pf[j], pfj[j]) } else { (0, 0, NEG, 0) };
            let fo = if up > 0 && uj > ii { up - o } else { NEG };
            let fx = if upf > NEG_HALF && upfj > ii { upf - e } else { NEG };
            let (fv, fj) = if fo >= fx { (fo, uj) } else { (fx, upfj) };
            let (dg, dj) = if i > 0 { (ph[j - 1], pj[j - 1]) } else { (0, 0) };
            let dvalid = dg > 0 && dj > ii;
            let dv = if dvalid { dg } else { 0 } + p;
            let dj2 = if dvalid { dj } else { j as i32 };
            let (h, hj) = if dv >= fv && dv >= ev {
                (dv, dj2)
            } else if fv >= ev {
                (fv, fj)
            } else {
                (ev, ej)
            };
            let (h, hj) = if h <= 0 { (0, 0) } else { (h, hj) };
            ch[j] = h;
            cj[j] = hj;
            cf[j] = fv;
            cfj[j] = fj;
            lh = h;
            lj = hj;
            best = best.max(h);
        }
        std::mem::swap(&mut ph, &mut ch);
        std::mem::swap(&mut pj, &mut cj);
        std::mem::swap(&mut pf, &mut cf);
        std::mem::swap(&mut pfj, &mut cfj);
    }
    best
}

pub fn rect_score(a: &[u8], b: &[u8], tab: &Tab, g: i32) -> i32 {
    let nb = b.len();
    let mut ph = vec![0i32; nb + 1];
    let mut ch = vec![0i32; nb + 1];
    let mut best = 0;
    for &ax in a {
        let row = &tab[ax as usize];
        ch[0] = 0;
        for y in 1..=nb {
            let h = (ph[y - 1] + row[b[y - 1] as usize]).max(ph[y] - g).max(ch[y - 1] - g).max(0);
            ch[y] = h;
            best = best.max(h);
        }
        std::mem::swap(&mut ph, &mut ch);
    }
    best
}
