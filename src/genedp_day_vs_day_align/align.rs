//! 第三版比对层：带状 + 仿射空位的动态规划，支持全局、首尾免罚（半全局）、局部三种模式，
//! 今天和昨天长度可以不同（逐笔级字母），带子沿 j = i·m/n 这条斜线走。
//!
//! 两种实现，结果逐字节相同：
//! - `align_scalar`：一次算一题，冻结的基准实现。
//! - `align_batch`：同一形状（n、m、带宽相同）的多题放进各自的“车道”一起算，
//!   每格的计分改用字母属性的算式（与计分表逐项相等，见单元测试），内层循环无分支，
//!   编译器自动向量化。分数、来源选择、并列时的偏好与基准完全一致，所以回溯路径也一致。
//!
//! 操作：D=两边对齐，X=今天多出一个（昨天留空），Y=昨天多出一个（今天留空）。

pub const SIZE: usize = 12; // 每个字母表最多 12 个字母（含静默字母 0）
pub const OP_D: u8 = 0;
pub const OP_X: u8 = 1;
pub const OP_Y: u8 = 2;
const NEG: i32 = -500_000_000;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Mode { Global, Semi, Local }

#[derive(Clone, Copy, Default, PartialEq, Eq, Debug)]
pub struct Aln { pub score: i32, pub i0: usize, pub j0: usize, pub i1: usize, pub j1: usize }

/// 第 i 行带子中心（今天第 i 格对应昨天的大致位置）
#[inline]
pub fn center(i: usize, n: usize, m: usize) -> usize {
    if n == m { i } else { (i as u64 * m as u64 / n as u64) as usize }
}

#[derive(Default)]
pub struct Work { h_prev: Vec<i32>, h_cur: Vec<i32>, x_prev: Vec<i32>, x_cur: Vec<i32>, tr: Vec<u8> }

/// 从回溯字节表恢复路径（逆序写入后翻转）。trace(i, p) 返回 (i, p) 格的回溯字节。
fn traceback(mut i: usize, mut j: usize, n: usize, m: usize, band: usize,
             trace: impl Fn(usize, usize) -> u8, ops: &mut Vec<u8>) -> (usize, usize) {
    ops.clear();
    let mut state = 0u8; // 0=H,1=X,2=Y
    loop {
        let p = j + band + 1 - center(i, n, m);
        let t = trace(i, p);
        match state {
            0 => match t & 3 {
                0 => break,
                1 => { ops.push(OP_D); i -= 1; j -= 1; }
                2 => { ops.push(OP_X); state = if t & 4 != 0 { 1 } else { 0 }; i -= 1; }
                _ => { ops.push(OP_Y); state = if t & 8 != 0 { 2 } else { 0 }; j -= 1; }
            },
            1 => { ops.push(OP_X); state = if t & 4 != 0 { 1 } else { 0 }; i -= 1; }
            _ => { ops.push(OP_Y); state = if t & 8 != 0 { 2 } else { 0 }; j -= 1; }
        }
    }
    ops.reverse();
    (i, j)
}

/// 基准实现（一次一题）。回溯字节：低 2 位来源（0 停止,1 D,2 X,3 Y）；4=X 延续；8=Y 延续。
pub fn align_scalar(a: &[u8], b: &[u8], tbl: &[i32], open: i32, ext: i32, band: usize, mode: Mode,
                    w: &mut Work, ops: &mut Vec<u8>) -> Aln {
    let (n, m) = (a.len(), b.len());
    let width = 2 * band + 1;
    let row = width + 2;
    for v in [&mut w.h_prev, &mut w.h_cur, &mut w.x_prev, &mut w.x_cur] { v.clear(); v.resize(row, NEG); }
    w.tr.clear(); w.tr.resize((n + 1) * width, 0);
    let global = mode == Mode::Global;
    // 第 0 行
    w.h_prev[band + 1] = 0;
    for j in 1..=band.min(m) {
        let p = band + 1 + j;
        if global {
            w.h_prev[p] = -(open + (j as i32 - 1) * ext);
            w.tr[p - 1] = 3 | if j > 1 { 8 } else { 0 };
        } else { w.h_prev[p] = 0; }
    }
    let (mut best, mut bi, mut bj) = (if mode == Mode::Semi { NEG } else { 0 }, 0usize, 0usize);
    if mode == Mode::Semi && n == 0 { best = 0; }
    for i in 1..=n {
        let base = i * width;
        let (ci, cp) = (center(i, n, m), center(i - 1, n, m));
        let d = ci - cp;
        let mut y_left = NEG;
        w.h_cur.iter_mut().for_each(|v| *v = NEG);
        w.x_cur.iter_mut().for_each(|v| *v = NEG);
        let jlo = ci.saturating_sub(band);
        let jhi = m.min(ci + band);
        let ai = a[i - 1] as usize * SIZE;
        for j in jlo..=jhi {
            let p = j + band + 1 - ci;
            if j == 0 {
                if global {
                    let v = -(open + (i as i32 - 1) * ext);
                    w.h_cur[p] = v; w.x_cur[p] = v;
                    w.tr[base + p - 1] = 2 | if i > 1 { 4 } else { 0 };
                } else {
                    w.h_cur[p] = 0;
                    if mode == Mode::Semi && (i == n || j == m) && 0 > best { best = 0; bi = i; bj = j; }
                }
                continue;
            }
            let pu = p + d;
            let (hu, xu) = if pu <= width { (w.h_prev[pu], w.x_prev[pu]) } else { (NEG, NEG) };
            let pd = p + d - 1;
            let hd = if pd <= width { w.h_prev[pd] } else { NEG };
            let (x1, x2) = (hu - open, xu - ext);
            let (xv, xc) = if x2 > x1 { (x2, 4u8) } else { (x1, 0u8) };
            let (y1, y2) = (w.h_cur[p - 1] - open, y_left - ext);
            let (yv, yc) = if y2 > y1 { (y2, 8u8) } else { (y1, 0u8) };
            let dd = hd + tbl[ai + b[j - 1] as usize];
            let (mut h, mut o) = if dd >= xv && dd >= yv { (dd, 1u8) } else if xv >= yv { (xv, 2u8) } else { (yv, 3u8) };
            if mode == Mode::Local && h <= 0 { h = 0; o = 0; }
            w.h_cur[p] = h; w.x_cur[p] = xv; y_left = yv;
            w.tr[base + p - 1] = o | xc | yc;
            match mode {
                Mode::Local => if h > best { best = h; bi = i; bj = j; },
                Mode::Semi => if (i == n || j == m) && h > best { best = h; bi = i; bj = j; },
                Mode::Global => {}
            }
        }
        std::mem::swap(&mut w.h_prev, &mut w.h_cur);
        std::mem::swap(&mut w.x_prev, &mut w.x_cur);
    }
    let (i1, j1, score) = match mode {
        Mode::Global => (n, m, w.h_prev[m + band + 1 - center(n, n, m)]),
        _ => (bi, bj, best),
    };
    let tr = &w.tr;
    let (i0, j0) = traceback(i1, j1, n, m, band, |i, p| tr[i * width + p - 1], ops);
    Aln { score, i0, j0, i1, j1 }
}

/// 一道比对题：今天、昨天的字母、字母表的属性格、模式和空位罚分。
#[derive(Clone, Copy)]
pub struct Prob<'a> { pub a: &'a [u8], pub b: &'a [u8], pub grid: (usize, usize), pub mode: Mode, pub open: i32, pub ext: i32 }

/// 字母 → (第一属性, 第二属性, 是否静默)。静默字母属性取 -99，保证不会和任何非静默字母相等。
#[inline]
fn attrs(x: u8, nb: usize) -> (i32, i32, i32) {
    if x == 0 { (-99, -99, 1) } else { (((x as usize - 1) / nb) as i32, ((x as usize - 1) % nb) as i32, 0) }
}

/// 批量缓冲：每格每车道一个值，整只股票复用。
pub struct BatchWork<const L: usize> {
    h_prev: Vec<[i32; L]>, h_cur: Vec<[i32; L]>, x_prev: Vec<[i32; L]>, x_cur: Vec<[i32; L]>,
    tr: Vec<[u8; L]>, ra: Vec<[i32; L]>, rb: Vec<[i32; L]>, rs: Vec<[i32; L]>,
    ca: Vec<[i32; L]>, cb: Vec<[i32; L]>, cs: Vec<[i32; L]>,
}
impl<const L: usize> Default for BatchWork<L> {
    fn default() -> Self {
        BatchWork { h_prev: vec![], h_cur: vec![], x_prev: vec![], x_cur: vec![], tr: vec![],
            ra: vec![], rb: vec![], rs: vec![], ca: vec![], cb: vec![], cs: vec![] }
    }
}

/// 批量实现：probs 里所有题必须同 n、同 m、同带宽，题数 ≤ L。返回每题的结果，路径写进 ops[k]。
pub fn align_batch<const L: usize>(probs: &[Prob], band: usize, w: &mut BatchWork<L>, ops: &mut [Vec<u8>]) -> Vec<Aln> {
    let np = probs.len();
    assert!(np >= 1 && np <= L);
    let (n, m) = (probs[0].a.len(), probs[0].b.len());
    let width = 2 * band + 1;
    let row = width + 2;
    // 不满的车道用第 0 题补齐，结果丢弃
    let pk = |k: usize| &probs[if k < np { k } else { 0 }];
    let mut open = [0i32; L]; let mut ext = [0i32; L];
    let mut is_local = [false; L]; let mut is_semi = [false; L]; let mut single = [false; L];
    for k in 0..L {
        let p = pk(k);
        debug_assert!(p.a.len() == n && p.b.len() == m);
        open[k] = p.open; ext[k] = p.ext;
        is_local[k] = p.mode == Mode::Local; is_semi[k] = p.mode == Mode::Semi;
        single[k] = p.grid.1 == 1;
    }
    // 字母属性按“位置 × 车道”排好
    let fill = |dst: &mut Vec<[i32; L]>, len: usize| { dst.clear(); dst.resize(len, [0; L]); };
    fill(&mut w.ra, n); fill(&mut w.rb, n); fill(&mut w.rs, n);
    fill(&mut w.ca, m); fill(&mut w.cb, m); fill(&mut w.cs, m);
    for k in 0..L {
        let p = pk(k);
        let nb = p.grid.1;
        for i in 0..n { let (x, y, s) = attrs(p.a[i], nb); w.ra[i][k] = x; w.rb[i][k] = y; w.rs[i][k] = s; }
        for j in 0..m { let (x, y, s) = attrs(p.b[j], nb); w.ca[j][k] = x; w.cb[j][k] = y; w.cs[j][k] = s; }
    }
    for v in [&mut w.h_prev, &mut w.h_cur, &mut w.x_prev, &mut w.x_cur] { v.clear(); v.resize(row, [NEG; L]); }
    w.tr.clear(); w.tr.resize((n + 1) * width, [0u8; L]);
    // 第 0 行
    w.h_prev[band + 1] = [0; L];
    for j in 1..=band.min(m) {
        let p = band + 1 + j;
        for k in 0..L {
            if !is_local[k] && !is_semi[k] {
                w.h_prev[p][k] = -(open[k] + (j as i32 - 1) * ext[k]);
                w.tr[p - 1][k] = 3 | if j > 1 { 8 } else { 0 };
            } else { w.h_prev[p][k] = 0; }
        }
    }
    let mut best = [0i32; L];
    for k in 0..L { if is_semi[k] { best[k] = if n == 0 { 0 } else { NEG }; } }
    let mut bi = [0i32; L]; let mut bj = [0i32; L];
    let neg = [NEG; L];
    for i in 1..=n {
        let base = i * width;
        let (ci, cp) = (center(i, n, m), center(i - 1, n, m));
        let d = ci - cp;
        let mut y_left = [NEG; L];
        w.h_cur.iter_mut().for_each(|v| *v = [NEG; L]);
        w.x_cur.iter_mut().for_each(|v| *v = [NEG; L]);
        let jlo = ci.saturating_sub(band);
        let jhi = m.min(ci + band);
        let (xa, xb, xs) = (w.ra[i - 1], w.rb[i - 1], w.rs[i - 1]);
        let last_row = i == n;
        for j in jlo..=jhi {
            let p = j + band + 1 - ci;
            if j == 0 {
                for k in 0..L {
                    if !is_local[k] && !is_semi[k] {
                        let v = -(open[k] + (i as i32 - 1) * ext[k]);
                        w.h_cur[p][k] = v; w.x_cur[p][k] = v;
                        w.tr[base + p - 1][k] = 2 | if i > 1 { 4 } else { 0 };
                    } else {
                        w.h_cur[p][k] = 0;
                        if is_semi[k] && (last_row || j == m) && 0 > best[k] { best[k] = 0; bi[k] = i as i32; bj[k] = j as i32; }
                    }
                }
                continue;
            }
            let pu = p + d;
            let (hu, xu) = if pu <= width { (w.h_prev[pu], w.x_prev[pu]) } else { (neg, neg) };
            let pd = p + d - 1;
            let hd = if pd <= width { w.h_prev[pd] } else { neg };
            let hl = w.h_cur[p - 1];
            let (ya, yb, ys) = (w.ca[j - 1], w.cb[j - 1], w.cs[j - 1]);
            let edge = last_row || j == m;
            let (ii, jj) = (i as i32, j as i32);
            let mut hn = [0i32; L]; let mut xn = [0i32; L]; let mut trn = [0u8; L];
            for k in 0..L {
                // 计分：与 build_score_table 逐项相等
                let both = xs[k] & ys[k];
                let one = xs[k] ^ ys[k];
                let same = (xa[k] == ya[k]) & (xb[k] == yb[k]);
                let da = xa[k] - ya[k];
                let partial = if single[k] { (da == 1) | (da == -1) } else { (xa[k] == ya[k]) | (xb[k] == yb[k]) };
                let s = if both == 1 { 1 } else if one == 1 { -2 } else if same { 4 } else if partial { 1 } else { -2 };
                let (x1, x2) = (hu[k] - open[k], xu[k] - ext[k]);
                let xc = x2 > x1;
                let xv = if xc { x2 } else { x1 };
                let (y1, y2) = (hl[k] - open[k], y_left[k] - ext[k]);
                let yc = y2 > y1;
                let yv = if yc { y2 } else { y1 };
                let dd = hd[k] + s;
                let dge = (dd >= xv) & (dd >= yv);
                let xge = xv >= yv;
                let mut h = if dge { dd } else if xge { xv } else { yv };
                let mut o: u8 = if dge { 1 } else if xge { 2 } else { 3 };
                let z = is_local[k] & (h <= 0);
                if z { h = 0; o = 0; }
                hn[k] = h; xn[k] = xv; y_left[k] = yv;
                trn[k] = o | ((xc as u8) << 2) | ((yc as u8) << 3);
                let upd = (is_local[k] & (h > best[k])) | (is_semi[k] & edge & (h > best[k]));
                if upd { best[k] = h; bi[k] = ii; bj[k] = jj; }
            }
            w.h_cur[p] = hn; w.x_cur[p] = xn; w.tr[base + p - 1] = trn;
        }
        std::mem::swap(&mut w.h_prev, &mut w.h_cur);
        std::mem::swap(&mut w.x_prev, &mut w.x_cur);
    }
    let gend = m + band + 1 - center(n, n, m);
    let mut out = Vec::with_capacity(np);
    let tr = &w.tr;
    for k in 0..np {
        let (i1, j1, score) = if is_local[k] || is_semi[k] { (bi[k] as usize, bj[k] as usize, best[k]) }
            else { (n, m, w.h_prev[gend][k]) };
        let (i0, j0) = traceback(i1, j1, n, m, band, |i, p| tr[i * width + p - 1][k], &mut ops[k]);
        out.push(Aln { score, i0, j0, i1, j1 });
    }
    out
}

/// 机器是否支持 512 位整数向量指令（支持就走 align_batch16_avx512）
pub fn has_avx512() -> bool {
    #[cfg(target_arch = "x86_64")]
    { is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw") && is_x86_feature_detected!("avx512vl") }
    #[cfg(not(target_arch = "x86_64"))]
    { false }
}

/// 16 车道批量实现的显式向量版：逐格逻辑与 align_batch / align_scalar 完全相同，
/// 只是 16 道题的同一格用一条 512 位指令一起算。调用前必须确认 has_avx512()。
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
pub unsafe fn align_batch16_avx512(probs: &[Prob], band: usize, w: &mut BatchWork<16>, ops: &mut [Vec<u8>]) -> Vec<Aln> {
    use std::arch::x86_64::*;
    const L: usize = 16;
    let np = probs.len();
    assert!(np >= 1 && np <= L);
    let (n, m) = (probs[0].a.len(), probs[0].b.len());
    let width = 2 * band + 1;
    let row = width + 2;
    let pk = |k: usize| &probs[if k < np { k } else { 0 }];
    let mut open = [0i32; L]; let mut ext = [0i32; L];
    let (mut local_m, mut semi_m, mut single_m): (u16, u16, u16) = (0, 0, 0);
    for k in 0..L {
        let p = pk(k);
        debug_assert!(p.a.len() == n && p.b.len() == m);
        open[k] = p.open; ext[k] = p.ext;
        if p.mode == Mode::Local { local_m |= 1 << k; }
        if p.mode == Mode::Semi { semi_m |= 1 << k; }
        if p.grid.1 == 1 { single_m |= 1 << k; }
    }
    let global_m: u16 = !(local_m | semi_m);
    let fill = |dst: &mut Vec<[i32; L]>, len: usize| { dst.clear(); dst.resize(len, [0; L]); };
    fill(&mut w.ra, n); fill(&mut w.rb, n); fill(&mut w.rs, n);
    fill(&mut w.ca, m); fill(&mut w.cb, m); fill(&mut w.cs, m);
    for k in 0..L {
        let p = pk(k);
        let nb = p.grid.1;
        for i in 0..n { let (x, y, s) = attrs(p.a[i], nb); w.ra[i][k] = x; w.rb[i][k] = y; w.rs[i][k] = s; }
        for j in 0..m { let (x, y, s) = attrs(p.b[j], nb); w.ca[j][k] = x; w.cb[j][k] = y; w.cs[j][k] = s; }
    }
    for v in [&mut w.h_prev, &mut w.h_cur, &mut w.x_prev, &mut w.x_cur] { v.clear(); v.resize(row, [NEG; L]); }
    // 回溯表不整块清零：回溯只会走到算过的格子，而算过的格子每个车道都会被写满；第 0 行单独清零
    if w.tr.len() < (n + 1) * width { w.tr.resize((n + 1) * width, [0u8; L]); }
    for p in 0..width { w.tr[p] = [0u8; L]; }
    w.h_prev[band + 1] = [0; L];
    for j in 1..=band.min(m) {
        let p = band + 1 + j;
        for k in 0..L {
            if global_m & (1 << k) != 0 {
                w.h_prev[p][k] = -(open[k] + (j as i32 - 1) * ext[k]);
                w.tr[p - 1][k] = 3 | if j > 1 { 8 } else { 0 };
            } else { w.h_prev[p][k] = 0; }
        }
    }
    let ld = |a: &[i32; L]| _mm512_loadu_si512(a.as_ptr() as *const _);
    let openv = ld(&open); let extv = ld(&ext);
    let negv = _mm512_set1_epi32(NEG);
    let zero = _mm512_setzero_si512();
    let (c1, c2, c3, c4, cm2) = (_mm512_set1_epi32(1), _mm512_set1_epi32(2), _mm512_set1_epi32(3), _mm512_set1_epi32(4), _mm512_set1_epi32(-2));
    let (c8, cx4) = (_mm512_set1_epi32(8), _mm512_set1_epi32(4));
    let mut bestv = _mm512_mask_blend_epi32(semi_m, zero, _mm512_set1_epi32(if n == 0 { 0 } else { NEG }));
    let mut biv = zero; let mut bjv = zero;
    for i in 1..=n {
        let base = i * width;
        let (ci, cp) = (center(i, n, m), center(i - 1, n, m));
        let d = ci - cp;
        let jlo = ci.saturating_sub(band);
        let jhi = m.min(ci + band);
        // 这一行算不到的格子置为不可达（只有矩阵边角才会出现）
        let (plo, phi) = (jlo + band + 1 - ci, jhi + band + 1 - ci);
        for p in 1..plo { w.h_cur[p] = [NEG; L]; w.x_cur[p] = [NEG; L]; }
        for p in phi + 1..=width { w.h_cur[p] = [NEG; L]; w.x_cur[p] = [NEG; L]; }
        let xa = ld(&w.ra[i - 1]); let xb = ld(&w.rb[i - 1]);
        let xs_m = _mm512_cmpeq_epi32_mask(ld(&w.rs[i - 1]), c1);
        let last_row = i == n;
        let iv = _mm512_set1_epi32(i as i32);
        let mut yl = negv;
        let mut jstart = jlo;
        if jlo == 0 {
            let p = band + 1 - ci;
            // -(open + (i-1)·ext)
            let gv = _mm512_sub_epi32(zero, _mm512_add_epi32(openv, _mm512_mullo_epi32(extv, _mm512_set1_epi32(i as i32 - 1))));
            let h = _mm512_mask_blend_epi32(global_m, zero, gv);
            let x = _mm512_mask_blend_epi32(global_m, negv, gv);
            _mm512_storeu_si512(w.h_cur[p].as_mut_ptr() as *mut _, h);
            _mm512_storeu_si512(w.x_cur[p].as_mut_ptr() as *mut _, x);
            let tv: u8 = 2 | if i > 1 { 4 } else { 0 };
            for k in 0..L { w.tr[base + p - 1][k] = if global_m & (1 << k) != 0 { tv } else { 0 }; }
            if last_row || m == 0 {
                let upd = _mm512_mask_cmpgt_epi32_mask(semi_m, zero, bestv);
                bestv = _mm512_mask_blend_epi32(upd, bestv, zero);
                biv = _mm512_mask_blend_epi32(upd, biv, iv);
                bjv = _mm512_mask_blend_epi32(upd, bjv, zero);
            }
            jstart = 1;
        }
        for j in jstart..=jhi {
            let p = j + band + 1 - ci;
            let pu = p + d;
            let (hu, xu) = if pu <= width { (ld(&w.h_prev[pu]), ld(&w.x_prev[pu])) } else { (negv, negv) };
            let pd = p + d - 1;
            let hd = if pd <= width { ld(&w.h_prev[pd]) } else { negv };
            let hl = ld(&w.h_cur[p - 1]);
            let ya = ld(&w.ca[j - 1]); let yb = ld(&w.cb[j - 1]);
            let ys_m = _mm512_cmpeq_epi32_mask(ld(&w.cs[j - 1]), c1);
            // 计分
            let both = xs_m & ys_m;
            let one = xs_m ^ ys_m;
            let eqa = _mm512_cmpeq_epi32_mask(xa, ya);
            let eqb = _mm512_cmpeq_epi32_mask(xb, yb);
            let adj = _mm512_cmpeq_epi32_mask(_mm512_abs_epi32(_mm512_sub_epi32(xa, ya)), c1);
            let partial = (single_m & adj) | (!single_m & (eqa | eqb));
            let mut s = _mm512_mask_blend_epi32(partial, cm2, c1);
            s = _mm512_mask_blend_epi32(eqa & eqb, s, c4);
            s = _mm512_mask_blend_epi32(one, s, cm2);
            s = _mm512_mask_blend_epi32(both, s, c1);
            // 递推
            let x1 = _mm512_sub_epi32(hu, openv); let x2 = _mm512_sub_epi32(xu, extv);
            let xc = _mm512_cmpgt_epi32_mask(x2, x1);
            let xv = _mm512_max_epi32(x1, x2);
            let y1 = _mm512_sub_epi32(hl, openv); let y2 = _mm512_sub_epi32(yl, extv);
            let yc = _mm512_cmpgt_epi32_mask(y2, y1);
            let yv = _mm512_max_epi32(y1, y2);
            let dd = _mm512_add_epi32(hd, s);
            let dge = _mm512_cmpge_epi32_mask(dd, xv) & _mm512_cmpge_epi32_mask(dd, yv);
            let xge = _mm512_cmpge_epi32_mask(xv, yv);
            let mut h = _mm512_mask_blend_epi32(dge, _mm512_mask_blend_epi32(xge, yv, xv), dd);
            let mut o = _mm512_mask_blend_epi32(dge, _mm512_mask_blend_epi32(xge, c3, c2), c1);
            let z = _mm512_mask_cmple_epi32_mask(local_m, h, zero);
            h = _mm512_mask_blend_epi32(z, h, zero);
            o = _mm512_mask_blend_epi32(z, o, zero);
            o = _mm512_or_si512(o, _mm512_maskz_mov_epi32(xc, cx4));
            o = _mm512_or_si512(o, _mm512_maskz_mov_epi32(yc, c8));
            _mm512_storeu_si512(w.h_cur[p].as_mut_ptr() as *mut _, h);
            _mm512_storeu_si512(w.x_cur[p].as_mut_ptr() as *mut _, xv);
            _mm_storeu_si128(w.tr[base + p - 1].as_mut_ptr() as *mut _, _mm512_cvtepi32_epi8(o));
            yl = yv;
            let cand = local_m | if last_row || j == m { semi_m } else { 0 };
            let upd = _mm512_mask_cmpgt_epi32_mask(cand, h, bestv);
            if upd != 0 {
                bestv = _mm512_mask_blend_epi32(upd, bestv, h);
                biv = _mm512_mask_blend_epi32(upd, biv, iv);
                bjv = _mm512_mask_blend_epi32(upd, bjv, _mm512_set1_epi32(j as i32));
            }
        }
        std::mem::swap(&mut w.h_prev, &mut w.h_cur);
        std::mem::swap(&mut w.x_prev, &mut w.x_cur);
    }
    let (mut best, mut bi, mut bj) = ([0i32; L], [0i32; L], [0i32; L]);
    _mm512_storeu_si512(best.as_mut_ptr() as *mut _, bestv);
    _mm512_storeu_si512(bi.as_mut_ptr() as *mut _, biv);
    _mm512_storeu_si512(bj.as_mut_ptr() as *mut _, bjv);
    let gend = m + band + 1 - center(n, n, m);
    let mut out = Vec::with_capacity(np);
    let tr = &w.tr;
    for k in 0..np {
        let (i1, j1, score) = if global_m & (1 << k) == 0 { (bi[k] as usize, bj[k] as usize, best[k]) }
            else { (n, m, w.h_prev[gend][k]) };
        let (i0, j0) = traceback(i1, j1, n, m, band, |i, p| tr[i * width + p - 1][k], &mut ops[k]);
        out.push(Aln { score, i0, j0, i1, j1 });
    }
    out
}

/// 位标记回溯表的一格：每个车道一位。z=停止，dge=来自对角，xge=来自上方（否则来自左方），xc/yc=空位延续。
#[derive(Clone, Copy, Default)]
pub struct MaskCell<M: Copy + Default> { z: M, dge: M, xge: M, xc: M, yc: M }

macro_rules! mask_trace_byte {
    ($c:expr, $k:expr) => {{
        let c = $c; let k = $k;
        let o: u8 = if (c.z >> k) & 1 != 0 { 0 } else if (c.dge >> k) & 1 != 0 { 1 } else if (c.xge >> k) & 1 != 0 { 2 } else { 3 };
        o | ((((c.xc >> k) & 1) as u8) << 2) | ((((c.yc >> k) & 1) as u8) << 3)
    }};
}

/// 位标记版批量缓冲（8 车道 i32 逐笔级用 / 32 车道 i16 格子级用）
pub struct MaskWork<T: Copy + Default, M: Copy + Default, const L: usize> {
    h_prev: Vec<[T; L]>, h_cur: Vec<[T; L]>, x_prev: Vec<[T; L]>, x_cur: Vec<[T; L]>,
    ra: Vec<[T; L]>, rb: Vec<[T; L]>, rs: Vec<[T; L]>, ca: Vec<[T; L]>, cb: Vec<[T; L]>, cs: Vec<[T; L]>,
    tr: Vec<MaskCell<M>>,
}
impl<T: Copy + Default, M: Copy + Default, const L: usize> Default for MaskWork<T, M, L> {
    fn default() -> Self { MaskWork { h_prev: vec![], h_cur: vec![], x_prev: vec![], x_cur: vec![], ra: vec![], rb: vec![],
        rs: vec![], ca: vec![], cb: vec![], cs: vec![], tr: vec![] } }
}

/// 两种位标记内核共用的准备：车道参数、字母属性、第 0 行。返回 (open, ext, local, semi, single, global) 车道参数。
macro_rules! mask_setup {
    ($probs:expr, $band:expr, $w:expr, $T:ty, $M:ty, $L:expr, $NEG:expr) => {{
        let probs = $probs; let band = $band; let w = &mut *$w;
        let np = probs.len();
        assert!(np >= 1 && np <= $L);
        let (n, m) = (probs[0].a.len(), probs[0].b.len());
        let width = 2 * band + 1;
        let row = width + 2;
        let pk = |k: usize| &probs[if k < np { k } else { 0 }];
        let mut open = [0 as $T; $L]; let mut ext = [0 as $T; $L];
        let (mut local_m, mut semi_m, mut single_m): ($M, $M, $M) = (0, 0, 0);
        for k in 0..$L {
            let p = pk(k);
            open[k] = p.open as $T; ext[k] = p.ext as $T;
            if p.mode == Mode::Local { local_m |= 1 << k; }
            if p.mode == Mode::Semi { semi_m |= 1 << k; }
            if p.grid.1 == 1 { single_m |= 1 << k; }
        }
        let global_m: $M = !(local_m | semi_m);
        for (dst, len) in [(&mut w.ra, n), (&mut w.rb, n), (&mut w.rs, n), (&mut w.ca, m), (&mut w.cb, m), (&mut w.cs, m)] {
            dst.clear(); dst.resize(len, [0 as $T; $L]);
        }
        for k in 0..$L {
            let p = pk(k);
            let nb = p.grid.1;
            for i in 0..n { let (x, y, s) = attrs(p.a[i], nb); w.ra[i][k] = x as $T; w.rb[i][k] = y as $T; w.rs[i][k] = s as $T; }
            for j in 0..m { let (x, y, s) = attrs(p.b[j], nb); w.ca[j][k] = x as $T; w.cb[j][k] = y as $T; w.cs[j][k] = s as $T; }
        }
        for v in [&mut w.h_prev, &mut w.h_cur, &mut w.x_prev, &mut w.x_cur] { v.clear(); v.resize(row, [$NEG; $L]); }
        if w.tr.len() < (n + 1) * width { w.tr.resize((n + 1) * width, MaskCell::default()); }
        // 第 0 行：全部车道默认“停止”；全局车道在 1..=band 列是来自左方的空位
        for p in 0..width { w.tr[p] = MaskCell { z: !0, ..MaskCell::default() }; }
        w.h_prev[band + 1] = [0 as $T; $L];
        for j in 1..=band.min(m) {
            let p = band + 1 + j;
            for k in 0..$L {
                if (global_m >> k) & 1 != 0 { w.h_prev[p][k] = -(open[k] as i32 + (j as i32 - 1) * ext[k] as i32) as $T; }
                else { w.h_prev[p][k] = 0 as $T; }
            }
            w.tr[p - 1] = MaskCell { z: !global_m, yc: if j > 1 { global_m } else { 0 }, ..MaskCell::default() };
        }
        (np, n, m, width, open, ext, local_m, semi_m, single_m, global_m)
    }};
}

/// 逐笔级专用：8 车道、32 位整数、256 位向量。逐格逻辑与 align_scalar 完全相同。
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx2")]
pub unsafe fn align_batch8_ymm(probs: &[Prob], band: usize, w: &mut MaskWork<i32, u8, 8>, ops: &mut [Vec<u8>]) -> Vec<Aln> {
    use std::arch::x86_64::*;
    let (np, n, m, width, open, ext, local_m, semi_m, single_m, global_m) = mask_setup!(probs, band, w, i32, u8, 8, NEG);
    let ld = |a: &[i32; 8]| _mm256_loadu_si256(a.as_ptr() as *const _);
    let openv = ld(&open); let extv = ld(&ext);
    let negv = _mm256_set1_epi32(NEG);
    let zero = _mm256_setzero_si256();
    let (c1, c4, cm2) = (_mm256_set1_epi32(1), _mm256_set1_epi32(4), _mm256_set1_epi32(-2));
    let mut bestv = _mm256_mask_blend_epi32(semi_m, zero, _mm256_set1_epi32(if n == 0 { 0 } else { NEG }));
    let mut biv = zero; let mut bjv = zero;
    for i in 1..=n {
        let base = i * width;
        let (ci, cp) = (center(i, n, m), center(i - 1, n, m));
        let d = ci - cp;
        let jlo = ci.saturating_sub(band);
        let jhi = m.min(ci + band);
        let (plo, phi) = (jlo + band + 1 - ci, jhi + band + 1 - ci);
        for p in 1..plo { w.h_cur[p] = [NEG; 8]; w.x_cur[p] = [NEG; 8]; }
        for p in phi + 1..=width { w.h_cur[p] = [NEG; 8]; w.x_cur[p] = [NEG; 8]; }
        let xa = ld(&w.ra[i - 1]); let xb = ld(&w.rb[i - 1]);
        let xs_m = _mm256_cmpeq_epi32_mask(ld(&w.rs[i - 1]), c1);
        let last_row = i == n;
        let iv = _mm256_set1_epi32(i as i32);
        let mut yl = negv;
        let mut jstart = jlo;
        if jlo == 0 {
            let p = band + 1 - ci;
            let gv = _mm256_sub_epi32(zero, _mm256_add_epi32(openv, _mm256_mullo_epi32(extv, _mm256_set1_epi32(i as i32 - 1))));
            _mm256_storeu_si256(w.h_cur[p].as_mut_ptr() as *mut _, _mm256_mask_blend_epi32(global_m, zero, gv));
            _mm256_storeu_si256(w.x_cur[p].as_mut_ptr() as *mut _, _mm256_mask_blend_epi32(global_m, negv, gv));
            *w.tr.get_unchecked_mut(base + p - 1) = MaskCell { z: !global_m, dge: 0, xge: global_m, xc: if i > 1 { global_m } else { 0 }, yc: 0 };
            if last_row || m == 0 {
                let upd = _mm256_mask_cmpgt_epi32_mask(semi_m, zero, bestv);
                bestv = _mm256_mask_blend_epi32(upd, bestv, zero);
                biv = _mm256_mask_blend_epi32(upd, biv, iv);
                bjv = _mm256_mask_blend_epi32(upd, bjv, zero);
            }
            jstart = 1;
        }
        let hp = w.h_prev.as_ptr(); let xp = w.x_prev.as_ptr(); let hc = w.h_cur.as_mut_ptr(); let xcp = w.x_cur.as_mut_ptr();
        let (cap, cbp, csp) = (w.ca.as_ptr(), w.cb.as_ptr(), w.cs.as_ptr());
        let trp = w.tr.as_mut_ptr();
        for j in jstart..=jhi {
            let p = j + band + 1 - ci;
            let pu = p + d;
            let (hu, xu) = if pu <= width { (ld(&*hp.add(pu)), ld(&*xp.add(pu))) } else { (negv, negv) };
            let pd = p + d - 1;
            let hd = if pd <= width { ld(&*hp.add(pd)) } else { negv };
            let hl = ld(&*hc.add(p - 1));
            let ya = ld(&*cap.add(j - 1)); let yb = ld(&*cbp.add(j - 1));
            let ys_m = _mm256_cmpeq_epi32_mask(ld(&*csp.add(j - 1)), c1);
            let both = xs_m & ys_m; let one = xs_m ^ ys_m;
            let eqa = _mm256_cmpeq_epi32_mask(xa, ya); let eqb = _mm256_cmpeq_epi32_mask(xb, yb);
            let adj = _mm256_cmpeq_epi32_mask(_mm256_abs_epi32(_mm256_sub_epi32(xa, ya)), c1);
            let partial = (single_m & adj) | (!single_m & (eqa | eqb));
            let mut s = _mm256_mask_blend_epi32(partial, cm2, c1);
            s = _mm256_mask_blend_epi32(eqa & eqb, s, c4);
            s = _mm256_mask_blend_epi32(one, s, cm2);
            s = _mm256_mask_blend_epi32(both, s, c1);
            let x1 = _mm256_sub_epi32(hu, openv); let x2 = _mm256_sub_epi32(xu, extv);
            let xc = _mm256_cmpgt_epi32_mask(x2, x1); let xv = _mm256_max_epi32(x1, x2);
            let y1 = _mm256_sub_epi32(hl, openv); let y2 = _mm256_sub_epi32(yl, extv);
            let yc = _mm256_cmpgt_epi32_mask(y2, y1); let yv = _mm256_max_epi32(y1, y2);
            let dd = _mm256_add_epi32(hd, s);
            let dge = _mm256_cmpge_epi32_mask(dd, xv) & _mm256_cmpge_epi32_mask(dd, yv);
            let xge = _mm256_cmpge_epi32_mask(xv, yv);
            let mut h = _mm256_mask_blend_epi32(dge, _mm256_mask_blend_epi32(xge, yv, xv), dd);
            let z = _mm256_mask_cmple_epi32_mask(local_m, h, zero);
            h = _mm256_mask_blend_epi32(z, h, zero);
            _mm256_storeu_si256((*hc.add(p)).as_mut_ptr() as *mut _, h);
            _mm256_storeu_si256((*xcp.add(p)).as_mut_ptr() as *mut _, xv);
            *trp.add(base + p - 1) = MaskCell { z, dge, xge, xc, yc };
            yl = yv;
            let cand = local_m | if last_row || j == m { semi_m } else { 0 };
            let upd = _mm256_mask_cmpgt_epi32_mask(cand, h, bestv);
            if upd != 0 {
                bestv = _mm256_mask_blend_epi32(upd, bestv, h);
                biv = _mm256_mask_blend_epi32(upd, biv, iv);
                bjv = _mm256_mask_blend_epi32(upd, bjv, _mm256_set1_epi32(j as i32));
            }
        }
        std::mem::swap(&mut w.h_prev, &mut w.h_cur);
        std::mem::swap(&mut w.x_prev, &mut w.x_cur);
    }
    let (mut best, mut bi, mut bj) = ([0i32; 8], [0i32; 8], [0i32; 8]);
    _mm256_storeu_si256(best.as_mut_ptr() as *mut _, bestv);
    _mm256_storeu_si256(bi.as_mut_ptr() as *mut _, biv);
    _mm256_storeu_si256(bj.as_mut_ptr() as *mut _, bjv);
    let gend = m + band + 1 - center(n, n, m);
    let mut out = Vec::with_capacity(np);
    let tr = &w.tr;
    for k in 0..np {
        let (i1, j1, score) = if (global_m >> k) & 1 == 0 { (bi[k] as usize, bj[k] as usize, best[k]) }
            else { (n, m, w.h_prev[gend][k]) };
        let (i0, j0) = traceback(i1, j1, n, m, band, |i, p| mask_trace_byte!(tr[i * width + p - 1], k), &mut ops[k]);
        out.push(Aln { score, i0, j0, i1, j1 });
    }
    out
}

/// 格子级专用：32 车道、16 位整数、512 位向量（饱和加减）。
/// 只在分数一定落在 16 位安全范围内时使用（见 grid_i16_safe），此时与 32 位实现每格数值相同。
pub const NEG16: i16 = -16000;
pub fn grid_i16_safe(n: usize, m: usize, open: i32, ext: i32) -> bool {
    // 可达格子的分数下界约为 -(4·len + open + ext·len)，上界 4·len；都远离 ±16000 时安全
    let len = n.max(m) as i32;
    4 * len + open + ext * len + 2 * len < 12_000
}
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
pub unsafe fn align_batch32_i16(probs: &[Prob], band: usize, w: &mut MaskWork<i16, u32, 32>, ops: &mut [Vec<u8>]) -> Vec<Aln> {
    use std::arch::x86_64::*;
    let (np, n, m, width, open, ext, local_m, semi_m, single_m, global_m) = mask_setup!(probs, band, w, i16, u32, 32, NEG16);
    let ld = |a: &[i16; 32]| _mm512_loadu_si512(a.as_ptr() as *const _);
    let openv = ld(&open); let extv = ld(&ext);
    let negv = _mm512_set1_epi16(NEG16);
    let zero = _mm512_setzero_si512();
    let (c1, c4, cm2) = (_mm512_set1_epi16(1), _mm512_set1_epi16(4), _mm512_set1_epi16(-2));
    let mut bestv = _mm512_mask_blend_epi16(semi_m, zero, _mm512_set1_epi16(if n == 0 { 0 } else { NEG16 }));
    let mut biv = zero; let mut bjv = zero;
    for i in 1..=n {
        let base = i * width;
        let (ci, cp) = (center(i, n, m), center(i - 1, n, m));
        let d = ci - cp;
        let jlo = ci.saturating_sub(band);
        let jhi = m.min(ci + band);
        let (plo, phi) = (jlo + band + 1 - ci, jhi + band + 1 - ci);
        for p in 1..plo { w.h_cur[p] = [NEG16; 32]; w.x_cur[p] = [NEG16; 32]; }
        for p in phi + 1..=width { w.h_cur[p] = [NEG16; 32]; w.x_cur[p] = [NEG16; 32]; }
        let xa = ld(&w.ra[i - 1]); let xb = ld(&w.rb[i - 1]);
        let xs_m = _mm512_cmpeq_epi16_mask(ld(&w.rs[i - 1]), c1);
        let last_row = i == n;
        let iv = _mm512_set1_epi16(i as i16);
        let mut yl = negv;
        let mut jstart = jlo;
        if jlo == 0 {
            let p = band + 1 - ci;
            let mut g = [0i16; 32];
            for k in 0..32 { g[k] = -(open[k] as i32 + (i as i32 - 1) * ext[k] as i32) as i16; }
            let gv = ld(&g);
            _mm512_storeu_si512(w.h_cur[p].as_mut_ptr() as *mut _, _mm512_mask_blend_epi16(global_m, zero, gv));
            _mm512_storeu_si512(w.x_cur[p].as_mut_ptr() as *mut _, _mm512_mask_blend_epi16(global_m, negv, gv));
            *w.tr.get_unchecked_mut(base + p - 1) = MaskCell { z: !global_m, dge: 0, xge: global_m, xc: if i > 1 { global_m } else { 0 }, yc: 0 };
            if last_row || m == 0 {
                let upd = _mm512_mask_cmpgt_epi16_mask(semi_m, zero, bestv);
                bestv = _mm512_mask_blend_epi16(upd, bestv, zero);
                biv = _mm512_mask_blend_epi16(upd, biv, iv);
                bjv = _mm512_mask_blend_epi16(upd, bjv, zero);
            }
            jstart = 1;
        }
        let hp = w.h_prev.as_ptr(); let xp = w.x_prev.as_ptr(); let hc = w.h_cur.as_mut_ptr(); let xcp = w.x_cur.as_mut_ptr();
        let (cap, cbp, csp) = (w.ca.as_ptr(), w.cb.as_ptr(), w.cs.as_ptr());
        let trp = w.tr.as_mut_ptr();
        for j in jstart..=jhi {
            let p = j + band + 1 - ci;
            let pu = p + d;
            let (hu, xu) = if pu <= width { (ld(&*hp.add(pu)), ld(&*xp.add(pu))) } else { (negv, negv) };
            let pd = p + d - 1;
            let hd = if pd <= width { ld(&*hp.add(pd)) } else { negv };
            let hl = ld(&*hc.add(p - 1));
            let ya = ld(&*cap.add(j - 1)); let yb = ld(&*cbp.add(j - 1));
            let ys_m = _mm512_cmpeq_epi16_mask(ld(&*csp.add(j - 1)), c1);
            let both = xs_m & ys_m; let one = xs_m ^ ys_m;
            let eqa = _mm512_cmpeq_epi16_mask(xa, ya); let eqb = _mm512_cmpeq_epi16_mask(xb, yb);
            let adj = _mm512_cmpeq_epi16_mask(_mm512_abs_epi16(_mm512_sub_epi16(xa, ya)), c1);
            let partial = (single_m & adj) | (!single_m & (eqa | eqb));
            let mut s = _mm512_mask_blend_epi16(partial, cm2, c1);
            s = _mm512_mask_blend_epi16(eqa & eqb, s, c4);
            s = _mm512_mask_blend_epi16(one, s, cm2);
            s = _mm512_mask_blend_epi16(both, s, c1);
            let x1 = _mm512_subs_epi16(hu, openv); let x2 = _mm512_subs_epi16(xu, extv);
            let xc = _mm512_cmpgt_epi16_mask(x2, x1); let xv = _mm512_max_epi16(x1, x2);
            let y1 = _mm512_subs_epi16(hl, openv); let y2 = _mm512_subs_epi16(yl, extv);
            let yc = _mm512_cmpgt_epi16_mask(y2, y1); let yv = _mm512_max_epi16(y1, y2);
            let dd = _mm512_adds_epi16(hd, s);
            let dge = _mm512_cmpge_epi16_mask(dd, xv) & _mm512_cmpge_epi16_mask(dd, yv);
            let xge = _mm512_cmpge_epi16_mask(xv, yv);
            let mut h = _mm512_mask_blend_epi16(dge, _mm512_mask_blend_epi16(xge, yv, xv), dd);
            let z = _mm512_mask_cmple_epi16_mask(local_m, h, zero);
            h = _mm512_mask_blend_epi16(z, h, zero);
            _mm512_storeu_si512((*hc.add(p)).as_mut_ptr() as *mut _, h);
            _mm512_storeu_si512((*xcp.add(p)).as_mut_ptr() as *mut _, xv);
            *trp.add(base + p - 1) = MaskCell { z, dge, xge, xc, yc };
            yl = yv;
            let cand = local_m | if last_row || j == m { semi_m } else { 0 };
            let upd = _mm512_mask_cmpgt_epi16_mask(cand, h, bestv);
            if upd != 0 {
                bestv = _mm512_mask_blend_epi16(upd, bestv, h);
                biv = _mm512_mask_blend_epi16(upd, biv, iv);
                bjv = _mm512_mask_blend_epi16(upd, bjv, _mm512_set1_epi16(j as i16));
            }
        }
        std::mem::swap(&mut w.h_prev, &mut w.h_cur);
        std::mem::swap(&mut w.x_prev, &mut w.x_cur);
    }
    let (mut best, mut bi, mut bj) = ([0i16; 32], [0i16; 32], [0i16; 32]);
    _mm512_storeu_si512(best.as_mut_ptr() as *mut _, bestv);
    _mm512_storeu_si512(bi.as_mut_ptr() as *mut _, biv);
    _mm512_storeu_si512(bj.as_mut_ptr() as *mut _, bjv);
    let gend = m + band + 1 - center(n, n, m);
    let mut out = Vec::with_capacity(np);
    let tr = &w.tr;
    for k in 0..np {
        let (i1, j1, score) = if (global_m >> k) & 1 == 0 { (bi[k] as usize, bj[k] as usize, best[k] as i32) }
            else { (n, m, w.h_prev[gend][k] as i32) };
        let (i0, j0) = traceback(i1, j1, n, m, band, |i, p| mask_trace_byte!(tr[i * width + p - 1], k), &mut ops[k]);
        out.push(Aln { score, i0, j0, i1, j1 });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::letters::build_score_table_grid;

    fn lcg(s: &mut u64) -> u64 { *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); *s >> 33 }

    #[test]
    fn hand_example() {
        // GATC 对 GTC（单属性 4 档字母，相邻档 +1 改为全不同才能对上手算：这里用 3×3 的对角字母）
        let mut t = vec![-1i32; SIZE * SIZE];
        for x in 0..SIZE { t[x * SIZE + x] = 1; }
        let (a, b) = ([1u8, 2, 3, 4], [1u8, 3, 4]);
        let (mut w, mut ops) = (Work::default(), Vec::new());
        let r = align_scalar(&a, &b, &t, 1, 1, 4, Mode::Global, &mut w, &mut ops);
        assert_eq!(r.score, 2);
        assert_eq!(ops, vec![OP_D, OP_X, OP_D, OP_D]);
    }

    /// 批量实现与基准实现：分数、起止点、路径完全相同（含不等长、斜带、三种模式、静默字母）。
    #[test]
    fn batch_equals_scalar() {
        let mut seed = 7u64;
        let grids = [(3usize, 3usize), (5, 1), (2, 3), (4, 2)];
        for round in 0..60 {
            let n = 5 + (lcg(&mut seed) % 120) as usize;
            let m = if round % 3 == 0 { n } else { 5 + (lcg(&mut seed) % 120) as usize };
            let band = if n == m { (n / 4).max(1) } else { 6 + (lcg(&mut seed) % 20) as usize };
            let ratio = (n.max(m) as f64 / n.min(m) as f64).ceil() as usize;
            if band < ratio + 1 { continue; }
            let mut seqs = Vec::new();
            for k in 0..16 {
                let g = grids[k % 4];
                let nl = (g.0 * g.1) as u64;
                let silent = k % 2 == 0;
                let mk = |len: usize, s: &mut u64| -> Vec<u8> { (0..len).map(|_| {
                    let r = lcg(s) % (nl + if silent { 2 } else { 0 });
                    if r >= nl { 0 } else { (r + 1) as u8 } }).collect() };
                let a = mk(n, &mut seed); let b = mk(m, &mut seed);
                seqs.push((a, b, g));
            }
            let modes = [Mode::Global, Mode::Semi, Mode::Local];
            let probs: Vec<Prob> = (0..16).map(|k| Prob { a: &seqs[k].0, b: &seqs[k].1, grid: seqs[k].2,
                mode: modes[k % 3], open: if k % 2 == 0 { 3 } else { 8 }, ext: if k % 2 == 0 { 1 } else { 2 } }).collect();
            let mut bw = BatchWork::<16>::default();
            let mut bops = vec![Vec::new(); 16];
            let res = align_batch(&probs, band, &mut bw, &mut bops);
            // 显式向量版（不足 16 题时补齐车道的情形也测：只取前 11 题）
            let mut vops = vec![Vec::new(); 16];
            let vres = if has_avx512() { unsafe { align_batch16_avx512(&probs, band, &mut bw, &mut vops) } } else { res.clone() };
            let mut pops = vec![Vec::new(); 11];
            let pres = if has_avx512() { unsafe { align_batch16_avx512(&probs[..11], band, &mut bw, &mut pops) } } else { res[..11].to_vec() };
            assert!(has_avx512(), "测试机应支持 512 位向量");
            // 8 车道 32 位（逐笔级用）：前 8 题
            let mut w8 = MaskWork::<i32, u8, 8>::default();
            let mut o8 = vec![Vec::new(); 8];
            let r8 = unsafe { align_batch8_ymm(&probs[..8], band, &mut w8, &mut o8) };
            // 32 车道 16 位（格子级用）：全部 16 题（另外 16 个车道补齐）
            let mut w32 = MaskWork::<i16, u32, 32>::default();
            let mut o32 = vec![Vec::new(); 16];
            let r32 = unsafe { align_batch32_i16(&probs, band, &mut w32, &mut o32) };
            for k in 0..16 {
                if k < 8 { assert_eq!(res[k], r8[k], "ymm round {round} lane {k}"); assert_eq!(bops[k], o8[k]); }
                assert_eq!(res[k], r32[k], "i16 round {round} lane {k}"); assert_eq!(bops[k], o32[k], "i16 ops {round} {k}");
            }
            for k in 0..16 {
                assert_eq!(res[k], vres[k], "avx512 round {round} lane {k}");
                assert_eq!(bops[k], vops[k], "avx512 ops round {round} lane {k}");
                if k < 11 { assert_eq!(res[k], pres[k]); assert_eq!(bops[k], pops[k]); }
                let tbl = build_score_table_grid(probs[k].grid);
                let (mut w, mut ops) = (Work::default(), Vec::new());
                let r = align_scalar(probs[k].a, probs[k].b, &tbl, probs[k].open, probs[k].ext, band, probs[k].mode, &mut w, &mut ops);
                assert_eq!(r, res[k], "round {round} lane {k} n={n} m={m} band={band}");
                assert_eq!(ops, bops[k], "round {round} lane {k}");
            }
            let _ = &vops;
        }
    }
}
