//! 提速引擎：把同长度、同参数的“只算最高分”比对 32 条一组并排算（每条序列占一个 16 位整数通道）。
//! 每个通道做的加减与取最大和标量版 dp::nested_score / forward_score / rect_score 完全相同，
//! 分数都远小于 16 位整数上限，所以结果逐位相同。
use crate::dp::{Tab, K, NEG, NEG_HALF};

pub const L: usize = 32;
type V = [i16; L];
const NEG16: i16 = NEG as i16;
const NEGH16: i16 = NEG_HALF as i16;

#[inline(always)]
fn splat(x: i16) -> V {
    [x; L]
}

/// 每行把各通道的打分行转成“按字母分组”的表：rows[b][l] = tab_l[a_l][b]
#[inline(always)]
fn build_rows(rows: &mut [V; K], tabs: &[Tab16], a: &[u8; L]) {
    for b in 0..K {
        for l in 0..L {
            rows[b][l] = tabs[l][a[l] as usize][b];
        }
    }
}

#[inline(always)]
fn lookup(rows: &[V; K], b: &[u8; L]) -> V {
    let mut p = splat(0);
    for k in 0..K {
        for l in 0..L {
            p[l] = if b[l] == k as u8 { rows[k][l] } else { p[l] };
        }
    }
    p
}

pub type Tab16 = [[i16; K]; K];
pub fn tab16(t: &Tab) -> Tab16 {
    let mut o = [[0i16; K]; K];
    for a in 0..K {
        for b in 0..K {
            o[a][b] = t[a][b] as i16;
        }
    }
    o
}

fn transpose(seqs: &[&[u8]], n: usize) -> Vec<[u8; L]> {
    let mut out = vec![[0u8; L]; n];
    for (l, s) in seqs.iter().enumerate() {
        for j in 0..n {
            out[j][l] = s[j];
        }
    }
    out
}

/// 嵌套比对最高分（批量）。seqs/tabs 长度 ≤ 32，所有序列长度相同
pub fn nested_score_batch(seqs: &[&[u8]], tabs: &[Tab16], o: i16, e: i16, minsep: usize) -> Vec<i32> {
    if std::arch::is_x86_feature_detected!("avx512bw") {
        unsafe { simd::nested(seqs, tabs, o, e, minsep) }
    } else {
        nested_score_batch_generic(seqs, tabs, o, e, minsep)
    }
}
pub fn nested_score_batch_generic(seqs: &[&[u8]], tabs: &[Tab16], o: i16, e: i16, minsep: usize) -> Vec<i32> {
    let m = seqs.len();
    let n = seqs[0].len();
    let mut ss: Vec<&[u8]> = seqs.to_vec();
    let mut tt: Vec<Tab16> = tabs.to_vec();
    while ss.len() < L {
        ss.push(seqs[0]);
        tt.push(tabs[0]);
    }
    let st = transpose(&ss, n);
    let mut ph = vec![splat(0); n + 1];
    let mut ch = vec![splat(0); n + 1];
    let mut pf = vec![splat(NEG16); n + 1];
    let mut cf = vec![splat(NEG16); n + 1];
    let mut rows = [splat(0); K];
    let mut best = splat(0);
    let ms = minsep.max(1);
    for i in (0..n).rev() {
        let start = i + ms;
        if start >= n {
            continue;
        }
        build_rows(&mut rows, &tt, &st[i]);
        ch[start - 1] = splat(0);
        cf[start - 1] = splat(NEG16);
        let mut lh = splat(0);
        let mut ee = splat(NEG16);
        for j in start..n {
            let p = lookup(&rows, &st[j]);
            let up = ph[j];
            let upf = pf[j];
            let dg = ph[j - 1];
            let mut h = splat(0);
            let mut ff = splat(0);
            for l in 0..L {
                ee[l] = (lh[l] - o).max(ee[l] - e);
                ff[l] = (up[l] - o).max(upf[l] - e);
                h[l] = (dg[l] + p[l]).max(ff[l]).max(ee[l]).max(0);
                best[l] = best[l].max(h[l]);
            }
            ch[j] = h;
            cf[j] = ff;
            lh = h;
        }
        std::mem::swap(&mut ph, &mut ch);
        std::mem::swap(&mut pf, &mut cf);
    }
    best[..m].iter().map(|&x| x as i32).collect()
}

/// 同向比对最高分（批量，含“两段不重叠”的起点记录）
pub fn forward_score_batch(seqs: &[&[u8]], tabs: &[Tab16], o: i16, e: i16, minsep: usize) -> Vec<i32> {
    if std::arch::is_x86_feature_detected!("avx512bw") {
        unsafe { simd::forward(seqs, tabs, o, e, minsep) }
    } else {
        forward_score_batch_generic(seqs, tabs, o, e, minsep)
    }
}
pub fn forward_score_batch_generic(seqs: &[&[u8]], tabs: &[Tab16], o: i16, e: i16, minsep: usize) -> Vec<i32> {
    let m = seqs.len();
    let n = seqs[0].len();
    let mut ss: Vec<&[u8]> = seqs.to_vec();
    let mut tt: Vec<Tab16> = tabs.to_vec();
    while ss.len() < L {
        ss.push(seqs[0]);
        tt.push(tabs[0]);
    }
    let st = transpose(&ss, n);
    let z = splat(0);
    let (mut ph, mut ch) = (vec![z; n + 1], vec![z; n + 1]);
    let (mut pj, mut cj) = (vec![z; n + 1], vec![z; n + 1]);
    let (mut pf, mut cf) = (vec![splat(NEG16); n + 1], vec![splat(NEG16); n + 1]);
    let (mut pfj, mut cfj) = (vec![z; n + 1], vec![z; n + 1]);
    let mut rows = [z; K];
    let mut best = z;
    let ms = minsep.max(1);
    for i in 0..n {
        let start = i + ms;
        if start >= n {
            break;
        }
        build_rows(&mut rows, &tt, &st[i]);
        let ii = i as i16;
        let first = i == 0;
        ch[start - 1] = z;
        cj[start - 1] = z;
        cf[start - 1] = splat(NEG16);
        cfj[start - 1] = z;
        let (mut lh, mut lj) = (z, z);
        let (mut ev, mut ej) = (splat(NEG16), z);
        for j in start..n {
            let p = lookup(&rows, &st[j]);
            let jj = j as i16;
            let (up, uj, upf, upfj) = if first { (z, z, splat(NEG16), z) } else { (ph[j], pj[j], pf[j], pfj[j]) };
            let (dg, dj) = if first { (z, z) } else { (ph[j - 1], pj[j - 1]) };
            let mut h = z;
            let mut hj = z;
            let mut fv = z;
            let mut fj = z;
            for l in 0..L {
                let eo = if lh[l] > 0 && lj[l] > ii { lh[l] - o } else { NEG16 };
                let ex = if ev[l] > NEGH16 && ej[l] > ii { ev[l] - e } else { NEG16 };
                let take_o = eo >= ex;
                ev[l] = if take_o { eo } else { ex };
                ej[l] = if take_o { lj[l] } else { ej[l] };
                let fo = if up[l] > 0 && uj[l] > ii { up[l] - o } else { NEG16 };
                let fx = if upf[l] > NEGH16 && upfj[l] > ii { upf[l] - e } else { NEG16 };
                let take_fo = fo >= fx;
                fv[l] = if take_fo { fo } else { fx };
                fj[l] = if take_fo { uj[l] } else { upfj[l] };
                let dvalid = dg[l] > 0 && dj[l] > ii;
                let dv = if dvalid { dg[l] } else { 0 } + p[l];
                let dj2 = if dvalid { dj[l] } else { jj };
                let c1 = dv >= fv[l] && dv >= ev[l];
                let c2 = fv[l] >= ev[l];
                let hv = if c1 { dv } else if c2 { fv[l] } else { ev[l] };
                let hjv = if c1 { dj2 } else if c2 { fj[l] } else { ej[l] };
                let pos = hv > 0;
                h[l] = if pos { hv } else { 0 };
                hj[l] = if pos { hjv } else { 0 };
                best[l] = best[l].max(h[l]);
            }
            ch[j] = h;
            cj[j] = hj;
            cf[j] = fv;
            cfj[j] = fj;
            lh = h;
            lj = hj;
        }
        std::mem::swap(&mut ph, &mut ch);
        std::mem::swap(&mut pj, &mut cj);
        std::mem::swap(&mut pf, &mut cf);
        std::mem::swap(&mut pfj, &mut cfj);
    }
    best[..m].iter().map(|&x| x as i32).collect()
}

/// 两序列局部比对最高分（批量，线性空位）
pub fn rect_score_batch(as_: &[&[u8]], bs: &[&[u8]], tabs: &[Tab16], g: i16) -> Vec<i32> {
    if std::arch::is_x86_feature_detected!("avx512bw") {
        unsafe { simd::rect(as_, bs, tabs, g) }
    } else {
        rect_score_batch_generic(as_, bs, tabs, g)
    }
}
pub fn rect_score_batch_generic(as_: &[&[u8]], bs: &[&[u8]], tabs: &[Tab16], g: i16) -> Vec<i32> {
    let m = as_.len();
    let na = as_[0].len();
    let nb = bs[0].len();
    let mut aa: Vec<&[u8]> = as_.to_vec();
    let mut bb: Vec<&[u8]> = bs.to_vec();
    let mut tt: Vec<Tab16> = tabs.to_vec();
    while aa.len() < L {
        aa.push(as_[0]);
        bb.push(bs[0]);
        tt.push(tabs[0]);
    }
    let at = transpose(&aa, na);
    let bt = transpose(&bb, nb);
    let z = splat(0);
    let mut ph = vec![z; nb + 1];
    let mut ch = vec![z; nb + 1];
    let mut rows = [z; K];
    let mut best = z;
    for x in 0..na {
        build_rows(&mut rows, &tt, &at[x]);
        ch[0] = z;
        for y in 1..=nb {
            let p = lookup(&rows, &bt[y - 1]);
            let dg = ph[y - 1];
            let up = ph[y];
            let lf = ch[y - 1];
            let mut h = z;
            for l in 0..L {
                h[l] = (dg[l] + p[l]).max(up[l] - g).max(lf[l] - g).max(0);
                best[l] = best[l].max(h[l]);
            }
            ch[y] = h;
        }
        std::mem::swap(&mut ph, &mut ch);
    }
    best[..m].iter().map(|&x| x as i32).collect()
}

/// AVX-512 版：一个 512 位寄存器装 32 个 16 位通道。逻辑与上面的通用版逐条对应。
mod simd {
    use super::{Tab16, K, L, NEG16, NEGH16};
    use std::arch::x86_64::*;

    type M = __m512i;

    #[inline(always)]
    unsafe fn ld(p: &[i16; L]) -> M {
        _mm512_loadu_si512(p.as_ptr() as *const _)
    }
    #[inline(always)]
    unsafe fn st(p: &mut [i16; L], v: M) {
        _mm512_storeu_si512(p.as_mut_ptr() as *mut _, v)
    }

    /// 补齐到 32 通道并转置：每列一个寄存器装 32 条序列在该位置的字母；同时预先算好每列“字母 == k”的掩码
    fn prep(seqs: &[&[u8]], n: usize) -> (Vec<[i16; L]>, Vec<[u32; K]>) {
        let mut t = vec![[0i16; L]; n];
        let mut masks = vec![[0u32; K]; n];
        for j in 0..n {
            for l in 0..L {
                let s = if l < seqs.len() { seqs[l] } else { seqs[0] };
                let x = s[j];
                t[j][l] = x as i16;
                masks[j][x as usize] |= 1u32 << l;
            }
        }
        (t, masks)
    }
    fn pad_tabs(tabs: &[Tab16]) -> Vec<Tab16> {
        (0..L).map(|l| if l < tabs.len() { tabs[l] } else { tabs[0] }).collect()
    }
    #[inline(always)]
    unsafe fn rows_of(rows: &mut [M; K], tabs: &[Tab16], a: &[i16; L]) {
        let mut tmp = [[0i16; L]; K];
        for b in 0..K {
            for l in 0..L {
                tmp[b][l] = tabs[l][a[l] as usize][b];
            }
            rows[b] = ld(&tmp[b]);
        }
    }
    #[inline(always)]
    unsafe fn lookup(rows: &[M; K], m: &[u32; K]) -> M {
        let mut p = rows[0];
        for k in 1..K {
            p = _mm512_mask_blend_epi16(m[k], p, rows[k]);
        }
        p
    }
    fn out(best: &[i16; L], m: usize) -> Vec<i32> {
        best[..m].iter().map(|&x| x as i32).collect()
    }

    #[target_feature(enable = "avx512f,avx512bw")]
    pub unsafe fn nested(seqs: &[&[u8]], tabs: &[Tab16], o: i16, e: i16, minsep: usize) -> Vec<i32> {
        let m = seqs.len();
        let n = seqs[0].len();
        let (st_, masks) = prep(seqs, n);
        let tt = pad_tabs(tabs);
        let zero = _mm512_setzero_si512();
        let negv = _mm512_set1_epi16(NEG16);
        let ov = _mm512_set1_epi16(o);
        let ev = _mm512_set1_epi16(e);
        let mut ph = vec![[0i16; L]; n + 1];
        let mut ch = vec![[0i16; L]; n + 1];
        let mut pf = vec![[NEG16; L]; n + 1];
        let mut cf = vec![[NEG16; L]; n + 1];
        let mut rows = [zero; K];
        let mut best = zero;
        let ms = minsep.max(1);
        for i in (0..n).rev() {
            let start = i + ms;
            if start >= n {
                continue;
            }
            rows_of(&mut rows, &tt, &st_[i]);
            st(&mut ch[start - 1], zero);
            st(&mut cf[start - 1], negv);
            let mut lh = zero;
            let mut ee = negv;
            let mut dg = ld(&ph[start - 1]);
            for j in start..n {
                let p = lookup(&rows, &masks[j]);
                let up = ld(&ph[j]);
                let upf = ld(&pf[j]);
                ee = _mm512_max_epi16(_mm512_sub_epi16(lh, ov), _mm512_sub_epi16(ee, ev));
                let ff = _mm512_max_epi16(_mm512_sub_epi16(up, ov), _mm512_sub_epi16(upf, ev));
                let h = _mm512_max_epi16(
                    _mm512_max_epi16(_mm512_add_epi16(dg, p), ff),
                    _mm512_max_epi16(ee, zero),
                );
                best = _mm512_max_epi16(best, h);
                st(&mut ch[j], h);
                st(&mut cf[j], ff);
                lh = h;
                dg = up;
            }
            std::mem::swap(&mut ph, &mut ch);
            std::mem::swap(&mut pf, &mut cf);
        }
        let mut b = [0i16; L];
        st(&mut b, best);
        out(&b, m)
    }

    #[target_feature(enable = "avx512f,avx512bw")]
    pub unsafe fn forward(seqs: &[&[u8]], tabs: &[Tab16], o: i16, e: i16, minsep: usize) -> Vec<i32> {
        let m = seqs.len();
        let n = seqs[0].len();
        let (st_, masks) = prep(seqs, n);
        let tt = pad_tabs(tabs);
        let zero = _mm512_setzero_si512();
        let negv = _mm512_set1_epi16(NEG16);
        let neghv = _mm512_set1_epi16(NEGH16);
        let ov = _mm512_set1_epi16(o);
        let ev_ = _mm512_set1_epi16(e);
        let z = [0i16; L];
        let (mut ph, mut ch) = (vec![z; n + 1], vec![z; n + 1]);
        let (mut pj, mut cj) = (vec![z; n + 1], vec![z; n + 1]);
        let (mut pf, mut cf) = (vec![[NEG16; L]; n + 1], vec![[NEG16; L]; n + 1]);
        let (mut pfj, mut cfj) = (vec![z; n + 1], vec![z; n + 1]);
        let mut rows = [zero; K];
        let mut best = zero;
        let ms = minsep.max(1);
        for i in 0..n {
            let start = i + ms;
            if start >= n {
                break;
            }
            rows_of(&mut rows, &tt, &st_[i]);
            let iv = _mm512_set1_epi16(i as i16);
            let first = i == 0;
            st(&mut ch[start - 1], zero);
            st(&mut cj[start - 1], zero);
            st(&mut cf[start - 1], negv);
            st(&mut cfj[start - 1], zero);
            let (mut lh, mut lj) = (zero, zero);
            let (mut evv, mut ej) = (negv, zero);
            for j in start..n {
                let p = lookup(&rows, &masks[j]);
                let jv = _mm512_set1_epi16(j as i16);
                let (up, uj, upf, upfj, dg, dj) = if first {
                    (zero, zero, negv, zero, zero, zero)
                } else {
                    (ld(&ph[j]), ld(&pj[j]), ld(&pf[j]), ld(&pfj[j]), ld(&ph[j - 1]), ld(&pj[j - 1]))
                };
                // 左
                let lvalid = _mm512_cmpgt_epi16_mask(lh, zero) & _mm512_cmpgt_epi16_mask(lj, iv);
                let evalid = _mm512_cmpgt_epi16_mask(evv, neghv) & _mm512_cmpgt_epi16_mask(ej, iv);
                let eo = _mm512_mask_blend_epi16(lvalid, negv, _mm512_sub_epi16(lh, ov));
                let ex = _mm512_mask_blend_epi16(evalid, negv, _mm512_sub_epi16(evv, ev_));
                let take_o = _mm512_cmpge_epi16_mask(eo, ex);
                evv = _mm512_mask_blend_epi16(take_o, ex, eo);
                ej = _mm512_mask_blend_epi16(take_o, ej, lj);
                // 上
                let uvalid = _mm512_cmpgt_epi16_mask(up, zero) & _mm512_cmpgt_epi16_mask(uj, iv);
                let fvalid = _mm512_cmpgt_epi16_mask(upf, neghv) & _mm512_cmpgt_epi16_mask(upfj, iv);
                let fo = _mm512_mask_blend_epi16(uvalid, negv, _mm512_sub_epi16(up, ov));
                let fx = _mm512_mask_blend_epi16(fvalid, negv, _mm512_sub_epi16(upf, ev_));
                let take_fo = _mm512_cmpge_epi16_mask(fo, fx);
                let fv = _mm512_mask_blend_epi16(take_fo, fx, fo);
                let fj = _mm512_mask_blend_epi16(take_fo, upfj, uj);
                // 对角
                let dvalid = _mm512_cmpgt_epi16_mask(dg, zero) & _mm512_cmpgt_epi16_mask(dj, iv);
                let dv = _mm512_add_epi16(_mm512_maskz_mov_epi16(dvalid, dg), p);
                let dj2 = _mm512_mask_blend_epi16(dvalid, jv, dj);
                let c1 = _mm512_cmpge_epi16_mask(dv, fv) & _mm512_cmpge_epi16_mask(dv, evv);
                let c2 = _mm512_cmpge_epi16_mask(fv, evv);
                let hv0 = _mm512_mask_blend_epi16(c2, evv, fv);
                let hj0 = _mm512_mask_blend_epi16(c2, ej, fj);
                let hv1 = _mm512_mask_blend_epi16(c1, hv0, dv);
                let hj1 = _mm512_mask_blend_epi16(c1, hj0, dj2);
                let pos = _mm512_cmpgt_epi16_mask(hv1, zero);
                let h = _mm512_maskz_mov_epi16(pos, hv1);
                let hj = _mm512_maskz_mov_epi16(pos, hj1);
                best = _mm512_max_epi16(best, h);
                st(&mut ch[j], h);
                st(&mut cj[j], hj);
                st(&mut cf[j], fv);
                st(&mut cfj[j], fj);
                lh = h;
                lj = hj;
            }
            std::mem::swap(&mut ph, &mut ch);
            std::mem::swap(&mut pj, &mut cj);
            std::mem::swap(&mut pf, &mut cf);
            std::mem::swap(&mut pfj, &mut cfj);
        }
        let mut b = [0i16; L];
        st(&mut b, best);
        out(&b, m)
    }

    #[target_feature(enable = "avx512f,avx512bw")]
    pub unsafe fn rect(as_: &[&[u8]], bs: &[&[u8]], tabs: &[Tab16], g: i16) -> Vec<i32> {
        let m = as_.len();
        let na = as_[0].len();
        let nb = bs[0].len();
        let (at, _) = prep(as_, na);
        let (_, bm) = prep(bs, nb);
        let tt = pad_tabs(tabs);
        let zero = _mm512_setzero_si512();
        let gv = _mm512_set1_epi16(g);
        let mut ph = vec![[0i16; L]; nb + 1];
        let mut ch = vec![[0i16; L]; nb + 1];
        let mut rows = [zero; K];
        let mut best = zero;
        for x in 0..na {
            rows_of(&mut rows, &tt, &at[x]);
            st(&mut ch[0], zero);
            let mut left = zero;
            let mut dg = ld(&ph[0]);
            for y in 1..=nb {
                let p = lookup(&rows, &bm[y - 1]);
                let up = ld(&ph[y]);
                let h = _mm512_max_epi16(
                    _mm512_max_epi16(_mm512_add_epi16(dg, p), _mm512_sub_epi16(up, gv)),
                    _mm512_max_epi16(_mm512_sub_epi16(left, gv), zero),
                );
                best = _mm512_max_epi16(best, h);
                st(&mut ch[y], h);
                left = h;
                dg = up;
            }
            std::mem::swap(&mut ph, &mut ch);
        }
        let mut b = [0i16; L];
        st(&mut b, best);
        out(&b, m)
    }
}
