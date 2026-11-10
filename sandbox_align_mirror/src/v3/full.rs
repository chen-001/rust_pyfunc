//! 提速引擎的“完整路径”比对：先用 AVX-512 一次算 32 条序列的分数，同时记下每格的选择方向；
//! 再从最高分那一格往回走，复原路径上的配对数、匹配数、空位数、先买后卖、配对量与起点。
//! 选择规则、最高分的先到先得、配对量的累加顺序（从中心往外）都与 dp.rs 的标量版一致，结果逐字节相同。
use crate::batch::{tab16, Tab16, L};
use crate::dp::{self, side_of, Best, Cell, Tab, K, NEG, NEG_HALF};
use std::arch::x86_64::*;

#[derive(Clone, Copy, Default)]
pub struct Dir {
    pub c1: u32,
    pub c2: u32,
    pub eo: u32,
    pub fo: u32,
    pub dp: u32,
}
#[inline]
fn bit(m: u32, l: usize) -> bool {
    (m >> l) & 1 == 1
}

pub struct Lane<'a> {
    pub a: &'a [u8],
    pub s: &'a [u8],
    pub vol: &'a [f32],
    pub tab: &'a Tab,
    pub center: bool,
}

pub struct Out {
    pub best: Best,
    pub center: Vec<i32>,
}

type M = __m512i;
#[inline(always)]
unsafe fn ld(p: &[i16; L]) -> M {
    _mm512_loadu_si512(p.as_ptr() as *const _)
}
#[inline(always)]
unsafe fn st(p: &mut [i16; L], v: M) {
    _mm512_storeu_si512(p.as_mut_ptr() as *mut _, v)
}

fn prep(seqs: &[&[u8]], n: usize) -> (Vec<[i16; L]>, Vec<[u32; K]>) {
    let mut t = vec![[0i16; L]; n];
    let mut masks = vec![[0u32; K]; n];
    for j in 0..n {
        for l in 0..L {
            let s = if l < seqs.len() { seqs[l] } else { seqs[0] };
            t[j][l] = s[j] as i16;
            masks[j][s[j] as usize] |= 1u32 << l;
        }
    }
    (t, masks)
}
#[inline(always)]
unsafe fn rows_of(rows: &mut [M; K], tabs: &[Tab16], a: &[i16; L]) {
    let mut tmp = [[0i16; L]; K];
    for b in 0..K {
        for l in 0..L {
            let t = if l < tabs.len() { &tabs[l] } else { &tabs[0] };
            tmp[b][l] = t[a[l] as usize][b];
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

struct Raw {
    best: [i16; L],
    bi: [i16; L],
    bj: [i16; L],
    dirs: Vec<Dir>,
    centers: Vec<Vec<i32>>,
}

#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn nested_raw(lanes: &[Lane], o: i16, e: i16, minsep: usize) -> Raw {
    let n = lanes[0].s.len();
    let seqs: Vec<&[u8]> = lanes.iter().map(|x| x.s).collect();
    let tabs: Vec<Tab16> = lanes.iter().map(|x| tab16(x.tab)).collect();
    let (st_, masks) = prep(&seqs, n);
    let zero = _mm512_setzero_si512();
    let negv = _mm512_set1_epi16(NEG as i16);
    let ov = _mm512_set1_epi16(o);
    let ev = _mm512_set1_epi16(e);
    let z = [0i16; L];
    let (mut ph, mut ch) = (vec![z; n + 1], vec![z; n + 1]);
    let (mut pk, mut ck) = (vec![z; n + 1], vec![z; n + 1]);
    let (mut pf, mut cf) = (vec![[NEG as i16; L]; n + 1], vec![[NEG as i16; L]; n + 1]);
    let (mut pfk, mut cfk) = (vec![z; n + 1], vec![z; n + 1]);
    let mut dirs = vec![Dir::default(); n * n];
    let cl: Vec<usize> = (0..lanes.len()).filter(|&l| lanes[l].center).collect();
    let mut centers: Vec<Vec<i32>> = (0..lanes.len()).map(|l| if lanes[l].center { vec![0; 2 * n + 1] } else { Vec::new() }).collect();
    let mut rows = [zero; K];
    let (mut best, mut bi, mut bj) = (zero, zero, zero);
    let ms = minsep.max(1);
    for i in (0..n).rev() {
        let start = i + ms;
        if start >= n {
            continue;
        }
        rows_of(&mut rows, &tabs, &st_[i]);
        let iv = _mm512_set1_epi16(i as i16);
        st(&mut ch[start - 1], zero);
        st(&mut ck[start - 1], zero);
        st(&mut cf[start - 1], negv);
        st(&mut cfk[start - 1], zero);
        let (mut lh, mut lk) = (zero, zero);
        let (mut ee, mut ek) = (negv, zero);
        for j in start..n {
            let p = lookup(&rows, &masks[j]);
            let (up, uk, upf, upfk) = (ld(&ph[j]), ld(&pk[j]), ld(&pf[j]), ld(&pfk[j]));
            let (dg, dk) = (ld(&ph[j - 1]), ld(&pk[j - 1]));
            let eo_v = _mm512_sub_epi16(lh, ov);
            let ex_v = _mm512_sub_epi16(ee, ev);
            let eo = _mm512_cmpge_epi16_mask(eo_v, ex_v);
            ee = _mm512_mask_blend_epi16(eo, ex_v, eo_v);
            ek = _mm512_mask_blend_epi16(eo, ek, lk);
            let fo_v = _mm512_sub_epi16(up, ov);
            let fx_v = _mm512_sub_epi16(upf, ev);
            let fo = _mm512_cmpge_epi16_mask(fo_v, fx_v);
            let ff = _mm512_mask_blend_epi16(fo, fx_v, fo_v);
            let fk = _mm512_mask_blend_epi16(fo, upfk, uk);
            let dv = _mm512_add_epi16(dg, p);
            let dpm = _mm512_cmpgt_epi16_mask(dg, zero);
            let key_new = _mm512_add_epi16(iv, _mm512_set1_epi16(j as i16));
            let dkey = _mm512_mask_blend_epi16(dpm, key_new, dk);
            let c1 = _mm512_cmpge_epi16_mask(dv, ff) & _mm512_cmpge_epi16_mask(dv, ee);
            let c2 = _mm512_cmpge_epi16_mask(ff, ee);
            let hv = _mm512_mask_blend_epi16(c1, _mm512_mask_blend_epi16(c2, ee, ff), dv);
            let hk = _mm512_mask_blend_epi16(c1, _mm512_mask_blend_epi16(c2, ek, fk), dkey);
            let pos = _mm512_cmpgt_epi16_mask(hv, zero);
            let h = _mm512_maskz_mov_epi16(pos, hv);
            let hk = _mm512_maskz_mov_epi16(pos, hk);
            let better = _mm512_cmpgt_epi16_mask(h, best);
            best = _mm512_mask_blend_epi16(better, best, h);
            bi = _mm512_mask_blend_epi16(better, bi, iv);
            bj = _mm512_mask_blend_epi16(better, bj, _mm512_set1_epi16(j as i16));
            st(&mut ch[j], h);
            st(&mut ck[j], hk);
            st(&mut cf[j], ff);
            st(&mut cfk[j], fk);
            dirs[i * n + j] = Dir { c1, c2, eo, fo, dp: dpm };
            lh = h;
            lk = hk;
        }
        // 镜像中心曲线：只对需要的通道逐格更新
        if !cl.is_empty() {
            for j in start..n {
                for &l in &cl {
                    let v = ch[j][l] as i32;
                    if v > 0 {
                        let k = ck[j][l] as usize;
                        if v > centers[l][k] {
                            centers[l][k] = v;
                        }
                    }
                }
            }
        }
        std::mem::swap(&mut ph, &mut ch);
        std::mem::swap(&mut pk, &mut ck);
        std::mem::swap(&mut pf, &mut cf);
        std::mem::swap(&mut pfk, &mut cfk);
    }
    let mut r = Raw { best: z, bi: z, bj: z, dirs, centers };
    st(&mut r.best, best);
    st(&mut r.bi, bi);
    st(&mut r.bj, bj);
    r
}

#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn forward_raw(lanes: &[Lane], o: i16, e: i16, minsep: usize) -> Raw {
    let n = lanes[0].s.len();
    let seqs: Vec<&[u8]> = lanes.iter().map(|x| x.s).collect();
    let tabs: Vec<Tab16> = lanes.iter().map(|x| tab16(x.tab)).collect();
    let (st_, masks) = prep(&seqs, n);
    let zero = _mm512_setzero_si512();
    let negv = _mm512_set1_epi16(NEG as i16);
    let neghv = _mm512_set1_epi16(NEG_HALF as i16);
    let ov = _mm512_set1_epi16(o);
    let ev_ = _mm512_set1_epi16(e);
    let z = [0i16; L];
    let (mut ph, mut ch) = (vec![z; n + 1], vec![z; n + 1]);
    let (mut pj, mut cj) = (vec![z; n + 1], vec![z; n + 1]);
    let (mut pf, mut cf) = (vec![[NEG as i16; L]; n + 1], vec![[NEG as i16; L]; n + 1]);
    let (mut pfj, mut cfj) = (vec![z; n + 1], vec![z; n + 1]);
    let mut dirs = vec![Dir::default(); n * n];
    let mut rows = [zero; K];
    let (mut best, mut bi, mut bj) = (zero, zero, zero);
    let ms = minsep.max(1);
    for i in 0..n {
        let start = i + ms;
        if start >= n {
            break;
        }
        rows_of(&mut rows, &tabs, &st_[i]);
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
            let lvalid = _mm512_cmpgt_epi16_mask(lh, zero) & _mm512_cmpgt_epi16_mask(lj, iv);
            let evalid = _mm512_cmpgt_epi16_mask(evv, neghv) & _mm512_cmpgt_epi16_mask(ej, iv);
            let eo_v = _mm512_mask_blend_epi16(lvalid, negv, _mm512_sub_epi16(lh, ov));
            let ex_v = _mm512_mask_blend_epi16(evalid, negv, _mm512_sub_epi16(evv, ev_));
            let eo = _mm512_cmpge_epi16_mask(eo_v, ex_v);
            evv = _mm512_mask_blend_epi16(eo, ex_v, eo_v);
            ej = _mm512_mask_blend_epi16(eo, ej, lj);
            let uvalid = _mm512_cmpgt_epi16_mask(up, zero) & _mm512_cmpgt_epi16_mask(uj, iv);
            let fvalid = _mm512_cmpgt_epi16_mask(upf, neghv) & _mm512_cmpgt_epi16_mask(upfj, iv);
            let fo_v = _mm512_mask_blend_epi16(uvalid, negv, _mm512_sub_epi16(up, ov));
            let fx_v = _mm512_mask_blend_epi16(fvalid, negv, _mm512_sub_epi16(upf, ev_));
            let fo = _mm512_cmpge_epi16_mask(fo_v, fx_v);
            let fv = _mm512_mask_blend_epi16(fo, fx_v, fo_v);
            let fj = _mm512_mask_blend_epi16(fo, upfj, uj);
            let dvalid = _mm512_cmpgt_epi16_mask(dg, zero) & _mm512_cmpgt_epi16_mask(dj, iv);
            let dv = _mm512_add_epi16(_mm512_maskz_mov_epi16(dvalid, dg), p);
            let dj2 = _mm512_mask_blend_epi16(dvalid, jv, dj);
            let c1 = _mm512_cmpge_epi16_mask(dv, fv) & _mm512_cmpge_epi16_mask(dv, evv);
            let c2 = _mm512_cmpge_epi16_mask(fv, evv);
            let hv = _mm512_mask_blend_epi16(c1, _mm512_mask_blend_epi16(c2, evv, fv), dv);
            let hj = _mm512_mask_blend_epi16(c1, _mm512_mask_blend_epi16(c2, ej, fj), dj2);
            let pos = _mm512_cmpgt_epi16_mask(hv, zero);
            let h = _mm512_maskz_mov_epi16(pos, hv);
            let hj = _mm512_maskz_mov_epi16(pos, hj);
            let better = _mm512_cmpgt_epi16_mask(h, best);
            best = _mm512_mask_blend_epi16(better, best, h);
            bi = _mm512_mask_blend_epi16(better, bi, iv);
            bj = _mm512_mask_blend_epi16(better, bj, jv);
            st(&mut ch[j], h);
            st(&mut cj[j], hj);
            st(&mut cf[j], fv);
            st(&mut cfj[j], fj);
            dirs[i * n + j] = Dir { c1, c2, eo, fo, dp: dvalid };
            lh = h;
            lj = hj;
        }
        std::mem::swap(&mut ph, &mut ch);
        std::mem::swap(&mut pj, &mut cj);
        std::mem::swap(&mut pf, &mut cf);
        std::mem::swap(&mut pfj, &mut cfj);
    }
    let mut r = Raw { best: z, bi: z, bj: z, dirs, centers: Vec::new() };
    st(&mut r.best, best);
    st(&mut r.bi, bi);
    st(&mut r.bj, bj);
    r
}

#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn rect_raw(lanes: &[Lane], g: i16) -> Raw {
    let na = lanes[0].a.len();
    let nb = lanes[0].s.len();
    let aa: Vec<&[u8]> = lanes.iter().map(|x| x.a).collect();
    let bb: Vec<&[u8]> = lanes.iter().map(|x| x.s).collect();
    let tabs: Vec<Tab16> = lanes.iter().map(|x| tab16(x.tab)).collect();
    let (at, _) = prep(&aa, na);
    let (_, bm) = prep(&bb, nb);
    let zero = _mm512_setzero_si512();
    let gv = _mm512_set1_epi16(g);
    let z = [0i16; L];
    let (mut ph, mut ch) = (vec![z; nb + 1], vec![z; nb + 1]);
    let mut dirs = vec![Dir::default(); na * nb];
    let mut rows = [zero; K];
    let (mut best, mut bi, mut bj) = (zero, zero, zero);
    for x in 0..na {
        rows_of(&mut rows, &tabs, &at[x]);
        let xv = _mm512_set1_epi16(x as i16);
        st(&mut ch[0], zero);
        let mut left = zero;
        let mut dg = ld(&ph[0]);
        for y in 1..=nb {
            let p = lookup(&rows, &bm[y - 1]);
            let up = ld(&ph[y]);
            let dv = _mm512_add_epi16(dg, p);
            let uv = _mm512_sub_epi16(up, gv);
            let lv = _mm512_sub_epi16(left, gv);
            let c1 = _mm512_cmpge_epi16_mask(dv, uv) & _mm512_cmpge_epi16_mask(dv, lv);
            let c2 = _mm512_cmpge_epi16_mask(uv, lv);
            let hv = _mm512_mask_blend_epi16(c1, _mm512_mask_blend_epi16(c2, lv, uv), dv);
            let h = _mm512_max_epi16(hv, zero);
            let dpm = _mm512_cmpgt_epi16_mask(dg, zero);
            let better = _mm512_cmpgt_epi16_mask(h, best);
            best = _mm512_mask_blend_epi16(better, best, h);
            bi = _mm512_mask_blend_epi16(better, bi, xv);
            bj = _mm512_mask_blend_epi16(better, bj, _mm512_set1_epi16((y - 1) as i16));
            st(&mut ch[y], h);
            dirs[x * nb + (y - 1)] = Dir { c1, c2, eo: 0, fo: 0, dp: dpm };
            left = h;
            dg = up;
        }
        std::mem::swap(&mut ph, &mut ch);
    }
    let mut r = Raw { best: z, bi: z, bj: z, dirs, centers: Vec::new() };
    st(&mut r.best, best);
    st(&mut r.bi, bi);
    st(&mut r.bj, bj);
    r
}

#[derive(Clone, Copy, PartialEq)]
enum St {
    H,
    E,
    F,
}

/// 从最高分格往回走。shape: 0 嵌套 1 同向 2 两序列
fn traceback(shape: u8, ln: &Lane, raw: &Raw, l: usize, width: usize) -> Best {
    let score = raw.best[l] as i32;
    if score <= 0 {
        return Best::default();
    }
    let (bi, bj) = (raw.bi[l] as usize, raw.bj[l] as usize);
    let (mut i, mut j) = (bi, bj);
    let (mut pairs, mut matched, mut gaps, mut bs) = (0u16, 0u16, 0u16, 0i16);
    let mut mvs: Vec<f64> = Vec::new();
    let mut stt = St::H;
    let (i0, j0);
    loop {
        let d = raw.dirs[i * width + j];
        match stt {
            St::H => {
                if bit(d.c1, l) {
                    pairs += 1;
                    let (la, lb) = if shape == 2 { (ln.a[i], ln.s[j]) } else { (ln.s[i], ln.s[j]) };
                    let p = ln.tab[la as usize][lb as usize];
                    if p > 0 {
                        matched += 1;
                        if shape == 2 {
                            bs += side_of(lb);
                            mvs.push(ln.vol[j] as f64);
                        } else {
                            bs += side_of(la);
                            mvs.push(ln.vol[i] as f64 + ln.vol[j] as f64);
                        }
                    }
                    if bit(d.dp, l) {
                        match shape {
                            0 => {
                                i += 1;
                                j -= 1;
                            }
                            _ => {
                                i -= 1;
                                j -= 1;
                            }
                        }
                    } else {
                        i0 = i;
                        j0 = j;
                        break;
                    }
                } else if bit(d.c2, l) {
                    stt = St::F;
                } else {
                    stt = St::E;
                }
            }
            St::F => {
                gaps = gaps.saturating_add(1);
                let open = shape == 2 || bit(d.fo, l);
                if shape == 0 {
                    i += 1;
                } else {
                    i -= 1;
                }
                if open {
                    stt = St::H;
                }
            }
            St::E => {
                gaps = gaps.saturating_add(1);
                let open = shape == 2 || bit(d.eo, l);
                j -= 1;
                if open {
                    stt = St::H;
                }
            }
        }
    }
    let mut mv = 0.0f64;
    for x in mvs.iter().rev() {
        mv += *x;
    }
    Best { score, i: bi, j: bj, c: Cell { v: score, i0: i0 as u16, j0: j0 as u16, pairs, matched, gaps, bs, mv } }
}

/// 一组参数相同、长度相同的比对（≤32 条）。shape: 0 嵌套 1 同向 2 两序列
pub fn run_group(shape: u8, lanes: &[Lane], o: i32, e: i32, minsep: usize) -> Vec<Out> {
    let raw = unsafe {
        match shape {
            0 => nested_raw(lanes, o as i16, e as i16, minsep),
            1 => forward_raw(lanes, o as i16, e as i16, minsep),
            _ => rect_raw(lanes, o as i16),
        }
    };
    let width = if shape == 2 { lanes[0].s.len() } else { lanes[0].s.len() };
    let mut raw = raw;
    let centers = std::mem::take(&mut raw.centers);
    lanes
        .iter()
        .enumerate()
        .map(|(l, ln)| Out {
            best: traceback(shape, ln, &raw, l, width),
            center: if ln.center { centers[l].clone() } else { Vec::new() },
        })
        .collect()
}

pub fn available() -> bool {
    std::arch::is_x86_feature_detected!("avx512bw")
}

// 让未用到的导入不报警
#[allow(dead_code)]
fn _unused(_: &dp::Scratch) {}
