//! 比对层：带状 + 仿射空位的全局/局部比对（动态规划），带回溯。
//! a 是今天的字母序列，b 是昨天的。操作：D=两边对齐，X=今天多出一个（昨天留空），Y=昨天多出一个（今天留空）。
//! 只保留两行分数，回溯信息放在预分配缓冲里，整只股票复用，核心循环不做堆分配。

pub const SIZE: usize = 12; // 每个字母表最多 12 个字母（含静默字母 0）
pub const OP_D: u8 = 0;
pub const OP_X: u8 = 1;
pub const OP_Y: u8 = 2;
const NEG: i32 = -500_000_000;

#[derive(Default)]
pub struct Work {
    h_prev: Vec<i32>, h_cur: Vec<i32>, x_prev: Vec<i32>, x_cur: Vec<i32>,
    tr: Vec<u8>,
    pub ops: Vec<u8>, // 回溯得到的操作序列（逆序写入后翻转为正序）
}

#[derive(Clone, Copy, Default)]
pub struct Aln { pub score: i32, pub i0: usize, pub j0: usize, pub i1: usize, pub j1: usize }

/// band：允许今天和昨天错开的最大格数；local=true 时做局部比对。
pub fn align(a: &[u8], b: &[u8], tbl: &[i32], open: i32, ext: i32, band: usize, local: bool, w: &mut Work) -> Aln {
    let (n, m) = (a.len(), b.len());
    let width = 2 * band + 1;
    let row = width + 2;
    for v in [&mut w.h_prev, &mut w.h_cur, &mut w.x_prev, &mut w.x_cur] { v.clear(); v.resize(row, NEG); }
    w.tr.clear(); w.tr.resize((n + 1) * width, 0);
    // 回溯字节：低 2 位来源（0 停止,1 D,2 X,3 Y）；4=X 延续；8=Y 延续
    // 第 0 行
    let c0 = band + 1;
    w.h_prev[c0] = 0;
    let mut y_left = NEG;
    for j in 1..=band.min(m) {
        let p = c0 + j;
        if local { w.h_prev[p] = 0; }
        else {
            let v = -(open + (j as i32 - 1) * ext);
            w.h_prev[p] = v;
            w.tr[p - 1] = 3 | if j > 1 { 8 } else { 0 };
        }
    }
    let _ = &mut y_left;
    let (mut best, mut bi, mut bj) = (0i32, 0usize, 0usize);
    for i in 1..=n {
        let base = i * width;
        y_left = NEG;
        w.h_cur.iter_mut().for_each(|v| *v = NEG);
        w.x_cur.iter_mut().for_each(|v| *v = NEG);
        let jlo = i.saturating_sub(band);
        let jhi = m.min(i + band);
        let ai = a[i - 1] as usize * SIZE;
        for j in jlo..=jhi {
            let p = j + band + 1 - i;
            if j == 0 {
                if local { w.h_cur[p] = 0; }
                else {
                    let v = -(open + (i as i32 - 1) * ext);
                    w.h_cur[p] = v; w.x_cur[p] = v;
                    w.tr[base + p - 1] = 2 | if i > 1 { 4 } else { 0 };
                }
                continue;
            }
            let (x1, x2) = (w.h_prev[p + 1] - open, w.x_prev[p + 1] - ext);
            let (xv, xc) = if x2 > x1 { (x2, 4u8) } else { (x1, 0u8) };
            let (y1, y2) = (w.h_cur[p - 1] - open, y_left - ext);
            let (yv, yc) = if y2 > y1 { (y2, 8u8) } else { (y1, 0u8) };
            let d = w.h_prev[p] + tbl[ai + b[j - 1] as usize];
            let (mut h, mut o) = if d >= xv && d >= yv { (d, 1u8) } else if xv >= yv { (xv, 2u8) } else { (yv, 3u8) };
            if local && h <= 0 { h = 0; o = 0; }
            w.h_cur[p] = h; w.x_cur[p] = xv; y_left = yv;
            w.tr[base + p - 1] = o | xc | yc;
            if local && h > best { best = h; bi = i; bj = j; }
        }
        std::mem::swap(&mut w.h_prev, &mut w.h_cur);
        std::mem::swap(&mut w.x_prev, &mut w.x_cur);
    }
    // 回溯
    w.ops.clear();
    let (mut i, mut j, score) = if local { (bi, bj, best) } else {
        (n, m, w.h_prev[m + band + 1 - n])
    };
    let (i1, j1) = (i, j);
    let mut state = 0u8; // 0=H,1=X,2=Y
    loop {
        if !local && i == 0 && j == 0 { break; }
        let p = j + band + 1 - i;
        let t = w.tr[i * width + p - 1];
        match state {
            0 => match t & 3 {
                0 => break,
                1 => { w.ops.push(OP_D); i -= 1; j -= 1; }
                2 => { w.ops.push(OP_X); state = if t & 4 != 0 { 1 } else { 0 }; i -= 1; }
                _ => { w.ops.push(OP_Y); state = if t & 8 != 0 { 2 } else { 0 }; j -= 1; }
            },
            1 => { w.ops.push(OP_X); state = if t & 4 != 0 { 1 } else { 0 }; i -= 1; }
            _ => { w.ops.push(OP_Y); state = if t & 8 != 0 { 2 } else { 0 }; j -= 1; }
        }
    }
    w.ops.reverse();
    Aln { score, i0: i, j0: j, i1, j1 }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn tbl() -> Vec<i32> {
        let mut t = vec![-1i32; SIZE * SIZE];
        for x in 0..SIZE { t[x * SIZE + x] = 1; }
        t
    }
    #[test]
    fn matches_hand_example() {
        // GATC 对 GTC，匹配 +1、错配 -1、空位 -1（等于无仿射），正确答案总分 2，路径 D X D D
        let (a, b) = ([1u8, 2, 3, 4], [1u8, 3, 4]);
        let mut w = Work::default();
        let r = align(&a, &b, &tbl(), 1, 1, 4, false, &mut w);
        assert_eq!(r.score, 2);
        assert_eq!(w.ops, vec![OP_D, OP_X, OP_D, OP_D]);
    }
    #[test]
    fn local_finds_core() {
        let (a, b) = ([5u8, 5, 1, 2, 3, 6], [7u8, 1, 2, 3, 7, 7]);
        let mut w = Work::default();
        let r = align(&a, &b, &tbl(), 2, 1, 6, true, &mut w);
        assert_eq!((r.score, r.i0, r.j0), (3, 2, 1));
        assert_eq!(w.ops, vec![OP_D, OP_D, OP_D]);
    }
    #[test]
    fn shift_within_band() {
        // 今天整体比昨天晚 2 格
        let (a, b) = ([9u8, 9, 1, 2, 3, 4, 5, 6], [1u8, 2, 3, 4, 5, 6, 8, 8]);
        let mut w = Work::default();
        let r = align(&a, &b, &tbl(), 1, 1, 3, false, &mut w);
        assert_eq!(w.ops.iter().filter(|&&o| o == OP_D).count(), 6);
        assert!(r.score > 0);
    }
}
