//! 配对折叠层：在一条字母序列上求“互不交叉”的最优配对（RNA 二级结构的 Nussinov 式动态规划）。
//!
//! 表 r 的含义：r[i][x] = 第 i 到第 x-1 个字母这一段的最优得分（x<=i 时为 0）。
//! 递推：第 j 个字母要么不配对，要么和某个 k 配对，把这一段切成 i..k-1、k+1..j-1 两块：
//!   r[i][j+1] = max( r[i][j],  max_k ( r[i][k] + w(k,j) + r[k+1][j] ) )
//! 对固定的 j，先算好 z[k] = w(k,j) + r[k+1][j]，内层就变成两段连续数组逐项相加再取最大，
//! 全部用整数，既能被 CPU 批量处理，也避免浮点累加带来的不确定性。
pub const NEG: i32 = i32::MIN / 2;

pub struct Table {
    stride: usize,
    r: Vec<i32>,
    z: Vec<i32>,
    stack: Vec<(usize, usize)>,
}

impl Table {
    pub fn new() -> Self {
        Table { stride: 0, r: Vec::new(), z: Vec::new(), stack: Vec::new() }
    }

    /// 求最优得分。w(k,j) 返回配对权重，不可配对时返回 NEG。
    pub fn solve<W: Fn(usize, usize) -> i32>(&mut self, n: usize, h: usize, active: &[bool], w: &W) -> i32 {
        let stride = n + 1;
        self.stride = stride;
        self.r.clear();
        self.r.resize(stride * stride, 0);
        self.z.clear();
        self.z.resize(n + 1, NEG);
        let r = &mut self.r[..];
        let z = &mut self.z[..];
        for j in 0..n {
            if !active[j] || j < h + 1 {
                for i in 0..=j {
                    r[i * stride + j + 1] = r[i * stride + j];
                }
                continue;
            }
            let kmax = j - h - 1;
            for k in 0..=kmax {
                let v = w(k, j);
                z[k] = if v > NEG / 2 { v + r[(k + 1) * stride + j] } else { NEG };
            }
            for i in 0..=j {
                let row = &mut r[i * stride..(i + 1) * stride];
                let mut best = row[j];
                if i <= kmax {
                    for (a, b) in row[i..=kmax].iter().zip(&z[i..=kmax]) {
                        best = best.max(*a + *b);
                    }
                }
                row[j + 1] = best;
            }
        }
        r[n]
    }

    /// 倒推配对方案，写进 partner（未配对为 -1）。
    /// 同分时的规则是固定的：优先让第 j 个不配对；否则 latest=true 选最靠近 j 的 k，false 选最远的。
    pub fn traceback<W: Fn(usize, usize) -> i32>(&mut self, n: usize, h: usize, w: &W, latest: bool, partner: &mut [i32]) {
        let stride = self.stride;
        let r = &self.r[..];
        partner[..n].iter_mut().for_each(|p| *p = -1);
        let stack = &mut self.stack;
        stack.clear();
        stack.push((0, n));
        while let Some((i, x0)) = stack.pop() {
            let mut x = x0;
            while x > i {
                let j = x - 1;
                let cur = r[i * stride + x];
                if cur == r[i * stride + j] {
                    x -= 1;
                    continue;
                }
                let kmax = j - h - 1; // 分数变大说明一定配了对，所以 j >= h+1
                let mut found = usize::MAX;
                let mut try_k = |k: usize| -> bool {
                    let v = w(k, j);
                    if v > NEG / 2 && r[i * stride + k] + v + r[(k + 1) * stride + j] == cur {
                        found = k;
                        true
                    } else {
                        false
                    }
                };
                if latest {
                    for k in (i..=kmax).rev() {
                        if try_k(k) { break; }
                    }
                } else {
                    for k in i..=kmax {
                        if try_k(k) { break; }
                    }
                }
                debug_assert!(found != usize::MAX);
                partner[found] = j as i32;
                partner[j] = found as i32;
                stack.push((found + 1, j));
                x = found;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rng(s: &mut u64) -> u64 {
        *s ^= *s << 13;
        *s ^= *s >> 7;
        *s ^= *s << 17;
        *s
    }

    /// 暴力枚举所有互不交叉的配对，作为对照。
    fn brute(i: usize, j_end: usize, h: usize, w: &dyn Fn(usize, usize) -> i32) -> i32 {
        if j_end <= i { return 0; }
        let j = j_end - 1;
        let mut best = brute(i, j, h, w);
        if j >= h + 1 {
            for k in i..=(j - h - 1) {
                let v = w(k, j);
                if v > NEG / 2 {
                    best = best.max(brute(i, k, h, w) + v + brute(k + 1, j, h, w));
                }
            }
        }
        best
    }

    #[test]
    fn dp_matches_bruteforce_and_traceback_is_valid() {
        let mut s = 88172645463325252u64;
        let mut t = Table::new();
        for case in 0..400 {
            let n = 4 + (rng(&mut s) % 11) as usize;
            let h = (rng(&mut s) % 4) as usize;
            let side: Vec<i8> = (0..n).map(|_| (rng(&mut s) % 3) as i8 - 1).collect();
            let vol: Vec<i32> = (0..n).map(|_| 1 + (rng(&mut s) % 9) as i32).collect();
            let active: Vec<bool> = side.iter().map(|&x| x != 0).collect();
            let w = |k: usize, j: usize| -> i32 {
                if side[k] * side[j] == -1 {
                    if case % 2 == 0 { 1 } else { vol[k].min(vol[j]) }
                } else { NEG }
            };
            let dp = t.solve(n, h, &active, &w);
            assert_eq!(dp, brute(0, n, h, &w));
            for latest in [true, false] {
                let mut p = vec![-1i32; n];
                t.traceback(n, h, &w, latest, &mut p);
                let mut sum = 0;
                let mut stack: Vec<usize> = Vec::new();
                for x in 0..n {
                    if p[x] > x as i32 {
                        assert!(p[x] as usize - x - 1 >= h);
                        sum += w(x, p[x] as usize);
                        stack.push(x);
                    } else if p[x] >= 0 {
                        assert_eq!(stack.pop(), Some(p[x] as usize)); // 配对必须互不交叉
                    }
                }
                assert_eq!(sum, dp);
            }
        }
    }

    /// 不加最短环长时，只要两种字母都有，计数配对恒等于“较少一方的字母数”。
    #[test]
    fn unconstrained_unit_pairing_equals_trivial_bound() {
        let mut s = 123456789u64;
        let mut t = Table::new();
        for _ in 0..300 {
            let n = 2 + (rng(&mut s) % 60) as usize;
            let side: Vec<i8> = (0..n).map(|_| if rng(&mut s) % 2 == 0 { 1 } else { -1 }).collect();
            let active = vec![true; n];
            let b = side.iter().filter(|&&x| x > 0).count() as i32;
            let c = n as i32 - b;
            let w = |k: usize, j: usize| if side[k] * side[j] == -1 { 1 } else { NEG };
            assert_eq!(t.solve(n, 0, &active, &w), b.min(c));
        }
    }
}
