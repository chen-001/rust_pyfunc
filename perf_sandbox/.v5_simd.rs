//! 热循环 AVX2 助手：pair 扫描里最内层的元素级运算与归约。
//! 每个函数都有标量兜底；运行时用 is_x86_feature_detected 选路。
//! 注意：SIMD 归约的浮点求和顺序与标量顺序不同（8 路并行累加器），
//! 结果与旧版在末位有差别，但同一版本下多次运行逐位相同（仍是确定性的）。

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[inline(always)]
pub fn have_avx2() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::is_x86_feature_detected!("avx2")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// out[x] = wb*p[x] + wc*q[x] + addm[x]（addm 里是 -inf 的位会把 X 屏蔽掉）
pub fn fill_h(out: &mut [f32], wb: f32, wc: f32, p: &[f32], q: &[f32], addm: &[f32]) {
    let n = out.len();
    #[cfg(target_arch = "x86_64")]
    unsafe {
        if have_avx2() {
            let wv = _mm256_set1_ps(wb);
            let cv = _mm256_set1_ps(wc);
            let mut x = 0;
            while x + 8 <= n {
                let a = _mm256_loadu_ps(p.as_ptr().add(x));
                let b = _mm256_loadu_ps(q.as_ptr().add(x));
                let m = _mm256_loadu_ps(addm.as_ptr().add(x));
                // 不用 fma：保持与标量 wb*p + wc*q 相同的两次舍入，
                // 保证 argmax 在并列/近并列边缘的选择与旧版逐位一致
                let r = _mm256_add_ps(
                    _mm256_add_ps(_mm256_mul_ps(wv, a), _mm256_mul_ps(cv, b)),
                    m,
                );
                _mm256_storeu_ps(out.as_mut_ptr().add(x), r);
                x += 8;
            }
            while x < n {
                out[x] = wb * p[x] + wc * q[x] + addm[x];
                x += 1;
            }
            return;
        }
    }
    for x in 0..n {
        out[x] = wb * p[x] + wc * q[x] + addm[x];
    }
}

/// g[x] = h[x] - max(p[x], q[x])
pub fn fill_gap(g: &mut [f32], h: &[f32], p: &[f32], q: &[f32]) {
    let n = g.len();
    #[cfg(target_arch = "x86_64")]
    unsafe {
        if have_avx2() {
            let mut x = 0;
            while x + 8 <= n {
                let hv = _mm256_loadu_ps(h.as_ptr().add(x));
                let a = _mm256_loadu_ps(p.as_ptr().add(x));
                let b = _mm256_loadu_ps(q.as_ptr().add(x));
                let r = _mm256_sub_ps(hv, _mm256_max_ps(a, b));
                _mm256_storeu_ps(g.as_mut_ptr().add(x), r);
                x += 8;
            }
            while x < n {
                g[x] = h[x] - if p[x] > q[x] { p[x] } else { q[x] };
                x += 1;
            }
            return;
        }
    }
    for x in 0..n {
        g[x] = h[x] - if p[x] > q[x] { p[x] } else { q[x] };
    }
}

/// 一趟算完 h、g 并累计「最大值」与「pos_stats 的三个和 + 计数」。
///
/// 等价于依次调用 fill_h、fill_gap、argmax_first（只取最大值那半段）、pos_stats 的
/// 累加段，归约顺序与尾巴的处理顺序与逐个调用时完全一致：
///   * 最大值：8 路 max 累加 → 按 lane 0..8 依次比较 → 尾巴按 x 升序比较（严格大于）；
///   * pos_stats：8 路和累加 → 按 lane 0..8 依次相加 → 尾巴按 x 升序累加；
///   * b、c 位置写 -inf（原代码在两次 fill 之后单独置位，此处等价）。
/// 返回 (最大值h, 最大值g, Σrelu(h), Σrelu(g), Σrelu(g)², #(g>0))。
pub fn fused_hg_stats(
    hb: &mut [f32],
    gb: &mut [f32],
    wb: f32,
    wc: f32,
    p: &[f32],
    q: &[f32],
    addm: &[f32],
    b: usize,
    c: usize,
) -> (f32, f32, f32, f32, f32, u32) {
    let n = hb.len();
    #[cfg(target_arch = "x86_64")]
    unsafe {
        if have_avx2() {
            let wv = _mm256_set1_ps(wb);
            let cv = _mm256_set1_ps(wc);
            let zero = _mm256_setzero_ps();
            let mut mh = _mm256_set1_ps(f32::NEG_INFINITY);
            let mut mg = _mm256_set1_ps(f32::NEG_INFINITY);
            let mut sh = _mm256_setzero_ps();
            let mut sg = _mm256_setzero_ps();
            let mut sq = _mm256_setzero_ps();
            let mut cnt = 0u32;
            let mut x = 0;
            let mut hbuf = [0f32; 8];
            let mut gbuf = [0f32; 8];
            while x + 8 <= n {
                let (hv, gv);
                if (b >= x && b < x + 8) || (c >= x && c < x + 8) {
                    // 含 b/c 的那块：这两个位置必须是 -inf，标量算这 8 个再按原 lane 累加
                    for i in 0..8 {
                        let xx = x + i;
                        let mut h = wb * p[xx] + wc * q[xx] + addm[xx];
                        let mut g = h - if p[xx] > q[xx] { p[xx] } else { q[xx] };
                        if xx == b || xx == c {
                            h = f32::NEG_INFINITY;
                            g = f32::NEG_INFINITY;
                        }
                        hbuf[i] = h;
                        gbuf[i] = g;
                    }
                    for i in 0..8 {
                        hb[x + i] = hbuf[i];
                        gb[x + i] = gbuf[i];
                    }
                    hv = _mm256_loadu_ps(hbuf.as_ptr());
                    gv = _mm256_loadu_ps(gbuf.as_ptr());
                } else {
                    let a = _mm256_loadu_ps(p.as_ptr().add(x));
                    let bv = _mm256_loadu_ps(q.as_ptr().add(x));
                    let m = _mm256_loadu_ps(addm.as_ptr().add(x));
                    // 不用 fma：与 fill_h 的两次舍入一致
                    hv = _mm256_add_ps(
                        _mm256_add_ps(_mm256_mul_ps(wv, a), _mm256_mul_ps(cv, bv)),
                        m,
                    );
                    gv = _mm256_sub_ps(hv, _mm256_max_ps(a, bv));
                    _mm256_storeu_ps(hb.as_mut_ptr().add(x), hv);
                    _mm256_storeu_ps(gb.as_mut_ptr().add(x), gv);
                }
                mh = _mm256_max_ps(mh, hv);
                mg = _mm256_max_ps(mg, gv);
                sh = _mm256_add_ps(sh, _mm256_max_ps(hv, zero));
                let gp = _mm256_max_ps(gv, zero);
                sg = _mm256_add_ps(sg, gp);
                sq = _mm256_add_ps(sq, _mm256_mul_ps(gp, gp));
                let mm = _mm256_movemask_ps(_mm256_cmp_ps::<_CMP_GT_OQ>(gv, zero)) as u32;
                cnt += mm.count_ones();
                x += 8;
            }
            let mut b1 = [0f32; 8];
            let mut b2 = [0f32; 8];
            _mm256_storeu_ps(b1.as_mut_ptr(), mh);
            _mm256_storeu_ps(b2.as_mut_ptr(), mg);
            let (mut mx_h, mut mx_g) = (f32::NEG_INFINITY, f32::NEG_INFINITY);
            for i in 0..8 {
                if b1[i] > mx_h {
                    mx_h = b1[i];
                }
                if b2[i] > mx_g {
                    mx_g = b2[i];
                }
            }
            let mut b3 = [0f32; 8];
            let mut b4 = [0f32; 8];
            let mut b5 = [0f32; 8];
            _mm256_storeu_ps(b3.as_mut_ptr(), sh);
            _mm256_storeu_ps(b4.as_mut_ptr(), sg);
            _mm256_storeu_ps(b5.as_mut_ptr(), sq);
            let (mut s1, mut s2, mut s3) = (0f32, 0f32, 0f32);
            for i in 0..8 {
                s1 += b3[i];
                s2 += b4[i];
                s3 += b5[i];
            }
            while x < n {
                let mut h = wb * p[x] + wc * q[x] + addm[x];
                let mut g = h - if p[x] > q[x] { p[x] } else { q[x] };
                if x == b || x == c {
                    h = f32::NEG_INFINITY;
                    g = f32::NEG_INFINITY;
                }
                hb[x] = h;
                gb[x] = g;
                if h > mx_h {
                    mx_h = h;
                }
                if h > 0.0 {
                    s1 += h;
                }
                if g > mx_g {
                    mx_g = g;
                }
                if g > 0.0 {
                    s2 += g;
                    s3 += g * g;
                    cnt += 1;
                }
                x += 1;
            }
            return (mx_h, mx_g, s1, s2, s3, cnt);
        }
    }
    let (mut mx_h, mut mx_g) = (f32::NEG_INFINITY, f32::NEG_INFINITY);
    let (mut s1, mut s2, mut s3, mut cnt) = (0f32, 0f32, 0f32, 0u32);
    for x in 0..n {
        let mut h = wb * p[x] + wc * q[x] + addm[x];
        let mut g = h - if p[x] > q[x] { p[x] } else { q[x] };
        if x == b || x == c {
            h = f32::NEG_INFINITY;
            g = f32::NEG_INFINITY;
        }
        hb[x] = h;
        gb[x] = g;
        if h > mx_h {
            mx_h = h;
        }
        if h > 0.0 {
            s1 += h;
        }
        if g > mx_g {
            mx_g = g;
        }
        if g > 0.0 {
            s2 += g;
            s3 += g * g;
            cnt += 1;
        }
    }
    (mx_h, mx_g, s1, s2, s3, cnt)
}

/// (首个最大值的 x, 最大值)。全是 -inf 时返回 (usize::MAX, -inf)，
/// 与旧标量版"找不到合法候选"的语义一致。
pub fn argmax_first(v: &[f32]) -> (usize, f32) {
    let n = v.len();
    #[cfg(target_arch = "x86_64")]
    unsafe {
        if have_avx2() {
            let mut m = _mm256_set1_ps(f32::NEG_INFINITY);
            let mut x = 0;
            while x + 8 <= n {
                m = _mm256_max_ps(m, _mm256_loadu_ps(v.as_ptr().add(x)));
                x += 8;
            }
            let mut buf = [0f32; 8];
            _mm256_storeu_ps(buf.as_mut_ptr(), m);
            let mut mx = f32::NEG_INFINITY;
            for &b in buf.iter() {
                if b > mx {
                    mx = b;
                }
            }
            while x < n {
                if v[x] > mx {
                    mx = v[x];
                }
                x += 1;
            }
            if mx == f32::NEG_INFINITY {
                return (usize::MAX, f32::NEG_INFINITY);
            }
            // 找第一个等于 mx 的下标（tie 取下标小的，与旧版 `>` 语义一致）
            let mv = _mm256_set1_ps(mx);
            let mut x = 0;
            while x + 8 <= n {
                let c = _mm256_cmp_ps::<_CMP_EQ_OQ>(_mm256_loadu_ps(v.as_ptr().add(x)), mv);
                let mask = _mm256_movemask_ps(c) as u32;
                if mask != 0 {
                    return (x + mask.trailing_zeros() as usize, mx);
                }
                x += 8;
            }
            while x < n {
                if v[x] == mx {
                    return (x, mx);
                }
                x += 1;
            }
            return (usize::MAX, mx);
        }
    }
    let mut mx = f32::NEG_INFINITY;
    let mut bi = usize::MAX;
    for (x, &vv) in v.iter().enumerate() {
        if vv > mx {
            mx = vv;
            bi = x;
        }
    }
    (bi, mx)
}

/// 一趟找出 h、g 里「第一个等于最大值」的下标，等价于两次 argmax_first 的后半段
/// （两次分别只扫一个数组）。扫描顺序、并列取小下标的语义都不变。
/// mx_g 为 -inf 时按原语义不扫 g、直接返回 usize::MAX。
pub fn argmax_first_two(h: &[f32], g: &[f32], mx_h: f32, mx_g: f32) -> (usize, usize) {
    let n = h.len();
    let find_g = mx_g != f32::NEG_INFINITY;
    #[cfg(target_arch = "x86_64")]
    unsafe {
        if have_avx2() {
            let mh = _mm256_set1_ps(mx_h);
            let mg = _mm256_set1_ps(mx_g);
            let mut bi = usize::MAX;
            let mut bsi = usize::MAX;
            let mut x = 0;
            while x + 8 <= n {
                if bi == usize::MAX {
                    let c = _mm256_cmp_ps::<_CMP_EQ_OQ>(_mm256_loadu_ps(h.as_ptr().add(x)), mh);
                    let mask = _mm256_movemask_ps(c) as u32;
                    if mask != 0 {
                        bi = x + mask.trailing_zeros() as usize;
                    }
                }
                if bsi == usize::MAX && find_g {
                    let c = _mm256_cmp_ps::<_CMP_EQ_OQ>(_mm256_loadu_ps(g.as_ptr().add(x)), mg);
                    let mask = _mm256_movemask_ps(c) as u32;
                    if mask != 0 {
                        bsi = x + mask.trailing_zeros() as usize;
                    }
                }
                if bi != usize::MAX && (bsi != usize::MAX || !find_g) {
                    return (bi, bsi);
                }
                x += 8;
            }
            while x < n {
                if bi == usize::MAX && h[x] == mx_h {
                    bi = x;
                }
                if bsi == usize::MAX && find_g && g[x] == mx_g {
                    bsi = x;
                }
                if bi != usize::MAX && (bsi != usize::MAX || !find_g) {
                    break;
                }
                x += 1;
            }
            return (bi, bsi);
        }
    }
    let mut bi = usize::MAX;
    let mut bsi = usize::MAX;
    for x in 0..n {
        if bi == usize::MAX && h[x] == mx_h {
            bi = x;
        }
        if bsi == usize::MAX && find_g && g[x] == mx_g {
            bsi = x;
        }
        if bi != usize::MAX && (bsi != usize::MAX || !find_g) {
            break;
        }
    }
    (bi, bsi)
}

/// (Σ relu(h), Σ relu(g), Σ relu(g)², #(g>0)) —— f32 SIMD 归约。
pub fn pos_stats(h: &[f32], g: &[f32]) -> (f32, f32, f32, u32) {
    let n = h.len();
    #[cfg(target_arch = "x86_64")]
    unsafe {
        if have_avx2() {
            let zero = _mm256_setzero_ps();
            let mut sh = _mm256_setzero_ps();
            let mut sg = _mm256_setzero_ps();
            let mut sq = _mm256_setzero_ps();
            let mut cnt = 0u32;
            let mut x = 0;
            while x + 8 <= n {
                let hv = _mm256_loadu_ps(h.as_ptr().add(x));
                let gv = _mm256_loadu_ps(g.as_ptr().add(x));
                sh = _mm256_add_ps(sh, _mm256_max_ps(hv, zero));
                let gp = _mm256_max_ps(gv, zero);
                sg = _mm256_add_ps(sg, gp);
                sq = _mm256_add_ps(sq, _mm256_mul_ps(gp, gp)); // 同标量 s3 += g*g 的舍入
                let m = _mm256_movemask_ps(_mm256_cmp_ps::<_CMP_GT_OQ>(gv, zero)) as u32;
                cnt += m.count_ones();
                x += 8;
            }
            let mut b1 = [0f32; 8];
            let mut b2 = [0f32; 8];
            let mut b3 = [0f32; 8];
            _mm256_storeu_ps(b1.as_mut_ptr(), sh);
            _mm256_storeu_ps(b2.as_mut_ptr(), sg);
            _mm256_storeu_ps(b3.as_mut_ptr(), sq);
            let (mut s1, mut s2, mut s3) = (0f32, 0f32, 0f32);
            for i in 0..8 {
                s1 += b1[i];
                s2 += b2[i];
                s3 += b3[i];
            }
            while x < n {
                if h[x] > 0.0 {
                    s1 += h[x];
                }
                if g[x] > 0.0 {
                    s2 += g[x];
                    s3 += g[x] * g[x];
                    cnt += 1;
                }
                x += 1;
            }
            return (s1, s2, s3, cnt);
        }
    }
    let (mut s1, mut s2, mut s3, mut c) = (0f32, 0f32, 0f32, 0u32);
    for x in 0..n {
        if h[x] > 0.0 {
            s1 += h[x];
        }
        if g[x] > 0.0 {
            s2 += g[x];
            s3 += g[x] * g[x];
            c += 1;
        }
    }
    (s1, s2, s3, c)
}

/// em[x] += w, gm[x] += w*g, g2m[x] += w*g²，其中 w = relu(h)*invh。
/// 三个累加器都是本块私有的 f32 数组（确定性靠块内顺序累加 + 按下标合并保证）。
/// 必须 w>0 才累加一阶二阶矩：掩码位 w=0 但 g=-inf，0×(-inf)=NaN 会污染全表。
pub fn mass_acc_h(em: &mut [f32], gm: &mut [f32], g2m: &mut [f32], h: &[f32], g: &[f32], invh: f32) {
    let n = em.len();
    for x in 0..n {
        if h[x] > 0.0 {
            let w = h[x] * invh;
            let gg = g[x];
            em[x] += w;
            gm[x] += w * gg;
            g2m[x] += w * gg * gg;
        }
    }
}

/// gap 口径同款：w = relu(g)*invg；w>0 时 g 必有限，安全乘。
pub fn mass_acc_g(sm: &mut [f32], sgm: &mut [f32], sg2m: &mut [f32], g: &[f32], invg: f32) {
    let n = sm.len();
    for x in 0..n {
        if g[x] > 0.0 {
            let w = g[x] * invg;
            sm[x] += w;
            sgm[x] += w * g[x];
            sg2m[x] += w * g[x] * g[x];
        }
    }
}

/// mass_acc_h 与 mass_acc_g 并成一趟：同一轮 x 先更新 h 口径的三个数组、再更新 gap
/// 口径的三个数组。六个数组互不相干、每个元素每个配对只加一次，所以每个累加器看到
/// 的加法序列和分开调用时逐位相同。do_h/do_g 对应原代码的两个「和大于 0 才累加」条件。
#[allow(clippy::too_many_arguments)]
pub fn mass_acc_hg(
    em: &mut [f32],
    gm: &mut [f32],
    g2m: &mut [f32],
    sm: &mut [f32],
    sgm: &mut [f32],
    sg2m: &mut [f32],
    h: &[f32],
    g: &[f32],
    invh: f32,
    invg: f32,
    do_h: bool,
    do_g: bool,
) {
    let n = h.len();
    for x in 0..n {
        if do_h {
            let hv = h[x];
            if hv > 0.0 {
                let w = hv * invh;
                let gg = g[x];
                em[x] += w;
                gm[x] += w * gg;
                g2m[x] += w * gg * gg;
            }
        }
        if do_g {
            let gv = g[x];
            if gv > 0.0 {
                let w = gv * invg;
                sm[x] += w;
                sgm[x] += w * gv;
                sg2m[x] += w * gv * gv;
            }
        }
    }
}

/// out[x] = (a[x] + b[x]) * w —— reduce pair_scan / pair_scan_topk 的 rho(X,U) 展开式。
pub fn mul_add(out: &mut [f32], a: &[f32], b: &[f32], w: f32) {
    let n = out.len();
    #[cfg(target_arch = "x86_64")]
    unsafe {
        if have_avx2() {
            let wv = _mm256_set1_ps(w);
            let mut x = 0;
            while x + 8 <= n {
                let r = _mm256_mul_ps(
                    _mm256_add_ps(
                        _mm256_loadu_ps(a.as_ptr().add(x)),
                        _mm256_loadu_ps(b.as_ptr().add(x)),
                    ),
                    wv,
                );
                _mm256_storeu_ps(out.as_mut_ptr().add(x), r);
                x += 8;
            }
            while x < n {
                out[x] = (a[x] + b[x]) * w;
                x += 1;
            }
            return;
        }
    }
    for x in 0..n {
        out[x] = (a[x] + b[x]) * w;
    }
}

/// best[x] = max(best[x], v[x])；对被更新的 x 把 idx[x] 写成 c。
/// SIMD 比较+取大做主活，更新下标只在命中的少数位置做标量写。
pub fn elem_max_upd_idx(best: &mut [f32], idx: &mut [i32], v: &[f32], c: i32) {
    let n = best.len();
    #[cfg(target_arch = "x86_64")]
    unsafe {
        if have_avx2() {
            let mut x = 0;
            while x + 8 <= n {
                let vv = _mm256_loadu_ps(v.as_ptr().add(x));
                let bb = _mm256_loadu_ps(best.as_ptr().add(x));
                let m = _mm256_movemask_ps(_mm256_cmp_ps::<_CMP_GT_OQ>(vv, bb)) as u32;
                if m != 0 {
                    _mm256_storeu_ps(best.as_mut_ptr().add(x), _mm256_max_ps(vv, bb));
                    let mut mm = m;
                    while mm != 0 {
                        let k = mm.trailing_zeros() as usize;
                        idx[x + k] = c;
                        mm &= mm - 1;
                    }
                }
                x += 8;
            }
            while x < n {
                if v[x] > best[x] {
                    best[x] = v[x];
                    idx[x] = c;
                }
                x += 1;
            }
            return;
        }
    }
    for x in 0..n {
        if v[x] > best[x] {
            best[x] = v[x];
            idx[x] = c;
        }
    }
}

/// best[x] = max(best[x], v[x]) 纯更新（不带下标）。
#[allow(dead_code)]
pub fn elem_max_upd(best: &mut [f32], v: &[f32]) -> u32 {
    let n = best.len();
    #[cfg(target_arch = "x86_64")]
    unsafe {
        if have_avx2() {
            let mut x = 0;
            let mut cnt = 0u32;
            while x + 8 <= n {
                let vv = _mm256_loadu_ps(v.as_ptr().add(x));
                let bb = _mm256_loadu_ps(best.as_ptr().add(x));
                let m = _mm256_movemask_ps(_mm256_cmp_ps::<_CMP_GT_OQ>(vv, bb)) as u32;
                if m != 0 {
                    _mm256_storeu_ps(best.as_mut_ptr().add(x), _mm256_max_ps(vv, bb));
                    cnt += m.count_ones();
                }
                x += 8;
            }
            while x < n {
                if v[x] > best[x] {
                    best[x] = v[x];
                    cnt += 1;
                }
                x += 1;
            }
            return cnt;
        }
    }
    let mut cnt = 0u32;
    for x in 0..n {
        if v[x] > best[x] {
            best[x] = v[x];
            cnt += 1;
        }
    }
    cnt
}

/// 数出 v[x] > refv[x] 的位置数（pair_scan_topk 的"击败单只"计数）。
#[allow(dead_code)]
pub fn count_gt(v: &[f32], refv: &[f32]) -> u64 {
    let n = v.len();
    #[cfg(target_arch = "x86_64")]
    unsafe {
        if have_avx2() {
            let mut cnt = 0u64;
            let mut x = 0;
            while x + 8 <= n {
                let m = _mm256_movemask_ps(_mm256_cmp_ps::<_CMP_GT_OQ>(
                    _mm256_loadu_ps(v.as_ptr().add(x)),
                    _mm256_loadu_ps(refv.as_ptr().add(x)),
                )) as u32;
                cnt += m.count_ones() as u64;
                x += 8;
            }
            while x < n {
                if v[x] > refv[x] {
                    cnt += 1;
                }
                x += 1;
            }
            return cnt;
        }
    }
    let mut cnt = 0u64;
    for x in 0..n {
        if v[x] > refv[x] {
            cnt += 1;
        }
    }
    cnt
}

/// 把 v[x] > refv[x] 的位置下标追加到 out（pair_scan_topk 需要知道是哪些 x 被击败）。
pub fn collect_gt_idx(out: &mut Vec<u32>, v: &[f32], refv: &[f32]) {
    let n = v.len();
    #[cfg(target_arch = "x86_64")]
    unsafe {
        if have_avx2() {
            let mut x = 0;
            while x + 8 <= n {
                let m = _mm256_movemask_ps(_mm256_cmp_ps::<_CMP_GT_OQ>(
                    _mm256_loadu_ps(v.as_ptr().add(x)),
                    _mm256_loadu_ps(refv.as_ptr().add(x)),
                )) as u32;
                let mut mm = m;
                while mm != 0 {
                    let k = mm.trailing_zeros() as usize;
                    out.push((x + k) as u32);
                    mm &= mm - 1;
                }
                x += 8;
            }
            while x < n {
                if v[x] > refv[x] {
                    out.push(x as u32);
                }
                x += 1;
            }
            return;
        }
    }
    for x in 0..n {
        if v[x] > refv[x] {
            out.push(x as u32);
        }
    }
}

/// beat[x] += (v[x] > refv[x])，i32 逐格计数（配对数 < 2^31，不会溢出）。
/// SIMD 掩码加：每 8 格一次 cmp+and+add，替代逐格分支。
pub fn add_cnt_gt_i32(beat: &mut [i32], v: &[f32], refv: &[f32]) {
    let n = v.len();
    #[cfg(target_arch = "x86_64")]
    unsafe {
        if have_avx2() {
            let one = _mm256_set1_epi32(1);
            let mut x = 0;
            while x + 8 <= n {
                let m = _mm256_castps_si256(_mm256_cmp_ps::<_CMP_GT_OQ>(
                    _mm256_loadu_ps(v.as_ptr().add(x)),
                    _mm256_loadu_ps(refv.as_ptr().add(x)),
                ));
                let cur = _mm256_loadu_si256(beat.as_ptr().add(x) as *const __m256i);
                _mm256_storeu_si256(
                    beat.as_mut_ptr().add(x) as *mut __m256i,
                    _mm256_add_epi32(cur, _mm256_and_si256(m, one)),
                );
                x += 8;
            }
            while x < n {
                beat[x] += (v[x] > refv[x]) as i32;
                x += 1;
            }
            return;
        }
    }
    for x in 0..n {
        beat[x] += (v[x] > refv[x]) as i32;
    }
}

/// 对每个 v[x] > refv[x] 的 x 调 f(x)——top-k 插入这类"命中才干活"的循环，
/// 先用 movemask 把 3964 格压成几个命中位，跳过绝大多数比较。
#[allow(dead_code)]
pub fn for_each_gt(v: &[f32], refv: &[f32], mut f: impl FnMut(usize)) {
    let n = v.len();
    #[cfg(target_arch = "x86_64")]
    unsafe {
        if have_avx2() {
            let mut x = 0;
            while x + 8 <= n {
                let m = _mm256_movemask_ps(_mm256_cmp_ps::<_CMP_GT_OQ>(
                    _mm256_loadu_ps(v.as_ptr().add(x)),
                    _mm256_loadu_ps(refv.as_ptr().add(x)),
                )) as u32;
                let mut mm = m;
                while mm != 0 {
                    f(x + mm.trailing_zeros() as usize);
                    mm &= mm - 1;
                }
                x += 8;
            }
            while x < n {
                if v[x] > refv[x] {
                    f(x);
                }
                x += 1;
            }
            return;
        }
    }
    for x in 0..n {
        if v[x] > refv[x] {
            f(x);
        }
    }
}

/// f32 → 保序 u32：正数翻符号位、负数全部取反，使整型序 = 浮点序。
/// 用于把 (score, idx) 打包成 u64 做整型 select_nth / sort，去掉浮点 comparator。
#[inline(always)]
pub fn fkey(v: f32) -> u32 {
    let b = v.to_bits();
    if b & 0x8000_0000 != 0 {
        !b
    } else {
        b | 0x8000_0000
    }
}

/// 打包"分数降序、下标升序"的 u64 键：升序比较 = 分数大的在前、同分按下标小的在前。
/// （!fkey 升序 = fkey 降序；低 32 位直接放下标，升序 = 下标升序。）
#[inline(always)]
pub fn pack_desc_idx_asc(score: f32, idx: u32) -> u64 {
    ((!fkey(score)) as u64) << 32 | idx as u64
}

/// 打包"分数降序、下标降序"的 u64 键（topk_adj 的并列规则：取下标大的）。
#[inline(always)]
pub fn pack_desc_idx_desc(score: f32, idx: u32) -> u64 {
    ((!fkey(score)) as u64) << 32 | (!(idx)) as u64
}
