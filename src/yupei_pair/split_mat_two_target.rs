/// Two target histories per AVX2 vector. Each four-lane accumulator keeps the
/// exact same source-cell order, multiply then add (no FMA), as the SSE kernel.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn compute_split_mats_two_targets(
    cells: &[Vec<(u32, [f32; 4])>], tau: f32, w: PmWeight,
) -> [Vec<f32>; 4] {
    use std::arch::x86_64::*;
    let n = cells.len();
    let (lp, lm) = w.lanes();
    let decay = (-(BUCKET_US as f32) / (tau * 1e6)).exp();
    let mut mats: [Vec<f32>; 4] = std::array::from_fn(|_| vec![0.0f32; n * n]);
    let npack = n.div_ceil(2);
    // [source0: target0 four lanes, target1 four lanes, source1: ...]
    let packs: Vec<std::sync::Mutex<Vec<f32>>> = (0..npack).into_par_iter()
        .map(|_| std::sync::Mutex::new(vec![0.0f32; n * 8 + 4 + BLOCK_BUCKETS * 4]))
        .collect();
    let mut cursors = vec![0usize; n];
    for block in 0..N_BLOCKS {
        let k0 = (block * BLOCK_BUCKETS) as u32;
        let k1 = ((k0 as usize + BLOCK_BUCKETS).min(N_BUCKETS)) as u32;
        let block_len = (k1 - k0) as usize;
        let counts: Vec<usize> = cells.par_iter().enumerate().map(|(i, st)| {
            let mut c = cursors[i];
            while c < st.len() && st[c].0 < k1 { c += 1; }
            c - cursors[i]
        }).collect();
        let total: usize = counts.iter().sum();
        if total == 0 {
            packs.par_iter().for_each(|pack| {
                let mut g = pack.lock().unwrap();
                decay_empty_block(&mut g[n * 8..n * 8 + 4], decay, block_len);
            });
            continue;
        }
        let mut offs = Vec::with_capacity(n + 1);
        let mut acc = 0usize;
        for &c in &counts { offs.push(acc); acc += c; }
        offs.push(acc);
        let mut buf: Vec<(u32, [f32; 4])> = vec![(0, [0.0f32; 4]); total];
        for i in 0..n {
            let (s, c) = (cursors[i], counts[i]);
            buf[offs[i]..offs[i] + c].copy_from_slice(&cells[i][s..s + c]);
            cursors[i] += c;
        }
        (0..npack).into_par_iter().for_each(|jp| {
            let mut g = packs[jp].lock().unwrap();
            let (pack, rest) = g.split_at_mut(n * 8);
            let (acc, gp) = rest.split_at_mut(4);
            for slot in 0..2 {
                let j = jp * 2 + slot;
                if j >= n { continue; }
                let (mut ap, mut am) = (acc[slot * 2], acc[slot * 2 + 1]);
                let (bcnt, boff) = (counts[j], offs[j]);
                let mut cur = 0usize;
                for k in 0..block_len {
                    let (mut up, mut um) = (0.0f32, 0.0f32);
                    while cur < bcnt && (buf[boff + cur].0 - k0) as usize == k {
                        let u = &buf[boff + cur].1;
                        up += u[lp]; um += u[lm]; cur += 1;
                    }
                    gp[k * 4 + slot * 2] = ap + 0.5 * up;
                    gp[k * 4 + slot * 2 + 1] = am + 0.5 * um;
                    ap = (ap + up) * decay;
                    am = (am + um) * decay;
                }
                acc[slot * 2] = ap; acc[slot * 2 + 1] = am;
            }
            unsafe {
                for i in 0..n {
                    let ca = counts[i];
                    if ca == 0 { continue; }
                    let base = i * 8;
                    let mut pv = _mm256_loadu_ps(pack.as_ptr().add(base));
                    for t in 0..ca {
                        let cell = &buf[offs[i] + t];
                        let kl = (cell.0 - k0) as usize;
                        let hist = _mm_loadu_ps(gp.as_ptr().add(kl * 4));
                        // Earlier-side direction first: [p0,p0,m0,m0,p1,p1,m1,m1].
                        let h8 = _mm256_insertf128_ps(_mm256_castps128_ps256(hist), hist, 1);
                        let h8 = _mm256_permutevar8x32_ps(h8, _mm256_setr_epi32(0, 0, 1, 1, 2, 2, 3, 3));
                        let up = *cell.1.get_unchecked(lp);
                        let um = *cell.1.get_unchecked(lm);
                        let weight = _mm256_setr_ps(up, um, up, um, up, um, up, um);
                        pv = _mm256_add_ps(pv, _mm256_mul_ps(weight, h8));
                    }
                    _mm256_storeu_ps(pack.as_mut_ptr().add(base), pv);
                }
            }
        });
    }
    struct Ptr(*mut f32);
    unsafe impl Send for Ptr {}
    unsafe impl Sync for Ptr {}
    let raw: Vec<Ptr> = mats.iter_mut().map(|v| Ptr(v.as_mut_ptr())).collect();
    (0..npack).into_par_iter().for_each(|jp| {
        let g = packs[jp].lock().unwrap();
        for slot in 0..2 {
            let j = jp * 2 + slot;
            if j >= n { continue; }
            for i in 0..n {
                for t in 0..4 {
                    unsafe { *raw[t].0.add(j * n + i) = g[i * 8 + slot * 4 + t]; }
                }
            }
        }
    });
    mats
}
