use rayon::prelude::*;
const BUCKET_US: i64 = 10_000;
const BLOCK_BUCKETS: usize = 4096;
const N_BLOCKS: usize = 8;
#[derive(Clone,Copy,Debug)]
enum PmWeight { Cnt, Vol }
impl PmWeight { fn lanes(self) -> (usize,usize) { match self { Self::Cnt => (0,1), Self::Vol => (2,3) } } }
fn compute_split_mats_sse(
    cells: &[Vec<(u32, [f32; 4])>],
    tau: f32,
    w: PmWeight,
) -> [Vec<f32>; 4] {
    use std::arch::x86_64::*;
    let n = cells.len();
    let (lp, lm) = w.lanes();
    let decay = (-(BUCKET_US as f32) / (tau * 1e6)).exp();
    let mut mats: [Vec<f32>; 4] = std::array::from_fn(|_| vec![0.0f32; n * n]);

    const GPAD: usize = BLOCK_BUCKETS * 2;
    let packs: Vec<std::sync::Mutex<Vec<f32>>> = (0..n)
        .into_par_iter()
        .map(|_| std::sync::Mutex::new(vec![0.0f32; n * 4 + 2 + GPAD]))
        .collect();

    let mut cursors = vec![0usize; n];
    for block in 0..N_BLOCKS {
        let k0 = (block * BLOCK_BUCKETS) as u32;
        let k1 = k0 + BLOCK_BUCKETS as u32;

        let counts: Vec<usize> = cells
            .par_iter()
            .enumerate()
            .map(|(i, st)| {
                let mut c = cursors[i];
                while c < st.len() && st[c].0 < k1 {
                    c += 1;
                }
                c - cursors[i]
            })
            .collect();
        let total: usize = counts.iter().sum();
        if total == 0 {
            continue;
        }
        let mut offs: Vec<usize> = Vec::with_capacity(n + 1);
        let mut acc = 0usize;
        for &c in counts.iter() {
            offs.push(acc);
            acc += c;
        }
        offs.push(acc);
        let mut buf: Vec<(u32, [f32; 4])> = vec![(0, [0.0f32; 4]); total];
        for i in 0..n {
            let s = cursors[i];
            let c = counts[i];
            buf[offs[i]..offs[i] + c].copy_from_slice(&cells[i][s..s + c]);
            cursors[i] += c;
        }

        let (buf_ref, offs_ref, counts_ref) = (&buf, &offs, &counts);
        (0..n).into_par_iter().for_each(|j| {
            let mut g = packs[j].lock().unwrap();
            let (pack, rest) = g.split_at_mut(n * 4);
            let (acc, gp) = rest.split_at_mut(2);
            let (mut ap, mut am) = (acc[0], acc[1]);
            let (bcnt, boff) = (counts_ref[j], offs_ref[j]);
            let mut cur = 0usize;
            for k in 0..BLOCK_BUCKETS {
                let (mut up, mut um) = (0.0f32, 0.0f32);
                while cur < bcnt && (buf_ref[boff + cur].0 - k0) as usize == k {
                    let u = &buf_ref[boff + cur].1;
                    up += u[lp];
                    um += u[lm];
                    cur += 1;
                }
                gp[k * 2] = ap + 0.5 * up;
                gp[k * 2 + 1] = am + 0.5 * um;
                ap = (ap + up) * decay;
                am = (am + um) * decay;
            }

            // 配对累加：四个分量打包成 [PP, PM, MP, MM]，一次乘加算完
            unsafe {
                for i in 0..n {
                    let ca = counts_ref[i];
                    if ca == 0 {
                        continue;
                    }
                    let base = i * 4;
                    let mut pv = _mm_loadu_ps(pack.as_ptr().add(base));
                    for t in 0..ca {
                        let cell = &buf_ref[offs_ref[i] + t];
                        let kl = (cell.0 - k0) as usize;
                        let gpk = [*gp.get_unchecked(kl * 2), *gp.get_unchecked(kl * 2 + 1)];
                        let up = *cell.1.get_unchecked(lp);
                        let um = *cell.1.get_unchecked(lm);
                        // e0=PP=up*gP  e1=PM=up*gM  e2=MP=um*gP  e3=MM=um*gM
                        let a = _mm_set_ps(um, um, up, up);
                        let b = _mm_set_ps(gpk[1], gpk[0], gpk[1], gpk[0]);
                        pv = _mm_add_ps(pv, _mm_mul_ps(a, b));
                    }
                    _mm_storeu_ps(pack.as_mut_ptr().add(base), pv);
                }
            }
            acc[0] = ap;
            acc[1] = am;
        });
    }

    struct Ptr(*mut f32);
    unsafe impl Send for Ptr {}
    unsafe impl Sync for Ptr {}
    let raw: Vec<Ptr> = mats.iter_mut().map(|v| Ptr(v.as_mut_ptr())).collect();
    (0..n).into_par_iter().for_each(|j| {
        let g = packs[j].lock().unwrap();
        let pack = &g[..n * 4];
        for i in 0..n {
            for t in 0..4 {
                unsafe {
                    *raw[t].0.add(j * n + i) = pack[i * 4 + t];
                }
            }
        }
    });
    mats
}

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
        let k1 = k0 + BLOCK_BUCKETS as u32;
        let counts: Vec<usize> = cells.par_iter().enumerate().map(|(i, st)| {
            let mut c = cursors[i];
            while c < st.len() && st[c].0 < k1 { c += 1; }
            c - cursors[i]
        }).collect();
        let total: usize = counts.iter().sum();
        if total == 0 { continue; } // Preserve the reference's empty-block semantics.
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
                for k in 0..BLOCK_BUCKETS {
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
                        // [p0,m0,p0,m0,p1,m1,p1,m1]
                        let h8 = _mm256_insertf128_ps(_mm256_castps128_ps256(hist), hist, 1);
                        let h8 = _mm256_permutevar8x32_ps(h8, _mm256_setr_epi32(0, 1, 0, 1, 2, 3, 2, 3));
                        let up = *cell.1.get_unchecked(lp);
                        let um = *cell.1.get_unchecked(lm);
                        let weight = _mm256_setr_ps(up, up, um, um, up, up, um, um);
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

fn main() {
    assert!(std::arch::is_x86_feature_detected!("avx2"));
    let mut seed = 0x694efc7820u64;
    let mut cases = 0;
    let mut values = 0;
    for n in [0usize, 1, 2, 17, 128, 513] {
        let cells: Vec<Vec<(u32,[f32;4])>> = (0..n).map(|i| {
            let mut v = Vec::new();
            for t in 0..N_BLOCKS*BLOCK_BUCKETS {
                seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17;
                if i % 23 == 0 || t / BLOCK_BUCKETS == 3 || seed % 128 > 3 { continue; }
                let p = ((seed >> 16) % 27) as f32;
                let m = ((seed >> 32) % 31) as f32;
                v.push((t as u32, [p, m, (p*107.3).sqrt(), (m*313.7).sqrt()]));
            }
            v
        }).collect();
        for w in [PmWeight::Cnt, PmWeight::Vol] {
            let t = std::time::Instant::now();
            let a = compute_split_mats_sse(&cells, 1.0, w);
            let original = t.elapsed().as_secs_f64();
            let t = std::time::Instant::now();
            let b = unsafe { compute_split_mats_two_targets(&cells, 1.0, w) };
            let candidate = t.elapsed().as_secs_f64();
            let mismatches = a.iter().zip(&b).map(|(a,b)| a.iter().zip(b).filter(|(x,y)| x.to_bits()!=y.to_bits()).count()).sum::<usize>();
            assert_eq!(mismatches, 0, "n={n}, w={w:?}");
            cases += 1; values += n*n*4;
            println!("n={n} weight={w:?} original={original:.6} candidate={candidate:.6} mismatches={mismatches}");
        }
    }
    println!("cases={cases} values={values} bit_mismatches=0");
}
