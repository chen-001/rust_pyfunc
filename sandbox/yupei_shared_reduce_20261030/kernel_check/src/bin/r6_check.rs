use rayon::prelude::*;
use std::time::Instant;
const K_TOPK: usize = 20;
struct SoftCfg { thr: f64 }
fn soft() -> SoftCfg { SoftCfg { thr: 2.0 } }
pub(crate) fn rank_avg(v: &[f64]) -> Vec<f64> {
    let m = v.len();
    let mut idx: Vec<usize> = (0..m).collect();
    idx.sort_by(|&a, &b| v[a].partial_cmp(&v[b]).unwrap().then(a.cmp(&b)));
    let mut r = vec![0.0f64; m];
    let mut i = 0usize;
    while i < m {
        let mut j = i;
        while j + 1 < m && v[idx[j + 1]] == v[idx[i]] {
            j += 1;
        }
        let avg = (i + j + 2) as f64 / 2.0; // 秩从 1 开始：i+1 .. j+1 的平均
        for k in i..=j {
            r[idx[k]] = avg;
        }
        i = j + 1;
    }
    r
}

/// Spearman，只在两边都有限的位置上算；样本不够返回空值。
pub(crate) fn spearman(x: &[f64], y: &[f64], min_n: usize) -> f64 {
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    for i in 0..x.len() {
        if x[i].is_finite() && y[i].is_finite() {
            xs.push(x[i]);
            ys.push(y[i]);
        }
    }
    if xs.len() < min_n {
        return f64::NAN;
    }
    let mut rx = rank_avg(&xs);
    let mut ry = rank_avg(&ys);
    let mx = rx.iter().sum::<f64>() / rx.len() as f64;
    let my = ry.iter().sum::<f64>() / ry.len() as f64;
    for v in rx.iter_mut() {
        *v -= mx;
    }
    for v in ry.iter_mut() {
        *v -= my;
    }
    let d = (rx.iter().map(|v| v * v).sum::<f64>() * ry.iter().map(|v| v * v).sum::<f64>()).sqrt();
    if d <= 0.0 {
        return f64::NAN;
    }
    rx.iter().zip(ry.iter()).map(|(a, b)| a * b).sum::<f64>() / d
}

/// nanmedian：忽略空值；偶数个取中间两个的均值，与 numpy 一致。
pub(crate) fn nanmedian(v: &[f64]) -> f64 {
    let mut f: Vec<f64> = v.iter().copied().filter(|x| x.is_finite()).collect();
    if f.is_empty() {
        return f64::NAN;
    }
    f.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let m = f.len();
    if m % 2 == 1 {
        f[m / 2]
    } else {
        (f[m / 2 - 1] + f[m / 2]) / 2.0
    }
}

pub(crate) struct R6 {
    pub(crate) gen_max: Vec<f64>,
    pub(crate) gen_top20: Vec<f64>,
    pub(crate) spec_max: Vec<f64>,
    pub(crate) spec_top20: Vec<f64>,
    pub(crate) zs_max: Vec<f64>,
    pub(crate) row_corr: Vec<f64>,
    pub(crate) spec_share: Vec<f64>,
    pub(crate) quad: Vec<f64>,
}

/// pair_reduce.py 的 r6_plane。
/// gen 侧用严格上三角（Zrow 与 ZSrow 只填上三角，所以最后一行的这四个数是空的）；
/// spec 侧用对称阵 Sc（对角填 0，所以 spec_max 下界是 0）。
pub(crate) fn r6_plane(n: usize, z: &[f32], zs: &[f32], ok: &[bool]) -> R6 {
    let t0 = Instant::now();
    // 每个配对的 (spec, 是否专才对) 与参与配对的次数
    let mut sc = vec![0.0f32; n * n]; // 对称，对角 0
    let mut sp = vec![0.0f32; n * n];
    let mut cnt_valid = vec![0.0f64; n];
    let zthr = soft().thr;
    for i in 0..n {
        if !ok[i] {
            continue;
        }
        for j in (i + 1)..n {
            if !ok[j] {
                continue;
            }
            let zl = z[i * n + j] as f64;
            let zsl = zs[i * n + j] as f64;
            let s = (zsl - zl) as f32;
            let p = if zsl > zthr && zl < -zthr { 1.0f32 } else { 0.0 };
            sc[i * n + j] = s;
            sc[j * n + i] = s;
            sp[i * n + j] = p;
            sp[j * n + i] = p;
            cnt_valid[i] += 1.0;
            cnt_valid[j] += 1.0;
        }
    }
    // 逐行：gen 侧只取 j > i，spec 侧取整行（含对角那个 0）
    let per_row: Vec<(f64, f64, f64, f64, f64, f64, f64)> = (0..n)
        .into_par_iter()
        .map(|i| {
            // gen（上三角、有效对）：Z 与 ZS 各一份
            let mut gv: Vec<f64> = Vec::new();
            let mut zv: Vec<f64> = Vec::new();
            if ok[i] {
                for j in (i + 1)..n {
                    if !ok[j] {
                        continue;
                    }
                    gv.push(z[i * n + j] as f64);
                    zv.push(zs[i * n + j] as f64);
                }
            }
            let gen_max = if gv.is_empty() {
                f64::NAN
            } else {
                gv.iter().copied().fold(f64::NEG_INFINITY, f64::max)
            };
            let gen_top20 = if gv.is_empty() {
                f64::NAN
            } else {
                let mut t = gv.clone();
                let k = K_TOPK.min(t.len());
                if k < t.len() {
                    let nth = t.len() - k;
                    t.select_nth_unstable_by(nth, |a, b| a.total_cmp(b));
                }
                t[t.len() - k..].iter().sum::<f64>() / k as f64
            };
            let zs_max = if zv.is_empty() {
                f64::NAN
            } else {
                zv.iter().copied().fold(f64::NEG_INFINITY, f64::max)
            };
            let row_corr = if gv.len() > 30 {
                spearman(&gv, &zv, 30)
            } else {
                f64::NAN
            };
            // spec：整行 n 个值（对角 0，无效位置 0）
            let mut spec_max = 0.0f64; // 对角那个 0 一定在里面
            let mut vals: Vec<f64> = Vec::with_capacity(n);
            for j in 0..n {
                if j == i {
                    vals.push(0.0);
                    continue;
                }
                let v = sc[i * n + j] as f64;
                vals.push(v);
                if v > spec_max {
                    spec_max = v;
                }
            }
            let mut t = vals;
            let k = K_TOPK.min(t.len());
            if k < t.len() {
                let nth = t.len() - k;
                t.select_nth_unstable_by(nth, |a, b| a.total_cmp(b));
            }
            let spec_top20 = t[t.len() - k..].iter().sum::<f64>() / k as f64;
            let share = if cnt_valid[i] > 0.0 {
                let s: f64 = (0..n).map(|j| sp[i * n + j] as f64).sum();
                s / cnt_valid[i].max(1.0)
            } else {
                f64::NAN
            };
            (gen_max, gen_top20, spec_max, spec_top20, zs_max, row_corr, share)
        })
        .collect();
    let gen_max: Vec<f64> = per_row.iter().map(|t| t.0).collect();
    let gen_top20: Vec<f64> = per_row.iter().map(|t| t.1).collect();
    let spec_max: Vec<f64> = per_row.iter().map(|t| t.2).collect();
    let spec_top20: Vec<f64> = per_row.iter().map(|t| t.3).collect();
    let zs_max: Vec<f64> = per_row.iter().map(|t| t.4).collect();
    let row_corr: Vec<f64> = per_row.iter().map(|t| t.5).collect();
    let spec_share: Vec<f64> = per_row.iter().map(|t| t.6).collect();
    let mg = nanmedian(&gen_max);
    let ms = nanmedian(&zs_max);
    let quad: Vec<f64> = (0..n)
        .map(|i| {
            let a = if gen_max[i] > mg { 2.0 } else { 0.0 };
            let b = if zs_max[i] > ms { 1.0 } else { 0.0 };
            a + b
        })
        .collect();
    if false {
        eprintln!("[reduce:p6] R6 平面 {:.1}s", t0.elapsed().as_secs_f32());
    }
    R6 {
        gen_max,
        gen_top20,
        spec_max,
        spec_top20,
        zs_max,
        row_corr,
        spec_share,
        quad,
    }
}pub(crate) fn r6_plane_streaming(n: usize, z: &[f32], zs: &[f32], ok: &[bool]) -> R6 {
    let t0 = Instant::now();
    let zthr = soft().thr;
    // Produce the exact original row directly: invalid pairs and diagonal zero,
    // canonical upper-triangle inputs, original order of values and additions.
    // 逐行：gen 侧只取 j > i，spec 侧取整行（含对角那个 0）
    let per_row: Vec<(f64, f64, f64, f64, f64, f64, f64)> = (0..n)
        .into_par_iter()
        .map(|i| {
            // gen（上三角、有效对）：Z 与 ZS 各一份
            let mut gv: Vec<f64> = Vec::new();
            let mut zv: Vec<f64> = Vec::new();
            if ok[i] {
                for j in (i + 1)..n {
                    if !ok[j] {
                        continue;
                    }
                    gv.push(z[i * n + j] as f64);
                    zv.push(zs[i * n + j] as f64);
                }
            }
            let gen_max = if gv.is_empty() {
                f64::NAN
            } else {
                gv.iter().copied().fold(f64::NEG_INFINITY, f64::max)
            };
            let gen_top20 = if gv.is_empty() {
                f64::NAN
            } else {
                let mut t = gv.clone();
                let k = K_TOPK.min(t.len());
                if k < t.len() {
                    let nth = t.len() - k;
                    t.select_nth_unstable_by(nth, |a, b| a.total_cmp(b));
                }
                t[t.len() - k..].iter().sum::<f64>() / k as f64
            };
            let zs_max = if zv.is_empty() {
                f64::NAN
            } else {
                zv.iter().copied().fold(f64::NEG_INFINITY, f64::max)
            };
            let row_corr = if gv.len() > 30 {
                spearman(&gv, &zv, 30)
            } else {
                f64::NAN
            };
            // spec：整行 n 个值（对角 0，无效位置 0）
            let mut spec_max = 0.0f64; // 对角那个 0 一定在里面
            let mut vals: Vec<f64> = Vec::with_capacity(n);
            let mut valid_count = 0.0f64;
            let mut spec_count = 0.0f64;
            for j in 0..n {
                if j == i {
                    vals.push(0.0);
                    continue;
                }
                let v = if ok[i] && ok[j] {
                    let ij = i.min(j) * n + i.max(j);
                    let zl = z[ij] as f64;
                    let zsl = zs[ij] as f64;
                    valid_count += 1.0;
                    if zsl > zthr && zl < -zthr { spec_count += 1.0; }
                    (zsl - zl) as f32 as f64
                } else { 0.0 };
                vals.push(v);
                if v > spec_max {
                    spec_max = v;
                }
            }
            let mut t = vals;
            let k = K_TOPK.min(t.len());
            if k < t.len() {
                let nth = t.len() - k;
                t.select_nth_unstable_by(nth, |a, b| a.total_cmp(b));
            }
            let spec_top20 = t[t.len() - k..].iter().sum::<f64>() / k as f64;
            let share = if valid_count > 0.0 {
                spec_count / valid_count.max(1.0)
            } else {
                f64::NAN
            };
            (gen_max, gen_top20, spec_max, spec_top20, zs_max, row_corr, share)
        })
        .collect();
    let gen_max: Vec<f64> = per_row.iter().map(|t| t.0).collect();
    let gen_top20: Vec<f64> = per_row.iter().map(|t| t.1).collect();
    let spec_max: Vec<f64> = per_row.iter().map(|t| t.2).collect();
    let spec_top20: Vec<f64> = per_row.iter().map(|t| t.3).collect();
    let zs_max: Vec<f64> = per_row.iter().map(|t| t.4).collect();
    let row_corr: Vec<f64> = per_row.iter().map(|t| t.5).collect();
    let spec_share: Vec<f64> = per_row.iter().map(|t| t.6).collect();
    let mg = nanmedian(&gen_max);
    let ms = nanmedian(&zs_max);
    let quad: Vec<f64> = (0..n)
        .map(|i| {
            let a = if gen_max[i] > mg { 2.0 } else { 0.0 };
            let b = if zs_max[i] > ms { 1.0 } else { 0.0 };
            a + b
        })
        .collect();
    if false {
        eprintln!("[reduce:p6] R6 平面 {:.1}s", t0.elapsed().as_secs_f32());
    }
    R6 {
        gen_max,
        gen_top20,
        spec_max,
        spec_top20,
        zs_max,
        row_corr,
        spec_share,
        quad,
    }
}
fn main() {
    let mut seed = 0x687a318bd16u64;
    let mut cases = 0; let mut values = 0;
    for n in [0usize, 1, 2, 17, 65, 128, 513] {
        for trial in 0..4 {
            let mut z = vec![0f32; n*n]; let mut zs = z.clone();
            for (a,b) in z.iter_mut().zip(&mut zs) {
                seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17;
                *a = ((seed >> 32) as i32 as f32 / (u32::MAX as f32)) * 17.0;
                *b = ((seed & 0xffffffff) as i32 as f32 / (u32::MAX as f32)) * 11.0;
                if trial % 2 == 0 && seed % 23 == 0 { *a = f32::NAN; }
                if trial % 2 == 0 && seed % 19 == 0 { *b = f32::INFINITY; }
            }
            let ok: Vec<bool> = (0..n).map(|i| trial % 2 == 0 || i % 7 != 0).collect();
            let t=Instant::now(); let a=r6_plane(n,&z,&zs,&ok); let old=t.elapsed().as_secs_f64();
            let t=Instant::now(); let b=r6_plane_streaming(n,&z,&zs,&ok); let new=t.elapsed().as_secs_f64();
            let ac=[a.gen_max,a.gen_top20,a.spec_max,a.spec_top20,a.zs_max,a.row_corr,a.spec_share,a.quad];
            let bc=[b.gen_max,b.gen_top20,b.spec_max,b.spec_top20,b.zs_max,b.row_corr,b.spec_share,b.quad];
            let mismatches=ac.iter().zip(&bc).map(|(a,b)|a.iter().zip(b).filter(|(x,y)|x.to_bits()!=y.to_bits()).count()).sum::<usize>();
            assert_eq!(mismatches,0,"n={n} trial={trial}");
            cases+=1;values+=n*8;
            println!("n={n} trial={trial} original={old:.6} candidate={new:.6} mismatches={mismatches}");
        }
    }
    println!("cases={cases} values={values} bit_mismatches=0");
}
