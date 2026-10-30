pub(crate) fn r6_plane_streaming(n: usize, z: &[f32], zs: &[f32], ok: &[bool]) -> R6 {
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
    if super::verbose() {
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
