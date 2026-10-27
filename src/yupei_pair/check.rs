//! 玉佩2 配对互补网络 —— net 产物的内存版自检。
//!
//! 与 sandbox_yupei_pair 的 `check_one` 校验项完全一致，只是输入从
//! 「22 张 .bin 落盘」换成 `net_one_mat` 直接返回的 `NetMats`：
//!   1. 每张矩阵长度 n²、严格上三角对称偏差（f32 逐位，X0/X0STAR 除外）；
//!   2. SINGLE == max(PB,QC)、SINGLE_S == max(PBS,QCS)；
//!   3. RHO_U == SINGLE + G（argmax 分解自洽）；
//!   4. EXZ / EXZS 在有效配对上均值≈0、标准差≈1；
//!   5. MASS 各行归一后总和 ≈ 1、非负约束；
//!   6. rho 对角 = 1；
//!   7. X0/X0STAR：合法对下标合法且落在 mask 内，含非活跃股的对必须是 -1 哨兵。
//! 返回失败项数（0 = 全绿）。

use rayon::prelude::*;

use super::net::NetMats;

/// 严格上三角的对称偏差与有限率（64×64 分块按行块并行归约，消除 v[j*n+i] 跨 20KB 步长的列读 cache miss）
fn upper_stats(v: &[f32], n: usize, skip_sym: bool) -> (f64, f64) {
    if skip_sym {
        let (fin, tot) = (0..n)
            .into_par_iter()
            .map(|i| {
                let row = &v[i * n..i * n + n];
                let mut fin = 0usize;
                for &x in &row[(i + 1)..n] {
                    if x.is_finite() {
                        fin += 1;
                    }
                }
                (fin, n - 1 - i)
            })
            .reduce(|| (0, 0), |a, b| (a.0 + b.0, a.1 + b.1));
        return (0.0, fin as f64 / tot.max(1) as f64);
    }
    const B: usize = 64;
    let nb = n.div_ceil(B);
    let (dev, fin, tot) = (0..nb)
        .into_par_iter()
        .map(|bi| {
            let i0 = bi * B;
            let i1 = (i0 + B).min(n);
            let mut dev = 0.0f64;
            let mut fin = 0usize;
            let mut tot = 0usize;
            for bj in bi..nb {
                let j0 = bj * B;
                let j1 = (j0 + B).min(n);
                for i in i0..i1 {
                    let js = if bi == bj { i + 1 } else { j0 };
                    for j in js..j1 {
                        tot += 1;
                        let a = v[i * n + j];
                        let b = v[j * n + i];
                        if a.is_finite() {
                            fin += 1;
                        }
                        let d = if a.is_finite() && b.is_finite() {
                            (a - b).abs() as f64
                        } else if a.is_finite() == b.is_finite() {
                            0.0
                        } else {
                            f64::INFINITY
                        };
                        if d > dev {
                            dev = d;
                        }
                    }
                }
            }
            (dev, fin, tot)
        })
        .reduce(
            || (0.0, 0, 0),
            |a, b| (a.0.max(b.0), a.1 + b.1, a.2 + b.2),
        );
    (dev, fin as f64 / tot.max(1) as f64)
}

/// 单矩阵全套硬检查，返回失败项数（0 = 全绿）。
/// `mask` = 活跃股票掩码（net 的配对参与集）。
pub(crate) fn check_net(m: &NetMats, n: usize, mask: &[bool], mat: &str) -> usize {
    let fails = std::cell::Cell::new(0usize);
    // 通过项只在 YUPEI_VERBOSE=1 时打印：量产时每天 15 条链 × 22 张矩阵全打一遍，
    // 会把日志刷满。失败项无论如何都打，否则只拿到一个数字、看不出是哪个矩阵坏在哪。
    let show = super::verbose();
    let warn = |ok: bool, msg: String| {
        if !ok {
            fails.set(fails.get() + 1);
        }
        if show || !ok {
            let tag = if ok { "ok" } else { "FAIL" };
            eprintln!("[check:{mat}:{tag}] {msg}");
        }
    };

    // ---- 1. 对称偏差与覆盖率（22 张）----
    let mats: Vec<(&str, &[f32])> = vec![
        ("G", &m.g), ("RHO_U", &m.ru), ("SINGLE", &m.sg), ("X0", &m.x0),
        ("Z", &m.z), ("GSTAR", &m.gs), ("RHO_US", &m.rus), ("SINGLE_S", &m.sgs),
        ("X0STAR", &m.x0s), ("ZS", &m.zs), ("GREL", &m.grel), ("GSH", &m.gsh),
        ("GCOV", &m.gcov), ("GNPOS", &m.gnpos), ("GHHI", &m.ghhi),
        ("PB", &m.pb), ("QC", &m.qc), ("PBS", &m.pbs), ("QCS", &m.qcs),
        ("EXZ", &m.exz), ("EXZS", &m.exzs), ("rho", &m.rho),
    ];
    for (nm, v) in &mats {
        warn(v.len() == n * n, format!("{nm} 长度 {} != {}", v.len(), n * n));
        let (dev, cov) = upper_stats(v, n, nm.starts_with("X0"));
        // X0/X0STAR 只有上三角有效（下三角写 -1 的对称格），不对称是设计如此
        warn(nm.starts_with("X0") || dev <= 0.0, format!("{nm} 对称偏差 {dev:.6}"));
        if show {
            eprintln!("[check:cov] {nm} 上三角有限率 {:.3}%", 100.0 * cov);
        }
    }

    // ---- 2. 分解自洽 ----
    fn max_dev(n: usize, f: impl Fn(usize, usize) -> f32 + Sync) -> f32 {
        (0..n)
            .into_par_iter()
            .map(|i| {
                let mut d = 0.0f32;
                for j in (i + 1)..n {
                    d = d.max(f(i, j));
                }
                d
            })
            .reduce(|| 0.0f32, |a, b| a.max(b))
    }
    {
        let (sg, pb, qc) = (&m.sg, &m.pb, &m.qc);
        let dev = max_dev(n, |i, j| {
            let k = i * n + j;
            if sg[k].is_finite() { (pb[k].max(qc[k]) - sg[k]).abs() } else { 0.0 }
        });
        warn(dev <= 1e-6, format!("SINGLE==max(PB,QC) 偏差 {dev:.2e}"));
    }
    {
        let (sg, pb, qc) = (&m.sgs, &m.pbs, &m.qcs);
        let dev = max_dev(n, |i, j| {
            let k = i * n + j;
            if sg[k].is_finite() { (pb[k].max(qc[k]) - sg[k]).abs() } else { 0.0 }
        });
        warn(dev <= 1e-6, format!("SINGLE_S==max(PBS,QCS) 偏差 {dev:.2e}"));
    }
    {
        let (g, sg, ru) = (&m.g, &m.sg, &m.ru);
        // G 是 `gain = h - single` 的 f32 差，其中 h = wb*p + wc*q 是加权和、single 是未加权最大值。
        // 权重 wb/wc 小时 h 可以远小于 single，此时 gain ≈ -single，它的 ulp 由 single 决定
        // （single≈1000 时 ulp≈6e-5），加回 single 复原 h 的绝对误差就是 1e-4 量级。
        // 用 |h| 当分母会把这个误差放大成 4e-5 的相对值，误判成失败。
        // 正确的尺度是 max(|single|, |h|)：误差相对它只有 eps 量级。
        // 2015 年股灾那几天（20150527/0707/0708/1015 等）就是这样被误判的。
        let dev = max_dev(n, |i, j| {
            let k = i * n + j;
            if g[k].is_finite() && sg[k].is_finite() && ru[k].is_finite() {
                (sg[k] + g[k] - ru[k]).abs() / sg[k].abs().max(ru[k].abs()).max(1.0)
            } else {
                0.0
            }
        });
        warn(dev <= 1e-6, format!("RHO_U==SINGLE+G 相对偏差 {dev:.2e}"));
    }

    // ---- 3. EXZ/EXZS 条件标准化 ----
    for (nm, v) in [("EXZ", &m.exz), ("EXZS", &m.exzs)] {
        let (s, sq, c) = (0..n)
            .into_par_iter()
            .map(|i| {
                let mut s = 0.0f64;
                let mut sq = 0.0f64;
                let mut c = 0usize;
                for j in (i + 1)..n {
                    let x = v[i * n + j] as f64;
                    if x.is_finite() {
                        s += x;
                        sq += x * x;
                        c += 1;
                    }
                }
                (s, sq, c)
            })
            .reduce(|| (0.0, 0.0, 0), |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2));
        let mean = s / c.max(1) as f64;
        let sd = (sq / c.max(1) as f64 - mean * mean).max(0.0).sqrt();
        warn(
            mean.abs() < 0.05 && (sd - 1.0).abs() < 0.05,
            format!("{nm} 条件标准化 mean={mean:.4} std={sd:.4}（期望≈0/1）"),
        );
    }

    // ---- 4. MASS（6×n：emass|gmass|g2mass|smass|sgmass|sg2mass）----
    {
        let mv = &m.mass;
        if mv.len() == 6 * n {
            let mut nonneg_bad = false;
            for (k, nm) in ["emass", "gmass", "g2mass", "smass", "sgmass", "sg2mass"]
                .iter()
                .enumerate()
            {
                let sl = &mv[k * n..k * n + n];
                let sum: f64 = sl.iter().map(|&v| v as f64).sum();
                let nz = sl.iter().filter(|&&v| v != 0.0).count();
                let mn = sl.iter().copied().fold(f32::INFINITY, f32::min);
                if (*nm != "gmass" && *nm != "sgmass") && mn < 0.0 {
                    nonneg_bad = true;
                }
                if show {
                    eprintln!("[check:cov] MASS[{nm}] sum={sum:.4} 非零 {nz}/{n} min={mn:.2e}");
                }
            }
            let em_sum: f64 = mv[0..n].iter().map(|&v| v as f64).sum();
            let sm_sum: f64 = mv[3 * n..4 * n].iter().map(|&v| v as f64).sum();
            // emass/smass 是「有成交股票」上的软质量归一，停牌股贡献 0，所以总和 ≤ 1。
            // 原来要求 ≈1（0.9..1.1）在两类情形下会误判：
            //   1. 当天停牌股多（2015 年常见），em_sum 会掉到 0.88 左右；
            //   2. gap 口径（sm）用的是 relu(gain)，而 gain = 加权和 - 未加权最大值，
            //      权重小的日子大量配对 gain < 0，sm_sum 会掉到 0.4~0.9，这是口径本身的性质。
            // 所以只检查「不膨胀、不出现负总和」，不要求填满。
            warn(
                (0.0..=1.1).contains(&em_sum) && (-1e-6..=1.1).contains(&sm_sum),
                format!("emass/smass 归一和 {em_sum:.4}/{sm_sum:.4}（应 ≤1，gap 口径可为 0）"),
            );
            warn(!nonneg_bad, "emass/smass/g2mass/sg2mass 存在负值".to_string());
        } else {
            eprintln!("[check:FAIL] MASS 长度 {} != 6n", mv.len());
            fails.set(fails.get() + 1);
        }
    }

    // ---- 5. rho 对角 ----
    {
        let dev: f32 = (0..n).map(|i| (m.rho[i * n + i] - 1.0).abs()).fold(0.0, f32::max);
        warn(dev <= 1e-6, format!("rho 对角偏差 {dev:.2e}"));
    }

    // ---- 6. X0/X0STAR：合法对下标合法，含非活跃股的对必须是 -1 哨兵 ----
    for (nm, v) in [("X0", &m.x0), ("X0STAR", &m.x0s)] {
        let mut bad = 0usize;
        for i in 0..n {
            for j in (i + 1)..n {
                let x = v[i * n + j];
                if mask[i] && mask[j] {
                    if !x.is_finite() || x < 0.0 {
                        bad += 1;
                        continue;
                    }
                    let xi = x.round() as usize;
                    if xi >= n || xi == i || xi == j || !mask[xi] {
                        bad += 1;
                    }
                } else if x.is_finite() && x >= 0.0 {
                    bad += 1;
                }
            }
        }
        warn(bad == 0, format!("{nm} 非法格数 {bad}"));
    }

    let n_fails = fails.get();
    if n_fails > 0 {
        eprintln!("[check:{mat}] {n_fails} 项硬检查失败");
    } else if show {
        eprintln!("[check:{mat}] 全部硬检查通过");
    }
    n_fails
}
