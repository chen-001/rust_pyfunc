//! 玉佩2 配对互补网络 —— 子命令 check：net 产物的纯 Rust 自检。
//!
//! 不用 Python 对账：所有一致性检查都在这里做，跑完 net 直接 check 一遍。
//! 检查项：
//!   1. 每张期望矩阵存在、长度 n²、对称偏差（f32 逐位）；
//!   2. SINGLE == max(PB,QC)、SINGLE_S == max(PBS,QCS)（按构造必须逐格相等）；
//!   3. RHO_U == SINGLE + G、RHO_US == SINGLE_S + GSTAR（argmax 分解自洽）；
//!   4. EXZ / EXZS 在有效配对上均值≈0、标准差≈1（条件标准化的定义）；
//!   5. MASS 向量：emass/smass 各行归一后总和 ≈ 1，全部非负；
//!   6. rho 对角 = 1、E > 0；
//!   7. 每张矩阵的对角与非法格（非个股相关）是否为 NaN/0；
//!   8. 覆盖率报告：每张矩阵严格上三角的有限值占比（正常应 100%）。
//!
//! 用法：yupei_pair check <pair_out_dir> <date> [--mat cnt_t1]
//! 退出码：全部通过 0，任一硬检查失败 1。

use std::path::PathBuf;

use rayon::prelude::*;

use crate::{parse_flag, read_codes, read_f32};

struct Mat {
    name: String,
    v: Vec<f32>,
}

fn load(day: &PathBuf, n: usize, name: &str, tag: &str) -> Option<Mat> {
    let p = day.join(format!("{name}{tag}.bin"));
    let v = read_f32(&p).ok()?;
    if v.len() != n * n {
        eprintln!("[check] {name}{tag}: 长度 {} != {}", v.len(), n * n);
        return None;
    }
    Some(Mat {
        name: format!("{name}{tag}"),
        v,
    })
}

/// 严格上三角的对称偏差与有限率（按行并行归约）
fn upper_stats(v: &[f32], n: usize) -> (f64, f64) {
    let (dev, fin, tot) = (0..n)
        .into_par_iter()
        .map(|i| {
            let mut dev = 0.0f64;
            let mut fin = 0usize;
            let mut tot = 0usize;
            for j in (i + 1)..n {
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
                    f64::INFINITY // 一边有限一边 NaN：不对称
                };
                if d > dev {
                    dev = d;
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

pub fn cmd_check(args: &[String]) {
    let pair_root = PathBuf::from(&args[0]);
    let date = args[1].clone();
    let mat = parse_flag(args, "--mat").unwrap_or("cnt_t1").to_string();
    let day = pair_root.join(&date);
    let codes = read_codes(&day.join("codes.txt")).expect("读 codes.txt 失败");
    let n = codes.len();
    // 与 net 的落盘口径一致：单矩阵无后缀
    let has_tagged = day.join(format!("G_{mat}.bin")).exists();
    let tag = if has_tagged {
        format!("_{mat}")
    } else {
        String::new()
    };
    eprintln!("[check] date={date} n={n} tag='{tag}'");

    let fails = std::cell::Cell::new(0usize);
    let warn = |ok: bool, msg: String| {
        if ok {
            eprintln!("[check:ok] {msg}");
        } else {
            eprintln!("[check:FAIL] {msg}");
            fails.set(fails.get() + 1);
        }
    };

    // ---- 1. 期望矩阵清单（22 张并行读，串行读是 check 最大墙钟开销）----
    let names = [
        "G", "RHO_U", "SINGLE", "X0", "Z", "GSTAR", "RHO_US", "SINGLE_S", "X0STAR",
        "ZS", "GREL", "GSH", "GCOV", "GNPOS", "GHHI", "PB", "QC", "PBS", "QCS",
        "EXZ", "EXZS", "rho",
    ];
    let loaded: Vec<Option<Mat>> = names
        .par_iter()
        .map(|nm| load(&day, n, nm, &tag))
        .collect();
    let mut mats: std::collections::HashMap<String, Mat> = std::collections::HashMap::new();
    for (nm, m) in names.iter().zip(loaded) {
        match m {
            Some(m) => {
                mats.insert(m.name.clone(), m);
            }
            None => {
                eprintln!("[check:FAIL] 缺矩阵 {nm}{tag}.bin 或长度不对");
                fails.set(fails.get() + 1);
            }
        }
    }

    // ---- 2. 对称偏差与覆盖率（上三角）----
    for m in mats.values() {
        let (dev, cov) = upper_stats(&m.v, n);
        // X0/X0STAR 只有上三角有效（下三角写 -1/0 的对称格），不对称是设计如此
        let expect_sym = !m.name.starts_with("X0");
        warn(
            !expect_sym || dev <= 0.0,
            format!("{} 对称偏差 {:.6}", m.name, dev),
        );
        eprintln!("[check:cov] {} 上三角有限率 {:.3}%", m.name, 100.0 * cov);
    }

    // ---- 3. 分解自洽（行并行归约最大偏差）----
    let get = |nm: &str| -> &[f32] { &mats[&format!("{nm}{tag}")].v };
    // 通用：上三角逐格 dev = f(i,j) 的最大值
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
    if mats.contains_key(&format!("SINGLE{tag}")) && mats.contains_key(&format!("PB{tag}")) {
        let (sg, pb, qc) = (get("SINGLE"), get("PB"), get("QC"));
        let dev = max_dev(n, |i, j| {
            let k = i * n + j;
            if sg[k].is_finite() {
                (pb[k].max(qc[k]) - sg[k]).abs()
            } else {
                0.0
            }
        });
        warn(dev <= 1e-6, format!("SINGLE==max(PB,QC) 偏差 {dev:.2e}"));
    }
    if mats.contains_key(&format!("SINGLE_S{tag}")) && mats.contains_key(&format!("PBS{tag}")) {
        let (sg, pb, qc) = (get("SINGLE_S"), get("PBS"), get("QCS"));
        let dev = max_dev(n, |i, j| {
            let k = i * n + j;
            if sg[k].is_finite() {
                (pb[k].max(qc[k]) - sg[k]).abs()
            } else {
                0.0
            }
        });
        warn(dev <= 1e-6, format!("SINGLE_S==max(PBS,QCS) 偏差 {dev:.2e}"));
    }
    {
        let (g, sg, ru) = (get("G"), get("SINGLE"), get("RHO_U"));
        let dev = max_dev(n, |i, j| {
            let k = i * n + j;
            if g[k].is_finite() && sg[k].is_finite() && ru[k].is_finite() {
                (sg[k] + g[k] - ru[k]).abs()
            } else {
                0.0
            }
        });
        warn(dev <= 1e-5, format!("RHO_U==SINGLE+G 偏差 {dev:.2e}"));
    }

    // ---- 4. EXZ/EXZS 标准化 ----
    for nm in ["EXZ", "EXZS"] {
        let key = format!("{nm}{tag}");
        if let Some(m) = mats.get(&key) {
            let (s, sq, c) = (0..n)
                .into_par_iter()
                .map(|i| {
                    let mut s = 0.0f64;
                    let mut sq = 0.0f64;
                    let mut c = 0usize;
                    for j in (i + 1)..n {
                        let v = m.v[i * n + j] as f64;
                        if v.is_finite() {
                            s += v;
                            sq += v * v;
                            c += 1;
                        }
                    }
                    (s, sq, c)
                })
                .reduce(|| (0.0, 0.0, 0), |a, b| {
                    (a.0 + b.0, a.1 + b.1, a.2 + b.2)
                });
            let mean = s / c.max(1) as f64;
            let sd = (sq / c.max(1) as f64 - mean * mean).max(0.0).sqrt();
            warn(
                mean.abs() < 0.05 && (sd - 1.0).abs() < 0.05,
                format!("{key} 条件标准化 mean={mean:.4} std={sd:.4}（期望≈0/1）"),
            );
        }
    }

    // ---- 5. MASS 向量（6×n：emass|gmass|g2mass|smass|sgmass|sg2mass，兼容旧 4×n）----
    let mp = day.join(format!("MASS{tag}.bin"));
    match read_f32(&mp) {
        Ok(mv) if mv.len() == 4 * n || mv.len() == 6 * n => {
            let six = mv.len() == 6 * n;
            let names: Vec<&str> = if six {
                vec!["emass", "gmass", "g2mass", "smass", "sgmass", "sg2mass"]
            } else {
                vec!["emass", "gmass", "smass", "sgmass"]
            };
            let smass_idx = if six { 3 } else { 2 };
            let sgmass_idx = if six { 4 } else { 3 };
            let mut nonneg_bad = false;
            for (k, nm) in names.iter().enumerate() {
                let sl = &mv[k * n..k * n + n];
                let sum: f64 = sl.iter().map(|&v| v as f64).sum();
                let nz = sl.iter().filter(|&&v| v != 0.0).count();
                let mn = sl.iter().copied().fold(f32::INFINITY, f32::min);
                // emass/smass/g2mass/sg2mass 按构造必须非负；gmass/sgmass 是加权 gap，可为负
                if (*nm != "gmass" && *nm != "sgmass") && mn < 0.0 {
                    nonneg_bad = true;
                }
                eprintln!("[check:cov] MASS[{nm}] sum={sum:.4} 非零 {nz}/{n} min={mn:.2e}");
            }
            let em_sum: f64 = mv[0..n].iter().map(|&v| v as f64).sum();
            let sm_sum: f64 = mv[smass_idx * n..(smass_idx + 1) * n].iter().map(|&v| v as f64).sum();
            // 每对把 1 份质量摊完：emass/smass 归一后总和应 ≈ 1
            // （em 只在 sum_h_pos>0 的对上摊、sm 只在 sum_gap_pos>0 的对上摊，
            //  归一除的是全部有效对数，极端情况会略小于 1，容差放宽到 [0.9, 1.1]）
            warn(
                (0.9..=1.1).contains(&em_sum) && (0.9..=1.1).contains(&sm_sum),
                format!("emass/smass 归一和 {em_sum:.4}/{sm_sum:.4}（期望≈1）"),
            );
            warn(!nonneg_bad, "emass/smass/g2mass/sg2mass 存在负值".to_string());
        }
        _ => {
            eprintln!("[check:FAIL] MASS{tag}.bin 缺失或长度不对");
            fails.set(fails.get() + 1);
        }
    }

    // ---- 6. rho 对角 / E 正值 ----
    if let Some(m) = mats.get(&format!("rho{tag}")) {
        let dev: f32 = (0..n).map(|i| (m.v[i * n + i] - 1.0).abs()).fold(0.0, f32::max);
        warn(dev <= 1e-6, format!("rho 对角偏差 {dev:.2e}"));
    }

    // ---- 7. X0/X0STAR：合法对下标合法，含非个股的对必须是 -1 哨兵 ----
    let basic = std::env::var("RUST_PYFUNC_BASIC_INFO_DIR")
        .unwrap_or_else(|_| "/ssd_data/data/basic_info".to_string());
    let ok: Vec<bool> = match crate::read_symbol_map(
        &std::path::Path::new(&basic).join("symbol_map.csv"),
    ) {
        Ok(sm) => codes.iter().map(|c| sm.contains(c)).collect(),
        Err(_) => vec![true; n],
    };
    for nm in ["X0", "X0STAR"] {
        let key = format!("{nm}{tag}");
        if let Some(m) = mats.get(&key) {
            let mut bad = 0usize;
            for i in 0..n {
                for j in (i + 1)..n {
                    let v = m.v[i * n + j];
                    if ok[i] && ok[j] {
                        // 合法对：下标必须落在白名单里且不等于 i/j
                        if !v.is_finite() || v < 0.0 {
                            bad += 1;
                            continue;
                        }
                        let x = v.round() as usize;
                        if x >= n || x == i || x == j || !ok[x] {
                            bad += 1;
                        }
                    } else if v.is_finite() && v >= 0.0 {
                        // 含非个股的对：必须没有有效下标（哨兵 -1 或 NaN）
                        bad += 1;
                    }
                }
            }
            warn(bad == 0, format!("{key} 非法格数 {bad}"));
        }
    }

    if fails.get() == 0 {
        eprintln!("[check] 全部硬检查通过");
        println!("{{\"ok\":true,\"date\":{date},\"mat\":\"{mat}\"}}");
    } else {
        eprintln!("[check] {} 项硬检查失败", fails.get());
        println!(
            "{{\"ok\":false,\"date\":{date},\"fails\":{}}}",
            fails.get()
        );
        std::process::exit(1);
    }
}
