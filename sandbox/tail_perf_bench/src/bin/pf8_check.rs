//! pf8_check：PfAcc（增量 + 开放寻址众数）与 engine::preflight_quality_check 的逐位对账 + 单线程计时。
//!
//! 覆盖：
//!  1. 真数据：≥4 个因子 ×（raw 13 面 + fold 13 面），块大小 64 / 997 / 2818 各喂一遍，
//!     三项统计（to_bits）+ passed 与生产全量结果逐位比较；
//!  2. free-mask 与 production 的 is_free 判定逐格一致（22.1M 格）；
//!  3. definitely_failed 早停安全性：任何前缀上提前返回 true ⇒ 最终必须 !passed；
//!  4. 快路径（majority_count_threshold ≥ n_stocks）：生产跳过众数、majority_count_mean
//!     返回 n_stocks —— zero/nan/passed 逐位一致且 majority 恰为 n_stocks；
//!  5. 合成边界：空日期 / 单格 / 全 NaN / ±0 / ±inf / 阈值边界；
//!  6. 计时：26 面 prod vs pf8（单线程，块大小 64，不含一次性 free-mask 建表）。
//!
//! 运行：V3_DATA=/home/chenzongwei/neu_lab/data_yupei ./target/release/pf8_check
//! 可选：PF8_NFACTORS=8 调整因子数（默认 4）。

use std::time::Instant;

use ndarray::{s, Array2, ArrayView2};

use tail_perf_bench::{engine, npy, pf8, preflightopt};

const DATA_YUPEI: &str = "/home/chenzongwei/neu_lab/data_yupei";
/// 生产 BLOCK_ROWS = 64；另测 997（不整除）与 2818（整张一行一块）覆盖边界。
const BLOCK_SIZES: [usize; 3] = [64, 997, 2818];
const THR_MAJ: f64 = 200.0;
const THR_ZERO: f64 = 0.1;
const THR_NAN: f64 = 0.04;

#[derive(Default, Clone, Copy)]
struct Report {
    pass: usize,
    fail: usize,
    faces: usize,
    early_trip: usize,
    early_false: usize,
    /// engine::preflight_quality_check（sandbox 复刻，HashMap 众数）
    prod_s: f64,
    /// preflightopt::preflight_prod（生产线上真正的 4 趟基数排序路径）
    prod_radix_s: f64,
    /// pf8：PfAcc push_block + finish（块大小 64）
    pf8_s: f64,
}

impl Report {
    fn merge(&mut self, o: &Report) {
        self.pass += o.pass;
        self.fail += o.fail;
        self.faces += o.faces;
        self.early_trip += o.early_trip;
        self.early_false += o.early_false;
        self.prod_s += o.prod_s;
        self.prod_radix_s += o.prod_radix_s;
        self.pf8_s += o.pf8_s;
    }
}

/// 单个面：prod 全量一遍 vs pf8 按三种块大小各喂一遍。
fn check_face(
    face: &Array2<f32>,
    mask: &pf8::FreeMask,
    restrict: &ArrayView2<f32>,
    thr_maj: f64,
    thr_zero: f64,
    thr_nan: f64,
) -> Report {
    let (n_dates, n_stocks) = face.dim();
    let mut rep = Report {
        faces: 1,
        ..Default::default()
    };

    let t = Instant::now();
    let prod = engine::preflight_quality_check(&face.view(), restrict, thr_maj, thr_zero, thr_nan);
    rep.prod_s = t.elapsed().as_secs_f64();

    // 生产线上真正的算法（4 趟基数排序）复刻；顺带断言它与 engine 版逐位一致。
    let t = Instant::now();
    let prod_radix =
        preflightopt::preflight_prod(&face.view(), restrict, thr_maj, thr_zero, thr_nan);
    rep.prod_radix_s = t.elapsed().as_secs_f64();
    if prod_radix.passed != prod.passed
        || prod_radix.majority_count_mean.to_bits() != prod.majority_count_mean.to_bits()
        || prod_radix.zero_ratio_mean.to_bits() != prod.zero_ratio_mean.to_bits()
        || prod_radix.nan_ratio_mean.to_bits() != prod.nan_ratio_mean.to_bits()
    {
        rep.fail += 1;
        println!("  [FAIL] 两个生产参照（engine HashMap / preflightopt 基数）互相不一致");
    }

    for (bi, &bs) in BLOCK_SIZES.iter().enumerate() {
        let mut acc = pf8::PfAcc::new(n_dates, n_stocks, thr_maj, thr_zero, thr_nan);
        let mut tripped = false;
        let t = Instant::now();
        let mut start = 0usize;
        while start < n_dates {
            let end = (start + bs).min(n_dates);
            acc.push_block(&face.slice(s![start..end, ..]), mask, start);
            if acc.definitely_failed() {
                tripped = true;
            }
            start = end;
        }
        let el = t.elapsed().as_secs_f64();
        if bi == 0 {
            rep.pf8_s = el; // 计时只取生产块大小 64
        }
        let r = acc.finish();

        let eq = r.passed == prod.passed
            && r.majority_count_mean.to_bits() == prod.majority_count_mean.to_bits()
            && r.zero_ratio_mean.to_bits() == prod.zero_ratio_mean.to_bits()
            && r.nan_ratio_mean.to_bits() == prod.nan_ratio_mean.to_bits();
        if eq {
            rep.pass += 1;
        } else {
            rep.fail += 1;
            println!(
                "  [FAIL] 逐位不一致 shape=({n_dates},{n_stocks}) bs={bs}\n         pf8 : passed={} maj={:.17e} zero={:.17e} nan={:.17e}\n         prod: passed={} maj={:.17e} zero={:.17e} nan={:.17e}",
                r.passed,
                r.majority_count_mean,
                r.zero_ratio_mean,
                r.nan_ratio_mean,
                prod.passed,
                prod.majority_count_mean,
                prod.zero_ratio_mean,
                prod.nan_ratio_mean
            );
        }
        if tripped {
            rep.early_trip = 1;
            if r.passed {
                rep.early_false += 1;
                println!("  [FAIL] definitely_failed 误报：bs={bs} 前缀提前返回 true，但最终 passed");
            }
        }
    }
    rep
}

/// 快路径：thr_maj = 1e9 ≥ n_stocks。
///
/// 参照仍是 `engine::preflight_quality_check`（sandbox 版没有快路径，会真算众数）：
/// 其 zero/nan 累加与生产快路径逐位相同；`passed = maj<=1e9 && zero<thr && nan<thr`，
/// 而 maj ≤ n_stocks < 1e9 恒真，所以 passed 与生产快路径等价。
/// 唯一要单独断言的是生产快路径的 majority_count_mean = n_stocks（占位）。
fn check_face_fastpath(
    face: &Array2<f32>,
    mask: &pf8::FreeMask,
    restrict: &ArrayView2<f32>,
) -> Report {
    let (n_dates, n_stocks) = face.dim();
    let thr_maj = 1.0e9f64;
    let mut rep = Report {
        faces: 1,
        ..Default::default()
    };
    let prod = engine::preflight_quality_check(&face.view(), restrict, thr_maj, THR_ZERO, THR_NAN);

    for (bi, &bs) in BLOCK_SIZES.iter().enumerate() {
        let mut acc = pf8::PfAcc::new(n_dates, n_stocks, thr_maj, THR_ZERO, THR_NAN);
        let t = Instant::now();
        let mut start = 0usize;
        while start < n_dates {
            let end = (start + bs).min(n_dates);
            acc.push_block(&face.slice(s![start..end, ..]), mask, start);
            start = end;
        }
        let el = t.elapsed().as_secs_f64();
        if bi == 0 {
            rep.pf8_s = el; // 无众数计数的「地板」耗时（pass1 + 掩码读）
        }
        let r = acc.finish();
        let eq = r.passed == prod.passed
            && r.zero_ratio_mean.to_bits() == prod.zero_ratio_mean.to_bits()
            && r.nan_ratio_mean.to_bits() == prod.nan_ratio_mean.to_bits()
            && r.majority_count_mean.to_bits() == (n_stocks as f64).to_bits();
        if eq {
            rep.pass += 1;
        } else {
            rep.fail += 1;
            println!(
                "  [FAIL] 快路径不一致 bs={bs}: pf8(passed={}, maj={}, zero={:.17e}, nan={:.17e}) prod(passed={}, zero={:.17e}, nan={:.17e})",
                r.passed,
                r.majority_count_mean,
                r.zero_ratio_mean,
                r.nan_ratio_mean,
                prod.passed,
                prod.zero_ratio_mean,
                prod.nan_ratio_mean
            );
        }
    }
    rep
}

/// 合成边界用例（小矩阵，覆盖空日期/单格/全 NaN/±0/±inf/阈值边界）。
fn synthetic_check() -> (usize, usize) {
    let shapes = [(0usize, 5usize), (1, 1), (1, 9), (3, 2), (7, 13), (64, 257)];
    let thr_list = [
        (0.5f64, 0.1f64, 0.04f64),
        (1.0e9, 0.1, 0.04), // 快路径
        (0.0, 0.0, 0.0),    // 阈值边界
    ];
    let mut pass = 0usize;
    let mut fail = 0usize;

    for (nd, ns) in shapes {
        let mut data = Array2::<f32>::from_elem((nd, ns), f32::NAN);
        let mut restr = Array2::<f32>::from_elem((nd, ns), 1.0f32);
        let mut rng: u32 = 0x1234_5678;
        for t in 0..nd {
            for si in 0..ns {
                rng = rng.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                data[[t, si]] = match (rng >> 8) % 8 {
                    0 => f32::NAN,
                    1 => 0.0,
                    2 => -0.0,
                    3 => 1.0,
                    4 => (rng % 100) as f32 / 100.0,
                    5 => f32::INFINITY,
                    6 => f32::NEG_INFINITY,
                    _ => 0.5,
                };
                restr[[t, si]] = if rng % 2 == 0 { 0.0 } else { 1.0 };
            }
        }
        let mask = pf8::build_free_mask(&restr.view());
        for (tm, tz, tn) in thr_list {
            let prod =
                engine::preflight_quality_check(&data.view(), &restr.view(), tm, tz, tn);
            for &bs in &[1usize, 3, 1000] {
                let mut acc = pf8::PfAcc::new(nd, ns, tm, tz, tn);
                let mut start = 0usize;
                while start < nd {
                    let end = (start + bs).min(nd);
                    acc.push_block(&data.slice(s![start..end, ..]), &mask, start);
                    start = end;
                }
                let r = acc.finish();
                let mut ok = r.passed == prod.passed
                    && r.zero_ratio_mean.to_bits() == prod.zero_ratio_mean.to_bits()
                    && r.nan_ratio_mean.to_bits() == prod.nan_ratio_mean.to_bits();
                if tm < ns as f64 {
                    ok &= r.majority_count_mean.to_bits() == prod.majority_count_mean.to_bits();
                } else {
                    ok &= r.majority_count_mean.to_bits() == (ns as f64).to_bits();
                }
                if ok {
                    pass += 1;
                } else {
                    fail += 1;
                    println!(
                        "  [FAIL] 合成用例 shape=({nd},{ns}) thr=({tm},{tz},{tn}) bs={bs}: pf8(passed={}, maj={}, zero={:.17e}, nan={:.17e}) prod(passed={}, maj={}, zero={:.17e}, nan={:.17e})",
                        r.passed, r.majority_count_mean, r.zero_ratio_mean, r.nan_ratio_mean,
                        prod.passed, prod.majority_count_mean, prod.zero_ratio_mean, prod.nan_ratio_mean
                    );
                }
            }
        }
    }
    println!("[合成] 边界用例（空日期/单格/全 NaN/±0/±inf/阈值边界，含快路径）：PASS {pass} / FAIL {fail}");
    (pass, fail)
}

/// 依次产出 13 个派生面：smooth_1 + 每窗口 mean/max/min/std。
fn for_each_face(
    ranked: &Array2<f32>,
    windows: &[usize],
    mut f: impl FnMut(&str, &Array2<f32>),
) {
    f("smooth_1", ranked);
    for &w in windows {
        let min_periods = std::cmp::max(1, w / 2);
        let (mean, max, min, std) = engine::rolling_stats_f32_serial(ranked, w, min_periods);
        f(&format!("mean_smooth_{w}"), &mean);
        f(&format!("max_smooth_{w}"), &max);
        f(&format!("min_smooth_{w}"), &min);
        f(&format!("std_smooth_{w}"), &std);
    }
}

fn main() {
    let dir = std::env::var("V3_DATA").unwrap_or_else(|_| DATA_YUPEI.to_string());
    let n_factors: usize = std::env::var("PF8_NFACTORS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(4);

    println!("=== pf8_check ===");
    println!("data = {dir}");
    let restrict = npy::as_f32_mat(npy::load(&format!("{dir}/restrict.npy")));
    let (n_dates, n_stocks) = restrict.dim();
    println!(
        "restrict = {n_dates} × {n_stocks}（{} MB）",
        n_dates * n_stocks * 4 / 1_000_000
    );

    // ---- free-mask：一次性建表 + 与生产 is_free 逐格对账 ----
    let t = Instant::now();
    let mask = pf8::build_free_mask(&restrict.view());
    let mask_ms = t.elapsed().as_secs_f64() * 1e3;
    let t = Instant::now();
    let mut mask_bad = 0usize;
    let mut free_total = 0usize;
    for ti in 0..n_dates {
        for si in 0..n_stocks {
            let v = restrict[[ti, si]];
            let want = v.is_finite() && v == 0.0;
            if mask.is_free(ti, si) != want {
                mask_bad += 1;
            }
            free_total += want as usize;
        }
    }
    let mask_verify_ms = t.elapsed().as_secs_f64() * 1e3;
    println!(
        "[mask] build_free_mask {mask_ms:.1} ms（一次性）；与生产判定逐格对账 {mask_verify_ms:.0} ms：不一致 {mask_bad} / {} 格（free {free_total}）",
        n_dates * n_stocks
    );

    // ---- 合成边界 ----
    let (syn_pass, syn_fail) = synthetic_check();

    // ---- 真数据：因子 ×（raw 13 面 + fold 13 面） ----
    let names_txt = std::fs::read_to_string(format!("{dir}/sample_names.txt")).unwrap();
    let mut all_names: Vec<String> = names_txt
        .lines()
        .map(|l| l.trim().to_string())
        .filter(|l| !l.is_empty())
        .collect();
    // sample_names.txt 之外的 factor_*.npy 也补进来（多覆盖几个因子，排序保证确定性）
    if let Ok(rd) = std::fs::read_dir(&dir) {
        let mut extra: Vec<String> = rd
            .filter_map(|e| e.ok())
            .filter_map(|e| e.file_name().into_string().ok())
            .filter(|f| f.starts_with("factor_") && f.ends_with(".npy"))
            .map(|f| f["factor_".len()..f.len() - 4].to_string())
            .filter(|n| !all_names.contains(n))
            .collect();
        extra.sort();
        all_names.extend(extra);
    }
    let use_names: Vec<&String> = all_names.iter().take(n_factors.max(1)).collect();
    println!(
        "因子数 = {} / {}；块大小 = {BLOCK_SIZES:?}；阈值 = (maj {THR_MAJ}, zero {THR_ZERO}, nan {THR_NAN})",
        use_names.len(),
        all_names.len()
    );

    let windows = vec![5usize, 10, 20];
    let mut main_rep = Report::default();
    let mut fast_rep = Report::default();
    let mut faces_total = 0usize;

    for (fi, name) in use_names.iter().enumerate() {
        let raw = npy::as_f32_mat(npy::load(&format!("{dir}/factor_{name}.npy")));
        assert_eq!(raw.dim(), (n_dates, n_stocks), "因子 {name} 形状不符");
        let mut f_rep = Report::default();
        let mut fast_used = 0usize;

        // raw 变体 13 面
        let ranked = engine::rank_and_fill_missing_cross_sectional_median(&raw, &restrict);
        let mut n_face = 0usize;
        for_each_face(&ranked, &windows, |_label, face| {
            f_rep.merge(&check_face(
                face, &mask, &restrict.view(), THR_MAJ, THR_ZERO, THR_NAN,
            ));
            // 快路径只在第 1 个因子的 raw 变体上测（对照需真算众数，较贵）
            if fi == 0 && fast_used < 13 {
                fast_rep.merge(&check_face_fastpath(face, &mask, &restrict.view()));
                fast_used += 1;
            }
            n_face += 1;
        });
        drop(ranked);

        // fold 变体 13 面（与 v8/production 一致：fold 用未排名的 raw）
        let folded = engine::build_fold_values(&raw);
        let ranked_fold = engine::rank_and_fill_missing_cross_sectional_median(&folded, &restrict);
        drop(folded);
        for_each_face(&ranked_fold, &windows, |_label, face| {
            f_rep.merge(&check_face(
                face, &mask, &restrict.view(), THR_MAJ, THR_ZERO, THR_NAN,
            ));
            n_face += 1;
        });
        drop(ranked_fold);

        println!(
            "[{}/{}] {name}: {n_face} 面 × {} 块大小 = {} 次逐位比较 → PASS {} / FAIL {}；早停触发 {} 面（误报 {}）；prod {:.2}s vs pf8 {:.2}s",
            fi + 1,
            use_names.len(),
            BLOCK_SIZES.len(),
            f_rep.pass + f_rep.fail,
            f_rep.pass,
            f_rep.fail,
            f_rep.early_trip,
            f_rep.early_false,
            f_rep.prod_s,
            f_rep.pf8_s
        );
        faces_total += n_face;
        main_rep.merge(&f_rep);
    }

    // ---- 汇总 ----
    let per_face_prod = main_rep.prod_s / main_rep.faces.max(1) as f64 * 1e3;
    let per_face_radix = main_rep.prod_radix_s / main_rep.faces.max(1) as f64 * 1e3;
    let per_face_pf8 = main_rep.pf8_s / main_rep.faces.max(1) as f64 * 1e3;
    let speedup = if main_rep.pf8_s > 0.0 {
        main_rep.prod_s / main_rep.pf8_s
    } else {
        f64::NAN
    };
    let speedup_radix = if main_rep.pf8_s > 0.0 {
        main_rep.prod_radix_s / main_rep.pf8_s
    } else {
        f64::NAN
    };
    let floor_ms = fast_rep.pf8_s / fast_rep.faces.max(1) as f64 * 1e3;
    let ok = main_rep.fail == 0
        && syn_fail == 0
        && mask_bad == 0
        && main_rep.early_false == 0
        && fast_rep.fail == 0;

    println!();
    println!("=== 汇总 ===");
    println!(
        "[1] 逐位对账（真数据）：{} 因子 × {faces_total} 面 × {} 块大小 = {} 次比较 → PASS {} / FAIL {}",
        use_names.len(),
        BLOCK_SIZES.len(),
        main_rep.pass + main_rep.fail,
        main_rep.pass,
        main_rep.fail
    );
    println!(
        "[2] free-mask 逐格对账：不一致 {mask_bad} / {} 格 → {}",
        n_dates * n_stocks,
        if mask_bad == 0 { "PASS" } else { "FAIL" }
    );
    println!(
        "[3] 合成边界：PASS {syn_pass} / FAIL {syn_fail} → {}",
        if syn_fail == 0 { "PASS" } else { "FAIL" }
    );
    println!(
        "[4] definitely_failed 早停：触发 {} 面；误报（提前 true 但最终 passed）{} 例 → {}",
        main_rep.early_trip,
        main_rep.early_false,
        if main_rep.early_false == 0 { "PASS" } else { "FAIL" }
    );
    println!(
        "[5] 快路径（thr_maj=1e9 ≥ n_stocks）：{} 面 × {} 块大小 → PASS {} / FAIL {} → {}",
        fast_rep.faces,
        BLOCK_SIZES.len(),
        fast_rep.pass,
        fast_rep.fail,
        if fast_rep.fail == 0 { "PASS" } else { "FAIL" }
    );
    println!("[6] 计时（单线程，块大小 64，不含一次性 free-mask 建表）：");
    println!(
        "      prod A engine::preflight_quality_check（HashMap 众数）: {:.3} s / {} 面 = {per_face_prod:.1} ms/面 → pf8 加速 {speedup:.2}×",
        main_rep.prod_s, main_rep.faces
    );
    println!(
        "      prod B preflightopt::preflight_prod（4 趟基数排序）  : {:.3} s / {} 面 = {per_face_radix:.1} ms/面 → pf8 加速 {speedup_radix:.2}×",
        main_rep.prod_radix_s, main_rep.faces
    );
    println!(
        "      pf8    PfAcc(push_block + finish)                     : {:.3} s / {} 面 = {per_face_pf8:.1} ms/面",
        main_rep.pf8_s, main_rep.faces
    );
    println!(
        "      参考：无众数计数的地板（快路径 pass1+掩码读）{floor_ms:.1} ms/面；free-mask 建表一次性 {mask_ms:.1} ms（{} 面摊销 {:.2} ms/面）",
        main_rep.faces,
        mask_ms / main_rep.faces.max(1) as f64
    );
    println!("总体：{}", if ok { "PASS" } else { "FAIL" });
    if !ok {
        std::process::exit(1);
    }
}
