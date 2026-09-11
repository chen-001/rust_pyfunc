//! neu8_check：验证 `neu8::neutralize_block`（按日期块中性化）与 `v3::v3_slot`（整张）
//! 逐位一致，并给出「全量 vs 按块」的计时。
//!
//! 构造 V3Shared 的方式与 src/main.rs::run_v3_bench 一致：
//!   `v2::build_shared(data_dir)` → `v3::v3_build(v2s)`
//!
//! 用法：
//!   V3_DATA=/home/chenzongwei/neu_lab/data_yupei ./target/release/neu8_check [因子数] [块1,块2,...]
//! 默认：2 个因子 × 13 个面，块大小 64,997,2818。

use std::time::Instant;

use ndarray::{s, Array2};

use tail_perf_bench::{engine, neu8, npy, v2, v3};

const DATA_DEFAULT: &str = "/home/chenzongwei/neu_lab/data_yupei";

/// 把 slot 按 `bs` 行分块，逐块调 `neu8::neutralize_block` 再拼回整张 (T,N)。
/// 返回（拼好的矩阵，总耗时）。
fn blocked_assemble(
    slot: &Array2<f32>,
    shared: &v3::V3Shared,
    bs: usize,
    scratch: &mut v3::V3Scratch,
) -> (Array2<f32>, f64) {
    let (t_rows, n) = slot.dim();
    let mut asm = Array2::<f32>::from_elem((t_rows, n), f32::NAN);
    let timer = Instant::now();
    let mut t0 = 0usize;
    while t0 < t_rows {
        let t1 = (t0 + bs).min(t_rows);
        let blk = slot.slice(s![t0..t1, ..]);
        let out = neu8::neutralize_block(&blk, shared, t0, t1, scratch);
        asm.slice_mut(s![t0..t1, ..]).assign(&out);
        t0 = t1;
    }
    (asm, timer.elapsed().as_secs_f64())
}

/// 旧占位实现（本次被替换掉的那个）：每块造一张 (T,N) 全 NaN 矩阵 → 把块塞进去 →
/// `v3::v3_slot` 跑**整张** → 截取块。仅用于对照计时。
fn placeholder_assemble(slot: &Array2<f32>, shared: &v3::V3Shared, bs: usize) -> (Array2<f32>, f64) {
    let (t_rows, n) = slot.dim();
    let mut asm = Array2::<f32>::from_elem((t_rows, n), f32::NAN);
    let timer = Instant::now();
    let mut t0 = 0usize;
    while t0 < t_rows {
        let t1 = (t0 + bs).min(t_rows);
        let mut full = Array2::<f32>::from_elem((t_rows, n), f32::NAN);
        full.slice_mut(s![t0..t1, ..]).assign(&slot.slice(s![t0..t1, ..]));
        let (out, _tv) = v3::v3_slot(full.view(), shared);
        asm.slice_mut(s![t0..t1, ..]).assign(&out.slice(s![t0..t1, ..]));
        t0 = t1;
    }
    (asm, timer.elapsed().as_secs_f64())
}

/// v8 的真实调用形态：块先落到 (block_rows, N) 的复用缓冲里，
/// 再传 `buf.slice(s![0..rows, ..])`（尾块不满），验证这种视图不会崩且结果一致。
fn v8_shape_assemble(
    slot: &Array2<f32>,
    shared: &v3::V3Shared,
    bs: usize,
    scratch: &mut v3::V3Scratch,
) -> Array2<f32> {
    let (t_rows, n) = slot.dim();
    let mut asm = Array2::<f32>::from_elem((t_rows, n), f32::NAN);
    let mut buf = Array2::<f32>::from_elem((bs, n), 0.0);
    let mut t0 = 0usize;
    while t0 < t_rows {
        let t1 = (t0 + bs).min(t_rows);
        let rows = t1 - t0;
        buf.slice_mut(s![0..rows, ..]).assign(&slot.slice(s![t0..t1, ..]));
        let blk = buf.slice(s![0..rows, ..]);
        let out = neu8::neutralize_block(&blk, shared, t0, t1, scratch);
        asm.slice_mut(s![t0..t1, ..]).assign(&out);
        t0 = t1;
    }
    asm
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n_factors: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(2);
    let blocks: Vec<usize> = match args.get(2) {
        Some(s) => s.split(',').filter_map(|x| x.parse().ok()).collect(),
        None => vec![64, 997, 2818],
    };
    let data = std::env::var("V3_DATA").unwrap_or_else(|_| DATA_DEFAULT.to_string());

    let timer = Instant::now();
    let v2s = v2::build_shared(&data);
    let t_v2 = timer.elapsed().as_secs_f64();
    let timer = Instant::now();
    let v3s = v3::v3_build(v2s);
    let t_v3 = timer.elapsed().as_secs_f64();
    let n = v3s.n_stocks;
    println!(
        "[setup] {data}\n         v2::build_shared={t_v2:.2}s v3::v3_build={t_v3:.2}s \
         T={} N={n} 非快路径日={} S==V 日={}",
        v3s.fast_ok.len(),
        v3s.fast_ok.iter().filter(|x| !**x).count(),
        v3s.identity_days
    );

    let names_txt = std::fs::read_to_string(format!("{data}/sample_names.txt")).unwrap();
    let names: Vec<&str> = names_txt
        .lines()
        .filter(|l| !l.trim().is_empty())
        .take(n_factors)
        .collect();
    assert!(!names.is_empty(), "sample_names.txt 没有因子名");
    let restrict_m = npy::as_f32_mat(npy::load(&format!("{data}/restrict.npy")));

    // 一份 scratch 跨块、跨面、跨因子复用（= v8.rs 每线程一份的用法）
    let mut scratch = v3::V3Scratch::new(n);

    let mut faces = 0usize;
    let mut pass_faces = 0usize;
    let mut mismatch_total = 0usize;
    let mut t_full_all = 0.0f64;
    let mut t_blk_all = vec![0.0f64; blocks.len()];
    let mut v8shape_ok = true;
    let mut ph_times: Vec<(usize, f64, f64)> = Vec::new();

    for nm in &names {
        let raw = npy::as_f32_mat(npy::load(&format!("{data}/factor_{nm}.npy")));
        let ranked = engine::rank_and_fill_missing_cross_sectional_median(&raw, &restrict_m);
        let (t_rows, ncols) = ranked.dim();
        assert_eq!(ncols, n);
        let mut slot_list: Vec<(String, Array2<f32>)> =
            vec![("smooth_1".to_string(), ranked.clone())];
        for &w in &[5usize, 10, 20] {
            let (m, x, mn, sd) = engine::rolling_stats_f32_rowmajor(&ranked, w, w / 2);
            slot_list.push((format!("mean_{w}"), m));
            slot_list.push((format!("max_{w}"), x));
            slot_list.push((format!("min_{w}"), mn));
            slot_list.push((format!("std_{w}"), sd));
        }
        println!("[{nm}] 面数={} T={t_rows} N={n}", slot_list.len());

        let mut t_full_factor = 0.0f64;
        let mut t_blk_factor = vec![0.0f64; blocks.len()];
        let mut factor_ok = true;

        for (fi, (tag, slot)) in slot_list.iter().enumerate() {
            let timer = Instant::now();
            let (full, _tv) = v3::v3_slot(slot.view(), &v3s);
            let t_full = timer.elapsed().as_secs_f64();
            t_full_factor += t_full;
            t_full_all += t_full;

            let mut line = format!("  {nm}::{tag} full={t_full:.3}s");
            let mut face_ok = true;
            let mut face_blk = vec![0.0f64; blocks.len()];
            for (bi, &bs) in blocks.iter().enumerate() {
                let (asm, t_blk) = blocked_assemble(slot, &v3s, bs, &mut scratch);
                face_blk[bi] = t_blk;
                t_blk_factor[bi] += t_blk;
                t_blk_all[bi] += t_blk;
                let (eq, mm) = v3::bitwise_equal(&full, &asm);
                if !eq {
                    face_ok = false;
                    factor_ok = false;
                    mismatch_total += mm;
                }
                line.push_str(&format!(
                    " | blk{bs}={t_blk:.3}s {} mm={mm}",
                    if eq { "PASS" } else { "FAIL" }
                ));
            }
            // v8 真实形态（缓冲切片 + 部分尾块）只对第一个面做一次
            if fi == 0 {
                let bs = blocks[0];
                let asm = v8_shape_assemble(slot, &v3s, bs, &mut scratch);
                let (eq, mm) = v3::bitwise_equal(&full, &asm);
                v8shape_ok &= eq;
                line.push_str(&format!(
                    " | v8形态(bs={bs},缓冲切片) {} mm={mm}",
                    if eq { "PASS" } else { "FAIL" }
                ));
                if !eq {
                    mismatch_total += mm;
                }
            }
            // 旧占位实现对照（只对第一个因子的第一个面做，证明去掉物化的收益）
            if nm == &names[0] && fi == 0 {
                for (bi, &bs) in blocks.iter().enumerate().take(2) {
                    let (asm_ph, t_ph) = placeholder_assemble(slot, &v3s, bs);
                    let (eq_ph, mm_ph) = v3::bitwise_equal(&full, &asm_ph);
                    ph_times.push((bs, t_ph, face_blk[bi]));
                    println!(
                        "  [对照] 旧占位实现(每块物化 (T,N) 全 NaN 矩阵) bs={bs}: {t_ph:.3}s  \
                         逐位 {} mm={mm_ph}",
                        if eq_ph { "PASS" } else { "FAIL" }
                    );
                }
            }
            faces += 1;
            if face_ok {
                pass_faces += 1;
            }
            println!("{line}");
        }
        println!(
            "[{nm}] 13 面合计：全量={t_full_factor:.3}s  按块={}  面一致={}",
            blocks
                .iter()
                .enumerate()
                .map(|(bi, bs)| format!("blk{bs}={:.3}s", t_blk_factor[bi]))
                .collect::<Vec<_>>()
                .join(" "),
            if factor_ok { "PASS" } else { "FAIL" }
        );
    }

    println!("\n==== 汇总 ====");
    println!(
        "因子={} {:?}  面数={faces}  块大小={blocks:?}",
        names.len(),
        names
    );
    println!(
        "逐位一致：{}  面级 PASS={pass_faces}/{faces}  不一致格数合计={mismatch_total}  \
         v8形态={}",
        if pass_faces == faces && mismatch_total == 0 && v8shape_ok {
            "PASS"
        } else {
            "FAIL"
        },
        if v8shape_ok { "PASS" } else { "FAIL" }
    );
    println!("计时（全部面合计，单位 s）：");
    println!("  全量 v3_slot           : {t_full_all:.3}");
    for (bi, &bs) in blocks.iter().enumerate() {
        println!(
            "  按块 neutralize_block bs={bs:<5}: {:.3}  (相对全量 {:.2}x)",
            t_blk_all[bi],
            t_blk_all[bi] / t_full_all
        );
    }
    if !ph_times.is_empty() {
        println!("对照（仅 {} 的 smooth_1 单面，不参与上面的合计）：", names[0]);
        for &(bs, t_ph, t_new) in &ph_times {
            println!(
                "  旧占位实现（每块物化 (T,N) 全 NaN 矩阵）bs={bs}: {t_ph:.3}s  \
                 vs 新按块 {t_new:.3}s  →  慢 {:.1}x",
                t_ph / t_new
            );
        }
    }
}
