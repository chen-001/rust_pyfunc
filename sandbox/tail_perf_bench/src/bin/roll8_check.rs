//! roll8_check：`roll8::rolling_block` 的分块流式实现 vs 生产 `engine::rolling_stats_f32_serial`。
//!
//! 验收内容：
//!   1) 逐位对账：data_yupei 的 3 个因子 × 3 个窗口，块大小 64 / 997 / 2818，
//!      把 `rolling_block` 按块拼起来，与 `rolling_stats_f32_serial` 全量输出逐位比较
//!      （NaN 位置也必须一致），并检查「只写 0..(t1-t0) 行」的尾部未被污染。
//!   2) 附加：乱序/跳块调用、换窗口集、NaN 加密样本、w=1 边界。
//!   3) 单线程计时：分块流式 vs 生产全量，给出加速比。
//!
//! 运行：
//!   cd sandbox/tail_perf_bench && PATH="$HOME/.cargo/bin:$PATH" cargo build --release
//!   V3_DATA=/home/chenzongwei/neu_lab/data_yupei ./target/release/roll8_check

use std::time::Instant;

use ndarray::{s, Array2};

use tail_perf_bench::{engine, npy, roll8, t8};

const DEFAULT_DATA: &str = "/home/chenzongwei/neu_lab/data_yupei";
/// out 缓冲的未写行哨兵值（用来验证「只写第 0..rows 行」）。
const SENTINEL: f32 = 12345.678;

fn load_names(dir: &str) -> Vec<String> {
    let p = format!("{dir}/sample_names.txt");
    if let Ok(txt) = std::fs::read_to_string(&p) {
        let v: Vec<String> = txt
            .lines()
            .map(|l| l.trim().to_string())
            .filter(|l| {
                !l.is_empty() && std::path::Path::new(&format!("{dir}/factor_{l}.npy")).exists()
            })
            .collect();
        if !v.is_empty() {
            return v;
        }
    }
    [
        "assort_cnt_t005_corr",
        "assort_cnt_t01_nb_str_minus_mean",
        "assort_cnt_t02_corr",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect()
}

/// 参考实现：13 张 (T,N) 全量面（slot 0 = ranked 本身；随后每窗口 mean/max/min/std）。
fn build_refs(ranked: &Array2<f32>, windows: &[usize]) -> Vec<Array2<f32>> {
    let mut refs = vec![ranked.clone()];
    for &w in windows {
        let (m, x, n2, sd) = engine::rolling_stats_f32_serial(ranked, w, std::cmp::max(1, w / 2));
        refs.push(m);
        refs.push(x);
        refs.push(n2);
        refs.push(sd);
    }
    refs
}

/// 比较 out 缓冲第 0..rows 行（对应 ranked 的 [t0,t1)）与参考矩阵的 [t0,t1)，返回不一致格数。
fn cmp_rows(got: &Array2<f32>, want: &Array2<f32>, t0: usize, rows: usize) -> usize {
    let n = want.ncols();
    let g = got.as_slice().expect("got 非连续内存");
    let w = want.as_slice().expect("want 非连续内存");
    let mut bad = 0usize;
    for r in 0..rows {
        let gs = &g[r * n..r * n + n];
        let ws = &w[(t0 + r) * n..(t0 + r) * n + n];
        for c in 0..n {
            let (a, b) = (gs[c], ws[c]);
            if !((a.is_nan() && b.is_nan()) || a.to_bits() == b.to_bits()) {
                bad += 1;
            }
        }
    }
    bad
}

struct CheckOutcome {
    bad_per_slot: Vec<usize>,
    tail_bad: usize,
}

impl CheckOutcome {
    fn pass(&self) -> bool {
        self.tail_bad == 0 && self.bad_per_slot.iter().all(|&b| b == 0)
    }
    fn total_bad(&self) -> usize {
        self.bad_per_slot.iter().sum()
    }
}

/// 按块调用 rolling_block 并与参考逐位比较；同时验证尾部哨兵未被写。
fn check_blocks(
    ranked: &Array2<f32>,
    windows: &[usize],
    block: usize,
    refs: &[Array2<f32>],
    scratch: &mut roll8::RollScratch,
) -> CheckOutcome {
    let (t, n) = ranked.dim();
    let ns = t8::slot_count(windows);
    let mut outs: Vec<Array2<f32>> = (0..ns)
        .map(|_| Array2::<f32>::from_elem((block, n), SENTINEL))
        .collect();
    let mut bad_per_slot = vec![0usize; ns];
    let mut tail_bad = 0usize;

    for t0 in (0..t).step_by(block) {
        let t1 = (t0 + block).min(t);
        let rows = t1 - t0;
        // 每次调用前把「本次不该写的尾部行」刷成哨兵，调用后验证未被写。
        if rows < block {
            for out in outs.iter_mut() {
                out.slice_mut(s![rows..block, ..]).fill(SENTINEL);
            }
        }
        roll8::rolling_block(ranked, windows, t0, t1, scratch, &mut outs);
        for si in 0..ns {
            bad_per_slot[si] += cmp_rows(&outs[si], &refs[si], t0, rows);
        }
        if rows < block {
            for out in outs.iter() {
                let sl = out.as_slice().unwrap();
                for &v in &sl[rows * n..] {
                    if v.to_bits() != SENTINEL.to_bits() {
                        tail_bad += 1;
                    }
                }
            }
        }
    }
    CheckOutcome {
        bad_per_slot,
        tail_bad,
    }
}

fn print_outcome(tag: &str, windows: &[usize], oc: &CheckOutcome) -> bool {
    let names = t8::derived_names_for_variant("v", windows);
    if oc.pass() {
        println!(
            "  {tag:<26} PASS  槽位 {} 个，不一致 0 格，尾部未写 0 格",
            oc.bad_per_slot.len()
        );
        true
    } else {
        println!(
            "  {tag:<26} FAIL  不一致 {} 格，尾部被写 {} 格",
            oc.total_bad(),
            oc.tail_bad
        );
        for (si, &b) in oc.bad_per_slot.iter().enumerate() {
            if b > 0 {
                println!("      slot {si} ({}) 不一致 {b} 格", names[si]);
            }
        }
        false
    }
}

/// 生产全量：每个窗口 4 张 (T,N) 矩阵（各自 NaN 预填），返回 (秒, 逐窗口秒)。
fn time_prod(ranked: &Array2<f32>, windows: &[usize], reps: usize) -> (f64, Vec<f64>) {
    let mut best = f64::INFINITY;
    let mut per_w = vec![f64::INFINITY; windows.len()];
    for _ in 0..reps {
        let t_all = Instant::now();
        for (wi, &w) in windows.iter().enumerate() {
            let t0 = Instant::now();
            let out = engine::rolling_stats_f32_serial(ranked, w, std::cmp::max(1, w / 2));
            per_w[wi] = per_w[wi].min(t0.elapsed().as_secs_f64());
            std::hint::black_box(&out);
        }
        best = best.min(t_all.elapsed().as_secs_f64());
    }
    (best, per_w)
}

/// 分块流式：生产块大小（默认 64）下走完整 T。
fn time_stream(ranked: &Array2<f32>, windows: &[usize], block: usize, reps: usize) -> f64 {
    let (t, n) = ranked.dim();
    let ns = t8::slot_count(windows);
    let mut scratch = roll8::RollScratch::new(t, n, windows, block);
    let mut outs: Vec<Array2<f32>> = (0..ns)
        .map(|_| Array2::<f32>::from_elem((block, n), 0.0))
        .collect();
    let mut best = f64::INFINITY;
    for _ in 0..reps {
        let st = Instant::now();
        for t0 in (0..t).step_by(block) {
            let t1 = (t0 + block).min(t);
            roll8::rolling_block(ranked, windows, t0, t1, &mut scratch, &mut outs);
        }
        best = best.min(st.elapsed().as_secs_f64());
        std::hint::black_box(&outs);
    }
    best
}

fn nan_ratio(a: &Array2<f32>) -> f64 {
    let total = a.len() as f64;
    let nan = a.iter().filter(|v| v.is_nan()).count() as f64;
    nan / total
}

fn main() {
    let dir = std::env::var("V3_DATA").unwrap_or_else(|_| DEFAULT_DATA.to_string());
    let restrict = npy::as_f32_mat(npy::load(&format!("{dir}/restrict.npy")));
    let names = load_names(&dir);
    let factors: Vec<(String, Array2<f32>)> = names
        .iter()
        .take(3)
        .map(|nm| {
            (
                nm.clone(),
                npy::as_f32_mat(npy::load(&format!("{dir}/factor_{nm}.npy"))),
            )
        })
        .collect();

    let windows: Vec<usize> = vec![5, 10, 20];
    let blocks: [usize; 3] = [64, 997, 2818];
    let (t, n) = factors[0].1.dim();
    // ROLL8_CHECK_QUICK=1：只跑计时（迭代调优用），跳过逐位对账。
    let quick = std::env::var("ROLL8_CHECK_QUICK").is_ok();

    println!("== roll8_check：分块流式 rolling vs 生产全量 ==");
    println!("数据目录 : {dir}");
    println!("形状     : T={t}  N={n}");
    println!("windows  : {windows:?}   blocks: {blocks:?}");
    println!();

    let mut all_pass = true;
    let mut ranked_list: Vec<Array2<f32>> = Vec::new();
    for (name, raw) in factors.iter() {
        ranked_list.push(engine::rank_and_fill_missing_cross_sectional_median(raw, &restrict));
        println!("[因子] {name}  ranked NaN 比例 {:.4}", nan_ratio(ranked_list.last().unwrap()));
    }
    println!();

    if !quick {
        // ---- 1) 逐位对账：3 因子 × 3 窗口 × 3 块大小 ----
        println!("---- 1) 逐位对账（vs engine::rolling_stats_f32_serial，含 NaN 位置）----");
        let mut scratch_by_block: Vec<roll8::RollScratch> = blocks
            .iter()
            .map(|&b| roll8::RollScratch::new(t, n, &windows, b))
            .collect();
        for (fi, (name, _)) in factors.iter().enumerate() {
            let ranked = &ranked_list[fi];
            let refs = build_refs(ranked, &windows);
            println!("  [{name}]");
            for (bi, &block) in blocks.iter().enumerate() {
                let oc = check_blocks(ranked, &windows, block, &refs, &mut scratch_by_block[bi]);
                all_pass &= print_outcome(&format!("block={block}"), &windows, &oc);
            }
        }
        println!();

        // ---- 2) 附加鲁棒性 ----
        println!("---- 2) 附加：乱序/跳块、换窗口集、NaN 加密、w=1 边界 ----");
        {
            // 乱序：先 [t/2, t) 再 [0, t/2)（首次调用即从 0 重放）
            let ranked = &ranked_list[0];
            let refs = build_refs(ranked, &windows);
            let ns = t8::slot_count(&windows);
            let mut scratch = roll8::RollScratch::new(t, n, &windows, 64);
            let mut outs: Vec<Array2<f32>> = (0..ns)
                .map(|_| Array2::<f32>::from_elem((t, n), SENTINEL))
                .collect();
            let half = t / 2;
            let mut bad = 0usize;
            roll8::rolling_block(ranked, &windows, half, t, &mut scratch, &mut outs);
            for si in 0..ns {
                bad += cmp_rows(&outs[si], &refs[si], half, t - half);
            }
            roll8::rolling_block(ranked, &windows, 0, half, &mut scratch, &mut outs);
            for si in 0..ns {
                bad += cmp_rows(&outs[si], &refs[si], 0, half);
            }
            println!("  {:<26} {}", "乱序 [t/2,t) → [0,t/2)", if bad == 0 { "PASS" } else { "FAIL" });
            if bad != 0 {
                println!("      不一致 {bad} 格");
            }
            all_pass &= bad == 0;
        }
        {
            // 换窗口集：同一个 scratch 从 [5,10,20] 切到 [1,2,7]（w=1 → min_periods=1、std 恒 NaN）
            let windows2: Vec<usize> = vec![1, 2, 7];
            let ranked = &ranked_list[0];
            let refs2 = build_refs(ranked, &windows2);
            let mut scratch = roll8::RollScratch::new(t, n, &windows, 64);
            let _ = check_blocks(ranked, &windows, 64, &build_refs(ranked, &windows), &mut scratch);
            let oc = check_blocks(ranked, &windows2, 997, &refs2, &mut scratch);
            all_pass &= print_outcome("换窗口集 [1,2,7]", &windows2, &oc);
        }
        {
            // NaN 加密：把 1/3 的格子清成 NaN（再额外注入 1/5 的 NaN），验证 NaN 位置逐位一致
            let ranked = &ranked_list[0];
            let mut dense = ranked.clone();
            for (i, v) in dense.iter_mut().enumerate() {
                if i % 3 == 0 {
                    *v = f32::NAN;
                } else if i % 5 == 0 {
                    *v = 0.0;
                }
            }
            println!("  {:<26} ranked NaN 比例 {:.4}", "NaN 加密样本", nan_ratio(&dense));
            let refs = build_refs(&dense, &windows);
            let mut scratch = roll8::RollScratch::new(t, n, &windows, 64);
            let oc = check_blocks(&dense, &windows, 64, &refs, &mut scratch);
            all_pass &= print_outcome("NaN 加密 block=64", &windows, &oc);
            let oc = check_blocks(&dense, &windows, 2818, &refs, &mut scratch);
            all_pass &= print_outcome("NaN 加密 block=2818", &windows, &oc);
        }
        {
            // 等值 / ±0.0 密集样本：前后缀极值法必须复刻 deque 的 `tail <= v` 弹尾语义
            // （值相等、含 +0.0/-0.0 时保留较新的那次出现）。
            let (tt, nn) = (600usize, 256usize);
            let mut tie = Array2::<f32>::zeros((tt, nn));
            let mut seed = 0x1234_5678_9abc_def0u64;
            let mut rnd = || {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                seed
            };
            let pool = [0.0f32, -0.0, 1.0, -1.0, 2.0, f32::NAN, 0.0, 1.0, -0.0];
            for v in tie.iter_mut() {
                *v = pool[(rnd() % pool.len() as u64) as usize];
            }
            let n_zero = tie.iter().filter(|v| v.to_bits() == 0.0f32.to_bits()).count();
            let n_neg_zero = tie.iter().filter(|v| v.to_bits() == (-0.0f32).to_bits()).count();
            println!(
                "  {:<26} +0.0 {} 格 / -0.0 {} 格 / NaN {} 格",
                "等值±0.0 密集样本",
                n_zero,
                n_neg_zero,
                tie.iter().filter(|v| v.is_nan()).count()
            );
            let refs = build_refs(&tie, &windows);
            let mut scratch = roll8::RollScratch::new(tt, nn, &windows, 64);
            let oc = check_blocks(&tie, &windows, 64, &refs, &mut scratch);
            all_pass &= print_outcome("等值±0.0 block=64", &windows, &oc);
            let oc = check_blocks(&tie, &windows, 997, &refs, &mut scratch);
            all_pass &= print_outcome("等值±0.0 block=997", &windows, &oc);
            let windows3: Vec<usize> = vec![1, 2, 3, 7];
            let refs3 = build_refs(&tie, &windows3);
            let oc = check_blocks(&tie, &windows3, 64, &refs3, &mut scratch);
            all_pass &= print_outcome("等值±0.0 w=[1,2,3,7]", &windows3, &oc);
        }
        {
            // 单调 / 常量 / 常量+NaN：极值算法的退化与最坏形状
            let (tt, nn) = (300usize, 128usize);
            let mut mono = Array2::<f32>::zeros((tt, nn));
            for r in 0..tt {
                for c in 0..nn {
                    mono[[r, c]] = match c % 4 {
                        0 => r as f32,        // 严格递增
                        1 => (tt - r) as f32, // 严格递减
                        2 => 5.0,             // 常量
                        _ => {
                            if r % 3 == 0 {
                                f32::NAN
                            } else {
                                5.0
                            }
                        } // 常量 + NaN
                    };
                }
            }
            let refs = build_refs(&mono, &windows);
            let mut scratch = roll8::RollScratch::new(tt, nn, &windows, 64);
            let oc = check_blocks(&mono, &windows, 64, &refs, &mut scratch);
            all_pass &= print_outcome("单调/常量/常量+NaN", &windows, &oc);
        }
        println!();
    }

    // ---- 3) 单线程计时 ----
    let reps = 3;
    println!("---- 3) 单线程计时（best-of-{reps}，3 因子）----");
    let mut prod_sum = 0.0f64;
    let mut stream_sum = [0.0f64; 3];
    let mut per_w_mean = vec![0.0f64; windows.len()];
    for (fi, (name, _)) in factors.iter().enumerate() {
        let ranked = &ranked_list[fi];
        let (prod, per_w) = time_prod(ranked, &windows, reps);
        prod_sum += prod;
        for (wi, &pw) in per_w.iter().enumerate() {
            per_w_mean[wi] += pw;
        }
        let mut line = format!("  [{name}] 生产全量 {prod:.3}s |");
        for (bi, &block) in blocks.iter().enumerate() {
            let st = time_stream(ranked, &windows, block, reps);
            stream_sum[bi] += st;
            line.push_str(&format!("  流式 block={block} {st:.3}s ({:.2}x)", prod / st));
        }
        println!("{line}");
    }
    let k = factors.len() as f64;
    println!();
    println!(
        "  生产全量（每窗口 4 张 (T,N) 矩阵，NaN 预填） 平均 {:.3}s/因子  逐窗口 {:.3}/{:.3}/{:.3}s",
        prod_sum / k,
        per_w_mean[0] / k,
        per_w_mean[1] / k,
        per_w_mean[2] / k
    );
    for (bi, &block) in blocks.iter().enumerate() {
        println!(
            "  分块流式 block={block:<5}                平均 {:.3}s/因子  加速比 {:.2}x",
            stream_sum[bi] / k,
            prod_sum / stream_sum[bi]
        );
    }
    println!();

    if quick {
        println!("== 结论: QUICK 模式（只计时，未跑对账）==");
    } else {
        println!(
            "== 结论: 逐位一致 {} ==",
            if all_pass { "PASS（全部用例 0 格不一致）" } else { "FAIL（见上）" }
        );
        if !all_pass {
            std::process::exit(1);
        }
    }
}
