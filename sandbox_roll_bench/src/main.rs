//! rolling_stats 访存模式对照实验：现状（按列、跨步 31KB）vs 分列块（行主序流式）。
//! 目标：验证 tail_v2_rank_roll_factor::rolling_stats_f32_serial 的「未计时 29 s/因子」
//! 主要来自列优先访存导致的 cache miss 放大，并测量分块改写的实际收益。
//! 两个实现必须逐位一致（NaN 感知比较）。

use ndarray::Array2;
use std::time::Instant;

// ---------- 现状实现（逐字照抄 tail_v2_rank_roll_factor.rs） ----------
fn rolling_stats_for_column(
    ranked: &Array2<f32>,
    col_idx: usize,
    window: usize,
    min_periods: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let n_rows = ranked.nrows();
    let mut mean_out = vec![f32::NAN; n_rows];
    let mut max_out = vec![f32::NAN; n_rows];
    let mut min_out = vec![f32::NAN; n_rows];
    let mut std_out = vec![f32::NAN; n_rows];

    let mut sum = 0.0f64;
    let mut sumsq = 0.0f64;
    let mut count = 0usize;
    let mut max_deque = std::collections::VecDeque::<(usize, f32)>::new();
    let mut min_deque = std::collections::VecDeque::<(usize, f32)>::new();

    for row_idx in 0..n_rows {
        let value = ranked[[row_idx, col_idx]];
        if !value.is_nan() {
            let value64 = value as f64;
            sum += value64;
            sumsq += value64 * value64;
            count += 1;
            while let Some((_, tail_val)) = max_deque.back() {
                if *tail_val <= value {
                    max_deque.pop_back();
                } else {
                    break;
                }
            }
            max_deque.push_back((row_idx, value));
            while let Some((_, tail_val)) = min_deque.back() {
                if *tail_val >= value {
                    min_deque.pop_back();
                } else {
                    break;
                }
            }
            min_deque.push_back((row_idx, value));
        }

        if row_idx >= window {
            let leave_idx = row_idx - window;
            let leave_value = ranked[[leave_idx, col_idx]];
            if !leave_value.is_nan() {
                let leave64 = leave_value as f64;
                sum -= leave64;
                sumsq -= leave64 * leave64;
                count -= 1;
            }
        }

        let valid_start = (row_idx + 1).saturating_sub(window);
        while let Some((idx, _)) = max_deque.front() {
            if *idx < valid_start {
                max_deque.pop_front();
            } else {
                break;
            }
        }
        while let Some((idx, _)) = min_deque.front() {
            if *idx < valid_start {
                min_deque.pop_front();
            } else {
                break;
            }
        }

        if count >= min_periods {
            let mean = sum / count as f64;
            mean_out[row_idx] = mean as f32;
            max_out[row_idx] = max_deque.front().map(|item| item.1).unwrap_or(f32::NAN);
            min_out[row_idx] = min_deque.front().map(|item| item.1).unwrap_or(f32::NAN);
            if count > 1 {
                let variance =
                    ((sumsq - (sum * sum) / count as f64) / (count as f64 - 1.0)).max(0.0);
                std_out[row_idx] = variance.sqrt() as f32;
            }
        }
    }

    (mean_out, max_out, min_out, std_out)
}

fn old_serial(
    ranked: &Array2<f32>,
    window: usize,
    min_periods: usize,
) -> (Array2<f32>, Array2<f32>, Array2<f32>, Array2<f32>) {
    let n_rows = ranked.nrows();
    let n_cols = ranked.ncols();
    let mut mean = Array2::<f32>::from_elem((n_rows, n_cols), f32::NAN);
    let mut max = Array2::<f32>::from_elem((n_rows, n_cols), f32::NAN);
    let mut min = Array2::<f32>::from_elem((n_rows, n_cols), f32::NAN);
    let mut std = Array2::<f32>::from_elem((n_rows, n_cols), f32::NAN);
    for col_idx in 0..n_cols {
        let (mean_col, max_col, min_col, std_col) =
            rolling_stats_for_column(ranked, col_idx, window, min_periods);
        for row_idx in 0..n_rows {
            mean[[row_idx, col_idx]] = mean_col[row_idx];
            max[[row_idx, col_idx]] = max_col[row_idx];
            min[[row_idx, col_idx]] = min_col[row_idx];
            std[[row_idx, col_idx]] = std_col[row_idx];
        }
    }
    (mean, max, min, std)
}

// ---------- 分列块实现：行主序流式（读写都连续） ----------
fn new_blocked(
    ranked: &Array2<f32>,
    window: usize,
    min_periods: usize,
    block: usize,
) -> (Array2<f32>, Array2<f32>, Array2<f32>, Array2<f32>) {
    let (t, n) = ranked.dim();
    let mut mean = Array2::<f32>::from_elem((t, n), f32::NAN);
    let mut max = Array2::<f32>::from_elem((t, n), f32::NAN);
    let mut min = Array2::<f32>::from_elem((t, n), f32::NAN);
    let mut std = Array2::<f32>::from_elem((t, n), f32::NAN);

    let ring = window.next_power_of_two().max(1);
    let mask = ring - 1;

    let mut sum = vec![0.0f64; block];
    let mut sumsq = vec![0.0f64; block];
    let mut count = vec![0usize; block];
    let mut hmax = vec![0usize; block];
    let mut tmax = vec![0usize; block];
    let mut hmin = vec![0usize; block];
    let mut tmin = vec![0usize; block];
    let mut qimax = vec![0usize; block * ring];
    let mut qvmax = vec![0.0f32; block * ring];
    let mut qimin = vec![0usize; block * ring];
    let mut qvmin = vec![0.0f32; block * ring];

    for c0 in (0..n).step_by(block) {
        let b = block.min(n - c0);
        for j in 0..b {
            sum[j] = 0.0;
            sumsq[j] = 0.0;
            count[j] = 0;
            hmax[j] = 0;
            tmax[j] = 0;
            hmin[j] = 0;
            tmin[j] = 0;
        }
        for r in 0..t {
            let row = ranked.row(r);
            let src = row.as_slice().unwrap();
            for j in 0..b {
                let value = src[c0 + j];
                if !value.is_nan() {
                    let v64 = value as f64;
                    sum[j] += v64;
                    sumsq[j] += v64 * v64;
                    count[j] += 1;
                    let base = j * ring;
                    while tmax[j] > hmax[j] && qvmax[base + ((tmax[j] - 1) & mask)] <= value {
                        tmax[j] -= 1;
                    }
                    qimax[base + (tmax[j] & mask)] = r;
                    qvmax[base + (tmax[j] & mask)] = value;
                    tmax[j] += 1;
                    while tmin[j] > hmin[j] && qvmin[base + ((tmin[j] - 1) & mask)] >= value {
                        tmin[j] -= 1;
                    }
                    qimin[base + (tmin[j] & mask)] = r;
                    qvmin[base + (tmin[j] & mask)] = value;
                    tmin[j] += 1;
                }
            }
            if r >= window {
                let lrow = ranked.row(r - window);
                let lsrc = lrow.as_slice().unwrap();
                for j in 0..b {
                    let lv = lsrc[c0 + j];
                    if !lv.is_nan() {
                        let l64 = lv as f64;
                        sum[j] -= l64;
                        sumsq[j] -= l64 * l64;
                        count[j] -= 1;
                    }
                }
            }
            let valid_start = (r + 1).saturating_sub(window);
            let mut mrow = mean.row_mut(r);
            let mut xrow = max.row_mut(r);
            let mut nrow = min.row_mut(r);
            let mut srow = std.row_mut(r);
            let ms = mrow.as_slice_mut().unwrap();
            let xs = xrow.as_slice_mut().unwrap();
            let ns = nrow.as_slice_mut().unwrap();
            let ss = srow.as_slice_mut().unwrap();
            for j in 0..b {
                let base = j * ring;
                while tmax[j] > hmax[j] && qimax[base + (hmax[j] & mask)] < valid_start {
                    hmax[j] += 1;
                }
                while tmin[j] > hmin[j] && qimin[base + (hmin[j] & mask)] < valid_start {
                    hmin[j] += 1;
                }
                if count[j] >= min_periods {
                    let c = count[j] as f64;
                    let m = sum[j] / c;
                    ms[c0 + j] = m as f32;
                    xs[c0 + j] = if tmax[j] > hmax[j] {
                        qvmax[base + (hmax[j] & mask)]
                    } else {
                        f32::NAN
                    };
                    ns[c0 + j] = if tmin[j] > hmin[j] {
                        qvmin[base + (hmin[j] & mask)]
                    } else {
                        f32::NAN
                    };
                    if count[j] > 1 {
                        let variance =
                            ((sumsq[j] - (sum[j] * sum[j]) / c) / (c - 1.0)).max(0.0);
                        ss[c0 + j] = variance.sqrt() as f32;
                    }
                }
            }
        }
    }
    (mean, max, min, std)
}

// ---------- 变体 C：分块 + 免 NaN 预填（calloc 惰性零页 + 显式写 NaN） ----------
fn new_blocked_noprefill(
    ranked: &Array2<f32>,
    window: usize,
    min_periods: usize,
    block: usize,
) -> (Array2<f32>, Array2<f32>, Array2<f32>, Array2<f32>) {
    let (t, n) = ranked.dim();
    // 0.0 → calloc 惰性零页（不产生一次全量 memset）；所有格子后面都会被显式写入
    let mut mean = Array2::<f32>::from_elem((t, n), 0.0);
    let mut max = Array2::<f32>::from_elem((t, n), 0.0);
    let mut min = Array2::<f32>::from_elem((t, n), 0.0);
    let mut std = Array2::<f32>::from_elem((t, n), 0.0);

    let ring = window.next_power_of_two().max(1);
    let mask = ring - 1;

    let mut sum = vec![0.0f64; block];
    let mut sumsq = vec![0.0f64; block];
    let mut count = vec![0usize; block];
    let mut hmax = vec![0usize; block];
    let mut tmax = vec![0usize; block];
    let mut hmin = vec![0usize; block];
    let mut tmin = vec![0usize; block];
    let mut qimax = vec![0usize; block * ring];
    let mut qvmax = vec![0.0f32; block * ring];
    let mut qimin = vec![0usize; block * ring];
    let mut qvmin = vec![0.0f32; block * ring];
    let mut ring_val = vec![0.0f32; block * ring];

    for c0 in (0..n).step_by(block) {
        let b = block.min(n - c0);
        for j in 0..b {
            sum[j] = 0.0;
            sumsq[j] = 0.0;
            count[j] = 0;
            hmax[j] = 0;
            tmax[j] = 0;
            hmin[j] = 0;
            tmin[j] = 0;
        }
        for r in 0..t {
            let row = ranked.row(r);
            let src = row.as_slice().unwrap();
            for j in 0..b {
                let value = src[c0 + j];
                ring_val[j * ring + (r & mask)] = value;
                if !value.is_nan() {
                    let v64 = value as f64;
                    sum[j] += v64;
                    sumsq[j] += v64 * v64;
                    count[j] += 1;
                    let base = j * ring;
                    while tmax[j] > hmax[j] && qvmax[base + ((tmax[j] - 1) & mask)] <= value {
                        tmax[j] -= 1;
                    }
                    qimax[base + (tmax[j] & mask)] = r;
                    qvmax[base + (tmax[j] & mask)] = value;
                    tmax[j] += 1;
                    while tmin[j] > hmin[j] && qvmin[base + ((tmin[j] - 1) & mask)] >= value {
                        tmin[j] -= 1;
                    }
                    qimin[base + (tmin[j] & mask)] = r;
                    qvmin[base + (tmin[j] & mask)] = value;
                    tmin[j] += 1;
                }
            }
            if r >= window {
                // 离窗值来自本地环形缓冲（不再回读 ranked 的第 r-window 行）
                for j in 0..b {
                    let lv = ring_val[j * ring + ((r - window) & mask)];
                    if !lv.is_nan() {
                        let l64 = lv as f64;
                        sum[j] -= l64;
                        sumsq[j] -= l64 * l64;
                        count[j] -= 1;
                    }
                }
            }
            let valid_start = (r + 1).saturating_sub(window);
            let mut mrow = mean.row_mut(r);
            let mut xrow = max.row_mut(r);
            let mut nrow = min.row_mut(r);
            let mut srow = std.row_mut(r);
            let ms = mrow.as_slice_mut().unwrap();
            let xs = xrow.as_slice_mut().unwrap();
            let ns = nrow.as_slice_mut().unwrap();
            let ss = srow.as_slice_mut().unwrap();
            for j in 0..b {
                let base = j * ring;
                while tmax[j] > hmax[j] && qimax[base + (hmax[j] & mask)] < valid_start {
                    hmax[j] += 1;
                }
                while tmin[j] > hmin[j] && qimin[base + (hmin[j] & mask)] < valid_start {
                    hmin[j] += 1;
                }
                if count[j] >= min_periods {
                    let c = count[j] as f64;
                    let m = sum[j] / c;
                    ms[c0 + j] = m as f32;
                    xs[c0 + j] = if tmax[j] > hmax[j] {
                        qvmax[base + (hmax[j] & mask)]
                    } else {
                        f32::NAN
                    };
                    ns[c0 + j] = if tmin[j] > hmin[j] {
                        qvmin[base + (hmin[j] & mask)]
                    } else {
                        f32::NAN
                    };
                    if count[j] > 1 {
                        let variance =
                            ((sumsq[j] - (sum[j] * sum[j]) / c) / (c - 1.0)).max(0.0);
                        ss[c0 + j] = variance.sqrt() as f32;
                    } else {
                        ss[c0 + j] = f32::NAN;
                    }
                } else {
                    ms[c0 + j] = f32::NAN;
                    xs[c0 + j] = f32::NAN;
                    ns[c0 + j] = f32::NAN;
                    ss[c0 + j] = f32::NAN;
                }
            }
        }
    }
    (mean, max, min, std)
}

fn same(a: &Array2<f32>, b: &Array2<f32>) -> bool {
    a.iter()
        .zip(b.iter())
        .all(|(x, y)| (x.is_nan() && y.is_nan()) || x.to_bits() == y.to_bits())
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let t: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(2818);
    let n: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(7857);
    let block: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(128);
    let nthreads: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(1);

    // 造 rank 型数据：每日有效股票 ~52%，取值 1..n_valid（平均 rank 的半整数），
    // 与 yupei_dist 模板一致（T×N，NaN 47%）。
    let mut data = Array2::<f32>::from_elem((t, n), f32::NAN);
    let mut seed: u64 = 88172645463325252;
    let mut rnd = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };
    for r in 0..t {
        let mut row = data.row_mut(r);
        let s = row.as_slice_mut().unwrap();
        for v in s.iter_mut() {
            if rnd() % 100 < 52 {
                *v = ((rnd() % 7857) as f32) + 0.5;
            }
        }
    }
    println!(
        "模板 {} × {} = {:.1}M 格，NaN≈47%，并发 {} 线程",
        t,
        n,
        (t * n) as f64 / 1e6,
        nthreads
    );

    for window in [5usize, 10, 20] {
        let mp = std::cmp::max(1, window / 2);
        let t0 = Instant::now();
        let a = old_serial(&data, window, mp);
        let d_old = t0.elapsed().as_secs_f64();
        let t1 = Instant::now();
        let b = new_blocked(&data, window, mp, block);
        let d_new = t1.elapsed().as_secs_f64();
        let t2 = Instant::now();
        let c = new_blocked_noprefill(&data, window, mp, block);
        let d_c = t2.elapsed().as_secs_f64();
        let eq = same(&a.0, &b.0)
            && same(&a.1, &b.1)
            && same(&a.2, &b.2)
            && same(&a.3, &b.3)
            && same(&a.0, &c.0)
            && same(&a.1, &c.1)
            && same(&a.2, &c.2)
            && same(&a.3, &c.3);
        println!(
            "单线程 window={:>2}: 现状 {:.2}s | 分块 {:.2}s | 分块免预填 {:.2}s | 逐位一致 {}",
            window, d_old, d_new, d_c,
            if eq { "✓" } else { "✗" }
        );
    }

    if nthreads > 1 {
        // 并发对照：每线程在【线程内】分配并首次触碰自己的数据（NUMA 本地放置，
        // 与引擎 worker 行为一致），避免主线程统一分配造成的跨节点假象。
        for (label, which) in [("现状", 0usize), ("分块", 1), ("分块免预填", 2)] {
            let window = 10usize;
            let mp = std::cmp::max(1, window / 2);
            let t0 = Instant::now();
            std::thread::scope(|sc| {
                for k in 0..nthreads {
                    sc.spawn(move || {
                        let mut d = Array2::<f32>::from_elem((t, n), f32::NAN);
                        let mut s2 = 12345u64 + k as u64;
                        let mut r2 = || {
                            s2 ^= s2 << 13;
                            s2 ^= s2 >> 7;
                            s2 ^= s2 << 17;
                            s2
                        };
                        for r in 0..t {
                            let mut row = d.row_mut(r);
                            let sl = row.as_slice_mut().unwrap();
                            for v in sl.iter_mut() {
                                if r2() % 100 < 52 {
                                    *v = ((r2() % 7857) as f32) + 0.5;
                                }
                            }
                        }
                        match which {
                            0 => {
                                old_serial(&d, window, mp);
                            }
                            1 => {
                                new_blocked(&d, window, mp, block);
                            }
                            _ => {
                                new_blocked_noprefill(&d, window, mp, block);
                            }
                        }
                    });
                }
            });
            println!(
                "并发×{}（线程内分配）window={} {}: 墙钟 {:.2}s",
                nthreads,
                window,
                label,
                t0.elapsed().as_secs_f64()
            );
        }

        // 裸带宽标定：每线程流式 triad（读2写1），看机器实际能给多少 GB/s
        for k in [1usize, 24, 96, nthreads] {
            let bytes_per_thread = 88_400_000usize;
            let t0 = Instant::now();
            std::thread::scope(|sc| {
                for _ in 0..k {
                    sc.spawn(move || {
                        let len = bytes_per_thread / 4;
                        let mut a = vec![0.0f32; len];
                        let b = vec![1.0f32; len];
                        let mut c = vec![2.0f32; len];
                        for i in 0..len {
                            c[i] = a[i] + b[i];
                        }
                        a[0] = c[len - 1];
                    });
                }
            });
            let dt = t0.elapsed().as_secs_f64();
            let gb = (k * bytes_per_thread * 3) as f64 / 1e9;
            println!("裸带宽 ×{} 线程: {:.2}s → {:.1} GB/s 聚合", k, dt, gb / dt);
        }
    }
}
