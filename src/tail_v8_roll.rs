//! tail_v8_roll：生产库版「按日期块流式产出 13 个派生面」（v8 一档 / 二档生产者侧）。
//!
//! 现状（v7 `process_v7_variant`）：每个窗口先调 `rolling_stats_f32_serial` 算出完整
//! 4 张 (T,N) 矩阵（各自先 NaN 预填一遍），12 张落地后被 preflight / 中性化 / 回测各读一遍。
//! 目标：只产出第 [t0,t1) 行，写进调用方复用的 (block,n) 小缓冲，不物化大矩阵、不做 NaN 预填。
//!
//! 实现（与沙箱 `sandbox/tail_perf_bench/src/roll8.rs` 同一份、已逐位验证）：
//! 行主序逐行推进 + 每列 f64 累加器 + van Herk/Gil-Werman 前后缀极值法（滑窗 max/min
//! 零数据相关分支）+ 跨块状态 next_row（顺序调用 O(T)、乱序/跳块自动从 0 重放）+ 热路径零分配。
//! 详见下方 `stream` / `rolling_block` 的注释。
//!
//! 硬性要求：与 `tail_v2_rank_roll_factor::rolling_stats_f32_serial` **逐位一致**
//! （含 NaN 位置与 ±0.0 等值语义）。
//! 自测：`rp.tail_v8_selfcheck("roll", data_dir)`。

use ndarray::{s, Array2};

use crate::tail_v2_rank_roll_factor::rolling_stats_f32_serial;

/// 派生面个数 = 1 + 4 × windows.len()（slot 0 = ranked 本身，其余按窗口 4 个：mean/max/min/std）
pub fn slot_count(windows: &[usize]) -> usize {
    1 + 4 * windows.len()
}

/// 每线程一份的可复用缓冲/状态。`new` 之后跨块、跨 slot、跨因子复用。
pub struct RollScratch {
    pub t: usize,
    pub n: usize,
    pub windows: Vec<usize>,
    /// 离窗值环形缓冲：布局 `[slot * n + col]`，`slot = row & ring_mask`。
    pub ring: Vec<f32>,
    pub ring_mask: usize,

    // ---- 流式状态（跨块保留）----
    /// 已折叠进状态的行数：状态 = 处理完 [0, next_row) 之后。
    next_row: usize,
    /// 每窗口每列的窗口内累加器（布局 `[wi * n + col]`）。
    sum: Vec<f64>,
    sumsq: Vec<f64>,
    cnt: Vec<u32>,
    /// 每窗口每列：当前块 [bs, r] 的前缀极值（较新者胜平局），布局 `[wi * n + col]`。
    pre_max: Vec<f32>,
    pre_min: Vec<f32>,
    /// 每窗口的后缀极值：布局 `[wi][j * n + col]`，j ∈ 1..=w（j = w 恒为恒等元：
    /// 窗口恰好等于整块时用，永不被反向扫描写）。j = 0 永不被读。
    suf_off: Vec<usize>,
    suf_max: Vec<f32>,
    suf_min: Vec<f32>,
    /// 每窗口每列：后缀反向扫描的累加器（每次扫描前重置为恒等元）。
    acc_max: Vec<f32>,
    acc_min: Vec<f32>,
}

impl RollScratch {
    pub fn new(t: usize, n: usize, windows: &[usize], _block_rows: usize) -> Self {
        assert!(
            windows.iter().all(|&w| w >= 1),
            "tail_v8_roll::RollScratch 只支持 >= 1 的窗口（w=0 无意义）"
        );
        let max_w = windows.iter().copied().max().unwrap_or(0);
        // ring 需要 > max_w 个槽位：保证「当前行」与「离窗行」落在不同槽位。
        let ring_cap = (max_w + 1).next_power_of_two();
        let mut suf_off = Vec::with_capacity(windows.len());
        let mut suf_len = 0usize;
        for &w in windows {
            suf_off.push(suf_len);
            suf_len += (w + 1) * n; // j = 0..=w
        }
        let wn = windows.len() * n;
        Self {
            t,
            n,
            windows: windows.to_vec(),
            ring: vec![0.0; ring_cap * n],
            ring_mask: ring_cap - 1,
            next_row: 0,
            sum: vec![0.0; wn],
            sumsq: vec![0.0; wn],
            cnt: vec![0; wn],
            pre_max: vec![f32::NEG_INFINITY; wn],
            pre_min: vec![f32::INFINITY; wn],
            suf_off,
            suf_max: vec![f32::NEG_INFINITY; suf_len],
            suf_min: vec![f32::INFINITY; suf_len],
            acc_max: vec![f32::NEG_INFINITY; wn],
            acc_min: vec![f32::INFINITY; wn],
        }
    }

    /// 丢掉全部流式状态（下次调用从第 0 行重放）。
    fn reset(&mut self) {
        for x in self.sum.iter_mut() {
            *x = 0.0;
        }
        for x in self.sumsq.iter_mut() {
            *x = 0.0;
        }
        for x in self.cnt.iter_mut() {
            *x = 0;
        }
        for x in self.pre_max.iter_mut() {
            *x = f32::NEG_INFINITY;
        }
        for x in self.pre_min.iter_mut() {
            *x = f32::INFINITY;
        }
        for x in self.suf_max.iter_mut() {
            *x = f32::NEG_INFINITY;
        }
        for x in self.suf_min.iter_mut() {
            *x = f32::INFINITY;
        }
        self.next_row = 0;
    }

    /// 逐行推进 `[r0, r1)`。`WRITE=false` 时只推进状态（重放用），不碰 `out`。
    fn stream<const WRITE: bool>(
        &mut self,
        data: &[f32],
        windows: &[usize],
        out: &mut [Array2<f32>],
        t0: usize,
        r0: usize,
        r1: usize,
    ) {
        let n = self.n;
        let ring_mask = self.ring_mask;
        let suf_off = &self.suf_off;
        let sum = &mut self.sum;
        let sumsq = &mut self.sumsq;
        let cnt = &mut self.cnt;
        let pre_max = &mut self.pre_max;
        let pre_min = &mut self.pre_min;
        let suf_max = &mut self.suf_max;
        let suf_min = &mut self.suf_min;
        let acc_max = &mut self.acc_max;
        let acc_min = &mut self.acc_min;
        let ring = &mut self.ring;

        for r in r0..r1 {
            let row = &data[r * n..r * n + n];
            let ring_cur = (r & ring_mask) * n;

            for (wi, &w) in windows.iter().enumerate() {
                let min_periods = std::cmp::max(1, w / 2);
                let sbase = wi * n;
                let sb = suf_off[wi];
                // 绝对块对齐：块 = [bs, bs+w)，块边界是 w 的倍数（与调用方块大小无关）。
                let bs = (r / w) * w;
                let i = r - bs;
                let first = i == 0;
                let leave = if r >= w {
                    Some(((r - w) & ring_mask) * n)
                } else {
                    None
                };

                // 块起点：先把上一块 [bs-w, bs) 的后缀极值算好（O(w·n)，摊销每元素 1 次）。
                // 只算 j ∈ 1..w：j = 0 与 j = w 的槽位永远不被读；w = 1 时整块即窗口，无需后缀。
                if first && r > 0 && w > 1 {
                    let prev0 = (bs - w) * n;
                    for c in 0..n {
                        acc_max[sbase + c] = f32::NEG_INFINITY;
                        acc_min[sbase + c] = f32::INFINITY;
                    }
                    for j in (1..w).rev() {
                        let src = &data[prev0 + j * n..prev0 + j * n + n];
                        let dst = sb + j * n;
                        for c in 0..n {
                            let v = src[c];
                            let v_nan = v.is_nan();
                            let vm = if v_nan { f32::NEG_INFINITY } else { v };
                            let vn = if v_nan { f32::INFINITY } else { v };
                            // acc 是更靠后的行（更新），平局留给 acc
                            let am = acc_max[sbase + c];
                            let nm = if vm > am { vm } else { am };
                            acc_max[sbase + c] = nm;
                            suf_max[dst + c] = nm;
                            let an = acc_min[sbase + c];
                            let nn = if vn < an { vn } else { an };
                            acc_min[sbase + c] = nn;
                            suf_min[dst + c] = nn;
                        }
                    }
                }

                let (pm_p, px_p, pn_p, ps_p) = if WRITE {
                    // 行内偏移：out[si] 的第 (r-t0) 行首元素。
                    let b = (r - t0) * n;
                    unsafe {
                        (
                            out[1 + 4 * wi].as_mut_ptr().add(b),
                            out[2 + 4 * wi].as_mut_ptr().add(b),
                            out[3 + 4 * wi].as_mut_ptr().add(b),
                            out[4 + 4 * wi].as_mut_ptr().add(b),
                        )
                    }
                } else {
                    (
                        std::ptr::null_mut(),
                        std::ptr::null_mut(),
                        std::ptr::null_mut(),
                        std::ptr::null_mut(),
                    )
                };

                // 窗口 [r-w+1, r] 的「上一块后缀部分」在上一块里的起点下标 = i+1。
                let jn = sb + (i + 1) * n;

                for c in 0..n {
                    let v = row[c];
                    let si = sbase + c;
                    let v_nan = v.is_nan();

                    // NaN 不进累加器：加 0.0 / 乘 0.0 与「跳过」逐位等价（s、sq 从 +0.0 出发，
                    // 不会变成 -0.0），因此这里用选择而不是分支。
                    let v64 = if v_nan { 0.0 } else { v as f64 };
                    let mut s = sum[si];
                    let mut sq = sumsq[si];
                    let mut k = cnt[si];
                    s += v64;
                    sq += v64 * v64;
                    k += !v_nan as u32;

                    // 极值：NaN → 恒等元（max -inf / min +inf），等价于「NaN 不入队」。
                    let vmax = if v_nan { f32::NEG_INFINITY } else { v };
                    let vmin = if v_nan { f32::INFINITY } else { v };

                    // 块内前缀（块起点用恒等元；较新的 v 胜平局，对应 deque 的 `tail <= v` 弹尾）
                    let p0 = if first { f32::NEG_INFINITY } else { pre_max[si] };
                    let pm = if p0 > vmax { p0 } else { vmax };
                    pre_max[si] = pm;
                    let q0 = if first { f32::INFINITY } else { pre_min[si] };
                    let pn = if q0 < vmin { q0 } else { vmin };
                    pre_min[si] = pn;

                    // 上一块后缀（更旧）与当前块前缀（更新）合并：平局留给前缀（较新）。
                    let sm = suf_max[jn + c];
                    let mx = if sm > pm { sm } else { pm };
                    let sn = suf_min[jn + c];
                    let mn = if sn < pn { sn } else { pn };

                    if let Some(lb) = leave {
                        let lv = ring[lb + c];
                        let l_nan = lv.is_nan();
                        let l64 = if l_nan { 0.0 } else { lv as f64 };
                        s -= l64;
                        sq -= l64 * l64;
                        k -= !l_nan as u32;
                    }

                    if WRITE {
                        if k as usize >= min_periods {
                            let mean = s / k as f64;
                            let sd = if k > 1 {
                                (((sq - (s * s) / k as f64) / (k as f64 - 1.0)).max(0.0)).sqrt()
                                    as f32
                            } else {
                                f32::NAN
                            };
                            unsafe {
                                *pm_p.add(c) = mean as f32;
                                *px_p.add(c) = mx;
                                *pn_p.add(c) = mn;
                                *ps_p.add(c) = sd;
                            }
                        } else {
                            unsafe {
                                *pm_p.add(c) = f32::NAN;
                                *px_p.add(c) = f32::NAN;
                                *pn_p.add(c) = f32::NAN;
                                *ps_p.add(c) = f32::NAN;
                            }
                        }
                    }

                    sum[si] = s;
                    sumsq[si] = sq;
                    cnt[si] = k;
                }
            }

            // 本行写进 ring（供后续行的「离窗值」读取；本行内没有窗口会读它，w >= 1）。
            if !windows.is_empty() {
                ring[ring_cur..ring_cur + n].copy_from_slice(row);
            }
        }
    }
}

/// 计算 `ranked` 第 [t0,t1) 行、全部 `slot_count(windows)` 个派生面。
///
/// 契约：
/// - `out.len() == slot_count(windows)`；
/// - `out[i].ncols() == n`，且 `out[i].nrows() >= t1-t0`；本函数只写第 `0..(t1-t0)` 行；
/// - 结果与 `engine::rolling_stats_f32_serial(ranked, w, max(1,w/2))` 对应窗口的输出逐位一致；
/// - `ranked` 与所有 `out[i]` 需为 C 序连续内存（否则直接报错，不做静默降级）。
///
/// 顺序按块调用（t0 == 上一块的 t1）时状态跨调用保留，整体 O(T)；跳块 / 乱序则自动重放。
pub fn rolling_block(
    ranked: &Array2<f32>,
    windows: &[usize],
    t0: usize,
    t1: usize,
    scratch: &mut RollScratch,
    out: &mut [Array2<f32>],
) {
    let (t, n) = ranked.dim();
    let rows = t1 - t0;
    assert!(
        t0 <= t1 && t1 <= t && t1 <= u32::MAX as usize,
        "rolling_block 行区间非法: t0={t0} t1={t1} t={t}"
    );
    assert_eq!(out.len(), slot_count(windows), "out 槽位数与 windows 不匹配");
    for (i, a) in out.iter().enumerate() {
        assert_eq!(a.ncols(), n, "out[{i}] 列数与 ranked 不匹配");
        assert!(a.nrows() >= rows, "out[{i}] 行数不足");
        assert!(
            a.is_standard_layout(),
            "rolling_block 要求 out[{i}] 为 C 序连续内存"
        );
    }
    let data = ranked
        .as_slice()
        .expect("rolling_block 要求 ranked 为 C 序连续内存");

    // slot 0 = ranked 本身
    if rows > 0 {
        out[0]
            .slice_mut(s![0..rows, ..])
            .assign(&ranked.slice(s![t0..t1, ..]));
    }
    if windows.is_empty() {
        return;
    }

    // 换因子 / 换形状 → 重建状态；顺序调用 → 续算；否则从 0 重放。
    if scratch.n != n || scratch.windows != windows {
        *scratch = RollScratch::new(t, n, windows, 0);
    }
    scratch.t = t;
    let start = if scratch.next_row == t0 {
        t0
    } else {
        scratch.reset();
        0
    };

    if start < t1 {
        // 重放段（仅乱序/跳块时非空）：只推进状态，不写输出。
        if start < t0 {
            scratch.stream::<false>(data, windows, out, t0, start, t0.min(t1));
        }
        scratch.stream::<true>(data, windows, out, t0, start.max(t0), t1);
    }
    scratch.next_row = t1;
}

// ---------------- 自测：rp.tail_v8_selfcheck("roll", data_dir) ----------------

/// 参考实现：13 张 (T,N) 全量面（slot 0 = ranked 本身；随后每窗口 mean/max/min/std）。
fn build_want(ranked: &Array2<f32>, windows: &[usize]) -> Vec<Array2<f32>> {
    let mut want = vec![ranked.clone()];
    for &w in windows {
        let (mean, max, min, std) = rolling_stats_f32_serial(ranked, w, std::cmp::max(1, w / 2));
        want.push(mean);
        want.push(max);
        want.push(min);
        want.push(std);
    }
    want
}

/// out 缓冲的未写行哨兵值（验证「只写第 0..rows 行」）。
const SENTINEL: f32 = 12345.678;

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

/// 按块调用 rolling_block 并与参考逐位比较；同时验证尾部哨兵未被写。
/// 返回 (不一致格数, 尾部被写格数)。
fn check_blocks(
    ranked: &Array2<f32>,
    windows: &[usize],
    bs: usize,
    want: &[Array2<f32>],
    sc: &mut RollScratch,
) -> (usize, usize) {
    let (t, n) = ranked.dim();
    let ns = slot_count(windows);
    let mut buf: Vec<Array2<f32>> =
        (0..ns).map(|_| Array2::<f32>::from_elem((bs, n), SENTINEL)).collect();
    let mut bad = 0usize;
    let mut tail_bad = 0usize;
    let mut t0 = 0usize;
    while t0 < t {
        let t1 = (t0 + bs).min(t);
        let rows = t1 - t0;
        if rows < bs {
            for b in buf.iter_mut() {
                b.slice_mut(s![rows..bs, ..]).fill(SENTINEL);
            }
        }
        rolling_block(ranked, windows, t0, t1, sc, &mut buf);
        for i in 0..ns {
            bad += cmp_rows(&buf[i], &want[i], t0, rows);
        }
        if rows < bs {
            for b in buf.iter() {
                let sl = b.as_slice().unwrap();
                for &v in &sl[rows * n..] {
                    if v.to_bits() != SENTINEL.to_bits() {
                        tail_bad += 1;
                    }
                }
            }
        }
        t0 = t1;
    }
    (bad, tail_bad)
}

/// 确定性小 PRNG（自测造数据用）。
fn lcg(seed: &mut u64) -> u64 {
    *seed ^= *seed << 13;
    *seed ^= *seed >> 7;
    *seed ^= *seed << 17;
    *seed
}

/// 自测：对 data_dir 下的因子，分块拼接后与生产 `rolling_stats_f32_serial` 逐位比较，
/// 并给出「新实现 / 占位实现（每块重算全矩阵）/ 生产全量」的单线程计时。
pub fn selfcheck(data_dir: &str) -> String {
    use ndarray_npy::read_npy;
    use std::time::Instant;

    let mut lines: Vec<String> = Vec::new();
    let mut all_pass = true;
    let names: Vec<String> = std::fs::read_to_string(format!("{data_dir}/sample_names.txt"))
        .unwrap_or_default()
        .split_whitespace()
        .map(|s| s.to_string())
        .collect();
    let windows = vec![5usize, 10, 20];
    let blocks = [64usize, 997, 2818];
    let ns = slot_count(&windows);

    // ---- 1) 逐位对账：4 因子 × 3 窗口 × 块 64/997/2818 ----
    let mut first: Option<(String, Array2<f32>)> = None;
    let mut scs: Vec<Option<RollScratch>> = (0..blocks.len()).map(|_| None).collect();
    for nm in names.iter().take(4) {
        let ranked: Array2<f32> = match read_npy(format!("{data_dir}/factor_{nm}.npy")) {
            Ok(m) => m,
            Err(e) => {
                lines.push(format!("  [对账] {nm}: 读取失败 {e}"));
                all_pass = false;
                continue;
            }
        };
        let (t, n) = ranked.dim();
        let want = build_want(&ranked, &windows);
        for (bi, &bs) in blocks.iter().enumerate() {
            // scratch 跨因子复用（与 v8 流水线一致）：换因子时必须自动重放，结果仍逐位一致。
            let sc = scs[bi].get_or_insert_with(|| RollScratch::new(t, n, &windows, bs));
            let (bad, tail) = check_blocks(&ranked, &windows, bs, &want, sc);
            if bad > 0 || tail > 0 {
                all_pass = false;
            }
            lines.push(format!(
                "  [对账] {nm} block={bs}: 不一致 {bad} 格，尾部被写 {tail} 格"
            ));
        }
        if first.is_none() {
            first = Some((nm.clone(), ranked));
        }
    }

    // ---- 2) 附加用例：乱序/跳块、换窗口集、NaN 加密、等值±0.0、单调退化 ----
    if let Some((nm, ranked)) = first.as_ref() {
        let (t, n) = ranked.dim();
        let want = build_want(ranked, &windows);
        let mut sc = RollScratch::new(t, n, &windows, 64);
        let mut buf: Vec<Array2<f32>> =
            (0..ns).map(|_| Array2::<f32>::from_elem((t, n), SENTINEL)).collect();
        let half = t / 2;
        let mut bad = 0usize;
        rolling_block(ranked, &windows, half, t, &mut sc, &mut buf);
        for i in 0..ns {
            bad += cmp_rows(&buf[i], &want[i], half, t - half);
        }
        rolling_block(ranked, &windows, 0, half, &mut sc, &mut buf);
        for i in 0..ns {
            bad += cmp_rows(&buf[i], &want[i], 0, half);
        }
        if bad > 0 {
            all_pass = false;
        }
        lines.push(format!("  [对账] 乱序 [t/2,t)→[0,t/2)（{nm}）: 不一致 {bad} 格"));
    }
    {
        // 等值 / ±0.0 密集：前后缀极值法必须复刻 deque 的 `tail <= v` 弹尾语义
        // （值相等、含 +0.0/-0.0 时保留较新的那次出现）。
        let (tt, nn) = (600usize, 256usize);
        let mut tie = Array2::<f32>::zeros((tt, nn));
        let mut seed = 0x1234_5678_9abc_def0u64;
        let pool = [0.0f32, -0.0, 1.0, -1.0, 2.0, f32::NAN, 0.0, 1.0, -0.0];
        for v in tie.iter_mut() {
            *v = pool[(lcg(&mut seed) % pool.len() as u64) as usize];
        }
        let n_pos = tie.iter().filter(|v| v.to_bits() == 0.0f32.to_bits()).count();
        let n_neg = tie.iter().filter(|v| v.to_bits() == (-0.0f32).to_bits()).count();
        let want = build_want(&tie, &windows);
        let mut sc = RollScratch::new(tt, nn, &windows, 64);
        let (bad, tail) = check_blocks(&tie, &windows, 64, &want, &mut sc);
        let (bad2, _) = check_blocks(&tie, &windows, 997, &want, &mut sc);
        let windows3 = vec![1usize, 2, 3, 7];
        let want3 = build_want(&tie, &windows3);
        let (bad3, _) = check_blocks(&tie, &windows3, 64, &want3, &mut sc);
        if bad + bad2 + bad3 > 0 || tail > 0 {
            all_pass = false;
        }
        lines.push(format!(
            "  [对账] 等值±0.0 密集（+0.0 {n_pos} 格 / -0.0 {n_neg} 格）block=64/997 + w=[1,2,3,7]: 不一致 {} 格，尾部被写 {tail} 格",
            bad + bad2 + bad3
        ));
    }
    {
        // NaN 加密 + 单调/常量退化形状
        let (tt, nn) = (900usize, 512usize);
        let mut dense = Array2::<f32>::zeros((tt, nn));
        let mut seed = 0x9e37_79b9_7f4a_7c15u64;
        for (i, v) in dense.iter_mut().enumerate() {
            *v = if i % 3 == 0 {
                f32::NAN
            } else if i % 5 == 0 {
                0.0
            } else {
                (lcg(&mut seed) % 1000) as f32 / 100.0
            };
        }
        let want = build_want(&dense, &windows);
        let mut sc = RollScratch::new(tt, nn, &windows, 64);
        let (bad, _) = check_blocks(&dense, &windows, 64, &want, &mut sc);
        let (bad2, _) = check_blocks(&dense, &windows, 2818, &want, &mut sc);
        if bad + bad2 > 0 {
            all_pass = false;
        }
        lines.push(format!("  [对账] NaN 加密 block=64/2818: 不一致 {} 格", bad + bad2));

        let (mt, mn) = (300usize, 128usize);
        let mut mono = Array2::<f32>::zeros((mt, mn));
        for r in 0..mt {
            for c in 0..mn {
                mono[[r, c]] = match c % 4 {
                    0 => r as f32,
                    1 => (mt - r) as f32,
                    2 => 5.0,
                    _ => {
                        if r % 3 == 0 {
                            f32::NAN
                        } else {
                            5.0
                        }
                    }
                };
            }
        }
        let want = build_want(&mono, &windows);
        let mut sc = RollScratch::new(mt, mn, &windows, 64);
        let (bad, _) = check_blocks(&mono, &windows, 64, &want, &mut sc);
        if bad > 0 {
            all_pass = false;
        }
        lines.push(format!(
            "  [对账] 单调/常量/常量+NaN: 不一致 {bad} 格"
        ));
    }

    // ---- 3) 单线程计时（第一个因子）----
    if let Some((nm, ranked)) = first.as_ref() {
        let (t, n) = ranked.dim();
        // 生产全量：3 个窗口各 4 张 (T,N)（= 占位实现「一个块」的成本）
        let mut prod = f64::INFINITY;
        for _ in 0..2 {
            let tp = Instant::now();
            for &w in &windows {
                let out = rolling_stats_f32_serial(ranked, w, std::cmp::max(1, w / 2));
                std::hint::black_box(&out);
            }
            prod = prod.min(tp.elapsed().as_secs_f64());
        }
        // 新实现：block=64（v8 生产块大小）
        let mut new_s = f64::INFINITY;
        for _ in 0..2 {
            let mut sc = RollScratch::new(t, n, &windows, 64);
            let mut buf: Vec<Array2<f32>> = (0..ns)
                .map(|_| Array2::<f32>::from_elem((64, n), 0.0))
                .collect();
            let tn = Instant::now();
            let mut t0 = 0usize;
            while t0 < t {
                let t1 = (t0 + 64).min(t);
                rolling_block(ranked, &windows, t0, t1, &mut sc, &mut buf);
                t0 = t1;
            }
            new_s = new_s.min(tn.elapsed().as_secs_f64());
            std::hint::black_box(&buf);
        }
        // 占位实现（每块重算 12 张完整矩阵再截取）：block=2818（1 块）与 997（3 块）直接测；
        // block=64（45 块）默认按每块成本外推，设 V8_ROLL_FULL_TIME=1 时直接实测。
        let ph = |bs: usize| -> f64 {
            let mut buf: Vec<Array2<f32>> = (0..ns)
                .map(|_| Array2::<f32>::from_elem((bs, n), 0.0))
                .collect();
            let tp = Instant::now();
            let mut t0 = 0usize;
            while t0 < t {
                let t1 = (t0 + bs).min(t);
                let rows = t1 - t0;
                buf[0]
                    .slice_mut(s![0..rows, ..])
                    .assign(&ranked.slice(s![t0..t1, ..]));
                let mut si = 1usize;
                for &w in &windows {
                    let (mean, max, min, std) =
                        rolling_stats_f32_serial(ranked, w, std::cmp::max(1, w / 2));
                    for mat in [mean, max, min, std] {
                        buf[si]
                            .slice_mut(s![0..rows, ..])
                            .assign(&mat.slice(s![t0..t1, ..]));
                        si += 1;
                    }
                }
                t0 = t1;
            }
            let el = tp.elapsed().as_secs_f64();
            std::hint::black_box(&buf);
            el
        };
        let ph_2818 = ph(2818);
        let ph_997 = ph(997);
        let n_blocks_64 = t.div_ceil(64);
        lines.push(format!(
            "  [计时] {nm} T={t} N={n} windows={windows:?}"
        ));
        lines.push(format!("  [计时] 生产全量（3 窗口 serial，12 张 (T,N)）: {prod:.3}s"));
        lines.push(format!(
            "  [计时] 占位实现 block=2818（1 块）: {ph_2818:.3}s ／ block=997（3 块）: {ph_997:.3}s（每块 ~{:.3}s）",
            ph_997 / 3.0
        ));
        lines.push(format!(
            "  [计时] 新实现 block=64: {new_s:.3}s  → vs 生产全量 {:.2}×",
            prod / new_s
        ));
        if std::env::var("V8_ROLL_FULL_TIME").is_ok() {
            let ph_64 = ph(64);
            lines.push(format!(
                "  [计时] 占位实现 block=64（{n_blocks_64} 块，直接实测）: {ph_64:.3}s  → 新实现快 {:.0}×",
                ph_64 / new_s
            ));
        } else {
            lines.push(format!(
                "  [计时] 占位实现 block=64 外推: {n_blocks_64} 块 × {:.3}s ≈ {:.1}s  → 新实现快 ~{:.0}×（设 V8_ROLL_FULL_TIME=1 可实测）",
                ph_2818,
                n_blocks_64 as f64 * ph_2818,
                n_blocks_64 as f64 * ph_2818 / new_s
            ));
        }
    }

    lines.insert(
        0,
        format!(
            "[tail_v8_roll::selfcheck] 逐位一致: {}",
            if all_pass { "PASS" } else { "FAIL" }
        ),
    );
    lines.join("\n")
}
