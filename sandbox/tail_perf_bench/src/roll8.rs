//! roll8：按日期块流式产出 13 个派生面（一档 + 二档的「生产者」侧）。
//!
//! 现状（生产 v7）：每个窗口先算完整 4 张 (T,N) 矩阵（各自先 NaN 预填一遍），
//! 12 张矩阵落地后再被 preflight / 中性化 / 回测各读一遍。
//! 目标：只产出第 [t0,t1) 行，写进调用方复用的小缓冲（block × N），**不再物化大矩阵**，
//! 也不再有 NaN 预填那趟白写。
//!
//! 实现（行主序流式 + 前后缀极值法）：
//! - 逐行推进（(T,N) 为 C 序，逐行即顺序访存），每列维护自己的 f64 累加器；
//!   每个窗口的「加入当前值 → 减去离窗值」序列与 `engine::rolling_stats_f32_serial`
//!   逐行完全同序，所以 mean/std（含 NaN 位置）逐位一致。
//! - max/min 用 van Herk / Gil-Werman 前后缀极值法：把行按窗口长度 w 切成**绝对对齐**的块
//!   （块边界 = w 的倍数，与调用方块大小无关），窗口 [r-w+1, r] 必然跨
//!   「上一块的尾部 + 当前块的头部」，于是
//!     window_max = max(上一块从 L 起的后缀极值, 当前块 [bs, r] 的前缀极值)
//!   每元素固定 2~3 次比较、**零数据相关分支**（旧实现的单调队列靠 `tail <= v` / `tail >= v`
//!   弹尾，在 47% NaN 的数据上分支抖动大，实测吃掉 ~70% 时间）。
//! - 与 deque 语义严格对齐（逐位一致的关键）：
//!   * serial 的弹尾条件是 `tail_val <= value`（min 是 `>=`），即**值相等（含 ±0.0）时保留较新的那次出现**；
//!     这里所有比较都让「较新的一侧」胜平局（块内前缀较新、上一块后缀较旧，前缀部分永远比后缀部分新）。
//!   * NaN 只在非 NaN 时入队：这里把 NaN 映射成恒等元（max→-inf、min→+inf），
//!     对极值结果等价（窗口内无有效值时输出仍是 NaN，由 count < min_periods 兜住）。
//! - 后缀极值在块起点用 `ranked` 的上一块行重算（O(w·n)，摊销到每元素 1 次），
//!   因此极值状态只有「当前块前缀」需要跨块保留；`next_row` 仍保留顺序状态：
//!   顺序按块调用总工作量 O(T)，跳块 / 乱序 / 换因子时自动从第 0 行重放（结果仍逐位一致）。
//! - 热路径零分配：所有状态都在 `RollScratch` 里预分配，`rolling_block` 只写调用方的 `out`。
//!
//! 硬性要求：与 `engine::rolling_stats_f32_serial` **逐位一致**（NaN 位置也要一致）。
//! 自测入口：`src/bin/roll8_check.rs`（3 因子 × 3 窗口 × 块 64/997/2818 逐位对账 + 单线程计时）。

use ndarray::{s, Array2};

use crate::t8::slot_count;

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
            "roll8::RollScratch 只支持 >= 1 的窗口（w=0 无意义）"
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
/// - `out.len() == t8::slot_count(windows)`；
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
