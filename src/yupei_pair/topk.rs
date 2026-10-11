//! 玉佩2 配对互补网络 —— 子命令 topk：每对 (B,C) 取前 K 只 X，双口径多统计。
//!
//! 为什么加这一档：net 每对只留一只 X0（h 最大的那只），所以「按 X0 聚合」的因子
//! （reduce 里的 r1_explain_* 系列）只有被选中过的那部分股票有值，20241231 是
//! 922/3684 = 25%。这里把同一个 h 放宽成前 K 名，并新增 gap-top 口径：
//!
//!   * h-top（h_ 前缀）：按 h_X = rho(X, U_BC) 取前 K——「合体后最像谁」口径；
//!   * gap-top（g_ 前缀）：按 gap_X = h_X - max(rho[X][B], rho[X][C]) 取前 K——
//!     「合体能补、单只补不了的缺口」口径，与 net 的 X0* 是同一目标的前 K 版。
//!
//! 每档每口径四个统计：
//!   * count      入选次数（u32，确定性整数）；
//!   * gain_mean  入选时的平均缺口 gap（h-top 档与旧版 x0topk{K}_gain_mean 同一量）；
//!   * h_mean     入选时的平均 h（gap 口径下也记 h，便于两口径对照）；
//!   * mass_share Σ relu(本口径分数)/该对本口径前 K 的 relu 总和——连续的"在场强度"，
//!                不依赖硬入选，是覆盖率的软化版。
//!
//! K 默认九档 {1,2,5,10,20,50,100,200,500}；K=1 的 h-top 档与 net 的 X0+G 同量，
//! 留作对账锚点。h 与权重一字不改地照 net 的 best_x 写：
//!   h_X = wb * rho[X][B] + wc * rho[X][C]
//!   wb = sqrt(eB)/D, wc = sqrt(eC)/D, D = sqrt(eB + eC + 2*rho[B][C]*sqrt(eB*eC))
//! 候选集与 net 一样：X 不能是 B 或 C，且必须在股票白名单里。
//!
//! 内存版：`topk_one_mat` 直接吃 `net_one_mat` 的 rho/e，输出
//! (names, cols)——每列 n 长，非活跃股票位置 NaN。计算逻辑与
//! sandbox_yupei_pair/src/topk.rs 逐位一致。
//!
//! ## 确定性怎么保证的
//!
//! B 的下标范围按固定块数 NBLOCKS 切开（切法只依赖 n 与 NBLOCKS，与线程数无关），
//! 每块在一块自己的累加器上求和，全部并行跑完之后按下标从小到大依次合并。于是
//! 换线程数（16/128/512）得到的结果逐位相同，同一输入跑两遍也逐位相同。

use rayon::prelude::*;
use std::time::Instant;

/// B 分成多少块。固定值，不随线程数变，否则换线程数结果会变。
const NBLOCKS: usize = 512;
/// 每个累加器里每档每口径 4 个量：cnt(u32), sum_gap, sum_h, sum_mass。
struct Acc {
    // [口径2][档nlev][统计4? 分开存] —— 分四个平面存，索引 = (crit*nlev + l)*n + y
    cnt: Vec<u32>,
    sum_gap: Vec<f64>,
    sum_h: Vec<f64>,
    sum_mass: Vec<f64>,
    pairs: u64,
    skipped: u64,
}

impl Acc {
    fn new(n: usize, nlev: usize) -> Self {
        let l = 2 * nlev * n;
        Acc {
            cnt: vec![0u32; l],
            sum_gap: vec![0.0f64; l],
            sum_h: vec![0.0f64; l],
            sum_mass: vec![0.0f64; l],
            pairs: 0,
            skipped: 0,
        }
    }
    fn add(&mut self, o: &Acc) {
        for i in 0..self.cnt.len() {
            self.cnt[i] += o.cnt[i];
            self.sum_gap[i] += o.sum_gap[i];
            self.sum_h[i] += o.sum_h[i];
            self.sum_mass[i] += o.sum_mass[i];
        }
        self.pairs += o.pairs;
        self.skipped += o.skipped;
    }
}

// 排序规则：分数降序、同分下标升序——已由 pack_desc_idx_asc 的 u64
// 打包键整型序实现（与原 h_less 比较器逐位等价）。
/// 把一个口径的前 take 名累加进 acc 的对应平面。
/// score 是该口径的分数数组（h 或 gap），order 是按该分数排好的前 take 下标，
/// relu_pref[v] = Σ_{u<=v} relu(score[order[u]])，供各档归一化用。
#[allow(clippy::too_many_arguments)]
fn absorb_topk(
    acc: &mut Acc,
    n: usize,
    crit: usize,
    nlev: usize,
    ks: &[usize],
    order: &[u32],
    h: &[f32],
    gap: &[f32],
    relu_pref: &[f64],
) {
    for (l, &k) in ks.iter().enumerate() {
        let kk = k.min(order.len());
        if kk == 0 {
            continue;
        }
        let denom = relu_pref[kk - 1].max(1e-30);
        let base = (crit * nlev + l) * n;
        for v in 0..kk {
            let y = order[v] as usize;
            acc.cnt[base + y] += 1;
            acc.sum_gap[base + y] += gap[y] as f64;
            acc.sum_h[base + y] += h[y] as f64;
            acc.sum_mass[base + y] += (order_score(order[v], h, gap, crit)).max(0.0) as f64 / denom;
        }
    }
}

#[inline]
fn order_score(y: u32, h: &[f32], gap: &[f32], crit: usize) -> f32 {
    if crit == 0 {
        h[y as usize]
    } else {
        gap[y as usize]
    }
}

/// 扫 [b0, b1) 这一段 B，h-top 与 gap-top 两口径各档同时累加。
/// addm 是预建的 -inf 掩码（非个股位置）；n_valid = 个股数-2，对所有配对相同。
/// 与旧版差异仅工程层：h/gap 用 SIMD 物化、top-K 用 u64 打包键整型选择，
/// 排序序与 h_less 完全一致（分数降序、同分下标升序 → pack_desc_idx_asc 升序）。
fn block_accumulate(
    n: usize,
    rho: &[f32],
    e: &[f64],
    mask: &[bool],
    addm: &[f32],
    n_valid: usize,
    b0: usize,
    b1: usize,
    ks: &[usize],
    maxk: usize,
) -> Acc {
    let nlev = ks.len();
    let mut acc = Acc::new(n, nlev);
    // 本块自己的缓冲，热路径上不再分配
    let mut h = vec![0.0f32; n];
    let mut gap = vec![0.0f32; n];
    let mut kh = vec![0u64; n];
    let mut kg = vec![0u64; n];
    let mut idx = vec![0u32; maxk];
    let mut gidx = vec![0u32; maxk];
    let mut pref = vec![0.0f64; maxk];
    let take = n_valid.min(maxk);

    for b in b0..b1 {
        if !mask[b] {
            continue;
        }
        let p = &rho[b * n..b * n + n]; // rho[x][b]（rho 对称，取行与取列逐位相同）
        let eb = e[b];
        let sb = eb.sqrt();
        for c in (b + 1)..n {
            if !mask[c] {
                continue;
            }
            let ec = e[c];
            let sc = ec.sqrt();
            let r = rho[b * n + c] as f64; // rho[b][c]
            let d = (eb + ec + 2.0 * r * sb * sc).sqrt();
            if !(d > 0.0) {
                acc.skipped += 1;
                continue;
            }
            let wb = (sb / d) as f32;
            let wc = (sc / d) as f32;
            let q = &rho[c * n..c * n + n];

            // 物化 h 与 gap（SIMD）；B/C 与非个股位 -inf
            super::simd::fill_h(&mut h, wb, wc, p, q, addm);
            super::simd::fill_gap(&mut gap, &h, p, q);
            h[b] = f32::NEG_INFINITY;
            h[c] = f32::NEG_INFINITY;
            gap[b] = f32::NEG_INFINITY;
            gap[c] = f32::NEG_INFINITY;
            if take == 0 {
                continue;
            }
            // 打包 (score,idx) u64 键：升序整型序 = 分数降序+下标升序
            for x in 0..n {
                kh[x] = super::simd::pack_desc_idx_asc(h[x], x as u32);
                kg[x] = super::simd::pack_desc_idx_asc(gap[x], x as u32);
            }
            kh.select_nth_unstable(take - 1);
            kh[..take].sort_unstable();
            kg.select_nth_unstable(take - 1);
            kg[..take].sort_unstable();
            for v in 0..take {
                idx[v] = kh[v] as u32;
                gidx[v] = kg[v] as u32;
            }

            // h 口径的 relu 前缀和 + 累加
            let mut s = 0.0f64;
            for v in 0..take {
                let y = idx[v] as usize;
                s += h[y].max(0.0) as f64;
                pref[v] = s;
            }
            absorb_topk(&mut acc, n, 0, nlev, ks, &idx[..take], &h, &gap, &pref);
            // gap 口径的 relu 前缀和 + 累加
            s = 0.0;
            for v in 0..take {
                let y = gidx[v] as usize;
                s += gap[y].max(0.0) as f64;
                pref[v] = s;
            }
            absorb_topk(&mut acc, n, 1, nlev, ks, &gidx[..take], &h, &gap, &pref);
            acc.pairs += 1;
        }
    }
    acc
}

/// 单矩阵 topk 因子块：返回 (names, cols)，cols[j][i] = 股票 i 在第 j 个
/// K 档×口径×统计下的值，非活跃股票（mask=false）写 NaN。
/// 与 sandbox cmd_topk 的落盘口径一致：每口径四个量
/// (count, gain_mean, h_mean, mass_share)，顺序 = 口径 × K 档 × 4 统计。
/// names 顺序与 cols 一一对应，前缀 {pre}x0topk{K}_ / {pre}gtopk{K}_。
pub(crate) fn topk_one_mat(
    rho: &[f32],
    e: &[f64],
    n: usize,
    mask: &[bool],
    ks: &[usize],
    pre: &str,
) -> (Vec<String>, Vec<Vec<f32>>) {
    let t0 = Instant::now();
    let nlev = ks.len();
    let maxk = *ks.iter().max().unwrap();
    let n_stocks = mask.iter().filter(|&&v| v).count();
    let n_valid = n_stocks.saturating_sub(2);
    // -inf 掩码：非活跃位置
    let addm: Vec<f32> = (0..n)
        .map(|i| if mask[i] { 0.0 } else { f32::NEG_INFINITY })
        .collect();
    let nb = NBLOCKS.min(n).max(1);
    let bounds: Vec<(usize, usize)> =
        (0..nb).map(|k| (k * n / nb, (k + 1) * n / nb)).collect();
    let accs: Vec<Acc> = bounds
        .par_iter()
        .map(|&(b0, b1)| {
            block_accumulate(n, rho, e, mask, &addm, n_valid, b0, b1, ks, maxk)
        })
        .collect();
    let mut total = Acc::new(n, nlev);
    for a in accs.iter() {
        total.add(a);
    }
    let pairs = total.pairs;

    // names + cols：每口径四量，顺序 (crit × l × 4)
    let mut names = Vec::with_capacity(nlev * 4 * 2);
    for crit in 0..2 {
        let pf = if crit == 0 { "x0topk" } else { "gtopk" };
        for &k in ks.iter() {
            names.push(format!("{pre}{pf}{k}_count"));
            names.push(format!("{pre}{pf}{k}_gain_mean"));
            names.push(format!("{pre}{pf}{k}_h_mean"));
            names.push(format!("{pre}{pf}{k}_mass_share"));
        }
    }
    let mut cols: Vec<Vec<f32>> = Vec::with_capacity(nlev * 4 * 2);
    for crit in 0..2 {
        for l in 0..nlev {
            let base = (crit * nlev + l) * n;
            let (mut c_cnt, mut c_gain, mut c_h, mut c_ms) = (
                vec![f32::NAN; n], vec![f32::NAN; n], vec![f32::NAN; n], vec![f32::NAN; n]);
            for i in 0..n {
                if !mask[i] {
                    continue;
                }
                let c = total.cnt[base + i];
                c_cnt[i] = c as f32;
                c_gain[i] = if c > 0 { (total.sum_gap[base + i] / c as f64) as f32 } else { f32::NAN };
                c_h[i] = if c > 0 { (total.sum_h[base + i] / c as f64) as f32 } else { f32::NAN };
                c_ms[i] = (total.sum_mass[base + i] / pairs.max(1) as f64) as f32;
            }
            cols.push(c_cnt);
            cols.push(c_gain);
            cols.push(c_h);
            cols.push(c_ms);
        }
    }
    eprintln!(
        "[topk] {nlev} 档 × 2 口径 {:.1}s，pairs={pairs}",
        t0.elapsed().as_secs_f64()
    );
    (names, cols)
}
