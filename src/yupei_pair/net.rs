//! 玉佩2 双层网络（配对互补）—— net 阶段（内存版）。
//!
//! 与 sandbox_yupei_pair 的 `cmd_net` 计算逻辑逐位一致，差别只在边界：
//! 输入从「mats/*.bin 文件」换成 `compute_yupei_dist_partial` 在线产出的
//! S_dir 矩阵切片；输出从「22 个 .bin 落盘」换成 `NetMats` 内存结构。
//! 汇报段（top-G/Z/ZS 配对列表、summary.json）在函数版里没有意义，删除。
//!
//! 口径（与 pair_spec.txt 一致）：
//!   S[i][j] = S_dir[i][j] + S_dir[j][i]（i != j）；E_i = 2 * S_dir[i][i]；
//!   rho[i][j] = S[i][j] / sqrt(E_i E_j)；
//!   rho[X][U_BC] = wB*rho[X][B] + wC*rho[X][C]，
//!   wB = sqrt(eB)/D, wC = sqrt(eC)/D, D = sqrt(eB+eC+2*rho[B][C]*sqrt(eB*eC))。

use rayon::prelude::*;

use super::simd;
use super::topk::{absorb_topk, Acc as TopkAcc};

/// xorshift64*：固定种子，保证同一输入每次跑出同样的抽样。
struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        Rng(if seed == 0 { 0x9E3779B97F4A7C15 } else { seed })
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }
    fn below(&mut self, m: usize) -> usize {
        (self.next_u64() % (m as u64)) as usize
    }
}

/// mass 归一化分母的下限：低于它说明该口径没有有效权质量，直接停用。
const MASS_EPS: f32 = 1e-6;

pub(crate) fn decile_of(v: f64, cuts: &[f64]) -> usize {
    // cuts 是 9 个十分位切点，返回 0..9
    let mut d = 0usize;
    while d < cuts.len() && v > cuts[d] {
        d += 1;
    }
    d
}

/// 方向拆分时的三套取数矩阵（行 = 配对位置，列 = 第三只 X）。
/// 现有单矩阵路径传 None，三者都退化成同一个对称 rho。
#[derive(Clone, Copy)]
pub(crate) struct SplitRho<'a> {
    /// 配对第一只（方向 a）对 X（方向 c）：`b_row[b*n + x]`
    pub b_row: &'a [f32],
    /// 配对第二只（方向 b）对 X（方向 c）：`c_row[c*n + x]`
    pub c_row: &'a [f32],
    /// 配对两只之间（a 对 b）：`bc[b*n + c]`
    pub bc: &'a [f32],
}

/// net 一张输入矩阵的全部产物（内存版）。
/// 字段名与 reduce 的 MatSet / 旧落盘文件名一一对应。
pub(crate) struct NetMats {
    pub e: Vec<f64>,
    pub rho: Vec<f32>,
    pub g: Vec<f32>,
    pub ru: Vec<f32>,
    pub sg: Vec<f32>,
    pub x0: Vec<f32>,
    pub gs: Vec<f32>,
    pub rus: Vec<f32>,
    pub sgs: Vec<f32>,
    pub x0s: Vec<f32>,
    pub z: Vec<f32>,
    pub zs: Vec<f32>,
    pub grel: Vec<f32>,
    pub gsh: Vec<f32>,
    pub gcov: Vec<f32>,
    pub gnpos: Vec<f32>,
    pub ghhi: Vec<f32>,
    pub pb: Vec<f32>,
    pub qc: Vec<f32>,
    pub pbs: Vec<f32>,
    pub qcs: Vec<f32>,
    pub exz: Vec<f32>,
    pub exzs: Vec<f32>,
    /// 6×n：emass|gmass|g2mass|smass|sgmass|sg2mass（按对数归一）
    pub mass: Vec<f32>,
    /// 有效配对数（mask 内且 d>0）
    /// 空模型配对数/分层统计（诊断用，消费端暂未使用）
    #[allow(dead_code)]
    pub npair: u64,
    /// (mean, std, count) × 100 层 ×2 口径
    #[allow(dead_code)]
    pub strata: Vec<(f64, f64, usize)>,
    #[allow(dead_code)]
    pub strata_s: Vec<(f64, f64, usize)>,
    /// 与 topk 合并后，同一趟配对扫描里累计出来的 topk 结果（None = 没启用合并）。
    pub topk_total: Option<TopkAcc>,
}

struct Shared {
    g: *mut f32,
    ru: *mut f32,
    sg: *mut f32,
    x0: *mut f32,
    gs: *mut f32,
    rus: *mut f32,
    sgs: *mut f32,
    x0s: *mut f32,
    grel: *mut f32,
    gsh: *mut f32,
    gcov: *mut f32,
    gnpos: *mut f32,
    ghhi: *mut f32,
    pb: *mut f32,
    qc: *mut f32,
    pbs: *mut f32,
    qcs: *mut f32,
}
unsafe impl Send for Shared {}
unsafe impl Sync for Shared {}

/// 一对 (B,C) 一趟扫描的全部产出（空模型抽样用，与配对扫描同口径）。
#[derive(Clone, Copy)]
struct PairScan {
    bi: usize,
    h: f32,
    single: f32,
    bsi: usize,
    hs: f32,
    singles: f32,
}

/// 一趟扫过所有 X，同时求两个 argmax：
///   (1) X0 = argmax_X rho(X,U_BC)
///   (2) X0* = argmax_X [rho(X,U_BC) - max(rho(X,B),rho(X,C))]
#[inline]
fn best_x(n: usize, p: &[f32], q: &[f32], wb: f32, wc: f32, b: usize, c: usize, mask: &[bool]) -> PairScan {
    let mut s = PairScan {
        bi: usize::MAX,
        h: f32::NEG_INFINITY,
        single: 0.0,
        bsi: usize::MAX,
        hs: 0.0,
        singles: 0.0,
    };
    let (mut best_h_key, mut best_g_key) = (u64::MAX, u64::MAX);
    for x in 0..n {
        if x == b || x == c || !mask[x] {
            continue;
        }
        let h = wb * p[x] + wc * q[x] + 0.0;
        let single = if p[x] > q[x] { p[x] } else { q[x] };
        let gain = h - single;
        if !h.is_finite() || !gain.is_finite() { continue; }
        let h_key = simd::pack_desc_idx_asc(h, x as u32);
        if h_key < best_h_key {
            best_h_key = h_key;
            s.h = h;
            s.bi = x;
            s.single = single;
        }
        let g_key = simd::pack_desc_idx_asc(gain, x as u32);
        if g_key < best_g_key {
            best_g_key = g_key;
            s.bsi = x;
            s.singles = single;
            s.hs = h;
        }
    }
    if s.bi == usize::MAX {
        s.h = f32::NAN;
        s.single = f32::NAN;
    }
    if s.bsi == usize::MAX {
        s.hs = f32::NAN;
        s.singles = f32::NAN;
    }
    s
}

#[cfg(test)]
mod packed_best_tests {
    use super::*;
    #[test]
    fn scalar_null_model_uses_same_winners_and_actual_scores() {
        let n=129;
        for seed in 0..128 {
            let p:Vec<f32>=(0..n).map(|i|((i*13+seed*17) as f32).sin()*0.0002+0.3).collect();
            let q:Vec<f32>=(0..n).map(|i|((i*19+seed*11) as f32).cos()*0.0002+0.3).collect();
            let mask:Vec<_>=(0..n).map(|i|i%11!=0).collect();
            let addm:Vec<_>=mask.iter().map(|&m|if m {0.0}else{f32::NEG_INFINITY}).collect();
            let (b,c)=(3,7);
            let mut h=vec![0.0;n];let mut g=vec![0.0;n];
            simd::fused_hg_stats(&mut h,&mut g,0.7,0.4,&p,&q,&addm,b,c);
            let (bi,bsi)=simd::argmax_packed_two(&h,&g);
            let s=best_x(n,&p,&q,0.7,0.4,b,c,&mask);
            assert_eq!((s.bi,s.bsi),(bi,bsi));
            assert_eq!(s.h.to_bits(),h[bi].to_bits());
            assert_eq!(s.hs.to_bits(),h[bsi].to_bits());
            assert_eq!((s.h-s.single).to_bits(),g[bi].to_bits());
            assert_eq!((s.hs-s.singles).to_bits(),g[bsi].to_bits());
        }
    }
}

/// 单张有向矩阵的全套配对网络产物。
///
/// * `sdir` —— n×n 行主序有向交互矩阵（`compute_matrices` 的 cnt_*/vol_* 等）
/// * `mask` —— 参与配对与候选 X 的股票掩码（= 有成交的股票；无成交空股为 false）
/// * `dec`  —— 每股活跃度十分位（空模型分层键）
/// * `stock_idx` —— mask==true 的下标列表（空模型抽样池）
/// * `null_mc` —— 空模型蒙特卡洛抽样上限
/// * `exz_bins` —— (p,q) 条件网格边数（默认 10）
pub(crate) fn net_one_mat(
    sdir: &[f32],
    n: usize,
    dec: &[usize],
    mask: &[bool],
    stock_idx: &[usize],
    null_mc: usize,
    exz_bins: usize,
    mat_name: &str,
    topk_ks: Option<&[usize]>,
    split: Option<(&SplitRho, &[f64], &[f64])>,
) -> NetMats {
    let t0 = std::time::Instant::now();
    let total_started = t0;
    let stage_enabled = std::env::var("YUPEI_STAGE").is_ok();
    let stage = |name: &str, started: std::time::Instant| {
        if stage_enabled {
            eprintln!("[net-stage] name={mat_name} stage={name} seconds={:.6}",
                started.elapsed().as_secs_f64());
        }
    };
    // ---- E_i = 2 * S_dir[i][i] ----
    let e: Vec<f64> = (0..n).map(|i| 2.0 * sdir[i * n + i] as f64).collect();
    let n_bad = e.iter().filter(|&&v| !(v > 0.0)).count();

    // ---- rho（行主序）与 rhoT（列读用）----
    let mut rho = vec![0.0f32; n * n];
    rho.par_chunks_mut(n).enumerate().for_each(|(i, row)| {
        row[i] = 1.0;
        let si = e[i].sqrt();
        for j in (i + 1)..n {
            let sj = e[j].sqrt();
            row[j] = ((sdir[i * n + j] + sdir[j * n + i]) as f64 / (si * sj)) as f32;
        }
    });
    // 上三角算完后镜像到下三角（读刚写完的行，cache 热）
    for i in 0..n {
        for j in 0..i {
            rho[i * n + j] = rho[j * n + i];
        }
    }
    // rho is mirrored bit for bit above; its transpose is the same immutable data.
    let rhot = &rho;
    // 方向拆分: 三套取数矩阵分别代表 (a,c)、(b,c)、(a,b) 三个方向组合。
    // 不拆分时三者都等于 rhot（现有行为）。
    // 方向拆分时自能量也要分两套: B 用方向 a 的能量、C 用方向 b 的能量
    // （X 的能量已经吸收进 rho 的归一化里）。
    let (pb, pc, rbc, ea, eb2) = match split {
        Some((sp, ea, eb)) => (sp.b_row, sp.c_row, sp.bc, ea, eb),
        None => (
            rhot.as_slice(),
            rhot.as_slice(),
            rhot.as_slice(),
            e.as_slice(),
            e.as_slice(),
        ),
    };
    // 精度实验：设 YUPEI_BF16=1 时把「C 对 X」那一套相关值截断到 16 位精度（bf16），
    // 用来量出降低精度对最终因子值的影响。默认关闭，行为与原来逐位一致。
    let pc_trunc: Vec<f32>;
    let pc: &[f32] = if std::env::var("YUPEI_BF16").is_ok() {
        pc_trunc = pc
            .iter()
            .map(|&x| f32::from_bits(x.to_bits() & 0xFFFF_0000))
            .collect();
        &pc_trunc
    } else {
        pc
    };

    // ---- 全部配对的 X0 / RHO_U / SINGLE / G + 分数矩阵 + 软质量向量 ----
    let mut g = vec![0.0f32; n * n];
    let mut ru = vec![0.0f32; n * n];
    let mut sg = vec![0.0f32; n * n];
    // X0 未计算的格子写 -1 哨兵：0 是合法下标，写 0 会被下游当成"X0=codes[0]"。
    let mut x0 = vec![-1.0f32; n * n];
    let mut gs = vec![0.0f32; n * n];
    let mut rus = vec![0.0f32; n * n];
    let mut sgs = vec![0.0f32; n * n];
    let mut x0s = vec![-1.0f32; n * n];
    let mut grel = vec![f32::NAN; n * n];
    let mut gsh = vec![f32::NAN; n * n];
    let mut gcov = vec![f32::NAN; n * n];
    let mut gnpos = vec![f32::NAN; n * n];
    let mut ghhi = vec![f32::NAN; n * n];
    let mut pbv = vec![f32::NAN; n * n];
    let mut qcv = vec![f32::NAN; n * n];
    let mut pbsv = vec![f32::NAN; n * n];
    let mut qcsv = vec![f32::NAN; n * n];
    // 软质量向量（每股一个数）：emass=Σ_pairs relu(h)/Σrelu(h)，gmass=emass 加权 gap，
    // g2mass=emass 加权 gap²，smass/sgmass/sg2mass 是 gap 口径同款。
    // 按固定块切 B、块内顺序累加、按下标合并 → 确定性。
    const MBLK: usize = 256;
    let npair_eff;
    let mass_vectors: Vec<f32>;
    let mut topk_total: Option<TopkAcc>;
    let mirror_output = std::env::var("YUPEI_NET_MIRROR").as_deref() != Ok("0");
    stage("setup", t0);
    let scan_started = std::time::Instant::now();
    {
        let sh = Shared {
            g: g.as_mut_ptr(),
            ru: ru.as_mut_ptr(),
            sg: sg.as_mut_ptr(),
            x0: x0.as_mut_ptr(),
            gs: gs.as_mut_ptr(),
            rus: rus.as_mut_ptr(),
            sgs: sgs.as_mut_ptr(),
            x0s: x0s.as_mut_ptr(),
            grel: grel.as_mut_ptr(),
            gsh: gsh.as_mut_ptr(),
            gcov: gcov.as_mut_ptr(),
            gnpos: gnpos.as_mut_ptr(),
            ghhi: ghhi.as_mut_ptr(),
            pb: pbv.as_mut_ptr(),
            qc: qcv.as_mut_ptr(),
            pbs: pbsv.as_mut_ptr(),
            qcs: qcsv.as_mut_ptr(),
        };
        let e_ref = ea;
        let e_ref2 = eb2;
        let pb_ref = pb;
        let pc_ref = pc;
        let rbc_ref = rbc;
        let mask_ref = mask;
        let sh_ref: &Shared = &sh;
        let nb = MBLK.min(n).max(1);
        let bounds: Vec<(usize, usize)> =
            (0..nb).map(|k| (k * n / nb, (k + 1) * n / nb)).collect();
        // -inf 掩码：非活跃位置加 -inf，使 h/gap 天然落到 argmax 之外、relu 为 0。
        let addm: Vec<f32> = (0..n)
            .map(|i| if mask[i] { 0.0 } else { f32::NEG_INFINITY })
            .collect();
        let n_stocks = mask.iter().filter(|&&v| v).count();
        let n_valid_pair = (n_stocks as u32).saturating_sub(2);
        // ---- topk 合并到这一趟扫描：参数与 topk_one_mat 完全一致 ----
        let topk_on = topk_ks.is_some();
        let ks_ref: &[usize] = topk_ks.unwrap_or(&[]);
        let nlev_t = ks_ref.len();
        let maxk_t = ks_ref.iter().copied().max().unwrap_or(0);
        let take_t = if topk_on {
            n_stocks.saturating_sub(2).min(maxk_t)
        } else {
            0
        };
        topk_total = if topk_on {
            Some(TopkAcc::new(n, nlev_t))
        } else {
            None
        };
        // Exact batching by default; 1 retains the original per-pair path for
        // comparison. Pair order and the original reduction blocks stay fixed.
        let mass_batch4 = std::env::var("YUPEI_PAIR_MASS_BATCH").as_deref() != Ok("1");
        let mass_capacity = if mass_batch4 { 4 } else { 1 };
        type Mass = (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, u64, Vec<TopkAcc>, [u64; 4]);
        let pairs_started = std::time::Instant::now();
        let parts: Vec<Mass> = bounds
            .par_iter()
            .map(|&(b0, b1)| {
                let sh = sh_ref;
                let mut em = vec![0.0f32; n];
                let mut gm = vec![0.0f32; n];
                let mut g2m = vec![0.0f32; n];
                let mut sm = vec![0.0f32; n];
                let mut sgm = vec![0.0f32; n];
                let mut sg2m = vec![0.0f32; n];
                let mut npair = 0u64;
                let mut h_batch = vec![0.0f32; mass_capacity * n];
                let mut g_batch = vec![0.0f32; mass_capacity * n];
                let mut mass_scales = [crate::yupei_pair_mass_batch::PairScale {
                    invh: 1.0, invg: 1.0, do_h: false, do_g: false,
                }; 4];
                let mut pending_mass = 0usize;
                let mut mass_flags = [0u64; 4];
                // topk 的块内缓冲（热路径上不再分配）
                let mut kh: Vec<u64> = vec![0u64; n];
                let mut kg: Vec<u64> = vec![0u64; n];
                let mut mid_buf: Vec<u64> = vec![0u64; n];
                let mut idx: Vec<u32> = vec![0u32; maxk_t];
                let mut gidx: Vec<u32> = vec![0u32; maxk_t];
                let mut pref: Vec<f64> = vec![0.0f64; maxk_t];
                // 子块 = topk_one_mat 的 NBLOCKS 分区与 [b0,b1) 的交（512 是 256 的两倍细分，
                // 所以每个 256 块正好被切成两段），保证两边各自的累加分组不变。
                let mut subs: Vec<(usize, usize)> = Vec::new();
                if topk_on {
                    let nbt = super::topk::NBLOCKS.min(n).max(1);
                    for j in 0..nbt {
                        let s0 = (j * n / nbt).max(b0);
                        let s1 = ((j + 1) * n / nbt).min(b1);
                        if s0 < s1 {
                            subs.push((s0, s1));
                        }
                    }
                } else {
                    subs.push((b0, b1));
                }
                let mut tks: Vec<TopkAcc> = Vec::with_capacity(subs.len());
                for (s0, s1) in subs.into_iter() {
                    let mut tacc: Option<TopkAcc> =
                        if topk_on { Some(TopkAcc::new(n, nlev_t)) } else { None };
                    for b in s0..s1 {
                    if !mask_ref[b] {
                        continue;
                    }
                    let p = &pb_ref[b * n..b * n + n];
                    let eb = e_ref[b];
                    let sb = eb.sqrt();
                    for c in (b + 1)..n {
                        if !mask_ref[c] {
                            continue;
                        }
                        let ec = e_ref2[c];
                        let sc = ec.sqrt();
                        let r = rbc_ref[b * n + c] as f64;
                        let d = (eb + ec + 2.0 * r * sb * sc).sqrt();
                        if !(d > 0.0) {
                            continue;
                        }
                        let wb = (sb / d) as f32;
                        let wc = (sc / d) as f32;
                        let q = &pc_ref[c * n..c * n + n];
                        // Failed/empty pairs reuse this uncommitted slot. Full
                        // h/g, TopK and pair matrix outputs are computed now.
                        let slot = pending_mass * n;
                        let hb = &mut h_batch[slot..slot + n];
                        let gb = &mut g_batch[slot..slot + n];
                        // 一趟算完 h/g（b、c 位置写 -inf）及原顺序的三个和 + 计数。
                        // 此处只需要是否有有效值；省去后续不消费的两个最大值归约。
                        let (any_valid, sh_pos, sg_pos, sqg_pos, npos) = crate::yupei_pair_scan_stats::fused_hg_stats_no_max(
                            hb, gb, wb, wc, p, q, &addm, b, c,
                        );
                        if !any_valid {
                            if let Some(t) = tacc.as_mut() { t.skipped += 1; }
                            // No valid explanation: retain -1 winner sentinels and undefined scores.
                            unsafe {
                                for ptr in [sh.g, sh.ru, sh.sg, sh.gs, sh.rus, sh.sgs] {
                                    *ptr.add(b*n+c) = f32::NAN;
                                    if !mirror_output { *ptr.add(c*n+b) = f32::NAN; }
                                }
                            }
                            continue;
                        }
                        // ---- topk 合并段：h/gap 只算这一遍，按 topk 原来的顺序累计 ----
                        let mut topk_done = false;
                        if let Some(t) = tacc.as_mut() {
                            if take_t > 0 {
                                simd::select_topk_keys(&hb, take_t, &mut kh, &mut mid_buf);
                                simd::select_topk_keys(&gb, take_t, &mut kg, &mut mid_buf);
                                for v in 0..take_t {
                                    idx[v] = kh[v] as u32;
                                    gidx[v] = kg[v] as u32;
                                }
                                let mut sf = 0.0f64;
                                for v in 0..take_t {
                                    let y = idx[v] as usize;
                                    let sc = hb[y];
                                    if sc > 0.0 {
                                        sf += sc as f64;
                                    }
                                    pref[v] = sf;
                                }
                                absorb_topk(t, n, 0, nlev_t, ks_ref, &idx[..take_t], &hb, &gb, &pref);
                                sf = 0.0;
                                for v in 0..take_t {
                                    let y = gidx[v] as usize;
                                    let sc = gb[y];
                                    if sc > 0.0 {
                                        sf += sc as f64;
                                    }
                                    pref[v] = sf;
                                }
                                absorb_topk(t, n, 1, nlev_t, ks_ref, &gidx[..take_t], &hb, &gb, &pref);
                                t.pairs += 1;
                                topk_done = true;
                            }
                        }
                        // TopK and non-TopK paths use one quantized score/index order.
                        // The returned scores belong to the selected stocks themselves.
                        let (bi, bsi) = if topk_done {
                            (idx[0] as usize, gidx[0] as usize)
                        } else {
                            simd::argmax_packed_two(&hb, &gb)
                        };
                        if bi == usize::MAX || bsi == usize::MAX { continue; }
                        let h = hb[bi];
                        let gs_ = gb[bsi];
                        let (pbi, qci) = (p[bi], q[bi]);
                        let single = if pbi > qci { pbi } else { qci };
                        let (pbs, qcs) = (p[bsi], q[bsi]);
                        let singles = if pbs > qcs { pbs } else { qcs };
                        let gain = gb[bi];
                        let gains = gs_;
                        let rus_v = hb[bsi];
                        let nv = n_valid_pair.max(1) as f32;
                        let grel_v = if single > 1e-4 { gain / single } else { f32::NAN };
                        let gsh_v = if h > 1e-4 { gain / h } else { f32::NAN };
                        let ghhi_v = if sg_pos > 0.0 {
                            sqg_pos / (sg_pos * sg_pos)
                        } else {
                            f32::NAN
                        };
                        unsafe {
                            let put = |m: *mut f32, v: f32| {
                                *m.add(b * n + c) = v;
                                if !mirror_output { *m.add(c * n + b) = v; }
                            };
                            put(sh.g, gain);
                            put(sh.ru, h);
                            put(sh.sg, single);
                            put(sh.x0, bi as f32);
                            put(sh.gs, gains);
                            put(sh.rus, rus_v);
                            put(sh.sgs, singles);
                            put(sh.x0s, bsi as f32);
                            put(sh.grel, grel_v);
                            put(sh.gsh, gsh_v);
                            put(sh.gcov, sg_pos / nv);
                            put(sh.gnpos, npos as f32 / nv);
                            put(sh.ghhi, ghhi_v);
                            put(sh.pb, pbi);
                            put(sh.qc, qci);
                            put(sh.pbs, pbs);
                            put(sh.qcs, qcs);
                        }
                        npair += 1;
                        // Compute the original guarded inverses at exactly
                        // the old mass-call position, then preserve pair order.
                        let scale = crate::yupei_pair_mass_batch::PairScale {
                            invh: if sh_pos > MASS_EPS { 1.0 / sh_pos } else { 1.0 },
                            invg: if sg_pos > MASS_EPS { 1.0 / sg_pos } else { 1.0 },
                            do_h: sh_pos > MASS_EPS,
                            do_g: sg_pos > MASS_EPS,
                        };
                        if stage_enabled {
                            mass_flags[scale.do_h as usize | ((scale.do_g as usize) << 1)] += 1;
                        }
                        if mass_batch4 {
                            mass_scales[pending_mass] = scale;
                            pending_mass += 1;
                            if pending_mass == 4 {
                                crate::yupei_pair_mass_batch::mass_acc_hg_batch4(
                                    &mut em, &mut gm, &mut g2m, &mut sm, &mut sgm, &mut sg2m,
                                    &h_batch, &g_batch, &mass_scales,
                                );
                                pending_mass = 0;
                            }
                        } else {
                            simd::mass_acc_hg(
                                &mut em, &mut gm, &mut g2m, &mut sm, &mut sgm, &mut sg2m,
                                hb, gb, scale.invh, scale.invg, scale.do_h, scale.do_g,
                            );
                        }
                    }
                }
                    // Never carry pending mass across a TopK subblock or the
                    // original Mass block; no float reduction groups change.
                    if pending_mass != 0 {
                        crate::yupei_pair_mass_batch::mass_acc_hg_batch4(
                            &mut em, &mut gm, &mut g2m, &mut sm, &mut sgm, &mut sg2m,
                            &h_batch[..pending_mass * n], &g_batch[..pending_mass * n],
                            &mass_scales[..pending_mass],
                        );
                        pending_mass = 0;
                    }
                    if let Some(a) = tacc {
                        tks.push(a);
                    }
                }
                (em, gm, g2m, sm, sgm, sg2m, npair, tks, mass_flags)
            })
            .collect();
        stage("scan_pairs", pairs_started);
        let merge_started = std::time::Instant::now();
        let mut emass = vec![0.0f64; n];
        let mut gmass = vec![0.0f64; n];
        let mut g2mass = vec![0.0f64; n];
        let mut smass = vec![0.0f64; n];
        let mut sgmass = vec![0.0f64; n];
        let mut sg2mass = vec![0.0f64; n];
        let mut npair_tot = 0u64;
        let mut mass_flags_tot = [0u64; 4];
        for (em, gm, g2m, sm, sgm, sg2m, np, tks, flags) in parts.iter() {
            if let Some(t) = topk_total.as_mut() {
                for a in tks.iter() {
                    t.add(a);
                }
            }
            for x in 0..n {
                emass[x] += em[x] as f64;
                gmass[x] += gm[x] as f64;
                g2mass[x] += g2m[x] as f64;
                smass[x] += sm[x] as f64;
                sgmass[x] += sgm[x] as f64;
                sg2mass[x] += sg2m[x] as f64;
            }
            npair_tot += np;
            if stage_enabled {
                for kind in 0..4 { mass_flags_tot[kind] += flags[kind]; }
            }
        }
        if stage_enabled {
            eprintln!("[mass-flags] name={mat_name} batch={mass_capacity} none={} h_only={} g_only={} both={} pairs={npair_tot}",
                mass_flags_tot[0], mass_flags_tot[1], mass_flags_tot[2], mass_flags_tot[3]);
        }
        npair_eff = npair_tot;
        let inv_p = if npair_tot > 0 { 1.0 / npair_tot as f64 } else { f64::NAN };
        let mut mv = Vec::with_capacity(6 * n);
        for arr in [&emass, &gmass, &g2mass, &smass, &sgmass, &sg2mass] {
            for x in 0..n {
                mv.push((arr[x] * inv_p) as f32);
            }
        }
        mass_vectors = mv;
        stage("scan_reduce", merge_started);
    }
    stage("scan", scan_started);
    let mirror_started = std::time::Instant::now();
    if mirror_output {
        let started = std::time::Instant::now();
        [&mut g, &mut ru, &mut sg, &mut x0, &mut gs, &mut rus, &mut sgs,
         &mut x0s, &mut grel, &mut gsh, &mut gcov, &mut gnpos, &mut ghhi,
         &mut pbv, &mut qcv, &mut pbsv, &mut qcsv]
            .into_par_iter().for_each(|mat| super::mirror_stage::mirror_lower(mat, n));
        if std::env::var("YUPEI_STAGE").is_ok() {
            eprintln!("[mirror-stage] name={mat_name} seconds={:.6}", started.elapsed().as_secs_f64());
        }
    }
    stage("mirror", mirror_started);
    if super::verbose() {
        eprintln!(
            "[net:{mat_name}] 有效配对 {npair_eff}（自能量非正 {n_bad}），扫描 {:.1}s",
            t0.elapsed().as_secs_f32()
        );
    }

    // ---- 空模型：按 (B 活跃度档, C 活跃度档) 分层的随机配对 ----
    let t0 = std::time::Instant::now();
    let nstrata = 100usize;
    let per = (null_mc / nstrata).max(20);
    let mut rng = Rng::new(0xC0FFEE_1234_5678);
    let mut samples: Vec<(usize, f32, f32)> = Vec::with_capacity(nstrata * per);
    for _ in 0..(nstrata * per) {
        let b = stock_idx[rng.below(stock_idx.len())];
        let c = stock_idx[rng.below(stock_idx.len())];
        if b == c {
            continue;
        }
        let (b, c) = if b < c { (b, c) } else { (c, b) };
        let eb = ea[b];
        let ec = eb2[c];
        let sb = eb.sqrt();
        let sc = ec.sqrt();
        let r = rbc[b * n + c] as f64;
        let d = (eb + ec + 2.0 * r * sb * sc).sqrt();
        if !(d > 0.0) {
            continue;
        }
        // Zero/nonfinite scores retain the original scalar edge-case path.
        let (gv, gsv) = (g[b*n+c], gs[b*n+c]);
        let (sample_g, sample_gs) = if gv.is_finite() && gsv.is_finite()
            && gv != 0.0 && gsv != 0.0 && x0[b*n+c] >= 0.0 && x0s[b*n+c] >= 0.0 {
            (gv, gsv)
        } else {
            let p = &pb[b*n..b*n+n];
            let q = &pc[c*n..c*n+n];
            let s = best_x(n, p, q, (sb/d) as f32, (sc/d) as f32, b, c, mask);
            if s.bi == usize::MAX || s.bsi == usize::MAX { continue; }
            (s.h-s.single, s.hs-s.singles)
        };
        // 分层键必须与 (b,c) 的顺序无关，否则 Z 矩阵不对称
        let (d1, d2) = if dec[b] <= dec[c] { (dec[b], dec[c]) } else { (dec[c], dec[b]) };
        samples.push((d1 * 10 + d2, sample_g, sample_gs));
    }
    let mut cnt = vec![0usize; nstrata];
    let mut sum = vec![0.0f64; nstrata];
    let mut sumsq = vec![0.0f64; nstrata];
    let mut cnts = vec![0usize; nstrata];
    let mut sums2 = vec![0.0f64; nstrata];
    let mut sumsq2 = vec![0.0f64; nstrata];
    let (mut gsum, mut gsumsq) = (0.0f64, 0.0f64);
    let (mut hsum, mut hsumsq) = (0.0f64, 0.0f64);
    let mut gcnt = 0usize;
    for &(si, v, w) in samples.iter() {
        cnt[si] += 1;
        sum[si] += v as f64;
        sumsq[si] += (v as f64) * (v as f64);
        cnts[si] += 1;
        sums2[si] += w as f64;
        sumsq2[si] += (w as f64) * (w as f64);
        gsum += v as f64;
        gsumsq += (v as f64) * (v as f64);
        hsum += w as f64;
        hsumsq += (w as f64) * (w as f64);
        gcnt += 1;
    }
    let gmean = gsum / gcnt.max(1) as f64;
    let gstd = (gsumsq / gcnt.max(1) as f64 - gmean * gmean).max(0.0).sqrt();
    let hmean = hsum / gcnt.max(1) as f64;
    let hstd = (hsumsq / gcnt.max(1) as f64 - hmean * hmean).max(0.0).sqrt();
    let mut strata: Vec<(f64, f64, usize)> = Vec::with_capacity(nstrata);
    let mut strata_s: Vec<(f64, f64, usize)> = Vec::with_capacity(nstrata);
    for s in 0..nstrata {
        if cnt[s] >= 30 {
            let m = sum[s] / cnt[s] as f64;
            let sd = (sumsq[s] / cnt[s] as f64 - m * m).max(0.0).sqrt();
            strata.push((m, sd, cnt[s]));
        } else {
            strata.push((gmean, gstd, cnt[s]));
        }
        if cnts[s] >= 30 {
            let m = sums2[s] / cnts[s] as f64;
            let sd = (sumsq2[s] / cnts[s] as f64 - m * m).max(0.0).sqrt();
            strata_s.push((m, sd, cnts[s]));
        } else {
            strata_s.push((hmean, hstd, cnts[s]));
        }
    }
    if super::verbose() {
        eprintln!(
            "[net:{mat_name}] 空模型 {:.1}s（{} 层有效）",
            t0.elapsed().as_secs_f32(),
            strata.iter().filter(|s| s.2 >= 30).count()
        );
    }
    stage("null_mc", t0);

    // ---- Z = (G - 层均值) / 层标准差；ZS 同法，用 G* ----
    let z_started = std::time::Instant::now();
    let mut z = vec![0.0f32; n * n];
    let mut zs = vec![0.0f32; n * n];
    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }
            if !g[i*n+j].is_finite() { z[i*n+j] = f32::NAN; }
            if !gs[i*n+j].is_finite() { zs[i*n+j] = f32::NAN; }
            let (d1, d2) = if dec[i] <= dec[j] { (dec[i], dec[j]) } else { (dec[j], dec[i]) };
            let s = d1 * 10 + d2;
            let (m, sd, _) = strata[s];
            if sd > 1e-12 {
                z[i * n + j] = ((g[i * n + j] as f64 - m) / sd) as f32;
            }
            let (m2, sd2, _) = strata_s[s];
            if sd2 > 1e-12 {
                zs[i * n + j] = ((gs[i * n + j] as f64 - m2) / sd2) as f32;
            }
        }
    }
    stage("z", z_started);

    // ---- (p,q) 条件空模型：EXZ / EXZS ----
    // G 的 ~99% 方差来自 (p,q) 剖面；把分层变量换成
    // "(min(p,q), max(p,q)) 十分位联合网格"，扣掉相似度水平效应，
    // 剩下的是 B、C 低重叠（互补性）带来的那部分。
    let t0 = std::time::Instant::now();
    let bins = exz_bins;
    let ncell = bins * bins;
    let sort_baseline = std::env::var("YUPEI_EXZ_SORT_BASELINE").as_deref() == Ok("1");
    let cond_std = |kind: &str, pmat: &[f32], qmat: &[f32], wmat: &[f32]| -> Vec<f32> {
        let collect_started = std::time::Instant::now();
        const RB: usize = 256;
        let nrb = n.div_ceil(RB);
        let uv_pairs: Vec<(Vec<f64>, Vec<f64>)> = (0..nrb)
            .into_par_iter()
            .map(|rb| {
                let mut us: Vec<f64> = Vec::new();
                let mut vs: Vec<f64> = Vec::new();
                for i in (rb * RB)..((rb + 1) * RB).min(n) {
                    for j in (i + 1)..n {
                        let pv = pmat[i * n + j] as f64;
                        let qv = qmat[i * n + j] as f64;
                        if !(pv.is_finite() && qv.is_finite()) {
                            continue;
                        }
                        let (u, v) = if pv <= qv { (pv, qv) } else { (qv, pv) };
                        us.push(u);
                        vs.push(v);
                    }
                }
                (us, vs)
            })
            .collect();
        let mut us: Vec<f64> = Vec::new();
        let mut vs: Vec<f64> = Vec::new();
        for (a, b) in uv_pairs {
            us.extend_from_slice(&a);
            vs.extend_from_slice(&b);
        }
        let m = us.len();
        let mut out = vec![f32::NAN; n * n];
        if m < 100 {
            return out;
        }
        let collect_s = collect_started.elapsed().as_secs_f64();
        let quantile_started = std::time::Instant::now();
        let (cuts_u, cuts_v) = if sort_baseline {
            let mut su = us.clone();
            let mut sv = vs.clone();
            su.par_sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
            sv.par_sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
            (
                (1..bins).map(|k| su[(k * m / bins).min(m - 1)]).collect::<Vec<f64>>(),
                (1..bins).map(|k| sv[(k * m / bins).min(m - 1)]).collect::<Vec<f64>>(),
            )
        } else {
            // us/vs are not used again. Partition these buffers directly and
            // release them before aggregation instead of retaining four copies.
            let cuts = |mut values: Vec<f64>| {
                match super::exact_quantile::exact_quantile_cuts(&mut values, bins) {
                    Some(cuts) => cuts,
                    None => {
                        values.par_sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
                        (1..bins).map(|k| values[(k * m / bins).min(m - 1)]).collect()
                    }
                }
            };
            rayon::join(|| cuts(us), || cuts(vs))
        };
        let quantile_s = quantile_started.elapsed().as_secs_f64();
        let aggregate_started = std::time::Instant::now();
        let cell_of = |u: f64, v: f64| -> usize {
            let du = cuts_u.partition_point(|&c| u > c);
            let dv = cuts_v.partition_point(|&c| v > c);
            du * bins + dv
        };
        let agg: Vec<(Vec<u64>, Vec<f64>, Vec<f64>)> = (0..nrb)
            .into_par_iter()
            .map(|rb| {
                let mut cnt = vec![0u64; ncell];
                let mut sm = vec![0.0f64; ncell];
                let mut sq = vec![0.0f64; ncell];
                for i in (rb * RB)..((rb + 1) * RB).min(n) {
                    for j in (i + 1)..n {
                        let pv = pmat[i * n + j] as f64;
                        let qv = qmat[i * n + j] as f64;
                        let w = wmat[i * n + j] as f64;
                        if !(pv.is_finite() && qv.is_finite() && w.is_finite()) {
                            continue;
                        }
                        let (u, v) = if pv <= qv { (pv, qv) } else { (qv, pv) };
                        let c = cell_of(u, v);
                        cnt[c] += 1;
                        sm[c] += w;
                        sq[c] += w * w;
                    }
                }
                (cnt, sm, sq)
            })
            .collect();
        let mut cnt = vec![0u64; ncell];
        let mut sm = vec![0.0f64; ncell];
        let mut sq = vec![0.0f64; ncell];
        for (c, s, q) in agg.iter() {
            for k in 0..ncell {
                cnt[k] += c[k];
                sm[k] += s[k];
                sq[k] += q[k];
            }
        }
        let ntot = cnt.iter().sum::<u64>().max(1) as f64;
        let gm = sm.iter().sum::<f64>() / ntot;
        let gs2 = (sq.iter().sum::<f64>() / ntot - gm * gm).max(0.0).sqrt();
        let mut stats = vec![(gm, gs2); ncell];
        for c in 0..ncell {
            if cnt[c] >= 30 {
                let mm = sm[c] / cnt[c] as f64;
                let sd = (sq[c] / cnt[c] as f64 - mm * mm).max(0.0).sqrt();
                stats[c] = (mm, sd);
            }
        }
        let aggregate_s = aggregate_started.elapsed().as_secs_f64();
        let output_started = std::time::Instant::now();
        out.par_chunks_mut(n).enumerate().for_each(|(i, row)| {
            for j in (i + 1)..n {
                let pv = pmat[i * n + j] as f64;
                let qv = qmat[i * n + j] as f64;
                let w = wmat[i * n + j] as f64;
                if !(pv.is_finite() && qv.is_finite() && w.is_finite()) {
                    continue;
                }
                let (u, v) = if pv <= qv { (pv, qv) } else { (qv, pv) };
                let (mm, sd) = stats[cell_of(u, v)];
                if sd > 1e-12 {
                    row[j] = ((w - mm) / sd) as f32;
                }
            }
        });
        for i in 0..n {
            for j in 0..i {
                out[i * n + j] = out[j * n + i];
            }
        }
        if stage_enabled {
            eprintln!(
                "[net-exz-stage] name={mat_name} kind={kind} collect={collect_s:.6} quantile={quantile_s:.6} aggregate={aggregate_s:.6} output={:.6}",
                output_started.elapsed().as_secs_f64()
            );
        }
        out
    };
    let (exz, exzs) = rayon::join(
        || cond_std("g", &pbv, &qcv, &g),
        || cond_std("gs", &pbsv, &qcsv, &gs),
    );
    if super::verbose() {
        eprintln!(
            "[net:{mat_name}] 空模型 {} 抽样, G mean={gmean:.4} std={gstd:.4}; (p,q) 条件化 {:.1}s",
            gcnt,
            t0.elapsed().as_secs_f32()
        );
    }
    stage("exz", t0);
    stage("total", total_started);

    NetMats {
        e, rho, g, ru, sg, x0, gs, rus, sgs, x0s, z, zs,
        grel, gsh, gcov, gnpos, ghhi,
        pb: pbv, qc: qcv, pbs: pbsv, qcs: qcsv,
        exz, exzs,
        mass: mass_vectors,
        topk_total,
        npair: npair_eff,
        strata,
        strata_s,
    }
}
