//! 玉佩多种刻画方式的统一 pipeline。
//!
//! 同一天、同一次读数据，一次算出五类因子：
//!   1. 原版   —— 10 毫秒落桶，与 `yupei_pair::split` 逐位一致
//!   2. 订单级 —— 按挂单号合并成交后再落桶
//!   3. 聚簇   —— Otsu 在相邻成交间隔上找断点，簇时间按成交量加权
//!   4. 领先度 —— 在相关矩阵层把「两个方向相加」换成「相减」
//!   5. 横截面分解 —— 不做配对，直接对成交强度曲线做截断奇异值分解
//!
//! 前三种共用同一套下游：桶单元 → 四分量矩阵 → 9 张方向相关矩阵 → 15 条链
//! → 653 列。区别只在桶单元怎么生成，所以 653 列的生成代码只写一遍、跑三遍。
//!
//! 相对单跑一次原版，这里做的提速：
//!   - 逐笔只读一次，三套共用（单日约省 60 秒）
//!   - 四分量矩阵跳过「该股票该块没有成交」的衰减场构建
//!   - 事件序列落到 10 毫秒网格后复用同一套扫描，订单级事件数比固定网格少两个数量级

use std::io;

use pyo3::prelude::*;
use rayon::prelude::*;

use crate::yupei_dist::matrix_stage::{BUCKET_US, N_BUCKETS, StockPrep};
use crate::yupei_pair::check::check_net;
use crate::yupei_pair::net::{decile_of, net_one_mat, SplitRho};
use crate::yupei_pair::reduce::{matset_from_net, produce_one_mat, set_soft, Out, SoftCfg};
use crate::yupei_pair::split::{
    dir_tag, energy, rho_ab, BC_COMBOS, D_ALL, D_M, D_P, X_DIRS,
};
use crate::yupei_pair::split_mat::{
    compute_split_mats_from_cells, compute_split_mats_simd, PmWeight, MM, MP, PM, PP,
};
use crate::yupei_pair::topk::{finalize_total, topk_one_mat};

/// 定稿版本跑的五种方向组合。按回测结果里每个方向组合贡献的入选因子数挑的，
/// 这五种合计保住 90 个入选变体里的 51 个；换成"信息量最优"的那五种只能保住 25 个。
/// 名字拆开是「配对两只的方向 + 第三只的方向」：pp=主买主买、AA=不分方向、
/// pm=主买主卖、mp=主卖主买；末位 a/p/m 是第三只看全部/主买/主卖。
pub const V5_CHAINS: [&str; 5] = ["spppa", "spppp", "spAAp", "sppma", "spmpm"];

/// 上午收市对应的桶号：9:30 到 11:30 共 7200 秒。reader 已把下午平移 90 分钟。
const AM_CUT: u32 = 720_000;

/// 桶单元的三种来源。
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum CellMode {
    Base,
    Order,
    Otsu,
    /// 订单方向视角：通道按「这笔成交吃的是买单还是卖单」分，而不是按主动方分。
    /// 主买成交吃的是卖方挂单 → 记「卖单」通道；主卖成交吃的是买方挂单 → 记「买单」通道。
    OrderSide,
}

impl CellMode {
    fn tag(self) -> &'static str {
        match self {
            CellMode::Base => "",
            CellMode::Order => "or_",
            CellMode::Otsu => "oc_",
            CellMode::OrderSide => "os_",
        }
    }
}

/// 逐笔记录 → 桶单元。三套逻辑只在这一层分叉，后面全部复用。
///
/// `u4 = [cnt_P, cnt_M, vol_P, vol_M]`，主买 flag 66、主卖 flag 83，其余不进。
fn build_cells(
    recs: &[crate::fast_csv_reader::TradeRecord],
    day_start: i64,
    mode: CellMode,
) -> Vec<(u32, [f32; 4])> {
    if recs.is_empty() {
        return Vec::new();
    }
    let flip = |r: &crate::fast_csv_reader::TradeRecord| -> Option<([f32; 4], f32)> {
        let sv = (r.volume as f64).max(0.0).sqrt() as f32;
        match r.flag {
            66 => Some(([1.0, 0.0, sv, 0.0], sv)),
            83 => Some(([0.0, 1.0, 0.0, sv], sv)),
            _ => None,
        }
    };
    let bucket_of = |t: i64| -> Option<u32> {
        let off = t - day_start;
        if off < 0 || off >= N_BUCKETS as i64 * BUCKET_US {
            None
        } else {
            Some((off / BUCKET_US) as u32)
        }
    };

    // 事件序列：时间为桶号（浮点，用于订单/聚簇的加权落桶），强度为四路单位向量
    let mut pts: Vec<(f64, [f32; 4])> = Vec::with_capacity(recs.len());
    match mode {
        CellMode::Base => {
            for r in recs {
                if let (Some(b), Some((u, _))) = (bucket_of(r.time_us), flip(r)) {
                    pts.push((b as f64, u));
                }
            }
        }
        CellMode::Order | CellMode::OrderSide => {
            // 按被动方挂单号聚合：主买吃的是卖挂单，主卖吃的是买挂单。
            // 两种模式的聚合完全一样，区别只在最后把哪一路放进 P 通道：
            //   Order     —— P 放主买（主动方视角）
            //   OrderSide —— P 放买单（订单方向视角，主卖成交吃的就是买单）
            use std::collections::HashMap;
            let mut agg: HashMap<i64, (f64, f64, f64)> = HashMap::new(); // (加权时间, 主买量, 主卖量)
            for r in recs {
                let Some(b) = bucket_of(r.time_us) else { continue };
                let Some((u, sv)) = flip(r) else { continue };
                let oid = if r.flag == 66 { r.ask_order } else { r.bid_order };
                let key = if oid > 0 { oid } else { -r.time_us };
                let w = sv.max(1e-6) as f64;
                let e = agg.entry(key).or_insert((0.0, 0.0, 0.0));
                e.0 += b as f64 * w;
                if u[0] > 0.0 {
                    e.1 += w;
                } else {
                    e.2 += w;
                }
            }
            for (tw, sp, sm) in agg.into_values() {
                let tot = sp + sm;
                if tot <= 0.0 {
                    continue;
                }
                let b = tw / tot;
                let (chan_p, chan_m) = if mode == CellMode::OrderSide {
                    (sm, sp) // 买单 = 主卖成交的量；卖单 = 主买成交的量
                } else {
                    (sp, sm) // 主买 = 主买成交的量；主卖 = 主卖成交的量
                };
                let mut u4 = [0.0f32; 4];
                if chan_p > 0.0 {
                    u4[0] = 1.0;
                    u4[2] = chan_p as f32;
                }
                if chan_m > 0.0 {
                    u4[1] = 1.0;
                    u4[3] = chan_m as f32;
                }
                pts.push((b, u4));
            }
        }
        CellMode::Otsu => {
            // 相邻成交间隔取对数，Otsu 找让两类分得最开的阈值，间隔超过它就断开
            let mut seq: Vec<(u32, [f32; 4], f64)> = Vec::with_capacity(recs.len());
            for r in recs {
                if let (Some(b), Some((u, sv))) = (bucket_of(r.time_us), flip(r)) {
                    seq.push((b, u, sv as f64));
                }
            }
            seq.sort_by_key(|x| x.0);
            if seq.len() < 30 {
                for (b, u, _) in seq {
                    pts.push((b as f64, u));
                }
            } else {
                let mut gaps: Vec<f64> = Vec::with_capacity(seq.len());
                for k in 1..seq.len() {
                    let d = (seq[k].0 - seq[k - 1].0) as f64;
                    if d > 0.0 {
                        gaps.push(d.ln());
                    }
                }
                let thr = if gaps.len() >= 10 {
                    otsu_threshold(&gaps).exp()
                } else {
                    f64::INFINITY
                };
                let mut seg_start = 0usize;
                let mut flush = |a: usize, b: usize, pts: &mut Vec<(f64, [f32; 4])>| {
                    if b - a < 3 {
                        return;
                    }
                    let mut tw = 0.0f64;
                    let mut wsum = 0.0f64;
                    let mut u4 = [0.0f32; 4];
                    for t in a..b {
                        let w = seq[t].2;
                        tw += seq[t].0 as f64 * w;
                        wsum += w;
                        for i in 0..4 {
                            u4[i] += seq[t].1[i];
                        }
                    }
                    if wsum > 0.0 {
                        pts.push((tw / wsum, u4));
                    }
                };
                for k in 1..seq.len() {
                    if (seq[k].0 - seq[k - 1].0) as f64 > thr {
                        flush(seg_start, k, &mut pts);
                        seg_start = k;
                    }
                }
                flush(seg_start, seq.len(), &mut pts);
            }
        }
    }

    // 事件落回 10 毫秒桶：同一桶累加，保证与下游的固定网格完全兼容
    let mut ev: Vec<(u32, [f32; 4])> = pts
        .into_iter()
        .filter_map(|(b, u)| {
            let k = b.round();
            if k < 0.0 || k >= N_BUCKETS as f64 {
                None
            } else {
                Some((k as u32, u))
            }
        })
        .collect();
    ev.sort_by_key(|x| x.0);
    let mut cells: Vec<(u32, [f32; 4])> = Vec::with_capacity(ev.len());
    for (k, u) in ev {
        if let Some((bk, ua)) = cells.last_mut() {
            if *bk == k {
                for t in 0..4 {
                    ua[t] += u[t];
                }
                continue;
            }
        }
        cells.push((k, u));
    }
    cells
}


/// 一次遍历逐笔，同时生成四套桶单元。
///
/// 原来四套各扫一遍全量记录，四次遍历做的是同一件事的四个投影。这里合并成一次：
/// 基础桶、订单聚合、聚簇序列在同一趟里累出来，订单方向级从订单聚合结果交换通道派生。
fn build_all_cells(
    recs: &[crate::fast_csv_reader::TradeRecord],
    day_start: i64,
) -> (
    Vec<(u32, [f32; 4])>,
    Vec<(u32, [f32; 4])>,
    Vec<(u32, [f32; 4])>,
    Vec<(u32, [f32; 4])>,
) {
    use std::collections::HashMap;
    let mut base: Vec<(u32, [f32; 4])> = Vec::with_capacity(recs.len());
    let mut seq: Vec<(u32, [f32; 4], f64)> = Vec::with_capacity(recs.len());
    let mut agg: HashMap<i64, (f64, f64, f64)> = HashMap::with_capacity(recs.len() / 2);

    for r in recs {
        let off = r.time_us - day_start;
        if off < 0 || off >= N_BUCKETS as i64 * BUCKET_US {
            continue;
        }
        let b = (off / BUCKET_US) as u32;
        let sv = (r.volume as f64).max(0.0).sqrt() as f32;
        let mut u4 = [0.0f32; 4];
        match r.flag {
            66 => {
                u4[0] = 1.0;
                u4[2] = sv;
            }
            83 => {
                u4[1] = 1.0;
                u4[3] = sv;
            }
            _ => continue,
        }
        base.push((b, u4));
        seq.push((b, u4, sv as f64));
        let oid = if r.flag == 66 { r.ask_order } else { r.bid_order };
        let key = if oid > 0 { oid } else { -r.time_us };
        let w = (sv as f64).max(1e-6);
        let e = agg.entry(key).or_insert((0.0, 0.0, 0.0));
        e.0 += b as f64 * w;
        if r.flag == 66 {
            e.1 += w;
        } else {
            e.2 += w;
        }
    }

    // ---- 基础桶：排序后合并同桶 ----
    base.sort_by_key(|x| x.0);
    let mut base_out: Vec<(u32, [f32; 4])> = Vec::with_capacity(base.len());
    for (k, u) in base {
        if let Some((bk, ua)) = base_out.last_mut() {
            if *bk == k {
                for t in 0..4 {
                    ua[t] += u[t];
                }
                continue;
            }
        }
        base_out.push((k, u));
    }

    // ---- 订单聚合：双向派生（主动方视角与订单方向视角只差通道交换）----
    let mut order_pts: Vec<(f64, [f32; 4])> = Vec::with_capacity(agg.len());
    let mut side_pts: Vec<(f64, [f32; 4])> = Vec::with_capacity(agg.len());
    for (tw, sp, sm) in agg.into_values() {
        let tot = sp + sm;
        if tot <= 0.0 {
            continue;
        }
        let b = tw / tot;
        let mut uo = [0.0f32; 4];
        if sp > 0.0 {
            uo[0] = 1.0;
            uo[2] = sp as f32;
        }
        if sm > 0.0 {
            uo[1] = 1.0;
            uo[3] = sm as f32;
        }
        order_pts.push((b, uo));
        let mut us = [0.0f32; 4];
        if sm > 0.0 {
            us[0] = 1.0;
            us[2] = sm as f32;
        }
        if sp > 0.0 {
            us[1] = 1.0;
            us[3] = sp as f32;
        }
        side_pts.push((b, us));
    }

    // ---- 聚簇：Otsu 在间隔上找断点 ----
    let otsu_out = cluster_seq(seq);

    (
        base_out,
        bucketize(order_pts),
        bucketize(side_pts),
        otsu_out,
    )
}

/// 事件序列（时间为桶号浮点）→ 落回 10 毫秒桶并合并同桶。
fn bucketize(mut pts: Vec<(f64, [f32; 4])>) -> Vec<(u32, [f32; 4])> {
    let mut ev: Vec<(u32, [f32; 4])> = pts
        .drain(..)
        .filter_map(|(b, u)| {
            let k = b.round();
            if k < 0.0 || k >= N_BUCKETS as f64 {
                None
            } else {
                Some((k as u32, u))
            }
        })
        .collect();
    ev.sort_by_key(|x| x.0);
    let mut out: Vec<(u32, [f32; 4])> = Vec::with_capacity(ev.len());
    for (k, u) in ev {
        if let Some((bk, ua)) = out.last_mut() {
            if *bk == k {
                for t in 0..4 {
                    ua[t] += u[t];
                }
                continue;
            }
        }
        out.push((k, u));
    }
    out
}

/// 聚簇：按相邻成交间隔的 Otsu 断点切段，段内成交量加权出一个时间点。
fn cluster_seq(mut seq: Vec<(u32, [f32; 4], f64)>) -> Vec<(u32, [f32; 4])> {
    seq.sort_by_key(|x| x.0);
    if seq.len() < 30 {
        return bucketize(seq.into_iter().map(|(b, u, _)| (b as f64, u)).collect());
    }
    let mut gaps: Vec<f64> = Vec::with_capacity(seq.len());
    for k in 1..seq.len() {
        let d = (seq[k].0 - seq[k - 1].0) as f64;
        if d > 0.0 {
            gaps.push(d.ln());
        }
    }
    let thr = if gaps.len() >= 10 {
        otsu_threshold(&gaps).exp()
    } else {
        f64::INFINITY
    };
    let mut pts: Vec<(f64, [f32; 4])> = Vec::new();
    let mut seg_start = 0usize;
    let mut flush = |a: usize, b: usize, pts: &mut Vec<(f64, [f32; 4])>| {
        if b - a < 3 {
            return;
        }
        let mut tw = 0.0f64;
        let mut wsum = 0.0f64;
        let mut u4 = [0.0f32; 4];
        for t in a..b {
            let w = seq[t].2;
            tw += seq[t].0 as f64 * w;
            wsum += w;
            for i in 0..4 {
                u4[i] += seq[t].1[i];
            }
        }
        if wsum > 0.0 {
            pts.push((tw / wsum, u4));
        }
    };
    for k in 1..seq.len() {
        if (seq[k].0 - seq[k - 1].0) as f64 > thr {
            flush(seg_start, k, &mut pts);
            seg_start = k;
        }
    }
    flush(seg_start, seq.len(), &mut pts);
    bucketize(pts)
}

/// 一维 Otsu：在直方图上找让两类类间方差最大的切点。
fn otsu_threshold(x: &[f64]) -> f64 {
    let (lo, hi) = (
        x.iter().cloned().fold(f64::INFINITY, f64::min),
        x.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
    );
    if !(hi > lo) {
        return hi;
    }
    const BINS: usize = 64;
    let mut hist = [0.0f64; BINS];
    let scale = BINS as f64 / (hi - lo);
    for &v in x {
        let i = (((v - lo) * scale) as usize).min(BINS - 1);
        hist[i] += 1.0;
    }
    let total: f64 = hist.iter().sum();
    let mut sum_all = 0.0;
    for (i, &h) in hist.iter().enumerate() {
        sum_all += h * i as f64;
    }
    let (mut w0, mut sum0, mut best, mut cut) = (0.0, 0.0, -1.0, 0usize);
    for i in 0..BINS - 1 {
        w0 += hist[i];
        sum0 += hist[i] * i as f64;
        let w1 = total - w0;
        if w0 <= 0.0 || w1 <= 0.0 {
            continue;
        }
        let m0 = sum0 / w0;
        let m1 = (sum_all - sum0) / w1;
        let between = w0 * w1 * (m0 - m1) * (m0 - m1);
        if between > best {
            best = between;
            cut = i;
        }
    }
    lo + (cut as f64 + 0.5) / scale
}

/// 用一套桶单元跑 15 条链，产出 653 × 15 列。`prefix` 用于区分载体。
#[allow(clippy::too_many_arguments)]
fn run_chains(
    cells: &[Vec<(u32, [f32; 4])>],
    n: usize,
    tau: f32,
    w: PmWeight,
    dec: &[usize],
    industry_opt: Option<&[i32]>,
    lnt_opt: Option<&[f64]>,
    ks: &[usize],
    sig: SoftCfg,
    null_mc: usize,
    exz_bins: usize,
    prefix: &str,
    chains: Option<&[String]>,
    existing_m4: Option<&[Vec<f32>; 4]>,
) -> io::Result<(Vec<String>, Vec<Vec<f32>>)> {
    set_soft(sig);
    let has_lnt = lnt_opt.map_or(false, |x| x.iter().any(|v| v.is_finite()));
    let t_m4 = std::time::Instant::now();
    let computed_m4;
    let m4 = if let Some(m4) = existing_m4 {
        m4
    } else {
        computed_m4 = compute_split_mats_simd(cells, tau, w);
        &computed_m4
    };
    let m4_s = t_m4.elapsed().as_secs_f64();

    let dirs = [D_ALL, D_P, D_M];
    let mut rho_cache: Vec<Vec<f32>> = Vec::with_capacity(9);
    for &a in dirs.iter() {
        for &b in dirs.iter() {
            rho_cache.push(rho_ab(&m4, n, a, b));
        }
    }
    let ridx = |a: usize, b: usize| -> usize {
        dirs.iter().position(|&x| x == a).unwrap_or(0) * 3
            + dirs.iter().position(|&x| x == b).unwrap_or(0)
    };
    let rho_s = t_m4.elapsed().as_secs_f64() - m4_s;

    type ChainOut = io::Result<(Vec<String>, Vec<Vec<f32>>)>;
    let mut jobs: Vec<(usize, usize, usize, String)> = Vec::new();
    for &(a, b, tag) in BC_COMBOS.iter() {
        for &c_dir in X_DIRS.iter() {
            let short = format!("{tag}{}", dir_tag(c_dir));
            // chains 给了就只跑列出的那些（写 "AAa" 或 "spAAa" 都认）
            if let Some(list) = chains {
                if !list.iter().any(|x| x == &short || x == &format!("sp{short}")) {
                    continue;
                }
            }
            let label = format!("{prefix}sp{short}");
            jobs.push((a, b, c_dir, label));
        }
    }
    if jobs.is_empty() {
        return Err(io::Error::new(io::ErrorKind::InvalidInput, "chains 过滤后没有可跑的链"));
    }
    if std::env::var("YUPEI_STAGE").is_ok() {
        eprintln!("[stage] prefix={prefix} m4 {m4_s:.1}s rho {rho_s:.1}s chains {}", jobs.len());
    }

    let results: Vec<ChainOut> = jobs
        .par_iter()
        .map(|(a, b, c_dir, label)| {
            let (a, b, c_dir) = (*a, *b, *c_dir);
            let rbx = &rho_cache[ridx(a, c_dir)];
            let rcx = &rho_cache[ridx(b, c_dir)];
            let rbc = &rho_cache[ridx(a, b)];
            let e_a = energy(&m4, n, a);
            let e_b = energy(&m4, n, b);

            let mut rho_sym = rbc.clone();
            for i in 0..n {
                for j in 0..i {
                    rho_sym[i * n + j] = rho_sym[j * n + i];
                }
            }
            let sdir = crate::yupei_pair::split::fake_sdir(&e_a, &rho_sym, n);
            let mask: Vec<bool> = e_a.iter().map(|&v| v > 0.0).collect();
            let stock_idx: Vec<usize> = (0..n).filter(|&i| mask[i]).collect();
            let sp = SplitRho {
                b_row: rbx.as_slice(),
                c_row: rcx.as_slice(),
                bc: rbc.as_slice(),
            };
            let tn = std::time::Instant::now();
            let mut net = net_one_mat(
                &sdir,
                n,
                dec,
                &mask,
                &stock_idx,
                null_mc,
                exz_bins,
                label,
                Some(ks),
                Some((&sp, &e_a, &e_b)),
            );
            let net_s = tn.elapsed().as_secs_f64();
            let fails = check_net(&net, n, &mask, label);
            if fails > 0 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("check 失败 {fails} 项（chain={label}）"),
                ));
            }
            let tk_total = net.topk_total.take();
            let ms = matset_from_net(net, n);
            let mut out = Out::new();
            let pre = format!("{label}_");
            let tp = std::time::Instant::now();
            produce_one_mat(&ms, &pre, n, &mask, industry_opt, lnt_opt, has_lnt, &mut out);
            let produce_s = tp.elapsed().as_secs_f64();
            let tk_s0 = std::time::Instant::now();
            let (tn, tc) = match tk_total {
                Some(t) => finalize_total(&t, n, &mask, ks, &pre),
                None => topk_one_mat(&ms.rho, &ms.e, n, &mask, ks, &pre),
            };
            let mut names = out.names;
            names.extend(tn);
            let mut cols: Vec<Vec<f32>> = out
                .cols
                .into_iter()
                .map(|c| c.into_iter().map(|v| v as f32).collect())
                .collect();
            cols.extend(tc);
            if std::env::var("YUPEI_STAGE").is_ok() {
                eprintln!(
                    "[stage] {} net {:.1}s produce {:.1}s topk {:.1}s",
                    label,
                    net_s,
                    produce_s,
                    tk_s0.elapsed().as_secs_f64()
                );
            }
            Ok((names, cols))
        })
        .collect();

    let mut names = Vec::new();
    let mut cols = Vec::new();
    for r in results {
        let (ns, cs) = r?;
        names.extend(ns);
        cols.extend(cs);
    }
    Ok((names, cols))
}

/// 领先度：在相关矩阵层把「相加」换成「相减」，输出每只股票的行统计。
/// 这是旧版完全没有的角度，两个方向相减之后才有「谁更常先动」的符号。
fn lead_factors(m4: &[Vec<f32>; 4], n: usize) -> (Vec<String>, Vec<Vec<f32>>) {
    let all = {
        let mut v = vec![0.0f32; n * n];
        for t in [PP, PM, MP, MM] {
            for k in 0..n * n {
                v[k] += m4[t][k];
            }
        }
        v
    };
    let e = energy(&m4, n, D_ALL);
    let mask: Vec<bool> = e.iter().map(|&x| x > 0.0).collect();
    let rho = rho_ab(&m4, n, D_ALL, D_ALL);
    // 非对称部分：rho_ab - rho_ba，用原始交互矩阵重算
    let mut lead = vec![0.0f32; n * n];
    lead.par_chunks_mut(n).enumerate().for_each(|(i, row)| {
        let si = e[i].sqrt();
        if !(si > 0.0) {
            return;
        }
        for j in 0..n {
            if i == j {
                continue;
            }
            let sj = e[j].sqrt();
            if !(sj > 0.0) {
                continue;
            }
            let num = all[i * n + j] as f64 - all[j * n + i] as f64;
            let den = all[i * n + j] as f64 + all[j * n + i] as f64;
            row[j] = if den.abs() > 1e-12 {
                (num / den) as f32
            } else {
                0.0
            };
        }
    });
    let _ = rho;
    let stat_names = ["mean", "std", "skew", "max", "p10", "p90"];
    let stats: Vec<[f32; 6]> = (0..n)
        .into_par_iter()
        .map(|i| {
            if !mask[i] {
                return [f32::NAN; 6];
            }
            let mut v: Vec<f32> = (0..n).filter(|&j| j != i && mask[j]).map(|j| lead[i * n + j]).collect();
            if v.is_empty() {
                return [f32::NAN; 6];
            }
            let k = v.len();
            let mean = v.iter().map(|&x| x as f64).sum::<f64>() / k as f64;
            let (mut m2, mut m3) = (0.0f64, 0.0f64);
            for &x in v.iter() {
                let d = x as f64 - mean;
                m2 += d * d;
                m3 += d * d * d;
            }
            let var = m2 / k as f64;
            let sd = var.sqrt();
            let skew = if sd > 1e-12 { m3 / k as f64 / (var * sd) } else { 0.0 };
            v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let q = |p: f64| v[(((k - 1) as f64) * p).round() as usize];
            [
                mean as f32,
                sd as f32,
                skew as f32,
                v[k - 1],
                q(0.10),
                q(0.90),
            ]
        })
        .collect();
    let mut names = Vec::new();
    let mut cols = Vec::new();
    for (c, nm) in stat_names.iter().enumerate() {
        names.push(format!("ld_AA_{nm}"));
        cols.push((0..n).map(|i| stats[i][c]).collect());
    }
    (names, cols)
}

/// 横截面分解：不做配对，把每只股票的成交强度曲线裁成矩阵做截断奇异值分解，
/// 取前 `k` 个成分的载荷。它与配对框架正交，出的是少量互相正交的公共节奏因子。
fn svd_factors(
    cells: &[Vec<(u32, [f32; 4])>],
    n: usize,
    k: usize,
) -> (Vec<String>, Vec<Vec<f32>>) {
    const STEP: usize = 100; // 1 秒一格（10ms × 100）
    let nb = N_BUCKETS / STEP;
    let mut a = vec![0.0f64; n * nb];
    a.par_chunks_mut(nb).enumerate().for_each(|(i, row)| {
        for &(b, u) in cells[i].iter() {
            row[(b as usize / STEP).min(nb - 1)] += (u[0] + u[1]) as f64;
        }
    });
    // 每只股票标准化，让分解看节奏而不是看成交量
    for i in 0..n {
        let row = &mut a[i * nb..(i + 1) * nb];
        let mean = row.iter().sum::<f64>() / nb as f64;
        let var = row.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / nb as f64;
        let sd = var.sqrt();
        if sd > 1e-12 {
            for x in row.iter_mut() {
                *x = (*x - mean) / sd;
            }
        } else {
            for x in row.iter_mut() {
                *x = 0.0;
            }
        }
    }
    // 幂迭代求前 k 个右奇异向量的载荷（n × nb 乘 nb × n 的协方差太大，直接在
    // 数据矩阵上做 Jacobi 式正交迭代）
    let mut basis: Vec<Vec<f64>> = Vec::with_capacity(k);
    let mut load: Vec<Vec<f32>> = Vec::with_capacity(k);
    for _ in 0..k {
        let mut v: Vec<f64> = (0..nb).map(|j| 1.0 + ((j * 37) % 97) as f64 * 1e-3).collect();
        for _ in 0..60 {
            // u = A v
            let mut urow = vec![0.0f64; n];
            urow.par_iter_mut().enumerate().for_each(|(i, o)| {
                let row = &a[i * nb..(i + 1) * nb];
                *o = row.iter().zip(v.iter()).map(|(x, y)| x * y).sum();
            });
            // v = A^T u
            let mut vnew = vec![0.0f64; nb];
            // 每列独立并行；列内仍按原来的股票顺序累加，保持浮点结果逐位一致。
            const SVD_COLUMN_CHUNK: usize = 256;
            vnew.par_chunks_mut(SVD_COLUMN_CHUNK).enumerate().for_each(|(block, out)| {
                let j0 = block * SVD_COLUMN_CHUNK;
                for i in 0..n {
                    let row = &a[i * nb + j0..i * nb + j0 + out.len()];
                    let ui = urow[i];
                    for (dst, &x) in out.iter_mut().zip(row.iter()) {
                        *dst += x * ui;
                    }
                }
            });
            // 对已有基去正交
            for bv in basis.iter() {
                let dot: f64 = vnew.iter().zip(bv.iter()).map(|(x, y)| x * y).sum();
                for j in 0..nb {
                    vnew[j] -= dot * bv[j];
                }
            }
            let norm = vnew.iter().map(|x| x * x).sum::<f64>().sqrt();
            if norm < 1e-12 {
                break;
            }
            for j in 0..nb {
                vnew[j] /= norm;
            }
            v = vnew;
        }
        let mut urow = vec![0.0f64; n];
        urow.par_iter_mut().enumerate().for_each(|(i, o)| {
            let row = &a[i * nb..(i + 1) * nb];
            *o = row.iter().zip(v.iter()).map(|(x, y)| x * y).sum();
        });
        load.push(urow.iter().map(|&x| x as f32).collect());
        basis.push(v);
    }
    let mut names = Vec::new();
    for i in 0..k {
        names.push(format!("sv_comp{}", i + 1));
    }
    (names, load)
}

/// 五类载体一起算。返回 (codes, names, vals)。
#[allow(clippy::too_many_arguments)]
pub fn compute_multi_full(
    date: i64,
    tau: f64,
    weight: &str,
    ks: &[usize],
    sig: SoftCfg,
    null_mc: usize,
    exz_bins: usize,
    svd_k: usize,
    chains: Option<&[String]>,
) -> io::Result<(Vec<String>, Vec<String>, Vec<f32>)> {
    let total_started = std::time::Instant::now();
    let stage_enabled = std::env::var("YUPEI_STAGE").is_ok();
    let stage = |name: &str, started: std::time::Instant| {
        if stage_enabled {
            eprintln!("[multi-stage] name={name} seconds={:.6}", started.elapsed().as_secs_f64());
        }
    };
    let read_started = std::time::Instant::now();
    let codes = crate::data_paths::list_stock_codes(date, "transaction");
    if codes.is_empty() {
        return Err(io::Error::new(io::ErrorKind::Other, "当日无股票"));
    }
    let day_start = crate::yupei_dist::compute::day_start_us_of(date);
    let w = match weight {
        "cnt" => PmWeight::Cnt,
        "vol" => PmWeight::Vol,
        other => {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("weight 只支持 cnt / vol，收到 {other}"),
            ))
        }
    };

    // ---- 1. 逐笔只读一次，三套桶单元共用 ----
    let raw: Vec<Option<Vec<crate::fast_csv_reader::TradeRecord>>> = codes
        .par_iter()
        .map(|c| {
            crate::fast_csv_reader::read_trade_light_inner(c, date).ok()
        })
        .collect();
    let n = codes.len();
    stage("read", read_started);
    if stage_enabled {
        eprintln!("[multi-stage] date={date} stocks={n} thread_count={}", rayon::current_num_threads());
    }
    let industry_started = std::time::Instant::now();
    let n_trades: Vec<f64> = raw
        .iter()
        .map(|r| r.as_ref().map_or(f64::NAN, |v| v.len() as f64))
        .collect();

    // 上游协变量：活跃度分层 + ln(笔数) + 行业
    let mut sorted_nt = n_trades.clone();
    sorted_nt.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let cuts: Vec<f64> = (1..10)
        .map(|k| sorted_nt[(k * n / 10).min(n - 1)])
        .collect();
    let dec: Vec<usize> = n_trades.iter().map(|&v| decile_of(v, &cuts)).collect();
    let lnt: Vec<f64> = n_trades
        .iter()
        .map(|&v| if v.is_nan() { f64::NAN } else { v.max(1.0).ln() })
        .collect();
    let industry = crate::yupei_dist::industry::industry_of(date, &codes)?;
    let industry_i32: Vec<i32> = industry.iter().map(|&v| v as i32).collect();
    stage("industry", industry_started);
    let cells_started = std::time::Instant::now();

    // ---- 2. 三套桶单元（并行生成）----
    // 四套桶单元合并成一次遍历（原来四套各扫一遍全量记录）
    let built: Vec<(
        Vec<(u32, [f32; 4])>,
        Vec<(u32, [f32; 4])>,
        Vec<(u32, [f32; 4])>,
        Vec<(u32, [f32; 4])>,
    )> = raw
        .par_iter()
        .map(|r| match r {
            Some(v) => build_all_cells(v, day_start),
            None => (Vec::new(), Vec::new(), Vec::new(), Vec::new()),
        })
        .collect();
    let mut cells_base: Vec<Vec<(u32, [f32; 4])>> = Vec::with_capacity(n);
    let mut cells_order: Vec<Vec<(u32, [f32; 4])>> = Vec::with_capacity(n);
    let mut cells_otsu: Vec<Vec<(u32, [f32; 4])>> = Vec::with_capacity(n);
    let mut cells_side: Vec<Vec<(u32, [f32; 4])>> = Vec::with_capacity(n);
    for (a, b, c, d) in built {
        cells_base.push(a);
        cells_order.push(b);
        cells_otsu.push(c);
        cells_side.push(d);
    }
    drop(raw);
    stage("cells", cells_started);

    // ---- 3. 每套跑 15 条链 ----
    let mut names: Vec<String> = Vec::new();
    let mut cols: Vec<Vec<f32>> = Vec::new();
    let mut base_m4: Option<[Vec<f32>; 4]> = None;
    let mut order_m4: Option<[Vec<f32>; 4]> = None;
    for (mode, cells) in [
        (CellMode::Base, &cells_base),
        (CellMode::Order, &cells_order),
        (CellMode::Otsu, &cells_otsu),
        (CellMode::OrderSide, &cells_side),
    ] {
        let carrier_started = std::time::Instant::now();
        let m4_started = std::time::Instant::now();
        let mut other_m4: Option<[Vec<f32>; 4]> = None;
        match mode {
            // 基础矩阵在四套载体计算后继续用于领先度，避免重算。
            CellMode::Base => base_m4 = Some(compute_split_mats_simd(cells, tau as f32, w)),
            CellMode::Order => order_m4 = Some(compute_split_mats_simd(cells, tau as f32, w)),
            CellMode::Otsu => {
                // build_all_cells 的第三项实际是订单方向桶；保持现有 oc_/os_ 标签语义。
                // 交换 P/M 通道只需移动矩阵所有权，不复制数据，也不改变累加顺序。
                let [pp, pm, mp, mm] = order_m4.take().expect("order matrix computed");
                other_m4 = Some([mm, mp, pm, pp]);
            }
            CellMode::OrderSide => other_m4 = Some(compute_split_mats_simd(cells, tau as f32, w)),
        }
        stage(&format!("m4_prepare_{}", mode.tag()), m4_started);
        let cached_m4 = match mode {
            CellMode::Base => base_m4.as_ref(),
            CellMode::Order => order_m4.as_ref(),
            _ => other_m4.as_ref(),
        };
        let (ns, cs) = run_chains(
            cells,
            n,
            tau as f32,
            w,
            &dec,
            Some(&industry_i32),
            Some(&lnt),
            ks,
            sig,
            null_mc,
            exz_bins,
            mode.tag(),
            chains,
            cached_m4,
        )?;
        names.extend(ns);
        cols.extend(cs);
        stage(&format!("carrier_{}", mode.tag()), carrier_started);
    }

    // ---- 4. 领先度（矩阵层相减）与横截面分解 ----
    let lead_started = std::time::Instant::now();
    let (ln, lc) = lead_factors(base_m4.as_ref().expect("base matrix computed"), n);
    names.extend(ln);
    cols.extend(lc);
    drop(base_m4);
    stage("lead", lead_started);
    let svd_started = std::time::Instant::now();
    let (sn, sc) = svd_factors(&cells_base, n, svd_k);
    names.extend(sn);
    cols.extend(sc);
    stage("svd", svd_started);
    let assemble_started = std::time::Instant::now();

    // ---- 5. 组装成行主序 ----
    let f = names.len();
    let mut vals = vec![f32::NAN; n * f];
    vals.par_chunks_mut(f).enumerate().for_each(|(i, row)| {
        for (j, c) in cols.iter().enumerate() {
            row[j] = c[i];
        }
    });
    stage("assemble", assemble_started);
    stage("total_compute", total_started);
    Ok((codes, names, vals))
}

/// Python 入口。
#[pyfunction]
#[pyo3(signature = (date, tau=1.0, weight="cnt", ks=None, svd_k=8, chains=None, sig_temp=0.3, sig_thr=2.0, null_mc=20000, exz_bins=10))]
pub fn py_yupei_multi(
    date: i64,
    tau: f64,
    weight: &str,
    ks: Option<Vec<usize>>,
    svd_k: usize,
    chains: Option<Vec<String>>,
    sig_temp: f64,
    sig_thr: f64,
    null_mc: usize,
    exz_bins: usize,
) -> PyResult<(Vec<String>, Vec<String>, Vec<f32>)> {
    compute_multi_full(
        date,
        tau,
        weight,
        &ks.unwrap_or_else(|| vec![5, 20, 100]),
        SoftCfg {
            temp: sig_temp,
            thr: sig_thr,
        },
        null_mc,
        exz_bins,
        svd_k,
        chains.as_deref(),
    )
    .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("{e:?}")))
}


/// 定稿入口：四套载体（原版、订单级、聚簇、订单方向级）+ 五种方向组合，
/// 默认输出 13074 列。相对通用入口只是把 chains 钉死，其余完全相同。
///
/// date: 交易日 YYYYMMDD；tau: 衰减时间尺度（秒），默认 1.0。
/// weight: "cnt" 按笔数，"vol" 按成交量平方根，默认 "cnt"。
/// ks: 附加 TopK 因子的档位列表，默认 [5, 20, 100]；可传 [5, 20, 30]。
/// svd_k: 横截面分解成分数，默认 8；sig_temp / sig_thr: 软阈值参数，默认 0.3 / 2.0。
/// null_mc: 空模型抽样上限，默认 20000；exz_bins: 条件网格边数，默认 10。
/// 返回 (股票代码, 因子列名, 按股票逐行展开的因子值)。线程数由 RAYON_NUM_THREADS 控制。
#[pyfunction]
#[pyo3(signature = (date, tau=1.0, weight="cnt", ks=None, svd_k=8, sig_temp=0.3, sig_thr=2.0, null_mc=20000, exz_bins=10))]
pub fn py_yupei_multi_v5(
    date: i64,
    tau: f64,
    weight: &str,
    ks: Option<Vec<usize>>,
    svd_k: usize,
    sig_temp: f64,
    sig_thr: f64,
    null_mc: usize,
    exz_bins: usize,
) -> PyResult<(Vec<String>, Vec<String>, Vec<f32>)> {
    let chains: Vec<String> = V5_CHAINS.iter().map(|s| s.to_string()).collect();
    compute_multi_full(
        date,
        tau,
        weight,
        &ks.unwrap_or_else(|| vec![5, 20, 100]),
        SoftCfg {
            temp: sig_temp,
            thr: sig_thr,
        },
        null_mc,
        exz_bins,
        svd_k,
        Some(&chains),
    )
    .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("{e:?}")))
}


/// 基准：同一天、同一批桶单元，标量版与 SIMD 版各跑一遍四分量矩阵。
/// 返回 (标量耗时秒, SIMD 耗时秒, 两版最大绝对差)。
#[pyfunction]
#[pyo3(signature = (date, weight="cnt", tau=1.0))]
pub fn py_bench_split_mats(date: i64, weight: &str, tau: f64) -> PyResult<(f64, f64, f32)> {
    let codes = crate::data_paths::list_stock_codes(date, "transaction");
    if codes.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err("当日无股票"));
    }
    let day_start = crate::yupei_dist::compute::day_start_us_of(date);
    let w = match weight {
        "cnt" => PmWeight::Cnt,
        "vol" => PmWeight::Vol,
        other => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "weight 只支持 cnt / vol，收到 {other}"
            )))
        }
    };
    let raw: Vec<Option<Vec<crate::fast_csv_reader::TradeRecord>>> = codes
        .par_iter()
        .map(|c| crate::fast_csv_reader::read_trade_light_inner(c, date).ok())
        .collect();
    let cells: Vec<Vec<(u32, [f32; 4])>> = raw
        .par_iter()
        .map(|r| {
            r.as_ref()
                .map_or_else(Vec::new, |v| build_cells(v, day_start, CellMode::Base))
        })
        .collect();
    let t0 = std::time::Instant::now();
    let a = compute_split_mats_from_cells(&cells, tau as f32, w);
    let t1 = std::time::Instant::now();
    let b = compute_split_mats_simd(&cells, tau as f32, w);
    let t2 = std::time::Instant::now();
    let mut worst = 0.0f32;
    for k in 0..4 {
        for x in 0..a[k].len() {
            let d = (a[k][x] - b[k][x]).abs();
            if d > worst {
                worst = d;
            }
        }
    }
    Ok(((t1 - t0).as_secs_f64(), (t2 - t1).as_secs_f64(), worst))
}
