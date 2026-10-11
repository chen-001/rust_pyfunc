//! 玉佩2 配对互补网络 —— reduce 扩展块（legacy 77 列之外的新因子）。
//!
//! 设计原则（用户 2026-10-10 六点要求里的 2/3/4/5）：
//!   * 离散 → 连续：硬阈值计数一律配软版本（sigmoid/relu 权重、log1p、百分位）；
//!   * max → 稳健：argmax/最大值一律配 top-K 均值、分位数、截尾统计；
//!   * 每个角度展开：行统计 16 种 × 21 张矩阵、贪心篮子 16 步全留、图多 k 多核、
//!     残差多协变量多被解释矩阵、平面多组对照；
//!   * 覆盖度：新增 e_/s_ 软质量族（MASS 向量）天然 100% 覆盖，替代只有 25% 覆盖的
//!     r1_explain_* 硬选中族。
//!
//! 命名规则：列名 = {pre} + 块名，{pre} 在多矩阵时是 "cnt_t1_" 这类前缀。

use rayon::prelude::*;
use std::time::Instant;

use crate::reduce::{spearman, topk_adj_multi, MatSet, Out};

// ============================================================
// A. 行统计：每股在一张 N×N 分数矩阵上的配对分布形状
// ============================================================

/// 16 种行统计的名字（顺序与 row_stats16 返回值一致）。
pub(crate) const ROW_STAT_NAMES: [&str; 16] = [
    "mean", "std", "med", "p10", "p25", "p75", "p90", "p95", "max", "t5m", "t10m",
    "t20m", "t50m", "possh", "hhip", "skew",
];

/// 对一行（一只股票的全部配对分数，去掉自身那一格与非个股配对）算 16 个统计量。
/// 返回 (统计名, 值) 的定长数组；无有限值时全 NaN。
fn row_stats16(row: &[f32], i: usize, n: usize, ok: &[bool]) -> [f64; 16] {
    let mut v: Vec<f64> = Vec::with_capacity(n);
    for j in 0..n {
        if j == i || !ok[j] {
            continue;
        }
        let x = row[j] as f64;
        if x.is_finite() {
            v.push(x);
        }
    }
    let m = v.len();
    if m == 0 {
        return [f64::NAN; 16];
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let sum: f64 = v.iter().sum();
    let mean = sum / m as f64;
    let var = v.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / m as f64;
    let std = var.sqrt();
    let med = if m % 2 == 1 {
        v[m / 2]
    } else {
        (v[m / 2 - 1] + v[m / 2]) / 2.0
    };
    // 分位点：线性索引 p*(m-1)
    let pct = |p: f64| -> f64 {
        let idx = (p * (m - 1) as f64).round() as usize;
        v[idx.min(m - 1)]
    };
    let topk = |k: usize| -> f64 {
        let k = k.min(m);
        v[m - k..].iter().sum::<f64>() / k as f64
    };
    let possh = v.iter().filter(|&&x| x > 0.0).count() as f64 / m as f64;
    let sp: f64 = v.iter().copied().filter(|x| *x > 0.0).sum();
    let hhip = if sp > 0.0 {
        v.iter()
            .filter(|&&x| x > 0.0)
            .map(|x| {
                let w = x / sp;
                w * w
            })
            .sum()
    } else {
        f64::NAN
    };
    let skew = if std > 1e-12 && m > 2 {
        v.iter().map(|x| ((x - mean) / std).powi(3)).sum::<f64>() / m as f64
    } else {
        f64::NAN
    };
    [
        mean,
        std,
        med,
        pct(0.10),
        pct(0.25),
        pct(0.75),
        pct(0.90),
        pct(0.95),
        v[m - 1],
        topk(5),
        topk(10),
        topk(20),
        topk(50),
        possh,
        hhip,
        skew,
    ]
}

/// 矩阵清单：名字 + 引用。行统计对每张各出 16 列，命名 row_{MAT}_{stat}。
fn mat_catalog(ms: &MatSet) -> Vec<(&'static str, &Vec<f32>)> {
    vec![
        ("rho", &ms.rho),
        ("g", &ms.g),
        ("z", &ms.z),
        ("gs", &ms.gs),
        ("zs", &ms.zs),
        ("sc", &ms.sc),
        ("grel", &ms.grel),
        ("gsh", &ms.gsh),
        ("gcov", &ms.gcov),
        ("gnpos", &ms.gnpos),
        ("ghhi", &ms.ghhi),
        ("exz", &ms.exz),
        ("exzs", &ms.exzs),
        ("ru", &ms.ru),
        ("sgl", &ms.sg),
        ("rus", &ms.rus),
        ("sgs", &ms.sgs),
        ("pb", &ms.pb),
        ("qc", &ms.qc),
        ("pbs", &ms.pbs),
        ("qcs", &ms.qcs),
    ]
}

fn block_row_stats(ms: &MatSet, pre: &str, n: usize, ok: &[bool], out: &mut Out) {
    let t0 = Instant::now();
    for (mn, m) in mat_catalog(ms) {
        // 先并行算全部行的 16 统计，再逐统计装配列（按统计名循环，列顺序确定）
        let rows: Vec<[f64; 16]> = (0..n)
            .into_par_iter()
            .map(|i| {
                if !ok[i] {
                    return [f64::NAN; 16];
                }
                row_stats16(&m[i * n..i * n + n], i, n, ok)
            })
            .collect();
        for (si, sn) in ROW_STAT_NAMES.iter().enumerate() {
            out.push(
                &format!("{pre}row_{mn}_{sn}"),
                rows.iter().map(|r| r[si]).collect(),
            );
        }
    }
    eprintln!("[reduce2:A] 行统计 21×16 {:.1}s", t0.elapsed().as_secs_f32());
}

// ============================================================
// B. explain 族扩展：软质量 + 软阈值计数（coverage 修复 + 连续化）
// ============================================================

#[inline]
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// 被选为 X0/X0* 的配对的扩展统计。
/// 返回 (cnt_log1p, cnt_share, zsoft, w_p90, w_t5m)。
fn explain_extra(
    n: usize,
    x0sel: &[i32],
    w: &[f32],
    z: &[f32],
    ok: &[bool],
) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    // 每只入选股收一份 (w,z) 列表（全部配对约 680 万条 ×8B，内存可接受）。
    // 行块并行收集再按块序合并；表内顺序对下游统计（排序/求和）无影响，结果确定。
    const RB3: usize = 256;
    let nrb = n.div_ceil(RB3);
    let parts: Vec<(Vec<Vec<(f32, f32)>>, usize)> = (0..nrb)
        .into_par_iter()
        .map(|rb| {
            let mut sel: Vec<Vec<(f32, f32)>> = vec![Vec::new(); n];
            let mut tot = 0usize;
            for i in (rb * RB3)..((rb + 1) * RB3).min(n) {
                if !ok[i] {
                    continue;
                }
                for j in (i + 1)..n {
                    if !ok[j] {
                        continue;
                    }
                    let x = x0sel[i * n + j];
                    if x < 0 || x as usize >= n {
                        continue;
                    }
                    sel[x as usize].push((w[i * n + j], z[i * n + j]));
                    tot += 1;
                }
            }
            (sel, tot)
        })
        .collect();
    let mut sel: Vec<Vec<(f32, f32)>> = vec![Vec::new(); n];
    let mut total = 0usize;
    for (ps, t) in parts {
        total += t;
        for x in 0..n {
            if !ps[x].is_empty() {
                sel[x].extend_from_slice(&ps[x]);
            }
        }
    }
    let inv = if total > 0 { 1.0 / total as f64 } else { 0.0 };
    let mut log1p = vec![f64::NAN; n];
    let mut share = vec![f64::NAN; n];
    let mut zsoft = vec![f64::NAN; n];
    let mut wp90 = vec![f64::NAN; n];
    let mut wt5 = vec![f64::NAN; n];
    for x in 0..n {
        if !ok[x] {
            continue;
        }
        let s = &sel[x];
        if s.is_empty() {
            // 没被选中过：计数型给 0（连续语义下 0 是真实值，不是缺失）
            log1p[x] = 0.0;
            share[x] = 0.0;
            zsoft[x] = 0.0;
            continue;
        }
        log1p[x] = (s.len() as f64).ln_1p();
        share[x] = s.len() as f64 * inv;
        let sc = crate::reduce::soft();
        zsoft[x] = s
            .iter()
            .map(|&(_, zz)| sigmoid((zz as f64 - sc.thr) / sc.temp))
            .sum();
        let mut ws: Vec<f64> = s.iter().map(|&(a, _)| a as f64).collect();
        ws.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let m = ws.len();
        wp90[x] = ws[((0.90 * (m - 1) as f64).round() as usize).min(m - 1)];
        let k = 5.min(m);
        wt5[x] = ws[m - k..].iter().sum::<f64>() / k as f64;
    }
    (log1p, share, zsoft, wp90, wt5)
}

/// 作为配对成员（B/C 任一）的 Z 软质量：Σ_j max(z-thr,0) 与 Σ_j sigmoid((z-thr)/temp)。
fn partner_z_extra(n: usize, z: &[f32], ok: &[bool]) -> (Vec<f64>, Vec<f64>) {
    let mut mass = vec![f64::NAN; n];
    let mut soft = vec![f64::NAN; n];
    let sc = crate::reduce::soft();
    mass.par_iter_mut()
        .zip(soft.par_iter_mut())
        .enumerate()
        .for_each(|(i, (mm, ss))| {
            if !ok[i] {
                return;
            }
            let mut a = 0.0f64;
            let mut b = 0.0f64;
            for j in 0..n {
                if j == i {
                    continue;
                }
                let v = z[i * n + j] as f64;
                if v.is_finite() {
                    if v > sc.thr {
                        a += v - sc.thr;
                    }
                    b += sigmoid((v - sc.thr) / sc.temp);
                }
            }
            *mm = a;
            *ss = b;
        });
    (mass, soft)
}

fn block_explain(ms: &MatSet, pre: &str, n: usize, ok: &[bool], out: &mut Out) {
    let t0 = Instant::now();
    // X0 路（e_）与 X0* 路（es_）
    for (tag, x0sel, w, z) in [
        ("e", &ms.x0, &ms.g, &ms.z),
        ("es", &ms.x0s, &ms.gs, &ms.zs),
    ] {
        let (lg, sh, zs_, wp, wt) = explain_extra(n, x0sel, w, z, ok);
        out.push(&format!("{pre}{tag}_cnt_log1p"), lg);
        out.push(&format!("{pre}{tag}_cnt_share"), sh);
        out.push(&format!("{pre}{tag}_zsoft"), zs_);
        out.push(&format!("{pre}{tag}_w_p90"), wp);
        out.push(&format!("{pre}{tag}_w_t5m"), wt);
    }
    // 软质量向量（net 落盘的 MASS，6×n 布局）：emass|gmass|g2mass|smass|sgmass|sg2mass
    let med_of = |sl: &[f64]| -> f64 {
        let mut v: Vec<f64> = sl.iter().copied().filter(|x| x.is_finite()).collect();
        if v.is_empty() {
            return f64::NAN;
        }
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[v.len() / 2]
    };
    for (tag, k) in [("e", 0usize), ("s", 3usize)] {
        let em: Vec<f64> = (0..n).map(|i| ms.mass[k * n + i] as f64).collect();
        let gm: Vec<f64> = (0..n).map(|i| ms.mass[(k + 1) * n + i] as f64).collect();
        let g2: Vec<f64> = (0..n).map(|i| ms.mass[(k + 2) * n + i] as f64).collect();
        let me = med_of(&em);
        let mg = med_of(&gm);
        out.push(&format!("{pre}{tag}_mass"), em.clone());
        out.push(
            &format!("{pre}{tag}_mass_rel"),
            em.iter()
                .map(|&v| if me.is_finite() && me > 0.0 { v / me } else { f64::NAN })
                .collect(),
        );
        out.push(&format!("{pre}{tag}_gmass"), gm.clone());
        out.push(
            &format!("{pre}{tag}_gmass_rel"),
            gm.iter()
                .map(|&v| if mg.is_finite() && mg != 0.0 { v / mg } else { f64::NAN })
                .collect(),
        );
        // 无条件软版本（覆盖率修复）：责任加权平均增益 = gmass/emass，
        // 责任加权标准差 = sqrt(g2mass/emass - mean²)。emass>0 的股票全覆盖。
        let wmean: Vec<f64> = (0..n)
            .map(|i| {
                if ok[i] && em[i] > 0.0 {
                    gm[i] / em[i]
                } else {
                    f64::NAN
                }
            })
            .collect();
        let wstd: Vec<f64> = (0..n)
            .map(|i| {
                if ok[i] && em[i] > 0.0 && g2[i].is_finite() {
                    (g2[i] / em[i] - wmean[i] * wmean[i]).max(0.0).sqrt()
                } else {
                    f64::NAN
                }
            })
            .collect();
        out.push(&format!("{pre}{tag}_w_mean_soft"), wmean);
        out.push(&format!("{pre}{tag}_w_std_soft"), wstd);
    }
    // 作为配对成员的 Z 软质量
    let (m1, s1) = partner_z_extra(n, &ms.z, ok);
    out.push(&format!("{pre}pz_mass"), m1);
    out.push(&format!("{pre}pz_soft"), s1);
    let (m2, s2) = partner_z_extra(n, &ms.zs, ok);
    out.push(&format!("{pre}psz_mass"), m2);
    out.push(&format!("{pre}psz_soft"), s2);
    eprintln!("[reduce2:B] explain 扩展 {:.1}s", t0.elapsed().as_secs_f32());
}

// ============================================================
// C. bestcomp 前 5 档剖面（argmax 的稳健化）
// ============================================================

/// 每只股票沿行取前 5 搭档的值 + 若干导出量。
/// 返回 (top1..top5, mean3, mean5, margin12, mutual_pct, z_at_top1)。
fn bestcomp_profile(
    n: usize,
    w: &[f32],
    z: &[f32],
    ok: &[bool],
) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let rows: Vec<[f64; 10]> = (0..n)
        .into_par_iter()
        .map(|i| {
            let mut r = [f64::NAN; 10];
            if !ok[i] {
                return r;
            }
            // 行内前 5 名（值+下标）
            let mut top: Vec<(f32, usize)> = Vec::with_capacity(5);
            for j in 0..n {
                if j == i || !ok[j] {
                    continue;
                }
                let v = w[i * n + j];
                if !v.is_finite() {
                    continue;
                }
                if top.len() < 5 {
                    top.push((v, j));
                    top.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
                } else if v > top[4].0 {
                    top[4] = (v, j);
                    top.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
                }
            }
            if top.is_empty() {
                return r;
            }
            for k in 0..top.len().min(5) {
                r[k] = top[k].0 as f64;
            }
            let m = top.len();
            r[5] = top[..3.min(m)].iter().map(|t| t.0 as f64).sum::<f64>() / 3.0f64.min(m as f64);
            r[6] = top[..5.min(m)].iter().map(|t| t.0 as f64).sum::<f64>() / 5.0f64.min(m as f64);
            if m >= 2 {
                r[7] = top[0].0 as f64 - top[1].0 as f64; // margin12
            }
            // mutual_pct：我在最强搭档那一行里排第几的百分位（连续互惠度）
            let p = top[0].1;
            let g = top[0].0;
            let mut cnt = 0usize;
            let mut tot = 0usize;
            for j in 0..n {
                if j == p || !ok[j] {
                    continue;
                }
                let v = w[p * n + j];
                if v.is_finite() {
                    tot += 1;
                    if v > g {
                        cnt += 1;
                    }
                }
            }
            r[8] = if tot > 0 {
                1.0 - cnt as f64 / tot as f64
            } else {
                f64::NAN
            };
            r[9] = z[i * n + p] as f64;
            r
        })
        .collect();
    let col = |k: usize| rows.iter().map(|r| r[k]).collect::<Vec<f64>>();
    (col(0), col(1), col(2), col(3), col(4), col(5), col(6), col(7), col(8), col(9))
}

fn block_bestcomp(ms: &MatSet, pre: &str, n: usize, ok: &[bool], out: &mut Out) {
    let t0 = Instant::now();
    for (tag, w, z) in [("bc", &ms.g, &ms.z), ("bcs", &ms.gs, &ms.zs)] {
        let (t1, t2, t3, t4, t5, m3, m5, mg, mp, zt) = bestcomp_profile(n, w, z, ok);
        out.push(&format!("{pre}{tag}_top1"), t1);
        out.push(&format!("{pre}{tag}_top2"), t2);
        out.push(&format!("{pre}{tag}_top3"), t3);
        out.push(&format!("{pre}{tag}_top4"), t4);
        out.push(&format!("{pre}{tag}_top5"), t5);
        out.push(&format!("{pre}{tag}_mean3"), m3);
        out.push(&format!("{pre}{tag}_mean5"), m5);
        out.push(&format!("{pre}{tag}_margin12"), mg);
        out.push(&format!("{pre}{tag}_mutual_pct"), mp);
        out.push(&format!("{pre}{tag}_z_top1"), zt);
    }
    eprintln!("[reduce2:C] bestcomp 剖面 {:.1}s", t0.elapsed().as_secs_f32());
}

// ============================================================
// D. R2 扩展：每只 X 的前 50 配对（top-K 替代 argmax）+ 击败单只的配对数
// ============================================================

/// 与 pair_scan 同一公式，但每只 X 保留前 kmax 对 (v,b,c)，并数出 rhoU>best_single 的对数。
/// 精确性：全局 top-K 的成员必在其所在 B 块的块内 top-K 里，按块收再并集后取 top-K 是精确的。
/// 返回 (top: n×Vec<(v,b,c)>, nbeat: n)。
fn pair_scan_topk(
    n: usize,
    rho: &[f32],
    e: &[f64],
    single: &[f32],
    ok: &[bool],
    kmax: usize,
) -> (Vec<Vec<(f32, i32, i32)>>, Vec<i64>) {
    let t0 = Instant::now();
    let se: Vec<f32> = (0..n).map(|i| e[i].max(1e-30).sqrt() as f32).collect();
    let mut m = vec![0.0f32; n * n];
    m.par_chunks_mut(n).enumerate().for_each(|(x, row)| {
        for b in 0..n {
            row[b] = rho[x * n + b] * se[b];
        }
    });
    let mut wm = vec![0.0f32; n * n];
    wm.par_chunks_mut(n).enumerate().for_each(|(b, row)| {
        for c in 0..n {
            if b == c {
                continue;
            }
            let t = (2.0f32 * rho[b * n + c]) * se[b] * se[c];
            let den = (e[b] + e[c] + t as f64).max(1e-30).sqrt() as f32;
            row[c] = 1.0f32 / den;
        }
    });
    let mt = crate::reduce::transpose_f32(&m, n);
    eprintln!("[reduce2:D] 准备 M/W {:.1}s", t0.elapsed().as_secs_f32());

    const BLK: usize = 64;
    let nblk = n.div_ceil(BLK);
    let t0 = Instant::now();
    let blocks: Vec<(Vec<Vec<(f32, i32, i32)>>, Vec<i64>)> = (0..nblk)
        .into_par_iter()
        .map(|bk| {
            let b0 = bk * BLK;
            let b1 = (b0 + BLK).min(n);
            // 每 x 的块内 top-kmax：插入序存储 + 跟踪当前 min，内层只有一次比较；
            // 命中才插入/淘汰并重扫 50 元表。淘汰目标与旧版逐位一致——旧版列表
            // 全程降序排序、顶掉 tl[kmax-1]（稳定序里最靠尾的 min），等价于在
            // 插入序里淘汰"最后出现的最小值"：这里用 <= 取最后一个 min 再
            // remove+push，保持存活元素的插入顺序，块末再统一降序排序，
            // 与旧版成员集合及顺序完全相同。
            let mut top: Vec<Vec<(f32, i32, i32)>> = vec![Vec::new(); n];
            // tmin 初值 -inf：列表未满时任何有限值都能进；
            // 被 -inf 掩掉的 x（==b/c/非个股）-inf > -inf 为假，天然排除
            let mut tmin = vec![f32::NEG_INFINITY; n];
            let mut beat32 = vec![0i32; n];
            let mut hits: Vec<u32> = Vec::with_capacity(256);
            let mut a = vec![0.0f32; n];
            let mut vbuf = vec![0.0f32; n];
            for b in b0..b1 {
                if !ok[b] {
                    continue;
                }
                for x in 0..n {
                    a[x] = m[x * n + b];
                }
                a[b] = f32::NEG_INFINITY;
                for c in (b + 1)..n {
                    if !ok[c] {
                        continue;
                    }
                    let w = wm[b * n + c];
                    let row = &mt[c * n..c * n + n];
                    let saved = a[c];
                    a[c] = f32::NEG_INFINITY;
                    crate::simd::mul_add(&mut vbuf, &a, row, w);
                    // 击败计数：SIMD 掩码加（i32 足够：对数 < 2^31）
                    crate::simd::add_cnt_gt_i32(&mut beat32, &vbuf, &single);
                    // top-k 插入：movemask 命中位才进列表操作
                    hits.clear();
                    crate::simd::collect_gt_idx(&mut hits, &vbuf, &tmin);
                    for &x in &hits {
                        let x = x as usize;
                        let tl = &mut top[x];
                        if tl.len() < kmax {
                            tl.push((vbuf[x], b as i32, c as i32));
                        } else {
                            let mut mv = f32::INFINITY;
                            let mut last = 0usize;
                            for (t, e) in tl.iter().enumerate() {
                                if e.0 <= mv {
                                    mv = e.0;
                                    last = t;
                                }
                            }
                            tl.remove(last);
                            tl.push((vbuf[x], b as i32, c as i32));
                        }
                        tmin[x] = if tl.len() < kmax {
                            f32::NEG_INFINITY
                        } else {
                            tl.iter().map(|e| e.0).fold(f32::INFINITY, f32::min)
                        };
                    }
                    a[c] = saved;
                }
            }
            // 块末统一降序排序：与旧版"全程保持降序"的最终顺序一致（成员相同）
            for tl in top.iter_mut() {
                tl.sort_by(|u, vv| vv.0.partial_cmp(&u.0).unwrap());
            }
            let beat: Vec<i64> = beat32.iter().map(|&v| v as i64).collect();
            (top, beat)
        })
        .collect();
    // 并集合取全局 top-kmax（按下标序合并，确定性）
    let mut top: Vec<Vec<(f32, i32, i32)>> = vec![Vec::new(); n];
    let mut beat = vec![0i64; n];
    for (bt, bb) in blocks.iter() {
        for x in 0..n {
            beat[x] += bb[x];
            let tl = &mut top[x];
            if tl.len() + bt[x].len() <= kmax {
                tl.extend_from_slice(&bt[x]);
            } else {
                tl.extend_from_slice(&bt[x]);
                tl.sort_by(|u, vv| vv.0.partial_cmp(&u.0).unwrap());
                tl.truncate(kmax);
            }
        }
    }
    for x in 0..n {
        top[x].sort_by(|u, vv| vv.0.partial_cmp(&u.0).unwrap());
    }
    eprintln!("[reduce2:D] top-{kmax} 配对扫描 {:.1}s", t0.elapsed().as_secs_f32());
    (top, beat)
}

fn block_r2(
    ms: &MatSet,
    pre: &str,
    n: usize,
    ok: &[bool],
    industry: Option<&[i32]>,
    out: &mut Out,
) {
    let single = crate::reduce::best_single_of(n, &ms.rho, ok);
    let single64: Vec<f32> = single
        .iter()
        .map(|&v| if v.is_finite() { v } else { f32::NEG_INFINITY })
        .collect();
    let (top, beat) = pair_scan_topk(n, &ms.rho, &ms.e, &single64, ok, 50);
    let mut t5 = vec![f64::NAN; n];
    let mut t10 = vec![f64::NAN; n];
    let mut t20 = vec![f64::NAN; n];
    let mut t50 = vec![f64::NAN; n];
    let mut cross = vec![f64::NAN; n];
    let mut nbeat = vec![f64::NAN; n];
    for x in 0..n {
        if !ok[x] {
            continue;
        }
        let l = &top[x];
        if l.is_empty() {
            continue;
        }
        let mean_of = |k: usize| -> f64 {
            let k = k.min(l.len());
            l[..k].iter().map(|t| t.0 as f64).sum::<f64>() / k as f64
        };
        t5[x] = mean_of(5);
        t10[x] = mean_of(10);
        t20[x] = mean_of(20);
        t50[x] = mean_of(50);
        nbeat[x] = beat[x] as f64;
        if let Some(ind) = industry {
            let mut c = 0f64;
            let mut t = 0f64;
            for &(v, b, c2) in l.iter() {
                let _ = v;
                let (ib, ic) = (ind[b as usize], ind[c2 as usize]);
                if ib >= 0 && ic >= 0 {
                    t += 1.0;
                    if ib != ic {
                        c += 1.0;
                    }
                }
            }
            if t > 0.0 {
                cross[x] = c / t;
            }
        }
    }
    out.push(&format!("{pre}r2_pair_top5m"), t5);
    out.push(&format!("{pre}r2_pair_top10m"), t10);
    out.push(&format!("{pre}r2_pair_top20m"), t20);
    out.push(&format!("{pre}r2_pair_top50m"), t50);
    out.push(&format!("{pre}r2_nbeat_single"), nbeat);
    out.push(&format!("{pre}r2_cross_top50"), cross);
}

// ============================================================
// E. R3 扩展：贪心篮子 16 步全留 + 曲线形状参数
// ============================================================

/// r3_greedy 的参数化版：kmax 步，返回 (picks n×kmax, marg n×kmax, rhocurve n×(kmax+1))。
fn r3_greedy_k(
    n: usize,
    rho: &[f32],
    e: &[f64],
    ok: &[bool],
    kmax: usize,
) -> (Vec<usize>, Vec<f64>, Vec<f64>) {
    let t0 = Instant::now();
    let se: Vec<f64> = (0..n).map(|i| e[i].max(1e-30).sqrt()).collect();
    let r: Vec<f64> = rho.iter().map(|&v| v as f64).collect();
    let rt = crate::reduce::transpose_f64(&r, n);
    let mut dt = vec![0.0f64; n * n];
    let mut q = vec![0.0f64; n];
    let mut is_in = vec![false; n * n];
    let mut picks = vec![0usize; n * kmax];
    let mut marg = vec![f64::NAN; n * kmax];
    let mut rs = vec![0.0f64; n * (kmax + 1)];
    for step in 0..kmax {
        let picked: Vec<usize> = (0..n)
            .into_par_iter()
            .map(|x| {
                let qx = q[x];
                let num = dt[x * n + x];
                let rx = &r[x * n..x * n + n];
                let dx = &dt[x * n..x * n + n];
                let inn = &is_in[x * n..x * n + n];
                let mut bestv = f64::NEG_INFINITY;
                let mut pick = 0usize;
                for y in 0..n {
                    if y == x || !ok[y] || inn[y] {
                        continue;
                    }
                    let numc = num + rx[y] * se[y];
                    let denc = (qx + (2.0 * se[y]) * dx[y] + e[y]).max(1e-30).sqrt();
                    let v = numc / denc;
                    if v > bestv {
                        bestv = v;
                        pick = y;
                    }
                }
                pick
            })
            .collect();
        for x in 0..n {
            picks[x * kmax + step] = picked[x];
        }
        let q_new: Vec<f64> = (0..n)
            .into_par_iter()
            .map(|x| {
                let p = picked[x];
                q[x] + 2.0 * se[p] * dt[x * n + p] + e[p]
            })
            .collect();
        dt.par_chunks_mut(n).enumerate().for_each(|(x, dx)| {
            let p = picked[x];
            let sp = se[p];
            let rp = &rt[p * n..p * n + n];
            for y in 0..n {
                dx[y] += sp * rp[y];
            }
        });
        for x in 0..n {
            is_in[x * n + picked[x]] = true;
        }
        q = q_new;
        for x in 0..n {
            let v = dt[x * n + x] / q[x].max(1e-30).sqrt();
            rs[x * (kmax + 1) + step + 1] = v;
            marg[x * kmax + step] = v - rs[x * (kmax + 1) + step];
        }
    }
    eprintln!("[reduce2:E] R3 贪心 {kmax} 步 {:.1}s", t0.elapsed().as_secs_f32());
    (picks, marg, rs)
}

/// 增益序列首次跌破 thresh*marg1 的小数步数（线性插值，连续化 n_to_10pct）。
fn marg_cross_step(marg: &[f64], kmax: usize, thr: f64) -> f64 {
    let m1 = marg[0];
    if !m1.is_finite() || m1 == 0.0 {
        return f64::NAN;
    }
    let level = m1 * thr;
    for s in 0..kmax {
        let v = marg[s];
        if v.is_finite() && v < level {
            // 在 s-1 与 s 之间线性插值
            if s == 0 {
                return 1.0;
            }
            let prev = marg[s - 1];
            if prev.is_finite() && prev != v {
                return (s - 1) as f64 + (prev - level) / (prev - v) + 1.0;
            }
            return (s + 1) as f64;
        }
    }
    (kmax + 1) as f64
}

fn block_r3(ms: &MatSet, pre: &str, n: usize, ok: &[bool], out: &mut Out) {
    const K: usize = 16;
    let (picks, marg, rs) = r3_greedy_k(n, &ms.rho, &ms.e, ok, K);
    let r: Vec<f64> = ms.rho.iter().map(|&v| v as f64).collect();
    for s in 0..K {
        out.push(
            &format!("{pre}r3x_marg{}", s + 1),
            (0..n).map(|x| marg[x * K + s]).collect(),
        );
    }
    for (nm, step) in [("r3x_rho4", 4usize), ("r3x_rho8", 8), ("r3x_rho16", 16)] {
        out.push(
            &format!("{pre}{nm}"),
            (0..n).map(|x| rs[x * (K + 1) + step]).collect(),
        );
    }
    let mut auc = vec![f64::NAN; n];
    let mut half = vec![f64::NAN; n];
    let mut n10 = vec![f64::NAN; n];
    let mut ent = vec![f64::NAN; n];
    let mut avg_pr = vec![f64::NAN; n];
    let mut p90_pr = vec![f64::NAN; n];
    let mut gt05 = vec![f64::NAN; n];
    let mut dsoft = vec![f64::NAN; n];
    for x in 0..n {
        if !ok[x] {
            continue;
        }
        let m = &marg[x * K..x * K + K];
        let m1 = m[0];
        if m1.is_finite() && m1 != 0.0 {
            // 边际增益曲线下面积（占首步的比例，衡量篮子后劲儿）
            auc[x] = m.iter().filter(|v| v.is_finite()).sum::<f64>() / (K as f64 * m1.abs());
            half[x] = marg_cross_step(m, K, 0.5);
            n10[x] = marg_cross_step(m, K, 0.1);
            // 正增益部分的熵：增益是集中在前几步还是摊得均匀
            let sp: f64 = m.iter().copied().filter(|v| *v > 0.0).sum();
            if sp > 0.0 {
                let mut h = 0.0f64;
                let mut c = 0usize;
                for &v in m {
                    if v > 0.0 {
                        let w = v / sp;
                        h -= w * w.ln();
                        c += 1;
                    }
                }
                ent[x] = if c > 1 { h / (c as f64).ln() } else { 0.0 };
            }
        }
        // 篮子内部伙伴相似度（冗余度）
        let s = &picks[x * K..x * K + K];
        let mut vals: Vec<f64> = Vec::with_capacity(K * K);
        for i in 0..K {
            for j in 0..K {
                if i != j {
                    vals.push(r[s[i] * n + s[j]]);
                }
            }
        }
        let mean = vals.iter().sum::<f64>() / vals.len() as f64;
        vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
        avg_pr[x] = mean;
        p90_pr[x] = vals[(0.9 * (vals.len() - 1) as f64) as usize];
        gt05[x] = vals.iter().filter(|&&v| v > 0.5).count() as f64 / vals.len() as f64;
        dsoft[x] = if mean > 0.5 { (mean - 0.5) / 0.5 } else { 0.0 };
    }
    out.push(&format!("{pre}r3x_auc"), auc);
    out.push(&format!("{pre}r3x_halflife"), half);
    out.push(&format!("{pre}r3x_n10_interp"), n10);
    out.push(&format!("{pre}r3x_marg_entropy"), ent);
    out.push(&format!("{pre}r3x_avg_partner_rho"), avg_pr);
    out.push(&format!("{pre}r3x_p90_partner_rho"), p90_pr);
    out.push(&format!("{pre}r3x_prho_gt05_share"), gt05);
    out.push(&format!("{pre}r3x_degen_soft"), dsoft);
}

// ============================================================
// F. R4 扩展：5 张核 × 5 个 k 的图 + 相似图对照 + 聚类系数
// ============================================================

fn clustering(n: usize, adj: &[bool], i: usize) -> f64 {
    let mut nbrs = Vec::new();
    for j in 0..n {
        if adj[i * n + j] {
            nbrs.push(j);
        }
    }
    let d = nbrs.len();
    if d < 2 {
        return f64::NAN;
    }
    let mut linked = 0usize;
    for a in 0..d {
        for b in (a + 1)..d {
            if adj[nbrs[a] * n + nbrs[b]] {
                linked += 1;
            }
        }
    }
    linked as f64 / (d * (d - 1) / 2) as f64
}

fn block_r4(ms: &MatSet, pre: &str, n: usize, ok: &[bool], out: &mut Out) {
    let t0 = Instant::now();
    let ks = [5usize, 10, 20, 50, 100];
    let graphs: Vec<(&'static str, &Vec<f32>)> = vec![
        ("rho", &ms.rho),
        ("g", &ms.g),
        ("z", &ms.z),
        ("gs", &ms.gs),
        ("zs", &ms.zs),
    ];
    let n_ok = ok.iter().filter(|&&v| v).count();
    let kks: Vec<usize> = ks.iter().map(|&k| k.min(n_ok.saturating_sub(1))).collect();
    // 邻接矩阵只算 5 次（每图一次排序、5 档共享），adjs 存引用避免拷贝
    let adj_all: Vec<Vec<Vec<bool>>> = graphs
        .iter()
        .map(|(_, m)| topk_adj_multi(n, m, &kks, ok))
        .collect();
    let mut adjs: Vec<(&'static str, usize, &Vec<bool>)> = Vec::new();
    for (gi, (gn, _)) in graphs.iter().enumerate() {
        for (ki, &k) in ks.iter().enumerate() {
            let adj = &adj_all[gi][ki];
            let deg: Vec<f64> = (0..n)
                .into_par_iter()
                .map(|i| (0..n).filter(|&j| adj[i * n + j]).count() as f64)
                .collect();
            out.push(&format!("{pre}r4_deg_{gn}_k{k}"), deg);
            adjs.push((gn, k, adj));
        }
    }
    // 相似图(rho) vs 其他核的邻居重叠（同 k）
    for (gn, k, ag) in adjs.iter() {
        if *gn == "rho" {
            continue;
        }
        let ar = adjs
            .iter()
            .find(|(g2, k2, _)| *g2 == "rho" && *k2 == *k)
            .map(|(_, _, a)| a)
            .unwrap();
        let mut ov = vec![f64::NAN; n];
        let mut ja = vec![f64::NAN; n];
        ov.par_iter_mut()
            .zip(ja.par_iter_mut())
            .enumerate()
            .for_each(|(i, (o, j2))| {
                if !ok[i] {
                    return;
                }
                let mut inter = 0usize;
                let mut uni = 0usize;
                for j in 0..n {
                    let (a, b) = (ar[i * n + j], ag[i * n + j]);
                    inter += (a && b) as usize;
                    uni += (a || b) as usize;
                }
                *o = inter as f64 / *k as f64;
                if uni > 0 {
                    *j2 = inter as f64 / uni as f64;
                }
            });
        out.push(&format!("{pre}r4_ov_{gn}_k{k}"), ov);
        out.push(&format!("{pre}r4_jac_{gn}_k{k}"), ja);
    }
    // k=20 的聚类系数
    for (gn, k, adj) in adjs.iter() {
        if *k != 20 {
            continue;
        }
        let cl: Vec<f64> = (0..n)
            .into_par_iter()
            .map(|i| {
                if !ok[i] {
                    f64::NAN
                } else {
                    clustering(n, adj, i)
                }
            })
            .collect();
        out.push(&format!("{pre}r4_clust_{gn}"), cl);
    }
    eprintln!("[reduce2:F] 图扩展 {:.1}s", t0.elapsed().as_secs_f32());
}

// ============================================================
// G. R5 扩展：多被解释矩阵 × 7 协变量的残差行统计
// ============================================================

/// 通用线性求解（列主元高斯消元，k<=8）。
fn solve_lin(mut a: Vec<Vec<f64>>, mut b: Vec<f64>, k: usize) -> Vec<f64> {
    for col in 0..k {
        let mut piv = col;
        for r in (col + 1)..k {
            if a[r][col].abs() > a[piv][col].abs() {
                piv = r;
            }
        }
        if piv != col {
            a.swap(piv, col);
            b.swap(piv, col);
        }
        let d = a[col][col];
        if d.abs() < 1e-30 {
            continue;
        }
        for r in (col + 1)..k {
            let f = a[r][col] / d;
            if f == 0.0 {
                continue;
            }
            for c in col..k {
                a[r][c] -= f * a[col][c];
            }
            b[r] -= f * b[col];
        }
    }
    let mut x = vec![0.0f64; k];
    for col in (0..k).rev() {
        let mut s = b[col];
        for c in (col + 1)..k {
            s -= a[col][c] * x[c];
        }
        x[col] = s / a[col][col];
    }
    x
}

/// 把 wmat 对协变量回归后的残差行统计。
/// 协变量（每对 7 个）：1, r=rho(B,C), lnB, lnC, |lnB-lnC|, lnE_B, lnE_C。
/// 输出每行 8 个统计：mean,std,p90,t10m,t20m,max,hhi,possh。
fn resid_stats(
    n: usize,
    wmat: &[f32],
    rho: &[f32],
    lnt: &[f64],
    e: &[f64],
    ok: &[bool],
) -> Vec<[f64; 8]> {
    const K: usize = 7;
    let med = crate::reduce::nanmedian(lnt);
    let ln: Vec<f64> = lnt
        .iter()
        .map(|&v| if v.is_finite() { v } else { med })
        .collect();
    let le: Vec<f64> = e.iter().map(|&v| v.max(1.0).ln()).collect();
    let cov = |i: usize, j: usize| -> [f64; K] {
        let lb = ln[i];
        let lc = ln[j];
        [
            1.0,
            rho[i * n + j] as f64,
            lb,
            lc,
            (lb - lc).abs(),
            le[i],
            le[j],
        ]
    };
    // X'X / X'y 按行块并行累加，按块序合并（确定性；与串行仅差浮点结合顺序）
    const RB2: usize = 256;
    let nrb = n.div_ceil(RB2);
    let partials: Vec<([[f64; K]; K], [f64; K])> = (0..nrb)
        .into_par_iter()
        .map(|rb| {
            let mut xtx = [[0.0f64; K]; K];
            let mut xty = [0.0f64; K];
            for i in (rb * RB2)..((rb + 1) * RB2).min(n) {
                if !ok[i] {
                    continue;
                }
                for j in (i + 1)..n {
                    if !ok[j] {
                        continue;
                    }
                    let y = wmat[i * n + j] as f64;
                    if !y.is_finite() {
                        continue;
                    }
                    let d = cov(i, j);
                    for a in 0..K {
                        for b in 0..K {
                            xtx[a][b] += d[a] * d[b];
                        }
                        xty[a] += d[a] * y;
                    }
                }
            }
            (xtx, xty)
        })
        .collect();
    let mut xtx = vec![vec![0.0f64; K]; K];
    let mut xty = vec![0.0f64; K];
    for (px, py) in partials.iter() {
        for a in 0..K {
            for b in 0..K {
                xtx[a][b] += px[a][b];
            }
            xty[a] += py[a];
        }
    }
    for i in 0..K {
        xtx[i][i] += 1e-8;
    }
    let beta = solve_lin(xtx, xty, K);
    let mut res = vec![f32::NAN; n * n];
    res.par_chunks_mut(n).enumerate().for_each(|(i, row)| {
        if !ok[i] {
            return;
        }
        for j in (i + 1)..n {
            if !ok[j] {
                continue;
            }
            let y = wmat[i * n + j] as f64;
            if !y.is_finite() {
                continue;
            }
            let d = cov(i, j);
            let mut fit = 0.0f64;
            for a in 0..K {
                fit += beta[a] * d[a];
            }
            row[j] = (y - fit) as f32;
        }
    });
    for i in 0..n {
        for j in 0..i {
            res[i * n + j] = res[j * n + i];
        }
    }
    // 行统计 8 项（复用 row_stats16 再取需要的）
    (0..n)
        .into_par_iter()
        .map(|i| {
            let mut o = [f64::NAN; 8];
            if !ok[i] {
                return o;
            }
            let s = row_stats16(&res[i * n..i * n + n], i, n, ok);
            // mean,std,p90,t10m,t20m,max,hhi,possh
            o[0] = s[0];
            o[1] = s[1];
            o[2] = s[6];
            o[3] = s[10];
            o[4] = s[11];
            o[5] = s[8];
            o[6] = s[14];
            o[7] = s[13];
            o
        })
        .collect()
}

fn block_r5(
    ms: &MatSet,
    pre: &str,
    n: usize,
    ok: &[bool],
    lnt: Option<&[f64]>,
    has_lnt: bool,
    out: &mut Out,
) {
    if !has_lnt {
        return;
    }
    let t0 = Instant::now();
    let lntv = lnt.unwrap();
    let stat_names = ["mean", "std", "p90", "t10m", "t20m", "max", "hhi", "possh"];
    for (wn, w) in [
        ("g", &ms.g),
        ("gs", &ms.gs),
        ("z", &ms.z),
        ("zs", &ms.zs),
        ("sc", &ms.sc),
    ] {
        let stats = resid_stats(n, w, &ms.rho, lntv, &ms.e, ok);
        for (si, sn) in stat_names.iter().enumerate() {
            out.push(
                &format!("{pre}res_{wn}_{sn}"),
                stats.iter().map(|r| r[si]).collect(),
            );
        }
    }
    eprintln!("[reduce2:G] 残差扩展 {:.1}s", t0.elapsed().as_secs_f32());
}

// ============================================================
// H. R6 扩展：多组二维平面
// ============================================================

/// 一个 (a,b) 平面：spec = b-a 的行统计 + 软专才占比 + 行相关 + 连续象限。
/// 列：spec_{mean,p90,t20m,max}, softshare, rowcorr, genpct, specpct, prod → 9 列。
fn block_plane(
    pre: &str,
    tag: &str,
    a: &[f32],
    b: &[f32],
    n: usize,
    ok: &[bool],
    out: &mut Out,
) {
    // spec 矩阵
    let mut sc = vec![f32::NAN; n * n];
    for i in 0..n {
        for j in (i + 1)..n {
            let (x, y) = (a[i * n + j], b[i * n + j]);
            if x.is_finite() && y.is_finite() {
                sc[i * n + j] = y - x;
                sc[j * n + i] = y - x;
            }
        }
    }
    let stats: Vec<[f64; 16]> = (0..n)
        .into_par_iter()
        .map(|i| {
            if !ok[i] {
                [f64::NAN; 16]
            } else {
                row_stats16(&sc[i * n..i * n + n], i, n, ok)
            }
        })
        .collect();
    out.push(
        &format!("{pre}pl_{tag}_spec_mean"),
        stats.iter().map(|r| r[0]).collect(),
    );
    out.push(
        &format!("{pre}pl_{tag}_spec_p90"),
        stats.iter().map(|r| r[6]).collect(),
    );
    out.push(
        &format!("{pre}pl_{tag}_spec_t20m"),
        stats.iter().map(|r| r[11]).collect(),
    );
    out.push(
        &format!("{pre}pl_{tag}_spec_max"),
        stats.iter().map(|r| r[8]).collect(),
    );
    // 软专才占比：mean_j sigmoid((b-thr)/temp)*sigmoid((-a-thr)/temp)；行相关：a 与 b 的 spearman
    // （原实现此处隐式温度 1.0，已统一为全局 --sig-temp/--sig-thr 口径）
    let sc = crate::reduce::soft();
    let rows2: Vec<(f64, f64, f64)> = (0..n)
        .into_par_iter()
        .map(|i| {
            if !ok[i] {
                return (f64::NAN, f64::NAN, f64::NAN);
            }
            let mut sh = 0.0f64;
            let mut cnt = 0usize;
            let mut va = Vec::new();
            let mut vb = Vec::new();
            for j in 0..n {
                if j == i {
                    continue;
                }
                let (x, y) = (a[i * n + j] as f64, b[i * n + j] as f64);
                if x.is_finite() && y.is_finite() {
                    sh += sigmoid((y - sc.thr) / sc.temp) * sigmoid((-x - sc.thr) / sc.temp);
                    cnt += 1;
                    va.push(x);
                    vb.push(y);
                }
            }
            let share = if cnt > 0 { sh / cnt as f64 } else { f64::NAN };
            let corr = spearman(&va, &vb, 30);
            // gen 侧行 max(a)
            let gmax = va
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max);
            (share, corr, gmax)
        })
        .collect();
    out.push(
        &format!("{pre}pl_{tag}_softshare"),
        rows2.iter().map(|t| t.0).collect(),
    );
    out.push(
        &format!("{pre}pl_{tag}_rowcorr"),
        rows2.iter().map(|t| t.1).collect(),
    );
    // 连续象限：行 max(a) 与行 max(spec) 各自做横截面百分位，再相乘
    let pct_rank = |v: &[f64]| -> Vec<f64> {
        let mut idx: Vec<usize> = (0..n).filter(|&i| ok[i] && v[i].is_finite()).collect();
        idx.sort_by(|&a2, &b2| v[a2].partial_cmp(&v[b2]).unwrap().then(a2.cmp(&b2)));
        let mut r = vec![f64::NAN; n];
        let m = idx.len();
        for (rk, &i) in idx.iter().enumerate() {
            r[i] = if m > 1 { rk as f64 / (m - 1) as f64 } else { 0.5 };
        }
        r
    };
    let gen_max: Vec<f64> = rows2.iter().map(|t| t.2).collect();
    let spec_max: Vec<f64> = stats.iter().map(|r| r[8]).collect();
    let gp = pct_rank(&gen_max);
    let sp = pct_rank(&spec_max);
    out.push(&format!("{pre}pl_{tag}_genpct"), gp.clone());
    out.push(&format!("{pre}pl_{tag}_specpct"), sp.clone());
    out.push(
        &format!("{pre}pl_{tag}_prod"),
        (0..n)
            .map(|i| {
                if gp[i].is_finite() && sp[i].is_finite() {
                    gp[i] * sp[i]
                } else {
                    f64::NAN
                }
            })
            .collect(),
    );
}

fn block_planes(ms: &MatSet, pre: &str, n: usize, ok: &[bool], out: &mut Out) {
    let t0 = Instant::now();
    block_plane(pre, "zzs", &ms.z, &ms.zs, n, ok, out);       // 通才 Z vs 专才 ZS
    block_plane(pre, "zexz", &ms.z, &ms.exz, n, ok, out);     // 活跃度基线 Z vs (p,q) 基线 EXZ
    block_plane(pre, "zsexzs", &ms.zs, &ms.exzs, n, ok, out); // 专才 ZS vs 条件化专才 EXZS
    eprintln!("[reduce2:H] 平面扩展 {:.1}s", t0.elapsed().as_secs_f32());
}

// ============================================================
// 入口
// ============================================================

pub(crate) fn expand(
    ms: &MatSet,
    pre: &str,
    n: usize,
    ok: &[bool],
    industry: Option<&[i32]>,
    lnt: Option<&[f64]>,
    has_lnt: bool,
    out: &mut Out,
) {
    let t0 = Instant::now();
    block_row_stats(ms, pre, n, ok, out);
    block_explain(ms, pre, n, ok, out);
    block_bestcomp(ms, pre, n, ok, out);
    block_r2(ms, pre, n, ok, industry, out);
    block_r3(ms, pre, n, ok, out);
    block_r4(ms, pre, n, ok, out);
    block_r5(ms, pre, n, ok, lnt, has_lnt, out);
    block_planes(ms, pre, n, ok, out);
    eprintln!("[reduce2] 扩展块总 {:.1}s", t0.elapsed().as_secs_f32());
}
