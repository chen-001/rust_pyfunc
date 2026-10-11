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
//! 用法：
//!   yupei_pair topk <backup_root> <date> <outdir> [--mat cnt_t1] [--threads N] [--stocks-only]
//!                  [--k-list 1,2,5,10,20,50,100,200,500]
//!
//! 输出 <outdir>/<date>/topk/ 下的 names.txt、factors.bin（N×72，f32 行主序，只含
//! 白名单股票，行序与 reduce_rs 的 factors.bin 一致）、summary.json。
//!
//! ## 确定性怎么保证的
//!
//! B 的下标范围按固定块数 NBLOCKS 切开（切法只依赖 n 与 NBLOCKS，与线程数无关），
//! 每块在一块自己的累加器上求和，全部并行跑完之后按下标从小到大依次合并。于是
//! 换线程数（16/128/512）得到的结果逐位相同，同一输入跑两遍也逐位相同。

use rayon::prelude::*;
use std::path::PathBuf;
use std::time::Instant;

use crate::{parse_flag, read_codes, read_f32, read_symbol_map, write_f32};

/// 默认九档 K，顺序即输出顺序。
const KS_DEFAULT: [usize; 9] = [1, 2, 5, 10, 20, 50, 100, 200, 500];
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

/// 排序规则：分数大的在前，相同按下标小的在前。
#[inline]
fn h_less(ha: f32, ia: u32, hb: f32, ib: u32) -> std::cmp::Ordering {
    match ha.partial_cmp(&hb) {
        Some(std::cmp::Ordering::Greater) => std::cmp::Ordering::Less,
        Some(std::cmp::Ordering::Less) => std::cmp::Ordering::Greater,
        _ => ia.cmp(&ib),
    }
}

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
            crate::simd::fill_h(&mut h, wb, wc, p, q, addm);
            crate::simd::fill_gap(&mut gap, &h, p, q);
            h[b] = f32::NEG_INFINITY;
            h[c] = f32::NEG_INFINITY;
            gap[b] = f32::NEG_INFINITY;
            gap[c] = f32::NEG_INFINITY;
            if take == 0 {
                continue;
            }
            // 打包 (score,idx) u64 键：升序整型序 = 分数降序+下标升序
            for x in 0..n {
                kh[x] = crate::simd::pack_desc_idx_asc(h[x], x as u32);
                kg[x] = crate::simd::pack_desc_idx_asc(gap[x], x as u32);
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

pub fn cmd_topk(args: &[String]) {
    let t_all = Instant::now();
    let backup_root = PathBuf::from(&args[0]);
    let date: i64 = args[1].parse().expect("date");
    let outdir = PathBuf::from(&args[2]);
    let mats_arg = parse_flag(args, "--mat").unwrap_or("cnt_t1").to_string();
    let mats: Vec<String> = mats_arg.split(',').map(|s| s.trim().to_string()).collect();
    let multi = mats.len() > 1;
    if let Some(t) = parse_flag(args, "--threads") {
        let k: usize = t.parse().unwrap();
        let _ = rayon::ThreadPoolBuilder::new().num_threads(k).build_global();
    }
    let stocks_only = args.iter().any(|a| a == "--stocks-only");
    let ks: Vec<usize> = match parse_flag(args, "--k-list") {
        None => KS_DEFAULT.to_vec(),
        Some(s) => {
            let v: Vec<usize> = s.split(',').map(|x| x.trim().parse().expect("--k-list")).collect();
            if v.is_empty() || v.iter().any(|&k| k == 0) {
                panic!("--k-list 必须是若干正整数，收到 {s}");
            }
            v
        }
    };
    let maxk = *ks.iter().max().unwrap();

    let day = backup_root.join(date.to_string());
    let codes = read_codes(&day.join("codes.txt")).expect("读 codes.txt 失败");
    let n = codes.len();

    // 股票白名单（项目规则：非个股既不参与配对也不进候选，且在计算之前筛）
    let basic = std::env::var("RUST_PYFUNC_BASIC_INFO_DIR")
        .unwrap_or_else(|_| "/ssd_data/data/basic_info".to_string());
    let mut mask = vec![true; n];
    if stocks_only {
        let sm = read_symbol_map(&std::path::Path::new(&basic).join("symbol_map.csv"))
            .expect("读 symbol_map.csv 失败，不允许退回全算");
        for i in 0..n {
            mask[i] = sm.contains(&codes[i]);
        }
        let n_bad = mask.iter().filter(|&&v| !v).count();
        eprintln!("[topk:p1] n = {n}, date = {date}, 非股票 {n_bad} 只（基金/债券），已剔除");
        if n_bad == n {
            panic!("codes.txt 与股票表完全不相交，白名单对不上");
        }
    } else {
        eprintln!("[topk:p1] n = {n}, date = {date}，没给 --stocks-only：非个股也会参与配对");
    }
    let n_stocks = mask.iter().filter(|&&v| v).count();
    let keep: Vec<usize> = (0..n).filter(|&i| mask[i]).collect();
    let nk = keep.len();

    let dir = outdir.join(date.to_string()).join("topk");
    std::fs::create_dir_all(&dir).expect("建输出目录失败");

    // 各矩阵完全独立：读盘、扫描、落盘、汇总全部并行。
    // 多矩阵时文件名带 _{mat} 后缀、列名带 {mat}_ 前缀（与 reduce 前缀口径一致），
    // 避免旧版多矩阵同名文件互相覆盖的静默丢失。
    let results: Vec<(String, serde_json::Value)> = mats
        .par_iter()
        .map(|mat| {
            let tag = if multi { format!("_{mat}") } else { String::new() };
            let pre = if multi { format!("{mat}_") } else { String::new() };

            // ---- 读矩阵、算 E 与 rho（与 net 完全同一套） ----
            let t0 = Instant::now();
            let mpath = day.join("mats").join(format!("{mat}.bin"));
            let sdir = read_f32(&mpath).expect("读矩阵失败");
            if sdir.len() != n * n {
                panic!("矩阵 {mat} 有 {} 个元素，期望 {}", sdir.len(), n * n);
            }
            let e: Vec<f64> = (0..n).map(|i| 2.0 * sdir[i * n + i] as f64).collect();
            let n_bad_e = e.iter().filter(|&&v| !(v > 0.0)).count();
            let mut rho = vec![0.0f32; n * n];
            for i in 0..n {
                rho[i * n + i] = 1.0;
                let si = e[i].sqrt();
                for j in (i + 1)..n {
                    let sj = e[j].sqrt();
                    let v = (sdir[i * n + j] + sdir[j * n + i]) as f64 / (si * sj);
                    rho[i * n + j] = v as f32;
                    rho[j * n + i] = v as f32;
                }
            }
            drop(sdir);
            let t_read = t0.elapsed().as_secs_f64();
            eprintln!("[topk:p2] 矩阵 {mat} 读入并算完 rho {t_read:.1}s，自能量非正 {n_bad_e} 只");

            // ---- 分块并行累加，再按下标顺序合并 ----
            let nb = NBLOCKS.min(n).max(1);
            let bounds: Vec<(usize, usize)> =
                (0..nb).map(|k| (k * n / nb, (k + 1) * n / nb)).collect();
            // -inf 掩码与定长合法候选数：对全部有效配对都一样
            let addm: Vec<f32> = (0..n)
                .map(|i| if mask[i] { 0.0 } else { f32::NEG_INFINITY })
                .collect();
            let n_valid = n_stocks.saturating_sub(2);
            let t0 = Instant::now();
            let accs: Vec<Acc> = bounds
                .par_iter()
                .map(|&(b0, b1)| {
                    block_accumulate(n, &rho, &e, &mask, &addm, n_valid, b0, b1, &ks, maxk)
                })
                .collect();
            let t_scan = t0.elapsed().as_secs_f64();
            let pairs: u64 = accs.iter().map(|a| a.pairs).sum();
            let skipped: u64 = accs.iter().map(|a| a.skipped).sum();
            eprintln!(
                "[topk:p3:{mat}] {} 档 × 2 口径一趟扫完：{pairs} 对，跳过 {skipped} 对，{t_scan:.1}s",
                ks.len()
            );

            let t0 = Instant::now();
            let mut total = Acc::new(n, ks.len());
            for a in accs.iter() {
                total.add(a);
            }
            let t_merge = t0.elapsed().as_secs_f64();

            // ---- 落盘：只保留白名单股票，行序与 reduce_rs 的 factors.bin 一致 ----
            let t0 = Instant::now();
            let nlev = ks.len();
            let f = nlev * 4 * 2;
            let mut rows = Vec::with_capacity(nk * f);
            // 每口径四个量；crit0=h-top 沿用旧名 x0topk{K}_*，crit1=gap-top 用 gtopk{K}_*
            for &i in keep.iter() {
                for crit in 0..2 {
                    for l in 0..nlev {
                        let base = (crit * nlev + l) * n + i;
                        let c = total.cnt[base];
                        rows.push(c as f32);
                        rows.push(if c > 0 {
                            (total.sum_gap[base] / c as f64) as f32
                        } else {
                            f32::NAN
                        });
                        rows.push(if c > 0 {
                            (total.sum_h[base] / c as f64) as f32
                        } else {
                            f32::NAN
                        });
                        // mass_share：对全部配对取平均（没被选中就自然摊薄），总有限
                        rows.push((total.sum_mass[base] / pairs.max(1) as f64) as f32);
                    }
                }
            }

            {
                let mut s = String::new();
                for crit in 0..2 {
                    let pf = if crit == 0 { "x0topk" } else { "gtopk" };
                    for &k in ks.iter() {
                        s.push_str(&format!("{pre}{pf}{k}_count\n"));
                        s.push_str(&format!("{pre}{pf}{k}_gain_mean\n"));
                        s.push_str(&format!("{pre}{pf}{k}_h_mean\n"));
                        s.push_str(&format!("{pre}{pf}{k}_mass_share\n"));
                    }
                }
                std::fs::write(dir.join(format!("names{tag}.txt")), s).unwrap();
            }
            write_f32(&dir.join(format!("factors{tag}.bin")), &rows).unwrap();

            // ---- 汇总（覆盖度按口径分别报）----
            let mut levels: Vec<serde_json::Value> = Vec::new();
            for crit in 0..2 {
                let crit_name = if crit == 0 { "h-top" } else { "gap-top" };
                for (l, &k) in ks.iter().enumerate() {
                    let base = (crit * nlev + l) * n;
                    let mut sel: Vec<(u32, usize)> = keep
                        .iter()
                        .map(|&i| (total.cnt[base + i], i))
                        .filter(|&(c, _)| c > 0)
                        .collect();
                    let n_selected = sel.len();
                    let n_selections: u64 = sel.iter().map(|&(c, _)| c as u64).sum();
                    sel.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
                    let top10: Vec<serde_json::Value> = sel
                        .iter()
                        .take(10)
                        .map(|&(c, i)| serde_json::json!({"code": codes[i], "count": c}))
                        .collect();
                    let top10_sum: u64 = sel.iter().take(10).map(|&(c, _)| c as u64).sum();
                    let top1 = sel.first().map(|&(c, _)| c as u64).unwrap_or(0);
                    levels.push(serde_json::json!({
                        "crit": crit_name,
                        "K": k,
                        "n_selected": n_selected,
                        "coverage": n_selected as f64 / n_stocks.max(1) as f64,
                        "n_selections": n_selections,
                        "top1_count": top1,
                        "top10_share": top10_sum as f64 / n_selections.max(1) as f64,
                        "top10": top10,
                    }));
                    eprintln!(
                        "[topk:p4:{mat}] {crit_name} K={k:<3} 有值股票 {n_selected}/{n_stocks}（{:.1}%），选中次数 {n_selections}，前 10 占比 {:.3}",
                        100.0 * n_selected as f64 / n_stocks.max(1) as f64,
                        top10_sum as f64 / n_selections.max(1) as f64
                    );
                }
            }

            let summary = serde_json::json!({
                "date": date,
                "n": n,
                "n_stocks": n_stocks,
                "n_non_stock": n - n_stocks,
                "stocks_only": stocks_only,
                "mat": mat,
                "threads": rayon::current_num_threads(),
                "nblocks": nb,
                "K_list": ks,
                "n_pairs": pairs,
                "n_pairs_skipped": skipped,
                "matrix": mpath.to_string_lossy(),
                "codes_source": day.join("codes.txt").to_string_lossy(),
                "seconds_read": t_read,
                "seconds_scan": t_scan,
                "seconds_merge": t_merge,
                "seconds_write": t0.elapsed().as_secs_f64(),
                "levels": levels,
            });
            (mat.clone(), summary)
        })
        .collect();

    // 汇总落盘：单矩阵保持旧文件名 summary.json，多矩阵每矩阵一份
    for (m, v) in &results {
        let fname = if multi {
            format!("summary_{m}.json")
        } else {
            "summary.json".to_string()
        };
        std::fs::write(
            dir.join(fname),
            serde_json::to_string_pretty(v).unwrap(),
        )
        .unwrap();
    }

    eprintln!(
        "[topk:done] {:.1}s，输出 {}",
        t_all.elapsed().as_secs_f64(),
        dir.display()
    );
    println!(
        "{{\"ok\":true,\"date\":{date},\"n\":{n},\"n_stocks\":{nk},\"n_mats\":{},\"outdir\":\"{}\"}}",
        results.len(),
        dir.display()
    );
}
