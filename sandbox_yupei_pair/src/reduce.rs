//! 玉佩2 配对互补网络：把 N×N 中间矩阵压成每股一个因子值（子命令 reduce）。
//!
//! 这个文件是把 experiments/yupei2/pair_reduce.py 里的因子合成（77 列）完整搬进 Rust。
//! 对照基准就是那个脚本，逻辑一句一句对着写的，包括几个看起来奇怪的地方：
//!   * r6 的 Zrow/ZSrow 只填严格上三角，所以最后一行（下标 n-1）的 gen_max / gen_top20 /
//!     zs_max / row_corr 天然是空值，不是 bug。
//!   * r6 的 Sc 是对称阵且对角填 0，所以 spec_max 永远 >= 0（全负时取到对角那个 0）。
//!   * r2 的跨行业列在找不到合法配对时，bB=-1 会让 Python 取到 industry 的最后一个元素，
//!     这里照样绕一圈取（那些行最后整行写空值，结果看不出来）。
//!   * r3 贪心的方向：每一步先用更新前的 D 求 argmax，再用更新前的 D 更新 Q，最后才更新 D。
//!
//! ## 每个因子名对应哪个函数（自查用，77 个一个不落）
//!
//! | 列名 | 合成在 |
//! |---|---|
//! | r1_explain_count | `r1_role` → cnt |
//! | r1_explain_count_z2 | `r1_role` → cnt_z2 |
//! | r1_explain_gain_mean | `r1_role` → gmean |
//! | r1_explain_gain_max | `r1_role` → gmax |
//! | r1_explain_z_mean | `r1_role` → zmean |
//! | r1_as_partner_z2_count | `r1_role` → part_z2 |
//! | r1_bestcomp_gain / _z / _mutual / _rank | `r1_bestcomp` |
//! | s_r1_explain_count / _count_z2 / _gain_mean / _gain_max / _z_mean / _as_partner_z2_count | `r1_role`，换 GSTAR / ZS / X0STAR |
//! | s_r1_bestcomp_gain / _z / _mutual / _rank | `r1_bestcomp`，换 GSTAR / ZS / X0STAR |
//! | r2_best_single | `best_single_of` |
//! | r2_best_pair | `pair_scan`（best 输出） |
//! | r2_best_pair_Bidx / _Cidx | `pair_scan`（bi / ci 输出） |
//! | r2_pair_gain / _ratio | `three_way` |
//! | r2_best_pair_cross_industry | `three_way`（读 industry） |
//! | r3_marg1..marg4 / _n_to_10pct / _basket_rho8 | `r3_greedy`（marg / rs） |
//! | r3_avg_partner_rho / _max_partner_rho / _degenerate | `r3_greedy`（avg_pr / hub） |
//! | r4_deg_rho_topk / _deg_G_topk / _neighbor_overlap_k / _neighbor_jaccard / _quad | `topk_adj` + `r4_graphs` |
//! | r5_res_max / _top20_mean / _mean / _hhi | `r5_residual` |
//! | s_r5_res_max / _top20_mean / _mean / _hhi | `r5_residual`，换 GSTAR |
//! | p_gen_max / p_gen_top20 / p_spec_max / p_spec_top20 / p_zs_max / p_row_corr / p_spec_share / p_quad | `r6_plane` |
//! | null_r1_explain_gain_mean / _gain_max / _z_mean | `r1_role`，把 G 与 Z 都换成 Z |
//! | null_r1_bestcomp_gain / _mutual / _rank | `r1_bestcomp`，把 G 与 Z 都换成 Z |
//! | null_r5_res_max / _top20_mean / _mean / _hhi | `r5_residual`，把 G 换成 Z |
//! | null_s_r1_explain_gain_mean / _gain_max / _z_mean | `r1_role`，第二路把 GSTAR 与 ZS 都换成 ZS |
//! | null_s_r1_bestcomp_gain / _mutual / _rank | `r1_bestcomp`，第二路上同 |
//! | null_s_r5_res_max / _top20_mean / _mean / _hhi | `r5_residual`，第二路把 GSTAR 换成 ZS |
//!
//! ## 没搬的部分（都是诊断，不进 factors.bin）
//!
//! * diag.json 全部：矩阵检查、行业说明、因子分布、X0 集中度、EXPLAIN 释义表。
//! * R4 里的社群对比 `comm_of`（label_propagation + ari_and_agree）、边集对照 `ov_extra`、
//!   两个度序列的 Spearman（deg_sp）——只写进 diag，不影响 5 个 r4_ 列。
//! * RHO_U.bin / SINGLE.bin / RHO_US.bin / SINGLE_S.bin 只被 Python 的自查项用到，没读。
//! * 跨日稳定性诊断 diag_crossday、与老因子重叠 diag_old_factors、空模型诊断 diag_null_model。
//! * --selftest 的合成 fixture（那是 Python 自测用的，Rust 这边用真实两天对账）。
//!
//! ## 用法
//!
//! ```text
//! yupei_pair reduce <pair_out_dir> <date> <outdir> [--backup <root>] [--threads N]
//! ```
//! 输入 <pair_out_dir>/<date>/ 下的 codes.txt、rho.bin、E.bin、G.bin、Z.bin、X0.bin、
//! GSTAR.bin、ZS.bin、X0STAR.bin，以及 <backup_root>/<date>/ 的 stats.bin（成交笔数，
//! 每行 14 个 f64 的第 0 个）与 industry.bin（u64 n + n 个 i16，长度按 symbol_map.csv 的
//! 行序对齐到 codes 下标）。输出 <outdir>/<date>/reduce_rs/ 下的 names.txt、factors.bin
//! （N×F f32 行主序）、factors.json（给人看与给 Python 探索，缺失值写 null）。
//!
//! 确定性：所有并行都是「按块收集、再按下标顺序合并」，不遍历 HashMap。

use rayon::prelude::*;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::time::Instant;

use crate::{parse_flag, read_codes, read_f32, read_symbol_map, mmap_f32, MatData};

// pair_reduce.py 的默认配置：--k 20 / --max-k 8。
pub(crate) const K_TOPK: usize = 20;
pub(crate) const MAX_K: usize = 8;

/// sigmoid 软化与硬阈值的共享参数（--sig-temp / --sig-thr，E 组参数刷选用）。
/// reduce 启动时由 cmd_reduce 写入 OnceLock，全进程只设一次；所有用
/// sigmoid((z-thr)/temp) 或 z>thr 的统计（z2_count、zsoft、pz_soft、spec_soft 等）
/// 统一从 soft() 读取，保证同一套参数口径一致。
#[derive(Clone, Copy)]
pub(crate) struct SoftCfg {
    pub temp: f64,
    pub thr: f64,
}
static SOFT: OnceLock<SoftCfg> = OnceLock::new();
const SOFT_DEFAULT: SoftCfg = SoftCfg {
    temp: 0.3,
    thr: 2.0,
};
pub(crate) fn soft() -> SoftCfg {
    *SOFT.get().unwrap_or(&SOFT_DEFAULT)
}
/// 供 cmd_reduce 在启动时写入参数（重复设置时忽略，保持首次值）。
pub(crate) fn set_soft(cfg: SoftCfg) {
    let _ = SOFT.set(cfg);
}

// ============================================================
// 小工具
// ============================================================

/// 与 numpy 的负下标一致：-1 取最后一个元素（r2 的跨行业列要用）。
#[inline]
pub(crate) fn wrap(i: i32, n: usize) -> usize {
    if i < 0 {
        (n as i64 + i as i64) as usize
    } else {
        i as usize
    }
}

/// 二维方阵转置（[a*n+b] -> [b*n+a]）。
pub(crate) fn transpose_f32(a: &[f32], n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; n * n];
    const B: usize = 64;
    for bi in (0..n).step_by(B) {
        for bj in (0..n).step_by(B) {
            let i1 = (bi + B).min(n);
            let j1 = (bj + B).min(n);
            for i in bi..i1 {
                for j in bj..j1 {
                    out[j * n + i] = a[i * n + j];
                }
            }
        }
    }
    out
}

pub(crate) fn transpose_f64(a: &[f64], n: usize) -> Vec<f64> {
    let mut out = vec![0.0f64; n * n];
    const B: usize = 64;
    for bi in (0..n).step_by(B) {
        for bj in (0..n).step_by(B) {
            let i1 = (bi + B).min(n);
            let j1 = (bj + B).min(n);
            for i in bi..i1 {
                for j in bj..j1 {
                    out[j * n + i] = a[i * n + j];
                }
            }
        }
    }
    out
}

pub(crate) fn max_abs_diff_to_transpose(a: &[f32], n: usize) -> f64 {
    // 抽样对称性检查：每 64 行抽 1 行扫全行（含对角两侧），
    // 保留"发现不对称数据就报警"的防御力，但避免为检查而
    // fault-in 整张 62MB 映射（MatSet 装载路径，串行不嵌套 par）。
    let mut mx = 0.0f64;
    let mut i = 0usize;
    while i < n {
        for j in 0..n {
            let d = (a[i * n + j] as f64 - a[j * n + i] as f64).abs();
            if d > mx {
                mx = d;
            }
        }
        i += 64;
    }
    mx
}

/// 对称化：(M + M^T) * 0.5（f32 运算，与 numpy 的 F32(0.5) 一致），对角置 0。
pub(crate) fn symmetrize_f32(a: &mut [f32], n: usize) {
    for i in 0..n {
        for j in (i + 1)..n {
            let v = ((a[i * n + j] + a[j * n + i]) * 0.5f32) as f32;
            a[i * n + j] = v;
            a[j * n + i] = v;
        }
        a[i * n + i] = 0.0;
    }
}

/// 平均秩（scipy.stats.rankdata 默认的 average 口径），输入必须全是有限值。
pub(crate) fn rank_avg(v: &[f64]) -> Vec<f64> {
    let m = v.len();
    let mut idx: Vec<usize> = (0..m).collect();
    idx.sort_by(|&a, &b| v[a].partial_cmp(&v[b]).unwrap().then(a.cmp(&b)));
    let mut r = vec![0.0f64; m];
    let mut i = 0usize;
    while i < m {
        let mut j = i;
        while j + 1 < m && v[idx[j + 1]] == v[idx[i]] {
            j += 1;
        }
        let avg = (i + j + 2) as f64 / 2.0; // 秩从 1 开始：i+1 .. j+1 的平均
        for k in i..=j {
            r[idx[k]] = avg;
        }
        i = j + 1;
    }
    r
}

/// Spearman，只在两边都有限的位置上算；样本不够返回空值。
pub(crate) fn spearman(x: &[f64], y: &[f64], min_n: usize) -> f64 {
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    for i in 0..x.len() {
        if x[i].is_finite() && y[i].is_finite() {
            xs.push(x[i]);
            ys.push(y[i]);
        }
    }
    if xs.len() < min_n {
        return f64::NAN;
    }
    let mut rx = rank_avg(&xs);
    let mut ry = rank_avg(&ys);
    let mx = rx.iter().sum::<f64>() / rx.len() as f64;
    let my = ry.iter().sum::<f64>() / ry.len() as f64;
    for v in rx.iter_mut() {
        *v -= mx;
    }
    for v in ry.iter_mut() {
        *v -= my;
    }
    let d = (rx.iter().map(|v| v * v).sum::<f64>() * ry.iter().map(|v| v * v).sum::<f64>()).sqrt();
    if d <= 0.0 {
        return f64::NAN;
    }
    rx.iter().zip(ry.iter()).map(|(a, b)| a * b).sum::<f64>() / d
}

/// nanmedian：忽略空值；偶数个取中间两个的均值，与 numpy 一致。
pub(crate) fn nanmedian(v: &[f64]) -> f64 {
    let mut f: Vec<f64> = v.iter().copied().filter(|x| x.is_finite()).collect();
    if f.is_empty() {
        return f64::NAN;
    }
    f.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let m = f.len();
    if m % 2 == 1 {
        f[m / 2]
    } else {
        (f[m / 2 - 1] + f[m / 2]) / 2.0
    }
}

// ============================================================
// 输入
// ============================================================

/// X0.bin / X0STAR.bin：四舍五入取整，只在严格上三角有效，其余写 -1。
/// 保持串行：MatSet 装载在外层多矩阵 par 循环内，嵌套 par 只会饥饿排队。
pub(crate) fn read_x0(raw: &[f32], n: usize) -> Vec<i32> {
    let mut out = vec![-1i32; n * n];
    for i in 0..n {
        for j in (i + 1)..n {
            out[i * n + j] = (raw[i * n + j] as f64).round_ties_even() as i32;
        }
    }
    out
}

pub(crate) fn load_matrix(p: &Path, n: usize, what: &str) -> Vec<f32> {
    let a = read_f32(p).unwrap_or_else(|e| panic!("读 {what} 失败: {e}"));
    if a.len() != n * n {
        panic!("{what} 有 {} 个元素，期望 {}", a.len(), n * n);
    }
    a
}

/// industry.bin = u64 n + n 个 i16。长度等于股票数就直接用，否则按 symbol_map.csv 的
/// 行序（也就是它的 pos 列）映射到 codes 下标，对不上的写 -1。
pub(crate) fn load_industry(path: &Path, codes: &[String], n: usize) -> Option<Vec<i32>> {
    let data = std::fs::read(path).ok()?;
    if data.len() < 8 {
        panic!("industry.bin 太短: {}", path.display());
    }
    let n_ind = u64::from_le_bytes(data[0..8].try_into().unwrap()) as usize;
    if data.len() < 8 + n_ind * 2 {
        panic!("industry.bin 长度与 n 不符: {}", path.display());
    }
    let ind: Vec<i32> = (0..n_ind)
        .map(|i| i16::from_le_bytes([data[8 + i * 2], data[8 + i * 2 + 1]]) as i32)
        .collect();
    if n_ind == n {
        return Some(ind);
    }
    let smap = "/ssd_data/data/basic_info/symbol_map.csv";
    let txt = std::fs::read_to_string(smap).ok()?;
    let mut code_by_pos: Vec<&str> = Vec::new();
    for (i, l) in txt.lines().enumerate() {
        if i == 0 {
            continue;
        }
        let c = l.split(',').next().unwrap_or("").trim();
        if !c.is_empty() {
            code_by_pos.push(c);
        }
    }
    if n_ind > code_by_pos.len() {
        return None;
    }
    let mut c2i: HashMap<&str, usize> = HashMap::with_capacity(n_ind);
    for pos in 0..n_ind.min(code_by_pos.len()) {
        c2i.entry(code_by_pos[pos]).or_insert(pos);
    }
    Some(
        codes
            .iter()
            .map(|c| c2i.get(c.as_str()).map(|&p| ind[p]).unwrap_or(-1))
            .collect(),
    )
}

/// 备份目录的 stats.bin（u64 n + n×14 个 f64，第 0 列是成交笔数），按 codes.txt 对齐到 codes，
/// 再取对数。对齐不上写空值。
pub(crate) fn load_lnt(day: &Path, codes: &[String]) -> Option<Vec<f64>> {
    let sp = day.join("stats.bin");
    let cp = day.join("codes.txt");
    if !sp.exists() || !cp.exists() {
        return None;
    }
    let arr = crate::read_ntrades(&sp).expect("读 stats.bin 失败");
    let bcodes = read_codes(&cp).expect("读备份 codes.txt 失败");
    let mut c2v: HashMap<&str, f64> = HashMap::with_capacity(bcodes.len());
    for i in 0..arr.len().min(bcodes.len()) {
        c2v.entry(bcodes[i].as_str()).or_insert(arr[i]);
    }
    Some(
        codes
            .iter()
            .map(|c| match c2v.get(c.as_str()) {
                // numpy 的 np.maximum(nan, 1.0) 还是 nan，Rust 的 f64::max 会返回 1.0，这里要分开写
                Some(&v) if v.is_nan() => f64::NAN,
                Some(&v) => v.max(1.0).ln(),
                None => f64::NAN,
            })
            .collect(),
    )
}

// ============================================================
// R2：全配对扫描（N 趟 O(N^2)）
// ============================================================

/// 对每只 X 求 best_pair = max_{B<C, X 不属于 {B,C}, 三只都是个股} rho[X][U_BC]。
/// 等价式 rho[X][U_BC] = (M[X,B] + M[X,C]) * Winv[B,C]，
/// M[X,B] = se_B * rho[X][B]，Winv[B,C] = 1/sqrt(E_B + E_C + 2 rho[B,C] se_B se_C)。
///
/// 与 pair_reduce.py 的 pair_scan 一样用 f32 累加。并列取「B 最小、再 C 最小」，
/// 与 numpy 的（严格大于才更新 + 块内 argmax 取第一个）一致。
/// 按 B 分块并行，每块内部写下标顺序，最后按 B 升序合并，结果与单线程逐 B 跑完全一样。
pub(crate) fn pair_scan(n: usize, rho: &[f32], e: &[f64], ok: &[bool]) -> (Vec<f32>, Vec<i32>, Vec<i32>) {
    let t0 = Instant::now();
    let se: Vec<f32> = (0..n).map(|i| e[i].max(1e-30).sqrt() as f32).collect();
    let mut m = vec![0.0f32; n * n];
    m.par_chunks_mut(n).enumerate().for_each(|(x, row)| {
        for b in 0..n {
            row[b] = rho[x * n + b] * se[b];
        }
    });
    // Winv：den 在 f64 里算（E 是 f64），2*rho*se*se 三项乘积在 f32 里逐步舍入，与 numpy 相同
    let mut wm = vec![0.0f32; n * n];
    wm.par_chunks_mut(n).enumerate().for_each(|(b, row)| {
        for c in 0..n {
            if b == c {
                row[c] = 0.0;
                continue;
            }
            let t = (2.0f32 * rho[b * n + c]) * se[b] * se[c];
            let den = (e[b] + e[c] + t as f64).max(1e-30).sqrt() as f32;
            row[c] = 1.0f32 / den;
        }
    });
    let mt = transpose_f32(&m, n);
    eprintln!("[reduce:p2] pair_scan 准备 M/W {:.1}s", t0.elapsed().as_secs_f32());

    const BLK: usize = 64;
    let nblk = n.div_ceil(BLK);
    let t0 = Instant::now();
    // 每块输出两个平面：ov=各 B 下每 x 的最大值，oc=取得最大值的 C 下标。
    // 内层 (a+row)*w 用 SIMD 物化，比较更新用带掩码的 AVX2 助手。
    let blocks: Vec<(Vec<f32>, Vec<i32>)> = (0..nblk)
        .into_par_iter()
        .map(|bk| {
            let b0 = bk * BLK;
            let b1 = (b0 + BLK).min(n);
            let mut ov = vec![f32::NEG_INFINITY; (b1 - b0) * n];
            let mut oc = vec![-1i32; (b1 - b0) * n];
            let mut a = vec![0.0f32; n];
            let mut vbuf = vec![0.0f32; n];
            for b in b0..b1 {
                if !ok[b] {
                    continue;
                }
                let ob = &mut ov[(b - b0) * n..(b - b0) * n + n];
                let cb = &mut oc[(b - b0) * n..(b - b0) * n + n];
                for x in 0..n {
                    a[x] = m[x * n + b];
                }
                a[b] = f32::NEG_INFINITY; // X == B 不能算
                for c in (b + 1)..n {
                    if !ok[c] {
                        continue;
                    }
                    let w = wm[b * n + c];
                    let row = &mt[c * n..c * n + n];
                    let saved = a[c];
                    a[c] = f32::NEG_INFINITY; // C == X 不能算
                    crate::simd::mul_add(&mut vbuf, &a, row, w);
                    crate::simd::elem_max_upd_idx(ob, cb, &vbuf, c as i32);
                    a[c] = saved;
                }
            }
            (ov, oc)
        })
        .collect();
    let mut best = vec![f32::NEG_INFINITY; n];
    let mut bi = vec![-1i32; n];
    let mut ci = vec![-1i32; n];
    for (bk, (ov, oc)) in blocks.iter().enumerate() {
        let b0 = bk * BLK;
        let b1 = (b0 + BLK).min(n);
        for b in b0..b1 {
            if !ok[b] {
                continue;
            }
            let ob = &ov[(b - b0) * n..(b - b0) * n + n];
            let cb = &oc[(b - b0) * n..(b - b0) * n + n];
            for x in 0..n {
                if !ok[x] {
                    continue;
                }
                if ob[x] > best[x] {
                    best[x] = ob[x];
                    bi[x] = b as i32;
                    ci[x] = cb[x];
                }
            }
        }
    }
    eprintln!(
        "[reduce:p2] pair_scan 扫完 {} 对 {:.1}s",
        n * (n - 1) / 2,
        t0.elapsed().as_secs_f32()
    );
    (best, bi, ci)
}

/// 每只股票的最高单只 rho（对角与白名单外都置 -inf），并列取小下标。
pub(crate) fn best_single_of(n: usize, rho: &[f32], ok: &[bool]) -> Vec<f32> {
    let mut out = vec![f32::NEG_INFINITY; n];
    out.par_iter_mut().enumerate().for_each(|(i, o)| {
        if !ok[i] {
            return;
        }
        for j in 0..n {
            if j == i || !ok[j] {
                continue;
            }
            if rho[i * n + j] > *o {
                *o = rho[i * n + j];
            }
        }
    });
    out
}

// ============================================================
// R1：角色画像与最强搭档
// ============================================================

pub(crate) struct R1Role {
    pub(crate) count: Vec<f64>,
    pub(crate) count_z2: Vec<f64>,
    pub(crate) gain_mean: Vec<f64>,
    pub(crate) gain_max: Vec<f64>,
    pub(crate) z_mean: Vec<f64>,
    pub(crate) partner_z2: Vec<f64>,
}

/// pair_reduce.py 的 r1_role。把 G（或 GSTAR / Z / ZS）按「被选为 X0 的对」聚合。
pub(crate) fn r1_role(n: usize, x0: &[i32], pw: &[f32], pwz: &[f32], ok: &[bool]) -> R1Role {
    let mut count = vec![0.0f64; n];
    let mut sum_w = vec![0.0f64; n];
    let mut max_w = vec![f64::NEG_INFINITY; n];
    let mut sum_z = vec![0.0f64; n];
    let mut count_z2 = vec![0.0f64; n];
    let mut partner_z2 = vec![0.0f64; n];
    let mut n_good = 0usize;
    let zthr = soft().thr;
    for i in 0..n {
        if !ok[i] {
            continue;
        }
        for j in (i + 1)..n {
            if !ok[j] {
                continue;
            }
            // 作为伙伴出现在强互补对里：strong_pair 是上三角掩码，行数它的较小下标、
            // 列数它的较大下标，所以每只股票在一对里只被数到一次
            if pwz[i * n + j] as f64 > zthr {
                partner_z2[i] += 1.0;
                partner_z2[j] += 1.0;
            }
            let x = x0[i * n + j];
            if x < 0 || x as usize >= n {
                continue;
            }
            let x = x as usize;
            let w = pw[i * n + j] as f64;
            let z = pwz[i * n + j] as f64;
            count[x] += 1.0;
            sum_w[x] += w;
            if w > max_w[x] {
                max_w[x] = w;
            }
            sum_z[x] += z;
            if z > zthr {
                count_z2[x] += 1.0;
            }
            n_good += 1;
        }
    }
    if n_good == 0 {
        panic!("X0.bin 里没有一个合法的 X0");
    }
    let gain_mean: Vec<f64> = (0..n)
        .map(|x| if count[x] > 0.0 { sum_w[x] / count[x] } else { f64::NAN })
        .collect();
    let gain_max: Vec<f64> = (0..n)
        .map(|x| if count[x] > 0.0 { max_w[x] } else { f64::NAN })
        .collect();
    let z_mean: Vec<f64> = (0..n)
        .map(|x| if count[x] > 0.0 { sum_z[x] / count[x] } else { f64::NAN })
        .collect();
    R1Role {
        count,
        count_z2,
        gain_mean,
        gain_max,
        z_mean,
        partner_z2,
    }
}

pub(crate) struct R1Best {
    pub(crate) gain: Vec<f64>,
    pub(crate) z_of: Vec<f64>,
    pub(crate) mutual: Vec<f64>,
    pub(crate) rank: Vec<f64>,
}

/// pair_reduce.py 的 r1_bestcomp。最强搭档取权重矩阵行内最大（对角与白名单外置 -inf），
/// 互惠指两边的行内最大互指。
pub(crate) fn r1_bestcomp(n: usize, pw: &[f32], pwz: &[f32], ok: &[bool]) -> R1Best {
    let mut partner = vec![0usize; n];
    let mut gain = vec![f64::NAN; n];
    for i in 0..n {
        let mut bv = f32::NEG_INFINITY;
        let mut bp = 0usize;
        for j in 0..n {
            if j == i || !ok[i] || !ok[j] {
                continue;
            }
            if pw[i * n + j] > bv {
                bv = pw[i * n + j];
                bp = j;
            }
        }
        partner[i] = bp;
        gain[i] = if ok[i] { bv as f64 } else { f64::NAN };
    }
    let mut z_of = vec![f64::NAN; n];
    let mut mutual = vec![0.0f64; n];
    let mut rank = vec![0.0f64; n];
    let rows: Vec<(f64, f64, f64)> = (0..n)
        .into_par_iter()
        .map(|i| {
            if !ok[i] {
                return (f64::NAN, f64::NAN, f64::NAN);
            }
            let p = partner[i];
            let g = gain[i];
            let zz = pwz[i * n + p] as f64;
            let mut cnt = 0.0f64;
            for j in 0..n {
                // Python 那边 W 的对角与非个股列都置成 -inf，这里要照样跳过，
                // 否则 g 为负时会把对角那个 0 也数进去
                if j == p || !ok[j] {
                    continue;
                }
                if pw[p * n + j] as f64 > g {
                    cnt += 1.0;
                }
            }
            let mutual_v = if cnt == 0.0 { 1.0 } else { 0.0 };
            (zz, mutual_v, cnt)
        })
        .collect();
    for i in 0..n {
        z_of[i] = rows[i].0;
        mutual[i] = rows[i].1;
        rank[i] = rows[i].2;
    }
    R1Best {
        gain,
        z_of,
        mutual,
        rank,
    }
}

// ============================================================
// R2 三分解
// ============================================================

/// pair_reduce.py 的 r2_three_way。返回 (best_single, best_pair, pair_gain, gain_ratio, cross_industry)。
pub(crate) fn three_way(
    n: usize,
    single_f32: &[f32],
    best_pair: &[f32],
    bi: &[i32],
    ci: &[i32],
    industry: Option<&[i32]>,
) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let single: Vec<f64> = single_f32
        .iter()
        .map(|&v| if v.is_finite() { v as f64 } else { f64::NAN })
        .collect();
    let bp: Vec<f64> = best_pair
        .iter()
        .map(|&v| if v.is_finite() { v as f64 } else { f64::NAN })
        .collect();
    let bg: Vec<f64> = (0..n).map(|i| bp[i] - single[i]).collect();
    let ratio: Vec<f64> = (0..n)
        .map(|i| {
            if single[i].abs() > 1e-9 {
                bg[i] / single[i].abs()
            } else {
                f64::NAN
            }
        })
        .collect();
    let cross: Vec<f64> = match industry {
        None => vec![f64::NAN; n],
        Some(ind) => (0..n)
            .map(|x| {
                let ib = ind[wrap(bi[x], n)];
                let ic = ind[wrap(ci[x], n)];
                if ib >= 0 && ic >= 0 {
                    if ib != ic {
                        1.0
                    } else {
                        0.0
                    }
                } else {
                    f64::NAN
                }
            })
            .collect(),
    };
    (single, bp, bg, ratio, cross)
}

// ============================================================
// R3：子模贪心篮子
// ============================================================

pub(crate) struct R3 {
    pub(crate) marg: Vec<f64>,       // n × MAX_K，列优先存（marg[x*MAX_K+s]）
    pub(crate) n_to: Vec<f64>,
    pub(crate) basket_rho: Vec<f64>,
    pub(crate) avg_pr: Vec<f64>,
    pub(crate) hub: Vec<f64>,
    pub(crate) degen: Vec<f64>,
}

/// pair_reduce.py 的 r3_greedy。
/// rho[X][U_S] = (Σ_{Y∈S} se_Y rho[X][Y]) / sqrt(Q_x)，Q_x = Σ_{Y,Z∈S} se_Y se_Z rho[Y][Z]。
/// D[:,x] = rho @ c_x（c_x = se * 1_{S_x}），加入一只伙伴只要一次 O(N) 的列更新。
/// 这里把 D 转置成 Dt[x][Y] 存，让每一步的候选值扫描与列更新都是顺序读。
pub(crate) fn r3_greedy(n: usize, rho: &[f32], e: &[f64], ok: &[bool]) -> R3 {
    let t0 = Instant::now();
    let se: Vec<f64> = (0..n).map(|i| e[i].max(1e-30).sqrt()).collect();
    let r: Vec<f64> = rho.iter().map(|&v| v as f64).collect();
    let rt = transpose_f64(&r, n);
    let mut dt = vec![0.0f64; n * n]; // Dt[x*n+Y] = D[Y,x]
    let mut q = vec![0.0f64; n];
    let mut is_in = vec![false; n * n];
    let mut picks = vec![0usize; n * MAX_K];
    let mut marg = vec![f64::NAN; n * MAX_K];
    let mut rs = vec![0.0f64; n * (MAX_K + 1)];
    for step in 0..MAX_K {
        // 候选值：val[Y,x] = (num_x + se_Y R[X][Y]) / sqrt(Q_x + 2 se_Y D[Y,x] + E_Y)
        let picked: Vec<usize> = (0..n)
            .into_par_iter()
            .map(|x| {
                let qx = q[x];
                let num = dt[x * n + x];
                let rx = &r[x * n..x * n + n];
                let dx = &dt[x * n..x * n + n];
                let inn = &is_in[x * n..x * n + n];
                let mut bestv = f64::NEG_INFINITY;
                let mut pick = 0usize; // numpy argmax 全 -inf 时给 0
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
            picks[x * MAX_K + step] = picked[x];
        }
        // Q 用更新前的 D 更新
        let q_new: Vec<f64> = (0..n)
            .into_par_iter()
            .map(|x| {
                let p = picked[x];
                q[x] + 2.0 * se[p] * dt[x * n + p] + e[p]
            })
            .collect();
        // D 的列更新：D[:,x] += se_pick[x] * R[:, pick[x]]
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
        let (rho_new, marg_step) = {
            let mut rho_new = vec![0.0f64; n];
            let mut marg_step = vec![0.0f64; n];
            for x in 0..n {
                let num_new = dt[x * n + x];
                let v = num_new / q[x].max(1e-30).sqrt();
                rho_new[x] = v;
                marg_step[x] = v - rs[x * (MAX_K + 1) + step];
            }
            (rho_new, marg_step)
        };
        for x in 0..n {
            rs[x * (MAX_K + 1) + step + 1] = rho_new[x];
            marg[x * MAX_K + step] = marg_step[x];
        }
    }
    // 增益降到首步 10% 以下所需只数（首次命中即停，与 Python 的循环等价）
    let n_to: Vec<f64> = (0..n)
        .map(|x| {
            let thr = marg[x * MAX_K].abs() * 0.1;
            let mut v = (MAX_K + 1) as f64;
            for s in 0..MAX_K {
                if marg[x * MAX_K + s] < thr {
                    v = (s + 1) as f64;
                    break;
                }
            }
            v
        })
        .collect();
    let basket_rho: Vec<f64> = (0..n).map(|x| rs[x * (MAX_K + 1) + MAX_K]).collect();
    // 伙伴两两之间的平均 rho 与最大 rho（按行主序摊开，与 numpy 的布尔掩码顺序一致）
    let stats: Vec<(f64, f64)> = (0..n)
        .into_par_iter()
        .map(|x| {
            let s = &picks[x * MAX_K..x * MAX_K + MAX_K];
            if MAX_K < 2 {
                return (f64::NAN, f64::NAN);
            }
            let mut sum = 0.0f64;
            let mut cnt = 0usize;
            let mut mx = f64::NEG_INFINITY;
            for i in 0..MAX_K {
                for j in 0..MAX_K {
                    if i == j {
                        continue;
                    }
                    let v = r[s[i] * n + s[j]];
                    sum += v;
                    cnt += 1;
                    if v > mx {
                        mx = v;
                    }
                }
            }
            (sum / cnt as f64, mx)
        })
        .collect();
    let avg_pr: Vec<f64> = stats.iter().map(|t| t.0).collect();
    let hub: Vec<f64> = stats.iter().map(|t| t.1).collect();
    let degen: Vec<f64> = avg_pr
        .iter()
        .map(|&v| if v > 0.5 { 1.0 } else { 0.0 })
        .collect();
    eprintln!("[reduce:p3] R3 贪心 {} 步 {:.1}s", MAX_K, t0.elapsed().as_secs_f32());
    R3 {
        marg,
        n_to,
        basket_rho,
        avg_pr,
        hub,
        degen,
    }
}

// ============================================================
// R4：互补图 vs 相似图
// ============================================================

/// 每只股票取前 k 条边（自身与白名单外直接排除）做成无向图。
/// 非个股行不填（Python 那边填完也会被清掉，所以取法与它们无关）。
///
/// 关于并列。pair_reduce.py 用 np.argpartition 取前 k，第 k 名与第 k+1 名的值完全相等时
/// numpy 留下哪一个是不指定的：它走的是 x86-simd-sort 的 AVX512 ArgQSelect（float32 会命中这条
/// 分发），取舍由向量分区的内部顺序决定。实测口径：在 20241231 的真实 G 行上人为造 600 个
/// 边界并列，numpy 取较小下标 47%、取较大下标 51%（另有 2% 两个都进），也就是一枚硬币；
/// （20241230/20241231 六个矩阵一共只有 2 个真实边界并列，numpy 两次都取了较大的下标。）
/// 这里定死一条：按值降序，值相等时取**下标较大**的，与那两天参考输出的取舍一致。
/// 两天的真实数据上按这条规则取到的边集与 numpy 逐个相同；换一天遇到并列，可能差 1 条边
/// （影响 r4 的四个列各一两格，量级是 1 或 1/20）。
pub(crate) fn topk_adj(n: usize, sq: &[f32], k: usize, ok: &[bool]) -> Vec<bool> {
    let rows: Vec<Vec<usize>> = (0..n)
        .into_par_iter()
        .map(|i| {
            if !ok[i] {
                return Vec::new();
            }
            let mut cand: Vec<(f32, usize)> = Vec::with_capacity(n);
            for j in 0..n {
                if j == i || !ok[j] {
                    continue;
                }
                cand.push((sq[i * n + j], j));
            }
            let kk = k.min(cand.len());
            if kk == 0 {
                return Vec::new();
            }
            if kk < cand.len() {
                cand.select_nth_unstable_by(kk - 1, |a, b| {
                    b.0.total_cmp(&a.0).then(b.1.cmp(&a.1))
                });
                cand.truncate(kk);
            }
            cand.into_iter().map(|(_, j)| j).collect()
        })
        .collect();
    let mut adj = vec![false; n * n];
    for i in 0..n {
        for &j in &rows[i] {
            adj[i * n + j] = true;
            adj[j * n + i] = true;
        }
    }
    adj
}

/// topk_adj 的多 k 版本：每行 (score,下标) 打包成 u64 键只排一次序，
/// 多个 k 档共享排序结果取前缀——R4 要同时出 5 档图时比逐档跑快 ~5 倍。
/// 排序序与单 k 版完全一致（分数降序、同分下标大者靠前；fkey 的整型序
/// 与 total_cmp 一致，+NaN 排最前、-NaN 排最后，与旧版相同）。
/// 返回 adj[ki][i*n+j]，与 topk_adj 同样写成无向邻接。
pub(crate) fn topk_adj_multi(
    n: usize,
    sq: &[f32],
    ks: &[usize],
    ok: &[bool],
) -> Vec<Vec<bool>> {
    // 每行的 top-maxk 邻居下标（按序），前缀即各档答案
    let kmax = *ks.iter().max().unwrap_or(&0);
    let rows: Vec<Vec<u32>> = (0..n)
        .into_par_iter()
        .map(|i| {
            if !ok[i] {
                return Vec::new();
            }
            let row = &sq[i * n..i * n + n];
            let mut keys: Vec<u64> = Vec::with_capacity(n);
            for j in 0..n {
                if j == i || !ok[j] {
                    continue;
                }
                keys.push(crate::simd::pack_desc_idx_desc(row[j], j as u32));
            }
            let kk = kmax.min(keys.len());
            if kk == 0 {
                return Vec::new();
            }
            if kk < keys.len() {
                keys.select_nth_unstable(kk - 1);
                keys.truncate(kk);
            }
            keys.sort_unstable();
            keys.into_iter().map(|k| !k as u32).collect()
        })
        .collect();
    let mut out = vec![vec![false; n * n]; ks.len()];
    for (ki, &k) in ks.iter().enumerate() {
        let adj = &mut out[ki];
        for i in 0..n {
            for &j in rows[i].iter().take(k) {
                let j = j as usize;
                adj[i * n + j] = true;
                adj[j * n + i] = true;
            }
        }
    }
    out
}

pub(crate) struct R4 {
    pub(crate) deg_rho: Vec<f64>,
    pub(crate) deg_g: Vec<f64>,
    pub(crate) overlap: Vec<f64>,
    pub(crate) jaccard: Vec<f64>,
    pub(crate) quad: Vec<f64>,
}

/// pair_reduce.py 的 r4_graphs。只算进 factors.bin 的 5 列（rho 图与 G 图的关系）；
/// 社群对比（label_propagation / ARI）、边集对照、度序列 Spearman 只进 diag.json，没搬，
/// 所以 Z 图与 ZS/GS 图的 top-k 邻接在这里根本不需要建。
pub(crate) fn r4_graphs(n: usize, rho: &[f32], g: &[f32], ok: &[bool]) -> R4 {
    let t0 = Instant::now();
    let n_ok = ok.iter().filter(|&&v| v).count();
    let k = K_TOPK.min(n_ok.saturating_sub(1));
    let a_rho = topk_adj(n, rho, k, ok);
    let a_g = topk_adj(n, g, k, ok);
    let deg_rho: Vec<f64> = (0..n)
        .map(|i| (0..n).filter(|&j| a_rho[i * n + j]).count() as f64)
        .collect();
    let deg_g: Vec<f64> = (0..n)
        .map(|i| (0..n).filter(|&j| a_g[i * n + j]).count() as f64)
        .collect();
    let inter: Vec<f64> = (0..n)
        .map(|i| (0..n).filter(|&j| a_rho[i * n + j] && a_g[i * n + j]).count() as f64)
        .collect();
    let union: Vec<f64> = (0..n)
        .map(|i| (0..n).filter(|&j| a_rho[i * n + j] || a_g[i * n + j]).count() as f64)
        .collect();
    let overlap: Vec<f64> = inter.iter().map(|&v| v / k as f64).collect();
    let jaccard: Vec<f64> = (0..n)
        .map(|i| {
            if union[i] > 0.0 {
                inter[i] / union[i].max(1.0)
            } else {
                f64::NAN
            }
        })
        .collect();
    let deg_rho_ok: Vec<f64> = (0..n).filter(|&i| ok[i]).map(|i| deg_rho[i]).collect();
    let deg_g_ok: Vec<f64> = (0..n).filter(|&i| ok[i]).map(|i| deg_g[i]).collect();
    let med_rho = nanmedian(&deg_rho_ok);
    let med_g = nanmedian(&deg_g_ok);
    let quad: Vec<f64> = (0..n)
        .map(|i| {
            let a = if deg_rho[i] > med_rho { 2.0 } else { 0.0 };
            let b = if deg_g[i] > med_g { 1.0 } else { 0.0 };
            a + b
        })
        .collect();
    eprintln!("[reduce:p4] R4 图对照 {:.1}s", t0.elapsed().as_secs_f32());
    R4 {
        deg_rho,
        deg_g,
        overlap,
        jaccard,
        quad,
    }
}

// ============================================================
// R5：残差互补
// ============================================================

pub(crate) struct R5 {
    pub(crate) res_max: Vec<f64>,
    pub(crate) res_top20: Vec<f64>,
    pub(crate) res_mean: Vec<f64>,
    pub(crate) res_hhi: Vec<f64>,
}

/// 5×5 线性方程，列主元高斯消元。
pub(crate) fn solve5(mut a: [[f64; 5]; 5], mut b: [f64; 5]) -> [f64; 5] {
    for col in 0..5 {
        let mut piv = col;
        for r in (col + 1)..5 {
            if a[r][col].abs() > a[piv][col].abs() {
                piv = r;
            }
        }
        if piv != col {
            a.swap(piv, col);
            b.swap(piv, col);
        }
        let d = a[col][col];
        for r in (col + 1)..5 {
            let f = a[r][col] / d;
            if f == 0.0 {
                continue;
            }
            for c in col..5 {
                a[r][c] -= f * a[col][c];
            }
            b[r] -= f * b[col];
        }
    }
    let mut x = [0.0f64; 5];
    for col in (0..5).rev() {
        let mut s = b[col];
        for c in (col + 1)..5 {
            s -= a[col][c] * x[c];
        }
        x[col] = s / a[col][col];
    }
    x
}

/// 行统计：最大值 / 前 20 均值 / 均值 / 正残差 HHI，全在「有限值」上做。
pub(crate) fn row_stats(row: &[f32]) -> (f64, f64, f64, f64) {
    let fin: Vec<f32> = row.iter().copied().filter(|v| v.is_finite()).collect();
    if fin.is_empty() {
        return (f64::NAN, f64::NAN, f64::NAN, f64::NAN);
    }
    let mx = fin.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let mut top = fin.clone();
    let k = K_TOPK.min(top.len());
    if k < top.len() {
        let nth = top.len() - k;
        top.select_nth_unstable_by(nth, |a, b| a.total_cmp(b));
    }
    let t20 = top[top.len() - k..].iter().map(|&v| v as f64).sum::<f64>() / k as f64;
    let mean = fin.iter().map(|&v| v as f64).sum::<f64>() / fin.len() as f64;
    let pos: Vec<f64> = fin
        .iter()
        .filter(|&&v| v > 0.0)
        .map(|&v| v as f64)
        .collect();
    let hhi = if pos.is_empty() {
        f64::NAN
    } else {
        let s = pos.iter().sum::<f64>();
        pos.iter().map(|v| {
            let w = v / s;
            w * w
        })
        .sum::<f64>()
    };
    (mx, t20, mean, hhi)
}

/// pair_reduce.py 的 r5_residual。把 wmat（G / GSTAR / Z / ZS）对
/// (1, rho[B][C], log 成交笔数 B, log 成交笔数 C, |两者之差|) 回归，取残差的行内统计。
pub(crate) fn r5_residual(n: usize, wmat: &[f32], rho: &[f32], lnt: &[f64], ok: &[bool]) -> R5 {
    let t0 = Instant::now();
    let med = nanmedian(lnt);
    let ln: Vec<f64> = lnt
        .iter()
        .map(|&v| if v.is_finite() { v } else { med })
        .collect();
    // 第一趟：XtX 与 Xty
    let mut xtx = [[0.0f64; 5]; 5];
    let mut xty = [0.0f64; 5];
    let mut npair = 0usize;
    for i in 0..n {
        if !ok[i] {
            continue;
        }
        for j in (i + 1)..n {
            if !ok[j] {
                continue;
            }
            let r = rho[i * n + j] as f64;
            let lb = ln[i];
            let lc = ln[j];
            let d = [1.0, r, lb, lc, (lb - lc).abs()];
            let y = wmat[i * n + j] as f64;
            for a in 0..5 {
                for b in 0..5 {
                    xtx[a][b] += d[a] * d[b];
                }
                xty[a] += d[a] * y;
            }
            npair += 1;
        }
    }
    if npair == 0 {
        panic!("R5 没有可回归的配对");
    }
    let mut a = xtx;
    for i in 0..5 {
        a[i][i] += 1e-8;
    }
    let beta = solve5(a, xty);
    // 第二趟：残差，上三角填好再对称化（下三角照抄上三角，对角留空）
    let mut res = vec![f32::NAN; n * n];
    let mut cnt_fin = vec![0usize; n];
    for i in 0..n {
        if !ok[i] {
            continue;
        }
        for j in (i + 1)..n {
            if !ok[j] {
                continue;
            }
            let r = rho[i * n + j] as f64;
            let lb = ln[i];
            let lc = ln[j];
            let y = wmat[i * n + j] as f64;
            let v = (y
                - (beta[0] + beta[1] * r + beta[2] * lb + beta[3] * lc + beta[4] * (lb - lc).abs()))
                as f32;
            res[i * n + j] = v;
            res[j * n + i] = v;
            cnt_fin[i] += 1;
            cnt_fin[j] += 1;
        }
    }
    let stats: Vec<(f64, f64, f64, f64)> = (0..n)
        .into_par_iter()
        .map(|i| {
            if cnt_fin[i] == 0 {
                (f64::NAN, f64::NAN, f64::NAN, f64::NAN)
            } else {
                row_stats(&res[i * n..i * n + n])
            }
        })
        .collect();
    eprintln!("[reduce:p5] R5 残差回归 {:.1}s", t0.elapsed().as_secs_f32());
    R5 {
        res_max: stats.iter().map(|t| t.0).collect(),
        res_top20: stats.iter().map(|t| t.1).collect(),
        res_mean: stats.iter().map(|t| t.2).collect(),
        res_hhi: stats.iter().map(|t| t.3).collect(),
    }
}

// ============================================================
// R6：(Z, ZS) 二维平面
// ============================================================

pub(crate) struct R6 {
    pub(crate) gen_max: Vec<f64>,
    pub(crate) gen_top20: Vec<f64>,
    pub(crate) spec_max: Vec<f64>,
    pub(crate) spec_top20: Vec<f64>,
    pub(crate) zs_max: Vec<f64>,
    pub(crate) row_corr: Vec<f64>,
    pub(crate) spec_share: Vec<f64>,
    pub(crate) quad: Vec<f64>,
}

/// pair_reduce.py 的 r6_plane。
/// gen 侧用严格上三角（Zrow 与 ZSrow 只填上三角，所以最后一行的这四个数是空的）；
/// spec 侧用对称阵 Sc（对角填 0，所以 spec_max 下界是 0）。
pub(crate) fn r6_plane(n: usize, z: &[f32], zs: &[f32], ok: &[bool]) -> R6 {
    let t0 = Instant::now();
    // 每个配对的 (spec, 是否专才对) 与参与配对的次数
    let mut sc = vec![0.0f32; n * n]; // 对称，对角 0
    let mut sp = vec![0.0f32; n * n];
    let mut cnt_valid = vec![0.0f64; n];
    let zthr = soft().thr;
    for i in 0..n {
        if !ok[i] {
            continue;
        }
        for j in (i + 1)..n {
            if !ok[j] {
                continue;
            }
            let zl = z[i * n + j] as f64;
            let zsl = zs[i * n + j] as f64;
            let s = (zsl - zl) as f32;
            let p = if zsl > zthr && zl < -zthr { 1.0f32 } else { 0.0 };
            sc[i * n + j] = s;
            sc[j * n + i] = s;
            sp[i * n + j] = p;
            sp[j * n + i] = p;
            cnt_valid[i] += 1.0;
            cnt_valid[j] += 1.0;
        }
    }
    // 逐行：gen 侧只取 j > i，spec 侧取整行（含对角那个 0）
    let per_row: Vec<(f64, f64, f64, f64, f64, f64, f64)> = (0..n)
        .into_par_iter()
        .map(|i| {
            // gen（上三角、有效对）：Z 与 ZS 各一份
            let mut gv: Vec<f64> = Vec::new();
            let mut zv: Vec<f64> = Vec::new();
            if ok[i] {
                for j in (i + 1)..n {
                    if !ok[j] {
                        continue;
                    }
                    gv.push(z[i * n + j] as f64);
                    zv.push(zs[i * n + j] as f64);
                }
            }
            let gen_max = if gv.is_empty() {
                f64::NAN
            } else {
                gv.iter().copied().fold(f64::NEG_INFINITY, f64::max)
            };
            let gen_top20 = if gv.is_empty() {
                f64::NAN
            } else {
                let mut t = gv.clone();
                let k = K_TOPK.min(t.len());
                if k < t.len() {
                    let nth = t.len() - k;
                    t.select_nth_unstable_by(nth, |a, b| a.total_cmp(b));
                }
                t[t.len() - k..].iter().sum::<f64>() / k as f64
            };
            let zs_max = if zv.is_empty() {
                f64::NAN
            } else {
                zv.iter().copied().fold(f64::NEG_INFINITY, f64::max)
            };
            let row_corr = if gv.len() > 30 {
                spearman(&gv, &zv, 30)
            } else {
                f64::NAN
            };
            // spec：整行 n 个值（对角 0，无效位置 0）
            let mut spec_max = 0.0f64; // 对角那个 0 一定在里面
            let mut vals: Vec<f64> = Vec::with_capacity(n);
            for j in 0..n {
                if j == i {
                    vals.push(0.0);
                    continue;
                }
                let v = sc[i * n + j] as f64;
                vals.push(v);
                if v > spec_max {
                    spec_max = v;
                }
            }
            let mut t = vals;
            let k = K_TOPK.min(t.len());
            if k < t.len() {
                let nth = t.len() - k;
                t.select_nth_unstable_by(nth, |a, b| a.total_cmp(b));
            }
            let spec_top20 = t[t.len() - k..].iter().sum::<f64>() / k as f64;
            let share = if cnt_valid[i] > 0.0 {
                let s: f64 = (0..n).map(|j| sp[i * n + j] as f64).sum();
                s / cnt_valid[i].max(1.0)
            } else {
                f64::NAN
            };
            (gen_max, gen_top20, spec_max, spec_top20, zs_max, row_corr, share)
        })
        .collect();
    let gen_max: Vec<f64> = per_row.iter().map(|t| t.0).collect();
    let gen_top20: Vec<f64> = per_row.iter().map(|t| t.1).collect();
    let spec_max: Vec<f64> = per_row.iter().map(|t| t.2).collect();
    let spec_top20: Vec<f64> = per_row.iter().map(|t| t.3).collect();
    let zs_max: Vec<f64> = per_row.iter().map(|t| t.4).collect();
    let row_corr: Vec<f64> = per_row.iter().map(|t| t.5).collect();
    let spec_share: Vec<f64> = per_row.iter().map(|t| t.6).collect();
    let mg = nanmedian(&gen_max);
    let ms = nanmedian(&zs_max);
    let quad: Vec<f64> = (0..n)
        .map(|i| {
            let a = if gen_max[i] > mg { 2.0 } else { 0.0 };
            let b = if zs_max[i] > ms { 1.0 } else { 0.0 };
            a + b
        })
        .collect();
    eprintln!("[reduce:p6] R6 平面 {:.1}s", t0.elapsed().as_secs_f32());
    R6 {
        gen_max,
        gen_top20,
        spec_max,
        spec_top20,
        zs_max,
        row_corr,
        spec_share,
        quad,
    }
}

// ============================================================
// 列装配与输出
// ============================================================

pub(crate) struct Out {
    pub(crate) names: Vec<String>,
    pub(crate) cols: Vec<Vec<f64>>,
}

impl Out {
    pub(crate) fn new() -> Self {
        Out {
            names: Vec::new(),
            cols: Vec::new(),
        }
    }
    pub(crate) fn push(&mut self, name: &str, col: Vec<f64>) {
        self.names.push(name.to_string());
        self.cols.push(col);
    }
}

// ============================================================
// 单矩阵输入包（net 一个 --mat 的整套产物）
// ============================================================

pub(crate) struct MatSet {
    pub rho: MatData,
    pub e: Vec<f64>,
    pub g: MatData,
    pub z: MatData,
    pub gs: MatData,
    pub zs: MatData,
    pub x0: Vec<i32>,
    pub x0s: Vec<i32>,
    pub ru: MatData,
    pub rus: MatData,
    pub sg: MatData,
    pub sgs: MatData,
    pub grel: MatData,
    pub gsh: MatData,
    pub gcov: MatData,
    pub gnpos: MatData,
    pub ghhi: MatData,
    pub pb: MatData,
    pub qc: MatData,
    pub pbs: MatData,
    pub qcs: MatData,
    pub exz: MatData,
    pub exzs: MatData,
    pub sc: MatData,   // ZS - Z
    pub mass: MatData, // 6n：emass|gmass|g2mass|smass|sgmass|sg2mass（旧4n自动补NaN）
}

/// 装载一张矩阵：文件存在且长度正确时用只读 mmap 零拷贝挂入；
/// 缺失/坏文件用全 NaN owned Vec 兜底，让下游列变全空而不是崩掉。
fn load_mat_data(day: &Path, n: usize, nm: &str) -> MatData {
    let p = day.join(format!("{nm}.bin"));
    match mmap_f32(&p) {
        Ok(m) if m.as_f32().len() == n * n => MatData::Map(m),
        Ok(m) => {
            eprintln!("[reduce] {} 长度 {} != n*n={}，按缺失处理", p.display(), m.as_f32().len(), n * n);
            MatData::Own(vec![f32::NAN; n * n])
        }
        Err(_) => MatData::Own(vec![f32::NAN; n * n]),
    }
}

pub(crate) fn load_matset(day: &Path, n: usize, tag: &str) -> MatSet {
    // 22 张矩阵 mmap 零拷贝挂入：不再分配 22×62MB 用户缓冲，
    // 也没有内核 copy_to_user——mmap 调用本身 O(微秒)，
    // 页 fault-in 摊到后续并行计算里（共享 page cache 页）。
    const MATS: [&str; 21] = [
        "rho", "G", "Z", "GSTAR", "ZS", "X0", "X0STAR", "RHO_U", "RHO_US",
        "SINGLE", "SINGLE_S", "GREL", "GSH", "GCOV", "GNPOS", "GHHI", "PB",
        "QC", "PBS", "QCS", "EXZ",
    ];
    let loaded: Vec<MatData> = MATS
        .iter()
        .map(|nm| load_mat_data(day, n, &format!("{nm}{tag}")))
        .collect();
    let exzs = load_mat_data(day, n, &format!("EXZS{tag}"));
    let e = {
        let p = day.join(format!("E{tag}.bin"));
        let raw = std::fs::read(&p).expect(&format!("读 {} 失败", p.display()));
        if raw.len() != n * 8 {
            panic!("{} 长度不对: {} 字节, 期望 {}", p.display(), raw.len(), n * 8);
        }
        (0..n)
            .map(|i| f64::from_le_bytes(raw[i * 8..i * 8 + 8].try_into().unwrap()))
            .collect::<Vec<f64>>()
    };
    let mut it = loaded.into_iter();
    let rho = it.next().unwrap();
    let g = it.next().unwrap();
    let z = it.next().unwrap();
    let gs = it.next().unwrap();
    let zs = it.next().unwrap();
    let x0 = read_x0(&it.next().unwrap()[..], n);
    let x0s = read_x0(&it.next().unwrap()[..], n);
    let ru = it.next().unwrap();
    let rus = it.next().unwrap();
    let sg = it.next().unwrap();
    let sgs = it.next().unwrap();
    let grel = it.next().unwrap();
    let gsh = it.next().unwrap();
    let gcov = it.next().unwrap();
    let gnpos = it.next().unwrap();
    let ghhi = it.next().unwrap();
    let pb = it.next().unwrap();
    let qc = it.next().unwrap();
    let pbs = it.next().unwrap();
    let qcs = it.next().unwrap();
    let exz = it.next().unwrap();
    // Z / ZS：上下三角不一致时取平均对称化（真实数据这两份是一致的）。
    // mmap 是只读映射，需对称化时复制成 owned 再改。
    let z = if max_abs_diff_to_transpose(&z[..], n) > 1e-6 {
        let mut zv = z.to_vec();
        symmetrize_f32(&mut zv, n);
        MatData::Own(zv)
    } else {
        z
    };
    let zs = if max_abs_diff_to_transpose(&zs[..], n) > 1e-6 {
        let mut zv = zs.to_vec();
        symmetrize_f32(&mut zv, n);
        MatData::Own(zv)
    } else {
        zs
    };
    // sc 构建保持串行：load_matset 在外层多矩阵 par 循环内，
    // 嵌套 par 在 50 线程池上只会排队（任务级并行内不嵌套）。
    let mut sc = vec![f32::NAN; n * n];
    for i in 0..n {
        for j in (i + 1)..n {
            let (a, b) = (zs[i * n + j], z[i * n + j]);
            if a.is_finite() && b.is_finite() {
                sc[i * n + j] = a - b;
            }
        }
    }
    for i in 0..n {
        for j in 0..i {
            sc[i * n + j] = sc[j * n + i];
        }
    }
    // MASS 统一成 6×n 布局 emass|gmass|g2mass|smass|sgmass|sg2mass；
    // 旧输出只有 4×n（没有二阶矩），把 g2/sg2 两段填 NaN，下游软 std 自然落空。
    let mass = {
        let p = day.join(format!("MASS{tag}.bin"));
        match mmap_f32(&p) {
            Ok(m) if m.as_f32().len() == 6 * n => MatData::Map(m),
            Ok(m) if m.as_f32().len() == 4 * n => {
                let v = m.as_f32();
                let mut mv = vec![f32::NAN; 6 * n];
                mv[..2 * n].copy_from_slice(&v[..2 * n]); // emass, gmass
                mv[3 * n..5 * n].copy_from_slice(&v[2 * n..4 * n]); // smass, sgmass
                MatData::Own(mv)
            }
            _ => MatData::Own(vec![f32::NAN; 6 * n]),
        }
    };
    MatSet {
        rho,
        e,
        g,
        z,
        gs,
        zs,
        x0,
        x0s,
        ru,
        rus,
        sg,
        sgs,
        grel,
        gsh,
        gcov,
        gnpos,
        ghhi,
        pb,
        qc,
        pbs,
        qcs,
        exz,
        exzs,
        sc: MatData::Own(sc),
        mass,
    }
}

/// 一个矩阵全套的因子合成：legacy 77 列（逐字搬旧版）+ reduce2 扩展列。
fn produce_one_mat(
    ms: &MatSet,
    pre: &str,
    n: usize,
    ok: &[bool],
    industry: Option<&[i32]>,
    lnt: Option<&[f64]>,
    has_lnt: bool,
    out: &mut Out,
) {
    let (rho, e, g, gs) = (&ms.rho, &ms.e, &ms.g, &ms.gs);
    let (z, zs, x0, x0s) = (&ms.z, &ms.zs, &ms.x0, &ms.x0s);
    let p = |nm: &str| format!("{pre}{nm}");

    // ---------------- R1 第一路 ----------------
    let t0 = Instant::now();
    let r1 = r1_role(n, x0, g, z, ok);
    let bc = r1_bestcomp(n, g, z, ok);
    out.push(&p("r1_explain_count"), r1.count.clone());
    out.push(&p("r1_explain_count_z2"), r1.count_z2.clone());
    out.push(&p("r1_explain_gain_mean"), r1.gain_mean.clone());
    out.push(&p("r1_explain_gain_max"), r1.gain_max.clone());
    out.push(&p("r1_explain_z_mean"), r1.z_mean.clone());
    out.push(&p("r1_as_partner_z2_count"), r1.partner_z2.clone());
    out.push(&p("r1_bestcomp_gain"), bc.gain.clone());
    out.push(&p("r1_bestcomp_z"), bc.z_of.clone());
    out.push(&p("r1_bestcomp_mutual"), bc.mutual.clone());
    out.push(&p("r1_bestcomp_rank"), bc.rank.clone());
    eprintln!("[reduce:p7] R1 第一路 {:.1}s", t0.elapsed().as_secs_f32());

    // ---------------- R1 第二路（列名按 Python 的 sorted(s1.keys()) 排序）----------------
    let t0 = Instant::now();
    let s1 = r1_role(n, x0s, gs, zs, ok);
    let sbc = r1_bestcomp(n, gs, zs, ok);
    out.push(&p("s_r1_as_partner_z2_count"), s1.partner_z2.clone());
    out.push(&p("s_r1_bestcomp_gain"), sbc.gain.clone());
    out.push(&p("s_r1_bestcomp_mutual"), sbc.mutual.clone());
    out.push(&p("s_r1_bestcomp_rank"), sbc.rank.clone());
    out.push(&p("s_r1_bestcomp_z"), sbc.z_of.clone());
    out.push(&p("s_r1_explain_count"), s1.count.clone());
    out.push(&p("s_r1_explain_count_z2"), s1.count_z2.clone());
    out.push(&p("s_r1_explain_gain_max"), s1.gain_max.clone());
    out.push(&p("s_r1_explain_gain_mean"), s1.gain_mean.clone());
    out.push(&p("s_r1_explain_z_mean"), s1.z_mean.clone());
    eprintln!("[reduce:p8] R1 第二路 {:.1}s", t0.elapsed().as_secs_f32());

    // ---------------- R2 ----------------
    let t0 = Instant::now();
    let (best_pair, b_i, c_i) = pair_scan(n, rho, e, ok);
    let single = best_single_of(n, rho, ok);
    let (r2_single, r2_pair, r2_gain, r2_ratio, r2_cross) =
        three_way(n, &single, &best_pair, &b_i, &c_i, industry);
    out.push(&p("r2_best_single"), r2_single);
    out.push(&p("r2_best_pair"), r2_pair);
    out.push(&p("r2_pair_gain"), r2_gain);
    out.push(&p("r2_pair_gain_ratio"), r2_ratio);
    out.push(&p("r2_best_pair_cross_industry"), r2_cross);
    out.push(&p("r2_best_pair_Bidx"), b_i.iter().map(|&v| v as f64).collect());
    out.push(&p("r2_best_pair_Cidx"), c_i.iter().map(|&v| v as f64).collect());
    eprintln!("[reduce:p9] R2 三分解 {:.1}s", t0.elapsed().as_secs_f32());

    // ---------------- R3 ----------------
    let r3 = r3_greedy(n, rho, e, ok);
    for s in 0..4 {
        out.push(
            &p(&format!("r3_marg{}", s + 1)),
            (0..n).map(|x| r3.marg[x * MAX_K + s]).collect(),
        );
    }
    out.push(&p("r3_n_to_10pct"), r3.n_to.clone());
    out.push(&p("r3_basket_rho8"), r3.basket_rho.clone());
    out.push(&p("r3_avg_partner_rho"), r3.avg_pr.clone());
    out.push(&p("r3_max_partner_rho"), r3.hub.clone());
    out.push(&p("r3_degenerate"), r3.degen.clone());

    // ---------------- R4 ----------------
    let r4 = r4_graphs(n, rho, g, ok);
    out.push(&p("r4_deg_rho_topk"), r4.deg_rho.clone());
    out.push(&p("r4_deg_G_topk"), r4.deg_g.clone());
    out.push(&p("r4_neighbor_overlap_k"), r4.overlap.clone());
    out.push(&p("r4_neighbor_jaccard"), r4.jaccard.clone());
    out.push(&p("r4_quad"), r4.quad.clone());

    // ---------------- R5 两路 ----------------
    if has_lnt {
        let lntv = lnt.unwrap();
        let r5 = r5_residual(n, g, rho, lntv, ok);
        out.push(&p("r5_res_max"), r5.res_max);
        out.push(&p("r5_res_top20_mean"), r5.res_top20);
        out.push(&p("r5_res_mean"), r5.res_mean);
        out.push(&p("r5_res_hhi"), r5.res_hhi);
        let s5 = r5_residual(n, gs, rho, lntv, ok);
        out.push(&p("s_r5_res_hhi"), s5.res_hhi);
        out.push(&p("s_r5_res_max"), s5.res_max);
        out.push(&p("s_r5_res_mean"), s5.res_mean);
        out.push(&p("s_r5_res_top20_mean"), s5.res_top20);
    } else {
        eprintln!("[reduce] 没有成交笔数，跳过 r5_/s_r5_/null_r5_/null_s_r5_ 列");
    }

    // ---------------- R6 ----------------
    let r6 = r6_plane(n, z, zs, ok);
    out.push(&p("p_gen_max"), r6.gen_max);
    out.push(&p("p_gen_top20"), r6.gen_top20);
    out.push(&p("p_quad"), r6.quad);
    out.push(&p("p_row_corr"), r6.row_corr);
    out.push(&p("p_spec_max"), r6.spec_max);
    out.push(&p("p_spec_share"), r6.spec_share);
    out.push(&p("p_spec_top20"), r6.spec_top20);
    out.push(&p("p_zs_max"), r6.zs_max);

    // ---------------- 空模型对照（G -> Z，GSTAR -> ZS）----------------
    let t0 = Instant::now();
    let n1 = r1_role(n, x0, z, z, ok);
    let nbc = r1_bestcomp(n, z, z, ok);
    let ns1 = r1_role(n, x0s, zs, zs, ok);
    let nsbc = r1_bestcomp(n, zs, zs, ok);
    out.push(&p("null_r1_bestcomp_gain"), nbc.gain.clone());
    out.push(&p("null_r1_bestcomp_mutual"), nbc.mutual.clone());
    out.push(&p("null_r1_bestcomp_rank"), nbc.rank.clone());
    out.push(&p("null_r1_explain_gain_max"), n1.gain_max.clone());
    out.push(&p("null_r1_explain_gain_mean"), n1.gain_mean.clone());
    out.push(&p("null_r1_explain_z_mean"), n1.z_mean.clone());
    if has_lnt {
        let lntv = lnt.unwrap();
        let zr5 = r5_residual(n, z, rho, lntv, ok);
        out.push(&p("null_r5_res_hhi"), zr5.res_hhi);
        out.push(&p("null_r5_res_max"), zr5.res_max);
        out.push(&p("null_r5_res_mean"), zr5.res_mean);
        out.push(&p("null_r5_res_top20_mean"), zr5.res_top20);
    }
    out.push(&p("null_s_r1_bestcomp_gain"), nsbc.gain.clone());
    out.push(&p("null_s_r1_bestcomp_mutual"), nsbc.mutual.clone());
    out.push(&p("null_s_r1_bestcomp_rank"), nsbc.rank.clone());
    out.push(&p("null_s_r1_explain_gain_max"), ns1.gain_max.clone());
    out.push(&p("null_s_r1_explain_gain_mean"), ns1.gain_mean.clone());
    out.push(&p("null_s_r1_explain_z_mean"), ns1.z_mean.clone());
    if has_lnt {
        let lntv = lnt.unwrap();
        let zs5 = r5_residual(n, zs, rho, lntv, ok);
        out.push(&p("null_s_r5_res_hhi"), zs5.res_hhi);
        out.push(&p("null_s_r5_res_max"), zs5.res_max);
        out.push(&p("null_s_r5_res_mean"), zs5.res_mean);
        out.push(&p("null_s_r5_res_top20_mean"), zs5.res_top20);
    }
    eprintln!("[reduce:p10] 空模型对照 {:.1}s", t0.elapsed().as_secs_f32());

    // ---------------- 扩展块（reduce2）----------------
    crate::reduce2::expand(ms, pre, n, ok, industry, lnt, has_lnt, out);
}

pub fn cmd_reduce(args: &[String]) {
    let t_all = Instant::now();
    let pair_root = PathBuf::from(&args[0]);
    let date = args[1].clone();
    let outdir = PathBuf::from(&args[2]);
    let backup_root = PathBuf::from(
        parse_flag(args, "--backup")
            .unwrap_or("/hdd/user_home_unsafe/chenzongwei/yupei_dist_backup"),
    );
    if let Some(t) = parse_flag(args, "--threads") {
        let k: usize = t.parse().unwrap();
        let _ = rayon::ThreadPoolBuilder::new().num_threads(k).build_global();
    }
    let mats_arg = parse_flag(args, "--mat").unwrap_or("cnt_t1").to_string();
    let mats: Vec<String> = mats_arg.split(',').map(|s| s.trim().to_string()).collect();
    // E 组参数变体：sigmoid 温度与硬阈值（默认 0.3 / 2.0，与原硬编码一致）
    let sig_temp: f64 = parse_flag(args, "--sig-temp")
        .map(|s| s.parse().expect("--sig-temp 需为浮点数"))
        .unwrap_or(0.3);
    let sig_thr: f64 = parse_flag(args, "--sig-thr")
        .map(|s| s.parse().expect("--sig-thr 需为浮点数"))
        .unwrap_or(2.0);
    set_soft(SoftCfg {
        temp: sig_temp,
        thr: sig_thr,
    });

    let day = pair_root.join(&date);
    let codes = read_codes(&day.join("codes.txt")).expect("读 codes.txt 失败");
    let n = codes.len();
    eprintln!("[reduce] date = {date}, n = {n}, mats = {mats:?}, 线程 {}, sigmoid temp={sig_temp} thr={sig_thr}", rayon::current_num_threads());

    // 股票白名单与行业、成交笔数
    let basic = std::env::var("RUST_PYFUNC_BASIC_INFO_DIR")
        .unwrap_or_else(|_| "/ssd_data/data/basic_info".to_string());
    let sm = read_symbol_map(&Path::new(&basic).join("symbol_map.csv"))
        .expect("读 symbol_map.csv 失败，不允许退回全算");
    let ok: Vec<bool> = codes.iter().map(|c| sm.contains(c)).collect();
    let n_ok = ok.iter().filter(|&&v| v).count();
    eprintln!("[reduce] 白名单内股票 {n_ok}/{n}，非个股 {}", n - n_ok);

    let bday = backup_root.join(&date);
    let ind_path = {
        let p1 = day.join("industry.bin");
        let p2 = bday.join("industry.bin");
        if p1.exists() {
            Some(p1)
        } else if p2.exists() {
            Some(p2)
        } else {
            None
        }
    };
    let industry = ind_path.as_ref().and_then(|p| load_industry(p, &codes, n));
    match &industry {
        Some(v) => eprintln!(
            "[reduce] 行业来自 {}，有行业的 {} 只",
            ind_path.unwrap().display(),
            v.iter().filter(|&&x| x >= 0).count()
        ),
        None => eprintln!("[reduce] 没有 industry.bin，跨行业列全写空值"),
    }
    let lnt = load_lnt(&bday, &codes);
    match &lnt {
        Some(v) => eprintln!(
            "[reduce] 成交笔数 {}/{} 只有值",
            v.iter().filter(|x| x.is_finite()).count(),
            n
        ),
        None => eprintln!("[reduce] 备份目录缺 stats.bin 或 codes.txt，R5 相关列不输出"),
    }
    let has_lnt = lnt.as_ref().map(|v| v.iter().any(|x| x.is_finite())).unwrap_or(false);

    // 各矩阵之间完全独立：MatSet 装载与全部因子块都并行，
    // 每矩阵产一个局部 Out，最后按 --mat 顺序拼接列（确定性）。
    let multi = mats.len() > 1;
    // 文件名 tag 按目录里实际存在的文件探测，而不是按 --mat 个数推断：
    // 多矩阵目录下用单 mat 刷参数变体（如 E 组）时能找到 _cnt_t1 后缀文件；
    // 列名前缀跟随探测结果，保证与文件来源一致。
    let tags: Vec<String> = mats
        .iter()
        .map(|mat| {
            if multi || day.join(format!("rho_{mat}.bin")).exists() {
                format!("_{mat}")
            } else {
                String::new()
            }
        })
        .collect();
    // 注：MatSet 文件是 net 刚写出的页缓存热数据，后台预取反而与
    // 任务读盘竞争带宽（实测后段矩阵 50-78s 长尾），故不做预取——
    // 读盘优化靠 read_f32 单拷贝路径（省一次 62MB 分配+拷贝）。
    let outs: Vec<Out> = mats
        .par_iter()
        .zip(tags.par_iter())
        .map(|(mat, tag)| {
            let pre = if !tag.is_empty() {
                format!("{mat}_")
            } else {
                String::new()
            };
            let t0 = Instant::now();
            let ms = load_matset(&day, n, tag);
            eprintln!("[reduce:{mat}] MatSet 读入 {:.1}s", t0.elapsed().as_secs_f32());
            let mut out = Out::new();
            produce_one_mat(
                &ms,
                &pre,
                n,
                &ok,
                industry.as_deref(),
                lnt.as_deref(),
                has_lnt,
                &mut out,
            );
            out
        })
        .collect();
    let mut out = Out::new();
    for o in outs {
        out.names.extend(o.names);
        out.cols.extend(o.cols);
    }

    // ---------------- 落盘 ----------------
    let t0 = Instant::now();
    let f = out.names.len();
    let mut mat = vec![f32::NAN; n * f];
    for (j, col) in out.cols.iter().enumerate() {
        if col.len() != n {
            panic!("列 {} 长度 {}，期望 {}", out.names[j], col.len(), n);
        }
        for i in 0..n {
            // 非个股整行写空值：它们不参与配对，也不该给出因子值（列数仍然恒定）
            mat[i * f + j] = if ok[i] { col[i] as f32 } else { f32::NAN };
        }
    }
    let dir = outdir.join(&date).join("reduce_rs");
    std::fs::create_dir_all(&dir).expect("建输出目录失败");
    // 输出只保留股票白名单内的代码。非个股（基金/债券）整行都是空值，写出去只会让下游
    // 多出 280 行全空的垃圾行，cross-section-explore skill 的验收条款要求「输出代码必须
    // 全在白名单里」。代码表以外的行在 factors.bin 与 factors.json 里都不出现。
    let keep: Vec<usize> = (0..n).filter(|&i| ok[i]).collect();
    let nk = keep.len();
    // ---- 逐因子覆盖度报告：白名单内有限值占比，低于 95% 的列出来 ----
    {
        let mut cov: Vec<(f64, &str)> = Vec::with_capacity(f);
        for (j, col) in out.cols.iter().enumerate() {
            let c = keep.iter().filter(|&&i| col[i].is_finite()).count();
            cov.push((c as f64 / nk.max(1) as f64, out.names[j].as_str()));
        }
        cov.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        let n_low = cov.iter().filter(|(c, _)| *c < 0.95).count();
        eprintln!("[reduce:cov] 覆盖度 <95% 的因子 {n_low}/{f} 个，最低 15 个：");
        for (c, nm) in cov.iter().take(15) {
            eprintln!("[reduce:cov]   {:5.1}%  {}", 100.0 * c, nm);
        }
        // 覆盖度明细落盘，供验收核对
        let mut s = String::new();
        for (c, nm) in cov.iter() {
            s.push_str(&format!("{nm}\t{c:.4}\n"));
        }
        std::fs::write(outdir.join(&date).join("reduce_rs").join("coverage.txt"), s).ok();
    }
    {
        let mut s = String::new();
        for nm in &out.names {
            s.push_str(nm);
            s.push('\n');
        }
        std::fs::write(dir.join("names.txt"), s).unwrap();
    }
    {
        let mut rows = Vec::with_capacity(nk * f);
        for &i in keep.iter() {
            rows.extend_from_slice(&mat[i * f..i * f + f]);
        }
        let bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(rows.as_ptr() as *const u8, rows.len() * 4) };
        std::fs::write(dir.join("factors.bin"), bytes).unwrap();
    }
    {
        #[derive(serde::Serialize)]
        struct JsonOut {
            names: Vec<String>,
            codes: Vec<String>,
            vals: Vec<Vec<Option<f32>>>,
        }
        let vals: Vec<Vec<Option<f32>>> = keep
            .iter()
            .map(|&i| {
                (0..f)
                    .map(|j| {
                        let v = mat[i * f + j];
                        if v.is_finite() {
                            Some(v)
                        } else {
                            None
                        }
                    })
                    .collect()
            })
            .collect();
        let jo = JsonOut {
            names: out.names.clone(),
            codes: keep.iter().map(|&i| codes[i].clone()).collect(),
            vals,
        };
        let s = serde_json::to_string(&jo).expect("拼 JSON 失败");
        std::fs::write(dir.join("factors.json"), s).unwrap();
    }
    eprintln!("[reduce:p11] 写文件 {:.1}s", t0.elapsed().as_secs_f32());
    eprintln!(
        "[reduce:done] {} 只股票 × {} 个因子（另剔除 {} 只非个股），总 {:.1}s，输出 {}",
        nk,
        f,
        n - nk,
        t_all.elapsed().as_secs_f32(),
        dir.display()
    );
    println!(
        "{{\"ok\":true,\"date\":{},\"n\":{},\"n_factors\":{},\"outdir\":\"{}\"}}",
        date,
        n,
        f,
        dir.display()
    );
}
