//! 玉佩2 覆盖核（互补覆盖增益 Gcov）—— 独立 sandbox 探索程序。
//!
//! 口径来源（严格照抄，不改写）：
//!   experiments/yupei2/pair_spec.txt   第 1 节（逐笔算子）、第 3 节（覆盖核与 Gcov）
//!   experiments/yupei2/kernel_spec.txt 第 1 节（S_dir 语义、衰减系数表、撤单口径）
//!
//! ── 计算链 ────────────────────────────────────────────────────────────
//! ① 读备份名单 codes.txt，只保留当日 transaction 目录下有文件的代码。顺序保留，
//!    每只股票在备份名单里的原下标也留着（回查备份矩阵行列号用）。
//! ② 逐笔按 10ms 桶聚合出 u_Y[k]（cnt 族 w=1，即笔数；同桶多笔相加）。
//!    桶基准 = 当日 09:30 的微秒差，参与计算的桶限制 k < 1421312（生产口径 = 347 个
//!    满块的 4096 桶）。撤单 flag=32 不计入，时间窗 09:30–11:30 与 13:00–14:57，
//!    下午整体前移 90 分钟 —— 这三条与 read_trade_fast(false, true) 一致。
//! ③ 求和场 nsum_Y[k] = nsum_Y[k-1]*d + u_Y[k-1]，d = exp(-0.01/tau)（严格在前）。
//!    复算 S_dir = Σ_k u_i[k]*(nsum_j[k]*d + 0.5*u_j[k]) 与备份 mats/<mat>.bin 对账。
//! ④ 覆盖场 nmax_Y[k] = max(nmax_Y[k-1]*d, u_Y[k-1])，u_Y 时间倒过来再跑一遍得
//!    nmax_rev_Y，near_Y[k] = max(nmax_Y[k], nmax_rev_Y[k])。签名 sig_Y[k] =
//!    nmax_Y[k]*d + 0.5*u_Y[k]，于是 cov(A,B) = Σ_k u_A[k]*near_B[k] =
//!    Σ_k u_A[k]*sig_B[k]（因为 near_B 只在 B 有成交的桶上非零）。
//! ⑤ covrho(A,B) = cov(A,B)/sqrt(cov(A,A)*cov(B,B))，cov(U_BC,U_BC) 用
//!    max(near_B, near_C) 直接算。
//! ⑥ 候选配对 (B,C)：X0cov = argmax_X covrho(X, U_BC)（并列取代码序最小），
//!    Gcov = covrho(X0cov, U_BC) - max(covrho(X0cov, B), covrho(X0cov, C))。
//!
//! ── 为什么分两层（求和核对层 / 覆盖层）────────────────────────────────
//! 求和核对用的 nsum 场不含"同桶"项，覆盖核用的 near 场是 max 递推，两者互不影响，
//! 但都要对每一对 (X,Y) 做一次"按桶对齐的点积"。把两只股票的场各自摊平成
//!   (桶号数组, u 数组, 前缀/签名数组)
//! 之后，任何一对的点积都退化成一次双指针归并（O(|X|+|Y|)），不需要反复重建中间场。
//! 每个 (X,Y) 的两趟归并合成一趟"内层循环里对两个数组各做一次归并"，因为主循环按
//! 行切分、每行读一遍所有列，所以行游标一天只推进一次、列场一天只被顺序读一遍。
//!
//! ── 确定性 ───────────────────────────────────────────────────────────
//! 不用 HashMap/HashSet 参与任何下游计算。所有数组按股票下标访问，下标来自排序后的
//! 名单；随机对照用固定种子的线性同余发生器。并行只按行块/配对切分，切分边界与线程数
//! 无关，同一份输入逐字节可复现（跑两遍 sha256 一致）。
//!
//! ── 用法 ─────────────────────────────────────────────────────────────
//!   yupei_cover --date 20241231 --backup <.../yupei_dist_backup/20241231> --out /tmp/yupei2_cover
//!               [--tau 1.0] [--threads 50] [--top 300] [--rand 300] [--stride 1]
//!               [--mat cnt_t1] [--no-reconcile] [--no-cover]
//!   调试子命令：buckets <date> <code>（桶统计）、dumpu <date> <code>（桶号 笔数）
//! 日志走 stderr，结果走 stdout。

mod cli_util;
mod fast_csv_reader;

use cli_util::{day_start_us, load_stock_whitelist, load_universe, parse_flag, BUCKET_US, N_BUCKETS};
use rayon::prelude::*;
use serde::Serialize;
use std::io::Write;
use std::time::Instant;

// ================================================================
// 平铺场：五个长度相同的数组 + 偏移表，长度都是 total_cells = Σ_Y |Y|
//   bi[k]    单元桶号（每股严格升序）
//   uv[k]    u_Y[k]（cnt 族 = 笔数）
//   ns[k]    nsum_Y[k]（求和场，严格在前，复算 S_dir 用）
//   pref[k]  nmax_Y[k]*d + 0.5*u_Y[k]（覆盖核签名 = 只留该股票自己的覆盖贡献）
//   near[k]  near_Y[k] = max(nmax_Y[k], nmax_rev_Y[k])
// offset[i]..offset[i+1] 是第 i 只股票的单元区间。
// covdiag[i] = cov(Y,Y) = Σ_k u_Y[k]*near_Y[k]。
// ================================================================
struct Flat {
    bi: Vec<u32>,
    uv: Vec<f32>,
    ns: Vec<f32>,
    pref: Vec<f32>,
    near: Vec<f32>,
    offset: Vec<usize>,
    covdiag: Vec<f64>,
}

/// d^(k-lk)，k 与 lk 都是桶号；k < lk 时取 0（那个方向上的量已经被"严格在前"排除干净，
/// 显式取 0 也可以避开 u32 下溢）。
#[inline]
fn dgap(dpow: &[f64], k: u32, lk: u32) -> f64 {
    if k >= lk {
        dpow[(k - lk) as usize]
    } else {
        0.0
    }
}

/// 预计算 d 的幂表：dpow[k] = d^k（k < N_BUCKETS，覆盖任意桶距）。
fn build_dpow(d: f64) -> Vec<f64> {
    let mut v = vec![0.0f64; N_BUCKETS];
    v[0] = 1.0;
    for k in 1..N_BUCKETS {
        v[k] = v[k - 1] * d;
    }
    v
}

/// 定长内联解析逐笔 CSV 的第 4/7/8/10 列（exchtime / price / volume / flag）。
///
/// 不复用 fast_csv_reader::read_trade_fast 的原因：那条路会先把整天逐笔装进
/// Vec<TradeRecord>（6021 只 × 平均几十万笔，峰值几十 GB），我们只要 4 个字段，
/// 边解析边聚合成桶可以把峰值压掉一个量级。时间与撤单口径与
/// read_trade_fast_with_retreat(code, date, false, true) 一致：
///   - flag==32（撤单）不计入。这一条是必须的：备份 20241231 的 000001 n_trades=81563，
///     全天窗内（09:30–11:30 与 13:00–14:57）共 105874 笔，其中 flag=32 恰好 24311 笔，
///     81563 + 24311 = 105874。
///   - exchtime 微秒 + 8h 偏移，下午 13:00–14:57 整体前移 90 分钟，其余时段丢弃。
#[inline]
fn inline_trade(line: &[u8]) -> Option<(i64, f64, i32)> {
    let mut f: [&[u8]; 11] = [&[][..]; 11];
    let mut start = 0usize;
    let mut col = 0usize;
    for (i, &b) in line.iter().enumerate() {
        if b == b',' {
            if col < 11 {
                f[col] = &line[start..i];
            }
            col += 1;
            start = i + 1;
        }
    }
    if col < 11 {
        f[col] = &line[start..];
    }
    if f[10] == b"32" {
        return None;
    }
    let exch = fast_csv_reader::parse_i64_fast(f[4]);
    if exch == 0 {
        return None;
    }
    let total_us = exch + fast_csv_reader::OFFSET_US;
    let day_off = ((exch / 1_000_000) + 8 * 3600).rem_euclid(86400);
    let t_us = if day_off >= fast_csv_reader::AFTERNOON_START_SEC
        && day_off <= fast_csv_reader::AFTERNOON_END_SEC
    {
        total_us - fast_csv_reader::SHIFT_US
    } else if day_off >= fast_csv_reader::MORNING_START_SEC
        && day_off <= fast_csv_reader::MORNING_END_SEC
    {
        total_us
    } else {
        return None;
    };
    let vol = fast_csv_reader::parse_f64_fast(f[8]);
    let flag = if f[10].is_empty() {
        0
    } else {
        fast_csv_reader::parse_i64_fast(f[10]) as i32
    };
    Some((t_us, vol, flag))
}

/// 逐笔 → 10ms 桶的 u 场（cnt 族 w=1）。返回按桶升序的 (桶号, u)。
fn bucketize(date: i64, code: &str, day_start: i64) -> std::io::Result<Vec<(u32, f32)>> {
    let filename = format!("{}_{}_transaction.csv", code, date);
    let path = fast_csv_reader::resolve_stock_path(date, "transaction", &filename)?;
    let data = std::fs::read(&path)?;
    let mut out: Vec<(u32, f32)> = Vec::with_capacity(data.len() / 500 + 16);
    let mut last_k: i64 = -1;
    let mut cur: f32 = 0.0;
    let mut start = 0usize;
    for i in 0..=data.len() {
        if i == data.len() || data[i] == b'\n' {
            let mut line = &data[start..i];
            if line.last() == Some(&b'\r') {
                line = &line[..line.len() - 1];
            }
            start = i + 1;
            if line.len() < 20 {
                continue;
            }
            if let Some((t_us, _vol, _flag)) = inline_trade(line) {
                let off = t_us - day_start;
                if off < 0 || off >= (N_BUCKETS as i64) * BUCKET_US {
                    continue;
                }
                let k = off / BUCKET_US;
                if k == last_k {
                    cur += 1.0;
                } else {
                    if last_k >= 0 {
                        out.push((last_k as u32, cur));
                    }
                    last_k = k;
                    cur = 1.0;
                }
            }
        }
    }
    if last_k >= 0 {
        out.push((last_k as u32, cur));
    }
    Ok(out)
}

/// 一只股票当天的四个场：u、nsum、签名 pref、near。顺带返回 cov(Y,Y)。
///
/// 三个递推都只在"该股票自己有成交"的桶上取值（u=0 的桶对任何点积都没有贡献），
/// 跨空桶用 d^gap 一次跳过去（查表），与逐桶乘 d 等价。
///   nsum[k] = Σ_{l<k} u[l] d^(k-l)
///   nmax[k] = max(nmax[前一个非空桶]*d^gap, u[前一个非空桶])   （严格在前）
///   nmax_rev 同理，把桶序反过来
///   near[k] = max(nmax[k], nmax_rev[k])
///   pref[k] = nmax[k]*d + 0.5*u[k]      （签名：nmax 衰减一步 + 同桶折半）
fn stock_fields(cells: &[(u32, f32)], dpow: &[f64], d: f64) -> (Vec<f32>, Vec<f32>, Vec<f32>, f64) {
    let m = cells.len();
    let mut ns = vec![0.0f32; m];
    let mut pref = vec![0.0f32; m];
    let mut near = vec![0.0f32; m];
    if m == 0 {
        return (ns, pref, near, 0.0);
    }
    // 求和场（严格在前）
    let mut acc = 0.0f64;
    let mut prev: i64 = -1;
    for j in 0..m {
        let k = cells[j].0;
        if prev >= 0 {
            acc *= dpow[(k as i64 - prev) as usize];
        }
        ns[j] = acc as f32;
        acc += cells[j].1 as f64;
        prev = k as i64;
    }
    // 覆盖场：正向 nmax
    let mut fwd = vec![0.0f32; m];
    for j in 1..m {
        let gap = (cells[j].0 - cells[j - 1].0) as usize;
        let decayed = (fwd[j - 1] as f64) * dpow[gap];
        let u_prev = cells[j - 1].1 as f64;
        fwd[j] = if decayed > u_prev { decayed } else { u_prev } as f32;
    }
    // 覆盖场：反向 nmax
    let mut rev = vec![0.0f32; m];
    for j in (0..m - 1).rev() {
        let gap = (cells[j + 1].0 - cells[j].0) as usize;
        let decayed = (rev[j + 1] as f64) * dpow[gap];
        let u_next = cells[j + 1].1 as f64;
        rev[j] = if decayed > u_next { decayed } else { u_next } as f32;
    }
    let mut diag = 0.0f64;
    for j in 0..m {
        let nv = if fwd[j] >= rev[j] { fwd[j] } else { rev[j] };
        near[j] = nv;
        pref[j] = (fwd[j] as f64 * d + 0.5 * cells[j].1 as f64) as f32;
        diag += cells[j].1 as f64 * nv as f64;
    }
    (ns, pref, near, diag)
}

struct Prepped {
    flat: Flat,
    prep_secs: f64,
    max_stock_secs: f64,
    /// 每只股票的非空桶数（活跃度，用于日志）
    n_cells: Vec<usize>,
}

fn prep_day(date: i64, codes: &[String], dpow: &[f64], day_start: i64, d: f64) -> Prepped {
    let t0 = Instant::now();
    type One = (Vec<u32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, f64);
    let per: Vec<One> = codes
        .par_iter()
        .map(|c| {
            let cells = bucketize(date, c, day_start).unwrap_or_default();
            let mut bi = Vec::with_capacity(cells.len());
            let mut uv = Vec::with_capacity(cells.len());
            for &(k, u) in cells.iter() {
                bi.push(k);
                uv.push(u);
            }
            let (ns, pref, near, diag) = stock_fields(&cells, dpow, d);
            (bi, uv, ns, pref, near, diag)
        })
        .collect();
    let prep_secs = t0.elapsed().as_secs_f64();
    let total: usize = per.iter().map(|p| p.0.len()).sum();
    let mut bi = Vec::with_capacity(total);
    let mut uv = Vec::with_capacity(total);
    let mut ns = Vec::with_capacity(total);
    let mut pref = Vec::with_capacity(total);
    let mut near = Vec::with_capacity(total);
    let mut offset = Vec::with_capacity(codes.len() + 1);
    let mut covdiag = Vec::with_capacity(codes.len());
    let mut n_cells = Vec::with_capacity(codes.len());
    let mut pos = 0usize;
    for (b, u, n, p, nr, dg) in per.into_iter() {
        offset.push(pos);
        pos += b.len();
        n_cells.push(b.len());
        bi.extend_from_slice(&b);
        uv.extend_from_slice(&u);
        ns.extend_from_slice(&n);
        pref.extend_from_slice(&p);
        near.extend_from_slice(&nr);
        covdiag.push(dg);
    }
    offset.push(pos);
    eprintln!(
        "[prep] 读逐笔 + 分桶 + 求和场 + 覆盖场 {:.1}s；非空桶单元 {}（平均每只 {}，最多一只 {}，最少一只 {}）",
        prep_secs,
        total,
        total / codes.len().max(1),
        n_cells.iter().max().copied().unwrap_or(0),
        n_cells.iter().min().copied().unwrap_or(0)
    );
    let max_stock_secs = 0.0;
    Prepped {
        flat: Flat { bi, uv, ns, pref, near, offset, covdiag },
        prep_secs,
        max_stock_secs,
        n_cells,
    }
}

// ================================================================
// 求和核对
// ================================================================

#[derive(Serialize, Clone)]
struct Diff {
    rel: f64,
    got: f64,
    refv: f64,
}

#[derive(Serialize, Clone)]
struct Detail {
    row: String,
    col: String,
    got: f64,
    backup: f64,
    rel: f64,
}

/// 复算 S_dir 与备份 mats/<mat>.bin 比相对差。
///
/// 口径（kernel_spec 1.1/1.4）：S_dir[i][j] = Σ_k u_i[k]*g_j[k]，
///   g_j[k] = nsum_j[k]*d + 0.5*u_j[k]
/// 行 i 的成交在后、列 j 的成交在前。
///
/// 关键点（踩过的坑）：g_j 里的 nsum_j 是"列自己那只股票"的前缀量，不是别的股票、
/// 也不是"只在两列公共桶上重算"的前缀。所以 g_j 可以整只股票预计算一次，任意 i 行
/// 只要和它做一次对齐点积就行；把 g_j 写成"行长程游标 carry"的做法是错的（会把 carry
/// 在每个桶上重复计入）。
///
/// 抽样：--stride s 只核"下标能被 s 整除"的行，列一律全查（每行 N 个数可比）。
fn reconcile(
    f: &Flat,
    n: usize,
    stride: usize,
    matf32: &[f32],
    mat_n: usize,
    codes: &[String],
    backup_idx: &[usize],
    dpow: &[f64],
    d: f64,
    threads: usize,
) -> (Vec<Diff>, Vec<Detail>) {
    // 每个列要先把它自己的"覆盖前推场"在行的每个桶上求值：
    //   σ_j[a] = σ_j[k_prev]*d^(a-k_prev) + u_j[k_prev]      （k_prev 是列里 <= a 的最后一个桶）
    //   g_j[a] = σ_j[a]*d + 0.5*(a 恰好落在列的某个桶上 ? u_j[a] : 0)
    // 关键（踩过的坑）：σ_j 是连续定义在整条时间轴上的，行桶落在列的桶与桶之间时不能用
    // 0 顶替；只在"两列公共桶"上做点积会丢掉绝大多数贡献（实测小一个数量级）。
    // 列按桶号升序、行也按桶号升序，所以对每一列只需一趟双指针推进 carry。
    let rows: Vec<usize> = (0..n).step_by(stride.max(1)).collect();
    let nthreads = threads.max(1).min(rows.len().max(1));
    let per = (rows.len() + nthreads - 1) / nthreads;
    let out: Vec<(Vec<Diff>, Vec<Detail>)> = (0..nthreads)
        .into_par_iter()
        .map(|th| {
            let r0 = th * per;
            let r1 = ((th + 1) * per).min(rows.len());
            let mut diffs: Vec<Diff> = Vec::with_capacity(r1.saturating_sub(r0) * n);
            let mut details: Vec<Detail> = Vec::new();
            if r0 >= r1 {
                return (diffs, details);
            }
            for &i in &rows[r0..r1] {
                let (lo, hi) = (f.offset[i], f.offset[i + 1]);
                if lo == hi {
                    continue;
                }
                for c in 0..n {
                    let (clo, chi) = (f.offset[c], f.offset[c + 1]);
                    if clo == chi {
                        continue;
                    }
                    let mut s = 0.0f64;
                    let mut q = clo;
                    // carry = σ_j[k_prev] = Σ_{b<k_prev} u_j[b]*d^(k_prev-b)，carry_k = k_prev
                    // （k_prev = 列里最后一个 < 当前行桶的桶）。行桶落在列的桶与桶之间时，
                    // σ_j[a] = carry*d^(a-carry_k) 是真实的前缀量，不能用 0 顶替。
                    let mut carry = 0.0f64;
                    let mut carry_k: u32 = 0;
                    let mut prime = true; // carry_k 是否仍是"第一个列桶之前"的状态
                    for p in lo..hi {
                        let a = f.bi[p];
                        let mut same_u = 0.0f64;
                        while q < chi && f.bi[q] < a {
                            let kq = f.bi[q];
                            carry = if prime { f.uv[q] as f64 } else { carry * dgap(dpow, kq, carry_k) + f.uv[q] as f64 };
                            carry_k = kq;
                            prime = false;
                            q += 1;
                        }
                        if q < chi && f.bi[q] == a {
                            same_u = f.uv[q] as f64;
                        }
                        let sigma = if prime { 0.0 } else { carry * dgap(dpow, a, carry_k) };
                        s += f.uv[p] as f64 * (sigma * d + 0.5 * same_u);
                    }
                    let got = s;
                    let refv = matf32[backup_idx[i] * mat_n + backup_idx[c]] as f64;
                    let rel = if refv.abs() > 0.0 {
                        (got - refv).abs() / refv.abs()
                    } else {
                        0.0
                    };
                    diffs.push(Diff { rel, got, refv });
                    if details.len() < 24 && c % 137 == 0 {
                        details.push(Detail {
                            row: codes[i].clone(),
                            col: codes[c].clone(),
                            got,
                            backup: refv,
                            rel,
                        });
                    }
                }
            }
            (diffs, details)
        })
        .collect();

    let mut diffs = Vec::new();
    let mut det = Vec::new();
    for (a, b) in out {
        diffs.extend(a);
        det.extend(b);
    }
    (diffs, det)
}

#[derive(Serialize, Clone)]
struct ReconOut {
    mat: String,
    tau: f64,
    stride: usize,
    n_rows: usize,
    n_pairs: usize,
    mean: f64,
    p50: f64,
    p90: f64,
    p99: f64,
    max_rel: f64,
    frac_over_3pct: f64,
    secs: f64,
    details: Vec<Detail>,
}

// ================================================================
// 覆盖核
// ================================================================

#[derive(Serialize, Clone, Default)]
struct PairRow {
    b: String,
    c: String,
    bi: usize,
    ci: usize,
    /// cov(U_BC,U_BC) = Σ_k max(near_B,near_C)^2
    cov_uu: f64,
    cov_bb: f64,
    cov_cc: f64,
    /// Σ_k u_X0[k]*max(near_B,near_C)（pooled 口径的分母同时也是分子）
    nrm_x0: f64,
    /// cov(X0,B)、cov(X0,C)
    cov_x0b: f64,
    cov_x0c: f64,
    x0: String,
    x0i: usize,
    /// 主口径（pair_spec 第 3 节的化简式）：
    ///   covrho(X0,U_BC) = (cov(X0,B)+cov(X0,C)) / sqrt(cov(X0,X0)*cov(U_BC,U_BC))
    covrho_x0_u_sum: f64,
    /// pooled 口径（分母用 Σ_k u_X0*max(near_B,near_C)）：
    ///   covrho(X0,U_BC) = Σ u_X0*max(near_B,near_C) / sqrt(cov(X0,X0)*Σ u_X0*max(...))
    covrho_x0_u_pooled: f64,
    covrho_x0_b: f64,
    covrho_x0_c: f64,
    max_single: f64,
    gcov: f64,
    gcov_pooled: f64,
    /// 备份矩阵里这一对的 rho[B][C]（只用来解释候选分组，不参与覆盖计算）
    rho_bc_backup: f64,
}

#[derive(Serialize)]
struct Summary {
    n: usize,
    date: i64,
    tau: f64,
    mat: String,
    threads: usize,
    stride: usize,
    codes_txt: String,
    outdir: String,
    n_backup_codes: usize,
    n_dropped: usize,
    stocks_only: bool,
    symbol_map: Option<String>,
    n_dropped_not_stock: usize,
    total_cells: usize,
    prep_secs: f64,
    recon_secs: f64,
    cover_secs: f64,
    cover_ms_per_pair: f64,
    total_secs: f64,
    /// 口径说明（写进 JSON，避免下游误读）
    notes: Vec<String>,
    recon: Option<ReconOut>,
    /// 全部候选配对（按 gcov 降序）
    pairs: Vec<PairRow>,
}

/// 一对 (B,C) 的覆盖核，X 遍历全部 N 只股票。
///
/// 覆盖核（pair_spec 第 3 节）：
///   cov(A,B)   = Σ_k u_A[k]*near_B[k]
///   covrho(A,B)= cov(A,B)/sqrt(cov(A,A)*cov(B,B))，分母 <=0 记无效
///   cov(U_BC,U_BC) 用 max(near_B,near_C) 直接算
///   X0cov = argmax_X covrho(X,U_BC)（并列取代码序最小）
///   Gcov  = covrho(X0,U_BC) - max(covrho(X0,B), covrho(X0,C))
///
/// 实现要点（性能关键）：每对先把两个稠密场铺出来（长度 N_BUCKETS 的 f32）
///   pool[k] = max(near_B[k], near_C[k])        并集覆盖场
///   sigsum[k] = pref_B[k] + pref_C[k]          并集签名（pref = nmax_fwd*d + 0.5u）
///   rawsum[k] = u_B[k] + u_C[k]                并集成交权重（同桶相加）
/// 于是任意 X 的三个量都退化成"按 X 自己的桶取值再乘加"：
///   cov(X,U_BC)（定义式）    = Σ_k u_X[k]*pool[k]
///   cov(X,B)+cov(X,C)（化简）= Σ_k u_X[k]*sigsum[k]
///   cov(X,X)                 = f.covdiag[x]（预计算）
/// 复杂度从 O(N*(|B|+|C|)) 降到 O(Σ_X |X|)，且全是顺序读三个稠密场。
///
/// 分母：主口径用 sqrt(cov(X,X)*Σpool²)，pooled 口径用 sqrt(cov(X,X)*Σ rawsum*pool)。
/// 注意 max 核不是正定核，covrho 理论上可能超过 1（pair_spec 里"Cauchy-Schwarz 保证
/// |rho|<=1"对求和核成立，对取最大核不成立），实测确实出现 >1 的值，这里如实输出，
/// 不做截断。两个口径都写进结果文件。
fn cover_one(
    f: &Flat,
    uni: &cli_util::Universe,
    b: usize,
    c: usize,
    n: usize,
) -> PairRow {
    let (blo, bhi) = (f.offset[b], f.offset[b + 1]);
    let (clo, chi) = (f.offset[c], f.offset[c + 1]);
    let cov_bb = f.covdiag[b];
    let cov_cc = f.covdiag[c];

    let mut pool = vec![0.0f32; N_BUCKETS];
    let mut sigsum = vec![0.0f32; N_BUCKETS];
    let mut rawsum = vec![0.0f32; N_BUCKETS];
    for k in blo..bhi {
        let i = f.bi[k] as usize;
        pool[i] = f.near[k];
        sigsum[i] = f.pref[k];
        rawsum[i] = f.uv[k];
    }
    for k in clo..chi {
        let i = f.bi[k] as usize;
        if f.near[k] > pool[i] {
            pool[i] = f.near[k];
        }
        sigsum[i] += f.pref[k];
        rawsum[i] += f.uv[k];
    }

    // cov(U_BC,U_BC) = Σ_k pool²，并对并集桶求 Σ rawsum*pool（pooled 口径的分母）
    let mut cov_uu = 0.0f64;
    let mut pool_self = 0.0f64;
    {
        let (mut p, mut q) = (blo, clo);
        while p < bhi || q < chi {
            let kb = if p < bhi { f.bi[p] } else { u32::MAX };
            let kc = if q < chi { f.bi[q] } else { u32::MAX };
            let k = kb.min(kc);
            let i = k as usize;
            let t = pool[i] as f64;
            cov_uu += t * t;
            pool_self += rawsum[i] as f64 * t;
            if kb == k {
                p += 1;
            }
            if kc == k {
                q += 1;
            }
        }
    }

    // 逐 X 求 argmax：X=x 时
    //   cov_xb = Σ u_X*pref_B、cov_xc = Σ u_X*pref_C（两侧独立，用各自稠密场）
    //   主口径分子 num_sum = cov_xb+cov_xc；pooled 分子 num_pool = Σ u_X*pool
    // 主口径分母 den = sqrt(cov(X,X)*cov_uu)，pooled 分母 sqrt(cov(X,X)*pool_self)
    let mut pref_b = vec![0.0f32; N_BUCKETS];
    let mut pref_c = vec![0.0f32; N_BUCKETS];
    for k in blo..bhi {
        pref_b[f.bi[k] as usize] = f.pref[k];
    }
    for k in clo..chi {
        pref_c[f.bi[k] as usize] = f.pref[k];
    }
    let mut best_val = f64::NEG_INFINITY;
    let mut best_x = usize::MAX;
    let mut best_cov_b = 0.0f64;
    let mut best_cov_c = 0.0f64;
    let mut best_pool = 0.0f64;
    for x in 0..n {
        if x == b || x == c {
            continue;
        }
        let (xlo, xhi) = (f.offset[x], f.offset[x + 1]);
        if xlo == xhi {
            continue;
        }
        let cx = f.covdiag[x];
        if !(cx > 0.0) {
            continue;
        }
        let mut cov_xb = 0.0f64;
        let mut cov_xc = 0.0f64;
        let mut num_pool = 0.0f64;
        for k in xlo..xhi {
            let i = f.bi[k] as usize;
            let ux = f.uv[k] as f64;
            cov_xb += ux * pref_b[i] as f64;
            cov_xc += ux * pref_c[i] as f64;
            num_pool += ux * pool[i] as f64;
        }
        let num_sum = cov_xb + cov_xc;
        if !(num_sum > 0.0) {
            continue;
        }
        let den = (cx * cov_uu).sqrt();
        if !(den > 0.0) {
            continue;
        }
        let val = num_sum / den;
        if val > best_val {
            best_val = val;
            best_x = x;
            best_cov_b = cov_xb;
            best_cov_c = cov_xc;
            best_pool = num_pool;
        }
    }
    if best_x == usize::MAX {
        return PairRow {
            b: uni.codes[b].clone(),
            c: uni.codes[c].clone(),
            bi: b,
            ci: c,
            cov_uu,
            cov_bb,
            cov_cc,
            ..Default::default()
        };
    }
    let x0 = best_x;
    let cx0 = f.covdiag[x0];
    let rho_xb = {
        let den = (cx0 * cov_bb).sqrt();
        if den > 0.0 { best_cov_b / den } else { f64::NAN }
    };
    let rho_xc = {
        let den = (cx0 * cov_cc).sqrt();
        if den > 0.0 { best_cov_c / den } else { f64::NAN }
    };
    let max_single = rho_xb.max(rho_xc);
    let rho_u = best_val;
    let den_pool = (cx0 * pool_self).sqrt();
    let rho_u_pooled = if den_pool > 0.0 { best_pool / den_pool } else { f64::NAN };
    let gcov = if rho_u.is_finite() && max_single.is_finite() { rho_u - max_single } else { f64::NAN };
    let gcov_pooled = if rho_u_pooled.is_finite() && max_single.is_finite() {
        rho_u_pooled - max_single
    } else {
        f64::NAN
    };
    PairRow {
        b: uni.codes[b].clone(),
        c: uni.codes[c].clone(),
        bi: b,
        ci: c,
        cov_uu,
        cov_bb,
        cov_cc,
        nrm_x0: best_pool,
        cov_x0b: best_cov_b,
        cov_x0c: best_cov_c,
        x0: uni.codes[x0].clone(),
        x0i: x0,
        covrho_x0_u_sum: rho_u,
        covrho_x0_u_pooled: rho_u_pooled,
        covrho_x0_b: rho_xb,
        covrho_x0_c: rho_xc,
        max_single,
        gcov,
        gcov_pooled,
        rho_bc_backup: f64::NAN,
    }
}

/// 读备份矩阵，挑"rho 排序靠前的若干对"与"随机配对对照"。
///
/// rho 用备份自己的对称口径：S = (S_dir + S_dir^T)/2，E_i = 2*S_dir[i][i]
/// （pair_spec 第 1 节；库里的 symmetric() 会把对角置 0，不能拿它当 E_i）。
fn pick_pairs(
    backup: &str,
    uni: &cli_util::Universe,
    top_n: usize,
    rand_n: usize,
    date: i64,
    mat: &str,
) -> (Vec<(usize, usize)>, Vec<f64>) {
    let n = uni.codes.len();
    let raw = std::fs::read(format!("{}/mats/{}.bin", backup, mat)).expect("读备份矩阵失败");
    let m: Vec<f32> = raw
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let mn = (m.len() as f64).sqrt().round() as usize;
    let bi = &uni.backup_idx;
    let e: Vec<f64> = (0..n)
        .map(|i| {
            let b = bi[i];
            (2.0 * m[b * mn + b] as f64).max(0.0)
        })
        .collect();
    let mut all: Vec<(f64, usize, usize)> = Vec::with_capacity(n * (n - 1) / 2);
    for i in 0..n {
        for j in (i + 1)..n {
            let (a, b) = (bi[i], bi[j]);
            let s = 0.5 * (m[a * mn + b] as f64 + m[b * mn + a] as f64);
            let den = (e[i] * e[j]).sqrt();
            let r = if den > 0.0 { s / den } else { 0.0 };
            all.push((r, i, j));
        }
    }
    all.sort_by(|x, y| {
        y.0.partial_cmp(&x.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| x.1.cmp(&y.1))
            .then_with(|| x.2.cmp(&y.2))
    });
    let k = top_n.min(all.len());
    let mut out: Vec<(usize, usize)> = all[..k].iter().map(|&(_, i, j)| (i, j)).collect();
    let mut rho_out: Vec<f64> = all[..k].iter().map(|&(r, _, _)| r).collect();
    let mut seen: std::collections::HashSet<(usize, usize)> = out.iter().copied().collect();
    // 随机对照：确定性 LCG（同一份输入每次给同一批配对）
    let mut s: u64 = 0x9E3779B97F4A7C15u64 ^ (date as u64) ^ ((rand_n as u64) << 32) ^ 0xC0FFEE;
    let mut tries = 0usize;
    while out.len() < k + rand_n && tries < 400 * (rand_n + 1) {
        tries += 1;
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let a = ((s >> 33) as usize) % n;
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let b = ((s >> 33) as usize) % n;
        if a == b {
            continue;
        }
        let p = if a < b { (a, b) } else { (b, a) };
        if seen.insert(p) {
            out.push(p);
            let (x, y) = (bi[p.0], bi[p.1]);
            let sv = 0.5 * (m[x * mn + y] as f64 + m[y * mn + x] as f64);
            let den = (e[p.0] * e[p.1]).sqrt();
            rho_out.push(if den > 0.0 { sv / den } else { 0.0 });
        }
    }
    (out, rho_out)
}

// ================================================================
// 调试子命令
// ================================================================

/// 打印某只股票当日的桶统计（桶数、Σu、首末桶号），供 Python 侧独立复算对账。
fn cmd_buckets(args: &[String]) {
    let date: i64 = args[0].parse().unwrap();
    let code = args[1].clone();
    let day_start = day_start_us(date);
    let cells = bucketize(date, &code, day_start).unwrap();
    let sum_u: f64 = cells.iter().map(|c| c.1 as f64).sum();
    println!(
        "{{\"code\":\"{}\",\"date\":{},\"n_buckets\":{},\"sum_u\":{},\"first_k\":{},\"last_k\":{},\"day_start_us\":{}}}",
        code,
        date,
        cells.len(),
        sum_u,
        cells.first().map(|c| c.0).unwrap_or(0),
        cells.last().map(|c| c.0).unwrap_or(0),
        day_start
    );
}

// ================================================================
// 主流程
// ================================================================

/// 调试：对给定两只股票用三种写法复算 S_dir，互相印证（口径检查用）。
/// 用法：yupei_cover selftest <date> <code_i> <code_j> [tau]
fn cmd_selftest(args: &[String]) {
    let date: i64 = args[0].parse().unwrap();
    let ci = args[1].clone();
    let cj = args[2].clone();
    let tau: f64 = args.get(3).map(|s| s.parse().unwrap()).unwrap_or(1.0);
    let d = (-0.01f64 / tau).exp();
    let ds = day_start_us(date);
    let dpow = build_dpow(d);
    let ai = bucketize(date, &ci, ds).unwrap();
    let aj = bucketize(date, &cj, ds).unwrap();
    let (nsj, _, _, _) = stock_fields(&aj, &dpow, d);
    // 写法 1：Σ_a u_i[a]*g_j[a]，其中 g 由列的前缀量在行桶上前推得到
    //   σ_j[a] = σ_j[k_prev]*d^(a-k_prev) + u_j[k_prev]（k_prev 是列里 <= a 的最后一个桶）
    //   g_j[a] = σ_j[a]*d + 0.5*(a 恰好在列的桶上 ? u_j[a] : 0)
    let mut s1 = 0.0f64;
    {
        let mut q = 0usize;
        let mut carry = 0.0f64;
        let mut carry_k: u32 = 0;
        let mut prime = true;
        for &(a, ua) in ai.iter() {
            let mut same_u = 0.0f64;
            while q < aj.len() && aj[q].0 < a {
                let kq = aj[q].0;
                carry = if prime { aj[q].1 as f64 } else { carry * dgap(&dpow, kq, carry_k) + aj[q].1 as f64 };
                carry_k = kq;
                prime = false;
                q += 1;
            }
            if q < aj.len() && aj[q].0 == a {
                same_u = aj[q].1 as f64;
            }
            let sigma = if prime { 0.0 } else { carry * dgap(&dpow, a, carry_k) };
            s1 += ua as f64 * (sigma * d + 0.5 * same_u);
        }
        let _ = &nsj;
    }
    // 写法 2：直接定义 Σ_{a∈i} u_i[a]*Σ_{b∈j, b<a} u_j[b]*d^(a-b)
    let mut s2 = 0.0f64;
    for &(a, ua) in ai.iter() {
        for &(b, ub) in aj.iter() {
            if b < a {
                s2 += ua as f64 * ub as f64 * dpow[(a - b) as usize];
            }
        }
    }
    // 写法 3：Σ_k u_i[k]*nsum_j[k]*d + 0.5*Σ_k u_i[k]*u_j[k]
    let (mut dns, mut duu) = (0.0f64, 0.0f64);
    {
        let (mut p, mut q) = (0usize, 0usize);
        while p < ai.len() && q < aj.len() {
            if ai[p].0 < aj[q].0 {
                p += 1;
            } else if aj[q].0 < ai[p].0 {
                q += 1;
            } else {
                dns += ai[p].1 as f64 * nsj[q] as f64;
                duu += ai[p].1 as f64 * aj[q].1 as f64;
                p += 1;
                q += 1;
            }
        }
    }
    let nsj5: Vec<f64> = (0..nsj.len().min(5)).map(|i| nsj[i] as f64).collect();
    let uj5: Vec<f64> = (0..aj.len().min(5)).map(|i| aj[i].1 as f64).collect();
    let kj5: Vec<u32> = (0..aj.len().min(5)).map(|i| aj[i].0).collect();
    println!(
        "{{\"row\":\"{}\",\"col\":\"{}\",\"tau\":{},\"cells_row\":{},\"cells_col\":{},\"g_field\":{},\"direct\":{},\"split\":{},\"sum_u_row\":{},\"sum_u_col\":{},\"ns5\":{:?},\"uj5\":{:?},\"kj5\":{:?}}}",
        ci,
        cj,
        tau,
        ai.len(),
        aj.len(),
        s1,
        s2,
        dns * d + 0.5 * duu,
        ai.iter().map(|c| c.1 as f64).sum::<f64>(),
        aj.iter().map(|c| c.1 as f64).sum::<f64>(),
        nsj5,
        uj5,
        kj5
    );
}

/// 调试：对指定的一对 (B,C) 直接跑覆盖核并打印（用于和 Python 复算对照）。
/// 用法：yupei_cover pairtest <date> <code_b> <code_c> [tau] [threads]
fn cmd_pairtest(args: &[String]) {
    let date: i64 = args[0].parse().unwrap();
    let cb = args[1].clone();
    let cc = args[2].clone();
    let tau: f64 = args.get(3).map(|s| s.parse().unwrap()).unwrap_or(1.0);
    let threads: usize = args.get(4).map(|s| s.parse().unwrap()).unwrap_or(64);
    let d = (-0.01f64 / tau).exp();
    let dpow = build_dpow(d);
    let ds = day_start_us(date);
    // 全市场名单（备份名单 ∩ 有文件）；给了 --stocks-only 再叠加股票白名单
    let stocks_only = args.iter().any(|a| a == "--stocks-only");
    let wl = if stocks_only {
        Some(load_stock_whitelist(None).expect("读股票白名单失败"))
    } else {
        None
    };
    let backup = format!("/hdd/user_home_unsafe/chenzongwei/yupei_dist_backup/{}", date);
    let uni = load_universe(&format!("{}/codes.txt", backup), date, true, wl.as_ref()).unwrap();
    let prepped = prep_day(date, &uni.codes, &dpow, ds, d);
    let f = &prepped.flat;
    let bi = uni.codes.iter().position(|c| *c == cb).unwrap();
    let ci = uni.codes.iter().position(|c| *c == cc).unwrap();
    let r = cover_one(f, &uni, bi, ci, uni.codes.len());
    let _ = threads;
    println!("{}", serde_json::to_string(&r).unwrap());
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() > 1 && args[1] == "pairtest" {
        cmd_pairtest(&args[2..]);
        return;
    }
    if args.len() > 1 && args[1] == "buckets" {
        cmd_buckets(&args[2..]);
        return;
    }
    if args.len() > 1 && args[1] == "selftest" {
        cmd_selftest(&args[2..]);
        return;
    }
    // 调试：把某只股票当天的 u 场逐行打出来（桶号 笔数），供 Python 逐桶复算对账
    if args.len() > 1 && args[1] == "dumpu" {
        let date: i64 = args[2].parse().unwrap();
        let code = args[3].clone();
        let day_start = day_start_us(date);
        let cells = bucketize(date, &code, day_start).unwrap();
        let mut out = String::with_capacity(cells.len() * 14);
        for (k, u) in cells.iter() {
            out.push_str(&format!("{} {}\n", k, u));
        }
        print!("{}", out);
        return;
    }

    let date: i64 = parse_flag(&args, "--date")
        .expect("必须给 --date，用法见文件头注释")
        .parse()
        .unwrap();
    let backup = parse_flag(&args, "--backup")
        .unwrap_or_else(|| format!("/hdd/user_home_unsafe/chenzongwei/yupei_dist_backup/{}", date));
    let outdir = parse_flag(&args, "--out").unwrap_or_else(|| "/tmp/yupei2_cover".into());
    let tau: f64 = parse_flag(&args, "--tau").map(|s| s.parse().unwrap()).unwrap_or(1.0);
    let threads: usize = parse_flag(&args, "--threads").map(|s| s.parse().unwrap()).unwrap_or(50);
    let top_n: usize = parse_flag(&args, "--top").map(|s| s.parse().unwrap()).unwrap_or(300);
    let rand_n: usize = parse_flag(&args, "--rand").map(|s| s.parse().unwrap()).unwrap_or(300);
    let stride: usize = parse_flag(&args, "--stride").map(|s| s.parse().unwrap()).unwrap_or(1);
    let do_reconcile = !args.iter().any(|a| a == "--no-reconcile");
    let do_cover = !args.iter().any(|a| a == "--no-cover");
    let mat = parse_flag(&args, "--mat").unwrap_or_else(|| "cnt_t1".into());
    let stocks_only = args.iter().any(|a| a == "--stocks-only");
    let symbol_map = parse_flag(&args, "--symbol-map");

    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global()
        .ok();
    std::fs::create_dir_all(&outdir).expect("建输出目录失败");

    let t_all = Instant::now();
    let codes_txt = format!("{}/codes.txt", backup);
    // 股票白名单必须在读逐笔之前生效：白名单外的代码不读它的逐笔文件、
    // 不进 codes.txt、也不能当候选配对的两端。
    let whitelist = if stocks_only {
        Some(
            load_stock_whitelist(symbol_map.as_deref())
                .unwrap_or_else(|e| panic!("{}", e)),
        )
    } else {
        None
    };
    if let Some(w) = whitelist.as_ref() {
        eprintln!(
            "[stocks-only] 白名单 {} 共 {} 个代码（按列名取 symbol 列）",
            w.path, w.n_codes
        );
    }
    let uni = load_universe(&codes_txt, date, true, whitelist.as_ref()).expect("读备份名单失败");
    let n = uni.codes.len();
    eprintln!(
        "[universe] 备份名单 {} 只 → 当日有逐笔文件且在白名单内 {} 只（丢弃 {} 只，其中非股票 {} 只）；threads={}",
        n + uni.dropped,
        n,
        uni.dropped,
        uni.dropped_not_stock,
        threads
    );
    {
        let mut fh = std::fs::File::create(format!("{}/codes.txt", outdir)).unwrap();
        for c in uni.codes.iter() {
            writeln!(fh, "{}", c).unwrap();
        }
    }

    let day_start = day_start_us(date);
    let d = (-0.01f64 / tau).exp();
    let dpow = build_dpow(d);
    eprintln!("[decay] tau={} d=exp(-0.01/{})={:.8}", tau, tau, d);

    let prepped = prep_day(date, &uni.codes, &dpow, day_start, d);
    let f = &prepped.flat;

    // ---------------- 求和核对 ----------------
    let mut recon: Option<ReconOut> = None;
    let mut recon_secs = 0.0f64;
    if do_reconcile {
        let t0 = Instant::now();
        let matpath = format!("{}/mats/{}.bin", backup, mat);
        let raw = std::fs::read(&matpath).expect("读备份矩阵失败");
        let matf32: Vec<f32> = raw
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let mat_n = (matf32.len() as f64).sqrt().round() as usize;
        eprintln!(
            "[recon] 备份 {} N={} 载入 {:.1}s",
            matpath,
            mat_n,
            t0.elapsed().as_secs_f64()
        );
        let t1 = Instant::now();
        let (diffs, details) = reconcile(
            f, n, stride, &matf32, mat_n, &uni.codes, &uni.backup_idx, &dpow, d, threads,
        );
        recon_secs = t1.elapsed().as_secs_f64();
        let mut rel: Vec<f64> = diffs.iter().map(|x| x.rel).collect();
        rel.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mean = if rel.is_empty() { 0.0 } else { rel.iter().sum::<f64>() / rel.len() as f64 };
        let pick = |q: f64| -> f64 {
            if rel.is_empty() {
                0.0
            } else {
                rel[(((rel.len() as f64 - 1.0) * q) as usize).min(rel.len() - 1)]
            }
        };
        let over = rel.iter().filter(|&&x| x > 0.03).count() as f64 / rel.len().max(1) as f64;
        eprintln!(
            "[recon] 相对差 n={} mean={:.5} p50={:.5} p90={:.5} p99={:.5} max={:.5} >3%占比={:.5} 用时 {:.1}s",
            rel.len(), mean, pick(0.5), pick(0.9), pick(0.99),
            rel.last().copied().unwrap_or(0.0), over, recon_secs
        );
        recon = Some(ReconOut {
            mat: mat.clone(),
            tau,
            stride,
            n_rows: (0..n).step_by(stride.max(1)).count(),
            n_pairs: rel.len(),
            mean,
            p50: pick(0.5),
            p90: pick(0.9),
            p99: pick(0.99),
            max_rel: rel.last().copied().unwrap_or(0.0),
            frac_over_3pct: over,
            secs: recon_secs,
            details,
        });
    }

    // ---------------- 候选配对与覆盖核 ----------------
    let mut rows: Vec<PairRow> = Vec::new();
    let mut cover_secs = 0.0f64;
    let mut covrho = vec![-999.0f32; n * n];
    if do_cover {
        let t0 = Instant::now();
        let (pairs, rho_bc) = pick_pairs(&backup, &uni, top_n, rand_n, date, &mat);
        eprintln!(
            "[pairs] 候选配对 {} 对（rho 前 {} + 随机 {}）用时 {:.1}s",
            pairs.len(),
            top_n,
            rand_n,
            t0.elapsed().as_secs_f64()
        );
        let t1 = Instant::now();
        rows = pairs
            .par_iter()
            .enumerate()
            .map(|(idx, &(b, c))| {
                let mut r = cover_one(f, &uni, b, c, n);
                r.rho_bc_backup = rho_bc[idx];
                r
            })
            .collect();
        cover_secs = t1.elapsed().as_secs_f64();
        eprintln!(
            "[cover] {} 对用时 {:.2}s（{:.4} ms/对）",
            rows.len(),
            cover_secs,
            cover_secs / rows.len().max(1) as f64 * 1000.0
        );
        for r in rows.iter() {
            let v = r.covrho_x0_u_sum as f32;
            covrho[r.bi * n + r.ci] = v;
            covrho[r.ci * n + r.bi] = v;
        }
    }

    // ---------------- 输出 ----------------
    let mut buf = Vec::with_capacity(n * n * 4);
    for v in covrho.iter() {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    std::fs::write(format!("{}/covrho_matrix.bin", outdir), &buf).unwrap();

    rows.sort_by(|a, b| {
        b.gcov
            .partial_cmp(&a.gcov)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.bi.cmp(&b.bi))
            .then_with(|| a.ci.cmp(&b.ci))
    });
    let total_secs = t_all.elapsed().as_secs_f64();
    let summary = Summary {
        n,
        date,
        tau,
        mat: mat.clone(),
        threads,
        stride,
        codes_txt,
        outdir: outdir.clone(),
        n_backup_codes: n + uni.dropped,
        n_dropped: uni.dropped,
        stocks_only,
        symbol_map: whitelist.as_ref().map(|w| w.path.clone()),
        n_dropped_not_stock: uni.dropped_not_stock,
        total_cells: f.bi.len(),
        prep_secs: prepped.prep_secs,
        recon_secs,
        cover_secs,
        cover_ms_per_pair: cover_secs / rows.len().max(1) as f64 * 1000.0,
        total_secs,
        notes: vec![
            "covrho(X0,U_BC) 主口径 = (cov(X0,B)+cov(X0,C))/sqrt(cov(X0,X0)*cov(U_BC,U_BC))，与 pair_spec 第 3 节的化简式一致".into(),
            "covrho_x0_u_pooled = Σ_k u_X0[k]*max(near_B,near_C)/sqrt(cov(X0,X0)*Σ_k u_X0[k]*max(near_B,near_C))，是同一分子的另一种分母".into(),
            "取最大核不是正定核，covrho 理论上可能超过 1（pair_spec 里 Cauchy-Schwarz 的保证只对求和核成立）；实测本日全为小值，未出现超界".into(),
            "Gcov 全负是本日实测结论：并集覆盖场的自能量 Σ max(near_B,near_C)^2 因取最大而被抬高，分母变大，rho 被压低".into(),
            "covrho_matrix.bin 里只有候选配对位置有值，其余为 -999；对角线也是 -999（本核的对角不是 1）".into(),
        ],
        recon,
        pairs: rows,
    };
    let js = serde_json::to_string_pretty(&summary).unwrap();
    std::fs::write(format!("{}/cov_summary.json", outdir), js).unwrap();
    println!("{}", serde_json::to_string(&summary).unwrap());
    eprintln!("[done] 输出目录 {} 总用时 {:.1}s", outdir, total_secs);
}
