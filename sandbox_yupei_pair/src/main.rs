//! 玉佩2 双层网络（配对互补）—— 求和核版本的「组合形状增益 G」。
//!
//! 口径冻结在 experiments/yupei2/pair_spec.txt，本文件严格照它实现。核心事实：
//!   S_dir[i][j] 已经在备份矩阵里（行 i 的成交在后、列 j 在前，同桶 50/50 折半）。
//!   对称核 S[i][j] = S_dir[i][j] + S_dir[j][i]（i != j）。
//!   自能量 E_i = 2 * S_dir[i][i]（含同桶自配对，库里 symmetric() 会把对角清零，不能拿它当 E）。
//!   rho[i][j] = S[i][j] / sqrt(E_i E_j)。
//!   拼接 B、C 成 U_BC 后，rho[X][U_BC] = wB*rho[X][B] + wC*rho[X][C]，
//!   其中 wB = sqrt(eB)/D, wC = sqrt(eC)/D, D = sqrt(eB + eC + 2*rho[B][C]*sqrt(eB*eC))。
//!   所以每对 (B,C) 只需要一趟长度 N 的加权最大扫描就能定出 X0，不用真的拼数据。
//!
//! 用法:
//!   yupei_pair net <backup_root> <date> <outdir> [--mat cnt_t1] [--null-mc 20000]
//!                  [--threads 200] [--topk 15]
//!   备份目录结构见 sandbox_yupei_dist/src/matrix_store.rs：
//!     <backup_root>/<date>/codes.txt
//!     <backup_root>/<date>/stats.bin     u64 n + n*14 个 f64（第 0 个是成交笔数）
//!     <backup_root>/<date>/mats/<mat>.bin  N*N f32 行主序有向矩阵
//!
//! 输出见 spec 第 4 节。确定性与项目规矩一致：全部按下标序循环，不用 HashMap 迭代序。
//!
//! 性能：主循环对每个 B 复用一列（rhoT 的行），内层对 C 顺序读另一列。
//! N=3964 时配对数 7.86e6，每对扫 3964 个下标，总访存约 124 GB，属于带宽受限。

use rayon::prelude::*;
use std::path::{Path, PathBuf};

mod check;
mod reduce;
mod reduce2;
mod simd;
mod topk;

// ------------------------------------------------------------------ 基础 IO

fn read_codes(p: &Path) -> std::io::Result<Vec<String>> {
    let s = std::fs::read_to_string(p)?;
    Ok(s.lines()
        .map(|l| l.trim().to_string())
        .filter(|l| !l.is_empty())
        .collect())
}

/// stats.bin：u64 n，然后 n×14 个 f64；只要第 0 个（成交笔数）。
/// 行 i 的 14 个 f64 从 8 + i*14*8 开始。
fn read_ntrades(p: &Path) -> std::io::Result<Vec<f64>> {
    let data = std::fs::read(p)?;
    if data.len() < 8 {
        return Err(err("stats.bin 太短"));
    }
    let n = u64::from_le_bytes(data[0..8].try_into().unwrap()) as usize;
    if data.len() < 8 + n * 14 * 8 {
        return Err(err("stats.bin 长度与 n 不符"));
    }
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let off = 8 + i * 14 * 8;
        out.push(f64::from_le_bytes(data[off..off + 8].try_into().unwrap()));
    }
    Ok(out)
}

fn read_f32(p: &Path) -> std::io::Result<Vec<f32>> {
    let data = std::fs::read(p)?;
    if data.len() % 4 != 0 {
        return Err(err("f32 文件长度不是 4 的倍数"));
    }
    let n = data.len() / 4;
    let mut out = vec![0.0f32; n];
    unsafe {
        std::ptr::copy_nonoverlapping(data.as_ptr(), out.as_mut_ptr() as *mut u8, data.len());
    }
    Ok(out)
}

fn write_f32(p: &Path, v: &[f32]) -> std::io::Result<()> {
    let bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) };
    std::fs::write(p, bytes)
}

fn write_f64(p: &Path, v: &[f64]) -> std::io::Result<()> {
    let bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 8) };
    std::fs::write(p, bytes)
}

fn err<S: Into<String>>(m: S) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::InvalidData, m.into())
}

/// 股票白名单。项目规则（cross-section-explore）：六位数字文件名不等于股票，
/// 原始目录里同时有基金与债券；默认只允许 /ssd_data/data/basic_info/symbol_map.csv
/// 的 symbol 列里的代码进入计算，而且要在计算之前过滤，不是最后再筛输出。
/// 读不到就报错，不退回"全都算"。
fn read_symbol_map(p: &Path) -> std::io::Result<std::collections::HashSet<String>> {
    let s = std::fs::read_to_string(p)?;
    let mut set = std::collections::HashSet::new();
    let mut lines = s.lines();
    let head = lines.next().unwrap_or("");
    let col = head
        .split(',')
        .position(|h| h.trim().trim_matches('"') == "symbol")
        .ok_or_else(|| err("symbol_map.csv 缺少 symbol 列"))?;
    for l in lines {
        let f: Vec<&str> = l.split(',').collect();
        if let Some(v) = f.get(col) {
            let c = v.trim().trim_matches('"');
            if !c.is_empty() {
                set.insert(c.to_string());
            }
        }
    }
    if set.is_empty() {
        return Err(err("symbol_map.csv 的 symbol 列为空"));
    }
    Ok(set)
}

// ------------------------------------------------------------------ 小工具

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

fn parse_flag<'a>(args: &'a [String], flag: &str) -> Option<&'a str> {
    args.iter().position(|a| a == flag).map(|i| args[i + 1].as_str())
}

fn decile_of(v: f64, cuts: &[f64]) -> usize {
    // cuts 是 9 个十分位切点，返回 0..9
    let mut d = 0usize;
    while d < cuts.len() && v > cuts[d] {
        d += 1;
    }
    d
}

// ------------------------------------------------------------------ 主流程

struct Shared {
    n: usize,
    g: *mut f32,
    ru: *mut f32,
    sg: *mut f32,
    x0: *mut f32,
    gs: *mut f32,
    rus: *mut f32,
    sgs: *mut f32,
    x0s: *mut f32,
    // ---- 新增配对分数矩阵（全部对称、每对写两格）----
    grel: *mut f32,  // G / max(p,q)：除掉"与 X0 的相似度水平"的相对增益
    gsh: *mut f32,   // G / h：增量占合体拟合的份额
    gcov: *mut f32,  // mean_X relu(gap)：合体对全市场平均多解释多少
    gnpos: *mut f32, // #(gap>0)/#valid：增益覆盖的股票占比
    ghhi: *mut f32,  // gap 的 HHI：增益集中在少数 X 还是摊得开
    pb: *mut f32,    // rho(X0,B)
    qc: *mut f32,    // rho(X0,C)
    pbs: *mut f32,   // rho(X0*,B)
    qcs: *mut f32,   // rho(X0*,C)
}
unsafe impl Send for Shared {}
unsafe impl Sync for Shared {}

/// 一对 (B,C) 一趟扫描的全部产出。
/// 除了两个 argmax（X0 按 h 最大、X0* 按 gap 最大），还带：
///   * 两个代表点各自的 (p,q) = (rho(X,B), rho(X,C))，供 (p,q) 条件空模型用；
///   * gap 对全体 X 的分布统计（relu 和、relu 平方和、>0 个数），供覆盖型分数用；
///   * sum_h_pos = Σ relu(h)、sum_gap_pos = Σ relu(gap)，供软质量归一化用。
#[derive(Clone, Copy)]
struct PairScan {
    bi: usize,   // X0 下标
    h: f32,      // rho(X0,U)
    single: f32, // max(p,q) at X0
    pb: f32,     // rho(X0,B)
    qc: f32,     // rho(X0,C)
    bsi: usize,  // X0* 下标
    hs: f32,     // h at X0*
    singles: f32,// max(p,q) at X0*
    pbs: f32,
    qcs: f32,
    sum_gap_pos: f32,   // Σ relu(gap)
    sumsq_gap_pos: f32, // Σ relu(gap)^2
    n_gap_pos: u32,     // #(gap>0)
    sum_h_pos: f32,     // Σ relu(h)
    n_valid: u32,       // 合法候选数
}

/// 一趟扫过所有 X，同时求两个 argmax 和 gap/h 的全 X 分布统计：
///   (1) X0 = argmax_X rho(X,U_BC)            —— 这对股票拼起来对谁最好，用户原规则
///   (2) X0* = argmax_X [rho(X,U_BC) - max(rho(X,B),rho(X,C))]
///                                            —— 这对股票在谁身上最能补上单只补不了的缺口
/// 两者一般不是同一只股票。
#[inline]
fn best_x(n: usize, p: &[f32], q: &[f32], wb: f32, wc: f32, b: usize, c: usize,
          mask: &[bool])
    -> PairScan {
    let mut s = PairScan {
        bi: usize::MAX, h: f32::NEG_INFINITY, single: 0.0, pb: 0.0, qc: 0.0,
        bsi: usize::MAX, hs: 0.0, singles: 0.0, pbs: 0.0, qcs: 0.0,
        sum_gap_pos: 0.0, sumsq_gap_pos: 0.0, n_gap_pos: 0,
        sum_h_pos: 0.0, n_valid: 0,
    };
    let mut bgs = f32::NEG_INFINITY;
    for x in 0..n {
        if x == b || x == c || !mask[x] {
            continue;
        }
        s.n_valid += 1;
        let h = wb * p[x] + wc * q[x];
        let single = if p[x] > q[x] { p[x] } else { q[x] };
        if h > s.h {
            s.h = h;
            s.bi = x;
            s.single = single;
            s.pb = p[x];
            s.qc = q[x];
        }
        let gain = h - single;
        if gain > bgs {
            bgs = gain;
            s.bsi = x;
            s.singles = single;
            s.hs = h;
            s.pbs = p[x];
            s.qcs = q[x];
        }
        // gap 覆盖统计与软质量分母，同一趟顺手做完
        if h > 0.0 {
            s.sum_h_pos += h;
        }
        if gain > 0.0 {
            s.sum_gap_pos += gain;
            s.sumsq_gap_pos += gain * gain;
            s.n_gap_pos += 1;
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

fn cmd_net(args: &[String]) {
    let t_all = std::time::Instant::now();
    let backup_root = PathBuf::from(&args[0]);
    let date: i64 = args[1].parse().expect("date");
    let outdir = PathBuf::from(&args[2]);
    let mats_arg = parse_flag(args, "--mat").unwrap_or("cnt_t1").to_string();
    let mats: Vec<String> = mats_arg.split(',').map(|s| s.trim().to_string()).collect();
    let null_mc: usize = parse_flag(args, "--null-mc").map(|s| s.parse().unwrap()).unwrap_or(20000);
    let topk: usize = parse_flag(args, "--topk").map(|s| s.parse().unwrap()).unwrap_or(15);
    if let Some(t) = parse_flag(args, "--threads") {
        let k: usize = t.parse().unwrap();
        let _ = rayon::ThreadPoolBuilder::new().num_threads(k).build_global();
    }

    let day = backup_root.join(date.to_string());
    let codes = read_codes(&day.join("codes.txt")).expect("读 codes.txt 失败");
    let n = codes.len();
    let ntrades = read_ntrades(&day.join("stats.bin")).expect("读 stats.bin 失败");
    if ntrades.len() != n {
        panic!("stats.bin 的 n={} 与 codes.txt 的 n={} 不一致", ntrades.len(), n);
    }
    eprintln!("[p1] n = {n}, date = {date}");

    let out = outdir.join(date.to_string());
    std::fs::create_dir_all(&out).expect("建输出目录失败");
    {
        let mut s = String::new();
        for c in &codes {
            s.push_str(c);
            s.push('\n');
        }
        std::fs::write(out.join("codes.txt"), s).unwrap();
    }

    // 成交笔数的十分位切点（用于空模型分层）
    let mut sorted_nt: Vec<f64> = ntrades.clone();
    sorted_nt.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let cuts: Vec<f64> = (1..10)
        .map(|k| sorted_nt[(k * n / 10).min(n - 1)])
        .collect();
    let dec: Vec<usize> = ntrades.iter().map(|&v| decile_of(v, &cuts)).collect();

    // ---- 股票白名单（项目规则：基金/债券不能进计算，且在计算之前筛）----
    let stocks_only = args.iter().any(|a| a == "--stocks-only");
    let basic = std::env::var("RUST_PYFUNC_BASIC_INFO_DIR")
        .unwrap_or_else(|_| "/ssd_data/data/basic_info".to_string());
    let mut is_stock = vec![true; n];
    if stocks_only {
        let sm = read_symbol_map(&Path::new(&basic).join("symbol_map.csv"))
            .expect("读 symbol_map.csv 失败，不允许退回全算");
        for i in 0..n {
            is_stock[i] = sm.contains(&codes[i]);
        }
        let n_bad = is_stock.iter().filter(|&&v| !v).count();
        eprintln!("[p1b] 非股票代码 {n_bad} 个（基金/债券），已从配对与 X 候选集里剔除");
        if n_bad == n {
            panic!("codes.txt 与股票表完全不相交，白名单对不上");
        }
    }
    let stock_idx: Vec<usize> = (0..n).filter(|&i| is_stock[i]).collect();

    let mut summary = serde_json::Map::new();
    summary.insert("n".into(), serde_json::json!(n));
    summary.insert("date".into(), serde_json::json!(date));
    summary.insert("codes_source".into(), serde_json::json!(day.join("codes.txt").to_string_lossy()));
    summary.insert("null_mc".into(), serde_json::json!(null_mc));
    summary.insert("stocks_only".into(), serde_json::json!(stocks_only));
    summary.insert("n_non_stock".into(),
        serde_json::json!(is_stock.iter().filter(|&&v| !v).count()));

    // 各矩阵之间完全独立：读盘、配对扫描、EXZ、落盘、汇报全部并行。
    // 每矩阵返回一份 summary 片段，最后按 --mat 顺序合并，保证输出确定。
    let n_mats = mats.len();
    let run_sums: Vec<(String, serde_json::Value)> = mats
        .par_iter()
        .map(|mat| {
        let t0 = std::time::Instant::now();
        let mpath = day.join("mats").join(format!("{mat}.bin"));
        let sdir = read_f32(&mpath).expect("读矩阵失败");
        if sdir.len() != n * n {
            panic!("矩阵 {mat} 有 {} 个元素，期望 {}", sdir.len(), n * n);
        }
        eprintln!("[p2:{mat}] 矩阵已读入 {:.1}s", t0.elapsed().as_secs_f32());

        // ---- E_i = 2 * S_dir[i][i] ----
        let e: Vec<f64> = (0..n).map(|i| 2.0 * sdir[i * n + i] as f64).collect();
        let n_bad = e.iter().filter(|&&v| !(v > 0.0)).count();
        eprintln!("[p2:{mat}] 自能量非正的股票 = {n_bad}");

        // ---- rho（行主序）与 rhoT（列读用）----
        let t0 = std::time::Instant::now();
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
        // 对称核自检：rho 必须在 [-1.001, 1.001]
        let mut over = 0usize;
        let mut maxabs = 0.0f32;
        for &v in rho.iter() {
            let a = v.abs();
            if a > maxabs {
                maxabs = a;
            }
            if a > 1.001 {
                over += 1;
            }
        }
        eprintln!("[p2:{mat}] |rho| 最大 = {maxabs:.5}, 超过 1.001 的格子 = {over}");
        // 转置：按输出行分块并行，rhot[j][i] = rho[i][j]
        let mut rhot = vec![0.0f32; n * n];
        {
            const B: usize = 64;
            let nb = n.div_ceil(B);
            rhot
                .par_chunks_mut(B * n)
                .enumerate()
                .for_each(|(bj, rows)| {
                    for (rj, row) in rows.chunks_mut(n).enumerate() {
                        let j = bj * B + rj;
                        for bi in 0..nb {
                            let i0 = bi * B;
                            let i1 = (i0 + B).min(n);
                            for i in i0..i1 {
                                row[i] = rho[i * n + j];
                            }
                        }
                    }
                });
        }
        eprintln!("[p2:{mat}] rho + 转置 {:.1}s", t0.elapsed().as_secs_f32());

        // ---- 全部配对的 X0 / RHO_U / SINGLE / G + 新增分数矩阵 + 软质量向量 ----
        let t0 = std::time::Instant::now();
        let mut g = vec![0.0f32; n * n];
        let mut ru = vec![0.0f32; n * n];
        let mut sg = vec![0.0f32; n * n];
        // X0 未计算的格子必须写 -1 哨兵：0 是合法下标（codes[0]），
        // 含非个股的配对约 107 万对若写 0，会被 reduce 当成"X0 是 codes[0]"污染聚合。
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
        // g2mass=emass 加权 gap²（算责任加权 std 用），smass/sgmass/sg2mass 是 gap 口径同款。
        // 按固定块切 B、块内顺序累加、按下标合并 → 确定性。
        const MBLK: usize = 256;
        let npair_eff;
        let mass_vectors: Vec<f32>; // 6×n：emass|gmass|g2mass|smass|sgmass|sg2mass（按对数归一）
        {
            let sh = Shared {
                n,
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
            let e_ref = &e;
            let rhot_ref = &rhot;
            let mask_ref = &is_stock;
            // sh 先取成引用再进闭包：否则 2021 版闭包会按字段拆开捕获，把 *mut f32 单独捕进去，
            // 那个类型不是 Sync。取成 &Shared 之后整体捕获，Shared 上已经 unsafe impl Sync。
            let sh_ref: &Shared = &sh;
            let nb = MBLK.min(n).max(1);
            let bounds: Vec<(usize, usize)> =
                (0..nb).map(|k| (k * n / nb, (k + 1) * n / nb)).collect();
            // -inf 掩码：非个股位置加 -inf，使 h/gap 天然落到 argmax 之外、relu 为 0。
            // 比在每对里逐个判断 !mask[x] 快得多，还让 fill_h 可以整行 SIMD。
            let addm: Vec<f32> = (0..n)
                .map(|i| if is_stock[i] { 0.0 } else { f32::NEG_INFINITY })
                .collect();
            // 合法候选数对每对都一样：全体个股减去 B、C 自身
            let n_stocks = is_stock.iter().filter(|&&v| v).count();
            let n_valid_pair = (n_stocks as u32).saturating_sub(2);
            // f32 分块累加器：每块私有、块内顺序累加、最后按下标合并——确定性不变。
            // 用 f32 是因为六份 f64 的每对每 x 读写是最贵的内存流量，砍半立竿见影。
            type Mass = (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, u64);
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
                    // 本块私有的 h/gap 物化缓冲，热路径上不再分配
                    let mut hb = vec![0.0f32; n];
                    let mut gb = vec![0.0f32; n];
                    for b in b0..b1 {
                        if !mask_ref[b] {
                            continue;
                        }
                        let p = &rhot_ref[b * n..b * n + n];
                        let eb = e_ref[b];
                        let sb = eb.sqrt();
                        for c in (b + 1)..n {
                            if !mask_ref[c] {
                                continue;
                            }
                            let ec = e_ref[c];
                            let sc = ec.sqrt();
                            let r = rhot_ref[b * n + c] as f64; // rho[b][c]
                            let d = (eb + ec + 2.0 * r * sb * sc).sqrt();
                            if !(d > 0.0) {
                                continue;
                            }
                            let wb = (sb / d) as f32;
                            let wc = (sc / d) as f32;
                            let q = &rhot_ref[c * n..c * n + n];
                            // 一趟物化：hb = wb*p + wc*q + addm；gb = hb - max(p,q)。
                            // 非个股与 B/C 自己写 -inf，argmax/relu 自然排除。
                            crate::simd::fill_h(&mut hb, wb, wc, p, q, &addm);
                            crate::simd::fill_gap(&mut gb, &hb, p, q);
                            hb[b] = f32::NEG_INFINITY;
                            hb[c] = f32::NEG_INFINITY;
                            gb[b] = f32::NEG_INFINITY;
                            gb[c] = f32::NEG_INFINITY;
                            // X0 = argmax h；X0* = argmax gap（同取下标小的在前）
                            let (bi, h) = crate::simd::argmax_first(&hb);
                            if bi == usize::MAX {
                                continue;
                            }
                            let (bsi, gs_) = crate::simd::argmax_first(&gb);
                            let (sh_pos, sg_pos, sqg_pos, npos) =
                                crate::simd::pos_stats(&hb, &gb);
                            let (pbi, qci) = (p[bi], q[bi]);
                            let single = if pbi > qci { pbi } else { qci };
                            let (pbs, qcs) = (p[bsi], q[bsi]);
                            let singles = if pbs > qcs { pbs } else { qcs };
                            let gain = h - single;
                            // GSTAR = argmax-gap 处的 gap 值本身（= h - max(p,q)，
                            // 不要再减 singles）；RHO_US = 该处的 h = gap + singles。
                            let gains = gs_;
                            let rus_v = gs_ + singles;
                            let nv = n_valid_pair.max(1) as f32;
                            let grel_v = if single > 1e-4 {
                                gain / single
                            } else {
                                f32::NAN
                            };
                            let gsh_v = if h > 1e-4 {
                                gain / h
                            } else {
                                f32::NAN
                            };
                            let ghhi_v = if sg_pos > 0.0 {
                                sqg_pos / (sg_pos * sg_pos)
                            } else {
                                f32::NAN
                            };
                            unsafe {
                                let put = |m: *mut f32, v: f32| {
                                    *m.add(b * n + c) = v;
                                    *m.add(c * n + b) = v;
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
                            // 软质量分配（每对把 1 份质量按 relu(h) 占比摊给全体 X；
                            // gap 口径同理按 relu(gap) 摊）。这是 explain_* 族低覆盖的根治：
                            // 不再只有 argmax 那只股票被记到，每只跟配对形状沾边的 X 都分到质量。
                            if sh_pos > 0.0 {
                                crate::simd::mass_acc_h(
                                    &mut em, &mut gm, &mut g2m, &hb, &gb, 1.0 / sh_pos,
                                );
                            }
                            if sg_pos > 0.0 {
                                crate::simd::mass_acc_g(
                                    &mut sm, &mut sgm, &mut sg2m, &gb, 1.0 / sg_pos,
                                );
                            }
                        }
                    }
                    (em, gm, g2m, sm, sgm, sg2m, npair)
                })
                .collect();
            let mut emass = vec![0.0f64; n];
            let mut gmass = vec![0.0f64; n];
            let mut g2mass = vec![0.0f64; n];
            let mut smass = vec![0.0f64; n];
            let mut sgmass = vec![0.0f64; n];
            let mut sg2mass = vec![0.0f64; n];
            let mut npair_tot = 0u64;
            // 按下标顺序合并，保证确定性
            for (em, gm, g2m, sm, sgm, sg2m, np) in parts.iter() {
                for x in 0..n {
                    emass[x] += em[x] as f64;
                    gmass[x] += gm[x] as f64;
                    g2mass[x] += g2m[x] as f64;
                    smass[x] += sm[x] as f64;
                    sgmass[x] += sgm[x] as f64;
                    sg2mass[x] += sg2m[x] as f64;
                }
                npair_tot += np;
            }
            npair_eff = npair_tot;
            // 归一化成"平均每对分到的质量/增益"，写进 mass_vectors
            let inv_p = if npair_tot > 0 { 1.0 / npair_tot as f64 } else { 0.0 };
            let mut mv = Vec::with_capacity(6 * n);
            for arr in [&emass, &gmass, &g2mass, &smass, &sgmass, &sg2mass] {
                for x in 0..n {
                    mv.push((arr[x] * inv_p) as f32);
                }
            }
            mass_vectors = mv;
        }
        eprintln!(
            "[p2:{mat}] 全部 {} 对配对（有效 {}）+ 软质量 {:.1}s",
            n * (n - 1) / 2,
            npair_eff,
            t0.elapsed().as_secs_f32()
        );

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
            let eb = e[b];
            let ec = e[c];
            let sb = eb.sqrt();
            let sc = ec.sqrt();
            let r = rhot[b * n + c] as f64;
            let d = (eb + ec + 2.0 * r * sb * sc).sqrt();
            if !(d > 0.0) {
                continue;
            }
            let p = &rhot[b * n..b * n + n];
            let q = &rhot[c * n..c * n + n];
            let s = best_x(n, p, q, (sb / d) as f32, (sc / d) as f32, b, c, &is_stock);
            if s.bi == usize::MAX || s.bsi == usize::MAX {
                continue;
            }
            // 分层键必须与 (b,c) 的顺序无关，否则 Z 矩阵不对称（实测最大偏差 18.0）
            let (d1, d2) = if dec[b] <= dec[c] { (dec[b], dec[c]) } else { (dec[c], dec[b]) };
            samples.push((d1 * 10 + d2, s.h - s.single, s.hs - s.singles));
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
        eprintln!(
            "[p2:{mat}] 空模型 {} 个抽样配对, 全局 G 均值 {:.5} 标准差 {:.5}; G* 均值 {:.5} 标准差 {:.5} {:.1}s",
            gcnt, gmean, gstd, hmean, hstd, t0.elapsed().as_secs_f32()
        );

        // ---- Z = (G - 层均值) / 层标准差；ZS 同法，用 G* ----
        let mut z = vec![0.0f32; n * n];
        let mut zs = vec![0.0f32; n * n];
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    continue;
                }
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

        // ---- (p,q) 条件空模型：EXZ / EXZS ----
        // 规格书 7.1 的结论：G 的 99% 方差来自 (p,q) 剖面（两只股各自跟 X0 有多像），
        // 配对自身的互补特质只剩零头。这里把 Z 的分层变量从"活跃度档"换成
        // "(min(p,q), max(p,q)) 十分位联合网格"：在同一格里的配对，两只成员对 X0 的
        // 相似度水平相同，Z 扣掉的就是相似度水平效应，剩下的是 B、C 之间低重叠
        // 带来的那部分（r 的信息），也就是真正的"互补性显著度"。
        // 不需要蒙特卡洛：真实配对本身就铺满网格，直接用同格配对的经验均值/标准差。
        let t0 = std::time::Instant::now();
        // (min(p,q), max(p,q)) 各 bins×bins 联合分位网格，参数 --exz-bins（默认 10）
        let bins: usize = parse_flag(args, "--exz-bins")
            .map(|s| s.parse().expect("--exz-bins"))
            .unwrap_or(10);
        let ncell = bins * bins;
        let cond_std = |pmat: &[f32], qmat: &[f32], wmat: &[f32]| -> Vec<f32> {
            // u = min(p,q), v = max(p,q) 的分位切点（只在合法配对上取）。
            // 三个 n² 循环全部 rayon 并行：行块 fold/reduce 合并，确定性与串行版一致
            // （计数/求和都是按格累加，合并顺序固定为行块下标序）。
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
            let mut su = us.clone();
            let mut sv = vs.clone();
            su.par_sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
            sv.par_sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
            let cuts_u: Vec<f64> = (1..bins).map(|k| su[(k * m / bins).min(m - 1)]).collect();
            let cuts_v: Vec<f64> = (1..bins).map(|k| sv[(k * m / bins).min(m - 1)]).collect();
            let cell_of = |u: f64, v: f64| -> usize {
                // 分位格下标：第一个 >= 该值的切点位置（与旧 decile_of 语义一致）
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
            // 上三角写完后镜像到下三角
            for i in 0..n {
                for j in 0..i {
                    out[i * n + j] = out[j * n + i];
                }
            }
            out
        };
        let (exz, exzs) = rayon::join(
            || cond_std(&pbv, &qcv, &g),
            || cond_std(&pbsv, &qcsv, &gs),
        );
        eprintln!("[p2:{mat}] (p,q) 条件标准化 EXZ/EXZS {:.1}s", t0.elapsed().as_secs_f32());

        // ---- 落盘（23 个文件并行写，磁盘带宽能同时吃多路）----
        let t0 = std::time::Instant::now();
        let tag = if n_mats > 1 { format!("_{mat}") } else { String::new() };
        {
            let writes: Vec<(&str, &[f32])> = vec![
                ("G", &g), ("RHO_U", &ru), ("SINGLE", &sg), ("X0", &x0),
                ("Z", &z), ("GSTAR", &gs), ("RHO_US", &rus), ("SINGLE_S", &sgs),
                ("X0STAR", &x0s), ("ZS", &zs), ("GREL", &grel), ("GSH", &gsh),
                ("GCOV", &gcov), ("GNPOS", &gnpos), ("GHHI", &ghhi),
                ("PB", &pbv), ("QC", &qcv), ("PBS", &pbsv), ("QCS", &qcsv),
                ("EXZ", &exz), ("EXZS", &exzs), ("rho", &rho),
            ];
            writes.par_iter().for_each(|(nm, v)| {
                write_f32(&out.join(format!("{nm}{tag}.bin")), v).unwrap();
            });
            write_f64(&out.join(format!("E{tag}.bin")), &e).unwrap();
            // 软质量向量：emass|gmass|g2mass|smass|sgmass|sg2mass 各 n 个 f32
            write_f32(&out.join(format!("MASS{tag}.bin")), &mass_vectors).unwrap();
        }
        eprintln!("[p2:{mat}] 落盘 {:.1}s", t0.elapsed().as_secs_f32());
        let t0 = std::time::Instant::now();

        // ---- 汇报用：按 G、Z、ZS 的头部配对 ----
        // 注意变量名不要和上面的 gs/zs 矩阵撞，这里统一用 gq/zq 存分位用的向量。
        let pairs: Vec<(usize, usize, f32, f32, f32, usize, f32, f32, usize)> = (0..n)
            .into_par_iter()
            .map(|b| {
                let mut v = Vec::new();
                for c in (b + 1)..n {
                    let bi = x0[b * n + c] as usize;
                    if bi >= n {
                        continue;
                    }
                    let bsi = x0s[b * n + c] as usize;
                    v.push((
                        b, c, g[b * n + c], z[b * n + c], ru[b * n + c], bi,
                        zs[b * n + c], gs[b * n + c], bsi,
                    ));
                }
                v
            })
            .reduce(Vec::new, |mut a, mut b| {
                a.append(&mut b);
                a
            });
        // top-15 只需要部分排序：select_nth 切出头部再对小段排序，
        // 分位数同理只要次序统计量，不做 780 万元组的全排序。
        let top15 = |v: &mut Vec<(usize, usize, f32, f32, f32, usize, f32, f32, usize)>,
                     key: fn(&(_, _, f32, f32, f32, usize, f32, f32, usize)) -> f32| {
            let tk = topk.min(v.len());
            if tk < v.len() {
                v.select_nth_unstable_by(tk, |a, b| key(b).partial_cmp(&key(a)).unwrap());
            }
            v[..tk].sort_unstable_by(|a, b| key(b).partial_cmp(&key(a)).unwrap());
        };
        let mut by_g = pairs.clone();
        top15(&mut by_g, |t| t.2);
        let mut by_z = pairs.clone();
        top15(&mut by_z, |t| t.3);
        let mut by_zs = pairs.clone();
        top15(&mut by_zs, |t| t.6);
        let gq: Vec<f32> = pairs.iter().map(|p| p.2).collect();
        let zq: Vec<f32> = pairs.iter().map(|p| p.3).collect();
        let zsq: Vec<f32> = pairs.iter().map(|p| p.6).collect();
        let q = |v: &[f32], p: f64| -> f32 {
            let mut s = v.to_vec();
            let k = ((s.len() as f64) * p) as usize % s.len();
            s.select_nth_unstable_by(k, |a, b| a.partial_cmp(b).unwrap());
            s[k]
        };
        eprintln!(
            "[p2:{mat}] G 分位 p10/p50/p90 = {:.4}/{:.4}/{:.4};  Z>2 占比 {:.3}%  ZS>2 占比 {:.3}%",
            q(&gq, 0.10), q(&gq, 0.50), q(&gq, 0.90),
            100.0 * zq.iter().filter(|&&v| v > 2.0).count() as f64 / zq.len() as f64,
            100.0 * zsq.iter().filter(|&&v| v > 2.0).count() as f64 / zsq.len() as f64,
        );
        eprintln!("[p2:{mat}] 按 G 最大的 {} 对:", topk);
        for t in by_g.iter().take(topk) {
            eprintln!(
                "   {} {}  G={:.4} Z={:.2} ZS={:.2} rhoU={:.4} X0={}",
                codes[t.0], codes[t.1], t.2, t.3, t.6, t.4, codes[t.5]
            );
        }
        eprintln!("[p2:{mat}] 按 Z 最大的 {} 对:", topk);
        for t in by_z.iter().take(topk) {
            eprintln!(
                "   {} {}  Z={:.2} G={:.4} ZS={:.2} rhoU={:.4} X0={}",
                codes[t.0], codes[t.1], t.3, t.2, t.6, t.4, codes[t.5]
            );
        }
        eprintln!("[p2:{mat}] 按 ZS（最大补缺口）最大的 {} 对:", topk);
        for t in by_zs.iter().take(topk) {
            eprintln!(
                "   {} {}  ZS={:.2} G*={:.4} Z={:.2} X0*={}",
                codes[t.0], codes[t.1], t.6, t.7, t.3, codes[t.8]
            );
        }
        eprintln!("[p2:{mat}] 汇报段 {:.1}s", t0.elapsed().as_secs_f32());

        let strata_json: Vec<serde_json::Value> = strata
            .iter()
            .enumerate()
            .map(|(s, &(m, sd, c))| {
                serde_json::json!({"stratum": s, "decile_b": s / 10, "decile_c": s % 10,
                                   "mean": m, "std": sd, "count": c})
            })
            .collect();
        (
            mat.clone(),
            serde_json::json!({
                "mat": mat,
                "n": n,
                "abs_rho_max": maxabs,
                "rho_over_1": over,
                "n_bad_energy": n_bad,
                "G_p10": q(&gq, 0.10), "G_p50": q(&gq, 0.50), "G_p90": q(&gq, 0.90),
                "Z_gt2_share": zq.iter().filter(|&&v| v > 2.0).count() as f64 / zq.len() as f64,
                "ZS_gt2_share": zsq.iter().filter(|&&v| v > 2.0).count() as f64 / zsq.len() as f64,
                "null_global_mean": gmean, "null_global_std": gstd,
                "null_global_mean_star": hmean, "null_global_std_star": hstd,
                "null_samples": gcnt,
                "strata": strata_json,
                "topG": by_g.iter().take(topk).map(|t| serde_json::json!({
                    "b": codes[t.0], "c": codes[t.1], "G": t.2, "Z": t.3, "ZS": t.6,
                    "rhoU": t.4, "X0": codes[t.5]})).collect::<Vec<_>>(),
                "topZ": by_z.iter().take(topk).map(|t| serde_json::json!({
                    "b": codes[t.0], "c": codes[t.1], "Z": t.3, "G": t.2, "ZS": t.6,
                    "rhoU": t.4, "X0": codes[t.5]})).collect::<Vec<_>>(),
                "topZS": by_zs.iter().take(topk).map(|t| serde_json::json!({
                    "b": codes[t.0], "c": codes[t.1], "ZS": t.6, "Gstar": t.7, "Z": t.3,
                    "X0star": codes[t.8]})).collect::<Vec<_>>(),
            }),
        )
        })
        .collect();
    for (m, v) in run_sums {
        summary.insert(format!("run_{m}"), v);
    }

    std::fs::write(
        out.join("summary.json"),
        serde_json::to_string_pretty(&summary).unwrap(),
    )
    .unwrap();
    eprintln!("[done] {:.1}s, 输出目录 {}", t_all.elapsed().as_secs_f32(), out.display());
    println!(
        "{{\"ok\":true,\"date\":{date},\"n\":{n},\"outdir\":\"{}\"}}",
        out.display()
    );
}

fn main() {
    // 参数个数的检查与原来完全一样（net 与 reduce 都要 5 个），net 的行为不动
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 5 {
        eprintln!("用法: yupei_pair net <backup_root> <date> <outdir> [--mat cnt_t1] [--null-mc 20000] [--threads 200]");
        eprintln!("      yupei_pair reduce <pair_out_dir> <date> <outdir> [--backup <root>] [--threads N] [--mat cnt_t1,...]");
        eprintln!("      yupei_pair topk <backup_root> <date> <outdir> [--mat cnt_t1] [--threads N] [--stocks-only] [--k-list 1,2,5,10,20,50,100]");
        eprintln!("      yupei_pair check <pair_out_dir> <date> [--mat cnt_t1]");
        std::process::exit(1);
    }
    match args[1].as_str() {
        "net" => cmd_net(&args[2..]),
        "reduce" => reduce::cmd_reduce(&args[2..]),
        "topk" => topk::cmd_topk(&args[2..]),
        "check" => check::cmd_check(&args[2..]),
        other => {
            eprintln!("未知子命令: {other}");
            std::process::exit(1);
        }
    }
}
