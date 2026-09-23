//! switch_moment 横截面因子（hm95「切换时刻贡献」的 Rust 版）。
//!
//! 输入一个交易日，输出当天全市场每只股票的 15067 个因子值。
//! 由 Level2 还原分钟字段；旧分钟库的历史缺失值口径无法仅从 Level2 完全推断，
//! 因此历史全量结果不保证与 Python go 函数版逐位一致。
//!
//! 计算流程：Level2 逐笔成交和盘口快照先按旧分钟 H5 的时间边界还原 22 个字段，
//! 再派生 19 个字段并计算每字段 793 个横截面因子。股票轴来自 basic_info/symbol_map.csv。
//! 历史分钟 H5 只用于验收，不参与正式计算。
//!
//! 照搬 numpy 语义的两处：`np.divide(a, b, out=zeros, where=b!=0)` 在 b=0 时取 0；
//! `np.sign` 在 ±0 上返回 ±0（不是 ±1）。

use crate::corr_contribution_factors::compute_all_factors;
use ndarray::Array2;
use pyo3::prelude::*;
use rayon::prelude::*;
use std::collections::HashMap;
use std::io;
use std::sync::{Arc, LazyLock, Mutex};

/// 贡献度因子取相关最高/最低的 K 对。
const TOP_K: usize = 300;
/// 每个字段派生的基础因子数。
const N_BASE: usize = 793;
/// 派生字段数。
const N_FIELDS: usize = 19;
/// 每只股票的因子总数 = 19 × 793。
pub const N_FACTORS: usize = N_FIELDS * N_BASE;

/// go() 里 fields 字典的 19 个键（未排序；实际计算按字典序）。
const FIELD_NAMES: [&str; 19] = [
    "buy_ratio",
    "turnover",
    "spread",
    "vol_ratio",
    "amplitude",
    "min_ret",
    "net_buy_ratio",
    "buy_count_ratio",
    "up_tick_ratio",
    "buy_sell_size_ratio",
    "obi_1",
    "obi_10",
    "depth_ratio",
    "bid_concentration",
    "vwap_dev",
    "trade_order_div",
    "price_vol_sync",
    "effective_spread",
    "bid_slope",
];

const STATS8: [&str; 8] = ["mean", "std", "skew", "kurt", "p5", "p95", "trend", "ac1"];
const STATS7: [&str; 7] = ["std", "skew", "kurt", "p5", "p95", "trend", "ac1"];
const METHODS: [&str; 4] = ["m1", "m2", "m3", "m4"];
const TYPES4: [&str; 4] = ["full", "prod", "top", "bot"];
const TYPES_HA: [&str; 2] = ["full", "prod"];

// ============================================================
// 因子名
// ============================================================

fn sorted_field_names() -> Vec<&'static str> {
    let mut v: Vec<&'static str> = FIELD_NAMES.to_vec();
    v.sort_unstable();
    v
}

/// 793 个基础因子名，顺序与 hm95.py 的 _gen_base_793() 完全一致。
/// 注意：compute_all_factors 实际产出的**顺序**与这里不同（ha_time_* / ha_mag_* 交错），
/// 名字集合相同（见单测），所以取值时按名字定位。
pub fn switch_moment_base_names() -> Vec<String> {
    let mut names: Vec<String> = Vec::with_capacity(N_BASE);
    for m in METHODS {
        for t in TYPES4 {
            for s in STATS8 {
                names.push(format!("{m}_{t}_{s}"));
            }
            for s in STATS8 {
                names.push(format!("{m}_{t}_cm_{s}"));
            }
            names.push(format!("{m}_{t}_fcorr"));
        }
        for t in TYPES_HA {
            names.push(format!("{m}_{t}_ha_corr"));
            for s in STATS8 {
                names.push(format!("{m}_{t}_ha_time_{s}"));
            }
            for s in STATS8 {
                names.push(format!("{m}_{t}_ha_mag_{s}"));
            }
        }
    }
    for s in STATS7 {
        names.push(format!("orth1_full_{s}"));
    }
    for s in STATS8 {
        names.push(format!("orth1_full_cm_{s}"));
    }
    for sub in ["top", "bot"] {
        for s in STATS8 {
            names.push(format!("orth1_full_{sub}_{s}"));
        }
        for s in STATS8 {
            names.push(format!("orth1_full_{sub}_cm_{s}"));
        }
    }
    for sub in ["top", "bot"] {
        for s in STATS7 {
            names.push(format!("orth1_{sub}_{s}"));
        }
        for s in STATS8 {
            names.push(format!("orth1_{sub}_cm_{s}"));
        }
    }
    for m in METHODS {
        for s in STATS7 {
            names.push(format!("{m}_orth2_full_{s}"));
        }
        for s in STATS8 {
            names.push(format!("{m}_orth2_full_cm_{s}"));
        }
        for sub in ["top", "bot"] {
            for s in STATS8 {
                names.push(format!("{m}_orth2_full_{sub}_{s}"));
            }
            for s in STATS8 {
                names.push(format!("{m}_orth2_full_{sub}_cm_{s}"));
            }
        }
        for sub in ["top", "bot"] {
            for s in STATS7 {
                names.push(format!("{m}_orth2_{sub}_{s}"));
            }
            for s in STATS8 {
                names.push(format!("{m}_orth2_{sub}_cm_{s}"));
            }
        }
    }
    names
}

/// 15067 个因子名：字段字典序 × 793 基础名序。
pub fn switch_moment_names() -> Vec<String> {
    let fields = sorted_field_names();
    let base = switch_moment_base_names();
    let mut out = Vec::with_capacity(N_FACTORS);
    for f in &fields {
        for b in &base {
            out.push(format!("{f}_{b}"));
        }
    }
    out
}

// ============================================================
// 逐元素算子（严格照搬 numpy 语义）
// ============================================================

/// np.divide(a, b, out=zeros, where=b != 0)：b 为 0（含 -0）时取 0，否则 a/b。
#[inline]
fn sd(a: f64, b: f64) -> f64 {
    if b != 0.0 {
        a / b
    } else {
        0.0
    }
}

/// np.sign：±0 返回 ±0，NaN 返回 NaN（不是 ±1）。
#[inline]
fn np_sign(x: f64) -> f64 {
    if x > 0.0 {
        1.0
    } else if x < 0.0 {
        -1.0
    } else {
        x
    }
}

/// np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
#[inline]
fn nan_to_zero(x: f64) -> f64 {
    if x.is_finite() {
        x
    } else {
        0.0
    }
}

// ============================================================
// 原始字段读取 + 派生字段
// ============================================================

#[inline]
fn get<'a>(raw: &'a HashMap<&'static str, Vec<f64>>, name: &str) -> &'a [f64] {
    raw.get(name)
        .expect("原始字段缺失（Level2 字段与派生字段不一致）")
        .as_slice()
}

/// 派生 19 个字段，按字段名字典序返回。
fn derive_fields(
    raw: &HashMap<&'static str, Vec<f64>>,
    t: usize,
    n: usize,
) -> Vec<(&'static str, Vec<f64>)> {
    let len = t * n;
    let mut out: Vec<(&'static str, Vec<f64>)> = Vec::with_capacity(N_FIELDS);

    for name in sorted_field_names() {
        let vals: Vec<f64> = match name {
            "buy_ratio" => {
                let a = get(raw, "act_buy_amount_sum");
                let b = get(raw, "amount");
                (0..len).map(|i| sd(a[i], b[i])).collect()
            }
            "turnover" => get(raw, "turnover").to_vec(),
            "spread" => get(raw, "spread_over_tick_size_mean").to_vec(),
            "vol_ratio" => {
                let v = get(raw, "volume");
                let mut denom = vec![0.0f64; n];
                for row in 0..t {
                    let base = row * n;
                    for c in 0..n {
                        denom[c] += v[base + c];
                    }
                }
                (0..len).map(|i| sd(v[i], denom[i % n])).collect()
            }
            "amplitude" => {
                let hi = get(raw, "high");
                let lo = get(raw, "low");
                let cl = get(raw, "close");
                (0..len).map(|i| sd(hi[i] - lo[i], cl[i])).collect()
            }
            "min_ret" => {
                let cl = get(raw, "close");
                let mut v = vec![0.0f64; len];
                for row in 1..t {
                    let cur = row * n;
                    let prev = (row - 1) * n;
                    for c in 0..n {
                        v[cur + c] = sd(cl[cur + c] - cl[prev + c], cl[prev + c]);
                    }
                }
                v
            }
            "net_buy_ratio" => {
                let a = get(raw, "act_buy_amount_sum");
                let b = get(raw, "act_sell_amount_sum");
                let d = get(raw, "amount");
                (0..len).map(|i| sd(a[i] - b[i], d[i])).collect()
            }
            "buy_count_ratio" => {
                let a = get(raw, "act_buy_count_sum");
                let b = get(raw, "act_sell_count_sum");
                (0..len).map(|i| sd(a[i], a[i] + b[i])).collect()
            }
            "up_tick_ratio" => {
                let a = get(raw, "up_tick_count");
                let b = get(raw, "down_tick_count");
                (0..len).map(|i| sd(a[i], a[i] + b[i])).collect()
            }
            "buy_sell_size_ratio" => {
                let ba = get(raw, "act_buy_amount_sum");
                let bc = get(raw, "act_buy_count_sum");
                let sa = get(raw, "act_sell_amount_sum");
                let sc = get(raw, "act_sell_count_sum");
                (0..len)
                    .map(|i| sd(sd(ba[i], bc[i]), sd(sa[i], sc[i])))
                    .collect()
            }
            "obi_1" => {
                let a = get(raw, "bid_vol1");
                let b = get(raw, "ask_vol1");
                (0..len).map(|i| sd(a[i] - b[i], a[i] + b[i])).collect()
            }
            "obi_10" => {
                let a = get(raw, "bid_size_10_mean");
                let b = get(raw, "ask_size_10_mean");
                (0..len).map(|i| sd(a[i] - b[i], a[i] + b[i])).collect()
            }
            "depth_ratio" => {
                let a = get(raw, "bid_size_10_mean");
                let b = get(raw, "ask_size_10_mean");
                (0..len).map(|i| sd(a[i], b[i])).collect()
            }
            "bid_concentration" => {
                let a = get(raw, "bid_vol1");
                let b = get(raw, "bid_size_10_mean");
                (0..len).map(|i| sd(a[i], b[i])).collect()
            }
            "vwap_dev" => {
                let av = get(raw, "ask_vwap10");
                let bv = get(raw, "bid_vwap10");
                let ap = get(raw, "ask_prc1");
                let bp = get(raw, "bid_prc1");
                let cl = get(raw, "close");
                (0..len)
                    .map(|i| {
                        let mid_vwap = (av[i] + bv[i]) / 2.0;
                        let mid_prc = (ap[i] + bp[i]) / 2.0;
                        sd(mid_vwap - mid_prc, cl[i])
                    })
                    .collect()
            }
            "trade_order_div" => {
                let a = get(raw, "act_buy_amount_sum");
                let b = get(raw, "act_sell_amount_sum");
                let d = get(raw, "amount");
                let ba = get(raw, "bid_size_10_mean");
                let bz = get(raw, "ask_size_10_mean");
                (0..len)
                    .map(|i| {
                        let net_buy_ratio = sd(a[i] - b[i], d[i]);
                        let obi_10 = sd(ba[i] - bz[i], ba[i] + bz[i]);
                        -net_buy_ratio * obi_10
                    })
                    .collect()
            }
            "price_vol_sync" => {
                let cl = get(raw, "close");
                let v = get(raw, "volume");
                let mut denom = vec![0.0f64; n];
                for row in 0..t {
                    let base = row * n;
                    for c in 0..n {
                        denom[c] += v[base + c];
                    }
                }
                let mut out = vec![0.0f64; len];
                for row in 1..t {
                    let cur = row * n;
                    let prev = (row - 1) * n;
                    for c in 0..n {
                        let r = sd(cl[cur + c] - cl[prev + c], cl[prev + c]);
                        out[cur + c] = np_sign(r) * sd(v[cur + c], denom[c]);
                    }
                }
                out
            }
            "effective_spread" => {
                let cl = get(raw, "close");
                let bp = get(raw, "bid_prc1");
                let ap = get(raw, "ask_prc1");
                (0..len)
                    .map(|i| {
                        let mid = (bp[i] + ap[i]) / 2.0;
                        sd((cl[i] - mid).abs(), cl[i])
                    })
                    .collect()
            }
            "bid_slope" => {
                let a = get(raw, "bid_vol1");
                let b = get(raw, "bid_vol5");
                (0..len).map(|i| sd(a[i], b[i])).collect()
            }
            other => unreachable!("未实现的派生字段: {other}"),
        };
        out.push((name, vals));
    }
    out
}

// ============================================================
// 主计算
// ============================================================

/// 算一个字段的 793 个因子，返回按基础因子名下标排列的 f32 值
/// （长度 = N_BASE × n，第 bi 个因子的值在 `[bi*n .. (bi+1)*n]`）。
fn compute_one_field(
    fname: &'static str,
    mut vals: Vec<f64>,
    t: usize,
    n: usize,
    base_idx: &HashMap<String, usize>,
    base: &[String],
) -> io::Result<Vec<f32>> {
    for v in vals.iter_mut() {
        *v = nan_to_zero(*v);
    }
    let arr = Array2::from_shape_vec((t, n), vals)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;
    let facs = compute_all_factors(arr, TOP_K, true);
    if facs.len() != N_BASE {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("字段 {fname} 产出 {} 个因子，期望 {N_BASE}", facs.len()),
        ));
    }
    let mut out = vec![f32::NAN; N_BASE * n];
    let mut seen = vec![false; N_BASE];
    for (nm, values) in facs.iter() {
        let bi = *base_idx.get(nm.as_str()).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("字段 {fname} 产出未知的基础因子名 {nm}"),
            )
        })?;
        if std::mem::replace(&mut seen[bi], true) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("字段 {fname} 的基础因子名 {nm} 重复"),
            ));
        }
        let dst = &mut out[bi * n..(bi + 1) * n];
        for s in 0..n {
            dst[s] = values[s] as f32;
        }
    }
    if let Some(miss) = seen.iter().position(|&s| !s) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("字段 {fname} 缺少基础因子 {}", base[miss]),
        ));
    }
    Ok(out)
}

/// 本进程可用的线程预算。优先读 RAYON_NUM_THREADS（pipeline worker 由
/// run_factor_pipeline_cross_section 设成 threads_per_worker），否则按机器核数。
/// 故意不碰 rayon 的全局池：本模块只用自己建的字段级小池，全局池一旦初始化就多一批线程。
fn thread_budget() -> usize {
    if let Ok(v) = std::env::var("RAYON_NUM_THREADS") {
        if let Ok(v) = v.parse::<usize>() {
            if v >= 1 {
                return v;
            }
        }
    }
    std::thread::available_parallelism()
        .map(|x| x.get())
        .unwrap_or(1)
}

/// 一个 worker 进程真正能喂满的线程数：19 个字段各 1 线程。
/// 实测 1 线程/字段时线程 ≈100% 忙，2 线程/字段只有 ~65% 忙，6 线程只有 ~34% 忙
/// （单字段内部约 69% 的活是串行的），所以「多给线程」换不来 CPU，只会让线程更闲。
/// 要让 n_jobs 等于实际 CPU 占用，就得**每字段 1 线程**，再用进程数去凑 n_jobs。
pub const WORKER_SATURATION: usize = N_FIELDS;

/// 父进程在批量阶段真正吃 CPU 的线程数（writer 池，见 factor_pipeline 的
/// WRITER_POOL_THREADS = 2）。要从 n_jobs 里扣掉，保证「实际 CPU 绝不超过 n_jobs」。
/// 实测 writer 是 I/O 为主、CPU 只在追加那几秒出现，所以扣 2 已经够保守。
pub const PARENT_CPU_THREADS: usize = 2;

/// 给定 n_jobs，选让「实际 CPU 最贴近 n_jobs」的 worker 进程数。
///
/// 每个 worker 最多喂满 WORKER_SATURATION(=19) 个线程（19 个字段各 1 线程），
/// 所以「能覆盖的 CPU」= w × min(19, 预算/w)，它随 w 增大而增大（受 19 封顶）。
/// 先求出预算内能覆盖的最大 CPU，再取「覆盖到该最大值 95% 以上」里 worker 数最少的那个
/// —— 既贴近 n_jobs，又不用把进程数堆到几百。
pub fn recommended_workers(n_jobs: usize) -> usize {
    let budget = n_jobs.saturating_sub(PARENT_CPU_THREADS).max(1);
    let max_cpu = (1..=256usize)
        .map_while(|w| {
            let per = budget / w;
            if per == 0 {
                None
            } else {
                Some(w * per.min(WORKER_SATURATION))
            }
        })
        .max()
        .unwrap_or(1);
    let threshold = max_cpu * 95 / 100;
    for w in 1..=256usize {
        let per = budget / w;
        if per == 0 {
            break;
        }
        if w * per.min(WORKER_SATURATION) >= threshold {
            return w;
        }
    }
    1
}

/// 给定 n_jobs 与 worker 数，每个 worker 的 CPU 预算（= 并发字段数，每字段固定 1 线程）。
/// 保证 n_workers × 本值 + PARENT_CPU_THREADS ≤ n_jobs。
pub fn worker_cpu_budget(n_jobs: usize, n_workers: usize) -> usize {
    (n_jobs.saturating_sub(PARENT_CPU_THREADS) / n_workers.max(1)).clamp(1, WORKER_SATURATION)
}

/// 并发字段数。默认 = min(19, 预算)——每字段 1 线程，线程满载，实际 CPU ≈ 预算。
/// 可用 RUST_PYFUNC_SWITCH_MOMENT_FIELD_CONC 覆盖（实验用）。
fn field_concurrency(usable: usize) -> usize {
    if let Ok(v) = std::env::var("RUST_PYFUNC_SWITCH_MOMENT_FIELD_CONC") {
        if let Ok(v) = v.parse::<usize>() {
            if v >= 1 {
                return v.min(N_FIELDS);
            }
        }
    }
    usable.clamp(1, N_FIELDS)
}

/// 每个并发字段的线程数。默认恒为 1（这是「n_jobs = 实际 CPU」的关键：
/// 每字段 1 线程时线程几乎 100% 忙，多给线程只会让线程空转）。
/// 可用 RUST_PYFUNC_SWITCH_MOMENT_FIELD_INNER 覆盖（实验用）。
fn field_inner_threads() -> usize {
    if let Ok(v) = std::env::var("RUST_PYFUNC_SWITCH_MOMENT_FIELD_INNER") {
        if let Ok(v) = v.parse::<usize>() {
            if v >= 1 {
                return v;
            }
        }
    }
    1
}

/// 字段级线程池：**正好 conc 个**池，每个 inner 线程。
/// 只建 conc 个（不是 19 个），否则多批处理时每个池都会被用到，线程数会翻几倍。
struct FieldPools {
    pools: Vec<Arc<rayon::ThreadPool>>,
}

static FIELD_POOLS: LazyLock<Mutex<HashMap<(usize, usize), Arc<FieldPools>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

fn get_field_pools(inner: usize, conc: usize) -> Arc<FieldPools> {
    let mut cache = FIELD_POOLS.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(p) = cache.get(&(inner, conc)) {
        return p.clone();
    }
    let pools: Vec<Arc<rayon::ThreadPool>> = (0..conc)
        .map(|_| {
            Arc::new(
                rayon::ThreadPoolBuilder::new()
                    .num_threads(inner)
                    .build()
                    .expect("建字段线程池失败"),
            )
        })
        .collect();
    let fp = Arc::new(FieldPools { pools });
    cache.insert((inner, conc), fp.clone());
    fp
}

/// 算一个交易日的全市场 switch_moment 因子。
///
/// 返回 `(codes, 扁平值)`，扁平值长度 = `codes.len() * N_FACTORS`，
/// 每只股票连续 N_FACTORS 个值，顺序与 `switch_moment_names()` 一致。
///
/// 并行结构（n_jobs 既是上限、也是实际 CPU 目标，偏差 ≤10% 且绝不超）：
/// - 本进程线程预算 = RAYON_NUM_THREADS，由 factor_pipeline 按
///   `worker_cpu_budget(n_jobs, n_workers)` 下发（已经扣掉父进程 writer 池那 4 个核）。
/// - **每字段固定 1 线程**，并发 conc = min(19, 预算)：实测 1 线程/字段时线程 ≈100% 忙，
///   所以实际 CPU ≈ 预算；多给线程只会让线程更闲（2 线程/字段 ~65% 忙，6 线程 ~34% 忙，
///   因为单字段内部约 69% 的活是串行的）。
/// - 于是「n_jobs = 实际 CPU」靠**进程数**凑：进程数 ≈ (n_jobs - 父进程 4) / 19，
///   见 recommended_workers()；n_jobs 小时（如 30）单进程喂不满 19，会自动多拆进程。
/// - 每个字段跑在自己的 1 线程 rayon 小池里：faer 的 GEMM 用 Parallelism::Rayon(0)，
///   在池里就等于只用这 1 个线程，不会偷偷多开线程。
/// - 用 pool.spawn + channel 投递，不额外开阻塞线程；本进程主线程在等结果。
///
/// 实测（20241231，单进程，采样 /proc/<pid>/task 的 utime+stime 增量）：
///   每字段 1 线程 → 20 线程存活、18.8 核（≈100% 忙）
///   每字段 2 线程 → 40 线程存活、约 25 核（≈65% 忙）
///   每字段 4 线程 → 78 线程存活、38 核（≈50% 忙）
///   每字段 6 线程 → 116 线程存活、39 核（≈34% 忙）
/// 注意：单进程最多约 19~21 核（19 个字段各 1 线程），要凑到 30 核必须拆 ≥2 个进程，
/// 所以 recommended_workers() 在 n_jobs=30 时会返回 2（2 × 13 字段 = 26 核 + 父进程 4 = 30）。
pub fn compute_switch_moment_full(date: i64) -> io::Result<(Vec<String>, Vec<f32>)> {
    let meta = crate::switch_moment_level2::meta()?;
    let (t, n, raw) = crate::switch_moment_level2::load(date, &meta)?;
    let fields = derive_fields(&raw, t, n);
    drop(raw);

    let base = switch_moment_base_names();
    // 基础因子名 → 列号。注意：compute_all_factors 的产出顺序与 hm95.py 的 _gen_base_793()
    // 不完全相同（ha_time_* 与 ha_mag_* 在 Rust 侧是交错的），Python 侧本来是按名字查字典，
    // 所以这里也必须按名字定位，不能按位置。
    let base_idx: Arc<HashMap<String, usize>> = Arc::new(
        base.iter()
            .enumerate()
            .map(|(i, s)| (s.clone(), i))
            .collect(),
    );
    let base = Arc::new(base);
    let mut flat = vec![f32::NAN; n * N_FACTORS];

    // n_jobs 既是上限也是目标：每字段 1 线程把预算喂满，实际 CPU ≈ 预算。
    let budget = thread_budget();
    let conc = field_concurrency(budget);
    let inner = field_inner_threads();
    let pools = get_field_pools(inner, conc);

    let mut slots: Vec<Option<(usize, &'static str, Vec<f64>)>> = fields
        .into_iter()
        .enumerate()
        .map(|(i, (nm, v))| Some((i, nm, v)))
        .collect();

    let mut cursor = 0usize;
    while cursor < N_FIELDS {
        let end = (cursor + conc).min(N_FIELDS);
        let (tx, rx) = std::sync::mpsc::channel::<(usize, io::Result<Vec<f32>>)>();
        let mut sent = 0usize;
        for k in cursor..end {
            let item = slots[k].take().expect("字段槽位被重复使用");
            let pool = pools.pools[k - cursor].clone();
            let tx = tx.clone();
            let base_idx = base_idx.clone();
            let base = base.clone();
            pool.spawn(move || {
                let (fi, fname, vals) = item;
                let r = compute_one_field(fname, vals, t, n, &base_idx, &base);
                let _ = tx.send((fi, r));
            });
            sent += 1;
        }
        drop(tx);
        for _ in 0..sent {
            let (fi, res) = rx.recv().expect("字段任务没有回传结果");
            let values = res?;
            let fbase = fi * N_BASE;
            for bi in 0..N_BASE {
                let col = fbase + bi;
                let src = &values[bi * n..(bi + 1) * n];
                for s in 0..n {
                    flat[s * N_FACTORS + col] = src[s];
                }
            }
        }
        cursor = end;
    }

    Ok((meta.codes.clone(), flat))
}

// ============================================================
// Python 接口
// ============================================================

#[pyfunction]
pub fn py_switch_moment_names() -> Vec<String> {
    switch_moment_names()
}

/// 单日单进程直算入口（核对用；正式计算走 run_factor_pipeline_cross_section）。
#[pyfunction]
pub fn py_switch_moment(py: Python<'_>, date: i64) -> PyResult<(Vec<String>, Vec<f32>)> {
    py.allow_threads(|| compute_switch_moment_full(date))
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn name_count_and_order() {
        assert_eq!(switch_moment_base_names().len(), N_BASE);
        assert_eq!(switch_moment_names().len(), N_FACTORS);
        assert_eq!(N_FACTORS, 15_067);
    }

    /// 手写的基础名必须与 compute_all_factors 实际产出的名字集合一致
    /// （顺序不同：Rust 侧 ha_time_* 与 ha_mag_* 是交错的，所以按名字定位而不是按位置）。
    #[test]
    fn base_names_match_producer() {
        let dummy = Array2::from_elem((2, 1), 0.0f64);
        let produced: Vec<String> = compute_all_factors(dummy, TOP_K, false)
            .into_iter()
            .map(|(n, _)| n)
            .collect();
        assert_eq!(produced.len(), N_BASE);
        let mut a = produced.clone();
        let mut b = switch_moment_base_names();
        a.sort();
        b.sort();
        assert_eq!(a, b);
    }
}
