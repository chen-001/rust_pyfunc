//! hm031「价格二叉树」—— 旧框架（design_whatever_out.TickDataBase）的 Rust 复现。
//!
//! 旧文件：/home/chenzongwei/pythoncode/成果hm31_价格二叉树/hm31_okay.py
//!
//! 旧代码要点（逐条对齐）：
//!
//! 【调用参数】get_many=0、read_full=0、change_time=1 → read_trade_data_single 返回单个 DataFrame，
//!   列含 exchtime（北京时间 Timestamp）、price、volume、flag；with_retreat=0 先丢掉 flag==32。
//!
//! 【窗口】四个 between_time 窗口 + 全天，都是真实钟点、闭区间、精确到微秒，不做下午平移：
//!   allday    = 不筛（连 09:25 集合竞价那些也算）
//!   open30    = [09:30:00, 10:00:00]
//!   morning   = [09:30:00, 11:30:00]
//!   afternoon = [13:00:00, 14:57:00]
//!   close30   = [14:30:00, 14:57:00]
//!
//! 【每个窗口】one(df)：
//!   df.shape[0] >= 5 才建树，否则返回 43 个 NaN。
//!   tree = rp.PriceTree(); tree.build_tree(df.exchtime.to_numpy(int), df.price(float), df.volume(float))
//!   取 tree.get_all_features() 里 43 项（顺序见下图），其中不少是两项/三项相除。
//!   ！！exchtime 传进去的是**纳秒**（Timestamp 的 int64 原值），Rust 侧对应 time_us * 1000。
//!   建树按数组顺序逐笔插入、不排序，所以行的先后不能动（= 文件行序）。
//!   非有限值（inf/nan）在旧代码里被 .replace(np.inf, np.nan) 抹成 NaN，这里出口统一转 NaN。
//!
//! 43 个特征 × 5 个窗口 = 215 个输出，顺序与旧代码 names_single 完全一致。
//!
//! ════════════════════════════════════════════════════════════════════
//! 记一处刻意偏差：_fold 因子的算法（已向 lead 报备并获准）
//! ════════════════════════════════════════════════════════════════════
//! 本文件只算 215 个原值；base 层那 4 个 _fold 因子在
//! 成果hm31_价格二叉树/hm31_okay_database.py 里按旧定义 abs(log(值) − 当日 mean(log(值))) 补算，
//! 没走框架自带的 `_fold`（框架是 abs(值 − 截面均值)，少一层 log）。
//!
//! 其中 completeness 这一项**刻意不与旧表达式等价**：
//! 旧写法 dwo.get_abs(np.log(fac)) 遇到 fac == 0 会先得 log(0) = -inf，
//! pandas 的截面 mean 被 -inf 污染成 -inf，当天整列 fold 全变成 inf。
//! 而 completeness 经常正好是 0——tree 的 depth >= 63 时 get_completeness 走
//! node_count / f64::MAX 那条分支，落到 f32 下溢成 0，
//! 实测 20250303 全市场 5117 只里 4840 只的 completeness_allday_unfold 正好为 0，
//! 照抄旧写法就是一个整列 inf 的废因子。
//! 本实现的口径：先把非有限的 log 值抹成 NaN，截面均值只在有限值上求。
//! 旧值本来有限的地方逐个对上，只在旧值退化的那些股票上不一样。
//! 另外 3 个 fold（wpl / avg_width / asl_ratio_avg_depth）的 unfold 值不会为 0，
//! 两种写法结果完全相同、不受这条偏差影响。
//!
//! ════════════════════════════════════════════════════════════════════
//! 说明：树本身的实现直接复用 crate::tree::PriceTree
//! ════════════════════════════════════════════════════════════════════
//! 旧代码调的就是 rust_pyfunc 里的 PriceTree，为了逐值一致这里不重写算法，直接调同一个实现。
//! 但 worker 进程里没有 Python 解释器，get_all_features 要 GIL，所以在 src/tree/mod.rs 里
//! 给 PriceTree 加了两个 pub(crate) 入口：build_tree_views（纯 Rust 建树，build_tree 现在
//! 只是它的薄包装）和 features_43（那 43 项按同一顺序算好）。原有行为一个字没改。

use crate::data_paths;
use crate::tree::PriceTree;
use std::fs::File;
use std::io::Read;

/// 逐笔 CSV 的列索引（0 基，与 fast_csv_reader 的常量一致）
const TRD_COL_EXCHTIME: usize = 4;
const TRD_COL_PRICE: usize = 7;
const TRD_COL_VOLUME: usize = 8;
const TRD_COL_FLAG: usize = 10;
const TRD_NCOL: usize = 15;

/// 秒 → 微秒
const US: i64 = 1_000_000;

/// 五个时段的真实钟点闭区间（当天微秒数）；allday 用整段
const WINDOW_NAMES: [&str; 5] = ["allday", "open30", "morning", "afternoon", "close30"];
const WINDOWS: [(i64, i64); 5] = [
    (i64::MIN, i64::MAX),
    (34200 * US, 36000 * US),
    (34200 * US, 41400 * US),
    (46800 * US, 53820 * US),
    (52200 * US, 53820 * US),
];

/// 每个窗口 43 个特征，五个窗口共 215 个
pub const N_FEAT: usize = 43;
pub const N_OUT: usize = N_FEAT * 5;

/// 建树的最少笔数（旧代码 `if df.shape[0]>=5`）
const MIN_ROWS: usize = 5;

/// 43 个特征名，顺序与 src/tree/mod.rs 的 PriceTree::features_43 严格一致
const FEATURE_NAMES: [&str; N_FEAT] = [
    "max_path_length",
    "avg_path_length",
    "avg_path_length_ratio",
    "max_subtree_nodes",
    "avg_subtree_nodes",
    "avg_subtree_nodes_ratio",
    "max_balance_factor",
    "avg_balance_factor",
    "avg_balance_factor_ratio",
    "skewness",
    "avg_volume_per_node",
    "avg_volume_per_node_ratio",
    "max_width",
    "avg_width",
    "avg_width_ratio",
    "critical_nodes",
    "critical_nodes_ratio_avg_width",
    "critical_nodes_ratio_avg_depth",
    "critical_nodes_ratio_avg_width_avg_depth",
    "critical_nodes_ratio_avg_path_length",
    "critical_nodes_ratio_total_nodes",
    "completeness",
    "density",
    "total_nodes",
    "internal_nodes",
    "internal_nodes_ratio",
    "avg_depth",
    "avg_depth_ratio",
    "asl",
    "asl_ratio_avg_depth",
    "asl_ratio_avg_width",
    "asl_ratio_avg_depth_avg_width",
    "wpl",
    "wpl_ratio_avg_depth",
    "wpl_ratio_avg_width",
    "wpl_ratio_avg_depth_avg_width",
    "wpl_ratio_avg_path_length",
    "diameter",
    "diameter_ratio_avg_depth",
    "diameter_ratio_avg_width",
    "degree_two_nodes",
    "degree_two_nodes_ratio",
    "degree_ratio",
];

pub fn names() -> Vec<String> {
    let mut out = Vec::with_capacity(N_OUT);
    for w in WINDOW_NAMES.iter() {
        for f in FEATURE_NAMES.iter() {
            out.push(format!("{}_{}_unfold", f, w));
        }
    }
    out
}

#[inline]
fn day_us(time_us: i64) -> i64 {
    time_us.rem_euclid(86_400_000_000)
}

#[inline]
fn parse_i64_at(b: &[u8]) -> i64 {
    let mut i = 0usize;
    while i < b.len() && b[i] == b' ' {
        i += 1;
    }
    let mut neg = false;
    if i < b.len() && (b[i] == b'-' || b[i] == b'+') {
        neg = b[i] == b'-';
        i += 1;
    }
    let mut v: i64 = 0;
    while i < b.len() {
        let c = b[i];
        if !(b'0'..=b'9').contains(&c) {
            break;
        }
        v = v * 10 + (c - b'0') as i64;
        i += 1;
    }
    if neg {
        -v
    } else {
        v
    }
}

/// 空单元格按 pandas 的读法给 NaN；其余交给 std 的正确舍入解析
#[inline]
fn parse_f64_at(b: &[u8]) -> f64 {
    if b.is_empty() {
        return f64::NAN;
    }
    std::str::from_utf8(b)
        .ok()
        .and_then(|s| s.trim().parse::<f64>().ok())
        .unwrap_or(f64::NAN)
}

/// 读 transaction CSV：返回 (当日微秒数, price, volume, flag)，保持文件行序。
fn read_trade_f64(code: &str, date: i64) -> Option<Vec<(i64, f64, f64, i32)>> {
    let filename = format!("{}_{}_transaction.csv", code, date);
    let mut path = None;
    for root in data_paths::level2_roots() {
        let p = root.join(date.to_string()).join("transaction").join(&filename);
        if p.exists() {
            path = Some(p);
            break;
        }
    }
    let path = path?;

    let mut content = String::new();
    File::open(&path).ok()?.read_to_string(&mut content).ok()?;

    let mut out = Vec::with_capacity(content.len() / 120 + 1);
    let mut lines = content.lines();
    lines.next(); // 表头
    for line in lines {
        if line.is_empty() {
            continue;
        }
        let bytes = line.as_bytes();
        let mut f: [&[u8]; TRD_NCOL] = [&[][..]; TRD_NCOL];
        let mut start = 0usize;
        let mut col = 0usize;
        for (i, &b) in bytes.iter().enumerate() {
            if b == b',' {
                if col < TRD_NCOL {
                    f[col] = &bytes[start..i];
                }
                col += 1;
                start = i + 1;
            }
        }
        if col < TRD_NCOL {
            f[col] = &bytes[start..];
        }
        let t_us = day_us(parse_i64_at(f[TRD_COL_EXCHTIME]) + 8 * 3600 * US);
        out.push((
            t_us,
            parse_f64_at(f[TRD_COL_PRICE]),
            parse_f64_at(f[TRD_COL_VOLUME]),
            parse_i64_at(f[TRD_COL_FLAG]) as i32,
        ));
    }
    Some(out)
}

/// 一个窗口的 43 个特征；笔数不足 5 或建树失败时给 43 个 NaN。
fn one_window(rows: &[&(i64, f64, f64, i32)]) -> [f64; N_FEAT] {
    if rows.len() < MIN_ROWS {
        return [f64::NAN; N_FEAT];
    }
    // 旧代码传的是 df.exchtime.to_numpy(dtype=int)，即 Timestamp 的 int64 纳秒
    let times: Vec<i64> = rows.iter().map(|r| r.0 * 1000).collect();
    let prices: Vec<f64> = rows.iter().map(|r| r.1).collect();
    let volumes: Vec<f64> = rows.iter().map(|r| r.2).collect();

    let mut tree = PriceTree::new_tree();
    tree.build_tree_views(
        ndarray::ArrayView1::from(&times),
        ndarray::ArrayView1::from(&prices),
        ndarray::ArrayView1::from(&volumes),
    );
    let feats = tree.features_43();
    // 旧代码在 cal_factor_one_day 里做过 .replace(np.inf, np.nan)，这里出口统一抹平
    let mut out = [f64::NAN; N_FEAT];
    for (i, &v) in feats.iter().enumerate() {
        out[i] = if v.is_finite() { v } else { f64::NAN };
    }
    out
}

/// 逐股入口：算不出来返回 None（上游会补 NaN，保证因子个数恒定）。
pub fn compute_stock(code: &str, date: i64) -> Option<Vec<f32>> {
    let trades = read_trade_f64(code, date)?;
    // with_retreat=0
    let rows: Vec<(i64, f64, f64, i32)> = trades.into_iter().filter(|r| r.3 != 32).collect();
    if rows.is_empty() {
        return None;
    }

    let mut out = Vec::with_capacity(N_OUT);
    for &(lo, hi) in WINDOWS.iter() {
        let sub: Vec<&(i64, f64, f64, i32)> = rows
            .iter()
            .filter(|r| r.0 >= lo && r.0 <= hi)
            .collect();
        out.extend(one_window(&sub).iter().map(|&v| v as f32));
    }
    debug_assert_eq!(out.len(), N_OUT);
    Some(out)
}

/// hm31 全是逐股因子，没有横截面版本。
pub fn compute_cross(_date: i64) -> Option<(Vec<String>, Vec<f32>)> {
    None
}
