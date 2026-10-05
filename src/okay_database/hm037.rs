//! hm37 订单体量与盈利 —— 旧框架（design_whatever_out.TickDataBase）的 Rust 复现。
//!
//! 旧代码要点（逐条对齐，见 成果hm37_订单体量与盈利/hm37_okay.py）：
//!
//!   参数：get_many=0 / read_full=0 / change_time=1 / cross_section=False，
//!        没有 adjust_afternoon，窗口用真实钟点，所以 Rust 侧 with_afternoon_adjust=false。
//!
//!   撤单过滤：read_trade_data_single(with_retreat=0) 丢 flag==32，按数值比较
//!        （2025 年文件里撤单标记写的是 "32.0"，一样过滤得掉）。
//!        注意：本文件**不用** read_trade_fast_inner，而是自己按 f64 解析 CSV，
//!        因为它的 price/volume/turnover 是 f32，会把 close/vwap-1 这种 1e-4 量级的
//!        量放大成 1e-4 的相对偏差（实测对账不过）。列索引与 fast_csv_reader 一致。
//!
//!   五个时段（between_time 两端都含，精确到纳秒；原始 exchtime 是整数微秒，
//!   所以按当日微秒比较与纳秒等价）：
//!        allday    = 整天**不做任何时间过滤**（含 09:15~09:30 集合竞价与尾盘）
//!        open30    = 09:30:00 ~ 10:00:00
//!        morning   = 09:30:00 ~ 11:30:00
//!        afternoon = 13:00:00 ~ 14:57:00
//!        close30   = 14:30:00 ~ 14:57:00
//!
//!   每个时段 sing(rows) 做同样 20 个指标：
//!        close_price = 该时段最后一笔的成交价
//!        asks1 = rows.groupby('ask_order').agg(turnover 求和, volume 求和)
//!                price = 总 turnover / 总 volume，再 p = close_price / price - 1
//!        bids1 同理（bid_order）
//!        然后 one() / two() / three() 各拿这份聚合表算相关系数（见下）。
//!
//!   数量：20 个指标 × 5 个时段 = 100，再拼一份逐项取绝对值 = 200（names 的顺序就是
//!        先 100 个 `_notabs`，再 100 个 `_abs`；段内是「时段优先、指标其次」）。
//!
//!   三个子函数（都是 Pearson，成对完整观测；样本 < 2 或某侧方差为 0 给 NaN）：
//!     one：group 数 >0 才算；volume 取自然对数；不筛体积。
//!          [asks_corr, bids_corr, asks_corr_large(>=q0.7), bids..., asks_corr_small(<q0.3),
//!           bids..., asks_corr_middle(0.3<=v<0.7), bids...]
//!     two：先按**原始** volume > 100 筛，再取对数。
//!          [asks_corr, bids_corr, asks_corr_large(>=q0.9), bids..., asks_corr_small(<q0.9),
//!           bids...]
//!     three：price_before = price / (5 个交易日前的开盘价) - 1，再筛 volume>100、取对数。
//!          [asks_corr_before, bids_corr_before, asks_corr_large_before(>=q0.9), bids...,
//!           asks_corr_small_before(<q0.9), bids...]
//!          整段包在 try 里，任何一步出错就整组 6 个 NaN；开盘价取不到时 price_before 全 NaN，
//!          相关也是 NaN，两条路径结果一样，所以 Rust 侧统一给 NaN。
//!
//!   分位点用 pandas 的线性插值（位置 = (n-1)*q）。
//!
//!   两个容易漏的旧行为（照抄，不要"优化"）：
//!     1. 任一时段为空 → sing 里 df.price.iloc[-1] 抛 IndexError → 整个股票日返回 None。
//!        所以五个时段必须都有成交，否则整条 NaN。
//!     2. 开盘价取的是「当前交易日往前数 5 个交易日」那一日的 S_DQ_OPEN，
//!        来自日频平台 {vars_root}/DailyPrice/S_DQ_OPEN.h5，已核对与旧框架读到的一致。
//!        （旧代码用 dwo.last_n_trading_date(date,5)，日历我用同一份 calendar_map.csv。）
//!
//!   names_single 200 个，无重名，不需要 _dup 后缀。

use std::collections::BTreeMap;
use std::fs::File;
use std::io::Read;
use std::sync::{Arc, LazyLock, Mutex};

const DAY_US: i64 = 86_400_000_000;
const SEC_US: i64 = 1_000_000;

// ── 逐笔 CSV 自己按 f64 解析 ───────────────────────────────────────────────
//
// 不用 read_trade_fast_inner：它返回的 price/volume/turnover 是 f32。
// 本系列的指标是 corr(close/vwap - 1, log 成交量)，close/vwap - 1 本身常常只有
// 1e-4 量级，f32 在 turnover 上的 ~1e-7 相对误差会被放大成 1e-4 量级的相对偏差，
// 对账直接不过（实测 #31/#71/#84 这些位置）。列索引与 fast_csv_reader 一致。
const TRD_COL_EXCHTIME: usize = 4;
const TRD_COL_PRICE: usize = 7;
const TRD_COL_VOLUME: usize = 8;
const TRD_COL_TURNOVER: usize = 9;
const TRD_COL_FLAG: usize = 10;
const TRD_COL_ASK_ORDER: usize = 13;
const TRD_COL_BID_ORDER: usize = 14;
const TRD_NCOL: usize = 15;

/// 一条逐笔成交（f64，只留计算要用的列）
#[derive(Clone, Copy)]
struct TradeRow {
    /// epoch 微秒，含 8 小时偏移（与 pandas 的 exchtime 同义）
    time_us: i64,
    price: f64,
    volume: f64,
    turnover: f64,
    flag: f64,
    bid_order: i64,
    ask_order: i64,
}

fn parse_f64_at(b: &[u8]) -> f64 {
    if b.is_empty() {
        return f64::NAN;
    }
    std::str::from_utf8(b)
        .ok()
        .and_then(|s| s.trim().parse::<f64>().ok())
        .unwrap_or(f64::NAN)
}

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

/// 读 transaction CSV（f64），保持文件原有行序，只跳表头。
/// 不过滤 exchtime==0 的行：pandas 读进来会把它当成 08:00，allday 时段是包含它的。
fn read_trade_f64(code: &str, date: i64) -> Option<Vec<TradeRow>> {
    let filename = format!("{}_{}_transaction.csv", code, date);
    let mut path = None;
    for root in crate::data_paths::level2_roots() {
        let p = root.join(date.to_string()).join("transaction").join(&filename);
        if p.exists() {
            path = Some(p);
            break;
        }
    }
    let path = path?;

    let mut content = String::new();
    File::open(&path).ok()?.read_to_string(&mut content).ok()?;

    let mut out: Vec<TradeRow> = Vec::with_capacity(content.len() / 120 + 1);
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
        out.push(TradeRow {
            time_us: parse_i64_at(f[TRD_COL_EXCHTIME]) + 8 * 3600 * SEC_US,
            price: parse_f64_at(f[TRD_COL_PRICE]),
            volume: parse_f64_at(f[TRD_COL_VOLUME]),
            turnover: parse_f64_at(f[TRD_COL_TURNOVER]),
            flag: parse_f64_at(f[TRD_COL_FLAG]),
            bid_order: parse_i64_at(f[TRD_COL_BID_ORDER]),
            ask_order: parse_i64_at(f[TRD_COL_ASK_ORDER]),
        });
    }
    Some(out)
}

/// 五个时段：(名字, 当日微秒下界, 当日微秒上界)，闭区间。
/// allday 用 [0, DAY_US] 表示不过滤。
const WINDOWS: [(&str, i64, i64); 5] = [
    ("allday", 0, DAY_US),
    ("open30", 34_200_000_000, 36_000_000_000),     // 09:30:00 ~ 10:00:00
    ("morning", 34_200_000_000, 41_400_000_000),    // 09:30:00 ~ 11:30:00
    ("afternoon", 46_800_000_000, 53_820_000_000),  // 13:00:00 ~ 14:57:00
    ("close30", 52_200_000_000, 53_820_000_000),    // 14:30:00 ~ 14:57:00
];

/// 每时段 20 个指标名（顺序 = 旧 names 的前 20 项）。
const BASE: [&str; 20] = [
    "asks_corr_one",
    "bids_corr_one",
    "asks_corr_large_one",
    "bids_corr_large_one",
    "asks_corr_small_one",
    "bids_corr_small_one",
    "asks_corr_middle_one",
    "bids_corr_middle_one",
    "asks_corr_two",
    "bids_corr_two",
    "asks_corr_large_two",
    "bids_corr_large_two",
    "asks_corr_small_two",
    "bids_corr_small_two",
    "asks_corr_before_three",
    "bids_corr_before_three",
    "asks_corr_large_before_three",
    "bids_corr_large_before_three",
    "asks_corr_small_before_three",
    "bids_corr_small_before_three",
];

const PER_WINDOW: usize = 20;
/// 5 个时段 × 20 个指标 = 100，再拼一份 abs 版 = 200。
pub const N_OUT: usize = PER_WINDOW * 5 * 2;

pub fn names() -> Vec<String> {
    let mut out = Vec::with_capacity(N_OUT);
    for w in WINDOWS.iter() {
        for b in BASE.iter() {
            out.push(format!("{}_{}_notabs", b, w.0));
        }
    }
    for w in WINDOWS.iter() {
        for b in BASE.iter() {
            out.push(format!("{}_{}_abs", b, w.0));
        }
    }
    debug_assert_eq!(out.len(), N_OUT);
    out
}

// ============================================================================
// pandas 语义小工具
// ============================================================================

/// 成对完整观测的 Pearson（只丢 NaN，与 pandas notna 一致）。
/// 有效样本 < 2、或任一侧方差 <= 0 时返回 NaN（等价 np.corrcoef 的 0/0）。
///
/// 额外挡一种情况：某一侧在有效样本里**逐位全等**时直接给 NaN。
/// 这是 numpy 的零方差行为（它算到的方差正好是 0），而我们自己顺序累加平方和
/// 会留下 ~1e-16 的舍入残差，不挡就会算出一个 1e-15 量级的假相关
/// （实测 000001@20231226 的 #24、000002@20231228 的 #25 就是这样）。
fn pearson_na(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len().min(y.len());
    let mut cx: Vec<f64> = Vec::with_capacity(n);
    let mut cy: Vec<f64> = Vec::with_capacity(n);
    for i in 0..n {
        if !x[i].is_nan() && !y[i].is_nan() {
            cx.push(x[i]);
            cy.push(y[i]);
        }
    }
    let m = cx.len();
    if m < 2 {
        return f64::NAN;
    }
    let x_const = cx.iter().all(|v| *v == cx[0]);
    let y_const = cy.iter().all(|v| *v == cy[0]);
    if x_const || y_const {
        return f64::NAN;
    }
    let mf = m as f64;
    let mut sx = 0.0f64;
    let mut sy = 0.0f64;
    for i in 0..m {
        sx += cx[i];
        sy += cy[i];
    }
    let mx = sx / mf;
    let my = sy / mf;
    let mut sxx = 0.0f64;
    let mut syy = 0.0f64;
    let mut sxy = 0.0f64;
    for i in 0..m {
        let dx = cx[i] - mx;
        let dy = cy[i] - my;
        sxx += dx * dx;
        syy += dy * dy;
        sxy += dx * dy;
    }
    if sxx <= 0.0 || syy <= 0.0 {
        return f64::NAN;
    }
    sxy / (sxx.sqrt() * syy.sqrt())
}

/// 只保留 mask 为真的位置再算相关（对应 pandas 的布尔切片）。
fn pearson_masked(x: &[f64], y: &[f64], keep: &dyn Fn(usize) -> bool) -> f64 {
    let n = x.len().min(y.len());
    let mut cx: Vec<f64> = Vec::new();
    let mut cy: Vec<f64> = Vec::new();
    for i in 0..n {
        if keep(i) {
            cx.push(x[i]);
            cy.push(y[i]);
        }
    }
    pearson_na(&cx, &cy)
}

/// pandas Series.quantile(q)：线性插值，位置 = (n-1)*q。空序列给 NaN。
fn quantile_linear(sorted: &[f64], q: f64) -> f64 {
    let n = sorted.len();
    if n == 0 {
        return f64::NAN;
    }
    if n == 1 {
        return sorted[0];
    }
    let pos = (n - 1) as f64 * q;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    if lo == hi {
        return sorted[lo.min(n - 1)];
    }
    let frac = pos - lo as f64;
    sorted[lo] + (sorted[hi] - sorted[lo]) * frac
}

fn sorted_vals(v: &[f64]) -> Vec<f64> {
    let mut s: Vec<f64> = v.iter().cloned().filter(|x| !x.is_nan()).collect();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    s
}

// ============================================================================
// 日频开盘价（{vars_root}/DailyPrice/S_DQ_OPEN.h5）
// ============================================================================

struct DailyMeta {
    /// calendar_map.csv 里每个交易日的行号
    date_to_idx: BTreeMap<i64, usize>,
    /// symbol_map.csv 里的列号
    code_to_pos: BTreeMap<String, usize>,
    n_cols: usize,
}

static DAILY_META: LazyLock<Option<DailyMeta>> = LazyLock::new(load_daily_meta);

fn load_daily_meta() -> Option<DailyMeta> {
    let dir = crate::data_paths::vars_path("DailyPrice");
    let cal = std::fs::read_to_string(dir.join("calendar_map.csv")).ok()?;
    let mut date_to_idx = BTreeMap::new();
    let mut row = 0usize;
    for line in cal.lines() {
        let s = line.trim();
        if s.is_empty() || !s.chars().all(|c| c.is_ascii_digit()) {
            continue; // 跳过表头 date_min
        }
        if let Ok(d) = s.parse::<i64>() {
            date_to_idx.insert(d, row);
            row += 1;
        }
    }
    if date_to_idx.is_empty() {
        return None;
    }
    let sym = std::fs::read_to_string(dir.join("symbol_map.csv")).ok()?;
    let mut code_to_pos = BTreeMap::new();
    let mut n_cols = 0usize;
    for line in sym.lines() {
        let s = line.trim();
        let mut it = s.split(',');
        let (Some(a), Some(b)) = (it.next(), it.next()) else {
            continue;
        };
        let Ok(pos) = b.trim().parse::<usize>() else {
            continue;
        };
        n_cols = n_cols.max(pos + 1);
        code_to_pos.insert(a.trim().to_string(), pos);
    }
    Some(DailyMeta {
        date_to_idx,
        code_to_pos,
        n_cols,
    })
}

/// date 往前 n 个交易日的日期；不够则 None。
fn n_days_before(meta: &DailyMeta, date: i64, n: usize) -> Option<i64> {
    let idx = *meta.date_to_idx.get(&date)?;
    if idx < n {
        return None;
    }
    let target = idx - n;
    meta.date_to_idx
        .iter()
        .find(|(_, &i)| i == target)
        .map(|(&d, _)| d)
}

/// 读某一行（某一天）的整行开盘价并缓存最近 4 天；返回 None 表示读不到。
#[cfg(feature = "hdf5")]
fn daily_open_row(day_idx: usize, n_cols: usize) -> Option<Arc<Vec<f64>>> {
    use hdf5_metno as hdf5;
    use ndarray::s;

    static ROW_CACHE: LazyLock<Mutex<Vec<(usize, Arc<Vec<f64>>)>>> =
        LazyLock::new(|| Mutex::new(Vec::new()));
    {
        let cache = ROW_CACHE.lock().unwrap_or_else(|e| e.into_inner());
        if let Some((_, row)) = cache.iter().find(|(i, _)| *i == day_idx) {
            return Some(row.clone());
        }
    }
    let path = crate::data_paths::vars_path("DailyPrice/S_DQ_OPEN.h5");
    let file = hdf5::File::open(&path).ok()?;
    let ds = file.dataset("data").ok()?;
    let shape = ds.shape();
    if day_idx >= shape[0] {
        return None;
    }
    let ncol = shape[1].min(n_cols);
    let arr: ndarray::Array2<f64> = ds.read_slice_2d(s![day_idx..day_idx + 1, 0..ncol]).ok()?;
    let mut row = vec![f64::NAN; n_cols];
    for c in 0..ncol {
        row[c] = arr[(0, c)];
    }
    let row = Arc::new(row);
    let mut cache = ROW_CACHE.lock().unwrap_or_else(|e| e.into_inner());
    if cache.len() >= 4 {
        cache.remove(0);
    }
    cache.push((day_idx, row.clone()));
    Some(row)
}

#[cfg(not(feature = "hdf5"))]
fn daily_open_row(_day_idx: usize, _n_cols: usize) -> Option<Arc<Vec<f64>>> {
    None
}

/// 5 个交易日前的开盘价；取不到返回 NaN。
fn open_price_before(date: i64, code: &str) -> f64 {
    let Some(meta) = DAILY_META.as_ref() else {
        return f64::NAN;
    };
    let Some(d5) = n_days_before(meta, date, 5) else {
        return f64::NAN;
    };
    let Some(&day_idx) = meta.date_to_idx.get(&d5) else {
        return f64::NAN;
    };
    let Some(&pos) = meta.code_to_pos.get(code) else {
        return f64::NAN;
    };
    match daily_open_row(day_idx, meta.n_cols) {
        Some(row) => row.get(pos).copied().unwrap_or(f64::NAN),
        None => f64::NAN,
    }
}

// ============================================================================
// 一个时段的聚合与指标
// ============================================================================

/// group by 订单号后的 (price, volume)：price = 总 turnover / 总 volume。
type Agg = (f64, f64);

fn group_agg(rows: &[&TradeRow], close_price: f64, side_ask: bool) -> Vec<Agg> {
    let mut m: BTreeMap<i64, (f64, f64)> = BTreeMap::new();
    for r in rows.iter() {
        let key = if side_ask { r.ask_order } else { r.bid_order };
        let e = m.entry(key).or_insert((0.0, 0.0));
        e.0 += r.turnover as f64;
        e.1 += r.volume as f64;
    }
    m.values()
        .map(|&(tn, vol)| {
            let vwap = tn / vol;
            (close_price / vwap - 1.0, vol)
        })
        .collect()
}

/// one()：不筛体积，直接用对数成交量；0.3 / 0.7 分位切三档。
fn metrics_one(asks: &[Agg], bids: &[Agg]) -> [f64; 8] {
    if asks.is_empty() || bids.is_empty() {
        return [f64::NAN; 8];
    }
    let a = one_side(asks);
    let b = one_side(bids);
    [a[0], b[0], a[1], b[1], a[2], b[2], a[3], b[3]]
}

fn one_side(agg: &[Agg]) -> [f64; 4] {
    let p: Vec<f64> = agg.iter().map(|x| x.0).collect();
    let lv: Vec<f64> = agg.iter().map(|x| x.1.ln()).collect();
    let sv = sorted_vals(&lv);
    let q70 = quantile_linear(&sv, 0.7);
    let q30 = quantile_linear(&sv, 0.3);
    let all = pearson_na(&p, &lv);
    let large = pearson_masked(&p, &lv, &|i| lv[i] >= q70);
    let small = pearson_masked(&p, &lv, &|i| lv[i] < q30);
    let middle = pearson_masked(&p, &lv, &|i| lv[i] >= q30 && lv[i] < q70);
    [all, large, small, middle]
}

/// two()：先按原始 volume > 100 筛，再取对数；0.9 分位切两档。
fn metrics_two(asks: &[Agg], bids: &[Agg]) -> [f64; 6] {
    if asks.is_empty() || bids.is_empty() {
        return [f64::NAN; 6];
    }
    let a = two_side(asks);
    let b = two_side(bids);
    [a[0], b[0], a[1], b[1], a[2], b[2]]
}

fn two_side(agg: &[Agg]) -> [f64; 3] {
    let sel: Vec<&Agg> = agg.iter().filter(|x| x.1 > 100.0).collect();
    let p: Vec<f64> = sel.iter().map(|x| x.0).collect();
    let lv: Vec<f64> = sel.iter().map(|x| x.1.ln()).collect();
    let sv = sorted_vals(&lv);
    let q90 = quantile_linear(&sv, 0.9);
    let all = pearson_na(&p, &lv);
    let large = pearson_masked(&p, &lv, &|i| lv[i] >= q90);
    let small = pearson_masked(&p, &lv, &|i| lv[i] < q90);
    [all, large, small]
}

/// three()：price_before = price / 开盘价 - 1，再按 volume>100 筛、取对数、0.9 分位切两档。
fn metrics_three(asks: &[Agg], bids: &[Agg], open_ref: f64) -> [f64; 6] {
    if asks.is_empty() || bids.is_empty() {
        return [f64::NAN; 6];
    }
    let a = three_side(asks, open_ref);
    let b = three_side(bids, open_ref);
    [a[0], b[0], a[1], b[1], a[2], b[2]]
}

fn three_side(agg: &[Agg], open_ref: f64) -> [f64; 3] {
    let sel: Vec<(f64, f64)> = agg
        .iter()
        .filter(|x| x.1 > 100.0)
        .map(|x| (x.0 / open_ref - 1.0, x.1.ln()))
        .collect();
    let p: Vec<f64> = sel.iter().map(|x| x.0).collect();
    let lv: Vec<f64> = sel.iter().map(|x| x.1).collect();
    let sv = sorted_vals(&lv);
    let q90 = quantile_linear(&sv, 0.9);
    let all = pearson_na(&p, &lv);
    let large = pearson_masked(&p, &lv, &|i| lv[i] >= q90);
    let small = pearson_masked(&p, &lv, &|i| lv[i] < q90);
    [all, large, small]
}

/// 一个时段的 20 个指标。
fn sing(rows: &[&TradeRow], open_ref: f64) -> [f64; PER_WINDOW] {
    let close_price = rows[rows.len() - 1].price as f64;
    let asks = group_agg(rows, close_price, true);
    let bids = group_agg(rows, close_price, false);
    let one = metrics_one(&asks, &bids);
    let two = metrics_two(&asks, &bids);
    let three = metrics_three(&asks, &bids, open_ref);
    let mut out = [f64::NAN; PER_WINDOW];
    out[..8].copy_from_slice(&one);
    out[8..14].copy_from_slice(&two);
    out[14..20].copy_from_slice(&three);
    out
}

// ============================================================================
// 入口
// ============================================================================

/// 逐股入口：算不出来返回 None（上游补 NaN，保证因子个数恒定）。
pub fn compute_stock(code: &str, date: i64) -> Option<Vec<f32>> {
    let trades = read_trade_f64(code, date)?;
    if trades.is_empty() {
        return None;
    }
    // 窗口只跟当日微秒有关，先算好每行的 day_us
    let day_us: Vec<i64> = trades
        .iter()
        .map(|r| r.time_us.rem_euclid(DAY_US))
        .collect();

    // 旧代码任一窗口为空就整体返回 None，这里照做：先确认五个窗口都非空。
    let mut window_rows: Vec<Vec<&TradeRow>> = Vec::with_capacity(5);
    for &(_, lo, hi) in WINDOWS.iter() {
        let rows: Vec<&TradeRow> = trades
            .iter()
            .enumerate()
            .filter(|(i, r)| r.flag != 32.0 && day_us[*i] >= lo && day_us[*i] <= hi)
            .map(|(_, r)| r)
            .collect();
        if rows.is_empty() {
            return None;
        }
        window_rows.push(rows);
    }

    let open_ref = open_price_before(date, code);

    let mut base: Vec<f64> = Vec::with_capacity(PER_WINDOW * 5);
    for rows in window_rows.iter() {
        base.extend(sing(rows, open_ref).iter().copied());
    }    let mut out: Vec<f32> = Vec::with_capacity(N_OUT);
    for v in base.iter() {
        out.push(*v as f32);
    }
    for v in base.iter() {
        out.push(v.abs() as f32);
    }
    debug_assert_eq!(out.len(), N_OUT);
    Some(out)
}

/// hm37 全是逐股因子，没有横截面版本。
pub fn compute_cross(_date: i64) -> Option<(Vec<String>, Vec<f32>)> {
    None
}
