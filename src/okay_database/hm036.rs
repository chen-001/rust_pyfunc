//! hm36 尾盘撤单 —— 旧框架（design_whatever_out.TickDataBase）的 Rust 复现。
//!
//! 旧代码要点（逐条对齐 /home/chenzongwei/pythoncode/成果hm36_尾盘撤单/hm36_okay.py 的 run_base）：
//!
//!   get_many=0、read_full=0、change_time=1 → read_trade_data_single 返回**单张** DataFrame。
//!   change_time=1 只把 exchtime/localtime 转成 Timestamp 并加一列 delay，不筛行；
//!   旧代码**没有**显式调 adjust_afternoon，窗口用的是真实钟点 14:54:00~14:57:00，
//!   所以 Rust 侧 with_afternoon_adjust=false。只做 deal_data 不做下午平移。
//!
//!   with_retreat=1 → 保留 flag==32 的撤单行（本系列的主题就是撤单，必须保留）。
//!
//!   计算流程（顺序不能变）：
//!     1. df = df.set_index('exchtime').between_time('14:54:00','14:57:00')
//!        两端都含，微秒精度（所以这里用闭区间，不能按整秒截断）
//!     2. df = df[df.flag == 32]           只留撤单行
//!     3. dura = (exchtime - exchtime.min()).dt.total_seconds()
//!        dura_volume = volume * dura
//!        ！基准 min 是**撤单子集**里最早那一笔，不是全窗口最早那一笔
//!     4. asks = df[df.ask_order != 0].groupby('ask_order').agg(dura_volume:sum, volume:sum)
//!        bids = df[df.bid_order != 0].groupby('bid_order').agg(dura_volume:sum, volume:sum)
//!        groupby 默认 sort=True → 分组键升序；组内按原行顺序累加
//!     5. 对 asks.dura_volume / bids.dura_volume / asks.volume / bids.volume 四条序列
//!        各取 autocorr(lag=1..10) 再取绝对值，共 40 个值
//!        autocorr(lag=k) = 序列与自身 shift(k) 的 Pearson（成对完整观测），
//!        等价于 pearson(x[..n-k], x[k..])；有效样本 <2 或任一侧方差为 0 返回 NaN
//!
//!   输出顺序（= 旧代码 names 的顺序）：
//!     asks_duravolume_corr1..10_abs, bids_duravolume_corr1..10_abs,
//!     asks_volume_corr1..10_abs,     bids_volume_corr1..10_abs
//!
//!   names_save（旧文件里要保存的 base 名，写在 hm36_okay_database.py 里）：
//!     bids_duravolume_corr1_abs, bids_volume_corr1_abs, bids_duravolume_corr9_abs,
//!     asks_duravolume_corr5_abs, bids_volume_corr6_abs, bids_volume_corr4_abs,
//!     bids_duravolume_corr3_abs, asks_duravolume_corr3_abs

use std::collections::BTreeMap;

use crate::fast_csv_reader::{read_trade_fast_inner, TradeRecord};

/// 撤单标记
const FLAG_RETREAT: i32 = 32;

/// 窗口 14:54:00 ~ 14:57:00（真实钟点，闭区间，微秒精度）
const LO_US: i64 = (14 * 3600 + 54 * 60) * 1_000_000;
const HI_US: i64 = (14 * 3600 + 57 * 60) * 1_000_000;

/// 四个序列 × 10 个滞后
const N_LAG: usize = 10;
pub const N_OUT: usize = 4 * N_LAG;

/// 北京时间的当日微秒数（time_us 已含 8 小时偏移）。
#[inline]
fn day_us(time_us: i64) -> i64 {
    time_us.rem_euclid(86_400_000_000)
}

pub fn names() -> Vec<String> {
    let mut out = Vec::with_capacity(N_OUT);
    for side in ["asks", "bids"] {
        for metric in ["duravolume", "volume"] {
            for lag in 1..=N_LAG {
                out.push(format!("{}_{}_corr{}_abs", side, metric, lag));
            }
        }
    }
    out
}

/// pandas 的 Series.corr（Pearson）：成对完整观测，任一为 NaN 就丢掉这一对。
/// 有效样本 < 2、或任一侧方差为 0 时返回 NaN。
fn pearson(x: &[f64], y: &[f64]) -> f64 {
    let mut n = 0usize;
    let mut sx = 0.0f64;
    let mut sy = 0.0f64;
    let mut sxx = 0.0f64;
    let mut syy = 0.0f64;
    let mut sxy = 0.0f64;
    for i in 0..x.len() {
        if !x[i].is_finite() || !y[i].is_finite() {
            continue;
        }
        n += 1;
        sx += x[i];
        sy += y[i];
        sxx += x[i] * x[i];
        syy += y[i] * y[i];
        sxy += x[i] * y[i];
    }
    if n < 2 {
        return f64::NAN;
    }
    let nf = n as f64;
    let cov = sxy - sx * sy / nf;
    let vx = sxx - sx * sx / nf;
    let vy = syy - sy * sy / nf;
    if vx <= 0.0 || vy <= 0.0 {
        return f64::NAN;
    }
    cov / (vx.sqrt() * vy.sqrt())
}

/// autocorr(lag=k)：序列与自身 shift(k) 的 Pearson，等价于 x[..n-k] 与 x[k..]。
fn autocorr(vals: &[f64], lag: usize) -> f64 {
    let n = vals.len();
    if n <= lag {
        return f64::NAN;
    }
    pearson(&vals[..n - lag], &vals[lag..])
}

/// 按订单编号分组求和，返回按编号升序排列的 (编号, 累计量) 列表。
///
/// 用 BTreeMap 而不是 HashMap：旧代码 groupby 默认 sort=True 按键升序，
/// 而且 HashMap 的迭代顺序在 Rust 里不确定，会影响 autocorr 这种对顺序敏感的统计量。
fn group_sum(keys: &[i64], vals: &[f64]) -> Vec<f64> {
    let mut map: BTreeMap<i64, f64> = BTreeMap::new();
    for (k, v) in keys.iter().zip(vals.iter()) {
        *map.entry(*k).or_insert(0.0) += *v;
    }
    map.into_values().collect()
}

/// 逐股入口：算不出来返回 None（上游会补 NaN，保证因子个数恒定）。
pub fn compute_stock(code: &str, date: i64) -> Option<Vec<f32>> {
    // with_afternoon_adjust=false：旧代码没有调 adjust_afternoon，窗口是真实钟点。
    //
    // with_retreat=true 是必须的：本系列要用 flag==32 的撤单行。
    // 不能依赖库里的 with_retreat 过滤，它按字节比 "32"，对 2025 年前后的 "32.0" 不生效，
    // 所以这里读全量、自己按数值判断 flag。
    let trades = read_trade_fast_inner(code, date, true, false, usize::MAX).ok()?;

    // 1+2：窗口内（闭区间）的撤单行，保持文件里的原始顺序
    let rows: Vec<&TradeRecord> = trades
        .iter()
        .filter(|r| r.flag == FLAG_RETREAT)
        .filter(|r| (LO_US..=HI_US).contains(&day_us(r.time_us)))
        .collect();

    // 空集在旧代码里也不报错：空 groupby 的 autocorr 全是 NaN，整条输出 NaN
    let mut ask_keys: Vec<i64> = Vec::new();
    let mut bid_keys: Vec<i64> = Vec::new();
    let mut dv: Vec<f64> = Vec::new();
    let mut vol: Vec<f64> = Vec::new();
    if !rows.is_empty() {
        // 3：dura 以撤单子集里最早那一笔为基准，单位秒
        let t0 = rows.iter().map(|r| day_us(r.time_us)).min().unwrap();
        for r in rows.iter() {
            let dura = (day_us(r.time_us) - t0) as f64 / 1_000_000.0;
            let v = r.volume as f64;
            // 旧代码里空的委托编号是 NaN，NaN != 0 为真但 groupby 会丢掉 NaN 键；
            // 这里读出来是 0，直接用 0 当缺失，下面按 != 0 过滤，两边结果一致。
            ask_keys.push(r.ask_order);
            bid_keys.push(r.bid_order);
            dv.push(v * dura);
            vol.push(v);
        }
    }

    // 4：按订单编号分组求和（升序）
    let mut out: Vec<f32> = Vec::with_capacity(N_OUT);
    // 顺序必须与 names() 一致：asks_duravolume、bids_duravolume、asks_volume、bids_volume
    for (keys, vals) in [(&ask_keys, &dv), (&bid_keys, &dv), (&ask_keys, &vol), (&bid_keys, &vol)] {
        let mut sel_keys: Vec<i64> = Vec::new();
        let mut sel_vals: Vec<f64> = Vec::new();
        for (k, v) in keys.iter().zip(vals.iter()) {
            if *k != 0 {
                sel_keys.push(*k);
                sel_vals.push(*v);
            }
        }
        let sums = group_sum(&sel_keys, &sel_vals);
        // 5：lag=1..10 的 |autocorr|
        for lag in 1..=N_LAG {
            out.push(autocorr(&sums, lag).abs() as f32);
        }
    }
    debug_assert_eq!(out.len(), N_OUT);
    Some(out)
}

/// hm36 全是逐股因子，没有横截面版本。
pub fn compute_cross(_date: i64) -> Option<(Vec<String>, Vec<f32>)> {
    None
}
