//! hm090「日落」—— 旧框架 go 函数的 Rust 复现。
//!
//! 旧文件：/home/chenzongwei/pythoncode/成果hm90_日落/hm90_okay.py（模块级 go(date, code)）
//!
//! ════════════════════════════════════════════════════════════════════
//! 一、旧 go 做的事（逐步对照）
//! ════════════════════════════════════════════════════════════════════
//! 1) trade1 = adjust_afternoon(read_trade(code, last_trading_day(date)))
//!    trade2 = adjust_afternoon(read_trade(code, date))
//! 2) prepare(trade) 取 7 列 f64：
//!      [exchtime.astype(int64)//1e9, price, volume, turnover, flag, bid_order, ask_order]
//!    第一列是**整数秒**（换 Timestamp 后 astype(int64) 是纳秒，再 //1e9）。
//! 3) result1, cols1 = calculate_order_pair_metrics_more(t2, t1)          # tolerance 默认 0.001
//!    result2, cols2 = calculate_order_pair_metrics_more_v2(t2, t1, tolerance=0.00001)
//!    两份都是 (配对数, 44) 的 f64 矩阵，44 列列名 = 旧文件 go() 里的 base_cols。
//! 4) vals1, _ = get_features_factors(DataFrame(result1, columns=cols1))   # Python 默认参数
//!    vals2, _ = get_features_factors(DataFrame(result2, columns=cols2))
//!    每份 1914 个值（含 lyapunov 组，44 列 × 43.5 + 946 个相关系数）。
//! 5) res = vals1 + vals2（3828 个），再按 names_save_unfold 取 67 个返回。
//!
//! ════════════════════════════════════════════════════════════════════
//! 二、和库里现成流水线的关系（重点，别直接套 pipeline_order_pair_hm90）
//! ════════════════════════════════════════════════════════════════════
//! factor_pipeline.rs 里的 pipeline_order_pair_hm90 看起来就是同一件事，实测有三处对不上，
//! 直接拿来当 hm090 会得到完全不同的值：
//!
//! · **撤单行**：流水线用 read_trade_fast_inner(code, date, false, true, ..)，
//!   with_retreat=false 时库里那条过滤是「按字节比 b"32"」，而 2025 年前后的成交明细里
//!   撤单标记写的是 "32.0"，过滤不生效；旧 read_trade 是 `df[df.flag != 32]` 的数值比较，
//!   两边行数差一大截（000001@20250303：旧 56193 行，流水线 75205 行）。
//!   本文件改成 with_retreat=true 读全部，再自己按 flag==32 丢掉。
//!
//! · **时间精度**：流水线走 parse_trades_from_fast，时间取的是 TradeRecord.time_sec（f32），
//!   epoch 秒 1.7e9 在 f32 下分辨率只有 ±128 秒，而「之后 10 秒窗口」「成交时长」这些量
//!   全都要精确到秒。本文件改用 order_pair_metrics_pipeline::*_exact_inner，
//!   时间取 time_us/1e6（整数秒），与旧框架的 //1e9 逐位一致。
//!
//! · **特征清单不同**：库里的 features（get_features_factors_rust_full，对应 Python 的
//!   with_lyapunov_exponent=False）关闭了 lyapunov，同时又多了 curvature / quad_coef
//!   两项（每列 2 个），所以每份不是旧代码的 1914 而是 1958，两份合计 3916。
//!   实测 `rp.verify_get_features_factors_rust` 在 44 列上返回 1958 个名字，
//!   前 21 组与 Python 的 get_features_names(with_lyapunov_exponent=False) 逐项一致，
//!   只是中间插了 curvature / quad_coef 两组。入选的 67 个名字一个都没落在
//!   lyapunov / curvature / quad_coef 上，所以按 3916 这套清单重算了 IDX 即可，
//!   值不受影响（下标由 `rp.verify_get_features_factors_rust` 返回的名字直接定位，
//!   不是估出来的）。
//!
//! ════════════════════════════════════════════════════════════════════
//! 三、下午平移与窗口（对齐 pandas 的 between_time）
//! ════════════════════════════════════════════════════════════════════
//! 旧 adjust_afternoon：set_index('exchtime').between_time(start, end)，
//! 上午 [09:30:00, 11:30:00]、下午 [13:00:00, 14:57:00]，**微秒级闭区间**，
//! 下午部分整体前移 90 分钟，最后 concat（上午全部在前、下午全部在后）。
//! 顺序有意义：order_pair 内部按时间排序用的是稳定排序，同秒的行保持输入顺序。
//! 本文件按 time_us 的当日偏移（rem_euclid(一天)）判窗口，微秒精度，不是按整秒截断。
//!
//! ════════════════════════════════════════════════════════════════════
//! 四、名字与重名
//! ════════════════════════════════════════════════════════════════════
//! names() 返回旧 go() 的输出顺序（= 旧文件 names_save 去掉 _fold 后的 67 项）。
//! 其中 v1_price_ratio_diff_mean_below_p10 出现两次（一项来自自己的 _fold 版本），
//! 按规范第二次加 _dup2 后缀，值不变、列数不变。
//!
//! 已知差异（不影响入选的 67 个）：market_fractal_dim_diff / abs_market_fractal_dim_diff
//! 两列在新旧两个库之间就有实现差异（000001@20250303 上 28965 行里 28633 行不同），
//! 入选名单里没有任何 fractal 相关的量，也没有以它为自变量的相关系数项。
//!
//! 另外，price/volume/turnover 在 Rust 里是 f32（旧框架是 f64），相对误差量级 1e-7，
//! 远在验收阈值 1e-4 之内。

use crate::data_paths;
use crate::fast_csv_reader;
use crate::features;
use crate::order_pair_metrics_pipeline;
use std::sync::LazyLock;

/// 旧 go 的 v1/v2 两份特征总长（各 1914，含 lyapunov）。只作对照记录，不参与计算。
#[allow(dead_code)]
const LEGACY_FULL: usize = 3828;
/// 库里 features 的实际输出长度：v1/v2 两份各 1958，合计 3916。
const FULL: usize = 3916;
/// 每份特征的长度（关了 lyapunov、多了 curvature/quad_coef）。
const HALF: usize = 1958;

const TOLERANCE_V1: f64 = 0.001;
const TOLERANCE_V2: f64 = 0.00001;

/// 一天的微秒数。
const DAY_US: i64 = 86_400_000_000;
/// 下午时段前移的微秒数。
const AFTERNOON_SHIFT_US: i64 = 90 * 60 * 1_000_000;
const MORNING_START_US: i64 = (9 * 3600 + 30 * 60) * 1_000_000;
const MORNING_END_US: i64 = (11 * 3600 + 30 * 60) * 1_000_000;
const AFTERNOON_START_US: i64 = (13 * 3600) * 1_000_000;
const AFTERNOON_END_US: i64 = (14 * 3600 + 57 * 60) * 1_000_000;

// NAMES 是旧 go() 的返回顺序；IDX 是每一项在 FULL（3916）向量里的下标。
// 3916 = [v1 的 1958 项] + [v2 的 1958 项]。1958 那套清单来自
// `rp.verify_get_features_factors_rust(44 列矩阵, base_cols)` 返回的名字，
// 与 worker 里 get_features_factors_rust_full(.., true) 用的是同一条代码路径。
// base_cols 就是旧文件 go() 里的那 44 列，顺序与 order_pair 返回的列名一致。
const NAMES: [&str; 67] = [
    "v1_abs_avg_trade_amount_diff_corr_trade_count",
    "v1_abs_avg_trade_amount_diff_p5",
    "v1_abs_buy_order_trend_diff_corr_abs_sell_order_trend_diff",
    "v1_abs_impact1_diff_corr_market_interval_cv_diff",
    "v1_abs_impact1_diff_mean",
    "v1_abs_impact2_diff_corr_abs_price_radiation_diff",
    "v1_abs_impact2_diff_std",
    "v1_abs_market_hurst_exp_diff_cv",
    "v1_abs_market_hurst_exp_diff_mean_above_p90",
    "v1_abs_market_hurst_exp_diff_p95",
    "v1_abs_opposite_direction_trend_diff_cv",
    "v1_abs_price_radiation_diff_corr_market_interval_cv_diff",
    "v1_abs_same_direction_trend_diff_corr_abs_sell_order_trend_diff",
    "v1_abs_same_direction_trend_diff_corr_trade_count",
    "v1_abs_signed_turnover_diff_p25",
    "v1_abs_turnover_diff_std",
    "v1_avg_trade_amount_diff_corr_market_lz_complexity_diff",
    "v1_buy_ratio_stability_median",
    "v1_impact1_diff_corr_price_radiation_diff",
    "v1_impact1_diff_trend",
    "v1_market_hurst_exp_diff_mean_below_p10",
    "v1_market_interval_cv_diff_kurt",
    "v1_opposite_direction_trend_diff_corr_buy_order_trend_diff",
    "v1_opposite_direction_trend_diff_corr_market_interval_cv_diff",
    "v1_opposite_direction_trend_diff_mean",
    "v1_opposite_direction_trend_diff_mean_above_p90",
    "v1_price_radiation_diff_corr_abs_impact2_diff",
    "v1_price_radiation_diff_corr_market_hurst_exp_diff",
    "v1_price_radiation_diff_corr_market_interval_cv_diff",
    "v1_price_ratio_diff_corr_avg_trade_amount_diff",
    "v1_price_ratio_diff_corr_price_radiation_diff",
    "v1_price_ratio_diff_corr_same_direction_trend_diff",
    "v1_price_ratio_diff_mean_below_p10",
    "v1_price_ratio_diff_mean_below_p10_dup2",
    "v1_price_ratio_diff_p5",
    "v1_same_direction_trend_diff_corr_trade_count",
    "v1_same_direction_trend_diff_mean_above_p90",
    "v1_today_order_id_corr_abs_avg_trade_amount_diff",
    "v1_today_order_id_entropy_1d",
    "v1_today_order_id_p25",
    "v1_today_order_id_skew",
    "v1_trade_count_mean",
    "v1_trade_duration_diff_max_range_product",
    "v1_turnover_diff_corr_trade_count",
    "v2_abs_avg_trade_amount_diff_autocorr1",
    "v2_abs_avg_trade_amount_diff_corr_abs_price_radiation_diff",
    "v2_abs_counterparty_order_diff_entropy_1d",
    "v2_abs_impact2_diff_mean_above_p90",
    "v2_abs_opposite_direction_trend_diff_corr_abs_buy_order_trend_diff",
    "v2_abs_price_ratio_diff_p25",
    "v2_abs_same_direction_trend_diff_corr_abs_sell_order_trend_diff",
    "v2_abs_signed_turnover_diff_corr_abs_trade_duration_diff",
    "v2_abs_signed_turnover_diff_p25",
    "v2_abs_signed_turnover_diff_p5",
    "v2_abs_signed_turnover_diff_trend",
    "v2_avg_trade_amount_diff_median",
    "v2_avg_trade_amount_diff_p25",
    "v2_market_hurst_exp_diff_kurt",
    "v2_opposite_direction_trend_diff_mean",
    "v2_price_radiation_diff_corr_abs_impact1_diff",
    "v2_price_radiation_diff_corr_market_interval_cv_diff",
    "v2_price_ratio_diff_corr_abs_avg_trade_amount_diff",
    "v2_price_ratio_diff_corr_abs_same_direction_trend_diff",
    "v2_price_ratio_diff_corr_avg_trade_amount_diff",
    "v2_today_order_id_autocorr1_abs",
    "v2_turnover_diff_corr_price_ratio_diff",
    "v2_turnover_diff_p25",
];

const IDX: [usize; 67] = [
    1654, 244, 1748, 1588, 21, 1599, 110, 480, 832, 392, 470, 1699, 1723, 1734, 283, 106, 1220,
    51, 1090, 577, 871, 213, 1391, 1413, 14, 806, 1308, 1321, 1323, 1049, 1052, 1054, 840, 840,
    224, 1390, 805, 903, 1870, 264, 132, 43, 1923, 1005, 2466, 3596, 3856, 2772, 3693, 2242,
    3681, 3489, 2241, 2197, 2549, 2010, 2230, 2169, 1972, 3265, 3281, 3023, 3028, 3007, 2486,
    2924, 2224,
];

pub fn names() -> Vec<String> {
    NAMES.iter().map(|s| s.to_string()).collect()
}

/// 交易日历（升序），只读一次。
static TRADING_DAYS: LazyLock<Vec<i64>> = LazyLock::new(|| {
    let path = data_paths::basic_info_path("").join("calendar.csv");
    std::fs::read_to_string(path)
        .map(|s| {
            s.lines()
                .filter_map(|x| x.trim().parse::<i64>().ok())
                .collect()
        })
        .unwrap_or_default()
});

/// 上一个交易日（严格小于 date 的最大交易日）。
fn last_trading_day(date: i64) -> Option<i64> {
    let days = TRADING_DAYS.as_slice();
    if days.is_empty() {
        return None;
    }
    let idx = days.partition_point(|&d| d < date);
    if idx == 0 {
        None
    } else {
        Some(days[idx - 1])
    }
}

/// 读一个交易日的逐笔，做旧框架那两步：丢掉撤单行 + 下午时段平移。
///
/// 不再用 read_trade_fast_inner 自带的 with_retreat / with_afternoon_adjust：
/// 前者的字节比较漏掉 "32.0" 形式的撤单标记，后者按整秒判窗口而旧代码是微秒闭区间。
fn read_day(code: &str, date: i64) -> Option<Vec<fast_csv_reader::TradeRecord>> {
    let raw = fast_csv_reader::read_trade_fast_inner(code, date, true, false, usize::MAX).ok()?;
    let mut morning: Vec<fast_csv_reader::TradeRecord> = Vec::with_capacity(raw.len());
    let mut afternoon: Vec<fast_csv_reader::TradeRecord> = Vec::new();
    for mut r in raw {
        if r.flag == 32 {
            continue;
        }
        let tod = r.time_us.rem_euclid(DAY_US);
        if (MORNING_START_US..=MORNING_END_US).contains(&tod) {
            morning.push(r);
        } else if (AFTERNOON_START_US..=AFTERNOON_END_US).contains(&tod) {
            r.time_us -= AFTERNOON_SHIFT_US;
            afternoon.push(r);
        }
    }
    morning.append(&mut afternoon);
    Some(morning)
}

pub fn compute_stock(code: &str, date: i64) -> Option<Vec<f32>> {
    let prev = last_trading_day(date)?;
    let t2 = read_day(code, date)?;
    let t1 = read_day(code, prev)?;

    let (result1, cols1) = order_pair_metrics_pipeline::calculate_order_pair_metrics_more_exact_inner(
        &t2,
        &t1,
        TOLERANCE_V1,
    );
    let (result2, cols2) =
        order_pair_metrics_pipeline::calculate_order_pair_metrics_more_v2_exact_inner(
            &t2,
            &t1,
            TOLERANCE_V2,
        );

    // 与流水线一致：空配对矩阵给半量 NaN，保证总长度恒定。
    let (vals1, _) = if result1.nrows() == 0 {
        (vec![f32::NAN; HALF], Vec::new())
    } else {
        features::get_features_factors_rust_full(&result1.mapv(|v| v as f32).view(), &cols1, true)
    };
    let (vals2, _) = if result2.nrows() == 0 {
        (vec![f32::NAN; HALF], Vec::new())
    } else {
        features::get_features_factors_rust_full(&result2.mapv(|v| v as f32).view(), &cols2, true)
    };

    let mut full = vals1;
    full.extend(vals2);
    if full.len() != FULL {
        // 长度不是预期的两份 1870：宁可整条给 NaN，也不要错位取值。
        return None;
    }
    Some(IDX.iter().map(|&i| full[i]).collect())
}

pub fn compute_cross(_date: i64) -> Option<(Vec<String>, Vec<f32>)> {
    None
}
