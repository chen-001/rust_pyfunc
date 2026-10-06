//! hm077「聚义厅（同向/反向订单）」—— 旧框架 go 函数的 Rust 复现。
//!
//! 旧文件：/home/chenzongwei/pythoncode/成果hm76_hm77聚义厅/hm77_okay.py（706 行，模块级 go(date, code)）
//! 与 hm076 是同一个 notebook 出来的姊妹系列，重活都在 `one(df, direction)` 里；
//! 差别只有两处（下面第二、三条），所以本文件的结构刻意与 src/okay_database/hm076.rs 对齐，
//! 读盘、pandas 语义、数值工具那几段是逐字搬过来的。
//!
//! ════════════════════════════════════════════════════════════════════
//! 一、旧 go 做的事（逐步对照）
//! ════════════════════════════════════════════════════════════════════
//!   df = dwo.read_trade(code, date)          # 默认参数，丢 flag==32；exchtime 变 Timestamp
//!   df = dwo.adjust_afternoon(df)            # 上午[09:30,11:30] + 下午[13:00,14:57]（下午整体-90min）
//!   df = df.sort_values(['volume','exchtime'])# 主键 volume、次键 exchtime，稳定
//!   res1 = one(df, 'same')；res2 = one(df, 'diff')；res = res1 + res2   # 各 960，共 1920
//!   names = 960 个基名；names = [i+'_sameside' for i in names] + [i+'_diffside' for i in names]
//!   res = {k:v for k,v in zip(names,res)}；res = [res[name] for name in names_save_unfold]
//!   return res                               # 55 个值（names_save 去掉 _fold 之后 55 项）
//!
//! one() 内部（旧文件第 52~436 行）：
//!   1) rp.calculate_order_time_gap_and_price_percentile_ultra_sorted_v2(
//!          volume, exchtime, price, flag, ask_order, bid_order, 100, direction)
//!      → (n, 27) 矩阵，列名是 27 个中文名，旧代码过 pypinyin(Style.TONE3) 后当列名，再 .dropna()。
//!   2) 按「买单标识」（第 24 列，0-based 下标 23）拆成 features_mai3dan1（==1）与
//!      features_mai4dan1（==0）。
//!   3) 每边给 12 个「时间间隔」列与 10 个「价格分位数」列各加一条**按行**的 trend 序列列
//!      （rp.trend_2d(..., axis=1)），拼成 24 列：13 + 11。
//!   4) 这 24 列上算：corrwith(订单价格/订单总量/订单编号/订单时间) 4×24、
//!      mean/std/skew/kurt 4×24、autocorr 24、按列的 trend 24 → 每边 10×24 = 240。
//!   5) groupby('订单总量').agg(mean) 得到 29 列（24 指标 + 5 常量），同样算那一整套 → 240。
//!   6) res = [240 + 240] × 2 + [240 + 240] × 2？不是 —— 实际顺序见下面第五条。
//!
//! ════════════════════════════════════════════════════════════════════
//! 二、与 hm076 的差别之一：走 v2 入口（按下单方向分买卖，不看成交 flag）
//! ════════════════════════════════════════════════════════════════════
//!   v2 的订单聚合是「ask_order[i] != 0 记一笔卖单、bid_order[i] != 0 记一笔买单」，
//!   两条 if 各自独立、顺序执行（同一行可能两条都命中），HashMap 只按 order_id 建键、
//!   不区分买卖 —— 所以同一个 id 出现在 ask 和 bid 里时会命中同一条订单记录。
//!   这一切照抄 src/order_records_ultra_sorted_v2_optimized.rs，一个字没改。
//!
//!   由此带来一个**必须原样保留**的后果：A 股逐笔里每行的 ask_order 与 bid_order 都非 0，
//!   而写回时先写 ask 分支（买单标识=0）、再写 bid 分支（买单标识=1），后者覆盖前者，
//!   于是「买单标识」恒等于 1，features_mai4dan1 恒为空表。
//!   实测 000001@20250303：dropna 后 52453 行，mai3dan1 取到 52453 行、mai4dan1 取到 0 行。
//!   空表那一支的取值口径（都实测过，不是猜的）：
//!     · mean/std/skew/kurt          → 全 NaN
//!     · corrwith（含 autocorr 那两块）→ 全 NaN
//!     · rp.trend_2d(空矩阵, axis=0) → 全 0.0（不是 NaN）
//!   Side::trend24/ttrend24 走 trend_1d，空列给 0.0，正好对上。
//!   55 个入选因子的下标全部落在 mai3dan1 那几块里（下面第五条列了块区间），
//!   空表那几块只作为占位值存在，保证列数恒定。
//!
//! ════════════════════════════════════════════════════════════════════
//! 三、与 hm076 的差别之二：没有 v3 那一段
//! ════════════════════════════════════════════════════════════════════
//!   hm076 最后还有 rp.calculate_trade_price_statistics_by_volume_v3 的 4×105 = 420 个值，
//!   hm77 的 one() 在 960 个值处直接 return（旧文件第 436~439 行），没有这一段。
//!   所以 ONE_LEN = 960、FULL_LEN = 1920，Day 里也不需要 presort_flag / vwap。
//!
//! ════════════════════════════════════════════════════════════════════
//! 四、必须照抄的几处 pandas 语义（与 hm076 同一套）
//! ════════════════════════════════════════════════════════════════════
//! · 时间：read_trade 先把 exchtime 微秒转成 pandas Timestamp（走 conversion.pyx 的
//!   cast_from_unit_vectorized，含 np.round(frac,9) 那一步），再 +8 小时；本文件逐行照抄。
//!   adjust_afternoon 的窗口是**纳秒级闭区间**。exchtime 传给 v2 时是 `.to_numpy(float)`，
//!   对 datetime64[ns] 就是纳秒值（1.7e18），v2 内部再 `/1e9` 变秒 —— 与 hm076 一致。
//! · 排序：sort_values(['volume','exchtime']) 是 np.lexsort，主键 volume、次键 exchtime，稳定。
//! · dropna 只丢 NaN。
//! · corrwith(Series)：走 Series.corr → nanops.nancorr → np.corrcoef（两遍法 + clip 到 [-1,1]），
//!   成对完整观测。
//! · autocorr 那两块走的是 **DataFrame.corrwith(DataFrame)**：左边 13 列（12 时间间隔 + trend），
//!   右边是 24 列的 shift(1)，pandas 给回 **24** 个值，后 11 个是 NaN（实测确认，不是 13 个）。
//!   值本身是逐列 num/dom，dom = (count-1)*std_l*std_r。shift(1) 是行位置下移，首行 NaN。
//! · groupby(...).agg({k:'mean'})：pandas 的 group_mean 是 Kahan 补偿求和再除以计数
//!   （不是增量均值、也不是 sum/count），NaN 不计入计数，键为 NaN 的行丢掉，分组键升序输出。
//! · mean/std/skew/kurt 走库里 features_f64 的 f64 通路，按名字取（hm076 实测这一档与
//!   pandas 直接算的最大相对偏差约 1e-9，远在 1e-4 验收线内）。
//!
//! ════════════════════════════════════════════════════════════════════
//! 五、res 的块结构（每块都是 24 个值，与旧 names 的 960 个基名一一对应）
//! ════════════════════════════════════════════════════════════════════
//!   [  0,  96) mai3dan1_corr × 4 个目标（订单价格 / 订单总量 / 订单编号 / 订单时间）
//!   [ 96, 192) mai4dan1_corr × 4
//!   [192, 336) mai3dan1 mean/std/skew/kurt（4×24）+ autocorr（24）+ trend（24）
//!   [336, 480) mai4dan1 同上
//!   [480, 576) together_mai3dan1 corr × 4
//!   [576, 672) together_mai4dan1 corr × 4
//!   [672, 816) together_mai3dan1 mean/std/skew/kurt + autocorr + trend
//!   [816, 960) together_mai4dan1 同上
//!   one() 拼装顺序 = corr66, corr83, stat66, ac66, tr66, stat83, ac83, tr83,
//!                   tcorr66, tcorr83, tstat66, tac66, ttr66, tstat83, tac83, ttr83
//!   （66 = 买单那一边，83 = 卖单那一边；变量名沿用 hm076，值由「买单标识」决定。）
//!
//! ════════════════════════════════════════════════════════════════════
//! 六、重名
//! ════════════════════════════════════════════════════════════════════
//!   names_save_unfold 的 55 项两两不同，不需要 _dup2；1920 个全名也两两不同。
//!   注意旧代码里那 960 个基名的措辞与它实际对应的算式并不总是一致（例如
//!   `mai3dan1_corr_..._jia4ge2` 有一部分落在 kurt 那一块），这是旧文件自己生成的标签，
//!   这里只保证「同一个下标给出同一个值」，不改名也不挪位置。

use crate::data_paths;
use crate::features_f64;
use ndarray::Array2;
// ────────────────────────── 常量 ──────────────────────────

/// 一天的纳秒数。
const DAY_NS: i64 = 86_400_000_000_000;
/// exchtime（微秒，UTC）转成 pandas Timestamp 后要加的北京时区偏移。
const OFFSET_NS: i64 = 8 * 3600 * 1_000_000_000;
/// 下午时段整体前移 90 分钟。
const SHIFT_NS: i64 = 90 * 60 * 1_000_000_000;
const MORNING_START_NS: i64 = 34_200_000_000_000; // 09:30:00
const MORNING_END_NS: i64 = 41_400_000_000_000; // 11:30:00
const AFTERNOON_START_NS: i64 = 46_800_000_000_000; // 13:00:00
const AFTERNOON_END_NS: i64 = 53_820_000_000_000; // 14:57:00

/// ultra_sorted 输出的 27 列列名（旧代码把中文列名过了 pypinyin，这里直接写拼音）。
/// 只作对照记录，计算里用下标访问。
#[allow(dead_code)]
const FEATS: [&str; 27] = [
    "zui4jin4shi2jian1jian4ge2",
    "ping2jun1shi2jian1jian4ge2_1%",
    "ping2jun1shi2jian1jian4ge2_2%",
    "ping2jun1shi2jian1jian4ge2_3%",
    "ping2jun1shi2jian1jian4ge2_4%",
    "ping2jun1shi2jian1jian4ge2_5%",
    "ping2jun1shi2jian1jian4ge2_10%",
    "ping2jun1shi2jian1jian4ge2_20%",
    "ping2jun1shi2jian1jian4ge2_30%",
    "ping2jun1shi2jian1jian4ge2_40%",
    "ping2jun1shi2jian1jian4ge2_50%",
    "ping2jun1shi2jian1jian4ge2_quan2bu4",
    "jia4ge2fen1wei4shu4_10%",
    "jia4ge2fen1wei4shu4_20%",
    "jia4ge2fen1wei4shu4_30%",
    "jia4ge2fen1wei4shu4_40%",
    "jia4ge2fen1wei4shu4_50%",
    "jia4ge2fen1wei4shu4_60%",
    "jia4ge2fen1wei4shu4_70%",
    "jia4ge2fen1wei4shu4_80%",
    "jia4ge2fen1wei4shu4_90%",
    "jia4ge2fen1wei4shu4_quan2bu4",
    "ding4dan1bian1hao4",
    "mai3dan1biao1zhi4",
    "ding4dan1zong3liang4",
    "ding4dan1shi2jian1",
    "ding4dan1jia4ge2",
];
/// 「买单标识」列（0-based）。
const C_BUY_FLAG: usize = 23;
/// 「订单总量」列（groupby 的键）。
const C_ORDER_VOLUME: usize = 24;

/// 参与全部统计的 24 列：13 个时间间隔列（含 trend 序列）+ 11 个价格分位数列（含 trend 序列）。
const N24: [&str; 24] = [
    "zui4jin4shi2jian1jian4ge2",
    "ping2jun1shi2jian1jian4ge2_1%",
    "ping2jun1shi2jian1jian4ge2_2%",
    "ping2jun1shi2jian1jian4ge2_3%",
    "ping2jun1shi2jian1jian4ge2_4%",
    "ping2jun1shi2jian1jian4ge2_5%",
    "ping2jun1shi2jian1jian4ge2_10%",
    "ping2jun1shi2jian1jian4ge2_20%",
    "ping2jun1shi2jian1jian4ge2_30%",
    "ping2jun1shi2jian1jian4ge2_40%",
    "ping2jun1shi2jian1jian4ge2_50%",
    "ping2jun1shi2jian1jian4ge2_quan2bu4",
    "shi2jian1jian4ge2_trendxu4lie4",
    "jia4ge2fen1wei4shu4_10%",
    "jia4ge2fen1wei4shu4_20%",
    "jia4ge2fen1wei4shu4_30%",
    "jia4ge2fen1wei4shu4_40%",
    "jia4ge2fen1wei4shu4_50%",
    "jia4ge2fen1wei4shu4_60%",
    "jia4ge2fen1wei4shu4_70%",
    "jia4ge2fen1wei4shu4_80%",
    "jia4ge2fen1wei4shu4_90%",
    "jia4ge2fen1wei4shu4_quan2bu4",
    "jia4ge2fen1wei4shu4_trendxu4lie4",
];
/// 一个方向 960 个值里入选的 55 个名字（= 旧 names_save 去掉 `_fold`，顺序原样）。
const NAMES: [&str; 55] = [
    "mai3dan1shi2jian1jian4ge210perc_kurt_sameside",
    "mai3dan1_jia4ge2fen1wei4shu410perc_trend_volumeju4he2_sameside",
    "mai3dan1_jia4ge2fen1wei4shu4trendxu4lie4_trend_volumeju4he2_sameside",
    "mai3dan1shi2jian1jian4ge210perc_mean_volumeju4he2_diffside",
    "mai3dan1_corr_jia4ge2fen1wei4shu430perc_jia4ge2_sameside",
    "mai3dan1_jia4ge2fen1wei4shu490perc_kurt_sameside",
    "mai3dan1jia4ge2fen1wei4shu4trendxu4lie4_mean_sameside",
    "mai3dan1_jia4ge2fen1wei4shu480perc_kurt_diffside",
    "mai3dan1shi2jian1jian4ge2trendxu4lie4_trend_volumeju4he2_sameside",
    "mai3dan1shi2jian1jian4ge2trendxu4lie4_mean_volumeju4he2_diffside",
    "mai3dan1_corr_shi2jian1jian4ge21perc_jia4ge2_trend_diffside",
    "mai3dan1_corr_shi2jian1jian4ge2quan2bu4_jia4ge2_autocorr_diffside",
    "mai3dan1shi2jian1jian4ge240perc_kurt_diffside",
    "mai3dan1_corr_shi2jian1jian4ge2quan2bu4_jia4ge2_trend_volumeju4he2_diffside",
    "mai3dan1_jia4ge2fen1wei4shu460perc_trend_volumeju4he2_diffside",
    "mai3dan1_jia4ge2fen1wei4shu420perc_skew_diffside",
    "mai3dan1shi2jian1jian4ge25perc_mean_diffside",
    "mai3dan1_corr_shi2jian1jian4ge250perc_jia4ge2_trend_volumeju4he2_sameside",
    "mai3dan1jia4ge2fen1wei4shu4quan2bu4_std_sameside",
    "mai3dan1shi2jian1jian4ge2trendxu4lie4_std_diffside",
    "mai3dan1shi2jian1jian4ge22perc_mean_volumeju4he2_diffside",
    "mai3dan1shi2jian1jian4ge24perc_mean_diffside",
    "mai3dan1_jia4ge2fen1wei4shu490perc_trend_volumeju4he2_sameside",
    "mai3dan1_corr_jia4ge2fen1wei4shu450perc_cheng2jiao1shi2jian1_diffside",
    "mai3dan1shi2jian1jian4ge220perc_mean_volumeju4he2_diffside",
    "mai3dan1shi2jian1jian4ge2trendxu4lie4_skew_sameside",
    "mai3dan1shi2jian1jian4ge2trendxu4lie4_trend_diffside",
    "mai3dan1shi2jian1jian4ge23perc_kurt_sameside",
    "mai3dan1shi2jian1jian4ge250perc_kurt_diffside",
    "mai3dan1shi2jian1jian4ge2quan2bu4_mean_volumeju4he2_sameside",
    "mai3dan1_jia4ge2fen1wei4shu450perc_trend_volumeju4he2_diffside",
    "mai3dan1_corr_shi2jian1jian4ge250perc_jia4ge2_trend_volumeju4he2_diffside",
    "mai3dan1_corr_shi2jian1jian4ge2quan2bu4_jia4ge2_trend_volumeju4he2_sameside",
    "mai3dan1_corr_shi2jian1jian4ge2trendxu4lie4_jia4ge2_diffside",
    "mai3dan1_corr_jia4ge2fen1wei4shu4trendxu4lie4_ding4dan1bian1hao4_sameside",
    "mai3dan1_corr_zui4jin4shi2jian1jian4ge2_jia4ge2_sameside",
    "mai3dan1shi2jian1jian4ge2trendxu4lie4_skew_diffside",
    "mai3dan1_corr_shi2jian1jian4ge250perc_cheng2jiao1shi2jian1_diffside",
    "mai3dan1_corr_shi2jian1jian4ge2quan2bu4_ding4dan1bian1hao4_diffside",
    "mai3dan1_corr_jia4ge2fen1wei4shu4trendxu4lie4_ding4dan1bian1hao4_diffside",
    "mai3dan1shi2jian1jian4ge2trendxu4lie4_mean_diffside",
    "mai3dan1jia4ge2fen1wei4shu480perc_mean_diffside",
    "mai3dan1shi2jian1jian4ge21perc_mean_volumeju4he2_diffside",
    "mai3dan1shi2jian1jian4ge220perc_mean_sameside",
    "mai3dan1_jia4ge2fen1wei4shu450perc_kurt_diffside",
    "mai3dan1shi2jian1jian4ge2trendxu4lie4_autocorr_sameside",
    "mai3dan1_corr_jia4ge2fen1wei4shu480perc_cheng2jiao1shi2jian1_sameside",
    "mai3dan1_jia4ge2fen1wei4shu410perc_trend_diffside",
    "mai3dan1_jia4ge2fen1wei4shu460perc_trend_volumeju4he2_sameside",
    "mai3dan1_corr_shi2jian1jian4ge22perc_jia4ge2_diffside",
    "mai3dan1_jia4ge2fen1wei4shu4quan2bu4_trend_volumeju4he2_diffside",
    "mai3dan1_corr_shi2jian1jian4ge250perc_ding4dan1bian1hao4_diffside",
    "mai3dan1_corr_zui4jin4shi2jian1jian4ge2_ding4dan1bian1hao4_diffside",
    "mai3dan1_corr_shi2jian1jian4ge21perc_ding4dan1bian1hao4_diffside",
    "mai3dan1_jia4ge2fen1wei4shu4trendxu4lie4_trend_volumeju4he2_diffside",
];

/// 上面 55 个名字在「两个方向拼起来的 1920 个值」里的下标。
const IDX: [usize; 55] = [
    270, 805, 815, 1638, 15, 285, 215, 1244, 804, 1644, 1273, 1259, 1233, 1763,
    1770, 1214, 1157, 802, 238, 1188, 1634, 1156, 813, 1049, 1639, 252, 1284, 267,
    1234, 683, 1769, 1762, 803, 972, 71, 0, 1212, 1042, 1019, 1031, 1164, 1172,
    1633, 199, 1241, 300, 92, 1285, 810, 962, 1774, 1018, 1008, 1009, 1775,
];

/// 一个方向的输出长度（8 块 × 24 + 8 块 × 24，按上面第五条的拼装顺序）。
const ONE_LEN: usize = 960;
/// 两个方向拼起来的总长度。
const FULL_LEN: usize = ONE_LEN * 2;

pub fn names() -> Vec<String> {
    NAMES.iter().map(|s| s.to_string()).collect()
}
// ────────────────────────── 数值小工具 ──────────────────────────

/// numpy 的成对求和（PW_BLOCKSIZE=128、8 路展开），pandas 的 .sum()/.mean() 走这条。
fn pairwise_sum(a: &[f64]) -> f64 {
    let n = a.len();
    if n < 8 {
        let mut r = 0.0f64;
        for i in 0..n {
            r += a[i];
        }
        return r;
    }
    if n <= 128 {
        let mut r = [a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]];
        let mut i = 8usize;
        let stop = n - (n % 8);
        while i < stop {
            for k in 0..8 {
                r[k] += a[i + k];
            }
            i += 8;
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        while i < n {
            res += a[i];
            i += 1;
        }
        return res;
    }
    let mut n2 = n / 2;
    n2 -= n2 % 8;
    pairwise_sum(&a[..n2]) + pairwise_sum(&a[n2..])
}

/// np.corrcoef(a, b)[0,1]：两遍法（np.average 的成对求和 → 中心化 → 点积/(n-1)）
/// → `(c/sa)/sb` → clip 到 [-1,1]。成对完整观测由调用方事先丢掉 NaN。
fn np_corrcoef(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len();
    if n == 0 {
        return f64::NAN;
    }
    let fact = n as f64 - 1.0;
    let ma = pairwise_sum(a) / n as f64;
    let mb = pairwise_sum(b) / n as f64;
    let (mut daa, mut dbb, mut dab) = (0.0f64, 0.0f64, 0.0f64);
    for i in 0..n {
        let x = a[i] - ma;
        let y = b[i] - mb;
        daa += x * x;
        dbb += y * y;
        dab += x * y;
    }
    let r = (dab / fact) / (daa / fact).sqrt() / (dbb / fact).sqrt();
    if r.is_nan() {
        f64::NAN
    } else {
        r.clamp(-1.0, 1.0)
    }
}

/// Series.corr(other)：成对完整观测 + np.corrcoef。
fn series_corr(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    let mut x: Vec<f64> = Vec::with_capacity(n);
    let mut y: Vec<f64> = Vec::with_capacity(n);
    for i in 0..n {
        if !a[i].is_nan() && !b[i].is_nan() {
            x.push(a[i]);
            y.push(b[i]);
        }
    }
    if x.is_empty() {
        return f64::NAN;
    }
    np_corrcoef(&x, &y)
}

/// DataFrame.corrwith(DataFrame) 的逐列口径：num/dom，dom = (count-1)*std_l*std_r。
fn df_corrwith(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    let mut la: Vec<f64> = Vec::with_capacity(n);
    let mut lb: Vec<f64> = Vec::with_capacity(n);
    for i in 0..n {
        if !a[i].is_nan() && !b[i].is_nan() {
            la.push(a[i]);
            lb.push(b[i]);
        }
    }
    let cnt = la.len();
    if cnt == 0 {
        return f64::NAN;
    }
    let ma = pairwise_sum(&la) / cnt as f64;
    let mb = pairwise_sum(&lb) / cnt as f64;
    let (mut num, mut va, mut vb) = (0.0f64, 0.0f64, 0.0f64);
    for i in 0..cnt {
        let da = la[i] - ma;
        let db = lb[i] - mb;
        num += da * db;
        va += da * da;
        vb += db * db;
    }
    let d = cnt as f64 - 1.0;
    let dom = d * (va / d).sqrt() * (vb / d).sqrt();
    num / dom
}

/// rp.trend（trend_2d 的一维版）：丢掉非有限值后与「原始位置 +1」做 Pearson；
/// 样本 < 2 或任一侧方差为 0 给 0.0（不是 NaN）。
fn trend_1d(v: &[f64]) -> f64 {
    let mut values: Vec<f64> = Vec::new();
    let mut idx: Vec<f64> = Vec::new();
    for (i, &x) in v.iter().enumerate() {
        if x.is_finite() {
            values.push(x);
            idx.push((i + 1) as f64);
        }
    }
    let n = values.len();
    if n < 2 {
        return 0.0;
    }
    let mean_x = values.iter().sum::<f64>() / n as f64;
    let mean_y = idx.iter().sum::<f64>() / n as f64;
    let (mut cov, mut vx, mut vy) = (0.0f64, 0.0f64, 0.0f64);
    for i in 0..n {
        let dx = values[i] - mean_x;
        let dy = idx[i] - mean_y;
        cov += dx * dy;
        vx += dx * dx;
        vy += dy * dy;
    }
    if vx == 0.0 || vy == 0.0 {
        return 0.0;
    }
    cov / (vx.sqrt() * vy.sqrt())
}

/// 从 features_f64 的 (vals, names) 里按名字取值。
fn pick(vals: &[f64], names: &[String], want: &str) -> f64 {
    for (i, n) in names.iter().enumerate() {
        if n == want {
            return vals[i];
        }
    }
    f64::NAN
}
// ────────────────────────── 读数据 ──────────────────────────

#[derive(Clone, Copy)]
struct Row {
    ns: i64,
    price: f64,
    volume: f64,
    turnover: f64,
    flag: i32,
    ask: i64,
    bid: i64,
}

/// 去掉字段两端的空白（含行尾的 `\r`）。
fn trim_ascii(raw: &[u8]) -> &[u8] {
    let mut s = 0usize;
    let mut e = raw.len();
    while s < e && raw[s].is_ascii_whitespace() {
        s += 1;
    }
    while e > s && raw[e - 1].is_ascii_whitespace() {
        e -= 1;
    }
    &raw[s..e]
}

/// 取字段的整数前缀（`"32.0"` 给 32）。
fn int_prefix(raw: &[u8]) -> i64 {
    let b = trim_ascii(raw);
    let mut i = 0usize;
    let mut neg = false;
    if i < b.len() && (b[i] == b'-' || b[i] == b'+') {
        neg = b[i] == b'-';
        i += 1;
    }
    let mut v: i64 = 0;
    while i < b.len() && b[i].is_ascii_digit() {
        v = v * 10 + (b[i] - b'0') as i64;
        i += 1;
    }
    if neg {
        -v
    } else {
        v
    }
}

fn pow10_table() -> &'static [f64; 309] {
    static T: std::sync::OnceLock<[f64; 309]> = std::sync::OnceLock::new();
    T.get_or_init(|| {
        let mut t = [1.0f64; 309];
        for (k, v) in t.iter_mut().enumerate() {
            *v = format!("1e{}", k).parse::<f64>().unwrap();
        }
        t
    })
}

/// pandas C 解析器 `precise_xstrtod` 的逐行复刻（与 hm033/hm090 里那份一致）。
fn parse_f64_pandas(b: &[u8]) -> f64 {
    const MAX_DIGITS: usize = 17;
    let n = b.len();
    let mut p = 0usize;
    while p < n && matches!(b[p], b' ' | b'\t' | b'\r' | b'\n') {
        p += 1;
    }
    let mut negative = false;
    if p < n && (b[p] == b'-' || b[p] == b'+') {
        negative = b[p] == b'-';
        p += 1;
    }
    let mut number = 0.0f64;
    let mut exponent: i32 = 0;
    let mut num_digits = 0usize;
    let mut num_decimals = 0usize;
    while p < n && b[p].is_ascii_digit() {
        if num_digits < MAX_DIGITS {
            number = number * 10.0 + (b[p] - b'0') as f64;
            num_digits += 1;
        } else {
            exponent += 1;
        }
        p += 1;
    }
    if p < n && b[p] == b'.' {
        p += 1;
        while num_digits < MAX_DIGITS && p < n && b[p].is_ascii_digit() {
            number = number * 10.0 + (b[p] - b'0') as f64;
            p += 1;
            num_digits += 1;
            num_decimals += 1;
        }
        if num_digits >= MAX_DIGITS {
            while p < n && b[p].is_ascii_digit() {
                p += 1;
            }
        }
        exponent -= num_decimals as i32;
    }
    if num_digits == 0 {
        return 0.0;
    }
    if negative {
        number = -number;
    }
    if p < n && (b[p] | 0x20) == b'e' {
        p += 1;
        let mut eneg = false;
        if p < n && (b[p] == b'-' || b[p] == b'+') {
            eneg = b[p] == b'-';
            p += 1;
        }
        let mut nn: i32 = 0;
        let mut nd = 0usize;
        while nd < MAX_DIGITS && p < n && b[p].is_ascii_digit() {
            nn = nn * 10 + (b[p] - b'0') as i32;
            nd += 1;
            p += 1;
        }
        if eneg {
            exponent -= nn;
        } else {
            exponent += nn;
        }
    }
    if exponent > 308 {
        f64::INFINITY
    } else if exponent > 0 {
        number * pow10_table()[exponent as usize]
    } else if exponent < -308 {
        if exponent < -616 {
            0.0
        } else {
            let t = pow10_table();
            (number / t[(-308 - exponent) as usize]) / t[308]
        }
    } else {
        number / pow10_table()[(-exponent) as usize]
    }
}

/// pandas conversion.pyx 的 cast_from_unit_vectorized（unit='s' → 'ns'）：
///   base = (int64)val; frac = val - base; frac = np.round(frac, 9);
///   out = base*1e9 + (int64)(frac*1e9)
/// 不是「微秒 × 1000」，这条链自带 ±120ns 噪声。
fn pandas_time_ns(exchtime_us: f64) -> i64 {
    let val = exchtime_us / 1e6;
    let base = val as i64;
    let frac = val - base as f64;
    let y = (frac * 1e9).round_ties_even() / 1e9;
    base.wrapping_mul(1_000_000_000)
        .wrapping_add((y * 1e9) as i64)
}

/// 读一个交易日的逐笔，复刻 dwo.read_trade + dwo.adjust_afternoon。
///
/// 行序 = 上午全部（文件序）在前、下午全部（文件序）在后（下午整体减 90 分钟）。
fn read_day(code: &str, date: i64) -> Option<Vec<Row>> {
    const COL_EXCHTIME: usize = 4;
    const COL_PRICE: usize = 7;
    const COL_VOLUME: usize = 8;
    const COL_TURNOVER: usize = 9;
    const COL_FLAG: usize = 10;
    const COL_ASK_ORDER: usize = 13;
    const COL_BID_ORDER: usize = 14;

    let filename = format!("{}_{}_transaction.csv", code, date);
    let mut data = None;
    for root in data_paths::level2_roots() {
        let path = root
            .join(date.to_string())
            .join("transaction")
            .join(&filename);
        if let Ok(bytes) = std::fs::read(&path) {
            data = Some(bytes);
            break;
        }
    }
    let data = data?;

    let mut morning: Vec<Row> = Vec::new();
    let mut afternoon: Vec<Row> = Vec::new();
    for (lineno, line) in data.split(|&b| b == b'\n').enumerate() {
        if lineno == 0 || line.is_empty() {
            continue;
        }
        let mut fields: [&[u8]; 15] = [&[][..]; 15];
        let mut commas = 0usize;
        let mut start = 0usize;
        for (i, &b) in line.iter().enumerate() {
            if b == b',' {
                if commas < 15 {
                    fields[commas] = &line[start..i];
                }
                commas += 1;
                start = i + 1;
            }
        }
        if commas < 15 {
            fields[commas] = &line[start..];
        }
        if commas + 1 < 15 {
            continue;
        }
        let exchtime_us = parse_f64_pandas(trim_ascii(fields[COL_EXCHTIME]));
        if exchtime_us == 0.0 {
            continue;
        }
        let flag = int_prefix(fields[COL_FLAG]) as i32;
        if flag == 32 {
            continue;
        }
        let ns = pandas_time_ns(exchtime_us).wrapping_add(OFFSET_NS);
        let tod = ns.rem_euclid(DAY_NS);
        let mut row = Row {
            ns,
            price: parse_f64_pandas(trim_ascii(fields[COL_PRICE])),
            volume: parse_f64_pandas(trim_ascii(fields[COL_VOLUME])),
            turnover: parse_f64_pandas(trim_ascii(fields[COL_TURNOVER])),
            flag,
            ask: int_prefix(fields[COL_ASK_ORDER]),
            bid: int_prefix(fields[COL_BID_ORDER]),
        };
        if (MORNING_START_NS..=MORNING_END_NS).contains(&tod) {
            morning.push(row);
        } else if (AFTERNOON_START_NS..=AFTERNOON_END_NS).contains(&tod) {
            row.ns -= SHIFT_NS;
            afternoon.push(row);
        }
    }
    morning.append(&mut afternoon);
    if morning.is_empty() {
        return None;
    }
    Some(morning)
}
// ────────────────────────── 排序后的当日数据 ──────────────────────────

/// hm077 的 v2 通路只用 ns / price / volume / ask / bid；turnover 与 flag 保留下来是为了
/// 与 hm076 的 Day 保持同一形状（读盘阶段两者本就一样），标 dead_code 免得刷警告。
#[allow(dead_code)]
struct Day {
    ns: Vec<i64>,
    price: Vec<f64>,
    volume: Vec<f64>,
    turnover: Vec<f64>,
    flag: Vec<i32>,
    ask: Vec<i64>,
    bid: Vec<i64>,
}

fn build_day(rows: Vec<Row>) -> Day {
    let n = rows.len();
    let mut order: Vec<usize> = (0..n).collect();
    // np.lexsort((exchtime, volume))：主键 volume、次键 exchtime，稳定。
    order.sort_by(|&i, &j| {
        rows[i]
            .volume
            .partial_cmp(&rows[j].volume)
            .unwrap()
            .then(rows[i].ns.cmp(&rows[j].ns))
    });

    let mut d = Day {
        ns: Vec::with_capacity(n),
        price: Vec::with_capacity(n),
        volume: Vec::with_capacity(n),
        turnover: Vec::with_capacity(n),
        flag: Vec::with_capacity(n),
        ask: Vec::with_capacity(n),
        bid: Vec::with_capacity(n),
    };
    for &i in order.iter() {
        d.ns.push(rows[i].ns);
        d.price.push(rows[i].price);
        d.volume.push(rows[i].volume);
        d.turnover.push(rows[i].turnover);
        d.flag.push(rows[i].flag);
        d.ask.push(rows[i].ask);
        d.bid.push(rows[i].bid);
    }
    d
}
// ────────── rp.calculate_order_time_gap_and_price_percentile_ultra_sorted_v2 ──────────

/// v2 的订单分组：按「下单方向」分买卖两侧（不再用成交 flag）。
struct OrderVolumeGroupV2 {
    indices: Vec<usize>,
    times: Vec<f64>,
    prices: Vec<f64>,
    /// true = 买单（bid_order 那一支）
    order_types: Vec<bool>,
    ask_indices: Vec<usize>,
    bid_indices: Vec<usize>,
}

impl OrderVolumeGroupV2 {
    fn new() -> Self {
        Self {
            indices: Vec::new(),
            times: Vec::new(),
            prices: Vec::new(),
            order_types: Vec::new(),
            ask_indices: Vec::new(),
            bid_indices: Vec::new(),
        }
    }

    fn add_record(&mut self, orig_idx: usize, time: f64, price: f64, is_bid: bool) {
        let group_idx = self.indices.len();
        self.indices.push(orig_idx);
        self.times.push(time);
        self.prices.push(price);
        self.order_types.push(is_bid);
        if is_bid {
            self.bid_indices.push(group_idx);
        } else {
            self.ask_indices.push(group_idx);
        }
    }

    /// 基于订单类型的时间距离计算（照抄 v2 源码）。
    fn find_nearest_records_ultra_fast_v2(
        &self,
        current_group_idx: usize,
        target_indices: &[usize],
        max_records: usize,
    ) -> Vec<(f64, f64)> {
        if target_indices.is_empty() {
            return Vec::new();
        }
        let current_time = self.times[current_group_idx];
        let mut time_distances: Vec<(f64, f64)> = Vec::new();
        for &target_idx in target_indices.iter() {
            if target_idx != current_group_idx {
                let time_diff = (current_time - self.times[target_idx]).abs();
                let price = self.prices[target_idx];
                time_distances.push((time_diff, price));
            }
        }
        time_distances.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        if time_distances.len() > max_records {
            time_distances.truncate(max_records);
        }
        time_distances
    }

    /// 基于订单类型的批量计算指标（照抄 v2 源码）。
    fn compute_all_indicators_ultra_fast_v2(
        &self,
        results: &mut [f64],
        min_count: usize,
        use_flag: &str,
    ) {
        let group_size = self.indices.len();
        if group_size < min_count {
            return;
        }
        for current_group_idx in 0..group_size {
            let current_is_bid = self.order_types[current_group_idx];
            let target_indices: Vec<usize> = match use_flag {
                "same" => {
                    if current_is_bid {
                        self.bid_indices
                            .iter()
                            .filter(|&&idx| idx != current_group_idx)
                            .cloned()
                            .collect()
                    } else {
                        self.ask_indices
                            .iter()
                            .filter(|&&idx| idx != current_group_idx)
                            .cloned()
                            .collect()
                    }
                }
                "diff" => {
                    if current_is_bid {
                        self.ask_indices.clone()
                    } else {
                        self.bid_indices.clone()
                    }
                }
                _ => (0..group_size)
                    .filter(|&idx| idx != current_group_idx)
                    .collect(),
            };
            if target_indices.len() < min_count {
                continue;
            }
            let time_distances = self.find_nearest_records_ultra_fast_v2(
                current_group_idx,
                &target_indices,
                target_indices.len(),
            );
            if time_distances.len() >= min_count {
                let orig_idx = self.indices[current_group_idx];
                let current_price = self.prices[current_group_idx];
                calculate_order_indicators_ultra_fast(
                    &mut results[orig_idx * 22..orig_idx * 22 + 22],
                    &time_distances,
                    current_price,
                );
            }
        }
    }
}
/// 超高速指标计算（照抄 src/order_records_ultra_sorted.rs）。
fn calculate_order_indicators_ultra_fast(
    result_row: &mut [f64],
    time_distances: &[(f64, f64)],
    current_price: f64,
) {
    let total_count = time_distances.len();
    if total_count == 0 {
        return;
    }
    result_row[0] = time_distances[0].0;

    let percentages = [0.01, 0.02, 0.03, 0.04, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50];
    let mut cumulative_sum = 0.0;
    let mut last_count = 0;
    for (idx, &pct) in percentages.iter().enumerate() {
        let count = ((total_count as f64 * pct).ceil() as usize)
            .max(1)
            .min(total_count);
        for j in last_count..count {
            cumulative_sum += time_distances[j].0;
        }
        result_row[idx + 1] = cumulative_sum / count as f64;
        last_count = count;
    }
    for j in last_count..total_count {
        cumulative_sum += time_distances[j].0;
    }
    result_row[11] = cumulative_sum / total_count as f64;

    let price_percentages = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00];
    let mut all_prices: Vec<f64> = Vec::with_capacity(total_count);
    for &(_, price) in time_distances.iter() {
        all_prices.push(price);
    }
    all_prices.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    for (idx, &pct) in price_percentages.iter().enumerate() {
        let count = if pct == 1.0 {
            total_count
        } else {
            ((total_count as f64 * pct).ceil() as usize)
                .max(1)
                .min(total_count)
        };
        let prices_slice = &all_prices[..count];
        let rank = match prices_slice.binary_search_by(|&p| p.partial_cmp(&current_price).unwrap()) {
            Ok(pos) => pos,
            Err(pos) => pos,
        };
        result_row[idx + 12] = rank as f64 / prices_slice.len() as f64;
    }
}

fn find_volume_ranges(volumes: &[f64]) -> Vec<(f64, usize, usize)> {
    if volumes.is_empty() {
        return Vec::new();
    }
    let mut ranges = Vec::new();
    let mut current_volume = volumes[0];
    let mut start_idx = 0;
    for i in 1..volumes.len() {
        if volumes[i] != current_volume {
            ranges.push((current_volume, start_idx, i));
            current_volume = volumes[i];
            start_idx = i;
        }
    }
    ranges.push((current_volume, start_idx, volumes.len()));
    ranges
}
/// 平铺的 (n × 27) 矩阵，等价于 rp.calculate_order_time_gap_and_price_percentile_ultra_sorted_v2。
/// exchtime 传进来是纳秒 i64（df.exchtime.to_numpy(float) 的值），内部 `/ 1e9` 变秒。
fn ultra_sorted_v2(
    volume: &[f64],
    exchtime_ns: &[i64],
    price: &[f64],
    ask_order: &[i64],
    bid_order: &[i64],
    min_count: usize,
    use_flag: &str,
) -> Vec<f64> {
    let n = volume.len();
    let exchtime_data: Vec<f64> = exchtime_ns.iter().map(|&t| t as f64 / 1e9).collect();

    // 1) 订单聚合：ask 分支在前、bid 分支在后，各写各的；键只有 order_id。
    let mut orders: Vec<(i64, bool, f64, f64, f64)> = Vec::new(); // (order_id, is_bid, vol, price, time)
    let mut order_map: std::collections::HashMap<i64, usize> = std::collections::HashMap::new();
    for i in 0..n {
        if ask_order[i] != 0 {
            let order_id = ask_order[i];
            if let Some(&oi) = order_map.get(&order_id) {
                let old_vol = orders[oi].2;
                let old_price = orders[oi].3;
                let new_vol = old_vol + volume[i];
                orders[oi].2 = new_vol;
                orders[oi].3 = (old_vol * old_price + volume[i] * price[i]) / new_vol;
                orders[oi].4 = orders[oi].4.max(exchtime_data[i]);
            } else {
                let oi = orders.len();
                orders.push((order_id, false, volume[i], price[i], exchtime_data[i]));
                order_map.insert(order_id, oi);
            }
        }
        if bid_order[i] != 0 {
            let order_id = bid_order[i];
            if let Some(&oi) = order_map.get(&order_id) {
                let old_vol = orders[oi].2;
                let old_price = orders[oi].3;
                let new_vol = old_vol + volume[i];
                orders[oi].2 = new_vol;
                orders[oi].3 = (old_vol * old_price + volume[i] * price[i]) / new_vol;
                orders[oi].4 = orders[oi].4.max(exchtime_data[i]);
            } else {
                let oi = orders.len();
                orders.push((order_id, true, volume[i], price[i], exchtime_data[i]));
                order_map.insert(order_id, oi);
            }
        }
    }

    // 2) 按订单 volume、再按订单时间排序（不稳定排序，与源码一致）
    orders.sort_unstable_by(|a, b| {
        a.2.partial_cmp(&b.2)
            .unwrap()
            .then(a.4.partial_cmp(&b.4).unwrap())
    });

    // 3) 按订单 volume 分组
    let order_volumes: Vec<f64> = orders.iter().map(|o| o.2).collect();
    let order_ranges = find_volume_ranges(&order_volumes);

    let mut order_groups: Vec<OrderVolumeGroupV2> = Vec::new();
    for (_, start_idx, end_idx) in order_ranges.iter() {
        let mut group = OrderVolumeGroupV2::new();
        for i in *start_idx..*end_idx {
            let (_, is_bid, _, p, time) = orders[i];
            group.add_record(i, time, p, is_bid);
        }
        order_groups.push(group);
    }

    // 4) 逐组算 22 个指标
    let mut order_results = vec![f64::NAN; orders.len() * 22];
    for group in order_groups.iter() {
        group.compute_all_indicators_ultra_fast_v2(&mut order_results, min_count, use_flag);
    }

    // 5) 映射回逐笔行；ask 分支先写、bid 分支后写（后者覆盖前者，原样保留）
    let mut results = vec![f64::NAN; n * 27];
    for i in 0..n {
        if ask_order[i] != 0 {
            if let Some(&oi) = order_map.get(&ask_order[i]) {
                for j in 0..22 {
                    results[i * 27 + j] = order_results[oi * 22 + j];
                }
                results[i * 27 + 22] = ask_order[i] as f64;
                results[i * 27 + 23] = 0.0;
                results[i * 27 + 24] = orders[oi].2;
                results[i * 27 + 25] = orders[oi].4;
                results[i * 27 + 26] = orders[oi].3;
            }
        }
        if bid_order[i] != 0 {
            if let Some(&oi) = order_map.get(&bid_order[i]) {
                for j in 0..22 {
                    results[i * 27 + j] = order_results[oi * 22 + j];
                }
                results[i * 27 + 22] = bid_order[i] as f64;
                results[i * 27 + 23] = 1.0;
                results[i * 27 + 24] = orders[oi].2;
                results[i * 27 + 25] = orders[oi].4;
                results[i * 27 + 26] = orders[oi].3;
            }
        }
    }
    results
}
// ────────────────────────── one(direction) ──────────────────────────

/// pandas groupby(...).agg({k:'mean'}) 的 group_mean：Kahan 补偿求和 / 计数。
fn group_means(labels: &[i64], n_groups: usize, cols: &[Vec<f64>]) -> Vec<f64> {
    let k = cols.len();
    let mut sumx = vec![0.0f64; n_groups * k];
    let mut comp = vec![0.0f64; n_groups * k];
    let mut nobs = vec![0i64; n_groups * k];
    for (i, &lab) in labels.iter().enumerate() {
        if lab < 0 {
            continue;
        }
        let g = lab as usize;
        for j in 0..k {
            let val = cols[j][i];
            if val.is_nan() {
                continue;
            }
            nobs[g * k + j] += 1;
            let y = val - comp[g * k + j];
            let t = sumx[g * k + j] + y;
            let mut c = t - sumx[g * k + j] - y;
            if c.is_nan() {
                c = 0.0;
            }
            comp[g * k + j] = c;
            sumx[g * k + j] = t;
        }
    }
    let mut out = vec![f64::NAN; n_groups * k];
    for g in 0..n_groups {
        for j in 0..k {
            let c = nobs[g * k + j];
            if c != 0 {
                out[g * k + j] = sumx[g * k + j] / c as f64;
            }
        }
    }
    out
}

/// 一边（66 或 83）的 24 列 + 4 个相关目标 + groupby 键。
struct Side {
    m: usize,
    m24: Vec<f64>,
    t_price: Vec<f64>,
    t_vol: Vec<f64>,
    t_id: Vec<f64>,
    t_time: Vec<f64>,
    /// groupby 之后的 (G × 29) 矩阵（29 列 = 24 指标 + 5 常量）
    gm: Vec<f64>,
    g: usize,
}

/// 把 27 列展开成 24 列（13 时间间隔 + trend，11 价格分位 + trend）。
fn expand24(rows27: &[f64]) -> Vec<f64> {
    let mut out = Vec::with_capacity(24);
    out.extend_from_slice(&rows27[0..12]);
    out.push(trend_1d(&rows27[0..12]));
    out.extend_from_slice(&rows27[12..22]);
    out.push(trend_1d(&rows27[12..22]));
    out
}

impl Side {
    fn build(flat: &[f64], idx: &[usize]) -> Side {
        let m = idx.len();
        let mut m24 = vec![0.0f64; m * 24];
        for (k, &r) in idx.iter().enumerate() {
            let row = expand24(&flat[r * 27..r * 27 + 27]);
            m24[k * 24..k * 24 + 24].copy_from_slice(&row);
        }
        let t_price: Vec<f64> = idx.iter().map(|&r| flat[r * 27 + 26]).collect();
        let t_vol: Vec<f64> = idx.iter().map(|&r| flat[r * 27 + 24]).collect();
        let t_id: Vec<f64> = idx.iter().map(|&r| flat[r * 27 + 22]).collect();
        let t_time: Vec<f64> = idx.iter().map(|&r| flat[r * 27 + 25]).collect();
        let key: Vec<f64> = idx.iter().map(|&r| flat[r * 27 + C_ORDER_VOLUME]).collect();

        let mut uniq: Vec<f64> = key.iter().cloned().filter(|v| !v.is_nan()).collect();
        uniq.sort_by(|a, b| a.partial_cmp(b).unwrap());
        uniq.dedup();
        let g = uniq.len();
        let labels: Vec<i64> = key
            .iter()
            .map(|v| match uniq.binary_search_by(|x| x.partial_cmp(v).unwrap()) {
                Ok(p) => p as i64,
                Err(_) => -1,
            })
            .collect();
        let mut cols29: Vec<Vec<f64>> = Vec::with_capacity(29);
        for j in 0..24 {
            cols29.push((0..m).map(|k| m24[k * 24 + j]).collect());
        }
        // 5 个常量列：订单编号 / 买单标识 / 订单总量 / 订单时间 / 订单价格
        for &tc in &[22usize, 23, 24, 25, 26] {
            cols29.push(idx.iter().map(|&r| flat[r * 27 + tc]).collect());
        }
        let gm = if g > 0 {
            group_means(&labels, g, &cols29)
        } else {
            Vec::new()
        };
        Side {
            m,
            m24,
            t_price,
            t_vol,
            t_id,
            t_time,
            gm,
            g,
        }
    }

    fn col(&self, j: usize) -> Vec<f64> {
        (0..self.m).map(|k| self.m24[k * 24 + j]).collect()
    }

    fn tcol(&self, j: usize) -> Vec<f64> {
        (0..self.g).map(|r| self.gm[r * 29 + j]).collect()
    }

    fn t24col(&self, j: usize) -> Vec<f64> {
        (0..self.g).map(|r| self.gm[r * 29 + j]).collect()
    }

    /// 4 个 corr 块（订单价格 / 订单总量 / 订单编号 / 订单时间）。
    fn corr4(&self) -> Vec<f64> {
        let mut out = Vec::with_capacity(4 * 24);
        for target in [&self.t_price, &self.t_vol, &self.t_id, &self.t_time] {
            for j in 0..24 {
                out.push(series_corr(&self.col(j), target));
            }
        }
        out
    }

    fn tcorr4(&self) -> Vec<f64> {
        let mut out = Vec::with_capacity(4 * 24);
        for j_target in [28usize, 26, 24, 27] {
            let target = self.tcol(j_target);
            for j in 0..24 {
                out.push(series_corr(&self.t24col(j), &target));
            }
        }
        out
    }

    /// mean/std/skew/kurt（4×24），走 features_f64。
    fn stat4(&self) -> Vec<f64> {
        let (vals, names) = self.feats24();
        let cols: Vec<String> = N24.iter().map(|s| s.to_string()).collect();
        let mut out = Vec::with_capacity(4 * 24);
        for suffix in ["mean", "std", "skew", "kurt"] {
            for c in cols.iter() {
                out.push(pick(&vals, &names, &format!("{}_{}", c, suffix)));
            }
        }
        out
    }

    fn tstat4(&self) -> Vec<f64> {
        let mut t24 = vec![0.0f64; self.g * 24];
        for r in 0..self.g {
            t24[r * 24..r * 24 + 24].copy_from_slice(&self.gm[r * 29..r * 29 + 24]);
        }
        let cols: Vec<String> = N24.iter().map(|s| s.to_string()).collect();
        let (vals, names) = if self.g > 0 {
            let arr = Array2::from_shape_vec((self.g, 24), t24).ok().unwrap();
            features_f64::get_features_factors_rust_full(&arr.view(), &cols, false)
        } else {
            (Vec::new(), Vec::new())
        };
        let mut out = Vec::with_capacity(4 * 24);
        for suffix in ["mean", "std", "skew", "kurt"] {
            for c in cols.iter() {
                out.push(pick(&vals, &names, &format!("{}_{}", c, suffix)));
            }
        }
        out
    }

    fn feats24(&self) -> (Vec<f64>, Vec<String>) {
        let cols: Vec<String> = N24.iter().map(|s| s.to_string()).collect();
        if self.m == 0 {
            return (Vec::new(), Vec::new());
        }
        let arr = Array2::from_shape_vec((self.m, 24), self.m24.clone())
            .ok()
            .unwrap();
        features_f64::get_features_factors_rust_full(&arr.view(), &cols, false)
    }

    /// DataFrame.corrwith：前 13 列有值，后 11 列 NaN。
    fn autocorr24(&self) -> Vec<f64> {
        let mut out = vec![f64::NAN; 24];
        for j in 0..13 {
            let col = self.col(j);
            let mut sh = vec![f64::NAN; self.m];
            for k in 1..self.m {
                sh[k] = col[k - 1];
            }
            out[j] = df_corrwith(&col, &sh);
        }
        out
    }

    fn tautocorr24(&self) -> Vec<f64> {
        let mut out = vec![f64::NAN; 24];
        for j in 0..13 {
            let col = self.t24col(j);
            let mut sh = vec![f64::NAN; self.g];
            for k in 1..self.g {
                sh[k] = col[k - 1];
            }
            out[j] = df_corrwith(&col, &sh);
        }
        out
    }

    fn trend24(&self) -> Vec<f64> {
        (0..24).map(|j| trend_1d(&self.col(j))).collect()
    }

    fn ttrend24(&self) -> Vec<f64> {
        (0..24).map(|j| trend_1d(&self.t24col(j))).collect()
    }
}
fn one(d: &Day, direction: &str) -> Vec<f64> {
    let n = d.volume.len();
    let flat = ultra_sorted_v2(&d.volume, &d.ns, &d.price, &d.ask, &d.bid, 100, direction);

    // features.dropna()：27 列任意一列是 NaN 就把这一行丢掉
    let mut idx66: Vec<usize> = Vec::new();
    let mut idx83: Vec<usize> = Vec::new();
    for r in 0..n {
        let mut ok = true;
        for c in 0..27 {
            if flat[r * 27 + c].is_nan() {
                ok = false;
                break;
            }
        }
        if !ok {
            continue;
        }
        if flat[r * 27 + C_BUY_FLAG] == 1.0 {
            idx66.push(r);
        } else {
            idx83.push(r);
        }
    }

    let s66 = Side::build(&flat, &idx66);
    let s83 = Side::build(&flat, &idx83);

    let mut res: Vec<f64> = Vec::with_capacity(ONE_LEN);
    res.extend(s66.corr4());
    res.extend(s83.corr4());
    res.extend(s66.stat4());
    res.extend(s66.autocorr24());
    res.extend(s66.trend24());
    res.extend(s83.stat4());
    res.extend(s83.autocorr24());
    res.extend(s83.trend24());
    res.extend(s66.tcorr4());
    res.extend(s83.tcorr4());
    res.extend(s66.tstat4());
    res.extend(s66.tautocorr24());
    res.extend(s66.ttrend24());
    res.extend(s83.tstat4());
    res.extend(s83.tautocorr24());
    res.extend(s83.ttrend24());
    res
}

// ────────────────────────── 入口 ──────────────────────────

pub fn compute_stock(code: &str, date: i64) -> Option<Vec<f32>> {
    let rows = read_day(code, date)?;
    let day = build_day(rows);
    let res1 = one(&day, "same");
    let res2 = one(&day, "diff");
    let mut full = res1;
    full.extend(res2);
    if full.len() != FULL_LEN {
        // 长度不对说明中间某一块算空了，宁可整条给 NaN 也不要错位取值
        return None;
    }
    Some(IDX.iter().map(|&i| full[i] as f32).collect())
}

pub fn compute_cross(_date: i64) -> Option<(Vec<String>, Vec<f32>)> {
    None
}
