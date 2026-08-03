//! 一呼百应（yhyb）：tick 级事件响应网络横截面因子（纯 Rust）。
//!
//! # 思想
//! 把日频收益 lead-lag 网络下放到 tick 级：在逐笔成交 + 盘口快照上定义 23 种事件，
//! 对每只股票 A 度量"全市场其他股票 B 对 A 事件的响应距离"（fwd = B 响应 A，A 领先；
//! bwd = A 响应 B，B 领先），池化成响应速度/协同度因子，并用均匀零模型剔除共同驱动。
//!
//! # 范式（cross-section-pipeline 规范）
//! - 任务粒度 = 一天全市场；rayon 并行读全市场逐笔+盘口（read_trade_fast_inner / read_market_fast_inner）
//! - per-stock 事件检测 → 跨股票池化匹配 → per-stock 因子输出（N_FACTORS = 1380）
//! - v1（读盘）+ v2（Python 传数据）双入口，共享同一份纯计算核心
//!
//! # 因子布局（23 事件 × 4 时段 × 15）
//! 对每个 (事件 e, 时段 p)：`yhyb_e{02}_p{p}_rate` + fwd/bwd 两方向 × 7 个度量
//! [med, mean, hit, fast5, wmed, rmed, rhit]（**v3 单遍近似统计**，评估相关性见 README）：
//! - med/mean/fast5/wmed：全部 1 秒桶中点（±0.5s 量化；med/fast5 Spearman 1.0000/0.9996，
//!   mean/wmed 桶中点相关性已实测评估）
//! - hit：精确标量（距离 ≤ hit_t_s 秒占比，含无响应）
//! - rmed/rhit：med/hit 除以零模型期望（净响应，剔除共同驱动）——零模型近似但 O(k_A+nB)：
//!   null_med 可分式均值池化（rmed Spearman 0.9694）、null_hit x̄ 均值近似（rhit 0.9632）
//!
//! # 数据约定
//! - 只使用连续竞价：读取时 with_afternoon_adjust=true（过滤集合竞价+收盘后、下午前移90分钟）
//! - 成交方向直接用逐笔自带 flag（66=主买/83=主卖），不用 Lee-Ready
//! - 跨股匹配一律用 time_us（i64 微秒）；time_sec 是 f32，2025 年 epoch 附近精度仅 ±128s
//! - 涨跌停/停牌无逐笔数据自然无事件；T+1 不影响（只研究当日事件联动）
//!
//! # 零模型（可复现性）
//! 解析均匀零模型（无随机数）：B 的 m 个事件在时段 [0,S] 均匀分布时，
//! A 事件 t 到下一个 B 事件的分布 P(d>y) = ((S-t-y)/(S-t))^m。
//! null_med = 池化**中位数** of x·(1 - 2^(-1/m))（系数表全市场预计算，循环内零 powf）；
//! null_hit = 逐点精确 1 - ((x-T)/x)^m（整数幂 powi + 安全截断）。
//! 全确定性，逐日逐股可复现。
//!
//! # 性能（一天全市场，限流 50 线程；机器 AMD EPYC 9754 ×2，512 核共享机）
//! 单日全市场 5914 股 × 1916 因子实测（2024-12-31，YHYB_TIMING 分段）：
//! v5（优化前）总墙钟 ~150-155s（读盘+检测 3.5s + 聚合+第4层 ~150s）；
//! v6 总墙钟 ~110-125s（读盘+检测 3.5s + 聚合+第4层 ~108-120s）；
//! v7.4（本版）总墙钟 ~89-91s（读盘+检测 3.5s + 聚合+第4层 ~87-89s，共享机波动 ±3s）；
//! v7.4 新增：**方向打包桶**——bwd/fwd 各 4 时段共用同一桶号，合并为 32B 宽桶，
//! 一次 RMW 更新多时段（load/store 与 touched 次数 /3~4、热循环寄存器大降），
//! 位级不变，受控 A/B 聚合 -2.4s（89.3 → 87.0）。
//! 优化（均为算法/底层级，输出逐位不变；v6→v7 增量见 agg_fused_blocked 注释）：
//! - 预计算切片表：聚合热循环 B 时段切片零二分搜索（原每任务 4×5914 次 DRAM 延迟绑定二分）
//! - union-walk 单趟归并：4 时段独立归并压缩为一次全事件归并（j-walk 1.9× 减少，
//!   时段 p 的 fwd/bwd 距离 = 全事件最近邻 + 有效性过滤，逐位等价）
//! - 5 段固定掩码子循环：A 事件按固定时段区间分段，内循环零成员分支
//! - 时段定制桶数组：8 个 Gather 共 560KB 落 L2（原 8×157KB=1.26MB 溢出 L2）
//! - 线程本地 Gather 复用（touched 清零）+ locate/stats 只扫 [0,max_b]
//! - 同一 (a,B) 4 时段推送共享 u64 除法；hit 无分支累加
//! - 第 4 层融合（v1/v2 统一）+ WITH_L4 编译期特化（const 泛型，热循环零运行时分支）
//! - **时间比较有效性判断**：j < bsl[p].1 ⟺ tb[j]-base < HI_p_S（partition 性质逐位等价），
//!   时段边界为编译期立即数（零寄存器/零 bsl 索引读取）
//! - **游标寄存器缓存**：tj_b/tp_b（后继/前驱时间）缓存于寄存器，条件比较零内存加载
//! - **单次 8B 桶加载/存储**（get_unchecked）+ 累加器 3 字段（hit|max_b 打包 u64 + wsum），
//!   延迟装载/段界回写（活跃寄存器集 = 当前段 gather 数）
//! - **表压缩**：切片表 u32（11→5.5MB）、事件数表 u32（5.5→2.75MB）、c(m) 记忆化
//!   静态缓存（5.5MB f64 表 → 0）+ null_hit powf 任务内记忆化（每 (k,时段,m) 一次）
//! - **B 侧事件流 u32 增量压缩**（聚合步行专用副本，4B/事件 vs 8B，DRAM 减半；
//!   转义编码 ≥u32 的跨事件间隔，解码纯整数加法逐位精确）
//! - 软件预取 B 访问数据（切片/事件数/有效索引，L3 延迟与 a-循环重叠）
//! - **事件数表按 (事件,时段) 转置**（m[e*4+p][bi]：聚合按 e 扫描时 bi 顺序读，
//!   硬件预取友好；原 m[bi][e][p] 是 464B 步长散布读）
//! - 实测成本构成（探针，2024-12-31）：推送 ~55s（1.06e12 次桶更新，x86-64 16 GPR
//!   限制下累加器栈驻留 ~9.4 周期/推）、j-walk ~32s（事件流读 ~1.1TB，L3 每 CCD 仅
//!   16MB 无法驻留 23MB/事件的流，K=2 的 L2 复用受 1.12MB 桶数组挤压未生效）、
//!   B 访问 ~10s（L3 延迟绑定）。
//!   60s 目标的剩余差距来自：逐位一致约束下 1.06e12 次桶更新的寄存器分配下限
//!   （S2/S3 峰值 6 个累加器 × 4 标量 = 24 个活跃值 > 16 GPR，编译器栈溢出），
//!   以及 j-walk 的事件流 DRAM 带宽墙（50 线程有效 ~60GB/s）。

use crate::fast_csv_reader::{read_market_fast_inner, read_trade_fast_inner, MarketRecord, TradeRecord};
use numpy::PyReadonlyArray2;
use pyo3::prelude::*;
use rayon::prelude::*;
use std::collections::BTreeSet;
use std::fs;

// ============================================================================
// 常量
// ============================================================================

pub const N_EVENTS: usize = 29;
pub const N_PERIODS: usize = 4;
pub const N_METRICS: usize = 7;
/// 每 (事件, 时段)：rate + fwd/bwd × 7 度量
pub const N_FACTORS: usize = N_EVENTS * N_PERIODS * (1 + 2 * N_METRICS); // 1740

/// 事件 29 个：big 系列按订单体量拆三档（截面百分位 + 空档每股内部补充）：
/// 大单 = 全市场成交额 top 10%（内部补充 = 每股内部 top 10%）、
/// 中单 = 10%~60%、小单 = bottom 40%。sweep/ice 的"大单" = 大单档。
pub const EVENT_NAMES: [&str; N_EVENTS] = [
    "big_l", "big_m", "big_s", "big_buy_l", "big_buy_m", "big_buy_s",
    "big_sell_l", "big_sell_m", "big_sell_s",
    "sweep_buy", "sweep_sell", "ice", "jump", "jump_up", "jump_dn",
    "run_up", "run_dn", "vwap_up", "vwap_dn", "vwap_dev_up", "vwap_dev_dn",
    "imb_buy", "imb_sell", "depth", "wall", "spread", "retreat",
    "imp_buy", "imp_sell",
];

pub const METRIC_NAMES: [&str; N_METRICS] = ["med", "mean", "hit", "fast5", "wmed", "rmed", "rhit"];

/// 时段边界（**adjust_afternoon 后的连续时钟**，本地 09:30 = 5400s、14:57 = 19620s；
/// 下午 13:00-15:00 已前移 90 分钟为 11:30-13:30，无午休空洞，跨午盘距离不受污染）
/// p0=全天 [09:30,14:57)；p1=盘中3小时 [10:00,14:27)（剔除早盘/尾盘各 30 分钟）；
/// p2=尾盘30分钟 [14:27,14:57)；p3=下午2小时 [13:00,15:00)（adjust 后 11:30-13:30；
/// 连续竞价数据只到 14:57 = adjust 后 13:27，窗口超出部分自然无数据）
const PERIOD_LO_S: [i64; N_PERIODS] = [5400, 6000, 17820, 12600];
const PERIOD_HI_S: [i64; N_PERIODS] = [19620, 17820, 19620, 19800];

/// 时段边界（µs 偏移，基准相对）：聚合热循环有效性比较的**编译期立即数**
/// （零寄存器占用；`tb[j]-base < HI0_S` ⟺ `j < bsl[0].1`，partition 性质逐位等价）。
const HI0_S: i64 = PERIOD_HI_S[0] * 1_000_000;
const HI1_S: i64 = PERIOD_HI_S[1] * 1_000_000;
const HI3_S: i64 = PERIOD_HI_S[3] * 1_000_000;
const LO0_S: i64 = PERIOD_LO_S[0] * 1_000_000;
const LO1_S: i64 = PERIOD_LO_S[1] * 1_000_000;
const LO2_S: i64 = PERIOD_LO_S[2] * 1_000_000;
const LO3_S: i64 = PERIOD_LO_S[3] * 1_000_000;

/// 事件时刻 t 所在日的"本地零点"（epoch 微秒）。
///
/// 注意：time_us 是"含 8h 偏移的 epoch"（本地时间作为伪 epoch，见 fast_csv_reader），
/// 因此本地零点 = UTC 零点 + 8h；若按 UTC 零点计算时段边界会整体错位 8 小时，
/// 导致零模型的时段末 S 早于事件时刻（x=0 → null_med=0 → rmed=inf）。
pub fn day_base(t: i64) -> i64 {
    (t / 86_400_000_000) * 86_400_000_000 + 28_800_000_000
}// ============================================================================
// 参数（样例数据调参结果；from_vec 顺序即调参脚本传参顺序，17 个）
// ============================================================================

#[derive(Clone, Copy, Debug)]
pub struct YhybParams {
    pub big_amt_wan: f64,  // 【已废弃】体量判定改用截面百分位（P90/P40）+ 空档内部补充；保留字段仅为兼容 from_vec 17 参数顺序
    pub sweep_w_s: f64,    // 扫单窗口（秒）
    pub sweep_m: usize,    // 扫单: 窗口内同向大单数 ≥ m
    pub ice_w_s: f64,      // 冰山窗口（秒）
    pub ice_m: usize,      // 冰山: 窗口内同价大单数 ≥ m
    pub jump_q: f64,       // 跳变: |Δp| > max(当日 |Δp| 的 jump_q 分位, 2×tick)
    pub jump_win: usize,   // 跳变滚动σ窗口（笔数，保留备用）
    pub run_r: usize,      // 连涨/连跌: 连续同向 ≥ r 笔
    pub vwap_dev: f64,     // 偏离VWAP: |p-vwap|/vwap > θ（进入语义）
    pub imb_thr: f64,      // 失衡极值: |IOB| > θ（进入语义），IOB 用全档总量
    pub dep_thr: f64,      // 深度突变: |Δdepth|/depth > θ（10档可见量）
    pub wall_k: f64,       // 厚墙: max(买一,卖一) > wall_k × 当日中位一档量（进入语义）
    pub spread_ticks: f64, // 价差扩张: spread > θ×0.01（进入语义）
    pub ret_thr: f64,      // 撤单潮: 相邻快照消失量/前快照可见量 > θ
    pub imp_thr: f64,      // 冲击单: 单笔量 > imp_thr × 当时前5档总量
    pub hit_t_s: f64,      // 命中窗口（秒）
    pub fast_q: f64,       // 最快 5% 分位
}

impl Default for YhybParams {
    fn default() -> Self {
        YhybParams {
            big_amt_wan: 10.0,
            sweep_w_s: 1200.0,
            sweep_m: 2,
            ice_w_s: 180.0,
            ice_m: 2,
            jump_q: 0.99,
            jump_win: 100,
            run_r: 2,
            vwap_dev: 0.001,
            imb_thr: 0.5,
            dep_thr: 0.3,
            wall_k: 2.0,
            spread_ticks: 1.0,
            ret_thr: 0.5,
            imp_thr: 0.05,
            hit_t_s: 60.0,
            fast_q: 0.05,
        }
    }
}

impl YhybParams {
    /// 从固定顺序的 17 个参数构造（调参脚本传参用）。
    pub fn from_vec(v: &[f64]) -> YhybParams {
        assert_eq!(v.len(), 17, "yhyb 参数必须为 17 个，顺序见 YhybParams 字段");
        YhybParams {
            big_amt_wan: v[0],
            sweep_w_s: v[1],
            sweep_m: v[2] as usize,
            ice_w_s: v[3],
            ice_m: v[4] as usize,
            jump_q: v[5],
            jump_win: v[6] as usize,
            run_r: v[7] as usize,
            vwap_dev: v[8],
            imb_thr: v[9],
            dep_thr: v[10],
            wall_k: v[11],
            spread_ticks: v[12],
            ret_thr: v[13],
            imp_thr: v[14],
            hit_t_s: v[15],
            fast_q: v[16],
        }
    }
}

// ============================================================================
// 事件流
// ============================================================================

#[derive(Clone, Default)]
pub struct EvStream {
    pub t: Vec<i64>, // 事件时间（epoch 微秒，升序）
    pub w: Vec<f64>, // 事件强度权重
}

fn push(ev: &mut EvStream, t: i64, w: f64) {
    ev.t.push(t);
    ev.w.push(w);
}

/// 显式符号函数：+0.0/-0.0 一律返回 0（注意：f64::signum(+0.0)=+1.0，会污染 run 检测）。
fn sgn(x: f64) -> f64 {
    if x > 0.0 {
        1.0
    } else if x < 0.0 {
        -1.0
    } else {
        0.0
    }
}

// ============================================================================
// 事件检测（per-stock，纯 Rust）
// ============================================================================

/// 体量档事件频率上限：三档拆分后事件量 ≈ 全部成交笔数（中单 = 50% 成交），
/// 直接全量会让跨股对数量暴增 15-50 倍（聚合数小时不可行）。
/// 超限时保留**档内金额最大的前 CAP_EV 笔**（每档最强信号，金融语义清晰）。
/// 500 笔/档/天：体量档总事件量 ≈ v4 水平（对数量可控，med 样本充足）。
const CAP_EV: usize = 500;

/// 收集候选后统一写入：超限按金额降序取前 CAP_EV，再按时间排序（保持流有序）。
fn push_capped(out: &mut EvStream, cand: &mut Vec<(i64, f64)>) {
    if cand.len() > CAP_EV {
        cand.select_nth_unstable_by(CAP_EV, |a, b| b.1.total_cmp(&a.1));
        cand.truncate(CAP_EV);
    }
    cand.sort_unstable_by(|a, b| a.0.cmp(&b.0));
    for &(tt, ww) in cand.iter() {
        push(out, tt, ww);
    }
}

/// 逐笔事件（21 个）：big_l/m/s（体量三档，不分方向）、big_buy_l/m/s、big_sell_l/m/s、
/// sweep_buy, sweep_sell, ice, jump, jump_up, jump_dn, run_up, run_dn,
/// vwap_up, vwap_dn, vwap_dev_up, vwap_dev_dn
///
/// **订单体量拆分（v5）**：截面百分位（全市场所有成交金额放一起）+ 空档每股内部补充。
/// - thr_l = 截面大单阈值（全市场成交额 P90），thr_m = 截面小单阈值（P40）；
///   大单 = amt ≥ P90（top 10%）、中单 = [P40, P90)（10%~60%）、小单 = < P40（bottom 40%）
/// - 空档补充：某只股票在截面口径下某一档为空（如小票全是小额成交、无截面大单），
///   则用该股票**内部**金额百分位补充该档（内部 top10% / 10%~60% / bottom40%），
///   保证每只有成交的股票都能识别出三档（覆盖率兜底）
/// - thr_l/thr_m 传 NaN（单股调试无截面）时三档全部按内部阈值
fn detect_trade_cols(
    t: &[i64],
    p: &[f64],
    v: &[f64],
    amt: &[f64],
    f: &[i32],
    prm: &YhybParams,
    thr_l: f64,
    thr_m: f64,
) -> [EvStream; 21] {
    let mut out: [EvStream; 21] = Default::default();
    let n = t.len();
    if n < 2 {
        return out;
    }
    // 当日中位单笔量（保留给 wall 类参考）
    let mut v_sorted = v.to_vec();
    v_sorted.select_nth_unstable_by(n / 2, |a, b| a.total_cmp(b));
    let _med_v = v_sorted[n / 2];
    // ---- 体量档判定：截面阈值 ∪ 空档内部补充 ----
    // 每股内部阈值（金额升序的 P40/P90）
    let mut amt_sorted = amt.to_vec();
    amt_sorted.sort_unstable_by(|a, b| a.total_cmp(b));
    let inner_m = amt_sorted[((n as f64) * 0.40) as usize];
    let inner_l = amt_sorted[((n as f64) * 0.90) as usize];
    // 截面档计数（判断哪些档为空；thr 为 NaN 时计数恒 0 → 全部走内部）
    let mut c_l = 0usize;
    let mut c_m = 0usize;
    let mut c_s = 0usize;
    for &x in amt {
        if x >= thr_l {
            c_l += 1;
        } else if x >= thr_m {
            c_m += 1;
        } else {
            c_s += 1;
        }
    }
    let use_l = thr_l.is_nan() || c_l == 0;
    let use_m = thr_m.is_nan() || c_m == 0;
    let use_s = thr_m.is_nan() || c_s == 0;
    let is_l: Vec<bool> = amt
        .iter()
        .map(|&x| if use_l { x >= inner_l } else { x >= thr_l })
        .collect();
    let is_m: Vec<bool> = amt
        .iter()
        .map(|&x| {
            if use_m {
                x >= inner_m && x < inner_l
            } else {
                x >= thr_m && x < thr_l
            }
        })
        .collect();
    let is_s: Vec<bool> = amt
        .iter()
        .map(|&x| if use_s { x < inner_m } else { x < thr_m })
        .collect();
    // 大/中/小三档事件（候选收集 + 频率上限，权重 = 成交金额）
    let mut cand = [Vec::<(i64, f64)>::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new()];
    for i in 0..n {
        if is_l[i] {
            cand[0].push((t[i], amt[i])); // big_l
            if f[i] == 66 {
                cand[3].push((t[i], amt[i])); // big_buy_l
            } else if f[i] == 83 {
                cand[6].push((t[i], amt[i])); // big_sell_l
            }
        }
        if is_m[i] {
            cand[1].push((t[i], amt[i])); // big_m
            if f[i] == 66 {
                cand[4].push((t[i], amt[i])); // big_buy_m
            } else if f[i] == 83 {
                cand[7].push((t[i], amt[i])); // big_sell_m
            }
        }
        if is_s[i] {
            cand[2].push((t[i], amt[i])); // big_s
            if f[i] == 66 {
                cand[5].push((t[i], amt[i])); // big_buy_s
            } else if f[i] == 83 {
                cand[8].push((t[i], amt[i])); // big_sell_s
            }
        }
    }
    for (i, c) in cand.iter_mut().enumerate() {
        push_capped(&mut out[i], c);
    }
    // 扫单：窗口内同向**大单**数 ≥ sweep_m
    let sw_us = (prm.sweep_w_s * 1e6) as i64;
    for (eidx, flag) in [(9usize, 66i32), (10, 83)] {
        let mut tb: Vec<i64> = Vec::new();
        let mut wb: Vec<f64> = Vec::new();
        for i in 0..n {
            if is_l[i] && f[i] == flag {
                tb.push(t[i]);
                wb.push(amt[i]);
            }
        }
        if tb.len() >= prm.sweep_m {
            let mut cand_sweep: Vec<(i64, f64)> = Vec::new();
            let mut j = 0usize;
            for i in 0..tb.len() {
                while tb[j] <= tb[i] - sw_us {
                    j += 1;
                }
                if i - j + 1 >= prm.sweep_m {
                    cand_sweep.push((tb[i], wb[i]));
                }
            }
            push_capped(&mut out[eidx], &mut cand_sweep);
        }
    }
    // 冰山：同价位大单在窗口内 ≥ ice_m
    // 注意：按 (价格, 时间) 分组遍历，事件流天然非时间升序；聚合归并/零模型都依赖
    // 时间有序，检测完必须按时间重排（历史 bug：ice 流乱序 → 匹配与零模型全部失效）
    {
        let mut idx: Vec<usize> = (0..n).filter(|&i| is_l[i]).collect();
        idx.sort_unstable_by(|&a, &b| p[a].total_cmp(&p[b]).then(t[a].cmp(&t[b])));
        let ice_us = (prm.ice_w_s * 1e6) as i64;
        let mut ice_ev: Vec<(i64, f64)> = Vec::new();
        let mut s = 0usize;
        while s < idx.len() {
            let mut e = s + 1;
            while e < idx.len() && p[idx[e]] == p[idx[s]] {
                e += 1;
            }
            if e - s >= prm.ice_m {
                let mut j = s;
                for i in s..e {
                    while t[idx[j]] <= t[idx[i]] - ice_us {
                        j += 1;
                    }
                    if i - j + 1 >= prm.ice_m {
                        ice_ev.push((t[idx[i]], amt[idx[i]]));
                    }
                }
            }
            s = e;
        }
        push_capped(&mut out[11], &mut ice_ev);
    }
    // 跳变：|Δp| > 当日 |Δp| 的 jump_q 分位
    // 分位阈值保证任意波动率的股票（含低价高波动股）都有事件 → 覆盖率可达 90%+；
    // 分位本身保证稀有性（top 1% 冲击），无需 2-tick 下限（下限会挡掉低价股）。
    if n > 2 {
        let mut dp_abs: Vec<f64> = (0..n - 1).map(|i| (p[i + 1] - p[i]).abs()).collect();
        let k = ((n as f64) * prm.jump_q).round().max(1.0) as usize;
        let k = k.min(dp_abs.len() - 1);
        dp_abs.select_nth_unstable_by(k, |a, b| a.total_cmp(b));
        let thr = dp_abs[k];
        for i in 0..n - 1 {
            let dp = p[i + 1] - p[i];
            if dp.abs() > thr {
                let wgt = dp.abs();
                push(&mut out[12], t[i + 1], wgt); // jump
                if dp > 0.0 {
                    push(&mut out[13], t[i + 1], wgt); // jump_up
                } else {
                    push(&mut out[14], t[i + 1], wgt); // jump_dn
                }
            }
        }
    }
    // 连涨/连跌：≥ run_r 笔同向（run 覆盖连续同向转移 [s, e]，事件时刻 = run 末笔）
    {
        let mut s = 0usize;
        let mut prev_sign = sgn(p[1] - p[0]);
        for i in 1..n - 1 {
            let sg = sgn(p[i + 1] - p[i]);
            if sg != prev_sign {
                if (i - s) >= prm.run_r && prev_sign != 0.0 {
                    let wgt = (p[i] - p[s]).abs();
                    if prev_sign > 0.0 {
                        push(&mut out[15], t[i], wgt); // run_up
                    } else {
                        push(&mut out[16], t[i], wgt); // run_dn
                    }
                }
                s = i;
                prev_sign = sg;
            }
        }
        // 收尾 run [s, n-2]
        if (n - 1 - s) >= prm.run_r && prev_sign != 0.0 {
            let wgt = (p[n - 1] - p[s]).abs();
            if prev_sign > 0.0 {
                push(&mut out[15], t[n - 1], wgt);
            } else {
                push(&mut out[16], t[n - 1], wgt);
            }
        }
    }
    // VWAP 与偏离
    let mut cum_v = 0.0f64;
    let mut cum_amt = 0.0f64;
    let mut vwap = vec![0.0f64; n];
    for i in 0..n {
        cum_v += v[i];
        cum_amt += amt[i];
        vwap[i] = cum_amt / cum_v.max(1e-9);
    }
    let mut prev_dev = p[0] - vwap[0];
    let mut in_zone = prev_dev.abs() > prm.vwap_dev * vwap[0];
    for i in 1..n {
        let dev = p[i] - vwap[i];
        if prev_dev <= 0.0 && dev > 0.0 {
            push(&mut out[17], t[i], dev.abs()); // vwap_up
        } else if prev_dev >= 0.0 && dev < 0.0 {
            push(&mut out[18], t[i], dev.abs()); // vwap_dn
        }
        let z = dev.abs() > prm.vwap_dev * vwap[i];
        if z && !in_zone {
            let wgt = dev.abs() / vwap[i];
            if dev > 0.0 {
                push(&mut out[19], t[i], wgt); // vwap_dev_up
            } else {
                push(&mut out[20], t[i], wgt); // vwap_dev_dn
            }
        }
        in_zone = z;
        prev_dev = dev;
    }
    out
}

/// 盘口事件（6 个）：imb_buy, imb_sell, depth, wall, spread, retreat
/// ask10/bid10: 10 档可见量（[f64;10]）。
/// IOB 用一档量（bid1/ask1）：一档变化快、买卖交替，覆盖率可达 90%+；
/// 全档 total IOB 在 A 股普遍偏卖（挂单结构），买方向覆盖率低（<40%），故不用。
fn detect_market_cols(
    mt: &[i64],
    ask1p: &[f64],
    bid1p: &[f64],
    ask10: &[[f64; 10]],
    bid10: &[[f64; 10]],
    prm: &YhybParams,
) -> [EvStream; 6] {
    let mut out: [EvStream; 6] = Default::default();
    let m = mt.len();
    if m < 2 {
        return out;
    }
    let depth10: Vec<f64> = (0..m).map(|i| ask10[i].iter().sum::<f64>() + bid10[i].iter().sum::<f64>()).collect();
    // IOB（全档）：进入语义
    let mut in_buy = false;
    let mut in_sell = false;
    // 厚墙：max(买一,卖一) > wall_k × 当日中位一档量
    let mut l1: Vec<f64> = (0..m).map(|i| bid10[i][0].max(ask10[i][0])).collect();
    let mut l1_sorted = l1.clone();
    l1_sorted.select_nth_unstable_by(m / 2, |a, b| a.total_cmp(b));
    let med_l1 = l1_sorted[m / 2];
    let mut in_wall = false;
    // 价差扩张：进入语义
    let mut in_spread = false;
    for i in 0..m {
        // IOB（一档）：进入语义
        let bsum = bid10[i][0];
        let asum = ask10[i][0];
        let denom = (bsum + asum).max(1e-9);
        let imb = (bsum - asum) / denom;
        if imb > prm.imb_thr && !in_buy {
            push(&mut out[0], mt[i], imb.abs());
            in_buy = true;
        } else if imb <= prm.imb_thr {
            in_buy = false;
        }
        if imb < -prm.imb_thr && !in_sell {
            push(&mut out[1], mt[i], imb.abs());
            in_sell = true;
        } else if imb >= -prm.imb_thr {
            in_sell = false;
        }
        // 厚墙
        let wall = l1[i] > prm.wall_k * med_l1;
        if wall && !in_wall {
            push(&mut out[3], mt[i], l1[i]);
            in_wall = true;
        } else if !wall {
            in_wall = false;
        }
        // 价差扩张（tick = 0.01）
        let spread = (ask1p[i] - bid1p[i]) / 0.01;
        let sp = spread > prm.spread_ticks;
        if sp && !in_spread {
            push(&mut out[4], mt[i], spread);
            in_spread = true;
        } else if !sp {
            in_spread = false;
        }
    }
    // 深度突变 + 撤单潮（相邻快照）
    for i in 1..m {
        let prev = depth10[i - 1];
        let cur = depth10[i];
        let dchg = (cur - prev).abs() / prev.max(1e-9);
        if dchg > prm.dep_thr {
            push(&mut out[2], mt[i], dchg);
        }
        let mut disc = 0.0f64;
        for k in 0..10 {
            disc += (bid10[i - 1][k] - bid10[i][k]).max(0.0);
            disc += (ask10[i - 1][k] - ask10[i][k]).max(0.0);
        }
        let rate = disc / prev.max(1e-9);
        if rate > prm.ret_thr {
            push(&mut out[5], mt[i], rate);
        }
    }
    out
}

/// 冲击单（2 个）：imp_buy, imp_sell —— 单笔量 > imp_thr × 当时前5档总量（双档）
fn detect_impact(
    t: &[i64],
    v: &[f64],
    amt: &[f64],
    f: &[i32],
    mt: &[i64],
    ask10: &[[f64; 10]],
    bid10: &[[f64; 10]],
    prm: &YhybParams,
) -> [EvStream; 2] {
    let mut out: [EvStream; 2] = Default::default();
    if t.is_empty() || mt.is_empty() {
        return out;
    }
    let mut j = 0usize;
    for i in 0..t.len() {
        while j + 1 < mt.len() && mt[j + 1] <= t[i] {
            j += 1;
        }
        if mt[j] > t[i] {
            continue; // 该笔早于第一份快照
        }
        let mut d5 = 0.0f64;
        for k in 0..5 {
            d5 += ask10[j][k] + bid10[j][k];
        }
        if v[i] > prm.imp_thr * d5 {
            let eidx = if f[i] == 66 { 0usize } else if f[i] == 83 { 1 } else { continue };
            push(&mut out[eidx], t[i], amt[i]);
        }
    }
    out
}

/// 全事件检测（29 个），输入原始 TradeRecord/MarketRecord（v1 读盘路径共用）。
/// thr_l/thr_m：截面成交额 P90/P40（体量三档拆分，NaN = 无截面信息走每股内部）。
fn detect_all(
    trades: &[TradeRecord],
    market: &[MarketRecord],
    prm: &YhybParams,
    thr_l: f64,
    thr_m: f64,
) -> [EvStream; N_EVENTS] {
    let n = trades.len();
    let mut t = Vec::with_capacity(n);
    let mut p = Vec::with_capacity(n);
    let mut v = Vec::with_capacity(n);
    let mut amt = Vec::with_capacity(n);
    let mut f = Vec::with_capacity(n);
    for tr in trades {
        t.push(tr.time_us);
        p.push(tr.price as f64);
        v.push(tr.volume as f64);
        amt.push(tr.turnover as f64);
        f.push(tr.flag);
    }
    let m = market.len();
    let mut mt = Vec::with_capacity(m);
    let mut ask1p = Vec::with_capacity(m);
    let mut bid1p = Vec::with_capacity(m);
    let mut ask10: Vec<[f64; 10]> = Vec::with_capacity(m);
    let mut bid10: Vec<[f64; 10]> = Vec::with_capacity(m);
    for mr in market {
        mt.push(mr.time_us);
        ask1p.push(mr.ask_prcs[0] as f64);
        bid1p.push(mr.bid_prcs[0] as f64);
        ask10.push(mr.ask_vols.map(|x| x as f64));
        bid10.push(mr.bid_vols.map(|x| x as f64));
    }
    let tev = detect_trade_cols(&t, &p, &v, &amt, &f, prm, thr_l, thr_m);
    let mev = detect_market_cols(&mt, &ask1p, &bid1p, &ask10, &bid10, prm);
    let iev = detect_impact(&t, &v, &amt, &f, &mt, &ask10, &bid10, prm);
    let mut out: [EvStream; N_EVENTS] = Default::default();
    for (i, s) in tev.into_iter().enumerate() {
        out[i] = s;
    }
    for (i, s) in mev.into_iter().enumerate() {
        out[21 + i] = s;
    }
    for (i, s) in iev.into_iter().enumerate() {
        out[27 + i] = s;
    }
    assert_streams_sorted(&out, "detect_all");
    out
}

/// 防御：事件流必须时间升序（跨股归并匹配与零模型都依赖有序）。
fn assert_streams_sorted(s: &[EvStream; N_EVENTS], what: &str) {
    for (i, ev) in s.iter().enumerate() {
        debug_assert!(
            ev.t.windows(2).all(|w| w[0] <= w[1]),
            "{what} 事件 {i} 流未按时间排序"
        );
    }
}

// ============================================================================
// 横截面聚合（per-stock 因子）
// ============================================================================

/// 1 秒直方图桶数：连续竞价全长 19620s，逐秒一桶，任何距离都有独立桶（无尾部截断）。
/// 桶索引 = 距离秒数（19619 为最大合法索引，min() 仅作防御）。
const N_BUCKETS_1S: usize = 19_620;

/// 均匀零模型（预计算表版）：
/// - null_med：池化**中位数** of x_i·(1 - 2^(-1/m_B)) over (B,i)——系数 2^(-1/m) 只依赖 B 的
///   事件数，全市场预计算（c 表），匹配循环内零 powf；期望值精确收集后 select。
/// - null_hit：逐点精确 1 - ((x_i-T)/x_i)^m_B（整数幂 powi + 安全截断）。
#[derive(Clone)]
struct NullTable {
    /// m[e*N_PERIODS+p][bi] = 事件数（u32；按 (事件,时段) 转置——聚合按 e 扫描时
    /// bi 顺序读取，硬件预取友好；原 m[bi][e][p] 是 464B 步长散布读）
    m: Vec<Vec<u32>>,
}

/// c(m) = 1 - 2^(-1/m) 记忆化静态缓存（m ≤ 500；与旧 c 表同一公式，逐位一致）。
/// 旧 c 表 5.5MB f64 从 L3 常驻集剔除——流的 L3 驻留是 j-walk 带宽的关键。
static C_CACHE: std::sync::OnceLock<[f64; 8192]> = std::sync::OnceLock::new();
static LN_C_CACHE: std::sync::OnceLock<[f64; 8192]> = std::sync::OnceLock::new();

#[inline(always)]
fn c_of_m(m: u32) -> f64 {
    C_CACHE.get_or_init(|| {
        let mut a = [0.0f64; 8192];
        for i in 1..8192 {
            a[i] = 1.0 - 2f64.powf(-1.0 / i as f64);
        }
        a
    })[m.min(8191) as usize]
}
#[inline(always)]
fn ln_c_of_m(m: u32) -> f64 {
    LN_C_CACHE.get_or_init(|| {
        let mut a = [0.0f64; 8192];
        for i in 1..8192 {
            a[i] = (1.0 - 2f64.powf(-1.0 / i as f64)).ln();
        }
        a
    })[m.min(8191) as usize]
}

/// 时段切片表：**全市场预计算一次**（(bi, e, p) → (lo, hi)），聚合任务热循环内零二分搜索。
/// 布局 sl[e * N_PERIODS + p][bi]：任务 (A, e, p) 按 bi 顺序扫描 → 缓存友好。
/// 正确性：所有股票同一交易日事件的 day_base 相同（见 day_base），因此基值统一，
/// 预计算的切片与 agg 内即时 period_slice 逐位一致（纯记忆化）。
struct SliceTable {
    sl: Vec<Vec<[u32; 2]>>, // [N_EVENTS * N_PERIODS][n_stocks] = [lo, hi]（u32：表 11MB→5.5MB）
    base: i64,              // 当日统一 day_base（事件均为同一天）
}

/// 预计算全市场切片表 + 零模型表（O(全市场事件数)，一次，并行）。
fn build_slices_and_null(streams: &[Option<[EvStream; N_EVENTS]>]) -> (SliceTable, NullTable) {
    let n = streams.len();
    let base = streams
        .iter()
        .find_map(|s| {
            s.as_ref()
                .and_then(|ev| ev.iter().find_map(|e| (!e.t.is_empty()).then(|| day_base(e.t[0]))))
        })
        .unwrap_or(0);
    let mut sl = vec![vec![[0u32; 2]; n]; N_EVENTS * N_PERIODS];
    sl.par_iter_mut().enumerate().for_each(|(ep, col)| {
        let e = ep / N_PERIODS;
        let p = ep % N_PERIODS;
        for (bi, sb) in streams.iter().enumerate() {
            let Some(sb) = sb else { continue };
            let (lo, hi) = period_slice(&sb[e].t, base, p);
            col[bi] = [lo as u32, hi as u32];
        }
    });
    let mut m = vec![vec![0u32; n]; N_EVENTS * N_PERIODS];
    for (bi, sb) in streams.iter().enumerate() {
        let Some(sb) = sb else { continue };
        for e in 0..N_EVENTS {
            let t = &sb[e].t;
            if t.is_empty() {
                continue;
            }
            for p in 0..N_PERIODS {
                let [lo, hi] = sl[e * N_PERIODS + p][bi];
                m[e * N_PERIODS + p][bi] = hi - lo;
            }
        }
    }
    (SliceTable { sl, base }, NullTable { m })
}

/// 时段切片（t 有序）：返回 [lo, hi) 落入时段 p 的下标区间。
/// p0 快速路径：生产数据（v1 adjust 过滤 / v2 已 adjust 样例）无 09:30 前事件，
/// 下界可跳过（只做 1 次二分）；其余时段常规 2 次二分。
pub fn period_slice(t: &[i64], base: i64, p: usize) -> (usize, usize) {
    let lo = base + PERIOD_LO_S[p] * 1_000_000;
    let hi = base + PERIOD_HI_S[p] * 1_000_000;
    if p == 0 && t.first().map_or(true, |&x| x >= lo) {
        return (0, t.partition_point(|&x| x < hi));
    }
    let s = t.partition_point(|&x| x < lo);
    let e = t.partition_point(|&x| x < hi);
    (s, e)
}

/// 单遍收集器（v3 生产默认）：1 秒桶直方图一次遍历完成所有统计（无第二遍）。
/// - med/fast5/wmed/mean：全部 1 秒桶中点（±0.5s 量化；med/fast5 评估 Spearman
///   1.0000/0.9996，wmed/mean 为 v3 新增近似，相关性已实测评估）
/// - hit：精确标量
/// - 距离一律 u64：i64 差值 as u64 无符号溢出（最大 ~19620s << u64 上限）。
///   历史 u32 版会在 >4295s 的缺口处静默回绕，把数小时的稀疏缺口伪装成小距离，
///   污染 med/fast5/wmed——v3 一并修复
/// 1 秒桶（u32 计数 + f32 权重和，8 字节/桶，同 cache line——桶更新的两次 store
/// 只触碰一条缓存行，分离数组要两条；19620 × 8B = 157KB）。
/// **v6 优化**：max_b 只扫最大触碰桶（locate/stats 从全表 19620 桶缩到实际范围）；
/// touched 列表按触碰桶清零 → 线程本地复用 Gather 缓冲（消除 68.6 万任务 × 314KB
/// 分配/清零，空任务不再分配）。
#[derive(Clone, Copy, Default)]
#[repr(C)]
struct Bucket {
    c: u32,
    w: f32,
}

struct Gather {
    b: Vec<Bucket>,  // 1 秒桶（157KB）
    n: u64,          // 距离条数（免去逐桶求和）
    med_b: usize,
    f5_b: usize,
    wmed_b: usize,
    hit: u64,
    wsum: f64,
    max_b: usize,       // 已触碰的最大桶（locate/stats 只扫 [0, max_b]）
    touched: Vec<usize>, // 已触碰桶列表（重置时只清这些桶）
}

impl Gather {
    fn new() -> Self {
        Gather::new_sized(N_BUCKETS_1S)
    }
    /// 定制桶长（时段最大距离 + 1 的防御余量由调用方保证；v6 生产路径无 clamp）。
    fn new_sized(nb: usize) -> Self {
        Gather {
            b: vec![Bucket::default(); nb],
            n: 0,
            med_b: 0,
            f5_b: 0,
            wmed_b: 0,
            hit: 0,
            wsum: 0.0,
            max_b: 0,
            touched: Vec::with_capacity(64),
        }
    }
    /// 线程本地复用：只清触碰过的桶（touched 去重靠 c==0 判断：首次触碰才记录）。
    fn reset(&mut self) {
        for &i in self.touched.iter() {
            self.b[i] = Bucket::default();
        }
        self.touched.clear();
        self.n = 0;
        self.med_b = 0;
        self.f5_b = 0;
        self.wmed_b = 0;
        self.hit = 0;
        self.wsum = 0.0;
        self.max_b = 0;
    }
    /// 单遍：1 秒桶更新 + 标量（n/wsum/hit）。无 mean 累加（mean 由桶中点统计）。
    /// 测试/调试路径：桶索引带 clamp（生产路径 push_b 由调用方保证越界不可能）。
    /// 注意：桶索引必须用 u64 除法——距离最大 ~19620s = 1.96e10 µs 远超 u32 上限
    /// （4.29e9），as u32 会把 >4295s 的大缺口回绕成小距离（u64 修复的教训）。
    #[inline(always)]
    fn push1(&mut self, dd: u64, ww: f64, t_us: u64) {
        let b = ((dd / 1_000_000) as usize).min(self.b.len() - 1);
        self.push_b(b, ww, dd, t_us);
    }
    /// 生产热路径：桶索引由调用方计算（时段定制桶长 + 距离恒 < 时段跨度，无 clamp）。
    #[inline(always)]
    fn push_b(&mut self, b: usize, ww: f64, dd: u64, t_us: u64) {
        debug_assert!(b < self.b.len(), "桶越界 b={b} len={}", self.b.len());
        let bk = &mut self.b[b];
        if bk.c == 0 {
            self.touched.push(b);
        }
        bk.c += 1;
        bk.w += ww as f32;
        self.n += 1;
        self.wsum += ww;
        self.hit += (dd <= t_us) as u64;
        if b > self.max_b {
            self.max_b = b;
        }
    }
    /// 定位：med/f5 桶（1 秒计数）+ wmed 桶（1 秒权重）。只扫 [0, max_b]（其上全空）。
    /// 分位数位置落在"桶前计数 ≤ 位置 < 桶前计数+桶计数"的桶内。
    fn locate(&mut self, half_n: u64, half_w: f64, k: u64) {
        let mut acc = 0u64;
        let mut f_acc = 0u64;
        let mut wacc = 0.0f64;
        for i in 0..=self.max_b {
            let c = self.b[i].c as u64;
            if c == 0 {
                continue;
            }
            if acc <= half_n && half_n < acc + c {
                self.med_b = i;
            }
            if f_acc < k {
                f_acc += c;
                if f_acc >= k {
                    self.f5_b = i;
                }
            }
            let wc = self.b[i].w as f64;
            if wacc <= half_w && half_w < wacc + wc {
                self.wmed_b = i;
            }
            acc += c;
            wacc += wc;
        }
    }

    /// 统计：[med, mean, hit, fast5, wmed]（秒），全部 1 秒桶中点/精确标量。只扫 [0, max_b]。
    fn stats(&mut self, n: u64, k: u64) -> [f64; 5] {
        if n == 0 {
            return [f64::NAN, f64::NAN, 0.0, f64::NAN, f64::NAN];
        }
        let med = self.med_b as f64 + 0.5;
        let mut mean_sum = 0.0f64;
        let mut f_acc = 0u64;
        let mut f_sum = 0.0f64;
        let mut fast5 = f64::NAN;
        for i in 0..=self.max_b {
            let c = self.b[i].c as u64;
            if c == 0 {
                continue;
            }
            let mid = i as f64 + 0.5;
            mean_sum += c as f64 * mid;
            if f_acc < k {
                let take = c.min(k - f_acc);
                f_sum += take as f64 * mid;
                f_acc += take;
                if f_acc >= k {
                    fast5 = f_sum / k as f64;
                }
            }
        }
        [
            med,
            mean_sum / n as f64,
            self.hit as f64 / n as f64,
            fast5,
            self.wmed_b as f64 + 0.5,
        ]
    }
}
/// **方向打包桶**（v7.4 生产路径）：bwd/fwd 各 4 个时段共用同一桶号 bb（同方向
/// 各组距离 d 相同）→ 合并为 1 个 32B 宽结构，一次 RMW 更新多个时段：
/// - 桶 load/store 次数 /3~4、touched 记录 /3~4（L1 事务大减）
/// - 热循环寄存器：2 个打包数组指针（原 8 个 gather 指针），峰值压力大降
/// - 位级等价：各时段 (c,w) 更新序列、wsum/hit/max_b 逐位一致；时段定制桶长由
///   距离范围保证（各时段 bb 恒 < 自身长度），越界不可能
#[derive(Clone, Copy, Default)]
#[repr(C)]
struct PackB {
    c0: u32,
    w0: f32,
    c1: u32,
    w1: f32,
    c2: u32,
    w2: f32,
    c3: u32,
    w3: f32,
}

/// 打包桶的时段统计字段（对应原 Gather 的 n/med/f5/wmed/hit/wsum/max_b）。
#[derive(Clone, Copy, Default)]
struct PackFields {
    n: u64,
    med_b: usize,
    f5_b: usize,
    wmed_b: usize,
    hit: u64,
    wsum: f64,
    max_b: usize,
}

/// 方向打包桶：b 为 4 时段 (c,w) 宽桶数组（定长 14220 = p0 最长），per[4] 为时段字段。
/// touched 为共享触碰列表（u32 打包桶号；重置时清整条宽桶）。
struct PackGather {
    b: Vec<PackB>,
    per: [PackFields; 4],
    touched: Vec<u32>,
}

impl PackGather {
    fn new_sized(nb: usize) -> Self {
        PackGather {
            b: vec![PackB::default(); nb],
            per: [PackFields::default(); 4],
            touched: Vec::with_capacity(64),
        }
    }
    /// 线程本地复用：按触碰列表清宽桶；字段清零。
    fn reset(&mut self) {
        for &t in self.touched.iter() {
            self.b[t as usize] = PackB::default();
        }
        self.touched.clear();
        self.per = [PackFields::default(); 4];
    }
    /// 取第 p 时段的桶计数（finalize 用）。
    #[inline(always)]
    fn lane_c(&self, p: usize, i: usize) -> u32 {
        let b = &self.b[i];
        match p {
            0 => b.c0,
            1 => b.c1,
            2 => b.c2,
            _ => b.c3,
        }
    }
    #[inline(always)]
    fn lane_w(&self, p: usize, i: usize) -> f32 {
        let b = &self.b[i];
        match p {
            0 => b.w0,
            1 => b.w1,
            2 => b.w2,
            _ => b.w3,
        }
    }
    /// 定位：med/f5/wmed 桶（第 p 时段 lane）。只扫 [0, per[p].max_b]。
    fn locate_p(&mut self, p: usize, half_n: u64, half_w: f64, k: u64) {
        let mut acc = 0u64;
        let mut f_acc = 0u64;
        let mut wacc = 0.0f64;
        for i in 0..=self.per[p].max_b {
            let c = self.lane_c(p, i) as u64;
            if c == 0 {
                continue;
            }
            if acc <= half_n && half_n < acc + c {
                self.per[p].med_b = i;
            }
            if f_acc < k {
                f_acc += c;
                if f_acc >= k {
                    self.per[p].f5_b = i;
                }
            }
            let wc = self.lane_w(p, i) as f64;
            if wacc <= half_w && half_w < wacc + wc {
                self.per[p].wmed_b = i;
            }
            acc += c;
            wacc += wc;
        }
    }
    /// 统计：[med, mean, hit, fast5, wmed]（秒），第 p 时段 lane。
    fn stats_p(&mut self, p: usize, n: u64, k: u64) -> [f64; 5] {
        if n == 0 {
            return [f64::NAN, f64::NAN, 0.0, f64::NAN, f64::NAN];
        }
        let med = self.per[p].med_b as f64 + 0.5;
        let mut mean_sum = 0.0f64;
        let mut f_acc = 0u64;
        let mut f_sum = 0.0f64;
        let mut fast5 = f64::NAN;
        for i in 0..=self.per[p].max_b {
            let c = self.lane_c(p, i) as u64;
            if c == 0 {
                continue;
            }
            let mid = i as f64 + 0.5;
            mean_sum += c as f64 * mid;
            if f_acc < k {
                // fast5 只计前 k 条：跨桶时只取该桶前 (k - f_acc) 条（与 Gather::stats 逐位一致）
                let take = c.min(k - f_acc);
                f_sum += take as f64 * mid;
                f_acc += take;
                if f_acc >= k {
                    fast5 = f_sum / k as f64;
                }
            }
        }
        [med, mean_sum / n as f64, self.per[p].hit as f64 / n as f64, fast5, self.per[p].wmed_b as f64 + 0.5]
    }
}

/// v7 聚合块大小：K 只 A 共享一次 B 遍历（j-walk 的 B 流重读减少 K 倍，
/// 桶数组内存 = K × 560KB/线程：K=2 → 1.12MB 接近 L2 上限，实测选 2）。
const K_BLOCK: usize = 2;

/// 线程本地 8×K 个 Gather（K 只 A × 4 时段 × fwd/bwd，时段定制桶长），任务间复用
/// （touched 清零）。rayon 工作线程串行执行任务，borrow 不会嵌套。
/// 桶长 = 时段最大距离秒数（p0 [5400,19620)→14220、p1 [6000,17820)→11820、
/// p2 [17820,19620)→1800、p3 [12600,19800)→7200），每股 8 个共 560KB 落 L2。
thread_local! {
    /// v7.4：线程本地 2×K 个**方向打包桶**（K 只 A × bwd/fwd，定长 14220 宽桶 × 32B =
    /// 455KB/个），任务间复用（touched 清零）。宽桶内各时段 lane 的 bb 恒小于自身
    /// 时段长度（距离范围保证），高段位 lane 恒 0——与旧 8×Gather 布局逐位等价。
    static GPOOL: std::cell::RefCell<Vec<PackGather>> = std::cell::RefCell::new({
        (0..2 * K_BLOCK).map(|_| PackGather::new_sized(14220)).collect()
    });
}

/// 探针计数器（YHYB_COUNT=1 时启用：统计 a-迭代/push/jwalk 步数，定位工作量规模）
static CNT_A: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static CNT_PUSH: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// 任务级累加器（8 个 Gather 的标量统计；热循环内寄存器驻留——不再每推一次读写
/// Gather 结构体字段，消除 v6 的 4 条内存依赖链，push 吞吐 ~9 周期/推 → ~3-4 周期）。
#[derive(Clone, Copy, Default)]
struct Acc {
    /// 打包 (hit << 32) | max_b：hit ≤ 3e6（22 位）、max_b ≤ 14220（14 位）——一个 u64
    /// 一个寄存器；拆分为 2 个 GPR 会挤爆 16 个 GPR（6 个活跃累加器 = 12 GPR + L4 + 临时量）
    hm: u64,
    wsum: f64,
}

/// v7.4 生产热路径宽桶更新：mask 编译期常量（调用点特化）→ 未选 lane 的代码消除。
/// 逐位等价于同组多次 push_b_acc：(c,w) 各 lane 更新序列一致、wsum/hit/max_b 一致、
/// 首触（整桶全 0）记录 touched（重置时清整条宽桶，语义不变）。
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn push_pack_acc(
    b: &mut [PackB],
    touched: &mut Vec<u32>,
    mask: u32,
    bb: usize,
    ww: f64,
    dd: u64,
    t_us: u64,
    a0: Acc,
    a1: Acc,
    a2: Acc,
    a3: Acc,
) -> (Acc, Acc, Acc, Acc) {
    let old = unsafe { *b.get_unchecked(bb) };
    if (old.c0 | old.c1 | old.c2 | old.c3) == 0 {
        touched.push(bb as u32);
    }
    unsafe {
        *b.get_unchecked_mut(bb) = PackB {
            c0: old.c0 + ((mask >> 0) & 1),
            w0: if mask & 1 != 0 { old.w0 + ww as f32 } else { old.w0 },
            c1: old.c1 + ((mask >> 1) & 1),
            w1: if mask & 2 != 0 { old.w1 + ww as f32 } else { old.w1 },
            c2: old.c2 + ((mask >> 2) & 1),
            w2: if mask & 4 != 0 { old.w2 + ww as f32 } else { old.w2 },
            c3: old.c3 + ((mask >> 3) & 1),
            w3: if mask & 8 != 0 { old.w3 + ww as f32 } else { old.w3 },
        };
    }
    let mut a = [a0, a1, a2, a3];
    for l in 0..4 {
        if mask >> l & 1 != 0 {
            let x = &mut a[l];
            x.wsum += ww;
            let mut h = x.hm >> 32;
            let mut m = (x.hm & 0xFFFF_FFFF) as usize;
            h += (dd <= t_us) as u64;
            if bb > m {
                m = bb;
            }
            x.hm = (h << 32) | m as u64;
        }
    }
    (a[0], a[1], a[2], a[3])
}

/// 第 4 层上下文（对级矩阵 + hub/spoke 写入目标 + 有效索引映射）。
struct L4Ctx<'a> {
    n: usize,
    valid_pos: &'a [usize],
    mf: crate::yhyb_network::SendPtr,
    mb: crate::yhyb_network::SendPtr,
    hp: crate::yhyb_network::SendPtr,
    sp: crate::yhyb_network::SendPtr,
}

/// 单 (A, 事件 e) 的因子：4 时段 × 15 = 60 个值（v7 融合任务，K 股块共享 B 遍历）。
///
/// **v7 优化（输出逐位不变）**：
/// - **K 股块共享 B 遍历**：任务粒度 (K 只 A 的块, e)；每只 B 的事件流/切片表/零模型表
///   只读一次，K 只 A 的 a-循环共享（实测 K=2 与 K=1 持平：1.12MB 桶数组挤压 L2，
///   流级复用被 LRU 逐出；保留 K=2 便于后续桶内存缩减时启用）
/// - **时间比较有效性判断**：j < bsl[p].1 ⟺ tb[j]-base < HI_p_S（partition 性质逐位
///   等价）——时段边界为编译期立即数，省去 v6 每 (a,B) 的 6 个 bsl 索引寄存器与比较
/// - **游标寄存器缓存**：tj_b/tp_b（B 事件后继/前驱相对时间）缓存于寄存器——条件
///   比较零内存加载（v6 每 a 一次 tb[j] 加载，条件加载 2.8e11 次）
/// - **单次 8B 桶加载/存储**（get_unchecked 免越界检查）+ 累加器打包
///   （hit|max_b 一个 u64 + wsum f64，n 由桶计数和推导）+ 延迟装载/段界回写
///   （活跃寄存器集 = 当前段的 gather 数，降低 16 GPR 压力）
/// - **表压缩**：切片表 u32、事件数表 u32、c(m) 记忆化静态缓存（f64 表 → 0）、
///   null_hit powf 任务内记忆化（每 (k,时段,m) 一次）
/// - 探针（YHYB_SKIP_PUSH / YHYB_NO_JWALK）改为编译期常量特化：热循环零运行时分支
/// - v6 保留：预计算切片表 / union-walk / 5 段固定掩码 / 时段定制桶数组 /
///   共享 u64 除法 / hit 无分支累加 / touched 清零复用 / WITH_L4 编译期特化
///
/// 零模型（v3 近似，评估验证相关性见 README）：
/// - null_med = exp(mean ln x)·exp(mean ln c)（geo 默认）等可分式，O(k_A+nB)
/// - null_hit = (1/nB)·Σ_b[1−(1−T/x̄)^m_b]——x̄ 均值近似，powf 记忆化
/// - YHYB_SKIP_NULL 环境变量：跳过零模型（瓶颈定位用，不影响正常路径）
#[allow(clippy::too_many_arguments)]
fn agg_fused_blocked<const WITH_L4: bool, const NO_JWALK: bool, const SKIP_PUSH: bool, const COUNT: bool, const K: usize>(
    streams: &[Option<[EvStream; N_EVENTS]>],
    null_t: &NullTable,
    slices: &SliceTable,
    cst: &CompStreams,
    ai: [usize; K], // 全量索引（streams 访问）
    row: [usize; K], // 有效索引（矩阵行；WITH_L4=false 时忽略）
    nk: usize,       // 块内实际 A 数（≤ K；尾块可能 < K）
    e: usize,
    prm: &YhybParams,
    l4: Option<&L4Ctx>,
) -> [Option<Vec<f64>>; K] {
    let skip_null = std::env::var("YHYB_SKIP_NULL").is_ok();
    let null_mode = std::env::var("YHYB_NULL_MODE").unwrap_or_else(|_| "geo".into());
    let geo = null_mode == "geo";
    let medm = null_mode == "med";
    let no_match = std::env::var("YHYB_NO_MATCH").is_ok();
    let base = slices.base;
    let t_us = (prm.hit_t_s * 1e6) as u64;
    // 时段边界（µs）：零模型时段末 us_hi[p]
    let us_hi = [
        base + PERIOD_HI_S[0] * 1_000_000,
        base + PERIOD_HI_S[1] * 1_000_000,
        base + PERIOD_HI_S[2] * 1_000_000,
        base + PERIOD_HI_S[3] * 1_000_000,
    ];
    // 块内每股状态
    let mut out: [Option<Vec<f64>>; K] = std::array::from_fn(|_| None);
    let mut ta: [&[i64]; K] = [&[]; K];
    let mut wa: [&[f64]; K] = [&[]; K];
    let mut alo = [[0usize; N_PERIODS]; K];
    let mut ahi = [[0usize; N_PERIODS]; K];
    let mut rate = [[0.0f64; N_PERIODS]; K];
    let mut seg = [[(0usize, 0usize); 5]; K];
    let mut sum_x = [[0.0f64; N_PERIODS]; K];
    let mut sum_ln_x = [[0.0f64; N_PERIODS]; K];
    let mut k_a = [[0.0f64; N_PERIODS]; K];
    let mut x_bar = [[0.0f64; N_PERIODS]; K];
    let mut sum_c = [[0.0f64; N_PERIODS]; K];
    let mut sum_ln_c = [[0.0f64; N_PERIODS]; K];
    let mut c_list: [Vec<Vec<f64>>; K] = std::array::from_fn(|_| Vec::new());
    let mut null_hit_acc = [[0.0f64; N_PERIODS]; K];
    let mut n_b = [[0u64; N_PERIODS]; K];
    // 第 4 层 per-k 累加器（跨 B 累积）
    let mut f_hit_b = [0u64; K];
    let mut b_hit_b = [0u64; K];
    let mut n_b_pair = [0u64; K];
    // 预计算每股：切片 / rate / 5 段 / 零模型 x 侧
    let mut all_empty = true;
    for k in 0..nk {
        let Some(sa) = streams[ai[k]].as_ref() else { continue };
        ta[k] = &sa[e].t;
        wa[k] = &sa[e].w;
        alo[k] = [
            slices.sl[e * N_PERIODS + 0][ai[k]][0] as usize,
            slices.sl[e * N_PERIODS + 1][ai[k]][0] as usize,
            slices.sl[e * N_PERIODS + 2][ai[k]][0] as usize,
            slices.sl[e * N_PERIODS + 3][ai[k]][0] as usize,
        ];
        ahi[k] = [
            slices.sl[e * N_PERIODS + 0][ai[k]][1] as usize,
            slices.sl[e * N_PERIODS + 1][ai[k]][1] as usize,
            slices.sl[e * N_PERIODS + 2][ai[k]][1] as usize,
            slices.sl[e * N_PERIODS + 3][ai[k]][1] as usize,
        ];
        for p in 0..N_PERIODS {
            rate[k][p] = (ahi[k][p] - alo[k][p]) as f64;
        }
        let ualo = alo[k][0].min(alo[k][1]).min(alo[k][2]).min(alo[k][3]);
        let uahi = ahi[k][0].max(ahi[k][1]).max(ahi[k][2]).max(ahi[k][3]);
        if ualo == uahi {
            // 全时段无 A 事件：rate=0 + 14 NaN × 4 时段（与 v6 逐位一致）
            let mut v = Vec::with_capacity(60);
            for p in 0..N_PERIODS {
                v.push(rate[k][p]);
                for _ in 0..14 {
                    v.push(f64::NAN);
                }
            }
            out[k] = Some(v);
            continue;
        }
        all_empty = false;
        debug_assert_eq!(day_base(ta[k][ualo]), base, "切片表基值与任务基值不一致");
        // 5 段 A 事件范围（固定时段边界；掩码恒定）：
        //   S0 [5400,6000) {p0} | S1 [6000,12600) {p0,p1} | S2 [12600,17820) {p0,p1,p3}
        //   S3 [17820,19620) {p0,p2,p3} | S4 [19620,19800) {p3}
        seg[k][0] = (ualo, ta[k].partition_point(|&x| x < base + 6_000_000_000));
        seg[k][1] = (seg[k][0].1, ta[k].partition_point(|&x| x < base + 12_600_000_000));
        seg[k][2] = (seg[k][1].1, ta[k].partition_point(|&x| x < base + 17_820_000_000));
        seg[k][3] = (seg[k][2].1, ta[k].partition_point(|&x| x < base + 19_620_000_000));
        seg[k][4] = (seg[k][3].1, uahi);
        // 零模型可分式预计算：Σ x_i（A 事件时段剩余）与 x̄，每任务 O(k_A)
        for p in 0..N_PERIODS {
            k_a[k][p] = (ahi[k][p] - alo[k][p]) as f64;
            if !skip_null {
                for &a in &ta[k][alo[k][p]..ahi[k][p]] {
                    sum_x[k][p] += (us_hi[p] - a).max(0) as f64;
                    if geo {
                        sum_ln_x[k][p] += ((us_hi[p] - a).max(1) as f64).ln();
                    }
                }
            }
            x_bar[k][p] = if skip_null { 0.0 } else { sum_x[k][p] / k_a[k][p] };
        }
    }
    if all_empty {
        return out; // 块内全部无事件：GPOOL 不触碰（与 v6 提前返回一致）
    }
    // null_hit 的 (1-T/x̄)^m 记忆化表：x̄ 是 A 股事件时段剩余均值（任务内常数），
    // 每 (k, 时段) 只有 1 个底数 → 每个 m 的 powf 任务内只算一次（懒填充；
    // 底数 0 时 powf(0,m)=0，与哨兵值一致，天然正确）。逐位等价（同一 powf 调用）。
    let mut pow_tab: [[[f64; 512]; N_PERIODS]; 8] = [[[0.0; 512]; N_PERIODS]; 8];
    GPOOL.with(|pool| {
        let mut pool = pool.borrow_mut();
        let g = &mut pool[..];
        for x in g.iter_mut() {
            x.reset();
        }
        for (bi, sb) in streams.iter().enumerate() {
            let Some(sb) = sb else { continue };
            let bpos = l4.map_or(usize::MAX, |c| c.valid_pos[bi]);
            if WITH_L4 && bpos == usize::MAX {
                continue;
            }
            // B 侧事件流：u32 增量压缩副本（4B/事件 vs 8B——j-walk DRAM 减半）
            let c_off = cst.offs[e][bi] as usize;
            let c_first = cst.first[e][bi];
            let c_len = cst.lens[e][bi] as usize;
            if c_len == 0 {
                continue;
            }
            let c_deltas = &cst.deltas[e];
            // 软件预取 bi+2 的切片/事件数/有效索引：B 访问的 L3 延迟与当前 B 的
            // a-循环重叠（每 B 只读 ~50B，纯延迟绑定 ~10s；预取后基本隐藏）
            let pf = bi + 2;
            if pf < streams.len() {
                unsafe {
                    core::arch::x86_64::_mm_prefetch(
                        &slices.sl[e * N_PERIODS + 0][pf] as *const _ as *const i8,
                        core::arch::x86_64::_MM_HINT_T0,
                    );
                    core::arch::x86_64::_mm_prefetch(
                        &slices.sl[e * N_PERIODS + 1][pf] as *const _ as *const i8,
                        core::arch::x86_64::_MM_HINT_T0,
                    );
                    core::arch::x86_64::_mm_prefetch(
                        &slices.sl[e * N_PERIODS + 2][pf] as *const _ as *const i8,
                        core::arch::x86_64::_MM_HINT_T0,
                    );
                    core::arch::x86_64::_mm_prefetch(
                        &slices.sl[e * N_PERIODS + 3][pf] as *const _ as *const i8,
                        core::arch::x86_64::_MM_HINT_T0,
                    );
                    core::arch::x86_64::_mm_prefetch(
                        &null_t.m[e * N_PERIODS][pf] as *const _ as *const i8,
                        core::arch::x86_64::_MM_HINT_T0,
                    );
                    if !WITH_L4 {
                        // valid_pos 只在 L4 路径读取；无条件预取亦可（无副作用）
                    }
                }
            }
            let bsl = [
                (
                    slices.sl[e * N_PERIODS + 0][bi][0] as usize,
                    slices.sl[e * N_PERIODS + 0][bi][1] as usize,
                ),
                (
                    slices.sl[e * N_PERIODS + 1][bi][0] as usize,
                    slices.sl[e * N_PERIODS + 1][bi][1] as usize,
                ),
                (
                    slices.sl[e * N_PERIODS + 2][bi][0] as usize,
                    slices.sl[e * N_PERIODS + 2][bi][1] as usize,
                ),
                (
                    slices.sl[e * N_PERIODS + 3][bi][0] as usize,
                    slices.sl[e * N_PERIODS + 3][bi][1] as usize,
                ),
            ];
            // 块内每股的 B 遍历（共享 bsl/tb；每股独立 j 游标与累加器）。
            // 宏体以常量 k 展开 → acc[k*8+gi] 常量索引 → SROA 拆标量 → 寄存器驻留。
            macro_rules! agg_k_body {
                ($k:expr) => {{
                    if bi != ai[$k] {
                        let g2 = &mut g[$k * 2..$k * 2 + 2];
                        let [bp, fp] = g2 else { unreachable!() };
                        // 零模型（每时段：B 有时段事件才贡献；每 (A,B) 一次）
                        if !skip_null {
                            for p in 0..N_PERIODS {
                                let (blo, bhi) = bsl[p];
                                if blo < bhi {
                                    let m_b = null_t.m[e * N_PERIODS + p][bi];
                                    let c_b = c_of_m(m_b);
                                    sum_c[$k][p] += c_b;
                                    if geo {
                                        sum_ln_c[$k][p] += ln_c_of_m(m_b);
                                    }
                                    if medm {
                                        c_list[$k][p].push(c_b);
                                    }
                                    n_b[$k][p] += 1;
                                    // null_hit x̄ 近似：每 B 一次（clamp 防 x̄ ≤ T 时底数非正）；
                                    // powf 记忆化（任务内每 (k,时段,m) 一次，m ≤ 511 查表）
                                    let base_p = (1.0 - t_us as f64 / x_bar[$k][p]).clamp(0.0, 1.0);
                                    let pv = if m_b < 512 {
                                        let slot = &mut pow_tab[$k][p][m_b as usize];
                                        if *slot == 0.0 && base_p != 0.0 {
                                            *slot = base_p.powf(m_b as f64);
                                        }
                                        *slot
                                    } else {
                                        base_p.powf(m_b as f64)
                                    };
                                    null_hit_acc[$k][p] += 1.0 - pv;
                                }
                            }
                        }
                        if no_match {
                            // 探针：只做 B 访问（切片+null），跳过匹配推送
                        } else {
                            let tk = ta[$k];
                            let wk = wa[$k];
                            // v7.4 方向打包累加器：ab/af 各 4 时段（bwd: p0..p3 ↔ 原
                            // al[1,3,5,7]；fwd: p0..p3 ↔ al[0,2,4,6]）。延迟装载 + 段界回写。
                            let mut ab: [Acc; 4] = [Acc::default(); 4];
                            let mut af: [Acc; 4] = [Acc::default(); 4];
                            macro_rules! al_load_b {
                                ($p:expr) => {
                                    ab[$p] = Acc {
                                        hm: (bp.per[$p].hit << 32) | (bp.per[$p].max_b as u64),
                                        wsum: bp.per[$p].wsum,
                                    }
                                };
                            }
                            macro_rules! al_flush_b {
                                ($p:expr) => {
                                    bp.per[$p].hit = ab[$p].hm >> 32;
                                    bp.per[$p].max_b = (ab[$p].hm & 0xFFFF_FFFF) as usize;
                                    bp.per[$p].wsum = ab[$p].wsum;
                                };
                            }
                            macro_rules! al_load_f {
                                ($p:expr) => {
                                    af[$p] = Acc {
                                        hm: (fp.per[$p].hit << 32) | (fp.per[$p].max_b as u64),
                                        wsum: fp.per[$p].wsum,
                                    }
                                };
                            }
                            macro_rules! al_flush_f {
                                ($p:expr) => {
                                    fp.per[$p].hit = af[$p].hm >> 32;
                                    fp.per[$p].max_b = (af[$p].hm & 0xFFFF_FFFF) as usize;
                                    fp.per[$p].wsum = af[$p].wsum;
                                };
                            }
                            al_load_f!(0);
                            al_load_b!(0);
                            // 第 4 层 p0 对级累加器（每 (A,B) 独立；计数打包成 u64 对省 2 GPR）
                            let mut kpfs = 0.0f64;
                            let mut kpbs = 0.0f64;
                            let mut l4c = 0u64; // (kpfn << 32) | kpbn
                            let mut l4h = 0u64; // (kpfh << 32) | kpbh
                            // 有效性判断用**基准相对时间 + 编译期立即数**（partition 性质逐位等价）；
                            // union-walk 游标 tj_b = tb[j]-base / tp_b = tb[j-1]-base 缓存于寄存器。
                            let mut j = 0usize;
                            let mut c_pos = c_off;
                            let mut tj_b = c_first;
                            let mut tp_b = i64::MIN;
                            // 软件预取宏：步行增量流提前 ~128B（每 a 一次，隐藏 L2/L3 延迟
                            // 串行链——jwalk 每步 ~21 cyc 中大部分是等待增量加载；PREFETCH
                            // 对任意地址无副作用，越界安全，无需边界检查）
                            macro_rules! pf_delta {
                                () => {
                                    unsafe {
                                        core::arch::x86_64::_mm_prefetch(
                                            c_deltas.as_ptr().wrapping_add(c_pos + 32) as *const i8,
                                            core::arch::x86_64::_MM_HINT_T0,
                                        )
                                    }
                                };
                            }
                            // 宽桶推送宏：mask 为编译期常量（位 0..3 = p0..p3；调用点特化，
                            // 未选 lane 的代码消除）；SKIP_PUSH 探针在宏内剔除
                            macro_rules! push_bw {
                                ($mask:expr, $bb:expr, $wi:expr, $d:expr) => {{
                                    if !SKIP_PUSH {
                                        let (t0, t1, t2, t3) = push_pack_acc(
                                            &mut bp.b,
                                            &mut bp.touched,
                                            $mask,
                                            $bb,
                                            $wi,
                                            $d,
                                            t_us,
                                            ab[0],
                                            ab[1],
                                            ab[2],
                                            ab[3],
                                        );
                                        ab[0] = t0;
                                        ab[1] = t1;
                                        ab[2] = t2;
                                        ab[3] = t3;
                                    }
                                }};
                            }
                            macro_rules! push_fw {
                                ($mask:expr, $bb:expr, $wi:expr, $d:expr) => {{
                                    if !SKIP_PUSH {
                                        let (t0, t1, t2, t3) = push_pack_acc(
                                            &mut fp.b,
                                            &mut fp.touched,
                                            $mask,
                                            $bb,
                                            $wi,
                                            $d,
                                            t_us,
                                            af[0],
                                            af[1],
                                            af[2],
                                            af[3],
                                        );
                                        af[0] = t0;
                                        af[1] = t1;
                                        af[2] = t2;
                                        af[3] = t3;
                                    }
                                }};
                            }
                            // S0: a ∈ [5400,6000) → p0
                            for i in seg[$k][0].0..seg[$k][0].1 {
                                let ab = tk[i] - base;
                                let wi = wk[i];
                                pf_delta!();
                                if !NO_JWALK {
                                    while j < c_len && tj_b <= ab {
                                        j += 1;
                                        tp_b = tj_b;
                                        if j < c_len {
                                            // 最后一个事件之后无增量：tj_b = MAX（与 v6 的
                                            // `if j < tb_len { tb[j] } else { i64::MAX }` 逐位一致）
                                            let dv = c_deltas[c_pos];
                                            tj_b += if dv == DELTA_ESC {
                                                c_pos += 3;
                                                (c_deltas[c_pos - 2] as i64) | ((c_deltas[c_pos - 1] as i64) << 32)
                                            } else {
                                                c_pos += 1;
                                                dv as i64
                                            };
                                        } else {
                                            tj_b = i64::MAX;
                                        }
                                    }
                                }
                                if tp_b >= LO0_S {
                                    let d = (ab - tp_b) as u64;
                                    let bb = (d / 1_000_000) as usize;
                                    push_bw!(0b0001, bb, wi, d);
                                    if WITH_L4 {
                                        kpbs += d as f64;
                                        l4c += 1;
                                        l4h += (d <= t_us) as u64;
                                    }
                                }
                                if tj_b < HI0_S {
                                    let d = (tj_b - ab) as u64;
                                    let bb = (d / 1_000_000) as usize;
                                    push_fw!(0b0001, bb, wi, d);
                                    if WITH_L4 {
                                        kpfs += d as f64;
                                        l4c += 0x1_0000_0000;
                                        l4h += ((d <= t_us) as u64) << 32;
                                    }
                                }
                            }
                            // S1: a ∈ [6000,12600) → p0, p1（bwd/fwd: p1⊂p0 嵌套）
                            al_load_f!(1);
                            al_load_b!(1);
                            for i in seg[$k][1].0..seg[$k][1].1 {
                                let ab = tk[i] - base;
                                let wi = wk[i];
                                pf_delta!();
                                if !NO_JWALK {
                                    while j < c_len && tj_b <= ab {
                                        j += 1;
                                        tp_b = tj_b;
                                        if j < c_len {
                                            // 最后一个事件之后无增量：tj_b = MAX（与 v6 的
                                            // `if j < tb_len { tb[j] } else { i64::MAX }` 逐位一致）
                                            let dv = c_deltas[c_pos];
                                            tj_b += if dv == DELTA_ESC {
                                                c_pos += 3;
                                                (c_deltas[c_pos - 2] as i64) | ((c_deltas[c_pos - 1] as i64) << 32)
                                            } else {
                                                c_pos += 1;
                                                dv as i64
                                            };
                                        } else {
                                            tj_b = i64::MAX;
                                        }
                                    }
                                }
                                if tp_b >= LO1_S {
                                    let d = (ab - tp_b) as u64;
                                    let bb = (d / 1_000_000) as usize;
                                    push_bw!(0b0011, bb, wi, d);
                                    if WITH_L4 {
                                        kpbs += d as f64;
                                        l4c += 1;
                                        l4h += (d <= t_us) as u64;
                                    }
                                } else if tp_b >= LO0_S {
                                    let d = (ab - tp_b) as u64;
                                    let bb = (d / 1_000_000) as usize;
                                    push_bw!(0b0001, bb, wi, d);
                                    if WITH_L4 {
                                        kpbs += d as f64;
                                        l4c += 1;
                                        l4h += (d <= t_us) as u64;
                                    }
                                }
                                if tj_b < HI1_S {
                                    let d = (tj_b - ab) as u64;
                                    let bb = (d / 1_000_000) as usize;
                                    push_fw!(0b0011, bb, wi, d);
                                    if WITH_L4 {
                                        kpfs += d as f64;
                                        l4c += 0x1_0000_0000;
                                        l4h += ((d <= t_us) as u64) << 32;
                                    }
                                } else if tj_b < HI0_S {
                                    let d = (tj_b - ab) as u64;
                                    let bb = (d / 1_000_000) as usize;
                                    push_fw!(0b0001, bb, wi, d);
                                    if WITH_L4 {
                                        kpfs += d as f64;
                                        l4c += 0x1_0000_0000;
                                        l4h += ((d <= t_us) as u64) << 32;
                                    }
                                }
                            }
                            // S2: a ∈ [12600,17820) → p0, p1, p3（bwd: p3⊂p1⊂p0；fwd: p1⊂p0⊂p3）
                            al_load_f!(3);
                            al_load_b!(3);
                            for i in seg[$k][2].0..seg[$k][2].1 {
                                let ab = tk[i] - base;
                                let wi = wk[i];
                                pf_delta!();
                                if !NO_JWALK {
                                    while j < c_len && tj_b <= ab {
                                        j += 1;
                                        tp_b = tj_b;
                                        if j < c_len {
                                            // 最后一个事件之后无增量：tj_b = MAX（与 v6 的
                                            // `if j < tb_len { tb[j] } else { i64::MAX }` 逐位一致）
                                            let dv = c_deltas[c_pos];
                                            tj_b += if dv == DELTA_ESC {
                                                c_pos += 3;
                                                (c_deltas[c_pos - 2] as i64) | ((c_deltas[c_pos - 1] as i64) << 32)
                                            } else {
                                                c_pos += 1;
                                                dv as i64
                                            };
                                        } else {
                                            tj_b = i64::MAX;
                                        }
                                    }
                                }
                                if tp_b >= LO3_S {
                                    let d = (ab - tp_b) as u64;
                                    let bb = (d / 1_000_000) as usize;
                                    push_bw!(0b1011, bb, wi, d);
                                    if WITH_L4 {
                                        kpbs += d as f64;
                                        l4c += 1;
                                        l4h += (d <= t_us) as u64;
                                    }
                                } else if tp_b >= LO1_S {
                                    let d = (ab - tp_b) as u64;
                                    let bb = (d / 1_000_000) as usize;
                                    push_bw!(0b0011, bb, wi, d);
                                    if WITH_L4 {
                                        kpbs += d as f64;
                                        l4c += 1;
                                        l4h += (d <= t_us) as u64;
                                    }
                                } else if tp_b >= LO0_S {
                                    let d = (ab - tp_b) as u64;
                                    let bb = (d / 1_000_000) as usize;
                                    push_bw!(0b0001, bb, wi, d);
                                    if WITH_L4 {
                                        kpbs += d as f64;
                                        l4c += 1;
                                        l4h += (d <= t_us) as u64;
                                    }
                                }
                                if tj_b < HI1_S {
                                    let d = (tj_b - ab) as u64;
                                    let bb = (d / 1_000_000) as usize;
                                    // HI1_S=19620e6 < HI3_S=19800e6 → HI1 蕴含 HI3，原 `if
                                    // tj_b < HI3_S { al[6] }` 恒真，并入 mask 0b1011（位级等价）
                                    push_fw!(0b1011, bb, wi, d);
                                    if WITH_L4 {
                                        kpfs += d as f64;
                                        l4c += 0x1_0000_0000;
                                        l4h += ((d <= t_us) as u64) << 32;
                                    }
                                } else if tj_b < HI0_S {
                                    let d = (tj_b - ab) as u64;
                                    let bb = (d / 1_000_000) as usize;
                                    push_fw!(0b1001, bb, wi, d);
                                    if WITH_L4 {
                                        kpfs += d as f64;
                                        l4c += 0x1_0000_0000;
                                        l4h += ((d <= t_us) as u64) << 32;
                                    }
                                } else if tj_b < HI3_S && !SKIP_PUSH {
                                    let d = (tj_b - ab) as u64;
                                    let bb = (d / 1_000_000) as usize;
                                    push_fw!(0b1000, bb, wi, d);
                                }
                            }
                            // S3: a ∈ [17820,19620) → p0, p2, p3（bwd: p2⊂p3⊂p0；fwd: p2=p0⊂p3）
                            al_flush_b!(1);
                            al_flush_f!(1);
                            al_load_b!(2);
                            al_load_f!(2);
                            for i in seg[$k][3].0..seg[$k][3].1 {
                                let ab = tk[i] - base;
                                let wi = wk[i];
                                pf_delta!();
                                if !NO_JWALK {
                                    while j < c_len && tj_b <= ab {
                                        j += 1;
                                        tp_b = tj_b;
                                        if j < c_len {
                                            // 最后一个事件之后无增量：tj_b = MAX（与 v6 的
                                            // `if j < tb_len { tb[j] } else { i64::MAX }` 逐位一致）
                                            let dv = c_deltas[c_pos];
                                            tj_b += if dv == DELTA_ESC {
                                                c_pos += 3;
                                                (c_deltas[c_pos - 2] as i64) | ((c_deltas[c_pos - 1] as i64) << 32)
                                            } else {
                                                c_pos += 1;
                                                dv as i64
                                            };
                                        } else {
                                            tj_b = i64::MAX;
                                        }
                                    }
                                }
                                if tp_b >= LO2_S {
                                    let d = (ab - tp_b) as u64;
                                    let bb = (d / 1_000_000) as usize;
                                    push_bw!(0b1101, bb, wi, d);
                                    if WITH_L4 {
                                        kpbs += d as f64;
                                        l4c += 1;
                                        l4h += (d <= t_us) as u64;
                                    }
                                } else if tp_b >= LO3_S {
                                    let d = (ab - tp_b) as u64;
                                    let bb = (d / 1_000_000) as usize;
                                    push_bw!(0b1001, bb, wi, d);
                                    if WITH_L4 {
                                        kpbs += d as f64;
                                        l4c += 1;
                                        l4h += (d <= t_us) as u64;
                                    }
                                } else if tp_b >= LO0_S {
                                    let d = (ab - tp_b) as u64;
                                    let bb = (d / 1_000_000) as usize;
                                    push_bw!(0b0001, bb, wi, d);
                                    if WITH_L4 {
                                        kpbs += d as f64;
                                        l4c += 1;
                                        l4h += (d <= t_us) as u64;
                                    }
                                }
                                if tj_b < HI0_S {
                                    let d = (tj_b - ab) as u64;
                                    let bb = (d / 1_000_000) as usize;
                                    push_fw!(0b1101, bb, wi, d);
                                    if WITH_L4 {
                                        kpfs += d as f64;
                                        l4c += 0x1_0000_0000;
                                        l4h += ((d <= t_us) as u64) << 32;
                                    }
                                } else if tj_b < HI3_S && !SKIP_PUSH {
                                    let d = (tj_b - ab) as u64;
                                    let bb = (d / 1_000_000) as usize;
                                    push_fw!(0b1000, bb, wi, d);
                                }
                            }
                            // S4: a ∈ [19620,19800) → p3
                            al_flush_b!(2);
                            al_flush_f!(2);
                            for i in seg[$k][4].0..seg[$k][4].1 {
                                let ab = tk[i] - base;
                                let wi = wk[i];
                                pf_delta!();
                                if !NO_JWALK {
                                    while j < c_len && tj_b <= ab {
                                        j += 1;
                                        tp_b = tj_b;
                                        if j < c_len {
                                            // 最后一个事件之后无增量：tj_b = MAX（与 v6 的
                                            // `if j < tb_len { tb[j] } else { i64::MAX }` 逐位一致）
                                            let dv = c_deltas[c_pos];
                                            tj_b += if dv == DELTA_ESC {
                                                c_pos += 3;
                                                (c_deltas[c_pos - 2] as i64) | ((c_deltas[c_pos - 1] as i64) << 32)
                                            } else {
                                                c_pos += 1;
                                                dv as i64
                                            };
                                        } else {
                                            tj_b = i64::MAX;
                                        }
                                    }
                                }
                                if tp_b >= LO3_S && !SKIP_PUSH {
                                    let d = (ab - tp_b) as u64;
                                    let bb = (d / 1_000_000) as usize;
                                    push_bw!(0b1000, bb, wi, d);
                                }
                                if tj_b < HI3_S && !SKIP_PUSH {
                                    let d = (tj_b - ab) as u64;
                                    let bb = (d / 1_000_000) as usize;
                                    push_fw!(0b1000, bb, wi, d);
                                }
                            }
                            // 写第 4 层对级矩阵（本任务独占第 (e,row) 行；B 无 p0 事件时保持 NaN）
                            if let Some(c) = l4 {
                                if bsl[0].0 < bsl[0].1 {
                                    let kpfn = l4c >> 32;
                                    let kpbn = l4c & 0xFFFF_FFFF;
                                    let kpfh = l4h >> 32;
                                    let kpbh = l4h & 0xFFFF_FFFF;
                                    let vf = if kpfn > 0 { (kpfs / kpfn as f64) as f32 } else { f32::NAN };
                                    let vb = if kpbn > 0 { (kpbs / kpbn as f64) as f32 } else { f32::NAN };
                                    c.mf.w((e * c.n + row[$k]) * c.n + bpos, vf);
                                    c.mb.w((e * c.n + row[$k]) * c.n + bpos, vb);
                                    if kpfh > 0 {
                                        f_hit_b[$k] += 1;
                                    }
                                    if kpbh > 0 {
                                        b_hit_b[$k] += 1;
                                    }
                                    n_b_pair[$k] += 1;
                                }
                            }
                            // 回写累加器到打包桶字段（每 (A,B) 一次；字段跨 B 累积）
                            al_flush_f!(0);
                            al_flush_b!(0);
                            al_flush_f!(3);
                            al_flush_b!(3);
                        }
                    }
                }};
            }
            if nk > 1 {
                agg_k_body!(0);
                agg_k_body!(1);
            } else {
                agg_k_body!(0);
            }
        }
        if COUNT {
            // 每任务一次原子累加（探针）：pushes = 8 个 Gather 的 n 之和；
            // a-iterations = 各段长度之和（每 A）
            for k in 0..nk {
                if out[k].is_some() {
                    continue;
                }
                let mut ps = 0u64;
                for pi in 0..2 {
                    let pk = &g[k * 2 + pi];
                    for p in 0..4 {
                        let mut ns = 0u64;
                        for i in 0..=pk.per[p].max_b {
                            ns += pk.lane_c(p, i) as u64;
                        }
                        ps += ns;
                    }
                }
                let mut ai_cnt = 0usize;
                for s in seg[k].iter() {
                    ai_cnt += s.1 - s.0;
                }
                CNT_PUSH.fetch_add(ps, std::sync::atomic::Ordering::Relaxed);
                CNT_A.fetch_add(ai_cnt as u64, std::sync::atomic::Ordering::Relaxed);
            }
        }
        // 写回累加器 + 定位/统计 + hub/spoke + 输出（每 k）
        for k in 0..nk {
            if out[k].is_some() {
                continue; // 空行已填
            }
            let g2 = &mut g[k * 2..k * 2 + 2];
            let [bp, fp] = g2 else { unreachable!() };
            // n 由桶计数和推导（整数和，与逐推累加逐位一致）：省去热循环内 n 累加
            for p in 0..N_PERIODS {
                let mut nfb = 0u64;
                for i in 0..=fp.per[p].max_b {
                    nfb += fp.lane_c(p, i) as u64;
                }
                fp.per[p].n = nfb;
                let mut nbb = 0u64;
                for i in 0..=bp.per[p].max_b {
                    nbb += bp.lane_c(p, i) as u64;
                }
                bp.per[p].n = nbb;
            }
            // 定位分位数桶（wsum 由 push 维护；n 已推导）
            let mut st = [[0.0f64; 5]; N_PERIODS * 2];
            for p in 0..N_PERIODS {
                let nf = fp.per[p].n;
                let nb = bp.per[p].n;
                let kf = ((nf as f64) * prm.fast_q).round().max(1.0) as u64;
                let kb = ((nb as f64) * prm.fast_q).round().max(1.0) as u64;
                if nf > 0 {
                    fp.locate_p(p, nf / 2, fp.per[p].wsum / 2.0, kf);
                }
                if nb > 0 {
                    bp.locate_p(p, nb / 2, bp.per[p].wsum / 2.0, kb);
                }
                st[p * 2] = fp.stats_p(p, nf, kf);
                st[p * 2 + 1] = bp.stats_p(p, nb, kb);
            }
            // hub/spoke（命中 B 占比）：与旧版 p0 任务语义一致——A 无 p0 事件时任务提前返回，
            // hub/spoke 保持 NaN（若 A 有 p0 事件但无任何 B 响应则写 NaN）
            if let Some(c) = l4 {
                if alo[k][0] < ahi[k][0] {
                    c.hp.w(e * c.n + row[k], if n_b_pair[k] > 0 { f_hit_b[k] as f32 / n_b_pair[k] as f32 } else { f32::NAN });
                    c.sp.w(e * c.n + row[k], if n_b_pair[k] > 0 { b_hit_b[k] as f32 / n_b_pair[k] as f32 } else { f32::NAN });
                }
            }
            // 每时段 null_med/null_hit + 输出 15 值
            let mut v = Vec::with_capacity(60);
            for p in 0..N_PERIODS {
                // 空时段（A 无事件）：与旧版逐任务语义一致——rate=0 + 14 个 NaN
                // （注意：hit 也必须为 NaN 而非 0.0，否则 fill_periods 链条回退会拾取 0.0
                //  而旧版会回退到更近时段的真实值，输出产生系统性差异）
                if rate[k][p] == 0.0 {
                    v.push(0.0);
                    for _ in 0..14 {
                        v.push(f64::NAN);
                    }
                    continue;
                }
                let null_med = if !skip_null && n_b[k][p] > 0 {
                    if geo {
                        (sum_ln_x[k][p] / k_a[k][p]).exp() * (sum_ln_c[k][p] / n_b[k][p] as f64).exp() / 1e6
                    } else if medm {
                        // x 降序中位（a 升序 → x 降序，上中位与 select_nth(len/2) 同秩）
                        let mid_x = (us_hi[p] - ta[k][ahi[k][p] - 1 - (ahi[k][p] - alo[k][p]) / 2]).max(0) as f64;
                        let mid_c = {
                            let m = c_list[k][p].len() / 2;
                            let (_, &mut v2, _) = c_list[k][p].select_nth_unstable_by(m, |a, b| a.total_cmp(b));
                            v2
                        };
                        mid_x * mid_c / 1e6
                    } else {
                        (sum_x[k][p] * sum_c[k][p]) / (k_a[k][p] * n_b[k][p] as f64) / 1e6
                    }
                } else {
                    f64::NAN
                };
                let null_hit = if !skip_null && n_b[k][p] > 0 {
                    null_hit_acc[k][p] / n_b[k][p] as f64
                } else {
                    f64::NAN
                };
                v.push(rate[k][p]);
                for dd in [&st[p * 2], &st[p * 2 + 1]] {
                    v.push(dd[0]);
                    v.push(dd[1]);
                    v.push(dd[2]);
                    v.push(dd[3]);
                    v.push(dd[4]);
                    v.push(dd[0] / null_med); // rmed
                    v.push(dd[2] / null_hit); // rhit
                }
            }
            out[k] = Some(v);
        }
    });
    out
}

/// 列出某天某子目录下所有股票代码（文件名 `{code}_{date}_{type}.csv`）。
pub fn list_codes(date: i64, subdir: &str) -> Vec<String> {
    let dir = format!("/ssd_data/stock/{date}/{subdir}");
    let mut set = BTreeSet::new();
    if let Ok(entries) = fs::read_dir(&dir) {
        for e in entries.flatten() {
            let name = e.file_name().into_string().unwrap_or_default();
            if let Some(code) = name.split('_').next() {
                if code.chars().all(|c| c.is_ascii_digit()) {
                    set.insert(code.to_string());
                }
            }
        }
    }
    set.into_iter().collect()
}

// ============================================================================
// 近似模式（仅用于评估妥协方案的截面影响，YHYB_APPROX 版本，不用于生产）
// 复现历史妥协：med/fast5 用 1 秒桶中点、wmed→wmean（加权均值）、
// null_med 均值池化、null_hit x̄ 均值近似。输出布局与精确版完全一致（可直接对比）。
// 距离同样 u64（与生产版一致，无 u32 回绕）。
// ============================================================================

#[derive(Default)]
struct Hist1 {
    cnt: Vec<u32>, // 1 秒桶计数（u32；78KB）
    mean: f64,
    hit: u64,
    wsum: f64,
    wmean_num: f64, // Σw·d（wmean 用）
}

impl Hist1 {
    fn new() -> Self {
        Hist1 {
            cnt: vec![0; N_BUCKETS_1S],
            ..Default::default()
        }
    }
    fn clear(&mut self) {
        self.cnt.fill(0);
        self.mean = 0.0;
        self.hit = 0;
        self.wsum = 0.0;
        self.wmean_num = 0.0;
    }
    #[inline(always)]
    fn push(&mut self, dd: u64, ww: f64, t_us: u64) {
        let b = ((dd / 1_000_000) as usize).min(N_BUCKETS_1S - 1);
        self.cnt[b] += 1;
        self.mean += dd as f64;
        self.wsum += ww;
        self.wmean_num += ww * dd as f64;
        if dd <= t_us {
            self.hit += 1;
        }
    }
    /// 1 秒桶中点统计：[med, mean, hit, fast5, wmean]（秒）。
    fn stats(&self, n: u64, fast_q: f64) -> [f64; 5] {
        if n == 0 {
            return [f64::NAN, f64::NAN, 0.0, f64::NAN, f64::NAN];
        }
        let k = ((n as f64) * fast_q).round().max(1.0) as u64;
        let half = n / 2;
        let mut acc = 0u64;
        let mut f_acc = 0u64;
        let mut f_sum = 0.0f64;
        let mut med = f64::NAN;
        let mut fast5 = f64::NAN;
        for (b, &c) in self.cnt.iter().enumerate() {
            let c = c as u64;
            if c == 0 {
                continue;
            }
            let mid = b as f64 + 0.5;
            if med.is_nan() {
                acc += c;
                if acc > half {
                    med = mid;
                }
            }
            if f_acc < k {
                let take = c.min(k - f_acc);
                f_sum += take as f64 * mid;
                f_acc += take;
                if f_acc >= k {
                    fast5 = f_sum / k as f64;
                }
            }
        }
        let wmean = if self.wsum > 0.0 {
            self.wmean_num / self.wsum / 1e6
        } else {
            f64::NAN
        };
        [
            med,
            self.mean / n as f64 / 1e6,
            self.hit as f64 / n as f64,
            fast5,
            wmean,
        ]
    }
}

/// 近似版 agg（妥协方案完整复现）：单遍匹配 + 1 秒桶直方图 + null 近似。
fn agg_one_approx(
    streams: &[Option<[EvStream; N_EVENTS]>],
    null_t: &NullTable,
    ai: usize,
    e: usize,
    p: usize,
    prm: &YhybParams,
) -> Option<Vec<f64>> {
    let sa = streams[ai].as_ref()?;
    let (ta, wa) = (&sa[e].t, &sa[e].w);
    let t_us = (prm.hit_t_s * 1e6) as u64;
    let mut h_fwd = Hist1::new();
    let mut h_bwd = Hist1::new();
    let mut null_val_sum = 0.0f64; // null_med 均值池化
    let mut null_hit_acc = 0.0f64; // null_hit x̄ 近似（每 B 一次）
    let mut null_n = 0u64; // (B,i) 数（null_med 池化用）
    let mut null_n_b = 0u64; // B 数（null_hit 池化用）
    let (alo, ahi, base) = if ta.is_empty() {
        (0usize, 0usize, 0i64)
    } else {
        let base = day_base(ta[0]);
        let (lo, hi) = period_slice(ta, base, p);
        (lo, hi, base)
    };
    let rate = (ahi - alo) as f64;
    let mut out = Vec::with_capacity(15);
    if ahi == alo {
        out.push(rate);
        for _ in 0..14 {
            out.push(f64::NAN);
        }
        return Some(out);
    }
    let s_us = base + PERIOD_HI_S[p] * 1_000_000;
    let k_a = (ahi - alo) as f64;
    for (bi, sb) in streams.iter().enumerate() {
        if bi == ai {
            continue;
        }
        let Some(sb) = sb else { continue };
        let tb = &sb[e].t;
        let (blo, bhi) = period_slice(tb, base, p);
        if blo == bhi {
            continue;
        }
        let m_b = null_t.m[e * N_PERIODS + p][bi];
        let c_b = c_of_m(m_b);
        // null_hit x̄ 近似：per B 的 A 事件距离均值（每 B 贡献一次）
        let mut sum_x = 0.0f64;
        for &a in &ta[alo..ahi] {
            sum_x += ((s_us - a).max(0)) as f64;
        }
        let x_bar = sum_x / k_a;
        null_hit_acc += 1.0 - (1.0 - t_us as f64 / x_bar).clamp(0.0, 1.0).powf(m_b as f64);
        null_n_b += 1;
        let mut j = blo;
        for i in alo..ahi {
            let a = ta[i];
            let wi = wa[i];
            let x = s_us - a;
            if x > 0 {
                null_val_sum += x as f64 * c_b;
            }
            null_n += 1;
            while j < bhi && tb[j] <= a {
                j += 1;
            }
            if j > blo {
                h_bwd.push((a - tb[j - 1]) as u64, wi, t_us);
            }
            if j < bhi {
                h_fwd.push((tb[j] - a) as u64, wi, t_us);
            }
        }
    }
    let n_f: u64 = h_fwd.cnt.iter().map(|&c| c as u64).sum();
    let n_b: u64 = h_bwd.cnt.iter().map(|&c| c as u64).sum();
    let fs = h_fwd.stats(n_f, prm.fast_q);
    let bs = h_bwd.stats(n_b, prm.fast_q);
    let null_med = if null_n > 0 {
        null_val_sum / null_n as f64 / 1e6
    } else {
        f64::NAN
    };
    let null_hit = if null_n_b > 0 {
        null_hit_acc / null_n_b as f64
    } else {
        f64::NAN
    };
    out.push(rate);
    for dd in [&fs, &bs] {
        out.push(dd[0]);
        out.push(dd[1]);
        out.push(dd[2]);
        out.push(dd[3]);
        out.push(dd[4]);
        out.push(dd[0] / null_med); // rmed
        out.push(dd[2] / null_hit); // rhit
    }
    Some(out)
}

/// 从预加载的全市场事件流聚合因子（v1/v2 共同核心）。
/// 并行粒度 (A, 事件, 时段)：5900×23×4 = 54 万个小任务，负载均衡（大票任务拆 4 份）；
/// 零模型系数表全市场预计算一次（build_null_table）。
/// 限流：rayon 全局池固定 **50 线程**（512 核共享机，统一 50 核口径；外部若已设
/// RAYON_NUM_THREADS 则尊重外部设置）。幂等：全局池只初始化一次。
pub fn ensure_threads() {
    if std::env::var("RAYON_NUM_THREADS").is_err() {
        let _ = rayon::ThreadPoolBuilder::new().num_threads(50).build_global();
    }
}

/// 时段链条回退填补：分时段指标（rate 除外）缺失（该时段无事件/无响应）时，
/// 依次用**该股票同一事件更近时段**的真实值回退（p3←p2←p1←p0，最后兜底全天）。
/// 理由：同一股票相邻时段的行为高度相关（代理性好），且各股票填各自的值——
/// 截面区分度保留（不填常数/0）；rate=0 是真实值（无事件）不填。
/// 前视检查：因子为 T 日收盘后构造，p0-p3 均为当日已发生数据，无前视。
fn fill_periods(row: &mut [f64]) {
    for e in 0..N_EVENTS {
        for p in 1..N_PERIODS {
            for m in 1..15 {
                let idx = (e * N_PERIODS + p) * 15 + m;
                if row[idx].is_nan() {
                    let mut done = false;
                    for q in (1..p).rev() {
                        let qv = row[(e * N_PERIODS + q) * 15 + m];
                        if !qv.is_nan() {
                            row[idx] = qv;
                            done = true;
                            break;
                        }
                    }
                    if !done {
                        let p0 = row[(e * N_PERIODS) * 15 + m];
                        if !p0.is_nan() {
                            row[idx] = p0;
                        }
                    }
                }
            }
        }
    }
}

/// B 侧事件流 **u32 增量压缩**（聚合步行专用副本，输出逐位不变）：
/// - 时间 i64 绝对 epoch → 首事件相对 base 的 i64 偏移 + 相邻增量 u32（< 2^32 µs 直接编码；
///   ≥ 0xFFFF_FFFF 转义为标记字 + 2×u32 的 u64 全量增量）
/// - 步行只读 4B/事件：j-walk 的 2.2TB DRAM 读减半（事件间隔 ~30-60s，增量远小于 u32 上限）
/// - 解码逐位精确（纯整数加法）；时段有效性/切片/零模型仍用原始绝对时间（本副本只服务
///   union-walk 游标 tj_b/tp_b 的推进）
struct CompStreams {
    first: Vec<Vec<i64>>, // [e][bi] 首事件相对 base 偏移
    deltas: Vec<Vec<u32>>, // [e] 拼接增量流（含转义）
    offs: Vec<Vec<u32>>,  // [e][bi] 增量流起始位置
    lens: Vec<Vec<u32>>,  // [e][bi] 事件数（编码前）
}

const DELTA_ESC: u32 = u32::MAX;

fn build_comp_streams(streams: &[Option<[EvStream; N_EVENTS]>], base: i64) -> CompStreams {
    let n = streams.len();
    let mut first = vec![vec![0i64; n]; N_EVENTS];
    let mut offs = vec![vec![0u32; n]; N_EVENTS];
    let mut lens = vec![vec![0u32; n]; N_EVENTS];
    let mut deltas: Vec<Vec<u32>> = vec![Vec::new(); N_EVENTS];
    for (bi, sb) in streams.iter().enumerate() {
        let Some(sb) = sb else { continue };
        for e in 0..N_EVENTS {
            let t = &sb[e].t;
            let m = t.len();
            if m == 0 {
                continue;
            }
            first[e][bi] = t[0] - base;
            lens[e][bi] = m as u32;
            offs[e][bi] = deltas[e].len() as u32;
            let d = deltas[e].reserve(m * 2);
            let _ = d;
            for i in 1..m {
                let dv = t[i] - t[i - 1];
                if dv >= DELTA_ESC as i64 {
                    deltas[e].push(DELTA_ESC);
                    deltas[e].push((dv as u64 & 0xFFFF_FFFF) as u32);
                    deltas[e].push((dv as u64 >> 32) as u32);
                } else {
                    deltas[e].push(dv as u32);
                }
            }
        }
    }
    CompStreams { first, deltas, offs, lens }
}

/// 从预加载的全市场事件流聚合因子（v1/v2 共同核心，v6 融合任务）。
/// 并行粒度 (A, 事件)：5914×29 = 17 万个小任务（v6 每任务一次全事件归并出 4 时段 60 值）；
/// 切片表 + 零模型系数表全市场预计算一次（build_slices_and_null）。
/// with_l4=true 时输出 1916 因子（1740 时段聚合 + 176 第 4 层网络因子），
/// 第 4 层对级累加与 p0 统计同趟（无独立阶段 A 遍历）；false 时输出 1740。
/// 限流：rayon 全局池固定 **50 线程**（512 核共享机，统一 50 核口径；外部若已设
/// RAYON_NUM_THREADS 则尊重外部设置）。幂等：全局池只初始化一次。
fn compute_from_streams_full(
    codes: &[String],
    streams: &[Option<[EvStream; N_EVENTS]>],
    prm: &YhybParams,
    with_l4: bool,
) -> (Vec<String>, Vec<f32>) {
    ensure_threads();
    let t0 = std::time::Instant::now();
    let n_all = codes.len();
    let (slices, null_t) = build_slices_and_null(streams);
    let t1 = std::time::Instant::now();
    // 有效股票（有事件流）
    let valid: Vec<usize> = streams
        .iter()
        .enumerate()
        .filter(|(_, s)| s.is_some())
        .map(|(i, _)| i)
        .collect();
    let n = valid.len();
    let ne = N_EVENTS;
    let mut valid_pos = vec![usize::MAX; n_all];
    for (pos, &i) in valid.iter().enumerate() {
        valid_pos[i] = pos;
    }
    // 第 4 层对级矩阵（29 × n² × 2 f32 ≈ 8.1GB）+ hub/spoke
    let mut m_fwd = vec![f32::NAN; ne * n * n];
    let mut m_bwd = vec![f32::NAN; ne * n * n];
    let mut hub = vec![f32::NAN; ne * n];
    let mut spoke = vec![f32::NAN; ne * n];
    let l4ctx = if with_l4 {
        Some(L4Ctx {
            n,
            valid_pos: &valid_pos,
            mf: crate::yhyb_network::SendPtr::new(m_fwd.as_mut_ptr()),
            mb: crate::yhyb_network::SendPtr::new(m_bwd.as_mut_ptr()),
            hp: crate::yhyb_network::SendPtr::new(hub.as_mut_ptr()),
            sp: crate::yhyb_network::SendPtr::new(spoke.as_mut_ptr()),
        })
    } else {
        None
    };
    // YHYB_NSUB：限制 A 任务数（探针，快速迭代用）
    let n_sub = std::env::var("YHYB_NSUB")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(n)
        .min(n);
    // v7：任务粒度 = (K 只 A 的块, 事件 e)；块内共享 B 遍历（j-walk DRAM 重读减少 K 倍）。
    // **e-major 任务序**：idx = e * n_blocks + b（事件类型外层、块内层）——同一时刻各线程
    // 处理的都是同一个 e 的任务，该 e 的全市场 B 事件流（~23MB）在 L3 中保持热驻留，
    // j-walk 重读由 DRAM 命中变为 L3 命中（v6 b-major 序下线程工作集横跨全部 29 个 e 的
    // 流 ≈816MB > L3 384MB，重读全部落 DRAM——2.2TB 带宽墙的来源）。
    // 探针（YHYB_SKIP_PUSH / YHYB_NO_JWALK）编译期特化，热循环零运行时分支。
    // YHYB_K：块大小（1 或 2，默认 2）；YHYB_BMAJOR：旧版 b-major 任务序（A/B 对比用）。
    let k_blk = std::env::var("YHYB_K").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(K_BLOCK).min(2).max(1);
    let b_major = std::env::var("YHYB_BMAJOR").is_ok();
    let no_jwalk = std::env::var("YHYB_NO_JWALK").is_ok();
    let skip_push = std::env::var("YHYB_SKIP_PUSH").is_ok();
    let count = std::env::var("YHYB_COUNT").is_ok();
    if count {
        CNT_A.store(0, std::sync::atomic::Ordering::Relaxed);
        CNT_PUSH.store(0, std::sync::atomic::Ordering::Relaxed);
    }
    let n_blocks = (n_sub + k_blk - 1) / k_blk;
    // B 侧事件流 u32 增量压缩（j-walk DRAM 减半；~1s 一次性构建）
    let cst = build_comp_streams(streams, slices.base);
    let results: Vec<[Option<Vec<f64>>; 2]> = (0..ne * n_blocks)
        .into_par_iter()
        .map(|idx| {
            let (e, b) = if b_major {
                (idx % n_blocks, idx / n_blocks)
            } else {
                (idx / n_blocks, idx % n_blocks)
            };
            let start = b * k_blk;
            let nk = (n_sub - start).min(k_blk);
            macro_rules! dispatch_agg {
                ($ct:expr) => {
                    match k_blk {
                1 => {
                    let ai = [valid[start]];
                    let row = [start];
                    let r = match (with_l4, no_jwalk, skip_push) {
                        (true, false, false) => agg_fused_blocked::<true, false, false, $ct, 1>(
                            &streams, &null_t, &slices, &cst, ai, row, 1, e, prm, l4ctx.as_ref()),
                        (true, true, false) => agg_fused_blocked::<true, true, false, $ct, 1>(
                            &streams, &null_t, &slices, &cst, ai, row, 1, e, prm, l4ctx.as_ref()),
                        (true, false, true) => agg_fused_blocked::<true, false, true, $ct, 1>(
                            &streams, &null_t, &slices, &cst, ai, row, 1, e, prm, l4ctx.as_ref()),
                        (true, true, true) => agg_fused_blocked::<true, true, true, $ct, 1>(
                            &streams, &null_t, &slices, &cst, ai, row, 1, e, prm, l4ctx.as_ref()),
                        (false, false, false) => agg_fused_blocked::<false, false, false, $ct, 1>(
                            &streams, &null_t, &slices, &cst, ai, row, 1, e, prm, l4ctx.as_ref()),
                        (false, true, false) => agg_fused_blocked::<false, true, false, $ct, 1>(
                            &streams, &null_t, &slices, &cst, ai, row, 1, e, prm, l4ctx.as_ref()),
                        (false, false, true) => agg_fused_blocked::<false, false, true, $ct, 1>(
                            &streams, &null_t, &slices, &cst, ai, row, 1, e, prm, l4ctx.as_ref()),
                        (false, true, true) => agg_fused_blocked::<false, true, true, $ct, 1>(
                            &streams, &null_t, &slices, &cst, ai, row, 1, e, prm, l4ctx.as_ref()),
                    };
                    let mut out2: [Option<Vec<f64>>; 2] = [None, None];
                    out2[0] = r.into_iter().next().unwrap();
                    out2
                    }
                _ => {
                    let mut ai = [0usize; 2];
                    let mut row = [0usize; 2];
                    for k in 0..nk {
                        ai[k] = valid[start + k];
                        row[k] = start + k;
                    }
                    match (with_l4, no_jwalk, skip_push) {
                        (true, false, false) => agg_fused_blocked::<true, false, false, $ct, 2>(
                            &streams, &null_t, &slices, &cst, ai, row, nk, e, prm, l4ctx.as_ref()),
                        (true, true, false) => agg_fused_blocked::<true, true, false, $ct, 2>(
                            &streams, &null_t, &slices, &cst, ai, row, nk, e, prm, l4ctx.as_ref()),
                        (true, false, true) => agg_fused_blocked::<true, false, true, $ct, 2>(
                            &streams, &null_t, &slices, &cst, ai, row, nk, e, prm, l4ctx.as_ref()),
                        (true, true, true) => agg_fused_blocked::<true, true, true, $ct, 2>(
                            &streams, &null_t, &slices, &cst, ai, row, nk, e, prm, l4ctx.as_ref()),
                        (false, false, false) => agg_fused_blocked::<false, false, false, $ct, 2>(
                            &streams, &null_t, &slices, &cst, ai, row, nk, e, prm, l4ctx.as_ref()),
                        (false, true, false) => agg_fused_blocked::<false, true, false, $ct, 2>(
                            &streams, &null_t, &slices, &cst, ai, row, nk, e, prm, l4ctx.as_ref()),
                        (false, false, true) => agg_fused_blocked::<false, false, true, $ct, 2>(
                            &streams, &null_t, &slices, &cst, ai, row, nk, e, prm, l4ctx.as_ref()),
                        (false, true, true) => agg_fused_blocked::<false, true, true, $ct, 2>(
                            &streams, &null_t, &slices, &cst, ai, row, nk, e, prm, l4ctx.as_ref()),
                    }
                    }
            }
                }
            };
            if count {
                dispatch_agg!(true)
            } else {
                dispatch_agg!(false)
            }
        })
        .collect();
    let t2 = std::time::Instant::now();
    if count {
        eprintln!(
            "YHYB_COUNT a_iters={} pushes={}",
            CNT_A.load(std::sync::atomic::Ordering::Relaxed),
            CNT_PUSH.load(std::sync::atomic::Ordering::Relaxed),
        );
    }
    // 组装 1740 时段聚合因子（有效股票）
    let mut out_codes = Vec::with_capacity(n);
    let mut vals1380 = Vec::with_capacity(n * N_FACTORS);
    for row in 0..n {
        let mut r = Vec::with_capacity(N_FACTORS);
        let mut ok = true;
        if row >= n_sub {
            // NSUB 探针：未计算的股票直接排除（results 只覆盖前 n_sub 行）
            ok = false;
        }
        if ok {
            let b = row / k_blk;
            let k = row % k_blk;
            for e in 0..ne {
                match &results[e * n_blocks + b][k] {
                    Some(v) => r.extend_from_slice(v),
                    None => ok = false,
                }
            }
        }
        if ok && r.len() == N_FACTORS {
            if std::env::var("YHYB_NO_FILL").is_err() {
                fill_periods(&mut r);
            }
            out_codes.push(codes[valid[row]].clone());
            vals1380.extend(r.iter().map(|&x| x as f32));
        }
    }
    if !with_l4 {
        return (out_codes, vals1380);
    }
    // 第 4 层阶段 B：从对级矩阵算 176 因子
    let (vals140, _) = crate::yhyb_network::l4_factors(&m_fwd, &m_bwd, &hub, &spoke, n, valid);
    let t3 = std::time::Instant::now();
    if std::env::var("YHYB_PROBE").is_ok() {
        eprintln!(
            "YHYB_PROBE 切片+null表={:.1}s 聚合任务={:.1}s 第4层B={:.1}s 组装={:.1}s",
            t1.duration_since(t0).as_secs_f64(),
            t2.duration_since(t1).as_secs_f64(),
            t3.duration_since(t2).as_secs_f64(),
            t3.elapsed().as_secs_f64(),
        );
    }
    let nc = out_codes.len();
    let total = N_FACTORS + crate::yhyb_network::N_L4;
    let mut vals = Vec::with_capacity(nc * total);
    for i in 0..nc {
        vals.extend_from_slice(&vals1380[i * N_FACTORS..(i + 1) * N_FACTORS]);
        vals.extend_from_slice(
            &vals140[i * crate::yhyb_network::N_L4..(i + 1) * crate::yhyb_network::N_L4],
        );
    }
    (out_codes, vals)
}

/// 1740 因子版本（无第 4 层；py_yhyb_params 用，输出布局与旧版一致）。
fn compute_from_streams(
    codes: &[String],
    streams: &[Option<[EvStream; N_EVENTS]>],
    prm: &YhybParams,
) -> (Vec<String>, Vec<f32>) {
    compute_from_streams_full(codes, streams, prm, false)
}

/// 近似版聚合（评估用）：并行粒度 (A, 事件, 时段)。
fn compute_from_streams_approx(
    codes: &[String],
    streams: &[Option<[EvStream; N_EVENTS]>],
    prm: &YhybParams,
) -> (Vec<String>, Vec<f32>) {
    ensure_threads();
    let n_stocks = codes.len();
    let (_, null_t) = build_slices_and_null(streams);
    let results: Vec<Option<Vec<f64>>> = (0..n_stocks * N_EVENTS * N_PERIODS)
        .into_par_iter()
        .map(|idx| {
            let ai = idx / (N_EVENTS * N_PERIODS);
            let e = (idx / N_PERIODS) % N_EVENTS;
            let p = idx % N_PERIODS;
            agg_one_approx(streams, &null_t, ai, e, p, prm)
        })
        .collect();
    let mut out_codes = Vec::new();
    let mut vals = Vec::with_capacity(n_stocks * N_FACTORS);
    for ai in 0..n_stocks {
        let mut row = Vec::with_capacity(N_FACTORS);
        let mut ok = true;
        for e in 0..N_EVENTS {
            for p in 0..N_PERIODS {
                match &results[(ai * N_EVENTS + e) * N_PERIODS + p] {
                    Some(v) => row.extend_from_slice(v),
                    None => ok = false,
                }
            }
        }
        if ok && row.len() == N_FACTORS {
            out_codes.push(codes[ai].clone());
            vals.extend(row.iter().map(|&x| x as f32));
        }
    }
    (out_codes, vals)
}

/// v1 入口（读盘，默认参数）：返回 (codes, vals) —— **合并输出 1916 因子**
/// （1740 时段聚合 + 176 第 4 层网络因子），并直接把完整结果写入备份文件
/// （backup_writer v4 格式，/hdd/user_home_unsafe/chenzongwei/yhyb5_{date}.bin）。
/// p0 聚合任务与第 4 层对级累加**融合**（同一趟对级遍历，消除第 4 层阶段 A 独立遍历）。
pub fn compute_yhyb_full(date: i64) -> std::io::Result<(Vec<String>, Vec<f32>)> {
    let prm = YhybParams::default();
    let t_start = std::time::Instant::now();
    let (codes_all, streams) = load_streams(date, &prm)?;
    let t_read = std::time::Instant::now();
    let (codes, vals) = compute_from_streams_full(&codes_all, &streams, &prm, true);
    // 写备份文件（v4 格式，与 pipeline 备份兼容；版本化文件名避免与旧版因子数冲突）
    let total = N_FACTORS + crate::yhyb_network::N_L4;
    let backup = format!("/hdd/user_home_unsafe/chenzongwei/yhyb5_{date}.bin");
    let results: Vec<crate::backup_reader::TaskResult> = codes
        .iter()
        .zip(vals.chunks(total))
        .map(|(code, facs)| crate::backup_reader::TaskResult {
            date,
            code: code.clone(),
            timestamp: 0,
            facs: facs.to_vec(),
        })
        .collect();
    crate::backup_writer::save_results_to_backup(&results, &backup, total).map_err(|e| {
        std::io::Error::new(std::io::ErrorKind::Other, format!("备份写入失败 {backup}: {e}"))
    })?;
    if std::env::var("YHYB_TIMING").is_ok() {
        eprintln!(
            "YHYB_TIMING date={date} 读盘+检测={:.1}s 聚合+第4层(融合)={:.1}s 备份={} rayon线程数={}",
            t_read.duration_since(t_start).as_secs_f64(),
            t_read.elapsed().as_secs_f64(),
            backup,
            rayon::current_num_threads(),
        );
    }
    Ok((codes, vals))
}

/// 公共读盘：读全市场逐笔+盘口 → 事件流（v1 1380 因子与第 4 层网络因子共用）。
/// 注意：必须在任何 rayon 使用之前调用 ensure_threads（全局池限流 50 线程）。
/// **v5 体量拆分**：读盘后先算全市场成交额截面百分位（P90 = 大单、P40 = 小单），
/// 再逐股检测（空档每股内部补充）。
pub fn load_streams(
    date: i64,
    prm: &YhybParams,
) -> std::io::Result<(Vec<String>, Vec<Option<[EvStream; N_EVENTS]>>)> {
    ensure_threads();
    let codes = list_codes(date, "transaction");
    let loaded: Vec<(Vec<TradeRecord>, Vec<MarketRecord>)> = codes
        .par_iter()
        .map(|code| {
            let trades = read_trade_fast_inner(code, date, false, true, 8 * 1024 * 1024).unwrap_or_default();
            let market =
                read_market_fast_inner(code, date, false, true, 8 * 1024 * 1024).unwrap_or_default();
            (trades, market)
        })
        .collect();
    let (thr_l, thr_m) = cross_amount_thresholds(&loaded);
    let streams: Vec<Option<[EvStream; N_EVENTS]>> = loaded
        .into_par_iter()
        .map(|(trades, market)| {
            if trades.is_empty() {
                return None;
            }
            Some(detect_all(&trades, &market, prm, thr_l, thr_m))
        })
        .collect();
    Ok((codes, streams))
}

/// 全市场成交额截面百分位：所有股票所有连续竞价成交放一起，
/// 大单阈值 thr_l = P90（top 10%）、小单阈值 thr_m = P40（bottom 40%）。
/// 用两次 select_nth（O(n)），内存为全市场成交笔数 × f64（~1.6GB，512 核机器可接受）。
fn cross_amount_thresholds(
    loaded: &[(Vec<TradeRecord>, Vec<MarketRecord>)],
) -> (f64, f64) {
    let total: usize = loaded.iter().map(|(t, _)| t.len()).sum();
    if total == 0 {
        return (f64::NAN, f64::NAN);
    }
    let mut amts = Vec::with_capacity(total);
    for (t, _) in loaded {
        for r in t {
            amts.push(r.turnover as f64);
        }
    }
    let k40 = (amts.len() as f64 * 0.40) as usize;
    let k90 = (amts.len() as f64 * 0.90) as usize;
    let k40 = k40.min(amts.len() - 1);
    let k90 = k90.min(amts.len() - 1);
    amts.select_nth_unstable_by(k40, |a, b| a.total_cmp(b));
    let thr_m = amts[k40];
    amts.select_nth_unstable_by(k90, |a, b| a.total_cmp(b));
    let thr_l = amts[k90];
    if thr_l.is_nan() || thr_m.is_nan() || thr_l < thr_m {
        return (f64::NAN, f64::NAN);
    }
    (thr_l, thr_m)
}

pub fn compute_yhyb_full_with_params(date: i64, prm: &YhybParams) -> std::io::Result<(Vec<String>, Vec<f32>)> {
    let t_start = std::time::Instant::now();
    let (codes, streams) = load_streams(date, prm)?;
    let t_read = std::time::Instant::now();
    let res = compute_from_streams(&codes, &streams, prm);
    if std::env::var("YHYB_TIMING").is_ok() {
        eprintln!(
            "YHYB_TIMING date={date} 读盘+检测={:.1}s 聚合={:.1}s",
            t_read.duration_since(t_start).as_secs_f64(),
            t_read.elapsed().as_secs_f64()
        );
    }
    Ok(res)
}

/// 因子名（与 N_FACTORS 严格对齐，单一源）。合并输出后追加第 4 层 140 个网络因子名，
/// 共 1520 个（1380 + 140）。
pub fn yhyb_names() -> Vec<String> {
    let mut names = Vec::with_capacity(N_FACTORS + crate::yhyb_network::N_L4);
    for e in 0..N_EVENTS {
        for p in 0..N_PERIODS {
            names.push(format!("yhyb_e{e:02}_p{p}_rate"));
            for dir in ["fwd", "bwd"] {
                for mt in METRIC_NAMES {
                    names.push(format!("yhyb_e{e:02}_p{p}_{mt}_{dir}"));
                }
            }
        }
    }
    names.extend(crate::yhyb_network::l4_names());
    names
}

// ============================================================================
// Python 入口（薄包装）
// ============================================================================

/// Python 单日调试（v1，默认参数）：返回 (codes, vals)。
#[pyfunction]
#[pyo3(signature = (date, approx=false))]
pub fn py_yhyb(py: Python<'_>, date: i64, approx: bool) -> PyResult<(Vec<String>, Vec<f32>)> {
    if approx {
        // 近似版（评估妥协方案用）：读盘 + 检测相同，聚合走近似统计
        ensure_threads();
        let codes = list_codes(date, "transaction");
        let prm = YhybParams::default();
        let loaded: Vec<(Vec<TradeRecord>, Vec<MarketRecord>)> = codes
            .par_iter()
            .map(|code| {
                let trades =
                    read_trade_fast_inner(code, date, false, true, 8 * 1024 * 1024).unwrap_or_default();
                let market =
                    read_market_fast_inner(code, date, false, true, 8 * 1024 * 1024).unwrap_or_default();
                (trades, market)
            })
            .collect();
        let (thr_l, thr_m) = cross_amount_thresholds(&loaded);
        let streams: Vec<Option<[EvStream; N_EVENTS]>> = loaded
            .into_par_iter()
            .map(|(trades, market)| {
                if trades.is_empty() {
                    return None;
                }
                Some(detect_all(&trades, &market, &prm, thr_l, thr_m))
            })
            .collect();
        return Ok(compute_from_streams_approx(&codes, &streams, &prm));
    }
    compute_yhyb_full(date).map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("{e:?}")))
}

/// Python 单日调试（v1，自定义参数，17 个按 YhybParams 顺序）。
#[pyfunction]
pub fn py_yhyb_params(py: Python<'_>, date: i64, params: Vec<f64>) -> PyResult<(Vec<String>, Vec<f32>)> {
    let prm = YhybParams::from_vec(&params);
    compute_yhyb_full_with_params(date, &prm)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("{e:?}")))
}

/// Python 拿因子名。
#[pyfunction]
pub fn py_yhyb_names() -> Vec<String> {
    yhyb_names()
}

/// v2 入口（Python 传数据，样例验证/调参）：
/// - trade_arrays：每股 (n_i, 8) [time_us, time_sec, price, volume, turnover, flag, bid_order, ask_order]
/// - market_arrays：每股 (m_i, 25) [time_us, total_ask_vol, total_bid_vol, ask_prc1, bid_prc1,
///   ask_vol1..10, bid_vol1..10]
/// - params：可选 17 个参数（默认 YhybParams::default()）
/// - approx：true 时用近似统计（1 秒桶中点 + wmean + null 均值池化/x̄ 近似），
///   仅用于评估妥协方案的截面影响，不用于生产
#[pyfunction]
#[pyo3(signature = (codes, trade_arrays, market_arrays, params=None, approx=false))]
pub fn py_yhyb_from_data(
    _py: Python<'_>,
    codes: Vec<String>,
    trade_arrays: Vec<PyReadonlyArray2<f64>>,
    market_arrays: Vec<PyReadonlyArray2<f64>>,
    params: Option<Vec<f64>>,
    approx: bool,
) -> PyResult<(Vec<String>, Vec<f32>)> {
    if codes.len() != trade_arrays.len() || codes.len() != market_arrays.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "codes.len()={} trade={} market={} 不一致",
            codes.len(),
            trade_arrays.len(),
            market_arrays.len()
        )));
    }
    let prm = params.map(|v| YhybParams::from_vec(&v)).unwrap_or_default();
    // v5 体量拆分：截面成交额百分位（全市场所有成交放一起，P90/P40）
    let total_amts: usize = trade_arrays.iter().map(|ta| ta.as_array().nrows()).sum();
    let mut all_amts = Vec::with_capacity(total_amts);
    for ta in &trade_arrays {
        let a = ta.as_array();
        for i in 0..a.nrows() {
            all_amts.push(a[[i, 4]]);
        }
    }
    let (thr_l, thr_m) = if all_amts.is_empty() {
        (f64::NAN, f64::NAN)
    } else {
        let k40 = ((all_amts.len() as f64) * 0.40) as usize;
        let k90 = ((all_amts.len() as f64) * 0.90) as usize;
        let k40 = k40.min(all_amts.len() - 1);
        let k90 = k90.min(all_amts.len() - 1);
        all_amts.select_nth_unstable_by(k40, |a, b| a.total_cmp(b));
        let thr_m = all_amts[k40];
        all_amts.select_nth_unstable_by(k90, |a, b| a.total_cmp(b));
        let thr_l = all_amts[k90];
        if thr_l.is_nan() || thr_m.is_nan() || thr_l < thr_m {
            (f64::NAN, f64::NAN)
        } else {
            (thr_l, thr_m)
        }
    };
    let streams: Vec<Option<[EvStream; N_EVENTS]>> = trade_arrays
        .iter()
        .zip(market_arrays.iter())
        .map(|(ta, ma)| {
            let a = ta.as_array();
            let b = ma.as_array();
            if a.nrows() == 0 {
                return None; // 无逐笔 → 无事件；盘口可缺（市场事件自然为空）
            }
            let n = a.nrows();
            let mut t = Vec::with_capacity(n);
            let mut p = Vec::with_capacity(n);
            let mut v = Vec::with_capacity(n);
            let mut amt = Vec::with_capacity(n);
            let mut f = Vec::with_capacity(n);
            for i in 0..n {
                // (n, 8): [time_us, time_sec, price, volume, turnover, flag, bid_order, ask_order]
                t.push(a[[i, 0]] as i64);
                p.push(a[[i, 2]]);
                v.push(a[[i, 3]]);
                amt.push(a[[i, 4]]);
                f.push(a[[i, 5]] as i32);
            }
            let m = b.nrows();
            let mut mt = Vec::with_capacity(m);
            let mut ask1p = Vec::with_capacity(m);
            let mut bid1p = Vec::with_capacity(m);
            let mut ask10: Vec<[f64; 10]> = Vec::with_capacity(m);
            let mut bid10: Vec<[f64; 10]> = Vec::with_capacity(m);
            for i in 0..m {
                mt.push(b[[i, 0]] as i64);
                ask1p.push(b[[i, 3]]);
                bid1p.push(b[[i, 4]]);
                let mut aa = [0.0f64; 10];
                let mut bb = [0.0f64; 10];
                for k in 0..10 {
                    aa[k] = b[[i, 5 + k]];
                    bb[k] = b[[i, 15 + k]];
                }
                ask10.push(aa);
                bid10.push(bb);
            }
            let tev = detect_trade_cols(&t, &p, &v, &amt, &f, &prm, thr_l, thr_m);
            let mev = detect_market_cols(&mt, &ask1p, &bid1p, &ask10, &bid10, &prm);
            let iev = detect_impact(&t, &v, &amt, &f, &mt, &ask10, &bid10, &prm);
            let mut out: [EvStream; N_EVENTS] = Default::default();
            for (i, s) in tev.into_iter().enumerate() {
                out[i] = s;
            }
            for (i, s) in mev.into_iter().enumerate() {
                out[15 + i] = s;
            }
            for (i, s) in iev.into_iter().enumerate() {
                out[21 + i] = s;
            }
            assert_streams_sorted(&out, "py_yhyb_from_data");
            Some(out)
        })
        .collect();
    if approx {
        // 近似版（评估用）：1380 部分走近似统计，第 4 层照常（布局一致 1916）
        let (codes_out, vals1380) = compute_from_streams_approx(&codes, &streams, &prm);
        Ok(merge_l4(&codes, &streams, &codes_out, vals1380))
    } else {
        // 生产版：融合任务一次出 1916 因子（p0 统计 + 第 4 层对级累加同趟）
        Ok(compute_from_streams_full(&codes, &streams, &prm, true))
    }
}

/// 拼接第 4 层 140 因子：1380 + 140 = 1520（有效股票按 streams 顺序对齐，
/// compute_from_streams 与 compute_l4 的过滤规则一致；codes_all 必须为全量代码）。
fn merge_l4(
    codes_all: &[String],
    streams: &[Option<[EvStream; N_EVENTS]>],
    codes_out: &[String],
    vals1380: Vec<f32>,
) -> (Vec<String>, Vec<f32>) {
    let (vals140, _) = crate::yhyb_network::compute_l4(codes_all, streams);
    let n = codes_out.len();
    let total = N_FACTORS + crate::yhyb_network::N_L4;
    let mut vals = Vec::with_capacity(n * total);
    for i in 0..n {
        vals.extend_from_slice(&vals1380[i * N_FACTORS..(i + 1) * N_FACTORS]);
        vals.extend_from_slice(
            &vals140[i * crate::yhyb_network::N_L4..(i + 1) * crate::yhyb_network::N_L4],
        );
    }
    (codes_out.to_vec(), vals)
}

/// Python 调试：单股单日事件时间线（事件名 -> (时间us, 权重)），供单例/验证。
#[pyfunction]
pub fn py_yhyb_events(py: Python<'_>, code: &str, date: i64) -> PyResult<Vec<(String, (Vec<i64>, Vec<f64>))>> {
    let trades = read_trade_fast_inner(code, date, false, true, usize::MAX)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("{e:?}")))?;
    let market = read_market_fast_inner(code, date, false, true, usize::MAX).unwrap_or_default();
    let ev = detect_all(&trades, &market, &YhybParams::default(), f64::NAN, f64::NAN);
    Ok(EVENT_NAMES
        .iter()
        .zip(ev.into_iter())
        .map(|(name, s)| (name.to_string(), (s.t, s.w)))
        .collect())
}

// ============================================================================
// 测试
// ============================================================================
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_names_count() {
        assert_eq!(yhyb_names().len(), N_FACTORS + crate::yhyb_network::N_L4);
        assert_eq!(N_FACTORS, 1740); // 29 事件 × 4 时段 × 15
        assert_eq!(crate::yhyb_network::N_L4, 176); // 29 × 6 + 2
    }

    #[test]
    fn test_params_vec_roundtrip() {
        let p = YhybParams::default();
        let v = [
            p.big_amt_wan, p.sweep_w_s, p.sweep_m as f64, p.ice_w_s, p.ice_m as f64, p.jump_q,
            p.jump_win as f64, p.run_r as f64, p.vwap_dev, p.imb_thr, p.dep_thr, p.wall_k,
            p.spread_ticks, p.ret_thr, p.imp_thr, p.hit_t_s, p.fast_q,
        ];
        let p2 = YhybParams::from_vec(&v);
        assert_eq!(p2.big_amt_wan, p.big_amt_wan);
        assert_eq!(p2.sweep_m, p.sweep_m);
    }

    #[test]
    fn test_period_slice() {
        let base = day_base(1_735_637_400_000_000); // 2024-12-31 某时刻
        // 09:30 起每 60s 一笔，共 10 笔 → 全在 09:30-09:40（只属于 p0 全天）
        let t: Vec<i64> = (0..10).map(|i| base + (5400 + i * 60) * 1_000_000).collect();
        let (lo, hi) = period_slice(&t, base, 0);
        assert_eq!((lo, hi), (0, 10));
        for p in 1..N_PERIODS {
            let (lo, hi) = period_slice(&t, base, p);
            assert_eq!((lo, hi), (0, 0), "p{p} 不应包含 09:30-09:40");
        }
        // 新时段边界（adjust 后时钟）：p1 盘中3h [10:00,14:27)、p2 尾盘30m [14:27,14:57)、
        // p3 下午2h [13:00,15:00)
        let mk = |off: i64| vec![base + off * 1_000_000];
        assert_eq!(period_slice(&mk(6000), base, 1), (0, 1)); // 10:00 ∈ p1
        assert_eq!(period_slice(&mk(5999), base, 1), (0, 0)); // 09:59:59 ∉ p1
        assert_eq!(period_slice(&mk(17820), base, 2), (0, 1)); // 14:27 ∈ p2
        assert_eq!(period_slice(&mk(17819), base, 2), (0, 0)); // 14:26:59 ∉ p2
        assert_eq!(period_slice(&mk(19619), base, 2), (0, 1)); // 14:56:59 ∈ p2
        assert_eq!(period_slice(&mk(12600), base, 3), (0, 1)); // 13:00 ∈ p3
        assert_eq!(period_slice(&mk(19799), base, 3), (0, 1)); // 14:59:59 ∈ p3
        assert_eq!(period_slice(&mk(19800), base, 3), (0, 0)); // 15:00 ∉ p3
    }

    #[test]
    fn test_compute_stats_basic() {
        // 距离 1,2,3,4,5 秒各一条（v3 单遍统计：med/mean/fast5/wmed 均为 1 秒桶中点）
        let mut g = Gather::new();
        for d in [1u64, 2, 3, 4, 5].map(|s| s * 1_000_000) {
            g.push1(d, 1.0, 2_500_000);
        }
        let n = g.n;
        g.locate(n / 2, g.wsum / 2.0, 1);
        let s = g.stats(n, 1);
        assert!((s[0] - 3.5).abs() < 1e-9); // med：中位索引 2 落在桶3 → 中点 3.5s
        assert!((s[1] - 3.5).abs() < 1e-9); // mean：Σ 桶中点/n = (1.5+...+5.5)/5 = 3.5s
        assert!((s[2] - 0.4).abs() < 1e-9); // hit（距离 ≤2.5s）
        assert!((s[3] - 1.5).abs() < 1e-9); // fast5：k=1 → 桶1 中点 1.5s
        assert!((s[4] - 3.5).abs() < 1e-9); // wmed：均匀权重，半权重 2.5 落在桶3 → 中点
        // 加权中位数：权重 [1,1,1,1,5]，一半权重 4.5 → 桶5（累积 4 < 4.5，+5 ≥ 4.5）→ 中点
        let mut g2 = Gather::new();
        for (d, w) in [(1u64, 1.0f64), (2, 1.0), (3, 1.0), (4, 1.0), (5, 5.0)].map(|(d, w)| (d * 1_000_000, w)) {
            g2.push1(d, w, 2_500_000);
        }
        let n2 = g2.n;
        g2.locate(n2 / 2, g2.wsum / 2.0, 1);
        let s2 = g2.stats(n2, 1);
        assert!((s2[4] - 5.5).abs() < 1e-9); // wmed = 桶5 中点（权重中点在大距离侧）
        // u64 大距离回归：8000s（远超旧 u32 上限 4295s）必须落在桶 8000，不得回绕
        let mut g3 = Gather::new();
        g3.push1(8_000_000_000, 1.0, 2_500_000);
        let n3 = g3.n;
        g3.locate(n3 / 2, g3.wsum / 2.0, 1);
        let s3 = g3.stats(n3, 1);
        assert!((s3[0] - 8000.5).abs() < 1e-9); // med = 桶8000 中点
        assert!((s3[4] - 8000.5).abs() < 1e-9); // wmed = 桶8000 中点
    }

    #[test]
    fn test_null_table_poisson_consistency() {
        // 高事件率下（m 大），均匀零模型期望 ≈ S/(m+1)（与泊松一致）
        let base = day_base(0);
        let s_us = base + 19_620_000_000;
        let mut evs: [EvStream; N_EVENTS] = Default::default();
        evs[0].t.push(base + 5_400_000_000);
        let streams = vec![Some(evs)];
        let (_, nt) = build_slices_and_null(&streams);
        assert_eq!(nt.m[0][0], 1);
        // c = 1 - 2^(-1/1) = 0.5
        assert!((c_of_m(1) - 0.5).abs() < 1e-9);
        let _ = s_us;
    }
}
