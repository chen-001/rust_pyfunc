//! 关联-差异矩阵（关联与差异指标.md 的衰减事件场交互能量）—— 纯 Rust 高性能实现。
//!
//! 数学定义（见 关联与差异指标.md 第 11 节）:
//!   S_AB = Σ_{i∈A} Σ_{j∈B} w_i·w_j·e^{-|t_i−t_j|/τ}
//! 等价逐笔实现: 每笔成交权重 w_i（6 种设计），按 10ms 时间桶聚合为 u 场，
//! 用"带衰减的滚动累加器"（z 场）做 B-固定流式扫描:
//!   S_{A→B} = Σ_k u_A[k]·g_B[k],  g_B[k] = Σ_{l<k} u_B[l]·e^{-(k−l)Δ/τ}   (B 领先 A)
//! 一次扫描同时算全部 (权重设计 × τ) 组合（AVX2 8-lane 对 A 批处理）。
//!
//! 存储语义: 所有矩阵存"有向" S_dir[i][j] = "i 领先 j 的交互能量"（行主序 N×N f32,
//! 对角 0）。对称交互 = S_dir + S_dirᵀ；同一 10ms 桶内的成交对按 50/50 折半计入
//! 两个方向（S_dir[i][j] 含 +0.5·Σ_k u_i[k]u_j[k]）。
//!
//! 权重设计 w_i（6 族，2026-10：vol 族被 rvol 替换）:
//!   cnt    : w = 1                        （成交同步）
//!   rvol   : w = v / 该股前30秒成交总量    （相对放量同步；无基线时=0）
//!   logvol : w = ln(1+v)                  （大单同步，对数版）
//!   flow   : w = s·√v, s = 主动方向(66买/83卖) （方向性资金流同步）
//!   urg    : w = s·u, u = 订单编号差异常度（每股 z 标准化） （迫切订单同步）
//!   ext    : w = 1(|u|>q95)·s·√v          （极端迫切订单同步，桶级近似）
//! signed 族（flow/urg/ext）拆同向/反向: same（同买同卖）, opp（对手盘）。
//!
//! τ 集合（秒）: cnt {0.05,1,30}, rvol {0.05,1,30}, logvol {0.5,1,3,30},
//!   flow± {0.2,0.5,1,5}, urg± {1,5,30}, ext± {1,5} → 共 28 张 N×N 矩阵。
//!   （cnt 原 9τ / vol 原 6τ；2026-10 经穷举裁剪：弃用 τ 均可被保留档 ρ≥0.97 还原）
//!
//! 性能设计（内存友好原则）:
//!   - Δ=10ms 桶, 全天 1,422,000 桶; 每股票稀疏 (桶, u[10]) 单元（同桶合并）
//!   - 4096 桶分块: 块内建 (k_local, stock, u[9]) 倒排（k 排序）
//!   - B 并行（rayon 50 线程）; 每 B 流式扫桶: 21 个衰减累加器每桶 21 乘+21 加（无 exp）
//!   - 单元循环按 8 个 A 批处理（AVX2）: 31 次 FMA/单元, 结果向量化刷入 m-major 分级缓冲,
//!     块末一次性写回 21 张矩阵（写流量降 ~40 倍, 无逐单元随机 RMW）

use rayon::prelude::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;


pub const BUCKET_US: i64 = 10_000; // 10ms
pub const N_BUCKETS: usize = 1_422_000; // 14220s × 100
pub const BLOCK_BUCKETS: usize = 4096; // 块内桶数 (40.96s)
pub const N_BLOCKS: usize = N_BUCKETS / BLOCK_BUCKETS; // 348

/// 每单元 10 个 u 场: [cnt, vol(√v, 供 flow/ext), logvol, flow+, flow−, urg+, urg−, ext+, ext−, rvol]
/// rvol = v/该股过去30秒成交总量（相对量权重，替换旧 vol 族矩阵；家族位沿用 family=1）。
pub const U_CNT: usize = 0;
pub const U_VOL: usize = 1;
pub const U_LOGVOL: usize = 2;
pub const U_FLOW_P: usize = 3;
pub const U_FLOW_M: usize = 4;
pub const U_URG_P: usize = 5;
pub const U_URG_M: usize = 6;
pub const U_EXT_P: usize = 7;
pub const U_EXT_M: usize = 8;
pub const U_RVOL: usize = 9;
pub const N_U: usize = 10;

/// 矩阵规格（与 MATRIX_NAMES 一一对应）
#[derive(Clone, Copy, Debug)]
pub struct MatrixSpec {
    pub name: &'static str,
    pub family: u8, // 0=cnt 1=rvol 2=logvol 3=flow 4=urg 5=ext
    pub tau: f32,   // 秒
    pub signed: bool,
    pub same: bool, // signed 时: true=同向 false=反向
}

pub const MATRIX_SPECS: [MatrixSpec; 28] = [
    // cnt (w=1): 保留 τ = 0.05/1/30 s（其余 6 档实测可被这三档以 ρ≥0.97 还原）
    MatrixSpec { name: "cnt_t005", family: 0, tau: 0.05, signed: false, same: true },
    MatrixSpec { name: "cnt_t1", family: 0, tau: 1.0, signed: false, same: true },
    MatrixSpec { name: "cnt_t30", family: 0, tau: 30.0, signed: false, same: true },
    // rvol (w = v/过去30秒成交总量, 相对量; 替换旧 vol 族): τ = 0.05/1/30
    MatrixSpec { name: "rvol_t005", family: 1, tau: 0.05, signed: false, same: true },
    MatrixSpec { name: "rvol_t1", family: 1, tau: 1.0, signed: false, same: true },
    MatrixSpec { name: "rvol_t30", family: 1, tau: 30.0, signed: false, same: true },
    // logvol (w=ln(1+v)): τ = 0.5/1/3/30
    MatrixSpec { name: "logvol_t05", family: 2, tau: 0.5, signed: false, same: true },
    MatrixSpec { name: "logvol_t1", family: 2, tau: 1.0, signed: false, same: true },
    MatrixSpec { name: "logvol_t3", family: 2, tau: 3.0, signed: false, same: true },
    MatrixSpec { name: "logvol_t30", family: 2, tau: 30.0, signed: false, same: true },
    // flow (w=s·√v): τ = 0.2/0.5/1/5, same/opp
    MatrixSpec { name: "flow_same_t02", family: 3, tau: 0.2, signed: true, same: true },
    MatrixSpec { name: "flow_opp_t02", family: 3, tau: 0.2, signed: true, same: false },
    MatrixSpec { name: "flow_same_t05", family: 3, tau: 0.5, signed: true, same: true },
    MatrixSpec { name: "flow_opp_t05", family: 3, tau: 0.5, signed: true, same: false },
    MatrixSpec { name: "flow_same_t1", family: 3, tau: 1.0, signed: true, same: true },
    MatrixSpec { name: "flow_opp_t1", family: 3, tau: 1.0, signed: true, same: false },
    MatrixSpec { name: "flow_same_t5", family: 3, tau: 5.0, signed: true, same: true },
    MatrixSpec { name: "flow_opp_t5", family: 3, tau: 5.0, signed: true, same: false },
    // urg (w=s·u): τ = 1/5/30, same/opp
    MatrixSpec { name: "urg_same_t1", family: 4, tau: 1.0, signed: true, same: true },
    MatrixSpec { name: "urg_opp_t1", family: 4, tau: 1.0, signed: true, same: false },
    MatrixSpec { name: "urg_same_t5", family: 4, tau: 5.0, signed: true, same: true },
    MatrixSpec { name: "urg_opp_t5", family: 4, tau: 5.0, signed: true, same: false },
    MatrixSpec { name: "urg_same_t30", family: 4, tau: 30.0, signed: true, same: true },
    MatrixSpec { name: "urg_opp_t30", family: 4, tau: 30.0, signed: true, same: false },
    // ext (极端迫切): τ = 1/5, same/opp
    MatrixSpec { name: "ext_same_t1", family: 5, tau: 1.0, signed: true, same: true },
    MatrixSpec { name: "ext_opp_t1", family: 5, tau: 1.0, signed: true, same: false },
    MatrixSpec { name: "ext_same_t5", family: 5, tau: 5.0, signed: true, same: true },
    MatrixSpec { name: "ext_opp_t5", family: 5, tau: 5.0, signed: true, same: false },
];

pub const N_MATRICES: usize = MATRIX_SPECS.len();

/// 权重家族位（MATRIX_SPECS.family 的顺序：0=cnt 1=rvol 2=logvol 3=flow 4=urg 5=ext）
pub const FAM_CNT: u8 = 1 << 0;
pub const FAM_RVOL: u8 = 1 << 1;
pub const FAM_LOGVOL: u8 = 1 << 2;
pub const FAM_FLOW: u8 = 1 << 3;
pub const FAM_URG: u8 = 1 << 4;
pub const FAM_EXT: u8 = 1 << 5;
pub const FAM_ALL: u8 = FAM_CNT | FAM_RVOL | FAM_LOGVOL | FAM_FLOW | FAM_URG | FAM_EXT;

/// 请求的矩阵名 → 需要计算的家族位。
pub fn family_mask_of(names: &[String]) -> u8 {
    let mut m = 0u8;
    for n in names {
        if let Some(sp) = MATRIX_SPECS.iter().find(|sp| sp.name == n.as_str()) {
            m |= 1 << sp.family;
        }
    }
    m
}

/// 每股预处理结果: 稀疏桶单元 + 统计量
#[derive(Clone, Debug)]
pub struct StockPrep {
    pub code: String,
    /// (桶idx, u[9])，按桶升序，同桶已合并
    pub cells: Vec<(u32, [f32; N_U])>,
    /// 方向拆分的桶单元，桶号与 `cells` 一一对应:
    /// `[cnt_P, cnt_M, vol_P, vol_M]`（买笔计数/卖笔计数/买笔√v/卖笔√v）。
    /// 只给「玉佩2 配对」的主买/主卖拆分用；28 张矩阵仍走 `cells`（不区分方向）。
    /// 方向未知的笔（flag 既非 66 也非 83）不进这四个场。
    pub cells_pm: Vec<(u32, [f32; 4])>,
    pub n_trades: usize,
    pub amount: f64,
    pub total_vol: f64,
    pub imb: f64,
    pub ret: f64,
    pub vol30: f64,
    pub vwap: f64,
    pub q95u: f64,
    /// 每族权重绝对值总和（供零模型/盈余归一化用），序同 family
    pub sum_w: [f64; 6],
}

impl StockPrep {
    /// 空预处理（有逐笔文件但读不到 / 0 笔成交 / 全部成交在交易时段外）：
    /// 保留在 universe 中占一行，cells 为空 → 所有矩阵行/列自然全 0，统计量为 NaN。
    pub fn empty(code: String) -> Self {
        StockPrep {
            code,
            cells: Vec::new(),
            cells_pm: Vec::new(),
            n_trades: 0,
            amount: f64::NAN,
            total_vol: f64::NAN,
            imb: f64::NAN,
            ret: f64::NAN,
            vol30: f64::NAN,
            vwap: f64::NAN,
            q95u: f64::NAN,
            sum_w: [f64::NAN; 6],
        }
    }
}

/// 每股统计量汇总（供降维指标用，也写盘备份）
#[derive(Clone, Copy, Debug, Default)]
pub struct StockStats {
    pub n_trades: f64,
    pub amount: f64,
    pub total_vol: f64,
    pub imb: f64,
    pub ret: f64,
    pub vol30: f64,
    pub vwap: f64,
    pub q95u: f64,
    pub sum_w_cnt: f64,
    pub sum_w_rvol: f64,
    pub sum_w_logvol: f64,
    pub sum_w_flow: f64,
    pub sum_w_urg: f64,
    pub sum_w_ext: f64,
}

/// 每股预处理（并行调用）: 读逐笔 → 权重 → 稀疏桶单元 + 统计量
pub fn prep_stock(
    code: &str,
    recs: &[crate::fast_csv_reader::TradeRecord],
    day_start_us: i64,
    min_trades: usize,
) -> Option<StockPrep> {
    let n = recs.len();
    if n < min_trades {
        return None;
    }
    // ---- pass 1: 订单编号差异 r=(bid−ask)/(|bid|+|ask|) 的均值/标准差 ----
    let mut rs = Vec::with_capacity(n);
    let (mut sum_r, mut sum_r2) = (0.0f64, 0.0f64);
    for r in recs {
        let denom = (r.bid_order.abs() + r.ask_order.abs()) as f64;
        let rr = if denom > 0.0 {
            (r.bid_order - r.ask_order) as f64 / denom
        } else {
            0.0
        };
        rs.push(rr);
        sum_r += rr;
        sum_r2 += rr * rr;
    }
    let mean_r = sum_r / n as f64;
    let var_r = (sum_r2 - sum_r * sum_r / n as f64) / n as f64;
    let std_r = if var_r > 0.0 { var_r.sqrt() } else { 0.0 };

    // ---- pass 1b: 30 秒滑窗基线（rvol 权重分母）----
    // base[i] = 第 i 笔成交之前、时间戳落在 (time_i - 30s, time_i) 内的本股成交量之和（不含本笔）。
    // 笔间双指针滑动，窗口内求和；时间轴与桶索引同源（adjust_afternoon 压缩后的连续轴）。
    // 开盘第一笔 base=0 → rvol=0：基线不存在时权重归零，意即"该股此刻没有参照系"。
    const RVOL_WIN_US: i64 = 30_000_000; // 30 秒
    let mut bases: Vec<f64> = Vec::with_capacity(n);
    {
        let mut lo = 0usize;
        let mut running = 0.0f64;
        for (i, r) in recs.iter().enumerate() {
            let thr = r.time_us - RVOL_WIN_US;
            while lo < i && recs[lo].time_us <= thr {
                running -= (recs[lo].volume as f64).max(0.0);
                lo += 1;
            }
            bases.push(running);
            running += (r.volume as f64).max(0.0);
        }
    }

    // ---- pass 2: 每笔临时权重（桶idx, u, sv, s, rv）----
    struct Tmp {
        bidx: u32,
        u: f32,
        sv: f32,
        lv: f32,
        s: f32,
        rv: f32,
    }
    let mut tmp: Vec<Tmp> = Vec::with_capacity(n);
    let mut absus: Vec<f32> = Vec::with_capacity(n);
    let (mut sum_w_cnt, mut sum_w_logvol, mut sum_w_flow, mut sum_w_urg) =
        (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    let mut sum_w_rvol = 0.0f64;
    let (mut first_p, mut last_p) = (0.0f64, 0.0f64);
    let (mut total_vol, mut amount, mut sgn_vol) = (0.0f64, 0.0f64, 0.0f64);
    for (i, r) in recs.iter().enumerate() {
        let off = r.time_us - day_start_us;
        if off < 0 || off >= N_BUCKETS as i64 * BUCKET_US {
            continue;
        }
        let bidx = (off / BUCKET_US) as u32;
        let v = (r.volume as f64).max(0.0);
        let sv = v.sqrt() as f32;
        let lv = (1.0 + v).ln() as f32;
        let s: f32 = match r.flag {
            66 => 1.0,
            83 => -1.0,
            _ => 0.0,
        };
        let u = if std_r > 0.0 {
            ((rs[i] - mean_r) / std_r) as f32
        } else {
            0.0
        };
        // rvol = v/base，与沙箱版同形（sv² 还原 v，与已验证版本逐位一致）
        let rv = if bases[i] > 0.0 {
            ((sv * sv) as f64) / bases[i]
        } else {
            0.0
        } as f32;
        tmp.push(Tmp { bidx, u, sv, lv, s, rv });
        absus.push(u.abs());
        sum_w_cnt += 1.0;
        sum_w_rvol += rv as f64;
        sum_w_logvol += lv as f64;
        sum_w_flow += sv as f64;
        sum_w_urg += (s * u).abs() as f64;
        if first_p == 0.0 {
            first_p = r.price as f64;
        }
        last_p = r.price as f64;
        total_vol += v;
        amount += (r.turnover as f64).max(0.0);
        sgn_vol += (s as f64) * v;
    }
    if tmp.is_empty() {
        return None;
    }
    // ---- pass 3: q95(|u|) → 组装稀疏桶单元（含 ext 阈值判定）----
    absus.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let q95u = absus[(((absus.len() as f64) * 0.95) as usize).min(absus.len() - 1)] as f64;
    let mut cells: Vec<(u32, [f32; N_U])> = Vec::with_capacity(tmp.len());
    // 方向拆分场：[cnt_P, cnt_M, vol_P, vol_M]，桶号与 cells 一一对应。
    // 同一笔只进 P 或 M 一路（flag 66 买 / 83 卖；其它 flag 两边都不进）。
    let mut cells_pm: Vec<(u32, [f32; 4])> = Vec::with_capacity(tmp.len());
    let mut sum_w_ext = 0.0f64;
    for t in tmp {
        let mut u9 = [0.0f32; N_U];
        u9[U_CNT] = 1.0;
        u9[U_VOL] = t.sv;
        u9[U_LOGVOL] = t.lv;
        u9[U_RVOL] = t.rv;
        let mut u4 = [0.0f32; 4];
        if t.s > 0.0 {
            u4[0] = 1.0; // cnt_P
            u4[2] = t.sv; // vol_P
        } else if t.s < 0.0 {
            u4[1] = 1.0; // cnt_M
            u4[3] = t.sv; // vol_M
        }
        let w_urg = t.s * t.u;
        if t.s > 0.0 {
            u9[U_FLOW_P] = t.sv;
            u9[U_URG_P] = w_urg;
            if t.u.abs() as f64 > q95u {
                u9[U_EXT_P] = t.sv;
            }
        } else if t.s < 0.0 {
            u9[U_FLOW_M] = t.sv;
            u9[U_URG_M] = -w_urg;
            if t.u.abs() as f64 > q95u {
                u9[U_EXT_M] = t.sv;
            }
        }
        sum_w_ext += (u9[U_EXT_P] + u9[U_EXT_M]) as f64;
        if let Some((bk, ua)) = cells.last_mut() {
            if *bk == t.bidx {
                for k in 0..N_U {
                    ua[k] += u9[k];
                }
                let ub = &mut cells_pm.last_mut().expect("cells_pm 与 cells 同长").1;
                for k in 0..4 {
                    ub[k] += u4[k];
                }
                continue;
            }
        }
        cells.push((t.bidx, u9));
        cells_pm.push((t.bidx, u4));
    }
    let n_kept = cells
        .iter()
        .map(|(_, ua)| ua[U_CNT] as usize)
        .sum::<usize>();
    let vol30 = bucket_vol30(recs, day_start_us);
    Some(StockPrep {
        code: code.to_string(),
        cells,
        cells_pm,
        n_trades: n_kept,
        amount,
        total_vol,
        imb: if total_vol > 0.0 {
            sgn_vol / total_vol
        } else {
            0.0
        },
        ret: if first_p > 0.0 {
            last_p / first_p - 1.0
        } else {
            0.0
        },
        vol30,
        vwap: if total_vol > 0.0 {
            amount / total_vol
        } else {
            0.0
        },
        q95u,
        sum_w: [
            sum_w_cnt,
            sum_w_rvol,
            sum_w_logvol,
            sum_w_flow,
            sum_w_urg,
            sum_w_ext,
        ],
    })
}

/// 30s 桶收益标准差（last price per 30s 桶）
fn bucket_vol30(recs: &[crate::fast_csv_reader::TradeRecord], day_start_us: i64) -> f64 {
    const S30: usize = 474;
    let mut last_p = vec![0.0f64; S30];
    for r in recs {
        let off = r.time_us - day_start_us;
        if off < 0 {
            continue;
        }
        let i = (off / 30_000_000) as usize;
        if i < S30 {
            last_p[i] = r.price as f64;
        }
    }
    let mut rets = Vec::with_capacity(S30);
    let mut prev = 0.0f64;
    for &p in last_p.iter() {
        if p > 0.0 {
            if prev > 0.0 {
                rets.push(p / prev - 1.0);
            }
            prev = p;
        }
    }
    if rets.len() < 2 {
        return 0.0;
    }
    let m = rets.iter().sum::<f64>() / rets.len() as f64;
    let v = rets.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / rets.len() as f64;
    v.sqrt()
}

#[inline]
fn decay(tau: f32) -> f32 {
    (-(BUCKET_US as f32) / (tau * 1e6)).exp()
}

/// 全市场矩阵计算（B-tile 并行 + 块内 gp 场物化 + A-major 寄存器累积）。
///
/// 结构（每块）:
///   phase 1: 对 tile 内 64 只 B 流式扫桶, 把 21 个衰减 g 场物化为 gp[4096][21][64]
///            （L3 驻留, 每块每 tile ~22MB）
///   phase 2: A-major 循环: 每只 A 把 S[A][tile] 列段读入 accA[64][21]（L1）,
///            遍历 A 块内单元做 31 FMA/单元, 块末一次性写回 —— S 写流量降 ~300 倍
/// 并行: 块串行（348）× tile 并行（87 个, rayon）
/// 返回 21 张 N×N f32 有向矩阵（行主序, 对角 0）与每股统计量。

/// pack 宽度（A-major [a][48]）：6 个 8 通道块 = cnt(3) + rvol(3) + logvol(4) + flow±(8) + urg±(6) + ext±(4)
const PACK_W: usize = 48;
/// gp 槽步长（每桶 48 槽）
const GP_STRIDE: usize = 48;

#[cfg(target_arch = "x86_64")]
#[allow(clippy::too_many_arguments)]
fn phase1_simd<const NEED: u8>(
    gp: &mut [f32],
    acc_mem: &mut [f32],
    block_buf: &[(u32, [f32; N_U])],
    bcnt: usize,
    boff: usize,
    k0: u32
) {
    unsafe {
        // B 块内单元游标（块局部）
        let mut b_curs = 0usize;
        // B 的桶内 u
        let mut ub = [0.0f32; N_U];
        let zero = _mm256_setzero_ps();
        let half = _mm256_set1_ps(0.5);
        // 衰减系数按 48 槽打包（6 块 × 8 槽，未用槽 0 → acc 恒 0）
        // cnt 与 rvol 用同一 τ 网格 {0.05,1,30}，共用 d0
        let d0 = _mm256_setr_ps(
            decay(0.05),
            decay(1.0),
            decay(30.0),
            0.0, 0.0, 0.0, 0.0, 0.0,
        );
        let d1 = _mm256_setr_ps(
            decay(0.5),
            decay(1.0),
            decay(3.0),
            decay(30.0),
            0.0, 0.0, 0.0, 0.0,
        );
        let d2 = _mm256_setr_ps(
            decay(0.2),
            decay(0.2),
            decay(0.5),
            decay(0.5),
            decay(1.0),
            decay(1.0),
            decay(5.0),
            decay(5.0),
        );
        let d3 = _mm256_setr_ps(
            decay(1.0),
            decay(1.0),
            decay(5.0),
            decay(5.0),
            decay(30.0),
            decay(30.0),
            0.0,
            0.0,
        );
        let d4 = _mm256_setr_ps(
            decay(1.0),
            decay(1.0),
            decay(5.0),
            decay(5.0),
            0.0, 0.0, 0.0, 0.0,
        );
        let aptr = acc_mem.as_ptr();
        let mut a0 = _mm256_loadu_ps(aptr);
        let mut a1 = _mm256_loadu_ps(aptr.add(8));
        let mut a2 = _mm256_loadu_ps(aptr.add(16));
        let mut a3 = _mm256_loadu_ps(aptr.add(24));
        let mut a4 = _mm256_loadu_ps(aptr.add(32));
        let mut a5 = _mm256_loadu_ps(aptr.add(40));
        for k in 0..BLOCK_BUCKETS {
            for f in 0..N_U {
                ub[f] = 0.0;
            }
            while b_curs < bcnt
                && (block_buf[boff + b_curs].0 - k0) as usize == k
            {
                let c = &block_buf[boff + b_curs];
                for f in 0..N_U {
                    ub[f] += c.1[f];
                }
                b_curs += 1;
            }
            let gpk = gp.as_mut_ptr().add(k * GP_STRIDE);
            // 只算请求的家族：被跳过的家族 gp 槽保持 0（块首已清零），只喂给不被请求的矩阵。
            if NEED & FAM_CNT != 0 {
                let uc = _mm256_blend_ps(_mm256_set1_ps(ub[U_CNT]), zero, 0b11111000);
                let g0 = _mm256_add_ps(a0, _mm256_mul_ps(half, uc));
                _mm256_storeu_ps(gpk, g0);
                a0 = _mm256_mul_ps(_mm256_add_ps(a0, uc), d0);
            }
            if NEED & FAM_RVOL != 0 {
                let ur = _mm256_blend_ps(_mm256_set1_ps(ub[U_RVOL]), zero, 0b11111000);
                let g1 = _mm256_add_ps(a1, _mm256_mul_ps(half, ur));
                _mm256_storeu_ps(gpk.add(8), g1);
                a1 = _mm256_mul_ps(_mm256_add_ps(a1, ur), d0);
            }
            if NEED & FAM_LOGVOL != 0 {
                let ul = _mm256_blend_ps(_mm256_set1_ps(ub[U_LOGVOL]), zero, 0b11110000);
                let g2 = _mm256_add_ps(a2, _mm256_mul_ps(half, ul));
                _mm256_storeu_ps(gpk.add(16), g2);
                a2 = _mm256_mul_ps(_mm256_add_ps(a2, ul), d1);
            }
            if NEED & FAM_FLOW != 0 {
                let u3 = _mm256_unpacklo_ps(
                    _mm256_set1_ps(ub[U_FLOW_P]),
                    _mm256_set1_ps(ub[U_FLOW_M]),
                );
                let g3 = _mm256_add_ps(a3, _mm256_mul_ps(half, u3));
                _mm256_storeu_ps(gpk.add(24), g3);
                a3 = _mm256_mul_ps(_mm256_add_ps(a3, u3), d2);
            }
            if NEED & FAM_URG != 0 {
                let u4 = _mm256_blend_ps(
                    _mm256_unpacklo_ps(
                        _mm256_set1_ps(ub[U_URG_P]),
                        _mm256_set1_ps(ub[U_URG_M]),
                    ),
                    zero,
                    0b11000000,
                );
                let g4 = _mm256_add_ps(a4, _mm256_mul_ps(half, u4));
                _mm256_storeu_ps(gpk.add(32), g4);
                a4 = _mm256_mul_ps(_mm256_add_ps(a4, u4), d3);
            }
            if NEED & FAM_EXT != 0 {
                let u5 = _mm256_blend_ps(
                    _mm256_unpacklo_ps(
                        _mm256_set1_ps(ub[U_EXT_P]),
                        _mm256_set1_ps(ub[U_EXT_M]),
                    ),
                    zero,
                    0b11110000,
                );
                let g5 = _mm256_add_ps(a5, _mm256_mul_ps(half, u5));
                _mm256_storeu_ps(gpk.add(40), g5);
                a5 = _mm256_mul_ps(_mm256_add_ps(a5, u5), d4);
            }
        }
        // 块末 store 回 acc（跨块持久）
        let amut = acc_mem.as_mut_ptr();
        _mm256_storeu_ps(amut, a0);
        _mm256_storeu_ps(amut.add(8), a1);
        _mm256_storeu_ps(amut.add(16), a2);
        _mm256_storeu_ps(amut.add(24), a3);
        _mm256_storeu_ps(amut.add(32), a4);
        _mm256_storeu_ps(amut.add(40), a5);
    }
}
#[cfg(any(not(target_arch = "x86_64"), test))]
#[allow(clippy::too_many_arguments, dead_code)]
fn phase1_scalar<const _NEED: u8>(
    gp: &mut [f32],
    acc_mem: &mut [f32],
    block_buf: &[(u32, [f32; N_U])],
    bcnt: usize,
    boff: usize,
    k0: u32
) {
    {
        // 标量回退：f32 运算与 x86_64 SIMD 版逐槽同序（slot 序固定、无并行折叠），
        // 结果逐位一致；仅速度较慢（本文件仅被 yupei_dist 全市场场景调用）。
        // acc 跨块持久（packs 内存段; 块开始 load, 块末 store 回）
        let amut = acc_mem.as_mut_ptr();
        // 每槽独立 acc（7 块 × 8 槽；槽 8..16 仅用槽 8 标量）
        let mut a0: [f32; 8] = [0.0; 8];
        let mut a1: [f32; 8] = [0.0; 8];
        let mut a2: [f32; 8] = [0.0; 8];
        let mut a3: [f32; 8] = [0.0; 8];
        let mut a4: [f32; 8] = [0.0; 8];
        let mut a5: [f32; 8] = [0.0; 8];
        for s in 0..8 {
            a0[s] = unsafe { *amut.add(s) };
            a1[s] = unsafe { *amut.add(8 + s) };
            a2[s] = unsafe { *amut.add(16 + s) };
            a3[s] = unsafe { *amut.add(24 + s) };
            a4[s] = unsafe { *amut.add(32 + s) };
            a5[s] = unsafe { *amut.add(40 + s) };
        }
        // 衰减系数（与 SIMD 版同源；cnt/rvol 共用 d0）
        let d0: [f32; 8] = [
            decay(0.05), decay(1.0), decay(30.0), 0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        let d1: [f32; 8] = [
            decay(0.5), decay(1.0), decay(3.0), decay(30.0), 0.0, 0.0, 0.0, 0.0,
        ];
        let d2: [f32; 8] = [
            decay(0.2), decay(0.2), decay(0.5), decay(0.5),
            decay(1.0), decay(1.0), decay(5.0), decay(5.0),
        ];
        let d3: [f32; 8] = [
            decay(1.0), decay(1.0), decay(5.0), decay(5.0),
            decay(30.0), decay(30.0), 0.0, 0.0,
        ];
        let d4: [f32; 8] = [
            decay(1.0), decay(1.0), decay(5.0), decay(5.0),
            0.0, 0.0, 0.0, 0.0,
        ];
        let mut b_curs = 0usize;
        let mut ub = [0.0f32; N_U];
        for k in 0..BLOCK_BUCKETS {
            for f in 0..N_U {
                ub[f] = 0.0;
            }
            while b_curs < bcnt
                && (block_buf[boff + b_curs].0 - k0) as usize == k
            {
                let c = &block_buf[boff + b_curs];
                for f in 0..N_U {
                    ub[f] += c.1[f];
                }
                b_curs += 1;
            }
            let gpk = unsafe { gp.as_mut_ptr().add(k * GP_STRIDE) };
            // cnt（槽 0..2）
            for s in 0..8 {
                let u = if s < 3 { ub[U_CNT] } else { 0.0 };
                unsafe {
                    *gpk.add(s) = a0[s] + 0.5 * u;
                }
                a0[s] = (a0[s] + u) * d0[s];
            }
            // rvol（槽 8..10）
            for s in 0..8 {
                let u = if s < 3 { ub[U_RVOL] } else { 0.0 };
                unsafe {
                    *gpk.add(8 + s) = a1[s] + 0.5 * u;
                }
                a1[s] = (a1[s] + u) * d0[s];
            }
            // logvol（槽 16..19）
            for s in 0..8 {
                let u = if s < 4 { ub[U_LOGVOL] } else { 0.0 };
                unsafe {
                    *gpk.add(16 + s) = a2[s] + 0.5 * u;
                }
                a2[s] = (a2[s] + u) * d1[s];
            }
            // flow ±（槽 24..31: 偶数槽 P、奇数槽 M, 各 4 槽）
            for s in 0..8 {
                let u = match s {
                    0 | 2 | 4 | 6 => ub[U_FLOW_P],
                    1 | 3 | 5 | 7 => ub[U_FLOW_M],
                    _ => 0.0,
                };
                unsafe {
                    *gpk.add(24 + s) = a3[s] + 0.5 * u;
                }
                a3[s] = (a3[s] + u) * d2[s];
            }
            // urg ±（槽 32..37）
            for s in 0..8 {
                let u = if s < 6 {
                    if s % 2 == 0 {
                        ub[U_URG_P]
                    } else {
                        ub[U_URG_M]
                    }
                } else {
                    0.0
                };
                unsafe {
                    *gpk.add(32 + s) = a4[s] + 0.5 * u;
                }
                a4[s] = (a4[s] + u) * d3[s];
            }
            // ext ±（槽 40..43）
            for s in 0..8 {
                let u = if s < 4 {
                    if s % 2 == 0 {
                        ub[U_EXT_P]
                    } else {
                        ub[U_EXT_M]
                    }
                } else {
                    0.0
                };
                unsafe {
                    *gpk.add(40 + s) = a5[s] + 0.5 * u;
                }
                a5[s] = (a5[s] + u) * d4[s];
            }
        }
        // 块末 store 回 acc（跨块持久）
        for s in 0..8 {
            unsafe {
                *amut.add(s) = a0[s];
                *amut.add(8 + s) = a1[s];
                *amut.add(16 + s) = a2[s];
                *amut.add(24 + s) = a3[s];
                *amut.add(32 + s) = a4[s];
                *amut.add(40 + s) = a5[s];
            }
        }
    }
}
#[cfg(target_arch = "x86_64")]
#[allow(clippy::too_many_arguments)]
fn phase2_simd<const NEED: u8>(
    pack: &mut [f32],
    gp: &[f32],
    block_buf: &[(u32, [f32; N_U])],
    counts: &[usize],
    offs: &[usize],
    n: usize,
    k0: u32
) {
        let pack_ptr = pack.as_mut_ptr();
        let gpk0 = gp.as_ptr();
        const SUB: usize = 2;
        let sub_buckets = BLOCK_BUCKETS / SUB;
    {
    // ± 通道交换索引（flow/urg/ext 组内相邻通道互换）
    let idx_x = unsafe { _mm256_setr_epi32(1, 0, 3, 2, 5, 4, 7, 6) };
    for sub in 0..SUB {
        let k_lo = sub * sub_buckets;
        let k_hi = k_lo + sub_buckets;
        for a in 0..n {
            let ca = counts[a];
            if ca == 0 {
                continue;
            }
            let base = a * PACK_W;
            unsafe {
                let ca0 = offs[a];
                let mut j0 = 0usize;
                while j0 < ca && ((block_buf[ca0 + j0].0 - k0) as usize) < k_lo {
                    j0 += 1;
                }
                if j0 >= ca || ((block_buf[ca0 + j0].0 - k0) as usize) >= k_hi {
                    continue;
                }
                let mut p0 = if NEED & FAM_CNT != 0 { _mm256_loadu_ps(pack_ptr.add(base)) } else { _mm256_setzero_ps() };
                let mut p1 = if NEED & FAM_RVOL != 0 { _mm256_loadu_ps(pack_ptr.add(base + 8)) } else { _mm256_setzero_ps() };
                let mut p2 = if NEED & FAM_LOGVOL != 0 { _mm256_loadu_ps(pack_ptr.add(base + 16)) } else { _mm256_setzero_ps() };
                let mut p3 = if NEED & FAM_FLOW != 0 { _mm256_loadu_ps(pack_ptr.add(base + 24)) } else { _mm256_setzero_ps() };
                let mut p4 = if NEED & FAM_URG != 0 { _mm256_loadu_ps(pack_ptr.add(base + 32)) } else { _mm256_setzero_ps() };
                let mut p5 = if NEED & FAM_EXT != 0 { _mm256_loadu_ps(pack_ptr.add(base + 40)) } else { _mm256_setzero_ps() };
                let mut j = j0;
                while j < ca {
                    let cell = &block_buf[ca0 + j];
                    let kl = (cell.0 - k0) as usize;
                    if kl >= k_hi {
                        break;
                    }
                    let ua = &cell.1;
                    let gpk = gpk0.add(kl * GP_STRIDE);
                    // 只算请求的家族：被跳过的家族不读 gp、不做 FMA。
                    if NEED & FAM_CNT != 0 {
                        let uc = _mm256_set1_ps(ua[U_CNT]);
                        let g0 = _mm256_loadu_ps(gpk);
                        p0 = _mm256_fmadd_ps(uc, g0, p0);
                    }
                    if NEED & FAM_RVOL != 0 {
                        let ur = _mm256_set1_ps(ua[U_RVOL]);
                        let g1 = _mm256_loadu_ps(gpk.add(8));
                        p1 = _mm256_fmadd_ps(ur, g1, p1);
                    }
                    if NEED & FAM_LOGVOL != 0 {
                        let ul = _mm256_set1_ps(ua[U_LOGVOL]);
                        let g2 = _mm256_loadu_ps(gpk.add(16));
                        p2 = _mm256_fmadd_ps(ul, g2, p2);
                    }
                    if NEED & FAM_FLOW != 0 {
                        let fp = _mm256_set1_ps(ua[U_FLOW_P]);
                        let fm = _mm256_set1_ps(ua[U_FLOW_M]);
                        let g3 = _mm256_loadu_ps(gpk.add(24));
                        let g3x = _mm256_permutevar8x32_ps(g3, idx_x);
                        p3 = _mm256_fmadd_ps(fp, g3, _mm256_fmadd_ps(fm, g3x, p3));
                    }
                    if NEED & FAM_URG != 0 {
                        let up = _mm256_set1_ps(ua[U_URG_P]);
                        let um = _mm256_set1_ps(ua[U_URG_M]);
                        let g4 = _mm256_loadu_ps(gpk.add(32));
                        let g4x = _mm256_permutevar8x32_ps(g4, idx_x);
                        p4 = _mm256_fmadd_ps(up, g4, _mm256_fmadd_ps(um, g4x, p4));
                    }
                    if NEED & FAM_EXT != 0 {
                        let xp = _mm256_set1_ps(ua[U_EXT_P]);
                        let xm = _mm256_set1_ps(ua[U_EXT_M]);
                        let g5 = _mm256_loadu_ps(gpk.add(40));
                        let g5x = _mm256_permutevar8x32_ps(g5, idx_x);
                        p5 = _mm256_fmadd_ps(xp, g5, _mm256_fmadd_ps(xm, g5x, p5));
                    }
                    j += 1;
                }
                if NEED & FAM_CNT != 0 {
                    _mm256_storeu_ps(pack_ptr.add(base), p0);
                }
                if NEED & FAM_RVOL != 0 {
                    _mm256_storeu_ps(pack_ptr.add(base + 8), p1);
                }
                if NEED & FAM_LOGVOL != 0 {
                    _mm256_storeu_ps(pack_ptr.add(base + 16), p2);
                }
                if NEED & FAM_FLOW != 0 {
                    _mm256_storeu_ps(pack_ptr.add(base + 24), p3);
                }
                if NEED & FAM_URG != 0 {
                    _mm256_storeu_ps(pack_ptr.add(base + 32), p4);
                }
                if NEED & FAM_EXT != 0 {
                    _mm256_storeu_ps(pack_ptr.add(base + 40), p5);
                }
            }
        }
    }
    }
}
#[cfg(any(not(target_arch = "x86_64"), test))]
#[allow(clippy::too_many_arguments, dead_code)]
fn phase2_scalar<const _NEED: u8>(
    pack: &mut [f32],
    gp: &[f32],
    block_buf: &[(u32, [f32; N_U])],
    counts: &[usize],
    offs: &[usize],
    n: usize,
    k0: u32
) {
        let pack_ptr = pack.as_mut_ptr();
        let gpk0 = gp.as_ptr();
        const SUB: usize = 2;
        let sub_buckets = BLOCK_BUCKETS / SUB;
    {
        // 标量回退：与 SIMD 版逐位一致（mul_add 单次舍入、slot 序固定）
        for sub in 0..SUB {
            let k_lo = sub * sub_buckets;
            let k_hi = k_lo + sub_buckets;
            for a in 0..n {
                let ca = counts[a];
                if ca == 0 {
                    continue;
                }
                let base = a * PACK_W;
                unsafe {
                    let mut p0: [f32; 8] = [0.0; 8];
                    let mut p1: [f32; 8] = [0.0; 8];
                    let mut p2: [f32; 8] = [0.0; 8];
                    let mut p3: [f32; 8] = [0.0; 8];
                    let mut p4: [f32; 8] = [0.0; 8];
                    let mut p5: [f32; 8] = [0.0; 8];
                    for s in 0..8 {
                        p0[s] = *pack_ptr.add(base + s);
                        p1[s] = *pack_ptr.add(base + 8 + s);
                        p2[s] = *pack_ptr.add(base + 16 + s);
                        p3[s] = *pack_ptr.add(base + 24 + s);
                        p4[s] = *pack_ptr.add(base + 32 + s);
                        p5[s] = *pack_ptr.add(base + 40 + s);
                    }
                    let ca0 = offs[a];
                    let mut j0 = 0usize;
                    while j0 < ca
                        && ((block_buf[ca0 + j0].0 - k0) as usize) < k_lo
                    {
                        j0 += 1;
                    }
                    let mut j = j0;
                    while j < ca {
                        let cell = &block_buf[ca0 + j];
                        let kl = (cell.0 - k0) as usize;
                        if kl >= k_hi {
                            break;
                        }
                        let ua = &cell.1;
                        let gpk = gpk0.add(kl * GP_STRIDE);
                        // 6 块 × 8 槽全量 mul_add（未用槽 g 值恒 0 → 结果保留原值；
                        // 与 _mm256_fmadd_ps 对全部 lane 同构，逐位一致）
                        let uc = ua[U_CNT];
                        for s in 0..8 {
                            p0[s] = uc.mul_add(*gpk.add(s), p0[s]);
                        }
                        let ur = ua[U_RVOL];
                        for s in 0..8 {
                            p1[s] = ur.mul_add(*gpk.add(8 + s), p1[s]);
                        }
                        let ul = ua[U_LOGVOL];
                        for s in 0..8 {
                            p2[s] = ul.mul_add(*gpk.add(16 + s), p2[s]);
                        }
                        // signed 块: SIMD 版 p3 = fma(fp,g3, fma(fm,g3x,p3))，
                        // g3x 为相邻通道交换 → 嵌套 FMA 同序
                        let fp = ua[U_FLOW_P];
                        let fm = ua[U_FLOW_M];
                        for s in 0..8 {
                            let gp_ = *gpk.add(24 + s);
                            let gm_ = *gpk.add(24 + (s ^ 1));
                            p3[s] = fp.mul_add(gp_, fm.mul_add(gm_, p3[s]));
                        }
                        let up = ua[U_URG_P];
                        let um = ua[U_URG_M];
                        for s in 0..8 {
                            let gp_ = *gpk.add(32 + s);
                            let gm_ = *gpk.add(32 + (s ^ 1));
                            p4[s] = up.mul_add(gp_, um.mul_add(gm_, p4[s]));
                        }
                        let xp = ua[U_EXT_P];
                        let xm = ua[U_EXT_M];
                        for s in 0..8 {
                            let gp_ = *gpk.add(40 + s);
                            let gm_ = *gpk.add(40 + (s ^ 1));
                            p5[s] = xp.mul_add(gp_, xm.mul_add(gm_, p5[s]));
                        }
                        j += 1;
                    }
                    for s in 0..8 {
                        *pack_ptr.add(base + s) = p0[s];
                        *pack_ptr.add(base + 8 + s) = p1[s];
                        *pack_ptr.add(base + 16 + s) = p2[s];
                        *pack_ptr.add(base + 24 + s) = p3[s];
                        *pack_ptr.add(base + 32 + s) = p4[s];
                        *pack_ptr.add(base + 40 + s) = p5[s];
                    }
                }
            }
        }
    }
}


/// pack 宽度（A-major [a][64]）

/// gp 槽步长（每桶 56 槽）


pub fn compute_matrices(stocks: &[StockPrep], need: u8) -> (Vec<Vec<f32>>, Vec<StockStats>) {
    let n = stocks.len();
    let nm = N_MATRICES;
    let mut mats: Vec<Vec<f32>> = (0..nm)
        .into_par_iter()
        .map(|m| {
            if need & (1u8 << MATRIX_SPECS[m].family) != 0 {
                vec![0.0f32; n * n]
            } else {
                Vec::new()
            }
        })
        .collect();
    let stats: Vec<StockStats> = stocks
        .iter()
        .map(|s| StockStats {
            n_trades: s.n_trades as f64,
            amount: s.amount,
            total_vol: s.total_vol,
            imb: s.imb,
            ret: s.ret,
            vol30: s.vol30,
            vwap: s.vwap,
            q95u: s.q95u,
            sum_w_cnt: s.sum_w[0],
            sum_w_rvol: s.sum_w[1],
            sum_w_logvol: s.sum_w[2],
            sum_w_flow: s.sum_w[3],
            sum_w_urg: s.sum_w[4],
            sum_w_ext: s.sum_w[5],
        })
        .collect();

    // 每股全局 cells 游标（块间推进）
    let mut cursors: Vec<usize> = vec![0; n];

    // 持久 pack 池（每 tile 一个; 块串行 → 无锁竞争; 整日累积后日末刷入 S）
    // TILE=1; pack 块对齐 [a][48]: 6 块 × 8 通道（cnt, rvol, logvol, flow±, urg±, ext±）,
    // 块内通道 0..len-1 对应矩阵场; gp 场 [k][48] 按通道索引连续（块内非对齐 ymm 加载）。
    let ntiles = n;
    // 场 → 通道映射（flush 用）: 场序 = MATRIX_SPECS 序,
    //   场索引: cnt 0..3, rvol 3..6, logvol 6..10, flow 10..18(±交替), urg 18..24, ext 24..28
    fn field_channel(m: usize) -> usize {
        match m {
            0..=2 => m,
            3..=5 => 8 + (m - 3),
            6..=9 => 16 + (m - 6),
            10..=17 => 24 + (m - 10),
            18..=23 => 32 + (m - 18),
            _ => 40 + (m - 24),
        }
    }
    // gp 槽布局（GP_STRIDE=48, 每块 8 槽对齐, 未用槽恒 0）:
    //   0..3 cnt(3τ), 8..11 rvol(3τ), 16..20 logvol(4τ),
    //   24..32 flow±(4τ), 32..38 urg±(3τ), 40..44 ext±(2τ)
    // gp 跨块不持久，改为线程局部复用缓冲（初次分配即全 0，未用槽永不被写，
    // 已用槽在 phase1 每桶全覆盖重写，省掉 4.68 GB 分配与每块每股 1.63 TB 的清零）。
    thread_local! {
        static TL_GP: std::cell::RefCell<Vec<f32>> =
            std::cell::RefCell::new(vec![0.0f32; BLOCK_BUCKETS * GP_STRIDE]);
    }
    let packs: Vec<std::sync::Mutex<Vec<f32>>> = (0..ntiles)
        .into_par_iter()
        .map(|_| std::sync::Mutex::new(vec![0.0f32; n * PACK_W + GP_STRIDE]))
        .collect();

    for block in 0..N_BLOCKS {
        let k0 = block * BLOCK_BUCKETS;
        let k1 = k0 + BLOCK_BUCKETS;
        // ---- 1. 每股块内单元（游标推进, 并行）----
        let counts: Vec<usize> = stocks
            .par_iter()
            .enumerate()
            .map(|(i, st)| {
                let mut c = cursors[i];
                while c < st.cells.len() && (st.cells[c].0 as usize) < k1 {
                    c += 1;
                }
                c - cursors[i]
            })
            .collect();
        let total_cells: usize = counts.iter().sum();
        let mut offs: Vec<usize> = Vec::with_capacity(n + 1);
        let mut acc_off = 0usize;
        for &c in counts.iter() {
            offs.push(acc_off);
            acc_off += c;
        }
        offs.push(acc_off);
        let mut block_buf: Vec<(u32, [f32; N_U])> = vec![(0, [0.0f32; N_U]); total_cells];
        for i in 0..n {
            let src = &stocks[i].cells[cursors[i]..cursors[i] + counts[i]];
            let dst = &mut block_buf[offs[i]..offs[i] + counts[i]];
            for (d, s) in dst.iter_mut().zip(src.iter()) {
                *d = s.clone();
            }
        }
        for i in 0..n {
            cursors[i] += counts[i];
        }

        // ---- 2. tile 并行（TILE=1: 每 tile = 一只 B; A-major 块向量寄存器累积）----
        let block_buf_ref = &block_buf;
        let offs_ref = &offs;
        let counts_ref = &counts;
        (0..ntiles).into_par_iter().for_each(|b| {
            let mut g = packs[b].lock().unwrap();
            let (pack, acc_mem) = g.split_at_mut(n * PACK_W);
            TL_GP.with(|tl| {
                let mut gp_borrow = tl.borrow_mut();
                let gp = gp_borrow.as_mut_slice();
                // ---- phase 1: 扫桶推进 g 场（x86_64 AVX2 SIMD；其他架构标量回退，逐位一致）----
                let bcnt = counts_ref[b];
                let boff = offs_ref[b];
                #[cfg(target_arch = "x86_64")]
                if need == FAM_CNT {
                    phase1_simd::<{ FAM_CNT }>(&mut *gp, &mut *acc_mem, block_buf_ref, bcnt, boff, k0 as u32);
                } else if need == FAM_RVOL {
                    phase1_simd::<{ FAM_RVOL }>(&mut *gp, &mut *acc_mem, block_buf_ref, bcnt, boff, k0 as u32);
                } else if need == (FAM_CNT | FAM_RVOL) {
                    phase1_simd::<{ FAM_CNT | FAM_RVOL }>(&mut *gp, &mut *acc_mem, block_buf_ref, bcnt, boff, k0 as u32);
                } else {
                    phase1_simd::<{ FAM_ALL }>(&mut *gp, &mut *acc_mem, block_buf_ref, bcnt, boff, k0 as u32);
                }
                #[cfg(not(target_arch = "x86_64"))]
                if need == FAM_CNT {
                    phase1_scalar::<{ FAM_CNT }>(&mut *gp, &mut *acc_mem, block_buf_ref, bcnt, boff, k0 as u32);
                } else if need == FAM_RVOL {
                    phase1_scalar::<{ FAM_RVOL }>(&mut *gp, &mut *acc_mem, block_buf_ref, bcnt, boff, k0 as u32);
                } else if need == (FAM_CNT | FAM_RVOL) {
                    phase1_scalar::<{ FAM_CNT | FAM_RVOL }>(&mut *gp, &mut *acc_mem, block_buf_ref, bcnt, boff, k0 as u32);
                } else {
                    phase1_scalar::<{ FAM_ALL }>(&mut *gp, &mut *acc_mem, block_buf_ref, bcnt, boff, k0 as u32);
                }
                // ---- 2. A-major 累积（SUB=2 提升 gp L1 驻留）----
                #[cfg(target_arch = "x86_64")]
                if need == FAM_CNT {
                    phase2_simd::<{ FAM_CNT }>(pack, gp, block_buf_ref, counts_ref, offs_ref, n, k0 as u32);
                } else if need == FAM_RVOL {
                    phase2_simd::<{ FAM_RVOL }>(pack, gp, block_buf_ref, counts_ref, offs_ref, n, k0 as u32);
                } else if need == (FAM_CNT | FAM_RVOL) {
                    phase2_simd::<{ FAM_CNT | FAM_RVOL }>(pack, gp, block_buf_ref, counts_ref, offs_ref, n, k0 as u32);
                } else {
                    phase2_simd::<{ FAM_ALL }>(pack, gp, block_buf_ref, counts_ref, offs_ref, n, k0 as u32);
                }
                #[cfg(not(target_arch = "x86_64"))]
                if need == FAM_CNT {
                    phase2_scalar::<{ FAM_CNT }>(pack, gp, block_buf_ref, counts_ref, offs_ref, n, k0 as u32);
                } else if need == FAM_RVOL {
                    phase2_scalar::<{ FAM_RVOL }>(pack, gp, block_buf_ref, counts_ref, offs_ref, n, k0 as u32);
                } else if need == (FAM_CNT | FAM_RVOL) {
                    phase2_scalar::<{ FAM_CNT | FAM_RVOL }>(pack, gp, block_buf_ref, counts_ref, offs_ref, n, k0 as u32);
                } else {
                    phase2_scalar::<{ FAM_ALL }>(pack, gp, block_buf_ref, counts_ref, offs_ref, n, k0 as u32);
                }
            });
        });
    }
    // ---- 日末刷入 S[m][b][a]（并行; 每 tile 只写第 b 行, 行互不相交）----
    // 方向: pack[a] = Σ_k u_a[k]·g_b[k]，g_b[k] 只含 b 在桶 k 之前的历史权重，
    // 所以这一对里 **b 早、a 后**，即 b 领先 a。要满足"行领先列"的存储语义
    // （S_dir[i][j] = i 领先 j，见文件头），必须写到 S[b][a]，不是 S[a][b]。
    // 2026-10 修正: 旧实现写 S[a][b]，落盘方向与注释相反，导致 directionality /
    // leadership / leader_return 三个有向模块的"出/入"全部取到反的一侧
    // （合成两只股票实测: A 在 1.00s、B 在 2.00s 时值落在 S[A][B]，即列领先行）。
    struct MatPtr(*mut f32);
    unsafe impl Send for MatPtr {}
    unsafe impl Sync for MatPtr {}
    let mats_raw: Vec<MatPtr> = mats.iter_mut().map(|v| MatPtr(v.as_mut_ptr())).collect();
    (0..ntiles).into_par_iter().for_each(|b| {
        let g = packs[b].lock().unwrap();
        let pack = &g[..n * PACK_W];
        for m in 0..nm {
            // 不被请求的家族不写回：省掉 22 张（约 2.3 GB）的写流量
            if need & (1u8 << MATRIX_SPECS[m].family) == 0 {
                continue;
            }
            let ch = field_channel(m);
            let ptr = mats_raw[m].0;
            for a in 0..n {
                unsafe {
                    // 行写（连续地址）: b 领先 a → S[b][a]
                    *ptr.add(b * n + a) = pack[a * PACK_W + ch];
                }
            }
        }
    });
    (mats, stats)
}
#[cfg(test)]
mod tests {
    use super::*;

    /// 逐位一致性验证：phase1/phase2 的 SIMD 版与标量回退版必须产生完全相同的位模式。
    /// （x86_64 上同时编译两版对比；其他架构标量版即生产路径）
    #[test]
    fn scalar_fallback_bitwise_identical() {
        // 合成数据：3 只股票 × 每只 50 个桶单元（覆盖 9 个 u 场、± 通道、稀疏/密集桶）
        let mut rng_state: u64 = 0x9E3779B97F4A7C15;
        let mut next = move || {
            rng_state ^= rng_state << 13;
            rng_state ^= rng_state >> 7;
            rng_state ^= rng_state << 17;
            ((rng_state >> 32) as f64 / u32::MAX as f64) as f32
        };
        let n_stocks = 3usize;
        let mut block_buf: Vec<(u32, [f32; N_U])> = Vec::new();
        let mut counts = vec![0usize; n_stocks];
        let mut offs = vec![0usize; n_stocks + 1];
        let mut c = 0usize;
        for s in 0..n_stocks {
            offs[s] = c;
            for j in 0..50usize {
                let mut u9 = [0.0f32; N_U];
                for f in 0..N_U {
                    u9[f] = next() * 100.0;
                }
                // 桶号递增有序（块内跨 2000 桶，覆盖空桶跳过路径）
                block_buf.push((2000 + s as u32 * 500 + j as u32, u9));
                c += 1;
            }
            counts[s] = 50;
        }
        offs[n_stocks] = c;

        for _round in 0..5 {
            let k0r = (next() * 1500.0) as u32; // k0 < 全部桶号 → 无下溢
            // 随机预热 acc / pack / gp（模拟跨块持久状态）
            let mut g1: Vec<f32> = (0..BLOCK_BUCKETS * GP_STRIDE).map(|_| next() * 10.0).collect();
            let mut a1: Vec<f32> = (0..GP_STRIDE).map(|_| next() * 10.0).collect();
            let mut p1: Vec<f32> = (0..n_stocks * PACK_W).map(|_| next() * 10.0).collect();
            let (g2, a2, p2) = (g1.clone(), a1.clone(), p1.clone());

            #[cfg(target_arch = "x86_64")]
            {
                phase1_simd::<{ FAM_ALL }>(&mut g1, &mut a1, &block_buf, counts[0], offs[0], k0r);
                phase2_simd::<{ FAM_ALL }>(&mut p1, &g1, &block_buf, &counts, &offs, n_stocks, k0r);
            }
            let mut g2m = g2;
            let mut a2m = a2;
            let mut p2m = p2;
            phase1_scalar::<{ FAM_ALL }>(&mut g2m, &mut a2m, &block_buf, counts[0], offs[0], k0r);
            phase2_scalar::<{ FAM_ALL }>(&mut p2m, &g2m, &block_buf, &counts, &offs, n_stocks, k0r);

            #[cfg(target_arch = "x86_64")]
            {
                assert_eq!(a1, a2m, "phase1 acc 不一致 round={} k0={}", _round, k0r);
                assert_eq!(g1, g2m, "phase1 gp 不一致 round={} k0={}", _round, k0r);
                assert_eq!(p1, p2m, "phase2 pack 不一致 round={} k0={}", _round, k0r);
            }
            // 非 x86_64：无 SIMD 对照，仅验证标量版自身可重复（确定性）
            #[cfg(not(target_arch = "x86_64"))]
            {
                let mut g3 = g2m.clone();
                let mut a3 = a2m.clone();
                let mut p3 = p2m.clone();
                phase1_scalar::<{ FAM_ALL }>(&mut g3, &mut a3, &block_buf, counts[0], offs[0], k0r);
                phase2_scalar::<{ FAM_ALL }>(&mut p3, &g3, &block_buf, &counts, &offs, n_stocks, k0r);
                assert_eq!(a2m, a3, "标量版不确定 round={} k0={}", _round, k0r);
                assert_eq!(g2m, g3, "标量版不确定 round={} k0={}", _round, k0r);
                assert_eq!(p2m, p3, "标量版不确定 round={} k0={}", _round, k0r);
            }
        }
    }
}
