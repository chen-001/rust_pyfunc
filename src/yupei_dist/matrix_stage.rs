//! 关联-差异矩阵（关联与差异指标.md 的衰减事件场交互能量）—— 纯 Rust 高性能实现。
//!
//! 数学定义（见 关联与差异指标.md 第 11 节）:
//!   S_AB = Σ_{i∈A} Σ_{j∈B} w_i·w_j·e^{-|t_i−t_j|/τ}
//! 等价逐笔实现: 每笔成交权重 w_i（6 种设计），按 10ms 时间桶聚合为 u 场，
//! 用"带衰减的滚动累加器"（z 场）做 B-固定流式扫描:
//!   S_{A→B} = Σ_k u_A[k]·g_B[k],  g_B[k] = Σ_{l<k} u_B[l]·e^{-(k−l)Δ/τ}   (A 领先 B)
//! 一次扫描同时算全部 (权重设计 × τ) 组合（21 个 g 场，AVX2 8-lane 对 A 批处理）。
//!
//! 存储语义: 所有矩阵存"有向" S_dir[i][j] = "i 领先 j 的交互能量"（行主序 N×N f32,
//! 对角 0）。对称交互 = S_dir + S_dirᵀ；同一 10ms 桶内的成交对按 50/50 折半计入
//! 两个方向（S_dir[i][j] 含 +0.5·Σ_k u_i[k]u_j[k]）。
//!
//! 权重设计 w_i（6 族）:
//!   cnt    : w = 1                        （成交同步）
//!   vol    : w = √v                       （大单同步）
//!   logvol : w = ln(1+v)                  （大单同步，对数版）
//!   flow   : w = s·√v, s = 主动方向(66买/83卖) （方向性资金流同步）
//!   urg    : w = s·u, u = 订单编号差异常度（每股 z 标准化） （迫切订单同步）
//!   ext    : w = 1(|u|>q95)·s·√v          （极端迫切订单同步，桶级近似）
//! signed 族（flow/urg/ext）拆同向/反向: same（同买同卖）, opp（对手盘）。
//!
//! τ 集合（秒）: cnt {0.05,0.2,1,5,30}, vol {0.2,1,5,30}, logvol {1,30},
//!              flow {1,5}, urg {1,30}, ext {1} → 共 21 张 N×N 矩阵。
//!
//! 性能设计（内存友好原则）:
//!   - Δ=10ms 桶, 全天 1,422,000 桶; 每股票稀疏 (桶, u[9]) 单元（同桶合并）
//!   - 4096 桶分块: 块内建 (k_local, stock, u[9]) 倒排（k 排序）
//!   - B 并行（rayon 50 线程）; 每 B 流式扫桶: 21 个衰减累加器每桶 21 乘+21 加（无 exp）
//!   - 单元循环按 8 个 A 批处理（AVX2）: 31 次 FMA/单元, 结果向量化刷入 m-major 分级缓冲,
//!     块末一次性写回 21 张矩阵（写流量降 ~40 倍, 无逐单元随机 RMW）

use rayon::prelude::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use crate::fast_csv_reader::TradeRecord;

pub const BUCKET_US: i64 = 10_000; // 10ms
pub const N_BUCKETS: usize = 1_422_000; // 14220s × 100
pub const BLOCK_BUCKETS: usize = 4096; // 块内桶数 (40.96s)
pub const N_BLOCKS: usize = N_BUCKETS / BLOCK_BUCKETS; // 348

/// 每单元 9 个 u 场: [cnt, vol, logvol, flow+, flow−, urg+, urg−, ext+, ext−]
pub const U_CNT: usize = 0;
pub const U_VOL: usize = 1;
pub const U_LOGVOL: usize = 2;
pub const U_FLOW_P: usize = 3;
pub const U_FLOW_M: usize = 4;
pub const U_URG_P: usize = 5;
pub const U_URG_M: usize = 6;
pub const U_EXT_P: usize = 7;
pub const U_EXT_M: usize = 8;
pub const N_U: usize = 9;

/// 矩阵规格（与 MATRIX_NAMES 一一对应）
#[derive(Clone, Copy, Debug)]
pub struct MatrixSpec {
    pub name: &'static str,
    pub family: u8, // 0=cnt 1=vol 2=logvol 3=flow 4=urg 5=ext
    pub tau: f32,   // 秒
    pub signed: bool,
    pub same: bool, // signed 时: true=同向 false=反向
}

pub const MATRIX_SPECS: [MatrixSpec; 37] = [
    // cnt (w=1): τ = 0.05/0.1/0.2/0.5/1/3/5/10/30 s
    MatrixSpec {
        name: "cnt_t005",
        family: 0,
        tau: 0.05,
        signed: false,
        same: true,
    },
    MatrixSpec {
        name: "cnt_t01",
        family: 0,
        tau: 0.1,
        signed: false,
        same: true,
    },
    MatrixSpec {
        name: "cnt_t02",
        family: 0,
        tau: 0.2,
        signed: false,
        same: true,
    },
    MatrixSpec {
        name: "cnt_t05",
        family: 0,
        tau: 0.5,
        signed: false,
        same: true,
    },
    MatrixSpec {
        name: "cnt_t1",
        family: 0,
        tau: 1.0,
        signed: false,
        same: true,
    },
    MatrixSpec {
        name: "cnt_t3",
        family: 0,
        tau: 3.0,
        signed: false,
        same: true,
    },
    MatrixSpec {
        name: "cnt_t5",
        family: 0,
        tau: 5.0,
        signed: false,
        same: true,
    },
    MatrixSpec {
        name: "cnt_t10",
        family: 0,
        tau: 10.0,
        signed: false,
        same: true,
    },
    MatrixSpec {
        name: "cnt_t30",
        family: 0,
        tau: 30.0,
        signed: false,
        same: true,
    },
    // vol (w=√v): τ = 0.2/0.5/1/3/5/30
    MatrixSpec {
        name: "vol_t02",
        family: 1,
        tau: 0.2,
        signed: false,
        same: true,
    },
    MatrixSpec {
        name: "vol_t05",
        family: 1,
        tau: 0.5,
        signed: false,
        same: true,
    },
    MatrixSpec {
        name: "vol_t1",
        family: 1,
        tau: 1.0,
        signed: false,
        same: true,
    },
    MatrixSpec {
        name: "vol_t3",
        family: 1,
        tau: 3.0,
        signed: false,
        same: true,
    },
    MatrixSpec {
        name: "vol_t5",
        family: 1,
        tau: 5.0,
        signed: false,
        same: true,
    },
    MatrixSpec {
        name: "vol_t30",
        family: 1,
        tau: 30.0,
        signed: false,
        same: true,
    },
    // logvol (w=ln(1+v)): τ = 0.5/1/3/30
    MatrixSpec {
        name: "logvol_t05",
        family: 2,
        tau: 0.5,
        signed: false,
        same: true,
    },
    MatrixSpec {
        name: "logvol_t1",
        family: 2,
        tau: 1.0,
        signed: false,
        same: true,
    },
    MatrixSpec {
        name: "logvol_t3",
        family: 2,
        tau: 3.0,
        signed: false,
        same: true,
    },
    MatrixSpec {
        name: "logvol_t30",
        family: 2,
        tau: 30.0,
        signed: false,
        same: true,
    },
    // flow (w=s·√v): τ = 0.2/0.5/1/5, same/opp
    MatrixSpec {
        name: "flow_same_t02",
        family: 3,
        tau: 0.2,
        signed: true,
        same: true,
    },
    MatrixSpec {
        name: "flow_opp_t02",
        family: 3,
        tau: 0.2,
        signed: true,
        same: false,
    },
    MatrixSpec {
        name: "flow_same_t05",
        family: 3,
        tau: 0.5,
        signed: true,
        same: true,
    },
    MatrixSpec {
        name: "flow_opp_t05",
        family: 3,
        tau: 0.5,
        signed: true,
        same: false,
    },
    MatrixSpec {
        name: "flow_same_t1",
        family: 3,
        tau: 1.0,
        signed: true,
        same: true,
    },
    MatrixSpec {
        name: "flow_opp_t1",
        family: 3,
        tau: 1.0,
        signed: true,
        same: false,
    },
    MatrixSpec {
        name: "flow_same_t5",
        family: 3,
        tau: 5.0,
        signed: true,
        same: true,
    },
    MatrixSpec {
        name: "flow_opp_t5",
        family: 3,
        tau: 5.0,
        signed: true,
        same: false,
    },
    // urg (w=s·u): τ = 1/5/30, same/opp
    MatrixSpec {
        name: "urg_same_t1",
        family: 4,
        tau: 1.0,
        signed: true,
        same: true,
    },
    MatrixSpec {
        name: "urg_opp_t1",
        family: 4,
        tau: 1.0,
        signed: true,
        same: false,
    },
    MatrixSpec {
        name: "urg_same_t5",
        family: 4,
        tau: 5.0,
        signed: true,
        same: true,
    },
    MatrixSpec {
        name: "urg_opp_t5",
        family: 4,
        tau: 5.0,
        signed: true,
        same: false,
    },
    MatrixSpec {
        name: "urg_same_t30",
        family: 4,
        tau: 30.0,
        signed: true,
        same: true,
    },
    MatrixSpec {
        name: "urg_opp_t30",
        family: 4,
        tau: 30.0,
        signed: true,
        same: false,
    },
    // ext (极端迫切): τ = 1/5, same/opp
    MatrixSpec {
        name: "ext_same_t1",
        family: 5,
        tau: 1.0,
        signed: true,
        same: true,
    },
    MatrixSpec {
        name: "ext_opp_t1",
        family: 5,
        tau: 1.0,
        signed: true,
        same: false,
    },
    MatrixSpec {
        name: "ext_same_t5",
        family: 5,
        tau: 5.0,
        signed: true,
        same: true,
    },
    MatrixSpec {
        name: "ext_opp_t5",
        family: 5,
        tau: 5.0,
        signed: true,
        same: false,
    },
];

pub const N_MATRICES: usize = MATRIX_SPECS.len();

/// 每股预处理结果: 稀疏桶单元 + 统计量
#[derive(Clone, Debug)]
pub struct StockPrep {
    pub code: String,
    /// (桶idx, u[9])，按桶升序，同桶已合并
    pub cells: Vec<(u32, [f32; N_U])>,
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
    pub sum_w_vol: f64,
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

    // ---- pass 2: 每笔临时权重（桶idx, u, sv, s）----
    struct Tmp {
        bidx: u32,
        u: f32,
        sv: f32,
        lv: f32,
        s: f32,
    }
    let mut tmp: Vec<Tmp> = Vec::with_capacity(n);
    let mut absus: Vec<f32> = Vec::with_capacity(n);
    let (mut sum_w_cnt, mut sum_w_vol, mut sum_w_logvol, mut sum_w_flow, mut sum_w_urg) =
        (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64);
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
        tmp.push(Tmp { bidx, u, sv, lv, s });
        absus.push(u.abs());
        sum_w_cnt += 1.0;
        sum_w_vol += sv as f64;
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
    let mut sum_w_ext = 0.0f64;
    for t in tmp {
        let mut u9 = [0.0f32; N_U];
        u9[U_CNT] = 1.0;
        u9[U_VOL] = t.sv;
        u9[U_LOGVOL] = t.lv;
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
                continue;
            }
        }
        cells.push((t.bidx, u9));
    }
    let n_kept = cells
        .iter()
        .map(|(_, ua)| ua[U_CNT] as usize)
        .sum::<usize>();
    let vol30 = bucket_vol30(recs, day_start_us);
    Some(StockPrep {
        code: code.to_string(),
        cells,
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
            sum_w_vol,
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

/// g 场元数据: 每矩阵的 (u+ 场, u− 场, g+ 场, g− 场, 是否 signed)
/// g 场布局（21）: 0..5 cnt(τ序), 5..9 vol, 9..11 logvol,
///                11..15 flow(±, τ1,τ5), 15..19 urg(±, τ1,τ30), 19..21 ext(±, τ1)
struct GInfo {
    up: usize,
    um: usize,
    gp: usize,
    gm: usize,
}

fn g_info() -> Vec<GInfo> {
    let u_field: [usize; 6] = [U_CNT, U_VOL, U_LOGVOL, U_FLOW_P, U_URG_P, U_EXT_P];
    let tau_list: [f32; 5] = [0.05, 0.2, 1.0, 5.0, 30.0];
    let fam_g_start: [usize; 3] = [0, 5, 9];
    let mut out = Vec::with_capacity(N_MATRICES);
    for spec in MATRIX_SPECS.iter() {
        let fam = spec.family as usize;
        let up = u_field[fam];
        if !spec.signed {
            let ti = tau_list
                .iter()
                .position(|&t| (t - spec.tau).abs() < 1e-6)
                .unwrap();
            let gi = fam_g_start[fam] + ti;
            out.push(GInfo {
                up,
                um: up,
                gp: gi,
                gm: gi,
            });
        } else {
            let base = match fam {
                3 => 11usize,
                4 => 15,
                5 => 19,
                _ => unreachable!(),
            };
            let off = match (fam, spec.tau as i32) {
                (3, 1) => 0,
                (3, 5) => 2,
                (4, 1) => 0,
                (4, 30) => 2,
                (5, 1) => 0,
                _ => 0,
            };
            let (gp, gm) = if spec.same {
                (base + off, base + off + 1)
            } else {
                (base + off + 1, base + off)
            };
            out.push(GInfo {
                up,
                um: up + 1,
                gp,
                gm,
            });
        }
    }
    out
}

/// 21 个 g 场的衰减系数与 u 场映射（每场: u 场 idx, decay）
struct GField {
    uf: usize,
    dec: f32,
}

fn g_fields() -> [GField; 37] {
    // 无符号（19）: cnt9, vol6, logvol4; signed（18）: flow±×4τ, urg±×3τ, ext±×2τ
    let d = decay;
    [
        // cnt: 0.05 0.1 0.2 0.5 1 3 5 10 30
        GField {
            uf: U_CNT,
            dec: d(0.05),
        },
        GField {
            uf: U_CNT,
            dec: d(0.1),
        },
        GField {
            uf: U_CNT,
            dec: d(0.2),
        },
        GField {
            uf: U_CNT,
            dec: d(0.5),
        },
        GField {
            uf: U_CNT,
            dec: d(1.0),
        },
        GField {
            uf: U_CNT,
            dec: d(3.0),
        },
        GField {
            uf: U_CNT,
            dec: d(5.0),
        },
        GField {
            uf: U_CNT,
            dec: d(10.0),
        },
        GField {
            uf: U_CNT,
            dec: d(30.0),
        },
        // vol: 0.2 0.5 1 3 5 30
        GField {
            uf: U_VOL,
            dec: d(0.2),
        },
        GField {
            uf: U_VOL,
            dec: d(0.5),
        },
        GField {
            uf: U_VOL,
            dec: d(1.0),
        },
        GField {
            uf: U_VOL,
            dec: d(3.0),
        },
        GField {
            uf: U_VOL,
            dec: d(5.0),
        },
        GField {
            uf: U_VOL,
            dec: d(30.0),
        },
        // logvol: 0.5 1 3 30
        GField {
            uf: U_LOGVOL,
            dec: d(0.5),
        },
        GField {
            uf: U_LOGVOL,
            dec: d(1.0),
        },
        GField {
            uf: U_LOGVOL,
            dec: d(3.0),
        },
        GField {
            uf: U_LOGVOL,
            dec: d(30.0),
        },
        // flow ±: 0.2 0.5 1 5
        GField {
            uf: U_FLOW_P,
            dec: d(0.2),
        },
        GField {
            uf: U_FLOW_M,
            dec: d(0.2),
        },
        GField {
            uf: U_FLOW_P,
            dec: d(0.5),
        },
        GField {
            uf: U_FLOW_M,
            dec: d(0.5),
        },
        GField {
            uf: U_FLOW_P,
            dec: d(1.0),
        },
        GField {
            uf: U_FLOW_M,
            dec: d(1.0),
        },
        GField {
            uf: U_FLOW_P,
            dec: d(5.0),
        },
        GField {
            uf: U_FLOW_M,
            dec: d(5.0),
        },
        // urg ±: 1 5 30
        GField {
            uf: U_URG_P,
            dec: d(1.0),
        },
        GField {
            uf: U_URG_M,
            dec: d(1.0),
        },
        GField {
            uf: U_URG_P,
            dec: d(5.0),
        },
        GField {
            uf: U_URG_M,
            dec: d(5.0),
        },
        GField {
            uf: U_URG_P,
            dec: d(30.0),
        },
        GField {
            uf: U_URG_M,
            dec: d(30.0),
        },
        // ext ±: 1 5
        GField {
            uf: U_EXT_P,
            dec: d(1.0),
        },
        GField {
            uf: U_EXT_M,
            dec: d(1.0),
        },
        GField {
            uf: U_EXT_P,
            dec: d(5.0),
        },
        GField {
            uf: U_EXT_M,
            dec: d(5.0),
        },
    ]
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
pub fn compute_matrices(stocks: &[StockPrep]) -> (Vec<Vec<f32>>, Vec<StockStats>) {
    let n = stocks.len();
    let nm = N_MATRICES;
    let gfields = g_fields();
    let mut mats: Vec<Vec<f32>> = vec![vec![0.0f32; n * n]; nm];
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
            sum_w_vol: s.sum_w[1],
            sum_w_logvol: s.sum_w[2],
            sum_w_flow: s.sum_w[3],
            sum_w_urg: s.sum_w[4],
            sum_w_ext: s.sum_w[5],
        })
        .collect();

    // 每股全局 cells 游标（块间推进）
    let mut cursors: Vec<usize> = vec![0; n];

    // 持久 pack 池（每 tile 一个; 块串行 → 无锁竞争; 整日累积后日末刷入 S）
    // TILE=1; pack 块对齐 [a][64]: 7 块 × 8 通道（cnt×2, vol, logvol, flow, urg, ext）,
    // 块内通道 0..len-1 对应矩阵场; gp 场 [k][37] 按场索引连续（块内非对齐 ymm 加载）。
    let ntiles = n;
    const PACK_W: usize = 64;
    // 块配置: (场起始, 长度, u+ 场, u− 场, signed)
    // 场索引: cnt 0..9, vol 9..15, logvol 15..19, flow 19..27(±交替), urg 27..33, ext 33..37
    const BLOCKS: [(usize, usize, usize, usize, bool); 7] = [
        (0, 8, U_CNT, U_CNT, false),
        (8, 1, U_CNT, U_CNT, false),
        (9, 6, U_VOL, U_VOL, false),
        (15, 4, U_LOGVOL, U_LOGVOL, false),
        (19, 8, U_FLOW_P, U_FLOW_M, true),
        (27, 6, U_URG_P, U_URG_M, true),
        (33, 4, U_EXT_P, U_EXT_M, true),
    ];
    // 场 → 通道映射（flush 用）: 块 b 起始通道 = 前面各块 8 通道之和
    fn field_channel(m: usize) -> usize {
        match m {
            0..=8 => m,
            9..=14 => 16 + (m - 9),
            15..=18 => 24 + (m - 15),
            19..=26 => 32 + (m - 19),
            27..=32 => 40 + (m - 27),
            _ => 48 + (m - 33),
        }
    }
    // gp 槽布局（GP_STRIDE=56, 每块 8 槽对齐, 未用槽恒 0）:
    //   0..9 cnt(9τ), 9..16 空闲(块1), 16..24 vol(6τ), 24..32 logvol(4τ),
    //   32..40 flow±(4τ), 40..48 urg±(3τ), 48..56 ext±(2τ)
    const GP_STRIDE: usize = 56;
    let packs: Vec<std::sync::Mutex<Vec<f32>>> = (0..ntiles)
        .map(|_| std::sync::Mutex::new(vec![0.0f32; n * PACK_W + BLOCK_BUCKETS * GP_STRIDE + 56]))
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
        let gfields = &gfields;
        (0..ntiles).into_par_iter().for_each(|b| {
            let mut g = packs[b].lock().unwrap();
            let (pack, rest) = g.split_at_mut(n * PACK_W);
            let (gp, acc_mem) = rest.split_at_mut(BLOCK_BUCKETS * GP_STRIDE);
            // 未用槽一次性清零（跨块不复用; 已用槽每桶重写）
            unsafe {
                std::ptr::write_bytes(gp.as_mut_ptr(), 0, gp.len());
            }
            // B 块内单元游标（块局部）
            let mut b_curs = 0usize;
            // B 的桶内 u
            let mut ub = [0.0f32; N_U];
            // ---- phase 1: 扫桶推进 g 场（SIMD: 7×ymm acc, 每桶 ~30 向量操作; 56 槽布局）----
            let bcnt = counts_ref[b];
            let boff = offs_ref[b];
            unsafe {
                let zero = _mm256_setzero_ps();
                let half = _mm256_set1_ps(0.5);
                // 衰减系数按 56 槽打包（未用槽 0 → acc 恒 0）
                let d0 = _mm256_setr_ps(
                    decay(0.05),
                    decay(0.1),
                    decay(0.2),
                    decay(0.5),
                    decay(1.0),
                    decay(3.0),
                    decay(5.0),
                    decay(10.0),
                );
                let d1s = decay(30.0);
                let d2 = _mm256_setr_ps(
                    decay(0.2),
                    decay(0.5),
                    decay(1.0),
                    decay(3.0),
                    decay(5.0),
                    decay(30.0),
                    0.0,
                    0.0,
                );
                let d3 = _mm256_setr_ps(
                    decay(0.5),
                    decay(1.0),
                    decay(3.0),
                    decay(30.0),
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                );
                let d4 = _mm256_setr_ps(
                    decay(0.2),
                    decay(0.2),
                    decay(0.5),
                    decay(0.5),
                    decay(1.0),
                    decay(1.0),
                    decay(5.0),
                    decay(5.0),
                );
                let d5 = _mm256_setr_ps(
                    decay(1.0),
                    decay(1.0),
                    decay(5.0),
                    decay(5.0),
                    decay(30.0),
                    decay(30.0),
                    0.0,
                    0.0,
                );
                let d6 = _mm256_setr_ps(
                    decay(1.0),
                    decay(1.0),
                    decay(5.0),
                    decay(5.0),
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                );
                // acc 跨块持久（packs 内存段; 块开始 load, 块末 store 回）
                let aptr = acc_mem.as_ptr();
                let mut a0 = _mm256_loadu_ps(aptr);
                let mut a1s = *aptr.add(8);
                let mut a2 = _mm256_loadu_ps(aptr.add(16));
                let mut a3 = _mm256_loadu_ps(aptr.add(24));
                let mut a4 = _mm256_loadu_ps(aptr.add(32));
                let mut a5 = _mm256_loadu_ps(aptr.add(40));
                let mut a6 = _mm256_loadu_ps(aptr.add(48));
                for k in 0..BLOCK_BUCKETS {
                    for f in 0..N_U {
                        ub[f] = 0.0;
                    }
                    while b_curs < bcnt
                        && (block_buf_ref[boff + b_curs].0 - k0 as u32) as usize == k
                    {
                        let c = &block_buf_ref[boff + b_curs];
                        for f in 0..N_U {
                            ub[f] += c.1[f];
                        }
                        b_curs += 1;
                    }
                    let gpk = gp.as_mut_ptr().add(k * GP_STRIDE);
                    let uc = _mm256_set1_ps(ub[U_CNT]);
                    let uv = _mm256_blend_ps(_mm256_set1_ps(ub[U_VOL]), zero, 0b11000000);
                    let ul = _mm256_blend_ps(_mm256_set1_ps(ub[U_LOGVOL]), zero, 0b11110000);
                    let u4 = _mm256_unpacklo_ps(
                        _mm256_set1_ps(ub[U_FLOW_P]),
                        _mm256_set1_ps(ub[U_FLOW_M]),
                    );
                    let u5 = _mm256_blend_ps(
                        _mm256_unpacklo_ps(
                            _mm256_set1_ps(ub[U_URG_P]),
                            _mm256_set1_ps(ub[U_URG_M]),
                        ),
                        zero,
                        0b11000000,
                    );
                    let u6 = _mm256_blend_ps(
                        _mm256_unpacklo_ps(
                            _mm256_set1_ps(ub[U_EXT_P]),
                            _mm256_set1_ps(ub[U_EXT_M]),
                        ),
                        zero,
                        0b11110000,
                    );
                    // cnt 块0（槽 0..8）: gpk = acc + 0.5u; acc = (acc+u)·dec
                    let g0 = _mm256_add_ps(a0, _mm256_mul_ps(half, uc));
                    _mm256_storeu_ps(gpk, g0);
                    a0 = _mm256_mul_ps(_mm256_add_ps(a0, uc), d0);
                    // cnt_t30（槽 8, 标量）
                    let ucs = ub[U_CNT];
                    *gpk.add(8) = a1s + 0.5 * ucs;
                    a1s = (a1s + ucs) * d1s;
                    // vol（槽 16..24）
                    let g2 = _mm256_add_ps(a2, _mm256_mul_ps(half, uv));
                    _mm256_storeu_ps(gpk.add(16), g2);
                    a2 = _mm256_mul_ps(_mm256_add_ps(a2, uv), d2);
                    // logvol（槽 24..32）
                    let g3 = _mm256_add_ps(a3, _mm256_mul_ps(half, ul));
                    _mm256_storeu_ps(gpk.add(24), g3);
                    a3 = _mm256_mul_ps(_mm256_add_ps(a3, ul), d3);
                    // flow ±（槽 32..40, unpacklo 交错 P+/P−）
                    let g4 = _mm256_add_ps(a4, _mm256_mul_ps(half, u4));
                    _mm256_storeu_ps(gpk.add(32), g4);
                    a4 = _mm256_mul_ps(_mm256_add_ps(a4, u4), d4);
                    // urg ±（槽 40..48）
                    let g5 = _mm256_add_ps(a5, _mm256_mul_ps(half, u5));
                    _mm256_storeu_ps(gpk.add(40), g5);
                    a5 = _mm256_mul_ps(_mm256_add_ps(a5, u5), d5);
                    // ext ±（槽 48..56）
                    let g6 = _mm256_add_ps(a6, _mm256_mul_ps(half, u6));
                    _mm256_storeu_ps(gpk.add(48), g6);
                    a6 = _mm256_mul_ps(_mm256_add_ps(a6, u6), d6);
                }
                // 块末 store 回 acc（跨块持久）
                let amut = acc_mem.as_mut_ptr();
                _mm256_storeu_ps(amut, a0);
                *amut.add(8) = a1s;
                _mm256_storeu_ps(amut.add(16), a2);
                _mm256_storeu_ps(amut.add(24), a3);
                _mm256_storeu_ps(amut.add(32), a4);
                _mm256_storeu_ps(amut.add(40), a5);
                _mm256_storeu_ps(amut.add(48), a6);
            }
            // ---- phase 2: A-major 显式 AVX2（7 块 × 9 FMA + 3 permute 实现 same/opp; SUB=2 提升 gp L1 驻留）----
            let pack_ptr = pack.as_mut_ptr();
            let gpk0 = gp.as_ptr();
            const SUB: usize = 2;
            let sub_buckets = BLOCK_BUCKETS / SUB;
            // ± 通道交换索引（flow/urg/ext 组内相邻通道互换）
            let idx_x = unsafe { _mm256_setr_epi32(1, 0, 3, 2, 5, 4, 7, 6) };
            for sub in 0..SUB {
                let k_lo = sub * sub_buckets;
                let k_hi = k_lo + sub_buckets;
                for a in 0..n {
                    let ca = counts_ref[a];
                    if ca == 0 {
                        continue;
                    }
                    let base = a * PACK_W;
                    unsafe {
                        let mut p0 = _mm256_loadu_ps(pack_ptr.add(base));
                        let mut p1 = _mm256_loadu_ps(pack_ptr.add(base + 8));
                        let mut p2 = _mm256_loadu_ps(pack_ptr.add(base + 16));
                        let mut p3 = _mm256_loadu_ps(pack_ptr.add(base + 24));
                        let mut p4 = _mm256_loadu_ps(pack_ptr.add(base + 32));
                        let mut p5 = _mm256_loadu_ps(pack_ptr.add(base + 40));
                        let mut p6 = _mm256_loadu_ps(pack_ptr.add(base + 48));
                        let ca0 = offs_ref[a];
                        let mut j0 = 0usize;
                        while j0 < ca && ((block_buf_ref[ca0 + j0].0 - k0 as u32) as usize) < k_lo {
                            j0 += 1;
                        }
                        let mut j = j0;
                        while j < ca {
                            let cell = &block_buf_ref[ca0 + j];
                            let kl = (cell.0 - k0 as u32) as usize;
                            if kl >= k_hi {
                                break;
                            }
                            let ua = &cell.1;
                            // 8 个权重标量广播（每单元一次）
                            let uc = _mm256_set1_ps(ua[U_CNT]);
                            let uv = _mm256_set1_ps(ua[U_VOL]);
                            let ul = _mm256_set1_ps(ua[U_LOGVOL]);
                            let fp = _mm256_set1_ps(ua[U_FLOW_P]);
                            let fm = _mm256_set1_ps(ua[U_FLOW_M]);
                            let up = _mm256_set1_ps(ua[U_URG_P]);
                            let um = _mm256_set1_ps(ua[U_URG_M]);
                            let xp = _mm256_set1_ps(ua[U_EXT_P]);
                            let xm = _mm256_set1_ps(ua[U_EXT_M]);
                            let gpk = gpk0.add(kl * GP_STRIDE);
                            // 7 个 ymm 加载（56 槽, 块对齐; 未用槽恒 0 → 乘积恒 0）
                            let g0 = _mm256_loadu_ps(gpk); // cnt 8τ
                            let g1 = _mm256_loadu_ps(gpk.add(8)); // cnt_t30 + 0×7
                            let g2 = _mm256_loadu_ps(gpk.add(16)); // vol 6τ + 0×2
                            let g3 = _mm256_loadu_ps(gpk.add(24)); // logvol 4τ + 0×4
                            let g4 = _mm256_loadu_ps(gpk.add(32)); // flow ±
                            let g5 = _mm256_loadu_ps(gpk.add(40)); // urg ±
                            let g6 = _mm256_loadu_ps(gpk.add(48)); // ext ±
                            p0 = _mm256_fmadd_ps(uc, g0, p0);
                            p1 = _mm256_fmadd_ps(uc, g1, p1);
                            p2 = _mm256_fmadd_ps(uv, g2, p2);
                            p3 = _mm256_fmadd_ps(ul, g3, p3);
                            // signed 块: same = A+×B+ + A−×B−; opp = A+×B− + A−×B+（通道交换）
                            let g4x = _mm256_permutevar8x32_ps(g4, idx_x);
                            p4 = _mm256_fmadd_ps(fp, g4, _mm256_fmadd_ps(fm, g4x, p4));
                            let g5x = _mm256_permutevar8x32_ps(g5, idx_x);
                            p5 = _mm256_fmadd_ps(up, g5, _mm256_fmadd_ps(um, g5x, p5));
                            let g6x = _mm256_permutevar8x32_ps(g6, idx_x);
                            p6 = _mm256_fmadd_ps(xp, g6, _mm256_fmadd_ps(xm, g6x, p6));
                            j += 1;
                        }
                        _mm256_storeu_ps(pack_ptr.add(base), p0);
                        _mm256_storeu_ps(pack_ptr.add(base + 8), p1);
                        _mm256_storeu_ps(pack_ptr.add(base + 16), p2);
                        _mm256_storeu_ps(pack_ptr.add(base + 24), p3);
                        _mm256_storeu_ps(pack_ptr.add(base + 32), p4);
                        _mm256_storeu_ps(pack_ptr.add(base + 40), p5);
                        _mm256_storeu_ps(pack_ptr.add(base + 48), p6);
                    }
                }
            }
        });
    }
    // ---- 日末刷入 S[m][a][b]（并行; 每 tile 只写第 b 列, 列互不相交）----
    struct MatPtr(*mut f32);
    unsafe impl Send for MatPtr {}
    unsafe impl Sync for MatPtr {}
    let mats_raw: Vec<MatPtr> = mats.iter_mut().map(|v| MatPtr(v.as_mut_ptr())).collect();
    (0..ntiles).into_par_iter().for_each(|b| {
        let g = packs[b].lock().unwrap();
        let pack = &g[..n * PACK_W];
        for m in 0..nm {
            let ch = field_channel(m);
            let ptr = mats_raw[m].0;
            for a in 0..n {
                unsafe {
                    *ptr.add(a * n + b) = pack[a * PACK_W + ch];
                }
            }
        }
    });
    (mats, stats)
}
