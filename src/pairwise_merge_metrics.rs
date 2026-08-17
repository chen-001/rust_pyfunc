//! 配对合并流指标（从 sandbox_pair_x/src/main.rs 迁入主项目, 逐位一致）。
//!
//! 对两只股票的（已排序）成交时间序列 + 大单标记做双指针合并, 计算 4 个交互指标:
//!   X1 交替频率       两股成交事件到达顺序的交替占比（衡量谁主导节奏）
//!   X3 转移互信息     相邻事件 (股A→股A / A→B / B→A / B→B) 转移矩阵的互信息（nats）
//!   X8 大单聚集       相邻大单间隔 < 30s 的比例（10s 分箱直方图前 3 箱 / 总大单间隔）
//!   X9 间隔变异系数   合并流相邻事件间隔的 CV = std/mean
//!
//! 输入约定: times 升序; big 非零 = 大单。两股等长约束无（各取各长）。
//!
//! Python 调用:
//!   x1, x3, x8, x9 = rp.pairwise_merge_metrics(times_a, big_a, times_b, big_b)

use pyo3::prelude::*;

/// 4 个配对指标的输出
#[derive(Clone, Copy, Debug)]
pub struct PairOut {
    pub x1: f32,
    pub x3: f32,
    pub x8: f32,
    pub x9: f32,
}

/// 双指针合并两股 (时间, 大单标记) 流 → (X1 交替频率, X3 转移互信息, X8 大单聚集, X9 间隔CV)。
#[inline]
pub fn pairwise_merge_metrics(
    times_a: &[i64],
    big_a: &[u8],
    times_b: &[i64],
    big_b: &[u8],
) -> PairOut {
    let na = times_a.len();
    let nb = times_b.len();
    let n = na + nb;
    let mut trans: u64 = 0;
    let (mut c00, mut c01, mut c10, mut c11): (u64, u64, u64, u64) = (0, 0, 0, 0);
    let mut h = [0u64; 33]; // 大单间隔直方图: 10s 一箱, 第32箱为溢出
    let mut nbig_gaps: u64 = 0;
    let mut last_big: i64 = i64::MIN;
    let (mut gap_mean, mut gap_m2): (f64, f64) = (0.0, 0.0);
    let mut gap_cnt: u64 = 0;
    let (mut prev_t, mut prev_label): (i64, u8) = (0, 2);
    let mut has_prev = false;
    let (mut ia, mut ib) = (0usize, 0usize);
    while ia < na || ib < nb {
        let (label, t, big) = if ib >= nb || (ia < na && times_a[ia] <= times_b[ib]) {
            let l = (0u8, times_a[ia], big_a[ia] != 0);
            ia += 1;
            l
        } else {
            let l = (1u8, times_b[ib], big_b[ib] != 0);
            ib += 1;
            l
        };
        if has_prev {
            if label != prev_label {
                trans += 1;
            }
            match (prev_label, label) {
                (0, 0) => c00 += 1,
                (0, 1) => c01 += 1,
                (1, 0) => c10 += 1,
                _ => c11 += 1,
            }
            let g = (t - prev_t) as f64;
            gap_cnt += 1;
            let d = g - gap_mean;
            gap_mean += d / gap_cnt as f64;
            gap_m2 += d * (g - gap_mean);
        }
        if big {
            if last_big != i64::MIN {
                let bg = (t - last_big) / 10_000_000;
                let bin = if bg >= 32 { 32usize } else { bg as usize };
                h[bin] += 1;
                nbig_gaps += 1;
            }
            last_big = t;
        }
        prev_t = t;
        prev_label = label;
        has_prev = true;
    }
    // X1 交替频率
    let x1 = if n > 1 { trans as f32 / (n - 1) as f32 } else { 0.0 };
    // X3 转移互信息 (nats)
    let x3 = {
        if n <= 1 {
            0.0
        } else {
            let denom = (n - 1) as f64;
            let (p00, p01, p10, p11) = (
                c00 as f64 / denom,
                c01 as f64 / denom,
                c10 as f64 / denom,
                c11 as f64 / denom,
            );
            let (p0, p1) = (p00 + p01, p10 + p11);
            let mut mi = 0.0;
            for (pab, pa, pb) in [(p00, p0, p0), (p01, p0, p1), (p10, p1, p0), (p11, p1, p1)] {
                if pab > 0.0 && pa > 0.0 && pb > 0.0 {
                    mi += pab * (pab / (pa * pb)).ln();
                }
            }
            mi as f32
        }
    };
    // X8 大单聚集: 相邻大单间隔 < 30s 的比例
    let x8 = if nbig_gaps > 0 {
        (h[0] + h[1] + h[2]) as f32 / nbig_gaps as f32
    } else {
        0.0
    };
    // X9 间隔CV
    let x9 = if gap_cnt > 1 && gap_mean > 0.0 {
        ((gap_m2 / (gap_cnt - 1) as f64).sqrt() / gap_mean) as f32
    } else {
        0.0
    };
    PairOut { x1, x3, x8, x9 }
}

/// Python 入口: 输入两股 (times, big) 序列 → (x1, x3, x8, x9)。
#[pyfunction]
pub fn py_pairwise_merge_metrics(
    times_a: Vec<i64>,
    big_a: Vec<u8>,
    times_b: Vec<i64>,
    big_b: Vec<u8>,
) -> (f32, f32, f32, f32) {
    assert_eq!(times_a.len(), big_a.len(), "times_a 与 big_a 长度不一致");
    assert_eq!(times_b.len(), big_b.len(), "times_b 与 big_b 长度不一致");
    let o = pairwise_merge_metrics(&times_a, &big_a, &times_b, &big_b);
    (o.x1, o.x3, o.x8, o.x9)
}
