//! edge_dynamics: 边的稳定性/跨日动态（降维指标.md 十九~二十四）。
//! 用 prev（20241230）的 cnt_t1, 与当日 cnt_t1 对比（均用对称矩阵）:
//! edge_persistence=行内 Pearson 相关; retention=Top10 重叠率; new_partner=新伙伴强度;
//! edge_change/pos/neg=边强度变化; network_shock_max=最大近似 z 冲击;
//! top_shock_mean=Top5 冲击均值; turnover=1-Jaccard(Top10)。
//! 注: 只有单日前日, 冲击用 (S_t-S_{t-1})/(S_t+S_{t-1}+ε) 近似标准化（注释说明）。
//! prev 中缺失的股票输出 NaN。

use crate::yupei_dist::indicator_ctx::{IndicatorCtx, IndicatorResult};
use rayon::prelude::*;

pub fn name() -> &'static str {
    "edge_dynamics"
}

pub fn desc() -> &'static str {
    "跨日动态: 边持久性/保留率/新伙伴/边变化/冲击/换手 (cnt_t1 vs prev)"
}

pub fn compute(ctx: &IndicatorCtx) -> Vec<IndicatorResult> {
    let n = ctx.n();
    let mut out = Vec::new();
    let prev = match ctx.prev {
        Some(ref p) => p,
        None => {
            // 无前一日数据（批量首日）: 输出 NaN 列保持因子长度
            for mat in crate::yupei_dist::indicator_ctx::MATRIX_LIST {
                let nan = vec![f32::NAN; n];
                for s in ["edge_persistence","retention","new_partner","edge_change",
                          "pos_edge_change","neg_edge_change","network_shock_max",
                          "top_shock_mean","turnover"] {
                    out.push(IndicatorResult::new(format!("dyn_{mat}_{s}"), nan.clone()));
                }
            }
            return out;
        }
    };
    let pn = prev.set.n;
    // 当前股票 → prev 索引
    let map: Vec<Option<usize>> = ctx
        .codes()
        .iter()
        .map(|c| prev.prev_idx(c))
        .collect();
    let k = 10usize;

    for mat in crate::yupei_dist::indicator_ctx::MATRIX_LIST {
        let sym = match ctx.symmetric(mat) {
            Some(s) => s,
            None => continue,
        };
        // prev 对称矩阵（手动 S+S^T, 有向备份）
        let pd = match prev.set.mats.get(mat) {
            Some(m) => m.as_slice(),
            None => {
                // prev 缺该矩阵（如 signed 族未备份）: NaN 列保长度
                let nan = vec![f32::NAN; n];
                for sfx in ["edge_persistence","retention","new_partner","edge_change",
                            "pos_edge_change","neg_edge_change","network_shock_max",
                            "top_shock_mean","turnover"] {
                    out.push(IndicatorResult::new(format!("dyn_{mat}_{sfx}"), nan.clone()));
                }
                continue;
            }
        };
        let mut psym = vec![0.0f32; pn * pn];
        psym.par_chunks_mut(pn).enumerate().for_each(|(i, row)| {
            for j in 0..pn {
                if i != j {
                    row[j] = pd[i * pn + j] + pd[j * pn + i];
                }
            }
        });

    // 当日每行 top-10（并行预计算）
    let top_now: Vec<Vec<u32>> = (0..n)
        .into_par_iter()
        .map(|i| crate::yupei_dist::topk_util::topk_indices(&sym[i * n..(i + 1) * n], k))
        .collect();

    let rows: Vec<[f32; 9]> = (0..n)
        .into_par_iter()
        .map(|i| {
        let pi = match map[i] {
            Some(x) => x,
            None => return [f32::NAN; 9],
        };
        let row = &sym[i * n..(i + 1) * n];
        let prow = &psym[pi * pn..(pi + 1) * pn];
        // 只对两日都存在的股票 j 计算（map[j] 存在）
        let mut sx = 0.0f64;
        let mut sy = 0.0f64;
        let mut sxy = 0.0f64;
        let mut sxx = 0.0f64;
        let mut syy = 0.0f64;
        let mut cnt = 0usize;
        let mut sum_d = 0.0f64;
        let mut sum_pd = 0.0f64;
        let mut sum_pos = 0.0f64;
        let mut sum_neg = 0.0f64;
        let mut max_shock = 0.0f64;
        let mut shocks: Vec<f64> = Vec::with_capacity(16);
        for j in 0..n {
            if j == i {
                continue;
            }
            let pj = match map[j] {
                Some(x) => x,
                None => continue,
            };
            let x = row[j] as f64;
            let y = prow[pj] as f64;
            sx += x;
            sy += y;
            sxy += x * y;
            sxx += x * x;
            syy += y * y;
            cnt += 1;
            let d = x - y;
            sum_d += d;
            sum_pd += x + y;
            if d > 0.0 {
                sum_pos += d;
            } else {
                sum_neg += d;
            }
            let sh = d / (x + y + 1e-9);
            if sh.abs() > max_shock {
                max_shock = sh;
            }
            shocks.push(sh);
        }
        if cnt < 3 {
            return [f32::NAN; 9];
        }
        let c = cnt as f64;
        let num = sxy - sx * sy / c;
        let den = ((sxx - sx * sx / c) * (syy - sy * sy / c)).sqrt();
        let mut pers_v = f32::NAN;
        if den > 1e-12 {
            pers_v = (num / den) as f32;
        }
        // top-10 重叠
        let t_now = &top_now[i];
        let t_prev = crate::yupei_dist::topk_util::topk_indices(prow, k);
        let mut inter = 0usize;
        for &x in t_now.iter() {
            // 仅当 x 在 prev 中且是 prev top-10
            if let Some(pj) = map[x as usize] {
                if t_prev.contains(&(pj as u32)) {
                    inter += 1;
                }
            }
        }
        let reten_v = inter as f32 / k as f32;
        // 新伙伴: 当日 top-10 中不在 prev top-10 的
        let mut np = 0.0f64;
        for &x in t_now.iter() {
            let in_prev_top = map[x as usize]
                .map(|pj| t_prev.contains(&(pj as u32)))
                .unwrap_or(false);
            if !in_prev_top {
                np += row[x as usize] as f64;
            }
        }
        let newp_v = np as f32;
        let echg_v = sum_d as f32;
        let pchg_v = sum_pos as f32;
        let nchg_v = sum_neg as f32;
        let shock_v = max_shock as f32;
        let mut tshock_v = f32::NAN;
        if shocks.len() >= 5 {
            shocks.sort_by(|a, b| b.abs().partial_cmp(&a.abs()).unwrap_or(std::cmp::Ordering::Equal));
            tshock_v = (shocks.iter().take(5).sum::<f64>() / 5.0) as f32;
        } else if !shocks.is_empty() {
            tshock_v = (shocks.iter().sum::<f64>() / shocks.len() as f64) as f32;
        }
        // turnover = 1 - Jaccard
        let union = t_now.len() + t_prev.len() - inter;
        let turn_v = if union > 0 { 1.0 - inter as f32 / union as f32 } else { 0.0 };
        let _ = sum_pd;
        [
            pers_v, reten_v, newp_v, echg_v, pchg_v, nchg_v, shock_v, tshock_v, turn_v,
        ]
    }).collect();
    let pers: Vec<f32> = rows.iter().map(|r| r[0]).collect();
    let reten: Vec<f32> = rows.iter().map(|r| r[1]).collect();
    let newp: Vec<f32> = rows.iter().map(|r| r[2]).collect();
    let echg: Vec<f32> = rows.iter().map(|r| r[3]).collect();
    let pchg: Vec<f32> = rows.iter().map(|r| r[4]).collect();
    let nchg: Vec<f32> = rows.iter().map(|r| r[5]).collect();
    let shock: Vec<f32> = rows.iter().map(|r| r[6]).collect();
    let tshock: Vec<f32> = rows.iter().map(|r| r[7]).collect();
    let turn: Vec<f32> = rows.iter().map(|r| r[8]).collect();
        out.push(IndicatorResult::new(format!("dyn_{mat}_edge_persistence"), pers));
        out.push(IndicatorResult::new(format!("dyn_{mat}_retention"), reten));
        out.push(IndicatorResult::new(format!("dyn_{mat}_new_partner"), newp));
        out.push(IndicatorResult::new(format!("dyn_{mat}_edge_change"), echg));
        out.push(IndicatorResult::new(format!("dyn_{mat}_pos_edge_change"), pchg));
        out.push(IndicatorResult::new(format!("dyn_{mat}_neg_edge_change"), nchg));
        out.push(IndicatorResult::new(format!("dyn_{mat}_network_shock_max"), shock));
        out.push(IndicatorResult::new(format!("dyn_{mat}_top_shock_mean"), tshock));
        out.push(IndicatorResult::new(format!("dyn_{mat}_turnover"), turn));
    }
    out
}
