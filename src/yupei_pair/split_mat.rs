//! 方向拆分的四分量交互矩阵（玉佩2 主买/主卖拆分用）。
//!
//! 现有 28 张矩阵把每只股票的成交当一个整体；这里按**主动方向**把它拆成
//! 主买（P，flag=66）和主卖（M，flag=83）两套，两两组合成四个分量:
//!
//!   S_pq[i][j] = 「i 的 p 类成交」领先「j 的 q 类成交」的衰减交互能量
//!                 = Σ_k u_p[i][k] · g_q[j][k]
//!   g_q[j][k] = Σ_{l<k} u_q[j][l]·e^{-(k−l)Δ/τ} + 0.5·u_q[j][k]
//!
//! 存储语义与 matrix_stage 一致: **S[i][j] = i 领先 j**（行领先列）。
//! 所以 tile 固定的是"先动方 j"（提供历史 g），遍历的 i 是"后动方"，
//! 落盘写到 `mats[t][j*n + i]`。
//!
//! 两个可选的权重族（`PmWeight`）:
//!   - `Cnt`: u_p = 1（该方向有多少笔成交）→ 主买/主卖**成交同步**
//!   - `Vol`: u_p = √v（该方向的成交量）→ 主买/主卖的**放量同步**
//!
//! 用法: `compute_split_mats(&stocks, tau, PmWeight::Cnt)` → `[PP, PM, MP, MM]`。
//! 上层按方向组合把它们线性叠加: 例如"主买 vs 全部" = PP + PM。

use rayon::prelude::*;

use crate::yupei_dist::matrix_stage::{StockPrep, BLOCK_BUCKETS, BUCKET_US, N_BLOCKS};

/// 四分量下标: 0=PP（买领先买）, 1=PM（买领先卖）, 2=MP（卖领先买）, 3=MM（卖领先卖）。
pub const PP: usize = 0;
pub const PM: usize = 1;
pub const MP: usize = 2;
pub const MM: usize = 3;

/// 权重族：用笔数还是成交量。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PmWeight {
    /// u = 1（该方向的成交笔数）
    Cnt,
    /// u = √v（该方向的成交量开方）
    Vol,
}

impl PmWeight {
    /// 从 `cells_pm` 的 4 槽里取本族的两路（P, M）。
    #[inline]
    fn lanes(self) -> (usize, usize) {
        match self {
            PmWeight::Cnt => (0, 1),
            PmWeight::Vol => (2, 3),
        }
    }
}

/// 单日四分量矩阵: `[PP, PM, MP, MM]`，各 n×n 行主序，S[i][j] = i 领先 j。
pub fn compute_split_mats(stocks: &[StockPrep], tau: f32, w: PmWeight) -> [Vec<f32>; 4] {
    let n = stocks.len();
    let (lp, lm) = w.lanes();
    let decay = (-(BUCKET_US as f32) / (tau * 1e6)).exp();
    let mut mats: [Vec<f32>; 4] = std::array::from_fn(|_| vec![0.0f32; n * n]);

    // 每 tile（先动方 j）一份跨块持久的 4 分量累加器 + 本块的 g 场。
    // 累加器必须跨块保留（一个 (i,j) 对的能量分散在全天所有块里）。
    let packs: Vec<std::sync::Mutex<Vec<f32>>> = (0..n)
        .into_par_iter()
        .map(|_| std::sync::Mutex::new(vec![0.0f32; n * 4 + 2 + GPAD]))
        .collect();

    let mut cursors = vec![0usize; n];
    for block in 0..N_BLOCKS {
        let k0 = (block * BLOCK_BUCKETS) as u32;
        let k1 = k0 + BLOCK_BUCKETS as u32;

        // ---- 1. 每股块内 cell 数 + 拼块内缓冲（按股票连续）----
        let counts: Vec<usize> = stocks
            .par_iter()
            .enumerate()
            .map(|(i, st)| {
                let mut c = cursors[i];
                while c < st.cells_pm.len() && st.cells_pm[c].0 < k1 {
                    c += 1;
                }
                c - cursors[i]
            })
            .collect();
        let total: usize = counts.iter().sum();
        let mut offs: Vec<usize> = Vec::with_capacity(n + 1);
        let mut acc = 0usize;
        for &c in counts.iter() {
            offs.push(acc);
            acc += c;
        }
        offs.push(acc);
        let mut buf: Vec<(u32, [f32; 4])> = vec![(0, [0.0f32; 4]); total];
        for i in 0..n {
            let s = cursors[i];
            let c = counts[i];
            buf[offs[i]..offs[i] + c].copy_from_slice(&stocks[i].cells_pm[s..s + c]);
            cursors[i] += c;
        }

        // ---- 2. tile 并行：phase1 算 g 场，phase2 对每个 i 做 4 分量 FMA ----
        let buf_ref = &buf;
        let offs_ref = &offs;
        let counts_ref = &counts;
        (0..n).into_par_iter().for_each(|j| {
            let mut g = packs[j].lock().unwrap();
            let (pack, rest) = g.split_at_mut(n * 4);
            // 衰减累加器必须**跨块持久**：块末保留"已衰减到下一块第 0 桶之前"的状态，
            // 否则每个块边界都把跨块的历史权重丢一次（一天 348 个块，错误会累积）。
            let (acc, gp) = rest.split_at_mut(2);
            let mut ap = acc[0];
            let mut am = acc[1];
            let bcnt = counts_ref[j];
            let boff = offs_ref[j];
            let mut cur = 0usize;
            for k in 0..BLOCK_BUCKETS {
                let mut up = 0.0f32;
                let mut um = 0.0f32;
                while cur < bcnt && (buf_ref[boff + cur].0 - k0) as usize == k {
                    let u = &buf_ref[boff + cur].1;
                    up += u[lp];
                    um += u[lm];
                    cur += 1;
                }
                // 同桶 50/50 折半，与 28 张矩阵同口径
                gp[k * 2] = ap + 0.5 * up;
                gp[k * 2 + 1] = am + 0.5 * um;
                ap = (ap + up) * decay;
                am = (am + um) * decay;
            }

            // phase2: 后动方 i 遍历自己的块内 cell，累加 4 个分量
            for i in 0..n {
                let ca = counts_ref[i];
                if ca == 0 {
                    continue;
                }
                let base = i * 4;
                let mut p = [
                    pack[base],
                    pack[base + 1],
                    pack[base + 2],
                    pack[base + 3],
                ];
                for t in 0..ca {
                    let cell = &buf_ref[offs_ref[i] + t];
                    let kl = (cell.0 - k0) as usize;
                    let gpk = &gp[kl * 2..kl * 2 + 2];
                    let up = cell.1[lp];
                    let um = cell.1[lm];
                    p[PP] += up * gpk[0];
                    p[PM] += up * gpk[1];
                    p[MP] += um * gpk[0];
                    p[MM] += um * gpk[1];
                }
                pack[base] = p[PP];
                pack[base + 1] = p[PM];
                pack[base + 2] = p[MP];
                pack[base + 3] = p[MM];
            }
            // 累加器存回，供下一块接着衰减
            acc[0] = ap;
            acc[1] = am;
        });
    }

    // ---- 3. 日末刷入 mats[t][j*n + i]（每 tile 只写第 j 行，行互不相交）----
    struct Ptr(*mut f32);
    unsafe impl Send for Ptr {}
    unsafe impl Sync for Ptr {}
    let raw: Vec<Ptr> = mats.iter_mut().map(|v| Ptr(v.as_mut_ptr())).collect();
    (0..n).into_par_iter().for_each(|j| {
        let g = packs[j].lock().unwrap();
        let pack = &g[..n * 4];
        for i in 0..n {
            for t in 0..4 {
                unsafe {
                    *raw[t].0.add(j * n + i) = pack[i * 4 + t];
                }
            }
        }
    });
    mats
}

/// gp 场在 pack 尾部的槽数: 每桶 2 个场（gP, gM）。
const GPAD: usize = BLOCK_BUCKETS * 2;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::yupei_dist::matrix_stage::{
        compute_matrices, StockPrep, FAM_CNT, MATRIX_SPECS, N_U,
    };

    /// 造一只股票：cells 与 cells_pm 同步（同桶合并，方向和 = 全部）。
    fn mk(code: &str, buckets: &[(u32, f32, f32)]) -> StockPrep {
        let mut cells: Vec<(u32, [f32; N_U])> = Vec::new();
        let mut cells_pm: Vec<(u32, [f32; 4])> = Vec::new();
        for &(k, p, m) in buckets {
            // p = 买笔数, m = 卖笔数（cnt 族权重都是 1）
            let mut u9 = [0.0f32; N_U];
            u9[0] = p + m; // U_CNT
            let mut u4 = [0.0f32; 4];
            u4[0] = p;
            u4[1] = m;
            if let Some((bk, ua)) = cells.last_mut() {
                if *bk == k {
                    for t in 0..N_U {
                        ua[t] += u9[t];
                    }
                    let ub = &mut cells_pm.last_mut().unwrap().1;
                    for t in 0..4 {
                        ub[t] += u4[t];
                    }
                    continue;
                }
            }
            cells.push((k, u9));
            cells_pm.push((k, u4));
        }
        StockPrep {
            code: code.to_string(),
            cells,
            cells_pm,
            n_trades: 1,
            amount: 0.0,
            total_vol: 1.0,
            imb: 0.0,
            ret: 0.0,
            vol30: 0.0,
            vwap: 0.0,
            q95u: 0.0,
            sum_w: [1.0; 6],
        }
    }

    /// 恒等式: PP + PM + MP + MM 必须等于「不分方向」的同一族矩阵。
    #[test]
    fn split_sum_equals_full_matrix() {
        let stocks = vec![
            mk("A", &[(100, 3.0, 2.0), (101, 1.0, 4.0), (150, 0.0, 2.0)]),
            mk("B", &[(99, 5.0, 0.0), (200, 1.0, 1.0)]),
            mk("C", &[(100, 0.0, 0.0), (300, 2.0, 3.0), (301, 1.0, 1.0)]),
        ];
        let n = stocks.len();
        let m4 = compute_split_mats(&stocks, 1.0, PmWeight::Cnt);
        let (mats, _) = compute_matrices(&stocks, FAM_CNT);
        let cnt_idx = MATRIX_SPECS.iter().position(|s| s.name == "cnt_t1").unwrap();
        let full = &mats[cnt_idx];
        let mut worst = 0.0f32;
        for i in 0..n {
            for j in 0..n {
                let s = m4[PP][i * n + j] + m4[PM][i * n + j] + m4[MP][i * n + j] + m4[MM][i * n + j];
                let d = (s - full[i * n + j]).abs();
                if d > worst {
                    worst = d;
                }
            }
        }
        assert!(worst < 1e-5, "四分量之和与原矩阵最大差 {worst}");
        // 方向正确性: A 在桶 100 有买、B 在桶 99 有买 → B 领先 A，S[A][B] 应为 0、S[B][A] > 0
        assert!(m4[PP][0 * n + 1].abs() < 1e-9, "S[A][B] 应为 0");
        assert!(m4[PP][1 * n + 0] > 0.0, "S[B][A] 应 > 0");
    }
}
