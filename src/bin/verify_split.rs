//! 自检: 四分量之和必须等于「不分方向」的同一族矩阵；方向语义必须是行领先列。
use rust_pyfunc::yupei_dist::matrix_stage::{compute_matrices, StockPrep, FAM_CNT, MATRIX_SPECS, N_U};
use rust_pyfunc::yupei_pair::split_mat::{compute_split_mats, PmWeight, MM, MP, PM, PP};

fn mk(code: &str, buckets: &[(u32, f32, f32)]) -> StockPrep {
    let mut cells: Vec<(u32, [f32; N_U])> = Vec::new();
    let mut cells_pm: Vec<(u32, [f32; 4])> = Vec::new();
    for &(k, p, m) in buckets {
        let mut u9 = [0.0f32; N_U];
        u9[0] = p + m;
        let mut u4 = [0.0f32; 4];
        u4[0] = p;
        u4[1] = m;
        if let Some((bk, ua)) = cells.last_mut() {
            if *bk == k {
                for t in 0..N_U { ua[t] += u9[t]; }
                let ub = &mut cells_pm.last_mut().unwrap().1;
                for t in 0..4 { ub[t] += u4[t]; }
                continue;
            }
        }
        cells.push((k, u9));
        cells_pm.push((k, u4));
    }
    StockPrep {
        code: code.to_string(), cells, cells_pm, n_trades: 1,
        amount: 0.0, total_vol: 1.0, imb: 0.0, ret: 0.0, vol30: 0.0, vwap: 0.0, q95u: 0.0,
        sum_w: [1.0; 6],
    }
}

fn main() {
    let stocks = vec![
        mk("A", &[(100, 3.0, 2.0), (101, 1.0, 4.0), (150, 0.0, 2.0)]),
        mk("B", &[(99, 5.0, 0.0), (200, 1.0, 1.0)]),
        mk("C", &[(100, 0.0, 0.0), (300, 2.0, 3.0), (301, 1.0, 1.0)]),
    ];
    let n = stocks.len();
    let m4 = compute_split_mats(&stocks, 1.0, PmWeight::Cnt);
    let (mats, _) = compute_matrices(&stocks, FAM_CNT);
    let ci = MATRIX_SPECS.iter().position(|s| s.name == "cnt_t1").unwrap();
    let full = &mats[ci];
    let mut worst = 0.0f32;
    for i in 0..n {
        for j in 0..n {
            let s = m4[PP][i * n + j] + m4[PM][i * n + j] + m4[MP][i * n + j] + m4[MM][i * n + j];
            let d = (s - full[i * n + j]).abs();
            if d > worst { worst = d; }
        }
    }
    println!("四分量之和 vs cnt_t1 最大绝对差 = {worst:.3e}");
    println!("S[A][B] (A 桶100 买, B 桶99 买 → B 领先 A, 应为 0) = {:.6}", m4[PP][0 * n + 1]);
    println!("S[B][A] (应为 e^(-1/τ) 量级 > 0)                    = {:.6}", m4[PP][1 * n + 0]);
    for i in 0..n {
        let row: Vec<String> = (0..n).map(|j| format!("{:8.4}", full[i * n + j])).collect();
        println!("full[{i}] = [{}]", row.join(", "));
    }
}
