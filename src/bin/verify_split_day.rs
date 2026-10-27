//! 真实一天: 四分量之和 vs 现有 cnt_t1 矩阵的差。
use rust_pyfunc::yupei_dist::compute::compute_yupei_dist_partial_need;
use rust_pyfunc::yupei_dist::matrix_stage::FAM_CNT;
use rust_pyfunc::yupei_pair::split_mat::{compute_split_mats, PmWeight, MM, MP, PM, PP};

fn main() {
    let date: i64 = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(20251031);
    let (_codes, n, stocks, _stats, hm, _ind) =
        compute_yupei_dist_partial_need(date, FAM_CNT).expect("读数据失败");
    let full = hm.get("cnt_t1").expect("无 cnt_t1");
    let m4 = compute_split_mats(&stocks, 1.0, PmWeight::Cnt);
    let (mut worst_abs, mut worst_rel) = (0.0f64, 0.0f64);
    let (mut sum_abs, mut sum_ref, mut cnt) = (0.0f64, 0.0f64, 0u64);
    for i in 0..n {
        for j in 0..n {
            let k = i * n + j;
            let s = (m4[PP][k] + m4[PM][k] + m4[MP][k] + m4[MM][k]) as f64;
            let f = full[k] as f64;
            let d = (s - f).abs();
            if d > worst_abs { worst_abs = d; }
            let rel = d / f.abs().max(1e-12);
            if rel > worst_rel && f.abs() > 1e-6 { worst_rel = rel; }
            sum_abs += d;
            sum_ref += f.abs();
            cnt += 1;
        }
    }
    // 逐笔口径核对: 方向拆分场的笔数之和应当等于 cnt 场之和
    let mut sum_cnt = 0.0f64;
    let mut sum_pm = 0.0f64;
    let mut ncell = 0usize;
    let mut ncell_pm = 0usize;
    for st in &stocks {
        for c in &st.cells {
            sum_cnt += c.1[0] as f64;
        }
        for c in &st.cells_pm {
            sum_pm += (c.1[0] + c.1[1]) as f64;
        }
        ncell += st.cells.len();
        ncell_pm += st.cells_pm.len();
    }
    println!("Σu_cnt={sum_cnt:.0}  Σ(u_P+u_M)={sum_pm:.0}  比值={:.6}", sum_pm / sum_cnt.max(1.0));
    println!("cell 数: cells={ncell} cells_pm={ncell_pm}");
    println!("n={n} 格子={cnt}");
    println!("最大绝对差 = {worst_abs:.4e}");
    println!("最大相对差 = {worst_rel:.4e}");
    println!("平均绝对差 = {:.4e}   平均量级 = {:.4e}", sum_abs / cnt as f64, sum_ref / cnt as f64);
    println!("对角线样例: 我方 {:?}", (0..3).map(|i| m4[PP][i*n+i] + m4[PM][i*n+i] + m4[MP][i*n+i] + m4[MM][i*n+i]).collect::<Vec<_>>());
    println!("             原版 {:?}", (0..3).map(|i| full[i*n+i]).collect::<Vec<_>>());
}
