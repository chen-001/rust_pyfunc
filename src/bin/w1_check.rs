// w1dist 验证入口: 读 /tmp/.../rho_rvol_t1.f32 当"有向矩阵"喂给 IndicatorCtx,
// 喂一半(rho/2)使 ctx.symmetric 内部 dir+dir^T 恰好还原 rho（rho 本身对称）。
// 用法: w1_check <rho_f32_path> <out_txt>
// 输出: stdout 打印 n、列数、各矩阵耗时、w1_rvol_t1 校验和; out 文件逐行 w1 值。
use rust_pyfunc::yupei_dist::indicator_ctx::{IndicatorCtx, MatrixSet};
use rust_pyfunc::yupei_dist::indicators;
use rust_pyfunc::yupei_dist::matrix_stage::StockStats;
use std::collections::HashMap;
use std::time::Instant;

fn read_f32(path: &str) -> Vec<f32> {
    let bytes = std::fs::read(path).expect("read matrix file");
    assert!(bytes.len() % 4 == 0);
    let mut v = Vec::with_capacity(bytes.len() / 4);
    for c in bytes.chunks_exact(4) {
        v.push(f32::from_le_bytes([c[0], c[1], c[2], c[3]]));
    }
    v
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = &args[1];
    let out = &args[2];
    let dir = read_f32(path);
    let n = (dir.len() as f64).sqrt() as usize;
    assert_eq!(n * n, dir.len());
    let half: Vec<f32> = dir.iter().map(|&v| v * 0.5).collect();
    let mut mats = HashMap::new();
    // 只注册 rvol_t1 一张验证（compute 遍历 MATRIX_LIST 但缺矩阵会 skip）
    mats.insert("rvol_t1".to_string(), half.clone());
    let codes: Vec<String> = (0..n).map(|i| format!("{i:06}")).collect();
    let set = MatrixSet {
        n,
        codes,
        stats: vec![StockStats::default(); n],
        mats,
        industry: None,
    };
    let ctx = IndicatorCtx::new(&set, None);
    let t0 = Instant::now();
    let results = indicators::generated::w1dist::compute(&ctx);
    let ms = t0.elapsed().as_secs_f64() * 1000.0;
    let w1 = &results
        .iter()
        .find(|r| r.name == "w1_rvol_t1")
        .expect("w1_rvol_t1")
        .values;
    let sum: f64 = w1.iter().map(|&v| v as f64).filter(|v| v.is_finite()).sum();
    let nvalid = w1.iter().filter(|v| v.is_finite()).count();
    println!("n={n} cols={} w1_sum={sum:.6} valid={nvalid} total_ms={ms:.1}", results.len());
    std::fs::write(
        out,
        w1.iter()
            .map(|v| {
                if v.is_finite() {
                    format!("{v:.8}")
                } else {
                    "nan".to_string()
                }
            })
            .collect::<Vec<_>>()
            .join("\n")
            + "\n",
    )
    .unwrap();
}
