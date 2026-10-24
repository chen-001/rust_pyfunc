// maxst 验证入口: 读生产备份目录 <backup>/<date>/mats/*.bin 当"有向矩阵"喂给 IndicatorCtx。
// ctx.symmetric = dir+dirᵀ（生产未归一），maxst 内部再做行强度归一 → 验证归一化后 Prim 不退化成星形。
// 用法: maxst_check <mats_dir> <mat_name> <out_deg_txt>
// 输出: stdout 打印 n、deg 校验和、deg>5/>20 股数、耗时; out 文件逐行 deg。
use rust_pyfunc::yupei_dist::indicator_ctx::{IndicatorCtx, MatrixSet};
use rust_pyfunc::yupei_dist::indicators::generated::maxst;
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
    let dir = &args[1];
    let mat = &args[2];
    let out = &args[3];
    let data = read_f32(&format!("{dir}/{mat}.bin"));
    let n = (data.len() as f64).sqrt() as usize;
    assert_eq!(n * n, data.len());
    let mut mats = HashMap::new();
    mats.insert(mat.to_string(), data);
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
    let results = maxst::compute(&ctx);
    let ms = t0.elapsed().as_secs_f64() * 1000.0;
    let deg = &results
        .iter()
        .find(|r| r.name == format!("maxst_{mat}_deg"))
        .unwrap_or_else(|| panic!("maxst_{mat}_deg"))
        .values;
    let sum: f64 = deg.iter().map(|&v| v as f64).sum();
    let maxd = deg.iter().cloned().fold(0.0f32, f32::max);
    let gt5 = deg.iter().filter(|&&v| v > 5.0).count();
    let gt20 = deg.iter().filter(|&&v| v > 20.0).count();
    let med = {
        let mut s = deg.clone();
        s.sort_by(|a, b| a.partial_cmp(b).unwrap());
        s[n / 2]
    };
    println!(
        "mat={mat} n={n} deg_sum={sum:.0} deg_max={maxd:.0} deg_med={med:.0} gt5={gt5} gt20={gt20} total_ms={ms:.1}"
    );
    std::fs::write(
        out,
        deg.iter().map(|v| format!("{v:.0}")).collect::<Vec<_>>().join("\n") + "\n",
    )
    .unwrap();
}
