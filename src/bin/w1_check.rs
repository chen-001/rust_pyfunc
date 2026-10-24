// w1dist/jsmarg 验证入口: 读 /tmp/.../rho_rvol_t1.f32 当"有向矩阵"喂给 IndicatorCtx,
// 喂一半(rho/2)使 ctx.symmetric 内部 dir+dir^T 恰好还原 rho（rho 本身对称）。
// 用法: w1_check <rho_f32_path> <out_txt>
// 输出: stdout 打印 n、列数、耗时; out 文件逐行 "w1,w1_asym,jsmarg,consens"。
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
    mats.insert("rvol_t1".to_string(), half);
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
    let wres = indicators::generated::w1dist::compute(&ctx);
    let w_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let t1 = Instant::now();
    let jres = indicators::generated::jsmarg::compute(&ctx);
    let j_ms = t1.elapsed().as_secs_f64() * 1000.0;

    fn get<'a>(
        res: &'a [rust_pyfunc::yupei_dist::indicator_ctx::IndicatorResult],
        nm: &str,
    ) -> &'a Vec<f32> {
        &res.iter().find(|r| r.name == nm).expect(nm).values
    }
    let w1 = get(&wres, "w1_rvol_t1");
    let asym = get(&wres, "w1_rvol_t1_asym");
    let js = get(&jres, "jsmarg_rvol_t1");
    let cons = get(&jres, "consens_rvol_t1_top20");
    println!(
        "n={n} wcols={} jcols={} w_ms={w_ms:.0} j_ms={j_ms:.0}",
        wres.len(),
        jres.len()
    );
    let fmt = |v: f32| {
        if v.is_finite() {
            format!("{v:.8}")
        } else {
            "nan".to_string()
        }
    };
    let mut lines = String::with_capacity(n * 40);
    for i in 0..n {
        lines.push_str(&format!(
            "{},{},{},{}\n",
            fmt(w1[i]),
            fmt(asym[i]),
            fmt(js[i]),
            fmt(cons[i])
        ));
    }
    std::fs::write(out, lines).unwrap();
}
