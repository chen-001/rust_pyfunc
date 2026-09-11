//! store8 自测：与 Python 侧抽出的 npy 夹具逐位对账，并测单因子读取耗时。
use std::time::Instant;

use tail_perf_bench::{npy, store8::Store8};

fn main() {
    let meta = std::env::var("STORE_META").unwrap_or("/home/chenzongwei/neu_lab/store_meta".into());
    let store = std::env::var("STORE_DIR")
        .unwrap_or("/hdd/user_home_unsafe/chenzongwei/factor_store_yupei_dist".into());
    let lab = std::env::var("LAB_DIR").unwrap_or("/home/chenzongwei/neu_lab/data_yupei_real".into());

    let s = Store8::open(&meta, &store).expect("打开 store8");
    println!("store8: T={} N={} 因子={}", s.t, s.n, s.n_factors());

    let names = std::fs::read_to_string(format!("{lab}/sample_names.txt")).unwrap();
    let names: Vec<&str> = names.split_whitespace().collect();
    let mut pass = 0;
    let mut bad = 0usize;
    let t0 = Instant::now();
    for nm in names.iter().take(16) {
        let col = s.factor_names.iter().position(|x| x == nm).expect("因子名不在库中");
        let got = s.read_factor(col).expect("读因子失败");
        let want = npy::as_f32_mat(npy::load(&format!("{lab}/factor_{nm}.npy")));
        let mut mm = 0usize;
        for (a, b) in got.iter().zip(want.iter()) {
            let same = (a.is_nan() && b.is_nan()) || a.to_bits() == b.to_bits();
            if !same {
                mm += 1;
            }
        }
        if mm == 0 { pass += 1 } else { bad += mm; }
        println!("  {nm}: {}", if mm == 0 { "PASS".to_string() } else { format!("FAIL {mm} 格") });
    }
    println!("对账：PASS {pass}/{}  不一致格数 {bad}  用时 {:.1}s", names.len().min(16), t0.elapsed().as_secs_f64());

    // 顺序读 64 个因子的吞吐（HDD 冷读）
    let t0 = Instant::now();
    let k = 64.min(s.n_factors());
    let mut acc = 0.0f64;
    for c in 0..k {
        let m = s.read_factor(c).unwrap();
        acc += m[[0, 0]].is_nan() as u8 as f64;
    }
    let dt = t0.elapsed().as_secs_f64();
    println!("顺序读 {k} 个因子: {dt:.2}s → {:.3}s/因子（{:.0} MB/s 有效值段）", dt / k as f64,
             (k as f64 * 5.87e6) / dt / 1e6);
    println!("(哨兵 {acc})");
}
