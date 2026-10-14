// 截面 fold 计算：numpy 版 vs 纯 Rust 版，同一块内存布局上对比
// fold(v) = |log(v) - 当日(行)截面 mean(log(v))|，非有限的 log 不参与均值
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let nd: usize = args[1].parse().unwrap();
    let ns: usize = args[2].parse().unwrap();
    let reps: usize = args.get(3).map(|s| s.parse().unwrap()).unwrap_or(3);
    let n = nd * ns;
    let mut v: Vec<f64> = Vec::with_capacity(n);
    // 造数：大部分为正的对数正态，少量 0 / 负 / NaN
    let mut x: u64 = 88172645463325252;
    for i in 0..n {
        x ^= x << 13; x ^= x >> 7; x ^= x << 17;
        let u = ((x >> 11) as f64) / ((1u64 << 53) as f64);
        let val = (u * 1e6) as f64;
        v.push(if i % 977 == 0 { 0.0 } else if i % 30011 == 0 { f64::NAN } else { val });
    }
    println!("panel {} x {} = {} cells, {:.1} MB", nd, ns, n, n as f64 * 4.0 / 1e6);

    let mut out: Vec<f64> = vec![0.0; n];
    let t0 = Instant::now();
    for _ in 0..reps {
        // 第一遍：逐行求 log 的均值
        let mut row = 0usize;
        while row < nd {
            let s = row * ns;
            let e = s + ns;
            let mut sum = 0.0f64;
            let mut cnt = 0u64;
            for i in s..e {
                let val = v[i];
                if val > 0.0 {
                    let l = (val as f64).ln();
                    if l.is_finite() { sum += l; cnt += 1; }
                }
            }
            let mean = if cnt > 0 { sum / cnt as f64 } else { f64::NAN };
            // 第二遍：写回 |log - mean|
            for i in s..e {
                let val = v[i];
                let l = if val > 0.0 { (val as f64).ln() } else { f64::NAN };
                out[i] = if l.is_finite() { ((l - mean).abs()) as f64 } else { f64::NAN };
            }
            row += 1;
        }
    }
    let el = t0.elapsed().as_secs_f64() / reps as f64;
    println!("rust_full_ms={:.1}  ns_per_cell={:.2}  checksum={:.6e}", el * 1000.0, el / n as f64 * 1e9, out[n / 2] as f64);
}
