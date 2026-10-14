// hm076 单股单日拆阶段计时（sandbox 复刻，单进程）
#[path = "data_paths.rs"]
mod data_paths;
#[path = "features_f64.rs"]
mod features_f64;
mod time_series {
    pub mod lyapunov {
        pub fn lyapunov_exponent_inner<T>(_v: T) -> f64 {
            panic!("lyapunov path unexpectedly taken")
        }
    }
}
#[path = "hm076.rs"]
mod hm076;

use std::sync::atomic::Ordering;
use std::time::Instant;

const STAGE_NAMES: [&str; 10] = [
    "读原始数据 CSV + 解析",
    "build_day 变换/排序/分时段",
    "one() 两次合计",
    "  └ ultra_sorted 前半（建订单/排序/分组）",
    "  └ ultra_sorted 主循环（逐记录算指标）",
    "  └ ultra_sorted 末段（结果回填 n*27）",
    "  └ one(): dropna + Side::build",
    "  └ price_stats_v3",
    "  └ features_f64（4 块）",
    "  └ one(): 16 个 extend 块",
];

fn reset() {
    for i in 0..10 { hm076::T[i].store(0, Ordering::Relaxed); }
    for i in 0..2 { hm076::P[i].store(0, Ordering::Relaxed); }
    for c in [&hm076::C_GROUPS, &hm076::C_RECORDS, &hm076::C_ELEMS, &hm076::C_SORTS, &hm076::C_ALLOCS,
              &hm076::C_P_GROUPS, &hm076::C_P_RECORDS, &hm076::C_P_ELEMS, &hm076::C_P_SORTS] {
        c.store(0, Ordering::Relaxed);
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let code = args.get(1).cloned().unwrap_or_else(|| "000001".to_string());
    let date: i64 = args.get(2).map(|s| s.parse().unwrap()).unwrap_or(20220819);
    let reps: usize = args.get(3).map(|s| s.parse().unwrap()).unwrap_or(3);
    let dump = args.get(4).map(|s| s == "dump").unwrap_or(false);

    let mut best_total = f64::MAX;
    let mut best_stages = [0f64; 10];
    let mut best_p = [0f64; 2];
    let mut snap = [0u64; 9];
    let mut out_vals: Vec<f32> = Vec::new();
    let mut all_totals: Vec<f64> = Vec::new();

    for _ in 0..reps {
        reset();
        let t = Instant::now();
        let out = hm076::compute_stock(&code, date).expect("compute_stock 返回 None");
        let total = t.elapsed().as_secs_f64() * 1000.0;
        all_totals.push(total);
        let mut st = [0f64; 10];
        for i in 0..10 { st[i] = hm076::T[i].load(Ordering::Relaxed) as f64 / 1e6; }
        let pv = [hm076::P[0].load(Ordering::Relaxed) as f64 / 1e6,
                  hm076::P[1].load(Ordering::Relaxed) as f64 / 1e6];
        if total < best_total {
            best_total = total;
            best_stages = st;
            best_p = pv;
            snap = [hm076::C_GROUPS.load(Ordering::Relaxed), hm076::C_RECORDS.load(Ordering::Relaxed),
                    hm076::C_ELEMS.load(Ordering::Relaxed), hm076::C_SORTS.load(Ordering::Relaxed),
                    hm076::C_ALLOCS.load(Ordering::Relaxed), hm076::C_P_GROUPS.load(Ordering::Relaxed),
                    hm076::C_P_RECORDS.load(Ordering::Relaxed), hm076::C_P_ELEMS.load(Ordering::Relaxed),
                    hm076::C_P_SORTS.load(Ordering::Relaxed)];
            out_vals = out;
        }
    }

    let tot_s: Vec<String> = all_totals.iter().map(|x| format!("{:.0}", x)).collect();
    println!("== hm076 {} {}  重跑 {} 次，并行数=0（单进程，RAYON_NUM_THREADS=1） ==", code, date, reps);
    println!("每次总耗时: [{}] ms，取最快 {:.1} ms", tot_s.join(", "), best_total);
    println!("输出 {} 个值", out_vals.len());
    for i in 0..10 {
        println!("  T[{}] {:40} {:10.1} ms  {:5.1}%", i, STAGE_NAMES[i], best_stages[i], best_stages[i] / best_total * 100.0);
    }
    println!("  T[0]+T[1]+T[2] = {:.1} ms；T[3..9] 是两次 one() 之和 = {:.1} ms",
        best_stages[0] + best_stages[1] + best_stages[2],
        best_stages[3..10].iter().sum::<f64>());
    println!("  price_stats_v3 细分：组准备 P[0] {:.1} ms（{:.1}%），逐记录主循环 P[1] {:.1} ms（{:.1}%）",
        best_p[0], best_p[0] / best_total * 100.0, best_p[1], best_p[1] / best_total * 100.0);
    println!("  ultra_sorted 计数器：处理组 {}  记录 {}  距离元素 {}  sort {}  每记录 2 次 Vec 申请共 {}",
        snap[0], snap[1], snap[2], snap[3], snap[4]);
    println!("  price_stats_v3 计数器：处理组 {}  记录 {}  距离元素 {}  sort/select {}",
        snap[5], snap[6], snap[7], snap[8]);
    println!("  每个因子平均 {:.3} ms，是 1 秒标准的 {:.1} 倍", best_total / out_vals.len() as f64, best_total / 1000.0);
    if dump {
        let s: Vec<String> = out_vals.iter().map(|&x| format!("{:.17e}", x as f64)).collect();
        println!("DUMP={}", s.join(","));
    }
}
