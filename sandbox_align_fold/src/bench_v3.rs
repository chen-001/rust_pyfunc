//! 单线程测第三版无交叉配对核：基准 / 提速(32位) / 提速(16位)，并拆出权重调用与倒推的开销。只计时，不产出因子。
#[path = "fast_csv_reader.rs"] mod fast_csv_reader;
#[path = "fold.rs"] mod fold;
#[path = "v3_letters.rs"] mod v3_letters;
#[path = "v3_kernels.rs"] mod v3_kernels;
use std::time::Instant;
use v3_kernels::*;
fn main() {
    let tr = fast_csv_reader::read_trade_fast("000001", 20241231).unwrap();
    let mk = fast_csv_reader::read_market_fast("000001", 20241231).unwrap();
    let w = v3_letters::windows()[0].with_bar(30);
    let b = v3_letters::Bars::new(&tr, &mk, &w).unwrap();
    let vs = v3_letters::variants();
    let s = b.seq(&vs[1], true).unwrap();
    let c = make_ctx(&s, None);
    println!("n={} 活跃={}", c.n, c.act.len());
    let reps = 50;
    let mut k = Kern::new();
    for rule in [0usize, 1] {
        let h = rule_h(rule);
        let wf = |a: usize, b: usize| weight(&c, rule, a, b);
        let t = Instant::now();
        for _ in 0..reps { k.base.solve(c.n, h, &c.active, &wf); }
        let tb = t.elapsed().as_secs_f64() / reps as f64 * 1e3;
        let t = Instant::now();
        for _ in 0..reps { k.fast.solve(&c.act, &c.side, h, &wf); }
        let tf = t.elapsed().as_secs_f64() / reps as f64 * 1e3;
        let t = Instant::now();
        for _ in 0..reps { k.fast16.solve(&c.act, &c.side, h, &wf); }
        let t16 = t.elapsed().as_secs_f64() / reps as f64 * 1e3;
        let mut p = vec![-1i32; c.n + 1];
        let t = Instant::now();
        for _ in 0..reps { k.fast.traceback(&c.act, &wf, &mut p); }
        let ttb = t.elapsed().as_secs_f64() / reps as f64 * 1e3;
        let t = Instant::now();
        let mut acc = 0i64;
        for _ in 0..reps { for j in 0..c.act.len() { for kk in 0..j { acc += wf(c.act[kk], c.act[j]) as i64; } } }
        let tw = t.elapsed().as_secs_f64() / reps as f64 * 1e3 / 2.0;
        println!("规则{rule}: 基准 {tb:.3} ms, 提速32 {tf:.3} ms, 提速16 {t16:.3} ms, 倒推 {ttb:.3} ms, 权重调用约 {tw:.3} ms ({acc})");
    }
}
