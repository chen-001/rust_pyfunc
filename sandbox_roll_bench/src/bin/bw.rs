//! 裸内存带宽标定：分配/填充在计时区外，只计 triad 循环（读2写1）。
//! 用法: bw <线程数> <每线程MB> <重复次数>
use std::time::Instant;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let threads: usize = a.get(1).and_then(|s| s.parse().ok()).unwrap_or(96);
    let mb: usize = a.get(2).and_then(|s| s.parse().ok()).unwrap_or(88);
    let reps: usize = a.get(3).and_then(|s| s.parse().ok()).unwrap_or(5);
    let len = mb * 1024 * 1024 / 4;

    std::thread::scope(|sc| {
        let mut hs = Vec::new();
        for _ in 0..threads {
            hs.push(sc.spawn(move || {
                // 线程内分配 + 首次触碰（NUMA 本地）
                let mut x = vec![1.0f32; len];
                let mut y = vec![2.0f32; len];
                let mut z = vec![0.0f32; len];
                for i in 0..len {
                    z[i] = x[i] + y[i];
                }
                // 屏障对齐：用 channel 集合
                (x, y, z)
            }));
        }
        let bufs: Vec<(Vec<f32>, Vec<f32>, Vec<f32>)> =
            hs.into_iter().map(|h| h.join().unwrap()).collect();

        let t0 = Instant::now();
        std::thread::scope(|sc| {
            for (mut x, y, mut z) in bufs {
                sc.spawn(move || {
                    for _ in 0..reps {
                        for i in 0..len {
                            z[i] = x[i] + y[i];
                        }
                        x[0] = z[len - 1];
                    }
                });
            }
        });
        let dt = t0.elapsed().as_secs_f64();
        let gb = (threads * mb * 3 * reps) as f64 / 1024.0;
        println!(
            "×{} 线程 × {}MB × {}轮: {:.2}s → {:.1} GB/s 聚合（每线程 {:.2} GB/s）",
            threads,
            mb,
            reps,
            dt,
            gb / dt,
            gb / dt / threads as f64
        );
    });
}
