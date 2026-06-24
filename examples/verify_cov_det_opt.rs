//! cov_det/lu_det 栈数组优化的一致性验证。
//! 对比栈版与堆版在各种 d 值下的结果，证明优化不改变数值。
//! 运行：cargo run --release --example verify_cov_det_opt

use ndarray::Array2;

fn main() {
    println!("=== cov_det 栈数组优化一致性验证 ===\n");

    // 构造随机测试数据（d=10 和 d=20）
    for &d in &[3usize, 5, 10, 20] {
        for &n in &[d + 1, d * 2, d * 5] {
            let mut data = Vec::with_capacity(n * d);
            // 用确定性伪随机（避免随机种子影响）
            for i in 0..n * d {
                let v = ((i as f64 * 1.234567).sin() * 100.0 + (i as f64 * 0.987654).cos() * 50.0);
                data.push(v);
            }

            // 栈版（优化后的 cov_det）
            let det_stack = cov_det_stack(&data, n, d);
            // 堆版（原始 cov_det 逻辑）
            let det_heap = cov_det_heap(&data, n, d);

            let rel_diff = if det_heap.abs() > 1e-20 {
                ((det_stack - det_heap) / det_heap).abs()
            } else {
                (det_stack - det_heap).abs()
            };

            let ok = rel_diff < 1e-10 || (det_stack == 0.0 && det_heap == 0.0);
            println!(
                "d={:2} n={:3}: 栈版={:+.6e} 堆版={:+.6e} 相对误差={:.2e} {}",
                d, n, det_stack, det_heap, rel_diff, if ok { "✅" } else { "❌" }
            );
        }
    }

    // 测试 lu_det_stack vs lu_det
    println!("\n=== lu_det 一致性 ===");
    for &n in &[2usize, 5, 10, 20] {
        let mut a = Vec::with_capacity(n * n);
        for i in 0..n * n {
            a.push(((i as f64 * 3.14159).sin() + 2.0).abs());
        }
        let d1 = lu_det_stack(&a, n);
        let d2 = lu_det_heap(&a, n);
        let rel = if d2.abs() > 1e-20 { ((d1 - d2) / d2).abs() } else { (d1 - d2).abs() };
        println!("n={:2}: 栈版={:.6e} 堆版={:.6e} 误差={:.2e} {}", n, d1, d2, rel, if rel < 1e-12 { "✅" } else { "❌" });
    }

    println!("\n验证完成");
}

// ===== 复制栈版实现（与 observable_order_metrics.rs 的优化版一致）=====
const MAX_D: usize = 20;

fn cov_det_stack(data: &[f64], n: usize, d: usize) -> f64 {
    if n < d || d == 0 || n == 0 || d > MAX_D {
        return cov_det_heap(data, n, d);
    }
    let mut col_mean = [0.0f64; MAX_D];
    for i in 0..n {
        for j in 0..d {
            col_mean[j] += data[i * d + j];
        }
    }
    for j in 0..d {
        col_mean[j] /= n as f64;
    }
    let mut cov = [0.0f64; MAX_D * MAX_D];
    for i in 0..n {
        for j in 0..d {
            let dj = data[i * d + j] - col_mean[j];
            for k in j..d {
                let dk = data[i * d + k] - col_mean[k];
                cov[j * d + k] += dj * dk;
            }
        }
    }
    for j in 0..d {
        for k in 0..d {
            if k < j {
                cov[j * d + k] = cov[k * d + j];
            } else {
                cov[j * d + k] /= n as f64;
            }
        }
    }
    lu_det_stack(&cov[..d * d], d)
}

fn cov_det_heap(data: &[f64], n: usize, d: usize) -> f64 {
    if n < d || d == 0 || n == 0 {
        return f64::NAN;
    }
    let mut col_mean = vec![0.0f64; d];
    for i in 0..n {
        for j in 0..d {
            col_mean[j] += data[i * d + j];
        }
    }
    for j in 0..d {
        col_mean[j] /= n as f64;
    }
    let mut cov = vec![0.0f64; d * d];
    for i in 0..n {
        for j in 0..d {
            let dj = data[i * d + j] - col_mean[j];
            for k in j..d {
                let dk = data[i * d + k] - col_mean[k];
                cov[j * d + k] += dj * dk;
            }
        }
    }
    for j in 0..d {
        for k in 0..d {
            if k < j {
                cov[j * d + k] = cov[k * d + j];
            } else {
                cov[j * d + k] /= n as f64;
            }
        }
    }
    lu_det_heap(&cov, d)
}

fn lu_det_stack(a: &[f64], n: usize) -> f64 {
    if n > MAX_D {
        return lu_det_heap(a, n);
    }
    let mut m = [0.0f64; MAX_D * MAX_D];
    m[..n * n].copy_from_slice(&a[..n * n]);
    let mut det = 1.0f64;
    for k in 0..n {
        let pivot = m[k * n + k];
        if pivot.abs() < 1e-300 || !pivot.is_finite() {
            return 0.0;
        }
        det *= pivot;
        for i in (k + 1)..n {
            let factor = m[i * n + k] / pivot;
            for j in k..n {
                m[i * n + j] -= factor * m[k * n + j];
            }
        }
    }
    det
}

fn lu_det_heap(a: &[f64], n: usize) -> f64 {
    let mut m = a.to_vec();
    let mut det = 1.0f64;
    for k in 0..n {
        let pivot = m[k * n + k];
        if pivot.abs() < 1e-300 || !pivot.is_finite() {
            return 0.0;
        }
        det *= pivot;
        for i in (k + 1)..n {
            let factor = m[i * n + k] / pivot;
            for j in k..n {
                m[i * n + j] -= factor * m[k * n + j];
            }
        }
    }
    det
}
