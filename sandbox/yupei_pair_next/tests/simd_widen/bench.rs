// Local mass-update microbenchmark on captured real h/g arrays, NOT whole-day timing.
// 50 threads; fixed normalization constants; verify bit equality before timing.
#![allow(dead_code)]
#[path = "../../../../src/yupei_pair/simd.rs"]
mod old;
mod wide;
fn run(a: &mut [Vec<f32>; 6], h: &[f32], g: &[f32], mode: usize, dh: bool, dg: bool) {
    let [e, m, m2, s, g1, g2] = a;
    if mode == 0 {
        old::mass_acc_hg(e, m, m2, s, g1, g2, h, g, 0.031, 0.017, dh, dg)
    } else {
        unsafe { wide::mass_wide(e, m, m2, s, g1, g2, h, g, 0.031, 0.017, dh, dg) }
    }
}
fn main() {
    assert!(is_x86_feature_detected!("avx512f"));
    let mut pairs = Vec::new();
    for e in std::fs::read_dir("/tmp").unwrap() {
        let p = e.unwrap().path();
        let name = p.file_name().unwrap().to_string_lossy();
        if name.starts_with("yp_array_") && name.ends_with(".bin") {
            let bytes = std::fs::read(p).unwrap();
            if bytes.len() != 5107 * 2 * 4 {
                continue;
            }
            let v: Vec<f32> = bytes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect();
            pairs.push(v);
        }
    }
    assert!(
        !pairs.is_empty(),
        "No /tmp/yp_array_*.bin input captures found"
    );
    println!("real pairs {}", pairs.len());
    for n in [1, 7, 8, 9, 15, 16, 17, 5107] {
        for dh in [false, true] {
            for dg in [false, true] {
                let mut a: [Vec<f32>; 6] = std::array::from_fn(|_| vec![0.0; n]);
                let mut b = a.clone();
                for v in &pairs {
                    run(&mut a, &v[..n], &v[5107..5107 + n], 0, dh, dg);
                    run(&mut b, &v[..n], &v[5107..5107 + n], 1, dh, dg);
                }
                for (x, y) in a.iter().flatten().zip(b.iter().flatten()) {
                    assert_eq!(x.to_bits(), y.to_bits());
                }
            }
        }
    }
    let pairs = std::sync::Arc::new(pairs);
    for mode in [0, 1, 1, 0] {
        let barrier = std::sync::Arc::new(std::sync::Barrier::new(51));
        let mut workers = Vec::new();
        for _ in 0..50 {
            let data = pairs.clone();
            let b = barrier.clone();
            workers.push(std::thread::spawn(move || {
                let mut a: [Vec<f32>; 6] = std::array::from_fn(|_| vec![0.0; 5107]);
                b.wait();
                for _ in 0..3000 {
                    for v in data.iter() {
                        run(
                            std::hint::black_box(&mut a),
                            &v[..5107],
                            &v[5107..],
                            mode,
                            true,
                            true,
                        )
                    }
                }
                std::hint::black_box(a);
            }));
        }
        let t = std::time::Instant::now();
        barrier.wait();
        for w in workers {
            w.join().unwrap();
        }
        println!(
            "50 threads mode {mode} seconds {}",
            t.elapsed().as_secs_f64()
        );
    }
    println!("all bits equal");
}
