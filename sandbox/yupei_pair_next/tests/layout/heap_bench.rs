#[path = "../../src/yupei_pair/topk_heap.rs"]
mod heap;
use heap::{Entry, FlatTop};
fn old_insert(a: &mut Vec<Entry>, v: Entry, t: &mut f32) {
    if !(v.0 > *t) {
        return;
    }
    if a.len() < 50 {
        a.push(v);
    } else {
        let mut min = f32::INFINITY;
        let mut last = 0;
        for (i, e) in a.iter().enumerate() {
            if e.0 <= min {
                min = e.0;
                last = i;
            }
        }
        a.remove(last);
        a.push(v);
    }
    *t = if a.len() < 50 {
        f32::NEG_INFINITY
    } else {
        a.iter().map(|e| e.0).fold(f32::INFINITY, f32::min)
    };
}
fn main() {
    let mut streams = Vec::new();
    for e in std::fs::read_dir("/tmp").unwrap() {
        let p = e.unwrap().path();
        let name = p.file_name().unwrap().to_string_lossy();
        if name.starts_with("yp_array_") && name.ends_with(".bin") {
            let bytes = std::fs::read(p).unwrap();
            if bytes.len() != 5107 * 8 {
                continue;
            }
            for c in bytes.chunks(5107 * 4) {
                streams.push(
                    c.chunks_exact(4)
                        .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
                        .collect::<Vec<_>>(),
                );
            }
        }
    }
    assert!(!streams.is_empty());
    let mut seed = 17u64;
    for case in 0..100 {
        streams.push(
            (0..5107)
                .map(|i| {
                    seed ^= seed << 13;
                    seed ^= seed >> 7;
                    seed ^= seed << 17;
                    match case % 5 {
                        0 => (seed % 17) as f32 - 8.0,
                        1 => {
                            if i % 2 == 0 {
                                0.0
                            } else {
                                -0.0
                            }
                        }
                        2 => {
                            if i % 3 == 0 {
                                f32::NAN
                            } else {
                                f32::INFINITY
                            }
                        }
                        3 => (i % 71) as f32,
                        _ => f32::from_bits(seed as u32),
                    }
                })
                .collect(),
        );
    }
    for s in &streams {
        let mut a = Vec::new();
        let mut t = f32::NEG_INFINITY;
        let mut b = FlatTop::new(1, 50);
        for (i, &v) in s.iter().enumerate() {
            let e = (v, (i / 256) as i32, (256 + i % 256) as i32);
            old_insert(&mut a, e, &mut t);
            if v > b.threshold(0) {
                b.push(0, e);
            }
        }
        a.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        b.sort_rows();
        assert_eq!(a.len(), b.row(0).len());
        for (x, y) in a.iter().zip(b.row(0)) {
            assert_eq!(x.0.to_bits(), y.0.to_bits());
            assert_eq!((x.1, x.2), (y.1, y.2));
        }
    }
    println!("{} streams all entries and order exact", streams.len());
    if std::env::args().any(|a| a == "--check-only") { return; }
    for mode in [0, 1, 1, 0] {
        let t = std::time::Instant::now();
        for _ in 0..300 {
            for s in &streams {
                let mut a = Vec::new();
                let mut th = f32::NEG_INFINITY;
                let mut b = FlatTop::new(1, 50);
                for (i, &v) in s.iter().enumerate() {
                    let e = (v, (i / 256) as i32, (256 + i % 256) as i32);
                    if mode == 0 {
                        old_insert(&mut a, e, &mut th)
                    } else if v > b.threshold(0) {
                        b.push(0, e)
                    }
                }
                std::hint::black_box((a, b));
            }
        }
        println!("mode {mode} seconds {}", t.elapsed().as_secs_f64());
    }
}
