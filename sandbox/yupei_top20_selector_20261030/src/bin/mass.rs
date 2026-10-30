#![allow(dead_code)]
use std::{hint::black_box, time::Instant};
#[path = "../../../../src/yupei_pair/bitonic_sort.rs"] mod bitonic_sort;
#[path = "../../../../src/yupei_pair/simd.rs"] mod baseline;
#[path = "../mass_candidate.rs"] mod candidate;

type Mass = [Vec<f32>; 6];
fn apply(fast: bool, state: &mut Mass, h: &[f32], g: &[f32], ih: f32, ig: f32, dh: bool, dg: bool) {
    let [a,b,c,d,e,f] = state;
    if fast { candidate::mass_acc_hg_512(a,b,c,d,e,f,h,g,ih,ig,dh,dg); }
    else { baseline::mass_acc_hg(a,b,c,d,e,f,h,g,ih,ig,dh,dg); }
}
fn main() {
    let n = 5195;
    let pairs = (0..128).map(|r| {
        let h = (0..n).map(|i| if (i + r) % 31 == 0 { f32::NEG_INFINITY } else { ((i * 8171 + r * 8777) as f32).sin() }).collect::<Vec<_>>();
        let g = h.iter().enumerate().map(|(i, &v)| if i % 11 == 0 { 0.0 } else { v - 0.3 }).collect::<Vec<_>>();
        (h, g)
    }).collect::<Vec<_>>();
    let mut old: Mass = std::array::from_fn(|_| vec![0.0; n]);
    let mut new = old.clone();
    for len in [1,7,8,15,16,17,31,32,1000,5195] {
        for (h,g) in &pairs {
            for (dh,dg) in [(true,true), (true,false), (false,true), (false,false)] {
                apply(false,&mut old,&h[..len],&g[..len],0.005,0.0003,dh,dg);
                apply(true,&mut new,&h[..len],&g[..len],0.005,0.0003,dh,dg);
            }
        }
    }
    for (a,b) in old.iter().flatten().zip(new.iter().flatten()) {
        assert_eq!(a.to_bits(), b.to_bits());
    }
    let edge = [0.0, -0.0, f32::INFINITY, f32::NEG_INFINITY,
        f32::from_bits(0x7fc0_0001), f32::from_bits(0xffc1_2345),
        f32::from_bits(1), -f32::from_bits(1), 1e-30, -1e-30, 1e30, -1e30];
    let h: Vec<_> = (0..n).map(|i| edge[i % edge.len()]).collect();
    let g: Vec<_> = (0..n).map(|i| edge[(i / edge.len()) % edge.len()]).collect();
    for len in [1,7,8,15,16,17,31,32,1000,5195] {
        for (dh,dg) in [(true,true), (true,false), (false,true), (false,false)] {
            let mut a: Mass = std::array::from_fn(|_| vec![-0.0; n]);
            let mut b = a.clone();
            apply(false,&mut a,&h[..len],&g[..len],0.005,0.0003,dh,dg);
            apply(true,&mut b,&h[..len],&g[..len],0.005,0.0003,dh,dg);
            for (a,b) in a.iter().flatten().zip(b.iter().flatten()) { assert_eq!(a.to_bits(), b.to_bits()); }
        }
    }
    println!("{{\"kind\":\"correctness\",\"bit_mismatches\":0}}");
    for fast in [false,true,true,false] {
        let mut state: Mass = std::array::from_fn(|_| vec![0.0; n]);
        let start = Instant::now();
        for _ in 0..400 {
            for (h,g) in &pairs { apply(fast, &mut state, black_box(h), black_box(g),0.005,0.0003,true,true); }
        }
        let digest: f64 = black_box(&state).iter().flatten().map(|&x| x as f64).sum();
        println!("{{\"kind\":\"timing\",\"candidate\":{},\"calls\":51200,\"seconds\":{},\"digest\":{}}}",fast,start.elapsed().as_secs_f64(),digest);
    }
}
