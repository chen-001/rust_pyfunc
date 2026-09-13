use std::time::Instant;
const T: usize = 2276;
const N: usize = 5000;
const K: usize = 10;
const P: usize = 20;
const B: usize = 4;
fn main(){
    let total=T*N;
    let mut xdays=vec![0f64; total*K];
    for i in 0..xdays.len(){ xdays[i]=((i*2654435761usize)%1000003) as f64/1000003.0; }
    let mut ys: Vec<Vec<f64>>=(0..B).map(|b|{
        let mut v=vec![0f64; total];
        for i in 0..total { v[i]=((i*40503+b*7919)%999983) as f64/999983.0; }
        v
    }).collect();
    let mut xty=vec![0f64; B*P];
    let mut coef=vec![0f64; B*P];
    // synthetic precomputed xtx inverse diag to avoid nalgebra
    // baseline: per-slot loops, reading xdays per slot
    let t0=Instant::now();
    let mut acc=0f64;
    for t in 0..T {
        let xb=&xdays[t*N*K..(t+1)*N*K];
        for b in 0..B {
            let yb=&ys[b][t*N..(t+1)*N];
            let xr=&mut xty[b*P..b*P+P];
            xr.fill(0.0);
            for i in 0..N {
                let xrow=&xb[i*K..i*K+K];
                let yv=yb[i];
                for c in 0..K { xr[c]+=xrow[c]*yv; }
                xr[K]+=yv; // fake industry
            }
            // fake solve p^2
            for c in 0..P { coef[b*P+c]=xr[c]*0.001; }
            // prediction
            for i in 0..N {
                let xrow=&xb[i*K..i*K+K];
                let mut pred=coef[b*P+K];
                for c in 0..K { pred+=coef[b*P+c]*xrow[c]; }
                acc += yb[i]-pred;
            }
        }
    }
    let t1=t0.elapsed().as_secs_f64();
    // batched: date outer, position middle, slots inner; xrow loaded once
    let t2=Instant::now();
    let mut acc2=0f64;
    for t in 0..T {
        let xb=&xdays[t*N*K..(t+1)*N*K];
        xty.fill(0.0);
        for i in 0..N {
            let xrow=&xb[i*K..i*K+K];
            for b in 0..B {
                let yv=ys[b][t*N+i];
                let xr=&mut xty[b*P..b*P+P];
                for c in 0..K { xr[c]+=xrow[c]*yv; }
                xr[K]+=yv;
            }
        }
        for b in 0..B { for c in 0..P { coef[b*P+c]=xty[b*P+c]*0.001; } }
        for i in 0..N {
            let xrow=&xb[i*K..i*K+K];
            for b in 0..B {
                let mut pred=coef[b*P+K];
                for c in 0..K { pred+=coef[b*P+c]*xrow[c]; }
                acc2 += ys[b][t*N+i]-pred;
            }
        }
    }
    let t3=t2.elapsed().as_secs_f64();
    println!("baseline {:.3}s batched {:.3}s speedup {:.2}x acc {:.6} {:.6}",t1,t3,t1/t3,acc,acc2);
}
