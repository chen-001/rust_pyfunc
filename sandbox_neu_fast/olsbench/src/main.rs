use nalgebra::{Cholesky, DMatrix};
use std::time::Instant;

const T: usize = 2276;
const N: usize = 5438;
const NV: usize = 5000;
const K: usize = 10;
const B: usize = 4;

fn fill_rand(v: &mut [f64], seed: u64) {
    let mut s = seed;
    for x in v.iter_mut() {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        *x = ((s >> 11) as f64) / ((1u64 << 53) as f64) - 0.5;
    }
}

fn main() {
    // valid idx: random permutation-like indices; ensure sorted ascending like production
    let mut valid_idx: Vec<u32> = (0..NV as u32).collect();
    let mut s = 12345u64;
    for i in (1..NV).rev() {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let j = (s as usize) % (i + 1);
        valid_idx.swap(i, j);
    }
    valid_idx.sort_unstable();
    let xdays: Vec<f64> = {
        let mut v = vec![0.0f64; T * NV * K];
        fill_rand(&mut v, 999);
        v
    };
    let y_slots: Vec<Vec<f64>> = (0..B).map(|b| {
        let mut v = vec![0.0f64; T * N];
        fill_rand(&mut v, 1000 + b as u64);
        v
    }).collect();
    // precompute per-date Cholesky L (p=20) from synthetic xtx
    let p = 20usize;
    let mut chol_l: Vec<f64> = vec![0.0; T * p * p];
    {
        let mut xtx = vec![0.0f64; p * p];
        for i in 0..p {
            for j in 0..p {
                xtx[i*p+j] = if i==j { 10.0 + (i as f64) } else { 0.1/(1+(i as f64)+(j as f64)) };
            }
        }
        let m = DMatrix::from_row_slice(p,p,&xtx);
        let chol = Cholesky::new(m).unwrap();
        let l = chol.l();
        for i in 0..p { for j in 0..=i { chol_l[i*p+j]=l[(i,j)]; } }
    }
    let mut out = vec![0f32; T * N];

    // variant 1: production-like nalgebra solve + allocations
    let t0 = Instant::now();
    let mut y_buf = Vec::with_capacity(NV);
    let mut xty = vec![0f64; p];
    for t in 0..T {
        let yb = &y_slots[0][t*N..(t+1)*N];
        y_buf.clear();
        for &j in &valid_idx { y_buf.push(yb[j as usize]); }
        xty.fill(0.0);
        let xb = &xdays[t*NV*K..(t+1)*NV*K];
        for i in 0..NV {
            let xrow=&xb[i*K..i*K+K];
            let yv=y_buf[i];
            for c in 0..K { xty[c]+=xrow[c]*yv; }
            xty[K]+=yv;
        }
        // build xtx (synthetic)
        let mut xtx = vec![0.0f64; p*p];
        for i in 0..p { for j in 0..p { xtx[i*p+j] = if i==j { 10.0 + (i as f64) } else { 0.1/(1+(i as f64)+(j as f64)) }; } }
        let m = DMatrix::from_row_slice(p,p,&xtx);
        let rhs = DMatrix::from_column_slice(p,1,&xty);
        let chol = Cholesky::new(m).unwrap();
        let coef: Vec<f64> = chol.solve(&rhs).column(0).iter().copied().collect();
        for (pos,&j) in valid_idx.iter().enumerate() {
            let xrow=&xb[pos*K..pos*K+K];
            let mut pred=coef[K];
            for c in 0..K { pred+=coef[c]*xrow[c]; }
            out[t*N+j as usize] = (y_buf[pos]-pred) as f32;
        }
    }
    let t1 = t0.elapsed().as_secs_f64();

    // variant 2: custom flat solve, no allocations
    let t0 = Instant::now();
    let mut xty2 = vec![0f64; p];
    let mut coef2 = vec![0f64; p];
    let mut z = vec![0f64; p];
    for t in 0..T {
        let yb = &y_slots[0][t*N..(t+1)*N];
        y_buf.clear();
        for &j in &valid_idx { y_buf.push(yb[j as usize]); }
        xty2.fill(0.0);
        let xb = &xdays[t*NV*K..(t+1)*NV*K];
        for i in 0..NV {
            let xrow=&xb[i*K..i*K+K];
            let yv=y_buf[i];
            for c in 0..K { xty2[c]+=xrow[c]*yv; }
            xty2[K]+=yv;
        }
        let l=&chol_l[t*p*p..(t+1)*p*p];
        // forward substitution L z = xty
        for i in 0..p {
            let mut acc=xty2[i];
            for j in 0..i { acc-=l[i*p+j]*z[j]; }
            z[i]=acc/l[i*p+i];
        }
        // back substitution L^T coef = z
        for i in (0..p).rev() {
            let mut acc=z[i];
            for j in (i+1)..p { acc-=l[j*p+i]*coef2[j]; }
            coef2[i]=acc/l[i*p+i];
        }
        for (pos,&j) in valid_idx.iter().enumerate() {
            let xrow=&xb[pos*K..pos*K+K];
            let mut pred=coef2[K];
            for c in 0..K { pred+=coef2[c]*xrow[c]; }
            out[t*N+j as usize] = (y_buf[pos]-pred) as f32;
        }
    }
    let t2 = t0.elapsed().as_secs_f64();
    println!("variant1 (nalgebra) {:.3}s variant2 (flat custom) {:.3}s speedup {:.2}x out0={}", t1,t2,t1/t2,out[0]);
}
