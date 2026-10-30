#![allow(dead_code)]
use std::{hint::black_box, time::Instant};
// YUPEI_SELECTOR_SOURCE_ROOT selects a repository/sandbox root for post-migration checks.
include!(concat!(env!("OUT_DIR"), "/source_modules.rs"));
mod simd { pub use super::correct::*; }
fn verbose() -> bool { false }

fn rng(s: &mut u64) -> u32 { *s ^= *s << 13; *s ^= *s >> 7; *s ^= *s << 17; (*s >> 32) as u32 }
fn main() {
    let mut state = 73129387u64;
    let n = 5195;
    let count = 4096;
    let data: Vec<Vec<f32>> = (0..count).map(|j| (0..n).map(|i| {
        let bits = rng(&mut state);
        let u = (bits >> 8) as f32 / 16777216.0;
        match j % 16 {
            0 => u * 2.0 - 1.0,
            1 => u * u * u * 0.3,
            2 => 0.0,
            3 => ((i as f32)*0.013 + j as f32).sin()*0.1 + u*0.001,
            4 => (u*30.0).round()*0.0001 + ((rng(&mut state)>>8) as f32/16777216.0-0.5)*0.00009,
            5 => if u<0.95 {f32::NEG_INFINITY} else {u},
            6 => -u*0.002,
            7 => (u-0.5)*0.0001,
            8 => f32::from_bits(bits),
            9 => f32::from_bits([0x7fc00001,0xffc12345,0,0x80000000,0x7f800000,0xff800000][i%6]),
            10 => if i==0 {0.50001} else if (1000..1050).contains(&i) {0.50004} else {0.0},
            11 => f32::from_bits(0x3851b717u32.wrapping_add((bits%11).wrapping_sub(5))),
            12 => (u-0.5)*1e35,
            13 => (u-0.5)*32.0,
            14 => f32::from_bits(if bits&1==0 {1} else {0x80000001}),
            _ => -(u*30.0).round()*0.0001 + (u-0.5)*0.000099,
        }
    }).collect()).collect();
    let mut keys=vec![0u64;n]; let mut scratch=keys.clone(); let mut reference=keys.clone();
    let mut legacy_keys=keys.clone(); let mut legacy_different=0usize;
    let mut prefix_checks=0;
    for (j,v) in data.iter().enumerate() {
        for (i,&s) in v.iter().enumerate() {reference[i]=correct::pack_desc_idx_asc(s,i as u32);}
        reference.sort_unstable();
        let mut prior=Vec::new();
        for k in [1,5,20,30,100,500] {
            correct::select_topk_keys(v,k,&mut keys,&mut scratch);
            assert_eq!(&keys[..k],&reference[..k],"vector={j}, k={k}");
            assert_eq!(&keys[..prior.len()],prior.as_slice());
            prior=keys[..k].to_vec(); prefix_checks+=1;
            if k==20 {legacy::select_topk_keys(v,k,&mut legacy_keys,&mut scratch); legacy_different+=usize::from(keys[..k]!=legacy_keys[..k]);}
        }
    }
    for len in [1,7,31,128,1000,1024,5195] {
        for v in data.iter().take(32) {
            for (i,&s) in v[..len].iter().enumerate() {reference[i]=correct::pack_desc_idx_asc(s,i as u32);}
            reference[..len].sort_unstable();
            for k in [0,1,5,20,len].map(|k|k.min(len)) {
                correct::select_topk_keys(&v[..len],k,&mut keys[..len],&mut scratch[..len]);
                assert_eq!(&keys[..k],&reference[..k],"len={len} k={k}");
            }
        }
    }
    println!("{{\"kind\":\"correctness\",\"vectors\":{count},\"length\":{n},\"ks\":[1,5,20,30,100,500],\"reference_mismatches\":0,\"prefix_checks\":{prefix_checks},\"legacy_different_k20\":{legacy_different}}}");
    // Benchmark ordinary score distributions separately from intentionally pathological inputs.
    for pathological in [false,true] {
        let fixture:Vec<_>=data.iter().enumerate().filter(|(j,_)|(*j%16>=8)==pathological).map(|(_,v)|v).collect();
        for mode in [0,1,2,2,1,0] {
            let start=Instant::now();let mut digest=0u64;
            for _ in 0..12 {for v in &fixture {
                match mode {
                    0=>legacy::select_topk_keys(black_box(v),20,&mut keys,&mut scratch),
                    1=>correct::select_topk_keys(black_box(v),20,&mut keys,&mut scratch),
                    _=>{for (i,&s) in v.iter().enumerate(){keys[i]=correct::pack_desc_idx_asc(s,i as u32);}keys.select_nth_unstable(19);keys[..20].sort_unstable();}
                }
                digest=digest.wrapping_add(black_box(keys[0]));
            }}
            println!("{{\"kind\":\"timing\",\"pathological\":{pathological},\"mode\":{mode},\"calls\":{},\"seconds\":{},\"digest\":{digest}}}",fixture.len()*12,start.elapsed().as_secs_f64());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn assert_bits(a:&[f32],b:&[f32],label:&str) {
        assert_eq!(a.len(),b.len());
        for (i,(a,b)) in a.iter().zip(b).enumerate() {assert_eq!(a.to_bits(),b.to_bits(),"{label}[{i}]");}
    }
    fn assert_non_topk(a:&net::NetMats,b:&net::NetMats) {
        macro_rules! field {($($name:ident),*)=>{$(assert_bits(&a.$name,&b.$name,stringify!($name));)*};}
        field!(rho,g,ru,sg,x0,gs,rus,sgs,x0s,z,zs,grel,gsh,gcov,gnpos,ghhi,pb,qc,pbs,qcs,exz,exzs,mass);
        for (x,y) in a.e.iter().zip(&b.e) {assert_eq!(x.to_bits(),y.to_bits());}
        assert_eq!(a.npair,b.npair);
        for (x,y) in a.strata.iter().chain(&a.strata_s).zip(b.strata.iter().chain(&b.strata_s)) {
            assert_eq!((x.0.to_bits(),x.1.to_bits(),x.2),(y.0.to_bits(),y.1.to_bits(),y.2));
        }
    }
    #[test]
    fn common_topk_and_all_net_fields_do_not_depend_on_requested_ks() {
        let _=rayon::ThreadPoolBuilder::new().num_threads(2).build_global();
        for dense in [false,true] {
            let n=64;
            let features:Vec<Vec<f32>>=(0..n).map(|i|(0..8).map(|t|if dense {1.0+((i*17+t*13) as f32).sin()*0.0002} else {((i*17+t*13) as f32).sin()+0.8}).collect()).collect();
            let mut mat=vec![0.0;n*n];
            for i in 0..n {for j in 0..n {mat[i*n+j]=features[i].iter().zip(&features[j]).map(|(a,b)|a*b).sum();}}
            let mask:Vec<_>=(0..n).map(|i|i%11!=0).collect();
            let stocks:Vec<_>=(0..n).filter(|&i|mask[i]).collect();
            let dec:Vec<_>=(0..n).map(|i|i%10).collect();
            let make=|ks:Option<&[usize]>|net::net_one_mat(&mat,n,&dec,&mask,&stocks,2000,4,"probe",ks,None);
            let base=make(Some(&[5,20]));
            let (names,values)=topk::finalize_total(base.topk_total.as_ref().unwrap(),n,&mask,&[5,20],"");
            for ks in [&[5,20,30][..],&[5,20,100][..],&[20,5][..]] {
                let other=make(Some(ks)); assert_non_topk(&base,&other);
                let (other_names,other_values)=topk::finalize_total(other.topk_total.as_ref().unwrap(),n,&mask,ks,"");
                for (name,value) in names.iter().zip(&values) {let j=other_names.iter().position(|x|x==name).unwrap();assert_bits(value,&other_values[j],name);}
            }
            assert_non_topk(&base,&make(None));
            let (standalone_names,standalone)=topk::topk_one_mat(&base.rho,&base.e,n,&mask,&[5,20],"");
            assert_eq!(names,standalone_names);
            for (name,(a,b)) in names.iter().zip(values.iter().zip(&standalone)) {assert_bits(a,b,name);}
        }
    }
    #[test]
    fn absorb_mass_applies_relu_per_selected_stock() {
        let h=[0.0,-0.0,1e-6,-1e-6,2e-6,f32::NAN];
        let gap=[-0.1,0.0,0.1,-0.2,0.2,0.0];
        let order=[0,1,2,3,4,5];
        let pref=[0.0,0.0,1e-6f32 as f64,1e-6f32 as f64,3e-6f32 as f64,3e-6f32 as f64];
        let mut acc=topk::Acc::new(6,2);
        topk::absorb_topk(&mut acc,6,0,2,&[2,6],&order,&h,&gap,&pref);
        for i in 0..6 {
            let c=&acc.cells[6+i];assert_eq!(c.cnt,u32::from(h[i].is_finite()));
            let expected=if h[i]>0.0 {h[i] as f64/pref[5]} else {0.0};
            assert_eq!(c.sum_mass.to_bits(),expected.to_bits());
            assert_eq!(c.sum_gap.to_bits(),(if h[i].is_finite(){gap[i] as f64}else{0.0f64}).to_bits());
        }
    }
    #[test]
    fn nonfinite_candidates_never_win_or_count_and_empty_means_undefined() {
        let _=rayon::ThreadPoolBuilder::new().num_threads(2).build_global();
        assert_eq!(correct::argmax_packed_two(&[f32::NEG_INFINITY,f32::NAN],&[0.2,0.3]),(usize::MAX,usize::MAX));
        assert_eq!(correct::argmax_packed_two(&[0.5,0.4,f32::NAN],&[f32::INFINITY,0.1,0.2]),(1,1));
        let n=16;
        let mask=vec![true;n];let stock:Vec<_>=(0..n).collect();let dec=vec![0;n];
        let mut mat=vec![0.0;n*n];for i in 0..n {mat[i*n+i]=0.5;}
        let energy=vec![1.0;n];let bc=vec![0.0;n*n];
        for empty in [false,true] {
            let p:Vec<_>=(0..n*n).map(|i|if empty || i%n%3!=0 {f32::NAN}else{0.2}).collect();
            let q:Vec<_>=(0..n*n).map(|i|if empty || i%n%3!=0 {f32::INFINITY}else{0.3}).collect();
            let split=net::SplitRho{b_row:&p,c_row:&q,bc:&bc};
            let make=|ks:Option<&[usize]>|net::net_one_mat(&mat,n,&dec,&mask,&stock,2000,4,"nonfinite",ks,Some((&split,&energy,&energy)));
            let base=make(Some(&[5,20]));assert_non_topk(&base,&make(None));assert_non_topk(&base,&make(Some(&[5,20,30])));
            let acc=base.topk_total.as_ref().unwrap();
            if empty {
                assert_eq!(base.npair,0);assert_eq!(acc.pairs,0);
                assert!(base.x0.iter().all(|&x|x==-1.0));assert!(base.mass.iter().all(|x|x.is_nan()));
                assert!(acc.cells.iter().all(|x|x.cnt==0));
                let (_,cols)=topk::finalize_total(acc,n,&mask,&[5,20],"");
                for j in [1,2,3,5,6,7] {assert!(cols[j].iter().all(|x|x.is_nan()));}
            } else {
                assert!(base.x0.iter().filter(|&&x|x>=0.0).all(|&x|(x as usize)%3==0));
                for (i,c) in acc.cells.iter().enumerate() {if (i%n)%3!=0 {assert_eq!(c.cnt,0);}}
            }
        }
        // Standalone TopK uses the same finite-candidate policy, including all-empty input.
        for rho in [vec![f32::NAN;n*n],vec![f32::INFINITY;n*n]] {
            let (_,cols)=topk::topk_one_mat(&rho,&energy,n,&mask,&[5,20],"");
            assert!(cols[0].iter().all(|&x|x==0.0));assert!(cols[3].iter().all(|x|x.is_nan()));
        }
    }
}
