"""Opt-in compact predecessor AVX512; apply after binary_hoist_patch."""
from pathlib import Path

p = Path(__file__).parent / 'src/relaynet_fast.rs'
s = p.read_text()
if 'fn binary_parent_backprop_avx512(' in s:
    raise SystemExit('binary compact predecessor SIMD already applied')
start = s.index('// BEGIN EXACT BINARY DIRECTION EXPERIMENT')
end = s.index('// END EXACT BINARY DIRECTION EXPERIMENT', start)
f = s[start:end]
old = '''    let hoist=std::env::var_os("RELAYNET_BINARY_HOIST").is_some();'''
assert old in f
f = f.replace(old, old + '''
    #[cfg(target_arch="x86_64")]
    let parent_simd=record_predecessors && graph.sorted && n<=i32::MAX as usize
        && std::env::var_os("RELAYNET_BINARY_PARENT_SIMD").is_some()
        && std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("avx2");''', 1)
old = '''                if record_predecessors {
                    // For a fixed u'''
assert old in f
f = f.replace(old, '''                if record_predecessors {
                    #[cfg(target_arch="x86_64")]
                    if parent_simd {
                        unsafe {binary_parent_backprop_avx512(scratch.pred_head[u],
                            &scratch.pred_from,&scratch.pred_next,&scratch.sigma,&mut scratch.delta,u);}
                        if u!=src {bet[u]+=scratch.delta[u];}
                        continue;
                    }
                    // For a fixed u''', 1)
s = s[:start] + f + s[end:]
s += r'''

// Each linked-list entry is a distinct actual predecessor. SIMD is enabled only
// for strictly sorted adjacency (which also proves no duplicate neighbor IDs).
// Packing eight IDs changes no floating operation or per-target update order.
#[cfg(target_arch="x86_64")]
#[target_feature(enable="avx512f,avx2")]
unsafe fn binary_parent_backprop_avx512(mut link:u32,from:&[u32],next:&[u32],sigma:&[f64],delta:&mut[f64],u:usize){
    use std::arch::x86_64::*;
    let sigma_u=sigma[u];let factor=1.0+delta[u];
    let su=_mm512_set1_pd(sigma_u);let df=_mm512_set1_pd(factor);
    let mut buf=[0u32;8];
    while link!=u32::MAX {
        let mut count=0;
        while count<8 && link!=u32::MAX {
            buf[count]=from[link as usize];link=next[link as usize];count+=1;
        }
        if count==8 {
            let ids=_mm256_loadu_si256(buf.as_ptr() as *const __m256i);
            let sv=_mm512_i32gather_pd::<8>(ids,sigma.as_ptr());
            let previous=_mm512_i32gather_pd::<8>(ids,delta.as_ptr());
            let ratios=_mm512_div_pd(sv,su);
            let terms=_mm512_mul_pd(ratios,df);
            let updated=_mm512_add_pd(previous,terms);
            _mm512_i32scatter_pd::<8>(delta.as_mut_ptr(),ids,updated);
        } else {
            for &v in &buf[..count] {
                let v=v as usize;delta[v]+=(sigma[v]/sigma_u)*factor;
            }
        }
    }
}
'''
p.write_text(s)
print('Applied opt-in RELAYNET_BINARY_PARENT_SIMD; hardware and unique-ID guards enabled')
