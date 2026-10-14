p='/home/chenzongwei/rust_pyfunc/src/yupei_dist/matrix_stage.rs'
s=open(p).read()
old = """            #[cfg(target_arch = "x86_64")]
            {
                let g = gp;
                let am = acc_mem;
                let bb = block_buf_ref;
                if need == FAM_CNT {
                    phase1_simd::<{ FAM_CNT }>(g, am, bb, bcnt, boff, k0 as u32);
                } else if need == FAM_VOL {
                    phase1_simd::<{ FAM_VOL }>(g, am, bb, bcnt, boff, k0 as u32);
                } else if need == (FAM_CNT | FAM_VOL) {
                    phase1_simd::<{ FAM_CNT | FAM_VOL }>(g, am, bb, bcnt, boff, k0 as u32);
                } else {
                    phase1_simd::<{ FAM_ALL }>(g, am, bb, bcnt, boff, k0 as u32);
                }
            }
            #[cfg(not(target_arch = "x86_64"))]
            {
                let g = gp;
                let am = acc_mem;
                let bb = block_buf_ref;
                if need == FAM_CNT {
                    phase1_scalar::<{ FAM_CNT }>(g, am, bb, bcnt, boff, k0 as u32);
                } else if need == FAM_VOL {
                    phase1_scalar::<{ FAM_VOL }>(g, am, bb, bcnt, boff, k0 as u32);
                } else if need == (FAM_CNT | FAM_VOL) {
                    phase1_scalar::<{ FAM_CNT | FAM_VOL }>(g, am, bb, bcnt, boff, k0 as u32);
                } else {
                    phase1_scalar::<{ FAM_ALL }>(g, am, bb, bcnt, boff, k0 as u32);
                }
            }"""
new = """            #[cfg(target_arch = "x86_64")]
            if need == FAM_CNT {
                phase1_simd::<{ FAM_CNT }>(&mut *gp, &mut *acc_mem, block_buf_ref, bcnt, boff, k0 as u32);
            } else if need == FAM_VOL {
                phase1_simd::<{ FAM_VOL }>(&mut *gp, &mut *acc_mem, block_buf_ref, bcnt, boff, k0 as u32);
            } else if need == (FAM_CNT | FAM_VOL) {
                phase1_simd::<{ FAM_CNT | FAM_VOL }>(&mut *gp, &mut *acc_mem, block_buf_ref, bcnt, boff, k0 as u32);
            } else {
                phase1_simd::<{ FAM_ALL }>(&mut *gp, &mut *acc_mem, block_buf_ref, bcnt, boff, k0 as u32);
            }
            #[cfg(not(target_arch = "x86_64"))]
            if need == FAM_CNT {
                phase1_scalar::<{ FAM_CNT }>(&mut *gp, &mut *acc_mem, block_buf_ref, bcnt, boff, k0 as u32);
            } else if need == FAM_VOL {
                phase1_scalar::<{ FAM_VOL }>(&mut *gp, &mut *acc_mem, block_buf_ref, bcnt, boff, k0 as u32);
            } else if need == (FAM_CNT | FAM_VOL) {
                phase1_scalar::<{ FAM_CNT | FAM_VOL }>(&mut *gp, &mut *acc_mem, block_buf_ref, bcnt, boff, k0 as u32);
            } else {
                phase1_scalar::<{ FAM_ALL }>(&mut *gp, &mut *acc_mem, block_buf_ref, bcnt, boff, k0 as u32);
            }"""
assert old in s, 'MISS'
s = s.replace(old, new, 1)
open(p,'w').write(s)
print('OK')
