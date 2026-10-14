p='/home/chenzongwei/rust_pyfunc/src/yupei_dist/matrix_stage.rs'
s=open(p).read(); done=[]
def rep(old,new,tag):
    global s
    if new in s: done.append('SKIP '+tag); return
    assert old in s, 'MISS '+tag
    s=s.replace(old,new,1); done.append('OK '+tag)
rep("""fn phase1_simd(
    need: u8,""", """fn phase1_simd<const NEED: u8>(""", 'p1 const')
rep("""fn phase2_simd(
    need: u8,""", """fn phase2_simd<const NEED: u8>(""", 'p2 const')
rep("""fn phase1_scalar(
    _need: u8,""", """fn phase1_scalar<const _NEED: u8>(""", 'p1s const')
rep("""fn phase2_scalar(
    _need: u8,""", """fn phase2_scalar<const _NEED: u8>(""", 'p2s const')
n1 = s.count('if need & FAM_')
s = s.replace('if need & FAM_', 'if NEED & FAM_')
done.append('guards %d' % n1)
rep("""            #[cfg(target_arch = "x86_64")]
            phase1_simd(need, gp, acc_mem, block_buf_ref, bcnt, boff, k0 as u32);
            #[cfg(not(target_arch = "x86_64"))]
            phase1_scalar(need, gp, acc_mem, block_buf_ref, bcnt, boff, k0 as u32);""",
"""            #[cfg(target_arch = "x86_64")]
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
            }""", 'p1 dispatch')
rep("""            #[cfg(target_arch = "x86_64")]
            phase2_simd(need, pack, gp, block_buf_ref, counts_ref, offs_ref, n, k0 as u32);
            #[cfg(not(target_arch = "x86_64"))]
            phase2_scalar(need, pack, gp, block_buf_ref, counts_ref, offs_ref, n, k0 as u32);""",
"""            #[cfg(target_arch = "x86_64")]
            if need == FAM_CNT {
                phase2_simd::<{ FAM_CNT }>(pack, gp, block_buf_ref, counts_ref, offs_ref, n, k0 as u32);
            } else if need == FAM_VOL {
                phase2_simd::<{ FAM_VOL }>(pack, gp, block_buf_ref, counts_ref, offs_ref, n, k0 as u32);
            } else if need == (FAM_CNT | FAM_VOL) {
                phase2_simd::<{ FAM_CNT | FAM_VOL }>(pack, gp, block_buf_ref, counts_ref, offs_ref, n, k0 as u32);
            } else {
                phase2_simd::<{ FAM_ALL }>(pack, gp, block_buf_ref, counts_ref, offs_ref, n, k0 as u32);
            }
            #[cfg(not(target_arch = "x86_64"))]
            if need == FAM_CNT {
                phase2_scalar::<{ FAM_CNT }>(pack, gp, block_buf_ref, counts_ref, offs_ref, n, k0 as u32);
            } else if need == FAM_VOL {
                phase2_scalar::<{ FAM_VOL }>(pack, gp, block_buf_ref, counts_ref, offs_ref, n, k0 as u32);
            } else if need == (FAM_CNT | FAM_VOL) {
                phase2_scalar::<{ FAM_CNT | FAM_VOL }>(pack, gp, block_buf_ref, counts_ref, offs_ref, n, k0 as u32);
            } else {
                phase2_scalar::<{ FAM_ALL }>(pack, gp, block_buf_ref, counts_ref, offs_ref, n, k0 as u32);
            }""", 'p2 dispatch')
open(p,'w').write(s)
print('\n'.join(done))
