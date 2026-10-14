p='/home/chenzongwei/rust_pyfunc/src/yupei_dist/matrix_stage.rs'
s=open(p).read(); done=[]
def rep(old,new,tag):
    global s
    if new in s: done.append('SKIP '+tag); return
    assert old in s, 'MISS '+tag
    s=s.replace(old,new,1); done.append('OK '+tag)

rep("""fn phase2_simd(
    pack: &mut [f32],""","""fn phase2_simd(
    need: u8,
    pack: &mut [f32],""", 'p2 sig')
rep("""                    let ua = &cell.1;
                    // 8 个权重标量广播（每单元一次）
                    let uc = _mm256_set1_ps(ua[U_CNT]);
                    let uv = _mm256_set1_ps(ua[U_VOL]);
                    let ul = _mm256_set1_ps(ua[U_LOGVOL]);
                    let fp = _mm256_set1_ps(ua[U_FLOW_P]);
                    let fm = _mm256_set1_ps(ua[U_FLOW_M]);
                    let up = _mm256_set1_ps(ua[U_URG_P]);
                    let um = _mm256_set1_ps(ua[U_URG_M]);
                    let xp = _mm256_set1_ps(ua[U_EXT_P]);
                    let xm = _mm256_set1_ps(ua[U_EXT_M]);
                    let gpk = gpk0.add(kl * GP_STRIDE);
                    // 7 个 ymm 加载（56 槽, 块对齐; 未用槽恒 0 → 乘积恒 0）
                    let g0 = _mm256_loadu_ps(gpk); // cnt 8τ
                    let g1 = _mm256_loadu_ps(gpk.add(8)); // cnt_t30 + 0×7
                    let g2 = _mm256_loadu_ps(gpk.add(16)); // vol 6τ + 0×2
                    let g3 = _mm256_loadu_ps(gpk.add(24)); // logvol 4τ + 0×4
                    let g4 = _mm256_loadu_ps(gpk.add(32)); // flow ±
                    let g5 = _mm256_loadu_ps(gpk.add(40)); // urg ±
                    let g6 = _mm256_loadu_ps(gpk.add(48)); // ext ±
                    p0 = _mm256_fmadd_ps(uc, g0, p0);
                    p1 = _mm256_fmadd_ps(uc, g1, p1);
                    p2 = _mm256_fmadd_ps(uv, g2, p2);
                    p3 = _mm256_fmadd_ps(ul, g3, p3);
                    // signed 块: same = A+×B+ + A−×B−; opp = A+×B− + A−×B+（通道交换）
                    let g4x = _mm256_permutevar8x32_ps(g4, idx_x);
                    p4 = _mm256_fmadd_ps(fp, g4, _mm256_fmadd_ps(fm, g4x, p4));
                    let g5x = _mm256_permutevar8x32_ps(g5, idx_x);
                    p5 = _mm256_fmadd_ps(up, g5, _mm256_fmadd_ps(um, g5x, p5));
                    let g6x = _mm256_permutevar8x32_ps(g6, idx_x);
                    p6 = _mm256_fmadd_ps(xp, g6, _mm256_fmadd_ps(xm, g6x, p6));""",
"""                    let ua = &cell.1;
                    let gpk = gpk0.add(kl * GP_STRIDE);
                    // 只算请求的家族：被跳过的家族不读 gp、不做 FMA。
                    if need & FAM_CNT != 0 {
                        let uc = _mm256_set1_ps(ua[U_CNT]);
                        let g0 = _mm256_loadu_ps(gpk);
                        let g1 = _mm256_loadu_ps(gpk.add(8));
                        p0 = _mm256_fmadd_ps(uc, g0, p0);
                        p1 = _mm256_fmadd_ps(uc, g1, p1);
                    }
                    if need & FAM_VOL != 0 {
                        let uv = _mm256_set1_ps(ua[U_VOL]);
                        let g2 = _mm256_loadu_ps(gpk.add(16));
                        p2 = _mm256_fmadd_ps(uv, g2, p2);
                    }
                    if need & FAM_LOGVOL != 0 {
                        let ul = _mm256_set1_ps(ua[U_LOGVOL]);
                        let g3 = _mm256_loadu_ps(gpk.add(24));
                        p3 = _mm256_fmadd_ps(ul, g3, p3);
                    }
                    if need & FAM_FLOW != 0 {
                        let fp = _mm256_set1_ps(ua[U_FLOW_P]);
                        let fm = _mm256_set1_ps(ua[U_FLOW_M]);
                        let g4 = _mm256_loadu_ps(gpk.add(32));
                        let g4x = _mm256_permutevar8x32_ps(g4, idx_x);
                        p4 = _mm256_fmadd_ps(fp, g4, _mm256_fmadd_ps(fm, g4x, p4));
                    }
                    if need & FAM_URG != 0 {
                        let up = _mm256_set1_ps(ua[U_URG_P]);
                        let um = _mm256_set1_ps(ua[U_URG_M]);
                        let g5 = _mm256_loadu_ps(gpk.add(40));
                        let g5x = _mm256_permutevar8x32_ps(g5, idx_x);
                        p5 = _mm256_fmadd_ps(up, g5, _mm256_fmadd_ps(um, g5x, p5));
                    }
                    if need & FAM_EXT != 0 {
                        let xp = _mm256_set1_ps(ua[U_EXT_P]);
                        let xm = _mm256_set1_ps(ua[U_EXT_M]);
                        let g6 = _mm256_loadu_ps(gpk.add(48));
                        let g6x = _mm256_permutevar8x32_ps(g6, idx_x);
                        p6 = _mm256_fmadd_ps(xp, g6, _mm256_fmadd_ps(xm, g6x, p6));
                    }""", 'p2 mask')
rep("fn phase1_scalar(","fn phase1_scalar(\n    _need: u8,", 'p1s sig')
rep("fn phase2_scalar(","fn phase2_scalar(\n    _need: u8,", 'p2s sig')
rep("pub fn compute_matrices(stocks: &[StockPrep]) -> (Vec<Vec<f32>>, Vec<StockStats>) {",
    "pub fn compute_matrices(stocks: &[StockPrep], need: u8) -> (Vec<Vec<f32>>, Vec<StockStats>) {", 'cm sig')
rep("""            #[cfg(target_arch = "x86_64")]
            phase1_simd(gp, acc_mem, block_buf_ref, bcnt, boff, k0 as u32);
            #[cfg(not(target_arch = "x86_64"))]
            phase1_scalar(gp, acc_mem, block_buf_ref, bcnt, boff, k0 as u32);""",
"""            #[cfg(target_arch = "x86_64")]
            phase1_simd(need, gp, acc_mem, block_buf_ref, bcnt, boff, k0 as u32);
            #[cfg(not(target_arch = "x86_64"))]
            phase1_scalar(need, gp, acc_mem, block_buf_ref, bcnt, boff, k0 as u32);""", 'p1 call')
rep("""            #[cfg(target_arch = "x86_64")]
            phase2_simd(pack, gp, block_buf_ref, counts_ref, offs_ref, n, k0 as u32);
            #[cfg(not(target_arch = "x86_64"))]
            phase2_scalar(pack, gp, block_buf_ref, counts_ref, offs_ref, n, k0 as u32);""",
"""            #[cfg(target_arch = "x86_64")]
            phase2_simd(need, pack, gp, block_buf_ref, counts_ref, offs_ref, n, k0 as u32);
            #[cfg(not(target_arch = "x86_64"))]
            phase2_scalar(need, pack, gp, block_buf_ref, counts_ref, offs_ref, n, k0 as u32);""", 'p2 call')
rep("""        for m in 0..nm {
            let ch = field_channel(m);
            let ptr = mats_raw[m].0;""",
"""        for m in 0..nm {
            // 不被请求的家族不写回：省掉 22 张（约 2.3 GB）的写流量
            if need & (1u8 << MATRIX_SPECS[m].family) == 0 {
                continue;
            }
            let ch = field_channel(m);
            let ptr = mats_raw[m].0;""", 'flush skip')
open(p,'w').write(s)
print('\n'.join(done))
