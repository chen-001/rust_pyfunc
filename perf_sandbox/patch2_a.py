import re
p='/home/chenzongwei/rust_pyfunc/src/yupei_dist/matrix_stage.rs'
s=open(p).read(); done=[]
def rep(old,new,tag):
    global s
    if new in s: done.append('SKIP '+tag); return
    assert old in s, 'MISS '+tag
    s=s.replace(old,new,1); done.append('OK '+tag)

rep("pub const N_MATRICES: usize = MATRIX_SPECS.len();",
"""pub const N_MATRICES: usize = MATRIX_SPECS.len();

/// 权重家族位（MATRIX_SPECS.family 的顺序：0=cnt 1=vol 2=logvol 3=flow 4=urg 5=ext）
pub const FAM_CNT: u8 = 1 << 0;
pub const FAM_VOL: u8 = 1 << 1;
pub const FAM_LOGVOL: u8 = 1 << 2;
pub const FAM_FLOW: u8 = 1 << 3;
pub const FAM_URG: u8 = 1 << 4;
pub const FAM_EXT: u8 = 1 << 5;
pub const FAM_ALL: u8 = FAM_CNT | FAM_VOL | FAM_LOGVOL | FAM_FLOW | FAM_URG | FAM_EXT;

/// 请求的矩阵名 → 需要计算的家族位。
pub fn family_mask_of(names: &[String]) -> u8 {
    let mut m = 0u8;
    for n in names {
        if let Some(sp) = MATRIX_SPECS.iter().find(|sp| sp.name == n.as_str()) {
            m |= 1 << sp.family;
        }
    }
    m
}""", 'family bits')

# phase1_simd：运行时 need
rep("""fn phase1_simd(
    gp: &mut [f32],""","""fn phase1_simd(
    need: u8,
    gp: &mut [f32],""", 'p1 sig')
rep("""            let gpk = gp.as_mut_ptr().add(k * GP_STRIDE);
            let uc = _mm256_set1_ps(ub[U_CNT]);
            let uv = _mm256_blend_ps(_mm256_set1_ps(ub[U_VOL]), zero, 0b11000000);
            let ul = _mm256_blend_ps(_mm256_set1_ps(ub[U_LOGVOL]), zero, 0b11110000);
            let u4 = _mm256_unpacklo_ps(
                _mm256_set1_ps(ub[U_FLOW_P]),
                _mm256_set1_ps(ub[U_FLOW_M]),
            );
            let u5 = _mm256_blend_ps(
                _mm256_unpacklo_ps(
                    _mm256_set1_ps(ub[U_URG_P]),
                    _mm256_set1_ps(ub[U_URG_M]),
                ),
                zero,
                0b11000000,
            );
            let u6 = _mm256_blend_ps(
                _mm256_unpacklo_ps(
                    _mm256_set1_ps(ub[U_EXT_P]),
                    _mm256_set1_ps(ub[U_EXT_M]),
                ),
                zero,
                0b11110000,
            );
            // cnt 块0（槽 0..8）: gpk = acc + 0.5u; acc = (acc+u)·dec
            let g0 = _mm256_add_ps(a0, _mm256_mul_ps(half, uc));
            _mm256_storeu_ps(gpk, g0);
            a0 = _mm256_mul_ps(_mm256_add_ps(a0, uc), d0);
            // cnt_t30（槽 8, 标量）
            let ucs = ub[U_CNT];
            *gpk.add(8) = a1s + 0.5 * ucs;
            a1s = (a1s + ucs) * d1s;
            // vol（槽 16..24）
            let g2 = _mm256_add_ps(a2, _mm256_mul_ps(half, uv));
            _mm256_storeu_ps(gpk.add(16), g2);
            a2 = _mm256_mul_ps(_mm256_add_ps(a2, uv), d2);
            // logvol（槽 24..32）
            let g3 = _mm256_add_ps(a3, _mm256_mul_ps(half, ul));
            _mm256_storeu_ps(gpk.add(24), g3);
            a3 = _mm256_mul_ps(_mm256_add_ps(a3, ul), d3);
            // flow ±（槽 32..40, unpacklo 交错 P+/P−）
            let g4 = _mm256_add_ps(a4, _mm256_mul_ps(half, u4));
            _mm256_storeu_ps(gpk.add(32), g4);
            a4 = _mm256_mul_ps(_mm256_add_ps(a4, u4), d4);
            // urg ±（槽 40..48）
            let g5 = _mm256_add_ps(a5, _mm256_mul_ps(half, u5));
            _mm256_storeu_ps(gpk.add(40), g5);
            a5 = _mm256_mul_ps(_mm256_add_ps(a5, u5), d5);
            // ext ±（槽 48..56）
            let g6 = _mm256_add_ps(a6, _mm256_mul_ps(half, u6));
            _mm256_storeu_ps(gpk.add(48), g6);
            a6 = _mm256_mul_ps(_mm256_add_ps(a6, u6), d6);""",
"""            let gpk = gp.as_mut_ptr().add(k * GP_STRIDE);
            // 只算请求的家族：被跳过的家族 gp 槽保持 0（块首已清零），只喂给不被请求的矩阵。
            if need & FAM_CNT != 0 {
                let uc = _mm256_set1_ps(ub[U_CNT]);
                let g0 = _mm256_add_ps(a0, _mm256_mul_ps(half, uc));
                _mm256_storeu_ps(gpk, g0);
                a0 = _mm256_mul_ps(_mm256_add_ps(a0, uc), d0);
                let ucs = ub[U_CNT];
                *gpk.add(8) = a1s + 0.5 * ucs;
                a1s = (a1s + ucs) * d1s;
            }
            if need & FAM_VOL != 0 {
                let uv = _mm256_blend_ps(_mm256_set1_ps(ub[U_VOL]), zero, 0b11000000);
                let g2 = _mm256_add_ps(a2, _mm256_mul_ps(half, uv));
                _mm256_storeu_ps(gpk.add(16), g2);
                a2 = _mm256_mul_ps(_mm256_add_ps(a2, uv), d2);
            }
            if need & FAM_LOGVOL != 0 {
                let ul = _mm256_blend_ps(_mm256_set1_ps(ub[U_LOGVOL]), zero, 0b11110000);
                let g3 = _mm256_add_ps(a3, _mm256_mul_ps(half, ul));
                _mm256_storeu_ps(gpk.add(24), g3);
                a3 = _mm256_mul_ps(_mm256_add_ps(a3, ul), d3);
            }
            if need & FAM_FLOW != 0 {
                let u4 = _mm256_unpacklo_ps(
                    _mm256_set1_ps(ub[U_FLOW_P]),
                    _mm256_set1_ps(ub[U_FLOW_M]),
                );
                let g4 = _mm256_add_ps(a4, _mm256_mul_ps(half, u4));
                _mm256_storeu_ps(gpk.add(32), g4);
                a4 = _mm256_mul_ps(_mm256_add_ps(a4, u4), d4);
            }
            if need & FAM_URG != 0 {
                let u5 = _mm256_blend_ps(
                    _mm256_unpacklo_ps(
                        _mm256_set1_ps(ub[U_URG_P]),
                        _mm256_set1_ps(ub[U_URG_M]),
                    ),
                    zero,
                    0b11000000,
                );
                let g5 = _mm256_add_ps(a5, _mm256_mul_ps(half, u5));
                _mm256_storeu_ps(gpk.add(40), g5);
                a5 = _mm256_mul_ps(_mm256_add_ps(a5, u5), d5);
            }
            if need & FAM_EXT != 0 {
                let u6 = _mm256_blend_ps(
                    _mm256_unpacklo_ps(
                        _mm256_set1_ps(ub[U_EXT_P]),
                        _mm256_set1_ps(ub[U_EXT_M]),
                    ),
                    zero,
                    0b11110000,
                );
                let g6 = _mm256_add_ps(a6, _mm256_mul_ps(half, u6));
                _mm256_storeu_ps(gpk.add(48), g6);
                a6 = _mm256_mul_ps(_mm256_add_ps(a6, u6), d6);
            }""", 'p1 mask')
open(p,'w').write(s)
print('\n'.join(done))
