//! 手写分块 f32 GEMM: C[m×n] = A[m×k] · Bᵀ[n×k]
//! 即 C[i][j] = Σ_t A[i][t] · B[j][t]
//!
//! 设计（内存友好原则）:
//!   - B 预先转置为 Bᵀ(k×n, 行=t 列=j)，使内层对 j 的访问连续 → 自动向量化
//!   - 外层按 C 的行块(i-block) rayon 并行，每线程独立写自己的 C 行，无竞争
//!   - 内层 k 分块保证 A/B 瓦片驻留 L2，寄存器阻塞 4×8 提高浮点密度
//! 适用场景: N×T @ T×N 型相关/计数矩阵（T 很大时此布局带宽最优）

const TILE_M: usize = 64;
const TILE_N: usize = 64;
const TILE_K: usize = 512;

/// 转置: bt[k×n] = b[n×k]ᵀ（行主序）
pub fn transpose(n: usize, k: usize, b: &[f32], bt: &mut [f32]) {
    // 分块转置改善缓存
    const TB: usize = 64;
    for i0 in (0..n).step_by(TB) {
        let i1 = (i0 + TB).min(n);
        for j0 in (0..k).step_by(TB) {
            let j1 = (j0 + TB).min(k);
            for i in i0..i1 {
                let src = &b[i * k + j0..i * k + j1];
                for (jj, &v) in src.iter().enumerate() {
                    bt[(j0 + jj) * n + i] = v;
                }
            }
        }
    }
}

/// C[m×n] = A[m×k] · Bᵀ[n×k]；bt 为 k×n 的 B 转置（调用方提供）
/// c 初值会被覆盖（不累加）
pub fn gemm_abt(a: &[f32], bt: &[f32], m: usize, n: usize, k: usize, c: &mut [f32]) {
    use rayon::prelude::*;
    let nt_m = m.div_ceil(TILE_M);
    c.par_chunks_mut(TILE_M * n).enumerate().for_each(|(ti, cblock)| {
        let i0 = ti * TILE_M;
        let i1 = (i0 + TILE_M).min(m);
        let rows = i1 - i0;
        let nt_n = n.div_ceil(TILE_N);
        let mut acc = vec![0f32; rows * n];
        for k0 in (0..k).step_by(TILE_K) {
            let k1 = (k0 + TILE_K).min(k);
            let kb = k1 - k0;
            for tj in 0..nt_n {
                let j0 = tj * TILE_N;
                let j1 = (j0 + TILE_N).min(n);
                let nb = j1 - j0;
                // 寄存器阻塞微内核: 4 行 × 8 列 块
                let mut r0 = 0usize;
                while r0 < rows {
                    let r1 = (r0 + 4).min(rows);
                    let mut c0 = 0usize;
                    while c0 < nb {
                        let c1 = (c0 + 8).min(nb);
                        // acc 局部累加（4×8）
                        let mut acc4x8 = [0f32; 32];
                        let nr = r1 - r0;
                        for t in 0..kb {
                            let mut av = [0f32; 4];
                            for (rr, v) in av.iter_mut().enumerate().take(nr) {
                                *v = a[(i0 + r0 + rr) * k + k0 + t];
                            }
                            for cc in 0..(c1 - c0) {
                                let bv = bt[(k0 + t) * n + j0 + c0 + cc];
                                for rr in 0..(r1 - r0) {
                                    acc4x8[rr * 8 + cc] += av[rr] * bv;
                                }
                            }
                        }
                        for rr in 0..(r1 - r0) {
                            for cc in 0..(c1 - c0) {
                                acc[(r0 + rr) * n + j0 + c0 + cc] += acc4x8[rr * 8 + cc];
                            }
                        }
                        c0 = c1;
                    }
                    r0 = r1;
                }
            }
        }
        // 写回 C 行（每任务只写自己的行块，无竞争）
        for (rr, row) in acc.chunks(n).enumerate() {
            if rr < rows {
                cblock[rr * n..rr * n + n].copy_from_slice(&row[..n]);
            }
        }
    });
}

/// 简单版本（T 小时用）: C[m×n] = A[m×k] · Bᵀ[n×k]，b 为 n×k 原布局，内部转置
pub fn gemm_abt_simple(a: &[f32], b: &[f32], m: usize, n: usize, k: usize, c: &mut [f32]) {
    let mut bt = vec![0f32; k * n];
    transpose(n, k, b, &mut bt);
    gemm_abt(a, &bt, m, n, k, c);
}
