//! spectral: 谱隙与第二/三集体模式（统计物理扩展，20251031 原型验证版）。
//! 对对称矩阵 S_sym = S_dir + S_dirᵀ（对角 0）做幂迭代:
//!   λ1,v1 = 主模式; λ2,v2 = 去 v1 的 deflated 幂迭代; λ3,v3 = 去 v1,v2 的双 deflate。
//! 每股输出:
//!   eig2_{mat}_coord/part, eig3_{mat}_coord/part: 第二/三模式坐标与参与度 N·v²;
//!   spec_{mat}_gap       = (λ1−λ2)/λ1 标量广播;
//!   spec_{mat}_lam2share = λ2/Σ_ij S_ij 标量广播（Σλ_i 用总交互能量 ΣS 代理, sym 对角 0）。
//! 物理含义: gap 收缩 → 市场模式之外有竞争的集体模式（分裂/板块化）;
//! lam2share 是次模式相对总强度的能量占比; v2/v3 给出各次模式载股。
//! 确定性: splitmix64 固定种子 LCG 初始化（与 eigenvector.rs 同款）。

use crate::yupei_dist::indicator_ctx::{IndicatorCtx, IndicatorResult};
use rayon::prelude::*;

pub fn name() -> &'static str {
    "spectral"
}

pub fn desc() -> &'static str {
    "谱隙+次模式: deflated幂迭代 v2/v3/part/gap/lam2share (28矩阵)"
}

/// 确定性 LCG 初始化（与 eigenvector.rs 相同种子与步进）
fn seed_vec(n: usize) -> Vec<f64> {
    let mut seed: u64 = 0x9E3779B97F4A7C15;
    (0..n)
        .map(|_| {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((seed >> 33) as f64 / u32::MAX as f64) * 2.0 - 1.0
        })
        .collect()
}

/// 行并行 matvec（f32 矩阵 × f64 向量，f64 累加）
fn matvec(m: &[f32], v: &[f64], n: usize) -> Vec<f64> {
    let mut w = vec![0.0f64; n];
    w.par_chunks_mut(64).enumerate().for_each(|(ci, chunk)| {
        let i0 = ci * 64;
        for (ii, o) in chunk.iter_mut().enumerate() {
            let i = i0 + ii;
            let row = &m[i * n..(i + 1) * n];
            let mut acc = 0.0f64;
            for t in 0..n {
                acc += row[t] as f64 * v[t];
            }
            *o = acc;
        }
    });
    w
}

/// deflates 内每个向量都在每步正交化去除（v1, v2, ...）。
/// 返回 (Rayleigh 商 λ, 单位向量); 收敛: |Δλ| ≤ 1e-8·max(1,λ) 或达 iters 上限。
fn power_mode(m: &[f32], n: usize, iters: usize, deflates: &[&[f64]]) -> (f64, Vec<f64>) {
    let mut v = seed_vec(n);
    let mut lam = 0.0f64;
    for _ in 0..iters {
        let mut w = matvec(m, &v, n);
        for u in deflates {
            let dot: f64 = w.iter().zip(u.iter()).map(|(a, b)| a * b).sum();
            w.iter_mut().zip(u.iter()).for_each(|(a, b)| *a -= dot * b);
        }
        let l: f64 = w.iter().map(|&x| x * x).sum::<f64>().sqrt();
        if l < 1e-12 {
            break;
        }
        w.iter_mut().for_each(|x| *x /= l);
        v = w;
        if (l - lam).abs() <= 1e-8 * l.max(1.0) {
            break;
        }
        lam = l;
    }
    // Rayleigh 商得带符号 λ（deflate 后范数法只给 |λ|）
    let av = matvec(m, &v, n);
    let lam_rq: f64 = av.iter().zip(v.iter()).map(|(a, b)| a * b).sum();
    (lam_rq, v)
}

/// 符号统一: 最大绝对值元素为正
fn fix_sign(v: &mut [f64]) {
    let mut mi = 0usize;
    for (i, &x) in v.iter().enumerate() {
        if x.abs() > v[mi].abs() {
            mi = i;
        }
    }
    if v[mi] < 0.0 {
        v.iter_mut().for_each(|x| *x = -*x);
    }
}

pub fn compute(ctx: &IndicatorCtx) -> Vec<IndicatorResult> {
    let n = ctx.n();
    let mut out = Vec::new();
    if n == 0 {
        return out;
    }
    for mat in crate::yupei_dist::indicator_ctx::MATRIX_LIST {
        if !ctx.wants_any(&["eig2_", "eig3_", "spec_"], mat) {
            continue;
        }
        let sym = match ctx.symmetric(mat) {
            Some(s) => s,
            None => continue,
        };
        let (lam1, mut v1) = power_mode(&sym, n, 40, &[]);
        fix_sign(&mut v1);
        let (lam2, mut v2) = power_mode(&sym, n, 60, &[&v1[..]]);
        fix_sign(&mut v2);
        // v3 只在 eig3 列幸存时算（未幸存矩阵省一组 80 次双 deflate 幂迭代）
        let v3 = if ctx.wants("eig3_", mat) {
            let (_l3, mut v3) = power_mode(&sym, n, 80, &[&v1[..], &v2[..]]);
            fix_sign(&mut v3);
            Some(v3)
        } else {
            None
        };
        let gap = if lam1.abs() > 1e-12 {
            ((lam1 - lam2) / lam1) as f32
        } else {
            f32::NAN
        };
        // lam2share = λ2 / Σ_ij S_ij（总交互能量代理; sym 对角为 0 故 Tr=0 不能当分母）
        let sum_s: f64 = (0..n)
            .into_par_iter()
            .map(|i| sym[i * n..(i + 1) * n].iter().map(|&v| v as f64).sum::<f64>())
            .sum();
        let lam2share = if sum_s.abs() > 1e-12 {
            (lam2 / sum_s) as f32
        } else {
            f32::NAN
        };
        emit(ctx, &mut out, format!("eig2_{mat}_coord"),
            v2.iter().map(|&x| x as f32).collect());
        emit(ctx, &mut out, format!("eig2_{mat}_part"),
            v2.iter().map(|&x| (x * x * n as f64) as f32).collect());
        emit(ctx, &mut out, format!("spec_{mat}_gap"), vec![gap; n]);
        if let Some(ref v3) = v3 {
            emit(ctx, &mut out, format!("eig3_{mat}_coord"),
                v3.iter().map(|&x| x as f32).collect());
            emit(ctx, &mut out, format!("eig3_{mat}_part"),
                v3.iter().map(|&x| (x * x * n as f64) as f32).collect());
        }
        emit(ctx, &mut out, format!("spec_{mat}_lam2share"), vec![lam2share; n]);
    }
    out
}

/// 列级过滤: keep=None 全发; keep=Some 只发名单内列
/// （矩阵级 wants 只管"算不算", 幸存矩阵的未幸存兄弟列不外发）。
fn emit(ctx: &IndicatorCtx, out: &mut Vec<IndicatorResult>, name: String, values: Vec<f32>) {
    if let Some(k) = &ctx.keep {
        if !k.contains(name.as_str()) {
            return;
        }
    }
    out.push(IndicatorResult::new(name, values));
}
