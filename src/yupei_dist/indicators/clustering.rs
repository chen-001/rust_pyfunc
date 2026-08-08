//! clustering: 加权聚类系数（降维指标.md 十三）。
//! 加权版（Zhang-Horvath 型）: CC_i = Σ_{j<k∈N(i)} (ŵ_ij·ŵ_jk·ŵ_ki)^{1/3} / C(m,2),
//! 其中 ŵ_ij = S_ij / max(S) 归一化边权, N(i) = 每行 top-50 邻居。cnt_t1 与 vol_t1。

use crate::yupei_dist::indicator_ctx::{IndicatorCtx, IndicatorResult};
use rayon::prelude::*;

pub fn name() -> &'static str {
    "clustering"
}

pub fn desc() -> &'static str {
    "加权聚类系数 (cnt_t1, vol_t1; top-50 邻居)"
}

/// top-k 部分选择: 小顶堆保持 k 个最大（值降序, 同值索引升序, 确定性）
fn topk_minheap(row: &[f32], k: usize) -> Vec<u32> {
    use std::cmp::Ordering;
    let mut heap: Vec<u32> = Vec::with_capacity(k);
    let worse = |a: u32, b: u32| -> Ordering {
        // b 更差（更小）时 a 排在 b 前 —— 小顶堆: 堆顶是最差（最小）的
        row[b as usize]
            .partial_cmp(&row[a as usize])
            .unwrap_or(Ordering::Equal)
            .then_with(|| b.cmp(&a))
    };
    for j in 0..row.len() as u32 {
        if heap.len() < k {
            heap.push(j);
            // 上浮
            let mut idx = heap.len() - 1;
            while idx > 0 {
                let parent = (idx - 1) / 2;
                if worse(heap[idx], heap[parent]) == Ordering::Less {
                    heap.swap(idx, parent);
                    idx = parent;
                } else {
                    break;
                }
            }
        } else if worse(j, heap[0]) == Ordering::Less {
            // j 比堆顶（最差）更好: 替换堆顶并下沉
            heap[0] = j;
            let mut idx = 0;
            loop {
                let l = idx * 2 + 1;
                let r = l + 1;
                let mut m = idx;
                if l < heap.len() && worse(heap[l], heap[m]) == Ordering::Less {
                    m = l;
                }
                if r < heap.len() && worse(heap[r], heap[m]) == Ordering::Less {
                    m = r;
                }
                if m != idx {
                    heap.swap(idx, m);
                    idx = m;
                } else {
                    break;
                }
            }
        }
    }
    // 堆内排序（值降序, 同值索引升序）保证确定性输出
    heap.sort_unstable_by(|&a, &b| {
        row[b as usize]
            .partial_cmp(&row[a as usize])
            .unwrap_or(Ordering::Equal)
            .then_with(|| a.cmp(&b))
    });
    heap
}

pub fn compute(ctx: &IndicatorCtx) -> Vec<IndicatorResult> {
    let n = ctx.n();
    let mut out = Vec::new();
    let k = 50usize;
    for mat in crate::yupei_dist::indicator_ctx::MATRIX_LIST {
        let sym = match ctx.symmetric(mat) {
            Some(s) => s,
            None => continue,
        };
        // 每行 top-50 邻居（二叉堆部分选择, 确定性: 同值按索引升序; 行并行）
        let nbrs: Vec<Vec<u32>> = (0..n)
            .into_par_iter()
            .map(|i| topk_minheap(&sym[i * n..(i + 1) * n], k))
            .collect();
        // 全矩阵最大边权（归一化用）
        let max_w = sym.par_iter().cloned().fold(|| 0.0f32, f32::max).reduce(|| 0.0f32, f32::max).max(1e-9);
        let col: Vec<f32> = (0..n)
            .map(|i| {
                let nbr = &nbrs[i];
                let m = nbr.len();
                if m < 2 {
                    return 0.0f32;
                }
                let mut num = 0.0f64;
                for a in 0..m {
                    let ja = nbr[a] as usize;
                    for b in (a + 1)..m {
                        let jb = nbr[b] as usize;
                        let wij = (sym[i * n + ja] / max_w) as f64;
                        let wjk = (sym[ja * n + jb] / max_w) as f64;
                        let wki = (sym[jb * n + i] / max_w) as f64;
                        if wij > 0.0 && wjk > 0.0 && wki > 0.0 {
                            num += (wij * wjk * wki).cbrt();
                        }
                    }
                }
                let denom = (m * (m - 1) / 2) as f64;
                if denom > 0.0 {
                    (num / denom) as f32
                } else {
                    0.0
                }
            })
            .collect();
        out.push(IndicatorResult::new(format!("clu_{mat}_coef"), col));
    }
    out
}
