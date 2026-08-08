//! top-k 部分选择工具（确定性: 值降序, 同值索引升序）。
//! 小顶堆保持 k 个最大, 用于替代全排序的 O(n·log k) 方案。

/// 每行 top-k 索引（值降序, 同值索引升序; 确定性）。
/// row 长度 < k 时返回全部（仍按值降序）。
pub fn topk_indices(row: &[f32], k: usize) -> Vec<u32> {
    use std::cmp::Ordering;
    let n = row.len();
    let k = k.min(n);
    if k == 0 {
        return Vec::new();
    }
    // 小顶堆: 堆顶是最差（最小）候选
    let worse = |a: u32, b: u32| -> Ordering {
        row[b as usize]
            .partial_cmp(&row[a as usize])
            .unwrap_or(Ordering::Equal)
            .then_with(|| b.cmp(&a))
    };
    let mut heap: Vec<u32> = Vec::with_capacity(k);
    for j in 0..n as u32 {
        if heap.len() < k {
            heap.push(j);
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
    heap.sort_unstable_by(|&a, &b| {
        row[b as usize]
            .partial_cmp(&row[a as usize])
            .unwrap_or(Ordering::Equal)
            .then_with(|| a.cmp(&b))
    });
    heap
}
