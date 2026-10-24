//! top-k 部分选择工具（确定性: 值降序, 同值索引升序）。
//! 小顶堆保持 k 个最大, 用于替代全排序的 O(n·log k) 方案。

/// 每行 top-k 索引（值降序, 同值索引升序; 确定性）。
/// O(n) quickselect 划界 + O(k log k) 排序。
/// （旧的小顶堆实现方向写反，会返回"前 k 列+游走最大值"；已废弃。）
pub fn topk_select_indices(row: &[f32], k: usize) -> Vec<u32> {
    let n = row.len();
    let k = k.min(n);
    if k == 0 {
        return Vec::new();
    }
    let cmp = |&a: &u32, &b: &u32| {
        row[b as usize]
            .partial_cmp(&row[a as usize])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.cmp(&b))
    };
    let mut idx: Vec<u32> = (0..n as u32).collect();
    // 第 k 名（含）切分: 前 k 个恰为全序意义下的 top-k（比较器是总序, 无 Equal 歧义）
    idx.select_nth_unstable_by(k - 1, cmp);
    let mut head = idx[..k].to_vec();
    head.sort_unstable_by(cmp);
    head
}

/// 每行 top-k 索引（值降序, 同值索引升序; 确定性）。
/// row 长度 < k 时返回全部（仍按值降序）。
/// 2026-10 修复：原小顶堆方向写反（堆顶放"最好"，j 更好才替换，实际把 top-k 换成前 k 列）。
/// 现为 topk_select_indices 的别名；存量 motifs/cliqueness/starness/edge_dynamics/clustering
/// 的 top-k 邻居口径经此修复后恢复正确语义（因子值会变，属修复而非破坏）。
pub fn topk_indices(row: &[f32], k: usize) -> Vec<u32> {
    topk_select_indices(row, k)
}
