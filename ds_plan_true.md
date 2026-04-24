# build_backup_column_block_cache_single_thread 内存优化方案

## 问题诊断

当前实现存在 **两处内存峰值叠加**：

### 峰值1：v4 全量解压缓冲区（第735行）

```rust
let mut decompressed = Vec::with_capacity(record_count * record_size);
for chunk in &chunks {
    let decompressed_chunk = zstd::decode_all(compressed_data)?;
    decompressed.extend_from_slice(&decompressed_chunk);
}
```

一次性将所有压缩块解压到连续 `Vec<u8>` 中，大小 = `record_count * record_size`。

### 峰值2：全量 block_f32_data（第660行）

```rust
let mut block_f32_data: Vec<Vec<f32>> = vec![Vec::new(); block_count];
// ... 遍历 sort_index，按 block 分别 push f32 值
```

所有 block 的 f32 数据同时驻留内存，大小 = `record_count * factor_count * 4`。

### 叠加效应

以 `record_count=10,666,713, factor_count=24,192, block_count=756` 为例：
- decompressed: ~962 GB
- block_f32_data: ~961 GB
- **合计峰值: ~1.9 TB**（物理内存仅 1.5 TB）

两阶段设计导致 `decompressed` 在第735行分配后直到 `build_cache_from_v3_buffer` 返回前都不释放，
与第660行的 `block_f32_data` 同时存在，形成叠加峰值。

---

## 核心设计思想：分块组 + 逆索引

**关键约束**：缓存文件要求数据按 `(date_id, code_id)` 排序后写入。这意味着不能简单地对原始顺序的数据做流式写入。

**解决思路**：
1. 第一遍：构建字典 + sort_index（不变）
2. 构建 **逆索引** `position_in_sorted[original_row] = sorted_position`
3. 第二遍：**逐块组处理**——每次只处理 G 个 block，需要多少分配多少，写完立刻释放

这样做的原理：
- 有了逆索引，可以在**原始顺序**遍历数据时，直接 O(1) 定位每条记录在各 block 中的写入位置
- 每次只分配 G 个 block 的 f32 buffer，写完释放后再处理下一组

---

## 详细设计

### 第一步：新增逆索引构建

在 `build_cache_from_v3_buffer` 和 `build_cache_from_v2` 中，sort_index 构建完成后，新增：

```rust
// 构建逆索引：position_in_sorted[original_row] = sorted_position
let mut position_in_sorted: Vec<u32> = vec![0u32; record_count];
for (sorted_pos, &original_row) in sort_index.iter().enumerate() {
    position_in_sorted[original_row] = sorted_pos as u32;
}
// sort_index 现在可以释放（如果后续用不到的话）
```

**注意**：`sort_index` 仍需保留，因为写 meta.bin 时需要按排序后的顺序写 `(date_id, code_id)` 对。

内存开销：`position_in_sorted: Vec<u32>` = `record_count * 4` 字节。对于 10,666,713 条记录约 **41 MB**，可接受。

### 第二步：改造第二遍为逐块组处理

**核心循环结构**：

```rust
// 写入 meta.bin（按 sort_index 顺序）
let mut meta_writer = BufWriter::new(File::create(meta_path(cache_dir))?);
for &row in &sort_index {
    meta_writer.write_all(&row_date_ids[row].to_le_bytes())?;
    meta_writer.write_all(&row_code_ids[row].to_le_bytes())?;
}
meta_writer.flush()?;

// 逐块组处理 f32 数据
let group_size = (memory_budget_bytes / (record_count * block_cols * 4)).max(1);
for group_start in (0..block_count).step_by(group_size) {
    let group_end = min(group_start + group_size, block_count);
    let group_block_count = group_end - group_start;

    // 为当前组分配 f32 buffer
    let mut group_buffers: Vec<Vec<f32>> = (group_start..group_end)
        .map(|blk_idx| {
            let cols = min(block_cols, factor_count - blk_idx * block_cols);
            Vec::with_capacity(record_count * cols)
        })
        .collect();

    // 按原始顺序遍历所有记录，填充当前组的 buffer
    for row in 0..record_count {
        let sorted_pos = position_in_sorted[row] as usize;
        let offset = data_start + row * record_size;
        let record = &data[offset..offset + record_size];

        for (local_idx, blk_idx) in (group_start..group_end).enumerate() {
            let start_col = blk_idx * block_cols;
            let end_col = min(start_col + block_cols, factor_count);
            let cols = end_col - start_col;
            let base = sorted_pos * cols;
            for (j, col) in (start_col..end_col).enumerate() {
                let byte_offset = FACTOR_BASE_OFFSET + col * FACTOR_SIZE;
                let bits = u32::from_le_bytes(record[byte_offset..byte_offset + 4].try_into().unwrap());
                group_buffers[local_idx][base + j] = f32::from_bits(bits);
            }
        }
    }

    // 逐个 block 压缩写入，释放 buffer
    for (local_idx, blk_idx) in (group_start..group_end).enumerate() {
        let f32_slice = &group_buffers[local_idx];
        let raw_bytes = /* f32 转 bytes */;
        let compressed = zstd::encode_all(raw_bytes, 9)?;
        let mut w = BufWriter::new(File::create(block_path(cache_dir, blk_idx))?);
        w.write_all(&(raw_bytes.len() as u64).to_le_bytes())?;
        w.write_all(&compressed)?;
        w.flush()?;
    }
    // group_buffers 离开作用域，内存释放
}
```

**关键变化**：
- 遍历顺序从 `for &row in &sort_index`（排序后顺序）改为 `for row in 0..record_count`（原始顺序）
- 通过 `position_in_sorted[row]` 获取排序后的写入位置
- 外层循环从"一次处理所有 block"变为"每次处理 group_size 个 block"

### 第三步：改造 v4 解压逻辑

不再全量解压，改为**按需解压**——每次处理一个块组时，重新解压所有 v4 压缩块。

```rust
fn build_cache_from_v4(...) -> PyResult<CacheManifest> {
    let chunks = build_chunk_index_v4(mmap)?;

    // 第一遍：构建字典 + sort_index + position_in_sorted + 写 meta.bin
    // 这需要逐 chunk 解压，提取 date/code，但不保留因子数据
    // （或直接用现有的第一遍逻辑，因为 v3_buffer 的第一遍只需要 date 和 code）

    // 第二遍：逐块组处理
    for group_start in (0..block_count).step_by(group_size) {
        // 为当前组分配 buffer
        let mut group_buffers = ...;

        // 重新解压所有 chunk
        let mut global_row = 0usize;
        for chunk in &chunks {
            let compressed_data = &mmap[chunk.data_offset..];
            let decompressed = zstd::decode_all(compressed_data)?;
            for local_row in 0..chunk.record_count {
                let row = global_row + local_row;
                let sorted_pos = position_in_sorted[row] as usize;
                let offset = local_row * record_size;
                let record = &decompressed[offset..offset + record_size];
                // 填充 group_buffers
                ...
            }
            global_row += chunk.record_count;
            // decompressed 离开作用域，内存释放
        }

        // 写入当前组的 block 文件
        ...
    }
}
```

**v4 的权衡**：
- 需要重新解压所有 chunk 多次（每次块组一遍）
- 解压次数 = `ceil(block_count / group_size)`
- 用 CPU 换内存：解压更多次，但每次只需保留一个 chunk 的解压数据 + 当前块组的 f32 buffer

### group_size 自动计算

不再由用户手动指定 `group_size`，改为接受 **`memory_budget_gb`**（分配给 f32 buffer 的内存预算），内部自动计算：

```rust
let memory_budget_bytes = memory_budget_gb * 1024 * 1024 * 1024;
// 单个 block 的 f32 buffer 大小 = record_count * block_cols * 4
let group_size = (memory_budget_bytes / (record_count * block_cols * 4)).max(1);
group_size = group_size.min(block_count); // 不超过 block 总数
```

**`memory_budget_gb` 的默认值**：通过系统 API 读取物理内存总量，取 1/3 作为默认预算。例如：
- 1.5 TB 机器 → 默认 500 GB → group_size ≈ 400 → 仅需 2 遍扫描
- 256 GB 机器 → 默认 85 GB → group_size ≈ 68 → 约需 12 遍扫描
- 64 GB 机器 → 默认 21 GB → group_size ≈ 17 → 约需 45 遍扫描

用户也可以通过 Python 参数覆盖默认值。

### 预估耗时（以 10.67M 记录、24,192 因子、756 blocks 为例）

瓶颈是 zstd 解压 + 压缩的 CPU 时间，磁盘 I/O 次之：

| memory_budget_gb | group_size | v4 解压遍数 | 预估总耗时 |
|------------------|-----------|------------|----------|
| 500              | 400       | 3          | ~40 分钟  |
| 200              | 156       | 6          | ~80 分钟  |
| 50               | 39        | 21         | ~280 分钟 |
| 20               | 15        | 52         | ~700 分钟 |

**说明**：耗时估算基于单线程 zstd 解压/压缩吞吐约 3-4 GB/s。实际耗时受具体机器 CPU 和磁盘性能影响。用户可以根据能接受的耗时反推需要给多少 `memory_budget_gb`。

### v3 的情况

v3 使用 mmap，不需要重新解压，只需要重新遍历 mmap 数据（内存映射访问，OS 管理页换入换出）。
对于超大文件，可能会有页面抖动，但不影响正确性，且远好于 OOM。

---

## 内存峰值估算（改造后）

| 组件 | 大小（10M 记录） | 备注 |
|------|-----------------|------|
| sort_index | ~81 MB | `Vec<usize>` |
| position_in_sorted | ~41 MB | `Vec<u32>` |
| row_date_ids | ~20 MB | `Vec<u16>` |
| row_code_ids | ~20 MB | `Vec<u16>` |
| 单个块组 f32 buffer | ~`memory_budget_gb` GB | 用户指定上限 |
| v4 解压临时 buffer | ~chunk_record_count × record_size | 一个 chunk，释放快 |

固定开销约 162 MB + f32 buffer 由 `memory_budget_gb` 控制。例如默认取物理内存 1/3，1.5 TB 机器上峰值约 500 GB，远低于原来的 1.9 TB。

---

## 需要修改的函数

### 1. `build_cache_from_v3_buffer` (第581行)
- 新增 `position_in_sorted` 构建
- 将第二遍改为逐块组处理
- meta.bin 的写入与 f32 数据收集分离

### 2. `build_cache_from_v2` (第417行)
- 同样新增 `position_in_sorted` 构建
- 同样将第二遍改为逐块组处理

### 3. `build_cache_from_v4` (第720行)
- 删除全量解压逻辑
- 第一遍：逐 chunk 解压，构建字典 + sort_index
- 第二遍：逐块组处理，每遍重新解压所有 chunk
- 或者：将第一遍和第二遍融合（边解压边处理），减少解压次数

### 4. 新增参数 `memory_budget_gb`
在 `build_backup_column_block_cache_single_thread` 的 Python 接口中新增 `memory_budget_gb: Option<f64>` 参数。
- `None`（默认）：自动读取物理内存的 1/3 作为预算
- 指定值：按给定 GB 数分配 f32 buffer 内存

内部自动推导 `group_size`，无需用户关心 block 内部结构。

---

## 验证方案

1. **正确性验证**：用改造前后的代码分别构建缓存，比较所有 blk_*.bin 文件是否一致（MD5）
2. **内存验证**：用 `/usr/bin/time -v` 监控最大 RSS，确认峰值显著降低
3. **性能验证**：在 10M 记录的数据集上比较改造前后的总耗时
4. **边界测试**：用不同 `memory_budget_gb`（如 1/10/50/500）测试，验证结果一致性
