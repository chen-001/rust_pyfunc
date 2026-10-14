# py_yupei_pair 第三处提速：每个配对的七到九遍扫描并成三遍

改动位置：`src/yupei_pair/net.rs`（调用处）+ `src/yupei_pair/simd.rs`（新增三个函数）。
基线：v4（家族裁剪 + net/topk 合并扫描）之上。补丁见文末。

## 一、改了什么

一个配对 (B,C) 里，原来对 h、g 两个长度 n 的数组做了 9 遍独立扫描（n=5107）：

| 原来的遍 | 干什么 |
|---|---|
| 1 | `fill_h` 算 h = wb·p + wc·q + addm |
| 2 | `fill_gap` 算 g = h − max(p,q) |
| 3 | `argmax_first(hb)` 扫最大值 |
| 4 | `argmax_first(hb)` 再扫第一个等于最大值的下标 |
| 5 | `argmax_first(gb)` 扫最大值 |
| 6 | `argmax_first(gb)` 再扫下标 |
| 7 | `pos_stats` 扫 Σrelu(h)、Σrelu(g)、Σrelu(g)²、#(g>0) |
| 8 | `mass_acc_h` 扫六个矩数组里的三个 |
| 9 | `mass_acc_g` 扫另外三个 |

现在并成 3 遍：

- 第一遍 `fused_hg_stats`：算 h、g 并写回（b、c 位置直接写 −inf），同一趟里累计两个最大值的
  8 路 max 累加器、`pos_stats` 的三个 8 路和累加器与计数。遍 1~3 与 5、7 合成这一遍。
- 第二遍 `argmax_first_two`：一趟里同时找 h、g 里第一个等于最大值的下标。
- 第三遍 `mass_acc_hg`：一趟里对每个 x 先做 h 口径、再做 gap 口径的六个矩数组更新。

去掉了 6 遍数组扫描。数据都在 L2 内（每对 hb/gb 各 20KB，六个矩数组每个分块私有），
省的是 L2 读写次数与循环开销。

## 二、为什么数值逐位不变

1. h 的算式一字未动：向量路径仍是 `add(add(mul(wb,p), mul(wc,q)), addm)`，标量路径仍是
   `wb*p + wc*q + addm`（保持两次舍入，不用 fma）；g 仍是 `h − max(p,q)`，向量用
   `_mm256_max_ps`、标量用 `if p>q {p} else {q}`，与原 `fill_gap` 完全一样。
2. b、c 两位置原来是在 fill 之后单独置 −inf，现在在第一遍里直接写 −inf，argmax 与 topk
   看到的数组内容一致。
3. 最大值的归约顺序不变：8 路 max 累加 → 按 lane 0..8 依次比较 → 尾巴按 x 升序比较
   （严格大于，并列保留先到者）。
4. `pos_stats` 的归约顺序不变：8 路和累加 → 按 lane 0..8 依次相加 → 尾巴按 x 升序累加。
   尾巴仍是「先 lane 归约、再加尾巴」，没有把尾巴单独求和后一次加上去。
5. 下标的查找顺序不变：仍是 x 升序、先到先得；`mx_g` 为 −inf 时按原语义不扫 g。
   含 b/c 的那一个 8 宽块改成标量算这 8 个元素，但按原 lane 位置累加，不影响归约顺序。
6. 六个矩数组互不相干（三个 h 口径、三个 gap 口径），每个元素每个配对只被加一次，
   所以合并成一趟后每个数组看到的加法序列与分开调用时逐位相同；`h>0` / `g>0` 以及
   「和大于 0 才累加」两个条件都照旧。

## 三、验证方法

1. 单张 `mats=["cnt_t1"]`（5107 × 701 = 3,580,007 个值）与改前存档逐位比，要求 0 个不同。
2. 默认 15 张（5107 × 10515 = 53,699,505 个值）与改前存档逐位比，要求 0 个不同，
   并核对 NaN 位置、股票顺序、列名顺序。
3. 换线程数结果一致：10 并行与 50 并行各跑一遍，两份结果逐位比。
4. 标准测速：20241231、默认 15 张、50 并行、机器空载，与 423.41 秒（改前）和
   392.23 秒（v4）比。

## 四、怎么重打 / 回退

```bash
# 全量（v4 + 第三处），从干净的 HEAD 上打
cd /home/chenzongwei/rust_pyfunc && git checkout -- src/ && git apply perf_sandbox/yupei_opt_v5.patch
# 只打第三处（先打 v4）
git checkout -- src/ && git apply perf_sandbox/yupei_opt_v4.patch && git apply perf_sandbox/yupei_opt_v5_third_only.patch
# 编译
PATH=$HOME/.cargo/bin:$PATH cargo build --profile release-fast --jobs 8
# 回退
git checkout -- src/
```

## 五、收益

预期（读代码推算，未实测前）：第三处把扫描遍数从 9 降到 3，net 那段是本流程里最重的一段，
估省 40~75 秒（50 并行、空机器口径）。实测数字出来前这只是预估。
