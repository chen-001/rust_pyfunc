# py_yupei_pair 家族裁剪补丁（可审、可回退）

**状态**：补丁曾应用到工作区并编译到 perf_sandbox/yp_pkg（未部署），
工作区 src/ 已按 git HEAD 还原；生产 python/rust_pyfunc/rust_pyfunc.abi3.so 仍是 10-11 23:07 那份，没动过。

## 改了什么（三个文件，168 增 91 删）

- src/yupei_dist/matrix_stage.rs：新增六个权重家族位 FAM_CNT..FAM_EXT 和 FAM_ALL，
  以及 family_mask_of(矩阵名列表) 反推需要的家族位；phase1_simd / phase2_simd / 两个标量回退
  都加了一个编译期常量 NEED，每个家族的计算块用 `if NEED & FAM_X != 0` 包起来；
  compute_matrices 多一个 need 参数，只在 need == cnt|vol 时走裁剪版，其余一律全算。
- src/yupei_dist/compute.rs：新增 compute_yupei_dist_partial_need(date, need)，
  旧入口 compute_yupei_dist_partial(date) 变成拿 FAM_ALL 的薄包装，py_yupei_dist 那条路不受影响。
- src/yupei_pair/mod.rs：按本次请求的 mats 算 need 再调 _need 入口（need 为 0 时退回 FAM_ALL）。

## 为什么数值不变

- 每个家族自己的 gp 槽、pack 通道、累加顺序一字未改；跳过的家族只喂给**未被请求**的矩阵。
- gp 缓冲每块开头本来就会清零，跳过的家族槽保持 0，phase2 对它们的 FMA 结果也是 0，
  这些通道只属于未被请求的矩阵，输出里根本不会被读。
- need 是从请求的矩阵名反推的，天然覆盖请求的每一个矩阵，不存在“请求了却没算”的情况。
- 非 cnt|vol 的组合走全算路径，行为和改前完全一致。

## 预期省多少（读代码推算，未实测）

phase2 是上游的大头（50 并行那次：线程累计 9834 秒、占上游计算量 97%）。
每个单元原来读 7 个 256 位向量（56 槽）+ 10 次 FMA，裁剪后读 3 个 + 3 次 FMA，
预期上游 217 秒那段降到 80~110 秒（50 并行）。**必须实测，不写进结论。**

## 怎么验收

1. 复现守卫的 py_yupei_pair 基准：A 组 20241231 默认 15 张（5107 × 10515 = 5370 万个值），
   B 组 20241231 单张 cnt_t1（358 万个值），逐位比，不设容差；codes 和 names 顺序一起比。
2. 同一输入换并行数（10 与 50）逐位一致。
3. 改前改后各跑一遍，同一天、同并行数、同参数。

## 怎么应用 / 回退

- 应用：`cd /home/chenzongwei/rust_pyfunc && git apply perf_sandbox/yupei_fam_trim.patch`
  （或重放 patch_stage_1.py / patch_stage_2.py / patch_stage_3.py）
- 编译（不部署）：`cargo build --profile release-fast --jobs 8`，产物在 target/release-fast/
- 回退：`git checkout -- src/`
- **不要跑 alter.sh**，那会把产物部署到生产位置。

## 复现材料

- 完整 diff：perf_sandbox/yupei_fam_trim.patch（368 行）
- 三个改动后的文件副本：perf_sandbox/matrix_stage.patched.rs、compute.patched.rs、mod.patched.rs
- 补丁脚本：perf_sandbox/patch_stage_1.py、patch_stage_2.py、patch_stage_3.py
- 验证脚本：perf_sandbox/yp_verify_patch.py（对照 perf_sandbox/yp_out/vals.npy 逐位比）

## 还没做的（第二处重复计算）

net 和 topk 对同一批 1303 万个配对各扫一遍，用的是同一个算 h/gap 的函数、同样的配对顺序。
两处可以合并成一遍：外层仍按 net 的 256 块跑，每块内按 topk 的 512 边界分两段各自累计，
这样两边各自的累加分组都不变，数值应当逐位一致。省的估计 140~190 秒（占 868 秒的 16%~22%），
也是推算，要实测。**这部分还没写代码。**