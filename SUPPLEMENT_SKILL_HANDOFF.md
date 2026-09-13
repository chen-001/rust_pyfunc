# 补充因子流程交接文档（2026-08-27 第三轮）

日期：2026-08-27（晚）
状态：**GPT 评审整改完成；正式基线 cross_yhyb_v2 验证通过；两仓库基础设施已合并 main。剩试点与启用。**

---

## 1. 本轮完成（GPT 评审整改）

- P0-1 修 Multica 回复脚本语法错误（字符串未闭合，已 ast 校验）。
- P1-1 修最高相关阻挡因子比较索引错误（corr 与样本数误比）。
- P1-2 修 copy_subset 稀疏写丢模板轴：新增内部函数 copy_subset_write，全 NaN 日期/股票补占位行显式继承源轴；
  Rust 单测 + `examples/verify_copy_subset_axes.rs`（`cargo run --example verify_copy_subset_axes`）验证通过。
- P1-4 横截面 writer 收尾校验：成功写入日期数 == 请求日期数，空 batch/失败日期报错上抛。
- P1-6 折中方案落地：保留“相关性 pair 样本不足不阻挡”口径，新增自身 IC 历史闸门
  （代表因子有效 IC 日数 < min_ic_obs=60 → insufficient_ic_history）；结果表新增 ic_n_obs 列。
- 小项：prefill CSV 先于闸门列落盘；删除 --no-gate 与自定义阈值（硬闸门不可绕过）；
  evaluate MD 补 floor_source_names/corr_n_obs/ic_n_obs/insufficient 区块；
  SKILL A 回测参数全量从基线 config 复制 + 日期轴用初版 store 模板；
  SKILL B 重排为 合并→构建→验名→accepted group 注册（带回滚）；评估 store 描述改为实话；
  可行性报告加滞后指针。

## 2. 正式基线 cross_yhyb_v2（验证通过）

- gap5 入选 58 个、纯 neu_IC 入选 12 个（与旧 cache 复算一致）。
- 纯 IC 下限 `0.019235`（v2 新引擎口径；旧 cache 为 0.020080，差异来自引擎逻辑统一，属预期）。
- config 已含 engine_logic_version v3、min_valid、行业路径、.so sha256、8 个输入源指纹。
- raw parquet 导出（收尾阶段）后台进行中，日志 `cross_yhyb_v2_rerun_v2.log`。

## 3. 合并 main 完成

- rust_pyfunc：`9c72608`（含并行提交 83de46d factor_taskd PID 防护）。
- design_whatever：`2dc998a`。
- 两仓库 main 均未 push（按纪律等指示）。

## 4. 提交记录（分支）

rust_pyfunc：`3329ed6`（GPT 整改）/ `4edec95`（并发评审整改）/ `9c2d510` / `02912f1`
design_whatever：`04179c9` / `e764e68` / `c7dee9d`

## 5. 待办（下一轮）

1. 等基线 raw 导出收尾完成（bash-6 后台任务），确认 RESULT 行与 RUN_INFO。
2. 从 main 执行 `bash alter.sh release-fast` 重建正式 .so/worker。
3. 从 main 建 worktree，选一条真实 Multica 评论跑补充因子试点（Skill A 全流程 → Skill B 判定/回复/合并）。
4. 试点通过后正式启用两个 skill。

## 6. 关键口径备忘（不变项）

- 判定器比较 engine_logic_version（不比较 .so sha）；build sha 仅作审计记录。
- 覆盖硬闸门 0.85/0.60（restrict 分母）不可配置；nan_max_threshold=0.04 兜底；填充后 cover_rate>=0.97。
- 评估 store 不注册；accepted store 由 build_accepted_group.py 在合并构建验名之后注册。
