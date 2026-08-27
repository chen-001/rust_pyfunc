# 补充因子流程（Multica 建议 → 补充计算/回测 → 回复/合并）可行性分析报告

日期：2026-08-27
范围：`rust-pipeline-*` / `cross-section-pipeline` / `multica` skills、`rust_pyfunc` colblk 存储、`design_whatever.tail_pipeline_engine`、Multica API、`research_ext_ideas` 与 `sandbox_yhyb_topcap` 实际案例。

## 一、结论

**流程可以落地，但在调查前存在 4 个会直接阻断的框架缺口。** 现已按最小侵入方式补齐：

1. 初版 store 与补充因子无法放进同一目录并被 `tail_pipeline_engine` 一次读取；
2. 补充因子计算若复用系列根目录，会被初版 `_completed_dates` / 已完成 (date, code) 记录跳过；
3. `tail_pipeline_engine` 显式传 `names` 时会静默丢弃存储中缺失的因子，可能造成“无报错但漏因子”；
4. 用户期望的“回测前先填充缺失值，再用 nan_ratio 兜底”之前并未真正实现——填充只发生在中性化内部，raw_cover 和 preflight 仍在填充前执行。

上述 4 项均已修改并通过 `cargo check --lib`、Python API 冒烟测试。

## 二、关键代码事实（调查结果）

### 1. colblk 存储现状

- `run_factor_pipeline` / `run_factor_pipeline_cross_section` 写 8 个 shard，`finish_and_project` 投影后回测读连续列。
- `FactorStoreWriter::resume` 原来只校验因子数，不校验因子名；不同名但同长度会静默续写旧 store。
- `FactorStoreReader` 原来假设所有 shard 共用同一组 factor_names，没有“同目录多组因子”概念。
- `run_factor_pipeline_cross_section` 的断点续算用根目录 `_completed_dates`，所以补充因子直接指向系列根目录时，所有日期都显示已完成，补充因子一行都写不进去。
- `tail_pipeline_engine` 原逻辑：
  ```python
  factor_names = [n for n in names if n in name_to_idx]  # 静默丢缺失名
  ```
  这正是“不报错但遗漏补充因子”的隐患来源。

### 2. 回测前预处理现状

- `factor_neutralize_std.rs` 中确实有生产标准填充（rank → 行业 OLS 填充 → 行业中位填充 → restrict → 中性化），但它是**中性化路径**。
- `tail_pipeline_engine` 的实际顺序原为：
  1. 读 raw factor；
  2. **先**算 raw_cover 并淘汰；
  3. rank + rolling；
  4. **后** preflight（nan/zero/majority）；
  5. raw 回测；
  6. 中性化内部填充 + neu 回测。
- 所以“先填充、后 nan_ratio 兜底”不成立，必须改 Rust 引擎。

### 3. 真实 yhyb 初版现状

- store：`/hdd/user_home_unsafe/chenzongwei/factor_store_cross_yhyb`，1916 个初版因子，8 shard 已投影。
- 回测 cache：`/hdd/user_home_unsafe/chenzongwei/cross_yhyb_tail_v4`，gap5 最终入选 58 个 derived 因子。
- 修正后的严格口径：最终 gap5 名单中 channel=neu_ic 的有 13 个，其中 1 个同时通过普通 neu_ret；**排除收益通道后纯 neu_IC 入选 12 个，IC 下限为 `abs(IC_mean)=0.020080...`**。
- `sandbox_yhyb_topcap` 是 310 只行业头部股宇宙，按新规范属于不合规设计；正式补充化必须重写为全市场因子。

## 三、已实施的解决办法

### 方案 1：组合 store（同系列目录 = 初版 + 补充）

新增 `factor_groups.json` manifest 与组合读取：

```
factor_store_cross_yhyb/
├── factor_groups.json            # {"version":1,"groups":[{"name":"base","dir":"."},{"name":"supplement_x","dir":"supplement_x"}]}
├── shard_0..7/                   # 初版 group "."
└── supplement_x/shard_0..7/      # 补充 group
```

- `run_factor_pipeline_cross_section` 写补充因子时，`store_dir` 指向 `supplement_x`，不会触碰初版 `_completed_dates`。
- 补充 group 投影完成后调用 `rp.factor_store_v5_register_group(series_store, group_name, group_dir)`：
  - 校验 group 可打开、分片因子名一致、已投影；
  - 校验组合后因子名无重复；
  - 写坏 manifest 自动回滚。
- `FactorStoreReader` 自动识别 manifest，返回拼接后的全局因子名；`factor_store_v5_info`、`factor_store_v5_template`、`factor_store_v5_read_factor`、`factor_store_v5_export_factors_parquet`、tail engine 的 fast scatter 读取全部支持。
- 以后全量重跑：
  ```python
  dw.tail_pipeline_engine(
      colblk_store_dir=series_store,
      ver=f"{base_ver}_full_supplement_x",  # 新文件名/缓存
      names=None,                            # None = 全部初版+补充
      ...
      nan_max_threshold=0.04,
  )
  ```

### 方案 2：回测前填充 + nan_ratio 兜底

`tail_v5_pipeline::process_task_with_values_v7` 现改为：

1. 对 raw factor 先横截面 rank；
2. 缺失 rank 用当日中位 rank 填充；
3. 再做 raw_cover；
4. 再做 rolling/preflight（`nan_max_threshold=0.04` 只兜底）；
5. raw 回测；
6. 标准中性化（其中仍有生产填充）→ neu 回测。

覆盖三层保障：
1. 因子设计面向全市场；
2. 回测前 rank + 缺失填充；
3. `cover_rate=0.97`、`nan_max_threshold=0.04` 兜底。

### 方案 3：names 严格校验，杜绝静默遗漏

`design_whatever.tail_v4.run_tail_pipeline_engine` 显式传 `names` 时：
- 去重；
- 任何请求名不在组合 store 中则直接 `ValueError`，列出缺失因子；
- `names=None` 回测全部因子。

### 方案 4：Multica 读写

- `multica` skill 和 `scripts/multica_api.py` 从只读扩展为读写：新增 `post-comment`，POST `/api/issues/{id}/comments?workspace_id=...`，支持 `content_file` / stdin / `parent_id`。
- 服务端 route 已核实：`POST /api/issues/{id}/comments`，body 字段 `content` / `type` / `parent_id`。

## 四、两个新 skill

已创建：

- `/home/chenzongwei/.agents/skills/multica-factor-designer-and-runner`
  - 读 Multica 建议 → sandbox 小样本验证 → 正式接入 Rust → 只算补充因子 → 注册组合 store → 只回测补充因子 → 写 handoff JSON。
- `/home/chenzongwei/.agents/skills/multica-report-and-merge`
  - 用 `dw.gap5_neu_ic_floor` + `dw.evaluate_supplement_factors` 做严格判定；
  - 生成并发布 Multica 回复；
  - 决定并执行正式代码合并 / 不合并；
  - 给出未来全量重跑模板。

判定逻辑已固化为 `design_whatever/supplement_evaluation.py`：

- 补充 source factor 代表 = 其 gap5 neu 候选中 `abs(IC_mean)` 最大者（且 `ratio_mean >= cover_rate`）；
- 第一关：代表 `abs(IC) >` 初版“纯 neu_IC 入选”因子的最小 `abs(IC)`；
- 第二关：不存在初版入选因子同时满足 `abs(IC) 更高` 且 `gap5 neu IC 时间序列绝对相关 > 0.5`；
- 两关都过 → 值得补充。

## 五、已修改文件

Rust：
- `src/factor_store_v5.rs`：组合 store 读取、register_group（含模板轴一致性校验）、resume 严格校验因子名、append_batch 严格校验结果长度、project/decompress 组合根目录保护。
- `src/tail_v5_pipeline.rs`：回测前 rank + 缺失 rank 填充；metrics-only 全量指标保存；填充前/后覆盖率字段。
- `src/tail_backtest_engine.rs`：`save_all_metrics` 入口 + `prefill_coverage` 汇总返回。
- `src/factor_pipeline.rs`：禁止把组合 store 根目录当计算目录（防误清空/误续写）。
- `src/lib.rs`：注册 `factor_store_v5_register_group`。
- `python/rust_pyfunc/__init__.pyi`：`__all__` 增加新函数。

Python：
- `design_whatever/tail_v4.py`：tail engine names 严格校验、完整配置持久化、selection audit 落盘、metrics-only 汇总、prefill_coverage 落盘。
- `design_whatever/tail_v2_screen.py`：新增 `select_tail_v2_factors_with_audit`，输出选入通道归因。
- `design_whatever/tail_whatever.py`：暴露 `metrics_only` 参数。
- `design_whatever/supplement_evaluation.py`：audit 优先、严格纯 neu_IC 口径、metrics-only 强制判定、normalize fold source、expected names 全量回答。
- `design_whatever/__init__.py`：导出新判定函数。

Skills：
- `cross-section-pipeline` / `rust-pipeline-level2` / `rust-pipeline-minute`：
  - `cover_rate=0.97`、`nan_max_threshold=0.04`；
  - 覆盖率三层保障说明；
  - 补充因子追加与组合 store 指引。
- `multica`：增加 POST 写评论支持。
- 新建 `multica-factor-designer-and-runner`、`multica-report-and-merge`。

其他：
- 全库 `pythoncode` 与 tests 中 `nan_max_threshold=2` 已替换为 0.04；其余旧流程中的 0.05/0.1/0.2 也统一为 0.04。仅 `comparisons/tv4_vs_fp_compare/run_compare.py` 保留了 `nan_max_threshold=1.0`，因为该脚本是框架对比 harness，需要故意关闭 preflight 过滤。
- `/home/chenzongwei/pythoncode/一呼百应/yhyb_pipeline.py`：未来重跑模板改为 `cover_rate=0.97`、`nan_max_threshold=0.04`。

## 六、针对外部评审（GPT5.6 Sol）逐项修复记录

1. **每个补充因子必须被回答**：新增 `tail_pipeline_engine(metrics_only=True)`。
   - Rust engine 在 metrics-only 模式下保存所有通过 raw_cover 的 derived slot 的 `summary_*_all` 和 `ic_*_all`，即使 preflight 不过、收益/IC 不过候选线；
   - 候选文件仍只包含 preflight 通过且达标的因子，筛选语义不变；
   - 判定器强制读取 `summary_neu_gap5_all`，缺文件直接报错；对 raw_cover 就失败的 source factor 输出 `blocker="no_metrics"`。
2. **neu_IC 归因纠正**：新增 `select_tail_v2_factors_with_audit`，每次 tail engine 筛选都会写 `selected/selection_audit.parquet/json`。
   - `channel` 明确记录 `neu_ret_ic_more / neu_ic / raw_ic / neu_ret / raw_ret`；
   - `is_pure_neu_ic = (channel == "neu_ic") and not also_neu_ret`；
   - 真实 cross_yhyb 验证：13 个 neu_ic → 1 个与 neu_ret 重叠 → 12 个纯 IC，下限 0.020080。
3. **初版回测配置可复现**：`tail_v4_config.json` 升级 config_version=2，持久化完整 `selection_kwargs`、preflight 参数、factor_names 顺序、metrics_only 开关和 engine logic version。
4. **Skill A/B 边界重划**：Skill A 在 `supplement/<slug>` 临时分支接入 Rust pipeline，只为批量计算和 metrics-only 回测；Skill B 说值得才合并分支，不值得则 main 保持不动。
5. **存储写入安全**：`FactorStoreWriter::append_batch` 现在严格要求每条 TaskResult 因子数等于注册因子数，缺失/多余直接报错，不再补 NaN。
6. **组合注册模板校验**：`factor_store_v5_register_group` 现在要求补充 group 的 dates/stocks 与已有组合完全一致，防止模板漂移；Skill A 增加填充前覆盖率报告脚本和回测后 `prefill_coverage.parquet`。
7. **验证与回滚点**：本次改动将在 rust_pyfunc 和 design_whatever 分别建立独立分支提交；`cargo check --lib` 通过；metrics-only 已用真实 colblk 单因子短窗口端到端跑通（selected=0 时 all metrics 仍完整，判定器输出 no_eligible_derived）。

## 七、第二轮审核修复记录

1. **基线口径一致**：判定器新增 `_require_compatible_configs`，对比 base/supplement 的 `tail_v4_config.json` 中 engine_logic_version、windows/fold、日期范围、industry_neutralize、index_name、style_data_path、selection_kwargs、preflight。旧 `cross_yhyb` cache 会明确报错，必须先跑 `cross_yhyb_v2` 新基线。
2. **只合并值得的因子**：Skill A 改为在独立 git worktree 中按“一个因子一个 commit”开发；Skill B 生成 curated branch/commit，只挑选 `worth_supplementing=True` 因子的代码与注册 hunk，并检查 diff 不含不值得因子。
3. **expected names 强制闭环**：`expected_supplement_names` 改为必填 keyword-only 参数；判定结束断言结论集合与 expected 完全一致；CLI 支持 `--handoff-json` 自动读 `supplement_names`；Rust metrics-only 即使全部 raw_cover 失败也写出空 `_all` 文件。
4. **写入错误不再被吞**：Level2/minute/cross-section 三个 writer 均通过错误 channel 把 `append_batch` 错误传回主线程并返回 PyErr；分钟/横截面只有写入成功才 `mark_date_complete`；`append_batch` 先校验长度再删除旧投影。
5. **相关性最小样本**：默认 `min_corr_obs=60`，不足则 blocker=`insufficient_ic_overlap`，结果记录 `corr_n_obs`；阻挡因子改为选择相关性最高的已入选因子。
6. **无纯 neu_IC 时明确失败**：`gap5_neu_ic_floor` 不再回退到其他集合，直接报“IC 下限未定义”。
7. **环境隔离**：Skill A 使用 git worktree；metrics-only 缓存显式放 `/hdd/.../supplement_eval_cache/{supp_ver}`；metrics-only 跳过 raw parquet 导出与 postprocess。
8. **Multica 幂等**：新增 `post_reply_idempotent.py`，发布前查重 `[supplement-eval:<marker>]`，重复运行跳过。
9. **小项修复**：覆盖率脚本支持 `--restrict-npy`；全 NaN IC 输出 `no_valid_ic`；`parallel_computing.pyi` 补 `factor_store_v5_register_group` 签名；`test_tail_backtest_engine.py` 在 store 缺失时改为 SKIP；新增 `tests/test_supplement_pipeline_smoke.py`（3 个冒烟测试通过）。

## 八、验证记录

- `cargo check --lib`：通过（373 个历史 warning，无 error）。
- `cargo test --lib` 仍会被仓库里既有的无关 test 模块编译错误阻断（`difference_matrix.rs`、`column_correlation.rs`、`features.rs` 等旧测试代码，与本次改动无关）；本次组合 store 逻辑改用 Python API 冒烟测试覆盖。
- `bash alter.sh release-fast`：maturin develop + worker 部署成功。
- `rp.factor_store_v5_register_group`：两 group 冒烟测试成功，全局 3 因子读取正确，fast/slow 读取一致。
- 组合 store 重复因子名被正确拒绝。
- `dw.tail_pipeline_engine` 对缺失 names 已报错而非静默过滤；`factor_store_v5_export_factors_parquet` 同样对缺失因子名报错。
- `dw.gap5_neu_ic_floor("cross_yhyb")`：严格排除 neu_ret 重叠后，12 个纯 IC 入选，下限 0.020080。
- metrics-only 真实 colblk 单因子短窗口端到端：`tail_pipeline_engine(metrics_only=True)` 在候选 selected=0 的情况下仍生成 8 个 `summary_*_all` 文件 + 4 组 `ic_*_all`，判定器正确输出 `no_eligible_derived`；且已确认跳过 raw parquet 导出与 postprocess。
- 新增 `tests/test_supplement_pipeline_smoke.py`：3 passed；`tests/test_tail_backtest_engine.py` 在 store 缺失时由“收集失败”改为 pytest skip。
- 新版基线 `cross_yhyb_v2` 已启动重跑（后台进程，`n_jobs=200`，majority_count_threshold=10000，因为 yhyb 事件率因子天然有大量 0 值；selection/preflight 已写入 config_version=2）。完成后以 `cross_yhyb_v2` 作为唯一补充因子比较基线。
- `tail_v4_config.json` 已实测写入 config_version=2 的完整 selection/preflight/factor_names 配置。
- 回滚点：rust_pyfunc 分支 `feat/supplement-pipeline-fixes`；design_whatever 同名分支。未直接提交到 main。
- Multica 读 API 脚本保留；POST 脚本已编译通过（未实际发评论）。

## 九、仍需人工/后续注意

1. **Rust 正式接入仍是人工开发环节**：skill A 负责把 Multica 建议转成 sandbox，再由人/agent 按 5 处注册清单接入正式 pipeline；skill 无法自动发明因子公式。
2. **旧的已回测结果不会自动更新**：`cross_yhyb` 初版回测仍是旧参数旧预处理的结果；只有下一次全量重跑（新 ver）才会使用新填充和 0.04 阈值。
3. **310 股 sandbox 不能直接注册**：`sandbox_yhyb_topcap` 必须重写为全市场定义后才允许进入正式组合 store；否则 majority_count 会兜底剔除。
4. **全量重跑成本**：组合 store 投影都已就绪，`names=None` 回测所有因子；新缓存 ver 必须与初版 ver 不同。
5. **判定口径按 source factor 聚合**：本实现把一个补充 source factor 的所有 derived 候选取 max abs(IC) 作为代表；如果以后要逐 derived 因子判定，可在 `evaluate_supplement_factors` 中去掉 groupby 聚合。
