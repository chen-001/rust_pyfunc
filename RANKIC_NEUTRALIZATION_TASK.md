# RankIC 计算方案独立设计任务（问题陈述）

> **重要提示**：本任务要求独立完成数学推导、数值验证与设计建议。
> 请勿阅读或参考本仓库中任何已有的“分析/交接/实验”类文档与验证工程（如 *_HANDOFF、*_ALTERNATIVES、*_REPORT、sandbox 实验代码等的存在与否与内容），
> 也不要假设它们给出的任何结论；所有判断必须由你在本任务中从零推导并验证。

---

## 1. 任务目标

为一个 A 股因子研究流程中的“因子筛选”环节（高性能回测引擎）设计 RankIC 的计算方案。
背景动机：现有实现中，每个因子、每个派生 slot 都要独立执行一次“横截面中性化”计算，
耗时占比显著；备选思路是“只对收益率序列做一次中性化、所有因子共享”，从而大幅减少重复计算。

需要回答的核心问题：

1. 备选思路在数学与金融含义上是否成立？它与现状 IC 的关系是什么（等价 / 近似 / 其他）？
2. 是否存在一种方案，既保留“收益率侧只算一次”的提速特性，又使计算结果与现状 IC **完全一致**？
   若不存在，请给出数学上的原因，并给出在该维度上可达到的最佳方案及其代价。
3. 基于分析给出推荐方案（或推荐取舍），并给出可执行的落地路径。

## 2. 现状（作为问题的输入事实）

### 2.1 计算链路

现状“中性化 RankIC”（下称 IC_A）的完整链路（代码位置见 2.2）：

1. 原始因子 -> 横截面 rank + 缺失值中位秩填充；
2. 生成派生 slot（`_smooth_1` 及 mean/max/min/std 滚动窗口）；
3. 每个 slot 做“标准中性化”：
   - 再次横截面百分位秩；
   - 缺失值按行业分组填充、restrict 非开放股置空；
   - 对风格矩阵做 OLS 残差化（barra 10 风格 + 申万一级行业 one-hot 的截面回归，无截距语义）；
   - 残差再次百分位秩；
4. 对当期有效股票截面，计算上述结果（x_neu）与未来收益 r 的秩相关（ordinal 秩 + 秩差平方和公式）。

注：以上步骤序、数值语义（如秩的平均秩/ordinal 处理、tie 规则、日期对齐：信号 t-1 日、收益 t 日起的未来收益、风格暴露取信号日）以源码与可执行基线为准。

### 2.2 权威参照

- 现状引擎：`rust_pyfunc/src/tail_backtest_engine.rs`、`src/tail_v5_pipeline.rs`（`process_v7_slot`、`legacy_backtest_gap1_gap5_single_slot`、`legacy_spearman_correlation`）；
- 中性化：`rust_pyfunc/src/factor_neutralize_std.rs`（`neutralize_std_section` / `neutralize_std_block_py`）；
- 独立可调用的基线：`rp.neutralize_std_block_py` + 引擎输出的 `ic_neu_gap1/gap5.npy`（同输入下与 2.1 链路一致）。

### 2.3 性能观察

可自行 profiling 确认：现状耗时集中在“标准中性化”环节（每因子每 slot 全矩阵运算与多次排序），
IC 计算与预处理的其余部分占比较小。请以你测到的数据为准。

## 3. 备选思路（待验证）

“对收益率序列做中性化（同样风格回归的一次性截面残差化），再计算原始因子与中性化收益之间的秩相关”。
直觉优势：收益侧只算一次，所有因子共享；因子侧不再逐 slot 做风格回归。

要求：从第一性原理验证该思路与 IC_A 的关系，而不是假设其一致性或不一致性。

## 4. 任务要求

### 4.1 数学分析

- 形式化定义两种口径（或你发现的其他口径）与 IC_A 的关系；
- 推导“数值完全一致”的充要条件（若存在）；
- 若严格一致不可实现，量化其差异来源与量级（理论层面）。

### 4.2 数值验证（必须在 sandbox 中完成，不改主项目代码）

- 实现候选算法（至少含：现状复刻基线 + 备选思路 + 若存在的“精确等价”方案）；
- 用真实数据对照基线，量化：
  - 数值一致性（最大绝对差、逐位一致比例、时间序列相关系数）；
  - 方向/排序一致性（符号一致率、跨因子排序秩相关）；
  - 提速倍数（同输入、同规模、多次计时取稳定值）。
- 数据与口径请与基线严格对齐（日期、有效股票集合、风格矩阵、行业分组、秩/ tie 规则）。

### 4.3 输出

1. 分析报告：数学推导 + 金融含义 + 实证结果（含上述量化指标与实验设置）；
2. 推荐方案与落地路径（含对现有阈值/参数体系的迁移影响评估）；
3. 若“完全一致”与“显著提速”不可兼得：给出取舍全景（各方案的代价表）并说明推荐理由。

## 5. 环境与数据

- Python：`/home/chenzongwei/.conda/envs/chenzongwei311/bin/python`；
- 因子数据（colblk 存储）：`/hdd/user_home_unsafe/chenzongwei/factor_store_挂单猫0701c`（`rp.factor_store_v5_info` / `factor_store_v5_template` / `factor_store_v5_read_factor`）；
- 回测共享输入（收益/restrict 等 npy）：`/home/chenzongwei/pythoncode/_tail_v2_shared/backtest_inputs/000905_20170103_20260522_5438_6b8884a67f05d221/`；
- 风格数据：`/home/chenzongwei/database/barra/barra_daily_together_jason.parquet`（列 `date, code, value_0..value_9, ind_1..ind_31`）；
- 行业矩阵：`design_whatever.tail_v4._load_industry_matrix`（含默认行业 CSV 路径）；
- sandbox 规范：新建/使用 `rust_pyfunc/sandbox`（Cargo 依赖对齐主项目：pyo3 0.18 / numpy 0.18 / ndarray 0.15 / nalgebra 0.32，`bash build.sh` 安装为 `dev_sandbox`）；
- 生产口径对照方式：`rp.neutralize_std_block_py`（输入 factor_block / industry / restrict / style_data_path / dates / stocks / industry_neutralize）。

## 6. 验收最低标准（可自行强化）

- 数学结论可复核（推导链完整、符号与假设清晰）；
- 数值验证与基线同口径、同输入、可复现（给出输入准备与运行脚本）；
- “精确等价”类结论必须有逐位/机器精度级证据；“近似”类结论必须有量化差异（相关系数、符号一致率、排序一致率、偏差量级）；
- 提速结论给出测量条件（构建模式、线程数、规模、时长）。

---

*（本文件只描述问题，刻意不含任何先前结论；完成后请将交付物与结论记录在独立文件中，避免污染本问题陈述。）*
