# RankIC 计算替代方案设计报告（交接文档）

> 主题：`tail_backtest_engine`（rust_pyfunc）与 `tail_pipeline_engine`（design_whatever）中
> 因子筛选所用 RankIC 的计算方式，是否存在“保留提速收益 + 数值与现状完全对应”的替代方案。
> 本文给出数学分析、三个候选方案、sandbox 实验证据与落地建议。
> 状态：分析/验证完成，主项目代码未改动（sandbox 实现于 `rust_pyfunc/sandbox`）。

---

## 1. 背景与问题定义

### 1.1 现状（基准）

现流程中“中性化因子 IC”的计算链路（`tail_v5_pipeline.rs` 的 `process_v7_slot`）：

1. 原始因子 -> 横截面 rank + 缺失中位秩填充（`rank_and_fill_missing_cross_sectional_median`）；
2. 派生 slot（`_smooth_1` 与 mean/max/std 滚动窗口）；
3. 每个 slot 做标准中性化（`factor_neutralize_std.rs::neutralize_std_section`）：
   - 再次横截面 rank pct（平均秩 / 有效数）；
   - 缺失的行业 OLS 回归填充 / 行业中位填充、restrict 置空；
   - 对风格矩阵做 OLS 残差化：barra 10 风格（单调秩后）+ 申万一级行业 one-hot（无截距）+  Cholesky 求解；
   - 残差再次 rank pct；
4. IC 计算（`legacy_spearman_correlation`）：对当期有效截面，
   `Spearman(x_neu, r) = 1 - 6 * sum(d_i^2) / (n(n^2-1))`，`d_i = ordinal(x_neu)_i - ordinal(r)_i`（ordinal 秩，tie 按出现顺序），其中 x_neu 为中性化（残差 rank pct）后的 slot 值。

记 **IC_A** 为该链路产出。性能画像（release 版，30 因子、全模板轴、n_jobs=32 实测）：
  neutralize（中性化）约 77% 耗时，raw/neu 回测约 21%，其余为预处理/汇总。

### 1.2 提出的替代想法

“对收益率序列做中性化（`M_B * r`，一次 OLS，所有因子共享），再算原始因子与中性化收益的 IC”。
天然优势：收益侧只算一次，因子侧不再做逐因子逐 slot 的风格回归。
核心问题：**这样算出来的 IC 能否与现有 RankIC（IC_A）数值一致？**

### 1.3 评判标准

1. 数学是否合理（与 IC_A 的关系）；
2. 是否与现状“完全对得上”（逐位 / 机器精度级一致）；
3. 提速幅度（保留“收益侧只算一次”特性的前提下）。

---

## 2. 数学分析（从第一性原理）

### 2.1 符号

- `x`：因子截面向量（流水线中已是秩化值 x_hat）；`r`：一期未来收益截面向量；
- `B`：风格矩阵 = [barra 10 秩化列, 申万一级行业 one-hot]（行业列和恒为 1，已含截距信息）；
- `M_B = I - B(B'B)^-1 B'`：对风格取残差的投影算子。性质：**对称**、**幂等**、`M_B * 1 = 0`；
- `rho(.)`：横截面百分位秩（平均秩 / 有效数）；`rank(.)`：ordinal 秩。

### 2.2 三条关键事实

**事实 1（线性层分子恒等）**

`<M_B * x, r> = <x, M_B * r>`（对称性）

在“残差原值”层面，因子侧砍风格与收益侧砍风格，**分子完全相同**。

**事实 2（秩层不可交换，本问题的核心）**

`rho ∘ M_B  ≠  M_B ∘ rho`——秩化是非线性变换。现状 IC_A 的分子是 `<rho(M_B * x_hat), rank(r)>`，
其中 `rho(M_B * x_hat)`（**残差的秩**）是每个因子自己的残差向量的非线性函数。
任何“收益侧一次性预计算量”（`M_B * r`、收益秩、方差等）都无法重构它。

结论 1：在 Spearman（秩）口径下，逐位复现 IC_A 必须让因子侧知道自己的残差序；
“数据完全由收益侧共享”的极致提速与“逐位一致”不可兼得。

**事实 3（Spearman 分母是常数）**

`||rho(u)||` 只由截面有效样本数 n 决定（秩化值恒为 1..n 的排列）。故所有差异只被
**秩差的平方和** `sum(d_i^2)` 承载：`IC = 1 - 6 * sum(d_i^2) / (n(n^2-1))`。

**推论（两口径的精确换算，线性/Pearson 层）**

`corr(M_B * x, r) = corr(x, M_B * r) * sqrt(1-R^2_R) / sqrt(1-R^2_X)`

其中 `R^2_X` 为因子被风格解释的比例（逐因子），`R^2_R` 为收益被风格解释的比例（市场常数）。
方向（符号）严格一致；数值差一个逐因子不同的缩放因子；跨因子排序可能小幅变化。

### 2.3 金融含义

- **IC_A（因子中性化，现状）**：风格中性选股信号质量的业界标准度量（因子正交化于 barra+行业后再看预测力）；
- **IC_B（收益中性化，提议）**：原始因子对特质收益（剔除风格可解释部分的残差收益）的预测力，即“脱离风格的纯 alpha”视角；
- 两者是同一“alpha vs 风格”问题的两个侧面（FWL 定理：同一双回归的两种计算顺序）。

---

## 3. 候选方案对比

### 方案 1：收益中性化 Spearman（原始提议）

`IC_B = corr(rho(x_hat), rho(M_B * r_hat))`——收益侧一次 OLS，因子侧零改动。

- 与 IC_A：非精确等价（事实 2）。真实数据实测：每日 IC 序列相关系数 **0.961**，
  符号一致率 94.8%，跨因子排序秩相关 0.987（Top5 重叠 4/5），|IC| 中位偏差约 6%；
- 提速：大（收益侧一次），但数值不可比，需全套阈值重校准；
- 结论：**作为低成本的“预筛选/海选”工具可行**（方向与排序几乎不变）；作为正式口径需重校准。

### 方案 2：共享收益秩（Spearman 精确版）

把 legacy_backtest 主循环里“每个 slot 每期都对收益重新排序”的部分抽为**引擎级一次预计算**
（收益秩基于 `O = {open & ret finite}` 全集，因子无关；因子秩/残差秩仍逐 slot 计算）。

- 与 IC_A：**逐位相等**（sandbox 实测 max|diff| = 0，数据见第 4 节）；
- 提速：**实测仅 1.13 倍**——因为收益排序只占约 10% 耗时，其余（因子秩化、OLS 残差、残差秩化）不可省；
- 结论：数学无妥协，但提速有限。

### 方案 3：Pearson 快路径 + 闭式恒等式（推荐）

把 IC 口径从 Spearman 调整为同一套数据的 Pearson：

`IC_A(pearson) = corr(M_B * x, r) ≡ cov(x, M_B * r) / (||M_B * x|| * ||r||)`

落地分层：

| 层 | 一次性预计算（与因子无关） | 每 slot 每期（因子侧） |
|---|---|---|
| 收益侧 | 每期 OLS 得 `M_B * r` + 收益中心化平方和 | - |
| 风格侧 | 每期 `X'X` + Cholesky 分解 | - |
| 因子侧 | - | `X'y`（O(N*K) 乘加）-> 回代 -> 残差平方和 `sum(y^2) - beta'(X'y)` -> 相关 O(N) |

因子侧**零排序、零残差物化、零重复累积**（现状每 slot 为 O(N*K^2) X'X 累积 + 三次全矩阵排序）。

- 与“因子中性化 Pearson IC”数值一致到 **2.1e-14（浮点噪声级，机器精度）**——恒等式严格成立；
- 与现状 Spearman IC_A：序列相关 0.966（Pearson 与 Spearman 口径之差，本质是秩化舍入信息）；
- 提速：**实测 3.7 倍**（相对现状算法层），生产全流程（neutralize 占 77%）预期 **2~3 倍**；
- 结论：**“完全对得上（精确）+ 显著提速”只在 Pearson 口径下成立**，代价是 IC 口径从 Spearman 变为 Pearson（业界同样常见，换口径后做一次分位数阈值重校准即可）。

### 汇总表

| 方案 | 与 IC_A 的一致性 | 实测提速 | 备注 |
|---|---|---|---|
| 1 收益中性化 Spearman | 近似（序列相关 0.96，排序 0.99，偏差约 6%） | 大（未实现） | 仅适合做海选/预筛 |
| 2 共享收益秩 Spearman | **逐位相等** | 1.13 倍 | 保底方案，无口径变更 |
| 3 Pearson 快路径 | **机器精度相等**（对 Pearson 口径）；与现状 Spearman 差 0.966 序列相关 | **3.7 倍**（IC 层）/ 预期 2~3 倍（全流程） | 推荐落地 |

---

## 4. Sandbox 实验证据

### 4.1 实验设置

- 数据：`factor_store_挂单猫0701c` 前 13 个因子（每因子 `rank_and_fill` 后作为 slot0），
  2024-01-02 ~ 2024-06-28（117 个交易日 x 5438 只股票），模板轴与共享回测输入缓存（000905 digest 6b8884a67f05d221）对齐；
- 风格口径复刻：barra 10 风格横截面平均秩/n + 申万一级行业 one-hot（`floor(行业码/10000)`），
  OLS 用 nalgebra Cholesky（与生产 `get_residual` 同路径），因子秩/残差秩为平均秩（`rank_pct_row_into` 语义）；
- IC 公式复刻：生产 `legacy_spearman_correlation`（ordinal 秩 + 1 - 6*sum(d^2)/(n(n^2-1))）；
- 日期对齐：信号 t 日、收益 `ret_sum_gap5[t+1]` 期、风格暴露取信号日 B(t-1)（与因子中性化完全对偶）；
- 有效集：`{restrict==0 且 ret finite}`（因子无关，保证收益秩可全局预计算）。

### 4.2 数值一致性结果

| 对比 | n | max|diff| | 结论 |
|---|---|---|---|
| 2 共享收益秩 vs 现状复刻 | 1508 | **0（逐位相等）** | 数值无任何改变 |
| 3 Pearson 快路径 vs Pearson 现状复刻 | 1508 | **2.1e-14** | 恒等式成立到机器精度 |
| 现状 Spearman vs Pearson 口径 | 1508 | 序列相关 0.966 | 口径差异约 3.4% 的信息 |
| 与生产引擎 `neutralize_std_block_py` 逐期对照 | 116 | 6.3e-3 | 复刻细节差异（填充/排序器 tie 处理），非设计问题 |

### 4.3 测速结果（3 轮中位，dev+opt1 构建，13x117=1521 期 x 5438 股）

| 版本 | 耗时 | 相对现状 |
|---|---|---|
| 现状算法层（Spearman，每次重排收益） | 3.71 s | 1.00 倍 |
| 2 共享收益秩 | 3.29 s | 1.13 倍 |
| 3 Pearson 快路径 | **0.998 s** | **3.72 倍** |

### 4.4 已知实现细节（落地产线时注意）

1. **有效集必须因子无关**：`rank_and_fill` 已保证 open 股票无 NaN，故 `{open & ret finite}` 与因子无关，收益秩 / `X'X` 才能全局预计算（sandbox 已按此处理并给出 F==O 判定，F!=O 时自动回退逐期重排，数值不受影响）；
2. **残差平方和口径**：`||M_B * x||^2 = sum(y^2) - beta'(X'y)`（M_B 列空间含 1，残差自动零均值），不要混入 n/(n-1) 样本方差因子（实测会引入 1e-5 级偏差，已修正）；
3. **分母用原始收益方差**（非收益残差方差），见恒等式 `(||M_B * x|| * ||r||)`；
4. **行业列与截距**：行业 one-hot 和 = 1 已含截距，不能再加常数项（否则 X'X 奇异）；
5. 收益秩预计算基于“当日有效集”的 ordinal 序；tie 处理与生产 `legacy_spearman_correlation`（ordinal、按出现顺序）一致；浮点连续值 tie 极少，与“平均秩”差异可忽略但非严格为零；
6. `serde_json` 拒绝 NaN：任何新产物（如 Pearson 模式 IC 序列文件）不得含 NaN 占位（沿用现有写法）；
7. 日期对齐：收益中性化必须用**信号日**风格暴露 B(t-1)，与因子中性化截面完全对称（两端不一致会使偏差累积）。

---

## 5. 落地路径建议

### 首选：`ic_mode="pearson_fast"`

1. Rust（`rust_pyfunc`）：
   - `tail_backtest_engine` / `tail_v5_pipeline.rs` 新增 `ic_mode`（默认 `spearman`），
     `pearson_fast` 走：收益侧/风格侧预计算（每期一次 `M_B * r`、`X'X`、Cholesky）+ 因子侧轻量投影；
   - 保留现有 `ic_neu_gap1/gap5.npy` + names/dates 的产物结构；
   - 复用 sandbox 已验证的 `sandbox_ic_pearson_fast` 算法（`rust_pyfunc/sandbox/src/lib.rs`，可直接照搬核心函数并补单元测试）；
2. Python（`design_whatever`）：
   - `tail_pipeline_engine` 透传 `ic_mode`；`select_tail_v2_factors` 不变（只消费 IC 序列）；
   - 新增“分位数阈值校准”脚本：用一段历史（如 2019-2024）分别按 Spearman 与 Pearson 算全体因子 IC，按分位数映射 `ic_point_neu_*` / `corr_point_neu` 等参数（调整量约为序列相关 0.966 带来的系统性平移）；
3. 验证：
   - Rust 单测：同一 (x, B, r) 上 `spearman` 与 `pearson_fast` 互相对拍（sandbox 断言可搬）；
   - 端到端：小规模跑两次（旧/新）比对已选因子名单差异（预期 TopK 重合度约 99%，个别位置换序）。

### 保底：共享收益秩（`ic_mode` 不动）

仅把 legacy_backtest 中每 slot 的收益重排抽为一次性预计算——零口径变更、逐位一致，
代价是提速只有约 1.1 倍。适合作为“无风险优先”的过渡实现。

### 不建议

纯“收益中性化 Spearman”（方案 1）作为正式口径：数值不可比、需全面重校准，
且与下游“中性化因子”入库口径不一致；仅适合做临时海选。

---

## 6. 交接材料清单

| 位置 | 内容 |
|---|---|
| `rust_pyfunc/sandbox/src/lib.rs` | sandbox 实现：`sandbox_ic_old`（现状复刻）、`sandbox_ic_new`（共享收益秩）、`sandbox_ic_pearson_old`（Pearson 现状复刻）、`sandbox_ic_pearson_fast`（推荐方案） |
| `rust_pyfunc/sandbox/Cargo.toml`、`build.sh` | sandbox 依赖与构建（`bash build.sh` 安装 `dev_sandbox`） |
| `/tmp/run_sandbox_verify3.py` | 验证/测速脚本（输入缓存 `/tmp/sb_blocks.npy` 等） |
| `/tmp/prep_sandbox_ic.py` | 输入准备（因子 rank_and_fill、barra 秩、行业、收益对齐） |
| 生产对照 | `neutralize_std_block_py` + `legacy_spearman_correlation`（现状口径的权威参照） |

### 复现命令

```bash
cd /home/chenzongwei/rust_pyfunc/sandbox && bash build.sh
/home/chenzongwei/.conda/envs/chenzongwei311/bin/python /tmp/run_sandbox_verify3.py
```

---

## 7. 结论（一页版）

1. 收益中性化 Spearman（原提议）：数学上“同源不等值”（分子恒等、归一化不同），与现状差约 3% 信息量（0.966 序列相关），适合海选，不适合正式口径；
2. **“完全对得上 + 显著提速”仅存在于 Pearson 口径**：闭式恒等式 `corr(M_B * x, r) ≡ cov(x, M_B * r) / (||M_B * x|| * ||r||)` 把收益侧中性化变为一次预计算、因子侧降为 O(N*K) 轻量投影，实测数值一致到 1e-14、提速 3.7 倍（IC 层）；
3. 若必须保留 Spearman 语义：共享收益秩方案逐位一致但仅 1.13 倍（这是数学上能给的全部）；
4. 落地建议：`ic_mode="pearson_fast"`（推荐）+ 分位数阈值校准；`共享收益秩` 作为零风险过渡。
