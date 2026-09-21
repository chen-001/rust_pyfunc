# MPROB（概率版双边单调度）实现规格 —— 唯一权威定义

工作笔记，不是交付物。所有实现与校验都以本文件为准，有疑问先问 lead。

## 0. 背景

上一轮给回测引擎加了 `SSM`（双边单调度，5 段版）：10 组组均收益曲线归一化后的「净位移 ÷ 段内台阶幅度之和」，
取五段最小，值域 [-1,1]，只当门槛、不当排序键，落盘成 summary parquet 的一列。
本轮加它的「概率版」`MPROB`：不再用样本均值比较的符号，而是对 45 个组对各自算一条价差序列的
Newey-West t 统计量，转成正态概率，再等权平均。目的：让「10bp ± 1bp」比「0.01bp ± 5bp」拿到更高置信度。

## 1. Rust 函数（放 `src/tail_v5_pipeline.rs`，紧挨 `compute_ssm`）

```rust
pub(crate) fn compute_mprob(group_returns: &[Vec<f64>], portf_num: usize) -> f64
```

输入 `group_returns[d][t]`：第 d 组（d=0..9 对应第 1..10 组）的逐日收益，`t` 是回测窗口内的交易日。
返回标量，值域 [-1,1]；无法计算时返回 `f64::NAN`。

步骤必须逐条照做（顺序也照做）：

1. `portf_num != 10` 或 `group_returns.len() != 10` → `NaN`
2. `n = group_returns[0].len()`；`n < 2`，或任一列长度 ≠ n → `NaN`
3. 任一元素非有限（NaN/±Inf）→ `NaN`
4. 时间均值 `r[d] = group_returns[d]` 的算术平均
5. 定向（与 `compute_ssm` 同一套规则）：`r[9] < r[0]` 时整条倒序，即 `idx[d] = 9 - d`；否则 `idx[d] = d`
6. 滞后阶数 `L = min(n - 1, floor(4 * (n / 100)^(2/9)))`（Newey-West 1994 自动选阶；n=2424 时 L=8）
7. 对 45 个组对 `i < j`（i、j 是 0 基下标，共 45 对，等权）：
   - `d_t = group_returns[idx[j]][t] - group_returns[idx[i]][t]`，`t = 0..n-1`
   - `mean = Σ d_t / n`；`e_t = d_t - mean`
   - `gamma_0 = Σ e_t² / n`
   - `gamma_k = Σ_{t=k}^{n-1} e_t · e_{t-k} / n`（k = 1..L）
   - `nw_var = (gamma_0 + 2 * Σ_{k=1..L} (1 - k/(L+1)) * gamma_k) / n`
   - `nw_var` 非有限或 `<= 0` → 该对贡献 `0.0`
   - 否则 `t_stat = mean / sqrt(nw_var)`，该对贡献 `2 * Φ(t_stat) - 1`
8. `MPROB = Σ(45 个贡献) / 45.0`

`Φ` 是标准正态分布函数：`Φ(x) = 0.5 * (1 + erf(x / sqrt(2)))`。
erf 用 A&S 7.1.26 五系数近似（系数与 `src/copula.rs::erf` 完全相同，可直接照抄那段）：
最大绝对误差 1.5e-7，因此 MPROB 的绝对误差 ≤ 7.5e-8。这一点要写进文档注释。

### 为什么定向规则要和 SSM 一样

`MPROB` 对整条曲线取负会严格反号（`d_t` 全变号 → `t` 全变号 → `Φ(-t) = 1-Φ(t)` → `2Φ(t)-1` 反号），
所以「整条倒序」等价于乘一个符号。用与 SSM 相同的端点规则定向后，MPROB 就是方向无关的形状分，
和 SSM 在同一个约定下可比。**不要**改用别的定向规则。

### 空值参考分布（默认门槛的依据）

Python 独立模拟（20000 次，10 组 iid 标准正态，T=2424，同一套定义）：

| 分位 | 5% | 25% | 50% | 75% | 90% | 95% | 99% |
|---|---|---|---|---|---|---|---|
| MPROB | -0.1269 | -0.0005 | 0.0903 | 0.1845 | 0.2675 | **0.3149** | 0.4009 |

sd = 0.1343；T=1212 时 95% 分位 0.3141，基本不随 T 变。所以 Python 侧默认门槛取 **0.315**。

## 2. Rust 接线（改动点清单）

| 位置 | 改法 |
|---|---|
| `src/tail_v5_pipeline.rs` `LegacyBacktestResult.summary` | `[f64; 11]` → `[f64; 12]`，下标 11 = MPROB，下标 10 = SSM 不动 |
| 同文件 `default_legacy_backtest_result()` | `[f64::NAN; 11]` → `[f64::NAN; 12]` |
| 同文件两处构造 summary 的数组字面量（`legacy_backtest_single_factor_with_effective` 与 `..._opt`） | 末尾追加 `compute_mprob(&group_returns, portf_num),` |
| `src/tail_v8_backtest.rs` `BtAcc::finish` | 同上，末尾追加 |
| `SummaryRowRecord` | 在 `ssm` 之后加 `#[serde(rename = "MPROB", default = "default_nan_f64")] pub(crate) mprob: f64,` |
| `write_summary_parquet` | schema 末尾加 `Field::new("MPROB", DataType::Float64, true)`，列数组末尾加 `f64_col(|r| r.mprob)`，并更新函数上方的列序文档表 |
| `src/tail_v8_backtest.rs` selfcheck 逐位比较 | `for i in 0..11` → `for i in 0..12`（注释同步） |
| 单元测试 | 新增 `test_compute_mprob_definition`，见下 |

**不要**动 `SSM` 的任何现有逻辑、下标、门槛。**不要**动 `SUMMARY_COLUMNS`（那是 Python 侧的事）。

### 单元测试用例

用 `assert!((got - want).abs() < 1e-9)` 这类容差断言（不要用 `assert_eq!` 比浮点，上一轮 SSM 就栽在这）。

1. 完美阶梯：`[0,8,16,...,72]`（单日）→ 1.0
2. 反向阶梯：`[72,...,0]` → 1.0（定向后仍 1.0）
3. 全平：`[5.0; 10]`（多日）→ 0.0
4. `portf_num = 5` → NaN；空输入 → NaN；10 组但无日期 → NaN；含一个 NaN → NaN
5. 至少一条「真实感」用例：用 Python 参考实现（见第 4 节）算出的硬编码期望值，容差 1e-9

## 3. Python 接线（design_whatever）

与 `use_ssm` 完全对称，参数名 `use_mprob`，列名 `MPROB`，门槛参数 `mprob_point_neu_gap5 / mprob_point_neu_gap1`，
默认 **0.315**（见第 1 节）。`use_mprob=True` 时 MPROB **只当门槛、不当排序键**，排序仍是中性化 `|IC_mean|` 降序，
与 `use_ssm=True` 的行为结构一模一样；原始（raw）通道不受影响。

改动点：

- `tail_v2_screen.py::select_tail_v2_factors`：签名加三个参数；在 `_filter_cover` 之后、`_select_by_metric` 之前
  加与 `use_ssm` 并行的门槛块；`_gap_channel_sets` 里加 `if cfg.get("use_mprob")` 的池子过滤；
  docstring 补一段（照 `use_ssm` 那段写）。
- `tail_v4.py`：`SUMMARY_COLUMNS` 加 `"MPROB"`；`run_tail_pipeline_engine` 加 `use_mprob: bool = False`，
  config 处理照 `use_ssm` 那段（True 写进 selection_config，False 时把三个新键 pop 掉，保证既有 cache 零副作用）；
  `ic_only` 条件末尾加 `and not use_mprob`；`run_tail_pipeline_v5` 里也 pop 掉这三个新键；
  把「Rust 侧写的是 16 列」的注释更新成 17 列。
- `tail_v4_standalone.py`：同上对应位置。
- `tail_whatever.py`：`tail_pipeline_engine` wrapper 加 `use_mprob: bool = False` 并透传；docstring 更新。
- `tail_v3.py`：`SUMMARY_COLUMNS` 加 `"MPROB"`（与 SSM 对称）。
- `supplement_evaluation.py`：把 `_SSM_SELECTION_KEYS` 扩成同时含 mprob 的三个键（可改名），注释同步。

## 4. 校验要求

**数学层（独立于 Rust）**：Python 参考实现必须另写一份（`math.erf` 版），逐条照第 1 节，
与 Rust 在相同输入上对比。因为 Rust 用 A&S 近似，两者容差 1e-6；把 Rust 的 A&S erf 也照抄进 Python
参考后，两者应完全一致（1e-15 级）。

**集成层**：构建后跑一次真实 pipeline（小因子子集 + 短窗口），确认
`summary_neu_gap5_candidates.parquet` 有 `MPROB` 列、非空、落在 [-1,1]，
并与 Python 参考实现（用引擎自己的分组收益复算）逐位/近位一致。

**归零校验**：`use_mprob=False`（不传参数）时，入选名单必须与改动前逐位一致。
