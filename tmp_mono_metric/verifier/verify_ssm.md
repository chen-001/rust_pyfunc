# SSM 独立校验报告

校验对象：`src/tail_v5_pipeline.rs::compute_ssm`（双边单调度 SSM，五段取最小），
调用点 `legacy_backtest_single_factor_with_effective` / `..._effective_opt` /
`tail_v8_backtest.rs::BtAcc::finish`，落盘 `summary_from_row` → `SummaryRowRecord.ssm` → parquet 列 `SSM`。

**一句话结论**：`compute_ssm` 的实现与给定定义逐条一致，没有下标错位、方向条件错误或兜底逻辑错误；
但新加的单元测试 `test_compute_ssm_definition` 里有一条**硬编码期望值是错的**（精确相等断言必失败）。
端到端数值校验已完成：中性化 26/26 组合 SSM 与 parquet **逐位相同**。

---

## 1. 代码审查

### 1.1 与定义的一致性：一致

逐行核对 `src/tail_v5_pipeline.rs:964-1001`：

| 定义要求 | 代码 | 判定 |
|---|---|---|
| 10 组、每组逐日收益 → 时间平均得 r[0..10] | `r[d] = col.iter().sum() / n` | 一致 |
| `r[9] < r[0]` 则整条倒序 | `if r[9] < r[0] { r.reverse(); }` | 一致（注意是**端点比较**，不是整体斜率） |
| seg(1,10) | `r[0..10]`，9 个台阶 | 一致 |
| seg(1,5) | `r[0..5]` | 一致 |
| seg(6,10) | `r[5..10]` | 一致 |
| seg(1,4) | `r[0..4]`，3 个台阶 | 一致 |
| seg(7,10) | `r[6..10]` | 一致 |
| 净位移 ÷ 段内台阶幅度之和 | `(s[last]-s[0]) / Σ|s[k]-s[k-1]|` | 一致 |
| 取五段最小 | `ssm = seg(1,10)` 后逐个 `if v < ssm` | 一致 |

五段下标经我按 1 起闭区间语义单独推演并复算（见第 2 节），与描述完全对应，**没有错位**。

### 1.2 兜底与 NaN

- `portf_num != 10`、`group_returns.len() != 10`、无日期、各列长度不齐 → `NaN`。
- 任一 `r[d]` 非有限 → `NaN`（在 `reverse` 之前检查，正确）。
- 分母为 0 → 返回 `0.0`（`den <= 0.0`）。`den` 是绝对值之和，恒 ≥0 且不会 NaN（前面已保证 r 全有限），
  所以这个分支只可能是 `den == 0`，即整段全平。定义里 0/0 未规定；取 0.0 的后果是
  **任何一段全平就把 SSM 压到 ≤0**（低于 0.098 及格线），与「全平 → 0.0」的测试期望一致，属合理选择，不是缺陷。
- 调用路径上 `group_returns` 的元素只会是 0.0 或有限值之和/个数，因此 `NaN` 只会在 `ic_only`（组收益为空）时出现，
  与两个调用点里 `ic_only` 分支硬写 `f64::NAN` 一致。

### 1.3 发现的问题

**(1) 真实缺陷：新增测试里一条期望值是错的（测试本身必失败，不是实现错）**

`src/tail_v5_pipeline.rs:4412-4418`：

```rust
assert_eq!(
    compute_ssm(&one_day(&[-200.0,-160.0,-120.0,-80.0,-40.0,0.0,0.8,1.6,2.4,3.2]), 10),
    1.0
);
```

我把 `compute_ssm` 原样抄成独立文件用 rustc 1.91.1 编译执行（`-O` 与 debug 两种都试），实际值是
`0.9999999999999997`（bits `0x3feffffffffffffd`），`assert_eq!` 精确相等 → **false**。
原因是 seg(1,10) 的分母是 `40+40+40+40+40+0+0.8+1.6+0.8+0.8` 的浮点累加（=203.20000000000005），
而分子是 203.2，比值差 1 ULP。实现符合定义，是测试的期望值不该用精确相等。
修法：改成 `assert!((got - 1.0).abs() < 1e-12)`（同文件其余三条断言本来就是这么写的）。
其余三条用 `assert_eq!(..., 1.0)` 的用例（等距阶梯、反向阶梯、多日）实测恰好是精确 1.0，不会失败。

**(2) 注释与代码措辞不一致（不影响结果）**

文档注释写「整体斜率为负时整条倒序」，代码实际只比端点 `r[9] < r[0]`。
按任务给的定义代码是对的，注释措辞偏松，建议改为「末端组低于首端组时」。

**(3) 覆盖缺口（不影响正确性）**

`src/tail_v8_backtest.rs:576` 的自检只比 `summary[0..10]`（即下标 0-9），新增的第 11 个元素（下标 10）
不在 BtAcc 与生产路径的逐位对比范围内。建议把这个循环扩到 `0..11`。
另：`src/tail_v4_pipeline.rs` 有自己的 `LegacyBacktestResult`，`[f64;10] → [f64;11]` 不影响它。

**(4) 一条已验证的性质（非问题）**

五段集合在「整条倒序」下是闭合的（(1,5)↔(6,10)、(1,4)↔(7,10)），所以 SSM 对整条曲线取负完全不变，
方向统一条件只是约定，不会造成多空方向的系统性偏置。实测 `[0,8,...,72]` 与 `[72,...,8,0]` 都是 1.0。

---

## 2. 独立复现（Python，未参考任何现成脚本）

命令：`/home/chenzongwei/.conda/envs/chenzongwei311/bin/python /tmp/ssm_verify/py_ssm.py`

| 输入（10 组组均收益） | 我的 Python | 测试硬编码期望 | 对照 |
|---|---|---|---|
| `[0,8,16,24,32,40,48,56,64,72]` | `1.0` | `1.0`（精确） | 一致 |
| `[72,64,56,48,40,32,24,16,8,0]` | `1.0` | `1.0`（精确） | 一致 |
| `[-45,-35,-25,-15,-5,1,3,5,7,4]` | `0.14285714285714285` | `0.142857142857143`（tol 1e-12） | 一致（Δ1.4e-16） |
| `[-45,-35,-25,-15,-5,-9,1,3,5,7]` | `0.8666666666666667` | `0.866666666666667`（tol 1e-12） | 一致（Δ3.3e-16） |
| `[-40.9,-10.9,3.4,15.9,23.1,28.3,32.5,34.7,36.2,33.3]` | `0.12121212121212062` | `0.121212121212121`（tol 1e-12） | 一致（Δ3.7e-16） |
| `[-200,-160,-120,-80,-40,0,0.8,1.6,2.4,3.2]` | `0.9999999999999997` | `1.0`（**精确相等**） | **不一致 → 测试会失败** |
| `[5]*10` | `0.0` | `0.0`（精确） | 一致 |

补充用例：`portf_num=5` → NaN；空输入 → NaN；10 组但无日期 → NaN；多日取时间平均 → 1.0。均与测试期望一致。

逐段明细（供核对五段下标）：`[0,8,...,72]` 五段全是 1.0；`[-45,...,7,4]` 五段为
seg(1,10)=0.8909090909090909、seg(1,5)=1.0、seg(6,10)=0.3333333333333333、seg(1,4)=1.0、seg(7,10)=0.14285714285714285，最小取 seg(7,10)。

Rust 侧同一批输入的实测值（把 `compute_ssm` 逐字抄出、`rustc -O` 编译运行）：

```
stair-up    bits=0x3ff0000000000000  1.0
stair-down  bits=0x3ff0000000000000  1.0
g10-3rd     bits=0x3fc2492492492492  0.14285714285714285
break6      bits=0x3febbbbbbbbbbbbc  0.8666666666666667
real        bits=0x3fbf07c1f07c1edd  0.12121212121212062
shortx50    bits=0x3feffffffffffffd  0.9999999999999997   <- assert_eq 1.0 失败
flat        bits=0x0000000000000000  0.0
assert_eq shortx50 == 1.0 -> false
```

注意：`cargo test --lib` 被 13 个老模块的编译错误挡住，所以**没有**用「测试通过」当证据，
上面的期望值对照与 Rust 实测值都是人工/独立编译得到的。

---

## 3. 端到端数值校验（已完成，逐位相同）

**没有走 `do_fulltest_part=True`**（那条路我实测没跑出来：2 个因子的短跑因为回测输入缓存键包含因子集合、
换了因子名就要从头重算 ret/restrict，10 分钟卡在输入准备阶段）。改用更直接也更严格的办法：

用**引擎自己的输入与自己的预处理函数**在 Python 里复现引擎的分组收益，再和 parquet 的 `SSM` 列逐位比。

复现链路（每一步都用引擎的产物或引擎的 Rust 函数，不自己猜口径）：

1. 因子原值：引擎导出的 raw parquet `/hdd/user_home_unsafe/chenzongwei/factor_data/ssm_smoke_on/<源因子>.parquet`
2. 轴：`/tmp/ssm_smoke_on_tail_v4/meta/dates.npy`、`stocks.npy`
3. 收益/限制/指数：引擎指纹里的
   `/home/chenzongwei/pythoncode/_tail_v2_shared/backtest_inputs/000905_20230103_20231229_7786_69b4fe0d2dbbe32b/{ret_gap1,ret_gap5,ret_sum_gap1,ret_sum_gap5,restrict,index_ret}.npy`
4. 回测前预处理（rank + 缺失名次中位数填充 + rolling）：`rp.tail_v5_rank_fill_roll_block_f32`
5. 标准中性化：`rp.neutralize_std_block_py`（行业码矩阵用 `design_whatever.tail_v4._load_industry_matrix` 同一函数加载）
6. 分组收益规则按 Rust 源码重写（`local_t % gap == 0` 换 held 信号行、`restrict[held]==0`、
   平均秩、`floor(rank/n*10)` 分桶、空桶 0.0、`stocks_num < portf_num` 留 0.0），
   再用我的 Python `compute_ssm` 算 SSM。

**交叉验证（证明我复现出的分组收益就是引擎的）**：同一批分组收益还能算出
`annualized_return`、`hedge_annualized_return`、`ratio_mean` 三个 parquet 列，
它们与 parquet 的最大偏差 ≤ **1.7e-16**（浮点噪声级），说明分组收益逐位一致。

### 结果

主校验（两个源因子 × 13 个派生面 × gap5 × 中性化 = 26 组）：
`/tmp/ssm_verify/e2e_neu2.py`

```
比对 26 个 (factor, gap5, neu) 组合
SSM                最大绝对偏差 = 0.000e+00   逐位相同 26/26
annualized_return  最大绝对偏差 = 1.665e-16
hedge_ann_return   最大绝对偏差 = 1.943e-16
ratio_mean         最大绝对偏差 = 5.551e-16
```

用户给的两个因子逐位复现：

- `microcapm_price_log_return_3s_capm_residual_zscore_mean_mean_mean_smooth_10`：
  Python SSM = `0.9640469439390147`，parquet = `0.9640469439390147`，偏差 `0`（用户报 0.964047 ✓；IC_mean 实测 -0.039498 ✓）
- `microcapm_price_log_return_3s_capm_residual_zscore_mean_mean_max_smooth_10`：
  Python SSM = `-0.013622452670538616`，parquet 同值，偏差 `0`（用户报 -0.013622 ✓；IC_mean 实测 -0.040988 ✓）

更宽样本（10 个源因子 × 13 个派生面 × gap1/gap5 × rolled/neu，`/tmp/ssm_verify/e2e_all.py`，
跑到 7/10 个源因子、347 条时收尾，结果已足够）：

```
rolled gap1: n= 78 |ΔSSM|max=0.000e+00 逐位 78/78
rolled gap5: n= 87 |ΔSSM|max=0.000e+00 逐位 87/87
neu    gap1: n= 91 |ΔSSM|max=2.227e-03 逐位 86/91
neu    gap5: n= 91 |ΔSSM|max=2.302e-05 逐位 90/91
```

（raw/rolled 阶段 165/165 逐位相同；neu 阶段 177/182 逐位相同。）

### 5 个 neu 组合残留偏差的来源（不是 SSM 的问题）

证据链：

1. 这几个组合的 `annualized_return`、`hedge` 偏差仍在 1e-16，说明我的分组收益与引擎只在**中间桶**构成上不同。
2. **IC 探针**（`/tmp/ssm_verify/ic_probe.py`，用引擎逐字定义的 Spearman）：
   rolled 阶段我的 IC_mean 与 parquet 偏差 ≤5e-17（完全一致），
   neu 阶段**每一个**组合都差 ~1e-9 —— IC 只依赖信号的**序**，这直接证明我拿到的中性化矩阵与引擎的
   在少数位置次序不同。
3. 原因是中性化实现有两条：引擎 v8 融合路径走 `tail_v8_neu_v3::v3_slots_range`（v3），
   Python 侧只能调到 `rp.neutralize_std_block_py`（v2/section）。代码注释声称两者逐位一致，实测不然。
4. 敏感度实验（`/tmp/ssm_verify/sens_neu.py`）：这些 neu 因子的组均收益只有 ~1e-4 量级，
   **单个股票换一个桶**就能让 SSM 动 5.8e-06 ~ 1.1e-03；5 个残留组合的偏差量级完全被这个机制解释。
   同时给中性化矩阵加 1e-8 相对噪声（8 个种子）SSM 一个都不变（噪声不改变序），
   排除了「SSM 对浮点噪声不稳定」这一解释。

结论：残留偏差发生在**中性化矩阵本身**（v2 与 v3 两条实现的差异），与 `compute_ssm` 无关；
在能证明输入逐位一致的 rolled 阶段，SSM 是 165/165 逐位相同。

---

## 复现命令清单

```bash
# 代码审查：独立编译运行 compute_ssm（原样抄写）
cd /tmp/ssm_verify/rs && ~/.cargo/bin/rustc -O -o ssm_opt ssm.rs && ./ssm_opt

# 第 2 部分：7 组输入的 Python 复算
/home/chenzongwei/.conda/envs/chenzongwei311/bin/python /tmp/ssm_verify/py_ssm.py
/home/chenzongwei/.conda/envs/chenzongwei311/bin/python /tmp/ssm_verify/prec.py

# 第 3 部分：端到端（主校验 26 组）
/home/chenzongwei/.conda/envs/chenzongwei311/bin/python -u /tmp/ssm_verify/e2e_neu2.py
# 宽样本 347 组（约 10 分钟）
/home/chenzongwei/.conda/envs/chenzongwei311/bin/python -u /tmp/ssm_verify/e2e_all.py
# IC 探针（证明 neu 矩阵次序有差）
/home/chenzongwei/.conda/envs/chenzongwei311/bin/python -u /tmp/ssm_verify/ic_probe.py
# 单股换桶敏感度
/home/chenzongwei/.conda/envs/chenzongwei311/bin/python -u /tmp/ssm_verify/sens_neu.py
```

脚本与中间结果：`/tmp/ssm_verify/`（`py_ssm.py`、`prec.py`、`rs/ssm.rs`、`e2e_rolled.py`、
`e2e_neu.py`、`e2e_neu2.py`、`e2e_all.py`、`e2e_all.csv`、`ic_probe.py`、`sens_neu.py`、`diag_neu.py`）。
