# MPROB 独立校验报告（mprob-verifier / task-3）

**一句话结论**：`compute_mprob` 与 SPEC 第 1 节的定义在 199 个独立用例上**逐位一致**（含 NaN 判定），SPEC 第 2 节的 9 处接线全部到位；端到端用引擎自己的产物复现分组收益后，**无中性化的 rolled 阶段 72/72 逐位相同**，neu 阶段 78 组里 75 组逐位相同，剩余 3 组的偏差已定位到「v2/v3 两条中性化实现的次序差异让少数股票换桶」，与 MPROB 本身无关。SPEC 自身有两条疑点（第 2 节用例 1/2 的期望值、第 1 节的误差上界）。

被校验版本：`src/tail_v5_pipeline.rs` md5 `5397660e6207f98a3ef5ba09c8e77416`、`src/tail_v8_backtest.rs` md5 `d738669ee3d6bc4b9be7bcaf3b87ebd0`（校验期间未再改动；`rscheck/body.rs` 与 src 对应区间 md5 相同，均为 `ace8f27e88a02627f56ef6d01a584663`）。

---

## 1. 产物与复现命令

| 文件 | 说明 |
|---|---|
| `ref_mprob.py` | 独立 Python 参考实现（只看 SPEC，未看 Rust）。`compute_mprob(..., erf=erf_math)` 定义版、`erf_as`（A&S 7.1.26）、向量化 `compute_mprob_batch` |
| `null_dist.txt` / `null_extra.txt` | 空值参考分布复算输出 |
| `rscheck/body.rs` | 逐字从 `src/tail_v5_pipeline.rs` 抽出的 `erf_as_7_1_26` + `std_normal_cdf` + `compute_mprob`（sed 1007..1127） |
| `rscheck/main.rs`、`build.sh`、`mprob_rs` | 独立 Rust 跑测驱动（rustc -O，不依赖 cargo） |
| `rscheck/driver.py`、`cases.txt` | 199 个用例的生成与逐位比对 |
| `e2e/e2e_mprob.py`、`e2e_run2.log` | 端到端复算（rolled + neu 两阶段 × gap1/gap5，含 IC 序探针） |
| `e2e/sens_mprob.py`、`sens_run2.log` | 单股换桶敏感度实验（解释 neu 残留偏差） |

```bash
PY=/home/chenzongwei/.conda/envs/chenzongwei311/bin/python
cd /home/chenzongwei/rust_pyfunc/tmp_mono_metric/verify_mprob

$PY ref_mprob.py            # erf 两版本差、向量化 vs 字面、B/C/A 用例
$PY ref_mprob.py --null     # 20000 次空值模拟（约 45s）
cd rscheck && ./build.sh && $PY driver.py     # Rust vs Python 逐位比对（199 用例）

cd ../e2e
$PY -u e2e_mprob.py /tmp/mprob_smoke_on_tail_v4 \
    /hdd/user_home_unsafe/chenzongwei/factor_data/mprob_smoke_on rolled,neu 1,5
$PY -u sens_mprob.py        # 单股换桶敏感度
```

## 2. SPEC 第 1 节逐条核对（Rust 实现）

| # | SPEC 要求 | Rust 实现（`tail_v5_pipeline.rs`） | 结论 |
|---|---|---|---|
| 1 | `portf_num != 10` 或组数 != 10 → NaN | 1065 `portf_num != 10 \|\| group_returns.len() != portf_num` | ✅ 等价 |
| 2 | `n = len[0]`；`n < 2` 或任一列长 ≠ n → NaN | 1068-1071 | ✅ |
| 3 | 任一元素非有限 → NaN | 1072-1077 全元素 `is_finite` | ✅ |
| 4 | `r[d] = 时间均值` | 1079-1082 | ✅ |
| 5 | `r[9] < r[0]` 时 `idx[d] = 9-d` | 1083-1086；与 `compute_ssm` 983-985 同一端点规则（SSM 用 `r.reverse()`，对 idx 映射等价） | ✅ 方向一致 |
| 6 | `L = min(n-1, floor(4·(n/100)^(2/9)))` | 1088 `floor(...) as usize` 后 `.min(n-1)` | ✅ n=2→1、50→3、120→4、1212→6、2424→8 |
| 7a | `d_t = gr[idx[j]][t] - gr[idx[i]][t]`，i<j | 1093-1104 `a=gr[idx[i]]`、`b=gr[idx[j]]`、`b[t]-a[t]` | ✅ 无下标错位 |
| 7b | `gamma_k = Σ_{t=k}^{n-1} e_t·e_{t-k} / n` | 1111-1116 `for t in k..n`，`/= n` | ✅ 区间与除数都对 |
| 7c | Bartlett 权重 `1 - k/(L+1)` | 1117 | ✅ |
| 7d | `nw_var = (gamma_0 + 2Σ)/n` | 1119 | ✅ 再除 n = 均值的方差 |
| 7e | `nw_var` 非有限或 ≤0 → 该对贡献 0.0 | 1120-1122 `continue` | ✅ 等价（跳过 = 加 0） |
| 7f | `t = mean/√nw_var`，贡献 `2Φ(t)-1` | 1123 | ✅ |
| 8 | `Σ(45 对)/45` | 1126 | ✅ |
| erf | A&S 7.1.26 五系数，与 `copula.rs::erf` 同式 | 1009-1024 系数与表达式逐字相同 | ✅ |
| Φ | `0.5*(1+erf(x/√2))` | 1027-1029 | ✅ |
| 签名 | `pub(crate) fn compute_mprob(&[Vec<f64>], usize) -> f64` | 1064 | ✅ |

## 3. SPEC 第 2 节接线核对

| SPEC 要求 | 实际位置 | 结论 |
|---|---|---|
| `summary: [f64; 11]` → `[f64; 12]` | `tail_v5_pipeline.rs:546` | ✅ |
| `default_legacy_backtest_result` → `[f64::NAN; 12]` | `:946` | ✅ |
| 两处数组字面量末尾追加 | `:1491`（`legacy_backtest_single_factor_with_effective`）、`:1703`（`..._opt`） | ✅ 都在末尾（下标 11） |
| `ic_only` 分支 summary | `:1447-1459` 12 项，下标 10/11 = NaN | ✅ |
| `tail_v8_backtest.rs::BtAcc::finish` 末尾追加 | `:488` | ✅ |
| `SummaryRowRecord` 加 mprob（rename `"MPROB"` + `default_nan_f64`） | `:433-435`，紧跟 ssm | ✅ |
| `write_summary_parquet` schema + 列数组 + 列序文档表 | `:3478`（Field）、`:3515`（`f64_col(mprob)`）、`:3455`（文档表 `\| 16 \| MPROB \|`） | ✅ 17 列，列序 = 结构体字段序 |
| v8 selfcheck `0..11` → `0..12` | `tail_v8_backtest.rs:577-578`（注释同步） | ✅ |
| `values[10]/[11]` 回读 | `summary_from_row` 用 `values.get(10)/(11).unwrap_or(NAN)` | ✅ 旧 11 列缓存回读不 panic |
| 全仓残留 11 假设 | 无（`factor_neutralize_std.rs` / `tail_v8_neu_v3.rs` 的 `[f64;11]` 是 11 个风格列，无关） | ✅ |
| summary parquet 唯一写出点 | `tail_v5_pipeline.rs:3460`（8 处调用） | ✅ 见第 5 节观察 2 |

## 4. 实测

### 4.1 空值参考分布（20000 次、10 组 iid N(0,1)、T=2424）

| 分位 | 5% | 25% | 50% | 75% | 90% | 95% | 99% | sd |
|---|---|---|---|---|---|---|---|---|
| SPEC 表 | -0.1269 | -0.0005 | 0.0903 | 0.1845 | 0.2675 | **0.3149** | 0.4009 | 0.1343 |
| 本实现 seed 20240918 | -0.1272 | -0.0015 | 0.0897 | 0.1852 | 0.2652 | **0.3140** | 0.3995 | 0.1336 |

95% 分位跨 4 个种子（T=2424）：0.3140 / 0.3204 / 0.3165 / 0.3143（均值 0.3163，蒙特卡洛标准误约 0.002）；T=1212 = 0.3146。**SPEC 表整体落在噪声内，门槛 0.315 站得住。**

### 4.2 Rust vs Python 逐位对比（数学层）

逐字抽出 Rust 函数、`rustc -O` 独立编译，**199 个用例**：n ∈ {2,3,4,5,10,17,50,120,243,2424} × 漂移正负 × 尺度 {0.01,1,5}，纯噪声，AR(-1) 强均值回复，恒定阶梯，阶梯+1e-7 扰动（正反两向），真实感 A，rust-core 的 LCG 用例，以及 NaN/±Inf/列长不齐/组数 9 和 11/`portf_num` 5 和 0/n=0/n=1。

**结果：NaN 判定 10/10 一致，非 NaN 全部逐位相同（0 处不同）。**

### 4.3 期望值（Rust 单元测试的硬编码值）

| 用例 | 输入 | 期望值 |
|---|---|---|
| A | `default_rng(20240501).standard_normal((10,120)) + 0.02*d` | A&S 版 **0.20324338436656386**（math.erf 版 0.20324340391338103，差 1.95e-8） |
| B | `group_returns[d][t] = d`，n=50 | 0.0（恒定价差 → `nw_var=0` → 兜底） |
| C | 全 5.0，n=50 | 0.0 |
| D | n=30，`8*d + 1e-7*((d*37+t*17)%11)` | 1.0（正反两向都是 1.0；`nw_var≈4.7e-15`，t≈1.2e8，Φ 在 float64 饱和） |
| E | LCG 用例（n=2424） | **0.50229993788529792**（= `0.5022999378852979`，hex `0x1.012d751c00186p-1`），与 rust-core 逐位相同 |

A 的 1200 个数落盘 `/tmp/mprob_rs/realcase_A.{rs.txt,csv,npz}`；Rust 单元测试里嵌入的 `REALCASE_A` 1200 个数经提取比对，**1200/1200 逐位等于** seed 20240501 的重生成结果（没有粘贴错）。

### 4.4 端到端（引擎 parquet 对比）

smoke run：`/tmp/mprob_smoke_on_tail_v4`（12 个源因子 × 13 个非 fold 派生面 × 2 个 gap；源因子值取自引擎导出的 `/hdd/user_home_unsafe/chenzongwei/factor_data/mprob_smoke_on`，其中 3 个源因子有 parquet，共 78 个可比对组合）。

| 阶段 | 组合数 | MPROB 最大偏差 | MPROB 逐位相同 | SSM 逐位相同 | ann_ret 最大偏差 | IC_mean 最大偏差 |
|---|---|---|---|---|---|---|
| rolled gap1 | 35 | **0.000e+00** | **35/35** | 35/35 | 3.3e-16 | 4.9e-17 |
| rolled gap5 | 37 | **0.000e+00** | **37/37** | 37/37 | 3.3e-16 | 4.2e-17 |
| neu gap5 | 39 | **0.000e+00** | **39/39** | 39/39 | 1.7e-16 | 2.7e-08 |
| neu gap1 | 39 | 3.449e-05 | 36/39 | 37/39 | 1.269e-05 | 8.6e-09 |

`annualized_return` / `hedge_annualized_return` / `ratio_mean` 与 parquet 的偏差都在 1e-16 量级，说明我复现的分组收益就是引擎的分组收益（rolled 阶段信号输入逐位一致，可作为「复现链路正确」的交叉验证）。

产物侧：4 个 candidates parquet 各 17 列、`MPROB` 在最后一列，非空 312/312、312/312、298/298、282/282，取值范围全部落在 [-1,1]（neu_gap1 [-0.449979, 0.742018]、neu_gap5 [-0.433717, 0.778738]、rolled_gap1 [-0.471758, 0.833815]、rolled_gap5 [-0.322487, 0.826886]）。lead 给的三个锚点核对通过：`price_log_return_3s_mean_smooth_10` → SSM 0.964047 / MPROB 0.778738 / IC_mean -0.039498（该行在 neu gap5 的 39/39 逐位相同集合里）。

### 4.5 neu gap1 那 3 条残留偏差的来源（不是 MPROB 的问题）

| 组合（gap1） | py − parquet 的 MPROB | SSM | ann_ret | IC_mean |
|---|---|---|---|---|
| `observable_ratio_level_mean_smooth_5` | +3.449e-05 | -2.227e-03 | **-4.7e-17** | -3.744e-09 |
| `price_log_return_3s_std_smooth_20` | +6.290e-06 | -1.555e-05 | -1.269e-05 | +4.278e-09 |
| `spread_bps_std_smooth_10` | +2.943e-06 | 0.000e+00 | **-1.9e-17** | +1.101e-09 |

证据链：

1. **rolled 阶段（无中性化）MPROB 逐位相同 72/72**，所以算法本身没有问题。
2. **IC 探针**（引擎逐字定义的 Spearman，只依赖信号的序）：rolled 阶段偏差 ≤4.9e-17（浮点噪声级）；neu 阶段**每一行**都差 1e-9~3e-8，说明 Python 只能调到的 `rp.neutralize_std_block_py`（v2/section）与引擎 v8 走的 `tail_v8_neu_v3`（v3）在少数位置次序不同 —— 与上一轮 `verify_ssm.md` 的结论同源。
3. **敏感度实验**（`e2e/sens_mprob.py`，取偏差最大的 `observable_ratio_level_mean_smooth_5` gap1，在某个交易日把跨桶边界的相邻两个股票信号值互换，重算）：单次换一对，**|dMPROB| 中位 9.2e-05、最大 3.6e-04；|dSSM| 中位 9.5e-04、最大 3.4e-03；|dann| 在 15 次里有 11 次恰好为 0**。实测残留（|dMPROB|=3.4e-05、|dSSM|=2.2e-03、|dann|=4.7e-17）正好落在「少数股票换桶」能造成的范围内；`dann` 为 0 也解释了为什么第 1、3 条组合的 `annualized_return` 仍在 1e-17 —— 换桶发生在中间桶，多空两端没动。
4. neu gap5 是最好的反证：IC 也差 ~1e-8（序确实变了），但桶内互换不改变桶均值，所以分组收益逐位一致（1.7e-16）、**MPROB 39/39 逐位相同**。

结论：MPROB 在「输入分组收益与引擎逐位一致」的每一条上都逐位一致；3 条偏差全部伴随分组收益本身的变化，根因是中性化 v2/v3 的次序差异，与 `compute_mprob` 无关。

## 5. 观察（不是缺陷）

1. **归零校验**：`use_mprob=False` 那轮（`/tmp/mprob_smoke_off_tail_v4`）与 `use_mprob=True` 且门槛 -1.0 那轮，`gap1_selected.parquet` / `gap5_selected.parquet` 的入选名单**完全一致（各 5 条，交集 5/5）**。注意 off 轮走的是 `ic_only` 快路径（`annualized_return` 全 0、SSM/MPROB 全 NaN，符合 SPEC 的 ic_only → NaN 约定），所以它验证的是「门槛不误杀」，不是完整的逐位归零。
2. `src/tail_v4_pipeline.rs` 有自己独立的 `SummaryRowRecord` 和 `[f64;10]` summary，走 JSON 不写 parquet，连 SSM 都没有 —— 是另一条旧路径，SPEC 未列入，本次未改动。唯一写 summary parquet 的地方是 `tail_v5_pipeline.rs:3460`，所以 MPROB 一定会进生产 parquet。

## 6. SPEC 疑点

1. **第 2 节用例 1/2 的期望值 1.0 与第 1 节第 7 条兜底冲突。** 「完美阶梯 `[0,8,...,72]`（单日）」字面读法是 n=1，走第 2 条返回 NaN；读成「10 组各 n≥2 天、组值恒定」则每对价差逐日不变 → `e_t ≡ 0` → `gamma_0 = 0` → `nw_var = 0` → 兜底贡献 0.0，45 对全 0 → MPROB = 0.0。两条路都得不到 1.0。要取到 1.0，输入必须非退化（例如第 4.3 节的用例 D，实测正反两向都是 1.0）。rust-core 独立发现了同一问题，单元测试已改成「恒定阶梯 → 0.0」+「阶梯+1e-7 扰动 → 1.0」。
2. **第 1 节「MPROB 的绝对误差 ≤ 7.5e-8」小了 2 倍。** Φ 的误差 = 0.5 × 1.5e-7 = 7.5e-8，而贡献是 `2Φ-1`，系数 2 把它放大回 1.5e-7；45 对等权平均不降低最坏情形。文档注释应写 1.5e-7（实测 A 用例两版本差 1.95e-8，符合）。Rust 侧注释已按 1.5e-7 写。
3. 其余逐条复核无笔误：`L` 公式、`gamma_k` 的求和区间 `Σ_{t=k}^{n-1}`、Bartlett 权重 `1-k/(L+1)`、`nw_var` 再除 `n`（均值的方差）与 `t = mean/√nw_var` 自洽；定向「整条倒序 ≡ 45 对全部取负」严格成立（向量化版与字面版差 1.1e-16），定向不变性实测 `MPROB(A 倒序) == MPROB(A)`。
