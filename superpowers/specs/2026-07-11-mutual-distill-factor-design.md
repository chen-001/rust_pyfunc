# 多空互蒸馏因子（distill_mutual）设计文档

**日期**：2026-07-11
**作者**：chenzongwei + Claude（brainstorming 协作）
**状态**：设计完成，待用户审阅 → 转 writing-plans
**关联**：与 `2026-07-09-distillation-factor-design.md`（①+⑥）和 `2026-07-11-tick-quantizer-factor-design.md`（③）独立共存

---

## 0. 背景与动机

### 0.1 LLM 双向蒸馏 / 互蒸馏 + 温度参数

LLM 蒸馏的经典形式是单向 teacher→student。**Deep Mutual Learning (DML)** 提出双向蒸馏：两个模型互为 teacher/student，互相学习对方的输出分布，最终两者都提升。

核心机制：
- **互蒸馏**：A 学 B 的分布，B 同时学 A 的分布
- **温度参数 T**：softmax(B/T) 调控分布尖锐度，高 T 软化、低 T 尖锐
- **双向 KL**：KL(A‖B) + KL(B‖A) 衡量双方互相逼近的程度

### 0.2 迁移到 A 股市场

A 股市场上**主买流 B(t) 和主卖流 S(t) 天然互为 teacher/student**：
- 主买代表多头信念，主卖代表空头信念
- 双方互相影响：主买会触发主卖（获利了结），主卖会触发主买（抄底）
- 当日波动率 = 温度 T：高波动时双方信念都"软化"（更谨慎），低波动时都"尖锐"（信念极化）

### 0.3 为什么选 ② 多空互蒸馏

- **与 ①+⑥/③ 重叠最低**：主买主卖互动视角，完全新的维度
- **温度参数 T 是独有创新**：让因子族有"信念强度"维度，所有其他族都没有
- **纯 L2 trade，数据简单**：主买/主卖从 `TradeRecord.flag` 直接判定（66=主买, 83=主卖）
- **概念新颖**：多空互为 teacher/student + 波动率当温度的组合，文献里几乎没人系统做过
- **算子全部通用**：OLS / softmax / KL / JS / Hawkes η 都是教科书公式

### 0.4 与 ①+⑥/③ 的关系

完全独立的第三族因子：
- 独立模块（`distill_mutual_metrics.rs`）
- 独立 pipeline 名（`"distill_mutual"`）
- 独立因子前缀（`mP_/mQ_/mE_`）
- 三族可在同一次 pipeline run 中共存，联合筛选 1170 因子（290+480+400）

---

## 1. 核心约束

| # | 约束 | 来源 |
|---|---|---|
| 1 | **不依赖特质模块** | 用户要求。`hawkes_analysis` / `ghost_market_maker` / `observable_order_metrics` 等都禁用 |
| 2 | **只用成熟通用算子** | OLS / softmax / KL / JS / Hawkes η 闭式 / EWMA / 分位数 / 2×2 矩阵运算等教科书算子直接实现 |
| 3 | **纯 Rust 实现** | 数据读取 + 处理 + 因子输出全 Rust，采用 rust-pipeline-factor-pattern 三层结构 |
| 4 | **不并行** | 用户 CLAUDE.md 明确 |
| 5 | **确定性** | 不用 HashMap 遍历顺序参与下游计算；需要遍历时先排序 |
| 6 | **性能红线** | 000001 / 20220819 单股单日 ≤ 1.0 秒 |
| 7 | **首期只做日频横截面因子** | 每天每股 1 个值 |
| 8 | **与 ①+⑥/③ 隔离** | 独立模块、独立 pipeline 名、独立前缀 |

---

## 2. 总体架构

### 2.1 三层结构（rust-pipeline-factor-pattern 范式）

```
src/distill_mutual_metrics.rs (新文件)
├── compute_distill_mutual_full(code, date) -> io::Result<Vec<f32>>   ← 核心唯一真相源
│   ├── read_trade_fast_inner(...)   ← 连续竞价 trade（含 flag）
│   ├── extract_active_buy_sell()    ← 按 flag 分流 B/S（66→B, 83→S）
│   ├── extract_price_series()       ← 价格序列（用于温度 T）
│   ├── compute_volatility_levels()  ← 4 档温度 T
│   ├── branch_P::compute()          ← 120 因子（双向学习矩阵 OLS）
│   ├── branch_Q::compute()          ← 180 因子（温度软化 + 双向 KL）
│   ├── branch_E::compute()          ← 100 因子（双向 Hawkes 触发矩阵）
│   └── 拼接 → Vec<f32> 长度 = OUT_LEN (400)
├── distill_mutual_names() -> Vec<String>  ← 名字单一源（Rust 生成）
├── py_distill_mutual(code, date)          ← Python 入口（抛 PyIOError）
└── py_distill_mutual_names()              ← Python 拿名字

src/factor_pipeline.rs
└── pipeline_distill_mutual(date, code, params, _td, expected_len) -> Vec<f32>
                                            ← pipeline 入口（吞异常返 NaN）
```

### 2.2 5 处注册清单

| # | 文件 | 改动 |
|---|---|---|
| 1 | `src/distill_mutual_metrics.rs` | 新建：核心 + 名字 + Python 入口（~1800 行） |
| 2 | `src/lib.rs` | `pub mod distill_mutual_metrics;` + 注册 `py_distill_mutual` + `py_distill_mutual_names` |
| 3 | `src/factor_pipeline.rs` | `use crate::distill_mutual_metrics;` + `known` 加 `"distill_mutual"` + 参数解析 + 线程分发（~645 行）+ v6 入口（~1351/1394 行）+ 新增 `pipeline_distill_mutual` 函数 |
| 4 | `src/bin/worker_pipeline.rs` | `use` 引入 + 分发 `else if pipeline_name == "distill_mutual"` |
| 5 | `python/rust_pyfunc/__init__.pyi` | `def py_distill_mutual(...) -> List[float]: ...` + `__all__` 加名字 |

### 2.3 参数策略

**不引入新 Params**。所有参数（窗口网格、滞后阶、温度档、阈值、触发窗口）在 `distill_mutual_metrics.rs` 用 `const` 写死。

### 2.4 总体数据流

```
┌─────────────────────────────────────────────────────────┐
│                  共享数据底座 (shared)                   │
│  ① L2 trade 读取（连续竞价，已平移）                      │
│  ② 主买流 B(t) / 主卖流 S(t) 提取（flag 分流）           │
│  ③ 价格序列 P(t) 提取                                    │
│  ④ 温度 T 4 档计算（EWMA 波动率分位数）                  │
│  ⑤ 通用算子库（OLS / softmax / KL / JS / Hawkes η）     │
└─────────────────────────────────────────────────────────┘
              ↓ 喂给 3 个独立分支
┌──────────────┬──────────────┬────────────────────────┐
│ 分支 P       │ 分支 Q       │ 分支 E                 │
│ 双向学习矩阵 │ 温度软化+KL  │ 双向 Hawkes 触发矩阵   │
│ OLS 滞后回归 │ 4 温度档扫描 │ 3 阈值 × 3 窗口        │
│ 120 因子     │ 180 因子     │ 100 因子               │
└──────────────┴──────────────┴────────────────────────┘
              ↓ 各自输出因子值
┌─────────────────────────────────────────────────────────┐
│             统一回测框架 (tail_backtest)                 │
└─────────────────────────────────────────────────────────┘
```

---

## 3. 共享数据底座

### 3.1 数据来源

`fast_csv_reader` 提供：
- `read_trade_fast_inner(code, date, false, true, MAX)`：连续竞价 trade（已平移）

**关键**：② **只用 trade 数据**，不需要 snapshot。这让数据读取比 ①+⑥/③ 更简单（trade 数据量约为 snapshot 的 1/5）。

### 3.2 TradeRecord 字段（关键确认）

```rust
pub struct TradeRecord {
    pub time_sec: f32,      // epoch 秒
    pub price: f32,
    pub volume: f32,
    pub turnover: f32,
    pub flag: i32,          // 66=主买, 83=主卖, 32=撤单（已过滤）
    pub bid_order: i64,
    pub ask_order: i64,
}
```

**主买/主卖判定直接用 flag 字段**：
- `flag == 66`（ASCII 'B'）→ 主买 B
- `flag == 83`（ASCII 'S'）→ 主卖 S
- 其他 flag → 忽略（撤单已被 `read_trade_fast_inner` 过滤）

### 3.3 主买流 B(t) / 主卖流 S(t) 提取

对每笔 trade，按 flag 分流，按 time_sec 聚合到固定时间桶（默认 1 秒）：

```rust
struct ActiveBuySell {
    time_sec: Vec<f32>,    // 时间桶（1 秒粒度）
    b_volume: Vec<f32>,    // 每秒主买量
    s_volume: Vec<f32>,    // 每秒主卖量
}
```

**聚合算法**（O(N)）：
```
对每笔 trade (time_sec, volume, flag)：
  bucket = floor(time_sec) - day_start_sec
  if flag == 66: b_volume[bucket] += volume
  elif flag == 83: s_volume[bucket] += volume
```

### 3.4 温度 T 的 4 档定义

**基于当日 EWMA 波动率的分位数**：

1. 计算成交价 P(t) 的 logret 序列
2. 用 EWMA（span=5min）算波动率序列 vol(t)
3. 取 vol(t) 当日的 4 个分位数作为 4 档温度：
   - `T_low = vol.quantile(0.25)`
   - `T_mid = vol.quantile(0.50)`
   - `T_high = vol.quantile(0.75)`
   - `T_ultra = vol.quantile(0.95)`

**物理含义**：
- T_low（低波动时段）：双方信念尖锐，分布极化
- T_ultra（高波动时段）：双方信念软化，分布平坦

**温度 T 在分支 Q 中的作用**：
- soft_B = softmax(B(t) / T)
- soft_S = softmax(S(t) / T)
- 温度越高，soft_B / soft_S 越接近均匀分布，KL 越小

### 3.5 通用算子清单（全部教科书公式）

| 算子 | 公式 | 用途 |
|---|---|---|
| OLS | 正规方程 β = (XᵀX)⁻¹Xᵀy | 双向学习矩阵 |
| softmax | exp(x_i/T) / Σ exp(x_j/T) | 温度软化 |
| KL 散度 | Σ p·log(p/q) | 分布差异（不对称） |
| JS 散度 | 0.5·KL(p\|m) + 0.5·KL(q\|m) | 分布差异（对称） |
| EWMA std | exp加权标准差 | 波动率 |
| Hawkes η | ΣN_t / Σ事件数 | 闭式触发比 |
| 2×2 矩阵运算 | 行列式/迹/特征值（闭式） | 矩阵衍生 |
| 分位数 | 排序后取位置 | 温度档 |

---

## 4. 三个分支详细设计

### 4.1 分支 P：双向学习矩阵（120 因子）

**核心**：滑动窗口 OLS 拟合 2×2 矩阵

**学习矩阵定义**：
```
B(t) = α_BB·B(t-k) + α_BS·S(t-k) + ε_B
S(t) = α_SB·B(t-k) + α_SS·S(t-k) + ε_S
```

**参数网格**：
- 窗口 w ∈ {5, 15, 30, 60, 120} min（5 值）
- 滞后阶 k ∈ {1, 5, 15} min（3 值）

**因子族**：

**(a) 矩阵元素（2×2 × 5 窗口 × 3 滞后 = 60 个）**

对每个 (w, k) 组合：
- `mP_aBB_w{w}_k{k}`：B(t) 对 B(t-k) 的回归系数
- `mP_aBS_w{w}_k{k}`：B(t) 对 S(t-k) 的回归系数
- `mP_aSB_w{w}_k{k}`：S(t) 对 B(t-k) 的回归系数
- `mP_aSS_w{w}_k{k}`：S(t) 对 S(t-k) 的回归系数

**(b) 拟合质量（40 个）**

对主要的 (w, k) 组合（取 5 窗口 × 2 滞后 = 10 组合 × 4 元素 = 40 个）：
- `mP_r2_{elem}_w{w}_k{k}`：R²（4 元素 × 10 组合 = 40）

**(c) 矩阵衍生特征（20 个）**

对每个窗口 w（5 值）：
- `mP_asymmetry_w{w}`：α_BS - α_SB（多空学习不对称）
- `mP_det_w{w}`：det([α])（矩阵行列式）
- `mP_trace_w{w}`：tr([α])（矩阵迹）
- `mP_max_eig_w{w}`：最大特征值（2×2 闭式）

**因子数小计**：60 + 40 + 20 = **120**

### 4.2 分支 Q：温度软化 + 双向 KL（180 因子）

**核心**：B/S 归一化为概率分布，温度 T 软化后算双向 KL

**3 种分布维度**：
- **时间维 d=time**：B(t) 在窗口内按时刻归一化（每秒一个概率）
- **价格维 d=price**：B 按成交价桶归一化（分 10 个价格桶，按当日价格范围等分）
- **成交量维 d=volume**：B 按单笔成交量桶归一化（分 5 个量桶，如 [0,100], [100,500], [500,1000], [1000,5000], [5000,+]）

**参数网格**：
- 温度 T ∈ {T_low, T_mid, T_high, T_ultra}（4 档）
- 窗口 w ∈ {5, 15, 30, 60} min（4 值）
- 维度 d ∈ {time, price, volume}（3 维）

**因子族**：

**(a) 双向 KL（2 方向 × 4 温度 × 4 窗口 × 3 维度 = 96 个）**

对每个 (T, w, d) 组合：
- `mQ_KL_B_to_S_T{T}_w{w}_d{d}`：KL(soft_B ‖ soft_S)
- `mQ_KL_S_to_B_T{T}_w{w}_d{d}`：KL(soft_S ‖ soft_B)

**(b) KL 随 T 变化曲线（4 窗口 × 3 维度 × 3 特征 = 36 个）**

对每个 (w, d)：
- `mQ_KL_slope_w{w}_d{d}`：KL_B_to_S 随 T 的线性斜率
- `mQ_KL_curvature_w{w}_d{d}`：KL_B_to_S 随 T 的曲率（二阶差分）
- `mQ_KL_area_w{w}_d{d}`：KL_B_to_S 曲线下面积

**(c) JS 散度（12 个）**

对每个 (T, d)（4×3=12）：
- `mQ_JS_T{T}_d{d}`：JS(soft_B, soft_S)

**(d) 温度敏感性（36 个）**

对每个 (w, d)（4×3=12）× 3 对比 = 36：
- `mQ_KL_diff_low_high_w{w}_d{d}`：T_low vs T_high 的 KL 差值
- `mQ_KL_ratio_low_high_w{w}_d{d}`：T_low / T_high 的 KL 比值
- `mQ_KL_diff_mid_ultra_w{w}_d{d}`：T_mid vs T_ultra 的 KL 差值

**因子数小计**：96 + 36 + 12 + 36 = **180**

### 4.3 分支 E：双向 Hawkes 触发矩阵（100 因子）

**核心**：大单 B/S 事件互相触发，拟合 2×2 branching matrix

**事件定义**：
- 大买事件：单笔主买量 ≥ 当日成交量 thr 分位（thr ∈ {10%, 5%, 1%}）
- 大卖事件：单笔主卖量 ≥ 同阈值

**触发矩阵**：
```
η_BB: 大买事件触发后续大买事件
η_BS: 大买事件触发后续大卖事件
η_SB: 大卖事件触发后续大买事件
η_SS: 大卖事件触发后续大卖事件
```

**参数网格**：
- 阈值 thr ∈ {10%, 5%, 1%}（3 值）
- 触发窗口 τ ∈ {30s, 60s, 120s}（3 值，注意是秒）

**因子族**：

**(a) 触发矩阵元素（2×2 × 3 阈值 × 3 窗口 = 36 个）**

对每个 (thr, tau) 组合：
- `mE_eta_BB_thr{thr}_tau{tau}`、`mE_eta_BS_thr{thr}_tau{tau}`
- `mE_eta_SB_thr{thr}_tau{tau}`、`mE_eta_SS_thr{thr}_tau{tau}`

**(b) 基线强度（2 方向 × 3 阈值 × 3 窗口 = 18 个）**

- `mE_lambda_B_thr{thr}_tau{tau}`：大买事件基线强度
- `mE_lambda_S_thr{thr}_tau{tau}`：大卖事件基线强度

**(c) 衍生特征（46 个）**

- `mE_asymmetry_thr{thr}_tau{tau}`：η_BS - η_SB（3 阈值 × 3 窗口 = 9 个，多空触发不对称）
- `mE_branching_total_thr{thr}_tau{tau}`：η 总和（3 阈值 × 3 窗口 = 9 个，集体触发强度）
- `mE_two_layer_ratio_thr{thr}_tau{tau}`：η_BB / η_BS 比值（3 阈值 × 3 窗口 = 9 个，同类 vs 异类触发比）
- `mE_cross_threshold_consistency_{elem}`：跨阈值 η 的相关（4 元素 = 4 个，每元素在 3 阈值下的相关系数均值）
- `mE_dominant_direction_thr{thr}`：主导触发方向（3 阈值 = 3 个，B→S 主导还是 S→B 主导）
- `mE_diagonal_strength_{tau}`：对角元素 η_BB+η_SS 平均（3 窗口 = 3 个，自触发强度）
- `mE_offdiag_strength_{tau}`：非对角元素 η_BS+η_SB 平均（3 窗口 = 3 个，跨向触发强度）
- `mE_diag_offdiag_ratio`：对角 vs 非对角比（1 个，整体自触发倾向）
- `mE_lambda_asymmetry_thr{thr}`：λ_B - λ_S（3 阈值 = 3 个，基线强度不对称）
- `mE_event_intensity_ratio`：大买事件数 / 大卖事件数（2 个，全日 + 早盘各 1）

**(d) 闭式估计**（不做 EM 迭代）：
```
λ_X = X 事件总数 / 时段长度（秒）
η_XY = Σ (Y 事件在 X 事件后 τ 窗口内出现的次数) / Σ (X 事件数)
```

**因子数小计**：36 + 18 + 46 = **100**

### 4.4 总因子数量

| 分支 | 因子数 | 前缀 |
|---|---|---|
| P 双向学习矩阵 | 120 | `mP_` |
| Q 温度软化+KL | 180 | `mQ_` |
| E 双向 Hawkes | 100 | `mE_` |
| **合计** | **400** | |

**`OUT_LEN = 400`**

---

## 5. 数据流

### 5.1 单股单日数据流（核心 `compute_distill_mutual_full`）

```
输入：code: &str, date: i64

步骤 1：数据读取（≤150ms，只读 trade）
└── trade = read_trade_fast_inner(code, date, false, true, MAX)?
    └── 连续竞价 trade（已平移），含 flag 字段

步骤 2：共享底座计算（≤200ms）
├── (B(t), S(t)) = extract_active_buy_sell(trade)
│   └── 按 flag 分流：flag==66 → B, flag==83 → S
├── P(t) = extract_price_series(trade)
├── logret = compute_logret_series(P)
├── ewma_vol = ewma_std(logret, window=5min)
└── T_levels = compute_volatility_levels(ewma_vol)
    └── [T_low, T_mid, T_high, T_ultra] 4 档

步骤 3：分发到 3 个分支（≤600ms）
├── branch_P::compute(B, S, window_grid, lag_grid)
│   → Vec<f32> 长度 120
├── branch_Q::compute(B, S, T_levels, window_grid, dim_grid)
│   → Vec<f32> 长度 180
└── branch_E::compute(B, S, threshold_grid, trigger_window_grid)
    → Vec<f32> 长度 100

步骤 4：拼接（≤30ms）
└── out = Vec::with_capacity(400)
    ├── extend(branch_P_out)
    ├── extend(branch_Q_out)
    └── extend(branch_E_out)
    → 长度 = 400
```

### 5.2 批量数据流（生产级别）

通过 `run_factor_pipeline(pipeline="distill_mutual", tasks=[[date, code], ...])` 批量调用：
- worker 进程调 `pipeline_distill_mutual(date, code, params, _td, expected_len=400)`
- 输出：`(n_stocks × n_days, 400)` 的因子面板

### 5.3 数据缺口处理

| 情况 | 处理 |
|---|---|
| trade 数据为空 | 400 因子全 NaN |
| trade 记录数 < 100 | 400 因子全 NaN |
| 主买或主卖事件数为 0 | 相关分支因子 NaN |
| logret 序列长度 < 30 | T_levels 全 NaN，分支 Q 因子 NaN |
| 5min 窗口内主买/主卖不足 10 笔 | 该窗口 P/Q 因子 NaN |
| 大单事件数 < 10 | 分支 E 该阈值组合 NaN |
| 矩阵元素拟合时窗口样本不足 | 该组合因子 NaN |
| KL 计算时分母为 0 | 加 ε=1e-12 平滑 |
| 温度 T = 0（零波动日） | 所有温度档 NaN |

### 5.4 关键时序约定

- `adjust_afternoon=true`
- 连续竞价段（平移后）：day_ts ∈ [34200, 48420]
- 时间窗口网格 w ∈ {5, 15, 30, 60, 120} min
- 滞后阶 k ∈ {1, 5, 15} min
- 触发窗口 τ ∈ {30, 60, 120} 秒

---

## 6. 误差处理与数值稳定性

### 6.1 数值稳定性

| 风险点 | 防护 |
|---|---|
| OLS 拟合时矩阵奇异（窗口样本不足或共线性） | 正规方程 + 对角扰动（ridge 项 ε=1e-8），失败返 NaN |
| softmax 溢出（B/S 值极大） | 减去最大值后再 exp（标准 softmax 稳定化） |
| KL 散度 log(0) | 用 `p·log(p+ε) - p·log(q+ε)`，ε=1e-12 |
| 温度 T=0（零波动日） | T≤1e-8 时所有温度档置 NaN |
| Hawkes 事件数 < 10 | 该组合 η/λ 全 NaN |
| 2×2 矩阵特征值计算时判别式 < 0（复数特征值） | 取实部（2×2 实数矩阵的复特征值成共轭对，取实部合法） |
| B/S 归一化时总和为 0 | 整个分布置 NaN |
| 滞后阶 k 导致样本对齐错位 | 前置 check，窗口长度 < k+5 时 NaN |
| EWMA 初值不稳定 | 前 10 个样本用 expanding 代替 EWMA |
| 分位数计算时样本 < 20 | 返 NaN |
| 浮点除零 | 前置 check，分母≤0 时 NaN |

### 6.2 确定性约束（CLAUDE.md 强制）

- **禁止**用 HashMap 遍历顺序参与下游计算
- B(t)/S(t) 序列按 time_sec 严格升序排列
- 2×2 矩阵元素的输出顺序固定（[α_BB, α_BS, α_SB, α_SS]）
- 温度档顺序固定（[T_low, T_mid, T_high, T_ultra]）
- 3 分支的调用顺序固定（P → Q → E）
- 所有参数网格用 `const` 数组

### 6.3 边界条件

- 单股单日数据为空：返回全 NaN 的 Vec（长度 400）
- 部分数据缺失：相关因子 NaN，不 panic
- 时间戳异常（time_sec 超出 86400）：跳过该条记录

---

## 7. 性能预算（000001 / 20220819 ≤ 1s）

| 模块 | 预算 | 优化策略 |
|---|---|---|
| `read_trade_fast_inner` | ≤ 150ms | 已优化的 CSV 解析（trade 比 snapshot 小） |
| 共享底座（B/S 提取 + 价格 + 温度 + EWMA） | ≤ 200ms | 向量化遍历，flag 分流 O(N) |
| 分支 P（120 因子，OLS） | ≤ 200ms | 5 窗口 × 3 滞后 × 4 元素 = 60 次 OLS，每次 O(N) |
| 分支 Q（180 因子，softmax+KL） | ≤ 250ms | 4 温度 × 4 窗口 × 3 维度 = 48 组合，向量化 |
| 分支 E（100 因子，Hawkes 闭式） | ≤ 150ms | 3 阈值 × 3 窗口 × 4 元素 = 36 组合，O(N) |
| 拼接 + 返回 | ≤ 30ms | `Vec::with_capacity(400)` |
| **合计** | **≤ 980ms** | 留 20ms 安全垫 |

**强制约束**：
- 全程**不并行**（用户 CLAUDE.md 明确）
- OLS 用闭式解（正规方程），不迭代
- Hawkes 用闭式估计，不做 EM
- softmax 用稳定化版本（减最大值）
- 禁止 `format!` 在热路径

### 7.1 性能优化备选

若 980ms 超 1s 红线：
- 分支 P 减少窗口：5→3（去掉 30/120 min），因子数 120→80
- 分支 Q 减少温度档：4→3（去掉 T_ultra），因子数 180→140
- 分支 E 减少阈值：3→2（去掉 1%），因子数 100→70
- 极端情况：分支缩减后合计 290，仍满足 OUT_LEN 一致性（需同步改 spec）

### 7.2 性能验收红线

- **硬性**：`compute_distill_mutual_full("000001", 20220819)` 在 < 1.0 秒内返回
- 验证方式：`time.perf_counter()` 计时，运行 3 次取中位数
- 失败处理：用 `factor-performance-checker` agent 诊断瓶颈

---

## 8. 测试设计

### 8.1 单元测试（每分支独立）

| 测试 | 验证内容 |
|---|---|
| `test_active_buy_sell_flag_parsing` | flag=66→B, flag=83→S, 其他→忽略 |
| `test_ols_synthetic_linear` | 已知线性关系的合成数据，OLS 系数恢复误差 < 1% |
| `test_softmax_temperature_effect` | T 大时分布更平坦，T 小时更尖锐 |
| `test_kl_known_distributions` | 已知分布的 KL 解析解 vs 计算值，误差 < 1% |
| `test_js_symmetry` | JS(p,q) == JS(q,p) |
| `test_hawkes_closed_form_synthetic` | 已知 η 的合成事件流，闭式估计误差 < 5% |
| `test_branch_P_determinism` | 相同输入运行 2 次，120 因子完全一致 |
| `test_branch_Q_temperature_sweep` | KL 随 T 单调性合理（通常 T 大 KL 小） |
| `test_branch_E_asymmetry` | η_BS ≠ η_SB 在合成数据下正确反映 |

### 8.2 集成测试

| 测试 | 验证内容 |
|---|---|
| `test_single_stock_single_day` | 真实 trade 跑 000001/20220819，输出 400 因子（NaN < 30%） |
| `test_against_python_baseline` | 与简化 Python 实现对比，分支 P 的 α 误差 < 5% |
| `test_pipeline_python_consistency` | `py_distill_mutual` 与 `run_factor_pipeline(pipeline="distill_mutual")` 逐字节相同 |

### 8.3 性能测试

```python
# tests/test_distill_mutual_perf.py
import rust_pyfunc as rp
import time

def test_perf_000001_20220819():
    rp.py_distill_mutual("000001", 20220819)  # 预热
    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        rp.py_distill_mutual("000001", 20220819)
        times.append(time.perf_counter() - t0)
    median = sorted(times)[1]
    assert median < 1.0, f"性能超时：中位数 {median:.3f}s > 1.0s"
```

### 8.4 回测验证（后续阶段）

| 测试 | 验证内容 |
|---|---|
| `test_backtest_ic_distribution` | 400 因子日频 IC 分布合理（多数 \|IC\| 在 0.01-0.05） |
| `test_backtest_top_factors` | Top 30 IC 因子分布在 3 个分支（不应集中在 1 个分支） |
| `test_correlation_with_distill_1_and_tick` | 与 ①+⑥/③ 因子的相关性 < 0.5（确认三族互补） |

---

## 9. 实现里程碑

| 阶段 | 里程碑 | 验收标准 |
|---|---|---|
| M1 | 共享底座 + 分支 P 跑通 | `py_distill_mutual` 返回长度 120，单股单日 < 0.5s |
| M2 | 分支 Q 加入 | 返回长度 300（P+Q），单股单日 < 0.8s |
| M3 | 分支 E 加入 | 返回长度 400（全部），单股单日 < 1s |
| M4 | pipeline 注册 + worker 分发 | `run_factor_pipeline(pipeline="distill_mutual")` 可批量跑 |
| M5 | 一致性验证 | py 入口与 pipeline 入口逐字节相同 |
| M6 | 回测验证 | 400 因子 IC 分布合理，Top 因子分布均衡 |
| M7 | 三族联合回测 | 1170 因子（290+480+400）联合筛选，确认三族互补 |

---

## 10. 文件清单

### 10.1 新建文件

| 文件 | 行数估计 | 内容 |
|---|---|---|
| `src/distill_mutual_metrics.rs` | ~1800 | 核心 + 名字 + Python 入口 |
| `tests/test_distill_mutual.py` | ~50 | 集成测试 |
| `tests/test_distill_mutual_perf.py` | ~20 | 性能测试 |

### 10.2 修改文件

| 文件 | 改动 |
|---|---|
| `src/lib.rs` | 加 `pub mod distill_mutual_metrics;` + 注册 2 个 py 函数 |
| `src/factor_pipeline.rs` | 加 `pipeline_distill_mutual` 函数 + 5 处注册点 |
| `src/bin/worker_pipeline.rs` | 加 worker 分发 |
| `python/rust_pyfunc/__init__.pyi` | 加类型声明 |

### 10.3 不修改文件

- 任何特质模块（`hawkes_analysis.rs` / `ghost_market_maker.rs` 等）
- `fast_csv_reader.rs`（仅调用，不修改）
- `distill_metrics.rs`（①+⑥ 模块，完全隔离）
- `distill_tick_metrics.rs`（③ 模块，完全隔离）

---

## 11. 风险与缓解

| 风险 | 缓解 |
|---|---|
| 1s 性能红线难达（980ms 留 20ms 安全垫） | 优先实现 M1-M2（P+Q=300 因子），若 M3（分支 E）超时则按 §7.1 备选方案缩减 |
| 温度 T 的 4 档定义依赖当日波动率分位数，可能不稳定 | 用 expanding 窗口代替全日分位数（滚动分位数），但性能略降 |
| KL 散度不对称（KL_B→S ≠ KL_S→B）可能让因子难以解释 | 同时输出 JS 散度（对称版），让用户选择 |
| 主买/主卖判定依赖 flag 字段，数据源 flag 缺失时全盘失效 | 前置 check，flag 字段全为 0 时返全 NaN |
| 3 分支因子高相关 | 不处理，让回测自然筛选；符合"人海战术" |
| 400 因子中很多无效 | 预期之内，靠 IC 筛选 |

---

## 12. 后续扩展路径（不在本 spec 范围）

- **方式 R（理性预期）**：把 B(t) 当多头信念，S(t) 当空头信念，拟合信念→价格传导
- **多档温度**：从 4 档扩展到 8 档（更细的温度扫描）
- **非线性学习矩阵**：用核回归代替 OLS，捕捉非线性互动
- **参数化**：若需调参，再建 `DistillMutualParams` 结构并改 bincode

预期最终因子数：本族 400 → 扩展后 600-1000。

---

## 13. 参考实现

- **范式参考**：`src/observable_order_metrics.rs` + `src/factor_pipeline.rs::pipeline_observable_order`
- **数据读取参考**：`src/fast_csv_reader.rs::read_trade_fast_inner`（`TradeRecord.flag` 字段直接判定主买/主卖）
- **①+⑥ 参考**：`docs/superpowers/specs/2026-07-09-distillation-factor-design.md`
- **③ 参考**：`docs/superpowers/specs/2026-07-11-tick-quantizer-factor-design.md`
- **skill 参考**：`rust-pipeline-factor-pattern`
- **编译命令**：`timeout 600s bash alter.sh 2>&1`

---

**本 spec 到此结束。待用户审阅后，调用 writing-plans skill 生成实现计划。**
