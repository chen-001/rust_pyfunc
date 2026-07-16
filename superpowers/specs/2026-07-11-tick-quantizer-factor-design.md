# Tick 量化器因子（distill_tick）设计文档

**日期**：2026-07-11
**作者**：chenzongwei + Claude（brainstorming 协作）
**状态**：设计完成，待用户审阅 → 转 writing-plans
**关联**：与 `2026-07-09-distillation-factor-design.md`（①+⑥ 集合竞价+遗忘）独立共存

---

## 0. 背景与动机

### 0.1 LLM 量化感知蒸馏（Quantization-Aware Distillation）

LLM 部署时要把浮点权重量化到低精度（如 INT8/INT4），量化误差会损失模型性能。**Quantization-Aware Training (QAT)** 在训练时模拟量化误差，让模型学会对量化鲁棒。

核心机制：
- **量化器**：把连续浮点映射到离散网格（如 round-to-nearest）
- **量化损失**：原始浮点分布 vs 量化后离散分布的距离
- **温度参数**：调控量化粒度，粗粒度 = 高温（更软化）

### 0.2 迁移到 A 股市场

A 股价格必须落在 **0.01 元 tick 网格**上——这是一个**强制量化器**。投资者"真实意图"是连续分布（愿意在 10.034 元买），但订单必须落在离散 tick（只能在 10.03 或 10.04 买）。**这个强制量化过程产生的误差本身就是蒸馏损失**。

### 0.3 为什么选 ③ tick 量化器

- **概念最新颖**：把 tick 网格当"强制量化器"是纯数学创新，文献里几乎没人这么做
- **与 ①+⑥ 完全互补**：①+⑥ 看价格回归/遗忘（时序），③ 看挂单分布形态（分布形状）
- **纯 L2 snapshot**：不依赖集合竞价段，不需要横截面分组表，不需要历史数据
- **算子全部通用**：Wasserstein / 基尼 / 熵 / HHI / KL / JS / Hurst 都是教科书公式
- **因子数量容易做大**：多 tick 尺度 × 多 σ 估计 × 多窗口，轻松 480+

### 0.4 与 ①+⑥ 的关系

完全独立的第二族因子：
- 独立模块（`distill_tick_metrics.rs` vs `distill_metrics.rs`）
- 独立 pipeline 名（`"distill_tick"` vs `"distill"`）
- 独立因子前缀（`qX_/qY_/qZ_` vs `dA_/dB_/dD_/dE_`）
- 两族可在同一次 pipeline run 中共存，联合筛选 770 因子

---

## 1. 核心约束

| # | 约束 | 来源 |
|---|---|---|
| 1 | **不依赖特质模块** | 用户要求。`hawkes_analysis` / `ghost_market_maker` / `observable_order_metrics` 等都禁用 |
| 2 | **只用成熟通用算子** | Wasserstein / 基尼 / Shannon 熵 / HHI / KL / JS / Hurst（R/S 法）/ OLS / rolling std 等教科书算子直接实现 |
| 3 | **纯 Rust 实现** | 数据读取 + 处理 + 因子输出全 Rust，采用 rust-pipeline-factor-pattern 三层结构 |
| 4 | **不并行** | 用户 CLAUDE.md 明确 |
| 5 | **确定性** | 不用 HashMap 遍历顺序参与下游计算；需要遍历时先排序 |
| 6 | **性能红线** | 000001 / 20220819 单股单日 ≤ 1.0 秒 |
| 7 | **首期只做日频横截面因子** | 每天每股 1 个值 |
| 8 | **与 ①+⑥ 隔离** | 独立模块、独立 pipeline 名、独立前缀 |

---

## 2. 总体架构

### 2.1 三层结构（rust-pipeline-factor-pattern 范式）

```
src/distill_tick_metrics.rs (新文件)
├── compute_distill_tick_full(code, date) -> io::Result<Vec<f32>>   ← 核心唯一真相源
│   ├── read_market_fast_inner(...)   ← 连续竞价 snapshot（含 10 档 ask/bid）
│   ├── read_trade_fast_inner(...)    ← 连续竞价 trade（用于 σ_short/σ_early）
│   ├── extract_orderbook_10levels()  ← 10 档提取
│   ├── resample_tick_scales()        ← 3 尺度重采样（10/5/2 档合并）
│   ├── compute_sigma_spread/short/early()  ← 3 种 σ
│   ├── compute_rolling_cumulative_orderbook()  ← 5min 累积挂单 vol 分布
│   ├── branch_X::compute()           ← 78 因子（瞬时盘口）
│   ├── branch_Y::compute()           ← 150 因子（滚动窗口）
│   ├── branch_Z::compute()           ← 252 因子（双窗口多 σ）
│   └── 拼接 → Vec<f32> 长度 = OUT_LEN (480)
├── distill_tick_names() -> Vec<String>  ← 名字单一源（Rust 生成）
├── py_distill_tick(code, date)          ← Python 入口（抛 PyIOError）
└── py_distill_tick_names()              ← Python 拿名字

src/factor_pipeline.rs
└── pipeline_distill_tick(date, code, params, _td, expected_len) -> Vec<f32>
                                          ← pipeline 入口（吞异常返 NaN）
```

### 2.2 5 处注册清单

| # | 文件 | 改动 |
|---|---|---|
| 1 | `src/distill_tick_metrics.rs` | 新建：核心 + 名字 + Python 入口（~2000 行） |
| 2 | `src/lib.rs` | `pub mod distill_tick_metrics;` + 注册 `py_distill_tick` + `py_distill_tick_names` |
| 3 | `src/factor_pipeline.rs` | `use crate::distill_tick_metrics;` + `known` 加 `"distill_tick"` + 参数解析 + 线程分发（~645 行）+ v6 入口（~1351/1394 行）+ 新增 `pipeline_distill_tick` 函数 |
| 4 | `src/bin/worker_pipeline.rs` | `use` 引入 + 分发 `else if pipeline_name == "distill_tick"` |
| 5 | `python/rust_pyfunc/__init__.pyi` | `def py_distill_tick(...) -> List[float]: ...` + `__all__` 加名字 |

### 2.3 参数策略

**不引入新 Params**。所有参数（tick 尺度、σ 窗口、滚动窗口大小）在 `distill_tick_metrics.rs` 用 `const` 写死。

**理由**：
- 建新 Params 要改 `TaskMessage::Init` 的 bincode 结构，风险大
- 符合用户 CLAUDE.md "无必要不写兼容性代码" 原则
- 后续如需调参，再做参数化

### 2.4 总体数据流

```
┌─────────────────────────────────────────────────────────┐
│                  共享数据底座 (shared)                   │
│  ① L2 snapshot 读取（连续竞价，已平移）                   │
│  ② L2 trade 读取（用于 σ_short / σ_early）                │
│  ③ 10 档 ask/bid 挂单提取                                │
│  ④ 多 tick 尺度重采样（10/5/2 档合并）                    │
│  ⑤ 3 种 σ 计算（spread / short / early）                 │
│  ⑥ 5min 滚动累积挂单 vol 分布                            │
│  ⑦ 通用算子库（Wasserstein/基尼/熵/HHI/JS/KL/Hurst）    │
└─────────────────────────────────────────────────────────┘
              ↓ 喂给 3 个独立分支
┌──────────────┬──────────────┬────────────────────────┐
│ 分支 X       │ 分支 Y       │ 分支 Z                 │
│ 瞬时盘口     │ 5min 滚动    │ 双窗口 + 3 种 σ        │
│ σ=spread     │ σ=short_vol  │ spread/short/early     │
│ 78 因子      │ 150 因子     │ 252 因子               │
└──────────────┴──────────────┴────────────────────────┘
              ↓ 各自输出因子值
┌─────────────────────────────────────────────────────────┐
│             统一回测框架 (tail_backtest)                 │
└─────────────────────────────────────────────────────────┘
```

---

## 3. 共享数据底座与多 tick 尺度

### 3.1 数据来源

`fast_csv_reader` 提供：
- `read_market_fast_inner(code, date, false, true, MAX)`：连续竞价 snapshot（已平移），含 10 档 ask/bid 量价
- `read_trade_fast_inner(code, date, false, true, MAX)`：连续竞价 trade

**注意**：与 ①+⑥ 不同，tick 量化器**不需要集合竞价段**（with_afternoon_adjust=false 那次读取），也不需要前一交易日数据。数据读取比 ①+⑥ 简单。

### 3.2 10 档挂单提取

每个 `MarketRecord` 包含：
- `ask_prcs[10]` / `ask_vols[10]`：10 档卖价/卖量
- `bid_prcs[10]` / `bid_vols[10]`：10 档买价/买量

提取为 `OrderBookSnapshot`：
```rust
struct OrderBookSnapshot {
    mid: f32,                    // (ask1 + bid1) / 2
    spread: f32,                 // (ask1 - bid1) / mid
    ask_vols: [f32; 10],         // 原始 10 档卖量
    bid_vols: [f32; 10],         // 原始 10 档买量
    time_sec: f32,
}
```

### 3.3 多 tick 尺度重采样（关键定义）

A 股 tick = 0.01 元（股价 < 20 元）。10 档 ask 覆盖 0.10 元 = 10 个 tick。

**3 个尺度**（都在 10 档范围内）：
- **尺度 S1（0.01 元 tick，最细）**：原始 10 档，每档 1 桶 → 10 个桶
- **尺度 S2（0.02 元 tick）**：每 2 档合并 → 5 个桶
- **尺度 S3（0.05 元 tick，最粗）**：每 5 档合并 → 2 个桶

**为什么不用 0.10/0.50/1.00 元**：超出 10 档范围，无法做分布分析（只有 1 个桶无意义）。

**每个尺度有 ask + bid 两个分布**，共 3 尺度 × 2 边 = **6 个分布**。

**重采样算法**（S1→S2→S3）：
```
S2[i] = S1[2i] + S1[2i+1]    for i in 0..5
S3[i] = S1[5i] + ... + S1[5i+4]  for i in 0..2
```

**vol 守恒**：合并后总 vol 不变（S2 总和 = S1 总和 = S3 总和）。

### 3.4 3 种 σ 定义（teacher 分布的尺度参数）

| σ | 定义 | 时间粒度 | 物理含义 |
|---|---|---|---|
| σ_spread | (ask1 - bid1) / mid | 每 snapshot 一个值 | 瞬时流动性 |
| σ_short | 5min rolling std of trade logret | 每时刻一个值 | 短期波动 |
| σ_early | 当天前 30min（9:30-10:00）logret std | 全天一个标量 | 当天基准紧张度 |

**Teacher 分布**：每个时刻、每个 σ，构造 `Laplace(loc=mid, scale=σ)` 连续分布。

**为什么用 Laplace 而非 Gaussian**：金融价格变动分布尖峰厚尾，Laplace 比 Gaussian 更贴近。

**σ_early 的"前 30min"定义**：day_ts ∈ [34200, 36000]（9:30-10:00）。

### 3.5 5min 滚动累积挂单 vol 分布

对每个时刻 t，取 [t-5min, t] 窗口内的所有 snapshot，按 tick 桶累积 vol：
- 窗口内每个 snapshot 的 10 档 vol 按"新挂单 - 撤单"计算净变化
- 累积到对应的 tick 桶
- 得到 5min 累积挂单 vol 分布（10 档）

简化实现：直接累加窗口内每个 snapshot 的瞬时 vol（近似，不追踪挂单生命周期）。

### 3.6 Wasserstein-1 距离（1D 闭式解）

对连续 Laplace(loc=mid, scale=σ) CDF 与离散挂单分布 CDF：
```
W1 = Σ_i |F_continuous(x_i) - F_discrete(x_i)| · Δx
```
其中 Δx 是 tick 间距（0.01 元），F 是累积分布函数。

**闭式优势**：O(N) 复杂度（N=10 档），无需采样，数值稳定。

### 3.7 通用算子清单（全部教科书公式）

| 算子 | 公式 | 用途 |
|---|---|---|
| Wasserstein-1 | CDF 差积分 | 量化损失 |
| 基尼系数 | Σ\|x_i - x_j\| / (2N²·mean) | 集中度 |
| Shannon 熵 | -Σ p·log(p) | 分布不确定性 |
| HHI | Σ p² | 集中度（赫芬达尔） |
| KL 散度 | Σ p·log(p/q) | 分布差异（不对称） |
| JS 散度 | 0.5·KL(p\|m) + 0.5·KL(q\|m) | 分布差异（对称） |
| Hurst 指数 | R/S 法 | 持续性 |
| 分形维数 | box-counting | 多尺度自相似 |
| OLS 斜率 | 最小二乘 | 趋势 |
| rolling std | 滑动窗口标准差 | 波动率 |

---

## 4. 三个分支详细设计

### 4.1 分支 X：瞬时盘口多尺度量化器（78 因子）

**核心**：每个 snapshot 瞬时计算，σ = σ_spread

**因子族**：

**(a) 单分布量化损失（6 分布 × 8 度量 = 48 个）**

对每个分布 d ∈ {ask_S1, bid_S1, ask_S2, bid_S2, ask_S3, bid_S3}：
- `qX_wasserstein_{d}`：Laplace(mid, σ_spread) vs 挂单分布的 W1 距离
- `qX_gini_{d}`：挂单 vol 的基尼系数
- `qX_entropy_{d}`：Shannon 熵（归一化到 [0,1]）
- `qX_hhi_{d}`：赫芬达尔指数
- `qX_concentration_{d}`：前 2 档 vol 占总 vol 比例
- `qX_peak_pos_{d}`：最大 vol 所在档位（1-10，S1）/（1-5，S2）/（1-2，S3）
- `qX_skew_{d}`：分布偏度
- `qX_kurt_{d}`：分布峰度

**(b) 跨尺度曲线特征（3 尺度 × 2 边 × 4 特征 = 24 个）**

对每个边 side ∈ {ask, bid}：
- `qX_wasserstein_slope_{side}`：W1 随尺度粗化（S1→S2→S3）的线性斜率
- `qX_gini_slope_{side}`：基尼随尺度的斜率
- `qX_entropy_slope_{side}`：熵随尺度的斜率
- `qX_curve_area_{side}`：W1 曲线（3 尺度）下面积

**(c) 跨边不对称特征（6 个）**

- `qX_ask_bid_wasserstein_diff_S{1,2}`：S1/S2 尺度的 ask-bid W1 差（2 个；去掉 S3，因 S3 只有 2 档信息量低）
- `qX_ask_bid_gini_diff_S{1,2}`：S1/S2 尺度的 ask-bid 基尼差（2 个；同上理由）
- `qX_ask_bid_concentration_ratio`：ask vs bid 前 2 档集中度比
- `qX_ask_bid_peak_shift`：ask 峰值档位 - bid 峰值档位（S1 尺度）

**因子数小计**：48 + 24 + 6 = **78**

**输出形态**：分支 X 的 80 个因子值，是当天全天所有 snapshot 的"日内统计"（如 W1 的均值/std/趋势等）。具体地：
- 单分布度量取"日内均值"作为日频因子值
- 跨尺度/跨边特征取"日内均值"
- 不做时序展开（时序展开在分支 Y 做）

### 4.2 分支 Y：滚动窗口多尺度量化器（150 因子）

**核心**：5min 滚动窗口，σ = σ_short，捕捉时序动态

**因子族**：

**(a) 单分布量化损失 + 时序统计（6 分布 × 15 度量 = 90 个）**

对每个分布 d，在 5min 窗口内计算：
- 8 个基础度量（同分支 X 的 a，取窗口内均值）
- 7 个时序统计：
  - `qY_wasserstein_mean_{d}` / `qY_wasserstein_std_{d}`：窗口内 W1 序列的均值/std
  - `qY_wasserstein_trend_{d}`：W1 线性趋势斜率
  - `qY_wasserstein_autocorr_{d}`：W1 的 1 阶自相关
  - `qY_wasserstein_hurst_{d}`：W1 的 Hurst 指数
  - `qY_gini_mean_{d}` / `qY_gini_std_{d}`：基尼均值/std

**(b) 跨尺度曲线特征 + 时序（3 尺度 × 2 边 × 10 特征 = 60 个）**

对每个尺度对 × 边：
- 5 个静态特征（同分支 X 的 b，取窗口内均值）
- 5 个时序特征：
  - `qY_slope_mean_{side}` / `qY_slope_std_{side}`：W1 斜率序列的均值/std
  - `qY_curve_area_mean_{side}` / `qY_curve_area_std_{side}`：面积均值/std
  - `qY_slope_trend_{side}`：斜率序列本身的变化趋势

**因子数小计**：90 + 60 = **150**

**输出形态**：分支 Y 的 150 个因子值，是当天所有 5min 窗口的"窗口间统计"（如全天各窗口 W1 均值的均值/std）。

### 4.3 分支 Z：双时间尺度 + 多 σ 估计（252 因子）

**核心**：瞬时 + 5min 双窗口，3 种 σ（spread/short/early）

**因子族**：

**(a) 单尺度量化损失（3σ × 3 尺度 × 2 窗口 × 8 度量，取 144 个）**

对每个 σ ∈ {spread, short, early} × 尺度 S{1,2,3} × 窗口 w ∈ {inst, roll5m}：
- 8 个基础度量（W1 / 基尼 / 熵 / HHI / 集中度 / 峰位 / 偏度 / 峰度）
- 取 ask + bid 两边的均值，共 3×3×2×8 = 144 个

**(b) 多尺度曲线特征（3σ × 2 窗口 × 10 特征 = 60 个）**

对每个 σ × 窗口：
- W1 斜率 / 曲率（二阶差分）/ 拐点尺度 / 单调性（符号变化次数）/ 面积
- 分形维数 / 极值尺度 / 跨尺度方差 / 凹凸性 / 拐点 W1 值

**(c) σ 敏感度（3 尺度 × 2 窗口 × 3 对比 = 18 个）**

对每个尺度 × 窗口：
- `qZ_sigma_diff_spread_short_S{s}_{w}`：σ_spread vs σ_short 的 W1 差值
- `qZ_sigma_diff_short_early_S{s}_{w}`：σ_short vs σ_early 的 W1 差值
- `qZ_sigma_diff_spread_early_S{s}_{w}`：σ_spread vs σ_early 的 W1 差值

**(d) 跨尺度互信息（2 窗口 × 3 尺度对 × 2 度量 = 12 个）**

对每个窗口 × 尺度对（S1-S2, S2-S3, S1-S3）：
- `qZ_js_divergence_{pair}_{w}`：JS 散度
- `qZ_kl_divergence_{pair}_{w}`：KL 散度（ask→bid 方向）

**(e) 跨窗口对比（3σ × 3 尺度 × 2 度量 = 18 个）**

对每个 σ × 尺度：
- `qZ_window_diff_wasserstein_{sigma}_S{s}`：瞬时 vs 5min 的 W1 差
- `qZ_window_ratio_wasserstein_{sigma}_S{s}`：瞬时 / 5min 的 W1 比

**因子数小计**：144 + 60 + 18 + 12 + 18 = **252**

### 4.4 总因子数量

| 分支 | 因子数 | 前缀 |
|---|---|---|
| X 瞬时盘口 | 78 | `qX_` |
| Y 滚动窗口 | 150 | `qY_` |
| Z 双窗口多 σ | 252 | `qZ_` |
| **合计** | **480** | |

**`OUT_LEN = 480`**

---

## 5. 数据流

### 5.1 单股单日数据流（核心 `compute_distill_tick_full`）

```
输入：code: &str, date: i64

步骤 1：数据读取（≤200ms）
├── market = read_market_fast_inner(code, date, false, true, MAX)?
│   └── 连续竞价 snapshot（已平移），含 10 档 ask/bid
└── trade = read_trade_fast_inner(code, date, false, true, MAX)?
    └── 连续竞价 trade（用于 σ_short / σ_early）

步骤 2：共享底座计算（≤250ms）
├── orderbook = extract_orderbook_10levels(market)
│   └── 每 snapshot 的 (mid, spread, ask_vols[10], bid_vols[10])
├── tick_scales = resample_tick_scales(orderbook)
│   └── 3 尺度 × 2 边 × N snapshots 的挂单分布
├── sigma_spread_t = compute_sigma_spread(orderbook)  # 瞬时序列
├── sigma_short_t = compute_sigma_short(trade, window=5min)  # 5min rolling
├── sigma_early = compute_sigma_early(trade, first_30min_window)  # 标量
└── rolling_orderbook = compute_rolling_cumulative_orderbook(market, window=5min)
    └── 5min 累积挂单 vol 分布

步骤 3：分发到 3 个分支（≤430ms）
├── branch_X::compute(orderbook, tick_scales, sigma_spread_t)
│   → Vec<f32> 长度 78
├── branch_Y::compute(rolling_orderbook, tick_scales, sigma_short_t)
│   → Vec<f32> 长度 150
└── branch_Z::compute(orderbook, rolling_orderbook, tick_scales,
│                     sigma_spread_t, sigma_short_t, sigma_early)
    → Vec<f32> 长度 252

步骤 4：拼接（≤50ms）
└── out = Vec::with_capacity(480)
    ├── extend(branch_X_out)
    ├── extend(branch_Y_out)
    └── extend(branch_Z_out)
    → 长度 = 480
```

### 5.2 批量数据流（生产级别）

通过 `run_factor_pipeline(pipeline="distill_tick", tasks=[[date, code], ...])` 批量调用：
- worker 进程调 `pipeline_distill_tick(date, code, params, _td, expected_len=480)`
- 输出：`(n_stocks × n_days, 480)` 的因子面板

### 5.3 数据缺口处理

| 情况 | 处理 |
|---|---|
| market 数据为空 | 480 因子全 NaN |
| market 记录数 < 100（交易日不完整） | 480 因子全 NaN |
| 某时刻 10 档挂单全 0（涨跌停） | 该 snapshot 的分支 X 因子 NaN，不影响其他时刻 |
| trade 数据为空 | σ_short / σ_early = NaN，涉及这两个 σ 的因子 NaN |
| 当天前 30min trade 不足 30 笔 | σ_early = NaN |
| 5min 窗口内 snapshot 不足 10 个 | 分支 Y/Z 该窗口的因子 NaN |
| σ_spread = 0（涨停/跌停锁死） | 该时刻分支 X 因子 NaN |
| mid ≤ 0（异常数据） | 整行剔除 |

### 5.4 关键时序约定

- `adjust_afternoon=true`（与 ①+⑥ 一致）
- 连续竞价段（平移后）：day_ts = time_sec % 86400 ∈ [34200, 48420]
- 5min 滚动窗口：从开盘向后滑，窗口内 snapshot 数随时间增长（开盘瞬间窗口可能不足，此时 NaN）
- "前 30min"定义：day_ts ∈ [34200, 36000]（9:30-10:00）

---

## 6. 误差处理与数值稳定性

### 6.1 数值稳定性

| 风险点 | 防护 |
|---|---|
| Wasserstein 计算时 σ=0（涨跌停锁死） | 前置 check，σ≤1e-8 时该时刻因子 NaN |
| 挂单 vol 全 0（盘口空） | 分布归一化时分母为 0，置 NaN |
| Wasserstein 数值溢出（σ 极大） | 设上限 σ_max=0.1（10% 价格波动），超过截断 |
| 基尼系数计算时分布只有 1 个非零点 | G=1.0（完全集中），不算 NaN |
| Shannon 熵遇到 log(0) | 用 `x·log(x+ε)`，ε=1e-12 |
| KL/JS 散度分母为 0 | 加 ε 平滑 |
| Hurst 指数计算时序列 < 20 个点 | 返 NaN |
| 5min 窗口 snapshot < 10 | 该窗口因子 NaN |
| 滚动 std 窗口不足 | 样本 < 5 时 NaN |
| 分形维数 box-counting 框数不足 | < 3 个尺度时 NaN |
| 浮点除零（mid=0） | 前置 check，mid≤0 时整行 NaN |

### 6.2 确定性约束（CLAUDE.md 强制）

- **禁止**用 HashMap 遍历顺序参与下游计算
- 挂单分布的桶顺序用固定 `Vec<f32>`（按档位 1→10 排序），不用 HashMap
- 多 tick 尺度重采样用显式档位合并（`Vec<usize>` 固定顺序）
- 3 种 σ 的计算顺序固定（spread → short → early）
- 3 分支的调用顺序固定（X → Y → Z）
- 所有参数网格用 `const` 数组

### 6.3 边界条件

- 单股单日数据为空：返回全 NaN 的 Vec（长度 480）
- 部分数据缺失：相关因子 NaN，不 panic
- 时间戳异常（time_sec 超出 86400）：跳过该条记录

---

## 7. 性能预算（000001 / 20220819 ≤ 1s）

| 模块 | 预算 | 优化策略 |
|---|---|---|
| `read_market_fast_inner` | ≤ 100ms | 已优化的 CSV 解析 |
| `read_trade_fast_inner` | ≤ 100ms | 同上 |
| 共享底座（10 档提取 + 3 尺度重采样 + 3 种 σ + 5min 累积） | ≤ 250ms | 向量化遍历，σ_early 一次性算 |
| 分支 X（78 因子，瞬时） | ≤ 80ms | 每 snapshot 6 分布并行矩阵化 |
| 分支 Y（150 因子，5min 滚动） | ≤ 150ms | 滚动窗口增量更新 |
| 分支 Z（252 因子，双窗口多 σ） | ≤ 200ms | 复用底座的 tick_scales 和 σ，不重算 |
| 拼接 + 返回 | ≤ 50ms | `Vec::with_capacity(480)` |
| **合计** | **≤ 930ms** | 留 70ms 安全垫 |

**强制约束**：
- 全程**不并行**（用户 CLAUDE.md 明确）
- Wasserstein 用 1D 闭式（CDF 差积分），不做采样
- 5min 滚动用增量更新，不重复扫描
- 禁止 `format!` 在热路径

### 7.1 性能验收红线

- **硬性**：`compute_distill_tick_full("000001", 20220819)` 在 < 1.0 秒内返回
- 验证方式：`time.perf_counter()` 计时，运行 3 次取中位数
- 失败处理：用 `factor-performance-checker` agent 诊断瓶颈

**优化备选**（若 930ms 超 1s 红线）：
- 分支 Z 的单尺度量化损失（144 个）减少 σ×尺度组合：只保留 2σ × 3 尺度 × 2 窗口 = 96 个
- 分支 Y 的时序统计减少：只保留 mean/std/trend 3 个时序统计
- 5min 累积挂单 vol 用近似（每 10 个 snapshot 采样一次）

---

## 8. 测试设计

### 8.1 单元测试（每分支独立）

| 测试 | 验证内容 |
|---|---|
| `test_wasserstein_synthetic_laplace` | 已知 Laplace vs 均匀分布的 W1 解析解 vs 计算值，误差 < 1% |
| `test_gini_known_distributions` | 均匀分布 G≈0.9（10 档），单点集中 G=1.0，全均 G=0 |
| `test_entropy_normalization` | 归一化熵 ∈ [0, 1]，全均=1，单点=0 |
| `test_tick_scale_resample` | S1(10 档) → S2(5 档) → S3(2 档) 的 vol 守恒（合并后总和不变） |
| `test_sigma_spread_zero_case` | 涨跌停（spread=0）时分支 X 因子全 NaN |
| `test_branch_X_determinism` | 相同输入运行 2 次，78 因子完全一致 |
| `test_branch_Y_rolling_window` | 5min 窗口边界正确（开盘瞬间窗口不足时 NaN） |
| `test_branch_Z_multi_sigma` | 3 种 σ 都非负，σ_spread ≤ σ_short 通常成立 |

### 8.2 集成测试

| 测试 | 验证内容 |
|---|---|
| `test_single_stock_single_day` | 真实 L2 跑 000001/20220819，输出 480 因子（NaN < 30%） |
| `test_against_python_baseline` | 与简化 Python 实现对比，分支 X 的 W1 误差 < 5% |
| `test_pipeline_python_consistency` | `py_distill_tick` 与 `run_factor_pipeline(pipeline="distill_tick")` 逐字节相同 |

### 8.3 性能测试

```python
# tests/test_distill_tick_perf.py
import rust_pyfunc as rp
import time

def test_perf_000001_20220819():
    rp.py_distill_tick("000001", 20220819)  # 预热
    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        rp.py_distill_tick("000001", 20220819)
        times.append(time.perf_counter() - t0)
    median = sorted(times)[1]
    assert median < 1.0, f"性能超时：中位数 {median:.3f}s > 1.0s"
```

### 8.4 回测验证（后续阶段）

| 测试 | 验证内容 |
|---|---|
| `test_backtest_ic_distribution` | 480 因子日频 IC 分布合理（多数 \|IC\| 在 0.01-0.05） |
| `test_backtest_top_factors` | Top 30 IC 因子分布在 3 个分支（不应集中在 1 个分支） |
| `test_correlation_with_distill_1` | 与 ①+⑥ 的 290 因子的相关性 < 0.5（确认两族互补） |

---

## 9. 实现里程碑

| 阶段 | 里程碑 | 验收标准 |
|---|---|---|
| M1 | 共享底座 + 分支 X 跑通 | `py_distill_tick` 返回长度 78，单股单日 < 0.4s |
| M2 | 分支 Y 加入 | 返回长度 228（X+Y），单股单日 < 0.7s |
| M3 | 分支 Z 加入 | 返回长度 480（全部），单股单日 < 1s |
| M4 | pipeline 注册 + worker 分发 | `run_factor_pipeline(pipeline="distill_tick")` 可批量跑 |
| M5 | 一致性验证 | py 入口与 pipeline 入口逐字节相同 |
| M6 | 回测验证 | 480 因子 IC 分布合理，Top 因子分布均衡 |
| M7 | 与 ①+⑥ 联合回测 | 770 因子（290+480）联合筛选，确认两族互补 |

---

## 10. 文件清单

### 10.1 新建文件

| 文件 | 行数估计 | 内容 |
|---|---|---|
| `src/distill_tick_metrics.rs` | ~2000 | 核心 + 名字 + Python 入口 |
| `tests/test_distill_tick.py` | ~60 | 集成测试 |
| `tests/test_distill_tick_perf.py` | ~20 | 性能测试 |

### 10.2 修改文件

| 文件 | 改动 |
|---|---|
| `src/lib.rs` | 加 `pub mod distill_tick_metrics;` + 注册 2 个 py 函数 |
| `src/factor_pipeline.rs` | 加 `pipeline_distill_tick` 函数 + 5 处注册点 |
| `src/bin/worker_pipeline.rs` | 加 worker 分发 |
| `python/rust_pyfunc/__init__.pyi` | 加类型声明 |

### 10.3 不修改文件

- 任何特质模块（`hawkes_analysis.rs` / `ghost_market_maker.rs` 等）
- `fast_csv_reader.rs`（仅调用，不修改）
- `distill_metrics.rs`（①+⑥ 的模块，完全隔离）

---

## 11. 风险与缓解

| 风险 | 缓解 |
|---|---|
| 1s 性能红线难达（480 因子比 ①+⑥ 的 290 多） | 优先实现 M1-M2（X+Y=228 因子），若 M3（分支 Z）超时则按 §7.1 备选方案缩减 |
| Laplace 假设不完美 | 3 种 σ 互相对冲（spread/short/early 捕捉不同时间尺度），即使一种 σ 下 Laplace 不完美，其他 σ 仍提供信号 |
| 3 尺度（10/5/2 档）信息量有限 | 后续可扩展到非均匀合并（如 S2 = 档 1-2 + 档 3-5 + 档 6-10） |
| 分支 X/Y/Z 因子高相关 | 不处理，让回测自然筛选；符合"人海战术" |
| 480 因子中很多无效 | 预期之内，靠 IC 筛选 |
| Wasserstein 闭式解数值误差 | 单元测试用已知解析解验证（误差 < 1%） |

---

## 12. 后续扩展路径（不在本 spec 范围）

- **非均匀 tick 合并**：按盘口形态自适应合并档位
- **方向 B（大单 vs 小单）**：把分支扩展到"知情 vs 散户"的量化器视角
- **方向 C（成交 vs 挂单）**：把 teacher 换成成交价分布
- **参数化**：若需调参，再建 `DistillTickParams` 结构并改 bincode

预期最终因子数：本族 480 → 扩展后 800-1500。

---

## 13. 参考实现

- **范式参考**：`src/observable_order_metrics.rs` + `src/factor_pipeline.rs::pipeline_observable_order`
- **数据读取参考**：`src/fast_csv_reader.rs::read_market_fast_inner` / `read_trade_fast_inner`
- **①+⑥ 参考**：`docs/superpowers/specs/2026-07-09-distillation-factor-design.md`（即将实现）
- **skill 参考**：`rust-pipeline-factor-pattern`
- **编译命令**：`timeout 600s bash alter.sh 2>&1`

---

**本 spec 到此结束。待用户审阅后，调用 writing-plans skill 生成实现计划。**
