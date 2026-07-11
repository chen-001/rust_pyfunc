# 蒸馏思想因子（distill）设计文档

**日期**：2026-07-09
**作者**：chenzongwei + Claude（brainstorming 协作）
**状态**：设计完成，待用户审阅 → 转 writing-plans

---

## 0. 背景与动机

### 0.1 LLM 蒸馏思想的核心机制

大模型蒸馏（Knowledge Distillation）包含若干核心机制：
- **Soft Label**：teacher 输出的概率分布（而非 hard label），携带"不确定性"信息
- **温度参数 T**：调控分布的尖锐程度
- **Feature Alignment**：对齐 teacher/student 的中间层表示
- **Catastrophic Forgetting**：持续学习中"学了新的忘了旧的"
- **Online Distillation**：多模型互相教
- **Cross-Model Distillation**：跨架构蒸馏（大→小、teacher→student 网络）

### 0.2 迁移到 A 股市场

把上述机制映射到 A 股 L2/分钟数据上，可以构造一整套"投资者学习股票规律"的因子族。详见研究笔记：
`/home/chenzongwei/pythoncode/蒸馏的应用/蒸馏思想因子集.md`

该笔记列出了 8 大类、18 个细分方向。本 spec 锁定其中**最有代表性的组合**做深：
- **① 集合竞价 = Few-Shot Teacher**：集合竞价是开盘前全市场一次性撮合得到的"少样本、高密度预测"
- **⑥ 灾难性遗忘**：早盘 teacher 信号在午盘/尾盘的失效速度

二者组合形成一条完整的"投资者学习 → 遗忘"曲线，是蒸馏思想最完整的体现。

### 0.3 为什么选 ①+⑥

- **概念新颖**：集合竞价在文献里几乎只被当作价格点，从未被显式当作 teacher 模型
- **信息丰富**：集合竞价可展开成 5+ 维 teacher 信号（不止价格回归速度）
- **单股 L2 闭环**：无横截面依赖，不需要板块/概念映射表，工程可控
- **验证抽象框架**：跑通后可批量复制到笔记中的其他 16 个方向

---

## 1. 核心约束

| # | 约束 | 来源 |
|---|---|---|
| 1 | **不依赖特质模块** | 用户要求。`hawkes_analysis` / `ghost_market_maker` / `passive_order_features` / `observable_order_metrics` 等都禁用，因为它们携带过强先验 |
| 2 | **只用成熟通用算子** | OLS 回归、EWMA、softmax、KL 散度、基尼系数、赫芬达尔指数、Hurst 指数（R/S 法）、Kyle lambda、Hawkes λ/η 闭式估计等教科书算子可直接实现 |
| 3 | **纯 Rust 实现** | 数据读取 + 处理 + 因子输出全 Rust，采用 rust-pipeline-factor-pattern 三层结构 |
| 4 | **不并行** | 用户 CLAUDE.md 明确："优化函数性能时不要使用并行" |
| 5 | **确定性** | 不用 HashMap 遍历顺序参与下游计算；需要遍历时先排序 |
| 6 | **性能红线** | 000001 / 20220819 单股单日 ≤ 1.0 秒 |
| 7 | **首期只做日频横截面因子** | 每天每股 1 个值，不做 T0 日内 |

---

## 2. 总体架构

### 2.1 三层结构（rust-pipeline-factor-pattern 范式）

```
src/distill_metrics.rs (新文件)
├── compute_distill_full(code, date) -> io::Result<Vec<f32>>   ← 核心唯一真相源
│   ├── read_market_fast_inner(...)   ← L2 snapshot（含集合竞价段）
│   ├── read_trade_fast_inner(...)    ← L2 trade
│   ├── extract_auction_teacher()     ← 5 维 teacher 信号
│   ├── compute_student_series()      ← student 度量（OFI / 主买占比 / 价格）
│   ├── branch_a::compute()           ← 50 因子（单维曲线拟合）
│   ├── branch_b::compute()           ← 70 因子（多维独立遗忘）
│   ├── branch_d::compute()           ← 120 因子（分阶段衰减）
│   ├── branch_e::compute()           ← 50 因子（双层 Hawkes）
│   └── 拼接 → Vec<f32> 长度 = OUT_LEN (290)
├── distill_names() -> Vec<String>    ← 名字单一源（Rust 生成）
├── py_distill(code, date)            ← Python 入口（抛 PyIOError）
└── py_distill_names()                ← Python 拿名字

src/factor_pipeline.rs
└── pipeline_distill(date, code, params, _td, expected_len) -> Vec<f32>
                                      ← pipeline 入口（吞异常返 NaN）
```

### 2.2 5 处注册清单

| # | 文件 | 改动 |
|---|---|---|
| 1 | `src/distill_metrics.rs` | 新建：核心 + 名字 + Python 入口（~1500 行） |
| 2 | `src/lib.rs` | `pub mod distill_metrics;` + 注册 `py_distill` + `py_distill_names` |
| 3 | `src/factor_pipeline.rs` | `use crate::distill_metrics;` + `known` 加 `"distill"` + 参数解析 + 线程分发（~645 行）+ v6 入口（~1351/1394 行）+ 新增 `pipeline_distill` 函数 |
| 4 | `src/bin/worker_pipeline.rs` | `use` 引入 + 分发 `else if pipeline_name == "distill"` |
| 5 | `python/rust_pyfunc/__init__.pyi` | `def py_distill(...) -> List[float]: ...` + `__all__` 加名字 |

### 2.3 参数策略

**不引入新 Params**。所有参数（窗口网格、阈值网格、时段划分）在 `distill_metrics.rs` 用 `const` 写死。

**理由**：
- 建新 Params 要改 `TaskMessage::Init` 的 bincode 结构，风险大
- 符合用户 CLAUDE.md "无必要不写兼容性代码" 原则
- 后续如需调参，再做参数化

### 2.4 共享数据底座

| 组件 | 输入 | 输出 | 算子 |
|---|---|---|---|
| `read_market_fast_inner`（无平移版） | code, date, with_afternoon_adjust=false | `Vec<MarketRecord>` 含集合竞价段 | 已有 |
| `read_market_fast_inner`（平移版） | code, date, with_afternoon_adjust=true | `Vec<MarketRecord>` 连续竞价专用 | 已有 |
| `read_trade_fast_inner`（平移版） | code, date, with_afternoon_adjust=true | `Vec<TradeRecord>` | 已有 |
| `extract_auction_teacher` | market_raw, trade, market_prev | `TeacherSignals`（5 维） | 按 time_sec % 86400 过滤集合竞价段 |
| `compute_student_series` | market, trade | `StudentData`（OFI / 主买占比 / 价格） | 向量化遍历 |

**关键时序约定**：
- `adjust_afternoon=true`（与现有 L2 处理一致）
- 4 时段划分（用平移后的时间）：
  - T1: 9:30-10:00（开盘冲击期）
  - T2: 10:00-11:30（早盘消化期）
  - T3: 11:30-12:30（午盘期，平移后）
  - T4: 12:30-14:27（尾盘期，平移后）
- 集合竞价段（9:15-9:25）不受 `adjust_afternoon` 影响

---

## 3. Teacher 信号定义（核心，所有分支共用）

### 3.1 数据来源（重要）

`fast_csv_reader` 的 `read_market_fast_inner` / `read_trade_fast_inner` 两个函数都有 `with_afternoon_adjust` 参数：
- `with_afternoon_adjust=true`：会过滤掉集合竞价段（9:15-9:25），只保留连续竞价
- `with_afternoon_adjust=false`：保留全集合竞价段，但时段未做下午平移

且集合竞价段（9:15-9:25）的 `MarketRecord` 里 `last_prc/volume/turnover` **都是 0**（因为还没成交），只有 `ask_prcs[0]` / `bid_prcs[0]` 有值（虚拟开盘参考价）。

**因此数据读取策略必须分两次**：

| 用途 | 调用方式 | 得到 |
|---|---|---|
| 集合竞价 teacher 提取 | `read_market_fast_inner(code, date, true, false, MAX)` | 含集合竞价段（last_prc 全 0，但 ask1/bid1 有值） |
| 连续竞价 student 计算 | `read_market_fast_inner(code, date, false, true, MAX)` | 已过滤涨跌停 + 已平移下午 |
| 连续竞价 trade | `read_trade_fast_inner(code, date, false, true, MAX)` | 已平移 |

**昨收 yclose 获取**：必须从**前一交易日**的 market 数据读，调用 `read_market_fast_inner(code, prev_date, true, false, MAX)`，取连续竞价段最后一条 `last_prc`。前一交易日通过 `trading_day_utils`（如不可用则从 trade_dates 列表反查）获得。

### 3.2 5 维 teacher 信号

| 维度 | 符号 | 定义 | 物理含义 |
|---|---|---|---|
| 价格维 | `P_auct` | 9:24-9:25 末段集合竞价 mid price = (ask1+bid1)/2 | teacher 的均值预测 |
| 成交量维 | `V_auct` | 9:30 开盘瞬间（前 5 秒）trade 累计成交量 | teacher 的置信度反指标（反映集合竞价撮合规模） |
| 不确定性维 | `σ_auct` | 9:20-9:25 集合竞价 mid price 的标准差 | teacher 的不确定性 |
| 修正方向维 | `Δ_auct` | `logret(mid_end_9:25 / mid_open_9:20)` | teacher 在撮合末段对自己预测的修正 |
| 历史偏离维 | `δ_yc` | `logret(P_auct / yclose)` | teacher 对"昨日学习结果"的修正强度 |

### 3.3 时间戳约定

`MarketRecord.time_sec` / `TradeRecord.time_sec` 是 **Unix epoch 秒**（含日期信息，不是当天秒数）。
过滤集合竞价段用：`(time_sec % 86400) ∈ [33300, 33900]`（即 9:15-9:25 当天秒数）。
过滤连续竞价段（平移后）用：`(time_sec % 86400) ∈ [34200, 48420]`（9:30 到平移后的 14:27）。

### 3.4 异常处理

| 情况 | 处理 |
|---|---|
| 集合竞价段记录数 < 5 | 所有 teacher 信号 NaN，全部 290 因子 NaN |
| `σ_auct` 样本 < 3 | `σ_auct = NaN`，涉及 σ 的因子 NaN |
| `Δ_auct` 计算时 `mid_open_9:20` 缺失 | `Δ_auct = NaN` |
| 昨收缺失（前一交易日数据不可得） | `δ_yc = NaN`，其他 4 维正常 |
| `V_auct` 计算时 9:30 前 5 秒无 trade | `V_auct = NaN`，扩展到 9:30 前 30 秒仍无则置 NaN |

---

## 4. 四个分支详细设计

### 4.1 分支 A：单维曲线拟合（50 因子）

**Student 信号**：
`ε(t) = logret(P_t / P_auct)`，t ∈ [0, 240min]，1min 重采样

**参数网格**：
- 收敛窗口 k ∈ {5, 15, 30, 60, 120} min（5 值）
- 突破阈值 thr ∈ {0.3%, 0.5%, 1%, 2%, 3%}（5 值）

**因子族**：

**(a) 基础拟合（5 窗口 × 5 度量 = 25 个）**

对每个 k，取 ε(t) 在 [0, k] 区间，拟合 `ε(t) = ε(0)·exp(−t/τ_A)`：
- `dA_tau_k{k}`：收敛时间常数 τ（分钟），上限 240，下限 0.5
- `dA_eps0_k{k}`：初始偏离 ε(0)
- `dA_resid_std_k{k}`：拟合残差标准差
- `dA_r2_k{k}`：拟合 R²
- `dA_hurst_eps_k{k}`：ε 在 [0,k] 的 Hurst 指数（R/S 法）

**(b) 统计降维（10 个）**

对全日 ε(t) 序列遍历：
- `dA_eps_std / min / max / skew / kurt / q10 / q25 / q50 / q75 / q90`（不含 mean，避免与 q50 高度冗余）

**(c) 突破特征（5 阈值 × 3 度量 = 15 个）**

对每个 thr：
- `dA_break_cnt_thr{thr}`：开盘后 30min 内突破 [P_auct ± thr] 的次数
- `dA_break_first_time_thr{thr}`：首次突破时间（相对 9:30 的分钟数）
- `dA_break_rebound_thr{thr}`：突破后 5min 内的回弹幅度

**因子数小计**：25（基础拟合）+ 10（统计降维，去掉与 q50 重复的 mean）+ 15（突破特征）= **50**

### 4.2 分支 B：多维独立遗忘（70 因子）

**Student 信号**：5 维 teacher 各自对应的 student 度量
- 价格维：`ε_price(t) = logret(P_t / P_auct)`
- 成交量维：`ε_vol(t) = V_continuous(t) / V_auct`（连续竞价累计成交量比）
- 信念维：`ε_belief(t) = rolling_corr(主买占比_t, Δ_auct, window=15min)`
- 不确定性维：`ε_sigma(t) = rolling_std(logret(P_t), window=5min)`
- 历史偏离维：`ε_yc(t) = rolling_beta(logret(P_t), δ_yc, window=15min)`

**因子族**：

**(a) 每维独立基础（5 维 × 7 度量 = 35 个）**

对每维 v ∈ {price, vol, belief, sigma, yc}，拟合 `ε_v(t) = ε_v(0)·exp(−t/τ_v)`：
- `dB_tau_{v}`：τ_v
- `dB_eps0_{v}`：ε_v(0)
- `dB_hurst_{v}`：ε_v 的 Hurst
- `dB_half_life_{v}`：半衰期 t_half = τ·ln2
- `dB_residual_1min_{v}`：1min 时残留强度 |ε_v(1)|
- `dB_residual_15min_{v}`：15min 时残留强度
- `dB_residual_60min_{v}`：60min 时残留强度

**(b) 每维细化窗口 τ（5 维 × 4 窗口 = 20 个）**

对每维 v，分别在 k ∈ {5, 15, 30, 60} 分钟窗口内单独拟合 τ：
- `dB_tau_{v}_k5 / k15 / k30 / k60`

**(c) 跨维组合（10 个）**

- `dB_tau_mean / std / min / max / range / skew / kurt`：τ 向量的 7 个统计量
- `dB_tau_rank_mean`：5 个 τ 在横截面排名（当日全市场）的均值
- `dB_tau_rank_std`：同上的标准差
- `dB_tau_cv`：τ 的变异系数 std/mean

**因子数小计**：35 + 20 + 9 + 6 = **70**

**(d) 维度遗忘顺序（6 个）**

- `dB_first_forgotten_dim`：τ 最小的维度编号（1-5）
- `dB_last_forgotten_dim`：τ 最大的维度编号（1-5）
- `dB_forgetting_order_entropy`：遗忘顺序的排列熵
- `dB_dim_consistency`：5 维 τ 排序在横截面的稳定性
- `dB_dim_extremeness`：max(τ)/min(τ) 的比值
- `dB_dim_balance`：5 维 τ 的基尼系数（越接近 0 越均衡）

### 4.3 分支 D：分阶段衰减（120 因子，主力分支）

**时段划分**：
- T1: 9:30-10:00（30min，开盘冲击）
- T2: 10:00-11:30（90min，早盘消化）
- T3: 11:30-12:30（60min，午盘，平移后）
- T4: 12:30-14:27（117min，尾盘，平移后）

**Teacher 信号组合**：5 维 teacher 的非空子集
- 单维：5 种（P_auct / V_auct / σ_auct / Δ_auct / δ_yc）
- 2 维组合：选信息量大的 5 种（P+Δ / P+δ / V+Δ / V+δ / σ+Δ）
- 3 维组合：选 3 种（P+V+Δ / P+V+δ / P+Δ+δ）
- 4+ 维组合：选 2 种（P+V+Δ+δ / 全 5 维加权）
- **合计 15 种组合**

**因子族**：

**(a) IC 度量（4 时段 × 15 组合 = 60 个）**

对每个时段 T_i 和每个组合 C_j，计算 IC(teacher_C_j, 该时段累计收益)：
- `dD_IC_T{T_i}_C{j}`：原始 IC 值（4×15=60 个）

**(b) 衰减曲线（15 组合 × 4 度量 = 60 个）**

对每个组合 C_j：
- `dD_decay_tau_C{j}`：4 时段 IC 序列的指数拟合 τ
- `dD_residual_ratio_C{j}`：IC_T4 / IC_T1
- `dD_shape_class_C{j}`：形状分类（0=单调衰减, 1=振荡, 2=反转）
  - 判定：IC 序列符号变化次数 = 0 → 0；1-2 次 → 1；≥3 次 → 2
- `dD_slope_T1_T4_C{j}`：(IC_T4 - IC_T1) / 4 线性斜率

**因子数小计**：60 + 60 = **120**

### 4.4 分支 E：双层 Hawkes 蒸馏（50 因子）

**事件定义网格**：
- 大单阈值 thr ∈ {前 10%, 5%, 1% 成交量}（3 值）
- 方向事件：主买大单 / 主卖大单（2 种）

**双层结构**：

第一层（集合竞价 → 开盘 5min）：
- 事件流：9:30-9:35 的大单成交（按 thr 过滤）
- Hawkes 参数：λ₁（基线强度）、η₁（分支比）

第二层（开盘 5min 信号 → 后续时段）：
- 事件流：9:35-15:00 的大单成交
- 用开盘 5min OFI 作为"二级 teacher"标记
- Hawkes 参数：λ₂、η₂

**闭式估计（不做 EM 迭代）**：
```
λ = 事件总数 / 时段长度（秒）
η = Σ N_t / Σ 事件总数，其中 N_t = 每个事件后触发窗口内的后续事件数
```

**因子族**：

**(a) Hawkes 参数（4 事件 × 4 参数 = 16 个）**

事件 ∈ {主买_10%, 主卖_10%, 主买_5%, 主卖_5%}（共 4 个，按"方向 × 阈值"枚举）：
- `dE_lambda1_{event}`、`dE_eta1_{event}`
- `dE_lambda2_{event}`、`dE_eta2_{event}`

**(b) 衍生 per-event（4 事件 × 5 度量 = 20 个）**

对每个 event：
- `dE_absorption_strength_{event}`：λ₁ · η₁
- `dE_propagation_strength_{event}`：λ₂ · η₂
- `dE_learning_ratio_{event}`：absorption / propagation
- `dE_eta_decay_{event}`：η₁ - η₂（学习强度的衰减）
- `dE_lambda_ratio_{event}`：λ₁ / λ₂

**(c) 跨事件聚合（10 个）**

- `dE_eta_cv_buy`：主买方向 η₁ 在 3 阈值下的变异系数
- `dE_eta_cv_sell`：主卖方向 η₁ 在 3 阈值下的变异系数
- `dE_eta_cv_buy_2`：主买方向 η₂ 在 3 阈值下的变异系数
- `dE_eta_cv_sell_2`：主卖方向 η₂ 在 3 阈值下的变异系数
- `dE_cross_consistency_10_5`：10% 与 5% 阈值下 η₁ 的相关
- `dE_cross_consistency_5_1`：5% 与 1% 阈值下 η₁ 的相关
- `dE_direction_asymmetry_10`：主买 vs 主卖 η₁ 在 10% 阈值下的不对称性
- `dE_direction_asymmetry_5`：同上 5% 阈值
- `dE_absorption_strength_mean`：4 事件 absorption_strength 的均值
- `dE_propagation_strength_mean`：4 事件 propagation_strength 的均值

**(d) 形状分类（4 个）**

- `dE_pattern_label`：双层 Hawkes 触发模式的聚类标签（0=正常, 1=暴涨式吸收, 2=慢热式, 3=反转式）
- `dE_pattern_score`：聚类置信度
- `dE_two_layer_balance`：第一层与第二层事件密度比
- `dE_event_volatility`：4 事件计数的变异系数

**因子数小计**：16 + 20 + 10 + 4 = **50**

### 4.5 总因子数量

| 分支 | 因子数 | 前缀 |
|---|---|---|
| A 单维曲线拟合 | 50 | `dA_` |
| B 多维独立遗忘 | 70 | `dB_` |
| D 分阶段衰减 | 120 | `dD_` |
| E 双层 Hawkes | 50 | `dE_` |
| **合计** | **290** | |

**`OUT_LEN = 290`**

---

## 5. 数据流

### 5.1 单股单日数据流（核心 `compute_distill_full`）

```
输入：code: &str, date: i64, prev_date: i64（前一交易日）

步骤 1：数据读取（≤300ms，读 3 次）
├── market_raw = read_market_fast_inner(code, date, true, false, MAX)?
│   └── 含集合竞价段（last_prc=0，但 ask1/bid1 有值）
├── market = read_market_fast_inner(code, date, false, true, MAX)?
│   └── 已过滤涨跌停 + 已平移下午（连续竞价专用）
├── trade = read_trade_fast_inner(code, date, false, true, MAX)?
└── market_prev = read_market_fast_inner(code, prev_date, true, false, MAX)?
    └── 取连续竞价段最后一条 last_prc 作为 yclose

步骤 2：集合竞价 teacher 提取（≤20ms）
├── day_ts = market_raw.time_sec.map(|t| t % 86400)
├── auct_seg = market_raw 中 day_ts ∈ [33300, 33900] 的子集
├── yclose = market_prev 连续竞价段最后一条 last_prc（缺失则 NaN）
├── teacher = TeacherSignals {
│       P_auct = (ask1_end_9:25 + bid1_end_9:25) / 2,
│       V_auct = trade 中 day_ts ∈ [34200, 34205] 的累计 volume,
│       sigma_auct = std(mid_9:20_9:25),
│       delta_auct = logret(mid_end_9:25 / mid_open_9:20),
│       delta_yc = logret(P_auct / yclose)
│   }
└── 若 auct_seg.len() < 5：返回全 NaN 的 Vec<f32>（长度 290）

步骤 3：student 度量（≤50ms）
├── cont_seg = market（已平移，直接用）
├── price_t = cont_seg.last_prc 按 1min 重采样
├── ofi_t = compute_ofi(cont_seg, window=1min)
├── active_buy_ratio_t = compute_active_buy_ratio(trade, window=1min)
└── student = StudentData { price_t, ofi_t, active_buy_ratio_t, trade }

步骤 4：分发到 4 个分支（≤600ms）
├── branch_a::compute(&teacher, &student) → Vec<f32> 长度 50
├── branch_b::compute(&teacher, &student) → Vec<f32> 长度 70
├── branch_d::compute(&teacher, &student) → Vec<f32> 长度 120
└── branch_e::compute(&teacher, &student) → Vec<f32> 长度 50

步骤 5：拼接（≤30ms）
└── out = Vec::with_capacity(290)
    ├── extend(branch_a_out)
    ├── extend(branch_b_out)
    ├── extend(branch_d_out)
    └── extend(branch_e_out)
    → 长度 = 290
```

**prev_date 获取**：
- 调用方（pipeline / py 入口）传入，不在 `compute_distill_full` 内部解决
- pipeline 模式：从 `trading_days` 列表反查（`_trading_days` 参数已传入 `pipeline_distill`）
- py 模式：用户传入或调用方负责

### 5.2 批量数据流（生产级别）

通过 `run_factor_pipeline(pipeline="distill", tasks=[[date, code], ...])` 批量调用：
- worker 进程拿到 task → 调 `pipeline_distill(date, code, params, _td, expected_len=290)`
- `pipeline_distill` 内部调 `compute_distill_full`，错误吞掉返 NaN
- 输出：`(n_stocks × n_days, 290)` 的因子面板

### 5.3 数据缺口处理

| 情况 | 处理 |
|---|---|
| 集合竞价缺失（停牌、新股） | 290 因子全 NaN |
| 集合竞价段 < 5 条记录 | 290 因子全 NaN |
| 连续竞价不完整（临时停牌） | 分支 A/B/D 用现有数据拟合；分支 E 需 ≥50 笔成交才计算 |
| 昨收缺失 | δ_yc = NaN，其他 4 维正常；涉及 δ_yc 的因子 NaN |
| σ_auct 样本 < 3 | σ_auct = NaN，涉及 σ 的因子 NaN |
| Hawkes 事件数 < 10 | 该 event 的 Hawkes 因子 NaN |

---

## 6. 误差处理与数值稳定性

### 6.1 数值稳定性

| 风险点 | 防护 |
|---|---|
| 拟合 `ε(t)=ε₀·exp(−t/τ)` 时 τ→∞ | 设上限 τ_max=240min，超过则报 NaN |
| τ→0（瞬时收敛） | 设下限 τ_min=0.5min，低于用线性拟合代替 |
| σ_auct=0 | 置 NaN，跳过 σ 维度因子 |
| logret 出现 inf（P_t≤0 或 P_auct≤0） | 前置 check，价格 ≤0 整行 NaN |
| Hawkes 拟合事件不足 | 事件数 < 10 时返 NaN |
| 滚动 IC 窗口样本不足 | 样本 < 30 时返 NaN |
| 除零（基尼系数、HHI 等） | 分母为 0 时返 NaN |

### 6.2 确定性约束（CLAUDE.md 强制）

- **禁止**用 HashMap 遍历顺序参与下游计算
- 横截面排名、相关矩阵等用 dict/HashMap 时必须先 sort
- teacher 信号 5 维组合枚举：用显式 `Vec<[usize; N]>`（固定顺序），不用 HashSet 迭代
- 4 时段划分、5 窗口网格等全部用 `const` 数组

### 6.3 边界条件

- 单股单日数据为空：返回全 NaN 的 Vec（长度 290）
- 部分数据缺失：相关因子 NaN，不 panic
- 时间戳异常（time_sec 超出 86400）：跳过该条记录

---

## 7. 性能预算（000001 / 20220819 ≤ 1s）

| 模块 | 预算 | 优化策略 |
|---|---|---|
| `read_market_fast_inner`（无平移版，取集合竞价） | ≤ 100ms | 已优化的 CSV 解析 |
| `read_market_fast_inner`（平移版，连续竞价） | ≤ 100ms | 复用 CSV 缓存（同 code+date） |
| `read_trade_fast_inner` | ≤ 100ms | 同上 |
| `read_market_fast_inner`（前一交易日） | ≤ 100ms | 仅取最后一条 last_prc（如读取慢可只读末尾 N 行） |
| 集合竞价段过滤 + teacher 提取 | ≤ 20ms | 一次遍历，按 time_sec % 86400 分桶 |
| student 度量（OFI / 主买占比） | ≤ 50ms | 向量化遍历 |
| 分支 A（50 因子） | ≤ 80ms | 5 窗口 × 10 度量矩阵化 |
| 分支 B（70 因子） | ≤ 120ms | 5 维 teacher 各跑一次轻量拟合 |
| 分支 D（120 因子） | ≤ 200ms | 15 组合预计算成矩阵，4 时段 IC 用矩阵乘法 |
| 分支 E（50 因子） | ≤ 150ms | Hawkes η 用闭式估计 |
| 拼接 + 返回 | ≤ 30ms | `Vec::with_capacity(290)` |
| **合计** | **≤ 950ms** | 留 50ms 安全垫 |

**优化备选**（若 950ms 超 1s 红线）：
- 前一交易日 yclose 缓存：用 `factor_store_v5` 缓存昨收，避免每次重读 CSV（可省 100ms）
- market 两次读取合并：自己实现一个读原始全量（含集合竞价）的变体，再做下午平移过滤（可省 100ms）

**强制约束**：
- 全程**不并行**（用户 CLAUDE.md 明确）
- 所有循环用 Rust 原生迭代器，避免不必要的 allocation
- 禁止 `format!` 在热路径
- Hawkes 用闭式估计，不做迭代优化

### 7.1 性能验收红线

- **硬性**：`compute_distill_full("000001", 20220819)` 在 < 1.0 秒内返回
- 验证方式：`time.perf_counter()` 计时，运行 3 次取中位数
- 失败处理：用 `factor-performance-checker` agent 诊断瓶颈

---

## 8. 测试设计

### 8.1 单元测试（每分支独立）

| 测试 | 验证内容 |
|---|---|
| `test_auction_extraction` | 集合竞价 5 维信号提取正确（合成数据验证边界） |
| `test_branch_A_decay_fit` | 已知 τ 的合成 ε(t) 能被正确拟合 |
| `test_branch_B_multi_dim` | 5 维 τ 向量顺序与维度对应正确 |
| `test_branch_D_phase_IC` | 4 时段 IC 在已知数据下符合解析值 |
| `test_branch_E_hawkes` | Hawkes 参数在合成事件流上恢复正确 |
| `test_determinism` | 相同输入运行 2 次，结果完全一致 |

### 8.2 集成测试

| 测试 | 验证内容 |
|---|---|
| `test_single_stock_single_day` | 真实 L2 跑 000001/20220819，输出 290 因子（NaN < 30%） |
| `test_against_baseline` | 与简化 Python 实现对比，分支 A 的 τ 误差 < 5% |
| `test_pipeline_python_consistency` | `py_distill` 与 `run_factor_pipeline(pipeline="distill")` 逐字节相同 |

### 8.3 性能测试

```python
# tests/test_distill_perf.py
import rust_pyfunc as rp
import time

def test_perf_000001_20220819():
    # 预热（首次调用含数据读取缓存）
    rp.py_distill("000001", 20220819)
    # 计时
    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        rp.py_distill("000001", 20220819)
        times.append(time.perf_counter() - t0)
    median = sorted(times)[1]
    assert median < 1.0, f"性能超时：中位数 {median:.3f}s > 1.0s"
```

### 8.4 回测验证（后续阶段）

| 测试 | 验证内容 |
|---|---|
| `test_backtest_ic_distribution` | 290 因子日频 IC 分布合理（多数 \|IC\| 在 0.01-0.05） |
| `test_backtest_top_factors` | Top 20 IC 因子分布在 4 个分支（不应集中在 1 个分支） |

---

## 9. 实现里程碑

| 阶段 | 里程碑 | 验收标准 |
|---|---|---|
| M1 | 共享底座 + 分支 A 跑通 | `py_distill` 返回长度 50（先单独测 A），单股单日 < 0.3s |
| M2 | 分支 B 加入 | 返回长度 120（A+B） |
| M3 | 分支 D 加入 | 返回长度 240（A+B+D） |
| M4 | 分支 E 加入 | 返回长度 290（全部），单股单日 < 1s |
| M5 | pipeline 注册 + worker 分发 | `run_factor_pipeline(pipeline="distill")` 可批量跑 |
| M6 | 一致性验证 | py 入口与 pipeline 入口逐字节相同 |
| M7 | 回测验证 | 290 因子 IC 分布合理，Top 因子分布均衡 |

---

## 10. 文件清单

### 10.1 新建文件

| 文件 | 行数估计 | 内容 |
|---|---|---|
| `src/distill_metrics.rs` | ~1500 | 核心 + 名字 + Python 入口 |
| `tests/test_distill.py` | ~50 | 集成测试 |
| `tests/test_distill_perf.py` | ~20 | 性能测试 |

### 10.2 修改文件

| 文件 | 改动 |
|---|---|
| `src/lib.rs` | 加 `pub mod distill_metrics;` + 注册 2 个 py 函数 |
| `src/factor_pipeline.rs` | 加 `pipeline_distill` 函数 + 5 处注册点 |
| `src/bin/worker_pipeline.rs` | 加 worker 分发 |
| `python/rust_pyfunc/__init__.pyi` | 加类型声明 |

### 10.3 不修改文件

- 任何特质模块（`hawkes_analysis.rs` / `ghost_market_maker.rs` 等）
- `fast_csv_reader.rs`（仅调用，不修改）
- `features.rs`（如需降维算子，在 `distill_metrics.rs` 内部实现）

---

## 11. 风险与缓解

| 风险 | 缓解 |
|---|---|
| 1s 性能红线难达 | 优先实现 M1-M3（A+B+D=240 因子），若 M4（分支 E）超时则降级分支 E |
| Hawkes 闭式估计误差大 | 接受，因子研究阶段不要求高精度，只要稳定可复现 |
| 4 时段划分不稳定 | 后续可加入时段敏感性分析（不同时段划分下 IC 的稳定性） |
| 集合竞价 5 维 teacher 信号相关性高 | 不处理，让回测自然筛选；高相关因子会在 IC 加权时自动降权 |
| 290 因子中很多无效 | 这是预期的，"人海战术"的本质就是靠数量取胜，靠 IC 筛选 |

---

## 12. 后续扩展路径（不在本 spec 范围）

本 spec 跑通后，可复制到研究笔记中的其他方向：

- **③ tick 量化器**：纯 L2 算子，类似三层结构
- **⑤ 多时间尺度共振**：分钟级数据，类似三层结构
- **⑧ 跨股蒸馏** 12 个子方向：需要横截面分组表，工程更复杂
- **参数化**：若需调参，再建 `DistillParams` 结构并改 bincode

预期最终因子数：首期 290 → 扩展后 1000-3000，符合 `factor-quantity-checker` 推荐范围。

---

## 13. 参考实现

- **范式参考**：`src/observable_order_metrics.rs` + `src/factor_pipeline.rs::pipeline_observable_order`
- **数据读取参考**：`src/fast_csv_reader.rs::read_market_fast_inner` / `read_trade_fast_inner`
- **skill 参考**：`rust-pipeline-factor-pattern`
- **编译命令**：`timeout 600s bash alter.sh 2>&1`

---

**本 spec 到此结束。待用户审阅后，调用 writing-plans skill 生成实现计划。**
