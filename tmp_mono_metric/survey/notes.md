# 调研：已有的「分组收益单调性」指标，以及对 Rank IC 的批评

调研时间盒：25 分钟（web_search + web_fetch）。每条给出 URL 与「能否直接用来对称衡量 10 组单调性」的判定。
判定分三类：**能直接用** / **需要改造** / **不适用**。
标注 `[已抓全文]` 的是我实际打开读过内容的页面；`[仅搜索摘要]` 表示只看到检索结果摘要，未逐字核对页面。

---

## 1. 学术界：排序组合单调性检验

### 1.1 Patton & Timmermann (2010, JFE) 单调性检验 — 需要改造

出处：论文 *Monotonicity in asset returns: New tests with applications to the term structure, the CAPM, and portfolio sorts*, Journal of Financial Economics 98(3), 605–625，doi: [10.1016/j.jfineco.2010.06.006](https://doi.org/10.1016/j.jfineco.2010.06.006)；工作论文 PDF 见 [Duke 页面](https://public.econ.duke.edu/~ap172/Patton_Timmermann_sorts_24dec09.pdf)（PDF 无法直接抓取正文）。R 实现见 CRAN 包 `monotonicity`。

R 文档给出的三组检验定义 `[已抓全文]`：

- **MR 检验（monotonic relationship）**，见 [monoRelation 文档](https://search.r-project.org/CRAN/refmans/monotonicity/html/monoRelation.html)：
  记相邻组合收益差 `Δ_i = E[r_(i,t)] − E[r_(i−1,t)]`，检验
  `H0: Δ ≤ 0` vs `H1: min_(i=1..N) Δ_i > 0`（即「逐级严格递增」）。
  同时给出「只用相邻组合」和「用所有两两组合比较」两个版本的 p 值。
- **Up-Down 检验**，见 [monoUpDown 文档](https://search.r-project.org/CRAN/refmans/monotonicity/html/monoUpDown.html)：
  对相邻差分的**平方和**与**绝对值和**分别构造统计量，分别检验递增/递减两种模式，用 Politis-Romano 平稳 bootstrap 出 p 值，并给出学生化版本。
- **Wolak (1989) 不等式约束检验 + Bonferroni 版**，见 [monoSummary 文档](https://search.r-project.org/CRAN/refmans/monotonicity/html/monoSummary.html)：
  输出 `TopMinusBottom`（顶组减底组均值）、`t_stat`/`t_pval`、`MR_pval`、`MRall_pval`、`UP_pval`、`DOWN_pval`、`Wolak_pval`、`Bonferroni_pval`。该 R 实现限制最多 15 个组合，**10 组正好在范围内**。

判定：**需要改造**。它是显著性检验（p 值），不是连续得分；且方向由 `increasing` 参数指定，本身不区分多空两侧。若把它当作「10 组是否单调」的判据可以直接用 p 值；若要一个对多空公平的 0–1 分，需要在它之上另设得分函数。

### 1.2 Isotonic regression / PAVA 拟合优度 — 需要改造

- 定义与算法：保序回归就是在约束 `x_1 ≤ x_2 ≤ … ≤ x_n` 下最小化加权平方误差 `Σ w_i (y_i − x_i)²`；一维有序情形用 PAVA（pool-adjacent-violators）求解。[Wikipedia: Isotonic regression](https://en.wikipedia.org/wiki/Isotonic_regression) `[已抓全文]`。中文条目见 [保序回归](https://zh.wikipedia.org/wiki/%E4%BF%9D%E5%BA%8F%E5%9B%9E%E5%BD%92)。
- 「拟合优度」这一提法确实有专门文献：[Goodness of fit test for isotonic regression](http://www.numdam.org/item/?id=PS_2001__5__119_0/)（numdam 记录）`[仅搜索摘要]`，另有 [把保序回归投影化归到 PAVA 的文章](https://amstat.tandfonline.com/doi/figure/10.1080/02331934.2022.2103415) `[仅搜索摘要]`。

判定：**需要改造**。把 10 组收益 `y` 对组序 1..10 做保序回归，可得单调拟合值；用一个拟合优度标量（例如 `R²_iso = 1 − SSE_iso / SST`，或与无约束均值比较的 F 型统计量）就是「单调性得分」。注意 `R²_iso ≥ 0` 且对递增/递减都对称（做一次反向拟合取大者），这一点比 Spearman 更适合「两侧公平」。缺点是没有任何现成的因子评价实现，需要自己写（PAVA 只有几十行）。

### 1.3 Kendall tau — 能直接用

- 定义：基于一致对 P 与不一致对 Q，`τ_b = (P − Q) / sqrt((n0 − n1)(n0 − n2))`，其中 n0 = n(n−1)/2，n1、n2 分别处理 x、y 上的并列。范围 [−1, 1]，0 表示无关联。[Wikipedia: Kendall rank correlation coefficient](https://en.wikipedia.org/wiki/Kendall_rank_correlation_coefficient) `[仅搜索摘要]`；R 实现见 [KendallTauB 文档](https://search.r-project.org/CRAN/refmans/DescTools/html/KendallTauB.html) `[仅搜索摘要]`。

判定：**能直接用**。把「10 组 × 组内均值收益」当成 10 个观测，`tau(组序, 组均收益)` 就是标准的单调性得分：值域对称、对递增递减取号、完全单调时取 ±1、乱序时趋近 0。**这正是 Jonckheere-Terpstra 在 10 组均值这一特例下的统计量**（见 1.5）。

### 1.4 Spearman ρ on group returns — 能直接用（社区主流做法，见第 2 节）

`ρ = 1 − 6 Σ d_i² / (n(n²−1))`，d_i 为组序与组均收益的秩差。等价于对两组秩做 Pearson。判定：**能直接用**，但对 10 个点的秩相关来说，它和 Kendall tau 一样对称，缺点是只捕捉秩的一致性、对「中间组乱序」和「尾部乱序」同等惩罚，不区分尾部（这恰好是用户想解决的问题之一，见第 4 节）。

### 1.5 Jonckheere-Terpstra 趋势检验（重点）— 能直接用（作为检验）/ 需要改造（作为得分）

**定义**：检验 k 个独立样本是否来自同一总体，备择假设是「按先验顺序的位置参数递增」：
`H0: F_1(x) = F_2(x) = … = F_k(x)`，`Ha: F_1(x) ≥ F_2(x) ≥ … ≥ F_k(x)`（至少一个严格）。k = 10 就是「10 个有序分组的单调趋势检验」。

**统计量**（[NIST Dataplot 参考页](https://www.itl.nist.gov/div898/software/dataplot/refman1/auxillar/jonck.htm) `[已抓全文]`）：

```
S = Σ_{i=1..k−1} Σ_{j=i+1..k} p_ij  −  Σ_{i=1..k−1} Σ_{j=i+1..k} n_i n_j
```
其中 `p_ij` 是「X_i < X_j 成立的次数」（即第 i 组每个观测小于第 j 组每个观测的计数和）。

**零分布**（同页）：E[S] = 0；无跨组并列时
```
Var(S) = [ n²(2n+3) − Σ_i n_i²(2n_i+3) ] / 18      （n = Σ n_i）
```
调整统计量 `S_adj = S / sqrt(Var(S))`，与标准正态比较，单侧（上尾）检验。
有跨组并列时不加连续性校正，方差要用列联表行列边际量 `t_i`、`u_i` 另算（[Wikipedia: Jonckheere's trend test](https://en.wikipedia.org/wiki/Jonckheere%27s_trend_test) `[已抓全文]`，其中还给了 S = P − Q 的「直接计数法」和「航海法」两种算法、连续性校正 `S_c = |S| − 1`、以及 k=3 的数值例）。
此外还有精确表（Jonckheere 给出 k 从 3 到 6、等样本量 m 从 2 到 5 的临界值）与**置换检验**两种求 p 值的方式（NIST 页面给的是 4000 次置换的参考分布）。

**实现参考**：
- R `SAGx::JT.test`（[文档](https://api.rdocumentation.org/packages/SAGx/versions/1.46.0/topics/JT.test) `[已抓全文]`）：输入「基因 × 样本」矩阵 + 有序分组标签，正态近似出 p 值，并额外给出 `trend`（与分组的秩相关）和 `S1`（预测强度）。
- R `fastJT`（高效 JT 统计量，[rdrr 文档](https://rdrr.io/cran/fastJT/man/fastJT-package.html) `[仅搜索摘要]`）。
- 方差公式的专门讨论：[A note on the variances of the tests of Kendall, Jonckheere, and Terpstra](https://docta.ucm.es/rest/oai/request?verb=GetRecord&metadataPrefix=qdc&identifier=oai:docta.ucm.es:20.500.14352/51428) `[仅搜索摘要]`。

**在 10 个有序组上的用法（我的推导，非引文）**：JT 的原设计里每组有很多观测。用日频因子时有两种接法：
1. **组内个股收益全用**：每天把全市场个股按因子分 10 组，直接对「10 组 × 当日个股收益」跑 JT，得到当日的 `S_adj` 或 p 值，再按时间聚合（如取 `mean(S_adj)` 或对 p 值做 Fisher 合并）。这样保留了组内样本量，`Var(S)` 公式可用；缺点是 JT 检验的是分布位置的有序性，不是「组均收益的单调性」的线性强度，且它对尾部两组样本量不敏感（等分位分组时样本量天然相等，问题不大）。
2. **只用 10 个组均收益**：此时每组只有 1 个观测，JT 退化为 10 个点上的 Kendall `S`（S = P − Q），其置换零分布是精确可枚举的（10! 个排列，或直接用 Kendall 的精确分布）。这一版本最贴合「10 组单调性」，**推荐**：`S / max|S|` 就是范围 [−1, 1]、完全单调取 ±1、对多空两侧对称的得分。

判定：**能直接用**（作为「10 个有序组是否单调」的检验，有成熟公式、零分布、R 实现）；**需要改造**（若只想要一个可比较的连续得分，用 `S_adj` 或 `S/S_max`，并明确是逐日聚合还是全样本聚合）。

### 1.6 Romano-Wolf — 未找到（与单调性直接结合的用法）

Romano-Wolf 是多重检验的 stepdown 方法（bootstrap 控制 FWER），检索中出现的都是「多个假设同时检验」语境。**未找到**把它直接用来构造「分组收益单调性指标」的公开做法。与 Patton-Timmermann 的 Bonferroni 版属于同类用途（多组合比较时控制族错误率），判定：**不适用**（不是单调性度量）。

### 1.7 附带发现：学术界目前报告的是两套指标，不含单调性得分

DSPO（*An End-to-End Framework for Direct Sorted Portfolio Construction*, arXiv 2405.15833，[ar5iv 全文](https://ar5iv.labs.arxiv.org/html/2405.15833) `[已抓全文]`）把评价指标明确分成两类：**representation-based（RankIC、RankICIR）** 与 **portfolio-based（Long/LS Return、Information Ratio）**，A 股因为做空受限只报 Long。它提出的是 MonLR 损失函数（用 tanh 平滑的成对符号似然），不是单调性指标。

判定：**不适用**，但它是「RankIC 与组合收益被分开报告、没有统一单调性得分」这一现状的近期证据。

---

## 2. 中文量化社区：「分组单调性 / 单调性得分」的具体算法

### 2.1 PandaAI/QuantSkills 的 `skill-factor-evaluate` — 能直接用（最具体的公开实现）

出处：[quantskills/skill-factor-evaluate README](https://github.com/quantskills/skill-factor-evaluate) `[已抓全文]`，算法在 [references/metrics.md](https://raw.githubusercontent.com/quantskills/skill-factor-evaluate/main/references/metrics.md) `[已抓全文]`。

它的做法就是最主流的「单调性得分」：

```python
def monotonicity(signal, fwd_ret, n_groups=5):
    rank = signal.rank(axis=1, pct=True)
    group_rets = []
    for q in range(n_groups):
        mask = (rank > q/n_groups) & (rank <= (q+1)/n_groups)
        group_rets.append(fwd_ret.where(mask).mean(axis=1).mean())
    return float(np.corrcoef(np.arange(n_groups), group_rets)[0, 1])
```

即：**按因子分 n 组（5 或 10）→ 每组算时序平均收益 → 组序与组均收益求相关系数**，值域 [−1, 1]，完全单调取 ±1。报告里的主分公式给单调性权重 0.10（`score = 0.20*IC + 0.30*Sharpe + 0.30*年化收益 + 0.20*MDD + 0.10*单调性 + 0.10*换手`）。

两处需要注意：函数名叫 `monotonicity`、注释写 Spearman，代码实际用的是 `np.corrcoef`（Pearson，因为输入是 1..n 与组均收益，秩相关与 Pearson 在「组序本身即等距」时接近但不相同）；另外它是 5/10 分组的**多头视角**打分，没有任何多空对称处理。

判定：**能直接用**（这就是用户要的「分组单调性得分」的社区标准做法），**需要改造**才能做到「对多空两侧公平」（见第 4 节）。

### 2.2 叩富网：《因子单调性（Monotonicity）：检验策略逻辑真伪的照妖镜》— 不适用（只有定性流程）

出处：[licai.cofool.com/user/guide_view_3413778.html](https://licai.cofool.com/user/guide_view_3413778.html) `[已抓全文]`。给出的「标准五步法」：全市场扫描 → 强制切 5 等分 → 固定持有期跟踪各组净值 → 把 5 条曲线画在一张图 → **看曲线是否「像整齐的阶梯」、是否交叉缠绕**。它明确把「第 1 组好、第 5 组也好、中间 2/3/4 组差」称为「单调性破裂」，并指出 IC 高但单调性破裂是常见陷阱。

判定：**不适用**（无公式、无得分，靠看图），但它是中文社区把「单调性」当作 IC 之外独立质检项的典型表述，可作为问题动机的引用。

### 2.3 聚宽社区（颖硕转石川）：IC 的陷阱与两种改进 — 需要改造

出处：[《用 IC 评价因子效果靠谱吗？》](https://joinquant.com/community/post/detailMobile?postId=14431) `[已抓全文]`（转自石川「量化投资与机器学习」，原文链接在文内：https://mp.weixin.qq.com/s/B-CI22w7CWsA_vVsss8srQ `[仅搜索摘要]`）。

核心例子：10 只股票、因子从大到小排序，构造两组收益率序列，**两组的 IC 完全相同（都是 0.2909）、回归斜率也相同（0.0058）**，但一组是「因子越大收益越高」，另一组是「因子越大收益越差」——只看 IC 完全看不出差别。它提出的两种改进：
1. **先分组再算 IC**：按因子把股票分 n 档（如 10 档），把每档当投资组合，再算「组合收益 vs 组合因子值」的 IC / Rank IC（等权或市值加权）。理由：因子是一揽子股票的共同暴露，组合收益噪音更小。
2. **加权 IC（weighted IC）**：按因子业务方向排序后，给样本按指数衰减赋权（文中系数 0.9，权重 `w^i / Σ w^i`），再算加权相关系数；公式（评论区给出实现）：

```
weighted_ic = (A1 − A2·A3) / (sqrt(B1 − A2²)·sqrt(B2 − A3²))
A1=Σ w·x·y, A2=Σ w·x, A3=Σ w·y, B1=Σ w·x², B2=Σ w·y²
```

判定：**需要改造**。方法 1 是「分组 IC」，跟分组单调性是同一族思路；方法 2 的指数衰减权重是**刻意偏向头部（多头端）**的，方向恰好与「对多空两侧公平」相反——如果要两侧对称，需要把权重改成对两端对称（如按 |秩 − 中位| 赋权）。

### 2.4 CSDN：《单因子测试 IC 表现和分组效果矛盾》— 不适用（只有解释）

出处：[blog.csdn.net/Baiempirical/article/details/137541729](https://blog.csdn.net/Baiempirical/article/details/137541729) `[已抓全文]`。解释了两种矛盾：①IC 不佳但多空组合好（IC 是整段样本平均，受异常值/短期噪声/非线性影响）；②IC 好但分组单调性差（非线性、噪声、因子时效性）。判定：**不适用**（无算法），但它是中文社区公开讨论「IC 与分组单调性不一致」的直接证据。

### 2.5 其他中文来源（未逐字核对）

- [BigQuant：从相关关系到指数增强——谈 IC 系数与股票权重的联系](https://sysubs.bigquant.com/square/paper/55ab532e-8e6c-4436-9337-b9e1d554da12) `[仅搜索摘要]`
- [聚宽：【因子组合 一】筛去因子极差组框架](https://www.joinquant.com/view/community/detail/50c5d5aeb682329b644a5d5ab9ffbf76) `[仅搜索摘要]`
- [华泰金工：双目标遗传规划应用于行业轮动](https://m.jrj.com.cn/madapter/stock/2024/05/24071340760829.shtml) `[仅搜索摘要]`，其中明确写「|IC| 是量化投资中最常用的因子评价指标」
- [CSDN：《投资-347》因子有效性检验](https://blog.csdn.net/HiWangWenBing/article/details/155141174) `[仅搜索摘要]`
- [百度云：量化投资单因子回测神器解析 — Alphalens](https://cloud.baidu.com/article/3791336) `[仅搜索摘要]`：Alphalens 是社区默认的因子评估工具，其输出是 IC 序列 + 分位组合收益曲线，**本身不提供单调性得分**。

**「单调性得分」这个词本身**：检索到的公开实现只有 2.1 的 `monotonicity()`（组序 vs 组均收益相关系数）；其余中文材料是「看图判断」，**未找到**第二个带公式的公开「单调性得分」定义。

---

## 3. 对 Rank IC 的已知批评

1. **相同 IC 可以是完全相反的收益结构**（最强的一条）：聚宽/石川文中 0.2909 的例子，两组收益的 IC 与回归斜率完全相同，但一组因子越大越好、另一组因子越大越差。出处：[joinquant postId=14431](https://joinquant.com/community/post/detailMobile?postId=14431) `[已抓全文]`。
2. **IC 是整段样本的单一标量，会被异常值与噪声主导**；它不区分「预测能力来自头部还是尾部」：CSDN 文（[链接](https://blog.csdn.net/Baiempirical/article/details/137541729)）与 Quantopian 论坛帖 [Good IC but poor quantiles? When to move from research to backtest?](https://quantopian-archive.netlify.app/forum/threads/good-ic-but-poor-quantiles-when-to-move-from-research-to-backtest) `[已抓全文]`（提问者 IC≈0.02、整体分位图看似单调，但**按行业分组的分位图很差**，说明 IC 抹平了子样本差异）。
3. **IC 与多空收益不等价**：DSPO（[ar5iv](https://ar5iv.labs.arxiv.org/html/2405.15833)）把 RankIC/RankICIR 与 Long/LS Return/IR 作为两套独立指标报告；A 股因做空受限只报 Long，进一步说明「IC 一个数」不足以描述多空两侧。
4. **IC 与分组单调性会系统性不一致**：叩富网文章直接称之为「单调性破裂」的陷阱（[链接](https://licai.cofool.com/user/guide_view_3413778.html)）。

判定：这些批评与用户的观察一致，**但都是定性论述，没有任何一篇给出替代 Rank IC 的单一对称标量**。

---

## 4. 是否已有人提出「多空两侧对称 / 均衡」的因子评价指标

**结论：未找到公开命名的「long-short balanced IC / 两侧 IC / half-IC / 对称 IC」指标。** 用 4 组关键词检索（英文 "asymmetric IC"、"long-short balanced IC"、"symmetric factor evaluation metric"；中文「多空 不对称 因子评价 指标」「两侧 IC」）均无命中。

找到的最接近的替代品是**「把因子拆成多头腿与空头腿分别评估」**，而不是合成一个对称标量：

- **Blitz, Baltussen & van Vliet (2020), "When Equity Factors Drop Their Shorts"**, *Financial Analysts Journal*：[Taylor & Francis 全文](https://www.tandfonline.com/doi/full/10.1080/0015198X.2020.1779560) `[仅搜索摘要]`、[CFA Institute 摘要](https://rpc.cfainstitute.org/research/financial-analysts-journal/2020/when-equity-factors-drop-their-shorts) `[仅搜索摘要]`。做法：把 HML/WML/RMW/CMA/VOL 各自拆成「多头组合 − 市场」与「市场 − 空头组合」两条腿，分别算收益、Sharpe、alpha、尾部风险；结论是多数因子的溢价集中在多头腿、空头腿分散化效果差、空头腿尾部风险约为多头腿两倍。
- **中文解读 + 反驳（动态对冲）**：徐杨《股票因子的多与空》，[金融界](https://opinion.jrj.com.cn/2020/01/19131828701132.shtml) `[已抓全文]`。该文指出原论文用 1:1 对冲使空头腿残留市场负暴露、低估空头腿；改用滚动 10 年 beta 动态对冲后，**空头腿相对多头腿出现正溢价，最优夏普组合把 44.1% 权重分给空头腿**（1:1 对冲时只有 5%）。这是「多空两侧不对称」这一现象最直接的公开讨论。
- 另外检索到的 [顶端优化模型：A股做空受限下的因子空头问题](https://zhuanlan.zhihu.com/p/2002759835247215742) `[仅搜索摘要]`（知乎 403，未能抓取正文），从标题看是同一问题在 A 股做空受限背景下的讨论。

判定：**这类研究解决的是「如何分别评估两条腿」，没有解决「如何用一个指标公平地衡量 10 组单调性」**——这个缺口是真实存在的。

---

## 5. 汇总表

| 候选指标 | 公式/做法要点 | 出处 | 判定 |
|---|---|---|---|
| 组序 vs 组均收益相关（社区「单调性得分」） | `corr(arange(n_groups), group_mean_ret)`，值域 [−1,1] | [quantskills metrics.md](https://raw.githubusercontent.com/quantskills/skill-factor-evaluate/main/references/metrics.md) | 能直接用（10 组现成可套）；未处理多空对称 |
| Kendall τ_b（10 组均值上） | `(P−Q)/sqrt((n0−n1)(n0−n2))` | [Wikipedia](https://en.wikipedia.org/wiki/Kendall_rank_correlation_coefficient) | 能直接用；对称、有号、范围 [−1,1] |
| Spearman ρ（10 组均值上） | `1 − 6Σd²/(n(n²−1))` | 社区标准做法（同上） | 能直接用；与 τ 同样对称，但对尾部不敏感 |
| Goodman-Kruskal γ | `(P−Q)/(P+Q)`，忽略并列，范围 [−1,1] | [DescTools 文档](https://search.r-project.org/CRAN/refmans/DescTools/html/GoodmanKruskalGamma.html) | 能直接用（并列多时 γ 比 τ 更稳） |
| Jonckheere-Terpstra | `S = Σ_{i<j}(p_ij − n_i n_j)`；`Var(S)=[n²(2n+3)−Σn_i²(2n_i+3)]/18`；`S/√Var` 对标准正态，单侧 | [NIST Dataplot](https://www.itl.nist.gov/div898/software/dataplot/refman1/auxillar/jonck.htm)、[Wikipedia](https://en.wikipedia.org/wiki/Jonckheere%27s_trend_test)、[SAGx::JT.test](https://api.rdocumentation.org/packages/SAGx/versions/1.46.0/topics/JT.test) | 能直接用（检验）+ 需要改造（得分）；10 组均值特例即 Kendall S，置换零分布可精确枚举 |
| Patton-Timmermann MR / Up-Down / Wolak | MR: `H0: Δ≤0` vs `H1: min Δ_i>0`；Up-Down: 差分平方和/绝对值和 + 平稳 bootstrap | [monoRelation](https://search.r-project.org/CRAN/refmans/monotonicity/html/monoRelation.html)、[monoUpDown](https://search.r-project.org/CRAN/refmans/monotonicity/html/monoUpDown.html)、[monoSummary](https://search.r-project.org/CRAN/refmans/monotonicity/html/monoSummary.html) | 需要改造（是 p 值不是得分；最多 15 组，10 组可用） |
| Isotonic regression / PAVA 拟合优度 | 约束 `x_1≤…≤x_n` 下最小化 `Σw_i(y_i−x_i)²`，用 `R²_iso` 或 F 型统计量当得分 | [Wikipedia](https://en.wikipedia.org/wiki/Isotonic_regression)、[numdam goodness-of-fit](http://www.numdam.org/item/?id=PS_2001__5__119_0/) | 需要改造（无现成因子实现，需自写 PAVA） |
| 加权 IC（石川） | 按因子排序后指数衰减赋权，再算加权相关 | [joinquant 14431](https://joinquant.com/community/post/detailMobile?postId=14431) | 需要改造（默认偏向多头端，与「两侧公平」方向相反） |
| 分组 IC | 先分 n 档，再用组合收益与组合因子值算 IC | 同上 | 需要改造 |
| 多空腿分解评估 | 因子拆成 long leg / short leg，各自对冲后比收益、Sharpe、alpha、尾部 | [FAJ 2020](https://www.tandfonline.com/doi/full/10.1080/0015198X.2020.1779560)、[徐杨解读](https://opinion.jrj.com.cn/2020/01/19131828701132.shtml) | 需要改造（是两条腿分别评估，不是单一对称标量） |
| Romano-Wolf / Bonferroni | 多重检验 stepdown | — | 不适用（不是单调性度量） |
| 「long-short balanced IC / 两侧 IC / half-IC」 | — | — | **未找到** |

---

## 6. 结论（哪些能直接用，哪些确实还缺）

1. **能直接用**：Jonckheere-Terpstra（`S = Σ_{i<j}(p_ij − n_i n_j)`，零分布方差 `[n²(2n+3) − Σ n_i²(2n_i+3)]/18`，单侧正态近似，R 的 `SAGx::JT.test` 是现成实现，10 组完全在适用范围内）——它就是「10 个有序分组的单调趋势检验」，直接命中需求。
2. **能直接用（作为得分）**：把 10 组均收益当 10 个点，算 Kendall τ_b 或 Goodman-Kruskal γ，两者都有号、对称、范围 [−1,1]，完全单调取 ±1；社区版的「单调性得分」就是组序与组均收益的相关系数（quantskills 的 `monotonicity()`）。
3. **能直接用（作为检验）**：Patton-Timmermann 的 MR / Up-Down / Wolak 检验有现成 R 包（`monotonicity`，上限 15 组），给的是 p 值而非得分；Isotonic regression 有定义与算法但**没有**现成的因子评价实现，要自己写 PAVA。
4. **确实还缺**：没有任何公开命名的「多空两侧对称 / 均衡」因子评价指标（long-short balanced IC、两侧 IC、half-IC 均**未找到**）；已有的处理方式是**把多头腿与空头腿拆开分别评估**（Blitz-Baltussen-van Vliet 2020 及其动态对冲反驳），而 Rank IC 被批评为单一标量、无法区分收益结构，这些批评都停留在定性层面，没有给出替代标量。因此「用一个对称标量衡量 10 组单调性」这一步需要自己定义。
