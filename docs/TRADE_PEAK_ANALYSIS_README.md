# 交易高峰模式分析函数使用说明

## 功能概述

`trade_peak_analysis` 函数用于分析交易数据中的高峰模式，识别成交量的局部高峰，并在时间窗口内分析相关的小峰模式，计算16个统计指标来描述高峰-小峰的特征。

## 函数签名

```python
import rust_pyfunc as rpf

result_matrix, feature_names = rpf.trade_peak_analysis(
    exchtime,        # 纳秒时间戳数组 (np.int64)
    volume,          # 成交量数组 (np.float64)  
    flag,            # 交易标志数组 (np.int32)
    top_tier1,       # 高峰百分比阈值 (float, 如0.01表示前1%)
    top_tier2,       # 小峰百分比阈值 (float, 如0.10表示前10%)
    time_window,     # 时间窗口大小 (float, 秒)
    flag_different,  # 是否只考虑不同flag的小峰 (bool)
    with_forth       # 是否考虑前后时间窗口 (bool)
)
```

## 返回值

- **result_matrix**: `numpy.ndarray` - N行16列的数组，每行对应一个局部高峰的16个统计指标
- **feature_names**: `list[str]` - 包含16个特征名称的字符串列表

## 16个统计特征

| 列索引 | 特征名称 | 描述 |
|--------|----------|------|
| 0 | 小峰成交量总和比值 | 时间窗口内小峰成交量总和与高峰成交量的比值 |
| 1 | 小峰平均成交量比值 | 小峰平均成交量与高峰成交量的比值 |
| 2 | 小峰个数 | 时间窗口内识别到的小峰数量 |
| 3 | 时间间隔均值秒 | 小峰与高峰时间间隔的均值（秒） |
| 4 | 成交量时间相关系数 | 小峰成交量与时间间隔的皮尔逊相关系数 |
| 5 | DTW距离 | 小峰成交量与时间间隔的动态时间规整距离 |
| 6 | 成交量变异系数 | 小峰成交量的标准差/均值 |
| 7 | 成交量偏度 | 小峰成交量分布的偏度 |
| 8 | 成交量峰度 | 小峰成交量分布的峰度 |
| 9 | 成交量趋势 | 小峰成交量的趋势性（与序列索引的相关性） |
| 10 | 成交量自相关 | 小峰成交量的一阶自相关系数 |
| 11 | 时间变异系数 | 时间间隔的标准差/均值 |
| 12 | 时间偏度 | 时间间隔分布的偏度 |
| 13 | 时间峰度 | 时间间隔分布的峰度 |
| 14 | 时间趋势 | 时间间隔的趋势性 |
| 15 | 时间自相关 | 时间间隔的一阶自相关系数 |

## 快速使用示例

```python
import numpy as np
import pandas as pd
import rust_pyfunc as rpf

# 假设已有交易数据
exchtime = trade['exchtime'].values.astype('datetime64[ns]').astype(np.int64)
volume = trade['volume'].values.astype(np.float64)
flag = trade['flag'].values.astype(np.int32)

# 执行分析
result_matrix, feature_names = rpf.trade_peak_analysis(
    exchtime=exchtime,
    volume=volume,
    flag=flag,
    top_tier1=0.01,      # 前1%为高峰
    top_tier2=0.10,      # 前10%为小峰
    time_window=30.0,    # 30秒时间窗口
    flag_different=True, # 只考虑不同flag的小峰
    with_forth=False     # 只考虑高峰后的时间窗口
)

# 构建DataFrame
df = pd.DataFrame(result_matrix, columns=feature_names)
print(f"识别到 {len(df)} 个高峰")
print(df.head())

# 导出结果
df.to_csv('trade_peaks.csv', index=False)
```

## 真实数据使用示例

```python
import pure_ocean_breeze.jason as p

# 读取真实股票数据
trade = p.read_trade('000001', 20220819)

# 数据预处理
n_sample = 10000
exchtime = trade['exchtime'].values[:n_sample].astype('datetime64[ns]').astype(np.int64)
volume = trade['volume'].values[:n_sample].astype(np.float64)
flag = trade['flag'].values[:n_sample].astype(np.int32)

# 执行分析
result_matrix, feature_names = rpf.trade_peak_analysis(
    exchtime, volume, flag,
    top_tier1=0.005,     # 前0.5%为高峰（更严格）
    top_tier2=0.05,      # 前5%为小峰
    time_window=15.0,    # 15秒时间窗口
    flag_different=True,
    with_forth=False
)

# 构建DataFrame并分析
df = pd.DataFrame(result_matrix, columns=feature_names)

# 找出最活跃的高峰
most_active = df.loc[df['小峰个数'].idxmax()]
print(f"最活跃高峰: 小峰个数={most_active['小峰个数']:.0f}, "
      f"时间间隔={most_active['时间间隔均值秒']:.2f}秒")
```

## 参数说明

### top_tier1 (高峰阈值)
- 类型: `float`
- 范围: `0.001 - 0.1`
- 说明: 用于识别局部高峰的成交量百分比阈值
- 建议: 0.01 (前1%) 用于一般分析，0.005 (前0.5%) 用于严格筛选

### top_tier2 (小峰阈值)  
- 类型: `float`
- 范围: `0.01 - 0.3`
- 说明: 用于识别小峰的成交量百分比阈值
- 建议: 0.10 (前10%) 用于一般分析，0.05 (前5%) 用于严格筛选

### time_window (时间窗口)
- 类型: `float`
- 单位: 秒
- 范围: `1.0 - 300.0`
- 说明: 分析高峰前后的时间窗口大小
- 建议: 15-30秒适合高频分析，60-120秒适合中频分析

### flag_different (标志筛选)
- 类型: `bool`
- 说明: 
  - `True`: 只考虑与高峰交易标志不同的小峰
  - `False`: 考虑所有符合条件的小峰
- 建议: `True` 用于分析对手盘反应，`False` 用于完整模式分析

### with_forth (双向分析)
- 类型: `bool`
- 说明:
  - `True`: 考虑高峰前后的时间窗口
  - `False`: 只考虑高峰后的时间窗口
- 建议: `False` 用于因果分析，`True` 用于完整时间模式分析

## 性能特点

- **极高速度**: Rust实现，处理10,000条记录仅需1-2毫秒
- **内存效率**: 优化的数据结构，内存占用极小
- **数值精度**: 高精度浮点计算，确保统计指标准确性
- **可扩展性**: 支持处理百万级数据

## 应用场景

1. **算法交易**: 识别市场微观结构模式
2. **风险管理**: 分析大单冲击后的市场反应
3. **量化研究**: 提取高频交易特征用于机器学习
4. **市场分析**: 研究不同时段的交易行为模式
5. **异常检测**: 识别异常的交易聚集模式

## 注意事项

1. **时间格式**: 输入时间必须是纳秒级时间戳（int64）
2. **数据完整性**: 确保时间、成交量、标志数组长度一致
3. **参数合理性**: top_tier1 应小于 top_tier2
4. **结果解释**: 当小峰个数为0时，其他指标均为0或NaN
5. **内存管理**: 对于超大数据集，建议分批处理

## 更新日志

- **v0.21.2**: 
  - 新增返回格式：(N行16列矩阵, 特征名称列表)
  - 完美适配pandas DataFrame构建
  - 优化统计计算精度
  - 添加中文特征名称支持