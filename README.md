# Rust_Pyfunc

一些用Python计算起来很慢的指标，这里用Rust来实现，提升计算速度。

## 安装
```shell
pip install rust_pyfunc
```

## 使用
```python
import rust_pyfunc as rp
```

## 功能列表（以下内容为windsurf生成）

### 1. 时间序列分析

#### 1.1 DTW动态时间规整 (dtw_distance)
计算两个时间序列之间的DTW（动态时间规整）距离。

#### 1.2 转移熵 (transfer_entropy)
计算从序列x到序列y的转移熵，用于衡量时间序列之间的因果关系。

#### 1.3 趋势计算 (trend 和 trend_fast)
计算时间序列的趋势。

### 2. 统计分析

#### 2.1 最小二乘回归 (ols 和 ols_predict)
执行最小二乘回归分析。

#### 2.2 区间统计 (min_range_loop 和 max_range_loop)
计算滑动窗口内的最小值和最大值。

### 3. 文本分析

#### 3.1 句子向量化 (vectorize_sentences 和 vectorize_sentences_list)
将句子转换为词频向量。

#### 3.2 Jaccard相似度 (jaccard_similarity)
计算两个句子之间的Jaccard相似度。

### 4. 序列分析

#### 4.1 分段识别 (identify_segments)
识别序列中的连续分段。


#### 4.2 最大范围乘积 (find_max_range_product)
寻找序列中乘积最大的区间。


### 5. 文本编辑距离

#### 5.1 最小词编辑距离 (min_word_edit_distance)
计算两个句子之间的最小词编辑距离。


### 6. 时间序列峰值分析

#### 6.1 局部峰值识别 (find_local_peaks_within_window)
在给定窗口内识别时间序列的局部峰值。


### 7. Pandas扩展功能

#### 7.1 滚动窗口统计 (rolling_window_stat)
对时间序列进行滚动窗口统计分析。

```python
import pandas as pd
import rust_pyfunc as rp

# 准备数据
a = pd.Series([1, 2, 3, 4, 4],index=pd.date_range("2024-01-01 00:00:00", "2024-01-01 00:00:04", freq="1s"))

# 计算滚动窗口统计
a.rolling_future('2s').mean()
```

### 8. 价格树分析

#### 8.1 价格树 (PriceTree)
用于分析价格序列的树形结构。


## 注意事项

1. 所有函数都经过Rust优化，相比Python原生实现有显著的性能提升
2. 输入数据需要符合函数要求的格式和类型
3. 部分函数（如`transfer_entropy`）的参数需要根据具体场景调整以获得最佳结果
4. 对于需要处理大量数据的场景，建议使用numpy数组作为输入以获得更好的性能
5. 在使用`PriceTree`等复杂数据结构时，注意及时释放资源


## 贡献指南

欢迎提交Issue和Pull Request来改进这个项目。在提交代码前，请确保：

1. 代码经过充分测试
2. 添加了适当的文档和示例
3. 遵循项目的代码风格

## License

MIT License
