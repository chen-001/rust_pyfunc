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

## 功能列表

### 1. 时间序列分析

#### 1.1 DTW动态时间规整 (dtw_distance)
计算两个时间序列之间的DTW（动态时间规整）距离。

```python
import rust_pyfunc as rp

# 示例数据
a = [1, 2, 3, 4]
b = [3, 9, 8, 6, 5]

# 计算DTW距离
distance = rp.dtw_distance(a, b)
print(f"DTW距离: {distance}")
```

#### 1.2 转移熵 (transfer_entropy)
计算从序列x到序列y的转移熵，用于衡量时间序列之间的因果关系。

```python
import numpy as np
from rust_pyfunc import transfer_entropy

# 创建两个相关的时间序列
x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
y = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])  # y比x滞后一个时间步

# 计算转移熵
k = 2  # 考虑过去2个时间步
c = 4  # 将数据离散化为4个等级
te = transfer_entropy(x, y, k, c)
print(f"从x到y的转移熵: {te}")

# 反向计算
te_reverse = transfer_entropy(y, x, k, c)
print(f"从y到x的转移熵: {te_reverse}")
```

#### 1.3 趋势计算 (trend 和 trend_fast)
计算时间序列的趋势。

```python
from rust_pyfunc import trend, trend_fast

# 准备数据
data = [1.0, 2.1, 1.9, 3.2, 4.0, 3.8, 4.5]

# 使用trend函数（更准确但较慢）
trend_result = trend(data)
print(f"趋势（标准版）: {trend_result}")

# 使用trend_fast函数（更快但精度略低）
trend_fast_result = trend_fast(np.array(data,dtype=float))
print(f"趋势（快速版）: {trend_fast_result}")
```

### 2. 统计分析

#### 2.1 最小二乘回归 (ols 和 ols_predict)
执行最小二乘回归分析。

```python
from rust_pyfunc import ols, ols_predict
import numpy as np

# 准备数据
X = np.array([[1, 2], [3, 4], [5, 6]])  # 特征矩阵
y = np.array([2.1, 3.8, 5.2])           # 目标变量

# 训练模型
coefficients = ols(X, y)
print(f"回归系数: {coefficients}")

# 预测新数据
X_new = np.array([[2, 3], [4, 5]])
predictions = ols_predict(X,y, X_new)
print(f"预测结果: {predictions}")
```

#### 2.2 区间统计 (min_range_loop 和 max_range_loop)
计算滑动窗口内的最小值和最大值。

```python
from rust_pyfunc import min_range_loop, max_range_loop

# 准备数据
data = [1.0, 4.0, 2.0, 5.0, 3.0, 6.0, 2.0]

# 计算滑动窗口最小值
min_values = min_range_loop(data)
print(f"滑动窗口最小值: {min_values}")

# 计算滑动窗口最大值
max_values = max_range_loop(data)
print(f"滑动窗口最大值: {max_values}")
```

### 3. 文本分析

#### 3.1 句子向量化 (vectorize_sentences 和 vectorize_sentences_list)
将句子转换为词频向量。

```python
from rust_pyfunc import vectorize_sentences, vectorize_sentences_list

# 两个句子的向量化
s1 = "The quick brown fox"
s2 = "The lazy brown dog"
v1, v2 = vectorize_sentences(s1, s2)
print(f"句子1的词频向量: {v1}")
print(f"句子2的词频向量: {v2}")

# 多个句子的向量化
sentences = [
    "The quick brown fox",
    "The lazy brown dog",
    "A quick brown fox jumps"
]
vectors = vectorize_sentences_list(sentences)
for i, vec in enumerate(vectors):
    print(f"句子{i+1}的词频向量: {vec}")
```

#### 3.2 Jaccard相似度 (jaccard_similarity)
计算两个句子之间的Jaccard相似度。

```python
from rust_pyfunc import jaccard_similarity

# 测试完全相同的句子
s1 = "The quick brown fox"
s2 = "The quick brown fox"
sim1 = jaccard_similarity(s1, s2)
print(f"完全相同的句子相似度: {sim1}")  # 输出: 1.0

# 测试部分相同的句子
s3 = "The lazy brown dog"
sim2 = jaccard_similarity(s1, s3)
print(f"部分相同的句子相似度: {sim2}")  # 输出: 0.4

# 测试完全不同的句子
s4 = "Hello world example"
sim3 = jaccard_similarity(s1, s4)
print(f"完全不同的句子相似度: {sim3}")  # 输出: 0.0
```

### 4. 序列分析

#### 4.1 分段识别 (identify_segments)
识别序列中的连续分段。

```python
from rust_pyfunc import identify_segments

# 准备数据
data = [1, 1, 1, 2, 2, 3, 3, 3, 1, 1]

# 识别连续分段
segments = identify_segments(data)
print(f"连续分段: {segments}")
# 输出形如: [(1, 3), (2, 2), (3, 3), (1, 2)]
# 表示：值1连续出现3次，值2连续出现2次，值3连续出现3次，值1连续出现2次
```

#### 4.2 最大范围乘积 (find_max_range_product)
寻找序列中乘积最大的区间。

```python
from rust_pyfunc import find_max_range_product

# 准备数据
data = [2.0, -3.0, 4.0, -1.0, 2.0, 1.0, -5.0, 4.0]

# 查找最大乘积区间
start_idx, end_idx, max_product = find_max_range_product(np.array(data,dtype=float))
print(f"最大乘积区间: [{start_idx}, {end_idx}]")
print(f"最大乘积值: {max_product}")
```

## 注意事项

1. 所有函数都经过Rust优化，相比Python原生实现有显著的性能提升
2. 输入数据需要符合函数要求的格式和类型
3. 部分函数（如`transfer_entropy`）的参数需要根据具体场景调整以获得最佳结果
4. 文本处理函数会自动进行大小写转换和标点符号处理

## 性能建议

1. 对于大规模数据，优先使用带有"fast"后缀的函数版本
2. 文本处理时，建议预先进行数据清洗
3. 时间序列分析时，注意数据的预处理（如归一化）可能会影响结果

## 贡献指南

欢迎提交Issue和Pull Request来改进这个项目。在提交代码前，请确保：

1. 代码经过充分测试
2. 添加了适当的文档和示例
3. 遵循项目的代码风格

## License

MIT License
