# Rust_Pyfunc [![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/chen-001/rust_pyfunc)

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

#### 1.1 趋势分析 (trend 和 trend_fast)
计算时间序列的趋势，即输入数组与自然数序列(1, 2, ..., n)之间的皮尔逊相关系数。

```python
import numpy as np
import rust_pyfunc as rp

# 创建测试数据
data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

# 计算趋势
trend_value = rp.trend(data)  # 通用版本
trend_fast_value = rp.trend_fast(data)  # 高性能版本，仅支持numpy数组
```

#### 1.2 DTW动态时间规整 (dtw_distance)
计算两个序列之间的动态时间规整(DTW)距离，可处理不等长序列。

```python
import rust_pyfunc as rp

# 创建两个测试序列
s1 = [1.0, 2.0, 3.0, 4.0]
s2 = [1.0, 2.0, 2.5, 3.0, 4.0]

# 计算DTW距离
distance = rp.dtw_distance(s1, s2)
# 使用Sakoe-Chiba带宽限制计算
distance_with_radius = rp.dtw_distance(s1, s2, radius=2)
```

#### 1.3 转移熵 (transfer_entropy)
计算从序列x到序列y的转移熵，用于衡量时间序列间的因果关系。

```python
import rust_pyfunc as rp

# 创建两个测试序列
x = [1.0, 2.0, 3.0, 4.0, 5.0]
y = [1.0, 2.0, 3.0, 4.0, 5.0]

# 计算转移熵，k是历史长度，c是离散化类别数
te = rp.transfer_entropy(x, y, k=2, c=3)
```

### 2. 统计分析

#### 2.1 最小二乘回归 (ols, ols_predict 和 ols_residuals)
执行最小二乘回归分析、预测和残差计算。

```python
import numpy as np
import rust_pyfunc as rp

# 创建设计矩阵和响应变量
X = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float64)
y = np.array([1, 2, 3], dtype=np.float64)

# 计算回归系数
coefficients = rp.ols(X, y)
# 不计算R²值
coefficients = rp.ols(X, y, calculate_r2=False)

# 预测新数据
X_pred = np.array([[7, 8], [9, 10]], dtype=np.float64)
predictions = rp.ols_predict(X, y, X_pred)

# 计算回归残差
residuals = rp.ols_residuals(X, y)
print(f"残差: {residuals}")  # 模型拟合得好的话，残差应该接近零
```

其中：
- `ols`: 计算线性回归系数
- `ols_predict`: 使用回归模型进行预测
- `ols_residuals`: 计算实际值与预测值之差（残差）

#### 2.2 滚动窗口统计 (rolling_window_stat)
对时间序列进行基于时间窗口的滚动统计分析。

```python
import numpy as np
import rust_pyfunc as rp

# 创建时间和值序列
times = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

# 计算2秒窗口的均值
means = rp.rolling_window_stat(times, values, window=2.0, stat_type="mean")

# 其他统计类型
maxs = rp.rolling_window_stat(times, values, window=2.0, stat_type="max")
mins = rp.rolling_window_stat(times, values, window=2.0, stat_type="min")
sums = rp.rolling_window_stat(times, values, window=2.0, stat_type="sum")
std = rp.rolling_window_stat(times, values, window=2.0, stat_type="std")
trend = rp.rolling_window_stat(times, values, window=2.0, stat_type="trend_time")
```

#### 2.3 滚动波动率 (rolling_volatility)
计算价格序列的滚动波动率，支持自定义回溯窗口和采样间隔。

```python
import numpy as np
import rust_pyfunc as rp

# 创建价格序列
prices = np.array([100.0, 102.0, 101.0, 103.0, 102.0, 104.0, 103.0])

# 计算滚动波动率，向前查看5个点，每隔1个点取样
vol = rp.rolling_volatility(prices, lookback=5, interval=1)

# 指定最小样本数
vol_min_periods = rp.rolling_volatility(prices, lookback=5, interval=1, min_periods=3)
```

### 3. 序列分析

#### 3.1 连续段识别 (identify_segments)
识别数组中的连续相等值段，并为每个段分配唯一标识符。

```python
import numpy as np
import rust_pyfunc as rp

# 创建测试数组
arr = np.array([1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0])

# 识别连续段
segments = rp.identify_segments(arr)
# 输出: [1, 1, 2, 2, 2, 3, 3]
```

#### 3.2 最大范围乘积 (find_max_range_product)
找到数组中一对索引(x, y)，使得min(arr[x], arr[y]) * |x-y|的值最大。

```python
import numpy as np
import rust_pyfunc as rp

# 创建测试数组
arr = np.array([5.0, 3.0, 1.0, 4.0, 6.0])

# 查找最大范围乘积
x, y, max_product = rp.find_max_range_product(arr)
```

#### 3.3 区间循环分析 (max_range_loop 和 min_range_loop)
计算序列中每个位置结尾的最长连续子序列长度。

```python
import rust_pyfunc as rp

# 创建测试序列
s = [5.0, 3.0, 1.0, 4.0, 6.0]

# 计算最大值在末尾的最长连续子序列长度
max_lengths = rp.max_range_loop(s)

# 计算最小值在末尾的最长连续子序列长度
min_lengths = rp.min_range_loop(s)

# 允许相等值
max_lengths_with_equal = rp.max_range_loop(s, allow_equal=True)
```

### 4. 文本分析

#### 4.1 文本向量化 (vectorize_sentences)
将两个句子转换为词频向量。

```python
import rust_pyfunc as rp

# 创建两个测试句子
sentence1 = "这是一个测试句子"
sentence2 = "这是另一个测试句子"

# 转换为词频向量
vector1, vector2 = rp.vectorize_sentences(sentence1, sentence2)
```

#### 4.2 Jaccard相似度 (jaccard_similarity)
计算两个句子之间的Jaccard相似度。

```python
import rust_pyfunc as rp

# 创建两个测试句子
sentence1 = "这是一个测试句子"
sentence2 = "这是另一个测试句子"

# 计算Jaccard相似度
similarity = rp.jaccard_similarity(sentence1, sentence2)
```

#### 4.3 最小词编辑距离 (min_word_edit_distance)
计算将一个句子转换为另一个句子所需的最少单词操作次数。

```python
import rust_pyfunc as rp

# 创建两个测试句子
sentence1 = "这是一个测试句子"
sentence2 = "这是另一个测试句子"

# 计算最小词编辑距离
distance = rp.min_word_edit_distance(sentence1, sentence2)
```

### 5. 时间序列峰值分析

#### 5.1 局部峰值识别 (find_local_peaks_within_window)
查找时间序列中价格在指定时间窗口内为局部最大值的点。

```python
import numpy as np
import rust_pyfunc as rp

# 创建时间和价格序列
times = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
prices = np.array([1.0, 3.0, 2.0, 4.0, 1.0])

# 查找2秒窗口内的局部峰值
peaks = rp.find_local_peaks_within_window(times, prices, window=2.0)
```

### 6. 半能量时间分析

#### 6.1 半能量时间 (find_half_energy_time)
计算每一行在其后指定时间窗口内的价格变动能量，并找出首次达到最终能量一半时所需的时间。

```python
import numpy as np
import rust_pyfunc as rp

# 创建时间和价格序列
times = np.array([1.0, 1.1, 1.2, 1.3, 1.4])
prices = np.array([10.0, 10.2, 10.5, 10.3, 10.1])

# 计算达到一半能量所需时间
half_energy_times = rp.find_half_energy_time(times, prices, time_window=5.0)
```

### 7. 协同交易分析

#### 7.1 协同交易量统计 (find_follow_volume_sum_same_price)
计算每一行在其后时间窗口内具有相同price和volume的行的volume总和。

```python
import numpy as np
import rust_pyfunc as rp
import pandas as pd

# 创建示例数据
df = pd.DataFrame({
    'exchtime': [1.0, 1.05, 1.08, 1.15, 1.2],
    'price': [10.0, 10.0, 10.0, 11.0, 10.0],
    'volume': [100, 100, 100, 200, 100]
})

# 计算follow列
df['follow'] = rp.find_follow_volume_sum_same_price(
    df['exchtime'].values,
    df['price'].values,
    df['volume'].values
)
```

#### 7.2 协同交易组标记 (mark_follow_groups)
标记时间窗口内具有相同price和volume的行组。

```python
import numpy as np
import rust_pyfunc as rp
import pandas as pd

# 创建示例数据
df = pd.DataFrame({
    'exchtime': [1.0, 1.05, 1.08, 1.15, 1.2],
    'price': [10.0, 10.0, 10.0, 11.0, 10.0],
    'volume': [100, 100, 100, 200, 100]
})

# 标记协同交易组
df['group'] = rp.mark_follow_groups(
    df['exchtime'].values,
    df['price'].values,
    df['volume'].values
)
```

### 8. 价格树分析

#### 8.1 价格树 (PriceTree)
用于分析价格序列的层次关系和分布特征。

```python
import numpy as np
import rust_pyfunc as rp

# 创建时间、价格和成交量序列
times = np.array([1000, 1001, 1002], dtype=np.int64)
prices = np.array([10.0, 11.0, 10.5], dtype=np.float64)
volumes = np.array([100.0, 200.0, 150.0], dtype=np.float64)

# 创建价格树
tree = rp.PriceTree()
tree.build_tree(times, prices, volumes)

# 获取树结构
structure = tree.get_tree_structure()
print(structure)

# 获取统计特征
stats = tree.get_tree_statistics()
for name, value in stats:
    print(f"{name}: {value}")

# 访问特定属性
total_nodes = tree.total_nodes
leaf_nodes = tree.leaf_nodes
avg_depth = tree.avg_depth
```

### 9. 熵变分析

#### 9.1 创新高点熵变 (calculate_shannon_entropy_change)
计算价格创新高时的香农熵变化。

```python
import numpy as np
import rust_pyfunc as rp

# 创建测试数据
exchtime = np.array([1e9, 2e9, 3e9, 4e9], dtype=np.float64)
order = np.array([100, 200, 300, 400], dtype=np.int64)
volume = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float64)
price = np.array([100.0, 102.0, 101.0, 103.0], dtype=np.float64)

# 计算3秒窗口的香农熵变
entropy_changes = rp.calculate_shannon_entropy_change(exchtime, order, volume, price, 3.0)

# 只计算价格最高的2个点的熵变
top_entropy_changes = rp.calculate_shannon_entropy_change(exchtime, order, volume, price, 3.0, top_k=2)
```

#### 9.2 创新低点熵变 (calculate_shannon_entropy_change_at_low)
计算价格创新低时的香农熵变化。

```python
import numpy as np
import rust_pyfunc as rp

# 创建测试数据
exchtime = np.array([1e9, 2e9, 3e9, 4e9], dtype=np.float64)
order = np.array([100, 200, 300, 400], dtype=np.int64)
volume = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float64)
price = np.array([100.0, 102.0, 101.0, 103.0], dtype=np.float64)

# 计算3秒窗口的香农熵变
entropy_changes = rp.calculate_shannon_entropy_change_at_low(exchtime, order, volume, price, 3.0)

# 只计算价格最低的2个点的熵变
bottom_entropy_changes = rp.calculate_shannon_entropy_change_at_low(exchtime, order, volume, price, 3.0, bottom_k=2)
```

### 10. 数值计算

#### 10.1 最速曲线 (brachistochrone_curve)
计算最速曲线（投掷线）并返回x_series对应的y坐标。

```python
import numpy as np
import rust_pyfunc as rp
import pandas as pd
import matplotlib.pyplot as plt

# 创建x序列
x_vals = np.linspace(0, 5, 100)

# 计算从点(0,0)到点(5,-3)的最速曲线
y_vals = rp.brachistochrone_curve(0.0, 0.0, 5.0, -3.0, x_vals)

# 设置超时时间
try:
    y_vals = rp.brachistochrone_curve(0.0, 0.0, 5.0, -3.0, x_vals, timeout_seconds=5.0)
except RuntimeError as e:
    print(f"计算超时: {e}")

# 绘制曲线
plt.figure(figsize=(10, 6))
plt.plot(x_vals, y_vals)
plt.scatter([0, 5], [0, -3], color='red', s=50)
plt.grid(True)
plt.title('最速曲线 (Brachistochrone Curve)')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
```

#### 10.2 最大特征值计算 (compute_max_eigenvalue)
计算二维方阵的最大特征值和对应的特征向量。

```python
import numpy as np
import rust_pyfunc as rp

# 创建测试矩阵
matrix = np.array([[4.0, -1.0], 
                    [-1.0, 3.0]], dtype=np.float64)

# 计算最大特征值和特征向量
eigenvalue, eigenvector = rp.compute_max_eigenvalue(matrix)
print(f"最大特征值: {eigenvalue}")
print(f"对应的特征向量: {eigenvector}")
```

## 注意事项

1. 所有函数都经过Rust优化，相比Python原生实现有显著的性能提升
2. 输入数据需要符合函数要求的格式和类型
3. 部分函数（如`transfer_entropy`）的参数需要根据具体场景调整以获得最佳结果
4. 对于需要处理大量数据的场景，建议使用numpy数组作为输入以获得更好的性能
5. 在使用`PriceTree`等复杂数据结构时，注意及时释放资源

## License

MIT License
