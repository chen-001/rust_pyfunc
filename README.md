# Rust_Pyfunc
---

一些用python计算起来很慢的指标，这里用rust来实现，提升计算速度。

* 安装
```shell
pip install rust_pyfunc
```

* 使用
```python
import rust_pyfunc as rp
```

* 目前支持的指标：
1. DTW动态时间规整
```python
import rust_pyfunc as rp

a=[1,2,3,4]
b=[3,9,8,6,5]
res=rp.dtw_distance(a,b)
```

2. transfer_entropy转移熵
```python
import rust_pyfunc as rp

a=[1,2,3,4]
b=[3,9,8,6]
res=rp.transfer_entropy(a,b)
```

