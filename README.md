# rust_pyfunc

用 Rust 写的高性能 Python 库。专治各种计算密集场景：金融数据分析、时间序列处理、统计计算——Python 搞不定的性能问题，这里管。

## 安装

```shell
pip install rust_pyfunc
```

## 使用

```python
import rust_pyfunc as rp
```

## 开发

```bash
pip install maturin
maturin develop --release
```

## 项目结构

```
src/          # Rust 源码
python/       # Python 包 & 类型提示
```

## License

MIT
