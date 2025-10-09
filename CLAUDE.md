# CLAUDE.md

always respond in Chinese.

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A high-performance Python library implementing computationally intensive algorithms in Rust using PyO3 bindings. The library focuses on financial data analysis, time series processing, statistical calculations, and mathematical functions that are significantly faster than pure Python implementations.

## Development Environment

**Python Environment:**
- Python path: `/home/chenzongwei/.conda/envs/chenzongwei311/bin/python`
- Pip path: `/home/chenzongwei/.conda/envs/chenzongwei311/bin/pip`
- Maturin path: `/home/chenzongwei/.conda/envs/chenzongwei311/bin/maturin`

## Common Development Commands

### Build and Development

- 增加新函数后，要在对应的*.pyi中添加函数声明
- 使用./alter.sh 2>&1来构建项目并查看成功或报错信息

### Testing

- 生成测试文件时，不要直接生成在rust_pyfunc文件夹下，而是存储在tests文件夹中。
- 在编写了rust新函数后，请使用python代码实现同样的功能，并比较二者的是否一致，以及速度差异如何。

### Documentation
```bash
# Generate API documentation (requires jinja2, markdown, numpy, pandas, graphviz, IPython)
python docs_generator.py
```

## Architecture

### Core Structure
- **`src/lib.rs`** - Main PyO3 module definition with all function exports
- **`src/`** - Rust implementation modules:
  - `time_series/` - DTW, trend analysis, peak detection, rolling calculations
  - `statistics/` - OLS regression, rolling statistics, eigenvalue calculations
  - `sequence/` - Segment identification, range analysis, entropy calculations
  - `text/` - Text similarity and vectorization functions
  - `tree/` - Price tree data structure for hierarchical analysis
  - `pandas_ext/` - Pandas integration utilities
  - `error/` - Custom error handling

### Python Integration
- **`python/rust_pyfunc/__init__.py`** - Python package entry point
- **`python/rust_pyfunc/*.py`** - Additional Python utilities and pandas extensions
- **`python/rust_pyfunc/rust_pyfunc.pyi`** - Type stubs for IDE support

### Key Dependencies
- **PyO3** - Rust-Python bindings
- **maturin** - Build system for Rust-Python packages
- **ndarray/numpy** - Array operations
- **nalgebra** - Linear algebra
- **rayon** - Parallel processing

## Development Guidelines

### Code Style
- When adding new Rust functions to be called from Python, update the corresponding `.pyi` file for proper type hints and IDE support
- Use Altair or Plotly for data visualization, avoid Matplotlib
- Add appropriate comments for code readability
- Only modify code relevant to the specific changes being made

### Adding New Functions
1. Implement the function in the appropriate `src/` module
2. Add the function export in `src/lib.rs` (line ~21-65)
3. Update `python/rust_pyfunc/rust_pyfunc.pyi` with proper type hints
4. Add documentation and examples following the existing pattern
5. Create test cases in `tests/` directory

### Performance Optimization
- Use `#[pyfunction]` macro for Python-callable functions
- Leverage `rayon` for parallel processing where appropriate
- Use SIMD instructions when available (`packed_simd_2`)
- Profile with `criterion` benchmarks (in `[dev-dependencies]`)

### Release Configuration
The project uses aggressive optimization settings:
```toml
[profile.release]
lto = true
codegen-units = 1
panic = "abort"
opt-level = 3
```

## CI/CD
- Multi-platform builds (Linux, macOS, Windows) via GitHub Actions
- Automatic documentation deployment to GitHub Pages
- PyPI publishing on tag creation

## Stock Data Context
The project includes utilities for working with Chinese stock market data through the `design_whatever` library, supporting L2 tick data, market snapshots, and minute-level aggregations.


### 数据结构说明

**Trade Data字段**：
- exchtime: 交易时间
- price: 成交价格
- volume: 成交量 
- turnover: 成交金额
- flag: 交易标志 (66=主买, 83=主卖, 32=撤单)
- ask_order/bid_order: 订单编号

**Market Data字段**：
- 基础信息: symbol, exchtime, last_prc, prev_close, open, high, low
- 限价信息: high_limited, low_limited  
- 成交统计: volume, turnover, num_trades
- 十档行情: ask_prc1-10, ask_vol1-10, bid_prc1-10, bid_vol1-10
- 加权价格: weighted_ask_prc, weighted_bid_prc

### 使用注意事项
- 日期格式统一使用8位整数：YYYYMMDD (如 20220819)
- symbols参数为股票代码列表，如 ['000001', '600000']
- 所有时间字段均为datetime64[ns]格式，便于pandas时间序列分析
- 分钟数据返回格式：index为时间，columns为股票代码
```

## 代码开发指南

### 代码组织原则
- **写新函数时,不要在mod.rs中追加,可以创建一个新的文件,然后在lib.rs中添加导入即可**

### 开发备忘录
- **写了新函数之后,只要写一些测试文件即可,不需要写演示文件**

### 代码命名规范
- **写新函数时，函数名字要具体，可以简单概括函数的核心计算内容与逻辑**