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
使用alter.sh来完成构建（即替代直接的maturin develop）
增加新函数后，要在rust_pyfunc.pyi中添加函数声明

- 使用./alter.sh 2>&1来构建项目并查看成功或报错信息

### Testing
生成测试文件时，不要直接生成在rust_pyfunc文件夹下，而是存储在tests文件夹中。
在编写了rust新函数后，请使用python代码实现同样的功能，并比较二者的是否一致，以及速度差异如何。

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



## 股票数据获取接口

### 核心数据读取函数
使用 `design_whatever` (简称 `dw`) 模块进行各类股票数据读取：

#### 逐笔成交数据 (L2 Trade Data)
```python
import design_whatever as dw

# 读取逐笔成交数据（不包含撤单）
# flag=66: 主动买入, flag=83: 主动卖出
dw.read_l2_trade_data(start_date=20220819, symbols=['000001'], with_retreat=0)
# 若是在class Go(dw.TickDataBase):中读取数据，则使用下面语句
# Go是对每个股票每天的数据循环读取计算的class，因此在每个循环中，下面语句读取的就是当前循环的股票和日期下的逐笔成交数据
self.read_trade_data_single(with_retreat=0)

# 读取逐笔成交数据（包含撤单）  
# flag=66: 主动买入, flag=83: 主动卖出, flag=32: 撤单
dw.read_l2_trade_data(start_date=20220819, symbols=['000001'], with_retreat=1)
# 若是在class Go(dw.TickDataBase):中读取数据，则使用下面语句
# Go是对每个股票每天的数据循环读取计算的class，因此在每个循环中，下面语句读取的就是当前循环的股票和日期下的逐笔成交数据
self.read_trade_data_single(with_retreat=1)
```

#### 盘口快照数据 (L2 Market Data)
```python
# 标准盘口快照格式（包含10档买卖价格/挂单量）
dw.read_l2_market_data(start_date=20220819, symbols=['000001'])
# 若是在class Go(dw.TickDataBase):中读取数据，则使用下面语句
# Go是对每个股票每天的数据循环读取计算的class，因此在每个循环中，下面语句读取的就是当前循环的股票和日期下的盘口快照数据
self.read_market_data_single()

# 挂单价格-挂单量对格式
asks, bids = dw.read_l2_market_data_price_vol_pair(date=20220819, symbols=['000001'])
# 若是在class Go(dw.TickDataBase):中读取数据，则使用下面语句
# Go是对每个股票每天的数据循环读取计算的class，因此在每个循环中，下面语句读取的就是当前循环的股票和日期下的盘口快照价格挂单量格式数据
self.read_market_data_single_price_vol_pair()
```

#### 分钟数据 (Minute Data)
```python
# 读取分钟级别聚合数据，支持多个字段
dw.read_minute_data(field='close', start_date=20220819, end_date=20220819)
# 若是在class Go(dw.TickDataBase):中读取数据，则使用下面语句
# Go是对每个股票每天的数据循环读取计算的class，因此在每个循环中，下面语句读取的就是当前循环的股票和日期下的盘口快照数据
self.read_market_data_single()
```

**分钟数据可用字段详表**：

| 字段类别 | 字段名 | 描述 |
|---------|--------|------|
| **基础价格** | close | 1分钟最后一秒的last price |
| | open | 1分钟第一秒的last price |
| | high | 1分钟内最高价 |
| | low | 1分钟内最低价|
| **成交统计** | volume | 1分钟内transaction的net_b_vol+net_s_vol |
| | amount | 1分钟内transaction的主买交易金额+主卖交易金额 |
| | up_tick_count | 1分钟内的uptick成交的笔数 |
| | down_tick_count | 1分钟内的downtick成交的笔数 |
| **主动买卖** | act_buy_amount_sum | 1分钟内主买成交金额总和 |
| | act_sell_amount_sum | 1分钟内主卖成交金额总和 |
| | act_buy_vol_sum | 1分钟内主买volume总和 |
| | act_sell_vol_sum | 1分钟内主卖volume总和 |
| | act_buy_count_sum | 1分钟内主买笔数总和 |
| | act_sell_count_sum | 1分钟内主卖笔数总和 |
| **盘口均值** | bid_size_1_mean | 1分钟内bid_size_1的均值 |
| | ask_size_1_mean | 1分钟内ask_size_1的均值 |
| | bid_size_6_mean | 1分钟内1-6档行情的bid_vol之和的均值 |
| | ask_size_6_mean | 1分钟内1-6档行情的ask_vol之和的均值 |
| | bid_size_10_mean | 1分钟内1-10档行情的bid_vol之和的均值 |
| | ask_size_10_mean | 1分钟内1-10档行情的ask_vol之和的均值 |
| | bid_vwap_mean | 1分钟内1-10档vwap（bid）的均值 |
| | ask_vwap_mean | 1分钟内1-10档vwap（ask）的均值 |
| **买方价格** | bid_prc1~bid_prc10 | 1分钟快照最后一帧的买方1-10档价格 |
| **买方挂单量** | bid_vol1~bid_vol10 | 1分钟快照最后一帧的买方1-10档挂单量 |
| **卖方价格** | ask_prc1~ask_prc10 | 1分钟快照最后一帧的卖方1-10档价格 |
| **卖方挂单量** | ask_vol1~ask_vol10 | 1分钟快照最后一帧的卖方1-10档挂单量 |
| **加权平均价** | bid_vwap3 | 1分钟快照最后一帧买方1-3档的volume加权平均价 |
| | bid_vwap5 | 1分钟快照最后一帧买方1-5档的volume加权平均价 |
| | bid_vwap10 | 1分钟快照最后一帧买方1-10档的volume加权平均价 |
| | ask_vwap3 | 1分钟快照最后一帧卖方1-3档的volume加权平均价 |
| | ask_vwap5 | 1分钟快照最后一帧卖方1-5档的volume加权平均价 |
| | ask_vwap10 | 1分钟快照最后一帧卖方1-10档的volume加权平均价 |

**使用示例**：
```python
# 读取基础OHLC数据
ohlc_data = dw.read_minute_data(field='close', start_date=20220819, end_date=20220819)

# 读取主动买卖数据  
buy_amount = dw.read_minute_data(field='act_buy_amount_sum', start_date=20220819, end_date=20220819)
sell_amount = dw.read_minute_data(field='act_sell_amount_sum', start_date=20220819, end_date=20220819)

# 读取盘口数据
bid1_price = dw.read_minute_data(field='bid_prc1', start_date=20220819, end_date=20220819)
ask1_volume = dw.read_minute_data(field='ask_vol1', start_date=20220819, end_date=20220819)

# 读取加权平均价数据
bid_vwap = dw.read_minute_data(field='bid_vwap5', start_date=20220819, end_date=20220819)
```

#### 辅助查询函数
```python
# 获取可用的时间-股票对
dw.read_level2_list(start_date=20170101, end_date=20250228)
```

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