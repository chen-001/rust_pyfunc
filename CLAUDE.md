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
```bash
# Primary development command - builds Rust and installs Python package
maturin develop --release
```
增加新函数后，要在rust_pyfunc.pyi中添加函数声明



### Testing
```bash
# Run individual test files (no unified test framework)
python tests/simple_performance_test.py
python tests/test_dtw_optimization.py
python tests/test_corrwith_performance.py
python tests/full_performance_test.py
```

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