# CRUSH.md

本仓库是基于 Rust + PyO3 + maturin 的高性能 Python 扩展库。以下为在本仓库中常用的构建/测试命令与代码风格规范，供智能体快速参考与执行。

## 构建/发布/文档
- 构建并查看日志: ./alter.sh 2>&1
- 本地开发构建(生成可加载的Python扩展): /home/chenzongwei/.conda/envs/chenzongwei311/bin/maturin develop -r
- 仅构建wheel: /home/chenzongwei/.conda/envs/chenzongwei311/bin/maturin build -r
- 生成API文档(需依赖jinja2/markdown/numpy/pandas/graphviz/IPython): python docs_generator.py

## 测试
- 建议将测试文件放在 tests/ 下
- 运行全部Rust单元测试: cargo test --all --release
- 运行指定测试(按模块名或测试名过滤): cargo test --release <pattern>
- Python侧一致性/性能对比测试: pytest -q tests  或 python -m pytest -q tests

## Lint/类型检查
- Rust格式化: cargo fmt --all
- Rust静态检查: cargo clippy --all-targets --all-features -D warnings
- Python类型检查: mypy --config-file pyproject.toml python

## 代码风格(综合 Cursor 规则与项目约定)
- 导入/模块
  - Rust: 按标准库、第三方、crate内模块顺序分组；pub 接口集中于 src/lib.rs 暴露；新增函数新建文件，不追加到 mod.rs；在 python/*.pyi 中补充声明
  - Python: 类型存根在 python/rust_pyfunc/*.pyi，同步更新签名与文档
- 格式/命名
  - Rust: rustfmt 默认；类型/结构体 PascalCase，函数/变量 snake_case，常量 SCREAMING_SNAKE_CASE
  - Python: PEP8；公开API具备明确类型注解并在 .pyi 中维护
- 类型/错误
  - Rust: 返回 Result<T, crate::error::PyFuncError> 或使用 PyResult<T>；避免 unwrap/expect；必要时使用 anyhow->错误信息转换到 PyErr
  - Python: 不泄漏密钥与路径；参数校验后再入 FFI
- 并行/性能
  - 合理使用 rayon；避免不必要的分配与拷贝；优先 ndarray/nalgebra 原地操作；SIMD 可用 packed_simd_2
- 文档
  - Rust 使用 /// 注释；Python 在 .pyi 中补充文档字符串；新增函数需补 docs/ 对应页面或示例

## 环境路径
- Python: /home/chenzongwei/.conda/envs/chenzongwei311/bin/python
- Pip: /home/chenzongwei/.conda/envs/chenzongwei311/bin/pip
- Maturin: /home/chenzongwei/.conda/envs/chenzongwei311/bin/maturin
