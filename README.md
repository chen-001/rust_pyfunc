# Rust_Pyfunc [![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/chen-001/rust_pyfunc)

一个专注于高性能计算的Python库，通过Rust实现计算密集型算法，专门为金融数据分析、时间序列处理和统计计算提供显著的速度提升。

## 安装
```shell
pip install rust_pyfunc
```

## 使用
```python
import rust_pyfunc as rp
```

## 贡献指南

### 贡献原则

1. **性能优先**: 只有能显著提升性能的函数才考虑用Rust实现
2. **安全第一**: 所有代码必须通过内存安全检查，避免越界访问
3. **接口清晰**: Python接口要简洁易用，类型提示完整
4. **文档完善**: 每个函数都需要详细的文档和使用示例
5. **测试充分**: 必须包含正确性测试和性能对比测试

### 欢迎所有类型的贡献

我们欢迎并感谢所有形式的贡献！无论你的技能水平如何，都有适合你的贡献方式：

🚀 **功能贡献**:
- 新的算法实现（数值计算、统计分析、机器学习等）
- 性能优化（加速现有函数、并行化处理）
- 新模块开发（创建全新的功能领域）

📝 **文档贡献**:
- 改进函数文档和使用示例
- 添加教程和最佳实践指南
- 翻译文档到其他语言

🧪 **测试和质量保证**:
- 添加更多测试用例
- 改进测试覆盖率
- 发现和修复bug

🔧 **工程改进**:
- CI/CD流程优化
- 构建系统改进
- 配置文件和脚本优化

💡 **想法和建议**:
- 提出新功能请求
- 报告问题和改进建议
- 参与设计讨论

### 开发环境设置

**必要工具**:
```bash
# 1. 安装Rust工具链
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env

# 2. 确保Python环境（支持Python 3.8+）
# 可以使用现有的Python环境，无需创建新环境
python --version  # 确认版本 >= 3.8

# 3. 安装构建工具
pip install maturin

# 4. 安装可选的开发依赖
pip install numpy pandas pytest  # 用于测试和验证
```

**Fork和克隆项目**:
```bash
# 1. 在GitHub上Fork项目到你的账户
# 访问 https://github.com/original-repo/rust_pyfunc 点击Fork

# 2. 克隆你的Fork
git clone https://github.com/your-username/rust_pyfunc.git
cd rust_pyfunc

# 3. 添加原始仓库为upstream
git remote add upstream https://github.com/original-repo/rust_pyfunc.git

# 4. 验证环境
maturin --version
```

### Maturin构建系统使用指南

Maturin是Rust-Python项目的专用构建工具，以下是常用命令：

**基础构建命令**:
```bash
# 开发模式构建（推荐用于开发）
maturin develop
# 等价于项目中的 ./alter.sh

# 查看详细构建输出
maturin develop --verbose

# 释放模式构建（用于最终发布）
maturin develop --release
```

**构建选项说明**:
```bash
# 仅构建Rust部分，不安装到Python
maturin build

# 构建wheel包
maturin build --release

# 指定Python解释器
maturin develop --python /path/to/python

# 构建时显示更多信息
maturin develop --verbose
```

**常见问题解决**:
```bash
# 如果遇到链接错误，尝试清理后重新构建
cargo clean
maturin develop

# 如果Python找不到模块，确认安装成功
python -c "import rust_pyfunc; print('✅ 导入成功')"

# 查看构建的wheel文件
maturin build && ls target/wheels/
```

**性能优化构建**:
```bash
# 最高性能的发布构建
maturin build --release --strip

# 使用项目的优化配置
maturin develop --release  # 使用Cargo.toml中的[profile.release]设置
```

### 添加新函数的步骤

#### 第一步：规划和设计

1. **选择或创建模块**：
   
   **现有模块**（可选择加入）：
   - `time_series/` - 时间序列分析
   - `statistics/` - 统计计算
   - `sequence/` - 序列分析
   - `text/` - 文本处理
   - `parallel_computing/` - 并行计算
   - `trading_analysis/` - 交易分析

   **创建新模块**（推荐方式）：
   ```bash
   # 创建新的功能模块
   mkdir src/your_new_module
   touch src/your_new_module/mod.rs
   touch src/your_new_module/your_function.rs
   ```

2. **编写Python参考实现**（重要！用于验证正确性）：
```python
# 示例：在tests/目录下创建测试文件
def python_prototype(data):
    """Python实现版本，用于验证正确性"""
    # 实现算法逻辑
    return result
```

#### 第二步：Rust实现

1. **创建或修改Rust模块**（推荐创建新文件）：
```rust
// src/your_module/your_function.rs
use pyo3::prelude::*;
use numpy::{PyArray1, PyReadonlyArray1};

#[pyfunction]
pub fn your_function_name(
    input: PyReadonlyArray1<f64>,
    param: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    // Rust实现
    // 注意：添加边界检查和错误处理
    todo!()
}
```

2. **在模块中声明函数**：
```rust
// src/your_new_module/mod.rs
pub mod your_function;
pub use your_function::*;
```

3. **在lib.rs中添加模块和导出函数**：
```rust
// src/lib.rs 
// 添加模块声明（文件顶部附近）
mod your_new_module;

// 在#[pymodule]函数中导出（约21-65行）
#[pymodule]
fn rust_pyfunc(_py: Python, m: &PyModule) -> PyResult<()> {
    // ... existing functions ...
    m.add_function(wrap_pyfunction!(your_new_module::your_function_name, m)?)?;
    Ok(())
}
```

**新模块的完整结构**：
```
src/your_new_module/
├── mod.rs          # 模块声明文件
├── your_function.rs # 具体函数实现
└── utils.rs        # （可选）辅助函数
```

#### 第三步：添加类型提示

**创建新的类型提示文件**（如果是新模块）：
```python
# python/rust_pyfunc/your_new_module.pyi
"""Your new module type hints"""
from typing import Optional
import numpy as np
from numpy.typing import NDArray

def your_function_name(
    input: NDArray[np.float64],
    param: float,
) -> NDArray[np.float64]:
    """详细的函数文档
    
    参数说明：
    ----------
    input : NDArray[np.float64]
        输入数据描述
    param : float
        参数描述
        
    返回值：
    -------
    NDArray[np.float64]
        返回值描述
        
    示例：
    -------
    >>> import rust_pyfunc as rp
    >>> result = rp.your_function_name(data, 1.5)
    """
    ...
```

如果是加入现有模块，则在对应的.pyi文件中添加函数声明即可。

#### 第四步：编写测试

**在tests/目录下创建测试文件**：
```python
# tests/test_your_function.py
import numpy as np
import rust_pyfunc as rp
import time

def python_reference_implementation(data, param=1.5):
    """Python参考实现 - 必须先实现这个！
    
    这个函数是验证Rust实现正确性的金标准。
    请用最直观、最容易理解的方式实现算法逻辑。
    """
    import numpy as np
    # 这里实现你的算法逻辑
    # 例如：计算某种移动平均
    result = []
    for i in range(len(data)):
        if i == 0:
            result.append(data[i])
        else:
            # 指数移动平均示例
            result.append(param * data[i] + (1 - param) * result[-1])
    return np.array(result)

def test_correctness():
    """正确性验证 - 最重要的测试！"""
    # 生成多种测试数据
    test_cases = [
        np.random.randn(100),           # 随机数据
        np.arange(50, dtype=float),     # 递增序列
        np.ones(30) * 5.0,              # 常数序列
        np.array([1.0, -1.0] * 25),     # 交替序列
    ]
    
    for i, data in enumerate(test_cases):
        print(f"🧪 测试用例 {i+1}: 长度={len(data)}")
        
        # Python参考实现
        python_result = python_reference_implementation(data, 1.5)
        
        # Rust实现
        rust_result = rp.your_function_name(data, 1.5)
        
        # 严格比较结果
        np.testing.assert_allclose(
            rust_result, python_result, 
            rtol=1e-12, atol=1e-15,
            err_msg=f"测试用例 {i+1} 失败"
        )
        print(f"   ✅ 测试用例 {i+1} 通过")
    
    print("🎉 所有正确性测试通过！")

def test_performance():
    """性能对比测试 - 展示Rust的优势"""
    print("🏃‍♂️ 开始性能测试...")
    
    # 多种规模的测试数据
    test_sizes = [1000, 10000, 100000]
    
    for size in test_sizes:
        print(f"\n📊 测试数据规模: {size:,} 个元素")
        data = np.random.randn(size)
        
        # 预热（避免首次调用的开销）
        _ = python_reference_implementation(data[:100])
        _ = rp.your_function_name(data[:100], 1.5)
        
        # 测试Python版本
        print("   ⏱️  测试Python实现...")
        python_times = []
        for _ in range(10):  # 多次测试取平均
            start = time.perf_counter()
            python_result = python_reference_implementation(data, 1.5)
            python_times.append(time.perf_counter() - start)
        avg_python_time = sum(python_times) / len(python_times)
        
        # 测试Rust版本
        print("   ⚡ 测试Rust实现...")
        rust_times = []
        for _ in range(10):  # 多次测试取平均
            start = time.perf_counter()
            rust_result = rp.your_function_name(data, 1.5)
            rust_times.append(time.perf_counter() - start)
        avg_rust_time = sum(rust_times) / len(rust_times)
        
        # 计算性能提升
        speedup = avg_python_time / avg_rust_time
        
        print(f"   📈 Python: {avg_python_time*1000:.2f}ms")
        print(f"   🚀 Rust:   {avg_rust_time*1000:.2f}ms")
        print(f"   ⚡ 加速比: {speedup:.1f}x")
        
        # 验证结果一致性
        np.testing.assert_allclose(rust_result, python_result, rtol=1e-10)
        print(f"   ✅ 结果验证通过")
    
    print(f"\n🎯 性能测试完成！Rust实现展现了显著的性能优势。")

if __name__ == "__main__":
    test_correctness()
    test_performance()
```

#### 第五步：构建和测试

```bash
# 构建项目
maturin develop
# 或使用项目的快捷脚本
./alter.sh 2>&1

# 运行测试
python tests/test_your_function.py

# 验证导入
python -c "import rust_pyfunc as rp; print('✅ 模块导入成功')"
python -c "import rust_pyfunc as rp; print(dir(rp))"  # 查看所有可用函数
```

**重要提醒**：
- ⚠️ **必须先写Python参考实现**，用于验证Rust版本的正确性
- 🧪 **多种测试用例**：随机数据、边界条件、特殊值
- ⚡ **性能测试**：展示Rust相对于Python的加速效果
- 📝 **详细文档**：函数用途、参数、返回值、使用示例

### 代码规范

#### Rust代码规范

```rust
// ✅ 好的示例
#[pyfunction]
pub fn calculate_moving_average(
    data: PyReadonlyArray1<f64>,
    window: usize,
) -> PyResult<Py<PyArray1<f64>>> {
    let data = data.as_array();
    let n = data.len();
    
    // 边界检查
    if window == 0 || window > n {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "窗口大小必须在1到数据长度之间"
        ));
    }
    
    let mut result = Vec::with_capacity(n - window + 1);
    
    // 使用迭代器和并行处理（如果适用）
    for i in 0..=(n - window) {
        let sum: f64 = data.slice(s![i..i + window]).sum();
        result.push(sum / window as f64);
    }
    
    Python::with_gil(|py| {
        Ok(PyArray1::from_vec(py, result).to_owned())
    })
}
```

#### 必须遵循的安全规范

1. **边界检查**: 所有数组访问都要检查边界
2. **错误处理**: 使用`PyResult`和适当的错误类型
3. **内存管理**: 避免内存泄漏，正确使用`Python::with_gil`
4. **并发安全**: 如果使用并行，确保线程安全

#### 性能优化建议

```rust
// 使用rayon并行处理
use rayon::prelude::*;

// 并行版本
let result: Vec<f64> = (0..=n-window)
    .into_par_iter()
    .map(|i| {
        let sum: f64 = data.slice(s![i..i + window]).sum();
        sum / window as f64
    })
    .collect();
```

### 文档要求

每个函数都需要包含：

1. **功能描述**: 清晰说明函数用途
2. **参数说明**: 每个参数的类型、含义、约束
3. **返回值说明**: 返回值类型和含义
4. **性能特性**: 时间复杂度、预期速度提升
5. **使用示例**: 至少2个实际使用案例
6. **注意事项**: 使用限制、边界条件

### 贡献流程（Fork + Pull Request）

#### 第一步：Fork和设置

```bash
# 1. 在GitHub上Fork项目
# 访问 https://github.com/original-owner/rust_pyfunc
# 点击右上角的 "Fork" 按钮

# 2. 克隆你的Fork到本地
git clone https://github.com/your-username/rust_pyfunc.git
cd rust_pyfunc

# 3. 添加原仓库为upstream（保持同步用）
git remote add upstream https://github.com/original-owner/rust_pyfunc.git

# 4. 验证远程仓库配置
git remote -v
# origin    https://github.com/your-username/rust_pyfunc.git (fetch)
# origin    https://github.com/your-username/rust_pyfunc.git (push)
# upstream  https://github.com/original-owner/rust_pyfunc.git (fetch)
# upstream  https://github.com/original-owner/rust_pyfunc.git (push)
```

#### 第二步：创建功能分支

```bash
# 1. 确保在main分支且是最新的
git checkout main
git pull upstream main

# 2. 创建并切换到功能分支
git checkout -b feature/your-function-name
# 或者创建修复分支
git checkout -b fix/issue-description
```

#### 第三步：开发和提交

```bash
# 开发你的功能...
# 按照前面的步骤添加Rust实现、类型提示、测试等

# 分阶段提交，保持提交历史清晰
git add src/your_new_module/
git commit -m "feat: 添加your_function_name的Rust实现

- 实现高性能算法XYZ
- 支持多种数据类型输入
- 包含完整的错误处理"

git add python/rust_pyfunc/your_new_module.pyi
git commit -m "docs: 添加your_function_name的类型提示

- 完整的函数签名和文档
- 详细的参数说明和示例"

git add tests/test_your_function.py
git commit -m "test: 添加your_function_name的测试

- 正确性验证测试
- 多场景性能对比测试
- 边界条件测试"
```

#### 第四步：测试和验证

```bash
# 构建项目
maturin develop

# 运行你的测试
python tests/test_your_function.py

# 运行所有测试（确保没有破坏现有功能）
python -m pytest tests/ -v

# 检查代码格式（如果项目有linting配置）
cargo fmt --check
cargo clippy
```

#### 第五步：推送和创建Pull Request

```bash
# 推送到你的Fork
git push origin feature/your-function-name

# 如果是第一次推送这个分支
git push -u origin feature/your-function-name
```

然后在GitHub上：
1. 访问你的Fork页面
2. GitHub会提示创建Pull Request，点击 "Compare & pull request"
3. 填写PR信息（见下面的模板）

#### Pull Request模板

**标题格式**：
- `feat: 添加your_function_name函数` （新功能）
- `fix: 修复issue_description` （bug修复）
- `docs: 改进documentation_part` （文档改进）
- `test: 添加test_description` （测试改进）

**PR描述模板**：
```markdown
## 📝 变更描述
简要描述此PR的目的和实现的功能

## ✨ 新增功能
- 实现了高性能的XXX算法
- 支持YYY数据类型
- 提供ZZZ接口

## 🚀 性能提升
- 相比Python实现提升 XX 倍
- 处理100万数据点仅需 XX ms
- 内存使用减少 XX%

## 🧪 测试情况
- [x] 正确性测试通过
- [x] 性能测试完成
- [x] 边界条件测试
- [x] 现有功能回归测试

## 📊 性能测试结果
```
数据规模    | Python时间 | Rust时间  | 加速比
----------|-----------|----------|-------
1,000     | 10.5ms    | 0.8ms    | 13.1x
10,000    | 105ms     | 7.2ms    | 14.6x
100,000   | 1.05s     | 72ms     | 14.6x
```

## 🔗 相关Issue
修复 #issue_number （如果有相关issue）

## ✅ 检查清单
- [x] 代码编译通过
- [x] 所有测试通过
- [x] 添加了类型提示
- [x] 更新了文档
- [x] 遵循了代码规范
```

### 示例：完整的贡献流程

假设我们要添加一个计算指数移动平均(EMA)的函数，完整的贡献过程如下：

#### 准备阶段
```bash
# 1. Fork项目并克隆
git clone https://github.com/your-username/rust_pyfunc.git
cd rust_pyfunc
git remote add upstream https://github.com/original-owner/rust_pyfunc.git

# 2. 创建功能分支
git checkout main
git pull upstream main
git checkout -b feature/exponential-moving-average
```

#### 开发阶段
```bash
# 3. 创建新模块结构
mkdir -p src/moving_averages
touch src/moving_averages/mod.rs
touch src/moving_averages/ema.rs

# 4. 实现Python参考版本（重要！）
# tests/test_ema.py - 先写Python实现用于验证
```

**Python参考实现**：
```python
# tests/test_ema.py
import numpy as np

def python_ema(data, alpha=0.1):
    """Python参考实现 - 指数移动平均"""
    result = np.zeros_like(data)
    result[0] = data[0]
    for i in range(1, len(data)):
        result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
    return result
```

**Rust实现**：
```rust
// src/moving_averages/ema.rs
use pyo3::prelude::*;
use numpy::{PyArray1, PyReadonlyArray1};

#[pyfunction]
pub fn exponential_moving_average(
    data: PyReadonlyArray1<f64>,
    alpha: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    let data = data.as_array();
    let n = data.len();
    
    if alpha <= 0.0 || alpha > 1.0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "alpha必须在(0, 1]范围内"
        ));
    }
    
    let mut result = Vec::with_capacity(n);
    if n > 0 {
        result.push(data[0]);
        for i in 1..n {
            let ema = alpha * data[i] + (1.0 - alpha) * result[i-1];
            result.push(ema);
        }
    }
    
    Python::with_gil(|py| {
        Ok(PyArray1::from_vec(py, result).to_owned())
    })
}
```

#### 集成和测试
```bash
# 5. 添加到模块系统
echo "pub mod ema;" >> src/moving_averages/mod.rs
echo "pub use ema::*;" >> src/moving_averages/mod.rs

# 在src/lib.rs中添加
# mod moving_averages;
# m.add_function(wrap_pyfunction!(moving_averages::exponential_moving_average, m)?)?;

# 6. 创建类型提示文件
# python/rust_pyfunc/moving_averages.pyi

# 7. 构建和测试
maturin develop
python tests/test_ema.py

# 8. 分阶段提交
git add src/moving_averages/
git commit -m "feat: 实现指数移动平均(EMA)的Rust核心算法

- 高性能EMA计算，支持任意alpha参数
- 完整的边界检查和错误处理
- 内存高效的向量化实现"

git add python/rust_pyfunc/moving_averages.pyi
git commit -m "docs: 添加EMA函数的类型提示和文档

- 完整的函数签名和参数说明
- 详细的使用示例和注意事项"

git add tests/test_ema.py
git commit -m "test: 添加EMA函数的完整测试套件

- Python参考实现用于正确性验证
- 多规模性能基准测试
- 边界条件和错误处理测试
- 测试结果：比纯Python实现快25倍"
```

#### 提交PR
```bash
# 9. 推送到你的Fork
git push -u origin feature/exponential-moving-average

# 10. 在GitHub上创建Pull Request
# 标题：feat: 添加指数移动平均(EMA)计算函数
# 使用前面提供的PR模板填写描述
```

**预期性能测试结果**：
```
📊 EMA性能测试结果:
数据规模    | Python时间 | Rust时间  | 加速比
----------|-----------|----------|-------
1,000     | 2.1ms     | 0.08ms   | 26.3x
10,000    | 21ms      | 0.8ms    | 26.3x
100,000   | 210ms     | 8ms      | 26.3x
```

这个完整示例展示了：
- ✅ Fork + PR的标准协作流程
- ✅ 创建新模块的完整过程
- ✅ Python参考实现的重要性
- ✅ 分阶段清晰的git提交历史
- ✅ 完整的测试和验证流程

### 贡献者支持

如果在贡献过程中遇到问题：

1. **查看现有代码**: 参考`src/time_series/`等模块的实现
2. **运行测试**: 使用`python -m pytest tests/`检查回归
3. **性能分析**: 使用`criterion`进行基准测试
4. **提出Issue**: 在GitHub上描述遇到的问题

我们欢迎任何形式的贡献，从bug修复到新功能实现！

## License

MIT License
