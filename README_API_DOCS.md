# Rust PyFunc API文档生成工具

本项目提供了一个自动化工具，用于为`rust_pyfunc`库生成详细的API文档，并部署到GitHub Pages上。生成的文档包含每个函数的详细说明、参数信息以及基于真实Python运行结果的使用示例。

## 功能特点

- 自动提取所有公开函数的信息
- 解析函数文档字符串，识别参数和返回值说明
- 运行实际代码示例，获取真实结果
- 生成美观的HTML文档
- 通过GitHub Actions自动部署到GitHub Pages

## 安装依赖

首先，需要安装必要的Python依赖：

```bash
pip install jinja2 markdown numpy
```

另外，确保已经安装了`rust_pyfunc`包：

```bash
cd /path/to/rust_pyfunc
pip install -e .
```

## 使用方法

### 步骤1：运行文档生成器

```bash
python docs_generator.py
```

这个脚本会：
1. 提取`rust_pyfunc`库中所有公开函数
2. 解析它们的文档字符串
3. 运行代码示例获取实际结果
4. 生成HTML文档到`docs`目录

### 步骤2：本地预览

生成文档后，可以在浏览器中打开`docs/index.html`文件进行预览。

### 步骤3：部署到GitHub Pages

有两种方式部署到GitHub Pages：

#### 方式一：手动部署

1. 将生成的`docs`目录推送到GitHub仓库
2. 在GitHub仓库设置中，启用GitHub Pages，并选择`gh-pages`分支作为源

#### 方式二：使用GitHub Actions自动部署

本项目已配置GitHub Actions工作流，可以在推送到`main`分支时自动部署文档。

1. 确保`.github/workflows/deploy.yml`文件已添加到仓库
2. 将更改推送到`main`分支
3. GitHub Actions会自动构建文档并部署到`gh-pages`分支

## 文档结构

生成的API文档具有以下结构：

- **首页**：按类别列出所有函数
  - 文本处理函数
  - 序列分析函数
  - 统计分析函数
  - 时间序列函数
  - 其他函数
  
- **函数详情页**：每个函数的详细信息
  - 函数描述
  - 参数详情
  - 返回值说明
  - 使用示例（带输入和实际输出）

## 自定义文档

### 添加更多函数示例

如需为特定函数添加更多示例，可以修改`docs_generator.py`文件中的`generate_examples_for_func`函数：

```python
def generate_examples_for_func(func_name: str) -> List[Dict[str, Any]]:
    # 现有代码...
    
    elif func_name == "你的函数名":
        examples_args = [
            (参数1, 参数2, ...),  # 示例1
            (参数1, 参数2, ...),  # 示例2
        ]
    
    # 现有代码...
```

### 修改文档样式

可以通过修改`copy_static_files`函数中的CSS内容来自定义文档样式。

### 更改文档组织结构

如需更改文档的组织方式，可以修改`generate_html_docs`函数中的分类逻辑和模板文件。

## 故障排除

### 常见问题

**问题**：生成文档时出现导入错误
**解决方法**：确保已正确安装`rust_pyfunc`包，并且Python路径正确

**问题**：某些函数没有示例
**解决方法**：在`generate_examples_for_func`函数中为该函数添加示例参数

**问题**：GitHub Actions部署失败
**解决方法**：检查工作流文件，确保设置了正确的权限

## 贡献

欢迎提交问题和改进建议！如果您有任何意见或建议，请提交Issue或Pull Request。

## 许可

与主项目相同的许可证 