import os
import sys
import inspect
import importlib
import re
import json
from typing import Dict, List, Any, Tuple, Optional
import markdown
import jinja2
import numpy as np
from pathlib import Path

# 确保我们可以导入rust_pyfunc模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入rust_pyfunc模块
import rust_pyfunc

def get_docstring(func) -> str:
    """获取函数的文档字符串"""
    doc = inspect.getdoc(func)
    return doc if doc else "无文档"

def parse_docstring(docstring: str) -> Dict[str, str]:
    """解析函数文档字符串，提取参数和返回值信息"""
    result = {
        "description": "",
        "parameters": [],
        "returns": ""
    }
    
    # 分割文档字符串的主要部分
    parts = re.split(r'参数说明：|返回值：', docstring)
    
    if len(parts) >= 1:
        result["description"] = parts[0].strip()
    
    if len(parts) >= 2:
        # 解析参数
        param_section = parts[1].strip()
        param_blocks = re.findall(r'(\w+)\s*:\s*([^\n]+)(?:\n\s+([^-][^\n]+))?', param_section)
        for name, type_info, description in param_blocks:
            result["parameters"].append({
                "name": name,
                "type": type_info.strip(),
                "description": description.strip() if description else ""
            })
    
    if len(parts) >= 3:
        # 解析返回值
        result["returns"] = parts[2].strip()
    
    return result

def get_function_info(func) -> Dict[str, Any]:
    """获取函数的信息，包括签名、文档等"""
    info = {}
    info["name"] = func.__name__
    
    # 获取函数签名
    try:
        signature = inspect.signature(func)
        params = []
        for name, param in signature.parameters.items():
            param_info = {
                "name": name,
                "default": str(param.default) if param.default is not inspect.Parameter.empty else None,
                "kind": str(param.kind),
                "annotation": str(param.annotation) if param.annotation is not inspect.Parameter.empty else None
            }
            params.append(param_info)
        info["signature"] = {
            "parameters": params,
            "return_annotation": str(signature.return_annotation) if signature.return_annotation is not inspect.Parameter.empty else None
        }
    except (ValueError, TypeError):
        info["signature"] = {"parameters": [], "return_annotation": None}
    
    # 获取文档
    docstring = get_docstring(func)
    info["docstring"] = docstring
    info["parsed_doc"] = parse_docstring(docstring)
    
    return info

def run_example(func_name: str, args_list: List[Tuple]) -> List[Dict[str, Any]]:
    """运行函数示例并获取结果"""
    examples = []
    func = getattr(rust_pyfunc, func_name)
    
    for args in args_list:
        example = {
            "args": args,
            "result": None,
            "error": None
        }
        
        try:
            # 运行函数并获取结果
            result = func(*args)
            
            # 对numpy数组特殊处理
            if isinstance(result, np.ndarray):
                if result.size > 10:
                    # 对于大数组，只显示前几个元素
                    example["result"] = f"数组，形状={result.shape}, 类型={result.dtype}\n前几个元素: {result.flatten()[:5]}..."
                else:
                    example["result"] = str(result)
            else:
                example["result"] = repr(result)
        except Exception as e:
            example["error"] = str(e)
        
        examples.append(example)
    
    return examples

def generate_examples_for_func(func_name: str) -> List[Dict[str, Any]]:
    """为函数生成示例"""
    # 根据函数名选择合适的示例参数
    examples_args = []
    
    if func_name == "trend":
        examples_args = [
            ([1.0, 2.0, 3.0, 4.0, 5.0],),  # 完美上升趋势
            ([5.0, 4.0, 3.0, 2.0, 1.0],),  # 完美下降趋势
            ([1.0, 3.0, 2.0, 5.0, 4.0],),  # 混合趋势
        ]
    elif func_name == "trend_fast":
        examples_args = [
            (np.array([1.0, 2.0, 3.0, 4.0, 5.0]),),
            (np.array([5.0, 4.0, 3.0, 2.0, 1.0]),),
            (np.array([1.0, 3.0, 2.0, 5.0, 4.0]),),
        ]
    elif func_name == "identify_segments":
        examples_args = [
            (np.array([1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0]),),
            (np.array([5.0, 5.0, 5.0, 5.0, 5.0]),),
        ]
    elif func_name == "find_max_range_product":
        examples_args = [
            (np.array([3.0, 1.0, 6.0, 4.0, 2.0, 8.0]),),
            (np.array([10.0, 8.0, 6.0, 4.0, 2.0]),),
        ]
    elif func_name == "vectorize_sentences":
        examples_args = [
            ("这是第一个句子", "这是第二个句子"),
            ("机器学习很有趣", "深度学习也很有趣"),
        ]
    elif func_name == "jaccard_similarity":
        examples_args = [
            ("机器学习算法", "深度学习算法"),
            ("Python编程", "Python编程语言"),
            ("完全不同的句子", "毫无共同点的文本"),
        ]
    elif func_name == "min_word_edit_distance":
        examples_args = [
            ("这是一个测试句子", "这是另一个测试句子"),
            ("深度学习算法", "机器学习算法"),
        ]
    elif func_name == "dtw_distance":
        examples_args = [
            ([1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 2.5, 3.0, 4.0]),
            ([1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]),
            ([1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0], 1),
        ]
    elif func_name == "transfer_entropy":
        examples_args = [
            ([1.0, 2.0, 3.0, 4.0, 5.0], [1.5, 2.5, 3.5, 4.5, 5.5], 2, 3),
            ([1.0, 2.0, 3.0, 4.0, 5.0], [5.0, 4.0, 3.0, 2.0, 1.0], 2, 3),
        ]
    elif func_name == "ols":
        examples_args = [
            (np.array([[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]]), np.array([2.0, 3.0, 4.0])),
            (np.array([[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]]), np.array([2.0, 3.0, 4.0]), False),
        ]
    elif func_name == "ols_predict":
        examples_args = [
            (np.array([[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]]), np.array([2.0, 3.0, 4.0]), np.array([[1.0, 4.0], [1.0, 5.0]])),
        ]
    elif func_name == "ols_residuals":
        examples_args = [
            (np.array([[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]]), np.array([2.0, 3.0, 4.0])),
        ]
    elif func_name == "min_range_loop":
        examples_args = [
            (np.array([3.0, 1.0, 4.0, 2.0, 5.0]), 3),
        ]
    elif func_name == "max_range_loop":
        examples_args = [
            (np.array([3.0, 1.0, 4.0, 2.0, 5.0]), 3),
        ]
    elif func_name == "rolling_volatility":
        examples_args = [
            (np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]), 3),
        ]
    elif func_name == "rolling_cv":
        examples_args = [
            (np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]), 3),
        ]
    elif func_name == "rolling_qcv":
        examples_args = [
            (np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]), 3),
        ]
    elif func_name == "vectorize_sentences_list":
        examples_args = [
            (["这是第一个句子", "这是第二个句子", "这是第三个句子"],),
        ]
    # 继续为其他函数添加示例...
    else:
        # 对于没有特定示例的函数，返回空列表
        return []
    
    return run_example(func_name, examples_args)

def get_all_functions() -> List[Dict[str, Any]]:
    """获取rust_pyfunc模块中的所有函数信息"""
    functions = []
    
    # 获取模块中的所有属性
    module_attrs = dir(rust_pyfunc)
    
    # 过滤出函数
    for attr_name in module_attrs:
        attr = getattr(rust_pyfunc, attr_name)
        
        # 检查是否为函数或方法
        if callable(attr) and not attr_name.startswith("_"):
            try:
                func_info = get_function_info(attr)
                
                # 生成函数示例
                func_info["examples"] = generate_examples_for_func(attr_name)
                
                functions.append(func_info)
            except Exception as e:
                print(f"处理函数 {attr_name} 时出错: {e}")
    
    # 按名称排序
    functions.sort(key=lambda x: x["name"])
    
    return functions

def generate_html_docs(functions: List[Dict[str, Any]], output_dir: str):
    """生成HTML文档"""
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载模板
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader("templates"),
        autoescape=jinja2.select_autoescape(['html'])
    )
    
    # 分类函数
    categorized_functions = {
        "text": [],
        "sequence": [],
        "statistics": [],
        "time_series": [],
        "other": []
    }
    
    for func in functions:
        name = func["name"]
        if any(name.startswith(prefix) for prefix in ["vectorize_", "jaccard_", "min_word_"]):
            categorized_functions["text"].append(func)
        elif any(name.startswith(prefix) for prefix in ["identify_", "find_", "compute_"]):
            categorized_functions["sequence"].append(func)
        elif any(name in name for name in ["ols", "rolling_", "volatility", "cv"]):
            categorized_functions["statistics"].append(func)
        elif any(name in name for name in ["trend", "dtw", "peaks", "entropy"]):
            categorized_functions["time_series"].append(func)
        else:
            categorized_functions["other"].append(func)
    
    # 渲染索引页面
    index_template = env.get_template("index.html")
    index_html = index_template.render(
        functions=functions,
        categorized_functions=categorized_functions
    )
    
    with open(os.path.join(output_dir, "index.html"), "w", encoding="utf-8") as f:
        f.write(index_html)
    
    # 为每个函数生成单独的页面
    function_template = env.get_template("function.html")
    for func in functions:
        func_html = function_template.render(function=func)
        
        with open(os.path.join(output_dir, f"{func['name']}.html"), "w", encoding="utf-8") as f:
            f.write(func_html)
    
    # 复制CSS和JS文件
    copy_static_files(output_dir)

def copy_static_files(output_dir: str):
    """复制静态文件到输出目录"""
    static_dir = os.path.join(output_dir, "static")
    os.makedirs(static_dir, exist_ok=True)
    
    # 创建CSS文件
    css_content = """
    body {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
        line-height: 1.6;
        color: #333;
        max-width: 1200px;
        margin: 0 auto;
        padding: 20px;
    }
    
    h1, h2, h3, h4 {
        color: #0366d6;
    }
    
    .container {
        display: flex;
        flex-wrap: wrap;
    }
    
    .sidebar {
        width: 250px;
        padding-right: 20px;
    }
    
    .content {
        flex: 1;
        min-width: 300px;
    }
    
    .function-list {
        list-style-type: none;
        padding: 0;
    }
    
    .function-list li {
        margin-bottom: 8px;
    }
    
    .function-list a {
        text-decoration: none;
        color: #0366d6;
    }
    
    .function-list a:hover {
        text-decoration: underline;
    }
    
    .function-item {
        border: 1px solid #e1e4e8;
        border-radius: 6px;
        padding: 20px;
        margin-bottom: 20px;
    }
    
    .function-name {
        margin-top: 0;
    }
    
    .parameter {
        margin-bottom: 10px;
    }
    
    .parameter-name {
        font-weight: bold;
    }
    
    .parameter-type {
        color: #6a737d;
    }
    
    .example {
        background-color: #f6f8fa;
        padding: 10px;
        border-radius: 3px;
        margin-bottom: 10px;
        font-family: SFMono-Regular, Consolas, "Liberation Mono", Menlo, monospace;
        font-size: 14px;
    }
    
    .category {
        margin-bottom: 30px;
    }
    
    .category-title {
        border-bottom: 1px solid #e1e4e8;
        padding-bottom: 10px;
    }
    
    code {
        font-family: SFMono-Regular, Consolas, "Liberation Mono", Menlo, monospace;
        background-color: #f6f8fa;
        padding: 2px 4px;
        border-radius: 3px;
    }
    
    .navbar {
        background-color: #24292e;
        padding: 15px 20px;
        margin-bottom: 20px;
        border-radius: 6px;
    }
    
    .navbar-title {
        color: white;
        font-size: 24px;
        margin: 0;
    }
    
    .navbar-subtitle {
        color: #c8c9cb;
        margin: 0;
    }
    """
    
    with open(os.path.join(static_dir, "style.css"), "w", encoding="utf-8") as f:
        f.write(css_content)

def create_templates():
    """创建HTML模板文件"""
    templates_dir = "templates"
    os.makedirs(templates_dir, exist_ok=True)
    
    # 创建基础模板
    base_template = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{% block title %}Rust PyFunc API文档{% endblock %}</title>
    <link rel="stylesheet" href="static/style.css">
</head>
<body>
    <div class="navbar">
        <h1 class="navbar-title">Rust PyFunc</h1>
        <p class="navbar-subtitle">高性能Python函数集合 - API文档</p>
    </div>
    
    {% block content %}{% endblock %}
</body>
</html>
"""
    
    # 创建索引页模板
    index_template = """{% extends "base.html" %}

{% block content %}
<div class="container">
    <div class="sidebar">
        <h2>函数分类</h2>
        <ul class="function-list">
            <li><a href="#text">文本处理</a></li>
            <li><a href="#sequence">序列分析</a></li>
            <li><a href="#statistics">统计分析</a></li>
            <li><a href="#time_series">时间序列</a></li>
            <li><a href="#other">其他函数</a></li>
        </ul>
    </div>
    
    <div class="content">
        <h1>API 文档</h1>
        <p>本文档提供了Rust PyFunc库中所有公开函数的详细说明和使用示例。这些示例基于真实的Python运行结果生成。</p>
        
        <div id="text" class="category">
            <h2 class="category-title">文本处理函数</h2>
            {% for func in categorized_functions.text %}
            <div class="function-item">
                <h3 class="function-name"><a href="{{ func.name }}.html">{{ func.name }}</a></h3>
                <p>{{ func.parsed_doc.description }}</p>
                <a href="{{ func.name }}.html">查看详情</a>
            </div>
            {% endfor %}
        </div>
        
        <div id="sequence" class="category">
            <h2 class="category-title">序列分析函数</h2>
            {% for func in categorized_functions.sequence %}
            <div class="function-item">
                <h3 class="function-name"><a href="{{ func.name }}.html">{{ func.name }}</a></h3>
                <p>{{ func.parsed_doc.description }}</p>
                <a href="{{ func.name }}.html">查看详情</a>
            </div>
            {% endfor %}
        </div>
        
        <div id="statistics" class="category">
            <h2 class="category-title">统计分析函数</h2>
            {% for func in categorized_functions.statistics %}
            <div class="function-item">
                <h3 class="function-name"><a href="{{ func.name }}.html">{{ func.name }}</a></h3>
                <p>{{ func.parsed_doc.description }}</p>
                <a href="{{ func.name }}.html">查看详情</a>
            </div>
            {% endfor %}
        </div>
        
        <div id="time_series" class="category">
            <h2 class="category-title">时间序列函数</h2>
            {% for func in categorized_functions.time_series %}
            <div class="function-item">
                <h3 class="function-name"><a href="{{ func.name }}.html">{{ func.name }}</a></h3>
                <p>{{ func.parsed_doc.description }}</p>
                <a href="{{ func.name }}.html">查看详情</a>
            </div>
            {% endfor %}
        </div>
        
        <div id="other" class="category">
            <h2 class="category-title">其他函数</h2>
            {% for func in categorized_functions.other %}
            <div class="function-item">
                <h3 class="function-name"><a href="{{ func.name }}.html">{{ func.name }}</a></h3>
                <p>{{ func.parsed_doc.description }}</p>
                <a href="{{ func.name }}.html">查看详情</a>
            </div>
            {% endfor %}
        </div>
    </div>
</div>
{% endblock %}
"""
    
    # 创建函数详情页模板
    function_template = """{% extends "base.html" %}

{% block title %}{{ function.name }} - Rust PyFunc API文档{% endblock %}

{% block content %}
<div class="container">
    <div class="sidebar">
        <h3>导航</h3>
        <p><a href="index.html">返回首页</a></p>
    </div>
    
    <div class="content">
        <h1>{{ function.name }}</h1>
        
        <div class="function-description">
            <h2>描述</h2>
            <p>{{ function.parsed_doc.description }}</p>
        </div>
        
        <div class="function-signature">
            <h2>函数签名</h2>
            <code>{{ function.name }}(
            {%- for param in function.signature.parameters -%}
                {{ param.name }}{% if not loop.last %}, {% endif %}
            {%- endfor -%}
            ) -> {{ function.signature.return_annotation }}</code>
        </div>
        
        <div class="function-parameters">
            <h2>参数</h2>
            {% if function.parsed_doc.parameters %}
                {% for param in function.parsed_doc.parameters %}
                <div class="parameter">
                    <span class="parameter-name">{{ param.name }}</span>
                    <span class="parameter-type">({{ param.type }})</span>
                    <p>{{ param.description }}</p>
                </div>
                {% endfor %}
            {% else %}
                <p>此函数没有参数</p>
            {% endif %}
        </div>
        
        <div class="function-returns">
            <h2>返回值</h2>
            <p>{{ function.parsed_doc.returns }}</p>
        </div>
        
        <div class="function-examples">
            <h2>示例</h2>
            {% if function.examples %}
                {% for example in function.examples %}
                <div class="example">
                    <p><strong>输入:</strong></p>
                    <code>{{ function.name }}(
                    {%- for arg in example.args -%}
                        {{ arg }}{% if not loop.last %}, {% endif %}
                    {%- endfor -%}
                    )</code>
                    
                    {% if example.error %}
                    <p><strong>错误:</strong></p>
                    <code>{{ example.error }}</code>
                    {% else %}
                    <p><strong>输出:</strong></p>
                    <code>{{ example.result }}</code>
                    {% endif %}
                </div>
                {% endfor %}
            {% else %}
                <p>暂无示例</p>
            {% endif %}
        </div>
    </div>
</div>
{% endblock %}
"""
    
    # 写入模板文件
    with open(os.path.join(templates_dir, "base.html"), "w", encoding="utf-8") as f:
        f.write(base_template)
    
    with open(os.path.join(templates_dir, "index.html"), "w", encoding="utf-8") as f:
        f.write(index_template)
    
    with open(os.path.join(templates_dir, "function.html"), "w", encoding="utf-8") as f:
        f.write(function_template)

def create_github_workflow():
    """创建GitHub Actions工作流文件，用于自动部署到GitHub Pages"""
    github_dir = ".github/workflows"
    os.makedirs(github_dir, exist_ok=True)
    
    workflow_content = """name: Deploy API Docs

on:
  push:
    branches: [ main ]
  workflow_dispatch:

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.9'
      
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install maturin jinja2 markdown numpy
          pip install -e .
      
      - name: Generate documentation
        run: |
          python docs_generator.py
      
      - name: Deploy to GitHub Pages
        uses: JamesIves/github-pages-deploy-action@4.1.4
        with:
          branch: gh-pages
          folder: docs
"""
    
    with open(os.path.join(github_dir, "deploy.yml"), "w", encoding="utf-8") as f:
        f.write(workflow_content)

def main():
    """主函数"""
    print("开始生成API文档...")
    
    # 创建HTML模板
    create_templates()
    
    # 获取所有函数信息
    functions = get_all_functions()
    
    # 生成文档
    generate_html_docs(functions, "docs")
    
    # 创建GitHub Actions工作流
    create_github_workflow()
    
    print(f"文档生成完成，共包含 {len(functions)} 个函数。")
    print("文档已输出到 docs 目录。")
    print("要查看文档，请打开 docs/index.html 文件。")
    print("要部署到GitHub Pages，请提交更改并推送到GitHub仓库。")

if __name__ == "__main__":
    main() 