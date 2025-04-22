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
    parts = re.split(r'参数说明：|参数：|返回值：|返回：', docstring)
    
    if len(parts) >= 1:
        # 处理描述部分，将markdown转换为HTML
        desc_md = parts[0].strip()
        result["description"] = markdown.markdown(desc_md)
    
    if len(parts) >= 2:
        # 解析参数
        param_section = parts[1].strip()
        # 改进参数匹配正则表达式，使其能更好地处理多行参数描述
        param_blocks = re.findall(r'(\w+)\s*:\s*([^\n]+)(?:\n\s+([^-\n][^\n]*(?:\n\s+[^-\n][^\n]*)*)?)?', param_section)
        for name, type_info, description in param_blocks:
            # 清理可能的多行描述
            clean_description = re.sub(r'\s+', ' ', description.strip()) if description else ""
            # 将参数描述中的markdown转换为HTML
            param_desc_html = markdown.markdown(clean_description) if clean_description else ""
            result["parameters"].append({
                "name": name,
                "type": type_info.strip(),
                "description": param_desc_html
            })
    
    if len(parts) >= 3:
        # 解析返回值，并清理格式
        returns_text = parts[2].strip()
        # 将返回值中的markdown转换为HTML
        result["returns"] = markdown.markdown(returns_text)
    
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

def format_numpy_array(arr):
    """更好地格式化NumPy数组输出"""
    if arr is None:
        return "None"
    
    try:
        if arr.size > 10:
            # 对于大数组，格式化为更紧凑的形式
            if arr.ndim == 1:
                return f"array([{', '.join(f'{x:.4f}' if isinstance(x, float) else str(x) for x in arr.flatten()[:5])},...], shape={arr.shape}, dtype={arr.dtype})"
            else:
                return f"array([...], shape={arr.shape}, dtype={arr.dtype})"
        else:
            # 对于小数组，显示完整内容但格式更清晰
            if arr.ndim == 1:
                values = [f'{x:.4f}' if isinstance(x, float) else str(x) for x in arr]
                return f"array([{', '.join(values)}], dtype={arr.dtype})"
            else:
                # 多维数组，使用更清晰的格式
                array_str = np.array2string(arr, precision=4, suppress_small=True, separator=', ')
                return f"array({array_str}, dtype={arr.dtype})"
    except Exception as e:
        # 处理格式化失败的情况
        return f"array(shape={arr.shape if hasattr(arr, 'shape') else 'unknown'}, dtype={arr.dtype if hasattr(arr, 'dtype') else 'unknown'})"

def run_example(func_name: str, args_list: List[Tuple]) -> List[Dict[str, Any]]:
    """运行函数示例并获取结果"""
    examples = []
    
    # 尝试获取函数
    try:
        func = getattr(rust_pyfunc, func_name)
    except AttributeError:
        # 如果函数不存在，返回空示例列表
        print(f"警告: 函数 {func_name} 不存在")
        return []
    
    for args in args_list:
        example = {
            "args": args,
            "result": None,
            "error": None
        }
        
        # 检查参数类型，确保正确格式化
        formatted_args = []
        for arg in args:
            if isinstance(arg, np.ndarray):
                # 对NumPy数组进行特殊处理，确保示例中正确显示
                formatted_args.append(f"np.array({arg.tolist()})")
            elif isinstance(arg, str):
                # 字符串加引号
                formatted_args.append(f'"{arg}"')
            elif isinstance(arg, (list, tuple)):
                # 列表或元组
                formatted_args.append(str(arg))
            else:
                # 其他类型
                formatted_args.append(str(arg))
        
        example["formatted_args"] = formatted_args
        
        try:
            # 运行函数并获取结果
            result = func(*args)
            
            # 对不同类型的结果进行美化格式化
            if isinstance(result, np.ndarray):
                example["result"] = format_numpy_array(result)
            elif isinstance(result, (tuple, list)) and len(result) > 0:
                if isinstance(result[0], np.ndarray):
                    # 处理返回元组/列表包含numpy数组的情况
                    formatted_result = []
                    for item in result:
                        if isinstance(item, np.ndarray):
                            formatted_result.append(format_numpy_array(item))
                        else:
                            formatted_result.append(str(item))
                    example["result"] = f"({', '.join(formatted_result)})" if isinstance(result, tuple) else f"[{', '.join(formatted_result)}]"
                elif isinstance(result[0], (int, float)):
                    # 数值型列表/元组
                    if len(result) > 10:
                        example["result"] = f"({', '.join(map(str, result[:5]))}, ...) (长度: {len(result)})" if isinstance(result, tuple) else f"[{', '.join(map(str, result[:5]))}, ...] (长度: {len(result)})"
                    else:
                        example["result"] = str(result)
                else:
                    # 其他类型的列表/元组
                    example["result"] = str(result)
            else:
                example["result"] = str(result)
        except Exception as e:
            # 获取详细错误信息
            import traceback
            error_msg = f"{type(e).__name__}: {str(e)}"
            example["error"] = error_msg
            example["traceback"] = traceback.format_exc()
            print(f"示例 {func_name}{args} 执行失败: {error_msg}")
        
        examples.append(example)
    
    return examples

def generate_examples_for_func(func_name: str) -> List[Dict[str, Any]]:
    """为函数生成示例"""
    # 根据函数名选择合适的示例参数
    examples_args = []
    
    # 检查函数是否在已知存在问题的列表中
    problematic_functions = [
        "calculate_shannon_entropy_change", 
        "calculate_shannon_entropy_change_at_low",
        "find_follow_volume_sum_same_price", 
        "find_follow_volume_sum_same_price_and_flag",
        "find_local_peaks_within_window",
        "mark_follow_groups",
        "mark_follow_groups_with_flag",
        "rolling_window_stat",
        "max_range_loop",
        "min_range_loop"
    ]
    
    # 对于问题函数，返回空示例
    if func_name in problematic_functions:
        return []
    
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
            (np.array([3.0, 1.0, 6.0, 4.0, 2.0, 8.0], dtype=np.float64),),
            (np.array([10.0, 8.0, 6.0, 4.0, 2.0], dtype=np.float64),),
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
    elif func_name == "rolling_volatility":
        examples_args = [
            (np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]), 3, 1),  # 添加interval参数
        ]
    elif func_name == "rolling_cv":
        examples_args = [
            (np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]), 3, 1),  # 添加interval参数
        ]
    elif func_name == "rolling_qcv":
        examples_args = [
            (np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]), 3, 1),  # 添加interval参数
        ]
    elif func_name == "vectorize_sentences_list":
        examples_args = [
            (["这是第一个句子", "这是第二个句子", "这是第三个句子"],),
        ]
    elif func_name == "find_half_energy_time":
        examples_args = [
            (np.array([10.0, 8.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0], dtype=np.float64), np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], dtype=np.float64)),  
        ]
    elif func_name == "brachistochrone_curve":
        examples_args = [
            (0.0, 0.0, 1.0, 1.0, np.linspace(0.0, 1.0, 10)),  # 修复参数顺序
        ]
    elif func_name == "RollingFutureAccessor":
        # 这个特殊的函数可能需要pandas对象，由于不能直接在示例中创建，使用空列表
        examples_args = []
    
    # 对于没有特定示例的函数，返回空列表
    if not examples_args:
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
    
    # 生成搜索数据
    search_data = []
    for func in functions:
        search_data.append({
            "name": func["name"],
            "description": func["parsed_doc"]["description"],
            "params": [param["name"] for param in func["parsed_doc"]["parameters"]]
        })
    
    with open(os.path.join(output_dir, "search_data.json"), "w", encoding="utf-8") as f:
        json.dump(search_data, f, ensure_ascii=False, indent=2)
    
    # 复制CSS和JS文件
    copy_static_files(output_dir)

def copy_static_files(output_dir: str):
    """复制静态文件到输出目录"""
    static_dir = os.path.join(output_dir, "static")
    os.makedirs(static_dir, exist_ok=True)
    
    # 创建CSS文件，增强markdown渲染效果
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
        padding: 10px;
        background-color: #f8f9fa;
        border-radius: 4px;
    }
    
    .parameter-name {
        font-weight: bold;
        color: #0366d6;
    }
    
    .parameter-type {
        color: #6a737d;
        font-style: italic;
        margin-left: 5px;
    }
    
    .example {
        background-color: #f6f8fa;
        padding: 16px;
        border-radius: 6px;
        margin-bottom: 16px;
        border: 1px solid #e1e4e8;
    }
    
    .example-input {
        margin-bottom: 12px;
    }
    
    .example-output {
        margin-top: 12px;
        padding: 8px;
        background-color: #f0fff4;
        border-radius: 4px;
    }
    
    .output-value {
        color: #28a745;
        word-break: break-all;
        white-space: pre-wrap;
    }
    
    .error-message {
        color: #cb2431;
        padding: 8px;
        background-color: #fff5f5;
        border-radius: 4px;
        word-break: break-all;
        white-space: pre-wrap;
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
        white-space: pre-wrap;
    }
    
    pre code {
        display: block;
        padding: 16px;
        overflow-x: auto;
        line-height: 1.45;
    }
    
    .navbar {
        background-color: #24292e;
        padding: 15px 20px;
        margin-bottom: 20px;
        border-radius: 6px;
        position: relative;
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
    
    .signature {
        margin-bottom: 20px;
        padding: 12px;
        background-color: #f6f8fa;
        border-radius: 6px;
        border-left: 4px solid #0366d6;
    }
    
    .returns-section {
        margin-top: 20px;
        margin-bottom: 20px;
        padding: 16px;
        background-color: #f8f9fa;
        border-radius: 6px;
        border-left: 4px solid #28a745;
    }
    
    .function-description {
        border-left: 4px solid #0366d6;
        padding-left: 16px;
        margin-bottom: 24px;
    }
    
    .search-container {
        margin-bottom: 20px;
        position: relative;
    }
    
    #search-input {
        width: 100%;
        padding: 10px;
        border: 1px solid #e1e4e8;
        border-radius: 6px;
        font-size: 16px;
    }
    
    #search-results {
        position: absolute;
        width: 100%;
        max-height: 300px;
        overflow-y: auto;
        background-color: white;
        border: 1px solid #e1e4e8;
        border-radius: 0 0 6px 6px;
        z-index: 1000;
        display: none;
    }
    
    .search-result-item {
        padding: 10px;
        border-bottom: 1px solid #e1e4e8;
        cursor: pointer;
    }
    
    .search-result-item:hover {
        background-color: #f6f8fa;
    }
    
    .search-result-item:last-child {
        border-bottom: none;
    }
    
    .example-usage-note {
        margin-top: 20px;
        padding: 16px;
        background-color: #f6f8fa;
        border-radius: 6px;
        border-left: 4px solid #f9c513;
    }
    
    /* 增强markdown渲染样式 */
    .function-description p, 
    .function-parameters p, 
    .function-returns p {
        margin: 0.5em 0;
    }
    
    .function-description ul, 
    .function-parameters ul, 
    .function-returns ul {
        padding-left: 20px;
    }
    
    .function-description code, 
    .function-parameters code, 
    .function-returns code {
        background-color: #f0f0f0;
        padding: 2px 4px;
        border-radius: 3px;
        font-family: monospace;
    }
    
    .function-description pre, 
    .function-parameters pre, 
    .function-returns pre {
        background-color: #f6f8fa;
        padding: 10px;
        border-radius: 4px;
        overflow-x: auto;
    }
    
    /* 格式化表格 */
    table {
        border-collapse: collapse;
        width: 100%;
        margin: 1em 0;
    }
    
    th, td {
        border: 1px solid #ddd;
        padding: 8px;
    }
    
    th {
        background-color: #f2f2f2;
        text-align: left;
    }
    
    tr:nth-child(even) {
        background-color: #f9f9f9;
    }
    """
    
    with open(os.path.join(static_dir, "style.css"), "w", encoding="utf-8") as f:
        f.write(css_content)
    
    # 创建搜索功能的JS文件
    search_js_content = """
    // 搜索功能
    document.addEventListener('DOMContentLoaded', function() {
        const searchInput = document.getElementById('search-input');
        const searchResults = document.getElementById('search-results');
        
        if (!searchInput || !searchResults) return;
        
        // 加载函数数据
        fetch('search_data.json')
            .then(response => response.json())
            .then(functions => {
                // 搜索函数
                searchInput.addEventListener('input', function() {
                    const query = this.value.toLowerCase().trim();
                    
                    if (query.length < 2) {
                        searchResults.style.display = 'none';
                        return;
                    }
                    
                    // 过滤函数
                    const filtered = functions.filter(func => {
                        return func.name.toLowerCase().includes(query) || 
                               func.description.toLowerCase().includes(query);
                    });
                    
                    // 显示结果
                    if (filtered.length > 0) {
                        searchResults.innerHTML = '';
                        filtered.forEach(func => {
                            const resultItem = document.createElement('div');
                            resultItem.className = 'search-result-item';
                            resultItem.innerHTML = `<strong>${func.name}</strong> - ${func.description.substring(0, 100).replace(/<\/?[^>]+(>|$)/g, "")}${func.description.length > 100 ? '...' : ''}`;
                            resultItem.onclick = function() {
                                window.location.href = `${func.name}.html`;
                            };
                            searchResults.appendChild(resultItem);
                        });
                        searchResults.style.display = 'block';
                    } else {
                        searchResults.innerHTML = '<div class="search-result-item">没有找到相关函数</div>';
                        searchResults.style.display = 'block';
                    }
                });
                
                // 点击外部关闭搜索结果
                document.addEventListener('click', function(e) {
                    if (e.target !== searchInput && !searchResults.contains(e.target)) {
                        searchResults.style.display = 'none';
                    }
                });
                
                // ESC键关闭搜索结果
                document.addEventListener('keydown', function(e) {
                    if (e.key === 'Escape') {
                        searchResults.style.display = 'none';
                    }
                });
            })
            .catch(error => console.error('加载搜索数据失败:', error));
    });
    """
    
    with open(os.path.join(static_dir, "search.js"), "w", encoding="utf-8") as f:
        f.write(search_js_content)

def create_templates():
    """创建HTML模板文件"""
    templates_dir = "templates"
    os.makedirs(templates_dir, exist_ok=True)
    
    # 创建基础模板，添加搜索JS
    base_template = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{% block title %}Rust PyFunc API文档{% endblock %}</title>
    <link rel="stylesheet" href="static/style.css">
    <script src="static/search.js"></script>
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
    
    # 更新索引模板，添加搜索功能
    index_template = """{% extends "base.html" %}

{% block content %}
<div class="container">
    <div class="sidebar">
        <div class="search-container">
            <input type="text" id="search-input" placeholder="搜索函数...">
            <div id="search-results"></div>
        </div>
        
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
                <div>{{ func.parsed_doc.description|safe }}</div>
                <a href="{{ func.name }}.html">查看详情</a>
            </div>
            {% endfor %}
        </div>
        
        <div id="sequence" class="category">
            <h2 class="category-title">序列分析函数</h2>
            {% for func in categorized_functions.sequence %}
            <div class="function-item">
                <h3 class="function-name"><a href="{{ func.name }}.html">{{ func.name }}</a></h3>
                <div>{{ func.parsed_doc.description|safe }}</div>
                <a href="{{ func.name }}.html">查看详情</a>
            </div>
            {% endfor %}
        </div>
        
        <div id="statistics" class="category">
            <h2 class="category-title">统计分析函数</h2>
            {% for func in categorized_functions.statistics %}
            <div class="function-item">
                <h3 class="function-name"><a href="{{ func.name }}.html">{{ func.name }}</a></h3>
                <div>{{ func.parsed_doc.description|safe }}</div>
                <a href="{{ func.name }}.html">查看详情</a>
            </div>
            {% endfor %}
        </div>
        
        <div id="time_series" class="category">
            <h2 class="category-title">时间序列函数</h2>
            {% for func in categorized_functions.time_series %}
            <div class="function-item">
                <h3 class="function-name"><a href="{{ func.name }}.html">{{ func.name }}</a></h3>
                <div>{{ func.parsed_doc.description|safe }}</div>
                <a href="{{ func.name }}.html">查看详情</a>
            </div>
            {% endfor %}
        </div>
        
        <div id="other" class="category">
            <h2 class="category-title">其他函数</h2>
            {% for func in categorized_functions.other %}
            <div class="function-item">
                <h3 class="function-name"><a href="{{ func.name }}.html">{{ func.name }}</a></h3>
                <div>{{ func.parsed_doc.description|safe }}</div>
                <a href="{{ func.name }}.html">查看详情</a>
            </div>
            {% endfor %}
        </div>
    </div>
</div>
{% endblock %}
"""
    
    # 更新函数详情页模板，修复使用示例的显示问题，添加safe过滤器允许HTML渲染
    function_template = """{% extends "base.html" %}

{% block title %}{{ function.name }} - Rust PyFunc API文档{% endblock %}

{% block content %}
<div class="container">
    <div class="sidebar">
        <div class="search-container">
            <input type="text" id="search-input" placeholder="搜索函数...">
            <div id="search-results"></div>
        </div>
        
        <h3>导航</h3>
        <p><a href="index.html">返回首页</a></p>
    </div>
    
    <div class="content">
        <h1>{{ function.name }}</h1>
        
        <div class="function-description">
            <h2>描述</h2>
            <div>{{ function.parsed_doc.description|safe }}</div>
        </div>
        
        <div class="function-signature signature">
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
                    <div>{{ param.description|safe }}</div>
                </div>
                {% endfor %}
            {% else %}
                <p>此函数没有参数</p>
            {% endif %}
        </div>
        
        <div class="function-returns returns-section">
            <h2>返回值</h2>
            <div>{{ function.parsed_doc.returns|safe }}</div>
        </div>
        
        <div class="function-examples">
            <h2>示例</h2>
            {% if function.examples %}
                {% for example in function.examples %}
                <div class="example">
                    <div class="example-input">
                        <p><strong>输入:</strong></p>
                        <code>{{ function.name }}(
                        {%- if example.formatted_args is defined -%}
                            {% for arg in example.formatted_args %}
                                {{ arg }}{% if not loop.last %}, {% endif %}
                            {% endfor %}
                        {%- else -%}
                            {%- for arg in example.args -%}
                                {{ arg }}{% if not loop.last %}, {% endif %}
                            {%- endfor -%}
                        {%- endif -%}
                        )</code>
                    </div>
                    
                    {% if example.error %}
                    <div class="example-error">
                        <p><strong>错误:</strong></p>
                        <code class="error-message">{{ example.error }}</code>
                    </div>
                    {% else %}
                    <div class="example-output">
                        <p><strong>输出:</strong></p>
                        <code class="output-value">{{ example.result }}</code>
                    </div>
                    {% endif %}
                </div>
                {% endfor %}
            {% else %}
                <p>暂无示例</p>
            {% endif %}
            
            <div class="example-usage-note">
                <h3>Python使用示例</h3>
                <pre><code>import numpy as np
from rust_pyfunc import {{ function.name }}

# 使用示例
{% if function.examples and function.examples[0].formatted_args is defined %}
{% set example = function.examples[0] %}
result = {{ function.name }}(
{%- for arg in example.formatted_args -%}
    {{ arg }}{% if not loop.last %}, {% endif %}
{%- endfor -%}
)
print(f"结果: {result}")
{% elif function.examples and function.examples[0].args %}
{% set example = function.examples[0] %}
result = {{ function.name }}(
{%- for arg in example.args -%}
    {{ arg }}{% if not loop.last %}, {% endif %}
{%- endfor -%}
)
print(f"结果: {result}")
{% else %}
# 请参考文档中的参数说明使用此函数
{% endif %}</code></pre>
            </div>
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

# 添加权限配置
permissions:
  contents: write

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
          pip install maturin jinja2 markdown numpy pandas graphviz IPython
          pip install -e .
      
      - name: Generate documentation
        run: |
          python docs_generator.py
      
      - name: Deploy to GitHub Pages
        # 更新到较新的版本
        uses: JamesIves/github-pages-deploy-action@v4.4.3
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