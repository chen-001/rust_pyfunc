#!/usr/bin/env python3
"""
调试万里长征导入问题
"""

import sys
import os

def debug_wanli_import():
    """调试万里长征导入问题"""
    print("=== 调试万里长征导入 ===")
    
    wanli_path = '/home/chenzongwei/pythoncode/万里长征'
    wanli_file = os.path.join(wanli_path, '万里长征放宽参数20.py')
    
    print(f"检查文件是否存在: {wanli_file}")
    if not os.path.exists(wanli_file):
        print("❌ 文件不存在")
        return False
    
    print("✅ 文件存在")
    
    # 读取文件内容，查看是否有模块级别的执行代码
    with open(wanli_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    print(f"文件大小: {len(content)} 字符")
    
    # 检查是否有模块级别的dw.run_factor调用
    if 'dw.run_factor(' in content:
        print("⚠️ 发现模块级别的dw.run_factor调用")
        
        # 找到这个调用的位置
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if 'dw.run_factor(' in line:
                print(f"第{i+1}行: {line.strip()}")
                
                # 检查这行是否在函数内部
                # 简单检查：看前面的行是否有函数定义
                in_function = False
                for j in range(max(0, i-20), i):
                    if lines[j].strip().startswith('def '):
                        # 检查缩进级别
                        def_indent = len(lines[j]) - len(lines[j].lstrip())
                        current_indent = len(line) - len(line.lstrip())
                        if current_indent > def_indent:
                            in_function = True
                            break
                
                if not in_function:
                    print("❌ 这是模块级别的调用！导入时就会执行")
                    return False
                else:
                    print("✅ 这是函数内部的调用")
    
    print("尝试添加路径并导入...")
    sys.path.insert(0, wanli_path)
    
    try:
        print("正在导入万里长征模块...")
        import 万里长征放宽参数20 as wanli_module
        print("✅ 成功导入万里长征模块")
        
        # 检查是否有go函数
        if hasattr(wanli_module, 'go'):
            print("✅ 找到go函数")
            return True
        else:
            print("❌ 没有找到go函数")
            return False
            
    except Exception as e:
        print(f"❌ 导入失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    debug_wanli_import()