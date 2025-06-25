#!/usr/bin/env python3
"""
提取万里长征go函数进行测试
"""

import os
import sys
import tempfile

def extract_and_test_wanli_go():
    """提取万里长征go函数并测试"""
    print("=== 提取万里长征go函数 ===")
    
    # 读取万里长征脚本，但只提取go函数部分
    wanli_file = '/home/chenzongwei/pythoncode/万里长征/万里长征放宽参数20.py'
    
    with open(wanli_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 找到go函数的开始和结束
    go_start = content.find('def go(date,code):')
    if go_start == -1:
        print("❌ 找不到go函数定义")
        return False
    
    # 找到下一个顶级定义或文件结束来确定go函数的结束
    lines = content[go_start:].split('\n')
    go_lines = []
    indent_level = None
    
    for i, line in enumerate(lines):
        if i == 0:  # def go(date,code): 这一行
            go_lines.append(line)
            continue
            
        if line.strip() == '':  # 空行
            go_lines.append(line)
            continue
            
        # 检查缩进级别
        if line.strip():  # 非空行
            current_indent = len(line) - len(line.lstrip())
            
            if indent_level is None and current_indent > 0:
                indent_level = current_indent
                go_lines.append(line)
            elif indent_level is not None and current_indent >= indent_level:
                go_lines.append(line)
            elif indent_level is not None and current_indent < indent_level and line.strip() != '':
                # 到达函数结束
                break
            else:
                go_lines.append(line)
    
    go_function_code = '\n'.join(go_lines)
    print(f"✅ 提取了go函数，代码长度: {len(go_function_code)} 字符")
    print(f"前几行预览:")
    for i, line in enumerate(go_lines[:10]):
        print(f"  {i+1}: {line}")
    
    # 创建一个独立的模块来测试
    test_module_code = f'''
import design_whatever as dw

{go_function_code}
'''
    
    # 写入临时文件
    temp_module_path = '/tmp/extracted_wanli_go.py'
    with open(temp_module_path, 'w', encoding='utf-8') as f:
        f.write(test_module_code)
    
    print(f"✅ 创建临时模块: {temp_module_path}")
    
    # 现在测试提取的go函数
    sys.path.insert(0, '/tmp')
    
    try:
        print("尝试导入提取的go函数...")
        import extracted_wanli_go
        print("✅ 成功导入提取的go函数！")
        
        # 添加项目路径
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'python'))
        
        # 现在测试这个函数
        test_args = [
            [20170101, "000001"],
            [20170101, "000002"],
        ]
        
        with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as tmp_file:
            backup_file = tmp_file.name
        
        try:
            import rust_pyfunc
            
            print("开始测试提取的万里长征go函数...")
            results = rust_pyfunc.run_pools(
                extracted_wanli_go.go,
                test_args,
                backup_file=backup_file,
                num_threads=2
            )
            
            print(f"✅ 测试成功，结果数量: {len(results)}")
            
            # 检查结果
            for i, result in enumerate(results):
                print(f"结果{i}: 长度={len(result)}")
            
            return True
            
        except Exception as e:
            print(f"❌ 测试失败: {e}")
            print("现在可以看到万里长征go函数的具体错误了！")
            import traceback
            traceback.print_exc()
            return False
            
        finally:
            if os.path.exists(backup_file):
                os.unlink(backup_file)
        
    except Exception as e:
        print(f"❌ 导入提取的go函数失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # 清理临时文件
        if os.path.exists(temp_module_path):
            os.unlink(temp_module_path)

if __name__ == "__main__":
    print("开始提取和测试万里长征go函数...")
    success = extract_and_test_wanli_go()
    print(f"\n测试结果: {'✅ 成功' if success else '❌ 失败'}")