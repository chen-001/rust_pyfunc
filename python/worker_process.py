import sys
import json
import base64
import traceback
import math
import hashlib
import gc
import weakref


CALCULATE_FUNCTION = None
FUNCTION_CODE_HASH = None  # 跟踪函数代码版本
IMPORTED_MODULES = set()   # 跟踪用户导入的模块
BASE_MODULES = set(sys.modules.keys())  # 记录基础系统模块

def clean_numeric_values(values):
    """清理数值列表中的NaN/Inf值，转换为特殊字符串以便JSON序列化"""
    cleaned = []
    for value in values:
        try:
            if isinstance(value, (int, float)):
                if math.isnan(value):
                    cleaned.append("__NaN__")  # 特殊标记表示NaN
                elif math.isinf(value):
                    cleaned.append("__NaN__")  # 无穷也转为NaN标记
                else:
                    cleaned.append(float(value))  # 确保是float类型
            else:
                # 尝试转换为数值
                num_value = float(value)
                if math.isnan(num_value):
                    cleaned.append("__NaN__")  # NaN标记
                elif math.isinf(num_value):
                    cleaned.append("__NaN__")  # 无穷转为NaN标记
                else:
                    cleaned.append(num_value)
        except (ValueError, TypeError):
            # 如果无法转换为数值，转换为NaN标记
            cleaned.append("__NaN__")
    return cleaned

def cleanup_imported_modules():
    """清理用户导入的模块，避免内存累积"""
    global IMPORTED_MODULES
    
    # 清理用户导入的模块（保留系统基础模块）
    modules_to_remove = []
    for module_name in sys.modules.keys():
        if module_name not in BASE_MODULES and module_name not in ('__main__', '__builtins__'):
            modules_to_remove.append(module_name)
            IMPORTED_MODULES.add(module_name)
    
    for module_name in modules_to_remove:
        try:
            del sys.modules[module_name]
        except KeyError:
            pass  # 模块已被删除
    
    # 强制垃圾收集
    gc.collect()

def get_restricted_globals():
    """获取受限的全局命名空间，包含必要的内建函数"""
    import builtins
    
    # 创建受限的全局命名空间，包含安全的内建函数和异常类
    restricted_globals = {
        '__builtins__': {
            # 基本数据类型
            'len': builtins.len,
            'range': builtins.range,
            'list': builtins.list,
            'dict': builtins.dict,
            'tuple': builtins.tuple,
            'set': builtins.set,
            'str': builtins.str,
            'int': builtins.int,
            'float': builtins.float,
            'bool': builtins.bool,
            'bytes': builtins.bytes,
            'bytearray': builtins.bytearray,
            'complex': builtins.complex,
            
            # 基本函数
            'max': builtins.max,
            'min': builtins.min,
            'sum': builtins.sum,
            'abs': builtins.abs,
            'round': builtins.round,
            'pow': builtins.pow,
            'divmod': builtins.divmod,
            'print': builtins.print,
            'enumerate': builtins.enumerate,
            'zip': builtins.zip,
            'sorted': builtins.sorted,
            'reversed': builtins.reversed,
            'isinstance': builtins.isinstance,
            'issubclass': builtins.issubclass,
            'hasattr': builtins.hasattr,
            'getattr': builtins.getattr,
            'setattr': builtins.setattr,
            'delattr': builtins.delattr,
            'callable': builtins.callable,
            'iter': builtins.iter,
            'next': builtins.next,
            'map': builtins.map,
            'filter': builtins.filter,
            'any': builtins.any,
            'all': builtins.all,
            'hash': builtins.hash,  # 添加hash函数
            
            # 异常类 - 这是关键的修复！
            'Exception': builtins.Exception,
            'BaseException': builtins.BaseException,
            'SystemExit': builtins.SystemExit,
            'KeyboardInterrupt': builtins.KeyboardInterrupt,
            'GeneratorExit': builtins.GeneratorExit,
            'StopIteration': builtins.StopIteration,
            'StopAsyncIteration': builtins.StopAsyncIteration,
            'ArithmeticError': builtins.ArithmeticError,
            'LookupError': builtins.LookupError,
            'AssertionError': builtins.AssertionError,
            'AttributeError': builtins.AttributeError,
            'BufferError': builtins.BufferError,
            'EOFError': builtins.EOFError,
            'ImportError': builtins.ImportError,
            'ModuleNotFoundError': builtins.ModuleNotFoundError,
            'IndexError': builtins.IndexError,
            'KeyError': builtins.KeyError,
            'MemoryError': builtins.MemoryError,
            'NameError': builtins.NameError,
            'UnboundLocalError': builtins.UnboundLocalError,
            'OSError': builtins.OSError,
            'OverflowError': builtins.OverflowError,
            'RuntimeError': builtins.RuntimeError,
            'RecursionError': builtins.RecursionError,
            'NotImplementedError': builtins.NotImplementedError,
            'SyntaxError': builtins.SyntaxError,
            'IndentationError': builtins.IndentationError,
            'TabError': builtins.TabError,
            'ReferenceError': builtins.ReferenceError,
            'SystemError': builtins.SystemError,
            'TypeError': builtins.TypeError,
            'ValueError': builtins.ValueError,
            'UnicodeError': builtins.UnicodeError,
            'UnicodeDecodeError': builtins.UnicodeDecodeError,
            'UnicodeEncodeError': builtins.UnicodeEncodeError,
            'UnicodeTranslateError': builtins.UnicodeTranslateError,
            'ZeroDivisionError': builtins.ZeroDivisionError,
            'FloatingPointError': builtins.FloatingPointError,
            
            # 其他必要的内建对象
            'None': None,
            'True': True,
            'False': False,
            '__import__': builtins.__import__,  # 允许import，但会被跟踪
        }
    }
    return restricted_globals

def set_function(function_code):
    """设置全局计算函数（优化版本：防止重复加载和内存累积）"""
    global CALCULATE_FUNCTION, FUNCTION_CODE_HASH, IMPORTED_MODULES
    
    if not function_code:
        return
    
    try:
        # 计算函数代码哈希，避免重复加载
        code_hash = hashlib.md5(function_code.encode('utf-8')).hexdigest()
        
        if FUNCTION_CODE_HASH == code_hash:
            # 函数代码未变化，跳过重新加载
            return
        
        # 清理之前的导入模块（保留系统模块）
        if FUNCTION_CODE_HASH is not None:  # 不是第一次加载
            cleanup_imported_modules()
        
        # 检查是源代码还是dill序列化字符串
        if "def " in function_code:
            # 使用受限的命名空间执行用户代码
            restricted_globals = get_restricted_globals()
            
            # 记录执行前的模块列表
            modules_before = set(sys.modules.keys())
            
            # 执行用户函数代码
            exec(function_code, restricted_globals)
            
            # 记录新导入的模块
            modules_after = set(sys.modules.keys())
            new_modules = modules_after - modules_before
            IMPORTED_MODULES.update(new_modules)
            
            # 寻找定义的第一个函数
            func_name = [name for name, obj in restricted_globals.items() 
                        if callable(obj) and not name.startswith("__")][0]
            CALCULATE_FUNCTION = restricted_globals[func_name]
            
            # 清理局部命名空间（保留函数引用）
            restricted_globals.clear()
            
        else:
            # 假设是dill序列化的
            import dill
            from base64 import b64decode
            decoded_bytes = b64decode(function_code.encode('utf-8'))
            CALCULATE_FUNCTION = dill.loads(decoded_bytes)
        
        # 更新函数代码哈希
        FUNCTION_CODE_HASH = code_hash
        
        # 强制垃圾收集
        gc.collect()
        
    except Exception as e:
        # 如果在函数加载阶段就失败，记录错误信息
        error_message = f"Failed to load function: {e}\n{traceback.format_exc()}"
        CALCULATE_FUNCTION = error_message  # 将错误信息存起来，在执行时报告
        # 不更新哈希，保持之前的状态


def execute_tasks(tasks):
    """执行任务列表（优化版本：避免内存累积）"""
    global CALCULATE_FUNCTION
    
    if not callable(CALCULATE_FUNCTION):
        error_msg = f"CALCULATE_FUNCTION is not valid: {CALCULATE_FUNCTION}"
        # 避免创建大列表，直接返回
        return {
            "results": [[] for _ in range(len(tasks))], 
            "errors": [error_msg for _ in range(len(tasks))], 
            "task_count": len(tasks)
        }

    # 使用生成器和流式处理，避免累积大量结果
    results = []
    errors = []
    
    # 预分配准确大小，避免动态扩展
    results = [None] * len(tasks)
    errors = [None] * len(tasks)
    
    for i, task in enumerate(tasks):
        try:
            date = task['date']
            code = task['code']
            
            # 简化为纯函数调用，不支持Go类
            facs = CALCULATE_FUNCTION(date, code)
            
            if not isinstance(facs, list):
                facs = list(facs)
            
            # 清理NaN/Inf值
            cleaned_facs = clean_numeric_values(facs)
            results[i] = cleaned_facs
            errors[i] = None  # 无错误
            
            # 定期垃圾收集和内存清理（每20个任务）
            if (i + 1) % 20 == 0:
                gc.collect()
                # 清理已处理的中间变量
                facs = None
                cleaned_facs = None
                
        except Exception as e:
            error_message = f"Error processing task {task}: {e}\n{traceback.format_exc()}"
            errors[i] = error_message
            results[i] = []
    
    # 过滤None值，创建最终的紧凑列表
    final_results = []
    final_errors = []
    
    for i in range(len(tasks)):
        final_results.append(results[i] if results[i] is not None else [])
        final_errors.append(errors[i] if errors[i] is not None else "")
    
    # 立即清理临时数组
    results = None
    errors = None
    
    # 最终内存清理
    gc.collect()
    
    # 内存监控（仅对大批次）
    try:
        if len(final_results) >= 50:
            import psutil
            import os
            process = psutil.Process(os.getpid())
            memory_mb = process.memory_info().rss / 1024 / 1024
            if memory_mb > 800:  # 降低警告阈值到800MB
                final_errors.append(f"Warning: Worker process memory usage: {memory_mb:.1f}MB")
    except ImportError:
        pass
    except Exception:
        pass

    return {"results": final_results, "errors": final_errors, "task_count": len(final_results)}

def main():
    """主工作循环"""
    current_tasks = []
    task_count = 0  # 添加任务计数器
    for line in sys.stdin:
        try:
            line = line.strip()
            if not line:
                continue

            command_data = json.loads(line)
            command_type = list(command_data.keys())[0]
            command_value = list(command_data.values())[0]

            if command_type == "Task":
                current_tasks.append(command_value)
            elif command_type == "FunctionCode":
                set_function(command_value)
            elif command_type == "Execute":
                if current_tasks:
                    response = execute_tasks(current_tasks)
                    print(json.dumps(response), flush=True)
                    task_count += len(current_tasks)  # 更新任务计数器
                    current_tasks = []  # 强制重新分配而非clear()
                    
                    # 每1万任务进行内存清理和进度报告
                    if task_count % 10000 == 0:
                        gc.collect()
                        # 获取内存使用情况
                        try:
                            import psutil
                            import os
                            process = psutil.Process(os.getpid())
                            memory_mb = process.memory_info().rss / 1024 / 1024
                            print(f"Progress: {task_count} tasks completed, memory: {memory_mb:.1f}MB", file=sys.stderr)
                        except ImportError:
                            print(f"Progress: {task_count} tasks completed", file=sys.stderr)
                        except Exception:
                            print(f"Progress: {task_count} tasks completed", file=sys.stderr)
            elif command_type == "Ping":
                print(json.dumps({"status": "pong"}), flush=True)
            elif command_type == "Exit":
                break
        except (json.JSONDecodeError, IndexError):
            continue
        except Exception as e:
            error_response = {
                "results": [],
                "errors": [f"Main loop error: {e}\n{traceback.format_exc()}"],
                "task_count": len(current_tasks)
            }
            print(json.dumps(error_response), flush=True)
            current_tasks = []


if __name__ == "__main__":
    main() 