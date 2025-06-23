import sys
import json
import base64
import traceback

CALCULATE_FUNCTION = None
GO_CLASS_INSTANCE = None

def set_go_class(serialized_go_class):
    """反序列化并设置全局的go_class实例"""
    global GO_CLASS_INSTANCE
    if serialized_go_class:
        try:
            import dill
            decoded_bytes = base64.b64decode(serialized_go_class.encode('utf-8'))
            GO_CLASS_INSTANCE = dill.loads(decoded_bytes)
        except Exception as e:
            # 在这里我们不抛出异常，而是在执行时报告错误
            GO_CLASS_INSTANCE = f"Failed to deserialize go_class: {e}"

def set_function(function_code):
    """设置全局计算函数"""
    global CALCULATE_FUNCTION
    if function_code:
        try:
            # 检查是源代码还是dill序列化字符串
            if "def " in function_code:
                # 动态执行以定义函数
                exec_globals = {}
                exec(function_code, exec_globals)
                # 寻找定义的第一个函数
                func_name = [name for name, obj in exec_globals.items() if callable(obj) and not name.startswith("__")][0]
                CALCULATE_FUNCTION = exec_globals[func_name]
            else:
                # 假设是dill序列化的
                import dill
                from base64 import b64decode
                decoded_bytes = b64decode(function_code.encode('utf-8'))
                CALCULATE_FUNCTION = dill.loads(decoded_bytes)
        except Exception as e:
            # 如果在函数加载阶段就失败，很难恢复
            error_message = f"Failed to load function: {e}\n{traceback.format_exc()}"
            CALCULATE_FUNCTION = error_message # 将错误信息存起来，在执行时报告


def execute_tasks(tasks):
    """执行任务列表"""
    global GO_CLASS_INSTANCE, CALCULATE_FUNCTION
    results = []
    errors = []
    
    if not callable(CALCULATE_FUNCTION):
        error_msg = f"CALCULATE_FUNCTION is not valid: {CALCULATE_FUNCTION}"
        for _ in tasks:
            errors.append(error_msg)
            results.append([])
        return {"results": results, "errors": errors, "task_count": len(tasks)}

    for task in tasks:
        try:
            date = task['date']
            code = task['code']
            
            if isinstance(GO_CLASS_INSTANCE, str) and GO_CLASS_INSTANCE.startswith("Failed to deserialize"):
                raise TypeError(GO_CLASS_INSTANCE)
            
            if GO_CLASS_INSTANCE is not None:
                facs = CALCULATE_FUNCTION(GO_CLASS_INSTANCE, date, code)
            else:
                facs = CALCULATE_FUNCTION(date, code)
            
            if not isinstance(facs, list):
                facs = list(facs)

            results.append(facs)
        except Exception as e:
            error_message = f"Error processing task {task}: {e}\n{traceback.format_exc()}"
            errors.append(error_message)
            results.append([])

    return {"results": results, "errors": errors, "task_count": len(tasks)}

def main():
    """主工作循环"""
    current_tasks = []
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
            elif command_type == "GoClass":
                set_go_class(command_value)
            elif command_type == "FunctionCode":
                set_function(command_value)
            elif command_type == "Execute":
                if current_tasks:
                    response = execute_tasks(current_tasks)
                    print(json.dumps(response), flush=True)
                    current_tasks = []
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