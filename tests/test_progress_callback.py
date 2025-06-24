#!/usr/bin/env python3
"""
测试进度回调功能是否正常工作
"""

import time
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))
sys.path.insert(0, '/home/chenzongwei/design_whatever')

import rust_pyfunc as rf
from tool_whatever import WebTqdmforRust

def simple_test_function(date, code):
    """简单的测试函数"""
    time.sleep(0.02)  # 模拟20ms的计算时间
    return [1.0, 2.0, 3.0]

def test_progress_callback():
    """测试进度回调功能"""
    
    print("=== 测试进度回调功能 ===")
    
    # 准备测试数据
    test_args = []
    for date in range(20240101, 20240105):  # 4天
        for code in ['000001', '000002', '600000']:  # 3只股票
            test_args.append([date, code])
    
    total_tasks = len(test_args)
    print(f"总任务数: {total_tasks}")
    
    # 创建进度回调对象
    progress_callback = WebTqdmforRust(
        total=total_tasks, 
        name="测试异步流水线进度", 
        server_url='http://localhost:5101'
    )
    
    start_time = time.time()
    
    try:
        results = rf.run_pools(
            func=simple_test_function,
            args=test_args,
            num_threads=4,
            backup_file=None,
            progress_callback=progress_callback
        )
        
        end_time = time.time()
        total_time = end_time - start_time
        
        print(f"✅ 任务执行成功!")
        print(f"总耗时: {total_time:.2f}秒")
        print(f"结果形状: {results.shape}")
        
        # 验证结果
        assert results.shape[0] == total_tasks, f"结果行数不匹配: {results.shape[0]} vs {total_tasks}"
        
        print("✅ 进度回调测试通过!")
        print("请检查 http://localhost:5101 页面查看进度条是否正常显示")
        
        return True
        
    except Exception as e:
        progress_callback.set_error(str(e))
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_progress_callback()
    if success:
        print("\n🎉 进度回调功能测试成功!")
        print("💡 确保 5101 端口的 Web 服务正在运行以查看进度条")
    else:
        print("\n❌ 进度回调功能测试失败")
        sys.exit(1)