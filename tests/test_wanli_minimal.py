#!/usr/bin/env python3
"""
最小化万里长征测试 - 只测试几个任务
"""

import os
import sys
import tempfile

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'python'))
sys.path.insert(0, '/home/chenzongwei/pythoncode/万里长征')

def test_minimal_wanli():
    """最小化万里长征测试"""
    print("=== 最小化万里长征测试 ===")
    
    # 导入万里长征的go函数
    from 万里长征放宽参数20 import go as wanli_go
    print("✅ 成功导入万里长征go函数")
    
    # 手动测试几个任务，绕过dw.run_factor的进度条系统
    test_args = [
        [20170101, "000001"],  # 这个可能会出错
        [20170101, "000002"],  # 这个也可能会出错
    ]
    
    with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as tmp_file:
        backup_file = tmp_file.name
    
    try:
        import rust_pyfunc
        
        print("开始最小化万里长征测试（绕过dw.run_factor）...")
        print("使用我们的多进程系统直接调用万里长征go函数...")
        
        results = rust_pyfunc.run_pools(
            wanli_go,
            test_args,
            backup_file=backup_file,
            num_threads=2
        )
        
        print(f"✅ 最小化万里长征测试成功，结果数量: {len(results)}")
        
        # 检查结果
        for i, result in enumerate(results):
            print(f"结果{i}: 长度={len(result)}, 前5个值={result[:5] if len(result) >= 5 else result}")
        
        return True
        
    except Exception as e:
        print(f"❌ 最小化万里长征测试失败: {e}")
        print("现在应该能看到万里长征go函数的详细错误信息了！")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        if os.path.exists(backup_file):
            os.unlink(backup_file)

if __name__ == "__main__":
    print("开始最小化万里长征测试...")
    success = test_minimal_wanli()
    print(f"\n测试结果: {'✅ 成功' if success else '❌ 失败'}")
    
    if not success:
        print("\n关键信息：")
        print("1. 如果看到详细错误信息，说明错误报告系统工作正常")
        print("2. 如果卡住不动，说明万里长征go函数内部有死循环或长时间处理")
        print("3. 我们绕过了dw.run_factor，直接使用rust_pyfunc.run_pools调用")