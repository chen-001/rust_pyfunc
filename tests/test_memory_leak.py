#!/usr/bin/env python3
"""
简化的内存泄漏检测脚本
"""

import os
import sys
import tempfile
import time
import psutil

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

def get_memory_usage():
    """获取当前进程内存使用量(MB)"""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024

def test_memory_leak():
    """测试内存泄漏"""
    print("开始内存泄漏检测...")
    
    # 测试函数
    def test_function(date, code):
        # 创建一些数据模拟真实计算
        data = [float(date + i) for i in range(100)]  # 100个浮点数
        return data[:10]  # 返回前10个
    
    try:
        import rust_pyfunc
        
        # 在模块导入后测量初始内存，避免模块加载的影响
        initial_memory = get_memory_usage()
        print(f"模块导入后初始内存: {initial_memory:.1f}MB")
        
        # 多轮测试，观察内存变化趋势
        for round_num in range(5):
            print(f"\n=== 第{round_num + 1}轮测试 ===")
            
            # 创建临时备份文件
            with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as tmp_file:
                backup_file = tmp_file.name
            
            try:
                # 每轮测试1000个任务
                test_args = [[20240101 + i, f"code_{i}"] for i in range(1000)]
                
                before_round = get_memory_usage()
                print(f"轮次开始前内存: {before_round:.1f}MB")
                
                # 执行计算
                results = rust_pyfunc.run_pools(
                    test_function,
                    test_args,
                    backup_file=backup_file,
                    num_threads=2,
                    backup_batch_size=50  # 小批次流式处理
                )
                
                after_round = get_memory_usage()
                print(f"轮次结束后内存: {after_round:.1f}MB")
                print(f"本轮内存增长: {after_round - before_round:.1f}MB")
                print(f"结果数量: {len(results)}")
                
                # 强制垃圾回收
                import gc
                gc.collect()
                
                after_gc = get_memory_usage()
                print(f"垃圾回收后内存: {after_gc:.1f}MB")
                
            finally:
                # 清理临时文件
                if os.path.exists(backup_file):
                    os.unlink(backup_file)
        
        final_memory = get_memory_usage()
        total_increase = final_memory - initial_memory
        
        print(f"\n=== 总体结果 ===")
        print(f"初始内存: {initial_memory:.1f}MB")
        print(f"最终内存: {final_memory:.1f}MB")
        print(f"总内存增长: {total_increase:.1f}MB")
        
        # 判断是否有严重内存泄漏
        if total_increase < 50:  # 小于50MB认为正常
            print("✅ 内存使用正常，无明显内存泄漏")
            return True
        elif total_increase < 100:  # 50-100MB警告
            print("⚠️ 内存增长较多，可能有轻微内存泄漏")
            return True
        else:  # 大于100MB认为有问题
            print("❌ 内存增长过多，存在明显内存泄漏")
            return False
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_memory_leak()
    sys.exit(0 if success else 1)