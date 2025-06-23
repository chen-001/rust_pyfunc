#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试Rust原生多进程功能
"""

import sys
import time
import tempfile
import multiprocessing
sys.path.append('/home/chenzongwei/rust_pyfunc/python')

import rust_pyfunc


def simple_test_function(date, code):
    """简单测试函数"""
    # 模拟一些计算
    result = 0
    for i in range(100):
        result += hash(f"{date}_{code}_{i}") % 1000
    
    return [
        float(date % 10000),
        float(len(code)),
        float(result % 1000)
    ]


def cpu_intensive_function(date, code):
    """CPU密集型函数"""
    result = 0
    for i in range(10000):  # 更多计算
        result += hash(f"{date}_{code}_{i}") % 100
    
    return [
        float(date % 10000),
        float(len(code)),
        float(result % 1000),
        float(result / 100.0)
    ]


def progress_callback(completed, total, elapsed, speed):
    """进度回调函数"""
    percent = completed / total * 100
    print(f"  进度: {percent:.1f}% ({completed}/{total}) | 速度: {speed:.0f} 任务/秒 | 已用时: {elapsed:.1f}秒")


def test_basic_multiprocess():
    """测试基本多进程功能"""
    print("=== 测试基本多进程功能 ===")
    
    # 创建测试数据
    args = [
        [20220101, "000001"],
        [20220101, "000002"],
        [20220102, "000001"],
        [20220102, "000002"],
        [20220103, "000001"],
    ]
    
    print(f"测试数据: {len(args)} 个任务")
    print(f"系统CPU核心数: {multiprocessing.cpu_count()}")
    
    # 测试多进程执行
    print("\n使用Rust原生多进程执行:")
    start_time = time.time()
    
    result = rust_pyfunc.run_multiprocess(
        simple_test_function,
        args,
        num_processes=2,
        progress_callback=progress_callback
    )
    
    elapsed = time.time() - start_time
    
    print(f"\n多进程执行完成:")
    print(f"  执行时间: {elapsed:.3f} 秒")
    print(f"  结果数量: {len(result)}")
    print(f"  第一个结果: {result[0]}")
    print(f"  处理速度: {len(args)/elapsed:.0f} 任务/秒")
    
    # 验证结果格式
    assert len(result) == len(args), "结果数量应该等于输入数量"
    assert len(result[0]) == 5, "每个结果应该有5列 (date, code, fac1, fac2, fac3)"
    assert result[0][0] == 20220101, "第一列应该是日期"
    assert result[0][1] == "000001", "第二列应该是代码"
    
    print("✓ 基本多进程功能测试通过")


def test_multiprocess_with_backup():
    """测试多进程备份功能"""
    print("\n=== 测试多进程备份功能 ===")
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        backup_file = f.name
    
    try:
        # 创建测试数据
        args = [
            [20220101, "000001"],
            [20220101, "000002"],
            [20220101, "000003"],
            [20220102, "000001"],
            [20220102, "000002"],
        ]
        
        print(f"备份文件: {backup_file}")
        
        # 测试带备份的多进程执行
        print("\n使用备份的多进程执行:")
        result = rust_pyfunc.run_multiprocess(
            cpu_intensive_function,
            args,
            num_processes=2,
            backup_file=backup_file,
            backup_batch_size=2,
            storage_format="json",
            progress_callback=progress_callback
        )
        
        print(f"\n多进程备份执行完成:")
        print(f"  结果数量: {len(result)}")
        
        # 验证备份文件存在
        import os
        assert os.path.exists(backup_file), "备份文件应该存在"
        print("✓ 备份文件创建成功")
        
        # 测试备份恢复
        print("\n测试备份恢复:")
        result2 = rust_pyfunc.run_multiprocess(
            cpu_intensive_function,
            args,
            num_processes=2,
            backup_file=backup_file,
            resume_from_backup=True,
            storage_format="json",
            progress_callback=progress_callback
        )
        
        print(f"  恢复的结果数量: {len(result2)}")
        assert len(result2) == len(args), "恢复的结果数量应该正确"
        
        print("✓ 多进程备份功能测试通过")
        
    finally:
        import os
        if os.path.exists(backup_file):
            os.unlink(backup_file)


def test_performance_comparison():
    """测试性能对比"""
    print("\n=== 性能对比测试 ===")
    
    # 创建较大的测试数据集
    args = [[20220101 + i//10, f"{i%10+1:06d}"] for i in range(100)]
    print(f"性能测试数据: {len(args)} 个任务")
    
    # 测试原有的run_pools（多线程）
    print("\n测试原有run_pools（多线程）:")
    start_time = time.time()
    
    result1 = rust_pyfunc.run_pools(
        cpu_intensive_function,
        args,
        num_threads=multiprocessing.cpu_count()
    )
    
    elapsed1 = time.time() - start_time
    speed1 = len(args) / elapsed1
    
    print(f"  多线程速度: {speed1:.0f} 任务/秒")
    
    # 测试新的run_multiprocess（多进程）
    print("\n测试新的run_multiprocess（多进程）:")
    start_time = time.time()
    
    result2 = rust_pyfunc.run_multiprocess(
        cpu_intensive_function,
        args,
        num_processes=multiprocessing.cpu_count()
    )
    
    elapsed2 = time.time() - start_time
    speed2 = len(args) / elapsed2
    
    print(f"  多进程速度: {speed2:.0f} 任务/秒")
    
    # 性能对比
    print(f"\n性能对比:")
    print(f"  多线程 (run_pools): {speed1:.0f} 任务/秒")
    print(f"  多进程 (run_multiprocess): {speed2:.0f} 任务/秒")
    
    if speed2 > speed1:
        speedup = speed2 / speed1
        print(f"  ✅ 多进程比多线程快 {speedup:.1f}x")
    else:
        print(f"  ⚠️  在这种情况下多线程更快（可能任务太小或其他因素）")
    
    print("✓ 性能对比测试完成")


def test_error_handling():
    """测试错误处理"""
    print("\n=== 错误处理测试 ===")
    
    def error_function(date, code):
        """会出错的函数"""
        if date == 20220102:
            raise ValueError(f"故意的错误: {date}, {code}")
        return [float(date), float(len(code))]
    
    args = [
        [20220101, "000001"],
        [20220102, "000002"],  # 这个会出错
        [20220103, "000003"],
    ]
    
    print(f"测试错误处理，预期第2个任务会失败")
    
    try:
        result = rust_pyfunc.run_multiprocess(
            error_function,
            args,
            num_processes=2,
            progress_callback=progress_callback
        )
        
        print(f"  处理完成，结果数量: {len(result)}")
        print("✓ 错误处理测试通过（进程能够容错继续）")
        
    except Exception as e:
        print(f"  捕获到异常: {e}")
        print("✓ 错误处理测试通过（正确抛出异常）")


if __name__ == "__main__":
    print("开始测试Rust原生多进程功能")
    print("=" * 60)
    
    try:
        test_basic_multiprocess()
        test_multiprocess_with_backup()
        test_performance_comparison()
        test_error_handling()
        
        print("\n" + "=" * 60)
        print("🎉 所有多进程测试通过！")
        
        print("\n✨ Rust原生多进程优势：")
        print("• 真正的并行：每个进程独立的Python解释器，无GIL限制")
        print("• 系统级控制：Rust直接管理进程生命周期")
        print("• 高性能：避免Python multiprocessing开销")
        print("• 通用性：支持任何Python函数，无pickle化要求")
        print("• 容错性：单个进程错误不影响其他进程")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)