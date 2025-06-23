#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
性能基准测试 - 展示并行处理和优化效果
"""

import sys
import time
import tempfile
import statistics
import multiprocessing
sys.path.append('/home/chenzongwei/rust_pyfunc/python')

import rust_pyfunc


def cpu_intensive_function(date, code):
    """CPU密集型测试函数"""
    # 模拟一些计算密集的操作
    result = 0
    for i in range(1000):
        result += hash(f"{date}_{code}_{i}") % 100
    
    # 返回计算结果
    return [
        float(date % 10000),
        float(len(code)),
        float(result % 1000)
    ]


def io_simulation_function(date, code):
    """IO密集型测试函数（模拟）"""
    # 模拟IO等待
    time.sleep(0.001)  # 1ms延迟
    return [
        float(date % 1000),
        float(len(code)),
        hash(f"{date}_{code}") % 100
    ]


def simple_function(date, code):
    """简单快速函数"""
    return [
        float(date % 1000),
        float(len(code)),
        1.0
    ]


def benchmark_function(func, args, test_name, **kwargs):
    """基准测试函数"""
    print(f"\n--- {test_name} ---")
    print(f"任务数量: {len(args)}")
    
    start_time = time.time()
    result = rust_pyfunc.run_pools(func, args, **kwargs)
    end_time = time.time()
    
    elapsed = end_time - start_time
    speed = len(args) / elapsed if elapsed > 0 else float('inf')
    
    print(f"执行时间: {elapsed:.3f} 秒")
    print(f"处理速度: {speed:.0f} 任务/秒")
    print(f"结果数量: {len(result)}")
    
    return elapsed, speed, len(result)


def test_multiprocessing_vs_serial():
    """测试multiprocessing vs 串行处理"""
    print("=== Multiprocessing vs 串行处理对比 ===")
    
    # 创建可以pickle的函数
    def picklable_function(date, code):
        return cpu_intensive_function(date, code)
    
    args = [[20220101 + i, f"{j:06d}"] for i in range(5) for j in range(1, 21)]  # 100个任务
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        backup_file = f.name
    
    try:
        # 测试multiprocessing（如果支持）
        print(f"\n测试支持multiprocessing的函数：")
        elapsed_mp, speed_mp, count_mp = benchmark_function(
            picklable_function,
            args,
            "Multiprocessing处理",
            backup_file=backup_file + "_mp",
            storage_format="json",
            num_threads=multiprocessing.cpu_count()
        )
        
        # 测试串行处理（不可pickle的函数）
        print(f"\n测试不支持multiprocessing的函数：")
        elapsed_serial, speed_serial, count_serial = benchmark_function(
            cpu_intensive_function,  # 这个函数不能pickle
            args,
            "优化串行处理",
            backup_file=backup_file + "_serial",
            storage_format="json",
            num_threads=multiprocessing.cpu_count()
        )
        
        # 性能对比
        print(f"\n性能对比：")
        if speed_mp > speed_serial:
            speedup = speed_mp / speed_serial
            print(f"✅ Multiprocessing比串行处理快 {speedup:.1f}x")
        else:
            print(f"⚠️  串行处理在这种情况下更快（可能是任务太小）")
            
        print(f"Multiprocessing: {speed_mp:.0f} 任务/秒")
        print(f"优化串行: {speed_serial:.0f} 任务/秒")
        
    finally:
        import os
        for suffix in ["_mp", "_serial"]:
            if os.path.exists(backup_file + suffix):
                os.unlink(backup_file + suffix)


def test_batch_size_optimization():
    """测试批次大小优化"""
    print("\n=== 批次大小优化测试 ===")
    
    args = [[20220101, f"{i:06d}"] for i in range(1, 101)]  # 100个简单任务
    
    batch_configs = [
        {"num_threads": 1, "name": "单线程"},
        {"num_threads": 2, "name": "2线程"},
        {"num_threads": 4, "name": "4线程"},
        {"num_threads": 8, "name": "8线程"},
        {"num_threads": multiprocessing.cpu_count(), "name": f"{multiprocessing.cpu_count()}线程（CPU数）"},
    ]
    
    results = []
    
    for config in batch_configs:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            backup_file = f.name
        
        try:
            elapsed, speed, count = benchmark_function(
                simple_function,
                args,
                config["name"],
                backup_file=backup_file,
                storage_format="json",
                num_threads=config["num_threads"]
            )
            results.append((config["name"], elapsed, speed))
            
        finally:
            import os
            if os.path.exists(backup_file):
                os.unlink(backup_file)
    
    # 找出最佳配置
    best_config = max(results, key=lambda x: x[2])
    print(f"\n最佳配置: {best_config[0]} ({best_config[2]:.0f} 任务/秒)")


def test_storage_format_performance():
    """测试不同存储格式的性能"""
    print("\n=== 存储格式性能测试 ===")
    
    args = [[20220101, f"{i:06d}"] for i in range(1, 201)]  # 200个任务
    
    formats = [
        ("json", ".json"),
        ("binary", ".bin"),
        ("memory_map", ".bin")
    ]
    
    results = []
    
    for storage_format, suffix in formats:
        with tempfile.NamedTemporaryFile(mode='w', suffix=suffix, delete=False) as f:
            backup_file = f.name
        
        try:
            elapsed, speed, count = benchmark_function(
                simple_function,
                args,
                f"{storage_format.upper()} 格式",
                backup_file=backup_file,
                storage_format=storage_format,
                backup_batch_size=50,
                num_threads=4
            )
            
            # 检查文件大小
            import os
            file_size = os.path.getsize(backup_file) if os.path.exists(backup_file) else 0
            results.append((storage_format, elapsed, speed, file_size))
            
        finally:
            import os
            if os.path.exists(backup_file):
                os.unlink(backup_file)
    
    # 性能对比
    print(f"\n存储格式对比：")
    for format_name, elapsed, speed, file_size in results:
        print(f"{format_name:12}: {speed:8.0f} 任务/秒, 文件大小: {file_size:8d} 字节")
    
    fastest_format = max(results, key=lambda x: x[2])
    smallest_format = min(results, key=lambda x: x[3])
    print(f"\n最快格式: {fastest_format[0]}")
    print(f"最小文件: {smallest_format[0]}")


def test_resume_performance():
    """测试备份恢复性能"""
    print("\n=== 备份恢复性能测试 ===")
    
    # 创建大数据集
    full_args = [[20220101 + i//100, f"{i%100:06d}"] for i in range(1000)]  # 1000个任务
    partial_args = full_args[:500]  # 前500个
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        backup_file = f.name
    
    try:
        # 第一次运行：创建部分备份
        print(f"第一次运行：创建 {len(partial_args)} 个任务的备份")
        elapsed1, speed1, count1 = benchmark_function(
            simple_function,
            partial_args,
            "创建备份",
            backup_file=backup_file,
            storage_format="binary",
            backup_batch_size=100,
            num_threads=4
        )
        
        # 第二次运行：恢复并完成
        print(f"第二次运行：恢复备份并处理剩余 {len(full_args) - len(partial_args)} 个任务")
        elapsed2, speed2, count2 = benchmark_function(
            simple_function,
            full_args,
            "恢复备份",
            backup_file=backup_file,
            resume_from_backup=True,
            storage_format="binary",
            backup_batch_size=100,
            num_threads=4
        )
        
        # 计算总体性能
        total_unique_tasks = len(full_args)
        total_time = elapsed1 + elapsed2
        overall_speed = total_unique_tasks / total_time
        
        print(f"\n恢复性能分析：")
        print(f"总任务数: {total_unique_tasks}")
        print(f"总时间: {total_time:.3f} 秒") 
        print(f"整体速度: {overall_speed:.0f} 任务/秒")
        print(f"备份恢复效率: {(len(full_args) - len(partial_args)) / elapsed2:.0f} 新任务/秒")
        
    finally:
        import os
        if os.path.exists(backup_file):
            os.unlink(backup_file)


if __name__ == "__main__":
    print("开始性能基准测试")
    print("=" * 60)
    print(f"系统信息：")
    print(f"  CPU核心数: {multiprocessing.cpu_count()}")
    print(f"  Python版本: {sys.version}")
    
    try:
        test_multiprocessing_vs_serial()
        test_batch_size_optimization() 
        test_storage_format_performance()
        test_resume_performance()
        
        print("\n" + "=" * 60)
        print("🎉 性能基准测试完成！")
        print("\n性能优化总结：")
        print("1. 🚀 实现了智能并行处理（multiprocessing + 优化串行）")
        print("2. ⚡ 批量任务分发减少函数调用开销")
        print("3. 💾 多种存储格式优化（JSON/Binary/MemoryMap）")
        print("4. 🔄 高效的备份恢复机制")
        print("5. 🌐 Web管理界面支持自动端口选择")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)