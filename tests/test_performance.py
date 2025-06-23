#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import tempfile
import os
import time
sys.path.append('/home/chenzongwei/rust_pyfunc/python')

import rust_pyfunc

def simple_analysis(date, code):
    """模拟简单的因子计算"""
    return [float(date % 1000), float(len(code)), 3.14159, 2.71828, 1.41421]

def performance_test(num_tasks=1000):
    print(f"🚀 性能测试 - {num_tasks} 个任务")
    print("=" * 60)
    
    # 生成测试数据
    args = [(20220101 + i, f"{1000000 + i:06d}") for i in range(num_tasks)]
    
    formats = ["json", "binary", "memory_map"]
    results = {}
    
    for fmt in formats:
        print(f"\n--- 测试 {fmt} 格式 ---")
        
        with tempfile.NamedTemporaryFile(mode='w', suffix=f'.{fmt}', delete=False) as f:
            backup_file = f.name
        
        try:
            # 执行测试
            start_time = time.time()
            result = rust_pyfunc.run_pools(
                simple_analysis,
                args,
                backup_file=backup_file,
                storage_format=fmt,
                num_threads=1,
                backup_batch_size=100
            )
            execution_time = time.time() - start_time
            
            # 查询测试
            start_time = time.time()
            backup_data = rust_pyfunc.query_backup(backup_file, storage_format=fmt)
            query_time = time.time() - start_time
            
            # 文件大小
            file_size = os.path.getsize(backup_file) if os.path.exists(backup_file) else 0
            
            results[fmt] = {
                'execution_time': execution_time,
                'query_time': query_time,
                'file_size': file_size,
                'tasks_per_second': num_tasks / execution_time if execution_time > 0 else 0,
                'success': len(result) == num_tasks and len(backup_data) == num_tasks
            }
            
            print(f"✓ 执行时间: {execution_time:.3f}秒 ({results[fmt]['tasks_per_second']:.0f} 任务/秒)")
            print(f"✓ 查询时间: {query_time:.3f}秒")
            print(f"✓ 文件大小: {file_size:,} 字节 ({file_size/1024:.1f} KB)")
            print(f"✓ 数据完整性: {'通过' if results[fmt]['success'] else '失败'}")
            
        except Exception as e:
            print(f"❌ {fmt} 测试失败: {e}")
            results[fmt] = {'success': False}
        finally:
            if os.path.exists(backup_file):
                os.unlink(backup_file)
    
    # 总结报告
    print("\n" + "=" * 60)
    print("📊 性能测试总结")
    print("=" * 60)
    
    success_count = sum(1 for r in results.values() if r.get('success', False))
    print(f"成功格式: {success_count}/{len(formats)}")
    
    if success_count > 0:
        print("\n性能对比:")
        print(f"{'格式':<12} {'执行(秒)':<10} {'速度(任务/秒)':<15} {'查询(秒)':<10} {'大小(KB)':<10}")
        print("-" * 60)
        
        for fmt, data in results.items():
            if data.get('success', False):
                print(f"{fmt:<12} {data['execution_time']:<10.3f} {data['tasks_per_second']:<15.0f} "
                      f"{data['query_time']:<10.3f} {data['file_size']/1024:<10.1f}")
        
        # 找出最快的格式
        fastest_exec = min((fmt for fmt, data in results.items() if data.get('success', False)), 
                          key=lambda fmt: results[fmt]['execution_time'])
        fastest_query = min((fmt for fmt, data in results.items() if data.get('success', False)), 
                           key=lambda fmt: results[fmt]['query_time'])
        smallest_size = min((fmt for fmt, data in results.items() if data.get('success', False)), 
                           key=lambda fmt: results[fmt]['file_size'])
        
        print(f"\n🏆 最佳性能:")
        print(f"  最快执行: {fastest_exec}")
        print(f"  最快查询: {fastest_query}")
        print(f"  最小文件: {smallest_size}")
    
    return results

if __name__ == "__main__":
    # 小规模测试
    performance_test(100)
    
    print("\n\n")
    
    # 中等规模测试
    performance_test(1000)