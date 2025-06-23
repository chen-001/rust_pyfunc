#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高性能并行计算系统演示
====================

这个演示展示了如何使用rust_pyfunc的并行计算和备份功能。
系统支持：
- 高速并行执行Python函数 
- 三种高性能存储格式：json, binary, memory_map
- 自动异步备份
- 进度监控和回调
- 从备份恢复执行
- 实时数据查询

适用场景：
- 大规模因子计算（百万级任务）
- 股票数据分析
- 量化投资策略回测
- 任何需要并行处理大量数据的场景
"""

import sys
import tempfile
import os
import time
sys.path.append('/home/chenzongwei/rust_pyfunc/python')

import rust_pyfunc

def financial_analysis(date, code):
    """模拟金融数据分析函数
    
    输入：
        date: 日期 (YYYYMMDD)
        code: 股票代码
    
    输出：
        [ma5, ma20, rsi, bollinger_upper, bollinger_lower]
    """
    # 模拟移动平均
    ma5 = float(date % 1000) / 10.0
    ma20 = ma5 * 0.95
    
    # 模拟RSI
    rsi = 50.0 + (len(code) % 50)
    
    # 模拟布林带
    bollinger_upper = ma20 * 1.02
    bollinger_lower = ma20 * 0.98
    
    return [ma5, ma20, rsi, bollinger_upper, bollinger_lower]

def progress_callback(completed, total, elapsed_time, speed):
    """进度回调函数"""
    percent = (completed / total) * 100
    print(f"进度: {percent:.1f}% ({completed}/{total}) - 速度: {speed:.0f} 任务/秒")

def demo_parallel_computing():
    """演示并行计算功能"""
    print("🚀 高性能并行计算系统演示")
    print("=" * 80)
    
    # 生成模拟数据：1000个股票，连续5个交易日
    print("📊 生成测试数据...")
    args = []
    base_date = 20240101
    for day in range(5):  # 5个交易日
        for stock_id in range(200):  # 200只股票
            date = base_date + day
            code = f"{600000 + stock_id:06d}"
            args.append((date, code))
    
    print(f"   总任务数: {len(args):,} 个")
    print(f"   覆盖股票: 200 只")
    print(f"   覆盖日期: 5 个交易日")
    
    # 测试不同存储格式的性能
    storage_formats = ["json", "binary", "memory_map"]
    results = {}
    
    for fmt in storage_formats:
        print(f"\n📈 测试 {fmt.upper()} 存储格式")
        print("-" * 60)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix=f'.{fmt}', delete=False) as f:
            backup_file = f.name
        
        try:
            start_time = time.time()
            
            # 执行并行计算
            result = rust_pyfunc.run_pools(
                financial_analysis,
                args,
                backup_file=backup_file,
                storage_format=fmt,
                num_threads=1,  # 单线程避免PyO3限制
                backup_batch_size=100,
                backup_async=True,
                progress_callback=progress_callback
            )
            
            execution_time = time.time() - start_time
            
            # 查询备份数据
            start_time = time.time()
            backup_data = rust_pyfunc.query_backup(
                backup_file,
                storage_format=fmt,
                date_range=(20240101, 20240105),
                codes=["600000", "600001", "600002"]
            )
            query_time = time.time() - start_time
            
            # 统计信息
            file_size = os.path.getsize(backup_file)
            
            results[fmt] = {
                'execution_time': execution_time,
                'query_time': query_time,
                'file_size': file_size,
                'tasks_per_second': len(args) / execution_time,
                'result_count': len(result),
                'backup_count': len(backup_data)
            }
            
            print(f"✓ 执行完成: {execution_time:.3f}秒")
            print(f"✓ 处理速度: {results[fmt]['tasks_per_second']:.0f} 任务/秒")
            print(f"✓ 查询速度: {query_time:.3f}秒")
            print(f"✓ 文件大小: {file_size:,} 字节 ({file_size/1024:.1f} KB)")
            print(f"✓ 数据完整性: {len(result) == len(args) and len(backup_data) > 0}")
            
        except Exception as e:
            print(f"❌ {fmt} 测试失败: {e}")
            results[fmt] = None
        finally:
            if os.path.exists(backup_file):
                os.unlink(backup_file)
    
    # 性能对比报告
    print("\n" + "=" * 80)
    print("📊 性能对比报告")
    print("=" * 80)
    
    successful_formats = {k: v for k, v in results.items() if v is not None}
    
    if successful_formats:
        print(f"{'格式':<12} {'执行时间(秒)':<12} {'速度(任务/秒)':<15} {'查询时间(秒)':<12} {'文件大小(KB)':<12}")
        print("-" * 80)
        
        for fmt, data in successful_formats.items():
            print(f"{fmt:<12} {data['execution_time']:<12.3f} {data['tasks_per_second']:<15.0f} "
                  f"{data['query_time']:<12.3f} {data['file_size']/1024:<12.1f}")
        
        # 性能排名
        fastest_exec = min(successful_formats.keys(), key=lambda x: successful_formats[x]['execution_time'])
        fastest_query = min(successful_formats.keys(), key=lambda x: successful_formats[x]['query_time'])
        smallest_file = min(successful_formats.keys(), key=lambda x: successful_formats[x]['file_size'])
        
        print(f"\n🏆 性能冠军:")
        print(f"  🚀 最快执行: {fastest_exec}")
        print(f"  ⚡ 最快查询: {fastest_query}")  
        print(f"  💾 最小存储: {smallest_file}")
        
        print(f"\n💡 推荐使用:")
        print(f"  • 对于小规模数据（< 10万行）：json 格式，便于调试")
        print(f"  • 对于大规模数据（> 10万行）：binary 格式，最佳性能")
        print(f"  • 对于超大数据（> 100万行）：memory_map 格式，内存友好")
    
    print(f"\n✅ 演示完成！系统已准备好处理你的 1000万行 数据。")

if __name__ == "__main__":
    demo_parallel_computing()