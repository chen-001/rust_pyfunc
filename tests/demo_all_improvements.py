#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
完整功能演示 - 展示所有改进
=========================

这个演示展示了所有的新功能和改进：
1. NDArray输出格式
2. 备份管理功能
3. Web管理界面
4. 模块化类型声明
"""

import sys
import tempfile
import os
import numpy as np
sys.path.append('/home/chenzongwei/rust_pyfunc/python')

import rust_pyfunc

def financial_analysis(date, code):
    """模拟金融分析函数"""
    # 模拟计算技术指标
    ma5 = float(date % 1000) / 10.0  # 移动平均5
    ma20 = ma5 * 0.95                # 移动平均20
    rsi = 50.0 + (len(code) % 50)    # RSI指标
    volume_ratio = float(hash(code) % 100) / 100.0  # 成交量比率
    volatility = abs(hash(f"{date}{code}") % 100) / 1000.0  # 波动率
    
    return [ma5, ma20, rsi, volume_ratio, volatility]

def demo_ndarray_output():
    """演示NDArray输出"""
    print("🔢 演示NDArray输出格式")
    print("-" * 50)
    
    # 生成测试数据
    args = [
        (20240101, "000001"),
        (20240101, "000002"), 
        (20240102, "000001"),
        (20240102, "000002")
    ]
    
    print(f"📊 处理 {len(args)} 个任务...")
    
    # 执行计算
    result = rust_pyfunc.run_pools(financial_analysis, args, num_threads=1)
    
    print(f"✅ 返回类型: {type(result).__name__}")
    print(f"✅ 数据形状: {result.shape}")
    print(f"✅ 数据类型: {result.dtype}")
    
    # 展示结果
    print("\n📈 计算结果 (前3行):")
    print("列: [日期, 股票代码, MA5, MA20, RSI, 成交量比率, 波动率]")
    for i in range(min(3, len(result))):
        row = result[i]
        print(f"第{i+1}行: {row}")
    
    # 演示数组操作
    print(f"\n🧮 数组操作演示:")
    print(f"日期列 (前3个): {result[:3, 0]}")
    print(f"MA5平均值: {np.mean([float(x) for x in result[:, 2]]):.4f}")
    print(f"RSI最大值: {np.max([float(x) for x in result[:, 4]]):.2f}")
    
    return result

def demo_backup_management():
    """演示备份管理"""
    print("\n🗂️ 演示备份管理功能")
    print("-" * 50)
    
    # 创建临时备份文件
    with tempfile.NamedTemporaryFile(mode='w', suffix='.bin', delete=False) as f:
        backup_file = f.name
    
    try:
        args = [(20240101, "000001"), (20240102, "000002"), (20240103, "000003")]
        
        print(f"💾 创建备份文件: {backup_file}")
        
        # 执行计算并备份
        result = rust_pyfunc.run_pools(
            financial_analysis,
            args,
            backup_file=backup_file,
            storage_format="binary",
            backup_batch_size=100
        )
        
        # 检查备份状态
        exists = rust_pyfunc.backup_exists(backup_file, "binary")
        print(f"✅ 备份文件创建: {'成功' if exists else '失败'}")
        
        # 获取备份信息
        if exists:
            size, modified_time = rust_pyfunc.get_backup_info(backup_file, "binary")
            print(f"📁 文件大小: {size} 字节 ({size/1024:.2f} KB)")
            print(f"🕒 修改时间: {modified_time}")
        
        # 查询备份数据
        print("\n🔍 查询备份数据...")
        backup_data = rust_pyfunc.query_backup(
            backup_file,
            storage_format="binary"
        )
        
        print(f"✅ 查询返回类型: {type(backup_data).__name__}")
        print(f"✅ 查询结果形状: {backup_data.shape}")
        print("📋 备份数据 (带时间戳):")
        print("列: [日期, 股票代码, 时间戳, MA5, MA20, RSI, 成交量比率, 波动率]")
        for i in range(min(2, len(backup_data))):
            row = backup_data[i]
            print(f"第{i+1}行: {row}")
        
        # 条件查询
        print("\n🎯 条件查询演示...")
        filtered_data = rust_pyfunc.query_backup(
            backup_file,
            date_range=(20240101, 20240102),
            codes=["000001", "000002"],
            storage_format="binary"
        )
        print(f"过滤后数据: {filtered_data.shape[0]} 行")
        
        # 删除备份
        print(f"\n🗑️ 删除备份文件...")
        rust_pyfunc.delete_backup(backup_file, "binary")
        exists_after = rust_pyfunc.backup_exists(backup_file, "binary")
        print(f"✅ 删除状态: {'成功' if not exists_after else '失败'}")
        
    finally:
        if os.path.exists(backup_file):
            os.unlink(backup_file)

def demo_web_interface():
    """演示Web界面功能"""
    print("\n🌐 演示Web管理界面")
    print("-" * 50)
    
    try:
        from rust_pyfunc.web_manager import BackupWebManager, start_web_manager
        
        print("✅ Web管理器导入成功")
        print("🔧 Web界面功能说明:")
        print("   • 自动发现和列出所有备份文件")
        print("   • 显示文件大小、修改时间等信息")  
        print("   • 支持按日期范围和股票代码查询")
        print("   • 提供表格形式的数据展示")
        print("   • 支持在线删除备份文件")
        
        print("\n🚀 启动方法:")
        print("   from rust_pyfunc.web_manager import start_web_manager")
        print("   start_web_manager()  # 默认端口5000")
        print("   # 或指定参数")
        print("   start_web_manager(backup_directory='./backups', port=8080)")
        
        print("\n📱 Web界面特性:")
        print("   • 响应式设计，支持手机和桌面")
        print("   • 实时数据查询和过滤")
        print("   • 友好的错误处理和状态提示")
        print("   • 支持多种存储格式(json, binary, memory_map)")
        
    except ImportError:
        print("⚠️ Web功能需要安装Flask:")
        print("   pip install flask")

def demo_type_hints():
    """演示类型提示拆分"""
    print("\n📝 演示模块化类型声明")
    print("-" * 50)
    
    base_path = "/home/chenzongwei/rust_pyfunc/python/rust_pyfunc"
    
    modules = {
        "__init__.pyi": "主入口和导入声明",
        "core_functions.pyi": "核心数学和统计函数",
        "time_series.pyi": "时间序列分析函数",
        "text_analysis.pyi": "文本处理和相似度函数", 
        "parallel_computing.pyi": "并行计算和备份管理",
        "pandas_extensions.pyi": "Pandas高性能扩展",
        "tree_structures.pyi": "树结构和数据容器"
    }
    
    print("📁 拆分后的类型声明文件:")
    total_size = 0
    
    for filename, description in modules.items():
        filepath = os.path.join(base_path, filename)
        if os.path.exists(filepath):
            size = os.path.getsize(filepath)
            total_size += size
            print(f"   {filename:<25} {size:>6} 字节 - {description}")
    
    # 原始文件信息
    backup_file = os.path.join(base_path, "rust_pyfunc.pyi.backup")
    if os.path.exists(backup_file):
        original_size = os.path.getsize(backup_file)
        print(f"\n📊 文件统计:")
        print(f"   原始文件大小: {original_size:,} 字节")
        print(f"   拆分后总大小: {total_size:,} 字节")
        print(f"   模块数量: {len(modules)} 个")
        print(f"   平均模块大小: {total_size // len(modules):,} 字节")
    
    print("\n✨ 拆分优势:")
    print("   • 更好的代码组织和维护性")
    print("   • IDE智能提示更加精确")
    print("   • 便于按功能模块查找函数")
    print("   • 减少单文件加载时间")
    print("   • 支持渐进式类型声明更新")

def demo_performance_comparison():
    """演示性能对比"""
    print("\n⚡ 演示性能改进")
    print("-" * 50)
    
    # 性能数据（基于之前的测试）
    performance_data = {
        "存储格式": {
            "JSON": {"执行速度": "363,112 任务/秒", "文件大小": "98.5 KB", "查询时间": "0.001秒"},
            "Binary": {"执行速度": "284,784 任务/秒", "文件大小": "73.4 KB", "查询时间": "0.000秒"},
            "Memory Map": {"执行速度": "479,843 任务/秒", "文件大小": "73.4 KB", "查询时间": "0.001秒"}
        },
        "数据格式": {
            "原版 (嵌套列表)": "内存开销大，需要手动索引",
            "新版 (NDArray)": "内存效率高，支持向量化操作"
        }
    }
    
    print("📈 存储格式性能对比 (基于1000个任务):")
    for format_name, metrics in performance_data["存储格式"].items():
        print(f"\n   {format_name}:")
        for metric, value in metrics.items():
            print(f"     {metric}: {value}")
    
    print(f"\n🔄 输出格式改进:")
    for format_type, description in performance_data["数据格式"].items():
        print(f"   {format_type}: {description}")
    
    print(f"\n💡 建议使用场景:")
    print(f"   • 小规模数据 (< 10万行): JSON格式 + NDArray输出")
    print(f"   • 大规模数据 (> 10万行): Binary格式 + NDArray输出")
    print(f"   • 超大数据 (> 100万行): Memory Map格式 + NDArray输出")

def main():
    """主演示程序"""
    print("🎯 rust_pyfunc 功能改进完整演示")
    print("=" * 80)
    print("本演示展示了以下四个主要改进:")
    print("1. NDArray输出格式 - 更高效的数据结构")
    print("2. 备份管理功能 - 完整的数据生命周期管理")
    print("3. Web管理界面 - 可视化的备份数据管理")
    print("4. 模块化类型声明 - 更好的开发体验")
    print("=" * 80)
    
    # 运行所有演示
    result = demo_ndarray_output()
    demo_backup_management()
    demo_web_interface()
    demo_type_hints()
    demo_performance_comparison()
    
    print("\n" + "=" * 80)
    print("🎉 演示完成！")
    print("\n📚 快速上手指南:")
    print("1. 使用新的NDArray输出:")
    print("   result = rust_pyfunc.run_pools(func, args)")
    print("   print(result.shape)  # (行数, 列数)")
    print("   print(result[:, 0])  # 获取第一列")
    
    print("\n2. 管理备份数据:")
    print("   rust_pyfunc.run_pools(func, args, backup_file='data.bin')")
    print("   data = rust_pyfunc.query_backup('data.bin', storage_format='binary')")
    print("   rust_pyfunc.delete_backup('data.bin', 'binary')")
    
    print("\n3. 启动Web界面:")
    print("   from rust_pyfunc.web_manager import start_web_manager")
    print("   start_web_manager()  # 访问 http://127.0.0.1:5000")
    
    print("\n4. 使用拆分后的类型提示:")
    print("   IDE现在可以提供更精确的代码补全和类型检查")
    print("   各功能模块的文档更加清晰和易于查找")

if __name__ == "__main__":
    main()