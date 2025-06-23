#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试并行池功能
"""

import time
import os
import tempfile
import sys
import random
sys.path.append('/home/chenzongwei/rust_pyfunc/python')

import rust_pyfunc


def simple_analysis(date, code):
    """简单的分析函数"""
    # 模拟一些计算
    time.sleep(0.001)  # 1ms的模拟计算时间
    
    # 返回固定长度的因子列表
    return [
        float(date % 1000),  # 第一个因子：日期的后三位
        float(len(code)),    # 第二个因子：代码长度
        random.random()      # 第三个因子：随机数
    ]


def complex_analysis(date, code):
    """复杂一些的分析函数"""
    # 模拟更复杂的计算
    time.sleep(0.005)  # 5ms的模拟计算时间
    
    # 基于日期和代码的一些计算
    base_value = hash(f"{date}_{code}") % 10000
    
    return [
        float(base_value),
        float(base_value ** 0.5),
        float(base_value / 100.0),
        float(date + int(code) if code.isdigit() else date),
        random.uniform(-1, 1)
    ]


def progress_callback(completed, total, elapsed_time, speed):
    """进度回调函数"""
    percent = completed / total * 100
    print(f"  进度回调: {percent:.1f}% ({completed}/{total}), 速度: {speed:.0f} 任务/秒, 已用时: {elapsed_time:.1f}秒")


def test_basic_functionality():
    """测试基本功能"""
    print("=== 测试基本功能 ===")
    
    # 创建测试数据
    args = [
        (20220101, "000001"),
        (20220101, "000002"),
        (20220102, "000001"),
        (20220102, "000002"),
        (20220103, "000001"),
    ]
    
    print(f"测试数据: {args}")
    
    # 测试不带备份的并行执行
    print("\n1. 测试不带备份的并行执行:")
    start_time = time.time()
    result = rust_pyfunc.run_pools(
        simple_analysis,
        args,
        num_threads=2
    )
    elapsed = time.time() - start_time
    
    print(f"结果数量: {len(result)}")
    print(f"第一个结果: {result[0]}")
    print(f"执行时间: {elapsed:.3f}秒")
    
    # 验证结果格式
    assert len(result) == len(args), "结果数量应该等于输入数量"
    assert len(result[0]) == 5, "每个结果应该有5列 (date, code, fac1, fac2, fac3)"
    assert result[0][0] == 20220101, "第一列应该是日期"
    assert result[0][1] == "000001", "第二列应该是代码"
    print("✓ 基本功能测试通过")


def test_backup_functionality():
    """测试备份功能"""
    print("\n=== 测试备份功能 ===")
    
    # 创建临时文件
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        backup_file = f.name
    
    try:
        # 创建测试数据
        args = [
            (20220101, "000001"),
            (20220101, "000002"),
            (20220101, "000003"),
            (20220102, "000001"),
            (20220102, "000002"),
        ]
        
        print(f"备份文件: {backup_file}")
        
        # 测试带备份的并行执行
        print("\n1. 测试带备份的并行执行:")
        result1 = rust_pyfunc.run_pools(
            complex_analysis,
            args,
            backup_file=backup_file,
            backup_batch_size=2,
            storage_format="json",
            num_threads=2
        )
        
        print(f"第一次执行结果数量: {len(result1)}")
        
        # 验证备份文件存在
        assert os.path.exists(backup_file), "备份文件应该存在"
        print("✓ 备份文件创建成功")
        
        # 测试查询备份数据
        print("\n2. 测试查询备份数据:")
        backup_data = rust_pyfunc.query_backup(backup_file, storage_format="json")
        print(f"备份数据数量: {len(backup_data)}")
        print(f"备份数据样例: {backup_data[0]}")
        
        # 验证备份数据格式 (包含timestamp)
        assert len(backup_data) == len(args), "备份数据数量应该等于原始数据数量"
        assert len(backup_data[0]) == 8, "备份数据应该有8列 (date, code, timestamp, fac1-fac5)"
        print("✓ 备份数据查询成功")
        
        # 测试从备份恢复
        print("\n3. 测试从备份恢复:")
        result2 = rust_pyfunc.run_pools(
            complex_analysis,
            args,
            backup_file=backup_file,
            resume_from_backup=True,
            storage_format="json",
            num_threads=2
        )
        
        print(f"从备份恢复的结果数量: {len(result2)}")
        
        # 验证结果一致性 (去除timestamp列后比较)
        for i in range(len(result1)):
            for j in range(len(result1[i])):
                if isinstance(result1[i][j], float) and isinstance(result2[i][j], float):
                    assert abs(result1[i][j] - result2[i][j]) < 1e-10, f"恢复的结果应该与原始结果一致: {result1[i][j]} vs {result2[i][j]}"
                else:
                    assert result1[i][j] == result2[i][j], f"恢复的结果应该与原始结果一致: {result1[i][j]} vs {result2[i][j]}"
        
        print("✓ 从备份恢复测试通过")
        
    finally:
        # 清理临时文件
        if os.path.exists(backup_file):
            os.unlink(backup_file)


def test_progress_callback():
    """测试进度回调功能"""
    print("\n=== 测试进度回调功能 ===")
    
    # 创建测试数据
    args = [(20220101, f"{i:06d}") for i in range(1, 21)]  # 20个任务
    
    print(f"测试数据数量: {len(args)}")
    
    # 测试带进度回调的执行
    print("\n执行中，观察进度回调:")
    result = rust_pyfunc.run_pools(
        simple_analysis,
        args,
        num_threads=4,
        progress_callback=progress_callback
    )
    
    print(f"最终结果数量: {len(result)}")
    assert len(result) == len(args), "结果数量应该等于输入数量"
    print("✓ 进度回调测试通过")


def test_large_dataset():
    """测试大数据集"""
    print("\n=== 测试大数据集 ===")
    
    # 创建较大的测试数据集
    args = []
    for date in range(20220101, 20220111):  # 10天
        for code_num in range(1, 101):  # 每天100只股票
            args.append((date, f"{code_num:06d}"))
    
    print(f"大数据集大小: {len(args)} 个任务")
    
    # 创建临时备份文件
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        backup_file = f.name
    
    try:
        print("\n执行大数据集并行计算:")
        start_time = time.time()
        
        result = rust_pyfunc.run_pools(
            simple_analysis,
            args,
            num_threads=8,
            backup_file=backup_file,
            backup_batch_size=100,
            storage_format="json",
            progress_callback=progress_callback
        )
        
        elapsed = time.time() - start_time
        speed = len(args) / elapsed
        
        print(f"\n大数据集执行完成:")
        print(f"  任务数量: {len(args)}")
        print(f"  结果数量: {len(result)}")
        print(f"  执行时间: {elapsed:.2f}秒")
        print(f"  平均速度: {speed:.0f} 任务/秒")
        
        assert len(result) == len(args), "结果数量应该等于输入数量"
        print("✓ 大数据集测试通过")
        
    finally:
        # 清理临时文件
        if os.path.exists(backup_file):
            os.unlink(backup_file)


def test_storage_formats():
    """测试不同存储格式"""
    print("\n=== 测试存储格式 ===")
    
    args = [
        (20220101, "000001"),
        (20220101, "000002"),
        (20220102, "000001"),
    ]
    
    formats = [
        ("sqlite", ".db"),
        ("memory_map", ".bin"),
        ("parquet", ".parquet")
    ]
    
    for storage_format, suffix in formats:
        print(f"\n测试 {storage_format} 格式:")
        
        # 创建临时文件
        with tempfile.NamedTemporaryFile(mode='w', suffix=suffix, delete=False) as f:
            backup_file = f.name
        
        try:
            print(f"  备份文件: {backup_file}")
            
            # 使用指定格式进行备份
            result = rust_pyfunc.run_pools(
                simple_analysis,
                args,
                backup_file=backup_file,
                storage_format=storage_format,
                num_threads=2
            )
            
            print(f"  存储结果数量: {len(result)}")
            
            # 验证备份文件存在
            assert os.path.exists(backup_file), f"{storage_format}备份文件应该存在"
            
            # 查询备份数据
            backup_data = rust_pyfunc.query_backup(backup_file, storage_format=storage_format)
            print(f"  备份数据数量: {len(backup_data)}")
            
            assert len(backup_data) == len(args), f"{storage_format}备份数据数量应该正确"
            print(f"✓ {storage_format}存储格式测试通过")
            
        finally:
            # 清理临时文件
            if os.path.exists(backup_file):
                os.unlink(backup_file)


def main():
    """主测试函数"""
    print("开始测试rust_pyfunc并行池功能")
    print("=" * 50)
    
    try:
        test_basic_functionality()
        test_backup_functionality()
        test_progress_callback()
        test_storage_formats()
        test_large_dataset()
        
        print("\n" + "=" * 50)
        print("🎉 所有测试通过！")
        print("rust_pyfunc并行池功能工作正常")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()