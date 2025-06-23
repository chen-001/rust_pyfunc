#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
详细测试resume_from_backup功能
"""

import sys
import tempfile
import os
import time
sys.path.append('/home/chenzongwei/rust_pyfunc/python')

import rust_pyfunc


def create_test_function():
    """创建一个测试函数，每次调用时记录"""
    call_log = []
    
    def test_func(date, code):
        # 记录函数调用
        call_log.append((date, code))
        # 模拟一些计算时间
        time.sleep(0.001)
        # 返回基于输入的唯一结果
        return [float(date % 1000), float(len(code)), hash(f"{date}_{code}") % 100]
    
    test_func.call_log = call_log
    return test_func


def test_resume_from_backup_basic():
    """测试基本的备份恢复功能"""
    print("=== 测试基本备份恢复功能 ===")
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        backup_file = f.name
    
    try:
        # 创建完整的测试数据集
        full_args = [
            [20220101, "000001"],
            [20220101, "000002"], 
            [20220101, "000003"],
            [20220102, "000001"],
            [20220102, "000002"],
            [20220102, "000003"],
        ]
        
        print(f"完整任务集: {len(full_args)} 个任务")
        
        # 第一次运行：只执行部分任务
        partial_args = full_args[:3]  # 只执行前3个
        test_func1 = create_test_function()
        
        print(f"\n第一次运行：执行 {len(partial_args)} 个任务")
        result1 = rust_pyfunc.run_pools(
            test_func1,
            partial_args,
            backup_file=backup_file,
            backup_batch_size=2,
            storage_format="json",
            num_threads=2
        )
        
        print(f"第一次运行完成：")
        print(f"  - 结果数量: {len(result1)}")
        print(f"  - 函数调用次数: {len(test_func1.call_log)}")
        print(f"  - 调用记录: {test_func1.call_log}")
        
        # 验证备份文件存在
        assert os.path.exists(backup_file), "备份文件应该存在"
        
        # 查询备份数据
        backup_data = rust_pyfunc.query_backup(backup_file, storage_format="json")
        print(f"  - 备份数据数量: {len(backup_data)}")
        
        # 第二次运行：使用resume_from_backup执行完整任务集
        test_func2 = create_test_function()
        
        print(f"\n第二次运行：使用resume_from_backup执行完整任务集")
        result2 = rust_pyfunc.run_pools(
            test_func2,
            full_args,
            backup_file=backup_file,
            resume_from_backup=True,
            storage_format="json",
            num_threads=2
        )
        
        print(f"第二次运行完成：")
        print(f"  - 结果数量: {len(result2)}")
        print(f"  - 函数调用次数: {len(test_func2.call_log)}")
        print(f"  - 调用记录: {test_func2.call_log}")
        
        # 验证关键点
        expected_new_calls = len(full_args) - len(partial_args)
        assert len(test_func2.call_log) == expected_new_calls, f"应该只调用 {expected_new_calls} 次新任务，实际调用了 {len(test_func2.call_log)} 次"
        
        # 验证只调用了未备份的任务
        expected_new_tasks = set((date, code) for date, code in full_args[3:])
        actual_new_tasks = set(test_func2.call_log)
        assert actual_new_tasks == expected_new_tasks, f"新任务不匹配：期望 {expected_new_tasks}，实际 {actual_new_tasks}"
        
        # 验证结果数量正确
        assert len(result2) == len(full_args), f"结果数量应该是 {len(full_args)}，实际是 {len(result2)}"
        
        print("✓ 基本备份恢复功能正常")
        
    finally:
        if os.path.exists(backup_file):
            os.unlink(backup_file)


def test_resume_empty_backup():
    """测试空备份文件的恢复"""
    print("\n=== 测试空备份文件恢复 ===")
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        backup_file = f.name
    
    try:
        # 删除空文件，模拟没有备份
        os.unlink(backup_file)
        
        args = [
            [20220101, "000001"],
            [20220101, "000002"],
        ]
        
        test_func = create_test_function()
        
        result = rust_pyfunc.run_pools(
            test_func,
            args,
            backup_file=backup_file,
            resume_from_backup=True,  # 即使没有备份文件也应该正常工作
            storage_format="json",
            num_threads=2
        )
        
        print(f"空备份恢复测试完成：")
        print(f"  - 结果数量: {len(result)}")
        print(f"  - 函数调用次数: {len(test_func.call_log)}")
        
        # 应该执行所有任务
        assert len(test_func.call_log) == len(args), "应该执行所有任务"
        assert len(result) == len(args), "结果数量应该正确"
        
        print("✓ 空备份文件恢复功能正常")
        
    finally:
        if os.path.exists(backup_file):
            os.unlink(backup_file)


def test_resume_complete_backup():
    """测试完全备份的恢复（不应该执行任何新任务）"""
    print("\n=== 测试完全备份恢复 ===")
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        backup_file = f.name
    
    try:
        args = [
            [20220101, "000001"],
            [20220101, "000002"],
        ]
        
        # 第一次运行：创建完整备份
        test_func1 = create_test_function()
        result1 = rust_pyfunc.run_pools(
            test_func1,
            args,
            backup_file=backup_file,
            storage_format="json",
            num_threads=2
        )
        
        print(f"第一次运行创建完整备份：")
        print(f"  - 函数调用次数: {len(test_func1.call_log)}")
        
        # 第二次运行：恢复相同的任务
        test_func2 = create_test_function()
        result2 = rust_pyfunc.run_pools(
            test_func2,
            args,  # 完全相同的任务
            backup_file=backup_file,
            resume_from_backup=True,
            storage_format="json",
            num_threads=2
        )
        
        print(f"第二次运行（完全恢复）：")
        print(f"  - 结果数量: {len(result2)}")
        print(f"  - 函数调用次数: {len(test_func2.call_log)}")
        
        # 不应该调用任何新函数
        assert len(test_func2.call_log) == 0, f"不应该调用任何新函数，实际调用了 {len(test_func2.call_log)} 次"
        assert len(result2) == len(args), "结果数量应该正确"
        
        print("✓ 完全备份恢复功能正常")
        
    finally:
        if os.path.exists(backup_file):
            os.unlink(backup_file)


def test_resume_different_storage_formats():
    """测试不同存储格式的备份恢复"""
    print("\n=== 测试不同存储格式 ===")
    
    args = [
        [20220101, "000001"],
        [20220101, "000002"],
        [20220102, "000001"],
    ]
    
    formats = ["json", "binary", "memory_map"]
    
    for storage_format in formats:
        print(f"\n测试 {storage_format} 格式：")
        
        suffix = ".json" if storage_format == "json" else ".bin"
        with tempfile.NamedTemporaryFile(mode='w', suffix=suffix, delete=False) as f:
            backup_file = f.name
        
        try:
            # 第一次运行：部分任务
            partial_args = args[:2]
            test_func1 = create_test_function()
            
            result1 = rust_pyfunc.run_pools(
                test_func1,
                partial_args,
                backup_file=backup_file,
                storage_format=storage_format,
                num_threads=2
            )
            
            # 第二次运行：恢复并完成
            test_func2 = create_test_function()
            
            result2 = rust_pyfunc.run_pools(
                test_func2,
                args,
                backup_file=backup_file,
                resume_from_backup=True,
                storage_format=storage_format,
                num_threads=2
            )
            
            print(f"  - 第一次调用: {len(test_func1.call_log)} 次")
            print(f"  - 第二次调用: {len(test_func2.call_log)} 次")
            print(f"  - 最终结果: {len(result2)} 个")
            
            expected_new_calls = len(args) - len(partial_args)
            assert len(test_func2.call_log) == expected_new_calls, f"{storage_format}格式：期望 {expected_new_calls} 次新调用，实际 {len(test_func2.call_log)} 次"
            assert len(result2) == len(args), f"{storage_format}格式：结果数量不正确"
            
            print(f"  ✓ {storage_format} 格式正常")
            
        finally:
            if os.path.exists(backup_file):
                os.unlink(backup_file)


if __name__ == "__main__":
    print("开始详细测试resume_from_backup功能")
    print("=" * 60)
    
    try:
        test_resume_from_backup_basic()
        test_resume_empty_backup()
        test_resume_complete_backup()
        test_resume_different_storage_formats()
        
        print("\n" + "=" * 60)
        print("🎉 所有resume_from_backup测试通过！")
        print("\nresume_from_backup功能工作原理：")
        print("1. 📋 读取备份文件中已存在的(date, code)组合")
        print("2. 🔍 从输入参数中过滤掉已计算的任务")
        print("3. ⚡ 只计算剩余的新任务")
        print("4. 🔗 将现有结果与新结果合并输出")
        print("5. ✅ 完全符合用户期望的功能！")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)