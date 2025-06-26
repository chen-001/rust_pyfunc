#!/usr/bin/env python3
"""
测试备份文件修复效果
"""
import os
import sys
import numpy as np

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'python'))

def test_backup_file_reading():
    """测试备份文件读取功能"""
    print("🔍 测试备份文件读取修复效果...")
    
    # 测试文件路径
    backup_file = "/home/chenzongwei/pythoncode/万里长征/backup_pool_万里长征放宽参数20.bin"
    
    if not os.path.exists(backup_file):
        print(f"❌ 备份文件不存在: {backup_file}")
        return False
    
    try:
        import rust_pyfunc
        print(f"✅ 成功导入 rust_pyfunc")
        
        # 获取文件大小
        file_size = os.path.getsize(backup_file)
        print(f"📁 备份文件大小: {file_size:,} 字节")
        
        # 尝试读取备份数据
        print("🔄 正在读取备份数据...")
        backup_array = rust_pyfunc.query_backup(
            backup_file=backup_file,
            date_range=None,
            codes=None,
            storage_format="binary"
        )
        
        if backup_array.size == 0:
            print("⚠️ 备份文件为空或无法解析")
            return False
        
        # 分析读取的数据
        num_rows, num_cols = backup_array.shape
        print(f"✅ 成功读取备份数据: {num_rows:,} 行 × {num_cols} 列")
        
        # 分析日期范围
        if num_rows > 0:
            dates = backup_array[:, 0].astype(int)
            min_date, max_date = dates.min(), dates.max()
            print(f"📅 数据日期范围: {min_date} - {max_date}")
            
            # 检查数据连续性
            unique_dates = len(np.unique(dates))
            print(f"📊 包含 {unique_dates:,} 个不同日期的数据")
            
            # 检查最新数据时间
            if num_cols >= 3:  # 有timestamp列
                timestamps = backup_array[:, 2].astype(int)
                latest_timestamp = timestamps.max()
                if latest_timestamp > 0:
                    import datetime
                    latest_time = datetime.datetime.fromtimestamp(latest_timestamp)
                    print(f"⏰ 最新数据计算时间: {latest_time}")
            
            # 采样显示一些数据
            print(f"📝 数据样本 (前5行):")
            for i in range(min(5, num_rows)):
                row = backup_array[i]
                date = int(row[0])
                code = str(row[1])
                timestamp = int(row[2]) if num_cols >= 3 else 0
                factors = row[3:] if num_cols > 3 else []
                print(f"   行{i+1}: 日期={date}, 代码={code}, 时间戳={timestamp}, 因子数={len(factors)}")
        
        return True
        
    except Exception as e:
        print(f"❌ 读取备份文件失败: {e}")
        import traceback
        print(f"🔍 详细错误信息:")
        traceback.print_exc()
        return False

def test_progress_estimation():
    """测试进度估算功能"""
    print("\n🔍 测试进度估算修复效果...")
    
    backup_file = "/home/chenzongwei/pythoncode/万里长征/backup_pool_万里长征放宽参数20.bin"
    
    if not os.path.exists(backup_file):
        print(f"❌ 备份文件不存在: {backup_file}")
        return False
    
    try:
        # 模拟进度监控器的估算逻辑
        from progress_monitor import BackupFileMonitor
        
        # 创建监控器实例 (使用较小的总任务数进行测试)
        monitor = BackupFileMonitor(
            backup_file=backup_file,
            total_tasks=1000000,  # 假设100万个任务
            task_name="测试任务"
        )
        
        # 测试进度估算
        print("🔄 正在估算进度...")
        completed = monitor.estimate_progress_from_backup()
        
        print(f"✅ 估算完成任务数: {completed:,}")
        
        # 测试文件大小估算
        print("🔄 正在测试智能文件大小估算...")
        file_size_estimate = monitor.smart_estimate_from_file_size()
        print(f"📊 基于文件大小估算: {file_size_estimate:,}")
        
        return True
        
    except Exception as e:
        print(f"❌ 进度估算失败: {e}")
        import traceback
        print(f"🔍 详细错误信息:")
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("🚀 开始测试备份文件修复效果\n")
    
    # 测试1: 备份文件读取
    success1 = test_backup_file_reading()
    
    # 测试2: 进度估算
    success2 = test_progress_estimation()
    
    # 总结
    print(f"\n📋 测试结果总结:")
    print(f"   备份文件读取: {'✅ 成功' if success1 else '❌ 失败'}")
    print(f"   进度估算功能: {'✅ 成功' if success2 else '❌ 失败'}")
    
    if success1 and success2:
        print(f"\n🎉 所有测试通过！备份文件修复成功！")
        return True
    else:
        print(f"\n⚠️ 部分测试失败，需要进一步调试")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)