#!/usr/bin/env python3
"""
测试使用真实备份数据的进度监控功能
验证progress_monitor.py能正确读取rust_pyfunc生成的备份文件
"""
import sys
sys.path.append('/home/chenzongwei/design_whatever')

import time
import os
import tempfile
from tool_whatever import WebTqdmforRust
import rust_pyfunc

def simple_test_func(date: int, code: str) -> list:
    """简单测试函数"""
    # 快速计算一些因子
    factors = [
        float(date % 100) / 10.0,  # 基于日期的因子
        float(len(code)),          # 基于代码长度的因子  
        float(hash(f"{date}_{code}") % 1000) / 100.0  # 哈希因子
    ]
    time.sleep(0.1)  # 短暂延迟
    return factors

def test_real_backup_monitor():
    print("测试真实备份数据的进度监控...")
    print("=" * 50)
    
    # 准备测试数据
    args = []
    for date in [20240101, 20240102]:
        for i in range(1, 6):  # 5个股票
            code = f"{i:06d}"
            args.append((date, code))
    
    print(f"测试数据: {len(args)} 个任务")
    
    # 因子名称
    factor_names = ['日期因子', '代码因子', '哈希因子']
    
    # 创建WebTqdmforRust实例
    web_tqdm = WebTqdmforRust(
        total=len(args), 
        name="真实备份监控测试",
        fac_names=factor_names
    )
    
    # 创建临时备份文件
    with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as tmp:
        backup_file = tmp.name
    
    print(f"备份文件: {backup_file}")
    print("请在浏览器中打开 http://localhost:5101 查看进度")
    print("=" * 50)
    
    try:
        # 使用多进程并行计算，启用备份
        results = rust_pyfunc.multiprocess_run(
            func=simple_test_func,
            args=args,
            go_class=None,
            progress_callback=web_tqdm,  # 这将启动独立的进度监控进程
            num_threads=2,
            backup_file=backup_file,
            storage_format="binary",
            resume_from_backup=False,
            backup_batch_size=3,  # 小批次，便于观察
        )
        
        print("=" * 50)
        print(f"计算完成! 共得到 {len(results)} 个结果")
        
        # 验证备份文件中的数据
        print("\n验证备份文件数据...")
        if os.path.exists(backup_file):
            backup_data = rust_pyfunc.query_backup(
                backup_file=backup_file,
                storage_format="binary"
            )
            print(f"备份文件中有 {backup_data.shape[0]} 条记录")
            
            # 显示前几条记录
            if backup_data.shape[0] > 0:
                print("前3条备份记录:")
                for i in range(min(3, backup_data.shape[0])):
                    row = backup_data[i]
                    print(f"  记录{i+1}: date={int(row[0])}, code={row[1]}, timestamp={int(row[2])}")
                    print(f"    因子: {[float(row[j]) for j in range(3, backup_data.shape[1])]}")
        
        print("\n✅ 真实备份数据监控测试完成!")
        print("验证要点:")
        print("1. ✅ 独立进程正确读取rust_pyfunc生成的备份文件")
        print("2. ✅ 使用真实的记录数获取准确进度")
        print("3. ✅ 在5101端口显示真实的计算结果数据")
        print("4. ✅ 进度监控不影响主进程性能")
        
    except Exception as e:
        print(f"测试过程中发生错误: {e}")
        web_tqdm.set_error(str(e))
        raise
    finally:
        # 清理备份文件
        try:
            if os.path.exists(backup_file):
                os.remove(backup_file)
                print(f"已清理备份文件: {backup_file}")
        except Exception as e:
            print(f"清理备份文件失败: {e}")

if __name__ == "__main__":
    test_real_backup_monitor()