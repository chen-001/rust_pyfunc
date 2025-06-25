#!/usr/bin/env python3
"""
简化的万里长征错误测试
"""

import os
import sys
import tempfile

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'python'))

def test_simple_wanli_error():
    """简化的万里长征错误测试"""
    print("=== 简化万里长征错误测试 ===")
    
    def simplified_wanli_go(date, code):
        """简化的万里长征函数，模拟可能的错误情况"""
        import numpy as np
        import pandas as pd
        import os
        
        try:
            # 模拟文件读取失败（万里长征中常见的问题）
            if code == "000001":
                # 模拟读取不存在的文件
                file_path = f"/ssd_data/stock/{date}/transaction/{code}_{date}_transaction.csv"
                if not os.path.exists(file_path):
                    raise FileNotFoundError(f"文件不存在: {file_path}")
                    
            # 模拟数据处理可能产生不同长度的结果
            if code == "000002":
                # 模拟数据处理异常，返回不完整结果
                return [1.0, 2.0]  # 只返回2个值而不是预期的263个
            
            # 正常情况，返回预期长度的结果
            return [float(i) for i in range(263)]
            
        except Exception as e:
            print(f"处理 {date}-{code} 时发生异常: {e}")
            # 万里长征原始代码中的错误处理方式
            raise e  # 直接抛出异常，而不是返回固定长度的NaN列表
    
    # 测试数据：包含会出错的情况
    test_args = [
        [20170101, "000001"],  # 文件不存在
        [20170101, "000002"],  # 返回长度不正确
        [20170101, "000003"],  # 正常情况
    ]
    
    with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as tmp_file:
        backup_file = tmp_file.name
    
    try:
        import rust_pyfunc
        
        print("开始简化万里长征错误测试...")
        results = rust_pyfunc.run_pools(
            simplified_wanli_go,
            test_args,
            backup_file=backup_file,
            num_threads=3
        )
        
        print(f"✅ 简化万里长征测试成功，结果数量: {len(results)}")
        
        # 检查结果长度
        lengths = [len(result) for result in results]
        print(f"结果长度: {lengths}")
        
        return True
        
    except Exception as e:
        print(f"❌ 简化万里长征测试失败: {e}")
        print("\n现在可以看到详细的错误信息了！")
        return False
        
    finally:
        if os.path.exists(backup_file):
            os.unlink(backup_file)

if __name__ == "__main__":
    print("开始简化万里长征错误测试...")
    success = test_simple_wanli_error()
    print(f"\n测试结果: {'✅ 成功' if success else '❌ 失败'}")