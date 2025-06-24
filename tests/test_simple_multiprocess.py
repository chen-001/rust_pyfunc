#!/usr/bin/env python3
"""
简单的多进程测试，不使用design_whatever
"""

import os
import sys
import tempfile
import time

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

def test_simple_multiprocess():
    """测试简单的多进程功能"""
    print("开始简单多进程测试...")
    
    def simple_func(date, code):
        """简单的测试函数"""
        # 模拟一些计算
        result = []
        for i in range(50):  # 返回50个结果
            result.append(float(date + i))
        return result
    
    # 创建测试任务
    test_args = [[20240101 + i, f"TEST{i:06d}"] for i in range(100)]
    
    with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as tmp_file:
        backup_file = tmp_file.name
    
    try:
        import rust_pyfunc
        
        print(f"测试任务数: {len(test_args)}")
        print(f"备份文件: {backup_file}")
        
        start_time = time.time()
        
        # 执行计算
        results = rust_pyfunc.run_pools(
            simple_func,
            test_args,
            backup_file=backup_file,
            num_threads=3,
            backup_batch_size=20
        )
        
        end_time = time.time()
        
        print(f"计算完成!")
        print(f"耗时: {end_time - start_time:.2f}秒")
        print(f"结果数量: {len(results)}")
        
        if len(results) > 0:
            print(f"第一个结果: {results[0]}")
            print(f"结果长度: {len(results[0])}")
        
        # 验证结果
        if len(results) == len(test_args):
            print("✅ 结果数量正确")
            
            # 验证第一个结果
            if len(results[0]) >= 52:  # date, code, + 50个facs
                print("✅ 结果格式正确")
                return True
            else:
                print(f"❌ 结果格式错误，长度: {len(results[0])}")
                return False
        else:
            print(f"❌ 结果数量错误: 期望{len(test_args)}, 实际{len(results)}")
            return False
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        if os.path.exists(backup_file):
            os.unlink(backup_file)

if __name__ == "__main__":
    success = test_simple_multiprocess()
    print(f"\n{'=' * 50}")
    if success:
        print("🎉 简单多进程测试通过!")
        sys.exit(0)
    else:
        print("❌ 简单多进程测试失败!")
        sys.exit(1)