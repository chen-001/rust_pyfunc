#!/usr/bin/env python3
"""
最终诊断测试 - 确认pure_ocean_breeze导入问题
"""

import os
import sys
import tempfile

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

def test_pure_ocean_breeze_data_read():
    """确认pure_ocean_breeze.jason.data.read_data导入问题"""
    print("=== 确认pure_ocean_breeze.jason.data.read_data导入问题 ===")
    
    def problematic_func(date, code):
        # 这与万里长征文件中的导入完全一致
        import pure_ocean_breeze.jason.data.read_data as p
        return [1.0, 2.0, 3.0]
    
    test_args = [[20240101, "TEST001"]]
    
    with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as tmp_file:
        backup_file = tmp_file.name
    
    try:
        import rust_pyfunc
        
        print("开始测试pure_ocean_breeze.jason.data.read_data导入...")
        results = rust_pyfunc.run_pools(
            problematic_func,
            test_args,
            backup_file=backup_file,
            num_threads=1
        )
        
        print(f"✅ 测试通过，结果数量: {len(results)}")
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        if os.path.exists(backup_file):
            os.unlink(backup_file)

if __name__ == "__main__":
    print("开始最终诊断测试...")
    success = test_pure_ocean_breeze_data_read()
    
    if not success:
        print("\n🔍 问题确认：pure_ocean_breeze.jason.data.read_data 导入会导致子进程卡死")
        print("\n💡 解决方案建议：")
        print("1. 在主进程中预先读取数据，传递给rust_pyfunc")
        print("2. 或者使用不同的数据读取方式")
        print("3. 或者在子进程启动前设置特殊的环境变量")
    else:
        print("\n✅ 测试通过，可能是其他问题")