#!/usr/bin/env python3
"""
测试万里长征脚本，检查错误报告
"""

import os
import sys
import tempfile

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'python'))

def test_wanli_error_reporting():
    """测试万里长征脚本的错误报告"""
    print("=== 测试万里长征脚本错误报告 ===")
    
    # 导入万里长征函数
    sys.path.insert(0, '/home/chenzongwei/pythoncode/万里长征')
    
    try:
        from 万里长征放宽参数20 import go as wanli_go
        print("✅ 成功导入万里长征函数")
    except Exception as e:
        print(f"❌ 导入万里长征函数失败: {e}")
        return False
    
    # 少量测试数据
    test_args = [
        [20170101, "000001"],
        [20170101, "000002"],
    ]
    
    with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as tmp_file:
        backup_file = tmp_file.name
    
    try:
        import rust_pyfunc
        
        print("开始万里长征错误测试...")
        results = rust_pyfunc.run_pools(
            wanli_go,
            test_args,
            backup_file=backup_file,
            num_threads=2
        )
        
        print(f"✅ 万里长征测试成功，结果数量: {len(results)}")
        return True
        
    except Exception as e:
        print(f"❌ 万里长征测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        if os.path.exists(backup_file):
            os.unlink(backup_file)

if __name__ == "__main__":
    print("开始测试万里长征错误报告...")
    success = test_wanli_error_reporting()
    print(f"\n测试结果: {'✅ 成功' if success else '❌ 失败'}")
    
    if not success:
        print("\n现在应该能看到详细的错误信息了！")