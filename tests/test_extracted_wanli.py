#!/usr/bin/env python3
"""
测试提取的万里长征go函数
"""

import os
import sys
import tempfile

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'python'))

def test_extracted_wanli():
    """测试提取的万里长征go函数"""
    print("=== 测试提取的万里长征go函数 ===")
    
    # 导入提取的万里长征go函数
    from wanli_go_extracted import go as extracted_wanli_go
    print("✅ 成功导入提取的万里长征go函数")
    
    # 测试数据：包含可能出错的情况
    test_args = [
        [20170101, "000001"],  # 可能文件不存在
        [20170101, "000002"],  # 可能文件不存在  
        [20170101, "999999"],  # 肯定文件不存在
    ]
    
    with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as tmp_file:
        backup_file = tmp_file.name
    
    try:
        import rust_pyfunc
        
        print("开始测试提取的万里长征go函数...")
        print("这应该能显示详细的错误信息...")
        
        results = rust_pyfunc.run_pools(
            extracted_wanli_go,
            test_args,
            backup_file=backup_file,
            num_threads=3
        )
        
        print(f"✅ 测试成功，结果数量: {len(results)}")
        
        # 检查结果
        for i, result in enumerate(results):
            if result:
                print(f"结果{i}: 长度={len(result)}, 前5个值={result[:5] if len(result) >= 5 else result}")
            else:
                print(f"结果{i}: 空结果")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        print("\n🎯 现在可以看到万里长征的详细错误信息了：")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        if os.path.exists(backup_file):
            os.unlink(backup_file)

if __name__ == "__main__":
    print("开始测试提取的万里长征go函数...")
    print("这将测试真实的万里长征逻辑，但避免模块级别的dw.run_factor调用")
    print()
    
    success = test_extracted_wanli()
    print(f"\n测试结果: {'✅ 成功' if success else '❌ 失败'}")
    
    if not success:
        print("\n📋 关键发现：")
        print("1. 现在可以看到万里长征go函数的详细错误信息")
        print("2. 错误报告系统正常工作")
        print("3. 万里长征的问题已经得到诊断")