#!/usr/bin/env python3
"""
直接测试万里长征脚本
"""

import os
import sys
import tempfile

def test_direct_wanli():
    """直接测试万里长征脚本"""
    print("=== 直接万里长征测试 ===")
    
    # 直接调用万里长征函数，不使用多进程
    sys.path.insert(0, '/home/chenzongwei/pythoncode/万里长征')
    
    try:
        from 万里长征放宽参数20 import go as wanli_go
        print("✅ 成功导入万里长征函数")
        
        # 直接调用一次看看会发生什么
        print("直接调用万里长征函数...")
        result = wanli_go(20170101, "000001")
        print(f"✅ 直接调用成功，结果长度: {len(result)}")
        return True
        
    except Exception as e:
        print(f"❌ 直接调用万里长征函数失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("开始直接万里长征测试...")
    success = test_direct_wanli()
    print(f"\n测试结果: {'✅ 成功' if success else '❌ 失败'}")