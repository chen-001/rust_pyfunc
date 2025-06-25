#!/usr/bin/env python3
"""
测试修改后的万里长征go函数
"""

import os
import sys
import tempfile

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'python'))
sys.path.insert(0, '/home/chenzongwei/pythoncode/万里长征')

def test_fixed_wanli_go():
    """测试修改后的万里长征go函数"""
    print("=== 测试修改后的万里长征go函数 ===")
    
    try:
        # 现在应该可以正常导入了
        from 万里长征放宽参数20 import go as wanli_go
        print("✅ 成功导入万里长征go函数")
        
        # 测试数据：包含可能出错的情况
        test_args = [
            [20170101, "000001"],  # 可能数据文件不存在
            [20170101, "000002"],  # 可能数据文件不存在  
        ]
        
        with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as tmp_file:
            backup_file = tmp_file.name
        
        try:
            import rust_pyfunc
            
            print("开始测试修改后的万里长征go函数...")
            print("注意观察终端输出的错误信息！")
            
            results = rust_pyfunc.run_pools(
                wanli_go,
                test_args,
                backup_file=backup_file,
                num_threads=2
            )
            
            print(f"✅ 测试成功，结果数量: {len(results)}")
            
            # 检查结果
            for i, result in enumerate(results):
                if result and len(result) > 2:
                    values = result[2:]  # 跳过date和code
                    print(f"结果{i}: 总长度={len(result)}, 值长度={len(values)}")
                else:
                    print(f"结果{i}: 长度={len(result) if result else 0}")
            
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
        
    except Exception as e:
        print(f"❌ 导入万里长征函数失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("开始测试修改后的万里长征go函数...")
    print("万里长征脚本已经修改，dw.run_factor现在在if __name__ == '__main__'中")
    print("错误信息将显示在这个终端窗口中")
    print()
    
    success = test_fixed_wanli_go()
    print(f"\n测试结果: {'✅ 成功' if success else '❌ 失败'}")
    
    if not success:
        print("\n📋 查看错误信息的位置：")
        print("1. 终端输出（主要）- 直接在这个窗口中显示")
        print("2. 包含详细的Python异常堆栈跟踪")
        print("3. 工作进程状态诊断信息")
        print("4. 现在应该能看到万里长征函数的具体错误了！")