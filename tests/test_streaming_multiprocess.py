#!/usr/bin/env python3
"""
测试流式多进程处理功能
"""

import os
import sys
import tempfile
import time
import shutil

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

def test_streaming_multiprocess():
    """测试流式多进程处理"""
    print("开始测试流式多进程处理...")
    
    # 测试数据 - 需要转换为list格式，因为rust_pyfunc期望PyList
    test_args = [[20240101 + i, f"00000{i%10}"] for i in range(100)]  # 100个测试任务
    
    def simple_calculate(date, code):
        """简单的计算函数"""
        # 模拟计算过程
        time.sleep(0.001)  # 1ms的计算时间
        return [float(date), float(len(code)), float(date % 1000)]
    
    # 创建临时备份文件
    with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as tmp_file:
        backup_file = tmp_file.name
    
    try:
        import rust_pyfunc
        
        # 执行流式多进程计算
        print(f"执行{len(test_args)}个任务的流式多进程计算...")
        start_time = time.time()
        
        results = rust_pyfunc.run_pools(
            simple_calculate,
            test_args,
            backup_file=backup_file,
            num_threads=2,
            backup_batch_size=20  # 流式处理：小批次
        )
        
        end_time = time.time()
        print(f"计算完成，耗时: {end_time - start_time:.2f}秒")
        print(f"结果数量: {len(results)}")
        
        # 验证结果
        if len(results) == len(test_args):
            print("✅ 结果数量正确")
        else:
            print(f"❌ 结果数量错误: 期望{len(test_args)}, 实际{len(results)}")
            return False
            
        # 验证备份文件是否存在
        if os.path.exists(backup_file):
            backup_size = os.path.getsize(backup_file)
            print(f"✅ 备份文件存在，大小: {backup_size} bytes")
        else:
            print("❌ 备份文件不存在")
            return False
            
        # 验证部分结果的正确性
        if len(results) > 0:
            print(f"第一个结果: {results[0]}, 长度: {len(results[0])}, 类型: {type(results[0])}")
            
        for i, (result, arg) in enumerate(zip(results[:5], test_args[:5])):
            date, code = arg[0], arg[1]
            expected = [float(date), float(len(code)), float(date % 1000)]
            
            print(f"结果{i}: {result}, 长度: {len(result)}")
            
            # 检查结果格式
            if len(result) >= 5:  # date, code, fac1, fac2, fac3
                # 结果格式：result[0]=date, result[1]=code, result[2:]=facs
                actual_facs = [float(result[j]) for j in range(2, len(result))]
                if (abs(actual_facs[0] - expected[0]) < 0.001 and 
                    abs(actual_facs[1] - expected[1]) < 0.001 and 
                    abs(actual_facs[2] - expected[2]) < 0.001):
                    print(f"✅ 结果{i}正确: {actual_facs}")
                else:
                    print(f"❌ 结果{i}错误: 期望{expected}, 实际{actual_facs}")
                    return False
            else:
                print(f"❌ 结果{i}格式错误: 长度{len(result)}, 期望至少5")
                return False
        
        print("✅ 流式多进程处理测试通过")
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # 清理临时文件
        if os.path.exists(backup_file):
            os.unlink(backup_file)


def test_memory_usage():
    """测试内存使用情况"""
    print("\n开始测试内存使用...")
    
    try:
        import psutil
        import gc
        
        # 获取当前进程
        process = psutil.Process()
        
        # 记录初始内存
        gc.collect()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        print(f"初始内存使用: {initial_memory:.1f} MB")
        
        # 执行计算
        test_args = [[20240101 + i, f"stock_{i}"] for i in range(1000)]  # 1000个任务
        
        def memory_test_func(date, code):
            # 创建一些数据模拟计算
            return [float(date + i) for i in range(50)]  # 返回50个浮点数
        
        with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as tmp_file:
            backup_file = tmp_file.name
        
        try:
            import rust_pyfunc
            
            results = rust_pyfunc.run_pools(
                memory_test_func,
                test_args,
                backup_file=backup_file,
                num_threads=2,
                backup_batch_size=50  # 流式处理：小批次
            )
            
            # 记录计算后内存
            gc.collect()
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = final_memory - initial_memory
            
            print(f"计算后内存使用: {final_memory:.1f} MB")
            print(f"内存增长: {memory_increase:.1f} MB")
            
            # 验证内存增长是否合理（流式处理应该内存增长很小）
            if memory_increase < 50:  # 小于50MB增长
                print("✅ 内存使用正常（流式处理）")
                return True
            else:
                print(f"⚠️ 内存增长较大: {memory_increase:.1f} MB")
                return False
                
        finally:
            if os.path.exists(backup_file):
                os.unlink(backup_file)
                
    except ImportError:
        print("⚠️ 无法导入psutil，跳过内存测试")
        return True
    except Exception as e:
        print(f"❌ 内存测试失败: {e}")
        return False


if __name__ == "__main__":
    print("=" * 50)
    print("流式多进程处理测试")
    print("=" * 50)
    
    success = True
    
    # 基本功能测试
    success &= test_streaming_multiprocess()
    
    # 内存使用测试
    success &= test_memory_usage()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 所有测试通过！")
        sys.exit(0)
    else:
        print("❌ 部分测试失败")
        sys.exit(1)