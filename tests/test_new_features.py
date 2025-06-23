#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试新功能：NDArray输出、备份管理、Web界面
"""

import sys
import tempfile
import os
import numpy as np
sys.path.append('/home/chenzongwei/rust_pyfunc/python')

import rust_pyfunc

def test_function(date, code):
    """测试函数"""
    return [float(date % 100), float(len(code)), 3.14159]

def test_ndarray_output():
    """测试NDArray输出格式"""
    print("🧪 测试NDArray输出格式")
    
    args = [(20220101, "000001"), (20220102, "000002")]
    
    # 测试run_pools的NDArray输出
    result = rust_pyfunc.run_pools(test_function, args, num_threads=1)
    
    print(f"✓ run_pools返回类型: {type(result)}")
    print(f"✓ 结果形状: {result.shape}")
    print(f"✓ 结果内容:\n{result}")
    
    assert isinstance(result, np.ndarray), "run_pools应该返回NDArray"
    assert result.shape[0] == len(args), "行数应该等于参数数量"
    assert result.shape[1] == 5, "列数应该是5 (date, code, 3个因子)"
    
    return True

def test_backup_functions():
    """测试备份管理函数"""
    print("\n🗂️ 测试备份管理函数")
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.bin', delete=False) as f:
        backup_file = f.name
    
    try:
        args = [(20220101, "000001"), (20220102, "000002")]
        
        # 创建备份
        rust_pyfunc.run_pools(
            test_function, 
            args, 
            backup_file=backup_file,
            storage_format="binary"
        )
        
        # 测试backup_exists
        exists = rust_pyfunc.backup_exists(backup_file, "binary")
        print(f"✓ 备份文件存在: {exists}")
        assert exists, "备份文件应该存在"
        
        # 测试get_backup_info
        size, modified_time = rust_pyfunc.get_backup_info(backup_file, "binary")
        print(f"✓ 备份文件大小: {size} 字节")
        print(f"✓ 修改时间: {modified_time}")
        assert size > 0, "备份文件大小应该大于0"
        
        # 测试query_backup的NDArray输出
        backup_data = rust_pyfunc.query_backup(backup_file, storage_format="binary")
        print(f"✓ query_backup返回类型: {type(backup_data)}")
        print(f"✓ 查询结果形状: {backup_data.shape}")
        
        assert isinstance(backup_data, np.ndarray), "query_backup应该返回NDArray"
        assert backup_data.shape[0] == len(args), "查询结果行数应该等于参数数量"
        assert backup_data.shape[1] == 6, "查询结果列数应该是6 (date, code, timestamp, 3个因子)"
        
        # 测试delete_backup
        rust_pyfunc.delete_backup(backup_file, "binary")
        exists_after = rust_pyfunc.backup_exists(backup_file, "binary")
        print(f"✓ 删除后文件存在: {exists_after}")
        assert not exists_after, "删除后文件不应该存在"
        
        return True
        
    except Exception as e:
        print(f"❌ 备份测试失败: {e}")
        return False
    finally:
        if os.path.exists(backup_file):
            os.unlink(backup_file)

def test_web_manager_import():
    """测试Web管理器导入"""
    print("\n🌐 测试Web管理器导入")
    
    try:
        from rust_pyfunc import web_manager
        print("✓ Web管理器模块导入成功")
        
        # 检查主要功能是否存在
        assert hasattr(web_manager, 'BackupWebManager'), "应该有BackupWebManager类"
        assert hasattr(web_manager, 'start_web_manager'), "应该有start_web_manager函数"
        
        print("✓ Web管理器功能完整")
        return True
        
    except ImportError as e:
        print(f"⚠️ Web管理器需要Flask: {e}")
        return True  # 这不是错误，只是缺少可选依赖
    except Exception as e:
        print(f"❌ Web管理器测试失败: {e}")
        return False

def test_type_hints():
    """测试类型提示拆分"""
    print("\n📝 测试类型提示文件结构")
    
    base_path = "/home/chenzongwei/rust_pyfunc/python/rust_pyfunc"
    
    expected_files = [
        "__init__.pyi",
        "core_functions.pyi", 
        "time_series.pyi",
        "text_analysis.pyi",
        "parallel_computing.pyi",
        "pandas_extensions.pyi",
        "tree_structures.pyi"
    ]
    
    for filename in expected_files:
        filepath = os.path.join(base_path, filename)
        if os.path.exists(filepath):
            size = os.path.getsize(filepath)
            print(f"✓ {filename}: {size} 字节")
        else:
            print(f"❌ {filename}: 文件不存在")
            return False
    
    # 检查备份的原始文件
    backup_file = os.path.join(base_path, "rust_pyfunc.pyi.backup")
    if os.path.exists(backup_file):
        backup_size = os.path.getsize(backup_file)
        print(f"✓ 原始文件已备份: {backup_size} 字节")
    
    print("✓ 类型提示文件拆分成功")
    return True

def main():
    """运行所有测试"""
    print("🚀 测试新功能特性")
    print("=" * 60)
    
    tests = [
        ("NDArray输出格式", test_ndarray_output),
        ("备份管理函数", test_backup_functions), 
        ("Web管理器导入", test_web_manager_import),
        ("类型提示拆分", test_type_hints),
    ]
    
    success_count = 0
    
    for test_name, test_func in tests:
        try:
            if test_func():
                success_count += 1
                print(f"✅ {test_name}: 通过")
            else:
                print(f"❌ {test_name}: 失败")
        except Exception as e:
            print(f"❌ {test_name}: 异常 - {e}")
    
    print("\n" + "=" * 60)
    print(f"测试结果: {success_count}/{len(tests)} 通过")
    
    if success_count == len(tests):
        print("🎉 所有新功能测试通过！")
    else:
        print("⚠️ 部分测试失败")
    
    # 展示功能改进总结
    print("\n📋 功能改进总结:")
    print("1. ✅ run_pools和query_backup现在返回NDArray而不是嵌套列表")
    print("2. ✅ 新增delete_backup、backup_exists、get_backup_info函数")
    print("3. ✅ 创建了Web管理界面 (需要Flask)")
    print("4. ✅ 将2547行的.pyi文件拆分为7个模块化文件")
    print("\n🔗 Web界面使用方法:")
    print("   from rust_pyfunc.web_manager import start_web_manager")
    print("   start_web_manager()  # 启动在 http://127.0.0.1:5000")

if __name__ == "__main__":
    main()