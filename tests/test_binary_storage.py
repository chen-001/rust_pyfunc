#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import tempfile
import os
sys.path.append('/home/chenzongwei/rust_pyfunc/python')

import rust_pyfunc

def simple_analysis(date, code):
    return [float(date % 100), float(len(code)), 3.14]

def test_binary_storage():
    print("测试二进制存储...")
    with tempfile.NamedTemporaryFile(mode='w', suffix='.bin', delete=False) as f:
        backup_file = f.name
    
    try:
        # 小规模测试
        args = [(20220101, "000001"), (20220102, "000002")]
        
        result = rust_pyfunc.run_pools(
            simple_analysis,
            args,
            backup_file=backup_file,
            storage_format="binary",
            num_threads=1
        )
        
        print(f"✓ 执行完成，结果数量: {len(result)}")
        print(f"结果示例: {result[0] if result else 'None'}")
        
        # 查询备份数据
        backup_data = rust_pyfunc.query_backup(backup_file, storage_format="binary")
        print(f"✓ 备份数据数量: {len(backup_data)}")
        print(f"备份示例: {backup_data[0][:5] if backup_data else 'None'}")  # 只显示前5个元素
        
        # 验证数据完整性
        assert len(backup_data) == len(args), f"备份数据数量不匹配: {len(backup_data)} != {len(args)}"
        assert len(result) == len(args), f"结果数量不匹配: {len(result)} != {len(args)}"
        
        print("🎉 二进制存储测试成功！")
        return True
        
    except Exception as e:
        print(f"❌ 二进制存储测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        if os.path.exists(backup_file):
            os.unlink(backup_file)

if __name__ == "__main__":
    test_binary_storage()