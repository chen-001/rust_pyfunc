#!/usr/bin/env python3
"""
简单的design_whatever测试
"""

import os
import sys
import tempfile

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

try:
    import design_whatever as dw
    
    class SimpleGo(dw.TickDataBase):
        def get_factor(self):
            # 简化的计算，不读取大数据
            return [1.0, 2.0, 3.0]
    
    print("开始简单design_whatever测试...")
    
    # 创建少量测试数据
    test_args = [[20240101, "000001"], [20240102, "000002"]]
    
    with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as tmp_file:
        backup_file = tmp_file.name
    
    try:
        import rust_pyfunc
        
        print(f"测试任务数: {len(test_args)}")
        
        # 直接调用run_pools而不是dw.run_factor
        results = rust_pyfunc.run_pools(
            lambda go_instance, date, code: go_instance.get_factor(),
            test_args,
            go_class=SimpleGo(),
            backup_file=backup_file,
            num_threads=2,
            backup_batch_size=1
        )
        
        print(f"计算完成!")
        print(f"结果数量: {len(results)}")
        if len(results) > 0:
            print(f"第一个结果: {results[0]}")
        
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        if os.path.exists(backup_file):
            os.unlink(backup_file)

except ImportError as e:
    print(f"无法导入design_whatever: {e}")
    print("跳过design_whatever测试")