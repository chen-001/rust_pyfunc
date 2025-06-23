#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调试特定情况下的Broken pipe问题
"""

import sys
sys.path.append('/home/chenzongwei/design_whatever')
sys.path.append('/home/chenzongwei/rust_pyfunc/python')

try:
    import design_whatever as dw

    class Go(dw.TickDataBase):
        def get_factor(self):
            # 简化的实现，避免实际数据读取
            # df=self.read_trade_data_single(with_retreat=0)
            res=[float(i) for i in range(150)]
            return [float(i) for i in res]

    print("🔍 测试简化版本...")

    names = ['a'+str(i) for i in range(150)]
    
    # 使用更小的数据集测试
    facs = dw.run_factor(
        Go,
        'pcm115',
        names,
        n_jobs=4,  # 减少进程数
        start_date=20170101,
        end_date=20170102,  # 只测试一天
        look_back_window=1,
        via_parquet=1,
        level2_single_stock=1,
        backup_chunk_size=100,  # 减小chunk大小
        change_time=1,
        recalcu_questdb=0,
        recalcu_file=0,
        producer_consumer_mode=1,
        rust_pool=1  # 使用我们的Rust多进程
    )
    
    print(f"✅ 执行成功，结果: {len(facs) if facs else 0}")

except Exception as e:
    print(f"❌ 发生错误: {e}")
    import traceback
    traceback.print_exc()