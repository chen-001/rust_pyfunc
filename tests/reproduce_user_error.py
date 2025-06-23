#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
重现用户报告的Broken pipe错误
"""

import sys
sys.path.append('/home/chenzongwei/design_whatever')
sys.path.append('/home/chenzongwei/rust_pyfunc/python')

try:
    import design_whatever as dw

    class Go(dw.TickDataBase):
        def get_factor(self):
            # try:
                df=self.read_trade_data_single(with_retreat=0)
                res=[float(i) for i in range(150)]
                        # print({f:i for f,i in zip(names,res)})
                return [float(i) for i in res]
            # except Exception as e:
            #     raise
            #     ...

    print("🔬 尝试重现用户的Broken pipe错误...")

    names = ['a'+str(i) for i in range(150)]
    facs = dw.run_factor(
        Go,
        'pcm115',
        names,
        n_jobs=50,
        start_date=20170101,
        end_date=20170201,
        look_back_window=1,
        via_parquet=1,
        level2_single_stock=1,
        backup_chunk_size=8000,
        change_time=1,
        recalcu_questdb=0,
        recalcu_file=0,
        producer_consumer_mode=1,
        rust_pool=1
    )
    
    print("✅ 执行成功")

except Exception as e:
    print(f"❌ 发生错误: {e}")
    import traceback
    traceback.print_exc()