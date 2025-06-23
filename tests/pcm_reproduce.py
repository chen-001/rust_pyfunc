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