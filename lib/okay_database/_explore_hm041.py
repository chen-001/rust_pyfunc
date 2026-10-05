"""临时探索脚本：把 hm41 旧代码的 pandas 语义逐步打出来（用完删）。"""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "/home/chenzongwei/pythoncode")
import design_whatever_out as dwo

np.set_printoptions(precision=12)


class Probe(dwo.TickDataBase):
    def get_factor(self):
        df_in = self.read_market_data_single()
        print("read_market_data_single ->", None if df_in is None else df_in.shape)
        asks, bids = self.read_market_data_single_price_vol_pair()
        print("asks", asks.shape, list(asks.columns), asks.dtypes.to_dict())
        print("bids", bids.shape, list(bids.columns))
        print(asks.head(3))
        asks = asks.set_index("exchtime").between_time("09:30", "14:57").reset_index()
        asks = asks[asks.number < 6]
        af = asks.drop_duplicates(subset="price", keep="first").set_index("price")
        al = asks.drop_duplicates(subset="price", keep="last").set_index("price")
        print("af cols", list(af.columns), "n", af.shape)
        print(af.head(3))
        cm = af.corr()
        print("af.corr() cols", list(cm.columns))
        print(cm)
        dura = (al.exchtime - af.exchtime).dt.total_seconds()
        print("dura name", dura.name, "index name", dura.index.name)
        rd = dura.reset_index()
        print("dura.reset_index cols", list(rd.columns))
        print(rd.corr())
        print("asks_first_sum", af.vol.sum(), "asks_last_sum", al.vol.sum())
        # merge 检查
        afl = pd.merge(
            af.rename(columns={"vol": "first_vol"}),
            al.rename(columns={"vol": "last_vol"}),
            on=["price"],
        )
        print("merge cols", list(afl.columns), afl.shape)
        print(afl.head(3))
        return None


if __name__ == "__main__":
    d = int(sys.argv[1]) if len(sys.argv) > 1 else 20250303
    c = sys.argv[2] if len(sys.argv) > 2 else "000001"
    rc = Probe(1, d, d, 1, symbol=c, get_many=0, read_full=0, change_time=1)
    rc.get_factor()
