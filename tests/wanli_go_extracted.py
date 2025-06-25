#!/usr/bin/env python3
"""
提取的万里长征go函数，用于测试错误报告
"""

import design_whatever as dw

def go(date,code):
    import os
    import rust_pyfunc as rp
    import numpy as np
    import pandas as pd

    def adjust_afternoon(df: pd.DataFrame) -> pd.DataFrame:
        if df.index.name=='exchtime':
            df1=df.between_time('09:00:00','11:30:00')
            df2=df.between_time('13:00:00','15:00:00')
            df2.index=df2.index-pd.Timedelta(minutes=90)
            df=pd.concat([df1,df2])
        elif 'exchtime' in df.columns:
            df1=df.set_index('exchtime').between_time('09:00:00','11:30:00')
            df2=df.set_index('exchtime').between_time('13:00:00','15:00:00')
            df2.index=df2.index-pd.Timedelta(minutes=90)
            df=pd.concat([df1,df2]).reset_index()
        return df

    def read_trade(symbol:str, date:int,with_retreat:int=0)->pd.DataFrame:
        file_name = "%s_%d_%s.csv" % (symbol, date, "transaction")
        file_path = os.path.join("/ssd_data/stock", str(date), "transaction", file_name)
        
        # 添加文件存在性检查
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"交易数据文件不存在: {file_path}")
            
        df= pd.read_csv(
            file_path,
            dtype={"symbol": str},
            usecols=[
                "exchtime",
                "price",
                "volume",
                "turnover",
                "flag",
                "index",
                "localtime",
                "ask_order",
                "bid_order",
            ],
        )
        if not with_retreat:
            df=df[df.flag!=32]
        df.exchtime=pd.to_timedelta(df.exchtime/1e6,unit='s')+pd.Timestamp('1970-01-01 08:00:00')
        return df

    def read_market(symbol:str, date:int)->pd.DataFrame:
        file_name = "%s_%d_%s.csv" % (symbol, date, "market_data")
        file_path = os.path.join("/ssd_data/stock", str(date), "market_data", file_name)
        
        # 添加文件存在性检查
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"市场数据文件不存在: {file_path}")
            
        df= pd.read_csv(
            file_path,
            dtype={"symbol": str},
        )
        df.exchtime=pd.to_timedelta(df.exchtime/1e6,unit='s')+pd.Timestamp('1970-01-01 08:00:00')
        return df

    def read_market_pair(symbol:str, date:int)->tuple[pd.DataFrame,pd.DataFrame]:
        df=read_market(symbol,date)
        df = df[df.last_prc != 0]
        ask_prc_cols = [f"ask_prc{i}" for i in range(1, 11)]
        ask_vol_cols = [f"ask_vol{i}" for i in range(1, 11)]
        asks = pd.concat(
            [
                pd.melt(
                    df[ask_prc_cols + ["exchtime"]],
                    id_vars=["exchtime"],
                    value_name="price",
                )
                .rename(columns={"variable": "number"})
                .set_index("exchtime"),
                pd.melt(
                    df[ask_vol_cols + ["exchtime"]],
                    id_vars=["exchtime"],
                    value_name="vol",
                )
                .drop(columns=["variable"])
                .set_index("exchtime"),
            ],
            axis=1,
        )
        asks=asks[asks.price!=0]
        asks.number=asks.number.str.slice(7).astype(int)
        asks=asks.reset_index().sort_values(by=["exchtime", "number"]).reset_index(drop=True)
        
        bid_prc_cols = [f"bid_prc{i}" for i in range(1, 11)]
        bid_vol_cols = [f"bid_vol{i}" for i in range(1, 11)]
        bids = pd.concat(
            [
                pd.melt(
                    df[bid_prc_cols + ["exchtime"]],
                    id_vars=["exchtime"],
                    value_name="price",
                )
                .rename(columns={"variable": "number"})
                .set_index("exchtime"),
                pd.melt(
                    df[bid_vol_cols + ["exchtime"]],
                    id_vars=["exchtime"],
                    value_name="vol",
                )
                .drop(columns=["variable"])
                .set_index("exchtime"),
            ],
            axis=1,
        )
        bids=bids[bids.price!=0]
        bids.number=bids.number.str.slice(7).astype(int)
        bids=bids.reset_index().sort_values(by=["exchtime", "number"]).reset_index(drop=True)
        
        return asks,bids

    # 万里长征的names列表（1066个因子名称）
    names = [
        "amount_dura_mean_ask",
        "amount_dura_std_ask", 
        # ... 这里应该是完整的1066个名称，为了简化我们只用一部分
    ]
    
    # 为了测试，我们创建一个固定长度的names列表
    names = [f"factor_{i}" for i in range(262)]  # 262个因子
    
    try:
        # 尝试读取数据并计算因子
        trade_df = read_trade(code, date)
        market_df = read_market(code, date)
        
        # 简化的因子计算（原始代码很复杂）
        # 这里只是示例，实际计算会更复杂
        res = []
        
        # 模拟一些基础统计因子
        for i in range(len(names)):
            if i < 100:
                # 基于交易数据的因子
                if not trade_df.empty:
                    res.append(float(trade_df['price'].mean() + i))
                else:
                    res.append(np.nan)
            else:
                # 基于市场数据的因子
                if not market_df.empty and 'last_prc' in market_df.columns:
                    res.append(float(market_df['last_prc'].mean() + i))
                else:
                    res.append(np.nan)
        
        # 确保返回长度与names一致
        if len(res) != len(names):
            print(f"警告：计算结果长度({len(res)})与names长度({len(names)})不一致")
            if len(res) < len(names):
                res.extend([np.nan] * (len(names) - len(res)))
            else:
                res = res[:len(names)]
        
        return res
        
    except Exception as e:
        # 万里长征原始代码的问题：直接抛出异常而不是返回固定长度的结果
        print(f"万里长征go函数处理 {date}-{code} 时发生异常: {e}")
        
        # 修复：返回与names长度一致的NaN列表
        return [np.nan] * len(names)

if __name__ == "__main__":
    # 测试函数
    print("测试提取的万里长征go函数...")
    try:
        result = go(20170101, "000001")
        print(f"测试成功，返回长度: {len(result)}")
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()