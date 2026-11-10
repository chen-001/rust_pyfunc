"""第二轮 D：用 20241024（逐笔和委托都有的日子）核对沪深撤单字段，并验证“订单编号差”能否代替撤单龄。"""
import os
os.environ["OMP_NUM_THREADS"]="1"
import pandas as pd, numpy as np
d=20241024
for code in ["000001","300750","600000","688001"]:
    t=pd.read_csv(f"/nvme/level2/{d}/transaction/{code}_{d}_transaction.csv",dtype={"symbol":str})
    o=pd.read_csv(f"/hdd/order_data/{d}/order/{code}_{d}_order.csv",dtype={"symbol":str})
    print(f"\n== {code} 逐笔 {len(t)} 委托 {len(o)}")
    print("委托 order_type/func_code:", o.groupby(["order_type","func_code"]).size().to_dict())
    c=t[t.flag==32].copy()
    print("撤单笔数",len(c),"价格为0占比",round((c.price==0).mean(),4),"量>0占比",round((c.volume>0).mean(),4),"func_code",c.func_code.value_counts().to_dict())
    c["side"]=np.where(c.bid_order>0,"bid","ask"); c["oid"]=np.where(c.bid_order>0,c.bid_order,c.ask_order)
    print("两侧同时非零",int(((c.bid_order>0)&(c.ask_order>0)).sum()),"两侧皆零",int(((c.bid_order==0)&(c.ask_order==0)).sum()))
    # 连续竞价内
    sec=(c.time//1000).astype(int)
    k=((sec>=93000)&(sec<113000))|((sec>=130000)&(sec<145700))
    m=c[k].merge(o[["order","exchtime","price","volume"]].rename(columns={"order":"oid","exchtime":"o_t","price":"o_p","volume":"o_v"}),on="oid",how="left")
    print("连续竞价内撤单",len(m),"能对上委托的占比",round(m.o_t.notna().mean(),4))
    m=m.dropna(subset=["o_t"]); 
    m["age_sec"]=(m.exchtime-m.o_t)/1e6
    print("撤单龄(秒) 中位",round(m.age_sec.median(),1),"p10/p90",round(m.age_sec.quantile(.1),1),round(m.age_sec.quantile(.9),1),"负值占比",round((m.age_sec<0).mean(),4))
    print("撤单量=原委托量占比",round((m.volume==m.o_v).mean(),3),"撤单量<原委托量占比",round((m.volume<m.o_v).mean(),3))
    # 编号差：撤单发生时，本股票最新已知订单编号 与 被撤编号 的差
    ev=t[t.flag!=0][["exchtime","index","bid_order","ask_order"]].copy()
    ev["maxid"]=ev[["bid_order","ask_order"]].max(axis=1).cummax()
    cur=ev.set_index("index").maxid
    m["cur"]=m["index"].map(cur)
    m["id_lag"]=m.cur-m.oid
    v=m[(m.id_lag>=0)&(m.age_sec>=0)]
    print("编号差>=0占比",round((m.id_lag>=0).mean(),4),"编号差与真实撤单龄 秩相关",round(v.id_lag.rank().corr(v.age_sec.rank()),4),"样本",len(v))
    print("订单编号是否逐股连续：相邻委托编号差中位",o["order"].diff().median())
