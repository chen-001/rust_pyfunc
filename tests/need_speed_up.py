import design_whatever as dw
import pandas as pd
import rust_pyfunc as rp
import numpy as np
import tqdm

closes1=dw.read_minute_data('close',20220819,20220819).reset_index(drop=True)
closes=(closes1-closes1.mean())/closes1.std()
rets=closes/closes.shift(1)-1
vols=rets.rolling(15).std()
vols_lag1=vols.shift(1)
vols_lag2=vols.shift(2)
vols_lag3=vols.shift(3)
vols_lag4=vols.shift(4)
vols_lag5=vols.shift(5)
many=dw.merge_many([vols,vols_lag1,vols_lag2,vols_lag3,vols_lag4,vols_lag5],names=['vols','vols_lag1','vols_lag2','vols_lag3','vols_lag4','vols_lag5'],how='inner')
def ols(df:pd.DataFrame):
    betas=rp.ols(df[['vols_lag1','vols_lag2','vols_lag3','vols_lag4','vols_lag5']].to_numpy(dtype=float),df['vols'].to_numpy(dtype=float))
    return pd.Series({'beta0':betas[0],'beta1':betas[1],'beta2':betas[2],'beta3':betas[3],'beta4':betas[4],'beta5':betas[5],'r2':betas[6]})
tqdm.auto.tqdm.pandas()
betas=many.groupby('code').progress_apply(ols)
betas,r2_pri=betas[['beta0','beta1','beta2','beta3','beta4','beta5']].T,betas[['r2']]
betas.index=['intercept','vols_lag1','vols_lag2','vols_lag3','vols_lag4','vols_lag5']
r2_pri.columns=['r2_pri']
many=many.assign(intercept=1)

def predict(df:pd.DataFrame):
    xs=df[['intercept','vols_lag1','vols_lag2','vols_lag3','vols_lag4','vols_lag5']]
    y_pred=xs.dot(betas).T
    resi=(y_pred-df.vols).T**2
    resi=resi.sum()
    r2=1-resi/df.vols.var()/df.shape[0]
    return r2

resis:pd.DataFrame=many.groupby('code').progress_apply(predict).replace([np.inf,-np.inf],np.nan)
resis=resis.where(resis>=0,np.nan)