import json, numpy as np, collections
meta=json.load(open('meta.json')); names=np.array(meta['names'])
PREF=['cnt_t005','cnt_t01','cnt_t02','cnt_t05','cnt_t1','cnt_t3','cnt_t5','cnt_t10','cnt_t30','vol_t02','vol_t05','vol_t1','vol_t3','vol_t5','vol_t30']
met=np.empty(len(names),object)
for i,nm in enumerate(names):
    for p in PREF:
        if nm.startswith(p+'_'):
            met[i]=nm[len(p)+1:]; break
alive=np.load('alive_0.9900_abs.npy'); c=collections.Counter(met[~alive]); tot=collections.Counter(met)
full=sorted([m for m,n in c.items() if n==tot[m]])
print('COUNT', len(full))
print(' '.join(full))
