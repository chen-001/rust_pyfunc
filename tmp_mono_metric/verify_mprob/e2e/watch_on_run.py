import glob, time, os, pandas as pd
ROOT="/tmp/mprob_smoke_on_tail_v4"
t0=time.time()
while time.time()-t0 < 1500:
    ready=False
    for p in sorted(glob.glob(f"{ROOT}/metrics/summary_*_candidates.parquet")):
        try:
            d=pd.read_parquet(p)
        except Exception as e:
            continue
        if len(d) and "MPROB" in d.columns and int(d.MPROB.notna().sum())>0:
            print(f"READY {p} rows={len(d)} MPROB 非空={int(d.MPROB.notna().sum())} 范围=({float(d.MPROB.min()):.6f},{float(d.MPROB.max()):.6f})", flush=True)
            ready=True
    if ready:
        print("ALL_READY", flush=True); break
    time.sleep(20)
else:
    print("TIMEOUT 未等到非空 MPROB", flush=True)
    for p in sorted(glob.glob(f"{ROOT}/metrics/*.parquet")):
        try:
            d=pd.read_parquet(p); print(" ",os.path.basename(p),"rows",len(d),"MPROB非空",int(d.MPROB.notna().sum()) if "MPROB" in d.columns else "无列")
        except Exception as e: print(" ",p,"ERR",e)
