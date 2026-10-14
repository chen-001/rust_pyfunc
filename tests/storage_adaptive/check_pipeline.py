"""Real worker entry points on two-stock input links, plus backtest IO strategy equivalence."""
import json,os,shutil,subprocess,sys,tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import rust_pyfunc as rp
import storage_sandbox as sb
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from test_supplement_fixes import _make_tiny_style_vars_dir
out=[]
with tempfile.TemporaryDirectory(prefix='codex_storage_fault_pipeline_') as tmp:
 base=Path(tmp);data=base/'level2';days=[20220819,20220822];codes=['000001','000002']
 for d in days:
  for sub in ['transaction','market_data']:
   folder=data/str(d)/sub;folder.mkdir(parents=True)
   for code in codes:
    source=Path('/ssd_data/stock')/str(d)/sub/f'{code}_{d}_{sub}.csv'
    if source.exists():(folder/source.name).symlink_to(source)
 for label,root in [('HDD','/hdd/user_home_unsafe/chenzongwei'),('NAS','/nas197/user_home/chenzongwei')]:
  with tempfile.TemporaryDirectory(prefix='codex_storage_fault_pipeline_',dir=root) as work:
   store=str(Path(work)/'cross')
   common=dict(pipeline='cross_section_example',n_jobs=4,expected_result_length=3,trading_days=days,bind_cores=False,store_dir=store,store_factor_names=rp.py_cross_section_example_names(),update_mode=True,incremental_projection=True,data_root=str(data))
   rp.run_factor_pipeline_cross_section(tasks=[days[0]],**common)
   first=rp.factor_store_v5_info(store)['record_count'];assert first==2
   if label=='NAS':
    # Fail only publication of the completion marker, after data and receipt are durable.
    opts=dict(common,tasks=[days[1]])
    child='import json,sys,rust_pyfunc as r; r.run_factor_pipeline_cross_section(**json.loads(sys.argv[1]))'
    env=dict(os.environ,LD_PRELOAD='/tmp/storage_io_fault.so',STORAGE_TEST_FAULT='marker')
    failed=subprocess.run([sys.executable,'-c',child,json.dumps(opts)],env=env,capture_output=True,text=True)
    assert failed.returncode!=0 and '失败' in failed.stderr,failed.stderr
    assert rp.factor_store_v5_info(store)['record_count']==4
   rp.run_factor_pipeline_cross_section(tasks=days,**common)
   assert rp.factor_store_v5_info(store)['record_count']==4
   assert set((Path(store)/'_completed_dates').read_text().split())==set(map(str,days))
   cross=[rp.factor_store_v5_read_factor(store,i)['factor'].copy() for i in range(3)]
   broken=Path(work)/'missing_shard';shutil.copytree(store,broken);shutil.rmtree(broken/'shard_7')
   try:rp.factor_store_v5_info(str(broken))
   except OSError:pass
   else:raise AssertionError('missing registered shard accepted')

   stock_store=str(Path(work)/'stock')
   args=dict(pipeline='extreme_point_fit',tasks=[[d,c] for d in days for c in codes],n_jobs=2,backup_file='',expected_result_length=26976,trading_days=days,params={'with_curvature':False},update_mode=True,bind_cores=False,store_dir=stock_store,store_factor_names=[f'f{i}' for i in range(26976)],incremental_projection=True,data_root=str(data),export_n_jobs=2)
   rp.run_factor_pipeline(**args);assert rp.factor_store_v5_info(stock_store)['record_count']==4
   rp.run_factor_pipeline(**args);assert rp.factor_store_v5_info(stock_store)['record_count']==4
   stock=np.asarray(sb.read_matrix(stock_store,list(range(26976)),True),dtype=np.float32)
   if label=='HDD':gold_cross=cross;gold_stock=stock
   else:
    for a,b in zip(gold_cross,cross):np.testing.assert_array_equal(a,b)
    np.testing.assert_array_equal(gold_stock,stock)
   out.append(dict(device=label,cross_rows=4,stock_rows=4,resume_no_duplicates=True,worker_values_match=True))
 # Synthetic factor/return/style arrays isolate storage from market calculations.
 ds=[int(d.strftime('%Y%m%d')) for d in pd.date_range('2022-01-01',periods=60)]
 cs=[f'{i:06}' for i in range(1,65)];rng=np.random.default_rng(37)
 factors=rng.normal(size=(len(ds)*len(cs),16)).astype(np.float32)
 style=_make_tiny_style_vars_dir(base/'style',ds,cs)
 inputs=base/'inputs';inputs.mkdir()
 for key in ['ret_gap1','ret_sum_gap1','ret_gap5','ret_sum_gap5']:np.save(inputs/f'{key}.npy',rng.normal(0,.02,(len(ds),len(cs))).astype(np.float32))
 np.save(inputs/'restrict.npy',np.zeros((len(ds),len(cs)),np.float32));np.save(inputs/'index_ret.npy',rng.normal(0,.01,len(ds)).astype(np.float32))
 names=[f'f{i}' for i in range(16)]
 with tempfile.TemporaryDirectory(prefix='codex_storage_fault_backtest_',dir='/nas197/user_home/chenzongwei') as work:
  store=str(Path(work)/'store');sb.append_store(store,np.repeat(ds,len(cs)).tolist(),cs*len(ds),factors.tolist(),names,True)
  common=dict(colblk_store_dir=store,factor_names=names,factor_paths=[f'{store}::{i}' for i in range(16)],dates=ds,stocks=[c+'.SZ' for c in cs],windows=[5,10],fold=False,n_jobs=4,min_valid=5,style_vars_dir=style,backtest_start=ds[0],cover_rate=.5,majority_count_threshold=200.,zero_max_threshold=1.,nan_max_threshold=1.,save_all_metrics=True,industry_neutralize=True,industry_matrix=np.tile((np.arange(64)%31+1).astype(np.float64),(60,1)),ic_only=True,
   ret_point_neu_gap5=0.,ret_point_neu_gap1=0.,ic_point_neu_gap5=0.,ic_point_neu_gap1=0.,ret_point_gap5=0.,ret_point_gap1=0.,ic_point_gap5=0.,ic_point_gap1=0.,ic_more_important_gap5=None,ic_more_important_gap1=None,**{f'{key}_path':str(inputs/f'{key}.npy') for key in ['ret_gap1','ret_sum_gap1','ret_gap5','ret_sum_gap5','restrict','index_ret']})
  results=[]
  for batch in [1,4]:
   cache=base/f'engine_{batch}';results.append(rp.tail_backtest_engine(cache_root=str(cache),io_batch_size=batch,**common))
   assert results[-1]['processed_sources']==16
  p1=base/'engine_1';p4=base/'engine_4'
  for p in p1.rglob('*.parquet'):
   q=p4/p.relative_to(p1);assert q.exists()
   a=pd.read_parquet(p);b=pd.read_parquet(q)
   if 'factor_name' in a.columns:
    assert not a['factor_name'].duplicated().any() and not b['factor_name'].duplicated().any()
    a=a.set_index('factor_name').sort_index();b=b.set_index('factor_name').sort_index()
   pd.testing.assert_frame_equal(a,b,check_exact=True)
  for p in p1.rglob('*.npy'):
   q=p4/p.relative_to(p1);np.testing.assert_array_equal(np.load(p,allow_pickle=True),np.load(q,allow_pickle=True))
  assert results[0]['candidate_counts']==results[1]['candidate_counts']
  assert results[0]['candidate_counts']['neu_gap5']>0,results
  assert results[1]['io_read_groups']==4 and results[0]['io_read_groups']==16
  out.append(dict(backtest_16_factors=True,metrics_identical=True,IC_identical=True,read_groups=[16,4]))
print(json.dumps(out,ensure_ascii=False));Path('/tmp/storage_pipeline_checks.json').write_text(json.dumps(out,indent=2))
