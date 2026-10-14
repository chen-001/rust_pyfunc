"""Sandbox/installed-reader checks. All stores are disposable; no existing data is touched."""
import importlib,json,os,shutil,subprocess,sys,tempfile,time
from pathlib import Path
import numpy as np
import storage_sandbox as sb
reader=importlib.import_module(sys.argv[1] if len(sys.argv)>1 else 'storage_sandbox')
results=[]
names=['f0','f1','f2','f3']
codes=[f'{i:06}' for i in range(1000)]
values=(np.arange(1000,dtype=np.float32)[:,None]*np.float32(.001)+np.arange(4,dtype=np.float32)).tolist()
for label,root in [('HDD','/hdd/user_home_unsafe/chenzongwei'),('NAS','/nas197/user_home/chenzongwei')]:
 with tempfile.TemporaryDirectory(prefix='codex_storage_fault_',dir=root) as tmp:
  empty=str(Path(tmp)/'empty');sb.append_store(empty,[],[],[],names,True)
  sb.append_store(empty,[20230102]*1000,codes,values,names,True)
  assert len(reader.factor_store_v5_read_factor(empty,0)['factor'])==1000
  store=str(Path(tmp)/'store')
  for day in range(5):sb.append_store(store,[20230102+day]*1000,codes,values,names,True)
  info=reader.factor_store_v5_info(store)
  assert info['record_count']==5000 and info['is_projected']
  for col in range(4):
   got=reader.factor_store_v5_read_factor(store,col)['factor']
   np.testing.assert_array_equal(np.sort(got),np.sort(np.tile(np.asarray(values,dtype=np.float32)[:,col],5)))
  np.testing.assert_array_equal(sb.read_matrix(store,list(range(4)),False),sb.read_matrix(store,list(range(4)),True))
  assert len(list(Path(store).glob('factors.proj.delta.[0-9]*')))==(4 if label=='NAS' else 1)
  if label=='NAS':
   # Unpublished/orphan numeric files must never participate in a committed projection.
   shutil.copy(next(Path(store).glob('factors.proj.delta.[0-9]*')),Path(store)/'factors.proj.delta.999999999')
   assert len(reader.factor_store_v5_read_factor(store,0)['factor'])==5000
   assert (Path(store)/'chunks.checkpoint').exists()
   dirty=Path(tmp)/'dirty';shutil.copytree(store,dirty)
   original_dates=(dirty/'dates.bin').read_bytes()
   with (dirty/'dates.bin').open('ab') as f:f.write(b'partial')
   with (dirty/'factors.colblk').open('ab') as f:f.write(b'\0'*32)
   sb.append_store(str(dirty),[],[],[],names,True)
   assert (dirty/'dates.bin').read_bytes()==original_dates
   assert len(reader.factor_store_v5_read_factor(str(dirty),0)['factor'])==5000
   bad=Path(tmp)/'bad_checkpoint';shutil.copytree(store,bad)
   cp=json.loads((bad/'chunks.checkpoint').read_text());cp['rows']=0
   (bad/'chunks.checkpoint').write_text(json.dumps(cp))
   before=(bad/'dates.bin').read_bytes()
   try:sb.append_store(str(bad),[],[],[],names,True)
   except OSError:pass
   else:raise AssertionError('bad checkpoint accepted')
   assert (bad/'dates.bin').read_bytes()==before

  child="import importlib,sys; m=importlib.import_module(sys.argv[1]); m.factor_store_v5_read_factor(sys.argv[2],0)"
  env=dict(os.environ,LD_PRELOAD='/tmp/storage_io_fault.so',STORAGE_TEST_FAULT='read')
  r=subprocess.run([sys.executable,'-c',child,reader.__name__,store],env=env,capture_output=True,text=True)
  assert r.returncode!=0 and ('Input/output error' in r.stderr or 'os error 5' in r.stderr),r.stderr
  sync_child="import storage_sandbox as s,sys; s.append_store(sys.argv[1],[20230201],['000001'],[[1.,2.,3.,4.]],['f0','f1','f2','f3'],True)"
  env=dict(os.environ,LD_PRELOAD='/tmp/storage_io_fault.so',STORAGE_TEST_FAULT='sync')
  r=subprocess.run([sys.executable,'-c',sync_child,str(Path(tmp)/'sync_failure')],env=env,capture_output=True,text=True)
  assert r.returncode!=0 and '失败' in r.stderr,r.stderr
  # Existing projections must fail loudly if their value section is truncated.
  damaged=str(Path(tmp)/'damaged');shutil.copytree(store,damaged)
  q=Path(damaged)/'factors.proj';q.write_bytes(q.read_bytes()[:-4])
  try:reader.factor_store_v5_read_factor(damaged,0)
  except OSError:pass
  else:raise AssertionError('truncated projection accepted')
  # Copying an entire store between media retains readability and future append compatibility.
  other='/nas197/user_home/chenzongwei' if label=='HDD' else '/hdd/user_home_unsafe/chenzongwei'
  with tempfile.TemporaryDirectory(prefix='codex_storage_fault_',dir=other) as dest:
   migrated=str(Path(dest)/'store');shutil.copytree(store,migrated)
   sb.append_store(migrated,[20230107]*1000,codes,values,names,True)
   assert len(reader.factor_store_v5_read_factor(migrated,0)['factor'])==6000
   back=str(Path(tmp)/'roundtrip');shutil.copytree(migrated,back)
   sb.append_store(back,[20230108]*1000,codes,values,names,True)
   assert len(reader.factor_store_v5_read_factor(back,0)['factor'])==7000

  results.append({'device':label,'rows':5000,'deltas':info['proj_deltas'],'normal_values_match':True,'EIO_raises':True,'fsync_raises':True,'truncation_raises':True,'media_migration':True})
# Bounded compaction with a low test threshold, and group-local batching.
with tempfile.TemporaryDirectory(prefix='codex_storage_fault_',dir='/nas197/user_home/chenzongwei') as tmp:
 os.environ['RUST_PYFUNC_NAS_DELTA_SEGMENTS']='3';root=Path(tmp)
 for g,ns in [('a',names),('b',['g0','g1','g2','g3'])]:
  store=str(root/g)
  for day in range(7):sb.append_store(store,[20230102+day]*1000,codes,values,ns,True)
  assert sb.factor_store_v5_info(store)['proj_deltas']<=2
 sb.factor_store_v5_register_group(str(root),'a','a');sb.factor_store_v5_register_group(str(root),'b','b')
 np.testing.assert_array_equal(sb.read_matrix(str(root),list(range(8)),True),sb.read_matrix(str(root),list(range(8)),False))
 assert len(reader.factor_store_v5_read_factor(str(root),4)['factor'])==7000
 del os.environ['RUST_PYFUNC_NAS_DELTA_SEGMENTS']
results.append({'compaction':True,'group_batching':True,'reader':reader.__name__})
for path,expected in [('/nas197/user_home/chenzongwei/new_store','nas'),('/nas197/user_home_unsafe/chenzongwei/new_store','local_btrfs'),('/hdd/user_home_unsafe/chenzongwei/new_store','local_btrfs')]:
 assert reader.factor_store_storage_profile(path)['kind']==expected
print(json.dumps(results,ensure_ascii=False))
Path('/tmp/storage_adaptive_checks.json').write_text(json.dumps(results,ensure_ascii=False,indent=2))
