"""Path choices and wrapper forwarding; no factor computation is started."""
import ast,os
from pathlib import Path
from unittest.mock import patch
import design_whatever as dw
import design_whatever.tail_whatever as tw
import rust_pyfunc as rp
roots=['/nas197/user_home/chenzongwei','/hdd/user_home_unsafe/chenzongwei']
for root in roots:
 p=dw.resolve_factor_storage_paths(root+'/factor_store_hm110','hm110')
 assert p['profile']['network']==(root==roots[0])
 assert rp.factor_store_storage_profile(p['temp_root'])['network'] is False
 assert p['selected_parquet_dir'].startswith(root)
 for n in range(95,111):
  file=next(Path('/home/chenzongwei/pythoncode').glob(f'成果hm{n}_*/hm{n}.py'))
  env={'os':os,'dw':dw};tree=ast.parse(file.read_text())
  with patch.dict(os.environ,{'FACTOR_STORE_ROOT':root}):
   for node in tree.body:
    if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name) and node.targets[0].id in ('ver','industry_neutralize','store_root','store_dir','tail_root'):
     exec(compile(ast.Module([node],type_ignores=[]),str(file),'exec'),env)
  assert env['store_dir'].startswith(root)
  assert rp.factor_store_storage_profile(env['tail_root'])['network'] is False
# Explicit overrides win, and the user-facing wrapper forwards storage options to the native engine.
with patch('design_whatever.tail_v4.run_tail_pipeline_engine',return_value=([],[],[],{'selected_parquet_dir':'/tmp/custom_export'})) as engine,patch.object(tw,'tail_pipeline_v4_postprocess') as post:
 tw.tail_pipeline_engine(colblk_store_dir=roots[0]+'/future_store',ver='storage_test',temp_root='/tmp/custom_cache',selected_parquet_dir='/tmp/custom_export',io_concurrency=3,io_batch_size=2,io_batch_bytes=123456,do_base_part=False,do_fulltest_part=False,do_doc_part=False,return_run_info=True)
 kw=engine.call_args.kwargs
 assert kw['temp_root']=='/tmp/custom_cache' and kw['selected_parquet_dir']=='/tmp/custom_export'
 assert (kw['io_concurrency'],kw['io_batch_size'],kw['io_batch_bytes'])==(3,2,123456)
 assert post.call_args.kwargs['temp_root']=='/tmp/custom_cache'
assert dw.resolve_factor_storage_paths('/nas197/user_home_unsafe/chenzongwei/future','alias')['profile']['network'] is False
print('PASS: 32 script configurations, mount/symlink detection, cache/export choices, explicit overrides, wrapper forwarding')
