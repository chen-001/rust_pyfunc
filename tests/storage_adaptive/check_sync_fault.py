"""投影强制落盘确认（fsync）的故障注入检查：一次失败必须报错、分段确认、每次失败都不回归。

为什么确认失败不许重试（2026-10-16 独立核验）：往 NAS 写 64 GiB，第一次确认在 192.6 秒报 EIO，
同一个句柄上紧接着再确认一次 3.4 秒就返回成功，但读回时 65536 块里有 28582 块内容不对，服务端
真实长度比客户端报的少约 64 MB（两轮都复现）。软挂载超时之后那批写已经丢了，再问一次拿到的
成功是假的，会把「确定失败」变成「静默成功且产物损坏」。扛住对面停顿靠的是分段确认（用例二）。"""
import json,os,subprocess,sys,tempfile
from pathlib import Path
OUT='/tmp/storage_sync_fault_checks.json'
SO='/tmp/storage_io_fault.so'
NAS='/nas197/user_home/chenzongwei'
# 注入器只对路径含 codex_storage_fault_ 的文件生效，所以临时 store 必须建在带该前缀的目录下。
CHILD="""
import numpy as np,storage_sandbox as sb,sys
rows=1000;names=['f0','f1','f2','f3'];codes=['%06d'%i for i in range(rows)]
values=(np.arange(rows,dtype=np.float32)[:,None]*np.float32(.001)+np.arange(4,dtype=np.float32)).tolist()
sb.append_store(sys.argv[1],[20230102]*rows,codes,values,names,True)
"""
results=[]
def dump():Path(OUT).write_text(json.dumps(results,ensure_ascii=False,indent=2))
# 三个用例都跑完再抛异常：某一个红（比如实现还没重建）不能挡住其它用例的结果。
def case(name,ok,detail,**extra):
 results.append({'case':name,'ok':bool(ok),'detail':detail,**extra});dump()
def report():
 bad=[r for r in results if not r['ok']]
 if bad:raise AssertionError('；'.join('%s 失败：%s'%(r['case'],r['detail'].split('\n')[0]) for r in bad))
def run(store,fault,chunk=None):
 env=dict(os.environ,LD_PRELOAD=SO,STORAGE_TEST_FAULT=fault)
 env.pop('RUST_PYFUNC_PROJ_SYNC_CHUNK',None)
 if chunk:env['RUST_PYFUNC_PROJ_SYNC_CHUNK']=str(chunk)
 return subprocess.run([sys.executable,'-c',CHILD,store],env=env,capture_output=True,text=True)
def proj_syncs(stderr):return sum(1 for l in stderr.splitlines() if l.startswith('FAULT_FSYNC') and 'factors.proj.tmp' in l)
def tail(text,n=6):return '\n'.join(text.strip().splitlines()[-n:])
with tempfile.TemporaryDirectory(prefix='codex_storage_fault_',dir=NAS) as tmp:
 # 用例一：sync_once 让进程内第一次对目标文件的确认报 EIO。新建 store 时这第一次落在
 # factors.atomic.tmp（idx 发布）上，和投影走同一个确认函数。要求它必须让进程非零退出：
 # 那次写入可能已经丢了，重问一次拿到的成功不能当成落盘成功。
 r=run(str(Path(tmp)/'single'),'sync_once')
 case('sync_once_must_fail',r.returncode!=0 and '失败' in r.stderr and '次才成功' not in r.stderr,
      '注入一次确认失败后，进程退出码 %d（要求非 0），stderr %s「失败」字样、%s「次才成功」这类重试成功字样。'
      '缺的是「第一次确认失败就报错退出」，不能重问一次把假成功当落盘成功。\n%s'
      %(r.returncode,'含' if '失败' in r.stderr else '不含','含' if '次才成功' in r.stderr else '不含',tail(r.stderr)),rc=r.returncode)
 r4=run(str(Path(tmp)/'seg4096'),'sync_log',4096)
 n4=proj_syncs(r4.stderr)
 r1=run(str(Path(tmp)/'seg1024'),'sync_log',1024)
 n1=proj_syncs(r1.stderr)
 case('sync_log_segmented',r4.returncode==0 and r1.returncode==0 and n4>=2 and n1>n4,
      'factors.proj.tmp 的确认次数：RUST_PYFUNC_PROJ_SYNC_CHUNK=4096 时 %d 次（要求 ≥2），=1024 时 %d 次（要求多于 4096 时的次数）。'
      '两次都少，说明投影还没分段确认；次数不随段大小变化，说明该环境变量没被用上。退出码 %d/%d。\n%s'
      %(n4,n1,r4.returncode,r1.returncode,tail(r4.stderr)),syncs_4096=n4,syncs_1024=n1)
 rs=run(str(Path(tmp)/'sync_fail'),'sync')
 case('sync_always_fails',rs.returncode!=0 and '失败' in rs.stderr,
      '每次确认都注入 EIO 时，进程退出码 %d（要求非 0），stderr %s「失败」字样。缺的是「确认失败必须报错退出」。\n%s'
      %(rs.returncode,'含' if '失败' in rs.stderr else '不含',tail(rs.stderr)),rc=rs.returncode)
report()
print(json.dumps(results,ensure_ascii=False))
