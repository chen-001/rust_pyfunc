import os, subprocess
import numpy as np
import rust_pyfunc as rp
os.chdir('/home/chenzongwei/rust_pyfunc/perf_sandbox/hm32_bench')
cases = [(20250303,'000001'),(20250304,'600519'),(20180601,'000002'),(20190613,'300750'),
         (20201215,'002594'),(20220819,'000001'),(20241231,'000001'),(20231020,'300750'),
         (20210630,'688981'),(20170213,'601398'),(20220819,'600000'),(20241231,'300750')]
tot = dict(old_mismatch=0, new_mismatch=0, values=0, max_rel_old=0.0, max_rel_new=0.0)
for date, code in cases:
    lib = np.array(rp.okay_compute_stock('hm032', code, date), dtype=np.float32)
    r = subprocess.run(['./hm32_bench', str(date), code, 'dump'], capture_output=True, text=True)
    dn = df = None
    for line in r.stdout.splitlines():
        if line.startswith('DUMP_NAIVE='): dn = np.array([float(x) for x in line[11:].split(',')])
        if line.startswith('DUMP_FAST='): df = np.array([float(x) for x in line[10:].split(',')])
    nan_ok = any('nan=true' in l for l in r.stdout.splitlines())
    f32_lib = lib
    f32_old = dn.astype(np.float32)
    f32_new = df.astype(np.float32)
    mo = int((f32_old != f32_lib).sum())
    valid = bool(np.all(np.isfinite(df)))
    mn = int((f32_new != f32_lib).sum()) if valid else -1
    with np.errstate(invalid='ignore', divide='ignore'):
        rel_old = np.nanmax(np.abs((dn - f32_lib.astype(np.float64)) / np.where(f32_lib == 0, np.nan, f32_lib)))
        rel_new = np.nanmax(np.abs((df - f32_lib.astype(np.float64)) / np.where(f32_lib == 0, np.nan, f32_lib))) if valid else float('nan')
    print(f'{date} {code} nonfinite_session={nan_ok} f32输出变了: 旧版={mo} 新版={mn} 110个值  最大相对差 旧={rel_old:.2e} 新={rel_new:.2e}', flush=True)
    if valid:
        tot['old_mismatch'] += mo; tot['new_mismatch'] += mn; tot['values'] += 110
        tot['max_rel_old'] = max(tot['max_rel_old'], rel_old); tot['max_rel_new'] = max(tot['max_rel_new'], rel_new)
print('==== 汇总 ====')
print('有效样本 %d 组 / %d 个 f32 输出值' % (tot['values']//110, tot['values']))
print('旧算法写成 f32 后与库里存的值不同的个数 = %d' % tot['old_mismatch'])
print('新算法写成 f32 后与库里存的值不同的个数 = %d' % tot['new_mismatch'])
print('最大相对差 旧=%.3e 新=%.3e' % (tot['max_rel_old'], tot['max_rel_new']))
