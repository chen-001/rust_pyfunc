import json, os, subprocess
import numpy as np
import rust_pyfunc as rp

os.chdir('/home/chenzongwei/rust_pyfunc/perf_sandbox/hm32_bench')
cases = [
    (20250303, '000001'), (20250303, '000016'), (20250303, '600519'),
    (20250304, '600519'), (20180601, '000002'), (20190613, '300750'),
    (20201215, '002594'), (20220819, '000001'), (20241231, '000001'),
    (20231020, '300750'), (20210630, '688981'), (20170213, '601398'),
]
rows = []
for date, code in cases:
    p = f'/ssd_data/stock/{date}/market_data/{code}_{date}_market_data.csv'
    if not os.path.exists(p):
        rows.append((date, code, 'FILE_MISSING', None, None, None)); continue
    lib = np.array(rp.okay_compute_stock('hm032', code, date), dtype=np.float64)
    r = subprocess.run(['./hm32_bench', str(date), code, 'dump'], capture_output=True, text=True)
    dn = df = None
    for line in r.stdout.splitlines():
        if line.startswith('DUMP_NAIVE='): dn = np.array([float(x) for x in line[11:].split(',')])
        if line.startswith('DUMP_FAST='): df = np.array([float(x) for x in line[10:].split(',')])
    if dn is None or df is None:
        rows.append((date, code, 'BENCH_FAIL:' + r.stdout[-160:], None, None, None)); continue
    def md(a, b):
        if len(a) != len(b): return float('nan')
        d = np.abs(a - b)
        d = d[np.isfinite(d)]
        return float(d.max()) if d.size else 0.0
    rows.append((date, code, f'len={len(dn)} maxabs={np.nanmax(np.abs(dn)):.4g}',
                 md(lib, dn), md(lib, df), md(dn, df)))
    print(f'{date} {code} len={len(dn)} |lib-naive|={md(lib,dn):.3e} |lib-fast|={md(lib,df):.3e} |naive-fast|={md(dn,df):.3e}', flush=True)
print('==== SUMMARY ====')
ln = [r[3] for r in rows if r[3] is not None]
lf = [r[4] for r in rows if r[4] is not None]
nf = [r[5] for r in rows if r[5] is not None]
print('cases', len(rows), 'ok', len(ln))
print('max |lib - naive| over all cells = %.3e' % max(ln))
print('max |lib - fast|  over all cells = %.3e' % max(lf))
print('max |naive - fast| over all cells = %.3e' % max(nf))
print('identical? lib-vs-naive==lib-vs-fast for every case:', all(abs(a - b) < 1e-30 for a, b in zip(ln, lf)))
json.dump(rows, open('verify_group2.json', 'w'))
