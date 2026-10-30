"""Check named carriers and deterministic f32 volumes against a Python reference."""
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BASE = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else ROOT / 'sandbox/yupei_correctness_20261030'
SOURCE = BASE / 'src/yupei_multi/mod.rs'
code = SOURCE.read_text()
body = code[code.index('#[derive(Clone, Copy, PartialEq, Eq)]\npub enum CellMode'):code.index('/// 用一套桶单元跑 15 条链')]
# A deliberately different order from stock/order-key order, with collisions in buckets.
records = [(100000, 2., 66, 30, 0), (100000, 1., 66, 10, 0),
           (100000, 2., 66, 20, 0), (110000, 9., 83, 0, 40),
           (120000, 4., 66, 30, 0), (120000, 3., 83, 0, 50)]
records += [(i * 10000 + (i % 3) * 100, float((i % 9) + 1), 66 if i % 2 else 83,
             100 + i // 3, 200 + i // 4) for i in range(50)]
records += [(-1, 1., 66, 999, 0), (14220000000, 1., 66, 999, 0),
            (50000, 1., 32, 999, 999)]


def bucketize(points):
    rows = []
    for b, u in points:
        k = int(np.floor(b + .5))  # Rust round, for nonnegative bucket coordinates.
        if 0 <= k < 1422000:
            rows.append((k, np.array(u, dtype=np.float32)))
    result = []
    for k, u in sorted(rows, key=lambda x: x[0]):
        if result and result[-1][0] == k:
            result[-1][1] += u
        else:
            result.append([k, u.copy()])
    return [[k, list(map(int, u.view(np.uint32)))] for k, u in result]


def reference(mode):
    points, orders = [], {}
    for time, volume, flag, ask, bid in records:
        if not 0 <= time < 14220000000 or flag not in (66, 83):
            continue
        b = time // 10000
        w = float(np.float32(np.sqrt(max(float(np.float32(volume)), 0))))
        if mode == 'base':
            points.append((b, [1, 0, w, 0] if flag == 66 else [0, 1, 0, w]))
        else:
            oid = ask if flag == 66 else bid
            key = oid if oid > 0 else -time
            v = orders.setdefault(key, [0., 0., 0.])
            v[0] += b * max(w, 1e-6)
            v[1 if flag == 66 else 2] += max(w, 1e-6)
    for _, (tw, p, m) in sorted(orders.items()):
        if mode == 'side':
            p, m = m, p
        points.append((tw / (p + m), [float(p > 0), float(m > 0), p, m]))
    return bucketize(points)


fixture = ','.join(f'TradeRecord{{time_us:{t},volume:{v}f32,flag:{f},ask_order:{a},bid_order:{b}}}'
                   for t, v, f, a, b in records)
native = '''
mod fast_csv_reader { #[derive(Clone)] pub struct TradeRecord {
 pub time_us:i64,pub volume:f32,pub flag:i32,pub ask_order:i64,pub bid_order:i64 } }
const BUCKET_US:i64=10000; const N_BUCKETS:usize=1422000;
__BODY__
fn print_cells(name:&str,v:&[(u32,[f32;4])]) {
 let rows:Vec<String>=v.iter().map(|(b,u)|format!("[{},[{}, {}, {}, {}]]",b,u[0].to_bits(),u[1].to_bits(),u[2].to_bits(),u[3].to_bits())).collect();
 println!("{}=[{}]",name,rows.join(","));
}
fn main(){
 use fast_csv_reader::TradeRecord;
 let recs=vec![__FIXTURE__];
 let first=build_all_cells(&recs,0);
 for _ in 0..500 {
  let v=build_all_cells(&recs,0);
  for (actual,expected,mode) in [(&v.base,&first.base,CellMode::Base),(&v.order,&first.order,CellMode::Order),(&v.otsu,&first.otsu,CellMode::Otsu),(&v.side,&first.side,CellMode::OrderSide)] {
   assert_eq!(actual,expected);assert_eq!(*actual,build_cells(&recs,0,mode));
  }
 }
 assert_eq!(CellMode::Otsu.tag(),"oc_");assert_eq!(CellMode::OrderSide.tag(),"os_");
 assert_ne!(first.otsu,first.side); // fixture actually distinguishes the carrier labels
 print_cells("base",&first.base);print_cells("order",&first.order);print_cells("side",&first.side);
}
'''.replace('__BODY__', body).replace('__FIXTURE__', fixture)
with tempfile.TemporaryDirectory(prefix='yupei_carrier_fix_') as tmp:
    tmp = Path(tmp)
    (tmp / 'check.rs').write_text(native)
    subprocess.run(['/home/chenzongwei/.cargo/bin/rustc', '+1.97.0', '--edition=2021', '-Awarnings',
                    '-C', 'opt-level=2', str(tmp / 'check.rs'), '-o', str(tmp / 'check')], check=True)
    runs = [subprocess.check_output([str(tmp / 'check')], text=True) for _ in range(3)]
    assert len(set(runs)) == 1
    actual = dict((k, json.loads(v)) for k, v in (line.split('=', 1) for line in runs[0].splitlines()))
    for mode in ['base', 'order', 'side']:
        assert actual[mode] == reference(mode), mode
result = dict(status='PASS',processes=3,repetitions_per_process=500,carriers_checked=4,
              python_reference=['base','order','side'],independent_builder_comparison='all four carriers',
              distinct_cluster_and_side=True,order_bits=actual['order'],source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest())
output = ROOT / 'reports/yupei_correctness_20261030/carrier_regression.json'
output.write_text(json.dumps(result,indent=2))
print(json.dumps({k:v for k,v in result.items() if k!='order_bits'}))
