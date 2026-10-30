"""Compare saved daily panels by exact labels, including every nonfinite position."""
import json
from pathlib import Path
import re
import sys

import numpy as np

reference, candidate, output = map(Path, sys.argv[1:4])
left = json.loads((reference / "labels.json").read_text())
right = json.loads((candidate / "labels.json").read_text())
assert left["codes"] == right["codes"], "Stock universe/order differs"
assert len(set(left["names"])) == len(left["names"])
assert len(set(right["names"])) == len(right["names"])
a = np.ascontiguousarray(np.load(reference / "values.npy").T)
b = np.ascontiguousarray(np.load(candidate / "values.npy").T)
lookup = {name: i for i, name in enumerate(left["names"])}
candidate_names = set(right["names"])
groups = {key: dict(columns=0, values=0, changed_columns=0, bit_mismatches=0,
                    nan_mask_mismatches=0, inf_mask_mismatches=0)
          for key in ("shared_topk", "non_topk")}
changed = []
for j, name in enumerate(right["names"]):
    if name not in lookup:
        continue
    x, y = a[lookup[name]], b[j]
    group = "shared_topk" if re.search(r"(?:x0topk|gtopk)\d+_", name) else "non_topk"
    bits = int(np.count_nonzero(x.view(np.uint32) != y.view(np.uint32)))
    nan = int(np.count_nonzero(np.isnan(x) != np.isnan(y)))
    inf = int(np.count_nonzero(np.isinf(x) != np.isinf(y)))
    g = groups[group]
    g["columns"] += 1
    g["values"] += len(x)
    g["changed_columns"] += bits > 0
    g["bit_mismatches"] += bits
    g["nan_mask_mismatches"] += nan
    g["inf_mask_mismatches"] += inf
    if bits:
        valid = np.isfinite(x) & np.isfinite(y)
        delta = np.abs(x[valid].astype(float) - y[valid].astype(float))
        changed.append(dict(name=name, group=group, bit_mismatches=bits,
                            nan_mask_mismatches=nan, inf_mask_mismatches=inf,
                            max_abs_diff=float(delta.max()) if len(delta) else None))
result = dict(reference=str(reference), candidate=str(candidate), stocks=len(left["codes"]),
              same_column_order=left["names"] == right["names"], groups=groups,
              removed=[n for n in left["names"] if n not in candidate_names],
              added=[n for n in right["names"] if n not in lookup], changed=changed)
output.write_text(json.dumps(result, ensure_ascii=False, indent=2))
print(json.dumps({k: v for k, v in result.items() if k not in ("removed", "added", "changed")},
                 ensure_ascii=False, indent=2))
print(f"removed={len(result['removed'])}, added={len(result['added'])}")
