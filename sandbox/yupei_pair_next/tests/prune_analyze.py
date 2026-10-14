"""Analyze actual multi-day outputs and intermediate samples.
Reproduction: in sandbox/yupei_pair_next apply `patch -p1 < tests/prune_audit.patch`,
then `bash build.sh`. Run tests/benchmark_yupei_pair_exact.py using this isolated
package and YUPEI_PRUNE_AUDIT=/tmp/yp_prune_audit/intermediates for the three dates.
YUPEI_PRUNE_TOPK_CAP=100 or 0 explicitly opts into dropping output columns;
unset it to retain all outputs. Audit runs must not replace the installed package.
Undefined correlations raise; exact equality uses raw float32 bits, not tolerance.
"""
import collections
import hashlib
import json
import re
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

root = Path('/tmp/yp_prune_audit')
dates = [20220819, 20240930, 20241231]
metas = [json.loads((root / str(d) / 'result.json').read_text()) for d in dates]
arrays = [np.load(root / str(d) / 'values.npy', mmap_mode='r') for d in dates]
names = metas[0]['names']
assert all(m['names'] == names for m in metas)
# Each column's three daily byte strings must match, including NaNs and signed zeros.
groups = collections.defaultdict(list)
constant = []
for j, name in enumerate(names):
    h = hashlib.sha256()
    for a in arrays:
        h.update(a[:, j].tobytes())
    groups[h.digest()].append(name)
    if all(np.unique(a[:, j][np.isfinite(a[:, j])]).size <= 1 for a in arrays):
        constant.append(name)
duplicates = [g for g in groups.values() if len(g) > 1]
for group in duplicates:
    for name in group[1:]:
        for a in arrays:
            assert np.array_equal(a[:, names.index(group[0])].view("u4"), a[:, names.index(name)].view("u4"))
constant_set = set(constant)
nonconstant = [g for g in duplicates if not any(x in constant_set for x in g)]

def corr(x, y):
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if len(x) < 30 or np.ptp(x) == 0 or np.ptp(y) == 0:
        raise ValueError(f'Undefined correlation: overlap={len(x)}, finite={valid.sum()}, variances={np.var(x)},{np.var(y)}')
    return float(spearmanr(x, y).statistic)

intermediate = []
taus = []
ks = []
sampled_matrix_groups = {}
for date, meta, a in zip(dates, metas, arrays):
    folder = root / 'intermediates' / str(date)
    mats = {}
    for path in sorted(folder.glob('*.json')):
        m = json.loads(path.read_text())
        v = np.fromfile(path.with_name(path.stem + '_matrices.bin'), dtype='<f4').reshape(len(m['fields']), -1)
        mats[path.stem] = (m, dict(zip(m['fields'], v)))
        hg = np.fromfile(path.with_name(path.stem + '_hg.bin'), dtype='<f4').reshape(16, 2, m['n'])
        overlap = []
        for h, g in hg:
            ih = np.argsort(-h, kind='stable')[:500]
            ig = np.argsort(-g, kind='stable')[:500]
            overlap.append(len(np.intersect1d(ih, ig)) / 500)
        record = {k: m[k] for k in ['date','mat','n','active','all_valid_pairs','differences','residual_valid_mask_differences','sample_triples','net_r2_bits_differ']}
        record['hg_top500_overlap_mean'] = float(np.mean(overlap))
        record['g_gs_spearman_sample'] = corr(mats[path.stem][1]['g'], mats[path.stem][1]['gs'])
        record['z_zs_spearman_sample'] = corr(mats[path.stem][1]['z'], mats[path.stem][1]['zs'])
        intermediate.append(record)
    matrix_hashes = collections.defaultdict(list)
    reference_pairs = next(iter(mats.values()))[0]['samples']
    for mat, (m, fields) in mats.items():
        assert m['samples'] == reference_pairs
        for field, values in fields.items():
            matrix_hashes[hashlib.sha256(values.tobytes()).digest()].append(mat+'.'+field)
    sampled_matrix_groups[str(date)] = [g for g in matrix_hashes.values() if len(g)>1]
    for family, ts in [('cnt',['005','01','02','05','1','3','5','10','30']),('vol',['02','05','1','3','5','30'])]:
        for t1,t2 in zip(ts,ts[1:]):
            m1,v1=mats[family+'_t'+t1];m2,v2=mats[family+'_t'+t2]
            assert m1['samples']==m2['samples'], 'Pair sample alignment differs'
            for field in ['rho','g','gs','z','zs']:
                taus.append(dict(date=date,left=family+'_t'+t1,right=family+'_t'+t2,field=field,spearman=corr(v1[field],v2[field]),bit_differences=int(np.count_nonzero(v1[field].view('u4')!=v2[field].view('u4')))))
    index = {n:i for i,n in enumerate(names)}
    for mat in mats:
        for pf in ['x0topk','gtopk']:
            for stat in ['count','gain_mean','h_mean','mass_share']:
                left=f'{mat}_{pf}100_{stat}';right=f'{mat}_{pf}500_{stat}'
                x=a[:,index[left]];y=a[:,index[right]]
                ks.append(dict(date=date,left=left,right=right,spearman=corr(x,y),bits_differ=int(np.count_nonzero(x.view('u4')!=y.view('u4'))),left_finite=int(np.isfinite(x).sum()),right_finite=int(np.isfinite(y).sum())))
null_checks = []
for date in dates:
    for mat, total, d1, d2 in re.findall(r'\[prune:null:([^\]]+)\] \[(\d+), (\d+), (\d+)\]', (root / f'{date}.log').read_text()):
        null_checks.append(dict(date=date,mat=mat,samples=int(total),g_differences=int(d1),gs_differences=int(d2)))
summary = dict(sampled_matrix_duplicate_groups=sampled_matrix_groups,null_checks=null_checks,dates=dates,shapes=[list(a.shape) for a in arrays],duplicate_groups=duplicates,nonconstant_duplicate_groups=nonconstant,constant_columns=constant,removable_exact_aliases=sum(len(g)-1 for g in duplicates),nonconstant_aliases=sum(len(g)-1 for g in nonconstant),intermediate=intermediate,tau_comparisons=taus,k_comparisons=ks)
(root/'analysis.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
print(json.dumps({k:v for k,v in summary.items() if k not in ['duplicate_groups','nonconstant_duplicate_groups','constant_columns','intermediate','tau_comparisons','k_comparisons','null_checks']},ensure_ascii=False))
