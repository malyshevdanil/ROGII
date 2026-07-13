"""Dry-run harness for the extracted WARP-blend cell: mimics the pipeline's globals (CFG, OUT,
list_wells, POSTPROCESSORS) using the local 3-well test stub, builds a dummy submission.csv (flat
baseline), then executes the extracted cell exactly as it would run inside the real notebook."""
import os, glob, json, pathlib
import pandas as pd, numpy as np

os.environ['ROGII_WARP_BLEND'] = '0.15'

class CFG:
    DATA = pathlib.Path('d:/ROGII/data')

OUT = pathlib.Path('C:/Users/user/AppData/Local/Temp/claude/d--ROGII/6761a874-94c4-40fb-a605-d49c9b58b718/scratchpad/dryrun_out')
OUT.mkdir(exist_ok=True)

def list_wells(split):
    base = CFG.DATA / split
    return sorted({p.name.split('__')[0] for p in base.glob('*__horizontal_well.csv')})

POSTPROCESSORS = []

# build a dummy submission.csv (flat baseline) using sample_submission.csv id order
sample = pd.read_csv(CFG.DATA / 'sample_submission.csv')
def split_id(series):
    s = series.astype(str); parts = s.str.rsplit('_', n=1, expand=True)
    return parts[0], parts[1].astype(int)
sample['well'], sample['row_idx'] = split_id(sample['id'])
vals = {}
for wid in list_wells('test'):
    hw = pd.read_csv(CFG.DATA / 'test' / f'{wid}__horizontal_well.csv')
    kn = hw[hw['TVT_input'].notna()]
    lt = float(kn['TVT_input'].iloc[-1]) if len(kn) else 0.0
    g = sample[sample['well'] == wid]
    for rid, ridx in zip(g['id'].astype(str), g['row_idx'].astype(int)):
        if 0 <= ridx < len(hw):
            vals[rid] = lt
sub = sample[['id']].copy()
sub['tvt'] = sub['id'].astype(str).map(vals).astype(float)
sub['tvt'] = sub['tvt'].fillna(sub['tvt'].mean())
sub.to_csv(OUT / 'submission.csv', index=False)
print('dummy submission written:', sub.shape, 'rows; unique wells:', sample['well'].nunique())

before = pd.read_csv(OUT / 'submission.csv')
print('BEFORE sample:\n', before.head(3))

exec(open('extracted_warp_cell.py', encoding='utf-8').read())

after = pd.read_csv(OUT / 'submission.csv')
print('AFTER sample:\n', after.head(3))
diff = (after['tvt'] - before['tvt']).abs()
print('rows changed (>1e-6):', (diff > 1e-6).sum(), '/', len(after))
print('POSTPROCESSORS:', POSTPROCESSORS)
