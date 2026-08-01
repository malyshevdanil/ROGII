"""Build file 65: an honest offline harness around file51's pipeline.

Why this exists: the 3 visible test wells are byte-identical duplicates of train wells, so the
contact-override resolves them exactly and every preview/self-check number measures a leak rather
than skill. This wraps the real pipeline so it runs against leak-free held-out train wells instead,
and scores every intermediate stage against the withheld truth -- one run gives an honest read on
which layers actually help, with no competition submission spent.
"""
import json

SCRATCH = r'C:\Users\user\AppData\Local\Temp\claude\d--ROGII\6761a874-94c4-40fb-a605-d49c9b58b718\scratchpad'
nb = json.load(open(SCRATCH + r'\rogii_51_kernel\51_gru_theirbase_blend.ipynb', encoding='utf-8'))
cells = nb['cells']


def code_cell(src):
    return {'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [],
            'source': src.splitlines(keepends=True)}


builder = r'''# ============== HONEST OFFLINE HARNESS: synthetic leak-free held-out test set ===================
# The 3 real visible test wells are byte-identical duplicates of train wells (verified: MD/GR/Z
# allclose 1e-9, train.TVT == test.TVT_input on every known row), so the contact-override resolves
# them EXACTLY (RMSE 0.0000) and preview scores measure a leak, not skill -- which is why preview
# never tracked the real LB and every idea cost a real submission to test.
# This cell builds a leak-free stand-in: take N train wells, mask them exactly like real test wells
# (contiguous known prefix, 20-33% of rows, formation columns and TVT dropped), place them in a
# synthetic test/ dir, and REMOVE them from the train/ pool so no duplicate can be looked up.
# Everything downstream runs unmodified; the withheld TVT is kept aside for scoring at the end.
import os, glob, random
from pathlib import Path
import numpy as np, pandas as pd

HARNESS_N_WELLS  = 12
HARNESS_SEED     = 7
HARNESS_KNOWN_LO = 0.204   # observed known-prefix fractions on the real test wells:
HARNESS_KNOWN_HI = 0.326   # 0.204 / 0.273 / 0.326
TEST_SCHEMA      = ['MD', 'X', 'Y', 'Z', 'GR', 'TVT_input']

_SRC  = Path(COMPETITION_DATA_ROOT)
_HARN = Path('/tmp/rogii_harness')          # /tmp, not /kaggle/working: keeps it out of kernel output
for _sub in ('train', 'test'):
    (_HARN / _sub).mkdir(parents=True, exist_ok=True)

_all_wells = sorted(Path(p).name.replace('__horizontal_well.csv', '')
                    for p in glob.glob(str(_SRC / 'train' / '*__horizontal_well.csv')))
_rng = random.Random(HARNESS_SEED)
HARNESS_WELLS = sorted(_rng.sample(_all_wells, HARNESS_N_WELLS))
print('harness: %d train wells available, holding out %d' % (len(_all_wells), HARNESS_N_WELLS))

_truth_rows, _sample_ids = [], []
for _w in HARNESS_WELLS:
    _hw = pd.read_csv(_SRC / 'train' / ('%s__horizontal_well.csv' % _w))
    _n = len(_hw)
    _k = int(round(_n * _rng.uniform(HARNESS_KNOWN_LO, HARNESS_KNOWN_HI)))
    _tvt = _hw['TVT'].to_numpy(float)
    _masked = _hw.copy()
    _masked['TVT_input'] = np.where(np.arange(_n) < _k, _tvt, np.nan)
    _masked[TEST_SCHEMA].to_csv(_HARN / 'test' / ('%s__horizontal_well.csv' % _w), index=False)
    pd.read_csv(_SRC / 'train' / ('%s__typewell.csv' % _w)).to_csv(
        _HARN / 'test' / ('%s__typewell.csv' % _w), index=False)
    for _i in range(_k, _n):
        _sample_ids.append('%s_%d' % (_w, _i))
        _truth_rows.append(('%s_%d' % (_w, _i), float(_tvt[_i])))
    print('  %s: n=%d known=%d (%.1f%%) eval=%d' % (_w, _n, _k, 100.0 * _k / _n, _n - _k))

_held = set(HARNESS_WELLS)
_linked = 0
for _w in _all_wells:
    if _w in _held:
        continue                                # the point of the harness: no lookup-able duplicate
    for _suf in ('__horizontal_well.csv', '__typewell.csv'):
        _src_f = _SRC / 'train' / ('%s%s' % (_w, _suf))
        _dst_f = _HARN / 'train' / ('%s%s' % (_w, _suf))
        if _dst_f.exists() or not _src_f.exists():
            continue
        try:
            os.symlink(_src_f, _dst_f)
        except Exception:
            _dst_f.write_bytes(_src_f.read_bytes())
    _linked += 1

pd.DataFrame({'id': _sample_ids, 'tvt': 0.0}).to_csv(_HARN / 'sample_submission.csv', index=False)
pd.DataFrame(_truth_rows, columns=['id', 'true_tvt']).to_csv('/kaggle/working/harness_truth.csv',
                                                             index=False)

COMPETITION_DATA_ROOT = str(_HARN)          # every downstream stage now reads the leak-free set
print('harness: train pool=%d wells (held-out removed), eval rows=%d' % (_linked, len(_sample_ids)))
print('harness: COMPETITION_DATA_ROOT ->', COMPETITION_DATA_ROOT)
'''

cells.insert(1, code_cell(builder))

# --- drop the held-out wells from the artifact-derived train_df -------------------------------
i_build = next(i for i, c in enumerate(cells) if 'test_df = build_dataset' in ''.join(c['source']))
src = ''.join(cells[i_build]['source'])
anchor = "features = [c for c in train_df.columns if c not in {'well','id','target'}]"
assert anchor in src, 'features anchor not found in cell %d' % i_build
guard = (
    "# HARNESS: the artifact train.csv still holds the held-out wells -- drop them so the GBM/ridge\n"
    "# anchor never sees their labels (masking the synthetic test dir alone would not prevent this).\n"
    "_held_wells = set(globals().get('HARNESS_WELLS', []))\n"
    "if _held_wells:\n"
    "    _n_before = len(train_df)\n"
    "    train_df = train_df[~train_df['well'].astype(str).isin(_held_wells)].reset_index(drop=True)\n"
    "    print('harness: dropped %d artifact rows for %d held-out wells'\n"
    "          % (_n_before - len(train_df), len(_held_wells)))\n\n" + anchor)
cells[i_build]['source'] = src.replace(anchor, guard).splitlines(keepends=True)
print('patched train_df guard into cell', i_build)


def snapshot(tag):
    return code_cell(
        "# HARNESS snapshot: %s\n"
        "import shutil as _snap_sh\n"
        "from pathlib import Path as _SnapPath\n"
        "_snap_w = _SnapPath('/kaggle/working')\n"
        "try:\n"
        "    _snap_sh.copy(_snap_w / 'submission.csv', _snap_w / 'harness_stage_%s.csv')\n"
        "    print('harness snapshot: %s')\n"
        "except Exception as _snap_e:\n"
        "    print('harness snapshot %s skipped:', _snap_e)\n" % (tag, tag, tag, tag))


i_beam = next(i for i, c in enumerate(cells)
              if ''.join(c['source']).strip().startswith('# --- PORTABLE BEAM2'))
tags = ['00_pre_blends', '01_after_beam2', '02_after_neighbor', '03_after_warp', '04_after_gru']
cells.insert(i_beam, snapshot(tags[0]))
for k in range(1, 5):
    cells.insert(i_beam + 2 * k, snapshot(tags[k]))
print('inserted 5 stage snapshots around the blend cells at', i_beam)

scorer = r'''# ================ HARNESS SCORING: honest per-stage RMSE on leak-free wells =====================
import glob as _sc_glob, os as _sc_os
import numpy as _sc_np, pandas as _sc_pd

_truth = _sc_pd.read_csv('/kaggle/working/harness_truth.csv',
                         dtype={'id': 'string'}).set_index('id')['true_tvt']

def _sc_eval(path):
    df = _sc_pd.read_csv(path, dtype={'id': 'string'})
    if 'tvt' not in df.columns or 'id' not in df.columns:
        return None
    df['true'] = df['id'].astype(str).map(_truth)
    df = df.dropna(subset=['true'])
    if df.empty:
        return None
    err = df['tvt'].astype(float).to_numpy() - df['true'].astype(float).to_numpy()
    pooled = float(_sc_np.sqrt(_sc_np.mean(err ** 2)))
    per = (df.assign(w=df['id'].astype(str).str.rsplit('_', n=1).str[0], e2=err ** 2)
             .groupby('w')['e2'].mean().pow(0.5))
    return pooled, float(per.mean()), per

_paths = sorted(_sc_glob.glob('/kaggle/working/harness_stage_*.csv'))
_paths += sorted(_sc_glob.glob('/kaggle/working/submission*.csv'))

print('=' * 78)
print('%-48s %9s %9s' % ('stage', 'pooled', 'perwell'))
print('=' * 78)
_rows = []
for _p in _paths:
    try:
        _r = _sc_eval(_p)
    except Exception:
        _r = None
    if _r is None:
        continue
    print('%-48s %9.4f %9.4f' % (_sc_os.path.basename(_p), _r[0], _r[1]))
    _rows.append(dict(stage=_sc_os.path.basename(_p), pooled_rmse=_r[0], per_well_rmse=_r[1]))
print('=' * 78)
_sc_pd.DataFrame(_rows).to_csv('/kaggle/working/harness_scores.csv', index=False)

_final = _sc_eval('/kaggle/working/submission.csv')
if _final is not None:
    print('\nper-well RMSE of the FINAL submission:')
    print(_final[2].sort_values(ascending=False).to_string())
'''
cells.append(code_cell(scorer))

nb['cells'] = cells
out = SCRATCH + r'\rogii_65_kernel\65_offline_harness.ipynb'
json.dump(nb, open(out, 'w', encoding='utf-8'))
print('total cells:', len(cells))

import ast
bad = 0
for i, c in enumerate(cells):
    if c['cell_type'] != 'code':
        continue
    try:
        ast.parse(''.join(c['source']))
    except SyntaxError:
        bad += 1
print('cells failing local py311 parse (py312 f-strings expected):', bad)
